"""The tools agents are allowed to call.

Every tool is a plain function that reads the curated and serving data and
returns a dictionary. Agents never write SQL themselves: questions about spend
go through the fixed, parameterised queries below, which keeps answers
repeatable and stops a model from scanning or exposing data it should not.
"""

from __future__ import annotations

import copy
from functools import lru_cache

import duckdb
import pandas as pd

from copilot import config
from copilot.contracts.extract import extract_terms
from copilot.contracts.search import ContractIndex, load_index
from copilot.quote_check import check_quote


def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    for name, folder in [("po", config.SERVING / "po_scored.parquet"),
                         ("vendors", config.CURATED / "vendors.parquet"),
                         ("materials", config.CURATED / "materials.parquet"),
                         ("scores", config.SERVING / "vendor_score.parquet"),
                         ("quotes", config.CURATED / "rfq_quotes.parquet")]:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{folder}')")
    return con.execute(sql, params or {}).df()


def _records(df: pd.DataFrame) -> list[dict]:
    return df.round(4).to_dict("records")


def latest_date() -> str:
    """The most recent purchase date in the data, as YYYY-MM-DD."""
    return str(_query("SELECT MAX(po_date)::DATE AS d FROM po")["d"][0])[:10]


# ---- spend questions ---------------------------------------------------------

def top_vendors_by_spend(year: int | None = None, category: str | None = None, n: int = 5
                         ) -> dict:
    """Vendors ranked by total spend, optionally for one year or material category."""
    rows = _query("""
        SELECT p.vendor_id, v.vendor_name, ROUND(SUM(p.qty * p.unit_price_usd), 0) AS spend_usd,
               COUNT(*) AS po_lines
        FROM po p JOIN vendors v USING (vendor_id) JOIN materials m USING (material_id)
        WHERE ($year IS NULL OR YEAR(p.po_date) = $year)
          AND ($category IS NULL OR m.category = $category)
        GROUP BY 1, 2 ORDER BY spend_usd DESC LIMIT $n""",
                  {"year": year, "category": category, "n": n})
    return {"tool": "top_vendors_by_spend", "year": year, "category": category,
            "rows": _records(rows)}


def most_expensive_vendors(year: int | None = None, category: str | None = None,
                           material_id: str | None = None, n: int = 5) -> dict:
    """Vendors ranked by how far their prices sit above should-cost."""
    rows = _query("""
        SELECT p.vendor_id, v.vendor_name,
               ROUND(100 * (MEDIAN(p.unit_price_usd / p.should_cost_p50) - 1), 1)
                 AS pct_above_should_cost,
               ROUND(SUM(p.qty * (p.unit_price_usd - p.should_cost_p50)), 0) AS excess_spend_usd,
               COUNT(*) AS po_lines
        FROM po p JOIN vendors v USING (vendor_id) JOIN materials m USING (material_id)
        WHERE ($year IS NULL OR YEAR(p.po_date) = $year)
          AND ($category IS NULL OR m.category = $category)
          AND ($material IS NULL OR p.material_id = $material)
        GROUP BY 1, 2 HAVING COUNT(*) >= 5
        ORDER BY pct_above_should_cost DESC LIMIT $n""",
                  {"year": year, "category": category, "material": material_id, "n": n})
    return {"tool": "most_expensive_vendors", "year": year, "category": category,
            "material_id": material_id, "rows": _records(rows)}


def late_vendors(year: int | None = None, n: int = 5) -> dict:
    """Vendors ranked by the share of delivered lines that arrived after the promised date."""
    rows = _query("""
        SELECT p.vendor_id, v.vendor_name,
               ROUND(100 * AVG(CASE WHEN p.received_date > p.promised_date THEN 1.0 ELSE 0 END), 1)
                 AS late_pct,
               COUNT(*) AS delivered_lines
        FROM po p JOIN vendors v USING (vendor_id)
        WHERE p.received_date IS NOT NULL AND ($year IS NULL OR YEAR(p.po_date) = $year)
        GROUP BY 1, 2 HAVING COUNT(*) >= 20
        ORDER BY late_pct DESC LIMIT $n""", {"year": year, "n": n})
    return {"tool": "late_vendors", "year": year, "rows": _records(rows)}


def spend_by_category(year: int | None = None) -> dict:
    """Total spend and number of vendors per material category."""
    rows = _query("""
        SELECT m.category, ROUND(SUM(p.qty * p.unit_price_usd), 0) AS spend_usd,
               COUNT(DISTINCT p.vendor_id) AS vendors
        FROM po p JOIN materials m USING (material_id)
        WHERE ($year IS NULL OR YEAR(p.po_date) = $year)
        GROUP BY 1 ORDER BY spend_usd DESC""", {"year": year})
    return {"tool": "spend_by_category", "year": year, "rows": _records(rows)}


# ---- vendors and contracts ---------------------------------------------------

def vendor_profile(vendor_id: str) -> dict:
    """Scorecard for one vendor: delivery, quality, price level, credit rating and rank."""
    rows = _query("SELECT * FROM scores WHERE vendor_id = $v", {"v": vendor_id})
    if rows.empty:
        return {"tool": "vendor_profile", "vendor_id": vendor_id, "found": False}
    rank = _query("SELECT COUNT(*) + 1 AS r FROM scores WHERE total_score > "
                  "(SELECT total_score FROM scores WHERE vendor_id = $v)", {"v": vendor_id})
    total = _query("SELECT COUNT(*) AS n FROM scores")
    return {"tool": "vendor_profile", "found": True, **_records(rows)[0],
            "rank": int(rank["r"][0]), "of": int(total["n"][0])}


@lru_cache(maxsize=512)
def _terms(vendor_id: str, index: ContractIndex) -> dict:
    return extract_terms(vendor_id, index)


def contract_terms(vendor_id: str) -> dict:
    """Structured commercial terms from a vendor's contract, with clause citations."""
    return {"tool": "contract_terms", **copy.deepcopy(_terms(vendor_id, load_index()))}


def search_contracts(question: str, vendor_id: str | None = None, k: int = 3) -> dict:
    """Find the contract clauses that best answer a question, with citations."""
    return {"tool": "search_contracts", "question": question, "vendor_id": vendor_id,
            "results": load_index().search(question, vendor_id, k)}


def rebate_status(vendor_id: str, year: int) -> dict:
    """Has annual spend with this vendor reached a rebate tier, and how far is the next one?"""
    terms = contract_terms(vendor_id)
    spend = float(_query("SELECT COALESCE(SUM(qty * unit_price_usd), 0) AS s FROM po "
                         "WHERE vendor_id = $v AND YEAR(po_date) = $y",
                         {"v": vendor_id, "y": year})["s"][0])
    tiers = terms.get("rebate_tiers", [])
    reached = [t for t in tiers if spend >= t["threshold_usd"]]
    ahead = [t for t in tiers if spend < t["threshold_usd"]]
    out = {"tool": "rebate_status", "vendor_id": vendor_id, "year": year,
           "spend_usd": round(spend, 0), "has_rebate": bool(tiers),
           "citation": terms.get("citations", {}).get("rebate")}
    if reached:
        best = reached[-1]
        out.update(rebate_pct_reached=best["rebate_pct"],
                   rebate_due_usd=round(spend * best["rebate_pct"] / 100, 0),
                   must_be_claimed=terms.get("rebate_must_be_claimed", False))
    if ahead:
        nxt = ahead[0]
        out.update(next_tier_usd=nxt["threshold_usd"], next_tier_pct=nxt["rebate_pct"],
                   gap_to_next_tier_usd=round(nxt["threshold_usd"] - spend, 0))
    return out


# ---- quotes ------------------------------------------------------------------

def quote_check(material_id: str, qty: float, price: float, currency: str = "USD",
                vendor_id: str | None = None, date: str | None = None) -> dict:
    """Compare a quoted unit price with the should-cost range for that material,
    quantity and date (YYYY-MM-DD). Returns the verdict, market trend and vendor record."""
    return {"tool": "quote_check", **check_quote(material_id, qty, price, currency, vendor_id,
                                                 date)}


def rfq_summary(rfq_id: str) -> dict:
    """All quotes received for one sourcing event, each checked against should-cost."""
    quotes = _query("SELECT * FROM quotes WHERE rfq_id = $r ORDER BY quoted_price_usd",
                    {"r": rfq_id})
    if quotes.empty:
        return {"tool": "rfq_summary", "rfq_id": rfq_id, "found": False}
    checked = []
    for q in quotes.itertuples():
        c = check_quote(q.material_id, q.qty, q.quoted_price_usd, "USD", q.vendor_id,
                        f"{q.quote_date:%Y-%m-%d}")
        c.update(lead_time_days=int(q.lead_time_days), payment_terms=q.payment_terms)
        checked.append(c)
    return {"tool": "rfq_summary", "rfq_id": rfq_id, "found": True,
            "material_id": checked[0]["material_id"], "qty": checked[0]["qty"],
            "quote_date": checked[0]["quote_date"], "quotes": checked}
