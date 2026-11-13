"""Clean the raw source files into the curated zone.

Each step fixes one kind of mess and records how many rows it touched. Rows
that cannot be trusted are not dropped silently: they go to a quarantine file
with the reason, so a person can review them.

Run with:  python -m copilot.pipeline.clean
"""

from __future__ import annotations

import difflib
import json
import re

import pandas as pd

from copilot import config
from copilot.contracts.search import build_chunks
from copilot.pipeline import deidentify

LEGAL_WORDS = {"co", "ltd", "company", "limited", "inc", "corp", "plc"}
CURRENCY_MAP = {"thb": "THB", "baht": "THB", "฿": "THB", "usd": "USD", "us$": "USD",
                "eur": "EUR", "€": "EUR", "euro": "EUR"}
UOM_MAP = {"kg": "kg", "kgs": "kg", "ton": "ton", "mt": "ton", "g": "g", "pcs": "pcs",
           "pcs.": "pcs", "ea": "ea"}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def normalize_name(name: str) -> str:
    """'  SIAM STEEL CO.,LTD ' and 'Siam Steel Company Limited' both become 'siam steel'."""
    words = re.sub(r"[^a-z0-9 ]", " ", str(name).lower()).split()
    return " ".join(w for w in words if w not in LEGAL_WORDS)


def resolve_vendors(names: pd.Series, master: pd.DataFrame, cutoff: float = 0.88) -> pd.Series:
    """Map free-text vendor names to vendor_id: exact match on the normalised name,
    then a fuzzy match to catch typos. Unmatched names return NA."""
    key_to_id = dict(zip(master["vendor_name"].map(normalize_name), master["vendor_id"],
                         strict=True))
    keys = names.map(normalize_name)
    lookup = {}
    for key in keys.unique():
        if key in key_to_id:
            lookup[key] = key_to_id[key]
        else:
            close = difflib.get_close_matches(key, list(key_to_id), n=1, cutoff=cutoff)
            lookup[key] = key_to_id[close[0]] if close else pd.NA
    return keys.map(lookup)


def parse_number(text: pd.Series) -> pd.Series:
    """'฿1,234.50' -> 1234.5 ; '' -> NaN."""
    cleaned = text.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_date(text: str):
    """Handle ISO, day-first, '5 Apr 2023' and Thai Buddhist-era years (CE + 543)."""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return pd.NaT
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return pd.Timestamp(s)
        if m := re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s):
            day, month, year = (int(g) for g in m.groups())
        elif m := re.fullmatch(r"(\d{1,2}) ([A-Za-z]{3}) (\d{4})", s):
            day, month, year = int(m[1]), MONTHS[m[2].lower()], int(m[3])
        else:
            return pd.NaT
        if year > 2400:  # Buddhist era
            year -= 543
        return pd.Timestamp(year=year, month=month, day=day)
    except (ValueError, KeyError):
        return pd.NaT


def clean_po_lines(raw: pd.DataFrame, vendors: pd.DataFrame, indices: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (clean rows, quarantined rows with a reason, data-quality counts)."""
    report = {"rows_in": len(raw)}
    df = raw.drop_duplicates().copy()
    report["duplicates_removed"] = len(raw) - len(df)

    df["vendor_id"] = resolve_vendors(df["vendor_name"], vendors)
    df["material_id"] = df["material_code"].astype(str).str.strip().str.upper()
    df["qty_raw"] = parse_number(df["qty"])
    df["price_raw"] = parse_number(df["unit_price"])
    df["currency"] = df["currency"].astype(str).str.strip().str.lower().map(CURRENCY_MAP)
    df["uom"] = df["uom"].astype(str).str.strip().str.lower().map(UOM_MAP)
    for col in ["po_date", "promised_date", "received_date"]:
        df[col] = pd.to_datetime(df[col].map(parse_date))

    # Standardise units (kg or pieces) and currency (USD at the month's rate).
    factor = df["uom"].map(config.UOM_TO_BASE)
    df["qty"] = df["qty_raw"] * factor
    month = df["po_date"].dt.to_period("M").dt.to_timestamp()
    fx = indices.set_index("month")[["thb_per_usd", "eur_per_usd"]].reindex(month)
    rate = pd.Series(1.0, index=df.index)
    rate[df["currency"] == "THB"] = fx["thb_per_usd"].to_numpy()[df["currency"] == "THB"]
    rate[df["currency"] == "EUR"] = fx["eur_per_usd"].to_numpy()[df["currency"] == "EUR"]
    df["unit_price_usd"] = df["price_raw"] / factor / rate

    # Decide which rows cannot be trusted. First matching reason wins.
    median = df.groupby("material_id")["unit_price_usd"].transform("median")
    reasons = pd.Series(pd.NA, index=df.index, dtype="object")
    checks = [
        ("unknown_vendor", df["vendor_id"].isna()),
        ("missing_qty_or_price", df["qty_raw"].isna() | df["price_raw"].isna()),
        ("negative_or_zero_qty", df["qty_raw"] <= 0),
        ("unparseable_date", df["po_date"].isna()),
        ("unknown_unit_or_currency", df["uom"].isna() | df["currency"].isna()),
        ("price_outlier", (df["unit_price_usd"] > 20 * median)
         | (df["unit_price_usd"] < median / 20)),
    ]
    for reason, mask in checks:
        hit = mask.fillna(False) & reasons.isna()
        reasons[hit] = reason
        report[f"quarantined_{reason}"] = int(hit.sum())

    bad = df[reasons.notna()].assign(reason=reasons[reasons.notna()])
    good = df[reasons.isna()]
    cols = ["po_number", "line_no", "vendor_id", "material_id", "qty", "unit_price_usd",
            "po_date", "promised_date", "received_date", "qc_pass_rate"]
    good = good[cols].astype({"line_no": "int64", "qty": "float64", "unit_price_usd": "float64",
                              "qc_pass_rate": "float64"})
    report["rows_out"] = len(good)
    report["rows_quarantined"] = len(bad)
    return good.reset_index(drop=True), bad[list(raw.columns) + ["reason"]], report


def run() -> dict:
    for folder in (config.CURATED, config.QUARANTINE, config.REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    vendors = pd.read_csv(config.RAW / "vendor_master.csv", dtype=str)
    indices = pd.read_csv(config.RAW / "market_indices.csv", parse_dates=["month"])
    raw_po = pd.read_csv(config.RAW / "erp_po_lines.csv", dtype=str, keep_default_na=False)

    po, bad, report = clean_po_lines(raw_po, vendors, indices)
    po.to_parquet(config.CURATED / "po_lines.parquet", index=False)
    bad.to_csv(config.QUARANTINE / "po_lines_rejected.csv", index=False)

    deidentify.deidentify_vendors(vendors).to_parquet(config.CURATED / "vendors.parquet",
                                                      index=False)
    pcard = pd.read_csv(config.RAW / "pcard_transactions.csv", dtype={"card_pan": str})
    deidentify.deidentify_pcard(pcard).to_parquet(config.CURATED / "pcard_transactions.parquet",
                                                  index=False)
    indices.to_parquet(config.CURATED / "market_indices.parquet", index=False)
    pd.read_csv(config.RAW / "material_master.csv").to_parquet(
        config.CURATED / "materials.parquet", index=False)
    pd.read_csv(config.RAW / "rfq_quotes.csv", parse_dates=["quote_date"]).to_parquet(
        config.CURATED / "rfq_quotes.parquet", index=False)
    pd.read_csv(config.RAW / "negotiation_outcomes.csv").to_parquet(
        config.CURATED / "negotiation_outcomes.parquet", index=False)

    chunks = build_chunks(config.RAW / "contracts")
    chunks.to_parquet(config.CURATED / "contract_chunks.parquet", index=False)
    report["contract_clauses_indexed"] = len(chunks)

    (config.REPORTS / "data_quality.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    result = run()
    print("Data quality report")
    for key, value in result.items():
        print(f"  {key:<40} {value:>8,}")
