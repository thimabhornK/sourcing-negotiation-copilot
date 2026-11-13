"""Pull the commercial terms out of a vendor's contract.

Each term is found by retrieving the relevant clause and then reading the
numbers from its text with patterns. A language model can replace the patterns
for contracts with freer wording; the output shape stays the same, and each
value keeps the citation of the clause it came from.
"""

from __future__ import annotations

import re

from copilot.contracts.search import ContractIndex, load_index

QUERIES = {
    "payment": "payment terms invoice net days",
    "rebate": "annual volume rebate purchases spend",
    "price": "prices reviewed commodity index adjustment",
    "late": "late delivery delay liquidated damages",
    "termination": "terminate agreement written notice",
}
PCT = re.compile(r"(\d+(?:\.\d+)?)%")
USD = re.compile(r"USD\s*([\d,]+)")
DAYS = re.compile(r"(\d+)\s+(?:calendar\s+)?days")
MONTHS = re.compile(r"every\s+(\d+)\s+months", re.I)


def _numbers(pattern: re.Pattern, text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in pattern.findall(text)]


def extract_terms(vendor_id: str, index: ContractIndex | None = None) -> dict:
    index = index or load_index()
    found = {name: (index.search(q, vendor_id, k=1) or [None])[0] for name, q in QUERIES.items()}
    if not any(found.values()):
        return {"vendor_id": vendor_id, "found": False}
    out: dict = {"vendor_id": vendor_id, "found": True, "citations": {}}

    def cite(name: str) -> str:
        out["citations"][name] = found[name]["citation"]
        return found[name]["text"]

    if found["payment"]:
        text = cite("payment")
        days = _numbers(DAYS, text)
        out["payment_days"] = int(days[0]) if days else None
        pct = _numbers(PCT, text)
        out["early_pay_discount_pct"] = pct[0] if pct else 0.0
        out["early_pay_days"] = int(days[-1]) if pct and len(days) > 1 else 0
    if found["rebate"]:
        text = cite("rebate")
        # Thresholds and percentages can appear in either order, so pair them by size.
        tiers = list(zip(sorted(_numbers(USD, text)), sorted(_numbers(PCT, text)), strict=False))
        out["rebate_tiers"] = [{"threshold_usd": t, "rebate_pct": p} for t, p in tiers]
        out["rebate_must_be_claimed"] = "claim" in text.lower()
    if found["price"]:
        text = cite("price")
        months = _numbers(MONTHS, text)
        band = _numbers(PCT, text)
        out["price_review_months"] = int(months[0]) if months else None
        out["index_linked"] = bool(band) and "no index-based" not in text.lower()
        out["index_band_pct"] = band[0] if out["index_linked"] else None
    if found["late"]:
        pct = _numbers(PCT, cite("late"))
        out["late_penalty_pct_per_week"] = pct[0] if pct else None
        out["late_penalty_cap_pct"] = pct[1] if len(pct) > 1 else None
    if found["termination"]:
        days = _numbers(DAYS, cite("termination"))
        out["termination_notice_days"] = int(days[0]) if days else None
    return out
