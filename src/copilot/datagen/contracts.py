"""Generate one supply agreement per vendor.

Commercial terms are drawn first as structured data (the ground truth), then
written out as contract text with varied wording and several clauses that have
nothing to do with price. Retrieval and extraction are scored against the
structured terms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PAYMENT = [
    "The Buyer shall pay each undisputed invoice within {payment_days} days of the invoice date "
    "(net {payment_days}).",
    "Payment terms are net {payment_days} days from receipt of a correct invoice.",
    "Invoices fall due {payment_days} calendar days after the invoice date.",
]
EARLY_PAY = ("A discount of {early_pay_discount_pct:g}% applies to any invoice settled within "
             "{early_pay_days} days.")
REBATE_ONE = [
    "If total purchases in a calendar year exceed USD {rebate_tier1_usd:,.0f}, the Supplier shall "
    "pay a rebate of {rebate_tier1_pct:g}% of that year's purchases.",
    "Annual volume rebate: {rebate_tier1_pct:g}% once annual spend passes "
    "USD {rebate_tier1_usd:,.0f}.",
]
REBATE_TWO = [
    "If total purchases in a calendar year exceed USD {rebate_tier1_usd:,.0f}, the Supplier shall "
    "pay a rebate of {rebate_tier1_pct:g}% of that year's purchases. If they exceed "
    "USD {rebate_tier2_usd:,.0f}, the rebate is {rebate_tier2_pct:g}%.",
    "Annual volume rebate: {rebate_tier1_pct:g}% once annual spend passes "
    "USD {rebate_tier1_usd:,.0f}; {rebate_tier2_pct:g}% once it passes "
    "USD {rebate_tier2_usd:,.0f}.",
]
REBATE_CLAIM = ("The rebate is not paid automatically: the Buyer must claim it in writing after "
                "the end of the calendar year.")
PRICE_LINKED = [
    "Prices are reviewed every {price_review_months} months. If the reference commodity index "
    "has moved by more than {index_band_pct:g}% since the last review, either party may request "
    "an adjustment in proportion to the material share of the price.",
    "Every {price_review_months} months the parties review prices. A change of more than "
    "{index_band_pct:g}% in the reference commodity index entitles either party to a "
    "proportionate adjustment.",
]
PRICE_FIXED = ["Prices are fixed and are reviewed every {price_review_months} months by mutual "
               "agreement. No index-based adjustment applies."]
LATE = [
    "For each full week of delay the Supplier shall pay liquidated damages of "
    "{late_penalty_pct_per_week:g}% of the value of the delayed goods, capped at "
    "{late_penalty_cap_pct:g}%.",
    "Late deliveries attract liquidated damages of {late_penalty_pct_per_week:g}% of the delayed "
    "order value per week, up to a maximum of {late_penalty_cap_pct:g}%.",
]
TERMINATION = ["Either party may terminate this Agreement for convenience on "
               "{termination_notice_days} days' written notice."]
WARRANTY = ["The Supplier warrants the goods against defects in material and workmanship for "
            "{warranty_months} months from delivery."]
FILLER = {
    "Definitions": "\"Goods\" means the materials and parts listed in Schedule 1. \"Order\" "
                   "means a purchase order issued by the Buyer under this Agreement.",
    "Scope of Supply": "The Supplier shall supply the Goods ordered by the Buyer from time to "
                       "time. This Agreement does not oblige the Buyer to order any minimum "
                       "quantity.",
    "Quality and Inspection": "Goods must conform to the agreed specification. The Buyer may "
                              "inspect and reject non-conforming Goods within 14 days of receipt.",
    "Confidentiality": "Each party shall keep the other's commercial and technical information "
                       "confidential for five years after termination.",
    "Force Majeure": "Neither party is liable for failure caused by events beyond its "
                     "reasonable control, provided it notifies the other party promptly.",
    "Governing Law": "This Agreement is governed by the laws of Thailand. Disputes are referred "
                     "to arbitration in Bangkok.",
}
ORDER = ["Definitions", "Scope of Supply", "Price and Price Adjustment", "Payment Terms",
         "Volume Rebate", "Delivery and Late Delivery", "Quality and Inspection", "Warranty",
         "Confidentiality", "Force Majeure", "Term and Termination", "Governing Law"]


def make_terms(vendors: pd.DataFrame, po: pd.DataFrame, rng: np.random.Generator
               ) -> pd.DataFrame:
    """Draw commercial terms per vendor. Rebate thresholds sit near real annual spend,
    so some vendors have passed a tier and others are just short of one."""
    last_year = po[po["po_date"].dt.year == po["po_date"].dt.year.max()]
    spend = (last_year["qty"] * last_year["unit_price_usd"]).groupby(last_year["vendor_id"]).sum()
    rows = []
    for i, v in enumerate(vendors.itertuples(), start=1):
        base = float(spend.get(v.vendor_id, 100_000.0))
        tiers = int(rng.choice([0, 1, 2], p=[0.25, 0.35, 0.40]))
        tier1 = round(base * rng.uniform(0.6, 1.3), -3) if tiers else 0.0
        early = float(rng.choice([0, 1, 2], p=[0.5, 0.25, 0.25]))
        rows.append({
            "contract_id": f"CT-{i:04d}", "vendor_id": v.vendor_id,
            "payment_days": int(rng.choice([30, 45, 60, 90])),
            "early_pay_discount_pct": early, "early_pay_days": 10 if early else 0,
            "rebate_tier1_usd": tier1,
            "rebate_tier1_pct": float(rng.choice([1.0, 1.5, 2.0])) if tiers else 0.0,
            "rebate_tier2_usd": round(tier1 * rng.uniform(1.4, 2.0), -3) if tiers == 2 else 0.0,
            "rebate_tier2_pct": float(rng.choice([2.5, 3.0, 4.0])) if tiers == 2 else 0.0,
            "index_linked": bool(rng.random() < 0.6),
            "price_review_months": int(rng.choice([3, 6, 12])),
            "index_band_pct": float(rng.choice([3, 5, 10])),
            "late_penalty_pct_per_week": float(rng.choice([0.5, 1.0, 1.5])),
            "late_penalty_cap_pct": float(rng.choice([5, 10])),
            "termination_notice_days": int(rng.choice([30, 60, 90])),
            "warranty_months": int(rng.choice([12, 18, 24])),
            "variant": int(rng.integers(0, 1000)),
        })
    return pd.DataFrame(rows)


def render_contract(t: dict, vendor_name: str) -> str:
    """Write one agreement as Markdown, one numbered clause per section."""
    def pick(options):
        return options[t["variant"] % len(options)].format(**t)

    payment = pick(PAYMENT)
    if t["early_pay_discount_pct"]:
        payment += " " + EARLY_PAY.format(**t)
    if t["rebate_tier2_usd"]:
        rebate = pick(REBATE_TWO) + " " + REBATE_CLAIM
    elif t["rebate_tier1_usd"]:
        rebate = pick(REBATE_ONE) + " " + REBATE_CLAIM
    else:
        rebate = "No volume rebate applies under this Agreement."
    body = dict(FILLER)
    body.update({
        "Price and Price Adjustment": pick(PRICE_LINKED if t["index_linked"] else PRICE_FIXED),
        "Payment Terms": payment,
        "Volume Rebate": rebate,
        "Delivery and Late Delivery": "The Supplier shall deliver on the date stated in each "
                                      "Order. " + pick(LATE),
        "Warranty": pick(WARRANTY),
        "Term and Termination": "This Agreement runs for three years from its effective date. "
                                + pick(TERMINATION),
    })
    lines = [f"# Supply Agreement {t['contract_id']}", "",
             f"Between the Buyer and {vendor_name} (the \"Supplier\"), vendor {t['vendor_id']}.",
             ""]
    for n, title in enumerate(ORDER, start=1):
        lines += [f"## {n}. {title}", "", body[title], ""]
    return "\n".join(lines)


def write_contracts(terms: pd.DataFrame, vendors: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    names = vendors.set_index("vendor_id")["vendor_name"]
    for t in terms.to_dict("records"):
        (folder / f"{t['contract_id']}.md").write_text(render_contract(t, names[t["vendor_id"]]))
