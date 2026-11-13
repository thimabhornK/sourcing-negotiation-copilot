"""Generate a synthetic but realistic procurement dataset.

The generator first builds a clean "true" world (materials, vendors, market
indices, prices driven by a cost formula), then exports it the way real source
systems look: inconsistent vendor names, mixed units and currencies, several
date formats, duplicates, blanks and typos. The cleaning pipeline has to undo
that mess, and because we know the truth we can measure how well it does.

Everything here is fictional. Market indices are simulated, not real prices.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from copilot import config
from copilot.datagen import mess

INDICES = {
    # index_id: (start level in USD per metric ton, monthly volatility)
    "steel_hrc": (750.0, 0.055),
    "aluminium": (2100.0, 0.045),
    "copper": (8000.0, 0.050),
    "pp_resin": (1250.0, 0.050),
    "kraft_paper": (820.0, 0.035),
}
CATEGORIES = {
    # category: (index_id, base unit, kg of commodity per unit range, conversion cost range USD)
    "steel_sheet": ("steel_hrc", "kg", (1.02, 1.10), (0.10, 0.35)),
    "steel_part": ("steel_hrc", "pcs", (0.20, 4.00), (0.30, 3.00)),
    "aluminium_part": ("aluminium", "pcs", (0.05, 1.50), (0.25, 2.50)),
    "copper_wire": ("copper", "kg", (1.01, 1.06), (0.40, 1.20)),
    "plastic_resin": ("pp_resin", "kg", (1.00, 1.03), (0.05, 0.20)),
    "plastic_part": ("pp_resin", "pcs", (0.02, 0.60), (0.05, 0.80)),
    "packaging": ("kraft_paper", "pcs", (0.10, 1.20), (0.03, 0.25)),
}
VENDOR_WORDS_A = ["Siam", "Bangkok", "Thai", "Eastern", "Golden", "Pacific", "Union", "Royal",
                  "Asia", "Mekong", "Summit", "Delta", "Prime", "Lotus", "Central", "Northern"]
VENDOR_WORDS_B = ["Precision", "Steel", "Metal", "Polymer", "Industrial", "Packaging", "Wire",
                  "Forming", "Alloy", "Plastics", "Components", "Casting", "Supply", "Tech"]
COUNTRIES = {"TH": ("THB", 0.60), "CN": ("USD", 0.18), "JP": ("USD", 0.08), "DE": ("EUR", 0.07),
             "VN": ("USD", 0.07)}


def months() -> pd.DatetimeIndex:
    return pd.date_range(config.START_MONTH, config.END_MONTH, freq="MS")


def make_indices(rng: np.random.Generator) -> pd.DataFrame:
    """Monthly commodity, labour and FX series as a mean-reverting random walk."""
    idx = months()
    out = {"month": idx}
    for name, (start, vol) in INDICES.items():
        level, series = np.log(start), []
        for _ in idx:
            level += 0.04 * (np.log(start) - level) + rng.normal(0, vol)
            series.append(np.exp(level))
        out[name] = np.round(series, 2)
    out["labor_index"] = np.round(100 * np.cumprod(1 + rng.normal(0.003, 0.002, len(idx))), 2)
    out["thb_per_usd"] = np.round(33 * np.exp(np.cumsum(rng.normal(0, 0.012, len(idx)))), 3)
    out["eur_per_usd"] = np.round(0.90 * np.exp(np.cumsum(rng.normal(0, 0.010, len(idx)))), 4)
    return pd.DataFrame(out)


def make_materials(rng: np.random.Generator, n: int) -> pd.DataFrame:
    rows = []
    cats = list(CATEGORIES)
    for i in range(n):
        cat = cats[i % len(cats)]
        index_id, uom, kg_rng, conv_rng = CATEGORIES[cat]
        rows.append({
            "material_id": f"MAT-{i + 1:04d}",
            "category": cat,
            "index_id": index_id,
            "base_uom": uom,
            "kg_per_unit": round(rng.uniform(*kg_rng), 3),
            "conversion_cost_usd": round(rng.uniform(*conv_rng), 3),
            "ref_qty": int(rng.choice([500, 1000, 2000, 5000, 10000])),
            "description": f"{cat.replace('_', ' ').title()} spec {rng.integers(100, 999)}",
        })
    return pd.DataFrame(rows)


def make_vendors(rng: np.random.Generator, n: int) -> pd.DataFrame:
    rows, used = [], set()
    cats = list(CATEGORIES)
    countries = list(COUNTRIES)
    weights = [COUNTRIES[c][1] for c in countries]
    while len(rows) < n:
        name = f"{rng.choice(VENDOR_WORDS_A)} {rng.choice(VENDOR_WORDS_B)} Co., Ltd."
        if name in used:
            name = name.replace(" Co., Ltd.", f" {rng.choice(VENDOR_WORDS_B)} Co., Ltd.")
        if name in used:
            continue
        used.add(name)
        i = len(rows) + 1
        country = str(rng.choice(countries, p=weights))
        slug = name.split(" Co.")[0].lower().replace(" ", "")
        rows.append({
            "vendor_id": f"V-{i:04d}",
            "vendor_name": name,
            "country": country,
            "currency": COUNTRIES[country][0],
            "categories": "|".join(rng.choice(cats, size=rng.integers(1, 3), replace=False)),
            "tax_id": "".join(map(str, rng.integers(0, 10, 13))),
            "bank_account": "".join(map(str, rng.integers(0, 10, 10))),
            "contact_email": f"sales@{slug}.example",
            "credit_rating": str(rng.choice(["A", "B", "C", "D"], p=[0.3, 0.4, 0.2, 0.1])),
            # Hidden behaviour, never exported to the raw files:
            "hidden_margin": rng.uniform(0.08, 0.25),
            "hidden_pass_through": rng.uniform(0.3, 1.0),  # share of index drops passed on
            "hidden_late_rate": rng.beta(2, 8),
            "hidden_quality": rng.beta(30, 2),
        })
    return pd.DataFrame(rows)


def sticky_index(index: np.ndarray, pass_through: float) -> np.ndarray:
    """Vendor's view of an index: rises are passed on in full, drops only partly."""
    eff = np.empty_like(index)
    eff[0] = index[0]
    for t in range(1, len(index)):
        eff[t] = index[t] if index[t] >= eff[t - 1] else (
            eff[t - 1] - pass_through * (eff[t - 1] - index[t]))
    return eff


def fair_price_usd(mat: pd.DataFrame, vend: pd.DataFrame, month_pos: np.ndarray,
                   qty: np.ndarray, indices: pd.DataFrame,
                   eff: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    """Price a vendor charges with no noise: cost formula times the vendor's margin."""
    lag = np.clip(month_pos - 1, 0, None)
    idx_level = np.array([eff[(v, i)][m] for v, i, m in
                          zip(vend["vendor_id"], mat["index_id"], lag, strict=True)])
    labor = indices["labor_index"].to_numpy()[lag] / 100
    cost = (mat["kg_per_unit"].to_numpy() * idx_level / 1000
            + mat["conversion_cost_usd"].to_numpy() * labor)
    volume = (qty / mat["ref_qty"].to_numpy()) ** -0.06
    return cost * volume * (1 + vend["hidden_margin"].to_numpy())


def pick_pairs(rng, n, materials, vendors):
    """Pick n (material, vendor) pairs where the vendor supplies the category."""
    by_cat = {c: vendors[vendors["categories"].str.contains(c)].index.to_numpy()
              for c in CATEGORIES}
    mat_ix = rng.integers(0, len(materials), n)
    ven_ix = np.array([rng.choice(by_cat[c]) for c in materials["category"].to_numpy()[mat_ix]])
    return (materials.iloc[mat_ix].reset_index(drop=True),
            vendors.iloc[ven_ix].reset_index(drop=True))


def generate(n_po: int = 60_000, n_rfq: int = 2_000, n_materials: int = 56, n_vendors: int = 80,
             n_pcard: int = 3_000, seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = make_indices(rng)
    materials = make_materials(rng, n_materials)
    vendors = make_vendors(rng, n_vendors)
    # Make sure every category has at least three vendors so RFQs have competition.
    for k, cat in enumerate(CATEGORIES):
        for j in range(3):
            row = (k * 3 + j) % n_vendors
            if cat not in vendors.at[row, "categories"]:
                vendors.at[row, "categories"] += f"|{cat}"
    eff = {(v.vendor_id, i): sticky_index(indices[i].to_numpy(), v.hidden_pass_through)
           for v in vendors.itertuples() for i in INDICES}
    month_start = indices["month"].to_numpy()
    n_months = len(indices)

    # ---- purchase order lines -------------------------------------------------
    mat, ven = pick_pairs(rng, n_po, materials, vendors)
    mpos = rng.integers(0, n_months, n_po)
    po_date = pd.to_datetime(month_start[mpos]) + pd.to_timedelta(rng.integers(0, 28, n_po), "D")
    qty = np.maximum(1, np.round(mat["ref_qty"].to_numpy() * rng.lognormal(0, 0.6, n_po)))
    price = fair_price_usd(mat, ven, mpos, qty, indices, eff) * rng.lognormal(0, 0.03, n_po)
    lead = rng.integers(14, 60, n_po)
    late = rng.random(n_po) < ven["hidden_late_rate"].to_numpy()
    delay = np.where(late, rng.integers(1, 21, n_po), -rng.integers(0, 4, n_po))
    po = pd.DataFrame({
        "po_number": [f"PO{700000 + i // 3}" for i in range(n_po)],
        "line_no": [i % 3 + 1 for i in range(n_po)],
        "vendor_id": ven["vendor_id"], "vendor_name": ven["vendor_name"],
        "currency": ven["currency"], "material_id": mat["material_id"],
        "description": mat["description"], "base_uom": mat["base_uom"],
        "qty": qty, "unit_price_usd": price, "po_date": po_date,
        "promised_date": po_date + pd.to_timedelta(lead, "D"),
        "received_date": po_date + pd.to_timedelta(lead + delay, "D"),
        "qc_pass_rate": np.round(np.clip(rng.normal(ven["hidden_quality"], 0.02), 0.5, 1.0), 3),
        "_mpos": mpos,
    })

    # ---- RFQ quotes, labels and negotiated outcomes ---------------------------
    quotes, labels, outcomes = [], [], []
    test_pos = int(np.searchsorted(month_start, np.datetime64(config.TEST_START)))
    for r in range(n_rfq):
        m = materials.iloc[rng.integers(0, n_materials)]
        pool = vendors[vendors["categories"].str.contains(m.category)]
        chosen = pool.sample(min(len(pool), int(rng.integers(3, 6))), random_state=int(seed + r))
        # A third of events fall in the held-out year so evaluation has enough quotes.
        pos = (int(rng.integers(test_pos, n_months)) if rng.random() < 0.33
               else int(rng.integers(12, test_pos)))
        date = pd.Timestamp(month_start[pos]) + pd.Timedelta(days=int(rng.integers(0, 28)))
        q = float(max(1, round(m.ref_qty * rng.lognormal(0, 0.5))))
        k = len(chosen)
        fair = fair_price_usd(pd.DataFrame([m] * k).reset_index(drop=True),
                              chosen.reset_index(drop=True), np.full(k, pos), np.full(k, q),
                              indices, eff)
        over = rng.random(k) < 0.15
        quoted = fair * np.where(over, 1 + rng.uniform(0.12, 0.35, k), rng.lognormal(0, 0.03, k))
        rfq_id = f"RFQ-{r + 1:05d}"
        for v, p, o in zip(chosen.itertuples(), quoted, over, strict=True):
            quotes.append({
                "rfq_id": rfq_id, "vendor_id": v.vendor_id, "material_id": m.material_id,
                "qty": q, "quoted_price_usd": round(float(p), 4), "quote_date": date.date(),
                "lead_time_days": int(rng.integers(14, 60)),
                "payment_terms": str(rng.choice(["NET30", "NET45", "NET60", "NET90"])),
            })
            labels.append({"rfq_id": rfq_id, "vendor_id": v.vendor_id, "is_overpriced": bool(o)})
        best = int(np.argmin(quoted))
        outcomes.append({
            "rfq_id": rfq_id, "material_id": m.material_id,
            "award_vendor_id": chosen.iloc[best].vendor_id,
            "opening_quote_usd": round(float(quoted[best]), 4),
            "final_price_usd": round(float(quoted[best] * (1 - rng.uniform(0, 0.06))), 4),
            "cycle_days": int(rng.integers(5, 30)),
        })

    # ---- purchasing card transactions (card data that must be protected) ------
    pc_vendor = vendors.sample(n_pcard, replace=True, random_state=seed).reset_index(drop=True)
    first = ["Somchai", "Anong", "Kittipong", "Malee", "Pim", "Narin", "Suda", "Wichai"]
    last = ["Srisuk", "Thongdee", "Wongsa", "Chaiyo", "Rattanakul", "Boonmee"]
    holders = [f"{rng.choice(first)} {rng.choice(last)}" for _ in range(25)]
    cards = ["4000" + "".join(map(str, rng.integers(0, 10, 12))) for _ in holders]
    who = rng.integers(0, len(holders), n_pcard)
    pcard = pd.DataFrame({
        "txn_id": [f"TXN{i + 1:07d}" for i in range(n_pcard)],
        "card_pan": [cards[w] for w in who],
        "cardholder_name": [holders[w] for w in who],
        "merchant_name": pc_vendor["vendor_name"],
        "amount_thb": np.round(rng.lognormal(8, 1, n_pcard), 2),
        "txn_date": pd.to_datetime(month_start[rng.integers(0, n_months, n_pcard)]).date,
    })

    public_vendors = vendors[[c for c in vendors.columns if not c.startswith("hidden_")]]
    return {"indices": indices, "materials": materials, "vendors": public_vendors,
            "po_clean": po, "quotes": pd.DataFrame(quotes), "labels": pd.DataFrame(labels),
            "outcomes": pd.DataFrame(outcomes), "pcard": pcard}


def write_raw(tables: dict[str, pd.DataFrame], seed: int = config.SEED) -> None:
    """Write source-system style files to data/raw, making the PO export messy."""
    rng = np.random.default_rng(seed + 1)
    config.RAW.mkdir(parents=True, exist_ok=True)
    tables["indices"].to_csv(config.RAW / "market_indices.csv", index=False)
    tables["materials"].to_csv(config.RAW / "material_master.csv", index=False)
    tables["vendors"].to_csv(config.RAW / "vendor_master.csv", index=False)
    tables["quotes"].to_csv(config.RAW / "rfq_quotes.csv", index=False)
    tables["outcomes"].to_csv(config.RAW / "negotiation_outcomes.csv", index=False)
    tables["pcard"].to_csv(config.RAW / "pcard_transactions.csv", index=False)
    # Ground truth for evaluation only. A real customer would not have this file.
    tables["labels"].to_csv(config.RAW / "eval_quote_labels.csv", index=False)
    messy = mess.make_messy_po(tables["po_clean"], tables["indices"], rng)
    messy.to_csv(config.RAW / "erp_po_lines.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--po-lines", type=int, default=60_000)
    ap.add_argument("--rfqs", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    tables = generate(n_po=args.po_lines, n_rfq=args.rfqs, seed=args.seed)
    write_raw(tables, seed=args.seed)
    print(f"Wrote raw files to {config.RAW}")
    for name in ["po_clean", "quotes", "vendors", "materials", "pcard"]:
        print(f"  {name:<10} {len(tables[name]):>7,} rows")


if __name__ == "__main__":
    main()
