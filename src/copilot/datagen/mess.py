"""Turn clean purchase-order lines into a messy ERP export.

Each function below injects one kind of problem that real procurement data has.
The cleaning pipeline in `copilot.pipeline.clean` has a matching fix for each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UOM_SPELLINGS = {
    "kg": ["kg", "KG", "Kg", "kgs"],
    "ton": ["ton", "TON", "MT"],
    "g": ["g", "G"],
    "pcs": ["pcs", "PCS", "ea", "EA", "Pcs."],
}
CURRENCY_SPELLINGS = {"THB": ["THB", "Baht", "฿", "thb"], "USD": ["USD", "US$", "usd"],
                      "EUR": ["EUR", "€", "Euro"]}
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def messy_vendor_name(name: str, rng: np.random.Generator) -> str:
    """Same vendor, typed by different people in different ways."""
    core = name.replace(" Co., Ltd.", "")
    suffix = rng.choice([" Co., Ltd.", " CO.,LTD", " Company Limited", " Co Ltd", "", " co.,ltd."])
    out = core + suffix
    style = rng.random()
    if style < 0.15:
        out = out.upper()
    elif style < 0.25:
        out = out.lower()
    if rng.random() < 0.10:
        out = f"  {out} "
    if rng.random() < 0.03 and len(core) > 5:  # typo: swap two neighbouring letters
        i = int(rng.integers(1, len(core) - 2))
        out = out[:i] + out[i + 1] + out[i] + out[i + 2:]
    return out


def messy_date(d: pd.Timestamp, rng: np.random.Generator) -> str:
    """ISO, day-first, text month, or Thai Buddhist-era year (CE + 543)."""
    style = rng.random()
    if style < 0.55:
        return d.strftime("%Y-%m-%d")
    if style < 0.75:
        return d.strftime("%d/%m/%Y")
    if style < 0.90:
        return f"{d.day} {MONTH_ABBR[d.month - 1]} {d.year}"
    return f"{d.day:02d}/{d.month:02d}/{d.year + 543}"


def messy_price(value: float, symbol: str, rng: np.random.Generator) -> str:
    text = f"{value:,.6f}" if rng.random() < 0.3 else f"{value:.6f}"
    if symbol in {"฿", "US$", "€"} and rng.random() < 0.5:
        return f"{symbol}{text}"
    return text


def make_messy_po(po: pd.DataFrame, indices: pd.DataFrame, rng: np.random.Generator
                  ) -> pd.DataFrame:
    n = len(po)
    fx = {"USD": np.ones(n),
          "THB": indices["thb_per_usd"].to_numpy()[po["_mpos"]],
          "EUR": indices["eur_per_usd"].to_numpy()[po["_mpos"]]}
    rate = np.select([po["currency"] == c for c in fx], list(fx.values()))

    # Weight-based lines are sometimes entered in tons or grams instead of kilograms.
    uom = po["base_uom"].to_numpy().copy()
    roll = rng.random(n)
    uom[(po["base_uom"] == "kg") & (roll < 0.15)] = "ton"
    uom[(po["base_uom"] == "kg") & (roll > 0.97)] = "g"
    factor = np.select([uom == "ton", uom == "g"], [1000.0, 0.001], default=1.0)
    qty = po["qty"].to_numpy() / factor
    price_local = po["unit_price_usd"].to_numpy() * rate * factor

    cur_text = [rng.choice(CURRENCY_SPELLINGS[c]) for c in po["currency"]]
    out = pd.DataFrame({
        "po_number": po["po_number"],
        "line_no": po["line_no"],
        "vendor_name": [messy_vendor_name(v, rng) for v in po["vendor_name"]],
        "material_code": [m.lower() if rng.random() < 0.1 else (f" {m} " if rng.random() < 0.05
                          else m) for m in po["material_id"]],
        "description": po["description"],
        "qty": [(f"{q:,.6f}" if rng.random() < 0.2 else f"{q:.6f}").rstrip("0").rstrip(".")
                for q in qty],
        "uom": [rng.choice(UOM_SPELLINGS[u]) for u in uom],
        "unit_price": [messy_price(p, c, rng) for p, c in zip(price_local, cur_text, strict=True)],
        "currency": cur_text,
        "po_date": [messy_date(d, rng) for d in po["po_date"]],
        "promised_date": [messy_date(d, rng) for d in po["promised_date"]],
        "received_date": [messy_date(d, rng) for d in po["received_date"]],
        "qc_pass_rate": po["qc_pass_rate"],
    })

    # Blanks, returns entered as negative quantities, and decimal-point slips.
    out.loc[rng.random(n) < 0.005, "qty"] = ""
    out.loc[rng.random(n) < 0.005, "unit_price"] = ""
    out.loc[rng.random(n) < 0.03, "received_date"] = ""  # goods not yet received
    neg = rng.random(n) < 0.005
    out.loc[neg, "qty"] = "-" + out.loc[neg, "qty"]
    slip = (rng.random(n) < 0.003) & (out["unit_price"] != "")
    out.loc[slip, "unit_price"] = [f"{p * 100:.6f}" for p in price_local[slip]]
    out.loc[rng.random(n) < 0.002, "vendor_name"] = "MISC VENDOR - SEE NOTE"

    # The same line exported twice.
    dupes = out.sample(frac=0.02, random_state=int(rng.integers(0, 1 << 31)))
    return pd.concat([out, dupes]).sample(frac=1, random_state=7).reset_index(drop=True)
