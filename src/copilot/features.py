"""Build model features for a purchase or a quote.

A should-cost estimate answers "what should this material cost on this date at
this quantity", so the features describe the material, the market and the
volume, and deliberately leave out who the vendor is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["material_code", "category_code", "is_per_kg", "log_qty", "log_qty_vs_ref",
            "index_lag0", "index_lag1", "index_lag3", "index_change_3m", "labor_lag1"]
CATEGORICAL = ["material_code", "category_code"]


def build_features(rows: pd.DataFrame, materials: pd.DataFrame, indices: pd.DataFrame,
                   date_col: str) -> pd.DataFrame:
    """rows needs material_id, qty and a date column. Returns one feature row per input row."""
    mats = materials.reset_index(drop=True)
    mat_code = {m: i for i, m in enumerate(mats["material_id"])}
    cat_code = {c: i for i, c in enumerate(sorted(mats["category"].unique()))}
    df = rows[["material_id", "qty", date_col]].merge(
        mats[["material_id", "category", "index_id", "base_uom", "ref_qty"]],
        on="material_id", how="left", validate="many_to_one")

    idx = indices.sort_values("month").reset_index(drop=True)
    month = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    pos = np.searchsorted(idx["month"].to_numpy(), month.to_numpy())
    pos = np.clip(pos, 0, len(idx) - 1)

    def level(lag: int) -> np.ndarray:
        p = np.clip(pos - lag, 0, None)
        out = np.empty(len(df))
        for name in df["index_id"].unique():
            mask = (df["index_id"] == name).to_numpy()
            out[mask] = idx[name].to_numpy()[p[mask]]
        return out

    out = pd.DataFrame({
        "material_code": df["material_id"].map(mat_code).astype("int64"),
        "category_code": df["category"].map(cat_code).astype("int64"),
        "is_per_kg": (df["base_uom"] == "kg").astype("int64"),
        "log_qty": np.log(df["qty"].astype(float)),
        "log_qty_vs_ref": np.log(df["qty"].astype(float) / df["ref_qty"]),
        "index_lag0": level(0), "index_lag1": level(1), "index_lag3": level(3),
        "labor_lag1": idx["labor_index"].to_numpy()[np.clip(pos - 1, 0, None)],
    })
    out["index_change_3m"] = out["index_lag0"] / out["index_lag3"] - 1
    return out[FEATURES]
