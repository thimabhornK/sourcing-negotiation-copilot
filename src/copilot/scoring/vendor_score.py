"""Build the vendor scorecard with SQL (DuckDB locally, BigQuery in the cloud).

Run with:  python -m copilot.scoring.vendor_score
"""

from __future__ import annotations

import duckdb
import pandas as pd

from copilot import config
from copilot.features import build_features
from copilot.model.should_cost import load_model

# Weights are configuration, not code: a raw-material buyer may care more about
# price, a spare-parts buyer more about delivery.
WEIGHTS = {"w_delivery": 0.30, "w_quality": 0.25, "w_price": 0.25, "w_risk": 0.20}
SQL_PATH = config.ROOT / "sql" / "vendor_score.sql"


def add_should_cost(po: pd.DataFrame, materials: pd.DataFrame, indices: pd.DataFrame
                    ) -> pd.DataFrame:
    """Attach the should-cost estimate to every purchase line."""
    X = build_features(po, materials, indices, "po_date")
    return po.assign(should_cost_p50=load_model().predict(X)["p50"].to_numpy())


def score_vendors(po: pd.DataFrame, vendors: pd.DataFrame, materials: pd.DataFrame,
                  indices: pd.DataFrame, weights: dict[str, float] | None = None,
                  po_scored: pd.DataFrame | None = None) -> pd.DataFrame:
    if po_scored is None:
        po_scored = add_should_cost(po, materials, indices)
    con = duckdb.connect()
    con.register("po_scored", po_scored)
    con.register("vendors", vendors)
    return con.execute(SQL_PATH.read_text(), weights or WEIGHTS).df()


def run() -> pd.DataFrame:
    read = lambda name: pd.read_parquet(config.CURATED / f"{name}.parquet")  # noqa: E731
    materials, indices = read("materials"), read("market_indices")
    po_scored = add_should_cost(read("po_lines"), materials, indices)
    scores = score_vendors(po_scored, read("vendors"), materials, indices, po_scored=po_scored)
    config.SERVING.mkdir(parents=True, exist_ok=True)
    po_scored.to_parquet(config.SERVING / "po_scored.parquet", index=False)
    scores.to_parquet(config.SERVING / "vendor_score.parquet", index=False)
    return scores


if __name__ == "__main__":
    table = run()
    cols = ["vendor_id", "vendor_name", "on_time_rate", "quality_rate", "price_ratio",
            "total_score"]
    print("Top 5 vendors\n", table[cols].head(5).to_string(index=False))
    print("\nBottom 5 vendors\n", table[cols].tail(5).to_string(index=False))
