"""Run the serving layer on BigQuery.

  python -m copilot.cloud.bigquery load      upload the curated tables
  python -m copilot.cloud.bigquery parity    run every query on BigQuery and on
                                             DuckDB, compare, and write
                                             reports/bigquery_parity.json

Works in the BigQuery sandbox, which needs no billing account. The sandbox has
no DML and tables expire after 60 days, so tables are written with load jobs
and replaced whole.

Settings (environment variables):
  COPILOT_BQ_PROJECT    Google Cloud project ID (required)
  COPILOT_BQ_DATASET    dataset name, default sourcing_copilot
  COPILOT_BQ_LOCATION   dataset location, default asia-southeast1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import lru_cache

import pandas as pd

from copilot import config

SQL_DIR = config.ROOT / "sql" / "bigquery"
TABLES = {"po_scored": config.SERVING, "vendors": config.CURATED, "materials": config.CURATED,
          "market_indices": config.CURATED, "rfq_quotes": config.CURATED}
BQ_TYPES = {int: "INT64", float: "FLOAT64", str: "STRING"}
# Parameters that may be None need a declared type, because BigQuery cannot infer one.
PARAM_TYPES = {"year": "INT64", "category": "STRING", "material": "STRING"}


def settings() -> dict:
    project = os.environ.get("COPILOT_BQ_PROJECT")
    if not project:
        raise SystemExit("Set COPILOT_BQ_PROJECT to your Google Cloud project ID.")
    return {"project": project,
            "dataset": os.environ.get("COPILOT_BQ_DATASET", "sourcing_copilot"),
            "location": os.environ.get("COPILOT_BQ_LOCATION", "asia-southeast1")}


@lru_cache(maxsize=1)
def client():
    from google.cloud import bigquery

    s = settings()
    return bigquery.Client(project=s["project"], location=s["location"])


def render_sql(name: str, dataset: str, limit: int = 5) -> str:
    """Fill in the dataset and the row limit. The limit is forced to an integer here
    and written into the text, so it does not depend on parameter support in LIMIT."""
    text = (SQL_DIR / f"{name}.sql").read_text()
    return text.replace("{dataset}", dataset).replace("{limit}", str(int(limit)))


def query_parameters(params: dict) -> list:
    from google.cloud import bigquery

    return [bigquery.ScalarQueryParameter(
        key, PARAM_TYPES.get(key) or BQ_TYPES[type(value)], value)
        for key, value in params.items() if key != "n"]


def query(name: str, params: dict) -> tuple[pd.DataFrame, dict]:
    """Run sql/bigquery/<name>.sql. Returns the rows and the job statistics."""
    from google.cloud import bigquery

    s = settings()
    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters(params),
                                         use_query_cache=False)
    start = time.perf_counter()
    sql = render_sql(name, f"{s['project']}.{s['dataset']}", params.get("n", 5))
    job = client().query(sql, job_config=job_config)
    rows = job.result().to_dataframe()
    stats = {"wall_ms": round(1000 * (time.perf_counter() - start), 0),
             "bytes_processed": int(job.total_bytes_processed or 0),
             "slot_ms": int(job.slot_millis or 0)}
    return rows, stats


def load() -> dict:
    """Upload the curated and serving tables, replacing what is there."""
    from google.cloud import bigquery

    s = settings()
    bq = client()
    dataset = bigquery.Dataset(f"{s['project']}.{s['dataset']}")
    dataset.location = s["location"]
    bq.create_dataset(dataset, exists_ok=True)
    job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET,
                                        write_disposition="WRITE_TRUNCATE")
    loaded = {}
    for table, folder in TABLES.items():
        with (folder / f"{table}.parquet").open("rb") as f:
            job = bq.load_table_from_file(f, f"{s['project']}.{s['dataset']}.{table}",
                                          job_config=job_config)
        job.result()
        loaded[table] = int(bq.get_table(f"{s['project']}.{s['dataset']}.{table}").num_rows)
        print(f"  loaded {table:<16} {loaded[table]:>8,} rows")
    return loaded


def compare(local: pd.DataFrame, cloud: pd.DataFrame) -> dict:
    """Same rows in the same order, with numbers equal to within rounding?"""
    out = {"rows_local": len(local), "rows_bigquery": len(cloud)}
    if len(local) != len(cloud) or list(local.columns) != list(cloud.columns):
        return {**out, "match": False, "reason": "different shape or columns"}
    text = [c for c in local.columns if not pd.api.types.is_numeric_dtype(local[c])]
    numeric = [c for c in local.columns if c not in text]
    same_keys = all((local[c].astype(str).to_numpy() == cloud[c].astype(str).to_numpy()).all()
                    for c in text)
    worst_abs = worst_rel = 0.0
    for c in numeric:
        a, b = local[c].astype(float), cloud[c].astype(float)
        gap = (a - b).abs()
        worst_abs = max(worst_abs, float(gap.max()))
        # one unit in the last rounded digit, or one part in a million on large sums
        worst_rel = max(worst_rel, float((gap / a.abs().clip(lower=1.0)).max()))
    close = worst_abs <= 0.11 or worst_rel <= 1e-6
    return {**out, "same_keys_and_order": bool(same_keys),
            "max_abs_difference": round(worst_abs, 6),
            "max_relative_difference": round(worst_rel, 9),
            "match": bool(same_keys and close)}


def parity_cases() -> list[tuple[str, dict, object]]:
    """Each query with its parameters and the local function that answers the same thing."""
    from copilot import tools
    from copilot.scoring.vendor_score import WEIGHTS

    def local_scores():
        return pd.read_parquet(config.SERVING / "vendor_score.parquet")

    def rows(fn, **kw):
        return lambda: pd.DataFrame(fn(**kw)["rows"])

    return [
        ("vendor_score", dict(WEIGHTS), local_scores),
        ("top_vendors_by_spend", {"year": 2025, "category": None, "n": 10},
         rows(tools.top_vendors_by_spend, year=2025, n=10)),
        ("top_vendors_by_spend", {"year": None, "category": "steel_sheet", "n": 5},
         rows(tools.top_vendors_by_spend, category="steel_sheet", n=5)),
        ("most_expensive_vendors",
         {"year": 2025, "category": "steel_sheet", "material": None, "n": 10},
         rows(tools.most_expensive_vendors, year=2025, category="steel_sheet", n=10)),
        ("late_vendors", {"year": 2025, "n": 10}, rows(tools.late_vendors, year=2025, n=10)),
        ("spend_by_category", {"year": 2025}, rows(tools.spend_by_category, year=2025)),
    ]


def parity() -> dict:
    os.environ["COPILOT_BACKEND"] = "local"  # the local side of the comparison
    s = settings()
    results = []
    for name, params, local_fn in parity_cases():
        cloud, stats = query(name, params)
        check = compare(local_fn().reset_index(drop=True), cloud.reset_index(drop=True))
        results.append({"query": name, "params": params, **check, **stats})
        print(f"  {name:<24} match={check['match']}  {stats['wall_ms']:.0f} ms  "
              f"{stats['bytes_processed']:,} bytes")
    report = {"dataset": s["dataset"], "location": s["location"],
              "note": "A difference of one unit in the last rounded digit can occur, because "
                      "the two engines add floating-point numbers in a different order.",
              "all_match": all(r["match"] for r in results), "queries": results}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "bigquery_parity.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the serving layer on BigQuery")
    ap.add_argument("command", choices=["load", "parity"])
    args = ap.parse_args()
    if args.command == "load":
        load()
    else:
        report = parity()
        print("All queries match." if report["all_match"] else "Some queries differ, see "
              "reports/bigquery_parity.json")


if __name__ == "__main__":
    main()
