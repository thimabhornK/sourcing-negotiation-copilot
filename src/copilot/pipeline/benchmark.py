"""Time the purchase-order cleaning step on pandas, Spark and BigQuery, and check
that all three give the same answer.

  python -m copilot.pipeline.benchmark                      pandas and Spark, 1x and 10x
  python -m copilot.pipeline.benchmark --scales 1 10 100
  python -m copilot.pipeline.benchmark --bigquery           add BigQuery (sandbox is enough)

Larger inputs are made by repeating the ERP export with new PO numbers, so a
copy at 10x has exactly ten times the rows, duplicates and bad rows of the
original. This stands in for combining the exports of several plants.

Each engine reads the raw CSV, cleans it and computes a summary that touches
every output column (counts by reason, totals). At 1x the full set of clean
rows is also compared row by row. Results go to reports/clean_benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import pandas as pd

from copilot import config
from copilot.pipeline import clean

BENCH = config.DATA / "bench"
KEY = ["po_number", "line_no"]


def scaled_copy(scale: int):
    """Write (once) and return the path of the ERP export repeated `scale` times."""
    src = config.RAW / "erp_po_lines.csv"
    if scale == 1:
        return src
    BENCH.mkdir(parents=True, exist_ok=True)
    path = BENCH / f"erp_po_lines_x{scale}.csv"
    if not path.exists() or path.stat().st_mtime < src.stat().st_mtime:
        raw = pd.read_csv(src, dtype=str, keep_default_na=False)
        with path.open("w", encoding="utf-8", newline="") as f:
            for i in range(scale):
                part = raw if i == 0 else raw.assign(po_number=raw["po_number"] + f"-{i:03d}")
                part.to_csv(f, index=False, header=(i == 0))
    return path


def _inputs():
    vendors = pd.read_csv(config.RAW / "vendor_master.csv", dtype=str)
    indices = pd.read_csv(config.RAW / "market_indices.csv", parse_dates=["month"])
    return vendors, indices


def summarise(good: pd.DataFrame, reasons: pd.Series) -> dict:
    """Counts by reason plus totals over the clean rows, in one engine-neutral shape."""
    out = {"rows_clean": len(good), **{f"quarantined_{k}": int(v)
                                       for k, v in reasons.value_counts().sort_index().items()},
           "sum_qty": float(good["qty"].sum()),
           "sum_unit_price_usd": float(good["unit_price_usd"].sum()),
           "n_po_date": int(good["po_date"].notna().sum()),
           "n_promised_date": int(good["promised_date"].notna().sum()),
           "n_received_date": int(good["received_date"].notna().sum()),
           "sum_qc_pass_rate": float(good["qc_pass_rate"].sum()),
           "n_vendors": int(good["vendor_id"].nunique())}
    return out


def from_sql_summary(rows: pd.DataFrame) -> dict:
    """Turn the SUMMARY query result (one row per reason) into summarise()'s shape."""
    rows = rows.astype({"reason": object})
    good = rows[rows["reason"].isna()].iloc[0]
    out = {"rows_clean": int(good["n_rows"]),
           **{f"quarantined_{r}": int(n) for r, n in
              rows[rows["reason"].notna()].sort_values("reason")[["reason", "n_rows"]]
              .itertuples(index=False)},
           "sum_qty": float(good["sum_qty"]),
           "sum_unit_price_usd": float(good["sum_unit_price_usd"]),
           "n_po_date": int(good["n_po_date"]), "n_promised_date": int(good["n_promised_date"]),
           "n_received_date": int(good["n_received_date"]),
           "sum_qc_pass_rate": float(good["sum_qc_pass_rate"]),
           "n_vendors": int(good["n_vendors"])}
    return out


def same_summary(a: dict, b: dict) -> bool:
    """Counts equal; totals equal to within float addition order (1 part in 1e9)."""
    if a.keys() != b.keys():
        return False
    for k in a:
        if isinstance(a[k], float):
            if abs(a[k] - b[k]) > 1e-9 * max(1.0, abs(a[k])):
                return False
        elif a[k] != b[k]:
            return False
    return True


def same_rows(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Row-by-row comparison of two sets of clean rows, matched on PO number and line."""
    a = a.copy()
    b = b.copy()
    for df in (a, b):
        for c in ["po_date", "promised_date", "received_date"]:
            df[c] = pd.to_datetime(df[c])
        df["line_no"] = df["line_no"].astype("int64")
    m = a.merge(b, on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    out = {"rows_a": len(a), "rows_b": len(b), "unmatched": int((m["_merge"] != "both").sum())}
    m = m[m["_merge"] == "both"]
    worst = 0.0
    different = 0
    for c in [c for c in a.columns if c not in KEY]:
        x, y = m[f"{c}_a"], m[f"{c}_b"]
        if pd.api.types.is_float_dtype(x):
            gap = (x - y.astype(float)).abs()
            worst = max(worst, float(gap.max()))
            different += int(((gap > 1e-9) | (x.isna() != y.isna())).sum())
        else:
            different += int(((x.astype(str) != y.astype(str)) & ~(x.isna() & y.isna())).sum())
    out.update(values_different=different, max_abs_difference=worst,
               match=out["unmatched"] == 0 and different == 0)
    return out


def run_pandas(path, vendors, indices):
    start = time.perf_counter()
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    good, bad, _ = clean.clean_po_lines(raw, vendors, indices)
    summary = summarise(good, bad["reason"])
    return good, summary, time.perf_counter() - start


def run_spark(spark, path, vendors, indices, rows: bool):
    from pyspark.sql import functions as F

    from copilot.pipeline import clean_spark

    start = time.perf_counter()
    result = clean_spark.clean_po_lines(spark, clean_spark.read_raw(spark, path), vendors,
                                        indices)
    agg = (result.groupBy("reason").agg(
        F.count("*").alias("n_rows"), F.sum("qty").alias("sum_qty"),
        F.sum("unit_price_usd").alias("sum_unit_price_usd"),
        F.count("po_date").alias("n_po_date"), F.count("promised_date").alias("n_promised_date"),
        F.count("received_date").alias("n_received_date"),
        F.sum("qc_pass_rate").alias("sum_qc_pass_rate"),
        F.countDistinct("vendor_id").alias("n_vendors")).toPandas())
    seconds = time.perf_counter() - start
    good = result.where("reason IS NULL").drop("reason").toPandas() if rows else None
    return good, from_sql_summary(agg), seconds


def run(scales=(1, 10), engines=("pandas", "spark"), bigquery_scales=(1, 10)) -> dict:
    vendors, indices = _inputs()
    report = {"machine": {"os": platform.system(), "cpu_count": _cpus(),
                          "python": platform.python_version()},
              "note": "Seconds include reading the CSV. Spark runs in local mode on this "
                      "machine; its start-up time is reported once and not included. "
                      "BigQuery time excludes uploading the file.",
              "runs": []}
    spark = None
    if "spark" in engines:
        from copilot.pipeline import clean_spark

        t = time.perf_counter()
        spark = clean_spark.session()
        spark.range(1).count()
        report["spark_startup_seconds"] = round(time.perf_counter() - t, 1)
        report["spark_version"] = spark.version

    for scale in scales:
        path = scaled_copy(scale)
        entry = {"scale": scale}
        base_rows, base, secs = run_pandas(path, vendors, indices)
        entry["rows_in"] = int(sum(1 for _ in path.open(encoding="utf-8")) - 1)
        entry["pandas"] = {"seconds": round(secs, 2), "summary": base}
        print(f"  {scale:>4}x  pandas    {secs:8.1f} s")
        if spark is not None:
            rows, summ, secs = run_spark(spark, path, vendors, indices, rows=scale == 1)
            entry["spark"] = {"seconds": round(secs, 2), "same_summary_as_pandas":
                              same_summary(base, summ)}
            if rows is not None:
                entry["spark"]["row_check"] = same_rows(base_rows, rows)
            print(f"  {scale:>4}x  spark     {secs:8.1f} s  "
                  f"same={entry['spark']['same_summary_as_pandas']}")
        if "bigquery" in engines and scale in bigquery_scales:
            entry["bigquery"] = _bigquery(path, scale, vendors, base, base_rows)
        report["runs"].append(entry)
        del base_rows

    if spark is not None:
        spark.stop()
    # A BigQuery run without Spark goes to its own file, so it does not replace
    # the Spark timings measured on another machine.
    name = "clean_benchmark.json" if spark is not None else "clean_benchmark_bigquery.json"
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["file"] = f"reports/{name}"
    return report


def _bigquery(path, scale, vendors, base, base_rows) -> dict:
    from copilot.pipeline import clean_sql

    table = "erp_po_lines" if scale == 1 else f"erp_po_lines_x{scale}"
    loaded = clean_sql.load_raw(path, table, vendors)
    rows, stats = clean_sql.run_bigquery(table, summary=True)
    out = {"seconds": round(stats["wall_ms"] / 1000, 2), "bytes_processed":
           stats["bytes_processed"], "slot_ms": stats["slot_ms"],
           "upload_seconds": loaded["load_seconds"],
           "same_summary_as_pandas": same_summary(base, from_sql_summary(rows))}
    if scale == 1:
        good, _ = clean_sql.run_bigquery(table, summary=False)
        out["row_check"] = same_rows(base_rows, good.drop(columns="reason"))
    print(f"  {scale:>4}x  bigquery  {out['seconds']:8.1f} s  "
          f"same={out['same_summary_as_pandas']}  ({stats['bytes_processed']:,} bytes)")
    return out


def _cpus() -> int:
    import os

    return os.cpu_count() or 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--no-spark", action="store_true", help="pandas only")
    ap.add_argument("--bigquery", action="store_true",
                    help="also run on BigQuery (needs COPILOT_BQ_PROJECT)")
    ap.add_argument("--bigquery-scales", type=int, nargs="+", default=[1, 10])
    args = ap.parse_args()
    engines = ["pandas"] + ([] if args.no_spark else ["spark"]) + (
        ["bigquery"] if args.bigquery else [])
    report = run(args.scales, engines, args.bigquery_scales)
    ok = all(r[e].get("same_summary_as_pandas", True) and
             r[e].get("row_check", {}).get("match", True)
             for r in report["runs"] for e in ("spark", "bigquery") if e in r)
    print("All engines agree." if ok else f"Engines disagree, see {report['file']}")


if __name__ == "__main__":
    main()
