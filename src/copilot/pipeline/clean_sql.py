"""The purchase-order cleaning step as BigQuery SQL (sql/bigquery/clean_po_lines.sql).

Two ways to run it:

- `run_bigquery`: on BigQuery, against the raw export loaded as a STRING table.
- `run_duckdb`: the same SQL translated to DuckDB, with the four small helper
  functions written out by hand. No cloud account is needed, so the tests use
  it to check the SQL's logic against the pandas version on every commit.

Vendor names are resolved by the same Python resolver as the other versions
and passed in as a small `vendor_name_map` table. Fuzzy matching against a
master list is a lookup a data steward maintains, not something to redo in SQL
for every row.
"""

from __future__ import annotations

import time

import pandas as pd

from copilot import config
from copilot.pipeline.clean import resolve_vendors

SQL_FILE = config.ROOT / "sql" / "bigquery" / "clean_po_lines.sql"
FINAL = "SELECT * FROM cleaned WHERE reason IS NULL;"
# Counts and totals over every output column, so the engine has to compute all of
# them; used for timing and for comparing versions at sizes too big to download.
SUMMARY = """SELECT reason, COUNT(*) AS n_rows, SUM(qty) AS sum_qty,
       SUM(unit_price_usd) AS sum_unit_price_usd, COUNT(po_date) AS n_po_date,
       COUNT(promised_date) AS n_promised_date, COUNT(received_date) AS n_received_date,
       SUM(qc_pass_rate) AS sum_qc_pass_rate, COUNT(DISTINCT vendor_id) AS n_vendors
FROM cleaned GROUP BY reason ORDER BY reason;"""

DUCKDB_HELPERS = r"""
CREATE OR REPLACE MACRO be_to_ce(y) AS CASE WHEN y > 2400 THEN y - 543 ELSE y END;
CREATE OR REPLACE MACRO month_number(m) AS
  list_position(['jan','feb','mar','apr','may','jun',
                 'jul','aug','sep','oct','nov','dec'], lower(m));
CREATE OR REPLACE MACRO safe_ymd(y, m, d) AS try_cast(printf('%04d-%02d-%02d', y, m, d) AS DATE);
CREATE OR REPLACE MACRO parse_erp_date(raw) AS CASE
  WHEN regexp_full_match(trim(raw), '\d{4}-\d{2}-\d{2}') THEN try_cast(trim(raw) AS DATE)
  WHEN regexp_full_match(trim(raw), '\d{1,2}/\d{1,2}/\d{4}') THEN safe_ymd(
    be_to_ce(CAST(regexp_extract(trim(raw), '/(\d{4})$', 1) AS INT)),
    CAST(regexp_extract(trim(raw), '^\d{1,2}/(\d{1,2})/', 1) AS INT),
    CAST(regexp_extract(trim(raw), '^(\d{1,2})/', 1) AS INT))
  WHEN regexp_full_match(trim(raw), '\d{1,2} [A-Za-z]{3} \d{4}') THEN safe_ymd(
    be_to_ce(CAST(regexp_extract(trim(raw), ' (\d{4})$', 1) AS INT)),
    month_number(regexp_extract(trim(raw), ' ([A-Za-z]{3}) ', 1)),
    CAST(regexp_extract(trim(raw), '^(\d{1,2}) ', 1) AS INT))
  END;
CREATE OR REPLACE MACRO parse_number(raw) AS
  try_cast(regexp_replace(raw, '[^0-9.\-]', '', 'g') AS DOUBLE);
"""


def render(raw_dataset: str, dataset: str, source: str = "erp_po_lines",
           summary: bool = False) -> str:
    text = SQL_FILE.read_text(encoding="utf-8")
    if summary:
        text = text.replace(FINAL, SUMMARY)
    return (text.replace("{raw}", raw_dataset).replace("{dataset}", dataset)
            .replace("{source}", source))


def vendor_name_map(names: pd.Series, vendors: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct raw vendor name that resolves to a vendor."""
    names = pd.Series(pd.unique(names), dtype=str)
    out = pd.DataFrame({"vendor_name": names,
                        "vendor_id": resolve_vendors(names, vendors).astype(object)})
    return out[out["vendor_id"].notna()].reset_index(drop=True)


def run_duckdb(erp: pd.DataFrame, vendors: pd.DataFrame, indices: pd.DataFrame,
               summary: bool = False) -> pd.DataFrame:
    """Run the BigQuery SQL on DuckDB. Only the statement after the helper
    functions is translated; the helpers are the hand-written macros above."""
    import duckdb
    import sqlglot

    text = render("", "", "erp_po_lines", summary=summary).replace("`.", "`")
    body = text[text.index("WITH deduplicated"):].strip().rstrip(";")
    query = sqlglot.transpile(body.replace("``", ""), read="bigquery", write="duckdb")[0]
    con = duckdb.connect()
    con.execute(DUCKDB_HELPERS)
    con.register("erp_po_lines", erp)
    con.register("vendor_name_map", vendor_name_map(erp["vendor_name"], vendors))
    con.register("market_indices", indices)
    return con.execute(query).df()


def load_raw(path, table: str, vendors: pd.DataFrame) -> dict:
    """Upload a raw ERP export (every column as STRING) and the vendor-name map
    into the raw dataset. Returns rows and seconds taken."""
    from google.cloud import bigquery

    from copilot.cloud import bigquery as bq

    s = bq.settings()
    client = bq.client()
    raw = bq.ensure_dataset(s["raw_dataset"])
    header = pd.read_csv(path, nrows=0).columns
    start = time.perf_counter()
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
        allow_quoted_newlines=False, write_disposition="WRITE_TRUNCATE",
        schema=[bigquery.SchemaField(c, "STRING") for c in header])
    with open(path, "rb") as f:
        client.load_table_from_file(f, f"{raw}.{table}", job_config=job_config).result()
    erp_names = pd.read_csv(path, usecols=["vendor_name"], dtype=str,
                            keep_default_na=False)["vendor_name"]
    client.load_table_from_dataframe(
        vendor_name_map(erp_names, vendors), f"{raw}.vendor_name_map",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")).result()
    return {"rows": int(client.get_table(f"{raw}.{table}").num_rows),
            "load_seconds": round(time.perf_counter() - start, 1)}


def run_bigquery(table: str, summary: bool = True) -> tuple[pd.DataFrame, dict]:
    """Run the cleaning SQL on BigQuery. Returns the result and job statistics."""
    from google.cloud import bigquery

    from copilot.cloud import bigquery as bq

    s = bq.settings()
    client = bq.client()
    sql = render(f"{s['project']}.{s['raw_dataset']}", f"{s['project']}.{s['dataset']}",
                 table, summary=summary)
    start = time.perf_counter()
    job = client.query(sql, job_config=bigquery.QueryJobConfig(use_query_cache=False))
    rows = job.result().to_dataframe(create_bqstorage_client=False)
    # A script runs as a parent job; its children hold the statistics.
    children = list(client.list_jobs(parent_job=job.job_id))
    stats = {"wall_ms": round(1000 * (time.perf_counter() - start), 0),
             "bytes_processed": int(sum(c.total_bytes_processed or 0 for c in children)
                                    or job.total_bytes_processed or 0),
             "slot_ms": int(sum(getattr(c, "slot_millis", 0) or 0 for c in children)
                            or job.slot_millis or 0)}
    return rows, stats
