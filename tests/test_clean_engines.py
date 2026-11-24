"""The cleaning step has three versions: pandas, Spark and BigQuery SQL. They must agree."""

import re
import shutil

import pandas as pd
import pytest

from copilot import config
from copilot.pipeline import benchmark, clean, clean_sql


@pytest.fixture(scope="module")
def raw_inputs(world):
    vendors = pd.read_csv(config.RAW / "vendor_master.csv", dtype=str)
    indices = pd.read_csv(config.RAW / "market_indices.csv", parse_dates=["month"])
    erp = pd.read_csv(config.RAW / "erp_po_lines.csv", dtype=str, keep_default_na=False)
    good, bad, _ = clean.clean_po_lines(erp, vendors, indices)
    return erp, vendors, indices, good, bad


def test_clean_sql_is_valid_bigquery():
    sqlglot = pytest.importorskip("sqlglot")
    sql = clean_sql.render("p.raw", "p.curated", "erp_po_lines")
    assert not any(p in sql for p in ("{raw}", "{dataset}", "{source}"))
    for summary in (False, True):
        sqlglot.parse(clean_sql.render("p.raw", "p.curated", summary=summary), read="bigquery")


def test_clean_sql_uses_the_same_unit_and_currency_maps_as_python():
    sql = clean_sql.SQL_FILE.read_text(encoding="utf-8")
    pairs = dict(re.findall(r"WHEN '([^']+)' THEN '([A-Za-z]+)'", sql))
    for raw, std in {**clean.CURRENCY_MAP, **clean.UOM_MAP}.items():
        assert pairs[raw] == std
    factors = dict(re.findall(r"WHEN '([a-z]+)' THEN ([0-9.]+)", sql))
    assert {k: float(v) for k, v in factors.items()} == config.UOM_TO_BASE


def test_clean_sql_returns_the_same_rows_as_pandas(raw_inputs):
    pytest.importorskip("sqlglot")
    erp, vendors, indices, good, bad = raw_inputs
    rows = clean_sql.run_duckdb(erp, vendors, indices)
    check = benchmark.same_rows(good, rows.drop(columns="reason"))
    assert check["match"], check
    summary = clean_sql.run_duckdb(erp, vendors, indices, summary=True)
    assert benchmark.same_summary(benchmark.summarise(good, bad["reason"]),
                                  benchmark.from_sql_summary(summary))


def test_scaled_copy_multiplies_every_count(world, raw_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "BENCH", tmp_path)
    erp, vendors, indices, good, bad = raw_inputs
    path = benchmark.scaled_copy(3)
    _, summary, _ = benchmark.run_pandas(path, vendors, indices)
    base = benchmark.summarise(good, bad["reason"])
    assert summary["rows_clean"] == 3 * base["rows_clean"]
    assert summary["quarantined_price_outlier"] == 3 * base["quarantined_price_outlier"]


def test_same_rows_reports_a_changed_value(raw_inputs):
    good = raw_inputs[3]
    changed = good.copy()
    changed.loc[0, "unit_price_usd"] += 0.01
    assert benchmark.same_rows(good, good)["match"]
    assert not benchmark.same_rows(good, changed)["match"]


@pytest.mark.skipif(shutil.which("java") is None, reason="Spark needs Java")
def test_spark_returns_the_same_rows_as_pandas(raw_inputs):
    pytest.importorskip("pyspark")
    from copilot.pipeline import clean_spark

    erp, vendors, indices, good, bad = raw_inputs
    spark = clean_spark.session(cores="2", memory="1g")
    rows, summary, _ = benchmark.run_spark(spark, config.RAW / "erp_po_lines.csv", vendors,
                                           indices, rows=True)
    assert benchmark.same_rows(good, rows)["match"]
    assert benchmark.same_summary(benchmark.summarise(good, bad["reason"]), summary)
