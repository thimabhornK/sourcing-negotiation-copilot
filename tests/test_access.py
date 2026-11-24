"""The agents' views must never expose identifiers, and the access check must
try both what the agent may and may not read."""

import pandas as pd
import pytest

from copilot import config
from copilot.cloud import access


def test_agent_views_leave_out_every_sensitive_column():
    for columns in access.AGENT_VIEWS.values():
        assert not access.SENSITIVE & set(columns)


def test_agent_views_list_only_columns_that_exist(trained):
    folders = {"po_scored": config.SERVING}
    for table, columns in access.AGENT_VIEWS.items():
        actual = pd.read_parquet(folders.get(table, config.CURATED) / f"{table}.parquet").columns
        assert set(columns) <= set(actual), table


def test_view_sql_is_valid_bigquery():
    sqlglot = pytest.importorskip("sqlglot")
    for table in access.AGENT_VIEWS:
        sqlglot.parse_one(access.view_sql("p.curated", table), read="bigquery")


def test_access_check_expects_views_allowed_and_tables_denied(monkeypatch):
    monkeypatch.setenv("COPILOT_BQ_PROJECT", "p")
    plan = access.attempts()
    assert any(allowed for _, _, allowed in plan)
    denied = [sql for _, sql, allowed in plan if not allowed]
    assert any("_raw." in sql for sql in denied) and any("p.sourcing_copilot." in sql
                                                         for sql in denied)
