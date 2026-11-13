import pandas as pd
import pytest

from copilot.agents.llm import OfflineLLM
from copilot.cloud import bigquery as bq
from copilot.eval import live

QUERIES = ["vendor_score", "top_vendors_by_spend", "most_expensive_vendors", "late_vendors",
           "spend_by_category"]


@pytest.mark.parametrize("name", QUERIES)
def test_bigquery_sql_is_valid_syntax(name):
    sqlglot = pytest.importorskip("sqlglot")
    sql = bq.render_sql(name, "my-project.sourcing_copilot", limit=7)
    assert "{" not in sql and "my-project.sourcing_copilot.po_scored" in sql
    sqlglot.parse_one(sql, read="bigquery")


def test_limit_is_always_an_integer():
    assert "LIMIT 7" in bq.render_sql("late_vendors", "p.d", limit="7")
    with pytest.raises(ValueError):
        bq.render_sql("late_vendors", "p.d", limit="7; DROP TABLE x")


def test_null_parameters_get_a_declared_type():
    pytest.importorskip("google.cloud.bigquery")
    params = {p.name: (p.type_, p.value) for p in bq.query_parameters(
        {"year": None, "category": "steel_sheet", "n": 5, "w_price": 0.25})}
    assert params == {"year": ("INT64", None), "category": ("STRING", "steel_sheet"),
                      "w_price": ("FLOAT64", 0.25)}


def test_compare_accepts_rounding_noise_and_rejects_real_differences():
    local = pd.DataFrame({"vendor_id": ["V-1", "V-2"], "spend_usd": [9_182_934.0, 100.0],
                          "score": [80.1, 70.0]})
    assert bq.compare(local, local.assign(score=[80.2, 70.0]))["match"]
    assert not bq.compare(local, local.assign(score=[81.5, 70.0]))["match"]
    assert not bq.compare(local, local.assign(vendor_id=["V-2", "V-1"]))["match"]
    assert not bq.compare(local, local.head(1))["match"]


def test_every_bigquery_query_has_a_local_counterpart(trained):
    names = {name for name, _, _ in bq.parity_cases()}
    assert names == set(QUERIES)
    for _, _, local_fn in bq.parity_cases():
        assert len(local_fn()) > 0


class CountsTokens(OfflineLLM):
    name = "fake-model"

    def write(self, question, draft, facts, span, feedback=""):
        span.update(tokens_in=120, tokens_out=40, cost_usd=0.0002)
        return draft


def test_live_evaluation_harness_runs_with_any_model(trained):
    report = live.run(CountsTokens(), pause=0, limit=6)
    answers = report["answers"]
    assert report["model"] == "fake-model" and report["routing"]["cases"] == 6
    assert answers["turns"] == 6 and answers["passed_number_review_first_time"] == 1.0
    assert answers["tokens_per_turn"] == {"in": 120, "out": 40}
    assert answers["cost_per_1000_turns_usd"] == 0.2
