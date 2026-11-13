import pandas as pd
import pytest

from copilot import config
from copilot.model import should_cost
from copilot.quote_check import check_quote


def test_model_beats_the_last_price_rule_of_thumb(trained):
    assert trained["mape_model_a_cost_formula"] < trained["mape_baseline_last_price"]
    assert trained["mape_model_a_cost_formula"] < 0.10


def test_band_is_ordered_and_flags_are_useful(trained):
    model = should_cost.load_model()
    assert 0.04 <= model.flag_threshold <= 0.30
    assert trained["flag_precision"] > 0.7 and trained["flag_recall"] > 0.7


def test_vendor_scores_are_in_range(trained):
    scores = pd.read_parquet(config.SERVING / "vendor_score.parquet")
    assert scores["total_score"].between(0, 100).all()
    assert scores["vendor_id"].is_unique


def test_check_quote_verdicts(trained):
    fair = check_quote("MAT-0001", 2000, 1.0, date="2025-06-15")["should_cost_usd"]
    assert fair["p10"] < fair["p50"] < fair["p90"]
    high = check_quote("MAT-0001", 2000, fair["p50"] * 1.4, date="2025-06-15")
    ok = check_quote("MAT-0001", 2000, fair["p50"], date="2025-06-15")
    assert high["verdict"] == "OVERPRICED" and high["potential_saving_usd"] > 0
    assert ok["verdict"] == "WITHIN EXPECTED RANGE"


def test_check_quote_converts_currency(trained):
    usd = check_quote("MAT-0001", 2000, 2.0, "USD", date="2025-06-15")
    thb = check_quote("MAT-0001", 2000, 2.0, "THB", date="2025-06-15")
    assert thb["quoted_price_usd"] < usd["quoted_price_usd"] / 20


def test_check_quote_rejects_unknown_material(trained):
    with pytest.raises(ValueError):
        check_quote("MAT-9999", 100, 1.0)
