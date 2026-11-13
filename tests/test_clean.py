import pandas as pd
import pytest

from copilot import config
from copilot.pipeline import clean


@pytest.mark.parametrize("raw", [
    "Siam Steel Co., Ltd.", "  SIAM STEEL CO.,LTD ", "siam steel company limited", "Siam Steel",
])
def test_vendor_name_variants_normalise_to_one_key(raw):
    assert clean.normalize_name(raw) == "siam steel"


@pytest.mark.parametrize("text,expected", [
    ("2023-04-05", "2023-04-05"),
    ("05/04/2023", "2023-04-05"),   # day first
    ("5 Apr 2023", "2023-04-05"),
    ("05/04/2566", "2023-04-05"),   # Thai Buddhist era
])
def test_parse_date_formats(text, expected):
    assert clean.parse_date(text) == pd.Timestamp(expected)


def test_parse_date_rejects_garbage():
    assert pd.isna(clean.parse_date("31/02/2023"))
    assert pd.isna(clean.parse_date(""))


def test_parse_number_strips_symbols_and_separators():
    out = clean.parse_number(pd.Series(["฿1,234.50", "US$9.68", "", "-12"]))
    assert out.iloc[0] == 1234.5 and out.iloc[1] == 9.68
    assert pd.isna(out.iloc[2]) and out.iloc[3] == -12


def test_fuzzy_match_recovers_typo_but_not_unknown_vendor():
    master = pd.DataFrame({"vendor_id": ["V-1"], "vendor_name": ["Siam Precision Co., Ltd."]})
    ids = clean.resolve_vendors(pd.Series(["Siam Preicsion", "MISC VENDOR - SEE NOTE"]), master)
    assert ids.iloc[0] == "V-1" and pd.isna(ids.iloc[1])


def test_pipeline_recovers_the_true_values(world):
    report = clean.run()
    cleaned = pd.read_parquet(config.CURATED / "po_lines.parquet")
    truth = world["po_clean"]
    merged = cleaned.merge(truth, on=["po_number", "line_no"], suffixes=("", "_true"))

    assert not cleaned.duplicated(["po_number", "line_no"]).any()
    assert report["rows_out"] + report["rows_quarantined"] + report["duplicates_removed"] \
        == report["rows_in"]
    assert (merged["vendor_id"] == merged["vendor_id_true"]).mean() > 0.995
    assert (merged["po_date"] == merged["po_date_true"]).all()
    price_error = (merged["unit_price_usd"] / merged["unit_price_usd_true"] - 1).abs()
    qty_error = (merged["qty"] / merged["qty_true"] - 1).abs()
    assert price_error.max() < 0.01 and qty_error.max() < 1e-6
    assert (cleaned["qty"] > 0).all()


def test_rejected_rows_keep_a_reason(world):
    clean.run()
    rejected = pd.read_csv(config.QUARANTINE / "po_lines_rejected.csv")
    assert len(rejected) > 0 and rejected["reason"].notna().all()
