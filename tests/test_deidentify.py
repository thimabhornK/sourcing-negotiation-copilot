import pandas as pd

from copilot.pipeline import deidentify


def test_tokens_are_stable_and_hide_the_value():
    a, b = deidentify.tokenize("1234567890123"), deidentify.tokenize("1234567890123")
    assert a == b and "1234567890123" not in a
    assert deidentify.tokenize("1234567890124") != a


def test_token_depends_on_the_key(monkeypatch):
    before = deidentify.tokenize("secret")
    monkeypatch.setenv("COPILOT_DEID_KEY", "another-key")
    assert deidentify.tokenize("secret") != before


def test_pcard_output_has_no_card_number_or_name():
    raw = pd.DataFrame({"txn_id": ["T1"], "card_pan": ["4000123412341234"],
                        "cardholder_name": ["Somchai Srisuk"], "amount_thb": [100.0]})
    out = deidentify.deidentify_pcard(raw)
    assert "card_pan" not in out and "cardholder_name" not in out
    assert out.loc[0, "card_last4"] == "1234"
    assert "4000123412341234" not in out.to_csv() and "Somchai" not in out.to_csv()


def test_email_is_masked():
    assert deidentify.mask_email("sales@vendor.example") == "s***@vendor.example"
