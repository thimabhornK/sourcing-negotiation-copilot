import pytest

from copilot.negotiation import simulator as sim
from copilot.negotiation.playbook import build_playbook, draft_email
from copilot.quote_check import check_quote


@pytest.fixture(scope="module")
def overpriced(trained):
    p50 = check_quote("MAT-0001", 2000, 1.0, date="2025-06-15")["should_cost_usd"]["p50"]
    return build_playbook("MAT-0001", 2000, round(p50 * 1.3, 4), "V-0001", date="2025-06-15")


def test_playbook_prices_are_in_a_sensible_order(overpriced):
    pb = overpriced
    quoted = pb["quote"]["quoted_price_usd"]
    assert pb["opening_ask_usd"] < pb["target_price_usd"] <= pb["walk_away_usd"] <= quoted
    assert pb["saving_at_target_usd"] > 0
    assert any(lv["lever"] == "should_cost" for lv in pb["levers"])


def test_fair_quote_is_not_pushed_hard(trained):
    p50 = check_quote("MAT-0001", 2000, 1.0, date="2025-06-15")["should_cost_usd"]["p50"]
    pb = build_playbook("MAT-0001", 2000, round(p50 * 0.98, 4), "V-0001", date="2025-06-15")
    assert pb["price_is_fair"]
    assert pb["target_price_usd"] >= 0.98 * pb["quote"]["quoted_price_usd"]


def test_email_does_not_leak_internal_numbers_or_competitor_names(overpriced):
    email = draft_email(overpriced)
    assert f"{overpriced['quote']['should_cost_usd']['p50']:.4f}" not in email
    for alt in overpriced["alternatives"]:
        assert alt["vendor_id"] not in email
        assert f"{alt['price_usd']:.4f}" not in email


def test_vendor_never_goes_below_its_limit():
    for persona in sim.PERSONAS:
        vendor = sim.VendorSimulator(quote_usd=1.60, fair_price_usd=1.40, persona=persona,
                                     competing_price_usd=1.30, market_change_pct=-5.0)
        for _ in range(sim.MAX_ROUNDS):
            turn = vendor.respond(0.50, set(sim.EVIDENCE_WORDS), should_cost_usd=1.35)
        assert turn["status"] == "final_offer"
        assert turn["price_usd"] >= 1.40 * (1 - 0.04) - 1e-6
        assert turn["price_usd"] < 1.60


def test_evidence_moves_the_vendor_more_than_asking():
    def first_counter(evidence):
        vendor = sim.VendorSimulator(1.60, 1.40, "flexible", market_change_pct=-5.0)
        return vendor.respond(1.00, evidence, should_cost_usd=1.35)["price_usd"]

    assert first_counter({"should_cost", "index_trend"}) < first_counter(set())


def test_false_evidence_is_ignored():
    vendor = sim.VendorSimulator(1.60, 1.40, "flexible", market_change_pct=+6.0)
    turn = vendor.respond(1.00, {"index_trend", "competing_offer"}, should_cost_usd=None)
    assert turn["credible_evidence"] == []


def test_vendor_accepts_a_good_enough_offer():
    vendor = sim.VendorSimulator(1.60, 1.40, "flexible")
    assert vendor.respond(1.59)["status"] == "accepted"
    assert vendor.closed


def test_parse_buyer_message():
    offer, evidence = sim.parse_buyer_message(
        "We can offer 1.35. Our cost model and a competing quote both point lower.")
    assert offer == 1.35
    assert {"should_cost", "competing_offer"} <= evidence


def test_playbook_strategy_beats_asking_for_five_percent(overpriced):
    quoted = overpriced["quote"]["quoted_price_usd"]
    results = {}
    for strategy in ("ask_5pct", "playbook"):
        vendor = sim.VendorSimulator(quoted, quoted / 1.3, "firm", market_change_pct=-3.0)
        results[strategy] = sim.run_auto(vendor, strategy, overpriced)["saving_pct"]
    assert results["playbook"] > results["ask_5pct"]
