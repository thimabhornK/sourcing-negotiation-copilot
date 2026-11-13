import pandas as pd

from copilot import config, tools
from copilot.contracts.extract import extract_terms
from copilot.contracts.search import load_index, normalise


def test_every_contract_is_split_into_twelve_clauses(trained):
    chunks = pd.read_parquet(config.CURATED / "contract_chunks.parquet")
    assert (chunks.groupby("contract_id").size() == 12).all()
    assert chunks["vendor_id"].notna().all()


def test_vocabulary_maps_buyer_words_to_contract_words():
    assert "late" in normalise("What if the shipment is delayed?")
    assert "terminate" in normalise("Can we cancel?")


def test_search_finds_the_right_clause_and_cites_it(trained):
    hit = load_index().search("What happens if they deliver late?", "V-0001", k=1)[0]
    assert hit["title"] == "Delivery and Late Delivery"
    assert hit["citation"] == "CT-0001 clause 6"


def test_search_stays_inside_the_vendor_asked_about(trained):
    hits = load_index().search("payment terms", "V-0005", k=3)
    assert {h["vendor_id"] for h in hits} == {"V-0005"}


def test_extracted_terms_match_the_ground_truth(trained):
    truth = pd.read_csv(config.RAW / "eval_contract_terms.csv").set_index("vendor_id")
    for vendor_id in truth.index[:20]:
        got, want = extract_terms(vendor_id), truth.loc[vendor_id]
        assert got["payment_days"] == want["payment_days"]
        assert got["termination_notice_days"] == want["termination_notice_days"]
        assert got["index_linked"] == bool(want["index_linked"])
        assert len(got["rebate_tiers"]) == int(want["rebate_tier1_usd"] > 0) + int(
            want["rebate_tier2_usd"] > 0)


def test_rebate_status_adds_up(trained):
    truth = pd.read_csv(config.RAW / "eval_contract_terms.csv")
    vendor_id = truth[truth["rebate_tier1_usd"] > 0]["vendor_id"].iloc[0]
    status = tools.rebate_status(vendor_id, 2025)
    assert status["has_rebate"] and status["citation"].endswith("clause 5")
    if "gap_to_next_tier_usd" in status:
        assert abs(status["next_tier_usd"] - status["spend_usd"]
                   - status["gap_to_next_tier_usd"]) <= 1
