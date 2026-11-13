import json

import pytest

from copilot.agents.llm import OfflineLLM
from copilot.agents.orchestrator import Session
from copilot.agents.reviewer import review
from copilot.agents.router import extract_params, route_by_rules
from copilot.agents.tracing import Tracer
from copilot.eval.cases import ROUTING

QUOTE = "Is 1.80 USD fair for 2000 kg of MAT-0001 from V-0001 on 2025-06-15?"


@pytest.fixture()
def session(trained):
    return Session(llm=OfflineLLM())


def test_params_are_pulled_out_of_a_message(trained):
    p = extract_params("V-0010 quoted ฿55 for 2 tons of MAT-0001 on 2025-06-15")
    assert p == {"material_id": "MAT-0001", "vendor_id": "V-0010", "date": "2025-06-15",
                 "qty": 2000.0, "price": 55.0, "currency": "THB"}


def test_vendor_can_be_named_instead_of_numbered(trained, world):
    name = world["vendors"].iloc[4]["vendor_name"].replace("Co., Ltd.", "").strip()
    assert extract_params(f"How is {name.upper()} doing?")["vendor_id"] == "V-0005"


def test_routing_rules_handle_most_of_the_regression_set(trained):
    right = sum(route_by_rules(m, extract_params(m)) == want for m, want in ROUTING)
    assert right / len(ROUTING) >= 0.9


def test_reviewer_accepts_numbers_from_facts_and_rejects_others():
    facts = {"price": 1.4567, "on_time_rate": 0.82, "note": "spend is 12,500 USD"}
    assert review("Price 1.46 USD, 82% on time, spend 12,500.", facts)["ok"]
    bad = review("Price 1.52 USD, 82% on time.", facts)
    assert not bad["ok"] and bad["unsupported_numbers"] == ["1.52"]


def test_reviewer_ignores_identifiers_and_dates():
    assert review("See CT-0042 clause 5 for MAT-0007 on 2025-06-15.", {})["ok"]


def test_every_mode_answers_and_passes_review(session):
    script = [("Top 3 vendors by spend in 2025", "top_vendors"),
              ("Who delivers late the most?", "late_vendors"),
              ("How is V-0009 performing?", "vendor_profile"),
              ("What are the payment terms with V-0003?", "contract_question"),
              (QUOTE, "check_quote"),
              ("Build a negotiation playbook for that", "negotiate"),
              ("Run the negotiation automatically", "simulate_auto")]
    for message, intent in script:
        reply = session.handle(message)
        assert reply.intent == intent
        assert reply.review["ok"], (message, reply.review)
        assert reply.facts and reply.trace[0]["agent"] == "supervisor"
    assert "CT-0003 clause 4" in session.handle("payment terms in the V-0003 contract?").text


def test_follow_up_reuses_the_last_quote(session):
    session.handle(QUOTE)
    reply = session.handle("How should I negotiate this?")
    assert reply.intent == "negotiate"
    assert reply.facts[0]["quote"]["quoted_price_usd"] == 1.80
    assert "Open at" in reply.text


def test_missing_details_are_asked_for(session):
    reply = session.handle("Is this quote fair?")
    assert reply.intent == "check_quote" and "I need" in reply.text


def test_unknown_material_is_reported_not_raised(session):
    assert "Unknown material" in session.handle("Is 2 USD fair for 100 kg of MAT-9999?").text


def test_practice_negotiation_runs_to_a_debrief(session):
    session.handle(QUOTE)
    assert session.handle("Let me practise against the vendor").intent == "simulate_start"
    reply = session.handle("We can offer 1.20. Our cost model points lower.")
    assert reply.intent == "simulate_turn" and "Coach:" in reply.text
    end = session.handle("stop")
    assert end.intent == "simulate_end" and "Debrief" in end.text
    assert session.handle("hello").intent == "help"


class MakesUpANumber(OfflineLLM):
    """A writer that invents a figure on its first attempt, or on every attempt."""

    def __init__(self, always: bool) -> None:
        self.always, self.calls = always, 0

    def write(self, question, draft, facts, span, feedback=""):
        self.calls += 1
        return draft + "\nThat is 987,654.32 USD a year." if self.always or not feedback \
            else draft


def test_reviewer_sends_a_made_up_number_back_to_the_writer(trained):
    llm = MakesUpANumber(always=False)
    reply = Session(llm=llm).handle("Top 3 vendors by spend in 2025")
    assert llm.calls == 2 and reply.review["ok"] and "987,654.32" not in reply.text


def test_answer_falls_back_to_the_draft_after_two_failed_reviews(trained):
    llm = MakesUpANumber(always=True)
    reply = Session(llm=llm).handle("Top 3 vendors by spend in 2025")
    assert llm.calls == 2 and reply.review["fell_back_to_draft"]
    assert "987,654.32" not in reply.text


def test_traces_are_written_as_json_lines(trained, tmp_path):
    path = tmp_path / "trace.jsonl"
    Session(llm=OfflineLLM(), tracer=Tracer(path)).handle("Who delivers late the most?")
    spans = [json.loads(line) for line in path.read_text().splitlines()]
    assert [s["agent"] for s in spans] == ["supervisor", "spend_analyst", "writer", "reviewer"]
    assert all(s["ms"] >= 0 and "tokens_in" in s for s in spans)


def test_adk_agents_and_tool_schemas_build(trained):
    pytest.importorskip("google.adk")
    from google.adk.tools import FunctionTool

    from copilot.agents.adk_app import root_agent

    assert {a.name for a in root_agent.sub_agents} == {
        "spend_analyst", "contract_analyst", "market_analyst", "strategist"}
    for agent in root_agent.sub_agents:
        for tool in agent.tools:
            assert FunctionTool(tool)._get_declaration().name == tool.__name__
