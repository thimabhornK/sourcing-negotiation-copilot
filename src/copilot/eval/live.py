"""Evaluate the language-model path and write reports/agent_eval_live.json.

  export GEMINI_API_KEY=...
  python -m copilot.eval.live              full run, about 10 minutes on a free key
  python -m copilot.eval.live --limit 10   a quick look

The offline evaluation (copilot.eval.run) covers everything that does not need
a model. This one measures what only a real model can show: whether it routes
better than the rules, whether its wording keeps the facts intact, and what a
turn costs in tokens, money and time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import time

from copilot import config
from copilot.agents.llm import GeminiLLM
from copilot.agents.orchestrator import Session
from copilot.agents.router import extract_params, route_by_rules
from copilot.eval.cases import ROUTING

QUOTE = "Is 1.62 USD fair for 5000 kg of MAT-0001 from V-0002 on 2025-06-15?"
CONVERSATIONS = [
    ["Top 5 vendors by spend in 2025",
     "Which vendors charge the most above should-cost for steel sheet in 2025?",
     "Who delivers late the most in 2025?", "Break down 2025 spend by category"],
    ["How is V-0009 performing?", "What are the payment terms with V-0003?",
     "What happens if V-0002 delivers late?", "Are we close to a rebate with V-0002 in 2025?",
     "How much notice do we need to give to end the contract with V-0010?"],
    [QUOTE, "Build a negotiation playbook for that", "Run the negotiation automatically"],
    ["Compare the quotes in RFQ-01990", "How do I negotiate RFQ-01990?"],
    ["Is 4.10 USD a good price for 800 pcs of MAT-0012 from V-0007 on 2025-03-10?",
     "What should we offer back?"],
]
CITATION = re.compile(r"CT-\d{4} clause \d+")
VERDICT = re.compile(r"OVERPRICED|WITHIN EXPECTED RANGE|BELOW EXPECTED RANGE")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)] if ordered else 0.0


def _share(turns: list[dict], applies: str, kept: str) -> dict:
    relevant = [t for t in turns if t[applies]]
    return {"turns": len(relevant),
            "share": round(sum(t[kept] for t in relevant) / len(relevant), 4) if relevant
            else None}


def eval_routing(llm, pause: float, limit: int | None) -> dict:
    cases = ROUTING[:limit] if limit else ROUTING
    model_right = rules_right = fallbacks = 0
    misrouted = []
    for message, want in cases:
        params, span = extract_params(message), {}
        got = llm.route(message, params, span)
        fallbacks += "fallback" in span
        model_right += got == want
        rules_right += route_by_rules(message, params) == want
        if got != want:
            misrouted.append({"message": message, "expected": want, "got": got})
        time.sleep(pause)
    return {"cases": len(cases), "model_accuracy": round(model_right / len(cases), 4),
            "rules_accuracy": round(rules_right / len(cases), 4),
            "model_calls_that_fell_back_to_rules": fallbacks, "misrouted_by_model": misrouted}


def eval_answers(llm, pause: float, limit: int | None) -> dict:
    """Run whole conversations and check what the model did to each draft answer."""
    turns = []
    for conversation in CONVERSATIONS:
        session = Session(llm=llm)
        for message in conversation:
            if limit and len(turns) >= limit:
                break
            reply = session.handle(message)
            writes = [s for s in reply.trace if s["agent"] == "writer"]
            if not writes:  # the turn asked for missing details; nothing was written
                continue
            turns.append({
                "message": message, "intent": reply.intent,
                "passed_first_review": reply.review.get("attempts") == 1 and reply.review["ok"],
                "needed_retry": reply.review.get("attempts", 0) > 1 and reply.review["ok"],
                "fell_back_to_template": bool(reply.review.get("fell_back_to_draft")),
                "model_error": any("fallback" in s for s in writes),
                "has_citation": bool(CITATION.search(reply.draft)),
                "citations_kept": set(CITATION.findall(reply.draft))
                <= set(CITATION.findall(reply.text)),
                "has_verdict": bool(VERDICT.search(reply.draft)),
                "verdict_kept": set(VERDICT.findall(reply.draft))
                <= set(VERDICT.findall(reply.text)),
                "ms": reply.usage["ms"], "tokens_in": reply.usage["tokens_in"],
                "tokens_out": reply.usage["tokens_out"], "cost_usd": reply.usage["cost_usd"],
                "answer": reply.text})
            time.sleep(pause)
    n = len(turns) or 1
    return {
        "turns": len(turns),
        "passed_number_review_first_time": round(sum(t["passed_first_review"] for t in turns) / n,
                                                 4),
        "passed_after_one_retry": round(sum(t["needed_retry"] for t in turns) / n, 4),
        "replaced_by_template_answer": round(sum(t["fell_back_to_template"] for t in turns) / n,
                                             4),
        "turns_with_a_model_error": sum(t["model_error"] for t in turns),
        "contract_citations_kept": _share(turns, "has_citation", "citations_kept"),
        "price_verdict_kept": _share(turns, "has_verdict", "verdict_kept"),
        "latency_ms": {"median": statistics.median(t["ms"] for t in turns) if turns else 0,
                       "p95": percentile([t["ms"] for t in turns], 0.95)},
        "tokens_per_turn": {"in": round(sum(t["tokens_in"] for t in turns) / n),
                            "out": round(sum(t["tokens_out"] for t in turns) / n)},
        "cost_per_turn_usd": round(sum(t["cost_usd"] for t in turns) / n, 6),
        "cost_per_1000_turns_usd": round(1000 * sum(t["cost_usd"] for t in turns) / n, 2),
        "examples": [{k: t[k] for k in ("message", "intent", "answer")} for t in turns],
    }


def run(llm=None, pause: float = 6.0, limit: int | None = None) -> dict:
    llm = llm or GeminiLLM()
    report = {"model": llm.name, "date": dt.date.today().isoformat(),
              "note": "Cost uses the list prices set in COPILOT_PRICE_IN_PER_M and "
                      "COPILOT_PRICE_OUT_PER_M. A free-tier key is not charged.",
              "routing": eval_routing(llm, pause, limit),
              "answers": eval_answers(llm, pause, limit)}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "agent_eval_live.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the Gemini path")
    ap.add_argument("--limit", type=int, help="only the first N routing cases and N turns")
    ap.add_argument("--pause", type=float, default=6.0,
                    help="seconds between model calls, to stay inside free-tier rate limits")
    args = ap.parse_args()
    report = run(pause=args.pause, limit=args.limit)
    summary = {k: v for k, v in report["answers"].items() if k != "examples"}
    print(json.dumps({"model": report["model"], "routing": {
        k: v for k, v in report["routing"].items() if k != "misrouted_by_model"},
        "answers": summary}, indent=2))
    print("Full report: reports/agent_eval_live.json")


if __name__ == "__main__":
    main()
