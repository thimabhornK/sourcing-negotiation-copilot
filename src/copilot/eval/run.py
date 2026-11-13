"""Evaluate the agent layer and write reports/agent_eval.json.

Run with:  python -m copilot.eval.run

Everything here runs offline with the rule-based router and writer. It
measures the parts that do not depend on a language model: routing rules,
contract retrieval and extraction, the reviewer, and negotiation strategies
against the simulated vendor.
"""

from __future__ import annotations

import json
import re
import statistics

import numpy as np
import pandas as pd

from copilot import config, tools
from copilot.agents import writer
from copilot.agents.llm import OfflineLLM
from copilot.agents.orchestrator import Session
from copilot.agents.reviewer import NUMBER, review
from copilot.agents.router import extract_params, route_by_rules
from copilot.contracts.extract import extract_terms
from copilot.contracts.search import ContractIndex
from copilot.eval.cases import QUESTIONS, ROUTING
from copilot.negotiation import simulator as sim
from copilot.negotiation.playbook import build_playbook, draft_email


def eval_routing() -> dict:
    wrong = [(m, want, got) for m, want in ROUTING
             if (got := route_by_rules(m, extract_params(m))) != want]
    return {"cases": len(ROUTING), "accuracy": round(1 - len(wrong) / len(ROUTING), 4),
            "misrouted": [{"message": m, "expected": w, "got": g} for m, w, g in wrong]}


def eval_retrieval() -> dict:
    chunks = pd.read_parquet(config.CURATED / "contract_chunks.parquet")
    vendors = sorted(chunks["vendor_id"].unique())
    out = {"questions": len(vendors) * len(QUESTIONS)}
    for label, normalise in (("plain_tfidf", False), ("with_vocabulary", True)):
        index = ContractIndex(chunks, normalise_vocabulary=normalise)
        top1 = top3 = 0
        for i, vendor_id in enumerate(vendors):
            for title, variants in QUESTIONS.items():
                found = [r["title"] for r in index.search(variants[i % len(variants)],
                                                          vendor_id, 3)]
                top1 += found[:1] == [title]
                top3 += title in found
        out[label] = {"hit_at_1": round(top1 / out["questions"], 4),
                      "hit_at_3": round(top3 / out["questions"], 4)}
    return out


def eval_extraction() -> dict:
    truth = pd.read_csv(config.RAW / "eval_contract_terms.csv")
    checked = correct = 0
    for t in truth.to_dict("records"):
        got = extract_terms(t["vendor_id"])
        tiers = [{"threshold_usd": t[f"rebate_tier{i}_usd"], "rebate_pct": t[f"rebate_tier{i}_pct"]}
                 for i in (1, 2) if t[f"rebate_tier{i}_usd"]]
        pairs = [(got.get("payment_days"), t["payment_days"]),
                 (got.get("early_pay_discount_pct", 0), t["early_pay_discount_pct"]),
                 (got.get("rebate_tiers"), tiers),
                 (got.get("index_linked"), bool(t["index_linked"])),
                 (got.get("late_penalty_pct_per_week"), t["late_penalty_pct_per_week"]),
                 (got.get("termination_notice_days"), t["termination_notice_days"])]
        checked += len(pairs)
        correct += sum(a == b for a, b in pairs)
    return {"contracts": len(truth), "fields_checked": checked,
            "field_accuracy": round(correct / checked, 4)}


def _corrupt(text: str, rng: np.random.Generator) -> str | None:
    """Change one figure in an answer, keeping its format, as a careless model might."""
    spots = [m for m in NUMBER.finditer(text)
             if "." in m.group(0) or abs(float(m.group(0).replace(",", ""))) > 31]
    spots = [m for m in spots if not re.search(r"[A-Z]-$", text[max(m.start() - 5, 0):m.start()])
             and not re.match(r"-\d{2}", text[m.end():m.end() + 3])]
    if not spots:
        return None
    m = spots[int(rng.integers(len(spots)))]
    token = m.group(0)
    decimals = len(token.split(".")[1]) if "." in token else 0
    value = float(token.replace(",", "")) * rng.uniform(1.04, 1.30)
    new = f"{value:+,.{decimals}f}" if token[0] in "+-" else f"{value:,.{decimals}f}"
    return text[:m.start()] + new + text[m.end():]


def eval_reviewer(quotes: pd.DataFrame, n: int = 100) -> dict:
    rng = np.random.default_rng(config.SEED)
    sample = quotes.sample(n, random_state=config.SEED)
    clean_total = false_alarms = corrupted_total = caught = 0
    for i, q in enumerate(sample.itertuples()):
        date = f"{q.quote_date:%Y-%m-%d}"
        if i % 2:
            facts = build_playbook(q.material_id, q.qty, q.quoted_price_usd, q.vendor_id,
                                   date=date, rfq_id=q.rfq_id)
            text = writer.playbook(facts, draft_email(facts))
        else:
            facts = tools.quote_check(q.material_id, q.qty, q.quoted_price_usd, "USD",
                                      q.vendor_id, date)
            text = writer.render_quote(facts)
        clean_total += 1
        false_alarms += not review(text, facts)["ok"]
        bad = _corrupt(text, rng)
        if bad:
            corrupted_total += 1
            caught += not review(bad, facts)["ok"]
    return {"answers": clean_total,
            "false_alarm_rate": round(false_alarms / clean_total, 4),
            "corrupted_answers": corrupted_total,
            "catch_rate": round(caught / corrupted_total, 4)}


def eval_negotiation(quotes: pd.DataFrame, per_group: int = 200) -> dict:
    """Negotiate a sample of test-period quotes one-to-one with the quoting vendor.

    Competing quotes are left out on purpose, so the comparison isolates how well each
    strategy negotiates and not which vendor it picks. The simulated vendor's limit is
    its normal no-markup price, which the buyer never sees.
    """
    labels = pd.read_csv(config.RAW / "eval_quote_labels.csv")
    events = quotes.merge(labels, on=["rfq_id", "vendor_id"])
    events = pd.concat([part.sample(min(per_group, len(part)), random_state=config.SEED)
                        for _, part in events.groupby("is_overpriced")])
    rows = []
    for e in events.itertuples():
        pb = build_playbook(e.material_id, e.qty, e.quoted_price_usd, e.vendor_id,
                            date=f"{e.quote_date:%Y-%m-%d}")
        row = {"overpriced": bool(e.is_overpriced),
               "room": 100 * max(1 - e.fair_price_usd / e.quoted_price_usd, 0)}
        for strategy in ("ask_5pct", "playbook"):
            vendor = sim.VendorSimulator(
                quote_usd=e.quoted_price_usd, fair_price_usd=e.fair_price_usd,
                persona=sim.persona_for(e.vendor_id),
                competing_price_usd=pb["batna"]["price_usd"] if pb["batna"] else None,
                market_change_pct=pb["quote"]["market"]["change_12m_pct"])
            result = sim.run_auto(vendor, strategy, pb)
            row[strategy] = result["saving_pct"]
            row[f"{strategy}_rounds"] = result["rounds"]
        rows.append(row)
    df = pd.DataFrame(rows)

    def block(part: pd.DataFrame) -> dict:
        return {"quotes": len(part),
                "avg_saving_pct": {k: round(float(part[k].mean()), 2)
                                   for k in ("ask_5pct", "playbook")},
                "avg_gap_to_vendor_normal_price_pct": round(float(part["room"].mean()), 2),
                "avg_rounds": {k: round(float(part[f"{k}_rounds"].mean()), 2)
                               for k in ("ask_5pct", "playbook")},
                "playbook_better_share": round(
                    float((part["playbook"] > part["ask_5pct"] + 1e-9).mean()), 4),
                "playbook_worse_share": round(
                    float((part["playbook"] < part["ask_5pct"] - 1e-9).mean()), 4)}

    return {"note": "Simulated vendor with fixed rules that concedes more when given true "
                    "evidence. This compares strategies under identical conditions. It is "
                    "not a forecast of savings with real vendors.",
            "overpriced_quotes": block(df[df["overpriced"]]),
            "fair_quotes": block(df[~df["overpriced"]])}


def eval_latency() -> dict:
    script = ["Top 5 vendors by spend in 2025", "How is V-0009 performing?",
              "What are the payment terms with V-0003?",
              "Are we close to a rebate with V-0002 in 2025?",
              "Is 1.62 USD fair for 5000 kg of MAT-0001 from V-0002 on 2025-06-15?",
              "Build a negotiation playbook for that", "Run the negotiation automatically"]
    session = Session(llm=OfflineLLM())
    for message in script:  # warm the caches so the timing reflects steady state
        session.handle(message)
    session = Session(llm=OfflineLLM())
    timings = {}
    for message in script:
        reply = session.handle(message)
        timings[reply.intent] = reply.usage["ms"]
    return {"note": "Offline path: tools and rules only, no language-model call.",
            "ms_by_intent": timings, "median_ms": statistics.median(timings.values())}


def run() -> dict:
    quotes = pd.read_parquet(config.CURATED / "rfq_quotes.parquet")
    test = quotes[quotes["quote_date"] >= config.TEST_START]
    report = {"routing": eval_routing(), "retrieval": eval_retrieval(),
              "extraction": eval_extraction(), "reviewer": eval_reviewer(test),
              "negotiation_simulation": eval_negotiation(test), "latency": eval_latency()}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "agent_eval.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
