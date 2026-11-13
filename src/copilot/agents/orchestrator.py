"""Orchestrate one chat turn across the agents.

  Supervisor   decides what the user wants and which specialists to call
  Specialists  market, vendor, contract and spend analysts; each one calls tools
  Strategist   turns the specialists' findings into a negotiation playbook
  Writer       drafts the answer; a language model may reword it
  Reviewer     checks every number against the tool results, up to two attempts

Each step is traced. The specialists and the strategist are deterministic, so
the same question always gets the same figures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from copilot import tools
from copilot.agents import writer
from copilot.agents.llm import default_llm
from copilot.agents.reviewer import review
from copilot.agents.router import MODES, extract_params
from copilot.agents.tracing import Tracer
from copilot.negotiation import simulator as sim
from copilot.negotiation.playbook import build_playbook, draft_email

MAX_REVIEW_ATTEMPTS = 2
QUOTE_KEYS = ("material_id", "qty", "price", "currency", "vendor_id", "date", "rfq_id")
QUOTE_EXAMPLE = "Is 1.62 USD fair for 5000 kg of MAT-0001 from V-0002?"


@dataclass
class Reply:
    text: str
    intent: str
    mode: str
    facts: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    review: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


class Session:
    """One conversation. Remembers the last quote so follow-ups can refer to it."""

    def __init__(self, llm=None, tracer: Tracer | None = None) -> None:
        self.llm = llm or default_llm()
        self.tracer = tracer or Tracer()
        self.quote: dict = {}
        self.vendor_id: str | None = None
        self.playbook: dict | None = None
        self.sim: sim.VendorSimulator | None = None

    # ---- one turn ------------------------------------------------------------

    def handle(self, message: str) -> Reply:
        t = self.tracer
        t.start_turn()
        with t.span("supervisor", "route") as span:
            params = extract_params(message)
            if self.sim and not self.sim.closed:
                intent = ("simulate_end" if re.search(r"\b(stop|quit|exit|end)\b", message, re.I)
                          else "simulate_turn")
            else:
                intent = self.llm.route(message, params, span)
            span.update(intent=intent, params=params)
        self._remember(params)

        facts: list = []
        try:
            draft = getattr(self, f"_{intent}")(message, params, facts)
        except ValueError as exc:  # unknown material, date outside the data, and so on
            draft, facts = f"I could not do that: {exc}.", []

        text, verdict = self._write_and_review(message, draft, facts, intent)
        return Reply(text=text, intent=intent, mode=MODES.get(intent, "Ask"), facts=facts,
                     trace=list(t.spans), review=verdict, usage=t.summary())

    def _remember(self, params: dict) -> None:
        if params.get("vendor_id"):
            self.vendor_id = params["vendor_id"]
        for key in ("material_id", "rfq_id"):  # a new material or event starts a new quote
            if params.get(key) and params[key] != self.quote.get(key):
                self.quote = {}
        self.quote.update({k: params[k] for k in QUOTE_KEYS if k in params})

    def _write_and_review(self, question: str, draft: str, facts: list, intent: str
                          ) -> tuple[str, dict]:
        t = self.tracer
        if intent in ("help", "simulate_turn", "simulate_end") or not facts:
            return draft, {"ok": True, "unsupported_numbers": [], "attempts": 0}
        feedback, verdict = "", {}
        for attempt in range(1, MAX_REVIEW_ATTEMPTS + 1):
            with t.span("writer", "compose", attempt=attempt) as span:
                text = self.llm.write(question, draft, facts, span, feedback)
            with t.span("reviewer", "check_numbers", attempt=attempt) as span:
                verdict = review(text, [facts, draft])
                span.update(ok=verdict["ok"], unsupported=verdict["unsupported_numbers"])
            verdict["attempts"] = attempt
            if verdict["ok"]:
                return text, verdict
            feedback = ("these numbers are not in the facts: "
                        + ", ".join(verdict["unsupported_numbers"]))
        verdict["fell_back_to_draft"] = True
        return draft, verdict

    def _call(self, agent: str, fn, facts: list, **kwargs) -> dict:
        with self.tracer.span(agent, fn.__name__, args=kwargs):
            result = fn(**kwargs)
        facts.append(result)
        return result

    # ---- Ask -------------------------------------------------------------------

    def _top_vendors(self, message, p, facts):
        return writer.table(self._call("spend_analyst", tools.top_vendors_by_spend, facts,
                                       year=p.get("year"), category=p.get("category"),
                                       n=p.get("n", 5)))

    def _expensive_vendors(self, message, p, facts):
        return writer.table(self._call("spend_analyst", tools.most_expensive_vendors, facts,
                                       year=p.get("year"), category=p.get("category"),
                                       material_id=p.get("material_id"), n=p.get("n", 5)))

    def _late_vendors(self, message, p, facts):
        return writer.table(self._call("spend_analyst", tools.late_vendors, facts,
                                       year=p.get("year"), n=p.get("n", 5)))

    def _spend_by_category(self, message, p, facts):
        return writer.table(self._call("spend_analyst", tools.spend_by_category, facts,
                                       year=p.get("year")))

    def _vendor_profile(self, message, p, facts):
        if not self.vendor_id:
            return writer.need("a vendor", "How is V-0009 performing?")
        return writer.vendor(self._call("vendor_analyst", tools.vendor_profile, facts,
                                        vendor_id=self.vendor_id))

    def _contract_question(self, message, p, facts):
        found = self._call("contract_analyst", tools.search_contracts, facts,
                           question=message, vendor_id=self.vendor_id, k=3)
        return writer.contract(found, None)

    def _rebate_status(self, message, p, facts):
        if not self.vendor_id:
            return writer.need("a vendor", "Are we close to a rebate with V-0002 in 2025?")
        year = p.get("year") or int(tools.latest_date()[:4])
        return writer.rebate(self._call("contract_analyst", tools.rebate_status, facts,
                                        vendor_id=self.vendor_id, year=year))

    # ---- Check a quote ---------------------------------------------------------

    def _quote_args(self, need_vendor: bool = False) -> dict | None:
        q = self.quote
        required = {"material_id", "qty", "price"} | ({"vendor_id"} if need_vendor else set())
        merged = {**q, "vendor_id": q.get("vendor_id") or self.vendor_id}
        if not all(merged.get(k) for k in required):
            return None
        return {"material_id": merged["material_id"], "qty": merged["qty"],
                "price": merged["price"], "currency": merged.get("currency", "USD"),
                "vendor_id": merged["vendor_id"], "date": merged.get("date")}

    def _check_quote(self, message, p, facts):
        args = self._quote_args()
        if not args:
            return writer.need("the material, the quantity and the quoted price", QUOTE_EXAMPLE)
        result = self._call("market_analyst", tools.quote_check, facts, **args)
        text = writer.render_quote(result)
        if result["verdict"] == "OVERPRICED":
            text += "\nAsk for a negotiation playbook to see how to bring this down."
        return text

    def _rfq_compare(self, message, p, facts):
        if not p.get("rfq_id"):
            return writer.need("a sourcing event", "Compare the quotes in RFQ-01990")
        return writer.rfq(self._call("market_analyst", tools.rfq_summary, facts,
                                     rfq_id=p["rfq_id"]))

    # ---- Negotiate -------------------------------------------------------------

    def _build_playbook(self, facts: list) -> dict | None:
        if self.quote.get("rfq_id") and not self.quote.get("price"):
            event = tools.rfq_summary(self.quote["rfq_id"])
            if event.get("found"):
                picks = [q for q in event["quotes"]
                         if q["vendor"]["vendor_id"] == self.vendor_id] or event["quotes"][:1]
                best = picks[0]
                self.quote.update(material_id=best["material_id"], qty=best["qty"],
                                  price=best["quoted_price_usd"], currency="USD",
                                  vendor_id=best["vendor"]["vendor_id"], date=best["quote_date"])
        args = self._quote_args(need_vendor=True)
        if not args:
            return None
        with self.tracer.span("strategist", "build_playbook", args=args):
            self.playbook = build_playbook(**args, rfq_id=self.quote.get("rfq_id"))
        facts.append(self.playbook)
        return self.playbook

    def _negotiate(self, message, p, facts):
        pb = self._build_playbook(facts)
        if not pb:
            return writer.need("the material, quantity, quoted price and vendor",
                               "How should I negotiate 1.62 USD for 5000 kg of MAT-0001 "
                               "from V-0002?")
        return writer.playbook(pb, draft_email(pb))

    def _new_simulator(self, pb: dict) -> sim.VendorSimulator:
        """The vendor's limit is hidden from the buyer. Outside the evaluation it is set
        from how this vendor has priced against should-cost in the past."""
        q = pb["quote"]
        usual = min(max(q.get("vendor", {}).get("price_ratio", 1.0), 0.95), 1.08)
        return sim.VendorSimulator(
            quote_usd=q["quoted_price_usd"], fair_price_usd=q["should_cost_usd"]["p50"] * usual,
            persona=sim.persona_for(pb["vendor_id"]), vendor_name=pb["vendor_name"],
            competing_price_usd=pb["batna"]["price_usd"] if pb["batna"] else None,
            market_change_pct=q["market"]["change_12m_pct"])

    def _simulate_start(self, message, p, facts):
        pb = self._build_playbook(facts)
        if not pb:
            return writer.need("a quote to practise on", QUOTE_EXAMPLE + " Then ask to practise.")
        self.sim = self._new_simulator(pb)
        q = pb["quote"]
        return (f"Practice negotiation. I will play {pb['vendor_name']}, following fixed rules "
                f"with a hidden price limit, so treat this as rehearsal.\n"
                f"{pb['vendor_name']}: Our price is {q['quoted_price_usd']:.4f} USD per "
                f"{q['unit']} for {q['qty']:,.0f} {q['unit']}.\n"
                f"Coach: open near {pb['opening_ask_usd']:.4f}, aim for "
                f"{pb['target_price_usd']:.4f}, walk away above {pb['walk_away_usd']:.4f}.\n"
                f"Type your offer and your arguments, for example: \"We can offer "
                f"{pb['opening_ask_usd']:.2f}. Our cost model and the index both point lower.\" "
                f"Type stop to end.")

    def _simulate_turn(self, message, p, facts):
        pb = self.playbook
        offer, evidence = sim.parse_buyer_message(message)
        with self.tracer.span("vendor_simulator", "respond", offer=offer,
                              evidence=sorted(evidence)):
            turn = self.sim.respond(offer, evidence, pb["quote"]["should_cost_usd"]["p50"])
        facts.append(turn)
        lines = [turn["message"]]
        if self.sim.closed:
            lines.append(self._debrief())
        else:
            unused = [lv["lever"].replace("_", " ") for lv in pb["levers"]
                      if lv["lever"] not in self.sim.seen]
            tip = f" You have not used: {', '.join(unused)}." if unused else ""
            lines.append(f"Coach: round {turn['round']} of {sim.MAX_ROUNDS}. Target "
                         f"{pb['target_price_usd']:.4f}, walk away above "
                         f"{pb['walk_away_usd']:.4f}.{tip}")
        return "\n".join(lines)

    def _debrief(self) -> str:
        pb, s = self.playbook, self.sim
        quote = s.quote_usd
        verdict = ("at or below target" if s.price <= pb["target_price_usd"] + 1e-9 else
                   "between target and walk-away" if s.price <= pb["walk_away_usd"] + 1e-9 else
                   "above your walk-away price, so the alternative is the better choice")
        return (f"Debrief: {quote:.4f} down to {s.price:.4f} USD in {s.round} rounds, "
                f"{100 * (1 - s.price / quote):.1f}% off the quote and {verdict}.")

    def _simulate_end(self, message, p, facts):
        self.sim.closed = True
        return "Practice ended. " + self._debrief()

    def _simulate_auto(self, message, p, facts):
        pb = self._build_playbook(facts)
        if not pb:
            return writer.need("a quote to negotiate", QUOTE_EXAMPLE)
        runs = []
        for strategy in ("ask_5pct", "playbook"):
            with self.tracer.span("vendor_simulator", "run_auto", strategy=strategy):
                runs.append(sim.run_auto(self._new_simulator(pb), strategy, pb))
        facts.append({"tool": "simulate_auto", "runs": runs})
        names = {"ask_5pct": "Ask for 5% off, no evidence", "playbook": "Follow the playbook"}
        lines = [f"Simulated negotiation with {pb['vendor_name']} (rule-based vendor, quote "
                 f"{pb['quote']['quoted_price_usd']:.4f} USD):"]
        lines += [f"  {names[r['strategy']]}: {r['price_paid_usd']:.4f} USD, "
                  f"{r['saving_pct']:.2f}% saved in {r['rounds']} rounds "
                  f"({r['outcome'].replace('_', ' ')})" for r in runs]
        lines.append("These figures come from a simulation with fixed rules. They compare "
                     "strategies under the same conditions and are not a forecast.")
        return "\n".join(lines)

    def _help(self, message, p, facts):
        return writer.HELP
