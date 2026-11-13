"""A rule-based vendor to practise against.

The simulated vendor has a hidden reservation price and a persona. It concedes
part of the remaining gap each round, and concedes more when the buyer brings
evidence it cannot dismiss. It never goes below its reservation price.

This is a simulation with fixed rules. It is useful for comparing negotiation
strategies under identical conditions and for rehearsal. It does not predict
how a real vendor will behave.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

PERSONAS = {
    # base: share of the remaining gap conceded per round with no evidence
    # flex: how far below its normal price the vendor will go
    "firm": {"base": 0.10, "flex": 0.00, "evidence": 0.12},
    "flexible": {"base": 0.25, "flex": 0.03, "evidence": 0.15},
    "volume": {"base": 0.12, "flex": 0.01, "evidence": 0.12},
}
EVIDENCE_WORDS = {
    "should_cost": r"should[- ]cost|cost model|cost breakdown|our estimate",
    "index_trend": r"index|market (has|is|price)|commodity|steel price|copper price|resin price",
    "competing_offer": r"compet|another (vendor|supplier|quote)|other (vendor|supplier|quote)"
                       r"|alternative",
    "volume_commitment": r"commit|volume|next (two|2|three|3) orders|long[- ]term|annual",
    "early_payment": r"early pay|pay (earlier|faster|within)|payment in",
}
MAX_ROUNDS = 5


def persona_for(vendor_id: str) -> str:
    """Stable persona per vendor, so repeated runs are comparable."""
    digest = int(hashlib.sha256(vendor_id.encode()).hexdigest(), 16)
    return list(PERSONAS)[digest % len(PERSONAS)]


def parse_buyer_message(text: str) -> tuple[float | None, set[str]]:
    """Pull the offered price and the kinds of evidence out of a typed message."""
    evidence = {name for name, pattern in EVIDENCE_WORDS.items()
                if re.search(pattern, text, re.IGNORECASE)}
    offer = None
    m = re.search(r"(?:offer|pay|at|close at|accept|do|propose)\s*(?:you\s*)?(?:usd|us\$|\$)?\s*"
                  r"(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        offer = float(m.group(1))
    return offer, evidence


@dataclass
class VendorSimulator:
    quote_usd: float
    fair_price_usd: float
    persona: str = "flexible"
    vendor_name: str = "Vendor"
    competing_price_usd: float | None = None
    market_change_pct: float = 0.0
    price: float = field(init=False)
    round: int = field(default=0, init=False)
    closed: bool = field(default=False, init=False)
    history: list[dict] = field(default_factory=list, init=False)
    seen: set[str] = field(default_factory=set, init=False)  # arguments heard so far

    def __post_init__(self) -> None:
        self.price = self.quote_usd

    def _reservation(self, evidence: set[str]) -> float:
        p = PERSONAS[self.persona]
        flex = p["flex"] + (0.03 if self.persona == "volume" and "volume_commitment" in evidence
                            else 0.0)
        return min(self.fair_price_usd * (1 - flex), self.quote_usd * 0.995)

    def _credible(self, evidence: set[str], should_cost: float | None) -> set[str]:
        """The vendor only gives ground on evidence that is actually true."""
        ok = set()
        if "should_cost" in evidence and should_cost and should_cost < self.price:
            ok.add("should_cost")
        if "index_trend" in evidence and self.market_change_pct < 0:
            ok.add("index_trend")
        if ("competing_offer" in evidence and self.competing_price_usd
                and self.competing_price_usd < self.price):
            ok.add("competing_offer")
        ok |= evidence & {"volume_commitment", "early_payment"}
        return ok

    def respond(self, offer_usd: float | None, evidence: set[str] | None = None,
                should_cost_usd: float | None = None) -> dict:
        if self.closed:
            return {"status": "closed", "price_usd": round(self.price, 4),
                    "message": "This negotiation has already ended."}
        self.seen |= evidence or set()
        evidence = set(self.seen)
        self.round += 1
        credible = self._credible(evidence, should_cost_usd)
        p = PERSONAS[self.persona]
        rate = min(p["base"] + p["evidence"] * len(credible), 0.9)
        floor = self._reservation(evidence)
        counter = max(floor, self.price - rate * (self.price - floor))

        if offer_usd is not None and offer_usd >= counter:
            self.price, self.closed = min(offer_usd, self.price), True
            status, msg = "accepted", f"Agreed at {self.price:.4f} USD. We will confirm in writing."
        else:
            moved = self.price - counter
            self.price = counter
            final = self.round >= MAX_ROUNDS
            self.closed = final
            status = "final_offer" if final else "counter"
            if moved < 1e-6:
                why = "We are already at our limit on this item."
            elif credible:
                why = "We have taken your points on " + ", ".join(
                    sorted(c.replace("_", " ") for c in credible)) + " into account."
            else:
                why = "We can move a little as a gesture of goodwill."
            msg = (f"{why} {'Our final price is' if final else 'We can offer'} "
                   f"{self.price:.4f} USD.")
        turn = {"round": self.round, "buyer_offer_usd": offer_usd, "evidence": sorted(evidence),
                "credible_evidence": sorted(credible), "status": status,
                "price_usd": round(self.price, 4), "message": f"{self.vendor_name}: {msg}"}
        self.history.append(turn)
        return turn


def run_auto(sim: VendorSimulator, strategy: str, playbook: dict | None = None) -> dict:
    """Play a whole negotiation with a scripted buyer.

    'ask_5pct' asks for 5% off with no supporting evidence, which is the common habit.
    'playbook' opens at the playbook's opening ask, brings its evidence, and moves
    towards the walk-away price round by round.
    """
    quote = sim.quote_usd
    if strategy == "ask_5pct":
        offers = [quote * 0.95] * MAX_ROUNDS
        evidence, should_cost, limit = set(), None, quote
    else:
        opening, limit = playbook["opening_ask_usd"], playbook["walk_away_usd"]
        target = playbook["target_price_usd"]
        steps = [0, 0.4, 0.7, 0.9, 1.0]
        offers = [opening + s * (target - opening) for s in steps]
        evidence = {lv["lever"] for lv in playbook["levers"]} | {"volume_commitment"}
        should_cost = playbook["quote"]["should_cost_usd"]["p50"]
    last = None
    for offer in offers:
        last = sim.respond(offer, evidence, should_cost)
        if sim.closed:
            break
    final = last["price_usd"]
    batna = playbook["batna"]["price_usd"] if playbook and playbook.get("batna") else None
    firm = playbook and playbook.get("batna") and playbook["batna"]["firm"]
    if last["status"] != "accepted" and strategy == "playbook" and final > limit and firm:
        outcome, paid = "walked_away_to_alternative", batna
    else:
        outcome, paid = "deal", final
    return {"strategy": strategy, "outcome": outcome, "price_paid_usd": round(paid, 4),
            "rounds": sim.round, "saving_pct": round(100 * (1 - paid / quote), 2)}
