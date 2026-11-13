"""Supervisor logic that needs no language model: pull identifiers and numbers
out of a message, and choose an intent by rule.

The rules are the offline fallback and the baseline a language-model router
has to beat. Identifiers (material, vendor, RFQ) are always taken from here,
because a regular expression does not invent an ID.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from copilot import config
from copilot.pipeline.clean import normalize_name

INTENTS = {
    "top_vendors": "Which vendors we spend the most with",
    "expensive_vendors": "Which vendors charge the most above should-cost",
    "late_vendors": "Which vendors deliver late most often",
    "spend_by_category": "Spend broken down by material category",
    "vendor_profile": "Score and track record of one vendor",
    "contract_question": "A question answered by a vendor's contract",
    "rebate_status": "Whether a volume rebate has been earned or is close",
    "check_quote": "Is a quoted price fair",
    "rfq_compare": "Compare all quotes received for one sourcing event",
    "negotiate": "Build a negotiation playbook for a quote",
    "simulate_start": "Practise the negotiation against a simulated vendor",
    "simulate_auto": "Run the simulated negotiation automatically and compare strategies",
    "help": "Anything else",
}
MODES = {"check_quote": "Check a quote", "rfq_compare": "Check a quote",
         "negotiate": "Negotiate", "simulate_start": "Negotiate", "simulate_auto": "Negotiate",
         "simulate_turn": "Negotiate", "simulate_end": "Negotiate", "help": "Help"}

RULES = [
    ("simulate_auto", r"\bauto|compare strateg|both strateg"),
    ("simulate_start", r"practi[cs]e|simulat|role[- ]?play|rehears|pretend to be"),
    ("negotiate", r"negotiat|playbook|counter[- ]?offer|push back|leverage|talking points"
                  r"|what should (i|we) (ask|offer|say)|how (do|should) (i|we) (respond|reply)"),
    ("rfq_compare", r"rfq-\d+"),
    ("expensive_vendors", r"(which|who|what) (vendors?|suppliers?).*(expensive|charg|overpric"
                          r"|above should)"),
    ("rebate_status", r"rebate"),
    ("contract_question", r"contract|clause|agreement|payment term|when (do|must|should) we pay"
                          r"|how (long|many days).*(pay|notice)|terminat|penalt|damages|warrant"
                          r"|price (review|adjust)|early[- ]pay|net \d+|what (happens|if)"
                          r"|allowed to|are we entitled|can we (cancel|exit|end)"),
    ("check_quote", r"quot|overpriced|fair price|should[- ]cost|too (high|expensive|much)"
                    r"|good price|reasonable|worth"),
    ("late_vendors", r"\blate\b|on[- ]time|delay|deliver"),
    ("expensive_vendors", r"expensive|overcharg|overpric|above should|priciest|most costly"),
    ("spend_by_category", r"categor"),
    ("top_vendors", r"spend|spent|biggest|largest|\btop\b|most money|how much (do|did) we buy"),
    ("vendor_profile", r"score|profile|perform|reliab|track record|tell me about|how is|how good"),
]


@lru_cache(maxsize=4)
def _vendor_names(path: str) -> list[tuple[str, str]]:
    v = pd.read_parquet(path, columns=["vendor_id", "vendor_name"])
    pairs = [(normalize_name(n), i) for n, i in zip(v["vendor_name"], v["vendor_id"], strict=True)]
    return sorted(pairs, key=lambda p: -len(p[0]))


@lru_cache(maxsize=4)
def _categories(path: str) -> list[str]:
    return sorted(pd.read_parquet(path, columns=["category"])["category"].unique())


def extract_params(message: str) -> dict:
    """Identifiers, quantity, price, currency, date and year mentioned in a message."""
    out: dict = {}
    text = message
    if m := re.search(r"\bMAT-\d{4}\b", text, re.I):
        out["material_id"] = m.group(0).upper()
    if m := re.search(r"\bRFQ-\d{5}\b", text, re.I):
        out["rfq_id"] = m.group(0).upper()
    if m := re.search(r"\bV-\d{4}\b", text, re.I):
        out["vendor_id"] = m.group(0).upper()
    else:
        norm = f" {normalize_name(text)} "
        for name, vendor_id in _vendor_names(str(config.CURATED / "vendors.parquet")):
            if f" {name} " in norm:
                out["vendor_id"] = vendor_id
                break
    if m := re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text):
        out["date"] = m.group(0)
    text = re.sub(r"\b(MAT|RFQ|V|CT)-\d+\b|\b20\d{2}-\d{2}-\d{2}\b", " ", text, flags=re.I)
    if m := re.search(r"\b(20[12]\d)\b", text):
        out["year"] = int(m.group(1))
        text = text.replace(m.group(0), " ", 1)
    if m := re.search(r"\btop\s+(\d{1,2})\b", text, re.I):
        out["n"] = int(m.group(1))
        text = text.replace(m.group(0), " ", 1)
    lowered = message.lower()
    for cat in _categories(str(config.CURATED / "materials.parquet")):
        if cat.replace("_", " ") in lowered or cat in lowered:
            out["category"] = cat
            break

    num = r"(\d[\d,]*(?:\.\d+)?)"
    if m := (re.search(num + r"\s*(kg|kgs|pcs|pieces|units|tons?|mt|ea)\b", text, re.I)
             or re.search(r"\b(?:qty|quantity|volume)\s*(?:of|is|:)?\s*" + num + r"()", text,
                          re.I)):
        qty = float(m.group(1).replace(",", ""))
        out["qty"] = qty * (1000 if m.group(2).lower() in ("ton", "tons", "mt") else 1)
        text = text.replace(m.group(0), " ", 1)
    cur = r"(usd|us\$|\$|thb|baht|฿|eur|euro|€)"
    if m := (re.search(cur + r"\s*" + num, text, re.I)
             or re.search(num + r"\s*" + cur, text, re.I)
             or re.search(r"(?:\bat|price(?: of| is)?|quot\w*|@|\bfor)\s*" + num, text, re.I)):
        groups = [g for g in m.groups() if g]
        number = next(g for g in groups if g[0].isdigit())
        out["price"] = float(number.replace(",", ""))
        symbol = next((g.lower() for g in groups if not g[0].isdigit()), "")
        out["currency"] = ("THB" if symbol in ("thb", "baht", "฿") else
                           "EUR" if symbol in ("eur", "euro", "€") else "USD")
    return out


def route_by_rules(message: str, params: dict) -> str:
    text = message.lower()
    for intent, pattern in RULES:
        if re.search(pattern, text):
            if intent == "rfq_compare" and re.search(RULES[2][1], text):
                return "negotiate"
            if intent == "late_vendors" and params.get("vendor_id"):
                return "vendor_profile"
            if intent in ("top_vendors", "expensive_vendors") and params.get("vendor_id") \
                    and not re.search(r"vendors|suppliers|who|which", text):
                return "vendor_profile"
            return intent
    if {"material_id", "price"} <= params.keys():
        return "check_quote"
    if params.get("vendor_id"):
        return "vendor_profile"
    return "help"
