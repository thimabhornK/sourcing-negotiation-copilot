"""The language-model boundary.

The orchestrator talks to a model through two calls only: classify a message
into an intent, and reword a draft answer. Both have a safe fallback, so the
system still answers when the model is unavailable or wrong.

OfflineLLM uses rules and returns the draft unchanged. It needs no API key and
is what the tests and the evaluation run on.

GeminiLLM calls Gemini through the google-genai SDK. Set GEMINI_API_KEY (or
GOOGLE_API_KEY) and optionally COPILOT_GEMINI_MODEL.
"""

from __future__ import annotations

import json
import os

from copilot.agents.router import INTENTS, route_by_rules

ROUTER_PROMPT = """You route messages for a procurement assistant.
Choose exactly one intent for the user's message.

Intents:
{intents}

Reply with JSON only: {{"intent": "<one of the intent names>"}}"""

WRITER_PROMPT = """You are a procurement analyst writing to a buyer.
Rewrite the DRAFT so it reads naturally and directly answers the QUESTION.

Rules:
- Use only numbers that appear in the DRAFT or FACTS. Never calculate a new number.
- Keep every contract citation (for example "CT-0002 clause 4") exactly as written.
- Keep the verdict and the recommended prices unchanged.
- Plain text, no Markdown tables.
{feedback}"""


class OfflineLLM:
    name = "offline-rules"

    def route(self, message: str, params: dict, span: dict) -> str:
        return route_by_rules(message, params)

    def write(self, question: str, draft: str, facts: list, span: dict,
              feedback: str = "") -> str:
        return draft


class GeminiLLM:
    """Gemini for routing and wording. Tools still produce every number."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from google import genai  # imported here so the offline path has no dependency

        self.name = model or os.environ.get("COPILOT_GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
        # USD per million tokens, used only for the cost shown in traces. Set these
        # to the current list price of the model you use.
        self.price_in = float(os.environ.get("COPILOT_PRICE_IN_PER_M", "0.30"))
        self.price_out = float(os.environ.get("COPILOT_PRICE_OUT_PER_M", "2.50"))

    def _generate(self, system: str, user: str, span: dict, as_json: bool = False) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system, temperature=0.2,
            response_mime_type="application/json" if as_json else "text/plain")
        response = self.client.models.generate_content(model=self.name, contents=user,
                                                       config=config)
        usage = response.usage_metadata
        tokens_in = getattr(usage, "prompt_token_count", 0) or 0
        tokens_out = getattr(usage, "candidates_token_count", 0) or 0
        span.update(model=self.name, tokens_in=tokens_in, tokens_out=tokens_out,
                    cost_usd=round((tokens_in * self.price_in + tokens_out * self.price_out)
                                   / 1e6, 6))
        return response.text or ""

    def route(self, message: str, params: dict, span: dict) -> str:
        fallback = route_by_rules(message, params)
        intents = "\n".join(f"- {name}: {desc}" for name, desc in INTENTS.items())
        try:
            raw = self._generate(ROUTER_PROMPT.format(intents=intents), message, span,
                                 as_json=True)
            intent = json.loads(raw).get("intent")
        except Exception as exc:  # network, quota or malformed JSON: fall back to rules
            span["fallback"] = f"{type(exc).__name__}"
            return fallback
        if intent not in INTENTS:
            span["fallback"] = "unknown_intent"
            return fallback
        return intent

    def write(self, question: str, draft: str, facts: list, span: dict,
              feedback: str = "") -> str:
        note = f"\nA reviewer rejected the previous attempt: {feedback}" if feedback else ""
        user = (f"QUESTION:\n{question}\n\nDRAFT:\n{draft}\n\nFACTS:\n"
                f"{json.dumps(facts, default=str)[:12000]}")
        try:
            return self._generate(WRITER_PROMPT.format(feedback=note), user, span) or draft
        except Exception as exc:
            span["fallback"] = f"{type(exc).__name__}"
            return draft


def default_llm():
    """Gemini when a key is configured, otherwise the offline rules."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        try:
            return GeminiLLM()
        except Exception:
            return OfflineLLM()
    return OfflineLLM()
