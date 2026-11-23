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
import re
import time

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


KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def describe_error(exc: Exception) -> str:
    """A short, key-free description of a failed model call, kept in the trace."""
    text = KEY_PATTERN.sub("[api key removed]", " ".join(str(exc).split()))
    return f"{type(exc).__name__}: {text[:300]}"


class OfflineLLM:
    name = "offline-rules"

    def route(self, message: str, params: dict, span: dict) -> str:
        return route_by_rules(message, params)

    def write(self, question: str, draft: str, facts: list, span: dict,
              feedback: str = "") -> str:
        return draft


DEFAULT_MODEL = "gemini-3.5-flash"
# Paid-tier list prices in USD per million tokens (input, output), from the Gemini
# API pricing page on 2026-11-13. Used only for the cost shown in traces; a
# free-tier key is not charged. Override with COPILOT_PRICE_IN_PER_M and
# COPILOT_PRICE_OUT_PER_M for a model that is not listed or after a price change.
PRICES = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}


class GeminiLLM:
    """Gemini for routing and wording. Tools still produce every number."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from google import genai  # imported here so the offline path has no dependency
        from google.genai import types

        self.name = model or os.environ.get("COPILOT_GEMINI_MODEL", DEFAULT_MODEL)
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"],
            # Without a timeout one stuck request can hold a run for minutes.
            http_options=types.HttpOptions(
                timeout=int(float(os.environ.get("COPILOT_TIMEOUT_S", "60")) * 1000)))
        listed_in, listed_out = PRICES.get(self.name, (0.0, 0.0))
        self.price_in = float(os.environ.get("COPILOT_PRICE_IN_PER_M", listed_in))
        self.price_out = float(os.environ.get("COPILOT_PRICE_OUT_PER_M", listed_out))
        self.price_known = self.name in PRICES or "COPILOT_PRICE_IN_PER_M" in os.environ
        self.retry_wait_s = float(os.environ.get("COPILOT_RETRY_WAIT_S", "30"))

    def _generate(self, system: str, user: str, span: dict, as_json: bool = False) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system, temperature=0.2,
            response_mime_type="application/json" if as_json else "text/plain")
        response = None
        for attempt in range(3):  # free-tier keys are rate limited or slow: wait and try again
            try:
                response = self.client.models.generate_content(model=self.name, contents=user,
                                                               config=config)
                break
            except Exception as exc:
                text = f"{type(exc).__name__} {exc}".lower()
                limited = "429" in text or "resource_exhausted" in text
                timed_out = "timeout" in text or "timed out" in text
                if not (limited or timed_out) or attempt == 2:
                    raise
                if timed_out:
                    span["timeouts"] = attempt + 1
                    continue  # a slow request is retried at once
                span["rate_limit_waits"] = attempt + 1
                time.sleep(self.retry_wait_s)
        usage = response.usage_metadata
        tokens_in = getattr(usage, "prompt_token_count", 0) or 0
        # Newer models "think" before answering. Thinking tokens are billed as output.
        thinking = getattr(usage, "thoughts_token_count", 0) or 0
        tokens_out = (getattr(usage, "candidates_token_count", 0) or 0) + thinking
        span.update(model=self.name, tokens_in=tokens_in, tokens_out=tokens_out,
                    thinking_tokens=thinking,
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
            span["fallback"] = describe_error(exc)
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
            span["fallback"] = describe_error(exc)
            return draft


def default_llm():
    """Gemini when a key is configured, otherwise the offline rules."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        try:
            return GeminiLLM()
        except Exception:
            return OfflineLLM()
    return OfflineLLM()
