"""Minimal tracing: one span per agent step, with latency, tokens and cost.

Spans are kept in memory for the current turn and can be appended to a JSONL
file (set COPILOT_TRACE_FILE). On Google Cloud the same spans map onto
OpenTelemetry and Cloud Trace.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class Tracer:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else (
            Path(os.environ["COPILOT_TRACE_FILE"]) if os.environ.get("COPILOT_TRACE_FILE")
            else None)
        self.spans: list[dict] = []
        self.turn = 0

    def start_turn(self) -> None:
        self.turn += 1
        self.spans = []

    @contextmanager
    def span(self, agent: str, step: str, **attrs):
        record = {"turn": self.turn, "agent": agent, "step": step, **attrs,
                  "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
        start = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["ms"] = round(1000 * (time.perf_counter() - start), 1)
            self.spans.append(record)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps(record, default=str) + "\n")

    def summary(self) -> dict:
        return {"steps": len(self.spans),
                "ms": round(sum(s["ms"] for s in self.spans), 1),
                "tokens_in": sum(s["tokens_in"] for s in self.spans),
                "tokens_out": sum(s["tokens_out"] for s in self.spans),
                "cost_usd": round(sum(s["cost_usd"] for s in self.spans), 6)}
