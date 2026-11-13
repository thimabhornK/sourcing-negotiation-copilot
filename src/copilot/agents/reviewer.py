"""Reviewer: check that every number in an answer comes from a tool result.

Language models are good at wording and unreliable at figures. The reviewer
extracts every number from the draft answer and looks for it in the facts the
tools returned. A number with no source is reported, and the answer is sent
back to the writer or replaced by the deterministic draft.
"""

from __future__ import annotations

import re

NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")
IGNORE = re.compile(r"\b(?:MAT|RFQ|V|CT)-\d+\b|\b20\d{2}-\d{2}-\d{2}\b|\bclause \d+\b"
                    r"|\bround \d+\b|\bnet \d+\b|\btop \d+\b", re.I)
SMALL_INTEGERS = set(range(0, 13))


def _collect(value, out: set[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int | float):
        out.add(float(value))
    elif isinstance(value, str):
        for m in NUMBER.findall(value):
            out.add(float(m.replace(",", "")))
    elif isinstance(value, dict):
        for v in value.values():
            _collect(v, out)
    elif isinstance(value, list | tuple | set):
        for v in value:
            _collect(v, out)


def known_numbers(facts) -> set[float]:
    """Every number in the facts, plus the two conversions the writers use:
    a rate shown as a percentage, and a ratio shown as a percentage difference."""
    base: set[float] = set()
    _collect(facts, base)
    derived = set(base)
    for n in base:
        if 0 <= n <= 2:
            derived.add(n * 100)
            derived.add((n - 1) * 100)
    return derived


def review(text: str, facts) -> dict:
    known = known_numbers(facts)
    unsupported = []
    for token in NUMBER.findall(IGNORE.sub(" ", text)):
        clean = token.replace(",", "").lstrip("+")
        value = float(clean)
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        if decimals == 0 and abs(value) in SMALL_INTEGERS:
            continue
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(abs(k) - abs(value)) <= tolerance for k in known):
            unsupported.append(token)
    return {"ok": not unsupported, "unsupported_numbers": unsupported}
