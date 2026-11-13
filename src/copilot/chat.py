"""Chat with the copilot in a terminal.

  python -m copilot.chat                      interactive
  python -m copilot.chat --ask "Top 5 vendors by spend in 2025"
  python -m copilot.chat --trace              also print each agent step

Without an API key the chat runs on rules and templates. With GEMINI_API_KEY
set, Gemini routes messages and words the answers, and the reviewer checks its
numbers against the tool results.
"""

from __future__ import annotations

import argparse

from copilot.agents.orchestrator import Reply, Session


def show(reply: Reply, trace: bool) -> None:
    print(f"\n[{reply.mode}]\n{reply.text}\n")
    if trace:
        for s in reply.trace:
            extra = f"  tokens {s['tokens_in']}+{s['tokens_out']}" if s["tokens_in"] else ""
            print(f"    {s['agent']:<17} {s['step']:<24} {s['ms']:>7.1f} ms{extra}")
        u = reply.usage
        print(f"    total {u['ms']:.1f} ms, {u['tokens_in'] + u['tokens_out']} tokens, "
              f"{u['cost_usd']:.6f} USD\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sourcing and negotiation copilot")
    ap.add_argument("--ask", action="append", help="ask one question and exit (repeatable)")
    ap.add_argument("--trace", action="store_true", help="print the agent steps for each turn")
    args = ap.parse_args()
    session = Session()
    if args.ask:
        for question in args.ask:
            print(f"> {question}")
            show(session.handle(question), args.trace)
        return
    print(f"Sourcing copilot ({session.llm.name}). Type help for examples, quit to leave.")
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in ("quit", "exit") and not (session.sim and not session.sim.closed):
            break
        if message:
            show(session.handle(message), args.trace)


if __name__ == "__main__":
    main()
