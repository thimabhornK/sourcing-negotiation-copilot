"""Writer: turn tool results into a draft answer with fixed templates.

The draft is always correct by construction, because it only formats values
the tools returned. A language model may reword it afterwards, and the
reviewer checks that the rewording did not change a number.
"""

from __future__ import annotations

from copilot.quote_check import render as render_quote

HELP = """I can help in three ways.

Ask about spend, vendors and contracts
  "Top 5 vendors by spend in 2025"
  "Which vendors charge the most above should-cost for steel sheet?"
  "What are the payment terms with V-0003?"
  "Are we close to a rebate with V-0002 in 2025?"

Check a quote
  "Is 1.62 USD fair for 5000 kg of MAT-0001 from V-0002?"
  "Compare the quotes in RFQ-01990"

Negotiate
  "Build a negotiation playbook for that quote"
  "Let me practise against the vendor"
  "Run the negotiation automatically"
"""


def _scope(f: dict) -> str:
    bits = [str(f["year"])] if f.get("year") else ["all years"]
    if f.get("category"):
        bits.append(f["category"].replace("_", " "))
    if f.get("material_id"):
        bits.append(f["material_id"])
    return ", ".join(bits)


def table(f: dict) -> str:
    rows = f["rows"]
    if not rows:
        return f"No purchases match ({_scope(f)})."
    if f["tool"] == "top_vendors_by_spend":
        head = f"Vendors with the highest spend ({_scope(f)}):"
        body = [f"  {i}. {r['vendor_name']} ({r['vendor_id']}): {r['spend_usd']:,.0f} USD across "
                f"{r['po_lines']:,.0f} order lines" for i, r in enumerate(rows, 1)]
    elif f["tool"] == "most_expensive_vendors":
        head = f"Vendors priced furthest above should-cost ({_scope(f)}):"
        body = [f"  {i}. {r['vendor_name']} ({r['vendor_id']}): typically "
                f"{r['pct_above_should_cost']:+.1f}% against should-cost, "
                f"{r['excess_spend_usd']:,.0f} USD paid above it on {r['po_lines']:,.0f} lines"
                for i, r in enumerate(rows, 1)]
    elif f["tool"] == "late_vendors":
        head = f"Vendors that deliver late most often ({_scope(f)}):"
        body = [f"  {i}. {r['vendor_name']} ({r['vendor_id']}): {r['late_pct']:.1f}% of "
                f"{r['delivered_lines']:,.0f} delivered lines were late"
                for i, r in enumerate(rows, 1)]
    else:
        head = f"Spend by category ({_scope(f)}):"
        body = [f"  {r['category'].replace('_', ' ')}: {r['spend_usd']:,.0f} USD from "
                f"{r['vendors']:,.0f} vendors" for r in rows]
    return "\n".join([head, *body])


def vendor(f: dict) -> str:
    if not f.get("found"):
        return f"I could not find vendor {f['vendor_id']}."
    return (f"{f['vendor_name']} ({f['vendor_id']}) scores {f['total_score']:.1f} out of 100, "
            f"rank {f['rank']} of {f['of']}.\n"
            f"  Delivery   {100 * f['on_time_rate']:.0f}% of lines on time\n"
            f"  Quality    {100 * f['quality_rate']:.1f}% pass rate\n"
            f"  Price      {100 * (f['price_ratio'] - 1):+.1f}% against should-cost\n"
            f"  Risk       credit rating {f['credit_rating']}")


def contract(search: dict, terms: dict | None) -> str:
    results = search["results"]
    if not results:
        return "I could not find a contract clause that answers this."
    top = results[0]
    lines = [f"{top['text']}", f"  Source: {top['citation']} ({top['title']})"]
    return "\n".join(lines)


def rebate(f: dict) -> str:
    if not f["has_rebate"]:
        return f"The contract with {f['vendor_id']} has no volume rebate ({f['citation']})."
    lines = [f"Spend with {f['vendor_id']} in {f['year']} is {f['spend_usd']:,.0f} USD."]
    if f.get("rebate_due_usd"):
        lines.append(f"A {f['rebate_pct_reached']:g}% rebate has been earned, about "
                     f"{f['rebate_due_usd']:,.0f} USD."
                     + (" It must be claimed in writing." if f.get("must_be_claimed") else ""))
    if f.get("gap_to_next_tier_usd") is not None:
        lines.append(f"The next tier ({f['next_tier_pct']:g}% at {f['next_tier_usd']:,.0f} USD) "
                     f"is {f['gap_to_next_tier_usd']:,.0f} USD away.")
    lines.append(f"  Source: {f['citation']}")
    return "\n".join(lines)


def rfq(f: dict) -> str:
    if not f.get("found"):
        return f"I could not find {f['rfq_id']}."
    lines = [f"{f['rfq_id']}: {f['qty']:,.0f} of {f['material_id']}, quoted on {f['quote_date']}.",
             f"Should-cost is {f['quotes'][0]['should_cost_usd']['p50']:.4f} USD."]
    for q in f["quotes"]:
        v = q["vendor"]
        lines.append(f"  {v['vendor_name']} ({v['vendor_id']}): {q['quoted_price_usd']:.4f} USD, "
                     f"{q['gap_vs_should_cost_pct']:+.1f}% against should-cost, {q['verdict']}. "
                     f"Vendor score {v['total_score']:.1f}, lead time {q['lead_time_days']} days.")
    return "\n".join(lines)


def playbook(pb: dict, email: str) -> str:
    q = pb["quote"]
    lines = [f"Negotiation playbook: {pb['vendor_name']}, {q['material_id']}, "
             f"{q['qty']:,.0f} {q['unit']}",
             f"  Quoted        {q['quoted_price_usd']:.4f} USD ({q['gap_vs_should_cost_pct']:+.1f}%"
             f" against should-cost, {q['verdict']})",
             f"  Open at       {pb['opening_ask_usd']:.4f} USD",
             f"  Target        {pb['target_price_usd']:.4f} USD "
             f"(saves {pb['saving_at_target_usd']:,.2f} USD on this order)",
             f"  Walk away at  {pb['walk_away_usd']:.4f} USD"]
    if b := pb["batna"]:
        kind = "" if b["firm"] else ", a past price and not a live offer"
        lines.append(f"  Alternative   {b.get('vendor_name', b['vendor_id'])} at "
                     f"{b['price_usd']:.4f} USD ({b['source']}{kind})")
    else:
        lines.append("  Alternative   none on record, so the walk-away price is the top of the "
                     "acceptable range")
    if pb["price_is_fair"]:
        lines.append("The price is already at or below should-cost. Ask for a small "
                     "improvement, then put your effort into terms.")
    lines.append("Arguments to use:")
    for lv in pb["levers"]:
        cite = f" [{lv['citation']}]" if lv.get("citation") else ""
        lines.append(f"  - {lv['argument']}{cite}")
    lines.append("What to trade:")
    for t in pb["trades"]:
        cite = f" [{t['citation']}]" if t.get("citation") else ""
        lines.append(f"  - Give: {t['give']}. Get: {t['get']}.{cite}")
    lines += ["", "Draft email:", email]
    return "\n".join(lines)


def need(what: str, example: str) -> str:
    return f"I need {what} to do that. For example: \"{example}\""


__all__ = ["HELP", "contract", "need", "playbook", "rebate", "render_quote", "rfq", "table",
           "vendor"]
