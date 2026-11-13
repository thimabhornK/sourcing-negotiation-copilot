"""Build a negotiation playbook for one quote.

The playbook is assembled by rules from tool outputs: the should-cost check,
the vendor's contract, rebate position and competing quotes. No language model
is involved, so every number can be traced to a tool result and the same
inputs always give the same plan.
"""

from __future__ import annotations

from copilot import tools


def _alternatives(quote: dict, vendor_id: str | None, rfq_id: str | None) -> list[dict]:
    """Other ways to buy the same material: competing quotes in the same sourcing
    event if there is one, otherwise what other vendors were paid recently."""
    if rfq_id:
        event = tools.rfq_summary(rfq_id)
        if event.get("found"):
            return [{"vendor_id": q["vendor"]["vendor_id"],
                     "vendor_name": q["vendor"]["vendor_name"],
                     "price_usd": q["quoted_price_usd"], "source": f"quote in {rfq_id}",
                     "firm": True}
                    for q in event["quotes"] if q["vendor"]["vendor_id"] != vendor_id]
    seen: dict[str, dict] = {}
    for p in reversed(quote["recent_purchases"]):
        if p["vendor_id"] != vendor_id and p["vendor_id"] not in seen:
            seen[p["vendor_id"]] = {"vendor_id": p["vendor_id"], "price_usd": p["unit_price_usd"],
                                    "source": f"purchase on {p['date']}", "firm": False}
    return sorted(seen.values(), key=lambda a: a["price_usd"])


def build_playbook(material_id: str, qty: float, price: float, vendor_id: str,
                   currency: str = "USD", date: str | None = None,
                   rfq_id: str | None = None) -> dict:
    """Negotiation plan for one quote: opening ask, target and walk-away price, the
    arguments to use with their contract citations, and what to trade."""
    quote = tools.quote_check(material_id, qty, price, currency, vendor_id, date)
    quoted, sc = quote["quoted_price_usd"], quote["should_cost_usd"]
    year = int(quote["quote_date"][:4])
    terms = tools.contract_terms(vendor_id)
    rebate = tools.rebate_status(vendor_id, year)
    alternatives = _alternatives(quote, vendor_id, rfq_id)
    batna = min(alternatives, key=lambda a: a["price_usd"]) if alternatives else None
    cites = terms.get("citations", {})

    # Price positions. Target: should-cost, or the best alternative if that is lower.
    # Only a live competing quote counts as a firm alternative. A past purchase from
    # another vendor is an argument, but not something we can walk away to.
    firm = batna["price_usd"] if batna and batna["firm"] else None
    price_is_fair = quoted <= sc["p50"] and not (firm and firm < quoted)
    if price_is_fair:  # little room on price: ask for a small improvement, then trade terms
        target, opening = quoted * 0.99, quoted * 0.97
    else:
        target = min(quoted, sc["p50"], firm or quoted)
        opening = target * (1 - (sc["p50"] - sc["p10"]) / (2 * sc["p50"]))
    ceiling = sc["p50"] * (1 + quote["flag_threshold_pct"] / 100)
    walk_away = max(target, min(quoted, firm or ceiling))

    levers = []
    if not price_is_fair:
        levers.append({
            "lever": "should_cost",
            "argument": f"Our cost model puts this material at {sc['p50']:.4f} USD per "
                        f"{quote['unit']} (range {sc['p10']:.4f} to {sc['p90']:.4f}). The quote "
                        f"is {quote['gap_vs_should_cost_pct']:+.1f}% against that."})
    mk = quote["market"]
    if mk["change_12m_pct"] < 0:
        arg = (f"The {mk['index']} index has moved {mk['change_12m_pct']:+.1f}% in 12 months "
               f"and {mk['change_3m_pct']:+.1f}% in 3 months.")
        lever = {"lever": "index_trend", "argument": arg}
        if terms.get("index_linked") and abs(mk["change_12m_pct"]) > terms.get("index_band_pct", 0):
            lever["argument"] += (f" The contract allows a price adjustment when the index moves "
                                  f"more than {terms['index_band_pct']:g}%.")
            lever["citation"] = cites.get("price")
        levers.append(lever)
    if batna and batna["price_usd"] < quoted:
        levers.append({
            "lever": "competing_offer",
            "argument": f"{batna.get('vendor_name', batna['vendor_id'])} "
                        f"{'is at' if batna['firm'] else 'was paid'} "
                        f"{batna['price_usd']:.4f} USD ({batna['source']}), "
                        f"{100 * (1 - batna['price_usd'] / quoted):.1f}% below this quote."})
    order_value = quoted * qty
    if rebate.get("gap_to_next_tier_usd") is not None:
        gap = rebate["gap_to_next_tier_usd"]
        if order_value >= gap:
            levers.append({
                "lever": "rebate_tier",
                "argument": f"This order ({order_value:,.0f} USD) takes {year} spend past "
                            f"{rebate['next_tier_usd']:,.0f} USD, which triggers the "
                            f"{rebate['next_tier_pct']:g}% annual rebate. Confirm it in writing.",
                "citation": rebate["citation"]})
        elif gap <= 3 * order_value:
            levers.append({
                "lever": "rebate_tier",
                "argument": f"{year} spend is {gap:,.0f} USD short of the "
                            f"{rebate['next_tier_pct']:g}% rebate tier. Offer to bring forward "
                            f"volume in exchange for applying the rebate now.",
                "citation": rebate["citation"]})
    if rebate.get("rebate_due_usd") and rebate.get("must_be_claimed"):
        levers.append({
            "lever": "rebate_claim",
            "argument": f"A {rebate['rebate_pct_reached']:g}% rebate (about "
                        f"{rebate['rebate_due_usd']:,.0f} USD) has been earned for {year} and "
                        f"must be claimed in writing.",
            "citation": rebate["citation"]})
    vendor = quote.get("vendor")
    if vendor and vendor["on_time_rate"] < 0.85:
        levers.append({
            "lever": "delivery_record",
            "argument": f"Only {100 * vendor['on_time_rate']:.0f}% of this vendor's lines arrived "
                        f"on time. Late delivery carries damages of "
                        f"{terms.get('late_penalty_pct_per_week', 0):g}% per week under the "
                        f"contract.",
            "citation": cites.get("late")})

    trades = []
    if terms.get("early_pay_discount_pct"):
        trades.append({"give": f"Pay within {terms['early_pay_days']} days",
                       "get": f"The {terms['early_pay_discount_pct']:g}% early-payment discount "
                              f"already in the contract",
                       "citation": cites.get("payment")})
    elif terms.get("payment_days", 0) >= 60:
        trades.append({"give": f"Shorten payment from {terms['payment_days']} to 30 days",
                       "get": "A further 1% off the unit price", "citation": cites.get("payment")})
    trades.append({"give": "Commit the next two orders at the agreed price",
                   "get": "Price held at target for six months"})

    return {
        "tool": "build_playbook", "vendor_id": vendor_id,
        "vendor_name": vendor["vendor_name"] if vendor else vendor_id,
        "quote": quote, "price_is_fair": price_is_fair,
        "target_price_usd": round(target, 4), "opening_ask_usd": round(opening, 4),
        "walk_away_usd": round(walk_away, 4),
        "saving_at_target_usd": round((quoted - target) * qty, 2),
        "order_value_usd": round(order_value, 0),
        "batna": batna, "alternatives": alternatives[:3],
        "levers": levers, "trades": trades,
        "contract": {k: terms.get(k) for k in ("payment_days", "index_linked",
                                               "price_review_months", "termination_notice_days")},
        "rebate": rebate,
    }


def draft_email(pb: dict) -> str:
    q = pb["quote"]
    # What goes to the vendor is deliberately vaguer than the internal playbook:
    # no competitor names, no competitor prices, no exact internal estimate.
    mk = q["market"]
    external = {
        "should_cost": "Our cost analysis, based on material content and published index "
                       "levels, does not support the quoted price.",
        "index_trend": f"The {mk['index']} index has moved {mk['change_12m_pct']:+.1f}% over "
                       f"the last 12 months and {mk['change_3m_pct']:+.1f}% over the last 3.",
        "competing_offer": "We hold a materially lower offer for the same specification."
                           if pb["batna"] and pb["batna"]["firm"] else
                           "We have recently bought the same specification at a materially "
                           "lower price.",
    }
    points = "\n".join(f"- {external[lv['lever']]}" for lv in pb["levers"]
                       if lv["lever"] in external)
    ask = (f"We would like to close at {pb['opening_ask_usd']:.4f} USD per {q['unit']}."
           if not pb["price_is_fair"] else
           f"The price is close to our expectations. We would like to close at "
           f"{pb['opening_ask_usd']:.4f} USD per {q['unit']} and to discuss terms.")
    return (f"Subject: {q['material_id']} quotation, {q['qty']:,.0f} {q['unit']}\n\n"
            f"Dear {pb['vendor_name']} team,\n\n"
            f"Thank you for your quotation of {q['quoted_price_usd']:.4f} USD per {q['unit']}.\n"
            f"Before we can award this order we need to address the following:\n{points}\n\n"
            f"{ask} Please send your revised offer by the end of this week.\n\nBest regards,\n"
            f"Procurement")
