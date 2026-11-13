"""Check one vendor quote against the should-cost model.

Example:
  python -m copilot.quote_check --material MAT-0001 --qty 5000 --price 3.10 \
      --currency USD --vendor V-0001 --date 2025-06-15

Every number in the output comes from the curated data or the model. This is
the deterministic core that the chat agents will call as a tool, so the
language model never has to do arithmetic itself.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

import pandas as pd

from copilot import config
from copilot.features import build_features
from copilot.model.should_cost import load_model


@lru_cache(maxsize=16)
def _cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def _read(folder, name: str) -> pd.DataFrame:
    """Read a table once and reuse it until the file changes."""
    path = folder / f"{name}.parquet"
    return _cached(str(path), path.stat().st_mtime)


def check_quote(material_id: str, qty: float, price: float, currency: str = "USD",
                vendor_id: str | None = None, date: str | None = None) -> dict:
    materials = _read(config.CURATED, "materials")
    indices = _read(config.CURATED, "market_indices")
    po = _read(config.CURATED, "po_lines")
    if material_id not in set(materials["material_id"]):
        raise ValueError(f"Unknown material: {material_id}")
    when = pd.Timestamp(date) if date else indices["month"].max()
    month = when.to_period("M").to_timestamp()
    if not (indices["month"].min() <= month <= indices["month"].max()):
        raise ValueError(f"No market data for {month:%Y-%m}")
    market = indices.set_index("month")

    rate = {"USD": 1.0, "THB": market.at[month, "thb_per_usd"],
            "EUR": market.at[month, "eur_per_usd"]}.get(currency.upper())
    if rate is None:
        raise ValueError(f"Unsupported currency: {currency}")
    price_usd = price / rate

    model = load_model()
    row = pd.DataFrame([{"material_id": material_id, "qty": qty, "quote_date": when}])
    band = model.predict(build_features(row, materials, indices, "quote_date")).iloc[0]
    gap = price_usd / band["p50"] - 1
    if gap > model.flag_threshold:
        verdict = "OVERPRICED"
    elif price_usd < band["p10"]:
        verdict = "BELOW EXPECTED RANGE"
    else:
        verdict = "WITHIN EXPECTED RANGE"

    mat = materials.set_index("material_id").loc[material_id]
    pos = market.index.get_loc(month)
    level = market[mat["index_id"]]
    history = (po[(po["material_id"] == material_id) & (po["po_date"] < when)]
               .sort_values("po_date").tail(5))
    result = {
        "material_id": material_id, "category": mat["category"], "unit": mat["base_uom"],
        "qty": qty, "quote_date": f"{when:%Y-%m-%d}",
        "quoted_price_usd": round(price_usd, 4),
        "should_cost_usd": {k: round(float(band[k]), 4) for k in ("p10", "p50", "p90")},
        "gap_vs_should_cost_pct": round(100 * gap, 1),
        "verdict": verdict,
        "flag_threshold_pct": round(100 * model.flag_threshold, 1),
        "potential_saving_usd": round(max(price_usd - band["p50"], 0) * qty, 2),
        "market": {
            "index": mat["index_id"],
            "level": float(level.iloc[pos]),
            "change_3m_pct": round(100 * (level.iloc[pos] / level.iloc[max(pos - 3, 0)] - 1), 1),
            "change_12m_pct": round(100 * (level.iloc[pos] / level.iloc[max(pos - 12, 0)] - 1),
                                    1),
        },
        "recent_purchases": [
            {"date": f"{r.po_date:%Y-%m-%d}", "vendor_id": r.vendor_id, "qty": r.qty,
             "unit_price_usd": round(r.unit_price_usd, 4)} for r in history.itertuples()],
    }
    score_path = config.SERVING / "vendor_score.parquet"
    if vendor_id and score_path.exists():
        scores = _read(config.SERVING, "vendor_score").set_index("vendor_id")
        if vendor_id in scores.index:
            s = scores.loc[vendor_id]
            result["vendor"] = {
                "vendor_id": vendor_id, "vendor_name": s["vendor_name"],
                "total_score": float(s["total_score"]), "on_time_rate": float(s["on_time_rate"]),
                "price_ratio": float(s["price_ratio"]),
                "rank": int(scores["total_score"].rank(ascending=False)[vendor_id]),
                "of": len(scores),
            }
    return result


def render(r: dict) -> str:
    sc, mk = r["should_cost_usd"], r["market"]
    lines = [
        f"Quote check: {r['material_id']} ({r['category']}), {r['qty']:,.0f} {r['unit']}"
        f" on {r['quote_date']}",
        f"  Quoted price        {r['quoted_price_usd']:.4f} USD per {r['unit']}",
        f"  Should-cost         {sc['p50']:.4f}  (range {sc['p10']:.4f} to {sc['p90']:.4f})",
        f"  Gap vs should-cost  {r['gap_vs_should_cost_pct']:+.1f}%"
        f"  (flag above +{r['flag_threshold_pct']:.0f}%)",
        f"  Verdict             {r['verdict']}",
        f"  Potential saving    {r['potential_saving_usd']:,.2f} USD if negotiated to should-cost",
        f"  Market ({mk['index']})  {mk['change_3m_pct']:+.1f}% in 3 months,"
        f" {mk['change_12m_pct']:+.1f}% in 12 months",
    ]
    if v := r.get("vendor"):
        lines.append(f"  Vendor              {v['vendor_name']}: score {v['total_score']:.1f}"
                     f" (rank {v['rank']} of {v['of']}), on time {100 * v['on_time_rate']:.0f}%,"
                     f" usually priced {100 * (v['price_ratio'] - 1):+.1f}% vs should-cost")
    if r["recent_purchases"]:
        lines.append("  Recent purchases of this material:")
        lines += [f"    {p['date']}  {p['vendor_id']}  {p['qty']:>10,.0f}  "
                  f"{p['unit_price_usd']:.4f} USD" for p in r["recent_purchases"]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check a vendor quote against should-cost")
    ap.add_argument("--material", required=True)
    ap.add_argument("--qty", type=float, required=True)
    ap.add_argument("--price", type=float, required=True, help="unit price in --currency")
    ap.add_argument("--currency", default="USD", choices=["USD", "THB", "EUR"])
    ap.add_argument("--vendor")
    ap.add_argument("--date", help="YYYY-MM-DD, defaults to the latest month with market data")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()
    result = check_quote(args.material, args.qty, args.price, args.currency, args.vendor,
                         args.date)
    print(json.dumps(result, indent=2) if args.json else render(result))


if __name__ == "__main__":
    main()
