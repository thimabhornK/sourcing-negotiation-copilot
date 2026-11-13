"""Train and evaluate the should-cost model.

Two models are compared against the rule of thumb buyers use today:

  baseline  last price paid for the material, adjusted by the index change
  model A   a transparent cost formula per material (linear regression):
            price = a * commodity index + b * labour index, with a volume term
  model B   gradient-boosted trees with quantile loss, giving a low / mid / high
            range (p10 / p50 / p90) instead of a single number

Data up to the end of 2024 is used for training; 2025 is held out. Splitting by
time matters: a random split would let the model peek at future prices.

Run with:  python -m copilot.model.should_cost
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from copilot import config
from copilot.features import CATEGORICAL, FEATURES, build_features

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
MODEL_FILE = "should_cost.pkl"


def mape(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(predicted - actual) / actual))


class CostFormulaModel:
    """Model A: one small linear model per material, easy to explain to a buyer."""

    def fit(self, X: pd.DataFrame, price: np.ndarray) -> CostFormulaModel:
        self.models_ = {}
        for code, part in X.groupby("material_code"):
            # Remove the volume effect first, then fit cost = a*index + b*labour.
            slope = np.polyfit(part["log_qty_vs_ref"], np.log(price[part.index]), 1)[0]
            adj = price[part.index] / np.exp(slope * part["log_qty_vs_ref"])
            lin = LinearRegression().fit(part[["index_lag1", "labor_lag1"]], adj)
            self.models_[code] = (slope, lin)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(X))
        for code, part in X.groupby("material_code"):
            slope, lin = self.models_[code]
            pos = X.index.get_indexer(part.index)
            out[pos] = (lin.predict(part[["index_lag1", "labor_lag1"]])
                        * np.exp(slope * part["log_qty_vs_ref"]))
        return out


class QuantileBoostedModel:
    """Model B: three boosted-tree models, one per quantile, on log price."""

    def fit(self, X: pd.DataFrame, price: np.ndarray) -> QuantileBoostedModel:
        self.models_ = {}
        for name, q in QUANTILES.items():
            self.models_[name] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=400, learning_rate=0.06,
                categorical_features=CATEGORICAL, random_state=config.SEED,
            ).fit(X[FEATURES], np.log(price))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: np.exp(m.predict(X[FEATURES]))
                             for name, m in self.models_.items()}, index=X.index)


class ShouldCostModel:
    """What the rest of the system uses: a mid estimate with a low / high band.

    The mid estimate comes from the cost formula, because it follows the market
    index outside the range seen in training (trees cannot extrapolate). The band
    width comes from the boosted quantile model.
    """

    def __init__(self, formula: CostFormulaModel, boosted: QuantileBoostedModel,
                 flag_threshold: float = 0.10):
        self.formula, self.boosted, self.flag_threshold = formula, boosted, flag_threshold

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        band = self.boosted.predict(X)
        mid = self.formula.predict(X)
        return pd.DataFrame({"p10": mid * band["p10"] / band["p50"], "p50": mid,
                             "p90": mid * band["p90"] / band["p50"]}, index=X.index)


def last_price_baseline(po: pd.DataFrame, indices: pd.DataFrame, materials: pd.DataFrame,
                        rows: pd.DataFrame, date_col: str) -> np.ndarray:
    """Buyer's rule of thumb: last price paid before the date, scaled by the index move."""
    idx = indices.set_index("month")
    index_id = materials.set_index("material_id")["index_id"]

    def index_at(material_ids, dates):
        month = pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp()
        return np.array([idx.at[m, index_id[mat]] for mat, m in
                         zip(material_ids, month, strict=True)])

    hist = po.sort_values("po_date")[["material_id", "po_date", "unit_price_usd"]]
    target = rows[["material_id", date_col]].reset_index().sort_values(date_col)
    joined = pd.merge_asof(target, hist, left_on=date_col, right_on="po_date",
                           by="material_id", allow_exact_matches=False).set_index("index")
    joined = joined.loc[rows.index]
    then = index_at(joined["material_id"], joined["po_date"].fillna(joined[date_col]))
    now = index_at(joined["material_id"], joined[date_col])
    return (joined["unit_price_usd"] * now / then).to_numpy()


def best_threshold(gap: np.ndarray, label: np.ndarray) -> float:
    """Pick the 'this much above should-cost is overpriced' cut-off with the best F1."""
    best, best_f1 = 0.10, -1.0
    for t in np.arange(0.04, 0.30, 0.01):
        flag = gap > t
        tp = (flag & label).sum()
        if tp == 0:
            continue
        precision, recall = tp / flag.sum(), tp / label.sum()
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return round(best, 2)


def run() -> dict:
    po = pd.read_parquet(config.CURATED / "po_lines.parquet")
    materials = pd.read_parquet(config.CURATED / "materials.parquet")
    indices = pd.read_parquet(config.CURATED / "market_indices.parquet")
    quotes = pd.read_parquet(config.CURATED / "rfq_quotes.parquet")
    labels = pd.read_csv(config.RAW / "eval_quote_labels.csv")
    quotes = quotes.merge(labels, on=["rfq_id", "vendor_id"], validate="one_to_one")

    cut = pd.Timestamp(config.TEST_START)
    train = po[po["po_date"] < cut].reset_index(drop=True)
    test = po[po["po_date"] >= cut].reset_index(drop=True)
    X_train = build_features(train, materials, indices, "po_date")
    X_test = build_features(test, materials, indices, "po_date")
    y_train, y_test = train["unit_price_usd"].to_numpy(), test["unit_price_usd"].to_numpy()

    formula = CostFormulaModel().fit(X_train, y_train)
    boosted = QuantileBoostedModel().fit(X_train, y_train)
    model = ShouldCostModel(formula, boosted)

    pred = model.predict(X_test)
    boosted_pred = boosted.predict(X_test)
    baseline = last_price_baseline(po, indices, materials, test, "po_date")
    report = {
        "train_rows": len(train), "test_rows": len(test), "test_period": "2025",
        "mape_baseline_last_price": round(mape(y_test, baseline), 4),
        "mape_model_a_cost_formula": round(mape(y_test, pred["p50"]), 4),
        "mape_model_b_boosted_p50": round(mape(y_test, boosted_pred["p50"]), 4),
        "band_coverage_p10_p90": round(float(((y_test >= pred["p10"])
                                              & (y_test <= pred["p90"])).mean()), 4),
    }

    # Overpriced-quote detection: tune the cut-off on older quotes, test on 2025 quotes.
    Xq = build_features(quotes, materials, indices, "quote_date")
    gap = (quotes["quoted_price_usd"] / model.predict(Xq)["p50"] - 1).to_numpy()
    label = quotes["is_overpriced"].to_numpy()
    is_test = (quotes["quote_date"] >= cut).to_numpy()
    model.flag_threshold = best_threshold(gap[~is_test], label[~is_test])
    flag = gap[is_test] > model.flag_threshold
    tp = int((flag & label[is_test]).sum())
    report.update({
        "flag_threshold": model.flag_threshold,
        "test_quotes": int(is_test.sum()),
        "test_quotes_overpriced": int(label[is_test].sum()),
        "flag_precision": round(tp / max(int(flag.sum()), 1), 4),
        "flag_recall": round(tp / max(int(label[is_test].sum()), 1), 4),
    })

    config.MODELS.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    with open(config.MODELS / MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    (config.REPORTS / "should_cost_eval.json").write_text(json.dumps(report, indent=2))
    return report


def load_model() -> ShouldCostModel:
    with open(config.MODELS / MODEL_FILE, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Import through the package so the saved model can be loaded from other modules.
    from copilot.model import should_cost as _module

    result = _module.run()
    print("Should-cost evaluation (held-out year)")
    for key, value in result.items():
        print(f"  {key:<32} {value}")
