"""Split, fit, evaluate. The model scores only TRANSFER and CASH_OUT (the types fraud can occur in);
every other type is passed through by policy, and fraud outside scope is counted, not hidden.

Time layout (the label for a transaction at hour t is confirmed at t + delay):

    fit  [..........] purge  val [......] purge  test [..........]
                      (delay)               (delay)

Rows whose confirmation lands inside the next block are purged, so no block is trained or tuned on a
label that the next block's clock has not yet produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import cost, features
from .synth import RULE_AMOUNT

TARGET = "isFraud"


def temporal_split(
    df: pd.DataFrame, val_start: int, test_start: int, delay: int = features.LABEL_DELAY_HOURS
):
    """Returns (fit, val, test, n_purged)."""
    if not 0 < val_start < test_start:
        raise ValueError("need 0 < val_start < test_start")
    fit = df[df["step"] + delay <= val_start]
    val = df[(df["step"] >= val_start) & (df["step"] + delay <= test_start)]
    test = df[df["step"] >= test_start]
    return fit, val, test, len(df) - len(fit) - len(val) - len(test)


def _pipeline(model, numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]
    )
    return Pipeline([("pre", pre), ("model", model)])


def _models(seed: int) -> dict:
    return {
        "logistic": LogisticRegression(max_iter=2000),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=200, random_state=seed
        ),
    }


def recall_at_alert_rate(y: np.ndarray, score: np.ndarray, rate: float) -> float:
    """Share of all fraud found within the top `rate` fraction of transactions by score."""
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, round(len(y) * rate))
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].sum() / max(1, y.sum()))


def bootstrap_ci(y, score, fn, n_boot: int = 500, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(fn(y[idx], score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def split_points(df: pd.DataFrame) -> tuple[int, int]:
    mx = int(df["step"].max())
    return int(mx * 0.5), int(mx * 0.75)


def evaluate(
    feat: pd.DataFrame,
    seed: int = 42,
    shuffle_labels: bool = False,
    review_cost: float = 200.0,
    cost_grid: tuple[float, ...] = (50.0, 200.0, 800.0),
) -> dict:
    features.assert_no_leakage(features.FEATURES)
    el = feat[feat["type"].isin(features.ELIGIBLE_TYPES)].reset_index(drop=True)
    vs, ts = split_points(feat)
    fit, val, test, purged = temporal_split(el, vs, ts)
    y_fit, y_val, y_te = (x[TARGET].to_numpy() for x in (fit, val, test))
    if shuffle_labels:  # negative control: signal must vanish
        rng = np.random.default_rng(seed)
        y_fit, y_val, y_te = rng.permutation(y_fit), rng.permutation(y_val), rng.permutation(y_te)
    if min(y_fit.sum(), y_val.sum(), y_te.sum()) == 0:
        raise ValueError("a block has no fraud: cannot fit, tune or evaluate")

    prevalence = float(y_te.mean())
    out = {
        "split": {"val_start": vs, "test_start": ts, "label_delay_hours": features.LABEL_DELAY_HOURS,
                  "n_fit": len(fit), "n_val": len(val), "n_test": len(test), "n_purged": purged},
        "fraud_counts": {"fit": int(y_fit.sum()), "val": int(y_val.sum()), "test": int(y_te.sum())},
        "test_prevalence": prevalence,
        "fraud_outside_scope": int(((feat[TARGET] == 1) & ~feat["type"].isin(features.ELIGIBLE_TYPES)).sum()),
        "models": {},
    }  # fmt: skip
    fitted, val_ap, test_scores, val_scores = {}, {}, {}, {}
    for name, model in _models(seed).items():
        pipe = _pipeline(model, features.NUMERIC, features.CATEGORICAL).fit(fit[features.FEATURES], y_fit)
        sv, st = (pipe.predict_proba(x[features.FEATURES])[:, 1] for x in (val, test))
        val_scores[name], test_scores[name] = sv, st
        val_ap[name] = float(average_precision_score(y_val, sv))
        lo, hi = bootstrap_ci(y_te, st, average_precision_score)
        out["models"][name] = {
            "val_pr_auc": val_ap[name],
            "pr_auc": float(average_precision_score(y_te, st)),
            "pr_auc_ci": [lo, hi],
            "roc_auc": float(roc_auc_score(y_te, st)),
            "recall_at_1pct_alerts": recall_at_alert_rate(y_te, st, 0.01),
        }
        fitted[name] = pipe
    best = max(val_ap, key=val_ap.get)  # chosen on validation, never on test
    out["selected_model"] = best

    amt_val, amt_te = val["amount"].to_numpy(), test["amount"].to_numpy()
    rule = ((test["type"] == "TRANSFER") & (test["amount"] > RULE_AMOUNT)).to_numpy().astype(float)
    sens = {}
    for rc in sorted(set(cost_grid) | {review_cost}):
        cut = cost.best_cut(y_val, val_scores[best], amt_val, rc)  # threshold from validation only
        res = cost.apply(y_te, test_scores[best], amt_te, cut["threshold"], rc)
        res["threshold"] = cut["threshold"]
        res["rule_cost"] = cost.apply(y_te, rule, amt_te, 1.0, rc)["cost"]
        sens[str(int(rc)) if float(rc).is_integer() else str(rc)] = res
    key = str(int(review_cost)) if float(review_cost).is_integer() else str(review_cost)
    out["decision"] = {"review_cost": review_cost, "at_review_cost": sens[key], "sensitivity": sens}

    if not shuffle_labels:
        leaky = features.NUMERIC + features.LEAK_DEMO
        pipe = _pipeline(_models(seed)["gradient_boosting"], leaky, features.CATEGORICAL)
        pipe.fit(fit[leaky + features.CATEGORICAL], y_fit)
        s = pipe.predict_proba(test[leaky + features.CATEGORICAL])[:, 1]
        out["ablation_with_ledger_artefact"] = {
            "pr_auc": float(average_precision_score(y_te, s)),
            "roc_auc": float(roc_auc_score(y_te, s)),
        }
        out["curves"] = {
            "y_test": y_te.tolist(), "score_clean": test_scores[best].tolist(), "score_leaky": s.tolist(),
            "amount_test": amt_te.tolist(),
        }  # fmt: skip
        rule_stats = cost.apply(y_te, rule, amt_te, 1.0, review_cost)
        out["rule_baseline"] = {k: rule_stats[k] for k in ("alerts", "recall", "precision", "cost")}
    return out
