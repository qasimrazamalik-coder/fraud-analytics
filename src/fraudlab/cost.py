"""Cost-based operating point.

Flagging a transaction costs `review_cost` (an analyst looks at it). Not flagging a fraud costs its
amount. Assumption, stated rather than hidden: a flagged fraud is stopped in full.
"""

from __future__ import annotations

import numpy as np


def best_cut(y: np.ndarray, score: np.ndarray, amount: np.ndarray, review_cost: float) -> dict:
    """Choose how many top-scored transactions to flag so total cost is lowest.

    Cuts are only allowed between distinct score values, so tied transactions are flagged together.
    Returns the threshold (flag when score >= threshold; inf means flag nothing) and the cost.
    """
    y, score, amount = np.asarray(y), np.asarray(score, float), np.asarray(amount, float)
    order = np.argsort(-score, kind="stable")
    s = score[order]
    caught = np.concatenate([[0.0], np.cumsum(np.where(y[order] == 1, amount[order], 0.0))])
    total_fraud = float(amount[y == 1].sum())
    k = np.arange(len(s) + 1)
    cost = (total_fraud - caught) + k * review_cost
    allowed = np.ones(len(s) + 1, dtype=bool)
    allowed[1:-1] = s[:-1] != s[1:]
    cost = np.where(allowed, cost, np.inf)
    best = int(np.argmin(cost))  # first minimum: the smaller alert volume wins a tie
    thr = float("inf") if best == 0 else float(s[best - 1])
    return {"threshold": thr, "flagged": best, "cost": float(cost[best])}


def apply(y: np.ndarray, score: np.ndarray, amount: np.ndarray, threshold: float, review_cost: float) -> dict:
    y, score, amount = np.asarray(y), np.asarray(score, float), np.asarray(amount, float)
    flagged = score >= threshold
    fraud = y == 1
    caught = float(amount[flagged & fraud].sum())
    total_fraud = float(amount[fraud].sum())
    n_flag = int(flagged.sum())
    cost = (total_fraud - caught) + n_flag * review_cost
    return {
        "alerts": n_flag,
        "alert_rate": n_flag / len(y),
        "recall": float((flagged & fraud).sum() / max(1, fraud.sum())),
        "precision": float((flagged & fraud).sum() / max(1, n_flag)),
        "fraud_amount_total": total_fraud,
        "fraud_amount_caught": caught,
        "cost": cost,
        "cost_without_model": total_fraud,
        "saving": total_fraud - cost,
    }
