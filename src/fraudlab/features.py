"""Point-in-time features. Every feature uses only rows that precede the transaction in time order,
and the one label-derived feature honours the confirmation delay.

Fraud labels do not exist when a payment is authorised: a chargeback or an analyst confirms them
`LABEL_DELAY_HOURS` later. A feature that counts "earlier fraud to this destination" is only legal
if it counts frauds that would already have been confirmed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ELIGIBLE_TYPES = ("TRANSFER", "CASH_OUT")
LABEL_DELAY_HOURS = 48
NUMERIC = [
    "log_amount", "amount_to_balance", "hour_of_day", "orig_tx_count_prior",
    "dest_prior_inbound", "dest_is_new", "dest_confirmed_fraud_prior", "log_dest_old_balance",
]  # fmt: skip
CATEGORICAL = ["type"]
FEATURES = NUMERIC + CATEGORICAL
# Not available when the decision is made, or the label itself.
LEAKY = {"isFraud", "newbalanceDest", "error_balance_dest"}
LEAK_DEMO = ["error_balance_dest"]  # added only by the ablation that shows what the trap is worth


def assert_no_leakage(features: list[str]) -> None:
    bad = sorted(set(features) & LEAKY)
    if bad:
        raise ValueError(f"leaky features: {bad}")


def _confirmed_fraud_prior(d: pd.DataFrame, delay: int) -> np.ndarray:
    """Per row: frauds into the same destination with step + delay <= this row's step."""
    out = np.zeros(len(d), dtype=int)
    fr = d.loc[d["isFraud"] == 1]
    if fr.empty:
        return out
    fsteps = {k: np.sort(g["step"].to_numpy()) for k, g in fr.groupby("nameDest")}
    sub = d.loc[d["nameDest"].isin(fsteps)]
    for dest, g in sub.groupby("nameDest"):
        out[g.index.to_numpy()] = np.searchsorted(fsteps[dest], g["step"].to_numpy() - delay, side="right")
    return out


def build(df: pd.DataFrame, label_delay: int = LABEL_DELAY_HOURS) -> pd.DataFrame:
    """Sort by time (ties keep file order) and add the features. Counts use ALL transaction types:
    history is about the account, not about which rows the model will later score."""
    if label_delay <= 0:
        raise ValueError("label_delay must be positive: a label cannot be known at the event itself")
    d = df.copy()
    d["_row"] = np.arange(len(d))
    d = d.sort_values(["step", "_row"], kind="stable").drop(columns="_row").reset_index(drop=True)
    d["log_amount"] = np.log1p(d["amount"])
    d["amount_to_balance"] = (d["amount"] / (d["oldbalanceOrg"] + 1)).clip(upper=2.0)
    d["hour_of_day"] = (d["step"] - 1) % 24
    d["orig_tx_count_prior"] = d.groupby("nameOrig").cumcount()
    d["dest_prior_inbound"] = d.groupby("nameDest").cumcount()
    d["dest_is_new"] = (d["dest_prior_inbound"] == 0).astype(int)
    d["log_dest_old_balance"] = np.log1p(d["oldbalanceDest"])
    d["dest_confirmed_fraud_prior"] = _confirmed_fraud_prior(d, label_delay)
    d["error_balance_dest"] = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]
    return d
