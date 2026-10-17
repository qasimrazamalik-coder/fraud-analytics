"""Data contract for the transaction table. Collects every violation, then raises once."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
]  # fmt: skip
TYPES = {"PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"}
MONEY = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


class DataContractError(Exception):
    pass


def violations(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    v: list[str] = []
    if len(df) == 0:
        return ["table is empty"]
    for c in REQUIRED:
        n = int(df[c].isna().sum())
        if n:
            v.append(f"{c}: {n} null values")
    if v:
        return v  # type and range checks are meaningless with nulls
    if not np.issubdtype(df["step"].dtype, np.integer) or (df["step"] < 1).any():
        v.append("step must be an integer >= 1 (hours since the start)")
    bad_type = sorted(set(df["type"]) - TYPES)
    if bad_type:
        v.append(f"unknown transaction types: {bad_type}")
    for c in MONEY:
        if (df[c] < 0).any():
            v.append(f"{c}: {int((df[c] < 0).sum())} negative values")
    if not set(df["isFraud"].unique()) <= {0, 1}:
        v.append("isFraud must be 0 or 1")
    elif df["isFraud"].sum() == 0:
        v.append("no fraud rows: nothing to learn or evaluate")
    return v


def fraud_outside_scope(df: pd.DataFrame, eligible=("TRANSFER", "CASH_OUT")) -> int:
    """Fraud rows the model can never score. Reported, not fatal: the real data has none."""
    return int(((df["isFraud"] == 1) & ~df["type"].isin(eligible)).sum())


def validate(df: pd.DataFrame) -> None:
    v = violations(df)
    if v:
        raise DataContractError("; ".join(v))
