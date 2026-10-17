"""Seeded synthetic generator in the PaySim column layout (SYNTHETIC: not real transactions).

What it imitates, and what it does not:
- Same columns as the Kaggle PaySim file, so the real CSV is a drop-in replacement.
- Fraud only occurs on TRANSFER and CASH_OUT, drains most of the victim's balance, skews to night hours,
  often goes to a small pool of reused "mule" destination accounts, and is sometimes repeated within hours.
- Balances are drawn per transaction, not carried through a ledger (PaySim's are not consistent either).
- ARTEFACT, on purpose: 70% of fraudulent TRANSFERs are never credited to the destination
  (newbalanceDest == oldbalanceDest). That is a post-event ledger effect that would not exist when the
  decision is made. It is the trap the leakage guard and the ablation are built to expose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TYPES = ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
TYPE_P = [0.34, 0.35, 0.22, 0.08, 0.01]
HOURS = 720  # 30 days
RULE_AMOUNT = 200_000  # the flag rule PaySim ships as isFlaggedFraud
COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]  # fmt: skip


def generate(n: int = 300_000, seed: int = 42, fraud_rate: float = 0.0025) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n * fraud_rate))
    n_legit = n - n_fraud
    n_cust = max(50, n // 8)

    # ---- legitimate traffic
    typ = rng.choice(TYPES, n_legit, p=TYPE_P)
    mu = {"PAYMENT": 7.5, "CASH_OUT": 10.0, "CASH_IN": 10.5, "TRANSFER": 10.0, "DEBIT": 7.0}
    sg = {"PAYMENT": 1.0, "CASH_OUT": 1.2, "CASH_IN": 1.0, "TRANSFER": 1.5, "DEBIT": 1.0}
    amount = np.array([rng.lognormal(mu[t], sg[t]) for t in typ]).round(2)
    old_org = rng.lognormal(8.5, 1.5, n_legit).round(2)
    debit = typ != "CASH_IN"
    amount = np.where(debit, np.minimum(amount, old_org), amount).round(2)
    new_org = np.where(debit, old_org - amount, old_org + amount).round(2)
    orig = np.array([f"C{i}" for i in rng.integers(0, n_cust, n_legit)])
    n_mule = 150
    mule_ids = np.array([f"C9{i:04d}" for i in range(n_mule)])
    dest = np.empty(n_legit, dtype=object)
    merchant = np.isin(typ, ["PAYMENT", "DEBIT"])
    dest[merchant] = [f"M{i}" for i in rng.integers(0, n_cust // 4, merchant.sum())]
    cust_dest = ~merchant
    dest[cust_dest] = [f"C{i}" for i in rng.integers(0, n_cust, cust_dest.sum())]
    # mules also receive a little legitimate traffic, so a reused destination is not proof of fraud
    to_mule = cust_dest & (rng.random(n_legit) < 0.004)
    dest[to_mule] = rng.choice(mule_ids, to_mule.sum())
    old_dest = np.where(merchant, 0.0, rng.lognormal(9.0, 2.0, n_legit)).round(2)
    credited = np.isin(typ, ["TRANSFER", "CASH_OUT"])
    new_dest = np.where(credited, old_dest + amount, old_dest).round(2)
    step = rng.integers(1, HOURS + 1, n_legit)
    legit = pd.DataFrame(
        dict(step=step, type=typ, amount=amount, nameOrig=orig, oldbalanceOrg=old_org,
             newbalanceOrig=new_org, nameDest=dest, oldbalanceDest=old_dest,
             newbalanceDest=new_dest, isFraud=0)
    )  # fmt: skip

    # ---- fraud: some events come as a pair (a second drain on the same account within 1-3 hours)
    n_events = int(n_fraud * 0.8)
    ev_hour = np.where(
        rng.random(n_events) < 0.5, rng.integers(0, 7, n_events), rng.integers(0, 24, n_events)
    )
    ev_day = rng.integers(0, HOURS // 24, n_events)
    ev_step = (ev_day * 24 + ev_hour + 1).astype(int)
    ev_org = np.array([f"C{i}" for i in rng.integers(0, n_cust, n_events)])
    frames = []
    for k in range(n_fraud):
        if k < n_events:
            base = k
            s, o = ev_step[base], ev_org[base]
        else:  # repeat on a prior event's account
            base = rng.integers(0, n_events)
            s, o = min(HOURS, ev_step[base] + rng.integers(1, 4)), ev_org[base]
        bal = float(rng.lognormal(10.5, 1.2))
        drain = rng.random() < 0.7
        amt = bal if drain else bal * rng.uniform(0.3, 0.9)
        t = "TRANSFER" if rng.random() < 0.6 else "CASH_OUT"
        d = rng.choice(mule_ids) if rng.random() < 0.55 else f"C{rng.integers(0, n_cust)}"
        od = float(rng.lognormal(9.0, 2.0))
        artefact = t == "TRANSFER" and rng.random() < 0.7
        frames.append(
            (s, t, round(amt, 2), o, round(bal, 2), round(bal - amt, 2), d, round(od, 2),
             round(od if artefact else od + amt, 2), 1)
        )  # fmt: skip
    fraud = pd.DataFrame(frames, columns=legit.columns)

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).sort_values("step", kind="stable").reset_index(drop=True)
    df["isFlaggedFraud"] = ((df["type"] == "TRANSFER") & (df["amount"] > RULE_AMOUNT)).astype(int)
    return df[COLUMNS]


def write(out_dir: str | Path, n: int = 300_000, seed: int = 42) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = generate(n, seed)
    df.to_csv(out / "paysim_like.csv", index=False)
    return len(df)
