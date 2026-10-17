# ADR-001: Score TRANSFER and CASH_OUT only, and treat post-event ledger columns as leakage

Status: accepted. Figures come from the bundled **synthetic** data (300,000 transactions, seed 42).

## Context

Fraud in the PaySim layout only ever occurs on TRANSFER and CASH_OUT. Scoring PAYMENT, CASH_IN and
DEBIT adds volume and no positives, and it dilutes every precision figure. Separately, the table carries
`newbalanceDest`: the destination's balance after the event. When a transfer is blocked or reversed the
credit never lands, so `newbalanceDest == oldbalanceDest` becomes a near-perfect proxy for the label. It is
a property of how the data was produced, not something a payment processor knows before authorising.

## Decision

- The model scores TRANSFER and CASH_OUT. Everything else passes through by policy. Fraud outside
  those types is counted and reported, not hidden.
- `newbalanceDest` and anything derived from it are forbidden as features (`features.LEAKY`). The
  ablation re-adds one derived column to show the size of the trap.
- History features (counts of earlier transactions per account, whether the destination is new) use only
  rows before the transaction in time order. A test cuts the table at a step and checks that features of
  earlier rows do not change.
- The label-derived feature "frauds already confirmed into this destination" counts only frauds whose
  confirmation time (`step + 48`) has passed. A test pins the boundary: a fraud at step 10 is visible at
  step 58 and not at 57.

## Consequences

- On the synthetic data the leak lifts gradient boosting from PR-AUC about 0.49 to about 0.79. Reporting
  the higher number would have been wrong and would have looked better.
- On the real PaySim file the same guard applies, and the ablation will likely show a different gap.
  That comparison is part of the laptop checklist, not a claim made here.
- The 48-hour delay is an assumption about how long chargebacks and manual confirmation take. It is a
  constant (`LABEL_DELAY_HOURS`), so a sensitivity run is a one-line change.
