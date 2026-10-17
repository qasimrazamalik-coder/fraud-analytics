# Data contract: transaction table

Enforced by `fraudlab.contract.validate` before any feature is built. All violations are collected and
reported together; the CLI exits with code 2.

| Column | Type | Rule |
|---|---|---|
| `step` | integer | >= 1, hours since the start of the window. Rows are ordered by it. |
| `type` | text | One of PAYMENT, CASH_OUT, CASH_IN, TRANSFER, DEBIT |
| `amount` | number | >= 0 |
| `nameOrig`, `nameDest` | text | Not null. Account identifiers. |
| `oldbalanceOrg`, `newbalanceOrig` | number | >= 0 |
| `oldbalanceDest`, `newbalanceDest` | number | >= 0 |
| `isFraud` | 0 or 1 | At least one fraud row in the table |
| `isFlaggedFraud` | 0 or 1 | Optional. Only used to compare against the amount rule; never a feature. |

Provenance is not a column. It is a required CLI flag (`--provenance synthetic|real`) printed into
`reports/metrics.json` and the README block, so a synthetic result cannot be shown as a real one.

## Reported, not fatal

- Fraud rows outside TRANSFER and CASH_OUT. The model scores only those two types (see ADR-001), so these
  rows are unreachable. The count is printed in the README block when it is not zero.

## Columns the model may not use

`newbalanceDest`, `error_balance_dest` (= `oldbalanceDest + amount - newbalanceDest`) and `isFraud`.
They describe what the ledger looked like after the event, or the label itself. The guard is
`features.assert_no_leakage`, and the ablation in the README shows what using them would be worth.
