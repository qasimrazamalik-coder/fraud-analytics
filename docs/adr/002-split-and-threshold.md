# ADR-002: Three time blocks with purges, and a threshold chosen by cost on validation

Status: accepted.

## Context

A random split lets a model train on hours that come after the ones it is tested on, and fraud rings
reuse accounts across hours. Labels also arrive late: a transaction at hour t is only known to be fraud
at about t + 48. And a model's PR-AUC does not say where to cut: that depends on what a review costs and
what a missed fraud costs.

## Decision

- Fit, validation and test are consecutive time blocks (50% / 25% / 25% of the observed hours).
- A row is purged from a block when its label is confirmed after the next block starts, so no block is
  fitted or tuned on a label the next block's clock has not produced yet.
- The model is selected on validation PR-AUC. The alert threshold is the cut that minimises
  `missed fraud amount + alerts x review cost` on validation. Test is touched once, with that threshold.
- Review cost is an input. The README shows three values, because the right threshold moves with it.
- Assumption: an alert stops the fraud in full. Real systems stop some and lose some, so the savings
  figure is an upper bound on what the policy can do.

## Consequences

- With about 200 fraud rows in test, PR-AUC carries a bootstrap interval of roughly +/- 0.07 (seed 42:
  0.42 to 0.57 for gradient boosting). In that run the intervals of logistic regression and gradient
  boosting overlap, so the ranking of the two models is not established by this test block alone.
- A test rewrites the test-block labels and checks that the threshold and the validation scores do not
  move. That is what "chosen on validation only" means in code.
- Amounts are in currency units of the data. The 200 default review cost is an illustration, not a
  benchmark from any company.
