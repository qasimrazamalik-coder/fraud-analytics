# Metric definitions

| Metric | Definition | Why this one |
|---|---|---|
| PR-AUC | Average precision of the fraud score on the test block, with a 500-resample percentile bootstrap interval (seed 7) | Fraud is under 1% of rows; ROC-AUC rewards ranking the many legitimate rows and hides the cost of false alerts |
| ROC-AUC | Standard | Reported only for comparison with other write-ups |
| Fraud in top 1% by score | Share of all test fraud inside the top 1% of rows by score | Maps to a fixed analyst capacity |
| Alert rate | Flagged rows / test rows | What operations feels |
| Precision, recall at the operating point | At the cost-optimal threshold chosen on validation | The decision, not the ranking |
| Cost with model | Unflagged fraud amount + alerts x review cost | One number that trades both errors |
| Cost, no model | Total fraud amount in test | The baseline every model must beat |
| Cost, amount rule | Same cost if only TRANSFERs above 200,000 are flagged (the `isFlaggedFraud` rule) | The simple rule a model has to justify itself against |
| Saving | Cost no model - cost with model | Upper bound: assumes a flagged fraud is stopped in full |
