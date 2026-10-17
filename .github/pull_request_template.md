## What changed and why

<!-- One paragraph. The decision or metric this affects, not the diff. -->

## Leakage and split

- [ ] No new feature uses a column listed in `features.LEAKY`
- [ ] A new history feature has a test showing earlier rows do not change when later rows are removed
- [ ] A new label-derived feature honours the confirmation delay and has a boundary test
- [ ] Any model change reports the shuffled-label control and the out-of-time split, not a random split

## Evidence

- [ ] `make verify` passes locally
- [ ] New logic has a test that fails without the change
- [ ] If numbers in the README moved: `make run readme` re-run and committed

## Data

- [ ] Provenance flag on any new result is correct (`synthetic` or `real`)
