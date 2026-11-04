# Board A: P4 round two

Reviewed `feat/p4-round2` at `8ca2e25` against the Board A recommendations and the frozen P4 protocol. **REVISE; do not merge.**

## Blocking findings

1. `src/wsp/predictor/train.py:475-476`: the pre-existing top-level `ci_method` key now describes paired cluster intervals, but the existing `targets` and `statnn` marginal intervals still bootstrap edits. Existing consumers reading `ci_method` alongside those fields will misreport their uncertainty. Preserve the existing key's meaning and label the paired method separately; test the association between each reported interval and its method.
2. `src/wsp/predictor/audit.py:139-158`: family centring is performed after pooling all outer-fold predictions, by family alone. When a family appears in several test folds this subtracts cross-fold rather than fold-specific means, so fold calibration offsets can influence the reported within-family correlation. Report the requested within-outer-test-fold centred correlations, or explicitly provide and label both estimands.

## Evidence and limits

Inner selection uses all surviving inner folds, with a fresh fixed-budget outer-train fit and train-context normalizers. The splits module is unchanged, though data-level split handling has changed; a complete independent leakage audit remains necessary. `tests/test_p4_predictors.py:558` checks device identity does not enter collated model inputs. Delta-norm uses training-only isotonic regression on effective Frobenius norm, with monotonic direction chosen on training labels. The paired bootstrap samples ancestry clusters with replacement and carries their paired rows; the paired method description says so. Per-family metrics are computed inside each test fold. The M2 matched model applies the same model to singleton sequences; a CPU instantiation at width 32 and pooled dimension 64 has 42,117 parameters for each model.

Focused CPU predictor tests: 38 passed. Real-data two-epoch CPU M1 runs on main and the branch were started under `boardA-p4main` and `boardA-p4r2`; both completed. The branch retains all main top-level metrics keys and adds baselines, paired_differences, per_family, family_centered, out_of_support and marginal_ci_method; all five audit structures were populated. The semantic change to ci_method remains blocking. A full default-compatibility assessment remains outstanding. The branch deliberately changes selection/refit and constant-coordinate handling, so existing numerical results cannot be assumed unchanged. This review does not establish that all leakage paths or all requested protocol details are correct; re-review after the blocking fixes and completion of those checks.
