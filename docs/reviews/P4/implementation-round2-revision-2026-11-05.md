# P4 round two implementation revision

Both blocking reporting findings are fixed. Each metric containing ci95 carries
its own ci_method; the legacy top-level edit-bootstrap description is restored.
family_centered now centres family × outer test fold and averages finite fold
Spearman correlations using fold sample counts. Undefined fold correlations are
excluded from that average. family_centered_pooled retains the pooled estimand.

Regression tests before implementation: 2 failed (incorrect method label and
missing fold-centred implementation). After: 39 predictor tests passed.
Full default CPU suite: 492 passed, 1 skipped, 8 deselected. Ruff check passed.

## Compatibility evidence — FAILED, unresolved

Rerun from this worktree after sourcing scripts/env.sh:

    CUDA_VISIBLE_DEVICES="" python scripts/check_p4_compat.py

The script archives main into a temporary source directory and runs that source
and the current branch in isolated Python processes on identical synthetic
features/evaluations. Both runs use CPU M1, epochs=2, n_boot=20, seed=0,
n_splits=3, d_model=16, batch_size=4. Every existing dictionary field is compared
recursively, permitting new keys and treating NaN pairs as equal.
Output directories must be fresh when --out is supplied.

Observed output:

    Compared 15 pre-existing top-level keys; 106 differences.
    /targets/acc/mae/value: 0.35272759121532243 != 0.21635750758772096
    /targets/acc/rmse/value: 0.5251093758936585 != 0.2963913671636671
    /targets/acc/spearman/value: -0.06611969519707062 != -0.25835658678855367
    AssertionError: default compatibility failed

Full local artifacts: /tmp/wsp-p4-compat-73k0c_e3 (metrics, logs,
differences.txt). Main source revision: 3f52db0; branch predecessor: 8ca2e25.
The requested numerical equality does not hold. The branch's pre-existing
selection/refit changes remain; this revision does not revert those protocol
changes or conceal differences in the compatibility assertion.
