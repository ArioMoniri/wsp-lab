# P4 round two implementation revision — 2026-11-05

Board A's re-review at 713f3e3 was read in full from the canonical checkout
(the review was absent from this branch). Restored StatNN model, n, n_folds,
n_features with fit_eval definitions: model name, total input rows, supplied
fold count, input feature width. Added skipped_folds separately. Baselines
use the same seed in every fold again. Zero/one finite training label explicitly
skips the target/fold, logs why, and records skipped_reason; no mean substitutes.
The same skip reporting is now available in fit_eval.

Python/NumPy/torch seeds are set by seed_all; torch deterministic algorithms
and deterministic cuDNN flags are explicit. LightGBM already sets random_state,
deterministic=True, force_row_wise=True and n_jobs=2; retained these settings.

## Test-first evidence

The new parametrized audit contract failed in all three cases before changes:
missing metadata and unsupported-fold behavior. After implementation all three
passed (39 deselected). Cases cover zero, one and three finite labels, verify
identical seed 17 across folds, and ensure skipped predictions remain NaN.
Predictor suite including ancestry-context checks: 45 passed.
Full CPU suite and ruff results are recorded below after completion.

## Compatibility residuals

Command (after sourcing scripts/env.sh, CPU only):

    CUDA_VISIBLE_DEVICES="" python scripts/check_p4_compat.py --repeat-self --out /tmp/p4-revision-compat-direct

Main at comparison: 694625922eb4c3931c475d28a659e57e23f79e14.
Settings: M1, epochs=2, n_boot=20, seed=0, n_splits=3, d_model=16,
batch_size=4, identical synthetic data. No performance claim follows from this
fixture. Artifacts contain both metrics, repeat metrics, logs and all leaf/list
differences. The strict compatibility assertion still exits nonzero for the
intentional protocol differences; it has not been weakened.

| Residual paths | Count | Class and responsible protocol change |
|---|---:|---|
| targets/acc metrics and CI endpoints | 9 | (a) All-inner-fold median epoch selection, fresh outer-train refit, train-constant coordinates zeroed; edit bootstrap on changed predictions. |
| h2_by_step_semantics/sequential/acc metrics and CI endpoints | 9 | (a) Same changed predictions/bootstrap, restricted to the sequential stratum. |
| length_ge_4/targets/acc MAE/RMSE and CI endpoints | 6 | (a) Same prediction/bootstrap changes on long chains. |
| statnn/targets/acc metrics and CI endpoints | 9 | (a) Baseline now fits full outer-train rows instead of first inner-train subset; original same-seed-per-fold schedule restored, no unsupported folds in this fixture. |
| folds/0,1,2/train_ids lengths and indexed entries | 27 | (a) Legacy fields describe full outer-train refit membership/order. |
| folds/0,1,2/validation_ids lengths | 3 | (a) Refit has no validation set; IDs move to inner_selection. |
| folds/0,1,2/normalization_context_ids lengths and entries | 27 | (a) Outer-train ancestry contexts replace first inner-train contexts for refit. |
| folds/0,1,2/history/0,1/train_loss | 6 | (a) Fresh refit uses different rows, fitted normalizer and selected epoch budget. |
| folds/0,1,2/history/0,1/validation_loss | 6 | (a) Null for fixed-budget refit; inner validation histories remain in inner_selection. |

Total: **102 class (a), 0 class (b), 0 class (c) observed**.
Four missing metadata differences are eliminated. The nine baseline differences
are now isolated to intended matched-refit training, not an extra seed schedule.
The DECISIONS addendum records all old → new rules requested by Board A.

--repeat-self runs the branch twice in separate processes with identical inputs,
arguments, seed and thread settings, then compares metrics.json bytes (including
NaNs and key ordering). Result: **bitwise identical**, repeat-self.json true.
This establishes repeatability for this CPU fixture, not all devices/configs.

## Leakage checks and limits

Read train.py, audit.py, baselines.py and data.py. Selection fits each normalizer
only on inner-train ancestry contexts; final refit fits outer-train contexts.
Controls fit on finite training labels only; device means use training categories,
and isotonic direction uses training labels. Inputs are extracted safetensor
stats/SVD, hashed edit descriptors (excluding seed/identity/labels), and source
stats for source relations; target rates and runtime device IDs do not enter
predictor features. Zoo snapshot and extraction provenance remain recorded.

New parametrized family/seed/trajectory tests check inner-training context
disjointness from validation and outer-test ancestry contexts and outer-training
context disjointness from test contexts. Existing tests additionally cover
mixed-family unlabeled ancestors, source/successor semantics, named purge and
chain-free holdouts, matched references, future-edit masking, constant handling,
test-independent preprocessing and device exclusion. These checks found no
leakage in the covered paths. Full-ancestry cluster exclusion can reduce
family/seed/holdout training support; the trajectory compatibility fixture
does not exercise that numerical effect. Real extraction provenance and every
possible zoo topology are outside this synthetic verification.

Final validation: default CPU suite **495 passed, 1 skipped, 8 deselected**
(91.84 s; collected before the three new split-context cases were added).
The subsequent complete predictor suite **45 passed** includes all three new
split-context cases. Ruff check passed. No class-(b) or observed nondeterminism
remains; numerical equality with main remains intentionally false as documented.
