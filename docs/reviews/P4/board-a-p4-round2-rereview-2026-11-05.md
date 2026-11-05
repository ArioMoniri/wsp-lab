# Board A: P4 round two re-review

Reviewed `feat/p4-round2` at `713f3e3` against the prior review and implementation note. **REVISE; do not merge.** The archived comparison uses main `3f52db0`, not the current main head; its 106 entries are individual leaf or list-length differences, not 106 independent experiments. The fixture is synthetic, so these are compatibility diagnostics, not research results. The two logs report validation MSE for main and refit training MSE for the branch; these are different quantities and do not establish comparable performance.

## Compatibility classification

The following groups exhaust the paths in `/tmp/wsp-p4-compat-73k0c_e3/differences.txt`. Counts count each reported line once. Class (a) is conditional on the required DECISIONS addendum; (b) blocks; (c) is unobserved nondeterminism, not simply a deliberate seed change.

| Paths | Count | Class and cause |
|---|---:|---|
| `/targets/acc/{mae,rmse,spearman}/{value,ci95/*}` | 9 | (a) Predictions change with all-inner-fold epoch selection, fresh outer-train refit, and zeroing train-constant coordinates. Edit bootstrap applied to changed predictions changes intervals. |
| `/h2_by_step_semantics/sequential/acc/{mae,rmse,spearman}/{value,ci95/*}` | 9 | (a) Same predictions and bootstrap, restricted to this stratum; in this fixture the stratum covers the same rows. |
| `/length_ge_4/targets/acc/{mae,rmse}/{value,ci95/*}` | 6 | (a) Same changed predictions and bootstrap on the long-chain subset. |
| `/statnn/{model,n,n_folds,n_features}` | 4 | (b) Keys are **missing**, not changed numerical counts. `fit_eval` used to emit model, total input rows, number of folds and input feature width; `audit_run` constructs only `targets`. No evidence of a changed dataset row filter is needed to explain their disappearance. Restore the legacy fields and semantics, with separate names for any new counts. |
| `/statnn/targets/acc/{mae,rmse,spearman}/{value,ci95/*}` | 9 | (b), mixed causes: baseline training expands from the first inner-training subset to outer training, an intended matched-refit change; however `fit_eval` used the same `seed` in each fold and `audit_run` uses `seed + f` with stochastic LightGBM subsampling. That separate, undocumented seed schedule is not required by the recommendation. Restore the seed rule and rerun to isolate the intended contribution; do not attribute all nine changes solely to refitting. The new fallback for fewer than two finite training labels also differs and needs explicit treatment. |
| `/folds/{0,1,2}/train_ids` length and indexed entries | 27 | (a) Legacy fields now describe the full outer-train refit, rather than the first inner-training subset; index differences follow the changed membership/order. |
| `/folds/{0,1,2}/validation_ids` length | 3 | (a) Refit has no validation set; inner validation IDs are recorded under `inner_selection`. |
| `/folds/{0,1,2}/normalization_context_ids` length and indexed entries | 27 | (a) Normalizer now fits outer-train ancestry contexts for the final refit instead of first inner-train contexts. |
| `/folds/{0,1,2}/history/{0,1}/train_loss` | 6 | (a) Different training rows, normalizer, epoch budget and resulting fit; these are refit training losses. |
| `/folds/{0,1,2}/history/{0,1}/validation_loss` | 6 | (a) Final refit has no validation loss, so it is null; inner histories retain validation losses. |

Totals: (a) 93, (b) 13, (c) 0 observed. This classification does not prove bitwise repeatability: no repeated same-revision run was performed. Seed/order nondeterminism remains to be ruled out by repeat runs after the baseline fix. The dataset's `clean_training` also adds full-ancestry cluster exclusion for family/seed holdouts; it is a deliberate leakage guard, not evidence that this trajectory-split fixture lost rows. Document that change and audit its effect on affected splits.

## Previous blockers and leakage

The CI association blocker is closed: `metrics_with_ci` attaches the edit method to each marginal interval, `paired_cluster_difference` attaches the cluster method to paired intervals, and the recursive saved-output test checks each `ci95` against its method. The legacy top-level label is restored. The centring blocker is closed: `family_centered_correlations` centres within family and outer test fold, reports fold-n weighted finite fold correlations, and separately labels `family_centered_pooled`. The fold-offset regression test verifies the distinction. Focused CPU tests: 2 passed, 37 deselected.

Selection fits a fresh normalizer on each inner train, selects epochs without outer-test labels, then refits a fresh model and normalizer on outer train; controls use training labels. This inspection found no direct test-label use in these paths, but it is not a complete independent leakage audit of all split modes, feature provenance and ancestry context. Do not infer a global no-leakage guarantee from this review.

## Required before merge

1. Restore the four legacy StatNN metadata keys with their prior definitions. Preserve the original fold seed schedule or explicitly justify and approve a new protocol separately; resolve the small-training-set fallback semantics. Re-run the compatibility comparison and classify residual differences, including deterministic repeat runs.
2. Add a P4 addendum to `docs/DECISIONS.md` giving old → new rules: first surviving inner split and its trained model → all surviving inner folds select the median best epoch and a fresh fixed-budget outer-train refit; inner-train normalizer → separately fitted inner normalizers and an outer-train refit normalizer, with train-constant coordinates zeroed and optional robust scaling; baseline training follows the refit rows; legacy fold fields now describe refit and validation details move to inner selection; full-ancestry cluster exclusion in affected holdouts; marginal edit versus paired ancestry-cluster bootstrap; within-fold versus pooled family centring. State that these rules change numerical results and define any deliberately retained baseline seed or fallback change.
3. Complete leakage checks across split modes, ancestry-derived contexts and feature provenance. Merge is appropriate only after the actual residual differences are acceptable and documented and those checks find no leakage.
