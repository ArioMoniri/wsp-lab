# Board A re-review — feat/p3-features

Reviewer: reviewer-ml. Date: 2026-11-04. Head: 59db2188adce0765af9d8254fbe064cace1718a9. Reviewed `git diff fbf60e5...59db218` in the branch worktree. Verdict: **REVISE**. One remaining blocking finding.

Read reviewer-ml.md, REVIEW_PROTOCOL.md and the prior Board A review. No implementation changes made. CPU suite: `source scripts/env.sh && pytest -q`: **227 passed, 8 deselected, 6 warnings in 17.72s**. Independently reran the real-entry enumeration test: **1 passed**, no skip. No GPU or extraction jobs launched.

## Remaining blocker — F3 partially closed

**Medium, blocking:** `src/wsp/features/extract.py:485-487` constructs the revision set outside entry failure isolation, but `_check_edit` validates only that provenance is a dictionary, not the type of its consumed `resolved_base_revision` field. A malformed nested value still aborts the whole batch.

Independent temporary-toy reproduction: build_toy_zoo with toy_base_state; replace the first entry's provenance with `{"resolved_base_revision": []}`; call run_extract with injected base, ids[:2], n_layers=4. Result: `TypeError: unhashable type: 'list'`; healthy second entry has no features.json, and no output manifest exists. This is the same pre-worker failure-isolation class as original F3, despite the passing regressions for invalid JSON and top-level field types. Validate this consumed nested field per entry before revision discovery, record the entry failure and continue; add a regression asserting healthy output, persisted failure reason, and nonzero CLI exit. Do not merely catch the global set-construction error.

## Closure evidence for prior findings

- **F1 closed:** run_baselines reads full zoo manifest ancestry before filtering targets. `test_trajectory_folds_keep_chain_together_when_middle_entry_has_no_target` exercises the actual baseline path. `test_trajectory_groups_survive_removed_intermediate_entries` covers branching through two untargeted intermediates and every fold for 2/3/4 folds. Conflict and reachable-cycle rejection are tested. Root traversal keeps retained ancestors and descendants together.
- **F2 closed:** feature_version 3; 15 explicit types including catch-all; default dimensions 1098 stats / 120 spectrum. Hybrid projection/state, norms and embeddings receive per-type spectra. Tiny installed Qwen3.5 model enumeration and extraction tests pass. `test_real_abliteration_and_lora_changed_tensors_land_in_per_type_pools` passed independently against the available real entries without skipping.
- **F3 partial:** invalid JSON, invalid top-level edit fields, malformed manifest rows, healthy continuation, persisted failures, nonzero CLI exit and real build-header recognition all have passing regressions. Nested provenance reproduction above prevents closure.
- **F4 closed for supported bases:** identity contains full config, threads, torch/numpy versions and base commit or injected-state content hash; CLI base files bind filename plus SHA256. Parameterized tests reject config/rank/projection/seed/version/thread/base changes, test library versions and changed base content, and require actionable refusal messages. Per-entry identity prevents stale skipping when the run manifest is missing. Workers remain outside numerical identity as intended.

## Additional requested checks

LoRA factor validation now matches eval.zoo_compat for DoRA, orientation, bias, modules_to_save, lora_bias, unsupported tensors and unpaired factors; arithmetic remains float32 B @ A followed by scaling. New rejection tests cover orientation/bias/modules_to_save. Entry artifact dispatch remains a separate inherited difference: zoo.apply chooses dense when present, whereas eval enforces delta_format and rejects a stray dense file in a LoRA entry; this is not a new adapter-factor validation regression.

`--delta-from-base` now refuses any target missing a base metric instead of silently dropping it; the missing-ppl and complete-base regression passes.

Read `data/runs/p3/smoke2-w3.log` and `smoke2-w1.log`: both report 3 completed, zero failures (104.2s / 178.7s). Independently compared all nine files (features.safetensors, pooled.safetensors, features.json for 04ce1ebc4749f1c8, 8cd3182b6e55dcbf, b8430ae622b22440) under data/features/p3-smoke-w1 and p3-smoke-w3: all byte-identical. Toy worker equality regression also passes. Timing/log messages alone were not treated as proof of equality.

No data paths occur in the reviewed committed diff; branch worktree data remains untracked. Main was clean before adding this review. No new H1 performance conclusion: the real baseline evidence gate remains outside this correctness re-review.

**REVISE — remaining blocker: nested malformed provenance bypasses metadata failure isolation (F3).**
