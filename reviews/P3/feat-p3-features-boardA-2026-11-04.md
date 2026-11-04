# Board A — feat/p3-features

Reviewer: reviewer-ml. Date: 2026-11-04. Reviewed head: fbf60e5c833f2851286aa7b6521623db0c7e5170, diff main...feat/p3-features in its worktree. Verdict: **REVISE**. Blocking: **yes**.

Read reviewer-ml instructions, REVIEW_PROTOCOL, HYPOTHESIS, ROADMAP P3, DECISIONS D1, and TDD. No implementation changes made. CPU only; no GPU activity or real extraction run.

## Findings by severity

### F1 — High, blocking: trajectory ancestry is lost when intermediate edits lack targets

`src/wsp/predictor/baselines.py:282,289`; `src/wsp/predictor/splits.py:29`.
Splits are constructed after filtering to evaluated edits. Root traversal has only these rows, so a missing intermediate becomes an external root, disconnected from its actual ancestor. With full ancestry a -> intermediate -> z, and independent b,c, filtering out intermediate yields groups a, intermediate, b, c. Reproduced retained rows [a,z,b,c] with two folds: train [0,3], test [1,2], placing a and z across folds. This leaks cumulative edit information in exactly the incomplete-evaluation setting load_targets supports. Build connected components from the complete feature/zoo ancestry before filtering targets, then carry group IDs into folds. Existing tests check one-hop links and shared external parents, but never an omitted intermediate between retained ancestor and descendant. Add that regression and a multistep ancestry assertion across every fold.

### F2 — High, blocking: hybrid projections vanish from the SVD baseline

`src/wsp/features/pool.py:30`.
The regex recognizes q/k/v/o/gate/up/down/out_proj only. Installed Qwen3.5 implementation uses linear_attn.in_proj_qkv, in_proj_z, in_proj_b, in_proj_a; none matches. They enter global stats only, and enter no pooled spectrum at all (there is no global spectrum pool). Therefore an edit confined to these projections has an all-zero mlp_svd feature vector despite nonzero delta. out_proj is correctly mapped to o; feature_version 2 correctly records that change, but is insufficient for full hybrid coverage. Define explicit fused-QKV and gated-delta groups or an explicit catch-all spectrum group, document semantics, test all installed hybrid names, and bump feature_version when changing the representation.

Global-stats-only inventory under the current regex: in_proj_qkv/z/b/a; linear_attn.conv1d.weight, A_log, dt_bias, norm.weight; layer input/post-attention norms; embedding/final norm/lm_head and any other nonmatching names if edited. Per-tensor files retain these; pooled spectrum does not. The session note acknowledges in_proj_* global-only, but does not resolve the information loss.

### F3 — Medium, blocking: malformed metadata aborts the batch before failure isolation

`src/wsp/features/extract.py:345-351`.
edit0 and the revision comprehension parse edit.json outside _work's exception handler. Reproduced a first ready toy entry with edit.json containing `{`: run_extract raises JSONDecodeError and does not process the healthy second entry or write a failure manifest. Without an injected base, any malformed todo edit can trigger this in revision discovery. Validate metadata per entry, record its failure, and continue healthy compatible entries; retain deliberate global rejection for incompatible base revisions. Current corruption test only corrupts delta.safetensors, so misses this case.

### F4 — Medium, blocking: resume configuration omits a known numerical setting

`src/wsp/features/extract.py:323-329,365-377`.
threads affects LAPACK output by the implementation's own documented observation, but is outside cfg/config_hash and absent from the manifest. Resuming the same output with --threads 2 after --threads 1 silently skips old entries and permits new ones with different numerics under one hash. Include numerical execution settings in the resume identity (workers may remain excluded), reject a changed thread count, and test this. Base identity also needs binding to resume: a different --base-file with the same feature config currently reuses old completed entries; validate base provenance before skipping.

### F5 — Medium, nonblocking inherited divergence: feature/eval LoRA validation differs

`src/wsp/features/extract.py:124`; `src/wsp/zoo/apply.py:61`; `src/wsp/eval/zoo_compat.py:55-73`.
Supported vanilla LoRA arithmetic agrees exactly, but features use zoo.apply whereas eval uses zoo_compat. The former accepts fan_in_fan_out/bias/modules_to_save configurations and ignores non-factor adapter tensors that eval rejects. Reproduced fan_in_fan_out=True: feature densifier returns two tensors; eval raises NotImplementedError. The validation gap predates this refactor, so it is not a behavior regression in lora_factors. Prefer a common validated path or test and document the supported intersection before consuming external adapters.

## Checks and evidence

- `source scripts/env.sh && pytest -q`: **184 passed, 8 deselected, 6 warnings in 16.05s**. GPU/slow tiers intentionally excluded. Log: /tmp/p3-boardA-tests.log (local, not committed).
- D1: effective_delta uses fp32 W+d, casts to base dtype, subtracts fp32 W. Default base loader is bf16. Direct CPU comparison against eval.run.apply_delta on bf16 tensors gives identical effective tensors and all four shared rounding-stat fields. Aggregate loss fraction is weighted by nonzero entries; norms aggregate squared per-tensor norms. Injected --base-file follows its stored dtype rather than forcing bf16; callers must supply a bf16 base for the advertised D1 experiment.
- Supported toy LoRA: feature and eval densifiers are torch.equal for every tensor. Both compute `(B.float() @ A.float()) * scaling`. The apply.py refactor preserves this order, dtype, resolution, and scaling logic relative to main. No behavior change found for supported inputs. Auxiliary canonical tokens scale B before QR; those describe unrounded LoRA, while primary svd/stats describe effective deltas, as documented.
- Existing GL(r) canonical-token tests pass, including extraction-level refactorization; stats depend on the dense effective delta. Sign fixing uses the largest absolute U coordinate; projections use a local seeded generator. These are approximate floating-point invariances, not guarantees at bf16 rounding boundaries or for repeated singular values (basis rotations are not removed by sign fixing). No new exact universal GL(r) claim is justified.
- Worker claim independently exercised by the passing test_output_bytes_do_not_depend_on_worker_count: workers 1 vs 2, fixed threads, four toy entries, exact bytes for features.safetensors, pooled.safetensors and features.json. Real-zoo worker equality was not independently rerun; session evidence is author-reported only.
- Family/seed GroupKFold enforces the requested grouping. Trajectory links are correct when every intermediate is present, and siblings sharing a missing immediate parent stay together. F1 covers the missing transitive case. Separate family/seed splits do not claim simultaneous ancestry isolation.
- Baseline bootstrap resamples paired edits with replacement, calculates each metric per resample, and takes 2.5/97.5 percentiles. Spearman uses average ranks for ties and correlation of ranks; constant cases return NaN. Seeds reach LightGBM and MLP; input scaling and MLP target scaling are fitted within each outer train fold, with no outer-test target/feature leakage. CIs are edit-bootstrap intervals conditional on OOF predictions, not refitted-model or trajectory-cluster intervals.
- --delta-from-base subtracts matching base metrics before fitting. Missing base metric keys silently yield missing targets while stale target names remain; recommend rejecting/reporting absent keys and adding a CLI regression (not currently tested).
- Header rows lacking id are skipped; test passes. Rank/config changes reject resume. Entry delta corruption is recorded and healthy entries continue; CLI exits nonzero after processing failures. F3/F4 limit the stronger batch/resume claims. Failure IDs persist in manifest; detailed reasons are returned/logged.
- Diff contains source/tests plus board/session text only; no data files or apparent credentials committed. Worktree's untracked data was left untouched. P3 real baseline gate remains outstanding: branch session explicitly reports synthetic-only baseline results, not real performance evidence.

## Verdict

**REVISE — blocking F1–F4.** Repair ancestry grouping, hybrid pooling coverage, metadata failure isolation, and resume numerical/base identity; add targeted regressions and obtain a fresh Board A review before merge. No conclusion about H1 performance is supported yet.
