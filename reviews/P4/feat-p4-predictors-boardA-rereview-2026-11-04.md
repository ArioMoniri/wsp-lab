# Board A re-review: feat/p4-predictors 6674f10

Verdict: REVISE. Blocking: yes (1 item, B1). Blockers 1–3 from the 8559479 review are closed.

Reviewer: reviewer-ml (Board A), new pass against 8559479. Worktree `.worktrees/feat/p4-predictors` at 6674f10.
Scope: predictor files in `git diff 8559479...6674f10` (`src/wsp/predictor/{data,train,splits,baselines,selftest,ssm_torch}.py`, `models/__init__.py`, `tests/test_p4_predictors.py`, `docs/MODEL_CANDIDATES.md` §P4 protocol, `docs/DECISIONS.md` D5). The range also carries main's merged H4 reader work, which is out of scope here. No code was edited. No GPU jobs were run.

## What I ran

- `source scripts/env.sh && pytest -q`: **319 passed, 8 deselected** (20 s). Matches the session note.
- `python -m wsp.predictor.train --model m2 --selftest`: exit 0, backend `loop`, 3 folds.
- `python -m wsp.predictor.train --model m5 --selftest`: exit 0, backend `torch_attention`, 3 folds.
- Real-data refusal (`--model m1 --features data/features/qwen3.5-0.8b --evals data/evals/qwen3.5-0.8b --split family --out <tmp>`): exit 1, `zero entries have targets with required predecessor targets; missing_predecessor_targets=0`, no output directory created.
- Independent purge audit on `data/zoo/qwen3.5-0.8b/manifest.jsonl` (scripts in `/tmp/reviewer-ml/`, not in the repo). The manifest is still growing during this review (1242 → 1243 lines), so all counts are a snapshot: 1235 unique ids plus 8 header rows with no id; no duplicate ids; no dangling `parent_id`.
- Parameter counts built on CPU with the real dimensions (token 1032, layers 25, stats 1098).

## Status of the blockers from 8559479

1. **Relations (closed).** Successor and source edges come from the manifest plus `edit.json` (`data.py:read_relations`). Only `chain` parent links are successors (505 edges: 333 sequential, 172 `lora_vs_base`). All 80 `task_vector` parents are sources. `FeatureDataset.chain()` walks `self.successors`, and M3 adjacency pairs come only from those chain batches, so task vectors never become history or transition pairs. `test_relations_sources_and_flagged_lora` covers the source, the length-1 chain, and the side-info stats.
   - `lora_vs_base` is defensible. The builder (`src/wsp/zoo/build.py:715-735`) adds a base-relative LoRA increment onto the accumulated state, so θ_t − θ_{t−1} is exactly that increment. "Successor" means the snapshot follows its predecessor, not that the LoRA was optimised against it. This is documented in `MODEL_CANDIDATES.md` §P4 and in the `read_relations` docstring. H2 results should be stratified by `step_semantics` (see N2).
2. **Ancestry-clean holdouts (closed for the purge itself; see B1 for the fold design).** Reproduced below: zero violations for all three holdouts. Purge runs before `Normalizer` and M3 pretraining, and the StatNN baseline uses the same purged folds (`train.py:289`, `:304`).
3. **Per-step Δy target (closed).** Default `--target delta`, predecessor taken from `successors` or `base`. Entries with a missing predecessor target are excluded and counted (`missing_predecessor_targets` in `metrics.json` and `manifest.json`). `--target level` is explicit. Bounds are checked before subtraction. Heads are linear in both modes. `test_step_delta_level_and_missing_predecessor` covers it.

## Remaining items you asked about

- **M5 spec (closed).** One tensor attention layer plus one causal edit layer gives two attention layers in total, dropout 0.2 on both (`models/__init__.py` `TensorEncoder`, `transformer(d, 1, 0.2)`). Positions appear only in M5.
- **m1_matched control (closed).** Built both models: `m1_matched − m5 = 4` parameters (2,148,364 − 2,148,360), dropout 0.2 on both. The test `test_m5_layers_dropout_matched_control_and_descriptor` asserts the same 4-parameter gap.
- **M6 collapse metrics (implemented; acceptance see B1).** ERank is computed on unmasked training representations, with a same-initial-encoder ablation using a frozen random linear predictor (documented). Collapse threshold ERank < 2, per-fold gate ERank ≥ 2× ablation, zero collapsed fold seeds.
- **Descriptor cleanup (closed).** `length`, `inner_seed` and `seed` are excluded (`data.py:descriptor`), with a test. See N4 for the remaining chain-parameter issue.
- **`--selftest` (closed).** Writes under `/tmp/wsp-p4-synthetic-*`, overrides `out` so it never reaches a results path, and marks `synthetic: true` in config and manifests. An explicit `mamba_ssm` request raises `ImportError` when unavailable, so there is no silent fallback.

## Blocking

**B1. The H3 acceptance gate can pass on a non-LoRA fold, and the LoRA holdout is not a named fold.**

- `src/wsp/predictor/train.py:316-331`: `transfer_pass |= count >= 6` iterates over every outer fold. `m6_acceptance.accepted` then requires only `family_split`, the collapse gate, and that `transfer_pass` is true. Any fold can set `h3_transfer_gate` true, including the fold that holds out `chain`, `abliteration` or `random_dir`.
- `splits.py:make_splits` runs `GroupKFold` on top-level `family`, so the fold composition is decided by group sizes, not by design. I simulated it on the real manifest (all rows treated as labeled):
  - `n_splits=5`: folds `{chain}`, `{abliteration}`, `{random_dir}`, `{lora_sft}`, `{task_vector, heretic_ara}`. LoRA is isolated only by balancing on today's counts.
  - `n_splits=3`: folds `{chain}`, `{abliteration, task_vector}`, `{heretic_ara, lora_sft, random_dir}`. The LoRA test rows share a fold with random_dir, which then leaves the training set. That contradicts H3 ("trained on abliteration + random-direction edits").
- Failure scenario: a `--split family` run where the chain-held-out fold reaches ρ ≥ 0.5 on six benchmarks writes `m6_acceptance.accepted: true` and `h3_transfer_gate: true`, and H3 is reported as passing for a holdout that never contained LoRA.
- The `MODEL_CANDIDATES.md` §P4 text says "for one held-out family". The code follows that wording, but H3 is LoRA-specific, so the wording needs tightening too.
- **Fix:** add an explicit `--holdout <family>` (or a leave-one-family-out loop that names each test family). Set the H3 transfer gate on the fold whose test families are exactly `{lora_sft}`, and let `accepted` depend only on that fold. Write the held-out family names into `metrics.json`. Add tests that (a) a 3-fold family split does not pool `lora_sft` with another family for the H3 claim, and (b) a non-LoRA fold passing six benchmarks leaves `accepted` false.

## Non-blocking

- **N1. Holdout cost.** See the section below.
- **N2. H2 stratification.** `lora_vs_base` is 172 of 505 successor edges (about 34 %) and 225 of 675 chain entries. Report H2 separately for `sequential` and `lora_vs_base` steps, since the second is additive composition rather than sequential optimisation.
- **N3. Stale spec.** `MODEL_CANDIDATES.md` line ~413 still says M5 uses a "sigmoid head, as in TrajectoryPredictor". The §P4 protocol and the code use linear heads. Fix the older text.
- **N4. Chain descriptor is inconsistent.** `inner_params` is a dict, so `descriptor()` hashes the whole string as one categorical bucket. The inner edit's numeric hyperparameters (for example abliteration `max_weight`) reach chain rows only as a per-edit hash, while standalone rows get log-scaled numbers. This does not leak, but chain rows do not describe their inner edit. Flatten `inner_params` into prefixed numeric keys.
- **N5. Split audit mixes outer and inner plans.** `folds()` appends an audit entry on every call, and `plans` calls it with `min(3, n_splits)`. Label the entries outer or inner so the manifest's audit can be read as a per-fold record.
- **N6. Legacy chain metadata.** 20 chain rows lack `step_semantics`; 16 of them have a parent and are classed as successors by the family fallback. The manifest records the missing flag, as the protocol says. Confirm these are early pipeline entries before reporting H2 on them.
- **N7. Snapshot drift.** The manifest grew during this review. Re-run the purge audit on a frozen manifest before results. The session note's 1219-row figures are older.
- **N8. CUDA and Mamba paths remain unexercised.** The explicit `mamba_ssm` backend and `selective_scan` have no GPU test. The protocol already gates CUDA on `p1-zoo-full` finishing. Keep that gate.
- **N9. Selftest banner.** The synthetic label is in config, manifests and the log, but not in `results.md`. Add a header line.
- **N10. Bootstrap CIs resample edits, not trajectories.** Already disclosed in `results.md`. For chain-containing folds the CIs are anti-conservative.

## Holdout cost (reproduced)

Purge audit on the real manifest (candidate training = all rows outside the held family, as in the session note):

| held out | test | candidate train | dropped | kept | violations |
|---|---:|---:|---:|---:|---:|
| lora_sft | 121 | 1114 | 470 (42 %) | 644 | 0 |
| abliteration | 205 | 1030 | 454 (44 %) | 576 | 0 |
| random_dir | 153 | 1082 | 479 (44 %) | 603 | 0 |

Violations were counted by an independent walk of all parent edges (successor and source), checking each kept row's ancestry for the held family or `inner_family`. The count is zero for all three.

Composition of the `lora_sft` holdout's training set: all 205 abliteration and all 153 random_dir standalone rows, 233 chain prefixes that come before the first LoRA step, 52 task vectors, 1 heretic. Every dropped row is a chain step or task-vector child downstream of a held-out step. The purge therefore removes chain history specifically, which is the signal H2 uses. It does not remove any training family.

**Is H3 still testable?** Yes, with two caveats.

- The test fold is intact: 121 standalone LoRA rows. The Spearman standard error at n = 121 is about 0.07, so the ρ ≥ 0.5 threshold is resolvable, but differences smaller than about 0.13 are not.
- The claim is only valid on the fold B1 identifies. Any H3 number must come from the `lora_sft`-only holdout.

**Should a chain-free variant be added?** Yes, as a non-blocking sensitivity arm. Train only on chain-free rows (single-step edits and task vectors whose parent is not a chain). That gives 411 rows for the `lora_sft` holdout, 324 for `abliteration`, and 386 for `random_dir`. Report it next to the purge set. It answers whether the 233 chain-prefix rows help H3, and it is the cheapest check on whether the purge cost is driving any result.

## Verified, no issue

- Chain structure: 154 chain roots (step 0, no parent), 521 non-root chain entries with a parent, no step > 0 root, no dangling parents, no duplicate ids.
- Successor-only chaining and M3 pairs (see Relations above).
- Targets: missing targets stay NaN and are masked; no target is fabricated; PPL and KL are refused by the [0,1] bound check.
- Baselines: StatNN is fit on the same purged `baseline_folds` as the neural head.
- Stop-gradient on the M3/M6 alignment target (`test_jepa_alignment_target_is_detached`) and `selective_scan` chunk equivalence (`test_chunked_scan_matches_loop_and_gradients`) now have tests. Mixed-target refusal has a test.
- Early stopping uses only inner validation of training edits. The test fold is predicted once after the best state is restored.

## Required before re-review

Fix B1 with the tests named there. Re-run `pytest -q` and the real-data refusal. Re-review as a new Board A pass, not as an edit to this one. N1–N10 are not required for this pass, but N3 and N5 are cheap and should go in with the B1 fix.
