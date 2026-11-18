# Stage W2: pre-declared plan (18 Nov 2026; EXPLORATORY; transcribes RULING-wme-post-stagew Q4)

> **Status.** Development work, nothing registered or confirmatory. This file is committed **before any stage-W2 data
> exists**: no `wme-w2` panel has been built, no W2 adapter trained, no W2 generation made and no `wsp-wme-w2-*`
> instance launched. Code is committed before the data that tests it. Part A stays held; `data/evals`, sealed splits,
> `docs/PREREG_*` and A5/F12-pinned files are not touched. It transcribes Q4 **without loosening**. Where this plan
> adds a rule, the rule is stricter than or orthogonal to Q4, and it says so.

**Binding inputs.** `docs/council/2026-11-18/RULING-wme-post-stagew.md` (Q4 is the spec), `RULING-wme-stagew.md`
(stage-W plan: recipe, features, baselines, nulls, execution), `RULING-wme-post-pilot3.md` (P2, P5),
`docs/council/2026-11-17/SYNTHESIS-r11.md` (R4 trajectory claim and nulls, R5 state, R6 shrinkage, R9 execution),
`docs/reviews/wme/wme-stagew-2026-11-18.md` and `.json` (EXPLORATORY; the stage-W measurements and rates).

## Q4, transcribed

| Q4 text | Where this plan implements it |
|---|---|
| Nested world model: "pooled same-path-type curve + an edit-informed residual, shrunk toward 0" | §6 |
| "Write a pre-declared plan before any W2 data exists" | this file (committed before the panel, training and generation) |
| "≥ 40 fresh paths with fresh seeds, H1/H3-type, over training step × λ × layer mask" | §3–§4: 15 runs × 3 paths = 45; ≥ 40 complete paths is a pass condition (§7) |
| fresh panels disjoint from all exposures incl. `wme-p3*` and stage W | §2 |
| "Primary contrast: nested WM vs the pooled curve, few-shot k = 2 and zero-shot, leave-one-run-out" | §7 |
| "the same nulls as stage W" | §8: N1 shuffled edit features, N2 path-label permutation, N3 no-edit-information world |
| "Pass: LB of the improvement over the pooled curve > 0 in either setting, Holm-adjusted, every null at size ≤ 0.035" | §7–§8 |
| "Cap: ≤ 10 GPU-h, ≤ $60, instances named `wsp-wme-w2-*`" | §5 |
| also report vs persistence, LC-PFN, parametric curves, StatNN, static map, stage-W's unnested WM | §9 (descriptive) |
| validate the nested model on stage-W data only as a code check | §10 (non-evidential) |
| consequence of a pass / of a fail | §11 |

## 0. Done before this plan

- Stage W's panel `wme-w` is a registered all-split exposure in `wm.exposure.EXPOSURES`; `wm.sep27.REQUIRED_EXPOSURES`
  now lists it, and the tests map every stage-W item source (the `wme-w` dev panel; the canary, timing, teacher and
  R1 feature items, all `wme-p3h`) to an all-split registered exposure (commit f449692).

## 1. Target

Qwen/Qwen3.5-9B @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, bf16, text only, chat template with
`enable_thinking=False`; 32 layers; the residual writers `mlp.down_proj` and `linear_attn.out_proj`/`self_attn.o_proj`
(64 tensors). Generation budget B = 2,048 tokens for both benchmarks. All unchanged from stage W.

## 2. Fresh panel `wme-w2`

- **Salt** `wme-w2-panel-20261118`. Panels `mmlupro_gen` (MMLU-Pro @ b189ec7) and `supergpqa_gen` (m-a-p/SuperGPQA @
  4430d44, `SuperGPQA-all.jsonl`), loaded exactly as the stage-W / pilot-3 builders load them; one item per
  near-duplicate group.
- **Exclusion, asserted before anything is written.** Every exposure in `wm.exposure.EXPOSURES` at near-duplicate-group
  level (word-5-gram MinHash, Jaccard ≥ 0.7), from manifest hash lists only (sealed files are never opened). The
  builder refuses unless `wme-v0-p1`, `wme-p2`, `calib27-20261116`, `wme-p2-smoke`, `wme-p3s`, `wme-p3h` and
  **`wme-w`** are all registered (`wm.stagew2.REQUIRED_EXPOSURES`).
- **Split.** `dev` only, **N = 500 per benchmark** (as stage W), the first 500 groups in salted rank order.
- After the build, `wme-w2` (all splits) is registered in `wm.exposure.EXPOSURES` and in `sep27.REQUIRED_EXPOSURES`
  **before any W2 generation**, so any later panel (an H-T3 confirmatory panel, `wme-c2-gen`, 27B) excludes it. The
  report states the fresh groups left per benchmark.

## 3. Training runs (H1-type, fresh seeds)

- **Recipe (stage W's, frozen).** LoRA rank 8, α 16, on every layer's residual writers; AdamW, warm-up 10, batch 8 rows,
  2 epochs; loss on response tokens plus `<|im_end|>`; non-thinking template.
- **Data.** The 828 gold-verified pilot-3 teacher rows (`wme-p3h` construction items; exposed, never evaluated on).
  64 rows (32 per benchmark; seed `wme-w2:feature-items`) are held out of every run as R1 feature items; the other 764
  are dealt into **three disjoint shards T1, T2, T3** (seed `wme-w2:shards`; 255/255/254 rows, 64 steps per run).
- **Runs.** 15 runs r01…r15 = every (shard, learning rate) cell of {T1, T2, T3} × {5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4}
  (r_i = (T_{i mod 3}, lr_{i mod 5})); fresh seeds `sha256("wme-w2:lora:<run>")` (initialisation) and
  `sha256("wme-w2:lora-order:<run>")` (row order). Wider learning-rate and data variation than stage W is a declared
  design choice (more between-run variation for an edit-informed residual to explain); it is fixed here, before data.
- **H3-type paths are not run** (H3 failed, P2; as stage W). "H1/H3-type" is met by H1-type paths.

## 4. Paths and members

The stage-W grid thinned to the points j ∈ {1, 2, 3, 4, 6, 8} (6 points per path; x = j in stage-W units):

| Path type (per run) | Varies | Fixed |
|---|---|---|
| S (training step) | t = round(j/8 · steps) (8, 16, 24, 32, 48, 64 at 64 steps) | λ 1, all layers |
| L (dose) | λ = j/4 ∈ {0.25, 0.5, 0.75, 1, 1.5, 2} | final t, all layers |
| M (mask growth) | layers 0 … 4j−1 (fine groups g1 ∪ … ∪ gj), j ∈ {1, 2, 3, 4, 6, 8} | final t, λ 1 |

The point (final t, λ 1, all layers) is shared by the three paths: **16 unique members per run, 240 in total,
45 paths**. No merge paths (Q4 does not ask for them; they cost members). Thinning 8 → 6 points is a declared cost
choice: 15 runs × 22 members at 8 points would need ≈ 8.7 h of evaluation alone (stage-W mean 95 s per member),
above the cap with overheads.

**Instance slots.** Three slots of five runs each, a = r01–r05, b = r06–r10, c = r11–r15 (every slot has every
learning rate). **Queue per slot (frozen):** canary → base → base re-run (reversed item order; slot a only; null
discordance) → the slot's members **run-major** (r by r; S, then L, then M within a run) → canary repeat. Run-major
order means a deadline cuts at most the last run of a slot. A path enters the analysis only if all 6 points were
evaluated.

## 5. Platform, caps and execution (R9)

- **Lambda `gpu_1x_h100_sxm5` only** (no PCIe fallback); up to **3 instances in parallel**, one per slot, named
  `wsp-wme-w2-<slot>-20261118[-k]`. Only instances this track creates are launched or terminated.
- **Dead-man per slot.** Each slot has its own terminator state (`~/.local/state/wsp-wme-w2-<slot>`, outside the repo)
  with cumulative `t_max_hours` **3.33** and `max_usd` **19.90** (margin 300 s, idle limit 1,800 s, max price
  $5.50/h): the three slots together are capped at **9.99 instance-h and $59.70**, inside Q4's ≤ 10 GPU-h and ≤ $60.
  Instance-hours are counted from the launch request to the API-confirmed termination.
- **Progress-based liveness (R9.5).** The remote job appends a line to `out/w2/progress.jsonl` only when a unit of work
  completes (setup step, block of training steps, feature block, evaluated member). The AWS heartbeat resets the
  terminator's idle clock only if that file grew within the last 30 minutes; a hung job therefore ends the instance
  through the idle limit. On `DONE` or `FAILED` the driver pulls, verifies sha256 sums, requests termination and
  confirms `terminated` through the API.
- **In-job deadline.** The slot's queue stops at its instance's launch + **3.10 h** (no member is started that would
  end after it), leaving the canary repeat, the checksums and the sync before the terminator's 3.33 h cap.
- **Per-instance isolation.** Each instance builds its own uv venv (vLLM 0.34.0 and its torch build; driver ≥ 580 or
  the cuda-compat path); versions go to the ledger. Locally, per-track venvs (`scripts/venv_track.sh`:
  `wme-stagew2`; LC-PFN in its own Python-3.10 CPU venv `wme-stagew2-lc`) with `scripts/env.sh`.
- **Base canary per instance.** Each instance first generates the 64 exposed `wme-p3h` holdout canary items (as stage
  W) and records the sha256 of the token ids, and repeats it at the end; the within-instance bitwise repeat and the
  cross-instance equality are reported. Every instance also evaluates the base on the `wme-w2` panel; slot a's base
  defines the item cells and y0 of the analysis (if slot a has no base, the first of b, c that has one); the other
  bases are reported as a platform check.
- **Frozen numerics (R9.6).** vLLM in-process, prefix caching off, seed 0, `gpu_memory_utilization` 0.88,
  `max_num_seqs` 256, `max_model_len` 8,192, greedy, `max_new_tokens` 2,048; one `generate` call per benchmark per
  member with all 500 prompts in salted rank order; benchmark order MMLU-Pro then SuperGPQA; identical for every member
  and every instance.
- **Raw generations.** For every item: strict-parser correctness, flexible re-parse, truncation flag, length, the
  final 400 characters and the extracted answer (`evals.jsonl`), plus the **full generated text** of every item of
  every evaluation (`raw/<sha1(key)[:16]>.jsonl.gz`).
- **Checksums after the final write.** The remote job writes its final progress line, then `SHA256SUMS` of every file
  under `out/w2` as the last file written (stage W's ordering slip fixed), then `STATUS`. Locally every analysis
  artefact is checksummed after its final write (`manifest.json`).
- **Expected cost (estimate from stage-W measured rates, not a result).** Per slot ≈ boot 0.19 h + setup 0.16 h +
  training 5 × 64 steps × ≤ 2.84 s + features + canary/base ≈ 0.35 h + 80 members × 95 s ≈ 2.11 h → ≈ 2.8 h;
  three slots ≈ 8.5 instance-h ≈ $42 at $4.89/h.

## 6. The nested world model

For a held-out path p of type t ∈ {S, L, M} with points j = 1…6 and endpoints e ∈ {accuracy, truncation} × {MMLU-Pro,
SuperGPQA} (4):

- **Pooled curve (the comparator, exactly stage W's pooled baseline).** c_t(j) = mean over the training runs' paths of
  type t of y(j) (equivalently y0 + mean change).
- **Edit-informed residual.** On the training paths: inputs Δφ(m) = φ(m) − φ̄_t(j) (the stage-W member features φ, 7:
  R0 Frobenius norms of the realised delta per coarse layer group, R1 J1 projections of the exact θ₀ gradients of
  the stop, target and margin objectives; φ̄_t(j) = mean φ over the training paths of type t at point j), standardised
  by their root mean square over the training rows; targets r = y − c_t(j) (4 endpoints). Ridge map
  g(m) = Δφ̃(m)ᵀ B, α = 1.0 (stage W's static-map α), no intercept, shared by the three types.
- **Shrinkage toward 0.** Zero-shot (k = 0): ŷ(j) = c_t(j) + s·g(m_j). Few-shot (k = 2): the combined curve offset by
  its mean residual on the k given points, ŷ(j) = c_t(j) + s·g(m_j) + mean_{i ≤ k}[y(i) − c_t(i) − s·g(m_i)] (the
  pooled baseline uses the same offset with s = 0). s ∈ {0, 0.25, 0.5, 0.75, 1} is chosen **per setting**, by
  leave-one-run-out folds **inside the outer fold's training runs only** (inner pooled curve and ridge refitted on
  the inner training runs), minimising the mean path score; ties within 0.5 % go to the smaller s (toward the pooled
  curve). With s = 0 the model is the pooled curve, so it is nested.
- Nothing else is tuned. α, the feature set, the grid and the tie rule are fixed here.

## 7. Primary contrast and the pass rule

- **Cross-fitting.** Leave one run out: the pooled curve, the residual and s are fitted on the other runs' complete
  S/L/M paths; the held-out run's paths are predicted.
- **Settings.** zero-shot k = 0 (features only) and few-shot k = 2 (first 2 real points given). k = 4 is not run.
- **Path score.** E_A,p = mean over the points j = k+1…6 and the 4 endpoints of |ŷ − y| / SE, SE = √(max(y(1 − y),
  1/N)/N) of the measured point, N = 500 (stage W's score).
- **Improvement.** d_p = E_pooled,p − E_nested,p (positive favours the nested model); reported with
  C = 1 − mean E_nested / mean E_pooled.
- **Test (stricter than stage W).** The unit is the **run** (a run's three paths share their final point and their
  training): one-sided sign-flip of the run-mean improvements with run weight = its number of complete paths
  (statistic = mean d over paths), exact enumeration (≤ 16 runs; 10,000 seeded flips above that). The lower bound is
  obtained by inverting the same test (`wm.signflip.test`).
- **Holm over the two settings**, family α = 0.025 one-sided: the smaller p is tested at 0.0125, the other at 0.025 if
  the first is rejected. The Holm-adjusted lower bound of a setting is the test-inversion bound at its Holm level.
- **PASS** if and only if all of:
  1. ≥ 40 complete S/L/M paths entered the analysis;
  2. in at least one setting, Holm rejects **and** the Holm-adjusted lower bound of the mean improvement is > 0;
  3. every null (§8) has size ≤ 0.035 for this whole rule (conditions 2 applied to null data).

  Otherwise **FAIL**. (Q4's pass rule plus the minimum-path condition, which is stricter.)

## 8. Nulls (the same three as stage W)

- **N1 shuffled edit features:** φ trajectories permuted across the S/L/M paths (`wm.stagew.null_shuffle_features`).
- **N2 path-label permutation:** outcome curves reassigned among paths of the same type
  (`wm.stagew.null_permute_paths`).
- **N3 no-edit-information world:** each member's counts are drawn binomially per base item cell from its path type's
  pooled per-cell finish and correct|finish rates at its point (over all W2 paths of the type), plus a path random
  effect on each logit (variance τ² per benchmark by method of moments: variance of the per-path MLE offsets minus their
  mean sampling variance); features and the base are the real ones, so outcomes carry no edit information.
- **Size** = share of null replicates in which the whole pass rule (condition 2, the full leave-one-run-out pipeline
  with its inner shrinkage choice refitted) holds; **1,000 replicates per null** (seeded). Every null must have size
  ≤ 0.035; per-setting rejection rates and the 95 % Clopper–Pearson upper bounds are reported.

## 9. Secondary comparators (descriptive; reported whatever the result)

On the same complete paths, folds and path scores, with the stage-W implementations (`wm.stagew.analyse`, settings
k ∈ {0, 2}, inner leave-one-run-out folds for the R6 zero-shot shrinkage of the learned arms):

| Comparator | k = 0 | k = 2 |
|---|---|---|
| persistence (base at k = 0; last real point at k = 2) | ✓ | ✓ |
| 2-parameter power law (challenger) | — | ✓ |
| parametric learning-curve ensemble (lin, log2, exp2, pow2) | — | ✓ |
| LC-PFN (`lcpfn` 0.1.3, `EMSIZE512_NLAYERS12_NBUCKETS1000`, x = 12.5 j, base anchor at x = 0) | — | ✓ |
| StatNN (MLP 32 tanh, α 1e-2, 5 seeds) | ✓ | ✓ |
| static map (ridge α 1.0) | ✓ | ✓ |
| stage-W's unnested WM (the R5 per-item state, penalty 1.0, path offsets at k = 2) | ✓ | ✓ |

For each: mean E, C = 1 − E_nested/E_comparator, mean d (comparator − nested) overall and per path type, the
run-cluster sign-flip p and its lower bound at α = 0.025 (unadjusted; descriptive).

## 10. Code check on stage-W data (non-evidential)

Before any W2 data, the nested model, its nulls (200 replicates each) and the comparator merge are run on the stage-W
run (`data/wme/stagew/run-20261118`) thinned to the W2 grid (4 runs, 12 paths). With 4 runs the exact run-level
sign-flip cannot reach p < 1/16, so the check cannot "pass"; it only shows that the code runs end to end on real
data. It is labelled **non-evidential** wherever it appears. It may change the code only to fix a bug (logged);
it does not change any design choice above.

## 11. Consequences (Q4)

- **Pass.** The report gives a priced confirmatory **H-T3** design (nested WM vs pooled curve) on a separate sealed path
  set: fresh seeds (namespace `wme-ht3`), a fresh panel `wme-ht3` (salt fixed in the amendment; N = 500 per benchmark;
  disjoint from every exposure, `wme-w2` included), the W2 recipe, grid, settings and test. Size: the smallest number of
  runs R ∈ {15, 18, 21, 24, 27, 30} whose run-cluster bootstrap power is ≥ 0.8 for every passing setting, resampling W2's
  run-level improvement vectors (each run's paths as a unit) shifted to **half** the observed mean improvement (a
  winner's-curse guard), tested at α = 0.0125, 2,000 replicates. Price at W2's measured unit rates (member evaluation,
  training, per-instance overhead), $ = instance-h × $4.89 × 1.10. If no R ≤ 30 reaches 0.8, the report says "pass, but
  not powered for a confirmation at ≤ 30 runs" and prices nothing. H-T3 then enters PREREG v2 only by a dated
  amendment written before its paths exist (Q4).
- **Fail.** The paper reports (a) as a negative result: on this model, edit-specific trajectory prediction does not beat
  path-type averages. No further spending on (a) for the 9B model (Q4).

## 12. Stop rules

1. Each slot's queue stops at its launch + 3.10 h; the canary repeat, checksums and sync always run.
2. A failed bitwise restore of θ₀ in the engine, a non-finite training loss or a vLLM/HF writer mismatch stops that
   slot's job (`FAILED:<step>`); nothing after it is generated.
3. A canary repeat that is not bitwise identical, or cross-instance canaries that differ, are reported; the analysis
   proceeds.
4. **Relaunch.** At most one relaunch per slot after an infrastructure failure, inside that slot's cumulative cap, only
   if ≥ 1.0 h of it is left; it resumes from the synced `out/w2` (completed members are skipped by key; partial ones
   are discarded; a new canary is run first; the deadline is its launch + the slot's remaining hours − 0.23 h).
5. **Top-up.** If the three slots end with < 40 complete paths and ≥ 1.0 h and ≥ $5 of the total caps are left, one
   top-up instance `wsp-wme-w2-t-20261118` (own terminator; t_max = 9.99 h − hours used, max_usd = $59.70 − $ spent)
   resumes the slot with the most incomplete paths, as in rule 4.
6. If < 40 complete paths remain after rules 4–5, condition 1 of the pass rule fails: the attempt is reported as
   **FAIL** (a bounded attempt; Q4).

## 13. Outputs

`data/wme/stagew2/run-20261118/{a,b,c}/` (synced instance outputs, uncommitted) and `data/wme/stagew2/run-20261118/`
analysis files with `manifest.json` (seeds, config hash, git SHA, model revision, versions, sha256 of every artefact);
`docs/reviews/wme/wme-stagew2-2026-11-18.md` and `.json`, generated by `scripts/wme_w2_report.py`, marked
EXPLORATORY: per-setting errors and lower bounds for the nested model and every comparator, the nulls, pass/fail
under §7, the code check (non-evidential), the ledger (instance ids terminated, versions, instance-h, $), and, if
pass, the priced H-T3 design.

## 14. Pilot-informed choices and deviations from stage W

The H1 recipe and its teacher data; B = 2,048; N = 500 per benchmark; the stage-W features, score and comparator
settings. New in W2 and fixed here: 6-point grid; 15 runs over 3 shards × 5 learning rates; three parallel slots with
run-major queues; base re-run on slot a only; no merge paths; run-level clustering of the test; the nested model of §6;
1,000 replicates per null; the H-T3 sizing rule of §11. Stage-W outcomes informed the nested design (Q4); they never
test a W2 claim.
