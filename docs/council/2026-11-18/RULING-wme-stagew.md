# Stage W: pre-declared plan (18 Nov 2026; EXPLORATORY; transcribes SYNTHESIS-r11 R4–R7, R9 and RULING-wme-post-pilot3 P5)

> **Status.** Development work, nothing registered or confirmatory. This file is committed **before any stage-W data
> exists**: no `wme-w` panel has been built, no stage-W adapter trained, no stage-W generation made and no stage-W
> instance launched. Code is committed before the data that tests it. Part A stays held; `data/evals`, sealed
> splits, `docs/PREREG_*` and A5/F12-pinned files are not touched. It transcribes R4 (H-T2), R5 (state), R6 (flag 1),
> R7 (exposure), R9 (execution) and P5 **without loosening**. Where this plan adds a rule, the rule is stricter
> than or orthogonal to the binding text, and it says so.

**Binding inputs.** `docs/council/2026-11-18/RULING-wme-post-pilot3.md` (P1–P5), `docs/council/2026-11-17/SYNTHESIS-r11.md`
(R1–R10), the round-11 seat reports, `docs/reviews/wme/wme-pilot3-2026-11-17.md` (EXPLORATORY) and
`docs/council/2026-11-17/RULING-wme-pilot3.md` (pilot-3 recipe and scoring), `RULING-wme-v2b.md` (A8 caps, E2 design).

**Purpose (P5).** (1) Fresh H1-type training paths with fresh seeds over checkpoints × λ × layer-group masks (and
merges if cheap), each member measured for accuracy **and** truncation on a fresh dev panel. (2) Fit the R5
budget-aware state against the R4 baselines and the three nulls, few-shot and zero-shot. (3) Simulate power for H-T2
(registration needs power ≥ 0.8 with ≤ 40 paths) and for the E2 co-primary "WM vs direct multi-fidelity best-of-N at
total-compute parity" on the sycophancy library (co-primary only if power ≥ 0.8). (4) Price the confirmatory design.

## 0. Done before this plan (R7)

- `wme-p3s` and `wme-p3h` are registered in `wm.exposure.EXPOSURES` with **all splits** (the holdout split, the
  unscored rest of every dev pool and the construction items behind the teacher/STaR/RL outputs included), with
  tests. `wm.sep27.REQUIRED_EXPOSURES` lists them too, so the 27B panel cannot be built without them.

## 1. Target, platform and caps

- **Target.** Qwen/Qwen3.5-9B @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, bf16, text only, chat template with
  `enable_thinking=False`. 32 layers; residual writers `mlp.down_proj` and `linear_attn.out_proj`/`self_attn.o_proj`
  (64 tensors), as pilot 3.
- **Platform.** Lambda `gpu_1x_h100_sxm5` only (no PCIe fallback, so the platform matches the confirmatory runs;
  wait for capacity). Instances are named `wsp-wme-w-<phase>-20261118[-k]`; only instances this track creates are
  launched or terminated. Every instance is launched and ended through the dead-man terminator
  (`scripts/lambda_deadman.py`) with its own state directory (`~/.local/state/wsp-wme-w`, outside the repo).
- **Liveness is progress-based (R9.5, R3(d), pro-engineer §3.2).** The remote job appends a line to
  `out/w/progress.jsonl` only when a unit of work completes (a setup step, a block of training steps, a feature
  block, an evaluated member). The AWS-side heartbeat resets the terminator's idle clock **only** if that file grew
  within the last 30 minutes. A hung process therefore stops the heartbeat and the terminator's idle limit
  (1,800 s) ends the instance even if nobody is watching. On `DONE` the driver syncs, verifies sha256 sums, requests
  termination and confirms `terminated` through the API.
- **Per-instance isolation (R9.5).** Each instance builds its own uv venv (vLLM 0.34.0 and its torch build; the
  driver ≥ 580 or the cuda-compat path of pilot 3); versions go to the run ledger. Locally, a per-track venv
  (`scripts/venv_track.sh`; never the shared `.venv`) with `scripts/env.sh` for CUDA isolation. LC-PFN runs in its
  own local CPU venv (it needs Python 3.10 and torch ≤ 1.11).
- **Base canary (R9.5).** Each instance first generates 64 base items (exposed `wme-p3h` holdout MMLU-Pro items, so
  no fresh item is spent) and records the sha256 of the token ids; the canary is repeated at the end of the
  evaluation queue and the bitwise match is reported. All stage-W points are generated on one instance, so every
  pair shares a platform.
- **Frozen numerics (R9.6).** vLLM in-process, prefix caching off, `seed` 0, `gpu_memory_utilization` 0.88,
  `max_num_seqs` 256, `max_model_len` 8,192, greedy, `max_new_tokens` = B = 2,048 for both benchmarks (the pilot-3
  frozen budget). One `generate` call per benchmark per member with all N prompts in salted rank order; benchmark
  order MMLU-Pro then SuperGPQA. The engine configuration, item order and batch are identical for every member.
- **Raw generations.** For every item: correctness (strict pilot-3 parser), the flexible re-parse, truncation flag,
  response length, the final 400 characters and the extracted answer.

| Cap (all binding) | Value |
|---|---|
| Stage W | ≤ 4.0 H100 instance-hours, everything from launch to confirmed termination except the timing-pilot segment (boot, install, download, training, features, evaluation, sync) |
| Timing pilot (P-9B-eval) | ≤ 0.5 instance-hours on the same instance after stage W, and ≤ $11 |
| P-27B-gen | **not run** in stage W (needs a 2× H100 worker; 27B starts only after the 9B result is sealed, R9.6). Its share of the $11 is unused |
| Total stage-W Lambda | ≤ $35 including any failed launch (terminator `max_usd` 30, cumulative `t_max_hours` 4.7) |

## 2. Fresh panel `wme-w` (R7)

- **Salt** `wme-w-panel-20261118`. Panels `mmlupro_gen` (MMLU-Pro @ b189ec7) and `supergpqa_gen` (m-a-p/SuperGPQA @
  4430d44, `SuperGPQA-all.jsonl`), loaded exactly as the pilot-3 builder loads them; one item per near-duplicate group.
- **Exclusion, asserted before anything is written.** Every exposure in `wm.exposure.EXPOSURES` (pilot 1, pilot 2
  with sealed hash lists, the 27B calibration, the smoke runs, **`wme-p3s` and `wme-p3h`, all splits**) at
  near-duplicate-group level (word-5-gram MinHash, Jaccard ≥ 0.7). Sealed files are never opened; only manifest hash
  lists. The reserved wme-c1 pools (`sycophancy_real`, the harmful pools, WikiText-2) are not in these sources and
  are not touched; the MMLU-Pro family is drawn only into `mmlupro_gen`.
- **Split.** `dev` only: **N = 500 per benchmark** (R4's simulation grid uses N = 500), first 500 groups in salted
  rank order. No sealed split (stage W has no confirmatory read) and no construction split (§3 reuses construction
  data). The builder writes `manifest.json` with every used hash, so `wme-w` is registered as an exposure before
  `wme-c2-gen` is built. The report states the fresh groups left per benchmark against the `wme-c2-gen` sizes
  (3,463 MMLU-Pro, 1,784 SuperGPQA).

## 3. Training paths (H1-type, fresh seeds)

- **Recipe (frozen from pilot 3, pilot-informed).** LoRA rank 8, α 16, on every layer's residual writers; AdamW,
  warm-up 10, batch 8 rows, 2 epochs; loss on response tokens plus `<|im_end|>`; non-thinking template.
- **Data.** The 828 gold-verified thinking-mode final answers (≤ B tokens) that pilot 3 generated on `wme-p3h`
  construction items (`h/teacher.jsonl`). They are construction data and are never evaluated on. Reusing them
  instead of regenerating them saves ≈ 0.54 GPU-h; this is a declared pilot-informed choice.
  - 64 rows (32 MMLU-Pro, 32 SuperGPQA; seed `wme-w:feature-items`) are held out of every run as R1 feature items.
  - The other 764 rows are split into two disjoint halves A and B (seed `wme-w:shards`).
- **Runs (fresh seeds, namespace `wme-w`, `sha256("wme-w:<label>")`):** r1 = (A, lr 1e-4), r2 = (B, lr 1e-4),
  r3 = (A, lr 2e-4), r4 = (B, lr 2e-4). Factors are saved at the 8 path checkpoints (fractions j/8 of the run's
  steps, j = 1…8) and synced back.
- **H3-type paths are not run.** H3 failed (P2), so SVF scales are not in the R5 library (R5, R8); P5's
  "H1/H3-type" is met by H1-type paths. H4 projections are not measured (R8) and are not in the library.

## 4. Library members and paths (R4, R5)

A member is m = (run r, checkpoint t, dose λ, layer-group mask G, merge weight w), realised as
ΔW(m) = λ · Σ_{L ∈ G} [w·ΔW_{r,L}(t) + (1 − w)·ΔW_{r′,L}(t)] on the residual writers. Fine layer groups g1…g8 are
layers 0–3, 4–7, …, 28–31; coarse groups (features) are layers 0–7, 8–15, 16–23, 24–31.

| Path type (per run) | Varies | Fixed | Points (8) |
|---|---|---|---|
| S_r (training step) | t = round(j/8 · steps) | λ 1, all layers | j = 1…8 |
| L_r (dose) | λ ∈ {0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2} | final t, all layers | 8 |
| M_r (mask growth) | G = g1 ∪ … ∪ gj | final t, λ 1 | j = 1…8 |
| X_{12}, X_{34} (merge, only if cheap) | w ∈ {1, 6/7, …, 1/7, 0} between the two runs' final checkpoints | λ 1, all layers | 8 |

The point (r, final t, λ 1, all layers) is shared by S_r, L_r and M_r and evaluated once: 22 unique points per run,
88 for four runs; the merge paths add 12. The H-T2 units are the 12 S/L/M paths. A merge path has no base-anchored
per-path curve, so X paths are not H-T2 units: their members are predicted zero-shot with both parent runs held out
and reported descriptively (R4's "unseen masks and merges").

**Evaluation queue (priority, frozen):** canary → base → base re-run (reversed item order; null discordance) →
S_r1…S_r4 → L_r1…L_r4 → M_r1…M_r4 → X_{12}, X_{34} → canary repeat. The queue stops at the in-job deadline
(instance launch + 3.75 h), always keeping time for the canary repeat and the sync. A path enters the analysis only
if all 8 of its points were evaluated; the report lists every path that is incomplete.

## 5. State and features (R5)

**Features (R0/R1 only; nothing is teacher-forced through an edited model).** R2 passes (teacher-forced edited
passes) are not used in stage W; if a later design uses them they are charged as a step (R5).

- **R0, member level (CPU arithmetic on ΔW):** the Frobenius norm of ΔW(m) restricted to each coarse group (4).
- **R1, member level (exact derivatives at θ₀ by autodiff; no finite differences):** for each of the 64 feature
  items i and three objectives, the gradient ∇f_{o,i}(θ₀) of
  - (a) the stop log-probability log p(`<|im_end|>` | prompt, concise target) (the termination hazard after the answer),
  - (b) the mean token log-probability of the concise target,
  - (c) the answer-letter margin after "The answer is (": logit of the gold letter minus the log-sum-exp of the others;

  projected on every realised delta: J1_o(m) = mean_i ⟨∇f_{o,i}(θ₀), ΔW(m)⟩ (3 features). J1 is linear in ΔW, so
  every λ, mask and merge member is exact from per-layer inner products with the saved checkpoints.
  Second-order jets (J2) are **not** computed (cost, and P1's open S1/S2 numerical defect); declared omitted.
- **R0, item level (the base reference on the same dev items, shared by all arms):** base truncation T_i, base
  correctness C_i and the base response length ℓ_i/B in 6 quantile bins among untruncated items. Item cells
  c = (T, C, ℓ-bin), 13 per benchmark.

**WM (the R5 per-item state, P(finish ≤ B) × P(correct | finish)).** Per benchmark and member:

- logit P(finish | c, m) = a_c + (β + η·ℓ_c + ζ·T_c)ᵀ φ(m) + δ_f;
- logit P(correct | finish, c, m) = b_c + (κ + ν·ℓ_c)ᵀ φ(m) + δ_c;
- predicted accuracy = Σ_c n_c P(f) P(c|f) / N; predicted truncation = 1 − Σ_c n_c P(f) / N.

φ(m) = the 7 features, standardised on the training members. Grouped-binomial likelihood over (member, cell) counts
with an L2 penalty 1.0 on every φ coefficient (cell intercepts unpenalised). δ_f = δ_c = 0 at k = 0; for k ≥ 1 they
are fitted on the held-out path's first k points (penalty 1.0), everything else fixed.

## 6. H-T2 analysis on stage-W data (R4; descriptive here)

**Cross-fitting.** Leave one run out: the WM and every learned baseline are fitted on the unique members of the
other runs' S/L/M paths plus the base point; the held-out run's paths are predicted. Each held-out path p is a unit.

**Settings.** zero-shot k = 0 (only R0/R1 features of the path's members) and few-shot k = 2 and k = 4 (the first k
real points of the path are given). Endpoints: accuracy and truncation on both benchmarks (4).

**Baselines (all use the same real points).**

| Baseline | k = 0 | k = 2, 4 |
|---|---|---|
| persistence (last real point; at k = 0 the base value, i.e. "no change") | ✓ | ✓ |
| 2-parameter power law y = y_base + b·(x/x₁)^c, c ∈ [0.01, 3] (challenger) | — | ✓ |
| parametric learning-curve ensemble (equal-weight mean of the converged fits of lin, log2, exp2, pow2 at k ≥ 2, plus pow3 a − b·x^(−c) and exp3 at k = 4; Domhan-style) | — | ✓ |
| LC-PFN (`lcpfn` 0.1.3, model `EMSIZE512_NLAYERS12_NBUCKETS1000`, available offline; posterior mean; base anchor at x = 0, point j at x = 12.5·j; truncation fed as 1 − rate) | — | ✓ |
| pooled cross-path curve of the same path type, offset by the mean residual of the k points (freeze-thaw-type, world-model seat B3; added, stricter) | ✓ | ✓ |
| StatNN: MLP (32 hidden, tanh, α 1e-2, L-BFGS, 5 seeds averaged) on φ(m) → the 4 endpoint changes | ✓ | ✓ (offset by the mean residual of the k points) |
| static map: ridge (α 1.0) linear map φ(m) → the 4 endpoint changes | ✓ | ✓ (offset by the mean residual of the k points) |

Every arm uses the k given points in the same way except the per-path curve fits, which use only them: the WM
through its two path offsets δ, StatNN, the static map and the pooled curve through a per-endpoint offset.

Path coordinates x: S → training step, L → λ, M → j, X → 1 − w.

**R6 symmetric shrinkage (flag 1, transcribed).** No generated-accuracy "raw field" exists, so the analogue is the
no-change reference: in the zero-shot setting every learned arm (WM, StatNN, static map) predicts
base + s·(model − base), s ∈ {0, 0.25, 0.5, 0.75, 1}, s chosen by the same inner leave-one-run-out folds and the same
loss (mean |error|/SE), ties within 0.5 % to the smaller s, for every arm alike.

**Error and statistic.** e = prediction − measured at points j = k+1…8; SE = √(max(y(1 − y), 1/N)/N) of the measured
point. Path score E_A,p = mean |e|/SE over its points and the 4 endpoints. Contrast vs baseline B: d_p = E_B,p − E_WM,p,
C_B = 1 − mean E_WM / mean E_B. Test: one-sided sign-flip over paths (exact if P ≤ 16, else 10,000 flips), α 0.025;
within a setting the WM must beat **every** baseline of that setting (intersection–union); Holm over the three
settings. In stage W these are descriptive (12 held-out paths from 4 runs, exploratory).

**Nulls the WM must not beat (R4).**
- N1 shuffled edit features: φ blocks permuted across paths (20 seeded permutations); WM, StatNN and static refitted.
- N2 path-label permutation: outcome curves reassigned to other paths of the same type (20 seeded derangements).
- N3 no-edit-information world: a semi-synthetic world in which curves are the pooled curve of their type plus a path
  random effect and binomial noise, independent of φ (simulation, §7).

Reported: per baseline, per setting, per path type: mean |e|/SE, C, d̄, sign-flip p; the null results; the
world-model seat's stage-W gate (Spearman LB > 0 between WM-predicted and measured accuracy change, and WM beating
StatNN in |error|), descriptive.

## 7. H-T2 power simulation (R4: power ≥ 0.8 with ≤ 40 paths)

- **Alternative world (calibrated on stage W).** Truth for a simulated path = the WM fitted on all complete stage-W
  paths, evaluated on a stage-W path's feature trajectory (resampled) scaled by a path factor LogNormal(0, 0.3),
  plus a path random effect on each logit with variance τ² estimated by method of moments from the cross-fitted
  WM residuals net of sampling noise. Counts per cell are drawn binomially at N = 500 per benchmark with the stage-W
  base cell sizes. The full pipeline (WM, every baseline, cross-fitting by run, every setting) is refitted per
  replicate. The simulated runs carry 3 paths each (S, L, M), as stage W.
- **Null worlds:** N1 and N2 applied to alternative-world data, and N3 (pooled curve + τ, no φ dependence).
- **Grid** P ∈ {16, 24, 32, 40} paths (P/3 rounded up to whole runs; R4's grid), 8 points per path, N = 500;
  ≥ 400 replicates per (P, world). Cross-fitting uses 4 folds by run, as stage W's leave-one-run-out.
  LC-PFN stays in the comparator set; it is excluded from the simulation only if its CPU cost exceeds 2 s per path
  fit, and the report then states that the simulated power omits it.
- **Futility (CPU only).** P = 40 is simulated first. If, after 400 replicates, the one-sided 95 % Clopper–Pearson
  upper bound of the power is < 0.8 for every setting, the smaller P are not simulated (with more paths the test
  can only gain power under this design) and H-T2 is "not powered".
- **Second estimate (empirical).** Resample P held-out paths with replacement from the observed stage-W d-vectors
  and apply the same IU sign-flip and Holm rule (2,000 replicates).
- **Registration rule (pre-declared; stricter than R4: both estimates must pass).** A setting is registrable at the
  smallest P ≤ 40 where the simulated power ≥ 0.8, the empirical power ≥ 0.8, and every null rejection rate is
  ≤ 0.035. H-T2 is recommended for registration with the registrable settings (Holm re-applied over them) and the
  priced design at that P. If no setting is registrable, H-T2 is registered **"unresolved — not powered"** and the
  paper does not claim (a) (R4).

## 8. E2 power simulation: WM vs direct multi-fidelity best-of-N at total-compute parity (R1, P3)

On the sycophancy library of the v2b selected design (n_rep 8 → 24 episodes, 8 per family; B = 48 full dev
evaluations per arm; dev 64; test 96; family-stratified sign-flip on episode paired differences, 10,000 flips).

- **Library per episode.** 512 candidates; (true gain, guard breach) drawn by resampling the 32 pilot-2 sycophancy
  E2 candidates of the episode's family pool with jitter (SD 0.5 × the robust between-candidate SD), dev and test
  noise from `syc_se_64` (v1 inputs).
- **Shared planner (both arms).** Two-rung successive halving: screen at 16 dev items, promote the top quarter to all
  64; winner = largest dev LCB among full-dev candidates that pass the dev guard, else the base; winner scored on the
  96 test items with the guarded score. The only differences: the WM arm screens candidates in WM-prior order, best-of-N
  screens a random order without a prior, and the budgets differ.
- **Total-compute parity.** The WM arm has B = 48 full-evaluation equivalents. Best-of-N gets B plus the WM's charges
  converted at the v2b unit (t_bundle 14.58 s): the whole bank (12.08 GPU-h) and the R1 field compute (2,782 s), both
  amortised over the 24 sycophancy episodes.
- **WM prior quality (planning input, exposed pilot-2 data).** ρ̂ = Spearman between the static field and the
  observed change on the pilot-2 `sycophancy_real` R2 steps; WM prior = ρ̂·z(true gain) + √(1 − ρ̂²)·noise.
  Sensitivities: ρ ∈ {0.5ρ̂, ρ̂, √ρ̂} and "no compute charge".
- **Rule.** Power at α = 0.0125 (Holm's first step within the E2 family, the conservative bound) under ρ̂ at parity,
  with size ≤ 0.035 in the exchangeable null (both arms the same procedure and budget), ≥ 1,000 replicates.
  Power ≥ 0.8 → **co-primary**; otherwise a **registered secondary**, reported whatever the result (R1).

## 9. Stop rules

1. The in-job deadline (launch + 3.75 h) stops the evaluation queue; the canary repeat and the sync always run.
   The stage-W ledger is cut off at 4.0 instance-h by the terminator (`t_max_hours` 4.7 covers the timing pilot).
2. A failed bitwise restore of θ₀ in the engine, a non-finite training loss or a vLLM/HF writer mismatch stops the
   job (`FAILED:<step>`); nothing after it is generated.
3. A canary repeat that is not bitwise identical is reported; the analysis proceeds (all points on one instance).
4. At most one relaunch after an infrastructure failure, only if the cumulative ledger leaves ≥ 1.5 h of the
   stage-W cap; it resumes from the synced `out/` (completed members are skipped by key; partial members are
   discarded).
5. If fewer than 3 runs have complete S and L paths, the H-T2 analysis is reported descriptively and the
   simulation uses only the complete paths; the registration rule (§7) is unchanged.
6. R9.2 pricing: every measured unit rate (one member evaluation, one training run, the timing-pilot rates) is
   compared with its R9 / pro-engineer estimate; a rate above 1.5× its estimate is flagged in the report and the
   confirmatory price uses the measured rate (re-pricing by amendment before any confirmatory data exist).

## 10. Outputs

`data/wme/stagew/run-20261118/` (uncommitted; `manifest.json` with seeds, config hash, git SHA, model revision,
versions); `docs/reviews/wme/wme-stagew-2026-11-18.md` and `.json`, generated by `scripts/wme_w_report.py`, marked
EXPLORATORY: per-baseline errors for H-T2 few-shot and zero-shot, the null results, both power simulations and the
resulting registration recommendations, the priced confirmatory design (GPU-h, $), the ledger (instance ids
terminated, versions), caveats.

## 11. Pilot-informed choices (for PREREG v2)

The H1 recipe and its teacher data; B = 2,048; N = 500 per point; the path types and grids (8 points; λ grid;
8 fine / 4 coarse layer groups; lr 1e-4 and 2e-4; data halves); k ∈ {0, 2, 4}; the feature set and penalties; the
baseline set and their fixed settings; the simulation's calibration from stage W. Stage-W outcomes may train later
models and size later designs; they never test a registered claim.
