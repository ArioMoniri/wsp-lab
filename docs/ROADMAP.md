# Roadmap — adopted programme, 5 November 2026

This programme supersedes conflicting historical phases below. TF-v1 yes; core ≤30 local GPU-h; additional history no; new Lambda $0. Existing A10 liability remains open. [Frozen registration](PREREG_H1_TF.md) controls all gates. None is checked by this design revision; Board A remains REVISE pending evidence.

| Date (2026) | Deliverable / owner | Required gate |
|---|---|---|
| **6 Nov** | Ten-entry pilot / Eval, Metrology, ledger owner | Inventory/source/item freeze before TF scoring; measured cold/warm/reference/retry/closure overhead; median≤2 min and full projected scope fits remaining 19-h scoring/30-h core caps. Reconcile A10 bill/liability. Reprice optional scope/re-pilot or stop if it cannot fit. |
| **7 Nov** | Legacy bridge / Eval, Metrology | 100 legacy items; median≤.02 nats, p99≤.2, argmax≥99%; failure versions endpoint/rebuilds matched references. Independently measured repeat/device/uncertainty gates in registration §6; bridge alone is insufficient. |
| **8–11 Nov** | Qualified fresh panel / Zoo, Metrology | Hash-ranked 110-candidate LoRA/task inventory plus chain closure, ≥100 genuinely fresh source components, disjoint 1,000-item primary; measured reliability lower bound≥.90 and paired uncertainty upper bounds≤δ/2. Certify provenance/device support, quarantine, missingness and all ancestors. |
| **12–13 Nov** | Residual estimand / Predictor, Metrology | D3-only fixed pipeline, three prediction-averaged seeds, pre-unsealing TF-specific ≥80% power and qualification; once-only sealed H1a, paired/joint 4,000-replicate CIs, θ≥.15/lower>0/half-width≤θ/2. Missing gates mean untested; Board A acceptance pending means exploratory. |
| **14–16 Nov** | Directional screen / Features | Candidate frozen before H1a unsealing; D3-only development and matched tree/destroyed-frame controls. +.05 residual-rank and ≥5% MAE continuation screen, not powered falsifier. Sealed H1b descriptive only; no fresh-label feature development. |
| **17–18 Nov** | History census / Modeling, Metrology | Additional 10 h declined: cached closure/cost/design only, no history compute. Future mask/shuffle study needs separate funding, TF power and length-one CI inside ±.05 equivalence margin. |
| **19 Nov** | Report freeze / Chair | All outcomes/exclusions, per-cell baseline/collapse audit, manifests, actual spend and unresolved gates. Stop at cap, no automatic extension. |

- [ ] Pilot timing, measured provenance and bill reconciliation.
- [ ] Bridge and prospective measured qualification.
- [ ] Source/item manifests, frozen models and TF-specific power.
- [ ] Once-only H1a and descriptive directional screen.
- [ ] Cached history census and final claim/budget report; H2 compute parked.

Budget: 19.00 scoring +4.22 cache repair +2.11 directional extraction +2.00 fits +2.67 reserve = **30.00 local GPU-h**, all overhead included. TF 1–2 min and directional 33 s remain unmeasured assumptions. [Registration §9](PREREG_H1_TF.md#9-budget-reconciliation-and-scaling-stop) reconciles allocations and $95.71 closed Lambda spend plus the unresolved A10 bill. Measurement/benchmark work is an accepted outcome. Later powered H2/H3 are needed for dynamics/transfer; H4–H6 remain parked under [HYPOTHESIS](HYPOTHESIS.md).

## Historical roadmap — 3–4 November (preserved, not current authorization)


Gates are pass/fail. A phase is done when its DoD (`docs/DOD.md`) is met and its gate script is green.

## P0 — Infra (today)
- [x] Server awake, GPU RTX PRO 4500 Blackwell 32 GB, CUDA 13.2, Python 3.12
- [x] Repo `~/projects/wsp-lab`, uv venv, pyproject, tmux job convention, worktrees under `.worktrees/`
- [x] `pytest -q` green (CPU), `pytest -m gpu` green (incl. Mamba-2 CUDA backend); GitLab CI fully green on the self-hosted GPU runner (pipeline 3025374435, 4 Nov 2026)
- [x] HF token set (`hf auth login`) — needed for gated repos only
- Gate: `scripts/gate_p0.sh`

## P1 — Edit zoo on a tiny model (Qwen3.5-0.8B, local, ~2 days GPU)
Edit families (each parameterised, seeded, logged as JSON "edit descriptors"):
1. Random orthogonal-direction ablation per layer (control family)
2. Refusal-direction ablation with Heretic kernels (direction_index, max_weight, position…) — sampled
   from the Optuna search space so we get *hundreds* of partially-abliterated variants, not one
3. Heretic ARA (default modifier in 2.0) at several `preserve_good_behavior_weight` settings
4. LoRA SFT on GSM8K / Alpaca-style data at varied steps, LR, rank (W2T-style ARC-LoRA zoo)
5. Task-vector scaling and merges between (1)–(4)
6. `llm_edit`: editor-LLM proposed edits (Astra / Claude / cheap controls) with a bounded forward-pass budget, stored like any other delta (H5)
7. Trajectories: chains of 2–8 edits sampled from the above, since H2 needs sequences
Target zoo size: ≥ 600 single edits + ≥ 150 trajectories.
- [x] Builder merged 4 Nov 2026 (6 families, Heretic-parity kernels, identity/eligibility provenance); local full build started (tmux p1-zoo-full); heretic_ara family: first Lambda run stopped 4 Nov (adapter export bug, D6 addendum); 2 entries quarantined; rebuild after fix/heretic-ara-export
- Gate: zoo manifest validates, every entry has `delta.safetensors` (or LoRA) + `edit.json` + `eval.json`

## P2 — Cheap eval battery (runs for every zoo entry)
Benchmarks chosen for speed (each ≤ 2 min on 0.8B with vLLM/HF generate, batch 64):
- GSM8K 200-item fixed subset (acc), MATH-500 100-item subset (acc), IFEval 100 (prompt-level strict)
- MMLU-Pro 300 fixed (acc, 1 sample), HellaSwag 300 (loglik), WikiText-2 perplexity (first 64 × 1024 tok)
- Refusal rate on 100 harmful + over-refusal on 100 harmless (Heretic's own sets), KL-to-base on 100 harmless
- Loop rate: share of generations hitting max tokens (the blog's key Qwen3.5-0.8B failure signal)
All with fixed seeds, temperature 0 unless the task card forbids. Store per-item results, not just means.
- Gate: base model numbers reproduce within ±1.5 pt across two runs.


### Batch evaluation throughput (4 Nov 2026)
`python -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b` defaults to
`configs/eval_fast.yaml`; `--ids`, `--families`, `--eligibility`, token and batch overrides
are supported. Completed results with the same effective configuration are skipped;
`--resume` makes that intent explicit. OOM retries use B, floor(B/2), floor(B/4).

Run `20261104-032002-1c2cc588`: two production random-direction entries,
`bccb9ab10b5661e0` and `adf416337f5a2277`, batch 4, fast config restricted to
4 items per benchmark and 32 generated tokens while `p1-zoo-full` shares the GPU.
Measured entry times: 0.2997 and 0.3022 minutes; mean 0.301 minutes/entry.
Board A's D2 calibration estimates ≈ 9–12 min/entry, ≈ 120 GPU-h for
700 entries, and ≈ 50 h for the D3 stratified subset. These estimates are pending
clean measurement of one full D2 entry after the GPU is released.
Follow-ups before a full run: measure KL context-transfer cost and allocator growth;
run the batch base reproduction gate twice; decide WikiText 8 vs 64 windows explicitly.
Missing-eligibility legacy rows remain excluded. Entry load/apply errors abort the run;
restart repairs the index and resumes completed production entries.

## P3 — Features + baselines
- φ_stats: per-tensor (||Δ||_F, ||Δ||_F/||θ||_F, top-8 singular values of Δ, spectral entropy, sign stats,
  per-block cosine(θ_{t−1}, θ_t)) → ~40 floats × n_tensors
- φ_svd: W2T-style canonical tokens: for each adapted matrix, top-r (U_k, σ_k, V_k) with r = 8,
  projected to 64-d via a frozen random projection per shape class
- Baselines: delta-norm linear, StatNN (LightGBM on φ_stats), MLP on φ_svd
- [x] Extractor, hybrid-aware pooling (feature_version 3: 1098 stats + 120 spectrum dims), ancestry-grouped splits and baselines merged 4 Nov 2026 (bf99575); full-zoo extraction running (tmux p3-extract)
- Gate: baselines evaluated with 5-fold split *by edit family* and *by seed*; numbers in `data/runs/p3/`

## P4 — Predictors (the deliverable)
- M1 stateless: per-tensor token Transformer (W2T-like) → Δy
- M2 SSM: Mamba-2 block stack (mamba-ssm if it compiles on CUDA 13.2 / cc 12.0; else pure-torch
  selective-scan fallback in `wsp/predictor/ssm_torch.py`) over the edit sequence; input per step =
  pooled φ_svd(Δθ_t) ⊕ edit descriptor embedding; output = y_t for all K benchmarks
- M3 JEPA: encoder E_θ on φ_svd(θ_t), predictor P(E_θ(θ_{t−1}), e_t) ≈ sg(E_θ(θ_t)), VICReg variance term;
  linear head on latent → y_t; trained in two stages like JEPAWG
- [x] Trainers for M1, M2, M3, M5, M6 (+ parameter-matched M1 control, --holdout gate, --selftest) merged 4 Nov 2026; waiting for eval.json targets
- M5 (Board C pick, competes with M2 on H2): trajectory Transformer, per-tensor W2T tokens + causally-masked
  attention over edits, ≤ 2 layers d=256; acceptance ≥ 15 % rel. MAE reduction vs M1 on T ≥ 4 and ≥ M2 on all splits
- M6 (Board C pick, competes with M3 on H1/H3): Tabular-JEPA pretraining on φ_stats with ResidualPred, dz=32,
  frozen encoder + MLP head; acceptance H3 transfer ρ ≥ 0.5 on ≥ 6 of K benchmarks, no collapsed seeds
  (details: docs/MODEL_CANDIDATES.md)
- [x] M4 harness merged 4 Nov 2026 (c6655ec): src/wsp/agents/reader.py + score.py; waits for eval.json targets
- M4 LLM-reader: GPT-6 Astra / Claude Fable 5.1 given the same φ_stats + top SVD components as text,
  asked for Δy; prompt in `src/wsp/agents/reader_prompt.md`
- Gate: H1–H4 tables filled in `docs/RESULTS.md` with CIs (bootstrap over edits)

## P5 — Scale to 20–30B (Lambda GPUs)
- Qwen3.8-27B family: base-instruct, `huihui-ai/Huihui-Qwen3.8-27B-abliterated`,
  `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored`, own Heretic run at 3 KL targets,
  plus Gemma-4-12B and Nanbeige4.3-33B as second families
- 27B bf16 = 54 GB → 1× H100 80 GB for eval; Heretic on 27B needs 2× H100 or 1× B200 (180 GB)
- Budget estimate: eval battery ≈ 25 min per variant on H100; 40 variants ≈ 17 h ≈ $70. Heretic 27B
  200 trials ≈ 6 h on B200 ≈ $42 per run. Total P5 ≲ $300.
- Deliverable: does the predictor trained on 0.8B/4B/9B zoos transfer to 27B deltas (zero-shot ρ)?
- Gate: `docs/RESULTS.md` §27B

## P4b — Multi-base-model zoos (H6, added 4 Nov 2026)
- Build zoos for LFM3-1.7B (hybrid, downloaded), Qwen3.5-4B, Gemma-3-270M-it (gated; licence accepted 4 Nov, both -it and base downloaded) with the
  same families and configs; feature pooling must map each architecture's tensor types (audit per model).
- Train on two models, test on the third (leave-one-model-out), plus the ≤ 20-edit calibration arm.
- Board C (4 Nov 2026, docs/MODEL_CANDIDATES.md §H6): screening run = Qwen3.5-0.8B, LFM3-1.7B, Gemma-3-270M-it,
  leave-one-model-out with n = 100 labelled edits per model (300 total), all transforms fit on training folds only;
  two competing representation runs: (A) dimensionless spectral / change-set descriptors with broad role pools,
  relative depth and occupancy flags; (B) the same descriptors fed to a shared Tabular-JEPA/JEPAWG encoder with per-model adapters (zero-shot shared latent + ≤ 20-edit adapted arm); current pooled stats
  as the ablation; report ρ/MAE and the ≤ 20-edit calibration gain.

## P6 — Closing the loop (promoted from optional, 4 Nov 2026)
Forecast-then-verify: the predictor scores every proposed edit before evaluation; it then gates/ranks
candidate edits (weight-space Bayesian optimisation) and acts as critic for `llm_optimizer` runs where the
editor LLM replaces Adam/Muon steps. Target: replicate the blog's loop fix with far fewer forward passes (H5).
Board C order (4 Nov 2026, docs/MODEL_CANDIDATES.md §LLM-in-the-loop): C first in shadow mode (critic forecasts
logged, not acting), then A+C (editor edits gated by critic), B (LLM as optimiser) last. Prerequisites: zoo ≥ 600 edits
/ 150 trajectories and H1 ρ ≥ 0.6 before any critic-gated episode.

## Benchmarks: why these
Blog uses MATH-500, GSM8K, IFEval, MMLU-Pro, loop rate on Qwen3.5-0.8B; we keep those for comparability
and add refusal/KL because of the alignment sub-study.

## P4c — Protein-LM sidecar (ESM contact maps; user idea, 4 Nov 2026; council verdict: do later)
Board C (docs/reviews/ESM/board-c-esm-2026-11-04.md): bounded H6 sub-study; deterministic contact precision removes
the sampled-generation device noise, but ESM2's released contact predictor is a ~121-parameter logistic head over
APC-corrected attention maps, so head-only refitting is the mandatory first control. Board A
(docs/reviews/ESM/board-a-esm-2026-11-04.md): REVISE — admissible only as a pre-registered sidecar; controls: untouched
8M and 650M, head-only tuning, equal-budget LoRA/distillation from 650M logits, matched random edits, single-draw
best-of-N, a non-LLM search over the same weights, sham reload; identity-clustered protein splits with cluster
bootstrap; editor replicates as a crossed random effect. Pilot: ESM2-8M, frozen clustered CASP-derived test set,
20 proposals, target < 1 GPU-hour (estimate). Scheduled after the H5 Stage-1 pilot; not before.


## 10 Nov 2026 ordering amendment (before freeze)

[Chair synthesis](council/2026-11-10/SYNTHESIS.md), implemented by A5 F5;
Board A reviews at the freeze. This ordering governs the bounded programme;
no unchecked gate is marked complete and no compute starts with this document.

1. Freeze: Board A item 7 accepts the merged pre-G2 evidence plus F5 and A5.1 hashes.
2. G2: fields, chart, probe cache and 20-predecessor audit (≤3 local GPU-h cap).
3. CPU power/yield simulation on G2 development predictions, before G3.
4. G3 Part A: unchanged forty edits, 120 variants plus 19 references against the
   closed snapshot; apply the ordered A5.5 rules and F5.7 reporting.
5. G4 Part B: unchanged primary plus CPU LoL L/R/K/S and anchored K+F secondaries,
   with field-only/anchored-only comparators; descriptive, no funding power.
6. G5 shadow editor pilot: F5.4 validation/rebuild/reversibility contract and four
   access flags; mandatory 32-draw/block random best-of-N with nested 1/8/32 curves.
   Caps remain ≤10 local GPU-h plus ≤6 for best-of-N; stop/reprice at caps.

In parallel after freeze, run the capped F5.3 4B diagnostic on Lambda H100,
20 outcome-blind fixed-radius Gaussian edits plus shared base, 64-window WikiText;
A3 device qualification first or fast-tier-only reporting. The 1.82 TF-v1 GPU-h
linear size extrapolation is unmeasured and excludes overhead/qualification.
It neither delays nor conditions Part A and never gates a purchase; stop scale
expansion if inconclusive. This narrowly scoped parallel Lambda item supersedes
the historical no-new-Lambda wording only for this diagnostic, not a larger zoo.
G7 remains conditional on primary go; G8 needs separate pricing/power. Funding
stops on registered no-go; whole-line stopping needs all F5.5 predicates.
