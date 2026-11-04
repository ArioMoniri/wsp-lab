# Roadmap (cut as of 3 Nov 2026)

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
- [x] Builder merged 4 Nov 2026 (6 families, Heretic-parity kernels, identity/eligibility provenance); local full build started (tmux p1-zoo-full); heretic_ara family started on a Lambda 1× A100-SXM4 40 GB (`gpu_1x_a100_sxm4`, $1.99/h, 4 Nov 2026 06:13 UTC; see `board/lambda_spend.md`)
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
- Build zoos for LFM3-1.7B (hybrid, downloaded), Qwen3.5-4B, Gemma-3-270M-it (gated; token available) with the
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
