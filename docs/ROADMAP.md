# Roadmap (cut as of 3 Nov 2026)

Gates are pass/fail. A phase is done when its DoD (`docs/DOD.md`) is met and its gate script is green.

## P0 — Infra (today)
- [x] Server awake, GPU RTX PRO 4500 Blackwell 32 GB, CUDA 13.2, Python 3.12
- [x] Repo `~/projects/wsp-lab`, uv venv, pyproject, tmux job convention, worktrees under `.worktrees/`
- [ ] `pytest -q` green (CPU), `pytest -m gpu` green
- [ ] HF token set (`hf auth login`) — needed for gated repos only
- Gate: `scripts/gate_p0.sh`

## P1 — Edit zoo on a tiny model (Qwen3.5-0.8B, local, ~2 days GPU)
Edit families (each parameterised, seeded, logged as JSON "edit descriptors"):
1. Random orthogonal-direction ablation per layer (control family)
2. Refusal-direction ablation with Heretic kernels (direction_index, max_weight, position…) — sampled
   from the Optuna search space so we get *hundreds* of partially-abliterated variants, not one
3. Heretic ARA (default modifier in 2.0) at several `preserve_good_behavior_weight` settings
4. LoRA SFT on GSM8K / Alpaca-style data at varied steps, LR, rank (W2T-style ARC-LoRA zoo)
5. Task-vector scaling and merges between (1)–(4)
6. Trajectories: chains of 2–8 edits sampled from the above, since H2 needs sequences
Target zoo size: ≥ 600 single edits + ≥ 150 trajectories.
- Gate: zoo manifest validates, every entry has `delta.safetensors` (or LoRA) + `edit.json` + `eval.json`

## P2 — Cheap eval battery (runs for every zoo entry)
Benchmarks chosen for speed (each ≤ 2 min on 0.8B with vLLM/HF generate, batch 64):
- GSM8K 200-item fixed subset (acc), MATH-500 100-item subset (acc), IFEval 100 (prompt-level strict)
- MMLU-Pro 300 fixed (acc, 1 sample), HellaSwag 300 (loglik), WikiText-2 perplexity (first 64 × 1024 tok)
- Refusal rate on 100 harmful + over-refusal on 100 harmless (Heretic's own sets), KL-to-base on 100 harmless
- Loop rate: share of generations hitting max tokens (the blog's key Qwen3.5-0.8B failure signal)
All with fixed seeds, temperature 0 unless the task card forbids. Store per-item results, not just means.
- Gate: base model numbers reproduce within ±1.5 pt across two runs.

## P3 — Features + baselines
- φ_stats: per-tensor (||Δ||_F, ||Δ||_F/||θ||_F, top-8 singular values of Δ, spectral entropy, sign stats,
  per-block cosine(θ_{t−1}, θ_t)) → ~40 floats × n_tensors
- φ_svd: W2T-style canonical tokens: for each adapted matrix, top-r (U_k, σ_k, V_k) with r = 8,
  projected to 64-d via a frozen random projection per shape class
- Baselines: delta-norm linear, StatNN (LightGBM on φ_stats), MLP on φ_svd
- Gate: baselines evaluated with 5-fold split *by edit family* and *by seed*; numbers in `data/runs/p3/`

## P4 — Predictors (the deliverable)
- M1 stateless: per-tensor token Transformer (W2T-like) → Δy
- M2 SSM: Mamba-2 block stack (mamba-ssm if it compiles on CUDA 13.2 / cc 12.0; else pure-torch
  selective-scan fallback in `wsp/predictor/ssm_torch.py`) over the edit sequence; input per step =
  pooled φ_svd(Δθ_t) ⊕ edit descriptor embedding; output = y_t for all K benchmarks
- M3 JEPA: encoder E_θ on φ_svd(θ_t), predictor P(E_θ(θ_{t−1}), e_t) ≈ sg(E_θ(θ_t)), VICReg variance term;
  linear head on latent → y_t; trained in two stages like JEPAWG
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

## P6 (optional) — Closing the loop
Use the predictor as a surrogate to *choose* edits (weight-space Bayesian optimisation), replicating the
blog's "fix the loop" result without 50 forward passes.

## Benchmarks: why these
Blog uses MATH-500, GSM8K, IFEval, MMLU-Pro, loop rate on Qwen3.5-0.8B; we keep those for comparability
and add refusal/KL because of the alignment sub-study.
