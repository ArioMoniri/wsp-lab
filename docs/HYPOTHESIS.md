# Hypotheses

Base object: a base model θ₀ and an edit sequence e₁…e_T producing θ₁…θ_T. Each θ_t has a performance
vector y_t ∈ ℝ^K over K cheap benchmarks (see `docs/ROADMAP.md` §Benchmarks). Δθ_t = θ_t − θ_{t−1}.

## H1 — Readability (replicates the blog, cheaply)
Weight deltas carry enough information to predict the *sign and rough magnitude* of Δy_t
without running the model. Test: a predictor f(φ(Δθ_t), φ(θ_{t−1})) → Δy_t achieves Spearman ρ ≥ 0.6
per-benchmark on held-out edits of the *same* base model, beating a weight-statistics baseline (StatNN)
and a "norm of delta" baseline.
Falsified if ρ < 0.3 on all benchmarks after honest tuning.

## H2 — Sequence matters (the SSM claim)
A stateful sequence predictor over the edit trajectory (Mamba-2/Mamba-3 block or a small state-transformer)
beats a stateless per-edit predictor with the same features, because performance after edit t depends on
accumulated drift (e.g. second abliteration pass hurts more after a LoRA merge).
Test: ≥ 15 % relative MAE reduction on trajectories of length ≥ 4.
Falsified if the stateless model is within noise.

## H3 — JEPA latent generalises across edit families
A JEPA-style weight encoder (predict embedding of θ_t from embedding of θ_{t−1} + edit descriptor,
VICReg anti-collapse, à la JEPAWG) transfers: trained on abliteration + random-direction edits, it still
ranks LoRA fine-tunes correctly (ρ ≥ 0.5) without seeing LoRA edits.
Falsified if transfer ρ < 0.2.

## H4 — LLM-as-reader vs. trained predictor
Given the same weight delta (as text/summary statistics + SVD components), a frontier LLM asked to predict
Δy (the blog's "native reading") is less accurate than the trained predictor on in-distribution edits, but
more robust to novel edit families. Measured with the same held-out sets, both directions reported.

## Alignment-edit sub-study (user request)
For Qwen3.5-0.8B/4B/9B and Qwen3.8-27B: compare **base-instruct**, **classic abliterated** (huihui-style
directional ablation), and **heretic** (Optuna-tuned ARA / projected abliteration, KL-aware) on
capability benchmarks, refusal rate and KL-to-base. Then ask: can the predictor tell, from Δθ alone,
which method was used and how much capability was lost? This directly tests "auditing without evaluation".

## Non-goals
- No new abliteration technique; we only consume published edits and standard Heretic runs.
- No claim about gradient-free optimisation; we predict, we do not optimise (Phase 5 optional).

## H5 — Editor models and LLM-in-the-loop optimisation (added 4 Nov 2026, user request)
Edits need not come from algorithms only. Two further edit sources enter the zoo and the predictor's
forecast-then-verify protocol:
- `llm_edit`: an editor LLM (GPT-6 Astra, Claude, cheaper controls) receives the weights in a Python sandbox
  plus a natural-language goal and proposes a weight edit (blog Experiment 8 setting, bounded forward passes).
- `llm_optimizer`: the editor LLM replaces or steers the optimiser step (proposes parameter updates or
  per-step hyper-parameters instead of Adam/Muon) for a short fine-tuning run; the predictor acts as critic
  before each proposed step is applied.
H5: with the predictor as critic, LLM-proposed edits/steps reach a given benchmark gain with fewer real
evaluations than (a) the same editor without the critic and (b) Adam/Muon with equal forward-pass budget.
Falsified if the critic does not reduce evaluations by ≥ 30 % at equal final score. Board C assesses the
design before any run (docs/MODEL_CANDIDATES.md §LLM-in-the-loop).

## P4 measurement qualification

P3 currently supplies cumulative-from-base delta features. H2 therefore measures
history beyond the latest cumulative snapshot, on successor trajectories only;
M1 already sees accumulated drift in that snapshot. For H3, M3's transition target
is the latent of these cumulative-from-base features, not absolute θ_t features.
Family holdouts exclude all descendants/sources with a held-out-family step in
full manifest ancestry, including unlabeled steps and chain `inner_family` values.
H1 uses measured predecessor-relative Δy; absolute levels are a separate baseline.

## H6 — Cross-base-model generalisation (added 4 Nov 2026, user request)
A predictor trained on edits of one base model should carry over to edits of a *different* base model when
its inputs are shape-agnostic (pooled per-tensor-type statistics, singular spectra, rounding statistics) and
edit descriptors are family-level only. H6: with the zoo extended to at least three base models of different
families and sizes (Qwen3.5-0.8B, LFM3-1.7B, Qwen3.5-4B; Gemma-3-270M as a small control), a predictor
trained on two of them reaches Spearman ρ ≥ 0.4 on the held-out model's edits for ≥ 6 of K benchmarks
without any target from that model. Falsified if ρ < 0.2 on all benchmarks. This is expected to be the hardest
hypothesis; a per-model calibration arm (≤ 20 labelled edits from the new model) is reported alongside.
