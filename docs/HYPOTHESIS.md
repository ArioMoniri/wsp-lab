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
