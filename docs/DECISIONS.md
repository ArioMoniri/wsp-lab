# Decisions

## D1 (4 Nov 2026) — bf16 rounding of small deltas
Finding (reviews/P2/fix-eval-lora-boardA-2026-11-04.md F2): merging a LoRA delta into bf16 weights rounds away
~87 % of non-zero delta entries for a 5-step LoRA; rounding alone moves logits by ~40 % of the intended effect.
Decision: the subject models are evaluated in bf16 as deployed. The zoo therefore stores the **effective** delta
Δ_eff = bf16(W + Δ) − W (what actually changed), together with `rounding_stats` (fraction of entries lost,
‖Δ‖ vs ‖Δ_eff‖). Predictor features use Δ_eff. The eval loader must apply Δ_eff exactly (upcast W to fp32, add,
cast to bf16) and record the stats in its manifest. Any fp32-merge variant is a separate, labelled experiment.
