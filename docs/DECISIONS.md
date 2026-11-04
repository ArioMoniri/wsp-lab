# Decisions

## D1 (4 Nov 2026) — bf16 rounding of small deltas
Finding (reviews/P2/fix-eval-lora-boardA-2026-11-04.md F2): merging a LoRA delta into bf16 weights rounds away
~87 % of non-zero delta entries for a 5-step LoRA; rounding alone moves logits by ~40 % of the intended effect.
Decision: the subject models are evaluated in bf16 as deployed. The zoo therefore stores the **effective** delta
Δ_eff = bf16(W + Δ) − W (what actually changed), together with `rounding_stats` (fraction of entries lost,
‖Δ‖ vs ‖Δ_eff‖). Predictor features use Δ_eff. The eval loader must apply Δ_eff exactly (upcast W to fp32, add,
cast to bf16) and record the stats in its manifest. Any fp32-merge variant is a separate, labelled experiment.

## D2 (4 Nov 2026) — eval tiers and token budgets
The base model loops in ~90 % of generations, so generative benchmarks cost max_new_tokens × n tokens per
entry regardless of quality. Tiers: **fast** (per-entry zoo screening): gsm8k 50 / math500 25 / ifeval 25 /
mmlu_pro 100 / hellaswag 100 / refusal 50 / kl 50, max_new_tokens **1024**, batch 32 — loop rate is still
measured as hit-1024; **cheap** (base and selected entries): the P2 battery at 4096; **faithful** (headline
comparisons with the blog): 32,768 tokens, few entries. Throughput work (vLLM or HF static-cache batching) is a
perf-challenger task; the predictor targets are the fast-tier scores, with cheap-tier scores used to
calibrate on a subset.
