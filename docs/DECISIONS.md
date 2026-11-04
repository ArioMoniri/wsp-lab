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

## D3 (4 Nov 2026) — evaluation budget and order
Board A's calibrated estimate (reviews/P2/feat-p2-batch-boardA-2026-11-04.md): ~9–12 min per entry under D2 on
this GPU, ≈ 120 GPU-hours for 700 entries. Order of work: (1) measure one clean D2 entry when the GPU is free;
(2) evaluate a stratified subset first — random_dir 60, abliteration 100, lora_sft 60, task_vector 30, chain
steps 50 (≈ 300 entries, ≈ 50 h) — enough for H1/H2 baselines; (3) pursue vLLM (if Qwen3.5 is supported) or a
Lambda H100 to finish the remainder; (4) refusal/KL prompts stay at 50/50 because they are cheap (short
continuations). The 32k faithful tier is reserved for the base model and ≤ 10 selected entries.

## D4 (4 Nov 2026) — Lambda GPU spend approved
User approved Lambda Cloud usage with a cap of **$150**: (1) Heretic ARA zoo family on one GPU (launched 4 Nov 2026 as a 1×A100-SXM4 40 GB instance at $1.99/h, the only type with capacity), (2) faster
evaluation of the D3 subset if local timing is slow. Rules: instances named `wsp-lab-*`, launched with a
per-hour price cap, results pushed to the private HF datasets, instance terminated as soon as its job ends;
spend ledger in `board/lambda_spend.md`.

## D5 (4 Nov 2026) — trajectory relations and per-step targets
Board A (reviews/P4/feat-p4-predictors-boardA-2026-11-04.md) showed that `parent_id` means different things per
family: for chain steps the parent is the predecessor state (`successor` edge); for task vectors the parent is
the *source* of a scaled delta applied to the base (`source` edge), not a predecessor. Predictors that model
history (M2, M5) and JEPA transitions (M3) use `successor` edges only. H1's target is the per-step change
Δy_t = y_t − y_{t−1} (predecessor = base for single-step edits); absolute level is an explicit secondary target.
A family holdout drops every entry whose ancestry contains the held-out family.
