# Council round 10, headroom seat: from log-prob headroom to benchmark accuracy on Qwen3.5-9B

**Status.** Post hoc. Every test is proposed for fresh `wme-p4` dev panels: new salt, and near-duplicate-disjoint from all pilot and wme-c1 items. None is for pilot-2 data. No GPU, model download, `data/` write or sealed split was used.

## BLUF

1. Pilot-2 GSM8K "headroom" is headroom on a **no-CoT** format: the prompt says "Give only the final numeric answer", with the prefix "The answer is " and thinking off (`src/wsp/wm/panel.py` `gsm8k_user`, `GSM8K_ANSWER_PREFIX`). Model-card GSM8K/MATH-style accuracy is a different regime.
2. The fetched Qwen3.5-9B card lists MMLU-Pro, GPQA-D, SuperGPQA, IFEval and HMMT, but **not GSM8K or MATH**. Its scores are not labelled by mode, and thinking is the default.
3. The largest documented headroom for non-thinking hybrids is **the model's own thinking mode**. Qwen3-8B scores 62.0 thinking vs 39.3 non-thinking on GPQA-D, and 97.4 vs 87.4 on MATH-500. NO-THINK RL gains approach the THINK accuracy but do not exceed it.
4. Top proposal: **T2N**, self-distilling θ₀'s thinking-mode verified answers into non-thinking mode under a frozen token budget. Its ceiling can be measured at R0.
5. Second: **RAFT/RLVR in the SVF action family**, with a mandatory **random-reward control**. Spurious rewards lift Qwen models.
6. The guard is miscalibrated. In 3 of 5 LoRA breaches (the sycophancy LoRAs), the harmless mean is not significantly above 0: that is **non-inferiority unproven at n = 64**, not over-refusal. The two stronger math LoRAs (λ 0.5 and 1) do shift. Use a powered, generated over-refusal guard.
7. Guard-preserving edits at R0: project LoRA deltas orthogonal to the refusal direction or onto the harmless null space.
8. To be tested on real accuracy, the world model must predict an **R1 proxy validated against R2 accuracy**: the log-prob of the base model's own verified solutions.
9. Power: at Miller's illustrative variance, a 3 pp paired effect needs about 969 items. IFEval (541 prompts) can detect about 4 pp only, so IFEval is secondary.
10. Batched generation is a prerequisite. At the measured unbatched rate of about 15 tok/s, a 500 × 512-token panel costs about 4.7 GPU-h per candidate.

## 1. What our data say (repo)

- **E2.** Qualifying counts were GSM8K 3/32, MATH 4/32 and sycophancy 10/32 (`docs/reviews/wme/wme-pilot2-2026-11-16.md` §5). Pilot 1 found no GSM8K or MMLU-Pro headroom (`wme-v0-pilot-2026-11-16.md`).
- **The guard** (`src/wsp/wm/p2design.py` L58–66 and L313; `panel2.pp_value`). It measures the mean change in sigmoid(margin), where the margin is the refusal-phrase logsumexp minus the compliance-phrase logsumexp. A breach is mean + 1.96 SE > 0.01, on 64 dev items.
- **Per-candidate guard values** (`wme-pilot2-2026-11-16.json`):
  - `lora:sycophancy_real:0.25`: +0.64 pp, SE 0.46, UCB 1.54. It breaches, but the harm is not significant.
  - `lora:gsm8k_gold_lp:0.25`: +0.28 pp, UCB 0.81. It passes.
  - `lora:mathcomp_gold_lp:1`: +3.17 pp, SE 1.00. This is a real shift.
- **Base rate.** The mean base harmless margin is −7.84 nats, so sigmoid ≈ 0.04 % (computed). The guard is a teacher-forced proxy, not a refusal rate.
- **Generation and cost.**
  - The 9B unbatched decode rate is about 15.1 tok/s (1,009 tokens in 66.8 s; `wme-ph100-2026-11-17.json`).
  - GSM8K generation used 20 tokens for 8 prompts, which means direct answers.
  - Lambda costs $4.89/h (`wme-27b-calib-2026-11-16.md`).

## 2. External evidence (fetched)

**Mode gap and transfer.**
- [Qwen3 report](https://arxiv.org/html/2505.09388v1), Qwen3-8B thinking vs non-thinking: GPQA-D 62.0/39.3; MMLU-Redux 87.5/79.5; AIME'24 76.0/29.1; IFEval-strict 85.0/83.0.
- [Thinking leakage](https://arxiv.org/abs/2610.14270): NO-THINK GRPO on Qwen3-1.7/4/8B mostly reactivates THINK mode, and "approaches THINK-mode accuracy but does not exceed it". [2510.12680](https://arxiv.org/abs/2510.12680) also finds reasoning leaking into no-think mode.
- [U-OPSD](https://arxiv.org/abs/2608.06296): Qwen3-8B non-thinking +10.7 % on a 5-benchmark math average.
- Steering: [2510.07364](https://arxiv.org/abs/2510.07364) recovers about 76 % of the RL gap. [ThinkEdit](https://arxiv.org/abs/2503.22048) edits 0.2 % of parameters for +3.34 % math. [SEAL](https://arxiv.org/abs/2504.07986) reports up to +11 %.

**RL and self-training.**
- [1-shot RLVR](https://arxiv.org/abs/2504.20571): MATH500 36.0 → 73.6 on Qwen2.5-Math-1.5B.
- [Spurious rewards](https://arxiv.org/abs/2506.10947): random rewards give +21.4 against +29.1 for true rewards on Qwen2.5-Math-7B. The effect is Qwen-specific.
- [RL updates 5–30 % of parameters](https://arxiv.org/abs/2505.11711).
- [RAFT](https://arxiv.org/abs/2504.11343) is competitive with GRPO.

**Spectral edits and merging.**
- [Transformer² SVF](https://arxiv.org/html/2501.06252v3): REINFORCE on singular-value scales with a KL penalty. Llama3-8B-Instruct GSM8K 75.89 → 79.15 with 0.58 M parameters.
- [Evolutionary merge](https://arxiv.org/html/2403.13187v1): 30.0 → 52.0 MGSM-JA, but it used 1,000 CMA-ES trials with real fitness.
- Exploring WSP, 9B held-out IFEval +1.85 pp (`round9-editor.md`).

**IFEval.** [Steering](https://arxiv.org/html/2410.12877v2) boosted 2 of 4 models when the instruction was present.

**Guards.**
- [Refusal is one direction](https://arxiv.org/abs/2406.11717); adding it makes models refuse harmless prompts.
- [AlphaEdit](https://arxiv.org/abs/2410.02355) uses null-space projection.
- [Safe LoRA](https://arxiv.org/abs/2405.16833) needs the base weights, and [Qwen3.5-9B-Base exists](https://huggingface.co/Qwen/Qwen3.5-9B-Base).
- [Shallow alignment](https://arxiv.org/abs/2406.05946) puts safety in the first tokens.
- [XSTest](https://arxiv.org/abs/2308.01263) has 250 safe and 200 unsafe prompts.

**Power.** [Miller](https://arxiv.org/html/2411.00640v1) uses paired differences, n = (z_{α/2}+z_β)²(ω²+σ²/K)/δ², and gives about 969 items for 3 pp at ω² = 1/9.

**Fetch log.**
- 2403.13187v3 html returned 404; v1 was used.
- Every arXiv `/abs/` link above was an abstract-only fetch, and no full-paper audit is claimed for it. Tulu 3 (2411.15124) gave no numbers, so none is claimed.

## 3. Proposals

**Common rules.**
- Edits are trained on construction items only.
- The world model ranks candidates at R0/R1.
- Every candidate is judged at **R2 by generated accuracy** on fresh `wme-p4` dev panels.
- The sealed final split is registered before any data.
- Decoding is frozen in advance: non-thinking, one sampling choice, K samples and a fixed `max_new_tokens`.

**M0. Measurement switch (prerequisite).**
- **Objectives.** Generated accuracy on GSM8K-format CoT (drawn from the GSM8K train pool) and on MMLU-Pro and SuperGPQA multiple choice. Both MC benchmarks are on the card.
- **R1 proxy.** The log-prob of θ₀'s own verified-correct sampled solutions, and their margin over incorrect ones. The paths are R0 artefacts, so proxy fields are R1.
- **Test.** Spearman correlation between predicted Δproxy and measured Δaccuracy, over at least 32 R2 candidates.
- **Kill.** LB ≤ 0. Then the world model makes no accuracy claim, and candidates are screened by R2 directly.
- **Needs.** P-H100b batched pricing (RULING-wme-v2b A9).

**G1. Guard recalibration.**
- **Change.** The acceptance guard becomes the generated over-refusal rate on XSTest-style plus panel harmless prompts, using the frozen `wm.genaudit` refusal rule. Generated harmful refusal and WikiText are kept. The first-token margin stays as an R1 screen only.
- **Sizing.** Guard N comes from measured discordance. *Illustration:* 2 % discordance with a 1 pp margin needs about 770 items; the 2 % is an assumption.
- **Test.** On fresh dev, inert edits breach at ≤ 5 %, and a refusal-direction positive control is detected.
- **Kill.** Either check fails.

**G2. Projected LoRA (R0).** Three projections, each a CPU operation or a base-pass artefact:
1. (I − r̂r̂ᵀ)ΔW, removing the refusal direction.
2. ΔW(I − P_K), where K comes from base activations on guard construction prompts (AlphaEdit-style).
3. Optionally, the Safe-LoRA projection using W_instruct − W_base.

- **Test.** Paired against the unprojected LoRA: G1 is non-inferior, and at least 70 % of the gain is retained (proposed).
- **Kill.** Retention < 50 %, or no reduction in breaches.

**P1. T2N self-distillation (top proposal).**
- **Teacher.** θ₀ in thinking mode on construction prompts, gold-verified (RAFT filter).
- **Student.** A rank-8 LoRA trained in the non-thinking template, then G2-projected. Its budget is frozen at the base non-thinking p95 length.
- **Ceiling.** The thinking-mode accuracy of θ₀, measured at R0.
- **Candidates.** 32 (λ × data mix × projection), ranked by the M0 proxy.
- **Test.** Paired Δaccuracy LCB > 0 on MMLU-Pro or GSM8K-CoT (order frozen), with G1 passing. Length and "wait" rate are reported.
- **Kill.** No LCB > 0, or a gain only through over-budget length. That would be mode leakage, not enhancement.
- **Cost.** ≤ 6 H100-h (about $29) plus ≤ 1 local GPU-h.

**P2. RAFT/RLVR with SVF (C7 W2).**
- **Method.** REINFORCE on z with a KL penalty to θ₀, non-thinking, 1–16 shots.
- **Controls.** A random-reward arm and a format-only arm are mandatory.
- **Kill.** True reward ≤ random reward. That is Qwen prior amplification, not enhancement.
- **Cost.** ≤ 8 local GPU-h, or ≤ $40.

**P3. Thinking-direction weight steer (R1).**
- **Method.** The contrast between think and no-think residuals at the first answer token, compiled into a rank-one late-layer writer shift. ThinkEdit-style head edits are a variant.
- **Test.** 8 settings at a fixed budget.
- **Kill.** All 8 have LCB ≤ 0, or anomalies exceed 2 %.

**P4. Fresh verifiable instruction following.**
- **Why not IFEval directly.** At n = 541 IFEval can resolve about 4.0 pp (computed), against the roughly 1.85 pp effect expected on 9B.
- **Design.** Dev panels compose IFEval instruction types onto fresh prompts. The edits are IF vectors, CWS/TFTV and constraint RLVR. IFEval itself is a Holm-adjusted secondary endpoint.

**P5. World-model merge search (R0).**
- **Method.** sep-CMA-ES (C6) over coefficients of the G2-projected library {T2N, SVF, IF, instruct−base}, run inside R0/R1. The world model stands in for Sakana's 1,000 real-fitness trials. The top 4 go to R2.
- **Kill.** The merge is no better than the best donor, or no better than best-of-N at equal budget.

## 4. Power (Miller's formula; ω² = 1/9 is illustrative and must be measured)

| n | 541 | 969 | 1,319 | 2,000 | 2,548 |
|---|---|---|---|---|---|
| MDE (α 0.05, power 0.8, K = 1) | 4.0 pp | 3.0 pp | 2.6 pp | 2.1 pp | 1.85 pp |

GPQA-D is too small for a primary endpoint.

**Order.** M0 and G1, then G2, then P1, then P2, then P3/P4, then P5. Only winners are ported to 27B, where the T2N ceiling is re-measured at R0.

## Summary table

| proposal | mechanism | access tier | test on fresh data | cost (GPU-h, $) | kill criterion |
|---|---|---|---|---|---|
| M0 accuracy objectives + self-path proxy | Generated accuracy as the objective; R1 proxy from the base model's verified solution paths | Proxy R0/R1; accuracy R2 | Spearman of predicted Δproxy vs measured Δacc over ≥ 32 candidates, `wme-p4` dev | ≤ 2 local after P-H100b batching; $0 | LB ≤ 0, so no world-model accuracy claim |
| G1 guard recalibration | Generated over-refusal (XSTest-style) and harmful refusal, powered non-inferiority; first-token margin kept as a screen | R2 guard, R1 screen | Inert-edit null breach ≤ 5 %; refusal-direction positive control detected | ≤ 1 local | Null breach > 5 % or positive control missed |
| G2 projected LoRA | Refusal-direction orthogonal, harmless null-space and Safe-LoRA projections of ΔW | R0 | Paired vs unprojected: G1 non-inferior, ≥ 70 % gain retained | < 0.5 local + CPU | Retention < 50 % or no breach reduction |
| P1 T2N self-distillation | θ₀ thinking-mode verified answers → non-thinking LoRA under a frozen token budget | Teacher R0; ranking R1; evaluation R2 | Paired Δacc LCB > 0 on MMLU-Pro or GSM8K-CoT, G1 passes, length reported | ≤ 6 H100-h (about $29) + ≤ 1 local + evaluation | No LCB > 0, or gain only through over-budget length |
| P2 RAFT/RLVR-SVF | REINFORCE on singular-value scales, KL to θ₀, 1–16-shot | Ranking R1; evaluation R2 | Exceed random-reward and format-only controls, LCB > 0 | ≤ 8 local or ≤ $40 | True reward ≤ random reward |
| P3 thinking-direction weight steer | Think − no-think residual contrast compiled into a rank-one writer shift | R1 | Δacc LCB > 0 at a fixed budget, anomalies ≤ 2 % | ≤ 1 local + evaluation | All 8 settings LCB ≤ 0 |
| P4 fresh verifiable instruction following | IF vectors, CWS/TFTV, constraint RLVR; IFEval proper secondary | R1 / R2 | Fresh composed constraints, LCB > 0; IFEval Holm-adjusted | ≤ 4 local | No LCB > 0 or anomaly > 2 % |
| P5 world-model merge search | sep-CMA-ES over library coefficients inside R0/R1, top 4 to R2 | R0/R1 planning | Beat best donor and best-of-N at equal real-evaluation budget | CPU + 4 × evaluation | ≤ best donor or ≤ best-of-N |
