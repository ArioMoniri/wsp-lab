# Round 15, methods / design seat (23 Nov 2026)

**BLUF.**
1. Neither idea is a world-model problem by default: summed saliency (magnitude, Wanda = q_A × ½, OBS, Fisher a + q) already predicts deletion damage with zero fit. A learned model counts only if it beats a **first-order stack** on held-out units.
2. The stack is the pruning analogue of r13's "pooled curve + dose"; for unlearning it is the **pooled layer profile + edit size**.
3. **Qwen3.5-0.8B** for both: Apache-2.0, pre-norm, fits 16 GB, measured TF-v1 rate 1.04 min/entry. 9B is 18 GB bf16 and C1's; MedGemma has post-norm cancellation, 13.5 % batch flips and HAI-DEF limits.
4. **P1:** 256 masks (128 fit, 128 held out) at 30 % and 50 %; primary is held-out within-sparsity Kendall τ, ranker minus stack. About 7 GPU-h, $0.
5. **U1:** 200 synthetic people with synthetic medical records, inserted by full fine-tuning; 40 eval episodes. Primary: does a predicted weight set beat random and saliency-matched sets at equal count and norm? About 30 GPU-h, $0.
6. Power: P1 needs ~120 held-out masks for Δτ ≈ 0.10; U1 needs 40–47 people for a 10 pp gap under Holm over two contrasts.
7. Separate exploratory preregistrations, after C1, each behind a timed pilot with stop-and-reprice; no Lambda.
8. Confounds: suppression vs forgetting, dose, norm invariances, batch non-determinism, insertion-delta leakage.

## Q2. What a world model must beat

F9-P (HYPOTHESIS ~l.231; Board A F9a.4) already registers zero-fit τ of a + q, q_A (= Wanda up to ½, F9a.8) and −‖W⊙M‖². r13: beyond-dose prediction did not identify (one-axis bank; inner CV switched off GRU and SSM). M′ never ran M-A (42.26/42.0 GPU-h).

A learned model's only job is the **non-additive part** of realised damage: cross-layer interactions, compensation, invariances. **Mandatory baseline:** a ridge stack of per-layer sums of the four saliencies plus per-layer counts, fitted on the training split. Shuffled-feature and within-sparsity permutation nulls apply.

## Q3. P1: prunability study

- **Model:** Qwen3.5-0.8B (752.4 M text params; 18 GatedDeltaNet + 6 full-attention; pre-norm; `docs/ARCHITECTURES.md`); embeddings and norms preserved.
- **Units:** unstructured masks at 30 % and 50 % eligible sparsity, 128 per rung. 10/20 % are dropped as primaries (likely under the 0.00487-nat WikiText half-width). Masks vary at matched count by
  per-layer allocation (registered seeded Dirichlet); criterion (magnitude, Wanda, OBS, random, 50/50 mixtures); module class (MLP, DeltaNet, full attention). Families are split evenly between halves, so no predictor is scored only on its own masks.
- **Ground truth:** realised TF-v1 WikiText ΔNLL (`configs/eval_tf_v1.yaml`); MMLU descriptive; 8 masks rescored for the floor.
- **Predictors** (all R0, via `access.R0Guard`): magnitude; Wanda / q_A (frozen activation factor); OBS w²/[H⁻¹]ᵢᵢ; K-FAC Fisher a + q over five objectives (`response_field.contract`). **Learned ranker:** ridge or GP on the same per-layer features plus pairwise layer interactions, at matched capacity (r13 R4(iv)).
- **Primary metric:** held-out Δτ = τ(ranker) − τ(stack), averaged within sparsity, with a 4,000-draw paired bootstrap over masks stratified by rung.
- **Secondaries:** top-10 % precision; cross-fitted residual correlation with y − stack; transfer to 32 Qwen3.5-4B masks (descriptive).
- **Power** (CPU Monte Carlo, Gaussian copula, 1,500 reps, seed 20261123). Assume ρ(stack) 0.7, ρ(WM) 0.8 and ρ(stack, WM) 0.85, so E[Δτ] ≈ 0.10. Power is then 0.43, 0.67 and **0.81** at 60, 90 and 120 held-out masks. At ρ 0.8 / 0.85 / 0.9 (Δτ ≈ 0.06) it is only 0.50 at 120. The residual test (Fisher z) needs n = 124 for ρ = 0.25. So 128 held-out masks detect Δτ ≈ 0.10 and nothing smaller. These correlations are **assumptions**, since we have no 0.8B pruning data; the pilot measures ρ(stack) first.
- **Cost:** scoring 256 × 1.04 min = 4.4 GPU-h; construction ≤ 1; fields (or identity-checked reuse) ≤ 1; pilot 0.3. Total about **7 GPU-h, cap 8**, $0. M′ ran at 1.5–2.4× its Gate 1 estimates on the shared host, so stop-and-reprice applies.
- **Kill rules:**
  - **K-P1 (pilot):** stack τ ≥ 0.90 in both rungs: no headroom; report F9-P descriptively, stop.
  - **K-P2 (pilot):** replicate floor ≥ half the within-rung ΔNLL spread: stop.
  - **K-P3:** Δτ lower bound ≤ 0: "no world-model value over first-order saliency".

## Q4. U1: targeted unlearning study

- **Data (CPU, seeded, synthetic only):**
  - **200 fictitious people.** Names are combinatorial and rejected if the base model completes any attribute above chance.
  - **20 attributes each:** 10 biographical (birth year; fictitious city, employer and occupation) and 10 synthetic medical (diagnosis and medication from fixed lists, dose, allergy, blood type, surgery year, fictitious clinic and physician, one lab value).
  - **Four phrasings per attribute:** two for training, one held-out paraphrase, and one held-out alternative format (reverse or indirect question, or 4-way multiple choice).
  - **Hard neighbours** share half of a target's values. Nothing real or scraped; records stay under `data/`.
- **Insertion:** full fine-tune of a local Qwen3.5-0.8B with embeddings frozen and 8-bit or bf16 Adam states, which fits in 16 GB. LoRA insertion is rejected: it would hand the predictors the location.
  - **Gate U-0:** training-phrasing accuracy ≥ 80 %, held-out paraphrase ≥ 60 %, and the base at chance. If 0.8B fails, fall back to Qwen3.5-4B with all-module LoRA, merged and labelled.
- **Split:** 40 dev targets (fit the localiser, freeze k), 40 eval targets, 40 neighbours (scored only), 80 in the retain-loss pool.
- **Episode:** forget one person by NPO + retain-KL for T steps, restricted to weight set S and rescaled to fixed ‖ΔW‖; zeroing S is descriptive.
- **Arms at equal |S| = k and equal norm:**
  1. **First-order localiser:** top-k of |g_forget ⊙ W| divided by the retain Fisher diagonal.
  2. **Learned localiser (the world model):** trained on dev episodes' realised outcomes for layer-group-masked edits.
  3. **Mandatory baseline:** pooled dev layer profile plus generic saliency within layers.
  4. Uniform random.
  5. **Saliency-matched random:** same per-layer counts and the same Wanda- and magnitude-decile histogram as arm 1.
  6. **Unlocalised reference:** NPO on all weights at matched norm.

  Task-vector negation reads the insertion delta: **oracle** arm (R2) only; θ_pre-insertion is never R0.
- **Primary:** held-out paraphrase forget accuracy (teacher-forced multiple-choice log-prob); arm 1 vs arms 4 and 5, paired over 40 eval people, Holm. **World-model test (secondary):** arm 2 vs arms 3 and 1.
- **Retain guards:** hard-neighbour accuracy ≥ arm 5's minus 3 pp; WikiText ΔNLL and an MMLU subset non-inferior to arm 6.
- **Suppression vs forgetting** ("forgotten" needs all three): a relearning attack (5 steps on half the attributes, score the other half), cross-format recovery, and a linear probe trained on retain people's residual stream.
- **Power** (paired normal approximation; per-person SD = binomial noise plus an assumed between-person SD), 10 pp at power 0.8:

  | facts per person | between-person SD | n, α 0.05 | n, Holm α 0.025 |
  |---|---|---|---|
  | 10 | 0.15 | 58 | 70 |
  | 20 | 0.15 | 39 | 47 |
  | 20 | 0.10 | 29 | 35 |

  With 20 facts, 40 eval people suffice if the between-person SD ≤ 0.13; dev re-estimates it and re-sizes before eval opens.
- **Cost (unmeasured estimate; the pilot prices it):** insertion 1–2 GPU-h; 480 episode-runs (80 people × 6 arms) at about 3 min each including the relearning attack, about 24; pilot ≤ 3. Total about **30 GPU-h, cap 36**, $0.
- **Kill rules:**
  - **K-U0:** insertion fails Gate U-0 on both 0.8B and the 4B fallback.
  - **K-U1 (dev):** arm 1 beats arm 5 by < 5 pp: localisation adds nothing; report descriptively.
  - **K-U2 (dev):** arm 2 ≤ arm 3: drop the world-model arm.
  - **K-U3:** relearning recovers ≥ 80 % of base accuracy in every arm: suppression only, no forgetting claim.

## Reuse vs new

**Reused:** `response_field` (factors, `contract`, `cross_term`), `c1/field.RCache`, `access.R0Guard`, `wmm/provenance`, `wmm/canary` (sketch, batch-1 canary), `wmm/lora_micro` (4B fallback), and from `mprime` the Ledger, `run_schedule`, BatchRegistry, `random_direction` and `scale_to`; the TF-v1 harness.

**New:** Wanda, OBS and SparseGPT code (none in `src/`; F9a.4 recipes), the mask generator, the person generator, full-FT insertion, masked NPO, relearning and probe evaluations, the saliency-matched sampler, and the stack and ranker.

## Confounds

- **Dose:** equal count ≠ equal effect; control by norm matching, saliency-matched nulls and a dev curve over k ∈ {0.01, 0.1, 1 %}.
- **Invariances:** Qwen's QK-norm makes q/k rescales near-inert; Gemma post-norms cancel writer rescales (Gate 0: +0.010 vs +12.27 nats), so on Gemma ‖ΔW‖ is not dose and deletion ≠ rescale.
- **Non-determinism:** batch composition flips 13.5 % of MedGemma greedy answers; use teacher-forced scoring, fixed batches and a measured 0.8B rerun floor.
- **Insertion artefact:** fine-tuned facts may be more local than pretrained ones; no privacy guarantee for real data follows.
- **Construction-predictor coupling:** criterion families are balanced across the splits.

## Decisions

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Qwen3.5-0.8B for both studies | Apache-2.0, pre-norm, fits ≤ 16 GB, measured 1.04 min/entry; 9B is 18 GB and is C1's; MedGemma has post-norm, 13.5 % flips, HAI-DEF | ARCHITECTURES/MODELS check | 0, $0 | 4B only as the U1 insertion fallback |
| First-order stack = mandatory baseline | q_A ≡ Wanda; r13 dose lesson | fitted on training masks only | 0 | — |
| P1: 256 masks, held-out Δτ | n = 120 gives power 0.81 at Δτ ≈ 0.10 | pilot of 16 masks, then primary | ~7, cap 8, $0 | K-P1 τ ≥ 0.90; K-P2 floor; K-P3 LB ≤ 0 |
| U1: 200 synthetic people, 40 eval episodes | 40–47 people detect 10 pp with Holm | Gate U-0 insertion; dev pilot | ~30, cap 36, $0 | K-U0–K-U3; report suppression only |
| Predicted set vs random and saliency-matched at equal k and norm | separates localisation from dose and magnitude | paired Holm contrasts | inside U1 | arm 1 < arm 5 + 5 pp on dev: stop |
| Task-vector negation as oracle only | reads the insertion delta (R2) | access tier label | inside U1 | — |
| Order: C1 → P1 → U1; nothing on Lambda | C1 priority; everything fits local | user approval of each prereg | — | any overrun: stop and reprice |
