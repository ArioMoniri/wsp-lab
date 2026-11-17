# Round 10, systems seat (17 Nov 2026): Fugu-like systems for slope transport and headroom

**Read:** BRIEF-r10; r9 addendum, pro-engineer, challenger, SYNTHESIS-r9; `docs/reviews/wme/wme-pilot2-2026-11-16.md`, `wme-arms-2026-11-17.md`; RULING-wme-v2-design, v2b; `src/wsp/wm/`. No GPU, no `data/`, no sealed split, no commit. Everything below is **post hoc**. It is declared here and tested only on fresh p3 steps or fresh dev panels (new salt).

## BLUF

1. **Slope (a).** The pooled pilot-2 cut is mostly harmful refusal (0.79); sycophancy is −0.26 (pilot-2 §4). **P1** replaces the single global step rule with an R0 router that picks a rule per (objective, family), in the manner of Fugu's selection head and RouteLLM.
2. **P2** goes past C4(b1) with R1 third-order directional jets at θ₀ plus a dose Padé for saturating margins.
3. **Headroom (b).** The strongest pilot-2 LoRAs breached **only** the harmless-refusal guard (pilot-2 §5). **P4** is a guard-orthogonal layer-group merge family, searched CycleQD/Model-Swarms-style inside R0.
4. **P5** is STaR-style verified self-distillation on construction items. It scores generated accuracy, not gold log-probability.
5. **Planning.** **P3** is BO with the WM as prior mean and multi-fidelity item screens, shared by both primary arms. An outcome-only BO enters as a competitor.
6. **P6** replaces the fixed mutation fraction with AB-MCTS adaptive branching (TreeQuest).
7. **Competitor.** **P7** evolves edit recipes with an LLM (ShinkaEvolve/AlphaEvolve-style). It proposes candidates, whereas the Fugu team only ranks them.
8. **P8** is a COMs-style conservative surrogate.
9. **Rejected:** Text-to-LoRA, DnD as a generator, VeLO, DGM/AI Scientist, tangent-space fine-tuning (§4).
10. **The swarm and evolution methods I fetched all used ≫ 32 real evaluations**, so they run inside R0 and never on the ledger.

## 1. External evidence (all fetched; abstract pages without numbers contribute no numbers)

| System | Verified claim |
|---|---|
| Evolutionary Model Merge (https://arxiv.org/html/2403.13187) | **Search:** CMA-ES over DARE-TIES density/weight; 1,000 trials; fitness on 1,069 JA-GSM8K items. **Result:** MGSM-JA 52.0 vs sources 9.6/18.4/30.0. Per-layer configs gave "no clear gains" |
| CycleQD (https://arxiv.org/abs/2410.14735) | **Method:** each task metric is quality in turn; the others are behaviour descriptors. Merge crossover, SVD mutation. **Model:** LLaMA3-8B |
| Model Swarms (https://arxiv.org/html/2410.11163) | **Method:** personal-best, global-best and global-worst moves. **Setup:** 10 LoRA experts on Gemma-7B; N = 20; ≤ 50 iterations; 200-item utility |
| Text-to-LoRA (https://arxiv.org/html/2506.06105) | **Training:** 479 tasks. **GSM8K:** base 40.9, multi-task LoRA 47.5, T2L 45.2–47.3 |
| Drag-and-Drop (https://arxiv.org/html/2506.16406) | **Training:** checkpoints saved every step; Qwen2.5. **1.5B, generated vs mean training LoRAs:** GSM8K 42.9→66.3; MATH 14.8→23.9 |
| Transformer² (https://arxiv.org/html/2501.06252) | **Method:** few-shot CEM over z-mixtures. **Llama3-8B MATH:** 24.54 → 25.47 |
| AB-MCTS (https://arxiv.org/abs/2503.04412); TreeQuest (https://github.com/SakanaAI/treequest) | **AB-MCTS:** wider/deeper branching beats repeated sampling and MCTS (no numbers). **TreeQuest:** Apache-2.0; ask_batch/tell; batch ≤ 5 advised |
| ShinkaEvolve (https://arxiv.org/abs/2509.19349) | **Method:** novelty rejection, parent sampling, bandit over LLMs. **Result:** SOTA circle packing in 150 samples |
| AlphaEvolve (https://arxiv.org/abs/2506.13131) | **Method:** LLM code edits plus evaluators. **Result:** 4×4 complex matmul in 48 multiplications |
| Multi-fidelity merging (https://arxiv.org/abs/2502.04030) | **Result:** merges found in "< 500 search steps" using multi-fidelity approximations |
| RouteLLM (https://arxiv.org/abs/2406.18665) | **Method:** learned strong/weak router. **Result:** ">2x" cost savings |
| COMs (https://arxiv.org/abs/2107.06882) | **Method:** a surrogate that underestimates out-of-distribution designs, against exploitation |
| STaR (https://arxiv.org/abs/2203.14465) | **Method:** fine-tune on self-generated rationales that reach correct answers |
| VeLO (https://arxiv.org/abs/2211.09760); DGM (https://arxiv.org/abs/2505.22954); AI Scientist (https://arxiv.org/abs/2408.06292); tangent-space task arithmetic (https://arxiv.org/abs/2305.12827) | **VeLO:** ≈ 4,000 TPU-months of meta-training. **DGM:** SWE-bench 20.0 → 50.0%. **AI Scientist:** < $15 per paper. **Tangent space:** linearised fine-tuning |

**Fetch ledger.**
- No fetch failed.
- AP-BMM (2512.09972) was seen in search results only and is not relied on.
- FunSearch was not fetched.
- The AB-MCTS posterior is unverified.

## 2. Slope transport (a)

### P1. A step-rule router (Fugu Stage-1 / RouteLLM analogue; R0)

**Problem.**
- Pilot-2 cuts range from 0.79 (harmful) to −0.26 (sycophancy), with GSM8K at 0.04 (pilot-2 §4).
- Pilot-1's left-point rule cut −4.8% (round9-challenger §3.2).
- A single global rule averages cells where transport helps with cells where it hurts.

**Mechanism.** A multinomial-logistic router chooses among {A0 static step, C4(b1) HVP slope, C4(b2) learned slope, P2 jet slope} for each step.
- **Inputs (R0 only):** objective id, family id, h, recipe norms, field a/q, and |E_{t−1}·D_t|.
- **Targets:** softmax(−|err|/SE_step), i.e. r9 W1's τ = SE soft target applied to rules.
- **Abstaining from transport is allowed.** That is what fixes sycophancy.

**Test (fresh p3).**
- Train on h ≤ 2. Test on h ≥ 2 in held-out closure components (`wm.uncertainty.component_selection_split`).
- **Pass:** a cut of ≥ 20% vs A0, **and** above the best single rule with component sign-flip LB > 0.
- Also report the result with harmful refusal excluded.

**Kill.**
- The routed cut is ≤ the best single rule's.
- Any objective is worse than A0 with LB > 0.
- If the router chooses by family alone, a fixed table replaces it.

**Cost.** CPU. It reuses C3/C4 p3 measurement (≤ 2 GPU-h, already declared).

### P2. Third-order directional jets at θ₀ (R1)

**Mechanism.** The end slope ⟨g(θ_t), D_t⟩ is R2. Expanding it at θ₀ gives ⟨g0, D⟩ + DᵀH0E_t + ½T0[E_t, E_t, D].
- The terms are nested forward-over-reverse JVPs at θ₀ along recipe directions known before any edit runs, so this is R1.
- C4(b1) stops at H0. The refusal margins carry the cut and saturate, so a [1/1] dose Padé with a pole guard is fitted to the jet coefficients.

**Test.** The incremental cut over C4(b1) on the same fresh p3 steps.

**Kill.**
- The incremental cut is < 10%.
- Poles fall inside the visited doses on > 5% of steps.
- Preflight memory exceeds 28 GB. Preflight runs on 0.8B construction items first (H4).

**Cost.** Unmeasured, so capped at ≤ 1 local GPU-h.

## 3. Headroom (b), planning and competitors

### P4. Guard-orthogonal layer-group merges, searched by quality diversity in R0

**Fact (pilot-2 §5).**

| LoRA | Gain | LCB | Only breach |
|---|---|---|---|
| Math λ 0.5 | 6.01 | 4.53 | `refusal_margin_harmless` |
| Math λ 1 | 6.34 | 4.70 | `refusal_margin_harmless` |
| Sycophancy λ 0.25 | 2.89 | 2.43 | `refusal_margin_harmless` |

Under value-scale guards, GSM8K would have 1 qualifier.

**Family.** Δ = Σ_g λ_g·LoRA_obj|_g + ν·proj_remove(r_harmless).
- The sum is over 4 layer groups.
- LoRA restriction to a group is free (`wm.lora.lowrank_actions`).
- r_harmless is a construction-only harmful-minus-harmless contrast per writer.
- That gives about 6 coordinates. Evolutionary Merge found finer granularity unhelpful.

**Search.**
- A MAP-Elites/CycleQD archive in R0. Quality is the WM-predicted LCB.
- Descriptors are the predicted harmless-refusal pp and WikiText ΔNLL.
- Moves follow Model Swarms.
- Only elites spend real bundles.

**Test (fresh dev panel).** 32 elites each for GSM8K and math.
- ≥ 2/32 must qualify under **both** pp and pilot-1 value-scale guards.
- The best guarded elite must beat the best guarded single candidate.
- The C2 generation audit is required.

**Kill.** Either test fails, or anomalies exceed 2%.

**Cost.**
- 64 bundles × 32.96 s (`wme-powersim-pricing-2026-11-17.json`) = 0.59 GPU-h.
- LoRA retrains: 138 s + 324 s (pilot-2 §5).
- Total ≈ 0.72 local GPU-h (computed).

### P5. Verified self-distillation (STaR-style; construction only)

**Mechanism.**
- Qwen3.5-9B samples solutions to construction GSM8K/MATH items with `enable_thinking=False`. Its own thinking-on traces may serve as teacher, so it stays a single LLM.
- Answer-verified samples are kept, and a rank-8 LoRA is trained on them.
- Every step is checkpointed: DnD found better points than the mean training LoRA.

**What it measures.** Generated exact-match accuracy, which C2 requires, instead of "The answer is X" log-probability. It is a cheaper SFT stage before C5(4) RL.

**Test.** Fresh dev. 32 (λ, checkpoint) candidates chosen on construction-validation. ≥ 2/32 must be guarded with accuracy LCB > 0.

**Kill.**
- Fewer than 2 qualify.
- Anomalies exceed 2%.
- Gains occur only on truncated items. Sycophancy hit max tokens on 7/8 in `wme-ph100-2026-11-17.json`.

**Cost.** The CoT length is unmeasured, so the cost is a cap: ≤ 3 H100-h ($14.67 at $4.89/h, `wme-pipeline-2026-11-17.md` §7), plus ≤ 0.3 GPU-h training.

### P3. WM-prior multi-fidelity BO (shared planner, plus a competitor)

**Mechanism.**
- A GP over shared R(E) models the residual y − ŷ_arm, with the arm's predictor (WM or StatNN) as prior mean.
- Fidelities are 4 → 32 → full-dev items, with successive halving (https://arxiv.org/abs/2502.04030).
- This transfers real outcomes **across candidates**; C7's filter conditions on taps.
- The code is identical in both arms, which keeps r9 W3 fairness.

**Competitor.** An outcome-only, SMAC-style random-forest BO with no field access joins family D.

**Precondition (pre-freeze ruling).** `BUNDLE_DEFINITION` must charge partial fidelities pro rata.

**Test.** `wm.planner.synthetic_regret` at B = 32, then descriptive fresh-dev episodes.

**Kill.** ≤ the C6 planner.

**Cost.** About 2 CPU-h.

### P6. AB-MCTS outer policy

**Mechanism.**
- "Wider" means a fresh pool draw. "Deeper" means mutating an incumbent.
- This replaces the fixed MUTATION_FRACTION (`arms/pool.py`), applied identically to every arm.
- Rounds are 8 bundles, while TreeQuest advises ≤ 5, so test both 8 and 2 × 4.

**Kill.** No synthetic regret gain over 0.5.

**Cost.** CPU.

### P7. LLM-evolved edit recipes (competitor arm)

**Mechanism.**
- An LLM writes short programs that map the R0 descriptions (`arms.pool`) to new recipes within the primitive library.
- It uses ShinkaEvolve-style novelty rejection and parent sampling.
- Fitness comes from **real ledger evaluations only**.
- Access is declared like `fugu_team`.

**Expectations.** ShinkaEvolve needed 150 samples against our 32, so the arm is likely under-budgeted. Report it either way.

**Cost.** Tokens. A team round is ≈ 6.6–9k (`wme-arms-2026-11-17.md` §3). Probe-price P7 rounds before setting the cap.

### P8. COMs surrogate

**Mechanism.** Penalise predictions on surrogate-ascent edits, in both arms.

**Kill.** No synthetic regret gain over the support penalty alone.

**Cost.** CPU.

## 4. Rejected

| System | Reason |
|---|---|
| Text-to-LoRA | Its value is amortisation over 479 tasks, and on GSM8K it does not beat multi-task LoRA (47.3 vs 47.5). We have 3 objectives |
| DnD as a generator | It needs a large checkpoint corpus and has only been shown at ≤ 7B Qwen2.5. Its checkpoint lesson is kept in P5 |
| Transformer² mixing | +0.93 MATH, and it is already C7 W2 |
| Evolutionary Merge / Model Swarms on the ledger | 1,000 trials, or 20 × 50 iterations, against 32 bundles. R0 only (P4) |
| VeLO | Its meta-training cost is out of reach, and it has no prediction or planning role |
| DGM, AI Scientist | They are research-automation harnesses, not E2 arms |
| Tangent-space fine-tuning | It is evaluated via a JVP, not as a plain θ₀ + Δ edit |

## Summary table

| proposal | mechanism | access tier | test on fresh data | cost (GPU-h, $) | kill criterion |
|---|---|---|---|---|---|
| P1 step-rule router | Soft-target router over {A0, HVP, learned, jet} per (objective, family) | R0 | Fresh p3, h ≤ 2 → held-out components; ≥ 20% and > best rule, LB > 0; also without harmful | CPU (reuses C3/C4 ≤ 2 GPU-h) | ≤ best single rule; any objective worse than A0 |
| P2 directional jets | Nested-JVP third-order slope at θ₀ + dose Padé | R1 | Fresh p3, increment over C4(b1) | ≤ 1 local GPU-h cap | < 10%; poles > 5%; > 28 GB |
| P4 guard-orthogonal QD merges | Layer-group LoRA λ_g + proj_remove(harmless), CycleQD/Swarm in R0 | R0 search; elites real | Fresh dev: ≥ 2/32 under both guard scales; beats best single; generation audit | ≈ 0.72 local GPU-h | Either fails; anomalies > 2% |
| P5 verified self-distillation | STaR-style LoRA on verified construction samples; per-step checkpoints | Edit family | Fresh dev generated accuracy, ≥ 2/32 guarded | ≤ 3 H100-h ($14.67) + 0.3 GPU-h | < 2; anomalies > 2%; truncation-only gains |
| P3 WM-prior MF-BO | GP residual on R(E), arm predictor as mean; 4 → 32 → full | R0 + ledger | Synthetic regret B = 32; descriptive fresh dev | ≈ 2 CPU-h | ≤ C6 planner |
| P3c outcome-only BO | SMAC-style RF + Hyperband, no field | Ledger | Family-D arm, pro-rata bundles | Within ledger | Reported either way |
| P6 AB-MCTS policy | Adaptive wider/deeper | Shared outer loop | Synthetic regret, batch 8 and 2 × 4 | CPU | No gain over 0.5 |
| P7 evolved recipes | ShinkaEvolve-style program mutation | R0 descriptions + own results | Competitor, same ledger, tokens charged | Tokens (probe first) | Reported either way |
| P8 COMs surrogate | Conservative penalty | R0 | Synthetic regret vs support penalty | CPU | No gain |
