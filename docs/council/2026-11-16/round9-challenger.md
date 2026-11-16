# Round 9 Challenger-Engineer Report: Adversarial Red-Team Audit of WM-E

## 1. Bottom Line Up Front (BLUF)
1. WM-E v0 failed all three core gates in Pilot 1 (Go-T 0.496, Go-R -4.8%, Go-E 1/24) because local weight edits on 9B behave quadratically inside trust regions and chaotically outside them.
2. Trajectory claims risk being an artefact of bookkeeping leakage: the step operator receives exact realised doses and overlaps ($), allowing an MLP to memorize cumulative dose without learning real dynamics.
3. Left-point slope transport degraded accuracy (-4.8% error cut); the trapezoid slope cut (+34.8%) is an expensive oracle illusion requiring 61s/state and disk-bound tensor rehashing that inflates bank cost to 24.1 GPU-h.
4. The headroom gate failed on math (0/24 qualifying) because teacher-forced gold log-probability provides zero headroom on instruct models and fails to transfer to autoregressive generation.
5. 21 of 24 candidate edits breached guardrails; optimizing tiny log-probability shifts causes catastrophic refusal collapse or reasoning loop degeneration undetected by small dev panels.
6. Comparing WM-E against zero-shot WSR readers tests against a straw man; Kocabay (2026) showed unassisted LLMs predict ghost behavioral changes for 53-88% of completely inert edits.
7. A learned multi-agent orchestrator like Sakana Fugu (TRINITY/Conductor) coordinating Thinker/Worker/Verifier roles with access lists would decisively outperform WM-E at equal inference budget.
8. A direct best-of-N search with multi-fidelity gating uses zero surrogate training compute and avoids all world-model approximation errors, dominating WM-E when bank construction costs are amortized honestly.
9. Decisive falsification requires a shuffled-bookkeeping ablation, an autoregressive generation audit, and an unamortized compute-matched contest against direct evaluation and activation steering.
10. A skeptic will only be convinced if WM-E beats a cumulative static baseline by >= 2.0 SE on generation accuracy on held-out test splits with zero bookkeeping leakage and zero guardrail degradation.

## 2. External Evidence & Literature Verification
In accordance with council instructions, three primary external sources were fetched and analyzed:
- **Sakana Fugu Product & Architecture** (https://sakana.ai/fugu/): Learned multi-agent orchestration via a single API. Fugu coordinates specialized models to achieve frontier performance on agentic benchmarks (e.g. SWEFish, GDP.pdf) without manual workflow engineering. Fetched successfully.
- **Sakana Fugu Technical Report** (arXiv:2606.21228, https://arxiv.org/abs/2606.21228): Details two-stage training: SFT on temperature-scaled soft targets followed by sep-CMA-ES evolution. Fugu-Ultra builds on Conductor (arXiv:2512.04388) and TRINITY (arXiv:2512.04695), training an RL orchestrator (GRPO) to generate workflows with strict access lists to prevent orchestration collapse. Fetched successfully.
- **Weight-Space Reading (WSR) in Language Models** (Kocabay, 2026, alphaXiv:2611.weight-space-reading-language-models, https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models): Audits weight reading across 9B–30B models. Findings: (1) no conclusive self-reading advantage (+0.085 to +0.114 CI crosses zero); (2) unassisted readers hallucinate behavioral changes for 53–88% of inert edits (e.g. 40 null interventions on Muse-30B); (3) destructive benchmarks measure damage detection rather than semantic reading. Fetched successfully.
- **Fetch Failures:** None. All three URLs were fetched and parsed with zero errors.

## 3. Adversarial Red-Team Audit of WM-E Claims & Pilot-1 Artifacts

### 3.1 The Trajectory Gate (Go-T) & Bookkeeping Leakage
In Pilot 1 (docs/reviews/wme/wme-v0-pilot-2026-11-16.md), Go-T failed: pooled resolvability was 0.496 (< 0.50), and static rollout error at h=3 was 1.91 SE (< 2.0 SE). Per docs/council/2026-11-16/RULING-wme-pilot1.md: **static cumulative prediction suffices**.

The trajectory hypothesis suffers from three structural flaws:
1. **Quadratic Dominance in the Trust Region:** Inside the stable edit radius, loss surfaces are well-approximated by second-order Taylor expansions. In src/wsp/wm/model.py, WM-E nests upon raw response field F(E) = <g0, E> + 0.5 E^T H0 E. Because this field is exact for quadratic dynamics, the learned latent step operator h_t = h_{t-1} + f(h_{t-1}, xl_t, e_t) has negligible non-linear dynamics to model.
2. **Bookkeeping Leakage:** In src/wsp/wm/model.py, the step operator receives step-local bookkeeping xl_t (realised norm D_t and signed overlap with E_{t-1}), while static receives cumulative bookkeeping. Any apparent WM-E gain over static is an artifact of xl_t shortcutting composition order rather than modeling latent weight dynamics.
3. **Selection Bias in Fresh Roots:** Discarding pilot-1 roots and widening ladders in wme-p2 (RULING-wme-pilot1.md) is a post-hoc search for non-linearities. Forcing models outside the trust region until WikiText Delta NLL spikes (> 1.0 nat/token) tests damage detection, not trajectory modeling.

### 3.2 The Slope-Transport Gate (Go-R): Oracle Illusion & Systems Breakdown
Pilot 1 revealed a slope transport failure: left-point transport produced a **-4.8% cut** (increasing error relative to A0). The council pivoted to a post-hoc trapezoid slope cut of +34.8% (docs/reviews/wme/wme-v0-pilot-2026-11-16.md).

This trapezoid result is an unviable oracle illusion:
- **Prohibitive Cost:** Trapezoid transport requires true intermediate action derivatives A_t = <grad y(W_t), E>. Collecting oracle gradients in P4 took 61s/state, inflating bank cost from 7.84 to **24.07 GPU-h** (docs/reviews/wme/wme-v0-pilot-2026-11-16.md).
- **Disk I/O Thrashing:** P4 froze initially because re-hashing intermediate checkpoints at 100 MB/s thrashed ~109 GB of disk I/O (docs/reviews/wme/wme-v0-pilot-2026-11-16.md).
- **Circularity in Planning:** At R0, intermediate states W_t are unexecuted. WM-E must predict hat{A}_t using C(h_t). If h_t drifts, predicted slopes compound rollout error, accelerating divergence.

### 3.3 Headroom (Go-E), Log-Probability Gaming, and Guardrail Collapse
Pilot 1 showed a headroom blackout: 0/24 candidates qualified on GSM8K, 0/24 on MMLU-Pro, and 21/24 breached registered guardrails (docs/reviews/wme/wme-v0-pilot-2026-11-16.md).

1. **Log-Probability vs Generation:** In instruct models, gold cross-entropy is saturated. Log-probability does not transfer to autoregressive generation: an edit improving gold log-prob by 0.01 nats often destroys generation entropy, inducing loops or refusal collapse.
2. **Guardrail Gaming:** The only candidate passing Pilot 1 was on sycophancy (1/24). Yet sycophancy reductions via steering or projection removal (src/wsp/wm/actions.py) degrade instruction adherence or trigger safety collapses. With 21/24 candidates breaching guardrails, surrogate search is actively hazardous.
3. **Boundary Leaks:** While src/wsp/wm/plan.py implements boundary checks via PlannerItemStore, task vectors trained on construction data can leak behavioral biases if problem templates overlap.

### 3.4 The WSR Comparison: Straw-Man Baselines vs Frontier Multi-Agent Systems
The planned E2/E3 contrast against Weight-Space Reading (SYNTHESIS-r8.md D7) is flawed:
1. **Straw-Man Baseline:** As shown by Kocabay (2026) (https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models), zero-shot LLM readers have high false-positive rates, predicting changes for 53% to 88% of inert edits. Comparing WM-E against an unassisted reader benchmarks against an uncalibrated straw man that hallucinates non-existent weight effects.
2. **True Competitor: Multi-Agent Orchestrators:** The real benchmark is a Sakana Fugu-style orchestrator (https://sakana.ai/fugu/, arXiv:2606.21228). Fugu-Ultra coordinates diverse LLMs through learned scaffolds with access lists (Conductor) and evolutionary optimization (TRINITY sep-CMA-ES). A Fugu system deploying specialized agents (Weight-Inspector, Gradient-Statistician, Adversarial-Verifier) would out-reason WM-E's 32-dim latent vector without requiring fragile surrogate dynamics.

## 4. Radically Different Skeptic Alternatives (at Equal Budget)

A skeptic asserts that WM-E's latent trajectory architecture (h_t in R^{32}) is unnecessary. At equal budget, two radical alternatives will decisively beat WM-E:

### 4.1 Alternative A: Multi-Fidelity Direct Best-of-N Search (Zero World Model)
- **Mechanism:** Eliminate the world model entirely. Allocate the bank construction budget (7.84 to 24.07 GPU-h) directly to real target evaluations.
- **Two-Stage Multi-Fidelity Filter:**
  - *Stage 1 (Screening):* Evaluate candidate edits on 4 fast items (latency ~1.2s). Discard any candidate with negative gain or WikiText degradation.
  - *Stage 2 (Verification):* Fully evaluate surviving candidates on the 32-item development panel (23.7s/state, docs/reviews/wme/wme-v0-pilot-2026-11-16.md).
- **Budget Justification:** The 24 GPU-h required for the oracle bank buys over **3,600 real state evaluations**. Direct search over small-LoRA task vectors (lambda in {0.25, 0.5, 1.0}) with 3,600 real evaluations will uncover far superior, verified optima than an R0-guided planner navigating a biased surrogate surface.

### 4.2 Alternative B: Dynamic Activation Steering with Contrastive Gating
- **Mechanism:** Abandon weight editing. Modifying 28 GB of bf16 weights in place causes parameter damage and requires complex restoration mechanics (src/wsp/wm/actions.py EditSession).
- **Approach:** Apply inference-time activation steering at layers >= 20, gated dynamically by a lightweight contrastive linear probe on the residual stream.
- **Advantages:** Zero disk I/O overhead (avoids P4 disk rehashing), perfect base capability preservation when inactive, and instantaneous rollback without weight checkpointing.

### 4.3 Alternative C: Fugu-Style Learned Multi-Agent Orchestrator
- **Mechanism:** Deploy an evolved/RL orchestrator structured like TRINITY (arXiv:2512.04695) and Conductor (arXiv:2512.04388): Thinker proposes edit hypotheses from gradient statistics; Worker realises rank-one or LoRA directions; Verifier enforces access lists and evaluates guardrails on held-out safety prompts.
- **Fair E2 Entry:** Under src/wsp/wm/plan.py BudgetLedger, grant Fugu the identical budget of 32 real evaluation bundles. Leveraging semantic reasoning over weight summaries rather than a brittle 32-dim latent vector, Fugu will generate higher-quality candidate edits while maintaining strict guardrail adherence.

## 5. Decisive Falsification Experiments & Conviction Threshold

The council must execute these decisive, low-cost falsification experiments:

### 5.1 Shuffled-Bookkeeping and Inactive-Weight Null Test
- **Protocol:**
  1. *Shuffled Bookkeeping:* On held-out compositions (h=3, 4), shuffle step bookkeeping xl_t and edit order while holding cumulative edit E_t fixed. If WM-E prediction error does not significantly worsen (p > 0.05), WM-E has learned zero trajectory dynamics and acts as a static cumulative lookup.
  2. *Null / Dead-Neuron Edit Control:* Apply rank-one edits to zero-activation features or weights cancelled by post-sublayer RMSNorm (as in Muse-30B, Kocabay 2026). If WM-E predicts non-zero behavioral shifts, it suffers from ghost-prediction hallucinations.
- **Cost:** < 0.2 CPU-h.

### 5.2 Autoregressive Generation Audit
- **Protocol:** Take top-3 candidate edits selected by WM-E on development log-probability objectives. Run full autoregressive generation (temperature 0.7, 512 tokens, 64 items) across GSM8K and Sycophancy.
- **Falsification Metric:** If generation accuracy fails to improve over the unedited base, or if repetition loops / refusal anomalies increase by > 2%, the log-probability objective is falsified as an optimization target.
- **Cost:** ~0.4 GPU-h.

### 5.3 Unamortized Compute-Matched Arm Comparison
- **Protocol:** Enforce honest accounting in src/wsp/wm/plan.py: charge the world-model arm with full bank construction cost (7.84 to 24.07 GPU-h). Provide the Direct Best-of-N arm with an equivalent GPU-h budget for direct evaluation.
- **Falsification Metric:** If Direct Best-of-N matches or exceeds WM-E test gain, the world model is economically unviable.

### 5.4 What Result Would Convince a Skeptic?
A skeptical engineer will concede WM-E represents a genuine breakthrough if and only if four conditions are met simultaneously:
1. **Trajectory Superiority:** On held-out compositions (h >= 3) with non-quadratic curvature, WM-E beats the refitted-F3u static baseline by >= 2.0 SE in paired evaluation, with shuffled-bookkeeping controls confirming the gain stems from dynamic state transitions.
2. **Generation Transfer:** Candidate edits selected by WM-E demonstrate statistically significant (p < 0.01) improvements in *end-to-end autoregressive generation accuracy* on held-out test splits.
3. **Zero Guardrail Degradation:** Winning edits maintain WikiText Delta NLL <= 0.01 nats/token and zero statistically significant degradation on harmful/harmless refusal margins.
4. **Economic Margin:** WM-E outperforms an unamortized compute-matched Best-of-N direct evaluation arm at equal total GPU hours.

## 6. Summary Table

| proposal | mechanism | test | cost | kill criterion |
|---|---|---|---|---|
| Shuffled-Bookkeeping Null Test | Permute xl_t and step order while fixing cumulative E_t | Paired prediction error difference on held-out 3-step and 4-step compositions | 0.2 CPU-h | WM-E error does not degrade (p > 0.05); confirms dynamics are bookkeeping artifacts |
| Dead-Neuron / RMSNorm Invariant Control | Apply edits to inactive rows and post-norm scale-cancelled weights | Evaluate WM-E predicted shift hat{y} on known zero-impact interventions | 0.1 CPU-h | Predicted shift |hat{y}| > 2 sigma; confirms ghost-prediction hallucination |
| Autoregressive Generation Audit | Decode 64 held-out prompts with top WM-E candidate edits | Measure greedy/sampled accuracy and repetition/refusal anomalies against base | 0.4 GPU-h | Generation accuracy gain <= 0 or loop/refusal anomaly rate > 2% |
| Multi-Fidelity Direct Best-of-N Arm | 2-stage screening (4-item quick filter -> 32-item panel) without world model | Head-to-head capability gain on held-out test split against WM-E | Matched to bank GPU-h (~7.8h) | Direct search achieves equal or superior test gain to WM-E |
| Fugu-Style Multi-Agent Benchmark | TRINITY-style Thinker/Worker/Verifier agents with access lists | E2 contrast under identical BudgetLedger (32 real-eval bundles) | Zero GPU overhead (API-driven) | Fugu multi-agent candidate selection matches or beats WM-E test gain |
| Unamortized Bank Cost Gate | Include 7.84–24.07 GPU-h bank construction in WM-E total budget | Net capability gain per total GPU-h consumed across all arms | Accounting only | WM-E net efficiency (gain / total GPU-h) < 1.0x direct evaluation |
