# Council Round 10: Challenger-Engineer Report
## Adversarial Red-Team Audit of Slope Transport, Headroom, and Frontier Competitors

**Seat:** Challenger-Engineer (Adversarial Red Team, Independent Vendor Perspective)  
**Worktree:** `/home/ubuntu/projects/wsp-lab/.worktrees/council/2026-11-17-r10/`  
**Binding Rules:** `docs/council/2026-11-17/BRIEF-r10.md`  
**Literature Status:** Fetch failures: None (all external claims verified via fetched URLs).

---

### 10-Line BLUF
1. Both weak results are fatal measurement artefacts of proxy choice and metric scaling, not real limits of weight dynamics.
2. Slope transport pass (34.6% cut) is a pooling illusion driven solely by harmful refusal (0.790 cut); 5 of 6 objectives fail.
3. Curvature cancels in contrastive sycophancy (Δ log P), rendering the R2 trapezoid oracle worse than static A0 (−0.256 cut).
4. Trust regions were never stressed: max ΔNLL was 0.127 nats (vs declared >1.0 nat stop), trapping ladders in linear regimes.
5. Headroom failure on GSM8K/MATH stems from teacher-forcing under enable_thinking=False, forcing zero-shot mental calculation.
6. Literature proves next-token SFT degrades math reasoning (75.9% to 60.5% on 8B); only RL/policy-gradient lifts accuracy.
7. Harmless-refusal 1 pp guardrail on sigmoid scale is mathematically hypersensitive, falsely disqualifying all strong LoRAs.
8. Frontier systems (Transformer² SVF, Evolutionary Merging, Text-to-LoRA, AB-MCTS) crush R0 world models at equal budget.
9. Post-hoc threshold loosening or refusal-dominated pooling are illegitimate rescues that must be barred by hard kill rules.
10. Challenger conviction requires true CoT generation gains (Δacc > 0) and outperforming unamortized Direct Best-of-N search.

---

### 1. Dissecting Weak Results: Physics vs. Measurement Artefacts

#### 1.1 Slope Transport (Go-R2): A Pooling Illusion and Curvature Cancellation
In pilot 1 (`docs/reviews/wme/wme-v0-pilot-2026-11-16.md`), slope transport failed (−4.8%). In pilot 2 (`docs/reviews/wme/wme-pilot2-2026-11-16.md` §4), Go-R2 nominally cut pooled error by 34.6% (2.36 → 1.54 SE). This pass is entirely artefactual:

1. **Curvature Cancellation in Contrastive Objectives:**  
   In `src/wsp/wm/panel.py` (lines 442–446), `sycophancy_real` is $\Delta \log P = \log P(\text{correct}|\text{neutral}) - \log P(\text{correct}|\text{suggested})$. Shared curvature cancels: $\nabla^2 \Delta \log P \approx \nabla^2 \log P_{\text{neutral}} - \nabla^2 \log P_{\text{suggested}} \approx 0$. Because true curvature is zero, evaluating end-state gradient $\nabla f(\theta_t)$ merely injects sample gradient noise. The trapezoid oracle **worsened error by 25.6%** (cut: **−0.256**, `docs/reviews/wme/wme-pilot2-2026-11-16.md` §4).
2. **Pooling Distortion Driven by a Single Non-Linearity:**  
   The 34.6% pooled cut is driven almost entirely by `refusal_margin_harmful` (cut = **0.790**). The other five objectives show negligible or negative transport:
   `gsm8k_gold_lp`: **0.041**, `mmlupro_gold_lp`: **0.023**, `wikitext_nll`: **0.144**, `refusal_margin_harmless`: **0.128**, `sycophancy_real`: **−0.256**. Harmful refusal moves massively (static error 9.77 SE at $h \ge 3$, §3) due to steep logsumexp saturation. Claiming slope transport succeeds when 5 of 6 fail is a pooling illusion.
3. **Truncated Trust Region:**  
   `docs/reviews/wme/wme-pilot2-2026-11-16.md` §6 notes that no ladder reached the $\Delta\text{NLL} > 1.0$ nat stop rule; maximum observed $\Delta\text{NLL}$ was **0.127 nats** (`scale_down15` at dose −4). Probing $\le 0.127$ nats traps the model in the quasi-linear bowl where $A_0$ dominates.
4. **Access Tier Violation:**  
   The trapezoid requires $\nabla f(\theta_t)$, an **R2 oracle**. If the R2 oracle fails on 5 of 6 objectives, R0/R1 approximations (C4 HVP $H_0 D$ or learned hypersolvers) cannot succeed.

*Decisive Fresh-Data Experiment:* Test R1 exact HVP ($H_0 D$) against static $A_0$ on 50 fresh p3 composition steps ($h \ge 2$) pushed to $\Delta\text{NLL} \in [0.5, 1.0]$ nats.  
*Kill Rule:* Kill slope transport if fewer than 3 of 5 non-refusal objectives achieve an error cut $\ge 15\%$, or if sycophancy cut remains $\le 0$. (Cost: 0.5 GPU-h, ~$2.50).

---

#### 1.2 GSM8K / MATH Headroom (Go-E2): CoT Deprivation and the Guardrail Trap
In pilot 1 (`docs/reviews/wme/wme-v0-pilot-2026-11-16.md`), Go-E failed (0 of 24 candidates for GSM8K/MMLU-Pro). In pilot 2 (`docs/reviews/wme/wme-pilot2-2026-11-16.md` §5), GSM8K qualified 3 of 32 candidates (LoRAs) and MATH 4 of 32. Under value-scale margins, GSM8K drops to 1 of 32. This failure is an evaluation artefact:

1. **Chain-of-Thought Deprivation (`enable_thinking=False`):**  
   In `src/wsp/wm/panel.py` (lines 308–310), GSM8K and MATH are scored by appending `"The answer is "` and measuring teacher-forced log-probability of final numerical tokens with `enable_thinking=False`. A 9B instruct model cannot solve multi-step math in a single forward pass without reasoning tokens. Base dev gold log-probabilities are catastrophically low: **−2.521 nats** on GSM8K and **−10.975 nats** on MATH ($\approx 1.7 \times 10^{-5}$, §5). This measures raw token memorization, not math reasoning.
2. **LoRA SFT Exposure Bias:**  
   LoRAs trained in `src/wsp/wm/lora.py` (rank 8, 256 steps) on teacher-forced answers nudged GSM8K gold log-prob by only 0.14–0.26 nats. SFT on final answers shifts formatting without building reasoning chains.
3. **The Sigmoid Guardrail Trap on Harmless Refusal:**  
   In `src/wsp/wm/panel2.py` (line 269), `pp_value` maps refusal margins via $\sigma(\text{margin}) = 1 / (1 + e^{-\text{margin}})$. Harmless prompts have baseline margins around [−4, −3], where sigmoid slope is steepest ($d\sigma/dz \approx 0.04$). A benign 0.5-nat logit fluctuation shifts probability by $>1.5$ pp. In §5, **all strong sycophancy LoRAs (LCB 2.09–2.42) and math LoRAs at $\lambda \in \{0.5, 1.0\}$ were disqualified** solely by tripping this 1 pp harmless-refusal threshold, despite zero real over-refusal.
4. **Action Space Mismatch:**  
   Single-layer steering shifts (`down12-15`) cannot inject multi-step logic. Expecting rank-one activation shifts to solve competition math without reasoning tokens is a category error.

*Decisive Fresh-Data Experiment:* Evaluate the 3 GSM8K LoRAs and 4 MATH candidates on a fresh dev panel ($N=128$) using **autoregressive CoT generation** (`enable_thinking=True`, greedy, 256 tokens) alongside log-prob, with empirical bootstrap-t null bounds.  
*Kill Rule:* If held-out CoT generation accuracy gain is $\le 0\%$ across all candidates, declare zero genuine headroom. (Cost: 0.6 GPU-h, ~$3.00).

---

### 2. External Research: Weight Edits and Frontier Competitors

#### 2.1 What Actually Raises GSM8K / MATH Benchmark Accuracy on ~9B Models
Literature establishes clear mechanics for raising held-out reasoning accuracy via weight modification:

1. **Task Arithmetic and Pruning (DARE / TIES):**  
   DARE ([arXiv:2311.03099](https://arxiv.org/abs/2311.03099)) and TIES-Merging ([arXiv:2306.01708](https://arxiv.org/abs/2306.01708)) demonstrate that 90–99% of fine-tuned delta parameters are redundant or destructive. Merging math task vectors ([arXiv:2212.04089](https://arxiv.org/abs/2212.04089)) into 7B/8B models yields gains only when sign interference is trimmed.
2. **Evolutionary Cross-Domain Merging:**  
   Evolutionary Model Merging (Sakana AI, [arXiv:2403.13187](https://arxiv.org/abs/2403.13187)) proved evolving layer paths and weights across Japanese LLMs and WizardMath-7B produced massive gains on MGSM-JA (16.4% → 55.2%), preserving general capability without training.
3. **The SFT vs. RL Dichotomy in Singular Value Fine-Tuning (Transformer²):**  
   In Transformer² (Sakana AI, [arXiv:2501.06252](https://arxiv.org/abs/2501.06252), ICLR 2025), Sun et al. adapted LLaMA-3-8B-Instruct (base GSM8K: 75.89%):
   - **Next-Token SFT crashed GSM8K accuracy to 60.50%** and MATH to 18.52% (Table 4).
   - **Policy Gradient / RL on Singular Values (SVF) lifted GSM8K to 79.23%** and MATH to 25.04% with only 0.58M parameters.  
   *Takeaway:* Teacher-forced SFT (`src/wsp/wm/lora.py`) inherently degrades reasoning; verifiable reward RL is required.

#### 2.2 Can a World Model Predict/Plan Reasoning Edits at R0/R1?
**No.** At R0/R1, a world model operates solely on base-model tangent contractions ($A_0, H_0 D$). It cannot simulate multi-token autoregressive branching, exposure bias, or dynamic execution feedback across 256+ tokens. Predicting single-token teacher-forced log-prob at R0 has near-zero rank correlation with multi-token problem-solving accuracy. Planning reasoning edits at R0 is an ill-posed surrogate search.

#### 2.3 Frontier Competitors That Beat a World Model at Equal Budget
Confirmatory bank construction is priced at **28.4 to 35 GPU-h** (`RULING-wme-resources.md` §1). At equal budget, a skeptic expects the following competitors to decisively beat WM-E:

1. **Sakana Fugu / TRINITY / Conductor (Learned Orchestrators):**  
   Fugu ([arXiv:2606.21228](https://arxiv.org/abs/2606.21228)), Conductor ([arXiv:2512.04388](https://arxiv.org/abs/2512.04388)), and TRINITY ([arXiv:2512.04695](https://arxiv.org/abs/2512.04695)) deploy Thinker, Worker, and Verifier roles with access lists. TRINITY proves sep-CMA-ES over candidate edits outperforms surrogate modeling under tight evaluation budgets.
2. **Transformer² / Dynamic SVF:**  
   Transformer² ([arXiv:2501.06252](https://arxiv.org/abs/2501.06252)) precomputes modular SVF expert vectors and dynamically mixes them via two-pass prompt dispatch at test time, adding only 3–7% latency while beating static LoRAs.
3. **Text-to-LoRA & Drag-and-Drop Hypernetworks:**  
   Text-to-LoRA ([arXiv:2506.06105](https://arxiv.org/abs/2506.06105), ICML 2025) and Drag-and-Drop LLMs ([arXiv:2503.18123](https://arxiv.org/abs/2503.18123)) train hypernetworks generating task-specific LoRA weights ($\Delta W = BA$) in a single forward pass directly from task instructions, eliminating iterative search.
4. **AB-MCTS (TreeQuest):**  
   Sakana AI’s Adaptive Branching MCTS ([arXiv:2503.04412](https://arxiv.org/abs/2503.04412)) dynamically balances "going wider" (sampling edits) and "going deeper" (refinement) using Thompson sampling on external verifier feedback.
5. **ShinkaEvolve / AlphaEvolve:**  
   ShinkaEvolve ([arXiv:2509.19349](https://arxiv.org/abs/2509.19349), ICLR 2026) uses bandit-based LLM ensembles and novelty rejection sampling for sample-efficient program and prompt search in <150 evaluations.
6. **Direct Multi-Fidelity Best-of-N Search:**  
   The 35 GPU-h bank buys **>4,500 real state evaluations** (at 24.8 s/state, `docs/reviews/wme/wme-v0-pilot-2026-11-16.md` BLUF). A two-stage filter (4-item screen → 32 validation items) searching over small LoRA scales and merged task vectors directly explores true capability space with zero surrogate bias.

---

### 3. Post-Hoc Rescues vs. Pre-Declared Rigor

Every proposal after pilot 2 is post hoc (`docs/council/2026-11-17/BRIEF-r10.md`). We must distinguish illegitimate rescues from valid engineering:

#### Illegitimate Post-Hoc Rescues (Flagged & Prohibited):
1. **Loosening Guardrails:** Expanding harmless-refusal threshold from 1 pp to 2 pp or switching scales after observing LoRA rejections.
2. **Refusal-Dominated Pooling:** Claiming slope transport "works" via pooled metrics when 5 of 6 cognitive objectives fail.
3. **Proxy Substitution:** Retaining gold log-prob proxies when autoregressive generation accuracy shows zero gain.
4. **Simulator Alternative Tuning:** Modifying synthetic simulation scenarios to match pilot-2 residuals (strictly barred by `docs/council/2026-11-17/RULING-wme-v2b.md` A7).

#### Mandatory Prevention Protocols:
- **Strict Pre-Declaration:** Freeze candidate spaces, seeds, and statistical thresholds in `docs/PREREG_WME.md` before generating confirmatory data.
- **Fresh Data Exclusivity:** All tests must use fresh `wme-p3` splits; pilot-1 and pilot-2 items are quarantined.
- **Automated Kill Rules:** Enforce hard programmatic termination upon control failure.

---

### 4. Conviction Threshold: What Convinces a Skeptic?

A skeptic will concede WM-E is scientifically valid if and only if all four criteria hold on sealed confirmatory data:

1. **Disaggregated Slope Transport:** R1 HVP ($H_0 D$) cuts error by $\ge 20\%$ relative to $A_0$ on $\ge 3$ of 5 non-refusal objectives at $h \ge 2$ on fresh p3 steps with $\Delta\text{NLL} \in [0.5, 1.0]$.
2. **Autoregressive Generation Transfer:** Winning edits selected by the world model demonstrate statistically significant ($p < 0.01$) improvements in end-to-end CoT generation accuracy on held-out test splits.
3. **Impeccable Safety Preservation:** Winning edits produce zero guardrail breaches under empirical bootstrap-t bounds on refusal margins and WikiText perplexity.
4. **Economic Parity:** World-model planning matches or outperforms Direct Multi-Fidelity Best-of-N search under an unamortized budget ledger charged with full bank construction compute.

---

### 5. Proposals and Kill Criteria Table

| proposal | mechanism | access tier | test on fresh data | cost (GPU-h, $) | kill criterion |
|---|---|---|---|---|---|
| **Disaggregated Slope Test** | Replace pooled Go-R2 with per-objective R1 exact HVP ($H_0 D$) | R1 | 50 fresh p3 steps ($h \ge 2$) with $\Delta\text{NLL} \in [0.5, 1.0]$ | 0.5 GPU-h (~$2.50) | Fewer than 3 of 5 non-refusal objectives achieve $\ge 15\%$ error cut, or sycophancy cut $\le 0$ |
| **CoT Generation Headroom Audit** | Autoregressive decoding (greedy, 256 tokens, `enable_thinking=True`) on GSM8K/MATH | R2 (eval) | Fresh dev panel ($N=128$) across 32 candidate edits | 0.6 GPU-h (~$3.00) | Zero candidate edits achieve held-out generation accuracy gain $\Delta\text{acc} > 0$ over base |
| **Empirical Bootstrap-t Guardrails** | Replace 1 pp sigmoid scale with empirical bootstrap-t bounds on unedited model | R0 (calc) | Null bootstrap distribution on 128 harmless/harmful prompt splits | <0.1 CPU-h (~$0.00) | False-positive rate on null distribution $> 5\%$ at $\alpha = 0.05$ |
| **Dynamic SVF Action Family** | Adapt singular values ($\Delta W = U \text{diag}(\sigma \odot z) V^T$) via RL reward | R0 / R1 | Compare SVF against rank-8 LoRA SFT on fresh dev panel | 0.8 GPU-h (~$4.00) | SVF candidate resolvability $< 0.40$ or gain $\le$ LoRA SFT |
| **Direct Best-of-N Competitor Arm** | 2-stage multi-fidelity filter (4-item screen → 32-item panel) without world model | R2 (eval) | Head-to-head capability gain on held-out test split | Matched to bank (35 GPU-h, ~$175) | Direct search matches or exceeds WM-E test gain at equal total compute |
| **Fugu-Style Verifier Orchestrator** | Conductor/TRINITY access-list scaffold with sep-CMA-ES candidate generation | R0 / API | E2 contrast under identical BudgetLedger (32 evaluation bundles) | API tokens (~$10) | Fugu multi-agent selection matches or beats WM-E test score |
| **Curvature Cancellation Audit** | Evaluate paired difference vs single-ended log-prob curvature | R1 | Test $\nabla^2$ magnitude on sycophancy prompt pairs vs individual prompts | <0.2 CPU-h (~$0.00) | Sycophancy Hessian norm exceeds individual prompt Hessian norm |
