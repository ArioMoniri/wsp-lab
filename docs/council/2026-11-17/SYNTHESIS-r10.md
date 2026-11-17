# Round 10 synthesis and chair ruling, 17 Nov 2026: slope transport, benchmark headroom, Fugu-like systems

**Seats and reports.**

| Seat | Vendor | Report |
|---|---|---|
| slope | Claude | `round10-slope.md` |
| headroom | Claude | `round10-headroom.md` |
| systems (Fugu-like) | Claude | `round10-systems.md` |
| challenger-engineer | Gemini 3.8 Flash | `round10-challenger.md` |

The brief is `BRIEF-r10.md`.

**Status.** This round was held *after* the pilot-2 results, so every decision below is post hoc. Each stage is
declared here, before any data that tests it, and runs only on **fresh** p3 steps or fresh dev panels. Pilot-2
data may be used for diagnosis and never for testing a fix. Part A stays held.

## BLUF

1. **Slope transport.**
   - Pilot 2's 34.6 % oracle cut is mostly harmful refusal. Without it the oracle cuts 4.7 %, by the slope seat's
     arithmetic on `docs/reviews/wme/wme-pilot2-2026-11-16.json`.
   - The cause: K-FAC/Fisher curvature is positive semidefinite, and the static step carries no curvature for
     margins and sycophancy (`SIGN_Q` = 0 in `scripts/wme_pilot2.py`).
   - Fix: an exact R1 second directional derivative and a cross-derivative at θ₀ (S2), plus a per-objective
     step-rule router (P1). Kill rule: if both fail, slope transport is dropped for good.
2. **Headroom on GSM8K and MATH was mostly mis-measured.**
   - The objective was the log-probability of the gold answer.
   - The prompt said "Give only the final numeric answer".
   - Thinking mode was off.
   - The harmless-refusal guard was underpowered: a breach meant a failure to *prove* there was no harm.

   The next headroom stage scores **generated accuracy** on benchmarks from the model card.
3. **New edit families, in order:**
   1. distil the model's own thinking mode into non-thinking mode;
   2. verified self-distillation (STaR);
   3. SVF with policy-gradient RL anchored by KL;
   4. guard-orthogonal LoRA projection.
4. **Fugu-like systems.** Planner upgrades (multi-fidelity BO, AB-MCTS, a COMs-style conservative surrogate) must
   first pass synthetic regret tests. Two new descriptive competitors enter E3: direct multi-fidelity best-of-N,
   and LLM-evolved edit recipes.
5. **Several challenger claims are rejected as factually wrong** (§D9). Its valid points are adopted.

## Decisions

### D1. Go-R is reported per objective

Go-R is reported **per objective and on resolvable pairs**. A pooled cut is never the only number reported. This
applies retroactively to the pilot-2 description and to every future gate.

### D2. Slope stage S (fresh `wme-p3` steps, H100)

| Arm | What | Tier |
|---|---|---|
| S2 (primary) | Per-item second directional derivative along D_t at θ₀ (nested forward-mode JVP) plus a cross-derivative with earlier edits, on 96 field items | R1 |
| S1 | Exact HVP at θ₀ on the support layers, replacing K-FAC ½eᵀH₀e | R1 |
| S3 | Per-objective shrinkage and an on/off switch, fixed on construction steps before test steps are read | R0 |
| P1 | Router over {static, S1, S2, static + learned residual}, per objective × edit family, fitted on construction steps | R0 |
| S5 | Learned end-slope corrector (hypersolver style, 1/SE² soft targets) | R0 |
| Oracle reference | Trapezoid oracle, used only to compute the fresh cut | R2 (reference only) |

**Tier ruling.** Only derivatives taken at θ₀ by automatic differentiation count as R1. Finite differences at
θ₀ ± εD are R2 and are not allowed.

**Gate.** The test is per objective, restricted to objectives whose fresh oracle cut is ≥ 0.20.
- **Pass:** S2 or S1 recovers ≥ 50 % of the fresh oracle cut.
- **Kill:** it recovers < 25 %.

P1 must beat the best single global rule (lower bound > 0). S5 must beat both S2 and 25 % of the oracle cut.

**Consequence.**
- If S1 and S2 both fail, slope transport is dropped (as SYNTHESIS-r9 C4 already says). The per-state oracle
  gradients then leave the bank.
- If they pass, transport is kept only for the objectives that passed. Every other objective uses static plus a
  learned residual on WM-CP.

**Cap.** ≤ 4.5 H100 GPU-h (≈ $22). The JVP cost factors are assumptions and must be measured in the first
0.25 GPU-h; stop if the measured cost is more than 2× the estimate.

### D3. Headroom stage H: measurement first

- **Objectives.** Generated accuracy under a step-by-step prompt with thinking off, a frozen token budget, and
  answer-verified scoring.
  - **Primary candidates:** MMLU-Pro and SuperGPQA in multiple-choice form (on the fetched model card).
  - **Secondary:** GSM8K and MATH L4–5 (not on the model card, but standard).
  - **Secondary:** IFEval, as fresh panels built from its instruction types.
  - GPQA-D has only 198 items, so it is descriptive only.
- **Proxy for the world model.** At R0 it predicts the log-probability of the *base model's own verified-correct
  generations*. Kill this proxy if, over ≥ 32 candidates, its correlation with the measured accuracy change has a
  lower bound ≤ 0.
- **Ceiling.** The base model's thinking-mode accuracy on the same fresh dev panels is measured first. It is the
  ceiling, and it is never used to select candidates.
- **Generation pricing.** P-H100b batched-generation pricing (track A, RULING-wme-v2b A10) must be in hand before
  item counts are fixed. N is sized by measured variance.

### D4. Edit families, in this order

Each stage passes with ≥ 2 of 32 guarded dev candidates whose accuracy-gain LCB is > 0.

| # | Family | Data | Controls | Cap |
|---|---|---|---|---|
| H1 | Thinking→non-thinking distillation LoRA | gold-verified thinking-mode answers to construction items, under the frozen budget | the same LoRA on *non-thinking* verified answers (= H2); a length-matched control | ≤ 6 H100-h |
| H2 | Verified self-distillation (STaR), checkpointed every step | the model's own verified non-thinking answers | the base model | ≤ 3 H100-h |
| H3 | SVF singular-value scales trained by policy gradient, KL-anchored to θ₀ | construction items with verifiable rewards | a random-reward arm and a format-only arm, which must both lose | ≤ 8 H100-h |
| H4 | Guard-orthogonal projection of H1–H3 and pilot-type LoRAs (refusal-direction orthogonal or harmless-activation null space; Safe-LoRA against Qwen3.5-9B-Base) | — | the unprojected edit | ≤ 1 GPU-h |

Fail rules:
- **H1 fails** if its gain comes only from responses that exceed the token budget.
- **H3 fails** if it does not beat both controls.

**SVF must use policy gradient, not next-token loss.** In Transformer² Table 4 (https://arxiv.org/html/2501.06252,
verified), on Llama-3-8B-Instruct:

| Setting | GSM8K | MATH |
|---|---|---|
| Base | 75.89 | — |
| SVF, next-token loss | 60.50 | 18.52 |
| SVF, policy gradient (MLP + attention) | 79.23 | 25.04 |
| LoRA, next-token loss | 77.18 | — |

LoRA with next-token loss does **not** hurt there (77.18 vs base 75.89).

If H1–H4 all fail: there is no accuracy-enhancement claim. Headroom is reported as the limiting finding, and the
sycophancy H-E1 stands as registered.

### D5. Guard

The pilot-2 guard (first-token harmless refusal margin, 1 pp) is kept descriptively but no longer gates.

**Replacement.**
- A **generated over-refusal rate** on fresh XSTest-style harmless prompts, plus harmful-compliance on fresh
  harmful prompts.
- Equivalence margins are calibrated by a base-model bootstrap (the challenger's bootstrap-t proposal).
- Item counts are sized so that a null edit passes with probability ≥ 0.8.

**Why.** The sycophancy LoRA breaches were failures to prove harmlessness on 64 items (λ 0.25: +0.64 pp,
SE 0.46; headroom seat). They were not shown over-refusal. The math LoRAs at λ 0.5 and 1 did shift the guard
mean, by 1.2 and 3.2 pp. That is a real effect, which H4 targets.

### D6. Planner upgrades (systems seat)

Each upgrade is used only after it passes a synthetic regret test at B = 32 against the frozen CEM, and it is
**shared identically by the WM and StatNN arms**:

- P3: multi-fidelity BO, with the world model or StatNN as the prior mean and screens at 4 → 32 → full dev;
- P6: AB-MCTS adaptive branching, replacing the fixed 0.5 mutation fraction;
- P8: a COMs-style conservative surrogate.

P3 needs a pre-freeze rule: partial-fidelity real evaluations are charged pro rata to the E2 ledger.

### D7. Competitors (descriptive E3; the E2 primary contrast stays WM vs StatNN)

- **Direct multi-fidelity best-of-N without a world model**, from the challenger and systems seats. It uses the
  same 4 → 32 → full ladder and the same pro-rata ledger, with unamortised cost reported.
- **LLM-evolved edit recipes** (P7, in the style of ShinkaEvolve and AlphaEvolve). It writes new candidates on the
  same 32-evaluation ledger, with tokens charged. It is Claude-only, like the other reader arms.

### D8. Rejected or limited

- **Text-to-LoRA and Drag-and-Drop as generators** are rejected. Text-to-LoRA's GSM8K result is 47.3 against 47.5
  for a multi-task LoRA (systems seat). Only Drag-and-Drop's per-step checkpoint lesson is kept.
- **VeLO, the Darwin Gödel Machine and The AI Scientist** are rejected.
- **Tangent-space fine-tuning** is rejected as an edit family. Its insight survives as S2.
- **Evolutionary merge and Model Swarms** may run only inside R0. They need ~1,000 real trials, against our 32.

### D9. Challenger claims rejected after verification

| Claim | Verdict |
|---|---|
| "Next-token SFT crashed GSM8K to 60.50" | Misattributed. That row is **SVF with next-token loss**. LoRA with next-token loss scores 77.18 > 75.89 (Transformer² Table 4, fetched). |
| The harmless guard sits "at the steepest point of the sigmoid", with margins of about −4 to −3 | **False.** The base dev mean margin is −7.84 nats (`wme-pilot2-2026-11-16.json` `E2.9b.base_dev_mean`), i.e. ≈ 0.04 %, in the flat tail. The real problem is power (D5). |
| The bank costs 35 GPU-h | Stale. RULING-wme-v2b prices the 9B bank at about 18 GPU-h. |
| Frontier systems "crush" R0 world models at equal budget, and TRINITY "proves" sep-CMA-ES beats surrogates | Assertions without an equal-budget result for edit prediction. They enter as D7 competitors to be measured, not as findings. |
| "Fetch failures: none" | Not trusted. Claims used here were re-verified independently. |

**Adopted from the challenger:**
- per-objective reporting (D1);
- the measurement-artefact diagnosis (D3);
- bootstrap guard calibration (D5);
- a direct best-of-N competitor (D7);
- its conviction criterion: a real generation accuracy gain, and a WM win over unamortised direct search.

### D10. Ordering and budget

- **Track C (new)** runs stages S and H as "pilot 3", in parallel with track A's v2b work.
- **PREREG v2 is not frozen until pilot 3 reports.** Its outcome decides, by the RULING-wme-v2b A8 priority order,
  whether accuracy enhancement enters as a registered claim.
- **Caps for pilot 3:** ≤ 25 GPU-h-eq and ≤ $150 on Lambda. Both fall inside the ≤ $1,000 9B authorisation.
- **Lambda rules:** only `wsp-wme-*` instances, with the dead-man terminator.

## Table

| Proposal | Mechanism | Access tier | Test on fresh data | Cost | Kill criterion |
|---|---|---|---|---|---|
| S2 directional curvature | JVP of the 2nd and cross derivatives at θ₀ | R1 | wme-p3 steps, per objective | ≤ 4.5 H100-h total for S | < 25 % of the oracle cut |
| S1 exact HVP | replaces PSD K-FAC | R1 | same | in S | < 25 % |
| P1 step-rule router | per objective × family choice of rule | R0 | same | CPU | does not beat the best global rule |
| S5 learned end slope | hypersolver, soft targets | R0 | same | CPU | does not beat S2 and 25 % of the oracle |
| H-measurement | generated accuracy, verified-own-answer proxy | R0 proxy | fresh dev panels | in H | proxy–Δacc correlation LB ≤ 0 |
| H1 thinking→non-thinking LoRA | self-distillation of the model's own thinking mode | edit at R2 | fresh dev | ≤ 6 H100-h | < 2/32, or gain only over budget |
| H2 STaR | verified self-distillation | R2 | fresh dev | ≤ 3 H100-h | < 2/32 |
| H3 SVF policy gradient | RL on singular values, KL anchor | R2 | fresh dev | ≤ 8 H100-h | does not beat the random-reward and format-only controls |
| H4 guard-orthogonal | refusal-orthogonal / null-space projection | R0 projection | fresh dev | ≤ 1 GPU-h | < 2/32 under the new guard |
| New guard | generated over-refusal, bootstrap margins | R2 eval | fresh harmless/harmful | in H | null-edit pass rate < 0.8 |
| P3 / P6 / P8 planners | MF-BO, AB-MCTS, COMs | R0 | synthetic regret at B = 32 | CPU | ≤ CEM |
| Direct MF best-of-N; P7 evolved recipes | competitors | R2 / reader | same ledger | in E3 | descriptive |
