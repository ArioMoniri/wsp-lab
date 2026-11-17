# Round 11 synthesis and chair ruling, 17 Nov 2026: the plan after v2b and pilot 3

**Seats and reports.**

| Seat | Model | Report |
|---|---|---|
| methods | Claude | `round11-methods.md` |
| world-model | Claude | `round11-world-model.md` |
| pro-engineer | Claude | `round11-pro-engineer.md` |
| challenger | Gemini 3.8 Flash | `round11-challenger.md` |

**Inputs.** Brief: `BRIEF-r11.md`. Interim data: `r11-interim-pilot3-dev.md`.

**Status.** Pilot 3 is still running (H3, H4 and the S rerun). Everything here is design, made before any
confirmatory data exists. All pilot-3 numbers are exploratory dev numbers, exposed to selection. Part A stays held.

## BLUF

1. **H1's dev gain is budget relief, not better reasoning.** The world-model seat found this, and the chair
   re-computed it from `data/wme/pilot3/run-20261117/h/evals.jsonl`.

   | H1 at step 208, λ 1 | MMLU-Pro | SuperGPQA |
   |---|---|---|
   | All items | 0.682 → 0.726 | 0.354 → 0.447 |
   | Items untruncated in both models | 0.853 → 0.829 (−0.024, SE 0.009; n = 783) | 0.566 → 0.545 (−0.021, SE 0.014; n = 613) |
   | Items the base truncated: H1 accuracy | 0.38 | 0.30 |

   On the items the base truncated, the length-matched control and H2 still truncate 74–81 % and score
   0.11–0.17. So **the length-matched control did not control the mechanism.**
2. **Consequence.** Any accuracy-enhancement claim is worded as **accuracy at a frozen non-thinking token budget**.
   A registered 2× budget arm decides whether the paper may say more (R2).
3. **New claim set** (R1), with sycophancy E1 as a secondary:
   - H-E1g, generated accuracy at the frozen budget;
   - H-E2g, world-model-guided search over an H1-type edit library;
   - H-T2, trajectory prediction along held-out edit paths.

   "(a) + (b) shown" may be claimed only if both H-T2 and an E-claim pass.
4. **Stage W pilot comes before PREREG v2.** It uses fresh paths and fresh seeds, at ≤ 4 GPU-h plus timing pilots
   of ≤ $11. It tests the budget-aware world-model state and simulates power for H-T2 and H-E2g.
5. **Immediate checks inside the pilot-3 cap** (R3):
   - base and H1 at a 4,096-token budget;
   - raw generations kept for a parser audit;
   - a guard-scorer audit. The base is flagged as refusing about 0.9 of harmless prompts, so the guard is
     descriptive until it is audited.
6. **9B plan.** About 81 of 90 GPU-h and about $700 of $1,000, on at most 4 concurrent 1× H100 SXM5. Sycophancy
   E2 is cut first if a cap binds. The 27B run starts only after the 9B result is sealed, on a 2× H100 worker.

## Rulings

### R1. Claim set and priority (replaces the RULING-wme-v2b A8 ranking)

**Priority:** {H-T2, H-E1g, H-E2g} > {H-T2, H-E1g} > {H-E1g, H-E2g} > {H-E1g} > {sycophancy H-E1}.

- **H-E1g.** One frozen H1-type recipe, retrained with fresh seeds, against the base on fresh sealed MMLU-Pro and
  SuperGPQA generation panels (`wme-c2-gen`). The test is paired and one-sided. The two benchmarks are Holm-adjusted
  at a family α of 0.025.
  - Sizing follows the methods seat: half the de-biased mean dev gain, Wilson-upper discordance from `sizing.json`,
    α = 0.0125 per benchmark, power 0.8. That gives N = 3,463 MMLU-Pro and 1,784 SuperGPQA items, a read cost of
    about 0.45 GPU-h.
  - If the source pools cannot supply fresh items at those sizes, N drops to the largest fresh pool. Power is then
    reported as is, and that is declared before the freeze.
- **H-E2g.** Shared multi-fidelity search over the edit library (R5), with 8 full real evaluations per arm.
  - **Primary:** the world model vs StatNN, on the same inputs and planner, so that the contrast isolates the
    world-model structure.
  - **Co-primary (Holm within the E2 family):** the world model vs direct multi-fidelity best-of-N at **total
    compute parity**. The world model is charged its R1 derivative compute and its bank. This answers the
    challenger: a win over StatNN alone does not show the world model is worth its cost.
  - **Descriptive:** the "last checkpoint at λ 1" default, and the reader competitors (SYNTHESIS-r10 D7).
  - The co-primary is kept only if stage W's simulation powers it (≥ 0.8). Otherwise it is a registered secondary
    and is reported whatever the result.
- **H-T2.** See R4.
- **Sycophancy.** H-E1 (TriviaQA generation, N = 1,000, already powered in v2b) stays as a registered secondary.
  Sycophancy E2 is dropped if any cap binds.
- **Old H-T.** It becomes descriptive on the bank (v2b).

### R2. Budget wording rule (pre-declared)

Every accuracy endpoint is registered at the frozen budget, 2,048 new tokens with thinking off. Alongside it, the
paper registers a **secondary at 4,096 tokens** and the **untruncated-in-both paired difference**.

- If H1's gain at 4,096 has a lower bound > 0, and its untruncated-in-both difference has a lower bound > −1 pp, the
  paper may say **"improves accuracy"**.
- Otherwise it must say **"improves accuracy at a fixed inference budget (token efficiency)"** and report the
  mechanism.

This rule also binds the 27B run.

### R3. Pilot-3 amendment 2 (outcome-blind for the new checks; inside the $150 / 25 GPU-h caps)

Track C appends this to `RULING-wme-pilot3.md` and commits it before the new checks run.

- **(a) Budget check.** Base and H1 (step 208 and step 104, λ 1), each at 4,096 tokens, on the same dev panels.
  This is descriptive and costs about 0.5 GPU-h.
- **(b) Raw generations.** Keep raw generations, or at least the final 400 characters plus the extracted answer,
  for every remaining and re-run evaluation. Re-parse the base and H1 with a flexible extractor (challenger proposal)
  and report the residual gap. The challenger's "parser mimicry" claim is not supported by the untruncated-in-both
  result, but it is cheap to test.
- **(c) Guard audit.** On 100 base harmless and 50 harmful guard generations, a blind label by two Claude judges
  plus a rule-based check. The guard gates nothing until its agreement with the audit is ≥ 0.9.
- **(d) Status files.** Fix the stale local `h/STATUS` handling, and make heartbeat liveness depend on progress
  (pro-engineer).

### R4. H-T2 trajectory claim (world-model seat; challenger baseline added)

**Task.** Predict held-out accuracy *and* truncation rate along fresh edit paths: training checkpoints × λ ×
layer-group masks. Two settings:

- **few-shot:** the first k real points of a path are given;
- **zero-shot:** only R0/R1 features of the edit are given.

**Matched baselines** (all use the same real points):
- persistence;
- per-path parametric learning curves, including a 2-parameter power law (challenger);
- LC-PFN;
- StatNN;
- the static map, for the zero-shot setting.

**Nulls the world model must not beat:**
- shuffled edit features;
- a path-label permutation;
- a synthetic world where curves carry no edit information.

**What makes it a world model:** it reads the edit and predicts library members that no per-path fit can reach,
such as unseen masks and merges. Registration requires stage-W-simulated power ≥ 0.8 with ≤ 40 paths. Otherwise
it is registered as **not powered**, and the paper does not claim (a).

### R5. World-model state for accuracy (world-model seat)

The registered D3 proxy (log-prob of the base's own verified answers) runs against the dev accuracy change
(ρ −0.81 and −0.88, world-model seat). It is retired as a primary feature. That is consistent with the D3 kill rule;
the formal kill is read on the final report.

The new per-item state is **P(finish within budget) × P(correct | finish)**:

| Input | Tier |
|---|---|
| Cheap base signals (lengths, partial-generation statistics) | R0 |
| Derivatives along the edit direction at θ₀ | R1 |
| A teacher-forced pass of an edited model | R2, charged as a step |

**The library:** checkpoint × λ × layer-group mask × merges × H4 projection × SVF scale. The SVF scale is included
only if H3 passes.

### R6. Flag 1

Every comparator is shrunk toward the raw field on the same grid, folds and loss as WM-CP, with ties going to the
exact field. Gate: quadratic-null size ≤ 0.035. This is the methods and world-model seats' reading of PREREG
§6.1.2.

### R7. Fresh data and exposure

- Build new sealed panels `wme-c2-gen`.
- Add `wme-p3h` and `wme-p3s` (all splits) to `wm.exposure.EXPOSURES`.
- Retrain every pilot-3 adapter that is used with fresh seeds.
- PREREG v2 lists every pilot-informed choice.
- The guard pools are exhausted (pro-engineer). New guard sources are pinned before the freeze; the PREREG author
  fetches and pins them.

### R8. Pre-declared consequences of the pending pilot-3 results

| Pending item | Outcome | Consequence |
|---|---|---|
| H3 (SVF policy gradient) | Pass: beats both controls, ≥ 2/32 | SVF scales join the library (≤ +6 GPU-h) |
| | Fail or undecidable | Dropped |
| H4 (guard-orthogonal) | Pass | Projection becomes a free library operator |
| | Not run or partial | "Not measured", and not in the library |
| S (slope rerun) | Pass for an objective | Transport kept for that objective only, within the bank cap |
| | Partial or fail | Transport dropped permanently. No further S spending (challenger) |
| D3 proxy gate | Killed | Recorded; R5 state used |
| R3(a) budget check | Base at 4,096 ≥ H1 at 2,048 − 1 pp on both benchmarks | H1 is pre-classified as token efficiency (R2 wording expected) |
| R3(c) guard audit | Agreement < 0.9 | Guards descriptive, and a new scorer is built before the freeze |

### R9. Execution (pro-engineer)

1. **Pilot 3** finishes by its deadline, 18 Nov 05:25 UTC, ≤ $150.
2. **Stage W and two timing pilots:** ≤ 4 GPU-h and ≤ $11. Any measured rate more than 1.5× its estimate is
   re-priced by amendment before its data exist.
3. **PREREG v2:** the author fills in R1–R8 and amendments A1–A11.
4. **Board A** re-reviews, then the **freeze**.
5. **9B run:** the bank (≈ 21.9 of 35 GPU-h) and confirmatory runs (≈ 81.5 of 90 GPU-h-eq), on at most 4 concurrent
   1× H100 SXM5.
   - Each instance has its own dead-man state, a base canary, frozen batch and order, and a per-track venv.
   - There is no PCIe fallback for confirmatory runs, so the platform stays identical.
6. **27B** starts after the 9B result is sealed, on a 2× H100 worker.
   - E1g and H-T2 are in scope.
   - E2g runs only if the measured cost ratio is ≤ 2.6× the 9B cost; otherwise it is descriptive.

**Totals and kills.**
- 9B: ≤ 90 GPU-h-eq and ≤ $1,000; the estimate is about $700.
- Any gate overrun of more than 1.5× stops that stage for re-pricing.
- All costs here are estimates scaled from the measured pilot-3 unit of 226–312 s per candidate evaluation.

### R10. Challenger points rejected or modified

| Challenger point | Ruling |
|---|---|
| "Drop stage S" | Not now. The rerun is already paid for within the cap, and R8 applies. No further S work. |
| "Demote H-T to descriptive" | Accepted for the old H-T. H-T2 must earn its registration through stage W. |
| "StatNN is a strawman" | Partly accepted: best-of-N is now co-primary at compute parity. StatNN stays as the primary contrast because it isolates the structure. |
| "Regex parser mimicry explains H1" | Not supported: H1 is lower on untruncated-in-both items. Audited cheaply in R3(b). |

## Table

| Decision | Rationale | Test or gate | Cost | Kill or fallback |
|---|---|---|---|---|
| H-E1g at frozen budget, with R2 wording | H1 gain is budget relief | Paired Holm on sealed `wme-c2-gen` | ≈ 0.45 GPU-h read | Fail → sycophancy H-E1 remains |
| R3 checks (4,096 tokens, parser, guard audit) | Mechanism and scorer validity | Descriptive, pre-declared consequences | ≤ 0.7 GPU-h, inside pilot 3 | Guard descriptive if agreement < 0.9 |
| Stage W | New state, H-T2 and E2g power unknown | Fresh paths; simulated power | ≤ 4 GPU-h + ≤ $11 | H-T2 or E2g co-primary unregistered if power < 0.8 |
| H-E2g: world model vs StatNN, and vs best-of-N at parity | (b) needs the world model to beat direct search | Shared multi-fidelity planner, 8 real evaluations per arm | ≈ 22 GPU-h | Best-of-N becomes a secondary if unpowered |
| H-T2 | Physically valid trajectory claim | Held-out paths vs curve fits and nulls | ≈ 4.8 GPU-h | Not powered → (a) not claimed |
| Flag 1: symmetric shrinkage | Fair static comparator | Quadratic null ≤ 0.035 | CPU | Redesign before the freeze |
| 9B confirmatory | Within user authorisation | Board A, then freeze | ≈ 81.5 GPU-h, ≈ $700 | Cut sycophancy E2 first |
| 27B replication | User goal | Launch gate after 9B sealed | ≈ $155–300 for E1g + H-T2 | E2g descriptive if cost ratio > 2.6 |
