# PREREG_WME: WM-E, a single-LLM weight-edit world model for trajectory prediction and enhancement

**Status: DRAFT v0 (chair), 17 Nov 2026.**

- **Not frozen.** The draft becomes binding once three things have happened: (i) independent review (Board A role); (ii) the outcome-blind power simulation (§8) has set every N it leaves open; (iii) a dated freeze commit with a manifest hash is made.
- **Before the freeze.** No confirmatory outcome may be generated.
- **After the freeze.** The file is append-only. Changes are dated amendments, made before the data they govern.

WM-E is separate from the A5 programme (`docs/PREREG_H1_TF.md`). Part A remains held. Nothing here releases, informs or uses any A5 outcome.

## 0. Provenance (exploratory history that this registration builds on)

**Council rounds.**
- `docs/council/2026-11-16/SYNTHESIS-r8.md`: the design (D1–D9).
- `RULING-wme-pilot1.md`: the pilot-2 pre-declaration.
- `SYNTHESIS-r9.md`: contingencies, controls C1–C10. It was written before the pilot-2 result.

**Pilot 1** (`docs/reviews/wme/wme-v0-pilot-2026-11-16.md`):
- Go-M, Go-N and Go-C passed.
- Go-T, Go-R and Go-E failed.
- The bank was not built.

**Pilot 2** (`docs/reviews/wme/wme-pilot2-2026-11-16.md`, merged 8825b15; code committed 1b49d03 before data):

| Gate | Result |
|---|---|
| Go-T2 | Pass. 0.540 of steps resolvable; primary endpoint 2.17 SE |
| Go-R2 | Pass. Trapezoid oracle cut 0.346 |
| Go-E2 sycophancy_real | Pass, 10/32 |
| Go-E2 GSM8K | Pass, 3/32 |
| Go-E2 MATH L4–5 | Pass, 4/32 |

Under RULING §4, the primary enhancement objective is `sycophancy_real`. All pilot results are **exploratory and exposed**. None is confirmatory evidence.

**Known limits carried forward:**
- **(a) Pilot ladders never left the trust region.** Maximum WikiText ΔNLL was 0.127.
- **(b) The GSM8K pass depends on the pp-equivalent guardrail reading.** That reading was fixed in code before data; §5 ratifies it.
- **(c) Pilot-2 peak GPU memory came within 6 MB of the 28 GB cap.** §9 requires the preflight before the bank.

**27B engineering** (`docs/reviews/wme/wme-27b-calib-2026-11-16.md`): feasible on one 80 GB GPU, provided the field is collected in 2-layer groups with items of at most 768 tokens.

## 1. Target and access tiers

**Target:** Qwen/Qwen3.5-9B @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Text-only, bf16, `enable_thinking=False`.
- Vision and MTP tensors are frozen and never loaded.
- θ₀ identity is as in pilot 1.

**"Without running the edit" means tier R0.**
- R0: any artefact computed at θ₀ independently of the candidate, plus CPU arithmetic on D.
- R1: base passes with inputs that depend on D. These are a separately charged arm.
- R2: any pass at θ ≠ θ₀. Allowed only for (a) bank training labels, (b) charged oracle arms, and (c) tap observations of states already evaluated for real.
- Enforcement: `wm.access` raises on any non-θ₀ artefact used by `predict` or `plan`.

## 2. Hypotheses

**H-T (trajectory).** On held-out whole trajectories, WM-E predicts cumulative outcome change better than the strongest same-input static predictor of the cumulative edit.
- Training uses horizon ≤ 2; testing uses horizons 3–4.

**H-E1 (enhanced).** The WM-E-planned edit improves Qwen3.5-9B on sealed test items of the primary objective (`sycophancy_real`: lower sycophancy gap).
- Every guardrail must be non-inferior.
- The gain must be confirmed in generated answers.

**H-E2 (guidance helped).** At an equal real-evaluation budget, WM-E-guided search finds better test-scoring edits than StatNN-guided search.

**E3 (descriptive, always reported).**
- Competitiveness against best-of-N direct search, best-of-N steering, LoRA SFT, and a Fugu-style reader team.
- Total compute: amortised and unamortised.

## 3. Data, items and splits (fixed before any bank or confirmatory data)

**Confirmatory panel `wme-c1`.** It is built from the same pinned dataset revisions and hashes as pilot 2, with a new salt `wme-c1-panel-20261117`.
- Splits, disjoint by normalised-text hash: construction, predictor-validation, search-development and sealed test.
- Exclusions: every item that appears in any split of pilot 1 or pilot 2, including both pilots' sealed tests, by hash. The pilot-2 sealed test (`49813882…`) is held unread as a **reserve replication set**.
- Each split's content hash is committed at the freeze.
- Test items are unreadable by planner code, enforced by the realpath and `O_NOFOLLOW` guard. Only the sealed-test reader can open them, and only after winners are frozen.

**Objectives.**
- **Primary E1 objective:** `sycophancy_real`.
- **Secondary E1 objectives (Holm-adjusted):** GSM8K gold log-probability, and competition-math L4–5 gold log-probability.
- **Trajectory endpoints:** the pilot-2 six.
  - WikiText ΔNLL;
  - harmful refusal;
  - harmless refusal;
  - `sycophancy_real`;
  - GSM8K;
  - MMLU-Pro.

## 4. Trajectory bank (training data; not confirmatory outcomes)

**Families.** The six families of SYNTHESIS-r8 D4.
- Scale.
- Row scale (vocabulary).
- proj_remove (contrast plus matched random).
- Norm-matched noise.
- Steering compiled into writers.
- Rank-8 LoRA task vectors on construction items.

**Size.** 16 roots per family, 96 in total.
- Roots are drawn in seed namespace `wme-bank` and disjoint from every pilot root and direction.

**Ladders.**
- Signed doses ±{0.25, 0.5, 1, 2, 4, **8**} × the pilot-2 nominals.
- The ×8 step is added because limit (a) showed pilot ladders never left the trust region.
- Each side stops at WikiText ΔNLL > 1.0, and that state is kept as a labelled endpoint.

**Compositions.** 24 AB/BA bundles extended to 4 steps.

**Units.** Independence units are **source-closure components** (`src/wsp/msrm/closure.py`), never rows.

**Folds.**
- Five outer folds by component, with families stratified.
- The held-out trajectory test set is the outer fold.
- A leave-one-family-out analysis is secondary.

**Cost.** Priced at 7.8 GPU-h locally (pilot-1 formula). The bank is re-priced after the ×8 step and preflight, and stops if it exceeds 12 GPU-h.

## 5. Guardrails (ratified from pilot 2's code, `panel2.pp_value`)

| Objective | Rule |
|---|---|
| WikiText | ΔNLL UCB ≤ 0.01 nats/token |
| Higher-is-better objectives | pp-equivalent LCB ≥ −0.01 |
| Harmless refusal and sycophancy (lower-is-better) | pp-equivalent UCB ≤ +0.01 |

- Intervals are paired, item-cluster bootstrap, 95%.
- Guardrails are evaluated simultaneously, with Bonferroni across guardrails.

## 6. Primary analyses and decision rules

### H-T

**Contrast.** C_T = 1 − Σ|e_WM| / Σ|e_static| over held-out (state with h ∈ {3, 4}, objective) pairs.
- Errors are scaled by paired SE.
- "static" is the better, chosen on inner folds, of StatNN and refitted F3u, given the same cumulative-edit inputs.
- The lower bound is from a component-cluster bootstrap-t, 95% one-sided.

**Supported when all of the following hold:**
- the lower bound of C_T is > 0;
- the **C2 controls** pass:
  - **shuffled-bookkeeping null:** WM error degrades significantly when step-local inputs and order are permuted with E_T held fixed;
  - **inert-edit control:** |ŷ| ≤ 2σ on dead-row and post-norm-cancelled edits;
- results are also reported separately inside and outside the trust region (WikiText ΔNLL ≤ 0.1 vs > 0.1).

If any C2 control fails, H-T is void, whatever C_T is.

### H-E1

**Planner.**
- R0-scored beam search with sep-CMA-ES, LCB guardrails and a support penalty (SYNTHESIS-r9 C6).
- Search space: the bank families on development items.
- Budget: **B = 32 real-evaluation bundles per episode**. B is confirmed by the §8 simulation.

**Selection and test.**
- One winner is frozen per arm.
- It is scored once on sealed `wme-c1` test items.

**Supported when all of the following hold:**
- the paired 95% lower bound of the primary gain is > 0;
- all guardrails pass (§5);
- **generation audit:** greedy, non-thinking generation on fresh held-out prompts shows a positive gain in non-sycophantic answers, with loop and refusal anomalies ≤ 2%.

**Secondaries.** GSM8K and MATH L4–5 use the same rules, Holm-adjusted.

### H-E2

**Episodes.** Primary objective × 3 families × seeds. The number of episodes is set by the §8 simulation.

**Contrast.** The test-score difference between the WM-guided and StatNN-guided winners, paired across episodes.
- This is **the single primary contrast**.
- Episode-cluster bootstrap; supported when the lower bound is > 0.

**Secondary arms (Holm).**
- random;
- field-ranked;
- scripted reader;
- WSR-style LLM reader;
- Fugu-style reader team;
- direct best-of-N with a two-stage screen;
- best-of-N steering;
- LoRA SFT with ≤ 32 checkpoint evaluations;
- WM greedy one-step, which isolates planning.

### Uncertainty prerequisites (SYNTHESIS-r9 C1)

These must pass before any analysis:
- the σ floor;
- StatNN σ;
- closure bootstrap;
- conformal coverage at h = 3–4 within 10 pp.

If coverage fails, E2 uses ranks only.

## 7. Arms and fairness

- Every ranking arm uses the same candidate pool, B, development items, guardrails and planner code; only the scorer differs.
- LLM tokens, training compute and bank cost are charged and reported (E3).
- The R2 oracle and re-linearised arms are reported as oracles only.

## 8. Outcome-blind power simulation (sets the open Ns before the freeze)

**Inputs.** Simulate from:
- pilot-2 paired SE per objective;
- the component-size distribution of the bank design;
- two effect scenarios:
  - **null:** WM equals the static predictor;
  - **alternative:** C_T = 0.15, and an E2 gain of 0.5 × the pilot-2 between-candidate SD on development items.

**Requirements.**
- Size ≤ 0.07.
- Coverage error ≤ 0.07, both tails.
- Lower binomial bound on power ≥ 0.80.
- These apply to H-T, H-E1 and H-E2 separately, with ≥ 1,000 replicates per cell.

**Outputs.**
- The smallest bank component count and episode count that pass.
- If none fits within 12 GPU-h of bank plus 10 GPU-h of confirmatory evaluation, the claim is registered as **"unresolved — not powered"** and is not run.

## 9. Engineering prerequisites (before the bank)

- **Preflight (H4).** Run the longest items first, and cap at ≤ 27 GB GPU, giving ≥ 1 GB margin against the 28 GB limit.
- **Resumable units (H5).**
  - Content-addressed, with hashes checked on resume.
  - A resident apply/evaluate/restore server.
- **The C1 σ fixes, test-first.**

## 10. 27B replication (registered now, run after the 9B confirmatory analysis is sealed)

**Target and hardware.** Qwen/Qwen3.8-27B @ `37561816e66b5b9153b1b64c10968c18c1e6f4a0`, on Lambda 80 GB GPUs.
- Field collected in 2-layer groups, items ≤ 768 tokens.
- Dead-man terminator in place.

**Design.** The same hypotheses, rules and controls as 9B, with:
- a fresh panel salt `wme-c27`;
- a fresh bank in namespace `wme-bank27`;
- 27B's own pilot gates (T2, R2, E2) on fresh roots.

**Separation.**
- 27B is never pooled with 9B.
- 27B's own primary objective is chosen by the same RULING §3 rule on its development items.

## 11. What would count against the study (stated in advance)

| Outcome | How it is reported |
|---|---|
| C_T lower bound ≤ 0, or a C2 control fails | "Static cumulative prediction suffices on Qwen3.5-9B for these families." |
| E1 fails | "No verified enhancement." |
| E2 fails while E1 passes | "Enhanced, but world-model guidance not shown to beat StatNN guidance." |
| E3 | Reported in full, including any loss to direct best-of-N or LoRA SFT. |
