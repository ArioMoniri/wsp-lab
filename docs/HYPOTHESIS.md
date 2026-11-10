# Hypotheses — amended 6 November 2026

## Dated changelog

**2026-11-05 — Board A revision:** H1 becomes H1a/H1b; H2 becomes closure-first within-M2 history ablation with placebo equivalence; H3 requires independent operator/source holdout; H4–H6 are parked. Adopted defaults: TF-v1 yes; ≤30 local GPU-h yes; additional history no; new Lambda no. All new thresholds are prospective criteria. [PREREG_H1_TF](PREREG_H1_TF.md) is binding; historical wording below remains archival. Original H1 was not demonstrated, H2/H3 are undetermined, H4–H6 untested ([Board A](reviews/P4/board-a-synthesis-and-v2-2026-11-05.md)). No new qualification or power result is claimed.

**2026-11-06 — Board A re-qualification adjudication; PREREG amendment A4:** on the 229 D3 stratum entries MMLU-Pro gold log-prob failed A2.1 (ii) (max h 0.1137 > 0.10) and (iii) (P upper 97.5% 0.2264 > 0.20), so **H1a is untested, not falsified**; no sealed MMLU-Pro scoring (decision (c)); the sealed partition stays unspent. **H1w** (a separately named exploratory claim on the qualified WikiText log-PPL endpoint) is **registered** by A4; it is not a replacement for H1a and may not be cited as H1a. No H1w estimate existed when A4 was committed. H1b is unaffected ([Board A](reviews/P5/board-a-requalification-adjudication-2026-11-06.md), [PREREG A4](PREREG_H1_TF.md)). The same day, the 6 November data-loss incident (D8) left H2 **not testable on the surviving zoo** ([incident review](reviews/P5/board-a-incident-d8-2026-11-06.md), section 3).

**Status: untested (not falsified).** MMLU-Pro gold log-prob is unqualified (A2.1 (ii) and (iii) fail), the §5 gate was not met, and no sealed item will be scored (decision (c)). The D3 development estimate (θ̂ = 0.129, all intervals within 0.01 of zero) is exploratory.

## H1a — residual TF readability (primary)

> On sealed fresh source-closed components, estimate row-weighted relative residual MAE reduction of fixed StatNN versus training-fitted isotonic norm after training-only family, device and log-norm adjustment for predecessor-relative MMLU gold-letter log-prob.

Primary: normalized MMLU gold-letter log-prob, 1,000 disjoint items. θ=1−Σ|r−p_StatNN|/Σ|r−p_isotonic|. Pass only θ≥.15, lower 95% bound>0, half-width≤θ/2, ≥100 genuinely fresh source-closed components, measured qualification and prospective TF-specific ≥80% power at .15. Paired 4,000-component and joint component/item intervals must both pass. Qualified adequately powered upper bounds<.05 kill practical readability for this endpoint only; intermediate/wide intervals are inconclusive. Failed qualification/count/power means untested. Registration §§1–6 freezes the full pipeline and numerical guard.

## H1w — WikiText log-perplexity sensitivity (exploratory, registered by A4)

> On the development stratum {lora_sft, task_vector, chain}, estimate the row-weighted relative residual MAE reduction of fixed StatNN versus training-fitted isotonic norm, after training-only family, device and log-norm adjustment, for the predecessor-relative change in token-weighted WikiText log-perplexity (nats; 64 windows × 1,024 tokens, pinned revision).

**Status: exploratory, registered (A4, 6 Nov 2026).** A conditional predictive-loss contrast about generic distributional sensitivity; it is not task-likelihood readability, not H1a, and may not be cited as H1a. All six intervals are reported and none is decisive; "distinguishable from zero" only if the lower bounds of all six are > 0; no pass/kill label. It stays exploratory unless the A2.3 gate (a)–(e) is met on the WikiText D3 OOF losses; fresh scoring would first need a committed sealed WikiText window panel and the A3 bitwise re-check. Qualification (A2.1) is already met for WikiText (max h 0.0128 against δ = 0.0488). Registration A4 governs.

**Gate result (6 Nov 2026, [h1w-gate](reviews/P5/h1w-gate-2026-11-06.md)):** θ̂ = 0.0071 (229 rows, 120 components), all six intervals straddle zero (not distinguishable from zero); the registered-scale A2.3 gate FAILS on power only (lower bound at θ = 0.15: 0.73 conditional, 0.60 joint, against 0.80; null size and coverage pass for bootstrap-t); flexible-nuisance θ̂ = −0.049. H1w stays exploratory; no fresh scoring is recommended.

## H1b — superiority over StatNN (descriptive screen)

One base-anchored signed-projection candidate/head versus original and augmented-feature StatNN, identical labels/information/splits/seeds/budget. Stronger criterion: ≥5% residual MAE gain over both trees, positive paired intervals and MMLU ρ≥.6. The present screen is descriptive; later confirmation needs untouched outcomes, TF-specific ≥80% power and multiplicity protection. Freeze all features before any H1a unsealing, develop on D3 only. A tree winning is useful representation evidence, not world-model superiority. Registration §7 governs.

## H2 — closure-first within-M2 history ablation (compute parked)

**Status: not testable on the surviving zoo (6 Nov 2026, D8).** The registered test needs complete lineages of length four or more. After the artefact loss, 99 lineages have length four or more, only six keep their first four steps' delta bytes, and three of those six are evaluated ([incident review](reviews/P5/board-a-incident-d8-2026-11-06.md), section 3). Parked; no compute is authorised.

Compare true/masked/causally shuffled history inside the same M2, matched initialization/training budget and identical current cumulative state plus incremental edit; include lag/trend controls. Require complete measured ancestry/source closure, length≥4, complete-run holdout and new TF variance-based ≥80% power. Future pass: ≥15% residual MAE gain versus mask, lower joint 95% bound>0, also beat shuffle/lag. Length-one placebo's entire 95% CI must lie within prospective **[−.05,+.05] relative-loss equivalence margin**; non-significance is insufficient. Failed equivalence, SD>1.0, missing closure or unaffordable power stops unresolved; qualified powered upper bound<.05 rejects useful history gain for this design. Forty closures are feasibility only. Additional 10 h declined: cached census/design only this programme; future execution separately funded. Registration §7 governs.

## H3 — unseen edit-operator transfer (conditional, parked)

Reopen after qualified H1 gates with separate budget and power. Hold out merge operator and all sources/descendants from preprocessing, fitting and unlabeled pretraining; report parent-known and source/parent-unseen arms separately. Planned minimum 100 independent held-out components is a floor, followed by ≥80% paired-effect power. JEPA success requires qualified MMLU ρ≥.5, positive residual-MAE advantage over same-input supervised and StatNN controls, and explicit noncollapse qualification. Powered rank upper bound<.2 rejects transfer; insufficient independent support means untested. Original ablation/random→LoRA remains undetermined: M2 LoRA holdout and M3 rank logging do not demonstrate collapse-qualified transfer; M6 fails its acceptance gate (Board A §A).

## H4–H6 — parked; conditions for reopening

- **H4:** pinned reader identity/access, separate token budget/consent, identical frozen summaries, blinded edits, replicated prompts and powered in-family/novel-operator panels. Original directional claim needs predictor superiority in-family and reader superiority on novel operators, both with paired intervals. No paid comparison here.
- **H5:** offline harness only. Reopen after measurement/sandbox requirements, proposal-distribution calibration, editor consent/budget, priced sealed scoring and power. Preserve ≥30% real-evaluation savings at equal final score, counting all editor/probe/amortized costs. Shadow scores are not success; no production rejection. [Prior Board A protocol](reviews/H5/board-a-prereg-2026-11-04.md) and [H5 protocol](H5_PROTOCOL.md) remain additional gates.
- **H6:** reopen after fixed-base signal, pinned accessible models, shared-coordinate audit and separate budget/power. Whole-model holdout retains original ≥.4 rank on ≥6 qualified endpoints, calibration separate. Fewer endpoints requires explicit narrowed amendment, not original success. No second-base compute here.

## Historical hypotheses — preserved 3–4 November wording

The following original definitions and proposed scale work are archival, not current authorization. Target migration does not establish historical hypothesis success.


Base object: a base model θ₀ and an edit sequence e₁…e_T producing θ₁…θ_T. Each θ_t has a performance
vector y_t ∈ ℝ^K over K cheap benchmarks (see `docs/ROADMAP.md` §Benchmarks). Δθ_t = θ_t − θ_{t−1}.

## H1 — Readability (replicates the blog, cheaply)
Weight deltas carry enough information to predict the *sign and rough magnitude* of Δy_t
without running the model. Test: a predictor f(φ(Δθ_t), φ(θ_{t−1})) → Δy_t achieves Spearman ρ ≥ 0.6
per-benchmark on held-out edits of the *same* base model, beating a weight-statistics baseline (StatNN)
and a "norm of delta" baseline.
Falsified if ρ < 0.3 on all benchmarks after honest tuning.

## H2 — Sequence matters (the SSM claim)
A stateful sequence predictor over the edit trajectory (Mamba-2/Mamba-3 block or a small state-transformer)
beats a stateless per-edit predictor with the same features, because performance after edit t depends on
accumulated drift (e.g. second abliteration pass hurts more after a LoRA merge).
Test: ≥ 15 % relative MAE reduction on trajectories of length ≥ 4.
Falsified if the stateless model is within noise.

## H3 — JEPA latent generalises across edit families
A JEPA-style weight encoder (predict embedding of θ_t from embedding of θ_{t−1} + edit descriptor,
VICReg anti-collapse, à la JEPAWG) transfers: trained on abliteration + random-direction edits, it still
ranks LoRA fine-tunes correctly (ρ ≥ 0.5) without seeing LoRA edits.
Falsified if transfer ρ < 0.2.

## H4 — LLM-as-reader vs. trained predictor
Given the same weight delta (as text/summary statistics + SVD components), a frontier LLM asked to predict
Δy (the blog's "native reading") is less accurate than the trained predictor on in-distribution edits, but
more robust to novel edit families. Measured with the same held-out sets, both directions reported.

## Alignment-edit sub-study (user request)
For Qwen3.5-0.8B/4B/9B and Qwen3.8-27B: compare **base-instruct**, **classic abliterated** (huihui-style
directional ablation), and **heretic** (Optuna-tuned ARA / projected abliteration, KL-aware) on
capability benchmarks, refusal rate and KL-to-base. Then ask: can the predictor tell, from Δθ alone,
which method was used and how much capability was lost? This directly tests "auditing without evaluation".

## Non-goals
- No new abliteration technique; we only consume published edits and standard Heretic runs.
- No claim about gradient-free optimisation; we predict, we do not optimise (Phase 5 optional).

## H5 — Editor models and LLM-in-the-loop optimisation (added 4 Nov 2026, user request)
Edits need not come from algorithms only. Two further edit sources enter the zoo and the predictor's
forecast-then-verify protocol:
- `llm_edit`: an editor LLM (GPT-6 Astra, Claude, cheaper controls) receives the weights in a Python sandbox
  plus a natural-language goal and proposes a weight edit (blog Experiment 8 setting, bounded forward passes).
- `llm_optimizer`: the editor LLM replaces or steers the optimiser step (proposes parameter updates or
  per-step hyper-parameters instead of Adam/Muon) for a short fine-tuning run; the predictor acts as critic
  before each proposed step is applied.
H5: with the predictor as critic, LLM-proposed edits/steps reach a given benchmark gain with fewer real
evaluations than (a) the same editor without the critic and (b) Adam/Muon with equal forward-pass budget.
Falsified if the critic does not reduce evaluations by ≥ 30 % at equal final score. Board C assesses the
design before any run (docs/MODEL_CANDIDATES.md §LLM-in-the-loop).

*Note, 11 Nov 2026 (council round 4, [F8.7](PREREG_H1_TF.md); [synthesis](council/2026-11-11/SYNTHESIS.md) §§1, 4).*
The pilot gains a scripted-estimator arm: a difference-of-means direction plus linear-programme column/scale
selection, no LLM, given the same localisation the editor receives; a second variant chooses layers by field
ranking. Information supplied to any arm (layers, budget, scale) is logged as prior information, not discovery.
GPT-6.1 Sol (medium) is registered as the practical editor beside Astra, which is kept for reproduction; no silent
substitution. Evaluation savings are reported per proposal stream (LLM, scripted, random best-of-N). The ≥ 30 %
paired-savings criterion concerns the critic and is unchanged. Nothing here authorises an episode.

## P4 measurement qualification

P3 currently supplies cumulative-from-base delta features. H2 therefore measures
history beyond the latest cumulative snapshot, on successor trajectories only;
M1 already sees accumulated drift in that snapshot. For H3, M3's transition target
is the latent of these cumulative-from-base features, not absolute θ_t features.
Family holdouts exclude all descendants/sources with a held-out-family step in
full manifest ancestry, including unlabeled steps and chain `inner_family` values.
H1 uses measured predecessor-relative Δy; absolute levels are a separate baseline.

## H6 — Cross-base-model generalisation (added 4 Nov 2026, user request)
A predictor trained on edits of one base model should carry over to edits of a *different* base model when
its inputs are shape-agnostic (pooled per-tensor-type statistics, singular spectra, rounding statistics) and
edit descriptors are family-level only. H6: with the zoo extended to at least three base models of different
families and sizes (Qwen3.5-0.8B, LFM3-1.7B, Qwen3.5-4B; Gemma-3-270M as a small control), a predictor
trained on two of them reaches Spearman ρ ≥ 0.4 on the held-out model's edits for ≥ 6 of K benchmarks
without any target from that model. Falsified if ρ < 0.2 on all benchmarks. This is expected to be the hardest
hypothesis; a per-model calibration arm (≤ 20 labelled edits from the new model) is reported alongside.

## Amendment — 7 Nov 2026: H1-OR, oriented local response

**Status: exploratory draft for Board A.** [PREREG Appendix A5](PREREG_H1_TF.md#appendix-a5--oriented-local-response-h1-or) governs.
Authority: [chair synthesis version 3.1](council/2026-11-07/SYNTHESIS.md), §§3–6.
H1-OR supersedes the unfrozen H1b candidate prospectively; it changes no historical result or wording.
H1a stays **untested**; H1w stays **inconclusive** (synthesis §§1, 5).

Hypothesis: oriented base-sensitivity features predict local WikiText response to effective weight edits beyond matched field-absent controls.
Part A compares signed predictions with realised intact/rotated and ladder responses, conditional on the frozen field and fixed panel.
Part B estimates C=1−A_cand/A_aug, a same-information representation contrast, plus matched field-ablation and destruction contrasts.
The field is the single primary arm; chart and probe-response arms are secondary and cannot be promoted after outcomes (synthesis §4.2).
The inherited 0.15/0.05 funding thresholds are chair choices for this new estimand, not measured effects (synthesis §4.2).
A4.4's no-kill wording governs Part B; scientific and funding conclusions remain separate.

WikiText is sensitivity, not task readability; MMLU remains descriptive until qualified (synthesis §2).
Success will be reported as “weights plus base sensitivity” or “weights plus predecessor”.
The project's original claim, prediction from the weights and the edit alone, stays untested (synthesis verdict).
D3 WikiText θ̂=0.0071 and MMLU θ̂=0.129 were known before this design (synthesis §§1, 4.2).
The old sealed partition stays unspent and is not reused for G7 (synthesis §5).

H2 stays parked: the closure ceiling is 242 depth-at-least-four rows in 99 lineages, not certified independent components (synthesis §3 G6).
Closure is census; any history experiment needs its own prospective amendment.

H5's critic consumes validated incremental effective deltas, exact coordinates, rounding loss and pinned predecessors (synthesis §§2, 6).
Cached post-gating inputs are separately labelled activation assistance; no rationale is an input.
Keep all attempts, including invalid and rejected ones, in the ledger; preserve immutable snapshots and an accept/abstain adapter.
G5 is feasibility only; G8 needs fresh episodes, frozen calibration/rejection and joint savings/noninferiority/guardrail criteria (synthesis §§3, 4.4).
A powered negative structural contrast rejects only the registered representation's incremental benefit on that population.
It does not reject structural mechanisms generally; deployment decisions are separate (synthesis §4.4).


## 10 Nov 2026 prospective amendment — H1-OR and H5 (before freeze)

The [10 Nov synthesis](council/2026-11-10/SYNTHESIS.md) is implemented by
[PREREG A5 amendment F5.1–F5.7](PREREG_H1_TF.md); Board A reviews at freeze.
H1-OR keeps the base-anchored field primary. Add descriptive CPU LoL secondaries
L/R/K/S on realised-increment SVD factors (raw stored adapters sensitivity only),
LoRA-only and 386-row inherited folds, and K+F versus field-only/anchored-only.
Use zero-at-zero compact heads, nested early stopping and paired A5.4 ratio
intervals; no secondary funds a purchase or becomes primary after outcomes.
The family-only stop requires L:S upper bounds ≤0. The 0.8B line stops only
jointly when (a) Part A has ≥15 resolved rotation pairs with agreement upper
bound ≤0.6, (b) every registered Part B orientation contrast has upper bound
<0.05 at simulated power ≥0.80, and (c) L:S paired upper bounds ≤0 on LoRA
rows, with valid conditional/joint bounds as F5.5 specifies. Any missing/failing
predicate leaves the line unresolved; registered no-go still stops funding.
Alternative stops field funding and is never evidence of wrong signs (F5.7).

H5 gains F5.4's critic contract: critic schedules/abstains, measured held-out
support improvement authorises acceptance; rebuild each accepted update from
pinned base, verify/reload, compare matched random-permutation updates, and retain
rollback plus complete attempts. Log separate Python, forward, hidden-state and
feedback flags. Add matched random best-of-N, 32 draws per source block with
nested 1/8/32 curves, development-only single-winner selection and held-out
validation; report ensembles separately. Actual-incumbent targets and base-relative
storage remain distinct. A5.7's ≥30% realised savings and conjunctive deployment
gates remain binding; G5 is shadow feasibility and production stays disabled.
The 4B pilot is a parallel Lambda diagnostic under F5.3, never a gate or H6 result.
No compute is started and no historical hypothesis status is repaired.

## 11 Nov 2026 prospective note — H1-OR (council round 4, before any G3 outcome)

The [11 Nov synthesis](council/2026-11-11/SYNTHESIS.md) is recorded as [PREREG A5 amendment F8](PREREG_H1_TF.md); Board A reviews F8
before any F8 entry is scored, and Part A does not wait. The registered arms, Part A selection and F5.5 are unchanged.
F8 adds two descriptive panels with **no decision authority** (F8.6): an external-edit panel (F8-E; the released Astra patch,
published LP edits, the aiXiv difference-of-means reconstruction) and a family-transfer panel (F8-F; gate deletion, TFTV rank-one,
a Concept-Country-style rank-one). They cannot satisfy, fail or unresolve any F5.5 predicate or A5.5 step and fund nothing.
F8.8: the access-ladder report states Rane's ordering (weights 42 %, activations 61 %, outputs 64 % domain recovery on
Qwen3.5-0.8B LoRAs, arXiv 2611.07610) as a prior in one sentence; it is not an effect-size revision, and the field is described
as weights plus frozen receiver sensitivity, never data-free. No historical hypothesis status changes.

## 11 Nov 2026 (council round 5) prospective note — H5 and H1-OR (before any G3 outcome)

The [round-5 synthesis](council/2026-11-11/SYNTHESIS-r5.md) is recorded as [PREREG A5 amendment F9](PREREG_H1_TF.md), written while
Part A staging stood at 27 of 40 edits and before any G3, rotation, ladder or sealed outcome existed. The registered arms, the
Part A selection, F5.5 and the H5 success criterion (A5.7) are unchanged; nothing is rewritten above. Board A reviews F9 with the F8a
adopting amendment; Part A does not wait.

**H5 gains, inside the G5 caps:**

- **Field localisation (F9.4).** A fifth access flag, `field_localisation_access`, records whether the editor receives the field's
  per-tensor ranking (gradient saliency g⊙W, candidate-specific g⊙D, cached down-projection input norms). The editor's own verbal site
  ranking is logged **before** it sees that ranking. Paired localisation variants: scripted estimator with field localisation;
  scripted estimator on the LLM's chosen layers; LLM editor with and without the field ranking; a shuffled-localisation control at
  matched information and opportunity. The LLM is credited with localisation value only if the LLM-localised variant beats the
  field-localised variant on held-out gain per real evaluation, paired, under A5.7 accounting. *[Superseded 11 Nov 2026 by F9a.5(a) — see S4 in the "11 Nov 2026 (F9a reconciliation)" section below.]* Attribution patching is a new metered
  artefact and is not registered.
- **Test-time LoRA shadow sub-stream (F9.5).** 32 frozen development queries, one isolated base-relative LoRA each, the critic
  forecasting collateral change only, every forecast logged under F9.1 before scoring and all 32 scored on TF-v1. Shadow replay can
  reject an uneconomic gate; it cannot demonstrate realised savings, which remain A5.7's prospective paired G8 claim. It is not a G8
  arm.
- **Generation-endpoint prerequisite (F9.6).** No H5 math-editor claim may be made until the F8-E MATH-500 generation endpoint
  (natural-completion rate and strict correctness together, thinking-off baseline) is qualified. Until then it is not an H5 endpoint,
  and it never alters F5.5.

**H1-OR gains:**

- **Forecast-before-modify log (F9.1) and calibration statistics (F9.2).** An append-only, hash-committed JSONL row is written before
  scoring for every edit outside Part A (F8 entries, F9 panels, every H5 proposal including rejected ones, test-time adapters),
  holding a, q, κ, the anchor type, the selection status, the chart output or abstention and the G2 hashes. The primary statistic is
  E₀ = 1 − Σ(Δy − s·a)² / ΣΔy², with sign agreement, calibration intercept and slope, coverage and the predicted-resolvable × resolved
  2×2 as mandatory secondaries, clustered by independent source component. Below 30 independent clusters per family the log is
  entries only; 30–99 is exploratory; a precision statement needs ≥ 100.
- **Pruning ladder (F9.3, F9-P) as a descriptive family.** Unstructured deletion at 10, 20 and 50 % from Wanda, transplanted
  SparseGPT and count-matched random masks (18 entries), three native SparseGPT entries with compensation as a separately labelled
  family, and a 24-mask ranking panel (Kendall τ of realised WikiText change against a + q, q_A and −‖W⊙M‖²), within sparsity only.
  It runs after Part A and F8-F.

**F9.7: no decision authority.** F9 panels and statistics cannot satisfy, fail or unresolve any F5.5 predicate or A5.5 step, cannot
change A5.7, and fund nothing. A JEPAWG-style latent model is not registered; it may be proposed by a later amendment only if the
F9.1 log shows structured residuals the first-order terms cannot represent. No historical hypothesis status changes.


## 11 Nov 2026 (F9a reconciliation) — H5 and H1-OR prospective note reconciled with F9a (before any G3 outcome)

Source: [PREREG A5 amendment F9a](PREREG_H1_TF.md) F9a.5–F9a.6 (adopted 11 Nov 2026, before any G3 outcome). F9a.5(a) requires `docs/HYPOTHESIS.md` to be reconciled before the H5 pilot. The registered arms, Part A selection, F5.5 and the H5 success criterion (A5.7) are unchanged; no historical hypothesis status changes. The same clauses appear in `docs/H5_DESIGN.md` and `docs/H5_PROTOCOL.md`.

**(a) F9a.5 and F9a.6, quoted verbatim from [PREREG A5 amendment F9a](PREREG_H1_TF.md)** (`docs/PREREG_H1_TF.md`, clauses F9a.5–F9a.6, main `e3920b1`):

> **F9a.5 — H5 reconciliation (condition 5).** (a) *Documents.* The governing H5 texts are `docs/H5_DESIGN.md` "11 Nov 2026 (council
> round 5) amendment — localisation flag, shadow LoRA sub-stream, re-anchoring" and `docs/H5_PROTOCOL.md` "11 Nov 2026 (council
> round 5) amendment — paired localisation episodes, shadow LoRA stream, re-anchoring" (both added by `2ab8fab`, after Board A reviewed
> `f181f0a`), `docs/HYPOTHESIS.md` "11 Nov 2026 (council round 5) prospective note — H5 and H1-OR", and the round-4 11 Nov sections
> (F8.7), each read as amended by F8a.6 and by F9a.5–F9a.6. Those two clauses are appended verbatim with F8a.6 to both H5 documents,
> and HYPOTHESIS is reconciled, before the H5 pilot. Superseded there: "with a missed-harm upper bound ≤ 5 % (A5.7)" (replaced by
> F9a.6), "run by a pinned script with no model calls" (F8a.6(a)) and "measured about 43 s per row" (F9a.5(e)). (b) *Access and
> training meters,* pinned per episode in the G5 manifest before the pilot: `python_tool_access`, `forward_access`,
> `hidden_state_access`, `feedback_access`, `field_localisation_access`, `backward_enabled`, `gradient_exposed_to_editor`. F9.4
> episodes: `backward_enabled = 0`; `gradient_exposed_to_editor = 1` exactly when `field_localisation_access = 1` (g⊙W and g⊙D derive
> from the frozen G2 gradient; no fresh backward pass). `lora_shadow`: python 1, forward 1, hidden-state 0, feedback 1 (F9.5), where
> feedback means the next-token training loss on the query context only (no TF-v1, held-out or sealed signal reaches training);
> `backward_enabled = 1`; `gradient_exposed_to_editor = 0` (no LLM editor; only the harness optimiser consumes gradients). This
> permission covers only the harness-run training of `lora_shadow`: no LLM editor arm gains backward passes, and no other arm's
> registered access changes (A5.7's gradient-based comparators keep theirs). Meters per arm: forward requests and tokens, backward
> passes and tokens, optimiser steps, GPU-seconds and wall time. Optimiser, learning rate, step, rank and target-module limits are
> pinned in the G5 manifest before the first adapter is trained, chosen without any shadow outcome; F9a sets no values. (c)
> *Localisation sequencing and controls.* Candidate-independent inputs (per-tensor g⊙W, cached down-projection input norms) may be
> released before the first candidate; candidate-specific g⊙D exists only after that candidate's realised delta and is returned as
> metered feedback on it, never guiding its own construction. The verbal site ranking is requested and logged with its hash before any
> field ranking is released (harness-enforced) and is never a critic input. The shuffled control permutes tensor labels over the same
> ranking positions, one permutation per episode drawn before the episode from `PCG64(domain_seed("20261111:f9-shuffle:", <episode
> id>))`, at matched length, support, opportunities and cost. (d) *Paired causal contrast.* The only localisation-value contrast is
> the scripted estimator on the LLM's logged layers (variant 2) against the scripted estimator with field localisation (variant 1):
> same editor, localisation alone differs. LLM with versus without the field ranking and real versus shuffled ranking are reported
> descriptively and support no claim. Endpoint: GPE_x = Σ_e gain_{x,e} / Σ_e E_{x,e}, with gain the change in A5.7's bounded held-out
> score (percentage points) of the final incumbent over the starting incumbent and E all real evaluator calls including failed calls
> and audits (A5.7); Δ_loc = GPE₂ − GPE₁ with cluster influence u_k = (g₂,k − GPE₂·E₂,k)/ΣE₂ − (g₁,k − GPE₁·E₁,k)/ΣE₁ in A5.7's paired
> episode-cluster bootstrap-t (4,000 common draws, PCG64 seed 20261107, its failure rules). A zero or nonfinite denominator makes
> Δ_loc undefined, never a success. Localisation value is credited only if the two-sided 95 % lower bound of Δ_loc is > 0; a point
> improvement alone is insufficient; there is one primary contrast and no multiplicity correction. Aggregate sensitivity is not causal
> localisation or faithful self-report, and comparisons at unequal access stay labelled (F8a.6(a)). (e) *Increments, gradients,
> re-anchoring.* Candidates stay stored base-relative (F5.4). A forecast for an acceptance decision is relative to the actual
> incumbent: the realised increment W_cand − W_inc (bf16 states, fp32 difference) contracted with an exact incumbent gradient
> (`gradient_anchor = incumbent-exact`, fresh gradient hash) and base factors (`factor_anchor = base-factor`), or, labelled
> `approx:difference_of_base_forecasts`, f_base(W_cand − W₀) − f_base(W_inc − W₀). Re-anchoring does not turn a submitted base patch
> into an incumbent increment, and an exact gradient contraction is not an exact finite-edit forecast. All re-anchoring work (gradient
> recomputation, loads, failures) is metered and charged to the critic arm's net GPU (A5.7). The historical timings are not future
> rates: 854.718742 s over 20 rows, mean 42.736 s, with different predecessor supports and including load and save/hash work
> (`docs/reviews/council/board-a-g2-2026-11-11.md`), and 13305.825513 s over 141 donors, 94.3676 s each including I/O (power
> `fields_manifest.json` summary, per Board A); per-objective re-anchoring and factored-LoRA costs are unmeasured, measured in the
> pilot and repriced under F8a.6(f). (f) *LoRA scope and groups.* The shadow forecasts completed adapters' collateral TF-v1 change
> only, not on-query gain or whether training is worthwhile; a TF surrogate cannot supply A5.7's bounded task-score endpoint.
> Query/training contexts, support, calibration, acceptance and sealed groups are disjoint by item id and source document; query
> contexts are disjoint from every TF-v1 scored item, the G2 gradient panels (including the 11 WikiText articles and the 500 reserved
> groups, A5.3.2(iii)) and any sealed panel; overlap fails closed. (g) *Contract preserved.* All of A5.7 (arms, primary comparator,
> ratio-of-totals savings, noninferiority, guardrails, the missed-proposal guardrail with its all-rejected denominator and all-zero
> unresolved case, net GPU, prospective joint power ≥ 0.80, unresolved versus kill), the F5.4 critic contract, F8.7's per-stream
> reporting and F8a.6's retained arms and matched-information rule stand. F8a.6(f)'s drop order gains, first, `lora_shadow` as one
> unit, then the F9.4 paired set, then spec §6.6's order. G5 stays shadow-only.
>
> **F9a.6 — shadow limits (condition 6).** (a) *Harm, defined separately.* Adapter j is harmful if its WikiText contrast Δy_j = y(W₀ +
> D_j) − y(W₀) under `configs/eval_tf_v1.yaml` is resolved (A5.3 paired block interval excludes 0) and Δy_j > 0; other TF-v1 endpoints
> are descriptive; a missing or nonfinite score is `unresolved`. This is not A5.7's missed qualifying beneficial proposal, which a
> collateral forecast cannot estimate; A5.7's guardrail, denominator and unresolved cases stay unchanged. (b) *Gate frozen before
> shadow outcomes* (in the G5 manifest, before the first adapter is trained): evaluate adapter j if its zero-fit forecast a_j + q_j
> (WikiText field, `base-exact`, `base-factor`) exceeds h\*, otherwise skip; a field abstention is evaluated. One gate, never tuned on
> the 32 outcomes. (c) *All 32 scored.* Every adapter is trained and scored on TF-v1 whatever the gate says; the gate is applied
> counterfactually. (d) *Read-outs.* Missed harm M_h = #(skipped ∧ harmful)/#skipped (undefined if nothing is skipped) with its exact
> one-sided 95 % Clopper–Pearson upper bound, and #(skipped ∧ harmful)/#harmful. The report states that even 32 independent skips with
> zero misses give an upper bound of 1 − 0.05^(1/32) = 0.08937 > 0.05, and fewer skips or dependence make it worse. Replay saving
> = #skipped/32 verification evaluations, descriptive, with training cost charged in both arms; E₀ and sign agreement follow F9a.2. (e)
> *Limits.* No ≤ 5 % certification, no realised-savings claim and no G8 gate or condition comes from this pilot. The shadow may report
> a gate as uneconomic, descriptively, when measured forecast cost per adapter is at least the measured verification cost per skipped
> adapter. Training, audits and failures stay in both arms' cost accounting; A5.7's unresolved cases are retained.

Clause guide (each item is in the quoted text above): access and training meters — F9a.5(b); localisation sequencing and the shuffled control — F9a.5(c); paired causal
contrast — F9a.5(d); base-relative versus incumbent increments, exact-gradient provenance and charged re-anchoring — F9a.5(e); LoRA scope and disjoint groups — F9a.5(f);
the full A5.7 / F5.4 / F8.7 contract preserved, drop order and shadow-only G5 — F9a.5(g); missed harm defined separately from missed beneficial proposals — F9a.6(a); collateral
gate frozen before shadow outcomes — F9a.6(b); all 32 adapters scored — F9a.6(c); read-outs and the 0.08937 bound — F9a.6(d); no ≤ 5 % certification, no realised-savings claim, no G8
gate, training and audits charged in both arms — F9a.6(e).

**(b) Superseded sentences in the earlier 11 Nov sections.** The earlier text is not deleted; each sentence below is marked in place with a bracketed note. Replacement sentences
are restatements of F9a.5–F9a.6 above; the quoted clauses govern.

- The round-5 HYPOTHESIS note contains neither "no model calls", "≈ 43 s per row" nor "≤ 5 %" wording, so S1–S3 have nothing to mark here (they are marked in `docs/H5_DESIGN.md` and `docs/H5_PROTOCOL.md`; the same replacements apply to any HYPOTHESIS reading).

- **S4 (localisation credit; F9a.5(d), read as amended by F9a.5)** — `docs/HYPOTHESIS.md`, round-5 note, **Field localisation (F9.4)**. Superseded sentence (verbatim):
  > The LLM is credited with localisation value only if the LLM-localised variant beats the field-localised variant on held-out gain per real evaluation, paired, under A5.7 accounting.

  Replaced under F9a by:
  > Localisation value is credited only if the two-sided 95 % lower bound of Δ_loc = GPE₂ − GPE₁ is > 0 (scripted estimator on the LLM's logged layers, variant 2, against the scripted estimator with field localisation, variant 1; A5.7's paired episode-cluster bootstrap-t, 4,000 common draws, PCG64 seed 20261107); a point improvement alone is insufficient. LLM with versus without the field ranking, and real versus shuffled ranking, are reported descriptively and support no claim (F9a.5(d)).

- **Read as amended (no sentence replaced).** The round-5 note's "Shadow replay can reject an uneconomic gate; it cannot demonstrate realised savings" is consistent with F9a.6(e) and stays; it is bounded by F9a.6(e)'s "uneconomic gate" criterion (descriptive: measured forecast cost per adapter at least the measured verification cost per skipped adapter) and by the missed-harm definition in F9a.6(a)–(d).


**(c) The ≈ 43 s per row re-anchoring timing is context, not a registered rate.** The figure quoted in the earlier 11 Nov sections ("about 43 s per row") is a historical F6-audit
measurement: 854.718742 s over 20 rows, mean 42.736 s, with different predecessor supports and including load and save/hash work
(`docs/reviews/council/board-a-g2-2026-11-11.md`); the other measured contraction is 13305.825513 s over 141 donors, 94.3676 s each including I/O (power
`fields_manifest.json` summary, per Board A). Under F9a.5(e) neither is a future rate. Per-objective re-anchoring and factored-LoRA costs are unmeasured; they are
measured in the pilot and repriced under F8a.6(f). Every second of re-anchoring is metered and charged to the critic arm's net GPU; nothing in this document registers a per-row cost.

**Consistency.** The same F9a.5–F9a.6 text, the same replacement sentences S1–S4 (each listed in the document where the superseded sentence occurs) and this timing statement are appended to `docs/H5_DESIGN.md`, `docs/H5_PROTOCOL.md` and
`docs/HYPOTHESIS.md`; if any wording elsewhere in the three documents differs from the quoted clauses, the quoted clauses govern. Nothing here registers a new arm, cap,
threshold or access value, authorises an episode, or changes A5.7, F5.4, F8.7 or the H5 success criterion. This section does not itself append F8a.6; F8a.6(a)'s replacement
sentence is quoted under S1 only because F9a.5(a) cites it. G5 stays shadow-only.

## 11 Nov 2026 (F8a.6 verbatim append)

F8a.6 (A5, `docs/PREREG_H1_TF.md`) requires its own text to be appended to the H5 documents. It follows verbatim; the F9a reconciliation section above supersedes nothing in it.

**F8a.6 — H5 reconciliation (condition 6).** Spec §6 (difference-of-means inputs, LP objective, deterministic selection and
ties, α grid, S-loc/S-field, treatments, Sol pin, per-stream contract, drop order) is adopted as the H5 scripted-arm contract and
must be appended verbatim with the overrides below to `docs/H5_DESIGN.md` and `docs/H5_PROTOCOL.md`, with `docs/HYPOTHESIS.md`
reconciled, before the H5 pilot. (a) *Access wording.* "run by a pinned script with no model calls … and nothing else the editor
does not" (`docs/H5_PROTOCOL.md` 11 Nov section; same text in `docs/H5_DESIGN.md`) is replaced by: "makes no LLM editor or provider
calls; it makes metered diagnostic model forwards (base generations and hidden activations, spec §6.1) through the same
forward-request meter, charged to the arm; S-field additionally receives the frozen G2 field as setup access". The actual access
flags of every arm (spec §6.3 table, plus F9.4's `field_localisation_access`) are pinned in the G5 manifest before the pilot.
Comparisons at unequal access are labelled and descriptive; they do not meet the matched-information requirement for an
editor-superiority claim. A T0 script given |L| and B is conditional on that prior, not unrestricted discovery. (b) *Retained arms.*
Astra stays the registered primary (A5.7 primary comparison: Astra with versus without the critic); Haiku, sham reload, random
ranking and the F5.4/FP5 nested 1/8/32 best-of-N are retained with every other A5.7/G5 comparator; arms are dropped only
prospectively for funding, never by observed performance. (c) *Complete A5.7 estimator per stream*, unchanged: ratio of total real
calls including failed calls and rejected-proposal audits; both totals finite and positive at the point and in every draw;
connected episodes clustered beyond the goal × seed × budget tuple when source-, direction- or prompt-connected; paired
episode-cluster bootstrap-t with 4,000 common draws, PCG64 seed 20261107, and its failure rules; bounded score endpoint with equal
episode weighting; missed-proposal denominator of all rejected candidates including failed attempts, unauditable rejected
attempts counted as missed, zero rejected attempts undefined; net GPU saving charging diagnostics, gradients, probes, failed
attempts and amortised preparation (including the G2 fields' 1.62 GPU-h, `g2_manifest_final.json` → `caps.fields.used_hours`);
prospective joint power ≥ 0.80; unresolved versus kill distinctions. Critic and no-critic share seeds and opportunity budgets
within a stream; realised calls are observed, not forced equal. A zero or nonfinite denominator makes a stream's savings undefined,
never a success. (d) *Frozen choices.* The α-selection metric is FP5's development-winner objective (mean paired improvement on
development support relative to the actual incumbent under A5.7's feasibility/safety predicates), ties within 1e-12 relative to
the smaller α; the proposal-training, support-validation, acceptance-slice, calibration and sealed splits (spec §6.1; F5.4; FP5;
A5.7) and the actual-incumbent comparisons are frozen in the G5 manifest before calibration. (e) *F9.4.* The F9.4 paired set
(scripted with field localisation, scripted on the LLM's layers, LLM with and without the field ranking, shuffled-localisation
control) uses the same estimator and access pinning; the LLM is credited with localisation value only under F9.4's paired rule.
(f) *Cost and drop order.* Every H5 stream here is unpriced (F8a.8). The pilot is repriced inside the existing G5 caps (≤ 10 GPU-h
plus ≤ 6 for best-of-N) before launch; if it does not fit, streams are dropped in this fixed order: the F9.4 paired set as one unit
(registered last), then spec §6.6's order (S-field; T1 as one unit; Sol T0). Astra, Haiku, sham and random best-of-N are never
dropped; dropped streams are reported as unfunded. G5 stays shadow-only, G8 keeps its existing eligibility (G5 feasibility plus
G3/G4 go), and every A5.7 rejection and acceptance safeguard is retained. No episode is authorised by this block.

