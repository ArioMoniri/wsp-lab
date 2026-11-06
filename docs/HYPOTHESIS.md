# Hypotheses — amended 6 November 2026

## Dated changelog

**2026-11-05 — Board A revision:** H1 becomes H1a/H1b; H2 becomes closure-first within-M2 history ablation with placebo equivalence; H3 requires independent operator/source holdout; H4–H6 are parked. Adopted defaults: TF-v1 yes; ≤30 local GPU-h yes; additional history no; new Lambda no. All new thresholds are prospective criteria. [PREREG_H1_TF](PREREG_H1_TF.md) is binding; historical wording below remains archival. Original H1 was not demonstrated, H2/H3 are undetermined, H4–H6 untested ([Board A](reviews/P4/board-a-synthesis-and-v2-2026-11-05.md)). No new qualification or power result is claimed.

**2026-11-06 — Board A re-qualification adjudication; PREREG amendment A4:** on the 229 D3 stratum entries MMLU-Pro gold log-prob failed A2.1 (ii) (max h 0.1137 > 0.10) and (iii) (P upper 97.5% 0.2264 > 0.20), so **H1a is untested, not falsified**; no sealed MMLU-Pro scoring (decision (c)); the sealed partition stays unspent. **H1w** (a separately named exploratory claim on the qualified WikiText log-PPL endpoint) is **registered** by A4; it is not a replacement for H1a and may not be cited as H1a. No H1w estimate existed when A4 was committed. H1b is unaffected ([Board A](reviews/P5/board-a-requalification-adjudication-2026-11-06.md), [PREREG A4](PREREG_H1_TF.md)).

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
