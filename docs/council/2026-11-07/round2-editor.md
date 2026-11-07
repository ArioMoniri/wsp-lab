# Round 2 — editor

All budgets, counts and decision thresholds below are prospective design choices, not measured results. Repository paths identify inspected evidence. This report authorizes no execution.

## Concessions

To representation: my cached activation contraction needs task sensitivity and a same-support gradient comparator. Otherwise the critic could simply rediscover a gradient step. To world-model: preserve predecessor-relative direction and exact support; another layer-token encoder is insufficient. To metrology: deterministic scoring does not establish item resolution, and WikiText cannot substitute for task success. MMLU qualification is unresolved; neither restored chain bytes nor my original small episode panel establishes confirmation power (`docs/council/2026-11-07/round1-{representation,world-model,metrology}.md`; `docs/HYPOTHESIS.md`). I withdraw any implication that the original shadow panel could establish H5.

## Rebuttals with evidence

Representation's zero-shot sum of objective gradient and positive Fisher quadratic needs correction. Its own specification uses model-sampled WikiText Fisher factors for several different objectives (`docs/council/2026-11-07/round1-representation.md`, §3). Such a quadratic is not automatically each objective's signed Hessian correction. Use it as a trust-region feature; learn objective-specific residual corrections. Likewise “label-free” means no edited-model outcome labels: computing gold-task gradients still uses task supervision.

World-model's directional intervention is valuable, but its sampled source blocks do not establish transfer to editor-selected winners. That report explicitly reserves a separate editor panel (§Models editing models). Selection calibration must be episode-held-out, including support selection, rather than inferred from source-edit prediction.

Metrology is right about qualification, but per-item prediction is a diagnostic route, not the operational H5 endpoint. H5 requires at least 30% fewer real evaluations at equal final score against the same editor and Adam/Muon (`docs/HYPOTHESIS.md`, H5). A residual-loss win alone cannot satisfy it.

My round-1 implementation assumption also needs narrowing: `src/wsp/editor/driver.py::run_episode` calls `validate_production` for every provider except exact `ScriptedEditor`. A live-editor pilot cannot quietly use the offline exception. Numerical caches and sensitivity fields require an external critic adapter; they are not an existing harness capability.

## Revised single proposal: support-conditioned local response critic

Consume validated effective delta, exact changed coordinates, rounding loss, pinned predecessor and structured operation metadata. For MLP writers cache post-gating inputs on a frozen, task-scoring-disjoint probe panel; compute signed delta-times-input responses. Add representation's per-objective gradient contractions and Fisher magnitude, plus world-model's predecessor/base-aligned coordinate-block contractions and displacement. Use a regularized residual head over this local chart, constrained to predict zero change for zero delta. Treat Fisher magnitude as uncertainty/context, not signed task curvature. Recompute at a changed predecessor or abstain. Start with writers only.

Rationale: **no**. Preserve pre-evaluation transcripts for audit; no rationale ablation in this first purchase. Predict item-score changes, aggregate task gain and collateral loss, and estimate probability of reaching the frozen goal within guardrails. WikiText is initially a distribution-sensitivity diagnostic; loop repair requires separately qualified generation outcomes.

Calibrate intervals on entire held-out episodes, sharing item resamples across candidates and references. Freeze thresholds on development data and validate them on a separate calibration partition. Reject only when the upper utility bound cannot reach the goal; uncertain, novel-support or out-of-radius edits receive real evaluation. Require the one-sided 95% upper bound on missed-qualifying-proposal frequency among rejected candidates to be ≤5%, with episode clustering and selected-candidate reporting. Insufficient calibration evidence means no rejection. Shadow-score every candidate during the pilot; never claim its counterfactual savings as realized savings.

## First pilot: hard ceiling 10 GPU-hours

Use existing `ScriptedEditor`, `driver.run_episode`, API validation, sandbox, `Episode` failure logging and zoo registration. Prepare frozen scripts outside scoring; an external adapter consumes registered effective deltas. Keep production disabled. This is a harness/structure pilot, explicitly not evidence about live LLM proposal quality (`src/wsp/editor/driver.py`, `api.py`, `episode.py`).

Proposed ten independent source/support blocks, each with exactly these six proposal arms:

| Arm | Proposals |
|---|---:|
| A: localized scripted writer edit D | 10 |
| B: same-support gradient step, effective norm matched to A | 10 |
| C: support-preserving orientation randomization, norm matched | 10 |
| D: −D | 10 |
| E: D/2 | 10 |
| F: D/4 | 10 |

Total 60 attempted proposals; failures remain in denominators and are not replaced. Split whole blocks 6/2/2 for fitting/calibration/held-out feasibility; these counts cannot certify the rejection bound. On identical rows compare the full critic, field-only, chart-only, StatNN, augmented StatNN, descriptors, norm, activation-only and correspondence-shuffled critic. No extra proposals for predictor ablations.

Score WikiText sensitivity; collect TF-v1 task vectors only under the qualification/amendment gates, otherwise leave task claims untested. Price conservatively using full TF scoring: 60×1 minute = 1 GPU-hour; reserve 60×54 seconds = 0.9 hours for extraction even if CPU; both anchors are in `docs/council/2026-11-07/BRIEF.md`. Allocate hard caps of 2 hours to fields/cache, 2 to generation/probes, 2 to references/repeats and 2.1 contingency: total 10. These are allocations, not throughput claims. Stop at any exhausted allocation; an incomplete pilot is a feasibility failure, not evidence of prediction failure.

## Decisive H5 test and kill criterion

After qualification, freeze critic and rejection policy; run fresh, paired live-editor episodes. Compare critic-guided search, identical editor without critic, random ranking, same-support gradient search, single-draw best-of-N and Adam/Muon. Best-of-N gets one score draw per candidate, identical candidate opportunities and equal real-evaluation budget; no repeated-draw archive maximum. Charge editor diagnostics, gradients, probes, failed attempts and amortized training; separately match GPU-seconds.

Pass requires ≥30% evaluation savings with paired 95% lower bound ≥30%, final-score difference lower bound ≥−1 percentage point (prospective noninferiority margin), guardrails satisfied, rejection bound ≤5%, and positive net GPU savings. Require ≥80% simulated joint episode/item power before execution. Otherwise no H5 success. Pilot timing and variance must determine episode count and price: total GPU-hours = setup + episodes × measured per-arm GPU-hours + sealed scoring. No honest fixed confirmatory price exists yet; the only purchased design here is the capped 10-hour pilot.

Kill useful critic deployment if a qualified, powered test gives savings upper bound <30%, or noninferiority fails. Kill the structural mechanism if its powered incremental residual-MAE gain upper bound is <5% against augmented StatNN and field-only. Calibration, power or budget failure leaves H5 unresolved and stops deployment.
