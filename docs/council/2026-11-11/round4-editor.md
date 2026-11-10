1. Preserve Part A, its controls, and F5.5; external editing successes do not validate a critic (§6).
2. Add an outcome-blind external-edit panel, separate from both registered parts (§1).
3. Only the released Astra patch is presently an exact-replay candidate; recipes require reconstruction (§1).
4. Add a no-LLM difference-of-means plus LP comparator alongside random best-of-N (§5).
5. Treat supplied layers, scales, and budgets as prior information, not autonomous discovery (§5).
6. Prefer a separately registered GPT-6.1 Sol editor comparison; retain Astra’s reproduction identity (§5).
7. Keep ≥30% same-editor savings, but require comparison with the cheapest qualifying scripted method (§5).
8. Admit deletion and activation-derived edits to shadow testing; calibrate before rejection (§3).
9. Report activation access explicitly and keep new spectral features outside frozen arms (§2–4).
10. No new panel rescues funding or alters the joint line-stop rule (§6).

# Editor — round 4, 11 November 2026

Prospective recommendations only. Repository authority: `docs/PREREG_H1_TF.md` A5/F5.4–F5.5, `docs/HYPOTHESIS.md` H5, `docs/H5_DESIGN.md`, `docs/H5_PROTOCOL.md`. This updates my `docs/council/2026-11-10/round3-editor.md` position under `docs/council/2026-11-10/SYNTHESIS.md`; amendments require F8 before G3 exposure (`docs/council/2026-11-11/BRIEF.md`).

## 1. External edits: replay is not rediscovery

Yes: freeze identities, eligibility, base/tokenizer revisions, effective bf16 deltas, endpoint signs, chart predictions and abstentions **before our scores**. “Outcome-blind” means blind to our endpoints, not to published gains. Group related recipes by source; they are not independent discoveries.

Flags below are `(python_tool_access, forward_access, hidden_state_access, feedback_access)` during construction, excluding the sealed evaluator. Exact artifact replay needs `(1,0,0,0)`; reconstructing directions generally needs `(1,1,1,1)` with development-only feedback. Cached activations retain their acquisition provenance and cost.

| Edit | Exactness on our base; required construction access |
|---|---|
| Astra two-tensor patch | Exact replay is possible **conditional on** obtaining the published artifact, matching the pinned base and verifying resulting tensor hashes. The prose alone omits directions/columns/scale (`docs/H5_RESEARCH.md`). Replay `(1,0,0,0)`; rediscovery `(1,1,1,1)`. [Experiment 8](https://rikiyat.github.io/blog/weight-space-programming/) describes loop-direction subtraction, not a complete byte recipe. |
| WSP-Scope column | [Project page](https://wsp-scope.github.io/) specifies layer 23, column 507 and W′=W−u e₅₀₇ᵀ, with 1,023 changed bf16 values. Without numerical u, exact reproduction is unavailable; estimating u gives a new approximation. Reconstruction `(1,1,1,1)`; a released patch would permit replay. |
| aiXiv difference-of-means | [Abstract](https://aixiv.science/abs/aixiv.261103.000004) specifies an estimator, not sufficient prompts/statistics/bytes for exact replay. Re-estimation on our support is an approximate methodological reproduction, `(1,1,1,1)`; feedback-free fixed contrasts can use `(1,1,1,0)`, separately labeled. |
| Barry LP | Exactness unverified: primary page inaccessible; no coefficient/code artifact audited here. Proposed reconstruction needs `(1,1,1,1)` for direction estimation and validation; LP solving on frozen statistics alone needs `(1,0,0,0)`. Do not call this Barry replication until its objective/constraints are recovered. |
| SWEET | [Paper](https://arxiv.org/abs/2611.10974): counterfactual activation contrasts localize MLP neurons/heads and suppress contributions. Porting to our base is a recipe replication, not replay of published weights; `(1,1,1,1)`, with explicit head/module allowlists. |
| TFTV | [Paper](https://arxiv.org/abs/2609.09054): forward statistics compile steering into rank-one weight changes. New receiver/statistics imply new bytes; `(1,1,1,0)` for fixed support/strength, feedback flag on if tuned. |

The brief's 4–8 entries cost approximately **0.08–0.16 GPU-h for TF scoring alone**, derived from its ≈0.02/entry H0 rate; references, extraction, probes and reconstruction are extra (`docs/council/2026-11-11/BRIEF.md`). Unavailable artifacts remain unavailable, never invented replacements. Pin a bounded panel, not a quota filled with near-duplicates.

Use qualified WikiText; report MMLU descriptively because its TF gold-log-prob endpoint is unqualified (`docs/reviews/P5/tf-v1-requalification-2026-11-06.md`). This small dependent panel diagnoses gross sign/OOD failures, not calibrated accuracy or powered superiority. Wrong signs on resolved TF effects challenge transfer; unresolved TF effects cannot do so. Loop repair can improve generation without improving teacher-forced likelihood; external math gains prescribe no WikiText sign. No Part A/B pooling or gate rescue.

## 2. Access ladder

[Rane](https://arxiv.org/abs/2611.07610) reports frozen LODO domain classification at 42% from weight statistics, 61% from activations, 64% from outputs. This strengthens my qualitative prior for activation-assisted editing; it supplies **no numerical effect-size prior** for benchmark-change prediction or our base-gradient field. Domain identity is another target; weight statistics are not all weight information.

Report chart-only, field-only, hybrid and activation-assisted comparisons with that prior stated. A cached base-gradient field uses data/gradient access during construction even if inference consumes only deltas; distinguish setup access from per-candidate access and charge both (`docs/PREREG_H1_TF.md` A5; `docs/H5_DESIGN.md`).

## 3. Editor proposals and critic coverage

Produce base-relative gate deletions `D=−M⊙W`, activation-derived rank-one `D=uvᵀ`, and localized SWEET suppressions within the tensor allowlist; preserve effective bf16 rounding, code hashes, support, provenance and all failures (`docs/H5_DESIGN.md`). [Deletion Tuning](https://arxiv.org/abs/2611.10866) **learns** masks under post-training: an editor's heuristic mask is not its exact training-free reproduction. No-gradient H5 cannot silently import mask-training gradients.

For sparse deletion, first-order signed response is `⟨g,D⟩`; K-FAC magnitude is `tr(G D A Dᵀ)`. Sparse coordinates do not imply small curvature or low rank; correlations and gating can invalidate local predictions. Fisher magnitude is not signed loss change. Use actual gate-projection factors, never down-projection substitutes; missing coverage means abstention.

Prefer a small family-transfer panel to reopening scale diagnostics: it directly tests editor support mismatch. The 4B diagnostic is closed with unresolved effects and failed pairing (`docs/reviews/scale/scale4b-diagnostic-2026-11-10.md`). Freeze recipe/support/scales without endpoint selection; score all, group by construction source, retain sham and matched controls. Price scoring at the same TF rate, but measure construction separately.

**Training on these families is not required before a shadow pilot**: untouched families are precisely a transfer test. Production rejection requires independent proposal-distribution calibration and the missed-winner bound; if coverage fails, collect separate training/calibration episodes and re-freeze, never train on this test panel. G5 remains shadow-only; rationale is logged, not a predictive input (`docs/PREREG_H1_TF.md` F5.4/A5.7).

## 4. Receiver-relative features

[Behavioral Rate](https://arxiv.org/abs/2611.06598) establishes receiver-dependent compressibility and data-conditioned correction spectra, not recovery from raw adapter singular values alone. Candidate diagnostic: per-objective correction-gradient singular spectrum and the edit's overlap with its leading subspaces, computed on permitted support. This is an analogy requiring validation, not an established equivalent to their spectrum.

Compute/report it without changing predictions if desired; inserting it into a fitted frozen field changes the arm and requires a separately registered exploratory extension (`docs/PREREG_H1_TF.md` A5/F5 schema pins). A chart might predict compressed-minus-corrupted improvement from signed contractions of both effective deltas; it cannot infer that compression helps merely because rank falls. Defer that experiment until outcome-blind compression settings and receiver-specific labels exist.

## 5. Scripted baseline, editor and savings

**Add the scripted estimator.** aiXiv recovers 93% of Astra's practice-set gain after receiving layers, column budget and scale; this is conditional repair, not complete rediscovery. Its Opus/Sonnet runs stopped after 18–23 requests ([source](https://aixiv.science/abs/aixiv.261103.000004)). Freeze difference-of-means direction extraction, then LP column/scale selection with support/norm constraints, deterministic tie-breaking and development-only validation. Separate supplied-localization and discovery conditions; equalize supplied information with LLM arms. Keep random best-of-N, nested 1/8/32, unchanged (`docs/PREREG_H1_TF.md` F5.4).

Barry's “8× fewer weights” and Concept Country's orthogonal-control results remain **brief-reported, unverified here**: [Barry URL](https://nathan.rs/posts/weight-space-lp/) and [Concept Country URL](https://concept.country/research/weight-space-programming) failed web fetch and returned HTTP 403 directly. Neither quantitative claim is evidence in this recommendation. **No change to Part A controls:** no flaw has been established. The reported orthogonal control motivates an optional H5 control, not replacement of spectrum-matched rotation.

[IWIBench Table 2](https://arxiv.org/html/2611.08647) places Sol medium/high above Astra in macro reward, with task/control tradeoffs. Choose **Sol medium for a new practical-editor comparison**, retain Astra for registered reproduction, retain the cheap control, and pin versions; no silent substitution (`docs/H5_PROTOCOL.md`). This is not proof of Qwen superiority. Log voluntary early stops and failures, not just nominal budgets. [Biswas](https://glitch401.github.io/writing/editing-a-models-weights/) independently motivates scripted controls and untouched validation, without establishing score noninferiority.

Keep A5.7's **≥30% paired savings lower bound against the same editor**, score lower bound ≥−1 point, missed-winner upper bound ≤5%, positive net GPU savings and prospective joint power ≥0.80. Add a separate practical comparison against the cheapest qualifying script at equal final score/guardrails, charging diagnostics, validation, LP CPU time, editor tokens and amortized critic cost. A script needing no critic still needs verification. If it needs no search evaluations, percentage search savings against it are undefined; report absolute costs. Beating an expensive LLM's evaluation count alone cannot justify deploying a costlier critic system. Do not retrospectively redefine H5 success.

## 6. Line-stop and exact paper additions

Keep F5.5 verbatim: jointly require ≥15 resolved Part A pairs with CP agreement upper bound ≤0.6; all registered Part B orientation contrasts' conditional/joint upper bounds <0.05 at power ≥0.80; valid learned-versus-spectrum upper bounds ≤0. Missing/invalid/underpowered predicates leave the line unresolved; funding stops separately on registered no-go/Part A alternative (`docs/PREREG_H1_TF.md` F5.5). External repairs neither satisfy nor defeat this conjunction.

Add these sentences to the measurement paper, with the linked sources above:

“Activation-assisted editing and held-out delta-to-benchmark prediction are distinct tasks; successful loop repair does not establish a weight-space critic.”

“Conditional scripted reconstruction recovers much of the published editing gain, motivating equal-information scripted estimators alongside matched random search.”

“Domain-fingerprinting evidence favors activation and output features over pooled weight statistics, but does not estimate benchmark-change predictability.”

“Deletion masks and activation-derived rank-one edits broaden the proposal distribution; deployment requires calibration on that distribution.”

“Our registered line-stop concerns the tested 0.8B representations and endpoints, not universal impossibility of weight editing.”

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---|---|
| External panel | Exact patches/declared reconstructions | Proposed F8; frozen signs, separate TF endpoints | ≈0.08–0.16 scoring; overhead extra | Missing bytes exclude exact replay; no funding power |
| Family transfer | Deletions/rank-one/suppression | Proposed F8; untouched families, shadow forecasts | ≈0.02/entry scoring; construction unpriced | Uncalibrated family prohibits rejection |
| Script + Sol comparison | Conventional estimator versus LLM search | Proposed F8; preserve F5.4 best-of-N | Reprice within existing G5 ≤10 + random ≤6 caps; otherwise defer | Cost overrun stops/reprices; no superiority without matched information |
| G8 critic | Schedule fewer evaluations | A5.7 unchanged; add scripted cost comparison | Unpriced until G5 | Powered savings upper bound <30% or demonstrated score inferiority |
| Line-stop | Joint structural evidence | F5.5 unchanged | No added GPU | All three valid predicates jointly; funding separate |
