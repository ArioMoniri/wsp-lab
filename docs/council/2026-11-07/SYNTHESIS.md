# Council decision — 7 November 2026 (chair synthesis, version 3.1)

Branch `council/2026-11-07`. Reviews: version 1 and version 2 by Boards A, B and C
(`docs/reviews/council/board-{a,b,c}-synthesis-2026-11-07.md`, `board-{a,b,c}-synthesis-v2-2026-11-07.md`); version 3 by Board A
(`board-a-synthesis-v3-2026-11-07.md`, ACCEPT WITH REQUIRED EDITS as a programme proposal). Version 3.1 applies those four edits. Inputs, read in order: [BRIEF](BRIEF.md); round 1 [world-model](round1-world-model.md),
[representation](round1-representation.md), [metrology](round1-metrology.md), [editor](round1-editor.md); round 2
[world-model](round2-world-model.md), [representation](round2-representation.md), [metrology](round2-metrology.md),
[editor](round2-editor.md); the 5 Nov [SYNTHESIS](../2026-11-05/SYNTHESIS.md).

**What this document is.** A programme proposal with its decision rules. It launches nothing. It is not the statistical
registration: that is appendix A5 of `docs/PREREG_H1_TF.md`, a separate deliverable listed in section 5, which Board A reviews
and which is frozen by hash **before gate G2 computes any field**. Section 4 fixes the rules A5 must contain; section 6 lists
the schema fields A5 must fill in.

**Thresholds.** 0.15 and 0.05 are inherited from PREREG §4 and §7 and are re-used here, by chair choice, for a new estimand
(section 4.2). 0.5 and 0.6 (section 4.1) and the 15-pair floor are chair choices. All were fixed before any new outcome exists.

**Terms.** *StatNN*: a gradient-boosted tree on pooled weight statistics. *Descriptor-only*: the same tree on the edit
descriptor alone (family, norm, support). *Residual MAE*: the mean absolute error of a predictor after the registered nuisance
model (family, device, log-norm) has been fitted on the training fold and subtracted. *θ*: PREREG §4's relative residual-error
reduction of StatNN against an isotonic fit on the norm. *TF-v1*: the teacher-forced scoring tier (no sampling). *H1a*: residual
readability of MMLU gold-letter log-prob; *H1w*: residual sensitivity of WikiText log-perplexity. *bf16*: the 16-bit storage
format of the weights. *Effective delta* Δ_eff = bf16(W + Δ) − W: the change actually stored. *Support*: the tensors and
coordinate blocks an edit changes. *Half-width*: half the width of a 95 % interval. *Tripwire*: PREREG's item-precision stop rule.
*Census*: a count of what exists. *Fisher*: the expected outer product of the loss gradient; *K-FAC*: its per-layer
Kronecker-factored approximation, one factor for layer inputs (A) and one for output gradients (G). *Field*: the
base-anchored sensitivity field of section 2. *Chart*: the oriented local response chart of section 2. *Ladder*: the same
edit scored at strengths 0.25 and 0.5. *Destruction control*: the same representation with its informative part destroyed
(a random field with matched norms; a field with layers permuted). *A1-lite*: the TF-v1 KL vector used as a cheap
activation-side comparator.

## Verdict

The negative and neutral results of 4–6 November do not show that no learnable response to a weight edit exists. They tested
pooled statistics, descriptors and neural encoders whose direction tokens sit in a random projection frame. They used one base
model, 297 rows, and a measurement tier that could not resolve the task endpoint. The four members agree that the next step is a
controlled intervention before another encoder is trained. The representation member argues that the missing ingredient is the
sign of an edit's effect relative to the base model, which pooled statistics barely encode. The other three treat this as a
mechanism worth testing, not an established cause.

Both proposed structures read more than the edit. The field uses gradients of the base model on labelled panels. The chart uses
the predecessor's weights. A success will be reported as "weights plus base sensitivity" or "weights plus predecessor", never as
weight reading alone. The project's original claim, prediction from the weights and the edit alone, stays untested.

The decision experiment is small. Part A is a rotation-and-ladder intervention on 40 existing edits, about 2 GPU-hours. Part B is
an exploratory retrospective on the regenerated teacher-forced panel, on CPU. Together they decide whether sealed rows are
bought. The other spend (regeneration, rescoring, the editor harness pilot, chain closure) is infrastructure and census. The user
approved it on 7 November; it is not conditioned on Part A.

## 1. What each negative tested

| Result (brief) | What it tested | Verdict | Strength | File |
|---|---|---|---|---|
| 1 trees ≥ neural; descriptor-only within 0.07 | a pooled-statistics tree, descriptors, and encoders on randomly framed directions; 297 rows, one base | family and norm explain most of the rank; no tested representation shows weight-specific signal beyond descriptors | conditional; does not test base- or predecessor-anchored directions | `docs/reviews/P4/d3v2r-summary-2026-11-06.md` |
| 2 held-out family | pooled rank across families | confounded estimate: between-family calibration offsets dominate the pooled ρ; an offset-free family test has not been run | not a structure test | `docs/RESULTS.md` (P4 section) |
| 3 history (H2) | M2 vs stateless M2 | never tested: 18 rows / 10 clusters at length ≥ 4; length-one placebo 0.1265 [0.039, 0.206] | sound | `docs/council/2026-11-05/round2_metrology_output.txt` §E |
| 4a H1a | residual readability | untested: endpoint unqualified | sound | `docs/reviews/P5/tf-v1-requalification-2026-11-06.md` |
| 4b H1w | residual sensitivity of pooled statistics | inconclusive in both directions: θ̂ near zero, all six intervals include zero, power below the gate | bears on one contrast for one endpoint | `docs/reviews/P5/h1w-gate-2026-11-06.md` |
| 5 measurement facts | sampled benchmarks | most zoo edits are behaviourally null; the labelled signal lives in a few families | measurement | `docs/RESULTS.md`, `docs/council/2026-11-05/metrology_budget_output.txt` |
| 6 tree importances | association | bucket-0 stable rank and entropy act largely as family and support indicators | weak clue | `docs/council/2026-11-05/evidence-statnn-importances.md` |
| 7 the blog | diagnosis and repair with feedback | Experiment 7 failed from weights alone in four attempts and succeeded with attention patterns; Experiment 8 allowed diagnostic forward passes; neither predicts a held-out benchmark change from weights | verified by three members in round 1 | https://rikiyat.github.io/blog/weight-space-programming/ |

Evidence behind rows 4a and 4b:

| Quantity | Value | Gate | File |
|---|---|---|---|
| MMLU max item half-width | 0.1137 | ≤ 0.10 | `tf-v1-requalification-2026-11-06.md` §A |
| MMLU tripwire upper bound | 0.2264 | ≤ 0.20 | same |
| WikiText max half-width | 0.01282 | ≤ 0.04879 | `h1w-gate-2026-11-06.md` §2 |
| WikiText reliability lower bound | 0.9923 | ≥ 0.90 | same |
| H1w θ̂ | 0.0071 | — | same §1 |
| H1w power lower bound at θ = 0.15, conditional / joint | 0.728 / 0.600 | ≥ 0.80 | same §4 |

The headline "the negatives tested the representation, not structure" is the hypothesis Parts A and B test. The evidence supports
"does not exclude a structure", no more.

## 2. The two structures, and where the members disagree

**Sensitivity field (representation).** Computed once on the base, before any outcome label is seen. For each objective k
(WikiText NLL, MMLU gold-letter log-prob, HellaSwag, refusal margin) a gradient g_k on a panel disjoint by text group from every
scored item, plus K-FAC factors (A_t, G_t) per supported linear module. Each edited tensor t becomes a token: the signed
contraction ⟨g_k, ΔW_t⟩, the magnitude term q_t = ½ tr(G_t ΔW_t A_t ΔW_tᵀ), the Frobenius norm, the layer index and the role.
"Label-free" means no edited-model outcome labels; the gold-task gradients still use task supervision.

**Oriented local response chart (world-model).** Consumes the actual predecessor W, the displacement W − W₀ and the incremental
effective edit D separately. Adds signed coordinate-block contractions ⟨D, P_out W P_in⟩ with frozen projectors, exact layer,
support, spectral tail and rounding loss. A small within-layer set encoder with a response head constrained to zero at zero edit.

**Where they stand.** The representation member holds that the field is the first-order expansion of the chart's response map.
The world-model member calls them complementary, keeps the chart as its primary test, and requires a field-only comparator before
field tokens are fed to the chart. Board C adds that a weight contraction is not the loss Jacobian and a Fisher factor is not the
Hessian. The chair adopts Board C's framing: the two are competing representations, run on identical rows and folds in Part B,
and neither is assumed to contain the other.

**Corrections the chair adopts** (raised by the world-model and editor members; the representation member's round 2 does not
address them). q_t is nonnegative for covariance factors and is not a signed curvature correction for every endpoint. The sign of
a zero-shot prediction therefore comes from ⟨g_k, ΔW⟩ alone; q_t enters only as a trust-region feature. The chain cross term is a
stateless feature of the current displacement; it belongs in the identical-current-state control and never in an H2 claim.

**Models editing models (editor).** The critic consumes the validated effective delta, the exact changed coordinates, rounding
loss, the pinned predecessor and structured operation metadata. For MLP writers it adds cached post-gating inputs on a frozen
scoring-disjoint probe panel. Field and chart tokens may be added as separately named arms. It predicts item score changes,
aggregate task gain and collateral loss, with episode-clustered calibrated abstention. It never reads the editor's rationale;
transcripts are kept for audit. Every proposal, including failed and rejected ones, stays in the denominator.

### Adjudicated disagreements

| Disagreement | Positions | Chair decision | Reverses it |
|---|---|---|---|
| Endpoint | WikiText first (representation; metrology r2, on a fresh sealed panel) vs MMLU decisive (world-model r2; metrology r1) | WikiText first as the mechanism endpoint; MMLU co-registered but descriptive until it qualifies. A WikiText-only pass is sensitivity, not task readability. | MMLU qualifies before G2; then A5 registers one co-primary rule and one arm family before any new endpoint outcome. |
| Buy sealed rows now | 20 + 100 blocks as a floor (world-model) vs intervention first (representation, metrology) | Intervention first. The world-model's own arithmetic needs ≈ 262 clusters for a 15 % feature-contrast effect (`../2026-11-05/SYNTHESIS.md` §2); 100 blocks is a floor, not power. | Part A and B go, and a development-only simulation shows ≥ 0.80 joint and conditional power at ≤ 300 blocks. |
| Loss- vs weight-anchoring | field vs chart | Not adjudicated. Both arms, their destruction controls and the probe-response comparator run on identical rows and folds (Part B). The field is the single primary arm of this programme; the chart is a registered secondary and cannot be promoted after outcomes. | A later amendment, written before new outcomes, registers a chart-primary programme on the same rules. |
| Is the editor critic a weight-only test | editor r1 | No. Cached D·x is a writer-local response. It is an activation-assisted comparator with probe-only and probe-plus-descriptor controls and measured probe cost. | A field-only or chart-only arm matches it on identical rows. |
| Do the regenerated chains make H2 testable | — | No. The ceiling is 242 rows in 99 lineages against a historical stand-in of 162 lineages (section 3, G6). H2 stays parked; closure is funded as census. Any history ablation needs its own amendment and is not authorised here. | Independent new lineages with measured paired TF-v1 loss variance and ≥ 0.80 power, plus a length-one equivalence test. |
| Rationale as critic input | — | Excluded (unanimous). A blinded rationale ablation may follow a positive delta-only result, never precede it. | A delta-only critic passes and a pre-registered blinded ablation shows a further gain. |
| Metrology's power verdict | metrology r2: none of the three decisive tests is powered as written; D3 outcomes have already been examined | Accepted. Part B is exploratory development evidence. Part A is called decisive only after a development-only simulation of its yield and power. | The simulation shows ≥ 0.80 power for the registered Part A contrast. |

## 3. Programme and gates

Hardware: the local RTX PRO 4500 (device H0). TF-v1 scoring is qualified on H0 only. An H100 may run fast-tier census work;
TF-v1 on any other device needs its own qualification first. Figures marked *allocation* are caps under a stop-and-reprice rule,
not measurements. Measured units: 1.0–1.04 min per TF-v1 entry warm on H0; 54.3 CPU-s per feature extraction; 7.5 min per
fast-tier entry on H0 (`docs/reviews/P5/d8-regeneration-2026-11-06.md` §§2, 5; BRIEF).

**Exposure ledger (all gates).** Every use of a zoo component (scoring, qualification, development fitting, gradient or factor
estimation, probe caching, pilot, tuning) is recorded with its role. Any component used for development in any gate, with its
full closure (ancestry, aliases, shared adapters and training runs, task-vector operands, generated direction identities), is
ineligible for G7 freshness. This covers G0, G1, G2, G3 (the 40 components), G4, G5 (the pilot groups) and G6. Auxiliary panels
(gradients, factors, probes) are certified disjoint by text group from development, calibration and sealed scored items.
Committed manifests (source, alias, component, text group), model/code/config hashes, seeds and panel/cache/projector/control
identities precede the affected computation. No rerun or regeneration restores freshness.

| Gate | What | Decision rule | Cost |
|---|---|---|---|
| **G0** chain regeneration (running) | 358 snapshots replayed from edit.json; install; features; full-set P4 rerun (target 345 matched rows) as the X1 fingerprint; TF-v1 G1 for the 2 lost predecessors and the 14 missing (y, h) rows | random_dir and abliteration steps must be bitwise. A re-trained LoRA step is accepted within a **norm** relative error of 5e-3 against `edit.json` (correction 7 Nov, after the measurement: the matrix relative Frobenius error of a re-trained step is 2.8e-2–1.5e-1 in the pilot, so no matrix-level tolerance is met and none is claimed) and is labelled "re-trained"; its descendants carry the label and are excluded from fingerprint claims. Pilot evidence (`docs/reviews/P5/d8-regeneration-2026-11-06/chain_replay_pilot.json`, branch `feat/d8-chain-replay`, commit ae7548a): 8 of 11 steps bitwise; for the three LoRA steps only the **norm** differences were recorded (≤ 9.1e-4 relative), which bound the matrix error from below, not above. The matrix metric was added to the comparator afterwards and will be measured on the same chains before any fingerprint claim | LoRA re-training *allocation* ≈ 3 GPU-h; extraction 358 × 54.3 s = 5.4 CPU-h; P4 rerun 26.6 run-wall-h (historical sum of run wall-clock, up to 4 workers × 2 threads; core-hours not established) |
| **G1** P6d rescoring | TF-v1 D3 stratum panel at the frozen source. Scoring entries: 316 recoverable (239 non-chain + 77 chain; 167 stratum rows + 157 controls = 324 logical rows, 8 reused alias records) priced at ≈ 5.5 GPU-h in `docs/reviews/P5/board-a-incident-d8-2026-11-06.md`, plus 62 chain rows that depend on G0 at ≈ 1.05 GPU-h: 378 scoring entries, 386 logical rows. A canonical mapping delta → logical row → alias → actual reference is a launch artefact | records bitwise with the surviving fingerprints (G1/G2 rule of the incident review) | 5.5 + 1.05 |
| **G2** fields, chart features, probe cache | A5 frozen first. Then g_k and K-FAC factors (schema, section 6); chart projectors; probe cache; exact-gradient audit at 20 chain predecessors: relative error of ⟨g, D⟩ against the base-plus-cross-term approximation ≤ 0.1, with an absolute floor of 1e-3 nats on the denominator, on ≥ 16 of 20; fewer is a hard stop of G2 pending a new amendment (section 4.0). Peak VRAM, throughput, CPU/disk bytes and reference-score costs recorded | hard caps; exceeding one stops the gate and triggers repricing, never a silent cut in coverage; a supported module without hooks fails the gate | *allocation* ≤ 2 GPU-h fields and cache, ≤ 1 GPU-h audit |
| **G3 Part A** rotation and ladder | 30 lora_sft and 10 chain D3 edits, one per certified source component, chosen by sorted id within effective-norm tertiles (ties by id) before any new variant outcome. Per edit: one rotation R at s = 1 and the edit at s ∈ {0.25, 0.5}, each applied at the actual predecessor and rounded. 120 new TF-v1 WikiText scores on the registered 64 windows, plus the predecessor references that are not already on file (enumerated in A5) | section 4.1 | 120 × 1.04 min ≈ 2.1 GPU-h plus references (priced in A5) |
| **G4 Part B** retrospective (exploratory) | registered H1w folds, nuisance and intervals on the G1 panel (386 logical rows if G0 lands, else 324) | section 4.2; go / screen / no-go / unresolved for buying sealed rows; no kill (PREREG A4.4) | CPU |
| **G5** editor harness pilot | `ScriptedEditor` only, production disabled; 10 source/support blocks × 6 arms (D, same-support gradient step, orientation randomisation, −D, D/2, D/4) = 60 proposals; whole connected episode/source/direction/prompt groups split 6/2/2; arms: full critic, field-only, chart-only, StatNN, augmented StatNN, descriptors, norm, activation-only, correspondence-shuffled | feasibility and calibration variance only; no rejection, no H5 claim; an exhausted allocation is a feasibility failure, not evidence about prediction | *allocation* ≤ 10 GPU-h: 1.0 scoring, 0.9 reserve for extraction (CPU in practice), 2 fields/cache, 2 generation/probes, 2 references/repeats, 2.1 contingency |
| **G6** chain closure (census) | TF-v1 scoring of the 296 never-evaluated chain entries (358 lost − 62 already evaluated); features | census only. Ceiling after closure: 242 depth ≥ 4 rows in 99 roots, up from 36 rows in 16 roots (`round2_metrology_output.txt` §A). The 162-lineage figure is the §E stand-in from 18 rows / 10 clusters on the old endpoint, not a certified TF-v1 requirement. Roots bound independent source components from above. Every row scored here is exposed | 296 × 1.0 min ≈ 4.9 GPU-h; 296 × 54.3 s ≈ 4.5 CPU-h |
| **G7** sealed purchase (conditional on G3 go and G4 go) | fresh source-closed component blocks, each with the edit and its rotation at strengths {−1, ½, 1}; a new sealed WikiText window panel disjoint from the 64 development windows; the single frozen arm, all fields, caches, projectors, controls and decision rules hashed before any sealed outcome, including sealed qualification | section 4.3 | conditionally costed: scoring 20 development blocks (120 rows) plus 100–300 sealed blocks (600–1,800 rows) at 1.04 min per row = 12.5–33.3 GPU-h; overheads *allocation* 2 GPU-h (14.5–35.3 if fully spent) |
| **G8** H5 prospective trial (conditional on G5 and a G3/G4 go) | fresh live-editor episodes; critic, calibration and rejection policy frozen before the first episode; calibration partition separate from evaluation | section 4.4 | not yet priced; set from G5 timing |

Allocations G0–G6 sum to 29.55 GPU-h as listed (3 + 6.55 + 3 + 2.1 + 10 + 4.9); without the 0.9 reserve that is CPU in
practice, **≈ 29 GPU-h** on H0. This is a cap, inside the user's approval; it is not a verification of that approval's scope.
CPU work is listed per gate and not summed, because the P4 figure is a run-wall sum.

## 4. Decision rules (content of PREREG appendix A5)

### 4.0 Decision function (one ordered evaluation; the first matching branch decides)

Equality conventions are stated per rule below: point thresholds written with ≥ or ≤ are inclusive; "lower bound > 0" and
"lower bound > 0.5" are strict. There is no blanket rule.

1. **Validity.** Missing manifests or hashes, a failed qualification, an exhausted allocation, or a development null-size
   simulation above 0.07 at nominal 0.05 → the gate is **unresolved**; nothing downstream runs. The G2 predecessor audit has one
   action: fewer than 16 of 20 passing predecessors is a **hard stop** of G2 pending a new amendment (exact-predecessor
   gradients may be registered there with their cost); it does not silently disable the approximation.
2. **Ladder alternative.** At s = 0.25, ≥ 15 resolved pairs and agreement upper bound ≤ 0.6 → **alternative**: the field is not
   pursued, no sealed purchase on the field arm, the probe-response comparator becomes development work only (it has no
   purchase criterion until a new amendment gives it one). This branch overrides a rotation go.
3. **Part A evidence.** Fewer than 15 resolved rotation pairs → **Part A unresolved**. Interval includes 0.5 → **Part A
   unresolved**. Interval entirely below 0.5 → **Part A unresolved**, with the observed disagreement reported (no scientific
   kill). Otherwise (≥ 15 pairs, lower bound > 0.5) → **Part A go**.
4. **Part B eligibility.** Part B is always computed and reported. It can fund only if Part A went and both destruction
   controls are beaten (lower bound > 0 on both); otherwise the funding decision is **unresolved** regardless of C, and the
   extension-only exception below does not apply.
5. **Funding (if eligible), evaluated in this order:** if C ≥ 0.15 and lower 95 % > 0 → **go**; else if C ≥ 0.05 → **screen**
   (Part A extension only, never sealed rows); else if upper 95 % < 0.05 and simulated power at C = 0.15 ≥ 0.80 → **no-go**
   (funding stops; not a scientific falsifier); else → **unresolved**. Intervals are never assumed to contain their point
   estimate; the branches are evaluated as written.
Scientific consequences are stated separately from funding consequences in every report.

### 4.1 Part A

*Analysis time.* One analysis, after all 120 variant scores and their references are on file and the ledger is closed.

*Null and claim.* The null is that the sign of ⟨g, D − R⟩ agrees with the sign of y(W + D) − y(W + R) with probability ½,
conditional on the frozen field and the fixed 64-window panel. The claim is agreement above chance under that conditioning; an
item-population claim is not made. The constant-sign comparator ("every trained edit degrades more than its rotation") is
reported with the same interval; it is not a registered comparison and no superiority over it is claimed.

*Unit and interval.* One pair per certified source component (40 components). Two-sided 95 % Clopper–Pearson interval on the
agreement count among resolved pairs. Lower bound > 0.5 requires at least 12 of 15 (lower bound 0.5191) or 27 of 40 (0.5087);
Board A's exact tails. A zero prediction counts as a disagreement. A missing or audit-excluded score removes the pair from the
resolved denominator and is listed in the full-panel ledger.

*Resolution.* d_i = y(W_i + D_i) − y(W_i + R_i) on the same windows. Its interval comes from a paired block bootstrap over the
32 two-window blocks with common draws across the edit, its rotation and the predecessor reference (fixed seed, 4,000 draws,
percentile interval). A pair is **resolved** if that interval excludes zero. The agreement estimand is conditional on resolution.
All 40 pairs are reported. The null size with the resolution step included is validated on a development-only simulation before
outcomes are read; if the simulated size exceeds 0.07 at nominal 0.05, the rule is unresolved and A5 is amended before scoring.

*Yield.* Fewer than 15 resolved pairs is **unresolved**: no replacement, no resampling. Board A's illustration: at an independent
resolution probability of 0.375 the chance of ≥ 15 resolved among 40 is ≈ 0.56. Yield and agreement power are simulated from
development contrast and noise assumptions before this purchase is called decisive.

*Go.* Part A **goes** when the rotation contrast has ≥ 15 resolved pairs and its lower bound exceeds 0.5; every other outcome is
listed in section 4.0, step 3.

*Ladder.* The contrast is y(W_i + s·Δ_i) − y(W_i) at s = 0.25 and s = 0.5, predicted sign from ⟨g, bf16(W_i + s·Δ_i) − W_i⟩,
same resolution rule, same interval, separate denominators per strength, reported as descriptive except for the alternative rule
of section 4.0, step 2 (upper bound ≤ 0.6 at s = 0.25 on ≥ 15 resolved pairs), which overrides a rotation go. A lower bound
below 0.5 alone is uninformative and triggers nothing. A pass at s = 0.25 with a fail at s ≥ 1 is a pattern to investigate, not a
conclusion.

*Rotation.* Per edited tensor t, the incremental effective edit is the realised bf16 difference D_t = bf16(W_t + Δ_t) − W_t at the
actual predecessor, subtracted in fp32 from the two stored bf16 states. Take the rectangular coordinate block of changed rows and
columns and compute the thin SVD of D_t on that block in fp32 (full SVD of the realised block; the raw LoRA factors may be used
only if they reproduce the realised block to a relative Frobenius error ≤ 1e-3, checked per tensor). Draw Q_out and Q_in as
Haar-orthogonal matrices on that block from a registered seed (one seed per edit, shared across its tensors so that a direction
common to several layers is rotated coherently; this choice is recorded). Form R_t = Q_out U Σ Vᵀ Q_inᵀ, apply W_t + R_t and
round to bf16. The scored object is bf16(W + R) − W. Preserved before rounding: singular values, Frobenius norm, rank,
confinement to the block, tensor names and shapes. Not preserved, and logged: coordinate moments, sign fractions, cosine to the
predecessor, directional tokens, and anything after rounding. A rotated LoRA is not an SFT-trained edit; "spectrum-matched"
refers to the pre-rounding construction. Post-rounding audit per tensor and per edit: relative Frobenius norm mismatch
|‖R_eff‖ − ‖D_eff‖| / ‖D_eff‖ ≤ 2e-2; relative spectrum mismatch (Frobenius distance of the sorted singular values) ≤ 5e-2; rank
within ±1 at relative tolerance 1e-3; block support identical. A failing pair is excluded by this rule, and the count rule is
"resolved among non-excluded pairs". Non-2D tensors, tensors with a zero or rank-degenerate block, and chain increments that are
not single-matrix edits are excluded from rotation and listed; coverage is reported per edit. Scaled edits are built by applying
s·Δ_t at W_t and rounding, not by scaling D_t. A nuisance-controlled sensitivity (logged v4 differences as covariates) is reported
next to the main contrast.

*Exposure.* D3 outcomes for the 40 edits are already known; selection is "before any new variant outcome", and Part A is
exploratory.

### 4.2 Part B

*Panel.* The G1 records, registered H1w folds, train-only nuisance (family, device, log-norm), seeds 0/1/2 averaged before the
loss, outer-held-out predictions audited for nested out-of-fold construction.

*Primary contrast (one).* C = 1 − A_cand / A_aug, where A is the row sum of held-out absolute residual errors after the same
training-only nuisance fit and seed averaging; cand = **StatNN + field tokens**; aug = **augmented StatNN**, the same tree with
the same settings, tuning budget and folds on identical inputs: v4 pooled statistics plus the deterministic per-tensor
transformation of section 6 (ordered exact-layer and role scalar columns of the chart and of the field, padded by a fixed rule),
so that the comparator receives every scalar the candidate receives. C therefore measures the candidate pipeline's performance
relative to a **same-information comparator**; it does not by itself identify the information the field adds. A5 must name the
actual difference between the two pipelines. **Field-information contrast (registered, decision-relevant):** C_abl = 1 −
A_cand / A_no-field, where the no-field arm is the identical pipeline (architecture, chart and predecessor access, rows, folds,
tuning opportunity) with the field columns removed; same interval. "Both destruction controls beaten" in section 4.0 is read as:
C_rand, C_perm **and** C_abl all have lower bound > 0. The random-field and permuted-field controls test particular destruction
hypotheses; the intact no-field ablation is what identifies the field's contribution. Interval: paired ancestry-cluster bootstrap-t on log(A_cand / A_aug), fixed
seed and draws, resampling whole components; conditional and joint item procedures as in PREREG A2.3, adapted to the A_aug
denominator; denominator failure (A_aug = 0 in a resample) is recorded and the draw dropped, with the count reported.

*Destruction controls.* C_rand = 1 − A_cand / A_rand-field and C_perm = 1 − A_cand / A_perm-field, same interval, alongside
C_abl above; "beaten" means lower bound > 0 on all three. *Secondary arms* (descriptive, no decision): zero-shot field (sign from ⟨g, ΔW⟩ only), ≤ 3-coefficient
field calibration, chart with its own destruction controls (permuted projectors; random-frame chart), A1-lite, descriptor-only,
delta-norm, original StatNN, the existing M1 and canonical LoRA tokens on the LoRA-only rows of the same folds (with their own
intervals and an inconclusive region), and the probe-response comparator (linear ΔW·x on a fixed scoring-disjoint 64-prompt
panel plus a regularised head; probe-only and probe-plus-descriptor controls; probe cost timed).

*Thresholds.* 0.15 is PREREG §4's θ threshold and 0.05 its §7 screen, fixed on 5 November before the H1w and MMLU estimates were
known. Their reuse for C is a chair choice made before any new outcome; A5 records it as such. The MMLU θ̂ of 0.129
(`docs/reviews/P5/h1a-companions-2026-11-06.md`) was known when this programme was designed; A5 discloses it.

*Decision.* Section 4.0, steps 4–5. A no-go stops funding and is not a scientific falsifier, because Part B reuses examined
outcomes and PREREG A4.4 forbids pass/kill language on this endpoint. There is no chart fallback in this programme. A chart-primary programme needs a new amendment written before new outcomes.

*Disclosure.* The D3 WikiText θ̂ and MMLU θ̂ were known before this design. The field and chart code, the arm list and the
thresholds are frozen by hash in A5 before G2 computes the first field.

### 4.3 G7 sealed purchase

Scientific success keeps the PREREG §§4–5 contract, re-stated for C on the new denominator: endpoint qualified on the new
sealed panel; item precision; conditional and joint calibrated intervals; independent component count; outcome-free power lower
bound ≥ 0.80 from a development-only simulation; C ≥ 0.15 with lower bound > 0 for the single frozen arm against augmented
StatNN; both destruction controls beaten; the intact-versus-rotated contrast positive with lower bound > 0. Source graph closure
as in the exposure ledger. A3 source-change qualification and matched actual-predecessor, device, dtype and batch identity are
required. No tuning, replacement or arm change after exposure. The panel is sized between 100 and 300 blocks by the simulation;
insufficient power at 300 is "unresolved", not a larger purchase.

### 4.4 G8 H5 trial

Arms: critic-guided search; the same editor without critic; random ranking; same-support gradient search; single-draw best-of-N
at equal real-evaluation budget (one score draw per candidate, no repeated-draw archive); Adam/Muon at equal forward budget. The
primary comparator is the same editor without critic; savings and non-inferiority must hold jointly against it; the other arms
are reported. Pass: realised real-evaluation savings ≥ 30 % with paired 95 % lower bound ≥ 30 %; final-score difference lower
bound ≥ −1 percentage point; guardrails met; one-sided 95 % upper bound on missed qualifying proposals among rejected ≤ 5 % with
episode clustering, denominator = all rejected candidates including failed attempts; positive net GPU saving after charging
diagnostics, gradients, probes, failed attempts and amortised training. Episode count and budget are set from G5 variance to
≥ 0.80 simulated joint power before any episode runs. **Kill** critic deployment only if a powered test shows a savings upper
bound < 30 % or demonstrated inferiority (upper bound of the score difference < −1 point); an interval too wide to show
non-inferiority is unresolved. A powered incremental residual-MAE gain with upper bound < 0.05 against augmented StatNN **and** field-only, on the contrast
fixed in A5 before any episode, **rejects the registered structural representation's incremental predictive benefit on that
population**; it does not kill the structural mechanism in general, and deployment kill is a separate decision. Calibration, power or
budget failure leaves H5 unresolved.

## 5. What this changes in the record

| File | Change |
|---|---|
| `docs/PREREG_H1_TF.md` | **Appendix A5** (the registration; Board A reviews it; frozen by hash before G2): sections 4.0–4.4 verbatim; the schema of section 6; the exposure ledger; enumerated references and their costs for G3; statement that A4.4 governs Part B; A4.6-style disclosure (D3 WikiText θ̂, MMLU θ̂ 0.129, design written after both); disposition of the unspent sealed partition `ebf0379a15d9…` (kept sealed, not reused for G7); §7's freeze-before-unsealing rule kept. |
| `docs/HYPOTHESIS.md` | New exploratory hypothesis **H1-OR** (oriented local response) superseding the unfrozen H1b candidate of PREREG §7; H1a stays "untested" and H1w "inconclusive" verbatim; the access-claim sentence of the verdict; H2 parked with the 242/99 ceiling; H5 critic contract of section 2. |
| `docs/ROADMAP.md` | Append gates G0–G8 under a dated heading; the 6–19 Nov table stays; G7 conditionally costed, G8 unpriced. |
| `docs/RESULTS.md` | H1w wording "FAIL" → "inconclusive (power below gate)"; the BRIEF's "H1w FAIL" is not quoted again. |
| `docs/DECISIONS.md` | D9: user's 7 Nov GPU approval and Lambda cap change; effective bf16 as the critic input; chain-replay exactness classes and the matrix-error metric. |
| `docs/H5_DESIGN.md`, `docs/H5_PROTOCOL.md` | Critic contract, G5 arms, G8 arms, external adapter (because `validate_production` blocks live providers in `src/wsp/editor/driver.py`). |
| `docs/MODEL_CANDIDATES.md` | Board C's module plan and ranked alternatives (`docs/reviews/council/board-c-synthesis-2026-11-07.md`, `board-c-synthesis-v2-2026-11-07.md`). |
| `paper/refs/verified.yaml` | Blog URL excerpt for Experiments 7–8 before any paper text cites them; `scripts/verify_refs.py` must exit 0. |
| `paper/` | Paper 1 (measurement and negative-result paper) reports the follow-up programme as designed and as run, whatever the outcome. Parts A and B appear as an exploratory section with the A5 disclosure; a failed Part A is reported with its intervals. A second paper is justified only if G7 runs and passes its own registration. Paper 1 is not edited until G3/G4 report, but its limitations paragraph already discloses that the follow-up was designed after the H1w and MMLU results were seen. The split is by evidence level, never by sign. |

## 6. Implementation plan and schema (Board C, adopted; filled in by A5)

New modules: `src/wsp/features/response_field.py` (streamed objective backprop at the pinned checkpoint, fp32 K-FAC factors,
coverage and unsupported-module policy, versioned identities), `response_chart.py` (seeded block projectors; predecessor,
displacement and increment contractions), `rotation.py` (section 4.1 construction and audits), `src/wsp/eval/response_intervention.py`
(variants, manifests, TF-v1 reuse through the normal runner, matched references, costs), `src/wsp/editor/critic.py` plus a
driver integration point before registration (trusted adapter over validated deltas and pinned weights; immutable hashed
effective-delta snapshots; panel and predecessor provenance; an all-attempt ledger that logs invalid attempts without a delta;
cost meter; accept / abstain contract; separate probe hooks and cache; no rationale). `extract.py` gains opt-in versioned
per-tensor tokens with the realised-increment convention (successor minus predecessor, both stored bf16 states, subtracted in
fp32; never a cumulative delta re-applied), with pinned state identities and lineage. `predictor/models/__init__.py` gains the
masked typed set encoder and the zero-at-zero head. CPU tests for rounding, chain state, contractions, rotation audits,
provenance and editor decisions.

Schema fields A5 must fix before G2: objective list with sign and normalisation (NLL versus gold log-prob) and token/window
weighting; Fisher estimator and sampling objective; factor dimensions, damping, eigen-truncation and omitted-tail handling;
module coverage with full-attention and GatedDeltaNet roles kept distinct and the missing-hook action (gate failure); fp32 or
float64 spectral algebra; base versus exact-predecessor computation and the cross-term audit (section 3, G2); panel, cache and
projector hashes; chart projector dimension, seed, distribution, normalisation, masks, non-matrix policy and epsilon; the
deterministic per-tensor transformation into augmented-StatNN columns (exact layer and role order, padding, masks, pooling,
predecessor/displacement and signed field inputs); tree settings, folds, seeds and tuning budget shared by candidate and
comparator; readout fits (zero-shot linear sign; ≤ 3-coefficient calibration); references for G3 and their reuse or cost.

## 7. Risks

Taylor validity radius (the ladder measures it; a small-strength pass with a large-strength fail is a pattern to investigate).
Reuse of examined D3 outcomes in Part B (exploratory only). Part B power below 0.80 on 324–386 rows, so a no-go may be an
underpowered null. Unmeasured field, cache and rotation costs (hard caps, stop and reprice). Rotation controls that leak spectrum
or support mismatch after bf16 rounding (audited, excluded by a frozen rule). K-FAC coverage gaps on fused or recurrent modules
(gate failure, never silent zero factors). The access claim differing from the project goal. MMLU never qualifying at this item
budget (then the project reports sensitivity, not task readability). Provider cascade (fallbacks per `docs/ORCHESTRATOR.md`).
