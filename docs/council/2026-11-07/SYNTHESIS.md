# Council decision — 7 November 2026 (chair synthesis, version 2)

Branch `council/2026-11-07`. Version 1 was reviewed by Boards A, B and C (`docs/reviews/council/board-{a,b,c}-synthesis-2026-11-07.md`,
all three REVISE); this version answers every required edit. Inputs, read in order: [BRIEF](BRIEF.md); round 1
[world-model](round1-world-model.md), [representation](round1-representation.md), [metrology](round1-metrology.md),
[editor](round1-editor.md); round 2 [world-model](round2-world-model.md), [representation](round2-representation.md),
[metrology](round2-metrology.md), [editor](round2-editor.md); the 5 Nov [SYNTHESIS](../2026-11-05/SYNTHESIS.md).

This is a decision proposal. It launches nothing. Every threshold in sections 3 and 4 (0.15, 0.05, 0.5, 0.6, 0.7, 15 pairs,
≥ 80 % power) is a chair choice made before any new outcome exists, not a measurement. Every number with a file next to it is
a quotation from that file. Terms used below: *StatNN* is a gradient-boosted tree on pooled weight statistics; *descriptor-only*
is the same tree on the edit descriptor (family, norm, support) alone; *TF-v1* is the teacher-forced scoring tier (no sampling);
*bf16* is the 16-bit storage format of the weights; the *effective delta* Δ_eff = bf16(W + Δ) − W is the change that is actually
stored; the *support* of an edit is the set of tensors and coordinate blocks it changes; *θ* is the relative residual-error
reduction of `docs/PREREG_H1_TF.md` §4; a *half-width* is half the width of a 95 % interval; the *tripwire* is the
PREREG item-precision stop rule; a *census* is a count of what exists, not a measurement of an effect.

## Verdict

The negative and neutral results of 4–6 November do not show that no learnable response to a weight edit exists. They tested
pooled statistics, descriptors and neural encoders whose direction tokens sit in a random projection frame, on one base model
and 297 rows, with a measurement tier that could not resolve the task endpoint. The four members agree that the next step is a
controlled intervention, not another encoder. The representation member argues that the missing ingredient is the sign of an
edit's effect relative to the base model, which pooled statistics barely encode; the other three treat this as a mechanism worth
testing, not an established cause.

Both proposed structures read more than the edit. The *sensitivity field* uses gradients of the base model on labelled panels;
the *local response chart* uses the predecessor's weights. A success will be reported as "weights plus base sensitivity" or
"weights plus predecessor", never as weight reading alone. The project's original claim, prediction from the weights and the
edit alone, stays untested by this programme.

The decision experiment is small: a rotation-and-ladder intervention on 40 existing edits (Part A, about 2 GPU-hours) and an
exploratory retrospective on the regenerated teacher-forced panel (Part B, CPU). Together they decide whether sealed rows are
bought. The rest of the spend (regeneration, rescoring, the editor harness pilot, chain closure) is infrastructure and census,
approved by the user on 7 November, and is not conditioned on Part A.

## 1. What each negative tested

| Result (brief) | What it tested | Verdict | Strength | File |
|---|---|---|---|---|
| 1 trees ≥ neural; descriptor-only within 0.07 | a pooled-statistics tree, descriptors, and encoders on randomly framed directions; 297 rows, one base | family and norm explain most of the rank; no tested representation shows weight-specific signal beyond descriptors | conditional; does not test base- or predecessor-anchored directions | `docs/reviews/P4/d3v2r-summary-2026-11-06.md` |
| 2 held-out family | pooled rank across families | confounded estimate: between-family calibration offsets dominate the pooled ρ; an offset-free family test has not been run | not a structure test | `docs/RESULTS.md` (P4 section) |
| 3 history (H2) | M2 vs stateless M2 | never tested: 18 rows / 10 clusters at length ≥ 4; length-one placebo 0.1265 [0.039, 0.206] | sound | `docs/council/2026-11-05/round2_metrology_output.txt` §E |
| 4a H1a (MMLU gold log-prob) | residual readability | untested: endpoint unqualified | sound | `docs/reviews/P5/tf-v1-requalification-2026-11-06.md` |
| 4b H1w (WikiText) | residual sensitivity of pooled statistics | inconclusive in both directions: θ̂ near zero, all six intervals include zero, power below the gate | bears on one contrast for one endpoint | `docs/reviews/P5/h1w-gate-2026-11-06.md` |
| 5 measurement facts | sampled benchmarks | most zoo edits are behaviourally null, so the labelled signal lives in a few families | measurement | `docs/RESULTS.md`, `docs/council/2026-11-05/metrology_budget_output.txt` |
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

The headline "the negatives tested the representation, not structure" is the hypothesis Parts A and B test. The evidence
supports "does not exclude a structure", no more.

## 2. The two structures, and where the members disagree

**Sensitivity field (representation).** Computed once on the base, before any outcome label is seen. For each objective k
(WikiText NLL, MMLU gold-letter log-prob, HellaSwag, refusal margin) a gradient g_k on a panel disjoint by text group from every
scored item, plus K-FAC factors (A_t for inputs, G_t for output gradients) per supported linear module. Each edited tensor t
becomes a token: the signed contraction ⟨g_k, ΔW_t⟩, the magnitude term q_t = ½ tr(G_t ΔW_t A_t ΔW_tᵀ), the Frobenius norm, the
layer index and the role. "Label-free" means no edited-model outcome labels; the gold-task gradients still use task supervision.

**Oriented local response chart (world-model).** Consumes the actual predecessor W, the displacement W − W₀ and the incremental
effective edit D separately, plus signed coordinate-block contractions ⟨D, P_out W P_in⟩ with frozen projectors, exact layer,
support, spectral tail and rounding loss, through a small within-layer set encoder with a response head constrained to zero at
zero edit.

**Where they stand.** The representation member holds that the field is the first-order expansion of the chart's response map.
The world-model member calls them complementary, keeps the weight-contraction chart as its primary test, and requires a
field-only comparator before field tokens are fed to the chart. Board C adds that a weight contraction is not the loss
Jacobian and a Fisher factor is not the Hessian, so the two are *competing representations* until measured as such. The chair
adopts Board C's framing: both run on identical rows and folds in Part B, and neither is assumed to contain the other.

**Corrections the chair adopts** (raised by the world-model and editor members; the representation member's round 2 does not
address them): q_t is nonnegative for covariance factors and is not a signed curvature correction for every endpoint, so the
sign of a zero-shot prediction comes from ⟨g_k, ΔW⟩ alone and q_t enters only as a trust-region feature; the chain cross term
is a stateless feature of the current displacement and belongs in the identical-current-state control, never in an H2 claim.

**Models editing models (editor).** The critic consumes the validated effective delta, the exact changed coordinates, rounding
loss, the pinned predecessor, structured operation metadata and, for MLP writers, cached post-gating inputs on a frozen
scoring-disjoint probe panel; it may add field and chart tokens as separately named arms. It predicts item score changes,
aggregate task gain and collateral loss, with episode-clustered calibrated abstention. It never reads the editor's rationale;
transcripts are kept for audit. Every proposal, including failed and rejected ones, stays in the denominator.

### Adjudicated disagreements

| Disagreement | Positions | Chair decision | Reverses it |
|---|---|---|---|
| Endpoint | WikiText first (representation; metrology r2, on a fresh sealed panel) vs MMLU decisive (world-model r2; metrology r1) | WikiText first as the mechanism endpoint; MMLU co-registered but descriptive until it qualifies. A WikiText-only pass is sensitivity, not task readability. | MMLU requalifies before Part A runs; then one co-primary rule is registered in A5 before unsealing. |
| Buy sealed rows now | 20 + 100 blocks as a floor (world-model) vs intervention first (representation, metrology) | Intervention first. The world-model's own arithmetic needs ≈ 262 clusters for a 15 % feature-contrast effect (`../2026-11-05/SYNTHESIS.md` §2); 100 blocks is a floor, not power. | Part A and B go, and a development-only simulation shows ≥ 0.80 joint and conditional power at ≤ 300 blocks. |
| Loss- vs weight-anchoring | field vs chart | Not adjudicated. Both arms, both destruction controls and the probe-response comparator on identical rows and folds (Part B). | One arm beats the other and both controls with a paired lower bound > 0 on the registered contrast. |
| Is the editor critic a weight-only test | editor r1 | No. Cached D·x is a writer-local response; it is an activation-assisted comparator with probe-only and probe-plus-descriptor controls and measured probe cost. | A field-only or chart-only arm matches it on identical rows. |
| Do the regenerated chains make H2 testable | — | No. The ceiling is 242 rows in 99 lineages against a historical stand-in of 162 lineages (section 3, G6). H2 stays parked; closure is funded as census. Any history ablation needs its own amendment and is not authorised here. | Independent new lineages with measured paired TF-v1 loss variance and ≥ 0.80 power, plus a length-one equivalence test. |
| Rationale as critic input | — | Excluded (unanimous). A blinded rationale ablation may follow a positive delta-only result, never precede it. | A delta-only critic passes and a pre-registered blinded ablation shows a further gain. |
| Metrology's power verdict | metrology r2: none of the three decisive tests is powered as written; D3 outcomes have already been examined | Accepted. Part B is exploratory development evidence; Part A is sized by simulation before it is called decisive. | A development-only simulation shows ≥ 0.80 power for the registered Part A contrast. |

## 3. Programme and gates

Hardware: the local RTX PRO 4500 (device H0). TF-v1 scoring is qualified on H0 only; an H100 may run fast-tier census work but
TF-v1 on any other device needs its own qualification first. GPU-hour figures marked *allocation* are caps with a stop-and-reprice
rule, not measurements; the measured units are 1.0–1.04 min per TF-v1 entry warm on H0, 54.3 CPU-s per feature extraction,
7.5 min per fast-tier entry on H0 (`docs/reviews/P5/d8-regeneration-2026-11-06.md` §§2, 5; BRIEF).

| Gate | What | Decision rule | Cost |
|---|---|---|---|
| **G0** chain regeneration (running) | 358 snapshots replayed from edit.json; install; features; full-set P4 rerun (target 345 matched rows) as the X1 fingerprint; TF-v1 G1 for the 2 lost predecessors and the 14 missing (y, h) rows | random_dir and abliteration steps must be bitwise; a re-trained LoRA step is accepted within a relative Frobenius tolerance of 5e-3 and labelled "re-trained"; descendants of a re-trained step are labelled likewise and excluded from fingerprint claims. Pilot evidence: `docs/reviews/P5/d8-regeneration-2026-11-06/chain_replay_pilot.json` on branch `feat/d8-chain-replay` (not yet merged into this branch): 8 of 11 steps bitwise, LoRA steps within 1e-3 relative Frobenius | LoRA re-training *allocation* ≈ 3 GPU-h (unmeasured at scale); extraction 358 × 54.3 s = 5.4 CPU-h; P4 rerun 26.6 run-wall-h (historical sum of run wall-clock, up to 4 workers × 2 threads; core-hours not established) |
| **G1** P6d rescoring | TF-v1 D3 stratum panel at the frozen source: 316 recoverable scoring entries (167 stratum rows + 157 controls, less aliases) priced at ≈ 5.5 GPU-h in `docs/reviews/P5/board-a-incident-d8-2026-11-06.md`; the 62 chain rows that depend on G0 add ≈ 1.05 GPU-h; canonical deltas, aliases, predecessors and references reconciled before launch | records bitwise with the surviving fingerprints (G1/G2 rule of the incident review) | 5.5 + 1.05 |
| **G2** fields, chart features, probe cache | g_k and K-FAC factors (schema in section 4); chart projectors; probe-cache for the comparator; exact-gradient audit at 20 chain predecessors with a frozen tolerance (relative error of ⟨g, D⟩ against the base-plus-cross-term approximation ≤ 0.1 on ≥ 16 of 20, else the base approximation is not used for chains); peak VRAM, throughput and bytes recorded | hard caps; exceeding one stops the gate and triggers repricing, never a silent cut in coverage | *allocation* ≤ 2 GPU-h fields and cache, ≤ 1 GPU-h audit |
| **G3 Part A** rotation and ladder | 30 lora_sft and 10 chain D3 edits, one per certified source component, chosen by sorted id within effective-norm tertiles before any new variant outcome; per edit one support-preserving spectrum-matched rotation R at s = 1 and the edit itself at s ∈ {0.25, 0.5}, each applied at the actual predecessor and rounded; 120 new TF-v1 WikiText scores on the registered 64 windows | section 4.1 | 120 × 1.04 min ≈ 2.1 GPU-h plus references |
| **G4 Part B** retrospective (exploratory) | registered H1w folds, nuisance and intervals on the G1 panel (229 + 157 rows if G0 lands, else 167 + 157) | section 4.2; go / no-go for buying sealed rows; no kill (PREREG A4.4) | CPU |
| **G5** editor harness pilot | `ScriptedEditor` only, production disabled; 10 source/support blocks × 6 arms (D, same-support gradient step, orientation randomisation, −D, D/2, D/4) = 60 proposals; whole connected episode/source/direction/prompt groups split 6/2/2; arms: full critic, field-only, chart-only, StatNN, augmented StatNN, descriptors, norm, activation-only, correspondence-shuffled | feasibility and calibration variance only; no rejection, no H5 claim; an exhausted allocation is a feasibility failure, not evidence about prediction | *allocation* ≤ 10 GPU-h: 1.0 scoring, 0.9 reserve for extraction (CPU in practice), 2 fields/cache, 2 generation/probes, 2 references/repeats, 2.1 contingency |
| **G6** chain closure (census) | TF-v1 scoring of the 296 never-evaluated chain entries (358 lost − 62 already evaluated); features | census only. Ceiling after closure: 242 depth ≥ 4 rows in 99 roots, up from 36 rows in 16 roots (`round2_metrology_output.txt` §A). The 162-lineage figure is the §E stand-in from 18 rows / 10 clusters on the old endpoint, not a certified TF-v1 requirement. Every row scored here is *exposed* and excluded from G7 freshness | 296 × 1.0 min ≈ 4.9 GPU-h; 296 × 54.3 s ≈ 4.5 CPU-h |
| **G7** sealed purchase (conditional on G3 and G4 go) | fresh source-closed component blocks, each with the edit and its orientation control at strengths {−1, ½, 1}; new sealed WikiText window panel disjoint from the 64 development windows; one frozen predictor, all fields, caches, projectors, controls and decision rules hashed before any sealed outcome | section 4.3 | scoring only: 20 development blocks (120 rows) plus 100–300 sealed blocks (600–1,800 rows) at 1.04 min per row = 12.5–33.3 GPU-h; overheads *allocation* 2 GPU-h |
| **G8** H5 prospective trial (conditional on G5 and a G3/G4 go) | fresh live-editor episodes with the critic and rejection policy frozen | section 4.4 | priced from G5 timing; no fixed price exists yet |

Sum of the allocations G0–G6: ≈ 28 GPU-h on H0 (3 + 6.55 + 3 + 2.1 + 10 + 4.9 = 29.6 before the 0.9 CPU reserve is removed),
inside the user's approval. CPU work is listed separately above and is not summed into a core-hour total because the P4 figure is
a run-wall sum.

## 4. Registered decision rules (to be frozen as PREREG appendix A5 before G3 runs)

### 4.1 Part A

*Null and comparator.* The null is that the sign of ⟨g, D − R⟩ agrees with the sign of the measured WikiText difference
y(W + D) − y(W + R) with probability ½. The constant-sign comparator ("every trained edit degrades more than its rotation") is
reported alongside, with the same interval.

*Unit and interval.* One pair per certified source component (40 components). Two-sided 95 % Clopper–Pearson interval on the
agreement count among resolved pairs. Lower bound > 0.5 requires at least 12 of 15 or 27 of 40 agreements (exact binomial tails,
computed by Board A). Ties (zero prediction) and missing scores count as disagreements.

*Resolution.* d_i is the same-window difference; its half-width h_pair,i comes from a paired block bootstrap over the 32
two-window blocks with common draws across the edit, its rotation and the predecessor reference, never from h_D, h_R or their
maximum. A pair is resolved if |d_i| > h_pair,i. The agreement estimand is conditional on resolution; all 40 pairs are reported,
and the null size with the resolution step included is validated on a development-only simulation before outcomes are read.

*Yield.* Fewer than 15 resolved pairs is **unresolved**: no replacement, no resampling. Board A's illustration: at an
independent resolution probability of 0.375 the chance of ≥ 15 resolved among 40 is ≈ 0.56; the yield and the agreement power are
simulated from development contrast and noise assumptions before this purchase is called decisive.

*Ladder.* Sign agreement between the predicted and measured direction at s = 0.25 and s = 0.5, same resolution rule and
Clopper–Pearson interval, reported as descriptive; a lower bound < 0.5 at s = 0.25 on ≥ 15 resolved pairs sends the programme to
the probe-response alternative (section 4.2) and stops field purchases, but kills nothing.

*Rotation.* Per edited tensor t with incremental effective edit D_t = bf16(W_t + Δ_t) − W_t at the actual predecessor: take the
rectangular coordinate block of changed rows and columns, compute the thin SVD D_t|block = U Σ Vᵀ in fp32 (from the raw LoRA
factors where they exist, from the realised matrix otherwise), draw Q_out and Q_in as Haar-orthogonal matrices on that block
from a registered seed, form R_t = Q_out U Σ Vᵀ Q_inᵀ, apply W_t + R_t and round to bf16. Preserved before rounding: singular
values, Frobenius norm, rank, confinement to the block, tensor names and shapes. Not preserved, and logged: coordinate moments,
sign fractions, cosine to the predecessor, cross-layer coherence of a shared direction, directional tokens, and anything after
rounding. Post-rounding audit: relative Frobenius mismatch ≤ 2e-2 and support identical at block level, else the pair is
excluded by a rule fixed before outcomes and the exclusion is reported. Non-2D and non-matrix tensors are excluded from
rotation and listed. Scaled edits are built by applying s·Δ_t at W_t and rounding, not by scaling D_t.

*Exposure.* D3 outcomes for the 40 edits are already known; the selection is "before any new variant outcome", and Part A is
exploratory.

### 4.2 Part B

*Panel.* The G1 records, registered H1w folds, train-only nuisance (family, device, log-norm), seeds 0/1/2 averaged before the
loss, outer-held-out predictions audited for nested out-of-fold construction.

*Primary contrast (one).* Relative residual-MAE reduction of **StatNN + field tokens** over **augmented StatNN** (the same tree
on identical inputs: v4 pooled statistics plus the ordered exact-layer scalar columns of the chart, padded by a fixed rule), row-summed,
with a paired ancestry-cluster bootstrap-t interval. This is the chair's primary arm; it is the cheapest arm that isolates the
field's information. All other arms are descriptive: zero-shot field (sign from ⟨g, ΔW⟩ only), ≤ 3-coefficient field
calibration, chart, norm-matched random field, layer-permuted field, A1-lite (TF-v1 KL vector), descriptor-only, delta-norm,
original StatNN, and the probe-response comparator (linear ΔW·x on a fixed scoring-disjoint 64-prompt panel plus a regularised
head; probe-only and probe-plus-descriptor controls; probe cost timed).

*Why 0.15 and 0.05.* 0.15 is the PREREG §4 threshold for θ, fixed on 5 November before the H1w and MMLU estimates were known;
0.05 is the PREREG §7 screen. Both are reused, not chosen after the fact. The MMLU θ̂ of 0.129 (`docs/reviews/P5/h1a-companions-2026-11-06.md`)
was known when this programme was designed; A5 discloses it.

*Decision.* **Go** (buy sealed rows): primary contrast ≥ 0.15 with lower 95 % > 0, and the field arm beats both destruction
controls on the same contrast with lower bound > 0, and Part A is not unresolved and not sent to the alternative. **Screen**
(funds a Part A extension only): ≥ 0.05 with lower bound ≤ 0. **No-go**: upper 95 % < 0.05 **and** the development-only
simulated power at 0.15 is ≥ 0.80; otherwise **unresolved**. There is no kill: Part B reuses examined outcomes and PREREG A4.4
forbids pass/kill language on this endpoint. If the chart arm passes the same contrast and the field arm does not, the chart
becomes the frozen G7 arm; the choice is made once, on Part B, and written into A5 before G7.

*Disclosure.* The D3 WikiText θ̂ and MMLU θ̂ were known before this design; the field and chart code, arm list and thresholds are
frozen by hash before the first field is computed.

### 4.3 G7 sealed purchase

Scientific success keeps the PREREG §§4–5 contract: endpoint qualified on the new sealed panel, item precision, conditional and
joint calibrated intervals, independent component count, outcome-free power lower bound ≥ 0.80, one frozen arm against augmented
StatNN on the registered contrast, and the intact-versus-rotated contrast positive with lower bound > 0. Source graph closure
extends through parents, shared adapters and training runs, task-vector operands and generated direction identities. Any
component exposed by G0, G4 or G6 is development and excluded. No tuning, replacement or arm change after exposure; a
development-only simulation sizes the panel between 100 and 300 blocks; insufficient power at 300 is "unresolved", not a
larger purchase.

### 4.4 G8 H5 trial

Arms: critic-guided search; the same editor without critic; random ranking; same-support gradient search; single-draw best-of-N
at equal real-evaluation budget (one score draw per candidate, no repeated-draw archive); Adam/Muon at equal forward budget.
Pass: realised real-evaluation savings ≥ 30 % with paired 95 % lower bound ≥ 30 %, final-score difference lower bound ≥ −1
percentage point, guardrails met, one-sided 95 % upper bound on missed qualifying proposals among rejected ≤ 5 % with episode
clustering, positive net GPU saving after charging diagnostics, gradients, probes, failed attempts and amortised training.
Episode count and budget are set from G5 variance to ≥ 0.80 simulated joint power before any episode runs. Kill critic
deployment if a powered test gives a savings upper bound < 30 % or non-inferiority fails; kill the structural mechanism if its
powered incremental residual-MAE gain upper bound is < 0.05 against augmented StatNN **and** field-only. Calibration, power or
budget failure leaves H5 unresolved.

## 5. What this changes in the record

| File | Change |
|---|---|
| `docs/HYPOTHESIS.md` | New exploratory hypothesis **H1-OR** (oriented local response) superseding the unfrozen H1b candidate of PREREG §7; H1a stays "untested" and H1w "inconclusive" verbatim; access-claim sentence of the verdict; H2 parked with the 242/99 ceiling; H5 critic contract of section 2. |
| `docs/PREREG_H1_TF.md` | Appendix A5: Part A and Part B rules of section 4, frozen by hash before G3; statement that A4.4 governs Part B; A4.6-style disclosure (D3 WikiText θ̂, MMLU θ̂ 0.129, design written after both); disposition of the unspent sealed partition `ebf0379a15d9…` (kept sealed, not reused for G7); §7's freeze-before-unsealing rule kept. |
| `docs/ROADMAP.md` | Append gates G0–G8 under a dated heading; the 6–19 Nov table stays; G7 and G8 marked conditional and unpriced. |
| `docs/RESULTS.md` | H1w wording "FAIL" → "inconclusive (power below gate)"; the BRIEF's "H1w FAIL" is not quoted again. |
| `docs/DECISIONS.md` | D9: user's 7 Nov GPU approval and Lambda cap change; effective-bf16 as the critic input; chain-replay exactness classes. |
| `docs/H5_DESIGN.md`, `docs/H5_PROTOCOL.md` | Critic contract, G5 arms, G8 arms, external adapter (because `validate_production` blocks live providers in `src/wsp/editor/driver.py`). |
| `docs/MODEL_CANDIDATES.md` | Board C's module plan and ranked alternatives (`docs/reviews/council/board-c-synthesis-2026-11-07.md`). |
| `paper/refs/verified.yaml` | Blog URL excerpt for Experiments 7–8 before any paper text cites them; `scripts/verify_refs.py` must exit 0. |
| `paper/` | Paper 1 (measurement and negative-result paper) reports the follow-up programme as designed and as run, whatever the outcome: Parts A and B appear as an exploratory section with the A5 disclosure; a failed Part A is reported with its intervals. A second paper is justified only if G7 runs and passes its own registration. Paper 1 is not edited until G3/G4 report, but its limitations paragraph already discloses that the follow-up was designed after the H1w and MMLU results were seen. The split is by evidence level, never by sign. |

## 6. Implementation plan (Board C, adopted)

New modules `src/wsp/features/response_field.py` (streamed objective backprop at the pinned checkpoint, fp32 K-FAC factors,
coverage and unsupported-module policy, versioned identities), `response_chart.py` (seeded block projectors, predecessor /
displacement / increment contractions), `rotation.py` (section 4.1 construction and audits), `src/wsp/eval/response_intervention.py`
(variants, manifests, TF-v1 reuse, matched references, costs), `src/wsp/editor/critic.py` plus a driver integration point
before registration (trusted adapter over validated deltas and pinned weights, all-attempt ledger, cost meter, accept / abstain
contract, no rationale); `extract.py` gains opt-in versioned per-tensor tokens with the realised-increment convention (successor
minus predecessor in bf16, never a cumulative delta re-applied); `predictor/models/__init__.py` gains the masked typed set encoder
and zero-at-zero head. CPU tests for rounding, chain state, contractions, rotation audits, provenance and editor decisions.
Field schema to freeze in A5: factor estimator and sampling objective, token weighting, module coverage (full-attention and
GatedDeltaNet roles kept distinct), damping and eigen-truncation, fp32 or float64 spectral algebra, panel and cache hashes,
calibration loss and fold fitting; chart projectors: dimension, seed, distribution, epsilon.

## 7. Risks

Taylor validity radius (the ladder measures it; a pass at s = 0.25 with a fail at s ≥ 1 means the structure is right and the
edits too large); reuse of examined D3 outcomes in Part B (exploratory only); Part B power below 0.80 on 167–229 rows, so a no-go
may be an underpowered null; unmeasured field, cache and rotation costs (hard caps, stop and reprice); rotation controls that leak
spectrum or support mismatch after bf16 rounding (audited, excluded by a frozen rule); K-FAC coverage gaps on fused or recurrent
modules (no silent zero factors); the access claim differing from the project goal; MMLU never qualifying at this item budget (then
the project reports sensitivity, not task readability); and the provider cascade (fallbacks per `docs/ORCHESTRATOR.md`).
