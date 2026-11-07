# Council decision — 7 November 2026 (chair synthesis)

Branch `council/2026-11-07`. Inputs, read in order: [BRIEF](BRIEF.md); round 1 [world-model](round1-world-model.md),
[representation](round1-representation.md), [metrology](round1-metrology.md), [editor](round1-editor.md); round 2
[world-model](round2-world-model.md), [representation](round2-representation.md), [metrology](round2-metrology.md),
[editor](round2-editor.md); the 5 Nov [SYNTHESIS](../2026-11-05/SYNTHESIS.md). This is a decision proposal for Boards A, B
and C. It launches nothing. Every count and threshold below is prospective unless a file is cited.

**Verdict.** The negative and neutral results of 4–6 November tested the *pooled-statistics representation* and the
*measurement tier*, not the existence of a learnable weight-space response. All four members agree on this and on the
missing ingredient: the pooled features are nearly sign-blind and never look at the base model, so they cannot say in
which direction an edit moves a score at matched norm, family and support. The council proposes one structure, the
**oriented local response** of the base (or predecessor) model to an effective bf16 edit, in two complementary forms
that are tested side by side, and one decisive purchase before any larger spend: a **rotation-and-ladder intervention**
on existing edits (≈2 GPU-h) plus a **retrospective on the regenerated TF-v1 panel**. A sealed confirmatory panel is
bought only if both pass and a simulation shows ≥80 % power. For models editing models, the critic consumes the
effective delta, exact support and cached writer inputs, never the editor's rationale, and H5 is decided only by
realised evaluation savings in a prospective trial. H2 remains parked: regeneration restores eligibility, not power.

## 1. What the council agrees on

1. **Which negatives are structure tests.** Of the seven results in the brief: result 1 (trees ≥ neural, descriptor-only
   within 0.07) is a conditional negative about the *tested heads and features* on 297 rows, not about weight
   readability (`docs/reviews/P4/d3v2r-summary-2026-11-06.md`). Result 2 (family split) is an estimand problem:
   between-family calibration offsets dominate the pooled ρ (`docs/RESULTS.md`, P4 section). Result 3 (H2) was never
   tested: 18 rows / 10 clusters at length ≥ 4 and a nonzero length-one placebo
   (`docs/council/2026-11-05/round2_metrology_output.txt`). Result 4 splits: H1a is *untested* (MMLU gold log-prob
   unqualified, max item half-width 0.1137 > 0.10, `docs/reviews/P5/tf-v1-requalification-2026-11-06.md`); H1w is
   *inconclusive* (θ̂ 0.0071, power lower bounds 0.728 conditional / 0.600 joint < 0.80,
   `docs/reviews/P5/h1w-gate-2026-11-06.md`) and is the only negative that bears on the pooled representation itself.
   Results 5 and 6 are measurement facts and a weak structure clue. Result 7: the blog never demonstrated a held-out
   Δ-only predictor; Experiments 7 and 8 used attention or behavioural feedback (fetched by two members independently,
   https://rikiyat.github.io/blog/weight-space-programming/).
2. **What the delta carries that pooling drops.** Sign and orientation relative to what the base is sensitive to;
   curvature-weighted magnitude; cross-layer coherence of a shared direction; exact layer and spectral tail;
   incremental versus cumulative chain state (`round1-representation.md` §1, with `src/wsp/features/stats.py`,
   `pool.py`, `svd.py::_proj`, `src/wsp/features/extract.py` docstring). M1's direction tokens live in a random
   projection frame and had to learn relevance from ≤ 345 labels; the 5 Nov council already conceded that pooling
   alone does not explain the neural failure (`../2026-11-05/round2-features.md`). The refinement is *anchoring*.
3. **One structure, two anchors.** (a) The **base-anchored sensitivity field** (representation): label-free, computed
   once on the base, per-objective gradients on panels disjoint from every scored item plus K-FAC factors; per-tensor
   tokens ⟨g_k, ΔW_t⟩, a Fisher magnitude term, norms, layer, role, on Δ_eff = bf16(W+Δ) − W. (b) The **oriented local
   response chart** (world-model): actual predecessor W, displacement W − W₀ and incremental D consumed separately,
   signed coordinate-block contractions ⟨D, P_out W P_in⟩, exact layer/support, a small within-layer set encoder with
   a zero-at-zero-edit response head. Both members agree these are the same object at different orders (the field is
   the Taylor expansion of the chart's response map; `round2-representation.md` §4, `round2-world-model.md` §4). The
   one real dispute, loss-anchoring versus weight-anchoring, is **measured, not adjudicated**: both run on identical
   rows and folds in the retrospective (§3, Part B).
4. **Corrections accepted by the proposers.** The Fisher quadratic is nonnegative for covariance factors and is not a
   signed Hessian correction for every endpoint; it is a trust-region / magnitude feature, and signed response is
   learned per objective (world-model and editor objections, `round2-world-model.md` §2, `round2-editor.md`; accepted
   here). "Label-free" means no edited-model outcome labels; gold-task gradients still use task supervision. The chain
   cross term is a stateless feature and belongs in the identical-current-state control, not in an H2 claim.
5. **Endpoint discipline.** WikiText log-PPL on 64 disjoint 1,024-token windows is the only qualified TF-v1 label
   (max half-width 0.01282 vs δ 0.04879, reliability lower bound 0.9923, `docs/reviews/P5/h1w-gate-2026-11-06.md`);
   it measures distribution sensitivity, not task readability. MMLU-Pro gold-letter log-prob on 1,000 disjoint items
   is the task endpoint and is **blocked** until item, tripwire, repeat and reliability gates pass
   (`docs/PREREG_H1_TF.md` §§1–2, A4.1). Every structure test uses the same frozen window/item ids, the actual
   predecessor, the actual effective delta, and the mandatory baselines StatNN, descriptor-only, delta-norm and
   augmented StatNN on identical inputs (a tree-only gain is representation value; only a gain over augmented StatNN
   is encoder value, `round2-world-model.md` §3). Gradients and probe caches are computed on windows/items disjoint
   from every scored outcome.
6. **Models editing models.** The critic reads the validated effective delta, exact changed coordinates, rounding
   loss, pinned predecessor, structured operation metadata and, for MLP writers, cached post-gating inputs on a
   frozen scoring-disjoint probe panel; it adds the field contractions and the chart contractions; it predicts item
   score changes, aggregate task gain and collateral loss, with episode-clustered calibrated abstention. **Rationale:
   no** (transcripts preserved for audit). Every proposal, including failed and rejected ones, stays in the
   denominator. H5 is decided only by a prospective trial: ≥ 30 % fewer real evaluations at equal final score
   (paired lower bound ≥ 30 %, noninferiority margin −1 point), against the same editor without critic, random
   ranking, a same-support gradient search, single-draw best-of-N at equal evaluation budget and Adam/Muon
   (`docs/HYPOTHESIS.md` H5, `docs/H5_DESIGN.md`, `round2-editor.md`). Note from the editor member: `driver.run_episode`
   calls `validate_production` for every provider except `ScriptedEditor`, so a live-editor pilot cannot use the
   offline path; caches and fields need an external critic adapter (`src/wsp/editor/driver.py`).
7. **H2 after regeneration.** The 358 regenerated snapshots (`feat/d8-chain-replay`) restore eligibility, not labels
   or independence. Ceiling after evaluating the 296 never-evaluated chain entries: 242 rows in 99 lineages of length
   ≥ 4, against the 162-lineage estimate for 80 % power at a 15 % effect
   (`docs/council/2026-11-05/round2_metrology_output.txt` §A; `round2-metrology.md` §H2). H2 stays parked as a
   confirmatory hypothesis; closure evaluation is funded as census and as the fingerprint of the full zoo (§3, G0/G6).

## 2. Adjudicated disagreements

| Disagreement | Chair decision | Reverses it |
|---|---|---|
| WikiText-first (representation) vs MMLU-only (metrology r1, world-model r1) | **WikiText first, MMLU descriptive alongside.** A structure that cannot read the qualified sensitivity label should die before MMLU qualification is bought; passing WikiText alone is sensitivity, not task readability, and is reported as such. | MMLU requalifies before Part A runs: then MMLU becomes co-primary with its own power simulation. |
| Buy 720 sealed rows now (world-model r1) vs intervention on existing edits first (representation, metrology) | **Intervention first.** The world-model's own arithmetic needs ≈ 262 clusters for a 15 % feature-contrast effect (`../2026-11-05/SYNTHESIS.md` §2); 100 blocks is a floor, not power. Sealed rows are sized by simulation after Part A/B. | Part A and B pass and the simulation shows ≥ 80 % joint and conditional power at ≤ 300 blocks. |
| Loss-anchored field vs weight-anchored chart | **Not adjudicated; measured.** Both arms, plus norm-matched random field, layer-permuted field and the TF-v1 KL vector (A1-lite), on identical rows and folds in Part B. | One arm beats the other and both controls with a paired lower bound > 0. |
| Editor critic sufficient for the weight-only question (editor r1) | **No.** Cached `D x` is a writer-local response; it is an activation-assisted comparator with probe-only and probe-plus-descriptor controls and measured probe cost, not a weight-only result. | A field-only or chart-only arm matches it on identical rows. |
| Regenerated chains make H2 testable | **No.** 242/99 < 162 lineages; eight or forty-eight training runs cannot resolve a 15 % history effect. Exploratory within-M2 ablation with an identical-current-state control is allowed once closure is evaluated; no confirmatory claim. | Independent new lineages are bought to ≥ 162 with measured paired loss variance. |
| Rationale as critic input | **Excluded.** Unanimous. A blinded rationale ablation may follow a positive delta-only result, never precede it. | A delta-only critic passes and a pre-registered blinded ablation shows a further gain. |

## 3. Programme and gates (local RTX PRO 4500 unless stated; GPU time approved by the user on 7 Nov 2026)

| Gate | What | Pass / stop | GPU-h (measured unit) |
|---|---|---|---|
| **G0** chain regeneration (running) | 358 snapshots replayed from edit.json; install; features; full-set P4 rerun (345 rows) as the X1 fingerprint; TF-v1 G1 for the two lost predecessors and the 14 missing (y, h) rows | random_dir/abliteration bitwise; LoRA within bf16 rounding (pilot `docs/reviews/P5/d8-regeneration-2026-11-06/chain_replay_pilot.json`) | ≈ 3 (LoRA re-training) + CPU 5.4 (54 s/entry) + 26.6 run-wall-h CPU |
| **G1** P6d rescoring | TF-v1 D3 stratum panel (229 + 157 controls once chains are back; 167 + 157 without) at the frozen source | records bitwise with the surviving fingerprints (G1/G2 rule) | 5.5 (priced in `docs/reviews/P5/board-a-incident-d8-2026-11-06.md`) |
| **G2** fields and chart features | gradients g_k on disjoint dev panels, K-FAC factors, contraction tokens; chart contractions; exact-gradient audit at 20 predecessors | timed; cap, stop and reprice if exceeded | ≤ 2 (cap) + ≤ 1 audit |
| **G3 Part A** rotation and ladder | 30 lora_sft + 10 chain D3 edits chosen by sorted id within norm tertiles before outcomes; per edit one support-preserving spectrum-matched rotation R at s = 1 and D at s ∈ {0.25, 0.5}; 120 TF-v1 evaluations | pass: sign agreement of ⟨g, D − R⟩ with the measured WikiText difference on resolved pairs (|y(D) − y(R)| > item half-width), lower 95 % binomial bound > 0.5, ≥ 15 resolved pairs; ladder sign agreement ≥ 0.7 at s = 0.25. Kill: ladder < 0.7, or ≥ 15 pairs with upper bound ≤ 0.6 | 2.0 (1.0 min/entry warm) |
| **G4 Part B** retrospective | registered H1w folds, nuisance and six intervals on the G1 panel; arms: StatNN, descriptor-only, delta-norm, field (zero-shot, ≤ 3-coefficient calibration, StatNN+field), augmented StatNN on chart features, chart, random field, layer-permuted field, A1-lite | pass to buy sealed rows: θ(StatNN+field or chart) − θ(StatNN) ≥ 0.15 with paired lower 95 % > 0, beats both destruction controls, **and** G3 passed. ≥ 5 % residual-MAE gain with lower bound ≤ 0 is a screen (funds a Part A extension only). Kill: gain upper 95 % < 0.05, or ties the random field | CPU |
| **G5** editor pilot | `ScriptedEditor` only, production disabled; 10 source/support blocks × 6 arms (D, same-support gradient step, orientation randomisation, −D, D/2, D/4) = 60 proposals; blocks split 6/2/2; critic arms full / field-only / chart-only / StatNN / augmented StatNN / descriptors / norm / activation-only / correspondence-shuffled | feasibility and calibration variance only; no rejection, no H5 claim; stop at any exhausted allocation | ≤ 10 hard cap (1.0 scoring, 0.9 extraction, 2 fields/cache, 2 generation/probes, 2 references, 2.1 contingency) |
| **G6** chain closure | TF-v1 scoring of the 296 never-evaluated chain entries; features | census for H2 (242 rows / 99 lineages ceiling); full-zoo fingerprint | ≈ 4.9 (+ 4.4 CPU core-h) |
| **G7** sealed purchase (conditional) | fresh source-closed component blocks with edit + orientation control at strengths {−1, ½, 1}; model frozen before outcomes; MMLU co-primary only if requalified | sized by simulation to ≥ 80 % joint and conditional power; floor 100 blocks, cap 300 (1,920 rows, ≈ 33 GPU-h); stop unresolved if the cap is insufficient | 12.5–33 |
| **G8** H5 prospective trial (conditional on G5 and a positive G3/G4) | fresh live-editor episodes, critic frozen; arms in §1.6 | ≥ 30 % realised savings, noninferiority, rejection bound ≤ 5 %, net GPU saving | priced from G5 timing; no fixed price yet |

Committed now (G0–G6): ≈ 28 local GPU-h plus ≈ 40 CPU core-h, inside the user's approval. Lambda is used only to
parallelise G6 or G7 if wall-clock matters (H100 ≈ 3.1 min/entry effective at the fast tier; TF-v1 cost on an H100 is
unmeasured and must be timed before being booked). G7 and G8 need Board A acceptance of their registrations and the
measured power before any spend.

## 4. What this changes in the record

- `docs/HYPOTHESIS.md`: add a dated amendment "H1-OR (oriented local response)": the registered estimand of H1a/H1w
  with the field/chart inputs, WikiText-first, MMLU descriptive until requalified; H2 remains parked with the 242/99
  ceiling recorded; H5 gets the critic contract of §1.6.
- `docs/ROADMAP.md`: replace the 6–19 Nov table with gates G0–G8.
- `docs/PREREG_H1_TF.md`: appendix A5 registering Part A (edit selection rule, rotation construction, strengths,
  resolved-pair rule, binomial bound), Part B arms and the destruction controls, frozen before G3 runs.
- Paper: no change to the measurement paper until G3/G4 report; a positive result becomes a second paper, a negative
  one a section of the first.

## 5. Risks the council names

Taylor validity radius (edits may be too large for a first-order field; the ladder measures it, and a pass at s = 0.25
with a fail at s ≥ 1 sends the programme to the probe-response alternative A1); unmeasured field and probe costs
(hard caps, stop and reprice); rotation construction leaking spectrum mismatch into the controls (residual v4
differences are logged and reported); MMLU never qualifying at this item budget (then the project reports sensitivity,
not task readability); and the provider cascade (every delegate falls back per `docs/ORCHESTRATOR.md`).
