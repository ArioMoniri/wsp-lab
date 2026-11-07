# Round 2 — world-model

Recommendation: retain the oriented local response chart, but establish directional representation value before buying a neural dynamical atlas. This is a design revision, not an experiment. Paths below are repository-relative; counts and budgets explicitly described as planned are proposals, not observations.

## 1. Concessions

- **Representation, `round1-representation.md` §1, “Why M1 did not fix it”:** base anchoring is a more specific intervention than adding another transformer. I now require a same-input tree and a capacity-matched random-frame comparison before attributing improvement to the encoder. Existing random projections and layer/type embeddings are implemented in `src/wsp/features/svd.py::_proj` and `src/wsp/predictor/models/__init__.py::TensorEncoder`.
- **Metrology, `round1-metrology.md` §2, “Protocol and decision” / “Exact label and cost”:** task readability needs the registered predecessor-relative normalized gold-letter endpoint and disjoint items. WikiText cannot replace it. Qualification currently fails despite exact repeats (`docs/reviews/P5/tf-v1-requalification-2026-11-06.md`, §A). My sealed purchase is conditional on a dated amendment, qualification and contrast-specific power, not merely the proposed component count.
- **Editor, `round1-editor.md`, “Smallest discriminating experiment and price” / “H5 decision and the order-statistic control”:** extraction is processing time, and shadow selection is not realized savings. I withdraw my round-1 combined GPU-hour totals: extraction used CPU workers (`docs/reviews/P5/d8-regeneration-2026-11-06.md`, §2). Editor transfer requires episode-held-out proposals and prospective cost accounting, not extrapolation from zoo rank.

## 2. Rebuttals

**Against representation §2's “clean structure negative.”** H1w is useful conditional evidence, not a decisive structure rejection: its estimate is 0.0071, but the bootstrap-t power lower bounds are 0.728 conditional and 0.600 joint, below 0.80; all six intervals cross zero. The verified gate explicitly forbids pass/kill labels (`docs/reviews/P5/h1w-gate-2026-11-06.md`, §§1,7). Changing the representation does not automatically power the new paired contrast.

**Against representation §3's zero-shot quadratic readout.** Its proposed `q=tr(G D A Dᵀ)/2` is nonnegative for covariance factors. A signed gold-log-prob response need not have nonnegative curvature. Therefore Fisher sensitivity cannot simply replace the objective Hessian with the same positive sign for every endpoint. Retain `q` as a magnitude/trust-region feature; estimate signed response separately. This is an algebraic objection to the displayed formula in `round1-representation.md` §3, not an empirical claim. Likewise its chain cross term depends on current displacement and incremental edit, not uniquely on history: it belongs in the stateless control before any H2 claim.

**Against making metrology §2's layer-set encoder the new structure by itself.** Existing M1 already includes typed, layer-indexed directional tokens (`src/wsp/predictor/models/__init__.py::TensorEncoder`; `src/wsp/features/svd.py`). The missing comparison is informative anchoring versus random anchoring on identical inputs and losses.

**Against treating editor's primary critic as sufficient for this question.** Its cached `D x` captures a local writer response, not the complete downstream response; the editor explicitly acknowledges this limitation in “Primary structure and two alternatives.” It is a useful separately named activation-assisted comparator, requiring probe-only and probe-plus-descriptor controls and measured probe costs. Its success alone would not establish weight-only prediction.

## 3. Revised single proposal: oriented local response chart

Consume actual predecessor `W`, cumulative displacement `W−W₀`, and incremental effective-bf16 edit `D`, separately. This corrects the cumulative chain-feature mismatch documented in `src/wsp/features/extract.py`, opening docstring. Preserve exact layer/module/support, signed low-rank directions, dense supported sparse slices, spectral tail and rounding loss. Add signed contractions `<D,Pout W Pin>` with frozen coordinate-block projectors, also against `W₀` and `W−W₀`.

Use a small within-layer set encoder, ordered cross-layer interaction and supervised response head. Enforce zero response at zero edit. Predict item-conditioned normalized gold-letter score changes, using frozen text features so unseen items remain representable; aggregate exactly as TF-v1. No JEPA reconstruction or weight decoder in the primary purchase.

**Mechanism:** equal-norm edits can differ in alignment with the predecessor and in cross-layer interference. These signed interactions could predict residual effects beyond family/support descriptors. Compare original StatNN, descriptor-only, delta-norm, exact-layer scalar tree, augmented StatNN on identical chart features, and original-input M1. A tree-only gain establishes representation value; only gain over augmented StatNN establishes this encoder's added value.

**Decisive test.** Retain the intervention from `round1-world-model.md`, “Smallest decisive purchase”: planned 20 development source blocks, then 100 fresh sealed blocks, each containing an edit and support-preserving orientation control at strengths `{−1, 1/2, 1}`: 120 development and 600 sealed rows. Reuse source deltas; whole shared-source/ancestry closure stays in one partition. Match post-bf16 effective norm/spectrum, report residual mismatch, and never select by observed effect. Freeze one model before sealed outcomes.

Use registered 1,000-item MMLU gold-letter panels with disjoint development/sealed groups and actual-predecessor references (`docs/PREREG_H1_TF.md`, §§1–2). Qualification must be re-established first. WikiText remains diagnostic. Require planned ≥15% paired residual-MAE reduction against both original and augmented StatNN, superiority to descriptor-only, positive component and joint-item 95% lower bounds, and a positive intact-versus-scrambled-direction contrast. Registered H1b additionally retains MMLU Spearman ≥0.6 (`docs/HYPOTHESIS.md`, H1).

Pilot loss simulations must demonstrate ≥80% power with the interval calibration requirements in `docs/PREREG_H1_TF.md`, §5/A2.3; 100 blocks is a floor, not proof. Retain my round-1 outcome-free resizing cap of 300 sealed blocks; stop unresolved if insufficient.

**Price:** measured warm scoring is approximately 1.04 min/entry; extraction 54.3 CPU-seconds/entry (`docs/reviews/P5/d8-regeneration-2026-11-06.md`, §§2,5). Derived scoring budgets are approximately 2.08 GPU-h for the pilot, 12.48 for 720 total rows, or 33.28 for the 1,920-row cap. Extraction separately costs approximately 1.81, 10.86 or 28.96 CPU core-hours. Retain the planned 2 GPU-h overhead cap, now covering references/repeats, custom features and predictor training; if insufficient, stop and reprice before expansion. No new SFT, fast-tier validation or H2 scoring is included.

**Kill:** qualified, adequately powered upper 95% gain bound below 5% rejects useful chart superiority on this endpoint (`docs/PREREG_H1_TF.md`, §4). Wide intervals, failed qualification or exhausted overhead mean unresolved. H5 remains a later episode-held-out test with rationale excluded, failed attempts counted, equal-budget best-of-N controls and ≥30% realized evaluation savings (`docs/H5_DESIGN.md`; `docs/HYPOTHESIS.md`, H5).

## 4. Relation to the base-anchored sensitivity field

Complementary: the field supplies objective-specific local sensitivity; the chart supplies predecessor-conditioned, support-aware interactions and a learned signed response. Frozen field contractions could become chart coordinates, but would change the access claim to weights plus base sensitivity and require a field-only comparator. The weight-contraction chart remains the primary test. A field's chain cross term is a stateless feature, while genuine history must beat an identical-current-state control. Neither approach licenses intrinsic path dependence at identical final weights.
