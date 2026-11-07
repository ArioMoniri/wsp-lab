1. Primary: a base-anchored, direction-sensitive local state/edit chart with a small supervised response head.
2. JEPAWG supplies controlled-population methodology, not evidence for a universal LM edit law.
3. Preserve exact layer, module, support and signed bilinear contractions with the predecessor weights.
4. Predict predecessor-relative item-score changes; benchmark prediction must survive disjoint items.
5. A new token transformer alone repeats M1: directional tokens and layer embeddings already exist.
6. Decisive intervention: direction versus matched-spectrum rotation at held-out strengths and sources.
7. Start with 20 development source blocks; buy 100 sealed blocks only after qualification and power.
8. Proposed 720 new rows cost 23.2 local GPU-hours in scoring/extraction proxies, plus explicit overheads.
9. Regenerating 358 deltas restores potential closure, not independent histories or measured H2 power.
10. For LLM edits, score the executable delta first; rationale is a separately controlled information arm.

# World-model recommendation — 7 November 2026

All counts, thresholds and budgets below are prospective unless attributed. No experiments were run. Paths are repository-relative.

## What the external structures actually license

I independently fetched all five requested papers and the blog. [Weight-Space Physics, §4](https://arxiv.org/html/2607.07127) pairs physical couplings with trained sampler weights, aligns their embeddings using covariance-free VICReg, freezes the encoders/predictor, then fits a weight decoder. Its multi-seed experiment includes separately initialized conditional networks; it is inaccurate to describe every population in this paper as sharing one initialization. Physical parameter indexing is the essential supervision, not merely many weight snapshots.

[Bifurcations, §§3.1–4.3](https://arxiv.org/html/2610.30847) uses parameter-indexed forecasting networks fine-tuned from a shared conditional forecaster. Its decoded-weight response is sensitivity to physical parameters, not edit time or benchmark sensitivity. Known bifurcations provide independent validation. My inference: reproduce the controlled intervention and external validation before importing its latent geometry. A smooth weight reconstruction can ignore a rare behaviorally important direction.

[W2T, §§3–4](https://arxiv.org/html/2603.15990) canonicalizes LoRA updates with QR/SVD and models rank and position tokens. [LoL, §§2–3](https://arxiv.org/html/2410.04207) instead builds GL-equivariant processing for the factorization freedom. Neither symmetry aligns unrelated base-model neuron coordinates. Degenerate singular subspaces also need subspace-invariant contractions rather than trusting a uniquely oriented singular vector. [GMN, §2](https://arxiv.org/html/2312.04501) constructs parameter graphs with permutation-equivariant message passing. That is a principled architecture prior; its experiments do not establish a cheap predictor for our hybrid 0.8B model.

This builds on, rather than repeats, `docs/council/2026-11-05/round1-jepawg.md`, `round2-jepawg.md` and `SYNTHESIS.md`: the missing experiment is an intervention that identifies directional response beyond descriptors, not another unlabeled geometry visualization.

## Which negatives bear on structure?

| Brief result | Interpretation | Single separating measurement where attribution is unresolved |
|---|---|---|
| 1: trees beat neural heads | Conditional architecture/representation negative, mixed with label resolution and sample size. It rejects the tested heads' superiority, not weight readability. | Same-input tree versus chart on qualified, source-held-out directional residual losses. |
| 2: family transfer | Mixed distribution shift, calibration and structure; pooled rank is insufficient. | Within-held-out-family paired MAE against norm/support controls with no test-family calibration. |
| 3: history | Data/contrast failure, not evidence against dynamics. | Complete length≥4 lineages with within-model mask/shuffle/placebo contrasts and joint loss intervals. |
| 4: TF-v1 | MMLU is a measurement gate failure; WikiText is a conditional negative for pooled-statistic sensitivity. | Item-panel uncertainty and paired directional-model losses on the same endpoint. |
| 5: repeats/nulls | Measurement and intervention coverage. Exact repeatability does not imply resolvable edit effects. | Joint item/component precision for the matched-direction contrast. |
| 6: tree importance | Predictive association; neither causal mechanism nor proof that direction is irrelevant. | Spectrum/support-matched rotations with measured outcomes. |
| 7: blog | Task-definition mismatch, not a test of our predictor. | Prospective forecast of unseen proposals before behavioral evaluation. |

Sources for these project claims: result 1–2, `docs/reviews/P4/d3v2r-summary-2026-11-06.md`; 3 and 6, `docs/council/2026-11-05/SYNTHESIS.md`; 4, `docs/HYPOTHESIS.md`; 5, `docs/RESULTS.md` and `docs/council/2026-11-07/BRIEF.md`; 7, verified blog below. Crucially, the recovered summary has **no M3-family metrics**, and individual M2 family cells are not all bounded by the brief's pooled range. Treat missing/NaN cells as missing, not transfer failures.

## Primary: an oriented local response chart

For predecessor W and effective incremental edit D, encode separately the cumulative predecessor displacement W−W₀ and D. The current extractor's chain features are cumulative-from-base; its LoRA-factor tokens describe unrounded factors while generic SVD tokens describe effective rounded deltas (`src/wsp/features/extract.py`). Never substitute either for the actual incremental effective edit.

Use per-matrix tokens containing exact layer/type, changed-column/row support, rounding losses, singular energy and residual tail energy. Add signed contractions

`b_ab(W,D) = <D, P_out,a W P_in,b> / (||W||F + epsilon)`

where P are fixed, hashed coordinate-block projectors, plus contractions with W₀ and W−W₀. Compute from low-rank factors without forming full projected matrices; jointly flipping SVD factor signs leaves the contraction unchanged. Sum over degenerate singular blocks. Keep sparse-support summaries independently: a sparse edit need not be low rank. Nonmatrix norms/gates receive separate scalar tokens. No arbitrary neuron permutation of D alone is an invariance: it changes its alignment with fixed W.

A small typed set encoder pools within layers, followed by ordered layer interaction and a regularized head. “Set” means invariance to token serialization, not to exchanging functional layers. Qwen has 24 layers mixing 18 GatedDeltaNet and six full-attention blocks; retain their identity and writer/reader distinction (`docs/ARCHITECTURES.md`).

This is more than the existing architecture only if the predecessor contractions help: `src/wsp/features/svd.py` already emits rank-wise projected directions; `src/wsp/predictor/models/__init__.py` already supplies TensorEncoder layer/type embeddings and M1 attention. M3 predicts the next cumulative-feature latent from a descriptor; it does not implement JEPAWG's physical parameter population. `pool.py` reduces exact-layer information to depth buckets; `stats.py` uses only the leading spectrum in its stable-rank statistic. Prefer code to stale dimensions/type counts in `docs/MODEL_CANDIDATES.md` and extractor docstrings.

The chart is `z(W+D)=z(W)+A(z(W),D)`, with `A(z,0)=0`. Its decoder predicts per-item normalized answer-score changes, then aggregates to Δ gold log-prob and Δ accuracy. Supervised task loss is primary; train-only JEPA successor alignment is an ablation against the identical supervised encoder. A weight hyperdecoder may reconstruct only local update coefficients, never the entire model as the first purchase.

A dynamical atlas consists of overlapping neighborhoods indexed by predecessor displacement, edit support/direction and strength. Test intermediate-strength interpolation, two-step composition, and rollout error. Do not impose conserved norm, energy, monotonic capability or commutativity. For fixed evaluation state and identical final weights there is no intrinsic dependence on the path taken; history can help only because our compressed state omits information. Optimizer state becomes an additional input only when forecasting the next optimizer-produced edit.

## Two alternatives

**A. Local GMN.** Build a typed graph of changed channels and connected predecessor blocks, explicitly representing residual, gating and mixer relations. Message passing may learn cross-layer interference that pooled statistics lose. Compare with the primary chart on identical selected inputs, and test simultaneous function-preserving relabeling of W and D. A compressed block graph is an approximation, not the full GMN symmetry theorem. Reject full per-weight graph construction until a memory/runtime audit fits the cap. This is the stronger symmetry option when channel correspondences become the bottleneck ([GMN](https://arxiv.org/html/2312.04501)).

**B. Probe-conditioned response hypernetwork.** Cache predecessor hidden responses on 32 frozen, development-only prompts; condition a small hypernetwork producing local response coefficients on these responses and directional edit tokens. Train to predict held-out item-score changes. It might beat StatNN because functional context disambiguates equally sized edits. Include probe-only regression, probe+descriptor regression, and augmented StatNN; otherwise any gain might simply purchase a partial evaluation. It is explicitly activation-assisted, with no claim of weight-only inference. Test a coefficient decoder against direct regression before adding JEPA reconstruction. This is a proposed adaptation, not a demonstrated result of either JEPAWG paper.

## Smallest decisive purchase, with gates

**Existing development data:** use eligible LoRA/task rows and restored chain states from the planned 1,259-row zoo; null families are controls, not the primary learning population. The 358 restored delta files alone do not supply labels or independent roots (`docs/council/2026-11-07/BRIEF.md`). Audit byte identity, source closure and predecessor labels before training. Cache-only fitting may start with the recovered 297 rows/270 clusters; these are development-only (`docs/reviews/P4/d3v2r-summary-2026-11-06.md`).

**New intervention:** 20 independent development source blocks, then 100 fresh sealed source blocks. Per block generate D and one support-preserving, spectrum-matched orientation control R at strengths {−1, 1/2, 1}: six rows, hence 120 pilot and 600 sealed rows. Use signed row/column transformations within the declared support; log post-bf16 norm/spectrum mismatch and condition on effective statistics. Freeze strength scale from development resolution, never sealed outcomes. Include source runs from LoRA/task and localized writer edits; no claimed LLM-proposal transfer until a separately held-out editor-episode panel exists. Blocks sharing adapter/task-vector sources are one component, potentially requiring more sources to meet the count.

**Labels:** predecessor-relative TF-v1 normalized MMLU gold-letter vectors, 1,000 disjoint development and 1,000 sealed items; WikiText log-PPL on 64×1,024-token windows is diagnostic. Existing MMLU qualification failed; these proposed item counts are starting designs, not a waiver (`docs/HYPOTHESIS.md`). Pilot must pass frozen item/repeat/precision qualification before sealed scoring. No generated MATH/IFEval primary; loop repair remains a different endpoint.

**Splits and controls:** whole source/ancestry block holdout, including unlabeled pretraining; hold out intermediate strength within development, then entirely fresh directions at confirmation. Training-only nuisance adjustment for family/device/log-norm plus stronger support/task controls. Required comparisons: StatNN, descriptor-only, delta-norm, original-input M1, exact-layer scalar tree, augmented-input StatNN, same-input supervised versus JEPA chart, and direction-scrambled chart. Select one architecture on development. Family-holdout is separate; base remains fixed, so no H6 claim. Probe alternative adds the three probe baselines above.

**Decision:** prospective ≥15% residual MAE reduction over both original and augmented StatNN, lower joint component/item 95% bound >0, and gain that disappears on direction scrambling. Report ordinary and within-family ranks, but do not substitute rank for this test. The stronger H1b MMLU ρ≥.6 criterion remains required to claim that registered screen (`docs/HYPOTHESIS.md`).

Power is unknown. With 100 independent blocks, a normal paired-loss approximation gives 80% power at two-sided .05 for a .15 improvement only if normalized paired-loss SD≤`.15*sqrt(100)/(1.96+.84)=.536`. Estimate that SD from source-held-out pilot predictions; validate with joint component/item simulations, denominator uncertainty and null-size checks. Existing feature-contrast stand-in SD≈.87 instead suggests roughly 264 blocks by the same approximation, not 100 (`docs/council/2026-11-05/SYNTHESIS.md`). Permit a prospectively priced cap of 300 sealed blocks only if simulations justify it. Otherwise stop untested. Qualified, powered upper bound <5% kills useful chart superiority on this endpoint; a wide interval is inconclusive.

**Prices:** using brief rates, pilot 120 rows costs 2.0 h TF +1.8 h extraction =3.8 local GPU-h proxy; full 720 costs 12.0+10.8=23.2 h; cap 1,920 costs 32.0+28.8=60.8 h. These exclude references, repeats, new contractions and edit construction, whose incremental runtime must be measured in the pilot. Budget an additional 2 h measured-overhead cap before resizing. Existing predictor rate is 1–10 CPU minutes/run; six frozen runs project 0.1–1 CPU-h, not GPU-h. No SFT generation is purchased: reuse source deltas. Full fast-tier validation of 720 rows would add 90 local h or 37.2 effective H100 worker-hours at the brief's rates; omit it initially. Probe/GMN alternatives each get a 2 GPU-h pilot cap, not a fabricated throughput estimate.

H2 needs complete evaluated length≥4 closures, true/masked/shuffled history within the same model, lag controls and length-one equivalence within [−.05,+.05]. Restoration is feasibility, not sufficiency; census independent roots and simulate ≥80% power at the registered 15% effect before pricing missing evaluations (`docs/HYPOTHESIS.md`).

## Models editing models

[Blog Experiment 7](https://rikiyat.github.io/blog/weight-space-programming/) reports four weights-only failures followed by attention-assisted identification. Experiment 8 permits diagnostic forwards and reports localized MLP writer repair in layers 15 and 19; its additional Gemma experiment reports no established comparable improvement. These are interactive diagnosis/repair demonstrations, not held-out Δ-only forecasting.

For `llm_edit`, consume executable effective delta, support and predecessor first. Treat rationale as an optional blinded arm: delta-only, rationale-only, both, shuffled rationale; split by entire editor episode/provider/prompt lineage. Post-evaluation transcript text is unavailable at prediction time. Mechanistic proposals may occupy a narrow support unlike sampled kernels, so recalibrate uncertainty on all attempts, including failures and rejected proposals. Preserve `docs/H5_DESIGN.md`'s equal-budget best-of-N controls against the Meta^n order-statistic trap. Critic savings must count probes, editor cost and missed winners, meeting ≥30% real-evaluation savings at equal final score before claiming H5 (`docs/HYPOTHESIS.md`).

| structure | mechanism | decisive test | cost | kill criterion |
|---|---|---|---|---|
| Oriented local chart, primary | Predecessor-aligned direction × layer interactions | Fresh matched-spectrum source blocks; beat both trees and lose gain under direction scrambling | 3.8 h pilot; 23.2 h nominal scoring/extraction; 60.8 h capped, plus stated overhead | Qualified powered gain upper bound <5%; failed qualification means untested |
| Local GMN | Typed connectivity and coordinated channel symmetry | Same panel/inputs; simultaneous relabeling and interaction ablations | Reuse labels; 2 GPU-h implementation pilot cap | No paired advantage or graph fails resource cap |
| Probe response hypernetwork | Functional context conditions local response coefficients | Beat probe-only, probe+descriptor and augmented tree on disjoint items | Reuse labels; 2 GPU-h probe pilot cap; full cost measured before expansion | No incremental gain or probe cost erases evaluation savings |
