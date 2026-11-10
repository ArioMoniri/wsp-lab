1. Register an external-edit panel separately; a two-tensor edit is one zoo entry, not two observations (§1).
2. Freeze realised deltas, field signs and chart predictions before TF-v1 scoring; unavailable directions stay unavailable (§1).
3. Rane’s 42/61/64 result favors activation assistance as a prior, not a transferable effect-size estimate (§2).
4. Our field retains objective alignment absent from Rane’s summaries; its incremental gain remains unmeasured (§2).
5. Sparse deletion is not low rank; its K-FAC contraction retains interactions between deleted coordinates (§3).
6. SWEET and TFTV can be rank-one per matrix, making norm-matched spectra direction-blind (§3).
7. The four-dimensional chart preserves some orientation but its exported columns omit the raw projected blocks (§3).
8. G2 cannot reconstruct correction-gradient covariance; mean-gradient spectra are a distinct, cheaper diagnostic (§4).
9. Add a scripted H5 estimator comparator; retain the ≥30% evaluation-savings criterion and Part A controls (§5).
10. Keep F5.5 unchanged: small external panels diagnose transfer, not programme survival (§6).

Representation, round 4, 11 November 2026. Recommendations below are proposed amendments, not executed tests. Repo paths are relative to this worktree.

### 1. External edits as zoo objects and sign tests

**Yes, as an F8 descriptive panel.** Represent the released Astra patch as one base-root entry containing both named dense deltas, with source/checkpoint hashes, exact base revision, changed columns, nominal factors, realised bf16 support/norm/rank, and shared discovery provenance. The existing loader accepts dense `delta.safetensors` plus `edit.json`; semantic family registration must be explicit rather than disguised as LoRA (`src/wsp/zoo/apply.py`, `src/wsp/schema.py`). Derive each scored increment as `fp32(bf16(W₀+Δ))−fp32(W₀)`; keep both tensors together and related scripted descendants in one source cluster (`src/wsp/features/response_chart.py`; `docs/PREREG_H1_TF.md` A5.2–A5.3).

For nominal column edit D=−u eⱼᵀ, field sign is −uᵀg[:,j]; q=½(uᵀGu)Aⱼⱼ; chart block is −(Poutᵀu)(eⱼᵀPin). Two matrices yield two tokens, summed first-order prediction, and per-tensor q—not a full cross-matrix Hessian (`response_field.py::contract`; chart implementation, both under `src/wsp/features/`). No rationale text enters the predictor.

The [Scope paper](https://arxiv.org/html/2611.09382), §7.1, specifies its column intervention but does not supply u: shape and changed-value count cannot recover orientation. Include it only when actual bytes become available. The [aiXiv replication](https://aixiv.science/abs/aixiv.261103.000004) supplies a checkable difference-of-means recipe, but shares supplied layers/budget/scale with Astra; these are not independent discoveries.

Freeze eligible byte identities, endpoint-specific signs, chart predictions, source clusters and missing entries before scoring. Score WikiText and descriptive MMLU with shared qualified references; apply the registered item-block resolution procedure, report every unresolved entry, and never use familiar MATH gains as endpoint labels (`docs/PREREG_H1_TF.md` A5.3–A5.5). At the brief’s ≈0.02 GPU-h/entry, 4–8 entries cost ≈0.08–0.16 GPU-h scoring, plus unreused references/extraction/qualification (`docs/council/2026-11-11/BRIEF.md`). That sample cannot satisfy Part A’s 15-resolved-pair floor. Value: detect gross sign reversals or abstention on external geometry. Failure on resolved TF effects challenges transfer/locality; null TF effects cannot refute a generated looping repair or establish missing capability.

### 2. Weights versus activations

[Rane §4.3/Table 2](https://arxiv.org/html/2611.07610) reports frozen LODO domain recovery on Qwen3.5-0.8B: weights 42%, activations 61%, outputs 64%. White-box inputs comprise 48 summaries: block-type norm mean/std/max/sum, a 24-point normalized depth profile, eight normalized mean singular values, and four global norm/ratio/count scalars; singular values come from the QR-reduced LoRA core. They discard singular directions. Clustered white-box uncertainty includes chance; the access ordering is suggestive, not a universal hierarchy.

They are weaker **in directional content**, not demonstrably worse predictors of our endpoint. Our signed objective contractions, curvature magnitudes and layer-specific chart coordinates contain information those summaries omit (`src/wsp/features/response_field.py`, `response_chart.py`). But the field uses auxiliary base gradients and Fisher statistics: call it *weights plus frozen receiver sensitivity*, not data-free weight reading.

**Prior:** expect activation-assisted prediction to outperform field-only prediction, while retaining a plausible modest field gain from objective alignment; neither gain is quantified. Reduce optimism about weight-only summaries without importing the accuracy gap into residual-error reduction C. Domain classification differs from signed benchmark change. Our neural seed-split results still trail StatNN (`docs/reviews/P4/d3v2f-summary-2026-11-08.md`). Report chart-only, field-only, hybrid and probe-assisted access explicitly, with matched ablations, costs and unchanged secondary status; the registered probe is linear D·x, not a full edited-model activation evaluation (`docs/PREREG_H1_TF.md` A5.4/F5.1–F5.2).

### 3. New families: what survives featurisation

Write g for a named-score gradient, A/G for input/output Fisher factors. The implemented field uses s=⟨g,D⟩ and q=½tr(GDADᵀ), not a signed Hessian Taylor term (`src/wsp/features/response_field.py`).

**Deletion.** With keep-mask M, D=−(1−M)⊙W. [Deletion Tuning](https://arxiv.org/html/2611.10866) learns gate-projection masks. If deleted entries are indexed (a,b), q=½Σ DₐᵦD𝚌𝚍Gₐ𝚌Aᵦ𝚍. Sparsity does not diagonalize this expression or imply low rank. The chart sees PoutᵀDPin; StatNN sees nonzero fraction, moments and spectrum in gate/depth pools, losing which coordinate was removed. Neither small support nor a small norm implies behavioral neutrality.

**SWEET.** Its [directional orthogonalization](https://arxiv.org/html/2611.10974), eq. 9, gives D=−ddᵀWS for selected neuron columns/head subblocks S. One direction yields rank at most one in that matrix; combined directions need not. s=−dᵀgSWᵀd and q=½(dᵀGd)(dᵀWSASWᵀd) for symmetric column selector S. Chart coordinates retain projected direction and location; pooled statistics retain suppression magnitude but no objective alignment.

**TFTV.** [Eq. 2](https://arxiv.org/html/2609.09054) constructs D=αs̄vᵀ, where v=Σᵢsign(μᵀvᵢ)σᵢvᵢ from W’s SVD and expected module input μ. Thus s=αs̄ᵀgv and q=½α²(s̄ᵀGs̄)(vᵀAv); chart block is α(Poutᵀs̄)(vᵀPin).

These are algebraic predictions from the cited recipes and implementation, not zoo measurements. For any nonzero exact rank-one D, normalized singular spectrum is (1,0,…), stable rank one, entropy zero, tail beyond eight zero. At fixed norm a spectrum-only readout is degenerate for direction, especially D versus −D. Full StatNN is not spectrum-only: moments, sparsity and predecessor cosine can differ (`src/wsp/features/stats.py`, `pool.py`). Deletion is not generically spectrally degenerate; a single-row/column mask is a special case. bf16 rounding can increase rank for dense outer products; audit realised spectra rather than imposing rank one.

Implementation caveat: chart tokens compute raw 4×4 blocks, but `to_columns` exports their elementwise products with base/predecessor/displacement blocks plus scalar diagnostics. At a base-root E=0, displacement products vanish and base/predecessor products coincide. Direction can also vanish in the random projection’s nullspace (`src/wsp/features/response_chart.py`).

Propose two fixed candidates per family: six entries, ≈0.12 GPU-h scoring plus construction/references. Freeze recipe data and strengths without endpoint selection; evaluate as family-held-out predictions, no refitting. A learned deletion mask needs optimization: its construction cost is unmeasured, not “free.” Prefer available masks and forward-only recipes first. This targets family transfer more directly than more Gaussian 4B edits, but information/GPU-h is a prior: that diagnostic had unresolved effects and failed device pairing (`docs/reviews/scale/scale4b-diagnostic-2026-11-10.md`).

### 4. Correction spectrum: missing statistic, missing artefacts

[Behavioral Rate, Appendix B](https://arxiv.org/html/2611.06598) distinguishes mean μ, uncentered J=E[gᵢgᵢᵀ], and covariance C=J−μμᵀ; its empirical spectrum is a projected output-head proxy, not a full Hessian-whitened information rate.

G2 stores averaged objective gradients and separately model-sampled Fisher K-FAC factors; it discards individual score gradients. The probe cache contains down-projection inputs, not objective-labelled output-head corrections (`response_field.py::_collect_objective`, `scripts/g2_fields.py::run_probe`). Therefore exact J/C cannot be recovered from existing artefacts. SVD of stored matrix gradients, or eigenvalues of A⊗G, are computable CPU diagnostics but must not be called correction covariance.

For an actual feature, replay the frozen auxiliary objective items, project each normalized item gradient into pinned chart frames, flatten/concatenate as zᵢ, and compute eigenvalues of ZᵀZ/n (or its item Gram matrix); center separately for covariance. Match WikiText token weighting and equal-item objectives. Store spectrum/tail/logdet with a fixed scale; report a projected proxy. This requires new backward passes and prospective cost measurement. A per-objective constant spectrum alone cannot rank edits on one receiver; use edit energy/alignment in those correction eigendirections. Add only a separately versioned descriptive diagnostic, with identical columns supplied to augmented StatNN, never silently alter frozen arms (`docs/PREREG_H1_TF.md` A5.4.1).

Compression recovery becomes a testable chart prediction only with both corrupted and compressed realised deltas, frozen predictions before their endpoint scores, and local-validity checks. A scalar rank/rate cannot identify which harmful direction compression removes; no existing recovery result is claimed here.

### 5. Controls and H5

**No Part A control change:** no flaw found in its spectrum/support/rounding audits (`docs/PREREG_H1_TF.md` A5.3). Concept Country and Barry’s specified URLs failed web retrieval and direct fetch returned HTTP 403; I cannot independently substantiate their numerical/control claims. The Concept Country control described in `docs/council/2026-11-11/BRIEF.md`, if verified, tests an analogous directional hypothesis, but is not our spectrum-matched rotation.

Add scripted difference-of-means and LP proposals beside random best-of-N, with matched data, tensor support, scale/search budget and charged evaluations; separate supplied-location from discovery runs. aiXiv’s fetched abstract reports 93% gain recovery conditional on supplied choices. [Biswas](https://glitch401.github.io/writing/editing-a-models-weights/) also reports effective scripted length edits and selection-sensitive accuracy claims. [IWIBench Table 2](https://arxiv.org/html/2611.08647) ranks Sol medium/high above Astra, with task heterogeneity; editor reward does not measure critic savings. Keep ≥30% realised savings and held-out acceptance; charge estimator preparation too (`docs/council/2026-11-10/SYNTHESIS.md` §1/F5.4).

### 6. Line stop and exact related-work additions

Keep F5.5 and FP6’s enumerated contrasts/validity/power unchanged. F6’s exact predecessor gradients do not make base-factor q exact; F7’s exact-donor restriction is reported in the brief, but an F7 block is absent from this checkout’s A5: reconcile its registration before relying on it (`docs/PREREG_H1_TF.md` F6; `docs/council/2026-11-11/BRIEF.md`; `docs/prereg/a5_manifests/power/fields_manifest.json`). External panels cannot rescue funding or retrospectively extend the conjunction.

Add these sentences, with the linked sources above:

“Rane’s frozen domain-recovery experiment favors activation and output summaries over direction-free weight summaries; it does not estimate signed edit-response prediction.”

“Deletion Tuning, SWEET and TFTV motivate sparse-mask and activation-derived edit-transfer tests, distinct from predicting effects of unseen deltas.”

“Receiver-relative correction spectra describe correction diversity; our frozen field retains mean objective alignment and model-sampled Fisher magnitudes, not per-example correction covariance.”

“Localized WSP gains and scripted replications motivate external-edit controls but do not establish a held-out weight-delta response predictor.”

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---|---|
| External panel | Exact oriented deltas | Proposed F8: frozen signs/chart, TF-v1 | ≈0.08–0.16 scoring + overhead | No expansion if unresolved; resolved sign reversals flag transfer failure |
| Six family entries | Mask/suppression/steering transfer | Proposed F8: family-held-out, no refit | ≈0.12 scoring; construction unpriced | Stop/reprice construction; no efficacy claim from unresolved effects |
| Correction diagnostic | Gradient diversity/alignment | Proposed versioned auxiliary diagnostic | CPU proxy 0; exact replay unpriced | Do not launch exact variant without price; discard covariance claim without per-item gradients |
| Scripted H5 comparator | Separate estimation from LLM search | Proposed G5 amendment, matched access | Unpriced; measure before launch | Retain ≥30% savings requirement; no budget overrun |
| Frozen programme | Orientation beyond spectrum | A5.3–A5.5, F5.5/FP6, F6 | Existing allocations | Only existing joint rule; invalid/underpowered means unresolved |
