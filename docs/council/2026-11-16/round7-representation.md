1. Develop a scripted-statistics residual response model for the existing 0.8B base.
2. Preserve signed field information; norms and spectra alone discard response direction.
3. Use a small nonlinear head to learn corrections and interactions between depth groups.
4. Treat matching the LLM reader with scripts as an empirical question, currently unanswered.
5. Weight access plus tools does not identify which computations caused a reader's success.
6. Start with already-exposed, non-Part-A root outcomes; freeze source-connected folds first.
7. Predict TF-v1 endpoint change in nats; report direction only as a secondary diagnostic.
8. Keep an LLM reader as one bounded diagnostic alternative, with matched information controls.
9. F12 failed qualification; no purchase, repeat or extension follows from this recommendation.
10. Any confirmation requires a new prospective registration, after Part A and at the user's decision.

## What representation was actually read?

[WSR abstract](https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models) and [paper](https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models.pdf), fetched 16 November: subtle-edit balanced accuracy was 0.730 versus 0.578, difference +0.152, CI [+0.056,+0.256], on 10 changed interventions/18 changed labels. Single cell runs and uncorrected contrasts make this exploratory evidence for the **weights-plus-tools condition**, not a demonstrated reasoning mechanism. Self-reading remains inconclusive. Muse's 40 inert edits attracted 53–88% change predictions. Its post-normalization explains some scaling invariance, not every inert intervention. Labels require p<0.01 plus 10% token-length or 0.05 rate shifts. These are behavioral detection thresholds, not our nats floor. Bootstrap intervals omit reader-run variation; related dose variants further limit independence. The paper does not compare against a scripted-statistics reader. Its appendix permits norms, SVD, projections and matrix products; these permissions do not establish which statistics readers used successfully.

The fetched [repository](https://github.com/suayptalha/wsr) explicitly excludes paper results. In [sandbox_lib.py](https://raw.githubusercontent.com/suayptalha/wsr/main/wsr/sandbox_lib.py), `W.info` returns shape, dtype, Frobenius norm, mean, standard deviation and absolute maximum; `W.get` exposes CPU tensors. In [read.py](https://raw.githubusercontent.com/suayptalha/wsr/main/wsr/read.py), the weights arm also receives configuration, tokenizer and replacement tensors/rows; text receives structural edit descriptions. Thus an agent can recover numerical dose, concentration, effective rank, row/token identity and alignment with base directions. This is a richer information channel than an operation description. The helper and prompt are evidence of available inputs, not a released attribution study of actual tool trajectories. Freeze and replay those trajectories before claiming an exhaustive script reproduces the reader.

[labels.py](https://raw.githubusercontent.com/suayptalha/wsr/main/wsr/labels.py) implements its named Welch test with population variances and a normal-reference `erfc` calculation, not finite-degree-of-freedom Welch-t calibration. [sandbox.py](https://raw.githubusercontent.com/suayptalha/wsr/main/wsr/sandbox.py) supplies read-only mounts and CPU execution; local mode is explicitly unisolated. A future adaptation should expose only allowlisted inputs through a container, not mount this checkout or invoke the measurement pipeline.

[Ahmed's abstract](https://www.alphaxiv.org/abs/2611.08070) and [PDF](https://pdfs.assets.alphaxiv.org/2611.08070v1.pdf), also fetched, separate real edge arithmetic from guessed semantics: output relabeling preserved recovered groups but broke interpreted class meanings. This is a small-CNN, mostly single-run, tool-free audit, not a refutation of WSR. It motivates opaque edit IDs, removal of intended-effect prose, and controls for metadata priors. A general reader accessing model-specific numbers is not automatically a learned model-specific response model; that requires supervised response prediction and source-held-out validation.

Fetch record: all required pages, both PDFs via the web reader, and the four linked raw code files were retrieved. A separate direct Python fetch of the WSR PDF returned HTTP 403; the web-reader PDF succeeded.

## Existing representation and its limits

`src/wsp/features/stats.py` already computes 18 delta features: norm, relative norm, old/new cosine, moments, extrema, sign/nonzero fractions, eight singular values, spectral entropy and a rank summary. Its `stable_rank` uses only the top-eight energy: label that truncation explicitly. `extract.py` uses the realised rounded delta and records rounding loss; its documentation warns that LoRA-factor tokens describe the unrounded factorization. `svd.py` canonicalizes factors, while `response_chart.py:contractions` retains signed base/predecessor contractions and spectral-tail diagnostics. `pool.py` separates 17 tensor roles and four depth buckets. Derive widths from the current schema, not stale prose in extraction documentation.

These statistics cannot generally identify loss response: equal-norm, equal-spectrum edits may align differently with the base gradient. Keep a=Σ〈g,D〉 and q=Σ½tr(GDADᵀ), and their depth decomposition (`response_field.py`; `docs/PREREG_H1_TF.md`, A5.8.1/F12.1). The field already carries model-specific task information; q is a nonnegative Fisher/Gauss–Newton magnitude, not signed curvature. Its three-coefficient readout cannot express cross-tensor interactions (`docs/council/2026-11-11/SYNTHESIS-r6.md`). A learned response correction is the useful next level of the user's thesis; learning the known transition W+D adds little.

Earlier seed-split StatNN correlations 0.56/0.66 exceeded matched neural encoders, but these were MMLU/HellaSwag rate targets, not WikiText nats; family transfer was poor and neural re-instantiation drift substantial (`docs/reviews/P4/d3v2f-summary-2026-11-08.md`). This supports a compact baseline-first design, not a universal statistical sufficiency claim.

## One buildable model: scripted residual reader

**Proposed design, not a registered modification.** Compute a fixed 48-vector per root edit:

- 18 global means of existing per-tensor delta statistics;
- a and q, plus a and q summed separately in each of four depth buckets: 10 values;
- 17 role shares of total squared delta norm, in `pool.py:TYPES` order;
- maximum tensor share of squared norm, fraction of intended nonzeros lost to rounding, and fraction of base tensors changed: three values.

All share denominators have an explicit zero-edit convention. Use realised D throughout; contract cached, scoring-disjoint base fields, with provenance and support checks. Missing required fields invalidate a row for every compared arm. Do not silently encode missingness as a zero response. No new base-field collection or target forward pass is authorized here.

Fit an outer-training-only F3 baseline. Transform the 48 inputs with train-only signed-asinh scales and standardization; feed `48 → 8 GELU → 1`. The residual is h(x)−h(x_zero), so its correction vanishes exactly at zero edit. This is **401 head parameters**, calculated as 48×8+8+8+1, plus F3's three coefficients. Predict F3+h in nats/token with L1 training on inner-OOF F3 residuals. Refit F3 and the selected head using outer-training rows only. Adopt F12.2's inner component folds, seeds and stopping recipe, and F12.3's optimizer settings as development defaults; any departure is logged before fitting. This nonlinear pooling permits interactions among depth-group summaries but cannot recover every tensor interaction or arbitrary semantic edit effect.

**Population.** F12.2 records 123 already-observed roots outside the 52 Part A components, in 89 source components; the stricter development-status subset has 79 roots/52 components. Begin with the stricter subset; explicitly exposed old-seal components require a provenance decision before inclusion. The preregistration and `docs/PROGRESS.md` correct the census to **689 records**, including the base, versus the brief's 694. Neither is an eligible independent training count. Exclude Part A's entire connected components, fresh/unselected and still-sealed components, non-root steps, aliases crossing split boundaries, missing-provenance rows and outcome-dependent row selection. Use approved outcome exports later; this research opens no evaluation files. Never train on all census records indiscriminately.

Freeze component closure through shared sources, directions, ancestry and duplicate delta bytes before folds. Use nested component-held-out predictions, equal tuning opportunities and a separate leave-one-family-out diagnostic. Existing outcomes are exploratory development evidence; no cross-validation result becomes confirmation.

**Baselines and attribution.** Compare field F3; existing augmented StatNN; and a scripted ridge residual over exactly the same 48 inputs. The latter isolates nonlinear readout benefit. Ablate every field-derived input together to measure field contribution. The one alternative is an LLM reader diagnostic on the same allowed development rows: identical text-only edit descriptions; descriptions plus the frozen numeric table; and weights/tools with a logged bounded loop. Give neutral framing, endpoint units and sign convention, and require numeric prediction plus uncertainty. Compare the table-only reader with the scripted head, then inspect incremental tool computations and add them to the script only within development. Match those extra features for a new held-out comparison. A tie would support sufficiency of those measured statistics on that population; it would not prove identical mechanisms. Do not add an LLM-to-head hybrid until there is reproducible extra information to justify it.

Test first: realised-bf16 no-ops, LoRA gauge equivalence, tensor-order invariance, norm-share closure, signed-a reversal under exact ±D, q nonnegativity, residual zero at zero, preprocessing isolation, component-disjointness, and rejection of prohibited paths. Score failures and abstentions explicitly.

## Measurement, qualification and cost

Predict continuous ΔNLL; report sign outside a development-frozen uncertainty region descriptively. “Same” must mean unresolved at the chosen resolution, not proven equivalence. A5/F9a records 0.00487 nats as an upper component-bootstrap bound on panel half-width; F8a.2 calls h*=0.0049397 a historical screen with reproduction obligations. Neither licenses converting WSR behavior labels into TF-v1 truth (`docs/PREREG_H1_TF.md`, F9a/F8a.2).

F12's alternative produced C=0.172 against F3 but 0.045 against StatNN, below its 0.05 point threshold; ordered detection was 218/1000, lower bound 0.180. All 14 gates failed. Even F3-only marginal detection was 424/1000; ridge-bias stress and undercoverage independently block purchase (`docs/reviews/council/board-a-f12-sim-report-2026-11-14.md`). Removing StatNN cannot repair interval calibration. Adding dependent chain rows cannot manufacture independent roots.

For any **new** proposed registration, use the outcome-blind F12 simulator machinery as a pricing tool, after its Meter.end/projection obligations are repaired. Freeze comparator, meaningful margin and component population first; preserve shared-reference/item noise, refit pipelines, test ridge-bias stress and both coverage tails. Require power ≥0.80 and stress/coverage error bounds ≤0.07 under prospectively reviewed rules. Consider more genuinely independent roots or lower-noise targets, but compute the minimum viable design; no qualifying sample-size number was computed here. A sign endpoint changes the scientific claim and needs separate calibration. The failed F12 run remains closed; no new simulator execution is authorized by this report.

Suggested development caps, **unmeasured planning limits**, are 8 CPU-h, zero GPU-h and zero LLM tokens for the scripted candidate. The optional reader audit has a separate 200,000 total-token cap including prompts/tool responses, within the same CPU cap; stop and reprice on exhaustion. Use cached features and a metered small CPU pilot before expensive extraction. Full qualification simulation needs its own priced cap: the prior run consumed 102.01 CPU-h (`PREREG`, chair acknowledgement). Stop development if provenance fails, calibration fails, or outer-fold residual MAE fails to improve on scripted ridge and StatNN; this is a development decision, not a scientific null result.

F8/F9/H5/F12-I remain unchanged. This model could later supply an H5 critic, but prediction gains do not establish search savings or guardrail compliance (A5.7). Confirmation requires a fresh prospective panel after Part A under existing ordering and an explicit user decision.

| proposal | mechanism | registered test | cost | kill criterion |
|---|---|---|---|---|
| Scripted residual reader | Signed field + weight statistics; 401-parameter correction | None added; new prospective panel only | Proposed ≤8 CPU-h; 0 GPU-h; 0 tokens; simulation separately priced | No reproducible gain over ridge/StatNN, provenance failure, or failed qualification |
| Optional reader diagnostic | Text/table/weights comparison with logged computations | Exploratory only; no registered-plan changes | Within CPU cap; proposed ≤200k tokens; 0 GPU-h | No incremental predictive information or cap exhausted |
