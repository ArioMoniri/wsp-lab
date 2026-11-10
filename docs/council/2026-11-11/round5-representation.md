1. Use one receiver-anchored interface with endpoint-specific prediction heads (§1).
2. Preserve realised increments, predecessor displacement, tensor identity and missingness (§1).
3. Extend forecast logging; calibrate signed magnitude, not only resolved signs (§2).
4. Propose a separate descriptive 10/30/50% mask ladder, leaving F8 unchanged (§3).
5. Separate SparseGPT mask transplantation from compensated pruning (§3).
6. Test field ranking within sparsity; do not assume locality at large deletion (§3).
7. Treat the four-dimensional chart and pooled StatNN as lossy controls (§3).
8. Give H5 objective saliency; attribution patching requires new measurements (§4).
9. Test verbal localisation incrementally; runtime LoRA gating saves validation, not construction (§§4–5).
10. Keep F5.5 and H5 success unchanged; qualify and price generation separately (§§6–7).

Representation, round 5, 11 November 2026. These are proposals for F9, not executed tests. Paths are worktree-relative; A5 denotes `docs/PREREG_H1_TF.md`.

### 1. One interface, several endpoints

Consume per-tensor tokens containing receiver/predecessor hashes, layer, role, shape, realised D, displacement E, objective-specific a/q, chart, statistics and coverage. D is fp32 subtraction of stored bf16 successor/predecessor; E is predecessor minus base. LoRA uses realised BA, chains snapshot differences, deletion −W⊙M, rank-one uvᵀ, runtime LoRA its completed adapter. Include rounding, source and recipe provenance; exclude rationale (`src/wsp/features/response_chart.py`, `response_field.py`; A5.7–A5.8).

Freeze receiver, G2 panels/factors/projectors, TF-v1 endpoints and splits. Learn a shared residual predictor with separate endpoint heads and explicit unsupported-family abstention. Online calibration may update only between prospectively separated episodes with versioned training inventories, never on evaluation feedback. Retain same-information augmented StatNN. L/R/K/S and K+F:F/K already isolate orientation/sensitivity; deletion needs its own rank/schema rule, not fabricated LoRA ranks (A5.4, F5.1–F5.2 and schema pins). Neural seed-split predictors still trail StatNN (`docs/reviews/P4/d3v2f-summary-2026-11-08.md`).

F6 withdrew g₀+GEA for chains: use exact predecessor gradients where registered, flag base-factor q, otherwise abstain or price extraction. A JEPAWG-style latent transition could encode accumulated state and nonlocal interactions, requiring held-out trajectory supervision; it cannot recover information discarded by the encoder. [IDM](https://arxiv.org/abs/2610.13991) forecasts agent modifications, not arbitrary weight deltas.

### 2. Forecast calibration

Propose F9-L for all future non-Part-A proposals: hash realised delta, receiver/predecessor, field/chart/model versions, endpoint orientation, a, per-tensor q/coverage, κ, predicted change/sign, abstention, source, timestamp and decision before scoring. At a=0 record κ undefined/infinite explicitly. Failed construction gets a row without delta. Extend F8.3 to H5, pruning and rejected attempts; predeclare rejection-audit sampling/probabilities to avoid acceptance-only calibration.

Report per-family/endpoint observed signed change y=α+βa: intercept/slope (ideal 0,1), mean signed residual y−a and MAE over **all** scored proposals, including unresolved effects. Use source/episode-cluster uncertainty; constant a or insufficient clusters makes inference undefined. Report resolution fraction and conditional sign agreement separately (A5.3). Signs are not probabilities: no Brier score without a frozen probability map. q is Fisher magnitude, not signed Hessian correction or confidence interval (`response_field.py`). Any fitted a/q calibration uses calibration-only rows and A5.4’s ≤3-coefficient descriptive allowance. These sampling/reporting rules exceed F8.3.

### 3. Pruning: representation and limits

**Yes, as separate F9-P after existing ledgers.** [Reasoning pruning](https://arxiv.org/abs/2604.25098) motivates unstructured tests on other receivers/generation endpoints. [Deletion Tuning](https://arxiv.org/abs/2611.10866) learns tiny gate masks; it does not validate large outcome-blind deletions. [OBC-Prune](https://arxiv.org/abs/2609.17890) reweights calibration activations using successful/unsuccessful rollouts for existing pruning algorithms. [Router sensitivity](https://arxiv.org/abs/2608.07890) concerns MoE experts. [PPF](https://arxiv.org/abs/2508.02381) predicts structured pruning policies; its reported error/speed is not a baseline estimate for arbitrary masks here.

Predeclare identical supported linear-weight inventory, per-row allocation, calibration IDs, seeds and tie-breaking for 10/30/50% deletion. Report eligible-weight and total-model sparsity separately; preserve unsupported tensors/embeddings. Three methods × three levels give nine pure-mask entries: Wanda, SparseGPT-derived masks transplanted onto original W, seeded random. Hold counts/support allocation equal and freeze masks without our endpoint selection. [Wanda](https://arxiv.org/abs/2306.11695) uses weight magnitude times input activation norm. G2 A’s diagonal supplies corresponding second moments, explicitly labelled with its panel weighting; sequential recalibration requires new forwards (`response_field.py`).

**SparseGPT caveat:** its [algorithm](https://arxiv.org/html/2301.00774v3) compensates surviving weights. Native D=−W⊙M+C, with C on survivors, so a=−Σ_M gᵢⱼWᵢⱼ+⟨g,C⟩. Add three native compensated entries; otherwise call the treatment “SparseGPT-mask transplantation.” Contract the complete D: q includes deletion–compensation cross terms.

For pure deletion a=−Σ_M gᵢⱼWᵢⱼ exactly contracts the frozen gradient. Per matrix,
q=½tr(GDADᵀ)=½Σᵢⱼₖₗ GᵢₖAⱼₗWᵢⱼWₖₗMᵢⱼMₖₗ.
Unlike additive a, q retains off-diagonal mask interactions. Summed per-tensor q omits cross-tensor coupling; unsupported curvature is missing, never zero (`response_field.py::contract`).

Can these rank methods? **Plausibly locally, not guaranteed at these sparsities.** Rank a for signed TF response, q separately as trust magnitude; an a+q NLL heuristic is descriptive only. Near-zero g ties masks; signed cancellation hides large opposing saliencies. Equal a/q masks remain indistinguishable despite higher-order differences. Factor truncation loses directions; simultaneous multilayer deletion changes activations. Sparsity percentage is not a small-step guarantee.

Freeze pairwise forecasts **within each sparsity**; compare observed paired WikiText differences with A5.3 window blocks, reporting ties/unresolved and MMLU descriptively. Compare field, frozen chart, StatNN, descriptor-only and delta-norm without refitting; unsupported levels abstain. Correlation across sparsities cannot establish mask ranking. Shared receiver/calibration/recipe dependence prevents treating twelve entries as twelve independent components.

**Chart:** PoutᵀDPin is only 4×4. Exported columns are elementwise products with projected predecessor/base/displacement, not raw blocks. At roots E=0, displacement products vanish and base/predecessor products duplicate; raw ⟨D,W⟩=−Σ_M W²=−‖D‖² supplies deleted energy. Projection-null directions vanish; small projected base coordinates attenuate others. Layer/role orientation can distinguish some energy-matched masks, but cannot encode full coordinate identity (`response_chart.py::contractions/to_columns`).

**StatNN:** sparsity, norm, moments, cosine and spectra distinguish distributions, without objective alignment. Mean pooling within type/depth buckets discards exact layer/coordinate alignment. Sparse masks need not be low-rank; matching summaries can hide different saliency. Implementation “stable_rank” uses retained top-eight energy, not full spectral energy (`src/wsp/features/stats.py`, `pool.py`). Equal-norm exact rank-one edits share spectra, but full StatNN can still distinguish moments/cosines; bf16 rounding can increase rank. Neither encoder can restore discarded information.

Scoring projection: twelve entries plus base ×1.04 min/60 ≈ **0.226 GPU-h**, using measured upper pilot entry time (`docs/RESULTS.md`, TF-v1 round 3); loading/requalification extra. Construction, full CPU spectra/contractions and SparseGPT integration remain unmeasured. Propose 0.5 GPU-h construction stop-and-reprice cap, not a runtime estimate. This separate panel omits rotations/amplitude ladders and makes no F8.4 four-component claim; full F8 packaging needs another explicit count/price. Unstructured zeros alone demonstrate no storage/latency reduction.

### 4. Localisation: saliency first

Require objective-labelled localisation evidence, not introspection. Frozen gradients permit CPU g, g⊙W deletion saliency and candidate-specific g⊙D by tensor/row/column. Cached down-projection inputs permit D x response norms. G2 discards item gradients and caches module inputs, not full causal traces (`response_field.py::_collect_objective`; `scripts/g2_fields.py::run_probe`).

Attribution patching needs clean/corrupted task pairs, intermediate activations, downstream metric gradients and patch sites; causal validation additionally needs intervention forwards. Decision/testimony similarity requires both tasks. These are new metered artefacts. [Inside](https://arxiv.org/abs/2610.07186) studies attribution similarity in trained-preference tasks; [multi-component tracing](https://arxiv.org/abs/2606.03085) explicitly intervenes. Neither makes aggregate saliency causal localisation.

[OWMI](https://arxiv.org/abs/2608.20569) finds chance verbal discrimination alongside informative activation probes, not universal impossibility. [Gate/router heads](https://arxiv.org/abs/2609.35108) separate detection from reporting/location. Log verbal localisation as treatment/audit outside critic inputs (A5.7). Randomise matched H5 episodes: saliency-only scripted proposals versus identical-access LLM proposals, plus LLM without saliency. Match support, scale, diagnostics and best-of-N opportunities. Predeclare incremental LLM value as a positive lower 95% episode-cluster bound on utility improvement at matched charged cost versus saliency-only; also report cost at matched utility. A wide interval is unresolved; attribution agreement is insufficient. This is an editor-value test, separate from critic calibration. Size from G5 variance, keeping connected episodes together; no power estimate exists yet (`docs/H5_DESIGN.md`).

### 5. Runtime LoRA

Register shadow feasibility, not deployment. [Agentic-TTT](https://arxiv.org/abs/2610.12002) motivates policy gating; [QueST](https://arxiv.org/abs/2605.13369) uses query-conditioned PEFT. Cheapest comparison constructs one fixed adapter stream, freezes forecasts, then replays critic versus evaluate-every-proposal with common truth labels and charged construction. The critic saves validation after construction, not construction itself. Separate audit label cost from projected operational savings. Retain A5.7: paired savings lower bound ≥30%, final-score difference lower bound ≥−1 point, missed-qualifying rejection upper bound ≤5%, and positive net GPU savings. Power/sample count comes from G5, not this small replay. Adapter construction/critic timings remain unmeasured (`docs/H5_DESIGN.md`; A5.7).

### 6. Math endpoint

Propose F8-E-only descriptive natural-completion and paired fixed-budget MATH with thinking-on/off base controls. Do not infer that all gains are repairs from TF evidence. Sampled math variability is documented (`docs/RESULTS.md`, P2 retest); qualify repeated-seed paired changes under A3-style reliability. Price = pilot + checkpoint-count × measured same-budget seconds/checkpoint /3600. No matching generation rate is established here: defer purchase pending pilot; retain TF-v1.

### 7. Exact paper sentences; unchanged decisions

“Structured pruning-policy surrogates motivate prospective forecasting but do not establish arbitrary-delta transfer.” (PPF above.)

“Objective saliency measures frozen receiver sensitivity; verbal localisation requires independent mechanistic validation.” (OWMI above.)

“Mask-only and compensated pruning are distinct edit families, and Fisher magnitude is not signed curvature.” (SparseGPT; `response_field.py`.)

Change no sentence of F5.5/FP6 or A5.7’s ≥30% savings, non-inferiority, rejection-safety and net-cost requirements. F9 cannot rescue funding (A5 F8.6; `docs/council/2026-11-11/SYNTHESIS.md`).

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---|---|
| F9-L log | Signed calibration | Proposed per-family α/β, residual MAE | 0 GPU for existing-field contraction; new predecessor gradients extra | Missing hash blocks scoring; degenerate fit undefined |
| F9-P masks | Alignment versus summaries | Proposed within-sparsity paired TF ranking | ≈0.226 scoring + overhead; construction cap 0.5 | Cap breach: reprice; unresolved: no expansion claim |
| H5 localisation | Saliency versus verbal selection | Proposed matched episode ablation | CPU reuse 0; new patching unpriced | No incremental value: no LLM-localisation claim |
| Runtime LoRA | Gate validation | Proposed shadow cost replay | Unpriced; meter before launch | Apply A5.7; no positive net saving: no deployment |
| F8-E generation | Termination/accuracy | Proposed repeatability qualification | Unpriced pilot | Reliability/budget failure: unresolved |
