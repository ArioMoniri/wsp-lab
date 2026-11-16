1. Develop one Qwen3.5-9B response model, conditional on a measured feasibility/headroom pilot.
2. Predict full trajectories from base information; intermediate target execution breaks the forward-free claim.
3. Build signed dose ladders and genuine noncommuting compositions, not just correlated checkpoint rows.
4. Prioritize benign capability steering and small task vectors; retain WSR operations as controls.
5. Count source-closure components, never seeds alone, as independent units.
6. Optimize verified capability under explicit noninferiority constraints; loss improvement alone is insufficient.
7. Compare world-model planning with static predictors, random search, best-of-N steering and LoRA SFT.
8. Match real-evaluation budgets and separately charge construction, training, tokens and bank amortization.
9. Freeze one winner per arm before sealed testing; distinguish enhancement from planner superiority.
10. Register WM-E separately; Part A stays held, and A5/F5.5/F8/F9 remain unchanged.

# Editor — round 8, 16 November 2026

**Status.** Research recommendations only. All new counts, doses, margins and caps below are proposed design choices, not measured results. Read `BRIEF-r8.md`, `SYNTHESIS-r7.md`, `round7-editor.md` and `docs/PREREG_H1_TF.md` (A5), especially F5.5/F8/F9. The [MSRM report](../../reviews/msrm/msrm1-dev-v0-2026-11-16.md) gives C=0.378 against F3u, description-only C=0.037, 123 rows but 29 closure groups, and k2b firing. That supports testing learned residuals, not assuming field superiority or successful dynamics.

## Evidence and target

The [Qwen card](https://huggingface.co/Qwen/Qwen3.5-9B) specifies a 9B language model with vision encoder, 32 layers, and interleaved Gated DeltaNet/attention blocks. Restrict edits to audited text-model writers; do not assume ordinary attention tooling covers every block. Keep the instruction checkpoint, revision, tokenizer, thinking mode and decoding fixed. Base-checkpoint SAEs require validation before use on this checkpoint.

Choose 9B for direct [WSR](https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models) comparability, but reject MATH-500 as the automatic primary objective. [Exploring WSP](https://arxiv.org/html/2611.09382v1), Table 2, reports a 9B math development gain of 29.69 percentage points but held-out −0.35; the original checkpoint was retained. Its IFEval/SciBench improvements motivate testing headroom, not assuming transfer. Prefer a prespecified difficult math/science task with construction/dev/test partitions; use MATH-500 descriptively.

[Sparks](https://rikiyat.github.io/blog/weight-space-programming/) used practice-problem feedback for its Qwen enhancement, and distinguishes releasing existing ability through loop repair from adding mathematical ability. [Task arithmetic](https://arxiv.org/abs/2212.04089) motivates signed fine-tuning deltas; [contrastive weight steering](https://arxiv.org/abs/2511.05408) uses opposing fine-tunes; [TFTV](https://arxiv.org/abs/2609.09054) maps activation directions into rank-one weight edits using forward statistics. These are candidate generators, not evidence that our capability objective will improve. [Steering persistence](https://arxiv.org/abs/2608.24988) warns that behavior can recover despite persistent weight edits: norm similarity is not a behavioral state.

Fetch audit: all linked landing pages and code fetched successfully today. The [WSR PDF URL](https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models.pdf) returned “not accessible via this tool”; use its landing page and released code, without claiming a fresh full-PDF audit.

## Q3: bank and trajectory semantics

The released [operations](https://raw.githubusercontent.com/suayptalha/wsr/main/wsr/ops.py) implement scaling, row scaling, projection removal, norm-matched noise, norm gains and replacement deltas. Its [general pools](https://raw.githubusercontent.com/suayptalha/wsr/main/wsr/pools.py) and [paper pools](https://raw.githubusercontent.com/suayptalha/wsr/main/wsr/paper/pools.py) select subtle-dose parents using prior behavioral changes. Reproduce that only in a separately labelled WSR replication; our predictive bank must freeze parents before outcomes.

Proposed development bank: **96 root recipes**, 16 each in six families:

- Block attenuation/amplification on supported MLP/attention writers.
- Vocabulary-row scaling, with decoded tokens and tied-weight checks.
- Projection removal from unembedding or benign activation contrasts, plus matched random directions.
- Seeded Gaussian noise matched on support and realized Frobenius norm.
- Benign difference-of-means steering compiled into selected writer columns/rank-one updates.
- Small LoRA SFT task deltas from construction-only examples; preserve provenance of each training run.

For each root define a family-specific nominal magnitude from a separate pilot, then two signed branches at cumulative doses ±{0.25,0.5,1,2}: eight states/root, **768 states**. For scaling use W(s)=[1+s(f−1)]W₀; for additive edits W(s)=W₀+sD. Construct each ladder endpoint from W₀, recording actual transition deltas; do not accidentally multiply successive absolute doses. Projection coefficients and total displacement receive pilot-frozen caps. Negative removal means amplification and must be labelled accordingly. Exact identity, rounded-away edits and tiny unresolved changes remain distinct.

Add **24 paired-order composition bundles**, AB versus BA, extending each order to four steps: at most **192 state visits**, hence ≤960 nonbase visits before deduplication. Include overlapping projection removals and steering followed by writer scaling; disjoint fixed additive edits commute and belong as negative controls. Record A+B versus sequential application, inverse/recovery tests and each bf16 materialization. Compare endpoint effects with additive single-edit forecasts to expose interactions.

Group every shared direction, training corpus subset, parent, ancestor and identical realized delta by transitive closure. Different random seeds over the same direction are not independent. Partition components before composition so compositions cannot bridge train/test; family-transfer tests exclude mixed-family relatives. The 96 roots are a workload proposal, not 96 guaranteed independent units or a powered confirmatory N.

Measure continuous general-text ΔNLL and task answer likelihood on every unique state, plus multiple-choice accuracy where valid. Generation correctness, completion, length and IFEval need a frozen stratified subset of whole trajectories, including failures. Do not use gold-answer likelihood as proof of better generated reasoning. Repeat a pilot subset to estimate uncertainty; reuse neither 0.8B timing nor its NLL floor for 9B.

Prediction means frozen base weights/features plus action recipes, CPU algebra allowed, **no intermediate forward/backward passes, activations or observed scores**. Re-linearized field is an explicitly execution-assisted, charged comparator. Evaluate horizon 1–4 cumulative and incremental errors against fixed field, summed static MSRM, cumulative-delta StatNN, description-only and identity predictors. Rollouts must beat a static predictor given the full cumulative edit; pure dose extrapolation is insufficient. Report component-level intervals and order sensitivity. If effects fall below pilot noise or dynamics cannot beat static prediction, stop the rollout claim.

## Q4: enhancement experiment

Use a frozen library of benign steering/task directions, supported layers and doses; beam-search sequences of ≤4 actions. Primary search excludes refusal-removal and vocabulary shortcuts; these remain stress/control families. Limit total realized displacement and permitted tensors, retaining base as an available decision.

The planner maximizes predicted task correctness minus a calibrated uncertainty penalty, subject to predicted guardrail feasibility. Freeze penalty and uncertainty calibration on development data. Predictions only nominate candidates: actual development evaluation determines acceptance. Proposed constraints are upper confidence bound on general-text ΔNLL ≤0.01 nats/token and lower confidence bounds on capability/IFEval/appropriate-refusal changes ≥−1 percentage point. Measure harmful compliance and benign over-refusal separately. These margins require pilot precision and scientific review before registration; wide intervals mean unresolved feasibility, not a pass.

Allocate **32 identical real-evaluation bundles per arm per development episode**, with common items, decoding and paired seeds; report incumbents at budgets 1/4/8/16/32. A bundle includes the task and guard panels, and counts every trial, rejection and intermediate state evaluated. Arms:

1. World-model rollout ranking.
2. Random selection from the identical frozen candidate pool.
3. Field ranking.
4. Same-information StatNN/static residual ranking.
5. WSR-style reader ranking.
6. Best-of-32 directly evaluated steering, tuning layers and doses without a learned dynamics model.
7. LoRA SFT: construction-only training, development selection among rank/step/learning-rate settings, at most 32 checkpoint evaluations.

Give LoRA the same training examples used to construct task vectors; charge every training run. Also compare world-model beam search with greedy one-step ranking using the same predictor, isolating planning from predictor quality. Equal evaluation count does not mean equal cost: report generated tokens, scoring tokens, training/feature GPU-hours, reader tokens and bank cost both unamortized and amortized across episodes. No universal savings claim if the bank costs more than it saves.

Use disjoint construction, predictor-validation, search-development and final test items; deduplicate related problems across partitions. Freeze predictors, search rules, guard margins, candidate libraries and test sizes before confirmation. Each arm chooses one checkpoint using development only, then tests once. Primary enhancement requires paired 95% lower bound on correctness gain >0 and simultaneous guardrail noninferiority. Failure to detect degradation is insufficient. Planner superiority additionally requires a prespecified paired advantage over the strongest static search baseline across fresh episodes; competitiveness against LoRA/steering must be reported even if unfavorable. Use hierarchical episode/item inference and outcome-blind power simulation; neither 32 trials nor 96 roots supplies power automatically.

## Pilot, WSR and governance

First build CPU tests for dtype-realized deltas, compact-row reconstruction, identity, composition order, inverse drift, closure splitting and forecast-before-score enforcement. Then propose a separately authorized **8 GPU-hour pilot cap**: two roots/family, a composition subset, repeated evaluations, short-context gradients and generation throughput. A nominal 9B bf16 model is about 18 GB of weights (calculation); this is not a gradient-memory estimate. Stream shards, retain one model, collect writer gradients/K-FAC layerwise, truncate/offload factors, and measure both host/GPU peaks. Switch to 4B only by a documented pre-confirmation target decision, discarding cross-model pooling.

Price the bank as H≈(960t_likelihood+N_generation·t_generation+T_features+T_training)/3600, with seconds measured by the pilot and deduplicated counts. Start with Transformers; add a separate vLLM environment only if end-to-end generation including edit reloads measurably benefits. Run long future jobs through `agent-tmux`.

For WSR use identical realized edits, labels, token decoding and permitted statistics. The [harness](https://raw.githubusercontent.com/suayptalha/wsr/main/wsr/harness.py) offers tool/direct/constant/random conditions, not a learned scripted-statistics comparator. Add one. Propose 20 cases × three reader conditions (text, weights, weights+world-model) × 4,000 total billed tokens = **240,000-token cap**. Report world-model training access separately; it is not zero-shot weight-only replication.

WM-E needs its own dated preregistration before confirmatory outcomes. A5 F5.5 governs the 0.8B conjunction; F8/F9 are descriptive and retain Part A→F8-E→F8-F→F9-P ordering. This new study neither releases held Part A nor supplies missing predicates, funding fallback, or H5 success. Keep independent manifests and fresh panels; do not repurpose held outcomes.

| proposal | mechanism | test | cost | kill criterion |
|---|---|---|---|---|
| Dose/composition bank | Realized actions and source closure | Identity/order/interaction and noise qualification | ≤960 visits; pilot-priced | Mostly unresolved effects or closure leakage |
| Rollout model | Learned state transitions | Frozen horizon errors versus cumulative static predictor | Bank plus metered fitting | No incremental predictive value |
| Enhancement | Constrained candidate planning | Sealed gain and simultaneous guardrails | 32 bundles/arm/episode | No verified gain or guard breach |
| Strong alternatives | Direct steering and LoRA SFT | Equal evaluation budget; full cost ledger | Training charged separately | Planner advantage disappears |
| Separate WM-E governance | Fresh registration and panels | Outcome-blind power/coverage simulation | Proposed 8 GPU-hour pilot cap | Infeasible precision, memory or throughput |
