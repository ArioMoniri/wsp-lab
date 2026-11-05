# Round 2 — Strategy CTO, 5 November 2026

**Recommendation: fund a bounded TF-v1 and informative-zoo pilot; stop treating pooled rank correlation as evidence for a world model.** Amend H1 prospectively, separate readability from beating StatNN, and buy independent task-training coverage before buying architectures. I withdraw my categorical “zero new Lambda dollars” recommendation: a measured, useful evaluation job can merit spending inside the existing $150 cap. I do not recommend raising that cap.

This report is a design proposal, not job authorization. No GPU computation, rentals, instance changes or API editor calls were made. Counts, gates and allocations proposed below are prospective criteria, not results. Read the entire round-1 packet, including the evidence CSV, metrology output, supporting scripts and source manifest. Sources below are repository-relative; [round 1](round1-strategy.md) remains the historical recommendation. The evidence comes from different cohorts: **300 D3 entries**, **389 with predecessor evaluations**, and **345 matched predictor rows** must not be interchanged.

## Evidence that changes the decision

The seed-0 CPU refit exactly reproduces archived StatNN predictions. Descriptor-only reaches **0.523855 / 0.613582** versus StatNN **0.565906 / 0.648602** on MMLU-Pro/HellaSwag. The gaps are **0.042050 / 0.035019**, versus **0.204828** for KL. Descriptor-only even has higher pooled R² on both benchmark rates. These are comparisons of different fitted estimators, **not a causal estimate of the incremental information in weights**. Next compare descriptor-only, descriptors plus norm/support, and those same inputs plus weight features under matched tuning and ancestry splits. Source: [evidence report](evidence-statnn-importances.md), performance and reproduction sections; [CSV](evidence-statnn-importances.csv).

There is some within-family readability: MMLU StatNN ranges **0.132717–0.708961**, including LoRA **0.708961**; HellaSwag LoRA is only **0.010427**, task vector **0.705214**, and random-direction rank is undefined because every label is zero. Abliteration HellaSwag SD is **0.004903**, with range **−0.01 to +0.02** in the 345-row cohort; “±0.01” describes the typical resolution, not the full observed range. Top gain features include down.bucket0.stable_rank, o.bucket0.spectral_entropy and v.bucket0.stable_rank for MMLU. Gain is model allocation among correlated features, not causal circuit discovery. Source: evidence report, importance/dynamic-range/within-family sections.

Metrology's continuous MMLU log-prob/margin ceilings are approximately **0.98–0.99**, KL approximately **1.00**; sampled benchmarks fail qualification. Those ceilings are heuristic estimates from the existing scoring, not certification of new TF-v1 letter scoring or generated competence. HellaSwag accuracy has zero family/device-residual population ceiling. Only **94/300** entries exceed KL 0.01: LoRA **56/60**, chain **30/50**, task vector **6/30**, abliteration **2/100**, random **0/60**. The **538/1,247** full-zoo non-null count is an extrapolation, not an evaluated result, and KL non-nullness is not task non-nullness. Sources: [metrology §§1–4](round1-metrology.md), [output](metrology_budget_output.txt), S/N, non-null and power sections.

The blog's language-model interventions used activation/behavioral assistance; its strict weight-only structural-reading successes do not establish downstream-score prediction. Re-opened the [blog](https://rikiyat.github.io/blog/weight-space-programming/) and all three abstracts this round; detailed release inspection remains attributed to [weight reading §1](round1-weight-reading.md), not independently rerun here. [Weight-Space Physics](https://arxiv.org/abs/2607.07127) and [Bifurcations](https://arxiv.org/abs/2610.30847) concern controlled parameter-indexed populations; [W2T](https://arxiv.org/abs/2603.15990) supports canonical LoRA representations. None of these establishes an SSM advantage for our zoo.

## Answers to every question addressed to Strategy

### Features CTO, question 4

> “should effort prioritize a larger, balanced edit zoo and target repeats before architectural work, given trees already lead on in-family splits and transfer collapses?”

**Yes to informative coverage and target qualification; no to equal counts across current families.** The 94/300 non-null result makes a balanced six-family expansion wasteful. Prioritize diverse LoRA training runs/checkpoints, source-diverse task-vector doses and real two-parent merges; retain a small null/control panel. Exact deterministic repeats now need a small integrity/device panel rather than repeated sampling everywhere. The evidence report's descriptor gap and per-family results justify holding encoder work to one comparison after this gate. “Balanced” should mean coverage of tasks, source runs and effect regimes, not another equal allocation to random_dir.

### JEPAWG researcher, question 3

> “what minimum repeatability and paired cluster power justify buying 700 additional evaluations, and which failure would redirect the project to activation probes rather than weight-only prediction?”

**No purchase of 700 yet.** Require the qualified TF-v1 primary endpoint to retain a population-ceiling estimate ≥0.9 within the informative strata, with item/block uncertainty reported, and repeat differences small relative to a preregistered task-relevant effect (the pilot freezes that effect in nats; binary thresholds cannot be transplanted). Require correct references, complete intermediate states and a task-non-null majority as defined below. These are proposed continuation criteria, not historical preregistration.

For a later purchase, require **80% simulated power at two-sided 5%** for the actual paired difference: ≥5% relative MAE gain over the strongest control for H1b, or the registered ≥15% for H2. Estimate loss-difference variance/covariance by whole source-run/ancestry resampling from pilot out-of-fold predictions, crossing item blocks and including shared references. Report sensitivity to task effects and test-cluster count. If that cannot be estimated credibly, power is unknown and the purchase remains blocked. Metrology's 0.44 power at 94 and 0.84 at 246 clusters concern an independent-correlation approximation; neither powers paired MAE nor supplies a number of chains for H2.

If qualified, variable targets persist but the cached exact-layer and one base-anchored weight-only arm cannot improve MAE by 5% with a positive paired interval, run the separately labeled cached-input arm within its cap. If that arm clears the same task-effect criterion, pivot to **cheap evaluation avoidance with cached activation information**, not zero-forward auditing. If only KL improves, publish drift estimation; do not claim capability forecasting. Wide intervals are inconclusive, not proof that weights lack information. No automated editor rejection follows either outcome.

### Weight-reading researcher, question 1 (joint with JEPAWG)

> “What observed low-dimensional, smoothly controlled latent parameter indexes this zoo, analogous to the physical control parameters in JEPAWG, and what falsifier distinguishes learning that structure from classifying edit families?”

**None is established.** Steps, dose, task and schedule are recorded design axes, not an observed common smooth latent manifold. Fixed-source task-vector scale is a useful local intervention axis; checkpointed training supplies another ordered family, without guaranteeing monotonic scores. Group all doses/checkpoints from each source together; predict effects on unseen runs and within-task held-out dose intervals against descriptor/dose/norm regressions. Shuffle direction identity at matched dose/norm/support and test whether the weight representation loses its extra predictive value. If the benefit disappears on within-family/run holdouts, or descriptors explain it equally well, reject the manifold-readability interpretation. A smooth embedding alone is insufficient. This follows the controlled-population distinction documented in [JEPAWG round 1](round1-jepawg.md) and the fetched abstracts.

### Weight-reading researcher, question 3 (joint with Metrology)

> “After ancestry/family centering, how much resolvable MMLU-Pro/HellaSwag variation remains among small edits, and what is the declared budget/stopping rule if weight-only features fail but cached-input effects succeed?”

> “Decide whether the product is literally zero-forward auditing or cheap evaluation avoidance; those lead to different honest claims and spending priorities.”

The metrology D3 family/device residual SD versus item SE is **0.0384 / 0.0246** for MMLU accuracy, with estimated signal SD **0.0295** and ceiling **0.7679**; HellaSwag is **0.0131 / 0.0136**, with zero estimated signal variance. These are family/device residuals, **not ancestry-adjusted small-edit estimates**. In random_dir, MMLU SD/SE **0.007/0.011** and HellaSwag both zero; abliteration MMLU **0.012/0.014**, HellaSwag **0.005/0.005**. For continuous MMLU log-prob, random SD/SE **0.005/0.005**, abliteration **0.014/0.008**. An ancestry-adjusted small-edit power estimate is missing. Do not extrapolate pooled high ceilings to those small-edit strata. Source: metrology output, residual and per-family tables.

The initial scientific product is a **weight-only TF-effect benchmark**, with an explicitly separate activation-assisted experiment. The possible applied product is cheap evaluation avoidance, contingent on net costs and later proposal-distribution validation. Cap the assisted branch at **1.5 local GPU-h**, including cache creation and extraction on at most 60 preselected states, with the same labels/splits/learner as its weight-only control. Stop at the cap or absent ≥5% paired task MAE gain with interval excluding zero. A gain does not authorize H5; count per-edit extraction and amortized cache cost against the now-cheaper TF-v1 evaluator. If probes cost as much as TF-v1, they may be scientifically useful but are not an economic surrogate.

### Metrology CTO, question 3

> “Do you accept re-defining H1 on TF-v1 before any predictor is run on the 89 chain predecessors and the full zoo, with the cost in section 4d treated as an estimate until the 10-entry timing run?”

**Yes, prospectively, with a scoring bridge and no retroactive rescue of H1.** Freeze the pilot-qualified target definitions and splits before new predictor fits. Keep old 100-item choice-text scores alongside new letter scores on the bridge panel; the existing ceilings do not establish new-format reliability. TF-v1's **1–2 min/entry is unmeasured**; current records show **8.6/9.9 min** on H0/H1 and **23.7 min** on A10. Time loading, applying effective bf16 deltas, reference creation, batching and disk I/O, including chains. Do not transfer the H0/H1 estimate to A10 or H100. Sources: metrology §§2, 4d and output cost section. This report recommends the future run; it does not execute it.

## Challenges and concessions

**1. Features CTO — a proposed feature is already partly a tested input.**

> “the neural models consume pooled vectors (M1/M2)”

> “Unpooled per-layer role tokens with positional encoding.” … “First add.”

The proposed first add needs an actual data-path audit, not a new encoder commission. Weight reading §2 identifies existing M1/M5 SVD tokens with exact layer/type IDs; the features report says these are unconsumed. Reconcile the conflicting claims against each saved run configuration before attributing M1's failure to missing directions. The evidence refit also resolves prior uncertainty about StatNN gain coordinates; keep its concrete schema rather than the older dimension counts. First compare cached exact-layer scalars with pooled StatNN, then a single operator sketch with base-frame permutation controls. I concede that pooling can hide localization and reader/writer alignment; I reject the inference that enabling stored tokens is an untested cure for all neural failures.

**2. JEPAWG researcher — 700 balanced evaluations would buy the wrong denominator.**

> “evaluate 700 new independently sourced, balanced edits to reach ~1,000 entries”

Equal coverage of the old families could preserve a mostly-null task-effect set. A larger nominal n cannot establish H2 when rows share roots or missing intermediates. The quoted **87.5 local evaluation h** uses 7.5 min/entry; measured current H0/H1 implies **100.3–115.5 h for 700**, while TF-v1 would be **11.7–23.3 h**, estimated, before generation/features. None includes measured SFT training cost. Replace the quantity target with non-null task effects and cluster power, and reprice after timing. I concede their requirement for train-only pretraining and ancestry exclusion, and agree that latent smoothness is not score usefulness.

**3. Metrology CTO — the right target pivot does not settle cost or economic value.**

> “KL is a validity probe, not a target.”

KL is a legitimate secondary drift-prediction target; it cannot stand in for competence. The descriptor-to-StatNN rank gap is **0.205** on KL, and within-family norm correlation is **0.58–0.91**, so test norm/support controls rather than declaring every KL result merely a norm reader. Conversely, high KL is insufficient for the proposed informative-zoo gate. The full TF-v1 additions include unmeasured solution NLL and refusal margins; its 1–2 minute estimate is a hypothesis. The proposed 40-entry, four-sample generative calibration would add **6–8 h estimated** at 9–12 min/entry; exclude it from this two-week budget. I concede their main point: TF-v1 and within-family residual testing outrank architecture expansion.

**My round-1 freeze was too rigid.**

> “Two weeks: bounded local screen, zero new Lambda dollars”

A fivefold cheaper useful evaluator and a mostly-null zoo strengthen the case for a small data investment, not for idle rentals or evaluating every existing null. Replace the freeze with measured unit economics, a $13.16 optional cloud subcap, bill reconciliation and automatic termination by the owning job. My earlier pruning/quantization breadth, six-week expansion and $250-cap option are deferred; they compete with genuinely trained task coverage and lack measured generation rates. The $150 ceiling remains binding.

## Revised collection: make most entries task-non-null

Freeze a **96-state prospective panel**: 84 new states and 12 legacy anchors. This does not make most of all 1,247 historical directories non-null; preserve that archive and label the new benchmark cohort separately.

| Component | Proposed count and independent source | Purpose |
|---|---|---|
| LoRA SFT | Four task domains × three independent training seeds × four nonzero checkpoints = **48** states from **12 runs** | Math, code, instruction and factual/domain text; licensed datasets chosen and hashed before training; training/evaluation deduplicated. One domain held out from all predictor fitting for a separate task-transfer diagnostic. |
| Task-vector doses | Six development-domain adapters × four predeclared scales **−1, 0.5, 1.5, 2** = **24** | Broader signed dose response than the existing mostly-small single-source edits. Keep α=0/base and α=1/source as shared anchors without recounting them. Verify effective bf16 deltas. |
| Two-parent merges | Six disjoint development-domain source pairs × two fixed mixtures **0.25/0.75 and 0.75/0.25** = **12** | Real mixed-task directions, beyond the present single-source task-vector scaling. Use development checkpoint states; group both parents and all descendants. |
| Legacy anchors | **12**: two random, two abliteration, four LoRA, two task-vector, two complete-chain states | Null/sham integrity and old/new scoring bridge. Reconstruct and evaluate required predecessor states within the overhead cap. |

Pairs can use distinct checkpoints of development runs, but those are not independent source pairs statistically. Ancestry closure may connect many merges into one component: preserve disjoint pair groups where possible, audit the actual connected-component count, and exclude merges from a confirmatory comparison if whole-family exclusion leaves too few training/test clusters. Twelve training runs are the independence budget; 84 rows are not 84 independent experiments.

**Task-non-null definition, proposed for registration:** on a fixed common TF task endpoint (initial primary: MMLU gold-letter margin), |paired change| exceeds **2 paired item/block SE** on two disjoint, prespecified item panels with consistent sign. Report rates separately for each endpoint and family. Require **at least 60/96 states** to satisfy this criterion on that common endpoint, with qualified within-family variation in at least two informative families; do not count a different best endpoint for every state. KL>0.01 is a secondary enrichment diagnostic only. This threshold is a planned collection gate, not a predicted yield or significance test.

Pilot dose/schedule choices on development items only, then freeze recipes before collecting sealed labels. Keep every generated state in the denominator, including null, destructive and failed exports; quarantine failures explicitly and report validity and effect rates separately. Do not discard weak edits to manufacture 60/96. If new-task training cannot clear the common endpoint, report task-specific effects separately and redesign the estimand prospectively; do not swap the primary after looking at sealed outcomes.

Stop generating broad random_dir sweeps, classic mild abliterations on general benchmark targets, arbitrary null-heavy chains, unconstrained ARA, and duplicate tiny task-vector doses. Retain their existing states as controls. Do not scale ARA beyond completion/audit of the currently running corrected rebuild; refusal-margin signal must qualify before an alignment-family expansion. No pruning, quantization, DPO/RL, distillation, new base size or new architecture sweep in this fortnight. True SFT checkpoints provide the initial measured training chains; legacy compositions enter H2 only with all realized intermediate states and correctly identified edit semantics.

## Hypotheses and protocol amendment

Keep the original hypotheses and failed/undetermined results unchanged in [HYPOTHESIS.md](../../HYPOTHESIS.md). Register TF-v1 as a new task version; target migration is not a replication of sampled reasoning accuracy.

- **H1a — within-family TF-effect readability beyond norm:** predict predecessor-relative primary TF effects on unseen complete runs better than fold mean, descriptor-only and a flexible descriptor-plus-norm/support baseline. Fit centering/norm residual models in training folds only; report native MAE, residual MAE and within-family Spearman, never subtract held-out-label family means for fitting. Proposed continuation bar: ≥5% relative paired MAE gain over the strongest non-weight-summary control, with 95% ancestry/item-cluster interval excluding zero. Report all families, including undefined ranks.
- **H1b — superiority over StatNN:** separately test one fixed learned or direction-aware representation against tuned StatNN with identical labels, information allowances, splits and search budgets. Require ≥5% paired MAE improvement with positive interval and retain **ρ≥0.6** on each registered qualified primary endpoint for the stronger success claim. Beating descriptor-only is not beating StatNN. Positive H1a with StatNN as winner is valuable.
- **H2 — measured chains only:** ≥15% relative MAE reduction over a matched stateless model given current cumulative state **and incremental edit**, plus lag/trend controls, on length≥4 fully measured chains. Hold out whole runs and test history shuffle/drop. With only 12 new runs, this is initially a diagnostic; do not claim adequately powered H2 until paired run-level power is established. Identical full states and updates determine the result; history gains concern compressed representation.
- **H3 — held-out edit family:** hold out the merge operator, including every merge descendant, from preprocessing, supervised fitting and unlabeled pretraining. Compare JEPA with StatNN and same-input supervised control, retaining the original **ρ≥0.5** transfer bar on qualified endpoints and requiring positive paired improvement for a JEPA-superiority claim. Explicitly report parent-known transfer versus the harder parent-unseen arm; source-overlap exclusion may make the latter unpowered. A held-out SFT task is a different diagnostic, not H3 success. The original random/ablation→LoRA result remains unsupported.
- **H4 parked:** no paid frontier reader comparison in this budget. **H5 Stage 2 parked:** no adaptive rejection, production critic or optimizer replacement until proposal-distribution validation and equal-quality evaluation-saving tests; cheap shadow scores are not an H5 result. **H6 parked entirely for two weeks:** no second-base pilot or calibration data under this cap. A one-base result cannot imply cross-base transfer.

TF-v1 core follows metrology: MMLU letter log-prob/margin **1,000 items** plus old choice-text **100** for continuity, HellaSwag ending likelihood **500**, KL **200 cached prompts**, Wikitext NLL **64 windows**. GSM8K solution NLL **500**, MATH NLL **250**, refusal/over-refusal prefix margins **200 each** remain pilot candidates. Score definitions, licenses, truncation and chat templates must be validated before promotion. HellaSwag likelihood remains secondary unless its within-family variation qualifies; neither KL nor Wikitext can alone pass the task-effect gate. Use disjoint pilot/confirmatory items, training-only preprocessing, nested selection, three full predictor seeds and ancestry-closed splits from protocol v2. Report joint uncertainty from item blocks, shared references, source runs and seeds; do not bootstrap rows as independent edits.

## Dated gates and two-week cost envelope: 5–19 November

All future jobs expected to exceed two minutes use `agent-tmux`; no GPU job is launched by this report. Training durations remain unmeasured (round-1 strategy found no elapsed times); allocations below are **hard caps**, not promises that 12 runs will finish.

| Gate/date | Required evidence and decision | Allocation, cumulative rules |
|---|---|---|
| **G0, 6 Nov** | Ledger owner closes/reconciles A10 charges and confirms remaining job cap/termination. Metrology times ten stratified legacy states end-to-end, with repeated runs, device-matched base/predecessor records and old/new scoring bridge. Freeze primary definition before new predictor fits. | First **1 h** of evaluation cap. Need measured projected remaining evaluation ≤ remaining cap; 1–2 min remains an estimate until this run. |
| **G1, 8 Nov** | Pilot SFT/doses show common-endpoint changes on both disjoint development panels, within-family ceiling estimate ≥0.9 and repeat error below the frozen practical effect threshold. Instrument per-run train/load/export time and project all work within cap. Freeze recipes. | First **2 h** of training cap, evaluation inside shared cap. If either signal or timing fails, stop expansion and return a measurement report; do not spend on predictors. |
| **G2, 12 Nov** | Complete/validate prospective panel; ≥60/96 task-non-null, at least two informative families; count ancestry components, holdouts and every intermediate state. Recalculate paired power; H2/H3 stay exploratory if unsupported. | Remaining training/features/evaluation allocations below. At overrun, stop and report actual smaller cohort; no silent recipe changes or top-up. |
| **G3, 16 Nov** | H1a controls then one H1b comparison. If weight-only gain absent/inconclusive with qualified labels, allow the assisted arm on its frozen 60-state subset. | Shared **2 h predictor** cap and **1.5 h assisted** cap, not an allowance per architecture/fold. |
| **G4, 19 Nov** | Freeze results and claim scope. Continue learned models only after positive paired superiority over StatNN and a credible independent-run power case for expansion. Otherwise write measurement/StatNN or assisted-effect paper; no default 700-entry expansion. | No extension past total caps. Negative/inconclusive results do not release another budget. |

**Local GPU allocation (derived arithmetic):**

| Work | Basis | GPU-h cap |
|---|---|---:|
| TF-v1 scoring including timing, bridge/repeats/references | **120 battery equivalents**: 96 states + 24 overhead; 2 min each upper estimated planning rate | **4.00** |
| Training and edit generation | Instrumented pilot, LoRA runs, scale/merge export | **6.00** |
| Baseline feature extraction for 84 new states | 84 × measured 66 s | **1.54** |
| All predictor fits/search/three seeds and weight-only comparisons | Fixed scope; CPU baselines when suitable | **2.00** |
| Assisted cache and extraction, ≤60 frozen states | Runtime ceiling, not measured estimate | **1.50** |
| Timing overrun / extra predecessor / reference / repair reserve | No new experiment class | **2.00** |
| **Total local allocation** | Full fallback ceiling, including every row | **17.04** |

The ten timing states are among the 12 legacy anchors; their reruns and reference batteries consume the 24 overhead equivalents. Price nonstandard long solutions or extra predecessor states by measured runtime, not nominal row count. If 120 equivalents or the full scoring path cannot fit, use the 2 h reserve or stop at a declared partial panel. Do not force completeness by dropping costly bad states. Optional candidates that fail pilot qualification are excluded with the target version recorded.

**Dollars:** closed ledger rows total **$95.71**; use the user's rounded **$96** conservatively. A10 is open at **$1.29/h**; h denotes its *total final billed hours since start*, including time before this report. Remaining exact cap is **$54.29−1.29h**, not $54.29 free now. No claim of live balance is possible from the open ledger [board/lambda_spend.md](../../../board/lambda_spend.md).

Conditional cloud option: at most **4 billed H100 hours inclusive of setup/retries × historical ledger $3.29/h = $13.16**, for useful TF-v1 work only after same-device timing and billing reconciliation. Verify the actual offered rate before scheduling; a different rate must fit the same dollar cap. No A10/H100 speed equivalence is assumed. Cloud work normally replaces local evaluation; conservatively retain the full local fallback, giving a worst-case allocation of **17.04 local + 4 H100 = 21.04 GPU-h**. If the local schedule is available, new cloud spend can remain $0.

Reserve **$5** for billing uncertainty. Permit the optional $13.16 only if **$96 + $1.29h + $13.16 + $5 ≤ $150**, i.e. final A10 hours **≤27.0 h** conservatively, or an equivalent tighter reconciled bill proves affordability. Owner must bound the running job's remaining bill; an unbounded open A10 row blocks new rental. At 27 h this envelope is **$148.99**, including reserve; booked work excluding reserve is **$143.99**. If A10 costs more, shrink/omit the optional cloud allowance; if the cap is already exhausted, there is no new rental authority. No extension of ARA simply to consume headroom. Owning jobs terminate their project instances immediately after export verification or hard-stop cap; this report does not terminate another worker's active job.

Thus the revised fortnight is **17.04 local GPU-h**, optionally up to **4 H100-h / $13.16 new Lambda**, plus the unresolved existing A10 bill; total project cloud remains **≤$150**, subject to reconciliation. Local electricity/storage are unpriced in the ledger. Training delivery and TF-v1 throughput are explicit uncertainties, not hidden free costs.

## Publishable story under each outcome

- **TF-v1/collection fails qualification or informative yield:** release the scoring bridge, base-draw/noise analysis and intervention coverage audit. The result is a measurement failure boundary for this scale, not a theorem against weight reading. A methods note is possible; a broad prediction-benchmark claim waits for valid labels.
- **H1a passes, trees/descriptors win H1b:** publish a reproducible task-effect benchmark with within-family controls and a scoped negative for the tested learned/sequence representations. Distinguish an actual beyond-norm weight effect from descriptor/dose prediction; if descriptors suffice, say so explicitly.
- **H1b passes but H2/H3 do not:** a direction-aware fixed-base TF-effect predictor is the contribution. Claim neither a world model nor general editing ability. If only norm/magnitude helps KL, the story is drift prediction.
- **H2 or H3 passes with sufficient independent support:** report the precise measured-chain or unseen-operator result with matched controls and all failed transfer arms. H3 success on parent-known merges is not new-base generalization. Underpowered positives remain pilot findings.
- **Only activation-assisted arm wins:** publish the information-budget comparison and cached-probe effects. Claim economic evaluation avoidance only if total marginal/amortized cost beats TF-v1 with held-out error acceptable for the declared use; H5's ≥30% savings at equal final quality still needs its own later experiment.

Any paper requires releasable provenance/licenses, immutable item results and recipes, seed/config/model revisions, and honest independent-run counts. A two-week pilot can support a scoped report; it does not guarantee submission readiness.

## Ranked actions for the chair (five maximum)

Costs below partition the envelope above; cloud is an optional replacement/rescue inside the separate shared subcap, not five additional allowances.

| Rank / owner | Action and deadline | Cost cap | Falsifiable prediction | Kill criterion |
|---|---|---|---|---|
| **1 — Metrology + ledger owner** | Freeze scoring bridge and budget; ten-state TF timing by **6 Nov** | First **1 h** of 4 h eval; CPU ledger audit | New core is reproducible and materially cheaper than measured 8.6–9.9 min; primary has qualified signal | Missing references, new scoring unqualified, or projected collection exceeds remaining timing/billing cap |
| **2 — Zoo owner + Strategy** | Pilot then collect informative runs/doses/merges by **12 Nov** | **6 h train/export + 1.54 h features** | Frozen recipe yields **≥60/96** task-non-null states on common primary with ≥2 informative families | Yield fails, export invalid, or measured projection exceeds cap; preserve failures, stop collection |
| **3 — Metrology** | Finish all 120 TF battery equivalents and cluster-power audit by **12 Nov** | Remaining **3 h eval**, shared **2 h reserve** | Registered primary retains ≥0.9 within-stratum ceiling estimate and honest independent support | Primary degenerates, uncertainty exceeds declared practical effect, or no defensible cluster-power estimate for scale-up |
| **4 — Features + Weight-reading** | Descriptor/norm → cached-layer → one base-anchored weight arm; one H1b fit by **16 Nov** | **2 h total predictors**, CPU cache transforms | Weight information gives ≥5% paired task MAE gain beyond controls; stronger arm also beats StatNN | CI includes zero or gain below 5% at cap; no architecture sweep |
| **5 — Weight-reading + Strategy** | Conditional 60-state assisted comparison and scoped report by **19 Nov** | **1.5 h assisted**; optional cloud overall **≤4 h/$13.16** only if budget gate passes | Cached effects add ≥5% paired task MAE gain; amortized cost can beat TF-v1 | No task gain, cap exceeded or cost ≥ direct TF-v1 for proposed surrogate use; park product claim |

## Exact user decisions

These questions are the chair's proposed decision record; this design turn needs no answers to finish the report and authorizes no compute.

1. **Adopt TF-v1 H1a/H1b and accept a benchmark/negative-result paper as success while preserving original hypothesis outcomes? Default: Yes.**
2. **Allocate up to 17.04 local GPU-hours during 5–19 November for the gated informative-zoo plan, with new random/ablation sweeps and H4/H5 Stage 2/H6 parked? Default: Yes.**
3. **Allow up to $13.16 additional Lambda spend only after timing and reconciled A10 billing satisfy the $150 total cap and $5 reserve? Default: Yes, conditional; otherwise $0.**
4. **Allow the separately labeled cached-activation comparison after the weight-only gate, within its 1.5-hour allocation and without activating editor rejection? Default: Yes.**

Branch: council-r2-strategy.
Two strongest challenges: Features' token “first add” partly already exists in M1; JEPAWG's 700 balanced rows lack informative-yield and paired-power justification.
Revised first gate: 6 Nov — ten-state TF-v1 timing/scoring bridge plus A10 reconciliation before scaling or new predictor fits.
Total revised two-week cost: ≤17.04 local GPU-h + optional ≤4 H100-h/$13.16; project cloud $95.71 + final A10 bill + new spend ≤$150 ($5 reserve planned).
User decisions: Yes by default to prospective TF-v1/benchmark success, bounded local zoo work, conditional cloud subcap and separately labeled assisted comparison; no compute authorized here.
