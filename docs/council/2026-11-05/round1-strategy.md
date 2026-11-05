# Round 1 — Strategy CTO, 5 November 2026

**Verdict: re-scope to a measurement-first weight-edit benchmark. Stop architecture expansion until real training trajectories produce qualified, variable targets.** StatNN winning is a useful result; it is not a reason to keep changing architectures until something wins. The current evidence does not establish that weight-only prediction is impossible, nor that a dedicated world model is necessary.

This is a design recommendation, not authorization to run jobs. No GPU work, cloud operations or commercial editor calls were performed. Proposed counts, runtime caps and decision thresholds below are planning criteria, not observations. All paths are repository-relative. Read the council brief first, followed by the requested hypotheses, roadmap, candidate/H6/editor assessments, identity/config, results, Board A D3 review, H5 protocol, spend ledger and lessons.

## 1. What zoo do we actually have?

I computed the following with `. scripts/env.sh` and `python docs/council/2026-11-05/strategy_audit.py` (CPU). Audit output and input hashes: `data/runs/council/r1-strategy-20261105/{audit.txt,manifest.json}`. The script reads descriptors and saved out-of-fold predictions, never model weights or test examples for fitting. Live data are shared across worktrees; this is the manifest snapshot with SHA-256 `6175c3728f03859f54c373bbec79f49564df4a10a3db94ba6db7fd366d986bcd`.

### Counts, norms and diversity

The manifest has **1,255 lines: eight headers and 1,247 unique edit entries**, rather than the brief's 1,235. There are **1,205 production entries**, 12 smoke entries and 30 without manifest eligibility. Production excludes every current Heretic row. This snapshot therefore does not yet contain a qualified six-family zoo; do not count the running correction as completed evidence.

| Family | All entries | Production | Production effective-delta Frobenius norm: min / median / max |
|---|---:|---:|---|
| random_dir | 153 | 150 | 0.2714 / 1.0660 / 3.5317 |
| abliteration | 205 | 200 | 0.2409 / 1.1112 / 3.1432 |
| heretic_ara | 13 | 0 | Not applicable; 12 smoke, one unclassified |
| lora_sft | 121 | 120 | 0.3027 / 1.2913 / 12.6604 |
| task_vector | 80 | 80 | 0.0024 / 0.6256 / 2.5531 |
| chain | 675 | 655 | 0.2766 / 2.3938 / 13.3224 |

These are the descriptors' recorded `delta_fro_norm`, not a fresh tensor-level recomputation; cumulative chain norms are not incremental step norms. A norm without base-weight normalization or support information does not establish functional importance.

All 150 production random edits use the two configured tensor roles, with 48/55/47 using one/two/four directions. Abliteration has 73 MLP-only, 55 attention-only and 72 combined edits. SFT has only **two datasets**: 64 GSM8K and 56 Alpaca; ranks 4/8/16/32 have 35/26/34/25 entries; steps 50/150/400/800 have 24/39/30/27 entries. Their sum is **40,650 optimization steps**, not GPU time. Broad hyperparameter sweeps on two tasks do not substitute for task diversity. Source: manifest plus every corresponding `edit.json`; the configuration specifies the sampling ranges, not the realized sample.

Descriptor examples inspected include `2d1de332e36d8f00/edit.json` (800-step rank-32 Alpaca SFT; norm 1.0077), `420e1a3b5a644506/edit.json` (scaled single-source task vector, scale 1.18884), and legacy `7f9c11a8ebcc2286/edit.json` (two-trial ARA; norm approximately 6.67 billion). The last is failure evidence, not desirable target diversity. The current config's ARA minimum is ten trials, whereas `docs/ZOO_IDENTITY.md` describes a default of twenty: qualify the actual config/reuse identity, not prose defaults.

### Dynamic range: mostly near-null controls, with a real SFT signal

I reconstructed predecessor-relative label distributions from `data/runs/p4/d3v2-m2-trajectory/seed-0/fold-*.parquet`: **345 edits, 278 ancestry clusters**. These are the labels actually used by that revised suite, not all 1,247 stored edits and not 389 independent labeled examples. A value of 0.01 is one percentage point.

| Family (n) | MMLU-Pro Δy min / median / max; distinct values | HellaSwag Δy min / median / max; distinct values |
|---|---|---|
| random_dir (60) | −.01 / .00 / .02; 4 | .00 / .00 / .00; 1 |
| abliteration (98) | −.04 / .00 / .03; 8 | −.01 / .00 / .02; 4 |
| lora_sft (59) | −.19 / −.06 / .03; 20 | .01 / .03 / .07; 7 |
| task_vector (30) | −.11 / .00 / .03; 9 | −.02 / .00 / .04; 6 |
| chain (98) | −.14 / .00 / .08; 15 | −.03 / .00 / .05; 9 |

**All random-direction HellaSwag labels are zero; 78.6% of abliteration HellaSwag labels are zero.** Ranking the former is undefined. MMLU-Pro random-edit SD is .00690, below one item of its 100-item score resolution. LoRA MMLU-Pro SD is .04838. There is learnable-looking variation, but concentrated in SFT and some chains; broad pooled correlations can reward identifying SFT rather than predicting fine-grained effects. Positive HellaSwag and negative MMLU-Pro SFT changes also argue against reducing the target to one scalar “capability.” These ranges do not prove generalization to new items.

The latest saved predictions confirm the brief's direction. Recomputed three-seed trajectory-split mean Spearman (MMLU-Pro / HellaSwag): **StatNN .561 / .659; M1-matched .446 / .456; M2 .353 / .316; M3 .148 / .111; delta-norm .206 / .179**. Source: respective `d3v2-*-trajectory/seed-{0,1,2}/fold-*.parquet`; StatNN values here use the matched saved comparator. These descriptive means are not confidence intervals or a formal paired test. Do not copy the older Board A numbers as current results: that review audited the first `d3-*` suite, with missing predecessors and weaker uncertainty accounting. Its cautions remain relevant.

`docs/RESULTS.md` seed-1 panel establishes exact repeatability of teacher-forced/likelihood endpoints on that host, while generation-seed changes disrupt sampled-score rankings. Exact repeatability on a fixed subset does not establish adequate between-edit variation, population accuracy or a numeric Spearman ceiling. Keep sampled GSM8K/MATH/IFEval descriptive at this tier. Use continuous per-item likelihood margins, paired log-likelihood differences, KL and log-PPL alongside discrete accuracy, and qualify each separately. KL is behavior drift, not task competence; it must not silently replace capability.

### Are the chains trajectories?

The 655 production chain snapshots have **150 roots**, not 655 independent trajectories. Their inner steps comprise 217 abliterations, 213 random directions and 225 LoRAs; 430 steps are `sequential`, 225 `lora_vs_base`. Realized descriptor lengths are two through seven, despite a config permitting up to eight. Source: audit and `docs/ZOO_IDENTITY.md`.

These are legitimate interventions for studying order and interference, but they are **arbitrary compositions rather than optimizer trajectories**. A LoRA trained against θ₀ and added after another edit is not training on the resulting model. The task-vector implementation sampled here scales one source; it does not supply diverse two-parent merges. Keep these controls, identify their semantics, and build true sequential fine-tuning with checkpointed optimizer/data state. The revised label set has only **18 length≥4 positions in ten ancestry clusters**. It improves on the old review's one usable position but does not establish a powered history effect.

### What to build instead

The first priority is task SFT with checkpoints along a run, not hundreds more random projections. Proposed first screen: four diverse training datasets × two seeds, four nonzero checkpoints each, with training/test deduplication and dataset licenses checked before selection. Include math, instruction following, code and factual/domain text; attach a domain-specific held-out teacher-forced probe to each, plus common general probes. At least one task must be absent from predictor fitting. Checkpoint counts are correlated samples: splits and uncertainty must hold out whole runs/datasets.

For a six-week core, propose **eight datasets × three seeds × two schedules = 48 independent runs**, five checkpoints per run = 240 endpoints. Add 80 real transformations: 24 pruning variants, 24 quantization variants and 32 two-source merges from independent task adapters. Match support/norm random controls on a smaller subset, not the majority. Pruning and quantization should span mild to destructive settings, with sham reload/dtype controls; report executable format, effective reconstructed bf16 delta and actual inference numerics separately. Merges must group both parents and descendants in splits. Label any catastrophic endpoints separately so a predictor cannot win simply by detecting broken models.

DPO is the next candidate, ahead of online RL: preference tuning with checkpoints on the current SFT state would add genuinely different objectives. Reserve an optional **12 runs × four checkpoints = 48 endpoints** only after preference data/access and a measured runtime pilot exist. Distillation similarly needs a named licensed teacher, cached target provenance and an equal-data SFT control; propose **eight runs × four checkpoints = 32 endpoints** as a later extension. Online RL adds rollout and reward-model confounds and is parked. Neither DPO nor distillation is funded in the core six-week totals.

Expected effect is increased *within-task* signal and controlled trajectories, not a promised predictor gain. Retain small gains, reversals, forgetting and null edits; do not curate only dramatic improvements or use sealed scores to choose training recipes. Freeze the post-pilot recipe before confirmatory collection.

## 2. Amend the hypotheses without rewriting the historical result

Keep original H1–H6 and their failures/undetermined status in the record. Version the following as prospective amendments; never apply easier thresholds retroactively.

| Hypothesis | Status at present scale | Proposed amended statement and test |
|---|---|---|
| H1 readability | Testable on qualified variable fixed-base targets; world-model superiority not demonstrated | **H1a:** weight-derived features predict predecessor-relative teacher-forced changes on unseen task-training runs better than fold mean, descriptors and delta norm. **H1b:** a learned representation improves on StatNN on the identical splits/features budget. Retain the original .6 Spearman bar on each declared primary variable endpoint for H1b, plus paired ancestry-cluster MAE superiority; report H1a independently if trees win. Likelihood margins are a new target, not a replication of generated reasoning accuracy. |
| H2 history | Inadequately tested on real trajectories | Given current cumulative state features plus the proposed next update, history of **actual sequential training** reduces MAE by at least the existing 15% bar against a capacity/feature-matched stateless predictor and a lag/trend baseline, on held-out complete runs of length≥4. Shuffle/drop history controls test genuine temporal information. Equal full weights and edit determine behavior; any history benefit is compensation for compressed state, not fundamental path dependence. |
| H3 JEPA transfer | Original random/ablation→LoRA domain leap unsupported | A JEPA representation trained on several genuine task-training/edit distributions ranks a predeclared unseen task or edit objective at ρ≥.5 on qualified primary endpoints and improves on supervised same-input StatNN/MLP. Hold out all source runs; no target-family unsupervised pretraining in the strict transfer arm. Stage task transfer before SFT→DPO transfer. Park the original two-control-family→LoRA claim as an unsuccessful test, not the main programme. |
| H4 reader comparison | Small blinded screen possible, not a priority | Compare a weights-summary-only LLM reader with StatNN and the learned encoder on frozen in-task and new-task edits, equal information, blinded identities, replicated prompts. Report both directions without presupposing the winner. Any activation/probe-assisted arm is a separate information budget. Park paid execution until model access and publication consent are explicit. |
| H5 critic/editor | Park production gating | Preserve the ≥30% real-evaluation reduction at equal final score, but qualify the surrogate **on editor-proposal distribution**, with coverage and missed-winner checks. Shadow scoring of already evaluated proposals is permitted research; adaptive rejection and optimizer replacement wait for predictive/calibration gates and the H5 protocol's sandbox/measurement preconditions. StatNN may be the eventual critic; no architecture entitlement. |
| H6 transfer | Untestable with one base; cross-architecture claim premature | **H6a:** first test transfer between two Qwen sizes with matched task mixtures, role/depth occupancy audits and dimensionless descriptors; report both directions and separate ≤20-label calibration. Use the existing .4 rank target only on qualified shared endpoints; replace “six of K” with an explicitly frozen endpoint set because six noisy targets cannot qualify it. Two sizes support only this pair claim. **H6b:** three-model leave-one-model-out across architectures remains a later hypothesis, requiring a third model and no target-model pretraining. |

Source constraints: `docs/HYPOTHESIS.md`, its P4 qualification, `docs/MODEL_CANDIDATES.md` H6/LLM sections and `docs/H5_PROTOCOL.md`. Do not use unaligned shape-seeded projection coordinates for H6; canonical LoRA factors do not align unrelated base networks. Same-family size transfer is already a difficult experiment and is not included as a free add-on to the six-week budget.

## 3. Fit to the primary sources and publishable outcomes

I fetched the [blog](https://rikiyat.github.io/blog/weight-space-programming/) and its [released-log README](https://github.com/RikiyaT/lm_understand-logs). Experiment 7 needed attention probes after weights-only failures; Experiment 8 used diagnostic model forwards to guide a localized repair. Neither establishes generic weights-only score prediction. The release excludes earlier prompt trials, so published attempts do not estimate an unconditional editor success rate. This motivates a sparse-layer direction-preserving feature control and explicit information budgets, not H1's claim to replicate the blog through pooled statistics. I inspected the page/README, not independently replayed its code or scores.

The [Weight-Space Physics abstract, 2607.07127](https://arxiv.org/abs/2607.07127) describes couplings mapped to weights on a controlled physical manifold, with observable structure and known ground truth. [Bifurcations from Weight Space, 2610.30847](https://arxiv.org/abs/2610.30847) uses physically indexed trained-network zoos and weight derivatives to locate transitions. My inference: emulate controlled data generation and ground-truth axes before emulating the encoder. Arbitrary LLM edits and coarse benchmark deltas lack those organizing conditions. Neither abstract establishes LLM score forecasting or an SSM advantage.

The [W2T abstract, 2603.15990](https://arxiv.org/abs/2603.15990) and [linked repository](https://github.com/xiaolonghan2000/Weight2Token) support testing canonical representations of task-trained adapters. I verified the abstract's QR/SVD factorization motivation and repository identity, not its tables or code performance. Direction-preserving per-layer tokens versus pooled summaries should be one bounded comparison on the *same real adapters*. Do not treat our pooled Tabular-JEPA as a replication of either JEPAWG paper.

**The honest paper if StatNN keeps winning:** “Measuring and predicting behavioral changes from small-language-model weight edits.” Its contributions would be a reproducible intervention zoo, reliability/target-resolution findings, and a controlled negative result for the tested representations/sequence predictors. It needs:

- Versioned base/edit/config identities, public reconstruction recipes, licenses, failure/quarantine records and immutable per-item labels; clear counts of independent runs versus checkpoints and qualified versus exploratory targets.
- A crossed device/seed/item reliability analysis, paired predecessor targets, continuous and discrete outcomes, effect-range plots and uncertainty that separates item variability, shared references, ancestry and training seeds.
- Strong supervised/statistical/descriptor controls, equal tuning budgets, frozen held-out datasets and edit operators, per-family results, learning curves and a localized direction-preserving control. Explain when trees succeed and where every method fails.
- At least a bounded real-training/task generalization study so this is more than a rerun of a small synthetic perturbation suite. A second base strengthens the scope but is not necessary for a narrowly stated one-base paper.

[TMLR's acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html) prioritize convincing evidence and reader interest, not a mandatory new state of the art. A benchmark/reliability paper can qualify; acceptance is not guaranteed. The actionable insight must survive the easy criticism that our zoo contains near-null controls and our features erased directional information. Failure of M2/M3 is evidence about these representations, data and targets, not a theorem that weights cannot be read.

**The ambitious paper:** a direction-aware state representation predicts task effects across unseen training runs/objectives, genuinely benefits from history beyond current compressed state, and transfers across a model-size pair. It needs paired superiority over tuned StatNN, noncollapsed JEPA latents, multiple full seeds, untouched families/tasks, power justified at the independent-run level and a prospective external validation set. An optimization claim additionally needs the H5 equal-quality/evaluation-saving experiment, including editor costs and missed-good-proposal risk. More architecture names or a successful loop-fix replay do not supply this evidence.

## 4. Plans, cost and stop rules

### Accounting before scheduling

The closed `board/lambda_spend.md` rows sum to **$95.71**, consistent with the user's rounded $96. The open A10 row charges $1.29/h from its recorded start, but has no final duration. Thus spend is **$95.71 + $1.29 × h**, where h is its eventual billed hours; remaining cap is **$54.29 − $1.29 × h**. This audit did not query or change the instance, so it cannot certify present spend. Reconcile that row before any new rental. Do not substitute a calendar-time guess for the bill or terminate another agent's active job.

The brief supplies 7.5 local GPU-min/evaluation, 3.1 H100 effective GPU-min/evaluation with three workers, 66 seconds/feature extraction and 1–10 min/predictor run. These are workload-specific throughput anchors, not training rates or multi-size scaling laws. Ledger H100 rate is $3.29/h (historical project rate, not a live quote); A10 throughput is not assumed equal to H100. All estimates exclude storage/tax and local electricity, which the project does not price.

**Important missing datum:** I examined all 121 LoRA `edit.json` files. None has a duration/timing field. The sample `2d1de332e36d8f00/train.json` supplies loss/steps, not elapsed time; its `generator_edit.json` also does not supply timing. Therefore “LoRA GPU-hours from edit.json” cannot be computed honestly. Instrument the next run: device, warmup/load/train/export times, token count and optimizer-step count. Until then the training allocations below are **hard runtime caps**, not measured estimates or guaranteed completion. DPO, RL, distillation and 4B costs remain pilot-gated.

### Two weeks: bounded local screen, zero new Lambda dollars

| When | Concrete milestone | Local GPU-hour allocation |
|---|---|---:|
| Days 1–2 | Freeze identities/targets and reconcile spend; qualify 12 representative edited states plus matched references under a repeat panel; instrument SFT pilot. Include null, moderate and large edits without selecting on sealed results. | Included in the budgets below |
| Days 3–7, only after gate 1 | Four tasks × two seeds × four checkpoints = 32 new edited states. Save optimizer/data state so these are real trajectories. | Training **cap 4 h** |
| Across days 1–10 | 32 new endpoint batteries + 24 qualification/reference/repeat batteries = **56 fast-tier equivalents**. | **7.00 h** |
| Days 7–12 | New-checkpoint feature extraction, common continuous-target/item expansion and provenance repairs. Stop expansion at its cap; report actual item counts. | **0.587 h features + cap 6 h** measurement reserve |
| Days 11–14 | StatNN, descriptor/norm controls, one direction-preserving encoder and matched history ablation; three full seeds within a fixed search budget. | Predictor **cap 2 h** |

**Total allocation 19.59 local GPU-h; incremental Lambda $0.** Total cloud spend after this plan is the existing $95.71 plus the running A10 bill, not zero. The 56-battery throughput is conservatively the full fast tier even if generation is omitted; measure the teacher-forced path before spending the reserve. The pilot cannot guarantee all eight training runs complete in 4 h: if it predicts an overrun, stop, return measured cost and narrow the screen prospectively. Do not silently cut trajectories or manufacture a finished milestone.

**Gate 1, day 2:** no predictor experiment proceeds until the intended primary target has verified matched-reference provenance, measurable variation and item/repeat qualification under the existing Board A rule. Retain its rank/ICC interval requirements where estimable; if repeats are constant/tied, use paired item uncertainty and report ranking as unqualified rather than declaring a trivial pass. Require at least one variable task endpoint and one common endpoint; no claimed broad capability test if only KL qualifies. If failure persists, spend the remaining allocation on measurement/reporting, not another encoder.

**Gate 2, day 7:** the real SFT screen must show resolvable paired changes on untouched item panels in more than one task. If all changes are below uncertainty, enlarge measurement within the reserve or conclude this scale cannot support the proposed prediction task. If variation exists only across tasks, redesign collection before scaling rather than accepting task-ID classification as readability.

**Gate 3, day 14:** a paired ancestry-cluster interval for MAE improvement over StatNN must exclude zero on a predeclared variable endpoint to justify a larger learned-encoder effort. This is a continuation gate, not the full H1/H2 success bar. If intervals overlap, call it inconclusive; choose the benchmark paper and stop architecture exploration at the cap. If StatNN wins, preserve it as the deployable baseline. Do not relabel a negative learned-model result as a bad baseline.

### Six weeks: conditional real-zoo benchmark, with an explicit fallback

Weeks 3–4, conditional on signal qualification: collect the 48-run/240-checkpoint SFT core described above. Count the two-week eight runs only if they match the frozen protocol; otherwise preserve them as pilot data. Weeks 4–5: add the 80 pruning/quantization/merge endpoints, complete run/dataset holdouts and the direction/history ablations. Week 6: freeze the benchmark, reproduce tables and write the narrowly scoped paper; qualify claims before drafting the abstract.

For **320 endpoints + 48 repeat/reference batteries = 368 fast-tier equivalents**, the total six-week core allocation (including the two-week work, not added on top) is:

- Evaluation: **46.00 local GPU-h**; feature extraction for 320 edited states: **5.87 h**.
- Training/edit generation: **cap 24 h**, predictors: **cap 6 h**, expanded target/reliability work: **cap 16 h**.
- **Total 97.87 local GPU-h; incremental Lambda $0** if executed locally. This assumes scheduling capacity over six weeks, not exclusive ownership of the shared GPU. If the measured training rate overruns the cap, reduce the registered collection before launch and report reduced scope; do not promise 48 runs regardless of cost.

An optional cloud accelerator moves only the 368 fast-tier batteries to an H100: **19.01 H100-h × $3.29 = $62.55**; allow a separately stated two-hour provisioning/retry cap, **$6.58**, giving **$69.13 maximum planned new Lambda** for this option. Training/features remain local; never transfer the local SFT cap to H100 as a measured runtime. The remaining $54.29 before A10 charges cannot fund this option. Request a **$250 total Lambda cap** if acceleration is wanted, conditional on the reconciled ledger leaving at least $69.13; otherwise keep the local plan. This is a funding option, not a rental request made by this review.

Optional extensions are excluded from those totals: the 48 DPO endpoints alone cost 6.00 local evaluation h + .88 feature h (or $8.16 H100 evaluation), **plus unmeasured training**; the 32 distillation endpoints cost 4.00 local evaluation h + .59 feature h (or $5.44 H100 evaluation), **plus teacher generation/training**. Pruning/quantization/merge generation must fit the 24-h core cap after the timing pilot. No 27B, protein sidecar, online RL or full H5 reproduction enters either core plan. H6a requires a separately costed 4B pilot and accessible pinned weights; do not extrapolate 0.8B timing to 4B.

At week 4, stop learned-model expansion if paired StatNN superiority remains absent on a frozen task holdout despite valid signal. At week 6, publish/release the measurement benchmark with scoped negatives, or explicitly defer submission if release/reliability is incomplete. Calendar time and money already spent are not continuation criteria.

### Decisions needed from the user

1. Keep the $150 Lambda cap and use the local plan, or raise it to $250 for the conditional evaluation accelerator after reconciling A10 billing? Local GPU availability also needs a scheduled allocation from the project owner.
2. Approve the benchmark/measurement outcome as a successful project even if the world model never beats StatNN. Otherwise no bounded scientific stop rule is credible.
3. Confirm available/licensed training datasets and, for later H6, access to a pinned Qwen size pair. For any later H4/H5 episode, explicitly choose commercial editor identity, API/token budget and transcript-publication consent. For distillation, name an accessible teacher and permitted data outputs. Neither editor access nor a 4B allocation is assumed in the core plan.

## 5. Three questions to the council

1. **Weight-reading researcher + features CTO:** can we hold layer/support and singular directions fixed while changing their orientation relative to base weights, producing matched pooled norms/spectra but different qualified task effects? What single direction-preserving feature comparison would distinguish representation loss from insufficient zoo signal within the two-week cap?
2. **JEPAWG researcher:** what is our analogue of a known physical parameter manifold, and which observable would falsify that analogue? Would task/dataset/step-indexed genuine training runs support a prospective JEPA-vs-supervised comparison, or are we importing the architecture without its source of identifiability?
3. **Metrology CTO, with features CTO:** which continuous predecessor-relative endpoints have enough within-task variation on independent item panels, and how many independent runs—not checkpoint rows—are required to resolve the registered 15% history MAE effect? Please specify the smallest panel and its measured cost before approving a larger zoo.

Branch: council-r1-strategy.
Current zoo: useful controls plus task signal, but dominated by near-null edits and correlated arbitrary compositions.
Amend first: split H1 into qualified teacher-forced readability and a separate prospective test of superiority to StatNN.
First gate: matched-reference provenance, nondegenerate target variation and Board A measurement qualification before new predictors.
Two-week cost: at most 19.59 local GPU-h allocated, $0 new Lambda; existing $95.71 plus running A10 billing remains.
User decision: accept the benchmark-first stop rule and choose local execution under $150 or a reconciled $250 cloud cap.
