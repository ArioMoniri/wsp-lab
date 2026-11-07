# TF-v1 pre-registration — 5 November 2026

Status: frozen prospective design, pending Board A acceptance and measured execution gates. This document reports no new qualification, timing or power result. User decisions adopted: TF-v1 programme **yes**, core **30 local GPU-h**, additional history **no**, new Lambda **no**. The registration commit fixes the rules below; execution manifests must be committed before scoring, not filled from outcomes. A missing manifest or failed gate prevents confirmation. Changes require a dated amendment before affected outcomes, never retrospective repair.

Authority: [Board A revision](reviews/P4/board-a-synthesis-and-v2-2026-11-05.md) supersedes conflicting proposals in [synthesis](council/2026-11-05/SYNTHESIS.md), [metrology](council/2026-11-05/round2-metrology.md) and [strategy](council/2026-11-05/round2-strategy.md). [DECISIONS](DECISIONS.md) D1 and D2 preserve effective-bf16 and measured-reference semantics. The [4 November qualification protocol](reviews/H5/board-a-prereg-2026-11-04.md) remains binding for stochastic endpoints. All new tolerances/margins specified here are prospective decision criteria, not measured numbers.

## 1. Primary claim and target

Board A's approved H1a wording, verbatim:

> On sealed fresh source-closed components, estimate row-weighted relative residual MAE reduction of fixed StatNN versus training-fitted isotonic norm after training-only family, device and log-norm adjustment for predecessor-relative MMLU gold-letter log-prob.

This is a conditional predictive loss contrast, not a causal weight contribution or universal within-family law. For item j, normalize the frozen answer-letter score vector with logsumexp; take the gold entry. For edit i, y_i is the mean of **1,000** paired edited-minus-actual-predecessor normalized gold-letter log-prob values, in nats. Root edits use the matched base. The score vector, gold index, option count, prompts, tokenizer, model revision, answer-letter tokenization and aggregation code are hashed. Invalid letter tokenization stops the scorer; do not substitute option-text likelihood. Incremental effective bf16 norm is measured against the actual predecessor; cumulative-from-base feature provenance is recorded separately.

Non-null means the outcome-free family stratum `{lora_sft, task_vector, chain}`, not a test on observed KL or task change. All valid registered rows, including zero effects and destructive edits, enter the denominator.

## 2. Sealed source/ancestry and item partitions

Before **any new TF-v1 scoring**, Zoo/Metrology must commit an immutable inventory and partition manifest with seed `20261105`, registration SHA, config/input hashes, model revision and canonical IDs. The following algorithm, rather than an asserted current independent count, is frozen.

1. Snapshot canonical `data/zoo/qwen3.5-0.8b/manifest.jsonl`, each `edit.json`, effective-delta content hashes, generator records, all previous evaluation/predictor membership and D3 IDs. Record the snapshot hash and corrected-ARA cohort cutoff. No newly arriving rebuild is added midstream.
2. Build a graph over **all** entries, including unlabeled ancestors. Join actual parent/predecessor edges, chain inner steps, task-vector/merge operand edges, shared trained adapter/checkpoint/source-run identities, and duplicate effective-delta/source hashes. Trace those edges transitively to complete closure. Shared immutable base model alone and generic dataset membership do not join all edits; shared trained state or common training run does. Record dataset/task sharing as a support diagnostic. Missing/ambiguous sources, cycles or conflicting predecessors make the entire connected component ineligible until resolved without outcomes. A nominal root is not the independence unit.
3. Development comprises every component touching D3, prior evaluated/predictor/feature-development panels, or the timing/qualification pilot. Pilot entries come from D3. Unlabeled ancestors and source-connected siblings of development rows cannot become fresh. Deduplicate exact copies, keep a canonical lexicographically smallest ID and a complete alias ledger. Shared reference records are retained once with many links; base references do not count as independent edits.
4. Audit the **110 previously unevaluated** candidate entries (60 `lora_sft`, 50 `task_vector`) from the metrology snapshot, and existing chain lineages. Membership is reconstructed from that snapshot, not a live query after evaluation; failure to reproduce the inventory stops confirmation. The reported 98 roots/75 D3-new roots plus about 30 new chain roots are planning counts, not source-closure certification ([metrology §1.1](council/2026-11-05/round2-metrology.md)).
5. Component ID is SHA256 of the canonical sorted member/source-identity list. Rank by SHA256 of UTF-8 `20261105:H1a:` followed by component ID; ties use component ID. First include all eligible fresh components containing any of the 110 candidates, then append eligible fresh chain components in this hash order until **100** components are reached. If the first set already exceeds 100, retain all. Chain additions use all existing eligible stratum rows in their full lineage, not a favorable endpoint checkpoint. The complete predecessor/source closure is priced and measured even where ancestors are not primary stratum rows. No KL/effect/rank-based selection, no substitutions after scoring. If insufficient components or closure cost exceeds cap, stop with a census and H1a untested.
6. Commit selected IDs, all closure/reference IDs, hash ranks, development/sealed flags, exclusion reasons, sources, device assignment and expected row/component counts. Keep sealed item records and metrics inaccessible to feature developers. Any exposure contaminates that component for future confirmation; do not replace it to rescue the current claim. Report all exclusions and resulting count; <100 invalidates confirmation.

Items: hash-sort canonical dataset-revision/item-ID fingerprints using the same seed and distinct `TF-v1:item:` domain, after deduplicating normalized text/problem groups. Remove every legacy, pilot and calibration item/group from confirmation. The first 1,000 eligible MMLU groups form the sealed panel; remaining groups supply disjoint development/qualification items. Freeze item IDs, revisions, group definitions and hashes before scoring. D3 training and qualification use only development items; the frozen predictor is evaluated once on the sealed items and fresh components. This tests both unseen components and items. Legacy 100-item bridge is separate. Shared text/variants form one resampling block; do not split near duplicates across panels. No sufficient disjoint item pool means no confirmatory endpoint. Secondary panels (HellaSwag 500; KL 200 continuations; WikiText 64 windows) get distinct hash domains and explicit overlap/exclusion manifests.

## 3. Frozen fitting, aggregation and missingness

Training data are measured, matched-reference TF-v1 **D3-only** rows on development items; historically inferred-runtime labels never enter. Use all eligible D3 families with supported design levels; the fresh evaluation stratum remains §1. Feature input is the original pooled statistics block loaded by `load_feature_table` in [baselines.py](../src/wsp/predictor/baselines.py), no directional augmentation for H1a. Commit its exact schema/order/hash and source revision; do not combine the historical 1,098-coordinate layout and later layouts. Missing cache or schema mismatch must be repaired before fitting, not imputed from sealed data.

Five source-closed inner folds: hash-sort development components with `20261105:inner:` and assign round-robin. Every fold's nuisance and feature preprocessing use training components only, including unlabeled source closure. Fit OLS with intercept, lexicographically sorted treatment-coded family/device dummies (first training level reference), and `log1p(||Δ_eff||_F)`; this fixes zero norm at zero. OLS uses the Moore–Penrose solution in float64. Audit design rank and family/device support; unresolved confounding that defeats the registered adjustment stops confirmation. Unseen family/device levels fail closed, without held-out-mean centering or imputation. A one-device claim must be fixed before scoring and trained/qualified on that device.

Train residual learners on inner-OOF residuals. Fit the final nuisance on all development rows for fresh residuals. Isotonic is increasing, unweighted, on raw incremental norm with `out_of_bounds=clip`; training ties are pooled. StatNN is LightGBM with the existing [fixed defaults](../src/wsp/predictor/baselines.py): 300 trees, learning rate .05, 8 leaves, min child samples 5, subsample .8/frequency 1, column sample .5, L2 1, two threads, deterministic row-wise. No search, early stopping or sealed fitting. Seeds **0,1,2**, identical folds and inputs; average the three **predictions per row before absolute loss**, never average selected scores or treat seeds as independent components. Isotonic/nuisance are shared deterministic fits. Report individual-seed estimates/range separately, not as CIs. Any missing seed prevents the primary analysis.

Audit and report inner-OOF versus full-refit nuisance prediction/residual distributions on development only; this distribution mismatch is a limitation, not a reason to recalibrate on fresh labels. Preprocessing uses no fresh target statistics. Infrastructure retries preserve immutable failed attempts and identical inputs; no selective rerun of poor scientific outcomes. Missing items/references, nonfinite features or predictions and unsupported cells are logged. Primary needs complete paired 1,000-item records; no denominator shrinkage by silent available-case scoring. Post-freeze exclusions require reasons, counts and a full-panel missingness sensitivity; no success if the ≥100 valid-component floor or qualification fails.

Mandatory descriptive sensitivities: equal-component-weight loss ratio; family×norm flexible nuisance (training-only cubic spline with knots at development norm quartiles, family interactions); descriptor/task/rank/steps/support controls; within-device/family support and device×edit diagnostics. They do not replace the primary.

## 4. Loss, denominator, uncertainty and decisions

Let r_i = y_i − m_D3(i), A = Σ_i |r_i − mean_s p_StatNN,s(i)| and B = Σ_i |r_i − p_isotonic(i)|. Primary θ = 1 − A/B, with **row sums**, preserving all rows of each selected component. Freeze the numerical denominator tolerance at **1e-12 nats per row**: B/n ≤ tolerance is undefined, including in resamples; never clamp B or report a zero gain. This is a numerical guard, not evidence of endpoint resolution.

Conditional component CI: **4,000 paired source/ancestry-component percentile bootstrap replicates**, seed 20261105, sampling selected components with replacement at the original component count; carry all their rows, both predictions and shared reference links together. Recompute the full row-summed ratio each time, not a mean of component ratios. Report 2.5/97.5 percentiles, valid count, all denominator failures, and component-size distribution. A nonfinite/denominator-failing replicate makes the confirmatory precision gate unmet; retain valid-only intervals as diagnostics rather than retrying until 4,000 pass.

Item-population claim additionally requires a **4,000-replicate joint component/item-block bootstrap**: independently resample components and sealed independent problem blocks, use one common item draw across every edit and predecessor/reference (including shared base), recompute paired y, residuals and losses against fixed predictions. Keep the D3 fit fixed: this CI covers fresh components/items conditional on the fitted pipeline and development set, not population training/refitting variation. Also report item-only intervals and seed-wise joint intervals. No per-token independence or independently re-drawn shared base. Confirmatory decisions must satisfy both component-only and joint intervals; precision half-width means (upper−lower)/2 for each. Report training-seed variation separately; it is not covered by the fixed-fit bootstrap.

One endpoint, one contrast, two-sided α=.05: no multiplicity correction for H1a. Margin, HellaSwag, subgroups, alternatives and accuracy are secondary/descriptive; no best endpoint/seed/model selection. Any later promoted confirmatory family needs a prospective fixed contrast list, fresh untouched outcomes and Holm control at .05; this registration promotes none.

Pass only with θ≥.15, lower 95% bound>0, half-width≤θ/2, ≥100 valid genuinely fresh components, measured qualification and prospectively demonstrated ≥80% power at .15. Kill practical readability **for this endpoint/design only** when upper 95% bound<.05 on a qualified adequately powered panel (both intervals). Otherwise intermediate/wide results are inconclusive. Qualification/count/power failure means untested; stop at the cap without a universal unreadability claim.

## 5. TF-specific power gate (before fresh outcomes)

Metrology must publish reproducible code/config/seed and a development-only report before sealed execution. Obtain paired per-item TF-v1 D3 OOF residual/prediction losses under §3, source components, actual size distribution, item/reference covariance and the qualified-device support. Use the actual sealed component counts/sizes (metadata only), development family/device strata, joint component/item-block draws and three fixed seeds. Generate paired-loss alternatives calibrated so the **row-summed ratio** is .15; retain empirical comparator loss and within-component/shared-item covariance, and report the calibration and nonnegative-loss construction. These are power-design simulations, never scientific result rows or invented observations. Include null calibration at θ=0, denominator failures, and conservative variance/cluster-size sensitivities, including OOF/full-refit mismatch.

Use 4,000 simulated panels per scenario with 4,000 bootstrap draws for each CI, frozen seed 20261105. Estimate rejection probability (lower bound>0 at two-sided 5%) at .15 and report its binomial uncertainty and precision-gate attainment separately. Require the lower 95% binomial power bound ≥.80 for both conditional and joint procedures before confirmation; freeze the achieved panel-size requirement before unsealing. The .15 point criterion is a decision threshold, not a claim that ≥80% of estimates equal/exceed their true mean. If power or planned precision is inadequate, expand only outcome-free closure within cap before scoring, otherwise stop untested. The old **79 accuracy clusters** is not TF power; ≥100 is a floor, not proof. No achieved power report exists in this documentation revision ([Board A](reviews/P4/board-a-synthesis-and-v2-2026-11-05.md)).

## 6. Prospective deterministic qualification and measured validation

Board A's approved deterministic qualification rule, verbatim:

> prospectively require measured provenance, bounded paired repeat and device error, lower 95% signal-reliability bound ≥.90, and upper 95% paired item and repeat-error bounds ≤δ/2 on a variable representative panel.

Freeze endpoint, effect scale, panel and CI procedure before executions. For MMLU predecessor-relative log-prob, this registration sets the smallest meaningful **endpoint effect δ=.10 nats** (a new planned resolution criterion, distinct from θ=.15 relative predictive loss). KL δ=.01 nats and WikiText δ=log(1.05) retain [the prior protocol](reviews/H5/board-a-prereg-2026-11-04.md); these diagnostics cannot pass H1a. Margins/other likelihoods stay unqualified/descriptive until their own effect scales are prospectively fixed.

Measured validation uses D3-only pilot/representative panels with matched measured identity: device, actual batch/order, dtype, code, source tree, model/tokenizer revision, items, templates and real predecessor run hashes. Never qualify from backfilled inferred runtime or copied scalar snapshots. Independent forced repeat executions in immutable run directories on every used device; base/reference executed independently and paired throughout. Start with metrology's ten-entry rule (4 LoRA: two nearest median legacy KL/two largest; 3 chains including ≥4 nodes; 2 task; 1 largest-KL abliteration; unspecified choices/ties sorted ID), then expand on the prespecified development hash order if bounds lack support. Legacy KL selects this development pilot only, never the fresh panel.

Maximum same-device repeat discrepancy ≤1e-4 nats for MMLU/log-PPL, ≤1e-5 for KL; failures investigated and qualification withheld. Cross-device SD of paired centred differences ≤.2 times between-entry SD for MMLU, ≤.1 for KL/WikiText; otherwise restrict claims to the independently qualified device before fresh execution. Missing second device is single-device scope, not a transport pass.

Use 4,000 component bootstrap draws for lower 95% signal-reliability bound and upper 95% uncertainty bounds. On training-only family/device residual effects, estimate signal variance as between-entry variance minus paired item/reference and repeat-error variance, with reliability signal/(signal+error). Report untruncated estimates/negative signal as failures. Bootstrap independent items/blocks jointly across edits and shared references to estimate relevant paired endpoint SE and 95% uncertainty half-width; its upper 95% component-bootstrap bound must be ≤δ/2. The upper 95% bound on the repeat-error **95th percentile** must independently be ≤δ/2. Reliability lower bound must be ≥.90. Show bias, tails, ties, host×edit interaction, variance components, panel composition and sampling uncertainty. Constant/undefined panels fail. Small panels unable to bound errors must expand within cap or remain unqualified; projections are not validation. Keep the stochastic rank/ICC and sampling-variance rules unchanged.

The 7 Nov bridge preserves 100 legacy items and compares old option-text with new letter scoring: median absolute difference≤.02 nats, p99≤.2, argmax agreement≥99%. Failure explicitly versions a new endpoint and rebuilds all matched references; never pools old and new values. Successful bridge alone does not qualify signal or effect resolution.

## 7. H1b and closure-first H2

H1b: one frozen base-frame signed-projection candidate (`U16ᵀ ΔW V16`, exact layer/role, outcome-free hashed frames and degeneracy policy) with a small regularized head versus original StatNN **and StatNN on the augmented features**. Same components/items, seeds, nuisance, information and total tuning budget. Freeze candidate/head/config/permutation controls on D3 before **any H1a unsealing**; if absent, no sealed H1b run. Directional screen on 14–16 Nov uses development only: ≥.05 residual-rank increment and ≥5% residual MAE gain with destroyed-frame controls losing the gain. H1b's stronger criterion is ≥5% residual MAE improvement over both trees, positive paired intervals, and ρ≥.6 on the sole MMLU primary. This is a descriptive screen here, not an additional confirmatory claim; a later powered ≥80% TF paired-loss design with multiplicity protection and untouched data is necessary. Upper bound<5% rejects useful superiority only on that adequately powered design; wide intervals stop expansion inconclusively. H1a labels cannot train the screen. A tree winning is valid representation evidence, not learned-world-model superiority ([synthesis §3](council/2026-11-05/SYNTHESIS.md)).

H2 is **parked for compute** under the adopted no-extra-10-h decision. On 17–18 Nov deliver only a cached closure census/cost/power design, no new scoring or training. Future activation needs separate funding and amendment. Start from existing long lineages, complete all actual predecessor/source closure and optimizer/checkpoint provenance; deduplicate already measured H1 states. About 40 closures are feasibility, not confirmation; the 162-lineage/SD .68 calculation is an accuracy stand-in, not TF power.

Future H2 estimand: held-out length≥4 complete-run residual MAE reduction of true-history M2 versus within-M2 masked history, with identical architecture, initialization, features, current cumulative state, incremental edit and training budget; also beat causal within-lineage shuffle and lag/trend controls. Shuffle only available past history, preserving family/source blocks; hold out full connected runs. Predeclare ≥15% improvement, lower paired joint 95% bound>0, fresh variance-based ≥80% power and qualified effects. Length-one placebo relative-loss contrast must have its **entire 95% CI inside [−.05,+.05]**, a new planned equivalence margin (one third of the 15% H2 target). Non-significance is insufficient. Matched initialization should give equivalent length-one execution paths; audit directly. Failed equivalence, SD>1.0, absent closure/provenance or unaffordable power stops H2 unresolved; qualified powered upper bound<.05 kills useful history gain for this design only. Arbitrary edit chains test compressed-state history, not optimizer dynamics. No cross-architecture comparison establishes history.

## 8. Descriptive evidence and quarantine ledger

Historical accuracy, including MMLU/HellaSwag, sampled reasoning, keyword refusal and capped loop rates, is descriptive-only; no TF or capability confirmation inferred. KL is a validity/sensitivity outcome and WikiText diagnostic. Inferred-runtime historical records are quarantined from qualification, training and confirmation until independently remeasured with provenance; strict string matching of inferred fields is insufficient ([DECISIONS D2](DECISIONS.md)).

Baseline aggregation correction: [v2 summary](reviews/P4/d3v2-summary-2026-11-05.md) neural entries are finite seed-point means/ranges; **StatNN .57/.65 is seed 0**, not a three-seed mean. Dashes mean missing outputs, NaN remains undefined. [Board A §A](reviews/P4/board-a-synthesis-and-v2-2026-11-05.md) audited all 75 seed files across 25 runs for nonempty per-family/centred/paired objects, not complete per-cell validity. Publish each run/seed/fold/target's finite n, rows/components, skips, baseline aggregation, paired CI conditioning and exclusions before any expanded v2 claim; that per-cell audit remains an execution deliverable. Five M2 family-purge runs are separate one-fold diagnostics, not seeds/M3 transfer. M3 ranks are logged without an M6 acceptance gate; M6 fails its all-seed gate on all standard splits (rank≥2 and ≥2×frozen-predictor rank), with no LoRA acceptance run. Neither gives a collapse-qualified transfer claim. Centring test labels is descriptive, never deployable prediction. H1 not demonstrated, H2/H3 undetermined, H4–H6 untested.

All twelve entries below share failure reason **unconstrained refusals-first selection and row-normalized-surrogate/unnormalized-deployment scale mismatch**, flag `ara_unconstrained_surrogate_scale`, `eligibility: smoke`; all are excluded from production fitting, qualification and confirmation, and retained only as explicit diagnostics. Source: [exhaustive forensic table](reviews/P1/heretic-ara-rebuild-forensic-2026-11-05.md), [D6 addendum 3](DECISIONS.md).

| Flagged ID | Disposition |
|---|---|
| `bbf6af9653df23c2` | Diagnostic only; excluded |
| `770d16bb686c8226` | Diagnostic only; excluded |
| `39b0bc05af270314` | Diagnostic only; excluded |
| `cb43afcdcf968a92` | Diagnostic only; excluded |
| `eeda0ec4d3dc118b` | Diagnostic only; excluded |
| `2b3c19e6158e0788` | Diagnostic only; excluded |
| `0b5a6ef591cee209` | Diagnostic only; excluded |
| `5c1c957dab65753f` | Diagnostic only; excluded |
| `d9da6daecd7c8d6d` | Diagnostic only; excluded |
| `cec362b7c5eaf0f1` | Diagnostic only; excluded |
| `cbc856f1b3ed9f7b` | Diagnostic only; excluded |
| `69c7390859177261` | Diagnostic only; excluded |

A corrected ARA rebuild is a separately versioned future cohort with new hashes, independent measured qualification and review; it does not repair historical labels. Its arrival never substitutes fresh panel members. Flag-connected chains are ineligible; record closure exclusions explicitly.

## 9. Budget reconciliation and scaling stop

This table replaces strategy's 17.04-h proposal and the synthesis's optional 40-h/paid replacement paths for the adopted programme. Caps are occupied local device time, including failures/retries, not utilization measurements or billed-cost forecasts. Anchors: [metrology output §C](council/2026-11-05/round2_metrology_output.txt), [synthesis accounting](council/2026-11-05/SYNTHESIS.md), [spend ledger](../board/lambda_spend.md).

| Work | Basis / uncertainty | Adopted cap |
|---|---|---:|
| Pilot, bridge, D3 and fresh TF scoring | Nominal 552 runs × assumed 1–2 min =9.20–18.40 h; actual closure count governs; reference/cold/retry/qualification/item overhead included | 19.00 local h |
| Baseline cache repair | Up to ~230 states × measured 66 s =4.22 h, rounded | 4.22 local h |
| One directional extraction including frames | ~230×proposed 33 s =2.11 h; timing unmeasured | 2.11 local h |
| All three-seed fits/screens | Measured historical 1–10 min/run anchor, CPU preferred | 2.00 local h |
| Closure/device/overrun reserve | Pooled reserve, not another experiment | 2.67 local h |
| **Core total** | Hard ceiling, no automatic extension | **30.00 local h** |
| Additional history | User default no; census/design only | **0 additional h** |
| New Lambda | User default no, optional $13.16 path disabled | **$0** |
| Closed Lambda ledger | Recorded closed rows | **$95.71** |
| Existing open A10 | $1.29×h, h = total final billed hours; unreconciled | **$1.29h** |
| Billing reserve / total cap | Require 95.71+1.29h+5≤150; h≤38.20 conservatively | **$5 / $150** |

At the assumed two minutes/run only **0.60 h** remains inside 19 h for all scoring overhead. TF throughput has not been measured; old H1 fast-tier 9.92 min gives 552×9.92/60=91.26 h and is not an affordable fallback. The 6 Nov ten-entry pilot must report cold/warm load, references, scoring, retry and closure costs, plus total committed/remaining cap. Warm median≤2 min is necessary but not sufficient: the projected full programme including qualification/power and every overhead must fit remaining allocations before scaling. At 2–3 min reduce only optional secondary scope and reprice; >3 min remove MATH NLL then HellaSwag 500→250 and re-pilot. Never reduce primary items/components or drop expensive outcomes to manufacture confirmation. Stop with measurement results if it does not fit. Ten-entry directional timing must likewise fit its allocation before extraction scales.

Ledger owner must reconcile actual final A10 charge and bounded remaining liability at the first gate; an open row is not free balance. No rental query/termination or bill reconciliation was performed by this design task. Owner closes the existing rental after verified export/job completion under repository rules. If the total cap/reserve inequality fails, escalate; no new spend is authorized. Jobs >~2 minutes must use `agent-tmux`; all artifacts use canonical data symlinks and immutable manifests. Final reporting gate is **19 Nov 2026**, with no automatic extension.

## 10. Amendments (dated; each precedes the outcomes it affects)

### A1 — 5 November 2026 (before any fresh-panel scoring; no fresh component has been scored)

Status: records the as-built TF-v1 tier, resolves the KL δ/2 scope that Board A's re-review found unregistered, registers the gate-1 result and the budget consequence, and fixes the interpretation of the §2 closure algorithm that the first inventory run required. It changes no registered effect size (MMLU δ=.10, KL δ=.01, WikiText δ=log 1.05, θ=.15) and no confirmatory threshold. Measured numbers are in [RESULTS](RESULTS.md) ("TF-v1 timing pilot (round 2)", "TF-v1 G1", "TF-v1 G2", "TF-v1 batching probe"); source `source_hash` 12ee875a68c9.

**A1.1 As-built tier (supersedes the 128-token, 500-item and first-token wording in §§2, 6 and 9 and in the 5 Nov D2 addendum).**

| Endpoint | As built | Status |
|---|---|---|
| MMLU-Pro gold-letter log-prob | 1,000 items, hash-ranked development panel, letter tokens single and distinct or the scorer stops, batch 1 | primary endpoint; qualification target |
| KL(base‖edited) | 200 prompts × up to 64 greedy base-continuation tokens, exact full-vocabulary bf16 logits from a hash-verified cache, averaged over positions then unweighted over prompts (not 200 continuations of 128 tokens); direction base‖edited | validity/sensitivity outcome; cannot pass H1a |
| HellaSwag | 250 items, nested prefix of the 500-item ranking, char-normalised log-likelihood | descriptive, unqualified |
| WikiText | 64 windows × 1,024 tokens, pinned revision, token-weighted log-PPL | diagnostic; cannot pass H1a |
| Refusal / over-refusal | 50 harmful prompts + 50 harmless prompts, phrase-set margin = logsumexp(6 refusal phrases) − logsumexp(7 compliance phrases), phrases scored over all their tokens | descriptive and uncalibrated; **not a refusal measure** until frozen from a disjoint calibration set. The artefacts carry no such flag (the source is frozen); the label lives in the config, DECISIONS and this section |

Development items (the 1,000/250/200/50 panels above and the 300 legacy items) are barred from the sealed panel (§2). The per-item panel hash domain `…:TF-v1:pilot-dev:20261105` is the development salt; the sealed items use the distinct `TF-v1:item:` domain and normalised-text-group deduplication (§2), and **are not built by this amendment**; no sealed item exists yet and none may be scored until the item manifest is committed.

**A1.2 KL δ/2 scope (choice: report but exclude high-KL entries from the KL qualification; δ is not widened).** The paired-item 95% half-width of an entry's KL scales with the entry's own KL (the pilot's KL .323 entry has half-width .017, 3.4 × δ/2 = .005; the other nine are ≤ .0018), so an absolute δ/2 bound cannot hold for large edits. Registered rule: an entry is *in the KL gate* if its measured KL ≤ 10δ = **0.10 nats**; the gate (δ/2 on the item half-width for every in-gate entry, δ/2 on the upper 97.5th-percentile component-bootstrap bound of the panel-mean half-width of in-gate entries, repeat ≤ 1e-5, reliability ≥ .90 on in-gate entries) applies to them. Entries above 0.10 are listed with their absolute and relative (half-width / KL) uncertainty and get no pass/fail. If more than 20% of the panel's entries exceed 0.10, KL is declared unqualified for that panel. Justification: δ=.01 is a smallest meaningful change in the low-divergence regime where the H1a stratum sits; widening δ would loosen the gate for every entry to accommodate a few, whereas an outcome-free-threshold exclusion keeps the gate strict where it is meaningful and discloses the excluded set and the 20% tripwire. The 0.10 threshold is a function of δ (10δ), not of a pilot quantile. KL cannot pass H1a either way, so this scope affects only the validity/sensitivity role.

**A1.3 δ/2 scope for the other endpoints (clarification of §6).** For MMLU gold log-prob and WikiText log-PPL the δ/2 bound is checked two ways, both required: every panel entry's own item half-width ≤ δ/2, and the 97.5th-percentile component-bootstrap bound of the panel-mean half-width ≤ δ/2. (The pilot's largest MMLU half-width, .0436, is 87% of δ/2 = .05; the strict per-entry reading is therefore live.) "Upper 95% bound" means the 97.5th percentile of the 4,000 draws, seed 20261105. Item draws are common across all edits and the shared reference; chain entries use their actual predecessor as the reference.

**A1.4 Gate 1 and budget (§9 sufficiency test).** Warm median 1.805 min/entry (range 1.781–1.833) on 10 pilot entries passes the 2.0 min line; G1 re-measured 1.78 min median over the repeat pass. The 552-run nominal projection is ≈16.6 h scoring + ≈1.5 h pilots + overheads ≈ 19.0 h with no retry slack. **At the actual closure count the sufficiency test is not met without a scope or source change:** the D3 labelled set is 389 evaluated entries (381 unique deltas) and the fresh closure is 255 unique deltas (278 entries) per the sealed partition, 636 unique entries in all, of which 23 are already scored; 613 remain ≈ 18.4 h at 1.805 min. With consumed time (pilots 1.5 h, G1 0.9 h, G2 and probe 0.1 h = 2.5 h) and ≈0.5 h process overhead the projection is ≈**21.5 h against the 19.0 h cap**, an overrun of ≈2.5 h that would consume almost all of the 2.67 h pooled reserve (not an experiment line). Per §9 ("stop with measurement results if it does not fit"), **H1a fresh scoring does not start under the as-built source until one of the following is recorded in a further dated amendment:** (a) the batching remedy of A1.5, or (b) an owner-approved reduction of optional scope or D3 label rows that leaves the §4 primary items/components intact, or (c) an owner-approved cap increase. Nothing in this amendment starts, authorises or schedules compute.

**A1.5 Optional batching remedy (measured, not implemented).** Scoring the 4 HellaSwag endings and the refusal/compliance phrases in one padded forward saves 43.7 s per entry (1.805 → ≈1.08 min; ≈7 h over 613 entries; projected total ≈15.4 h including a repeated pilot and G1) at the price of a different bf16 numeric path (HellaSwag deviations ≤ .010 with 1 of 250 argmax changes; phrase log-prob deviations ≤ .39, margins ≤ .29; the batched path repeats bitwise). MMLU-Pro, KL and WikiText stay batch 1 and are untouched, so the primary endpoint's semantics do not change. Adoption is permitted only as a source change made before any reference is generated, followed by a new warm timing pilot, a repeated G1 on the §6 composition and a repeated G2 on the final `source_hash`; the artefacts under `source_hash` 12ee875a68c9 then become development evidence only. See RESULTS "TF-v1 batching probe".

**A1.6 Qualification gates executed before the panel (§6).** G1 (same-device bitwise repeat) on the §6 composition: the composition is fixed by the §6 rule, not by the ten D3 controls (these contain no chain), and was derived by `scripts/tf_v1_qualification_panel.py` (committed at `docs/prereg/qualification_panel.json`): 4 LoRA (2 nearest the median legacy KL, 2 largest), 3 chains at step ≥ 3 from distinct lineages (first by sorted ID), 2 task-vector (first by sorted ID), 1 largest-KL abliteration, plus the three chain predecessors the predecessor-relative endpoint needs. Result: 14 of 14 repeats bitwise-identical. G2 same-item bridge on the 100+100 legacy items, 11 records: passes (MMLU argmax ≥ 99%, median |Δ| ≈ 0, p99 .066; HellaSwag bitwise equal). G3 (second device) cannot run: single-device scope (DECISIONS, "H0 is the sole scoring device"). Reliability and δ/2 gates: `docs/reviews/P5/tf-v1-qualification-2026-11-05.md`. The §6 panel is expanded from the ten composition entries by the ten already-measured D3 pilot controls (20 y-bearing entries, development; not the prespecified hash-order expansion), labelled as a deviation there.

**A1.7 Closure-algorithm interpretation (§2 items 2, 3 and 5).** (i) Joins: parent/predecessor edges, equal generator (trained-adapter) identity, byte-equal stored deltas. A chain's step-0 delta is byte-identical to a standalone random_dir/abliteration delta with the same inner seed in 101 cases; these are exact copies, joined, scored once (canonical lexicographically smallest id, alias ledger in the manifest). The generation `reuse_key` is a recipe-class key shared by whole families and is not a source identity. The refusal-direction fingerprint (one artefact shared by every abliteration-derived entry) and the dataset are **support diagnostics, not joins**, as §2 says for shared dataset; joining on the fingerprint would merge every abliteration-derived component into one. (ii) A component is development if any member is D3-evaluated or a pilot/qualification entry (weight-only feature extraction exposes no outcome and is not membership); excluded if any member has non-production or missing eligibility, a flagged ARA id (§8), a missing artefact, a missing or inconsistent predecessor; otherwise fresh. (iii) The 110 candidates are the 60 eligible lora_sft and 50 task_vector entries without an evaluation (a 61st unevaluated lora_sft, `04ce1ebc4749f1c8`, is a legacy entry without eligibility and is excluded). (iv) Non-stratum members (abliteration, random_dir) of selected components are closure/reference entries priced and scored once, not rows. (v) The partition manifest is built by `scripts/build_h1a_panel.py` and sealed by its content hash before any fresh scoring.

### A2 — 5 November 2026 (before any D3 re-scoring under the final source and before any fresh-panel scoring; no fresh component has been scored)

Status: transcribes Board A's adjudication ([board-a-qualification-adjudication-2026-11-05.md](reviews/P5/board-a-qualification-adjudication-2026-11-05.md), decisions 1–5 and the six non-blocking items) as prospective rules. It changes no registered effect size (MMLU δ=.10, KL δ=.01, WikiText δ=log 1.05, θ=.15) and widens no δ. Where the wording below is a quotation of the Board it is verbatim. Disclosure: the 20-entry development panel was found to fail the A1.3 per-entry reading *before* this amendment was written; the "interim passes" below were obtained after that failure was known, are reported as interim, and do not qualify the endpoint. This amendment starts no compute by itself; the scoring it permits is listed in A2.4–A2.5 and governed by the gates there.

**A2.1 Item-uncertainty condition for MMLU-Pro gold log-prob and WikiText log-PPL (replaces the first clause of A1.3; the rest of A1.3 and the repeat and reliability bounds stand).**

> For panel entry i let h_i be its 95% half-width from the joint item draw (2.5/97.5 percentiles, 4,000 draws, seed 20261105, one common draw across edits and the shared reference; WikiText blocks of 2 windows). The endpoint satisfies the item-uncertainty condition iff all of the following hold on the qualification panel:
> (i) *panel mean:* the 97.5th percentile of the 4,000-draw component-bootstrap of mean_i h_i is ≤ δ/2 (unchanged from A1.3);
> (ii) *ceiling:* max_i h_i ≤ δ;
> (iii) *tripwire:* P ≤ 0.20, where P = Σ_f w_f p_f, f ∈ {lora_sft, task_vector, chain}, p_f is the share of family-f panel entries with h_i > δ/2, and w_f is family f's share of stratum rows in the sealed partition (metadata only; chain 161/241 = 0.668, lora_sft 55/241 = 0.228, task_vector 25/241 = 0.104 at the 100-component partition, recomputed from the final manifest). For the registered qualification P is the 97.5th percentile of the family-stratified component bootstrap; for interim reports it is the point value;
> (iv) *disclosure:* every entry with h_i > δ/2 is listed with y_i, h_i and h_i/|y_i|; every H1a result is reported with and without those rows (descriptive leave-out sensitivity), under both the component-only and the joint interval.
> Failure of (i), (ii) or (iii) leaves the endpoint unqualified (H1a untested). The A1.3 per-entry result (every entry ≤ δ/2) is still computed and reported descriptively. **The registered qualification panel is the stratum-family D3 entries (lora_sft, task_vector, chain) scored under the final `source_hash`, with the §6 repeat bounds from the repeated G1**, the reliability bound recomputed with the nuisance refit inside each bootstrap draw. The 20-entry development panel is an interim report and does not qualify the endpoint under A2.1.
> **Fresh-stage audit.** After scoring, the same half-widths and P are computed on the sealed items for the fresh rows. If P_fresh > 0.20 the item-population claim is withdrawn and only the conditional (fixed-item) claim stands, labelled as such. This audit cannot create or remove qualification.

Weights: the final manifest (106 components, 267 stratum rows) gives w = chain 187/267 = 0.700, lora_sft 55/267 = 0.206, task_vector 25/267 = 0.094; the qualification code recomputes them from the manifest. The 20% is the A1.2 constant, reused so that no second threshold is tuned after seeing a 10% exceedance rate; (ii) and the leave-out sensitivity in (iv) carry the weight, not the tripwire. Interim, not qualification (20-entry panel, 100-component weights): (i) 0.0312 ≤ 0.050, (ii) max 0.0893 ≤ 0.100, (iii) P = 0.076 ≤ 0.20. The registered qualification is **re-run on the 229 D3 stratum entries (139 chain, 60 lora_sft, 30 task_vector) under the final source**. Evidence that reverses A2.1 to the strict A1.3 reading (endpoint unqualified, H1a untested): on that re-run, panel-mean bound > δ/2, any h_i > δ, P (upper 97.5%) > 0.20, reliability lower bound < 0.90 with the refit nuisance, any repeat discrepancy > 1e-4, or exceedances that are not confined to large-effect LoRAs (a scorer problem rather than a property of the edit).

**A2.2 KL (supersedes the A1.2 role; the A1.2 outcome stands).**

> KL(base‖edited) is a descriptive secondary. Its A1.2 qualification outcome (unqualified, tripwire 25% > 20%) is reported as recorded. KL carries no pass/fail claim anywhere and is reported per entry with absolute and relative (half-width/KL) uncertainty. KL, task change and any other outcome must not be used to include, exclude or reweight rows of the H1a panel; any destructive-edit sensitivity analysis is descriptive.

No redefinition and no further qualification effort. The stage stays in the scoring path (7.7 s per entry, verified cache, the only validity signal for destroyed models). The KL cache root `2ed18bda284b3a89…` must be unchanged by the A2.4 source change.

**A2.3 Status rule and interval procedure for H1a; expansion to 106 components.**

H1a is registered **exploratory (power-limited)** by default. It is upgraded to confirmatory iff the following are all met in a committed development-only report **before any fresh component is scored**; otherwise it stays exploratory and the §4 pass/kill labels are not applied (estimates and intervals only, wording "estimated" or "not distinguishable", no kill claim).

> *Gate (all required):* (a) on the D3 out-of-fold TF-v1 losses (empirical comparator loss, empirical within-component and shared-item covariance, actual component sizes) the lower 95% binomial power bound at θ = 0.15 is ≥ 0.80 for **both** the conditional and the joint component/item-block procedure, using the interval procedure below; (b) under the null (θ = 0), one-sided rejection ≤ 0.035 and, at θ = 0.15, two-sided 95% coverage ≥ 0.93, in the empirical scenario and in the variance-inflated sensitivity scenarios, for both procedures; (c) fewer than 0.1% of draws with a denominator failure; (d) qualification (A2.1) passes; (e) the report also gives P(decision pass) at true θ = 0.15, 0.20 and 0.25.
> *Primary interval (replaces the §4 percentile bootstrap):* cluster **bootstrap-t on ln R**, R = ΣA/ΣB over rows, resampling source/ancestry components with replacement at the original count, 4,000 draws, seed 20261105. Per replicate compute t\* = (ln R\* − ln R̂)/se\*, with se\* the linearised cluster-robust standard error computed on that replicate: with component influence u_c = A_c/ΣA − B_c/ΣB, se² = n/(n−1) Σ_c u_c². The interval for ln R is [ln R̂ − q₀.₉₇₅(t\*)·se, ln R̂ − q₀.₀₂₅(t\*)·se]; the θ interval is 1 − exp of its endpoints, reversed. The joint procedure uses the same construction with the two-way (component × sealed item-block) cluster-robust standard error V = V_comp + V_item − V_both. Reported alongside, not decisive: BCa (acceleration from the delete-one-component jackknife) and the registered percentile interval.
> *Fallback order, fixed now:* if the primary fails (b) in the simulation, BCa is tried against the same criterion; if neither passes, H1a is exploratory. The choice is made on simulated data only, never on fresh outcomes.

The §5 gate is binding on the D3 OOF losses after D3 is scored and fitted, with the joint procedure; its interval procedure is the one above, not the percentile bootstrap. A CPU re-run of the interval procedures on the existing proxy grid precedes D3 scoring only as a courtesy to the owner (if neither procedure controls null rejection at ≤ 0.035 under the heavy-tailed scenarios, expect exploratory status). The gate report also gives n_eff = (ΣB_c)²/ΣB_c² and the top-five-component share of ΣB (diagnostics, not thresholds), and states in the limitation language that the D3 fit uses the 229 stratum rows plus the 160 null-family controls, a small training set for the StatNN-versus-isotonic comparison. §5's "power at 0.15" is the power of the rejection test, not of the §4 decision rule (at a true θ exactly at the threshold P(θ̂ ≥ 0.15) ≈ ½), hence item (e).

*Expansion to 106.* The six fresh chain components that remained unselected (26 stratum rows, 26 unique deltas, in hash order after the 100th) are added by the §5 outcome-free closure expansion, for floor margin (a single post-freeze exclusion no longer breaks the ≥ 100 valid-component floor); the floor stays 100. Re-sealed partition: 106 confirmatory components, 267 stratum rows (chain 187, lora_sft 55, task_vector 25), 305 closure entries, 281 unique deltas; the six spares are listed in the manifest as `spare_component_ids`. The stable seal is **`partition_sha256` = `ebf0379a15d95b592ee8c227836406f3365a56a3ef46e6978d369c2d42c0ff3f`** (it hashes the component→status map of all 483 components; it supersedes the 100-component seal `df4689983f1b…`, and only the six spares changed status, fresh_unselected → confirmatory; all other component ids, ranks, members and the 101-group alias ledger are unchanged). `manifest_content_sha256` includes `registration.git_head`, changes with every commit and is not the seal.

**A2.4 Batching and the pilot gate (replaces A1.5's "optional, not implemented"; cost decision).**

> *Source change.* HellaSwag endings (4) and the 13 refusal/compliance phrases per prompt are scored in one right-padded forward per item or prompt, with the actual batch recorded in each record. No edit to the letter scorer, `kl_values`, the WikiText path, or the four `SCORER_HASH` functions (`last_logits`, `generate_entry`, `entry_digest`, `content_root`), so the KL cache stays valid (root `2ed18bda284b3a89…` must be unchanged; no 5.4 GiB rebuild). Tests: batched-versus-batch-1 deviation within the probe maxima (HellaSwag ≤ 0.010, phrase log-prob ≤ 0.39, margin ≤ 0.30), bitwise repeat of the batched path, and a diff check that the primary paths are untouched.
> *Freeze.* Commit, record the new `source_hash` in A2, generate no reference before the pilot. Artefacts under `12ee875a68c9` are development-only and are not mixed with new labels (provenance requires the same `source_hash`). The 23 entries already scored are re-scored under the new source by the pilot and G1.
> *New warm timing pilot* (base plus the 10 D3 controls, one process). Pass: warm median ≤ **1.40 min**; MMLU-Pro, KL and WikiText records bitwise-equal to the old primaries (run `20261105-144427-fb58ffe6`); HellaSwag and phrase deviations within the probe limits.
> *G1 repeat* on the §6 composition (13 entries plus base, primary and repeat in fresh processes). Pass: max discrepancy ≤ 1e-4 (KL 1e-5), expected exactly 0.
> *G2 re-run* on the final source; the bridge script calls the final batched HellaSwag path. The registered G2 criteria concern MMLU-Pro only and are expected to reproduce unchanged. The worst entry's argmax agreement is exactly 99% against a 99% limit (a bf16 tie, granularity 1/100); **a one-item shortfall is investigated, not automatically a stop.**
> *Fallbacks, in order, if the pilot gate fails or any unchanged path is not bitwise-equal:* (1) drop the refusal and over-refusal stages by config; (2) additionally subsample null-family D3 rows: abliteration 100 → 20 and random_dir 60 → 30 (110 entries), hash-ordered with the `20261105:inner:` domain, no lora_sft/task_vector/chain row removed. (3) §9: stop with measurement results.

Projected total ≈ 15.5 h of the 19.0 h cap with the six spares (613 entries × 1.08 min = 11.0 h; consumed 2.5 h; new pilot, G1, G2 ≈ 1.0 h; overhead ≈ 0.5 h; spares 0.47 h). Break-even warm median for 19.0 h with no slack is 1.47 min, so the 1.40 min gate leaves ≈ 0.9 h of retry slack at the line.

**A2.5 The six non-blocking items (all closed here).**

1. *Alias fan-out.* For 54 of the 101 alias groups the canonical (scored) id is a standalone random_dir/abliteration, not the chain step 0 that is the stratum row. Scored id = canonical id; the reference of a chain entry is the scored id of its predecessor; every label/reference lookup resolves through the manifest `alias_ledger`. Implemented in `scripts/tfv1_scoring_plan.py` (`canonical_map`, `scored_id`, `reference_of`), with a test (`tests/test_tf_v1_prepanel.py`) and a dry-run that confirms **381 unique D3 deltas, 255 fresh deltas (281 with the six spares), each scored once**.
2. *Seal on the right hash.* `partition_sha256` (above) is the seal; `manifest_content_sha256` is not.
3. *Task-vector reference convention (frozen in the manifest `reference_convention`).* Only chain entries are predecessor-relative; lora_sft, task_vector, abliteration and random_dir are relative to the matched base. A task vector's stored delta is the full scaled delta from the base (the parent is the *source* of the vector, not a predecessor model state), so no unevaluated parent needs scoring and the +0.25 h of parent scoring is not spent.
4. *Sealed item panel and secondary panels.* The sealed MMLU-Pro item panel (`TF-v1:item:` domain, text-group deduplication, disjointness) is **not built by this amendment** and no sealed item may be scored until its manifest is committed. The secondary panels (HellaSwag 250, KL 200 prompts × 64 tokens, WikiText 64 windows, refusal/over-refusal 50 + 50) **keep the development panels as descriptive endpoints for fresh rows too**, so no new KL cache is needed; the dev-panel qualification does not cover the fresh MMLU rows (hence the fresh-stage audit in A2.1).
5. *Qualification-script limits for the re-run.* The 20-entry qualification (a) did not refit the nuisance regression inside each bootstrap draw (in-sample residuals, 6 parameters on 20 points), (b) used the cumulative rather than the incremental norm, and (c) expanded the §6 composition by the ten D3 controls rather than the prespecified hash order. All three are fixed by the D3-scale re-run (`scripts/tfv1_requalify.py`: refit per draw, incremental effective norm, the panel is the full stratum so no ad hoc expansion), and are stated as limits of the interim report.
6. *Power script limits.* The proxy resamples from 20 in-sample residuals with a CV of 1.97 and lacks the joint procedure; the proxy re-run and the D3-OOF gate (A2.3) replace it.

A2 is recorded on branch `feat/tf-v1-batching`; the post-batching `source_hash` and the measured pilot/G1/G2 results are appended to RESULTS ("TF-v1 pilot round 3", "G1 (round 3)", "G2 (round 3)") and referenced in A2.6 below once measured.

**A2.6 Source freeze and execution record (5 November 2026, evening; recorded before any D3 label was produced under the new source).**

- **New `source_hash`: `a224779bf873d96c92be7f70287b017db9139f487c039bb3fcf0d3b74b2fe178`** (round-2 source `12ee875a68c9…`, whose artefacts are development-only). Branch `feat/tf-v1-batching`, source commit `dd2a609`; `src/wsp` has not changed since (the hash is a function of `src/wsp/**/*.py` only). The KL cache identity (`SCORER_HASH` `7c42ea73fa33…`) and content root `2ed18bda284b3a89…` are unchanged; the primary scoring paths are pinned by `tests/test_tf_v1.py::test_primary_scoring_paths_are_untouched_by_batching`.
- **Pilot round 3 (gate A2.4):** warm median **1.003 min** (range 0.994–1.040) against the 1.40 min line: PASS; MMLU-Pro, KL and WikiText records bitwise identical to run `20261105-144427-fb58ffe6` for all 11 entries; recorded batch 4 (HellaSwag) and 13 (phrases) in every record. **G1 (round 3): 14 of 14 repeats bitwise identical** (maxima exactly 0). **G2 (round 3): PASS**, unchanged (MMLU-Pro bridge records bitwise equal to the round-2 bridge; worst-entry argmax agreement exactly 99%). Details: RESULTS "TF-v1 pilot round 3", "G1 (round 3)", "G2 (round 3)".
- **One A2.4 clause is not met and is disclosed, not repaired.** A2.4 lists "HellaSwag and phrase deviations within the probe limits" among the pilot's pass conditions. The probe maxima were measured on the unedited base alone; over the 11 pilot records the worst batched-versus-batch-1 deviations are HellaSwag 0.0148 (limit 0.010), phrase log-prob 0.526 (0.39) and margin 0.371 (0.30), i.e. 1.1–1.5 × the limits, one to two bf16 logit steps on edited models. These stages are descriptive (HellaSwag 250; uncalibrated refusal margins) and never enter H1a. The execution gate (warm median and bitwise-identical MMLU-Pro/KL/WikiText) passed, so D3 scoring was queued; Board A may accept the exceedance or invoke fallback 1 (drop the refusal stages by config) or a revert for HellaSwag, and the cost of acting later is a D3 re-score of the affected config.
- **Scoring queue.** `agent-tmux tfv1-d3-stratum` scores the 229 D3 stratum entries (phase A) then the null-family D3 controls the §3 fit needs (phase B; 381 unique D3 deltas in all, minus the 23 already scored by the pilot and G1). **No fresh component is scored, including the six spares** (fresh scoring starts only after the A2.3 gate report is committed). `scripts/tfv1_requalify.py` runs A2.1 on the stratum entries and the A2.3 gate on the D3 OOF losses once phase A (and, for the §3 fit, phase B) is complete.

### A3 (pending Board A, 5 Nov 2026 23:00 UTC) — source change after the A2 freeze
`fix/ara-delta-keys` (merged on main) adds a load-time delta key mapping (`language_model_prefix_v1`) in
src/wsp/zoo/common.py, used by the evaluation apply path and feature extraction, so the 24 heretic_ara deltas load.
It changes `source_hash`. The D3 stratum scoring continues on the frozen A2 source `a224779bf873` from its worktree.
Before any fresh confirmatory scoring under the new source, the A2.4 bitwise checks (MMLU-Pro, KL, WikiText on the
11 pilot records) must be repeated and recorded; non-prefixed deltas take an identity mapping, so identity is expected.

### A4 — 6 November 2026 (H1w; committed before any WikiText out-of-fold loss, θ̂ or interval is computed and before any fresh outcome)

Status: transcribes Board A's adjudication ([board-a-requalification-adjudication-2026-11-06.md](reviews/P5/board-a-requalification-adjudication-2026-11-06.md), decisions 1–5, §1b) as a prospective amendment. Owner decision recorded here: decision (c) is accepted (no sealed MMLU-Pro item is built or scored; the sealed partition `ebf0379a15d9…` stays unspent) and A4 is accepted. H1w is a **separately named exploratory claim**; it is **not a replacement for H1a**, and it is not a confirmatory claim unless the A4.5 gate is met. All six intervals are reported and none is decisive (A4.4). This amendment authorises no GPU compute. The wording below is Board A's, verbatim.

> **A4 — H1w, an exploratory claim on the qualified WikiText endpoint (6 November 2026; precedes any WikiText out-of-fold loss, θ̂ or interval, and any fresh outcome).**
> **A4.1** *H1a status.* On the 229 D3 stratum entries under `source_hash` a224779bf873 the MMLU-Pro gold log-prob endpoint fails A2.1 (ii) (max h 0.1137 > 0.10) and (iii) (P upper 97.5% 0.2264 > 0.20). H1a is recorded as **untested**. A2.1, δ = 0.10, the §1 wording and the §4 decision rule are unchanged and are not re-applied to any other endpoint. No sealed MMLU-Pro item is built or scored by this amendment.
> **A4.2** *Claim.* "On the development stratum {lora_sft, task_vector, chain}, estimate the row-weighted relative residual MAE reduction of fixed StatNN versus training-fitted isotonic norm, after training-only family, device and log-norm adjustment, for the predecessor-relative change in token-weighted WikiText log-perplexity (nats; 64 windows × 1,024 tokens, pinned revision)." It is a conditional predictive-loss contrast about generic distributional sensitivity. It is not task-likelihood readability, is not H1a, and may not be cited as H1a (§6 stands).
> **A4.3** *Pipeline.* Identical to §§3–4 and A2.3 with y replaced by the WikiText paired mean (same folds, seeds, nuisance, row-summed θ, denominator tolerance, 157/160 null controls with the three legacy exclusions listed). Mandatory descriptive companions: family×spline(norm) flexible nuisance; equal-component-weight ratio; θ by family and by incremental-norm tertile (weight-only strata); leave-out of any row with h_i > δ/2 (expected none).
> **A4.4** *Intervals.* Report conditional and joint intervals by bootstrap-t (A2.3 primary), BCa and percentile, each next to its simulated null size and coverage. No interval is decisive. Wording rule: "distinguishable from zero" only if the lower bounds of all six are > 0; otherwise "not distinguishable"; no pass/kill label and no "kill" language.
> **A4.5** *Gate and status.* Re-run the A2.3 gate (CPU only, registered scale) on the WikiText D3 OOF losses, with qualification (A2.1) already met. H1w is **exploratory** unless (a)–(e) are all met with the A2.3 primary or fallback. Fresh scoring of H1w requires, before any scoring, a committed sealed WikiText window panel disjoint from the 64 development windows and the A3 bitwise re-check on the post-A3 source.
> **A4.6** *Disclosure.* H1w is proposed because WikiText qualified and MMLU did not, with the MMLU D3 θ̂ = 0.129 known. The WikiText D3 θ̂ was not computed before this amendment.

Implementation notes (no change to the wording above): (i) the WikiText gate is `scripts/tfv1_requalify.py --endpoint wikitext_log_ppl`; WikiText windows are resampled in blocks of two (A2.1), so the joint procedure resamples the 32 two-window blocks, each block value being the mean of its two windows (which leaves the paired mean unchanged); (ii) the 157 null-family controls are the 160 registered controls minus the three legacy rows already excluded for the MMLU fit (the `157/160` of A4.3); (iii) the H1a status, the A2.3 gate and the §4 rule for MMLU-Pro are not re-run or relaxed by anything above; (iv) the Board A "reversal evidence" (§3 of its adjudication) is not a rule of this amendment, it only says what would make a fresh WikiText run worth ≈ 1.9 GPU-h (derived, not measured) — that run still needs the A4.5 sealed-window panel and the A3 re-check first, and is not authorised here.

### Appendix A5 — oriented local response (H1-OR)

Dated amendment: **7 Nov 2026**. **Draft for Board A; not an execution freeze.**
This appendix repairs no historical registration or result retrospectively. It starts no compute.

Authority and citation key:
[S31](council/2026-11-07/SYNTHESIS.md) means chair synthesis version 3.1, §§3–6.
[BA3](reviews/council/board-a-synthesis-v3-2026-11-07.md) means Board A's required edits and twelve A5 obligations.
[BC2](reviews/council/board-c-synthesis-v2-2026-11-07.md) means Board C's required edits 1–3.
Verbatim rule blocks below cite their S31 section for every copied quantity.
All new numerical choices are planned criteria, not measurements.

#### A5.1 Scope and frozen-hash list

**A5.1.1 — BA3 obligation 1: version and freeze.** H1-OR is exploratory; the field is the sole primary arm.
The chart cannot be promoted after outcomes (S31 §§2, 4.2).
WikiText tests sensitivity; MMLU is descriptive until qualified.
H1a stays “untested”; H1w stays “inconclusive” (S31 §§1, 5).
Report success as “weights plus base sensitivity” or “weights plus predecessor”.
The project's original claim, prediction from the weights and the edit alone, stays untested (S31 verdict).

Before G2 computes its first field, Board A must accept and hash this registration and a completed freeze manifest.
That manifest pins:

- Field, chart, rotation, intervention, extraction, inference and critic-adapter code, dependencies, source commit and schema.
- Arm list, thresholds, ordered decisions, tree/head settings, transformations, tuning budgets, nuisance, folds and seed averaging.
- Model/tokenizer revisions, effective/raw snapshots, exact support, component closure, aliases, predecessors, references and exclusions.
- Scored/gradient/factor/probe panels, text groups, windows, qualification runs, cache-input specifications, projector/control identities and seeds.
- Device, dtype, actual batch/order, A3 source-change qualification, resource caps, reference reuse and retry accounting.

Missing hashes or manifests block execution; a declaration of future freezing is not a freeze (S31 §§3, 5–6; BA3 obligation 1).
This draft supplies the A5.3.2 simulation companion, but does not certify selected IDs, governed field implementations,
panel certification or measured resource audits.
Those evidence obligations remain blocking until committed without using governed outcomes.
A5.11 requires the pre-G2 companion package in `docs/prereg/a5_manifests/`, with each file listed here by sha256 before G2;
this draft has no such inventory and is not frozen. Cache-content hashes and resource measurements generated by G2 are committed
before downstream use and before G3, rather than required before the gate that creates them.
The procedural freeze in A5.3.2 fixes code/schema before G0/G1; development residual hashes follow closure, and
label-free development field predictions/hashes follow G2 before G3. Neither generated input is a pre-procedural-freeze value.
MMLU promotion needs a new prospective endpoint/arm family, multiplicity and joint decision rule before new endpoint outcomes (BA3 obligation 1).

#### A5.2 Exposure ledger, G0–G8

**A5.2.1 — BA3 obligation 3: source and item manifests.** Record every use, including failures, retries and qualification.
Join ancestry, aliases, shared adapters/checkpoints/training runs, task-vector operands and generated direction identities transitively.
Generic dataset and immutable base sharing alone remain support diagnostics (§2).
Each ledger entry records canonical ID, connected component, aliases, role, time, predecessor, strength, fold, text groups and hashes.
Pin actual predecessor, device/dtype/batch and matched source-change qualification before affected work.
Auxiliary gradient, factor and probe panels must be text-group disjoint from development, calibration and sealed scored items (S31 §3).

| Gate | Exposure role and accounting |
|---|---|
| G0 | Replay, installation, features, fingerprint rerun, qualification and full connected closure. |
| G1 | Scoring, aliases, shared references, qualification and missing-record recovery. |
| G2 | Field/factor estimation, chart construction, probe caching and predecessor audit. |
| G3 | Every selected component, original, rotation, ladder and predecessor; exclusions stay exposed. |
| G4 | Fitting, validation, tuning, retrospective outcomes and all fitted transforms. |
| G5 | Pilot source/support/episode/direction/prompt groups; invalid and rejected attempts included. |
| G6 | Closure census and connected relatives; all scored rows are exposed. |
| G7 | Qualification itself opens sealed outcomes; log every subsequent sealed use. |
| G8 | Separate calibration/evaluation groups; live feedback cannot refit the frozen critic. |

Development exposure at any G0–G6 gate bars full closure from G7 freshness (S31 §3).
Regeneration, rerunning or relabeling never restores freshness.
G7/G8 exposed components cannot become fresh in a later confirmation.

**A5.2.2 — BA3 obligation 12: canonical counts and costs.** Commit delta → logical row → alias → actual-reference mapping.
Prove reuse for the eight alias records by complete scoring identity, not equal scalar values (S31 §3 G1).
S31 §3 G1 gives 316+62=378 scoring entries and 386 logical rows; without dependent rows, 324 logical rows.
Enumerate the two lost predecessors, fourteen missing records, G0/G1 overlap and all retries (S31 §3 G0–G1).
No reuse certification is asserted here.
Random-direction and abliteration replay must be bitwise.
LoRA replay needs matrix relative Frobenius error ≤5e-3; label retrained descendants and exclude them from fingerprints (S31 §3 G0).
Norm-only pilot differences do not establish matrix-error compliance (BA3 N1).
Retain 29.55 GPU-h allocation, or 28.65 excluding the CPU reserve; these are not measurements (BA3 obligation 12).
G7 scoring and overhead remain separate; G8 pricing comes from G5.
Before G3, measure/price fields, sequence counts, CPU/disk bytes, peak VRAM, throughput, rotations, audits and references (BC2 edit 2).
Exceeding caps stops and reprices, never silently cuts coverage (S31 §3).

#### A5.3 Part A

The following block is verbatim S31 §4.1; that source supplies every copied number.

*Analysis time.* One analysis, after all 120 variant scores and their references are on file and the ledger is closed.

*Null and claim.* The null is that the sign of ⟨g, D − R⟩ agrees with the sign of y(W + D) − y(W + R) with probability ½,
conditional on the frozen field and the fixed 64-window panel. The claim is agreement above chance under that conditioning; an
item-population claim is not made. The constant-sign comparator ("every trained edit degrades more than its rotation") is
reported with the same interval; it is not a registered comparison and no superiority over it is claimed.

*Unit and interval.* One pair per certified source component (40 components). Two-sided 95 % Clopper–Pearson interval on the
agreement count among resolved pairs. Lower bound > 0.5 requires at least 12 of 15 (lower bound 0.5191) or 27 of 40 (0.5087);
Board A's exact tails. A zero prediction counts as a disagreement. A missing or audit-excluded score removes the pair from the
resolved denominator and is listed in the full-panel ledger.

*Resolution.* d_i = y(W_i + D_i) − y(W_i + R_i) on the same windows. Its interval comes from a paired block bootstrap over the
32 two-window blocks with common draws across the edit, its rotation and the predecessor reference (fixed seed, 4,000 draws,
percentile interval). A pair is **resolved** if that interval excludes zero. The agreement estimand is conditional on resolution.
All 40 pairs are reported. The null size with the resolution step included is validated on a development-only simulation before
outcomes are read; if the simulated size exceeds 0.07 at nominal 0.05, the rule is unresolved and A5 is amended before scoring.

*Yield.* Fewer than 15 resolved pairs is **unresolved**: no replacement, no resampling. Board A's illustration: at an independent
resolution probability of 0.375 the chance of ≥ 15 resolved among 40 is ≈ 0.56. Yield and agreement power are simulated from
development contrast and noise assumptions before this purchase is called decisive.

*Go.* Part A **goes** when the rotation contrast has ≥ 15 resolved pairs and its lower bound exceeds 0.5; every other outcome is
listed in section 4.0, step 3.

*Ladder.* The contrast is y(W_i + s·Δ_i) − y(W_i) at s = 0.25 and s = 0.5, predicted sign from ⟨g, bf16(W_i + s·Δ_i) − W_i⟩,
same resolution rule, same interval, separate denominators per strength, reported as descriptive except for the alternative rule
of section 4.0, step 2 (upper bound ≤ 0.6 at s = 0.25 on ≥ 15 resolved pairs), which overrides a rotation go. A lower bound
below 0.5 alone is uninformative and triggers nothing. A pass at s = 0.25 with a fail at s ≥ 1 is a pattern to investigate, not a
conclusion.

*Rotation.* Per edited tensor t, the incremental effective edit is the realised bf16 difference D_t = bf16(W_t + Δ_t) − W_t at the
actual predecessor, subtracted in fp32 from the two stored bf16 states. Take the rectangular coordinate block of changed rows and
columns and compute the thin SVD of D_t on that block in fp32 (full SVD of the realised block; the raw LoRA factors may be used
only if they reproduce the realised block to a relative Frobenius error ≤ 1e-3, checked per tensor). Draw Q_out and Q_in as
Haar-orthogonal matrices on that block from a registered seed (one seed per edit, shared across its tensors so that a direction
common to several layers is rotated coherently; this choice is recorded). Form R_t = Q_out U Σ Vᵀ Q_inᵀ, apply W_t + R_t and
round to bf16. The scored object is bf16(W + R) − W. Preserved before rounding: singular values, Frobenius norm, rank,
confinement to the block, tensor names and shapes. Not preserved, and logged: coordinate moments, sign fractions, cosine to the
predecessor, directional tokens, and anything after rounding. A rotated LoRA is not an SFT-trained edit; "spectrum-matched"
refers to the pre-rounding construction. Post-rounding audit per tensor and per edit: relative Frobenius norm mismatch
|‖R_eff‖ − ‖D_eff‖| / ‖D_eff‖ ≤ 2e-2; relative spectrum mismatch (Frobenius distance of the sorted singular values) ≤ 5e-2; rank
within ±1 at relative tolerance 1e-3; block support identical. A failing pair is excluded by this rule, and the count rule is
"resolved among non-excluded pairs". Non-2D tensors, tensors with a zero or rank-degenerate block, and chain increments that are
not single-matrix edits are excluded from rotation and listed; coverage is reported per edit. Scaled edits are built by applying
s·Δ_t at W_t and rounding, not by scaling D_t. A nuisance-controlled sensitivity (logged v4 differences as covariates) is reported
next to the main contrast.

*Exposure.* D3 outcomes for the 40 edits are already known; selection is "before any new variant outcome", and Part A is
exploratory.

**A5.3.1 — BA3 obligation 4: selection and accounting.** Use metadata/effective norms only, before new variant outcome access.
The following deterministic algorithm fixes global deduplication before cell assignment. A component spanning families or
tertiles is represented only by its lexicographically smallest eligible canonical edit ID, irrespective of its other rows.
Eligibility uses certified closure, supported family and pre-outcome feature-manifest availability, never response scores.
Effective norm is the manifest's Frobenius norm of the full realised incremental edit, not the raw adapter norm.

```text
rows = sort(eligible_metadata_rows, key=canonical_edit_id)
representatives = first row for each certified_source_component in rows
for family in [LoRA, chain]:
    ordered = sort(representatives in family, key=(feature_manifest.effective_norm, canonical_edit_id))
    N = len(ordered); b1 = floor(N/3); b2 = floor(2*N/3)
    cells = [ordered[0:b1], ordered[b1:b2], ordered[b2:N]]  # zero-based, half-open
    quotas = [10,10,10] if family == LoRA else [4,3,3]
    for cell, quota in zip(cells, quotas):
        if len(cell) < quota: STOP unresolved; do not replace across cells
        selected += sort(cell, key=canonical_edit_id)[0:quota]
assert len(selected) == 40 and len(unique(selected.component_id)) == 40
write selection manifest, ranks, boundaries, rejected rows and closure certificates
```

Nonfinite/missing manifest norms or uncertified sources block selection rather than being silently dropped.
Freeze the eligible inventory and its exclusion ledger before running this algorithm.
Planned low/middle/high allocations are LoRA 10/10/10 and chain 4/3/3.
These allocations are **chair default, Board A may change before freeze**; family totals follow S31 §3 G3.
No cross-cell replacement; a shortage stops selection.
Commit the forty actual IDs, family/tertile allocation, closure proofs, tensor coverage and outcome-independent exclusion reasons (BA3 obligation 4).
Their enumeration is outstanding; this draft cannot certify it without data access.
No G3 launch is permitted without it.

Keep all forty in the ledger (S31 §4.1).
Missing/nonfinite scores and audit exclusions remove only that contrast from its resolved denominator, with explicit reasons.
Use separate rotation and ladder-strength denominators; no borrowing between strengths.
An interval touching zero is unresolved; a zero prediction is disagreement (S31 §4.1).
Close the ledger before the single analysis; restrict variant-outcome access until closure.

**A5.3.2 — BA3 obligation 5: inference and calibration.** CP means exact equal-tail, two-sided binomial intervals, including boundary cases.
Bootstrap percentiles use linear interpolation at 0.025/0.975.
PCG64 with seed 20261107 and that quantile convention are **chair default, Board A may change before freeze**.
The block count, draw count and CP examples are S31 §4.1's rules, not measurements.

Freeze A5's code, input schemas, selection inventory/cutpoints and configuration before exposure; this freezes the
algorithm, not field values that G2 has not yet produced. The mandatory order is
**freeze(A5) → G0/G1 closure → NULL-SIZE → G2 → POWER-AND-YIELD → G3**.
G1 TF-v1 development records supply the window-level paired residuals (edit, rotation and predecessor reference), norms,
certified source closure and audit/missingness flags after G0/G1 closure. Commit their actual inventory/hashes before
NULL-SIZE. No sealed or G3 outcomes supply noise, exclusions, effect sizes or tuning; missing inputs leave calibration
unresolved. The NULL-SIZE simulation runs before G2 and requires no field: under the null predicted sign is independent
of outcome, so use fair-coin signs. Its empirical size must be ≤0.07 at nominal 0.05, otherwise stop and amend before G2.
POWER-AND-YIELD runs after G2 computes the label-free field on development rows and before G3. Computing that field on
development edits before G3 is permitted. Add these development predictions and their generated hashes to the input package
then; do not require nonexistent field values pre-G2. Simulated ordered Part A power must be ≥0.80 at the registered
alternative, otherwise Part A is **not decisive** and a resizing amendment is required before G3. Report exact binomial
uncertainty, but the acceptance threshold applies to simulated power itself, not its lower confidence bound.

The companion artifact is `scripts/a5_simulation.py`, with synthetic checks in `tests/test_a5_simulation.py`.
It uses CPU/numpy only and reads only supplied files. A JSON input has `inventory` and `donors` row lists with
`id`, `component`, `family`, `norm`, optional `eligible` (default true); donor rows add `residuals[64,3]` in
[edit, rotation, predecessor] order and `audit[3]` in [rotation, 0.25, 0.5] order (default true).
NPZ may instead hold that same JSON as scalar Unicode `payload` (no pickle). The optional field JSON maps donor row IDs
to three finite predictions in that contrast order; missing predictions block POWER-AND-YIELD, zero predictions disagree.
Run `python scripts/a5_simulation.py INPUT --output SIZE.json` before G2 and add `--fields FIELD.json` after G2.
The latter also reproduces NULL-SIZE; its draws do not depend on field values. Registered runs use 10,000 replicates per
simulation, 32 two-window blocks and 4,000 paired resolution draws; API overrides exist only for synthetic tests.

Use A5.3.1 global deduplication and selection on the frozen inventory in every replicate. Freeze each family's two
rank cutpoints as the `(effective_norm, canonical_edit_id)` keys at ranks floor(N/3) and floor(2N/3); comparison at a
cutpoint belongs to the higher half-open cell. Donor membership reuses those cutpoints, never recomputes them on donors.
Within each matching (family, tertile) cell, a donor source cluster is represented by its lexicographically smallest
eligible canonical edit ID in that cell. Sort these representatives by ID, sample donor clusters with replacement for
selected slots, and retain all rows/arms/references of the cluster together for provenance. Only its representative row's
paired residual block becomes the slot response: never average its other rows. A cluster spanning cells can represent
each matching cell through its own eligible representative there. No donor cell or selection shortage permits replacement
across cells. Repeated donors represent distinct simulated component slots; each slot has one component sign draw.

Centre the input per window on the predecessor reference: U_edit = E_edit − E_reference and
U_rotation = E_rotation − E_reference. Rotation contrast is the sign-flipped paired difference
U_edit − U_rotation. Ladder contrasts under the null are the representative's U_edit block scaled by s = 0.25 or 0.5,
with that same sign flip. One independent fair sign per simulated component per replicate is shared across all three
contrasts; there is no panel-wide sign. Null prediction signs can therefore be fixed +1 without loss of generality.
For the registered alternative, retain the donor contrast's absolute mean and centred paired noise, replacing its mean
by abs(mean) × sign(field prediction) × B, where P(B=+1)=0.75. Draw one B per simulated component and share it across
all three contrasts, preserving their prediction-sign differences and noise covariance; zero predictions remain zero.

Derive exactly `numpy.random.SeedSequence(20261107).spawn(3)` and construct PCG64 generators in the fixed order
[panels, signs, resolution]. In each simulation/replicate, select donors in family/tertile/slot order, sample 32 common
block indices with replacement for the outer panel, draw the component signs, then draw a common [4000,32] array of
resolution indices. Resolution draws are shared across all slots/arms/contrasts; use paired bootstrap means and linearly
interpolated .025/.975 percentiles. Constant block arrays have exact point intervals and consume no resolution variates.
Apply donor audit and missing/nonfinite masks per contrast; no replacement. Resolved means the interval strictly excludes
zero. Count agreement with each contrast's prediction and compute exact equal-tail two-sided CP intervals separately.
Report each contrast's two-sided rejection size (CP excludes .5), positive-direction rejection, yield, conditional coverage
of the generating mean and failure counts, with exact binomial uncertainty for size/yield/power. Unresolved replicates
count as non-rejections. Apply the complete Part A ordered indicator: validity first; then ≥15 resolved at 0.25 and
CP upper ≤0.6 gives **alternative**, overriding rotation; otherwise ≥15 rotation pairs and CP lower >0.5 gives **go**;
otherwise **unresolved**. Report this whole decision's go frequency in addition to the individual tests; both individual
two-sided sizes and whole-decision error must be ≤0.07. This Part A simulation grants no Part B funding eligibility.
Save configuration, donor/mask/count/decision traces and input provenance in the calibration package. Planned sensitivity
runs use constant positive donor means (constant-sign family effect), cell-specific masks/noise (norm-dependent resolution),
and twice the centred noise with means unchanged (variance inflation); run these as separately labelled frozen scenarios,
not as substitutes for the registered alternative. Neither empirical calibration nor a power pass is asserted here.
The claim remains conditional on the frozen field and fixed panel; constant-sign comparison remains descriptive.

**A5.3.3 — BA3 obligation 6; BC2 edit 1: realised rotation and sensitivity.** Use fp32 subtraction of pinned stored bf16 states.
Use the full thin SVD of the realised block; do not truncate its spectrum.
A raw-factor substitute must first meet S31 §4.1's per-tensor reconstruction tolerance ≤1e-3; otherwise use the realised SVD.
Numerical rank counts singular values strictly above 1e-3 times the largest (S31 §4.1 rank tolerance).
Equality is excluded from numerical rank.
`rank_degenerate` means numerical rank zero at that audit tolerance; low rank is not an exclusion.
LoRA blocks of rank 4–32 on 1,024-wide blocks remain eligible. This resolves the ambiguous “rank-degenerate” wording in the
copied A5.3 rotation block under the chair-endorsed interpretation.
The construction is R = Q_out D_block Q_inᵀ = Q_out U Σ Vᵀ Q_inᵀ for any SVD of the realised block.
Repeated singular values and choice of singular basis are irrelevant; SVD serves the reconstruction/rank/spectrum audits.
This matches `src/wsp/features/rotation.py` on main at commit `88215cb`; pin the eventual code hash before G2.

Rows/columns are changed if any realised increment entry is exactly nonzero.
Block support means the allowed rectangle; report actual nonzero coordinate masks separately, including post-rounding changes.
Identical block support does not imply identical coordinate sparsity.
Any unsupported, non-2D (`non_2d`), nonfinite (`non_finite`), zero (`zero_block`) or numerical-rank-zero
(`rank_degenerate`) edited tensor excludes the whole rotation pair, rather than leaving it unrotated.
Record each reason separately; non-single-matrix objects use `not_single_matrix`, with chain coverage separately tabulated.
This edit-level aggregation is **chair default, Board A may change before freeze**, implementing S31 §4.1 and BC2 edit 1.
Chain eligibility requires a single-matrix incremental edit; list heterogeneous-chain exclusions and coverage.

Derive each edit's RNG seed from SHA256 of `20261107:rotation:` plus canonical ID (first eight bytes, big-endian,
reduced modulo 2^63). Initialise a torch generator and clone its initial state for each tensor, as `rotate_block` does.
Draw an out×out Gaussian array then an in×in array in that order; sign-corrected QR gives Q_out then Q_in.
Equal shapes share frames; unequal shapes consume that same stream at their own dimensions, without a nested-frame claim.
QR uses positive diagonal signs (zero diagonal sign replaced by +1).
This mapping is **chair default, Board A may change before freeze** (BA3 obligation 6).
Audit orthogonality and full-SVD reconstruction residual ≤1e-5 relative Frobenius error.
That residual tolerance is **chair default, Board A may change before freeze**.
Norm mismatch divides by ||D||; spectrum mismatch divides by the Euclidean norm of D's sorted singular values.
Zero denominators fail eligibility; all tensors must meet S31's norm/spectrum/rank/support criteria.
Report per-tensor audits plus maximum and pooled edit diagnostics; a single tensor failure excludes the pair.
Also report ||R_eff−R_raw||/||D||, reconstruction residual, exact support and unrotated-tensor count.
Spectrum matching is pre-rounding unless the realised audit supports it (BC2 edit 1).

For descriptive nuisance sensitivity, fit ridge logistic agreement on family, log1p effective norm and rotated-minus-original v4 differences.
Use train-only standardisation, penalty 1, no tuning: **chair default, Board A may change before freeze** (BA3 obligation 6).
This diagnoses matching confounds; it replaces neither CP inference nor the primary funding rule.

**A5.3.4 — references (BA3 obligations 4, 12; BC2 edit 3).** The launch table must enumerate the following objects by selected ID:

| Object | Reference/reuse rule |
|---|---|
| Original full-strength D | Reuse G1 only with identical effective snapshot, windows and scoring identity; otherwise score it. |
| Actual predecessor | Root base or pinned chain predecessor; deduplicate by complete scoring identity only. |
| Rotation and ladders | Three new variants per selected edit; S31 §3 G3's 120 entries. |

Each row lists original run hash, predecessor ID/hash, reusable record hash, missing-score flag and charged minutes.
Let n_D count missing original records and n_P count unique missing predecessors.
Additional scoring costs (n_D+n_P)×1.04 minutes; total costs (120+n_D+n_P)×1.04 minutes (S31 §3 G3).
All forty original records are required, even when no new full-strength scoring is needed (BC2 edit 3).
Exact predecessor IDs, n_D, n_P and reuse proofs remain outstanding launch blockers, not assumed zero costs.
Preprocessing and retries are charged separately under stop-and-reprice.

#### A5.4 Part B

The following block is verbatim S31 §4.2; that source supplies every copied number.

*Panel.* The G1 records, registered H1w folds, train-only nuisance (family, device, log-norm), seeds 0/1/2 averaged before the
loss, outer-held-out predictions audited for nested out-of-fold construction.

*Primary contrast (one).* C = 1 − A_cand / A_aug, where A is the row sum of held-out absolute residual errors after the same
training-only nuisance fit and seed averaging; cand = **StatNN + field tokens**; aug = **augmented StatNN**, the same tree with
the same settings, tuning budget and folds on identical inputs: v4 pooled statistics plus the deterministic per-tensor
transformation of section 6 (ordered exact-layer and role scalar columns of the chart and of the field, padded by a fixed rule),
so that the comparator receives every scalar the candidate receives. C therefore measures the candidate pipeline's performance
relative to a **same-information comparator**; it does not by itself identify the information the field adds. A5 must name the
actual difference between the two pipelines. **Field-information contrast (registered, decision-relevant):** C_abl = 1 −
A_cand / A_no-field, where the no-field arm is the identical pipeline (architecture, chart and predecessor access, rows, folds,
tuning opportunity) with the field columns removed; same interval. "Both destruction controls beaten" in section 4.0 is read as:
C_rand, C_perm **and** C_abl all have lower bound > 0. The random-field and permuted-field controls test particular destruction
hypotheses; the intact no-field ablation is what identifies the field's contribution. Interval: paired ancestry-cluster bootstrap-t on log(A_cand / A_aug), fixed
seed and draws, resampling whole components; conditional and joint item procedures as in PREREG A2.3, adapted to the A_aug
denominator; denominator failure (A_aug = 0 in a resample) is recorded and the draw dropped, with the count reported.

*Destruction controls.* C_rand = 1 − A_cand / A_rand-field and C_perm = 1 − A_cand / A_perm-field, same interval, alongside
C_abl above; "beaten" means lower bound > 0 on all three. *Secondary arms* (descriptive, no decision): zero-shot field (sign from ⟨g, ΔW⟩ only), ≤ 3-coefficient
field calibration, chart with its own destruction controls (permuted projectors; random-frame chart), A1-lite, descriptor-only,
delta-norm, original StatNN, the existing M1 and canonical LoRA tokens on the LoRA-only rows of the same folds (with their own
intervals and an inconclusive region), and the probe-response comparator (linear ΔW·x on a fixed scoring-disjoint 64-prompt
panel plus a regularised head; probe-only and probe-plus-descriptor controls; probe cost timed).

*Thresholds.* 0.15 is PREREG §4's θ threshold and 0.05 its §7 screen, fixed on 5 November before the H1w and MMLU estimates were
known. Their reuse for C is a chair choice made before any new outcome; A5 records it as such. The MMLU θ̂ of 0.129
(`docs/reviews/P5/h1a-companions-2026-11-06.md`) was known when this programme was designed; A5 discloses it.

*Decision.* Section 4.0, steps 4–5. A no-go stops funding and is not a scientific falsifier, because Part B reuses examined
outcomes and PREREG A4.4 forbids pass/kill language on this endpoint. There is no chart fallback in this programme. A chart-primary programme needs a new amendment written before new outcomes.

*Disclosure.* The D3 WikiText θ̂ and MMLU θ̂ were known before this design. The field and chart code, the arm list and the
thresholds are frozen by hash in A5 before G2 computes the first field.

**A5.4.1 — BA3 obligation 8: information contract.** Candidate and augmented tree receive the same underlying scalar information.
Candidate uses v4 plus masked within-role mean/sum/maximum token summaries; augmented tree uses the full ordered unpooled scalar columns.
Candidate summaries are deterministic functions of augmented inputs, including masks and tensor dimensions.
This pooling choice is **chair default, Board A may change before freeze**, resolving S31 §4.2 and BA3 E2.
C measures predictive use of supplied information; only the matched field-absent contrast identifies field contribution.
C_abl removes all field-derived columns and summaries, holding architecture, chart/predecessor access, rows, folds and tuning fixed.
No duplicate field proxy remains in that ablation.

Use H1w source-closed folds and nested OOF training targets, with §3's train-only OLS nuisance and preprocessing.
Nuisance includes intercept, family/device treatment dummies and log1p incremental effective norm; unsupported levels fail closed.
Average seed predictions before absolute loss; seeds are not independent observations (S31 §4.2; §3).
There is no tuning or early stopping; all pipelines have the same fixed fitting opportunity.

**A5.4.2 — BA3 obligation 9: ratio inference.** Let A_x=Σ_i |r_i−mean_s p_x,s(i)|, a row sum retaining whole components.
For each comparator, use L=log(A_cand/A_comparator).
Apply A2.3's bootstrap-t construction and component influences u_c=A_c/ΣA−B_c/ΣB.
Map [L_lo,L_hi] to [1−exp(L_hi),1−exp(L_lo)], reversing endpoints.
Conditional draws resample components; joint draws also resample common item blocks across rows and shared references.
Joint studentisation uses A2.3's V_comp+V_item−V_both.
Evaluation draws hold fitted nuisance/predictors fixed; qualification simulations refit nuisance as A2 requires.

Use 4,000 draws, PCG64 seed 20261107 and linear 0.025/0.975 quantiles.
These choices are **chair default, Board A may change before freeze**, completing S31 §4.2.
For the point estimate and every draw, n is the sampled logical-row count including component multiplicities.
Require finite A_cand and A_comparator, both positive, with A_x/n >1e-12 nats for each arm; equality at the guard is invalid.
Count and report nonpositive A values, guarded near-zero values and nonfinite values separately, by arm and procedure.
Equal positive finite losses above the guard give L=log(1)=0 and C=0: these are valid null draws.
A zero log ratio never invalidates a draw; a nonfinite log ratio or nonpositive/nonfinite studentising variance does.
At least two distinct certified source components are required in the point data and every resample; fewer fails validity.
The minimum usable count is all 4,000 draws: **chair default, Board A may change before freeze** (BA3 obligation 9).
Record failed draws; valid-only intervals remain diagnostics, never funding or scientific success evidence.
Do not retry, clamp losses, floor variances or replace invalid draws.
This strengthens S31's “drop and report” into an explicit unresolved action.
Apply identical numerical rules to C_abl, C_rand, C_perm and secondary comparator intervals.
Failure of C, C_abl, C_rand or C_perm blocks primary funding; a failed descriptive-only interval is labelled undefined
and reported with its failure count, without invalidating the otherwise defined primary family.

Funding needs both conditional and joint bootstrap-t lower bounds to pass, including all three controls.
No-go requires both upper bounds and adequate power for both procedures.
This interpretation is **chair default, Board A may change before freeze** (BA3 obligation 2).
A4.4 additionally governs all six conditional/joint bootstrap-t, BCa and percentile intervals and their scientific wording.
Simulate the actual augmented-denominator C, not isotonic θ, using A2.3's null calibration, coverage and failure gates.
Retain empirical paired-loss covariance, component sizes, shared-item dependence and conservative variance scenarios (§5; BA3 obligation 9).
No calibration, power or coverage result is asserted here.

**A5.4.3 — secondary comparators (BC2 edit 3).** M1 and canonical tokens use identical eligible LoRA-only rows and inherited folds.
Compare each with LoRA-only augmented StatNN under the same nuisance, seed averaging and ratio interval procedure.
Intervals including zero are inconclusive, never equivalence evidence; these arms cannot fund G7 (S31 §4.2).
Publish eligible-row exclusions before fitting.
Probe-response and probe-only/probe-plus-descriptor controls receive paired ancestry-cluster intervals and timed probe costs.
An interval including zero is inconclusive; no secondary promotion follows outcomes.

**A5.4.4 — disclosure (A4.6 style).** This design follows known D3 WikiText θ̂=0.0071 (S31 §1).
MMLU θ̂=0.129 was also known (S31 §4.2, citing the H1a companions review).
The inherited 0.15/0.05 thresholds concern another estimand; reusing them for C is chair choice (S31 §4.2).
Neither is a measured oriented-response effect.

#### A5.5 Ordered if/else decision function

**A5.5.1 — BA3 obligation 2: executable priority.** The following block is verbatim S31 §4.0.
Evaluate once in order; a terminating branch wins, and otherwise continue to the next step.
Invalid/undefined statistics, including A5.4.2 failures, enter validity first.

Equality conventions are stated per rule below: point thresholds written with ≥ or ≤ are inclusive; "lower bound > 0" and
"lower bound > 0.5" are strict. There is no blanket rule.

1. **Validity.** Missing manifests or hashes, a failed qualification, an exhausted allocation, or a development null-size
   simulation above 0.07 at nominal 0.05 → the gate is **unresolved**; nothing downstream runs. The G2 predecessor audit has one
   action: fewer than 16 of 20 passing predecessors is a **hard stop** of G2 pending a new amendment (exact-predecessor
   gradients may be registered there with their cost); it does not silently disable the approximation.
2. **Ladder alternative.** At s = 0.25, ≥ 15 resolved pairs and agreement upper bound ≤ 0.6 → **alternative**: the field is not
   pursued, no sealed purchase on the field arm, the probe-response comparator becomes development work only (it has no
   purchase criterion until a new amendment gives it one). This branch overrides a rotation go.
3. **Part A evidence.** Fewer than 15 resolved rotation pairs → **Part A unresolved**. Interval includes 0.5 → **Part A
   unresolved**. Interval entirely below 0.5 → **Part A unresolved**, with the observed disagreement reported (no scientific
   kill). Otherwise (≥ 15 pairs, lower bound > 0.5) → **Part A go**.
4. **Part B eligibility.** Part B is always computed and reported. It can fund only if Part A went and both destruction
   controls are eligible under A5.8.3 and beaten, and C_abl is valid and beaten (strict lower bound > 0 for each of
   C_rand, C_perm and C_abl in both conditional and joint intervals); otherwise the funding decision is **unresolved** regardless of C, and the
   extension-only exception below does not apply.
5. **Funding (if eligible), evaluated in this order:** if C ≥ 0.15 and lower 95 % > 0 → **go**; else if C ≥ 0.05 → **screen**
   (Part A extension only, never sealed rows); else if upper 95 % < 0.05 and simulated power at C = 0.15 ≥ 0.80 → **no-go**
   (funding stops; not a scientific falsifier); else → **unresolved**. Intervals are never assumed to contain their point
   estimate; the branches are evaluated as written.
Scientific consequences are stated separately from funding consequences in every report.

Part B reporting is always required once valid upstream inputs exist, even when Part A evidence is unresolved.
This does not override the validity hard stop or permit funding without Part A go.
Equality follows each displayed operator; touching zero fails a strict positive-bound rule.
Exactly fifteen resolved pairs meet the floor; exactly 0.6 meets the ladder upper-bound rule (S31 §4.0).
Exactly 0.15/0.05 meet point thresholds; upper bound exactly 0.05 does not meet no-go (S31 §4.0).
Exactly 0.80 meets the power threshold; audit count exactly sixteen passes (S31 §§3–4).
Funding, scientific wording and deployment decisions are reported separately (BA3 E1–E4).

Step 1 is the following validity checklist; each checked predicate must link its evidence in the gate report:

| Predicate (all applicable checks must pass) | Required evidence artefact |
|---|---|
| Pre-G2 registration accepted; identities/code/config pinned and hashes verified | A5.1 sha256 inventory, A5.11 manifests and Board A acceptance |
| Source closure, alias/reference reuse, panel disjointness and gate freshness certified | Closure/alias, text-group, reference and exposure manifests |
| Actual predecessor/device/dtype/batch and endpoint/source-change qualification pass | Matched qualification report with run IDs and scoring hashes |
| Allocation not exhausted; references, failures and retries fully charged | Cost ledger, measured resource report and reference inventory |
| G2 has ≥16/20 predecessor passes per objective at error ≤0.1 with 1e-3-nat floor; hooks/factors valid | Per-predecessor audit and field-schema audit; <16 hard-stops pending amendment |
| Selection complete and rotation/support audits accounted for | Forty-ID selection ledger, per-tensor audit/exclusion report |
| Development null size ≤0.07 at nominal 0.05; applicable power/precision/coverage gates pass | Hashed simulation code/config, input inventory and calibration report |
| Point and all 4,000 draws finite, valid variances, ≥2 distinct components and ratio guards pass | Per-contrast conditional/joint inference diagnostics and failed-draw counts |
| Both destruction arms pass their own eligibility checks; C/C_abl/C_rand/C_perm defined | Frozen control manifests and destruction/coverage audit (A5.8.3) |
| G7/G8 additional count, qualification, episode denominator and calibration gates pass when applicable | Fresh purchase/episode manifest, endpoint report and prospective sizing report |

A primary-family numerical failure or ineligible destruction arm makes funding unresolved; it does not suppress valid
Part B estimates. A failed descriptive-only arm is reported undefined without blocking a valid primary family.
The ≥15 resolved Part A floor belongs to step 3, not an invented independent-component sufficiency claim for Part B.
G7 additionally requires ≥100 fresh independent components; G8 requires ≥2 independent episode clusters and its frozen
G5-derived count. Distinct-cluster and all-draw validity checks also apply within their respective resamples.

#### A5.6 G7 sealed purchase

**A5.6.1 — BA3 obligation 10: sealed contract.** The following block is verbatim S31 §4.3.

Scientific success keeps the PREREG §§4–5 contract, re-stated for C on the new denominator: endpoint qualified on the new
sealed panel; item precision; conditional and joint calibrated intervals; independent component count; outcome-free power lower
bound ≥ 0.80 from a development-only simulation; C ≥ 0.15 with lower bound > 0 for the single frozen arm against augmented
StatNN; both destruction controls beaten; the intact-versus-rotated contrast positive with lower bound > 0. Source graph closure
as in the exposure ledger. A3 source-change qualification and matched actual-predecessor, device, dtype and batch identity are
required. No tuning, replacement or arm change after exposure. The panel is sized between 100 and 300 blocks by the simulation;
insufficient power at 300 is "unresolved", not a larger purchase.

G7 needs G3 go and G4 go, never screen or alternative (S31 §3 G7; §4.0).
Freeze one arm, fresh source-closed blocks and a new disjoint WikiText item panel before any outcome, including sealed qualification.
Each block has intact/rotated edits at strengths {−1, ½, 1} (S31 §3 G7).
Define d_i as intact-minus-rotated predecessor-relative NLL change at strength 1.
The primary orientation estimand is mean_i sign(〈g,D_i−R_i〉)d_i; zero predictions contribute zero, never disappear.
For component c and equal-weight item block j define z_cj=sign(〈g,D_c−R_c〉)×(ΔNLL_intact,cj−ΔNLL_rotated,cj),
using common predecessor references; d_c is the mean over the fixed item blocks. The point estimate is T=mean_c mean_j z_cj,
one equal weight per independent component. With m components, h_c=mean_j z_cj−T and
V_comp=Σ_c h_c²/[m(m−1)]. Conditional SE=sqrt(V_comp).
For J item blocks let h_j=mean_c z_cj−T and h_cj=z_cj−T;
V_item=Σ_j h_j²/[J(J−1)], V_both=Σ_cj h_cj²/[mJ(mJ−1)].
Joint SE=sqrt(V_comp+V_item−V_both), the A2.3 two-way cluster sandwich with finite-cluster corrections.
Conditional draws resample whole components m times; joint draws also resample J common item blocks, preserving all paired
edits/references. Recompute T*, cluster influences and SE* on each draw, treating sampled cluster copies as bootstrap units.
Use t*=(T*−T)/SE* and [T−q_.975(t*) SE, T−q_.025(t*) SE], 4,000 draws, PCG64 seed 20261107.
These signed responses have no positivity, loss-ratio or near-zero-loss guard. Require finite responses and strictly positive
finite point/replicate variances, ≥2 distinct source clusters (and ≥2 distinct item blocks for joint draws), and all 4,000
valid draws. Report every failure; no clipping, redrawing or valid-only success interval. Failure leaves G7 unresolved.
This estimand choice is **chair default, Board A may change before freeze** (BA3 obligation 10).
Require signed-mean lower bounds strictly >0 for both conditional and joint intervals, with shared references.
Use A5.4.2 for ratio endpoints only; signed-mean validity follows the construction immediately above.
The independent-component floor is 100; simulation chooses the fixed count up to 300, never outcomes (S31 §4.3; §§4–5).
New-panel endpoint qualification and A2.1 item precision remain mandatory, including fresh-stage tripwire disclosure.
Apply §§4–5 precision and power gates to C and the frozen control/rotation family; all success requirements are conjunctive.
C_rand, C_perm and C_abl must each have positive conditional and joint lower bounds (S31 §§4.2–4.3; BA3 obligation 2).
A development-only simulation must calibrate joint success across these correlated contrasts, retaining shared item/reference covariance.
Before purchase, freeze its empirical development component/item residual arrays, signed alternative means, ratio effects,
noise/variance sensitivity settings and code/config hashes. Resample whole components and common item blocks, refit where
required by A2, and run the actual signed-mean and ratio procedures. Report null size at nominal 0.05 (must be ≤0.07),
coverage, failure rate and binomial uncertainty from 10,000 null replicates and 4,000 alternative panels, seed 20261107.
Report rejection-test power separately from the probability of meeting every point threshold, interval, control and precision
requirement; the lower binomial bound on that conjunctive success probability must reach 0.80. Missing development inputs
or degenerate variance cannot be replaced by a success assumption.
Size and report power for the actual contrasts; inability to meet power at 300 blocks is unresolved (S31 §4.3).
No replacement, tuning, enlarged purchase or arm change after exposure.
Record the conditional scoring and overhead caps separately (S31 §3 G7); no G7 purchase is authorised by this draft.

#### A5.7 G8 prospective H5 trial

**A5.7.1 — BA3 obligation 11: live contract.** The following block is verbatim S31 §4.4.

Arms: critic-guided search; the same editor without critic; random ranking; same-support gradient search; single-draw best-of-N
at equal real-evaluation budget (one score draw per candidate, no repeated-draw archive); Adam/Muon at equal forward budget. The
primary comparator is the same editor without critic; savings and non-inferiority must hold jointly against it; the other arms
are reported. Pass: realised real-evaluation savings ≥ 30 % with paired 95 % lower bound ≥ 30 %; final-score difference lower
bound ≥ −1 percentage point; guardrails met; one-sided 95 % upper bound on missed qualifying proposals among rejected ≤ 5 % with
episode clustering, denominator = all rejected candidates including failed attempts; positive net GPU saving after charging
diagnostics, gradients, probes, failed attempts and amortised training. Episode count and budget are set from G5 variance to
≥ 0.80 simulated joint power before any episode runs. **Kill** critic deployment only if a powered test shows a savings upper
bound < 30 % or demonstrated inferiority (upper bound of the score difference < −1 point); an interval too wide to show
non-inferiority is unresolved. A powered incremental residual-MAE gain with upper bound < 0.05 against augmented StatNN **and** field-only, on the contrast
fixed in A5 before any episode, **rejects the registered structural representation's incremental predictive benefit on that
population**; it does not kill the structural mechanism in general, and deployment kill is a separate decision. Calibration, power or
budget failure leaves H5 unresolved.

G8 additionally needs G5 feasibility and G3/G4 go (S31 §3 G8).
Before live evaluation, commit fresh episode/item groups, calibration/evaluation split, editor/critic hashes, rejection policy, count and budget.
Choose those counts from G5 variance and prospective joint-power simulation, never evaluation outcomes (S31 §4.4).
Source/direction/prompt-connected episodes stay together in splits and paired clustered inference.
Calibration episodes cannot enter evaluation; no evaluation feedback updates calibration.

Log every attempt, including invalid proposals without deltas, failed attempts and rejected proposals (S31 §§2, 6).
Pin immutable validated effective snapshots, exact coordinates, rounding loss and predecessor/panel provenance before critic access.
The trusted adapter returns accept/abstain, meters costs and uses separately hashed probe hooks/cache with an intervention path.
No rationale or editor transcript is a critic input; retain transcripts only for audit (S31 §§2, 6; BC2 edit 2).
Audit missing outcomes through blinded adjudication; missing outcomes cannot count as successful savings.
For the missed-proposal guardrail, conservatively count unauditable rejected attempts as missed qualifying proposals.
That missingness policy is **chair default, Board A may change before freeze** (BA3 obligation 11).
Charge failed attempts, diagnostics, gradients, probes and amortised training over the registered evaluation episodes.

G8 endpoint definitions and inference (planned chair defaults, Board A may change before freeze):
Each fresh evaluation episode runs all arms on the same task/start state, budget and scoring panel, with frozen seeds.
Connected episodes form one cluster; aggregate all episode totals within a connected cluster before resampling. Pair by episode,
never by unpaired proposals. Let E_c,e and E_0,e count all real evaluator calls, including failed calls and rejected-proposal
outcome audits, for critic and no-critic arms. Savings S=1−Σ_e E_c,e/Σ_e E_0,e is a ratio of totals, not a mean of episode ratios;
require both totals finite and positive at point and in every draw, and report their actual denominators.
The final score F_x,e is the fraction of correct items on the frozen qualified task panel, in [0,1], of the final incumbent;
100×mean_e(F_c,e−F_0,e) is the equally episode-weighted difference in percentage points. A failed episode retains its starting
incumbent; an unauditable final score leaves evaluation unresolved rather than removing the episode. G5 must certify this
bounded endpoint and its panel before G8; an incompatible task requires amendment rather than a change of score units.

A proposal qualifies if a real evaluator on that panel finds a strictly positive score improvement over its actual incumbent,
it satisfies the frozen editor's feasibility/support constraints and it passes the registered harmful/harmless safety checks.
Guardrails require zero confirmed safety/feasibility violations in accepted proposals, valid immutable snapshots and complete
attempt/cost accounting; freeze the task-specific safety predicates and adjudicator in the G8 manifest before calibration.
A missed proposal is a rejected qualifying proposal, plus every unauditable rejected/failed attempt under the conservative
policy above. M=Σ missed/Σ rejected, with denominator all rejected candidates including failed attempts, not audited-only
candidates. Zero rejected attempts makes this guardrail undefined and G8 unresolved. Report both counts and coverage.
Net saving is Σ(GPU_seconds_no-critic−GPU_seconds_critic), charging all listed costs; require it strictly positive.

Use episode-cluster bootstrap-t: resample connected episode clusters at their original count, retaining paired arms/items.
For S, studentise L=log(ΣE_c/ΣE_0) using cluster influence u_k=E_c,k/ΣE_c−E_0,k/ΣE_0 and
SE²=K/(K−1)Σ_k u_k²; invert and transform endpoints as in A5.4.2.
For the score difference, let a_k=Σ_episodes-in-k 100(F_c−F_0), b_k=number of episodes in k;
D=Σa/Σb, u_k=(a_k−D b_k)/Σb and the same cluster SE² applies. This preserves equal episode weights despite unequal cluster sizes.
For M use a_k=missed_k, b_k=rejected_k, u_k=(a_k−M b_k)/Σb and the same SE².
For every endpoint recompute its estimate and SE* on each of 4,000 common paired draws, PCG64 seed 20261107;
t*=(estimate*−estimate)/SE*. Two-sided intervals use estimate−q_.975 SE and estimate−q_.025 SE.
For M the one-sided upper bound is M−q_.05(t*) SE; require it ≤0.05. No clipping or continuity correction.
Nonfinite values, zero denominators, nonpositive point/replicate variances, <2 distinct episode clusters or any invalid draw
leave that endpoint and the conjunctive G8 success unresolved; report failures and never redraw. This includes a degenerate
all-zero missed count: it cannot certify the guardrail without a prospective calibrated amendment.
Net-GPU saving uses the same paired cluster-total accounting and is reported with its interval and cost denominators.

The residual-MAE target is the measured proposal score change relative to its actual incumbent, after the identical
training-only nuisance adjustment. The evaluation population is every attempted proposal in the fresh evaluation episodes,
including rejected proposals with blinded audit outcomes; failed/unauditable targets block structural inference rather than
being silently removed. Critic, augmented StatNN and field-only make frozen, matched predictions for those same targets
before outcomes. A_x sums absolute residual error over that common population; each attempt gets equal weight, not each seed.
For these ratios A5.4.2 applies with connected episode clusters replacing ancestry components, n=sampled attempts including
cluster multiplicities. Conditional inference holds scoring items fixed; joint inference also resamples common task-item
blocks and uses A2.3's two-way variance, maintaining paired predictions/references.
Freeze all development-only G5 residual/cost arrays, alternative effects, missingness assumptions and simulation code/config
before calibration episodes. Simulate paired whole-episode clusters and common items, run these exact endpoint procedures,
and report null size (≤0.07 at nominal 0.05), coverage, failures and binomial power uncertainty with 10,000 null replicates
and 4,000 alternative panels, seed 20261107. Select the fixed count/budget only from those simulations, with lower binomial
bound ≥0.80 for joint success. Report rejection-test power separately from the probability of simultaneously satisfying
savings point/interval, score noninferiority, missed-proposal bound, net cost and guardrails. Calibration episodes set only
registered critic/rejection calibration parameters; they cannot change endpoints, inference or evaluation sizing after exposure.
No G5-dependent count, effect assumption or success result is supplied by this draft.

The powered structural contrasts are critic residual-MAE gain against augmented StatNN and against field-only, each C=1−A_critic/A_comparator.
Use the matched attempted-proposal losses and paired episode/item procedure defined above, with both conditional and joint bounds.
This contrast specification is **chair default, Board A may change before freeze** (S31 §4.4; BA3 obligation 11).
A powered upper bound <0.05 for both rejects only that registered representation's incremental benefit on that population.
It does not reject structural mechanisms generally; deployment rejection is a separate savings/inferiority decision (S31 §4.4; BA3 E4).
Exactly 30% savings and lower bound, −1-point noninferiority lower bound, and 5% missed-proposal upper bound satisfy inclusive limits.
Equality at an inferiority upper bound does not meet the strict rejection rule (S31 §4.4).
Calibration, power, missingness or budget failures leave H5 unresolved.

#### A5.8 Complete field, chart, probe and augmented-StatNN schema

**A5.8.1 — BA3 obligation 7; BC2 edit 2: field and state.** The following completes S31 §6's open schema.
Every new concrete choice in this section is **chair default, Board A may change before freeze**.
Values are planned settings, not observations; S31 §6 delegates their selection to A5.

| Field | Frozen draft value and interpretation |
|---|---|
| WikiText objective | Mean NLL per valid next token, nats; token-weighted over equal-length windows; lower is better. |
| MMLU objective | Mean normalized gold-letter log-prob per item, nats; equal item weights; higher is better; §1 scoring semantics. |
| HellaSwag objective | Gold-option character-normalised log-likelihood, equal item weights; higher is better; descriptive only. |
| Refusal objective | Mean refusal-minus-compliance phrase margin, equal prompt weights; retain harmful/harmless roles separately; descriptive, uncalibrated. |
| Gradient sign | Differentiate the named score itself; signed contraction predicts that score's change, without relabeling NLL as gain. |
| Auxiliary sampling | Hash-sort text groups with `20261107:field:`; 64 groups per objective, separate from all scored panels; one item/prompt per group. |
| Auxiliary lengths | WikiText windows use 1,024 next-token positions (A4.2 convention); other prompts truncate at 1,024 tokens, keeping gold scoring positions. |
| Fisher estimator | Model-expected loss-gradient outer products, one model-sampled target per valid token, fixed RNG; no edited-outcome labels. |
| Factors | A=mean xxᵀ including bias coordinate where present; G=mean output-loss-gradient outer product, valid-position weighting. |
| Factor shapes | For an out×in linear map, A is (in+bias) square and G is out square; store exact dimensions and counts. |
| Coverage | Every supported edited linear map; full-attention Q/K/V/O, GatedDeltaNet maps, MLP gate/up/down are distinct typed roles. |
| Missing hooks | Any supported module lacking a valid hook fails G2; no silent zero factors. |
| Damping | Add 1e-3×trace(F)/dim(F) to each factor diagonal; zero-trace factors retain an explicit zero flag. |
| Truncation | Keep eigenvectors through 99% nonnegative eigenvalue energy; retain omitted trace/energy as tail features, never call it exact curvature. |
| Precision | fp32 gradients, factors and spectral algebra; finite/symmetry/PSD audits; materially negative eigenvalues fail the gate. |
| Negative-eigenvalue tolerance | Absolute negative eigenvalue ≤1e-6×max(trace/dimension,1e-12) may be rounded to zero; larger violations fail. |
| Normalisation | Store raw contractions and magnitudes; training-fold z-scores for fitted features only, no outcome-dependent token normalisation. |
| Base versus predecessor | Base field primary; predecessor, base displacement E=W−W₀ and incremental D stay separate. |
| Cross term | Approximate gradient g(W)≈g₀+G E A; prediction Σ_t[〈g₀,t,D_t〉+tr(D_tᵀG_tE_tA_t)]. |
| Trust-region term | q_t=½tr(G_t D_t A_t D_tᵀ), nonnegative magnitude only; never a signed Hessian correction. |

Audit twenty chain predecessors selected by sorted canonical ID, without outcomes (count from S31 §3 G2).
Compare exact predecessor contraction with the base-plus-cross-term approximation.
Relative error divides absolute contraction error by max(|exact contraction|,1e-3 nats).
Error ≤0.1 on at least sixteen of twenty is required (S31 §3 G2).
For multiple objectives require that count separately per objective; selection and this aggregation are chair defaults under this section's label.
Fewer passing predecessors hard-stops G2 pending amendment, never silently disables the approximation (S31 §4.0; BA3 E3).
The Fisher approximation is not a Hessian and the weight contraction is not a loss Jacobian (S31 §2; BC2 edit 2).

**A5.8.2 — chart schema.** All new values here are **chair default, Board A may change before freeze** (S31 §6).
Projector dimension is 32 per side, seed 20261107, Gaussian then QR with positive diagonal convention.
For smaller dimensions use min(32,dimension); pad unused coordinates with zero and a presence mask.
P_out has shape 32×out and P_in has shape in×32 after padding; active columns/rows are orthonormal.
Revision 7 Nov 2026 (chair, after the implementation in commit 8ee91f3, before freeze): the default projector dimension is **4** per side, not 32. With 32 the augmented-StatNN layout of A5.8.3 has 318 slots and 571,836 columns, unusable for a tree on a few hundred rows; with 4 each matrix slot carries 3 × 16 contraction columns (≈ 9,000 over 186 matrix slots). Every "32" in this clause reads 4; "min(32, dimension)" reads min(4, dimension). Board A may change before freeze.
For each matrix X∈{W,E,D}, store P_out X P_in, plus signed 〈P_out D P_in,P_out W P_in〉 and displacement contractions.
Also store raw 〈D,W〉, 〈D,E〉, norms, rounding loss, support, role, layer and spectral/tail features.
Use epsilon 1e-12 for dimensionless normalisations; zero denominators produce explicit zero/presence flags, never hidden imputations.
Vectors use identity-side projection and a vector role; scalars use a scalar role, with presence masks.
No non-matrix chart token becomes a rotation-eligible matrix by reshaping.
Predecessor and successor identities/lineage are pinned, with fp32 subtraction of stored bf16 states (BC2 edit 2).

The secondary chart head is a masked within-layer set encoder, width 32, mean pooling and a linear response head.
Evaluate f(W,E,D)−f(W,E,0), enforcing zero at zero edit exactly.
Train for 100 epochs, batch 16, Adam learning rate 0.001, no early stopping or search.
These architecture/training choices are **chair default, Board A may change before freeze**; training-scale values match configs/p4_predictors.json.
Pure field, pure chart and hybrid arms stay separately named; no assumed representation containment (S31 §2).

**A5.8.3 — columns, controls and probes.** All new choices here are **chair default, Board A may change before freeze** (S31 §6).
Order tensors by numerical layer, exact role string and full tensor name; order objective columns by the table above.
Flatten all field/chart entries in row-major order; include norms, dimensions, predecessor/displacement inputs, exact support and tail flags.
Use the architecture's fixed tensor slots, zero padding and explicit presence masks for variable tensor sets.
Never pool exact layer/role identity away in augmented StatNN; candidate pooling follows A5.4.1.

C_rand and C_perm use the identical candidate pipeline, replacing only the field and recomputing every field-derived transform.
Random field uses independent per-tensor Gaussian gradients, rescaled to match the intact gradient's Frobenius norm
for each tensor token (not one global norm); zero-norm tokens remain zero with explicit flags.
Random covariance factors use Gaussian eigenframes with intact spectra; regenerate all derived field scalars consistently.
Permuted field permutes layer indices within role and exact shape-compatible groups using a registered seeded derangement;
whole gradient/factor tuples move together. Singleton/no-compatible groups remain fixed and are disclosed, never reshaped.
Use domain-separated `20261107:random-field:` and `20261107:permuted-field:` seeds; freeze generated control manifests and
transforms before outcomes. No seed search after a failure.

Eligibility is checked separately for each arm without outcome access. For token t let g_t be the intact gradient and h_t
its control; a changed token has ||h_t−g_t||_F >1e-6 ||g_t||_F and ||g_t||_F>0. Let coverage be the sum of ||g_t||_F² over
changed tokens divided by that sum over all supported tokens; singleton/fixed tokens stay in the denominator.
Require coverage ≥0.90 (equality passes) and weighted absolute cosine
Σ_t ||g_t||_F² |〈g_t,h_t〉|/(||g_t||_F ||h_t||_F) / Σ_t ||g_t||_F² ≤0.20 (equality passes), over nonzero intact tokens.
A zero/missing total norm, a nonfinite quantity, or a zero control norm for a positive-norm intact token fails eligibility.
Random tokens must additionally match intact norms to relative error ≤1e-6 and pass finite/factor-schema audits;
permuted tokens must pass the registered bijection, role/shape, unchanged-multiset and factor-schema audits.
Compute over the full frozen edited-token inventory, not only changed or compatible groups. A singleton-only inventory or
no compatible group fails the permuted arm; low changed coverage cannot be hidden by removing fixed tokens.
Write seed, token hashes, norms, cosines, coverage, compatibility groups and schema checks to the destruction-audit artefact.
These cutoffs are planned eligibility criteria, not measured destruction claims. An arm is eligible for the “beaten” test
only if its own checks pass and its A5.4.2 intervals are valid. Inadequate destruction makes funding unresolved/no funding,
even with positive control intervals; both eligible arms plus intact no-field ablation must have strictly positive lower
bounds in both conditional and joint intervals.
Chart destruction uses permuted compatible projector identities and independently seeded random frames, reported separately.

Zero-shot sign is the signed linear contraction alone; q never changes its sign.
Field calibration is intercept plus summed signed contraction plus summed q, ridge penalty 1, training folds only.
This uses no more than S31 §4.2's three coefficients.
Probe response uses cached post-gating x for MLP writers, D·x, mean and RMS per supported output coordinate.
Use a separate scoring-disjoint 64-prompt cache (S31 §4.2), hashed panel/predecessor/hook identities and matched probe-only controls.
Fit a ridge head, penalty 1, training-fold standardisation and no search; record cache generation and inference costs.
A1-lite uses the frozen TF-v1 KL vector only, retaining its activation-assisted label (S31 §§2, 4.2).

**A5.8.4 — fitting defaults and config discrepancy.** configs/p4_predictors.json has no tree-specific parameters.
It supplies n_splits=5, seed=0, CPU, standard scaling and neural training settings; these are not LightGBM settings.
Copy tree settings from §3 and src/wsp/predictor/baselines.py instead: 300 trees, learning rate 0.05, eight leaves, min-child five.
Also use subsample 0.8/frequency one, column fraction 0.5, L2 one, two threads, deterministic row-wise execution (§3).
Seeds 0/1/2 and no tuning follow S31 §4.2.
This source resolution is **chair default, Board A may change before freeze**; do not claim absent JSON tree settings were copied.
Freeze the complete resolved config with code/schema hashes before G2 (BA3 obligation 8).

#### A5.9 Old seal and Part B wording

The unspent partition `ebf0379a15d95b592ee8c227836406f3365a56a3ef46e6978d369c2d42c0ff3f` stays sealed (A2.3; S31 §5).
It is not reused for G7; G7 requires newly certified fresh closure and a new item panel.
A4.4 governs Part B, including all six intervals, calibration disclosure and no pass/kill wording (S31 §§4.2, 5).
Funding “go”, “screen”, “no-go” and “unresolved” are not scientific falsifiers.

#### A5.10 New amendments and acceptance crosswalk

Any change to endpoint/orientation, scope, estimand, arm, information access, controls, thresholds, selection, exclusions or inference needs a dated prospective amendment.
Changes to source closure, panels, preprocessing, seeds, schema, predecessor approximation or resource-driven coverage likewise need amendment before affected outcomes.
Board A may change labelled chair defaults before freeze, recording the accepted version and hashes.
After freeze, a code/config change requires versioning and any matched qualification before new outcomes, never retrospective repair.
Infrastructure retries retain identical inputs and immutable failed-attempt records; they cannot select scientific outcomes.
Unresolved manifests, code, calibration or resource evidence prevent acceptance/execution, rather than becoming assumed results.

BA3's twelve obligations have numbered homes: 1→A5.1.1; 2→A5.5.1; 3→A5.2.1; 4→A5.3.1; 5→A5.3.2; 6→A5.3.3.
The remaining homes are 7→A5.8.1; 8→A5.4.1; 9→A5.4.2; 10→A5.6.1; 11→A5.7.1; 12→A5.2.2.
Cross-references supplement those clauses; none of the obligations is waived by this draft.


#### A5.11 Mandatory manifest package and freeze blocker

Before G2, commit `docs/prereg/a5_manifests/` and list each file by path and sha256 in A5.1 for Board A acceptance.
The package must contain actual edit IDs, forty selected component IDs and selection ranks, closure/alias/tensor-coverage
proofs, text-group hashes and disjointness certificates, exact predecessor and reference run IDs, all eight alias reuse
proofs, missing-reference counts n_D/n_P with priced costs/retries/preprocessing, model/schema/config identities and code hashes.
Include `scripts/a5_simulation.py`, `tests/test_a5_simulation.py`, their code/config hashes, selection-inventory cutpoints,
and control seeds/eligibility specifications. NULL-SIZE inputs are development window-level paired residuals from G1 TF-v1
records plus resolution settings, norms, certified closure and audit/missing flags, available only after G0/G1 closure;
commit their provenance, inventory/hashes and the pre-G2 null-size report then. POWER-AND-YIELD adds G2's development
field predictions, with edit IDs, model/schema/code identity, generation provenance and actual file hashes after G2 and
before G3, plus its power/yield report. The order is freeze(A5) → G0/G1 → null-size → G2 → power/yield → G3;
pre-G2 specifications freeze code/schema, not the later field values. Do not fabricate IDs, hashes, audits or passes.
G2-produced cache-content hashes and resource measurements are committed before downstream use and before G3; their input
specifications/code belong in the pre-G2 package. G7 purchase identities and G8 G5-dependent sizing remain prospective gates,
fixed before their own exposure. The procedural freeze pins code/schema before G0/G1; the completed pre-G2 evidence
package, including post-closure residual hashes and null-size results, must be committed, listed by sha256 in A5.1 and accepted
by Board A before G2. Post-G2 field values/hashes and power/yield results complete the pre-G3 evidence gate.
This clause is a blocking draft milestone, not fulfillment of the missing identity/reference evidence or authority to compute. No hashes, selected IDs or completed certifications are asserted here.
