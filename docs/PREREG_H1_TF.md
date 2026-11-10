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

Dated amendment: **7 Nov 2026**. **Draft for Board A; not an execution freeze.** (7 Nov status. The freeze-pass
inventory of 10 Nov 2026 is A5.1.2; acceptance and execution remain Board A acts.)
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
the 7 Nov draft had no such inventory and was not frozen (the 10 Nov 2026 inventory is A5.1.2). Cache-content hashes and resource measurements generated by G2 are committed
before downstream use and before G3, rather than required before the gate that creates them.
The procedural freeze in A5.3.2 fixes code/schema before G0/G1; development residual hashes follow closure, and
label-free development field predictions/hashes follow G2 before G3. Neither generated input is a pre-procedural-freeze value.
MMLU promotion needs a new prospective endpoint/arm family, multiplicity and joint decision rule before new endpoint outcomes (BA3 obligation 1).

**A5.1.2 — freeze-pass inventory (10 Nov 2026; [BA-FRZ](reviews/council/board-a-freeze-2026-11-10.md) item 7).** This
replaces, as of 10 Nov 2026, A5.1.1's 7 Nov statement that no inventory existed; that wording above is kept as history.
It is the external final inventory for Board A's post-commit audit, not Board A's acceptance. Executed in the review's
order without G2/G3 outcomes, GPU or new governed outcomes:

(i)–(ii) Edits 1–6 resolved prospectively: A5.2.3, A5.9.1, "F5 schema pins (10 Nov 2026)" and "A5 freeze-pass
amendments (10 Nov 2026)". The two sensitivities (FP2) were run. Old raw evidence was preserved, and the registered
run was re-run with the amended script and reproduced byte for byte.
(iii) `code_hashes.json` (schema `a5-code-hashes-v2`, sha256 `708c6fabd8fcbcfe4d0d8f4d38ed19eddae2797ffa8c5a89411d75053abfe2c9`) pins 57 files from the working
tree committed with it (`git_sha` = parent `4fd1e2a`): builders, simulation/validator, A5 tests, every transitive `wsp`
import of a pinned module, the primary fitting code/config (`src/wsp/predictor/baselines.py`,
`configs/p4_predictors.json`) and the fixed selection/panel/qualification inputs held in git. It also pins the resolved
TF-v1 config, model/tokenizer revision and environment.
(iv) `scripts/a5_build_manifests.py --pin-worktree` ran against the closed snapshot, whose live-input pins matched. It
reproduced `components.jsonl`, `references.json` and `text_groups.json` byte for byte, and the same forty IDs (sorted
sha256 `e2bd6cb2956c0fbe64194b3f1ebbb481248ecec45b8a729a59a54b9a2bac9d6b`), forty distinct components, the four
`(norm, id)` cutpoints, n_D = 19 and n_P = 0 (`check_procedural_freeze` passed). A full `scripts/a5_build_panels.py` rebuild from
the offline cache reproduced `nullsize_inputs.json` byte for byte. It also reproduced every panel ID, rank, certificate
and count of `panels.json`, differing in one line only: `inputs.code.sha256`, the builder's own provenance hash
(committed `f218b72e…a2b29f4`; amended builder `cfba620e79b7c3b7c98c0d92d53ec4b2fe7a2ee9f323f8586f8f7a67e1502534`). The
committed `panels.json` is therefore kept byte-identical with its historical producer identity. The other companion
files and README section were regenerated with `--from-committed`, which reproduces it exactly, and
`--from-committed --check` reports every file identical. `control_seeds.json` changed (chart projector seeds registered,
FP6), and `linestop_power_spec.json` is new.
(v) `scripts/a5_simulation.py --inputs-spec` reproduced the raw `validation.json` byte for byte (valid: 796 inventory rows,
179 eligible, 141 donors, 0 masked windows, 0 incomplete pairs, 423 unavailable audit entries). The NULL-SIZE reports were
regenerated: the three earlier trace files, `inputs_provenance.json` and `validation.json` are byte-identical, and two
trace files are new. `scripts/a5_snapshot.py --evidence` then rewrote only `snapshot.json:content.a5_pre_g2_evidence`.
(vi) `selection.json` was refreshed only in `inputs.snapshot.sha256` (now `1b2af25cde1bd7ad0427fec2d0ee22d34f75651ba3fb979176464f8fad701da3`). Its 7 Nov procedural-freeze
block is untouched, including code-map digest `2b93ec10b728c981aaee126d9f4d5e5ded70210b08f842bf8906a454ac900a95`. Both
READMEs were regenerated.
(vii) The SHA256s below were computed from the committed bytes.
(viii) **Re-pin for the A5.8.1 amendment (10 Nov 2026, before audit; "A5.8.1 amendment (10 Nov 2026, before audit):
eigenvalue tolerance" at the end of this document).** `scripts/a5_build_manifests.py --pin-worktree` was re-run against
the closed snapshot, into a copy of the package, from parent `b7f10a1`. It reproduced `components.jsonl`,
`selection.json`, `references.json` and `text_groups.json` byte for byte; only `code_hashes.json` and `README.md` (which
lists the former's sha256) changed, and only those two were copied into the package. Nothing else was regenerated:
`panels.json`, `nullsize_inputs.json`, `control_seeds.json`, `linestop_power_spec.json`, `snapshot.json` and every
`nullsize/` report and trace are untouched. `code_hashes.json` (schema `a5-code-hashes-v2`, sha256
`701ae646eff3a00713140b8100ac28c83a81973c7f296909f9ce18984a130744`, `git_sha` = parent `b7f10a1`) now pins 64 files: the
new `src/wsp/features/response_field.py` bytes, the builder itself (its pin list grew), and seven newly pinned paths,
`scripts/g2_fields.py`, `scripts/g2_fields.sh`, `scripts/scale4b_build_edits.py`, `scripts/lambda_job_scale4b.sh`,
`scripts/scale4b_h0_pair.sh`, `scripts/scale4b_report.py` and `configs/eval_tf_v1_wikitext.yaml`. Its environment block
now also records the fitting libraries lightgbm 4.7.0, scikit-learn 1.9.1 and scipy 1.18.1. This code map supersedes
`708c6fab…abfe2c9` as the current one; that digest and the 7 Nov and 8 Nov digests remain historical identities. The
amendment commit changes only `docs/PREREG_H1_TF.md`, these two package files, `src/wsp/features/response_field.py`,
`scripts/g2_fields.py`, `scripts/a5_build_manifests.py`, `tests/test_response_field.py` and `tests/test_g2_fields.py`
(the last two are not code-map members). The table rows below carry the new literal hashes with old/new identity.
(ix) **Re-pin for the 4B builder alignment (10 Nov 2026, before any 4B scoring; "A5 note (10 Nov 2026): 4B builder
aligned to F5.3/FP4" at the end of this document).** `scripts/a5_build_manifests.py --pin-worktree` was re-run against
the closed snapshot, into a copy of the package, from parent `43283ec`. It reproduced `components.jsonl`,
`selection.json`, `references.json` and `text_groups.json` byte for byte; only `code_hashes.json` and `README.md`
changed, and only those two were copied into the package. `code_hashes.json` (schema `a5-code-hashes-v2`, sha256
`ce07f3b5464457ff96356e58d57279de93cf5392a52bf31d7ce1c8068cc6f8af`, `git_sha` = parent `43283ec`) still pins 64 files;
it differs from `701ae646…130744` only in `git_sha` and the four changed 4B scripts, `scripts/scale4b_build_edits.py`,
`scripts/scale4b_report.py`, `scripts/lambda_job_scale4b.sh` and `scripts/scale4b_h0_pair.sh`. `701ae646…130744`
remains a historical identity. The table rows below carry the new literal hashes with old/new identity.

*Identities.* Historical: 7 Nov procedural-freeze code map `2b93ec10…900a95` and 8 Nov evidence code map
`05a2d244a0d15bb8ef70cb62e89611c783447737b724f755ad98ed9b08ffeabe`. This freeze's code map is `708c6fabd8fcbcfe4d0d8f4d38ed19eddae2797ffa8c5a89411d75053abfe2c9`. Content
pins (`selection.json:inputs`): feature content index `af893deab180568858487f4aac9481f249eda1f9acc86b9d8652d563d95ee3fc`,
TF-v1 record content index `124e9b9245b78ff480f145a8dd89086e7dae21822e5d46bad06a5d46a8c66064`, record index
`c8c5dff72d88b8fdd9b7240b80cc75ab5becebb3bea89dc1c5f5e2e68a320750`. Snapshot closed 2026-11-08T00:59:56Z, base
`19f5cbf781cb0955fdfe482c12ec523cc2873dc6`, per-file identities in `snapshot.json`. Effective config hash
`79d13d456efee587281566168ebd598bcffe136d3c3c7f8f13a05692350014f6`. Model and tokenizer Qwen/Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`. Environment: Python 3.12.3, numpy 2.5.4, torch 2.15.0, transformers 5.22.1,
tokenizers 0.23.3, datasets 5.1.0, safetensors 0.8.0, pyyaml 6.0.3 (`code_hashes.json`); fitting libraries lightgbm 4.7.0,
scikit-learn 1.9.1 and scipy 1.18.1 are recorded here. `docs/PREREG_H1_TF.md`'s identity is its blob in the freeze
commit; it carries no self-hash.
*Resolved primary fitting settings (A5.8.4).* These come from §3 and `src/wsp/predictor/baselines.py`, not from
`configs/p4_predictors.json`, which has no tree settings. LightGBM `LGBMRegressor`: n_estimators 300, learning_rate 0.05,
num_leaves 8, min_child_samples 5, subsample 0.8, subsample_freq 1, colsample_bytree 0.5, reg_lambda 1.0, n_jobs 2,
deterministic, force_row_wise, random_state ∈ {0, 1, 2}, with predictions averaged before the absolute loss. Isotonic:
increasing, unweighted, `out_of_bounds="clip"`. Nuisance: OLS (Moore–Penrose, float64) with intercept, treatment-coded
family/device and log1p incremental effective norm, on training components only. Folds: §3's five source-closed folds
(`20261105:inner:`); no tuning or early stopping in the primary.
*Staged, not pinned here (each needs a dated amendment with literal hashes before the affected execution).* LoL/K+F
code and settings (F5 schema pins); the line-stop power implementation and report (FP6); 4B model/tokenizer/window
identities and qualification run IDs (FP4); the pre-G5/G8 adapter manifest (FP5); G2 field/cache hashes, resource
measurements and POWER-AND-YIELD; G3 alias reuse proofs and tensor/rotation audits (F3).
*Change allowlist.* Relative to parent `4fd1e2a`, this freeze commit changes only `docs/PREREG_H1_TF.md`, package files
under `docs/prereg/a5_manifests/` (each listed below), and the five A5 scripts and four A5 tests marked "changed" in the source table. No
fixed panel, ranking or qualification input changed. Raw data paths are hash pins only; the two new raw outputs were
written under `data/runs/a5/nullsize/`, the one writable data directory, and nothing there was overwritten. The
baseline `docs/PROGRESS.md` difference (`e3c577f` → `6f3d571`) predates this commit and is not touched.

*Committed companion package (every file).*

| Path | SHA256 | Identity |
|---|---|---|
| `docs/prereg/a5_manifests/README.md` | `225f0caeff23bc7f5bf58115f63f4f08294729484979d8912c947e6662766120` | changed 2026-11-10 (4B builder alignment re-pin); old `a6adac9a57858dc43c1bed7da27afab6ef9bda5b5713b961950e2e2968d3d373` (A5.8.1 amendment re-pin), before that `46996d88e854362314fff1df4e6cfe56aa0399ae308f43b8e88638e3c0890018` (freeze pass), earlier `3bc5132b717a35b4cde20fa723a9a9d68f9e4f04c01dbcd603eb884295d8a5bc` |
| `docs/prereg/a5_manifests/code_hashes.json` | `ce07f3b5464457ff96356e58d57279de93cf5392a52bf31d7ce1c8068cc6f8af` | changed 2026-11-10 (4B builder alignment re-pin); old `701ae646eff3a00713140b8100ac28c83a81973c7f296909f9ce18984a130744` (A5.8.1 amendment re-pin), before that `708c6fabd8fcbcfe4d0d8f4d38ed19eddae2797ffa8c5a89411d75053abfe2c9` (freeze pass), earlier `05a2d244a0d15bb8ef70cb62e89611c783447737b724f755ad98ed9b08ffeabe` |
| `docs/prereg/a5_manifests/components.jsonl` | `6b4eff8d61a4b1c4d83f0c058cb4df5c6c5398e2c6656a8b58b38f8b46251dec` | unchanged |
| `docs/prereg/a5_manifests/control_seeds.json` | `d357ed51031877f817baeb4204cf93b38c66fdfcfd95bd41803d93d5160c12d5` | changed 2026-11-10; old `8e2cbae160e8e8448b60545b0f79c0680f1fdd9ccf005f30d1c04907fe4bd28b` |
| `docs/prereg/a5_manifests/linestop_power_spec.json` | `7bfff5203133bc3481ac34808454297ce9fa090e2dd9ff11fd742af7e214cae9` | new (2026-11-10) |
| `docs/prereg/a5_manifests/nullsize/README.md` | `727dfc07fa7fe57e2a06b21682e1430f26809cc828f8875436253e0ea676e812` | changed 2026-11-10; old `a63534e6b16ae7c9167959113594e68266e02f56027c5e4a20ece952d0025174` |
| `docs/prereg/a5_manifests/nullsize/cell_mask_noise_traces.jsonl.gz` | `ea79b0c563ff2b0f949a2c608821815e04c980ab331a56b8242c56a1b831d90d` | new (2026-11-10) |
| `docs/prereg/a5_manifests/nullsize/constant_positive_mean_traces.jsonl.gz` | `2c691cfdcb09e9698eef771b5e4de292253d3c8f299bf3cbcba46182cba8801d` | new (2026-11-10) |
| `docs/prereg/a5_manifests/nullsize/inputs_provenance.json` | `ea160aa4088c56db0aa27046daf3e21eb5865b3950b98e70b4eff9242e1eb1a3` | unchanged |
| `docs/prereg/a5_manifests/nullsize/registered_traces.jsonl.gz` | `34079bb88397386542261bb311720a1509bca76bb53ca3e93f71cabd05582aab` | unchanged |
| `docs/prereg/a5_manifests/nullsize/report.json` | `1bed2c9531e4230864a1ebbe7e4023e0ffafcf704a59c40e93f9a75bd469dfa6` | changed 2026-11-10; old `b6ea5d134790090afeec9ad080531965d3f1b4682df3eb88bc6875d589bf9228` |
| `docs/prereg/a5_manifests/nullsize/unavailable_audit_fails_traces.jsonl.gz` | `105ec8aab49b7bb5e9dba5d8034d4c724b62c870252efcd3d9e4a2af2849d95b` | unchanged |
| `docs/prereg/a5_manifests/nullsize/validation.json` | `7a9f72876e73f5ffba2852f925a8aa3ae52a5c9261d55af942b322b1e0606136` | unchanged |
| `docs/prereg/a5_manifests/nullsize/variance_x2_traces.jsonl.gz` | `2c7e497ff72223e86e1c24b1e1691cb1af8a6d2e223d54adc131e1573cdfe582` | unchanged |
| `docs/prereg/a5_manifests/nullsize_inputs.json` | `6e58b589e32491b36fd70e679b20242538ea8e958b9b9072ce0e30b444508885` | unchanged |
| `docs/prereg/a5_manifests/panels.json` | `c291b68516952c4f342bfc7f36843a83f0482df7d721a00e46287597bad108cd` | unchanged |
| `docs/prereg/a5_manifests/references.json` | `5b6991f299c1eee049de51bc1e4f00c71c1dfc58114cdf6549fb5d246850564e` | unchanged |
| `docs/prereg/a5_manifests/selection.json` | `393530365f87f981d5da3ecb2779c89b8e7bf98105b6c45d1824bf290e908c63` | changed 2026-11-10; old `13e5fc6ab78ba1fa08a6a043691605759c1a027c279fc2efd795943053540bb9` |
| `docs/prereg/a5_manifests/snapshot.json` | `1b2af25cde1bd7ad0427fec2d0ee22d34f75651ba3fb979176464f8fad701da3` | changed 2026-11-10; old `030517947267287875ee55847d24732c0e30f0140a83bfd3fdbd9a58264a71c1` |
| `docs/prereg/a5_manifests/text_groups.json` | `f769afcade5a80571daf3a9578b1c03584b46a5abbbba07b32d43a7e9035e7e1` | unchanged |

*Raw evidence (hash pins only; not committed).*

| Path | SHA256 | Identity |
|---|---|---|
| `data/runs/a5/nullsize/cell_mask_noise.json` | `b2da17e774b5bcd65a518dff21e9bdb3a5fe67d1395ab8abc95e49c8ddca1fe5` | new (2026-11-10) |
| `data/runs/a5/nullsize/constant_positive_mean.json` | `d64ce74bd0e29d000264ac6edde8ac53ad981fdcc1c4dc20265017a76f7173d2` | new (2026-11-10) |
| `data/runs/a5/nullsize/inputs.json` | `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de` | unchanged (Board A review pin) |
| `data/runs/a5/nullsize/registered.json` | `452c078aae520dfedbcf25a44a9390305612d32ade3793840f3118043e4239e8` | unchanged (Board A review pin) |
| `data/runs/a5/nullsize/unavailable_audit_fails.json` | `279c31f8eda4ffef535802befab261ec018bcf363ce9c4dd128e5fb38305790d` | unchanged (Board A review pin) |
| `data/runs/a5/nullsize/validation.json` | `7a9f72876e73f5ffba2852f925a8aa3ae52a5c9261d55af942b322b1e0606136` | unchanged (Board A review pin) |
| `data/runs/a5/nullsize/variance_x2.json` | `c8d9b0783127d7cba31edeccd3befaaf56df0ad92ee3e776e9191870b8e8745b` | unchanged (Board A review pin) |
| `data/runs/d8/a5_incremental_norms.jsonl` | `6d8954f1aefde5b0af63944c1286f73f86d7f5b35dbe2ec2d8d1f470edc7a11a` | unchanged (Board A review pin) |
| `data/zoo/qwen3.5-0.8b/artefact_status.jsonl` | `376a829c3d5839604c61319f41123ef3d4e544b05586988f9012c7c7b02c1878` | unchanged (Board A review pin) |
| `data/zoo/qwen3.5-0.8b/manifest.jsonl` | `71a396e5bab2e53f8f74b0a8a0c646e1a791ef92bb3cf3f3df5c51c8010140e7` | unchanged (Board A review pin) |

*Source, config and test identities (the complete `code_hashes.json` map, pinned from the working tree committed with it).*

| Path | SHA256 | Identity |
|---|---|---|
| `configs/editor_qwen3.5-0.8b.yaml` | `cd53d67beb2cb231351b3e2dcc8bdc76ad2278d0da0cd6f720e08ae749f41ed4` | unchanged |
| `configs/eval_subsets/harmless_alpaca_tf_v1.txt` | `026ce60174fb8f227c6dc433d85c0574c06a1caf62e5b7228ad153f4de4f169c` | unchanged |
| `configs/eval_subsets/hellaswag.txt` | `938f0b691e5d90e705da6ac88586a766c1f5ab00ffa77fe129a0ce546c8ab507` | unchanged |
| `configs/eval_subsets/hellaswag_tf_v1.txt` | `a48427f88a2ff5dcb02d0f3c02ccb8fc472455ebb336fbe40b48206c45cc706e` | unchanged |
| `configs/eval_subsets/mmlu_pro.txt` | `45aa5c72467f0cbbee7df5deb5c104fde3a4c880d3a60eea533e709dabce62ac` | unchanged |
| `configs/eval_subsets/mmlu_pro_tf_v1.txt` | `a7bef4f45d9c888673868999a5a611a8d31d9c8e956347bc43cf4e0ebfaddbea` | unchanged |
| `configs/eval_tf_v1.yaml` | `5c20cabe8e35b4116adeed4df650cf27461e7ab144ef66c0b60c397b13b00317` | unchanged |
| `configs/eval_tf_v1_wikitext.yaml` | `5fd90d015b206d10780e791a710563ee41fbbd3e00deeb53ec4b65abd617ac0c` | newly pinned 2026-11-10 (A5.8.1 amendment); bytes unchanged since `b7f10a1` |
| `configs/p4_predictors.json` | `c217b5a5b012689475609782bfb68d862cbfb4864b1233093be7d355a7e04cfe` | unchanged |
| `docs/prereg/panel_manifest.json` | `7c72bfd3280954a82de1161e640a870da5f3a876c03b22247656701ed6af2c67` | unchanged |
| `docs/reviews/P5/d8-regeneration-2026-11-06/chain_replay_install.json` | `b5f5b713690142b4c700f28886006c870ffdbed86baeb499c70125fee976a6dd` | unchanged |
| `docs/reviews/P5/d8-regeneration-2026-11-06/chain_replay_pilot.json` | `30e672b4b81c7b52b96fe78d9bdb10d47d38e1065339551a87777a235a5f4621` | unchanged |
| `docs/reviews/P5/d8-regeneration-2026-11-06/chain_replay_restore.json` | `78f59f514be22fa0315953ea26814610ce2a5301ed04cda7b7063965f41264ea` | unchanged |
| `docs/reviews/P5/d8-regeneration-2026-11-06/g1_report.json` | `356f4fa15ef961a6ab001d1ef284dc8ba2d1b13b1a62786d16e2cf839f480f2e` | unchanged |
| `docs/reviews/P5/d8-regeneration-2026-11-06/g1b_report.json` | `45f152b2614348d4038a49f8aaaccc6cce46840202fe1ec90bc0e18f97bd30ff` | unchanged |
| `scripts/a5_build_manifests.py` | `c7dfedb1201307314059c03249905eeabbb4cbe4262a6370c3e84f1103ac0e05` | changed 2026-11-10 (A5.8.1 amendment: pin list and environment packages); old `e9f0e6faf3096e50f3977c030e74b49f61dc4800d2ba501c2de408beef3bd769` (freeze pass), earlier `2f994d7bf2f32110f909732b3cdb430abeaae89be411e636cfb1c99b8a3fbcab` |
| `scripts/a5_build_panels.py` | `cfba620e79b7c3b7c98c0d92d53ec4b2fe7a2ee9f323f8586f8f7a67e1502534` | changed 2026-11-10; old `f218b72e927a620fa4b1b1a5bc72e6543bb43846a96c5be12377eaae8a2b29f4` |
| `scripts/a5_incremental_norms.py` | `fbea8c8e9ee147e18f6ca0d6f8d8232866ebc417dd1b5acabda7f2105bc26c3b` | unchanged |
| `scripts/a5_nullsize_inputs.py` | `d3f6718ec5b8ec0fe47ce515e738fa692c8c469fd87efccc0e7ab5deb9de9a42` | changed 2026-11-10; old `b09c07a15d9965874e22228477c948a29c359e7b09da560432d663e485229169` |
| `scripts/a5_simulation.py` | `3b86668657273b188abdaa2e5d19eec832abe0fdb21f19b002fc5c0504da8fe6` | changed 2026-11-10; old `ecde1dd7cb99e2b6dc450da5bcbc229e417e21c75b2b94a2a9db61f6b3870668` |
| `scripts/a5_snapshot.py` | `a0e68302e1256ab55594e12223677d527bc4fa830451194d61e28aebaecaed72` | changed 2026-11-10; old `a7518b416ca21259cef584db28492298de8aa2146ee3c58e0ea165664e6653b3` |
| `scripts/g2_fields.py` | `4856f9938633b94138fabc3a592962ea93b2e274045aa71cc30bedf64b5e52f9` | newly pinned 2026-11-10 (A5.8.1 amendment); bytes changed in that commit (clamp statistics), old blob at `b7f10a1` `c1366df579a02d060ec9081fe0e77e6ff51e3cb92826ffd6259add8769b37e39` |
| `scripts/g2_fields.sh` | `13b761dac0a0682ed7f12f521ef0e1cfd31eeb613489aac68155f08b66f4c133` | newly pinned 2026-11-10 (A5.8.1 amendment); bytes unchanged since `b7f10a1` |
| `scripts/lambda_job_scale4b.sh` | `45a59075ecbc986706a660839e0e4ab25e13c712b2c6072288512d98624b41a2` | changed 2026-11-10 (4B builder alignment to F5.3/FP4); old `2cd984f19ae78447aed0f79757486a18963c96379acf94e37f2f93ea32cc88f8` (newly pinned at the A5.8.1 amendment; bytes unchanged since `b7f10a1`) |
| `scripts/scale4b_build_edits.py` | `e520377a5ee7244b5d787e51ff9718a4bb4cb3a798f5bf743374612bd8114ea1` | changed 2026-11-10 (4B builder alignment to F5.3/FP4); old `55669ed20f1f502cd3567b35d5bb444b0994de8a5ab0dae54db1b81df7f9fa0d` (newly pinned at the A5.8.1 amendment; bytes unchanged since `b7f10a1`) |
| `scripts/scale4b_h0_pair.sh` | `0bfe4819ae1eae4584f2aa0ba2227813023810cb46b78075fada5b0276446be5` | changed 2026-11-10 (4B builder alignment to F5.3/FP4); old `317da12dac9324afb12becb2dfdc56d4573c220ea019477c4be5090cac45fda1` (newly pinned at the A5.8.1 amendment; bytes unchanged since `b7f10a1`) |
| `scripts/scale4b_report.py` | `90279c5ebfe8e6a67713cc19d0f7057ba9ffcb11e8c5f0306f3f91a2662f4a5e` | changed 2026-11-10 (4B builder alignment to F5.3/FP4); old `27b15f45ada44f61855c51e1914a2f3262c2f8a89647100122122d55718f1440` (newly pinned at the A5.8.1 amendment; bytes unchanged since `b7f10a1`) |
| `src/wsp/__init__.py` | `69d834204c661b1dd2cecf98720126c72e3bade7e235eeb82541dcc983e9cff9` | unchanged |
| `src/wsp/eval/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | unchanged |
| `src/wsp/eval/batch.py` | `122b0fbac62e3b94d7a5c356bd2f45692a18f96ea1a7811c45763b0863e789c6` | unchanged |
| `src/wsp/eval/ifeval/__init__.py` | `0f203c7c20602839e1964adbb9a78ca513ce77384e011191d23e9513c8c9e776` | unchanged |
| `src/wsp/eval/ifeval/instructions.py` | `fb84b3187977399b26e83b87e78da1d9567878b11b98d8777de0b8e55a58efa3` | unchanged |
| `src/wsp/eval/ifeval/instructions_registry.py` | `7548ab8ad441c942826976b5188c3bb604da62bea272b82b82327dd60c2fc925` | unchanged |
| `src/wsp/eval/ifeval/instructions_util.py` | `0f57e19fe72273d501d14ddb74c94b345533497132ae53f32e61cac1a40fadce` | unchanged |
| `src/wsp/eval/ifeval/utils.py` | `3557c87b0d6bb290e34c49fc0cf88320421f9251109d70ced87ea8fef341032b` | unchanged |
| `src/wsp/eval/metadata.py` | `d6f97b9317d394e8f4c19dc6f098021cbfc4ff27aa5f2b2cf380b5afcde313c7` | unchanged |
| `src/wsp/eval/refusal.py` | `b416b0f816117dfedd4b826572eef2c87c1761e98dac145581052fecd2354db0` | unchanged |
| `src/wsp/eval/repeats.py` | `cd6e4c04502bffb979d248a0c7b8ef255dbd191b964875a10a6fa02728d26e64` | unchanged |
| `src/wsp/eval/response_intervention.py` | `7cf453f263b92b3dc056109e06766b7dbd3dc606ad463f11aa4ccdec1b73f903` | unchanged |
| `src/wsp/eval/run.py` | `24d01a9e7cb846da34e63bf157f8a49bde817570ce474e3120999f267e39c69a` | unchanged |
| `src/wsp/eval/subsets.py` | `5f8b55ad555aa910c3ca2024cc830d5bda479021a0a01de268d5f27a5c26f24f` | unchanged |
| `src/wsp/eval/tf_v1.py` | `29125dac45ddf7ffa7a490c0daead1ea06236835f09a0f34f7b38579043caef9` | unchanged |
| `src/wsp/eval/zoo_compat.py` | `28763035e98344b5ef1cca24a6f20c25b38e6d74ef9596ef913bb1258f87c969` | unchanged |
| `src/wsp/features/__init__.py` | `9e0306865c2d23f151673293aa577718b7771b085d15fdb6378c29890b5600a0` | unchanged |
| `src/wsp/features/extract.py` | `11de545c5ba8b4186df91ff218992a595fdbc8dcd343fd4b6b3a8e6278626906` | unchanged |
| `src/wsp/features/pool.py` | `85a6341a446f32761531e0f0a193101114201d9eb08b0e05e6a6d1d42ee94390` | unchanged |
| `src/wsp/features/response_chart.py` | `f23397779c13d2d63cb9db79f126886612eb3deb67d1d97b433ead7ec5bcc0c6` | unchanged |
| `src/wsp/features/response_field.py` | `98d81bbc300e715fe27a02e04711c0d036ae7ef9342d6bddbcaa227d91f725bc` | changed 2026-11-10 (A5.8.1 amendment: fp64 eigendecomposition, λ_max-relative tolerance, schema 2); old `94bbe9b89ee51b8a444ca9859cb3415182c38dca61ce90133df8beef5bb3bd4d` |
| `src/wsp/features/rotation.py` | `f3ff12426ad89dac69073b2d06ca034f83de7257edd248d2a8063a194b1897e3` | unchanged |
| `src/wsp/features/stats.py` | `f1d4df606fd49ac789a6a7e64c34c93ee0bbc543bac9faf5bf57db03bd04911b` | unchanged |
| `src/wsp/features/svd.py` | `48a56004747da7a217850acae1f9e98c8a64adf5c57e6b51c71357fafe14c3fc` | unchanged |
| `src/wsp/predictor/__init__.py` | `1aa5fc9caf7a4504032f9896cb4b4d56ce037b4df4990cc4e1462f92356a46f7` | unchanged |
| `src/wsp/predictor/baselines.py` | `c1602d20db9b5e3fdb6558a769b98b495cdb0e256228f745751ebe197c9aa6bc` | unchanged |
| `src/wsp/predictor/splits.py` | `1eeec3071c26e73bbb9e4ae8abb42ba0a271e82c3540e442772e83bed6dc868f` | unchanged |
| `src/wsp/schema.py` | `2c8be072934e002bf17e3ae80ab167e65f3c2b9c827a963a25d4b2d76376b996` | unchanged |
| `src/wsp/zoo/__init__.py` | `682c6a58968670c8692f2219a206263c37a65c2ec7c6fca9fd27a4a690ac0e53` | unchanged |
| `src/wsp/zoo/apply.py` | `4dbeb10e9d7024508baa376b915a90072170428f87663b32865ff4543cd72005` | unchanged |
| `src/wsp/zoo/arch.py` | `1cb721530a05a0bc33c7dc9cd41667805b10e65c4dc7dcb71fa3558ce8c304f2` | unchanged |
| `src/wsp/zoo/common.py` | `738e7f4edea6e880fe0a076d2a7592033e7bf809bf277fbda2cc2804ecea4318` | unchanged |
| `tests/test_a5_build_manifests.py` | `946e63d451baeaaec2393db3fb53f923005641a44c6557e954f47fb074851902` | changed 2026-11-10; old `ad5ef44bb3549500b0b2588f1ca3497fdc9a5052f5a4a1557a2af76379547683` |
| `tests/test_a5_build_panels.py` | `95ba2f3073ecc3c6f5e89fcc039f02d1d20952ba6266b727460ca976343736e2` | changed 2026-11-10; old `84f734f3dcdfb97d10087d76ff9787e6a8064119f02843166cbd186d66f800f4` |
| `tests/test_a5_nullsize_inputs.py` | `dfa40c03abbbce67f790fa2d96032a9862683e5ef63213c825eef9505ea11208` | changed 2026-11-10; old `dadddb13bebaa28ac1c50a63e6e59e425a53c992fea500e123f4d833e0940125` |
| `tests/test_a5_simulation.py` | `721a92c21adaf39d0b5c317ee5ba8e6c60a2b4f63ba1b837fc3f21272cf65072` | changed 2026-11-10; old `c1bae99ae8565143c688bc5dabbee8a979904ae4a8ff7b14318d5128948a4709` |
| `tests/test_a5_snapshot.py` | `85d27f488d489934e703bc77336614ef7882120f39c0de6ec346264a99ac5dcc` | unchanged |

*Other fixed documents.*

| Path | SHA256 | Identity |
|---|---|---|
| `docs/prereg/qualification_panel.json` | `f0ad7f80d8d1d69c67b175c8e286628e7d0a0c56db39cef90869b28653018a16` | unchanged |
| `docs/prereg/h1w_gate_report.json` | `039b55857f90f92de4c3e190c69c3f15bd9e598c8680c0aa5dedfe0dab955dbd` | unchanged |
| `docs/reviews/council/board-a-freeze-2026-11-10.md` | `62ec0fb8fe49670c8aae1cff0360b461127078b8f6127804cd04849fb8445edc` | unchanged |

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

**A5.2.3 — closure reconciliation (10 Nov 2026; [BA-FRZ](reviews/council/board-a-freeze-2026-11-10.md) edit 1).**
Appended after the closed snapshot (`snapshot.json`, closed 2026-11-08T00:59:56Z, base `19f5cbf`); the snapshot content
is unchanged and the 7 Nov wording above stays as written. (i) The G6 census scored 187 members of 37 old-confirmatory
components (`snapshot.json:old_seal`); they are G6-exposed under this ledger, and their old status label cannot restore
freshness (A5.9.1). (ii) The primaries of the two regenerated predecessors `b7edbb9f8b339f52` and `86ea1fe0ec777cf8`
carry census run id `20261107-105825-56ef1597`, not a later run of `scripts/d8_g1_predecessors.sh` (its three logs are
empty); their `repeats/g1b` records were written earlier in the same session (log `d8-g6-census-20261107-102943.log`)
and equal the primary files byte for byte. The G1 closure is 14 bitwise-repeated entries including base. (iii) All forty
registered edits and their actual predecessors have current bytes; `983bb1066d0fd006` and `bc6edb1e3c913a10` are held
only through alias holders, and G3 staging must load and prove them under F3(i) before use. The three re-trained picks and
their predecessors remain flagged, development-exposed and are not historical bitwise replays. (iv) The
`snapshot.json:cost_ledger` rows overlap and mix units (run-wall-h, GPU-h, approximate GPU-h, CPU-h, summed run-wall-h);
the 0.05 GPU-h predecessor-primary row lies inside the 11.2 h census row. They are disclosed wall/CPU/approximate GPU
charges, not additive GPU-hours and not a reconciled cash spend. The G0/G1 history predates the procedural freeze and is
not re-dated.

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
Their enumeration was outstanding on 7 Nov; it is now `selection.json` (F1(a); listed in A5.1.2, 10 Nov 2026).
No G3 launch is permitted without it.

Keep all forty in the ledger (S31 §4.1).
Missing/nonfinite scores and audit exclusions remove only that contrast from its resolved denominator, with explicit reasons.
Use separate rotation and ladder-strength denominators; no borrowing between strengths.
An interval touching zero is unresolved; a zero prediction is disagreement (S31 §4.1).
Close the ledger before the single analysis; restrict variant-outcome access until closure.

**A5.3.1 clarification (chair ruling, 7 Nov 2026, before freeze; Board A confirms at freeze).** An edit whose actual predecessor state is lost (no stored or aliased bytes) has no incremental feature manifest and is therefore **ineligible** for selection; it does not block the selection. Root edits use the effective-delta norm `features.json:norms.fro_effective`; chain steps use the predecessor-relative norm of `runs/d8/a5_incremental_norms.jsonl`. Rows without an `eligibility` field are ineligible. Only D3-evaluated development members are eligible (sealed confirmatory components stay closed, A5.9). Under this ruling the selection of `docs/prereg/a5_manifests/selection.json` (`conditional`, 30 lora_sft 10/10/10 and 10 chain 4/3/3 by tertile, 40 distinct components; 3 chain picks with re-trained lineage, flagged) is the registered Part A selection; `references.json` and `text_groups.json` are rebuilt with `scripts/a5_build_manifests.py` after the G6 census closes and before G3, and their hashes enter A5.1 at freeze.

**A5.3.2 and A5.8.1 clarifications (chair rulings, 7 Nov 2026, before freeze; Board A confirms at freeze).** (i) The harmful side of the refusal-margin gradient panel has 20 prompts, the whole pool left after removing every scored and extraction prompt (`docs/prereg/a5_manifests/panels.json`); the 44 extraction prompts are not used, because they entered the construction of the abliteration edits. Refusal margin is a secondary objective; its field tokens are reported, never decisive. (ii) The pre-G2 null-size simulation cannot use edit/rotation/predecessor triples (no rotation is scored before G3). Its donor residual blocks are the per-window paired differences y(W_i + D_i) − y(W_i) between an evaluated edit and its actual predecessor (or the base for roots) from the TF-v1 records on file (`nullsize_inputs.json`), centred per pair; the rotation pair shares the predecessor and the windows, so this is the registered proxy for its noise structure, and the post-G2 power simulation uses the same blocks. (iii) The gradient panels are ranked lists of 500 (MMLU-Pro, HellaSwag) whose first 64 entries are the A5.8.1 sample; the WikiText gradient panel joins whole held-out articles (11 of the 60 test articles) until 64 windows are reached; the 500 reserved groups count as G2 exposure and any later sealed panel excludes them and the 11 articles.

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
Exact predecessor IDs, n_D, n_P and reuse proofs were outstanding launch blockers on 7 Nov, not assumed zero costs.
As of 10 Nov 2026, predecessor IDs and n_D = 19, n_P = 0 are in `references.json` (A5.1.2); alias reuse proofs remain G3 staging blockers (F3(i)).
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
**A5.9.1 — old-seal status after the G6 census (10 Nov 2026; [BA-FRZ](reviews/council/board-a-freeze-2026-11-10.md) edit 1).**
The sentence above is the 7 Nov historical wording. Since then the G6 census scored 187 members of 37 components of that
partition (`snapshot.json:old_seal`, A5.2.3). "Unspent" and "stays sealed" are therefore no longer a current certificate
for those components; they are G6-exposed. The partition remains barred from G7 (this clause) and from Part A selection
(A5.3.1 selects D3 development members only), so the forty picks are unaffected.

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
This clause is a blocking draft milestone, not fulfillment of the missing identity/reference evidence or authority to compute. No hashes, selected IDs or completed certifications are asserted here (7 Nov wording; the package is listed by sha256 in A5.1.2 on 10 Nov 2026, and Board A acceptance remains outstanding).

#### A5 freeze-prep amendments (7 Nov 2026)

Dated prospective amendment after Board A's A5 package review, ACCEPT WITH REQUIRED EDITS
([BA-PKG](reviews/council/board-a-a5-package-2026-11-07.md)); it executes that review's §8 checklist items 1 and 2 only.
It was written without reading any G3 outcome (none exists). Earlier A5 text is not rewritten; where it conflicts with
this block, this block governs prospectively. It authorises no compute and asserts no calibration, power or audit pass.

**F1 — registered adoptions (BA-PKG items 1–3, 5).**
(a) *Selection.* The forty picks recorded in `docs/prereg/a5_manifests/selection.json` (formerly labelled `conditional`)
are the registered Part A selection: 30 lora_sft 10/10/10 and 10 chain 4/3/3 by tertile, forty distinct components,
from 60 LoRA and 39 chain globally deduplicated representatives. The frozen `(effective_norm, canonical_edit_id)`
cutpoint keys are LoRA `(1.0580346482037204, f4b10903b8e09e24)`, `(2.2141261523655484, f8c2cf0c6b9ffa8c)` and chain
`(0.7725259468377004, 7af5977862ec021f)`, `(1.3049782925751559, 14cdaabee246493b)`. The exclusion of `1952501f1b2bae55`
(actual predecessor `c60bc6ef765e5e84`) is a prospective availability exclusion and requires a positive proof: the
predecessor has an inventory and artefact-status sidecar row, no stored bytes, sidecar `delta_present` false, empty
`alias_of`, and no alias-ledger holder with bytes (`scripts/a5_build_manifests.py:lost_predecessor_proof`; no error
string is inspected). It licenses no other drop: a missing norm of an otherwise eligible, reconstructible edit still
blocks. The re-trained chain picks `057dcfbedea3251f`, `0bc0a0c4ad0520e0` and `05531ce1db4df63f` are retained without
replacement; their estimand concerns the actual current successor and predecessor bytes, not bitwise replay of the
lost history; they and their connected relatives remain development-exposed and ineligible for fresh G7 use. The
later rebuild against the closed snapshot must reproduce exactly these forty IDs, forty distinct components and these
cutpoints (`check_procedural_freeze`); if G6 closure changes independence, selection or ranks, stop and amend
prospectively, never rerank silently.
(b) *Harmful refusal panel.* The 20-prompt harmful panel (the whole unexposed pool) is registered and overrides
A5.8.1's generic 64-group clause for `refusal_margin_harmful` only. The exception is enforced in the schema
(`panels.json` `registered_size` 20) and by the consumer (`src/wsp/features/response_field.py:registered_field_items`,
which refuses 64 or any other count). Prompt weights are equal separately within the harmful and harmless roles; the
two roles are never pooled as 84 observations. The objective stays descriptive and cannot become decisive or be
promoted after outcomes; any gate applied per objective still applies.
(c) *Extraction fill rejected.* The 44 direction-extraction prompts entered the construction of the abliteration edits.
Their full normalized-text groups are excluded; generated outputs list them only as `extraction_fill_rejected`.
(d) *Ranked panels.* The 500 ranked MMLU-Pro and HellaSwag groups are frozen and the field uses exactly the first 64 in
rank order, never another prefix. All reserved members are G2 exposure even where only the prefix is used: 512
reserved MMLU-Pro item IDs (500 groups) and 589 reserved HellaSwag IDs.
(e) *WikiText.* The whole-article rule is registered: whole held-out articles in rank order until 64 windows, i.e. 11
of the 60 test articles, all 11 exposed including unused tails; 12 further articles are development-touched and 37
remain an upper candidate pool, not a freshness certificate. Windows can share or straddle articles, so 64 windows are
not 64 independent text groups; article membership, tokenization, join and boundary rules are preserved, and any
item-level inference must respect the article grouping.

**F2 — executable NULL-SIZE proxy (BA-PKG item 4; supersedes A5.3.2's "(edit, rotation and predecessor reference)"
residual sentence, the `residuals[64,3]` payload and the `U_edit − U_rotation` construction).**
*Donor.* For each eligible development pair i whose edit and actual predecessor (base for roots) both have TF-v1
primary records with the same scoring identity (config hash, model revision, source hash, device) on the same 64
registered windows, U_i[w] = y(W_i + D_i)[w] − y(W_i)[w], with y the per-window mean NLL (`extra.mean_nll`).
*Masks and missingness.* `window_mask[i,w]` is false where either record lacks window w or its value is nonfinite; the
entry is then NaN/null and is never imputed. A pair with any masked window has all three simulated contrasts missing in
every replicate (counted as failures, removed from the resolved denominators, no replacement), whether or not the
outer panel redraws that block. A row lacking either record is not a donor and is listed in `provenance.missing`.
*Centring versus predecessor subtraction.* Predecessor subtraction defines U_i; centring is a separate step within the
pair over its valid windows: `centred_noise[i,w]` = U_i[w] − m_i with m_i the mean of the valid U_i[w]. The nonzero
donor effect m_i is pinned separately as `donor_mean[i]`. The payload holds `centred_noise` and `window_mask` as
[n_pairs, 64] arrays and `donor_mean` as [n_pairs], row-aligned with the donor rows; blocks are the 32 two-window
means (2k, 2k+1) of the centred noise.
*Three contrasts.* With scales a = (1, 0.25, 0.5) for (rotation, ladder 0.25, ladder 0.5), slot k drawing donor i:
null x_c = a_c · σ_k · (m_i + noise block_i), with one fair-coin σ_k per simulated component per replicate shared by the
mean and noise of all three contrasts; registered alternative x_c = a_c · (|m_i| · sign(prediction_c) · B_k + noise
block_i), with one B_k per component, P(B_k = +1) = 0.75, and noise not sign-flipped. This registers the alternative
scale: |m_i| for the rotation contrast and a_c|m_i| for the ladders. The resolution step (32 blocks, 4,000 paired
draws, A5.3.2 seeds/streams) is applied to each contrast separately.
*What is shared.* One donor block, mean and noise, per slot is shared by the three contrasts; the outer 32-block panel
draw and the [4000, 32] resolution draws are common to all slots and contrasts. Because the ladder blocks are
rescalings of the rotation block, under this proxy their null resolution and agreement coincide with the rotation
contrast; the proxy carries no strength-specific or rotation-specific noise.
*Audits.* Each donor carries explicit `audit[3]` in (rotation, 0.25, 0.5) order and `audit_evidence[3]` in {computed,
unavailable}; neither is defaulted. The rotation audit is the A5.3.3 outcome-free audit of the donor's bytes; the ladder
audit is the scaled-edit construction. Unavailable evidence, including the eight alias reuse proofs and G3 staging
audits, is never a default pass. The registered run counts it as a pass (more pairs, conservative for size) and the
frozen scenario `unavailable_audit_fails` counts it as a failure; both are reported. `variance_x2` (centred noise
doubled, means unchanged) is the frozen variance-inflation scenario; A5.3.2's constant-positive-mean and
cell-specific mask/noise sensitivities stand.
*Timing.* The payload (inventory, donor rows, noise, masks, means, audits and evidence, provenance) is materialised once
from the closed G0/G1 snapshot, validated with `scripts/a5_simulation.py --inputs-spec`, hashed and committed before
NULL-SIZE. Masks and audit flags are never updated from G2 or G3 outcomes; POWER-AND-YIELD uses the identical file.
*Scope.* The proxy is not an observed rotation contrast: in the real comparison the predecessor cancels and the proxy
misses rotation-specific heteroskedasticity, edit/rotation covariance, orientation-dependent resolution, rounding and
nonlinear ladder response; its variance can err in either direction. NULL-SIZE and POWER-AND-YIELD are conditional on
this surrogate and the sign-flip DGP, are not an empirical certification of the rotation null, and are not tuned after
G3. No rotation record is fabricated and predecessor data is never relabelled as a rotation. The ≤0.07 gates stand.
Implementation: `scripts/a5_simulation.py` (`donor_block`, `contrasts`, `donor_inputs`, `simulate`, `validate_inputs`)
and `docs/prereg/a5_manifests/nullsize_inputs.json` (schema `a5-nullsize-inputs-spec-v2`, `proxy.allowed_sources` =
`tf_v1_edit_minus_actual_predecessor`).

**F3 — A5.11 and A5.3.1 staging reconciliation (BA-PKG item 7).** The eight alias reuse proofs and the realised
tensor-coverage, single-matrix and rotation audits are G3 staging outputs, not pre-G2 package items; A5.11's list and
A5.3.1's "commit … closure proofs, tensor coverage" are read accordingly. Frozen now:
(i) *Alias/reference reuse proof.* Algorithm: a record is reused only if the holder's current entry content hash equals
the record's `entry_hashes`, the record is complete and primary, its scoring identity matches (effective config hash,
model revision, benchmark counts), and its 64 window token hashes equal `text_groups.json`
(`a5_build_manifests.reference_status`). Inputs: `references.json` rows, holder bytes, `eval.json`/`records.jsonl`.
Output per proof: alias and holder IDs, holder content hash, recorded and current entry hashes, identity fields, window
check, failed checks, verdict. Gate: G3 staging, before the affected reuse. Fail-closed: a failed or missing proof
forbids reuse; the reference is scored new and charged under n_D/n_P with stop-and-reprice, or, if the bytes cannot be
identified, the pair is excluded with its reason and no replacement. Any reuse already relied on by G0/G1 or by pre-G2
donors needs its proof before that use.
(ii) *Tensor coverage, single-matrix and rotation audits.* Algorithm: A5.3.3 on the pinned stored bf16 states of the
selected edit and its actual predecessor, seeds per `control_seeds.json`, code `src/wsp/features/rotation.py` at its
pinned hash. Output per edit: edit and predecessor IDs, seed, per-tensor name/shape/role, block rows and columns,
numerical rank, norm and spectrum mismatch, rank change, support identity, reconstruction residual, reason code
(`non_2d`, `non_finite`, `zero_block`, `rank_degenerate`, `not_single_matrix`, unsupported) or pass, coverage counts,
unrotated-tensor count and pair verdict. Gate: G3 staging, before any rotation of that edit is scored. Fail-closed: any
tensor failure or a missing audit excludes the whole rotation pair from the rotation denominator with its reason; the
pair stays in the forty-ID ledger and is not replaced.
Metadata-level closure, the selected IDs, actual predecessor availability and the outcome-independent coverage rules
remain freeze blockers; realised G3 reports do not.

**F4 — future sealed builder obligation (BA-PKG item 6).** Any future sealed builder (the `TF-v1:item:` MMLU-Pro panel,
any sealed HellaSwag or prompt panel, the A4.5/A5.6 sealed WikiText panel and any G7 panel) must, before G7
qualification opens any outcome: load the frozen A5 manifests by their A5.1 sha256 and the complete exposure ledger
(A5.2); exclude all 500 reserved MMLU-Pro groups (all 512 reserved item IDs) and every legacy, development, pilot,
calibration and later-exposed group before ranking (a different hash domain is not enough); exclude all 589 reserved
HellaSwag IDs; exclude the 12 development-touched and the 11 auxiliary WikiText articles whole, not only used windows or
identical token hashes, plus later calibration/pilot exposures; exclude every exposed prompt group (gradient, factor,
probe, scored refusal/harmless) and every direction-extraction group; reject any group or article intersection,
fail-closed; and write and commit a certificate (inputs by sha256, exclusion counts, zero-overlap checks per excluded
set). Until such a builder runs, future-seal statements in `panels.json` are obligations (`future_seal_obligations`),
not computed certificates.

**F5 — procedural freeze (BA-PKG checklist item 2).** The procedural freeze is dated 7 Nov 2026 and is distinct from, and
earlier than, the pre-G2 evidence freeze (checklist items 3–7); no evidence hash existed at this date and none is
claimed. `selection.json:procedural_freeze` records the inventory hash, the cutpoints above, the forty IDs and their
hash, the re-trained picks and the sha256 of `code_hashes.json`. `code_hashes.json` (schema `a5-code-hashes-v2`) pins
the bytes committed with this amendment (its `git_sha` is their parent) of both builders, the simulation/validator,
the A5 tests, the imported selection/grouping/reference modules and `configs/eval_tf_v1.yaml`, together with that
config's effective hash as the TF-v1 records store it, the model and tokenizer revision from
`configs/editor_qwen3.5-0.8b.yaml`, and the Python, numpy, torch and transformers versions. Not pinned at this freeze:
feature-file and donor/record contents and the closed G0/G1/G6 snapshot; `selection.json:inputs` remains a partial input
index, not a full immutable snapshot. Existing G0/G1/G6 observations keep their actual dates and exposure history.
Generated labels now read "registered" where the rulings register them and "rejected" for the extraction fill (the
pre-ruling literal selection run is kept as a superseded audit record); components, references, text groups and panel
IDs were not rebuilt against live data, which remains checklist items 4–5 after the census.


#### A5 amendment F5 (10 Nov 2026, before freeze; Board A reviews at the freeze)

Authority: [10 Nov chair synthesis](council/2026-11-10/SYNTHESIS.md), §§1, 4–6, and
its four round-3 reports. This prospective block appends to, and does not repair,
the historical registration. The earlier **F5 — procedural freeze** retains its
7 Nov meaning; F5.1–F5.7 below identify this separate 10 Nov amendment. No G2/G3
outcome is used. New settings are planned chair defaults for Board A review, not
measurements. A5.1 hashes/acceptance remain required before execution. The field
remains sole primary; A5.5's ordered funding decision is unchanged.

**F5.1 — secondary LoL family (extends A5.4.3).** Register L (learned-frame
GL-set head), R (random-frame control), K (K-FAC-anchored frame), and S
(singular-values-only control), computed on CPU. Per supported matrix use the
realised increment D = fp32(bf16(W + Δ)) − fp32(W) at its pinned actual
predecessor; truncate its SVD at the recorded rank: r for LoRA/chain-LoRA,
1 for random-direction/abliteration, 32 for task vectors, with omitted energy
recorded. Unsupported/missing-rank rows are listed before fitting, never silently
assigned raw factors. Raw α/r-scaled stored adapters are a separately named
sensitivity on the authentic-factor subset, never substituted for scored D.

Following round3-representation §3, L uses GL-equivariant left/right frames
shared within role across layers, width 4 with layer/role embeddings; R uses
frozen Haar frames of width 8, SHA256 domain `20261110:lol-random:` plus tensor
identity (first eight bytes, big-endian, modulo 2^63; PCG64, positive-diagonal QR).
K uses frozen top-8 G2 WikiText K-FAC eigenframes, with block
M = (Λ_G^(1/2) Q_Gᵀ B)(A Q_A Λ_A^(1/2)); S sees only log singular values
1–8 and tail energy, with zero/presence flags. Pin eigenbasis/tie conventions and
frame hashes before fitting. Gauge refactorization (B,A)→(BQ,Q⁻¹A) must leave
predictions invariant; audit with fixed well-conditioned Q, retaining discrepancies.

The compact shared readout uses padded/masked tokens (vec(M), log spectrum,
tail energy, layer/24, full-attention flag), a 75→16→16 MLP with role embedding,
masked sum/mean pooling and a linear endpoint head: about 1.6k parameters
(1,617 for the representation report's one-endpoint head). Learned frame
parameters are additional: do not call the entire L encoder 1.6k (up/down
frames alone have 36,864 in that report). S masks directional inputs; padding
and masks are fixed, not fitted to outcomes. Return f(D)−f(0), exactly zero at
zero edit; use explicit zero flags for log spectra. Freeze complete input dimensions
and actual per-arm parameter counts under A5.1, including the F5.2 fusion inputs.

Fit L1 loss to training-only nuisance residuals, AdamW lr 1e-3, weight decay
1e-2, batch 16, at most 300 epochs. Use four ancestry-component inner folds
inside each outer training fold, patience 30; take median inner best epoch,
then refit outer training rows. Seeds 0/1/2 are prediction-averaged before loss.
This nested early stopping is a secondary-only exception to A5.4.1; no outer
held-out row selects epochs, transforms or architecture. WikiText is the endpoint;
MMLU remains descriptive/unqualified. Evaluate on both identical eligible LoRA-only
rows/inherited folds and the 386-logical-row G1 H1w folds (eligibility exclusions
reported, not 386 independent examples). No sealed rows enter development.
Compare L/R/K/S and each against same-row augmented StatNN; the secondary tree
comparator gets inner-only tree count choice {100,300,600}, all other A5.8.4
settings fixed. Report conditional and joint paired ratio intervals exactly as
A5.4.2 (C_X:Y = 1−A_X/A_Y, ancestry clusters/common items, guards and failures).
The family is descriptive/exploratory, cannot fund G7 or any other purchase,
and cannot replace the primary after outcomes. Intervals crossing zero are
inconclusive. The **family-only kill** is valid L-versus-S paired upper bound
≤0 on LoRA-only rows in both conditional and joint procedures; undefined or wide
intervals are unresolved. This alone does not stop the entire 0.8B line.

**F5.2 — anchored GL-set head.** Register K+F, taking the K blocks above plus
per-tensor field tokens (signed gradient contractions, q and predecessor cross
term c) before pooling, with field-only F and anchored-only K comparators.
Use F5.1's compact zero-at-zero head, L1/nested recipe, rows, folds and paired
A5.4.2 intervals. Keep masks/role/metadata and training opportunity matched;
remove every K-derived input for F and every field-derived input for K.
Report K+F:F and K+F:K beside K:R and L:S; include each against augmented
StatNN with all corresponding scalar inputs. These are secondary descriptive
contrasts with no funding or promotion power. This does not change the primary
tree/no-field/destruction-control family in A5.4.

**F5.3 — capped 4B scale diagnostic.** Register 20 outcome-blind Gaussian edits
of pinned Qwen3.5-4B plus one shared 4B base. Before scoring, hash the model,
tokenizer, supports and 20 seeds. Planned chair default: radius ρ=0.005,
D=ρ||W_support||_F Z/||Z||_F, Z iid standard Gaussian on all supported floating-point
linear-weight matrices; zero/nonfinite norms fail closed. Pin the exact tensor list
and realised bf16 rounding diagnostics. Derive each seed from SHA256 of
`20261110:4b-gaussian:` plus draw index 0–19 (first eight bytes, big-endian,
modulo 2^63; PCG64). No radius or support selection from wins.
Score the same 64-window WikiText log-PPL/mean-NLL endpoint, with shared-reference
paired per-edit changes and resolution intervals under A5.3, and report every
edit, improvement fraction (NLL reduction with paired interval excluding zero),
exact binomial uncertainty and unresolved fraction. No predictor is fitted.
Run on a Lambda H100 in parallel with local G2/G3 after the registration freeze;
first time and qualify the actual 4B endpoint/device under A3, including matched
identity/repeat and precision. Otherwise label all results **fast-tier only**,
never qualified TF-v1. Record all panel/source exposure, qualification, failures,
references and measured costs under A5.2/F4; no exposed group becomes fresh later.
This is a diagnostic of density/effect resolution, never a gate, never G7, and
cannot delay, condition or resize Part A. Stop scale expansion if effects remain
inconclusive or qualification fails; no automatic larger zoo. The **1.82 TF-v1
GPU-h** estimate is an **unmeasured linear size extrapolation**:
21 × (4/0.8) × 1.04 min / 60, excluding cold load, qualification/repeats and edit
construction. The planned Lambda cap is 2 GPU-h including qualification;
stop/reprice before exceeding it, never report a favourable incomplete subset
as the completed diagnostic. This amendment starts no GPU job.

**F5.4 — H5 critic contract.** The critic schedules real evaluation, ranks or
abstains; measured held-out support improvement authorises acceptance. Pin immutable
base, incumbent and candidate hashes. Diagnose/train proposals on proposal support,
validate on disjoint held-out support, rebuild every accepted update from the same
pinned base (never compound patches), reload/verify the effective delta, and validate
the rebuilt candidate on an untouched acceptance slice if its bytes changed.
Only a measured qualifying improvement over the actual incumbent under A5.7's
feasibility/safety predicates is accepted; otherwise retain/restore the incumbent
(or base initially). Preserve base-relative storage and actual-incumbent targets
separately. Keep rollback/reversibility checks and a complete ledger including
invalid, rejected, failed, no-op and reverted attempts. Reused support feedback is
adaptive development; sealed outcomes never select updates or refit the critic.

Log four separate Boolean treatment flags, fixed across matched critic/no-critic
arms: `python_tool_access`, `forward_access`, `hidden_state_access`,
`feedback_access`, with actual calls, code/decode hashes and costs. Python access
does not imply forward access; hidden-state and feedback access are independently
specified. Compare architecture-compatible random permutations of proposed updates,
matched on changed count/norm and validation/tuning opportunities; distinguish
these from critic-feature permutations.

A matched random **best-of-N** arm is mandatory for any editor-advantage claim:
32 independent Gaussian draws per source block, matched effective support/norm,
fixed seed order and scale before outcomes, with nested prefixes N=1/8/32.
Select each prefix's single winner using development support only, then evaluate
on held-out items under the frozen policy. Match editor opportunities at each N
and report equal-real-evaluation and equal-GPU costs, all draws/failures included.
Retain existing G5 arms; G5 is shadow feasibility, not production rejection.
Single-model selection (K=1) is separate from RandOpt-style top-K ensembles;
any ensemble requires its own registration and multi-pass inference accounting.
A5.7's ≥30% realised-savings, score noninferiority, missed-winner, power and net
compute contract is unchanged; forecasts never substitute for measured acceptance.

**F5.5 — joint line-stop rule at 0.8B.** Stop this weight-space world-model line
at 0.8B and publish the measurement paper as its final word only when all three
valid predicates hold jointly:
(a) Part A resolves ≥15 rotation pairs with agreement CP upper bound ≤0.6;
(b) every registered orientation contrast in Part B has both conditional and joint
95% upper bounds <0.05 at simulated power ≥0.80 for both procedures;
(c) learned L does not beat S on LoRA-only rows, meaning valid conditional and
joint L:S paired upper bounds ≤0 as in F5.1.
For (b), enumerate before fitting C_abl, C_rand, C_perm, K:R, L:S,
K+F:F and K+F:K, plus the registered chart intact-versus-random/permuted-frame
contrasts; use A5.4.2 and freeze the development-only power simulation for these
actual ratios at C=0.15 with shared component/item covariance. Invalid/ineligible
controls or missing power cannot satisfy (b). Apply A5.5 validity first. Any one
predicate failing, missing, underpowered or undefined leaves the line **unresolved**,
not killed. This bounded line-stop is separate from funding and is no universal
scientific falsifier: Part B still reuses examined outcomes under A4.4. Funding
stops on a registered no-go regardless of this conjunction; the Part A alternative
also stops field funding under A5.5. No secondary supplies a funding fallback.

**F5.6 — brief corrections.** Withdraw the LoL “50,000 times faster” claim;
it was not found in the fetched text. WeightLoop beats random edits in **8 of 8**
settings and the original predictor in **7 of 8**, not 7 of 8 versus random.
No chain LoRA factors are stored: chain steps are dense snapshots; their F5.1
factors are reconstructed realised-increment SVDs, never original adapters.
The 122 stored adapters are **121 lora_sft + 1 heretic_ara** (round3-representation
§2); stored adapters are not independent labelled rows or a power calculation.

**F5.7 — null-size caveats and interpretation.** Record the evidence in
`docs/prereg/a5_manifests/nullsize/report.json` and its README: the three
contrasts rescale one donor block, so this is **one test, not three**; the
registered null returns alternative in 1,939/10,000 replicates (**19.4%**, exact
rate 0.1939), beside go 180/10,000 (0.018); the LoRA low tertile has one donor
reused for all ten slots; donor means dominate within-pair noise (SD about
0.004 nats per synthesis §5), explaining proxy yield 1.0. These are limitations,
not rotation-null or power certification. Keep the registered ≤0.07 size gate.
Every Part A report must show the alternative branch's null count/rate and
binomial interval next to the ordered go rate, for the registered and reported
sensitivity scenarios; distinguish those null rates from observed branch outcomes.
An **alternative** outcome is a **funding stop for the field**, never evidence
of wrong signs. Do not reinterpret its high null frequency as directional
falsification or alter A5.5's branch order retrospectively.

#### F5 schema pins (10 Nov 2026)

Prospective, before any G2 field, K-FAC factor, LoL/K+F implementation or fit exists ([BA-FRZ](reviews/council/board-a-freeze-2026-11-10.md)
edit 3). It completes F5.1–F5.2 and, where it conflicts with them, governs. Chair decisions are marked. No implementation
of L, R, K, S, F or K+F exists at this freeze: they are **staged secondaries**, and their code and a settings file
reproducing every value below must be committed and listed by sha256 in A5.1 by a dated amendment before any affected
fitting. Missing code, settings or hashes leave the family unresolved.

*Matched width (chair decision).* Every frame has width **4**: L's learned frames, R's Haar frames and K's eigenframes.
This replaces F5.1's width-8 R and top-8 K so that the directional block has the same dimension (4×4 = 16) in L, R and K.
Widths are matched; L still differs from R and K by fitting its frames, which is the intended contrast. The spectrum
columns keep σ_1..σ_8.

*Increment, rank and energy.* For each supported matrix tensor t (12 roles, 186 slots: `mlp.{gate,up,down}_proj` ×24,
`self_attn.{q,k,v,o}_proj` ×6, `linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}` ×18, as
`src/wsp/zoo/arch.py`), D_t = fp32(bf16(W_t + Δ_t)) − fp32(W_t) at the pinned actual predecessor. Full thin SVD
`torch.linalg.svd(D_t, full_matrices=False)` in fp32 on CPU, singular values descending. Recorded rank r: the adapter's `r`
for LoRA and chain-LoRA steps, 1 for random-direction/abliteration, 32 for task vectors; unknown r excludes the row
before fitting. Balanced factors B = U_r diag(σ_r^½), A = diag(σ_r^½) V_rᵀ, so BA is the rank-r truncation and sign or
basis choices inside the SVD cancel. Omitted energy Σ_{j>r} σ_j² / Σ_j σ_j² is recorded as a diagnostic, not an input.
A tie at the truncation (σ_r − σ_{r+1} ≤ 1e-6 σ_1) is flagged `rank_cutoff_tie` and kept.

*Token schema (39 columns, this order).* 0–15 vec(M_t) row-major (4×4); 16–23 log σ_j(D_t), j = 1..8, from the
untruncated spectrum, 0 where absent; 24–31 presence flags 1[σ_j > 1e-3 σ_1] (the A5.3.3 rank tolerance); 32 tail energy
Σ_{j>8} σ_j² / Σ_j σ_j² (0 when D_t = 0); 33 layer/24 with layer index 0–23; 34 full-attention flag; 35 the endpoint
objective's field contraction ⟨g_t, D_t⟩ (WikiText head: `wikitext_nll`; descriptive MMLU head: `mmlu_gold_log_prob`);
36 q_t; 37 cross term c_t (0 for roots); 38 field presence flag (1 iff 35–37 are computed and finite, otherwise 35–38 are 0).

*Directional blocks.* L: M_t = (Φ_ρᵀ B)(A Ψ_ρ), Φ_ρ ∈ ℝ^{out×4}, Ψ_ρ ∈ ℝ^{in×4} learned per role ρ and shared across
layers, initialised for training seed s by Haar frames from seed(`20261110:lol-learned-init:`, `<role>:<s>`). R:
M_t = (Q_outᵀ B)(A Q_in), Haar frames from seed(`20261110:lol-random:`, `<exact tensor name>`), numpy
`Generator(PCG64(seed))`, `standard_normal((out, 4))` then `standard_normal((in, 4))`, QR with positive diagonal (zero → +1).
K: M_t = (Λ_G^½ U_Gᵀ B)(A U_A Λ_A^½) with the top-4 eigenpairs of the base model's G2 WikiText K-FAC factors of tensor t
(A is in×in; the Qwen3.5 linear maps have no bias). seed(domain, key) is the `control_seeds.json` rule (SHA256, first eight
bytes big-endian, mod 2^63). Frames are written with per-tensor SHA256 to a frame manifest before fitting.

*Eigen conventions.* Factors in fp64, symmetrised ½(F + Fᵀ), undamped, `torch.linalg.eigh` on CPU; eigenvalues descending
with index order on exact ties; A5.8.1's negative-eigenvalue tolerance rounds small negatives to 0 (larger violations
already fail G2); Λ^½ of a zero eigenvalue is 0. Sign: each eigenvector's largest-|·| entry is positive (lowest index on
ties). A cutoff tie (λ_4 − λ_5 ≤ 1e-9 λ_1) masks that tensor's K block with flag `k_cutoff_tie` (reported, no
replacement); ties inside the top four are flagged `k_inner_tie` and kept.

*Per-arm masks (fixed, never fitted).* L, R, K: columns 0–34; S: 16–34 (directional block zeroed); F: 16–38; K+F: 0–38
with K blocks. F carries no K-derived input and K no field-derived input. Masked columns are exact zeros.

*Zero flags and pooling.* A token is present (m_t = 1) iff tensor t is supported and D_t ≠ 0 after rounding; absent
tensors are padding. Head: h1 = GELU(W1 x_t + b1 + E[ρ_t]) (16), h2 = GELU(W2 h1 + b2) (16), with E a 12×16 role
embedding and exact `torch.nn.GELU()`; pooled = [Σ_t m_t h2_t, Σ_t m_t h2_t / max(1, Σ_t m_t)] (32); f = w·pooled + b0;
the prediction is f(D) − f(0) = w·pooled, exactly zero at zero edit. Separate models per endpoint.

*Complete parameter counts per endpoint model* (nominal; "effective" counts only weights attached to unmasked columns):

| Arm | Head nominal | Head effective | Learned frame parameters | Total nominal |
|---|---|---|---|---|
| L | 1,137 | 1,073 | 161,920 (81,920 on the six LoRA roles; up/down 36,864) | 163,057 |
| R | 1,137 | 1,073 | 0 | 1,137 |
| K | 1,137 | 1,073 | 0 | 1,137 |
| S | 1,137 | 817 | 0 | 1,137 |
| F | 1,137 | 881 | 0 | 1,137 |
| K+F | 1,137 | 1,137 | 0 | 1,137 |

Head: W1 39×16 + b1 16 = 640; role embedding 192; W2 + b2 = 272; linear head 33. This replaces F5.1's 75-input head and
1,617-parameter count. L's frames: Σ_ρ 4(out_ρ + in_ρ) over the 12 roles.

*Gauge audit.* For every held-out row and supported tensor with factors, Q = H diag(d), H Haar r×r from
seed(`20261110:lol-gauge:`, `<canonical edit id>:<exact tensor name>`) via `torch.Generator('cpu').manual_seed`,
`randn(r, r)` in fp64, positive-diagonal QR; d_j = 2^((j mod 3) − 1), so cond(Q) ≤ 4. (B, A) → (BQ, Q⁻¹A) in fp64, cast
to fp32. Every fitted L, R, K and K+F model (each seed, each outer fold) is evaluated on both inputs. Tolerance: |Δf| ≤
1e-5 + 1e-4 |f| nats per row. All discrepancies are retained; any violation makes that arm's contrasts undefined.

*Folds, stopping and refit.* Outer folds: the registered H1w source-closed folds (§3, A4), and their restriction to
LoRA-only rows. Inner folds: the outer-training ancestry components hash-sorted by sha256(`20261110:lol-inner:` +
component id) ascending and assigned round-robin to 4 folds. Per seed s ∈ {0, 1, 2} (`torch.manual_seed(s)` before
module construction in a fixed order), each inner fold trains on the other three for up to 300 epochs. Its best epoch is
the argmin of held-out L1 (earliest on ties), with stopping after 30 epochs without strict improvement. The refit epoch
count is floor((e_(2) + e_(3)) / 2) of the four inner best epochs (at least 1); refit on the outer training rows; average
the three seeds' predictions before the loss. Eligibility is set once for all arms: a row with any supported changed
tensor lacking r, a finite SVD or a K frame is excluded from every arm, and the exclusions are listed before fitting.

*Augmented comparators.* Arm X's same-row augmented StatNN receives v4 pooled statistics plus X's masked 39-column token
for every one of the 186 matrix slots (zero padding plus a presence column), flattened in A5.8.3 order. L's learned frames
cannot be given to a tree, so L's comparator uses the R-frame directional columns; this is disclosed, not called
identical information. The tree count is chosen on the same inner folds from {100, 300, 600} (lowest mean inner L1,
smaller count on ties); every other A5.8.4 setting is fixed.

#### A5 freeze-pass amendments (10 Nov 2026)

Authority: [BA-FRZ](reviews/council/board-a-freeze-2026-11-10.md), required edits 1–6, executed in its item-7 order. This
prospective block is appended and repairs nothing historical; where it conflicts with earlier A5 text it governs.
No G2/G3 outcome was read (none exists), no field was computed and no GPU was used. Settings marked "chair decision" were
left open by the review and are planned criteria, not measurements. The schema pins of edit 3 are the separate block
"F5 schema pins (10 Nov 2026)" above.

**FP1 — closure reconciliation (edit 1).** Recorded as A5.2.3 and A5.9.1 and in the package README section "Closure
reconciliation (10 Nov 2026)", generated by `scripts/a5_build_manifests.py`. In short: 187 census-scored members of 37
old-confirmatory components are exposed, so "unspent / stays sealed" is historical wording, not a current certificate.
The two regenerated predecessors' primaries come from census run `20261107-105825-56ef1597`, and their earlier g1b repeats
are byte-identical. All forty edits and predecessors are available, two only through alias holders that G3 staging
must prove. Cost rows overlap and mix units, and the 0.05 h predecessor-primary row is inside the 11.2 h census.

**FP2 — NULL-SIZE sensitivities run; alternative-branch interpretation (edit 2).** F2's "A5.3.2's constant-positive-mean
and cell-specific mask/noise sensitivities stand" is executed. Both are frozen, outcome-independent scenarios of
`scripts/a5_simulation.py` (`SCENARIOS`) on the unchanged payload `data/runs/a5/nullsize/inputs.json`, run at registered
scale (10,000 replicates, 4,000 resolution draws, `SeedSequence(20261107).spawn(3)`). Each consumes every random stream
exactly as the registered run does.
(i) `constant_positive_mean` (chair decision on the constant): every donor mean is replaced by its family's positive
constant, the mean |donor_mean| over that family's unique donor-cell representatives. This is the registered
alternative's |m_i| scale averaged within family, giving LoRA 0.080957 and chain 0.023787 nats. Noise, masks,
audits, selection and slots are unchanged. Under the fair-coin null this equals a fixed positive truth with independent
fair-coin prediction signs.
(ii) `cell_mask_noise` (chair decision): in the single-donor LoRA low-tertile cell, the ten slots are cut to the registered
minimum of one slot per donor component (1; the first slot in slot order). The other nine are masked as missing in all
three contrasts, with no replacement. That cell's centred noise is doubled. The review names no cell-specific noise scale,
so the chair's fallback of noise ×2 in that cell only is used. All other cells are unchanged.
Results (exact two-sided 95% Clopper–Pearson intervals; the three contrasts coincide under the proxy):

| Scenario | Two-sided size (each contrast) | Ordered go | Alternative (funding-only) | Yield ≥15 | Decisions go / alt / unresolved |
|---|---|---|---|---|---|
| `registered` | 0.0372 [0.0336, 0.0411] | 0.0180 [0.0155, 0.0208] | 1939/10,000 = 0.1939 [0.18619, 0.20179] | 1.0000 | 180 / 1939 / 7881 |
| `variance_x2` | 0.0347 [0.0312, 0.0385] | 0.0162 [0.0138, 0.0189] | 1884/10,000 = 0.1884 [0.18078, 0.19620] | 1.0000 | 162 / 1884 / 7954 |
| `unavailable_audit_fails` | 0.0000 [0.0000, 0.0004] | 0.0000 [0.0000, 0.0004] | 0/10,000 = 0.0000 [0.00000, 0.00037] | 0.0000 | 0 / 0 / 10000 (vacuous: nothing resolves) |
| `constant_positive_mean` | 0.0406 [0.0368, 0.0447] | 0.0191 [0.0165, 0.0220] | 2177/10,000 = 0.2177 [0.20964, 0.22592] | 1.0000 | 191 / 2177 / 7632 |
| `cell_mask_noise` | 0.0307 [0.0274, 0.0343] | 0.0155 [0.0132, 0.0181] | 1544/10,000 = 0.1544 [0.14737, 0.16163] | 1.0000 | 155 / 1544 / 8301 |

The registered gate (ordered go rate and every individual two-sided size ≤ 0.07, registered run) is unchanged and passes.
Every sensitivity scenario's go rate and sizes are also ≤ 0.07. Raw outputs: `data/runs/a5/nullsize/<scenario>.json`
(hash pins in A5.1.2; the 8 Nov raw outputs are unchanged). Reports and gzip traces: `docs/prereg/a5_manifests/nullsize/`.
Determinism check: the registered scenario was re-run with the amended script, to a scratch path outside `data/`, and
reproduced the 8 Nov raw output byte for byte (sha256 `452c078aae520dfedbcf25a44a9390305612d32ade3793840f3118043e4239e8`).
In `cell_mask_noise`, the nine masked slots account for its 90,000 audit/nonfinite failures per contrast, so 31 slots
stay active and yield is still 1.0. These sensitivities share the proxy's limits (F2, F5.7). They are not a
rotation-null certification and add no donor diversity to the LoRA-low cell.
*Positive-go size versus funding-only alternative.* The ≤ 0.07 gates are positive-go error control: ordered go and
each individual two-sided test. The **alternative** branch is a funding-only stop for the field (F5.7). Under this
fair-coin null (p = .5), agreement lies inside the region that rule is meant to stop funding, so its null rate is not
a type-I or wrong-sign error, and no ≤ 0.07 gate applies to it. A5.3.2's "whole-decision error" reads as the ordered
go rate. No decisive negative inference (statistical exclusion of useful agreement) and no wrong-sign claim is
registered. Either would first need its own boundary-null calibration at p = .60, with a false-alternative rate ≤ 0.07
at nominal 0.05 including resolution, masks and branch order, plus prospective power, all registered before outcomes.
None is registered here.
*F5.7 reporting rule (amended).* Every Part A report shows, for the registered run and for each of the four
sensitivity scenarios, the alternative branch's null count, rate and exact CP 95% interval beside the ordered go rate,
as in the table above. It keeps these null rates distinct from observed branch outcomes. A5.5's branch order is unchanged.

**FP3 — F5.1–F5.2 schema.** See "F5 schema pins (10 Nov 2026)".

**FP4 — 4B paired qualification panel (edit 4; completes F5.3, never transferred from the 0.8B pilot).**
*Panel.* The shared Qwen3.5-4B base plus five edits named by F5.3 draw index: 0, 4, 8, 12 and 16 (seeds from
`20261110:4b-gaussian:` + index, as F5.3). The choice is outcome-blind; scoring them is exposure under A5.2/F4.
*Devices and repeats.* The panel is scored on H0 and on the Lambda H100. Each device gets two independent repeats, each
in a fresh process with a fresh model load and its own immutable run id (project `YYYYMMDD-HHMMSS-<hash>` form, repeat
labels `q4b-r1`, `q4b-r2`). Failed attempts are kept. If H0 cannot host the panel, qualification has single-device scope
(H100-only) and no transport is claimed. A transport claim needs the matched centred-paired device error SD ≤ 0.1 ×
the between-edit SD.
*Identity, pinned before scoring and checked per run.* Device: GPU model, driver, CUDA, torch. dtype bf16. Actual batch
size, composition and window order (registered order). The effective config hash (TF-v1 config with the 4B model
substituted, recomputed by `wsp.eval.batch.effective_config_hash`). Model repo and revision sha. Tokenizer revision. The
64-window identity, re-derived with the 4B tokenizer: per-window token-id hashes, equal to `text_groups.json` or else
pinned separately before scoring.
*Criteria (§6/A2 paired shared-base item-block).* δ = log 1.05 nats. For each edit's paired change against the shared
base, the upper 95% bound of the paired uncertainty half-width must be ≤ δ/2. The bound uses 4,000 joint
component/item-block draws, PCG64 seed 20261110, linear .025/.975 quantiles. Repeat discrepancy must be ≤ 1e-4 nats
(maximum absolute primary − repeat difference of the per-window mean NLL). The reliability lower bound must be ≥ 0.90
(between-entry signal over signal plus paired item, reference and repeat error). Constant or undefined panels fail.
*Fail-closed within the cap.* If the bounds are undefined or not met, the panel may expand once to draw indices 1, 5,
9, 13 and 17 inside the 2 GPU-h Lambda cap. If it still fails, or the cap would be exceeded, the diagnostic is
**unqualified** and every 4B result is fast-tier only. Stop and reprice before exceeding the cap. A failed or unqualified
panel never becomes qualified TF-v1 by relabelling. A3 source-change checks are additional, not a substitute. A fast
tier, if used, has its own pinned identity (batch, dtype, order) and label.

**FP5 — F5.4 best-of-N executable rules (edit 5; pins of the pre-G5/G8 adapter manifest, not G2 outcomes and not
permission to run G5 production rejection).**
*Draw seeds.* seed(`20261110:h5-best-of-n:`, `<episode id>:<source block id>:<draw index>`), draw index 0–31, using the
`control_seeds.json` derivation rule. `torch.Generator('cpu').manual_seed(seed)`, `randn` in fp32 over the block shape.
The draw order is the index order.
*Matched support and norm.* A source block is one tensor's editor proposal. Its support S is the set of coordinates
changed by the editor's realised bf16 increment. A draw is D = ‖D_editor‖_F (Z ⊙ S) / ‖Z ⊙ S‖_F, applied as
bf16(W + D) − W. After rounding, the realised effective norm must match the editor's realised norm to relative error
≤ 2e-2. The realised changed-coordinate count must be within [0.95, 1.05] of the editor's, inside S. Otherwise the
draw is invalid.
*Permutations and compatibility.* The architecture-compatible permutation arm moves whole proposed updates across
layers within groups of identical role string and shape. It uses the `scripts/a5_build_panels.py:derangement` of
seed(`20261110:h5-permutation:`, `<episode id>:<role>:<shape>`); singleton groups stay fixed and are disclosed. It is
reported separately from critic-feature permutations. Draws never leave their source block's tensor and support.
*Development winner.* The objective is the measured mean paired improvement on development (proposal) support relative
to the actual incumbent, under A5.7's feasibility/safety predicates. The winner of prefix N is the argmax over valid
draws with index < N; ties within 1e-12 relative go to the lowest index. A prefix without a valid draw yields no
candidate, and the incumbent is retained.
*Invalid draws.* Norm/support failures, nonfinite values and evaluation failures stay in the ledger with a reason and
count toward the 32. There is no hidden replacement or redraw.
*Equal opportunity and accounting.* All 32 draws are evaluated once on development support; prefixes are nested. Prefix
N is charged N development evaluations plus one held-out evaluation of its winner. The editor arm gets the same
development-evaluation count at each N. Real-evaluation counts and GPU-seconds are reported per prefix, failures included.
*Capability logging.* Beside the four Boolean flags, log actual use: Python calls, forward passes, hidden-state reads,
feedback reads, the episode feedback history (hash chain), code/decode identity hashes, and calls and costs.
*Acceptance slices.* Held-out support is split before G5 into disjoint acceptance slices, hash-sorted by
sha256(`20261110:h5-acceptance:` + group id), each used at most once in that order. A changed-byte rebuild consumes the
next unused slice. When the slices are exhausted, acceptance fails closed and the incumbent is retained. Reused
validation is adaptive development, never acceptance evidence.

**FP6 — F5.5 enumeration, chart seeds and exclusion power (edit 6).** (b) is the conjunction over exactly these 13 members,
each with its conditional and joint A5.4.2 upper bound < 0.05 (ratio 1 − A_numerator / A_denominator):

| Member | Numerator / denominator | Rows | Control eligibility |
|---|---|---|---|
| C_abl | cand / no-field | H1w 386 | intact ablation; A5.4.1 contract, A5.4.2 validity |
| C_rand | cand / random field | H1w 386 | A5.8.3 random-field destruction eligibility |
| C_perm | cand / permuted field | H1w 386 | A5.8.3 permuted-field destruction eligibility |
| C_chart_rand | chart / chart random frame | H1w 386 | chart random-frame eligibility (`control_seeds.json:chart_projectors`) |
| C_chart_perm | chart / chart permuted projector | H1w 386 | chart permuted-projector eligibility and bijection audits |
| K:R, L:S, K+F:F, K+F:K | as named | H1w 386 and LoRA-only (8 members) | schema-pin mask, frame and gauge audits |

"Chart" is the A5.8.2 secondary chart arm on the identical rows and folds. Its controls are now registered in
`control_seeds.json:chart_projectors`: the random frame uses `20261110:chart-random-frame:` + exact tensor name and
`block_projectors(out, in, 4, seed)`. The permuted projector uses a derangement of intact projector identities within
role-and-shape groups, from `20261110:chart-permuted-projector:` + `<role>:<shape>`. Eligibility is A5.8.3's coverage
≥ 0.90 and weighted |cos| ≤ 0.20 on the projected-edit blocks. The earlier "not registered here" status is superseded.
An invalid or ineligible member, or missing power, leaves (b) unresolved.
*Upper-bound exclusion power.* The frozen specification is `docs/prereg/a5_manifests/linestop_power_spec.json`. It
computes P(upper bound < 0.05) for both procedures at the registered sizes, at true C = 0 (small-effect alternative,
required ≥ 0.80 for every member and procedure) and at the boundary C = 0.05 (required ≤ 0.035). C = 0.025 and 0.15
are also reported. It uses the development D3 WikiText H1w out-of-fold comparator losses with the
`scripts/tf_v1_power.py` nonnegative multiplicative construction. The registered cell is s = 0.5, ρ = 0.5, with a
sensitivity grid, 4,000 panels × 4,000 draws, and `SeedSequence(20261110)`. The registered C = 0.15 superiority power
is retained and reported separately. The specification is **not run**: its code must be hashed and its report committed
before Part B fitting, otherwise (b) is unresolved. (a) and (c) keep their own validity rules and are not powered by it.
No scientific negative claim is contemplated, so the conjunction is a **bounded programme/resource retirement on examined
data**, not a universal or fresh falsifier. There is no .5 ladder substitution and no secondary funding fallback.

#### A5.8.1 amendment (10 Nov 2026, before audit): eigenvalue tolerance

*Finding.* The CPU dry run of the G2 driver (`scripts/g2_fields.py --toy`, commit `604129b`) ran on toy panels: a tiny
random Qwen3.5 and synthetic panels with the registered sizes. The A5.8.1 rule as implemented in
`src/wsp/features/response_field.py` (schema 1) symmetrised and decomposed each factor in fp32, required
max|F − Fᵀ| ≤ 1e-6 × trace/dim (plus 1e-6 relative), and failed the gate when the smallest eigenvalue was below
−1e-6 × trace/dim. That rule fails on rank-deficient K-FAC factors. MMLU has 64 rows against 1,024–6,144 factor
dimensions, and the refusal factors are also rank-deficient. fp32 rounding leaves their null-space eigenvalues near
−1e-7 × λ_max, which is far below −1e-6 × trace/dim because trace/dim is small when most eigenvalues are zero.
WikiText and HellaSwag passed. The failure is numerical; it is not a materially indefinite factor. The rule below was
decided after this dry run on toy panels. No real field was computed under either rule, and no G2 outcome exists.

*New rule (supersedes the "Precision" and "Negative-eigenvalue tolerance" rows of the A5.8.1 table; the F5 schema
pins' eigen conventions use this tolerance).*
(a) Each stored fp32 factor is cast to float64. The symmetry audit requires max|F − Fᵀ| ≤ 1e-6 × max|F|, relative to
the largest absolute entry. The factor is then symmetrised ½(F + Fᵀ) and decomposed with `torch.linalg.eigh`, all in
float64. Storage stays fp32: gradients, factors and the kept (damped, truncated) factor.
(b) The tolerance is relative to the largest eigenvalue λ_max. Eigenvalues in [−1e-6 × λ_max, 0) are clamped to 0 and
counted. An eigenvalue below −1e-6 × λ_max fails the gate as before, and so does any negative eigenvalue when
λ_max ≤ 0.
(c) The field artifact records the count of clamped eigenvalues and the most negative ratio λ_min/λ_max (0 if none).
These go in each factor's audit (`n_clamped`, `most_negative_ratio`, `lambda_max`) and in each factored tensor's row
of the coverage block (`eigen_clamps`). The G2 manifest records them per objective (`fields.<objective>.eigen_clamps`:
factors and eigenvalues clamped, the most negative ratio and its factor).
(d) Damping (1e-3 × trace/dim of the clamped spectrum) and the 99% energy truncation of the damped spectrum are
unchanged.
(e) The `FieldArtifact` schema version is 2, and schema-1 artifacts are refused on load.
On a well-conditioned factor, the float64 path reproduces the schema-1 kept factor and contractions to 1e-6 relative
(test). A 64 × 1,024 fp32 factor that schema 1 rejected now passes with its clamps recorded (test). A materially
negative factor still raises (test).
In the same toy dry run under the new rule, all five objectives pass. The toy MMLU field clamped 541 of 1,948
eigenvalues in 41 of its 62 factors, with most negative ratio −3.78e-7 (`model.layers.1.mlp.up_proj` A). These are toy
values, not G2 measurements.

*Re-pin.* `response_field.py` is re-pinned, and `code_hashes.json` now also pins the G2 driver and launcher and the
4B scripts and config. The new literal hashes and the old/new identities are in A5.1.2 (viii).

*F5.3 confirmations (chair, 10 Nov 2026, before any 4B scoring).*
(i) "Attention output projection" means each layer's attention writer: `self_attn.o_proj` on full-attention layers
and `linear_attn.out_proj` on GatedDeltaNet layers, as `wsp.zoo.arch` resolves the `attn_out` role alias. Each 4B edit
therefore has 12 tensors, `mlp.down_proj` plus the attention writer on each of layers [11, 16, 20, 23, 25, 28]
(`scripts/scale4b_build_edits.py`, default `--support role`).
(ii) The edit that is recorded and applied is the realised bf16 edit, bf16(W + stored) − W, whose radius is about
0.82e-3 for the intended 1e-3. Every edit records its realised radius, changed-element fraction and cosine to the
intended Gaussian, per tensor and pooled.
*Noted, not resolved here.* F5.3's text above states a planned radius ρ = 0.005, and the pinned builder's
`RADIUS = 1e-3`. The builder seeds edits as 20261110 + i, while F5.3 and FP4 state a SHA256-of-`20261110:4b-gaussian:`
rule. Its A3 pairing set is the first five by sha256 rank, while FP4 states draw indices 0, 4, 8, 12 and 16. These
differences are flagged for the chair and Board A before any 4B scoring.

#### A5 note (10 Nov 2026): 4B builder aligned to F5.3/FP4

*Ruling (chair, 10 Nov 2026, before any 4B scoring).* The registered text governs the code. The three discrepancies
noted at the end of "A5.8.1 amendment (10 Nov 2026, before audit)" are resolved in favour of F5.3 and FP4; the chair's
1e-3 radius default is withdrawn. No 4B edit has been built or scored, so no outcome informed this note.
*Changes* (`scripts/scale4b_build_edits.py`, kernel `gaussian_relative_frobenius_per_tensor_pcg64_bf16_effective_v2`):
(1) **Radius.** ρ = 0.005 of each touched tensor's Frobenius norm, so the pooled intended radius over the support is
also ρ. The realised bf16 edit is recorded as before: realised radius, changed-element fraction and cosine to the
intended Gaussian, per tensor and pooled. On a CPU toy tensor (not the 4B model), ρ = 0.005 gives realised radius
about 0.0052, changed fraction about 0.68 and cosine about 0.95.
(2) **Seeds.** Draw i (0–19) has seed int.from_bytes(sha256(`20261110:4b-gaussian:` + str(i))[:8], big-endian) mod
2^63, as F5.3 states. Draw 0's seed is 3916065655914066609; draw 2's unsigned value 14737898880838895577 reduces to
5514526843984119769. Per tensor, the seed is the same rule applied to `<draw seed>:<layer>:<module path>`, with the
namespace-free key from `wsp.zoo.arch.split_key`. Z is `numpy.random.Generator(PCG64(seed)).standard_normal` in float64
over the tensor shape. A tensor's draw depends only on the draw index and its own key.
(3) **Pairing.** The FP4 panel is draw indices 0, 4, 8, 12 and 16. The single expansion, 1, 5, 9, 13 and 17, is
recorded in the manifest header and used only under FP4's fail-closed rule. The sha256-rank pairing set is withdrawn.
*FP4 fields also aligned.* The two same-device repeats are labelled `q4b-r1` and `q4b-r2`. Each runs in a fresh
process with a fresh model load and its own immutable run id (`YYYYMMDD-HHMMSS-<8 hex>`); the earlier `r1` repeat is
withdrawn. On the H100 the panel (base + 5) and both repeats run before the remaining 15 edits. The Lambda cap
defaults to 2 GPU-h and cannot be set higher (it was 3.0). Each device writes `device_identity.json` (GPU model,
driver, CUDA, torch). `scripts/scale4b_report.py` checks, per run: the device id; dtype bf16; batch size; the effective
config hash recomputed by `wsp.eval.batch.effective_config_hash`; model repo and revision; window order; and per-window
token-id hashes equal to `text_groups.json` or to a separate pin given before scoring. The FP4 1e-4-nat repeat
discrepancy is added beside the bitwise A2.4/A3 check. The report does not compute FP4's half-width or reliability
bounds, and states that it alone never establishes qualification.
*Re-pin.* A5.1.2 (ix).

#### A5.1.2 scope reconciliation (10 Nov 2026, after audit)

Executes the two blocking repairs of Board A's post-commit freeze audit
([BA-FRZ-AUDIT](reviews/council/board-a-freeze-audit-2026-11-10.md), "Blocking repairs before another freeze
acceptance"), and nothing else. This block is appended; earlier text is not rewritten. It was written without GPU work,
without computing any field and without any G2 outcome; data were read only. It is not Board A's acceptance. Board A's
post-commit check of this commit is still required.

*Repair 1: registered multi-objective predecessor audit (A5.8.1, A5.5.1, F1(b)).* The driver pinned at A5.1.2 (viii)/(ix)
audited only `wikitext_nll` (`AUDIT_OBJECTIVE`). That does not meet A5.8.1's "for multiple objectives require that count
separately per objective", A5.5.1's validity row ("≥16/20 predecessor passes per objective") or F1(b)'s "any gate applied
per objective still applies". `scripts/g2_fields.py` now implements the registered rule, and the rule is unchanged:
(1) For each of the five field objectives (`wikitext_nll`, `mmlu_gold_log_prob`, `hellaswag_gold_char_log_likelihood`,
`refusal_margin_harmful`, `refusal_margin_harmless`), separately, the stage audits the same twenty identified
predecessors (chain picks' actual predecessors and their ancestors, sorted by canonical ID; unchanged selection). Each row
records predecessor, successor, pick and alias-holder identities; exact contraction; approximate = base contraction + cross
term, with both parts recorded; increment and displacement hashes; denominator max(|exact|, 1e-3 nats); and relative error.
(2) Each objective gets its own verdict: pass iff ≥16 of 20 rows have finite error ≤0.1 (inclusive) *and* twenty rows exist.
(3) A short audit is a hard stop whatever the pass count and whatever the overall `complete` flag. This covers fewer than
twenty identified predecessors, or fewer than twenty rows for any objective. A short selection hard-stops before any audit
compute is spent.
(4) Any failed required objective sets `HARD_STOP` in the manifest and gives exit code 2. A pass on one objective (for
example WikiText) never stands for another.
(5) Exact gradients are computed one objective at a time at each predecessor, through the existing gradient-only path, with
no graph retention. The model is restored to base after every row.
(6) Before the first row, the manifest records the cost estimate as 5 objectives × 20 predecessors = 100 rows. Each row is
estimated at 0.5 × that objective's measured field wall time, because the exact gradient is one of the field's two
forward/backward passes over the same items. The estimate is checked against the registered aggregate 1 GPU-h audit cap,
and re-checked before each later objective using the largest measured/estimated row-time ratio so far. If the estimate
exceeds the cap, the run stops as incomplete (`audit_gpu_hours_estimate`) and is repriced prospectively. The budget is
never multiplied.
(7) The stage resumes per objective. A completed objective's file is verified by hash, kept, and never re-run (no retry to a
pass). An interrupted objective restarts, and its earlier partial rows stay in the attempt history. Resuming with a
different audit implementation after a completed objective is refused.
Outputs: `predecessor_audit/<objective>.json` per objective, and `predecessor_audit.json`, which holds the overall verdict
and the per-objective file hashes. Manifest schema: `audit.objectives.<objective>`, `audit.required_objectives`,
`audit.pairs`, `audit.cost_estimate`, `audit.cost_checks` and `hard_stop.failed_objectives`/`short_objectives`, with
audit schema `g2-predecessor-audit-v2`. Regression tests in `tests/test_g2_fields.py` cover per-objective verdicts, the
short-audit hard stop, the any-objective-failure hard stop, the cost estimate, per-objective resume and the manifest schema.
*Toy CPU dry run* (`scripts/g2_fields.py --toy`: a tiny random Qwen3.5, synthetic panels and synthetic chain deltas; toy
numbers, not G2 measurements). The run ended `HARD_STOP` with exit code 2, and every objective failed its own count.
WikiText passed 5/20, MMLU 14/20, HellaSwag 6/20, harmful refusal 8/20 and harmless refusal 7/20. The toy cost estimate
was 0.0186 h for 100 rows. The measured toy audit took 0.0113 h, about 0.6 × the estimate. On these random toy deltas the
first-order base-plus-cross approximation is poor; this says nothing about the real fields.

*Repair 2: commits in `b7f10a1..a0f352b` (`git log --oneline`).* Independently motivated commits are kept separate from the
governed freeze/amendment diff. History is not rewritten, and the independent commits are not absorbed into any allowlist
wildcard. **This range does not contain only allowlisted repairs.** Board A is asked to approve this explicit, expanded
audit scope.

| Commit | Subject | Files | Class |
|---|---|---|---|
| `a0f352b` | reviews: Board A post-commit freeze audit | `docs/reviews/council/board-a-freeze-audit-2026-11-10.md` | independent: review report |
| `1965703` | progress: 4B alignment merged; freeze audit requested | `docs/PROGRESS.md` | independent: PROGRESS entry |
| `7ecd186` | merge prereg/a5-4b-align | contents of `5958498` (diff to first parent identical) | governed amendment merge (A5.1.2 (ix)) |
| `5958498` | 4B diagnostic builder aligned to F5.3/FP4; re-pin | `docs/PREREG_H1_TF.md`, package `README.md`/`code_hashes.json`, `scripts/scale4b_build_edits.py`, `scripts/scale4b_report.py`, `scripts/lambda_job_scale4b.sh`, `scripts/scale4b_h0_pair.sh`, `tests/test_scale4b.py` | governed amendment ("A5 note (10 Nov 2026): 4B builder aligned to F5.3/FP4") |
| `21292fe` | progress: eigenvalue amendment merged; 4B ruling; backup corrected | `docs/PROGRESS.md` | independent: PROGRESS entry |
| `43283ec` | merge prereg/a5-eig | contents of `b67cf34` (diff to first parent identical) | governed amendment merge (A5.1.2 (viii)) |
| `b67cf34` | A5.8.1 amendment: fp64 eigendecomposition and λ_max-relative tolerance; re-pin | `docs/PREREG_H1_TF.md`, package `README.md`/`code_hashes.json`, `scripts/a5_build_manifests.py`, `scripts/g2_fields.py`, `src/wsp/features/response_field.py`, `tests/test_g2_fields.py`, `tests/test_response_field.py` | governed amendment ("A5.8.1 amendment (10 Nov 2026, before audit)") |
| `4d51297` | D8 addendum 3 note: symlink fix and lifecycle rule | `docs/DECISIONS.md` | independent: DECISIONS addendum note |
| `32e8b70` | backup: do not follow symlinks | `scripts/backup_sync.sh` | independent: backup infrastructure |
| `48ad76b` | progress: freeze edits merged; eigenvalue amendment in progress | `docs/PROGRESS.md` | independent: PROGRESS entry |

*Verification.* The inventory has 98 paths: the 97 path rows of A5.1.2's tables plus `docs/PREREG_H1_TF.md`. For each
path, `git diff --stat b7f10a1..a0f352b -- <path>` was run. It is empty for 88 paths, including every raw-evidence path,
which is not in git. It is non-empty for exactly ten paths: `docs/PREREG_H1_TF.md`,
`docs/prereg/a5_manifests/README.md`, `docs/prereg/a5_manifests/code_hashes.json`, `scripts/a5_build_manifests.py`,
`scripts/g2_fields.py`, `src/wsp/features/response_field.py`, `scripts/scale4b_build_edits.py`, `scripts/scale4b_report.py`,
`scripts/lambda_job_scale4b.sh` and `scripts/scale4b_h0_pair.sh`. `git log b7f10a1..a0f352b -- <path>` attributes each of
those ten paths only to `b67cf34` and/or `5958498`. For each of the six independent commits (`48ad76b`, `32e8b70`,
`4d51297`, `21292fe`, `1965703`, `a0f352b`), `git diff --name-only <c>^ <c> -- <98 paths>` is empty. **The independent
commits touched no inventoried file.** In the range, the changed files outside the inventory are the three governed test
files below, which are not code-map members, and the independent files `docs/PROGRESS.md`, `docs/DECISIONS.md`,
`scripts/backup_sync.sh` and the audit report.

*Missing test-file identities (audit item 7) and this repair's files.* SHA256 of committed bytes. None of these tests is a
`code_hashes.json` member.

| Path | SHA256 | Identity |
|---|---|---|
| `tests/test_response_field.py` | `4dfb62eb3be29440b89963dd50dbda305714a59660274afdbdb01add771d02f4` | changed 2026-11-10 by `b67cf34` (A5.8.1 amendment: fp64/λ_max-tolerance regressions); old `9e805e1450a18f8dba0ab78dfaca3c4008126ae056883721675eace3daf41965` at `b7f10a1`; unchanged since |
| `tests/test_scale4b.py` | `3c4bcc153ba78b19aa2d03548b2b81760eba0f4fdb54155b2480dc1e0039714a` | changed 2026-11-10 by `5958498` (4B alignment to F5.3/FP4); old `a9087dd25399d2d26467c7a5b6467e4df693484a68e665115117c7b64ba52776` at `b7f10a1`; unchanged since |
| `tests/test_g2_fields.py` | `25724b715edcadcaa92f19cba03cf3c4e223125b1c39e02d2a460a3e81aed593` | changed 2026-11-10 by this repair (per-objective audit regressions); old `a35880b3cdfe665ec8ae558ebc6c2bbedf70f69bce051830e5305aa60fa08d8b` (`b67cf34`, A5.8.1 amendment), earlier `8c4c63c16ad71f3db5a55ee547da677ec90ba8334bea8c8955a8a8a5790cb5db` at `b7f10a1` |
| `scripts/g2_fields.py` | `16a2ee79ea86fb87ef138deac9839395915a69b854f19c43a3db3ae14a503726` | changed 2026-11-10 by this repair (registered per-objective predecessor audit); old `4856f9938633b94138fabc3a592962ea93b2e274045aa71cc30bedf64b5e52f9` (A5.8.1 amendment), earlier `c1366df579a02d060ec9081fe0e77e6ff51e3cb92826ffd6259add8769b37e39` at `b7f10a1` |

*Re-pin (x) (10 Nov 2026, after audit).* `scripts/a5_build_manifests.py --pin-worktree` was re-run against the closed
snapshot, into a copy of the package, from parent `a0f352b`. Of the regenerated files, `components.jsonl`,
`selection.json`, `references.json` and `text_groups.json` were byte for byte identical, as was every other package file in
the copy. Only `code_hashes.json` and `README.md` changed, and only those two were copied into the package.
`code_hashes.json` (schema `a5-code-hashes-v2`, sha256 `e1ac62fde2c1928ad660eb833df2884c100b11db5884d537df527fb2251f4075`,
`git_sha` = parent `a0f352b`) still pins 64 files. It differs from `ce07f3b5…c6f8af` only in `git_sha` and the
`scripts/g2_fields.py` pin. This is now the current code map. `ce07f3b5464457ff96356e58d57279de93cf5392a52bf31d7ce1c8068cc6f8af`,
`701ae646…130744` and `708c6fab…abfe2c9` remain historical identities. The *Identities* sentence "This freeze's code map
is `708c…`" names the initial freeze-pass identity. This commit changes only `docs/PREREG_H1_TF.md`, the two package files
below, `scripts/g2_fields.py` and `tests/test_g2_fields.py`. These rows supersede the corresponding current-hash entries of
A5.1.2's tables.

| Path | SHA256 | Identity |
|---|---|---|
| `docs/prereg/a5_manifests/README.md` | `3766416ce2e3e61465c72d6b42f84dc569188cde7bae79033b65e17c39002296` | changed 2026-11-10 (audit-repair re-pin); old `225f0caeff23bc7f5bf58115f63f4f08294729484979d8912c947e6662766120` (4B builder alignment re-pin) |
| `docs/prereg/a5_manifests/code_hashes.json` | `e1ac62fde2c1928ad660eb833df2884c100b11db5884d537df527fb2251f4075` | changed 2026-11-10 (audit-repair re-pin); old `ce07f3b5464457ff96356e58d57279de93cf5392a52bf31d7ce1c8068cc6f8af` (4B builder alignment re-pin) |
| `scripts/g2_fields.py` | `16a2ee79ea86fb87ef138deac9839395915a69b854f19c43a3db3ae14a503726` | changed 2026-11-10 (this repair); old `4856f9938633b94138fabc3a592962ea93b2e274045aa71cc30bedf64b5e52f9` |

#### A5.8.1 note (10 Nov 2026, after G2 fields): audit cost check uses the measured first row

This block is appended; earlier text is not rewritten. It was written without GPU work. G2 data were read only. It
repairs the audit stage's cost check under the stop-and-reprice rule of A5.8.1 (S31 §3) and nothing else. It is not
Board A's acceptance.

*The stopped run.* G2 attempt 1 (`data/runs/g2/g2_manifest.json`, 13:49:48Z-15:16:43Z, driver `scripts/g2_fields.py`
`16a2ee79…a503726` under code map `e1ac62fd…f4075`) ended `incomplete`. The `fields`, `projectors` and `probe` stages
are `complete`. All five fields passed their factor audits and have recorded file hashes. Field wall time was 525-1,649 s
per objective, and fields plus probe cache used 1.448 of the 2.0 GPU-h cap. The `audit` stage stopped before its first
row (`incomplete_reason`: `audit_gpu_hours_estimate` at `audit:wikitext_nll`), because the pre-audit estimate was
11.70 GPU-h against the registered 1.0 GPU-h audit cap. That estimate was 100 rows × 0.5 × the objective's field wall
time. **Zero audit rows were run.** The fields, projectors and probe cache are valid and are kept. The 0.5 × field-wall
row cost was a guess, not a measurement. Most of the field wall time is CPU K-FAC accumulation, which an exact-gradient
pass over the same items does not do.

*New rule (driver only).* (1) The field-based estimate is still computed and recorded (`audit.cost_estimate`), but it is
no longer the stop criterion. The cost check runs after the first registered row of each objective. That row is computed
exactly as registered and is row 1 of 20, not a separate pilot. The check is: audit hours used + the measured first-row
time × the objective's 19 remaining rows + each later objective at 20 × 0.5 × its field wall time × the largest
measured/estimated row-time ratio so far (this objective's included) must be ≤ the 1 GPU-h audit cap. Otherwise the run
stops `incomplete` (`audit_gpu_hours_estimate`). Each check is recorded in `audit.cost_checks`. (2) Before an objective's
first row, its one-row estimate (0.5 × field wall time × the largest measured/estimated ratio of objectives already
audited, 1.0 before any) must be ≤ 0.1 × the audit cap (360 s at 1 GPU-h). Otherwise the objective is not run in this
attempt. It is recorded as `cost_guard` and the run ends `incomplete` (`audit_cost_guard`); it is not dropped. The other
objectives continue, and a resume re-evaluates the guarded objective with the ratios measured by then. Before every later
row, audit hours used + the objective's mean measured row time × its remaining rows must be ≤ the cap. Otherwise the run
stops `incomplete` (`audit_gpu_hours_row_guard`) before that row. Rows already computed stay in the manifest, and in the
attempt history on resume; no row is dropped and the budget is never multiplied. (3) Every audit row records its wall
seconds, GPU seconds (wall time on the exclusive GPU) and `torch.cuda.max_memory_allocated`. Field, projector and probe
records now also carry GPU seconds and peak memory. A quantity a run did not record is reported as "not captured in this
run" and never estimated (`resource_summary`). In attempt 1, the field and probe records do contain peak memory and GPU
hours; the projector record has only wall seconds and bytes written. (4) `--resume --stages audit` first verifies the
file hashes of the completed fields, projectors and probe cache, and refuses on any mismatch. It never re-runs them, and
it then runs only the audit stage.

*Expected effect on the resume (from attempt 1's field wall times, not an outcome).* The uncalibrated one-row estimates
are 824 s for `wikitext_nll` and 368 s for `hellaswag_gold_char_log_likelihood`, both above 360 s; the other three
objectives are 263-345 s. `wikitext_nll` is first in the registered order, so with no measured ratio yet it is
cost-guarded on the first resume. `hellaswag_gold_char_log_likelihood` is evaluated after `mmlu_gold_log_prob` has been
measured. A second identical resume evaluates `wikitext_nll` with the measured ratios. Every audited row is still
checked against the same cap.

*Unchanged.* No registered quantity, threshold, panel or predecessor set changes. That includes: the five objectives; the
twenty identified predecessors and their selection rule; exact contraction versus base contraction plus cross term;
relative error ≤ 0.1 with the 1e-3-nat floor; ≥16 of 20 per objective; the short-audit and any-failure hard stops; the
1.0 GPU-h audit and 2.0 GPU-h fields-and-cache caps; the field configuration; and the 0.5 field-fraction constant used
by the estimate. Regression tests in `tests/test_g2_fields.py` cover the first-row-then-estimate order, the 0.1 × cap
guard, the incomplete stop after a measured overrun, peak memory and GPU seconds in field and row resources, the
not-captured reader, and an audit-only resume on hash-verified artefacts.

*Re-pin (xi) (10 Nov 2026, after G2 fields).* `scripts/a5_build_manifests.py --pin-worktree` was re-run into a copy of
the package from parent `31400a0`. Every regenerated or copied package file was byte for byte identical except
`code_hashes.json` and `README.md`, and only those two were copied into the package. `code_hashes.json` (schema
`a5-code-hashes-v2`, `git_sha` = parent `31400a0`) still pins 64 files. It differs from `e1ac62fd…f4075` only in
`git_sha` and the `scripts/g2_fields.py` pin. It is now the current code map, and `e1ac62fd…f4075` is a historical
identity (the code map of G2 attempt 1). These rows supersede the corresponding current-hash entries of A5.1.2's tables.
This commit changes only `docs/PREREG_H1_TF.md`, the two package files, `scripts/g2_fields.py` and
`tests/test_g2_fields.py`.

| Path | SHA256 | Identity |
|---|---|---|
| `docs/prereg/a5_manifests/README.md` | `5467fac25399a36e7828b610b3f7a381c30ede8007bdfcafa727af117b36a4a3` | changed 2026-11-10 (A5.8.1 note re-pin, after G2 fields); old `3766416ce2e3e61465c72d6b42f84dc569188cde7bae79033b65e17c39002296` (audit-repair re-pin) |
| `docs/prereg/a5_manifests/code_hashes.json` | `5be50667fd728332954a4b1add9640860a9495ce74bc34c70ac01c10a9e30c25` | changed 2026-11-10 (A5.8.1 note re-pin, after G2 fields); old `e1ac62fde2c1928ad660eb833df2884c100b11db5884d537df527fb2251f4075` (audit-repair re-pin; G2 attempt 1) |
| `scripts/g2_fields.py` | `6125b519381da53a6064da81bc108aedeb2bd7a1a9b5a2b6e9460dfb1c2100e9` | changed 2026-11-10 (this note); old `16a2ee79ea86fb87ef138deac9839395915a69b854f19c43a3db3ae14a503726` (audit repair; G2 attempt 1) |
| `tests/test_g2_fields.py` | `a0ba30033ed10f6dfb89cbf3d133ccb547fba183670dbbb0ef9fc6b644ede510` | changed 2026-11-10 (this note; not a code-map member); old `25724b715edcadcaa92f19cba03cf3c4e223125b1c39e02d2a460a3e81aed593` |
| `scripts/g2_fields.py` | `cb1835181ae0d8202edcaa72586e38dcc3a8822ce6127f104e5c123a86838723` | changed 2026-11-10 (ascending order and end-of-run guard re-check, paragraph below); old `6125b519381da53a6064da81bc108aedeb2bd7a1a9b5a2b6e9460dfb1c2100e9` (this note) |
| `tests/test_g2_fields.py` | `f7be4d4aeb83fe0741a7890005480376514aa2394234f9239894980397a8adaa` | changed 2026-11-10 (paragraph below; not a code-map member); old `a0ba30033ed10f6dfb89cbf3d133ccb547fba183670dbbb0ef9fc6b644ede510` (this note) |
| `docs/prereg/a5_manifests/code_hashes.json` | `d3b11bab664e0822ec63b4f059703d6418ff09847c49681fdd7f710617e6a5c7` | changed 2026-11-10 (re-pin (xii), paragraph below); old `5be50667fd728332954a4b1add9640860a9495ce74bc34c70ac01c10a9e30c25` (re-pin (xi)) |
| `docs/prereg/a5_manifests/README.md` | `0454be009511b5761991e7ef27bd351ee1f25f772ca0fcc7d2d01b19997549f5` | changed 2026-11-10 (re-pin (xii), paragraph below); old `5467fac25399a36e7828b610b3f7a381c30ede8007bdfcafa727af117b36a4a3` (re-pin (xi)) |

*Addendum (10 Nov 2026, chair decision on the cost guard).* This paragraph is appended; the text above is not rewritten. It
was written without GPU work, and G2 data were read only. It changes the driver only. (1) The audit stage now processes the
required objectives in ascending order of the field-based one-row estimate (0.5 × field wall time), with ties kept in
registered order. With attempt 1's field wall times the order is `refusal_margin_harmful` (262.5 s),
`mmlu_gold_log_prob` (306.7 s), `refusal_margin_harmless` (344.7 s), `hellaswag_gold_char_log_likelihood` (367.5 s),
`wikitext_nll` (824.3 s). The cheapest objective is therefore measured first, and its measured/estimated ratio calibrates
the guard and cost checks of the others. (2) An objective skipped by the 0.1 × cap guard is re-checked at the end of the
same run, after every other objective has been processed. The re-check uses the largest measured/estimated ratio of the
objectives audited by then, and the objective is audited at that point if it passes. If it still fails, it stays
`cost_guard` and the run ends `incomplete` (`audit_cost_guard`). This way a single resume can complete all five objectives.
The manifest records the processing order (`audit.order`, with each objective's field-based row estimate) and every guard
decision (`audit.guard_decisions`, phase `in order` or `end-of-run re-check`). The two rules are named in
`config.audit.objective_order` and `config.audit.guard_recheck`. Nothing else changes: row 1 then the measured check, the
per-row guard, the resource records, resume and the verdict rules are as above. The registered order assumed under
*Expected effect* above no longer applies. Tests in `tests/test_g2_fields.py` cover the ascending order, a guarded
objective that is re-checked and audited after calibration within one run, and a still-failing re-check that leaves the
objective `cost_guard` and the run incomplete. *Re-pin (xii).* `scripts/a5_build_manifests.py --pin-worktree` was re-run
into a copy of the package from parent `8f81ab0`. Only `code_hashes.json` and `README.md` differed, and only those two
were copied into the package. `code_hashes.json` still pins 64 files and differs from `5be50667…30c25` only in `git_sha`
and the `scripts/g2_fields.py` pin. The last four rows of the table above record this re-pin. This commit changes only
`docs/PREREG_H1_TF.md`, the two package files, `scripts/g2_fields.py` and `tests/test_g2_fields.py`.

#### A5 amendment F6 (11 Nov 2026, after the G2 audit measurement; before any G3 outcome)

This block is appended; earlier text is not rewritten. It records chair decisions F6.1–F6.6, taken after the G2
predecessor audit's one measured row and before any G3 outcome. It was written without GPU work, and G2 data were read
only. It starts no compute. It is not Board A's acceptance: Board A reviews F6 before any exact-gradient row runs (F6.6).

*Facts (read from the files named here).* G2 attempt 2 resumed the audit stage alone (`--resume --stages audit`,
15:46:07Z–16:05:59Z on 10 Nov 2026) under code map `d3b11bab…a6e5c7` (re-pin (xii)), after hash verification of the
completed fields, projectors and probe cache. Its manifest is `data/runs/g2/g2_manifest.json`, sha256
`636a967466cfcc3c728c6828f145f2474e04e658625701ab6f505ff00809c022`. The committed copy
`docs/prereg/a5_manifests/g2/g2_manifest_run1.json` has the same sha256, which is the hash listed in
`docs/prereg/a5_manifests/g2/README.md`. In the ascending order of the A5.8.1 note's addendum, the audit started with
`refusal_margin_harmful`, the cheapest objective (field-based one-row estimate 262.5 s, within the 360 s guard). It ran
exactly one registered row: pick `00f8111cbad8218f`, predecessor `072b2a3ba72e828c`, successor `34ee195e55227908`,
neither through an alias. Exact contraction −0.021217 nats, approximation −0.014085 (base contraction −0.014070 plus
cross term −1.45e−5), relative error **0.336**, above the 0.1 rule. The row took 394.0 s wall inside the row computation
(`wall_seconds`), and its resource record is 400.8 s wall and GPU seconds, including the restore of the model to base.
Peak allocated memory was 2.77 GB, and the audit used 0.17 GPU-h (`caps.audit.used_hours` 0.1715). The measured
first-row check then projected 17.92 GPU-h with the driver's formula (19 remaining rows at the measured time, and the
other four objectives at their field-based estimates × 1.5268) against the registered 1.0 GPU-h audit cap. The run stopped
`incomplete` (`audit_gpu_hours_estimate`, "audit:refusal_margin_harmful (after row 1)"). At the measured row time, the
registered 100 rows (5 objectives × 20 predecessors) need about 11 GPU-h (100 × 394–401 s = 10.9–11.1 h), eleven times
the cap. The row is recorded in the manifest (`audit.objectives.refusal_margin_harmful.rows[0]` and
`resources_summary.audit_rows`). **No per-objective audit file exists:** the driver writes
`data/runs/g2/predecessor_audit/<objective>.json` only when an objective completes, so
`data/runs/g2/predecessor_audit/refusal_margin_harmful.json` was never written. The G2 README's mention of "the audit
file" therefore has no separate file behind it; the manifest copy is the audit record. Fields, projectors and probe cache
are complete and valid (1.45 GPU-h of the 2.0 GPU-h fields-and-cache cap).

**F6.1 — the registered audit is recorded as run and infeasible at its cap.** The A5.8.1 audit (twenty predecessors per
objective, five objectives) ran under its registered rules and stopped at its registered 1.0 GPU-h cap after one measured
row (facts above). Its single measured row fails the rule (0.336 > 0.1). A5.8.1 makes the approximation conditional on
the audit ("otherwise the base approximation is not used for chains"). By that fallback the base-plus-cross-term
approximation g(W) ≈ g₀ + G E A is **not used for any chain row**. The audit is not resumed and not re-run. A5.5.1's
validity row ("≥16/20 predecessor passes per objective") is answered by this disposition and F6.2, which register the
exact-predecessor gradients and their cost as A5.5.1 step 1 provides. The approximation is not silently disabled: it
is withdrawn here, by amendment, for chains.

**F6.2 — exact predecessor gradients for the ten registered chain picks.** For the ten chain picks of `selection.json`,
the Part A field tokens are computed at the actual predecessor state by exact gradient for two objectives only:
`wikitext_nll` (the mechanism endpoint) and `mmlu_gold_log_prob` (descriptive). That is 10 predecessors × 2 objectives =
**20 exact-gradient rows**. The new registered cap is **2.2 GPU-h** (measured 394 s per row × 20 = 2.19 h) under stop
and reprice. K-FAC factors are not computed at predecessors: the q_t magnitude term of these rows is taken from the base
factors and flagged `base-factor`. The HellaSwag and both refusal field tokens of these ten picks are masked (presence 0).
The thirty LoRA picks are unaffected, because their predecessor is the base and their tokens are the base field's.
*Noted, not changed.* (i) Two of the ten picks, `014f8a9ec78db2be` and `0559051354691a70`, are root chain edits whose
actual predecessor is the base. For them the exact gradient at the predecessor is the base gradient, recomputed by the
same exact-gradient path. Their four rows are still among the twenty. (ii) Two predecessor states load through alias
holders, `bc6edb1e3c913a10` via `4b84f551adaf52ba` and `983bb1066d0fd006` via `345a7db356689c69`. The alias reuse
proofs remain the G3 staging blockers they were (F3(i)). (iii) The audit row's GPU-seconds record (400.8 s) × 20 is
2.23 h, just above the cap. The measured checks of the stage below decide whether the twenty rows fit; an overrun stops
the stage and is repriced, and the cap is not raised.

**F6.3 — Part A sign predictor.** Part A's registered sign predictor stays ⟨g_wikitext, D − R⟩. For chain picks, g is the
exact WikiText gradient at the actual predecessor (F6.2). For LoRA picks it is the base field's gradient, as before.
Resolution, intervals, yield and the decision function are those of A5.3 and A5.5.1.

**F6.4 — Part B.** Chain rows other than the ten picks keep their base-anchored field tokens, flagged
`approximation unvalidated`. A registered sensitivity arm repeats the primary contrast with those rows excluded. It is
reported next to the primary contrast and does not replace it.

**F6.5 — the audit's cost constant.** The audit's cost-estimate constant, a row at 0.5 × the objective's field wall time,
underestimated the measured row. The measured/estimated ratio is **1.53** (400.8 s measured / 262.5 s estimated; the
driver's recorded `calibration_ratio` is 1.5268). This is recorded. The constant is not used to size F6.2, whose cap
comes from the measured row.

**F6.6 — unchanged.** No threshold, panel, selection or decision rule changes. That includes the panels and their
hashes, the forty picks and their predecessors, the field configuration and base fields, the rotation, ladder and
resolution rules, the intervals, the A5.5.1 decision order and every funding threshold. Board A reviews F6 before any
exact-gradient row runs.

*Implementation (driver only; `scripts/g2_fields.py`, stage `chain_exact`).* The stage is requested explicitly. It is not
in the default stages. For each registered chain pick, sorted by edit id, it sets the model once to the actual
predecessor: base plus the predecessor's cumulative delta, loaded through the alias holder when needed, or the base for a
root pick. The selection's predecessor must equal the lineage parent in `components.jsonl`. With the model there, it
computes the exact WikiText and then MMLU gradient. It uses the audit's gradient-only path (`exact_gradient`: same scorer,
item order, fp32 CPU accumulation, division by the total weight; no Fisher pass, no K-FAC). The gradient is taken on the
tensors of the pick's realised increment D = stored(pick) − stored(predecessor). These are the support of D, of its
rotation R and of the ladder edits (A5.3); D_t = 0 elsewhere. Each streamed panel must reproduce the verified panel's
content hash. Before the first row, the loaded model's state hash must equal the base fields' recorded state hash. Each
row is written as a FieldArtifact-like directory `data/runs/g2/chain_exact/<pick_id>/<objective>/`, with
`manifest.json` (identity and content hashes) and `tensors.pt` (gradients, no factors). It is written through a
temporary directory and renamed, and it loads with `FieldArtifact.load`. Its identity records anchor `actual
predecessor`, q_t `base-factor`, the objective and panel identities, the pick, predecessor and alias identities, the base
state hash, the predecessor-state, increment and displacement hashes, the support and the code hashes. Each manifest row
also records file hashes, the exact contraction ⟨g, D⟩, the gradient norm, wall and GPU seconds, state-load seconds and
peak CUDA memory. *Cap:* `--max-chain-exact-gpu-hours`, default 2.2; a real run refuses any higher value. After the
first row of each objective, a measured check projects used hours + each objective's remaining rows × its measured mean
row time. An objective not yet measured uses its field-based row estimate × the largest measured/field-based ratio. The
check must stay ≤ the cap, or the run stops `incomplete` (`chain_exact_gpu_hours_estimate`). Before every later row, the
same projection is the per-row guard (`chain_exact_gpu_hours_row_guard`). The audit's 0.1 × cap pre-objective guard is
not part of this stage. Computed rows stay on record and the budget is never multiplied. *Resume:* `--resume --stages
chain_exact` keeps completed rows, verified by hash and never re-run. A row directory left by an interrupted attempt
without a manifest row is moved aside (`<objective>.stale-attempt<N>`), never deleted, and recomputed. A row whose
gradient fails is recorded with its error, masked (presence 0) and not retried. A completed stage is verified and not
re-run. *Manifest:* block `chain_exact` (schema `g2-chain-exact-v1`: status, objectives, masked objectives, picks and
`picks_record`, per-pick `masks` for all five objectives, `policy`, `order`, `rows`, `cost_basis`, `cost_checks`,
`stale_dirs`, resources and the summary file `chain_exact/chain_exact.json` with its sha256), `caps.chain_exact` and
`config.chain_exact`. The first chain_exact run writes `audit_disposition` (F6.1): the audit's status, rows on record,
relative errors, the stop record and `approximation_used_for_chains: false`. From then on, a complete G2 needs fields,
projectors, probe cache and chain_exact, and the audit stage is refused. A real run refuses the audit stage in any case.
Regression tests in `tests/test_g2_fields.py`, on the toy (three chain picks with predecessors and one root pick, 8
rows), cover: the stage's rows, artefacts, identities and masks for the configured picks; equality of the stage's
gradient with `collect_field` at the predecessor; pick resolution, including the registered selection (10 picks, 2 with
base predecessors, 2 via alias); the stop after the measured first-row check; the per-row guard stop; resume without
re-running completed rows, with stale directories kept aside and a tampered completed row refused; a failed row masked and
not retried; F6 completion semantics and the audit refusal; the cost projection; and the CLI cap guard.
*Two accounting repairs in the same driver.* (1) The fields-and-cache budget clock ran for the whole attempt. G2 attempt
2 ran only the audit stage, yet its `caps.fields.used_hours` reads 1.620 h: 1.448 h of fields and probe cache plus the
0.172 h audit stage. Only attempts that run the fields or probe stage now record that budget, so chain_exact time is
charged to `caps.chain_exact` alone. The 1.620 h on record is not rewritten; the 1.448 h of fields and probe cache is the
measured quantity. (2) On resume, the attempt-history entry shared the `caps` object with the new attempt. Attempt 1's
history entry in the attempt-2 manifest therefore shows attempt 2's caps (`audit.used_hours` 0.1715, although attempt 1
ran no audit row). History entries are now copies. Both repairs have tests (`caps.fields` unchanged by a chain_exact-only
attempt; attempt 1's chain_exact hours kept in its history entry).

*A5.1.2 re-pin (xiii) (11 Nov 2026, F6).* `scripts/a5_build_manifests.py --pin-worktree` was re-run into a copy of the
package from parent `10ac9c5`. Every regenerated or copied package file was byte for byte identical except
`code_hashes.json` and `README.md`, and only those two were copied into the package. `code_hashes.json` (schema
`a5-code-hashes-v2`, `git_sha` = parent `10ac9c5`) still pins 64 files. It differs from `d3b11bab…a6e5c7` only in
`git_sha` and the `scripts/g2_fields.py` pin. It is now the current code map, and `d3b11bab…a6e5c7` is a historical
identity (the code map of G2 attempt 2). These rows supersede the corresponding current-hash entries of A5.1.2's tables.
This commit changes only `docs/PREREG_H1_TF.md`, the two package files, `scripts/g2_fields.py` and
`tests/test_g2_fields.py`.

| Path | SHA256 | Identity |
|---|---|---|
| `docs/prereg/a5_manifests/README.md` | `2fab2a1cd6c95301812f8760d36dc02f9cfef24a6eab50f93b35b16c3d013629` | changed 2026-11-11 (F6 re-pin (xiii)); old `0454be009511b5761991e7ef27bd351ee1f25f772ca0fcc7d2d01b19997549f5` (re-pin (xii)) |
| `docs/prereg/a5_manifests/code_hashes.json` | `6dd2615f9e8ea12544716f1cc585adf902c5f751fa1ed6fee0eb4494cd28d346` | changed 2026-11-11 (F6 re-pin (xiii)); old `d3b11bab664e0822ec63b4f059703d6418ff09847c49681fdd7f710617e6a5c7` (re-pin (xii); G2 attempt 2) |
| `scripts/g2_fields.py` | `6f14a2ac4c0dd9b2dc6b2ba2e022ec6cbe98a75e840dd414804d1febf3fb9f93` | changed 2026-11-11 (F6: stage `chain_exact`); old `cb1835181ae0d8202edcaa72586e38dcc3a8822ce6127f104e5c123a86838723` (re-pin (xii); G2 attempt 2) |
| `tests/test_g2_fields.py` | `67be8c51375fef4a3536fc1b4ed54843bbf83ab222c49a43941ff0966949a42d` | changed 2026-11-11 (F6; not a code-map member); old `f7be4d4aeb83fe0741a7890005480376514aa2394234f9239894980397a8adaa` |

*A5.1.2 re-pin (xiv) (Part A staging; run 2026-11-10 18:46–23:45 UTC, after Board A's G2 acceptance).* Board A's
G2 verdict (`docs/reviews/council/board-a-g2-2026-11-11.md`, item 5) authorises Part A preparation, not G3 scoring.
`wsp.eval.response_intervention prepare` gained `--selection` (stage the registered picks of `selection.json`,
checked against `registered.edit_ids` and the procedural-freeze id hash, never re-selected), `--references` (load an
alias-only predecessor from its bytes holder; the scoring plan checks the holder's bytes) and `--text-groups` (the
primary-record and 64-window checks of the F3(i) reuse proof). It also writes a per-tensor F3(ii) audit file, and
`score --manifest` now also scores the missing references. These changes were committed with tests in `7f8b453`,
and the staging ran from that clean commit. `scripts/a5_build_manifests.py --pin-worktree` was then re-run into a copy
of the package from `7f8b453`. Every regenerated or copied package file was byte for byte identical except
`code_hashes.json` and `README.md`, and only those two were copied into the package. `code_hashes.json` (schema
`a5-code-hashes-v2`, `git_sha` = `7f8b453`) still pins 64 files and differs from `6dd2615f…cd28d346` only in `git_sha`
and the `src/wsp/eval/response_intervention.py` pin. No rule, threshold, seed, selection or panel changes. The staging
outputs are `docs/prereg/a5_manifests/part_a/`; its README records what was staged and the audit verdicts. Under the
registered A5.3.3 post-rounding audit, 39 of the 40 rotation pairs are excluded (all 30 lora_sft and 9 of 10 chain;
reasons are per tensor). 81 variants are staged (1 rotation, 80 ladder). n_D = 19 and n_P = 0, as in
`references.json`, and the priced count is 100 entries, 104.00 min. No variant or reference was scored. These rows
supersede the corresponding current-hash entries of A5.1.2's tables.

| Path | SHA256 | Identity |
|---|---|---|
| `docs/prereg/a5_manifests/README.md` | `5c770dbe4d8549224c8202192aba3d2eb3e9f22426b8dcfd8b902f795a1194b7` | changed 2026-11-10 (re-pin (xiv)); old `2fab2a1cd6c95301812f8760d36dc02f9cfef24a6eab50f93b35b16c3d013629` (re-pin (xiii)) |
| `docs/prereg/a5_manifests/code_hashes.json` | `d4b2e0ab32dde329e8d8f15d53709eca741476b51c996096ab863fa93d1a637b` | changed 2026-11-10 (re-pin (xiv)); old `6dd2615f9e8ea12544716f1cc585adf902c5f751fa1ed6fee0eb4494cd28d346` (re-pin (xiii)) |
| `src/wsp/eval/response_intervention.py` | `b511d3d1ccb2c518317b59eafa4eea034df41ee90b85ab53709ef008736e353a` | changed 2026-11-10 (Part A staging options, commit `7f8b453`); old `7cf453f263b92b3dc056109e06766b7dbd3dc606ad463f11aa4ccdec1b73f903` |
| `tests/test_response_intervention.py` | `10fa97eeb7693fe10abc89dcb77aec2c4f96d798f71e657f562acdbcb65768e5` | changed 2026-11-10 (commit `7f8b453`; not a code-map member); old `675f710a4c1696f636b407f59b5b84f7f4e4144c7c36b476ae4df937224b37c8` |
| `docs/prereg/a5_manifests/part_a/manifest.json` | `d7785b35de1d7041edd2db139591562379db12e33c2532921c814735e8dde8d6` | new 2026-11-10 (Part A launch manifest, A5.11; schema `a5-part-a-launch-v1`) |
| `docs/prereg/a5_manifests/part_a/rotation_audits.jsonl` | `f05fec4265ffd93c8c537100993d23262be95141369240fcad0a7e16ff474d35` | new 2026-11-10 (F3(ii) per-tensor rotation audits, 2,534 rows) |
| `docs/prereg/a5_manifests/part_a/README.md` | `2fcbf517ccf26f2a1fe325ef31300d291bc0f254c7faaacd18154e327bceada1` | new 2026-11-10 (Part A staging record; no outcome) |
