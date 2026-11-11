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

#### A5 amendment F7 (11 Nov 2026, before any G3 outcome): power/yield donor pool

This block is appended; earlier text is not rewritten. It records chair ruling F7, taken after the power fields were built
(`docs/prereg/a5_manifests/power/fields_manifest.json`, sha256 `19e9de84…5e9a6b`; `data/runs/a5/power/fields.json`,
sha256 `85843ed2…cf3b9`) and before any G3 outcome. No G3, rotation, ladder or sealed outcome was read, and no GPU was used.

*Facts.* `fields.json` holds registered sign predictions for all 141 NULL-SIZE donors. Ten (the registered chain picks) use
the F6.2 `chain_exact` WikiText gradient at the actual predecessor. Fifty-one are root donors (predecessor = base: 25 LoRA,
26 chain) and use the G2 base WikiText gradient, which is the exact gradient at their predecessor. The other 80 are chain
donors whose predecessor is not the base. They use the base field and are flagged `base-approximation-unvalidated`. Board A's
F6 acceptance ([F6 review](reviews/council/board-a-f6-2026-11-11.md), condition 5) requires POWER-AND-YIELD to use accepted
exact WikiText chain predictions and does not accept base gradients for nonroot chain donors.

**F7.1 — registered pool.** The registered POWER-AND-YIELD run draws its donors only from the donors whose WikiText gradient
is exact: the 51 roots and the 10 picks, **61 donors** (LoRA 25; chain 26 roots + 10 picks = 36). The list is
`docs/prereg/a5_manifests/power/registered_donors.json` (sha256 `6e77bca615624a04cb43b3d11a035b5dc31e5a2f1cd918d42a35cc7786babbb5`; sorted-id sha256 `52366da3…07f1`). It is
derived from the manifest flags alone: no flag, and either `gradient_source` `chain_exact` (a registered pick) or
`base_field` with `predecessor_is_base`. The 80 excluded donors are exactly those flagged `base-approximation-unvalidated`.
Donor cells are rebuilt from the listed rows with the frozen cutpoints and the same one-row-per-component rule
(`donor_cells`). Per (family, tertile) cell, donor rows / donor components (full-pool components in brackets):

| Cell | LoRA:0 | LoRA:1 | LoRA:2 | chain:0 | chain:1 | chain:2 |
|---|---|---|---|---|---|---|
| Rows / components | 1 / 1 (1) | 9 / 9 (9) | 15 / 15 (15) | 12 / 11 (26) | 9 / 8 (20) | 15 / 14 (25) |

No cell is empty; an empty cell would have stopped the run for a further ruling. The LoRA cells are unchanged. Three chain
components contribute two exact rows to the same cell, so the 36 chain rows are 33 component representatives.

**F7.2 — unchanged.** The NULL-SIZE payload (`data/runs/a5/nullsize/inputs.json`, sha256 `67da223c…1c53de`), its masks,
audits and evidence, the selection, quotas and frozen cutpoints, the seeds and stream order (`SeedSequence(20261107)`,
spawn 3: panels, signs, resolution), 10,000 replicates and 4,000 resolution draws, the registered `.75` alternative, the
resolution and decision rules, and the ≥ 0.80 ordered-power gate are unchanged. The restriction applies to POWER-AND-YIELD
only: the NULL-SIZE pass inside the same run uses the full payload, as registered, and reproduces the registered NULL-SIZE
run exactly.

**F7.3 — sensitivity.** A separately labelled SENSITIVITY run uses all 141 donors with the flagged predictions. It is
descriptive. The registered decision (ordered simulated power ≥ 0.80 at the registered alternative) is read on the
registered run only.

*Implementation.* `scripts/a5_simulation.py --donor-ids FILE` (a JSON list, or an object with a `donor_ids` list; requires
`--fields`). `restricted_donor_cells` checks that every id is a unique, eligible payload donor with a field prediction,
counts rows per cell, stops on an empty cell (naming it), and rebuilds the cells. The output records `donor_pool` (count,
sorted-id sha256, per-cell rows/components/full-pool components, file path) and adds the file to `input_provenance`.
Without `--donor-ids` the output is byte-for-byte what it was. Tests in `tests/test_a5_simulation.py` cover the restriction
(power traces draw only listed donors; NULL-SIZE identical to the full-pool run), the cell-count report, the empty-cell
error, id validation and the CLI.

*Results (report: `docs/prereg/a5_manifests/power/report.json` and `README.md`).* Both runs used the code bytes pinned below (`scripts/a5_simulation.py` `04f283d8…c167f`), 10,000 replicates and 4,000 resolution draws (exact two-sided 95% CP intervals). **Registered** (61 donors; raw `data/runs/a5/power/power_registered.json`, sha256 `2b4806e56d405bdd87c233ed3b247bb0c7a2fe4393083ddcad144e3085424522`): ordered power (go) 8540/10,000 = **0.8540** [0.8469, 0.8609]; alternative 0/10,000 = 0.0000 [0.0000, 0.0004]; unresolved 1460/10,000 = 0.1460 [0.1391, 0.1531]; yield (≥ 15 resolved) 1.0000 in each contrast; two-sided power 0.8540 / 0.8525 / 0.8535. **Ordered power 0.8540 ≥ 0.80: Part A is decisive (A5.3.2).** **Sensitivity**, descriptive (141 donors; raw `data/runs/a5/power/power_all.json`, sha256 `aa0af06458cafabbae5dd9364f54f945e1a0f303b10ce3667fafea7be35a4009`): ordered power (go) 8501/10,000 = **0.8501** [0.8430, 0.8570]; alternative 0/10,000 = 0.0000 [0.0000, 0.0004]; unresolved 1499/10,000 = 0.1499 [0.1430, 0.1570]; yield (≥ 15 resolved) 1.0000 in each contrast; two-sided power 0.8501 / 0.8483 / 0.8490. In both outputs the NULL-SIZE pass equals the registered NULL-SIZE run exactly (go 180, alternative 1939, unresolved 7881 of 10,000). Yield 1.0 rests on the registered treatment of unavailable audit evidence as a pass (F2); the proxy limits of F2 and F5.7 apply. The raw outputs are not committed; the report holds their hashes.

*A5.1.2 re-pin (xiv) (11 Nov 2026, F7).* `scripts/a5_build_manifests.py --pin-worktree` was re-run into a copy of the
package from parent `c9d1873`. Every regenerated or copied package file was byte for byte identical except `code_hashes.json` and `README.md`, and only those two were copied into the package. `code_hashes.json` (schema `a5-code-hashes-v2`, `git_sha` = parent `c9d1873`) still pins 64 files. It differs from `6dd2615f…d28d346` only in `git_sha` and the `scripts/a5_simulation.py` and `tests/test_a5_simulation.py` pins. It is now the current code map, and `6dd2615f…d28d346` is a historical identity (the F6 code map). These rows supersede the corresponding current-hash entries of A5.1.2's tables. This commit changes only `docs/PREREG_H1_TF.md`, the two package files, `scripts/a5_simulation.py`, `tests/test_a5_simulation.py` and the new `docs/prereg/a5_manifests/power/` files `registered_donors.json`, `report.json` and `README.md`. `--verify-pins HEAD` passes after the commit.

| Path | SHA256 | Identity |
|---|---|---|
| `docs/prereg/a5_manifests/README.md` | `f51a290a322dd22fba129a064ce3fcd0d9364ef4c386319e3cd2c005d2c04ea4` | changed 2026-11-11 (F7 re-pin (xiv)); old `2fab2a1cd6c95301812f8760d36dc02f9cfef24a6eab50f93b35b16c3d013629` (F6 re-pin (xiii)) |
| `docs/prereg/a5_manifests/code_hashes.json` | `90a28ef7f43580a33af0d2c5ea11d8168c07e7baf5a5e73918f4d40c66b91ba0` | changed 2026-11-11 (F7 re-pin (xiv)); old `6dd2615f9e8ea12544716f1cc585adf902c5f751fa1ed6fee0eb4494cd28d346` (F6 re-pin (xiii)) |
| `scripts/a5_simulation.py` | `04f283d8756bc2d81e2e8bd700163f48d5ceff8487041a8969db48beda8c167f` | changed 2026-11-11 (F7: `--donor-ids`); old `3b86668657273b188abdaa2e5d19eec832abe0fdb21f19b002fc5c0504da8fe6` |
| `tests/test_a5_simulation.py` | `c69a912b7deb1767918452688d862bd076a6933afbe4020b314da27472ed4096` | changed 2026-11-11 (F7); old `721a92c21adaf39d0b5c317ee5ba8e6c60a2b4f63ba1b837fc3f21272cf65072` |
| `docs/prereg/a5_manifests/power/registered_donors.json` | `6e77bca615624a04cb43b3d11a035b5dc31e5a2f1cd918d42a35cc7786babbb5` | new 2026-11-11 (F7.1; not a code-map member) |
| `docs/prereg/a5_manifests/power/report.json` | `ff46de30fa8e4ff97a4034e8bdef78cb19bc49662073523500a43c773b55845a` | new 2026-11-11 (F7 results; not a code-map member) |

#### A5 amendment F8 (11 Nov 2026, before any G3 outcome): descriptive external-edit and family-transfer panels; H5 scripted arm

This block is appended; earlier text is not rewritten. It records the chair decision of council round 4
(`docs/council/2026-11-11/SYNTHESIS.md`, member reports in the same directory) on the weight-space editing papers and replications
that appeared after round 3. It was written while Part A staging was at 20 of 40 edits and before any G3, rotation, ladder or sealed
outcome had been read. It adds nothing to, and removes nothing from, the registered arms (A5.4), the Part A selection (A5.1.2),
the decision rules (A5.5), the line-stop rule (F5.5) or any seed, cap or pin. Board A reviews F8 before any F8 entry is scored;
Part A scoring does not wait on F8.

**F8.1 — external-edit panel (F8-E), descriptive.** A panel of weight edits of the pinned base made by other groups and selected on
their endpoints, never on ours. Entries: (i) the released Astra two-tensor patch, replayed exactly as one base-root zoo entry holding
both dense deltas, hash-pinned against the pinned base revision; the entry fails closed if any tensor outside the claimed two
differs from the pinned base; (ii) published linear-programme edits only if their bytes or a complete recipe (objective, constraints,
inputs) are public; (iii) the aiXiv difference-of-means recipe re-estimated on frozen development inputs, labelled a
reconstruction. The arXiv 2611.09382 column edit enters only if its direction u is released. Source clusters are declared before
scoring; the loop-removal family (Astra, aiXiv, LP) is one component. Entries whose bytes are unavailable are listed as missing,
never substituted.

**F8.2 — family-transfer panel (F8-F), descriptive.** Edits of the pinned base in families the zoo lacks, constructed from
development-only inputs with no endpoint selection: a Wanda-initialised deletion mask on gate projections at < 0.05 % of entries;
a count-matched random gate deletion; a training-free task-vector rank-one edit (arXiv 2609.09054) on o_proj; a rank-one
down-projection edit from residual-activation contrasts on a non-scored split (the Concept Country construction). Each entry
carries the registered rotation (A5.3) and the ladder s ∈ {0.25, 0.5, 1}. SWEET-style suppressions (arXiv 2611.10974) are deferred
until their counterfactual generation is priced; learned deletion masks (arXiv 2611.10866) are excluded as an unpriced purchase.
Construction is capped at 0.5 GPU-h under the stop-and-reprice rule.

**F8.3 — frozen predictions.** Before any F8 entry is scored, a dated manifest records for every entry: the realised bf16 delta
hashes and the A5.3.3 effective-rank audit; the field's first-order term a = ⟨g, D⟩ and its sign at each s; sign⟨g, D − R⟩ for the
rotation R; the curvature term q and κ = q/|a|; the descriptive quadratic-sign forecast (sign flip at s* = |a|/q); the chart
prediction; abstentions; and a minimum surviving-entry fraction after bf16 rounding (an entry whose realised change is zero counts as
a disagreement, A5.3). The field and chart used are the frozen G2 artefacts; nothing is refitted.

**F8.4 — CPU pre-check gate.** TF-v1 scoring is bought only if at least four components have |a| + q above the floor, defined as the
median paired-interval half-width of the registered null-size payload (`docs/prereg/a5_manifests/nullsize/`). Otherwise the panel is
reported unscored with its predictions.

**F8.5 — endpoints and order.** Scoring on H0 TF-v1 with the common base reference under `configs/eval_tf_v1_wikitext.yaml`;
WikiText decides for the panel, MMLU is descriptive (its gold-letter log-prob endpoint is unqualified). F8 entries are scored only
after the Part A ledger is sealed, in the order Part A → F8-E → F8-F. Every entry is reported, including unresolved ones; agreement
is summarised per independent source cluster with exact binomial and Clopper–Pearson intervals, and the report states that a
panel of this size cannot reject chance agreement.

**F8.6 — no decision authority.** The F8 panels cannot satisfy, fail or unresolve any F5.5 predicate or any A5.5 step, and fund
nothing. A resolved sign hit is a transfer precondition for the H5 critic; a resolved miss marks a locality or K-FAC failure for
that family; an unresolved entry says nothing. None speaks to the published generation-side effects, which teacher-forced
endpoints do not measure.

**F8.7 — H5 scripted-estimator arm and second editor.** The H5 pilot adds a no-LLM arm: a difference-of-means direction with
linear-programme column and scale selection, given exactly the localisation the LLM editor receives, plus a variant that chooses
layers by field ranking. All information supplied to any arm (layers, budget, scale) is logged as prior information, not
discovery. GPT-6.1 Sol (medium reasoning) is registered as the practical editor beside GPT-6 Astra, which is retained for
reproduction; versions are pinned; no silent substitution; voluntary early stops are logged. The ≥ 30 % paired evaluation-savings
criterion (A5.7, F5.4) is unchanged and is reported per proposal stream (LLM, scripted, random best-of-N); an editor-advantage
claim must beat both baselines at matched information with charged cost.

**F8.8 — access-ladder prior.** The access-ladder report (chart-only, field-only, hybrid, activation-assisted) states in one
sentence that on this base a frozen domain-recovery study found activation and output features more informative than spectral
weight summaries (arXiv 2611.07610), that this is a prior and not an effect-size estimate, and that the field is weights plus
frozen receiver sensitivity rather than a data-free readout.

**F8.9 — brief corrections recorded.** arXiv 2611.09382 names "long, unresolved reasoning" rather than loops and derives its
direction from token embeddings, not activations; the TFTV edit is rank one and norm-matched; arXiv 2611.10866 tests no Qwen3.5
backbone; the aiXiv 93 % recovery is conditional on supplied layers, budget and scale.

**F8.10 — post-G3 option, not registered.** A compression-recovery truncation test on non-pick LoRA rows (k ∈ {1, r/2},
≤ 0.5 GPU-h) may be registered by a later amendment after the G3 ledger closes.

#### A5 amendment F9 (11 Nov 2026, before any G3 outcome): forecast log, pruning ladder, H5 localisation and test-time-LoRA shadow, generation endpoint

This block is appended; earlier text is not rewritten. It records the chair decision of council round 5
(`docs/council/2026-11-11/SYNTHESIS-r5.md`; brief `BRIEF-r5.md`; member reports `round5-*.md`) on five threads the user raised:
edits that improve mathematics, models that edit their own weights while working, models reporting which parts they use, fewer
parameters at equal performance, and predicting all of these. It was written while Part A staging stood at 27 of 40 edits and before
any G3, rotation, ladder or sealed outcome had been read. It adds nothing to, and removes nothing from, the registered arms (A5.4),
the Part A selection (A5.1.2), the decision rules (A5.5), the line-stop rule (F5.5), the H5 success criterion (A5.7) or any seed,
cap or pin. Everything below is descriptive, runs after the Part A ledger is sealed and after the F8 panels, and is governed by the
F8a freeze, gate, cluster and scoring-configuration rules once F8a is adopted. Board A reviews F9 with F8a; Part A does not wait.

**F9.1 — forecast-before-modify log (F9-L).** For every edit the programme constructs or receives outside Part A (F8 entries, F9
panels, every H5 proposal including rejected ones, test-time adapters), one row is appended to a JSONL log **before** the entry is
scored or deployed, and each batch's SHA-256 is committed before its first scoring job. The row holds: incumbent hash; realised bf16
delta hash and recipe hash; anchor type (`base-exact`, `incumbent-exact`, `base-factor`); selection status (`outcome-blind`,
`field-selected`, `externally-selected`); per-tensor and summed a = ⟨g, D⟩ and q = ½tr(GDADᵀ) from the frozen G2 artefacts; κ =
q/|a| (recorded as undefined at a = 0, never regularised); s* = |a|/q when a < 0 < q; sign(a) at each registered scale; the
quadratic-sign forecast; top-k tensor shares; the chart output or an explicit abstention; G2 artefact hashes; timestamp; and the
decision taken. Rows for failed constructions carry no delta and are kept.

**F9.2 — calibration statistics and claim thresholds.** Per family and endpoint, clustered by independent source component:
primary E₀ = 1 − Σ(Δy − s·a)² / ΣΔy² (a is in nats, so κ is fixed at 1) with its bias² / conditional-MSE split; secondary: E₀ for
s·a + s²·q (descriptive, q is a Gauss–Newton magnitude), resolved-sign agreement under the A5.3 rule with exact Clopper–Pearson
intervals over clusters, the calibration intercept and slope of Δy on a, forecast coverage (fraction of non-abstaining forecasts) and
the predicted-resolvable × resolved 2×2 on the F8a floor. Component bootstrap, 4,000 draws, SeedSequence 20261107. A fitted κ̂ is
applied, frozen, only to the next family. Below 30 independent clusters per family the log is reported as entries only; at 30–99 any
statement is labelled exploratory; a precision statement requires ≥ 100. Only the A5.8.3 readout (intercept, Σa, Σq, ridge) and the
abstention quantiles may learn from scored log rows, between prospectively separated episodes, with a recalibration accepted only if
held-out loss falls and conditional MSE does not rise. No F9 statistic has A5.5 or F5.5 authority.

**F9.3 — sparsity ladder, native SparseGPT family and ranking panel (F9-P).** (a) Ladder: unstructured deletion of the pinned base at
10, 20 and 50 % of the supported linear-weight inventory, from three mask sources: Wanda scores built on CPU from the G2 WikiText
input factors on development windows (after the A5.8.1 99 %-energy truncation is checked), a SparseGPT-derived mask transplanted onto
the original weights (pure deletion), and a count-matched random mask; each entry carries its A5.3 rotation: 18 entries. (b) Three
native SparseGPT entries with compensation, D = −W⊙M + C, as a separately labelled family whose contraction uses the complete D.
(c) Ranking panel: 24 outcome-blind Wanda masks at 20 % global sparsity with per-layer ratios drawn from a registered seeded Dirichlet;
statistic: Kendall τ of realised WikiText change against a + q, against q_A (G = I), and against the magnitude ⟨D, W⟩ = −‖W⊙M‖²,
with zero fitted parameters. Comparisons are within sparsity only; eligible-weight and total-model sparsity are reported separately;
unsupported tensors and embeddings are preserved; recipes, calibration identifiers, seeds and tie-breaking are predeclared. Price at
the measured 1.04 min per TF-v1 entry: 45 entries ≈ 0.78 GPU-h scoring; construction capped at 0.5 GPU-h under stop-and-reprice;
SparseGPT solve cost unmeasured. Resolution floor for the ladder: the registered 64-window WikiText endpoint's qualified panel
half-width 0.00487 nats and per-entry maximum 0.01282 nats (`docs/reviews/P5/tf-v1-requalification-2026-11-06.md`).

**F9.4 — H5 measured localisation and verbal-localisation treatment.** A fifth access flag `field_localisation_access` governs
whether the editor receives the field's per-tensor ranking (gradient saliency g⊙W, candidate-specific g⊙D, cached down-projection
input norms); the editor's own verbal site ranking is logged before it sees that ranking. Paired H5 episodes: scripted estimator with
field localisation; scripted estimator on the LLM's chosen layers; LLM editor with and without the field ranking; a
shuffled-localisation control at matched information and opportunity. The LLM is credited with localisation value only if the
LLM-localised variant beats the field-localised variant on held-out gain per real evaluation, paired, under A5.7 accounting.
Attribution patching and causal validation are new metered artefacts and are not registered here. Inside the G5 caps.

**F9.5 — test-time LoRA shadow sub-stream.** Within G5: 32 frozen development queries, one isolated base-relative LoRA each trained
with next-token loss on the query context (no generation; no sealed item text), access flags (python 1, forward 1, hidden-state 0,
feedback 1) plus `backward_enabled`, `gradient_exposed_to_editor`, optimiser, steps, rank and module allowlists and training-data
provenance. The critic forecasts collateral change on the registered objectives relative to the actual incumbent; forecasts are
logged under F9.1 before scoring; all 32 are scored on TF-v1. Read-outs: E₀ and sign agreement; savings from evaluating only
above-floor forecasts at a missed-harm upper bound ≤ 5 %; forecast wall time measured and compared with evaluation time. Training
cost is paid in both arms; only verification evaluations count as savings. Shadow replay may reject an uneconomic gate; realised
savings remain A5.7's prospective paired G8 claim. Price ≈ 0.55 GPU-h scoring plus training capped at 0.5 GPU-h.

**F9.6 — F8-E generation endpoint (descriptive; prerequisite for any H5 math-editor claim).** For the exact published Astra edit
and any F8-E entry with a published generation claim: MATH-500, all 500 items, three arms (unedited thinking-on, edited thinking-on,
unedited thinking-off), fixed 8,192-token cap and decoding settings, natural-completion rate and strict correctness reported
together with their paired difference. Screen: five independent paired seeds per item per arm. Qualification: two independent
five-seed batches per item and arm plus a same-device same-seed replay with a pre-declared A3-style margin. Unpriced: no comparable
generation-throughput measurement exists; the qualification ceiling is 122.88 M generated tokens; a timed same-device pilot precedes
any purchase. This endpoint never alters F5.5 and is not an H5 endpoint until qualified.

**F9.7 — no decision authority.** F9 panels and statistics cannot satisfy, fail or unresolve any F5.5 predicate or A5.5 step, cannot
change A5.7, and fund nothing. A JEPAWG-style latent model is not registered; it may be proposed by a later amendment only if the
F9.1 log shows structured residuals the first-order terms cannot represent.

**F9.8 — brief corrections recorded.** arXiv 2508.02381 predicts a divergence to the dense model for structured channel pruning from
775 evaluated masks; arXiv 2611.03244's 82 % is in-sample and 66 % out of sample; arXiv 2610.13991's forecaster is an editable
program calling a frozen LLM, calibrated by online bias plus conditional MSE; arXiv 2604.25098 tests unstructured pruning at 10 % and
20 % only, without SparseGPT; arXiv 2610.12002's evaluation stream was deliberately enriched for test-time-training benefit.

**F9.9 — ordering and governance.** Order: Part A → F8-E → F8-F → F9-P; H5 items inside G5. F9 panels use the F8a freeze protocol,
exposure cutoff, gate, cluster rule and scoring configuration (the Part A default `configs/eval_tf_v1.yaml`, per Board A's F8-spec
ruling). Re-anchoring: after any accepted edit in H5, the incumbent's exact gradient is recomputed (measured ≈ 43 s per row in the F6
audit) and factors are flagged `base-factor`; no learned transition is registered.

#### A5 amendment F8a (11 Nov 2026, before any G3 outcome): F8 execution rules

This block is appended; earlier text is not rewritten. It adopts the F8 execution specification
`docs/design/F8_EXECUTION_SPEC.md` (added by commit `158e92cf036b24191fd1f6570f37afae9b8e3076`, file sha256
`d2b981f07258b947f476c774013b4bfd8085a3745ffd28d196406128ce9433d0`; "the spec") as binding F8 execution rules, with the
overrides required by Board A's review of the spec (`docs/reviews/council/board-a-f8-spec-2026-11-11.md`, commit
`8e8e63550c64253a42ff8f9569bd0ae029e6dfba`, sha256 `f6f967d007a067ae62db8fa6bc394d157e3425a390bde7708f9a713fc50b62a1`;
"Board A") and with Board A's earlier F8 review (`docs/reviews/council/board-a-f8-2026-11-11.md`, sha256
`4de06af0e5da86f93ddc0fc516a65bbd209523bccd977ea05b6a6e75e48cb0b7`). The spec is incorporated by that immutable reference; where
a clause below differs from the spec, the clause governs. It was written at main `f181f0a`, before any G3, rotation, ladder,
sealed or F8 outcome had been read; no Part A score exists to be read. It changes no registered arm (A5.4), Part A selection
(A5.1.2), decision rule (A5.5), line-stop rule (F5.5), H5 success criterion (A5.7), seed, cap or pin; the frozen A5, F5, F6 and F7
decisions stand, and Part A keeps priority and does not wait on F8. F8a adds F8's own scoring caps (F8a.8), which borrow from no
registered allocation. Like F8, it has no decision authority (F8.6). Adoption is design acceptance only: no replay is certified,
no manifest is frozen, and no F8 score or H5 episode is authorised by this block. Numbered clauses F8a.1–F8a.8 discharge Board A's
conditions 1–8 of the spec review in the same order.

**F8a.1 — freeze, hash chain and exposure (condition 1).** Spec §1 is adopted (one freeze of both panels before exposure; C1
before any F8 delta is contracted with the field; C2 before the first F8 TF-v1 score; disjoint development inputs, §1.4; no
scored-endpoint feedback; score order Part A → F8-E → F8-F) with these repairs. (a) *Acyclic chain.* C1 and C2 write disjoint
file sets under `docs/prereg/f8_manifests/`; no C1 file is rewritten after the C1 commit and no file contains its own commit's sha
or commit time. C1 files, in this order: 1 `exposure_c1.json`, 2 `construction_inputs.json`, 3 `candidates.json`, 4 the floor
payload copy (F8a.2), 5 `floor.json`, 6 `astra_qualification.json`, 7 `code_hashes_c1.json`. C2 supplements, in this order:
8 `floor_reproduction.json`, 9 `exposure_c2.json`, 10 `code_hashes_c2.json`, 11 `deltas.json`, 12 `predictions.jsonl`,
13 `gate.json`. `H_0 = sha256(utf8("f8-manifest-v1:" + S))`, where S is the full git sha of the commit that first adds this F8a
heading to `docs/PREREG_H1_TF.md` (`git log --reverse --format=%H -S '#### A5 amendment F8a' main -- docs/PREREG_H1_TF.md`,
first line); `H_k = sha256(H_{k-1} || sha256(file_k))` over raw 32-byte digests. `MANIFEST_C1.sha256` (C1) holds H_0–H_7;
`MANIFEST_C2.sha256` (C2) holds H_8–H_13, starting from H_7. The final chain head is H_13. The verifier recomputes both
manifests and checks that every C1 file's git blob at C2 equals its blob at the C1 commit; any mismatch refuses scoring. Actual
commit times are attested externally by git commit metadata on the merged branch, not by file contents. (b) *Code pinned for C2.*
`code_hashes_c1.json` pins every source file C2 will execute (construction, prediction, gate, verifier, identity verifier of
F8a.4). `code_hashes_c2.json` must equal it. If such code must change after C1 and before any F8 delta has been contracted, the
change is an auditable pre-exposure restart: C1's bytes are kept; a new file set is written in a dated restart subdirectory with
its own chain anchored to the previous chain head, the diff's sha256 and the reason; an attestation states that no contraction,
gate quantity or outcome existed. After any contraction value exists, no code change is allowed without a new dated amendment that
discloses that the contraction values were visible. (c) *Exposure attestations cover every decision-maker.* Spec §1.3's
definition of exposure is kept and includes summaries (any message, document or log line conveying a G3, rotation, ladder or F8
outcome). Validity condition 1 of spec §1.3 is replaced: C1 and C2 are pre-outcome only if no person, agent session or process
that makes or approves an F8 choice (the chair, every builder session, every recipe or parameter selector, every reviewer approving
an F8 parameter) has been exposed. `exposure_c1.json` and `exposure_c2.json` carry one attestation line per decision-maker, with
role, session id and the time through which it holds; `exposure_c2.json` covers the whole interval from F8a adoption to its own
generation. The F8 scoring record (written after C2, spec §1.3) adds an attestation that no exposure occurred between
`exposure_c2.json`'s generation time and the C2 commit time recorded by git. Command-log hashes and the mechanical grep of spec
§1.3 are retained as support for the attestations; they do not prove absence of exposure. (d) *After exposure.* If any attestation
cannot be given or the grep hits, the panel is exploratory and an explicitly exploratory amendment is required before anything is
scored (spec §1.3). (e) *Links.* The scoring record links the final chain head, the sealed Part A ledger hash (A5.3 "Analysis
time") and the F8-E then F8-F scoring run ids. No F8 score precedes the complete C1/C2 freeze, the gate and the sealed Part A
ledger.

**F8a.2 — combined gate and floor (condition 2).** *Gate.* One gate across one fixed E+F candidate set (spec §2.1, combined
reading adopted; the per-panel fallback is not adopted), over the six planned source-component slots C1–C6 of spec §2.2, with
the recipe defaults of spec §2.2 frozen as stated (C3: layer 11 o_proj, ρ = 0.05; C4: layer 12 down_proj, ρ = 0.05, labelled an
own-definition contrast, never a Concept Country reproduction; C5: ⌊0.0004·out·in⌋ lowest-Wanda entries of each of the 24
`mlp.gate_proj`; C6: count-matched random mask; 64 construction prompts/windows per recipe; ladder {0.25, 0.5, 1} plus the
registered rotation). `gate_pass ⇔ #{qualifying components} ≥ 4 ∧ f_surv ≥ 0.75` (spec §2.6), with `qualifies_c ⇔ Q_c finite ∧
Q_c > h*` strictly, `Q_c = |a_rep| + q_rep` (spec §2.4). Each representative is the first available entry in spec §2.2's
availability hierarchy, at s = 1 unrotated, versus the base; never chosen by field magnitude or by later resolution. Missing
components do not lower the four. The loop family (Astra, aiXiv, LP and descendants) is one component; F8-E alone can supply at
most two, so an F8-E purchase supported by F8-F components is disclosed and never reported as "E passed the gate". Passing the
screen neither measures resolvability nor certifies an effect: a negative a and positive q can cancel (spec §2.4); selection on the
screen is prediction-conditioned and disclosed. *Dependency graph.* `candidates.json` freezes, in C1, a dependency graph over
C1–C6 with typed edges: shared construction items (any overlap of the sorted item IDs in `construction_inputs.json`), shared
direction or activation statistic, shared generator or seed stream, and matched support or counts (C6 ← C5). Components joined by
a shared-item, shared-direction or shared-generator edge are merged into one component before the gate counts. A matched-count
edge alone does not merge C6 into C5; the stated rationale is that, conditional on the fixed tensors and per-tensor counts, C6's
positions come from an independent stream and use no C5 activation statistic. That is an assumption, not a certified independence
of outcome signs, and it is reported as such. *Survival.* Spec §2.5 is adopted, with: N_e includes every A5.3.3-excluded
rotation as non-surviving even though no rotation bytes exist; an entry whose intended norm is zero fails; a no-op entry is
non-surviving and stays a disagreement (F8a.3); N_e = 0 makes f_surv undefined and the gate fails. *Edge rules for the gate.*
Unsupported curvature contributes 0 to q_rep with `q_partial = true` (spec §2.4); such a component is labelled partial and is not
complete quadratic support. Only a gradient-field abstention (missing gradient for an edited tensor or nonfinite a) disqualifies a
representative; the mandatory chart abstention (F8a.3) never does. *Floor h\*.* Exactly spec §2.3: file
`data/runs/a5/nullsize/inputs.json`, sha256 `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de` (pinned in
`docs/prereg/a5_manifests/nullsize/report.json` → `inputs.sha256` and `inputs_provenance.json` → `payload_sha256`); fields
`centred_noise` [141, 64], `window_mask` [141, 64], `donors[j].{id, component, family, eligible}`, row j of the arrays being
`donors[j]`; units WikiText `extra.mean_nll`, nats per valid next token, 1,023 tokens per window, scale 1.0; pairing into the 32
two-window blocks (2k, 2k+1) by `scripts/a5_simulation.py:donor_block(noise, mask, 1.0)`; donor set every `eligible` row with
all 64 windows masked-in and finite (expected 141 pairs in 63 components, `report.json` → `sufficiency`); missingness: a donor
with any masked or nonfinite window is excluded and counted, never imputed (0 at this payload, `validation.json`); interval:
percentile bootstrap of the block mean, 4,000 draws, one common [4000, 32] index array across all donors (one call of
`a5_simulation.resolution` on the [32, n_donor] block matrix in payload row order), linear .025/.975 quantiles; RNG
`numpy.random.Generator(numpy.random.PCG64(1731840362851662234))`, the `domain_seed` of ("20261111:f8-floor:", "resolution");
per-pair h_j = (q.975 − q.025)/2; donor weighting by component: h_c = `numpy.median` of h_j within component c, h\* =
`numpy.median` of the h_c over components. h\* is a historical noise screen, not a calibrated resolution probability. No fresh
Part A or F8 outcome and no rewritten null calibration enters it. *Committed data and independent reproduction.* The payload is
not in git (`git ls-files` lists nothing under `data/`); the committed NULL-SIZE files hold provenance, aggregate counts and
traces, not the residual values or interval endpoints. Therefore, as C1 file 4 of F8a.1, a byte-identical copy of the raw output
`data/runs/a5/nullsize/inputs.json` (651,039 bytes; written 8 Nov 2026 by `python scripts/a5_nullsize_inputs.py materialise`,
whose log `data/runs/a5/nullsize/materialise.log` records the same sha256) is committed as
`docs/prereg/f8_manifests/floor_payload/nullsize_inputs.json`, with the floor code pinned in `code_hashes_c1.json`; the builder verifies the sha256 before
use and `floor.json` records the value, the payload and code sha256, numpy/Python versions, every h_c and the sensitivities. The
spec's 0.0049397 nats/token is an unreproduced design preview until recomputed. Before any F8 delta is contracted, a second run
by someone other than the builder, from a fresh clone of the C1 commit with no access to `data/`, recomputes h\* and every h_c from
the committed payload copy and committed code and writes `floor_reproduction.json`; any difference from `floor.json`, or from
0.0049397 in any printed digit, stops the build and is reported; the preview is never silently replaced. Pair-weighted and family
sensitivities (spec §2.3: 0.0011513 over 141 pairs; 0.0091528 LoRA, 0.0006664 chain) are published beside h\* and are never
selected among.

**F8a.3 — predictions, scaling and edge values (condition 3).** Spec §3 is adopted (schema §3.2, order §3.3, realised-bf16
contraction, rank/support audits, missing/no-op ledger, no refit), with these overrides. (a) *Scaling.* Each row's a_s and q_s are
contracted on that row's own realised increment D_eff(s), so they already carry the scale: under exact scaling D_eff(s) = s·D_eff(1)
one would have a_s = s·a_1 (linear in s) and q_s = s²·q_1 (quadratic in s), but bf16 rounding breaks exact scaling and only the
realised values are used. The row's quadratic-sign forecast is **sign(a_s + q_s)**, replacing spec §3.2's sign(a + s·q), which
applied the scale twice to curvature. A row-relative forecast for a further multiplier t of that row's realised direction is
t·a_s + t²·q_s, with flip point t\* = |a_s|/q_s defined only for a_s < 0 and q_s > 0. The absolute s\* = |a_1|/q_1 from the
s = 1 representative is a nominal full-strength-direction forecast; it is not a forecast for the bf16-rounded ladder points, and is
labelled so. This reading replaces F8.3's "sign flip at s\* = |a|/q". (b) *Edge rules (spec §3.4, completed).* a_s = 0: first-order
prediction 0, and a resolved zero prediction is a disagreement (A5.3). q_s = 0 with a_s ≠ 0: κ = 0, no t\*, forecast sign(a_s).
a_s = 0 and q_s > 0: κ is the computed edge value +∞, stored as JSON null with `kappa_status = "edge:a_zero_q_positive"`.
a_s = 0 and q_s = 0: κ null with `kappa_status = "undefined:a_zero_q_zero"`. A nonfinite field input or contraction: the field
abstains for that row (`field_status = "abstain:nonfinite"`), all derived values null with reason, and a representative so marked
does not qualify. t\*/s\* null with reason outside a < 0 < q; at t = t\* the forecast is 0 (reported, not counted). No-op entry
(realised change exactly zero): a = 0, q = 0, status `no_op`, counted as a disagreement, not scored. q stays an unsigned Fisher
magnitude, never a signed Hessian term (A5.8.1); the quadratic forecast is descriptive and never enters Part A, A5.5 or F5.5. Every
non-finite or undefined value is a JSON null or a defined status string, never NaN or Infinity. (c) *Partial q.* When
`q_partial` is true, a, a_per_tensor, the first-order sign and the partial q sum are reported, labelled partial; κ, t\*, s\* and the
quadratic-sign forecast abstain with `abstain:q_partial`. (d) *Field identities.* Both fields are pinned in full: `wikitext_nll`
identity `6fab5b29c1fb70f5b08e1ae09f098088467b3bc649a234a5b12f4bc91223a87e` (decisive, descriptively) and `mmlu_gold_log_prob`
identity `00c76046a3bdd27ed111a5b6a6aaa7177809d713e5931399a9cecbf2a535d581` (descriptive only), with
`g2_manifest_final.json` sha256 `c4ec7390aaf118ed785c390ce337a4077936a31204f97da54341bcfa54fd09ff` and the sha256 of each field's
`manifest.json` and `tensors.pt`, loaded through `FieldArtifact.load(..., expected_identity=...)`. (e) *Construction
degeneracies, fixed before C1.* A zero or nonfinite normalising norm (ŝ, û, v̂ or the C3 v) makes the construction fail:
the entry is `missing: construction_degenerate`, retried only with the same inputs and seed (spec §1.4). In C3's
v = Σᵢ sign(μᵀvᵢ) σᵢ vᵢ, sign(0) = 0, so a singular vector with exactly zero projection contributes nothing; this makes v independent
of the SVD's sign convention. Activation pooling for C3, C4 and C5: plain text without a chat template, each item truncated to
256 tokens (spec §2.2's C4 truncation applied to C3 as well), base-model forward in bf16 with fp32 accumulation; C3's μ and ŝ and
C4's means are means over all valid tokens within each item, then over items with equal item weight; C5's ‖X_j‖₂ is taken over all
valid token rows of the 64 windows. (f) *Chart.* Every F8 entry carries `chart_status = "abstain:no_frozen_chart_predictor"`
(spec §3.5); chart columns are hashed descriptively; no chart model is fitted for F8 and no Part B output is substituted later.

**F8a.4 — Astra replay qualification (condition 4).** Spec §4 steps A1–A11 are adopted unchanged as preconditions for including
the Astra entry: provenance (A1), form (A2), tokenizer/config allowlist (A3), complete tensor-name set (A4), shapes and dtypes
(A5), exactly the two claimed changed tensors (A6), byte-identical complement with complement hash (A7), changed coordinates with
prose mismatches recorded rather than failing (A8), dense zoo entry with bitwise round-trip (A9), full-state identity (A10) and
fresh-process reload determinism (A11). Spec A12 is replaced, because `wsp.eval.batch` has no dry-run option and the scorer's
`--dry-run` only emits commands: inclusion additionally requires a committed **CPU identity verifier**, added prospectively and
pinned in `code_hashes_c1.json`, that (i) loads the staged entry through the production apply path (`wsp.zoo.apply`, as the scorer
does) on CPU; (ii) computes, with the production functions, the full-state hash, `effective_config_hash` of
`configs/eval_tf_v1.yaml`, the model and tokenizer revisions, the tokenizer file hashes, the per-benchmark item counts of that
config and `entry_hashes` {`delta.safetensors`: sha256}; and (iii) compares them with the expected values and with the base
reference's recorded identity. No endpoint is run to obtain this evidence before C2. Reuse of an existing base reference requires
every identity field to match, not the config filename alone. *Provenance.* The target model and tokenizer revision
`2fc06364715b967f1860aea9cf38778875588b17` is taken from `docs/prereg/a5_manifests/g2/g2_manifest_final.json` → `model`.
`docs/prereg/a5_manifests/snapshot.json` (sha256 `1b2af25cde1bd7ad0427fec2d0ee22d34f75651ba3fb979176464f8fad701da3`, closed
2026-11-08T00:59:56Z) is linked as the immutable pre-G2 artefact/record hash index; its `base_ref`
`19f5cbf781cb0955fdfe482c12ec523cc2873dc6` is a repository revision, not a model pin, and the snapshot does not contain the HF
revision string. Any failure makes C1(i) `missing: replay not qualified` with the failing step; nothing is repaired into an exact
replay. This block certifies nothing; only `astra_qualification.json` built by committed code can. aiXiv stays a
reconstruction; unreleased or incomplete LP and Scope artefacts stay missing; loop descendants stay one component.

**F8a.5 — cluster rule and binomial wording (condition 5).** *Rule (spec §5.1 adopted).* Unit: one frozen component slot
C1–C6 (after any merge of F8a.2). Primary contrast: the representative's s = 1 ladder contrast y(W₀+D_eff(1)) − y(W₀) on WikiText,
against sign(a_rep). Rotation and s = 0.25/0.5 contrasts and mixed signs across scales are reported per entry and never change the
cluster outcome. *Status precedence*, applied in this order, first match wins: (1) `missing` (no bytes, failed replay, unfunded,
including `missing: unfunded (cap)` under F8a.8, or failed construction); (2) `disagree` if the representative is a no-op, whether or not the gate
passed; (3) `abstain:field` if the gradient field abstained (a chart abstention never applies here); (4) `unscored` if the gate
failed or the cap stop of F8a.8 left it unscored; (5) `unresolved` if the score is missing or nonfinite (with reason) or the
interval contains 0; (6) `agree` if resolved, pred ≠ 0 and the signs match; otherwise `disagree`. *Ledger.* All six slots are
reported every time, with their status and reason; N_c = 6 does not shrink when a slot is missing. k = #agree, n = #agree +
#disagree. n = 0 is no test and no scientific conclusion. *Independence.* `candidates.json` records in C1 whether the frozen
dependency graph (F8a.2) supports independent cluster Bernoulli signs, with the rationale; if not, only descriptive entry and
cluster counts are reported and no binomial inference is made. Conditional independence and fair-coin signs, and selection by
resolution, are assumptions, not certified by the gate. *Wording.* F8.5's sentence "the report states that a panel of this size
cannot reject chance agreement" and spec §5.2's replacement sentence are both replaced by: "Agreement is summarised over the n
resolved source clusters of N_c = 6 frozen slots, with the exact two-sided binomial test of p = ½ and the exact 95 %
Clopper–Pearson interval. At the 5 % level, evidence of above-chance agreement requires 6/6 (p = 0.03125, CP lower bound 0.541);
two-sided rejection also occurs at 0/6, which is evidence of anti-agreement, never of transfer; with n ≤ 5 neither tail rejects
(5/5 or 0/5: p = 0.0625; CP lower bound 0.478 at 5/5). The interval is a descriptive precision statement." The report always
states the actual k, n, p, interval and direction. If a slot is missing so that n ≤ 5, it says the test cannot reject at this n.
No result satisfies, fails or unresolves any A5.5 step or F5.5 predicate or becomes a G8 condition; a resolved agreement is a
transfer precondition only, a resolved disagreement is prediction failure on that fixed endpoint without identifying locality versus
K-FAC error, and an unresolved cluster says nothing (F8.6).

**F8a.6 — H5 reconciliation (condition 6).** Spec §6 (difference-of-means inputs, LP objective, deterministic selection and
ties, α grid, S-loc/S-field, treatments, Sol pin, per-stream contract, drop order) is adopted as the H5 scripted-arm contract and
must be appended verbatim with the overrides below to `docs/H5_DESIGN.md` and `docs/H5_PROTOCOL.md`, with `docs/HYPOTHESIS.md`
reconciled, before the H5 pilot. (a) *Access wording.* "run by a pinned script with no model calls … and nothing else the editor
does not" (`docs/H5_PROTOCOL.md` 11 Nov section; same text in `docs/H5_DESIGN.md`) is replaced by: "makes no LLM editor or provider
calls; it makes metered diagnostic model forwards (base generations and hidden activations, spec §6.1) through the same
forward-request meter, charged to the arm; S-field additionally receives the frozen G2 field as setup access". The actual access
flags of every arm (spec §6.3 table, plus F9.4's `field_localisation_access`) are pinned in the G5 manifest before the pilot.
Comparisons at unequal access are labelled and descriptive; they do not meet the matched-information requirement for an
editor-superiority claim. A T0 script given |L| and B is conditional on that prior, not unrestricted discovery. (b) *Retained arms.*
Astra stays the registered primary (A5.7 primary comparison: Astra with versus without the critic); Haiku, sham reload, random
ranking and the F5.4/FP5 nested 1/8/32 best-of-N are retained with every other A5.7/G5 comparator; arms are dropped only
prospectively for funding, never by observed performance. (c) *Complete A5.7 estimator per stream*, unchanged: ratio of total real
calls including failed calls and rejected-proposal audits; both totals finite and positive at the point and in every draw;
connected episodes clustered beyond the goal × seed × budget tuple when source-, direction- or prompt-connected; paired
episode-cluster bootstrap-t with 4,000 common draws, PCG64 seed 20261107, and its failure rules; bounded score endpoint with equal
episode weighting; missed-proposal denominator of all rejected candidates including failed attempts, unauditable rejected
attempts counted as missed, zero rejected attempts undefined; net GPU saving charging diagnostics, gradients, probes, failed
attempts and amortised preparation (including the G2 fields' 1.62 GPU-h, `g2_manifest_final.json` → `caps.fields.used_hours`);
prospective joint power ≥ 0.80; unresolved versus kill distinctions. Critic and no-critic share seeds and opportunity budgets
within a stream; realised calls are observed, not forced equal. A zero or nonfinite denominator makes a stream's savings undefined,
never a success. (d) *Frozen choices.* The α-selection metric is FP5's development-winner objective (mean paired improvement on
development support relative to the actual incumbent under A5.7's feasibility/safety predicates), ties within 1e-12 relative to
the smaller α; the proposal-training, support-validation, acceptance-slice, calibration and sealed splits (spec §6.1; F5.4; FP5;
A5.7) and the actual-incumbent comparisons are frozen in the G5 manifest before calibration. (e) *F9.4.* The F9.4 paired set
(scripted with field localisation, scripted on the LLM's layers, LLM with and without the field ranking, shuffled-localisation
control) uses the same estimator and access pinning; the LLM is credited with localisation value only under F9.4's paired rule.
(f) *Cost and drop order.* Every H5 stream here is unpriced (F8a.8). The pilot is repriced inside the existing G5 caps (≤ 10 GPU-h
plus ≤ 6 for best-of-N) before launch; if it does not fit, streams are dropped in this fixed order: the F9.4 paired set as one unit
(registered last), then spec §6.6's order (S-field; T1 as one unit; Sol T0). Astra, Haiku, sham and random best-of-N are never
dropped; dropped streams are reported as unfunded. G5 stays shadow-only, G8 keeps its existing eligibility (G5 feasibility plus
G3/G4 go), and every A5.7 rejection and acceptance safeguard is retained. No episode is authorised by this block.

**F8a.7 — scoring configuration (condition 7).** F8.5's `configs/eval_tf_v1_wikitext.yaml` is replaced by
`configs/eval_tf_v1.yaml` (file sha256 `5c20cabe8e35b4116adeed4df650cf27461e7ab144ef66c0b60c397b13b00317`), the Part A default
(`response_intervention.DEFAULT_CONFIG`) and the configuration F9.9 already names. Reason: the restricted file is the F5.3 4B
WikiText diagnostic whose header states that its config hash differs and its records never pool with full TF-v1 records; under it
(a) the descriptive MMLU endpoint of F8.5 would not be scored, (b) the common base reference could not be reused from records
with the full scoring identity, (c) the floor noise (F8a.2) comes from full-config records, and (d) the measured per-entry rates of
F8a.8 were measured on the full config, while no WikiText-only H0 rate exists in `docs/RESULTS.md` (spec §2.8; Board A §7). C1
records the effective scoring identity (`effective_config_hash`, model and tokenizer revision, and the item counts MMLU-Pro 1,000,
HellaSwag 250, KL 200, WikiText 64 windows of 1,024 tokens at revision `b08601e04326c79dfdd32d625aee71d232d685c3`, refusal 50,
over-refusal 50). WikiText decides for the panel descriptively; MMLU stays descriptive. Base reuse requires the complete identity
match of F8a.4. A WikiText-only redesign would need a separate endpoint and pricing amendment.

**F8a.8 — funding, pruning and provenance (condition 8).** *Caps.* F8-E scoring ≤ 0.3 GPU-h and F8-F scoring ≤ 0.6 GPU-h
(synthesis estimates, Board A F8 review condition 2), each inclusive of loads, base reference, retries and overhead; F8-F
construction ≤ 0.5 GPU-h (F8.2). Pricing uses 1.04 min per entry, 47.1 s per process cache load and 1.751 min for a cold base
entry (`docs/RESULTS.md` TF-v1 pilot rounds 2 and 3; spec §2.7). These are reference-rate estimates, not measured F8 runtimes.
*Pruning rule, fixed in C1 before any contraction.* The purchase population and the gating population are frozen together in
`candidates.json`. Entries are bought in whole members of four ({0.25, 0.5, 1, R}). Within each panel, members are added in this
order while the projected panel cost, charging one cache load and always a cold base reference, stays within the cap: first every
component's representative member (spec §2.2 order), then the remaining loop-family members in hierarchy order. For F8-E this
admits at most three members (12 entries, 0.25027 GPU-h; 16 entries would be 0.31960 > 0.3, Board A §8); F8-F's 16 entries
project to 0.29042, or 0.31960 if the base reference falls to F8-F. A member not admitted is listed `unscored: cap`, and a component
whose representative is not admitted is `missing: unfunded (cap)` for the gate and for F8a.5; no unbought entry supports the gate.
If the gate passes, every constructed, non-no-op, non-excluded entry of the purchase population is scored; "buys every frozen
entry" in spec §2.1 and §2.6 means every entry of this purchase population. *Stops.* Scoring of a panel stops before projected
overspend at 1.04 min per remaining entry; a wall-time watchdog stops it if any entry exceeds 2 × 1.04 min wall excluding the
process's single cache load, which is charged separately (the round-2 new-process entry measured 2.60 min including its load,
`docs/RESULTS.md`); construction stops at 0.5 GPU-h. Retries, new loads and cold references are charged inside the same cap.
Recovery: scored entries are kept and reported; unscored entries are listed `unscored: stopped`; continuation requires a dated
repricing amendment computed from timing and cost logs only, never score values, which may change funding but never the purchase
population, gate, floor or predictions. No Part A, G5 or G7 allocation is borrowed. *Status of every item in spec §7:*

| Spec §7 item | Price (GPU-h, H0) | Status under F8a |
|---|---|---|
| F8 floor, Astra qualification, field contraction, gate | 0 GPU-h (CPU; wall time, memory and storage are not zero) | funded (needs no GPU allocation) |
| F8-E scoring | 0.069–0.277 entries (+0.013 load, +0.029 cold base) | funded within ≤ 0.3, pruned to the purchase population above |
| F8-F scoring | 0.277 + 0.013 | funded within ≤ 0.6 |
| F8-F construction (activation statistics for C3–C5) | unmeasured | funded up to the 0.5 GPU-h cap; unpriced until measured; stop and reprice at the cap |
| F8-E aiXiv reconstruction | unmeasured (needs thinking-on generations) | **unfunded** as priced: C1(ii) is `missing: unfunded` unless a later amendment allocates it before C1 |
| H5 scripted/Sol streams (and F9.4 streams) | unmeasured: the 4096-token tier and diagnostic-forward latency are not measured; the 12.4 and 6.1 min/entry fast-tier references are not H5 rates | **unpriced** until the 4096-token tier is measured; repriced inside G5 ≤ 10 + ≤ 6 before the pilot; drop order F8a.6(f) |

Unpriced is not permission to spend to discover a price outside a bounded allocation. *Provenance.* Condition 7 of Board A's F8
review is carried forward as discharged through the retained-evidence README `docs/council/2026-11-11/evidence/README.md`
(sha256 `efb29eb199523764926e9810a800e1dc8e5942461996ab43cb916a165f17f4ce`) and is not reopened as a blocker. URL, version, access
date and file hashes of every recipe and download actually used go into `astra_qualification.json` and
`construction_inputs.json`. C4 is an own-definition contrast, not a certified reproduction; C1(iii) enters only with retrievable
bytes or a complete recipe; no number from a page that returned HTTP 403 to Board A is used as a threshold, funding predicate,
expected effect, component count or success claim.

| Board A condition | F8a clause |
|---|---|
| 1 Authority and freeze: named, dated, immutable spec reference; frozen A5/F5/F6/F7 and Part A priority; acyclic C1/C2 chain with separate C2 supplements; attestations for every decision-maker through C2; disjoint inputs; exploratory after exposure; sealed Part A ledger and score order | preamble; F8a.1 (a)–(e) |
| 2 Gate and floor: combined six-slot E+F gate, representative hierarchy, ≥ 4 independent components, strict floor, ≥ 0.75 survival with complete denominators; dependency map, construction parameters, seeds and purchase population frozen; exact committed payload and floor code/output; preview unreproduced until recomputed; sensitivities without choice | F8a.2 (gate, dependency graph, survival, edge rules, floor, committed data and independent reproduction); purchase population in F8a.8 |
| 3 Predictions: full schema and identities for both fields; realised bf16; row-wise quadratic scaling; relative versus absolute s\*; JSON-safe edge statuses; construction degeneracies and pooling; partial q; chart abstention; no refit | F8a.3 (a)–(f) |
| 4 Replay: A1–A11, real CPU A12 identity verifier, HF/G2/snapshot provenance, fail closed, reconstruction/missing labels | F8a.4 |
| 5 Cluster reporting: s = 1 representative rule, status precedence, six-slot ledger, independence justification, both binomial tails, n = 0, descriptive authority only | F8a.5 |
| 6 H5 reconciliation: §6 contract into H5 documents; access wording; matched information; complete A5.7 estimator; frozen splits and α metric; repricing and drop order; shadow-only G5 and G8 eligibility | F8a.6 (a)–(f) |
| 7 Configuration: `configs/eval_tf_v1.yaml` replaces F8.5's file; scoring identity and counts pinned; WikiText decisive descriptively, MMLU descriptive; reuse only on complete match | F8a.7 |
| 8 Funding and provenance: all-entry versus cap pruning resolved before the purchase freeze; caps with overhead; stop, retry and reprice policy; aiXiv status; H5 repricing and drop order; condition 7 carried forward; C4 labelling; no blocked-page numbers | F8a.8 (and F8a.6(f) for the H5 drop order) |

#### A5 amendment F9a (11 Nov 2026, before any G3 outcome): F9 execution rules

This block is appended; earlier text is not rewritten. It adopts F9 subject to Board A's F9 review
(`docs/reviews/council/board-a-f9-2026-11-11.md`, added by commit `fedc7c842b5def72b7d4d075429d2b6441bc9f14`, sha256
`fb06563b67f78789f70619d34a86e3dcad0f7ff5a44a030a4fc2a2b4fa52a5ff`; "Board A") and discharges that review's eight numbered
conditions in clauses F9a.1–F9a.8, in the same order. It was written at main `097b106` (F8a adopted by `e216067`), before any G3,
rotation, ladder, sealed, F8 or F9 outcome had been read; no Part A score exists to be read and none was opened. It changes no
registered arm (A5.4), Part A selection (A5.1.2), decision rule (A5.5), line-stop rule (F5.5), H5 success criterion (A5.7), seed,
cap or pin; the frozen A5, F5, F6, F7, F8 and F8a decisions stand, and Part A keeps priority and does not wait on F9. F9.7 is
retained in full: F9 panels and statistics cannot satisfy, fail or unresolve any F5.5 predicate or A5.5 step, cannot change A5.7 and
fund nothing. The bounded allocations named in F9a.4, F9a.5 and F9a.7 are F9's own, are funded only for the purchases named there,
and borrow from no registered allocation (F9.5's shadow stays inside the existing G5 cap). Where a clause below differs from F9
(including F9.1–F9.9 wording), the clause governs; F8a governs F8 and is applied to F9 clause by clause in F9a.1(d). Adoption is
design acceptance only: no manifest is frozen, no entry is constructed or scored, no adapter is trained, no generation is sampled
and no H5 episode is authorised by this block. Rules that are this block's own rather than Board A's are listed at the end for Board
A to confirm.

**F9a.1 — authority, provenance, order, F8a overrides and F9 purchase populations (condition 1).** (a) *Authority.* F9.7 is
retained; F9 adds descriptive panels, access treatments and a training permission (F9.5) that need this prospective amendment, and
calling them descriptive does not waive freezing. (b) *Provenance.* Member identities (role, provider, model, effort, retrieval
limits) are archived in `docs/council/2026-11-11/MEMBERS.md` (added by `c461ce021bd0fdd838f61575080c8cf7fb88255b`, sha256
`03081e305fdf3fa2441b8e9ae4be9590232ecb4064d1fb9150d7e6d2546da9fb`). The four round-5 reports Board A read are identified by Board
A's SHA-256 table: `round5-editor.md` `e093ff1fc2a7a56636053d1d60297a18467d046996f3eb32da2d2f5cae615b2d`, `round5-metrology.md`
`155b81a7e18d18720698a7bd2d1aef3f38a66a9587ffb020c296e61ec80108c8`, `round5-representation.md`
`cd860d3384c4808bc7fa3eadeaa78d67de57cc9477f484b8758681accd91c0e8`, `round5-world-model.md`
`1c708bc8cc70a0b701b1e1c5ee0843e219f24d5b3e65ff41daef466b1c43847a`. At `097b106` they are not tracked in git; the untracked copies
in the council worktree (`.worktrees/council/2026-11-11/docs/council/2026-11-11/`) were re-hashed by the session writing this block
and match the table. No F9a rule depends on a report's content beyond what SYNTHESIS-r5, F9 and Board A state. (c) *Order (repairs
F9's opening sentence).* F9-L is operative before any F8 score: its batch 0 holds the F8 rows (F9a.2(e)) and is committed after
F8a's C2 commit and before the first F8 TF-v1 score. F9's "runs after the Part A ledger is sealed and after the F8 panels" applies
only to F9-P scoring, F9.4, F9.5 and F9.6. Scoring order stays Part A → F8-E → F8-F → F9-P (F9.9); F9-P scoring starts only after
the F8 scoring record is closed (F8-F scored, unscored or stopped under F8a.5/F8a.8); F9.4 and F9.5 run inside G5 (shadow,
production disabled); the F9.6 pilot runs only after F8-E TF-v1 scoring closes (SYNTHESIS-r5 §2) and after the F9a.7 generation
manifest is committed. (d) *F8a overrides, by clause.* F8a.1 (freeze, acyclic hash chain, code pinning, restarts, exposure): F9-P
gets its own chain (F9a.4(a)); F9-L batch 0 rides on F8's chain; spec §1.3's exposure definition as amended by F8a.1(c) (summaries
count) and F8a.1(d) (exposure ⇒ explicitly exploratory amendment before scoring) apply to every F9 freeze. F8a.2: only the floor h\*
("Floor h\*" paragraph, with its committed payload and independent reproduction) is applied, as the operative floor for every F9
quantity that names a floor (F9.2's predicted-resolvable × resolved table, F9.5's gate); F8a.2's six-slot E+F gate, representative
hierarchy and dependency graph are **not** inherited by any F9 population, no F9 entry counts toward or against the F8 gate, and no
F8 entry supports an F9 purchase; F8a.2's survival definitions apply to F9-P as in F9a.4(e). F8a.3 (a)–(d) and (f) (realised
scaling, edge values, partial q, field identities, chart abstention) apply to every F9-L row; (e) is F8-only. F8a.5's status
precedence and binomial wording apply to any F9 cluster report with "six slots" read as the population's frozen component count,
together with its independence record. F8a.6 governs every H5 item of F9 (F9a.5). F8a.7: every F9 TF-v1 score uses
`configs/eval_tf_v1.yaml` (sha256 `5c20cabe8e35b4116adeed4df650cf27461e7ab144ef66c0b60c397b13b00317`) and reuses a base reference
only on the complete identity match of F8a.4. F8a.8: the stop-before-overspend, watchdog (2 × 1.04 min wall per entry excluding the
single cache load), retry-inside-cap, recovery, reprice-from-timing-and-cost-logs-only and no-borrowing rules apply to every F9
allocation, with its reference rates 1.04 min per TF-v1 entry, 47.1 s per cache load and 1.751 min for a cold base entry
(`docs/RESULTS.md` TF-v1 pilot rounds 2 and 3); 1.040 min is the round-3 warm maximum (median 1.003, range 0.994–1.040), a reference
rate and not a measured high-sparsity, LoRA or generation runtime. (e) *F9's own purchase populations.* No F9 gate is built by
counting components:

| F9 item | What is bought | Precondition and floor | Allocation | Attests (F8a.1(c) roles) |
|---|---|---|---|---|
| F9-L (F9.1–F9.2) | nothing (CPU) | batch commit before scheduling (F9a.2(e)) | 0 GPU-h | chair; every session that writes rows |
| F9-P (F9.3) | every constructed, non-no-op, non-excluded entry of the 45 frozen entries, in the purchase order of F9a.4(i) | F9-P C1 and C2 complete; sealed Part A ledger; F8 scoring record closed; f_surv ≥ 0.75 over the 45 slots; exposure attestations; floor h\* recorded per entry, never a gate | construction ≤ 0.5 GPU-h (F9.3); scoring ≤ 0.82227 GPU-h (F9a.4(i)) | chair, builder sessions, recipe/parameter selectors, the independent floor reproducer of F8a.2, approving reviewers |
| F9.4 paired set | inside G5 under F8a.6(f) | G5 manifest frozen; no floor | G5 ≤ 10 + ≤ 6 (existing) | chair, G5 manifest author, every editor-session operator |
| F9.5 shadow | all 32 adapters, trained and scored | G5 manifest frozen including the F9a.6 gate (floor h\*, counterfactual only) | training ≤ 0.5 GPU-h (F9.5), scoring ≤ 0.59693 GPU-h, both inside the existing G5 ≤ 10 | chair, G5 manifest author, training-harness operator |
| F9.6 generation | the timed pilot only | F9a.7 generation manifest committed; F8-E TF-v1 scoring closed; no floor | ≤ 0.5 GPU-h (F9a.7(h)) | chair, generation-manifest author |

F9-P has no magnitude gate: the predicted-resolvable flag |a_s| + q_s > h\* is recorded for every entry and never selects one; F9-P
has at most three source components (F9a.4(h)), so a component-count gate could only be met by manufacturing components. (f)
*Attestations.* Pre-outcome status of every F9 freeze requires one attestation line per decision-maker (F8a.1(c)); staging counts
("27 of 40") are context, not proof. The session that wrote this block attests that it opened no Part A score (none exists), no G3,
rotation, ladder, sealed, F8 or F9 outcome and no file under `data/`. A design frozen after any decision-maker's exposure, and any
proposal informed by an outcome, is exploratory and needs an explicitly exploratory amendment before it is scored.

**F9a.2 — executable forecasts and statistics (condition 2).** (a) *Realised scale.* Each row's a_s = ⟨g, D_eff(s)⟩ and q_s = ½tr(G
D_eff(s) A D_eff(s)ᵀ) are contracted on that row's own realised bf16 increment (fp32 difference of stored bf16 states, A5.3), so
they already carry the scale (a scales with s and q with s² only under exact scaling, which bf16 rounding breaks). The row's
first-order forecast is **a_s** and its quadratic forecast **a_s + q_s**; F9.2's "s·a" and "s·a + s²·q" are replaced by these
(F8a.3(a)). Nominal full-direction values s·a₁ and s·a₁ + s²·q₁ may be reported only labelled `nominal` and never enter E₀,
agreement, calibration or the readout. Every ladder point, sparsity and rotation is contracted separately. (b) *Contrast and
reference.* Each scored row carries one contrast type. `edit_vs_reference`: target Δy = y(W_ref + D) − y(W_ref), first-order
forecast a, with W_ref the base for root rows or the actual incumbent (F9a.5(e)). `edit_vs_rotation`: target Δy = y(W_ref + D) −
y(W_ref + R), first-order forecast ⟨g, D_eff − R_eff⟩ (A5.3), quadratic forecast (a_D + q_D) − (a_R + q_R). The two are different
estimands with separate E₀, agreement and calibration, never pooled. Orientation: y is the endpoint in its native units and sign;
for WikiText `extra.mean_nll` (nats per valid next token) positive Δy is a worse fit. (c) *Notation.* κ denotes only the curvature
ratio q_s/|a_s|, with F8a.3(b)'s edge values. F9.2's parenthesis "κ is fixed at 1" reads "the linear forecast coefficient β on a is
fixed at 1" (zero-fit), and its "fitted κ̂" reads "fitted readout coefficients", governed by F9a.3. q stays a nonnegative
Fisher/Gauss–Newton magnitude, never a signed Hessian correction or an interval. (d) *Schema `f9l-v1`.* Append-only JSONL with three
row kinds. A `forecast` row holds F9.1's fields plus: row id; schema version; population (F8-E, F8-F, F9-P, F9.4, F9.5, G5 stream
id); family; frozen source-component id; endpoint, objective-field identity and manifest/tensor hashes (F8a.3(d));
`effective_config_hash` and config sha256 (F8a.7); orientation; contrast type and reference hash; scale or sparsity;
`gradient_anchor` ∈ {`base-exact`, `incumbent-exact`} and `factor_anchor` ∈ {`base-factor`, `incumbent-factor`} as separate fields
(replacing F9.1's single anchor type; an `incumbent-exact` gradient with `base-factor` curvature is a valid pair); gradient hash;
predictor version (F9a.3); `field_status`, `kappa_status`, `q_partial`, `chart_status` (F8a.3); construction status (failed
constructions carry no delta and are kept). A `decision` row holds the forecast row id and hash, the decision (schedule, skip,
accept, reject, abstain, no-op, failed), the decision-maker and time; F9.1's "decision taken" moves here. An `outcome` row holds the
forecast row id, score run id, Δy, interval, resolution status and any missingness reason. (e) *Immutable pre-decision commitment.*
Forecast rows are written in batches; each batch file's sha256 is committed before any of its rows is scheduled, scored, deployed or
accepted (F9.1's "before scoring" tightened to "before scheduling"). Decision and outcome rows go in later, separate files
referencing the forecast hash. No row is edited or deleted; failures, rejections, abstentions, skips and no-ops stay. Batch 0 is
derived from F8a's `predictions.jsonl` without rewriting it. (f) *Undefined, degenerate and missing values* (JSON null plus a status
string; never NaN or Infinity, epsilon repair or redraw): ΣwΔy² = 0, including 0/0 → E₀ `undefined:zero_denominator`; missing or
nonfinite Δy → row `unresolved:missing_target`, kept in the attempted count and out of the E₀ sums; field abstention → no forecast,
counted against coverage; a bootstrap draw with a zero denominator or a nonfinite value is invalid and any invalid draw makes that
interval `undefined:invalid_draws` with the count reported (as A5.7); calibration intercept and slope (OLS of Δy on the forecast)
are `undefined:constant_forecast` at zero forecast variance and `undefined:n_lt_3` below three rows; a zero forecast is a
disagreement (A5.3); n = 0 resolved components is no test. (g) *E₀ and its decomposition.* Over rows with finite forecast f and
finite Δy, with weights w summing to 1: E₀ = 1 − Σw(Δy − f)²/ΣwΔy², an uncentred squared-error skill score against predicting zero;
it can be negative without bound and is not a centred R². With r = Δy − f and b = Σw·r: MSE = Σw·r² = b² + CMSE, CMSE = Σw(r − b)²,
the **centred residual MSE** under the same weights. F9.2's "conditional MSE" is renamed CMSE; no conditioning strata are
registered, so no conditional-error estimator is claimed. Reported denominators: attempted, scored-finite, abstaining, missing and
unresolved rows and coverage. E₀ uses every scored-finite row whatever its resolution; a resolved-only E₀ is never primary. (h)
*Weighting, representatives, uncertainty.* Primary weighting: equal weight per frozen source component, split equally over its
scored-finite rows; row-weighted E₀ is reported beside it. Sign agreement and Clopper–Pearson intervals use one Bernoulli per
component, its representative being the first row of a per-population hierarchy frozen before construction (F8: F8a.5's
representative; F9-P: the ρ = 0.10 unrotated entry of the mask source; F9.5: the adapter of the lowest frozen query id in the
component); rotations, scales, sparsities and related masks never add trials. E₀ and calibration intervals: component bootstrap,
4,000 draws, SeedSequence 20261107 (F9.2), conditional on the fixed endpoint items. An item-population statement requires a joint
interval that also resamples the 32 two-window blocks with common draws across all rows and their references (the A5.7/A2.3 two-way
structure); otherwise none is made. Every report gives the actual interval width and the effective concentration (largest and
top-five component shares of ΣwΔy²). (i) *Floor.* The operative floor is F8a.2's h\*. The P5 figures are historical context only: on
the 229-entry, 64-item D3 development panel of `wikitext_log_ppl`, 0.00487 is the **upper 97.5 % component-bootstrap bound on the
panel-mean half-width** and 0.01282 the **observed maximum per-entry half-width**
(`docs/reviews/P5/tf-v1-requalification-2026-11-06.md`, wikitext_log_ppl table). They replace F9.3's "Resolution floor for the
ladder" sentence, are not a floor for new masks and are not h\*. New entries get their own paired intervals (A5.3 resolution rule:
32 two-window blocks, 4,000 draws, percentile, resolved if the interval excludes 0); no recipe or threshold is adjusted from those
outcomes. (j) *Preserved.* The zero-fit β = 1 primary, chart abstention (`abstain:no_frozen_chart_predictor` unless a chart
predictor was frozen before the batch commit) and q's magnitude-only status.

**F9a.3 — leakage control (condition 3).** (a) *Scope.* Online learning is limited to the A5.8.3 readout (intercept, Σa, Σq, ridge)
and the abstention quantiles, for future descriptive episodes only. F8 and F9-P forecasts stay zero-fit; registered Part B fits stay
frozen; no committed forecast is ever refitted. (b) *Splits.* Before any row of a population is scored, its source-connected
components (F8a.2's typed edges: shared items, documents, prompts, directions, activation statistics, generators or seed streams,
plus shared queries) are assigned by a committed rule to one role, `train`, `swap_validation` or `evaluation`, recorded in that
population's manifest; connected rows never straddle roles. (c) *Order, update protocol and versions.* Population order is frozen
now: F8-E, F8-F, F9-P, then G5 episodes in the order frozen in the G5 manifest, then the F9.5 shadow. Predictor version v₀ is
zero-fit (β = 1, no intercept, no abstention quantiles). Version v_{k+1} may be fitted only after population k closes, on `train`
rows of populations ≤ k; it is assessed once on the `swap_validation` rows of population k (those rows are then retired into
training and never reused for another swap) and is applied, frozen, only to forecast batches not yet committed. Swap rule: v_{k+1}
replaces v_k only if its component-weighted MSE on the swap-validation rows is strictly lower and its CMSE is not higher; otherwise
v_k stays. Every candidate version, accepted or rejected, is hashed and logged with its training-row ids and ridge penalty (fixed in
the version manifest before its training rows exist); every forecast row names its version. A fitted scale applied to a next family
is reported separately from the zero-fit E₀ primary. (d) *Eligible rows.* Only rows whose outcome role is `development` in a frozen
role map: scored F8, F9-P, F9.5 and G5 development-support rows. Never eligible: Part A, sealed (A5.9, G7), G8 calibration or
evaluation episodes, A5.7 rejected-proposal audits, any H5 held-out evaluation used in an F9.4 or A5.7 comparison, and current
evaluation rows. (e) *Prohibited.* Audit, sealed or current-test feedback; repeated adaptive use of a holdout; retrospective refits;
response-conditioned inclusion (rows chosen by outcome, resolution or realised magnitude); selecting recipes, hyperparameters,
abstention levels or an update on outcomes and then assessing it on the same outcomes; tuning the F9.5 gate on its 32 outcomes;
endpoint-guided mask repair or recipe retries (F9a.4). (f) *30/100.* Below 30 independent components per family: entries only, and
no pooled calibration claim; 30–99: exploratory; ≥ 100: a precision statement is allowed only with its actual interval width. These
are minimum reporting conventions, not evidence of calibration or precision; 100 independent Bernoulli components give roughly ten
percentage points of worst-case sign precision and no bound on E₀ precision under concentrated losses (Board A).

**F9a.4 — pruning execution and funding (condition 4).** (a) *Freeze.* Directory `docs/prereg/f9p_manifests/`, two commits on
F8a.1(a)'s pattern. C1, before any F9-P contraction: 1 `exposure_c1.json`, 2 `inventory.json`, 3 `construction_inputs.json`, 4
`candidates.json`, 5 `code_hashes_c1.json`. C2, before the first F9-P score: 6 `construction_cost.json`, 7 `exposure_c2.json`, 8
`code_hashes_c2.json` (equal to C1's), 9 `deltas.json`, 10 `predictions.jsonl` (the F9-L batch), 11 `purchase.json`. H₀ =
sha256(utf8("f9p-manifest-v1:" + S)), S the full sha of the commit that first adds this F9a heading to `docs/PREREG_H1_TF.md` (`git
log --reverse --format=%H -S '#### A5 amendment F9a' main -- docs/PREREG_H1_TF.md`, first line); H_k = sha256(H_{k−1} ‖
sha256(file_k)); `MANIFEST_C1.sha256` holds H₀–H₅, `MANIFEST_C2.sha256` H₆–H₁₁; verifier, restart and code-change rules as
F8a.1(a)–(b). (b) *Inventory.* The supported linear-weight inventory is every 2-D tensor for which the frozen `wikitext_nll` field
(identity `6fab5b29c1fb70f5b08e1ae09f098088467b3bc649a234a5b12f4bc91223a87e`) holds a gradient and both curvature factors;
embeddings, output head, norms, biases and any tensor lacking a factor are preserved and listed. `inventory.json` lists name, shape,
element count n_t and support; N_supp = Σn_t and N_total = all parameters of the pinned base; eligible-weight sparsity K/N_supp and
total-model sparsity K/N_total are reported separately. (c) *The 45 entries, exactly* (no extra scale or rotation variants): ladder,
18 = ρ ∈ {0.10, 0.20, 0.50} × source {Wanda, SparseGPT-transplant, random} × {unrotated, A5.3 rotation}, s = 1 only; native, 3 =
compensated SparseGPT at ρ ∈ {0.10, 0.20, 0.50}, unrotated; ranking, 24 = Wanda masks at global ρ = 0.20 with Dirichlet layer
ratios, unrotated. (d) *Counts and recipes.* Ladder and native entries delete exactly k(ρ) = ⌊ρ·n_in⌋ entries in every output row of
every supported tensor, so all sources match counts by row, tensor and total at each ρ. Wanda: score S_ij = |W_ij|·√A_jj with A the
tensor's frozen WikiText input factor as loaded by `FieldArtifact.load`; the A5.8.1 99 %-energy truncation is checked first, and if
the loaded factor does not give the exact diagonal the omitted trace fraction is recorded per tensor and the source is labelled
`wanda:truncated_diagonal`; the k(ρ) lowest scores of each row are deleted, ties by ascending column index. Random: k(ρ) positions
per row uniformly without replacement from `numpy.random.Generator(numpy.random.PCG64(seed))`, seed =
`scripts/a5_build_panels.py:domain_seed("20261111:f9p:", "random:<ρ>")`, rows visited in A5.8.3 tensor order, then row-major.
SparseGPT: one solve per ρ, in the order 0.10, 0.20, 0.50; per tensor H = A + λI with the same frozen input factor A and λ = 1e-3 ×
trace(A)/dim (A5.8.1's damping rule); mask selection over each full row at k(ρ) by the saliency W_ij²/[H⁻¹]_jj², ties by ascending
column index; compensation by the sequential column-wise optimal-brain-surgeon update in fp32, with no column blocking, no
sequential layer-wise recalibration and no other option; reference code pinned in `code_hashes_c1.json`. Because H comes from the
frozen base input factor rather than fresh activations, this is a one-shot base-statistic SparseGPT variant, labelled so, not a
reproduction of sequential SparseGPT. Transplant entry: D = −W⊙M with M the solve's mask on the original weights; native entry: D =
bf16(W_sgpt) − W = −W⊙M + C, C the survivors' compensation. Calibration inputs (the G2 WikiText factor windows) are checked in C1
for item and article disjointness from the 64 scored WikiText windows and any sealed panel; overlap fails closed. (e) *Rotation,
rounding, survival, degeneracy.* Rotations follow A5.3's rotation rule with the per-edit seed `domain_seed("20261111:f9p:",
"rotation:<entry id>")` and the A5.3.3 audit; excluded rotations are listed and count as non-surviving (F8a.2). Deletion entries are
exact in bf16 (zero is representable), so D_eff = −W⊙M; native and rotated entries are rounded to bf16 and contracted on the
realised increment. f_surv = surviving entries / 45 with F8a.2's definitions; f_surv < 0.75 leaves F9-P unscored and reported with
its predictions. A no-op is a disagreement and is not scored. A nonfinite solve or an H singular after damping is `missing:
construction_degenerate`, retried only with identical inputs and seed; no endpoint-guided mask repair and no retry with an altered
recipe. (f) *Ranking-panel allocation.* K = Σ_t out_t·⌊0.20·n_in,t⌋ (the 20 % Wanda ladder entry's count). Layers ℓ are the decoder
blocks holding supported tensors (24 on this base, cf. the 24 `mlp.gate_proj` of F8a.2); N_ℓ = supported entries in block ℓ,
capacity c_ℓ = ⌊0.5·N_ℓ⌋, and each tensor's capacity ⌊0.5·n_t⌋. Draws m = 1…24 are taken in order from one
`Generator(PCG64(domain_seed("20261111:f9p:", "dirichlet")))`: π_m ~ Dirichlet(1, …, 1). Block budgets: targets K·π_ℓ; any block
over capacity is set to capacity and the remainder re-split in proportion to π over the others, repeated until none exceeds;
integers by largest remainder (ties to the lower ℓ) within capacities, so Σ_ℓ K_ℓ = K exactly. Within a block, K_ℓ goes to tensors
in proportion to n_t by the same water-filling and largest-remainder rule (ties in A5.8.3 tensor order); within a tensor, each row
gets ⌊K_t/out⌋ and the first (K_t mod out) rows by index one more; within a row the lowest Wanda scores are deleted, ties by
ascending column index. (g) *Contractions and ranking statistic.* Native entries are contracted on the complete D, so q includes the
deletion–compensation cross terms; a(−W⊙M), a(C) (which sum to a(D)) and q_cross = q(D) − q(−W⊙M) − q(C) are reported descriptively;
the native family is reported separately from transplant entries. For the ranking panel, Δy_m = y(W₀ + D_m) − y(W₀) on WikiText
`extra.mean_nll`, and the three predictors are oriented so that larger means more predicted damage: P1 = a_m + q_m; P2 = q_A,m =
½tr(D_m A D_mᵀ) (G = I; a reconstruction-objective comparator, not the task loss); P3 = ‖W⊙M_m‖²_F = −⟨D_m, W⟩ (F9.3's ⟨D, W⟩ with
its sign reversed). Statistic: Kendall τ-b between Δy and each predictor over the 24 masks, zero fitted parameters. Uncertainty:
paired block bootstrap over the 32 two-window blocks with common draws across the 24 masks and the base reference, 4,000 draws,
percentile, seed `domain_seed("20261111:f9p:", "tau")`, for each τ-b and for τ(P1) − τ(P2) and τ(P1) − τ(P3). The interval is
conditional on the 24 fixed masks and 64 windows; it is not mask-population or cross-family calibration. Comparisons are within
sparsity only; ladder and native entries never enter τ and sparsities are never pooled. (h) *Component dependence.* F9-P has three
source components: Wanda (Wanda ladder entries, their rotations and all 24 ranking masks, which share scores), SparseGPT (transplant
entries, their rotations and the native entries, which share solves) and random (one generator). `candidates.json` records this
graph with F8a.2's typed edges and records that it supports no binomial inference (F8a.5); F9-P is reported as entries with
per-entry intervals, entries only under F9a.3(f). (i) *Funding, stops and order.* Construction ≤ 0.5 GPU-h (F9.3) for any GPU use;
CPU work is charged 0 GPU-h and its wall time and peak memory are logged in `construction_cost.json`. The first solve (ρ = 0.10) is
the bounded measurement: if the projected cost of the remaining solves exceeds the remaining construction cap, construction stops
and every SparseGPT entry is `missing: unfunded (cap)`. Scoring allocation ≤ 0.82227 GPU-h = (45 × 1.04 min + 47.1 s cache load +
1.751 min cold base reference)/60, the overhead-inclusive projection at the reference rates with no retry margin; retries, new loads
and references are charged inside it; F8a.8's stops, watchdog, recovery and repricing apply. Purchase order: ladder by ρ ascending,
within ρ (Wanda, Wanda rotation, transplant, transplant rotation, random, random rotation); then native by ρ; then ranking masks in
draw order. Entries not reached are `unscored: stopped`; τ is computed only on the complete 24-mask panel. (j) *No compression
claim.* The panel concerns unstructured zeros; it supports no claim of equal-performance compression, reduced memory or dense
allocation, latency, or a smaller parameter inventory, and no cap is borrowed. The 10/20/50 % rungs are chair choices (F9a.8).

**F9a.5 — H5 reconciliation (condition 5).** (a) *Documents.* The governing H5 texts are `docs/H5_DESIGN.md` "11 Nov 2026 (council
round 5) amendment — localisation flag, shadow LoRA sub-stream, re-anchoring" and `docs/H5_PROTOCOL.md` "11 Nov 2026 (council
round 5) amendment — paired localisation episodes, shadow LoRA stream, re-anchoring" (both added by `2ab8fab`, after Board A reviewed
`f181f0a`), `docs/HYPOTHESIS.md` "11 Nov 2026 (council round 5) prospective note — H5 and H1-OR", and the round-4 11 Nov sections
(F8.7), each read as amended by F8a.6 and by F9a.5–F9a.6. Those two clauses are appended verbatim with F8a.6 to both H5 documents,
and HYPOTHESIS is reconciled, before the H5 pilot. Superseded there: "with a missed-harm upper bound ≤ 5 % (A5.7)" (replaced by
F9a.6), "run by a pinned script with no model calls" (F8a.6(a)) and "measured about 43 s per row" (F9a.5(e)). (b) *Access and
training meters,* pinned per episode in the G5 manifest before the pilot: `python_tool_access`, `forward_access`,
`hidden_state_access`, `feedback_access`, `field_localisation_access`, `backward_enabled`, `gradient_exposed_to_editor`. F9.4
episodes: `backward_enabled = 0`; `gradient_exposed_to_editor = 1` exactly when `field_localisation_access = 1` (g⊙W and g⊙D derive
from the frozen G2 gradient; no fresh backward pass). `lora_shadow`: python 1, forward 1, hidden-state 0, feedback 1 (F9.5), where
feedback means the next-token training loss on the query context only (no TF-v1, held-out or sealed signal reaches training);
`backward_enabled = 1`; `gradient_exposed_to_editor = 0` (no LLM editor; only the harness optimiser consumes gradients). This
permission covers only the harness-run training of `lora_shadow`: no LLM editor arm gains backward passes, and no other arm's
registered access changes (A5.7's gradient-based comparators keep theirs). Meters per arm: forward requests and tokens, backward
passes and tokens, optimiser steps, GPU-seconds and wall time. Optimiser, learning rate, step, rank and target-module limits are
pinned in the G5 manifest before the first adapter is trained, chosen without any shadow outcome; F9a sets no values. (c)
*Localisation sequencing and controls.* Candidate-independent inputs (per-tensor g⊙W, cached down-projection input norms) may be
released before the first candidate; candidate-specific g⊙D exists only after that candidate's realised delta and is returned as
metered feedback on it, never guiding its own construction. The verbal site ranking is requested and logged with its hash before any
field ranking is released (harness-enforced) and is never a critic input. The shuffled control permutes tensor labels over the same
ranking positions, one permutation per episode drawn before the episode from `PCG64(domain_seed("20261111:f9-shuffle:", <episode
id>))`, at matched length, support, opportunities and cost. (d) *Paired causal contrast.* The only localisation-value contrast is
the scripted estimator on the LLM's logged layers (variant 2) against the scripted estimator with field localisation (variant 1):
same editor, localisation alone differs. LLM with versus without the field ranking and real versus shuffled ranking are reported
descriptively and support no claim. Endpoint: GPE_x = Σ_e gain_{x,e} / Σ_e E_{x,e}, with gain the change in A5.7's bounded held-out
score (percentage points) of the final incumbent over the starting incumbent and E all real evaluator calls including failed calls
and audits (A5.7); Δ_loc = GPE₂ − GPE₁ with cluster influence u_k = (g₂,k − GPE₂·E₂,k)/ΣE₂ − (g₁,k − GPE₁·E₁,k)/ΣE₁ in A5.7's paired
episode-cluster bootstrap-t (4,000 common draws, PCG64 seed 20261107, its failure rules). A zero or nonfinite denominator makes
Δ_loc undefined, never a success. Localisation value is credited only if the two-sided 95 % lower bound of Δ_loc is > 0; a point
improvement alone is insufficient; there is one primary contrast and no multiplicity correction. Aggregate sensitivity is not causal
localisation or faithful self-report, and comparisons at unequal access stay labelled (F8a.6(a)). (e) *Increments, gradients,
re-anchoring.* Candidates stay stored base-relative (F5.4). A forecast for an acceptance decision is relative to the actual
incumbent: the realised increment W_cand − W_inc (bf16 states, fp32 difference) contracted with an exact incumbent gradient
(`gradient_anchor = incumbent-exact`, fresh gradient hash) and base factors (`factor_anchor = base-factor`), or, labelled
`approx:difference_of_base_forecasts`, f_base(W_cand − W₀) − f_base(W_inc − W₀). Re-anchoring does not turn a submitted base patch
into an incumbent increment, and an exact gradient contraction is not an exact finite-edit forecast. All re-anchoring work (gradient
recomputation, loads, failures) is metered and charged to the critic arm's net GPU (A5.7). The historical timings are not future
rates: 854.718742 s over 20 rows, mean 42.736 s, with different predecessor supports and including load and save/hash work
(`docs/reviews/council/board-a-g2-2026-11-11.md`), and 13305.825513 s over 141 donors, 94.3676 s each including I/O (power
`fields_manifest.json` summary, per Board A); per-objective re-anchoring and factored-LoRA costs are unmeasured, measured in the
pilot and repriced under F8a.6(f). (f) *LoRA scope and groups.* The shadow forecasts completed adapters' collateral TF-v1 change
only, not on-query gain or whether training is worthwhile; a TF surrogate cannot supply A5.7's bounded task-score endpoint.
Query/training contexts, support, calibration, acceptance and sealed groups are disjoint by item id and source document; query
contexts are disjoint from every TF-v1 scored item, the G2 gradient panels (including the 11 WikiText articles and the 500 reserved
groups, A5.3.2(iii)) and any sealed panel; overlap fails closed. (g) *Contract preserved.* All of A5.7 (arms, primary comparator,
ratio-of-totals savings, noninferiority, guardrails, the missed-proposal guardrail with its all-rejected denominator and all-zero
unresolved case, net GPU, prospective joint power ≥ 0.80, unresolved versus kill), the F5.4 critic contract, F8.7's per-stream
reporting and F8a.6's retained arms and matched-information rule stand. F8a.6(f)'s drop order gains, first, `lora_shadow` as one
unit, then the F9.4 paired set, then spec §6.6's order. G5 stays shadow-only.

**F9a.6 — shadow limits (condition 6).** (a) *Harm, defined separately.* Adapter j is harmful if its WikiText contrast Δy_j = y(W₀ +
D_j) − y(W₀) under `configs/eval_tf_v1.yaml` is resolved (A5.3 paired block interval excludes 0) and Δy_j > 0; other TF-v1 endpoints
are descriptive; a missing or nonfinite score is `unresolved`. This is not A5.7's missed qualifying beneficial proposal, which a
collateral forecast cannot estimate; A5.7's guardrail, denominator and unresolved cases stay unchanged. (b) *Gate frozen before
shadow outcomes* (in the G5 manifest, before the first adapter is trained): evaluate adapter j if its zero-fit forecast a_j + q_j
(WikiText field, `base-exact`, `base-factor`) exceeds h\*, otherwise skip; a field abstention is evaluated. One gate, never tuned on
the 32 outcomes. (c) *All 32 scored.* Every adapter is trained and scored on TF-v1 whatever the gate says; the gate is applied
counterfactually. (d) *Read-outs.* Missed harm M_h = #(skipped ∧ harmful)/#skipped (undefined if nothing is skipped) with its exact
one-sided 95 % Clopper–Pearson upper bound, and #(skipped ∧ harmful)/#harmful. The report states that even 32 independent skips with
zero misses give an upper bound of 1 − 0.05^(1/32) = 0.08937 > 0.05, and fewer skips or dependence make it worse. Replay saving
= #skipped/32 verification evaluations, descriptive, with training cost charged in both arms; E₀ and sign agreement follow F9a.2. (e)
*Limits.* No ≤ 5 % certification, no realised-savings claim and no G8 gate or condition comes from this pilot. The shadow may report
a gate as uneconomic, descriptively, when measured forecast cost per adapter is at least the measured verification cost per skipped
adapter. Training, audits and failures stay in both arms' cost accounting; A5.7's unresolved cases are retained.

**F9a.7 — generation qualification and purchase (condition 7).** (a) *Checkpoints.* The Astra edit only if F8a.4 qualifies its
replay, and any F8-E entry with a published generation claim whose bytes qualify; their number E is frozen before the pilot. (b)
*Frozen before any sampling,* in a committed generation manifest: model and tokenizer revision
`2fc06364715b967f1860aea9cf38778875588b17`; chat-template hash; thinking flag per arm (unedited thinking-on, edited thinking-on,
unedited thinking-off); decoding T 0.6, top-p 0.95, top-k 20, presence penalty 1.5 (the P2 settings in `docs/RESULTS.md`);
`max_new_tokens` 8,192; EOS token set; the repository's MATH-500 `exact_match` parser and scorer, pinned by code hash; engine, batch
size and device H0. (c) *Definitions.* Natural completion: generation ends with an EOS-set token before 8,192 generated tokens and,
in thinking-on arms, after the thinking block is closed. Strict correctness: the pinned parser extracts a final answer and the
pinned scorer marks it equal to the reference. Truncated, malformed and unparsed outputs are incorrect; the denominator is all 500
items × seeds; completion and correctness are reported together with their 2×2 per arm; completion is not correctness. (d) *Seeds
and pairing.* Per item and seed index, the seed is `domain_seed("20261111:f9-gen:", "<batch>:<seed index>:<item id>")`, shared by
all arms (paired); batch 1 and batch 2 seed indices are disjoint; the pilot uses its own stream and is never pooled. Paired
differences are per-item means over seeds; intervals by item bootstrap, 4,000 draws, PCG64 seed 20261107, percentile, with common
draws across arms so shared unedited-arm errors stay paired. (e) *Qualification rule.* Same form as A2.1 and A3's same-device check:
replay agreement (fraction of replayed generations token-identical to the originals), between-batch reliability of per-item rates,
and paired-difference half-width against a declared margin δ_gen per endpoint. The numeric limits and δ_gen are written in the
generation manifest before the pilot and confirmed by Board A; F9a sets no values. An unqualified endpoint is reported descriptively
and has no H5 use. (f) *Counts and ceiling.* The screen is qualification batch 1 (seed indices 1–5), reused, adding no generations;
batch 2 is seed indices 6–10; the replay regenerates seed index 1 for all 500 items in all arms on the same device. F9.6's 122.88 M
= 500 × 10 × 3 × 8,192 is the upper bound (every generation at the cap) for the **two-batch qualification panel of one edited
checkpoint with its two unedited arms**; it excludes the replay, the pilot and further checkpoints, and replaces F9.6's
"qualification ceiling" wording. With the unedited arms shared across the E edited checkpoints, the complete qualification upper
bound including replay is 500 × 11 × (2 + E) × 8,192 = 45.056 M × (2 + E) generated tokens (135.168 M at E = 1). (g) *Public items.*
MATH-500 items are public and already exposed (`docs/RESULTS.md` P2); they are never fresh H5 held-out, calibration or sealed
evidence, and every H5 panel excludes them. (h) *Funding.* Only a timed same-device pilot is funded: items in a frozen order
(`domain_seed("20261111:f9-gen:", "pilot-order")`), all arms interleaved per item, pilot seed stream, stopped at ≤ 0.5 GPU-h or
after 500 items, whichever comes first; it records tokens and wall time per generation. Complete repricing is a dated amendment
computed from pilot timing logs only; pilot outcomes change no frozen rule; no qualification batch runs before that amendment. The
endpoint has no funding or decision authority, never alters F5.5 and is not an H5 endpoint until qualified and adopted by amendment.

**F9a.8 — evidence wording (condition 8).** For every future use (reports, paper, H5 documents) the following replace the
corresponding text of F9, SYNTHESIS-r5 and the RESEARCH_NOTES round-5 section, which are not rewritten. (1) *PPF (arXiv
2508.02381).* "A predictor of structured (channel or layer) pruning-policy performance, measured as JS divergence to the original
model, on an evaluated dataset of 775 collected samples with 20 % used for testing" (excerpt in `paper/refs/verified.yaml`; Board
A's v2 check); "trained on 775 evaluated masks" (synthesis), "from 775 evaluated masks" (F9.8) and "a surrogate trained on evaluated
masks" (paper sentence 1) read so; its prediction error is not a perplexity error for our masks. (2) *IDM (arXiv 2610.13991)* was
submitted 26 Sep 2026, not 15 Oct (RESEARCH_NOTES); its CMSE is a centred residual MSE; our swap loss is F9a.3(c). (3) *P5 labels*
as F9a.2(i). (4) *Finite-edit exactness.* "The field is exact" for isolated adapters, deletion masks or root edits (synthesis
verdict, paper sentence 3, RESEARCH_NOTES item 6) reads: "for a root edit of the pinned base the gradient term a = ⟨g, D⟩ is the
exact first-order contraction at the base, with no predecessor displacement; neither a nor a + q is an exact forecast of a finite
edit's effect." (5) *Timing* as F9a.1(d) and F9a.5(e). (6) *Mathematics.* "Every small-model math gain so far is a termination or
robustness repair" reads: "the small-model math gains reviewed in round 5 include termination and robustness repairs; the council
did not establish that all are." (7) *Introspection.* Paper sentence 2 reads: "In one study of eight open-weight models (arXiv
2608.20569), verbal reports did not discriminate interventions from shams beyond chance, while linear probes on the same activations
recovered them; the same source reports a model fine-tuned for this report reaching near-perfect recovery, so no universal claim is
made." (8) *Wanda.* Paper sentence 4's "Wanda's score is the diagonal of our curvature term" reads: "the squared Wanda score is, up
to a constant, the diagonal term of q_A (the G = I curvature term), not of q." (9) *Rungs.* 10 and 20 % are where arXiv 2604.25098
reports unstructured Magnitude and Wanda pruning retaining performance on 7–8B models (Board A's v3 check); "50 where OBC-Prune
reports collapse" is attributed-only (the archived excerpt reports improvements at 40 and 50 %); all three rungs are chair choices.
(10) *Premises.* Board A verified PPF's count, arXiv 2611.03244's 82 % and 66 % and arXiv 2604.25098's 10/20 % design. Every other
numerical premise is either archived-source (text retained in `paper/refs/sources/` and excerpted in `paper/refs/verified.yaml` by
the chair, not checked by Board A) or attributed-only (no retained primary text); none is a threshold, expected effect, funding
premise or success premise:

| Premise | Where used | Status |
|---|---|---|
| OWMI: AUROC ≈ 0.5007 over 78,000 measurements; probes 75–95.8 % | synthesis Q4/sentence 2, RESEARCH_NOTES 11 | archived-source (`2608.20569.txt`) |
| PPF: 1.52 s versus 1 min 38 s per policy | RESEARCH_NOTES 20 | archived-source (`2508.02381.txt`) |
| OBC-Prune "collapse" at 50 % | synthesis Q3 | attributed-only; archived excerpt says the opposite at 40–50 % |
| Das replication: 10/30 versus 2/30 completion; 16/30 versus 9/30; McNemar p = 0.021 raw, 0.107 Holm | RESEARCH_NOTES 4, paper sentence 5 | attributed-only (dev.to blog; no retained copy) |
| Pruning laws: collapse past about 70–80 % | RESEARCH_NOTES 18 | attributed-only |
| aiXiv ≈ 93 % recovery | F8.9, H5 documents | attributed-only; conditional on supplied layers, budget and scale |
| Agentic-TTT and QueST gains | RESEARCH_NOTES 6–7 | archived-source; qualitative only |

The chair choices (30/100, 24 masks, 32 queries, 10/20/50 %, 8,192 tokens, five-seed batches and every cap) are planned criteria,
not measurements.

**Rules in this block that are the author's, not Board A's (for Board A to confirm).** (1) No magnitude gate for F9-P; f_surv ≥ 0.75
reused from F8a.2 as its survival precondition (F9a.1(e), F9a.4(e)). (2) Allocations: F9-P scoring ≤ 0.82227 GPU-h and F9.5 scoring
≤ 0.59693 GPU-h as overhead-inclusive projections at F8a.8's reference rates with no retry margin; the F9.6 pilot cap ≤ 0.5 GPU-h,
reusing F9's construction/training cap figure (F9a.1(e), F9a.4(i), F9a.7(h)). (3) Schema `f9l-v1`, the three row kinds and the split
of anchor into `gradient_anchor`/`factor_anchor` (F9a.2(d)–(e)). (4) Component-weighted E₀ as primary, row-weighted beside it; the
per-population representative hierarchy; the "any invalid draw ⇒ undefined" rule (F9a.2(f), (h)). (5) The update protocol: one-use
swap-validation populations retired into training, swap loss = component-weighted MSE, population order F8-E, F8-F, F9-P, G5, F9.5
(F9a.3(c)). (6) Per-row count rule k(ρ) = ⌊ρ·n_in⌋ for all F9-P sources; Wanda from √A_jj of the frozen factor with the
`truncated_diagonal` label; SparseGPT as a one-shot solve on the frozen factor with A5.8.1's damping, full-row mask selection and no
blocking (F9a.4(d)). (7) Dirichlet(1, …, 1) layer ratios with 50 % block and tensor capacities, water-filling and largest-remainder
integerisation; row allocation (F9a.4(f)). (8) Damage orientation P3 = ‖W⊙M‖²_F, Kendall τ-b, window-block bootstrap for τ, τ only
on the complete panel; purchase order (F9a.4(g), (i)). (9) Three F9-P components (F9a.4(h)). (10) Access values
`gradient_exposed_to_editor = 1` with field localisation and 0 in `lora_shadow`; `lora_shadow` first in the drop order (F9a.5(b),
(g)). (11) The localisation contrast variant 2 versus variant 1 with GPE as ratio of totals, its influence function and lower
bound > 0 (F9a.5(d)). (12) Harm defined on resolved positive WikiText change; the a + q > h\* shadow gate with abstentions evaluated; the
"uneconomic gate" descriptive criterion (F9a.6). (13) Generation: P2 decoding settings, completion/correctness definitions, seed
derivation, screen = batch 1, replay = seed index 1 for all items and arms, shared unedited arms across checkpoints,
qualification-rule form with values left to the manifest (F9a.7). (14) Wording item (8) on Wanda and q_A, and the premise status
table (F9a.8). (15) The domain-seed prefixes `20261111:f9p:`, `20261111:f9-shuffle:` and `20261111:f9-gen:`.

| Board A condition | F9a clause |
|---|---|
| 1 Authority/provenance: F9.7 and all decisions/pins/caps retained; reports archived with identities; logging-versus-panel order; F8a exposure/freeze/configuration overrides; F9's own gate/purchase population; decision-maker attestations | preamble; F9a.1 (a)–(f) |
| 2 Executable forecasts/statistics: realised scale, reference/rotation conventions, κ notation, anchor/version/schema fields, immutable pre-decision commitment, undefined/degenerate/missing values, MSE decomposition, weighting, representatives, uncertainty, denominators; zero-fit primaries, chart abstention, q magnitude-only; P5 widths not a floor | F9a.2 (a)–(j) |
| 3 Leakage control: frozen source-connected splits, family/episode order, update protocol, predictor versions, eligible development rows only; prohibitions; 30/100 as reporting conventions | F9a.3 (a)–(f) |
| 4 Pruning execution/funding: 45 entries, Dirichlet/count/solve/coverage/rounding recipes, τ/tie/orientation rules, native compensation, within-sparsity ranking, component dependence; overhead-inclusive bounded costs, stop/reprice; no compression claim | F9a.4 (a)–(j) |
| 5 H5 reconciliation: H5 documents with F8a overrides; access/training meters; localisation sequencing; paired causal contrast; matched-information controls; base-relative versus incumbent increments; exact-gradient provenance; charged re-anchoring; full A5.7/F5.4/F8.7 contract | F9a.5 (a)–(g) |
| 6 Shadow limits: missed harm separate from missed beneficial proposals; gate frozen before outcomes; all 32 scored; no ≤ 5 % certification, realised savings or G8 gate; costs in both arms; A5.7 unresolved cases | F9a.6 (a)–(e) |
| 7 Generation qualification/purchase: completion/correctness and qualification rules, independent seeds and pairing, public-item exclusion; token-ceiling scope, screen reuse, replay and checkpoint counts; bounded pilot before repricing; no F5.5 or automatic H5 effect | F9a.7 (a)–(h) |
| 8 Evidence wording: PPF count, P5 labels, finite-edit exactness, timing qualifications, universal math/introspection claims removed; unverified premises labelled and unused as premises | F9a.8 (1)–(10); F9a.2(i); F9a.5(e) |

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

*A5.1.2 re-pin (xv) (10 Nov 2026, 23:55 UTC; merge of `feat/part-a-prepare` into main after F7–F9a; before any G3 outcome).* The
staging branch's re-pin, written as "(xiv)" above, was built on a tree without F7's `scripts/a5_simulation.py` change; main's (xiv)
(F7) was built without the staging CLI. After merging, `scripts/a5_build_manifests.py --pin-worktree` was re-run into a copy of the
package on the merged tree and only `code_hashes.json` and the package README were copied back. Rows: `docs/prereg/a5_manifests/code_hashes.json`
`016de4c822c9e2f6517bc820e0f3aa8c143c08ea05c0874b1f844775310c4671` (old main `90a28ef7…`, old branch `d4b2e0ab…`);
`docs/prereg/a5_manifests/README.md` `461905e4e744645501c7ddff01638a334b80738a203f7474b671cf7676d49a16` (old main `f51a290a…`, old
branch `5c770dbe…`); `scripts/a5_simulation.py` unchanged from F7 (`04f283d8…`); `src/wsp/eval/response_intervention.py` as pinned by
the staging branch. No selection, seed, control, panel or decision file changed. Nothing has been scored.

#### A5 amendment F10 (11 Nov 2026, before any G3 outcome): rotation exclusion rule, realised diagnostics, chain-ladder convention

This block is appended; earlier text is not rewritten. It records chair rulings F10 (R1: remedy (d) of the design note),
the chain-ladder convention (R2) and the scoring hold (R3). They were taken after the Part A staging
(`docs/prereg/a5_manifests/part_a/`, staging code `7f8b453`, merged into main at `008f0af` with A5.1.2 re-pin (xv); audit
file sha256 `f05fec4265ffd93c8c537100993d23262be95141369240fcad0a7e16ff474d35`; staging README sha256
`2fcbf517ccf26f2a1fe325ef31300d291bc0f254c7faaacd18154e327bceada1`) and before any G3, rotation, ladder, reference or sealed
outcome existed. The analysis is `docs/design/ROTATION_YIELD_2026-11-11.md` (added by commit `ccf1e7b`, sha256
`fad409ccb1c3059fc4b8385252c1e5f883148571a7521f590e350948f0e5719b`; "the note"); F10.1–F10.8 adopt its §3.2 draft with the
edits listed in F10.12. The trigger is the staged audit yield (1 of 40 rotation pairs), a function of stored bytes and staging
metadata only. F10 changes no selection (A5.3.1, `selection.json`), seed (`control_seeds.json`, A5.3.3 seed mapping), prediction,
resolution, yield or decision rule (A5.3, A5.5), simulation gate (A5.3.2), cap or allocation (A5.3.4), and no pin: no file pinned
in A5.1.2 changes with this block. It changes only the rotation-pair exclusion rule (F10.3), what is reported beside it (F10.4,
F10.5), when a masked simulation must run (F10.6), and the reading of A5.3's ladder direction for eight chain picks (F10.9). It
takes effect for execution only after Board A accepts it (F10.10). If Board A rejects it, the registered reading applies
(remedy (e) of the note): the A5.3/A5.3.3 audit stands, the rotation contrast is unresolved by yield (1 < 15), and the ladders
and references are scored as staged (100 entries, 104.00 GPU-min, staging README); F10.9 and F10.10 still apply.

**F10.1 — Facts (the note §§1.1–1.4; staging README).** Under the A5.3/A5.3.3 realised audit, 39 of 40 pairs are excluded
(2,534 tensors, 535 pass; lora_sft 30/30 and chain 9/10 excluded; the only pass is `00f8111cbad8218f`). Per tensor, the realised
discrepancy ε_R = ‖R_eff − R‖/‖D‖ is the bf16 rounding of the predecessor W at the rotation's scale: it never exceeds
min(1, 1/(√12κ)) by more than 0.24 %, with κ = ‖D_block‖/√(Σ_block ulp(W)²). On the 2,188 tensors of the 32 root edits it
equals the edit's own rounding ε_D = ‖D − Δ‖/‖D‖ (median ratio 1.0003, 5–95 % 0.996–1.021). Realised norm and spectrum
mismatches are bounded by ε_R (Mirsky; holds in 2,534/2,534 rows). The 1e-3 rank counts bf16 noise: rank-16/32 LoRA edits have
D rank 437–1,024. The registered realised tolerances therefore exclude by effective norm and predecessor grid, not by rotation
quality. Every drawn rotation is exact to fp32 before rounding (staged maxima: orthogonality 8.7e-7, pre-rounding norm mismatch
1.43e-7; the five fp32 reconstruction residuals of 1.05e-5–1.40e-5 are 1.4e-15–4.0e-15 in fp64).

**F10.2 — Construction unchanged.** R = Q_out D_block Q_inᵀ (`rotation.rotate_block`), with the registered per-edit seeds, one
generator per edit whose initial state is cloned for each tensor, the QR sign convention, fp32 construction and rounding at the
actual predecessor (`rotation.apply_and_round`: fp32 W + R, then bf16) (A5.3, A5.3.3). The scored object bf16(W + R) − W, the
prediction ⟨g, D_eff − R_eff⟩, the "spectrum-matched (pre-rounding)" wording, the selection, the 40 pairs and their 10/10/10 and
4/3/3 tertile allocation, the seeds, the resolution rule, the yield rule (≥ 15 resolved), A5.5 and the ladders are unchanged.

**F10.3 — Exclusion audit.** It replaces the exclusion role of A5.3's post-rounding audit, A5.3.3's "all tensors must meet S31's
norm/spectrum/rank/support criteria", and F3(ii)'s per-tensor pass rule. Each edited tensor passes if and only if (i)–(iv) hold,
and the edit passes (v). A failure of any tensor or of (v) excludes the pair with its reason, without replacement (A5.3.1); the
pair stays in the forty-ID ledger. A missing audit fails closed (F3(ii)).
(i) *Construction*, computed in fp64 from the stored bf16 states and the fp32 construction (Q_out, Q_in and R as `rotate_block`
produces them): orthogonality error ‖QᵀQ − I‖_F/√n ≤ 1e-5 for each Q (as `_rotation_diagnostics` computes it); full-SVD
reconstruction residual of the realised block ≤ 1e-5 relative Frobenius; |‖R‖ − ‖D‖|/‖D‖ ≤ 1e-5; ‖σ(R) − σ(D)‖/‖σ(D)‖ ≤ 1e-5
(sorted singular values); and the numerical rank of R (1e-3 rule, A5.3.3, at R's own σ₁) equals that of D, except that
|rank R − rank D| may not exceed the number of singular values of D within 1e-5·σ₁(D) of the cut 1e-3·σ₁(D).
(ii) *Support*: D and R_eff are zero outside the changed block (unchanged rule; actual nonzero masks are reported, A5.3.3).
(iii) *Replay*: after the variant is written, R is recomputed from `torch.Generator().manual_seed(seed)` with the pinned code,
the pinned environment and the staged thread settings (`OMP_NUM_THREADS` = `MKL_NUM_THREADS` = 4), rounded with
`apply_and_round`, and the result must equal the R_eff bytes reloaded from the written `delta.safetensors` (roots) or
`increment.safetensors` (chain steps) bitwise. The written delta must reproduce the intended bf16 state under the scorer's
arithmetic bf16(base + delta) (the existing staging check).
(iv) *Rounding bound*, entrywise in fp64, with R the fp32 construction: |R_eff − R| ≤ min(|R|, ½·ulp(W + R_eff)) + ulp₃₂(W + R).
ulp is the bf16 spacing at the bf16 value (2^(e−8) for w = m·2^e, m ∈ [½, 1); `scripts/a5_rotation_yield_check.py:ulp_bf16`), and
ulp₃₂ the fp32 spacing (2^(e−24)) at the fp32 sum. Without the last term this is the round-to-nearest guarantee. The fp32 term
covers the registered two-step arithmetic (fp32 sum, then bf16), whose double rounding can exceed the single-rounding bound
by less than one fp32 spacing. A failure means a wrong grid, predecessor, dtype or arithmetic.
(v) *Sub-resolution guard*, per edit: the pooled realised norm mismatch |‖R_eff‖ − ‖D‖|/‖D‖, with norms pooled over all edited
tensors (`_edit_audit` `pooled_norm_mismatch`), is ≤ 0.5, so the rotation keeps at least half the edit's norm after rounding.
It is fixed a priori; the staged maximum is 0.0309 (`61d1f61e25212068`).
Existing pre-audit reasons (`non_2d`, `non_finite`, `zero_block`, `rank_degenerate`, `not_single_matrix`) are unchanged. New
reason codes: `construction_failed`, `support_failed`, `replay_failed`, `rounding_bound_failed`, `sub_resolution`. The registered
realised verdict (A5.3 tolerances) is kept as a logged field, never as a reason code.

**F10.4 — Realised diagnostics (reported, not exclusion criteria).** Per tensor and pooled per edit: the registered realised norm
mismatch (signed), spectrum mismatch and ranks at 1e-3 of D and R_eff; the ranks above max(1e-3·σ₁, 3τ) of D and R_eff, with
τ = √(mean_block ulp(W)²/12)·(√m + √n) for an m × n block; ε_R, ε̂ = min(1, 1/(√12κ)), κ, ε_D (roots) and the energy ratio
ρ = ‖R_eff‖²/‖D‖² (pooled: Σ_t‖R_eff,t‖²/Σ_t‖D_t‖²); the A5.3.3 items (‖R_eff − R‖/‖D‖, reconstruction residual, exact support,
unrotated-tensor count) and the logged moments as staged. "Spectrum-matched" refers to the pre-rounding construction. The
realised rotation equals the exact rotation up to bf16 rounding of W, of the same size as the edit's own rounding. No realised
spectrum match is claimed.

**F10.5 — Sensitivities (descriptive; the decision uses F10.3's set).** (a) The A5.3.3 nuisance-controlled sensitivity adds
log ρ (pooled per edit) as a covariate, with train-only standardisation and penalty 1 as registered. (b) Agreement and its CP
interval are reported by norm tertile. (c) They are also reported on the 30 pairs whose staged pooled realised norm mismatch
(`manifest.json:edit_audits`) is ≤ 2e-2: all except `0a841d4082532e8b`, `0bc0a0c4ad0520e0`, `0d055869527177b4`,
`1ab850a41051fe45`, `1fc51f7164052d50`, `2d1de332e36d8f00`, `453d8c08966438e0`, `455b881d5de4e6e0`, `61d1f61e25212068`,
`649bd035e79488e5` (all low tertile). (d) The registered constant-sign comparator is shown next to each. (e) They are also
reported on the 34 pairs that pass the noise-calibrated realised audit (norm ≤ 2e-2 + ε̂, spectrum ≤ 5e-2 + ε̂, rank above
max(1e-3·σ₁, 3τ) within ±1; the note §2.1 remedy (b)): all except `00f8111cbad8218f`, `0a841d4082532e8b`, `0bc0a0c4ad0520e0`,
`1ab850a41051fe45`, `61d1f61e25212068`, `649bd035e79488e5`. None of these is decisive or can change A5.5's outcome; each subset
uses the ≥ 15 floor only to say whether its descriptive interval is reported as resolved.

**F10.6 — Simulations.** If all 40 pairs pass F10.3, the actual audit state equals the audit assumption of the registered
NULL-SIZE and POWER-AND-YIELD runs (F2, F7; ordered power 0.8540). They stand without a rerun and remain conditional on the F2
proxy (Board A F7 conditions 2 and 4). This is not a conversion of simulated yield into audit evidence (condition 4): the
evidence is the F10 re-stage audit, and the frozen simulation payload is not changed. If any pair is excluded, then before
scoring, POWER-AND-YIELD (with its NULL-SIZE pass) is rerun as a separately labelled run with those selected slots masked in
every replicate, on the registered 61-donor pool, seeds and stream order (F7). The ≥ 0.80 ordered-power and ≤ 0.07 size gates
apply to that run (A5.3.2). It needs a `--slot-mask` option in `scripts/a5_simulation.py` with tests and a re-pin. The F7 runs
took 21 min 39 s and 22 min 27 s of CPU wall (the note §2.4). A failure makes Part A not decisive and needs a further amendment
before scoring.

**F10.7 — Execution.** The F10.3 audit and the F10.4 diagnostics are implemented in `src/wsp/eval/response_intervention.py` and
`src/wsp/features/rotation.py`, with tests, and pinned by A5.1.2 re-pin (xvi) before the re-stage. The note's draft says
"(xv)"; that number is taken by the `008f0af` merge re-pin. The rotation variants of all 40 edits are rebuilt under that code,
in a rotation-only mode. The R_eff bytes (`delta.safetensors` and, for chain steps, `increment.safetensors`) of the staged
passing pair `00f8111cbad8218f` must be bitwise identical to the staged hashes in `manifest.json:delta_sha256`. The 80 ladder
variants and the reuse proofs are re-hashed against the staged manifest, not rebuilt. The Part A manifest, audit file and README
are regenerated as a re-stage report, and their hashes enter A5.1.2.

**F10.8 — Exposure.** F10 is pre-outcome only if no decision-maker who drafts or approves it has been exposed to any G3,
rotation, ladder or reference score, or to a summary of one (F8a.1(c) definition, including messages, documents and log lines).
Part A scoring, including the 80 ladders and the 19 missing originals, starts only after Board A has ruled on F10 (F10.10).

**F10.9 — Chain-ladder convention (chair ruling R2).** A5.3 builds scaled edits "by applying s·Δ_t at W_t and rounding, not by
scaling D_t". For the 8 chain picks whose predecessor is not the base (`0027efdf2eae7405`, `00f8111cbad8218f`,
`05531ce1db4df63f`, `057dcfbedea3251f`, `0bc0a0c4ad0520e0`, `0d055869527177b4`, `13b02fe2673cc296`, `14cdaabee246493b`), no fp32
per-step Δ_t exists; only cumulative bf16 snapshots are stored (staging README, "Flag (chain ladders)"). The realised increment
D_t = bf16-state_t − bf16-state_{t−1} (fp32 subtraction) is Δ_t up to one bf16 rounding of W_t. The 16 ladder variants built as
bf16(W_t + s·D_t) − W_t for these picks (s = 0.25, 0.5; `scale_source: realised_increment` in `variant.json`) are accepted as the
registered A5.3 ladder, with the realised increment substituted for Δ_t. They are flagged `realised-increment ladder` in the
regenerated manifest and in the Part A report. The substitution is of rounding scale (same analysis as F10.1): D_t − Δ_t is one
bf16 rounding of W_t, whose uniform-rounding size relative to ‖D_t‖ is the pooled noise floor ratio √(Σ ulp²/12)/‖D_t‖, 0.066–0.392
for these eight edits (the note §1.4, column "pooled ε̂"; it overpredicts below κ ≈ 1, the note §4). On the 32 roots, where Δ is
stored, the measured ε_D tracks this size. Scaling D_t also scales its rounding noise, which the second bf16 rounding then adds
to, as for the rotation. The 30 lora_sft picks and the 2 root chain picks (`014f8a9ec78db2be`, `0559051354691a70`) scale their
stored Δ as registered; for the two root chain picks the stored Δ is already the realised increment (ε_D = 0), and they are not
flagged. The prediction is unchanged: ⟨g, bf16(W_t + s·D_t) − W_t⟩, the realised bf16 subtraction and never s·D_t itself (Board A
G2 item 2). Resolution, denominators (40 per strength), the step-2 alternative rule and A5.5 are unchanged.

**F10.10 — Order of operations and scoring hold (chair ruling R3; the note §§3.3–3.5).** No Part A entry (ladder, rotation or
reference) is scored before Board A has accepted F10, or has rejected it so that the registered reading (e) applies. Scoring the
ladders or the 19 missing originals first would expose the response size per edit, which correlates with every candidate audit
rule, and the step-2 ladder outcome, which overrides a rotation go (the note §3.5; F8a.1(c)). The order is:
1. F10 appended (this block) with the attestation in F10.11.
2. Board A reviews F10: accept, or reject (then (e): no re-stage; go to step 7 with 100 entries, F10.9 flags in the Part A report).
3. Code, tests and re-pin (xvi) (F10.7); `--slot-mask` only if step 5 needs it.
4. Re-stage the 40 rotations on CPU under `agent-tmux`, 4 threads, at the priced cost: about 1 h at the measured rate, upper
   bound 4 h 58 min 33 s (the full staging wall time); +35.8 GB under `data/runs/a5/part_a/` (404 GB free) (the note §3.4).
5. Re-stage report: F10.3 yield per pair with reasons, F10.4 diagnostics, the `00f8111cbad8218f` bitwise check, re-hashed
   ladders, the `realised-increment ladder` flags, and the updated priced count (40 − excluded + 80 + 19 entries × 1.04 min).
6. If fewer than 40 pairs pass: the slot-masked POWER-AND-YIELD run of F10.6 (about 22 min CPU), which must reach ≥ 0.80
   ordered power before scoring.
7. Score all entries in one campaign on the exclusive GPU through `agent-tmux`: 40 rotations + 80 ladders + 19 originals =
   139 entries, 144.56 GPU-min at 40/40 (A5.3.4), with staged bytes re-hashed first.
8. Close and seal the ledger, then the single analysis (A5.3 "Analysis time").

**F10.11 — Pre-outcome attestation.** Nothing has been scored. The staging manifest records `scored: false`; the scorer has
not been run; none of the 120 staged or excluded variant ids and none of the 19 missing originals has a directory under
`data/evals/qwen3.5-0.8b/tf_v1/` (checked by path existence only, 11 Nov 2026; nothing was opened). The staging was merged at
`008f0af` without scoring. The only persons and sessions that have seen the staging audit are the staging builder (README:
"No outcome exists"), the session that wrote the note (status: no outcome scored, opened or read), the chair (R1–R3 were taken
on the note and the staging README), and the session that wrote this block. None of them has seen any G3, rotation, ladder,
reference or sealed outcome or a summary of one, because none exists. This session read only `docs/`, `src/` and
`scripts/`, wrote nothing under `data/`, and ran CPU-only synthetic checks of the F10.3(i) and (iv) arithmetic (F10.12). Board A
adds its own attestation line on review (F8a.1(c)).

**F10.12 — Edits to the note's §3.2 draft, for implementability.** (a) F10.3(i): "the numerical rank of R equals that of D" now
admits rank differences caused only by singular values within 1e-5·σ₁(D) of the 1e-3 cut. With exact equality, fp32 construction
noise, far inside the 1e-5 spectrum tolerance, could still flip a singular value that straddles the cut. That would exclude
pairs by arithmetic, as the five fp32 reconstruction artefacts did. High-κ edits have dense spectra at the cut (D rank
437–1,024 against LoRA rank 16/32). The pre-rounding spectrum and rank were not computed at staging, so the window is set now,
at the chair's construction tolerance of 1e-5. (b) F10.3(iii): the replay
environment is fixed: pinned code and environment, staged thread settings, fresh seeded generator, comparison with the
reloaded written bytes. (c) F10.3(iv): "+ ulp₃₂(W + R)" is added. `apply_and_round` rounds twice (fp32 sum, then bf16). When the
fp32 sum lands exactly on a bf16 midpoint, ties-to-even can round away from the bf16 value nearest W + R. The error then exceeds
min(|R|, ½·ulp) by less than one fp32 spacing (|bf16(y) − y| ≤ min(|W − y|, ½·ulp) and |y − (W + R)| ≤ ½·ulp₃₂ for y the fp32
sum). Such ties occur at a small but positive rate per entry, and an edit has up to 72 tensors with blocks up to 1024 × 3584
(staging README; the note §1.3). Without the term, correct code would fail essentially every pair. A synthetic CPU check by this
session confirmed violations without the term and none with it (not committed; no data read). (d) Header, date and provenance: the staging is cited at its merge `008f0af`. (e) F10.7: the
re-pin is (xvi), not (xv). (f) The note's staged maximum pooled mismatch "0.031" is given as 0.0309. (g) F10.6 names the 61-donor
pool and Board A condition 4. (h) F10.9–F10.11 are new (R2, R3, attestation).

| clause | rule changed | yield | simulation effect |
|---|---|---|---|
| F10.1 facts | none | registered audit on staged rows: 1/40 pairs, 535/2,534 tensors | none |
| F10.2 construction | none (R, seeds, rounding, prediction, selection, yield rule, A5.5, ladders) | – | none |
| F10.3 exclusion audit | A5.3 post-rounding exclusion, A5.3.3 "all tensors must meet S31's criteria" and F3(ii) pass rule → fp64 construction, support, replay, rounding bound, sub-resolution guard | 40/40 on the staged rows for the measured items (orthogonality ≤ 8.7e-7, pre-rounding norm ≤ 1.43e-7, support 2,534/2,534, fp64 reconstruction ≤ 4.0e-15, pooled guard ≤ 0.0309 against 0.5; the note §2.1); pre-rounding spectrum and rank, replay and rounding bound are first computed at the re-stage | none if 40/40 (equals the F2/F7 audit assumption); otherwise F10.6 |
| F10.4 realised diagnostics | realised norm, spectrum and rank move from exclusion to reporting; adds κ, ε̂, ε_R, ε_D, ρ, noise-referenced rank | none (not exclusion criteria) | none |
| F10.5 sensitivities | adds log ρ covariate to the registered nuisance sensitivity; adds tertile and two subset reports | descriptive subsets: 30 pairs (pooled norm ≤ 2e-2), 34 pairs (noise-calibrated audit) | none; no decision authority |
| F10.6 simulations | adds a slot-masked POWER-AND-YIELD rerun (with NULL-SIZE pass) whenever fewer than 40 pairs pass | – | none at 40/40 (registered 0.8540 stands); otherwise one labelled rerun, about 22 min CPU, gates ≥ 0.80 and ≤ 0.07 |
| F10.7 execution | rotation-only re-stage; re-pin (xvi); bitwise check of `00f8111cbad8218f` | re-stage yield reported per pair | none |
| F10.8 exposure | none (applies F8a.1(c)) | – | none |
| F10.9 chain-ladder convention | A5.3 "s·Δ_t" read as s·D_t for the 8 non-root chain picks, flagged `realised-increment ladder` | ladders unchanged: 40 per strength, 16 variants flagged | none (simulated ladder contrasts already scale the realised-edit residual block, A5.3.2) |
| F10.10 order and hold | no Part A scoring before Board A rules on F10 | – | masked run, if needed, precedes scoring |
| Fallback (e) on rejection | none | 1/40 (rotation unresolved) | none; the go branch is unreachable |

#### A5 note F10a (11 Nov 2026, before any G3 outcome): methodological clarifications for F10 (Board A condition 1)

This block is appended; earlier text is not rewritten. It discharges the methodological-record part of condition 1 of Board A's F10
review (`docs/reviews/council/board-a-f10-2026-11-11.md`, commit `e9c6db7`, sha256
`f1c494ab608c464d29f9dc261dee8711e553ab0fee90a136581b0efb432142b3`; "the review"). The quotations in F10a.1 and F10a.2 are
copied verbatim from the review's section "Three implementability edits" (F10.3(i)) and section "Simulations and chain ladders"
(F10.9). F10a clarifies how F10 is read and executed; it changes no F10 operational rule except as stated in F10a.1, where it
adopts the review's own formula.

**F10a.1 — F10.3(i) rank criterion: a near-cut rank budget (Board A wording adopted verbatim).**
> Rank is discontinuous: exact equality is an inappropriate requirement for a fixed fp32 rotation even when fp64 audits find tiny
> norm/spectrum errors. The explicit rule is `abs(rank_R − rank_D) <= N_near`, where
> `N_near = count(abs(σ_j(D) − 0.001 σ_1(D)) <= 0.00001 σ_1(D))`; each rank uses its own largest singular value and strict `>` at
> its cut. Keep this rule and the independent fp64 norm/spectrum/reconstruction/orthogonality checks fixed before re-stage yields
> are seen. A count budget is not proof that individual crossings were caused only by near-cut values; describe it as a budget, and
> log the ranks, cuts and N_near. The Frobenius spectrum criterion alone is not an entrywise `1e-5 σ_1` perturbation theorem. Do
> not widen the window or add ±1 after seeing failures.

Reading. The last clause of F10.3(i) ("except that |rank R − rank D| may not exceed the number of singular values of D within
1e-5·σ₁(D) of the cut 1e-3·σ₁(D)") is this rule, with "within" inclusive (`<=`). rank_D counts σ_j(D) > 1e-3·σ₁(D) and rank_R counts
σ_j(R) > 1e-3·σ₁(R), both in fp64 (D from the stored bf16 states, R the fp32 construction before rounding). F10.12(a)'s phrase "rank differences caused only by
singular values within 1e-5·σ₁(D) of the 1e-3 cut" is read as a count budget, not as a causal claim about individual crossings: a
tensor that meets the budget is not thereby shown to have changed rank only through near-cut values, and the re-stage report says
"rank budget", not "rank match". Every tensor logs rank_D, rank_R, both cuts (1e-3·σ₁(D), 1e-3·σ₁(R)) and N_near. The window
(1e-5·σ₁(D)) and the budget are fixed; there is no ±1 allowance. The executable tests the review lists for this rule (equality at
the cut and window edges, both rank directions, own-σ₁ thresholds, N_near = 0, rejection above the budget) belong to condition 2.

**F10a.2 — F10.9 chain-ladder convention: uncertainty of the realised-increment ladder (Board A wording adopted verbatim).**
> **F10.9 is accepted as an explicit operational amendment, not literal compliance with the original “not by scaling D_t”
> clause.** […]
>
> For these eight non-root picks, raw Δ is unavailable: the root-edit rounding comparison does not measure their missing raw-step
> error. Calling D−Δ “one bf16 rounding” requires an identified underlying additive raw update and construction history; the quoted
> uniform-rounding floor is a heuristic scale, not a measured error or upper bound for those eight. Qualify F10.9 accordingly.
> Accepting this available direction changes the operational ladder intervention, potentially relevant to step 2; it does not
> establish equivalence of the unavailable raw ladder.

Reading. F10.9's statements that "D_t = … is Δ_t up to one bf16 rounding of W_t", that "the substitution is of rounding scale",
and that "D_t − Δ_t is one bf16 rounding of W_t, whose uniform-rounding size relative to ‖D_t‖ is … 0.066–0.392" are qualified as
follows: for the 8 non-root chain picks (`0027efdf2eae7405`, `00f8111cbad8218f`, `05531ce1db4df63f`, `057dcfbedea3251f`,
`0bc0a0c4ad0520e0`, `0d055869527177b4`, `13b02fe2673cc296`, `14cdaabee246493b`) the raw per-step Δ_t was not stored, so
D_t − Δ_t was not measured and is not bounded; "one bf16 rounding" holds only under the unverified assumption of an identified
additive raw update with a known construction history; 0.066–0.392 is a heuristic uniform-rounding scale, not a measured error or an
upper bound; and the root-edit comparison (ε_D on the 32 roots) does not measure the error of these eight. The 16 ladder variants
built as bf16(W_t + s·D_t) − W_t are the operational A5.3 ladder for these picks by explicit amendment, not a demonstrated
equivalent of the raw-Δ ladder that A5.3 describes. Their `realised-increment ladder` flag and this qualification are carried into
the re-stage report and the Part A report (conditions 4 and 6), and any step-2 statement that relies on them says so. F10.9's
operational rule is unchanged: the 16 variants as staged, the 2 root chain picks and the 30 lora_sft picks scaling their stored Δ,
the prediction ⟨g, bf16(W_t + s·D_t) − W_t⟩ on the realised increment, 40 slots per strength, the step-2 rule and A5.5. No
raw-step reconstruction or retraining is authorised.

**F10a.3 — Status of F10's thresholds: fixed before outcomes, not before audits.** The F10 parameters were fixed after the Part A
staging audit and before any outcome. Staging ran 10 Nov 2026 18:46–23:45 UTC (audit file sha256 `f05fec42…474d35`; README
`2fcbf517…ceada1`); the yield analysis (design note `fad409cc…f0e5719b`) recomputed audit quantities on all 2,534 staged rows and
the remedy yields between about 23:50 UTC 10 Nov and 01:01 UTC 11 Nov; F10 was committed at 01:12 UTC 11 Nov (`0af5a66`). The
parameters concerned are: the 1e-5 construction tolerances of F10.3(i) (orthogonality, fp64 reconstruction residual, pre-rounding
norm and spectrum) and its 1e-5·σ₁(D) near-cut window (F10a.1); the one-fp32-spacing slack ulp₃₂(W + R) of F10.3(iv); the 0.5
pooled half-norm guard of F10.3(v); and the replay settings of F10.3(iii) (fresh `manual_seed` generator per tensor, pinned code and
environment, `OMP_NUM_THREADS` = `MKL_NUM_THREADS` = 4, bitwise comparison with reloaded bytes). Their authors had seen the
staged audit rows, the 1/40 registered yield, the staged maxima F10.1 and F10.3(v) quote (orthogonality 8.7e-7, pre-rounding norm
1.43e-7, pooled mismatch 0.0309) and the remedy yields of the note §2. They are therefore audit-yield-informed: **fixed before
outcomes, not before audits.** No G3, rotation, ladder, missing-reference or sealed outcome, or a summary of one, existed when they
were fixed (attestations, F10a.5). The 39/40 registered failure, the remedy-yield comparisons and the original realised verdict
stay disclosed and logged (F10.3, last sentence; the review, "Pre-outcome legitimacy and attestation").

**F10a.4 — No retuning on the re-stage yield.** No F10 parameter may be changed after any re-stage verdict, yield or diagnostic has
been seen. This covers every parameter listed in F10a.3, the reason codes, the rank rule of F10a.1, the F10.4
diagnostic definitions, the F10.5 covariate, tertiles and subset lists, and the F10.6 masking rule. A tensor or pair that fails a
fixed tolerance is excluded with its reason code (F10.3), without replacement, and stays in the forty-ID ledger. Any other response
to a failure is a new, dated, disclosed amendment that states the yield and diagnostics already seen, is reviewed by Board A
before any scoring, and does not replace the fixed rule retroactively. A failure is never handled by widening a window, adding a
±1 rank allowance, changing the slack or the guard, choosing other seeds or thread settings, or re-running until a pair passes.

**F10a.5 — Exposure attestations.** One no-exposure attestation per decision-maker (F8a.1(c): role, session identity, coverage
time) is recorded in `docs/prereg/f10_attestations.md`: the chair; the staging builder (`feat/part-a-prepare`); the rotation-yield
analyst (`design/rotation-yield`); the F10 drafter (`prereg/f10`); the F10 code builder (`feat/f10-rotation-audit`, pending its
report); Board A (`review/board-a-f10`, quoted from the review); and the session that wrote this block (`prereg/f10a`). The file
also records the path check made for this block: none of the 120 Part A variant ids and none of the 19 missing originals has an
entry under `data/evals/qwen3.5-0.8b/tf_v1/`, and none of the 120 variant ids appears anywhere under `data/evals/`. The 19
missing originals do appear by name one level up, as symlinks `data/evals/qwen3.5-0.8b/<id>` created 6 Nov 2026 08:10 UTC into
their zoo entries, which hold pre-TF-v1 records written at zoo generation (file names listed only; contents not opened). These are
not TF-v1 or Part A records and were not produced by Part A; the file records them so that each attestation covers them. A
decision-maker who makes or approves a further F10 choice before scoring adds its own attestation there (the review, condition 7).
The code builder's attestation must be completed from its report before the implementation merge (condition 1).

**F10a.6 — Scope.** F10a changes no selection (A5.3.1, `selection.json`), seed (`control_seeds.json`, A5.3.3), prediction,
resolution, yield or decision rule (A5.3, A5.5), simulation gate (A5.3.2, F10.6), cap or allocation (A5.3.4), and no pin: no file
pinned in A5.1.2 changes with this block. It does not release scoring: F10.10 and conditions 2–7 of the review still hold every
Part A entry, including the 80 ladders and the 19 missing originals.

*A5.1.2 re-pin (xvi) (11 Nov 2026, about 02:30 UTC; merge of `feat/a5-slot-mask`, Board A F10 condition 5 implementation; before any
G3 outcome).* `scripts/a5_simulation.py` gained `--slot-mask <json>` (contrast-specific slot masking for the F10.6 slot-masked
NULL-SIZE / POWER-AND-YIELD run; the unmasked run is byte-identical without the option, regression-tested; 18 tests pass). Re-run
`scripts/a5_build_manifests.py --pin-worktree` into a copy; only `code_hashes.json` and the package README copied back. Rows:
`scripts/a5_simulation.py` `c4f0c55fd8e8c9bc5bb6521539f58db996f23e1a3cb37d99a19403e134d2fdf2` (old `04f283d8…`, F7);
`docs/prereg/a5_manifests/code_hashes.json` `67b1517bd61c8ec41f83f688d319a336673a218c9b3c07d266e5d96351cf59e2` (old `016de4c8…`, xv);
`docs/prereg/a5_manifests/README.md` `12f54157dedd179687880d527fff15b18fe903226b3db5ab2769a0cc4cdd8a74` (old `461905e4…`, xv). No
selection, seed, control, panel or decision file changed; no slot-masked run has been executed; nothing has been scored. A further
re-pin follows the F10 exclusion-rule code merge (Board A F10 condition 3).

#### A5 note F10b (11 Nov 2026, before any G3 outcome): adoption of the F10.5 sensitivity and interpretation specification

This block is appended; earlier text is not rewritten. It is the dated A5 note that Board A's review of F10a and the F10.5
specification (`docs/reviews/council/board-a-f10a-2026-11-11.md`, commit `778e250`, sha256
`0971c06f0a29fa50b0f745e1e63be9abe6978b623d8b63c89e677e0c4413d1a9`; "the review", part B, conditions B1 and B2) and the
specification's own header (`docs/design/F10_SENSITIVITY_SPEC.md`: "takes effect only when a dated A5 note adopts it by path and
sha256 before any Part A entry is scored") require. It discharges B1 in full and states how B2 is carried; it changes no statistic.

**F10b.1 — Adoption, by path, hash and commit.** `docs/design/F10_SENSITIVITY_SPEC.md` (453 lines) with
**sha256 `8624e06343bbb401ff3b6d783572b0c53e1172d0ea3f36aec3768a13701998d4`** is adopted as the frozen specification of every
F10.5 report (F10.5(a)–(e) and the interpretation rules Board A attached in condition 6 of its F10 review). Provenance: the
specification was committed as `e577c67` ("design: F10.5 sensitivity and interpretation specification (Board A F10 condition 6)")
and reached `main` through merge `2ee94c7` ("merge design/f10-sensitivity"). Chronology, stated so it cannot be misread: the
second parent of `2ee94c7`, and the commit the review fast-forwarded to and quotes, is the sibling commit `bc15691`; `e577c67` and
`bc15691` share parent `754b7eb`, carry the same subject, and `git show` of the specification path gives the identical sha256 above
at both. The only difference between the two commits is a root `README.md` hunk (2 insertions, 2 deletions) that is not part of the
specification. The adopted object is the file bytes with the sha256 above, which is what the review hashed; either commit id
identifies those bytes. Any later edit to the specification file requires a new dated A5 note adopting its new exact hash before
scoring (the review, B2). Report implementations must carry this hash and fail closed on a mismatch.

*Historical registration hash.* The specification's source table (§0) lists the registration `docs/PREREG_H1_TF.md` with sha256
`be3dcab22b745a04c20b3cd2c433222f1372dc223cd6f35f7dfcd0b44c58111a`. That is the registration as of commit `0af5a66` (the F10
block), before F10a. It is a historical hash and is not evidence of tampering. It is not the registration in force: F10a (commit
`9fdd8bb`) was appended after it, and the file at the parent of this commit has sha256
`898c3587706c3d9e14a7f1e9356eb9aadca0b57a367330872357c0a8c1ac9e59`. The specification's reports are read together with F10a
(F10a.1 near-cut rank budget; F10a.2 chain-ladder reading; F10a.3 and F10a.4 audit-yield-informed thresholds and the ban on
retuning), and F10a's qualifications are carried forward into every F10.5 report. The other source-table hashes in §0 match the
bytes the review reviewed.

**F10b.2 — Operational choices confirmed (the specification's §7.1 points 1–3, and the review's B1).** Board A accepts these
prospective choices as specified; they are not permission to choose columns, models or cuts after outcomes.

1. *"v4 differences".* The words "logged v4 differences" / "rotated-minus-original v4 differences" (A5.3, A5.3.3) are read as
   the five logged `_moments` fields `mean`, `std` (unbiased), `frac_pos`, `frac_nonzero` and `cos_to_predecessor`, as
   rotated-minus-original differences (`logged_r − logged_d`), averaged per edit over its edited tensors **weighted by element
   count** (the product of the full tensor shape), exactly as specification §1.2. "Directional tokens" are not logged by the pinned
   code and are **excluded**. The block is five columns, fixed by name now; no other reading (for example v4 StatNN pool features)
   may be substituted after outcomes.
2. *"Penalty 1".* λ = 1 on the **summed** log loss, objective Σ_i[−a_i ln p_i − (1−a_i) ln(1−p_i)] + ½·λ·‖β‖², with the
   **intercept unpenalised**; equivalent to `sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, fit_intercept=True)`
   (specification §1.3). Train-only standardisation of all eight columns in every full, leave-one-out and bootstrap fit; fixed
   folds and common bootstrap draws, as specified. No other λ, solver setting, column, interaction or transformation is tried.
3. *Two tertile sets.* (i) The **registered norm tertiles**, the A5.3.1 selection cells (`selection.json`; `manifest.json:selections[].tertile`),
   reported per family cell and pooled, exactly as F10.5(b) states (specification §2.1). (ii) **Additionally**, ln ρ tertiles
   (specification §2.2), whose **cut points and member ids are frozen from the staged manifest** (cut 1 = (0.01010406252666643,
   `057dcfbedea3251f`); cut 2 = (0.03502604499592402, `22659cefc904e29f`); rank cells 13 / 13 / 14). Re-stage ln ρ values are printed
   beside the frozen ids, and **any pair whose re-stage value would cross a cut is flagged `cell_drift`, never moved**. The ln ρ
   tertiles carry no greater authority than the registered norm tertiles, and no tertile report can reach the ≥ 15 floor; each is
   labelled `below floor (descriptive)`.
4. *Subset id-list hashes.* Under the repository's sorted-json convention (sha256 of `json.dumps(sorted(ids))` encoded as UTF-8,
   default separators; `scripts/a5_build_panels.py:sorted_sha`): **S30** (F10.5(c); staged pooled realised norm mismatch ≤ 2e-2;
   30 pairs) = `3da5f44f0c10307c2af7336c23fad430d13f6eb9347976e34de5d0810277267c`; **S34** (F10.5(e); noise-calibrated realised
   audit; 34 pairs) = `fb1c34e68d91c03b67b90ae1c0e867eaad32edadebf2ec4b615ce13515aedfa0`. Both hashes were recomputed from the
   JSON id lists in specification §3.1 and §3.2 for this note and match. The lists are frozen as ids; subset membership is never
   recomputed from outcomes or re-stage values; excluded, missing or unresolved pairs are disclosed and never replaced, and no
   subset becomes another primary denominator. Board A recomputed S30 from the manifest and accepted S34 as a frozen id list, not
   as an independently reproduced audit; this note claims no more. S30 ∩ S34 has 29 pairs, so the subsets are not independent checks.

**F10b.3 — Board A's two reporting corrections (the review, B2).** Neither changes a statistic or grants decision authority.

1. *Predicted-sign table is 3 × 2.* Board A: "§4.2's stated categories require a **3 × 2** sign table (including zero
   predictions), despite its '2 × 2' label". The specification states the categories as sign(p_i) ∈ {+, −, 0} by
   sign(d̂_i) ∈ {+, −}. Every report row that §4.2 requires therefore gives a **3 × 2** table: rows predicted positive / predicted
   negative / predicted zero (abstain) by columns resolved d̂ positive / d̂ negative. Field agreement is read off the same cells
   (agree = predicted + with d̂ +, or predicted − with d̂ −; the zero row is disagreement by A5.3), so each report also prints the
   resolved agree / disagree split of every predicted-sign row beside the counts, and zero-prediction pairs are counted separately
   inside the disagreements as §2.3 requires. Specification §4.2's "2 × 2" label is superseded by "3 × 2" for this purpose.
2. *Units.* The endpoint, d̂_i and the first-order forecast p_i of §6.2 (and every other statement of them in the specification)
   are in **nats per valid next token**, consistent with specification §0, and not unqualified total nats. Every report column
   and the final report label them so.

**F10b.4 — Status of the reports.** The F10.5 reports must be **implemented, tested and pinned in A5.1.2 before any Part A entry is
scored** (specification §7.1 point 4; the review, B2): CPU only; synthetic-input tests covering train-only transforms, one-class
training sets, degenerate columns, missing covariates, failed draws, CP boundary cases and the subset-hash check; code and the
specification hash pinned like any other Part A analysis code. They are executed **once**, in the single final report, with the
verbatim limitations paragraph of specification §5, the §4.3 non-superiority sentence and all ledgers (specification §6). They have
**no A5.5 or F5.5 authority**: no covariate model, tertile row, subset row or comparator can create, remove, rescue or reweight a
decision under A5.5, F5.5, F5.7 or the registered claim, and none authorises another primary denominator.

**F10b.5 — Scope and attestation.** F10b changes no selection (A5.3.1, `selection.json`), seed (`control_seeds.json`, A5.3.3),
prediction, resolution, yield or decision rule (A5.3, A5.5), simulation gate, cap or allocation (A5.3.4), and no pin: no file pinned in
A5.1.2 changes with this block. It does not release scoring: F10.10, conditions 2–7 of the original Board A F10 review and
condition A1 of the F10a review (completion of the code builder's attestation, entry 5 of the ledger, before the implementation merge)
all still hold every Part A entry, including the 80 ladders and the 19 missing originals. Exposure attestation of the drafter of this
note (role: pre-registration drafting sub-agent; session: worktree `prereg/f10b`, 11 Nov 2026, from its creation until the commit of
this block): it read only `docs/`, `src/`, `scripts/` text and committed metadata; it did not open anything under `data/evals`, did
not compute, score or receive any G3, rotation, ladder, missing-TF-v1-reference or sealed outcome or a summary of one, used no GPU and
wrote no data. The older 6 November D3 development records classified by A5.3 and by Board A's ruling are unaffected and not opened.
It attests only to its own session; the attestation ledger is not edited here and condition A1 remains open.

*A5.1.2 re-pin (xvii) (11 Nov 2026, about 04:50 UTC; merge of `feat/f10-sensitivity-reports` after Board A's acceptance with
conditions, `docs/reviews/council/board-a-part-a-analysis-2-2026-11-11.md`; before any G3 outcome).* The pin map in
`scripts/a5_build_manifests.py` gained four entries (`src/wsp/eval/part_a_analysis.py`, `scripts/part_a_analyse.py`,
`tests/test_part_a_analysis.py`, `docs/design/F10_SENSITIVITY_SPEC.md`); the builder was re-run with `--pin-worktree` into a copy and
only `code_hashes.json` and the package README were copied back. Rows: `src/wsp/eval/part_a_analysis.py`
`419dc07486279a4fe16cdefd90973042596ca12f65028e466172aa7d7d1d9bc9` (new); `scripts/part_a_analyse.py`
`a47a72f8b6485363b4c9ba8fc600b8564c31dfcb02e0c1a1106ffe734f07395d` (new); `tests/test_part_a_analysis.py`
`5adaeaf8ea66d8b64ef1e9cfd6964006eaea363ba58d1214d1d03befc6a48b4f` (new); `docs/design/F10_SENSITIVITY_SPEC.md`
`8624e06343bbb401ff3b6d783572b0c53e1172d0ea3f36aec3768a13701998d4` (new; equals the F10b hash); `scripts/a5_build_manifests.py`
`2ac41aab35df6838…` (changed: pin map only); `docs/prereg/a5_manifests/code_hashes.json`
`c8c3a048d5dee5444957e7dc9b11e79029edf1ea7bccf90e442a031de816e246` (old `67b1517b…`, xvi); `docs/prereg/a5_manifests/README.md`
`558aae097b19121926be96bd75602c61450e91e2a39cd27ce43485bfd4e9a55e` (old `12f54157…`, xvi). Input-inventory bindings for the analysis
(`text_groups.json`, the Part A manifest, the NULL-SIZE report) are recorded by hash in A5 note F10c. No selection, seed, control,
panel or decision file changed; nothing has been scored.

#### A5 note F10c (11 Nov 2026, before any G3 outcome): adoption of the Part A analysis implementation's operational choices

This block is appended; earlier text is not rewritten. It is the dated A5 note that Board A's re-review of the corrected Part A
analysis code (`docs/reviews/council/board-a-part-a-analysis-2-2026-11-11.md`, "the re-review"; verdict ACCEPT WITH CONDITIONS;
reviewed commit `2b03bda571bc46bce00b4b8aaf6b9c9c07424139`) requires in its condition 1: "Adopt the operational choices before
scoring. Append a dated A5 note explicitly adopting the three requested behaviours, sealed-ledger v2 and producer/evidence contract,
resolution stream, exact flat-step tie-break, `audit_missing`, whole-run subset-hash refusal, and validity/Part B/masked-report
interfaces. Carry the supplied-versus-computed Part B distinction and all existing F10b interpretation restrictions. No
retrospective choice from outcomes." It adopts the "Choices pending adopting note" listed in the module docstring of
`src/wsp/eval/part_a_analysis.py` and the items the builder listed in `docs/PROGRESS.md` (11 Nov 2026, 03:45 and 04:30 UTC), each
prospectively, in the numbered clauses below, with the re-review's disposition quoted where it ruled. The first review
(`docs/reviews/council/board-a-part-a-analysis-2026-11-11.md`, verdict REJECT, six corrections) is the earlier round; its
corrections are the ones the re-review found addressed. Everything below is read together with F10b, whose interpretation
restrictions (F10b.2–F10b.4: descriptive authority, no decision authority for any sensitivity output, 3 × 2 sign tables, units of
nats per valid next token) are carried unchanged.

**F10c.1 — Missing Part B leaves A5.5 step 1 unresolved.** Adopted. Board A: "A5.5 step 1 is unresolved. There is no Part A decision
without the mandatory Part B inference/control validity information. This tool retains supplied Part B; it does not fit it. Absence
is not eligibility and must not be represented as a computed Part B report." Consequently, when `--part-b` is not supplied, or the
supplied object has a false `valid` flag, a missing, nonfinite, reversed or malformed point or interval, failed or missing draws,
fewer than two distinct components or ineligible destruction controls, step 1 is unresolved and every later step (ladder
alternative, Part A evidence, Part B eligibility, funding) is unresolved with it. The supplied Part B object, its validity flags and
its evidence are printed in the JSON and Markdown on every branch, labelled as supplied and not computed by the tool. The
registered hard stop of G2 pending a new amendment is the stated consequence of a failed `g2_predecessor_audit`. No sensitivity
statistic enters validity.

**F10c.2 — A masked-calibration report supplied at 40/40 is refused.** Adopted. Board A: "Masked report at 40/40: refuse, rather
than silently ignore it. This prevents an inapplicable run from being mistaken for operative calibration. Below 40/40, require the
correctly bound masked report and preserve its gates and extra NULL-SIZE row." Operationally (`--masked-calibration`): at a
re-stage yield of 40/40 the registered F2/F7 simulations stand and a supplied masked report stops the run; below 40/40 a hashed,
separately labelled report from `scripts/a5_simulation.py --slot-mask` is mandatory, its rotation-mask ids and canonical mapping
sha256 must equal the re-stage exclusions, its hash must appear in the `null_size` validity evidence, its embedded NULL-SIZE row is
appended to the five frozen F5.7 rows without replacing one, and a failed calibration gate leaves step 1 unresolved (F10.6: "A
failure makes Part A not decisive"). The tool ingests and provenance-checks this report; it does not perform a calibration run.

**F10c.3 — Sealed-ledger schema `a5-part-a-sealed-ledger-v2`; fields and prediction rows.** Adopted. Board A: "Sealed ledger v2:
accept the exact ordered-window/block/token-hash and re-stage provenance contract described above. An outcome-free producer must
certify scoring identity and alignment; adoption does not make self-asserted hashes independent evidence." The schema is the one
documented in the module docstring: `schema`, `sealed`, `manifest_sha256`, `fields_sha256`, `text_groups_sha256` (the A5.1.2 digest
`f769afcade5a80571daf3a9578b1c03584b46a5abbbba07b32d43a7e9035e7e1`), `window_ids` equal to the string item ids `"0"` to `"63"` of
`text_groups.json` in registered order (ids 0–63 in order), `two_window_blocks` equal to its 32 adjacent two-window blocks
`[0,1]` … `[62,63]`, `window_token_ids_hashes` (64 per-window `token_ids_hash` values aligned with `window_ids`),
`restage_provenance` (hashes of the re-stage manifest, audit file and README), `references`, `variants` and `restage`
(per edit: `f10_3_verdict`, `pre_audit_exclusions`, `f10_3_exclusions`, `registered_realised_verdict`, nonempty `diagnostics`,
`variant_json_sha256`, `pooled_rho`, per-tensor moments). A re-stage entry that is absent, empty, partial or malformed, or a ledger
without complete `restage_provenance`, excludes the rotation pair as `audit_missing` (clause 6); a contradictory verdict or an
unknown reason code refuses the analysis; missing covariates drop a row from the complete-case covariate models only, with no
imputation. The fields file maps every selected edit id to three predictions in [rotation, ladder 0.25, ladder 0.5] order (A5.3.2),
optionally wrapped as `{"predictions": {...}}`; a missing row or a nonfinite or missing prediction fails the run, except an explicit
`null` prediction for an excluded variant. `mean_nll` is TF-v1 WikiText `extra.mean_nll` per window in nats per valid next token.
Adoption fixes the format, not the evidence: the ledger producer must be outcome-free and must certify scoring identity and array
alignment, and the trusted pre-score input inventory (clause 11) must bind the producer's inputs; a hash repeated inside the ledger
and the validity file does not by itself establish freezing.

**F10c.4 — Resolution stream: PCG64(20261107), one shared `[4000, 32]` index array.** Adopted. Board A: "Resolution stream: accept
PCG64(20261107) with one shared `[4000, 32]` index matrix across all analysed pairs/arms/contrasts, using paired differences and
linear percentile endpoints. `resolve` calls `sim.resolution` once on the stacked columns. The existing exact-degenerate fast path
returns identical endpoints without generating indices when every block column is constant; this is mathematically equivalent and
involves no alternative seed or retuning." The analysis draws one index array from `numpy.random.Generator(numpy.random.PCG64(
20261107))` and applies it to every pair, arm and contrast; 4,000 draws, 32 two-window blocks, linear .025/.975 percentiles; a
pair is resolved only if its interval strictly excludes zero. This is **not** the simulation's stream: `scripts/a5_simulation.py` derives its generators from `numpy.random.SeedSequence(20261107)
.spawn(3)`, which shares the integer 20261107 but not the generator state, so the analysis' index array is not bit-identical to
any index array the simulations drew. That is acceptable because the observed-data resolution is a single registered computation
whose only requirements are a fixed seed and draw count chosen before any outcome, common draws across pairs, arms and contrasts,
and the registered 32-block, 4,000-draw, linear-percentile procedure (`sim.resolution` itself is reused); nothing in A5.3, A5.5 or
the NULL-SIZE gates relies on the simulated and observed draws coinciding, and resampling the paired difference with common
indices is algebraically equivalent to common arm/reference draws for these linear contrasts. Board A did not require alignment
with the simulation's stream, so none is required or adopted; no alternative seed is permitted.

**F10c.5 — The λ = 1 solver's flat-step tie-break.** Adopted. Board A: "Flat-step solver tie-break: accept the existing numerical
refinement: normal objective decrease is accepted; a floating-point-flat step within `8 * eps * max(1, abs(f))` is accepted only if
maximum absolute gradient decreases. Fixed objective, lambda, tolerance, iteration limit and failure counting remain unchanged. The
objective comparison to sklearn and the new failure tests support this acceptance; this is not permission to tune a solver after
outcomes." The fit agrees with `sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0)` to 1e-6 on the tests' synthetic
problems. It is the F10b.2 item 2 model (summed log loss, λ = 1, unpenalised intercept, train-only population-SD standardisation in
every full, leave-one-out and bootstrap fit); nothing about it is tunable.

**F10c.6 — `audit_missing` as a fail-closed exclusion code.** Adopted. Board A: "`audit_missing`: accept as the fail-closed F10.3
rotation exclusion reason for absent/incomplete audit evidence. It remains visible in pair/subset ledgers and contributes to the
yield requiring masked calibration. It neither supplies missing covariates nor silently repairs an audit." It implements F10.3's
"A missing audit fails closed".

**F10c.7 — A wrong subset hash stops the run.** Adopted. Board A: "Wrong subset hash: stop the whole analysis before summarising
outcomes or writing reports. Frozen S30/S34 membership must not be replaced or silently omitted. The S30 mutation test deliberately
verifies refusal; the same guard covers S34. S34 remains acceptance of a frozen list, not independent reproduction of its historical
audit." The hashes are those of F10b.2 item 4 (S30 `3da5f44f…267c`, S34 `fb1c34e6…dfa0`), unchanged.

**F10c.8 — Part B contrasts and the seven A5.5 step-1 validity checks are supplied inputs.** Adopted, with the re-stated limit that
the tool retains and checks supplied Part B and never fits it (clause 1). Part B contrasts C, C_abl, C_rand and C_perm are supplied
by the Part B pipeline as precomputed conditional and joint intervals plus points, validity flags, per-procedure diagnostics (4,000
draws, failed draws, minimum distinct components ≥ 2), destruction-control eligibility and an evidence list, through `--part-b`.
The seven A5.5.1 step-1 predicates (`registration_pinned`, `source_closure`, `qualification`, `allocation`,
`g2_predecessor_audit`, `selection_and_audits`, `null_size`) are supplied through `--validity` as `{pass, evidence: [{path,
sha256}]}`; a predicate certifies step 1 only with `pass: true` and a non-empty, well-formed evidence list linking its required pins
(specification, staged manifest, text groups, fields and NULL-SIZE hashes for `registration_pinned`; the three re-stage provenance
hashes for `selection_and_audits`; the NULL-SIZE report and, below 40/40, the masked-calibration report hash for `null_size`).
Board A: "Missing Part B … must not be represented as a computed Part B report." The tool checks structure and hash links; it does
not re-derive the supplied evidence.

**F10c.9 — Runtime `--spec-sha256` check; staged correlations printed, not recomputed.** Adopted. Before reading any other input
the CLI hashes the specification bytes and requires them to equal the F10b.1 hash `8624e063…98d4` recorded in this registration,
the module constant and the required `--spec-sha256` argument; a mismatch, missing spec or missing flag stops the run with no
output. Board A: "`verify_spec` reads the file bytes and binds their SHA256 to F10b.1, the module constant and the required CLI
`--spec-sha256`." The staged ln ρ values and tertile cut keys are the frozen manifest values of F10b.2 item 3, printed as staged
and verified against the manifest hash; re-stage ln ρ values are printed beside them and are not used to move any id. The re-review
records no separate disposition for this point beyond its confirmation of the corrected report contents (its item 4); the
first review accepted that energy tertiles "verify the frozen cut keys and member IDs; re-stage drift is flagged without moving
IDs".

**F10c.10 — `cell_drift` for edit `057dcfbedea3251f` on ln ρ cut 1.** Adopted as already registered in F10b.2 item 3: edit
`057dcfbedea3251f` is the cut-1 member (staged ln ρ 0.01010406252666643); any downward re-stage drift that would carry it, or any
other pair, across a cut is flagged `cell_drift` and never moved. The flag has no decision authority.

**F10c.11 — Pin list and input inventory.** The analysis execution bytes to be pinned are, with the role and exact sha256 Board A
confirmed in the re-review ("Independently computed SHA256 values match all four builder prefixes") and which I recomputed for this
note:

| File | SHA256 | Role |
|---|---|---|
| `src/wsp/eval/part_a_analysis.py` | `419dc07486279a4fe16cdefd90973042596ca12f65028e466172aa7d7d1d9bc9` | Part A analysis implementation |
| `scripts/part_a_analyse.py` | `a47a72f8b6485363b4c9ba8fc600b8564c31dfcb02e0c1a1106ffe734f07395d` | Part A CLI |
| `tests/test_part_a_analysis.py` | `5adaeaf8ea66d8b64ef1e9cfd6964006eaea363ba58d1214d1d03befc6a48b4f` | Analysis and fail-closed tests |
| `docs/design/F10_SENSITIVITY_SPEC.md` | `8624e06343bbb401ff3b6d783572b0c53e1172d0ea3f36aec3768a13701998d4` | Adopted analysis specification (F10b.1) |

The existing pin of `scripts/a5_simulation.py` (`c4f0c55fd8e8c9bc5bb6521539f58db996f23e1a3cb37d99a19403e134d2fdf2`) is retained. The
four rows are **not** yet in `docs/prereg/a5_manifests/code_hashes.json`; adding them to it and to the builder's pin list, re-running
the targeted CPU suite and Ruff, and `scripts/a5_build_manifests.py --verify-pins HEAD` at the actual accepted execution commit are
the re-review's condition 2 and are done in the subsequent re-pin, not here. If any of the four files changes, the pins are
recomputed and a material change is reviewed again. Board A: "`text_groups.json`, the frozen Part A manifest and the NULL-SIZE
report belong in **A5.1.2's input inventory**, rather than being added as analysis-code rows to `code_hashes.json`." They are bound
there with their hashes, which I recomputed from the committed bytes and which match the re-review:

| Input | SHA256 |
|---|---|
| `docs/prereg/a5_manifests/text_groups.json` | `f769afcade5a80571daf3a9578b1c03584b46a5abbbba07b32d43a7e9035e7e1` |
| `docs/prereg/a5_manifests/part_a/manifest.json` | `d7785b35de1d7041edd2db139591562379db12e33c2532921c814735e8dde8d6` |
| `docs/prereg/a5_manifests/nullsize/report.json` | `1bed2c9531e4230864a1ebbe7e4023e0ffafcf704a59c40e93f9a75bd469dfa6` |

The accepted re-stage report, manifest and README, the per-edit variant metadata, the frozen prediction file, the scoring, panel and
configuration identity and any applicable mask or calibration inputs also require trusted pre-score inventory binding; keeping data
inputs outside the code-file map does not permit them to remain unfrozen.

**F10c.12 — Scope.** F10c changes no selection (A5.3.1, `selection.json`), seed (`control_seeds.json`, A5.3.3), prediction,
resolution, yield or decision rule (A5.3, A5.5), simulation gate, cap or allocation (A5.3.4), and no pin: no file pinned in A5.1.2
changes with this block. It authorises **no scoring**. Conditions 2 and 3 of the re-review (merge, re-test and pin the accepted
bytes; the other release conditions: attestations, outcome-free re-stage acceptance, applicable F10.6 calibration and every
remaining F10/F10a gate, including F10.10, conditions 2–7 of the original Board A F10 review and condition A1 of the F10a review)
remain open and hold every Part A entry, including the 80 ladders and the 19 missing originals. Nothing here is a choice made from
an outcome.

*Exposure attestation of the drafter of this note* (role: pre-registration drafting sub-agent; session: worktree `prereg/f10c`,
11 Nov 2026, from its creation until the commit of this block): it read only `docs/`, `src/`, `scripts/` and `tests/` text and
committed metadata (including the sha256 of the committed files listed above); it did not open anything under `data/evals`, did not
compute, score or receive any G3, rotation, ladder, missing-TF-v1-reference or sealed outcome or a summary of one, used no GPU and
wrote no data. It attests only to its own session; the attestation ledger is not edited here.

*A5 note F10c addendum (chair, 11 Nov 2026, about 05:10 UTC).* F10c clause 11 was written on a branch cut before re-pin (xvii) and says
the four analysis-code rows are not yet in `code_hashes.json`. They are: re-pin (xvii) (above) added `src/wsp/eval/part_a_analysis.py`
`419dc074…`, `scripts/part_a_analyse.py` `a47a72f8…`, `tests/test_part_a_analysis.py` `5adaeaf8…` and
`docs/design/F10_SENSITIVITY_SPEC.md` `8624e063…`, the same bytes F10c.11 cites. Nothing else in F10c changes.

*A5.1.2 re-pin (xviii) (11 Nov 2026, about 05:45 UTC; pre-execution freeze for the F10 rotation-only re-stage; before any G3 outcome).*
This re-pin supersedes the earlier F10 implementation pin references (re-pins (xvi) and (xvii) were used by the slot-mask and
analysis merges). Reviewed code: commit `060cc60` (`feat/f10-rotation-audit`), accepted with conditions by Board A
(`docs/reviews/council/board-a-f10-code-2-2026-11-11.md`); merged into main as `21646f2`; the execution SHA is the clean HEAD at
launch, recorded by the launch script in `docs/prereg/a5_manifests/part_a/restage_launch.json` and compared with the freeze. Pin map
extended by `scripts/a5_f10_dryrun.py`, `tests/test_rotation.py`, `tests/test_response_intervention.py`; the builder was re-run with
`--pin-worktree` into a copy and only `code_hashes.json` and the package README copied back. Rows: `src/wsp/features/rotation.py`
`8ac459934b0b145f62a3f0c81f68980c1f57d0ef7663d16e9ce1f693be376dba`; `src/wsp/eval/response_intervention.py`
`694f9f4eab769c731f269f33e1b2382902c505a8958dc0ad719981f00be10433`; `scripts/a5_f10_dryrun.py` `11bbc5f6…531ef`;
`tests/test_rotation.py` `7570224a…9325`; `tests/test_response_intervention.py` `c971a909…4e14`; `scripts/a5_build_manifests.py`
(pin map only); `docs/prereg/a5_manifests/code_hashes.json` `74321d76858274d2…` (old `c8c3a048…`, xvii);
`docs/prereg/a5_manifests/README.md` `692879c7cdada059…` (old `558aae09…`, xvii). Frozen execution: the Board A command verbatim
(rotation-only, `--audit-rule f10`, `--staged-manifest`, no `--evals`), `CUDA_VISIBLE_DEVICES=` (CPU only), `OMP_NUM_THREADS=4`,
`MKL_NUM_THREADS=4`, `HF_HUB_OFFLINE=1`, Torch intra-op threads 4 in construction and replay; environment identities (Python 3.12.3,
Torch 2.15.0+cu132, Torch build-config SHA-256, MKL available, CPU model, x86_64), selection/reference/control-seed/text-group hashes
and base revision are in `docs/prereg/a5_manifests/part_a/restage_freeze.json`. Preserved before execution: the a533 staging
manifest `d7785b35de1d7041…` as `part_a/manifest_a533.json` and its audits as `part_a/rotation_audits_a533.jsonl`; the 81 staged
delta hashes (80 ladders + the prior passing pair's rotation) stay in that manifest. Rulings carried: scorer round-trip failures are
`replay_failed` exclusions; the F10.3(v) pooled guard as registered; owning-check F10a reason codes with a `detail` field for missing
or non-finite information; constructor, seeds, selection and ladder bytes unchanged; pooled diagnostics as defined in F10.4. Scratch
evidence: the dry run under `data/runs/a5/f10_dryrun/condition2/` ran on an uncommitted tree (`src_dirty: true`) and is evidence of
behaviour only, not the execution pin. The inclusive equality at one fp32 spacing passes the registered rule; no F10c amendment is
needed (Board A). The regenerated manifest, audit and README hashes are entered after execution, in the re-stage report. Nothing has
been scored.

#### A5 amendment F11 (11 Nov 2026, before any G3 outcome): resizing by supplementary procedural selection after the F10.6 calibration failure

This block is appended; earlier text is not rewritten. It records the chair's F11 decision (`docs/PROGRESS.md`, entry "11 Nov 2026,
07:20 UTC. Chair decision on resizing (F11)": Option A of the council metrology report
`docs/council/2026-11-11/f11-metrology.md`, branch `council/f11-resize`, "the metrology report" below). It is the "resizing
amendment … required before G3" of A5.3.2 and the "further amendment before scoring" of F10.6 and Board A F10 condition 5
(`docs/reviews/council/board-a-f10-2026-11-11.md`). It takes effect for execution only after Board A accepts it. Until then, and
in every case until F11.4 passes, the scoring hold of F10.10, Board A F10 condition 5 and Board A F10-code condition 4
(`docs/reviews/council/board-a-f10-code-2-2026-11-11.md`) stays in place for every Part A entry, including ladders and references.
This commit adds the frozen supplementary list `docs/prereg/a5_manifests/selection_f11.json`, its builder
`scripts/a5_f11_supplement.py` and the tests `tests/test_a5_f11_supplement.py`. It changes no pinned file and makes no re-pin.

**F11.1 — Trigger and evidence.** All figures below are read from the cited files, and none of them is an outcome.
(a) *Yield.* The F10 rotation-only re-stage staged **31 of 40** rotation pairs (`docs/prereg/a5_manifests/part_a/manifest.json:coverage`:
`n_rotation_pairs_staged` 31, `n_rotation_pairs_excluded` 9; sha256 `59d839e351824fa154a665fcf105284ce5fdeed8423a5c0a2b7a60687b0d3b06`).
The 9 exclusions, by registered cell (`manifest.json:variants`, kind `rotation`, `excluded`; restated in
`selection_f11.json:registered_rotation_exclusions`), fall in three cells:
- LoRA-high (`lora_sft:2`), 3 of 10: `15a53586efc72a50`, `2233954b0cc3edc0`, `72d3756e968e3209`, each `rounding_bound_failed` on one tensor.
- chain-low (`chain:0`), 3 of 4: `057dcfbedea3251f`, `0bc0a0c4ad0520e0`, `0d055869527177b4`, `replay_failed`.
- chain-high (`chain:2`), 3 of 3: `0027efdf2eae7405`, `05531ce1db4df63f`, `14cdaabee246493b`, `replay_failed`.

LoRA-low, LoRA-middle and chain-middle lost none. All 80 ladder slots were kept.
(b) *Calibration.* The F10.6 slot-masked registered-scale run (10,000 replicates, 61-donor pool, mask
`docs/prereg/a5_manifests/part_a/slot_mask_f10.json` sha256 `86fa27a5c1838d0c0bd88ef530feba22eceafac698d0ec4506fbf05df83b698e`;
record `docs/prereg/a5_manifests/part_a/slotmask_calibration_record.json`; raw `data/runs/a5/power/power_slotmask_f10.json` sha256
`b14756828bc27cbbe14e1e590226820371a65ef5f83ad9622b2e6ccf04a3450a`) gave **ordered power 0.7514 < 0.80 (fail)**. Size passed:
ordered go 0.0192; two-sided 0.0377 / 0.0372 / 0.0372 (rotation, 0.25, 0.5). The null alternative rate was 0.1939, which is
reported and not size-gated (F5.7).
(c) *Sizing curve.* The chair's sizing runs are outcome-free synthetic masks that unmask subsets of the 9 excluded slots. They used
the same code, pool, seeds and replicates. They are labelled sizing in `docs/PROGRESS.md` (06:40 UTC entry), although their files
carry the F10.6 label string. They are not registered gates.
- `data/runs/a5/power/sizing_slots_34.json` (sha256 `e36a832b4361f34a1595bbe406d5962974a9f62269d6f1e6d6c17f643a4e4da6`): 34 available rotation slots, 0.7722 [0.7639, 0.7804], fail.
- `sizing_slots_37.json` (`ed039ab5495a6989520c62f9780725d37007bd4cd3bc65fa434ab6548e995adb`): 37 slots, 0.8251 [0.8175, 0.8325], pass.
- `sizing_slots_39.json` (`71de730fde8ea199d27d601af38e24cb3211c299343351dc46db84ad09dce484`): 39 slots, 0.8453 [0.8381, 0.8523], pass.

Null ordered go in those runs was 0.0156, 0.0180 and 0.0182. The anchors are 31 → 0.7514 (record above) and 40 → 0.8540
[0.8469, 0.8609] (`docs/prereg/a5_manifests/power/report.json`, `ordered_power`, `ordered_power_cp95`). The 0.80 crossing lies
between 34 and 37 available slots.
(d) *Per-cell resolution.* Per-slot resolution probabilities under the registered donor cells are about 1.00 for every LoRA cell,
0.906 for chain-high, 0.659 for chain-middle and 0.573 for chain-low (metrology report §1.3; result file
`data/runs/a5/f11_sizing/donor_cell_resolvability.json`, not committed, derived from the NULL-SIZE payload). Power therefore
depends on which cells hold the available slots, not only on how many there are. The emulator in the same section (planning only,
not a gate) reaches 0.80 at 36 slots only for high-cell-heavy allocations, and for every allocation at 38 (§1.4).
(e) *Pass-rate evidence* (`part_a/manifest.json`):
- LoRA overall 27/30, but LoRA-high 7/10;
- root chain picks 2/2 (`014f8a9ec78db2be`, `0559051354691a70`);
- non-root chain picks 2/8.

The `replay_failed` mechanism (scorer round trip on predecessor-zero / base-nonzero entries; PROGRESS 06:30 UTC) needs a
non-base predecessor and cannot arise for roots. Planning probabilities that the gate passes for k = 12 (4/4/4) are 0.86–1.00
across the metrology pass-rate grid p_L ∈ {0.7, 0.9}, p_C ∈ {0.5, 0.7, 1.0} (metrology report §2.3, emulator, not a gate).
A 500-replicate unregistered run with all twelve passing gave 0.872 [0.840, 0.900] (§1.2, `s_l2p4_c0p4_c2p4`).

**F11.2 — Supplementary selection rule (frozen).** Part A adds **k = 12** supplementary components, **four in each depleted cell**.
A depleted cell is a registered (family, tertile) cell that lost at least one rotation pair at the F10 re-stage. The depleted
cells are exactly LoRA-high (`lora_sft:2`), chain-low (`chain:0`) and chain-high (`chain:2`); the builder stops if the Part A
manifest gives a different set.

*Order.* Within each such cell, candidates are taken in the registered next-in-cell order `sorted(cell, key=canonical_edit_id)[quota:]`.
Here `cell` is the representative list `docs/prereg/a5_manifests/selection.json:registered.ledger.families.<family>.cells[tertile]`
and `quota` is the registered quota (10, 4 and 3). This is A5.3.1's own selection key, `sort(cell, key=canonical_edit_id)[0:quota]`,
continued past the quota. The builder checks that the sorted prefix equals the registered picks of each cell.

*Filters.* A candidate is selected if all of the following hold, from metadata only:
1. It is an eligible development row with certified closure (`components.jsonl`).
2. It is a **root** (predecessor = base: `components.jsonl:parent_id` null and `power/fields_manifest.json`
   `predecessor_is_base`; a disagreement between the two stops the builder).
3. It has an **exact WikiText gradient**: a `fields_manifest.json` donor with `gradient_source` `base_field` and no flags, listed in
   the F7.1 pool `power/registered_donors.json`. The G2 base field is exact at a root's predecessor (F7.1).
4. It is **not alias-only**: its own bytes are stored (`bytes_present` true and `bytes_via_alias` null in `components.jsonl`,
   `donor_via_alias` false in `fields_manifest.json`; a disagreement stops the builder).
5. It is not one of the registered forty.
6. Its component is not already represented.

The first four candidates that pass, in that order, are selected. Later candidates are recorded as `reserve_unreached` and are
never used.

*The list.* Builder: `scripts/a5_f11_supplement.py`. Output: `docs/prereg/a5_manifests/selection_f11.json` (schema
`a5-f11-supplement-v1`; per-candidate positions and skip reasons; input sha256s, all equal to the registered identities:
`selection.json` `39353036…0e908c63`, `components.jsonl` `6b4eff8d…46251dec`, `fields_manifest.json` `19e9de84…5e9a6b`,
`registered_donors.json` `6e77bca6…6babbb5`, `part_a/manifest.json` `59d839e3…0d3b06`; full strings in the file). The twelve ids are:
- LoRA-high, positions 10–13: `8d29ac2adb7fdda0`, `af6d608f8651fb46`, `b5d0ac5d76c3cad6`, `c12f6ec6bad59ba3`.
- chain-low: `342a1be080049e8f` (6), `388347cfb0b7fafa` (7), `45cd0be9fea1c139` (8), `aaf7a0660c6d5a73` (11). Skipped: positions
  4, 5 and 9 as non-roots without an exact gradient, and position 10, `7bc62b9f723359cb`, as alias-only (no `delta.safetensors` of
  its own; bytes via `b0b6b9cb18e49442`).
- chain-high: `15cdcf0bb126de17` (3), `17668df17662b3f7` (5), `1ac30f3c7a209fe6` (6), `2410c595817dc20b` (7). Skipped: position 4
  as a non-root.

Sorted-id sha256 (`sha256(json.dumps(sorted(ids)))`, the F7.1 convention) is
**`1848fe4cacbaacc9f643a179835770fefd3027a48cef7cf9d83358f15b85fdd9`**. The rerun is byte-identical.

*Fixed.* The list and its hash are fixed by this commit, before any supplementary staging. There is no top-up, substitution or
reserve use after any supplementary audit, diagnostic or yield has been seen. The forty registered pairs are not replaced:
- they stay in the ledger with their F10 verdicts;
- the 9 excluded rotation pairs stay excluded with their reasons (F10.3, "without replacement");
- the 31 staged rotations and the 80 staged ladders are untouched and are not rebuilt.

A supplementary exclusion is not replaced either.

**F11.3 — Staging under F10.** Each supplement is staged once, under the F10 audit rule (`--audit-rule f10`; F10.3 as read by
F10a.1). It gets a rotation with its registered A5.3.3 seed (`rotation_seed`, `control_seeds.json:rotation`) and two ladders,
s ∈ {0.25, 0.5}, built as registered for roots (stored Δ scaled at the base and rounded; no F10.9 flag arises). Exclusions follow
F10.3 with reason codes and without replacement. A valid ladder is kept whatever its rotation verdict, as for the forty.

Staging runs on CPU (`CUDA_VISIBLE_DEVICES=` empty) under `agent-tmux`, with the freeze and execution-record requirements of
A5.1.2 re-pin (xviii):
- the reviewed code is merged and pinned first;
- a pre-execution freeze file records the environment, threads (`OMP_NUM_THREADS` = `MKL_NUM_THREADS` = 4, Torch intra-op 4),
  `HF_HUB_OFFLINE=1`, input hashes including `selection_f11.json` and its sorted-id hash, and the command verbatim;
- a launch record holds the execution SHA (clean tree), start and end times and the exit code;
- the forty's manifest and audit files are preserved before execution, and any new manifest is written so that the forty's
  entries are byte-identical.

Staging the supplements needs the `prepare` path to accept `selection_f11.json` as an additional, frozen selection input. That
is a code change with tests and a re-pin, reviewed by Board A before execution; it is not made here.

Price (metrology report §3, from `part_a/README.md` and `part_a/restage_launch.json`): the full three-variant staging ran at
7.46 min per edit (4 h 58 min 33 s / 40), and the F10 rotation re-stage at 5.46 min per edit (02:44:00–06:22:28 UTC = 3 h 38 min
28 s / 40). So the 12 supplements cost about **1.5–2.6 h** of CPU. That is an upper bound, because 8 of the 12 are chain roots
with fewer edited tensors. The staging report gives per-pair F10.3 verdicts with reasons, F10.4 diagnostics, delta hashes and the
updated priced count. It contains no score.

**F11.4 — Calibration on the supplemented slot set.** Before any scoring, one separately labelled, registered-scale slot-masked
NULL-SIZE / POWER-AND-YIELD run is made on the **52-slot** set (the 40 registered and the 12 supplements, in their cells; per-cell
slots LoRA 10/10/14, chain 8/3/7).

Fixed settings:
- 10,000 replicates and 4,000 resolution draws;
- the F7.1 61-donor pool for POWER-AND-YIELD and the full payload for NULL-SIZE (F7.2);
- `SeedSequence(20261107)`, spawn 3, streams in the order panels, signs, resolution;
- the registered 0.75 alternative.

*Mask.* The mask is the F10 exclusions among the 52, per contrast: the 9 registered rotation exclusions plus every supplementary
rotation or ladder exclusion from the F11.3 staging report. It and its id-to-slot mapping are frozen and hashed before the run.

*Gates,* in this order: ordered power ≥ 0.80; then ordered go and each individual two-sided size ≤ 0.07 (A5.3.2; FP2). The null
rate of the step-2 alternative is reported next to the go rate, not size-gated (F5.7). No seed search, rerun, replicate change or
mask edit is allowed.

The run needs a supplement option in `scripts/a5_simulation.py`: the frozen `selection_f11.json` ids mapped to their cells,
slots appended after the registered slots in cell order, and mask ids resolved over all 52. The option needs tests (registered and
F10.6 outputs unchanged without it) and a re-pin before the run; it is not made here. Expected CPU wall is about 11–22 min (the
F10.6 run 06:25:29–06:36:01 UTC, `slotmask_calibration_record.json`; the F7 runs 21 min 39 s and 22 min 27 s, F10.6). **A failure
of either gate invokes F11.7.**

**F11.5 — Analysis.** If F11.4 passes, the registered Part A analysis (A5.3 resolution, yield and CP rules; A5.4; the ordered
decision function A5.5) runs once on the supplemented set as **the registered Part A**:
- one pair per component, 52 components;
- per-contrast denominators of 52 less exclusions, with separate rotation and ladder-strength denominators and no borrowing;
- the ≥ 15 resolved-pair floor, CP rule and step-2 alternative unchanged;
- analysis time after all supplemented variant scores and references are on file and the ledger is closed.

The donor pool of every simulation stays the F7.1 61 donors. Reported descriptively, with no decision authority:
- a **registered-40 sensitivity** with the same analysis on the original forty only (31 rotation pairs, 40 ladder pairs per
  strength), labelled as such;
- agreement split into supplements versus the forty and roots versus non-root chain steps (metrology report §3, bias (i));
- the F10.5 reports on the supplemented set. The subset lists of F10.5(c) and (e) stay as frozen on the forty (F10a.4); the
  supplements' values of the same predicates are shown beside them and do not enter those subsets.

Analysis-code changes needed for 52 components are code changes with tests and a re-pin, made before scoring.

**F11.6 — Pre-outcome attestation and bias statement.** No G3, rotation, ladder, reference or sealed outcome exists. Nothing has
been scored (`part_a/manifest.json:scored` false).

*Bias.* k = 12, the four-per-cell allocation and the root filter were chosen after the F10 re-stage yield (31/40), its reasons and
the F10.6 failure (0.7514) were known. They are **audit-yield-informed and outcome-blind**, the same status F10a.3 gives F10's
thresholds: "fixed before outcomes, not before audits".
- The root filter shifts the chain-low and chain-high supplementary slots from chain steps to first-step roots. The rotation
  contrast there then speaks to roots, not continued-chain increments; F11.5's split reports this.
- The order is the registered one and is not chosen. Still, the four LoRA-high supplements have effective norms 2.66–4.33
  (`selection_f11.json:supplements`), against 4.69–6.78 for the three excluded LoRA-high picks (`components.jsonl`). This is
  disclosed.
- `aaf7a0660c6d5a73` has re-trained lineage (`exactness` `re-trained`), like three of the forty.
- The not-alias-only filter is in the chair's F11 brief to this drafter. It moves the fourth chain-low pick from the metrology
  report's draft list (`7bc62b9f723359cb`, position 10, alias-only) to `aaf7a0660c6d5a73` (position 11). With the alias-only edit
  the draft list's sorted-id sha256 would be `5e4260b1…a45a41`; the frozen list is the one in F11.2.
- D3 originals of the supplements are known, as for the forty (A5.3 *Exposure*). They enter only as simulation donors.

*Attestation of this drafting session* (branch `prereg/f11`, worktree `.worktrees/prereg/f11`, 11 Nov 2026). It read:
- `docs/` (this file, PROGRESS, the metrology report, the Board A F10 and F10-code-2 reviews, `docs/prereg/a5_manifests/`);
- `scripts/` and `src/`;
- the zoo manifest `data/zoo/qwen3.5-0.8b/manifest.jsonl` and the `edit.json` files of four candidates, with a directory listing
  of those four entries;
- the outputs `data/runs/a5/power/sizing_slots_{34,37,39}.json`.

It opened nothing under `data/evals`, used no GPU, staged nothing, ran no simulation and wrote nothing under `data/`. It has seen
no G3, rotation, ladder, reference or sealed outcome, or a summary of one. The chair, the metrology member, the stager and Board A
each add an attestation line (F8a.1(c)) to `docs/prereg/f11_attestations.md` before staging.

**F11.7 — Fallback (Option B).** If F11.4 fails either gate, Part A proceeds as a **screen** without further amendment. The
rotation contrast is **not decisive** (A5.3.2). Scoring then covers the F10 state plus the supplementary variants already staged,
and the A5.5 validity row is read with the metrology report's Option B wording:

> The A5.5 validity predicate "applicable power … gates pass" is read as not applicable to the rotation contrast. The step-2
> ladder alternative keeps its funding-stop role. A rotation interval with lower bound > 0.5 is reported as "Part A rotation:
> positive screen, not decisive". It is not a Part A go, and it does not count toward G7 ("G3 go and G4 go, never screen", A5.6).
> Every other outcome is reported as registered.

The failed calibration's power is stated next to any rotation result. Part B is reported but cannot fund without a Part A go
(A5.5 step 4).

**F11.8 — Cost.**
- *CPU:* staging about 1.5–2.6 h (F11.3) and calibration about 11–22 min (F11.4).
- *GPU scoring:* 3 entries × 1.04 min (`part_a/manifest.json:scoring.minutes_per_entry`) per supplement, at most 36 entries =
  **37.44 min**.
- *Total:* at most 135.2 + 37.44 = **172.64 GPU-min**. The current priced count is 130 entries, 135.2 min
  (`part_a/manifest.json:scoring.n_priced`, `minutes`).
- n_P = 0, because every supplement's predecessor is the base. n_D = 0 is the planning value: all twelve are NULL-SIZE donors with
  TF-v1 originals (`fields_manifest.json:donors`). It holds subject to the A5.3.4 reuse proofs built at staging (identical
  effective snapshot, windows and scoring identity), with particular attention to the re-trained lineage `aaf7a0660c6d5a73`. A
  failed proof adds that original at 1.04 min (stop-and-reprice).
- Supplementary exclusions lower the GPU count.

**F11.9 — What does not change.**
- The decision rules (A5.5) and the line-stop rule (F5.5).
- The H5 success criterion (A5.7).
- Seeds (`control_seeds.json`; `SeedSequence(20261107)`), the registered 0.75 alternative, the resolution, yield and CP rules,
  and the ≥ 0.80 and ≤ 0.07 gates.
- Caps and allocations other than the Part A entry count priced in F11.8.
- The F7.1 donor pool.
- F10.3, F10a and the F10.9 flags.
- The forty registered pairs, their verdicts and staged bytes.
- `selection.json` and the frozen cutpoints.
- Every pin, except the re-pin that the F11.3–F11.5 code changes will entail; that re-pin is not made here.

| clause | rule changed | evidence | Board A gate |
|---|---|---|---|
| F11.1 trigger | none | 31/40 (`part_a/manifest.json`); 0.7514 (`slotmask_calibration_record.json`); sizing 34/37/39 → 0.7722/0.8251/0.8453 (`data/runs/a5/power/sizing_slots_*.json`); per-cell resolution (metrology §1.3) | accept the evidence as outcome-free |
| F11.2 selection | A5.3 "40 components" and A5.3.1 quotas extended by addition: +4 in LoRA-high, chain-low and chain-high (roots, exact gradient, not alias-only), next-in-cell canonical-ID order, k fixed | `selection_f11.json` (sorted-id sha256 `1848fe4c…85fdd9`); `scripts/a5_f11_supplement.py`; tests | accept the rule and the frozen list before staging; no top-up |
| F11.3 staging | none to F10; adds a frozen supplementary selection input to `prepare` | re-pin (xviii) freeze/launch pattern; priced 1.5–2.6 h CPU | review code, tests and re-pin; accept the staging report |
| F11.4 calibration | F10.6 masked run on 52 slots instead of 40 | F10.6 record; supplement option in `a5_simulation.py` | review code and re-pin; mask frozen before the run; ≥ 0.80 then ≤ 0.07 |
| F11.5 analysis | registered Part A = 52 components; registered-40 reported descriptively | A5.3, A5.4, A5.5 unchanged | accept the denominators and the sensitivity labels |
| F11.6 attestation / bias | none (applies F8a.1(c), F10a.3) | this block; `f11_attestations.md` | attestation lines of chair, metrology, drafter, stager and Board A |
| F11.7 fallback | A5.5 validity row read as not applicable to the rotation contrast, only if F11.4 fails | metrology report Option B | accept the wording now, before any calibration |
| F11.8 cost | Part A entries ≤ 166 (≤ 172.64 GPU-min) | `part_a/manifest.json:scoring` | accept the price |
| F11.9 unchanged | none | — | none |

*A5.1.2 hash entries after the F10 re-stage (chair, 11 Nov 2026, about 08:30 UTC; copied from `docs/prereg/a5_manifests/part_a/RESTAGE_A512_NOTE.md`).*


This file is not part of `docs/PREREG_H1_TF.md`. The paragraph below is meant to be appended to A5.1.2 after re-pin (xviii). The
chair appends it. The hashes were re-computed on 11 Nov 2026 at about 07:17 UTC from the files at branch `report/f10-restage`
(based on main `be02cc3`).

---

*A5.1.2 hash entries for the F10 re-stage (11 Nov 2026, about 07:17 UTC, after execution; the pre-execution freeze is re-pin
(xviii), 05:45 UTC entry, `restage_freeze.json` written 02:40:50 UTC; before any G3 outcome).*

The rotation-only re-stage ran 02:44:00–06:22:28 UTC at clean execution SHA `7bacec12604f1090168a79a6c6d55d4f548af4ec`
(exit 0). Its outputs were committed in `848d7a5`. All 71 pins of `code_hashes.json` (`74321d76858274d2…`) match at that SHA and
at `be02cc3`.

Regenerated and new files under `docs/prereg/a5_manifests/part_a/`:

- `manifest.json` (regenerated): `59d839e351824fa154a665fcf105284ce5fdeed8423a5c0a2b7a60687b0d3b06`
- `rotation_audits.jsonl` (regenerated; 2,534 rows): `4ec0928a91b5523a400b911f00f8045f267386c2589aa72a58816d565f5c97f3`
- `restage_launch.json` (launch and end record): `28dcba683dc215fe303d938a1c7b094b5d0856929f19fb720187b75bb1a59a7e`
- `restage_freeze.json` (pre-execution freeze, (xviii), for reference):
  `03dfaf2667d0addbd459c0662b8fa87a2304b36a054c36cf64ac29192ca48858`
- `slot_mask_f10.json`, F10.6 mask as used. Schema `a5-slot-mask-v1`, rewritten in `cf3eae8`. It masks the 9 excluded rotation
  slots; mapping sha256 `b7724a69315d8c8692a128504126251ac16945cf7b741ec60b3cdb6daa5cb329`. File:
  `86fa27a5c1838d0c0bd88ef530feba22eceafac698d0ec4506fbf05df83b698e`.
  - The first mask file, committed in `848d7a5`, was `a2ed6ae7e9ac8e08850ddbaded875f9fda65704ea261bdb50ddb080da1a2fe80`. The
    simulator rejected it for unknown keys, so nothing was simulated with it.
- `slot_mask_f10.provenance.json`: `0bb56b4bb05ce0718f48e7fc962a60e4378b06d9c46c9cf08e4b1ef78be08636`
- `slotmask_calibration_record.json`: `0c5fa8c38da42d63bd5762c307ec1c2254313501fbfd4763b3418eaa3eff6b51`
- `RESTAGE_REPORT.md` (the re-stage report, which serves as the F10.7 regenerated README):
  `61294c8666b4bbe4f938e76af7797631ef86613e77d431229bddcaf69d6d5f61`

Unchanged files:

- `README.md` (the a533 staging README): `2fcbf517ccf26f2a1fe325ef31300d291bc0f254c7faaacd18154e327bceada1`
- `manifest_a533.json`: `d7785b35de1d7041edd2db139591562379db12e33c2532921c814735e8dde8d6`
- `rotation_audits_a533.jsonl`: `f05fec4265ffd93c8c537100993d23262be95141369240fcad0a7e16ff474d35`

Calibration output: `data/runs/a5/power/power_slotmask_f10.json`,
`b14756828bc27cbbe14e1e590226820371a65ef5f83ad9622b2e6ccf04a3450a`. Run at `cf3eae8`, 06:25:29–06:36:01 UTC, with
`scripts/a5_simulation.py` `c4f0c55f…` (re-pin (xvi)). Result: ordered power 0.7514 < 0.80, **FAIL**; size passes.

Yield: 31 / 40 rotation pairs; 1,984 / 2,534 tensors rotated. The 80 ladder variants (96 files) and the prior passing pair's
delta (`951e8621…`) and increment (`78ec4689…`) are byte-identical to the staged hashes. 16 ladder entries are flagged
`realised_increment_ladder`. Priced count: 130 entries, 135.2 min.

Recorded discrepancies:

- The emitted Torch build-config SHA-256 (`07f0372b…`) differs from the frozen value (`d1b4e24c…`).
- The freeze's `base_revision` is empty. All 40 selected edits record `2fc06364…`, the model pin.

Nothing has been scored. The scoring hold remains (F10.6; Board A F10 condition 5).

*Chair rulings on the re-stage report (11 Nov 2026, 08:30 UTC).* (1) Freeze field mismatch: `restage_freeze.json` recorded the Torch
build-config SHA-256 as the hash of the UTF-8 text of `torch.__config__.show()` (`d1b4e24c…`); the staging code derives it
differently and the run recorded `07f0372b…`, which recomputes in the same environment; the nine other environment fields match, the
process environment was the same, and the code's derivation is authoritative from now on; the freeze's empty `base_revision` was a
wrong key in the chair's script, and the run records `2fc06364…` for all 40 edits, equal to the model pin. Disclosed; no bearing on
the audits. (2) NULL-SIZE block of the slot-masked run (go 192 / alternative 1939 / unresolved 7869) differs from the registered null
(180 / 1939 / 7881) only in the masked rotation contrast, as masking must produce; equality was never a gate (F10.6 states the
gates: size ≤ 0.07 on the masked run, which passed, and ordered power ≥ 0.80, which failed); the chair's 06:30 progress wording
("NULL-SIZE block equality") is withdrawn. (3) The three lora_sft `rounding_bound_failed` exclusions exceed the bound by thousands of
fp32 spacings on one entry each; they are not the inclusive-equality case; the 05:20 progress wording is corrected; the exclusions
stand under the no-retuning rule.

*Chair correction to ruling (1) of 08:30 UTC (11 Nov 2026, about 09:10 UTC; Board A re-stage review condition 1,
`docs/reviews/council/board-a-f10-restage-2026-11-11.md`).* The "different derivation" explanation is withdrawn. The staging code
(`src/wsp/features/rotation.py`, `torch_config_sha256 = digest(torch.__config__.show())`) and the chair's freeze script hash the same
text. The discrepancy arises because `torch.__config__.show()` reports the process's thread configuration: the freeze value
`d1b4e24c…` was computed by the chair in a shell WITHOUT the frozen environment variables, and the run's value `07f0372b…` is what the
frozen environment produces. Verified today by recomputing under `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
HF_HUB_OFFLINE=1`: `07f0372b…`, equal to the run's `restage.replay_environment.torch_config_sha256`. The discrepancy is therefore a
chair error in the freeze procedure, not an environment change; the run executed under the frozen settings (threads 4 recorded).
Ratified: the binding Torch build-config hash is `07f0372b…` computed under the frozen environment; future freezes must be computed
under the same environment variables as the run. Correction to the re-stage report's §2/§8 (condition 2): 93 tensor rows have
maximum excess exactly one fp32 spacing and pass the rounding check; 92 of them pass F10 overall; the remaining one
(`057dcfbedea3251f`, `model.layers.6.mlp.up_proj.weight`) is excluded for replay. The scoring hold and baseline evidence are
preserved; F11 staging and calibration remain separately gated.

#### A5 note F11a (11 Nov 2026, before any G3 outcome): F11 clarifications (Board A F11 condition 1)

This block is appended; earlier text is not rewritten. It discharges the methodological-freeze part of condition 1 of Board A's F11
review (`docs/reviews/council/board-a-f11-2026-11-11.md`, "the review"; merged in `7ba776d`). Quotations are copied verbatim from the
review's sections "Calibration and analysis" and "Fallback" and from its verdict line. F11a clarifies how F11 is read; it changes
no F11 operational rule except as stated in F11a.1 and F11a.2, where it adopts the review's own wording. Where F11.4, F11.5 or F11.7
read differently, F11a governs.

**F11a.1 — F11.7 fallback (Option B, screen): power failure only; a size failure holds scoring.** The review's verdict line:
> F11.7 needs the restricted fallback interpretation below; its literal "either gate" wording does not waive a size failure.

and its section "Fallback":
> Accept a prospective screen fallback for **power failure with all size and other applicable validity gates passing**. A5.5
> separately requires development null size ≤0.07; declaring rotation power inapplicable cannot waive that predicate. If either
> ordered-go size or any applicable individual two-sided size exceeds 0.07, validity stays unresolved and scoring remains held pending
> a new prospective ruling. Compute/report size even if the power check fails first, so fallback eligibility is known.

Accordingly:
- (a) The F11.7 fallback (Option B, screen) is available **only** when the F11.4 calibration fails the **POWER** gate (ordered power
  < 0.80) and every size gate (ordered-go size and each individual two-sided size ≤ 0.07) and every other applicable validity gate
  passes. F11.4's "A failure of either gate invokes F11.7" and F11.7's "If F11.4 fails either gate" are read in this restricted way.
- (b) A **SIZE** failure (ordered-go size or any individual two-sided size > 0.07) **never** falls back to the screen. Validity stays
  unresolved, the scoring hold stays in place for every Part A entry, and only a new amendment, with its own review, can release it.
- (c) The F11.4 run computes and reports every size rate even when the power gate fails first, so that fallback eligibility is
  known from the report.
- (d) *Exception to the introductory hold.* The F11 introduction holds scoring "in every case until F11.4 passes". As the review
  puts it: "after an accepted power-only failure report the explicitly authorized screen route may release scoring, subject to
  every other gate." Only that route is excepted. Board A must accept the power-only failure report first.
- (e) In the screen route only rotation power is waived: "no Part A go, G7 contribution or Part B funding is possible. The step-2
  ladder alternative retains priority and funding-stop meaning, and the ≥15 resolved floor still applies to a positive screen.
  Remaining validity checks and unresolved cases retain their registered handling. Report the failed power and all size results beside
  the screen. 'Every other outcome ... as registered' cannot silently restore step-3 go or downstream funding."

**F11a.2 — NULL-SIZE uses the full registered payload.** F11.5's sentence "The donor pool of every simulation stays the F7.1 61
donors" is qualified, as the review requires:
> The fixed 61 donors remain the POWER-AND-YIELD pool even when supplementary edits are members of it. [...] Crucially,
> **NULL-SIZE keeps the full payload under F7.2**. F11.5's sentence "donor pool of every simulation stays ... 61" must be qualified
> to POWER-AND-YIELD; F11.4's explicit full-payload NULL-SIZE rule governs.

Accordingly: (a) the NULL-SIZE pass of the F11.4 run uses the full registered payload, all 141 donors, exactly as F7 and the F10.6
run did. (b) The F7 restriction to the 61-donor pool applies to POWER-AND-YIELD only. (c) The 61 donors are fixed: no donor is added
or removed, self-membership of supplementary edits is kept, no residual is refitted on new outcomes and no leave-one-out sampling is
introduced.

**F11a.3 — The original forty are preserved; there is no replacement.** The forty registered pairs keep their IDs, verdicts and
staged bytes. The 31 staged rotations and the 80 staged ladders are untouched and are not rebuilt; the 9 excluded rotation pairs stay
excluded with their reasons and are not replaced (F10.3, "without replacement"). Supplementary edits are additions only. There is no
top-up, substitution, new order, new cutpoint or reserve use after any supplementary audit or diagnostic, and a supplementary
exclusion is not replaced (review, condition 1; F11.2 "Fixed"). The forty's manifest, audit history and bytes are preserved, and any
drift is rejected rather than overwritten (review, condition 2). The frozen list stays `selection_f11.json`: twelve IDs, sorted-ID
sha256 `1848fe4cacbaacc9f643a179835770fefd3027a48cef7cf9d83358f15b85fdd9`, file sha256
`ce746e7135dd94b087da86953beb02382e5706f8ff2dd17bd57c053aaf54793c` (the review regenerated it byte-for-byte). The draft metrology list
(with `7bc62b9f723359cb`) is superseded and is not a reserve.

**F11a.4 — Attestations.** The F11 exposure attestations are completed in `docs/prereg/f10_attestations.md` (the file carries the
F10 and F11 record; entries 10–13 are added with this note), not in the `docs/prereg/f11_attestations.md` that F11.6 names, which
is not created. Board A F11's own line is incorporated by reference (`board-a-f11-2026-11-11.md`, final paragraph, "Board A
exposure attestation"). Entries for the stager, the implementation sessions and any approving reviewer are added when those sessions
act, and the attestations are updated through the final methodological freeze before scoring (review, condition 6).

**F11a.5 — What this note does not do.** It changes no selection beyond F11.2's committed list, and no seed, decision rule, cap or
pin. It authorises no staging, calibration or scoring by itself. Execution still needs the review's conditions 2 to 6: the `prepare`
supplement path, the 52-slot simulation option and the 52-component analysis with the manifest-binding fix each need Board A review,
CPU tests and a re-pin, and the accepted F10 re-stage report is a separate gate. Nothing has been scored and no G3, rotation, ladder,
reference or sealed outcome exists (`part_a/manifest.json:scored` false).

*A5.1.2 re-pin (xix) (11 Nov 2026, about 10:20 UTC; merge of `feat/a5-sim-supplement` after Board A's acceptance with conditions,
`docs/reviews/council/board-a-f11-sim-2026-11-11.md`; before any G3 outcome).* `scripts/a5_simulation.py` gained `--supplement`
(52-slot design: slots 40–51 appended in cell order; the 40 unchanged; byte-identical output without the option, golden-hash
regression), `--registered-40-sensitivity` (descriptive, `decision_authority: false`), and the 52-id `--slot-mask` rules (ladder masks
allowed for supplementary ids only). Builder re-run with `--pin-worktree` into a copy; only `code_hashes.json` and the package README
copied back. Rows: `scripts/a5_simulation.py` `4ce3faa51889c4aa27f4d9580e7aab9343351954146cbbd76e162ccd47a7947a` (old `c4f0c55f…`,
xvi); `tests/test_a5_simulation.py` `d231c8b0264d48e7e75b991b5bf82ee2f71b846c0364063800bbea84920d2833`;
`docs/prereg/a5_manifests/code_hashes.json` `a0077ab6f0518ec586689b07dc1a203f315d740393506cfa6ae3b74a0fb88d95` (old `74321d76…`,
xviii); `docs/prereg/a5_manifests/README.md` `26ca1d2da368f321acd4addbefb84a091cbbe8e9c775c5eb2731c96bf2cb034b` (old `692879c7…`,
xviii). No selection, seed, control, panel or decision file changed; no F11.4 run executed; nothing scored. Further re-pins follow
the `prepare --supplement` merge (before staging) and the analysis F11-path merge (before scoring).

*A5.1.2 re-pin (xx) (11 Nov 2026, about 11:40 UTC; merge of `feat/f11-supplement-stage` after Board A's acceptance with conditions,
`docs/reviews/council/board-a-f11-stage-2026-11-11.md`; pre-execution freeze for the F11 supplementary staging; before any G3
outcome).* `src/wsp/eval/response_intervention.py` gained `prepare --supplement` (loader bound to the frozen `selection_f11.json`
and the F11.2 rule; the forty re-hashed before and after; merged 52-id manifest with `supplementary: true` flags; `*_pre_f11.*`
preserved; `--no-reference-proofs` default, proofs established by `score --manifest` before scoring), the `slot-mask` builder and the
deferred proof step. Builder re-run with `--pin-worktree` into a copy; only `code_hashes.json` and the package README copied back.
Rows: `src/wsp/eval/response_intervention.py` `e52ea009740751a3…` (old `694f9f4e…`, xviii); `tests/test_response_intervention.py`
`c7f202369469f56a…`; `docs/prereg/a5_manifests/code_hashes.json` `46dd52341df19ad0…` (old `a0077ab6…`, xix);
`docs/prereg/a5_manifests/README.md` `62e6841312b34710…` (old `26ca1d2d…`, xix). Frozen execution: Board A's command verbatim
(`prepare --supplement … --no-reference-proofs --audit-rule f10 --strengths 0.25 0.5 --config configs/eval_tf_v1.yaml`), CPU only,
`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`; the freeze record
`docs/prereg/a5_manifests/part_a/supplement_freeze.json` (sha256 `1094d170…` before the revision patch; re-hashed in the launch
record) was computed UNDER the frozen environment this time (Torch build-config `07f0372b…`, threads 4, Python 3.12.3, Torch
2.15.0+cu132, MKL available, CPU model, x86_64, model revision `2fc06364…` from the manifest), with the selection_f11 hashes, the
input hashes, the originals' manifest (`59d839e3…`) and audit (`4ec0928a…`) hashes and the 81-file staged byte inventory. The
execution SHA is the clean HEAD at launch, recorded in `supplement_launch.json`. Conservative priced count 178 entries. No
selection beyond F11.2's committed list, no seed, decision rule, cap or other pin changed; nothing scored.

*A5.1.2 re-pin (xxi) (11 Nov 2026, 08:16 UTC; merge of `fix/part-a-analysis-manifest-binding` after Board A's acceptance with
conditions, `docs/reviews/council/board-a-part-a-analysis-4-2026-11-11.md`; before any G3 outcome).* The Part A analysis now binds
the frozen subsets, cuts and cells to the preserved a533 manifest (`FROZEN_STAGED_MANIFEST_SHA256` = `d7785b35…`), takes the
execution manifest by explicit `--manifest-sha256` cross-checked with the sealed ledger, includes supplementary edits in the
registered contrasts and the F10.5 reports with a `supplementary` marker, produces the registered-40 descriptive report, accepts
per-contrast calibration masks, and implements the F11a screen route with size-failure priority. Builder re-run with
`--pin-worktree` into a copy; only `code_hashes.json` and the package README copied back. Rows: `src/wsp/eval/part_a_analysis.py`
`5a21e9959511d791337cc16180385f91d6fe2f3ff804ad16b0c5c1febf805ca1` (old `419dc074…`, xvii); `scripts/part_a_analyse.py`
`74bcff0f9f25be7aacf8cfadb3b0f2cdbef044455e2313f1dd6e52f40bce0b32` (old `a47a72f8…`); `tests/test_part_a_analysis.py`
`829932c50ac8a8cef3236102e7e9ce6a92f3aff705d8ad625e3111412e8f5e69` (old `5adaeaf8…`); `docs/prereg/a5_manifests/code_hashes.json`
`3447730883fe3030…` (old `46dd5234…`, xx); `docs/prereg/a5_manifests/README.md` `bb6173b11aee2322…` (old `62e68413…`, xx). The
conventions Board A requires by dated note (supplement cell assignment; screen-route acceptance input; forty-only size failure
refuses; 52-slot calibration always required with supplements; supplementary row refusals) are adopted in A5 note F11b. No
selection, seed, decision rule or cap changed; nothing scored. The supplementary staging launched at 08:06 UTC on execution SHA
8211ecf is unaffected (it does not import the analysis module).

#### A5 note F11b (11 Nov 2026, before any G3 outcome): adoption of the Part A analysis F11-path conventions

This block is appended; earlier text is not rewritten. It is the dated A5 note that Board A's fourth Part A analysis review
(`docs/reviews/council/board-a-part-a-analysis-4-2026-11-11.md`, "the review"; verdict ACCEPT WITH CONDITIONS; reviewed commit
`f57b70afffcd3f1cca281f50c365ef497fe05e74`, merged and re-pinned as (xxi)) requires in its condition 1: "Adopt the conventions
before scoring. A dated A5 note must adopt module choices 7 and 8: supplementary norm cells from the original selection cut keys
and ln rho cells from frozen energy cut keys applied to the supplement's own staged value, with no movement/re-cutting of the forty
and no supplement in S30/S34; and `f11a_screen_route: {pass: true, evidence: [{path, sha256}, ...]}` requiring linkage to the exact
accepted power-only calibration report. Preserve the existing accepted choices 1–6. State the forty-only size-refusal behavior, the
continued unresolved-validity/scoring hold and the requirement for actual Board A report acceptance. No post-outcome convention
choice." Everything is adopted prospectively. Choices 1–6 of the module docstring of `src/wsp/eval/part_a_analysis.py`, adopted in
F10c, stay in force unchanged, as do F10b.2–F10b.4 and F11a. Where this note states a convention, the executable at the pinned
bytes of (xxi) implements it; the clauses below are the registration, and the code is the means.

**F11b.1 — Supplementary cell assignment (module choice 7).** Adopted, as the review accepts it ("methodologically acceptable
prospectively"; condition 1). Each of the twelve supplementary edits joins the F10.5 reports under a cell assigned by a frozen rule
and is marked `supplementary`:
- (a) *Norm tertile.* The supplement's A5.3.1 norm tertile is the cell its `(effective_norm, id)` key falls in under the frozen
  selection cut keys of its family (`selection.json:procedural_freeze.cutpoints[family].keys`; the file is bound by
  `manifest_a533.json:selection_ledger.selection_json_sha256` = `393530365f87f981…`). A key equal to a cut key falls in the higher
  half-open cell. The cut keys are verified at run time against the a533 manifest, and the assignment is verified against all forty
  original rows before any supplement is assigned; the stated tertile of each supplement must equal the recomputed one, otherwise
  the run is refused.
- (b) *ln ρ cell.* The supplement's ln ρ cell is the frozen spec §2.2 energy cut keys (`LNRHO_CUTS`) applied to the supplement's
  **own staged** value ln ρ = 2·ln(1 + pooled_norm_mismatch) from its F11.3 staging record in the execution manifest (spec §1.1
  formula). It is not assigned from the re-stage value and not from any re-cut population. A supplement without a staged value has
  no ln ρ cell and is listed as unassigned. The re-stage ln ρ and `cell_drift` are displayed for supplements as for the forty;
  crossings of a cut are disclosed and never move a member.
- (c) *The forty are untouched.* No cut is recomputed with supplements in the population, no frozen member moves cell, and the forty's
  cell memberships stay anchored to the preserved a533 manifest (sha256 `d7785b35…`; norm-cell membership hashes verified).
- (d) *S30 and S34 are unchanged.* They remain the frozen registered-40 lists. Each supplement's values of the two subset
  predicates (staged `pooled_norm_mismatch` ≤ 2e-2, cross-checked with `in_subset_pooled_norm_2e-2`; `in_subset_noise_calibrated`
  or `noise_calibrated_passes`) are shown **beside** the lists, with their verdicts, and never enter either list, any subset
  estimate or any decision.
- (e) Supplementary rows join the covariate models M0/M1, every 3 × 2 sign table, the constant-sign comparator and the tertile
  reports, marked `supplementary`, all with authority none (descriptive). Nothing in F10b.2–F10b.4 is relaxed.

**F11b.2 — Screen-route acceptance input (module choice 8).** Adopted. The F11a.1 screen route (power-only failure of the
supplemented F11.4 calibration, every size rate passing) releases step 1 only with the following object in the `--validity` input,
under the key `f11a_screen_route`, exactly:

```json
{"f11a_screen_route": {"pass": true, "evidence": [{"path": "<repository-relative path>", "sha256": "<64 lowercase hex>"}]}}
```

- (a) `pass` must be the JSON boolean `true`; `evidence` must be a non-empty list of well-formed `{path, sha256}` objects, and at
  least one `sha256` must equal the sha256 of the exact supplemented, slot-masked F11.4 calibration report supplied as
  `--masked-calibration`. Absence of the key, `pass` other than `true`, a malformed or empty evidence list, or evidence that does not
  link that hash leaves step 1 **unresolved**, with the reason printed, and every later step unresolved with it. The input is
  consulted only when the calibration state is a power-only failure; on a pass of both gates it is not needed, and it never waives a
  size failure (clause 4).
- (b) *Who may issue it.* Only on **Board A's acceptance of the power-only failure report** (F11a.1(d): "Board A must accept the
  power-only failure report first"). The chair or a builder transcribes that acceptance into the object; the evidence list names the
  calibration report (by its sha256) and, as further entries, the Board A report that records the acceptance. The object is a
  producer's assertion, not independent proof. As the review states: "A syntactically valid JSON assertion is a trusted producer
  assertion, not independent proof that Board A accepted anything. Its real review record must be checked in the pre-score
  inventory." Condition 3 below requires that check. An object issued without a Board A report that accepts that exact calibration
  report by hash is void.
- (c) In the screen route only rotation power is waived (F11a.1(e)): no Part A go, G7 contribution or Part B funding is possible;
  the ladder alternative keeps priority and its funding-stop meaning; a positive screen still needs 15 resolved pairs; the failed
  power and all size rates are reported beside it. The A5.5 validity row is worded as F11.7 states. Other validity predicates and
  Part B inference/control validity still apply.

**F11b.3 — Forty-only size failure refuses.** Adopted as the review rules: "This is a stronger fail-closed operational expression
of the unresolved-validity/scoring hold, not a newly authorized decision or screen route. It does not resolve validity as
scientifically negative. Record this behavior in the adopting note so operators do not mistake refusal for missing analysis output
to work around." Accordingly, when the calibration state shows a genuine size failure (ordered-go size or any individual two-sided
size > 0.07), the analysis **refuses** with the message "hold: size failure requires a new amendment" and writes no report, on the
supplemented run (F11a.1(b)) and on the forty-only F10.6 run alike. Inconsistent gate flags and rates also refuse. A refusal is not
a missing output to be worked around, a Part A negative or an unresolved-validity report: scoring stays held for every Part A entry
until a new amendment with its own review releases it. Neither simultaneous power failure nor an `f11a_screen_route` input can
waive failed size. A forty-only F10.6 power failure (size passing) leaves step 1 unresolved, as in F10c; it has no screen route.

**F11b.4 — The scoring hold continues.** Nothing in this note releases scoring. The unresolved-validity and scoring holds stand
until the pre-score conditions below are met; the calibration and its route are accepted separately by Board A; the screen route
exists only after an actual Board A acceptance of a power-only failure report (clause 2(b)); other staging, calibration,
reference-reuse, pricing, qualification, exposure and release conditions remain in force until separately met.

**F11b.5 — The 52-slot calibration is always required when supplements exist.** Adopted, as the review rules: "Accepted and
necessary. Full audit yield does not make the forty-slot calibration applicable to fifty-two slots." With any supplement in the
selection, the analysis requires the supplemented slot-masked calibration report (label "slot-masked (F10.6); supplemented
(F11.4)") covering every selected slot, **even at full audit yield**: under the twelve-ID inventory this is the one registered-scale
52-slot run, not a forty-slot calibration reused for a larger panel. It carries a `supplement` block (supplementary ids' sorted-id
sha256, k, slot count), per-contrast masks that recompute from the execution manifest in the canonical availability mapping with
their sha256, the ordered power, the ordered-go size and the three two-sided size rates. Masks of an original-forty ladder are
refused (the eighty ladders are preserved, F11a.3); supplementary ladder exclusions at either strength are accepted independently of
rotation availability; the registered-40 sensitivity run is refused as calibration authority. Without supplements, F10c.2 and the
frozen registered evidence for the full-yield route are unchanged.

**F11b.6 — Supplementary row conditions; the twelve-ID population is bound by hash.** Adopted within the existing trusted
execution-manifest and input-inventory contract, as the review rules: "acceptable [...] not a general waiver of F11.2". The analysis
refuses any supplementary selection row that is not flagged `supplementary: true`, is not a root (predecessor `base`), is not in one
of the three depleted cells (`lora_sft:2`, `chain:0`, `chain:2`) or whose stated tertile differs from the frozen cut keys (clause
1(a)); it refuses a registered id flagged supplementary or a registered row that differs from its a533 row. It reports the sorted-id
sha256 of the supplementary ids and whether it equals the frozen `selection_f11` hash
`1848fe4cacbaacc9f643a179835770fefd3027a48cef7cf9d83358f15b85fdd9`. As the review states, the executable alone "does not enforce
k=12, four per depleted cell, distinct components, exact selected IDs or original staging-byte/verdict preservation"; a false list
match is reported rather than refused, and **a false list match is never an authorized production F11 run**. Those requirements
are enforced by the accepted staging evidence and the pre-score inventory (clause 7). No top-up or substitution is permitted.

**F11b.7 — What the analysis binds (Board A condition 3).** The Part A analysis of the F11 path binds **exactly the twelve-ID
supplementary population**, with the forty, through: (i) the `selection_f11` sorted-id sha256
`1848fe4cacbaacc9f643a179835770fefd3027a48cef7cf9d83358f15b85fdd9` (twelve IDs, four per depleted cell, distinct components, file
sha256 `ce746e71…4793c`); (ii) the **execution manifest**, passed by `--manifest` with the required `--manifest-sha256` and agreeing
with the sealed ledger's `manifest_sha256` and `restage_provenance`; (iii) the **frozen masks**: the per-contrast canonical
availability recomputed from that manifest and the mapping hash; and (iv) the **accepted calibration and route evidence**: the
one registered-scale 52-slot calibration report by sha256 and, on a power-only failure, the hash-linked `f11a_screen_route`
object backed by the actual Board A acceptance. The trusted pre-score inventory must, before scoring (review condition 3), verify:
exactly the twelve frozen IDs and the four-per-cell allocation; component uniqueness; the original forty's preserved identities,
staging bytes, nine rotation exclusions with reasons and eighty ladders; the a533 and final execution manifest hashes; the
ledger and field predictions, windows, producer and scoring identities; the accepted staging evidence; the actual per-contrast
masks and mapping; the 52-slot calibration report and its separate acceptance; and, if a power-only failure, the Board A record
and the hash-linked input. A size failure holds all scoring pending a new amendment. It must also exercise the analysis's
input-binding and tamper checks against the final accepted outcome-free manifest, using synthetic responses only. These
conditions discharge the analysis implementation gate only; they do not discharge any other staging, calibration, reference-reuse,
pricing, qualification, exposure or release condition. One sealed-ledger analysis with descriptive sensitivities is then run.

**F11b.8 — What this note does not do.** It changes no selection, seed, decision rule, cap or pin. It authorises no staging,
calibration or scoring, and it does not close F11 condition 5 (that closes only after the review's conditions 1–3 are discharged,
before scoring). No convention is chosen after any outcome: nothing has been scored, and no G3, rotation, ladder, reference or
sealed outcome exists (`part_a/manifest.json:scored` false). Condition 2 of the review (final-bytes verification and
`--verify-pins HEAD` at the real execution HEAD) is satisfied by its own later record, not by this note.

*No-exposure attestation (this session, 11 Nov 2026).* The drafter of F11b is Claude Sonnet 5.5, branch `prereg/f11b` (worktree
`.worktrees/prereg/f11b`), delegated by the chair. It read the review, the module docstring and two short passages of
`src/wsp/eval/part_a_analysis.py`, the F10c, F11, F11a, F10b and re-pin (xxi) text of this file, `docs/PROGRESS.md` and the
attestations file. It ran no scoring, test suite, simulation, staging or GPU job and wrote only under `docs/`. It opened nothing under
`data/evals/`, wrote nothing under `data/`, and has seen no G3, rotation, ladder, reference or sealed outcome or a summary of one;
none exists. It chose nothing: the clauses copy the review's rulings and the builder's recorded decisions. This attests this session
only; it is also entry 14 of `docs/prereg/f10_attestations.md`.

#### A5 note F11c (11 Nov 2026, before any G3 outcome): F11.8 clarification and disclosures

This block is appended; earlier text is not rewritten. It records one clarification of F11.8 that the pre-scoring inventory found
open (`docs/prereg/a5_manifests/part_a/PRESCORE_INVENTORY.md`, gate row "ST-1 dated F11.8 clarification (deferred proofs, 178,
stop-and-reprice)") and four disclosures about the F11.3 staging and the F11.4 calibration records. It is dated before any G3
outcome. Nothing has been scored, and no G3, rotation, ladder, reference or sealed outcome exists (`part_a/manifest.json:scored`
false).

**F11c.1 — F11.8 clarified: deferred proofs, the conservative priced count and the stop-and-reprice threshold.**
- (a) *Proofs are deferred to the scoring phase.* F11.8 says the planning value n_D = 0 "holds subject to the A5.3.4 reuse proofs
  built at staging". As executed, `prepare --supplement` ran with `--no-reference-proofs` (re-pin (xx); `supplement_launch.json`
  command), because the proof reads TF-v1 `eval.json` and `records.jsonl` of the twelve supplementary originals, and Board A's
  staging review rules that parsing them is exposure, not an outcome-blind exception under F8a.1(c)
  (`docs/reviews/council/board-a-f11-stage-2026-11-11.md`). So the reuse proofs of the 12 supplementary originals were **not**
  built at staging: `manifest.supplement.reference_proofs` records `computed` false, `paths_touched` empty and `deferred_to`
  "score --manifest (score_reference_proofs)". Wherever F11.8 reads "built at staging", it is read as "built in the scoring phase".
- (b) *How they are established.* The proofs are established by the proofs-only call `score --proofs-only` (the deferred proof
  step, `score_reference_proofs(..., write=True)`), run **before any GPU scoring** and only after the final methodological freeze
  and the exposure attestations through it (Board A stage review: it is "the first authorized scoring-phase action"). The record it
  writes is retained. A proven original is reused. A failed one is scored as a reference at 1.04 min and keeps its id (the twelve ids
  never change). `score --manifest` refuses to score any variant while its original stays unproven.
- (c) *Conservative priced count.* Until the proofs exist, the plan is priced on the assumption that all twelve originals must be
  scored: **178 entries = 147 variants + 19 references + 12 originals to prove or score**, at 1.04 min each (`part_a/manifest.json:scoring`:
  `n_variants` 147, `references_to_score` 19, `priced_unproven_as_references` 12, `n_priced` 178), **185.12 min**. If all twelve proofs
  succeed, the count falls to 166 entries, 172.64 min, the figure in F11.8 and in its table row. That is the lower bound, not the plan.
  Retries are **unpriced** (A5.3.4, F11.8, the stage review).
- (d) *Stop-and-reprice.* The chair sets the threshold at **1.25 × 185.12 = 231.4 min**. After the proofs-only call, the price is
  recomputed from its record. If it exceeds 231.4 min, scoring **halts**, and it resumes only after a dated reprice note appended to
  this file before any GPU command. At or below 231.4 min, the repriced plan is still recorded and accepted before GPU commands, as the
  stage review and the inventory (§6, §7) require. Exceeding any cap stops and reprices and never silently cuts coverage (A5.3.4,
  F11.8). Retries stay unpriced. The threshold changes no cap, count or selection: it is the point at which a price change needs a
  dated note rather than a plain record.
- (e) *Open gap.* `scripts/part_a_score_launch.sh` does not implement the proofs-only stop (inventory finding LS-1). It must do so, or
  the proofs-only call must be run separately, before any GPU scoring. This note does not authorise running it.

**F11c.2 — Disclosure: the F11.4 slot mapping was not a separate frozen file before the run.** Board A's simulation review
(`board-a-f11-sim-2026-11-11.md`, condition 2) required the explicit `slot_mask.slots` table to be saved and hashed as
`part_a/slot_mapping_f11.json` before RNG execution ("A mapping emitted only after simulation is too late"). That file was not written
before the F11.4 run (09:28–09:34 UTC, execution SHA `92b0338`). The mask `slot_mask_f11.json` (sha256 `5f4e7701…bbfc4`) was frozen and
committed before the run, as were the manifest, the audits and `selection_f11.json`. The slot mapping is a deterministic function of
those three inputs (the execution manifest, `selection_f11.json` and the mask). The inventory recomputed the mask's per-contrast
availability from the execution manifest (rotation 43, ladder_0.25 52, ladder_0.5 52), and the `mapping_sha256`
`171dfd20b47020dbc172d39eac87adf5c6efbe7864a0f3a784ef018fcb19741a` equals the calibration record and the simulator output (inventory
checks C1, C3). The slot table of the output (`json.dumps(slots, sort_keys=True)` sha256 `68b8b01d…620e`) was copied after the run into
`slot_mapping_f11.json` (its `note` says so). Board A rules whether this suffices for condition 2 and for the F11.4 gate. This note
does not rule on it, and the calibration stays unaccepted until Board A does.

**F11c.3 — Disclosure: main advanced during the staging run; the manifest's code SHA.** The supplementary staging was launched at
08:06:02 UTC (system clock) from a clean checkout at `8211ecf` and ended at 09:26:07 UTC, exit 0 (`supplement_launch.json`:
`execution_sha` `8211ecf4…`, `dirty_files` 0). The canonical checkout was merged forward during the run: main moved from `8211ecf` to
`887a514` (the merge of `prereg/f11b`, 08:28 UTC). `git diff 8211ecf..887a514` touches the analysis module, its script and tests, the pin
file and package README of re-pin (xxi), `docs/PREREG_H1_TF.md`, `docs/PROGRESS.md`, `docs/prereg/f10_attestations.md`, the root
README and the analysis-4 review. It touches no staging, `prepare`, rotation or audit module. The Python process had loaded its code
at its start: the pinned bytes of re-pin (xx), unaffected by the later merges (re-pin (xxi) already says the run "does not import
the analysis module"). The execution manifest records `code_sha` and `provenance.git_sha` `887a5146…`, because `prepare` reads
`HEAD` at write time (09:26 UTC). The run's identity is therefore `8211ecf` (launch record, freeze), and the manifest's code SHA is
the HEAD at write time. Neither is a change to the staged bytes: the forty's 249 files were re-hashed equal before and after.

**F11c.4 — Disclosure: the run records were enriched after the fact.** The launch and calibration records were first written without
some fields. At `516d9c9` (after both runs) they were completed. `supplement_launch.json` gained the command line, the log path and
log sha256, the launch-script path and sha256 and the output hashes (execution manifest and audit file). `f11_calibration_record.json`
gained the log path and log sha256, the launch-script sha256, the input hashes (`input_provenance`: five files), the mapping-file path
and an `environment` block, which is **empty** (`{}`); the Python, NumPy and thread settings of the F11.4 run are not in the record.
The additions identify things that existed before. They record no new run, and the run, its command and its outputs did not change.
They were made after the fact, not at launch. The launch wrappers, which were run from `/tmp`, were committed at `516d9c9` as
`scripts/part_a_f11_supplement_launch.sh` (sha256 `1dcc133a…92e4`) and `scripts/part_a_f11_calibration_launch.sh` (sha256
`00c3a8bb…b19b`). Both are byte-identical to the `/tmp` files that ran, and were not frozen or committed before the runs.

**F11c.5 — What this note does not do.** It changes no selection (the frozen twelve of `selection_f11.json` and the forty), seed,
decision rule, cap or pin, and makes no re-pin. It authorises no staging, calibration, proof read or scoring. It accepts neither the
staging report nor the F11.4 calibration, and releases no gate of the inventory. Scoring stays held for every Part A entry until
Board A's acceptances, the final methodological freeze and the attestations through it are on file.

*No-exposure attestation (this session, 11 Nov 2026).* The drafter of F11c is Claude Sonnet 5.5, branch `prereg/f11c` (worktree
`.worktrees/prereg/f11c`), delegated by the chair. It read the documents named above and wrote only under `docs/`. It opened nothing
under `data/evals/`, wrote nothing under `data/`, used no GPU and ran no scoring, staging, simulation or test suite. It has seen no
G3, rotation, ladder, reference or sealed outcome or a summary of one; none exists. It chose nothing: the clauses copy the
inventory, Board A's rulings and the chair's threshold. This attests this session only; it is also entry 21 of
`docs/prereg/f10_attestations.md`.
