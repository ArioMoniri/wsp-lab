# Board A: independent review of PREREG_WME DRAFT v0

Review date: 2026-11-17. Reviewed base: `1547c354f7d646014e5229d8db5c962d63fd7eec`.

**Verdict: ACCEPT WITH CONDITIONS.** All numbered changes below are required before the freeze. This verdict accepts the proposed research question, not the current draft as an executable registration. No confirmatory outcome should be generated under v0.

## Scope and evidence

Read the draft and all seven requested provenance documents: SYNTHESIS-r8, RULING-wme-pilot1, SYNTHESIS-r9, round9-challenger, and the pilot-1, pilot-2 and 27B calibration reports, all dated 2026-11-16. Inspected the relevant `src/wsp/wm/` implementation, especially `model.py`, `panel2.py`, `p2design.py`, `p2data.py`, `plan.py`, `access.py`, actions and construction-only LoRA checks; also inspected the referenced `src/wsp/msrm/closure.py`. This is a document and source audit, not verification of raw experimental outputs. No sealed test file or `data/evals` was opened; no GPU was used. The preregistration was not edited.

The draft gets several important choices right: pilot evidence is exploratory, cumulative-edit prediction is the comparator, H-E2 has one named primary comparator, guardrails and generation are conjunctive, and 27B is not pooled with 9B. The main deficiencies are incomplete separation and operational definitions, missing power inputs, and claim language stronger than the tests support.

## Required changes

### 1. Freeze an item-exposure and split contract (§§0, 3, 4, 10)

The planned normalized-text exclusions are appropriate but do not yet certify disjointness. Specify which split supplies direction construction, LoRA training, field/probe collection, bank labels, predictor tuning, conformal calibration, search feedback and final scoring. Bank-label training must not silently use predictor-validation or search-development outcomes. All transformations of one question, including neutral/suggested pairs, must stay together; define document/problem grouping and a frozen near-duplicate policy where exact normalized hashes are insufficient. Distinguish item separation from trajectory separation.

Publish hash-only exclusion and intersection audits against every pilot split, including construction/training items and known engineering/calibration exposures. Use existing manifests for sealed exclusions, never opening their content. §0's statement that all pilot results are exposed does not make pilot test items exposed: pilot 2 explicitly reports its test unread. §3 can exclude that set from the new panel while reserving it, but must label it an optional, separately registered replication, with no substitution for a failed primary test and no automatic claim of independence from pilot-trained directions/adapters.

Feasibility matters: pilot-2 report §1 gives 1,813 sycophancy questions and uses 1,344 across its four splits. Excluding all of them leaves at most 469 from that source, before any further exclusions. Freeze feasible new split sizes rather than inheriting pilot sizes or relaxing exclusions after inspection.

### 2. Certify root/direction closure across every partition (§§3–4, 6–7, 10)

A new seed namespace is not proof of fresh directions. Specify immutable root, contrast-source, adapter-training, direction and realized-delta identities, and joins for shared ancestry, direction, source and duplicate weights. Scale recipes can recur despite different seeds; several LoRA seeds can share one training source. Register whether source sharing joins components and justify the independence assumption. Apply exclusions against pilots and across outer train/validation/test components and enhancement episodes; declare any permitted shared frozen library and narrow the generalization claim accordingly.

The referenced closure implementation recognizes zoo random-direction and abliteration keys; it does not automatically recognize all new rank-one, contrast and LoRA provenance. Require a WM-specific mapping and audit. Keep every ladder, prefix, reverse order, dose and AB/BA composition in its full source closure. A composition joining two roots merges their components transitively; five folds may be impossible if bundles connect the bank. Publish the planned component graph/counts before outcomes, stratify components rather than pretending multi-family components have one family, and power on the resulting independent components. `model.selection_split` currently randomizes trajectories, not closure groups; inner shrinkage selection and ensemble resampling need the same closure discipline.

### 3. Make the H-T estimand and horizon restriction executable (§§2, 4, 6)

Freeze the state population, unique-state/repeated-visit policy, objective and horizon weights, treatment of stop endpoints, and the exact paired-SE denominator and floor. The current pooled sum weights large components and repeated states more heavily; say whether that is intended or use prespecified component weights. Define bootstrap-t studentization, resample count, fold aggregation, and treatment of zero static error or too few components.

Training on h ≤ 2 must include every learned stage: F3u refits, normalization, auxiliary grounding, shrinkage, model selection and feature selection. Do not train on late ladder states merely because each endpoint can be rebuilt directly from base. Define horizon consistently for ladders and compositions. Reserve any h = 3–4 coverage check to an independent calibration allocation, with no model tuning on outer test outcomes. Freeze algorithms, grids and seeds in the manifest.

### 4. Strengthen and fairly train the static comparator (§§2, 6–7)

Define static as a function of the full realized cumulative edit at θ₀, independent of its decomposition or history. Give it the same candidate-independent base features, cumulative oriented representation, training labels and tuning resources as WM. Capacity and auxiliary-target access must be matched or explicitly ablated; with WM receiving grounded observation/slope supervision, an outcome-only MLP comparison alone cannot identify the value of recurrence. Freeze the StatNN/F3u selection loss, scope and tie rule on inner folds only. Refitting F3u on all bank ladders, as in the exploratory gate, is not permitted on an outer held-out root.

`model.py` currently uses a raw-field nest and a static MLP on cumulative coordinates plus the final bookkeeping scalar; that is not by itself a complete implementation contract for arbitrary six-family 9B edits. Demonstrate representation sufficiency and identical cumulative information, especially for overlapping and bf16-realized compositions. If planning uses evaluated-state observations or online refits, give StatNN identical observations/refits and include the anchor-shift static comparator from r9 C7. A win then supports this frozen model comparison, not an inherent need for memory: r9 explicitly says the final weights are sufficient.

### 5. Specify a valid shuffled-bookkeeping test (§6 H-T; §§6–7 H-E2)

Freeze the permutation unit, eligible states, number/seeds of permutations, paired error statistic, alpha and component inference. Hold the *realized* final edit and target fixed. Simply permuting noncommuting actions can change the final weights, and independently shuffling local norms/overlaps can create inconsistent inputs. Degradation on malformed, out-of-distribution inputs establishes sensitivity to corruption, not learned dynamics. Separate that diagnostic from a valid fixed-endpoint control, and include consistent alternate decompositions/split-step invariance where available. Show the cumulative static prediction remains unchanged.

Preserve the adopted failure consequence without claiming that a passing corruption test proves dynamics. r9 C2 requires controls for trajectory **or E2** claims; v0 attaches its explicit void rule only to H-T. Enumerate which control gates each claim and propagate the relevant failures to E2.

### 6. Operationalize inert edits and trust-region strata (§§0, 4, 6)

Name the exact tensors, construction and checks that establish inertness. A row unobserved on a small probe panel is not necessarily dead, and RMSNorm epsilon, surrounding operations and bf16 rounding can invalidate exact cancellation. Require mathematical invariance or independent paired behavioral verification at a prespecified tolerance. Freeze control count, endpoint coverage, simultaneous pass rule and an absolute calibrated tolerance as well as |ŷ| ≤ 2σ; otherwise inflated uncertainty trivially passes. Identify whose σ is used and require the uncertainty checks first.

Correct “pilot ladders never left the trust region”: pilot 2's maximum 0.127 exceeds v0's 0.1 boundary. It did not reach the 1.0 stopping boundary and scarcely sampled the outer region. Freeze the data split/measurement used to assign strata, component minimums, and behavior when a stratum is sparse. Report both strata with uncertainty and a nondestructive subset. A pooled win driven by damaging states alone must be described as such, not as useful enhancement or general trajectory understanding.

### 7. Freeze E1 selection and the generation audit (§§2, 3, 6)

Identify the single E1 WM winner versus base, how it relates to the multiple E2 episode winners, and its development-only selection/tie/fallback rule. “One winner per arm” and “episodes” currently leave room to choose the most favorable test result. Freeze all episode winners before any final-test evaluation; no new search after seeing another episode's test score.

Define the generation item count, allocation, source grouping, prompts, answer parser/rubric, token/stop limits, treatment of truncations, loops and refusals, and paired statistical test. “Positive gain” is only a point-estimate criterion; require a prespecified positive lower bound for generated correctness/non-sycophancy and uncertainty-aware anomaly bounds. Specify whether ≤ 2% means absolute incidence or increase versus base, and its denominator. Greedy decoding is a prospective departure from r8's replicated decoding; justify and freeze it rather than treating one deterministic audit as equivalent. Define corresponding generation endpoints for GSM8K and math secondaries.

The neutral-minus-suggested log-probability gap can shrink by destroying neutral correctness. Register separate neutral and suggested accuracy/probability reporting and a neutral-performance preservation rule so “less sycophantic” cannot mean equally wrong under both prompts.

### 8. Declare multiplicity across claims and analyses (§§2–3, 5–6, 10)

Freeze the inferential family and alpha allocation or hierarchy for H-T, H-E1 and H-E2, objective secondaries, nine E2 secondary arms, controls and any 27B confirmation. Distinct scientific claims can have separate families, but then prohibit an unadjusted “any claim succeeded” study-level assertion. State exactly which p-values enter each Holm family and when tests are gated. Conjunctive benefit/guardrail/generation requirements do not themselves require a union correction; selectable winners, objectives or successful contrasts do. Keep trust-region and leave-family-out results descriptive unless separately allocated alpha. Specify one-/two-sided bounds consistently: H-T says one-sided 95%, E1 just says paired 95%, and E2 gives no level.

### 9. Define guardrails on explicit scales and populations (§§3, 5–6)

Enumerate the complete guard set per objective, including competition math, which is not one of the six trajectory endpoints, and which primary endpoint is excluded from its own guard set. Specify per-item transform **before** pairing/averaging: exp(summed gold log-probability), sigmoid(refusal margin), and exp(neutral) − exp(suggested) for sycophancy, following `panel2.pp_value`. Pin token aggregation, orientation, item weights, bootstrap group and simultaneous Bonferroni count/alpha.

This pp-equivalent reading was implemented before pilot-2 data, but is not an accuracy guarantee. Long exact-answer sequence probabilities can be tiny, making a one-point absolute probability margin permissive even for a large log-probability loss. Refusal phrase probability is also not actual refusal behavior. Prospectively justify these margins, report native scales and generated safety/capability outcomes, and either add meaningful preservation thresholds or restrict the claim to these surrogate guardrails. Never retune margins after new results. Preserve the report's sensitivity disclosure: GSM8K had 3 qualifiers under pp-equivalent bounds and 1 under the alternative value-scale reading.

### 10. Freeze E2 fairness, episodes and complete accounting (§§6–7)

Name the three families, define fresh episode roots and seeds, candidate library/pool, adaptive proposal rules, base fallback, rejection and tie rules, search/refit schedule and failed-episode handling. Specify the test-score scale and how infeasible winners enter the contrast; E2 must not reward an unsafe edit or drop losing episodes. Pair inference across independent episode/source components and account for common test items and shared directions: a seed alone is not an independence unit.

Define a real-evaluation bundle, including screen fractions, intermediate visits, repeated candidates, checkpoint selection, validation, oracle derivatives and generation. `BudgetLedger` caps entry counts but `report()` labels budgets identical even with unequal usage; it neither defines bundle equivalence nor enforces information parity. Specify matched uncertainty penalties and a symmetric ranks-only fallback if coverage fails, without silently dropping final guardrails. Resolve “must pass before any analysis” versus the allowed coverage-failure continuation. Preserve r9 C6's synthetic planner comparison/fallback and freeze its result before real search.

Separate rankers sharing a candidate pool from steering and SFT competitors whose action spaces differ. State matched construction-data/information access and checkpoint budgets. Report full bank, direction/adapter training, grounding slopes, features, failed attempts, planner compute, LLM tokens, evaluation and generation costs, with an explicit amortization denominator. E3 is descriptive under r9 C2; equal 32-bundle superiority does not establish total-compute superiority. Disclose information/training exposure for reader comparisons; a Fugu-style reader is not a replication of Fugu.

### 11. Replace the incomplete power specification (§§3–4, 6, 8–9)

§8 has no E1 effect scenario and no output setting item/test/generation Ns. A pilot SE alone does not determine covariance, guardrail feasibility or selected-winner variability. Specify a scientifically meaningful E1 benefit, generation transfer, distance to each noninferiority margin and anomaly rates. Freeze how the exploratory E2 SD is estimated without favorable candidate selection; 0.5 × that SD is an assumption, not measured guidance efficacy. C_T = 0.15 is also an assumed effect. Add smaller effects, no generation transfer, boundary guardrail/anomaly nulls, heavy tails, correlated objectives/items, unequal closure sizes, weak trust-region strata and shared-episode sources. Pilot winner effects must not be treated as unbiased alternatives.

Simulate the actual nested selection, folds, ratio statistic, bootstrap, multiplicity, conjunctive gates and ranks fallback. Calibrate false positives under claim-specific nulls: WM = static is an H-T null, not a zero-benefit E1 or boundary-unsafe null. Set a grid for independent components, episodes, each item split, generation items and B; resolve fixed 96 roots/B = 32 versus simulation-set values. Publish code, seeds, replicate counts, binomial confidence method and Monte Carlo uncertainty for size/coverage as well as power. Explain size ≤ 0.07 as a simulation tolerance for a specified nominal alpha, not permission to replace 5% tests. No simulation or N retuning on confirmatory outcomes.

Reprice all retained observations required by the frozen model. r8 grounding includes action derivatives; the 7.84 GPU-hour pilot price excludes per-state oracle gradients, while the report prices their inclusion at 24.07. Passing the trapezoid oracle gate does not make those labels free or prove learned R0 transport works. State which grounding targets are retained, their collection cost, wider-ladder/control/generation cost and the stopping policy before outcomes. Under budget infeasibility, retain “unresolved — not powered”; do not shrink N until a run fits and claim confirmation.

### 12. Disclose adaptive history and freeze the ×8 design (§§0, 3–4, 8)

Selecting `sycophancy_real` is outcome-dependent development but follows RULING §3's predeclared most-qualifiers rule: counts 10/3/4 select it without a tie. That is acceptable for a fresh confirmatory panel; it is not an independently selected endpoint before all pilots. Explain pilot 1's sycophancy recommendation and pilot 2's prospective selection, retain failed pilot gates, and do not treat selection/headroom as confirmation.

The ×8 dose is a new post-pilot design choice, not part of r8 or the pilot-2 declaration. It can be registered prospectively now. Freeze all six families' nominals (pilot 2 only specifies three), caps, signed realization, duplicate-state rules, composition doses and side-stop order. The positive projection cap already duplicates larger doses; adding ×8 need not add a distinct state. Freeze the design without repeated widening until a desired effect appears, and include its destructive-state sensitivity and cost in the power plan.

### 13. Complete 27B separation and exposure handling (§§0, 10)

The calibration report is engineering only: no scientific gates or generation endpoint were established. Exclude its construction/validation/timing exposures and edits, along with all earlier exposed items, from any claimed fresh 27B confirmation using manifest identities. A fresh salt or namespace alone is insufficient. Specify whether 27B tests independent-data replication or deliberately shared-item replication, and prohibit transfer of 9B test outcomes into 27B tuning.

Choosing 27B's own primary by its headroom gate tests a new selected objective if it differs from 9B; call that a separate experiment, not replication of the same primary enhancement claim. Freeze 27B's pilot-only selection, final panel, model/metric rules, power/Ns and amendments before its confirmatory outcomes, with explicit separation from any 9B-informed changes. Retain scientific gates and engineering limits, including the two-layer field collection configuration; the earlier timing calibration cannot replace them.

### 14. Prevent null-to-success reinterpretation (§§0, 6, 11)

Replace the unconditional §11 statement “Static cumulative prediction suffices” with “WM superiority over the registered static comparator was not established” when the interval crosses zero. That is not an equivalence test; a C2 failure instead invalidates the specified inference and does not establish static sufficiency. Distinguish benefit failure, guardrail failure, invalid control/calibration, insufficient power and engineering abort. Pilot-1's historical ruling can remain as a quoted exploratory consequence.

Keep E1 success separate from E2 success: an improved edit with no guidance advantage supports enhancement only, and E2 without E1 cannot be promoted to verified enhancement. Secondary wins, a descriptive E3 advantage, a favorable stratum, a later objective or 27B must not rescue a failed primary. Publish all registered contrasts, controls and adverse results. Append-only amendments made before some later data are still adaptations if earlier confirmatory results motivated them; preserve the original claim and label the new analysis/experiment accordingly.

## Freeze acceptance check

Acceptance requires a revised registration and hashed executable manifest resolving conditions 1–14, plus the outcome-blind simulation report and feasibility accounting. Source inspection shows useful infrastructure, not completed enforcement of this contract: current σ lacks the required residual floor, StaticPredictor has no σ, and inner selection is trajectory-based. The already required C1 fixes and closure-aware checks must be demonstrated before use. No conclusion in this review relies on the challenger's unsupported prediction that competing systems would outperform WM-E.
