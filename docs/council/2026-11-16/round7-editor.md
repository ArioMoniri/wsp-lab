1. Develop a scripted weight-statistics residual predictor, not another large encoder.
2. WSR supports a narrow benefit from weights plus tools; it does not isolate reasoning from statistics.
3. Preserve magnitude prediction in TF-v1 nats; report sign and resolution separately.
4. Develop only on already-observed, eligible non-Part-A root edits.
5. Scaling, projection removal and noise need distinct mechanism and source labels.
6. Intentional identity controls must remain separate from failed, rounded-away edits.
7. New WSR-style families belong in a future prospective panel, not silently inside F8/F9.
8. F12 failed qualification; neither more correlated rows nor dropping StatNN rescues that result.
9. Keep F8/F9, H5, F12-I and F5.5 unchanged; Part A retains priority.
10. Any confirmatory purchase requires a new owner-approved registration and passing simulation.

# Editor — 16 November 2026, round 7

**Scope and evidence.** This is a development recommendation, not execution authority. Repository references below use **A5** for `docs/PREREG_H1_TF.md`, **R6** for `docs/council/2026-11-11/SYNTHESIS-r6.md`, and **Board** for `docs/reviews/council/board-a-f12-sim-report-2026-11-14.md`. The prior editor position is `docs/council/2026-11-11/round6-editor.md`. All linked external sources were fetched on 16 November; no fetch failed. No evaluation files were opened.

## What WSR establishes

The fetched [paper](https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models.pdf), also available at the requested [landing page](https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models), reports 0.730 versus 0.578 balanced accuracy, difference 0.152 [0.056, 0.256], on subtle 27B edits. Evidence rests on ten changed interventions producing eighteen changed labels; cells ran once and comparisons were uncorrected. Recall improved while specificity fell. Muse's forty edits produced no detected changes, yet readers forecast changes for 53–88%; normalization explains only some inertness. Welch p<0.01 plus minimum effects (10% tokens or 0.05 rates) defines change; failure to detect is not equivalence. This is encouraging evidence for the combined access treatment, not general reliability or a demonstrated self-reading advantage.

My interpretation: edit-type priors plus computed geometry remain a live explanation. WSR does not test an equal-information scripted-statistics predictor. [Ahmed's audit](https://arxiv.org/html/2611.08070v1) found local arithmetic surviving label permutation while semantic interpretations failed; its small CNNs and tool-free setting do not refute WSR. They motivate anonymized identifiers and information-matched baselines here.

The [reader](https://github.com/suayptalha/wsr/blob/main/wsr/read.py) exposes base and replacement tensors read-only, permits CPU matrix calculations, and prohibits inference/training. The [harness](https://github.com/suayptalha/wsr/blob/main/wsr/harness.py) supports tool loops, direct text-only calls and constant/random baselines. Thus adaptation to our base is feasible in principle, not demonstrated by this review. Our cached field adds previously measured model-specific information; it is not a pure weight-only replication. Field, StatNN and a learned residual are restricted response models, not rollout simulators (R6 §1). H5 additionally tests whether forecasts improve economical verification, not whether explanations sound mechanistic (A5.7, F8.7).

## Edit families: correspondence is not registration

The following algebra comes from the fetched [edit implementation](https://github.com/suayptalha/wsr/blob/main/wsr/ops.py); coverage comes from A5 F8/F8a and F9/F9a.

| WSR mechanism | Relation to our zoo and panels | Development and future role |
|---|---|---|
| Block scaling: D=(f−1)W; f=0 ablates | Not interchangeable with F9-P's coordinate deletion or compensated SparseGPT | Future mild amplification/attenuation on matched supported writers; destructive ablations reported separately |
| Row scaling: selected rows receive (f−1)W | Output-token edits are not F8's selected-column residual edits; embeddings are preserved by F9-P | Reserve vocabulary-row scaling as future family transfer; verify token mapping, tied weights and field support first |
| Direction removal: D=−αuuᵀW, with orientation handled explicitly | Rank-one algebra overlaps abliteration; F8-F's task-vector and residual-contrast directions have different provenance | Develop on eligible observed abliteration roots; future unembedding-derived versus random-direction matched controls |
| Seeded Gaussian noise with prescribed Frobenius norm | Existing random_dir roots are useful controls, not certified equivalents of dense Gaussian perturbations | Future noise matched by realised norm, support and tensor site; independent seed streams |
| Small or unused-row controls | Not guaranteed exact identities; F8a counts rounded-away intended edits as non-surviving disagreements | Add explicit identity controls only in a new panel; retain tiny realised edits as unresolved unless evidence supports more |

The released [paper pools](https://github.com/suayptalha/wsr/blob/main/wsr/paper/pools.py) distinguish block, vocabulary, projection, noise and control families. Their subtle pool selects parents using earlier behavioral changes; dose siblings share parents. Eight pilot agent-edit files are not distributed. Do not treat that selection process as outcome-blind or assume those bytes are available.

For our future panel, freeze parent recipes, sites, dose ladders, directions, matching rules and all failures before target outcomes. Group shared directions, construction items, lineages, parent edits and aliases into connected source components; dose variants do not increase independent N. Match realised bf16 deltas, not intended fp32 norms. Include D=0, attenuation/amplification, direction removal and Gaussian controls; reserve vocabulary scaling and multisite compositions for transfer. Never call an unused vocabulary row behaviorally inert merely because it is absent from prompts: changing its output logit can change normalization.

Existing panels stay exactly as registered: F8-E exact/reconstructed external patches; F8-F tiny untrained gate masks and specified rank-one constructions; F9-P deletion, compensation and within-sparsity ranking. Learned masks and SWEET remain excluded/deferred. F9.6 handles generation separately. F12-I supplies split-edit interaction diagnostics under its own rules; disjoint tensors give zero field interaction, whereas column splits within a tensor can retain quadratic cross terms (A5 F12.6/F12a). The round-six learned-mask priority remains a later, priced challenge, not the next purchase.

## One model to develop

**Scripted response residual (proposed).** Predict TF-v1 WikiText ΔNLL in nats/token. From base weights, realised D and cached field terms, compute eight features in each of four buckets: early/late layers crossed with MLP/attention writers. Features: summed a, summed q, relative Frobenius displacement, changed-coordinate fraction, cosine(D,W), delta-energy-weighted stable rank, realised/intended norm ratio, and delta-energy-weighted depth. Fix unsupported/empty buckets to zero with an explicit eligibility report; a missing required gradient excludes the row from every arm. Stable rank is ||D||²_F/||D||²_2; use a fixed deterministic spectral routine. Zero denominators have pinned zero values and tests.

Use 32 standardized inputs and a width-eight GELU residual: prediction F3(D)−F3(0)+h(x(D))−h(x(0)). Nominal residual parameters: 8×32+8+8+1=273, plus three field coefficients; zero-edit output is exactly zero. Fit L1 with the F5 nested stopping/seed recipe as a development default, nesting field fitting and standardization inside each training split. Compare with zero-fit a, a+q, calibrated field, same-information StatNN, a scripted ridge reader over identical statistics without field features, and text-only forecasts. Field subtraction and all budgets must be identical where applicable.

**Development rows.** A5 F12.2/F12.10 and `docs/PROGRESS.md` correct the brief: 689 primary records (688 entries plus base), not 694; these are not independent development roots. Start from the documented 123 observed roots outside Part A, then conservatively exclude all confirmatory-status components: the documented subset is 79 roots in 52 components, before further eligibility checks. Admit only verified observed development rows; no fresh, status-less, excluded or Part-A-connected rows, no non-root transitions, and no new scoring of the 147 unobserved roots. Eligible families are chain roots, lora_sft, abliteration and random_dir (A5 F12.2). An implementer must reconcile an allowlist before label access; this memo has read no labels.

Use five outer component folds and four inner folds, deterministic hashes and three seeds. Audit duplicate deltas and shared generators before splitting. Report pooled out-of-fold absolute error, per-family errors, and leave-one-family-out diagnostics; scarcity may make family results entries-only. No F8/F9 outcome trains its own forecast. Existing exposed rows support exploration only; all final architecture/tuning choices must freeze before a separate future test.

**Test-first contract:** identity yields zero; scaling obeys the realised delta calculation; transpose conventions agree; compact rows round-trip; no unsupported tensor is silently dropped; parent/dose/alias groups never cross folds; shuffled outcomes remove learnable gain; all baselines receive identical eligible rows. Deny Part A and evaluation-directory mounts in any reader sandbox.

**Labels.** Keep continuous targets primary. A5 F8a.2's floor, independently reproduced as 0.0049397 nats/token (F12a), is a screening reference, not WSR's behavioral threshold or per-entry uncertainty. Report improved/worsened only when paired intervals resolve the sign; otherwise unresolved. Reserve “unchanged” for a prospectively specified equivalence rule or an exact identity. Do not increase accuracy by relabeling low-power observations as no change.

**One alternative:** a bounded WSR-style LLM reader over exactly the same permitted tensors/statistics, paired with text-only descriptions. Neutral framing, opaque candidate IDs, no forward passes or outcomes, machine-readable magnitude/sign/abstention outputs. It must beat the scripted reader to justify its extra cost. Do not develop a large all-row encoder or hybrid feature-generator first: equal-information incremental value is currently unknown.

## Power, cost and governance

Board reports all fourteen gates failed: ordered power 218/1000, lower bound 0.180; simulated gain 0.172 versus field but 0.045 versus StatNN. Field-only detection was itself only 424/1000. The failure combines a small second-comparator margin, ridge-bias stress and undercoverage; increasing N alone cannot repair invalid uncertainty. These are simulated contrasts, not measured edit outcomes.

Reuse the outcome-blind simulator methodology only under a new registration, including the required accounting fixes. Price independent-root expansion, lower-noise targets and a single prespecified StatNN primary comparison with field secondary. The latter changes the claim; it cannot rescue F12. Simulate the margin against StatNN directly, shared-item/base covariance, component dependence, nested fitting, stress and both interval tails. Require calibrated coverage/stress bounds ≤0.07 and the power lower bound ≥0.80 before purchase. Sign targets need their own prevalence/resolution simulation; they do not automatically improve power. No sample size is justified yet.

Proposed exploratory cap: 10 CPU-h, zero GPU-h and zero LLM tokens for the scripted model; stop/reprice if cached inputs are insufficient. Alternative cap: twenty cases, 10,000 total billed tokens each across reader and text conditions, hence 200,000 tokens, plus 5 CPU-h; these are budgets, not measured costs. Full simulation needs a timed pilot: the previous qualification recorded 102.01 CPU-h (Board), so “cheap” means cheaper than buying outcomes, not negligible. Future scoring is unpriced pending panel size and timing.

No registered plan changes. Any confirmation is a new prospective panel after Part A, preserving F5.5 ordering and existing obligations, at the owner's decision. F12 remains closed; F12-I is independent, never a rescue.

| proposal | mechanism | registered test | cost | kill criterion |
|---|---|---|---|---|
| Scripted residual | Geometry plus small learned correction | Exploratory only; future separate registration | Proposed ≤10 CPU-h; 0 GPU-h; 0 tokens | No held-out improvement over StatNN, failed controls, or cap exceeded |
| Reader alternative | Tool-derived prediction versus text-only | Unregistered bounded development comparison | ≤5 CPU-h; 0 local GPU-h; ≤200k tokens | No incremental value over script or poor identity specificity |
| Future matched edit panel | Mild scaling, directions, noise, identities; transfer families | New owner-approved registration after Part A | Pilot-priced CPU and scoring | Power/coverage/stress qualification fails |
| Existing F8/F9/H5/F12-I | Transfer, pruning, verified editing, interactions | Existing rules unchanged | Existing caps only | Existing gates and stop rules |
