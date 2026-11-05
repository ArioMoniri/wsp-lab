# Board A adjudication: TF-v1 qualification result, power and cost (main @ 035ee62)

5 November 2026. Reviewer: Board A (strict ML). **Verdict: the programme proceeds, but not as registered. Four prospective changes (amendment A2, wording below) are required before any further scoring; H1a is exploratory by default and becomes confirmatory only if a re-run §5 gate passes.** This review authorises no compute. It was written against [the qualification report](tf-v1-qualification-2026-11-05.md), [PREREG_H1_TF](../../PREREG_H1_TF.md) (A1, §§5, 6, 9), the sealed [partition manifest](../../prereg/panel_manifest.json), [RESULTS](../../RESULTS.md) (G1, G2, batching probe), the [council synthesis §3–4](../../council/2026-11-05/SYNTHESIS.md), my [P4 rule](../P4/board-a-synthesis-and-v2-2026-11-05.md) and my [re-review](board-a-tf-v1-rereview-2026-11-05.md).

What I did and did not do. I read the documents and the scripts (`build_h1a_panel.py`, `tf_v1_qualification.py`, `tf_v1_power.py`, `tf_v1_qualification_panel.py`). I ran CPU-only, metadata-only checks on the committed manifest, the zoo manifest, `delta_hashes.json` and the committed `qualification_report.json` and `power_report.json` (scratch scripts in `/tmp/boardA_adj`, nothing in the repo or `data/` modified). I did **not** re-run the qualification or the power simulation, and I did not touch the GPU. Every number I quote is from those committed reports or from the checks listed in §5.

## 0. Decisions at a glance

| # | Question | Decision |
|---|---|---|
| 1 | MMLU-Pro per-entry δ/2 | **(b)**, amended prospectively: panel-mean bound + hard ceiling h ≤ δ + stratum-weighted 20% tripwire (A2.1). δ is **not** widened. The 20-entry result is reported as passing the amended rule but is **not counted as qualification**; the registered qualification is re-run on the D3 stratum entries under the final source. |
| 2 | KL | A1.2 outcome stands (**not qualified**, tripwire tripped at 25% vs 20%). KL becomes a **descriptive secondary**; no redefinition, no further qualification effort. |
| 3 | Power / status | A confirmatory claim at 100 components is **not admissible on these numbers**. H1a is registered **exploratory (power-limited)** and is upgraded only by a re-run §5 gate (D3 OOF losses, joint procedure, size-controlled interval). The percentile bootstrap is replaced by a cluster **bootstrap-t on ln(ΣA/ΣB)** (BCa as the pre-declared fallback). |
| 4 | Cost | **Approve the batching change** (A1.5), with a measured pilot gate of warm median ≤ 1.40 min. Expected total ≈ 15.5 h against 19.0 h. Fallback, in order: drop refusal stages (config only), then subsample null-family D3 rows. |
| 5 | Partition / scripts | **No blocking inconsistency.** Closure, ranks, ids, the 101 aliases and the non-null definition reproduce. Six non-blocking items must close before scoring (§5). |

## 1. MMLU-Pro gold log-prob: the per-entry δ/2 rule

**Decision: (b), with exact wording and tripwire below. Not (a). Not (c).**

### Why not (a), keep A1.3 as written

(a) is internally consistent, and I would accept it if the owner prefers it: the endpoint would be unqualified and H1a would stop with measurement results (§9, A1.4). I do not recommend it, for four reasons.

1. The per-entry clause is not in the rule I approved. My P4 wording says "upper 95% paired item and repeat-error bounds ≤δ/2 on a variable representative panel": a panel-level bound. The per-entry reading entered through A1.3 as a same-day "clarification" after I had flagged the question as unregistered. It is a legitimate strict reading, but it is not the only one.
2. The two failing entries are the two largest-effect LoRAs by construction (the §6 rule picks the two largest legacy-KL LoRAs). `b5d0ac5d76c3cad6` has y = −0.80 with half-width 0.089 (11% of |y|); `c3ae3e8c8edca6b8` has y = −0.168 with half-width 0.061 (36% of |y|). The half-width comes from item heterogeneity (implied per-item SD of the paired difference ≈ 1.44 nats for the first), not from repeat noise (repeat error is exactly 0). A hand-assembled panel with extremes over-sampled by rule (19 distinct components for 20 entries) should not decide the programme alone.
3. The decisive information is available at near-zero marginal cost: all 229 D3 stratum entries (139 chain, 60 lora_sft, 30 task_vector) will be scored under the final source anyway, and their half-widths come for free from the joint item draw.
4. Item sampling error in a fixed-panel label is not measurement error for the conditional claim (the label is exactly reproducible); it matters for the item-population claim, which §4 already covers with the joint component/item-block bootstrap.

### Why not (c), widen δ

Passing would need δ/2 ≥ 0.0893, so δ ≥ 0.179. That is outcome-tuned by arithmetic. It also loosens the gate for the roughly two thirds of stratum rows that are chain steps (|y| ≈ 0.001–0.004 in the pilot, half-widths ≈ 0.003). δ = 0.10 stays. A different δ would be a new endpoint version with its own registration, not an amendment.

### Why not "scope by |y|" (A1.2's device)

A cut at a multiple of δ does not rescue the panel: `c3ae…` sits at 1.7 δ and `b5d0…` at 8 δ, so only a multiple ≤ 1.6 would pass, which is plainly fitted to the failure. A relative rule h_i ≤ κ|y_i| needs κ ≥ 0.36, also fitted. Rejected.

### Wording (A2.1, to replace the first clause of A1.3; the rest of A1.3 and the repeat and reliability bounds stand)

> **A2.1 Item-uncertainty condition for MMLU-Pro gold log-prob and WikiText log-PPL.** For panel entry i let h_i be its 95% half-width from the joint item draw (2.5/97.5 percentiles, 4,000 draws, seed 20261105, one common draw across edits and the shared reference; WikiText blocks of 2 windows). The endpoint satisfies the item-uncertainty condition iff all of the following hold on the qualification panel:
> (i) *panel mean:* the 97.5th percentile of the 4,000-draw component-bootstrap of mean_i h_i is ≤ δ/2 (unchanged from A1.3);
> (ii) *ceiling:* max_i h_i ≤ δ;
> (iii) *tripwire:* P ≤ 0.20, where P = Σ_f w_f p_f, f ∈ {lora_sft, task_vector, chain}, p_f is the share of family-f panel entries with h_i > δ/2, and w_f is family f's share of stratum rows in the sealed partition (metadata only; chain 161/241 = 0.668, lora_sft 55/241 = 0.228, task_vector 25/241 = 0.104 at the 100-component partition, recomputed from the final manifest). For the registered qualification P is the 97.5th percentile of the family-stratified component bootstrap; for interim reports it is the point value;
> (iv) *disclosure:* every entry with h_i > δ/2 is listed with y_i, h_i and h_i/|y_i|; every H1a result is reported with and without those rows (descriptive leave-out sensitivity), under both the component-only and the joint interval.
> Failure of (i), (ii) or (iii) leaves the endpoint unqualified (H1a untested). The A1.3 per-entry result (every entry ≤ δ/2) is still computed and reported descriptively. **The registered qualification panel is the stratum-family D3 entries (lora_sft, task_vector, chain) scored under the final `source_hash`, with the §6 repeat bounds from the repeated G1**, the reliability bound recomputed with the nuisance refit inside each bootstrap draw. The 20-entry development panel is an interim report and does not qualify the endpoint under A2.1.
> **Fresh-stage audit.** After scoring, the same half-widths and P are computed on the sealed items for the fresh rows. If P_fresh > 0.20 the item-population claim is withdrawn and only the conditional (fixed-item) claim stands, labelled as such. This audit cannot create or remove qualification.

Constants. The 20% is the A1.2 constant, reused so that no second threshold is tuned after seeing a 10% exceedance rate; δ is the registered effect. I note that 20% is lenient as a row count, and that it is the exceeding rows that carry most of the loss mass (the two over-limit entries hold 85% of the squared residual in the 20-entry panel). That is why (ii) and the mandatory leave-out sensitivity in (iv) carry the weight, not the tripwire.

### What the 20-entry panel scores under A2.1 (interim, not qualification)

(i) 0.0312 ≤ 0.050 pass; (ii) max 0.0893 ≤ 0.100 pass; (iii) P = 0.228 × (2/6) + 0.104 × 0 + 0.668 × 0 = 0.076 ≤ 0.20 pass. Reliability: point 0.991; dropping `b5d0…` gives 0.975 and dropping both over-limit entries gives 0.983 (my recomputation from the committed report), so the point estimate is not carried by the two LoRAs. The 10-entry composition's lower bound (0.770) and the in-sample nuisance (not refit per bootstrap draw, 19 clusters for 20 entries) are why the lower bound cannot be trusted at n = 20. These passes were obtained after the failure was known and are reported as such in A2.

### Evidence that would reverse this decision (to (a): unqualified, H1a untested)

- On the D3 stratum under the final source: panel-mean bound > δ/2, or any h_i > δ, or P (upper 97.5%) > 0.20, or reliability lower bound < 0.90 with the refit nuisance, or any repeat discrepancy > 1e-4.
- Exceedances that are not confined to large-effect LoRAs (for example chain or task rows with h_i > δ/2), which would point to a scorer problem rather than a property of the edit.
- A new finding that the item-population claim matters more than I have weighed it (for example, if the owner wants the abstract to claim generalisation to items, then (a)'s strict per-entry reading is the safer registration and the cost is H1a untested).
- I would never reverse towards (c): a different δ is a new endpoint.

## 2. KL

**Decision: A1.2's outcome stands; KL is a descriptive secondary.**

- A1.2 tripped (5/20 = 25% above 0.10 nats vs the 20% tripwire) and one in-gate entry (`b5d0…`, KL 0.0699, half-width 0.0103 > 0.005) fails the per-entry bound. A1.2 already said the 10δ threshold was the wrong shape: half-widths scale with the entry's own KL (relative width 0.05–0.15 over the entries above 0.03), so an absolute δ/2 cannot hold for large edits. I do not repair it retroactively.
- Do **not** redefine the endpoint to rescue it. Shorter prefixes cannot help: the pilot's prefix medians (16, 32, 64 tokens: 0.00144, 0.00153, 0.00160) are flat, so the half-width is set by prompt-level heterogeneity, not by positions per prompt. First-token-only would raise variance and force a cache rebuild and a source change for an endpoint that cannot pass H1a.
- Registered wording (A2.2): *"KL(base‖edited) is a descriptive secondary. Its A1.2 qualification outcome (unqualified, tripwire 25% > 20%) is reported as recorded. KL carries no pass/fail claim anywhere and is reported per entry with absolute and relative (half-width/KL) uncertainty. KL, task change and any other outcome must not be used to include, exclude or reweight rows of the H1a panel; any destructive-edit sensitivity analysis is descriptive."* The last sentence matters: PREREG §1 requires zero-effect and destructive rows to stay in the denominator, and an outcome-based filter would be selection on the outcome.
- Keep the stage in the scoring path: it costs 7.7 s per entry, uses the verified cache, and is the only validity signal for destroyed models.

Evidence that would reverse it: a decision by the owner to make KL a qualified validity endpoint (for example for a capability-preservation claim). That would be a new KL endpoint with a **relative** precision criterion (for example h_i ≤ max(δ/2, c·KL_i)) designed on D3 data and registered with its own tripwire before fresh scoring. It is outside this programme's cap and I do not recommend spending on it now.

## 3. Power: confirmatory or exploratory, and the interval procedure

### Admissibility

**A confirmatory claim at 100 components is not admissible on the current numbers.** The planning report is proxy-based (no D3 OOF TF-v1 losses exist) but it is already sufficient to refuse a confirmatory label:

- Lower-bound power at θ = 0.15 is ≥ 0.80 in one of six scenarios (A, 0.870) and 0.29–0.53 in the other five. The component sweep for scenario C reaches only 0.587 at 200 components. The partition cannot reach 200 (maximum 106).
- The percentile bootstrap is not size-controlled: null rejection 0.046–0.117 against a nominal 0.025 one-sided. The power figures therefore include false rejections; a calibrated test has less power than the table says.
- The §5 gate requires the lower binomial power bound ≥ 0.80 for **both** the conditional and the joint procedure. The joint procedure was not run.
- Even in scenario A the registered precision condition (half-width ≤ θ̂/2) is met in 48% of panels and the full decision rule in 42%. I add a point the report does not make: §5's "power at 0.15" is the power of the rejection test, not of the §4 decision rule. At a true θ exactly at the 0.15 threshold, P(θ̂ ≥ 0.15) is about one half, so the decision rule cannot exceed roughly 50% success there. The gate report must also show P(decision pass) at true θ of 0.20 and 0.25 so readers can see what the study can detect.
- The loss mass is concentrated: in the 20-entry proxy, two entries hold 85% of the squared residual and 57% of the absolute residual, and 161 of the 241 stratum rows are chain steps with near-zero effects. The row-summed ratio is effectively driven by the 80 lora_sft/task_vector rows. The effective number of independent units is well below 100, which is the mechanism behind the size distortion.

**Decision.** H1a is registered as **exploratory (power-limited)** by default. It is upgraded to confirmatory iff the following are all met in a committed development-only report **before any fresh component is scored**; otherwise it stays exploratory and the §4 pass/kill labels are not applied (estimates and intervals only, wording "estimated" or "not distinguishable", no kill claim).

> **A2.3 Status rule and interval procedure.**
> *Gate (all required):* (a) on the D3 out-of-fold TF-v1 losses (empirical comparator loss, empirical within-component and shared-item covariance, actual component sizes) the lower 95% binomial power bound at θ = 0.15 is ≥ 0.80 for **both** the conditional and the joint component/item-block procedure, using the interval procedure below; (b) under the null (θ = 0), one-sided rejection ≤ 0.035 and, at θ = 0.15, two-sided 95% coverage ≥ 0.93, in the empirical scenario and in the variance-inflated sensitivity scenarios, for both procedures; (c) fewer than 0.1% of draws with a denominator failure; (d) qualification (A2.1) passes; (e) the report also gives P(decision pass) at true θ = 0.15, 0.20 and 0.25.
> *Primary interval (replaces the §4 percentile bootstrap):* cluster **bootstrap-t on ln R**, R = ΣA/ΣB over rows, resampling source/ancestry components with replacement at the original count, 4,000 draws, seed 20261105. Per replicate compute t\* = (ln R\* − ln R̂)/se\*, with se\* the linearised cluster-robust standard error computed on that replicate: with component influence u_c = A_c/ΣA − B_c/ΣB, se² = n/(n−1) Σ_c u_c². The interval for ln R is [ln R̂ − q₀.₉₇₅(t\*)·se, ln R̂ − q₀.₀₂₅(t\*)·se]; the θ interval is 1 − exp of its endpoints, reversed. The joint procedure uses the same construction with the two-way (component × sealed item-block) cluster-robust standard error V = V_comp + V_item − V_both. Reported alongside, not decisive: BCa (acceleration from the delete-one-component jackknife) and the registered percentile interval.
> *Fallback order, fixed now:* if the primary fails (b) in the simulation, BCa is tried against the same criterion; if neither passes, H1a is exploratory. The choice is made on simulated data only, never on fresh outcomes.

Why bootstrap-t on the log ratio and not BCa first. The statistic is a ratio of sums with heavy-tailed cluster totals and about 100 clusters. A studentised pivot on the log scale removes first-order skewness from both the ratio and the cluster weights, and it errs towards wider intervals when a few clusters dominate, which is the failure we need to guard against. BCa corrects bias and skewness but needs a stable jackknife, which a handful of dominant clusters undermines. The studentiser is a closed-form delta-method quantity, so the cost is the same as the percentile bootstrap.

**Must the §5 gate be re-run with it before the panel? Yes, twice.** (1) Now, on the existing proxy grid (CPU only, no GPU-hours): confirm that at least one of the two procedures can control null rejection at ≤ 0.035 under the heavy-tailed scenarios D and F. If neither can, expect exploratory status and tell the owner before D3 scoring is committed. (2) Binding: on the D3 OOF losses after D3 is scored and fitted, with the joint procedure. §5 already requires the gate "before fresh outcomes"; I add that its interval procedure is the A2.3 one, not the percentile.

Smaller points for the gate report: report the effective cluster number n_eff = (ΣB_c)²/ΣB_c² and the share of ΣB held by the top five components (diagnostics, not thresholds); and note that the D3 fit uses 229 stratum rows plus 160 null-family controls, so the StatNN-versus-isotonic comparison rests on a small training set, which belongs in the limitation language.

### Expand the confirmatory set to 106

Six fresh chain components remain unselected (26 stratum rows, 26 unique deltas, all in hash order after the 100th). §5 permits "outcome-free closure expansion within cap before scoring". Adding them costs 26 × 1.08 min ≈ 0.47 h, raises rows from 241 to 267 and, more importantly, removes the zero margin above the ≥ 100 valid-component floor (a single post-freeze exclusion otherwise invalidates confirmation). Re-seal the partition (new `partition_sha256`) before the pilot, with the 6 listed as chain-fill by rank. Power gain is small (about 0.01); the value is the floor margin.

### Evidence that would reverse this decision

- Towards confirmatory: the D3-OOF gate in A2.3 passes (that is the stated route, not a reversal).
- Towards stopping before fresh scoring: the D3-OOF gate gives a conditional lower power bound below 0.50 for every interval procedure and the null size cannot be controlled. Then the 5.3 h of fresh scoring buys only an interval estimate; the owner may still choose to buy it, since a measurement paper is an accepted outcome (SYNTHESIS §5), but it should be bought knowingly.

## 4. Cost

**Decision: approve the batching change (A1.5) as a source change, with a measured gate; do not cut development rows now.**

Arithmetic (all from the committed numbers): 613 entries remaining × 1.805 min = 18.44 h; plus 2.5 h already consumed and ≈ 0.5 h process overhead = 21.4 h against 19.0 h. Batched: 613 × 1.08 min = 11.0 h; plus 2.5 h consumed, ≈ 1.0 h for the new pilot, G1 and G2, and ≈ 0.5 h overhead = **≈ 15.0 h**; with the six spare components (0.47 h) ≈ 15.5 h. Break-even warm median for 19.0 h with no slack is (19.0 − 2.5 − 1.0 − 0.5)/613 = 1.47 min, so the gate below leaves about 0.9 h of retry slack at the line.

Why batching and not cutting rows. The change touches only HellaSwag (descriptive, 250 items) and the refusal/over-refusal phrase stages (uncalibrated margins that the registration already says are not a refusal measure). MMLU-Pro, KL and WikiText stay batch 1, so the primary endpoint's semantics are unchanged and checkable bitwise. Cutting development rows reduces the predictor's training set (D3 stratum is only 229 rows) and departs from §3's "all eligible D3 families". The cost of batching is about one hour and the loss of the old artefacts as anything but development evidence, which is what they already are.

**Exact scope (A2.4).**

1. *Source change.* HellaSwag endings (4) and the 13 refusal/compliance phrases per prompt are scored in one right-padded forward per item or prompt, with the actual batch recorded in each record. No edit to the letter scorer, `kl_values`, the WikiText path, or the four `SCORER_HASH` functions (`last_logits`, `generate_entry`, `entry_digest`, `content_root`), so the KL cache stays valid (root `2ed18bda284b3a89…` must be unchanged; no 5.4 GiB rebuild). Tests: batched-versus-batch-1 deviation within the probe maxima (HellaSwag ≤ 0.010, phrase log-prob ≤ 0.39, margin ≤ 0.30), bitwise repeat of the batched path, and a diff check that the primary paths are untouched.
2. *Freeze.* Commit, record the new `source_hash` in A2, generate no reference before the pilot. Artefacts under `12ee875a68c9` are development-only and are not mixed with new labels (provenance requires the same `source_hash`). The 23 entries already scored are re-scored under the new source by the pilot and G1.
3. *New warm timing pilot* (base plus the 10 D3 controls, one process), ≈ 0.3 h. Pass: warm median ≤ **1.40 min**; MMLU-Pro, KL and WikiText records bitwise-equal to the old primaries; HellaSwag and phrase deviations within the probe limits. Bitwise equality of the unchanged paths is what lets the existing MMLU/KL/WikiText item evidence be carried as an identical measurement; the registered qualification is nevertheless re-run (§1).
4. *G1 repeat* on the §6 composition (13 entries plus base, primary and repeat in fresh processes), ≈ 0.6 h. Pass: max discrepancy ≤ 1e-4 (KL 1e-5), expected exactly 0.
5. *G2 re-run* on the final source, ≈ 0.05 h, and the bridge script must call the final batched HellaSwag path (otherwise it tests the old source). The registered G2 criteria concern MMLU-Pro only and are expected to reproduce unchanged. Note that the worst entry's argmax agreement is exactly 99% against a 99% limit (a bf16 tie, granularity 1/100); a repeat showing 98% would fail on a single tie, and the amendment should say that a one-item shortfall is investigated, not automatically a stop.

**Fallbacks, in order, if the pilot gate fails or any unchanged path is not bitwise-equal:** (1) drop the refusal and over-refusal stages by config (−24 s, 1.40 min; 613 entries ≈ 14.3 h, total ≈ 18.2 h with the pilot and overhead: fits with about 0.8 h slack); (2) additionally subsample null-family D3 rows: abliteration 100 → 20 and random_dir 60 → 30 (110 entries ≈ 3.3 h at 1.805 min), hash-ordered with the `20261105:inner:` domain, no lora_sft/task_vector/chain row removed. (3) §9: stop with measurement results. These follow §9's order ("reduce only optional secondary scope", never primary items or components).

Evidence that would reverse the decision: a pilot median above 1.40 min; any difference in MMLU-Pro/KL/WikiText records against the old primaries (an unintended source change: fix or revert); a non-repeatable batched path; or an owner decision to raise the cap (A1.4(c)).

## 5. Partition builder and qualification scripts: spot-check

**No blocking inconsistency.** What I reproduced from the committed manifest, the zoo manifest and `delta_hashes.json`:

| Check | Result |
|---|---|
| Component id = sha256(json.dumps([members, source_identities])) | 483/483 reproduce |
| Rank = sha256("20261105:H1a:" + id); list sorted by (rank, id) | 483/483 reproduce; sorted |
| Closure | 1,247 zoo entries, each in exactly one component; 0 duplicated members; 0 source identities shared across components; 0 parent edges and 0 generator ids crossing components |
| 101 aliases | 101 groups; canonical id is the lexicographic minimum in every group; every group lies inside a single component. Composition: 51 chain + random_dir, 46 abliteration + chain, 2 random_dir pairs, 2 groups with two chains sharing one delta (merged, which is conservative) |
| Selection reproduces §2.5 | 71 components with candidate lora/task, plus 29 chain-fill in rank order; no skipped fresh stratum component ranks below the last chain-fill; 6 fresh chain components remain (the 106 maximum); all 100 are non-null; **0 confirmatory members have a D3 evaluation** |
| Non-null definition | `has_stratum` = any member family in {lora_sft, task_vector, chain}, outcome-free; 106 fresh components with no stratum row are correctly excluded; 241 rows (chain 161, lora 55, task 25) |
| Development | 252 components, all 389 D3-evaluated entries inside them; 414 development members are unevaluated (335 chain) and never scored |
| Qualification panel | 20 entries fall in 19 distinct components (the bootstrap unit is therefore nearly the entry) |

Non-blocking items to close before the pilot or before fresh scoring:

1. **Alias fan-out.** For 54 of the 101 alias groups the canonical (scored) id is a standalone random_dir or abliteration, not the chain step-0 that is the stratum row. Label and reference lookups for the chain row, and for step-1 chain rows whose predecessor is the aliased step, must resolve through the alias ledger. Add a test, and a dry-run scoring list that confirms 381 unique D3 deltas and 255 (281 with the six spares) fresh deltas are scored once each.
2. **Seal on the right hash.** `manifest_content_sha256` includes `registration.git_head`, so it changes on any commit; `partition_sha256` (currently `df4689983f1b…`, to be recomputed after adding the six components) is the stable seal. Record it in A2.
3. **Task-vector reference convention.** The qualification script and the report use the base as reference for non-chain entries. If task-vector rows are to be parent-relative, 14 unique unevaluated parents of D3 task vectors need scoring (+0.25 h at 1.08 min). Freeze the convention in the manifest.
4. **Sealed item panel not built** (`TF-v1:item:` domain, text-group dedupe, disjointness). Fresh rows are scored on sealed items, so the dev-panel qualification does not cover them (see the fresh-stage audit in A2.1). Decide now that the secondary panels (HellaSwag, KL prompts, WikiText, phrases) keep the development panels as descriptive endpoints, so no new KL cache is needed.
5. **Qualification-script limits for the re-run.** The reliability bootstrap does not refit the nuisance regression per draw (in-sample residuals with 6 parameters on 20 points) and uses cumulative, not incremental, norm; the §6 expansion used the ten controls rather than the prespecified hash order. All three are fixed by the D3-scale re-run and should be stated in A2.
6. **Power script limits.** The proxy resamples from 20 in-sample residuals with a CV of 1.97 and lacks the joint procedure; the proxy re-run in §3 and the D3-OOF gate replace it.

## 6. Order of work and hours

| Step | GPU h | Gate |
|---|---:|---|
| 0. Commit A2 (A2.1–A2.4), batching source change, re-seal partition at 106, alias test, sealed item panel build, proxy re-run of the interval procedures (CPU) | 0 | Board A reads A2 before any scoring |
| 1. New warm pilot (0.3), repeated G1 (0.6), repeated G2 (0.05) on the final source | ≈ 1.0 | pilot ≤ 1.40 min; bitwise unchanged paths; G1 ≤ 1e-4; G2 unchanged |
| 2. D3 scoring: 381 unique minus the ≈ 20 already rescored, ≈ 358 entries at 1.08 min plus overhead | ≈ 6.7 | immutable run dirs, one source |
| 3. Re-qualification on D3 stratum (A2.1) and the §5 power gate on the D3 OOF losses (A2.3), CPU | 0 | decides confirmatory or exploratory, and whether the owner buys step 4 |
| 4. Fresh scoring on the sealed panel: 255 + 26 spare unique deltas | ≈ 5.3 | starts only after step 3 is committed |
| Consumed already | 2.5 | |
| **Total against the 19.0 h cap** | **≈ 15.5** | ≈ 3.5 h slack |

Before any fresh component is scored, the GPU hours that must have run are about **7.7 h** (1.0 + 6.7), and the first 1.0 h of that is the only part needed before D3 scoring may begin.

## 7. Summary

1. **MMLU decision:** (b), amended prospectively (A2.1): panel-mean bound ≤ δ/2, ceiling h ≤ δ, stratum-weighted tripwire P ≤ 0.20; δ stays at 0.10; the 20-entry result passes interim (0.0312, 0.0893, P = 0.076) but is not counted, and the registered qualification is re-run on 229 D3 stratum entries under the final source.
2. **KL decision:** A1.2 outcome stands (unqualified, 25% > 20%); KL becomes a descriptive secondary, never used to select or exclude H1a rows; no first-token or shorter-prefix redefinition.
3. **Confirmatory vs exploratory:** exploratory (power-limited) by default; upgraded only if the D3-OOF §5 gate passes with the cluster bootstrap-t on ln R (BCa fallback), joint procedure included, null size ≤ 0.035 and lower power bound ≥ 0.80; 106 components registered for floor margin.
4. **Cost decision:** approve batching (A2.4) with a pilot gate of warm median ≤ 1.40 min and bitwise-unchanged MMLU/KL/WikiText; total ≈ 15.5 h of 19.0 h; fallbacks are dropping refusal stages, then subsampling null-family D3 rows.
5. **Before the panel starts:** A2 and source freeze (0 GPU), pilot + G1 + G2 ≈ 1.0 h, D3 scoring ≈ 6.7 h, then CPU re-qualification and the re-run §5 gate; fresh scoring (≈ 5.3 h) starts only after that gate is committed, so ≈ 7.7 GPU-h precede it.
