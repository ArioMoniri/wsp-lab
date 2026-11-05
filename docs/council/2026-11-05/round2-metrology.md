# Round 2 — Metrology CTO (council, 5 Nov 2026)

Branch `council-r2-metrology` (worktree `.worktrees/council-r2-metrology`). CPU only; no forward passes, no GPU, no Lambda.
Every number below is in `docs/council/2026-11-05/round2_metrology_output.txt` (script `round2_metrology.py`) or
`round2_residual_estimand_output.txt` (script `round2_residual_estimand.py`), except where marked **R1** (copied from
`round1-metrology.md` §1a) or **estimate** (not measured). Costs use the recorded `eval.json` `minutes` field, not the brief's 7.5 min.

**Bottom line.**
1. The inference unit is the ancestry cluster, and the zoo has **646** of them (not 1,235 or 1,247). Only **281** are labelled today.
2. History (H2) is not testable on the current data. The M2-vs-stateless contrast has a placebo of 13% on rows with no history at all.
3. Within the label set, H1 has a real, cluster-robust residual signal on MMLU-Pro accuracy: after family, device and log‖Δ‖_F, the stats model cuts held-out residual MAE by 15% (95% CI 7–21%). It is absent on HellaSwag accuracy. This is exploratory, on data already used for P4.
4. Pre-register H1 as a residual estimand on a fresh, non-null panel of at least 100 clusters. Pre-register H2 only after a closure-first cluster count. Do not buy 700 evaluations for a 5% falsifier: it is not resolvable.

---

## 0. What was checked

- **Lineage census** (manifest, `edit.json`, `eval.json`): 1,247 entries, 389 evaluated, 646 ancestry roots. Ancestry clusters use the same rule as `wsp.predictor.splits.trajectory_groups` (root of the parent chain).
- **Measured cost:** `eval.json` `minutes`. H0 mean 8.63 min (n = 200), H1 9.92 (n = 179), H2 23.66 (n = 10); D3 H0 subset 7.01 (n = 111).
- **Stored fields:** counted in all 300 D3 `records.jsonl` files (`round2_metrology_output.txt` §D).
- **Paired P4 predictions** (`data/runs/p4/d3v2-m2-trajectory`, `d3v2-m2_matched-trajectory`, seeds 0–2, trajectory split). M2 is the history model; M2-matched is the stateless control. `trajectory_length` is the number of nodes in the lineage, root included (`src/wsp/predictor/data.py:416`).
- **Residual pilot** (`round2_residual_estimand.py`): the proposed primary estimand (§4.1), fitted on existing D3 rows with train-fold nuisance terms. StatNN is fixed-config, seed 0, no tuning.

Limits: the GPU is shared and the brief asks for CPU-only work, so no TF-v1 timing has been measured. Costs marked *estimate* come from 1–2 min per entry (R1 arithmetic) and must be replaced by the §4.6 pilot.

---

## 1. Answers to questions addressed to Metrology

### 1.1 Strategy CTO (round 1, Q3): smallest panel and measured cost to resolve a 15% history MAE effect

Estimator and power are in `round2_metrology_output.txt` §E and §H. Stand-in SD: the cluster-level SD of the M2-vs-stateless contrast on MMLU-Pro accuracy, length ≥ 4 rows (0.68, from 10 clusters, 18 rows). The 95% range on that SD is roughly ×0.69 to ×1.83, which gives 77–540 clusters. The point estimate is:

- **162 independent lineages, each with ≥ 4 nodes** (≈ 648 evaluated nodes), for 80% power at a true 15% MAE reduction, two-sided α = 0.05.
- **Measured cost of those evaluations at the current protocol:** 648 × 9.92 min ≈ **107 GPU-h**. Under TF-v1 (*estimate* 1–2 min per entry): **11–22 GPU-h**. Training of the 648 edits is not measured.
- **Cheapest route:** the ancestor closure of the 99 existing ≥4-node roots contains 426 unevaluated entries (539 in total). Evaluating them costs ≈ 70 GPU-h at the measured rate, or 7–14 GPU-h (*estimate*) under TF-v1, and needs no new training. That gives 99 clusters, which resolves a **19%** effect (2.8 × 0.68 / √99). The remaining 63 clusters are new 4-node lineages: 252 new edits.
- **Pilot to re-estimate the SD:** the first ~40 of the 99 closures (≈170 entries). It gives an SD within ±25%. Measured-rate cost ≈ 28 GPU-h; TF-v1 *estimate* ≈ 3–6 GPU-h.
- **Strategy's 8-run two-week design cannot resolve this:** MDE = 2.8 × 0.68 / √8 = **67%**. The 48-run core gives MDE = 27%.
- **Placebo:** on length-1 rows, where no history exists, M2 and M2-matched differ by **12.7%** (CI 3.9%–20.6%; 292 rows, 278 clusters). The length ≥ 4 contrast is 6.9% (CI −43% to +41%). The naive arm contrast is therefore not evidence about history. The H2 test must be history ablation inside M2 (§4.1).

**Smallest H1 panel (for the strategy's variable-endpoint gate):** the residual pilot (§4.1) gives an implied cluster SD of **0.47** on the non-null stratum (lora, chain, task). For 80% power at θ = 0.15 that needs **79 clusters**; for θ = 0.10 it needs 176. I recommend a fresh confirmatory panel of **≥ 100 non-null clusters**, which allows for the SD uncertainty. Cheapest realisation: 110 existing LoRA and task entries that are not yet evaluated (60 + 50). They sit on 98 clusters, 75 of them not already in D3. Adding ~30 chain lineages (new roots) reaches ~105. Measured-rate cost of the 110 entries ≈ 18 GPU-h; TF-v1 *estimate* 2–4 GPU-h. No new training.

### 1.2 Weight-reading researcher (round 1, Q3): residual variation, budget, zero-forward versus evaluation avoidance

- **How much resolvable variation remains after family and device centring?** Signal variance = residual SD² − item SE², and reliability at the TF-v1 item count = signal / (signal + SE_new²) (`round2_metrology_output.txt` §I; residual inputs from R1 §1a):

  | endpoint | signal SD | reliability at TF-v1 n | gate (≥ 0.90) |
  |---|---:|---:|:--:|
  | MMLU-Pro gold log-prob, 1,000 items | 0.279 nats | 0.996 | pass |
  | MMLU-Pro gold margin, 1,000 items | 0.432 nats | 0.997 | pass |
  | MMLU-Pro accuracy, 1,000 items | 0.029 | 0.929 | pass (small) |
  | HellaSwag gold log-prob, 500 items | 0.0039 /char | 0.959 | pass (small) |
  | HellaSwag accuracy | none (signal variance < 0) | — | fail |
  | KL to base, 200 prompts | 0.059 nats | ≈ 1.00 | pass |
  | wikitext log-ppl, 64 windows | 0.028 | 0.991 | pass |
  | GSM8K sampled accuracy, 500 items | 0.011 | 0.26 | fail |

  Items fix noise but not absent signal. The sampled GSM8K endpoint has no signal to recover.
- **Small edits:** random_dir has a residual SD of 0.007 on MMLU accuracy, below one item of resolution. Abliteration has 0.012 (R1 §1b). The non-null stratum carries essentially all the residual signal.
- **Budget and stopping rule.** Metrology cannot test a zero-forward claim, because every label requires a forward pass. A cached-activation probe is a separate information budget, so I propose a cap of **0.5 GPU-h** for caches and probe fits on the existing 281 clusters. Its stopping rule: on the non-null stratum, θ ≥ 0.15 with a lower 95% bound > 0 at ≥ 100 clusters. Otherwise stop, and keep only the cheap scalar-layer baseline.
- **Product:** cached-activation-assisted *evaluation avoidance*, not zero-forward auditing. It has to predict TF-v1 endpoints at a stated fraction of the TF-v1 cost. I agree with the weight-reading memo that the claim "zero forward" is not supported by any experiment here.

### 1.3 JEPAWG researcher (round 1, Q3): minimum repeatability and cluster power for 700 additional evaluations; redirect criterion

- **Repeatability:** TF-v1 endpoints are deterministic, so same-device repeat is exact by construction. Qualification therefore rests on cross-device checks (§4.5, G1 and G3), not on repeats.
- **Power:** the cluster SD of the StatNN-vs-descriptor-only MAE contrast is 0.87 over 278 clusters (`round2_metrology_output.txt` §F). The zoo has 646 clusters, and 700 new *independent* edits would add at most 700 more, so 1,346 at best. MDE at 1,346 clusters = 2.8 × 0.87 / √1,346 = **6.6%**. A 5% effect needs **2,355** clusters; a 10% effect needs 589.
- **Measured StatNN-vs-descriptor effect:** 4.1% MAE on MMLU (CI −6.3% to +14.0%), 5.5% on HellaSwag (CI −6.3% to +16.4%). The gap that the evidence memo calls "≈ 0.04" is not distinguishable from zero on MAE at this scale.
- **Recommendation:** do not buy the 700 evaluations for a 5% falsifier. Buy them only for an effect the decision depends on: 10% MAE needs ~590 clusters, 15% needs ~260 on the feature contrast.
- **Redirect criterion:** after ≥ 260 independent clusters, if the residual θ (§4.1) on the non-null stratum has an upper 95% bound below 0.05, redirect to the activation probes. Until then the weight-only residual remains open.

### 1.4 Features CTO (round 1 Q4 is addressed to Metrology): can the four admissible endpoints be qualified, and what gates feature experiments?

Board A's four admissible endpoints are MMLU-Pro, HellaSwag, KL and wikitext. The qualification gate is in §4.5. The reliability gate is **≥ 0.90** on signal variance at the TF-v1 item count. HellaSwag *accuracy* fails it and is retired. HellaSwag *log-prob* passes on paper at 500 items (0.96), but this needs confirmation in the pilot, because its signal (SD 0.0039 per character) is small. The repeat-based ranking rule in Board A is not usable for deterministic endpoints (see §2, challenge C2).

### 1.5 Strategy CTO (round 1, Q3 to Metrology): accept re-defining H1 on TF-v1 before any predictor is run?

Yes, with two changes from my round-1 text. (a) The redefinition must be a **residual** estimand (§4.1), not a pooled Spearman. (b) "Before any predictor is run" cannot be satisfied for the D3 panel, which has already been used for P4. So TF-v1 is exploratory on D3 and confirmatory only on fresh clusters. The strategy memo's split into H1a and H1b is the right shape.

---

## 2. Challenges to other members (with evidence), and concessions

**C1 — Strategy CTO, strategy memo §4 ("Gate 3" and the two-week table).** Quote: "Gate 3, day 14: a paired ancestry-cluster interval for MAE improvement over StatNN must exclude zero on a predeclared variable endpoint." And: "Four tasks × two seeds × four checkpoints = 32 new edited states."
*Flaw:* 32 states from 8 seeded runs form 8 independent clusters, because checkpoints within a run share ancestry. The MDE for a StatNN-vs-comparator MAE contrast is 2.8 × 0.87 / √8 ≈ 0.86, and for the history effect 0.67. Gate 3 is therefore a null test. It cannot pass unless the effect is ≥ 67%.
*Missing control:* a cluster-count gate. Gate 3 should read: "≥ 100 non-null clusters, and the θ CI half-width ≤ 0.5 × the θ point estimate."
*Also:* the two-week plan costs its 56 fast-tier batteries at 7.5 min each (7.00 h). Measured rates are 8.63 min (H0) and 9.92 min (H1), i.e. 8.1–9.3 h. The label "measured" should go.

**C2 — Strategy CTO, strategy memo §2 (H1a comparator set).** Quote: "H1a: weight-derived features predict predecessor-relative teacher-forced changes … better than fold mean, descriptors and delta norm."
*Flaw:* the comparator set omits the nuisance that matters, family + device + log‖Δ‖_F. Delta norm is a weak comparator: its pooled R² is about 0.06 (`evidence-statnn-importances.md`, pooled table). StatNN beats it by 37% MAE (MMLU, CI 31–44%) and 50% (HellaSwag). Both gains are real and say little about the features. On the residual, the same norm-based comparator is no better than predicting zero (MAE 0.0203 vs 0.0202 on MMLU). The correct comparator is isotonic(‖Δ‖_F) fitted to the residual.
*Evidence:* `round2_residual_estimand_output.txt`.

**C3 — Weight-reading memo, Rank 1 falsifier.** Quote: "original-plus-layer features reduce paired MAE by at least 5% … with ancestry-cluster CI for MAE improvement excluding zero." And: "Failure says location scalars alone do not rescue H1."
*Flaw:* a 5% MAE effect needs 2,355 clusters. The labelled set has 281, whose MDE is 14.5%. A null result at 5% is uninformative, and the design would read it as a refutation. This is a missing power check, not a wrong idea.
*Fix:* state the falsifier at 15% on the residual θ, with the cluster count written in.

**C4 — JEPAWG memo, planning and the 700-evaluation purchase.** Quote: "Screen at 1,000–3,000 independent ancestry clusters." And: "700 new independently sourced, balanced edits to reach ~1,000 entries."
*Flaw:* the zoo has 646 ancestry clusters (all 1,247 entries). Even if all 700 new edits were independent roots, the total is 1,346. The 1,000–3,000 planning range cannot be reached from this family design without new lineage types. The memo itself warns that entries are not clusters, but the planning figure still assumes they are.
*Missing control:* the geometry audit's permutation controls. A permutation that breaks family or cluster structure will inflate the difference between true and shuffled orders. Controls must preserve family and cluster blocks.

**Concessions**

- *Weight-reading:* correct that the blog's successful interventions used activations, feedback or both. Every label here needs a forward pass, so "weight-only" should be dropped from our claims. I also accept that the M1 token path keeps exact layer IDs (`data.py`); I did not verify that line.
- *JEPAWG:* correct that the history model needs order-shuffle and history-ablation controls. The placebo result (§1.1) shows why. This is a concession, not a disagreement.
- *Strategy:* correct that Board A's ranking rule (ρ ≥ 0.8 with a lower bound ≥ 0.6, ICC ≥ 0.75) cannot qualify deterministic endpoints; the ranking must be reported as unqualified. I also concede that the pooled-ρ critique is weaker than I wrote in round 1 for MMLU: pooled ρ is 0.566 and residual ρ 0.484 (§3). It is strong for HellaSwag: pooled 0.649, residual 0.258.

---

## 3. What changed my round-1 view

1. **Within-family norm.** Round 1 said H1 "must beat a within-family norm with |ρ| ≈ 0.5." That confused a raw norm correlation with a residual gain. After family, device and a log-norm term, the norm comparator's residual ρ is 0.10, and StatNN's is 0.48 on MMLU accuracy (`round2_residual_estimand_output.txt`). The residual MMLU signal exists and survives device centring (within-device ρ 0.50 on H0, 0.43 on H1). Exploratory only, on data already used by P4.
2. **HellaSwag.** Round 1 retired HellaSwag as a whole. Correct scope: retire HellaSwag *accuracy* (no signal). HellaSwag *log-prob* at 500 items passes the reliability gate on paper. I was wrong to lump them.
3. **Power for 0.6 vs 0.4.** Round 1's 0.44 (non-null, n = 94) is a two-sample comparison of two independent correlations. For a single predictor tested against ρ = 0.4 (one-sided α = 0.05), power is 0.82 at 94 entries and 0.79 at 86 clusters. Full zoo: 1.00 (`round2_metrology_output.txt` §G).
4. **R1 internal inconsistency.** R1's table gives KL item SE 0.0019 at 50 prompts; its §2 text gives 0.0036. Both give reliability ≥ 0.99 at 200 prompts. I use 0.0036 for the conservative case.
5. **Unit of inference.** Round 1 counted entries. The power that matters is clusters: 281 labelled, 646 in the zoo. Non-null D3 entries (94) map to 86 clusters, so the design effect is small there. Chains are the exception: 605 unevaluated production chain entries sit on 105 new clusters.

---

## 4. TF-v1, revised: pre-registration-ready spec

### 4.1 Primary estimands

**H1 — residual readability (primary).** Unit: ancestry cluster. Target y: predecessor-relative, device-matched TF-v1 endpoint. Primary endpoint: MMLU-Pro gold log-prob (1,000 items). Secondary: margin, and gold-solution NLL for GSM8K if it passes G4.

- **Nuisance (fitted on training folds only):** OLS of y on family dummies, device dummies and log‖Δ‖_F. Residual r = y − (nuisance fit).
- **Model arm:** predictor fitted to r on features (StatNN, fixed config, seed-averaged within cluster).
- **Comparator arm:** isotonic regression of r on ‖Δ‖_F (the norm comparator, monotone).
- **θ** = 1 − Σ_c Σ_i |r_i − r̂_i| / Σ_c Σ_i |r_i − r̂^C_i|. Positive θ means the features explain held-out residual error beyond family, device and norm.
- **Inference:** ancestry-cluster bootstrap, 4,000 resamples, percentile 95% CI.
- **Stratum (outcome-free):** the non-null stratum is defined by family (lora_sft, chain, task_vector), not by measured KL. Selecting on the outcome would bias θ.
- **Threshold:** H1 readability is supported if θ ≥ 0.15 with a lower 95% bound > 0 on ≥ 100 fresh non-null clusters. It is falsified if the upper 95% bound < 0.05 on the same panel.
- **Exploratory pilot (already run, P4 data, not confirmatory):**

  | target | stratum | rows | clusters | θ | 95% CI | residual ρ (StatNN / norm) |
  |---|---|---:|---:|---:|---|---|
  | MMLU-Pro accuracy | all families | 345 | 278 | 0.146 | 0.069–0.214 | 0.484 / 0.099 |
  | MMLU-Pro accuracy | non-null (lora, chain, task) | 187 | 128 | 0.192 | 0.109–0.273 | — |
  | HellaSwag accuracy | all families | 345 | 278 | −0.034 | −0.118–0.037 | 0.258 / 0.114 |
  | HellaSwag accuracy | non-null | 187 | 128 | −0.006 | −0.101–0.078 | — |

  Within-family residual ρ for StatNN, MMLU: abliteration 0.27, chain 0.54, lora 0.55, random_dir 0.57, task 0.45. The random_dir value is on the one-item floor and is not interpretable; the stratum excludes it.

**H2 — history (secondary, conditional on §4.3 closure).** Replace the M2-vs-stateless contrast with **history ablation inside M2** (same architecture, same seeds, history shuffled or dropped). Target: same TF-v1 endpoint. Rows: length ≥ 4 lineages.
- θ_H2 = 1 − Σ|e_M2| / Σ|e_M2-ablated|, cluster bootstrap.
- **Placebo:** the same statistic on length-1 rows must have a CI containing 0. If it does not, the arm contrast is invalid and H2 is not tested.
- **Threshold:** θ_H2 ≥ 0.15 with a lower bound > 0, on ≥ 162 clusters (SD to be re-estimated at the pilot). Falsified if the upper bound < 0.05.

### 4.2 Item sets and counts (frozen before the pilot)

Sampling uses the existing `wsp.eval.subsets.subset_ids(size, n, seed)` with seed **20261105**, excluding the legacy IDs in `configs/eval_subsets/`. Legacy items are kept as bridge items (§4.5 G2).

| endpoint | confirmatory items | legacy bridge | source |
|---|---:|---:|---|
| MMLU-Pro log-prob, margin (primary); accuracy (secondary, same items) | 1,000 (of 12,032 test; excludes legacy 300) | 100 | `TIGER-Lab/MMLU-Pro` test |
| HellaSwag per-character log-prob (secondary; accuracy retired) | 500 (of 10,042 validation; excludes legacy 300) | 100 | `Rowan/hellaswag` validation |
| GSM8K gold-solution NLL (exploratory until G4) | 500 (of 1,319 test; excludes legacy 200) | 0 | `openai/gsm8k` test |
| MATH-500 gold-solution NLL (exploratory until G4) | 250 (of the 500-item set) | 0 | `HuggingFaceH4/MATH-500` |
| Refusal first-token margin (exploratory; replaces keyword rate) | 200 harmful + 200 benign prompts (first 200 of each spec) | 50 + 50 | Heretic default prompt sets |
| KL to base (validity probe) | 200 prompts (first 200 benign) | 50 | same spec |
| WikiText NLL (diagnostic) | 64 windows (first 64, existing order) | 8 | `Salesforce/wikitext` test |

The refusal prefix sets (refusal vs compliance first tokens) must be frozen from a *calibration* set disjoint from the 400 confirmatory prompts. Freeze them before the pilot.

### 4.3 Cached-base requirements

- **Base run per device, dtype, batch and code hash.** A missing base means the entry is counted missing; there is no fallback to another device. This follows Board A's same-device rule.
- **Cache once per device:** every base scalar output for every confirmatory item (MMLU letter log-probs, HellaSwag ending scores and lengths, GSM8K/MATH gold NLL sum and token count, refusal prefix log-sum-exps, wikitext window NLL sums and token counts), and the 200 base KL continuations as token IDs.
- **Do not cache full base logits.** 200 prompts × 128 positions × ~250k vocabulary is too large. Keep the base model loaded and compute base KL forwards in-process on the cached continuations. The forward is cheap; the generation it replaces is not.
- **Removed cost:** the current protocol regenerates base continuations every entry (`src/wsp/eval/refusal.py`), about 256 decode steps. Caching removes that.
- **Calls:** the current code runs one unbatched forward per MMLU choice (`src/wsp/eval/run.py:263–275`, about 1,345 forwards per entry). TF-v1 runs one batched forward per MMLU item at the answer position, which yields all letter log-probs at once. The total number of items is about 4,600 batched sequence forwards per entry, against about 8,444 decode steps today.

### 4.4 What is stored per item

Enough to recompute every endpoint without rerunning the model:

- **MMLU-Pro:** the n-option raw log-prob vector (the full-vocabulary letter log-probs already in `extra.scores`), gold index from the dataset, n_options. Gold log-prob = score[gold] − logsumexp(scores). Margin = score[gold] − max over other options.
- **HellaSwag:** the 4 ending scores (per character, as now), each ending's character length, gold index. The existing `extra.scores` already has this.
- **GSM8K/MATH:** sum of token NLL on the gold solution, token count, and the per-token NLL vector (needed for block bootstrap).
- **Refusal:** log-sum-exp over refusal prefix tokens and over compliance prefix tokens at the first assistant position. Margin = difference.
- **KL:** full-sequence and first-token KL, continuation token count, and the continuation token IDs.
- **WikiText:** window NLL sum and token count (current records store mean NLL and n_tokens, from which the sum is recoverable).
- **Per entry:** device, base run ID, item and order hashes, batch, dtype, code hash, wall-clock by stage.

### 4.5 Qualification gates (numbers)

- **G0 Provenance.** 100% of items have a same-device base. Any missing item excludes the entry and is counted.
- **G1 Reproducibility.** Same-device, same-seed rerun on 2 entries: max |Δ log-prob| ≤ 1e-4 nats, KL ≤ 1e-5, wikitext log-ppl ≤ 1e-4. Failure stops the protocol.
- **G2 Bridge to the legacy scores.** On the 100 legacy MMLU items for the 10 pilot entries: median |Δ score| ≤ 0.02 nats, 99th percentile ≤ 0.2 nats, argmax agreement ≥ 99%. A failure means the new scoring is not the old scoring, and the base must be rebuilt for every entry.
- **G3 Cross-device.** For MMLU log-prob and margin, over the 10 pilot entries run on H0 and H1: SD of (Δ_H0 − Δ_H1) ≤ 0.2 × SD of Δ_H0 across entries (ICC ≥ 0.96). KL and wikitext: ≤ 0.1 ×. Failure restricts the confirmatory panel to one device.
- **G4 Reliability (signal, not repeat).** At the TF-v1 item count, reliability = signal variance / (signal variance + SE_new²) ≥ 0.90 on the family-and-device residual. The signal estimate must be recomputed on the full TF-v1 D3 run before the confirmatory set is frozen. Current estimates (§1.2 table): pass for MMLU log-prob, margin, KL, wikitext and HellaSwag log-prob; fail for HellaSwag accuracy and sampled GSM8K.
- **G5 Effect screen.** Where Board A gives δ, use R_b ≤ δ/2 and CI half-width ≤ δ/2 (binaries δ = 0.10; KL 0.01 nats; log-ppl log 1.05). For log-prob and margin, Board A has no δ; I propose CI half-width ≤ ½ × signal SD, and flag it as an amendment. MMLU log-prob at 100 items: 1.96 × 0.058 = 0.114 ≤ 0.14, so it passes already; at 1,000 items 0.036. G5 is therefore not what justifies the 1,000-item expansion. The expansion is justified by MMLU accuracy (reliability 0.57 at 100 items, 0.93 at 1,000) and by reducing attenuation on the log-prob primary (0.96 → 0.997).
- **G6 Stratum.** Non-null stratum = family in {lora_sft, task_vector, chain}, fixed before the pilot. Power is stated on clusters within that stratum.

**Retired ranking rule.** Board A's Spearman-and-ICC rule is replaced for deterministic endpoints by G3 and G4. This needs Board A sign-off.

### 4.6 The 10-entry timing pilot (before freezing)

- **Entries:** 10 from D3, fixed by rule: 4 lora_sft (the two with KL nearest the family median, two with the largest KL), 3 chain (one with a ≥ 4-node lineage), 2 task_vector, 1 abliteration (largest KL).
- **Runs:** H0, warm and cold; repeat 2 entries (G1); H1 for all 10 (G3); legacy bridge on the 100 legacy MMLU items (G2).
- **Timing:** wall-clock per stage (load, delta application, MMLU, HellaSwag, GSM8K/MATH NLL, refusal, KL, wikitext), per entry.
- **Gates:** median warm wall-clock ≤ 2.0 min/entry passes. > 3.0 min: drop MATH NLL first, then HellaSwag 500 → 250, and re-run the pilot.
- **Cost:** 22 entry-runs plus base caching. *Estimate* 0.4–0.7 GPU-h on TF-v1; 3.6 GPU-h at the measured fast-tier rate.

### 4.7 Power: non-null subset versus full zoo

Residual θ (MMLU, implied cluster SD 0.615 all families, 0.473 non-null). MDE = (1.96 + 0.84) × SD / √clusters, 80% power, two-sided α = 0.05.

| panel | clusters | MDE θ | note |
|---|---:|---:|---|
| existing D3, all families | 278 | 0.103 | exploratory, used by P4 |
| existing D3, non-null stratum | 128 | 0.117 | exploratory |
| fresh non-null, stage A (110 LoRA/task entries, 75 new clusters) | 75 | 0.153 | just below the 0.15 bar |
| fresh non-null, stage A + ~30 chain lineages | 105 | 0.129 | **minimum confirmatory** |
| full zoo, non-null (projected) | ~492 | 0.060 | requires evaluating the remaining 858 |
| full zoo, all families | 646 | 0.068 | |
| full zoo + 700 new independent edits | 1,346 | 0.047 | still above the 5% target |

Clusters for 80% power at θ = 0.15: 79 (non-null, SD 0.473), 132 (all families, SD 0.615), 262 (feature MAE contrast, SD 0.87). Stage A alone (75 new clusters) falls just short, which is why ~30 chain lineages are added.

One-sample ρ power (ρ = 0.6 vs H0 ρ = 0.4, one-sided α = 0.05): D3 non-null 0.82 at 94 entries, 0.79 at 86 clusters; full zoo non-null (projected) 1.00.

### 4.8 Cost, measured and estimated

| item | count | measured-rate GPU-h (eval.json) | TF-v1 estimate GPU-h |
|---|---:|---:|---:|
| current fast tier, H1 rate | 1 entry | 9.92 min | — |
| pilot | 22 entry-runs | ≈ 3.6 | 0.4–0.7 |
| stage A, fresh non-null | 110 entries (98 clusters) | ≈ 18 | 2–4 |
| stage B, chain lineages | ~30 lineages × ~4 entries ≈ 120 | ≈ 20 | 2–4 |
| D3 under TF-v1 (training labels) | 300 entries | ≈ 50 | 5–10 |
| H2 closure of 99 roots | 426 entries | ≈ 70 | 7–14 |
| H2 new lineages | 63 × 4 = 252 edits | ≈ 42 | 4–8 (training unmeasured) |
| full-zoo residual (remaining) | 858 entries | ≈ 142 | 14–29 |
| 5%-falsifier purchase | 700 entries | ≈ 116 | 12–24 (MDE 6.6%, not 5%) |

Training of any new edit is not measured. Strategy memo §4 says to instrument it; I agree.

---

## 5. Existing fast-tier fields: retired from H1, and what the paper must say

**Retired from H1 (confirmatory).** These fields remain in the records and may be reported as descriptive.
- GSM8K sampled accuracy (50 items, 1,024-token cap): repeat fails (R_b 0.14 vs δ/2 0.05), signal ≈ 0 (G4 reliability 0.26 at 500 items).
- MATH-500 accuracy (25 items): repeat fails; no population signal.
- IFEval accuracy (25 items): repeat fails (R_b 0.069 vs 0.05); no population signal.
- Refusal keyword rate (50): saturated (median 0, chain p90 0.28). Replaced by the TF-v1 first-token margin, which is exploratory until its pilot passes.
- Over-refusal keyword rate (50): repeat fails (R_b 0.099); no population signal.
- HellaSwag accuracy (100): no signal beyond item noise (signal variance < 0).
- MMLU-Pro accuracy (100): demoted to secondary, and re-scored at 1,000 items.
- loop_rate / `hit_max_tokens`: descriptive only. The 1,024-token cap truncates the output, so loops cannot be separated from long correct answers.
- WikiText 8-window perplexity: diagnostic, expanded to 64 windows.
- KL to base (50 prompts): validity probe, expanded to 200. First-token KL is descriptive.

**Retained in TF-v1:** MMLU-Pro log-prob and margin (primary), KL (validity probe), WikiText (diagnostic), HellaSwag log-prob (secondary), GSM8K and MATH gold-solution NLL and refusal margin (exploratory, gated by G4 and the pilot).

**What the paper must say.**
1. The sampled and keyword fields were measured. They failed Board A's repeat or population tests, so they are reported descriptively and do not carry any H1 claim. State the counts and the R_b values.
2. The TF-v1 endpoints were chosen **after** the D3 panel was analysed, because D3 variance is what set the item counts. Every result on D3 is exploratory. Confirmatory evidence comes only from fresh clusters.
3. The 1,024-token generation cap makes sampled outputs loop-sensitive. Say so, and do not read sampled Δ as capability change.
4. The item-noise floor: MMLU-Pro accuracy changes below 0.01 are below one item of resolution on 100 items. Report this next to any random_dir result.
5. The count of independent units is clusters (646 in the zoo, 281 labelled), not entries. Report entries and clusters side by side in every table.

---

## 6. Ranked actions I endorse

1. **Run the TF-v1 10-entry pilot (§4.6) before any freeze.** Owner: Metrology, with Features for the scoring code. Cost: 0.4–0.7 GPU-h estimate (3.6 at measured rate). Prediction: warm median ≤ 2.0 min/entry on H0; legacy-bridge median |Δ| ≤ 0.02 nats; determinism exact; cross-device ICC ≥ 0.96. Kill: median > 3.0 min/entry (cut items, re-pilot); bridge median > 0.05 nats (protocol break, rebuild every base); any non-reproducible item (stop).
2. **Build the fresh non-null H1 panel: 110 unevaluated LoRA/task entries (75 new clusters) plus ~30 chain lineages (~120 entries), to ≥ 100 clusters, with TF-v1 on the 300 D3 entries for training labels.** Owner: Strategy (panel), Metrology (estimand and analysis). Cost: ~530 entries, 9–18 GPU-h estimate (≈ 88 measured). No new training. Prediction: θ for MMLU log-prob in [0.10, 0.25], lower bound > 0. Kill: after ≥ 100 fresh non-null clusters, upper 95% bound < 0.05 → readability beyond norm is not supported for this endpoint; stop it, and keep H1 as a descriptive result.
3. **Rewrite H2 as history ablation inside M2, with a placebo, and start with the closure.** Owner: Strategy (design), JEPAWG (model), Metrology (analysis). Cost: closure of 426 entries (7–14 GPU-h estimate; ~70 measured); 40-lineage SD pilot first; the 63 new lineages (252 edits) only if the SD pilot gives a 15% MDE at ≤ 162 clusters. Prediction: placebo CI includes 0; θ_H2 in [0.05, 0.15] (the current contrast is 0.07 with a 0.13 placebo). Kill: pilot SD > 1.0 (MDE > 27% at 99 clusters): report H2 unresolved and stop; placebo CI excludes 0: the arm contrast is invalid, stop.
4. **Cap every new purchase by the MDE it must resolve, not by evaluation count.** Owner: Chair. Concretely: reject the 700-evaluation purchase for a 5% falsifier (MDE 6.6% even at 1,346 clusters); accept purchases sized to a decision effect (10% feature MAE needs ~590 clusters; 15% residual θ needs 79–132). Cost: 0. Prediction: none. Kill: any proposal whose pre-registered effect is below the MDE at its cluster count is rejected.
5. **Amend the strategy's gates and the field list.** Owner: Strategy and Chair. Replace Gate 3 with a cluster-count gate (§2 C1); retire the fields in §5 from H1; replace Board A's ranking rule for deterministic endpoints with G3 and G4 (needs Board A sign-off); correct the 7.5 min battery figure to the measured 8.6–9.9 min. Cost: 0. Prediction: none. Kill: if Board A rejects the G4/G3 substitution, keep the ranking rule and report the endpoints as unqualified.

---

## 7. Limits and open items

- **Not measured:** TF-v1 wall-clock, model load, delta application, training time for any new edit, GPU-level utilisation. The 10-entry pilot (§4.6) replaces the TF-v1 estimates.
- **Stand-in SDs.** The H2 cluster SD (0.68) comes from 10 clusters, 18 rows, MMLU accuracy; the residual SDs (0.615, 0.473) come from the pilot on the same accuracy endpoint. The TF-v1 log-prob SD is not measured. The SD CI for the H2 stand-in is roughly ×0.69 to ×1.83.
- **Seeds.** The residual pilot uses StatNN seed 0 only. The M2 contrasts use three predictor seeds, and their seed SD is 0.009 on the length ≥ 4 rows but 0.093 on all rows. Three seeds cannot characterise predictor-seed variance.
- **Post-hoc D3.** The residual pilot is exploratory on data already used by P4. Its θ is not a confirmatory result.
- **Known inconsistency.** R1's KL item SE is stated as 0.0019 at 50 prompts (table) and 0.0036 (text); both give reliability ≥ 0.99 at 200 prompts.
- **Not verified:** the M1 token path's exact-layer IDs (cited in the weight-reading memo); the refusal prefix sets (to be frozen from a calibration set); the claim that the current protocol's choice scoring is numerically identical to a single batched forward (G2 tests it).
- **Pronouns and names:** none used.

---

## Summary

1. **Branch:** `council-r2-metrology` (worktree `.worktrees/council-r2-metrology`). Memo `docs/council/2026-11-05/round2-metrology.md`; scripts and outputs `round2_metrology.py` / `round2_metrology_output.txt`, `round2_residual_estimand.py` / `round2_residual_estimand_output.txt`.
2. **Strongest challenges:** (i) Strategy Gate 3 uses 8 independent clusters, so its MDE is 67%; it is a null test by design (C1). (ii) The M2-vs-stateless "history" contrast has a 12.7% placebo on rows with no history at all, so H2 must be history ablation inside M2 (§1.1). The weight-reading 5% falsifier needs 2,355 clusters and is unresolvable (C3).
3. **Revised primary estimand:** the residual θ = 1 − Σ|r − r̂_StatNN| / Σ|r − r̂_isotonic(‖Δ‖_F)|, with r = y − (family + device + log‖Δ‖_F), fitted on training folds, ancestry-cluster bootstrap. Exploratory value θ = 0.146 (CI 0.07–0.21; non-null 0.19, CI 0.11–0.27) on MMLU accuracy; ~0 on HellaSwag accuracy. Confirmatory: θ ≥ 0.15 with lower bound > 0 on ≥ 100 fresh non-null clusters.
4. **What I changed my mind about:** the within-family norm is not the whole residual story for MMLU (the residual beyond family, device and log-norm is real, exploratory); HellaSwag accuracy is retired, but HellaSwag log-prob at 500 items passes reliability on paper; the 0.44 power figure was a two-sample criterion, and the one-sample figure is 0.82.
5. **Open disagreement for the chair:** whether to treat residual readability on MMLU as a confirmatory claim now, on ~100 fresh clusters (Metrology: yes, after the pilot; Strategy: benchmark-first, no predictor claim until qualified). Whether to fund H2 at all: the closure-first route costs 7–14 GPU-h (TF-v1 estimate) before any new training is justified.
