# Round 1 — Metrology CTO (council, 5 Nov 2026)

Branch `council-r1-metrology` (worktree `.worktrees/council-r1-metrology`). Every number below is in
`docs/council/2026-11-05/metrology_budget_output.txt`, produced by `docs/council/2026-11-05/metrology_budget.py`
(read-only over canonical `data/`; CPU only, no forward passes). Items marked **derived** are arithmetic on
recorded counts; **estimate** marks assumptions I did not measure.

**Bottom line.** On the fast tier, gsm8k, math500, ifeval and over-refusal carry no between-entry signal that
clears their own seed and item noise, and their family means are mostly base-draw artefacts. The likelihood
and teacher-forced fields carry real, deterministic signal. H1 should be re-defined on those targets, and its
bar set against a family- and norm-matched comparator rather than a pooled Spearman.

---

## 0. What was checked, and how

- **Sample.** The 300-entry D3 subset (abliteration 100, lora_sft 60, random_dir 60, task_vector 30, chain 50;
  `data/zoo/qwen3.5-0.8b/<id>/eval.json`, identified by `base_run_id` absent). The 89 chain predecessors enter only
  the composition and cost checks.
- **Matched references.** The three H0 base runs with records (`20261104-233637`, `-233616`, `-233657`) are
  item-identical (458 items each; compared item by item, outputs and scores included). The first-pass run
  `20261104-084939` has no records file and is not used. H0 entries use `20261104-233637-7c43da7c`; H1 uses
  `20261104-112635-8cee6d4a`; H2 uses `20261104-112739-7cbc5bbc`. These are different draws: gsm8k is 0.16 on H0 and
  0.06 on H1 at the same seed (`base/devices/*/*/eval.json`).
- **Integrity checks, in the script.** Item means of the stored `correct` field reproduce every `eval.json`
  aggregate (0 mismatches). Gold labels joined from the cached MMLU-Pro and HellaSwag datasets: argmax of the stored
  choice scores equals the stored correctness on 200/200 items for each benchmark.
- **Seed noise.** Seed-1 panel: 10 controls plus base on H0 (`data/runs/p2-batch/20261105-050543-aad6c310/`),
  noise measured as d = Δ(seed1) − Δ(seed0), SD/√2. The panel has 3 abliteration, 3 random_dir, 2 lora_sft, 2
  task_vector controls and **no chain entry**. Chain noise is extrapolated, not measured.
- **Device-centring.** Each device's mean Δ is removed before computing variance, since a device's base draw
  shifts all of its Δ. The "within family × device" rows are the stricter residual test.

Two definitions used throughout:

- **Signal variance** = var(Δy) − mean(paired item SE²). Paired item SE = SD(per-item paired difference)/√n.
- **Ceilings** (heuristic: corr(prediction, observed) ≤ √reliability). *Fixed-item ceiling* = √(1 − σ²_seed/var):
  the estimand H1 uses now, predicting the seed-0 score on these exact items. *Population ceiling* = √(1 − SE²_item/var):
  the estimand a benchmark claim needs.

---

## 1. Signal-to-noise budget (question 1)

### 1a. Per target, D3 subset (n = 300, device-centred)

| target | SD(Δy) | seed SD [95% CI] | item SE (mean) | fixed-item ceiling [CI] | population ceiling |
|---|---:|---|---:|---|---:|
| gsm8k (50 items, sampled) | 0.070 | 0.050 [0.035, 0.092] | 0.058 | 0.70 [0, 0.87] | 0.57 |
| math500 (25) | 0.050 | 0.027 [0.019, 0.049] | 0.070 | 0.84 [0.14, 0.93] | **0** |
| ifeval (25) | 0.034 | 0.025 [0.017, 0.045] | 0.039 | 0.69 [0, 0.87] | **0** |
| over-refusal keyword (50) | 0.060 | 0.036 [0.025, 0.065] | 0.061 | 0.80 [0, 0.91] | **0** |
| refusal keyword (50) | 0.081 | 0.012 [0.009, 0.023] | 0.023 | 0.99 | 0.96 |
| mmlu_pro accuracy (100) | 0.048 | 0 (exact) | 0.025 | 1.00 | 0.86 |
| hellaswag accuracy (100) | 0.019 | 0 (exact) | 0.014 | 1.00 | 0.69 |
| mmlu_pro gold log-prob, nats (100) | 0.325 | 0 | 0.058 | 1.00 | 0.98 |
| mmlu_pro gold margin (100) | 0.514 | 0 | 0.076 | 1.00 | 0.99 |
| hellaswag gold log-prob (100) | 0.0068 | 0 | 0.0018 | 1.00 | 0.97 |
| KL to base (50 prompts) | 0.066 | ≈0 | 0.0019 | 1.00 | 1.00 |
| wikitext log-ppl (8 windows) | 0.049 | 0 | 0.0074 | 1.00 | 0.99 |

The pooled rows include family structure. The residual beyond family and device (within family × device) is the
honest test of a predictor that already knows the family:

| target | residual SD | residual item SE | residual population ceiling |
|---|---:|---:|---:|
| mmlu_pro accuracy | 0.038 | 0.025 | 0.77 |
| hellaswag accuracy | 0.013 | 0.014 | **0** |
| gsm8k | 0.059 | 0.058 | 0.20 (fixed-item 0.51) |
| refusal keyword | 0.075 | 0.023 | 0.95 |
| mmlu_pro gold log-prob | 0.285 | 0.058 | 0.98 |
| mmlu_pro margin | 0.439 | 0.076 | 0.99 |
| hellaswag log-prob / margin | 0.0043 / 0.0053 | 0.0018 / 0.0028 | 0.91 / 0.85 |
| KL to base | 0.059 | 0.0019 | 1.00 |
| wikitext log-ppl | 0.029 | 0.0074 | 0.97 |

Source: `metrology_budget_output.txt`, sections "S/N budget" and "per-family budget".

### 1b. Per family (device-centred within family, D3 subset)

| target | chain | lora_sft | task_vector | abliteration | random_dir |
|---|---|---|---|---|---|
| gsm8k: SD / item SE / pop. ceiling | 0.093 / 0.066 / 0.70 | 0.071 / 0.062 / 0.47 | 0.058 / 0.054 / 0.36 | 0.034 / 0.052 / 0 | 0.042 / 0.055 / 0 |
| mmlu_pro acc: SD / SE / ceiling | 0.072 / 0.033 / 0.89 | 0.048 / 0.038 / 0.61 | 0.033 / 0.020 / 0.80 | 0.012 / 0.014 / 0 | 0.007 / 0.011 / 0 |
| hellaswag acc: SD / SE / ceiling | 0.026 / 0.021 / 0.58 | 0.012 / 0.021 / 0 | 0.015 / 0.010 / 0.74 | 0.005 / 0.005 / 0 | 0 |
| mmlu_pro log-prob: SD / SE / ceiling | 0.589 / 0.097 / 0.99 | 0.343 / 0.092 / 0.96 | 0.113 / 0.033 / 0.96 | 0.014 / 0.008 / 0.81 | 0.005 / 0.005 / 0.29 |
| refusal keyword: SD / SE / ceiling | 0.140 / 0.037 / 0.97 | 0.068 / 0.030 / 0.90 | 0.022 / 0.012 / 0.84 | 0.067 / 0.018 / 0.96 | 0.006 / 0.006 / 0 |
| math500: SD / SE / ceiling | 0.067 / 0.081 / 0 | 0.056 / 0.073 / 0 | 0.041 / 0.065 / 0 | 0.040 / 0.066 / 0 | 0.036 / 0.068 / 0 |

Source: "per-family budget" section. The seed term is pooled from the 10 controls for every family, so per-family
fixed-item ceilings are approximate.

### 1c. Answers

- **Which targets have real between-entry signal?**
  - *Yes, deterministic:* mmlu gold log-prob and margin, KL, and wikitext. Signal SD exceeds item SE by 4–35×
    (pooled), and the pooled population ceilings are 0.98–1.00.
  - *Yes, on the families where edits move the model:* mmlu accuracy (residual ceiling 0.77; zero in abliteration
    and random_dir). Refusal keyword is real but saturated.
  - *Family-only:* hellaswag accuracy. Family explains η² = 0.51 of pooled variance; residual SD 0.013 is below
    residual item SE 0.014, so its within-family ceiling is 0.
  - *At or below noise on the fast tier:* gsm8k, math500, ifeval, over-refusal. Population ceilings are 0 for three of
    four, and for gsm8k 0.57 with a seed CI running from 0 to 0.87. These four do not qualify under Board A's repeat
    rule (section 1d).
- **Is most Δy on mmlu_pro and hellaswag within ±0.02?** Yes: 70% (mmlu, 210 of 300) and 74% (hellaswag). The 30%
  and 26% outside that band are almost all lora_sft, chain, and a few task_vector entries (mmlu: 49 lora, 29 chain,
  6 task, 6 abliteration). ±0.02 is ~0.8 item SE for mmlu, so being inside the band is not evidence of no effect. It
  does mean most entries sit below what a 100-item binary can resolve.
- **Is the fast-tier Δy a base-draw artefact?** Largely, for the sampled benchmarks. Raw entry levels agree across
  devices within family (gsm8k: abliteration 0.079 H0 vs 0.081 H1; random_dir 0.094 vs 0.086; task_vector 0.087 vs
  0.094), while the bases differ by 0.10. So matched Δ differs by device through the base alone. Consequences:
  - random_dir, a null family (median KL 0.0006), has mean math500 Δ = −0.047, and its gsm8k Δ is −0.006 on average.
  - The lora_sft gsm8k mean Δ of +0.110 is mostly the H1 base. Its H0 members (n = 2) sit at the H0 base level.
  - Cross-device control check (aggregates, 10 entries): H1 − H0 on gsm8k has mean +0.004 and SD 0.059. Device is not
    what separates the two; the base draw is.
  - Device-matching does not remove base noise. It spreads one base draw across every entry on that device, which
    shifts the Δ of all 179 H1 entries together.
- **Cross-device stability for teacher-forced fields** (`cross-device aggregate` section): H1 − H0 SD is 0.010 for
  mmlu accuracy (max 0.02), 0.005 for hellaswag, 0.0003 for KL and 0.0004 for log-ppl. The continuous log-prob and
  margin cannot be checked across devices: device copies carry aggregate `eval.json` only, no item records.

### 1d. Board A's pre-registered repeat rule, applied to our panel (approximate)

Rule from `docs/reviews/H5/board-a-prereg-2026-11-04.md` ("Exact fast-tier qualification rule", lines 111–129):
R_b ≤ δ/2, δ = 0.10 for binaries, log 1.05 for log-ppl, 0.01 nats for KL. R_b is approximated as 1.96·SD(d) on 10
controls.

| target | δ/2 | R_b (1.96·SD) | median item CI half-width | repeat rule | comment |
|---|---:|---:|---:|---|---|
| gsm8k | 0.050 | 0.140 | 0.111 | fail | |
| math500 | 0.050 | 0.074 | 0.138 | fail | |
| ifeval | 0.050 | 0.069 | 0.078 | fail | |
| over-refusal | 0.050 | 0.099 | 0.119 | fail | |
| refusal keyword | 0.050 | 0.034 | 0.000 | pass | saturated; cannot rank small edits |
| mmlu_pro | 0.050 | 0.000 | 0.028 | pass | repeat is trivial (exact); between-entry test is 1b |
| hellaswag | 0.050 | 0.000 | 0.000 | pass | repeat is trivial; no residual signal (1a) |
| KL | 0.005 | 0.0003 | 0.0002 | pass | |
| wikitext | 0.024 | 0.000 | 0.002 | pass | diagnostic only per Board A |

A pass on repeatability is not evidence of between-entry signal. Board A's own list (MMLU-Pro and HellaSwag
provisional; refusal and KL conditional) matches these numbers, but hellaswag has no within-family signal, so it
should not carry H1.

---

## 2. Challenging the targets (question 2)

**Should H1 move to continuous, teacher-forced targets? Yes, with three conditions.**

1. **The primary comparison is the family- and norm-residual, not pooled ρ.** Pooled ρ mostly measures family offsets:
   family explains η² = 0.21–0.65 of device-centred variance on the continuous fields (η² = 0.35 on mmlu accuracy,
   0.50 on hellaswag). Within family, the weight-only norm already gives |ρ| ≈ 0.5 on those fields: ρ(‖Δ‖_F, mmlu
   gold log-prob) = −0.51 (chain, n = 50) and −0.43 (lora, n = 60); ρ(‖Δ‖_F, margin) = −0.54 and −0.42; ρ(‖Δ‖_F,
   refusal) = −0.49 (abliteration) to −0.56 (chain). BRIEF.md reports delta-norm alone ≈ 0.2 (pooled; RESULTS lists the P4 delta-norm
   comparator as pending). That is a family-offset artefact, not the norm's ceiling. H1 must beat this comparator within family, not just exceed 0.6 pooled.
2. **KL is a validity probe, not a target.** KL is a distance to base, and the norm predicts it well within family
   (ρ = 0.58–0.91 in every family). Scoring a predictor on KL tests whether it can read ‖Δ‖.
3. **Deterministic does not mean meaningful.** Continuous fields have zero seed noise, so their reliability says nothing
   about whether an edit changes benchmark capability. Claims must be limited to "teacher-forced likelihood on a fixed
   item set", stated next to the item count.

**Pre-registered set (TF-v1).** Every item is forward-only, with no sampling. The base is computed once per device and
cached. Item counts come from the paired-SE table in `metrology_budget_output.txt` ("items needed") and the per-item
SDs in 1a.

| target | items | definition | basis for the count | status |
|---|---:|---|---|---|
| MMLU-Pro gold log-prob and margin (letter scoring at "Answer:") | **1,000** | normalised log-prob of gold letter; gold minus best other | chain per-item SD 0.64 nats (choice-text variant, proxy) → paired SE 0.020 at 1,000 items; current 100 items give 0.064 | new scoring; the choice-text version is kept at 100 items for continuity |
| HellaSwag gold-ending log-prob (length-normalised) | 500 | same as current, larger item set | chain per-item SD 0.024 → SE 0.001 | current scoring, more items |
| GSM8K gold-solution NLL (mean per token) | 500 | teacher-forced on reference solutions | not measured | **pilot required** |
| MATH-500 gold-solution NLL | 250 | same | not measured | **pilot required** |
| KL to base, base continuations cached | **200** prompts | mean full-sequence KL | chain per-prompt SD ≈ 0.025 → SE 0.0018 (current 50: 0.0036) | expand; cache base continuations |
| Wikitext NLL | **64** windows | mean NLL, block bootstrap | per-window SD ≈ 0.031 → SE 0.004 (current 8: 0.011) | expand |
| Refusal and over-refusal first-token margin | 200 + 200 | log p(refusal prefix) − log p(compliance prefix) | not measured | **pilot required**; replaces saturated keyword rate as primary |

Total ≈ 0.75 M forward tokens per entry (**estimate**). Cost is in section 4.

**Not computable from our records:** refusal first-token margins (no logits stored), gold-solution NLL (not stored),
and cross-device checks on log-prob and margin (no device item records). Each needs a 10-entry pilot before the
pre-registration is frozen.

---

## 3. Effect sizes per family (question 3)

Matched Δ on the D3 subset. Source: "effect sizes per family" section.

| family | n | KL median / p90 | refusal keyword median / p90 | mmlu acc median / p90 | mmlu gold log-prob median / p90 (nats) | wikitext Δlog-ppl median / p90 |
|---|---:|---|---|---|---|---|
| random_dir | 60 | 0.0006 / 0.0011 | 0 / 0 | 0.010 / 0.010 | 0.004 / 0.008 | 0.0004 / 0.001 |
| abliteration | 100 | 0.0013 / 0.0038 | 0 / 0.040 | 0.010 / 0.020 | 0.006 / 0.016 | 0.002 / 0.011 |
| task_vector | 30 | 0.0006 / 0.032 | 0 / 0.022 | 0.010 / 0.072 | 0.006 / 0.115 | 0.001 / 0.104 |
| lora_sft | 60 | 0.041 / 0.090 | 0.020 / 0.140 | 0.060 / 0.121 | 0.224 / 0.743 | 0.105 / 0.121 |
| chain | 50 | 0.019 / 0.213 | 0.020 / 0.282 | 0.040 / 0.171 | 0.137 / 1.351 | 0.056 / 0.107 |

Mean Δ (matched) on the same subset: lora_sft gsm8k +0.110 (base artefact, section 1c); lora_sft mmlu acc −0.064;
lora_sft gold log-prob −0.29 nats; chain gold log-prob −0.38 nats; chain KL 0.083; lora_sft over-refusal −0.082.

**What the zoo produces.** random_dir is a null family. abliteration is small: KL median 0.0013 and refusal moves by
≤ 0.04 for 90% of entries. lora_sft is the consistent mid-range family. chain has the widest range (p90 KL 0.21,
p90 gold log-prob change 1.35 nats).

**Does the zoo span enough dynamic range to test a predictor?**

- For continuous targets, yes, but only on the non-null entries. 94 of 300 D3 entries have KL > 0.01 (lora 56/60,
  chain 30/50, task 6/30, abliteration 2/100, random 0/60). Full-zoo extrapolation: about 538 of 1,247 directories
  (from the family rates; `edit.json` family counts are chain 675, abliteration 205, random_dir 153, lora 121, task
  80, heretic_ara 13).
- For the sampled and binary targets at the fast tier, no. Entries clearing 2× total SE: mmlu 49, hellaswag 29,
  refusal 35, gsm8k 17, over-refusal 16 of 300. A predictor test on these is a test on fewer than 50 entries.

**Edit magnitudes needed.** Median KL and ‖Δ‖_F among entries whose Δy clears 2× total SE (script section
"entries with |Δ| > 2 total SE"):

| target | entries clearing | median KL | min KL | median ‖Δ‖_F | min ‖Δ‖_F |
|---|---:|---:|---:|---:|---:|
| mmlu accuracy | 49/300 | 0.042 | 0.004 | 2.6 | 0.25 |
| mmlu gold log-prob | 92/300 | 0.019 | 0.0003 | 1.5 | 0.25 |
| hellaswag accuracy | 29/300 | 0.072 | 0.006 | 2.2 | 0.45 |
| refusal keyword | 35/300 | 0.128 | 0.002 | 3.2 | 0.61 |
| gsm8k | 17/300 | 0.104 | 0.001 | 5.2 | 1.31 |
| wikitext log-ppl | 184/300 | 0.009 | 0.0003 | 1.4 | 0.06 |

The continuous log-prob field clears the bar at about half the median KL needed for binary accuracy (0.019 vs 0.042),
because its item SE is smaller. Going from 100 to 1,000 items cuts item SE by √10, which lowers that KL further.

---

## 4. Measurement redesign and costs (question 4)

### 4a. Keep, drop, add

- **Keep (secondary):** mmlu accuracy at 100 items (continuity); hellaswag accuracy (secondary, not H1); refusal and
  over-refusal keyword rates (descriptive); KL (expanded and cached); wikitext (expanded).
- **Drop from the per-entry fast tier:** the 1,024-token sampling of gsm8k, math500, ifeval, refusal and over-refusal.
  These account for 8,189 of the 8,444 decode steps per entry (**derived**, from recorded token counts with the 1,024
  cap: `decode_steps` in the script). The other 256 steps are base KL continuations, regenerated on every entry
  (`src/wsp/eval/refusal.py:65–80`). The likelihood scoring makes 1,345 unbatched single-sequence forwards per
  entry (`src/wsp/eval/run.py:263–275`).
- **Move to a frozen calibration subset (40 entries):** gsm8k, math500 and ifeval at 100 items × 4 samples at 1,024
  tokens. Refusal keyword stays descriptive there.
- **Add (TF-v1):** section 2's table.

### 4b. The sampling-noise arithmetic (**derived**, from scaling assumptions)

Seed SD scales as 1/√(n·k) for n items and k samples; item SE scales as 1/√n. For gsm8k, from the measured
0.050 seed SD and 0.058 item SE at 50 items × 1 sample:

- 100 items × 4 samples: seed SD ≈ 0.018, population SE ≈ 0.041.
- To bring population SE to 0.02 needs ~415 items × 4 samples. At 1,024 tokens that is too expensive per entry.

So the 4-sample design fixes seed noise but not item noise. Generative benchmarks become calibration instruments only.

### 4c. Power: ρ = 0.6 vs 0.4 (Fisher z, two-sided α = 0.05, 80% power)

From `metrology_budget_output.txt` ("power" section):

| n | SE(z) | 95% CI for ρ = 0.6 | 95% CI for ρ = 0.4 | P(0.6 vs 0.4 distinguished) |
|---|---:|---|---|---:|
| 300 (independent) | 0.058 | [0.52, 0.67] | [0.30, 0.49] | 0.91 |
| 300 (246 ancestry clusters, sensitivity) | 0.064 | [0.51, 0.67] | [0.29, 0.50] | 0.84 |
| 1,000 | 0.032 | [0.56, 0.64] | [0.35, 0.45] | 1.00 |
| 94 (non-null D3 entries) | 0.105 | [0.45, 0.72] | — | **0.44** |
| ≈538 (non-null full zoo, extrapolated) | 0.043 | [0.54, 0.65] | — | 0.99 |

Smallest ρ distinguishable from zero at 80% power: 0.16 (n = 300), 0.09 (n = 1,000).

Two points matter more than n:

- **Non-null count, not n, binds.** With 300 entries but only 94 non-null, the 0.6-vs-0.4 test has 44% power. Stratify by
  family and enlarge the non-null set (more chain and lora entries) before adding nulls.
- **Attenuation.** An observed ρ = true ρ × √reliability. At mmlu population reliability 0.74, a true 0.6 appears as
  0.52. For targets with population ceiling 0, no n produces a passable correlation.

### 4d. Costs

| setting | per entry | basis |
|---|---:|---|
| current fast tier, measured wall time | 8.6 min (H0), 9.9 min (H1), 23.7 min (H2 A10); brief quotes 7.5 min | `eval.json` `minutes` (`data/zoo`) |
| current decode steps | 8,444 | **derived**, records (`metrology_budget.py`, cost section) |
| implied wall time per decode step | 53–70 ms | **derived**; assumes generation dominates, not profiled |
| TF-v1 | ~1–2 min | **estimate:** 0.75 M tokens ≈ 1.2 × 10¹⁵ FLOP (2 × 0.8 B × tokens) → 12–40 s at 30–100 TFLOP/s effective, plus model load and delta application, which I did not measure |
| calibration: 4 samples × 100 items × 3 benchmarks at 1,024 | ~9–12 min | **estimate:** 10,240 steps at batch 128, assuming step time unchanged |
| frozen 4,096-token tier, 20 entries | ~11–14 min each, ~4–5 GPU-h total | **estimate:** 3 batches × 4,096 steps |
| D3 subset (300) | now 43–50 GPU-h; TF-v1 ~5–10 GPU-h | records; estimate |
| full zoo (1,247 dirs) | now ~180–206 GPU-h; TF-v1 ~21–42 GPU-h | records; estimate |

The GPU is shared (9% busy from another job at the time of checking), and the brief asks for CPU-only analysis, so
none of the TF-v1 figures are timed. The first action after pre-registration is a 10-entry timing run in tmux
(`agent-tmux`), which replaces the estimate.

### 4e. Seed panel

The 10-control panel gives a seed-SD CI of about 0.5×–1.8× (section 1a). To reach ±15% on each SD, use about 24 pairs,
at least 6 of them chain. Average the base over four seeds for sampled targets, since one base draw shifts every entry
on its device.

---

## 5. Three questions for other members

1. **Weight-reading researcher.** Within family, ‖Δ‖_F alone gives ρ = −0.51 on chain and −0.43 on lora for mmlu gold
   log-prob (`metrology_budget_output.txt`). Name the per-layer statistic you will commit to before seeing labels, and
   state the within-family ρ you expect it to add over ‖Δ‖_F on mmlu margin. Chain has n = 50, so the 95% CI on a ρ near 0.5 is
   about ±0.2 (Fisher z); what result would you accept as evidence?
2. **JEPAWG researcher.** Family explains η² = 0.21–0.65 of device-centred variance on the continuous fields, and only 94
   of 300 entries are non-null. Will you train and score on family-residual targets (Δ minus family × device mean) and
   make that the headline? Pooled ρ on these targets is mostly family offsets.
3. **Strategy CTO.** Board A's rule plus our panel makes H1 unreachable on gsm8k, math500, ifeval and over-refusal at the
   fast tier (population ceilings 0.57, 0, 0, 0; fixed-item CIs run from 0 to ~0.9). Hellaswag accuracy has no within-family
   signal. Do you accept re-defining H1 on TF-v1 before any predictor is run on the 89 chain predecessors and the full zoo,
   with the cost in section 4d treated as an estimate until the 10-entry timing run?

---

## 6. What I could not compute, and limits

- **Not computed:** refusal first-token margins and gold-solution NLL (no logits or reference solutions in records).
  Cross-device checks on log-prob and margin (device copies have no item records). Seed noise on chain (no chain entry
  in the panel) and on H1 or H2 (H0 only). Model-load overhead and any GPU timing (shared GPU, CPU-only brief).
- **Assumptions:** the 300 entries are treated as independent in the power table, with a 246-cluster sensitivity row.
  Chain steps share ancestry, so this is optimistic. Seed noise is taken as independent across entries (common random
  numbers would reduce Δ noise; not tested). Per-family ceilings use the pooled seed SD.
- **Noise estimates are wide.** Ten controls give seed SDs that are only roughly ±40%. Every ceiling for a sampled target
  inherits that: the gsm8k fixed-item ceiling runs from 0 to 0.87.
- **Counts differ from the brief.** The brief quotes 1,235 zoo edits; 1,247 directories have `edit.json`, including
  13 heretic_ara. D3 composition is 111 H0, 179 H1, 10 H2. Chain entries are all H0, so device and family are partly
  confounded in the subset. Device-centring is applied with that in mind, and the family × device residual is the
  primary check.
- **Verification.** The script runs in about 4 s on CPU. Every table in this memo is copied from its output file.

---

## Summary

1. **Branch:** `council-r1-metrology` (worktree `.worktrees/council-r1-metrology`; memo, script and output in
   `docs/council/2026-11-05/`).
2. **Targets with real between-entry signal:** gold log-prob and margin (pooled population ceilings 0.98 and 0.99; residual
   0.98 and 0.99), KL (1.00), wikitext (0.99), mmlu accuracy (0.86 pooled, 0.77 residual, zero in abliteration and random).
   Hellaswag accuracy has no residual signal (ceiling 0). gsm8k, math500, ifeval and over-refusal are at or below noise on
   the fast tier (population ceilings 0.57, 0, 0, 0).
3. **Pre-registered set (TF-v1):** MMLU-Pro gold log-prob and margin at 1,000 items; HellaSwag log-prob at 500; KL at 200
   prompts with cached base; wikitext at 64 windows; GSM8K and MATH gold-solution NLL and refusal margins (pilot first).
   Primary comparison: family and norm residual, against a ‖Δ‖_F comparator.
4. **Cost per entry:** about 1–2 min estimated for TF-v1 (≈0.75 M forward tokens, unmeasured), against 8.6–9.9 min measured
   now; 8,189 of 8,444 decode steps are sampled generation, so dropping it is most of the saving.
5. **Could current results be explained by noise alone?** Partly. The sampled-benchmark Δs and family means fail the
   repeat rule and show base-draw artefacts (random_dir, a null family, has math500 Δ −0.047). The likelihood signal is real,
   but within family it is mostly a norm effect, and hellaswag has none beyond family.
6. **Not computed:** refusal margins and gold-solution NLL (no stored logits); cross-device log-prob and margin; chain and H1
   seed noise; GPU timing and model-load overhead; cluster-adjusted power beyond a 246-cluster sensitivity row.
