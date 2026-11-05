# Results

(empty — filled by P2+ runs; every number carries run id, n, CI method)

| Phase | Table | Run id | Status |
|---|---|---|---|
| P2 | Base model reproduction vs blog table | p2/base-final-4096 | 4k pass done; 32k rerun pending |
| P3 | Baselines by family / by seed | — | pending |
| P4 | H1–H4, protocol v2 | `data/runs/p4/d3v2-*` (25 complete runs; `d3v2-m3-family` has no metrics) | H1 not demonstrated; H2, H3 undetermined; H4 untested. Table: paper, "Predictor suite under protocol v2" |
| P5 | 27B transfer | — | pending |

## P2 reproduction status

Implementation was merged to `main` on 3 Nov 2026 (merge commit `2b6904a`, Board A APPROVE). CPU tests and the eight-item, two-run GPU
determinism smoke test pass. Full benchmark numbers and blog gaps remain pending;
no measurements are inferred from the smoke test. The provisional `p2-base` job
was superseded before any benchmark completed, so it supplies no quoted scores.

`scripts/eval_p2.sh` runs the full 4096-token cheap battery followed by the
32768-token GSM8K/MATH-500 comparison sequentially, then generates the table from
complete per-item records. It records Wilson 95% intervals for binary metrics and
explicitly identifies the different loop-rate denominators. The reference blog
numbers supplied in the task are MATH-500 7.9%, GSM8K 13.8%, IFEval 21.0%, and
loop rate 93% at 32k; they are not reproduced results yet.

## P2 — Base model reproduction (Qwen/Qwen3.5-0.8B, thinking on)
Run id `p2/base-final-4096` (git 6d64980, model rev main); max_new_tokens 4096; sampling T 0.6, top-p 0.95, top-k 20, presence 1.5;
fixed subsets (configs/eval_subsets). Blog (Takehi et al. 2026, Table "Qwen3.5-0.8B base") used 32,768 tokens and several
samples per item; our 32k pass crashed (CUDA OOM under GPU contention) and will be rerun.

| Benchmark (n) | Ours @4k (%) | Blog @32k (%) |
|---|---|---|
| gsm8k | 15.5 | 13.8 |
| math500 | 6.0 | 7.9 |
| ifeval | 42.0 | 21.0 |
| mmlu_pro | 20.3 | — |
| hellaswag | 51.3 | — |
| loop rate (gsm8k+math500+ifeval) | 90.2 | 93.0 (MATH-500 only) |
| refusal rate (keyword, 100 harmful) | 99.0 | — |
| KL to base (self-check) | 0.000 | — |

Reading: GSM8K and MATH-500 are within a few points of the blog at a 8× smaller token budget; the loop rate
confirms the blog's central failure mode (≈ 90 % of generations hit the limit). IFEval is far above the blog's
21.0; the budget and checker differences must be reconciled in the 32k rerun before this number is used.
HellaSwag/MMLU-Pro are log-likelihood scored (no generation) and unaffected by the loop. Refusal 99/100 is
consistent with Heretic's reported 97/100 for Gemma-3-12B-class aligned models. Items: data/runs/p2/base-final-4096/records.jsonl.

### P2 batch smoke

Run `20261104-032002-1c2cc588`, resolved model commit
`2fc06364715b967f1860aea9cf38778875588b17`: base plus two production edits
completed all nine benchmarks, n=4 per benchmark (WikiText: 4 windows).
Batch 4, generation budget 32 (KL continuation budget 128). Entry times
0.2997 / 0.3022 minutes; descriptive mean 0.301, no confidence interval with n=2.
These are engineering smoke timings, not benchmark reproduction measurements.
See ROADMAP P2 for explicitly qualified workload projections.

## P2 — Clean fast-tier timing (D2) on one production abliteration entry
Run under tmux `post-zoo` (log `~/agent-logs/post-zoo-20261104-053857.log`), entry `4463696a4c62e8f3`, GPU otherwise idle:
**745 s = 12.4 min per entry** for the fast tier (gsm8k 50 / math500 25 / ifeval 25 / mmlu_pro 100 / hellaswag 100 /
refusal 50 / kl 50, max_new_tokens 1024, batch 32, HF generate). First edited-entry scores (percent): GSM8K 14.0,
MATH-500 8.0, IFEval 44.0, MMLU-Pro 28.0, HellaSwag 39.0, refusal 100, KL-to-base 0.001, loop rate 100.
Projection: D3 subset (300 entries) ≈ 62 h on this GPU with the HF path; all 1,247 manifest rows ≈ 258 h (1,247 × 745 s). The throughput
benchmark (HF static cache vs vLLM) runs next and decides the engine; Lambda A100 is the fallback once the ARA family ends.

## P2 — Engine throughput (docs/BENCH_EVAL.md, branch perf/eval-throughput 73f1419)
Same entry: HF batch 32 6.18 min/entry; vLLM 6.10 (single runs; four CPU extraction workers busy during both). The "HF best"
6.15 figure in the branch doc is the same configuration as HF batch 32 (noise). Static cache and torch.compile were not usable.
vLLM changed scores at stochastic settings (gsm8k 0.14→0.06, ifeval 0.44→0.36), so the HF path stays canonical. The earlier
12.4 min figure was measured while extraction also competed for the host. D3 subset started 4 Nov (tmux p2-d3-batch), projected ≈ 31 h.

## Interim D3 readout (4 Nov 2026, 20:40 UTC; 112 of 300 subset entries evaluated, fast tier, run p2-d3-*)
Not a result table: the subset is 37 % evaluated, family counts are unbalanced, and cross-host noise (below) is not
yet separated from edit effects. Family means over evaluated entries (base in the first row):

| family | n | gsm8k | math500 | ifeval | mmlu_pro | hellaswag | refusal | over-refusal | KL to base |
|---|---:|---|---|---|---|---|---|---|---|
| base | 1 | 0.160 | 0.160 | 0.400 | 0.280 | 0.390 | 1.000 | 0.100 | 0.000 |
| abliteration | 22 | 0.084 | 0.067 | 0.405 | 0.286 | 0.392 | 0.995 | 0.177 | 0.002 |
| chain | 50 | 0.143 | 0.083 | 0.398 | 0.227 | 0.415 | 0.906 | 0.124 | 0.083 |
| lora_sft | 8 | 0.138 | 0.075 | 0.390 | 0.236 | 0.434 | 0.982 | 0.118 | 0.043 |
| random_dir | 21 | 0.094 | 0.074 | 0.404 | 0.284 | 0.390 | 0.999 | 0.181 | 0.001 |
| task_vector | 11 | 0.087 | 0.065 | 0.404 | 0.258 | 0.395 | 0.996 | 0.175 | 0.008 |

Observations to be tested, not claims: single-step abliteration entries barely move refusal (0.995 vs 1.000) and
KL (0.002) at the fast tier, while chains do (refusal down to 0.38, KL up to 0.73); gsm8k and math500 fall under
every family, including random directions, by 0.06–0.09, which is within the cross-host spread below.

**Cross-host controls (local RTX PRO 4500 vs Lambda A10, same seed, batch 32, temperature 0.6).** Base model:
gsm8k 0.16 vs 0.06 (n=50), math500 0.16 vs 0.28 (n=25), ifeval 0.40 vs 0.44 (n=25), mmlu_pro 0.28 = 0.28 (n=100),
hellaswag 0.39 = 0.39 (n=100), wikitext ppl 22.90 vs 22.91. The largest base-only gap is 0.12 (math500); on the four shared edited entries the per-benchmark
gaps reach 0.16 (math500). Item counts alone give a binomial SE of 0.03–0.10 on the 25–100-item benchmarks (IFEval near 0.10; Board B
correction), so the fast tier cannot resolve single-edit effects of that size on gsm8k/math500/ifeval. Board A's
qualification rule (docs/reviews/H5/board-a-prereg-2026-11-04.md) provisionally admits mmlu_pro and hellaswag for
large-effect screening and refusal/KL as narrow probes; none is qualified until the repeat panels are analysed. Same-host test-retest (10 controls + base, tmux p2-retest)
and the H100's 10 controls are running; Δy will be computed against the same-host base (D3 addendum 3).

**P4 data-path check** (`data/runs/p4/smoke-m1-realdata`, M1, family split, 35 labeled edits): Spearman within
[−0.3, 0.55] with intervals crossing zero on every target except mmlu_pro (0.54 [0.23, 0.81]) and loop rate; at n=35
and the noise level above this is uninformative either way and is recorded only to show the pipeline runs on real
features and targets.

**Same-host test-retest (4 Nov, tmux p2-retest; run id p2-retest-20261104-202946).** The base model and the 10
control entries were re-evaluated on the same GPU with the same seed, config and actual batch size (32). All 11 pairs
are bit-identical on every metric (gsm8k, math500, ifeval, mmlu_pro, hellaswag, refusal, over-refusal, KL,
wikitext ppl, loop rate): mean and max absolute difference 0.000. Within a device the fast tier is therefore
deterministic; the cross-host gaps above are systematic hardware/kernel effects on the sampled generations, not
run-to-run noise. What this does *not* measure is sampling noise under a different seed (a different draw at
T = 0.6 from the same items); that requires a seed-1 repeat panel, scheduled once the `--repeat` runner
(branch fix/eval-board-a-h5) is merged. Item-count SE from Board B/Board A still applies to any single draw.

**Cross-host control panel, three devices (4 Nov, 22:00 UTC; same seed, config and batch 32; H100 PCIe shard
`lambda0`, A10 shard `lambda1`, local RTX PRO 4500).** 11 edited entries evaluated on both local and H100 (the 10 controls plus `013b7b61202fe188`), and 4 edited entries plus the base on local and
A10 (the A10's other control copies were lost, see board/lambda_spend.md). Mean / max absolute difference and Spearman
over entries, local vs H100 (n = 11): gsm8k 0.053 / 0.12 / ρ 0.30; math500 0.040 / 0.12 / 0.09; ifeval 0.022 / 0.08 /
0.29; over-refusal 0.049 / 0.14 / 0.39; mmlu_pro 0.008 / 0.02 / 0.75; hellaswag 0.003 / 0.01 / 0.76; KL 0.000 / 0.001
/ 1.00; loop rate 0.006 / 0.04 / 0.72. Base: gsm8k 0.16 vs 0.06, math500 0.16 vs 0.08, over-refusal 0.10 vs 0.20,
others within 0.01. Local vs A10 (4 edited entries plus base): same pattern (math500 max 0.16, ifeval max 0.12; mmlu_pro, hellaswag, KL,
refusal within 0.02). Reading, pending Board A's formal qualification (their rule needs ρ ≥ 0.8 with lower bound
≥ 0.6, and this panel has little between-entry variance): the likelihood-scored benchmarks (mmlu_pro, hellaswag),
KL and the keyword refusal rate transfer across devices; the sampled-generation benchmarks (gsm8k, math500,
ifeval, over-refusal) do not, and their Δy must be taken against the same-device base. Local records: the ten controls from `20261104-104702-5fac075c`; `013b7b61202fe188` and the first-pass base from `20261104-084939-f912077b` (source hash `d7cece35`, the controls' is `6f274a83`). Scripts: ad-hoc, to be replaced
by `wsp.eval.repeats.compare_pairs` once fix/eval-board-a-h5 merges.

## P4 — first complete predictor suite on the D3 subset (4 Nov 2026, 23:13–23:33 UTC; runs data/runs/p4/d3-*)
Board A verdict (docs/reviews/P4/board-a-d3-suite-2026-11-04.md): **REVISE — H1 undetermined and not demonstrated;
H2/H3 insufficient evidence.** Supportable statements, in Board A's wording:
> On the D3 fast-tier subset, 260 of 300 evaluated edits had matched-reference delta targets; all retained labels used
> inferred runtime metadata. In exploratory trajectory-grouped evaluation, M2 achieved Spearman .54 [.42,.63] on
> MMLU-Pro and .49 [.38,.60] on HellaSwag, compared with StatNN .61 and .67. Intervals resample edits conditional on
> the fitted models and do not correct chain dependence.
> The pre-registered H1 criterion was not demonstrated. Ranking qualification of the fast targets remains incomplete,
> and the P4 delta-norm comparator is pending. Pooled family-holdout correlations were sensitive to between-family
> calibration offsets and do not establish within-family transfer.
Cause of the negative family-split baselines: between-family calibration offsets in pooled correlations plus
extrapolation instability of the neural heads; the pooled family-holdout ρ is not a transfer measure. Full table:
docs/reviews/P4/d3-summary-2026-11-04.md. Next (ranked by Board A): finish the 89 chain-predecessor evaluations
(running), seed-1 repeat panel, nested model selection with fold-mean and delta-norm baselines and per-edit baseline
predictions, feature audit with descriptor-only and device-only controls, ancestry-cluster bootstrap, multiple seeds.

## P4 revised-protocol CPU smoke (run `p4-round2-m1-family-cpu-final`)

This is a two-supervised-epoch M1 engineering check on the D3 fast targets,
not a full rerun or hypothesis qualification. Family holdouts cover 260 labeled
edits; the revised nested selection refits the complete permitted outer training
fold. MMLU-Pro pooled MAE is 0.12732 (descriptive edit-bootstrap percentile 95% CI
[0.11306, 0.14054]); pooled Spearman is -0.28639 ([-0.40260, -0.12550]). These
marginal intervals resample edits conditional on fitted predictions. The paired
M1-minus-StatNN MAE difference is +0.09854 (95% CI [0.08707, 0.11224]), resampling
246 full-ancestry clusters with replacement and retaining paired predictions for
all edits in each sampled cluster. This smoke used only 20 bootstrap draws and
seed 0; the intervals are coarse and do not cover benchmark-item/repeat noise or
shared-base uncertainty. Existing label/runtime qualification limitations remain.

The per-outer-test-family MMLU-Pro model report is:

| fold | family | n | MAE | Spearman |
|---|---|---:|---:|---:|
| 0 | abliteration | 98 | 0.06142 | 0.02119 |
| 1 | random_dir | 60 | 0.09976 | -0.01739 |
| 2 | lora_sft | 59 | 0.24204 | -0.15744 |
| 3 | task_vector | 30 | 0.17134 | 0.34120 |
| 4 | chain | 13 | 0.12910 | 0.75192 |

Each fold parquet also retains StatNN, fold-mean, delta-norm, descriptor-only and
device-only predictions; metrics.json contains their per-family metrics, paired
differences, centered diagnostic correlations and training-range support checks.
All artifacts are under `data/runs/p4/p4-round2-m1-family-cpu-final/`.

**Seed-1 repeat panel, same host (5 Nov 05:05–06:40 UTC; run p2-batch/20261105-050543-aad6c310, config
eval_fast_seed1 = eval_fast with generation seed 1; 10 controls + base, batch 32, device H0).** Seed 0 vs seed 1, mean /
max absolute difference and Spearman over the 11 records: gsm8k 0.055 / 0.12 / ρ 0.15; math500 0.036 / 0.12 / 0.46;
ifeval 0.029 / 0.04 / 0.10; over-refusal 0.049 / 0.10 / 0.67; refusal 0.009 / 0.04 / 0.26; loop rate 0.006 / 0.02 /
0.66; mmlu_pro, hellaswag, KL and wikitext ppl identical (likelihood-scored or teacher-forced: no sampling). Base:
gsm8k 0.16 → 0.10, math500 0.16 → 0.04, ifeval 0.40 → 0.36. Reading: seed-to-seed sampling noise on one host is the
same size as the cross-device gaps above, so those gaps are sampling noise under an effectively different draw, not
a hardware bias; at the fast tier the sampled benchmarks cannot rank small edits (ρ 0.10–0.46 against themselves),
while the likelihood/teacher-forced metrics are exactly repeatable. Qualification per Board A's rule: mmlu_pro,
hellaswag, KL and ppl pass the repeat criterion trivially (zero difference) but still need the between-entry
variance check; gsm8k, math500, ifeval and over-refusal fail it at this tier and need more items or more samples.

## TF-v1 timing pilot

Measured run `20261105-131348-277686c3` on H0: base + the ten fixed IDs in `data/runs/d3_shards/controls.txt`; primary records (`repeat_id=null`), batch 1, no sampling. Native runtime/source/item metadata and same-device base matching are in each eval.json. Source may be dirty; source hashes identify the measured code.

| Entry | Wall min | Load s (shared once) | Peak allocated/reserved GiB | TF acc (1000) | Fast acc (100) | Gold log-prob | Records MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 11.499 | 6.57 | 3.913/5.791 | 0.201 | 0.280 | -2.442048 | 2.248 |
| b6fe7f59016cdda2 | 4.666 | 6.57 | 3.913/5.859 | 0.209 | 0.280 | -2.451112 | 2.279 |
| 3ddc3b4bf47ed994 | 4.544 | 6.57 | 3.913/5.807 | 0.192 | 0.260 | -2.467125 | 2.279 |
| 0a841d4082532e8b | 4.122 | 6.57 | 3.913/5.809 | 0.175 | 0.230 | -2.569524 | 2.280 |
| 3db97b535c996265 | 3.915 | 6.57 | 3.913/5.779 | 0.205 | 0.290 | -2.441007 | 2.280 |
| 7b8247d8d7ab7db9 | 4.525 | 6.57 | 3.913/5.779 | 0.206 | 0.280 | -2.438299 | 2.280 |
| 439dbfa62a14d55f | 4.403 | 6.57 | 3.913/5.793 | 0.207 | 0.280 | -2.460094 | 2.279 |
| af6d608f8651fb46 | 11.907 | 6.57 | 3.913/5.809 | 0.221 | 0.260 | -2.440941 | 2.280 |
| dd90d99e2ddc0e77 | 3.892 | 6.57 | 3.913/5.793 | 0.205 | 0.290 | -2.439536 | 2.280 |
| db5c3a7bfc19178c | 4.652 | 6.57 | 3.913/5.809 | 0.203 | 0.290 | -2.449961 | 2.280 |
| 1caef8eb4c640c3c | 4.571 | 6.57 | 3.913/5.793 | 0.209 | 0.300 | -2.449637 | 2.280 |

Median warm entry storage (records + eval/summary/manifest): 2.322 MiB; shared base logits cache: 10.452 GiB (not duplicated per entry). The pilot also left an orphan byte-identical duplicate cache of 10.452 GiB (see caveat).

Warm median wall time over 9 edits that reused a cache: **4.525 min/entry** (all ten edits: 4.534); entry `af6d608f8651fb46` hit a cache-key defect (git sha/dirty flag in the key; a WIP commit during the run changed them) that rebuilt a byte-identical base cache (KL stage ~10 min) and is excluded from the warm median; the following three edits reused that duplicate. Pilot code `source_hash` 07ebf7b04e8b; the key is fixed afterwards in a later commit.  Model load: **6.57 s** once per batch invocation (not charged eleven times). Base includes cold greedy-continuation cache construction. The council warm timing pass criterion is ≤2.0 minutes; >3.0 minutes calls for revision and re-pilot. This fixed-count pilot does not silently reduce the requested item counts. Per-stage times are in `data/evals/qwen3.5-0.8b/tf_v1/pilot_report.json`; wall includes weight restore/application but excludes shared model load and result serialization.

Legacy bridge (11 paired records): median |accuracy(1000 new) − accuracy(100 legacy)| = **0.0740** (7.40 percentage points). Legacy run `20261104-202947-1dcb69a2`. This compares different item sets; it is not a same-item numerical-parity or qualification test.

Gold log-prob versus TF accuracy across all 11 entries: Pearson r=0.8497, Spearman rho=0.4018; excluding base (10 edits): r=0.8694, rho=0.4329. Descriptive point estimates only: related controls are not independent components; no confirmatory CI or readability claim.

TF-v1 stores normalized MMLU letter vectors, per-character HellaSwag scores, paired item IDs, KL continuation IDs/cache hashes, window NLL/token counts, and refusal/compliance token sets and margins. The lexical first-token sets are frozen in the config, exploratory and not calibrated on a separate prompt panel. Refusal uses the requested 50+50 prompts, rather than the council proposal’s 200+200. This control panel contains 3 random-direction, 3 abliteration, 2 LoRA and 2 task-vector edits, with no chains; it replaces the council proposal’s stratified panel at the user’s explicit request. No gold-solution NLL, independent repeat, cross-device execution, fresh component qualification or same-item legacy bridge was requested or performed. Endpoints remain unqualified under Board A’s prospective rule.

TF-v1 pilot reading (measured; run `20261105-131348-277686c3`, H0, RTX PRO 4500, batch 1, host shared with other agent jobs): median warm stage times are MMLU-Pro ~33 s, HellaSwag ~71 s, KL ~120–167 s (200 cached 128-token continuations: 63 MB bf16 logit tensors loaded per prompt plus full-vocabulary KL), WikiText ~4 s, refusal margins ~2 s each. **The council timing gate fails:** warm median 4.5 min/entry is above both the 2.0 min pass line and the 3.0 min revision line, so the §4.6 rule (reduce items, then re-pilot) applies; no items were reduced here. KL is the largest stage and the obvious target (cache logits as top-k or per-token KL on device); this has not been tried. Cold base construction took 11.5 min (576 s KL cache). Base cache on disk is 10.45 GiB for 200 prompts, one-off per device/source hash; a byte-identical 10.45 GiB duplicate (key `bd82d791…`) remains from the cache-key defect and may be deleted. Entries ran on the pre-fix source (`source_hash` in each eval.json); the other §4.6 items (G1 repeats, G3 second device H1, same-item G2 bridge) were not run.

## TF-v1 timing pilot (round 2)

Run `20261105-144427-fb58ffe6` on H0 (RTX PRO 4500, batch 1, greedy, no sampling), committed source `02c2847` (`source_hash` 12ee875a68c9, tree clean). Base + the ten D3 control IDs in `data/runs/d3_shards/controls.txt`, one process; native metadata and same-run base matching in each eval.json. **Panel status: development/pilot** (items scored here are barred from the sealed panel). Config: MMLU-Pro 1,000, HellaSwag 250 (nested prefix of the 500-item ranking), KL 200 prompts x 64 greedy tokens (exact full vocabulary), WikiText 64, refusal 50 + 50 continuation-phrase margins. Round 1 (`20261105-131348-277686c3`, gate 1 failed at 4.525 min) is archived under `data/evals/qwen3.5-0.8b/tf_v1_round1_archive/`.

| Entry | Wall min | MMLU s | HellaSwag s | KL s | WikiText s | Refusal s | Over-refusal s | Peak alloc/reserved GiB | Storage MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 6.308 | 41.6 | 37.0 | 269.4 | 5.0 | 12.5 | 12.9 | 9.22/9.78 | 2.13 |
| b6fe7f59016cdda2 | 1.809 | 35.0 | 36.8 | 7.7 | 4.3 | 11.9 | 12.2 | 9.22/10.13 | 2.36 |
| 3ddc3b4bf47ed994 | 1.793 | 34.5 | 36.5 | 7.6 | 4.4 | 11.9 | 12.1 | 9.22/10.15 | 2.36 |
| 0a841d4082532e8b | 1.833 | 34.6 | 36.6 | 7.7 | 4.3 | 12.1 | 12.3 | 9.22/10.24 | 2.38 |
| 3db97b535c996265 | 1.798 | 34.9 | 36.5 | 7.7 | 4.3 | 12.1 | 12.3 | 9.22/10.12 | 2.36 |
| 7b8247d8d7ab7db9 | 1.809 | 34.6 | 36.9 | 8.0 | 4.6 | 12.1 | 12.2 | 9.22/10.12 | 2.36 |
| 439dbfa62a14d55f | 1.801 | 34.9 | 37.0 | 7.7 | 4.3 | 11.8 | 12.0 | 9.22/10.16 | 2.36 |
| af6d608f8651fb46 | 1.831 | 34.4 | 36.7 | 7.6 | 4.3 | 12.1 | 12.3 | 9.22/10.24 | 2.37 |
| dd90d99e2ddc0e77 | 1.814 | 35.0 | 37.1 | 7.7 | 4.3 | 12.1 | 12.3 | 9.22/10.16 | 2.37 |
| db5c3a7bfc19178c | 1.781 | 34.5 | 36.4 | 7.5 | 4.3 | 11.8 | 12.1 | 9.22/10.17 | 2.36 |
| 1caef8eb4c640c3c | 1.784 | 34.4 | 36.4 | 7.7 | 4.3 | 11.8 | 12.1 | 9.22/10.16 | 2.36 |

**Warm median wall time over 10 edits (cache resident, loaded and verified once in the process): 1.805 min/entry** (range 1.781 to 1.833). Warm stage medians (s): MMLU 34.6, HellaSwag 36.7, KL 7.7, WikiText 4.3, refusal 12.0, over-refusal 12.2. **Gate 1 (2.0 min pass line, 3.0 min revision line): PASS.**

Cold cost: base entry 6.31 min wall including cold greedy-continuation cache construction (261 s, mode `build`); model load 7.6 s once per process.
New-process cache cost (separate process, entry `b6fe7f59016cdda2`, label `load-probe`): cache load plus content verification **47.1 s**, entry wall 2.60 min. Repeat vs primary: max |ΔKL| = 0.00e+00, max |Δgold log-prob| = 0.00e+00, records identical: True.

Storage: median 2.36 MiB per entry (records 2.32 MiB, including per-position KL); shared base-logit cache 5.38 GiB on disk (manifest bytes 5.38 GiB, 3 prefix lengths recorded), content root `2ed18bda284b3a89…` recorded in every KL record, datasets and eval.json. Peak GPU allocated 9.22 GiB with the resident cache.

Mean per-prompt KL(base‖edited) by continuation prefix length across the ten edits (descriptive; for choosing the token cap): 16 tokens: median entry value 0.00144, 32 tokens: median entry value 0.00153, 64 tokens: median entry value 0.00160.

Not run: G1 bitwise repeat on two entries, same-item G2 bridge, G3 second device, the §6 ten-entry composition (these are the ten D3 controls: 3 random-direction, 3 abliteration, 2 LoRA, 2 task-vector), and the reliability procedure. Endpoints remain unqualified.

Reading and caveats (round 2). Gate 1 passes on the 2.0 min line (warm median 1.805 min) but not with the margin Board A asked for (target ≤1.8 min): at 1.805 min, 552 runs cost about 16.6 h of the 19.0 h scoring cap, leaving about 2.4 h for cold bases, retries and qualification repeats, so the full programme must still be repriced before scaling. The warm time is dominated by MMLU-Pro (35 s), HellaSwag (37 s) and the two refusal stages (12 s each; 13 continuation phrases, 7 forwards per prompt); KL is now 7.7 s because the cache is resident. The cache costs 47 s to load and verify in each new process (not per entry) and about 4.4 min to build cold. Refusal margins are phrase-scored but still uncalibrated and descriptive. The repeat of one entry in a separate process reproduced all records exactly; this is one observation, not the two-entry G1 test. Round-1 caches `5a5159df…` and `bd82d791…` (content-identical) were deleted after this cache was built and verified; the alias is recorded in `tf_v1_round1_archive/cache_alias.json`.

## TF-v1 G1 (bitwise repeat on the PREREG section 6 composition)

Date 5 Nov 2026, device H0 (RTX PRO 4500), batch 1, greedy, committed source (`source_hash` 12ee875a68c9, identical to the timing pilot; branch `feat/tf-v1-prepanel` @ 3b66f30 for the primary pass, @ c192f82 for the repeat pass, tree clean in both, `dirty: false` in every eval.json). Panel status: development.

**Composition.** PREREG §6 fixes the ten-entry composition by rule (4 LoRA: two nearest the median legacy KL of the D3-evaluated lora_sft entries and the two largest; 3 chains with >= 4 nodes; 2 task-vector; 1 largest-KL abliteration; unspecified choices and ties by sorted ID), it does not leave it to the ten D3 controls, which have no chain and no 4-node lineage and are therefore not used for G1. The rule is implemented in `scripts/tf_v1_qualification_panel.py` and the result is committed at `docs/prereg/qualification_panel.json`. Legacy KL (fast tier) selects this development pilot only.

| Group | Entries |
|---|---|
| LoRA, nearest median legacy KL (median 0.0413) | `2f61a4086395b719`, `b5d0ac5d76c3cad6` |
| LoRA, largest legacy KL | `d7f933d3b4e5d25e`, `c3ae3e8c8edca6b8` |
| Chain, step >= 3 (4th node or later), distinct lineages | `0027efdf2eae7405`, `00f8111cbad8218f`, `057dcfbedea3251f` |
| Task vector | `023b384a7799a787`, `02a1b5e21971290f` |
| Abliteration, largest legacy KL | `5c766ee2fa8bf915` |
| Chain predecessors (scoring references, needed for the predecessor-relative endpoint) | `b7edbb9f8b339f52`, `34ee195e55227908`, `86ea1fe0ec777cf8` |

The predecessor-relative MMLU endpoint needs each chain entry's actual predecessor, so G1 covers 13 entries plus the base (14 repeats). All 13 were scored in a new process (cache loaded and verified once, base reused from the pilot run `20261105-144427-fb58ffe6`, matched by the registered reference keys, `matched_base_missing: false`), then re-executed in a second fresh process as `repeats/g1` (immutable run directories, a new run id, the base re-executed and paired in the same process, cache verified again). Cache modes: `load` for the first entry of each process, `resident` after.

**Result: all 14 repeats are bitwise-identical to their primaries.** Complete `records.jsonl` equality (all 1,614 records per entry, every benchmark and field) holds for every entry; maximum absolute differences over all entries are exactly 0 for the MMLU-Pro gold log-prob and option log-probs (1,000 items), HellaSwag scores (250), per-prompt and per-position KL (200 x 64), WikiText mean NLL (64 windows) and refusal / over-refusal phrase margins (50 + 50). Limits registered in §6 are 1e-4 nats (MMLU, log-PPL) and 1e-5 (KL): met with zero discrepancy. Report: `data/evals/qwen3.5-0.8b/tf_v1/g1_report.json` (`scripts/tf_v1_g1_report.py`).

Both resident-cache and fresh-load modes are covered (the primary pass loaded the cache for its first entry; the repeat pass loaded it for the base). Warm wall time per entry in these runs was 1.77 to 1.94 min (median 1.80 min for the repeat pass) except where the delta-hashing job for the sealed partition read 255 GB from the same disk (primary-pass entries up to 1.94 min, first-entry cache loads 43 s and 217 s). The timing pilot, not this run, is the gate-1 measurement.

What G1 does not show: it is same-device repeat determinism on H0 (single device; see the D2 addendum "H0 is the sole scoring device"), not cross-device, cross-batch-size or cross-driver agreement, and it qualifies no endpoint by itself (reliability and delta/2 gates are in `docs/reviews/P5/tf-v1-qualification-2026-11-05.md`).

## TF-v1 G2 (same-item legacy bridge)

Date 5 Nov 2026, device H0, batch 1, source `source_hash` 12ee875a68c9 (unchanged: the bridge is `scripts/tf_v1_g2_bridge.py`, outside `src/wsp`), tree clean at c192f82 (`dirty: false`). The TF-v1 panels deliberately exclude the 300 legacy items, so the pilot could not compare scoring item by item. This run scores the fast tier's own **100 MMLU-Pro and 100 HellaSwag items** (the first 100 ids of `configs/eval_subsets/mmlu_pro.txt` and `hellaswag.txt`, exactly the fast tier's `n: 100`) with the TF-v1 scorers (`letter_scores`/`choice_record` for MMLU-Pro, the same `choice_scores(..., True)` call for HellaSwag) on the 11 pilot records (base + the ten D3 controls), and compares each item with the same entry's fast-tier `records.jsonl`. Outputs: `data/evals/qwen3.5-0.8b/tf_v1/bridge_g2/` (records per entry, `g2_report.json`).

| Entry | MMLU acc fast | MMLU acc TF-v1 | MMLU \|Δacc\| | MMLU correctness agreement | MMLU argmax agreement | MMLU gold-lp \|Δ\| median / p99 / max | HellaSwag acc fast / TF-v1 | HellaSwag correctness agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0a841d4082532e8b | 0.23 | 0.23 | 0.00 | 1.00 | 1.00 | 0.0e+00 / 0.036 / 0.053 | 0.43 / 0.43 | 1.00 |
| 1caef8eb4c640c3c | 0.30 | 0.30 | 0.00 | 1.00 | 1.00 | 0.0e+00 / 0.032 / 0.065 | 0.39 / 0.39 | 1.00 |
| 3db97b535c996265 | 0.29 | 0.29 | 0.00 | 1.00 | 1.00 | 1.1e-16 / 0.036 / 0.050 | 0.39 / 0.39 | 1.00 |
| 3ddc3b4bf47ed994 | 0.26 | 0.26 | 0.00 | 1.00 | 0.99 | 2.2e-16 / 0.019 / 0.055 | 0.39 / 0.39 | 1.00 |
| 439dbfa62a14d55f | 0.28 | 0.29 | 0.01 | 0.99 | 0.99 | 0.0e+00 / 0.029 / 0.099 | 0.38 / 0.38 | 1.00 |
| 7b8247d8d7ab7db9 | 0.28 | 0.28 | 0.00 | 1.00 | 1.00 | 0.0e+00 / 0.039 / 0.047 | 0.39 / 0.39 | 1.00 |
| af6d608f8651fb46 | 0.26 | 0.26 | 0.00 | 1.00 | 1.00 | 2.2e-16 / 0.025 / 0.085 | 0.43 / 0.43 | 1.00 |
| b6fe7f59016cdda2 | 0.28 | 0.28 | 0.00 | 1.00 | 1.00 | 0.0e+00 / 0.025 / 0.030 | 0.39 / 0.39 | 1.00 |
| base | 0.28 | 0.28 | 0.00 | 1.00 | 1.00 | 0.0e+00 / 0.066 / 0.075 | 0.39 / 0.39 | 1.00 |
| db5c3a7bfc19178c | 0.29 | 0.29 | 0.00 | 1.00 | 1.00 | 0.0e+00 / 0.013 / 0.045 | 0.39 / 0.39 | 1.00 |
| dd90d99e2ddc0e77 | 0.29 | 0.29 | 0.00 | 1.00 | 1.00 | 0.0e+00 / 0.033 / 0.044 | 0.39 / 0.39 | 1.00 |

**Agreement.** Per-item correctness agrees on 99.91% of the 1,100 MMLU-Pro (entry, item) pairs (worst entry 99%; one flip) and on 100.00% of the 1,100 HellaSwag pairs. Argmax (predicted choice) agrees on 99.82% for MMLU-Pro (worst entry 99%; two disagreements, both bf16 logit ties or one-step gaps: item 2168 in `3ddc3b4bf47ed994` has a 0.0625 gap in the fast tier and an exact tie in TF-v1, item 1889 in `439dbfa62a14d55f` the reverse) and 100.00% for HellaSwag.
**Accuracy |Δ| on the same items.** MMLU-Pro: mean 0.0009, max 0.01 (one item in 100, entry `439dbfa62a14d55f`), 10 of 11 entries exactly equal. HellaSwag: 0 for every entry.
**PREREG §6 bridge criteria (MMLU-Pro normalized gold-letter log-prob, worst entry for each statistic).** median |Δ| 2.2e-16 (limit .02), p99 0.066 (limit .2), argmax agreement 99% (limit 99%): **PASS**. HellaSwag scores are bitwise identical to the fast tier in all 11 entries.

What the difference is. About 47% of MMLU-Pro (entry, item) pairs (520 of 1,100) differ in the last digits, with median nonzero |Δ| of 1e-8 and a tail up to 0.099 nats. The fast tier already scored the answer letters (` A`..`J`) with a full-sequence forward and `choice_scores`; TF-v1 scores the same letter log-probs from a prefix-only forward that keeps only the last position (`logits_to_keep=1`). The model emits bf16 logits, so a different reduction order moves a logit by one bf16 step (0.0625 to 0.125 at logit magnitude 8 to 32), which is the 0.03 to 0.10 tail; it can also create or break exact ties in the argmax. So the bridge compares two near-identical scorers, not option-text versus letter likelihood as PREREG §6 words it: the legacy MMLU score was already letter-based. It is therefore a same-device implementation-parity test (it passes), and it does not show that old and new values pool across devices or batch sizes. Caveats: the fast-tier records carry backfilled (`inferred`/`transcribed`) runtime metadata (DECISIONS D2), so this compares item values, not matched-runtime references; `correct` on bf16 ties depends on the first-index rule in both tiers.

A G2 pass does not qualify signal or effect resolution (PREREG §6): the reliability and δ/2 gates are in `docs/reviews/P5/tf-v1-qualification-2026-11-05.md`.
