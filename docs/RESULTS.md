# Results

(empty — filled by P2+ runs; every number carries run id, n, CI method)

| Phase | Table | Run id | Status |
|---|---|---|---|
| P2 | Base model reproduction vs blog table | p2/base-final-4096 | 4k pass done; 32k rerun pending |
| P3 | Baselines by family / by seed | — | pending |
| P4 | H1–H4 | — | pending |
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
Projection: D3 subset (300 entries) ≈ 62 h on this GPU with the HF path; full zoo ≈ 255 h. The throughput
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
