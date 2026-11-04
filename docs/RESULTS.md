# Results

(empty — filled by P2+ runs; every number carries run id, n, CI method)

| Phase | Table | Run id | Status |
|---|---|---|---|
| P2 | Base model reproduction vs blog table | p2/base-final-4096 | 4k pass done; 32k rerun pending |
| P3 | Baselines by family / by seed | — | pending |
| P4 | H1–H4 | — | pending |
| P5 | 27B transfer | — | pending |

## P2 reproduction status

Implementation is on `feat/p2-eval`. CPU tests and the eight-item, two-run GPU
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
