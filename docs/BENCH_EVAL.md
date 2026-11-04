# Evaluation throughput investigation

D2 uses 1024 generation tokens, batch 32, and exactly `configs/eval_fast.yaml` subsets,
including WikiText and 50/50 refusal/KL prompts. Each path evaluates the same production
abliteration entry. Wall minutes exclude initial HF model loading and delta application;
they include all benchmark loading, generation, choice scoring, perplexity and KL work.
vLLM wall time additionally includes snapshot export and engine startup. Tokens/s counts
actual generated tokens (including EOS, excluding padding), divided by generation wall time;
it is not full-entry throughput. No faithful-tier projection is implied.

## Compatibility and environment

Checked 4 Nov 2026: the [vLLM model registry](https://docs.vllm.ai/en/latest/models/supported_models/)
and [Qwen3.5 implementation](https://docs.vllm.ai/en/latest/api/vllm/models/qwen3_5/)
include Qwen3.5's hybrid gated-delta architecture. The
[GPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
requires CUDA >=12.8 for Blackwell. These establish a candidate path, not proof that every
SM120 gated-delta kernel works. Installed vLLM 0.33.0 has `Qwen3_5ForCausalLM` in its registry;
its own dependencies select torch 2.14.0+cu130 and Transformers 5.20.2.
The installed torch reports CUDA 13.0 and includes `sm_120` in its compiled architecture list;
that verifies the torch wheel, while vLLM gated-delta kernel validation still needs the GPU probe. The isolated Python 3.12
`.venv-vllm` leaves `.venv` untouched. Full pins: `data/runs/perf/vllm-freeze.txt`;
installation log: `/home/ubuntu/agent-logs/perf-vllm-install-20261104-071212.log`.

The adapter exports the effective edited bf16 weights; vLLM generates while HF retains
teacher-forced choice/PPL/KL scoring. The same evaluator writes per-item `EvalRecord`s.
KL base continuations use a separate base snapshot and restore the edited HF weights.
This is a hybrid evaluation path, so engine startup is charged rather than hidden.
The worker uses eager execution, no prefix caching, a context cap of 8192 and 55% GPU memory.

## Measurements

The greedy probe warms all 32 lanes with 16 tokens, then measures 32 GSM8K prompts at
1024 tokens and temperature 0. Probe agreement compares decoded output, token count and
hit-budget flag, not stochastic D2 output. Static/compile is eligible only on exact HF
probe agreement. Production settings retain temperature 0.6 and presence penalty 1.5.

Full zoo denominator: **1205 production entries** at report time. D3 is 300 entries; the original D3 budget also referenced a 700-entry zoo. Projections multiply measured minutes/entry by count / 60 and assume serial entries with the same workload.

| Path | Tokens/s | Minutes/entry | D3 300 (h) | Full zoo (h) | Original 700 (h) |
| --- | ---: | ---: | ---: | ---: | ---: |
| hf-b32 | Not measured | Not measured | — | — | — |
| hf-best | Not measured | Not measured | — | — | — |
| vllm | Not measured | Not measured | — | — | — |

Recommendation pending GPU measurements. Do not infer throughput from CPU tests or model support listings.

## Validation and reproduction

CPU regressions cover left padding, prompt/continuation cropping, EOS/padding token counts,
partial batches, presence penalty, and scalar versus padded choice likelihoods. The slow
GPU test compares greedy B1 dynamic against B2 static on the real model. See
`data/runs/perf/cpu-tests.log` and `data/runs/perf/gpu-test.log`.

Run `agent-tmux perf-benchmark bash scripts/perf_eval.sh`. The job enforces the literal
reservation gate until neither `p1-zoo-full:` nor `post-zoo:` is listed. Gate log:
`/home/ubuntu/agent-logs/perf-wait-20261104-071148.log`; benchmark log:
`/home/ubuntu/agent-logs/perf-benchmark-20261104-071739.log`. Per-probe failures and full
entry logs are in `data/runs/perf/`. All perf outputs are isolated from canonical evals.
Run `python scripts/perf_report.py` after the job finishes to regenerate this table.
