<p align="center"><img src="assets/hero.svg" alt="edit trajectory → weight-delta tokens → state-space predictor → performance forecast" width="820"></p>

<h1 align="center">wsp-lab</h1>
<p align="center"><a href="https://gitlab.com/ArioMoniri/wsp-lab/-/pipelines"><img src="https://gitlab.com/ArioMoniri/wsp-lab/badges/main/pipeline.svg" alt="pipeline status"></a></p>

## The question

If you change the weights of a language model, can you tell from the weight change alone how its benchmark scores moved, without running the edited model?

The question comes from a blog post by Takehi et al. on weight-space programming (<https://rikiyat.github.io/blog/weight-space-programming/>), in which a frontier model read Qwen3.5-0.8B's weights and edited 0.016% of them, which raised MATH-500 from 7.9% to 48.1%. We are not trying to repeat that. We want to know whether a small model trained on many (weight delta, score change) pairs can do the forecasting step by itself, and whether it does better when it sees the whole sequence of edits instead of one edit at a time. The hypotheses, with the thresholds we committed to before running anything, are in [docs/HYPOTHESIS.md](docs/HYPOTHESIS.md).

## What exists today

The zoo builder, the evaluation battery and the feature extractor are written, tested and reviewed. The zoo is still growing, and apart from the base model and a two-entry pipeline smoke test none of it has been scored. No predictor has been trained yet.

The zoo is a set of edits to Qwen3.5-0.8B, each stored as a weight delta plus a JSON descriptor with seed, config hash, git SHA and the pinned base revision. There are just over a thousand entries; the table in the status section below gives the current count per family. Entries come from six families:

| family | what the edit is |
|---|---|
| `random_dir` | ablation of a random direction; the control |
| `abliteration` | refusal-direction removal using Heretic's kernel, with its parameters sampled rather than optimised |
| `heretic_ara` | Heretic ARA runs |
| `lora_sft` | short LoRA fine-tunes on GSM8K training data or on the first 5,000 Alpaca-cleaned examples |
| `task_vector` | scaled and merged task vectors |
| `chain` | two to eight edits applied in sequence, each drawn from the families above |

Heretic ARA is the slow one, at roughly eight minutes per trial on the shared 32 GB GPU, so it was moved to a rented Lambda GPU under a $150 cap. A first job was launched on 4 November; until it finishes and syncs, the zoo holds a single test entry from that family. Merging a LoRA delta into bf16 weights rounds away most of a small adapter: about 87% of the non-zero entries for a 5-step adapter in the check we ran. So we store the effective delta, `bf16(W + Δ) − W`, and evaluate that; [docs/DECISIONS.md](docs/DECISIONS.md) has the reasoning (D1).

The evaluation battery has three tiers because the base model gets stuck in repetition loops in most of its generations, which makes long generations expensive whatever the score. The fast tier screens zoo entries at 1,024 new tokens on small fixed subsets of GSM8K, MATH-500, IFEval, MMLU-Pro and HellaSwag, plus refusal rate and KL to the base model. The cheap tier runs the full subsets at 4,096 tokens. The faithful tier uses the blog's 32,768 tokens and is meant only for the base model and a handful of entries. Subset item lists are fixed under `configs/eval_subsets/`, and every per-item record is kept.

The base model has been measured once at the cheap tier, run `p2/base-final-4096`: GSM8K 15.5%, MATH-500 6.0%, IFEval 42.0%, and 90.2% of generations on the three generative benchmarks reaching the token limit. GSM8K and MATH-500 land near the blog's numbers at an eighth of its token budget, and the loop rate agrees with its main failure mode. The IFEval figure is about twice what the blog reports and we do not trust it until the 32k pass is rerun; that pass died with a CUDA out-of-memory error while the zoo build was using the GPU. Details and the full table are in [docs/RESULTS.md](docs/RESULTS.md).

Feature extraction turns each delta into per-tensor statistics and canonical SVD tokens in the style of W2T, then pools them to fixed-width vectors by tensor type, with extra care for Qwen3.5's hybrid linear-attention layers. Trajectory splits group entries by the root of their ancestry, so a chain and its parents never fall on opposite sides of a split. Extraction reads every valid manifest id, including `smoke` and `unlabelled` entries, and skips entries whose artefacts are not built yet, entries already extracted under the same configuration, and entries whose metadata fails validation (those are recorded as failed); filtering by eligibility is left to whoever trains on the features. Three baselines are written and tested on a toy zoo: ridge regression on the delta norm, LightGBM on the statistics, and an MLP on the spectrum. None has seen real scores, because no production entry has a fast- or cheap-tier `eval.json` yet. The only per-entry results so far are an engineering smoke run (`20261104-032002-1c2cc588`: the base model and two production edits, 4 items per benchmark, 32 new tokens), kept under `data/runs/p2-batch/legacy-smoke`. It tested the batch pipeline and its timing. It measured pipeline timing on four items per benchmark, far too few to say anything about the edits' effect on scores.

The sequence model will be a Mamba-2 stack. On this machine (CUDA 13.2, compute capability 12.0) neither `mamba-ssm` nor `causal-conv1d` built out of the box; `scripts/build_mamba.sh` patches both to C++20 and takes about eleven minutes. A forward and backward pass at batch 32, length 64, width 256 took 2.65 ms median with the CUDA kernels against 18.67 ms for a plain PyTorch loop, measured while another job held the GPU ([docs/BENCH_SSM.md](docs/BENCH_SSM.md)).

The review protocol asks for two reviews before a branch merges. Board A looks for ML errors such as leakage, unseeded randomness, silent fallbacks and missing provenance; Board B reads the way a journal reviewer would. Many branches were sent back for fixes, and PROGRESS records the ones from the first night. [docs/PROGRESS.md](docs/PROGRESS.md) is the dated log of what was tried and what the boards rejected, and [docs/REVIEW_PROTOCOL.md](docs/REVIEW_PROTOCOL.md) describes the process.

## Running it

```bash
git clone https://github.com/ArioMoniri/wsp-lab && cd wsp-lab
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
source scripts/env.sh && pytest -q          # CPU tier; GPU tests need -m gpu
bash scripts/build_mamba.sh                 # only for the CUDA Mamba-2 backend

python -m wsp.zoo.build --config configs/zoo_qwen3.5-0.8b_local.yaml   # every family except Heretic ARA
python -m wsp.eval.run --config configs/eval_cheap.yaml --model Qwen/Qwen3.5-0.8B --out data/runs/p2/<run-id>
python -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --subset d3        # fast tier on the D3 subset
python -m wsp.features.extract --zoo data/zoo/qwen3.5-0.8b --out data/features/qwen3.5-0.8b --workers 4
python scripts/update_readme_status.py                                  # refresh the status block below
```

Anything over two minutes should go through `agent-tmux <name> <cmd>` on the lab server. Weights, deltas and eval records stay out of git. They live in private Hugging Face datasets and sync with `scripts/hf_sync.sh`. A pre-push hook runs `scripts/secret_scan.sh --history`; pre-commit runs ruff and the CPU tests. CI runs on a self-hosted GitLab runner with GPU passthrough (`.gitlab-ci.yml`).

## What is not done

The fast tier has not been run over the zoo, so no zoo entry has a score a predictor could learn from. Board A's calibrated estimate was 9 to 12 minutes per entry, which came to about 120 GPU-hours for the 700 entries that existed then. The manifest now lists over 1,100, so the full run would take longer, and that figure is a projection from a smoke timing, not a measurement. Decision D3 therefore starts with a stratified subset of about 300 entries, roughly 50 hours. Still pending: a clean timing of one entry, the second base-model reproduction run that the P2 gate requires, and a rerun of the 32k base pass.

Next comes the first predictor comparison, starting with the baselines on D3 scores; [docs/ROADMAP.md](docs/ROADMAP.md) lays out the remaining phases and their gates. `src/wsp/predictor/` currently holds the scan kernels, splits and baselines, with no training script. The paper in `paper/` has an abstract, introduction, related work, and method, zoo and experiment sections that describe the design; it reports no experimental results.

## Limitations

**Coverage.** The zoo has one base model, Qwen3.5-0.8B, and one tokenizer, so nothing here says anything about larger models or other architectures. The 27B transfer test is meant to probe that and has not been started.

**Scoring.** Refusal rate is a keyword match on 100 prompts, and the base model refuses 99 of them, so mild edits have little room to show. The fast and cheap tiers use 25 to 300 items per benchmark, so a single entry's score change will often sit inside sampling noise, and many entries will differ from the base by less than that.

**The edits themselves.** Heretic's abliteration kernels are sampled from a hand-set parameter range instead of being found by its optimiser, so those entries are partial abliterations of widely varying strength, not points on the refusal/KL frontier Heretic would pick. Our kernel also skips the row normalisation that Heretic's default settings apply. The LoRA fine-tunes are short, and the bf16 rounding above removes part of the intended change. Chains reuse parents from the same zoo, so their targets are correlated even with ancestry-grouped splits.

**Review.** Every Board review so far was run by language models, not people, and the paper has not been through an outside reviewer.

## Layout

| Path | Contents |
|---|---|
| `src/wsp/zoo/`, `src/wsp/edits/` | edit generators, the Heretic-parity kernels, refusal-direction extraction, zoo builder and applier |
| `src/wsp/eval/` | benchmark runner, batch evaluator, IFEval checker, report generator |
| `src/wsp/features/`, `src/wsp/predictor/` | feature extraction and pooling, splits, baselines, selective-scan code |
| `configs/` | zoo and eval configs, fixed benchmark subsets |
| `docs/` | hypotheses, roadmap, decisions, results, infrastructure, writing and review rules |
| `board/`, `reviews/`, `sessions/`, `harness/` | agent message board, review records, session notes, lessons |
| `paper/` | TMLR manuscript sections as they pass review |

## Status

<!-- status:start -->

_Generated by `scripts/update_readme_status.py` from artefacts on 2026-11-04T23:53:19Z._

**Zoo** (`data/zoo/qwen3.5-0.8b`): 1235 entries in the manifest.

| family | production | unlabelled | total |
|---|---:|---:|---:|
| random_dir | 150 | 3 | 153 |
| abliteration | 200 | 5 | 205 |
| heretic_ara | 0 | 1 | 1 |
| lora_sft | 120 | 1 | 121 |
| task_vector | 80 | 0 | 80 |
| chain | 655 | 20 | 675 |

`smoke` entries come from short test builds and `unlabelled` rows predate the eligibility field. The batch evaluator takes only `production` entries by default; feature extraction reads every valid manifest id, whatever its eligibility, and skips entries whose files are not built yet.

**Features**: 1235 entries extracted (finished).

**Evals**: 112 index rows, 112 entries, 112 with a complete result.

**Base model** (Qwen3.5-0.8B, run `p2/base-final-4096`, 4096 new tokens, percent):

| GSM8K | MATH-500 | IFEval | MMLU-Pro | HellaSwag | loop rate | refusal rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15.5 | 6.0 | 42.0 | 20.3 | 51.3 | 90.2 | 99.0 |

**Lambda**: 1 open ledger row(s): 73522ff7f7c7422dae09f55f8055d9d1 (gpu_1x_a100_sxm4 (us-west-2), heretic_ara 12×10 rebuild with fixed exporter (100c28a), since 2026-11-04T10:16:06Z).

**CI** (GitLab, `main`): pipeline 3029227189 on `ffb5c9d`, status failed.

<!-- status:end -->
