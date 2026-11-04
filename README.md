<p align="center"><img src="assets/hero.svg" alt="edit trajectory → weight-delta tokens → state-space predictor → performance forecast" width="820"></p>
<p align="center"><img src="assets/pipeline.gif" alt="Animated weight-edit prediction pipeline" width="820"></p>

<h1 align="center">wsp-lab</h1>
<p align="center"><b>Can a small state-space model predict how an LLM's benchmark scores will change from the weight change alone?</b></p>
<p align="center"><a href="docs/HYPOTHESIS.md">Hypotheses</a> · <a href="docs/ROADMAP.md">Roadmap</a> · <a href="docs/MODELS.md">Models</a> · <a href="docs/RESULTS.md">Results</a> · <a href="paper/">Paper</a> · <a href="https://ariomoniri.github.io/wsp-lab/">Animated overview</a></p>

Weight-space programming (Takehi et al., MIT CSAIL, Nov 2026) showed a frontier LLM reading and writing weights
directly. This repository tests a cheaper, complementary claim: a dedicated predictor trained on a zoo of
(weight delta → performance delta) pairs can forecast benchmark changes after each edit in a trajectory, with
no forward pass of the edited model. Edits come from abliteration (classic and Heretic), LoRA fine-tunes,
task-vector merges and random-direction controls, on Qwen3.5-0.8B up to Qwen3.8-27B. Backbones compared:
a Mamba-2 state-space model over the edit sequence, a JEPA weight encoder (JEPAWG-style), a stateless
Transformer over W2T-style canonical SVD tokens, LightGBM on weight statistics, and frontier LLMs reading the
same deltas as text.

## Layout
| Path | Purpose |
|---|---|
| `src/wsp/` | `zoo` (edit families), `eval` (cheap battery), `features` (φ_stats, φ_svd), `predictor` (SSM / JEPA / Transformer), `agents` (LLM reader) |
| `configs/` | experiment configs; `tests/` pytest tiers (`-m gpu`, `-m slow`) |
| `docs/` | hypotheses, roadmap, models, infra, review protocol, writing & figure style, venue |
| `paper/` | TMLR manuscript built section-by-section after review-board approval |
| `board/`, `reviews/`, `sessions/`, `harness/` | agent message board mirror, review records, session journal, harness lessons |
| `assets/lottie/`, `site/` | animations (Lottie JSON) and the GitHub Pages overview |

## Use
```bash
git clone https://github.com/ArioMoniri/wsp-lab && cd wsp-lab
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate && pytest -q
python -m wsp.zoo.build --config configs/zoo_qwen3.5-0.8b.yaml      # build the edit zoo
python -m wsp.eval.run  --config configs/eval_cheap.yaml --zoo data/zoo/qwen3.5-0.8b
python -m wsp.predictor.train --config configs/predictor_ssm.yaml
```
CI runs on a self-hosted GitLab runner with GPU passthrough (`.gitlab-ci.yml`); GitHub Actions is not used.

## Status
P0 infrastructure complete (CI on self-hosted GPU runner, Mamba-2 kernels built for Blackwell); P2 eval battery merged; P1 edit zoo in final review. See `docs/ROADMAP.md` and `docs/RESULTS.md`.
