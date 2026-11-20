# WM-E stage W2 addendum: empirical noise floor of the path score E (20 Nov 2026)

> **DESCRIPTIVE addendum to W2** (`docs/reviews/wme/wme-stagew2-2026-11-18.md`), implementing SYNTHESIS-r13 R2
> (`docs/council/2026-11-20/SYNTHESIS-r13.md`). Nothing here is registered or confirmatory, and it does not change
> the W2 verdict (FAIL). CPU only, $0. Read only exposed W2 artefacts (`data/wme/stagew2/run-20261118`) and, as a
> secondary pair, the exposed stage-W run (`data/wme/stagew/run-20261118/remote/w`). `data/evals`, sealed splits
> and `docs/PREREG_*` were not touched. Machine-readable copy:
> `docs/reviews/wme/wme-stagew2-noisefloor-2026-11-20.json`.

## BLUF

- **Empirical floor (perfect predictor): E = 0.357, 95 % item-bootstrap CI [0.323, 0.389]**, from 1 W2 replicate
  pair (`base` vs `base_rerun`, slot a; 500 items per benchmark, 4 endpoints).
- This is less than half the pooled curve's E (0.771 at k = 0, 0.881 at k = 2) and well below the Gaussian bound
  √(2/π) ≈ 0.80. That bound assumes the run-to-run noise equals the binomial SE. In fact the replicate noise SD is
  only **0.45 SE** (CI [0.40, 0.49]): on a fixed panel, item sampling is shared by every member and is not noise
  for the predictor. Only the generation noise (vLLM batch non-invariance) is.
- **Verdict under the pre-declared rule:** the floor is far below pooled E − 0.05 (0.721 / 0.831) in both settings,
  so **(a) is not closed on 9B by the noise floor. The headroom is 0.414 at k = 0 and 0.523 at k = 2** (CI-end
  ranges [0.382, 0.448] and [0.491, 0.558]). Every sensitivity variant below leads to the same verdict.
- Most of the headroom is on truncation: about 0.5–0.7 per truncation endpoint, against about 0.22–0.35 per
  accuracy endpoint.

## 1. Definition

E is defined as in W2 (`wsp.wm.stagew2.score`): the mean of |pred − y| / SE(y) over the points after k and the
4 endpoints (accuracy and truncation on MMLU-Pro and SuperGPQA). SE(y) = √(max(y(1−y), 1/N)/N) with N = 500
(`wsp.wm.stagew.se_of`). A truncated response is never correct, as in `wsp.wm.stagew.cell_counts`.

A perfect predictor knows the expected value of a state's measurement on the fixed panel, so it scores only that
measurement's noise. For a replicate pair, let the item-level differences be d_i = x1_i − x2_i for each endpoint:

- σ = √(var(d) / 2N) is the single-measurement noise SD, since Var(y1 − y2) = var(d)/N spread over two noisy
  copies.
- **floor** = √(2/π) · σ · mean(1/SE(y1), 1/SE(y2)), averaged over the 4 endpoints with equal weight, as in E. This
  is the primary estimator. It uses all 500 item differences, not a single |y1 − y2| draw.
- **e_replicate** = mean over both directions of |y1 − y2| / SE(target). This is the E of predicting one measurement
  by its exact replicate (one draw per endpoint). Dividing by √2 gives **floor_plugin**, rescaled to one
  measurement's noise.
- CI: 10,000 item-bootstrap draws (items resampled in pairs within each benchmark; seed 20261120), percentile 95 %.
- Pooled E is recomputed per endpoint with the W2 code (leave-one-run-out, s = 0, `fit`/`predict`/`score`). It
  reproduces the reported values: 0.7708 (k = 0) and 0.8805 (k = 2) over 45 paths.

## 2. Replicate measurements available

| pair | state | relation | n items / benchmark | discordance MMLU-Pro / SuperGPQA | used |
|---|---|---|---|---|---|
| W2 a: `base` vs `base_rerun` | base model | same engine, reversed item order | 500 | 0.058 / 0.082 | **primary** |
| W2 a vs b, a vs c: `base` | base model | same order, different hosts | 500 | 0 / 0 (bitwise identical) | reported (floor 0) |
| stage W: `base` vs `base_rerun` | base model | stage-W panel (other items), same engine | 500 | 0.046 / 0.092 | secondary |
| smoke runs (W2 smoke-1915, stage-W smoke-0622) | base model | 12 items | 12 | — | not used (too small) |

The W2 run has no other repeated measurement. Every member was evaluated once, and the three paths of a run share
one evaluation of their common final point. Same-order bases are bitwise identical across hosts (and the canaries
repeat bitwise), so all the replicate noise comes from batch composition and order.

**n of replicate pairs: 1 in W2 (primary), 2 with the stage-W pair (secondary).**

## 3. Per-objective breakdown (primary W2 pair)

| endpoint | y base / rerun | discordance | σ/SE | **floor** [95 % CI] | e_replicate | floor_plugin | pooled E k = 0 | pooled E k = 2 | headroom k = 0 / k = 2 |
|---|---|---|---|---|---|---|---|---|---|
| acc MMLU-Pro | 0.686 / 0.676 | 0.058 | 0.365 | **0.291** [0.237, 0.340] | 0.480 | 0.339 | 0.514 | 0.639 | 0.223 / 0.348 |
| acc SuperGPQA | 0.362 / 0.348 | 0.082 | 0.423 | **0.337** [0.286, 0.386] | 0.654 | 0.463 | 0.564 | 0.677 | 0.227 / 0.340 |
| trunc MMLU-Pro | 0.168 / 0.168 | 0.080 | 0.535 | **0.427** [0.363, 0.489] | 0.000 | 0.000 | 0.938 | 1.157 | 0.511 / 0.730 |
| trunc SuperGPQA | 0.330 / 0.344 | 0.098 | 0.468 | **0.373** [0.321, 0.421] | 0.662 | 0.468 | 1.066 | 1.049 | 0.693 / 0.676 |
| **mean (= E)** | | | **0.448** [0.404, 0.488] | **0.357** [0.323, 0.389] | 0.449 | 0.318 | **0.771** | **0.881** | **0.414 / 0.523** |

"Discordance" is the share of items whose indicator differs between the two generations. The plug-in values come
from a single draw per endpoint: trunc MMLU-Pro happens to be exactly 0, and the bootstrap CI of the plug-in mean
is wide, [0.16, 0.91]. The flip-based floor is the stable estimate.

Stage-W secondary pair: floor 0.337 [0.301, 0.371]. Per endpoint: acc 0.268 / 0.354, trunc 0.370 / 0.359. Pooled
with W2 (2 pairs): **0.347 [0.322, 0.370]**.

## 4. Comparison

| quantity | value |
|---|---|
| empirical floor (W2 pair) | 0.357 [0.323, 0.389] |
| empirical floor (W2 + stage-W pairs) | 0.347 [0.322, 0.370] |
| floor at k = 2 if the perfect predictor is also offset by its mean residual on the 2 noisy given points (× √1.5) | 0.437 |
| theoretical Gaussian floor with noise = SE, √(2/π) | 0.798 |
| oracle E with SE-level noise (SYNTHESIS-r13 R2, simulated) | 0.690 |
| pooled E, k = 0 / k = 2 | 0.771 / 0.881 |
| nested WM E, k = 0 / k = 2 (W2) | 0.768 / 0.881 |

√(2/π) is not the right floor for this metric. It treats a fresh binomial draw of items as noise. The panel is
fixed, and the replicate SD is about 0.45 SE, which puts the floor at about 0.45 · 0.80 ≈ 0.36.

## 5. Verdict under the pre-declared rule

Rule (SYNTHESIS-r13 decision table): if floor ≥ pooled E − 0.05, (a) is closed on 9B for good; otherwise there is
headroom of size pooled E − floor.

| setting | pooled E | threshold (pooled − 0.05) | floor | closed? | headroom |
|---|---|---|---|---|---|
| k = 0 | 0.771 | 0.721 | 0.357 [0.323, 0.389] | **no** | **0.414** [0.382, 0.448] |
| k = 2 | 0.881 | 0.831 | 0.357 [0.323, 0.389] | **no** | **0.523** [0.491, 0.558] |

Sensitivity, all **not closed**:

- plug-in floor 0.318 (headroom 0.453 / 0.563);
- combined with the stage-W pair, 0.347 (0.423 / 0.533);
- k = 2 offset-protocol floor 0.437 (headroom 0.443);
- upper CI end 0.389.

Even the raw replicate predictor (e_replicate 0.449) is far below both thresholds.

**Consequence:** the noise floor does not close (a) on 9B. The W2 tie is not explained by measurement noise.
Between 0.41 and 0.52 of E above the floor is between-run variation in the true state that the pooled curve does
not capture, and most of it is in truncation. This is the descriptive input R2 asked for. It reopens nothing by
itself:

- W2 stays a FAIL.
- Whether an edit-informed predictor can reach that headroom is the question for the beyond-dose (a) design in
  study M (PREREG_WMM).
- The no-9B-dose-confirmation recommendation is unchanged.

## 6. Caveats

- One replicate pair in W2 (two with stage W), all at the base state. Noise at trained members may differ, for
  example if truncation-prone states are more order-sensitive. The bootstrap CI covers item sampling within the
  pair, not variation across states or across pairs.
- The replicate design (reversed item order) is one perturbation of batch composition. Members were evaluated
  in their own queue positions, and their real perturbation could be larger or smaller.
- The floor uses a Gaussian approximation for E|error| (CLT over 500 items). The test in
  `tests/test_wme_w2_noise_floor.py` checks that independent Bernoulli replicates recover √(2/π).
- The SE floor 1/N in `se_of` is inactive at the replicate pairs: every base y is in [0.15, 0.72].

## 7. Commands

```
cd /home/ubuntu/projects/wsp-lab && scripts/wt.sh new analysis/w2-noise-floor
cd .worktrees/analysis/w2-noise-floor && . scripts/env.sh
python -m pytest -q tests/test_wme_w2_noise_floor.py
python scripts/wme_w2_noise_floor.py --w2 data/wme/stagew2/run-20261118 \
    --stagew data/wme/stagew/run-20261118/remote/w \
    --json docs/reviews/wme/wme-stagew2-noisefloor-2026-11-20.json
```

The run took 6.6 CPU-seconds. No GPU and no Lambda were used, and the cost was $0.
