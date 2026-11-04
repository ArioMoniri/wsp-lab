# Board A — D3 P4 suite review, 4 November 2026

**REVISE — blocking for hypothesis or generalisation claims.** This first complete
D3 pass provides exploratory predictor measurements, not a qualified H1/H2/H3
success. Review requested on the main checkout; no training implementation was
changed. All parent run IDs below are under `data/runs/p4/` and have predictor git
`f90f5918b8d59b633c5054808dbd1d3931548025`.

## 1. Hypothesis verdicts and admissible endpoints

The H5 Board A pre-registration admits MMLU-Pro and HellaSwag **provisionally for
large-effect screening**, not yet for confirmatory ranking. Its ranking gate is
repeat-panel Spearman ≥ .8, lower 95% bound ≥ .6, ICC lower bound ≥ .75, plus the
effect-noise and paired-CI criteria. The current cross-device points .75/.76 do
not pass that gate. Keyword refusal and KL remain conditional probes; saturated
refusal is not a qualified ranking target. GSM8K, MATH exact-match, IFEval and
pooled hit-1024 are descriptive. No excluded target can rescue a hypothesis.

**H1: undetermined under the qualification rule at n=260; the observed numerical
pass criterion is not met.** There are 300 evaluated entries, 260 usable matched
reference Δy rows, 40 excluded for missing reference targets, and all 260 retained
rows rely on inferred runtime metadata. No confirmatorily qualified endpoint is
available. Even treating the two provisional screening endpoints as eligible,
M2 does not reach .6 on either in any standard split and loses to StatNN on both
under seed/trajectory splits. This is failure to demonstrate H1, not its stronger
pre-registered falsification condition (ρ < .3 on all benchmarks after honest
tuning). Absolute-level runs at n=300 answer a different question.

| Parent run / split | MMLU-Pro M2 ρ [edit-bootstrap 95% CI] | HellaSwag M2 ρ [CI] | StatNN MMLU-Pro / HellaSwag ρ |
|---|---:|---:|---:|
| `d3-m2-seed` | .50 [.38,.60] | .55 [.45,.64] | .58 / .68 |
| `d3-m2-trajectory` | .54 [.42,.63] | .49 [.38,.60] | .61 / .67 |
| `d3-m2-family` | .02 [−.09,.13] | .58 [.50,.65] | −.35 / −.49 |

These are pooled out-of-fold correlations, n=260 each, not five independent
replications. M1, M1-matched, M5 and M6 also fail to establish H1 on the admissible
screening pair (see the complete D3 summary). Family M2 keyword refusal is −.35
[−.46,−.24], descriptive and conditional, not supporting evidence. KL was excluded
from the rate-head dataset and has no P4 prediction result. Beating negative family
baselines alone cannot satisfy H1. The suite also lacks a matched delta-norm result.

**H2: undetermined, effectively untested.** Every completed Δy K-fold run has
**one** evaluated row at trajectory length ≥4. `d3-m2-trajectory` reports that row's
MMLU-Pro/HellaSwag absolute errors .07006/.01005; M1-matched reports .12356/.14886
and M5 .05463/.46198. A single row cannot test ≥15% relative MAE reduction or
estimate its uncertainty; its degenerate bootstrap intervals are not precision.
The 18 length≥4 rows in level runs do not replace predecessor-relative evidence.
M2 consumes pooled features plus descriptors, while M1/M5 consume tensor tokens
and source information: M2 versus M1 alone is not a same-feature history ablation.
The valid H2 estimand is history beyond the latest cumulative-from-base snapshot.

**H3: undetermined.** No M3 run produced metrics: the three `d3-m3-*` directories
contain only their manifest snapshots, consistent with no usable training
parent–child pairs after purging. The 89 predecessor evaluations in progress are
not evidence in this review. M2's named LoRA holdout is not JEPA: its actual test
n is **59**, not the summary's dataset-wide 260, with ρ .40 [.16,.63] MMLU-Pro and
.12 [−.19,.40] HellaSwag. Other named holdout test n are 98 abliteration, 60 random
direction, 30 task vector and 13 chain. Chain-holdout .65/.83 cannot stand in for
LoRA transfer. M6 has no named LoRA acceptance run; its collapse gate is false in
all three splits, with ERank<2 in 1/5 family, 4/5 seed and 2/5 trajectory fold
seeds. None passes its foldwise ≥2× frozen-predictor requirement. No H3 success or
falsification follows from these non-equivalent/incomplete tests.

## 2. Negative family baselines: investigation and cause

Reviewer audit run **`board-a-d3-audit-20261104`** archives scripts, outputs and a
manifest with SHA256 of 178 parent configuration/manifest/metric/result/parquet
files. Reproduce from repo root after `. scripts/env.sh` with
`python data/runs/p4/board-a-d3-audit-20261104/audit.py` and `extra.py`.
No neural retraining or test-label-based fitting was performed. StatNN predictions
are not saved in parent fold parquet files: I refitted deterministic LightGBM on
exact saved train/test IDs and recovered the reported correlations to <1e−8.
I opened every completed run's fold parquet files, checked unique (edit,target)
keys and recomputed each reported neural Spearman.

**Determined mechanism: out-of-family target offsets, amplified by the nested
training composition, produce reversed between-family ordering in pooled OOF
predictions. Neural models additionally extrapolate under severe feature shift.**
This is not evidence of a universal negative predictor within families. It is
neither a discovered global Δy sign inversion nor a reason to multiply predictions
by −1. Specific evidence:

* Family test sizes are 98/60/59/30/13 for abliteration/random/LoRA/task-vector/chain.
  Actual fit sizes are only 75/75/75/128/141, with validation sizes 60/98/98/98/98.
  `train.py:123–128` takes only the first inner fold and never refits on the whole
  outer training set. For the LoRA test fold, the 75 fitting rows have **constant
  HellaSwag Δy=0**. StatNN consequently predicts zero for all 59 LoRAs despite
  their mean observed Δy=.033729. The opposite folds learn larger target offsets
  from LoRA-containing fitting sets and overpredict near-zero families.
* A feature-free per-fold training-mean predictor already produces pooled
  ρ=−.5397 MMLU-Pro and −.7363 HellaSwag on these exact family folds. It produces
  only −.0591/−.0457 on the trajectory folds. Negative family OOF correlation can
  arise from this composition alone, without features or normalization.
* Neural training-only normalization has **zero test-ID overlap** in the two
  audited family runs. It normalizes features, not target signs. StatNN never uses
  this normalizer, ruling it out as the shared cause. However, on the LoRA test
  fold 21.99% of scalar stats coordinates exceed |z|=20 before clamping; maximum
  |z| is 18,699. This is genuine unseen-support extrapolation, not leakage.
  M1-matched's LoRA HellaSwag mean prediction is −1.0093 against +.03373;
  task-vector mean +1.3579 against +.00533. Its linear head permits these values.
* Directly reconstructing entry-minus-matched-reference values for M1-matched
  family, M2 family and M2 trajectory reproduces all 2,080 target cells per run
  within 2.39e−8. Task-vector source parents are not predecessor references;
  successor links are. This rules out a numerical global sign mismatch in these
  arrays, not incorrect inferred metadata or a host×edit effect.

Per-family residuals below are **mean prediction minus observation**, in rate
units. All numbers come from the audit of `d3-m2-family` and
`d3-m1_matched-family`; StatNN is identical in those runs.

| Test family (n) | Observed MMLU-Pro / HellaSwag mean Δy | StatNN residual MMLU-Pro / HellaSwag | M1-matched residual MMLU-Pro / HellaSwag | M2 residual MMLU-Pro / HellaSwag |
|---|---:|---:|---:|---:|
| abliteration (98) | .00020 / .00194 | −.01898 / .00560 | −.06621 / .02416 | −.14767 / .32892 |
| random direction (60) | .00300 / .00000 | −.03320 / .01024 | .08075 / .09460 | .19248 / −.18405 |
| LoRA SFT (59) | −.06339 / .03373 | .07035 / −.03373 | .16220 / −1.04304 | .05349 / .37565 |
| task vector (30) | −.01033 / .00533 | .00262 / .00561 | 1.59950 / 1.35256 | .02139 / −.07890 |
| chain (13) | −.02231 / .01154 | −.01116 / .00670 | .06711 / .07587 | −.02516 / .02392 |

Subtracting each family's observed and predicted means separately gives this
**post-hoc diagnostic**, not an eligible corrected prediction score (test labels
enter centering):

| Predictor | MMLU-Pro pooled → centered ρ | HellaSwag pooled → centered ρ |
|---|---:|---:|
| StatNN | −.3517 → .1271 | −.4903 → .2589 |
| M1-matched | −.2166 → −.0518 | −.5440 → .1448 |
| M2 | .0244 → .0445 | .5823 → .1833 |

Within-family StatNN MMLU-Pro ρ is .049/−.234/−.276/.343/.859 in the table's
family order. For HellaSwag it is −.012/undefined/undefined/.299/.601: random
direction observations and LoRA predictions are constant. Thus some actual
within-family failures remain; offsets do not explain every error. M2's LoRA
HellaSwag ρ is **−.273** despite its pooled +.582. Pooled signs in either direction
are poor evidence of transfer.

**Trust decision:** the stored family scores are arithmetically reproducible and
trustworthy as a description of this exact, poorly calibrated training protocol.
No model's pooled family score is trustworthy as confirmatory within-family
ranking or general edit-family transfer evidence. Do not discard the negatives,
flip their signs, normalize using test families, or promote M2's positive result.
Report fold/family-specific metrics and a fold-training-mean baseline, then rerun
a revised, prospectively frozen protocol. Normalization-only interventions would
be needed to quantify how much neural degradation is attributable to scaling;
this review establishes its support problem, not a causal percentage.

## 3. Baselines, controls and uncertainty

* `baselines.py:43–86,248–267` implements the roadmap's **StatNN as LightGBM on
  pooled φ_stats**, not a neural network: 300 trees, learning rate .05, eight
  leaves, minimum child size five, row/column subsampling, L2=1, deterministic
  seed. P4 uses the same targets and exact neural fitting/test IDs, excluding
  inner-validation rows (`train.py:318–330`). This matches the registered
  weight-statistics baseline. It does not get neural normalization. Persist its
  per-edit predictions in future runs; metrics-only storage obstructs audit.
* The registered delta-norm linear baseline exists as `delta_norm_ridge`: a
  training-fitted StandardScaler plus RidgeCV on effective Frobenius norm and
  relative effective Frobenius norm. **It is absent from every P4 metrics file
  and from `train.py:318–330`.** A P3 score using different labels/folds cannot
  fill this gap. Blocking: evaluate it on the frozen P4 targets and exact fold
  IDs before any claim of beating both baselines. No evidence supports saying
  both baselines were evaluated here.
* Manifests confirm M1-matched **2,185,740** versus M5 **2,185,736** parameters,
  a difference of four, across splits. `models/__init__.py:88–112` supplies the
  registered two stateless tensor layers versus one tensor plus one causal edit
  layer, both dropout .2. This matches capacity, not architecture or features
  against M2 (2,310,632 parameters, pooled input).
* `baselines.py:97–126` resamples paired (observed,predicted) **edits**, 1,000
  times at seed 0, with percentile 95% intervals and tied-rank Spearman. It does
  not refit models, resample outer folds, capture training-seed variance,
  preserve chain clusters/shared reference covariance, resample benchmark items,
  or account for endpoint/run selection. Fold seeds 0–4 are not five complete
  experiment seeds. Overlapping marginal CIs are not a paired model comparison.
  Use trajectory/ancestry-cluster resampling and paired differences, repeated
  outer seeds/splits, and separate item/repeat uncertainty.
* Configuration hashes recompute correctly in all 22 completed runs; M2 reports
  `mamba_ssm` in all five standard folds. Frozen manifest ancestry, split IDs,
  normalization contexts and feature/base revisions are recorded. SVD inputs
  are promoted to fp32 in `features/svd.py:30–35,49–53`; effective-rank SVD also
  uses fp32. These checks do not certify historical inferred runtime identity.

## 4. Ceiling analysis: what can and cannot be estimated

There is **no identified numeric attainable Spearman ceiling per Δy target** from
these artifacts. docs/RESULTS.md reports a small **level** panel (11 entries,
including base; only four A10 pairs), without the required centered-Δy ICC and
ranking CIs. Low panel variance/ties and host×edit effects prevent equating this
panel's correlation with predictor reliability. In particular, the prose there
saying likelihood/refusal/KL “transfer” is stronger than the registered qualification
supports. Identical-seed within-device repeats do not estimate seed variance or
uncertainty on new benchmark items.

For orientation only, the classical independent additive-noise Pearson model has
reliability r=Var(signal)/(Var(signal)+Var(noise)) and an oracle correlation ceiling
sqrt(r). It is not an exact Spearman law, and cross-device level Spearman is not
r. The following square roots are sensitivity heuristics computed from the actual
panel points, **not estimated ceilings or target qualifications**:

| Endpoint (fast item count) | Local–H100 panel level ρ | sqrt(panel ρ), heuristic only | Consequence |
|---|---:|---:|---|
| MMLU-Pro (100) | .75 | .87 | M2 .50–.54 and StatNN .58–.61 leave plausible headroom; none certifies a ceiling |
| HellaSwag (100) | .76 | .87 | M2 .49–.55 trails StatNN .67–.68; qualification still fails |
| GSM8K (50) | .30 | .55 | descriptive only; observed ranks may reflect device/family effects |
| MATH exact-match (25) | .09 | .30 | descriptive only; cannot conclude model adequacy from weak correlation |
| IFEval (25) | .29 | .54 | descriptive only |
| keyword over-refusal (50) | .39 | .62 | conditional; not a dependable ranking endpoint |
| pooled hit-1024 (100 mixed items) | .72 | .85 | descriptive truncation proxy, not MATH-only semantic looping |
| KL (50 prompts) | 1.00 | 1.00 | narrow panel agreement; no P4 head or demonstrated general ceiling |
| keyword refusal (50) | not supplied as a variable rank panel | unidentified | saturation makes rank qualification undefined/weak |

At n=100/50/25, maximal single-rate binomial SE is .05/.0707/.10.
For Δy use paired item discordance and covariance, not an independent-binomial
formula as a measurement. Fixed subsets can be predictable even with poor
population validity; neither these SEs nor device agreement is a mathematical
upper bound on fixed-subset Spearman. More edited entries reduce predictor
estimation variance but do not reduce per-edit item uncertainty. Estimate a
useful ceiling only after collecting variable same-device matched-reference
repeat panels, seed-1/2 repeats, per-item paired differences and Δy reliability
intervals. Shared base uncertainty and host interactions must remain explicit.

## 5. Ranked recommendations and permissible paper wording

1. **Repair and qualify measured predecessor-relative labels before scaling
   models.** Finish the 89 predecessor evaluations with recorded actual runtime,
   device, batch, item/order hashes and reference IDs; rebuild frozen Δy tables
   and count exclusions. Expand benchmark items/target variation and run independent
   seed-1 (then seed-2) panels. Qualify MMLU-Pro/HellaSwag under the exact existing
   rule; conditional refusal/KL need their own qualification. Never reinterpret
   metadata backfill as measured provenance. This is the single highest-priority fix.
2. Freeze revised split/training plans before reruns: preserve full ancestry/source
   grouping and purging, report family-specific n and metrics, inspect target
   variance in fitting/validation folds, and replace reliance on the first inner
   fold with nested selection plus a declared outer-train refit protocol. Fit all
   preprocessing only on permitted training contexts. Add the fold-mean and
   delta-norm baselines and retain all per-edit baseline predictions. Bootstrap
   paired differences by independent ancestry clusters; run multiple full seeds.
3. Audit features before architecture expansion: incremental versus cumulative
   delta features, source information, descriptors/family-only and device-only
   controls, train-only robust scaling/constant-coordinate handling, and out-of-
   support diagnostics. Do not inject device into a purported weights-only result.
   Use identical pooled features for M2 and a stateless capacity-matched control;
   use M5 versus M1-matched for the existing token-feature H2 comparison.
4. Then rerun M2/M5 on enough complete length≥4 trajectories for a powered paired
   H2 MAE comparison, and M3/named LoRA JEPA transfer after ancestry-pair coverage
   is restored. Diagnose M6 collapse before more seeds or any acceptance claim.
   Bigger models cannot resolve missing predecessors or unqualified targets.

Paper templates that are supportable now:

> On the D3 fast-tier subset, 260 of 300 evaluated edits had matched-reference
> delta targets; all retained labels used inferred runtime metadata. In exploratory
> trajectory-grouped evaluation, M2 achieved Spearman .54 [.42,.63] on MMLU-Pro
> and .49 [.38,.60] on HellaSwag, compared with StatNN .61 and .67. Intervals
> resample edits conditional on the fitted models and do not correct chain dependence.

> The pre-registered H1 criterion was not demonstrated. Ranking qualification
> of the fast targets remains incomplete, and the P4 delta-norm comparator is
> pending. Pooled family-holdout correlations were sensitive to between-family
> calibration offsets and do not establish within-family transfer.

> H2 and H3 remain undetermined: the delta-target suite contains only one labeled
> length-at-least-four trajectory position, and the trajectory-JEPA runs produced
> no completed result after ancestry purging.

The trajectory M2 numbers are **reportable only as these qualified exploratory
measurements**, with both baselines' status, n, tier, inferred provenance and CI
limitations. They are not evidence that sequence modeling beats StatNN or that H1
passes. Do not claim general capability, safety, loop repair or hardware-free
prediction from this suite. No gate is marked passed by this review.

## Verification and disposition

Inspected AGENTS.md, reviewer-ml role, REVIEW_PROTOCOL, HYPOTHESIS, the P4 candidate
protocol, H5 Board A qualification, RESULTS cross-device panel, D3 summary,
training/data/splits/baselines and all model implementations. Inspected all 22
completed runs' config/manifest/metrics/results and fold parquet predictions;
three M3 incomplete directories were checked separately. Audit output and frozen
parent-file hashes are under `data/runs/p4/board-a-d3-audit-20261104/` (local data,
not committed). Focused predictor tests: **36 passed**, with dependency deprecation
warnings; full CPU suite: **482 passed, 1 skipped, 8 deselected**, 51.74 s. No GPU
benchmark/training rerun, new cloud spend, push, or external board post was made.

**REVISE:** blocking items are label/ranking qualification, missing delta-norm
comparison, unsupported pooled family-transfer interpretation, and insufficient
H2/H3 data/design. Keep exploratory negative and positive results together.

H1 — UNDETERMINED under qualification; observed M2 .49–.55 trails StatNN .58–.68 and does not meet .6 on the screening pair.
H2 / H3 — H2 UNDETERMINED (one length≥4 labeled row); H3 UNDETERMINED (no completed M3 or qualifying JEPA LoRA transfer).
Negative family baselines — reversed family/fold offsets from shifted, sometimes constant fitting labels; neural support/scaling shift adds instability, not a global sign inversion.
Most important fix — collect and qualify measured matched-runtime predecessor Δy labels with expanded items and independent seed repeats.
Trajectory M2 — reportable as exploratory fixed-tier results only, not H1/H2 success or baseline superiority.
Tests/artefacts — 36 focused and 482 full-suite tests passed; 22 completed run bundles and all fold parquets, three incomplete M3 directories, deterministic baseline reconstruction and archived reviewer audit inspected.
