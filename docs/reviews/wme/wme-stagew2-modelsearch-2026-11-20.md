# WM-E stage W2: exploratory model search on the exposed 9B data (20 Nov 2026)

> **POST HOC / EXPLORATORY.** This implements SYNTHESIS-r13 addendum item 3
> (`docs/council/2026-11-20/SYNTHESIS-r13.md`). It runs on the exposed W2 data (`data/wme/stagew2/run-20261118`) that
> the council has already seen, CPU only, $0. Nothing here is registered or confirmatory, and it does not change the W2
> verdict (FAIL). Its only purpose is to **freeze one specification** for later confirmation on fresh data. Read only:
> `data/evals`, sealed splits and `docs/PREREG_*` were not touched.

## BLUF (POST HOC / EXPLORATORY)

- **Verdict: none beats pooled + dose.** None of the 5 pre-declared candidates beats C2 at either k: Holm over 10
  hypotheses, family α 0.025, and the best p of any candidate is 0.149. Nested inner CV shrank the
  per-layer J1 ridge, the GP, the GRU and the SSM back to C2 in most folds, and the GRU in **all 30** folds.
- **Frozen spec for fresh-data confirmation: C2 (pooled + per-(type, point, endpoint) log ‖ΔW‖ dose) vs C1 (pooled).**
  This is round-13 A1. It reproduces exactly on W2: E 0.678 vs 0.771 (k = 0, p 0.021) and 0.766 vs 0.881 (k = 2,
  p 0.003). In sample, it passes the W2 rule at both k (Holm-level LB 0.003 / 0.014). The nulls N1 (dose permuted),
  N2 and N3 all have size **0 / 1000** (CP upper 0.003).
- **Power to confirm it (α 0.0125, the W2 rule):** at the observed effect, **20 fresh runs give 0.96** (k = 2; 15
  runs give 0.83). At half the observed effect, even 60 runs give only 0.72. Plan **≥ 30 runs** (0.9985 at the
  observed effect, 0.26 at half). This confirms a recipe-dose effect only (r(log ‖ΔW‖, log lr) = 0.996), not
  edit-specific prediction.
- **The only non-zero candidate gain** is C4 at k = 2 (E 0.747, d +0.020, p 0.149). It comes from a partial or zero
  k-offset (ω < 1) on M paths (d +0.080), not from any feature. This is a hypothesis about the W2 offset protocol,
  not about weight-space information.
- **Headroom remains:** C2 0.678 vs the floor 0.357. Per-layer J1, nonlinear kernels and sequence memory do not
  capture any of it on 45 paths. The R4 conditions for an SSM claim are unchanged and unmet: C5b ≈ C2.

| id | candidate | E k = 0 | E k = 2 | d vs C2, k = 0 (p) | d vs C2, k = 2 (p) | Holm α (smallest) | beats C2 | inner-CV choice |
|---|---|---|---|---|---|---|---|---|
| C1 | pooled (reference) | 0.771 | 0.881 | −0.093 (0.979) | −0.114 (0.997) | descriptive | — | — |
| C2 | **pooled + dose** (baseline) | **0.678** | **0.766** | 0 | 0 | — | — | — |
| C3a | C2 + ridge on per-layer J1 (24) | 0.682 | 0.766 | −0.004 (1.000) | 0.000 (1.000) | 0.0025 | no | α = ∞ (= C2) in 14/15 and 15/15 folds |
| C3b | C2 + GP on [J1G, D] | 0.683 | 0.766 | −0.005 (1.000) | 0.000 (1.000) | 0.0025 | no | ρ = 0 (= C2) in 9/15 and 15/15 folds |
| C4 | per-endpoint heads (base, ω) | 0.685 | **0.747** | −0.007 (0.971) | **+0.020 (0.149)** | 0.0025 | no | k = 2: mostly ω 0 / 0.5 on accuracy and truncation-MMLU |
| C5a | C2 + GRU (248 / 588 params) | 0.678 | 0.766 | 0.000 (1.000) | 0.000 (1.000) | 0.0025 | no | off (= C2) in 15/15 and 15/15 folds |
| C5b | C2 + selective diagonal SSM (240 / 424) | 0.680 | 0.766 | −0.002 (1.000) | 0.000 (1.000) | 0.0025 | no | off in 14/15 and 15/15 folds |

The Holm α shown is the smallest level, 0.025/10. No p reached it, so Holm stopped at the first hypothesis.

## 0. Pre-declaration (committed before any candidate was fitted on the W2 data)

Section 0 was written and committed before `scripts/wme_w2_model_search.py` was run on the W2 data. The only results
seen beforehand were the published ones: W2 itself, the round-13 methods specification tests (pooled + dose 0.678 /
0.766) and the noise floor (0.357). Those are the reason this search is post hoc.

### 0.1 Data, protocol, metric

- **Data.** 45 complete S/L/M paths (15 runs x 3 types) x 6 points x 4 endpoints, loaded with `wsp.wm.stagew2.load_runs`
  and `arrays`.
- **Metric.** The W2 path score E (`wsp.wm.stagew2.score`): the mean |pred − y| / SE(y) over the points after k and the
  4 endpoints (accuracy and truncation, MMLU-Pro and SuperGPQA). Settings k = 0 and k = 2.
- **Outer protocol.** Leave-one-run-out (LORO), 15 folds. A held-out run's 3 paths are predicted from the other 14
  runs only.
- **k-point offset.** At k = 2, every candidate except C4 and C5 adds its mean residual on the 2 given points of the
  held-out path, exactly as in W2 (ω = 1). C4 tunes ω, and C5 reads the given points as inputs.
- **Tuning.** Every hyperparameter is chosen by **nested** LORO inside the 14 training runs (13 inner-training runs,
  1 inner-test run), separately for each k. The criterion is the inner mean path E. On ties within 0.5 %, the
  earlier grid entry wins; grids are ordered from most regularised to least, and the first entry is always the
  pooled + dose baseline itself, so every tuned candidate nests the baseline.
- **Cross-fitted residuals.** Residual learners (C3, C4's ridge entries, C5) are trained on the training paths'
  residuals against a pooled + dose baseline that was itself cross-fitted by LORO inside the same training runs, so
  the training residuals have out-of-sample scale.

### 0.2 Features (all from exposed W2 artefacts)

- **Dose** D = log ‖ΔW‖ of the member, = log √(Σ over the 4 R0 group norms²) (`W.member_features` columns 0–3). This is
  exactly the round-13 methods dose variable.
- **Per-layer J1** (J1G, 24 features): from `features.npz` `j1` [3 objectives, 64 items, checkpoint, 32 layers], the
  member's value is λ · mean over items, summed within the 8 fine layer groups (4 layers each), and zeroed for groups
  beyond the member's mask. This is the layer-resolved version of W2's `j1_*` features, which sum over layers.
- **R1 summaries** (3): W2's `j1_stop`, `j1_target`, `j1_margin`.
- Features are standardised within (type, point) with training-run statistics. A constant column becomes 0.

### 0.3 Candidates and grids

| id | candidate | definition | grid (inner-CV order) | free parameters per outer fold |
|---|---|---|---|---|
| C1 | pooled (reference) | per (type, point, endpoint) mean of the training paths (= `W2.fit` curve) | — | 72 |
| C2 | **pooled + dose** (mandatory baseline) | per (type, point, endpoint) OLS of y on [1, D] (the round-13 "per (type, point, endpoint)" row, 0.678 / 0.766) | — | 144 |
| C3a | C2 + ridge on per-layer J1 | per (type, endpoint) ridge, shared across points, of the cross-fitted C2 residual on J1G (24) | α ∈ {∞ (= C2), 1000, 100, 10, 1} | 144 + 288 |
| C3b | C2 + GP on per-layer J1 and dose | per (type, endpoint) GP posterior mean of the C2 residual, RBF kernel on standardised [J1G, D] (25 inputs), squared distance / 25 | (ℓ, ρ = signal/noise variance): (·, 0) = C2, then ℓ ∈ {2, 1, 0.5} x ρ ∈ {0.1, 0.3, 1, 3} | 144 + GP |
| C4 | per-endpoint heads | each endpoint independently picks its base model and k-offset weight by inner CV (separate accuracy and truncation heads) | base ∈ {C2, pooled, logit-dose (C2 on the logit scale), C2 + ridge α 1000, 100, 10}; ω ∈ {1, 0.5, 0} at k = 2 (ω = 1 only at k = 0); first entry (C2, ω 1) | ≤ 144 + 288 |
| C5a | C2 + tiny GRU | GRU over the 6 ordered points, inputs per point (13): type one-hot (3), point x/8, D, R1 summaries (3), the C2 residual in SE units at given points (4; 0 after k), a given-point flag. A zero-initialised linear readout gives a correction in SE units on top of C2 (with its ω = 1 offset at k = 2) | off (= C2), then (hidden h, L2 λ) ∈ {4, 8} x {0.1, 0.01, 0.001} | h 4: 248; h 8: 588 |
| C5b | C2 + selective diagonal SSM (Mamba-style) | same inputs and readout; h_t = a_t ⊙ h_{t−1} + Δ_t ⊙ (B u_t), a_t = exp(−Δ_t e^{A}), Δ_t = softplus(W_Δ u_t + b); y_t = C (h_t ⊙ silu(Z u_t + z)) + D_u u_t (amended, see below) | off (= C2), then {4, 8} x {0.1, 0.01, 0.001} | h 4: 240; h 8: 424 |

C5 training: full batch, Adam (learning rate 0.02), 300 steps, Huber loss (δ 1) in SE units on the points after k, an
L2 penalty λ Σθ², and an ensemble mean over 3 seeds. k = 0 and k = 2 are trained separately. Parameter counts are
matched in order of magnitude to C3a's 288 ridge coefficients. The ~45-path sample is the reason for the small sizes.

**Amendment A1 (20 Nov 2026, before any fit on W2 data).** As first declared, C5b had no output gate (y_t = C h_t +
D_u u_t). With that readout its state can only decay after the given points, so it failed the synthetic unit test
of extrapolating a path effect that grows along the path, which the GRU passes. The amendment adds the Mamba output
gate silu(Z u_t + z), raising the parameter counts from 184 / 312 to 240 / 424. Nothing else changed. The amendment
is committed together with the code, before the W2 run.

### 0.4 Tests and the selection rule

- **Contrast.** For each candidate c ≠ C2 and each k, d_p = E_C2,p − E_c,p (positive favours c).
- **Test.** One-sided run-cluster sign-flip (`W2.cluster_test`: exact over the 15 runs, run weight = path count) and
  its test-inversion lower bound.
- **Multiplicity.** **Holm over 10 hypotheses** (C3a, C3b, C4, C5a, C5b x k ∈ {0, 2}) at family α 0.025. C1 vs C2 is
  reported descriptively.
- **"Beats pooled + dose"** for a candidate means Holm rejects and the Holm-level lower bound is > 0 in at least one k
  (the W2 pass rule).
- **Frozen spec.** Among the candidates that beat C2, the one with the largest mean d averaged over k = 0 and k = 2.
  If none does, the verdict is **"none beats pooled + dose"**, and the specification frozen for confirmation is
  **C2 vs C1** (round-13 A1).
- **Nulls for the frozen spec**, the W2 N1–N3 adapted to the contrast actually frozen, with the pass rule's size
  required ≤ 0.035:
  - N1: the candidate's non-dose features are shuffled (whole trajectories, deranged within type). For C2 vs C1 this
    is the dose trajectory.
  - N2: outcome curves are deranged among paths of the same type.
  - N3: the W2 no-edit-information world (`W2.calibrate_n3` / `n3_world`), features kept.
  - Replicates: 1000 for C1–C4. For C5 the count is set by run time and stated.
- **Fresh-data power** for the frozen contrast: `W2.ht3_power` (run-cluster bootstrap of the per-path d, a run's
  paths as one unit, sign-flip at α 0.0125 = Holm over 2 k) at R ∈ {15, 20, 30, 45, 60} runs, with the effect at 1.0x
  and 0.5x the observed mean. The 0.5x row allows for winner's curse.
- **Reported for every candidate:** E at both k, d vs C2, p, Holm decision, the lower bound, E per endpoint, d per type
  and the chosen hyperparameters.

## 1. Results (POST HOC / EXPLORATORY)

The full run took 2,367 CPU-seconds (39 min, 4 workers, CPU only, $0). Every number below is from
`docs/reviews/wme/wme-stagew2-modelsearch-2026-11-20.json`.

### 1.1 Reproduction

C1 gives 0.7708 / 0.8805 and C2 gives 0.6781 / 0.7662. These match W2 and the round-13 methods row "per (type,
point, endpoint)" to 4 decimals, with the same sign-flip p (0.0207 / 0.0029).

### 1.2 E per endpoint (acc MMLU-Pro, acc SuperGPQA, trunc MMLU-Pro, trunc SuperGPQA)

| candidate | k = 0 | k = 2 |
|---|---|---|
| floor (noise-floor addendum) | 0.291, 0.337, 0.427, 0.373 | same |
| C1 pooled | 0.514, 0.564, 0.938, 1.066 | 0.639, 0.677, 1.157, 1.049 |
| C2 pooled + dose | 0.480, 0.536, 0.797, 0.900 | 0.550, 0.651, 0.956, 0.907 |
| C3a ridge | 0.481, 0.535, 0.802, 0.909 | = C2 |
| C3b GP | 0.482, 0.534, 0.806, 0.909 | = C2 |
| C4 per-endpoint | 0.484, 0.540, 0.808, 0.906 | 0.530, 0.640, 0.872, 0.943 |
| C5a GRU | = C2 | = C2 |
| C5b SSM | 0.482, 0.537, 0.798, 0.904 | = C2 |

Dose removes about 0.14–0.17 of truncation E at k = 0. Beyond dose, nothing moves truncation SuperGPQA (0.90 vs a
floor of 0.37), which is the largest remaining gap.

### 1.3 d vs C2 by path type (S / L / M; positive favours the candidate)

| candidate | k = 0 | k = 2 |
|---|---|---|
| C1 pooled | −0.199 / −0.080 / +0.001 | −0.308 / −0.055 / +0.020 |
| C3a ridge | −0.002 / −0.008 / −0.002 | 0 / 0 / 0 |
| C3b GP | +0.003 / −0.017 / −0.001 | 0 / 0 / 0 |
| C4 per-endpoint | +0.002 / −0.016 / −0.006 | +0.001 / −0.023 / **+0.080** |
| C5a GRU | 0 / 0 / 0 | 0 / 0 / 0 |
| C5b SSM | −0.001 / −0.003 / −0.003 | 0 / 0 / 0 |

Dose helps S and L paths and does nothing on M paths. The mechanism is not identified here.

### 1.4 Frozen contrast: C2 vs C1 (round-13 A1)

| k | E C2 | E C1 | d | p | Holm α | Holm rejects | LB at Holm α |
|---|---|---|---|---|---|---|---|
| 0 | 0.678 | 0.771 | 0.093 | 0.0207 | 0.025 | yes | 0.003 |
| 2 | 0.766 | 0.881 | 0.114 | 0.0029 | 0.0125 | yes | 0.014 |

The Holm-level LBs were recomputed from the JSON's per-path d with `W2.holm` / `W2.cluster_test`; the script stores
the LBs at α 0.025 (0.003 / 0.024).

**Nulls** (size = share of replicates passing the whole W2 rule; ≤ 0.035 required):

| null | replicates | size | CP upper 95 % | mean d k = 0 / k = 2 |
|---|---|---|---|---|
| N1 dose trajectory deranged within type | 1000 | 0.000 | 0.003 | −0.028 / −0.030 |
| N2 outcome curves deranged within type | 1000 | 0.000 | 0.003 | −0.027 / −0.030 |
| N3 no-edit-information world | 1000 | 0.000 | 0.003 | −0.032 / −0.036 |

**Fresh-data power** (`W2.ht3_power`, run-cluster bootstrap, sign-flip at α 0.0125, 2,000 replicates):

| runs R | k = 0, observed effect | k = 0, half effect | k = 2, observed effect | k = 2, half effect |
|---|---|---|---|---|
| 15 | 0.44 | 0.07 | 0.83 | 0.07 |
| 20 | 0.61 | 0.12 | **0.96** | 0.13 |
| 30 | 0.85 | 0.20 | 0.9985 | 0.26 |
| 45 | 0.98 | 0.36 | 1.00 | 0.52 |
| 60 | 0.9985 | 0.50 | 1.00 | 0.72 |

At the observed effect, the W2 rule (pass in either k) needs **about 20 fresh runs** for 0.8 power. If the true
effect is half the post-hoc estimate, **more than 60 runs** are needed. The methods seat priced 30 runs at about 17
GPU-h and $84; that is the user's decision under Q4. Study M remains the chair's preferred route.

For reference, the same bootstrap for the best non-significant point estimate (C4 vs C2) gives at most 0.46 power
at 60 runs (k = 2, observed effect) and 0 at k = 0, where d < 0.

## 2. Interpretation

- **The model search confirms the round-13 diagnosis.** The W2 bank identifies one within-type axis, dose. Given
  dose, per-layer J1 (24 features) carries no out-of-fold predictive information under ridge or a GP. A GRU or a
  Mamba-style SSM over the trajectory and the R1 sequence adds nothing either. Parameter count is not the binding
  constraint: inner CV turns these models off.
- **The headroom above the floor is real but not reachable with these features.** C2 sits at 0.678 against a floor
  of 0.357. Accuracy is close to its floor-plus-signal ceiling; truncation is not.
- **The C4 hint.** At k = 2, ignoring or halving the W2 k-point offset helps M paths (d +0.080). That fits the
  round-13 note that two early points inject noise while the signal sits late. It is not significant (p 0.149) and
  concerns the protocol, not (a).
- **SSM (R4).** No condition for an SSM claim is met: paths have 6 points, no beyond-dose variance was detected, and
  the matched-capacity ridge/GP comparators also tie C2.

## 3. Caveats

- POST HOC. The council had already seen this data, and the C2 specification was chosen among six after seeing it.
  The in-sample C2 pass is therefore not evidence; only fresh data can confirm it.
- The C5 ensembles use 3 seeds and 300 Adam steps. A different optimiser budget could change C5. Inner CV chose
  "off" almost everywhere, so a gain would have to be large to change the verdict.
- C3/C4 residual learners share coefficients across points within (type, endpoint). Per-point ridge was not tried.
- With 15 runs, the run-cluster sign-flip has a minimum attainable p of 1/32,768; power at small effects is
  limited by the design, not the test.
- N1–N3 are adapted to the frozen contrast (§0.4). N1 permutes dose, not W2's full feature vector.

## 4. Commands

```
cd /home/ubuntu/projects/wsp-lab && scripts/wt.sh new analysis/w2-model-search
cd .worktrees/analysis/w2-model-search && . scripts/env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests/test_wme_w2_model_search.py          # 19 passed
agent-tmux w2-model-search python scripts/wme_w2_model_search.py --w2 data/wme/stagew2/run-20261118 \
    --workers 4 --json docs/reviews/wme/wme-stagew2-modelsearch-2026-11-20.json
```

Read only: `data/wme/stagew2/run-20261118/{a,b,c}/w2`. No GPU, no Lambda, $0. `data/evals`, sealed splits and
`docs/PREREG_*` were not touched.
