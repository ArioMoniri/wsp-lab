# WM-E stage W2: exploratory model search on the exposed 9B data (20 Nov 2026)

> **POST HOC / EXPLORATORY.** This implements SYNTHESIS-r13 addendum item 3
> (`docs/council/2026-11-20/SYNTHESIS-r13.md`). It runs on the exposed W2 data (`data/wme/stagew2/run-20261118`) that
> the council has already seen, CPU only, $0. Nothing here is registered or confirmatory, and it does not change the W2
> verdict (FAIL). Its only purpose is to **freeze one specification** for later confirmation on fresh data. Read only:
> `data/evals`, sealed splits and `docs/PREREG_*` were not touched.

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
