# F10.5 sensitivity and interpretation specification (Board A F10 condition 6)

**Status: design specification, 11 Nov 2026 (server date). Not a registration text.** It freezes every report that
F10.5 of `docs/PREREG_H1_TF.md` requires, and the interpretation rules Board A attached to them
(`docs/reviews/council/board-a-f10-2026-11-11.md`, condition 6). It takes effect only when a dated A5 note adopts it by
path and sha256 **before any Part A entry is scored** (Board A F10 conditions 6 and 7; F10.10). `docs/PREREG_H1_TF.md`,
`docs/prereg/` and `data/` were not edited. No code was changed. Nothing under `data/evals/` was opened. No G3, rotation,
ladder, missing-reference or sealed outcome, or a summary of one, was seen in writing this spec; none exists
(`docs/prereg/a5_manifests/part_a/manifest.json:scored` = `false`). Every number below is read from, or recomputed from, a
repository file named next to it. Recomputations are of outcome-free staging metadata only.

**Exposure line.** This session (design sub-agent, worktree `design/f10-sensitivity`, 11 Nov 2026, until the commit of this
file) read only `docs/`, `src/` and `scripts/` files and the committed manifests under `docs/prereg/a5_manifests/`. It did not
open, compute, score or receive any G3, rotation, ladder, reference or sealed outcome or a summary of one. CPU only; no GPU;
no write outside this file.

## 0. Sources and scope

| Source | Path | sha256 |
|---|---|---|
| Registration (F10 block, A5.3–A5.5, F2, F5.5, F5.7, F7, F9a.2) | `docs/PREREG_H1_TF.md` | `be3dcab22b745a04c20b3cd2c433222f1372dc223cd6f35f7dfcd0b44c58111a` |
| Board A F10 ruling (condition 6; "Hypothesis, estimand and bias") | `docs/reviews/council/board-a-f10-2026-11-11.md` | `f1c494ab608c464d29f9dc261dee8711e553ab0fee90a136581b0efb432142b3` |
| Board A F7 ruling (conditions 2–4) | `docs/reviews/council/board-a-f7-2026-11-11.md` | `817dd45467b3e23167663ea20b40780cc79d9510377698c65aadca2cf13c5b19` |
| Rotation-yield design note (§1 diagnostics, §2.1, §2.3) | `docs/design/ROTATION_YIELD_2026-11-11.md` | `fad409ccb1c3059fc4b8385252c1e5f883148571a7521f590e350948f0e5719b` |
| Part A staging manifest | `docs/prereg/a5_manifests/part_a/manifest.json` | `d7785b35de1d7041edd2db139591562379db12e33c2532921c814735e8dde8d6` |
| Part A staging audits | `docs/prereg/a5_manifests/part_a/rotation_audits.jsonl` | `f05fec4265ffd93c8c537100993d23262be95141369240fcad0a7e16ff474d35` |
| Part A staging README | `docs/prereg/a5_manifests/part_a/README.md` | `2fcbf517ccf26f2a1fe325ef31300d291bc0f254c7faaacd18154e327bceada1` |
| NULL-SIZE report | `docs/prereg/a5_manifests/nullsize/report.json` | `1bed2c9531e4230864a1ebbe7e4023e0ffafcf704a59c40e93f9a75bd469dfa6` |
| Power field manifest (sign distribution) | `docs/prereg/a5_manifests/power/fields_manifest.json` | `19e9de8430ff2a135db848f7d0d5097b5b5724138e9ee815e807e5753e5f9a6b` |
| Power report | `docs/prereg/a5_manifests/power/report.json` | `ff46de30fa8e4ff97a4034e8bdef78cb19bc49662073523500a43c773b55845a` |
| Registered selection | `docs/prereg/a5_manifests/selection.json` | `393530365f87f981d5da3ecb2779c89b8e7bf98105b6c45d1824bf290e908c63` |
| Staging diagnostics code (`_moments`, `_edit_audit`) | `src/wsp/eval/response_intervention.py` | `b511d3d1ccb2c518317b59eafa4eea034df41ee90b85ab53709ef008736e353a` |
| Power prediction code (`p_rot = <g, D_eff − R_eff>`) | `scripts/a5_power_fields.py` | `bcf7d15d679e819d17f7784262bb22fbe10fece33bbb6757e8e08b44d3e592a8` |

**Scope.** Every report here concerns the **rotation contrast** d_i = y(W_i + D_i) − y(W_i + R_i) of A5.3, with y the
registered WikiText endpoint `extra.mean_nll` (nats per valid next token; positive Δy is a worse fit, F9a.2(b)) and the
registered prediction p_i = ⟨g, D_eff − R_eff⟩ (A5.3; F10.2). F10.5 adds sensitivities to the rotation exclusion rule, and ρ
is defined only for rotation pairs. The two ladder contrasts are analysed exactly as A5.3, A5.5 and F10.9 register them; no
F10.5 report applies to them, and their `realised-increment ladder` flags (F10.9) are carried in the limitations (§5).

**Population.** The *analysis set* is the set of rotation pairs that pass F10.3 at the re-stage, have finite scores for both
arms and are **resolved** under A5.3 (paired block bootstrap over 32 two-window blocks, 4,000 draws, percentile interval
excludes zero). Resolution status is taken from the single main analysis and is never recomputed for a sensitivity report.
Under fallback (e) (F10, header; Board A "Order and fallback"), the same reports are produced on whatever single pair passes
the original audit; every one is then below the ≥ 15 floor, and the covariate model of §1 is `not estimable`.

**Agreement.** For a resolved pair, a_i = 1 if sign(p_i) = sign(d̂_i) with both signs nonzero, else a_i = 0; a zero
prediction is a disagreement (A5.3). d̂_i is the point paired mean over the 64 registered windows.

---

## 1. Covariate analysis (F10.5(a))

### 1.1 Covariate

For each rotation pair i, the **log energy ratio**

  x_ρ,i = ln ρ_i, ρ_i = Σ_t ‖R_eff,t‖² / Σ_t ‖D_eff,t‖²,

summed over all edited tensors t of edit i (F10.4's pooled ρ; D_eff = D of the note's notation). Natural logarithm,
dimensionless. **Source for the model: the re-stage report's F10.4 pooled ρ** of each F10.3-passing pair, never the staged
manifest. F10.3(v) guarantees ρ_i ≥ 0.25 for every passing pair, so x_ρ,i ≥ −1.386 is finite.

Staged values, for reference and for the frozen tertile cuts of §2.2 only: `_edit_audit` stores
`pooled_norm_mismatch` = |‖R_eff‖ − ‖D‖|/‖D‖ with pooled norms (`src/wsp/eval/response_intervention.py:_edit_audit`), and
all forty signed pooled values are positive (note §1.4 table, column "pooled realised norm (signed)": +0.0007 to +0.0309).
Hence staged ln ρ_i = 2·ln(1 + `manifest.json:edit_audits[i].edit_audit.pooled_norm_mismatch`). Recomputed from the manifest:
range 0.0013847 (`15a53586efc72a50`) to 0.0609060 (`61d1f61e25212068`). The re-stage report must give
max_i |x_ρ,i(re-stage) − x_ρ,i(staged)|; F10.2 leaves the construction unchanged, so a nonzero difference is disclosed, not
corrected.

Outcome-free collinearity disclosure (staged values, 40 pairs): Pearson correlation of staged ln ρ with
ln(1 + effective norm) is −0.735, Spearman −0.832. ε_R is set by κ, hence by effective norm and the predecessor grid
(note §1.2). The ln ρ coefficient is therefore not separable from the norm term on these data, and the report must say so.

### 1.2 Models (the registered A5.3.3 sensitivity, then with x_ρ added)

Outcome: a_i ∈ {0, 1} of §0. Rows: the analysis set, restricted to the complete-case set of §1.4. Two models, fitted on the
same rows:

* **M0, registered** (A5.3.3: "ridge logistic agreement on family, log1p effective norm and rotated-minus-original v4
  differences"). Columns: (1) `chain` indicator (1 = chain, 0 = lora_sft; `manifest.json:selections[].family`);
  (2) ln(1 + effective norm), effective norm from `manifest.json:selections[].effective_norm` (A5.3.1 clarification: root
  `fro_effective`, chain predecessor-relative norm); (3)–(7) the five rotated-minus-original logged-moment differences
  Δmean, Δstd, Δfrac_pos, Δfrac_nonzero, Δcos_to_predecessor.
* **M1 = M0 + x_ρ** (F10.5(a)). Column (8) x_ρ of §1.1.

*Moment differences (operational reading, flagged for the adopting note).* A5.3 and A5.3.3 name "logged v4 differences" /
"rotated-minus-original v4 differences" without a column list. The only rotated-minus-original quantities the pinned staging
code logs are the five fields of `_moments` (`mean`, `std` (unbiased), `frac_pos`, `frac_nonzero`, `cos_to_predecessor`),
computed on the full tensor for D (`logged_d`) and R_eff (`logged_r`) in every rotation `variant.json`
(`src/wsp/eval/response_intervention.py:_moments`, `_rotation_diagnostics`; README "including the logged moments"). This spec
freezes those five as the operational block. "Directional tokens", listed as logged in A5.3, are not logged by the code and
are not included. For edit i and moment k, the per-edit column is the element-count-weighted mean over the edit's edited
tensors, Δm_k,i = Σ_t n_t (logged_r[t][k] − logged_d[t][k]) / Σ_t n_t, with n_t the product of the full tensor shape. The
values are read from the re-stage `variant.json`, whose hashes enter A5.1.2 (F10.7). If the adopting A5 note or Board A
prefers another reading (for example v4 StatNN pool features), it must be fixed by id and column before scoring. It may not
be chosen afterwards.

### 1.3 Standardisation, penalty, folds

* **Training set.** Every fit has a training multiset T: (a) the full fit, T = the complete-case analysis rows; (b) LOO fold
  j, T = those rows minus pair j; (c) bootstrap draw b, T = the drawn rows with multiplicity.
* **Train-only standardisation.** For each column c and each fit, μ_c = mean over T (with multiplicity), s_c = population SD
  over T (ddof = 0). Every row the fit touches (training rows and, in LOO, the held-out row) uses z = (x − μ_c)/s_c from T
  alone. All eight columns, including the indicator, are standardised by this rule. If s_c ≤ 1e-12·max(1, |μ_c|), the
  column is set to 0 for that fit and logged `degenerate_column:<c>` (for example, a single-family training set).
  Nothing outside T (excluded, unresolved or missing pairs, ladder rows, other subsets) enters μ or s.
* **Fit.** Minimise −Σ_{i∈T} [a_i ln p_i + (1 − a_i) ln(1 − p_i)] + ½·λ·‖β‖², with logit p_i = β₀ + βᵀz_i, **λ = 1**
  ("penalty 1", A5.3.3) and the intercept β₀ unpenalised. This is the objective of
  `sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, fit_intercept=True)`. Reference solver: Newton–Raphson
  from β = 0, β₀ = logit(mean_T a), with step halving whenever the objective does not decrease. Converged when
  max |gradient| ≤ 1e-10; at most 100 iterations; nonconvergence is a recorded fit failure. No other λ, solver setting,
  column, interaction or transformation is tried (§1.5).
* **Folds.** Leave-one-component-out. Each selected pair is its own certified source component (A5.3.1 asserts 40 distinct
  components; `selection_ledger.edit_ids_sorted_sha256` `e2bd6cb2…9d6b`), so the folds are leave-one-pair-out over the
  complete-case analysis rows in canonical edit-id order. They give out-of-fold diagnostics only; the point estimates come
  from the full fit.

### 1.4 Missing values

* A pair that F10.3 excludes, that has a missing or nonfinite score in either arm, or that is unresolved is **not a row**
  (it is not in the agreement denominator, A5.3). It stays in the forty-ID ledger (§6.1).
* A resolved pair with any missing or nonfinite value in columns (1)–(8) is dropped from **both** M0 and M1, so the two
  models share rows. It is listed with reason `covariate_missing:<column>`. No imputation, carry-forward or replacement.
  The raw agreement is also reported on the complete-case rows beside the full-analysis-set agreement.
* **Estimability.** A model is fitted and reported only if the complete-case rows number n_cc ≥ 15 (the A5.3 floor, used
  here as a reporting floor, F10.5) and contain both agreement classes. Otherwise both models are reported
  `not estimable: <reason>` and no coefficient is shown. A LOO fold or bootstrap draw whose training set has one class
  only, or whose fit fails, is a counted failure. It is not retried and not replaced.

### 1.5 No tuning

λ, the column set, the standardisation rule, the solver tolerances, the fold rule, the bootstrap seed and draw count are
fixed by this section before any outcome exists. No model selection, stepwise removal, alternative transform of ρ
(for example ρ itself, ε_R, or splines) or added interaction is run or reported in the Part A report. A report computed
any other way is outside this specification and may not appear under F10.5.

### 1.6 Estimands

All are descriptive and conditional on resolution, the frozen field and the fixed 64-window panel (A5.3 "Null and claim").

1. θ_raw = Σ a_i / n_cc on the complete-case rows (exact two-sided 95 % CP, §6.3).
2. θ_adj(M) = expit(β̂₀(M)) for M ∈ {M0, M1}: the model-implied agreement probability of a resolved pair at the training
   mean of every column.
3. Δθ = θ_adj(M1) − θ_adj(M0): how much adding x_ρ moves the adjusted agreement.
4. β̂_ρ (M1 only): change in the log-odds of agreement per training-SD of x_ρ, holding the other standardised columns fixed.
   Given §1.1's collinearity, it is not an energy effect isolated from norm.
5. LOO diagnostics: mean out-of-fold log loss and Brier score for intercept-only, M0 and M1, with failed folds counted.

### 1.7 Output table (one, fixed layout)

| row | n_cc | θ_raw [CP 95 %] | θ_adj [bootstrap 95 %] | Δθ vs M0 [bootstrap 95 %] | β̂_ρ [bootstrap 95 %] | other β̂ (8 or 7 values) | LOO log loss / Brier | failed draws / folds | degenerate columns |
|---|---|---|---|---|---|---|---|---|---|
| intercept only | | | – | – | – | – | | | – |
| M0 (registered) | | | | – | – | | | | |
| M1 (M0 + ln ρ) | | | | | | | | | |

The table is followed by the list of dropped rows with reasons, and by this sentence verbatim: *"Covariate adjustment is a
descriptive diagnostic of matching confounds. It does not certify an unbiased primary contrast, repair or validate its null,
establish superiority over the constant-sign comparator, or change the primary interval, denominator or decision (A5.3.3;
F10.5; Board A F10 condition 6)."*

---

## 2. Tertile reports (F10.5(b))

Both tertile partitions below are fixed by outcome-free staging metadata. Membership is frozen **by edit id** here and is
never recomputed from outcomes, resolution or re-stage values. Every tertile has at most 14 selected pairs, so **no tertile
report can reach the ≥ 15 floor**; each is always labelled `below floor (descriptive)`.

### 2.1 Registered norm tertiles (F10.5(b) as written)

The A5.3.1 selection cells (`manifest.json:selections[].tertile`, 0 = low, 1 = middle, 2 = high), reported per family
cell and pooled by tertile index:

| tertile | lora_sft ids | chain ids | pooled n | sorted-id sha256 (pooled) |
|---|---|---|---|---|
| low (0) | `0a841d4082532e8b`, `1ab850a41051fe45`, `1b2b671e5da5d625`, `1fc51f7164052d50`, `22659cefc904e29f`, `2d1de332e36d8f00`, `453d8c08966438e0`, `455b881d5de4e6e0`, `61d1f61e25212068`, `649bd035e79488e5` | `057dcfbedea3251f`, `0bc0a0c4ad0520e0`, `0d055869527177b4`, `13b02fe2673cc296` | 14 | `64d442b072f81da1fdfaf4a7ee8a0c05f97c762904e42561b9711cc9845c0317` |
| middle (1) | `19d9a98cfb2db6e2`, `2d7f5f78fac4a9ae`, `2f61a4086395b719`, `2fb8d03d2547f357`, `3499f7dcc5968d96`, `3996bea032c24200`, `4f918f6bbd264b34`, `5f80bd482d092938`, `7d03d89c0b0299e6`, `983c5e13c59e0022` | `00f8111cbad8218f`, `014f8a9ec78db2be`, `0559051354691a70` | 13 | `041b916bf2fabb6d68f5b31c4981553a50f23061a43ff4c1653618be32822bbd` |
| high (2) | `0712caa1d248514a`, `07f9a50b29a7ef5c`, `15a53586efc72a50`, `2233954b0cc3edc0`, `2971f30e5276edd9`, `34e407b92a902282`, `5abd76a640cc2b9d`, `72d3756e968e3209`, `85366c308158ea48`, `8cfc1393b97300ab` | `0027efdf2eae7405`, `05531ce1db4df63f`, `14cdaabee246493b` | 13 | `7c663b27747ebb9f8f70490caa40a994cf06f94add40a3654465551d666a95a1` |

### 2.2 Covariate tertiles (ln ρ), frozen cut points

These cells are added by this commission because condition 6 ties the covariate to tertile reporting. They are
supplementary to §2.1 and have no greater authority. Rule (the A5.3.1/A5.3.2 convention): sort the forty pairs by
(staged ln ρ, canonical edit id); N = 40, b1 = floor(N/3) = 13, b2 = floor(2N/3) = 26; cells are ranks [0, 13), [13, 26),
[26, 40), half-open, and a value equal to a cut key falls in the higher cell. Frozen cut keys (staged ln ρ from
`manifest.json`, §1.1):

* cut 1 = (0.01010406252666643, `057dcfbedea3251f`), i.e. pooled mismatch 0.005064814290961933; the largest value below
  it is 0.010020151811342375 (`0712caa1d248514a`);
* cut 2 = (0.03502604499592402, `22659cefc904e29f`), i.e. pooled mismatch 0.01766727463448136; the largest value below it
  is 0.03462397249341332 (`2fb8d03d2547f357`).

| ln ρ cell | n | ids (ascending ln ρ) | family/norm-tertile make-up | sorted-id sha256 |
|---|---|---|---|---|
| low energy | 13 | `15a53586efc72a50`, `34e407b92a902282`, `72d3756e968e3209`, `8cfc1393b97300ab`, `0027efdf2eae7405`, `2233954b0cc3edc0`, `2971f30e5276edd9`, `5abd76a640cc2b9d`, `14cdaabee246493b`, `0559051354691a70`, `00f8111cbad8218f`, `07f9a50b29a7ef5c`, `0712caa1d248514a` | lora/high 9, chain/middle 2, chain/high 2 | `2cc773dd85e33d3e36f068d97829960a2e9b43e4f90cdf47b56956ea7d630e96` |
| middle energy | 13 | `057dcfbedea3251f`, `85366c308158ea48`, `5f80bd482d092938`, `05531ce1db4df63f`, `014f8a9ec78db2be`, `7d03d89c0b0299e6`, `3996bea032c24200`, `13b02fe2673cc296`, `19d9a98cfb2db6e2`, `2d7f5f78fac4a9ae`, `2f61a4086395b719`, `4f918f6bbd264b34`, `2fb8d03d2547f357` | lora/middle 8, lora/high 1, chain/low 2, chain/middle 1, chain/high 1 | `37f9eacdba5885cc88c487285233896cc131dbb662a87d4845665388c6c54980` |
| high energy | 14 | `22659cefc904e29f`, `3499f7dcc5968d96`, `983c5e13c59e0022`, `1b2b671e5da5d625`, `2d1de332e36d8f00`, `453d8c08966438e0`, `455b881d5de4e6e0`, `1fc51f7164052d50`, `0d055869527177b4`, `649bd035e79488e5`, `1ab850a41051fe45`, `0a841d4082532e8b`, `0bc0a0c4ad0520e0`, `61d1f61e25212068` | lora/low 10, lora/middle 2, chain/low 2 | `b3decd34fa74e474c3e83df5f4f8253f06d76d5cc9ee6ac4760722aa1afad6c6` |

Membership is the id list above. The re-stage ln ρ values are printed beside it, and any pair whose re-stage value would
fall in another cell under the frozen cuts is flagged `cell_drift`. It is **not** moved. The make-up column shows that the
two partitions largely coincide (the high-energy cell holds all ten low-tertile LoRA edits). They are not independent
views.

### 2.3 What each tertile row reports

For each cell of §2.1 (three pooled rows plus six family rows) and §2.2 (three rows), with denominators as in §6.1: selected
n; F10.3-excluded n (by reason code); missing/nonfinite-score n; unresolved n; resolved n; field agreements and
disagreements (zero predictions counted separately inside the disagreements); field agreement with exact CP 95 %;
constant-sign comparator agreement with exact CP 95 % (§4); the predicted-sign split of the resolved pairs (§4.2); and the
label `below floor (descriptive)`. No test across tertiles, no trend statistic and no interval for a difference between
cells is computed.

---

## 3. The two pre-listed subsets (F10.5(c), (e))

Hash convention: sha256 of `json.dumps(sorted(ids))` encoded as UTF-8, Python default separators. This is the repository's
`sorted_sha` convention (`scripts/a5_build_panels.py:sorted_sha`; the same form gives
`manifest.json:selection_ledger.edit_ids_sorted_sha256` = `e2bd6cb2956c0fbe64194b3f1ebbb481248ecec45b8a729a59a54b9a2bac9d6b`
for the forty ids, which was recomputed and matches).

### 3.1 S30: staged pooled realised norm mismatch ≤ 2e-2 (F10.5(c))

Definition: `manifest.json:edit_audits[i].edit_audit.pooled_norm_mismatch` ≤ 2e-2 (staged bytes, not re-stage values).
Recomputed from the manifest: exactly 30 pairs. The 10 excluded pairs equal F10.5(c)'s list and all are low norm tertile
(8 lora_sft, 2 chain). Three values lie within 1e-3 of the threshold: `1b2b671e5da5d625` 0.019942 (in),
`2d1de332e36d8f00` 0.020080 and `453d8c08966438e0` 0.020101 (out). The threshold is the registered A5.3 norm tolerance, not
a value chosen here.

```json
["0027efdf2eae7405", "00f8111cbad8218f", "014f8a9ec78db2be", "05531ce1db4df63f", "0559051354691a70", "057dcfbedea3251f", "0712caa1d248514a", "07f9a50b29a7ef5c", "13b02fe2673cc296", "14cdaabee246493b", "15a53586efc72a50", "19d9a98cfb2db6e2", "1b2b671e5da5d625", "2233954b0cc3edc0", "22659cefc904e29f", "2971f30e5276edd9", "2d7f5f78fac4a9ae", "2f61a4086395b719", "2fb8d03d2547f357", "3499f7dcc5968d96", "34e407b92a902282", "3996bea032c24200", "4f918f6bbd264b34", "5abd76a640cc2b9d", "5f80bd482d092938", "72d3756e968e3209", "7d03d89c0b0299e6", "85366c308158ea48", "8cfc1393b97300ab", "983c5e13c59e0022"]
```

* **S30 sorted-id sha256: `3da5f44f0c10307c2af7336c23fad430d13f6eb9347976e34de5d0810277267c`**
* Complement (10 excluded): `0a841d4082532e8b`, `0bc0a0c4ad0520e0`, `0d055869527177b4`, `1ab850a41051fe45`,
  `1fc51f7164052d50`, `2d1de332e36d8f00`, `453d8c08966438e0`, `455b881d5de4e6e0`, `61d1f61e25212068`, `649bd035e79488e5`;
  sha256 `7ce32f86ac171dde7da77c4c37c4b18c9e8f18b0c851d4f3e87b826a56b52280`.

### 3.2 S34: passes the noise-calibrated realised audit (F10.5(e); note §2.1 remedy (b))

Definition: norm ≤ 2e-2 + ε̂, spectrum ≤ 5e-2 + ε̂, rank above max(1e-3·σ₁, 3τ) within ±1, on every tensor. The pass/fail
set is the one F10.5(e) and the note §2.1 list, computed on the note's uncommitted scratch output
`data/runs/a5/rotation_yield/full.json` (sha256 `49b09d6f8992dc338d01c322338cd22ad528b79386ebd4c07d6084864fd0053b`, note
§0.3). This spec does not recompute it, and the list is frozen as written there. The 6 excluded pairs are
`00f8111cbad8218f` (chain/middle, the only pair the registered audit passed) and five low-tertile pairs.

```json
["0027efdf2eae7405", "014f8a9ec78db2be", "05531ce1db4df63f", "0559051354691a70", "057dcfbedea3251f", "0712caa1d248514a", "07f9a50b29a7ef5c", "0d055869527177b4", "13b02fe2673cc296", "14cdaabee246493b", "15a53586efc72a50", "19d9a98cfb2db6e2", "1b2b671e5da5d625", "1fc51f7164052d50", "2233954b0cc3edc0", "22659cefc904e29f", "2971f30e5276edd9", "2d1de332e36d8f00", "2d7f5f78fac4a9ae", "2f61a4086395b719", "2fb8d03d2547f357", "3499f7dcc5968d96", "34e407b92a902282", "3996bea032c24200", "453d8c08966438e0", "455b881d5de4e6e0", "4f918f6bbd264b34", "5abd76a640cc2b9d", "5f80bd482d092938", "72d3756e968e3209", "7d03d89c0b0299e6", "85366c308158ea48", "8cfc1393b97300ab", "983c5e13c59e0022"]
```

* **S34 sorted-id sha256: `fb1c34e68d91c03b67b90ae1c0e867eaad32edadebf2ec4b615ce13515aedfa0`**
* Complement (6 excluded): `00f8111cbad8218f`, `0a841d4082532e8b`, `0bc0a0c4ad0520e0`, `1ab850a41051fe45`,
  `61d1f61e25212068`, `649bd035e79488e5`; sha256 `b926f27079df5720ecc1fce07c97b3e223069f920a361cb3cf7f89027eb198aa`.

S30 ∩ S34 has 29 pairs, and the two complements together hold 11 pairs. The subsets overlap heavily and are not two
independent checks. Board A: "The 30-pair subset removes ten low-tertile edits; it is not an independent confirmation."

### 3.3 Intersection with the F10 re-stage: disclose, never replace

Let X be the set of pairs that the F10 re-stage excludes (F10.3 reason codes or pre-audit reasons), plus pairs with a
missing or nonfinite score in either arm. For each subset S ∈ {S30, S34}, the report gives:

1. |S| (30 or 34, frozen) and its sha256 from §3.1/§3.2, recomputed at report time from the ids actually used. A mismatch
   voids the subset report;
2. S ∩ X, listed by id with each pair's reason (exclusion code or `missing_score`);
3. unresolved pairs in S \ X, listed by id;
4. resolved n in S \ X, field agreement with exact CP 95 %, constant-sign comparator agreement with exact CP 95 %, the
   predicted-sign split (§4.2), and the label `resolved (n ≥ 15)` or `below floor (n < 15)` (F10.5: the floor only labels
   the descriptive interval).

A pair in S ∩ X is **never** replaced by another pair, S is never redefined (by re-stage values, new thresholds or any
other list), and no further subset is reported under F10.5. The subset results are reported beside the F10.3 primary
contrast and cannot replace its denominator (Board A: subsets do not "authorize choosing another primary denominator after
outcomes").

---

## 4. Constant-sign comparator (F10.5(d))

### 4.1 Definition

The registered comparator (A5.3 "Null and claim") is "every trained edit degrades more than its rotation", i.e. predicted
sign +1 for every pair: y(W + D) > y(W + R), d_i > 0 (y a loss, §0). Its agreement on a set of resolved pairs is
c = #{d̂_i > 0}/n, and d̂_i = 0 counts as a disagreement. It is shown **next to every field agreement this spec
requires**: the main F10.3 contrast, every tertile row (§2.3), both subsets (§3.3) and the covariate table's θ_raw
(§1.7). It uses the same denominator and the same exact CP 95 % interval.

### 4.2 Required companion: predicted-sign split

Each of those rows also gives the 2 × 2 counts of sign(p_i) ∈ {+, −, 0} by sign(d̂_i) ∈ {+, −} among its resolved pairs.
Both the field agreement and the comparator agreement are functions of these counts. The row also gives the share of
negative predictions, which is the agreement the field would reach if every resolved d̂_i were negative. The note's §2.3
mechanism (extra rounding energy in R_eff pushing d_i negative under positive curvature) can raise field agreement only
through the predicted-sign balance, so the reader needs that balance beside the agreement. No paired test (for example
McNemar), no difference interval and no ranking is computed from this table.

### 4.3 Interpretation rule

The Part A report states verbatim: *"No superiority of the field over the constant-sign comparator is claimed or tested.
Covariate adjustment, tertile stratification and the pre-listed subsets do not establish such superiority, do not isolate
directional geometry from quantisation or curvature, and do not certify the rotation null empirically (A5.3; F10.5;
Board A F10 condition 6)."* No sentence of the report may compare the two agreements as better or worse, including in the
abstract or summary.

---

## 5. Limitations paragraph (verbatim in the Part A report)

The paragraph below must appear verbatim, with no edits, in the Part A report's limitations section. Outcome-dependent
quantities (actual yield, agreements) are reported elsewhere and are deliberately absent from it.

> **Limitations of the Part A rotation contrast and its F10.5 sensitivities.** (1) *Quantisation.* The scored rotation is
> R_eff = bf16(W + R) − W = R + E_R, a second bf16 rounding at the predecessor grid of the same size as the edit's own
> rounding (median ε_R/ε_D 1.0003 over 2,188 root-edit tensors; staged pooled realised norm mismatch +0.0007 to +0.0309;
> pooled ε_R up to 0.365, largest in the low norm tertile; `docs/design/ROTATION_YIELD_2026-11-11.md` §§1.2, 1.4). Its
> first-order part ⟨g, E_R⟩ is inside the registered prediction ⟨g, D_eff − R_eff⟩. The realised extra energy is not
> removed by design, and the exclusion rule (F10.3) does not test realised rotation quality. (2) *Curvature.* The
> second-order effect of that energy is not priced by the registered prediction. The F9 quadratic forecast's q is a
> nonnegative Gauss–Newton magnitude only, never a signed Hessian correction (A5.8.1; F9a.2(c)), and for the chain picks it
> uses base K-FAC factors (F6.2, `base-factor`). The constant-sign nuisance argument of the design note §2.3 is a plausible
> mechanism under positive, roughly isotropic curvature, not a theorem for every pair; fresh rounding depends on W and R and
> can correlate with the field or the prediction signs (Board A F10). (3) *Sign imbalance.* The registered power inputs
> carry rotation prediction signs 43 positive and 98 negative of 141 donors
> (`docs/prereg/a5_manifests/power/fields_manifest.json:summary.sign_distribution.rotation`). Negative is the direction in
> which the energy nuisance pushes d_i, so a constant-sign nuisance can raise agreement when predictions are imbalanced
> towards negative. The report's predicted-sign split shows the actual balance. (4) *No realised spectrum matching.*
> "Spectrum-matched" refers to the pre-rounding construction only. No realised norm, spectrum or rank match is claimed
> (F10.4; A5.3.3), and passing the F10.3 audit is not evidence of one. (5) *Chain ladders.* For the 8 non-root
> chain picks, the 16 ladder variants scale the realised increment D_t, not the unavailable raw step Δ_t
> (`realised-increment ladder`, F10.9). Equivalence to the registered raw-step ladder is not established: the 0.066–0.392
> noise-floor ratio is a heuristic scale, not a measured error or an upper bound (Board A F10, F10.9 ruling). (6) *Null
> branch frequency.* Under the registered NULL-SIZE proxy the ordered decision returns the funding-only **alternative**
> branch in 1,939 of 10,000 null replicates (rate 0.1939, exact CP 95 % [0.18619, 0.20179]), against ordered go 180/10,000
> (0.018) (`docs/prereg/a5_manifests/nullsize/report.json`; F5.7). An alternative outcome is a funding stop for the field,
> never evidence of wrong signs. (7) *Proxy and dependence (F7).* The NULL-SIZE and POWER-AND-YIELD results (ordered power
> 0.8540) are conditional on F2's edit-minus-actual-predecessor noise surrogate, the |donor mean| effect scale and the 0.75
> sign-agreement alternative. They are not empirical rotation-null or power certification. The single LoRA low-tertile
> donor is reused for all ten slots of that cell. The three contrasts rescale one donor block and under the null are one
> test, not three. Simulated yield 1.0 rests on treating 423 unavailable audit entries as passes, and the
> `unavailable_audit_fails` scenario resolves no pair (Board A F7 conditions 2–4). (8) *Scope.* The claim is agreement above
> chance conditional on the frozen field, the fixed 64-window panel and resolution, on development edits whose D3 outcomes
> were already known (A5.3 "Exposure"). It is not an item-population claim. Covariate adjustment, tertile and subset reports
> are descriptive: they neither certify an unbiased primary contrast nor validate its null, and none can change A5.5's
> outcome.

---

## 6. Denominators, units, uncertainty and authority

### 6.1 Denominators (every report)

Every F10.5 table carries the full forty-ID ledger for its set: **selected → pre-audit excluded (by code) → F10.3 excluded
(by code: `construction_failed`, `support_failed`, `replay_failed`, `rounding_bound_failed`, `sub_resolution`) → missing
or nonfinite score (by arm) → unresolved → resolved = agreements + disagreements (zero predictions shown inside the
disagreements)**. The registered realised verdict (A5.3 tolerances) is shown as a logged field, never as a reason (F10.3).
Every count has an id list in the report's ledger appendix. Denominator by report:

| report | universe | denominator of agreement |
|---|---|---|
| main contrast (reference only; registered, A5.3) | 40 selected | resolved F10.3-passing pairs |
| covariate models (§1) | 40 selected | complete-case resolved pairs n_cc |
| norm tertiles (§2.1) | 14 / 13 / 13 pooled; 10/10/10 LoRA, 4/3/3 chain | resolved pairs in the cell |
| ln ρ tertiles (§2.2) | 13 / 13 / 14 | resolved pairs in the cell |
| S30, S34 (§3) | 30, 34 | resolved pairs in S \ X |
| constant-sign comparator (§4) | as the row it sits beside | the same denominator |

Unresolved pairs, exclusions and missing scores are listed, never dropped from the ledger. No denominator borrows pairs
across cells, subsets or contrasts (A5.3.1).

### 6.2 Units

d̂_i in nats (WikiText `extra.mean_nll`, mean over the 64 registered windows). p_i in nats (first-order forecast).
Agreement: proportion of resolved pairs, where a pair is one certified source component. ln ρ: natural log, dimensionless.
Effective norm: Frobenius norm of the realised bf16 increment (fp32 subtraction), weight units. Model coefficients: log-odds
per training-SD of the column. θ_adj and Δθ: probability scale.

### 6.3 Uncertainty methods

| quantity | method |
|---|---|
| Every agreement proportion (field, comparator, any row) | exact equal-tail two-sided 95 % Clopper–Pearson (A5.3.2): lower = Beta⁻¹(0.025; x, n − x + 1) (0 if x = 0), upper = Beta⁻¹(0.975; x + 1, n − x) (1 if x = n); n = 0 reported `no resolved pairs` |
| Per-pair resolution (inherited, not recomputed) | registered paired block bootstrap, 32 two-window blocks, 4,000 draws, percentile interval (A5.3) |
| θ_adj, Δθ, β̂ (§1) | **component bootstrap**: resample the n_cc complete-case pairs (one pair = one component) with replacement, **4,000 draws**, generator `numpy.random.Generator(numpy.random.PCG64(20261107))`, one index matrix `rng.integers(0, n_cc, size=(4000, n_cc))` drawn once over rows in canonical edit-id order and shared by M0 and M1 (common draws); each draw re-standardises and refits per §1.3; percentile interval with linear interpolation at 0.025/0.975 (A5.3.2). Failed draws are counted, not retried or replaced; if any draw fails, the interval is labelled `diagnostic (k of 4000 draws failed)` |
| LOO diagnostics | point values only, with failed-fold counts |
| NULL-SIZE branch frequencies (§6.4) | copied from the frozen report, exact CP 95 % as recorded |

No other interval, p-value or test is computed for an F10.5 report.

### 6.4 F5.7 table (required beside every Part A decision report)

As F5.7's amended reporting rule requires, copied from `docs/prereg/a5_manifests/nullsize/report.json:scenarios`. These are
null rates under the proxy, distinct from observed branch outcomes:

| scenario | ordered go (null) [CP 95 %] | alternative (null) count / rate [CP 95 %] | go / alternative / unresolved |
|---|---|---|---|
| `registered` | 0.0180 [0.0155, 0.0208] | 1,939 / 0.1939 [0.18619, 0.20179] | 180 / 1,939 / 7,881 |
| `variance_x2` | 0.0162 [0.0138, 0.0189] | 1,884 / 0.1884 [0.18078, 0.19620] | 162 / 1,884 / 7,954 |
| `unavailable_audit_fails` | 0.0000 [0.0000, 0.0004] | 0 / 0.0000 [0.00000, 0.00037] | 0 / 0 / 10,000 (vacuous) |
| `constant_positive_mean` | 0.0191 [0.0165, 0.0220] | 2,177 / 0.2177 [0.20964, 0.22592] | 191 / 2,177 / 7,632 |
| `cell_mask_noise` | 0.0155 [0.0132, 0.0181] | 1,544 / 0.1544 [0.14737, 0.16163] | 155 / 1,544 / 8,301 |

If F10.6 forces a slot-masked POWER-AND-YIELD run, its embedded NULL-SIZE pass is added as a separately labelled row. It
replaces no row.

### 6.5 Authority

**None of the reports in §§1–4 has A5.5 or F5.5 authority.** They do not enter A5.5's validity step, the step-2 ladder
alternative, the step-3 Part A evidence rule or Part B eligibility (A5.4; A5.5 step 4). They do not enter the F5.5 joint
line-stop rule, any funding branch, or any gate of F10.6 or A5.3.2. A5.5 is evaluated once, on the F10.3 set, as
registered (F10.5: "the decision uses F10.3's set"; "None of these is decisive or can change A5.5's outcome"). If a
sensitivity disagrees with the main contrast, that is reported as a disagreement. It does not reverse, qualify or delay the
decision.

---

## 7. Board A condition 6 checklist

| # | Board A F10 condition 6 item (with the bias ruling it cites) | Frozen in |
|---|---|---|
| 1 | Freeze before scoring; discharge in the single final report | header (adoption by dated A5 note before scoring); §6.5 |
| 2 | Implement the fixed F10.5 **covariate** report | §1.1–§1.7 |
| 3 | … **tertile** reports | §2.1–§2.3 |
| 4 | … **subset** reports | §3.1–§3.3 |
| 5 | … **constant-sign** report | §4.1–§4.2 |
| 6 | Train-only standardisation | §1.3 (training multiset, μ/s from T only, degenerate columns) |
| 7 | No tuning | §1.5; fixed λ, columns, solver, folds, seed (§1.3, §6.3) |
| 8 | No decision authority | §6.5; §1.7 closing sentence; §3.3 |
| 9 | Preserve subset ID lists | §3.1, §3.2 (lists, sha256, complements) |
| 10 | Disclose any intersection with newly audit-excluded / missing rows; never replace them | §3.3 |
| 11 | Carry quantisation / curvature / sign-imbalance limitations | §5 items (1)–(3) |
| 12 | Absence of realised spectrum matching | §5 item (4) |
| 13 | Uncertain raw-chain equivalence (F10.9) | §5 item (5) |
| 14 | F5.7 null alternative frequencies | §5 item (6); §6.4 |
| 15 | F7 proxy / dependence limits | §5 item (7) |
| 16 | No superiority claim over the constant-sign comparator | §4.3; §1.7 closing sentence |
| 17 | No empirical null certification from covariate adjustment | §1.7 closing sentence; §4.3; §5 item (8) |
| 18 | All denominators, unresolved cases and exclusions stay in the forty-ID ledger | §6.1 |
| 19 | (bias ruling) Subsets do not authorise another primary denominator; 30-pair subset is not independent confirmation | §3.2 last paragraph; §3.3 |
| 20 | (bias ruling) Agreement above chance does not isolate directional geometry from quantisation/curvature | §4.3; §5 items (1)–(2) |

### 7.1 Open points for the adopting A5 note

1. Confirm the operational reading of "v4 differences" as the five logged `_moments` differences, weighted by element count
   (§1.2), or fix another column list by name before scoring.
2. Confirm the reading of "penalty 1" as λ = 1 on the summed log loss with an unpenalised intercept (§1.3).
3. Confirm that the covariate tertiles of §2.2 are reported in addition to F10.5(b)'s registered norm tertiles (§2.1),
   with no greater authority.
4. The implementing code (CPU only; synthetic-input tests covering one-class training sets, degenerate columns, missing
   covariates, failed draws, CP boundary cases and the subset-hash check) must be pinned in A5.1.2 before scoring, like any
   other Part A analysis code. This spec contains no code.
