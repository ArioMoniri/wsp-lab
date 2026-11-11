# World-model — round 6, 11 Nov 2026

No experiment run; nothing under data/ opened. Counts are recomputed from committed manifests (`components.jsonl`, `snapshot.json`, `predictions_f11.json`); no score read.

## BLUF

1. The field is a fixed second-order Taylor model: a linear unit with ~10⁸ weights g (this model's gradient) plus a block-diagonal Gauss–Newton quadratic, read out by 3 coefficients. "Single neuron" fits the readout, not the information.
2. It cannot represent cross-tensor interactions, negative curvature, higher orders, a per-edit gain Δy/a, or path dependence. Inside its trust region it predicts effects in nats; outside, only direction. Part A tests sign only.
3. No outcome evidence about the field exists: "33/33" compares two computations of a; power 0.8722 assumes a 0.75 alternative.
4. Part B's learned arms already are model-specific world models, but none is registered against the 3-coefficient field.
5. JEPAWG-style dynamics (b) already exists here as M3: ρ 0.15/0.13 (seed), 25 successor pairs. The weight transition W + D is known exactly; the unknown is y. Not recommended.
6. The 342-row loss is about components, not architecture. Chain-step targets sit below the noise floor; root edits do not.
7. Brief correction: only 147 development root entries can be scored without spending G7 freshness, at ≈ 2.5 GPU-h, not 565 at 9.8.
8. **F12** (register now, reads no Part A): field + learned residual head (1,137 parameters) against field-3 on 270 roots / 234 components, A5.4.2 intervals, outcome-free power first.
9. **F12-I**: 36-entry additivity panel (≈ 0.62 GPU-h). The field predicts zero interaction exactly; a resolved interaction directly supports the thesis.
10. Part A, Part B, A5.5 and F5.5 unchanged; F12 funds nothing.

Fetched 11 Nov (HTML truncated; † = summary-derived): [2606.13461](https://arxiv.org/html/2606.13461), [2611.11604](https://arxiv.org/html/2611.11604), [2610.13991](https://arxiv.org/html/2610.13991), [2508.02381](https://arxiv.org/html/2508.02381), [2607.07127](https://arxiv.org/html/2607.07127v1), [2610.30847](https://arxiv.org/html/2610.30847v1), [2608.15483](https://arxiv.org/html/2608.15483v1), [2605.06298](https://arxiv.org/html/2605.06298v2).

## Q1 — Function class

With D_t = fp32(bf16(W_t+Δ_t)) − fp32(W_t), `response_field.contract` gives a = Σ_t⟨g_t, D_t⟩ and q = Σ_t ½tr(G_t D_t A_t D_tᵀ) ≥ 0 (factors truncated at 99 % energy). The A5.8.3 readout is ŷ = β₀ + β₁a + β₂q (ridge 1): a quadratic in D whose coefficient tensors are fixed by the base's gradient and K-FAC Fisher; only β is learned. The code notes q's sum "omits weight/bias off-diagonal interactions and is not joint q".

**Cannot represent.**
- *Cross-tensor interactions.* a and q are additive across modules, so for disjoint tensors the field predicts Δy(D₁+D₂) − Δy(D₁) − Δy(D₂) = 0 exactly (rounding is elementwise). Multi-site edits (Sparks: two layers; Concept Country: nine) are its blind spot.
- *Negative curvature.* The GN term always predicts harm; ŷ(s) = β₁sa + β₂s²q changes sign at most once, at s* = |a|/q when a < 0 < q, improve-then-harm only.
- *Third order and beyond*: saturation, thresholds, termination (the teacher-forced field is blind to generation).
- *Per-edit gain*: one β₁ cannot track Δy/a varying with family, rank or norm.
- *Path*: F6 withdrew the chain cross term (audit error 0.336); only re-anchoring (≈ 43 s/row) restores exactness.

**The field's self-assessment** (outcome-free, `predictions_f11.json`): κ = q/|a| at s = 1 has median 0.023 (κ > 1 for 2 of 52); s* < 1 for 1 of 46 defined. Median predicted |a| is 0.228 nats for LoRA (34) and 0.00054 for chain (18). Rounding already makes a nonlinear in s: a(0.5)/a(1) has median 0.497, range −10.45 to 0.815. κ omits the non-GN Hessian and cross blocks, so it cannot certify locality.

**Answer.** A first-order readout predicts *effects* wherever the Taylor remainder is small next to |a|, and only *direction* beyond. Part A tests only direction; magnitude is tested only by F9's E₀ on F8/F9 panels. A predicted 0.228 nats for LoRA SFT is large: if realised changes are much smaller, signs may agree while E₀ < 0, and β₁ rescales only uniformly.

**Rounds 4–5 established** computational determinism (33/33 ladder, 25/25 rotation between two pipelines, `comparison_with_power_fields`) and power under an assumed DGP (power README: "not an empirical certification"). Nothing about outcomes.

## Q2 — Candidates (target: predecessor-relative WikiText Δy, this base only)

| | Inputs | Parameters | Supervision (components) | Prior (qualitative) | Test |
|---|---|---|---|---|---|
| (a) Part B: augmented StatNN, StatNN+field, K+F, probe ridge, A1-lite | v4 stats + ≈ 9,000 field/chart columns; 39-col tokens | tree 300×8 leaves; K+F 1,137 | H1w 386 logical rows, reused (A4.4) | already the object | C, C_abl, C_rand, C_perm, K+F:F; never vs field-3 |
| (b) JEPAWG dynamics | z_t = E(φ(W_t−W₀)); (z_t, edit) → z_{t+1}; head → Δy | M3 predictor 8,336 (my count) + encoder | 457 chain rows outside Part A, **73 components**, median 5 each | low | none |
| (c) Learned readout | per-tensor a_t, q_t, spectrum, role/layer, probe D·x | 1,137 + ridge | 270 roots / 234 components | moderate-low | **F12** |
| (d) Activation-assisted | TF-v1 KL vector or D·x | ridge | as (a) | highest; runs the edited model | A5.4.3, descriptive |

(a) are world models in information (this model's outcomes and field) but static regressors, and no contrast sets them against the field. That gap is the thesis's comparison.

Against (b):
- M3 is exactly this design (`src/wsp/predictor/models/__init__.py`, class `JEPA`: 16-d latent, VICReg, adjacent cumulative-delta pairs). It scored 0.15/0.13 and 0.12/0.07, with "98 of 521" pairs featured and "25 evaluated" (d3v2f).
- JEPAWG maps couplings to weights statically (2607.07127: 361 flows per lattice, d_z 16), and "outside the training range, nothing constrains the decoded weights" (2610.30847†).
- Bulk weight motion is predictable only in "localized, time-varying pockets" (2608.15483).
- NOVA (2605.06298) needs 33–43 M parameters and 8,000–324,313 sequences†, and models no edits.
- In editing, W_{t+1} = W_t + D is exact. The unknown is y (a reward model), whose analytic state update is re-anchoring.

## Q3 — Why encoders lost on 342 rows

- 342 rows in 278 clusters, and the family split lies within [−0.29, 0.23] for all models.
- Descriptor-only (0.51/0.61) nearly matches StatNN (0.56/0.66), so the signal is mostly descriptors.
- Re-instantiation moved neural ρ by a median 0.09 (21 of 49 by > 0.10), as large as the neural-to-tree gap.
- M3 saw 25 evaluated pairs (d3v2f).

What changes it is inputs and components. Field tokens hand the learner a per-tensor analytic prediction to correct. Noise: the WikiText panel-mean half-width bound of 0.00487 nats (`tf-v1-requalification-2026-11-06.md`) implies a per-entry SE ≈ 0.0025. Predicted chain-step |a| (0.00054) is below that; LoRA (0.228) far above. Step targets would teach a dynamics model mostly noise unless realised step effects dwarf the field's (unknown).

**Sample size.** At C = 0.15 (log 0.85 = −0.163), one-sided 0.025 and power 0.80, K ≈ 7.85σ²/0.0264 ≈ 297σ² components, where σ is the between-component SD of the relative loss difference. σ is unknown pre-outcome, hence a simulation.

**Closing rows.** TF-v1 primaries exist for 689 entries (648 chain) in 126 components (`snapshot.json`). Most of the remaining 1,259 − 689 are sealed "confirmatory" (106 components) or "fresh_unselected" (106); scoring them spends G7 freshness. Development, outside Part A, bytes present and unscored: 177 entries / 172 components, of which 147 are roots (79 abliteration, 49 random_dir, 19 lora_sft; the 30 task vectors are not roots). 147 × 1.04 min ≈ 2.55 GPU-h (`docs/RESULTS.md`), or 2.66 at the census rate of 11.2 GPU-h per 618 (`PROGRESS.md` 7 Nov).

## Q4 — Anchoring vs learning the response

Anchoring is model-specific by construction: the same D gets a different a on another base. The field is this model's local world model, with zero outcome fitting. The thesis's strong reading is that outcome supervision adds information beyond local geometry, and the programme touches it only indirectly: C_abl (field inside a tree), K+F:F, probe/A1-lite (activation access), F9 E₀ (zero-fit magnitude).

Missing:
1. A held-out learned-vs-field-3 head-to-head.
2. A design that nests the field, so a win belongs to the learned residual.
3. Fresh rows (Part B reuses examined H1w outcomes).
4. Leave-one-family-out transfer, where the field is family-agnostic and learned models scored ≈ 0 (d3v2f).

## Q5 — F12 (registrable now)

- **Rows.** Roots only, so the field is exact everywhere. 123 existing TF-v1 roots outside the 52 Part A components (105 chain root steps, 10 lora_sft, 4 abliteration, 4 random_dir; 89 components) plus the 147 purchasable roots: **270 rows / 234 components**. Confirmatory and fresh components, task vectors and non-root chain steps are never scored.
- **Folds.** 5 outer component folds, hash-sorted under `20261111:f12:`, stratified by family; inner folds, seeds 0/1/2 and stopping as in the F5 pins.
- **Endpoint.** TF-v1 WikiText mean-NLL Δy (`configs/eval_tf_v1.yaml`).
- **Models.** Baseline F3 is the A5.8.3 readout fitted on outer-training rows. Candidate WM-R = F3 + h_θ, with h the F5-pinned K+F head (39 columns, 1,137 parameters) plus probe D·x summaries. It is trained by L1 on F3's training residuals and is zero at zero edit.
- **Contrast.** C_WM = 1 − A_WM/A_F3, A5.4.2 conditional and joint bootstrap-t, 4,000 draws, seed 20261107.
- **Decision (scientific wording only).**

  | Result | Condition |
  |---|---|
  | Correction detected | C_WM ≥ 0.05 and both lower bounds > 0 |
  | None detectable | both upper bounds < 0.05 and power ≥ 0.80 at C = 0.15 |
  | Unresolved | otherwise |

  Descriptive alongside: zero-fit E₀, same-row augmented StatNN, leave-one-family-out C_WM.
- **Power, outcome-free, before purchase.** Simulate Δy = a + γh*(x) + ε on real tokens, with h* a random K+F-head function, ε from qualification noise, and γ set for oracle C = 0.15 (γ = 0 for the null). Gates: size ≤ 0.07 and power ≥ 0.80; otherwise F12 is not bought.
- **Cost.** ≈ 2.55 GPU-h scoring, 3 GPU-h cap. Fields on CPU, ≈ 7 CPU-h at the measured 13,305.8 s per 141 donors (`fields_manifest.json`).
- **F12-I.** 16 development lora_sft roots, each split into early and late layer halves, plus Astra split by tensor and Concept Country in halves: 36 entries ≈ 0.62 GPU-h. Statistic I = Δy(full) − Δy(part₁) − Δy(part₂) against the F8a floor; the field predicts I = 0.
- **Ordering.** Register now: F12 reads nothing from Part A and excludes its components. Score after the Part A ledger is sealed. No A5.5, F5.5 or funding authority.

## Q6 — Editing side

Ranked by how badly they should embarrass the field:
1. Coordinated multi-site edits (Astra in F8-E, Concept Country in F8-F); the interaction itself is untested, hence F12-I.
2. Large deletions (F9-P at 50 %).
3. Objective-trained masks: Deletion Tuning is selection-biased on a and excluded as unpriced (F8.2).
4. Termination edits, which are generation-side (F9.6).

Out of scope:
- RL editing (2606.13461) edits one 128×32 MNIST layer, with no outcome forecast: "editing all layers remains infeasible".
- 2611.11604 writes 1×16 or 2×32 networks, verified by Storm afterwards.
- The only learned forecasters work in low-dimensional modification spaces: PPF (775 masks, error < 0.0011 held out) and IDM (a program calling a frozen LLM).

**Brief corrections.**
- 0.45/0.50 and 0.42/0.56 are both m1_matched (W2T-style); the JEPA-style M3 scored 0.15/0.13.
- The chain has 675 entries in 93 components (618 is the census count; 99 counts lineages with ≥ 4 nodes).
- 689 TF-v1 primaries, not 694.

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---|---|
| F12 WM-R vs field-3 | field + learned residual (1,137) on roots | C_WM, A5.4.2, outcome-free power ≥ 0.80 | ≈ 2.55 (cap 3) | upper bounds < 0.05 at power ≥ 0.80 → close learned-WM line for roots |
| F12-I additivity | field predicts I = 0 across tensors | I vs F8a floor, 18 edits | ≈ 0.62 | no resolved I → no interaction target |
| F12 family transfer | learned vs family-agnostic field | descriptive C_WM per family | 0 | — |
| (b) JEPAWG dynamics | latent transition over chain | not registered | 0 | reopen only if F12-I resolves interactions |
| (a) Part B arms | learned, model-specific | A5.4/F5 unchanged | — | unchanged |
| 565-row closure | — | — | — | rejected: spends G7 freshness |
