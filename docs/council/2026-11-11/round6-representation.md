1. The analytic field compresses a model-wide gradient and Fisher geometry; its readout has a severe bottleneck (§1; RF).
2. First order can predict a differentiable endpoint locally; arbitrary finite edits need additional assumptions (§1).
3. Cross-tensor curvature and changed activation routes are absent; the chart restores only projected coordinates (§1; RC).
4. The 33/33 agreements compare forecasts with forecasts, not outcomes; Part A remains unscored (§1; INV).
5. Augmented StatNN and K+F already learn this model’s response from outcomes, in a restricted surrogate sense (§2; A5).
6. Prefer a regularised kernel residual over fixed field/chart/probe tokens to learning another large frame (§2).
7. More rows help only with eligible labels, source separation and resolved effects; 1,259 artifacts are not independent labels (§3; B6).
8. Root contrasts are my default target; chain steps need state and paired-noise evidence before priority (§3).
9. Register F12 now without reading Part A; execute after the existing programme, with power qualification (§5).
10. Learned gate masks and coordinated edits threaten locality most; F8 does not yet purchase learned masks (§6; A5).

Representation, round 6, 11 November 2026. Proposals below are unexecuted. **Local references:** A5 = `docs/PREREG_H1_TF.md`; RF/RC = `src/wsp/features/response_field.py` / `response_chart.py`; P4 = `docs/reviews/P4/d3v2f-summary-2026-11-08.md`; B6 = `docs/council/2026-11-11/BRIEF-r6.md`; INV = `docs/prereg/a5_manifests/part_a/PRESCORE_INVENTORY_2.md`. These abbreviations cite files throughout.

### 1. Function class and its blind spots (Q1)

For realised increments D, the analytic arm is

\[
a(D)=\sum_t\langle g_t,D_t\rangle,\quad q(D)=\tfrac12\sum_t\mathrm{tr}(G_tD_tA_tD_t^\top),\quad
\hat y=b_0+b_1a+b_2q.
\]

Zero-shot sign uses **a alone**; the three-coefficient ridge calibration is secondary. The Part B primary is a tree, not this equation (A5.4, A5.8.3). At a fixed anchor this is one fixed linear functional plus one fixed positive-semidefinite block quadratic, with at most three fitted coefficients. Calling it a single neuron describes the narrow readout, but hides the model-specific computation producing g/G/A (RF).

The gradient already contains downstream, non-local computational consequences of each infinitesimal parameter change. What is missing is **edit-dependent** downstream change. q includes within-matrix interactions, including non-diagonal input/output covariance; it excludes cross-tensor Hessian blocks and signed negative curvature. Cancellation between tensor contributions survives in a; arbitrary interactions between those contributions do not. Fisher magnitude is not the endpoint Hessian (RF; A5.4.2).

Along an ideal unrounded ray, a(sD)=sa and q(sD)=s²q. A zero-intercept quadratic can exhibit a nonzero crossing; zero-shot a cannot reverse sign for positive s. Realised bf16 ladders need separate contractions because rounding breaks exact scaling. Neither feature describes arbitrary multiple crossings, saturation, gating changes or autoregressive termination. A nonzero calibration intercept also forfeits exact zero at D=0 (A5.8.3; RC).

Endpoint response is path-independent given complete weights and fixed evaluation; apparent trajectory dependence comes from different predecessor states. Base anchoring cannot distinguish equal increments at different predecessors. Re-anchoring restores the local derivative, not a finite-step dynamics model; F6 withdrew the Fisher gradient-transport substitute (A5 F6; `docs/council/2026-11-11/SYNTHESIS-r5.md`).

The chart adds tensor location, predecessor displacement E, raw inner products, rounding/tail diagnostics and fixed 4×4 projected contractions. Token objects retain projected matrices, but `to_columns` exports their elementwise contraction blocks, not every projected matrix. Both views discard a large nullspace (RC). It creates learnable coordinates, not recovered full interactions.

INV reports 33/33 ladder forecast agreement twice and 25/25 rotation agreement with older power forecasts. These establish sign consistency only. The frozen prediction **summary** has 52 edits, nine rotation abstentions, 42 base-field and ten chain-exact sources (`docs/prereg/a5_manifests/part_a/predictions_f11.json`). No observed-effect validation follows. A differentiable endpoint admits a first-order local approximation; even local sign requires a nonzero derivative and sufficiently small remainder. Finite-edit success is empirical, currently unestablished here.

### 2. Concrete model-specific surrogates (Q2, Q4)

**(a) Existing arms.** Augmented StatNN consumes v4 statistics and ordered field/chart columns; K+F combines frozen K-FAC directional blocks and field tokens. These learn this receiver’s outcome response, so satisfy the weak, operational “world model” definition. They do not simulate internal computation. The later F5 schema pins supersede the approximate 1.6k head: K+F has **1,137** nominal parameters; L has **163,057**, including frames. Eligible rows are the registered 386 logical G1 rows and LoRA restriction, not the entire zoo (A5 F5.1–F5.2/schema pins). *Prior:* possible gain over the field; no positive-gain presumption over same-information StatNN. The tree uses A5.8.4’s 300-tree default. CPU fitting needs zero GPU-h.

**(b) Latent transitions.** Input z(predecessor), realised D and E; predict successor latent and endpoint increment. Width and full parameter count must be fixed before purchase; no justified architecture count exists yet. B6’s 618 chain entries/99 lineages are an inventory ceiling, not paired training examples. P4’s earlier dynamics arm had only 98 feature-complete successor pairs, 25 evaluated. *Prior:* lower chance of beating StatNN than fixed-token readouts until eligible transitions and held-out rollout errors are audited. Extraction/training cost is unpriced; defer construction.

**(c) Recommended readout.** Use a Gaussian-kernel ridge residual on fixed ordered per-tensor a/q, validity/anchor flags, chart values/masks, v4 statistics, and cached-probe D·x mean/RMS. Retain exact layer/role slots. For a proposed single-objective design with T field slots and H supported probe output coordinates, dimension is

\[
d=d_{v4}+56N_m+3N_v+2T+2H+d_{flags}.
\]

Here d_v4 is the frozen pooled-statistics width, derived from `pool.py` rather than the stale width in the extraction docstring. Chart values contain 48 contraction coordinates plus seven scalars and presence per matrix, two norms plus presence per vector; coordinate masks/field availability/anchor metadata contribute to the explicitly frozen d_flags (RC; `src/wsp/features/extract.py`; A5.8.3). This is a schema formula, not an asserted cache width. Include E; do not confuse base-relative P3 features with step-relative chart tokens (`extract.py`; RC).

Fit the residual after fold-local a/q calibration, subtract its zero-edit kernel prediction, and retain the calibration comparator unchanged. There are n_train kernel coefficients plus three calibration coefficients, no learned long-axis frame. Freeze bandwidth from training-only distances; ridge choice and preprocessing use four source-component inner folds inside the registered outer folds, with source/ancestor closure. All derivatives, probes and feature transforms must exclude outer scoring items. Same-information StatNN receives every scalar; field-absent/random/permuted versions remove/rebuild all proxies (A5.4).

No honest minimum-row guarantee follows from d or parameter count. Start with eligible development rows; require component-held-out learning curves and power qualification before adding labels. *Prior:* better than a global field if residuals depend on location/state; uncertain against StatNN.

Round 3’s identifiability problem remains: label-equivalent learned frames and feature collisions cannot be repaired by a larger head (`docs/council/2026-11-10/round3-representation.md`). [LoL](https://arxiv.org/abs/2410.04207) addresses factor gauge symmetry; [W2T](https://arxiv.org/html/2603.15990) canonicalises updates before tokenisation. Neither guarantees endpoint sufficiency. [Finite Probes](https://arxiv.org/abs/2609.33901) gives identification/universality under architectural assumptions, not a certificate for our finite compressed cache.

**(d) Activation-assisted reference.** Replace linear D·x with actual successor-minus-predecessor hidden responses on scoring-disjoint prompts; freeze layer/position summaries and fit the same kernel (n_train coefficients). This captures changed downstream routes but requires post-edit forwards. [Rane](https://arxiv.org/html/2611.07610) finds activation/output features more informative than weight statistics for domain classification; that motivates, but does not validate, endpoint regression. *Prior:* strongest nonlinear reference, unknown advantage over StatNN. Incremental forward/extraction cost is unpriced. Anchoring alone is model-specific measurement; outcome-trained readouts test response learning. Missing: an independently powered residual comparison and prospective state-conditioned transitions.

### 3. Rows, noise and target choice (Q3)

P4 has 342 rows/278 ancestry clusters: M1-matched seed 0.45/0.50 versus StatNN 0.56/0.66; descriptor-only 0.51/0.61. Matching layer identity did not eliminate the gap. Pooled 18-statistic tensor features include norms/spectrum and aggregate by role/depth; this supplies useful low-variance bias (`stats.py`, `pool.py`). Limited independent supervision and expensive representation learning are plausible explanations, not demonstrated causal diagnoses. P4 targets fast-tier rates, not WikiText TF-v1; its correlations cannot estimate F12’s gain.

B6 gives 694 TF-v1 records versus 1,259 artifacts: the nominal gap is 565, costing 565×1.04/60≈**9.8 GPU-h scoring only**. Reconcile revision, ancestry, labels and development/sealed eligibility before purchase; records are not automatically training rows. `docs/PROGRESS.md` separately reports 652/675 chain records and 95 complete lineages on 7 November, illustrating inventory-denominator differences.

The measured 0.00487-nat panel half-width and 0.01282 maximum are interval summaries, **not per-row Gaussian σ** (`docs/reviews/P5/tf-v1-requalification-2026-11-06.md`). More rows do not shrink each label’s shared-item uncertainty. Power must preserve item covariance and component sizes.

*Prior target choice:* root endpoint changes usually offer larger accumulated signal than adjacent steps. A step target has variance Var(y_next)+Var(y_prev)−2Cov(y_next,y_prev); common-item pairing may help, but small steps can remain unresolved. Chains test changing state more directly, yet 99 lineages are not 618 independent experiments (B6). Prioritise resolved root contrasts, then qualified paired steps; compare learning curves and paired precision before declaring steps more informative per row.

### 4. F12 proposal and editing stress cases (Q5–Q6)

Register the kernel design **now**, blind to Part A, with an immutable development-only eligibility/fold manifest. No sealed rows enter training; no zoo-wide relabelling by default. Execute after existing Part A/Part B ordering. Primary new contrast: absolute-residual reduction versus calibrated field; co-required comparison versus same-information augmented StatNN. Use A5.4.2 conditional/joint ratio intervals, C≥0.15 with lower bounds >0, and eligible no-field/random/permuted controls beaten. Simulate the entire conjunctive decision with refitted nuisance, empirical component/item covariance and inflated-noise sensitivity; require power lower bound ≥0.80. Underpowered/invalid means unresolved. Existing exposed outcomes permit a funding screen, not fresh confirmation (A5 A4.4). F12 cannot amend F5.5 or rescue failed existing gates.

Mechanism ranking is an inference: coordinated [learned gate deletions](https://arxiv.org/html/2611.10866) and sequential [RL masks/shifts](https://arxiv.org/html/2606.13461) most threaten fixed derivatives through changed routing/state. Sparse support does not imply a small functional displacement. Need support-aware hidden responses and cross-tensor interactions, or re-anchored state-conditioned dynamics. [SWEET](https://arxiv.org/abs/2611.10974) suppression and [TFTV](https://arxiv.org/abs/2609.09054) activation-derived rank-one edits need direction/location; low rank alone does not defeat the field. [Column edits](https://arxiv.org/abs/2611.09382) can still change generation nonlocally. [WeightLoop](https://arxiv.org/abs/2611.07319) uses feedback-driven programs; program complexity alone implies no prediction lower bound. [PPF](https://arxiv.org/html/2508.02381) learns from 775 masks in a restricted pruning space, not arbitrary deltas.

F8 includes external edits subject to released bytes, Wanda/random deletion and TFTV; learned masks are excluded, SWEET deferred. F9 adds pruning and generation diagnostics, not learned-mask purchase (A5 F8–F9).

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---|---|
| Existing K+F/StatNN | Fixed anchored tokens | A5.4, F5.1–2; unchanged | 0 fitting, CPU | Existing rules only |
| Proposed F12 kernel | Nonlinear token residual | Above; power ≥0.80 | 0 fitting, CPU; extraction unpriced | Powered upper C<0.05; otherwise unresolved |
| Optional label closure | More eligible supervision | New development manifest | ≈9.8 scoring; extras unpriced | Stop if eligibility/power fails |
| Activation/chain reference | Changed routes/state | Separately priced registration | Unpriced | No purchase without precision/power audit |
