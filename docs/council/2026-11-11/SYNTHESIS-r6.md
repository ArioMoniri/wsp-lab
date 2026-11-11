# Council decision — 11 November 2026, round 6 (chair synthesis): is the field a single neuron, and is a model-specific world model needed?

Branch `council/2026-11-11`. Inputs: [BRIEF-r6](BRIEF-r6.md); [world-model](round6-world-model.md), [representation](round6-representation.md),
[metrology](round6-metrology.md), [editor](round6-editor.md); rounds 4–5 syntheses; appendix A5 with amendments F1–F11e. Part A is
staged, frozen and released with conditions but **unscored and held**; no outcome exists. This synthesis changes the registration
only through the dated amendment F12 below, registered before any G3 outcome is read. Thresholds named here are chair choices unless
a file is cited.

## Verdict on the user's thesis

**The thesis is upheld in its first half and converted into a registered test in its second.** (1) The registered primary arm is,
as a function class, a fixed second-order Taylor model of this base: a linear functional of the realised delta whose ~10⁸
coefficients are the model's own WikiText gradient, plus a block-diagonal Gauss–Newton magnitude, read out by at most three learned
coefficients (A5.8.3). "Single neuron" is exactly right about the readout and wrong about the information: the coefficients are
model-specific by construction. What that readout cannot represent is now written down (section 1): interactions between tensors
(it predicts zero interaction for any split edit, exactly), negative curvature, a per-edit gain, sign changes other than one
improve-then-harm flip along the ladder, path dependence, and anything on the generation side. It predicts effect sizes only inside
a trust region and direction beyond it; Part A tests direction only. (2) No outcome evidence about the field exists: the 33/33 and
25/25 agreements compare two computations of the same prediction, and the 0.8722 power assumes the registered 0.75 alternative.
(3) The programme already contains model-specific learned predictors in the user's sense (Part B's augmented StatNN and the
K-FAC-anchored GL-set head, trained on this model's outcomes and tested on held-out source components), but **no registered
contrast ever sets a learned model-specific predictor against the three-coefficient field head to head**. That is the gap the user
has identified, and it is the comparison the thesis actually requires. Amendment F12 registers it now, outcome-blind, together
with the one experiment in which the field's answer is exactly zero and any resolved signal is direct evidence for the thesis: an
additivity panel of split edits. The council does not recommend a JEPAWG-style latent dynamics model: one was built (M3, ρ
0.15/0.13 on 25 evaluated successor pairs), the weight transition W + D is known exactly in editing, and the unknown is the
response, not the state.

## 1. Decisions by question

| Question | Decision | Dissent and handling |
|---|---|---|
| Q1 Is the field a single neuron? | **Readout yes, information no.** ŷ = β₀ + β₁a + β₂q with a = Σ_t⟨g_t, D_t⟩, q = Σ_t ½tr(G_t D_t A_t D_tᵀ) ≥ 0 (`src/wsp/features/response_field.py`). Cannot represent: cross-tensor interactions (additive by construction), negative curvature (q ≥ 0, at most one sign flip at s* = \|a\|/q), third-order effects and termination, a per-edit gain, path dependence (F6 withdrew the chain cross term). Self-assessment on the frozen predictions: κ median 0.023 at s = 1, s* < 1 for 1 of 46; predicted \|a\| median 0.228 nats (LoRA) and 0.00054 nats (chain), the latter below the 0.00487-nat panel floor. First order predicts effects where the remainder is small and direction beyond; Part A tests sign only (all four). | None. Editor adds that for a fixed-prompt differentiable loss first order predicts magnitude to leading order, so the trust region is the real question; adopted. |
| What rounds 4–5 established | Computational determinism of the prediction pipeline and simulated power under an assumed data-generating process. Nothing about outcomes. | None. |
| Q2 What is a model-specific world model here | f_θ(state, edit) trained on this base's observed edit–outcome pairs and tested on unseen source components. (a) Part B's augmented StatNN and K+F head qualify as restricted model-specific surrogates (static regressors, not rollouts); no contrast registers them against field-3. (b) JEPAWG-style latent dynamics: already built as M3 (`src/wsp/predictor/models/__init__.py`, 16-d latent, VICReg; ρ 0.15/0.13, 25 evaluated pairs, `d3v2f-summary`); 457 chain rows outside Part A in only 73 components; the transition is known exactly; **not recommended**. (c) A learned residual readout over per-tensor field tokens, chart coordinates and cached probe D·x summaries, nested in the field so that a win belongs to the learned part: **F12**. (d) Activation-assisted (forward passes after the edit): the upper-cost reference; runs the edited model; cost per candidate unmeasured; descriptive under A5.4.3 and the H5 ladder. | Representation prefers a regularised kernel residual over the same frozen tokens; editor a width-8 residual MLP (8d + 17 parameters); world-model the F5-pinned K+F head (1,137 parameters). Chair: F12's candidate is the F5-pinned K+F head plus probe summaries (already pinned, smallest parameter count, zero at zero edit); the kernel and MLP variants are registered as sensitivity arms under the same folds. |
| Q3 Why learned encoders lost on 342 rows | Too few independent source components (278 clusters), descriptors carrying most of the signal (descriptor-only 0.51/0.61 vs StatNN 0.56/0.66), re-instantiation drift of neural ρ (median 0.09) as large as the gap, and M3 seeing 25 evaluated pairs; not architecture. What changes it: inputs that hand the learner an analytic prediction to correct (field tokens) and more independent root components. Chain-step targets sit below the noise floor and would mostly teach noise; roots do not. Sample size: K ≈ 297σ² components at C = 0.15, with σ (between-component SD of the relative loss difference) unknown before outcomes, so an A5.4.2 simulation decides, not a guess (metrology). | Brief corrections adopted: TF-v1 primaries exist for 689 entries in 126 components; the chain has 675 entries in 93 source components; most of the 565 unscored entries are sealed confirmatory or fresh and scoring them would spend G7 freshness; only 147 development root entries can be scored safely (≈ 2.55 GPU-h, not 9.8). |
| Q4 Model-specific by construction? | Anchoring makes the field this model's local world model with zero outcome fitting; the thesis's strong reading is that outcome supervision adds information beyond local geometry. The programme touches that only indirectly (C_abl, C_rand, C_perm, K+F:F). Missing: a held-out learned-vs-field-3 head-to-head; a design that nests the field; fresh rows; leave-one-family-out transfer. F12 supplies all four. | None. |
| Q5 F12, registrable now | **Registered (section 3).** Roots only (the field is exact there): 123 existing TF-v1 roots outside the 52 Part A components plus the 147 purchasable development roots = 270 rows in 234 source components; five outer component folds hash-sorted under `20261111:f12:`, stratified by family, inner folds and seeds as in the F5 pins; endpoint TF-v1 WikiText mean-NLL Δy under `configs/eval_tf_v1.yaml`; baseline F3 = the A5.8.3 readout fitted on outer-training rows; candidate WM-R = F3 + h_θ with h the F5-pinned K+F head over 39 columns plus probe D·x summaries, L1 on F3's training residuals, zero at zero edit; contrast C_WM = 1 − A_WM/A_F3 and C_WM,S = 1 − A_WM/A_StatNN(same inputs), both with A5.4.2 conditional and joint bootstrap-t intervals (4,000 draws, seed 20261107); decision wording: "correction detected" if C ≥ 0.05 with both lower bounds > 0 against **both** comparators; "none detectable" if both upper bounds < 0.05 at simulated power ≥ 0.80; otherwise unresolved; descriptive alongside: zero-fit E₀, leave-one-family-out C_WM, kernel and MLP sensitivity arms. Outcome-free power simulation (Δy = a + γh*(x) + ε on real tokens, ε from the qualification noise, γ for oracle C = 0.15, γ = 0 null; gates size ≤ 0.07, power ≥ 0.80) **before any purchase**. Cost ≈ 2.55 GPU-h scoring (cap 3), fields on CPU ≈ 7 CPU-h. Reads nothing from Part A; scored after the Part A ledger is sealed; no A5.5, F5.5 or funding authority. | None. Metrology requires the smallest valid holdout to come from the simulation, not a guessed N; adopted. |
| F12-I additivity panel | **Registered.** 16 development lora_sft roots each split into early-layer and late-layer halves, plus the Astra edit split by tensor and the Concept-Country-style construction in halves: 36 entries ≈ 0.62 GPU-h. Statistic I = Δy(full) − Δy(part₁) − Δy(part₂) against the F8a floor; the field predicts I = 0 exactly; a resolved I is direct evidence for the thesis. Descriptive; no decision authority. | None. |
| Q6 The editing side | Edit mechanisms change the distribution and scale of D, not the definition of a. Most embarrassing for a first-order predictor: learned coordinated gate deletions at matched count and norm (multiplicative, interacting; F8 funds only untrained masks and excludes learned ones), multi-site edits (Sparks two layers, Concept Country nine; F12-I), feedback-driven multi-step programs and RL policies (no trajectory state in a base-anchored functional; outside scope), and anti-loop edits on the generation side (F9.6 endpoint). WeightLoop computes exact output changes from cached features for finite edits, which is the least embarrassing case and qualifies the brief's "no forecast" summary. | Editor's priority (learned gate masks at matched count/norm, source-held-out) is recorded as the first candidate for a later amendment once priced; not funded by F12. |
| Brief corrections | 689 TF-v1 records (not 694); chain 675 entries / 93 components (618 is the census count, 99 the lineages with ≥ 4 nodes); 147 scorable development roots at ≈ 2.55 GPU-h; the brief's "JEPA-style 0.42/0.56" was m1_matched, M3 scored 0.15/0.13; arXiv 2605.06298 is a video world model with no weight edits; WeightLoop forecasts exactly from cached features. | — |

## 2. Spend (H0; stop-and-reprice; nothing waits on F12; Part A unchanged)

| Item | GPU-h | Status |
|---|---|---|
| F12 power simulation | CPU | before purchase |
| F12 scoring of the 147 development roots | ≈ 2.55 (cap 3) | after the Part A ledger is sealed and the simulation passes |
| F12 field tokens for 270 roots | ≈ 7 CPU-h | with the simulation |
| F12-I additivity panel (36 entries) | ≈ 0.62 | after F8/F9 panels |

## 3. Amendment F12 (appended to A5 today, before any G3 outcome)

F12.1 purpose and thesis; F12.2 population (270 roots / 234 components; exclusions; folds; endpoint); F12.3 models (F3 baseline; WM-R
candidate; augmented StatNN comparator on the same inputs; kernel and MLP sensitivity arms); F12.4 contrasts and decision wording
(both comparators; A5.4.2 intervals; no A5.5/F5.5 authority); F12.5 outcome-free power simulation and purchase gate; F12.6 F12-I
additivity panel; F12.7 ordering (register now; score after the Part A ledger is sealed; forecasts logged under F9.1 before scoring);
F12.8 cost and caps; F12.9 what does not change; F12.10 brief corrections recorded. Text in `docs/PREREG_H1_TF.md`.

## 4. Sentences for the measurement paper

1. "Our primary predictor is a fixed second-order model of one base: a linear functional whose coefficients are that model's own
   gradient, a Gauss–Newton magnitude, and at most three fitted coefficients. It is model-specific in its information and
   single-readout in its function class; it cannot represent interactions between tensors, negative curvature or path dependence."
2. "No registered contrast in the original design set a learned model-specific predictor against that readout; amendment F12 adds
   it, outcome-blind, on held-out source components, together with an additivity panel in which the first-order prediction is exactly
   zero."
3. "A latent weight-dynamics model was trained on this zoo and did not predict held-out responses (ρ 0.15 / 0.13 on 25 evaluated
   successor pairs); in weight editing the weight transition is known exactly and the unknown is the response."

## 5. What this changes in the record

`docs/PREREG_H1_TF.md`: F12 appended. `docs/HYPOTHESIS.md`: H1-OR gains F12 and F12-I as registered secondary tests of the
single-readout limit. `docs/RESEARCH_NOTES.md`: arXiv 2606.13461 and 2611.11604 with their takeaways and the M3 correction. Board A
reviews F12 before its simulation is run; Part A remains held by the user and is unaffected.
