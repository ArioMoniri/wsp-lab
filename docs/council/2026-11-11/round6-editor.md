1. A small readout can predict local response; it cannot generally represent finite-edit behavior.
2. Exact delta contraction is not an exact endpoint forecast (A5 F9a.5).
3. Editors need proposal rules and verification; these papers do not establish a learned world-model requirement.
4. WeightLoop actually computes exact output changes from cached features; its ranking objective remains nonlinear.
5. Learned gate deletions are my strongest in-domain challenge to a first-order predictor.
6. F8 includes untrained masks, excludes learned masks, and defers SWEET (A5 F8.2).
7. Activation-assisted prediction pays candidate forwards before it can save verification.
8. H5 permits metered probes, but only measured support improvement permits acceptance (A5.7/F5.4).
9. Register F12 now, outcome-blind: a small nonlinear response head, then a separately powered critic comparison.
10. Preserve Part A/Part B ordering, F5.5 and H5’s complete success contract.

# Editor — round 6, 11 November 2026

Here **A5** means `docs/PREREG_H1_TF.md`; proposals below are not amendments or execution authority. Read `SYNTHESIS-r5.md` and `round5-editor.md` subject to the later F9a corrections reproduced in `docs/H5_DESIGN.md` and `docs/H5_PROTOCOL.md`.

## Q1: what the field represents

The zero-fit arm contracts the realised delta: a=Σ⟨g_t,D_t⟩, q=½Σtr(G_tD_tA_tD_tᵀ). The online readout has at most intercept, a and q coefficients (A5.8.3). Calling this a neuron hides its expensive, model-specific feature construction, but correctly notices its limited output function class. Linear response sums tensors; blockwise q includes within-tensor interactions, not general cross-layer curvature. Fisher magnitude is not a signed Hessian. Along sD, a scales as s and q as s²: the linear term cannot reverse sign for positive s; the quadratic can represent only restricted reversals. Neither captures arbitrary trajectory dependence or receiver drift.

For differentiable fixed-prompt loss, first-order response predicts magnitude to leading order, not merely direction; sign requires a nonzero leading term and sufficiently small remainder. Finite-scale accuracy and stopping behavior have no such guarantee. The 33/33 agreement is agreement between forecasts and `fields.json`, not scored outcomes (`docs/PROGRESS.md`, 11 Nov staging); Part A remains unscored (`BRIEF-r6.md`). It establishes implementation consistency, not universality.

## Q6: editing algebra versus response complexity

In the following, locality concerns the **endpoint**, not how few weights change. Every smooth endpoint has a local Taylor expansion; none of these edit recipes guarantees its finite step lies in that regime.

| Mechanism and primary source | Weight change and first-order limits | F8/F9 coverage (A5) |
|---|---|---|
| [Sparks, experiment 8](https://rikiyat.github.io/blog/weight-space-programming/) | Subtracts a loop direction from 64 down-projection columns **in each** of two layers. A shared-direction selected-column update has rank at most one per tensor; exact coefficients require released bytes. Sparse support does not bound autoregressive consequences: changing termination can change the whole continuation. Practice forwards guide search. | F8-E exact patch; generation separately F9.6. |
| [Exploring WSP](https://arxiv.org/html/2611.09382v1) | Primary math edit D=−u e₅₀₇ᵀ; dense rank-one reruns also occur. Here u comes from token embedding/readout contrasts, not residual activation differences. Fixed-input module output changes linearly; downstream reasoning need not. Development generation/teacher-forcing selects candidates. | F8-E conditional on direction availability; not a blanket registration of every intervention. |
| [aiXiv replication](https://aixiv.science/abs/aixiv.261103.000004) | Difference-of-means direction and selected columns reproduce much of the benefit **conditional on supplied layers, budget and scale**. Same low-rank algebra; selection success is not an endpoint forecast. | F8-E reconstruction; LP only with public bytes/complete recipe. |
| [SWEET](https://arxiv.org/html/2611.10974v1) | Selected neuron columns/head blocks receive D=−ddᵀW. Directional output suppression is exact for fixed module input. Original/perturbed generated responses provide directions and attribution; multi-layer upstream changes invalidate a fixed-input interpretation globally. | Explicitly deferred pending counterfactual-generation pricing, F8.2. |
| [TFTV](https://arxiv.org/html/2609.09054v1) | D=αs̄vᵀ, with v from weight SVD and signs of expected-input projections. Norm-matched rank-one arithmetic and mean-input steering do not imply linear behavioral response; independently estimated trait inputs weaken even the composition identity. | F8-F o_proj rank-one, rotation and ladder. |
| [Deletion Tuning](https://arxiv.org/html/2611.10866v1) | Frozen gate weights acquire trained soft-mask logits, deployed as binary keep/delete bits: D=−W⊙M_delete. SFT/GRPO/self-distillation optimize masks. Each deletion is a full coordinate change; the gate’s SiLU output multiplies the up branch. Small deleted fraction is not small functional displacement. | F8-F Wanda/random masks only; **learned masks excluded**. F9-P adds substantial pruning and compensation, not learned deletion training. |
| [WeightLoop](https://arxiv.org/html/2611.07319v1) | Fixed program attenuates selected output weights over a frozen encoder. Cached h gives exact Δprediction=s_y Δwᵀh, even for finite edits. Ranking/logistic loss is nonlinear. Support training/validation chooses updates; rebuilding starts from base. This is the least embarrassing family for linear output prediction. | F5.4 imports validation/rebuild discipline; its biological predictor is not an F8/F9 edit family. |
| [RL neural editing](https://arxiv.org/html/2606.13461v1) | MaskWorld multiplies weights; ShiftWorld adds updates, optionally factorized. Policy observes weights and previous reward, permits repeated actions, and receives task/utility rewards. Sequential receiver changes defeat a fixed-base local forecast; a PPO value estimate is not an explicit endpoint dynamics model. | LoRA algebra overlaps existing arms, but this RL proposal process is not registered by F8/F9. |
| [Policy programming/model checking](https://arxiv.org/html/2611.11604v1) | LLM writes entire small-network weights, optionally repairs against optimal actions, then uses Storm feedback. All-layer rewrites and discrete action switches are not local edits. Verification uses an explicit environment MDP, not a learned weight-response surrogate. | Outside these pinned-LLM panels. |

F8.2 also registers a Concept Country-style residual-contrast down-projection rank-one construction: D=uvᵀ is linear at fixed input, not through the remaining network. This describes our registered design, not independent verification of that paper. Concept Country and Barry’s LP pages were inaccessible in this fetch; I make no quantitative claims from them. LP selection changes chosen columns/scales, not the eventual delta interface.

**My priority:** learned gate masks at matched count/norm, with source-held-out evaluation. Their coordinated multiplicative changes offer a clean test of interactions missing from a and blockwise q. Whole-policy synthesis is more obviously nonlocal but answers a different task. Anti-loop edits most embarrass a claim about **generation** from a teacher-forced field; that is endpoint mismatch, not evidence that a nonlinear TF predictor wins. F9-P already probes substantial nonlocal deletions, so run its registered diagnostic before buying learned-mask training (A5 F8.2/F9.3/F9.6).

## Q2–Q3: candidates, information and cost

A model-specific response model is fθ(state, realised edit) trained on this receiver’s outcomes. Anchoring alone gives model specificity; outcome learning supplies the stronger thesis. Candidate priors below are judgments, not measured effect sizes.

Available inventory is 1,259 zoo entries, 694 TF records, and 618 chain entries in 99 lineages (`BRIEF-r6.md`). None is automatically an eligible independent training row; registration uses inherited G1 folds and source components (A5.4).

- **(a) Augmented StatNN / GL-set:** pooled statistics, per-tensor field/chart inputs and anchored geometry; tree complexity follows A5.8.4, compact GL head about 1.6k parameters, with frame/fusion counts additional (F5.1–2). These are model-specific response surrogates, though not rollout world models. Use eligible labelled G1 rows. Prior: modest residual gain over field plausible; GL beating same-information StatNN uncertain. Cached fitting can use CPU; new feature extraction is extra.
- **(b) Latent dynamics:** encode predecessor state, condition on realised increment, predict successor latent and endpoint. Proposed one-hidden-layer transition width h on state/edit widths d_z,d_e has h(d_z+d_e+1)+d_z(h+1) parameters, **plus encoder/readout**. Paired labelled transitions must be audited; 618 snapshots are not 618 independent transitions. Prior: weakest near-term gain over either comparator; more complexity and state-feature cost. F9.7 does not register it.
- **(c) Richer readout:** field tokens, chart and cached probe summaries into the small head specified below; eligible shared-label rows, nested component folds. Prior: interactions may beat field locally outside its linear regime; gain over StatNN remains uncertain. CPU fitting avoids a GPU purchase when inputs are cached.
- **(d) Activation-assisted upper-cost reference:** apply each immutable candidate, run frozen probe prompts, summarize hidden/output changes and predict disjoint endpoint change. Use the same head/folds as (c), adding probe dimensions to its parameter count; eligible rows require candidate-specific probes, whose count is not established. Prior: strongest informational opportunity over field and weight-only StatNN, **not** guaranteed performance dominance.

Per candidate, (d) costs load/apply + probe batches × measured forward time + activation extraction/transfer + head inference; incumbent probes are cached only under identical hashes. Amortized feature-training costs and audits are additional. No measured probe GPU-h rate exists in the cited H5 design; TF scoring’s 1.04 minutes/entry is not a probe timing (`docs/RESULTS.md`; `docs/H5_PROTOCOL.md`). H5 allows separately hashed metered probe hooks, pinned forward/hidden-state access, no transcript/rationale inputs, and no sealed feedback; predictions only schedule/abstain. Acceptance still requires held-out verification and, if rebuilt bytes change, an untouched acceptance slice (A5.7/F5.4). Post-edit probing saves neither construction nor already-paid forwards.

The fast-tier study has 342 rows/278 ancestry clusters and strong descriptor baselines; representation instability and weak family transfer support a variance/shortcut explanation, not proof that neurons cannot work (`docs/reviews/P4/d3v2f-summary-2026-11-08.md`). Closing the reported label gap costs approximately (1,259−694)×1.04/60=9.8 GPU-h, excluding overhead (`BRIEF-r6.md`; `docs/RESULTS.md`). Better inputs may matter more than correlated rows. The 0.00487-nat panel half-width is not independent per-row noise; use shared-item covariance in power simulations (`docs/reviews/P5/tf-v1-requalification-2026-11-06.md`). Transition differences add predecessor uncertainty.

## Q4–Q5: decisive critic test

The papers’ loops above require proposal construction plus evidence-based selection; they do not prove a separately learned model-specific world model necessary. WeightLoop’s exact cached-feature calculation directly qualifies the brief’s “no forecast” summary. H5 tests whether forecasting makes verification economical, separately from editor ability (`docs/H5_RESEARCH.md`; A5 F8.7).

**Proposed F12:** register now without Part A outcomes; execute after existing Part A/Part B obligations. Fit a residual MLP on (c)’s frozen d inputs, proposed width 8, giving 8d+17 parameters per endpoint; return f(D)−f(0). Reuse F5.1 nested four-fold component training, fixed seeds and training-only nuisance. Use only eligible G1 labels, inherited outer folds, connected donor/direction/lineage groups; leave whole edit families out descriptively. No F8/F9 outcome trains its own prediction.

Require C=1−A_F12/A_reference lower bounds >0.05 against **both** field-only and same-information augmented StatNN, conditional and joint A5.4.2 intervals (proposed meaningful margin). Match inputs/tuning; include field-absent/permuted controls. Freeze development-only component/common-item simulations at C=0.15, null-size checks and ≥0.80 power before scoring; insufficient power stays unresolved. A powered upper bound <0.05 against both rejects this arm’s incremental benefit, not all world models.

For the critic, offline TF improvement is insufficient: freeze the learned gate and compare identical proposal streams prospectively under A5.7—≥30% savings lower bound, score noninferiority, missed-proposal bound, safety and positive fully charged net GPU savings. F9a shadow LoRA remains collateral-only and cannot certify these. F12 changes neither F5.5 nor prior funding predicates.

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---|---|
| F12 response head | Nonlinear state/edit residual | Proposed paired tests above | 0 GPU fitting on cached inputs; extraction unpriced | Powered upper bounds <0.05 against both |
| Activation reference | Candidate probe responses | Access ladder; proposed matched F12 ablation | Meter pilot before purchase | No affordable net saving; predictive uncertainty unresolved |
| Learned deletion challenge | Optimized binary gate mask | Not registered; F8 excludes it | Unpriced | Coverage/pricing failure blocks purchase |
| Existing pruning diagnostic | Deletion/compensation | F9-P, descriptive | ≈0.78 scoring + ≤0.5 construction (SYNTHESIS-r5.md) | Gate/qualification failure; no F5.5 authority |
| Live learned critic | Schedule verified edits | A5.7, after prerequisites | G5 ≤10+≤6; G8 unpriced (H5_PROTOCOL.md) | Powered savings upper bound <30% or score upper bound <−1 point |
