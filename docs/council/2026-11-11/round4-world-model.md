1. Register external and family-transfer edits as one descriptive panel (F8), outside A5.5 and F5.5.
2. External edits are root edits on our exact base, so the base field is the exact anchor (no F6 problem).
3. The loop edits share one direction: one component, so they add no sign-test n beyond it.
4. Even 6/6 sign hits only just exclude chance (CP lower 0.541); 4/4 cannot (0.398).
5. For a rank-one edit −u vᵀ the field factorises exactly: a = −E[(uᵀδ)(vᵀx)], q = ½(uᵀGu)(vᵀAv).
6. Expected outcome for loop edits on WikiText: unresolved. A miss would bear on locality, not on the loop fix.
7. Add an outcome-blind check before scoring: buy scoring only for edits the field predicts are resolvable.
8. Register κ = q/|a| and a quadratic-sign forecast as descriptive: the field can predict its own failures.
9. Per GPU-hour the family panel beats the 4B diagnostic only if that check passes; otherwise skip it.
10. No change to Part A controls, H5 criterion or F5.5. Add a scripted-estimator arm to H5 and five related-work sentences.

# World-model — round 4, 11 November 2026

No experiment run. **A5** = `docs/PREREG_H1_TF.md` appendix A5. **Field** = the G2 base WikiText gradient
g plus K-FAC factors G, A (A5.8.1). F7's text is not on this branch. I use the donor counts in
`docs/prereg/a5_manifests/power/fields_manifest.json`.

**Sources read:** [2611.09382](https://arxiv.org/html/2611.09382), [2611.07610](https://arxiv.org/html/2611.07610),
[2611.10866](https://arxiv.org/html/2611.10866), [2611.06598](https://arxiv.org/html/2611.06598),
[2611.10974](https://arxiv.org/html/2611.10974), [2609.09054](https://arxiv.org/html/2609.09054),
[Concept Country](https://concept.country/research/weight-space-programming),
[aiXiv](https://aixiv.science/abs/aixiv.261103.000004) (abstract only), [Barry](https://nathan.rs/posts/weight-space-lp/),
[blog](https://rikiyat.github.io/blog/weight-space-programming/). *(extract)* marks facts from truncated HTML.

**Brief corrections.**
- **WSP-Scope.** The paper does not call its 0.8B edit loop removal. It names "long, unresolved reasoning" as the
  failure. Its u comes from token embeddings over two word sets, weighted by final-RMSNorm gains, not from
  activations *(extract)*.
- **Loop framing.** It belongs to the blog's Experiment 8 (85 of 96 practice answers ran to 32,768 tokens) and to
  Barry.
- **TFTV.** The edit is s̄(Σᵢ sign(μᵀvᵢ)σᵢvᵢ)ᵀ, rank one and norm-matched to W.
- **Deletion Tuning.** None of its backbones is Qwen3.5.

## Q1 — External edits as a sign test

**Valid as a base-anchored field test.**
- Every external edit is a root edit on Qwen3.5-0.8B, so g is the exact anchor and no cross term is needed. F6
  withdrew the cross term after a 0.336 audit error (`docs/reviews/council/board-a-f6-2026-11-11.md`).
- The edits were selected on MATH or IFEval, so the panel is outcome-blind on our endpoints.
- The A5.3 rotation transfers unchanged. On a single-column block it is a Haar-random direction in the same column,
  with the same norm. That is Concept Country's control.

**Staging check.** The checkpoint must differ from our pinned revision (`2fc0636…`) only in the claimed tensors, or the
panel fails closed.

**Decision value of n ≈ 4–8 is low as inference.**
- Two-sided CP lower bounds with all pairs agreeing: 4/4 0.398, 5/5 0.478, 6/6 0.541, 8/8 0.631. With one miss,
  7/8 gives 0.474.
- The loop edits are not independent. aiXiv's direction has cosine ≈ 0.78 to Astra's, and one of Barry's LP edits uses
  Astra's direction. Under A5.3.1 they are **one** component.
- WSP-Scope's layer-23 column edit is a second component only if u is released or fully determined by its recipe.
  We do not guess it.

**The edit's own mechanism.** For ΔW = −u vᵀ:
- a = −E[(uᵀδ)(vᵀx)] and q = ½(uᵀGu)(vᵀAv), with δ the output gradient and x the post-gating input.
- vᵀx is measurable outcome-free. The G2 probe cache holds `mlp.down_proj` inputs at all 24 layers
  (`docs/reviews/council/board-a-g2-2026-11-11.md`).

On WikiText prose a loop direction's neurons may be weakly active, so the likely result is **unresolved**, as in the 4B diagnostic (0 of 20 resolved improving,
`docs/reviews/scale/scale4b-diagnostic-2026-11-10.md`).

**What each outcome means:**

| outcome | what it says |
|---|---|
| hit | First-order WikiText response transfers to an LLM-designed, outcome-selected family. That is an H5-critic precondition, not Part A evidence. |
| miss | A locality or K-FAC failure. For sparsely firing units, E[(uᵀδ)²(vᵀx)²] ≠ E[(uᵀδ)²]E[(vᵀx)²]. Agreement at s = 0.25 with a miss at s = 1 marks a locality boundary. |
| unresolved | Says nothing. |

None speaks to the loop fix, which teacher-forced NLL cannot see, and none may enter A5.5 or F5.5.

**F8-E.** Pre-register sign(a) at s ∈ {0.25, 0.5, 1}, sign⟨g, D − R⟩, q and κ = q/|a|. Score everything on H0 TF-v1
(WikiText decides for the panel; MMLU descriptive) and report all entries.

## Q2 — Weights vs activations (Rane)

Rane's white-box arm is a 48-dimensional spectral summary. That is the same kind of readout as StatNN, which the
descriptor-only control nearly matches (0.51/0.61 vs 0.56/0.66, `docs/reviews/P4/d3v2f-summary-2026-11-08.md`).

Our field is different. ⟨g, D⟩ = E[δᵀDx] is a loss-weighted linear activation probe of the edit, which places it
between Rane's white-box and gray-box arms. The result therefore does not lower the field's expected effect, and I give
no numerical revision. Rane's task is domain identity on 36 runs, its white-box clustered CI (0.194–0.639) includes chance, and it is not response prediction.

**Yes:** state the prior in the access-ladder report (sentence 3 below).

## Q3 — New edit families: how the field should respond

**Sparse deletion, D = −W⊙M.**
- a = −Σ_M g_ij W_ij is the first-order Taylor pruning saliency.
- q = ½ tr(G(W⊙M)A(W⊙M)ᵀ) is a full quadratic form over the scattered mask, with no rank-one shortcut.
- Wanda's squared score, W_ij²‖X_j‖², is the input-side diagonal term W_ij²A_jj of q. So Wanda masks are chosen for
  low curvature energy.
- Deletion is bf16-exact: the effective delta is the intended one.
- **Prediction:** at equal count, q(Wanda) < q(random), and |ΔNLL| is smaller for Wanda. The q comparison is checkable
  on CPU before any scoring.
- A Wanda-initialised gate mask is not Deletion Tuning's learned mask. Training one is a separate, unpriced purchase.

**Rank-one activation-derived edits (TFTV, Concept-Country-style).**
- TFTV·μ = c·s̄, so vᵀAv is large by construction: the edit is an always-on residual bias. Expect κ ≥ 1.
- Under ΔL(s) ≈ s·a + s²·q with q ≥ 0 and a < 0, the sign flips at s* = |a|/q. The registered first-order sign is then
  expected to fail at s = 1 but hold at s = 0.25 when 1 < κ < 4.
- Register this **quadratic-sign forecast** as descriptive and panel-only. Part A keeps q unsigned (A5.8.1); the
  Fisher is a Gauss–Newton proxy, not a Hessian.

**bf16 caveat.** A small rank-one edit is not rank one after rounding: the 4B realised radius was 5.19e-3 against an
intended 5e-3. Ladder entries may round to no change, and a zero prediction counts as a disagreement (A5.3). F8 must
register a minimum surviving-entry fraction plus the A5.3.3 effective-rank audit before scoring.

**Per GPU-hour.** The 4B diagnostic cost $0.54 Lambda plus 0.2 local GPU-h (`docs/PROGRESS.md`, 11 Nov 03:00) and gave
an A3 fail and no predictor test. On H0 the panel is A3-qualified and every entry carries a prediction.
Six components (plus two extra loop members) × 4 entries is 32 entries, ≈ 0.64 GPU-h at the brief's 0.02 GPU-h per
entry. Edit construction is unmeasured.

It beats the 4B diagnostic **only if the edits resolve**. Gate: compute a, q and κ on CPU first. Score only if at least
four components have |a| + q above the floor, defined as the median paired-interval half-width from the registered
null-size payload (`docs/prereg/a5_manifests/nullsize/`).

Components:
1. Loop family (Astra diff and Barry's released LP edits).
2. WSP-Scope, if determinable.
3. TFTV on o_proj.
4. Concept-Country-style down_proj rank-one from a non-scored split.
5. Wanda gate deletion at < 0.05 %.
6. Random gate deletion matched on count.

SWEET is deferred (needs generated counterfactuals).

## Q4 — Receiver-relative information

Base anchoring already captures the receiver. g, G and A belong to the receiver; q is the edit's energy in the
receiver's curvature, and F5.2's whitened K blocks carry its direction.

The behavioural-rate statistic log det(I + αJ), J = E[gᵢgᵢᵀ] at θ₀, describes a dataset–receiver pair. For one base and
one objective it is constant across edits, so it has zero within-receiver variance as a per-edit feature. What we lack
is each edit's source-data empirical Fisher. That needs training-data access that chain and external edits lack, so it
would be a new arm. **Do not add it.**

Compression recovery is a **field** prediction (a spectral ladder, ⟨g, D_k − D⟩ + q), not a chart one. Our data make it
testable:
- For the 25 root LoRA donors, median |⟨g,D⟩| is 0.245 nats against 0.0026 for the rotation.
- All 25 have ⟨g,D⟩ < 0 (manifest `donors`).

So for LoRA, Part A's rotation contrast is essentially sign⟨g,D⟩, opposite to the constant-sign comparator (a reading
aid, not a change).

Truncation test: after the G3 ledger closes, non-pick LoRA rows only, k ∈ {1, r/2}, ≤ 0.5 GPU-h.

## Q5 — Controls and H5

**Part A controls: no change; no flaw found.** Concept Country matches norm and matrices with an orthogonal direction.
A5.3's Haar rotation also preserves spectrum, rank and support (A5.3.3). At 1,024 dimensions a Haar direction's cosine
to the original has standard deviation ≈ 1/32.

**H5: add a scripted-estimator arm.** Difference-of-means plus LP, no LLM, given the same localisation the editor gets.
A second variant picks its own layers by field ranking. aiXiv's 93 % recovery was conditional on Astra's layers, budget
and scale. For rank-one proposals the critic reads exactly a and q.

The ≥ 30 % savings criterion (A5.7, F5.4) concerns the critic and stays. Report savings per proposal stream (LLM,
scripted, random best-of-N). Editor-advantage claims must beat both baselines.

## Q6 — Line-stop rule and the paper

**F5.5 unchanged.** Add to F8: "External and family-transfer panels are descriptive; they cannot satisfy, fail or
unresolve any F5.5 predicate or A5.5 step."

Sentences for related work:
1. "Concurrent work edits the same base, Qwen3.5-0.8B, with column-sparse or rank-one changes that raise generation
   accuracy substantially (Hu et al. 2026, arXiv 2611.09382; Takehi et al. 2026). Scripted difference-of-means and
   linear-programming estimators recover most of that gain once layers and budget are given (aiXiv 261103.000004;
   Barry 2026). We predict teacher-forced response for a fixed zoo, not whether useful edits exist."
2. "Our finding that most zoo edits are behaviourally null describes the zoo's families and radii, not weight edits on
   this base."
3. "On the same base, spectral weight-delta statistics identified a LoRA's domain less well than activation or output
   features (Rane 2026, arXiv 2611.07610); our field contracts the delta with base gradients, an activation-weighted
   readout."
4. "Pre-registered norm-matched random-direction controls (Concept Country 2026) are the published analogue of our
   spectrum-matched rotation."
5. "Useful fine-tuning information is receiver-dependent (David et al., arXiv 2611.06598); our field is anchored to one
   receiver and makes no cross-base claim."

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---|---|
| F8-E external-edit panel | Exact base anchor; rank-one factorisation of a, q | Pre-scored sign(a), sign⟨g,D−R⟩, ladder, κ; loop edits = one component; checkpoint-diff check | inside F8 cap | < 4 components pass the resolvability gate → do not score; never enters A5.5/F5.5 |
| F8-T family-transfer panel | Deletion = Taylor saliency + mask q; steering rank-one has large vᵀAv | CPU check q(Wanda) < q(random); WikiText signs; quadratic-sign forecast; bf16 surviving-fraction rule | ≈ 0.64 scoring + construction; cap 2 | Same gate; fewer than half of predicted-resolvable entries resolve → stop family expansion |
| κ diagnostic | Linearity breaks where q ≳ \|a\| | Agreement for κ < 1 vs ≥ 1, panel only | 0 (CPU) | No difference between strata → drop the forecast |
| Spectral-truncation ladder | Response concentrated in leading directions | sign⟨g, D_k − D⟩, non-pick LoRA, post-G3 | ≤ 0.5 | Unresolved majority → stop |
| H5 scripted-estimator arm | Edit program = (direction, columns, scale) | DoM + LP at matched localisation, plus field-localised variant; savings per stream | in G5 cap | Editor advantage only if it beats scripted and best-of-N |
| Paper sentences | Scope honesty | Five sentences above | 0 | — |
