# Round 2: representation (7 Nov 2026)

Thresholds are prospective design choices; data numbers cite repo files.

## 1. Concessions

1. **WikiText is not task readability** (`round1-metrology.md` §2, "Exact label and cost"). A WikiText pass supports only "the field reads distribution sensitivity" (H1w estimand, `docs/PREREG_H1_TF.md` A4.2); MMLU arms run alongside as descriptive.
2. **Evaluate at the predecessor, not the base** (`round1-world-model.md`, "Primary"). For chains, my K-FAC cross term is only a first-order patch; I add an exact-gradient audit at 20 predecessors.
3. **The orientation control must act on outcomes, not just features** (`round1-world-model.md`, "Smallest decisive purchase"). A scored, spectrum-matched rotation of the edit beats my feature-only random field. Adopted as Part A.
4. **Consume Δ_eff = bf16(W+Δ) − W** (`round1-editor.md`, "What an editor proposal is"), and log rounding loss.
5. **Two corrections to my own round 1.**
   - P6d rescoring yields a **167-stratum-row** panel, not 229; the other 62 chain rows need replay plus ≈1.05 GPU-h scoring (`docs/reviews/P5/board-a-incident-d8-2026-11-06.md`, P6d and Totals).
   - A 5% gain is a **screen, not a falsifier**: at feature-contrast SD ≈ 0.87 it needs about 2,355 clusters (`docs/council/2026-11-05/SYNTHESIS.md` l.50).

## 2. Rebuttals

1. **Sequencing against metrology's MMLU-only endpoint.** MMLU is the task claim, but on D3 it is unqualified: max h 0.1137 > 0.10 and P 0.2264 > 0.20 (`docs/PREREG_H1_TF.md` A4.1), so any test on it is "untested" by construction. WikiText is the only qualified D3 label: max h 0.0128 vs δ 0.0488, reliability lower bound 0.9923 (`docs/reviews/P5/h1w-gate-2026-11-06.md` §2). A structure that fails it should die before MMLU qualification is bought.
2. **Against buying 720 rows first** (`round1-world-model.md`, 23.2 GPU-h nominal). Their own arithmetic needs about 264 blocks at SD 0.87, not 100 (SYNTHESIS l.50: about 262 for 15%). The rotation contrast runs on 40 existing D3 edits for about 2 GPU-h; large purchases should follow it.
3. **Hashed projectors repeat M1's frame problem.** The world model's b_ab = ⟨D, P_a W P_b⟩ uses fixed hashed projectors; M1's random JL frame (`src/wsp/features/svd.py::_proj`) did not beat the tree (`docs/reviews/P4/d3v2r-summary-2026-11-06.md`). W-anchoring says where weights are, not which directions the loss cares about, so relevance is still learned from ~300 labels; the field's costs none.
4. **The editor's ΔW·x critic is half of the field** (`round1-editor.md`, "Primary"). For a linear layer, ⟨g, ΔW⟩ = Σ_t δ_tᵀ ΔW x_t, where δ_t is the back-propagated output gradient. Their cache supplies x_t; they concede a logit lens "is not the downstream Jacobian". Backprop supplies it, for attention and GatedDeltaNet readers too.

## 3. Revised single proposal

**Structure.** A base-anchored sensitivity field, frozen before any label is seen.
- Gradients g_k for WikiText, MMLU gold-letter, HellaSwag and refusal, each on a panel disjoint from every scoring panel (`docs/PREREG_H1_TF.md` §2 and A1 table).
- K-FAC factors per tensor.
- Per-tensor tokens [⟨g_k, ΔW_t⟩, q_t, the chain cross term c_t, norms, layer, role] on Δ_eff.
- ⟨g, ΔW⟩ is basis-free: immune to degenerate subspaces and SVD sign flips.

Readouts: (a) zero-shot; (b) fold-fitted linear calibration, ≤3 coefficients per endpoint; (c) StatNN on [v4 ‖ field].

**Mechanism against StatNN and descriptors.** Family, norm, spectrum and support carry most of the tree's rank (`docs/council/2026-11-05/evidence-statnn-importances.md`). All four are fixed under a spectrum-matched rotation inside the support (residual v4 differences logged). The field is not, so it predicts the **sign** of an effect at matched norm.

**Part A: rotation and ladder (new, decisive for mechanism).**
- *Edits.* 30 lora_sft (LoRA 56/60 and chain 30/50 have KL > 0.01, `docs/council/2026-11-05/metrology_budget_output.txt` l.249) and 10 chain D3 edits. Picked by sorted ID within norm tertiles, before outcomes.
- *Variants.* Per edit: one rotation R = (Q_out U) Σ (Q_in V)ᵀ inside the support, at s = 1; plus the edit D itself at s ∈ {0.25, 0.5}.
- *Volume.* 40 + 80 = 120 TF-v1 evaluations. At the 1.003 min/entry warm median (`docs/RESULTS.md` l.329) that is ≈2.0 GPU-h.
- *Pass.* On pairs whose measured WikiText difference |y(D) − y(R)| exceeds the pair's item half-width, sign agreement between the predicted ⟨g, D−R⟩ + q(D) − q(R) and the measured difference must have a lower 95% binomial bound > 0.5. Fewer than 15 such pairs: **inconclusive**.
- *Ladder.* Sign agreement ≥ 0.7 at s = 0.25.

**Part B: D3 retrospective.**
- *Panel.* P6d recovered panel (167 stratum rows + 157 null-family controls), or 229 + 157 if the replay lands.
- *Machinery.* Registered H1w folds, nuisance and six intervals.
- *Arms.* StatNN; descriptor-only; delta-norm; field (a), (b), (c); norm-matched random field; layer-permuted field; the world model's W-contraction b_ab; A1-lite (TF-v1 KL vector).
- *Pass to buy fresh sealed rows.* θ(c) − θ(StatNN) ≥ 0.15 with paired lower 95% > 0, field arms beat both field controls, **and** Part A passes.
- *Screen only.* ≥5% residual-MAE gain with lower bound ≤ 0 (PREREG §7 screen) funds only a Part A extension.
- *Power caveat.* On 229 rows power lower bounds at θ = 0.15 are 0.73 conditional, 0.60 joint (`h1w-gate` §4); 167 rows will be lower; re-simulate before reading outcomes.

**Predecessor audit.** Exact g at 20 chain predecessors compared with g(θ₀) + cross term. Unmeasured; capped at 1 GPU-h.

**Cost.** P6d 5.5 (already priced) + optional replay 1.05 + fields ≤2 (cap) + Part A 2.0 + audit ≤1 = **≤11.6 GPU-h**, of which ≤5 is new.

**Kill** (any one):
1. Ladder sign agreement < 0.7 at s = 0.25 (Taylor structure wrong even locally).
2. Part A has ≥ 15 resolved pairs and its sign-agreement upper 95% bound is ≤ 0.6.
3. Part B gain upper 95% < 0.05.
4. Field (c) does not beat the random field (paired point ≤ 0).

Holding at s = 0.25 but failing at s ≥ 1: right structure, edits too large; move to probe-response (A1).

**H5.** Unchanged: beat a same-support projected-gradient step and equal-budget best-of-N; rationale only as a blinded arm (all members agree).

## 4. Field vs chart; compatibility with the metrology endpoint

**Field vs chart: complementary**, the same object at different orders.
- The world model's chart is z(W+D) = z(W) + A(z(W), D) with A(z, 0) = 0. My field is the analytic Taylor expansion of A: first order J(W)·D and a K-FAC quadratic, evaluated at W = θ₀. My chain cross term c_t is the first-order correction g(W_prev) ≈ g(θ₀) + H(W_prev − θ₀). The predecessor-aligned chart is this field re-expanded at each predecessor plus a learned nonlinear remainder.
- The field supplies label-free coordinates and a measured validity radius (the ladder); the chart learns the remainder from labels and should take field tokens as inputs instead of hashed projectors.
- The one real disagreement is the anchor: their ⟨D, PWP⟩ is weight-anchored, mine loss-anchored. Part B runs both on identical rows and folds.

**Metrology endpoint.** Predecessor-relative normalized MMLU gold-letter log-prob on 1,000 disjoint items is **compatible** with a WikiText-first test.
- Same pipeline, folds and nuisance; only y changes (PREREG A4.3).
- g_MMLU is computed on MMLU items disjoint from both the 1,000 development items and the sealed 1,000, frozen before any requalification.
- MMLU arms come free from the same P6d records and stay descriptive until MMLU qualifies.

WikiText decides whether the structure lives; MMLU whether it reads tasks. Passing only WikiText means sensitivity, not task readability.
