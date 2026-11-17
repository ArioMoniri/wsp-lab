# Round 10, slope seat (17 Nov 2026): the end-of-step slope at R0/R1

No GPU, downloads, `data/` access, sealed splits or PREREG edits. **P2** = `docs/reviews/wme/wme-pilot2-2026-11-16.md`; **P2.json** = its JSON. *Computed* = my arithmetic on `P2.json` `R2.rows` (h ≥ 2, 282 pairs). Pilot-2 numbers are used for **diagnosis only**. Every test runs on fresh `wme-p3` steps.

## BLUF

1. **The 35% is one objective.** Harmful refusal gives 92% of the pooled error reduction. Without it, the oracle cut is **0.047** (*computed*).
2. **For margins and sycophancy, A0 has no curvature at all:** `SIGN_Q` = 0 for them (`scripts/wme_pilot2.py`:66–73).
3. **For harmful refusal, in-step curvature ½(R−L) gives 0.56 of the 0.79 cut.** The cross-derivative gives 0.23 (*computed*).
4. **Cheapest R1 quantity:** the per-item second directional derivative along D_t at θ₀, from a nested forward JVP (S2).
5. **For sycophancy the slope is noise.** The end-slope term has corr −0.01 with the observed step. Transport's cut is −0.44 on unresolvable pairs and 0.00 on resolvable ones (*computed*).
6. **GSM8K and MMLU-Pro have nothing to recover.** Static error is 1.16 and 1.13 SE, against a noise floor of 0.80. The oracle cut is ≤ 0.04.
7. **Harmless refusal is miscalibration:** every predictor, the oracle included, under-predicts by an OLS factor of 1.4–1.6.
8. **Keep transport only for harmful refusal** (WikiText behind a gate), using the exact objective Hessian. Everywhere else, use static plus a learned outcome residual.
9. **The fresh tests cost ≈ 4.3 GPU-h, about $21 on H100.** Each proposal has its own kill criterion. If S1 and S2 both fail, transport is dropped for good (C4).
10. **Go-R is reported per objective and on resolvable pairs only** (r9 b3). It is never pooled alone.

## 1. Diagnosis

**Definitions.**
- L = ⟨g(θ_{t−1}), D_t⟩ and R = ⟨g(θ_t), D_t⟩. Both are R2 gradients on 24 construction items. Outcomes are scored on 64 validation items (`ph_r2`, `FIELD_ITEMS`).
- L − A0 is the cross-derivative E_{t−1}ᵀH̄D_t.
- R − L ≈ D_tᵀHD_t is the in-step curvature.

Cut vs static at h ≥ 2 (*computed*):

| Objective | cross (L) | static + ½(R−L) | trapezoid | static err/SE | ceiling* |
|---|---|---|---|---|---|
| harmful refusal | +0.23 | **+0.56** | **+0.79** | 5.69 | 0.86 |
| harmless refusal | +0.07 | +0.05 | +0.13 | 3.22 | 0.75 |
| WikiText | −0.31 | −0.28† | +0.14 | 1.67 | 0.52 |
| GSM8K | −0.32 | +0.02† | +0.04 | 1.16 | 0.31 |
| MMLU-Pro | −0.20 | +0.19† | +0.02 | 1.13 | 0.29 |
| sycophancy | −0.12 | −0.24 | −0.26 | 1.28 | 0.38 |
| pooled | +0.02 | +0.20 | +0.35 | 2.36 | — |

\* Ceiling = 1 − E|Z| / static error, with E|Z| = 0.80: the cut if only observation noise remained.

† Double-counts curvature, because the static field already holds K-FAC ½q for these objectives.

**Harmful refusal margin.**
- Steering steps move it 12–24 SE, and the first-order static step misses by 8–12 SE.
- The OLS ratio obs/static is 1.75, against 0.99 for the trapezoid.
- Corr(R−L, obs−L) = 0.99, so the slope change is large and deterministic, which exact curvature can capture.
- Steering strength theory predicts "non-monotonic effects" (https://arxiv.org/abs/2602.02712).

**Sycophancy.** The objective is a difference of two log-probabilities (P2 §1), so its gradient is the difference of two overlapping gradients.
- R has corr −0.01 with the observed step (OLS 0.06).
- The refitted F3u linear coefficient is −0.055 (P2.json `T2.F3u_coef`).
- `steer_syc15` steps are the worst case (0.71 → 2.39 SE).
- My hypothesis, tested by S3: cancellation leaves the 24-item slope dominated by item noise. Steerability "varies widely across inputs" (https://arxiv.org/abs/2407.12404).
- The fix is therefore measurement and gating, not integration.

**GSM8K and MMLU-Pro.** These sit in a lazy regime: max ΔNLL is 0.127 (P2 §6). The literature is mixed on how far linearisation holds:
- prompt-based fine-tuning "often induces kernel-based dynamics" (https://arxiv.org/abs/2210.05643);
- linearisation holds "not always", only with regularisation toward θ₀ (https://neurips.cc/virtual/2025/133860);
- CLIP shows a "non-linear advantage" (https://arxiv.org/html/2305.12827v3).

**Harmless refusal.** Static, L, R and the trapezoid all have OLS 1.41–1.62, and corr(R−L, obs−L) = −0.13. This calls for a per-objective gain (S6), not transport.

**RMSNorm.**
- Scale steps (`scale_down15`, n = 5) make the trapezoid worse on four objectives. The sample is too small to conclude anything.
- Normalisation makes gradients orthogonal to the weights, but not for attention weights (https://arxiv.org/html/2306.00700v2).
- Our writers feed a residual sum, so invariance is partial. S2 measures curvature along the action directly, so no separate correction is proposed.

**Why K-FAC fails here.**
- The field's curvature is a model-sampled Fisher (`src/wsp/wm/collect.py`). Like the Gauss–Newton Hessian, "unlike H, G is always positive semidefinite" (https://arxiv.org/pdf/2308.03296 §2).
- A PSD curvature cannot represent the indefinite curvature of margins or of log-prob differences.
- Block-diagonal LLM curvature shows "order-one relative error" (https://arxiv.org/html/2602.00816v1).
- Our D and E lie on layers 12–15, so a support-only HVP gives DᵀHD and EᵀHD *exactly*.

## 2. Proposals

**Common fresh test (F).**
- Steps: `wme-p3` roots disjoint from pilot 2, the same composition design (≈ 53 states, ≈ 47 steps at h ≥ 2), validation items.
- Reference: the R2 oracle trapezoid on all states, ≈ 0.65 GPU-h (53 × 44 s, *computed*).
- Reporting: per objective, resolvable-only, pooled, and pooled without harmful.
- Platform: Lambda H100, for shared numerics (`RULING-wme-resources.md` §2). Priced at $4.89/h (`docs/reviews/wme/wme-27b-calib-2026-11-16.json`), assuming H100 time ≤ local time.
- Learned parts are fit on p3 train closures and tested on held-out closures.

**S1. Exact objective-Hessian step (R1), which sharpens C4(b1).**
- Mechanism: v = H₀D_t by forward-over-reverse (`jvp∘grad`; jvp/jacfwd∘jacrev in https://docs.pytorch.org/docs/main/func.whirlwind_tour.html) on support parameters. Predict ΔF̂ = g₀·D + E_{t−1}·v + ½D·v.
- Preconditions (0.8B smoke run, construction items): forward-AD support in eager attention and the torch GatedDeltaNet fallback; ≤ 28 GB, or else H100.
- Cost: 53 × an assumed ≤ 3 × 44 s ≈ 1.9 GPU-h.
- Pass: pooled cut ≥ 20% **and** ≥ 50% of the fresh oracle cut on harmful refusal.
- Kill: < 25% on harmful refusal, or the preconditions fail.
- Finite-difference HVPs (https://arxiv.org/html/2602.00816v1 Eq. 1) evaluate at θ₀ ± εD, which is **R2** under our tiers unless the chair rules otherwise.

**S2. Directional jets, the primary recommendation.**
- Mechanism: a nested `jvp(jvp(F))` at θ₀ gives per-item scalars ∂²_D F and ∂_E∂_D F. There is no backward graph; forward mode is "exact and efficient" (https://arxiv.org/abs/2202.08587). S2a uses ∂²_D F only.
- Field size: 96 construction items instead of 24. Per-item values give the SE that S3 needs.
- Cost: ≈ 1.1 GPU-h, assuming ≤ 4 forwards per jet and ≈ 9 s per 24-item forward, scaled from pilot 1's 23.7 s per state. Both factors are to be measured.
- Pass: ≥ 50% of the oracle cut on harmful refusal. S2a suffices if it reaches ≥ 80% of S2.
- Kill: < 25%, or a mismatch with S1 above 1e-3 relative on a CPU toy.

**S3. Reliability gate (CPU).**
- Mechanism: ŷ = static + w·(S2 − static), with w = max(0, 1 − ŝ²/(Δ̂² + ŝ²)). Here ŝ is the split-half construction SE of the increment Δ̂.
- Per-objective switch: on only if the p3-train cut is > 0, fixed before p3-test is read.
- Pass: on held-out data, no objective worse than static by > 0.05, and the pooled cut ≥ S2.
- Kill: the first condition fails.

**S4. Padé margins (R1).**
- Mechanism: a third-order jet gives c₁, c₂, c₃. The [1/1] Padé, c₁s / (1 − (c₂/c₁)s), checked against c₃, gives a bounded dose response for the refusal margins.
- Cost: ≈ 0.6 GPU-h (assumed).
- Test: fresh p3 ladders, |dose| ≥ 2 and h ≥ 3, against S2.
- Kill: LB ≤ 0, or sign flips on more than 10% of steps.

**S5. Learned end-slope residual (hypersolver; C4 b2).**
- Target: r = R − (g₀ + H₀E_t)·D_t.
- Features: S2 jets, norms, family, layer, overlaps and the S3 SE.
- Precedent: hypersolvers fit the base solver's "local truncation errors" (https://arxiv.org/html/2007.09601v2).
- Training: two stages, ladders then compositions, with 1/SE² soft weights. This follows the soft-target idea reported for Fugu (https://di-zhang-llm.github.io/blog/how-fugu-is-implemented-a-technical-inspect/); the primary abstract confirms only fine-tuning, evolution and RL (https://arxiv.org/abs/2606.21228).
- Access: R2 labels for training only (`src/wsp/wm/access.py`).
- Kill: < 25% of the oracle cut, **or** no gain over S2 with LB > 0. The second condition is new.

**S6. Direct outcome residual instead of slope grounding.**
- Mechanism: WM-CP (`RULING-wme-v2-design.md` A1) integrates its residual on top of the S2 nest, with a per-objective gain and no slope labels.
- Decision rule: if S5 does not beat S6 with LB > 0, drop the per-state oracle gradients from the bank.
- Pricing: the bank is 7.84 GPU-h without them and 24.07 with them at pilot-1 pricing (`wme-v0-pilot-2026-11-16.md`), and 28.4 under the current pricing (`RULING-wme-resources.md`).
- Kill: worse than static (LB < 0).

**Per objective (declared now).**
- Harmful refusal: S1/S2 (+S4).
- WikiText: S2 behind S3.
- Harmless refusal: S6 gain only.
- GSM8K and MMLU-Pro: static.
- Sycophancy (the H-E1 primary): no transport. Use static + S6 with 96 field items, and rely on real evaluations for planning.

## 3. Literature beyond §1

- **Hessian-aware merging.** It links merge error to "mismatches in the gradients" (https://arxiv.org/abs/2310.12808). That supports curvature-corrected composition.
- **Influence functions.** They track the proximal response, not leave-one-out (https://arxiv.org/abs/2209.05364). Inverse curvature is the wrong tool for a known D.
- **Lanczos at θ₀.** Feasible at scale (https://arxiv.org/abs/2505.11564), but the top eigenvectors need not align with D. **Not proposed.**
- **LoRA in the NTK regime** (https://arxiv.org/abs/2402.11867). Relevant to E2's small-λ LoRAs, not to transport.

## 4. Fetch record (17 Nov)

| Status | Sources |
|---|---|
| Abstract only | 2210.05643, 2402.11867, 2202.08587, 2505.11564, 2602.02712, 2209.05364, 2310.12808, 2606.21228, 2407.12404 |
| Full HTML | 2305.12827v3 (first 100k chars), 2007.09601v2, 2306.00700v2, 2602.00816v1, NeurIPS 133860, the PyTorch tour, the Fugu blog |
| Partial | 2308.03296: the PDF was unparseable by WebFetch; the saved copy was read locally with `pdftotext` (§2) |

**Failed:**
- PMC11500277 (Δ-ML): reCAPTCHA.
- 2602.02343 PDF: unreadable.

Search-snippet-only claims are not used.

## Summary table

| proposal | mechanism | access tier | test on fresh data | cost (GPU-h, $) | kill criterion |
|---|---|---|---|---|---|
| F reference | R2 oracle trapezoid on p3 states | R2 (reference) | — | 0.65; ≤ $3.2 | — |
| S1 exact Hessian | support HVP; g₀D + EᵀH₀D + ½DᵀH₀D | R1 | pooled cut ≥ 20% and ≥ 50% of oracle on harmful | ≈ 1.9 (assumed); ≤ $9.3 | < 25% on harmful; preconditions fail |
| S2 jets (primary) | nested JVP; per-item ∂²_D F, ∂_E∂_D F; 96 items | R1 | oracle share on harmful and pooled; S2a ≥ 80% of S2 | ≈ 1.1 (assumed); ≤ $5.4 | < 25%; mismatch with S1 on a toy |
| S3 gate | split-half-SE shrinkage; per-objective switch | R0/R1 (CPU) | held-out: no objective < static − 0.05; pooled ≥ S2 | CPU | any objective < static − 0.05 |
| S4 Padé | 3rd-order jet; [1/1] Padé on margins | R1 | p3 ladders \|dose\| ≥ 2, h ≥ 3, vs S2 | ≈ 0.6 (assumed); ≤ $2.9 | LB ≤ 0, or > 10% sign flips |
| S5 learned end slope | hypersolver head; 2-stage; 1/SE² weights | R0/R1 predict, R2 labels | oracle share on held-out closures | CPU | < 25%, or no LB > 0 gain over S2 |
| S6 direct residual | WM-CP Δ-learning on the S2 nest plus gain | R0/R1 | S6 vs S5 vs static, held-out | CPU; saves the bank's oracle share | LB < 0 vs static |
