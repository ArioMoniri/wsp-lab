# Chair ruling, 18 Nov 2026: consequences of stage W, and the path to the freeze

**Input.** `docs/reviews/wme/wme-stagew-2026-11-18.md` and `.json` (EXPLORATORY), merged at eb08cf0. The pre-declared
rules are SYNTHESIS-r11 R1/R4, `RULING-wme-post-pilot3.md` P3/P5, and `RULING-wme-stagew.md`.

## Q1. H-T2 is not registered

- **Empirical power is 0 at every path count.** Simulated power reaches 0.89 only in the few-shot settings at 40
  paths, and only because the simulation treats the fitted world model as the truth. The rule requires both, so it
  fails.
- **On measured paths, a pooled same-path-type curve beats the world model in every setting.**

  | Setting | Pooled curve | World model |
  |---|---|---|
  | Zero-shot | 0.62 SE | 2.34 SE |
  | k = 2 | 0.88 SE | 2.10 SE |
  | k = 4 | 0.77 SE | 1.77 SE |

  StatNN also beats the world model at k = 2 and k = 4. The stage-W gate fails.
- **Consequence (R4):** the 9B confirmatory study does **not** claim (a). The old H-T and H-T2 run as labelled
  descriptive analyses.

## Q2. World model vs direct multi-fidelity best-of-N is a registered secondary

- At total-compute parity, power is 0.006 (prior quality 0.348). It stays below 0.07 under every sensitivity.
- It is reported whatever the result, and this is disclosed as a structural cost disadvantage of the bank.

## Q3. Registered claim set for PREREG v2

- **H-E1 (enhancement):** sycophancy, TriviaQA generation, N = 1,000, at the frozen batch size of 128. Power 0.884
  at 0.5× the generated effect.
- **H-E2:** world-model-guided vs StatNN-guided search, family-stratified sign-flip test. Power 0.828.
- **Secondaries:**
  - world model vs multi-fidelity best-of-N at compute parity (Q2);
  - token efficiency at 2,048 tokens for one H1 recipe retrained with fresh seeds, with the 4,096-token and
    untruncated-in-both analyses (P3 wording);
  - the E1 secondaries already listed in the v2b manifest, if they are within the caps.
- **Descriptive:**
  - the old H-T and H-T2;
  - the stage-S slope result;
  - inert-edit and trust-region controls (the C2 red team).
- **Design and cost:** the v2b manifest values (A8 rank 3), 65.7 GPU-h-eq, ≈ $354. Adding the secondaries must keep
  the total at ≤ 90 GPU-h-eq and ≤ $1,000.

## Q4. One bounded, pre-declared attempt to rescue (a): stage W2

Stage W found that path-type structure dominates. The obvious world-model design that respects this is **nested**:

> pooled same-path-type curve + an edit-informed residual, shrunk toward 0

This follows the F3u-nesting principle used throughout the programme. A nested model can lose to the pooled curve
only through residual overfit, which the shrinkage controls.

**The attempt:**
- Write a pre-declared plan before any W2 data exists.
- Use ≥ 40 fresh paths with fresh seeds, H1/H3-type, over training step × λ × layer mask.
- The primary contrast is the nested world model vs the pooled curve, few-shot k = 2 and zero-shot, using
  leave-one-run-out and the same nulls as stage W.
- **Pass:** the lower bound of the improvement over the pooled curve is > 0, in either setting, Holm-adjusted, with
  every null at size ≤ 0.035.
- **Cap:** ≤ 10 GPU-h, ≤ $60, instances named `wsp-wme-w2-*`.

**Consequence of a pass:**
- H-T3 (nested world model vs pooled curve) is added to PREREG v2 by a dated amendment, *before* its
  confirmatory paths exist.
- Those paths form a separate sealed set: new seeds and new panels, ≈ 9.5 GPU-h, ≈ $51.

**Consequence of a fail:**
- The paper reports (a) as a negative result: on this model, edit-specific trajectory prediction does not beat
  path-type averages.
- No further spending on (a) for the 9B model.

**Timing.** W2 runs in parallel with PREREG v2 writing. The freeze waits for the W2 outcome only to decide whether
H-T3 is included; the E-claims do not depend on it.

## Q5. Next steps

1. Track W2 (stage W2, ≤ $60) and the PREREG v2 author run in parallel.
2. Board A re-reviews PREREG v2, then the freeze.
3. The 9B bank and confirmatory runs, on at most 4 concurrent 1× H100 SXM5, under the R9 harness.
4. 27B under R9. Its scope is H-E1, plus H-T3 if registered.

**Lambda spend to date:** about $153, on top of the $14.92 engineering spend counted before track A. No
`wsp-wme-*` instance is live.
