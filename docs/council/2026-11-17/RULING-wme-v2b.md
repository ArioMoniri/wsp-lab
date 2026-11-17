# Chair ruling on WM-E v2b: the H-T alternative, claim-set priority, caps and the E1 effect size (17 Nov 2026, pre-freeze)

**Inputs:** `docs/reviews/wme/wme-v2-2026-11-17.md` §7 flags 1–5 (merged on main). No bank, trajectory, enhancement or confirmatory outcome exists. All figures below are synthetic or outcome-blind.

## A7. The H-T alternative must be physically possible

**Why v1's alternative is withdrawn.** v1's alternative made outcomes depend on the decomposition of the edit (γ = 3). The target system cannot do that: outcomes depend only on the realised θ_T (r8 D3; r9 world-model BLUF 1). An alternative the real system cannot produce is a **simulator specification error**. Replacing it is a correction, not re-tuning. It is withdrawn.

**Registered replacement alternative.**
- **Response.** A path-independent response that is non-quadratic in R(E_T):
  - a second-order field near zero;
  - plus a saturating, normalisation-type departure that grows with |E_T| along each family's dose direction;
  - with independent per-component random strength, so no single global curve exists.
- **Magnitude.** Fixed from **exposed exploratory pilot-2 evidence**: the median refitted-static error at h ≥ 3 was **2.17 paired SE**. The simulated departure is set so that the static map's held-out error at h = 3–4 has that median. It is not adjusted for power.
- **Paired null scenario.** A quadratic world in which WM-CP must not beat static, so C_T ≈ 0. It is included in the size and coverage gates alongside the existing null.

**Allowed before the freeze.**
- WM-CP's configuration and its K(E_T) rule may be chosen on **construction-split data or synthetic data only**, then frozen in the manifest.
- Nothing is chosen on the simulation's alternative outcomes.

## A8. Claim-set priority (resolves flag 2, and flag 3 with A9)

Priority follows the user's stated goals: trajectory prediction **and** enhancement using the world model.

| Rank | Claim set |
|---|---|
| 1 | {H-T, H-E1, H-E2} |
| 2 | {H-T, H-E1} |
| 3 | {H-E1, H-E2} |
| 4 | {H-E1} |

- Within a rank, the cheaper powered design wins.
- Unpowered claims are not run as confirmatory. They are registered **"unresolved — not powered"**, and their arms may run only as clearly labelled descriptive analyses.

## A9. Caps raised (authorised by the user: "as much resource as needed", Lambda)

The new caps, which replace RULING-wme-resources §1:
- confirmatory: **≤ 90 GPU-h-eq**;
- bank: still ≤ 35;
- Lambda for 9B: **≤ $1,000**.

One short **P-H100b** run is allowed, at most 1 instance-hour, to measure larger generation batches (flag 4) and re-price generation.

## A10. E1 effect size on TriviaQA (flag 5)

TriviaQA wrong answers come from a frozen within-type rule and are less plausible than the curated ones. The **E1 generation N is therefore set for power ≥ 0.80 at 0.5× the pilot-template effect**. The full effect is reported as a sensitivity analysis.

## A11. Re-simulation

- Re-run the cheap grid and the full-refit validation under A7–A10, using the §4(b) rule.
- Apply A8 to select the design.
- Report the achieved C_T and per-claim status.
- Then hand off to the PREREG v2 author.
