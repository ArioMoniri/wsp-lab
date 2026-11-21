# Chair ruling, 21 Nov 2026: Board A REJECT of PREREG_WMM, and an exploratory M′

**Input.** `docs/reviews/council/board-a-prereg-wmm-2026-11-21.md` (REJECT as confirmatory).

## What Board A found

**The Brier guard on generated answers measures no calibration.** On Gate 1 data it equals the accuracy drop to
within 2×10⁻⁵. Board A's outcome-blind redesign G-CAL is a proper Brier score on no-reasoning letter probabilities,
with the +0.01 margin and one-sided α 0.025.

**Even with G-CAL, joint-pass power falls short.**

| Claim | Joint-pass power | Limited by |
|---|---|---|
| M-SYC | ≈ 0.29–0.34 | the harm guard |
| M-TE | ≈ 0.21 | — |

Widening the margins would be margin-shopping, and Board A correctly rejects it.

## Decisions

**W1. Confirmatory M′ is killed** (ruling `RULING-wmm-gate1.md` item 4).
- No confirmatory claim will be made on MedGemma.

**W2. M′ goes ahead as a pre-registered EXPLORATORY study.** It keeps the same fresh sealed panels and the same frozen
plan, and every result is labelled exploratory.

The reasons:
- the user asked for enhancement to be tested in its strongest feasible form;
- the endpoints alone are well powered: adoption is 0.90 at half the dev effect, and the M-TE token and low-budget
  endpoints are 1.00;
- the guards are reported with their CIs and are never waived silently;
- the study runs locally at $0.

The paper may report it as an exploratory result: "on MedGemma, edit X reduced adoption by Y (CI); guard results Z". It
may not report it as a confirmed claim.

**W3. Conditions before freezing the exploratory plan.** These are Board A's conditions, adopted in full:
- G-CAL, including:
  - the ≤ 0.3 GPU-h SD measurement on exposed Gate 1 items;
  - non-vacuity gates;
  - a positive control.
- **Chair note on SYNTHESIS-r12 R4.** G-CAL is a teacher-forced calibration probe, used alongside the measured
  generated guards. It does not predict them, so R4's "measured on generated output, never predicted" is respected
  for harm, over-refusal and general ability.
- Optional panel enlargement, sealed before any data: ARC to about 6,000 items, benign to about 4,000. Adopt it only
  if it fits the compute cap after re-pricing.
- **TE-I** is exploratory only.
- **M-KD** stays conditional, with α fixed at 0.025/3 per family and an all-items superiority test added.
- **M-A** is declared likely-null and descriptive. Fix the SE = 0 → NaN bug for inert sham units with a test.
- **Compute.** Re-price the study (Board A puts the current estimate at about 1.3× too low) and set an explicit local
  cap of 30 GPU-h. If the re-priced plan exceeds it, drop the M-A bank size first.

**W4. Order.**
1. Implement W3.
2. A short Board A check of the conditions only.
3. Freeze the exploratory plan (tag `prereg-wmm-exploratory`).
4. Run locally. C1 keeps priority on everything shared.

## Addendum (21 Nov 2026): chair notes before the exploratory freeze

**C6 note (SYNTHESIS-r12 R4).**
- **What G-CAL is.** It is a teacher-forced calibration probe on no-reasoning letter probabilities, and is a guard in
  its own right. It is **not** a surrogate for any generated-output guard and does not predict one.
- **What stays measured on generated output.** Harmful compliance, benign over-refusal and general ability, as R4
  requires.
- **What G-CAL replaces.** The generated-answer Brier guard. Board A showed that guard reduced to an accuracy limit.

**Local compute cap and M-A.**
- The user explicitly chose the narrowed study *with* the edit-direction prediction test (M-A). Dropping M-A to 0 units
  to fit 30 GPU-h would remove a user-selected component.
- **The local cap rises to 42 GPU-h**, at $0 on the shared local GPU with ≤ 16 GB VRAM. C1 has priority.
- M-A is restored with its re-priced bank, about 10.7 GPU-h. If the 42 GPU-h cap still binds, cut in this order:
  1. M-A bank size, down to its declared minimum;
  2. then M-TE's ARC guard, with M-TE's claim then not made.
- M-A stays declared expected-null and descriptive.

**Expected guard failures are disclosed, not fixed.**
- On dev, the sycophancy LoRA worsens G-CAL by +0.040, against the +0.01 margin.
- The plan must state that the M-SYC edit is expected to fail G-CAL and the neutral-accuracy limit.
- The exploratory report is then the trade-off itself: the adoption reduction against its calibration and accuracy
  cost, each reported with its CI.
