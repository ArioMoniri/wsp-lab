# Study M′ (EXPLORATORY): phase reports 2–4 and final report (23 Nov 2026)

**Plan:** `docs/PREREG_WMM.md`, tag `prereg-wmm-exploratory` (`1904bbe`). **Driver:** `run/wmm-mprime`.
**Run:** `data/wmm/mprime-2026-11-21` (not committed). **Machine-readable:** `wmm-mprime-final-2026-11-23.json`.
**Base report:** `wmm-mprime-base-2026-11-22.md`.

Every outcome was read through the FC1 read-gate. Every result is **exploratory**. Intervals are two-sided 95 %
(Tango, paired). Claim tests use the registered one-sided levels (α_f = 0.025/3 for endpoints, 0.025 for guards and
non-inferiority).

**E4 status: not yet done.** The guard-scorer re-audit (two blind judges, 75 harmful + 75 benign per state, for base,
M-SYC and M-TE) has its packets built (450 items, internal only) but no judge has labelled them. Every harm/benign
guard result below is therefore **provisional on E4**.

## Ledger and cuts (FC2)

| Phase | GPU-h | Status |
|---|---|---|
| Job start / end (canaries, byte hashes) | 0.03 | done |
| Base | 20.55 | done |
| M-SYC (pool, training, 7 sets) | 16.13 | done |
| M-TE without ARC (pool 512, rewrite, training, 5 sets) | 5.55 | done |
| **Total** | **42.26** of 42.0 | hard stop reached after `mte:harm` (started at 41.81) |

**Cuts applied, in the registered order:**
1. **M-A minimum bank (20 units) and the M-A base/pool sets:** not measured (cap).
2. **M-TE ARC guard:** not measured (cap). The M-TE claim is therefore **not made**.
3. **M-A remaining 12 units:** not measured (cap).
4. TE-I: not run.

The schedule placed the M-A minimum before the M-TE ARC guard, so both were lost together once the ledger passed 42.0.
Cause: throughput ran at about 1.5–2.4× the Gate 1 estimates on the shared host (CPU contention; see the base report's
engineering note).

**Integrity.** Job-start and job-end weight sketches identical; batch-1 behavioural canary identical at both ends;
base sha256 from bytes `8ef19ef6…`; one hash-chained provenance record per realised edit (M-SYC, M-TE); every restore
returned the base bytes. No OOM; every set ran at batch 32.

## Phase 2: M-SYC (sycophancy LoRA, λ 1; 98 training rows, 26 steps)

**Exploratory wording.** On MedGemma 1.5 4B, the sycophancy LoRA reduced wrong-suggestion adoption on MedQA from
24.8 % to 9.9 %, **−14.9 pp (95 % CI −17.0 to −12.9)**. **The claim is not made: two guards failed** — G-CAL (as
expected) and harmful compliance (not expected).

| Condition | Estimate [95 % CI] | Registered test | Result |
|---|---|---|---|
| Adoption change (primary, n 1,544) | −14.9 pp [−17.0, −12.9] | UB < 0 at α_f | pass |
| Adoption change, parsed in both (n 1,452) | −15.3 pp [−17.4, −13.3] | UB < 0 at α_f | pass |
| Neutral accuracy, MedQA + MedMCQA (n 3,544) | −0.06 pp [−1.5, +1.4] | LB ≥ −3 pp | **pass** (expected fail; did not fail) |
| **G-CAL ΔBrier** (n 3,544) | **+0.043 [+0.039, +0.047]** | UB ≤ +0.01 | **FAIL** (expected) |
| **Harmful compliance** (n 1,700) | **+1.7 pp [+0.4, +3.0]** | UB ≤ +2 pp | **FAIL** (provisional on E4) |
| Benign over-refusal (n 1,500) | −0.7 pp [−1.5, −0.1] | UB ≤ +2 pp | pass |
| ARC general accuracy (n 2,161) | +0.5 pp [−1.4, +2.4] | LB ≥ −3 pp | pass |

Descriptive:
- Parse rate rose 1.9 pp [0.8, 3.1]; accuracy under a wrong suggestion rose 11.1 pp [8.8, 13.4].
- Generated-answer Brier change (≡ −Δaccuracy, F1): −0.004 [−0.018, +0.010].

**The trade-off (addendum).** Adoption fell about 15 pp at no measurable neutral-accuracy cost, but the edit made the
model's direct-answer letter probabilities worse calibrated (+0.043 Brier, four times the margin, matching dev +0.040),
and it **raised compliance with harmful requests by an estimated 1.7 pp**, with an upper bound above the 2 pp margin.
The harmful-compliance result rests on the frozen refusal rule; E4 must confirm it. Harmful outputs stay internal.

## Phase 3: M-TE (short-form self-rewrite LoRA, λ 1; 388 correct pool answers, 98 valid rewrites, 26 steps)

**Exploratory wording.** The token-efficiency LoRA cut mean response length on MedQA + MedMCQA from 1,170 to 294
tokens (Δ log-tokens −1.71 [−1.74, −1.69]) and raised accuracy at a 1,024-token budget by +23.4 pp [+21.6, +25.2],
but **lowered full-budget accuracy by 2.7 pp [−4.4, −1.1]**, failing its −2 pp non-inferiority limit. **The claim is
not made** (non-inferiority failed, and the ARC guard was cut by the cap).

| Condition | Estimate [95 % CI] | Registered test | Result |
|---|---|---|---|
| Δ log-tokens (n 3,544) | −1.71 [−1.74, −1.69] | UB < 0 at α_f | pass |
| Accuracy at 512 | +54.3 pp [+52.6, +55.9] (3.5 % → 57.8 %) | LB > 0, Holm within α_f | pass |
| Accuracy at 1,024 | +23.4 pp [+21.6, +25.2] (35.9 % → 59.2 %) | LB > 0, Holm within α_f | pass |
| **Full-budget accuracy, untruncated in both** (n 3,405) | **−2.7 pp [−4.4, −1.1]** | LB ≥ −2 pp | **FAIL** |
| G-CAL ΔBrier | −0.011 [−0.014, −0.007] | UB ≤ +0.01 | pass |
| Harmful compliance | −0.7 pp [−1.9, +0.6] | UB ≤ +2 pp | pass (provisional on E4) |
| Benign over-refusal | −0.4 pp [−1.2, +0.3] | UB ≤ +2 pp | pass (provisional on E4) |
| ARC general accuracy | — | LB ≥ −3 pp | **not measured (cap)** |

Descriptive: full-budget accuracy on all items (truncated = wrong) −2.7 pp [−4.3, −1.0]; generated-answer Brier
change +0.025 [+0.009, +0.041] (≡ −Δaccuracy, F1). The low-budget gains are mostly the base model running out of
budget (base accuracy at 512 is 3.5 %), not better reasoning.

## Phase 4: M-A

**Not measured.** The hard stop removed the whole M-A phase, including the declared 20-unit minimum. Registered
wording: "beyond-dose prediction not established on study M". No M-A data exist; the sealed `ma` panel and `pool_ma`
were not read.

## Final exploratory summary

1. **M-SYC:** large adoption reduction (−14.9 pp), neutral accuracy unchanged; claim not made because G-CAL failed
   (expected, disclosed in the plan) and harmful compliance rose beyond its margin (unexpected; provisional on E4).
2. **M-TE:** a ~4× token cut with large low-budget gains, at a −2.7 pp full-budget accuracy cost; claim not made
   (non-inferiority failed; ARC guard cut).
3. **M-A, TE-I:** not run (cap).
4. **Pending:** E4 judge audit of the harm/benign guards (base, M-SYC, M-TE), which can change the harm-compliance
   result for M-SYC.
