# Chair ruling, 18 Nov 2026: open issues in PREREG_WME DRAFT v2 (11d9828)

These answer the author's issues 1–8 in `docs/reviews/wme/prereg-v2-authoring-2026-11-18.md`. They are made before
any confirmatory data exists.

1. **C2 controls on H-E2 are descriptive, as Q3 says.** They are reported for every arm and do not gate H-E2. Two
   controls remain **validity preconditions** for the whole run, not tests:
   - the deterministic engineering checks: the inert-edit bitwise identity, and the 1-vs-2-GPU bitwise parity;
   - the base canary.

   A failed precondition halts the run for repair before any further outcome is read. It is not scored.
2. **H-E1 keeps its registered teacher-forced refusal guards.** The generated OR-Bench / LLM-LAT guards are
   reported descriptively for H-E1 and gate S-TE only.
3. **"≤ 90 GPU-h-eq" means bank plus confirmatory, combined.** The bank is re-priced without the per-state oracle
   gradients that P1 removed, which gives 76.43 GPU-h-eq with no H-T3. If H-T3 is added later by amendment, it must
   fit in the remaining 13.57. If it does not fit, the §II.14 cap-fit rule applies.
4. **The S-BoN parity budget is the measured one** (≈ 86.7 full-evaluation equivalents). Its power is recomputed at
   that budget and stated; the claim stays secondary.
5. **Availability re-audit at the panel build is mandatory**, counting every exposure (pilot 3, stage W, W2). If a
   pool is short, N drops to the largest fresh count, the power is restated before any outcome is read, and the
   freeze record states it. The shared MMLU-Pro pool is split across S-TE and the other panels by a salted
   deterministic assignment written before the build.
6. **R6 must be re-simulated before the freeze.**
   - Run the quadratic-null size, and the A7 alternative, with symmetric shrinkage in place, on CPU.
   - If the quadratic-null size is > 0.035, R6 is redesigned before the freeze. H-T is descriptive, so this binds
     only the descriptive analysis and any later H-T3.
7. **S-TE** stays as authored: analytic power, one training seed, guards at α 0.025, with N 523/200 as sized.
   It is a secondary.
8. **E2 is guided by WM-E**, the ensemble world model simulated in v2b. WM-CP is the H-T path model. Confirmed.

**Next steps.**
1. The author applies items 1–8 and runs items 3, 4 and 6 (CPU only).
2. Then Board A gives its independent review of v2.
3. Then the freeze.
