# Chair ruling, 21 Nov 2026: study M Gate 1 outcome and the narrowed study M′

**Input.** `docs/reviews/wmm/wmm-gate1-2026-11-20.md` and its `.json` (main `49d7a47`).

**Gate 1 verdict: KILL under rule 1, as written.**
- None of the 11 edits improved generated accuracy on MedQA or MedMCQA. The central doses moved it by −1.5 to −2.5 pp.
- This is recorded as a second-model replication of the 9B headroom finding: edits do not add accuracy to a strong,
  domain-tuned model.
- Study M's accuracy claims are closed.

**Surviving signal (descriptive, dev).**
- The sycophancy LoRA reduced wrong-suggestion adoption by 8.5 pp at λ 1 (95 % CI 2.8–14.2), from a base of 23.5 %.
  The reduction grows with dose.
- It passed the harmful-compliance, over-refusal and general-ability guards.
- It missed the Brier guard by 0.0009.

**User decision (21 Nov 2026): proceed with a narrowed study M′.** This is a **new** pre-registration,
`docs/PREREG_WMM.md`, written before any M′ data and frozen only after Board A. It uses fresh, unseen items only. Gate
0–1 items are exposed and may be used for development only.

1. **Primary, M-SYC.** The pre-specified sycophancy LoRA recipe, with the dose fixed in advance from Gate 1 dev
   (λ = 1), reduces wrong-suggestion adoption on fresh items. This is a second-model check of the 9B H-E1 direction,
   never pooled with it.
   - **Guards:** harmful compliance, over-refusal, general ability, and Brier. The Brier margin is fixed now at
     +0.01, unchanged; the Gate 1 near-miss is disclosed.
   - **Neutral accuracy:** must not fall by more than 3 pp.
2. **Exploratory, M-A.** The beyond-dose edit-direction prediction test of SYNTHESIS-r13 R5 and its addenda.
   - **Bank:** at least 30 matched-norm direction units.
   - **Target:** wrong-suggestion adoption and truncation.
   - **Baseline to beat:** pooled + dose.
   - **Features:** R0/R1 only, including length and EOS-logit features.
   - **Cost:** about 7–13 local GPU-h, priced from Gate 1 timing.
3. **Batch order is fixed** for every pair of compared generations: the same batch composition and order for base and
   edit. Gate 1 measured a 13.5 % disagreement floor caused by batch composition alone, and the prereg must state it.
4. **Sizing** follows the Gate 1 discordance, using the round-11 formula. The achieved power must be declared before
   the freeze. If power for M-SYC is below 0.5, study M′ is killed.
5. **Limits:** local GPU only, ≤ 16 GB, $0, wmm venv. No edits to wme-c1-registered files. C1 has priority.
