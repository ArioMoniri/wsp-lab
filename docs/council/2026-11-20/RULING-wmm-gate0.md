# Chair ruling, 20 Nov 2026: study M Gate 0 outcome

**Input.** `docs/reviews/wmm/wmm-gate0-2026-11-19.md` and its `.json` (main `162f0d8`).

**What Gate 0 found.**
- The port is clean.
- Every inert and null check is bitwise.
- Dose response is monotone for the task vector and for the 1.5 → 1 interpolation. It is not monotone (it peaks at λ = 1) for the two self-distilled LoRAs.
- The sycophancy LoRA reduces adoption of the suggested wrong answer by 17.2 points (95 % CI [−28.4, −6.0]), with neutral accuracy unchanged.
- Base MedQA under our template is **60.5 %** on 400 dev items, against the 1.5 card's 69.1. That miss of −8.6 points trips the third kill criterion.

The diagnostics point to answer format and length rather than knowledge:

| Diagnostic | Value |
|---|---|
| Answers truncated at 2,048 tokens | 8.75 % |
| Answers ending in `\boxed{X}`, which the strict extractor ignores | 5 % |
| Accuracy with the flexible extractor | 64.0 % |
| Accuracy on parsed answers only | 70.1 % |

The criterion text in `SYNTHESIS-r12.md` says **"kill or redesign"**. No study M panel exists yet and nothing is sealed, so a redesign of the *evaluation harness* is admissible. The model and the bank are not redesigned. The redesign must be declared here, before it is tested, and tested on items it was not tuned on.

## Decisions

**M1. One harness redesign, one attempt (Gate 0b).** The fix is declared now and is final:
- **Prompt:** the answer instruction follows the model card's evaluation format: zero-shot step-by-step, ending with "Final Answer: (X)".
- **Extractor:** it accepts "Final Answer: (X)", a trailing `\boxed{X}`, and a final line holding only the letter, in that precedence. Nothing else is accepted.
- **Budget:** the token budget is 4,096. Truncation is reported.

**Items:** Gate 0b runs on **400 fresh MedQA dev items** under a new salted designation. None of them may be among the 784 items Gate 0 already exposed.

**Pass rule:** strict accuracy is at least 64.1 % (card − 5) **and** truncation is at most 5 %.

**On failure:** the MedQA primary is dropped, and study M is killed unless MedMCQA, run under the same harness and the same rule against its card value of 59.8, passes. That fallback is declared here too. One attempt only; no further iteration.

**M2. The LoRA families stay in the bank.**
- Gate 0 counted 2 of 4 families as non-monotone, below the "3 or more" kill threshold, so the dose rule is met as written.
- The drift-based dose metric does not suit self-distilled LoRAs. This is declared now, after seeing their ladders, so it is marked **post hoc**:
  - At Gate 1, the LoRA families' dose is judged on their **target behaviour**: generated accuracy for H1, wrong-suggestion adoption for the sycophancy LoRA.
  - This is tested on fresh Gate 1 items.
  - The Gate 0 drift verdict ("fail") stays on record unchanged.

**M3. The k_proj × 2 inert control is removed for Gemma 3.** Gemma 3 normalises keys (k_norm), so this control is not exact on it. It is not a bitwise identity on this architecture. This follows from the architecture, not the outcome, so it is accepted. The removal is recorded.

**M4. Budget.** Gate 0 used 2.32 GPU-h of its ≤ 3.
- Gate 0b is allowed ≤ 1 GPU-h, locally, at $0.
- All isolation rules are unchanged:
  - ≤ 16 GB of VRAM;
  - the `wmm` venv;
  - no Lambda;
  - nothing touching C1.

**M5.** If Gate 0b passes, Gate 1 (the headroom probe) proceeds as in `SYNTHESIS-r12.md`, with the M1 harness frozen.
