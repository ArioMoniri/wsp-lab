# Study M′ (EXPLORATORY): phase report 1, base (22 Nov 2026)

**Plan:** `docs/PREREG_WMM.md`, frozen at tag `prereg-wmm-exploratory` (`1904bbe`).
**Driver:** `run/wmm-mprime` at `423615f`.
**Run:** `data/wmm/mprime-2026-11-21` (not committed). **Machine-readable:** `wmm-mprime-base-2026-11-22.json`.

Every outcome was read through the FC1 read-gate. Every result below is **exploratory**. The base phase has no
paired claims; it provides the reference for M-SYC and M-TE.

## Base, `google/medgemma-1.5-4b-it` @ `91850547`

Generation: frozen harness, greedy, fixed panel orders, batch 32. No OOM occurred.

| Measure (sealed panel) | Estimate [95 % Wilson CI] | n |
|---|---|---|
| MedQA neutral accuracy | 68.8 % [66.5, 71.1] | 1,544 |
| MedQA wrong-suggestion adoption | 24.8 % [22.7, 27.0] | 1,544 |
| MedQA accuracy under a wrong suggestion | 57.6 % [55.1, 60.0] | 1,544 |
| MedMCQA neutral accuracy | 57.0 % [54.9, 59.2] | 2,000 |
| ARC-Challenge non-health accuracy | 77.8 % [76.0, 79.5] | 2,161 |
| Harmful requests: refusal (frozen rule) | 59.4 % [57.0, 61.7] | 1,700 |
| Benign MedQuAD: refusal (frozen rule) | 1.9 % [1.3, 2.7] | 1,500 |

**Truncation at 4,096 tokens:**

| Set | Truncation |
|---|---|
| MedQA neutral | 2.46 % |
| MedQA suggested | 4.02 % |
| MedMCQA | 2.15 % |
| ARC | 1.71 % |
| Harmful (1,024 tokens) | 8.1 % |
| Benign (1,024 tokens) | 40.1 % |

- **B\*** (≤ 2 % truncation) is reached for ARC (3,072) but not for MedQA or MedMCQA at 4,096. It is descriptive; no
  M′ claim uses B\*.
- **Parse rates:** MedQA 97.3 %, MedMCQA 94.7 %.
- **Mean tokens:** MedQA 1,210, MedMCQA 1,140.

**G-CAL, base** (direct-answer letters, 3,544 items, batch 1):

| Measure | Value |
|---|---|
| Mean Brier score | 0.374 |
| Mean max-p | 0.849 |
| ECE | 0.313 |
| Direct-answer accuracy | 53.6 % |

These match the calibration run on exposed items (0.392 / 0.849).

**Descriptive generated-answer Brier** (≡ −Δaccuracy, F1; letter read after reasoning, p ≈ 1): MedQA 0.292,
MedMCQA 0.397.

## GPU-h ledger (cap 42.0; wall time with the model resident)

| Item | GPU-h | Estimate |
|---|---|---|
| Job start (load, canaries, byte hashes) | 0.03 | — |
| base `mq_neutral` | 4.75 | 2.81 |
| base `mq_suggested` | 6.92 | 2.83 |
| base `mm_neutral` | 4.26 | 2.85 |
| base G-CAL | 0.04 | 0.05 |
| base harm | 0.67 | 0.62 |
| base benign | 0.56 | 0.48 |
| base gen | 3.36 | 2.40 |
| **Base total** | **20.58** | 12.0 |

**Cuts applied:** none so far.

**Engineering note (chair decision, 22 Nov 04:33).** The M′ process was CPU-starved at nice 10:
- 8 cores, load average 20–25 from other sessions' jobs;
- GPU utilisation bursting between 0 % and 100 %;
- `mq_neutral` at 109 tokens/s.

Only the M′ process was reniced to 0 (fair share). No other process was touched, and the ledger is unchanged.
- Base sets after the renice ran at about 1.5× their estimates, against 1.7–2.4× before it.
- `mq_suggested` (6.92 h) finished just before the renice.

**Projection at about 1.5×.** M-SYC completes near 39–40 GPU-h. The hard stop would then cut most of M-TE, and the
whole M-A phase, its minimum included. Those parts would be reported "not measured". M-SYC itself is not projected to
be cut.

**Integrity.** The job-start canaries match the run's reference, and the base sha256 recomputed from bytes is
`8ef19ef6…`. The end-of-job canaries are checked when the job ends.
