# Chair ruling on WM-E: resources, platform, item supplements and the simulation plan (17 Nov 2026)

**When made.** This ruling was made before the freeze. No bank, trajectory, enhancement or confirmatory outcome existed when it was made, and none had been read.

**Inputs.**
- `docs/reviews/wme/wme-confirm-eng-2026-11-17.md` (merged at 6cbc1a9).
- `docs/PREREG_WME.md` DRAFT v1.
- Board A's review, conditions 1–14.

**Authority.** The user authorised "as much resource as needed", for the work to be done "complete and fast", and on Lambda using the stored API key.

## 1. Budget caps (these replace the v0 caps of 12 + 10 GPU-h)

The v0 caps were set before the grounding labels and the secondary arms had been priced. Under those caps the engineering report prices the study as **unresolved, not powered**. Because slope grounding passed Go-R2 and is part of the D3 design, it is kept.

| Item | New cap | Basis |
|---|---|---|
| 9B bank, including per-state oracle gradients for grounding | ≤ 35 GPU-h-equivalent | Priced at 28.4 |
| 9B confirmatory, all registered arms | ≤ 50 GPU-h-equivalent | Priced at 40.5 |
| Lambda spend for 9B | ≤ $800 | — |
| Lambda spend for 27B (bank priced at ≈ 9 GPU-h, ~$44, plus pilot and confirmatory) | ≤ $600 | Re-priced at the 27B registration |

**If a priced item exceeds its cap,** it is re-priced and recorded as a dated amendment before any of its data exist. The study does not shrink its N to fit a cap and then claim confirmation.

## 2. Platform

**Hardware.** The whole 9B confirmatory study runs on **one platform: Lambda H100 SXM5 80 GB**, so the bank and all tests share numerics.
- Up to 8 GPUs may be used, data-parallel over states.
- Partial results are reduced on the CPU in a fixed order, with a 1-vs-k parity test at rtol 1e-6 (pro-engineer L1–L2).
- The dead-man terminator is required (L4).

**Pilot values.** The pilot values (nominals and SEs) were measured on the local RTX PRO 4500. They are design inputs only.

**Local GPU.** The local GPU remains available for:
- engineering;
- re-measuring the per-state SE on H100 before the simulation is finalised (P-H100);
- engineering-only checks on construction items.

**P-H100.** This is a timing and noise pilot on H100, run on construction and validation items only, using no-op, inert and fixed timing edits. It gives the rates and SEs the simulation uses.

## 3. Item supplements, named before the freeze

**Sycophancy.** 467 question groups are left, which is feasible as proposed. No supplement.

**Harmful-refusal guardrail.** 127 groups are left, which binds the test split at 32 items. Pinned supplementary sources, in this order of preference:
1. AdvBench harmful behaviours;
2. HarmBench text behaviours;
3. JailbreakBench JBB-Behaviors.

Each source is pinned by revision and sha256, and deduplicated against all pilot and existing items by the frozen near-duplicate policy. The pool must reach at least the simulation-required test N. The source and N are chosen **only** by the outcome-blind simulation and the hash audit.

## 4. Simulation plan

**(a) Grid.** A grid over every open N (PREREG v1 §8) using the cheap simulator: about 0.14 CPU-h per 1,000-replicate grid. **The "source" closure policy, with 44 components, is primary.** The "concept" policy, with 30, is a sensitivity analysis.

**(b) Full-refit validation at the selected cell.** The pipeline is refit for every replicate, under both null and alternative, at about 90 CPU-h. The run is on Lambda CPU or locally, whichever is faster. This satisfies Board A condition 11, "simulate the actual nested pipeline".
- If the full-refit gates fail where the cheap grid passed, take the next larger grid cell.
- If no cell passes within the §1 caps, register **unresolved, not powered**.

**(c) Restored scenario.** The scenario for sparse trust-region strata is restored and implemented.

## 5. Planner

The C6 gate selected **CEM**: sep-CMA-ES did not beat it, with a regret difference of +0.004 [−0.021, 0.029]. This is frozen.

## 6. Engineering still required before the freeze

- The 9B planner adapter.
- Bank and confirmatory run scripts for multi-GPU Lambda.
- E2 episode definitions.
- The competitor arms: Fugu-style reader team, LoRA SFT, and best-of-N steering.
- Field coverage for vocabulary-row roots.
- 27B separation (condition 13).
- Inert-edit verification on 9B, using construction items.
- The direction-similarity audit.
- P-H100.
- Manifest completion.
