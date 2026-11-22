# wme-c1 run8: bank-phase PhasePause (1.25 × the priced bank), 22 Nov 2026

This is a registered pause, not an incident. No sealed file was read and no claim outcome exists. No incident is open.

## Sequence (clean checkout of `b2f1f71`; bundle sha256 `d2c2b6e1…`)

| step | result |
|---|---|
| Dead-man pre-checks l8 and l8m; preflight | pass |
| Parity, 2×H100 (driver 580.105.08) | PASS: every pass-rule check true, including `hf_gen_eq_committed`; hf_gen `7299ac4f…` on GPU0, GPU1 and 3/3 repeats; code identity pass; gate 273 passed; projection 100.28; terminated and confirmed |
| Main, 1×H100 (driver 570.148.08) | launched 09:39Z; heartbeat gate met |
| Resume (generation 1) | PASS: canaries equal the committed references; start canary passed |
| `resolve-incident` 569 | resolved, citing addendum 6 |
| `repair-field` (K3) | **all 9 field units bitwise equal** (g0–g7, vocab), re-collected deterministically on a new instance and driver against the generation-1 hashes recorded in run7; repair record written |
| Bank (generation 1) | 156 units in run8, with every chunk canary passing. v2x confirmed: at a closing canary the parent held 0.12 GiB reserved (run7: 23.7 GiB), with 78.3 GiB free against 60 GiB (HF) and 69.7 GiB (vLLM) required |
| `PhasePause` | raised before the next unit: `bank: above 1.25 x its priced 4.530551290639242 GPU-h` |
| `close` | closing canary passed, so the 12 results since the previous canary are bracketed; instance terminated, confirmed and reconciled |

## Ledger seconds (the basis of the chair's decision, §II.13 item 9, §II.14)

- **Bank-phase measured unit GPU time: 5.320 h.** Of this, generation 0 used 0.873 h (26 units, superseded) and generation 1 used 4.447 h (182 successful attempts: 166 eligible units plus run7's 16 retained-ineligible sides).
- **Pause threshold:** 1.25 × 4.5306 = 5.663 GPU-h. 5.320 h plus the next unit's estimate (0.35) exceeds it.
- **Bank units:** 210 registered; **166 eligible** under generation 1; **44 remaining**, each with 0 counted generation-1 attempts: 12 `bank_side`, 24 `bank_comp`, 8 `bank_inert`.
- **Combined ledger:** study = dead-man = 19.229 GPU-h-eq ($93.14). Cap 110, cap generation 0, no R8 raise.

## Resuming

The field must be re-materialised on the next instance by K3. Run8 showed this is bitwise across instances and drivers.
