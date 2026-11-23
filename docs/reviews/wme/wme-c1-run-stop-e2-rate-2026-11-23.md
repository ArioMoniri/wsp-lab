# wme-c1 run10: bank complete; E2 stopped under R9.2 (measured unit rate above 1.5 × its registered estimate), 23 Nov 2026

This is an engineering stop under a registered rule. No sealed file was read and no claim outcome exists. No incident is open.

## Sequence (clean checkout of `2b1cc17`; bundle sha256 `be8d6d3b…`)

| step | result |
|---|---|
| Dead-man pre-checks l10 and l10m; preflight | pass |
| Parity, 2×H100 (driver 570.148.08) | PASS: every pass-rule check true, including `hf_gen_eq_committed`; hf_gen `7299ac4f…` on GPU0, GPU1 and 3/3 repeats; code identity pass; gate 289 passed; registered projection 108.8; terminated and confirmed |
| Main, 1×H100 (driver 570.148.08) | launched 22:18Z |
| Resume (generation 1) | PASS: canaries equal the committed references |
| `resolve-incident` 2356 | resolved (record 2392), citing addendum 8 |
| `record-repair` | seq 2394: `inert:05:comp`, `code_defect_crash`, failure seq 2358, citing addendum 8 |
| `repair-field` (K3) | all 9 field units bitwise equal (fourth instance) |
| **Bank** | **complete: 210 of 210 units eligible under generation 1, 0 fallbacks.** Bank bundle `ba1b8416…` (1,026 states, 0 fallbacks; component graph `c8d50fe2…` equals the frozen manifest's). Bank-phase measured unit time 6.035 GPU-h (resumed, addendum 7) |
| Compensated inerts | all endpoint-bitwise equal to the base. `r_omitted`: `inert:05:comp` → `model.layers.12.mlp.up_proj.weight`; `inert:06:comp` → `model.layers.15.mlp.up_proj.weight`; `inert:07:comp` → `model.layers.15.self_attn.v_proj.weight` (the attention compensated pair: **v_proj, not up_proj**) |
| Prep | library 2.3 min, dev_base 0.4 min, fit 2.9 min; eligible |
| E2 | `e2:wm_e:00` 21.4 min and `e2:statnn:00` 19.0 min, both eligible; `e2:random:00` in flight when the controller was stopped (fenced at `close`; one started attempt, no outcome) |
| `close` | closing canary passed; instance terminated, confirmed and reconciled |

## Why E2 was stopped: R9.2 (§II.13 item 9)

> "A measured unit rate above 1.5× its registered estimate is re-priced by dated amendment before any data it governs (R9.2)."

- **The registered estimate of an E2 arm-episode** comes from the registered price (`powersim_v1.price`, v2b): B × t_bundle + planner_gpu_s. That is 48 × 14.580 s (`t_bundle_s`, `wme-powersim-v2b-2026-11-17.json`) + 0, which is **699.8 s (11.66 min)**.
- **Measured:** `wm_e:00` 1,284 s (**1.84 ×**); `statnn:00` 1,140 s (**1.63 ×**).

E2 has 264 arm-episodes (24 episodes × 11 arms). At the measured primary-arm rate, E2 alone would need roughly 85–90 GPU-h. The registered confirmatory price is 53.67 GPU-h, which pauses at 1.25 × = 67.1 GPU-h. Continuing would have generated the governed data before the amendment.

## State and ledger

- **Confirmatory phase:** measured unit time 0.766 GPU-h; 5 units eligible (prep and two E2 arm-episodes).
- **Ledger:** study = dead-man = **25.389 GPU-h-eq ($123.06)**. Cap 110, cap generation 0, no R8 raise. No `wsp-wme-c1` instance is live.
- **Projection at the measured rates (for the re-pricing):** about 25 spent + about 85–90 for E2 + S-BoN + sealed + E1 generation + S-TE. That is well above 110 GPU-h-eq; the absolute $1,000 limit corresponds to about 204 GPU-h-eq.
