# wme-c1 run14: E2 stopped under R9.2 at `e2:wm_e:02` (unit rate 1.68 × its re-priced estimate), 24 Nov 2026

This is an engineering stop under a registered rule. No sealed file was read and no claim outcome exists. No incident is open.

## Sequence

The run used a fresh clean checkout of `eeb324d` (merged main: v3c and addendum 13) at the fixed run10 checkout path. The record is `wme-c1-run14-checkout-2026-11-23.json`. The bundle sha256 is `b02f25c1…`.

| step | result |
|---|---|
| Dead-man pre-checks l14 and l14m; preflight | pass |
| Parity, first `up` | refused before any launch: the registered test gate failed 1 test while the host was heavily loaded (load average about 25; the gate took 15 min instead of about 2). A local re-run passed 314 of 314; the gate does not keep its output, so the test is not identified. No spend |
| Parity, 2×H100 (driver 570.148.08) | gate 314 passed; PASS: every pass-rule check is true; code identity `eeb324d`; reconciled 1.339 GPU-h-eq |
| Main, 1×H100 (driver 570.148.08) | launched 00:24Z |
| Resume (generation 1) | PASS: the start canary equals the persisted reference |
| `repair-field` (K3) | all 9 field units bitwise equal (eighth instance) |
| Key check | PASS (`f210cf6f…`, `83f62b48…`, `ae9fa6e4…`) |
| `verify-e2` | PASS (seq 2918): `wm_e:00` took 720.1 s, `statnn:00` 677.5 s, both timing-free equal; factor 1.30 |
| Projection before E2 | 35.008 + 84.135 + 3.046 = 122.189 GPU-h-eq, under the cap of 128.386 |
| `run --stages e2` | the v3c fix held across the chunk-boundary canary. 21 new arm-episodes completed, all eligible, including the `e2:random:00` retry. **`e2:wm_e:02` took 1,524.3 s**; the R9.2 watch stopped the controller at 05:40:45Z, with `e2:statnn:02` in flight (seq 3082) |
| `close` | the closing canary passed; terminated, confirmed and reconciled |

## E2 rates (unit time; registered estimate 699.8 s; re-priced 909.8 s; R9.2 limit 1.5 × re-priced = 1,364.7 s)

| arm | ep 00 | ep 01 | ep 02 |
|---|---|---|---|
| `wm_e` | 1,283.7 (run10, before the memos) | 621.5 | **1,524.3 (1.68 × re-priced)** |
| `statnn` | 1,138.0 (run10, before the memos) | 606.0 | in flight at the stop |
| `random` | 508.5 | 509.8 | |
| `field_ranked` | 591.0 | 555.9 | |
| `scripted_reader` | 523.4 | 528.3 | |
| `wsr_reader` | 425.8 | 91.1 | |
| `fugu_team` | 738.7 | 788.3 | |
| `wm_greedy` | 483.0 | 456.4 | |
| `direct_bon_two_stage` | 1,215.5 | 1,229.7 (1.35 × re-priced) | |
| `bon_steering` | 454.4 | 454.4 | |
| `lora_sft` | 638.7 | 623.9 | |

**Episode 01, all 11 arms after the memos:** 6,465 s, against 11 × 909.8 = 10,008 s re-priced, so 0.65 × in aggregate. `wm_e` varies widely between episodes after the memos: 621.5 s at ep 01 and 1,524.3 s at ep 02. `direct_bon_two_stage` sits steadily at about 1.35 ×.

## Reading of R9.2 (for the chair)

- Addendum 10 P3: "if any arm's measured rate exceeds 1.5 × its re-priced estimate, E2 stops again". §II.13 item 9: "a measured unit rate above 1.5 × its registered estimate".
- The watch applied the per-unit reading, the stricter one.
- On the arm-mean reading, nothing breaches: `wm_e` averages 1,143 s over its 3 units, 1.26 × re-priced; on the run14 units alone, 1,073 s, 1.18 ×.

## Interrupted attempt

`e2:statnn:02` (seq 3082) was started and fenced at `close`, with no payload and no outcome. It is the same case as run10's `e2:random:00` (addendum 13). It is counted by the log mechanics unless it can be disposed of. The only start without an outcome other than this one is seq 2705, which addendum 13 kept conservatively.

## Ledger

- Charged 39.517 GPU-h-eq ($212.56 at the ledger convention). Dead-man: 32.10 instance-hours, $191.76. Confirmatory measured unit time: 4.538 GPU-h.
- Cap 128.386 (generation 1, P4); no further raise. No `wsp-wme-c1` instance is live.
- **Projection (P4 rule):** charged 39.517 + re-priced remainder 80.366 + overheads 3.046 = **122.93 GPU-h-eq**, under the cap.

## Addendum (24 Nov 2026): $0 diagnosis of `wm_e` 621.5 s (ep 01) against 1,524.3 s (ep 02)

The diagnosis used CPU only and read only timing, ledger and R-cache fields, the registered library index and the construction library. No outcome field was read and no planner code was changed.

| unit | unit time | dev evals (48) | R-cache GPU seconds (misses) | planner CPU outside the R-cache |
|---|---|---|---|---|
| `wm_e:01` | 621.5 s | 442.6 s | 23.2 s (137) | about 155 s |
| `wm_e:02` | 1,524.3 s | **870.4 s** | **480.4 s (3,168)** | about 156 s |

**Cause: legitimate workload variation by primitive family (registered episode design), not a defect.**
- The 24 episodes cycle through three families: `steer`, `proj_remove`, `lora_tv`. Episodes 2, 5, 8, 11, 14, 17, 20 and 23 are `lora_tv`.
- A `lora_tv` primitive touches **16 weight tensors** (64 distinct per episode). A `steer` or `proj_remove` primitive touches 1 (6 distinct per episode).
- R(E) is cached per tensor and per exact action set on it, so `lora_tv` compositions multiply the misses: 3,168 against 137, and 480 s of GPU contraction against 23 s.
- Each dev evaluation applies the bundle and measures displacement norms over 16–64 large tensors, so evaluations take about twice as long (870 s against 443 s for 48).
- The planner's CPU time and the P2 memos are unchanged. Host contention was not involved: the planner runs on the instance and the GPU was busy throughout.

**Prediction for the other arms on `lora_tv` episodes.** Each arm's ep 00/01 time splits into evaluations, R-cache and the rest. On `lora_tv` episodes, evaluations scale by 870.4 / 442.6, and R(E)-using arms add the R-cache seconds observed for `lora_tv`. The model reproduces `wm_e:02` (1,506 s predicted, 1,524.3 s observed).
- `statnn`: about 1,520 s. `field_ranked`: about 1,440–1,454 s. Both are above the unchanged 1,364.7 s limit.
- `direct_bon_two_stage`: about 1,355–1,369 s, at the limit.
- All other arms: about 840–1,010 s.

A re-pricing of `wm_e` alone would therefore most likely stop E2 again at the next unit, `e2:statnn:02`.

**In-flight attempt.** `e2:statnn:02` (seq 3082): no payload, no outcome. It is kept on the conservative count (addendum 13), with 1 of 3 attempts used.
