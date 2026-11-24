# wme-c1 run16: E2 stopped under R9.2 at `e2:field_ranked:05` (1,638.8 s; addendum-15 limit 1,413.6 s), 24 Nov 2026

This is an engineering stop under a registered rule. No sealed file was read and no claim outcome exists. No incident is open.

## Sequence

The run used a fresh clean checkout of `41a9a9f` at the fixed run10 checkout path (`wme-c1-run16-checkout-2026-11-24.json`).

| step | result |
|---|---|
| Dead-man pre-checks l16 and l16m; preflight | pass |
| Parity, 2×H100 | gate 314 passed (output kept); PASS: every pass-rule check is true; code identity `41a9a9f`; reconciled 0.869 |
| Main, 1×H100 | launched 15:03Z, after about 2.7 h of polling for capacity; resume passed |
| `repair-field` (K3) | all 9 field units bitwise equal (tenth instance) |
| Key check | PASS |
| `verify-e2` | not re-run (`require_p2_verified` accepts seq 2918 under `same_code`) |
| E2 | 28 more arm-episodes eligible (59 in all), with no breach through episodes 2 (rest), 3 and 4 and episode 5's `wm_e` (1,416.9 s), `statnn` (1,231.9 s) and `random` (1,222.0 s). **`e2:field_ranked:05` took 1,638.8 s** against its addendum-15 limit of 1,413.6 s (1.5 × its episode-2 seconds, 942.4 s). The watch stopped the chain at 21:06:52Z, with `e2:scripted_reader:05` in flight (seq 3508) |
| `close` | the closing canary passed; terminated, confirmed and reconciled |

## $0 diagnosis (timing fields only)

| unit | unit time | dev evals (48) | R-cache seconds (misses) | planner seconds |
|---|---|---|---|---|
| `field_ranked:00` | 567.3 s | 449.2 s | 51.4 s (284) | 118.0 s |
| `field_ranked:01` | 524.8 s | 442.2 s | 24.8 s (128) | 82.5 s |
| `field_ranked:02` | 928.0 s | 778.5 s | 64.9 s (496) | 149.5 s |
| `field_ranked:05` | 1,620.3 s | 691.3 s | **840.6 s (6,880)** | 929.0 s |

The dev-evaluation cost on `lora_tv` repeats (691 s against 778 s). The difference is R(E): on episode 05 the field-ranked planner reached 6,880 distinct per-tensor compositions, against 496 on episode 02. A `lora_tv` primitive touches 16 tensors, so the number of distinct (tensor, action-set) contractions depends on which primitives the episode's subsample holds and how the search combines them. One episode-2 measurement per arm did not represent this arm's episode-5 work. `wm_e:05` (3,952 misses) and `statnn:05` (3,520 misses) rose too, but stayed under their limits. This is legitimate registered workload, not a defect.

## Attempts without an outcome

seq 2705 `e2:random:00`, 3082 `e2:statnn:02`, 3217 `e2:bon_steering:02` (now eligible) and 3508 `e2:scripted_reader:05` are kept on the conservative count.

## Ledger and projection

- Charged 51.704 GPU-h-eq ($278.12 at the ledger convention). Dead-man: 43.43 instance-hours, $251.18.
- Cap 128.386; no R8 record. No `wsp-wme-c1` instance is live.
- **Projection (addendum-15 rule, from the ledger):** charged 51.704 + log remainder 71.913 + `lora_tv` excess 11.604 + canaries 0.660 = 135.88; with one relaunch (2.674) **138.55 GPU-h-eq**.
- **Relaunch overhead since the run10 stop** (instance charge − unit time, all 13 instances from 23 Nov 05:48Z): 26.315 − 12.226 = **14.090 GPU-h-eq**.
