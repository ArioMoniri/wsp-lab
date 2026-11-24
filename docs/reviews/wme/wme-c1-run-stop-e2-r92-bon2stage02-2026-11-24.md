# wme-c1 run15: E2 stopped under R9.2 at `e2:direct_bon_two_stage:02` (4,892.7 s; limit 2,286.5 s), 24 Nov 2026

This is an engineering stop under a registered rule. No sealed file was read and no claim outcome exists. No incident is open.

## Sequence

The run used a fresh clean checkout of `c973cc4` at the fixed run10 checkout path. The records are `wme-c1-run15-checkout-2026-11-24.json` and `wme-c1-run15-deviation-2026-11-24.md`.

| step | result |
|---|---|
| Dead-man pre-checks l15 and l15m; preflight | pass |
| Parity, 2×H100 | gate 314 passed (output kept by `ctl_gate_tee.py`); PASS: every pass-rule check is true; code identity `c973cc4`; reconciled 0.859 |
| Main, 1×H100 | launched 06:49Z; resume passed |
| `repair-field` (K3) | all 9 field units bitwise equal (ninth instance) |
| Key check | PASS |
| `verify-e2` | not re-run: `require_p2_verified` accepts seq 2918 under `same_code` (evidence-only changes since `7509e4d`) |
| E2 | 8 more arm-episodes eligible (31 in all): episode 02 `random`, `field_ranked`, `scripted_reader`, `wsr_reader`, `fugu_team`, `wm_greedy`, `statnn` (1,378.0 s) and `direct_bon_two_stage`. **`e2:direct_bon_two_stage:02` took 4,892.7 s** against its addendum-14 `lora_tv` limit of 2,286.5 s. The 30 s watch stopped the chain at 11:04:02Z, with `e2:bon_steering:02` in flight (seq 3217) |
| `close` | the closing canary passed; terminated, confirmed and reconciled |

## $0 diagnosis (timing fields only)

`direct_bon_two_stage` screens 512 candidates on 4/64 of a bundle each, then runs 16 full evaluations.

| unit | unit time | 512 screens (sum; median) | 16 full evaluations |
|---|---|---|---|
| `:00` (`steer`) | 1,198.7 s | 1,052.3 s; 1.93 s | 143.8 s |
| `:01` (`proj_remove`) | 1,212.6 s | 1,065.6 s; 1.95 s | 144.5 s |
| `:02` (`lora_tv`) | 4,873.6 s | **4,626.0 s; 8.19 s** | 245.1 s |

A screen evaluates only 4/64 of a bundle, so its fixed per-candidate cost dominates: applying up to 4 primitives of 16 tensors each, hashing theta, and computing displacement norms over every touched tensor. On `lora_tv` that cost is about 4.2 × higher. This is legitimate registered workload, not a defect. The addendum-14 workload model treated this arm as evaluation-proportional (about 1,360 s predicted); that model is withdrawn by addendum 15.

## Price under the addendum-15 rule (for the chair)

The rule: estimate = max(909.8 s, the arm's own post-memo episode-2 seconds); `wm_e` 1,524.3 s.

- Episode-2 measured seconds: `wm_e` 1,524.3, `statnn` 1,378.0, `random` 1,060.5, `field_ranked` 942.4, `scripted_reader` 1,337.5, `wsr_reader` 211.9, `fugu_team` 1,270.3, `wm_greedy` 1,020.9, `direct_bon_two_stage` 4,892.7. `bon_steering` and `lora_sft` are not yet measured.
- `lora_tv` excess over episodes 5–23 for the measured arms: **11.955 GPU-h-eq**, against the 5.29 already counted. `direct_bon_two_stage` alone accounts for 7.74.
- **Projection:** charged 44.680 + log remainder 77.001 + `lora_tv` excess 11.955 + canaries 0.716 = 134.35; with one relaunch (P4 launch term 2.674) **137.03 GPU-h-eq, above the cap of 128.386**. The two unmeasured arms are not included.

## Attempts without an outcome

seq 2705 `e2:random:00`, seq 3082 `e2:statnn:02` and seq 3217 `e2:bon_steering:02` are each kept on the conservative count (addenda 13 and 14).

## Ledger

- Charged 44.680 GPU-h-eq ($240.33 at the ledger convention). Dead-man: 36.84 instance-hours, $216.92.
- Cap 128.386 (generation 1, P4); no R8 record. No `wsp-wme-c1` instance is live.
