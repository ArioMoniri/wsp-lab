# wme-c1 run12: halt at the first guard after the P4 cap raise (code defect), 23 Nov 2026

This is an engineering halt. No E2 unit was issued, no sealed file was read and no claim outcome exists. No incident is open: incident 2750 was resolved at seq 2799.

## Sequence

The run used a fresh clean checkout of `8e2e9b9` at the fixed run10 checkout path, `.worktrees/run/wme-c1-run10` (addendum 11). The `.venv` and `data` links are recorded in `wme-c1-run12-checkout-2026-11-23.json`. The bundle sha256 is `96a4ca20…`, code `8e2e9b9`.

| step | result |
|---|---|
| Dead-man pre-checks l12, l12m, l12m2; preflight | pass (no error events) |
| Parity, 2×H100 (after about 2 h of polling for capacity) | PASS: every pass-rule check is true, including `hf_gen_eq_committed`; code identity `8e2e9b9`; terminated and confirmed |
| Main, first launch (us-south-3) | the remote setup failed (`SETUP_FAILED:cuda-runtime`: the node's CUDA runtime did not come up); terminated and confirmed by the `up` script before any request; a new dead-man pre-check (l12m2) followed |
| Main, second launch (us-south-3, driver 570.148.08) | setup ok |
| Resume (generation 1) | PASS: the start canary equals the persisted reference |
| `repair-field` (K3) | all 9 field units bitwise equal (sixth instance) |
| `resolve-incident 2750` | resolved (seq 2799), citing addendum 11 |
| Key check | PASS: the recomputed keys equal the logged keys (`f210cf6f…`, `83f62b48…`, `ae9fa6e4…`) |
| **`verify-e2`** | **PASS** (seq 2803): `e2:wm_e:00` and `e2:statnn:00` equal their eligible payloads without timing fields (timing-free sha256 `d44a0bb0…` and `780ee066…`). The re-runs took 716.9 s and 677.2 s (run10: 1,283.7 s and 1,138.0 s). The `wm_e` rate is 1.024 × registered, so the P3 factor is 1.30 |
| `reprice --factor 1.30` | seq 2806 (confirmatory: the E2 part 51.609 GPU-h-eq; re-priced price 69.149) and seq 2808 (S-BoN, whole phase: 13.911) |
| **`cap-raise`** | **cap generation 1, seq 2812, rule P4: cap 128.386 GPU-h-eq** (+1.97 % against the approved 125.90; +3.68 % against 123.83). Basis: charged 29.530 + re-priced remainder 84.138 + overheads 3.046 = projection 116.714 |
| Dead-man re-base (the `up` rule) | policy `max_usd` 526.9 → 628.41, `t_max_hours` 105.0 → 122.46; watcher restarted on base `t_max` 128.386; 4 clean ticks, 0 error events |
| **`run --stages e2`** | **failed at guard construction:** `ValueError: a cap generation never lowers the cap (110.0 < 128.386)` |
| Instance | terminated through the dead-man API and confirmed. In the study log it was closed at its API-confirmed termination time and reconciled (`reconcile_instance`, dead-man source), so its charge stops growing: 1.631 GPU-h-eq |

## Cause: a code defect (since the cap-generation code; first reached by the P4 raise)

`ProductionGuard.__init__` calls `ledger.record_cap(cap)` with the registered §II.31 cap generation (110) on every construction.
- Before any raise this was an idempotent no-op.
- After a raise, `record_cap` refuses to record a lower cap, so no guard can be constructed: not for E2, not for `close`, not for any later stage.
- The same failure would have followed any R8 raise. No raise had happened in production before seq 2812.
- My v3a tests built a guard after a re-pricing, but not after a raise.

`ctl close` also constructs a guard, which is why I terminated the instance through the dead-man API instead. No unit had executed since the resume's start canary, so under R2 no closing canary was owed.

## Proposed fix (v3b; branch `run/wme-c1-9b-v3b`, not launched)

The constructor re-records the registered cap only if no persisted cap generation equals it field for field (`_cap_persisted`). An unpersisted lower cap is still refused.

- Tests (`tests/test_wme_c1_v3b.py`): a guard after a P4 raise and after an R8 raise; the registered cap recorded once; an unpersisted lower cap refused.
- Checked on a copy of the real study log: the guard constructs, no cap record is written, the cap in force stays 128.386, and the re-priced confirmatory price is 69.149.

A code change needs a new commit and parity before the run resumes.

## Ledger

- Charged 30.214 GPU-h-eq in the study log. The first main instance (setup failure, about 26 min on 1×H100) is not yet reconciled; that happens at the next resume.
- Dead-man: 23.91 instance-hours, $146.49.
- Cap 128.386 (generation 1, P4). No `wsp-wme-c1` instance is live.
- **Projection (P4 rule, at this ledger):** charged 30.214 + re-priced remainder 84.138 + overheads 3.046 = **117.40 GPU-h-eq**, under the cap of 128.386. The unreconciled first main instance adds about 0.44, and the next launch's parity and setup are inside the overhead term.
