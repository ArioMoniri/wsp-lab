# wme-c1 run6: halt at the bank stage (order incident) and an idle termination, 22 Nov 2026

This was an engineering halt: no unit attempt started under generation 1, and no field was collected. The study holds two open incidents: `order` (seq 338) and `worker_error` (seq 340).

## Sequence (clean checkout of `2fa4fa7`; bundle sha256 `8e0375d6…`)

| step | result |
|---|---|
| Parity, 2×H100 (driver 580.105.08) | PASS: every pass-rule check true, including `hf_gen_eq_committed`; hf_gen `7299ac4f…` on GPU0, GPU1 and 3/3 repeats; code identity pass; gate 248 passed; projection 96.42; terminated and confirmed |
| Main instance, 1×H100 (driver 570.148.08) | launched 02:01Z |
| `phasel-main --resume` (generation 0, pre-transition) | PASS: inert identity; resume canaries hf_gen, tf and vllm equal the committed references and the parity instance; start canary passed (owner epoch 14) |
| `field-timing` | C4 vocabulary probe passed under the deterministic context (104 s). Slowdown, deterministic over warm registered: total 1.012 (wikitext 0.923). C6: deterministic mmlupro g0 = `af39a617…`, equal to T2's on another instance. C5: 86.0 + 10.972 + 0.003 = 96.98 GPU-h-eq, below 110, so no R8 raise |
| `transition` | generation 1, contract `8f38d82c…` (seq 326); incident seq 277 resolved; start canary passed with the reference unchanged |
| `run --stages bank` | refused before any attempt: **`order` incident, seq 338** |
| Instance | terminated by the dead-man watcher as **idle**: requested 02:36:36Z, confirmed 02:41:45Z |
| `close` (operator) | its canary found the instance gone (`InstanceLost`), which recorded a `worker_error` incident (seq 340, K2) |
| Ledger | the instance was reconciled at its authoritative 0.677 h; study ledger = dead-man total = 11.146 GPU-h-eq |

## Cause 1: the §II.18 order monitor is not scoped to the contract generation (code defect)

The run-health monitor keeps its §II.18 stage pointer in `runs/wme-c1/health/state.json` (`"stage": 1`, which is `bank_states`). The generation-0 bank had reached the bank-state units before the run3 halt. The T4 transition superseded every generation-0 unit but left the pointer where it was. The first generation-1 unit, `field_group` (stage `bank_field`), was therefore "going back" and was refused with an `order` incident before `guard.begin`, so no attempt was used.

A fix needs the stage pointer (and its skipped-stage declarations) to be scoped to the current contract generation, so that a new generation starts at the first stage. That is a code change, so it needs a new clean commit and a parity run on it.

## Cause 2: the heartbeat job had no Python (operator setup error)

`scripts/wme_c1_hb.sh` runs `<worktree>/.venv/bin/python scripts/lambda_deadman.py`. The run4 worktree had a `.venv` symlink to the repository's environment. The run5 and run6 worktrees did not, because the operator created them without it.

Every heartbeat loop therefore failed ("No such file or directory"), and the dead-man's idle clock was never reset. The watcher terminated the main instance as idle 35 min after launch. It logged no error event, because the failure was in the heartbeat job, not in the watcher.

The run5 main, diagnostic and T2 instances escaped this only because each finished in about 30 min.

**Fix (operational).** Create the `.venv` link in every run worktree, and check that the heartbeat log shows "heartbeat (progress)" before polling starts. A hardening of `wme_c1_hb.sh` itself, so that it fails loudly or uses the environment's Python, is advisable.

## State

- Contract generation 1 (`8f38d82c…`) is current. `contract_state` = `current`, so no further transition is needed.
- No generation-1 unit exists, and the field is not collected. The next bank stage collects it fresh, and the C5 field gate has nothing to verify.
- Two open incidents (`order` seq 338, `worker_error` seq 340) must be resolved, with a diagnosis and the chair's report named, before any unit.
