# wme-c1 run4: halt at `phasel-main --resume` (fencing defect), 21 Nov 2026

Engineering halt. No unit ran and no result exists, so there is no scientific content. The registered state is unchanged apart from one `owner_epoch` record and dead-man ticks.

## What happened

| step | evidence |
|---|---|
| Phase L parity on clean `7af5724` (2×H100) | PASS: pass rule all true, `hf_gen_eq_committed` true, hf_gen `7299ac4f…` on both GPUs, 3/3 repeats; code identity pass; instance terminated and confirmed |
| main instance (1×H100) launched | 18:12:20Z; setup ok; code `7af5724`; bundle sha256 `c3da7910…`; lock reproduced; driver 570.148.08 |
| `phasel-main --resume` | opened the bank guard (new owner epoch), set the worker fence to that epoch, then sent `hello` at epoch 0 |
| worker | moved the request to `rejected/` and logged `rejected_stale`, as the R6 fence rule requires; nothing was answered |
| controller | waited for the hello answer; no progress, no incident record, no traceback |
| operator | diagnosed from the worker's `fence.json`, `rejected/`, `progress.jsonl` and `dispatch.jsonl`; killed the resume process at 18:33:13Z; termination requested 18:33:19Z and confirmed through the API at 18:40:03Z |

Study log after the halt: one `owner_epoch` record plus ticks. No `attempt_aborted`, incident or unit record was written, `attestation.json` is unchanged (B1), and all 26 bank units stay eligible for the resume. The dead-man watcher logged 0 error ticks.

## Cause

In `scripts/wme_c1_ctl.py` `cmd_phasel_resume`, the four requests after `tr.set_fence(g.epoch, inst)` (`hello`, `inert_identity`, `canary`, `timing`) carried epoch 0, copied from the first-launch path, where they run before any guard or fence exists. The v2s resume tests used a fake remote that ignored epochs, so they did not catch it. The drill drives the controller and worker directly, not this CLI path.

## Fix (v2t)

- All four requests carry `g.epoch`.
- Regression test `test_v2t_every_resume_request_carries_the_new_owner_epoch`: a fake remote rejects any request behind the worker fence with the rule `worker.Worker.handle` applies (fence epoch > request epoch → `rejected/`, never answered). It is red on `7af5724` and green with the fix.
- Every other fenced path (`cmd_phasel_main` after its guard, `cmd_close`, `Controller`, `study`, `ctl.run_canary`) already sends `guard.epoch`. The parity and first-launch calls at epoch 0 run before any fence exists.

## Note: driver

The main instance reported NVIDIA driver 570.148.08. Every earlier study instance had 580.105.08, including the parity instance of this run. The resume precondition compares the main instance's canaries (tf, hf_gen, vllm) with the committed references. A numeric effect of the driver would show there as a `canary_mismatch` incident before any unit runs.

## Follow-up FF (chair, same day): this class of hang fails fast

- **Rejected requests.** Whenever the worker refuses or aborts a request without a result, it writes `rejected/<id>.reason.json` before moving the request file. The reasons are `stale_epoch`, `fenced_while_running` and `malformed`. The controller checks for this file on every poll, in the same single round trip as the result check. Within one poll interval it raises a `request_rejected` run-health incident that names the reason.
- **Deadlines.** Every request has a response deadline per kind (`wsp.wm.c1.deadlines`).
  - Units: 8 × the guard's registered per-unit estimate, i.e. bank 2.8 h, confirmatory and S-BoN 6 h, S-TE 10 h.
  - Checks: hello 10 min; canary and timing 30 min; inert identity and parity 1 h.
  - When a deadline passes, the controller records a `request_deadline` incident, withdraws the request (R6) and halts.
- **Attempts.** Both are `IncidentAbort`: an in-flight unit attempt is recorded `attempt_aborted` (charged, never counted). The incident is written through the guard that `ctl.take_over` binds at every owner takeover. Before any guard exists (parity, first-launch checks), the failure is journalled and raised just the same.
- **Driver.** `code_and_state` records the instance's NVIDIA driver next to the parity instance's driver, in the parity, launch and resume evidence. A difference is informational; the canary comparison remains the binding test.
- **This halt under FF.** The run4 resume would have stopped within one poll (10 s) with `request_rejected` / `stale_epoch`, instead of waiting indefinitely.
