# wme-c1 run17: main instance terminated idle by the dead-man after a passing resume; operator error (27 Nov 2026)

**Sequence (UTC).**

| time | event |
|---|---|
| 18:10:57 | parity l17 launched, after about 10 h 53 min of polling |
| 18:37:36 | parity l17 PASS (code `7e2ae32`); terminated and confirmed |
| 18:37–18:39 | main dead-man pre-check l17m passed; `main.sh` started |
| 18:47:08 | main launched (us-southeast-1) |
| 19:10:15 | `phasel-main --resume` PASS (the start canary equals the persisted reference) |
| 19:10–20:10 | no request reached the instance |
| 20:10:24 | the dead-man terminated the main instance for idleness (idle limit 1,800 s) |
| 20:11:25 | termination confirmed |

**Cause (operator error).** I started the background waiter for the resume in the same step as `main.sh`, before its log file existed. `L=$(ls -t …wmec1main17-*.log)` was therefore empty, and the waiter's `grep -q 'M DONE' $L` read an empty stdin and could never succeed. I was not woken at 19:10, so the chain (`repair-field`, the key check, E2) never started. The stall alarm did not fire because it watches the chain session, which never started.

**Consequences.**
- No unit ran, no incident record was written, and no attempt was used.
- The dead-man acted as the safety net.
- About 1.4 h of 1×H100 was spent without use: 1.405 GPU-h-eq.
- In the study log the instance was closed at its API-confirmed termination by the ledger's `reconcile_instance` (dead-man source), as accepted in addendum 12. That stops its charge growing.
- Charged: 53.999 GPU-h-eq.

**Fix (operator scripts only).** Every log-watching waiter first waits until its log file exists, then resolves the path. The stall alarm also covers a resumed main instance with no chain running.

**Next.** Per the chair's ruling of 27 Nov (no parity reuse), a new main launch needs a new parity. No relaunch is made without a ruling.
