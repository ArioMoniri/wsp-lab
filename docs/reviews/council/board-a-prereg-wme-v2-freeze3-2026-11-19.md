# Board A — third WME v2 freeze re-check (v2i)

**Verdict: ACCEPT WITH CONDITIONS.** The stale-object freeze blocker B is closed for the serialized production path. A and changes 1–5/7 remain closed at their previously accepted scope. Acceptance requires the registration to state the supported concurrency and log-integrity boundaries below; the multi-instance deployment is a **launch precondition**, not a demand to build a distributed execution system before freezing the scientific plan. This is not authorization to launch four independent guards or to share one guard between concurrent threads.

Reviewed public main **70906cd721e3a01253958ffcc9ba25cb53039d1b**, 19 November 2026 UTC, as Board A, independent Codex methods seat. Scope: §II.27, `runguard.py`, `harness.py`, the regression/property tests, mutation scripts/reports, Update v2i, and targeted regressions for A and changes 1–7.

Only this review file was written in the repository. Probe scripts, reports and temporary fixture state were outside it. All Python verification used `CUDA_VISIBLE_DEVICES=''` and `PYTHONDONTWRITEBYTECODE=1`; pytest caching was disabled. Long jobs ran through `agent-tmux`. No sealed contents or `data/evals` were opened; no GPU, Lambda, secrets, commits, panel rebuilds or source-pool reconstruction were used. Numbers describing probes below are CPU enforcement fixtures, not study outcomes or launch attestations.

## Verification and closure of B

**105 tests passed, 1 slow test deselected, in 56.82 seconds**, with the existing constant-input correlation warning. Invocation: `.venv/bin/python -m pytest -q -p no:cacheprovider` on:

```text
tests/test_wme_e2run.py       tests/test_wme_panels_v2d.py
tests/test_wme_e2sens.py      tests/test_wme_guards_v2.py
tests/test_wme_runguard.py    tests/test_wme_sbon.py
tests/test_wme_prereg_v2.py   tests/test_wm_exposure.py
tests/test_wm_harness.py      tests/test_wm_p3gen.py
tests/test_wm_signflip.py     tests/test_wme_runguard_property.py
```

The invocation used `-m 'not slow'` and includes the default **60 sequences × 40 steps**, the original clock/contract/loss probes, all `test_probe_freeze2_*` tests, concurrent-process locking, and torn/truncated-log repair. Its temporary storage was disk-backed. No full-suite or confirmatory-stack run is claimed.

Independent public-API continuations used a retained parent guard while a forked child acquired ownership and separately performed (a) A → B → A transitions, (b) a loss disposition for the same instance id, or (c) three classified failures through `harness.run_units` into fallback. In every case the retained parent refused canary, resolve, harness execution, ledger mutation and retry allocation with `StaleGuard`; no callback ran and the study-log bytes were unchanged. Raw sequence numbers remained gap-free. These cross-process checks supplement the deterministic sequential reproductions in the named tests.

I also independently verified:

- Takeover inside a successful unit callback prevents the old owner from recording its result. The attempt remains counted; no result file or `attempt_ok` is admitted.
- Smaller/larger/smaller authoritative reconciliation never reduces the charge: a local ten-hour record remains ten after a one-hour correction, increases to twenty, then stays twenty after a two-hour correction.
- Replacing an attempt payload through `UnitStore.write`, including a valid replacement store checksum, is rejected by `resolve` with `ResultMismatch` against the study-log digest.

The structural closure is substantive: owner epoch and contract checks occur after refresh under the same OS lock as mutation; retry allocation, lifecycle, canary evidence and result selection derive from one global log. The three freeze2 public-API stale-snapshot failures no longer reproduce. An owner replaced during computation cannot publish eligible output. This does not imply that ownership takeover physically cancels an already running remote computation; termination/accounting remains a deployment obligation.

## Property oracle, coverage and mutations

Independent rerun of `scripts/wme_v2i_runguard_property.py --n 5000 --procs 5` completed **5,000 × 40 steps**, seeds **7,000,000–7,004,999**, with **zero failures in 96.0 seconds**. It used `TMPDIR=/dev/shm`; it exercises file operations and replay, not deployed-disk crash durability. An earlier disk-backed long run was interrupted to switch storage and is not counted as completed. The authors' disk-backed report remains separately identified evidence, not my independent disk-backed long-run result.

Coverage exactly reproduced the checked-in report: 18,508 unit executions; 206,224 checked returned results; 6,967 eligible, 1,349 ineligible and 4,438 superseded attempts; 2,507 fallbacks; 261 disposals; 3,177 losses; 5,523 transitions including 1,255 returns to an earlier digest; 22,186 owners; **16,048 stale refusals without log changes**; **1,300 mid-unit takeovers**; 917 backward clock moves during units; 6,147 clock halts and 5,944 recoveries; 2,512 torn-log refusals and 1,797 audited repairs; 26 cap-reaching sequences.

The oracle is materially more independent than v2h. `Model` parses raw bytes, checks canonical hashes/sequences, and independently derives current owner, generation, lifecycle, charges, brackets and first eligible/fallback selection without calling runguard's semantic helpers. I5 now checks `None` against an expected absence as well as positive results. Retained old objects, takeover during computation and torn-tail recovery directly address the previous coverage gap. I6 still includes an implementation-versus-implementation snapshot comparison, but adds model comparisons for important semantic fields; it is not the only oracle.

Limits remain: this is seeded testing, not an exhaustive proof or a fully independent event-generation specification. It uses one phase, one-GPU fixtures, constant canary references and truthful recovery records, not four live remote instances, varying per-engine references, arbitrary reconciliation data or power-loss fault injection. Its concurrency coverage is separate from the random sequences. Shared-object threads, in-flight budget reservations and earlier cached-prefix corruption are outside the generated operations. Positive coverage counts establish reached cases, not every ordering.

I independently reran `scripts/wme_v2i_property_mutations.py --n 30` in all three modes, seeds from 9,000,000; completed in **133.4 seconds**:

- Unmutated: all 30 sequences per mode pass; four concurrent processes produce **1,200 verified records and 1,200 charged seconds**.
- m4, no lock/no refresh: detected at sequence 1 in every mode. Without I0 the detection is the implementation's `LogCorrupt` during I6 replay, not independent result semantics.
- m5, no epoch fence: detected at sequence 1 with the explicit fencing assertion and sequence 26 in both semantic modes by I2's stale-owner execution assertion.
- m6, no OS lock: undetected in all 30 single-process sequences per mode; the concurrent-process check detects a broken chain at record 3, with all four writers failing.

These reproduce the reported detection mechanisms. I did not independently repeat all 300 controls per mode. Mutation sensitivity establishes sensitivity to these removals, not universal completeness or distributed safety.

## Conditions on the frozen wording

Two new bounded probes show that §II.27's universal claims are broader than the implementation. **Record these restrictions in the frozen registration; alternatively repair and re-check the broader guarantees.** There is no additional executable blocker for the serialized, append-only operating scope accepted here.

1. **Calls sharing a `StudyLog` object must be serialized, including across threads.** `StudyLog.locked` uses object-wide `_depth` (`runguard.py:370–385`), not a thread-aware reentrant mutex. Another thread can see nonzero depth and skip acquisition/refresh. In three independent trials, four threads each calling the same bare ledger's public `add` twenty times produced `LogCorrupt` or `FileNotFoundError` on the shared head temporary file. This does not contradict the passing separate-process/separate-object tests; it does contradict an unrestricted “every public operation is serialized” claim. One owner is not itself one thread. For this acceptance, require one serialized controller call stream and no shared-object concurrent entry. If shared-object threads are intended, a thread-aware lock and regression are required before that use.
2. **Cached refresh verifies the last cached line and appended tail, not the whole cached prefix.** `_refresh` (`:415–440`) does not reread earlier cached records. After an eligible fixture result, I changed an early owner-record byte without changing its length or the last line. The existing guard still returned its cached eligible result; a fresh `read_study` raised `LogCorrupt`. Narrow “any changed already-read record is detected on every operation” to the actual checks. Treat immutable-prefix storage as an operating assumption, perform verified fresh replay on restart/recovery, and do not describe hash chaining as protection against coordinated rollback of the log and head. Full-prefix checking on every call is optional hardening for this review's storage-damage scope, not a new stale-object blocker.

The current scientific claims and analysis need no change. These are required specification corrections, not production attestations supplied by this review. Unqualified ACCEPT FOR FREEZE would overstate the implementation until these boundaries are recorded.

## Explicit ruling on up to four instances and the combined cap

**Launch precondition, not inherently a freeze blocker.** “Up to four” permits serialized execution on one instance; scientific freeze does not require a distributed lock implementation. A single controller on one host with a genuinely local authoritative log is a defensible deployment architecture. The existing API does not establish safe parallel dispatch merely because that host owns the log: each guard binds one phase/instance, and constructing another guard revokes the first, including its in-flight completion rights.

A further public-API probe illustrates the budget gap for parallel dispatch: at 89 charged fixture GPU-hours with a 0.6-hour estimate, `begin('u1', ...)` and then `begin('u2', ...)` both succeed. They each check 89 + 0.6; neither reserves its estimate for the other. This is not an over-cap execution in the ordinary serial `run_units` loop. It means **serialized accounting calls alone do not reserve parallel work**. Likewise giving each of four dead-men the full `remaining_usd()` can multiply the permitted liability.

Before launch, the registration must expressly require:

- Exactly one authoritative study log and controller ownership domain, with all guard calls serialized and all workers forbidden from independently claiming ownership. A shared path or copied state on different machines is insufficient. Cross-host writers require an equivalent tested transactional lock/fence, not an assumption that local flock provides it.
- A bound mapping of physical instances, phases, attempts, contract generations and canary references to controller records. Remote outputs are admitted only through the current owner, with stale completion refusal. Controller takeover must also stop or account for old outstanding workers.
- All previously spent bank/study time and every active instance included in the combined charge; setup, downloads, idle, canaries, failures, retries and replacements remain charged. Worker sub-ledgers are evidence to reconcile into the controller, never independent cap authorities.
- For parallel dispatch, atomic aggregate reservations or a conservatively partitioned budget covering **all** outstanding work and active-instance exposure, with bounded termination latency and coordinated per-instance dead-man allowances. Reserve liability before work starts; reconcile monotonically afterward. Retrospective sub-ledger aggregation alone cannot enforce an absolute cap.
- Fail closed on loss of controller, state integrity, ownership or accounting connectivity; retain termination/accounting evidence. Demonstrate deployment restart, takeover, worker loss and simultaneous-dispatch cap enforcement with CPU fixtures before any confirmatory outcome.
- Until that parallel adapter/protocol is implemented and checked, launch only the supported serialized execution path, with an aggregate termination budget covering any other running instance. Do not work around epochs by repeatedly reconstructing guards while older work remains active.

A proposal requiring four autonomous guards today would not satisfy this acceptance. If the authors decline the serialized launch restriction and cannot demonstrate an equivalent aggregate protocol, launch remains blocked; it must not be relabelled optional hardening. These operational rules must be outcome-blind and cannot alter eligibility, retries, fallbacks or the absolute 90 GPU-h-eq/$1,000 combined limits.

## A, changes 1–7 and remaining launch evidence

Against freeze2's `ca65219`, changed top-level manifest fields are only `generated_from_commit`, `harness.runguard` and `prereg_v2`. `audit.panels_v2e`, `bank.closure_allocation` and `exposures.v2` are unchanged. The panel-audit and root-lineage reports are byte-identical, with SHA256 respectively `a251d79880d964a94c390d5e6604f53631d44938ef2b8f20426f30aab50f5311` and `0df383ae561f81afc42cfd24b0fd337edfb44797be656ef1f10dd3810281b2cf`. A stays closed at the accepted hash-only metadata/lineage scope; no sealed text or groups were independently rederived.

The manifest SHA256 is `4b125a307cfada3d6dd6436c177b07df353820fa9ad776b348ae340ccd692fef`; its property/mutation report pins match their files and §II.27. `e2run.py`, `guards_v2.py`, `arms/sbon.py` and `panels_v2d.py` are unchanged. Targeted tests find no regression in changes 1–5/7. R6 remains descriptive, C2 is not a gate, W2 FAIL excludes H-T3, H-E2 is conditional non-symmetry, and the accepted sizing, sparse-guard and S-BoN restrictions remain in force. Change 6/B is now closed for the stated execution scope.

Pinned-engine inert identity, actual 1-vs-2-GPU parity, committed per-engine canary references, real evidence/fallback binding, unique physical-instance identities, deployed persistence/repair and authoritative time/dead-man reconciliation remain **mandatory launch preconditions**, unperformed here. `EngineeringGuard` cannot supply confirmatory evidence. Stronger thread support, full-prefix validation, independently anchored log heads and directory-fsync/power-loss testing are optional hardening under the restricted scope; they become requirements if the deployment claims those stronger guarantees.

## Four-line summary

1. All 105 targeted tests and 5,000 × 40 property steps pass; independent cross-process stale-object reproductions are fenced.
2. A and changes 1–5/7 remain closed; mutation reruns reproduce m4/m5 detection and concurrent m6 detection.
3. Register serialized-call and cached-prefix limits; multi-instance ownership, reservations and aggregate dead-men are launch preconditions.
4. **Verdict: ACCEPT WITH CONDITIONS — B closed for serialized use; no autonomous parallel launch is approved.**
