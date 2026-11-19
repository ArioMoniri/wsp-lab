# Board A — second WME v2 freeze re-check (v2h)

**Verdict: REJECT — blocker B remains open because previously constructed guards can bypass persisted transitions, loss dispositions and retry exhaustion.** The three v2g reproductions are repaired when operations use the current reconstructed guard. Blocker A and fallback identity remain closed at their previously accepted scope.

Reviewed public `main` **ca65219e72f09f3d07e3a9d20db87f3561be36e1**, 19 November 2026 UTC, as the independent Codex methods seat. Scope: PREREG §II.26, `runguard.py`, `harness.py`, the named regression/property tests, Update v2h, manifest binding, and targeted regressions for A and changes 1–7.

Only this review file was written in the repository. Probe scripts and reports used temporary locations outside the repository; fixture values below are CPU enforcement tests, not study outcomes or production attestations. No sealed split contents or `data/evals` were opened. No GPU, Lambda, secrets, commits, panel rebuilds or source-pool reconstruction were used. All Python verification used `CUDA_VISIBLE_DEVICES=''` and disabled bytecode generation; pytest's cache provider was disabled.

## Verification

**98 tests passed, 1 slow test deselected, in 166.92 s**, with only the constant-input correlation warning in the proxy-gate fixture:

```text
tests/test_wme_e2run.py       tests/test_wme_panels_v2d.py
tests/test_wme_e2sens.py      tests/test_wme_guards_v2.py
tests/test_wme_runguard.py    tests/test_wme_sbon.py
tests/test_wme_prereg_v2.py   tests/test_wm_exposure.py
tests/test_wm_harness.py      tests/test_wm_p3gen.py
tests/test_wm_signflip.py     tests/test_wme_runguard_property.py
```

Invocation: `.venv/bin/python -m pytest -q -p no:cacheprovider` with those paths and `-m 'not slow'`. This includes the default 80 property sequences and all three `test_probe_freeze_*` tests. I also invoked each of the three original freeze probes separately, together with the fallback-identity probe; all passed.

I independently ran the registered property runner for **5,000 sequences × 40 steps**, seeds **7,000,000–7,004,999**, five processes through `agent-tmux`: **passed, zero failures, 18.4 s**. The completed run used `TMPDIR=/dev/shm`; file appends, reads and reconstruction were exercised, but this is not disk-durability evidence. An initial disk-backed full-size run was stopped before completion to switch temporary storage; it is not counted as a completed run. The default pytest run above used ordinary disk-backed temporary storage.

Coverage matched the authors' report: 22,840 unit starts; 9,264 eligible, 1,798 ineligible and 6,606 superseded attempts; 4,096 fallbacks; 378 disposals; 3,066 losses; 6,568 transitions including 1,702 returns to an earlier digest; 6,333 clock halts, 6,105 recoveries and 47 cap-reaching sequences. Independent mutation rerun (`scripts/wme_v2h_property_mutations.py --n 20`, output outside the repository) detected m1, m2 and m3 at sequences **2, 9 and 7**, respectively. The separate unmutated runs, rather than the mutation script's hard-coded `unmutated` field, establish the unmutated pass.

No full-suite or confirmatory-stack run is claimed.

## Original B gaps and new probes

| Probe | Independent observation | Assessment |
|---|---|---|
| Original clock rollback credit | Closing at hour 10 after opening at hour 100 raises `ClockInconsistent`; halt persists, relaunch is blocked, clearing requires termination reconciliation and a recovered clock. A replacement open for 91 hours cannot start a unit. | Original reproduction closed. |
| Original contract A → B → A | Freshly reconstructed guards advance generations 0 → 1 → 2; generation-0 result remains superseded; generation 2 needs a new start canary. | Original reproduction closed; stale-object continuation below remains a blocker. |
| Original post-loss canary resurrection | Reconstruction using the lost id rejects both canary recording and `run_units`; the old unbracketed attempt remains ineligible and the replacement attempt is selected. | Original reproduction closed; stale-object continuation below remains a blocker. |
| Gap 4 / fallback identity | Changed fallback payload is rejected on reconstruction; the supplied resume/idempotence regression passes. | Remains closed for registered payload identity; stale exhaustion counters are a distinct failure below. |
| Clock rollback during an open unit | Start at hour 100; callback moves the clock to hour 10. With the harness measurement clock tied to that fixture clock, negative measured duration raises `ClockInconsistent`. The next canary persists a halt; `resolve` then refuses. | No eligible result escapes in this probe. `succeeded` writes the unit payload before rejecting its negative duration, but does not append `attempt_ok`; the orphan payload is not eligibility evidence. This path alone does not persist the halt until the next tick. |
| Reconciliation subtracting time | Ten local hours remain ten after a one-hour authoritative record; a 20-hour record increases charge to 20; a subsequent two-hour record leaves it 20. Termination before launch is rejected. | No subtraction found. Per-instance max(local, all authoritative durations) preserves the charge. |
| Generation manipulation through repeated reconstruction | After A → B → A, four additional reconstructions retain generation 2. A new bracketed `.a2` is selected over generation-0 `.a1`. | Reconstruction alone does not reset generation; retaining an earlier object does bypass it. |
| Canary under another object using the same id | A fresh object built after loss rejects the canary and reopening. An object built before loss accepts the canary after another object records loss. | Freeze blocker; details below. Physical provenance of a caller-supplied id still requires launch binding. |
| Resolve order across generations | A current generation-2 guard returns its new result; the retained generation-0 guard returns the old result from the same unit store and state directory. | Freeze blocker; details below. |
| Fully bracketed result before loss | Loss preserves an already bracketed pre-loss result through direct resolution, as registered; fresh same-id reopening fails. | Expected behavior preserved. |
| Partial trailing write on restart | Appending incomplete JSON separately to `events.jsonl`, `ledger.jsonl` and `retry.jsonl` causes `JSONDecodeError` on reconstruction. | Fails closed; automatic recovery would need an audited repair. |
| Complete-record log truncation | Removing the retry-log contents after one persisted unit start reconstructs an empty attempt counter despite that start remaining in the event log. | No suffix-integrity/cross-log reconciliation guarantee. Durability hardening and deployed-restart verification, not an additional freeze blocker from this storage-corruption fixture alone. |

The last two probes intentionally modified only temporary fixture logs to model storage damage. All blocker reproductions below used **public API calls only**, without editing logs, object fields, generation counters or guard internals.

## Remaining freeze blocker: stale objects are not fenced against shared persisted state

`CombinedLedger.rows` is loaded once (`src/wsp/wm/runguard.py:134–142`), as are retry counters and `ProductionGuard.events` (`:573–580`). `validate()` only calls `_require_active()` (`:655–662`); that checks the object's cached ledger and loss events. It neither reloads the current contract nor rejects a stale generation. `resolve`, `begin` and `canary` also operate on cached state. `_seq()` allocates sequence numbers from that object's snapshot (`:632–646`), allowing overlapping objects to append duplicate sequence numbers.

These probes require no simultaneous execution or timing race: construct two objects, perform operations sequentially, then invoke the older object. There is no enforced exclusive ownership or documented, executable invalidation mechanism making the old object unusable.

### B2 continuation: a persisted transition does not supersede an older guard's view

1. Construct `g0` for contract A, open `i1`, pass its canary, compute a unit with fixture payload `{"generation": 0}`, and pass its closing canary.
2. Construct `g1` in the **same state directory** under contract B with a valid outcome-blind technical transition.
3. Construct `g2` there under A with another valid transition. Reconstructing it repeatedly keeps generation 2; its old result is superseded. Compute and bracket a new result with payload `{"generation": 2}`; its current resolution selects that result.
4. Call `harness.run_units` using the retained **g0**, same store and same unit. A callback that raises if called is never called. The returned value is **`{"u1": {"generation": 0}}`**.

Thus returning the old guard to the actual harness bypasses both supersession and current-generation first-valid selection. The added `validate()` call does not provide the contract validation claimed in §II.26. This is a runtime freeze blocker, not a request for real-engine launch evidence.

### B3 continuation: a retained same-id guard accepts post-loss closing evidence

1. Construct `live`, open `i1`, pass a start canary, compute `.a1`, and leave it pending.
2. Construct `other` for the same state directory and id. At hour 1, use `other.declare_lost(...)` with a registered technical cause, outcome-blind record and termination accounting.
3. Call **`live.canary(reference)`**. It succeeds because `live` still sees its cached active lifecycle and no loss disposition.
4. Call `harness.run_units(..., guard=live)`. It returns **`{"u1": {"unit": "u1"}}`** from the formerly pending `.a1` without computing a retry.

A guard freshly constructed after the loss correctly rejects a same-id canary. That does not protect a pre-existing guard: its own cached view accepts the post-loss evidence and the harness consumes the result. This demonstrates a return-time eligibility violation; it does not claim that a subsequently reconstructed guard necessarily accepts the conflicting appended events.

### Retry continuation: exhaustion/fallback is also bypassed

Construct `g` and a second guard `stale` after opening `i1` and recording its start canary. Using `g`, let the unit fail with three classified `MemoryError` exceptions; `run_units` records and returns the registered fallback. Then call `run_units` for the same unit with `stale` and a successful callback. The callback **executes a fourth time globally**, and returns pending attempt key **`.a1`**, because `stale` retained empty retry counters and no fallback event. Observed callback sequence: **failure, failure, failure, fourth execution**.

This is the same stale-state defect affecting the three-attempt limit, independent of payload-identity checks. Reusing stale attempt numbers also threatens per-attempt storage identity.

**Required executable closure:** fence or invalidate stale guards against the current persisted study state before returning results, accepting canaries, allocating attempts or writing events. Serialize ownership/state changes so validation and mutation cannot use different snapshots; preserve globally unique event and attempt ordering. An enforced exclusive-owner/epoch mechanism is one possible solution. Merely promising to discard old Python objects at launch does not close these public-API violations. Add deterministic regressions for all three sequential two-object reproductions and extend sequence generation to retain and reuse older guards.

## Property-test review

The test is useful and demonstrably detects the three injected historical defects. It is **not a fully independent reference model for I1–I6**, and its passing coverage does not cover the new failure:

- **I1:** true opening/closing times are separately tracked, providing an independent lower bound. Charged totals themselves come from `instance_charges`; the model uses one GPU and truthful authoritative recovery records. It does not generate arbitrary smaller reconciliation records; my separate probe covers that case.
- **I2:** the callback checks lifecycle, loss, halt and cap using `instance_state`, `_lost`, `halted` and `gpu_h`. These are implementation helpers, not an independently derived active-instance/cap oracle. Checks use one live guard snapshot.
- **I3:** the history is independently retained, but statuses come from `attempt_status` (`tests/test_wme_runguard_property.py:263`). This detects false-to-true changes that are observed in the sequence; it does not establish all expected statuses independently.
- **I4:** closure points and before/after canary brackets are recomputed from raw events, a substantive improvement. References still come from `g.references`, and “current generation” is compared against `g.generation`/`g.contract`, not an independently replayed latest persisted contract. An always-ineligible implementation could evade much of this implication without the aggregate positive-coverage assertions.
- **I5:** per-generation result stability is checked through `resolve`; fallback hashing uses `RG._digest`. There is no independent oracle selecting the earliest eligible attempt or requiring a result to exist when one should. A `None` resolution is simply skipped. The narrower per-generation stability condition is sensible, but the prose's unqualified “never changes” should be read with that restriction.
- **I6:** fresh reconstruction is compared with the original implementation's cached state and `unit_status`. This checks round-trip consistency, not independent semantics or crash durability.

Operation coverage is adequate for the original serial, single-current-object failures: it reaches successful and failed brackets, disposals, losses, return-to-digest transitions, cap checks, exhaustion and recovery. However, `reconstruct`, `transition` and `stale` replace `self.g`; they **never reuse the previous object after another object advances shared state**. “Stale” means a freshly read guard with an old instance id, not an old snapshot. The generator also advances clocks between operations, not during unit callbacks; uses constant canary references across contracts; and does not inject torn writes, suffix loss or append failures. Aggregate counters establish occurrences, not all adversarial orderings. Add independent raw-event contract/lifecycle/retry replay and retained-object operations; the present property pass cannot support unconditional closure of B.

## A and changes 1–7

Relative to the preceding freeze review's `f9b5d3c`, only `generated_from_commit`, `harness.runguard` and `prereg_v2` change at manifest top level. `audit.panels_v2e`, `bank.closure_allocation` and `exposures.v2` are unchanged. The lineage and panel-audit reports are byte-identical to that baseline; their current SHA256 values match the manifest:

- Panel audit: `a251d79880d964a94c390d5e6604f53631d44938ef2b8f20426f30aab50f5311`.
- Root lineage: `0df383ae561f81afc42cfd24b0fd337edfb44797be656ef1f10dd3810281b2cf`.
- Current manifest: `79844dc616df5599d226ad8c1b3623fe1a7ad4199773f21ed4615bf9538c8295`, matching §II.26. The two property/mutation report hashes also match §II.26.

A therefore remains closed at the previously accepted **hash-only metadata/lineage scope**; I did not rederive text groups or inspect sealed items. `e2run.py`, `guards_v2.py`, `arms/sbon.py` and `panels_v2d.py` have no changes from that baseline. Targeted regressions pass for changes 1–5 and 7. R6 remains descriptive only, C2 is not a gate, W2 FAIL excludes H-T3 registration, H-E2 remains a conditional non-symmetry claim, and the accepted sizing, sparse-guard and S-BoN restrictions remain operative. Change 6/B alone prevents freeze approval.

## Launch preconditions after executable B closure

These are **launch preconditions, not additional freeze blockers**, and were not performed here:

- Run and retain inert identity/output bitwise checks and 1-vs-2-GPU parity on the actual pinned confirmatory engines/configuration.
- Commit per-engine canary references; run start, periodic and closing canaries with the registered eligibility rules.
- Bind real evidence paths/digests, engine configuration, unique physical-instance identities, the shared study state/ledger and registered fallbacks before the first confirmatory unit.
- Verify outcome-blind technical-repair records and deployed restart/replacement persistence, including partial-write handling and retention of every charged attempt. Complete-record truncation detection and cross-log consistency checking remain durability hardening; the manual truncation probe is not evidence that an ordinary fsync'd restart loses records.
- Verify authoritative instance-time reconciliation and the combined-ledger dead-man budget, including setup, idle, failures and canaries.
- Use `ProductionGuard` for confirmatory orchestration; `EngineeringGuard` results cannot enter confirmatory evidence. Exercise the repaired guard-ownership/fencing path on deployment after the CPU blocker is fixed.

The findings are **not only launch preconditions or optional hardening**: the sequential stale-object reproductions above are executable freeze blockers, requiring repair and re-check before approval.

## Four-line summary

1. All 98 targeted CPU tests and 5,000 property sequences pass; mutations are detected at sequences 2, 9 and 7.
2. The three original freeze probes are repaired; A, fallback identity and changes 1–5/7 remain closed at their stated scope.
3. B remains open: stale guards return superseded/lost results and execute a fourth attempt after persisted exhaustion.
4. **Verdict: REJECT — fix stale-state enforcement before freeze; confirmatory-stack checks remain launch preconditions.**
