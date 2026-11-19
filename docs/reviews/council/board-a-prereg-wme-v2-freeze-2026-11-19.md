# Board A — narrow freeze re-check of WME v2g

**Verdict: REJECT — blocker B remains open; freeze is not approved. Blocker A remains closed.**

Reviewed `main` **f9b5d3c524ec5040b76227651f05dcb8a8d539dd**, 19 November 2026 UTC, as the independent Codex methods seat. Scope: the four B runtime gaps from my final review, §II.25 and its implementation/manifest/response, with targeted regression checks for A and changes 1–7. The four original reproductions now pass, but adversarial continuations expose three failures of the same accounting and eligibility contract. These require executable repairs before freeze, rather than operator promises at launch.

Only this review file was written in the repository. Independent probes used automatically removed temporary directories outside the repository, the actual `harness.run_units` and `ProductionGuard`, and CPU fixture payloads, clocks and attestations. Fixture attestations are not production-validity evidence. No sealed split contents or `data/evals` were opened. No GPU, Lambda, secrets, commits, source-pool reconstruction or panel rebuilds were used.

## Verification

With `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1` and pytest's cache provider disabled, **94 tests passed in 14.35 s** across:

```text
tests/test_wme_e2run.py       tests/test_wme_panels_v2d.py
tests/test_wme_e2sens.py      tests/test_wme_guards_v2.py
tests/test_wme_runguard.py    tests/test_wme_sbon.py
tests/test_wme_prereg_v2.py   tests/test_wm_exposure.py
tests/test_wm_harness.py      tests/test_wm_p3gen.py
tests/test_wm_signflip.py
```

This includes all five `test_probe_gap1_*`/gap2/gap3/gap4 tests and the combined failure/canary/restart/preemption/exhaustion/first-valid-selection test. The only warning was constant-input correlation in the proxy-gate fixture. Independent probes below additionally exercised the original four cases and continuations absent from that suite. No full-suite, long sensitivity simulation or confirmatory-stack run is claimed.

## B: evidence per gap

### 1. Instance lifecycle accounting — original cases repaired; clock handling still unsafe

Independent `run_units` probes confirmed: a passing canary without an open instance cannot launch a unit; reconstruction and duplicate opening preserve 89 controlled elapsed hours; 91 elapsed hours block another unit; closing an instance prevents further computation; reopening its closed id is refused. These agree with the supplied gap1 tests.

**Remaining blocker:** local closing timestamps are not validated against opening timestamps (`src/wsp/wm/runguard.py:175–180`). `instance_gpu_h` sums signed durations across instances (`:220–242`). A backward wall clock therefore creates a negative credit that offsets time on a replacement instance. Reproduction using only public guard methods and its injectable clock:

1. Set the CPU fixture clock to 100 hours; construct the guard and `open_instance()` for `i1`.
2. Move the clock backward to 10 hours; `close_instance()` succeeds and records **−90 instance-hours**.
3. Use the registered infrastructure relaunch to `i2`; open it and pass its start canary at hour 10.
4. Advance to hour 101. `i2` has been open **91 hours**, but the combined instance charge is **1 hour**. `run_units` executes a new unit instead of raising `CapExceeded`.

These are controlled CPU clock values, not study measurements. No ledger rows were edited. The default clock is `time.time`, so rollback affects the actual accounting path. Fail closed on inconsistent time and require trustworthy reconciliation before continuing; a negative duration must never subsidize another instance. Merely clamping negative time would remove this demonstrated credit but would not establish the lost elapsed time.

### 2. Resume contract binding — simple mismatch repaired; transition cycles revive old results

Independent probes confirmed that a changed engine digest raises `ContractMismatch`; a valid technical transition hides the earlier result and requires a canary under the changed contract. The supplied test also checks invalid transition causes and refusal to return to the old contract without another transition.

**Remaining blocker:** the contract identity is only the attestation content digest (`:492–514`); eligibility compares that digest without a transition generation (`:633–650`). Reproduction:

1. Under contract A, open `i1`, pass a start canary, execute a unit, then pass its closing canary. Its original payload is eligible.
2. Reconstruct with changed engine contract B and a valid `ContractTransition`. The original attempt becomes `superseded_contract` and `resolve` returns nothing.
3. Reconstruct with the original A attestation and a second valid, outcome-blind technical transition. Both transitions are persisted.
4. Without any new canary or unit computation, `run_units` returns the original A payload. A callback that would fail if invoked is never called.

This reverses supersession solely by returning to an earlier digest. It contradicts the manifest's “earlier results become superseded_contract, never eligible” promise and lets a transition rollback reuse old canary evidence. Require an irreversible disposition or a distinct transition generation for attempts and canaries; returning to a previous configuration must not silently reactivate its earlier results.

### 3. Lost end-canary recovery — recovery repaired; loss disposition is reversible through canary recording

Independent probes confirmed that a replacement instance's own start canary does **not** close the old pending attempt. `declare_lost` charges the original instance to its supplied termination, retains its unbracketed attempt as ineligible, and permits attempt `.a2` after the registered relaunch. In the fixture, the old instance was charged 1.5 hours. No closing canary was added by the disposition itself. The combined supplied test now includes genuine missing-end-canary recovery.

**Remaining blocker:** `canary()` accepts events for closed/lost instances (`:623–625`). `attempt_status` checks loss only when there is no later canary (`:648–650`). Reproduction continuing the lost-instance case:

1. Retain `i1.a1` as ineligible through `declare_lost`; retry on `i2` and close `i2`'s canary normally.
2. Reconstruct the guard with the old `i1` identifier. Its persisted lifecycle is still closed/lost; no reopening occurs.
3. Call `canary(reference)` on that guard. It succeeds. The previously lost, unbracketed `.a1` changes from `retained_ineligible` to `eligible`.
4. `run_units` consumes `.a1` because it is earlier than `.a2`; `resolve` runs before the lifecycle check in `begin`.

This uses no event-log mutation or forged JSON. A stale instance identifier after replacement is enough to admit purported closing evidence from an instance already declared lost. Ordinary canaries correctly tagged `i2` remain isolated. Reject post-loss canaries and make the disposition of attempts lacking a closing canary at loss irreversible. Preserve genuinely bracketed pre-loss results where allowed by the registration.

### 4. Fallback identity — closed for the requested scope

After three classified CPU failures, the original fallback payload and digest are persisted. Independent reconstruction with another payload raises `FallbackMismatch`. Reconstruction with the same payload, followed by two `run_units` calls, returns exactly the original fallback without invoking another attempt or appending a second fallback event. The attempt count remains three. The supplied gap4 and combined orchestration tests pass. No additional fallback blocker was found in this re-check.

## A and changes 1–7: targeted regression finding

Compared with the previously reviewed `624ff5b`, the only changed top-level manifest fields are `generated_from_commit`, `harness.runguard` and `prereg_v2`. The complete `audit.panels_v2e`, `bank.closure_allocation` and `exposures.v2` values are unchanged. The lineage and panel-audit reports are byte-identical to that baseline, and both current file SHA256 values match their manifest pins. The current manifest SHA256 is **`f25ea7795dbea8cf662d7f0d73c4bc285579f36c9e6cb273afded43978f6077d`**, matching §II.25.

Thus A's previously accepted hash-only lineage reconciliation and exclusion evidence have not changed; A remains closed at that review's stated metadata-only scope. I did not independently rederive text groups or inspect sealed items. `harness.py`, `e2run.py`, `guards_v2.py`, `arms/sbon.py` and `panels_v2d.py` are also byte-identical to the baseline. The selected tests found no regression in changes 1–5 or 7. The R6 descriptive-only ruling, removal of the C2 gate, W2 FAIL/H-T3 non-registration, conditional non-symmetry E2 claim, sizing/guard rules and S-BoN restrictions remain operative. Change 6/B alone remains unresolved by the findings above.

## Confirmatory-stack launch preconditions after B closes

These remain **launch preconditions, not additional freeze blockers**, and were not performed here:

- Run and retain inert identity/output bitwise checks and 1-vs-2-GPU parity on the actual pinned confirmatory engines and configuration.
- Establish and commit the per-engine canary references; execute start, periodic and closing canaries under the registered eligibility rules.
- Bind the real evidence paths/digests, engine configuration, shared study state/ledger and registered fallback payloads before the first confirmatory unit.
- Verify the outcome-blind technical-repair record and persistence through the deployed restart/replacement path.
- Verify authoritative instance-time reconciliation and the combined-ledger dead-man budget, including setup, idle, failures and canaries.
- Select `ProductionGuard` in confirmatory orchestration; `EngineeringGuard` results cannot enter confirmatory evidence.

These launch obligations do not defer the three demonstrated CPU enforcement failures. A repaired B implementation needs targeted adversarial re-check before freeze approval.

## Four-line summary

1. All 94 targeted CPU tests pass; independent probes confirm repairs of the four original reproductions.
2. A stays closed; no regression found in changes 1–5 or 7 at the permitted review scope.
3. B remains open: negative clock credit, contract-cycle resurrection and post-loss canary resurrection.
4. **Verdict: REJECT — freeze not approved; confirmatory-stack checks remain launch preconditions.**
