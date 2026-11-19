# Board A — final narrow re-check of WME v2f

**Verdict: ACCEPT WITH CONDITIONS — freeze remains blocked by B. A is closed.**

Reviewed `main` **624ff5b0af8bf235171b10c3036ba35562aa32e6**, 19 November 2026 UTC, as the independent Codex methods seat. This is a narrow re-check of the two confirmation blockers and regression review of changes 1–7. The runtime repair is substantial, but §II.24.2's assertion of complete enforcement is stronger than the executable evidence supports.

Only this review document was written in the repository. CPU probes used temporary directories outside it. No sealed split contents or `data/evals` were opened; lineage inspection used hash/count metadata only. No GPU, Lambda, secrets, commits or panel rebuilds were used. I inspected the lineage script but did not execute its source-pool reconstruction or its report-writing entry point.

## Verification and regression scope

With `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1` and pytest's cache provider disabled, **89 tests passed in 18.45 s** across `test_wme_e2run`, `test_wme_panels_v2d`, `test_wme_e2sens`, `test_wme_guards_v2`, `test_wme_runguard`, `test_wme_sbon`, `test_wme_prereg_v2`, `test_wm_exposure`, `test_wm_harness`, `test_wm_p3gen` and `test_wm_signflip`. The sole warning was constant-input correlation in the proxy-gate fixture. This selection includes the authors' combined failure/canary/restart/exhaustion test. It does not establish production-stack validity.

I compared changes since the previously reviewed `4307e1e`, including the builder-input extraction, non-writing lineage return, manifest overlay and runtime changes. The manifest changes are confined to `audit.panels_v2e`, `bank.closure_allocation`, `exposures.v2`, `generated_from_commit`, `harness.runguard` and `prereg_v2`. I found no additional regression in changes 1, 2, 4, 5 or 7 at the previously accepted registration/module scope. Change 3 closes below; change 6 remains blocked. W2 FAIL, H-T3 non-registration, the narrowed conditional non-sign-symmetry E2 claim, fixed sizing/guard rules and S-BoN restrictions remain operative. No full-suite, long sensitivity simulation or confirmatory driver run is claimed.

## A — closed at the permitted metadata level

Independent metadata checks verified:

- Manifest SHA256 `82e844b8dd49929c501bb447e5f984a10645cde2a1a36e1e00d8939b11aaa703` matches §II.24. The lineage report's file digest matches §II.24 and the manifest pin. Canonical hashing of its complete mapping reproduces `c28022f9c152a3443fc09705d9e09a281822d078d1a812269ed5e8fd9a9dfdc5`.
- All **1,761 distinct member hashes** are mapped. Their **1,728 original groups** exactly equal the union of contrast/adapter source-group commitments in the hash-bound roots metadata. There are **1,723 distinct final groups and representatives**. Roles reproduce **1,761 = 1,723 representatives + 25 in-pool members + 13 out-of-pool members**.
- Every mapped representative, final group and owner matches an actual `wme-c1` construction manifest entry. The reserved count is 1,723. Direct intersections against all 40 c1 manifest units give **zero root-member hashes and zero final root groups in evaluation splits**.
- Direct intersection against every c2-gen manifest unit gives **zero root-member hashes**. Both panel-manifest file digests match the lineage report's input pins. The bound v2e audit still reports **1,035 joint-policy pairs with zero hash/group overlaps** and zero evaluation exposure hits.

The c2-gen manifest publishes hashes without per-item group lists. Accordingly, c2 group exclusion is supported by the hash-bound joint-policy audit and inspected reservation/allocation implementation, not an independently recomputed text grouping. Likewise, I verified the published mapping and its metadata consistency, not a new source-text derivation of the union groups. These are the limits of this hash-only review; they do not reopen the requested lineage reconciliation.

§II.24.1 now registers group reservation with representative deduplication, rather than literal placement of every member. It explicitly authorizes only the disclosed construction exposure exception (422 written representatives / 428 member hashes), retains absolute evaluation exclusions and corrects the response's original-versus-final group wording. **Blocker A is closed.**

## B — remains a freeze blocker

The repair demonstrably fixes several original defects: `guard` is required; ordinary classified failures are charged; retry/relaunch counters and canary records survive reconstruction; successful attempts have separate records; pending/failed-canary results are not returned by `resolve`; and the tested chronological first-eligible selection works. Independent CPU failure probes also confirmed three failed attempts survive restart and charge nine controlled elapsed seconds in total. Missing/false launch attestations are rejected by the supplied tests.

However, independent probes against the actual `run_units`/`ProductionGuard` path found these remaining failures. The controlled clocks, digests and payloads below are CPU test inputs, not study outcomes.

1. **Authoritative instance accounting is optional and can be reset on restart.** `ProductionGuard.begin` does not require an open instance (`runguard.py:428`). With an attestation and passing canary but no `open_instance`, a probe advanced its accounting clock by 91 hours and successfully executed a unit; authoritative instance time remained zero. More decisively, opening an instance, advancing to 89 hours, reconstructing the guard and calling `open_instance` again reduced its authoritative time from **89 to 0 hours**: `CombinedLedger.instance_gpu_h` overwrites an existing open timestamp. This directly defeats the absolute combined cap during an ordinary startup/resume sequence. Require validated lifecycle registration before units, preserve the earliest active opening across restart, and reconcile interruptions/closure against authoritative instance records. Duplicate opening must be idempotent or rejected without discarding time. Pre-unit estimates cannot substitute for this accounting.

2. **Restart does not bind the registered launch/configuration contract.** The constructor validates the new attestation's shape, but, once any attestation event exists, neither compares it with the persisted one nor records the new one (`runguard.py:356–373`). I created an eligible result, reconstructed the guard with a different engine-config digest, and `run_units` returned the old payload without objection. The persisted canary reference/configuration and evidence must be checked against the registered contract on resume; permitted technical repairs need an explicit, auditable transition rather than silent replacement. This is an enforcement gap, not a request to rerun parity during this CPU review.

3. **A lost end canary has no executable recovery disposition.** After a successful pending attempt on one instance, I reconstructed the guard, used the registered relaunch, opened a replacement instance and passed its start canary. Retrying the unit still raised `CanaryMismatch` because the old instance's pending attempt blocks every new attempt (`runguard.py:433–440`). A preempted instance cannot supply its closing canary. Persist a technical disposition that retains that unbracketed attempt as ineligible, charges its instance time, preserves its attempt count and permits the remaining registered attempts/fallback. Do not fabricate a closing canary from the replacement instance. The supplied orchestration test closes the first instance's canary before relaunch and therefore misses this case.

4. **Fallback status persists, but fallback value/contract does not.** `fallback` records only unit/label; `resolve` reconstructs the payload from the current caller's `fallback_value` (`runguard.py:424–425,469–471`). After three charged technical failures, my first invocation returned its configured base fallback; reconstructing with a different fallback value changed the resumed result without executing another attempt. Persist and verify the registered fallback payload or immutable fallback identity, including across restart. A caller convention alone does not enforce fixed result consumption.

These are failures within the already registered accounting, restart and eligibility requirements. They require technical fixes and targeted CPU recovery tests before B can close. They cannot be deferred merely as operator launch checks. The passing happy-path tests and the authors' “all seven closed” status do not resolve them. **Freeze is not approved.**

## Confirmatory-stack launch preconditions after B closes

The following remain **launch preconditions, not additional freeze blockers**: run and retain inert identity/output bitwise checks and 1-vs-2-GPU parity on the actual pinned confirmatory engines/configuration; establish the committed per-engine canary references; run start, periodic and closing canaries with the registered eligibility rules; verify the outcome-blind technical-repair record and persistence across the deployed restart path; and verify authoritative instance-time reconciliation plus the combined-ledger dead-man budget. Bind the real evidence paths/digests, engine configuration, study state/ledger and registered fallbacks before the first confirmatory unit. Confirmatory orchestration must select `ProductionGuard`; explicit `EngineeringGuard` results cannot be admitted as evidence. None of these checks was performed here on GPU.

## Four-line summary

1. A closes: 1,761 = 1,723 + 25 + 13; metadata bindings and exclusion evidence verify.
2. All 89 selected CPU tests pass; no additional regression found in previously closed changes 1, 2, 4, 5 and 7.
3. B remains blocked by lifecycle accounting, resume contract binding, lost-canary recovery and mutable fallback replay.
4. **Verdict: ACCEPT WITH CONDITIONS — freeze remains blocked; stack validity checks remain launch preconditions.**
