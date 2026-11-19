# PREREG_WME v2d: response to Board A's v2 review (19 Nov 2026)

> **DESIGN, CODE AND ONE SYNTHETIC CPU SIMULATION; NO OUTCOMES.**
> - **What this answers:** `docs/reviews/council/board-a-prereg-wme-v2-2026-11-18.md` (ACCEPT WITH CONDITIONS,
>   required changes 1–7).
> - **Where:** branch `prereg/wme-v2d`, from main `665b94e`.
> - **What did not happen:** no GPU, no Lambda, no `data/evals`, no sealed split, no confirmatory outcome. Track W2
>   and its branch `feat/wme-stagew2` were not touched; nothing was cherry-picked. Nothing was merged or pushed.

## BLUF

| # | Required change | Status | Evidence |
|---|---|---|---|
| 1 | Incorporate the ruling addendum (R6 descriptive only; no C2 gate) | **Closed** | §II.22.1. `prereg_v2.r6_resimulation.consequence` now carries the addendum, with the old text kept under `consequence_superseded`. The issues-ruling digest is re-hashed: `fb74dc82…` → `7118a1d3…`. The v2c generator is updated (`2c0f2ef`). |
| 2 | One executable E2 configuration | **Closed** | §II.22.2, `wsp.wm.e2run`, `e2.config_effective`, `slope_transport` false in `model.wm.config` and S2/S3. CPU integration test `tests/test_wme_e2run.py` (5 tests). |
| 3 | Exposure, joint MMLU allocation, build order, audits | **Closed (v2e, 19 Nov)** | W2 merged and bound (`wme-w2` sha256 `b983bc20…`; FAIL, so H-T3 is not registered). The real joint allocation was run (sha256 `0051481c…`), `wme-c1` and `wme-c2-gen` were built, and the hash-only audits are clean (`docs/reviews/wme/wme-v2e-panel-audit-2026-11-19.json`). The S-TE MMLU-Pro shortfall was frozen at 3,398 and its power restated (0.792). Split hashes and the closure allocation are in the manifest. PREREG §II.23; see the "Update v2e" section below. |
| 4 | State the E2 null; sensitivity; freeze the failed-calibration consequence | **Closed; the consequence fired** | §II.22.4 and `docs/reviews/wme/wme-v2d-e2sens-2026-11-19.json`. With an asymmetric mean-zero null, sign-flip is 0.0715 and RB-E2 0.078, so the **H-E2 mean-advantage wording is withdrawn**. |
| 5 | S-TE sizing label; guard α 0.025 in code; sparse rule | **Closed** | §II.22.5, `wsp.wm.guards_v2`, `prereg_v2.te_n` (reproduces 3,463 and 1,784 exactly), `tests/test_wme_guards_v2.py` (7 tests). |
| 6 | Validity, repair and stopping contract | **Closed in code and text** | §II.22.6, `wsp.wm.runguard`, `harness.run_units(guard=)`, `tests/test_wme_runguard.py` (6 tests). P-H100 inert and parity evidence is cited. The checks re-run at launch. |
| 7 | S-BoN runner and Holm-threshold power | **Closed** | §II.22.7, `wsp.wm.arms.sbon`, `tests/test_wme_sbon.py` (4 tests). At B 86.7218: K 174, P 44, 86.5013 spent, 0.2205 forfeited. Power is 0.006, 0.024 and 0.037 at α 0.0025, 0.0125 and 0.025. |

**Commits.**
- `2c0f2ef`: all code and tests.
- `5944229`: the manifest overlay script and the sensitivity report.
- The report commit (branch head): `docs/PREREG_WME.md` §II.22, the manifest (sha256 `02877aa4…`) and its table, this
  response, and the session log.

## Change by change

### 1. The governing addendum

**R6.** §II.22.1 supersedes §II.21.6's "R6 must be redesigned before the freeze" and §II.21.9 row 4, explicitly. It
records:
- that R6 is retained only as a non-evidentiary descriptive diagnostic;
- the selection mechanism (118 vs 48 of 1,420 folds);
- the stopping caveat: fixed-N intervals at outcome-dependent stopping times carry no sequential coverage guarantee;
  the failure is the 600-replicate criterion (22 > 21);
- the prohibition on using any C_T, p-value or interval as evidence of WM-CP superiority.

**C2.** §II.18's "then C2 gates" is overridden. No C2 gate remains anywhere; the §II.22.6 validity preconditions
apply instead.

**Manifest and generator.**
- The consequence carries the addendum, and the superseded text is preserved.
- The digests are updated: issues ruling `7118a1d3…`, Board A v2 review.
- `scripts/wme_manifest_prereg_v2c.py` now emits the addendum consequence (`2c0f2ef`).
- The historical §II.21 text is unchanged.

### 2. The E2 configuration

| Board A's finding | Resolution |
|---|---|
| `slope_transport: true` in `model.wm.config` and S2/S3 | `E2_WM_CONFIG` sets `slope_transport=False`. The manifest now stores this effective config, plus `e2.config_effective`. |
| `PredictorArm` vs `V0RArm` | E2 uses `V0RArm`: realised-prefix R rows through one `r_of`, shared by the bank-label path. `PredictorArm` is not used. |
| R6 tie rule vs `model.select_shrink` | E2 registers `model.select_shrink` (exact ties) for both arms. R6 is not borrowed. |
| StatNN member and family | `model.MatchedStatic` only, with no family selection in E2. |
| Seeds and components | Members 0–4; component seed 20261117; component ids mandatory; calibration components excluded. |

**Integration test.** `tests/test_wme_e2run.py` covers the frozen defaults, mandatory components, component
bootstraps and a shared selection split, removal of the A block and slope targets, and bitwise-identical arm inputs
for overlapping and non-commuting orders, equal to the bank-label path.

### 3. Exposure and assignment (partially pending W2)

**W2.** `wme-w2` (on `feat/wme-stagew2`) is referenced as pending, with null slots for:
- its manifest hash;
- the W2 outcome;
- the H-T3 decision.

**Build order.** It is fixed so that W2 merges first and both builders refuse to run until `wme-w2` is registered.

**Joint allocation.** Implemented over the union of `wm.neardup` and `exposure.near_duplicate_groups` with salt
`wme-mmlupro-v2c`. It is self-hashing and consumed by both builders. The tests show each builder draws only its share,
and that the cross-panel audit catches an item or group shared across panels.

**Shortfall rule.**
- `wme-c1` raises on any shortfall.
- `wme-c2-gen` freezes min(required, fresh) and sets `restate_power`.

**Audit code.** Hash-only, covering counts, pairwise hash and joint-group overlaps, and exposures, for every source
including the generated guards.

**Not done (requires the W2 merge):** the real allocation file and its hash, the panel builds, the split hashes and
the published audits. Condition 1 therefore remains open.

### 4. The E2 null

**The stated null.** Joint sign symmetry of the 24 paired differences, conditional on the realised test items, the
declared library and the frozen predictors. The sign-flip test is not a test of equal means, and inference does not
generalise over items or libraries.

**The sensitivity run.**
- 13 scenarios, 4,000 replicates each, 10,000 flips; via agent-tmux on code `2c0f2ef`; 13 min.
- It reproduces the v2 null (0.027) and the alternative (0.828).
- Arm-by-item interactions leave the population-level rate at 0.025–0.032.
- Arm-by-family (library) contrasts give 0.084 and 0.176. That is why the wording is conditional on the library.

**The failure.** With an asymmetric mean-zero null (`asym_reversed`; MC mean −0.002 ± 0.004), the sign-flip test
rejects at **0.0715** and RB-E2 at **0.078**. RB-E2 also fails `sym_fallback_heavy` (0.047).

**The frozen consequence, now in force.**
- The mean-advantage wording is withdrawn.
- An H-E2 rejection is reported only as non-symmetry of the paired differences, favouring WM, conditional on these
  items and this library.
- RB-E2 is reported beside it.
- No test or α was changed.

**Power.** 0.828 is labelled model-based planning power.

### 5. S-TE and the sparse guards

**Sizing.** "0.8" is labelled analytic benchmark sizing at Holm's first level, before the guards. The exact inputs
reproduce 3,463 and 1,784:
- δ is the λ = 1 mean dev gain halved and rounded to 4 decimals (0.0214, 0.0353);
- d is the exact Wilson upper discordance (0.167275, 0.234994).

Board A's 3,457 came from also rounding d.

**Guard level.** `guards_v2.guard_test` runs at α 0.025. `p3gen.guard_test` takes `alpha` explicitly.

**Sparse rule.**
- **When it applies:** fewer than 20 non-zero paired differences, or SE = 0.
- **The bound:** the Clopper–Pearson upper limit of P(+1).
- **Why:** before choosing the threshold, the bootstrap-t was measured at about 0.15 pass rate at the boundary with
  about 4 discordant items in 200.
- **Check:** the test shows boundary-null pass rates ≤ α plus Monte Carlo tolerance.

### 6. Validity, repair and stopping

**Cited evidence.**
- P-H100: `inert_pass` true; 1-vs-2-GPU parity bitwise over 10,080 means.
- Stage W: canary bitwise on repeat.

**The bound contract.**
- **Canaries:** the canary per engine path, with the first confirmatory instance as reference.
- **Eligibility:** a unit needs a valid canary before and after it. Units before a failed canary are retained but
  ineligible, and the first canary-valid rerun is used whatever its value.
- **Repairs:** only the 7 technical causes, with an outcome-blind attestation.
- **Retries:** 3 attempts per unit counted across relaunches, separate from 1 relaunch per phase.
- **Stops:** an absolute combined 90 GPU-h-eq / $1,000 stop, checked by `harness.run_units` before every unit, plus
  the 1.25× phase pause. The dead-man `max_usd` comes from the combined ledger.

### 7. S-BoN

**Pinned in code (`arms/sbon`):**
- the screen count K (the largest K with K·c16 + ⌈K/4⌉·inc ≤ B_par);
- ⌈K/4⌉ promotions;
- ties to the earlier draw;
- promotion reuses the 16 screened groups and scores the remaining 48;
- the fractional residual and unused promotion slots are forfeited, never re-spent;
- cache hits by key or θ-hash cost 0 and are not counted;
- the seeded draw stream.

**Fixtures.** At B 86.7218: K 174, P 44, 86.5013 spent, 0.2205 forfeited. `plan.BudgetLedger` now accepts a
non-integer budget; it previously truncated to an integer, which would have silently cut B_par.

**Holm-threshold power.** 0.006 at α 0.0025, 0.024 at 0.0125, 0.037 at 0.025.

## Board A's optional suggestions

- **Effective-registration view:** partly addressed by the manifest's `e2.config_effective` and §II.22's
  supersession lines. A compact generated view is not written.
- **H-E2 scores before and after clipping, and fallback counts by arm and family:** not added in this amendment.
  Recommended for the reporting template.
- **S-TE untruncated-in-both reading:** §II.6.4 already reports it as descriptive. The non-causal reading Board A
  states is agreed.

## Open items

1. **Closed by v2e (19 Nov 2026; see below).** Originally: condition 1 stays open until the W2 merge, with:
   - the `wme-w2` hash and outcome and the H-T3 decision, by dated amendment;
   - the joint allocation and its hash;
   - both panel builds and their split hashes;
   - the hash-only audits;
   - power restated on any shortfall.
2. **H-E2 now supports only a conditional non-symmetry statement.** The chair may wish to note this change of
   scientific scope. It follows from the rule frozen before the run, and no method was changed after it.
3. **The launch preconditions run on the confirmatory stack** before any outcome: inert identity, parity and canaries.

## Hygiene

- The secret scan was clean before each commit.
- There are no IPs, hostnames or AI attribution lines beyond the required co-author trailer.
- `data/` is untracked, and nothing was added with `git add -A`.
- Every commit used `--no-verify` after a full CPU-suite run on the same tree via agent-tmux: 2,244 passed, and only
  the three pinned tripwires failed (`test_f12d_prereqs` ×2, `test_f12e_qualify` ×1).
- `tests/test_wme_prereg_v2.py`, which reads the manifest, was re-run after regeneration: 9 passed.

## Update v2e (19 Nov 2026): change 3 closed

> **What changed.** Stage W2 merged at `718c48f` and failed its pre-declared rule.
> - Branch `prereg/wme-v2e`, from main `718c48f`.
> - CPU builds via agent-tmux. No outcome was generated; no `data/evals` file was opened; no sealed file was read
>   back.
> - PREREG §II.23 records everything below.

| Board A item under change 3 | Evidence |
|---|---|
| W2 identity, exposure and chronology | `wme-w2` is in `wm.exposure.EXPOSURES` (manifest sha256 `b983bc20…`, content hash `6d533f94…`). It is excluded from both confirmatory builds, which run after it (`exposure.registry_before`). |
| W2 outcome and H-T3 decision | **FAIL**: zero-shot d 0.003, Holm lower bound −0.053; k = 2 d 0.000; nulls ≤ 0.011. **H-T3 is not registered** and is removed from every total. Registered total **66.93 GPU-h-eq, about $360**. The (a) negative-result rule is §II.23.1. |
| Salted joint MMLU allocation over the union of both policies, consumed by every builder | `panels_v2d.allocate` (salt `wme-mmlupro-v2c`), record sha256 `0051481c…`, file sha256 `c323eb13…`. Bank-root shard groups (830) are pre-assigned to c1 construction. c1 was dealt 235 groups so that 232 stay usable; c2-gen was dealt 3,463. Both builders restrict MMLU-Pro to their shares. Tests: `tests/test_wme_panels_v2d.py` (12). |
| Build-order contradiction | Resolved: W2 merges, then `wme-c1`, then `wme-c2-gen`. Builders refuse to run without `wme-w2`. |
| Hash-only exclusion, intersection and count audits for every source and split, incl. generated guards and cross-panel duplicates | `docs/reviews/wme/wme-v2e-panel-audit-2026-11-19.json`: **clean**. 1,035 (panel, split) pairs have no hash or union-policy group overlap. There is no exposure hit outside the 1,761 disclosed bank-root construction items. Per-source counts and item-hash digests are in §II.23.3. |
| Shortfall rule and power restatement before outcomes | c2-gen MMLU-Pro is frozen at **3,398 of 3,463**. Analytic power is restated: **0.792** at α 0.0125 and 0.863 at 0.025 (`prereg_v2.te_power`). Every other panel was built at its registered N. |
| Final split hashes and closure allocation bound | `panel.wme_c1.split_hashes`, `panel.wme_c2_gen.split_hashes`, `audit.panels_v2e`, `bank.closure_allocation` in the manifest (sha256 in §II.23). |

**A design issue found and fixed: the bank-root shards.**
- The 72 bank roots were sampled at θ₀ from the v2b-era construction remainder.
- A naive rebuild would have let some of their shard items reach sealed or dev splits.
- Their **1,728 original groups** (root-sampler group ids; 1,761 pool members) are reserved for `wme-c1`
  construction. Under union-policy grouping they form **1,723 final groups**, each with one written representative
  (group reservation with representative deduplication; corrected wording, v2f, PREREG §II.24.1).
- 422 written representatives (428 root member hashes) were later used by exploratory panels. That is disclosed under
  the registered construction-only exception; no root member or group enters an evaluation split.

**Discarded builds (nothing read; hashes kept under `data/wme/panels/v2e-build-attempt*`).**

| Build | What happened |
|---|---|
| 1 | c1 MMLU eligibility defect; stopped before writing |
| 2 | OR-Bench-80k missing from the offline cache; stopped at c2-gen |
| 3 | Audit not clean (single-policy grouping, per-panel c2 families, a non-root representative) |
| 4 | Audit clean, but the MMLU-Pro fresh field shard was 21 < 24 |

Each fix was committed and tested before the next build: `9447675`, `d3b703a`, `d3d83ad`. Salted, deterministic
assignment leaves no room to select favourable items.

**Status of all seven changes (as written at v2e; superseded by "Update v2f" below).** All are closed:
- 1, 2, 4, 5 and 7 by v2d;
- 6 by v2d, with the launch checks to re-run on the confirmatory stack;
- 3 by v2e.

## Update v2f (19 Nov 2026): Board A confirmation blockers A and B

> **Context.** Board A's confirmation (`docs/reviews/council/board-a-prereg-wme-v2-confirm-2026-11-19.md`) closed
> changes 1, 2, 4, 5 and 7 and left two freeze blockers.
> - Branch `prereg/wme-v2f`, from main `a4fa022`.
> - These are technical repairs only: no rebuild, no outcome, no sealed content read.
> - PREREG §II.24.

| Blocker | Closure evidence |
|---|---|
| **A. Root-shard lineage** | **Lineage.** `scripts/wme_v2f_root_lineage.py` re-runs the builder's own plan (`build_c1_panel_v2e(lineage=True)`, nothing written). It reproduces the built `wme-c1` hash and group lists in all 40 units (digests only), then maps every root member. **Counts:** 1,728 original groups and 1,761 members map to 1,723 final union groups and 1,723 written construction representatives. The remainder is 25 in-pool members and 13 members found only in the MMLU-Pro proxy pool (not a panel source). **Checks:** 0 root members or groups in any `wme-c1` evaluation split; 0 in `wme-c2-gen`. **Digest:** mapping sha256 `c28022f9c152a3443fc09705d9e09a281822d078d1a812269ed5e8fd9a9dfdc5` (report `docs/reviews/wme/wme-v2f-root-lineage-2026-11-19.json`). **Wording:** amended to "group reservation with representative deduplication". **Exception:** the construction-only exposure exception is registered (422 written / 428 member hashes); evaluation exclusions stay absolute. **Response bullet:** corrected above. |
| **B. Runtime contract** | **Guard.** `harness.run_units(*, guard)` is mandatory. `runguard.ProductionGuard` provides launch attestations, a persistent replayed event log, persistent retry and canary state, per-attempt result keys, eligibility-gated resume (first canary-valid in order; retained-ineligible kept; pending incomplete), charged failures, and an absolute cap on max(unit time, authoritative instance time). **Tests.** CPU tests in `tests/test_wme_runguard.py`, including the combined orchestration test (failure, failed end canary, process restart, relaunch, exhaustion/fallback, first-valid selection). Full suite: only the 3 pinned tripwires fail. |

**Status (as written at v2f; superseded by "Update v2g" below).** Changes 1, 2, 4, 5 and 7 were closed by Board A.
Change 3 is closed by v2e and the lineage above. Change 6 is closed in code by v2f, with the launch checks to run on
the confirmatory stack. Both blockers await Board A's check.

## Update v2g (19 Nov 2026): Board A final re-check, blocker B gaps 1-4

> **Context.** Board A's final re-check (`docs/reviews/council/board-a-prereg-wme-v2-final-2026-11-19.md`) closed
> blocker A and found four runtime gaps in B.
> - Branch `prereg/wme-v2g`, from main `d0d946c`.
> - These are technical repairs in `src/wsp/wm/runguard.py` (code `e217056`); PREREG §II.25.

| Gap | Repair | Regression test (`tests/test_wme_runguard.py`) |
|---|---|---|
| 1. Instance lifecycle: a unit ran without an instance (91 h); reopen reset 89 h to 0 h | `begin` requires a registered active instance (`InstanceNotOpen`). Opening is idempotent and keeps the earliest opening. Closed ids are never reopened. Authoritative records override local times via `reconcile_instance`. The cap uses that time. | `test_probe_gap1_no_unit_runs_without_an_open_instance`, `test_probe_gap1_restart_and_reopen_never_reset_instance_time` |
| 2. A changed engine-config digest returned the old payload | The full attestation digest is persisted. A mismatch raises `ContractMismatch` unless a persisted technical `ContractTransition` is recorded. Results and canaries are per contract (`superseded_contract`). | `test_probe_gap2_changed_engine_config_cannot_return_old_payload` |
| 3. A lost end canary blocked retries | `declare_lost` is a persisted technical disposition: an authoritative close (time charged), retained-ineligible attempts, no fabricated canary, attempt counts kept, and the remaining attempts or fallback allowed. | `test_probe_gap3_lost_end_canary_allows_the_remaining_attempts` |
| 4. A changed fallback value changed the resumed result | The fallback payload and its sha256 are persisted and verified (`FallbackMismatch`); `resolve` returns the persisted payload. | `test_probe_gap4_changed_fallback_value_is_refused_on_resume` |

**Combined test.** `test_orchestration_failure_canary_restart_preemption_exhaustion_and_first_valid_selection` now
includes a preempted-instance path.

**Suite.** 2,286 passed; only the 3 pinned tripwires fail.

**Status.** Blocker A was closed by Board A. Blocker B's four gaps are repaired in code and await Board A's check.
The confirmatory-stack checks remain launch preconditions. *(Superseded by Update v2h: Board A's freeze re-check closed
gap 4 and found three remaining failures.)*


## Update v2h (19 Nov 2026): Board A freeze re-check, three remaining runtime gaps

> **Context.** Board A's freeze re-check (`docs/reviews/council/board-a-prereg-wme-v2-freeze-2026-11-19.md`)
> rejected on three remaining blocker-B failures; gap 4 is closed.
> - Branch `prereg/wme-v2h`, from main `c62ca31`.
> - Technical repairs in `src/wsp/wm/runguard.py` and `src/wsp/wm/harness.py` (code `960de9c`); PREREG §II.26.

| Gap | Repair | Regression test (`tests/test_wme_runguard.py`) |
|---|---|---|
| 1. A backward clock created negative credit that offset a replacement instance | Fail closed: a backward reading persists `clock_halt` and raises `ClockInconsistent`; everything halts until every instance active at the halt is authoritatively terminated and the clock is back at or above its maximum. Local durations are never negative; each instance is charged max(local, authoritative) separately. No clamping. | `test_probe_freeze_clock_rollback_cannot_credit_another_instance` |
| 2. Transitioning away and back revived the old contract's results | Monotone generation per contract; attempts and canaries bind to (digest, generation); a return to an earlier digest is a new generation and never reactivates anything. | `test_probe_freeze_contract_rollback_does_not_revive_old_results` |
| 3. A canary under a stale lost instance id revived its pending attempt | `canary()` and every unit start (`begin`) need an active, non-lost instance; closure or loss irreversibly disposes of unbracketed attempts; post-closure canaries never count; `run_units` validates before any resolve. Genuinely bracketed pre-loss results keep their eligibility. | `test_probe_freeze_canary_after_loss_cannot_revive_an_attempt` |

**Property test.** `tests/test_wme_runguard_property.py` (seeded generator; `hypothesis` is not installed) runs
invariants I1–I6 after every step. The registered run was 5,000 sequences × 40 steps and passed with 0 failures. The
coverage counts are in §II.26.2.
- **Mutation check:** re-injecting each of the three defects is detected (at sequences 2, 9 and 7).
- **Additional runtime defects found:** none. Test weaknesses (coverage, invariants that trusted guard helpers) were
  found by the mutation check and fixed before the registered run.

**Suite.** 2,290 passed; only the 3 pinned tripwires fail.

**Status.** Blocker A and gap 4 were closed by Board A. Blocker B's three remaining gaps are repaired in code and await
Board A's check. The confirmatory-stack checks remain launch preconditions. *(Superseded by Update v2i: Board A's freeze re-check
2 found the stale-object root cause.)*


## Update v2i (19 Nov 2026): Board A freeze re-check 2, stale objects fenced

> **Context.** Board A's freeze re-check 2 (`docs/reviews/council/board-a-prereg-wme-v2-freeze2-2026-11-19.md`)
> rejected on one root cause: stale guard objects were not fenced against shared persisted state.
> - Branch `prereg/wme-v2i`, from main `e54776b`.
> - A structural repair in `src/wsp/wm/runguard.py` (code `a53fdbc`); PREREG §II.27.

**Design.**
- **One log.** One authoritative, append-only, fsync'd, hash-chained study log per state directory, with a global
  gap-free sequence and a head record. The ledger, clock, retry and relaunch counters, canaries, contract generations,
  lifecycle, loss dispositions, fallbacks and owner epoch are all replayed from it; the separate logs are merged.
- **Lock and re-read on every operation.** Every public operation takes an exclusive `fcntl.flock`, re-reads and
  verifies the tail, re-validates on that current state, then appends under the same lock.
- **Owner epoch.** Constructing a guard appends `owner_epoch`. Older objects raise `StaleGuard` before returning or
  writing anything.
- **Fail closed on log damage.** Torn, truncated or chain-broken logs raise `LogCorrupt`. A torn tail has one audited
  repair, which also ends the ownership.

| Board A reproduction | Regression test (`tests/test_wme_runguard.py`) |
|---|---|
| B2: a stale guard returned a superseded result | `test_probe_freeze2_stale_guard_cannot_return_a_superseded_result` |
| B3: a stale same-id guard accepted a post-loss canary | `test_probe_freeze2_stale_same_id_guard_cannot_accept_a_post_loss_canary` |
| A stale guard ran a fourth attempt after exhaustion | `test_probe_freeze2_stale_guard_cannot_run_a_fourth_attempt_after_exhaustion` |
| Duplicate sequence numbers | `test_probe_freeze2_overlapping_objects_never_duplicate_sequence_numbers` |

Further tests:
- `test_unfenced_views_always_act_on_the_current_log`
- `test_lock_serialises_concurrent_processes`
- `test_torn_or_truncated_log_fails_closed_until_reconciled`

**Property test.** The property test now keeps every guard object it constructs and reuses older ones at random. It
also takes over ownership during units and tears appends. Its invariants come from an independent replay of the raw
log bytes, which answers Board A's oracle remarks on I2–I6.
- **Run:** 5,000 sequences × 40 steps on disk-backed temporary storage; passed, 0 failures; 16,048 calls through older objects refused with the log unchanged; 1,300 takeovers during units; 2,512 operations refused on torn logs and 1,797 audited repairs.
- **Mutations:** removing the lock and the re-read (m4) is detected at sequence 1 in every mode; removing the epoch fence (m5) is detected at sequence 1 (full) and 26 (I2, without the fencing assertion); removing only the OS lock (m6) is not visible to single-process sequences and is detected by the concurrent-process check; unmutated controls pass 300 sequences in every mode and the concurrent check.
- **Additional defects found:** The property test found **no runtime defect** in the repaired guard: 0 failures in the registered run, the development runs and the default-suite runs. One test bug was found and fixed during development: the torn-log step consulted the model before calling a guard. m6 is not detectable by a single-process sequence, which cannot interleave inside an operation; the lock is covered by the concurrent-process check and by `test_lock_serialises_concurrent_processes`. With the I0 checks off, m4 is caught by the implementation's own fail-closed chain verification, not by a result-level invariant. With the lock and re-read in place, the epoch fence's removal is caught by I2's one-owner rule. These detections are reported as observed.

**Suite.** 2,297 passed; only the 3 pinned tripwires fail.

**Status.** Blocker A and fallback identity were closed by Board A. Blocker B's stale-object root cause is repaired in
code and awaits Board A's check. The confirmatory-stack checks remain launch preconditions, now including the
deployed ownership and fencing path and a host-local lock for the shared state.

## Update freeze (19 Nov 2026): Board A freeze re-check 3 accepted with conditions; FROZEN

> **Context.** Board A's freeze re-check 3 (`docs/reviews/council/board-a-prereg-wme-v2-freeze3-2026-11-19.md`) is
> ACCEPT WITH CONDITIONS: B is closed for the serialized production path.
> - Branch `prereg/wme-freeze`, from main `967d61d`.
> - PREREG §II.28 records the conditions and freezes the registration.

| Board A condition | Where recorded |
|---|---|
| 1. Calls sharing a `StudyLog` object must be serialized, including across threads | §II.28.1 item 1; manifest `harness.runguard.serialisation` |
| 2. A cached refresh verifies the last cached record and the appended tail. Immutable-prefix storage is an operating assumption. A verified fresh replay is required on every restart and recovery. The hash chain is not protection against a coordinated rollback of log and head. | §II.28.1 item 2; manifest `harness.runguard.log_integrity` |
| The launch restriction: serialized path only, no autonomous parallel launch | Chair decisions (1) and (2), verbatim, in §II.28.2; manifest `harness.runguard.supported_execution`, `caps.concurrency` and `caps.dead_man_allowance` |
| Launch preconditions | §II.28.3 (8 items, mandatory before the first confirmatory unit); manifest `harness.runguard.launch_preconditions` |

**Freeze.**
- Freeze conditions (i)–(iv) are all met (`scripts/wme_freeze_check.py`; report sha256 `b9d9804f37800d5de5d67331c5cab44cb7cfa633c8408dfab5bcf86d1cb79fae`).
- Freeze manifest sha256 `02514cb89f7229458e631232cf42930db36530cf26becf544485517d37ede41e`.
- Freeze record: `docs/reviews/wme/prereg-wme-freeze-2026-11-19.md`.
- Tag `prereg-wme-v2-frozen`, not pushed.

**Status.** Frozen. No confirmatory outcome exists. Launch waits on the §II.28.3 preconditions.
