# Board A — re-check of the v2n implementation of the §II.31 conditions (R1–R11, D1–D4)

**Verdicts**

| Decision | Verdict |
|---|---|
| **Re-running Phase L** (2-GPU `phasel-parity`, then `phasel-main` on the evidence it writes) | **ACCEPT WITH CONDITIONS**: P1 and P2 below. Both are documentary or procedural, and neither needs a code change. |
| **The first confirmatory unit** | **ACCEPT WITH CONDITIONS**: N1–N6 below, plus a Phase L that passes as registered. |

All of R1–R11 and D1–D4 are met in substance. The code has no blocker for Phase L. The one finding with a real
bias channel is N1: payload or log tracebacks are now *uncounted* aborts, which go beyond R4. It touches only
confirmatory units, because Phase L does not run through `run_units`.

**Scope.** I reviewed main **e538903** on 20 Nov 2026 UTC, as Board A. I read:
- `git diff af4face..e538903 -- docs/PREREG_WME.md` (§II.31.8);
- `git diff 861e821..e538903 -- src scripts tests`;
- the drill JSON and its sidecar;
- `RULING-wme-c1-phasel.md` (addenda 1 and 2);
- my earlier review `board-a-prereg-wme-v2l-2026-11-19.md`;
- the commit messages of 2291504 (v2n), cf48211/d3a2d01 (restoring the registered wm files) and 95a474e (R11).

**Handling.** I wrote only this file. I made no commits and used no GPU or Lambda. I opened no `data/evals` or
sealed split, and I edited no prereg or code. A copy of `861e821`'s `src tests` was extracted to `/tmp` only to
compare ruff counts.

## Verification

| Check | Result |
|---|---|
| Append-only | `git diff af4face..e538903 -- docs/PREREG_WME.md`: one hunk (`@@ -3809,3 +3809,443 @@`), **440 insertions, 0 removed lines**. |
| CPU tests (the given command, `CUDA_VISIBLE_DEVICES=`) | **135 passed, 1 skipped, 1 deselected, 185.6 s, exit 0.** Adding `tests/test_wmm_*.py` gives 62 passed, so **197 passed, 1 skipped** in total, the chair's figure. The skip is `tests/test_wme_c1_v2n.py:747`: the pinned 9B tokenizer is not in this cache. That is the Qwen-equivalence test for `cc13cd37` (see N2). |
| Drill evidence | `docs/reviews/wme/wme-c1-drill-v2n-2026-11-20.json` sha256 `08559317…54501` equals its `.sha256` sidecar. `all_pass: true`, 10/10 scenarios, including `H_run_health_halt` (technical cause `infrastructure` recorded, `RepairRequired` on the non-technical one) and `H_withdrawal_and_no_attempt_burn` (3 trips, 3 aborted, 0 failures, worker restarted). It was committed in 2291504, **before** R11 (95a474e) and the genaudit restore (cf48211). |
| `genaudit.py` | sha256 `52ad2bbd…` on main, equal to af4face. `PL.parser_check()` passes, and the five function digests equal `PARSER_DIGESTS`. `panel.py`, `p3gen.py`, `actions.py`, `controls.py` and `hf.py` are also byte-identical to af4face. |
| Ruff / `precommit.sh` | `ruff check src tests`: 21 errors. At 861e821 there were 51, since the unrelated I001s were fixed. v2n **adds** 5: `launchops.py:244,277` F401, `health.py:201,233` S110, `test_wme_c1_v2l.py:291` C408. `ruff format --check`: 9 files would be reformatted. `precommit.sh` stops at `ruff check` (`set -e`, `&&`), so pytest never runs from it. |

## Condition-by-condition

| Condition | Met? | Evidence (file:line) |
|---|---|---|
| **R1** Phase L pass rule and engine selection executed by the driver | **Yes** | `phasel.py:102-106`: `run_checks` issues parity, inert and, under C1(a), `gen_repeat` n = 3. `phasel.py:121-180`: `evaluate` takes pass-rule items 1–3 explicitly (`:134-141`, item 3 through `inert_hf_ok` `:109-113`) and calls `select_engine` with S_L and C (`:142-144`). `proceed` = selected engine == recorded `GEN_ENGINE` ∧ pass ∧ every other precondition, including `tf_canary_gpu0_eq_gpu1` (`:179`). Driver: `wme_c1_ctl.py:344` refuses a worker engine that differs from `GE.config()`; `:346-351` takes S_L from the dead-man ledger and C from `cap_state()`; everything is written to evidence (`:352-365`); the instance is terminated in `finally`; there is a non-zero exit unless `proceed` (`:370`). Tests: `test_wme_c1_v2n.py:113,127,150,202`, both outcomes through the driver. |
| **R2** canary set from the engine, run-tagged never-overwritten evidence, engine-config binding | **Yes** | `wme_c1_ctl.py:436` uses `GE.canary_engines()` (`genengine.py:336-339`). `phasel.py:81-88` `write_new` refuses an existing path, and every filename carries `utc_date()-tag`. `wme_c1_ctl.py:417` `check_binding` on the parity evidence; `:419` refuses unless it `proceed`ed with the recorded engine. Test: `:226,241`. Gap (N3): the main instance's canaries are not compared with the parity instance's. |
| **R3** cause preserved; a non-technical failure is retried only after a repair record | **Yes** | `health.py:509-511` records the incident and re-raises the exception **unwrapped**, so `attempt_fail.cause` comes from `ctl.technical` as without the monitor. `runguard.py:1363-1371`: `RepairRequired` unless a repair record follows the last failure seq; it is raised before `retry.attempt`, so no attempt is recorded. Tests: `:292,314`. Drill: `H_run_health_halt`. |
| **R4** incidents never consume attempts | **Yes for order, idle, telemetry and progress. Over-broad for tracebacks (N1).** | Order: `harness.py:195-196` → `health.py:520-521` → `check_order` before `guard.begin`, so no attempt record. Mid-flight: `runguard.py:1391-1404` writes `attempt_aborted` (charged, `status="aborted"`) and re-raises, which halts the run. `RetryPolicy` (`runguard.py:854-858`): attempt **numbers** stay `attempts+1` and are unique; the **limit** uses `attempts − aborted`. Test `:340` covers 4 trips with no `attempt_fail` and no fallback; `:376` shows that real failures still count. **But** `check_payload`/`check_log` (`health.py:420-440`, called at `:512-513` **after `fn` has returned a completed payload**) also raise `HealthIncident`, so the completed result is discarded, uncounted, and the unit re-run. R4 named only order, idle, telemetry and progress. See N1. |
| R4: can attempts minus aborts bias the H-E2 arms? | **No, for the channel R4 targeted. One residual channel remains (N1).** | Fallbacks (base winner, or "engineering abort") are now reached only through counted, cause-labelled failures, identically in every arm. Aborts mid-flight (idle or `no_progress`) produce no payload, so they cannot select on an outcome. A deterministic arm re-run reproduces itself. Abort time is charged to the shared ledger, and R8 makes the cap response mechanical, so differential abort cost cannot become differential loss. Residual: a nondeterministic unit (the LLM-reader arms, `study.py:485`) aborted *after completion* by a payload or log traceback gets an unbounded, uncounted redraw. That is an arm-specific resampling channel if the traceback is ever outcome-correlated (N1). |
| **R5** C1(b) HF token ids in, HF decode out | **Yes** | `genengine.py:320` uses `tok.encode(chat_prompt, add_special_tokens=False)`; `vllmops.py:93` feeds `prompt_token_ids`; `vllmops.py:104` uses HF `decode(ids, skip_special_tokens=True)` when `hf_text`. Test `:411`. |
| **R6** real withdrawal; skip not lost | **Yes** | `ctl.py:98-120`: cancel, fence to epoch + 1, wait up to 120 s on the heartbeat, else `restart()` (`wme_c1_ctl.py:277-285` kills the worker, vLLM child and gpu1 helper, then relaunches); invoked from `ctl.py:141`. Worker: `worker.py:51-55`, `:153-154`, `:216-219` tolerate a withdrawn file. Skip: `study.py:128-136` through the Study's monitor; `controller.py:313`; the monitor state is reloaded and saved under `flock` (`health.py:346-359`, used by `check_order` `:380+`). Tests: `:480,522,555,560,576`. Drill: `H_withdrawal_and_no_attempt_burn`, `restarted: true`. |
| **R7** kind table and progress | **Yes** | `health.py:88` `KIND_CLASS`, covering every kind (test `:584`). `e2arms.py:124` and the field-feature tick; `worker.py` `Ctx.tick`; `ProgressClient` (`study.py:485-500`) checks the controller stall before each LLM call. Tests: `:589,616,630,763`. |
| **R8** mechanical cap raise | **Yes** | `runguard.py:975-985` `r8_cap` gives min(prev + 1.25 × remainder, $1,000 / 5.379), or None if charged + remainder > $1,000. `:797-821` applies it inside `check_before`, only when `ledger.r8` (`:1084`, the §II.31 cap regime). `CapGeneration.check` (`:946-955`): a chair generation must carry rule `R8` and a basis that reproduces the cap, and `outcome_blind` must be True. Announcement is no longer a gate. Tests: `:666,693`. Observation O1 covers overruns. |
| **R9** parser digests at launch; restored state accepted; undocumented versions rejected | **Yes, with N2** | `phasel.py:54-75`: per-function digests via AST, file sha256, and `pass` = digests equal ∧ file ∈ `GENAUDIT_FILES` (`:38-44`). The restored `52ad2bbd…` is in the set, and the check passes on main (test `:703`, first assert). An undocumented file (any extra byte) gives `file_history None`, `pass False` (`:711-714`). A changed parser function gives `functions_unchanged False` (`:707-710`). It is enforced in `phasel-main` (`wme_c1_ctl.py:421-423`) and recorded as `wme-c1-launch-parsers-…json`. **Residual:** `cc13cd37…` (study M's 46967f8 bytes) is still accepted, although the chair has since ruled that study M never edits registered files and main no longer carries those bytes (N2). |
| **R10** drill re-run, committed with sha256 | **Yes, but stale against the final code (N5)** | The drill JSON plus sidecar are verified above. The scenario H extensions cover R3, R4 and R6. It predates R11 and the restore, and it does not record the commit it ran on. |
| **R11** (chair) deterministic-state restore | **Yes** | `genengine.py:202-243` `global_state`/`restore_state` cover deterministic algorithms and warn-only, cuDNN deterministic and benchmark, the 4 SDPA flags and the CUBLAS env. `deterministic()` snapshots inside the context and restores in `finally` (`:264`), also on an exception, leaving the env alone. `hf_generate` (`:286-297`) compares before and after every call, in both modes, and restores and **raises** on any leak, failing closed. Tests: `:827` (2 priors × 2 modes, including a raising generation), `:866`, `:878`, plus autouse fixtures in the c1 modules. No `wm` module imports `seed_all` or `wsp.eval`, so the worker's baseline is torch's default. **Not registered in the prereg:** §II.31.8 lists R1–R10 only (P1). |
| **D1** cap wording superseded; R8 rule | **Yes** | `PREREG_WME.md:4121-4147`. |
| **D2** addendum 2 verbatim; "no exemption" withdrawn; kind table | **Yes** | `PREREG_WME.md:4149-4185` (withdrawal at `:4165`). |
| **D3** Δ branch inert; C1(b) inert identity at stage-W settings | **Yes** | `PREREG_WME.md:4187-4192`. |
| **D4** parser-hash erratum; launch-report date | **Yes, but now partly outdated** | `PREREG_WME.md:4194-4221`. It describes `cc13cd37…` as the post-freeze state. Main has since been restored to `52ad2bbd…` (cf48211), which §II.31.8 does not record (P1). |

## Item-specific answers

**(a) R1.** Met. `select_engine` is unchanged and correct, and the Δ branch remains inert at C = 110. If C1(a) fails,
the run stops after termination (`wme_c1_ctl.py:370`). The switch to C1(b) is a recorded `GEN_ENGINE` commit followed by
a fresh `phasel-parity` under (b). That second run's S_L includes the first check instance, and its evidence binds the
new engine-config sha.

**(b) R3/R4.** Met. Attempts minus aborts is sound: numbering stays unique, aborts only lift the limit, and every
abort is charged. It does not bias the H-E2 arms through the fallback channel that motivated R4. N1 closes the one
residual channel: uncounted redraws of completed, nondeterministic units.

**(c) R6.** Met. The test and the drill use a live `LocalTransport` worker.

**(d) R8.** Met, and non-discretionary in code. O1 is a liveness observation.

**(e) R9.** The launch check accepts the restored `52ad2bbd…` and rejects any undocumented file or changed parser
function. It still accepts `cc13cd37…` (N2).

**(f) `before_begin`.**
- The hook is keyword-only and optional, and it defaults to `None`, which leaves the registered `run_units` behaviour
  unchanged. Its only production use is `HealthMonitor.check_order`. That function can only raise `HealthIncident`
  (which halts with no attempt recorded), append a registered `incident` record, and advance the monitor stage.
- It cannot begin, succeed, fail or resolve a unit. It narrows behaviour; it does not widen it.
- The frozen manifest registers *behaviour* ("every confirmatory unit runs through `harness.run_units` with
  `ProductionGuard`"), not the bytes of `harness.py`. The draft manifest's `source_sha256` for `harness.py`
  (`d24fef9d…`) already differed from af4face (`e7c88a10…`) before v2n, so no binding byte-check is broken.
- Two things are missing:
  - the hook is named nowhere in the prereg;
  - the launch attestation binds the engine config, but not the code identity (commit or bundle hash) that contains
    the hook.

  Neither blocks Phase L, which does not use `run_units`. Both are N4.

**(g) R11.** Met (see the table).

**(h) Anything blocking Phase L.** There is no code blocker. P1 and P2 are needed so that the Phase L evidence can
be attributed to the reviewed code.

**`precommit.sh`.**
- **Phase L: no condition.** The registered tests were run explicitly here (135 passed, 1 skipped) and by the chair
  (197 passed, 1 skipped).
- **Confirmatory: yes (N6).** A gate that never reaches pytest is not a gate, and v2n itself added 5 ruff errors in
  c1 files.

## Conditions

**Before the Phase L re-run:**

- **P1 (documentary; one dated §II.31 addendum, before Phase L).** Record:
  - R11: its scope (`global_state`, the fail-closed check in `hf_generate`) and its commit 95a474e;
  - the cf48211 restore: `genaudit.py`, `panel.py`, `p3gen.py`, `actions.py`, `controls.py` and `hf.py` are back to
    their af4face bytes, and `genaudit.py` is `52ad2bbd…` again;
  - that §II.31.8 D4's description of `cc13cd37…` is historical;
  - the `run_units(before_begin=)` hook and its contract (N4 (a)).
- **P2 (procedural).** Build the Phase L bundle from a clean tree at a named main commit that contains P1. Record
  that commit and the bundle's sha256 in the run log and the Phase L report. Run `phasel-main` from the same commit,
  using only parity evidence that commit produced. No code change is needed. If the commit is ever written into the
  evidence JSON, it is a new field, and that is N4 (b).

**Before the first confirmatory unit:**

- **N1 (bias; supersedes the R4 scope for tracebacks).** An attempt aborted by `check_payload` or `check_log`
  *after `fn` has returned a payload* must either:
  - (preferred) be recorded as `attempt_fail` with cause `None`. This uses the registered semantics: halt, a repair
    record, and a counted retry, which is also what addendum 2 answer 4 says ("the registered retry budget is
    unchanged");
  - or, if it stays uncounted:
    - carry the discarded payload's sha256 in `attempt_aborted`;
    - require that a deterministic unit's next payload equals it (a mismatch is an incident);
    - allow at most one such uncounted redraw per unit;
    - report the number of post-completion aborts per arm in H-E2.

  Add a CPU test with a reader-arm fixture whose first completed payload is aborted by a log traceback.
- **N2 (R9 narrowing).** Remove `cc13cd37…` from the accepted launch set, or accept it only together with a passing
  (not skipped) run of `test_r9_post_freeze_chat_change_is_qwen_equivalent` on the pinned tokenizer, recorded in the
  parser evidence. Main now carries `52ad2bbd…`, so the narrowing costs nothing. Also add the
  `test_registered_wm_files_are_untouched_by_study_m` check, or an equivalent in a `test_wme_c1_*` module, so that the
  C1 test command covers it.
- **N3 (cross-instance canary identity).** In `phasel-main`, compare the main instance's per-engine canary hashes (`tf`,
  `hf_gen` or `vllm_gen`, `vllm`) with the GPU-0 hashes of the bound parity evidence, and refuse a mismatch before
  `composite_canary` (`wme_c1_ctl.py:458`). Otherwise, an HF engine that is reproducible within one instance but not
  across instances would become the committed reference. The failure would then surface only as a start-canary
  mismatch on a paid confirmatory instance. Add a CPU test.
- **N4 (code identity and the hook).**
  - (a) Register the `before_begin` contract: it may only raise `IncidentAbort` and record incidents, and it never
    calls guard unit operations. Add a test that a `before_begin` raise leaves no `attempt_start`.
  - (b) Bind the controller commit (clean tree) and the worker bundle sha256 in the launch attestation evidence, and
    have `hello` report `GE.global_state()`, so that the worker's process globals at start are evidenced (R11).
- **N5 (R10 currency).** Re-run the CPU drill on the final pre-launch commit and record that commit in the JSON. Commit
  it with a sidecar.
- **N6 (gate).**
  - Fix the 5 ruff errors v2n introduced (`launchops.py:244,277`, `health.py:201,233`, `test_wme_c1_v2l.py:291`).
  - Make the pre-launch gate run the registered test command independently of the repo-wide ruff state. Record its
    result (counts and exit code) in the launch evidence.
  - The repo-wide pre-existing ruff and format debt does not need to be fixed for this study.

**Observation (no condition): O1, R8 under an overrun.**
- `priced_remainder` is Σ max(0, price − measured). Measured includes failed and aborted seconds.
- Once every remaining phase has overrun its price, the remainder is 0, R8 gives no raise, and the run stops at the
  cap even with dollars left.
- That is mechanical and outcome-blind, so it causes no bias, only a liveness cost.
- If the chair wants otherwise, a pre-data floor such as remainder ≥ the next unit's `est_gpu_h` would have to be
  registered before any data.

## BLUF

1. Phase L re-run: **ACCEPT WITH CONDITIONS**, P1 (dated addendum for R11, the genaudit restore and the hook) and P2
   (a clean, named commit for the bundle). There is no code blocker.
2. First confirmatory unit: **ACCEPT WITH CONDITIONS**, N1–N6.
3. Append-only holds: 440 lines added, 0 removed. Tests: 135 passed, 1 skipped (197 passed, 1 skipped with wmm). The
   drill sidecar `08559317…` verifies.
4. R1 is met. The driver runs `gen_repeat`, the GPU 0 and GPU 1 `hf_gen` hashes, inert item 3 and `select_engine`,
   records every input and output, and proceeds only if the selected engine equals `GEN_ENGINE`.
5. R3 and R4 are met. Causes survive, and a non-technical failure needs a repair record. Order, idle and progress
   incidents never consume attempts.
6. Attempts minus aborts does not bias the H-E2 arms via fallbacks. A payload or log traceback after completion is
   still an unbounded, uncounted redraw for the nondeterministic reader arms (N1).
7. R6 is met: fence, then kill and restart, verified with a live worker. The skip is locked and reloaded. R8 is
   mechanical and ledger-only.
8. R9 accepts the restored `52ad2bbd…` and rejects undocumented bytes, but still accepts study M's `cc13cd37…`. Drop
   it (N2).
9. The `before_begin` hook is safe and narrowing, but unregistered. Register it and bind the code identity (N4).
   `phasel-main` should also cross-check the canaries against the parity instance (N3).
10. R11 is met and fails closed. `precommit.sh` never reaches pytest, and v2n added 5 ruff errors: that is condition
    N6 for confirmatory, not for Phase L.
