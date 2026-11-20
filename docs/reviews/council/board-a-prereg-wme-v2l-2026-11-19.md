# Board A — check of amendment §II.31 (v2l) to the frozen PREREG_WME

**Verdict: ACCEPT WITH CONDITIONS.** The *design* content of §II.31 is outcome-blind, pre-data, append-only and
admissible under Part I §11.4. C1(a) and C1(b) preserve the registered E1 estimand. The cap raise to 110 cannot bias
anything as recorded. The *implementation* is not ready for the Phase L re-run, for three reasons:
- the registered C1(a)/(b) selection is not wired into the Phase L driver;
- the C1(b) launch path would refuse its own canary set;
- the run-health wrapper changes the registered retry semantics. Specifically, it loses the technical cause, burns
  attempts on non-failure incidents, and can drive units into their fallbacks.

Conditions R1–R6 must be met before Phase L is re-run. Conditions R7–R10 must be met before the first confirmatory
unit. Conditions D1–D4 are documentary and go into a dated §II.31 addendum.

**Scope.** Reviewed main **d385577**, 19–20 Nov 2026 UTC, as Board A. I read:
- the frozen text and the §II.31 diff;
- `RULING-wme-c1-phasel.md`, including addenda 1 and 2;
- the launch report (`.md` and `.json`) and the parity evidence;
- the v2l local report (`.md` and `.json`) and the v2l drill JSON;
- `git diff b817550..d385577 -- src scripts tests`.

**Handling.** I wrote only this file. I made no commits and used no GPU or Lambda. I opened no `data/evals` and no
sealed split, and I edited no prereg or code. The two small CPU probes below ran from `/tmp` with
`CUDA_VISIBLE_DEVICES=''` and changed nothing in the repository.

## Verification

**Tests.** I ran `WSP_VENV=wme-c1 . scripts/env.sh; CUDA_VISIBLE_DEVICES='' python -m pytest -q
tests/test_wme_c1_*.py tests/test_wme_runguard*.py` via `agent-tmux`. Result: **98 passed, 1 deselected (the `slow`
marker in `pyproject.toml` addopts), 67.97 s, exit 0.**

**Append-only.** `git diff af4face..d385577 -- docs/PREREG_WME.md`: **269 insertions, 0 removed lines**. All the added
lines are §II.31 (`docs/PREREG_WME.md:3875-4080`).

**Parser module.** `src/wsp/wm/genaudit.py` is byte-identical at af4face, b817550 and d385577 (sha256 `52ad2bbd…`).
v2l does not touch it. See finding 4 for the registered hash.

**Probe P1 (cause loss).** `ctl.technical(WorkerError(..., cause="out_of_memory"))` gives `out_of_memory`. The same
exception wrapped as `health.HealthIncident(...) from exc` gives **`None`**.

**Probe P2 (stale skip).** I built a long-lived `HealthMonitor`, as `Study.run` keeps one, and advanced it through
`sealed_eval`. I then called `HealthMonitor(runs).skip_stage("e1_gen")` on a separate instance, exactly as
`controller.stage_gen` does. The next `ste_calibrate` raised **an order incident: `skipped_without_declaration:
['e1_gen']`**.

## Answers to the eight questions

1. **Outcome-blind and pre-data: yes.**
   - The launch report and its JSON record "No confirmatory outcome exists; nothing sealed was read".
   - The parity evidence (sha256 `1e7bf367…`, verified) is labelled "exposed engineering items only". It contains
     no `sealed`, `eval` or `wme-c2` key.
   - There is no study log under `data/wme/runs/wme-c1/` (only `ctl`, `instances`), so no `ProductionGuard` unit has
     run.
   - The local test used the exposed `wme-p3h` holdout only.
   - The commit order is: launch report 15:22 → ruling 15:24 → addendum 15:32 → §II.31 15:57 → addendum 2 15:58, all
     19 Nov UTC.
   - Minor: the launch report's filename says 2026-11-20, while its commit is 19 Nov (D4).
2. **Append-only: yes.** There are 0 removed lines. One factual claim inside the appended text is wrong (finding 4).
3. **Pre-specified and mechanical: yes in text, no in code.**
   - §II.31.2 gives a closed trigger (pass rule, or Δ breach) evaluated once on exposed items. (b) is therefore
     legitimately "not an adaptation" under §11.4: no confirmatory result exists, and the switch depends only on
     engineering hashes and timings.
   - The code does not execute this rule (finding 1).
   - The throughput branch is effectively inert at C = 110. With P + S_L ≈ 91, it would need Δ > 19 GPU-h, i.e.
     r > 35. That is harmless, but say so (D3).
4. **The estimand is preserved: yes, subject to R5.**
   - Under C1(a), the engine, prompts (`panel.chat_prompt`), stop rule, batch 128, file order and base/winner pairing
     are unchanged. Only kernel numerics change, symmetrically for base and winner.
   - Under C1(b), greedy decoding, 128 tokens, the same prompt strings, the same stop ids and single-instance pairing
     are kept. vLLM batching is ruled irrelevant to pairing (addendum 2, answer 2), which I accept.
   - Residual risk: the parser input text under C1(b) is vLLM's incremental detokenisation (`c.text`,
     `vllmops.py:95`), not `tok.decode(ids, skip_special_tokens=True)` as in `genaudit.py:375`. Prompts are also
     re-tokenised by vLLM (`vllmops.py:87`). Either could make parser input differ from the HF definition for the same
     token ids (R5).
5. **The cap: consistent as recorded, with one unbounded path.**
   - The cap of 110 GPU-h-eq is the user's decision before any data. It is inside $1,000, which is 185.9 GPU-h-eq at
     5.379 $/GPU-h.
   - It changes no claim, and every arm is funded from the same ledger, so it cannot bias anything as recorded.
   - **Later chair raises** are discretionary and may happen after confirmatory results exist, for example after E2
     or the sealed read and before S-TE. The frozen §II.22.6 says no chair decision can override the stop
     (`PREREG_WME.md:2924`).
   - A discretionary raise or refusal that is conditional on seen results is optional stopping on the remaining
     claims. It must be made mechanical (R8, D1).
6. **Health rules and order, retries and losses: yes, there are real risks** (findings 2, 3, 5, 6). The text is
   fine. The code changes retry semantics and can convert halts into fallbacks.
7. **Code vs text: partly.**
   - Correct:
     - the deterministic scope (`genengine.deterministic`, which restores state);
     - process-wide cuBLAS, unset in the vLLM child;
     - the fla/kernel binding check;
     - `genaudit.py` untouched;
     - `GEN_ENGINE` inactive and bound into `engine_config`;
     - `CapGeneration`, which refuses lower caps, unannounced chair raises and caps above $1,000;
     - the ledger cap and the dead-man re-base per generation;
     - `HealthHalt` in `begin`.
   - Gaps: findings 1–7.
8. **Missing before Phase L:** R1–R6.

## Findings

**1. Critical: the §II.31.1 pass rule and the §II.31.2 selection are not executed by the Phase L driver.**
- `genengine.gen_pass` and `select_engine` are called only from tests. `grep` finds no caller in `src` or
  `scripts`.
- The `gen_repeat` handler exists (`launchops.py:281`), but `cmd_phasel_parity` and `cmd_phasel_main`
  (`scripts/wme_c1_ctl.py:296-470`) never call it.
- `cmd_phasel_main` checks only `par["parity"]["parity_bitwise"]` (teacher-forced chunks) and `inert["pass"]`
  (`wme_c1_ctl.py:401`). It never checks `canary_gpu0_eq_gpu1["hf_gen"]`. That is exactly the field that was `false`
  in the failed launch, which nonetheless recorded precondition 2 as "yes".
- Nothing writes r, Δ, P, S_L, C, the booleans or the selected engine to the evidence, although §II.31.2 requires it.
- As wired, pass rule items 1 and 2 are not enforced, and the (a)/(b) choice would be made by hand.

**2. Major: run-health wrapping erases the registered failure cause and changes retry semantics.**
- `HealthMonitor.wrap_fn` catches every exception and re-raises it as `HealthIncident(...) from exc`
  (`health.py:350-354`).
- `harness.run_units` passes that exception to `ProductionGuard.failed`, where `technical()` sees `HealthIncident`
  and returns `None` (probe P1). The `attempt_fail` record therefore carries `cause: null` for every error, technical
  or not.
- On resume, `begin` → `RetryPolicy.attempt(key, None)` accepts the retry.
- Result: a non-technical failure, which §II.22.6 and harness semantics say halts and is not retried as technical,
  becomes indistinguishable from a technical one and is retried after `incidents_resolved`. This contradicts
  §II.31.5 item 3 ("a technical failure that the registered retry policy would retry is retried").
- The audit trail of the 7 registered causes is also lost.

**3. Major: non-failure incidents burn attempts and can reach the registered fallback.**
- `check_order` runs inside `fn` (`health.py:347`), after `guard.begin` has recorded `attempt_start`. The
  idle/telemetry abort is also raised inside `fn` through `Remote.watch`.
- Each order, idle or telemetry incident therefore records a failed attempt with cause `None`. "A halt never consumes
  an attempt" (`PREREG_WME.md:4051`) holds only for the subsequent refusal, not for the tripping attempt.
- A unit that trips deterministically, such as a CPU-bound `e2_arm` that is idle for more than 15 minutes, or the
  stale-skip case in finding 5, is retried after each resolution. On the 4th `begin` it hits `RetryExhausted` and
  takes its fallback:
  - an E2 arm-episode becomes the base winner;
  - an S-TE or E1 unit becomes missing, i.e. "engineering abort" (§II.29 G17).
- This is a loss disposition caused by the monitor, not by the unit. It can be **differential between arms**: the
  WM-E arm's planner CPU time differs from the StatNN or direct arms, so it would bias H-E2.
- Addendum 2 answer 3 (CPU units get a progress heartbeat) reduces but does not remove this. Order and telemetry
  incidents still burn attempts, and `e2_arm` is a mixed CPU/GPU/LLM-reader kind with no registered classification.

**4. Major (documentary, pre-existing): the registered `gen.parsers` hash does not match the frozen `genaudit.py`.**
- §II.31.1 says "`genaudit.py` is unchanged, so the `gen.parsers` sha256 `43ec3beb…` still holds"
  (`PREREG_WME.md:3898`). That is false.
- `43ec3beb…` is `genaudit.py` at 2737136 (17 Nov 07:52). Commit 63ca07f (17 Nov 10:13, before the freeze and before
  any data) changed the file to `52ad2bbd…` by adding `_generate`/`generate_sampled`. The frozen manifest
  (`docs/prereg/wme_manifest.json:1028`) and §II.4 delta 4 (`:1902`) still cite the old hash.
- The diff touches only the generation functions. The five registered parser functions (`squad_norm`,
  `classify_v1`, `detect_loop_v1`, `audit_v1`, `audit_v1_single`) are byte-identical. So the estimand is intact, but
  the registered integrity check would fail if anyone ran it.
- No code verifies this hash at runtime.

**5. Major: the E1-not-designated skip is lost (false order incident on the S-TE path).**
- `controller.stage_gen` declares the skip on a fresh `HealthMonitor` (`controller.py:315`).
- `Study.run` keeps its own long-lived monitor (`study.py:167`). That monitor never reloads `state.json`, and its
  next `_save()` overwrites the skip.
- `ste_calibrate` then raises an order incident (probe P2). By finding 3, repeated resolution leads to
  `RetryExhausted`, and S-TE becomes "engineering abort".
- The path depends on an outcome (whether E1 is designated), so this inserts chair intervention and a possible loss
  conditional on a result.
- The unit test (`tests/test_wme_c1_v2l.py:432-453`) uses the same instance and a fresh one, so it misses this.

**6. Major: "withdrawing" an in-flight request does not stop it.**
- On an idle incident, `Remote` calls `transport.cancel(id)` (`ctl.py:113`). That only deletes the inbox file, and
  the transport docstring says so: "withdraw a not-yet-started request".
- The worker is single-threaded and keeps executing the started handler. `Ctx.check` only checks the fence epoch.
  When the handler finishes, `os.replace(path, done/…)` (`worker.py:193`) raises `FileNotFoundError` outside the
  try block and kills the worker loop.
- A hung GPU unit keeps the GPU and the resident model state until the dead-worker revive.
- §II.31.5 item 3 ("is withdrawn") is therefore not implemented for the case it targets. The test
  (`test_v2l.py:492`) has no running worker.

**7. Major: the C1(b) activation path is incomplete in the driver.**
- `cmd_phasel_main` hard-codes `set(eng) != {"tf","hf_gen","vllm"}` (`wme_c1_ctl.py:379`). Under
  `GEN_ENGINE="vllm"` the canary returns `vllm_gen`, so the launch would exit "PRECONDITION FAILED".
- Under C1(b), the inert identity is checked only at the stage-W 2,048-token settings without stop ids
  (`launchops.py:97-125`), not at the `vllm_gen`/E1 settings. Engine-level identity is acceptable, but this should be
  stated.
- Parity evidence is read from a fixed `DATE = "2026-11-19"` filename (`wme_c1_ctl.py:40`). A re-run of
  `phasel-parity` would **overwrite** the committed failed-launch evidence (`1e7bf367…`, bound in the launch report).
  If it is not re-run, `phasel-main` would read the stale evidence from before the engine change, which was produced
  without `CUBLAS_WORKSPACE_CONFIG`.

**8. Minor: §II.31 text predates addendum 2 and contradicts it.**
- §II.31.5 ends "no exemption is introduced here" (`:4066`). Addendum 2 answer 3 introduces the GPU-kind vs CPU-kind
  split and binds it on §II.31.
- The prereg must carry the binding text, including a registered table of kind → {GPU, CPU} for every unit kind in
  `health.STAGES`, and how mixed kinds (`e2_arm` with LLM-reader waits) are classified.

**9. Minor: the §II.22.6 / §II.28.2 cap wording is not explicitly superseded.**
- §II.31.4 replaces 90 with 110, but it does not name the frozen sentences it supersedes: "90 GPU-h-eq combined … no
  chair decision can override it" (`:2924`), and §II.28.2's "absolute 90 GPU-h-eq / $1,000 … unchanged" (`:3680`).
- It also leaves later raises discretionary (finding on Q5).

**10. Observations (no action needed).**
- The Δ branch is inert at C = 110.
- The local test cannot show that C1(a) cures the H100 nondeterminism. Its registered path was already reproducible
  locally, and its report says so honestly.
- `gen_pass` correctly also requires the parity GPU-0 hash to equal the repeat hash.
- The dead-man t_max now defaults to the remaining cap.
- `CapGeneration.check` enforces the $1,000 ceiling at the ledger convention.

## Conditions

**Before the Phase L re-run:**

- **R1 (finding 1).** Wire the rule into `phasel-parity`/`phasel-main`:
  1. Call `gen_repeat` (n = 3) on GPU 0.
  2. Take the GPU 0 and GPU 1 `hf_gen` hashes from `parity`.
  3. Take the deterministic HF inert rows from `inert_identity`.
  4. Compute `gen_pass` with item 3 included explicitly, not via `passed["pass"]` alone.
  5. Compute `select_engine` with S_L from the dead-man ledger and C from `cap_state()`.
  6. Write every input and output to a new evidence file and refuse to continue unless the recorded engine equals
     `GEN_ENGINE`.

  Add a CPU test that drives this with fake handlers for both outcomes.
- **R2 (finding 7).** Replace the hard-coded engine set with `genengine.canary_engines()`. Write Phase L evidence to
  new, run-tagged filenames, and never overwrite committed evidence. Bind the parity and inert evidence to the
  engine-config sha256 they were produced under, and refuse a mismatch.
- **R3 (finding 2).** Preserve the cause:
  - classify `exc.__cause__` (or carry the cause on `HealthIncident`) so that `attempt_fail.cause` is the registered
    technical cause or `None` exactly as without the monitor;
  - on resumption, a unit whose last cause is not technical is not retried except through a `RetryPolicy.repair`
    record (registered cause, commit, outcome-blind attestation).

  Test both cases.
- **R4 (finding 3).** Order, idle, telemetry and progress incidents must not consume an attempt. Either check order
  before `guard.begin`, or record such aborts with a registered `infrastructure`-type disposition that does not count
  toward the 3-attempt limit. Add a CPU test showing that 3 consecutive incident trips on one unit never reach
  `RetryExhausted` or the fallback.
- **R5 (Q4).** Under C1(b), recompute `text` as `tok.decode(ids, skip_special_tokens=True)` with the HF tokenizer
  before parsing. Pass prompts to vLLM as the HF-encoded token ids (`tok.encode(chat_prompt, add_special_tokens=False)`),
  or assert their equality. Unit-test parity of the text and ids path on a fixture.
- **R6 (findings 6, 5).** Make withdrawal real:
  - fence the in-flight request, by bumping the owner epoch so that `Ctx.check` raises `Fenced`, or by killing and
    reviving the worker process;
  - make `Worker.handle` tolerate a missing inbox file;
  - route `skip_stage` through the Study's monitor, or reload state under a lock before every `check_order`.

  Test both with a live `LocalTransport` worker.

**Before the first confirmatory unit:**

- **R7 (finding 8).** Implement addendum 2 answer 3 on the pending branch, with a registered kind → {GPU, CPU}
  table that covers every kind in `health.STAGES`. The 15-minute progress rule must use sub-unit progress
  (`ctx.progress`) that every CPU-heavy handler emits, including the E2 planner and the LLM-reader waits. Board A
  re-checks this diff.
- **R8 (Q5, finding 9).** Make later raises non-discretionary with respect to outcomes. Register now that when
  `CapExceeded` would fire and combined spend plus the priced remainder of the registered design is ≤ $1,000, the
  chair **must** raise, by the priced remainder × 1.25 or to the $1,000 ceiling, whichever is lower. Announcing to the
  user is a notification, not a gate. A raise or non-raise is decided and logged from ledger seconds only, with an
  outcome-blind attestation. Encode the attestation in `CapGeneration`.
- **R9 (finding 4).** Add a launch check that recomputes the per-function source digest of the five `gen.parsers`
  functions (and the file sha256 `52ad2bbd…`), and record both in the attestation evidence.
- **R10.** Re-run the CPU drill, including scenario H extended with R3, R4 and R6. Commit its JSON with sha256 before
  launch.

**Documentary (one dated §II.31 addendum, before Phase L):**

- **D1.** State that §II.31.4 supersedes the "90" and "no chair decision can override it" wording of §II.22.6 and
  §II.28.2, replacing it with the R8 rule. CapExceeded remains an engineering abort at the cap in force.
- **D2.** Record addendum 2 answers 1–4 verbatim. Withdraw "no exemption is introduced here".
- **D3.** Record that the Δ branch is inert at C = 110 (it needs r > 35). Record that C1(b)'s inert identity is
  checked at stage-W vLLM settings.
- **D4.** Erratum: the `gen.parsers` file hash is `52ad2bbd…` since 63ca07f (pre-freeze), and the five parser functions
  are unchanged from `43ec3beb…`. Note the launch report's filename date (20 Nov) against its commit date (19 Nov
  UTC).

## BLUF

1. Verdict: **ACCEPT WITH CONDITIONS**. The design is sound; the implementation is not ready for the Phase L re-run.
2. §II.31 is outcome-blind and pre-data: no study log, no sealed or eval access, exposed canary items only.
3. Append-only is confirmed: 269 lines added, 0 removed. `genaudit.py` is untouched by v2l.
4. C1(a) and C1(b) preserve the E1 estimand. C1(b) is legitimately not an adaptation. R5 fixes the parser-input text
   and tokenisation.
5. Critical: the Phase L driver never runs `gen_repeat`, `gen_pass` or `select_engine`, and ignores the HF GPU 0 vs
   GPU 1 hash (R1).
6. The run-health wrapper erases the technical cause (`cause: null`) and changes retry semantics (R3).
7. Order, idle and telemetry incidents burn attempts and can force fallbacks, possibly differently across E2 arms
   (R4, R7).
8. Withdrawing an in-flight request doesn't stop it, and the E1-skip declaration is lost, giving a false S-TE halt
   (R6).
9. The cap of 110 is fine. Later chair raises must become mechanical and ledger-only to rule out optional stopping
   (R8, D1).
10. The registered parser hash `43ec3beb…` has been stale since before the freeze (the parsers themselves are
    unchanged). Add an erratum and a launch check (R9, D4). Tests: 98 passed, 1 deselected.
