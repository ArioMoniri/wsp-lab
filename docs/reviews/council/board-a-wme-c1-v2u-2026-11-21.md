# Board A: quick check of the T4 contract transition (wme-c1 v2u), 21 Nov 2026

**Scope.** This is the quick check required by `RULING-wme-c1-bank-halt.md` addenda 2–4, before T4 runs.
- **Code reviewed:** `git diff 9e06d9b..481db13 -- src scripts tests`, on main at `481db13`.
- **Documents read:** PREREG_WME §II.31.11, `wme-c1-diag-g0-2026-11-21-l5d.json` and `wme-c1-transition-prep-2026-11-21.md`.
- **Also read:** the registered retry text (§II.22.6, the §II.26 guard table and contract generations) and the runguard module docstring.
- **How the board worked:** read-only. No GPU, no Lambda, no sealed file and nothing under `data/evals` was opened.
- **Study log:** read for record metadata only (types, units, generations, counts); no payload was read.

## Verdict: ACCEPT WITH CONDITIONS

The transition is sound in substance:
- The deterministic scope is correct and restores state.
- Supersession is complete and mechanical.
- The generation-0 attestation is protected.
- The hash-chained `load_att` is sound.
- The gates hold.

Four conditions must be met **before the parity run**, because they change code and parity must bind the final commit:
- **C1:** the attempt budget per generation;
- **C2:** a robust P2 rule for evidence-only commits;
- **C3:** the transition bound to the live instance and the worker's code;
- **C4:** a deterministic vocabulary probe.

Two reporting conditions apply before the transition (C5 and C6).

## Verification performed

- **CPU tests.** `tests/test_wme_c1_*.py tests/test_wme_runguard*.py` gave **222 passed, 1 skipped, 1 deselected**, exit 0, in 172 s.
- **Engine config at `481db13`.** Recomputed on CPU:
  - `ec_sha(field=False)` = `cd2b3496537492b9…`, which equals the generation-0 attestation's `engine_config_sha256`;
  - `ec_sha()` = `b98713ec2ff3ad38…`;
  - `contract_state()` on the live study = `pre_transition`.
- **Live study log** (300 records):
  - one `attestation`, generation 0, contract `0a8efdac…`, composite canary reference `c7d87cd5…`;
  - **26 units**, all in the bank phase: 8 field groups, vocab, `bank:base` and 16 sides;
  - each unit has exactly **1 `attempt_start` and 1 `attempt_ok`**, with no failure, no abort and no unit `fallback`;
  - one open `incident`, seq 277 (`field_repair_mismatch`);
  - no `relaunch` record;
  - 4 phase `fallback_registered` records and 1 `repair` (infrastructure, seq 264).
- **`attestation.json`.** It has sha256 `29d121f2…`. Both `.bak-*` copies are byte-identical to it.

## Findings by question

### Q1. Deterministic scope: correct

- **Scope.** `GE.field_collection` wraps exactly `C.collect_grouped` in `h_field_group` and `VF.collect_vocab_rows` in `h_field_vocab`, plus the scratch-only deterministic leg of `h_field_timing`.
- **K3 path.** `h_field_repair` delegates to those same handlers, so the K3 re-collection is deterministic too.
- **Kept factors.** `F.compute_kept` runs after the `with` block exits, so the kept-factor eigendecompositions keep the registered settings. The test asserts the global state at the kept step equals the state before.
- **Restore.** `deterministic()` snapshots `global_state()` and restores it in `finally`, so the R11 semantics hold, including on an exception (tested).
- **Process-wide setting.** `CUBLAS_WORKSPACE_CONFIG` is process-wide and set at `prepare_process`. That is the registered process state for every unit, not an expansion of scope.
- **Static guard.** `test_only_field_and_vocab_collection_enter_the_field_context` pins the call sites by scanning the source.
- **Remaining gap.** The vocabulary collection has **never run** under `use_deterministic_algorithms(True, warn_only=False)`. T2 covered `mmlupro` g0 only, and field-timing covers the six g0 objectives. A kernel with no deterministic implementation would raise inside a bank unit and cost an attempt (see C4).

### Q2. The transition

- **Generation-1 attestation: correct.**
  - It carries the same `inert_identity_bitwise`, `gpu_parity_bitwise` and composite `canary_reference` (`c7d87cd5…`), with the new engine config.
  - Its evidence is the union of the old evidence with the ruling, the T1–T3 prep JSON, the parity evidence and the timing evidence, each as `path sha256:…`.
  - The per-engine HF canary `7299ac4f…` is enforced in two places: by resume (`resume canaries differ from the committed references`) and by the parity C2 check (`hf_gen_eq_committed`).
  - The transition's start canary checks the composite that contains it. The reference is unchanged, as addendum 4 item 5 requires.
- **Supersession: complete.**
  - `StudyState.status` returns `superseded_contract` for any attempt whose (generation, contract) is not the current one.
  - All 26 generation-0 units are therefore superseded mechanically. All 26 are bank-phase or field units, so nothing outside the bank phase is affected.
  - The test asserts the set of statuses is `{superseded_contract}`.
- **Incident seq 277: resolved correctly.**
  - The transition refuses if any open incident other than `field_repair_mismatch` exists (tested with `gpu_idle`).
  - The `incidents_resolved` record cites addendum 4 and the transition's seq.
  - The transition description names the incident and its seq.
- **`attestation.json` (generation 0) is never rewritten.**
  - Its bytes are compared before and after the transition.
  - The new attestation goes to `attestation-g1.json` (refused if it exists) and, authoritatively, to the log.
  - Resume now also checks that `attestation.json` is the log's `contracts[0]` digest.
- **`load_att`: sound.**
  - It reads the last `attestation` or `contract_transition` record from the verified, hash-chained log.
  - It rebuilds the `LaunchAttestation` and fails closed unless its digest equals `s.contract`.
  - It falls back to the file only when no log exists.
  - **Crash window (minor).** If the process dies between the transition record and `PL.write_new(attestation-g1.json)`, the log is still authoritative and correct. Only the convenience copy is missing, and a rerun says "nothing to transition". This is acceptable; record it in the run notes if it happens.

### Q3. Attempt counts across a contract transition: a change is needed (C1)

- **What the code does.** It counts `attempt_start` records per unit across all generations. Each of the 26 superseded units would get 2 attempts in generation 1.
- **The registered text.** It reads "3 attempts in total, counted across relaunches" (§II.22.6) and "across processes, instances and relaunches" (§II.26). These enumerate the axes **within** one contract.
- **Contract generations.** The same registration, and the guard's own docstring, say that "attempts and canaries bind to (digest, generation)" and that supersession is irreversible.
- **The rulings.** Addendum 2 says superseded units are "superseded and **recomputed**".
- **Why supersession is not like canary ineligibility.** The closest registered analogy is "an attempt later made ineligible by a canary counts". That rule exists because a canary failure is a unit- or instance-level event within the contract: not counting it would allow unbounded reruns on a flaky instance. A contract transition is different:
  - it is a contract-level, chair-ruled, Board-checked, outcome-blind repair;
  - it supersedes **every** attempt of the generation at once;
  - the generation-1 unit is a different computation, a new field giving new bank values;
  - transitions cannot be looped cheaply, so the budget stays bounded.
- **The bias argument.** Selection by retrying is already prevented by the "first canary-eligible attempt, whatever its value" rule, so the attempt cap only bounds cost and termination.
  - Counting across generations creates **differential missingness by run order**. The 26 units that happened to run before the halt (all field groups, vocab, base and the first 16 sides) get 2 attempts. Bank units not yet run get 3.
  - Exhaustion makes a bank unit unlabelled, and exhausts a field unit (which stops the bank). That is an arbitrary, order-dependent difference introduced by an engineering defect.
  - It also falls hardest on the field units, which now carry the new SDPA-math memory risk.
  - Resetting the budget per generation is uniform, because all 26 have the same history, and it is outcome-blind.

**Ruling.** The per-generation reading is the registered one. The precise condition is C1.

### Q4. Cause `driver_or_cuda`: correct

- The defect is a non-reproducible CUDA backward kernel in registered mode: the D1 diagnostic located it at the layer-2 linear-attention block.
- Among the 7 registered technical causes it is the only one that fits. The guard's own example for a `ContractTransition` is a driver/CUDA change of the engine-config digest.
- It is outcome-blind, and the guard enforces that.

### Q5. P2 across evidence commits: acceptable in principle, not robust as written (C2)

- **Why the idea is safe.** The order produces exactly two docs-only evidence commits between the code commit and the transition:
  - parity writes its evidence uncommitted, and resume commits it together with the resume evidence;
  - field-timing commits its own evidence.
- **Weakness 1: renames.** `git diff --name-only` uses rename detection by default. A move from `src/…` to `docs/…` lists only the `docs/` path, so it passes.
- **Weakness 2: code inputs under `docs/`.** The bundle and the worker read files under `docs/`:
  - `docs/prereg/wme_manifest.json`, through `MO.manifest`;
  - `docs/reviews/wme/wme-c1-shards-2026-11-19.json`, the shard file read by `serve.py`.

  Both are in `BUNDLE_FILES`, and a change to either passes as "docs-only".
- **Weakness 3: no clean-tree check.** `cmd_transition` and `cmd_field_timing` do not check for a clean tracked tree, unlike `up`, resume and `code_identity`.

### Q6. Gates

- **`run` and `repair-field` refuse until the transition.**
  - `require_current_contract` raises before any remote contact (tested).
  - Live state: `pre_transition`.
- **Resume accepts generation 0 only as "the current config minus the field entry".**
  - `contract_state()` accepts only `ec_sha()` or `ec_sha(field=False)` and refuses anything else (tested).
  - `ec_sha(field=False)` reproduces `cd2b3496…` at `481db13` (verified above).
  - Parity must still bind to the new config, and P2 at resume stays strict, with parity equal to HEAD.
- **Recovery.**
  - After a loss under generation 1, `contract_state()` = `current`. The path is resume, then `repair-field`, through `h_field_repair`, which uses the deterministic handlers.
  - The bitwise K3 check compares against the generation-1 payloads' `file_sha256` and `kept_sha256`; vocab now records `file_sha256`, so there is no K3b.
  - A tampered hash halts with `FieldRepairMismatch` (tested).
  - **Untested assumption.** The deterministic collection is reproducible on a *different* instance. T2 was one instance, and K3 also compares the registered-mode eigendecompositions. A cross-instance mismatch fails closed, so it is safe, but it would force another transition (see C6).

### Q7. Order parity → resume → field-timing → transition → bank: nothing blocks

The board walked the order end to end:

| Step | What happens |
|---|---|
| Parity | Its evidence is left uncommitted. |
| Resume | Strict P2 passes, the worker code identity is checked and the evidence is committed. Docs-only commit 1. |
| Field-timing | A non-unit request under generation 0. Its evidence is committed. Docs-only commit 2. |
| Transition | `same_code` accepts both commits. The instance `open` is idempotent (`instance_resume`). The generation-1 start canary runs. |
| `run --stages bank` | Allowed by `require_current_contract`. |

Two conditions on the order:
- The C1–C4 code must land **before** parity.
- Under persistence option (ii), the first generation-1 field collection must run on the same live instance.

## Condition table

| # | Condition | Before | Owner | Kind | Blocking |
|---|---|---|---|---|---|
| C1 | Count the attempt budget per contract generation for units superseded by a transition (precise rule below) | parity | implementer, then chair note | code + PREREG note + tests | yes |
| C2 | Make the P2 evidence-only rule robust: no renames, no code inputs, a clean tree | parity | implementer | code + tests | yes |
| C3 | Bind the transition to the live instance and to the worker's code | parity | implementer | code + tests | yes |
| C4 | Run a scratch-only deterministic vocabulary collection probe before the transition | parity (code); transition (evidence) | implementer | code + evidence | yes |
| C5 | Project the slowdown under R8 against the 110 cap; tell the user if it goes over | transition | operator, then chair | evidence | yes, if over the cap |
| C6 | Compare field-timing's deterministic `mmlupro_gold_lp.g0` sha with T2's `af39a617…` and report it | transition | operator | evidence | no (report only) |

## Conditions

**C1. Attempt budget per generation.** In `runguard`:
- **The count.** For a unit `u` under the current (contract C, generation G), the counted attempts are the `attempt_start` records with `unit = u`, `generation = G` and `contract = C`, minus the `attempt_aborted` records whose `start_seq` points to such a start. `attempt()` refuses when counted + 1 > 3.
- **Attempt numbers stay global.** The next number is the unit's previous maximum plus 1, so keys stay unique (`.a2`, `.a3`, `.a4`).
- **`fallback()`.** It uses the same per-generation count, so the precondition "attempts remain" means fewer than 3 counted in G. This also removes its current use of raw starts without the R4 abort exclusion.
- **Failure causes.** `last_cause` and `last_fail_seq` are scoped to G, so a failure in an earlier generation can neither block nor license a retry in G.
- **Earlier fallbacks.** A unit `fallback` record from an earlier generation is never revived or discarded by this rule. The transition must **refuse** if the generation it supersedes holds any unit `fallback`. None exists today. Any other case needs a ruling.
- **Unchanged.** The relaunch counter (1 per phase) and every other counter stay global.
- **Tests:**
  - a unit with 1 attempt in generation 0 gets exactly 3 in generation 1 (`.a2`–`.a4`), then its fallback;
  - the limit of 3 within one generation is unchanged;
  - an earlier generation's failure cause does not affect generation 1;
  - the transition refuses when the superseded generation holds a fallback;
  - the existing contract-rollback probe still passes.
- **PREREG.** Add an append-only note under §II.31.11 stating this reading and citing this review.

If the chair rules for global counting instead, the board's view is that this is the less faithful and order-biased reading. It must then be recorded in PREREG with the consequence stated: 2 attempts for the 26 units.

**C2. Robust P2 for evidence-only commits.** `same_code(commit)` is true only if all of these hold:
- `commit` is HEAD or an ancestor of it;
- every entry of `git diff --no-renames --name-status commit HEAD` is under `docs/`;
- no entry is in `BUNDLE_FILES` or under `docs/prereg/`;
- `git status --porcelain --untracked-files=no` is empty at `field-timing` and at `transition`.

Add tests for a rename from `src/` to `docs/`, a change to `docs/prereg/wme_manifest.json` or to the shards JSON, and a dirty tree.

**C3. The transition binds the live instance and the worker's code.** Before any write, `cmd_transition` must check:
- the timing evidence's `instance` equals `LO.guard_instance(a.iid)`;
- a `hello` on that instance reports `code_sha` equal to the parity evidence's `controller_commit`, with `same_code` true for that commit.

Record the `hello` code identity in the transition evidence. `field-timing` records the same identity.

**C4. Deterministic vocabulary probe.**
- **Code.** Extend the non-unit `field_timing` request, or add a sibling non-unit request, with one scratch-only `collect_vocab_rows` under `GE.field_collection`. Delete it afterwards, record its time and `file_sha256`, and write nothing to the live field.
- **Evidence.** The transition requires this probe to be present and successful.
- **If a kernel raises.** Halt and report which kernel to the chair, as in T2. Never substitute a kernel.

**C5. R8 projection.** Field-timing reports the deterministic-over-warm-registered slowdown, per objective and in total, with wikitext as the stated risk. Before the transition, the operator projects the bank, field and recovery cost at that slowdown against the 110 cap. If it goes over, R8 applies and the user is told before anything else runs.

**C6. Cross-instance determinism (report only).**
- Field-timing's deterministic `mmlupro_gold_lp.g0.pt` sha256 is compared with T2's `af39a617…`, and the driver and torch versions are recorded.
- **If they are equal:** the K3 recovery on a new instance is supported by evidence.
- **If they differ:** a lost instance cannot be repaired bitwise, so recovery means another transition. The chair is told that persistence option (ii) is then the only safe path. This does not block the transition.

## Non-blocking observations

- **O1.** `cmd_run` performs no worker code-identity check of its own; it relies on resume. After C3 this is covered for T4. A general check in `run` is advisable later.
- **O2.** `field_timing` starts with a cold registered run, then deterministic, then a warm registered run. The deterministic run benefits from the warm-up, so the reported slowdown is slightly optimistic. Wikitext is the risk to watch.
- **O3.** If the process crashes after the transition record but before `attestation-g1.json` is written, only the convenience copy is missing (see Q2). The log remains authoritative.

## BLUF

1. **Verdict: ACCEPT WITH CONDITIONS.** C1–C4 must land before parity; C5 and C6 are evidence before the transition.
2. **Deterministic scope is correct.** It covers field and vocabulary collection only, with R11 restore; kept-factor eigendecompositions and all other units keep the registered settings.
3. **The transition is sound.** Generation 1 keeps the canary reference (composite `c7d87cd5…`, HF `7299ac4f…`), supersedes all 26 bank and field units, resolves seq 277 citing the ruling, and never rewrites `attestation.json`.
4. **`load_att` is sound.** It reads the last contract record from the hash-chained log and fails closed on any digest mismatch; `ec_sha(field=False)` reproduces `cd2b3496…` at `481db13`.
5. **Attempts (C1).** The budget should be counted per contract generation, so the 26 superseded units get 3 fresh attempts. Counting across generations is order-biased and does not follow the generation binding. The board gives the precise rule above.
6. **The cause `driver_or_cuda` is correct.**
7. **P2 across docs-only commits.** The idea is safe, but the check misses renames, code inputs under `docs/` and a dirty tree (C2). The transition must also bind the live instance and the worker's code (C3), and the vocabulary must be probed deterministically (C4).
8. **Order.** parity → resume → field-timing → transition → bank is mechanically unblocked: 222 tests pass on CPU, and the live study is in `pre_transition`.
