# Chair ruling, 21 Nov 2026: wme-c1 bank-phase halt (canary OOM)

**Input.** `docs/reviews/wme/wme-c1-run-halt-bank-2026-11-21.md`, on branch `run/wme-c1-phasel-l3` at `59e2e4c`.

## What happened

Phase L part 2 passed. The run then completed 26 bank-phase units, all of them eligible, with 0 failures.

The chunk-boundary HF canary ran out of GPU memory. The cause was that the worker process kept roughly 77–79 GiB of
field and R-cache tensors resident on the GPU. This happened inside a canary, not inside a unit, so no attempt was
consumed.

The worker was restarted and the closing canary passed. The instance has been terminated, and the Lambda API confirms
it.

**Ledger:** 6.219 GPU-h-eq, $29.94 at provider price, with 103.78 GPU-h-eq left under the 110 cap. No raise has
occurred.

## Decisions

All decisions are engineering only and were made before any sealed read.

**K1. Canary isolation.**
- **Fix:** every HF canary, and every other non-unit generation check, runs in a fresh child process on the GPU, as
  vLLM already does. The child receives only the model state it needs. Freeing memory in the parent alone is not
  sufficient.
- **Test:** a CPU-level test proves that the child path is the one used, and that a failure in the child is reported
  back to the parent.

**K2. Canary errors are run-health incidents.**
- A worker or child error inside a canary is written to the study log as a run-health incident, using the existing
  incident kinds, before the run halts.
- **Test:** a test covers this path.

**K3. Resuming the bank.**
- **Approved:** an outcome-blind field repair. Re-collect the 8 field groups and the vocabulary unit on the new
  instance, then verify each one bitwise against the hashes recorded in its eligible payload.
- **On any mismatch:** halt and report. Do not substitute the new values.
- **Budget:** about 0.5 GPU-h-eq, charged to the ledger.
- **Status of the units:** the repair re-materialises inputs. It creates no new unit result, and the original 9 units
  remain the registered results.

**K4. Code identity.**
- K1 and K2 change the code, so Phase L parity must re-run on the new clean main commit (about 0.85 GPU-h-eq). The
  rules P2 and N4 bind parity evidence to its commit.
- **Pass rule:** unchanged.
- **Extra requirement:** the HF canary hash must still equal `7299ac4f…`, the committed reference. Any other value is
  an incident and the run stops.

**K5. Order.**
1. Implement K1 and K2 with TDD, then commit.
2. The chair merges and names the clean commit.
3. Board A does a quick check of the K1/K2 diff only.
4. Phase L parity runs on that commit.
5. `phasel-main` runs.
6. The K3 field repair runs.
7. The bank resumes, then the remaining phases.

**Projection.** About 84–90 GPU-h-eq out of 110. No raise is expected. If R8 triggers a raise, it is announced to the
user.

## Addendum (21 Nov 2026): K3b, indirect verification of the vocabulary unit

**Problem.** The vocabulary unit's eligible payload recorded only a manifest hash. That hash includes wall-clock
timings, and the original manifest was lost with the instance. A direct bitwise check of a re-collected vocabulary
field is therefore impossible.

**Decision: K3b, indirect bitwise verification.** This is outcome-blind and adds no new results.

1. **Re-collect the vocabulary unit** on the new instance.
2. **Re-run a completed dependent unit.** Pick at least one bank-state unit that is already completed and eligible,
   and whose recorded result depends on the vocabulary field. If none depends on it, pick the earliest unit that does.
   Re-run it as a verification computation outside `run_units`, using the re-collected field and vocabulary.
3. **Compare.** Its output hashes must equal the hashes recorded in that unit's eligible payload, bitwise.
   - **Match:** the re-collected vocabulary is accepted. One `repair` record is written that names the verifying
     unit. The original unit results stay the registered results.
   - **Mismatch, or no dependent completed unit exists:** record an incident and halt, then report to the chair.
     Nothing is substituted.
4. **Cost.** The verification compute is charged to the ledger.

**Also required.** Every field-family payload records per-file data hashes from now on. This is already implemented
in v2r.

## Addendum 2 (21 Nov 2026): run5 repair mismatch on field g0 (mmlupro gradient file)

**Input.** `docs/reviews/wme/wme-c1-run-halt-repair-2026-11-21.md`, on branch `run/wme-c1-run5` at commit `68defa6`.

**D1. Diagnostic approved.** Before any further repair attempt:
- Re-collect field group g0 twice on one instance. Compare the two collections file by file, bitwise. Also compare
  each against the quarantined re-collection.
- Charge the work to the ledger, with a budget of about 0.5 GPU-h-eq.
- Writing an owner epoch is acceptable.
- Create no unit result and no repair record.

**D2. Consequences, declared now, before the diagnostic.**

**(a) The two collections differ from each other.** The cause is nondeterminism in the collection backward pass.
- Bitwise field repair is impossible on this stack, and the 26 eligible bank units cannot be verified against a
  re-collected field.
- Remedy: a **contract transition** under the registered guard mechanism.
  - All bank-phase units, field included, are superseded and recomputed.
  - Field collection runs under deterministic algorithms, if that makes it reproducible (test this first, on the same
    instance), with a stated slowdown.
  - The field is collected **once** and kept on a persistent volume for the rest of the run. If no persistent
    volume is available, the remaining bank work must finish on one instance.
- This is outcome-blind. Bank units are construction data: no sealed item has been read and no claim outcome exists.
- Board A does a quick check of the transition before it runs.

**(b) The two collections match each other but differ from run3.** The cause is a difference in process or input
state, for example the K1 model reload.
- Diagnose it by comparing the panel content hashes and the model-state hash with run3's recorded values.
- If the cause is found and fixed, retry K3 on a new commit, followed by a parity run.
- If no cause is found, apply (a).

**(c) Both collections match run3.** Treat run5's mismatch as transient. Retry K3 once. A second mismatch applies (a).

**Cost.** Option (a) means redoing about 4.5 GPU-h-eq of bank work. The 110 cap still holds; if R8 raises it, the
user is told.

## Addendum 3 (21 Nov 2026): branch (a) confirmed; the transition plan

**Input.** The D1 diagnostic is recorded in `wme-c1-diag-g0-2026-11-21-l5d.json` on branch `run/wme-c1-run5` at commit
`fa347ad`.
- On one instance, two collections of the `mmlupro_gold_lp` gradients for field g0 differed.
- The nondeterminism enters at the layer-2 linear-attention block, in the backward pass.
- Everything else matched bitwise: all other objectives, every K-FAC factor and every kept factor.

The pre-declared branch (a) therefore applies.

**T1. Close the ledger gap first.**
- Write the reconcile record for `1addc266`, using its API-confirmed 0.502 h. This stops the 0.756 GPU-h-eq phantom
  surplus.
- This is a ledger fix only.

**T2. Determinism probe** (diagnostic; it may run outside the repo).
- On one instance, run the `mmlupro` g0 collection 3 times under `torch.use_deterministic_algorithms(True)`, with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and deterministic SDPA.
- Report bitwise equality across the 3 runs and the slowdown.
- If a kernel raises because it has no deterministic implementation, report which kernel. Do not substitute a kernel.

**T3. Field persistence options.** Price them before any choice is made.
- (i) A Lambda persistent filesystem in the run's region. Report price and availability from the API and docs; the
  field needs about 204 GB.
- (ii) Keeping one instance alive from field collection to the end of E2 and S-BoN.

The chair chooses between them after T2. Any extra dollar cost counts toward the user's $1,000.

**T4. The transition itself** must be implemented and then checked quickly by Board A before it runs.
- Supersede every bank-phase unit through a registered contract transition.
- Collect the field once, deterministically if T2 passes.
- If T2 fails, collect the field once **without** bitwise repair. It is then never re-collected for the rest of the
  run, so persistence becomes mandatory.
- Recompute the bank.

This is outcome-blind: the bank is construction data, nothing sealed has been read, and no claim outcome exists.

## Addendum 4 (21 Nov 2026): T2 passed; persistence option (ii); T4 specification

**Input:** `docs/reviews/wme/wme-c1-transition-prep-2026-11-21.md`, at commit `4fc91d6`.
- **T1.** `1addc266` was reconciled at 0.502 h. The study ledger now equals the dead-man total, 9.607 GPU-h-eq.
- **T2.** Collection under the project's existing deterministic settings was bitwise reproducible in 3 of 3 runs. No
  kernel raised and none was substituted.

**P. Persistence: option (ii), approved.**
- Keep one instance alive from field collection through the end of S-BoN.
- No filesystem is created. That keeps launches free to use any region, and avoids an unapproved dollar cost.
- **If the instance is lost:** re-collect the field deterministically, then do a normal bitwise K3 repair against the
  hashes recorded in the transition contract.

**T4. Contract transition.** Implement it with TDD; Board A does a quick check before it runs.

1. **Field collection runs under the deterministic context.** This is the existing `deterministic()` context manager,
   with R11 restore semantics. It applies to field and vocabulary collection only. Every other unit and engine keeps
   its registered settings, and global state must be restored afterwards. A test covers this.
2. **Contract transition** under the registered guard mechanism:
   - every bank-phase unit and every field unit is superseded;
   - the open incident `field_repair_mismatch` (seq 277) is resolved by the transition record, which cites this ruling;
   - the new field's per-file hashes are recorded in the first new field payloads, which supports later K3 repair.
3. **Timing first.** Run one full deterministic collection of g0 across all six objectives. Report the slowdown
   against the registered collection, with wikitext's long sequences as the known risk. If the projection then goes
   over the 110 cap, R8 applies and the user is told.
4. **Recovery path.** A lost instance triggers deterministic re-collection plus bitwise K3. Both are tested.
5. **New commit.** Because the code changes, a parity run on the new clean commit is required. The canary reference
   stays `7299ac4f…`.

## Addendum 5 (22 Nov 2026): run6 bank-start halt

**Input.** `docs/reviews/wme/wme-c1-run-halt-bank-order-2026-11-22.md`, on branch `run/wme-c1-run6` at commit
`99fddd1`.

**What passed.** Parity, resume, field timing and the transition to generation 1 (seq 326) all passed. Two checks
from the earlier rulings came out as follows:
- **C5:** the projection is 96.98 GPU-h-eq. That is under the 110 cap, so no R8 raise is triggered.
- **C6:** the deterministic mmlupro hash equals T2's on a different instance. Bitwise K3 recovery across instances is
  therefore supported.

**What failed.** The bank stage was refused before any attempt by an `order` incident (seq 338). The order monitor's
stage pointer was not scoped to the contract generation, so it still held generation 0's position. Separately, the
heartbeat setup was broken: the worktree had no `.venv` link. The dead-man therefore terminated the idle main
instance. A close-canary that then ran against the dead instance recorded `worker_error` (seq 340).

No attempt was consumed and no field was collected.

**Decisions.** The v2w fix covers four items:
1. The order monitor is scoped to the contract generation.
2. A `resolve-incident` command is added.
3. Heartbeat hardening, plus a preflight that requires two live heartbeats before any unit is issued.
4. A new drill scenario I: transition followed by bank start.

Incidents 338 and 340 are resolved under this addendum, citing the diagnoses above.

## Addendum 6 (22 Nov 2026): run7, canary OOM mid-bank

**Input.** `docs/reviews/wme/wme-c1-run-halt-bank-oom-2026-11-22.md`, on branch `run/wme-c1-run7` at commit
`e9f2e0e`.

**Status before the halt.** The following all passed:
- preflight and the heartbeat gate;
- parity, with hf_gen `7299ac4f…`;
- the resume;
- the resolution of incidents 338 and 340;
- the deterministic field collection and `bank:base`;
- two bank chunks;
- the order monitor.

**The fault.** A chunk-closing canary ran out of memory in its child process. `Runtime.release()` emptied the CUDA
cache before dropping the R-cache, bank and field tensors, so the parent process still held about 24 GiB. This was
recorded as incident `worker_error` (seq 569).

The 16 bank sides that were waiting on that canary are `retained_ineligible`. Each consumed one generation-1
attempt, as registered. No sealed read and no claim outcome exists.

**Decisions.**
1. **The v2x fix is approved.**
   - `Runtime.release()` first drops every GPU reference, then runs `gc.collect()`, then `empty_cache()`.
   - The parent's free GPU memory is checked and recorded before any child process starts. If it is below the
     child's requirement, that is an incident and the child does not start.
   - A CPU test covers the release order.
   - A small **local-GPU** smoke test must also run before the next launch: allocate a stand-in tensor of at least
     4 GiB as R-cache or field, call `release()`, and verify the memory is returned.
     - It is engineering only and touches no study data.
     - Limits: ≤ 8 GB of VRAM, and it must coexist with the running study M′ job.
2. **Incident 569 is resolved** under this addendum. Diagnosis: the release order in `Runtime.release()`, fixed in
   v2x.
3. **Recovery.** Re-collect the generation-1 field deterministically and verify it with a bitwise K3 repair against
   its recorded per-file hashes. The 16 ineligible sides are re-run with their remaining 2 attempts. No other
   change.

## Addendum 7 (22 Nov 2026): chair decision on the registered bank PhasePause (§II.13 item 9, §II.14)

**Input.** `docs/reviews/wme/wme-c1-run-pause-bank-2026-11-22.md`, on branch `run/wme-c1-run8` at commit `c7db897`.

**Ledger seconds. These are the only basis for this decision.**

| Quantity | Value |
|---|---|
| Bank phase measured | 5.320 GPU-h (generation 0: 0.873, superseded; generation 1: 4.447) |
| Pause threshold | 5.663 GPU-h (1.25 × 4.531) |
| Remaining | 44 units, about 1 GPU-h of unit time plus about 1 GPU-h of instance overhead |
| Combined spent | 19.229 GPU-h-eq |
| Conservative projection | 105.2 GPU-h-eq, against the 110 cap |

The overrun comes from engineering re-execution: the superseded generation 0, and run7's 16 sides that became
ineligible. It does not come from the registered work itself.

**Decision: RESUME the bank phase beyond 1.25 × its price** for the remaining 44 registered units, within the absolute
cap.
- This decision uses no outcome. No bank result was inspected for it, and no sealed or claim data exists.
- R8 governs any later cap raise mechanically. Any raise is announced to the user.
- The C1 agent records this decision in the study log as the chair's PhasePause resumption, citing this addendum.

**Parity.** Reuse run8's parity evidence if the registered code accepts it for the unchanged commit `b2f1f71`.
Otherwise, re-run parity on `b2f1f71`.

## Addendum 8 (22 Nov 2026): run9, compensated inerts outside the field support

**Input.** `docs/reviews/wme/wme-c1-run-halt-bank-inert-2026-11-22.md`, on branch `run/wme-c1-run9` at commit `655b2ff`.

**Where the run stands.** 207 of the 210 bank units are eligible under generation 1. The 3 that are not are the
compensated inert units `inert:05/06/07:comp`. Each one edits `mlp.up_proj`, which is outside the R1 field support,
so the R(E) contraction raised `UnsupportedTensor` (seq 2356).

No sealed read has taken place, and no claim outcome exists.

**R1. Option (a).** For every bank unit, R(E) is contracted over the **supported tensors only**, meaning the field's
writers and the vocabulary rows.
- The omitted tensors are recorded per unit as `r_omitted: [tensor names]`.
- Rationale. At access tier R1 the world model can only see the collected field. A function-preserving edit whose
  compensation sits partly outside that support is the case the inert control exists to probe: does the WM
  falsely predict an effect from what it can see?
- The measured bank outcome for these units is still computed in full, exactly as registered.
- D-C2 and C2-I stay descriptive (§II.21.1). The D-C2 report must state that the R(E) for these 3 units excludes
  `up_proj`.
- No other unit is affected. No other registered bank or E2 unit has run into an unsupported tensor.
- This decision is outcome-blind and engineering only. Record it as a PREREG_WME §II.31.13 note, append-only.

**R2. The efficiency fix is approved.** When a chunk runs no unit, because every unit in it was already eligible,
skip its closing canary. A chunk with zero executed units has no result to bracket. A test covers this.

**R3. Incident 2356 is resolved** under this addendum. Diagnosis: R(E) support handling for the compensated inerts,
fixed by R1. The retry of `inert:05:comp` uses a repair record that cites this addendum.

**Addendum 8, erratum (22 Nov 2026).** R1 says "for every bank unit". The binding scope is **inert units only**, which
is how the chair instructed it and how v2z implements it. Every non-inert unit still raises on an unsupported tensor.

## Addendum 9 (23 Nov 2026): E2 stopped for re-pricing under R9.2

**Input.** `docs/reviews/wme/wme-c1-run-stop-e2-rate-2026-11-23.md`, on branch `run/wme-c1-run10` at commit
`c1fc5fa`.

**Where the run stands.**
- The bank is complete: 210 of 210 units are eligible, and prep is done.
- Two E2 arm-episodes have been measured, at 1.84× and 1.63× the registered estimate of 699.8 s.
- At that pace E2 alone is projected at about 85–90 GPU-h, against a confirmatory price of 53.67.
- No sealed read has happened and no claim outcome exists.

**P1. Profile at $0 before re-pricing.** This is CPU only, read-only, and uses the existing logs and heartbeats for
`wm_e:00` and `statnn:00`. Decompose each arm-episode's wall time into four parts:
- GPU-busy evaluation;
- planner and fit compute, split into CPU and GPU;
- waits on LLM readers;
- overheads.

Report the GPU-busy fraction.

**P2. Outcome-preserving speed-up (optional).** If a speed-up is available, it counts as engineering only if it
meets all of the following:
- it changes no registered quantity: arms, budgets, items, seeds or the planner algorithm;
- it reproduces both completed E2 units **bitwise**, in their full payloads with timing fields removed. This must be
  verified on the instance before any new E2 data;
- it comes with tests;
- it lands as a new commit, followed by parity.

**P3. Re-pricing amendment.** After P1 and P2, append a PREREG_WME §II.31.14 note, dated and append-only. It
records:
- the measured per-arm-episode rate, including any P2 speed-up;
- the resulting confirmatory price.

This is outcome-blind: only ledger seconds are used, and no outcome is read.

**P4. Cap.** If the re-priced projection exceeds 110, the chair raises the cap before the E2 units that need it.
- The raise is to the re-priced projection × 1.10, and never beyond the user's $1,000 total for the 9B study.
- It is announced to the user at the time it is made.
- No claim, α, N, margin or stopping rule changes.

## Addendum 10 (23 Nov 2026): P1 result, P2 approved, P3/P4 rules

**P1** (`docs/reviews/wme/wme-c1-e2-profile-2026-11-23.md`, branch `run/wme-c1-9b-p1`).
- The GPU work per arm-episode is about 450 s, under the registered 699.8 s.
- The whole overrun is CPU planner time, which the registered price did not include.
- That planner time is dominated by exact repeats: 86–87 % of rows repeat, and the R(E) calls cover only about 1k
  distinct tuples.

**P2 approved.** Two memos inside `e2arms.run_planner`:
- an R(E) memo keyed by the primitive-id tuple;
- a whole-batch prediction memo keyed by exact bytes.

Expected saving: about 5.3 GPU-h-eq, bringing `wm_e` to about 1.25× and `statnn` to about 1.07× of registered.

Two conditions:
- Row-level deduplication and any thread-count change are **excluded**, because neither is bitwise.
- Before any new E2 data, re-run `e2:wm_e:00` and `e2:statnn:00` on the instance as verification. Each must match its
  committed run10 payload exactly, ignoring timing fields only. A mismatch is an incident and halts the run.

**P3. Re-pricing**, recorded as a PREREG_WME §II.31.14 note, append-only and based on ledger seconds only.
- The E2 per-arm-episode estimate becomes the registered estimate × 1.30. This covers the post-memo `wm_e` rate
  measured in the verification run, rounded up to the nearest 0.05× if that rate turns out higher.
- The S-BoN estimate is scaled by the same factor, since it uses the same planner.
- Every other phase keeps its registered price.
- R9.2 still applies to the new estimate. If any arm's measured rate exceeds 1.5× **its re-priced estimate**, E2 stops
  again for re-pricing.

**P4. Cap.**
- After P3, compute the projection as spent plus the re-priced remainder of every phase, plus instance overheads.
- If the projection exceeds 110, the cap rises to the projection × 1.10, never beyond the user's $1,000 total.
- The raise is logged as a cap generation before the units it funds, and announced to the user by the chair.

## Addendum 11 (23 Nov 2026): run11, E2 unit keys depend on the checkout path

**Input.** `docs/reviews/wme/wme-c1-run-halt-e2-verify-key-2026-11-23.md`, on branch `run/wme-c1-run11` at commit
`2f9beb6`.

**Problem.** E2, prep, S-BoN, sealed and gen recipes carry absolute local paths, so their unit keys changed with the
checkout directory. `verify-e2` refused, as designed (seq 2750). No E2 data, no sealed read and no claim outcome
exist.

**Decision: option (a), with no code change.** From now on, every run is made from a fresh clean checkout of the
named commit **at the fixed run10 checkout path**. The agent recreates that directory for each run, and the
recomputed keys reproduce the logged keys exactly. The agent records the path in the P2 evidence.
- Option (b), path-independent keys, is deferred to after the study. It would remap every eligible key mid-run.
- Incident 2750 is resolved under this addendum. Diagnosis: path-dependent recipe keys, fixed procedurally by (a).
- Before `verify-e2`, check that the recomputed keys for `e2:wm_e:00`, `e2:statnn:00` and `e2:random:00` equal the
  logged keys. Any mismatch halts the run.

**Cap.** The P4 cap at the current ledger is 125.90 GPU-h-eq, +1.7 % against the announced 123.83. That is within
the 5 % bound, so it is approved. After `cap-raise`, re-base the dead-man's instance-hour limit with the same rule
`up` uses.

## Addendum 12 (23 Nov 2026): run12, guard construction after the P4 cap raise

**Input.** `docs/reviews/wme/wme-c1-run-halt-guard-cap-2026-11-23.md`, on branch `run/wme-c1-run12` at commit
`75c5462`. The fix is on branch `run/wme-c1-9b-v3b` at commits `23e1a65` (code) and `e2d3241` (drill evidence).

**Problem.** `ProductionGuard.__init__` re-recorded the registered §II.31 cap generation (110) on every
construction. After the P4 raise to 128.386 GPU-h-eq (cap generation 1, seq 2812), `record_cap` refused it ("a cap
generation never lowers the cap"). No guard could then be constructed, so `run --stages e2` failed before any unit
was issued. Steps 1–7 of the run12 GO had completed:
- parity passed;
- the resume passed;
- K3 was bitwise;
- incident 2750 was resolved (seq 2799);
- the key check passed;
- `verify-e2` passed (seq 2803);
- `reprice` recorded seqs 2806 and 2808;
- `cap-raise` recorded seq 2812;
- the dead-man was re-based.

No E2 data, no sealed read and no claim outcome exist.

**Decision 1: v3b approved as written.**
- The constructor re-records the registered cap only if no persisted cap generation equals it field for field
  (`_cap_persisted`).
- A lower cap absent from the log is still refused, and that test is kept.
- The change to `test_wme_c1_v2l::test_cap_generation_rules` is confirmed. Its intent was "an old guard cannot lower
  the cap", and that intent is preserved: the cap stays raised and no new record is written. The refusal on
  construction was the defect.
- It is a code change, so a new parity run precedes the resume.

**Decision 2: the first main node's setup failure is accepted as an infrastructure retry of the launch.** The node
failed with `SETUP_FAILED:cuda-runtime` before any request. No unit and no incident were involved, and termination
was confirmed. A fresh dead-man pre-check (l12m2) preceded the relaunch. The node's time, about 0.44 GPU-h-eq, is
reconciled at the next resume.

**Decision 3: closing the halted main instance through the ledger is accepted.** `ctl close` could not construct a
guard. The instance was therefore terminated through the dead-man API and confirmed, then closed in the study log at
its API-confirmed termination time by the ledger's own `reconcile_instance` (dead-man source), at 1.631 GPU-h-eq. No
unit had executed since the start canary, so under R2 no closing canary was owed.

**Resume.**
- Run from the merged main commit at the fixed run10 checkout path (addendum 11): pre-checks, preflight, parity, main
  launch, `phasel-main --resume`, `repair-field`, the key check.
- Re-run `verify-e2`, because `runguard.py` changed since `8e2e9b9`.
- Then `run --stages e2,…` and every later registered phase.
- `reprice` and the P4 cap generation (128.386) stand. No further raise is authorised: any further raise means stop
  and report first.

## Addendum 13 (23 Nov 2026): run13, a stale E2 environment after the canary's model release (incident 2868)

**Input.** `docs/reviews/wme/wme-c1-run-halt-e2-env-release-2026-11-23.md`, on branch `run/wme-c1-run13` at commit
`c531688`. The fix is on branch `run/wme-c1-9b-v3c` at commits `be46617` (code), `c0093b2` (drill evidence) and
the chunk-boundary test commit.

**Problem.** `Runtime.release()`, which runs before a canary's child engines, did not drop the cached E2 environment
(`rt._e2`). The next E2 request reused an Env holding the released model, and `e2:random:00` failed in
`modelops.at_base` (incident 2868; `attempt_fail` with cause None). `verify-e2` had passed (seq 2856). No E2 result,
no sealed read and no claim outcome exist.

**Decision 1: v3c approved.** `Runtime.release()` drops and closes the cached E2 Env. A test covers the run10 latent
path: a chunk-boundary canary between two E2 units rebuilds the Env through `env_for`. It fails on the pre-v3c code.

**Decision 2: the attempt budget of `e2:random:00`.** The run10 attempt (seq 2705, generation 1, epoch 46) was in
flight when E2 was stopped under R9.2. That stop was chair-ordered and was not a failure, and the attempt produced
no payload and has no outcome. Under §II.31.8 R4, the last sentence of §II.31.10 N1 and §II.31.5 item 3 ("a halt
never consumes an attempt"), it is ruled **uncounted**: charged, but not counted toward the 3. Today's code-defect
failure (seq 2867/2870) counts, and so does the retry after the repair record.

**Implementation note.** The log has only one writer of `attempt_aborted`: `ProductionGuard.failed` with an
`IncidentAbort`, and only for an attempt in flight under the current owner epoch. Disposing of the run10 attempt
would need a new write path, which this ruling forbids. The conservative count is therefore kept: the replayed
budget counts 2 of 3, so 1 attempt remains in generation 1. If that attempt fails, the registered fallback applies.

The fix to the count (a persisted disposition of an interrupted attempt from a closed owner) is deferred to after the
study, together with the path-independent keys (addendum 11, option (b)). The only unit with an R9.2-interrupted
attempt is `e2:random:00` (seq 2705). It is the only `attempt_start` in the log without an outcome.

**Decision 3: repair.** `resolve-incident 2868` and `record-repair --label e2:random:00 --cause code_defect_crash`,
both citing this addendum, are written before any further E2 data.

**Resume.**
- Run from the merged main commit at the fixed run10 checkout path: pre-checks, preflight, parity, main launch,
  `phasel-main --resume`, `repair-field`, the key check, a fresh `verify-e2`, then `run --stages e2,…`.
- No cap raise beyond 128.386 is authorised. Any further raise, or a projection above the cap, means stop and report
  first.

## Addendum 14 (24 Nov 2026): run14, the R9.2 stop at `e2:wm_e:02`; re-pricing `lora_tv` episodes; cap tripwires withdrawn

**Input.** `docs/reviews/wme/wme-c1-run-stop-e2-r92-wm_e02-2026-11-24.md`, with its $0 diagnosis, on branch
`run/wme-c1-run14` at commits `6f32b2e`, `931be75` and `79ee0d4`.

**1. Reading of R9.2.** The stop stands under the stricter per-unit reading (§II.13 item 9: "a measured unit rate").
`e2:wm_e:02` took 1,524.3 s, which is 1.68 × its re-priced estimate of 909.8 s and above the limit of 1,364.7 s.
Addendum 10 and §II.31.14 ("any arm's measured rate") would allow an arm-mean reading. On that reading `wm_e`
averages 1,143 s over its three units (1.26 ×), or 1.18 × over the post-memo units alone, and would not have breached.
The laxer reading is not adopted after seeing the timings.

**2. Diagnosis (accepted): legitimate workload variation, not a defect.** Episode 02 is a `lora_tv` episode, as are
episodes 2, 5, 8, 11, 14, 17, 20 and 23. Each of its primitives touches 16 weight tensors instead of 1. The R(E)
cache misses rose from 137 to 3,168, with 480 s of GPU contraction instead of 23 s, and the dev evaluations took
about twice as long (870 s against 443 s). Planner CPU time and the P2 memos are unchanged.

**3. Re-pricing (supersedes item 3 of the earlier draft of this ruling).**
- On `lora_tv` episodes only, `wm_e`, `statnn`, `field_ranked` and `direct_bon_two_stage` get a unit estimate of
  1,524.3 s and an R9.2 limit of 2,286.5 s. That is 31 remaining units: `wm_e` 7, `statnn` 8, `field_ranked` 8,
  `direct_bon_two_stage` 8.
- Every other arm, and these four arms on `steer` and `proj_remove` episodes, keeps 909.8 s and 1,364.7 s.
- For `statnn`, `field_ranked` and `direct_bon_two_stage` this rests on the workload model, not on a measured
  breach. The model was checked against one unit (`wm_e:02`: 1,506 s predicted, 1,524.3 s observed).
- This changes timing only. It is held in the operator watch (no `src/` change); the log-level phase price, the
  1.25 × PhasePause and the cap are unchanged.

**4. Projection and tripwires.**
- The P4-rule projection with the `lora_tv` re-price and one relaunch, which is already in the overhead term and is
  counted once, is **128.593 GPU-h-eq**:
  - charged 39.517;
  - re-priced remainder 80.366;
  - `lora_tv` excess 31 × 614.5 s = 5.292;
  - overheads 3.418, which is the registered P4 rule recomputed on the ledger (the last three launches now include
    run13 and run14).
- That exceeded the chair's projection gate by 0.207. The overhead value was not substituted and no estimate was
  lowered to fit.
- The chair's launch gate ("projection ≤ cap") and tripwire D(ii) ("charged + remaining above the cap means stop")
  are **withdrawn**, together with the claim that reaching the cap would by itself abort the study. Reason: after the
  one-shot P4 raise, §II.31.8 D1 leaves R8 as the only raise mechanism. R8 is mechanical: it applies when
  `CapExceeded` would fire and (charged + priced remainder) × 5.379 ≤ $1,000. It leaves no room for a discretionary
  raise or refusal. A chair gate that stopped short of R8 conflicted with it.
- Launching is allowed: charged (39.517) is far below the cap.

**5. Checks on a copy of the study log (no write to the real log).**
- At charged 128.0, `check_before` records an R8 cap generation instead of raising `CapExceeded`. The new cap is
  128.386 + 1.25 × the priced remainder.
- The priced remainder covers the guard's priced phases: bank, confirmatory (with its reprice record) and S-TE.
  S-BoN counts only inside the S-BoN stage's process. The `lora_tv` re-price is not included.
- Examples: remainder 30.99 gives a cap of 167.13; 4.99 gives 134.63; 0.85 gives 129.44; 0 gives `CapExceeded`
  (O1). The absolute ceiling is 185.9.
- The dead-man's limits are set at launch from the cap in force. After an R8 raise it would terminate the instance
  about 0.75 h later (its instance-hour and dollar limits equal the old cap) unless it is re-based. The operator
  watch therefore re-bases it on any new cap generation, with the same rule `up` uses: `tighten` with the new
  allowance and cap generation, then restart the watcher on the new cap, with clean ticks verified.

**6. Notifications and halts.**
- Report when charged first passes 110 and again at 120, with the projection. This is a notification only, not a
  gate.
- Report any R8 record immediately. The $1,000 ceiling (185.9) is absolute.
- Tripwire D(i) stays as an ordinary halt: any further R9.2 or incident halt is reported as usual.

**7. `e2:statnn:02`** (seq 3082) was in flight at the stop and has no payload. As in addendum 13, the conservative
count is kept: 1 of 3 attempts used, and no new write path.

**8.** `e2:wm_e:02`'s payload stays eligible. Nothing is redrawn.

**9. Operator changes (no `src/` change).**
- The watch is polled every ~30 s with the per-arm, per-family limits. It was unit-tested on the ledger: under the
  old limits it flags exactly `wm_e:02`; under the new limits it flags none.
- The launch gate's full output is written outside the repository (`ctl_gate_tee.py`).
- Residual risk: any stop still lands mid-unit and burns that unit's attempt, because stopping between units needs
  a `src/` change.

### Addendum 14, correction (24 Nov 2026, appended during run15)

Item 5 of addendum 14 is completed and corrected as follows. These are the R8 facts established on copies of the
study log before run15.
- At charged 128.0 (charged + 0.75 > cap), `check_before` records an R8 cap generation instead of raising
  `CapExceeded`. The new cap is the previous cap + 1.25 × the priced remainder, never beyond 185.9 GPU-h-eq.
- **When the priced remainder is 0 (O1), there is no raise and `check_before` raises `CapExceeded`.** This was
  reproduced with confirmatory and S-TE both measured above their prices.
- R8's priced remainder is Σ max(0, re-priced price − measured) over the priced phases of the guard in use: bank,
  confirmatory (with its `reprice` record) and S-TE. S-BoN is included only inside the S-BoN stage's process, where
  `set_sbon_price` runs. **The `lora_tv` re-price of addendum 14 item 3 is not in R8's remainder:** it is held only
  in the operator watch, and no log record carries it.
- Examples on copies at charged 128.0: remainder 30.99 → cap 167.13; 4.99 → 134.63; 0.85 → 129.44; 0 →
  `CapExceeded`.
- The dead-man's limits are set at launch from the cap in force. After an R8 raise it would terminate the instance
  about 0.75 h later unless it is re-based.
- Addendum 14 item 5's statement that "the operator watch therefore re-bases it on any new cap generation" is
  **withdrawn**. By the run15 ruling (item 2) the automatic re-base is disabled until it is tested and approved. On
  an R8 record the operator re-bases by hand with the P4 procedure; otherwise the dead-man terminating the instance
  is the accepted safe failure. The launch of run15 despite that report-first condition is recorded as a deviation
  in the run15 evidence.

## Addendum 15 (24 Nov 2026): run15, the R9.2 stop at `e2:direct_bon_two_stage:02`; measured `lora_tv` prices

**Input.** `docs/reviews/wme/wme-c1-run-stop-e2-r92-bon2stage02-2026-11-24.md` and the run15 deviation record, on
branch `run/wme-c1-run15` at commit `733b0d7`.

**1. The stop stands; the diagnosis is accepted.** `e2:direct_bon_two_stage:02` took 4,892.7 s against its
addendum-14 `lora_tv` limit of 2,286.5 s. Its 512 screens took 8.19 s each (median) instead of 1.93 s: a screen
evaluates 4/64 of a bundle, and its fixed per-candidate cost (applying up to 4 × 16 tensors, theta hash,
displacement norms) dominates. This is legitimate registered workload. The unit stays eligible and nothing is
redrawn.

Attempts without an outcome are kept on the conservative count (generation 1):

| unit | counted | left |
|---|---|---|
| `e2:bon_steering:02` (seq 3217) | 1 | 2 |
| `e2:statnn:02` | 2 | 1 (eligible) |
| `e2:random:00` | 3 | 0 (eligible) |

**2. The addendum-14 workload model is withdrawn as a pricing basis.** It failed on its first test:
- `scripted_reader` measured 1,337.5 s, `fugu_team` 1,270.3 s and `random` 1,060.5 s, against predictions of
  840–1,010 s;
- it over-predicted `field_ranked` (942.4 s, predicted about 1,440 s) and `statnn` (1,378.0 s, predicted about
  1,520 s);
- it put `direct_bon_two_stage` at about 1,360 s against 4,892.7 s measured.

**3. Measured rule for `lora_tv` episodes 5–23** (one rule for all 11 arms; a chair decision made before the data it
governs, from ledger seconds only, to avoid relaunches):
- **Estimate** = max(909.8 s, the arm's own measured post-memo episode-2 unit seconds). `wm_e` stays 1,524.3 s.
  R9.2 limit = 1.5 × the estimate.
- **`direct_bon_two_stage`:** 4,892.7 s, limit 7,339 s. This is an R9.2 re-price after a measured breach.
- **Every other arm:** the rule re-prices arms that are below the 1.5 × trigger, from one measurement per arm. It
  lowers `statnn` (1,378.0 s) and `field_ranked` (942.4 s) from 1,524.3 s.
- **`bon_steering` and `lora_sft`:** 909.8 s and 1,364.7 s until their own episode-2 unit completes. Then the same
  rule applies, computed mechanically from the ledger by the operator watch (`r92v3`) before episode 5. There is no
  further discretion.
- **Episode 2's remaining units** keep the addendum-14 limits.
- **`steer` and `proj_remove` episodes** keep 909.8 s and 1,364.7 s for every arm.
- This is the **last** pre-emptive `lora_tv` re-price. After it, a breach stops E2 under R9.2.
- No claim, α, N, margin, arm, budget, item, order or stopping-rule change.

**4. The chair's error.** The chair's estimate of about 4.9 GPU-h of `lora_tv` excess left `direct_bon_two_stage`
at a guess. The measured excess for episodes 5–23 is **11.955 GPU-h-eq**, and it is incomplete: `bon_steering` and
`lora_sft` are not yet measured.

**5. Projection, recorded as is.**
- Charged 44.680 + log remainder 77.001 + `lora_tv` excess 11.955 + canaries 0.716 + one relaunch 2.674 =
  **137.03 GPU-h-eq**, above the cap of 128.386.
- No estimate is lowered to fit and no chair raise is made: P4 is spent, and R8 is the only mechanism.
- Relaunch is authorised: charged 44.680 is within the cap in force. The user is told 137.

**6. R8 and the phase pause, checked on a copy of the log.** The ledger was walked forward from today's state at the
measured instance-to-unit ratio (1.02), after one relaunch.
- The confirmatory **1.25 × PhasePause** (69.149 × 1.25 = 86.44 measured) comes first, at charged about **126.9**.
  It is a registered, resumable pause (§II.13 item 9, §II.14).
- **R8 fires** at charged about 127.7, with a priced remainder of 1.845 (the unmeasured S-TE price), giving a cap
  of 130.69. It repeats (+2.31 each time) while that remainder is positive, up to the $1,000 ceiling (185.9). With
  a remainder of 0 there is no raise (O1).

**7. Stepping between units.** The controller CLI has no existing single-unit, single-chunk, episode or stop
option (`run` takes `--iid` and `--stages` only). So a watch stop still lands mid-unit and burns that unit's
attempt. No `src/` change is made, and no engineering speed-up of the screen path either.

**8. Operations.** The automatic dead-man re-base stays disabled. Reports go out at charged 110 and 120, on any R8
record, on any halt and when E2 is done. On any further halt, the report also states the total spent so far on
relaunch overhead.

## Addendum 16 (27 Nov 2026): run16, the R9.2 stop at `e2:field_ranked:05`; mechanical re-pricing replaces the stop

**Input.** `docs/reviews/wme/wme-c1-run-stop-e2-r92-fr05-2026-11-24.md`, on branch `run/wme-c1-run16` at commit
`317639c`.

**1. The stop and its diagnosis are accepted.** `e2:field_ranked:05` took 1,638.8 s, against its addendum-15 limit
of 1,413.6 s. Its R(E) cache reached 6,880 misses (840.6 s), against 496 (64.9 s) on episode 02. The unit stays
eligible and nothing is redrawn.

Attempts of the units with a start that has no outcome (conservative count, generation 1):

| unit | start without outcome | counted | left |
|---|---|---|---|
| `e2:random:00` | seq 2705 | 3 | 0 (eligible) |
| `e2:statnn:02` | seq 3082 | 2 | 1 (eligible) |
| `e2:bon_steering:02` | seq 3217 | 2 | 1 (eligible) |
| `e2:scripted_reader:05` | seq 3508 | 1 | 2 (to run) |

**2. Diagnosis of the pattern.**
- On `lora_tv` episodes, per-unit E2 time depends on which tensor combinations the episode's primitives reach (the
  R-cache misses). No per-arm price fixed in advance is stable.
- Five stops have cost **14.090 GPU-h-eq** of relaunch overhead plus burned attempts: the run10 rate stop, then
  `wm_e:02` (run14), `direct_bon_two_stage:02` (run15) and `field_ranked:05` (run16). Meanwhile aggregate E2 cost
  runs near or under its price.
- The stop-on-breach rule is a chair rule (addendum 10 P3: "E2 stops again"), not frozen text. §II.13 item 9
  requires a measured rate above 1.5 × to be re-priced by dated amendment before the data it governs. It does not
  require a stop.

**3. New rule for the rest of E2.** This replaces the addendum-10 stop. It is outcome-blind and concerns timing
only: no claim, α, N, margin, arm, budget, item, order or stopping-rule change for the claims.
- **a. Mechanical re-price, no stop.** When a completed unit exceeds 1.5 × its current estimate, its arm's estimate
  for the remaining episodes of the same family (`steer`, `proj_remove` or `lora_tv`) becomes max(current estimate,
  that unit's seconds). The watch writes this as a dated, sha256-chained, append-only record in an operator evidence
  file outside the live checkout (`~/wme-c1-ops/reprice/records.jsonl`), before the next unit of that arm starts.
  The pre-authorisation is this addendum. If the record cannot be written and verified, the watch stops the chain:
  it fails closed.
- **b. Defect stop kept.** Any unit above 4 × its current estimate, or above 3 h of wall time, still stops E2 and is
  reported.
- **c. Cost controls unchanged and binding:** the cap in force, R8, the $1,000 ceiling, the 1.25 × PhasePause and
  the dead-man.
- **d.** Every re-price record is committed with the run evidence at the next boundary, and listed in the E2-done
  report with the per-arm estimates.

**4. Test on the ledger (before launch).** Replaying the recorded units under the estimates in force when each ran
produces **re-price records, not stops**, for exactly the three units named, and stops on nothing:

| n | unit | family | estimate before | limit before | unit seconds | estimate after |
|---|---|---|---|---|---|---|
| 0 | `e2:wm_e:02` | `lora_tv` | 909.8 s | 1,364.7 s | 1,524.3 s | 1,524.3 s |
| 1 | `e2:direct_bon_two_stage:02` | `lora_tv` | 1,524.3 s | 2,286.5 s | 4,892.7 s | 4,892.7 s |
| 2 | `e2:field_ranked:05` | `lora_tv` | 942.4 s | 1,413.6 s | 1,638.8 s | 1,638.8 s |

The estimates in force were: addendum 10 for `wm_e:02`, addendum 14 for the rest of episode 02, addendum 15 for
episodes 5–23, and 909.8 s on `steer` and `proj_remove` episodes. These three records are the start of the live
record file.

Further tests on synthetic inputs:
- a broken record chain fails closed (exit 5);
- a unit above 4 × is a defect stop, with no record written;
- a re-price applies to the same arm and family only, and is idempotent;
- a write failure raises;
- an in-flight unit above 3 h is a defect.

**5. Projection.**
- With the `field_ranked/lora_tv` re-price: **137.04 GPU-h-eq** without a launch, **139.72** with one relaunch.
- The addendum-15 projection (138.55) is recorded as is.
- No chair raise; R8 is the mechanism. The automatic dead-man re-base stays disabled.
- Notifications at charged 110 and 120, on any R8 record, on any halt, and when E2 is done.

**6. Reporting latency.**
- The run16 stop was at 24 Nov 21:06:52Z. The instance was closed at 21:12:24Z and the stop report committed at
  21:17:52Z. The report message went out at about 21:18–21:20Z.
- A stall alarm is added: if the chain is stopped and no report has been marked sent within 30 min, an independent
  waiter wakes the operator.
