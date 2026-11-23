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
