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
