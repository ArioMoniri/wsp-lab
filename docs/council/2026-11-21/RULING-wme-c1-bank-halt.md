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
