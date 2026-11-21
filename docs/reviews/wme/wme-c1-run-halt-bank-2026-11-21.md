# wme-c1 confirmatory run: HALT in the bank phase, 21 Nov 2026

**Status: halted; the instance was closed and terminated (API-confirmed). Nothing is running. Waiting for the chair.**

No sealed file has been read. No E1, E2, S-BoN or S-TE unit has run.

## Phase L part 2: passed (14:16Z)

- **Main instance.** `7d6a1235…` (1×H100, us-southeast-1), launched 13:53:46Z through the guarded path.
- **Pre-launch dead-man check.** It passed twice:
  - `wme-c1-launch-deadman-2026-11-21-l3m.json`;
  - `…-l3m-recheck.json`, taken after the N6 gate refused the first attempt before any spend. That refusal was my job-script error: a git-hooks override leaked into the gate's temporary-repo tests.
- **Git-config incident.** Recorded in `wme-c1-launch-gitconfig-incident-2026-11-21-l3m.json`: no check or commit of this run fell in the window.
- **`phasel-main` (bound parity evidence l3).**
  - Inert identity: pass.
  - Canary engines {tf, hf_gen, vllm}.
  - N3 cross-instance canaries equal to parity GPU 0 (`hf_gen` `7299ac4f…`).
  - Bank re-priced (§II.30) at 4.531 GPU-h-eq.
  - Attestation contract `0a8efdac…`, composite canary reference `c7d87cd5…`.
  - Earlier check instances reconciled.
  - Start canary: pass.
  - Evidence commit `ed26b4a`.

## Bank phase: what ran

| What | Count / state |
|---|---|
| Field and pre units (8 field groups, vocabulary, bank base) | Done; chunk 1 closed with a passing canary |
| Bank state units | 16 done |
| Units in total | 26 `attempt_ok`, all **eligible** |
| Failed attempts | 0 |
| Open incidents | 0 |

## The halt

At 15:14:02Z the **composite chunk-boundary canary** (not a unit) failed on the worker with
`OutOfMemoryError: CUDA out of memory. Tried to allocate 2.36 GiB … this process has 77.36 GiB in use`.

**Diagnosis.**
- The worker keeps the bank-phase GPU state (field and R(E) cache tensors) resident. Its GPU memory grew from 26.7 GiB
  (field units) to 45 GiB (state units), and was 77–79 GiB at the canary.
- `serve.h_canary` runs the teacher-forced and `hf_gen` canaries on that same process, and calls `rt.release()` only
  before the vLLM part.
- The `hf_gen` canary under C1(a) (math SDPA; the 64 canary items in one batch, prompts up to 1,574 tokens) needs
  several GiB of transient memory.
- In Phase L the same canaries ran on a fresh worker and passed.

This is an engineering defect in the canary path's memory handling. It is not an engine or determinism failure.

**What I did.**
1. Restarted the worker process with the ctl's `restart()`, which freed the GPU memory. No unit ran.
2. Ran `close --phase bank`: the closing canary **passed** (equal to the reference), so the open chunk's units are
   bracketed and eligible.
3. Closed the instance, then terminated it and confirmed the termination through the API (15:26–15:27Z), and
   reconciled it.

## Ledger (study log)

- Charged: **6.219 GPU-h-eq**. That is the instance time for the Phase L checks and the main instance; unit time is
  0.873.
- Cost: **$33.45** at 4.89 × 1.10, or **$29.94** at the provider price.
- Cap in force: 110 (generation 0). Remaining: **103.78**. No R8 raise has happened.
- Dead-man watcher: 0 error events for the whole run.

## For the chair

1. **The fix (needs your approval; code change, so a new commit and bundle).** Before the HF canaries, `h_canary`
   should drop the resident bank-phase GPU tensors (field and R cache, then `torch.cuda.empty_cache()`), or run the
   canary in a fresh child process as vLLM does. Add a test that the canary runs with the field caches released.
2. **Resumption problem.** The field (204 GB on the instance disk) was lost with the instance. Its 9 units are
   eligible in the study log, so `run_units` will not recompute them, but the bank state units need the field files.
   - One option: an outcome-blind repair that re-collects the field on the new instance and verifies it bitwise
     against the `file_sha256` / `kept_sha256` recorded in the eligible field payloads, refusing on any mismatch.
   - That is about 0.5 GPU-h-eq.
3. **Gap.** A worker error raised in the canary path (rather than in a unit) stops the run but is not recorded as a
   run-health `incident` in the study log.
4. **Projection.** Within the earlier 80–86 GPU-h-eq envelope, plus the overhead so far:
   - about 2.4 GPU-h-eq of Phase L re-run and restart;
   - about 1 GPU-h-eq for the field re-collection and the repeated canaries.

   About 84–90 GPU-h-eq against the cap of 110.
