# Board A review: feat/p2-batch (head 9143644)

- Reviewer: reviewer-ml (Board A), read-only
- Scope: `git diff main...feat/p2-batch` (9 files: `src/wsp/eval/batch.py`, `run.py`, `refusal.py`,
  `tests/test_eval_batch.py`, `configs/eval_fast.yaml`, `docs/ROADMAP.md`, `docs/RESULTS.md`, `board/P2.md`,
  `sessions/2026-11-04-p2-batch.md`)
- Context read: `docs/REVIEW_PROTOCOL.md`, `docs/DECISIONS.md` (D1, D2), `docs/ROADMAP.md` §P2
- Ran: `source scripts/env.sh && pytest -q` in the worktree → **159 passed, 8 deselected** (14.5 s).
  No GPU job was run (the zoo build holds the GPU).
- Evidence inspected: `data/zoo/qwen3.5-0.8b/manifest.jsonl`, the two smoke entries
  (`bccb9ab10b5661e0`, `adf416337f5a2277`), `data/evals/qwen3.5-0.8b/index.jsonl`,
  `/home/ubuntu/agent-logs/p2-batch-smoke-20261104-031953.log`, `/home/ubuntu/agent-logs/p2-reproduce-20261103-232354.log`,
  `data/runs/p2/base-final-4096/`.

## Verdict

**REVISE.** The core mechanics are sound: weight restoration, skip keying, OOM continuation, and the revision
pin all check out. Five items are blocking. None needs a redesign, and the fixes are small. The branch should not
be merged, and the zoo-wide run should not start, until they are fixed and the smoke is re-run on the final commit.

## Blocking

### B1. Smoke and `--limit` runs write into canonical zoo entry directories (data loss risk, already happened)
- `src/wsp/eval/batch.py:113` (`out = Path("data/evals") / zoo.name`) and `batch.py:173–183`: for each entry,
  `dest = zoo / r["id"]`, and `write_results(dest, ...)` writes `records.jsonl`, `summary.json`, `manifest.json`,
  then `atomic_json(dest / "eval.json", ...)`. `--limit` changes only the signature; it does not change the
  destination.
- Evidence: `data/zoo/qwen3.5-0.8b/bccb9ab10b5661e0/` and `adf416337f5a2277/` now contain `eval.json`,
  `records.jsonl`, `summary.json` and `manifest.json` written 03:20–03:21 today by the `limit=4`, `max_new_tokens=32`
  smoke. Both are production entries (`edit.json` `eligibility: production`). `index.jsonl` lines 2–3 point at
  those directories.
- Failure scenario: a `--limit 4` run on any production entry overwrites its full-tier `records.jsonl` and
  `eval.json`. Any P3 loader that reads `records.jsonl` from the zoo tree ingests n=4, 32-token numbers as
  production data. The skip logic would re-run the entry later, so the damage is silent.
- Fix: write `--limit` (or any signature that is not the default fast tier) under `data/evals/<zoo>/…` only, or
  refuse `--limit` unless `--out` is given. Move the two smoke artifacts out of the zoo directories (the author
  should inspect them before moving; I did not touch them).

### B2. `git_sha` is the wrong repository's HEAD; smoke ran on uncommitted code
- `batch.py:157` uses `get_git_sha()` (`src/wsp/zoo/common.py:24`), which runs `git rev-parse HEAD` in the current
  working directory, not in the directory the code came from.
- Evidence: the smoke manifest records `git_sha: cff5a13…`. The batch module that ran came from the
  `feat/p2-batch` worktree (traceback path in the smoke log). Commit `9143644` is timestamped 03:21:40 UTC, after
  the smoke finished (03:20). The smoke therefore ran on code that was not yet committed, and its recorded sha
  does not identify that code.
- Fix: compute the sha from the package's repository (`Path(__file__).resolve().parents[…]`), record a dirty flag,
  and record a hash of the source files (`run.py` already has `source_hash`; batch does not). Then re-run the n=4
  smoke on the final commit and confirm the sha matches.

### B3. `eval.json` schema is incomplete and `index.jsonl` appends are not idempotent
- Missing fields: `eligibility` (requested in the review brief; absent from the manifest dict at
  `batch.py:153–171`, and `index.jsonl` lines show `eligibility` absent) and `family`.
- `batch.py:156`: `config_hash` hashes the config file bytes, not the effective configuration. The skip key is
  `signature`, which is the effective configuration (`batch.py:108`). The field name is misleading for the
  reproducibility chain (manifest ↔ effective config hash ↔ git sha).
- `batch.py:184`: every run appends one line per processed entry. Repeated OOM retries (`eval_status: "oom"`,
  which stays eligible for retry) append duplicate lines. A kill between `atomic_json` (`:183`) and the append
  leaves a complete `eval.json` that is missing from the index, and nothing repairs it. The index also mixes
  smoke and production lines with no key to separate them.
- Fix: upsert keyed by `(edit_id, signature)`, or regenerate `index.jsonl` from the per-entry `eval.json` files at
  the end of each run. Add `eligibility`, `family`, `signature`, and an effective-config hash to the manifest.

### B4. `configs/eval_fast.yaml` does not follow D2
- Line 2: `max_new_tokens: 4096`. D2 says 1024. Line 8: `batch_size: 4`. D2 says 32.
- This is a config-only change, but it is blocking. The batch runner defaults to this file
  (`batch.py:193`), and the ROADMAP text says the default is the fast tier. As written, a default run produces
  4096-token results and labels them as D2 screening; with `loop_rate` defined as hit-the-limit, those are not the
  predictor targets D2 defines.
- The subset sizes (gsm8k 50, math500 25, ifeval 25, mmlu_pro 100, hellaswag 100, refusal 50, over-refusal 50,
  KL 50) do match D2.
- Also remove two dead keys: `mmlu_pro.max_new_tokens` (line 23; MMLU-Pro is log-likelihood scored and the key is
  never read) and `loop_rate.from` (line 44; `aggregate()` hard-codes the generative benchmarks). Non-blocking, but
  they mislead readers.

### B5. The ROADMAP's 5,617 GPU-hour figure is wrong under D2 and overstated even for the config as written
- `docs/ROADMAP.md:46–49`. Method: 0.301 min/entry × (50/4) × (4096/32) = 1600× → 482 min/entry → × 700 entries =
  5,617 h.
- Problems: (a) the 1600× factor multiplies batch count by token budget, but the smoke's 288 decode steps include a
  fixed 128-token KL continuation that does not scale with the token budget; the real step ratio for the config as
  written is 218,752 / 288 ≈ 760×, so 1600× overstates by about 2×; (b) it uses 4096 tokens, which D2 has retired
  for the fast tier; (c) "700 entries" does not match the zoo: `manifest.jsonl` has 491 entries, 461 with
  `eligibility: production`.
- Under D2 the correct figure is about 100–150 GPU-hours for 700 entries (see §Throughput). This must be replaced
  before merge, because it is a planning number in a merged document.

## Non-blocking (should fix before the full run, or track)

- **N1. Completeness is not checked.** `select_entries` (`batch.py:61`) treats `eval_status == "complete"` plus a
  signature match as complete. It does not check that every configured benchmark is present with count equal to
  configured `n`. `load_items` (`subsets.py:40`) and `prompts()` (`refusal.py`) silently truncate `[:n]`, so an
  over-large `n` would produce a "complete" entry with fewer items. Add a count check.
- **N2. Signature excludes code version.** A fix to scoring code does not invalidate existing results, and there is
  no `--force`. `--resume` is a no-op (`batch.py:204` pops it). Include the source hash in the signature, or make
  `--resume` meaningful.
- **N3. OOM handling is narrow.** `retry_oom` (`batch.py:68–77`) catches only `torch.cuda.OutOfMemoryError` and only
  around `evaluate`. Two consequences: (a) the reduced batch persists into later entries, because
  `runner.generation["batch_size"] = batch` (`batch.py:149`) mutates the shared dict, so every later entry runs at
  the reduced batch and bf16 numerics change with padding (the manifest does record the batch used, which is
  good); (b) a non-OOM exception in `load_entry_delta` or `apply_zoo_entry` (`batch.py:136–140`) aborts the whole
  run instead of recording an error entry. Restart is idempotent, so this is acceptable, but it should be stated.
- **N4. Restoration tests are partial.** `test_restore_exact_base_between_edits` uses an fp32 `Linear` and no
  `base_context`. `test_batch_loads_once_restores_and_resumes` monkeypatches `apply_zoo_entry` and `evaluate`, so
  the real KL path (`refusal.py:78–95`, `batch.py:38–45`) is not exercised. Add a bf16 test that uses the real
  `apply_delta`, and one test through `evaluate_refusal` with `base_context`.
- **N5. Cost of `base_context` is unmeasured.** Each KL prompt enters `base_context` once, which copies every touched
  tensor to CPU and back (three transfers). For random_dir (8 small tensors) this is negligible. For full-model
  families (lora_sft, chain) it may be tens of seconds per entry at n=50. The smoke at n=4 could not show this. Measure
  it on the first full D2 entry.
- **N6. 30 manifest rows have no `eligibility` key** (chain 20, abliteration 5, random_dir 3, heretic_ara 1, lora_sft 1).
  `select_entries` skips them silently. Log the skipped count, and confirm they are legacy rows that should stay excluded.
- **N7. WikiText window count.** `eval_fast.yaml:28` uses `n_windows: 8`. ROADMAP P2 specifies 64 windows, and
  `base-final-4096` used 64. D2 does not list WikiText, so this needs an explicit decision.
- **N8. The base-reproduction gate has not been run on the batch runner.** The P2 gate (base within ±1.5 pt across
  two runs) applies to this runner. The smoke was n=4 at 32 tokens and says so. Run the base through the batch
  runner at D2 before the zoo-wide run. Note that `base-final-4096` has `model_revision: "main"`; the new runs
  record the resolved commit `2fc06364…`, which matches the zoo header.
- **N9. Relative output path.** `Path("data/evals")` (`batch.py:113`) resolves against the working directory.
  Launching from another directory silently writes elsewhere. Resolve from the repo root or take `--out`.
- **N10. Reserved GPU memory grows across entries.** The smoke log shows `gpu_reserved` 1.52 → 2.03 → 3.05 GB over
  three entries, with no `empty_cache` between them. Probably allocator fragmentation, not a leak, but watch it in
  the long run. Concurrent processes appending to one `index.jsonl` also need a lock (see B3).
- **N11. Test gaps.** Nothing tests: the `eval.json` field set, the OOM path's recorded state, index idempotency, or
  that `--limit` does not touch zoo directories (B1).

## Verified (what is correct)

1. **Base-weight restoration.** The keys captured for restoration come from the same loader that `apply_zoo_entry`
   uses (`load_entry_delta` → `apply_delta`). Capture happens after the restore at the top of each entry and before
   the apply (`batch.py:131–140`), and there is a final restore. `base_context` re-applies edits in a `finally`, so an
   exception inside the KL path does not leave base weights in place. Restore is a same-dtype `copy_`, so it is
   bitwise. The integration test asserts `weights == [0, 1, 1]` (base, then entry A, then entry B), which shows B is not
   contaminated by A.
2. **Skip and resume.** Skip requires `eval_status == "complete"` and a signature match. The signature covers the
   full config, the overridden generation (batch, max tokens), and `limit`. OOM entries are re-selected on the next
   run. File presence alone is not used. (Gaps are in B3 and N1–N2.)
3. **eval.json aggregates.** Means and `loop_rate` come from items only (`schema.aggregate`). `summary.counts` gives
   per-benchmark item counts. `run_id`, `max_new_tokens`, `rounding_stats`, and `batch_size` are recorded. The
   resolved model commit is recorded: `HFRunner` now resolves via `resolve_base_revision` (`run.py`), and the smoke
   manifest shows `model_revision 2fc06364…`, which equals the zoo header. (Gaps are in B2–B3.)
4. **OOM.** The entry is marked `eval_status: "oom"`, a manifest is written, and the loop continues. The index is not
   corrupted, since each line is a complete JSON object. Idempotency is not met (B3).
5. **Smoke skipped by default.** `eligibility` defaults to `production` (`batch.py:196`), and the selection test covers it.
   (The `--ids` path still filters on eligibility, so an explicit smoke id is not run by default.)
6. **No data or secrets committed.** The branch diff is 9 files, none under `data/`. `data/` is ignored (`.gitignore:43`).
   The worktree's untracked `data` entry is a symlink to the repo's data directory and is not committed. A grep of the
   diff for token patterns (`hf_`, `gho_`, `glpat-`, `sk-`, `secret_wsp_`, `AKIA`) is clean.

## Throughput estimate (my own, from logs)

Calibration. Generation dominates, so cost ≈ decode steps × time per step.

- Smoke (`batch 4`, 32 tokens, GPU shared with the zoo build): 5 generative calls × 32 steps plus one 128-step KL
  continuation = 288 decode steps in 0.30 min → **≈ 62 ms/step** (upper bound; all time attributed to decode).
- `base-final-4096` (`batch 64`, 4096 tokens): 4+2+2+2+2+2 = 14 batches (matches the 14 `generated` lines in the log)
  ≈ 16,384 + 4×8,192 + 256 = 49,408 decode steps. Records were last written 00:31; the run was launched at 23:24 →
  **≤ 81 ms/step**, plausibly 60–80 ms.
- Step time is therefore roughly batch-independent for this model (launch-bound). Throughput comes from fewer
  batches, and larger batches help.

**D2 fast tier, per entry (batch 32, 1024 tokens):**

| Benchmark | Items | Batches | Steps per batch | Decode steps |
|---|---|---|---|---|
| gsm8k | 50 | 2 | 1024 | 2048 |
| math500 | 25 | 1 | 1024 | 1024 |
| ifeval | 25 | 1 | 1024 | 1024 |
| refusal_rate | 50 | 2 | 1024 | 2048 |
| over_refusal | 50 | 2 | 1024 | 2048 |
| kl_to_base (base continuation) | 50 | 2 | 128 | 256 |
| **Total** | | | | **≈ 8,450** |

- Time ≈ 8,450 × 60–80 ms ≈ 8.5–11.3 min decode, plus about 0.5–1 min for MMLU-Pro/HellaSwag scoring, WikiText,
  and KL context swaps. **≈ 9–12 min per entry.**
- Smoke cross-check: 8,450 / 288 ≈ 29 × 0.30 min ≈ 8.8 min. Consistent.
- **Zoo-wide:** 461 current production rows → ≈ 70–90 GPU-h. 700 entries → ≈ 105–140 GPU-h. Planning figure:
  **~120 GPU-h for 700 entries** (range 100–150). If step time at batch 32 is 2× the batch-4 value, the upper end
  is about 250 GPU-h.
- For comparison, the config as written (4096 tokens, batch 4): 218,752 decode steps per entry → 3.6–4.9 GPU-h per
  entry → ≈ 2,500–3,400 GPU-h for 700 entries. The ROADMAP's 5,617 overstates even this.
- Caveats: the step time is from shared-GPU runs. A clean measurement is one full D2 entry (≈ 10 min) once the GPU
  is released, plus the base run for the N8 gate. Multiple worker processes on one GPU could also cut wall-clock,
  since per-step time is launch-bound.

## Requested changes before re-review

1. B1: isolate `--limit`/non-default output; move the two smoke artifacts out of the zoo directories (author to inspect first).
2. B2: correct `git_sha` provenance and add source hash and dirty flag; re-run the n=4 smoke on the final commit.
3. B3: add `eligibility`/`family`; rename or fix `config_hash`; make the index an idempotent upsert or rebuild it from `eval.json`.
4. B4: set `configs/eval_fast.yaml` to `max_new_tokens: 1024` and `batch_size: 32`; remove the dead keys.
5. B5: replace the ROADMAP 5,617 h with the D2 estimate (or delete it), with the method above.
6. Re-run `pytest -q`; add tests for B1, B3 and N4.

Non-blocking items N1–N11 can be tracked as follow-ups.
