# Board A re-review: feat/p2-batch (head a63fdaf)

- Reviewer: reviewer-ml (Board A), read-only. No code edited, no GPU job run (zoo build holds the GPU).
- Prior review: `reviews/P2/feat-p2-batch-boardA-2026-11-04.md` (REVISE, B1–B5).
- Scope: `git diff 9143644...a63fdaf` (10 files; code in `src/wsp/eval/batch.py`, tests in `tests/test_eval_batch.py`,
  `configs/eval_fast.yaml`, `configs/eval_subset_d3.yaml`, `docs/ROADMAP.md`, `board/P2.md`).
- Ran:
  - `source scripts/env.sh && pytest -q` in the worktree: **164 passed, 8 deselected** (14.7 s), exit 0.
  - `provenance()` from the worktree: `git_sha a63fdaf6…` (equals HEAD), `REPO` = worktree, `wsp.__file__` = worktree `src/wsp`.
  - D3 subset resolved with the branch's own `subset_ids` against `data/zoo/qwen3.5-0.8b/manifest.jsonl`
    (snapshot mtime 03:33:38 UTC; the zoo build is still writing).
  - On-disk inspection of the two zoo entries, `data/evals/qwen3.5-0.8b/` and `data/runs/p2-batch/legacy-smoke/`.

## Verdict

**REVISE.** Most of the prior blockers are now closed in code and covered by tests. Three items remain blocking: a B1
residue on disk (the legacy base smoke and three smoke rows still sit in production paths), a B2 dirty-flag defect plus
an unrun acceptance smoke, and B5's D3 planning numbers, which the D3 subset does not reproduce. All three are small.
None needs a redesign.

## Status of prior blockers

| # | Status | Basis |
|---|---|---|
| B1 | **Open (residue)** | Non-production runs no longer write into zoo entries (closed in code). The two zoo entries are clean. The legacy base smoke and the index rows still sit in production paths (see R1). |
| B2 | **Open (residue)** | Sha is correct and the source hash covers `src/wsp`. The dirty flag is always `True` on this worktree (R2a), and the required n=4 smoke on the final commit has not been run (R2b). |
| B3 | **Closed** | Manifest has `eligibility`, `family`, `parent_id`, `step_semantics`, `effective_config`, `config_hash` (over the effective config). Index upsert is keyed on `(edit_id, config_hash)` under `flock`. Test repairs the index after deletion, which simulates a kill after `eval.json` and before the index append. |
| B4 | **Closed** | `configs/eval_fast.yaml:2` is `1024`, `:8` is `32`. Dead keys removed. Subset sizes match D2. `test_d2_config` checks the two generation values. |
| B5 | **Open (residue)** | The 5,617 h figure is gone and the D2 figure (≈120 GPU-h / 700) is stated. The D3 claim ("≈300 entries, ≈50 h") does not match the subset that resolves (227 entries, §D3 below). See R3. |

## Blocking (remaining)

### R1. Legacy smoke artefacts are still in production paths (B1 residue)
- `data/evals/qwen3.5-0.8b/base/eval.json` is still on disk. It is the legacy smoke base (`limit: 4`,
  `max_new_tokens: 32`, `batch_size: 4`, `signature: null`, `run_id 20261104-032002-1c2cc588`). It was not moved, unlike
  the two entry directories.
- `data/evals/qwen3.5-0.8b/index.jsonl` (3 lines, rewritten 03:30) still lists `base`, `bccb9ab10b5661e0` and
  `adf416337f5a2277` with `limit: 4` and `max_new_tokens: 32`. Their `path` fields now point to
  `data/runs/p2-batch/legacy-smoke/…`, so they point at moved files.
- `src/wsp/eval/batch.py:218` (base upsert) and `:127` (`select_entries` upsert) only ever add or replace rows. They never
  remove the legacy rows, so the three smoke rows persist in the production index.
- Failure scenario: a P3 loader that reads `data/evals/<zoo>/index.jsonl` or `data/evals/<zoo>/base/eval.json` ingests
  n=4, 32-token numbers as production baselines. The row is not keyed to a signature, so nothing rejects it.
- Fix: move `data/evals/qwen3.5-0.8b/base/` to `data/runs/p2-batch/legacy-smoke/base/`. Rebuild
  `data/evals/qwen3.5-0.8b/index.jsonl` from the production `eval.json` files only (or drop the 3 legacy lines). Confirm
  `index.jsonl` contains only rows whose `signature` matches the current run. Add a test: a non-production row must not be
  written to the production index.

### R2. Dirty flag and acceptance smoke (B2 residue)
- **R2a (defect):** `src/wsp/eval/batch.py:25–36` (`provenance`) computes `dirty = bool(git status --porcelain)` over the
  whole worktree. The worktree has an untracked `data` symlink (`?? data`, created 03:17), so `dirty` is `True` for every
  run from this worktree. Verified: `provenance()` returns `dirty: True` on the committed HEAD. The flag cannot tell a clean
  run from a dirty one. Fix: `git status --porcelain -- src configs tests`. I checked that this pathspec returns empty on
  this worktree. Add a test that `dirty` is `False` when only an untracked file outside `src/configs/tests` exists.
- **R2b (acceptance not met):** my prior review asked for the n=4 smoke to be re-run on the final commit and for the sha
  to be confirmed. That has not happened, because the GPU is held (the author logged `p1-zoo-full` as still listed). This
  is a GPU gate, not a code change. It must run on the final commit before the zoo-wide run. I did not run it.

### R3. D3 subset and ROADMAP planning figure (B5 residue)
- `docs/ROADMAP.md:46–47` says "≈ 50 h for the D3 stratified subset", which implies ≈300 entries.
- `configs/eval_subset_d3.yaml` resolves to **227** entries on the current manifest. Table below. `chain` resolves to 0
  because all 20 chain rows lack `eligibility`. `task_vector` resolves to 7 of 30 because the pool is 7.
- At the stated 9–12 min/entry, 227 entries ≈ 34–45 GPU-h, not ≈50 h. The ROADMAP must state the resolved count and the
  shortfall.
- Decision needed (not mine to make): are the 20 legacy `chain` rows production? If yes, add `eligibility` to them. If no,
  accept the 227-entry subset and say so in the ROADMAP and in D3.

**D3 subset as resolved (manifest snapshot 03:33:38, 507 entries: 477 production, 30 missing eligibility, 8 headers):**

| family | D3 quota | production rows in manifest | selected |
|---|---|---|---|
| random_dir | 60 | 150 | 60 |
| abliteration | 100 | 200 | 100 |
| lora_sft | 60 | 120 | 60 |
| task_vector | 30 | 7 | 7 |
| chain | 50 | 0 (20 rows missing `eligibility`) | 0 |
| **total** | 300 | 477 | **227** |

The `ROADMAP` "700 entries" figure is also not the current production count (477). That is acceptable as a planning
assumption if labelled as one.

## Non-blocking (fix before the subset run or the zoo-wide run)

- **N-a. The D3 subset is not reproducible under zoo growth.** `subset_ids` (`batch.py:63–73`) draws from the live manifest.
  Adding a row to any family changes the random stream and therefore the members of every family. Persist the resolved id
  list in the run output (`data/evals/<zoo>/d3_subset.json`) before the subset run.
- **N-b. Manifest and eval.json disagree on status.** `write_results` writes `manifest.json` (`run.py:263`, called from
  `batch.py:281`) with `eval_status: "complete"`. The `incomplete` override (`batch.py:289`) is applied only to `eval.json`.
  Set the status before `write_results`, or make `write_results` take it.
- **N-c. Source hash in the signature forces full re-runs.** Signature = config hash + hash of all `src/wsp/**/*.py`
  (`batch.py:25–36`). Any source edit, including logging, invalidates every completed entry. At ≈120 GPU-h for 700, that is a
  real cost. Keep it deliberately and say so in the ROADMAP, or hash only the scoring modules (`eval/`, `schema.py`).
- **N-d. N4 still open.** `test_restore_exact_base_between_edits` is still fp32 with no `base_context`.
  `test_kl_uses_base_context_and_restores_bf16` uses a bf16 parameter but not `apply_delta`. Add one bf16 test that goes
  through the real `apply_delta`.
- **N-e. Two OOM and override paths are untested end-to-end.** (i) All three attempts fail, so `eval_status: "oom"` is
  written and the entry must be re-selected on the next run. (ii) `--max-new-tokens` or `--batch-size` alone must route to
  `data/runs/p2-batch/<run>/` (`batch.py:186–193`). Only the `--limit` path is tested.
- **N-f. KL base_context cost.** `refusal.py:70–95` calls `base_context()` once per KL prompt (n=50). Each call copies every
  edited tensor to CPU and back. This is still unmeasured (N5). Measure it on the first full D2 entry.
- **N-g. `--ids` combined with `--subset` drops ids silently** (`batch.py:196`). Log the dropped count.
- **N-h. `upsert_index` rewrites the whole file per entry** (`batch.py:40–49`, called from `select_entries`). That is O(N²)
  I/O at startup. Acceptable at 700, but a single bulk rebuild is cheaper.
- Still open from the prior review and tracked in the ROADMAP: N7 (WikiText 8 vs 64 windows, `eval_fast.yaml:27`), N8 (base
  reproduction gate through this runner, not run), N10 (allocator growth; `empty_cache` now added per entry, `batch.py:293`).

## Verified (correct)

1. **Zoo isolation (B1, code path).** Production is a conjunction (`batch.py:186–191`): no `--limit`, `--max-new-tokens`, or
   `--batch-size`; eligibility `production`; config is `configs/eval_fast.yaml`. Anything else writes to
   `data/runs/p2-batch/<run_id>`, and `select_entries` reads and writes only under that run. The on-disk zoo entries
   `bccb9ab10b5661e0` and `adf416337f5a2277` now hold only `delta.safetensors` and `edit.json`. Their smoke artefacts are in
   `data/runs/p2-batch/legacy-smoke/<id>/` (eval, manifest, records, summary), not deleted.
2. **Provenance sha (B2).** `REPO = Path(wsp.__file__).resolve().parents[2]` (`batch.py:22`) resolves to the worktree. The
   recorded sha `a63fdaf6…` equals HEAD. `test_package_provenance_ignores_cwd` checks cwd independence. (Dirty flag: R2a.)
3. **eval.json fields and effective-config hash (B3).** `config_hash` is hashed over `cfg` after the overrides, including
   generation, limit, and benchmark `n` (`batch.py:172–185`). `eligibility`, `family`, `parent_id`, `step_semantics`,
   `effective_config` and the source hash are in the manifest.
4. **Index idempotency under interruption (B3).** `upsert_index` takes `flock` on `index.lock`, reads, replaces by key, and
   atomically `replace`s. `test_batch_loads_once_restores_and_resumes` deletes `index.jsonl` after a complete run, reruns,
   and asserts it is repaired. That simulates a kill after `eval.json` and before the index append.
5. **Reduced OOM batch does not persist.** `execute` writes `runner.generation["batch_size"]` (`batch.py:249–250`), but
   each entry starts from the configured `generation["batch_size"]` (`batch.py:253`), not the mutated dict. The test asserts
   `("a", 4)`, `("a", 2)`, then `("b", 4)`.
6. **KL path has a unit test.** `test_kl_uses_base_context_and_restores_bf16` asserts that the base continuation is generated
   under base weights (`seen == [0]`), that KL is > 0, that the edited weight is restored to 1 afterwards, and that
   `generation["max_new_tokens"]` is restored.
7. **Counts match n for every configured benchmark.** `write_results` counts records per benchmark (`run.py:259–261`), and
   each benchmark emits one record per item (refusal, KL and wikitext included). `complete()` (`batch.py:52–60`) therefore
   matches the config.
8. **Restore and skip logic** unchanged and correct. Base is re-applied in a `finally`. Capture happens after restore.
9. **Config (B4).** `eval_fast.yaml`: `max_new_tokens 1024`, `batch_size 32`, sizes gsm8k 50 / math500 25 / ifeval 25 /
   mmlu_pro 100 / hellaswag 100 / refusal 50 / over-refusal 50 / KL 50, matching D2.

## Requested changes before the next re-review

1. R1: move `data/evals/qwen3.5-0.8b/base/` to legacy-smoke. Rebuild `data/evals/qwen3.5-0.8b/index.jsonl` from production
   `eval.json` only. Add a test that a non-production row never reaches the production index.
2. R2a: change the dirty pathspec to `-- src configs tests` and add the test described above.
3. R2b: run the n=4 smoke on the final commit once the GPU is free. Confirm `git_sha` equals that commit and `dirty` is `False`.
4. R3: correct `docs/ROADMAP.md:46–47` to the resolved D3 count (227) and its shortfall. Decide the `chain` eligibility question,
   and record the decision in `docs/DECISIONS.md`.
5. Non-blocking items N-a to N-h before the subset run or the zoo-wide run, as listed above.
