# Board A: H5 editor harness review (feat/h5-editor @ af22aee)

Date: 2026-11-04. Reviewer: Board A (strict ML correctness). Branch reviewed read-only in its worktree; no tracked file changed.
Scope: `git diff main...feat/h5-editor` (16 files), `src/wsp/editor/*`, `configs/editor_*.yaml`, `tests/editor/*`, against
docs/H5_DESIGN.md, docs/H5_PROTOCOL.md, docs/DECISIONS.md D1, and the Board A pre-registration (threat register and DoD).

**Verdict: REVISE (blocking B1–B6).** The branch may not merge as-is. Merging only after B1–B6 are fixed and re-reviewed,
and even then no production episode may run until the production items in N2, N3, N4 and N11 are closed.

## Method and evidence

- `pytest -q tests/editor` (worktree): 13 passed.
- Full CPU suite, `pytest -q -m "not gpu"`, on the branch: 416 passed, 8 deselected, exit 0, 47.5 s (main baseline per session note: 403; +13 editor tests).
- 35 adversarial probes, written in a scratch directory outside the repo and not committed. Each one records what the branch does; the findings below cite the probe results.
- Measurements of IPC payloads and episode-log size on a 151,936-vocabulary stand-in model (random logits) and a Qwen-sized weight tensor.
- No GPU run, no real checkpoint, no network provider. Qwen/Gemma tensor names were not checked against real state dicts.

## Blocking (must be fixed and re-reviewed)

**B1. Probe output path bypasses the layer/position restriction for real models.**
`src/wsp/editor/api.py:74-75` returns the model output unchanged when `isinstance(output, dict)`. transformers `ModelOutput`
subclasses `dict`, so real models take this path: logits stay as tensors, and any hidden states requested come back for every
layer and every position, not the allowlisted layers and last position (`api.py:78-80`). Probe result: a dict-like output with
hidden states returned 3 layers and the full 2-token sequence. Over IPC the response fails to serialize (`valid=False`), so no
data leaks today, but the real-model probe path is non-functional, and the obvious fix (`.tolist()` on the dict) would expose
unrestricted hidden states. The tests use `SimpleNamespace` (`tests/editor/test_loop.py:15-18`), so this path is never exercised.
Required: normalize the output explicitly (`logits`; `hidden_states[i][:, -1]` only for allowlisted `i`); add a ModelOutput-like test;
move output shape (layers, positions) into the config.

**B2. Unbounded IPC payloads, retained in the append-only episode log.**
`api.py:76` returns full-vocabulary logits for every position of every sequence; `api.py:39-43` (`weights.read`) and
`api.py:116-130` (`dispatch`) return full tensors with no size cap; `api.py:47-48` has no sequence-length cap; `sandbox.py:99-104` stores every response in
the transcript; `driver.py:56-62` writes the transcript into `episode.jsonl`. Measured: a 16-token probe on a 151,936-vocabulary
model produced 50.2 MB of JSON (20.6 bytes/float, 0.06 s). Extrapolated, a 512-token sequence is about 1.6 GB of JSON held by the
supervisor, which also holds the model. A single 4-token probe grew `episode.jsonl` to 12.5 MB. One `weights.read` of a Qwen-sized
down_proj (1024×3584) produced 43.8 MB of JSON. Failure scenario: a 50-call episode with 64-sequence batches writes gigabytes of
logits into the private log and can exhaust memory in the process that holds the model; the log stops being usable for replay.
Required: return only what the design allows (last position, or requested token IDs or top-k); cap response bytes and sequence length
in the API config; log hashes and summaries of responses, not raw arrays.

**B3. Filesystem metadata oracle in the sandbox.**
`src/wsp/editor/sandbox.py:28` allows `newfstatat`. Probe results: `os.stat` on a missing path raises `FileNotFoundError`
(existence oracle); `os.stat` on the repo data root succeeds (existence, size and mtime readable by the editor). The following are
correctly denied (EPERM): open, listdir, getcwd, openat via libc, fork, socket, setrlimit. Threat register row 1 requires denial of
filesystem calls, and H5_DESIGN requires no editor path to the repo/data root.
Required: drop `newfstatat` (keep `fstat`), and restrict `write` to fd 1 (see N10).

**B4. Fail-open editable-tensor default.**
`src/wsp/editor/driver.py:37`: `config.get("editable_tensors", list(weights))`. If the key is absent, every tensor in the supplied
state dict becomes readable and writable. Probe: with no `editable_tensors`, an edit to `embed` was accepted and registered
(events: prompt, completion, sandbox, registered). H5_DESIGN says "default to selected MLP output weights ... do not silently
broaden access"; the pre-registration threat row "Alter tokenizer, model code, config ..." applies.
Required: require the key; reject any delta name outside the configured list; check names against the loaded model.

**B5. The "replay" test does not replay a recorded proposal.**
`tests/editor/test_loop.py:53-86` runs a hard-coded `ScriptedEditor` string twice and compares IDs. It never reads a completion or
transcript from the episode log. H5_DESIGN's DoD requires that rerunning a recorded proposal reproduce the submitted delta, and the
session note says "replay identity" is tested. The mechanism itself works: a probe that read the logged completion code from
`episode.jsonl` and reran it reproduced the same delta and transcript. The test is missing, not the replay.
Required: add a test that replays from the log; correct `sessions/2026-11-04-h5-editor.md`.

**B6. The "sealed absent" test cannot fail.**
`tests/editor/test_loop.py:103-125`: `sealed_reference` is a config key that no code reads. The pool is passed explicitly and contains
only a train item, and the assertion searches the episode log for a string that was never written there. H5_DESIGN's DoD requires
"sealed-item IDs/content absent from editor IPC and logs" as a CPU test.
Required: a test in which sealed items exist in a store the driver must not touch, asserting absence from the IPC transcript and from
`episode.jsonl`; otherwise remove the claim.

## Non-blocking (fix before the first production episode unless noted)

- **N1. Config validation is late.** `zoo.py:18` rejects `base_revision: main` only inside `register()`, after the editor has spent
  budget and tokens (probe events: prompt, completion, sandbox, failed). The YAML uses `budgets` (a list) while code reads `budget`
  (`driver.py:41`), and there is no loader. Validate at load, before the first provider call.
- **N2. `production_enabled: false` is inert.** It appears in `configs/editor_qwen3.5-0.8b.yaml:23` and
  `configs/editor_gemma-3-270m.yaml:22` and is never read. The real fence is in code and holds: the provider registry contains only
  `scripted` (`providers.py:31-37`), `register()` hard-codes `eligibility="smoke"` (`zoo.py:39-40`), there is no CLI entry point, and
  eval selects production only (`src/wsp/eval/batch.py:116,157`). Enforce the flag or delete it, so it does not read as a control.
- **N3. Development pool is not pinned (production blocker).** `driver.py:18-28` checks only the schema. `version_required` and
  `sources` are unused. There are no item IDs or hashes, no disjointness check against sealed IDs, and no sealed reference. The
  pre-registration requires exact IDs or an immutable split hash before any episode.
- **N4. No episode-level meter; wall time not enforced during model calls (production blocker).** `run_episode` creates a fresh
  budget for each proposal (`driver.py:33-43`). Tokens are recorded but never capped. The wall timeout is checked only between
  messages (`sandbox.py:82-84`) and cannot preempt a supervisor-side forward (`api.py:68-70`). The session note acknowledges this.
- **N5. `delta.add` broadcasts.** `api.py:99-100`: a (1,2) tensor added to a (2,2) delta is accepted and applied to both rows (probe
  confirmed). H5_DESIGN requires exact name/shape match for `add`. Check the input shape before adding.
- **N6. Delta bounds are not configurable.** `driver.py:38-43` does not pass `max_element`/`max_norm`, so the defaults of 1.0 and 10.0
  (`api.py:11`) apply silently.
- **N7. `build()` charges a call that never launches.** `zoo.py:83` passes `model=None` to `run_episode`. `probe.forward` then raises
  `TypeError` after being charged (`api.py:65`, `api.py:69-70`; probe confirmed). Pre-launch failures should be free.
- **N8. Provenance gaps against H5_DESIGN.** No completion hash (only `code_hash`), no base checkpoint hash (only the revision), no
  timestamps in the descriptor (only in the log), API and sandbox versions are hard-coded (`driver.py:74-75`), and every proposal in an
  episode uses the same seed. `register()` drops the provenance of a duplicate entry (`zoo.py:50`); the log has the IDs, the zoo does not.
- **N9. D1 base cast.** `zoo.py:25` casts the base to bf16 before `effective_delta`. That matches `apply_delta` only for bf16 checkpoints.
  Assert the base dtype. D1 itself is verified (see below).
- **N10. `write` is allowed on any fd** (`sandbox.py:28`). Probe: writing to fd 1 produced a request in the transcript. There is no
  escalation, because requests still pass through `dispatch`, but the channel should be fd 1 only.
- **N11. Isolation is seccomp and rlimits under the same OS user** (`sandbox.py:61-72`). There is no separate principal, mount namespace,
  or network namespace. H5_DESIGN and the pre-registration call for a separate principal or container (production blocker).
- **N12. Unverified model names.** `editable_tensors` in both configs is not checked against real state dicts (none downloaded).
  `tests/editor/test_loop.py:89-100` checks only that the list is non-empty.
- **N13. Log integrity.** `episode.py:21-26` is append-only by convention (`O_APPEND`), with no hash chain. `driver.py:85-91` records only the
  exception type, losing the message.
- **N14. Session note overclaims.** `sessions/2026-11-04-h5-editor.md` says replay identity and sealed absence are tested (see B5, B6).

## Verified as correct

- **Meter** (`api.py:45-83`): a batch of B is charged B. A batch straddling the limit is rejected before the model runs, with nothing charged
  (`api.py:62-64`). Pre-launch validation errors (float IDs, wrong rank, negative IDs, mask shape, hidden states not enabled) are free and do
  not execute the model. A model exception after launch is charged. Logits and hidden states from one call are one charge.
- **Isolation at the syscall level** (`sandbox.py:15-49`): the worker has no `torch`, the environment is empty, `socket`, `fork`, `openat`,
  `setrlimit`, `getcwd` and `listdir` are denied, and `inference_mode` is used (`api.py:68`). No backward or gradient operation is exposed.
- **D1**: `register()` uses `effective_delta` (`zoo.py:10,25`), and `preview()` reuses it (`api.py:111`). On a bf16 base, the stored Δ_eff
  reproduces `eval.run.apply_delta` exactly (probe). The helper is reused, not reimplemented.
- **`llm_edit` registration**: added to `EDIT_FAMILIES` (`schema.py:24-27`), `FAMILY_ORDER`, and the plot palette. It is not in the default
  zoo configuration. `parent_id` is None (asserted in tests). The full CPU suite passes.
- **Fencing of the open items (item 5)**: the separate model worker, probe timeouts, and provider/split qualification are fenced in code (N2,
  N3, N4, N11). No code path can produce a production-eligible llm_edit entry, and only the scripted provider exists.

## Test counts observed

- `tests/editor`: 13 passed.
- Full CPU suite (`-m "not gpu"`): 416 passed, 8 deselected, 0 failed, exit 0, 47.5 s.
- Adversarial probes (scratch, not committed): 35 passed. Several assert the defects above, so passing means the behavior was reproduced.

## Summary

1. Verdict: REVISE (blocking B1–B6).
2. Blocking items: 6 (B1 probe output, B2 unbounded payloads, B3 stat oracle, B4 fail-open default, B5 replay test, B6 sealed test).
3. Worst finding: `src/wsp/editor/api.py:76`, full-vocabulary logits for every position, unbounded and retained in the episode log (about 50 MB per 16-token probe; 12.5 MB of log per 4-token probe).
4. Tests observed: 13 editor tests passed; full CPU suite 416 passed, 8 deselected, 0 failed (47.5 s); 35 adversarial probes passed.
5. Merge as-is: no. Fix and re-review B1–B6; production stays blocked by N2, N3, N4 and N11.
