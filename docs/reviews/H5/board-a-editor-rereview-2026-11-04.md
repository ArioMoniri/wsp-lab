# Board A: H5 editor harness re-review (feat/h5-editor @ 266af77)

Date: 2026-11-04. Reviewer: Board A (strict ML correctness). Re-review of the previous Board A review
(`docs/reviews/H5/board-a-editor-2026-11-04.md`, at af22aee). The branch was read in its worktree; no tracked
file in the worktree was changed. Adversarial probes were kept in a scratch directory outside the repo and are
not committed.

**Verdict: REVISE.** B1–B5 are closed. B6 is closed for the requirement as written, with a caveat. N2 is closed.
Three new blocking defects were found (R1, R2, R3). N3, N4 and N11 stay open. Merge: no. First pilot: not yet.
Section 6 lists what must be true first.

## 1. Method and evidence

- Read: AGENTS.md, `.claude/agents/reviewer-ml.md`, `docs/REVIEW_PROTOCOL.md`, `docs/H5_DESIGN.md` (including the
  implementation-boundary section added on this branch), `sessions/2026-11-04-h5-editor.md`, the previous Board A
  editor review, and the Board A pre-registration review (evaluator isolation and threat register).
- Editor tests (`. scripts/env.sh; pytest -q tests/editor`): 37 passed, 1 skipped (the root-only UID test, skipped as
  `ubuntu`).
- Full CPU suite (`pytest -q -m "not gpu"`, run in tmux): 440 passed, 1 skipped, 8 deselected, exit 0, 27.98 s.
  The session note says 439; the observed count is 440.
- Adversarial probes: about 20 probes, plus debugging scripts. Probes ran against the real Qwen3.5-0.8B architecture,
  built from the cached `config.json` with random initialisation (no checkpoint weights loaded), and against
  `google/gemma-3-270m-it` for tensor names. Tensor names were checked by parsing the cached safetensors headers
  (JSON only, no model code executed). Pinned revisions available on this host:
  Qwen3.5-0.8B `2fc06364715b967f1860aea9cf38778875588b17`, gemma-3-270m-it `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`.
- Harness notes: the first probe run gave invalid results for two reasons, both fixed before the reported runs. The
  worker's stderr is seccomp-denied, so an editor exception was lost and looked like exit 120. Real-model calls also
  hit the default 5 s `call_seconds`. Only results from the corrected harness are reported here.
- Real-model CPU forwards need the reference kernels. On this host `causal_conv1d` and `flash-linear-attention` are
  installed, and with them the Qwen3.5 CPU forward fails (CUDA-only kernels). The probe process blocked those two
  imports so transformers used its PyTorch fallback. The branch itself was not changed.

## 2. Status of previous items

| Item | Status | Evidence |
|---|---|---|
| B1 probe output path | **Closed** | `api.py:56-93`: dict-like outputs are read explicitly (`output[key]`); layers and positions are checked before launch. Real-architecture probe: logits only at the final position, shape `(1,1,248320)`; hidden entries contain only `norm`, `mean`, `projections`; layer 5 (not allowlisted) refused before launch. Caveat: default `max_vocab=200000` (`api.py:112`) refuses the real Qwen vocabulary of 248320. That is fail-closed, but the production config must set `probe.max_vocab` (Section 6). |
| B2 unbounded payloads/log | **Closed for the log**; see R2, R5 | Real probe: one final-position response is 3.12 MB of JSON. Three real probes produced a 3,150-byte episode log with a longest line of 2,365 bytes. Arrays are logged as sha256 plus shape (`sandbox.py:69-81`, `episode.py:14-30`). Per-call response is capped (`api.py:194-205`). R2 and R5 are the remaining gaps. |
| B3 stat oracle, fd-1 writes | **Closed** | Seccomp allowlist at `sandbox.py:36-38`; `write` restricted to the IPC fd at `sandbox.py:47-50`. Probe: `stat`/`lstat` on existing and missing paths both return EPERM (errno 1), so no existence oracle. Also EPERM: `open`, `os.open`, `listdir`, `scandir`, `getcwd`, `readlink`, `statvfs`, `fork`, `kill`, `uname`, `pipe`, `dup`, `sleep`, writes to fd 1 and fd 2, and `socket`. Raw libc `newfstatat`, `statx`, `openat`, `getdents64` return -1. `os.path.exists` returns False for both existing and missing paths. |
| B4 editable-tensor default | **Closed** | `validate_config` requires an explicit non-empty allowlist and checks every name against the state dict (`driver.py:46-55`). Probe: a config without the key is rejected; `set` and `add` on a non-allowlisted name are refused; `read` of it is refused; `list_tensors` returns `['w']` only. |
| B5 replay from log | **Closed** | `test_logged_replay_delta_meter_and_array_hashes` (`test_review.py:140`) reruns the logged completion code. Real-model probe: same entry ID, same delta-file SHA-256, logged code equals the source, and a tampered logged code gives a different entry. |
| B6 sealed-absent test | **Closed for the requirement as written; caveat** | `test_sealed_store_unreachable_and_detector_catches_injection` passes and checks absence from `episode.jsonl`, which includes the transcript. Probe: an injected sealed content/ID string in an editor-controlled field is flagged. An injected sealed **path** is not flagged, because the markers in `test_review.py:185` and `:204-206` cover only content and IDs. Denied file attempts leave no record (R4). The stale test `test_loop.py:112-125` still claims a sealed check using `sealed_reference`, a config key nothing reads. Remove it. |
| N1 late config validation | **Closed** | `run_episode` validates before any provider call (`driver.py:80`); `load_config` exists. The YAML templates still use a `budgets` list and `base_revision: main`. They fail closed, but they still look runnable (`test_loop.py:108` asserts the old list form). |
| N2 `production_enabled` inert | **Closed** | Read by `validate_config` and `run_episode` (`driver.py:65-66`, `:83-84`); non-scripted providers refuse without it (`providers.py:35-38`). `test_production_provider_gate`. |
| N3 development pool not pinned | **Open (protocol item)** | Schema, train-only sources and an immutable pool hash are enforced (`driver.py:20-30`, `:61-64`). Not enforced: disjointness from sealed item IDs (no sealed reference exists in code, so this must be a documented protocol statement), and `version_required` and `sources` are unused. |
| N4 episode meter, wall time | **Open** (R1, R3) | Episode meter across proposals works (`test_episode_meter_across_proposals`). Each forked probe is killed at its deadline (`api.py:246-256`). But a worker that stops reading stalls the supervisor past the wall limit (R1), and the second and later forked probes hang on real models (R3). |
| N5 `delta.add` broadcast | **Closed** | Probe: a (1,2) tensor added to a (2,2) delta is rejected (`api.py:285-289`). |
| N6 delta bounds from config | **Closed, with a low caveat** | Bounds are passed from config (`driver.py:97-98`), but defaults of 1.0 and 10.0 still apply silently when the keys are absent. |
| N7 unlaunched calls charged | **Closed** | Unlaunched calls are free; a launch failure is free (`api.py:240-242`); a launched failure is charged. Covered by `test_episode_deadline_kills_probe_and_launch_failure_free`. |
| N8 provenance gaps | **Open** (non-blocking for a pilot) | Probe on a real entry: no base-checkpoint hash in the descriptor; no start or end timestamps (only `wall_seconds`); `api_version` and `sandbox_version` are literals (`driver.py:135-136`); `meter_totals` holds the per-proposal count `api.used` (`driver.py:134`), not the episode meter; a duplicate entry keeps the first provenance (`zoo.py:50`). |
| N9 D1 base dtype | **Closed** | `test_checkpoint_dtype_effective_delta` covers fp32, fp16 and bf16; `register` uses the checkpoint dtype (`zoo.py:25`). |
| N10 `write` on any fd | **Closed** | Probe: write to fd 1 returns EPERM; `write` is allowed only on the IPC fd (`sandbox.py:47-50`). |
| N11 isolation principal | **Open** (production blocker) | Section 4. |
| N12 tensor names | **Closed (names); pinning remains** | Cached Qwen3.5-0.8B has `model.language_model.layers.15.mlp.down_proj.weight` and `...layers.19...` (both `[1024, 3584]`, BF16). Cached gemma-3-270m-it has `model.layers.15.mlp.down_proj.weight` (`[640, 2048]`, BF16; 18 layers). Both configs still say `base_revision: main` and `development_pool_hash: null`, so they cannot load until the SHAs and the pool hash are filled in. |
| N13 log integrity | **Open** (non-blocking for a pilot) | `Episode.append` is `O_APPEND` plus `fsync` with no chain (`episode.py:40-47`). A process with the same uid can rewrite the file. |
| N14 session note | **Mostly closed** | The note now describes the B5 and B6 tests accurately. It still reports 439 passed (observed 440), and the stale test in `test_loop.py` is not mentioned. |

## 3. New blocking findings

**R1. A worker that stops reading stalls the supervisor and bypasses the wall limit.**
`sandbox.py:155-156` writes each response with `proc.stdin.write` and no timeout. The wall check at `sandbox.py:132-134`
runs only at the top of the loop, so it never runs while the write is blocked.
- Probe: the editor writes a `weights.read` request whose response is about 70 KB (larger than the 64 KB pipe buffer),
  then blocks on a lock without reading stdin. With `wall_seconds=2` the call never returns; the outer `timeout 25`
  killed it (exit 124).
- Controls: a small response followed by the same block returns `TimeoutError` at 2.0 s. A worker that blocks on
  `stdin.read` is stopped at the wall limit (3.0 s).
- Failure scenario: an LLM-written proposal (or a bug) that stops reading its IPC channel wedges the supervisor, which
  also holds the model, indefinitely. The proposal never reaches its time limit and the N4 claim does not hold.
- Required fix: write responses with a deadline (non-blocking writes with a selector, or a writer thread with
  `join(timeout)` followed by a kill). Add a CPU test: a non-reading worker with a 100 KB response must return within
  wall plus 1 s.

**R2. `weights.read` materialises the slice before the size cap.**
`api.py:319` calls `self.read(**args)`, which allocates `value[slice]`. The byte check is at `api.py:320`, after the
allocation. The slice is a JSON list of indices, so a single request can ask for many rows.
- Measured: a 100,000-index slice (a 300 KB request) on a 1024×3584 float32 tensor raised peak RSS by 2.7 GB (1.43 GB
  of rows, 0.5 s), and was then refused with "weight response byte cap".
- Computed: the 1 MiB IPC line limit (`sandbox.py:140`) allows about 557,000 indices of `0,`. Those materialise about
  8 GB before the cap rejects them.
- Failure scenario: `weights.read(name, slice=list(range(N)))`, which looks like ordinary editor code, can exhaust
  memory in the supervisor that holds the model.
- Required fix: compute the output size from the index count and row size before indexing, and refuse above
  `max_bytes`. Add a test that asserts no allocation for an oversized index list.

**R3. Forked probes hang after the parent does parallel torch work (real models).**
`api.py:221-225` forks a probe child with no control of torch threads. A fresh process ran one forked probe on the
real Qwen3.5 architecture (24 layers, bf16, CPU reference kernels) in 8.6 s. After the parent runs one 512×512 matmul
(a parallel region; `register()` also runs parallel elementwise work in the parent, but I did not test that path directly), the next two
forked probes each hit the 120 s hard timeout. With `torch.set_num_threads(1)` set before the second fork, the same
sequence finished in 0.6 s each. The test suite does not see this, because the DictModel and TinyCausal probes never
enter a parallel region in the child.
- Failure scenario: a multi-proposal real-model episode works for proposal 1, then every later probe times out. Each
  timed-out probe is charged to the budget. The failure is recorded as `failed` (so it is not silent), but the
  mechanism the design relies on does not work for real models.
- Required fix: pin torch to one thread in the supervisor before forking, or in `_probe_worker`. Then add a regression
  test with a fixture whose forward does a 512×512 matmul, running two forked probes around a parent-side parallel op.
  Note that the single-thread child was also faster in this test (0.6 s against about 8.5 s), which needs explaining
  before the setting is adopted.

## 4. New non-blocking findings (fix before any confirmatory run)

- **R4. Denied calls are not audited.** The editor catches the EPERM (or the exception) and continues. The B6 probe
  made three denied attempts (open, stat, listdir); the sandbox transcript shows only the completed `delta.set`. The
  pre-registration threat register requires "deny/audit filesystem and tool calls". Required: an audit path for denied
  calls, for example auditd rules on the worker's uid for the denied syscalls (exit `-EPERM`), or seccomp user
  notification. Verify that audit is enabled on the production host.
- **R5. Editor-controlled strings are logged without a cap.** `driver.py:107` logs the full completion code, and
  `sandbox.py:149` keeps `rationale` raw in the transcript. Probe: ten `delta.set` calls with 900 KB rationales produced
  a 9.0 MB `episode.jsonl`. Required: cap rationale (for example 4 KB) and code length; store a hash beyond the cap.
- **R6. A forged `done` line truncates the transcript.** `sandbox.py:146-148` ends the API loop on any request equal to
  the done line. Requests the worker sends after that are never answered or logged. Probe: a `delta.set` after a forged
  `done` was dropped; the proposal was `valid=True` with a one-line transcript. The delta did not change, because the
  request was never executed, but the registered proposal's log is incomplete. Required: after the real `done`, require
  EOF on stdout and treat any further bytes as an error.
- **R7. Test gaps.** (a) The suite never runs a real checkpoint or a real-sized model (this is why R2 and R3 pass CI).
  (b) The negative isolation tests assert only `not valid`. A seccomp denial and a crash look the same (the traceback
  cannot leave the worker), so they cannot tell the two apart. Add positive assertions that the editor sees `EPERM`.
  (c) `test_loop.py:112-125` is the old sealed test and asserts on a config key nothing reads; delete it.
  (d) `test_loop.py:108` asserts the `budgets` list form, which the loader rejects.
- **Smaller points.** `meter_totals` is misnamed (N8). Probe memory: the forked probe child has no memory limit, and
  the forward allocates full logits for every position (batch × sequence × vocabulary). For a 512-token Qwen sequence
  that is about 0.25 GB of bf16 logits per call. This is computed, not measured, and is bounded by `max_sequence`.

## 5. Isolation: what it guarantees and whether it meets the requirement

**Mechanism.** The editor runs in a fresh `python -I -S` process (`sandbox.py:113-121`) with a cleared environment,
`cwd=/`, and stdout and stderr to the null device. It is confined by a seccomp allowlist with default action
`ERRNO(EPERM)` (`sandbox.py:36-38`). It allows 23 unconditional syscalls: `read`, `close`, `fstat`, `lseek`, memory
management, `futex`, signal handling, `getrandom`, the uid and pid getters, `clock_gettime`, and `exit`. `write` is
allowed only on the IPC descriptor (`sandbox.py:47-50`). The worker also has rlimits: CPU 2 s, address space 256 MB,
`RLIMIT_FSIZE` 0, core 0. Its only network path is the monkeypatched `socket.socket`, which is also blocked by seccomp.
Probe B3 confirmed each denial at the kernel level, including raw libc syscalls.

**What it guarantees (verified by probes):**
- The editor cannot open, stat, list or read the filesystem, including existence probing through `stat`.
- It cannot execute programs, fork, create sockets, send signals, make pipes or duplicate descriptors.
- It cannot write anywhere except the IPC channel. Its only inherited readable descriptor is stdin, the supervisor's responses.
- Its IPC requests are validated against the typed API before anything runs.
- The probe child is forked by the supervisor, not by the editor, so the editor cannot reach the model.

**What it does not guarantee:**
- **Same OS user.** The default runs as the supervisor's uid (`ubuntu` on this host). Any seccomp bypass, kernel exploit, or other process with that uid gets that uid's file access. On this host `/home/ubuntu` is mode 750 and the credentials file is 600, but the seccomp barrier is the only control in the default configuration.
- **No namespaces.** There is no mount, network, PID or cgroup isolation. Wall-time enforcement is also weak (R1).
- **Optional setuid is not exercised here.** With `require_separate_principal: true` and a root supervisor, the worker drops to `sandbox.uid` before seccomp. The only test of that path is skipped as `ubuntu`, so I could not verify it. I did not run anything as root.
- **The shipped configs refuse to run without root.** Both configs set `require_separate_principal: true`. As a non-root supervisor, every proposal fails with `ValueError: separate principal unavailable` (verified). The configs therefore cannot run on this host as shipped.
- **The model worker is trusted.** It is a fork of the supervisor and shares its memory, including the base weights.
- **No audit of denials (R4), no log integrity (N13).**
- **GPU is unsupported.** Non-CPU model tensors are refused (`api.py:185-188`), so a GPU episode cannot run through this backend.

**Verdict on the requirement "no editor path to sealed items, credentials, or network":**
- **Smoke: met**, at the level of editor code, on the paths probed (Section 3, B3 and B6). The pre-registration's own
  caveat still applies: the isolation is same-uid and seccomp-only.
- **Production (confirmatory): not met.** The separate-principal path is unexercised, there are no namespaces, the
  wall limit can be bypassed (R1), denials are not audited (R4), and no sealed evaluator exists yet on which the
  principal boundary could be tested. Production needs a separate principal (or a container or namespace) with the
  sealed store unreadable by the editor uid, the root path tested on a root-run host, and R1–R4 closed.

## 6. What must be true before a first pilot episode

A pilot is one real-model (Qwen3.5-0.8B) episode with frozen, reviewed scripted editor code. The branch has no network
provider adapter. It runs on CPU, on the development pool only. It is labelled a harness pilot and cannot count
toward confirmation (pre-registration: "Editor-only harness pilots cannot count toward confirmation").

Required before the pilot:
1. **R1, R2 and R3 fixed**, with the regression tests above, and re-reviewed.
2. **A principal decision for the pilot.** Either a root-run supervisor with a dedicated `sandbox.uid` (the root path is not yet exercised), or `require_separate_principal: false` set explicitly and recorded in the manifest. The second is acceptable only while no sealed material is on this host. As of today none is present under the repo's `data/` or in the home directory.
3. **Config values** (the shipped templates fail on each of these):
   - `base_revision: 2fc06364715b967f1860aea9cf38778875588b17`
   - `probe.max_vocab: 248320`. The default of 200000 refuses the real vocabulary; this was verified.
   - `probe.call_seconds` of at least 60. A forward takes about 8.5 s on the reference kernels, and the default of 5 s times out (verified).
   - `episode_limits.wall_seconds` of at least 600. The default of 60 s is too short for a 10-call episode at that speed.
   - a single `budget` of 10, 25 or 50 (not the `budgets` list)
   - `editable_tensors` set to the two names in Section 2 (N12)
   - `development_pool_hash` from `digest` of the exact train-only pool
   - `production_enabled: false` (the pilot stays smoke)
4. **Environment.** `causal_conv1d` and `flash-linear-attention` must not be importable, or a qualified CPU fallback must be recorded. With them installed, the CPU forward fails (verified). Record torch and transformers versions in the manifest.
5. **Pool disjointness.** The pool's item IDs, and a statement that none overlaps the sealed items, go into the pre-registration (N3).
6. **Keep the private episode log** as the source of record for the pilot. Do not report the descriptor fields that are missing (N8).

**Answers on the known open items:**
- **N8 (provenance) does not block a pilot.** The episode log has the completion, the transcript hash, the meter and the budget. The descriptor gaps (no base-checkpoint hash) must be closed before any entry is used in analysis.
- **N12 (tensor names) no longer blocks.** The names exist in the cached checkpoints at the pinned revisions. The production config must carry them and the SHA, and `validate_config` fails closed at load if a name is missing.
- **N13 (hash chain) does not block a pilot.** A process with the same uid can rewrite the log regardless, so a chain adds little until a separate principal exists. It is required before any result is reported.

**Production or confirmatory episodes** additionally need N11 closed (Section 5), R4 and R5 closed, the sealed evaluator in a separate principal or container, and an LLM provider adapter reviewed on its own.

## 7. Test counts observed

- `tests/editor`: 37 passed, 1 skipped (root-only UID test), 3.0 s.
- Full CPU suite (`-m "not gpu"`): 440 passed, 1 skipped, 8 deselected, exit 0, 27.98 s.
- Adversarial probes: about 20 in scratch, plus debugging scripts. Results cited above come from the corrected harness.
  Reproducing them requires the cached Qwen3.5-0.8B and gemma-3-270m-it configs and safetensors headers on this host.

## 8. Summary

1. Verdict: **REVISE**. B1–B5 closed; B6 closed for its stated requirement, with a path-detection caveat; N2 closed; new blocking R1–R3.
2. Still open: R1 (supervisor stall and wall-limit bypass), R2 (`weights.read` allocates before the cap), R3 (forked probes hang after parent parallel torch work); non-blocking R4–R7; N3 (disjointness), N4 (through R1), N8, N11 (production), N13.
3. Merge to main: **no**, until R1–R3 are fixed with tests and re-reviewed.
4. First pilot blocked by: R1–R3; the principal decision (the shipped configs refuse to run as non-root); the config values in Section 6 (`max_vocab`, `call_seconds`, wall, revision, one budget, pool hash); the CPU-kernel environment; and the pool-disjointness statement.
5. Tests observed: editor 37 passed, 1 skipped; full CPU 440 passed, 1 skipped, 8 deselected, exit 0; about 20 adversarial probes, scratch only, not committed.
