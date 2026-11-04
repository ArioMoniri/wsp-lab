# H5 `llm_edit` harness design

Status: Board C design for the Qwen3.5-0.8B Experiment 8 reproduction. This is a design, not an implementation or a result. Every runtime, token, and GPU figure below that is not directly identified as a repo measurement is **to be measured**.

## Decision

Implement a **code-executing Python REPL inside a disposable subprocess/container**. The editor writes Python against an intentionally small API; the orchestrator validates and persists the resulting tensor delta. The model/evaluation process owns the weights and the sealed evaluation data. The editor process receives only an allowlisted CPU tensor snapshot and a probe API backed by a separate evaluator process. It has no filesystem, network, shell, or direct access to eval storage. Its Python is still arbitrary code, so OS-level isolation and resource limits are required; a Python namespace restriction alone is not a sandbox.

The target reproduction is Qwen3.5-0.8B with no gradients and three pre-registered diagnostic-forward budgets: 10, 25, and 50. The goal prompt names the observable development objective (reduce loop rate while improving MATH-500 and IFEval), but does not include sealed item text, gold answers, per-item outputs, or per-item scores. The blog-reported outcome is motivation only; the project's base numbers and evaluation protocol remain the ones in `docs/RESULTS.md` and D2, with a faithful-tier comparison needed for a direct blog reproduction.

## API and isolation contract

Expose a versioned API such as `editor_api.v1`:

- `weights.list_tensors()` returns names, shapes, dtypes, and parameter roles for the permitted editable tensors. `weights.read(name, slice=None)` returns a detached, read-only CPU copy. Allow full raw tensors, matching the reported weight-reading setting, only for this small model and only if the memory/resource budget is established. Default to selected MLP output weights plus a pre-registered diagnostic set; do not silently broaden access. No mutable alias to evaluator storage is exposed.
- `probe.forward(input_ids, attention_mask=None)` accepts token IDs (or approved development prompts through a separately identified helper), runs inference mode on the frozen base, returns logits or explicitly requested hidden-state summaries, and charges the meter. It returns no training gradients, optimizer state, or eval item data. **Hidden-state probes are allowed** because raw-weight reading plus forward access is the intended setting; restrict layers/positions and output shape in the API config, log each request, and count its inference as forwards. Do not enable hidden-state access by accident: it is an explicit treatment flag and must be held fixed across editor arms.
- `delta.add(name, tensor)` / `delta.set(name, tensor)` capture proposed changes in float32 CPU memory. `delta.preview()` reports touched names, shapes, norm, sparsity, max absolute value, and estimated bf16 rounding; it does not evaluate or reveal scores. Require finite values, exact name/shape match, configured element/norm bounds, and a declared edit rationale. No direct state-dict writes.
- `budget.remaining`, `budget.used`, and `budget.limit` are read-only. Attempts over budget fail closed, are logged, and do not execute the model call.

**Meter semantics:** one charged forward is one example sequence passed through the frozen model once. A batch of B sequences costs B, regardless of whether dispatched in one batched kernel call. A prefill counts one per sequence; each autoregressive decode step counts one per live sequence. Token count and wall time are recorded separately. Logit and hidden-state requests for the same actual model invocation are one charge; a second invocation is another. Failed requests that never execute the model are not charged; executed calls that error after launch are charged conservatively. The 10/25/50 caps apply to the total per proposal, including retries and all API paths. There is no unmetered evaluator forward visible to the editor.

Editor code runs in a fresh process per proposal with no network, no host mounts except a bounded scratch directory, no credentials, CPU/RAM/time limits, and a read-only API transport. A supervisor checks API version, budget, output schema, wall-time cap, and process exit. The model worker is separate and has no access to sealed item storage through its editor-facing interface. The sealed benchmark split is opened only by the final evaluator after proposal freeze; item outputs and aggregate sealed scores never flow back into the running episode. Development prompts/items must be explicitly allowlisted, versioned, and disjoint by item ID from sealed items. The prompt/completion log is scrubbed of credentials and stored with access controls; public reports contain no raw private prompts.

## Delta capture and zoo representation

The proposal is a delta from the pinned base checkpoint, not from an earlier proposal. The writer computes `Δ_eff = bf16(W + Δ) - W` using the D1 merge rule, and stores `Δ_eff` as dense `delta.safetensors`. Preserve the submitted delta separately only in restricted episode logs if needed for audit; predictor features and standard evaluation use `Δ_eff`. Reject shape/name mismatch, NaN/Inf, empty delta, and configured magnitude violations before evaluation. An invalid or timed-out proposal remains in the episode trajectory as a failed attempt, but produces no production zoo entry.

Add `llm_edit` to `EDIT_FAMILIES` and build registration rather than disguising it as `random_dir`. Use the standard `EditDescriptor`: `family="llm_edit"`, pinned `base_model` and resolved `base_revision`, stable proposal seed, `parent_id` null for each independent base-relative candidate, touched tensor names and delta norm, and provenance containing editor/provider/model/version, prompt-template hash, episode and proposal IDs, API/sandbox versions, code/completion hashes, probe budget and used count, probe transcript hash, dtype/merge rule, base and config hashes, and all timestamps/durations. The content identity must include proposal digest and editor configuration so distinct completions cannot collide. Each valid proposal gets one zoo directory with `edit.json`, dense delta, and a manifest row; results are attached using the existing eval output format. Family eligibility distinguishes smoke from production and records reasons.

An episode is a trajectory record under `data/runs/llm_edit/<episode_id>/` containing ordered proposal references and outcomes. It is not a weight-space chain: candidates are independent edits from the same base. A later search can record proposal-parent/selection relations in trajectory metadata without changing each entry's `parent_id` semantics. Each entry retains the ordinary zoo parent field as null unless it truly applies to a predecessor state. Record all attempts, including failures and unselected candidates, to avoid survivor-only logs.

## Editor arms, controls, and cost accounting

- **Primary editor:** Codex `gpt-6-astra`, matching the blog's GPT-6 Astra editor identity in the project orchestrator. Pin the provider model ID/version available at run time and log it. If unavailable, do not silently substitute; register a separate arm.
- **Cheap LLM control:** Claude Haiku 5.5 through the configured Claude API instance. It tests whether any effect depends on the premium editor. Use the same prompt, API, and budgets.
- **Matched premium comparison:** Claude Opus 5.5 or Fable 5.1 may be a separately preregistered comparison, not a replacement for Astra. Sonnet is a mid-cost optional arm. OpenCode choices are not the default because the repo records a recent shared quota cap; record actual availability before scheduling.
- **Non-LLM controls:** (1) random-direction edits matched to the editor proposal's editable tensor/layer support, sparsity, and effective-delta norm, with diagnostic budget recorded; (2) best-of-N single-draw baseline, where N independent random-direction candidates (and, as a separately labeled control, N independent single-draw cheap-editor candidates) receive the same total real-evaluation budget and the best is selected by development evaluation. This tests the Meta^n archive/order-statistic warning. Do not compare a multi-proposal/editor archive against one draw. Report both fixed real-evaluation count and GPU-seconds.

**Cost per episode:** The repository's clean measured fast-tier evaluation is 12.4 minutes/entry and the throughput trial is 6.1 minutes/entry; the user-specified planning input is **7.5 minutes per entry**. For this design, budget one final fast-tier evaluation per valid proposal only when the arm's protocol calls for it: at 7.5 minutes each, N valid proposals imply 7.5 × N GPU-minutes, before diagnostic forwards. The actual episode calculation is `diagnostic_forward_gpu_seconds + full_eval_entries × measured_fast_tier_seconds`, with loading/caching and failed evaluations measured separately. Diagnostic forward latency is **to be measured**, not inferred from the 7.5-minute full benchmark. Editor input/output tokens, API latency, and provider charge are **to be measured**; record exact usage and do not invent dollar estimates. Publish both proposal-level and episode-level cost. For predictor-gated runs later, record rejected proposals and saved full evaluations; critic inference must not alter the editor's budget or interface.

## Gemma negative control

Run the same harness and preregistered budgets on `google/gemma-3-270m-it` using `configs/zoo_gemma-3-270m.yaml` as the architecture/config anchor. Pin its resolved revision, use its own tensor map and tokenizer, and do not transfer Qwen-specific layer indices or names. The blog's reported inability to find a fix is a negative-control hypothesis, not a guaranteed outcome. Preserve identical prompt goals where model-independent, with model-specific permitted tensor metadata. Include Gemma no-edit/base and matched random-direction controls; report failure to find a qualifying proposal as an outcome, not a missing run. Separate its results from Qwen in all aggregate claims.

## Alternatives considered

| Design | Advantages | Costs/risks | Decision |
|---|---|---|---|
| Tool-calling agent (named read/probe/edit tools) | Narrow schema, straightforward audit, fewer arbitrary-code risks. | Awkward for iterative tensor algebra, row/column projections, and custom diagnostics; provider tool-call semantics differ and may make this unlike the Python-kernel experiment. | Keep as a later safer ablation, not the initial reproduction. |
| Python code-executing REPL | Closest to the reported Python kernel; natural tensor inspection/transforms; completions and executed code can be replayed. | Arbitrary code requires strong process isolation; nondeterminism and provider-generated bugs need validation. | **Choose**, with a strict API façade and OS isolation. |
| In-process Python sandbox | Lowest call overhead and easy tensor sharing. | A Python sandbox is not a security boundary; accidental mutable access can corrupt base weights or expose sealed data. | Reject. |
| Disposable subprocess/container | Separates editor code, model worker, and evaluator; gives enforceable resource/network/filesystem limits and clean replay boundary. | IPC and tensor-copy cost; isolation setup and container availability must be qualified. | **Choose**. |

## Integration and future arm C

The editor driver should depend only on `EditorAPI` and a `ProposalResult` schema. It should not import predictor modules, know whether a critic is enabled, or receive predictor forecasts. Add a later proposal broker between completed editor proposals and full evaluation: arm C can consume the frozen proposal delta/features, return predicted score changes plus uncertainty and accept/reject/rank, and trigger real evaluation according to a preregistered uncertainty fallback. The broker writes its forecast, model/version, calibration state, decision, and eventual verification to trajectory logs. With C disabled, every eligible proposal follows the baseline evaluation policy. This preserves identical editor prompts, code execution, probe meter, and output deltas between critic and no-critic arms.

The evaluator remains the only component that reads benchmark data. Forecasting uses the existing predictor interface/features after delta validation. A forecast never overwrites measured results. Compare C against same-editor/no-C, random ranking, and equal-real-evaluation best-of-N controls, using the H5 threshold and equal-final-score criterion already specified in `docs/HYPOTHESIS.md` / `docs/MODEL_CANDIDATES.md`.

## Proposed module layout

- `src/wsp/editor/api.py`: typed read-only weight/probe API and delta collector; meter semantics.
- `src/wsp/editor/sandbox.py`: subprocess/container lifecycle, resource limits, IPC, timeout and fail-closed handling.
- `src/wsp/editor/driver.py`: provider-neutral episode/proposal orchestration; provider adapters implement a small completion protocol.
- `src/wsp/editor/providers.py`: explicit model/provider registry and usage accounting; no silent fallback.
- `src/wsp/editor/zoo.py`: delta validation, bf16 effective-delta conversion, descriptor/provenance and manifest registration.
- `src/wsp/editor/episode.py`: append-only proposal trajectory and replay metadata.
- `src/wsp/editor/eval_gate.py`: optional future forecast-then-verify broker; editor driver remains unchanged.
- `configs/editor_qwen3.5-0.8b.yaml`, `configs/editor_gemma-3-270m.yaml`: model, tensor allowlist, probe policy, budgets, limits, prompt hash, dev/sealed split references (sealed reference inaccessible to editor).
- `tests/editor/`: CPU-only API, meter, validation, serialization, zoo, isolation-protocol and deterministic replay tests, using a tiny local causal model fixture. Tests use synthetic tensors only in memory and never write research artefacts into docs/paper/assets/data.

## Tests and definition of done

CPU tests must establish: batch charges equal item count; prefill/decode and hidden-state calls charge exactly as specified; failed pre-launch calls do not charge and launched failures do; over-budget calls never execute; editor cannot mutate the base; unknown keys/shapes and nonfinite deltas fail closed; D1 effective bf16 delta matches `eval.run.apply_delta`; proposal digests yield stable distinct entry IDs; manifest/edit metadata round-trips; failed proposals remain in trajectories without production entries; sealed-item IDs/content are absent from editor IPC and logs; timeout/resource-limit exits are recorded; rerunning a recorded proposal reproduces the same submitted delta where deterministic execution is enabled. Add a Gemma config validation test without downloading weights.

DoD before any production episode:

1. Design implemented with provider/version pinning, no silent fallback, prompt/code/tool-call/token/time/budget logs, and reproducible manifests with seed, config hash, git SHA, and resolved model revision.
2. Isolation review demonstrates no editor path to sealed eval items, credentials, network, or mutable evaluator weights; Board A verifies split, accounting, and D1 merge semantics.
3. CPU test suite passes on tiny fixtures; an explicitly labeled tiny-model integration smoke proves subprocess IPC and delta registration without consuming GPU or writing research results.
4. Qwen and Gemma configs validate; the 10/25/50 budgets are tested against actual measured forward counts. Full benchmark protocol, item IDs, statistical plan, number of attempts, and selection rule are preregistered before opening sealed evaluation.
5. Pilot records per-call GPU time, peak memory, editor tokens/latency, failed proposals, fast-tier full-eval time, and complete episode cost. No cost claim is made until measured.
6. Report Qwen outcome and Gemma negative-control outcome independently, including all attempts and controls. A direct claim of reproducing blog gains requires a faithful 32k comparison and reconciled metric/loop denominators; fast-tier screening alone is not that reproduction.

## Biggest risk and preregistration handoff

The largest risk is **proposal-selection leakage and hidden evaluation access**: iterative editors can overfit visible development items, while a leaked sealed set invalidates the claimed generalization. Board A must preregister the exact development/sealed item IDs or immutable split hashes, item access boundary, prompt and stopping rule, proposal count and selection rule, score threshold for a qualifying fix, 10/25/50 treatment assignment, random seeds, matched best-of-N evaluation budget, Gemma negative-control decision rule, and final sealed-evaluation timing before the first episode.

## Implemented isolation and qualification boundary (Board A revision, 4 Nov 2026)

The current backend is Linux seccomp plus rlimits, not a container, chroot or mount
namespace. After Python/JSON initialization the editor cannot open files (including
scratch files), inspect path metadata (`stat`, `lstat`, `newfstatat`, `statx`), create
processes, execute programs or create sockets. Existing and missing host paths both
fail with EPERM. `fstat` can inspect inherited descriptors only. The only permitted
write descriptor is a dedicated IPC pipe; stdout/stderr cannot submit requests or
write logs. The supervisor validates every IPC operation against its allowlist.
No evaluator storage, credentials, model objects or mutable parent tensors are sent
to the editor. Transcript numeric arrays are hash/shape summaries; completion code
is retained privately so deterministic proposals can be replayed.

By default the editor still runs as the supervisor's OS user. This is **not isolation
from a hostile process belonging to that same user**, host administrators, kernel
exploits, shared-resource/timing side channels or a compromised supervisor/model.
No network namespace or filesystem namespace is claimed. With
`require_separate_principal: true`, a root supervisor must configure a positive,
different `sandbox.uid`; the worker clears supplementary groups and permanently
sets its gid/uid before seccomp and before executing editor code. Otherwise launch
and production qualification refuse. Templates require this flag. Deployments must
choose a dedicated unprivileged account, not an account shared with other jobs.
The root-only UID assertion is skipped in unprivileged CPU CI.

Probes execute in disposable **forkserver workers** for CPU fixtures and **spawn
workers** for checkpoint-backed models. Children explicitly use one intra-op and
one inter-op torch thread. This avoids inheriting the supervisor's parallel torch
locks and bounds CPU oversubscription; it is an isolation/reliability setting, not
a throughput claim. The sandbox itself remains a fresh exec (`python -I -S`),
with a new process group and an independent timer that kills that group even when
the supervisor is blocked writing IPC. Probe call/proposal/episode deadlines still
apply and launched failures count. Checkpoint workers reload the frozen base for
every call via `eval.run.HFRunner`; startup/loading time is charged and recorded.
`probe.device` selects `cpu` or `cuda`; `probe.kernels` selects `auto` or `reference`.
Reference Qwen kernels unwrap transformers' original torch functions, even with
causal-conv1d and FLA installed. Each worker owns its kernel policy; CUDA state is
never inherited from the supervisor. The trusted model worker is not a security
boundary for sealed storage. Production still requires a separate evaluator principal.

Denied syscalls use seccomp user notification. A supervisor thread returns EPERM
and counts denials by syscall number; every sandbox result includes enabled/error
status and counts. Unsupported notification or auditing errors fail closed. The
unprivileged host regression verifies open/stat/listdir/socket EPERM and counters.
This is supervisor-owned auditing, not an auditd deployment claim. Same-uid host
processes can still tamper with logs. Rationales are capped at 4096 characters;
completion code at 65536. Oversized strings get bounded prefixes, length and hashes
in the private log; rejected completion code is not executed. Valid completion code
is retained exactly for replay. Any bytes after the completion sentinel invalidate
that proposal, including a second sentinel.

`load_config(path, weights, pool)` validates the exact nonempty tensor allowlist,
resolved checkpoint revision, chosen scalar budget, and `development_pool_hash`
(SHA-256 of sorted-key JSON of the exact ordered approved records, via
`wsp.editor.episode.digest`) before provider use. Qwen's engineering pool is the
first 400 GSM8K training records at the pinned dataset revision. The held-out pool
is the current fixed evaluation subset records for all five configured benchmarks.
`python scripts/editor_integration.py --prepare-pools` reproduces both private JSON
files under `data/editor/pools` from the pinned dataset revisions and subset lists.
The Qwen YAML pins both hashes; any new sealed split requires qualification again.
These engineering choices do not replace scientific pre-registration or approval.
`run_episode(..., sealed_pool=...)` requires the pinned sealed records for trusted
preflight, rejects overlap by item ID or NFKC/casefold/whitespace-normalized prompt,
and records only the two hashes and disjointness statement in the episode manifest.
No sealed record is passed to the editor. The manifest also records seed, config
hash, git SHA, model revision, principal decision and torch/transformers versions.
Tests with no sealed pin are fixture-only; production requires a sealed pin.

Production remains disabled. The templates require a separate principal and cannot
launch unprivileged as shipped. A pilot still needs the §6 principal decision and
independent re-review. No network provider adapter is supplied. Real-model integration
is an explicit script, outside pytest: `python scripts/editor_integration.py
--smoke-real-model`. It checks GPU processes and lists tmux jobs first, then runs one
four-sequence GPU probe and prints load/forward/total timing. Do not launch it while
GPU evaluation jobs are active. This is a plumbing smoke, not an H5 outcome.

The `probe.v1` result contains logits only at requested positions (default: final
position), or top-k token IDs/log probabilities with k <= 64. Requested hidden
layers return only per-position norm, mean and up to 16 selected coordinate
projections. No full activation tensors cross IPC. Position/layer, batch, sequence,
model-dimension and conservative JSON byte bounds are checked before launch; actual
output dimensions and serialized bytes are also checked in the worker. Weight reads
share the response byte cap. Episode limits cover calls, sequences, tokens and
elapsed wall clock across proposals, in addition to each proposal's forward budget.
Measured elapsed wall time is logged but is not expected to be identical on replay;
deterministic call/sequence/token totals and effective-delta file hashes must match.
Delta preview and registration use the same effective_delta helper in the checkpoint's
original dtype, including bf16 checkpoints, rather than forcing all bases to bf16.
