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


## 10 Nov 2026 amendment — critic validation and matched search (before freeze)

Authority: PREREG A5 F5.4, [10 Nov synthesis](council/2026-11-10/SYNTHESIS.md).
This prospective section supplements the earlier design; it reports no execution.
G5 remains shadow feasibility, production disabled, and A5.7 controls any G8 trial.

Record four separate Boolean flags per treatment/attempt: `python_tool_access`,
`forward_access`, `hidden_state_access`, `feedback_access`. Pin them before the
run and hold them fixed across matched critic/no-critic arms. Log actual calls,
code/config/decode hashes, feedback exposure, sequences/tokens and all compute;
Python algebra alone does not imply model inference or hidden-state access.

The loop is forecast → schedule/abstain → measure on held-out support → rebuild
from pinned base → reload/verify → accept or roll back. Separate proposal-training,
support-validation, calibration and sealed items. Every candidate remains a
base-relative patch, never a compounded patch. Preserve actual incumbent hashes
and measure its relative improvement separately. If rebuilding changes bytes,
validate the rebuilt tensor on an untouched acceptance slice. Only measured
improvement satisfying the frozen support/feasibility/safety predicates authorises
acceptance; a critic prediction does not. Failure retains the incumbent, initially
the base. Test reversibility and deterministic reload; keep all invalid, failed,
rejected, no-op and reverted attempts. Compare architecture-compatible random
permutations of updates with matched support/count/norm and equal validation/tuning
budget, separately from feature permutations. Adaptive reused support is development;
sealed feedback cannot select edits or refit a frozen critic.

Retain the existing G5 arms and add matched random best-of-N: 32 independent
Gaussian draws per source block, effective support/norm matched, fixed scales,
seeds and draw order before outcomes. Use nested prefixes 1/8/32, select one winner
per prefix on development support, then evaluate on held-out items. Match editor
opportunities at each N; report equal-real-evaluation and equal-GPU comparisons,
charging all draws, diagnostics, failed attempts and selection. No editor advantage
claim is allowed without this comparator. Single-model selection (K=1) is separate
from top-K ensembles and their multi-pass inference costs; any ensemble is separately
registered. A5.7's ≥30% realised savings, noninferiority, missed-winner, net-cost and
prospective power conditions remain unchanged. This amendment authorises no episode.

## 11 Nov 2026 amendment — scripted-estimator arm, second editor, prior information (before any G3 outcome)

Authority: PREREG A5 F8.7, [11 Nov synthesis](council/2026-11-11/SYNTHESIS.md) §§1, 4.
This prospective section supplements the 10 Nov amendment; it reports no execution, authorises no episode, and
G5 stays shadow feasibility with production disabled. Inside the existing G5 ≤ 10 + ≤ 6 GPU-h caps; reprice before launch.

**Scripted-estimator arm.** A non-LLM proposal stream: a difference-of-means direction plus linear-programme column and
scale selection (the published replications' recipe: aiXiv 261103.000004; Barry 2026), run by a pinned script with no
model calls. It receives the same localisation (layers, matrix, column budget) that the editor receives, and nothing else
the editor does not. *[Superseded 11 Nov 2026 by F9a.5(a) — see S1 in the "11 Nov 2026 (F9a reconciliation)" section below.]* A second variant picks layers by field ranking (the base-anchored sensitivity field, not the editor's
choice). Both variants use the same forward-request meter, development/held-out split, rebuild-from-pinned-base and
rollback rules as every other arm, and are charged for every direction estimate, LP solve, forward request and failed attempt.

**Second editor.** GPT-6.1 Sol (medium reasoning) is registered as the practical editor beside Astra, which is retained
for reproduction of the blog protocol. Pin, before the first episode: provider, model identifier and version string,
reasoning-effort setting, system prompt, tool schema and decode settings, each hashed and logged per episode. No silent
substitution or provider fallback: an unavailable, renamed or changed model ends that arm and is reported, not replaced.
Voluntary early stops (an editor ending its episode before the forward-request budget) are logged with the stop reason and
the requests used, and are not discarded or imputed.

**Prior-information logging.** Whatever an arm is told beyond the weights and the goal is recorded per attempt as
prior information, not discovery: layers or matrices supplied, column budget, edit scale, direction source, and
whether it came from a published edit or from the field ranking. An arm's recovery is reported conditional on the
information supplied; the aiXiv ≈ 93 % recovery is conditional on Astra's layers, budget and scale and is not carried over.

**Per-stream savings reporting.** Evaluation savings are reported separately for each proposal stream: LLM editors (Astra,
Sol, cheap control), the scripted estimator (both variants) and random best-of-N, each with and without the critic, at
matched information and charged cost. The ≥ 30 % paired realised-savings criterion (A5.7, F5.4) concerns the critic and is
unchanged, as are the noninferiority, missed-winner, net-cost and power conditions. Any editor-advantage claim must beat both
the scripted and the random best-of-N baselines at matched information and charged cost.

## 11 Nov 2026 (council round 5) amendment — localisation flag, shadow LoRA sub-stream, re-anchoring (before any G3 outcome)

Source: [synthesis](council/2026-11-11/SYNTHESIS-r5.md) Q4, Q5 and §4; PREREG A5 amendment F9.4, F9.5, F9.9. Everything above stays; this
section adds interface and logging requirements inside the G5 caps and changes no registered arm, cap or criterion.

**Fifth access flag.** The per-episode access record gains `field_localisation_access` beside the Python, forward, hidden-state and
feedback flags. When 1, the editor receives the field's per-tensor ranking: gradient saliency g⊙W, the candidate-specific g⊙D, and
cached down-projection input norms (‖Dx‖). The ranking comes from the frozen G2 artefacts and the pinned base. The flag is logged per episode, hashed with the ranking payload, and counted as prior information (the 11 Nov
prior-information rule), never as discovery.

**Verbal-localisation logging order.** Before the editor is shown the field ranking, its own verbal site ranking (tensors or layers it
says it would edit, in order, with its stated reason) is requested and written to the episode log with a timestamp and hash. Only
then is the field ranking released. The order is enforced by the harness, not by the prompt. Verbal localisation is a logged treatment; it is not
used to gate anything. Attribution patching (clean/corrupted pairs, intermediate activations, patch sites) would be a new metered
artefact and is not part of this design.

**Shuffled-localisation control.** The control supplies a field ranking permuted across tensors with matched ranking length,
matched information content and the same number of opportunities, so that any benefit of real field localisation is separated from
the benefit of receiving some ranking.

**Test-time LoRA shadow sub-stream (G5).** An additional `lora_shadow` sub-stream, not a G8 arm. The harness trains one isolated
base-relative LoRA per frozen development query (32 queries) with next-token loss on the query context; no generation and no sealed
item text. Per-episode record: the four registered access flags (python 1, forward 1, hidden-state 0, feedback 1) plus
`backward_enabled` and `gradient_exposed_to_editor`; optimiser, step count, rank and target-module allowlists; training-data
provenance. The critic forecasts **collateral** change on the registered objectives
relative to the actual incumbent. It cannot forecast on-query gain and is not asked to. Each forecast is appended to the F9.1
forecast-before-modify log before scoring, and all 32 adapters are scored on TF-v1 under the Part A default configuration.
Forecast wall time is measured, because the only measured contraction (about 94 s CPU per donor) is slower than a TF-v1 entry and a
factored ⟨g, BA⟩ route is expected to be cheaper but is unmeasured.

**Accounting.** Training cost is charged in both arms (gated and ungated). Only verification evaluations count as savings. Savings
are computed only from evaluating adapters whose forecast is above the floor, with a missed-harm upper bound ≤ 5 % (A5.7). *[Superseded 11 Nov 2026 by F9a.5(a) — see S2 in the "11 Nov 2026 (F9a reconciliation)" section below.]* Shadow
replay may reject an uneconomic gate; it cannot demonstrate realised savings, which stay with A5.7's prospective paired G8 design.
Price: about 0.55 GPU-h scoring plus training capped at 0.5 GPU-h (F9.5).

**Re-anchoring after an accepted edit (F9.9).** When an H5 edit is accepted, the incumbent changes, so the base-anchored field no
longer describes the receiver. The harness recomputes the incumbent's exact gradient (measured about 43 s per row in the F6 audit,
`docs/reviews/council/board-a-g2-2026-11-11.md`) and flags the factors `base-factor` in the forecast log; no learned transition is
registered. *[Superseded 11 Nov 2026 by F9a.5(a) — see S3 in the "11 Nov 2026 (F9a reconciliation)" section below.]* Forecast rows record the anchor type (`base-exact`, `incumbent-exact`, `base-factor`) so that predecessor-displacement
effects are visible rather than modelled.


## 11 Nov 2026 (F9a reconciliation) — H5 design: access meters, localisation, shadow limits, re-anchoring (before any G3 outcome)

Source: [PREREG A5 amendment F9a](PREREG_H1_TF.md) F9a.5–F9a.6 (adopted 11 Nov 2026, before any G3 outcome). F9a.5(a) requires these two clauses to be copied into the H5 documents before the pilot and marks the superseded wording of the earlier 11 Nov sections of this file. Everything above stays; superseded sentences are marked in place.

**(a) F9a.5 and F9a.6, quoted verbatim from [PREREG A5 amendment F9a](PREREG_H1_TF.md)** (`docs/PREREG_H1_TF.md`, clauses F9a.5–F9a.6, main `e3920b1`):

> **F9a.5 — H5 reconciliation (condition 5).** (a) *Documents.* The governing H5 texts are `docs/H5_DESIGN.md` "11 Nov 2026 (council
> round 5) amendment — localisation flag, shadow LoRA sub-stream, re-anchoring" and `docs/H5_PROTOCOL.md` "11 Nov 2026 (council
> round 5) amendment — paired localisation episodes, shadow LoRA stream, re-anchoring" (both added by `2ab8fab`, after Board A reviewed
> `f181f0a`), `docs/HYPOTHESIS.md` "11 Nov 2026 (council round 5) prospective note — H5 and H1-OR", and the round-4 11 Nov sections
> (F8.7), each read as amended by F8a.6 and by F9a.5–F9a.6. Those two clauses are appended verbatim with F8a.6 to both H5 documents,
> and HYPOTHESIS is reconciled, before the H5 pilot. Superseded there: "with a missed-harm upper bound ≤ 5 % (A5.7)" (replaced by
> F9a.6), "run by a pinned script with no model calls" (F8a.6(a)) and "measured about 43 s per row" (F9a.5(e)). (b) *Access and
> training meters,* pinned per episode in the G5 manifest before the pilot: `python_tool_access`, `forward_access`,
> `hidden_state_access`, `feedback_access`, `field_localisation_access`, `backward_enabled`, `gradient_exposed_to_editor`. F9.4
> episodes: `backward_enabled = 0`; `gradient_exposed_to_editor = 1` exactly when `field_localisation_access = 1` (g⊙W and g⊙D derive
> from the frozen G2 gradient; no fresh backward pass). `lora_shadow`: python 1, forward 1, hidden-state 0, feedback 1 (F9.5), where
> feedback means the next-token training loss on the query context only (no TF-v1, held-out or sealed signal reaches training);
> `backward_enabled = 1`; `gradient_exposed_to_editor = 0` (no LLM editor; only the harness optimiser consumes gradients). This
> permission covers only the harness-run training of `lora_shadow`: no LLM editor arm gains backward passes, and no other arm's
> registered access changes (A5.7's gradient-based comparators keep theirs). Meters per arm: forward requests and tokens, backward
> passes and tokens, optimiser steps, GPU-seconds and wall time. Optimiser, learning rate, step, rank and target-module limits are
> pinned in the G5 manifest before the first adapter is trained, chosen without any shadow outcome; F9a sets no values. (c)
> *Localisation sequencing and controls.* Candidate-independent inputs (per-tensor g⊙W, cached down-projection input norms) may be
> released before the first candidate; candidate-specific g⊙D exists only after that candidate's realised delta and is returned as
> metered feedback on it, never guiding its own construction. The verbal site ranking is requested and logged with its hash before any
> field ranking is released (harness-enforced) and is never a critic input. The shuffled control permutes tensor labels over the same
> ranking positions, one permutation per episode drawn before the episode from `PCG64(domain_seed("20261111:f9-shuffle:", <episode
> id>))`, at matched length, support, opportunities and cost. (d) *Paired causal contrast.* The only localisation-value contrast is
> the scripted estimator on the LLM's logged layers (variant 2) against the scripted estimator with field localisation (variant 1):
> same editor, localisation alone differs. LLM with versus without the field ranking and real versus shuffled ranking are reported
> descriptively and support no claim. Endpoint: GPE_x = Σ_e gain_{x,e} / Σ_e E_{x,e}, with gain the change in A5.7's bounded held-out
> score (percentage points) of the final incumbent over the starting incumbent and E all real evaluator calls including failed calls
> and audits (A5.7); Δ_loc = GPE₂ − GPE₁ with cluster influence u_k = (g₂,k − GPE₂·E₂,k)/ΣE₂ − (g₁,k − GPE₁·E₁,k)/ΣE₁ in A5.7's paired
> episode-cluster bootstrap-t (4,000 common draws, PCG64 seed 20261107, its failure rules). A zero or nonfinite denominator makes
> Δ_loc undefined, never a success. Localisation value is credited only if the two-sided 95 % lower bound of Δ_loc is > 0; a point
> improvement alone is insufficient; there is one primary contrast and no multiplicity correction. Aggregate sensitivity is not causal
> localisation or faithful self-report, and comparisons at unequal access stay labelled (F8a.6(a)). (e) *Increments, gradients,
> re-anchoring.* Candidates stay stored base-relative (F5.4). A forecast for an acceptance decision is relative to the actual
> incumbent: the realised increment W_cand − W_inc (bf16 states, fp32 difference) contracted with an exact incumbent gradient
> (`gradient_anchor = incumbent-exact`, fresh gradient hash) and base factors (`factor_anchor = base-factor`), or, labelled
> `approx:difference_of_base_forecasts`, f_base(W_cand − W₀) − f_base(W_inc − W₀). Re-anchoring does not turn a submitted base patch
> into an incumbent increment, and an exact gradient contraction is not an exact finite-edit forecast. All re-anchoring work (gradient
> recomputation, loads, failures) is metered and charged to the critic arm's net GPU (A5.7). The historical timings are not future
> rates: 854.718742 s over 20 rows, mean 42.736 s, with different predecessor supports and including load and save/hash work
> (`docs/reviews/council/board-a-g2-2026-11-11.md`), and 13305.825513 s over 141 donors, 94.3676 s each including I/O (power
> `fields_manifest.json` summary, per Board A); per-objective re-anchoring and factored-LoRA costs are unmeasured, measured in the
> pilot and repriced under F8a.6(f). (f) *LoRA scope and groups.* The shadow forecasts completed adapters' collateral TF-v1 change
> only, not on-query gain or whether training is worthwhile; a TF surrogate cannot supply A5.7's bounded task-score endpoint.
> Query/training contexts, support, calibration, acceptance and sealed groups are disjoint by item id and source document; query
> contexts are disjoint from every TF-v1 scored item, the G2 gradient panels (including the 11 WikiText articles and the 500 reserved
> groups, A5.3.2(iii)) and any sealed panel; overlap fails closed. (g) *Contract preserved.* All of A5.7 (arms, primary comparator,
> ratio-of-totals savings, noninferiority, guardrails, the missed-proposal guardrail with its all-rejected denominator and all-zero
> unresolved case, net GPU, prospective joint power ≥ 0.80, unresolved versus kill), the F5.4 critic contract, F8.7's per-stream
> reporting and F8a.6's retained arms and matched-information rule stand. F8a.6(f)'s drop order gains, first, `lora_shadow` as one
> unit, then the F9.4 paired set, then spec §6.6's order. G5 stays shadow-only.
>
> **F9a.6 — shadow limits (condition 6).** (a) *Harm, defined separately.* Adapter j is harmful if its WikiText contrast Δy_j = y(W₀ +
> D_j) − y(W₀) under `configs/eval_tf_v1.yaml` is resolved (A5.3 paired block interval excludes 0) and Δy_j > 0; other TF-v1 endpoints
> are descriptive; a missing or nonfinite score is `unresolved`. This is not A5.7's missed qualifying beneficial proposal, which a
> collateral forecast cannot estimate; A5.7's guardrail, denominator and unresolved cases stay unchanged. (b) *Gate frozen before
> shadow outcomes* (in the G5 manifest, before the first adapter is trained): evaluate adapter j if its zero-fit forecast a_j + q_j
> (WikiText field, `base-exact`, `base-factor`) exceeds h\*, otherwise skip; a field abstention is evaluated. One gate, never tuned on
> the 32 outcomes. (c) *All 32 scored.* Every adapter is trained and scored on TF-v1 whatever the gate says; the gate is applied
> counterfactually. (d) *Read-outs.* Missed harm M_h = #(skipped ∧ harmful)/#skipped (undefined if nothing is skipped) with its exact
> one-sided 95 % Clopper–Pearson upper bound, and #(skipped ∧ harmful)/#harmful. The report states that even 32 independent skips with
> zero misses give an upper bound of 1 − 0.05^(1/32) = 0.08937 > 0.05, and fewer skips or dependence make it worse. Replay saving
> = #skipped/32 verification evaluations, descriptive, with training cost charged in both arms; E₀ and sign agreement follow F9a.2. (e)
> *Limits.* No ≤ 5 % certification, no realised-savings claim and no G8 gate or condition comes from this pilot. The shadow may report
> a gate as uneconomic, descriptively, when measured forecast cost per adapter is at least the measured verification cost per skipped
> adapter. Training, audits and failures stay in both arms' cost accounting; A5.7's unresolved cases are retained.

Clause guide (each item is in the quoted text above): access and training meters — F9a.5(b); localisation sequencing and the shuffled control — F9a.5(c); paired causal
contrast — F9a.5(d); base-relative versus incumbent increments, exact-gradient provenance and charged re-anchoring — F9a.5(e); LoRA scope and disjoint groups — F9a.5(f);
the full A5.7 / F5.4 / F8.7 contract preserved, drop order and shadow-only G5 — F9a.5(g); missed harm defined separately from missed beneficial proposals — F9a.6(a); collateral
gate frozen before shadow outcomes — F9a.6(b); all 32 adapters scored — F9a.6(c); read-outs and the 0.08937 bound — F9a.6(d); no ≤ 5 % certification, no realised-savings claim, no G8
gate, training and audits charged in both arms — F9a.6(e).

**(b) Superseded sentences in the earlier 11 Nov sections.** The earlier text is not deleted; each sentence below is marked in place with a bracketed note. Replacement sentences
are restatements of F9a.5–F9a.6 above; the quoted clauses govern.

- **S1 (access wording; F9a.5(a), F8a.6(a))** — `docs/H5_DESIGN.md`, "11 Nov 2026 amendment — scripted-estimator arm, second editor, prior information", **Scripted-estimator arm**. Superseded sentence (verbatim):
  > A non-LLM proposal stream … run by a pinned script with no model calls. It receives the same localisation (layers, matrix, column budget) that the editor receives, and nothing else the editor does not.

  Replaced under F9a by:
  > "makes no LLM editor or provider calls; it makes metered diagnostic model forwards (base generations and hidden activations, spec §6.1) through the same forward-request meter, charged to the arm; S-field additionally receives the frozen G2 field as setup access" (F8a.6(a)'s replacement text, which F9a.5(a) cites). The actual access flags of every arm are pinned in the G5 manifest before the pilot (F9a.5(b)); comparisons at unequal access are labelled and descriptive.

- **S2 (missed-harm bound and savings; F9a.5(a), F9a.6)** — `docs/H5_DESIGN.md`, round-5 section, **Accounting**. Superseded sentence (verbatim):
  > Savings are computed only from evaluating adapters whose forecast is above the floor, with a missed-harm upper bound ≤ 5 % (A5.7).

  Replaced under F9a by:
  > Under F9a.6 the gate (a_j + q_j > h\*, a field abstention evaluated) is frozen in the G5 manifest before the first adapter is trained and applied counterfactually; every one of the 32 adapters is trained and scored on TF-v1. Replay saving = #skipped/32 verification evaluations, descriptive only, with training cost charged in both arms. Missed harm M_h = #(skipped ∧ harmful)/#skipped (undefined if nothing is skipped) is reported with its exact one-sided 95 % Clopper–Pearson upper bound, descriptively, and #(skipped ∧ harmful)/#harmful is reported beside it; harm is a resolved positive WikiText change, defined separately from A5.7's missed qualifying beneficial proposals. No ≤ 5 % certification, no realised-savings claim and no G8 gate or condition comes from this pilot (F9a.6(a)–(e)).

- **S3 (re-anchoring timing; F9a.5(a), F9a.5(e))** — `docs/H5_DESIGN.md`, round-5 section, **Re-anchoring after an accepted edit (F9.9)**. Superseded sentence (verbatim):
  > The harness recomputes the incumbent's exact gradient (measured about 43 s per row in the F6 audit, `docs/reviews/council/board-a-g2-2026-11-11.md`) and flags the factors `base-factor` in the forecast log; no learned transition is registered.

  Replaced under F9a by:
  > After any accepted H5 edit, an acceptance forecast is made relative to the actual incumbent: the realised increment W_cand − W_inc contracted with an exact incumbent gradient (`gradient_anchor = incumbent-exact`, fresh gradient hash) and base factors (`factor_anchor = base-factor`), or, labelled `approx:difference_of_base_forecasts`, f_base(W_cand − W₀) − f_base(W_inc − W₀); candidates stay stored base-relative (F5.4). All re-anchoring work (gradient recomputation, loads, failures) is metered and charged to the critic arm's net GPU (A5.7). No re-anchoring rate is registered; per-objective re-anchoring and factored-LoRA costs are measured in the pilot and repriced under F8a.6(f) (F9a.5(e)).


**(c) The ≈ 43 s per row re-anchoring timing is context, not a registered rate.** The figure quoted in the earlier 11 Nov sections ("about 43 s per row") is a historical F6-audit
measurement: 854.718742 s over 20 rows, mean 42.736 s, with different predecessor supports and including load and save/hash work
(`docs/reviews/council/board-a-g2-2026-11-11.md`); the other measured contraction is 13305.825513 s over 141 donors, 94.3676 s each including I/O (power
`fields_manifest.json` summary, per Board A). Under F9a.5(e) neither is a future rate. Per-objective re-anchoring and factored-LoRA costs are unmeasured; they are
measured in the pilot and repriced under F8a.6(f). Every second of re-anchoring is metered and charged to the critic arm's net GPU; nothing in this document registers a per-row cost.

**Consistency.** The same F9a.5–F9a.6 text, the same replacement sentences S1–S4 (each listed in the document where the superseded sentence occurs) and this timing statement are appended to `docs/H5_DESIGN.md`, `docs/H5_PROTOCOL.md` and
`docs/HYPOTHESIS.md`; if any wording elsewhere in the three documents differs from the quoted clauses, the quoted clauses govern. Nothing here registers a new arm, cap,
threshold or access value, authorises an episode, or changes A5.7, F5.4, F8.7 or the H5 success criterion. This section does not itself append F8a.6; F8a.6(a)'s replacement
sentence is quoted under S1 only because F9a.5(a) cites it. G5 stays shadow-only.

## 11 Nov 2026 (F8a.6 verbatim append)

F8a.6 (A5, `docs/PREREG_H1_TF.md`) requires its own text to be appended to the H5 documents. It follows verbatim; the F9a reconciliation section above supersedes nothing in it.

**F8a.6 — H5 reconciliation (condition 6).** Spec §6 (difference-of-means inputs, LP objective, deterministic selection and
ties, α grid, S-loc/S-field, treatments, Sol pin, per-stream contract, drop order) is adopted as the H5 scripted-arm contract and
must be appended verbatim with the overrides below to `docs/H5_DESIGN.md` and `docs/H5_PROTOCOL.md`, with `docs/HYPOTHESIS.md`
reconciled, before the H5 pilot. (a) *Access wording.* "run by a pinned script with no model calls … and nothing else the editor
does not" (`docs/H5_PROTOCOL.md` 11 Nov section; same text in `docs/H5_DESIGN.md`) is replaced by: "makes no LLM editor or provider
calls; it makes metered diagnostic model forwards (base generations and hidden activations, spec §6.1) through the same
forward-request meter, charged to the arm; S-field additionally receives the frozen G2 field as setup access". The actual access
flags of every arm (spec §6.3 table, plus F9.4's `field_localisation_access`) are pinned in the G5 manifest before the pilot.
Comparisons at unequal access are labelled and descriptive; they do not meet the matched-information requirement for an
editor-superiority claim. A T0 script given |L| and B is conditional on that prior, not unrestricted discovery. (b) *Retained arms.*
Astra stays the registered primary (A5.7 primary comparison: Astra with versus without the critic); Haiku, sham reload, random
ranking and the F5.4/FP5 nested 1/8/32 best-of-N are retained with every other A5.7/G5 comparator; arms are dropped only
prospectively for funding, never by observed performance. (c) *Complete A5.7 estimator per stream*, unchanged: ratio of total real
calls including failed calls and rejected-proposal audits; both totals finite and positive at the point and in every draw;
connected episodes clustered beyond the goal × seed × budget tuple when source-, direction- or prompt-connected; paired
episode-cluster bootstrap-t with 4,000 common draws, PCG64 seed 20261107, and its failure rules; bounded score endpoint with equal
episode weighting; missed-proposal denominator of all rejected candidates including failed attempts, unauditable rejected
attempts counted as missed, zero rejected attempts undefined; net GPU saving charging diagnostics, gradients, probes, failed
attempts and amortised preparation (including the G2 fields' 1.62 GPU-h, `g2_manifest_final.json` → `caps.fields.used_hours`);
prospective joint power ≥ 0.80; unresolved versus kill distinctions. Critic and no-critic share seeds and opportunity budgets
within a stream; realised calls are observed, not forced equal. A zero or nonfinite denominator makes a stream's savings undefined,
never a success. (d) *Frozen choices.* The α-selection metric is FP5's development-winner objective (mean paired improvement on
development support relative to the actual incumbent under A5.7's feasibility/safety predicates), ties within 1e-12 relative to
the smaller α; the proposal-training, support-validation, acceptance-slice, calibration and sealed splits (spec §6.1; F5.4; FP5;
A5.7) and the actual-incumbent comparisons are frozen in the G5 manifest before calibration. (e) *F9.4.* The F9.4 paired set
(scripted with field localisation, scripted on the LLM's layers, LLM with and without the field ranking, shuffled-localisation
control) uses the same estimator and access pinning; the LLM is credited with localisation value only under F9.4's paired rule.
(f) *Cost and drop order.* Every H5 stream here is unpriced (F8a.8). The pilot is repriced inside the existing G5 caps (≤ 10 GPU-h
plus ≤ 6 for best-of-N) before launch; if it does not fit, streams are dropped in this fixed order: the F9.4 paired set as one unit
(registered last), then spec §6.6's order (S-field; T1 as one unit; Sol T0). Astra, Haiku, sham and random best-of-N are never
dropped; dropped streams are reported as unfunded. G5 stays shadow-only, G8 keeps its existing eligibility (G5 feasibility plus
G3/G4 go), and every A5.7 rejection and acceptance safeguard is retained. No episode is authorised by this block.


## 11 Nov 2026 (F8 execution spec §6, verbatim, with F8a.6 overrides)

F8a.6 (A5, `docs/PREREG_H1_TF.md`) requires spec §6 to be appended verbatim, with the overrides it lists, to the H5 documents. Source: `docs/design/F8_EXECUTION_SPEC.md` §6.1–§6.6 (subsections copied unchanged below, headings included), read at main `67642ab`. The earlier "11 Nov 2026 (F9a reconciliation)" and "11 Nov 2026 (F8a.6 verbatim append)" sections are not repeated here. Nothing here authorises an episode.

### 6.1 Difference-of-means inputs

- **Items.** The episode's *proposal-training split* of the visible pool (200 GSM8K-train + 200 MATH-train, seed 0;
  `docs/H5_PROTOCOL.md` "Adopted decisions" 4). Never support-validation, calibration, acceptance-slice or sealed items
  (F5.4; FP5 "Acceptance slices").
- **Generations.** Base completions under the pinned G5 decode settings and token cap T, thinking on, metered through the same
  forward-request meter as the editor and charged to the arm.
- **Labels.** loop = the harness's truncation flag is true (the completion reached T without terminating). non-loop =
  terminated. Labels come from the evaluator's validity/truncation fields (`docs/H5_PROTOCOL.md` "Adopted decisions" 2),
  never from correctness.
- **Minimum counts.** ≥ 8 completions per class. Otherwise the attempt is recorded as `no_direction` (a failed, charged
  attempt), the incumbent is retained, and nothing is substituted.
- **Activations.** For each layer l in scope:
  - r_l = residual stream after block l, over the last 256 generated tokens of each completion (all tokens if shorter);
  - mean within each completion, then mean over completions (equal completion weight);
  - u_l = normalize(mean_loop r_l − mean_nonloop r_l), in fp32.
- **Neuron statistics.** x_{l,j} = post-gating input of `mlp.down_proj` at neuron j over the same tokens. For column j of
  W_l = `model.layers.l.mlp.down_proj.weight`, its contribution to the u-projection at a token is p_j = (u_lᵀ W_l[:, j])·x_{l,j}.
  The column score is s_{l,j} = E_loop[p_j] − E_nonloop[p_j].

### 6.2 LP: objective, constraints, tie-breaking, edit and scale

```text
maximise    Σ_{l∈L} Σ_j s_{l,j} z_{l,j}
subject to  Σ_j z_{l,j} ≤ B_l   for each layer l ∈ L        (column budget per layer)
            0 ≤ z_{l,j} ≤ 1
solver      scipy.optimize.linprog(c = −s, A_ub, b_ub, bounds=(0,1), method="highs"), version pinned
```

- **Selection, deterministic.** For each l, take columns sorted by (−s_{l,j}, j) and keep the first min(B_l, #{s_{l,j} > 0}).
  This is an optimal vertex of the LP. Assert that its objective equals the solver's optimum within 1e-9 relative. On a
  mismatch the attempt fails closed (`lp_mismatch`). Columns with s ≤ 0 are never selected.
- **Edit.** For selected columns S_l: W'_l[:, j] = W_l[:, j] − α (u_lᵀ W_l[:, j]) u_l. Equivalently D_l = −α u_l u_lᵀ W_l S_l,
  rank one per tensor. Rebuilt from the pinned base, stored as Δ_eff = bf16(W + Δ) − W (`docs/H5_DESIGN.md` "Delta capture").
- **Scale α.** Either supplied (when the treatment supplies it), or chosen from the fixed grid (0.5, 1.0, 1.5, 2.0):
  - each grid point is one development-support evaluation, charged;
  - the winner is the argmax of measured development improvement, with ties within 1e-12 relative going to the smaller α;
  - acceptance follows F5.4/FP5 (held-out support, rebuild, untouched acceptance slice if the bytes changed).
- **Variants.**
  - **S-loc (supplied localisation):** L, matrix, B_l and α (if supplied) are exactly what the paired LLM treatment
    receives, logged as prior information.
  - **S-field (field-localised):** compute the candidate D_l for every l ∈ 0..23 at budget B and α = 1. Rank layers with the
    frozen G2 fields by r_l = a_l^{mmlu} − q_l^{mmlu}: the first-order predicted gain of the `mmlu_gold_log_prob` field
    (higher is better) minus its own curvature magnitude, both in nats per item. Take the top |L| by (−r_l, l). This ranking
    rule is a design default, chosen because MMLU is the only G2 objective tied to the reasoning task. The amendment may
    replace it before any episode, never after. |L| and B are supplied counts, logged as prior information.

### 6.3 Treatments, pairing and access flags

Each connected episode cluster k = (goal formulation × seed × budget) runs every funded arm from the same start state (the
base), the same proposal/validation/acceptance split and the same sealed panel, with frozen seeds. Critic and no-critic are
paired **within each stream**: same proposals, seeds and budget; the critic only schedules or abstains (A5.7, F5.4).

| Treatment | Arm (stream) | Information supplied (prior information) | python / forward / hidden / feedback |
|---|---|---|---|
| T0 discovery (blog protocol) | **Astra** (registered primary, reproduction; protocol unchanged) | goal + weights only | as pinned in the G5 manifest for LLM editors; identical across critic/no-critic |
| T0 | **Sol** (GPT-6.1 Sol, medium) | same prompt, tool schema and budgets as Astra | same as Astra |
| T0 | **Haiku** (cheap control, retained) | same as Astra | same as Astra |
| T0 | **S-field** | \|L\|, B (counts only) + G2 field artefact | 1 / 1 / 1 / 1 if α is tuned, else 1 / 1 / 1 / 0 |
| T0 | **Random best-of-N** (retained; FP5, nested N = 1/8/32, matched to each LLM editor's realised proposal blocks) | support/norm of the paired editor proposal | 1 / 0 / 0 / 1 (development-winner selection) |
| T0 | **Sham reload** (retained) | — | 0 / 0 / 0 / 0 |
| T1 supplied localisation | **Sol + loc** | layers 15, 19; `mlp.down_proj`; 64 columns per layer; source = published Astra edit. Exact names only if §4 PASS; otherwise the blog prose (`docs/H5_RESEARCH.md` line 10), labelled an inference | as Sol |
| T1 | **S-loc** | identical to Sol + loc | 1 / 1 / 1 / 1 (α tuned; α is not given in the prose) |
| T1 | **Random best-of-N** on the supplied support | same support | 1 / 0 / 0 / 1 |

- **Astra's reproduction treatment stays unchanged.** Astra does not run in T1 unless separately funded.
- **When access differs, the comparison is labelled.** If the LLM editors' pinned `hidden_state_access` is 0, every
  editor-versus-scripted comparison is labelled "scripted arm has hidden-state access". An editor advantage over a
  better-informed script remains reportable. An editor deficit is not attributed to the editor.
- **The G2 field is setup access.** It is logged with its measured G2 cost (fields 1.62 GPU-h,
  `g2_manifest_final.json` → `caps.fields.used_hours`) as amortised preparation, and is not charged per episode as forward
  access.

### 6.4 Sol pin

Recorded in the G5 manifest before the first episode, hashed and logged per episode:
- provider;
- model identifier exactly as the provider reports it;
- version or snapshot string;
- reasoning effort = "medium";
- decode settings;
- system prompt sha256;
- tool schema sha256;
- the provider's per-call reported model field.

If the provider offers no immutable snapshot id, the pin is the alias plus the pin date, and any per-call model field that
differs from the pinned one ends the Sol arm. The arm is reported as stopped and not replaced (`docs/H5_PROTOCOL.md` 11 Nov
"Second editor"). Voluntary early stops are logged with their reason and the requests used.

### 6.5 Per-stream savings and claims

For each stream x ∈ {Astra, Sol, Haiku, S-field, S-loc, Sol + loc, random best-of-N}, compute A5.7's contract on that stream's
own critic/no-critic pairs:
- S_x = 1 − Σ_e E_c,e / Σ_e E_0,e, with episode-cluster bootstrap-t;
- the ≥ 30 % point estimate and paired lower bound;
- the score noninferiority lower bound ≥ −1 point;
- the missed-qualifying-proposal upper bound ≤ 5 %;
- net GPU saving > 0;
- prospective joint power ≥ 0.80.

Further rules:
- **Zero denominators:** a zero or nonfinite denominator (e.g. S-loc with α supplied makes no search evaluations) makes S_x
  **undefined**, never a success. Absolute real-evaluation counts and GPU-h are reported for every stream.
- **Primary stream:** the A5.7 primary comparison stays **Astra with versus without the critic**. Other streams are
  reported. A stream's result is only called powered if its own prospective power simulation met the 0.80 bound. There is no
  retrospective best-stream selection: every funded stream is reported.
- **Editor-advantage claims** (F8.7; Board A condition 6) need, at matched treatment (T0 or T1) and labelled access: an
  episode-paired final-score difference with lower bound > 0 points against **both** the scripted stream and random
  best-of-N, with total charged cost (GPU-h, editor tokens and provider charge, reported separately) not higher.

**Retained unchanged:** G5 shadow-only status (production rejection disabled), G8 eligibility (G5 feasibility plus G3/G4
go), and all A5.7 rejection and acceptance safeguards.

### 6.6 Price

The H5 sealed scoring tier (4096 tokens) and the diagnostic-forward latency are **unmeasured**. `docs/H5_PROTOCOL.md`
"Staged plan" says they are measured before Stage 1. So every H5 item here is **unpriced**. The only measured reference rates
are the fast tier at 12.4 min/entry and the engine trial at 6.1 min/entry (`docs/RESULTS.md` "P2 — Clean fast-tier timing").
These are not the H5 tier.

The streams added here multiply the arm count. The pilot must be repriced inside the existing G5 caps (≤ 10 GPU-h + ≤ 6 for
best-of-N; `docs/council/2026-11-10/SYNTHESIS.md` §3) before launch. If the repriced plan exceeds the caps, streams are
dropped in this fixed order until it fits:
1. S-field;
2. T1 (Sol + loc together with S-loc and its random best-of-N, so T1 stays matched);
3. Sol T0.

Astra, Haiku, sham and random best-of-N are never dropped. Dropped streams are reported as unfunded.

**F8a.6 overrides of the spec §6 text above.** Quoted verbatim from F8a.6 (`docs/PREREG_H1_TF.md`); where an override and the spec text differ, the override governs.

| F8a.6 override | replaces spec text |
|---|---|
| (a) "makes no LLM editor or provider calls; it makes metered diagnostic model forwards (base generations and hidden activations, spec §6.1) through the same forward-request meter, charged to the arm; S-field additionally receives the frozen G2 field as setup access" | the earlier H5 wording "run by a pinned script with no model calls … and nothing else the editor does not" (not spec §6 text; §6.1, §6.3 already describe the metered forwards and the G2 field as setup access) |
| (a) "The actual access flags of every arm (spec §6.3 table, plus F9.4's `field_localisation_access`) are pinned in the G5 manifest before the pilot." | the fixed python / forward / hidden / feedback values of the §6.3 table, which are defaults until pinned and gain the `field_localisation_access` flag |
| (a) "Comparisons at unequal access are labelled and descriptive; they do not meet the matched-information requirement for an editor-superiority claim. A T0 script given \|L\| and B is conditional on that prior, not unrestricted discovery." | §6.3 "An editor advantage over a better-informed script remains reportable" and the §6.5 editor-advantage condition, as to matched information |
| (d) "The α-selection metric is FP5's development-winner objective (mean paired improvement on development support relative to the actual incumbent under A5.7's feasibility/safety predicates), ties within 1e-12 relative to the smaller α" | §6.2 "the winner is the argmax of measured development improvement, with ties within 1e-12 relative going to the smaller α" |
| (d) "the proposal-training, support-validation, acceptance-slice, calibration and sealed splits (spec §6.1; F5.4; FP5; A5.7) and the actual-incumbent comparisons are frozen in the G5 manifest before calibration" | §6.1 and §6.3 split statements, which are now frozen in the G5 manifest before calibration |
| (e) "The F9.4 paired set (scripted with field localisation, scripted on the LLM's layers, LLM with and without the field ranking, shuffled-localisation control) uses the same estimator and access pinning; the LLM is credited with localisation value only under F9.4's paired rule." | adds the F9.4 streams to §6.5's per-stream list; localisation value is credited only under F9.4's paired rule |
| (f) "the F9.4 paired set as one unit (registered last), then spec §6.6's order (S-field; T1 as one unit; Sol T0). Astra, Haiku, sham and random best-of-N are never dropped; dropped streams are reported as unfunded." | §6.6 "dropped in this fixed order until it fits: 1. S-field; 2. T1 …; 3. Sol T0": the F9.4 paired set goes first, then the §6.6 order |

**F9a precedence.** Where F9a.5–F9a.6 and the "11 Nov 2026 (F9a reconciliation)" section supersede a spec §6 sentence, F9a governs. One found: the §6.6 sentence "streams are dropped in this fixed order until it fits: 1. S-field; 2. T1 …; 3. Sol T0" is superseded as to order by F9a.5(g) ("F8a.6(f)'s drop order gains, first, `lora_shadow` as one unit, then the F9.4 paired set, then spec §6.6's order"); no other §6 sentence is superseded.
