# Board A: H5 editor harness final re-review (feat/h5-editor @ cebe504)

Date: 2026-11-04. Reviewer: Board A (strict ML correctness). Final re-review after
`docs/reviews/H5/board-a-editor-rereview-2026-11-04.md` (at 266af77, REVISE). The branch was read and tested in
its worktree (`.worktrees/feat/h5-editor`), which was not modified. Adversarial probes live in
`/home/ubuntu/boardA-h5-final-scratch/probes` (outside the repo, not committed). The old commit's source was
extracted with `git archive 266af77 src` into the same scratch area and used as a control.

**Verdict: APPROVE to merge to main with production disabled.** R1, R2 and R3 are closed with evidence, and R4–R7
are closed or partly closed with non-blocking caveats. The first pilot episode is **not yet** permitted: the
principal decision (§6.2), the pool statement and pool-composition conflict (§6.5), and a manifest entry (§6.4)
remain. Production stays blocked (N11, plus the latent gate gap in §3.1). This is a merge decision only; no pilot
was run.

## 1. Method and evidence

- Read: prior re-review; branch diff `266af77..cebe504` (`src/wsp/editor/{sandbox,api,driver,denials,zoo}.py`,
  both editor configs, `tests/editor/test_rereview.py`, `tests/editor/test_loop.py`, `scripts/editor_integration.py`,
  `sessions/2026-11-04-h5-editor.md`); `docs/H5_PROTOCOL.md`, `docs/H5_DESIGN.md`.
- Full CPU suite at cebe504, run in tmux session `boardA-h5-final` (`. scripts/env.sh`, `pytest -q -m "not gpu"`):
  **482 passed, 1 skipped, 8 deselected, exit 0, 50.73 s.**
- `tests/editor`: **47 passed, 1 skipped** (the setuid test, root-only).
- Named regression tests, all pass: `test_wall_monitor_kills_blocked_worker[write]`, `[sleep]`,
  `test_weight_cap_before_index_allocation`, `test_three_probes_after_parent_parallel_torch`.
- GPU before any real-model probe: 0%, 0 MiB; no eval, training or wsp process running. All real-model probes
  ran on CPU. I did not re-run the GPU smoke.
- Old-source controls: the R1 test fails on 266af77 (assert `2.0 s < 1.3 s`); the R3 test fails on 266af77
  (`TimeoutError: probe hard timeout`).

## 2. Status of prior items

| Item | Status | Evidence |
|---|---|---|
| **R1** supervisor stall, wall bypass | **Closed** | Independent monitor kills the process group at the wall limit, including during a blocked write (`sandbox.py`, `threading.Timer` + `killpg`). Probes on cebe504 (wall 2 s or 1.5 s): write-without-read with 25k floats (about 250 KB reply) returns at 2.0 s; 8k floats returns at 2.0 s; 1 float returns at 2.0 s; blocked on stdin returns at 1.5 s; busy loop (CPU limit 60) returns at 1.5 s. All `TimeoutError: wall time limit`, exit −9. Control at 266af77: the write-without-read case hangs (killed by outer `timeout` at 40 s). |
| **R2** `weights.read` allocates before cap | **Closed** | `read()` sizes list slices from `len(slice) × row_numel` and refuses before indexing (`api.py:199–209`). Direct: 300 KB index list on 1024×3584 fp32 raises `weight response byte cap` with **1.4 MB** peak-RSS growth (266af77: **2,737 MB**, no error from `read`). Through IPC: supervisor peak growth 2.2 MB (266af77: 2,737 MB). Boundary: 73 rows accepted, 74 refused (exact at `max_bytes` 8 MiB); bool and float indices rejected; negative in-range indices accepted. Zero-width rows bypass the cap (N-B below). |
| **R3** forked probes hang after parent parallel torch | **Closed for the pilot path** | Probe workers no longer use `fork`: the `forkserver` context for caller-supplied models and `spawn` for `ModelSpec` (`api.py:279–285`), and each probe pins torch to one thread (`api.py:63–64`). Synthetic control: 266af77 (fork), first probe after parent matmul hits the 60 s hard timeout; cebe504 gives three OK probes (1.02 s, 0.02 s, 0.02 s). **Real pilot path** (`ModelSpec` → spawn → `HFRunner`, Qwen3.5-0.8B at `2fc0636`, CPU, reference kernels; parent does a 512×512 matmul and a 1024×3584 elementwise op before each probe): three OK, each **9.5–9.7 s** wall (7.1 s model load in child, 0.33–0.34 s forward), 248,320 logits each. The object-model path on fork (266af77) with the real checkpoint was not re-run; the synthetic control reproduces the mechanism. |
| **R4** denials not audited | **Closed** (code and test; not separately probed) | Seccomp user notification counts every denied syscall and returns EPERM (`denials.py`); counts are recorded in `ProposalResult.denial_audit` and the episode log. Audit failure fails the proposal closed (`sandbox.py`, `denial_summary.get("error")`). Test: `test_seccomp_denials_audited_and_eperm`. Counts are per syscall number; paths and arguments are not recorded. |
| **R5** unbounded editor strings in log | **Closed** | `rationale` capped at 4096 (`api.py:343`); completion code over 65,536 bytes is logged as prefix + sha256 + length and refused (`driver.py:178–185`); transcript strings over 4096 are audited (`sandbox.audit`). Test: `test_code_and_rationale_caps` (900 KB code → episode log under 10 KB). |
| **R6** forged `done` truncates transcript | **Closed** | Probe: `done` then `delta.set` on cebe504 → `valid=False`, `ValueError: bytes after done`, delta not set. Control at 266af77 → `valid=True` (reproduced). Any byte after the real `done` is an error and EOF is required (`sandbox.py`). |
| **R7** test gaps | **Closed**, with caveats | (a) Real checkpoint: a script outside pytest (`scripts/editor_integration.py`) and the pilot probe above; the suite still does not run a real model (non-blocking). (b) Positive EPERM assertions present (`test_seccomp_denials_audited_and_eperm`). (c) Stale sealed test removed; no `sealed_reference` remains. (d) `test_loop.py` asserts the scalar budget. Caveat: `test_weight_storage_slice_allocation` uses tracemalloc, which does not see torch allocations; the profiler assertion carries the weight (N-C). |
| **N3** development pool not pinned | **Code closed; protocol open** | Pins and disjointness are enforced (`driver.py:35–61`, `run_episode` requires the sealed pool when `sealed_pool_hash` is set). Independent recomputation: development digest `9024bfdc…` = config pin; sealed digest `f151e05d…` = config pin; 400 development records (gsm8k train); 1000 sealed records (gsm8k test 200, math500 test 100, mmlu_pro test 300, hellaswag validation 300, ifeval train 100); ID overlap 0; NFKC/casefold/whitespace prompt overlap 0. The protocol statement and the composition conflict are open (§3.4, §5 item 5). |
| **N4** episode wall limit | **Closed** | Follows from R1 and R3: the wall limit holds on blocked writes and on probe forwards (`api.py` deadline, `sandbox.py` timer). |
| **N6** delta bounds default silently | **Open, low** | `config.get("max_element", 1.0)` and `max_norm` 10.0 still apply silently when keys are absent (`driver.py:163–164`). |
| **N8** provenance gaps | **Open** (not a pilot blocker per §6) | Unchanged: `meter_totals` holds the per-proposal count `api.used` (`driver.py:207`); `api_version` and `sandbox_version` are literals (`driver.py:208–209`); no base-checkpoint weight hash; no start or end timestamps. |
| **N11** isolation principal | **Open** (production blocker) | Shipped configs require a separate principal. On this host (uid 1000) the pilot dry run fails closed: `failed: ValueError: separate principal unavailable` before any process starts. Latent gate gap, §3.1. |
| **N13** log integrity | **Open** (not a pilot blocker; required before results are reported) | `Episode.append` is `O_APPEND` plus `fsync`, with no chain (`episode.py:40–47`), unchanged. |
| **N14** session note | **Closed** | The note now records 482 passed at the final commit and the real-model smoke evidence. |

## 3. New findings

**3.1 (latent, blocks enabling production; not the merge) — production gate does not require a separate principal.**
`validate_production` (`driver.py:64–74`) checks the production flag, the 40-hex revision and the 64-hex pool hash,
and checks the principal only when `require_separate_principal` is truthy. I verified:
`validate_production({production_enabled: True, base_revision: 40 hex, development_pool_hash: 64 hex,
require_separate_principal: False})` returns without error. This is not reachable today: `REGISTRY` contains only
`scripted`, and `provider("anthropic")` returns `provider unavailable`. Required before any non-scripted provider
or `production_enabled: true` is merged: production must always require a separate principal, a sealed pool pin
and a resolved revision, whatever the config flag says.

**3.2 (nit, unreachable with allowlisted tensors) — zero-width rows bypass the byte cap.**
`elements = len(slice) × value[0].numel()` is 0 for a `(3000, 0)` tensor, so `read("w", [0]*300000)` returns and its
JSON would be about 1.2 MB. The allowlisted tensors are 1024×3584. Fix: `elements = len(slice) × max(1, row_numel)`.

**3.3 (test quality) — the tracemalloc peak check measures nothing about torch.**
`test_weight_storage_slice_allocation` asserts `peak < 1024**2` with tracemalloc, which does not track torch tensor
memory. The profiler assertion on the same test is the meaningful one. Drop or label the tracemalloc assertion.

**3.4 (pilot blocker, protocol) — the pinned development pool conflicts with the pre-registration text.**
`docs/H5_PROTOCOL.md` §4 (on main) says the visible pool is "200 GSM8K-train + 200 MATH-train". The pinned pool is
400 GSM8K-train only (`development.json`: 400 records, all `source: gsm8k`). The same section says the sealed set is
"frozen by item-id hash" but does not state the pool hashes or the disjointness result, which §6.5 requires. The
principal must decide which composition is intended and amend the pre-registration or the pool.

**3.5 (nit) — declared pool keys are not enforced.** `development_pool.path`, `version_required`, `sources` and
`sealed_pool_path` in the Qwen config are not read by the loader. Identity rests on the hash pins, which is correct,
but the keys look enforced when they are not.

## 4. Real-model and configuration checks

- **Loader reuse.** `ModelSpec.load()` calls `wsp.eval.run.HFRunner` (the eval loader), with `device` and `kernels`
  from the config. `run_episode` builds a `ModelSpec` when no model object is given (`driver.py:151–160`), and
  `build()` passes `None`. The editor worker loads no model. The caller-supplied object path bypasses `HFRunner` and
  is for fixtures; a pilot must not use it.
- **Kernels.** `causal_conv1d` and `fla` are importable in this venv. The pilot forward succeeded with them present,
  because `HFRunner(kernels="reference")` replaces the Qwen kernels with torch fallbacks in the child only. §6 item 4's
  "qualified CPU fallback" is therefore the `kernels: reference` policy, now verified on the real checkpoint.
- **Shipped Qwen config.** `load_config` with the real development pool and the two editable names passes, with
  `budget 10`, episode `wall_seconds 600`, sandbox `wall_seconds 600`, `call_seconds 60`, `max_vocab 248320`,
  `base_revision 2fc0636…`, `production_enabled false`, `require_separate_principal true`. Both tensor names exist in
  the cached `model.safetensors.index.json`. Gemma's template is unchanged (`base_revision: main`, pool hash null,
  `wall_seconds` 5 and `call_seconds` 5); it fails closed and is not the pilot target.
- **Pilot gate dry run.** `run_episode` with `ScriptedEditor(["pass"])`, the shipped Qwen config, and the real pools
  (sealed pool supplied) returns `[None]`, with `failed: ValueError: separate principal unavailable`. No sandbox
  process was started. The gate fails closed as designed.
- **Manifest.** `run_episode` records seed, config hash, git SHA, model revision, principal flag, torch and
  transformers versions. It does not record the kernel policy or the kernel-package state explicitly (the
  `probe.kernels` value is only inside `config_hash`).

## 5. What must be true before the first pilot episode (§6 status)

| §6 item | Status | Detail |
|---|---|---|
| 1. R1–R3 fixed, tested, re-reviewed | **Met** | §2; probes on cebe504, plus the real-model R3 run. |
| 2. Principal decision | **Not met (principal's decision)** | Option (a): root-run supervisor with dedicated `sandbox.uid`. Not exercised here; the only test is skipped as `ubuntu`. Option (b): `require_separate_principal: false`. No longer acceptable under §6's own condition, because sealed material is now on this host: `data/editor/pools/sealed.json` (1000 records, mode 600, same uid as the supervisor). Under the shipped config the pilot fails closed. |
| 3. Config values | **Met for Qwen** | Verified by `load_config` (§4). Gemma remains an unfilled template (not the pilot target). |
| 4. Environment and manifest | **Substantially met; one small change** | Kernel fallback qualified on the real checkpoint. Versions recorded. Add the kernel policy and `causal_conv1d`/`fla` availability to the manifest explicitly. |
| 5. Pool disjointness in the pre-registration | **Not met** | Enforcement and recomputation are done (§2 N3). Pre-registration text not amended, and it conflicts with the pinned pool (§3.4). Principal's decision. |
| 6. Private episode log as source of record; no N8 fields reported | **Met (procedural)** | N8 fields still exist and must not be reported. |

Not pilot blockers per §6, but they **must be closed before any result is used**: N8 (provenance) and N13 (hash
chain). N11 and §3.1 block production only. N13 adds little while the log owner shares the editor's uid.

**Remaining before a first pilot episode:** (2) the principal decision; (5) amend the pre-registration and resolve the
pool composition; (4) add the kernel policy and package state to the manifest. The pilot is a smoke harness episode
and cannot count toward confirmation.

## 6. Merge decision

**APPROVE to merge to main with production disabled.** No merge-blocking defect remains. Conditions:
1. `production_enabled` stays `false` in both editor configs, and no non-scripted provider is registered.
2. §3.1 is fixed before any non-scripted provider or production flag is merged.
3. Main moved to `08e8263` during this review (another session's `board/usage.md` commit). This review's commit sits on top of it. It was not reviewed.

## 7. Not done

- GPU smoke not re-run (CPU only); no pilot episode run; no root-path run; no run of the object-model fork control on
  the real checkpoint.
- R4 and R5 were checked by code reading and the existing tests, not by separate adversarial probes.
- Scratch artefacts are not committed: `/home/ubuntu/boardA-h5-final-scratch/{probes,old}`.

## 8. Test counts observed

- Full CPU suite at cebe504 (`-m "not gpu"`): 482 passed, 1 skipped, 8 deselected, exit 0, 50.73 s.
- `tests/editor`: 47 passed, 1 skipped.
- Old-source controls (266af77): R1 test fails (2.0 s against a 1.3 s bound); R3 test fails (probe hard timeout).
- Probes on cebe504: R1 (5 cases) returned at the wall; R2 (direct, IPC, boundary) as in §2; R3 synthetic and real
  pilot path (three probes each) passed; R6 forged `done` rejected; pilot gate fails closed on principal.
