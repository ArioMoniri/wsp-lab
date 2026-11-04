# Board A review: feat/p1-heretic-lora (head ac85fc6)

- Reviewer: reviewer-ml (Board A), read-only. No code edited. No GPU job started.
- Diff: `git diff main...feat/p1-heretic-lora` (8 files, +897). Base main 904044c.
- Scope: `src/wsp/zoo/heretic_ara.py`, `src/wsp/zoo/lora_sft.py`, `tests/conftest.py`, `tests/test_heretic_ara.py`, `tests/test_lora_sft.py`, `configs/p1_generator_smoke.yaml`, `board/P1.md`, `sessions/2026-11-03-002.md`.
- Reference: installed `heretic-llm` 2.0.0 in the worktree `.venv` (`config.py`, `model.py`, `evaluator.py`, `plugin.py`, `modifiers/ara.py`, `main.py`).

**Verdict: REVISE.** Four blocking items (B1–B4). None is a design rejection. The heretic 2.0.0 integration is sound: every import and call resolves against the installed package, and the optimization loop order matches heretic's own.

## Checks run

- `source scripts/env.sh && pytest -q`: 42 passed, 4 deselected, 9.17 s (under the 60 s budget).
- `ruff check src tests`: all checks passed. `ruff format --check src tests`: 35 files already formatted.
- Read the installed heretic 2.0.0 source (see Reference above).
- Inspected the smoke artifacts under `data/runs/p1-generator-smoke/` (gitignored, not in the diff): `edit.json`, `manifest.json`, `study.json`, `train.json`, `adapter_config.json`, and the safetensors headers. Compared the tensor names with the cached base checkpoint index for `Qwen/Qwen3.5-0.8B`.
- Read both smoke logs. Each is two lines: the pytest summary and nothing else.
- Scanned the diff for IP addresses and key or token patterns. None found.

## Findings

### Blocking

**B1. The GPU smoke artifacts were produced by uncommitted code, and their `git_sha` names a different tree.** (blocking: yes)

- Evidence: `edit.json` and `manifest.json` in both smoke entries record `git_sha` a000d79, committed 2026-11-03 23:35:44. The next commit, ac85fc6 (23:40:17), rewrote `src/wsp/zoo/heretic_ara.py` (156 lines changed) and `src/wsp/zoo/lora_sft.py` (170 lines changed) and the tests. The adapter directories were written at 23:38–23:39, before ac85fc6 existed.
- Mechanism: `get_git_sha()` returns only HEAD (`src/wsp/zoo/heretic_ara.py:25-33`, `src/wsp/zoo/lora_sft.py:34-42`). It records no dirty flag and no tree hash.
- Failure scenario: an entry regenerated from ac85fc6 does not match the manifest that backs the reported smoke numbers. The manifest-to-git-SHA reproducibility check fails.
- Fix: re-run both smokes from the committed head through `agent-tmux`, regenerate the artifacts, and record a dirty flag or tree hash alongside the SHA.

**B2. `touched_tensors` uses PEFT wrapper names, not base-checkpoint tensor names.** (blocking: yes)

- `src/wsp/zoo/heretic_ara.py:81` and `src/wsp/zoo/lora_sft.py:63` build `f"{name}.weight"` from `peft_model.named_modules()`. The result is `base_model.model.model.language_model.layers.12.mlp.down_proj.weight` (see `edit.json` in both entries).
- The base checkpoint's keys are `model.language_model.layers.12.mlp.down_proj.weight`. I checked this against the cached `Qwen/Qwen3.5-0.8B` index (488 keys).
- Failure scenario: a consumer that joins `touched_tensors` with base `state_dict` keys matches zero tensors for every LoRA entry. The sibling branch writes `touched_tensors = sorted(delta.keys())`, where the keys come from `model.state_dict()` (`feat/p1-zoo-build` `src/wsp/zoo/build.py:93` and `:429`). The field this review was asked to check is unusable for LoRA entries.
- Test gap: `tests/test_heretic_ara.py:58` and `tests/test_lora_sft.py:40` assert only `len > 0`. `tests/test_lora_sft.py:151` asserts a substring. None of them would catch this.
- Fix: strip the `base_model.model.` prefix (or map names via `get_base_model()`). Add a test that every touched name is a key of the base model's `state_dict()`. The toy model in `tests/conftest.py` is enough for that test.

**B3. The base revision is unpinned. `"main"` is used silently and recorded as if it were a revision.** (blocking: yes)

- `src/wsp/zoo/heretic_ara.py:118` passes `model_commit = params.get("base_revision")`. When that is None, Heretic loads whatever `main` points to. `src/wsp/zoo/heretic_ara.py:213` and `src/wsp/zoo/lora_sft.py:110` default to `"main"`.
- `configs/p1_generator_smoke.yaml` has no `base_revision`. Both manifests record `model_revisions: {"Qwen/Qwen3.5-0.8B": "main"}`, and both `edit.json` files carry `base_revision: "main"`. `EditDescriptor.id` hashes `base_revision` (`src/wsp/schema.py:59`), so the entry ID does not pin the model either.
- The local HF cache resolves `refs/main` for this model to `2fc06364715b967f1860aea9cf38778875588b17`. The SHA was available when the artifacts were generated.
- Failure scenario: the same config and seed yields a different edit once upstream `main` moves. The recorded revision cannot identify the weights that were used.
- Fix: resolve the commit SHA at generation time, or require a 40-hex `base_revision` in configs. Raise on `"main"` or a missing value. Pass the same value to `model_commit`.

**B4. Refusal and KL extraction falls back silently to list position, and a missing KL becomes 0.0.** (blocking: yes)

- `src/wsp/zoo/heretic_ara.py:148-159`: refusals is the first score whose name contains "refusal" or "keyword", otherwise `scores[0]`. KL is the first score whose name contains "kl", otherwise `scores[1]`, otherwise `0.0`.
- Failure scenario: the KL scorer is configured with a `score_name` that lacks "kl" (for example "Divergence") and it is the only non-refusal scorer. Then `kl_div` becomes 0.0 (lines 156–159). `select_best_trial`'s filter `t.get("kl_divergence", 0.0) <= kl_divergence_target` (line 51) admits every trial, so the KL constraint is vacuous. The same mis-attribution applies to any second scorer whose name does not match.
- The defaults happen to match ("Refusals" from `heretic/scorers/keyword_rate.py:49-52`, "KL divergence" from `heretic/scorers/kl_divergence.py:39-40`), so the smoke numbers are correct. The fallback is still a silent failure path. The fakes in `tests/test_heretic_ara.py:163-166` use the same names, so the fallback is never exercised.
- Fix: look scorers up by plugin class or configured instance name and raise if one is missing. Remove the `0.0` default.

### Non-blocking (fix before the P1 zoo run)

- **N1 (Medium). `preserve_good_behavior_weight` is overwritten after Optuna suggested it.** `src/wsp/zoo/heretic_ara.py:138-140`. TPE still models the suggested value for that dimension, so its posterior over that dimension is noise and its guidance for the other parameters is distorted. The study records are truthful, because they use the overridden value. Heretic's own loop does not do this (`main.py:678-694`). Fix: make the weight fixed inside the trial, so the sampler sees the value that was evaluated.
- **N2 (Medium). An infeasible `kl_divergence_target` silently returns a violating trial.** `src/wsp/zoo/heretic_ara.py:54-60`. `study.json` (`src/wsp/zoo/heretic_ara.py:196-202`) does not record whether the constraint was met. `configs/p1_generator_smoke.yaml` does not set `kl_divergence_target`, so no artifact exercises the KL-target branch. Fix: record `kl_constraint_met` in `study.json`, and fail or flag when no trial qualifies. Set the target in the P1 config.
- **N3 (Medium). GPU memory release has no evidence and no failure path.** The logs contain only the pytest summary, and the session note's "GPU memory returned to baseline" has no readings behind it. Release happens only on success (`src/wsp/zoo/heretic_ara.py:228-231`, `src/wsp/zoo/lora_sft.py:223-226`); there is no `try`/`finally`. Fix: log `nvidia-smi` before and after in the `agent-tmux` log, and wrap the release in `try`/`finally`.
- **N4 (Medium, upstream, note for the write-up). Heretic's ARA objective does not score the exported model.** `heretic/modifiers/ara.py:344-351` applies per-row scaling inside the objective only. The exported model (`heretic/model.py:323-371`, `merge_and_unload`) and the evaluator use W + BA without that scaling. The generator's artifact is consistent with heretic's export, so the mismatch is inside heretic 2.0.0 and is not a bug in this branch. It should be stated in the P1 write-up and reported upstream.
- **N5 (Medium). Selection and evaluation splits overlap by design.** The ARA trials are scored on `test[:2]` for KeywordRate and KLDivergence (`configs/p1_generator_smoke.yaml:12-14`), and the best trial is picked from those scores (`src/wsp/zoo/heretic_ara.py:186`). Any `refusal_rate` or `kl_to_base` benchmark in the zoo must use prompts disjoint from these splits, or selection leaks into evaluation.
- **N6 (Medium). Dataset revisions are not pinned or recorded.** `src/wsp/zoo/lora_sft.py:138` and `:141` call `load_dataset` without a revision. The heretic prompt specs have no revision either. Neither `train.json` nor the manifests record a dataset revision or fingerprint. The GSM8K split is correct: train only, no test split (`lora_sft.py:141`).
- **N7 (Medium). The GPU smoke tests assert almost nothing.** `tests/test_heretic_ara.py:102-108` and `tests/test_lora_sft.py:83-89` check file existence and counts only. They do not check finite losses, a nonzero adapter norm, or that the scorer configs were applied. The ARA smoke's two trials both score Refusals 1.0 on two prompts, so they carry no signal.
- **N8 (Medium). Test coverage gaps.**
  - Real `Model`, `Evaluator` and scorer wiring never runs on CPU (`tests/test_heretic_ara.py:111-180`). The `scorer` config namespace (consumed at `heretic/evaluator.py:74`) is only proven by the GPU run.
  - The LoRA fake dataset ignores the dataset name (`tests/test_lora_sft.py:125-127`), so the GSM8K prompt formatting (`lora_sft.py:142`) is untested.
  - Nothing checks that the loss decreases on a fixed batch. The smoke losses are 2.12, 1.72, 2.37, 2.17, 2.54. Five steps at batch size 1 may be noise, but nothing tests learning.
  - `tests/conftest.py:5` changes torch's thread count globally at import time.
- **N9 (Low).** `src/wsp/zoo/lora_sft.py:137-145` validates the dataset name after the model has been loaded (`:116-121`). Parameter validation (`:108-109`) runs first, but dataset validation does not.
- **N10 (Low). LoRA target and label design.** The default targets (`lora_sft.py:100`) do not cover the Qwen3.5 linear-attention modules (`in_proj_qkv`, `out_proj`), so the default LoRA family never touches the GatedDeltaNet layers. Confirm this is intended. The labels (`lora_sft.py:171-172`) include the prompt tokens, and no EOS is appended (`:163-169`), so the model never learns to stop, which could feed `loop_rate`. Document or fix.
- **N11 (Low).** `src/wsp/zoo/heretic_ara.py:173` uses `min(10, n_trials)` for `n_startup_trials`. Heretic's default is 30 (`heretic/config.py:411-414`, used at `main.py:731`). Document the deviation or match the default.
- **N12 (Low).** `src/wsp/zoo/heretic_ara.py:130`: a `lora_rank` key inside `ara_settings` raises a `TypeError` (duplicate keyword).
- **N13 (Low).** `compute_adapter_stats` and `get_git_sha` are duplicated in both modules (`heretic_ara.py:25-33, 68-84`; `lora_sft.py:34-42, 50-66`). Move them to one module.
- **N14 (Low, public-repo hygiene).** `board/P1.md` and `sessions/2026-11-03-002.md` cite host paths under the agent-log directory. No IP addresses or keys were found in the diff. Prefer log basenames or repo-relative paths.
- **N15 (Low).** The worktree lists `.venv/` and `data/hf/` as untracked. Neither is in the diff. Confirm `.gitignore` covers them before merge.

## What checks out

- **heretic 2.0.0 API.** Every import and call resolves against the installed package: `InitOnlySettings` (`config.py:615`), `Model`, `Evaluator`, `Context`, `empty_cache`, `get_model_class`, `ara.Settings` / `ara.ARA` / `ara.Parameters`, `init`, `suggest_parameters`, `reset_model`, `modify_model`, `get_scores`, `get_objective_values`, `get_objective_directions`. The loop order (suggest, reset, modify, scores, objective) matches `main.py:678-694`. The TPE settings match, apart from `n_startup_trials` (N11). The `scorer` namespace is consumed (`evaluator.py:74`).
- **LoRA format.** `delta_format: "lora"` is set in `edit.json` and in `params`. `adapter_config.json` has `r=4` and `peft_type: LORA`. The adapter tensors are F32 (96 tensors for ARA, 24 for LoRA SFT), so the adapter weights are fp32 despite the bf16 base.
- **`study.json`.** Each trial stores `refusals`, `kl_divergence` and the full parameter set. When a KL target is set, `select_best_trial` filters with `<=` (`heretic_ara.py:51`). The tests cover both branches (`tests/test_heretic_ara.py:26-38`).
- **LoRA SFT.** Seeds are set for torch, CUDA, `random`, NumPy and transformers (`lora_sft.py:89-95`), and the prompt shuffle is seeded (`:149-150`). The base model loads in bf16 (`:119`). In the touched list, the q/v targets land only on `self_attn` layers (3, 7, 11, …), with no vision or MTP modules. The per-step loss is in `train.json`. Training uses the GSM8K train split only.
- **EditDescriptor.** `delta_fro_norm` is ‖B·A·scale‖_F (`lora_sft.py:199-211`, `heretic_ara.py:204-216`). That matches the exported weight change (BA) and the sibling branch's delta norm. `family`, `seed`, `base_model`, `params` and `parent_id` (null) are set correctly. `git_sha` is filled but stale (B1).
- **CPU tests are real.** The LBFGS optimization (ARA) and PEFT training run for real. Only `Model`, `Evaluator` and prompt loading are faked, at the I/O boundary. Determinism is asserted by comparing adapter tensors across two runs (`tests/test_heretic_ara.py:183-204`, `tests/test_lora_sft.py:131-152`). The base model is checked to stay frozen. Runtime 9.17 s.
- **No weights or data committed.** The diff is 8 text files. `data/runs/` and `*.safetensors` are gitignored, and the adapters exist only in the worktree.
- **Secrets.** No keys, tokens or IP addresses in the diff. The only pattern matches were tokenizer identifiers.

## Required before re-review

1. Fix B1–B4, then re-run both GPU smokes from the new committed head through `agent-tmux`, and record a clean-tree SHA.
2. Re-run `pytest -q` and `ruff` at the new head.
3. Post the re-review through the board. Non-blocking items may follow in the P1 sweep if the author records them as open in the session note.
