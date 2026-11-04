# Board A re-review: feat/p1-heretic-lora (head c6d48ee)

- Reviewer: reviewer-ml (Board A), read-only. No code edited. No GPU job started. The GPU was shared with a P2 eval job (pid 45677, `wsp.eval.run`) during this review; I did not touch it.
- Prior review: `reviews/P1/feat-p1-heretic-lora-boardA-2026-11-03.md` (verdict REVISE, B1–B4 blocking) at ac85fc6.
- Diff reviewed: `git diff ac85fc6...c6d48ee`, 18 files. Code and test changes: `src/wsp/zoo/common.py` (new), `src/wsp/zoo/heretic_ara.py`, `src/wsp/zoo/lora_sft.py`, `tests/test_heretic_ara.py`, `tests/test_lora_sft.py`, `tests/test_zoo_common.py` (new), `scripts/check_generator_provenance.py` (new), `configs/p1_generator_smoke.yaml`, `.gitignore`. The rest is board, session, and review text.
- `main` moved during the review, from 41c34fa to e3727d9. The branch does not include the main tip. Commit 9690067 on main scrubs host identifiers from the docs (see Hygiene).

**Verdict: APPROVE.** B1–B4 are closed. Each has a regression test and was checked against the stored artifacts. The non-blocking items below are open for the P1 sweep and must stay in the session note. One record needs correcting (N3).

## Checks run

- `source scripts/env.sh && pytest -q` in the worktree: **49 passed, 4 deselected** in 9.74 s (under the 60 s budget). The GPU tests were not run here.
- `ruff check src tests scripts`: all checks passed. `ruff format --check src tests scripts`: 38 files already formatted.
- `python3 -I scripts/check_generator_provenance.py data/runs/p1-generator-smoke`: exit 0. Both entries report clean committed SHA c6d48ee and base 2fc06364715b967f1860aea9cf38778875588b17.
- Read the two stable final logs (`data/runs/p1-generator-smoke/{heretic,lora}-smoke2-final.log`) and both `edit.json`, `study.json`, `manifest.json` files.
- Read the installed heretic 2.0.0 `evaluator.py`. `_scorer_entries` exists at lines 37 and 105, which the new `extract_trial_metrics` relies on.
- Checked every `touched_tensors` entry in both `edit.json` files against the base index `snapshots/2fc06364…/model.safetensors.index.json` (488 keys): 9 and 12 names, **0 missing**.

## Blocking items

**B1. Provenance. Closed.**
- `edit.json` and `manifest.json` in both entries record `git_sha` c6d48ee69a286dc08390d6249d472056e64028f8. That is the current HEAD.
- The commit c6d48ee was made at 23:53:47 UTC. The artifacts were written at 23:54:23–23:55:18. The commit existed before the run, and the run recorded it.
- `params.git_tree` d94f3a6 equals `git rev-parse c6d48ee^{tree}`. `params.git_dirty` is False in both entries.
- The stable logs under `data/runs/p1-generator-smoke/` open with `c6d48ee…`, and both record HEAD before and after.
- Regression: `tests/test_zoo_common.py::test_smoke_provenance_check_rejects_stale_sha` checks that a stale manifest SHA fails. The GPU tests (`test_heretic_ara.py:87`, `test_lora_sft.py:69`) assert `desc.git_sha == get_git_sha()`, that `git_dirty` is False, and that the manifest SHA matches.
- Caveat (non-blocking): `git_provenance()` in `common.py` uses `--untracked-files=no`. An untracked module imported by a run would not set `git_dirty`.

**B2. touched_tensors name space. Closed.**
- `common.py:compute_adapter_stats` now emits `name.removeprefix('base_model.model.') + '.weight'`. Example from `heretic_ara/edit.json`: `model.language_model.layers.10.mlp.down_proj.weight`. This matches the base key format. I checked all 21 entries against the real base index: 0 missing.
- Regression: `tests/test_zoo_common.py::test_touched_tensors_match_unwrapped_state_dict` checks `set(touched) <= set(toy_model.state_dict())`. Before this change, every name carried the PEFT prefix, so the test would fail.
- The GPU tests also check each touched name against the real checkpoint index (`test_heretic_ara.py:117`, `test_lora_sft.py:99`).

**B3. base_revision. Closed.**
- `common.py:resolve_base_revision` calls `snapshot_download` and returns the snapshot directory name. It raises unless the name is 40 hex characters. The loaders get the resolved SHA through `model_commit` (`heretic_ara.py:118,125-126`) and `revision` (`lora_sft.py:81`).
- Both `edit.json` files record `base_revision` 2fc06364715b967f1860aea9cf38778875588b17. This matches `refs/main` in the local HF cache. Both manifests record the same value.
- Regression: `tests/test_zoo_common.py::test_revision_is_snapshot_commit_not_requested_ref` checks that `"main"` resolves to a 40-hex commit and that an unresolved path raises.
- Caveat (non-blocking): the CPU `fake_heretic` and `fake_sft` fixtures monkeypatch `resolve_base_revision` to return `"a"*40`. The real resolution inside `generate()` is covered only by the GPU test.

**B4. Refusal and KL extraction. Closed.**
- `extract_trial_metrics` (`heretic_ara.py:69-87`) matches each `KeywordRate` and `KLDivergence` plugin instance to its output label. It raises when a label is missing, ambiguous, or nonfinite. The `0.0` default and list-position fallback are gone.
- Regression: `test_scores_use_plugin_output_names_not_position` renames and reorders the outputs and expects the right values. Removing the `Divergence` output raises `ValueError` matching "Divergence".
- `select_best_trial` now raises on nonfinite `refusals` and `kl_divergence` (`heretic_ara.py:38-41`); a missing key raises `KeyError` from `trial[key]`. The `kl_divergence_target` filter is tested by `test_select_best_trial_with_kl_target` (infeasible target now raises) and `test_fake_study_filters_violating_trial` (rejects the lower-refusal trial that exceeds the target and selects trial 2).
- Confirmed in the real artifact: `study.json` has `kl_constraint_met: true`, best trial 1, KL 0.00112 with target 1.0.

## Non-blocking (open for the P1 sweep; record them as open in the session note)

- **N3 (record correction, required).** `sessions/2026-11-03-003.md` reports "before/after readings 26881 / 32623 MiB". 32623 is the total-memory column, not an after reading. In the final ARA log (`heretic-smoke2-final.log`), memory reads 26883 MiB before and **28788 MiB after**. Pid 45677 (a P2 eval job, not this branch) held 26874 MiB throughout. "Returned to baseline" is not shown for the final ARA run, and the shared GPU prevents attribution. Correct the note. The `try`/`finally` cleanup is in place.
- **N2.** An infeasible `kl_divergence_target` now raises only after all trials have run. `kl_constraint_met` is always True when a target is set, so the field is redundant. Make the check before the sweep, or drop the field.
- **N5 (open).** ARA trials are scored on `test[:2]` and selection uses those scores. Any `refusal_rate` or `kl_to_base` benchmark must use disjoint prompts.
- **N6 (open).** Dataset revisions and fingerprints are not pinned or recorded (the `load_dataset` calls in `lora_sft.py`).
- **N7 (partial).** GPU tests now assert finite losses, provenance, touched names, and nonzero adapter norms. They still do not check that loss decreases. Both ARA trials score Refusals 1.0 on two prompts, so the smoke carries no effectiveness signal, as the session note says.
- **N8 (partial).** CPU tests now exercise real plugin classes through `_scorer_entries`. Still untested: GSM8K prompt formatting (the fake dataset ignores the name) and fixed-batch learning.
- **N10 (open).** Default LoRA targets do not reach the Qwen3.5 GatedDeltaNet linear-attention modules, and the labels include prompt tokens with no EOS. Confirm this before the zoo run.
- **N4 (upstream).** Heretic 2.0.0 optimizes a row-scaled objective and exports unscaled W+BA. Note it in the P1 write-up and report it upstream.
- **New, minor.** `extract_trial_metrics` reads heretic's private `_scorer_entries`. Pin `heretic-llm==2.0.0` or add a test that fails if the attribute disappears.
- **New, minor.** `test_selection_rejects_missing_kl` accepts `(ValueError, KeyError)`. It passes through `trial[key]` raising `KeyError`, not through an explicit check.
- **New, minor.** Dead `.get(..., inf)` defaults remain in the no-target branch of `select_best_trial`. The finite check runs first, so they never fire.
- **N14 (partial).** The branch's added lines contain no host identifiers. Historical host paths remain in `board/P1.md` and `sessions/2026-11-03-002.md`.
- **N15 (closed).** `.gitignore` now covers `.venv` and `data/hf`.
- Fixed: N1 (`FixedWeightTrial` keeps the sampler's dimension equal to the evaluated weight), N9 (dataset name validated before model load), N11 (`n_startup_trials=30`), N12 (`ara_settings` precedence over `lora_rank`), N13 (`common.py`).

## Hygiene and policy

- **Weights and data.** The since-review diff adds no weights, adapters, or data. `data/runs/` is ignored and the adapters exist only in the worktree.
- **Host identifiers in the branch's additions.** None. A scan of the added lines found only basenames in the new log references.
- **Pre-existing on main (not introduced here; owner decision needed).**
  - `main` (commit b422a3b and later, history) contains the EC2 instance ID and the private hostname in `docs/INFRA.md` and `sessions/2026-11-03-001.md`. Commit 9690067 scrubs them from the main tip, but they remain in the public git history. Removing them from history is a rewrite that needs your decision.
  - 33 `data/papers/*.txt` files are tracked on main. `.gitignore` covers only `data/papers/*.pdf`. AGENTS.md says "Never commit … data", so confirm whether these are allowed.
- **AGENTS.md "no demo, mock, placeholder, or synthetic values" in paper/, docs/, figures, tables, or analyses.** The branch's `docs/` and `configs/` additions contain no placeholder, demo, mock, synthetic, TBD, TODO, or dummy values. The smoke artifacts come from real runs (`mlabonne/*` splits, Qwen3.5-0.8B). The test fakes are test-only and outside the rule's scope.

## Notes on the review process

- I did not re-run the GPU tests. Their assertions are verified by reading and by the stored artifacts.
- The protocol says blocking items must be re-reviewed. None remain.
