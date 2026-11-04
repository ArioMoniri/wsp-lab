# Board A review: fix/heretic-ara-export (head d2193cf) — 2026-11-04

Reviewer: reviewer-ml (Board A). Read-only: no code edited, no GPU job started.
Scope: `git diff main...fix/heretic-ara-export` (6 files: `src/wsp/zoo/heretic_ara.py`, `src/wsp/zoo/apply.py`,
`tests/test_heretic_ara.py`, README, `board/P1.md`, `sessions/2026-11-04-p4-boardA-revise.md`).
Inputs: `docs/REVIEW_PROTOCOL.md`, `docs/DECISIONS.md` (D1, D6, D6 addendum), `reviews/P1/heretic-ara-densify-check-2026-11-04.md`,
smoke log `~/agent-logs/ara-fix-smoke-20261104-095732.log`, smoke entry `data/runs/p1-generator-smoke/heretic_ara/`,
Heretic source at `.venv/.../heretic/` (model.py, modifiers/ara.py, scorers/kl_divergence.py).

## Verdict: REVISE

Two blocking items (B1, B2), both small to fix. The export and verification logic is sound: the smoke reproduces, the
artifact is exactly the bf16 deployed delta, and the gate works on the staged file. The blockers are in the surrounding
contract: the zoo build path records the wrong format for every rebuilt entry, and a failed gate can leave a previous
passing artifact in place.

## Checks (what was verified)

| # | Requirement | Result | Evidence |
|---|---|---|---|
| 1 | Best trial re-applied once via Heretic API, evaluated in-process, `verified_*` recorded, not re-run via Optuna | PASS | `heretic_ara.py:207-214`: `reset_model` then `modify_model(best_params)` once; `get_merged_model()` then `get_scores()` on the merged model. `study.json` has `verified_refusals: 1.0`, `verified_kl: 0.001249964232556522`. `tests/test_heretic_ara.py:361-380` asserts 3 `modify_model` calls (2 trials + 1 re-apply). Note: re-apply is a re-optimisation with the same seed, not a replay of stored weights (see N2). |
| 2 | Delta = merged − base, fp32 then bf16 (D1), changed tensors only, dense, raw PEFT adapter unused | PASS | `merged_delta` (`heretic_ara.py:90-109`): bf16-round merged, fp32 diff against bf16 base, omit zero diffs. `apply.py:153-154` adds in fp32 and casts to bf16. Artifact check (CPU, read-only): 9 fp32 tensors, keys == `touched_tensors`, all in base index, and every delta is exactly on the bf16 grid (`bf16(W+Δ)−W == Δ`, all 9 True). `generate` no longer calls `save_pretrained`; `apply.load_entry_delta` prefers `delta.safetensors`. |
| 3 | `delta_fro_norm` from dense delta, plausible magnitude | PASS (magnitude plausible, see N3) | Recomputed from file: 0.00022312671213 vs `edit.json` 0.00022312671216. Per-tensor nonzero fraction 0.10–1.05%, max |Δ| 0.5–3.8e-6 (about one bf16 ulp for small weights). Relative norm about 5e-6 against base norms of 15–20 per tensor: near-identity, consistent with a 1-step, rank-4 edit whose fp32 change was mostly lost in bf16 rounding (D1). |
| 4 | Reload gate: base + delta via `load_edited_model`, KL on Heretic's good-evaluation prompts, written only if \|KL_reloaded − verified_kl\| ≤ 0.01, failures recorded with both numbers | PASS on logic; gate has holes (B2, N1) | Smoke log and `study.json` both show `verified_kl == reloaded_kl == 0.001249964232556522`, refusals 1.0 / 1.0. Reload reads the staged `delta.safetensors` back via `load_edited_model` (`heretic_ara.py:252`), so the on-disk file is what is checked. Nothing is published before the gate (`:263-274`). Failure writes `reloaded_kl`, `verified_kl`, `kl_reload_difference`, `verification_status: failed` (`:258-270`). KL scorer prompts are Heretic's `KLDivergence` set, default `harmless_alpaca` (`scorers/kl_divergence.py:24-30`), which is the good-evaluation set. Smoke uses only 2 prompts. |
| 5 | Row normalisation: does Heretic's merged model apply the objective's normalisation? Does the gate make it moot? | Answer below | See "Row normalisation" section. |
| 6 | Tests real, ruff clean, no weights or secrets | PASS | CPU run in tmux: `pytest -q` → `401 passed, 8 deselected` (gpu/slow/net excluded by `pyproject.toml:39`), exit 0. `ruff check .` → All checks passed. `ruff format --check .` → 372 files already formatted. Diff contains only 6 text files. `data/` is a symlink to the main repo's `data/`, and `data/runs/` is gitignored. Secret-pattern grep on the diff: no hits. Smoke was run at `38d1d58`; `git diff --quiet 38d1d58 d2193cf -- src tests` is clean, so the smoke tested the head code. |

## Row normalisation (item 5)

**Answer:** No. Heretic's merged model does not apply the normalisation used in its objective.

- The normalisation lives only inside `ARA.modify_model`'s `objective` (`heretic/modifiers/ara.py:345-349`, default
  `preserve_row_magnitudes=True`, `ara.py:71-75`, used at `ara.py:344`). It rescales each output row by `‖W_base row‖ / ‖(W_base + B@A) row‖` to
  compute the LBFGS loss. This is a surrogate model, not the model that gets scored.
- The scored model is the plain PEFT forward (`model.py` `get_logits` → `generate`). `grep normaliz model.py evaluator.py`
  finds nothing, so trial refusals and KL are measured without row normalisation.
- The export `get_merged_model` is `merge_and_unload()` (`model.py:323-370`): `W_base + scaling·B@A`, no row normalisation.

**Consequence for the stored artifact:** none. The exported weights are exactly the scored weights (verified in-process
and re-checked from the staged file by the reload gate). The stored `verified_kl`/`verified_refusals` describe the
artifact correctly. So the KL gate does make the question moot for correctness of what we store.

**Consequence for the search:** the optimiser follows a row-rescaled objective that neither the scorer nor the export uses.
The ARA edits are therefore not the row-norm-preserving edits the upstream ARA definition describes. This affects how
`heretic_ara` should be described in the paper, and it is recorded as N4.

## Blocking

### B1 — Zoo build path mislabels dense ARA entries as LoRA (blocking)
- `src/wsp/zoo/build.py:514` `_with_lora_extras` sets `params["delta_format"] = "lora"` for every LORA family, including `heretic_ara` (`LORA_FAMILIES`, `build.py:116`). `build.py:560` (`_lora_descriptor` provenance) and `build.py:305` (`_stored_format`) and `build.py:1017-1018` (manifest line) repeat the `lora` label.
- Reproduced on CPU (`~/agent-logs/ara-review-build-path-check.py`, with a real dense generator `edit.json`): `build._lora_descriptor` returns `params.delta_format = "lora"`, `provenance.delta_format = "lora"`, `_stored_format = "lora"`, while the entry directory holds `delta.safetensors` (dense).
- Failure scenario: the family rebuild the D6 addendum requires runs through `build_lora_entry` (`build.py:570`, called at `build.py:979`), so every rebuilt `heretic_ara` entry has `edit.json` and manifest rows that say `lora` for a dense artifact. `features/extract.py:158` detects the format from files, so features are correct. `zoo_compat.load_entry_delta` never sees a top-level `delta_format` (see N6) and falls back to file inference with a warning. Metadata, config hash and artifact disagree, which breaks the manifest ↔ config hash ↔ artifact integrity Board A checks.
- Fix: derive the stored format from `generated.params["delta_format"]` in `_lora_descriptor`/`_stored_format` (or set `dense` for `heretic_ara` in `_with_lora_extras`). Add a regression test in `tests/test_zoo_lora_build.py` that builds a `heretic_ara` entry from a dense generator output and asserts `delta_format == "dense"` everywhere, including the manifest line. Entries built before the fix are already quarantined.

### B2 — A failed gate can leave a previous passing artifact in place (blocking)
- `heretic_ara.py:247-274`: `generate()` never clears `out_dir`. On gate failure it writes only `study.json` (`verification_status: failed`) and raises. Any `edit.json`, `delta.safetensors`, or `manifest.json` from an earlier run in the same directory survives.
- Found by code reading (not reproduced): the only writes to those three files are on the success path and nothing deletes them. The toy failure test (`tests/test_heretic_ara.py:329-358`) uses a fresh `tmp_path`, so it cannot catch this.
- Reachable: the GPU smoke writes to a fixed directory (`tests/test_heretic_ara.py:95`), and `data/runs/p1-generator-smoke/heretic_ara/` already contains a pre-fix `lora/` from 2026-11-03 23:54 (N5).
- Failure scenario: run passes once; a later run with a mismatch raises; the directory now has a `failed` study and a passing `edit.json` + `delta.safetensors`, which `entry_has_artifacts` treats as usable.
- Fix: on failure, remove `edit.json`, `delta.safetensors`, `manifest.json` from `out_dir` before re-raising (or publish from a fresh directory only). Add a test: pre-create an `edit.json` in `out_dir`, force a mismatch, and assert the artifact files are gone. Delete the stale `lora/` from the smoke entry before any reuse.

## Non-blocking

- **N1 — gate tolerance is absolute and weak for this KL scale.** `heretic_ara.py:263` uses `|Δ KL| ≤ 0.01`. The smoke verified KL is 0.00125, so an artifact reloading at 0.0113 would pass (about 9× the KL). The gate is KL-only, so an artifact with a different weight set but equal KL also passes. The spec requires 0.01, so this is compliant. Recommended: add a free exact check, `torch.equal` on every tensor between the merged state and `reloaded_state` before loading. Optionally a relative tolerance.
- **N2 — trial KL and verified KL differ by 2× and the gap is not explained.** Trial 0 reports `kl_divergence 0.00256`; re-applied and merged bf16 gives `0.00125`. Plausible causes, not tested: D1 rounding (PEFT fp32 adapter path vs. bf16-rounded merge), and GPU non-determinism in the re-optimisation. The gate does not compare against the trial value. Recommended: record the KL of the unmerged re-applied PEFT model as well, so the gap can be attributed. Until then, treat trial KLs as upper bounds for deployed KL.
- **N3 — D1 rounding stats not stored in the zoo entry.** D1 says the zoo stores Δ_eff "together with `rounding_stats`". `heretic_ara` does not write them. They are computed downstream in `features/extract.py:112,251` and recorded by the eval loader (`eval/batch.py:340`), and other families' entries do not store them either. This is probably the intended location. The orchestrator should confirm that reading of D1. Without the pre-rounding magnitude, the plausibility of the 2.2e-4 norm cannot be fully judged.
- **N4 — row normalisation is not recorded and not documented.** See the section above. Recommended: record the effective `ara.Settings` (at least `preserve_row_magnitudes`) in `params`, because default values are not captured in the ID. Document that `heretic_ara` optimises a row-rescaled surrogate and that the scored/exported model is not rescaled.
- **N5 — stale pre-fix `lora/` in the smoke entry.** `data/runs/p1-generator-smoke/heretic_ara/lora/` (2026-11-03 23:54) is left from before the fix. Loaders prefer `delta.safetensors`, so it is not used. Remove it (tied to B2).
- **N6 — `zoo_compat.load_entry_delta` reads a top-level `delta_format` that `EditDescriptor` never writes** (`eval/zoo_compat.py:97-106`). It always falls back to file inference with a warning. This is pre-existing and not caused by this branch. It becomes material only if B1 is left unfixed.
- **N7 — smoke evidence is plumbing only.** Two trials, two KL prompts, two refusal prompts. Refusals stay at 1.0, the same as baseline, so no refusal reduction is shown. No efficacy claim should cite this entry.
- **N8 — GPU path is tested only by the smoke.** The rest of the suite runs against a fake Heretic `Model` and `Evaluator` (`tests/test_heretic_ara.py:143-239`). The real-merge path is covered only by the smoke run.

## Artifacts used for this review

- `~/agent-logs/ara-review-boardA-tests-100150.log` (pytest, ruff check, ruff format)
- `~/agent-logs/ara-review-artifact-check.py` (CPU read of smoke `delta.safetensors` against base shards)
- `~/agent-logs/ara-review-build-path-check.py` (CPU reproduction of B1)
- No GPU job was started; the D3 batch was not touched.
