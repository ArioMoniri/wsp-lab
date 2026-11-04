# Board A re-review — fix/eval-lora

- **Task:** fix/eval-lora (zoo LoRA entries evaluated the base model)
- **Head re-reviewed:** 43c7d9d (`git diff 28c80a9...43c7d9d`; prior review at 28c80a9)
- **Reviewer:** reviewer-ml (Board A)
- **Date:** 2026-11-04
- **Verdict:** REVISE
- **Blocking:** yes, three items (B1–B3). All are test-side and small. No code change to the loader is required.

## Summary

F1 is closed. The CPU `HFRunner` test now fails under both plausible reverts of the loader, so the suite can see the regression it was written for. D1 is implemented on the eval side: fp32 add, bf16 cast, and `rounding_stats` per tensor (fraction lost, ‖Δ‖, ‖Δ_eff‖, max lost). It is tested with a sub-ulp case. F6 and F7 are closed. F3, F4, F5 and F8 have their code in place, but their tests do not yet cover the cases they claim. Also, no test in the suite checks B@A orientation on asymmetric values, so the central merge math is verified only by my manual recomputation, not by the suite.

## Test runs

- `source scripts/env.sh && pytest -q` (worktree `.worktrees/fix/eval-lora`) → **80 passed, 5 deselected**, exit 0. Up from 77; the three new tests pass. Log: `~/.t3/scratch/.../boardA-rereview/pytest.log`.
- GPU real-adapter run: `WSP_REAL_LORA_ENTRY=/home/ubuntu/projects/wsp-lab/data/zoo/qwen3.5-0.8b/04ce1ebc4749f1c8 pytest -q -m gpu -k lora` → **2 passed, 83 deselected**, exit 0. The two tests are `tests/test_eval_lora.py::test_real_lora_smoke_changes_output` (the real-adapter test; it ran against the real entry) and `tests/test_lora_sft.py::test_generate_lora_sft_smoke_gpu` (selected by `-k lora`, see N8). Log: `.../boardA-rereview/pytest-gpu.log`.
- **F1 revert check** (scratch copies, branch untouched; `.../boardA-rereview/revert*/`). Each copy has the branch's `HFRunner` LoRA branch replaced, and I ran `tests/test_eval_lora.py::test_hfrunner_cpu_lora`:
  - `revert`: `PeftModel.from_pretrained(model, entry/"lora")`, as on `main`. → **FAILED**, `KeyError: 'transformer.h.0.attn.c_proj.weight'` (the PEFT wrapper prefixes state keys). PEFT also warns about missing adapter keys because the renamed adapter no longer matches `target_modules`.
  - `revert_merge`: the same, plus `.merge_and_unload()`. → **FAILED**, `torch.equal` False. The bf16 merge differs bitwise from the fp32-add-then-cast expectation. So this variant is caught by value, not only by key lookup.

  Both reverts fail. The GPU test would not catch either one, because it asserts only that logits changed. The CPU test is what closes F1.

## D1 check

- `apply_delta` (`src/wsp/eval/run.py`): `d` is upcast to fp32, `merged = (base_fp32 + d).to(bf16)`, `effective = merged.float() − base`. `lost = nonzero(d) & (effective == 0)`. Stats are per tensor: `fraction_nonzero_lost`, `delta_norm` (‖Δ‖), `effective_delta_norm` (‖Δ_eff‖), `max_abs_lost` (max |Δ − Δ_eff|). These match what D1 asked for. The stats are returned and written into the manifest.
- Applying W + Δ_eff in fp32 and casting to bf16 gives the same bits as bf16(W + Δ), so the eval-side application is consistent with D1.
- Sub-ulp test (`test_rounding_stats_sub_ulp`): W = 1 (bf16), Δ = [0.001, 0.125]. Expected fraction lost 0.5, max lost 0.001, ‖Δ_eff‖ 0.125. Correct; I checked the bf16 spacing at 1.0 by hand.
- Not covered: the manifest write in `main()` is not exercised by any test (see B1).
- **Zoo side of D1 is not in this diff.** D1 also says the zoo stores Δ_eff with `rounding_stats`. The canonical entry is still a LoRA adapter with no stored rounding stats. This is tracked in N2.

## Findings

### B1 — blocking: F5 is not tested for identity, only for presence
- **File:** `tests/test_eval_lora.py` (`test_hfrunner_cpu_lora`, last assertions) and `src/wsp/eval/run.py` (`HFRunner.__init__`, `main` manifest block).
- **Evidence:** the test asserts `edited.entry_hashes and edited.rounding_stats`. Those are truthiness checks. It never compares the hash to the file, and it never checks that `main()` writes `edit_id`, `delta_format`, `entry_hashes` or `rounding_stats` into `manifest.json`.
- **Failure scenario:** `entry_hashes` hashes the wrong file or a stale file, or the manifest drops the field. The suite still passes, and two runs on different adapters look identical in the manifest. That is the F5 scenario.
- **Fix:** assert `entry_hashes["lora/adapter_model.safetensors"] == sha256(file bytes)` and the same for `adapter_config.json`. Add a test that `main()` (or a small extracted manifest builder) includes the four fields.

### B2 — blocking: F3 and F4 have silent-drop paths with no test
- **File:** `src/wsp/eval/zoo_compat.py` (`lora_dense_delta`, `load_entry_delta`).
- **Evidence (code):**
  - `modules_to_save` and `lora_bias` raise. `lora_embedding_*` and `lora_magnitude_vector` raise through the `else` branch. None of these four has a test. `test_adapter_validation_and_dispatch` covers only unpaired, `fan_in_fan_out` and `bias`.
  - `delta_format` values other than `dense`/`lora` raise `ValueError`. No test covers this branch.
  - A `delta_format: "lora"` entry that also has a `delta.safetensors` file loads the LoRA and ignores the stray file. It is not raised, not warned, and not recorded in the manifest. That is a silent drop, and it is exactly the F4 scenario with the direction reversed.
- **Failure scenario:** an entry with `modules_to_save` or a stray dense delta is evaluated as if the extra tensors were not there. The manifest does not show that anything was ignored.
- **Fix:** (a) add tests for the four unsupported LoRA tensor kinds and for an unknown `delta_format`; (b) raise when `delta.safetensors` exists alongside `delta_format: "lora"` (or at least when `lora/` exists alongside `"dense"`). Either way, record the rejection in the test.

### B3 — blocking: the suite never checks B@A orientation on asymmetric values
- **File:** `tests/test_eval_lora.py` (`test_lora_eval_matches_manual_delta` and `test_hfrunner_cpu_lora`).
- **Evidence:** both tests set LoRA weights with `constant_(…, 0.1)`. The toy target is `o_proj` (32×32, square), and the CPU test's target is a square 16×16 projection. For constant A and B, `B @ A` equals `(B @ A).T`, and `A.T @ B.T` is the same matrix too. So a transposed or mis-oriented merge passes both tests. Only the scale `alpha/r` is checked. The real-adapter test asserts only `not torch.equal(base, edited)`. The independent B@A check was done by hand in my earlier review, not by a test in the suite.
- **Failure scenario:** a regression that merges `A @ B` (shape-compatible on square modules) or transposes the delta passes the whole suite. On non-square Qwen modules the shape check in `apply_delta` would catch it, so the real entry is partly guarded, but square modules and any future model are not.
- **Fix:** seed random A and B with non-symmetric values, use a non-square target (or a random square one), and assert against `(B.float() @ A.float()) * scale` computed inline in the test. Do this in the CPU test. In the GPU test, add the same equality check on one tensor (N6).

## Non-blocking

- **N1 — F4 does not dispatch on the real entry's schema.** The canonical entry `data/zoo/qwen3.5-0.8b/04ce1ebc4749f1c8/edit.json` has `delta_format` only under `params` and `provenance`. It has no top-level key. Every real eval therefore takes the fallback path: `load_entry_delta` emits `UserWarning: missing delta_format; inferring from files` and infers `lora`. The inferred value is correct, so the numbers are unaffected, but the explicit dispatch that F4 asked for is not what runs on real data. Decision for the zoo owner: write a top-level `delta_format` in the zoo descriptor on `feat/p1-zoo-build`, and make this loader accept it without fallback for new entries. Track it before any LoRA numbers go into the paper.
- **N2 — D1 zoo-side storage not implemented here.** D1 says the zoo stores Δ_eff with `rounding_stats`, and predictor features use Δ_eff. This branch implements only the eval-side application and stats. The canonical LoRA entry has no stored Δ_eff or rounding stats. Track this on the zoo branch before predictor features are built.
- **N3 — F5 hashing loads whole files into memory.** `hashlib.sha256(p.read_bytes())` reads the entire file. For a dense `delta.safetensors` of a large model that is a multi-GB transient allocation. Stream the hash in chunks.
- **N4 — F8 partly closed.** The CPU test now uses `alpha/r = 3`, so scale ≠ 1 is covered on the HFRunner path. `rank_pattern`, `alpha_pattern` and `use_rslora` are still not tested.
- **N5 — F3 gap on the zoo branch.** `src/wsp/zoo/apply.py` on `feat/p1-zoo-build` still has the old gap (unpaired tensors, `fan_in_fan_out`, `bias`). This branch fixed only its compatibility copy. Fix it on that branch, or re-diff the copies at merge (the prior review asked for this).
- **N6 — the GPU test is weak.** `test_real_lora_smoke_changes_output` asserts only that logits changed. A revert to unmerged PEFT would still pass it, which is why the CPU test is the F1 gate. Add the W + B@A equality check on at least one tensor (the earlier `gpu_check.py` did this for all 72).
- **N7 — hardcoded path.** The default entry in the GPU test is `/home/ubuntu/projects/wsp-lab/data/zoo/…`, which is machine-specific. Prefer a path relative to the repo or to `WSP_CANON`.
- **N8 — test writes into canonical data (side effect of this review).** `tests/test_lora_sft.py::test_generate_lora_sft_smoke_gpu` writes to `Path(__file__).parents[1] / "data/runs/p1-generator-smoke/lora_sft"`. In the worktree `data` is a symlink to canonical `data/`, so the write lands in canonical data. `-k lora` selects this test along with the real-adapter test. I ran the command as instructed without first checking what `-k lora` selects. That run rewrote `data/runs/p1-generator-smoke/lora_sft/` (new `edit.json` with id `0e8accac1beff06d`, mtime 2026-11-04 01:15:24). The previous contents were not backed up, and `data/` is untracked, so they cannot be restored from git. The zoo entry `04ce1ebc4749f1c8` is unaffected (mtime 2026-11-04 00:54:56). Fix: make the smoke test write under `tmp_path`, or point it at a run-scoped directory.
- **N9 — F7 closed.** The inner variable is renamed to `d`. No test needed.
- **F6 closed.** The suffix fallback is limited to `model.layers.` and `model.language_model.layers.`. The negative test covers an `other.layers.…` hit. The real entry resolves all 72 modules.

## Checked, no finding

- Dense and LoRA entries share one path: `HFRunner` → `apply_zoo_entry` → `load_entry_delta` → `apply_delta`.
- `apply_delta` validates every name and shape before any write.
- `lora_dense_delta` rejects `fan_in_fan_out` from config. PEFT itself writes `fan_in_fan_out: true` for Conv1D targets (I checked with a tiny GPT-2 adapter), so the config check catches the PEFT-written case. Only hand-edited configs slip past it, and the CPU test is one.
- The real entry's adapter config: `bias: none`, `fan_in_fan_out: false`, `modules_to_save: null`, `rank_pattern: {}`, `alpha_pattern: {}`, `use_rslora: false`, `use_dora: false`. The new checks do not reject it.

## Remaining blockers

1. **B1** — hash-value and manifest-content test for F5.
2. **B2** — tests for the unsupported LoRA tensor kinds and unknown `delta_format`, and a rejection (not silent ignore) for a stray `delta.safetensors` on a LoRA entry.
3. **B3** — a non-symmetric, non-square (or random square) orientation test for the merge, in the CPU test and, if possible, the GPU test.

N1–N8 do not block this merge, but N1 and N2 need owners before any LoRA-entry number or predictor feature goes out. N8 needs a fix so the next `-k lora` run does not rewrite canonical data.

## Verdict

**REVISE.** Fix B1–B3 and re-review. The loader changes themselves are sound; the gaps are in what the suite proves.
