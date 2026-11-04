# Board A review — fix/eval-lora

- **Task:** fix/eval-lora (zoo LoRA entries evaluated the base model; issue "eval: LoRA zoo entries evaluate the base model")
- **Head:** 28c80a9 (`git diff main...fix/eval-lora`, 3 files, +165/−9)
- **Reviewer:** reviewer-ml (Board A)
- **Date:** 2026-11-04
- **Verdict:** REVISE
- **Blocking:** yes (one item, F1)

## Summary

The fix works. The old `PeftModel.from_pretrained` path is gone, and dense and LoRA entries now share one
loader (`HFRunner` → `apply_zoo_entry` → `load_entry_delta`). A real `lora_sft` entry changes the logits, and
the merged weights match an independent recomputation of `W + B·A·alpha/r` exactly. The branch's own
real-adapter test, however, skips in every location this branch can reach, so the suite does not test the
behaviour the issue is about. That is the blocker. Everything else is a non-blocking finding.

## Test runs

- `source scripts/env.sh && pytest -q` (worktree `.worktrees/fix/eval-lora`) → **77 passed, 5 deselected**, exit 0.
  Log: `~/.t3/scratch/.../boardA-eval-lora/pytest.log`.
- `pytest -q -m gpu -rs tests/test_eval_lora.py` → **1 skipped**: `no lora_sft smoke adapter under data/zoo`.
  Log: `.../boardA-eval-lora/pytest-gpu.log`.
- Real-adapter check (`.../boardA-eval-lora/gpu_check.py`, log `gpu_check.log`, results `gpu_check_results.json`).
  Uses the branch's `HFRunner` and `apply_zoo_entry` on `Qwen/Qwen3.5-0.8B` (bf16, cached), prompt
  `"The capital of France is"` (5 tokens). Entry: `feat/p1-zoo-build/data/zoo/qwen3.5-0.8b/04ce1ebc4749f1c8`.
  - **Did the real adapter change the logits? Yes.** Base vs edited: `torch.equal` False; max |Δlogit| 0.2656;
    1,018,754 of 1,241,600 logit entries differ. Base forward is deterministic.
  - **Adapted tensors = adapter modules = 72.** 72 A/B pairs in the adapter. Exactly 72 state tensors changed,
    and that set equals the adapter-derived key set and `edit.json` `touched_tensors`.
  - **Math and dtype.** For all 72 tensors, the merged bf16 weight is bitwise equal to
    `(W_base.float() + (B.float() @ A.float()) * 1.0).to(bf16)`, recomputed from the safetensors with no
    branch code. 0 results are non-bf16.
  - **`HFRunner(zoo_entry=...)` path.** Logits equal the manual `apply_zoo_entry` result (max |Δ| 0.0);
    `edit_id` = `04ce1ebc4749f1c8`.
  - **Negative cases.** A layer index 999 raises `KeyError: LoRA module model.language_model.layers.999.mlp.down_proj.weight
    has no matching tensor in the base state`. A `model.visual.*` module key raises `KeyError`. A dense
    `delta.safetensors` with an unknown name raises `ValueError` and leaves all state tensors unchanged.
- Side measurement (`.../boardA-eval-lora/fp32_vs_bf16.py`, log `fp32_vs_bf16.log`): same entry, fp32 model.
  Intended LoRA effect (fp32 merge vs base): mean |Δlogit| 0.0231, max 0.317. bf16 rounding of the merged
  weights (quantized vs fp32 merge): mean 0.0091, max 0.123. Top-1 agreement is 1.0 on these 5 positions,
  which is too few positions to mean much.
- GPU was shared at check time (21.7 of 32.6 GB used, 100% util by another process). Peak use by my runs was
  about 3.2 GB (fp32 model), freed after each run.

## Findings (by severity)

### F1 — High — blocking: the branch's real-adapter test never runs
- **File:** `tests/test_eval_lora.py:51-73` (`test_real_lora_smoke_changes_output`)
- **Evidence:** the test globs `Path(__file__).parents[1] / "data" / "zoo"`. In this worktree `data` is a symlink
  to the canonical `data/`, and the entry exists only in `feat/p1-zoo-build`'s own `data/`. The GPU run
  reports `1 skipped … no lora_sft smoke adapter under data/zoo`.
- **Failure scenario:** the non-GPU tests only call `apply_zoo_entry` on a toy model. None of them goes through
  `HFRunner.__init__` (`src/wsp/eval/run.py:145-146`). Reverting `HFRunner` to `PeftModel.from_pretrained`, or
  deleting the `apply_zoo_entry` call, leaves the default suite green (77 passed). The issue cannot regress
  silently, but the suite cannot show it is fixed either.
- **Fix:** take the entry path from an explicit env var or fixture argument, and fail (not skip) when CUDA is
  present and the path is set but missing. Also add a CPU test that goes through `HFRunner` with a tiny
  generated LoRA adapter, so the loader call site is covered without a GPU.

### F2 — High — non-blocking for this merge; decide before LoRA-edit numbers go in the paper
- **File:** `src/wsp/eval/run.py:74-83` (`apply_delta`): `target.copy_((target.to(delta.dtype) + delta).to(target.dtype))`
- **Evidence:** 86.6% of nonzero LoRA delta entries round to no change in bf16 (`gpu_check_results.json`,
  `delta_entries_absorbed_by_bf16_rounding_frac`). Rounding alone moves logits by mean 0.009, against an intended
  mean effect of 0.023 (`fp32_vs_bf16.log`). The 5-step adapter's delta is near bf16 resolution.
- **Failure scenario:** a LoRA entry evaluated in bf16 reports an effect that is partly rounding error. Nothing
  flags it, and the eval number does not match `W + BA`.
- **Scope:** pre-existing. `main`'s `add_` and the zoo branch's `load_edited_model` both keep the base dtype
  (bf16), so the same rounding applies to every dense delta. This fix does not introduce it. It does mean the
  edit is not faithfully evaluated for any entry.
- **Decision needed (not a review call):** merge in fp32 and evaluate the touched tensors in fp32, or keep bf16
  and record the rounding error per entry in the manifest. The fp32 option costs memory and needs its own
  measurement on the larger models.

### F3 — Medium — non-blocking: `lora_dense_delta` drops things silently
- **File:** `src/wsp/eval/zoo_compat.py` (`lora_dense_delta`). The loop reads only `.lora_A.` keys and the
  config is checked only for `use_dora`.
- **Failure scenario:**
  - An orphan `lora_B` key, `modules_to_save`, `lora_embedding_*` or `lora_magnitude_vector` is skipped with no error.
  - `fan_in_fan_out=True` (Conv1D-style, weights stored transposed) is merged as if it were `nn.Linear`. On a
    square module, `apply_delta`'s shape check passes and the merge is silently wrong.
  - `bias != "none"` (bias updates ignored) is not rejected.
  The docstring says unmappable modules "never silently dropped", which holds only for the key-mapping case.
- **Real adapter:** unaffected. Its config has `bias: none`, `modules_to_save: null`, `fan_in_fan_out: false`.
- **Fix:** raise `NotImplementedError` for these config options and for any adapter tensor that is not
  `lora_A`/`lora_B` with a matching partner. The zoo branch's `apply.py` has the same gap, so fix both at merge.

### F4 — Medium — non-blocking: entry dispatch by file presence, not by `delta_format`
- **File:** `src/wsp/eval/zoo_compat.py` (`load_entry_delta`). The old `HFRunner` branched on
  `edit.json["delta_format"]`. The new code prefers `delta.safetensors` if present, otherwise reads `lora/`.
- **Failure scenario:** a `delta_format: "lora"` entry with a stray `delta.safetensors` is evaluated as dense
  with no warning. A `delta_format: "dense"` entry whose file is missing silently falls through to `lora/` if
  that directory exists.
- **Fix:** dispatch on `delta_format` (passed in from the descriptor) and raise on a mismatch.

### F5 — Medium — non-blocking: eval manifest does not identify the adapter
- **File:** `src/wsp/eval/run.py:350-362`. The manifest records `zoo_entry` (the path) and `model_revision`. It
  does not record the entry's `edit_id`, `delta_format`, or a hash of the adapter file.
- **Failure scenario:** the entry is rebuilt at the same path with different weights. `config_hash` and
  `git_sha` do not change, so two runs with different adapters look reproducible from the manifest.
- **Fix:** record `edit_id` and the sha256 of `lora/adapter_model.safetensors` (or of `edit.json`).

### F6 — Low — non-blocking: suffix fallback can bind outside the text stack
- **File:** `src/wsp/eval/zoo_compat.py` (`resolve_state_key`, suffix branch). When the exact and
  prefix-swapped candidates miss, it accepts any unique state key ending in `.{layer}.{rest}` (excluding only
  names containing `visual`).
- **Failure scenario:** if the text-stack tensor is absent, a unique match elsewhere (for example a
  multi-token-prediction head) is accepted silently. For this model all 72 modules resolved to the expected
  keys, so no harm is visible here.
- **Fix:** restrict suffix hits to the `model.` and `model.language_model.` prefixes.

### F7 — Low — nit: `apply_delta` rebinds `delta` inside the loop over `delta.items()`
- **File:** `src/wsp/eval/run.py:80-83`. Correct only because the iterator is created before the rebinding.
  Rename the inner variable (the zoo copy uses `d`).

### F8 — Low — nit: real adapter does not exercise `alpha/r`
- The real entry has alpha = r = 32, so its scale is 1.0. The `alpha/r` scaling (and the rsLoRA `alpha/sqrt(r)`
  branch) is covered only by the toy test (alpha 4, r 2, scale 2). That is correct, but it is the only
  coverage. Add a `rank_pattern`/`alpha_pattern` case when F1 is fixed.

## Checked, no finding

- **Single loading path.** Dense and LoRA entries both go through `apply_zoo_entry` → `load_entry_delta`. Only
  `HFRunner.__init__` calls it. `PeftModel` is no longer used in the eval path.
- **Key mapping.** The adapter keys `base_model.model.model.language_model.layers.N.*` map to
  `model.layers.N.*.weight` (the CausalLM state dict uses `model.layers`). All 72 map to the expected keys, and
  that set equals `edit.json` `touched_tensors`.
- **Math.** `B @ A` has shape (out, in), matching `weight`. `scale = alpha/r` (or `alpha/sqrt(r)` for rsLoRA),
  with the rank taken from `A.shape[0]`, so `rank_pattern` is handled. The merge runs in fp32 and casts once.
- **Compatibility copy.** `resolve_state_key`, `lora_dense_delta` and `load_entry_delta` in `zoo_compat.py` are
  identical to `feat/p1-zoo-build:src/wsp/zoo/apply.py` (checked with `diff`). The "replace with
  `wsp.zoo.apply` after the zoo merge" marker is accurate. Re-diff at merge time.
- **Validation before write.** `apply_delta` checks every name and shape before any write. A failing dense
  delta leaves the model unchanged (checked).

## Remaining blockers

1. **F1** (blocking): make the real-adapter test run or fail, and add a CPU test through `HFRunner`.
2. **F2** is not a merge blocker, but the fp32-vs-bf16 choice must be made before any LoRA-entry numbers
   enter the paper.

## Verdict

**REVISE.** Fix F1 and re-review. F2–F8 can go in the same revision or follow it.
