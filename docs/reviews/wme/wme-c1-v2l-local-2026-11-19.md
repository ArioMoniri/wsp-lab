# wme-c1 §II.31 (v2l) C1(a): local engineering test of the deterministic HF-generate engine, 19 Nov 2026

**Scope.** This is chair ruling `RULING-wme-c1-phasel` step C4.1(3). It ran on the local GPU (RTX PRO 4500 Blackwell,
shared with another job) at $0, with no Lambda. The model was Qwen3.5-0.8B, on exposed `hf_gen` canary items only
(wme-p3h holdout `mmlupro_gen`, file order). No `wme-c1` or `wme-c2-gen` item and no sealed file was touched.

**This is engineering evidence, not the Phase L decision.** The decision is made by the §II.31.1 pass rule on the pinned
H100 stack.

**Evidence.** `wme-c1-v2l-local-2026-11-19.json` (sha256 `6cc5d0d08b2f5fc8ced2069df081abe04aa89797044759c6909df9e341d60a84`).
It was produced by `scripts/wme_c1_v2l_local.py`, with the process capped at 8 GB VRAM
(`torch.cuda.set_per_process_memory_fraction`).

**Settings.**
- Batch 128 (all items fall in one batch), 128 new tokens, greedy, non-thinking.
- One unrecorded warm-up.
- Then 3 deterministic runs (`hf_deterministic`) interleaved with 3 registered-path runs (`hf_registered`, the
  throughput reference).
- Local stack: torch 2.15.0+cu132, transformers 5.23.0. This differs from the pinned lock (torch 2.14.0+cu130,
  transformers 5.22.2).

## Results

| Variant | Kernels | Items | Deterministic: 3 hashes | Registered: 3 hashes | Median s (det / reg) | Ratio | Peak VRAM |
|---|---|---|---|---|---|---|---|
| A (primary) | torch path, as the pinned lock binds it (fla blocked; production binding check passed) | 24 | `cc93a8902de3…` ×3, bitwise | `471c24ba2816…` ×3 | 5.55 / 4.88 | 1.137 | 3.52 GB |
| B | local `fla` chunk / recurrent kernels (production check bypassed for this local variant only; real bindings recorded) | 24 | `cce41f44db09…` ×3, bitwise | `fa8289fe4471…` ×3 | 4.95 / 4.31 | 1.149 | 3.18 GB |

- **Reproducibility.** The deterministic engine is 3/3 bitwise in both variants. A first execution, in a separate
  process, gave the same four hashes.
- **Different tokens from the registered path.** The deterministic and registered paths give different token ids
  (math SDPA vs the default SDPA kernel), as expected. The canary reference is measured under the active engine at the
  Phase L re-run.
- **Throughput.** The deterministic engine was about 14–15 % slower on this GPU. Timings come from a shared GPU
  (another job at about 97 % utilisation), so they are indicative only.
  - If the H100 ratio were similar (r ≈ 1.15), §II.31.2 gives Δ ≈ 0.15 × 0.545 ≈ 0.08 GPU-h. That is hypothetical;
    the Phase L re-run measures r.

## Limits (read before relying on this)

1. **The local registered path was already reproducible.** On this GPU with 0.8B it is 3/3 bitwise too, as in Phase 0.
   So this test cannot show that C1(a) removes the H100 nondeterminism. It shows only that the engine runs, is
   bitwise-stable, records its configuration, and what it costs.
2. **24 of the 64 items, not 64.** With the 8 GB limit, one left-padded prefill batch of the 64 canary items does not
   fit on either kernel path. The longest prompt is 1,574 tokens, and items 24 and 28 exceed 1,000 tokens. Every
   attempt ran out of memory at 8 GB:
   - 64 items on the torch path, twice;
   - 64 items with fla;
   - 32 items on the torch path.

   The first 24 items in file order (longest 549 tokens) fit as one batch. The H100 re-run uses all 64.
3. **No GPU 0 vs GPU 1 comparison.** There is a single local GPU; the comparison runs at the Phase L re-run.
4. **Different 0.8B and stack.** This is not the 9B, and the local stack is not the pinned stack.

## Bug found and fixed by this test

The production check "no fla / causal_conv1d / hub kernel bound" first read `__module__` of the module-level
gated-delta-net names. transformers wraps the chosen implementation with `functools.wraps`, so the names keep the torch
function's module even when `fla` is bound. With the `kernels` package installed, the names are also `Func` modules.

`genengine.resolved_module` now follows the wrapper's closure (`implementation`) and the `Func.forward` closure. Variant
B shows `fla.ops.gated_delta_rule.chunk` and `…fused_recurrent`, and the regression test
`test_kernel_bindings_resolve_the_real_implementation_not_the_wrapper_name` covers it.

On the pinned lock no `fla`, `causal_conv1d` or `kernels` package is installed, so the torch path is bound there either
way.
