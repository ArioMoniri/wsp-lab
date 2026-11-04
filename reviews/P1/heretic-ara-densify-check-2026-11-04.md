# Heretic ARA densification review — 2026-11-04

**Verdict: fix needed.** The PEFT densifier correctly represents the saved adapters; the extreme delta is in the saved factors, not a densification scaling or orientation error. The entries and associated KL labels need investigation and revalidation before use.

CPU only, CUDA hidden. Copied both entries from Lambda. Base Qwen/Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17` was loaded in bf16 then converted to fp32; adapter factors are fp32. Both configs have r=50, alpha=50 (scale 1), target modules down_proj, o_proj, out_proj, fan_in_fan_out=false, empty rank and alpha patterns, no RSLoRA or DoRA. Each adapter contains 48 B tensors of shape (1024,50), 24 A tensors (50,2048), and 24 A tensors (50,3584); 48 paired modules total, including zero-valued ones.

| Entry | Stored global Frobenius norm | CPU float32 B@A global norm |
|---|---:|---:|
| 770d16bb686c8226 | 623.3 | 623.24 |
| bbf6af9653df23c2 | 1,943,437,066.6 | 1,942,947,524 |

The ~0.025% difference in the second figure is consistent with different matrix-multiply precision/execution; its exact cause was not verified. It does not explain the magnitude. Neither fp16 norm overflow nor an inverted factor order explains these float32 results.

For each entry I compared the three largest module deltas with `PeftModel.from_pretrained(...).merge_and_unload()` on the CPU fp32 model. The merged-minus-base norms match the helper. The absolute Frobenius discrepancy is at most 1.6e-6 for the first entry and 0.83 for the second (below 5e-9 of the corresponding delta norm), consistent with addition/subtraction rounding. Examples:

| Entry and module | Base norm | Helper delta norm | Delta/base |
|---|---:|---:|---:|
| 770d, layer 11 self_attn.o_proj | 16.86 | 250.07 | 14.83 |
| 770d, layer 10 linear_attn.out_proj | 16.90 | 246.46 | 14.59 |
| 770d, layer 10 mlp.down_proj | 15.24 | 244.38 | 16.03 |
| bbf6, layer 14 mlp.down_proj | 15.14 | 1,932,835,328 | 127,640,347 |
| bbf6, layer 14 linear_attn.out_proj | 16.93 | 167,636,928 | 9,902,613 |
| bbf6, layer 10 mlp.down_proj | 15.24 | 102,222,672 | 6,706,368 |

Heretic initializes A in PEFT orientation and optimizes A and B with LBFGS. It creates PEFT adapters with alpha=r, and its export calls PEFT merge_and_unload. Its ARA objective optionally applies row normalization to the *objective outputs* using norms of W+B@A, but does not fold that normalization into saved A/B or PEFT's forward/merge. That discrepancy is a concrete upstream implementation concern; the precise cause of the enormous trained factors was not established. Our generator evaluates each trial, then resets and reruns the chosen parameters before saving, so the recorded trial KL does not establish the behavior of the exact saved factors if optimization varies between runs. A recorded KL of 0.028 is not substantiated by these extreme saved weight changes; weight norms alone cannot prove a particular output KL. No direct forward-pass KL was run in this CPU check.

The densification math is valid for the saved PEFT adapter and features computed from it describe that adapter. They should not be treated as validated features/evaluations of the scored trial or an intended row-normalized ARA edit. Quarantine both entries pending direct saved-adapter forward/KL checks and reproduction. Fix generation to preserve/export the exact evaluated trial and evaluate the final saved/reloaded adapter; investigate/fix Heretic's row-normalization semantics and optimizer instability, then regenerate affected entries and recompute features/evals and metadata. Do not change B@A scaling to hide the large norm.
