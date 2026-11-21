# wme-c1 run5: halt at `repair-field` (field:g0 mismatch), 21 Nov 2026

Engineering halt. No unit ran and no unit result was read. The registered state holds one open incident, `field_repair_mismatch` (study log seq 277).

## Sequence (clean checkout of `d3e31c0`; bundle sha256 `c1a3bbb5…`, CODE_SHA `d3e31c0`)

| step | result |
|---|---|
| dead-man pre-checks | l5 and l5m both pass (≥ 3 clean ticks after the policy write, no error event, policy loaded through the watcher's reload path) |
| Phase L parity, 2×H100 (driver 570.148.08) | PASS: every pass-rule check true, including `hf_gen_eq_committed`; hf_gen `7299ac4f…` on GPU0, GPU1 and all 3 repeats; code identity pass; gate 216 passed; instance terminated and confirmed (0.398 h) |
| main instance, 1×H100 (driver 580.105.08) | launched 20:24Z |
| `phasel-main --resume` | pass: inert identity; resume canaries hf_gen, tf and vllm all equal the committed references and the parity instance; code identity and worker globals pass; driver recorded (`same_as_parity` false: informational); evidence committed (`794fdbb`) under the persisted contract `0a8efdac…`; earlier instances reconciled; owner epoch 9; start canary pass |
| `repair-field` (K3) | field:g0 re-collected; **MISMATCH**; g0 quarantined on the worker (C4); `field_repair_mismatch` incident; exit 1 |
| operator | read-only evidence collected; instance terminated and confirmed through the API (0.502 h) |

## The mismatch

Five of the six objective files of group 0 reproduce bitwise: gsm8k_gold_lp, refusal_margin_harmful, refusal_margin_harmless, sycophancy_real and wikitext_nll. All kept-factor files (`kept_sha256`) reproduce bitwise.

One file differs: `mmlupro_gold_lp.g0.pt` has eligible `b5f7b8e4…` and re-collected `45d4e3f7…`.

A group file (`wsp.wm.collect.collect_grouped`) holds the objective's summed `gradients` and the K-FAC `factors` (A, G, count). The kept factors are computed from A and G only. Matching kept files therefore mean the factors reproduced, and the difference is in the mmlupro gradient tensors. This is an inference: the original file was on the run3 instance and no longer exists.

## What is ruled out (no GPU work)

- **Code.** No file on the collection path changed between run3's bank commit `5060e32` and `d3e31c0`: `collect.py`, `panel.py`, `panel2.py`, `scorers.py`, `c1/field.py`, `c1/modelops.py`, `c1/genengine.py`, `hf.py`. The only change in `serve.py` is K1: the HF canary runs in a child after `rt.release()`, so the worker reloads the model before the collection.
- **Driver.** Run3 and this main instance both ran 580.105.08.
- **Model.** The re-collected manifest records theta `61988209…`, which is theta0.
- **Worker log.** No traceback, error or nondeterminism warning.

## Open question for the chair

Two explanations remain:
1. Run-to-run nondeterminism in the mmlupro gold-log-prob backward pass. The field collection does not run under the deterministic-algorithms context.
2. A process-state difference introduced by K1, such as the model reload or a different allocation history.

A second re-collection of g0 on one instance would separate them: equal repeats point to (2) or to an input difference, unequal repeats to (1). That is new GPU work during a halt, and it would write an owner epoch to the study log, so it was not run.

The re-collected g0 manifest (identity, per-objective panel content hashes, timings) is kept outside the repository for that comparison.

## D1 diagnostic result (RULING-wme-c1-bank-halt addendum 2): D2 branch (a)

The run took place on 21 Nov 2026, 21:36–22:00Z.
- **Code and evidence.** Run from commit `68defa6` (clean tree; bundle sha256 `498a5d11…`; the worker's CODE_SHA equals the controller's). Evidence: `wme-c1-diag-g0-2026-11-21-l5d.json`. Driver script sha256 `d5a6b1c4…`, kept outside the repository.
- **Instance and ownership.** One 1×H100 instance (driver 570.148.08). Owner epoch 11.
- **What ran.** Two non-unit `field_repair` collections of g0 (A and B), each quarantined aside on disk afterwards. No unit result and no repair record.
- **Termination and charge.** The instance was terminated, confirmed and reconciled: 0.390 GPU-h-eq.

| file (g0) | A | B | run5 (quarantined) | run3 (eligible) |
|---|---|---|---|---|
| gsm8k_gold_lp | `4deea013` | `4deea013` | `4deea013` | `4deea013` |
| **mmlupro_gold_lp** | **`b5f7b8e4`** | **`703e5654`** | **`45d4e3f7`** | **`b5f7b8e4`** |
| refusal_margin_harmful | `21c74b89` | `21c74b89` | `21c74b89` | `21c74b89` |
| refusal_margin_harmless | `8bcfe92c` | `8bcfe92c` | `8bcfe92c` | `8bcfe92c` |
| sycophancy_real | `5879a153` | `5879a153` | `5879a153` | `5879a153` |
| wikitext_nll | `9ca5a2d5` | `9ca5a2d5` | `9ca5a2d5` | `9ca5a2d5` |

- **Every other hash matches.** All 48 kept-factor files are equal across A, B, run5 and run3. The panel content hashes are equal across A, B and run5. The model-state hash (`61988209…`) is equal across A, B, run5, the worker's theta0 and run3's recorded theta0. The `collect.py` implementation hash is equal across A, B and run5.
- **Tensor level, A against B (mmlupro):**
  - every K-FAC factor (A, G, count) is bitwise equal;
  - the gradients of layer 3 (`self_attn.o_proj`, `mlp.down_proj`) are equal;
  - the layer-2 gradients differ, with `mlp.down_proj` at 248 of 50.3 M elements (relative max difference 3.7e-6) and `linear_attn.out_proj` at 27 k elements;
  - layers 1 and 0 differ in millions of elements, with relative max differences of 1.2e-3 to 4.0e-3.

  The pattern is consistent with a nondeterministic operation in the backward pass, entering at the layer-2 linear-attention block and growing toward layer 0. It shows only on the mmlupro gold-log-prob objective.
- **Finding.** Two collections on one instance differ from each other. Four collections produced three distinct mmlupro files; collection A happened to reproduce run3. This is D2 **(a)**: the collection backward pass is nondeterministic on this stack.
