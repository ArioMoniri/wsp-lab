# 4B scale diagnostic (F5.3), 2026-11-10

**diagnostic, not a gate; exposure recorded.**

F5.3 of the 10 Nov council synthesis: 20 outcome-blind Gaussian edits of Qwen3.5-4B at a fixed relative
radius plus the shared base, scored on the 64-window WikiText log-PPL endpoint (TF-v1 restricted to its
WikiText stage, `configs/eval_tf_v1_wikitext.yaml`). It is a diagnostic of beneficial-edit density and
effect resolution at 4B; it is never a gate, is not part of G7, and does not touch Part A. If it is
inconclusive, scale expansion stops.

TF-v1 status of the H100 records: **not counted**: A3 pairing fail (cross-device SD ratio 0.662 > 0.1); FP4 identity pass.
FP4's paired half-width bound (<= log(1.05)/2) and reliability bound (>= 0.90) are not computed by this report; it never establishes FP4 qualification on its own.

## Design (fixed before scoring)

- Base `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; kernel `gaussian_relative_frobenius_per_tensor_pcg64_bf16_effective_v2`; radius rho = 0.005 of each touched tensor's Frobenius norm; stored delta is the bf16-effective delta.
- Layers [11, 16, 20, 23, 25, 28] of 32 (0.8B random_dir [8, 12, 15, 17, 19, 21] of 24, scaled by round(l * 32/24)); tensor specs ['mlp.down_proj', 'self_attn.o_proj'] (role mode), 12 tensors.
- 20 draws, seed sha256('20261110:4b-gaussian:' + str(draw index))[:8] big-endian mod 2^63, generator numpy.random.PCG64; design hash `fa4fbdeeb938`; FP4 panel (draw indices 0, 4, 8, 12, 16) `fff24173456d293b`, `531bbf717b19786a`, `0410161ef09ad88e`, `a467154753b880e2`, `bd39344ac30d427d` (FP4: draw indices 0, 4, 8, 12, 16 (expansion once: 1, 5, 9, 13, 17)).

## Result

Base log-PPL (H100): 2.379466 nats. Edits improving over the base (delta < 0): **3 of 20** (fraction 0.15, Clopper-Pearson 95% [0.0321, 0.379]); resolved improving 0, resolved worsening 2, the rest unresolved.

| id | draw | seed | panel | delta log-PPL (nats) | 95% block-bootstrap interval | resolved | realised r (pooled) | realised r min–max | changed elements |
|---|---|---|---|---|---|---|---|---|---|
| `fff24173456d293b` | 0 | 3916065655914066609 | yes | +6.527e-05 | [-2.011e-04, +3.215e-04] | unresolved | 5.1936e-03 | 5.1875e-03–5.1992e-03 | 0.684 |
| `fa34888ba16346cb` | 1 | 8517114996709935346 |  | +1.069e-05 | [-3.281e-04, +3.356e-04] | unresolved | 5.1936e-03 | 5.1872e-03–5.1997e-03 | 0.684 |
| `6e825d517c74f00d` | 2 | 5514526843984119769 |  | -3.395e-05 | [-3.444e-04, +2.499e-04] | unresolved | 5.1936e-03 | 5.1868e-03–5.2003e-03 | 0.684 |
| `eda507238dc5215f` | 3 | 6907741158792257525 |  | +3.372e-04 | [+1.189e-04, +5.721e-04] | worsening | 5.1936e-03 | 5.1874e-03–5.1998e-03 | 0.684 |
| `531bbf717b19786a` | 4 | 6240142443084749801 | yes | +2.468e-06 | [-3.007e-04, +3.117e-04] | unresolved | 5.1936e-03 | 5.1866e-03–5.2008e-03 | 0.684 |
| `cc289e807d895724` | 5 | 8975902875688875423 |  | +1.761e-04 | [-7.422e-05, +4.185e-04] | unresolved | 5.1935e-03 | 5.1874e-03–5.1993e-03 | 0.684 |
| `e264e1d8cbb82d1d` | 6 | 9022206078619920708 |  | -6.919e-05 | [-4.479e-04, +2.795e-04] | unresolved | 5.1937e-03 | 5.1877e-03–5.2005e-03 | 0.684 |
| `c51ebb73162abed9` | 7 | 1143832456716188922 |  | +1.056e-04 | [-1.668e-04, +3.583e-04] | unresolved | 5.1935e-03 | 5.1876e-03–5.1999e-03 | 0.684 |
| `0410161ef09ad88e` | 8 | 8344162535550948333 | yes | +2.368e-04 | [-4.272e-05, +5.328e-04] | unresolved | 5.1933e-03 | 5.1870e-03–5.1993e-03 | 0.684 |
| `21cea7248c7050a0` | 9 | 4039369277826701704 |  | +3.004e-04 | [-4.989e-05, +6.312e-04] | unresolved | 5.1938e-03 | 5.1872e-03–5.2003e-03 | 0.684 |
| `a8f3f19076573c3a` | 10 | 1240834983570332308 |  | +9.739e-05 | [-2.459e-04, +4.402e-04] | unresolved | 5.1934e-03 | 5.1876e-03–5.2003e-03 | 0.684 |
| `020579ae3d9e5e71` | 11 | 3656709543942745210 |  | +3.006e-04 | [-4.084e-05, +6.324e-04] | unresolved | 5.1935e-03 | 5.1880e-03–5.1996e-03 | 0.684 |
| `a467154753b880e2` | 12 | 6657596879467450163 | yes | +4.402e-05 | [-2.414e-04, +3.248e-04] | unresolved | 5.1936e-03 | 5.1869e-03–5.1999e-03 | 0.684 |
| `df0c06cf26998659` | 13 | 8381956151944289091 |  | +8.883e-05 | [-2.411e-04, +4.037e-04] | unresolved | 5.1935e-03 | 5.1870e-03–5.2005e-03 | 0.684 |
| `757cb4c98ec2aa62` | 14 | 8871778435691943367 |  | +2.478e-04 | [-7.904e-05, +5.533e-04] | unresolved | 5.1936e-03 | 5.1875e-03–5.2000e-03 | 0.684 |
| `59dc1ac4349d4b6d` | 15 | 3177895835607504974 |  | +2.316e-04 | [-7.688e-05, +5.494e-04] | unresolved | 5.1937e-03 | 5.1877e-03–5.2000e-03 | 0.684 |
| `bd39344ac30d427d` | 16 | 4348725117917686194 | yes | +2.754e-04 | [-5.304e-05, +6.168e-04] | unresolved | 5.1936e-03 | 5.1869e-03–5.1998e-03 | 0.684 |
| `f091ba26273448b4` | 17 | 7761534084188437654 |  | +3.251e-04 | [+3.443e-05, +6.039e-04] | worsening | 5.1935e-03 | 5.1863e-03–5.1998e-03 | 0.684 |
| `069f42cadb0ba658` | 18 | 3918046388836709123 |  | +1.546e-04 | [-1.358e-04, +4.150e-04] | unresolved | 5.1937e-03 | 5.1871e-03–5.2004e-03 | 0.684 |
| `068836804e2f9096` | 19 | 7638157043054532002 |  | -7.111e-05 | [-3.954e-04, +2.285e-04] | unresolved | 5.1938e-03 | 5.1871e-03–5.2010e-03 | 0.684 |

Realised pooled radius over the 20 entries: 5.1933e-03 to 5.1938e-03 (intended 0.005); bf16 rounding leaves part of each intended delta below half a unit in the last place, so the effective edit is the realised one, not the intended Gaussian. Largest fraction of elements where the applied weight differs from bf16(W + intended): 0.00182.

## A3 device pairing

Status: **fail**. cross-device SD ratio 0.662 > 0.1.

Cross-device (H100 minus H0) on 5 paired entries: centred SD 6.697e-05, between-entry SD 1.012e-04, ratio 0.662 (limit 0.1); mean offset +1.580e-05; base log-PPL difference +3.466e-04, base per-window max |diff| 6.480e-03.
- H100 same-device repeats (q4b-r1, q4b-r2): 12 of 12 bitwise; max |primary - repeat| 0 (FP4 limit 0.0001 nats).
- H0 same-device repeats (q4b-r1, q4b-r2): 12 of 12 bitwise; max |primary - repeat| 0 (FP4 limit 0.0001 nats).

## FP4 identity

Status: **pass**. Effective config hash `79e81e79d46cee9e980476000ab06ab0e585111e79d3363e2661526b96e98971`; window identity `docs/prereg/a5_manifests/text_groups.json`.
- H100: {"cuda": "12.8", "driver": "570.148.08", "dtype": "bfloat16", "eval_device_id": "H3", "gpu_model": "NVIDIA H100 PCIe", "torch": "2.11.0+cu128"}
- H0: {"cuda": "13.2", "driver": "595.112.07", "dtype": "bfloat16", "eval_device_id": "H0", "gpu_model": "NVIDIA RTX PRO 4500 Blackwell Server Edition", "torch": "2.15.0+cu132"}

## H100 timing

Device `H3` (NVIDIA H100 PCIe); model load 9.56 s; scoring wall 0.048 h over 21 evaluations; median 5.34 s per evaluation.
Job window: start 2026-11-10T13:56:26Z launch=2026-11-10T13:56:24Z cap_h=2.0 deadline=2026-11-10T15:56:24Z device=H3, end 2026-11-10T14:06:10Z rc=0. Ledger line: `| 2026-11-10T13:56Z | 2026-11-10T14:06Z | a6508259aa7049fbbdd7609aba484fed | gpu_1x_h100_pcie | 3.29 | 0.16 | 0.54 | F5.3 4B scale diagnostic: build 20 Gaussian edits + TF-v1 WikiText FP4 panel (base+5, repeats q4b-r1/q4b-r2) then 15 more, rev 736ca25d4a74, rc=0 |`.

| evaluation | wall s | WikiText stage s |
|---|---|---|
| `base` | 52.89 | 52.87 |
| `fff24173456d293b` | 5.562 | 5.025 |
| `fa34888ba16346cb` | 10.82 | 10.08 |
| `6e825d517c74f00d` | 5.296 | 5.015 |
| `eda507238dc5215f` | 5.288 | 4.987 |
| `531bbf717b19786a` | 5.336 | 5.051 |
| `cc289e807d895724` | 5.189 | 4.908 |
| `e264e1d8cbb82d1d` | 5.313 | 5.03 |
| `c51ebb73162abed9` | 8.403 | 8.112 |
| `0410161ef09ad88e` | 5.376 | 5.073 |
| `21cea7248c7050a0` | 5.437 | 5.083 |
| `a8f3f19076573c3a` | 5.329 | 5 |
| `020579ae3d9e5e71` | 5.27 | 4.961 |
| `a467154753b880e2` | 5.349 | 4.992 |
| `df0c06cf26998659` | 5.298 | 4.965 |
| `757cb4c98ec2aa62` | 7.688 | 7.341 |
| `59dc1ac4349d4b6d` | 5.244 | 4.946 |
| `bd39344ac30d427d` | 5.406 | 5.091 |
| `f091ba26273448b4` | 5.107 | 4.81 |
| `069f42cadb0ba658` | 5.131 | 4.842 |
| `068836804e2f9096` | 7.592 | 7.28 |

## Exposure

Exposed by this diagnostic: the 64 development WikiText windows (wikitext-2-raw-v1 test at the pinned revision, 1,024-token windows 0–63, the H1w development windows) scored on the Qwen3.5-4B base and 20 Gaussian edits. No sealed item, no A5 Part A edit and no 0.8B outcome is touched. H100 run ids: `20261110-140105-bf90e624`, `20261110-140420-165ae81a`. H0 run ids: `20261110-174315-f90969e5`.
