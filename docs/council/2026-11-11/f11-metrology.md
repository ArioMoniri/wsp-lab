# F11 resizing amendment — Metrology (single-member round)

Chair's question: how to size the A5.3.2 "resizing amendment" that has been required since the F10.6 slot-masked calibration failed
(ordered power 0.7514 < 0.80 at 31 of 40 rotation slots), before any Part A outcome exists.
Branch `council/f11-resize` from main `4c33fdb`. No GPU. Data read only. No file under `data/evals` was opened.
Own runs wrote only to `/home/ubuntu/wsp-data/runs/a5/f11_sizing/` (`data/runs/a5/f11_sizing/` through the repo symlink).

## BLUF

1. Registered-scale sizing runs give 31 → 0.7514, 34 → 0.7722 (fail), 37 → 0.8251 (pass) and 39 → 0.8453 (pass), against 0.8540 at 40/40. Size passed in every run (null go ≤ 0.0192). Files: `data/runs/a5/power/sizing_slots_{34,37,39}.json`.
2. Power depends on *where* the slots are, not only how many. Per-slot resolution probability from the registered donor cells is about 1.00 for LoRA, 0.91 for chain-high, 0.66 for chain-middle and 0.57 for chain-low (§1.3). A chain-low slot is worth about half a LoRA slot.
3. The ideal curve P(Bin(n, 0.75) ≥ k_min(n)) is a sawtooth (31: 0.771, 33: 0.819, 34: 0.789, 35: 0.858, 36: 0.833, 39: 0.846, 40: 0.897). Random chain resolution smooths it, but it means "more slots" is not automatically "more power".
4. Minimum available rotation slots that reach 0.80, with additions confined to the three cells that lost slots: **36**, and only for LoRA-high or chain-high-heavy allocations (10 of 21). **38** is the smallest count at which every allocation passes (emulator, §1.4).
5. Supplementary candidates exist in all three depleted cells, in frozen next-in-cell order (canonical ID). LoRA-high has 10 unselected roots (7 in the 61-donor pool), chain-low has 5 roots, chain-high has 6. All roots use the base WikiText gradient, which is exact at their predecessor, so the replay failure mode (non-root chain steps) cannot arise.
6. Pass-rate evidence: LoRA 27/30 overall but **7/10 in LoRA-high**, where all three `rounding_bound_failed` occurred. Root chain picks passed 2/2; non-root chain picks passed 2/8. The chain-root pass rate is unknown; I plan for 0.5–1.0.
7. **Option A (recommended): supplementary selection of k = 12, four per depleted cell (LoRA-high, chain-low roots, chain-high roots)**, in next-in-cell order. Fixed k, no top-up, full F10 staging (rotation + two ladders), and one registered-scale masked calibration that must reach ≥ 0.80. If that calibration fails, the pre-registered fallback is B. Gate-pass probability is 0.86–1.00 across the pass-rate grid (§2.3).
8. Cost of k = 12: CPU staging about 1.5–2.6 h, plus one about 22 min calibration. GPU scoring is at most 36 entries = 37.44 min, so 172.64 min in total against the current 135.2. n_D = 0, because all 12 are NULL-SIZE donors with TF-v1 originals on file.
9. Option B (accept "not decisive", score the 31 rotations, ladders and originals as a screen) is legitimate and costs 135.2 GPU-min. It needs explicit text for A5.5's validity row, and it forecloses a rotation go and therefore G7.
10. Main bias risk of A: k and the root filter are audit-yield-informed but outcome-blind, and chain cells shift from chain steps to roots. Board A will want attestations, a frozen ID list before staging, no top-up, a registered-40 sensitivity, and the fallback written in.

## 1. Power versus available rotation slots

### 1.1 Sizing runs (registered scale, 10,000 replicates; chair's runs)

All four runs below used the same code (`scripts/a5_simulation.py`, pinned bytes at `4c33fdb`), the same 61-donor pool
(`docs/prereg/a5_manifests/power/registered_donors.json`), seeds and stream order. Only the frozen slot mask differs.
Each mask unmasks a subset of the 9 F10-excluded ids (masks `/tmp/sizing_{34,37,39}.json`, read only).

| run | rotation slots available (L2, c0, c2) | ordered power [CP95] | gate ≥ 0.80 | emulator | null go | null two-sided (rot, .25, .5) | null alternative | raw sha256 |
|---|---|---|---|---|---|---|---|---|
| F10 registered (`power_slotmask_f10.json`) | 31 (7, 1, 0) | 0.7514 | fail | 0.751 | 0.0192 | 0.0377, 0.0372, 0.0372 | 0.1939 | `b1475682…3450a` |
| `sizing_slots_34.json` | 34 (10, 1, 0) | 0.7722 [0.7639, 0.7804] | fail | 0.775 | 0.0156 | 0.0324, 0.0372, 0.0372 | 0.1939 | `e36a832b…e4da6` |
| `sizing_slots_37.json` | 37 (10, 2, 2) | 0.8251 [0.8175, 0.8325] | pass | 0.825 | 0.0180 | 0.0369, 0.0372, 0.0372 | 0.1939 | `ed039ab5…95adb` |
| `sizing_slots_39.json` | 39 (10, 4, 2) | 0.8453 [0.8381, 0.8523] | pass | 0.845 | 0.0182 | 0.0379, 0.0372, 0.0372 | 0.1939 | `71de730f…ce484` |
| registered 40/40 (`power_registered.json`) | 40 (10, 4, 3) | 0.8540 [0.8469, 0.8609] | pass | 0.854 | 0.018 | — | 0.1939 | `2b4806e5…24522` |

The masks are: 34 restores the three LoRA:2 slots. 37 additionally restores `0027efdf`, `05531ce1` (chain:2) and `057dcfbe` (chain:0). 39 masks only `14cdaabe` (chain:2). Yield (≥ 15 resolved) was 1.0 in every contrast. Size passed in all three runs, and the alternative branch under the null is unchanged at 0.1939 because the ladders keep 40 slots.

**Curve (registered scale):** 31 → 0.7514, 34 → 0.7722, 37 → 0.8251, 39 → 0.8453, 40 → 0.8540. The 0.80 crossing lies between 34 and 37 available slots. The emulator (§1.3) predicts the three sizing runs to within 0.003 after its fixed −0.007 offset, which was set from the 31 and 40 anchors before these runs finished. I use it for the allocations not run at registered scale.

Registered anchors: 40/40 gives 0.8540 [0.8469, 0.8609] (`docs/prereg/a5_manifests/power/report.json`). 31/40 gives 0.7514 and size 0.0192 (`docs/prereg/a5_manifests/part_a/slotmask_calibration_record.json`).

### 1.2 Reduced-replicate runs (this member; 500 replicates; not registered)

`data/runs/a5/f11_sizing/code/f11_grid.py` imports `simulate()` from the pinned `scripts/a5_simulation.py`. It changes only (a) the slot mask and (b) per-cell quotas through the API override, which A5.3.2 reserves for synthetic tests (hence "not registered"). Quotas are raised to model supplementary slots, which `selection_cells` fills by next-in-cell canonical-ID order. The simulated response depends only on the slot's cell, so slot identity matters only for the mask. Outputs: `data/runs/a5/f11_sizing/<name>.json`, logs `grid_{mask,supp}.log`. Each run took 39–70 s. At 500 replicates the Monte Carlo SE is about 0.018 near 0.8, so these runs rank designs; they do not gate.

| run | design | slots (rotation avail. / per contrast) | ordered power [CP95] | emulator | mean resolved rot. | null go | null two-sided (rot, .25, .5) | null alternative |
|---|---|---|---|---|---|---|---|---|
| `m31_registered` | F10 mask (9 masked) | 31 / 40 | 0.734 [0.693, 0.772] | 0.752 | 29.62 | 0.030 | 0.062, 0.052, 0.052 | 0.180 |
| `m34_restore_lora2` | restore LoRA:2 3 | 34 / 40 | 0.770 [0.731, 0.806] | 0.775 | 32.62 | 0.018 | 0.034, 0.052, 0.052 | 0.180 |
| `m34_restore_chain2` | restore chain:2 3 | 34 / 40 | 0.750 [0.710, 0.787] | 0.771 | 32.34 | 0.028 | 0.058, 0.052, 0.052 | 0.180 |
| `m34_restore_chain0` | restore chain:0 3 | 34 / 40 | 0.782 [0.743, 0.817] | 0.764 | 31.39 | 0.026 | 0.046, 0.052, 0.052 | 0.180 |
| `m37_mask_lora2` | mask LoRA:2 3 only | 37 / 40 | 0.794 [0.756, 0.829] | 0.807 | 34.11 | 0.022 | 0.044, 0.052, 0.052 | 0.180 |
| `m37_mask_chain2` | mask chain:2 3 only | 37 / 40 | 0.814 [0.777, 0.847] | 0.812 | 34.39 | 0.024 | 0.042, 0.052, 0.052 | 0.180 |
| `m37_mask_chain0` | mask chain:0 3 only | 37 / 40 | 0.830 [0.794, 0.862] | 0.833 | 35.34 | 0.024 | 0.044, 0.052, 0.052 | 0.180 |
| `m40_none` | no mask | 40 / 40 | 0.858 [0.824, 0.887] | 0.854 | 37.11 | 0.028 | 0.052, 0.052, 0.052 | 0.180 |
| `s_l2p3` | +3 LoRA:2 | 34 / 43 | 0.778 [0.739, 0.814] | 0.775 | 32.68 | 0.028 | 0.040, 0.036, 0.036 | 0.188 |
| `s_l2p6` | +6 LoRA:2 | 37 / 46 | 0.832 [0.796, 0.864] | 0.837 | 35.50 | 0.020 | 0.044, 0.032, 0.032 | 0.194 |
| `s_l2p9` | +9 LoRA:2 | 40 / 49 | 0.860 [0.826, 0.889] | 0.859 | 38.65 | 0.024 | 0.050, 0.048, 0.048 | 0.250 |
| `s_c0p3_c2p3` | +3 chain:0, +3 chain:2 | 37 / 46 | 0.814 [0.777, 0.847] | 0.807 | 34.11 | 0.020 | 0.044, 0.050, 0.050 | 0.200 |
| `s_l2p3_c0p3_c2p3` | +3/+3/+3 (k=9) | 40 / 49 | 0.850 [0.816, 0.880] | 0.854 | 37.13 | 0.024 | 0.050, 0.046, 0.046 | 0.228 |
| `s_l2p3_c0p3_c2p3_fail1each` | +3/+3/+3, one failure per cell | 37 / 49 | 0.804 [0.766, 0.838] | 0.815 | 34.58 | 0.028 | 0.050, 0.046, 0.046 | 0.228 |
| `s_l2p4_c0p4_c2p4` | +4/+4/+4 (k=12) | 43 / 52 | 0.872 [0.840, 0.900] | 0.869 | 39.58 | 0.022 | 0.046, 0.040, 0.040 | 0.260 |
| `s_l2p5_c0p5_c2p6` | +5/+5/+6 (k=16) | 47 / 56 | 0.904 [0.875, 0.928] | 0.904 | 43.00 | 0.026 | 0.042, 0.046, 0.046 | 0.284 |

(The emulator column is offset-corrected, emulator − 0.007, §1.3.) The registered alternative gave 0 alternative decisions in every run.

Under the null, the step-2 alternative rate rises with the number of ladder slots: 0.180 at 40 slots, 0.228 at 49, 0.260 at 52 and 0.284 at 56. Under the registered alternative it stays at 0/500. Because the alternative branch is a funding stop and is not size-gated (F5.7), this is not a size failure. It must still be disclosed beside the go rate under F5.7's reporting rule, and it is one reason to consider C1.

### 1.3 Why cells matter: per-slot resolvability

F5.7 records that donor means dominate within-pair noise. Under the registered alternative, a resolved slot therefore agrees with its prediction with probability about 0.75 (the B draw). What varies by cell is the probability that the slot resolves at all. From the payload (`data/runs/a5/nullsize/inputs.json`), each registered-pool donor has z = |donor_mean| / SE(32 centred block means). For a Gaussian proxy, P(resolve) = 0.75·[Φ(z − 1.96) + Φ(−z − 1.96)] + 0.25·[same]. Result file: `data/runs/a5/f11_sizing/donor_cell_resolvability.json`.

| Cell (registered pool) | donor components | median z | min z | share z < 3 | median abs donor mean (nats) | P(resolved) per slot |
|---|---|---|---|---|---|---|
| LoRA:0 | 1 | 25.4 | 25.4 | 0.00 | 0.0964 | 1.000 |
| LoRA:1 | 9 | 22.0 | 17.3 | 0.00 | 0.0935 | 1.000 |
| LoRA:2 | 15 | 14.0 | 5.66 | 0.00 | 0.0717 | 1.000 |
| chain:0 | 11 | 1.99 | 0.53 | 0.55 | 0.0003 | 0.573 |
| chain:1 | 8 | 6.69 | 0.16 | 0.38 | 0.0022 | 0.659 |
| chain:2 | 14 | 15.4 | 0.54 | 0.14 | 0.0511 | 0.906 |

Check: 30 + 4(0.573) + 3(0.659) + 3(0.906) = 36.99 expected resolved rotation pairs at 40/40, against the 500-replicate mean of 37.11 (`m40_none`). At the F10 mask, 27 + 0.573 + 3(0.659) = 29.55, against 29.62 (`m31_registered`).

A fast emulator (`data/runs/a5/f11_sizing/code/f11_emulator.py`) draws a donor per slot from its cell, B with P = 0.75, and t = B·z + N(0, 1), with resolved iff |t| > 1.96. It then applies go iff ≥ 15 resolved and CP lower > 0.5, and ignores the ladder branch, which fired 0/10,000 times under the registered alternative. It reproduces the registered runs to within 0.007: 0.8601 vs 0.8540 at 40, and 0.7581 vs 0.7514 at 31. I subtract 0.007 for planning. Every planning number below is emulator-based and labelled as such; none is a gate.

### 1.4 Minimum slots and cell sensitivity

The ideal curve (all slots resolve, agreement Bin(n, 0.75)) is in `data/runs/a5/f11_sizing/binomial_curve.json`. Its first passing n are 30 (0.803), 33 (0.819) and 35 onward, with dips at 31, 32 and 34. Under the registered donors (emulator, offset-corrected; `emulator_min_slots.json`), with all additions confined to LoRA:2, chain:0 and chain:2 on top of the 31 F10 slots:

| available rotation slots | allocations | passing ≥ 0.80 | best allocation (L2, c0, c2) | worst |
|---|---|---|---|---|
| 34 | 10 | 0 | (10, 1, 0) 0.773 | (7, 4, 0) 0.760 |
| 35 | 15 | 0 | (11, 1, 0) 0.794 | (7, 5, 0) 0.767 |
| 36 | 21 | 10 | (12, 1, 0) 0.817 | (7, 6, 0) 0.775 |
| 37 | 27 | 23 | (13, 1, 0) 0.838 | (7, 6, 1) 0.791 |
| 38 | 32 | 32 | (14, 1, 0) 0.856 | (8, 6, 1) 0.810 |
| 39 | 36 | 36 | — | (7, 6, 3) 0.827 |

Sensitivity at a fixed 34 slots (paired 500-replicate runs, identical draws because only the mask changes): restoring LoRA:2 gives 0.770, chain:2 0.750, chain:0 0.782. The emulator gives 0.782 / 0.778 / 0.771. All three fail, and the ordering follows §1.3. The differences sit inside the Monte Carlo noise and the sawtooth, so I would not rank them more finely. A LoRA-only fix is the most efficient per slot, but it is cross-cell (§3, C2). The chain-low slots are the least informative: three of them buy about 1.7 resolved pairs.

## 2. Candidate pool for a supplementary procedural selection

### 2.1 Frozen order

A5.3.1 selects `sort(cell, key=canonical_edit_id)[0:quota]` within each (family, tertile) cell. `selection.json:registered.ledger.families.<f>.cells` stores every cell's full representative list. The next-in-cell order is therefore already fixed: `sorted(cell)[quota:]`. No new ordering is needed. Full listing with flags: `data/runs/a5/f11_sizing/candidates.json` (from `selection.json`, `components.jsonl`, `power/fields_manifest.json` and `power/registered_donors.json`).

Losses by cell (`part_a/manifest.json:variants`, rotation `excluded`): **LoRA:2 3/10** (`15a53586`, `2233954b`, `72d3756e`; `rounding_bound_failed`, one entry each, 3,495–12,834 fp32 spacings over the bound, `rotation_audits.jsonl`). **chain:0 3/4** (`057dcfbe`, `0bc0a0c4`, `0d055869`; `replay_failed`). **chain:2 3/3** (`0027efdf`, `05531ce1`, `14cdaabe`; `replay_failed`). chain:1 and LoRA:0–1 lost none.

### 2.2 Unselected candidates in the depleted cells (next-in-cell order)

Root = predecessor base (`components.jsonl:parent_id` null). Exact WikiText gradient = root (G2 base field) or registered chain pick (F7.1). Pool = in the 61-donor list. Donor = has TF-v1 edit and predecessor records (NULL-SIZE donor), so its original need not be scored (A5.3.4 reuse, subject to the reuse proof).

| Cell | pos | id | root | exact grad. | donor / pool | norm | note |
|---|---|---|---|---|---|---|---|
| LoRA:2 | 10 | `8d29ac2adb7fdda0` | yes | yes | yes / yes | 2.659 | |
| LoRA:2 | 11 | `af6d608f8651fb46` | yes | yes | yes / yes | 3.027 | |
| LoRA:2 | 12 | `b5d0ac5d76c3cad6` | yes | yes | yes / yes | 3.899 | |
| LoRA:2 | 13 | `c12f6ec6bad59ba3` | yes | yes | yes / yes | 4.330 | |
| LoRA:2 | 14–19 | `c3ae3e8c`, `d7f933d3`, `d8a9fc56`, `f66b8fbb`, `f8c2cf0c`, `fd69d121` | yes | yes | 4 of 6 donors | 2.21–12.66 | reserve |
| chain:0 | 4, 5 | `21574bfef4d3feed`, `2a62388970807a51` | **no** | no (base-approx.) | yes / no | 0.450, 0.707 | skipped by root rule |
| chain:0 | 6 | `342a1be080049e8f` | yes | yes | yes / yes | 0.277 | |
| chain:0 | 7 | `388347cfb0b7fafa` | yes | yes | yes / yes | 0.359 | |
| chain:0 | 8 | `45cd0be9fea1c139` | yes | yes | yes / yes | 0.361 | |
| chain:0 | 9 | `61d512bffde6513a` | **no** | no | yes / no | 0.668 | skipped |
| chain:0 | 10 | `7bc62b9f723359cb` | yes | yes | yes / yes | 0.676 | bytes via alias `b0b6b9cb18e49442` |
| chain:0 | 11 | `aaf7a0660c6d5a73` | yes | yes | yes / yes | 0.605 | re-trained lineage; 5th root, reserve |
| chain:0 | 12 | `bf0e7fc1c18fa991` | **no** | no | yes / no | 0.525 | skipped |
| chain:2 | 3 | `15cdcf0bb126de17` | yes | yes | yes / yes | 1.460 | |
| chain:2 | 4 | `163c830256b1941d` | **no** | no | yes / no | 1.351 | skipped |
| chain:2 | 5 | `17668df17662b3f7` | yes | yes | yes / yes | 2.341 | |
| chain:2 | 6 | `1ac30f3c7a209fe6` | yes | yes | yes / yes | 2.323 | |
| chain:2 | 7 | `2410c595817dc20b` | yes | yes | yes / yes | 1.953 | |
| chain:2 | 8, 10 | `26ea2129fc9db0f7`, `b943ccc4c39696a6` | yes | yes | yes / yes | 1.619, 1.847 | reserve; `b943…` re-trained |
| chain:2 | 9, 11, 12 | `542d767f`, `d79b4e07`, `fbe56a7d` | **no** | no | — | — | skipped |

Every row is `eligible`, `closure_certified` and `development`. Exactness is `bitwise` except for the two flagged re-trained rows (`components.jsonl`). Root candidates available: LoRA:2 10, chain:0 5, chain:2 6.

Non-root chain candidates would need new F6.2 `chain_exact` gradients (GPU) for exact predictions. Their predecessor arithmetic is also the one that produced `replay_failed`: 6 of the 8 non-root picks failed, and both non-root passes (`00f8111c`, `13b02fe2`) were small edits with 6 tensors each. Roots avoid both problems because the predecessor is the base.

### 2.3 Pass-rate assumptions and k

Measured F10 yield (`part_a/manifest.json`; CP 95 %): LoRA 27/30 [0.735, 0.979]. LoRA:0–1 20/20 [0.832, 1]. **LoRA:2 7/10 [0.348, 0.933]**. Root chain 2/2 [0.158, 1]. Non-root chain 2/8 [0.032, 0.651].

The three LoRA failures all fell in LoRA:2. The chance of that under no tertile effect is C(10,3)/C(30,3) = 0.030, so I use p_L = 0.7 as the planning value and 0.9 as the optimistic one. For chain roots the replay mechanism ("predecessor-zero / base-nonzero entries", PROGRESS 06:30) cannot occur. The only F10 failure seen on roots is the rounding bound, which appeared once per 220 M-entry LoRA edit. Chain roots have 29–38 M entries (the two root picks: 10 and 18 tensors, `rotation_audits.jsonl`) and smaller norms. I therefore expect p_C ≥ 0.9 but grid it at 0.5 / 0.7 / 1.0.

P(gate ≥ 0.80) is computed by binomial enumeration of the realised passes, times emulator power of each realised mask, with offset −0.007 (`emulator_designs.json`). E[power] is shown in brackets.

| k (L2 / c0 / c2) | p_L .7, p_C .5 | .7 / .7 | .7 / 1.0 | .9 / .5 | .9 / .7 | .9 / 1.0 |
|---|---|---|---|---|---|---|
| 6 (6 / 0 / 0), cross-cell | 0.42 [0.802] | 0.42 [0.802] | 0.42 [0.802] | 0.89 [0.826] | 0.89 [0.826] | 0.89 [0.826] |
| 6 (3 / 0 / 3) | 0.23 [0.788] | 0.42 [0.798] | 0.78 [0.814] | 0.40 [0.798] | 0.66 [0.809] | 0.97 [0.826] |
| 9 (3 / 3 / 3) | 0.54 [0.803] | 0.80 [0.820] | 1.00 [0.843] | 0.74 [0.815] | 0.93 [0.831] | 1.00 [0.851] |
| 10 (4 / 3 / 3) | 0.73 [0.817] | 0.90 [0.832] | 1.00 [0.850] | 0.92 [0.831] | 0.98 [0.844] | 1.00 [0.857] |
| 8 (4 / 0 / 4) | 0.60 [0.809] | 0.81 [0.823] | 0.99 [0.843] | 0.83 [0.825] | 0.95 [0.838] | 1.00 [0.853] |
| **12 (4 / 4 / 4)** | **0.86 [0.829]** | **0.97 [0.845]** | **1.00 [0.862]** | **0.97 [0.841]** | **1.00 [0.854]** | **1.00 [0.867]** |
| 13 (5 / 4 / 4) | 0.93 [0.839] | 0.99 [0.852] | 1.00 [0.867] | 0.99 [0.852] | 1.00 [0.861] | 1.00 [0.875] |
| 15 (5 / 5 / 5) | 0.97 [0.847] | 1.00 [0.861] | 1.00 [0.880] | 1.00 [0.858] | 1.00 [0.869] | 1.00 [0.891] |

k = 9 (3/3/3) with every candidate passing restores the registered per-cell rotation allocation exactly (10/10/10, 4/3/3). Its power is then the registered 0.8540, since the rotation response depends only on available slots per cell. But it has no margin for a single failure: one LoRA:2 failure plus one chain-root failure in each chain cell gives 0.804 at 500 replicates (`s_l2p3_c0p3_c2p3_fail1each`; emulator 0.815). k = 12 adds one buffer edit per depleted cell. It keeps P(gate) ≥ 0.86 even at the pessimistic corner (0.7 / 0.5), and ≥ 0.97 at p_C ≥ 0.7. The 500-replicate run `s_l2p4_c0p4_c2p4` (all 12 pass) gives 0.872 [0.840, 0.900] (emulator 0.870).

## 3. Amendment text options

Shared context: A5.3.2 ("otherwise Part A is not decisive and a resizing amendment is required before G3"), F10.6 ("A failure makes Part A not decisive and needs a further amendment before scoring"), Board A F10 condition 5 ("A failed gate … leaves the scoring hold in place pending a new amendment"), condition 3 ("No … replacement edits … is authorized by this gate") and the simulation spec ("do not resample replacements"), and F10a.4 (no retuning; "any other response to a failure is a new, dated, disclosed amendment … reviewed by Board A before any scoring").

### Option A — supplementary procedural selection (recommended)

Draft text:

> **F11.1 Supplement.** The forty registered pairs, their F10 verdicts and their eighty ladders are unchanged and stay in the ledger; the nine F10-excluded rotation pairs stay excluded, with reasons, and are not replaced. Part A adds k = 12 supplementary components, four in each cell that lost a rotation slot at the F10 re-stage (LoRA high, chain low, chain high). Within each such cell they are the next unselected representatives in the frozen A5.3.1 order (`sorted(cell, key=canonical_edit_id)[quota:]`, `selection.json:registered.ledger.families`), restricted to root edits (predecessor = base), whose registered WikiText prediction uses the G2 base field, which is exact at their predecessor (F7.1). The twelve IDs are: LoRA:2 `8d29ac2adb7fdda0`, `af6d608f8651fb46`, `b5d0ac5d76c3cad6`, `c12f6ec6bad59ba3`; chain:0 `342a1be080049e8f`, `388347cfb0b7fafa`, `45cd0be9fea1c139`, `7bc62b9f723359cb`; chain:2 `15cdcf0bb126de17`, `17668df17662b3f7`, `1ac30f3c7a209fe6`, `2410c595817dc20b`. The list and its sha256 are committed before any supplementary staging. k and the list are fixed: no top-up, substitution or reserve use after any supplementary audit is seen.
>
> **F11.2 Staging.** Each supplement is staged once under the pinned F10 code, environment and thread settings (F10.3, F10a), with its registered A5.3.3 seed, a rotation and two ladders (s = 0.25, 0.5) built as registered for roots. Exclusions follow F10.3 without replacement. Valid ladders are retained whatever the rotation verdict.
>
> **F11.3 Calibration.** `scripts/a5_simulation.py` gains a registered supplement option (frozen ID list → cells; quotas = registered + supplement), with tests and a re-pin. One separately labelled registered-scale run (10,000 replicates, 61-donor pool, registered seeds and stream order, NULL-SIZE pass on the full payload) masks every actually excluded rotation slot, the nine plus any supplementary exclusions. Gates: ordered go ≥ 0.80; ordered go and each two-sided size ≤ 0.07. No seed search or rerun.
>
> **F11.4 Fallback.** If F11.3 fails, Option B applies without further amendment.
>
> **F11.5 Reporting.** Ladder and rotation denominators become 52 per contrast. The step-2 alternative's null rate is re-reported (F5.7). The registered-40 subset (31 rotations) is reported as a labelled sensitivity, alongside agreement by root versus non-root and by supplementary versus registered. A5.3.4 is repriced.

- *A5 text changed:* A5.3 "40 components" and A5.3.1 quotas, by addition only. F10.3 "without replacement" is not touched (no replacement). F10.6 / condition 5 allow ladder slot count 40 + k. A5.3.4 entry count.
- *Pre-outcome legitimacy:* yes, under F8a.1(c) attestations. The trigger (audit yield 31/40, power 0.7514) contains no G3 outcome. The order is frozen since 7 Nov. The root rule is metadata-only. It is audit-informed, exactly like F10a.3's "fixed before outcomes, not before audits", and must be disclosed as such.
- *Cost (k = 12):* CPU staging per edit is 7.46 min for three variants (4 h 58 min 33 s / 40, `part_a/README.md`). The F10 rotation is 3.28–5.30 min (2 h 11 min – 3 h 32 min / 40, PROGRESS) or 5.46 min actual (3 h 38 min 28 s / 40, `restage_launch.json`). Total 7.46–12.9 min per edit, so **1.5–2.6 h** for 12. That is an upper bound, because 8 of the 12 are chain roots with 10–18 tensors against LoRA's 72. Calibration: about 11–22 min (slot-mask run 06:25:29–06:36:01; F7 runs 21 min 39 s / 22 min 27 s). Predictions: ≤ 94 s per edit at the F7 field-build rate (`fields_manifest.json:summary.wall_seconds` 13,305.8 / 141). GPU: 3 entries × 1.04 = 3.12 min per edit, **≤ 37.44 min**, with n_D = n_P = 0 since all 12 are donors with base predecessors. Total ≤ 172.64 GPU-min, against 135.2 now.
- *Bias risk:* (i) Composition shift: chain-low and chain-high rotation slots become roots, so the rotation contrast there speaks to first-step chain edits, not continued-chain increments (mitigated by the F11.5 split). (ii) Audit-pass selection correlates with edit properties, as in F10. LoRA-high failures may be norm-linked, and the supplements in LoRA:2 have lower norms (2.66–4.33) than the failures (4.69–6.78). This follows the frozen order, not a choice, but it is a disclosed property. (iii) k was chosen knowing the 31/40 yield and 0.7514 power. That is design information, not outcome information. (iv) D3 originals of the supplements are known, as for the forty (A5.3 *Exposure*), and are used only as simulation donors.
- *Expected Board A conditions:* attestations (F8a.1(c)) for chair, metrology, drafter and stager. Commit the ID list and hash before staging. No top-up or reserve use. Same pins, threads and F10a.4. Simulation code, tests and re-pin, then one registered run. Fallback pre-written. Registered-40 sensitivity and the root split. Alias proof for `7bc62b9f`. Re-hash before the scoring campaign.

### Option B — accept "not decisive" and score as a screen

> **F11-B.** The F10.6 calibration failed (0.7514 < 0.80; size 0.0192). Part A's rotation contrast is **not decisive** (A5.3.2). Scoring proceeds on the F10 state: 31 rotations, 80 ladders and 19 originals (130 entries, 135.2 GPU-min). The A5.5 validity predicate "applicable power … gates pass" is read as not applicable to the rotation contrast. The step-2 ladder alternative keeps its funding-stop role. A rotation interval with lower bound > 0.5 is reported as "Part A rotation: positive screen, not decisive". It is not a Part A go, and it does not count toward G7 ("G3 go and G4 go, never screen", A5.6). Every other outcome is reported as registered.

- *Text changed:* A5.5 validity row and step 3 labelling. No selection, mask or code change.
- *Legitimacy:* clean. The failure is accepted, nothing is retuned, and no edit is added.
- *Cost:* 135.2 GPU-min and no CPU.
- *Bias:* none added. The result is uninformative for funding, because a positive rotation result cannot fund G7.
- *Board A:* will want the "screen" label fixed before scoring, the 0.7514 power stated next to any result, and Part B wording confirmed (Part B is reported but cannot fund without a Part A go).

### Option C — other pre-outcome options considered

- **C1, rotation-only supplement:** A, but supplements enter the rotation contrast only, and the ladders stay at 40 per strength as condition 5 states. CPU cost is the rotation part (≈ 3.3–5.5 min per edit). GPU cost is 1.04 min per edit (12.48 for 12). It needs a ladder-unavailable slot state in the mask schema, which F10.6 says "would need a new ruling". It breaks the A5.3 unit of "one pair per component across contrasts". Rotation power is identical to A. This is a legitimate cheaper variant if Board A prefers to keep the ladders at exactly 40.
- **C2, LoRA-high-only supplement (k = 6):** about 0.83 if all pass, but P(gate) is only 0.42 at p_L = 0.7, and it is cross-cell (A5.3.1 "No cross-cell replacement"). Rejected.
- **C3, uniform quota increase across all six cells, literal order:** audit-blind in cell choice, which is attractive. But the next-in-order candidates in chain:0 and chain:1 are non-roots (`21574bfe`, `2a623889`, `06427a22`, `0b1bb525`). They need GPU `chain_exact` gradients and have shown 6/8 replay failures. Rejected on cost and yield.
- **C4, changing the inference or gate** (one-sided CP, lower 0.80, weaker alternative, more replicates, new seed): post-failure retuning of frozen F7.2 items. Not legitimate.
- **C5, repairing the chain scorer round-trip for the six `replay_failed` picks:** this changes scorer arithmetic after seeing the failures, and needs a new qualification and a Board A ruling under F10a.4. High cost, targets the failed items, and cannot be guaranteed outcome-free in effect. Not recommended.

## 4. Recommendation

**Option A with k = 12 (LoRA:2 + 4, chain:0 + 4 roots, chain:2 + 4 roots), the IDs listed in F11.1, fixed in advance, with B as the pre-registered fallback.** Reasons:

1. It is the smallest cell-faithful design whose gate-pass probability stays ≥ 0.86 across the plausible pass-rate grid (≥ 0.97 if chain roots pass at ≥ 0.7).
2. If everything passes, it restores or exceeds the registered per-cell rotation allocation (≥ 10/10/10, 4/3/3) and sits near the registered design's own 0.854.
3. It adds at most 37.44 GPU-min.

If Board A wants the minimal design, k = 9 (3/3/3) is defensible, but its gate-pass probability is 0.54–0.80 unless chain roots pass at about 1.0. I would not trade 3 edits (≤ 9.4 GPU-min) for that risk. If Board A rejects any supplement, B is the honest reading.

## Provenance

Read: `docs/PREREG_H1_TF.md` (A5.3, A5.3.1–A5.3.4, A5.5, F2, F5.7, F7, F10, F10a), `docs/reviews/council/board-a-f10-2026-11-11.md`, `docs/PROGRESS.md` (06:30, 06:40 entries), `docs/prereg/a5_manifests/{selection.json, components.jsonl}`, `docs/prereg/a5_manifests/power/{fields_manifest.json, registered_donors.json, report.json}`, `docs/prereg/a5_manifests/part_a/{manifest.json, rotation_audits.jsonl, slot_mask_f10.json, slotmask_calibration_record.json, restage_launch.json, README.md}`, `data/runs/a5/nullsize/inputs.json`, `data/runs/a5/power/fields.json`, `data/runs/a5/power/sizing_slots_{34,37,39}.json`. Not opened: anything under `data/evals`.
Written (not committed; outside git): `data/runs/a5/f11_sizing/` → `candidates.json`, `binomial_curve.json`, `donor_cell_resolvability.json`, `emulator_configs.json`, `emulator_designs.json`, `emulator_min_slots.json`, 16 grid outputs `<name>.json`, `grid_{mask,supp}.log`, `code/f11_grid.py`, `code/f11_emulator.py`.
Exposure: this member has seen no G3, rotation, ladder, reference or sealed outcome, nor a summary of one. Simulation inputs are the pre-G3 NULL-SIZE payload and power fields.
