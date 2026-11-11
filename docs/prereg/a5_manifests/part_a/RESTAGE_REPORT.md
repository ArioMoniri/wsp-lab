# Part A F10 rotation-only re-stage — report (outcome-free)

Written 11 Nov 2026 (UTC), after the re-stage (02:44–06:22 UTC) and after the slot-masked calibration (06:25–06:36 UTC). It is
for Board A and the chair's acceptance under Board A F10 condition 4 (`docs/reviews/council/board-a-f10-2026-11-11.md`) and
F10 code re-review conditions 2 and 4 (`docs/reviews/council/board-a-f10-code-2-2026-11-11.md`). It covers F10.3–F10.4, F10.6,
F10.7, F10.10 step 5 and F10a in `docs/PREREG_H1_TF.md`. **Nothing has been scored. This report does not release scoring.**

Every number below comes from a committed file, a log, or a re-hash done for this report. Paths are relative to the repository
root unless absolute. The re-hashes and comparisons were run by a read-only script (`/tmp/f10report/analyze.py`, not committed).
It read `docs/prereg/a5_manifests/part_a/*`, `data/runs/a5/part_a/qwen3.5-0.8b/<id>/{*.safetensors,variant.json}`,
`data/zoo/qwen3.5-0.8b/<edit>/edit.json`, `data/runs/a5/nullsize/registered.json` and `data/runs/a5/power/*.json`. Under
`data/evals` it only tested whether paths exist (§7).

The old staging README (`part_a/README.md`, sha256 `2fcbf517…ceada1`, identical to the hash preserved in `restage_freeze.json`)
is kept unchanged. Under F10.7 this file is the re-stage README.

## Summary

| item | result |
|---|---|
| Execution | `7bacec12604f1090168a79a6c6d55d4f548af4ec`, clean (0 dirty files). 02:44:00–06:22:28 UTC. Exit 0. CPU only, 4 threads |
| Pins at execution SHA | 71/71 files in `code_hashes.json` (`74321d76…`) match. They are unchanged at the current HEAD `be02cc3` |
| Freeze vs emitted environment | 9 of 10 fields match. **Mismatch: Torch build-config SHA-256.** Frozen `d1b4e24c…`, emitted `07f0372b…` (§1.4). The freeze's `base_revision` is empty `{}` (§1.3) |
| F10 yield | **31 / 40 rotation pairs pass.** 9 excluded: 6 chain `replay_failed` and 3 lora_sft `rounding_bound_failed` |
| Tensors | 2,534 edited; 1,984 rotated (in passing pairs); 79 failing tensors (76 `replay_failed`, 3 `rounding_bound_failed`). The other 471 tensors of the excluded pairs passed individually |
| Original a533 verdict | 1 / 40 pairs pass (`00f8111cbad8218f`); 535 / 2,534 tensors pass. Recomputed a533 diagnostic agrees on 2,534 / 2,534 tensors |
| Ladders | 80 variants, 96 files (80 deltas + 16 increments). Re-hashed on disk, they are byte-identical to `manifest.json`, `manifest_a533.json` and the freeze. 80 `variant.json` files are identical to the a533 manifest |
| Prior passing pair | `00f8111cbad8218f` → `21c6044695fed38e`: delta `951e8621…` and increment `78ec4689…` are identical on disk and in both manifests |
| Chain-ladder flags | 16 ladder entries carry `realised_increment_ladder: true` (8 non-root chain picks × 2 strengths) |
| Priced count | 111 variants + 19 references = 130 entries × 1.04 min = **135.2 min** |
| F10.6 slot-masked calibration | **FAIL.** Ordered power 0.7514 (CP95 0.7428–0.7598) < 0.80. Size passes (ordered go 0.0192; two-sided 0.0377 / 0.0372 / 0.0372 ≤ 0.07). The NULL-SIZE block is **not** identical to the registered null: go 192 / alt 1939 / unresolved 7869 against 180 / 1939 / 7881. The differences are confined to the masked rotation contrast (§6) |
| Exposure | Nothing scored (`scored: false`). 0 of the 120 variant ids and 0 of the 19 missing originals has an evaluation directory |

## 1. Execution record

### 1.1 Command, cwd, time, SHA

Source: `docs/prereg/a5_manifests/part_a/restage_launch.json` (sha256 `28dcba68…1a59a7e`) and the agent-tmux log
`/home/ubuntu/agent-logs/a5-f10-restage-20261111-024400.log`.

- agent-tmux job `a5-f10-restage`. It ran the launch script `/tmp/a5_f10_restage.sh` (outside the repository; sha256
  `ce25c17c…5410afb2`, mtime 02:44:00 UTC). The script records SHA and dirty count, refuses on a dirty tree, and runs the command
  under `/usr/bin/time -v`.
- cwd: `/home/ubuntu/projects/wsp-lab`.
- Exact command (launch record `command`). It is string-identical to the Board A code re-review command without the
  `agent-tmux a5-f10-restage` prefix, and to the inner command in `restage_freeze.json:command` (both checked by string equality):

`
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m wsp.eval.response_intervention prepare --rotation-only --audit-rule f10 --staged-manifest docs/prereg/a5_manifests/part_a/manifest.json --zoo data/zoo/qwen3.5-0.8b --selection docs/prereg/a5_manifests/selection.json --references docs/prereg/a5_manifests/references.json --out data/runs/a5/part_a --manifest-out docs/prereg/a5_manifests/part_a/manifest.json
`

- UTC start `2026-11-11T02:44:00Z`, end `2026-11-11T06:22:28Z`, `exit_code` 0. The log ends `RESTAGE_EXIT 0`. `time -v`:
  wall 3:38:27; user 39,937.74 s; system 1,257.24 s; 314 % CPU; max RSS 12,758,744 kB. `nproc` 8; load at start 5.18 5.58 7.44.
- Execution SHA `7bacec12604f1090168a79a6c6d55d4f548af4ec`, `dirty_files: 0`. The manifest's `provenance` agrees (`git_sha` =
  `code_sha` = `7bacec1…`, `dirty: false`, `source_hash` `e1ca5194…67d2511`).
- Order: the freeze was written `2026-11-11T02:40:50.775598Z` (`restage_freeze.json:frozen_at_utc`) and committed in `7bacec1`
  (02:41:26 UTC), before the launch at 02:44:00. The regenerated outputs were committed in `848d7a5`.
- `--evals` was not passed and ordinary `prepare` was not used. Mode `rotation_only`, `audit_rule` `f10` (`manifest.json:restage`).

### 1.2 Pinned source hashes at the execution SHA

`git show 7bacec1:docs/prereg/a5_manifests/code_hashes.json` (sha256 `74321d76858274d2…`, the (xviii) value; `git_sha`
`21646f2`, the merged reviewed code `060cc60`). I re-hashed each of its **71** pinned files from `git show 7bacec1:<path>`.
**All 71 match.** All 71 are also byte-identical at the current HEAD `be02cc3`, and so is `code_hashes.json`. Rows named by
re-pin (xviii) and the freeze (`restage_freeze.json:pinned_source_hashes`), all equal:

| file | sha256 (code_hashes at 7bacec1 = re-hash = freeze) |
|---|---|
| `src/wsp/features/rotation.py` | `8ac459934b0b145f62a3f0c81f68980c1f57d0ef7663d16e9ce1f693be376dba` |
| `src/wsp/eval/response_intervention.py` | `694f9f4eab769c731f269f33e1b2382902c505a8958dc0ad719981f00be10433` |
| `scripts/a5_f10_dryrun.py` | `11bbc5f66eff80e951c2c66e8954a048a3023fbde7163defffc6794bd9a531ef` |
| `tests/test_rotation.py` | `7570224a0b4c0dd8e9f1b0aacdaa930e2170d97fa1d1e9c6864bd60d13869325` |
| `tests/test_response_intervention.py` | `c971a909916d7a249e97af5a04e51fed6e57875d3b6236006c951df0f1bf4e14` |
| `scripts/a5_simulation.py` (used in §6; not in the freeze) | `c4f0c55fd8e8c9bc5bb6521539f58db996f23e1a3cb37d99a19403e134d2fdf2` |

Environment pins in `code_hashes.json`: python 3.12.3, torch 2.15.0, numpy 2.5.4, safetensors 0.8.0, transformers 5.22.1,
tokenizers 0.23.3, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, datasets 5.1.0, pyyaml 6.0.3. Model and tokenizer pin:
`Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`.

### 1.3 Frozen inputs and base revision

| input | freeze | re-hash at `7bacec1` | re-hash at HEAD | match |
|---|---|---|---|---|
| `docs/prereg/a5_manifests/selection.json` | `393530365f87f981d5da3ecb2779c89b8e7bf98105b6c45d1824bf290e908c63` | same | same | yes; it also equals `manifest.json:selection_ledger.selection_json_sha256` |
| `docs/prereg/a5_manifests/references.json` | `5b6991f299c1eee049de51bc1e4f00c71c1dfc58114cdf6549fb5d246850564e` | same | same | yes |
| `docs/prereg/a5_manifests/control_seeds.json` | `d357ed51031877f817baeb4204cf93b38c66fdfcfd95bd41803d93d5160c12d5` | same | same | yes |
| `docs/prereg/a5_manifests/text_groups.json` | `f769afcade5a80571daf3a9578b1c03584b46a5abbbba07b32d43a7e9035e7e1` | same | same | yes |
| base revision | **`{}` (empty; not captured by the freeze)** | – | – | **gap** |

Base revision: `restage_freeze.json:base_revision` is an empty object, so the freeze cannot be compared on this field. As
substitute evidence, the `base_revision` of all 40 selected edits (`data/zoo/qwen3.5-0.8b/<edit>/edit.json`) is
`2fc06364715b967f1860aea9cf38778875588b17`. That equals the `code_hashes.json` model pin. The selections, rotation seeds,
selection ledger, rotation variant ids and `staged_zoo` in `manifest.json` all equal those in `manifest_a533.json`.

### 1.4 Numerical environment: emitted vs freeze

Emitted `manifest.json:restage.replay_environment`. All 40 rotation `variant.json:replay_environment` records (31 staged and
9 excluded) are **identical** to it. All 2,534 tensor audits record `replay_threads` 4 (`rotation_audits.jsonl` and
`variant.json:tensor_audits`).

| field | `restage_freeze.json` | emitted (`restage.replay_environment` = every rotation `variant.json`) | match |
|---|---|---|---|
| CPU only | `cpu_only: true`, `CUDA_VISIBLE_DEVICES=""` | `CUDA_VISIBLE_DEVICES=` in the command (not emitted as a field) | yes (command) |
| `OMP_NUM_THREADS` | `"4"` | `"4"` (also `restage.thread_env`) | yes |
| `MKL_NUM_THREADS` | `"4"` | `"4"` | yes |
| Torch intra-op threads, construction | 4 (`torch_intra_op_threads_required`) | `torch_threads` 4 | yes |
| Torch intra-op threads, replay | 4 | `replay_threads` 4 | yes |
| Python | 3.12.3 | 3.12.3 | yes |
| Torch | 2.15.0+cu132 | 2.15.0+cu132 | yes |
| **Torch build-config SHA-256** | `torch_build_config_sha256` **`d1b4e24c0c5af63e8950d12d424ece45f6b125c7355f13a0a92181b58992e569`** | `torch_config_sha256` **`07f0372b5746fc761d40701ab808447f3703337e82f04fcb4846c45e71ba008d`** | **NO** |
| MKL available | true | true | yes |
| CPU model | Intel(R) Xeon(R) 6975P-C | Intel(R) Xeon(R) 6975P-C | yes |
| Machine | x86_64 | x86_64 | yes |
| `HF_HUB_OFFLINE` | `"1"` | `1` in the command (not emitted) | yes (command) |

**Freeze mismatch: Torch build-config SHA-256.** The emitted value is `sha256(torch.__config__.show())`, as defined in
`src/wsp/features/rotation.py:numerical_environment` (pinned `8ac45993…`). For this report I recomputed it in the same
`.venv`, with `CUDA_VISIBLE_DEVICES=` and 4 threads, and got `07f0372b…`, the emitted value. None of these candidate encodings
reproduces the frozen `d1b4e24c…`:

- `show()` plus a trailing newline: `ad474d19…`
- stripped: `b17dc8f1…`
- stripped plus a newline: `0f2eeb06…`
- `torch.__config__.parallel_info()`: `4947a84c…`

The freeze does not record how its value was derived, and the field names differ (`torch_build_config_sha256` against
`torch_config_sha256`). The other evidence available for the arithmetic environment, which bears on this field but does not
resolve it:

- Every other identity field matches, including the Torch version string.
- Replay bit-equality holds for all 2,534 tensors (`replay_bits_equal: true`, `prerounding_replay_bits_equal: true`).
- The prior passing pair's delta and increment are byte-identical to the staging made under code `7f8b453`.

**The chair should rule on this mismatch.** It is not resolved here.

## 2. Yield

Sources: `manifest.json` (`coverage`, `edit_audits`, `excluded_variant_ids`, `selections`) and `rotation_audits.jsonl`
(2,534 rows, sha256 `4ec0928a…5c97f3`).

- **Pairs: 31 pass, 9 excluded** (`coverage.n_rotation_pairs_staged` 31, `n_rotation_pairs_excluded` 9; log line
  `rotation pairs: 31 pass, 9 excluded`). By family: chain 10 edits, 6 excluded; lora_sft 30 edits, 3 excluded.
- **Tensors:** 2,534 edited (`coverage.n_tensors_edited`) and **1,984 rotated** (`n_tensors_rotated`; all tensors of the 31
  passing pairs). 2,455 rows pass individually; 79 fail:
  - `replay_failed` on **76** tensors (failed check `scorer_roundtrip` on all 76; `replay_bits_equal` is true on all 76, so the
    reloaded R_eff bytes equal the replay, and the failure is the scorer-arithmetic round-trip of F10.3(iii), last sentence).
    95 mismatching entries in total.
  - `rounding_bound_failed` on **3** tensors (failed check `rounding_bound`).
  - The other **471** tensors of the 9 excluded pairs pass individually and are excluded with their pair (whole-pair exclusion,
    A5.3.1).
  - No row fails construction or support. No row has `nonfinite_fields`. The bound violations total exactly 3 (the 3 failing
    rows).
- **Pre-audit reasons** (`non_2d`, `non_finite`, `zero_block`, `rank_degenerate`, `not_single_matrix`): none occur.
- **`detail`:** every pair-level exclusion entry has the keys `{reason, tensor}`. On all 2,534 rows `fail_closed_detail` and
  `replay_detail` are `null`. The F10a `detail` field applies to missing or non-finite information, and none occurred: every
  failure is a computed check result, not a fail-closed case.
- The 6 chain exclusions are 6 of the 8 non-root chain picks listed in F10.9. All 6 have `delta_convention`
  `realised_cumulative_from_base`. The other two non-root picks (`00f8111cbad8218f`, `13b02fe2673cc296`) pass, as do the two
  root chain picks. The 3 lora_sft exclusions are all tertile 2.
- Rounding-bound failures (one entry per tensor):

| edit | tensor | bound violations | max excess over the bound (fp32 spacings) | max ratio to the bound | `n_double_rounding` | `n_exceed_half_ulp_w` |
|---|---|---|---|---|---|---|
| `15a53586efc72a50` | `model.layers.11.self_attn.q_proj.weight` | 1 | 3,495.0 | 1.2133 | 27 | 179,493 |
| `2233954b0cc3edc0` | `model.layers.12.mlp.up_proj.weight` | 1 | 10,970.0 | 1.6696 | 35 | 162,907 |
| `72d3756e968e3209` | `model.layers.2.mlp.down_proj.weight` | 1 | 12,834.5 | 1.7834 | 33 | 119,992 |

  These three violations are **not** the inclusive one-spacing equality that Board A ruled on. One entry per tensor exceeds the
  registered bound by 3,495–12,834.5 fp32 spacings (1.21–1.78 × the bound). Across all **passing** rows the maximum recorded
  excess is exactly 1.0 spacing, on 93 tensors (`rounding_bound_passes: true`, inclusive `<=`, as ruled). `docs/PROGRESS.md`
  (06:30 entry) describes the three as "the exact one-fp32-spacing slack bit". The recorded magnitudes do not match that
  wording. The exclusions stand either way (§8). I flag the discrepancy so that the record describes them correctly.

### 2.1 Per pair

| edit | family | tertile | rotation variant | F10 verdict | tensors edited | rotated | excluded tensors | reason codes (tensors) | failed check | `detail` | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0027efdf2eae7405` | chain | 2 | `8506c63c5b858637` | **excluded** | 32 | 0 | 13 | `replay_failed` ×13 | `scorer_roundtrip` | null | 19 scorer round-trip mismatching entries |
| `00f8111cbad8218f` | chain | 1 | `21c6044695fed38e` | **pass** | 6 | 6 | 0 | – | – | – | – |
| `014f8a9ec78db2be` | chain | 1 | `366800d745feac2d` | **pass** | 18 | 18 | 0 | – | – | – | – |
| `05531ce1db4df63f` | chain | 2 | `ffe89cadb99d110c` | **excluded** | 72 | 0 | 24 | `replay_failed` ×24 | `scorer_roundtrip` | null | 30 scorer round-trip mismatching entries |
| `0559051354691a70` | chain | 1 | `d76a454f707156c4` | **pass** | 10 | 10 | 0 | – | – | – | – |
| `057dcfbedea3251f` | chain | 0 | `0bcb2f48c5fcbcd2` | **excluded** | 76 | 0 | 24 | `replay_failed` ×24 | `scorer_roundtrip` | null | 28 scorer round-trip mismatching entries |
| `0712caa1d248514a` | lora_sft | 2 | `2d78171abc3abb58` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `07f9a50b29a7ef5c` | lora_sft | 2 | `477350e0a0491020` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `0a841d4082532e8b` | lora_sft | 0 | `03bb09f5fef50b43` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `0bc0a0c4ad0520e0` | chain | 0 | `7cb4ef4f618014cc` | **excluded** | 72 | 0 | 9 | `replay_failed` ×9 | `scorer_roundtrip` | null | 9 scorer round-trip mismatching entries |
| `0d055869527177b4` | chain | 0 | `716996040438aed6` | **excluded** | 72 | 0 | 1 | `replay_failed` ×1 | `scorer_roundtrip` | null | 1 scorer round-trip mismatching entries |
| `13b02fe2673cc296` | chain | 0 | `ea9913d4ea1851b6` | **pass** | 6 | 6 | 0 | – | – | – | – |
| `14cdaabee246493b` | chain | 2 | `0824ab1a336c0c19` | **excluded** | 10 | 0 | 5 | `replay_failed` ×5 | `scorer_roundtrip` | null | 8 scorer round-trip mismatching entries |
| `15a53586efc72a50` | lora_sft | 2 | `ed51a348b7fd834b` | **excluded** | 72 | 0 | 1 | `rounding_bound_failed` ×1 | `rounding_bound` | null | 1 bound violation; max excess 3495 fp32 spacings |
| `19d9a98cfb2db6e2` | lora_sft | 1 | `b67496b7ac04369b` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `1ab850a41051fe45` | lora_sft | 0 | `6a554d08a2232e60` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `1b2b671e5da5d625` | lora_sft | 0 | `facea4b71fa37dbd` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `1fc51f7164052d50` | lora_sft | 0 | `c02e9c4a5018a292` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `2233954b0cc3edc0` | lora_sft | 2 | `417883272c745e7e` | **excluded** | 72 | 0 | 1 | `rounding_bound_failed` ×1 | `rounding_bound` | null | 1 bound violation; max excess 10970 fp32 spacings |
| `22659cefc904e29f` | lora_sft | 0 | `b61358b573b37a4f` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `2971f30e5276edd9` | lora_sft | 2 | `f951e61f0f1e7da0` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `2d1de332e36d8f00` | lora_sft | 0 | `6c28ad7a3f1f520e` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `2d7f5f78fac4a9ae` | lora_sft | 1 | `8c2822631244cdaf` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `2f61a4086395b719` | lora_sft | 1 | `0ff3a16f693343b1` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `2fb8d03d2547f357` | lora_sft | 1 | `805113b632ee5d3a` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `3499f7dcc5968d96` | lora_sft | 1 | `6b579dbe647afe42` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `34e407b92a902282` | lora_sft | 2 | `6dc4a23fb51b45ca` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `3996bea032c24200` | lora_sft | 1 | `b3b97e6ab82b82cb` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `453d8c08966438e0` | lora_sft | 0 | `281da7d0e1e0e473` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `455b881d5de4e6e0` | lora_sft | 0 | `0ff8adb2d7c5b40e` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `4f918f6bbd264b34` | lora_sft | 1 | `2ad1813877843ad7` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `5abd76a640cc2b9d` | lora_sft | 2 | `777bce482a34a0c1` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `5f80bd482d092938` | lora_sft | 1 | `ce9a93bfa840dfc8` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `61d1f61e25212068` | lora_sft | 0 | `5b6cb30f498e4eb9` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `649bd035e79488e5` | lora_sft | 0 | `0936224384434c66` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `72d3756e968e3209` | lora_sft | 2 | `9ebbe197b37dfdfd` | **excluded** | 72 | 0 | 1 | `rounding_bound_failed` ×1 | `rounding_bound` | null | 1 bound violation; max excess 12834.5 fp32 spacings |
| `7d03d89c0b0299e6` | lora_sft | 1 | `66d9430775129dd1` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `85366c308158ea48` | lora_sft | 2 | `c0b894745c4070bb` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `8cfc1393b97300ab` | lora_sft | 2 | `ef993c4862b8b0b1` | **pass** | 72 | 72 | 0 | – | – | – | – |
| `983c5e13c59e0022` | lora_sft | 1 | `a7751b617767a85c` | **pass** | 72 | 72 | 0 | – | – | – | – |

### 2.2 By family and tertile (selection cell)

`tensors excluded (failing)` counts failing tensors only. All tensors of an excluded pair are unrotated
(`edit_audits.<id>.unrotated_tensors`).

| group | pairs | pass | excluded | tensors edited | tensors rotated | tensors excluded (failing) |
|---|---|---|---|---|---|---|
| cell=chain:0 | 4 | 1 | 3 | 226 | 6 | 34 |
| cell=chain:1 | 3 | 3 | 0 | 34 | 34 | 0 |
| cell=chain:2 | 3 | 0 | 3 | 114 | 0 | 42 |
| cell=lora_sft:0 | 10 | 10 | 0 | 720 | 720 | 0 |
| cell=lora_sft:1 | 10 | 10 | 0 | 720 | 720 | 0 |
| cell=lora_sft:2 | 10 | 7 | 3 | 720 | 504 | 3 |
| family=chain | 10 | 4 | 6 | 374 | 40 | 76 |
| family=lora_sft | 30 | 27 | 3 | 2160 | 1944 | 3 |
| tertile=0 | 14 | 11 | 3 | 946 | 726 | 34 |
| tertile=1 | 13 | 13 | 0 | 754 | 754 | 0 |
| tertile=2 | 13 | 7 | 6 | 834 | 504 | 45 |


The excluded pairs leave 31 rotation slots: 10 + 10 + 7 in lora_sft tertiles 0, 1, 2 and 1 + 3 + 0 in chain tertiles 0, 1, 2.
These equal the slot-masked calibration's per-cell `available.rotation` counts (§6).

## 3. Original a533 verdict and F10 diagnostics per pair

Columns:

- **a533 pair verdict and failing tensors (preserved):** from `manifest_a533.json:edit_audits.<id>` (`pair_verdict`,
  `edit_audit.n_failing / n_tensors`).
- **a533 diagnostic, recomputed:** the re-stage's own `edit_audit.a533_diagnostic.failing_tensors`. It equals the preserved
  count for all 40 pairs. Per tensor, the re-stage's `a533_passes` agrees with the preserved `rotation_audits_a533.jsonl` on
  2,534 / 2,534 rows (same (edit, tensor) key set). The original a533 result is **1 / 40 pairs and 535 / 2,534 tensors**
  (`rotation_audits_a533.jsonl`: 535 `pass`, 1,999 `audit_failed`). It is a logged diagnostic, not an exclusion reason (F10.3).
- **F10.4 pooled diagnostics** from `manifest.json:edit_audits.<id>.edit_audit`, as defined in its `aggregation` block:
  - `pooled_norm_mismatch_signed` = (pooled ‖R_eff‖ − pooled ‖D‖)/pooled ‖D‖, the F10.3(v) guard quantity.
  - `log_energy_ratio` = ln Σ‖R_eff‖²/Σ‖D‖².
  - κ = pooled ‖D‖/√Σ ulp_bf16(W)².
  - ε̂ = min(1, 1/(√12 κ)).
  - ε_D for roots only. For the 8 non-root chain picks it is `unavailable_non_root`.
  - `noise-calibrated`: every tensor passes the noise-calibrated audit (F10.5(e)).
  - 30-subset (`in_subset_pooled_norm_2e-2`) and 34-subset (`in_subset_noise_calibrated`).

| edit | family:tertile | a533 pair verdict (preserved) | a533 failing tensors / edited (preserved) | a533 diagnostic, recomputed (failing) | F10 verdict | pooled norm mismatch (signed) | log energy ratio ln ρ | κ (pooled) | ε̂ (pooled) | ε_D (pooled) | noise-calibrated | 30-subset | 34-subset |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `0027efdf2eae7405` | chain:2 | excluded | 32 / 32 | 32 | **excluded** | +1.4117e-03 | +2.8214e-03 | 4.346 | 0.06643 | n/a (unavailable_non_root) | True | True | True |
| `00f8111cbad8218f` | chain:1 | pass | 0 / 6 | 0 | **pass** | +2.6028e-03 | +5.1987e-03 | 3.964 | 0.07282 | n/a (unavailable_non_root) | False | True | False |
| `014f8a9ec78db2be` | chain:1 | excluded | 18 / 18 | 18 | **pass** | +7.4668e-03 | +1.4878e-02 | 1.916 | 0.1507 | 0.0002317 | True | True | True |
| `05531ce1db4df63f` | chain:2 | excluded | 15 / 72 | 15 | **excluded** | +7.2433e-03 | +1.4434e-02 | 2.233 | 0.1293 | n/a (unavailable_non_root) | True | True | True |
| `0559051354691a70` | chain:1 | excluded | 9 / 10 | 9 | **pass** | +2.4717e-03 | +4.9374e-03 | 4.001 | 0.07215 | 0.0002768 | True | True | True |
| `057dcfbedea3251f` | chain:0 | excluded | 69 / 76 | 69 | **excluded** | +5.0648e-03 | +1.0104e-02 | 0.8583 | 0.3363 | n/a (unavailable_non_root) | True | True | True |
| `0712caa1d248514a` | lora_sft:2 | excluded | 27 / 72 | 27 | **pass** | +5.0226e-03 | +1.0020e-02 | 2.824 | 0.1022 | 0.1017 | True | True | True |
| `07f9a50b29a7ef5c` | lora_sft:2 | excluded | 23 / 72 | 23 | **pass** | +2.9773e-03 | +5.9457e-03 | 3.696 | 0.07811 | 0.07786 | True | True | True |
| `0a841d4082532e8b` | lora_sft:0 | excluded | 72 / 72 | 72 | **pass** | +2.9225e-02 | +5.7612e-02 | 0.7499 | 0.3849 | 0.331 | False | False | False |
| `0bc0a0c4ad0520e0` | chain:0 | excluded | 72 / 72 | 72 | **excluded** | +2.9649e-02 | +5.8436e-02 | 0.7369 | 0.3918 | n/a (unavailable_non_root) | False | False | False |
| `0d055869527177b4` | chain:0 | excluded | 72 / 72 | 72 | **excluded** | +2.6955e-02 | +5.3196e-02 | 0.7471 | 0.3864 | n/a (unavailable_non_root) | True | False | True |
| `13b02fe2673cc296` | chain:0 | excluded | 6 / 6 | 6 | **pass** | +9.8315e-03 | +1.9567e-02 | 1.678 | 0.172 | n/a (unavailable_non_root) | True | True | True |
| `14cdaabee246493b` | chain:2 | excluded | 8 / 10 | 8 | **excluded** | +2.4454e-03 | +4.8848e-03 | 4.031 | 0.07161 | n/a (unavailable_non_root) | True | True | True |
| `15a53586efc72a50` | lora_sft:2 | excluded | 44 / 72 | 44 | **excluded** | +6.9257e-04 | +1.3847e-03 | 7.743 | 0.03728 | 0.03726 | True | True | True |
| `19d9a98cfb2db6e2` | lora_sft:1 | excluded | 71 / 72 | 71 | **pass** | +1.2979e-02 | +2.5792e-02 | 1.495 | 0.1931 | 0.1826 | True | True | True |
| `1ab850a41051fe45` | lora_sft:0 | excluded | 72 / 72 | 72 | **pass** | +2.7327e-02 | +5.3921e-02 | 0.8126 | 0.3553 | 0.3092 | False | False | False |
| `1b2b671e5da5d625` | lora_sft:0 | excluded | 72 / 72 | 72 | **pass** | +1.9942e-02 | +3.9492e-02 | 1.115 | 0.259 | 0.2382 | True | True | True |
| `1fc51f7164052d50` | lora_sft:0 | excluded | 72 / 72 | 72 | **pass** | +2.5506e-02 | +5.0371e-02 | 0.7688 | 0.3755 | 0.3172 | True | False | True |
| `2233954b0cc3edc0` | lora_sft:2 | excluded | 22 / 72 | 22 | **excluded** | +1.4398e-03 | +2.8776e-03 | 5.362 | 0.05383 | 0.05374 | True | True | True |
| `22659cefc904e29f` | lora_sft:0 | excluded | 72 / 72 | 72 | **pass** | +1.7667e-02 | +3.5026e-02 | 1.177 | 0.2452 | 0.2247 | True | True | True |
| `2971f30e5276edd9` | lora_sft:2 | excluded | 23 / 72 | 23 | **pass** | +1.7890e-03 | +3.5748e-03 | 4.799 | 0.06015 | 0.06005 | True | True | True |
| `2d1de332e36d8f00` | lora_sft:0 | excluded | 72 / 72 | 72 | **pass** | +2.0080e-02 | +3.9762e-02 | 1.174 | 0.2459 | 0.2306 | True | False | True |
| `2d7f5f78fac4a9ae` | lora_sft:1 | excluded | 58 / 72 | 58 | **pass** | +1.3050e-02 | +2.5932e-02 | 1.639 | 0.1761 | 0.172 | True | True | True |
| `2f61a4086395b719` | lora_sft:1 | excluded | 72 / 72 | 72 | **pass** | +1.4458e-02 | +2.8708e-02 | 1.491 | 0.1937 | 0.187 | True | True | True |
| `2fb8d03d2547f357` | lora_sft:1 | excluded | 72 / 72 | 72 | **pass** | +1.7463e-02 | +3.4624e-02 | 1.308 | 0.2206 | 0.2101 | True | True | True |
| `3499f7dcc5968d96` | lora_sft:1 | excluded | 72 / 72 | 72 | **pass** | +1.7798e-02 | +3.5284e-02 | 1.298 | 0.2224 | 0.2116 | True | True | True |
| `34e407b92a902282` | lora_sft:2 | excluded | 60 / 72 | 60 | **pass** | +1.1307e-03 | +2.2601e-03 | 6.016 | 0.04799 | 0.04786 | True | True | True |
| `3996bea032c24200` | lora_sft:1 | excluded | 49 / 72 | 49 | **pass** | +8.6642e-03 | +1.7254e-02 | 2.028 | 0.1423 | 0.1392 | True | True | True |
| `453d8c08966438e0` | lora_sft:0 | excluded | 72 / 72 | 72 | **pass** | +2.0101e-02 | +3.9804e-02 | 1.028 | 0.2808 | 0.2503 | True | False | True |
| `455b881d5de4e6e0` | lora_sft:0 | excluded | 72 / 72 | 72 | **pass** | +2.2991e-02 | +4.5461e-02 | 0.9365 | 0.3082 | 0.2726 | True | False | True |
| `4f918f6bbd264b34` | lora_sft:1 | excluded | 72 / 72 | 72 | **pass** | +1.6628e-02 | +3.2983e-02 | 1.304 | 0.2213 | 0.2081 | True | True | True |
| `5abd76a640cc2b9d` | lora_sft:2 | excluded | 10 / 72 | 10 | **pass** | +2.3489e-03 | +4.6923e-03 | 4.189 | 0.06892 | 0.06875 | True | True | True |
| `5f80bd482d092938` | lora_sft:1 | excluded | 47 / 72 | 47 | **pass** | +7.1497e-03 | +1.4249e-02 | 2.216 | 0.1303 | 0.1272 | True | True | True |
| `61d1f61e25212068` | lora_sft:0 | excluded | 72 / 72 | 72 | **pass** | +3.0921e-02 | +6.0906e-02 | 0.6739 | 0.4284 | 0.3599 | False | False | False |
| `649bd035e79488e5` | lora_sft:0 | excluded | 72 / 72 | 72 | **pass** | +2.7080e-02 | +5.3440e-02 | 0.8314 | 0.3472 | 0.3053 | False | False | False |
| `72d3756e968e3209` | lora_sft:2 | excluded | 57 / 72 | 57 | **excluded** | +1.1445e-03 | +2.2877e-03 | 5.966 | 0.04839 | 0.04827 | True | True | True |
| `7d03d89c0b0299e6` | lora_sft:1 | excluded | 46 / 72 | 46 | **pass** | +8.2501e-03 | +1.6433e-02 | 2.148 | 0.1344 | 0.1329 | True | True | True |
| `85366c308158ea48` | lora_sft:2 | excluded | 38 / 72 | 38 | **pass** | +5.5809e-03 | +1.1131e-02 | 2.667 | 0.1082 | 0.1076 | True | True | True |
| `8cfc1393b97300ab` | lora_sft:2 | excluded | 43 / 72 | 43 | **pass** | +1.1840e-03 | +2.3667e-03 | 5.909 | 0.04886 | 0.04879 | True | True | True |
| `983c5e13c59e0022` | lora_sft:1 | excluded | 72 / 72 | 72 | **pass** | +1.8136e-02 | +3.5946e-02 | 1.285 | 0.2247 | 0.2142 | True | True | True |


Checks on these columns:

- **Subsets.** The 30-subset (`manifest.json:f10_subsets.pooled_norm_le_2e-2`) is exactly the 40 ids minus the 10 that F10.5(c)
  lists. The 34-subset (`f10_subsets.noise_calibrated`) is exactly the 40 minus the 6 that F10.5(e) lists. Both subsets were
  fixed from staged values and **intersect the F10 passing set in 23 and 26 pairs**.
- **Guard.** F10.3(v) passes on all 40 pairs. The maximum |pooled mismatch| is 0.030921 (`61d1f61e25212068`), against 0.5;
  F10.3(v) quotes the staged maximum as 0.0309.
- **Construction maxima over all 40 pairs** (F10.3(i), tolerance 1e-5):
  - orthogonality 8.70e-7
  - construction residual 4.96e-7
  - pre-rounding norm 1.43e-7
  - pre-rounding spectrum 1.68e-7
  - fp64 reconstruction 1.01e-14

## 4. Preserved evidence

### 4.1 Old manifest and audits

| file | sha256 (re-hash) | equals |
|---|---|---|
| `part_a/manifest_a533.json` | `d7785b35de1d7041edd2db139591562379db12e33c2532921c814735e8dde8d6` | `restage_freeze.json:preserved_before_restage.manifest_sha256`, `manifest.json:restage.preserved`, re-pin (xviii) |
| `part_a/rotation_audits_a533.jsonl` | `f05fec4265ffd93c8c537100993d23262be95141369240fcad0a7e16ff474d35` | freeze `rotation_audits_sha256`, `manifest.json:restage.preserved` |
| `part_a/README.md` (old staging README, unchanged) | `2fcbf517ccf26f2a1fe325ef31300d291bc0f254c7faaacd18154e327bceada1` | freeze `readme_sha256` |

### 4.2 The 80 ladder variants: re-hashed on disk

I re-hashed every `.safetensors` file of the 80 ladder variants under `data/runs/a5/part_a/qwen3.5-0.8b/<id>/`: 80
`delta.safetensors` and 16 `increment.safetensors` (the chain-pick ladders), **96 files**. Each was compared with four
references:

- `manifest.json:delta_sha256`
- `manifest_a533.json:delta_sha256`
- `restage_freeze.json:ladder_variant_hashes`
- the file on disk

**All 96 are byte-identical across all four.** No ladder directory holds a `.safetensors` file outside these. The freeze map has
one more id than the 80 ladders: `21c6044695fed38e`, the prior passing rotation (§4.3).

The 80 ladder `variant.json` files re-hash to `manifest_a533.json:variant_json_sha256`, `manifest.json:variant_json_sha256` and
`manifest.json:restage.ladders_rehashed.by_variant.<id>.variant_json` alike. The re-stage's own record agrees:
`restage.ladders_rehashed` reports `n` 80 and `all_equal_staged: true`, and each of its 80 entries has `equal_staged: true`.

### 4.3 Prior passing pair `00f8111cbad8218f` (rotation variant `21c6044695fed38e`)

| file | on disk (re-hash) | `manifest.json` | `manifest_a533.json` |
|---|---|---|---|
| `delta.safetensors` | `951e8621f34b7fce8a9fc48884b91bd90806acfd15802afa5c27786a5b4d197d` | same | same |
| `increment.safetensors` (chain increment) | `78ec4689a5110b7054df3e431cb096df613fb523dc8ab0c9d41766628f2fbe94` | same | same |

The re-stage's own record `restage.staged_pass_bitwise.21c6044695fed38e` has `equal: true`, and its `staged` and `restaged`
hashes are the same as above (F10.7). I also re-hashed the delta and increment files of the 30 other passing rotation variants
on disk. They equal their `manifest.json:delta_sha256` entries.

The 9 excluded rotation variants keep only `variant.json` on disk. Their delta and edit files were removed (whole-pair
exclusion), and they have no `delta_sha256` entry. `manifest.json:delta_sha256` has 111 entries (80 ladders + 31 rotations);
`manifest_a533.json` had 81.

### 4.4 The 16 realised-increment chain ladders (F10.9 / F10a.2)

`manifest.json:variants` has 80 `scale` entries. Exactly **16 carry `realised_increment_ladder: true`**, 8 at s = 0.25 and 8 at
s = 0.5:

`3177a34add5494f5`, `3701979a52879081`, `4c96c43a37f1ddb6`, `69440966f51b6863`, `6f1f66e1a6d1a383`, `71b8d911e58cdd05`,
`7b360fbf0494745b`, `8e307a2ad1f7eaf6`, `9c4639b034dfa9df`, `a27ba8a3c75802ce`, `ac5e9257cb8027f4`, `b60e5a17c81873ca`,
`c7efa4adaf790751`, `e442eff80f1b4316`, `ee83a8f491ee4ff5`, `efe63f9a93cb3ac5`.

The other 64 carry `false`. The same 16 are listed in `manifest.json:realised_increment_ladder_variant_ids`. Their
`source_id`s are exactly the 8 non-root chain picks of F10.9: `0027efdf2eae7405`, `00f8111cbad8218f`, `05531ce1db4df63f`,
`057dcfbedea3251f`, `0bc0a0c4ad0520e0`, `0d055869527177b4`, `13b02fe2673cc296`, `14cdaabee246493b`.

The flag sits in the regenerated manifest, not in the byte-identical staged `variant.json` files. Those carry
`scale_source: "realised_increment"`, and the set with that value is the same 16 variants (64 carry `raw_stored_delta`).
Compared with `manifest_a533.json:variants`, each of the 80 ladder entries differs **only** in the added
`realised_increment_ladder` key.

These 16 ladders are flagged ladders whose slots stay in the denominators. Six of their 8 source edits have an excluded rotation
pair (§2). The rotation exclusion does not remove ladder slots (F10.6).

### 4.5 Reference proofs and missing-reference lists

`manifest.json:scoring.carried_from_staged` lists 13 fields:

- `config`, `evals_dir`, `predecessors`, `originals`, `predecessor_bytes_holders`
- `missing_predecessors`, `missing_originals`, `n_P`, `n_D`
- `references_to_score`, `minutes_per_entry`, `reuse_proofs`, `window_check`

**Each is equal to the same field of `manifest_a533.json:scoring`**: 49 reuse proofs, 19 missing originals, 0 missing
predecessors. `reference_ids_needed` is equal too. Only `plan`, `n_variants`, `excluded_variants`, `n_priced` and `minutes` were
recomputed (`scoring.recomputed`). The re-stage states
`restage.evaluation_reads: "none: no evaluation record opened, no scorer run; reference status and reuse proofs carried from
manifest_a533.json"`.

Carrying the proofs preserves the earlier certification. It does not newly verify the evaluation contents (Board A code
re-review condition 2). This report did not open any evaluation record either (§7).

## 5. Priced count and scoring command

`manifest.json:scoring`:

- `n_variants` 111 (31 rotations + 80 ladders; `plan` has 111 entries)
- `n_D` 19, `n_P` 0, `references_to_score` 19
- `n_priced` 130, `minutes_per_entry` 1.04, **`minutes` 135.2**

130 × 1.04 = 135.2 GPU-min. For comparison, the a533 staging was 81 + 19 = 100 entries, 104.00 min
(`manifest_a533.json:scoring`). The registered 40/40 price was 139 entries, 144.56 min (F10.10 step 7). The 19 missing originals
are all lora_sft:

`1ab850a41051fe45`, `1b2b671e5da5d625`, `1fc51f7164052d50`, `22659cefc904e29f`, `2971f30e5276edd9`, `2d1de332e36d8f00`,
`2d7f5f78fac4a9ae`, `2fb8d03d2547f357`, `3499f7dcc5968d96`, `3996bea032c24200`, `453d8c08966438e0`, `455b881d5de4e6e0`,
`4f918f6bbd264b34`, `61d1f61e25212068`, `649bd035e79488e5`, `72d3756e968e3209`, `7d03d89c0b0299e6`, `85366c308158ea48`,
`983c5e13c59e0022`.

Scoring command, copied from the staging README (`part_a/README.md`, "Scoring later"). It is **not run and not authorised**. It
runs on the exclusive GPU, through agent-tmux, from the repository root, after G3 scoring is authorised:

```
agent-tmux a5-part-a-score env PYTHONPATH=src .venv/bin/python -m wsp.eval.response_intervention score \
  --stage data/runs/a5/part_a/qwen3.5-0.8b \
  --manifest docs/prereg/a5_manifests/part_a/manifest.json --config configs/eval_tf_v1.yaml
```

The README's counts beside that command (81 variants, 100 entries, 104.00 min) describe the a533 staging. Under F10 the same
command reads the regenerated `manifest.json`: 111 variants, 130 entries, 135.2 min. F10.7 requires the staged bytes to be
re-hashed against `manifest.json:delta_sha256` immediately before the campaign.

**Status: on hold.** The F10.6 calibration below failed. The chair has adopted the supplementary selection F11 (Option A;
`docs/PROGRESS.md`, 07:20 UTC entry). That would change this count, so 135.2 min is the count for the 40 registered selections
only.

## 6. Slot-masked calibration (F10.6; Board A F10 condition 5)

**Mask.** The mapping was frozen at re-stage commit `848d7a5` as `part_a/slot_mask_f10.json` (sha256 `a2ed6ae7…a2fe80`).

- **First attempt failed at input validation.** agent-tmux `a5-slotmask-f10`, HEAD `848d7a5`, 06:24:17–06:24:18 UTC; log
  `/home/ubuntu/agent-logs/a5-slotmask-f10-20261111-062417.log` ends `SLOTMASK_EXIT 1`. The error was
  `ValueError: slot mask has unknown key(s): ['basis', 'masked_edit_ids', 'n_rotation_masked', 'source_manifest_sha256']`. The
  frozen `a5-slot-mask-v1` format in `scripts/a5_simulation.py` is "a JSON object with exactly two keys". The first file carried
  four provenance keys besides `schema` and `available`, so nothing was simulated.
- **Rewritten mask (used).** Commit `cf3eae8` rewrote the mask to the two-key schema: `part_a/slot_mask_f10.json`, sha256
  **`86fa27a5c1838d0c0bd88ef530feba22eceafac698d0ec4506fbf05df83b698e`**. The provenance moved to
  `part_a/slot_mask_f10.provenance.json` (sha256 `0bb56b4b…e08636`).
- **The mapping is unchanged.** The canonical mapping sha256 that I computed from the first file's `available` (the simulator's
  `mapping_sha256` definition) is `b7724a69315d8c8692a128504126251ac16945cf7b741ec60b3cdb6daa5cb329`. The run reports the same
  `slot_mask.mapping_sha256`.
- **Masked rotation slots (9):** `0027efdf2eae7405`, `05531ce1db4df63f`, `057dcfbedea3251f`, `0bc0a0c4ad0520e0`,
  `0d055869527177b4`, `14cdaabee246493b`, `15a53586efc72a50`, `2233954b0cc3edc0`, `72d3756e968e3209`. These are exactly the
  9 F10-excluded pairs. No ladder slot is masked: 40 edits list both ladders, and 31 list `rotation`. The mask's
  `source_manifest_sha256` equals the regenerated `manifest.json` (`59d839e3…`).
- **Ordering caveat.** The rewrite and the rerun came after the first attempt had been rejected. The mapping content (the frozen
  object) is unchanged, but the file hash entered for the calibration differs from the hash committed with the re-stage.

**Registered-scale run (the one counted).**

- agent-tmux `a5-slotmask-f10-run2`; log `/home/ubuntu/agent-logs/a5-slotmask-f10-run2-20261111-062529.log`.
- HEAD `cf3eae8cfe468c8194881f19f4b0e7e284bd8dcd`, `dirty 0`. 06:25:29–06:36:01 UTC (wall 10:31.91), exit 0, `SLOTMASK_EXIT 0`.
- cwd `/home/ubuntu/projects/wsp-lab`. Command:

```
env PYTHONPATH=src .venv/bin/python scripts/a5_simulation.py data/runs/a5/nullsize/inputs.json --fields data/runs/a5/power/fields.json --donor-ids docs/prereg/a5_manifests/power/registered_donors.json --slot-mask docs/prereg/a5_manifests/part_a/slot_mask_f10.json --output data/runs/a5/power/power_slotmask_f10.json
```

Input hashes. The output's `input_provenance` equals my re-hash in each case:

| input | sha256 |
|---|---|
| `data/runs/a5/nullsize/inputs.json` | `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de` |
| `data/runs/a5/power/fields.json` | `85843ed28d3fd3ba5dc71e2ba8943fd022924cbc4f0b62dddbfb001cb2bcf3b9` |
| `docs/prereg/a5_manifests/power/registered_donors.json` | `6e77bca615624a04cb43b3d11a035b5dc31e5a2f1cd918d42a35cc7786babbb5` |
| `docs/prereg/a5_manifests/part_a/slot_mask_f10.json` | `86fa27a5c1838d0c0bd88ef530feba22eceafac698d0ec4506fbf05df83b698e` |
| `scripts/a5_simulation.py` at `cf3eae8` | `c4f0c55fd8e8c9bc5bb6521539f58db996f23e1a3cb37d99a19403e134d2fdf2` (= re-pin (xvi) pin) |

**Output.** `data/runs/a5/power/power_slotmask_f10.json`, sha256 `b14756828bc27cbbe14e1e590226820371a65ef5f83ad9622b2e6ccf04a3450a`
(re-hashed; it equals `part_a/slotmask_calibration_record.json:output_sha256`; record sha256 `0c5fa8c3…eff6b51`).

- Label "slot-masked (F10.6)", `scenario` `registered`, seed 20261107, 10,000 replicates, 4,000 resolution draws, minimum
  resolved 15.
- Donor pool 61 ids (`ids_sha256` `52366da3…107f1`).
- `stream_order`, `frozen_cell_cutpoints`, `quotas`, `seed`, `resolution_draws`, `noise_scale`, `unavailable_audit`, `proxy`,
  `donor_cluster_members` and `donor_pool` all equal those of the unmasked `data/runs/a5/power/power_registered.json`.

**Available slots per contrast** (`slot_mask.available_slots`): rotation **31**, ladder 0.25 **40**, ladder 0.5 **40**. Per cell
(rotation / ladder 0.25 / ladder 0.5): LoRA:0 10/10/10, LoRA:1 10/10/10, LoRA:2 7/10/10, chain:0 1/4/4, chain:1 3/3/3,
chain:2 0/3/3.

**NULL-SIZE block against the registered null** (`data/runs/a5/nullsize/registered.json`, sha256 `452c078a…4239e8`; equal to the
embedded null of `power_registered.json`):

| quantity | registered (unmasked) | slot-masked | equal |
|---|---|---|---|
| decisions go / alternative / unresolved | 180 / 1939 / 7881 | **192 / 1939 / 7869** | **no** (go +12, unresolved −12; alternative identical) |
| ordered go rate (CP95) | 0.0180 (0.0155–0.0208) | 0.0192 (0.0166–0.0221) | no |
| two-sided rejection rotation / ladder 0.25 / ladder 0.5 | 0.0372 / 0.0372 / 0.0372 | 0.0377 / 0.0372 / 0.0372 | ladders yes; rotation no |
| contrast rejection rates | 0.018 / 0.018 / 0.018 | 0.0192 / 0.018 / 0.018 | ladders yes; rotation no |
| `audit_or_nonfinite_counts` | 0 / 0 / 0 | 90,000 / 0 / 0 (= 9 masked slots × 10,000) | – (by construction) |
| `accepted` | true | true | – |

The masked NULL-SIZE block is **not byte-equal** to the registered null. Every difference is in the rotation contrast, which the
mask removes 9 slots from. The ladder columns and the alternative-branch count are identical. That pattern is what masking alone
should produce, with streams and draws unchanged. Whether "equality" was meant as a gate is for the chair. F10.6 and Board A
condition 5 gate size at ≤ 0.07, and that gate passes.

**Power (POWER-AND-YIELD, 61-donor pool):**

- **ordered go power 0.7514, CP95 [0.7428, 0.7598]**; decisions go 7514 / alternative 0 / unresolved 2486
- two-sided rejection rates rotation / ladder 0.25 / ladder 0.5 = 0.7514 / 0.8525 / 0.8535
- registered unmasked: 0.8540 / 0.8525 / 0.8535; the ladder values are identical

| gate (F10.6, A5.3.2, Board A condition 5) | threshold | value | result |
|---|---|---|---|
| ordered go power | ≥ 0.80 | 0.7514 (CP95 upper 0.7598) | **FAIL** |
| applicable size: ordered go | ≤ 0.07 | 0.0192 | pass |
| applicable size: each two-sided test | ≤ 0.07 | 0.0377 / 0.0372 / 0.0372 | pass |
| funding-only alternative rate | not gated (F5.7) | 0.1939 | disclosed |

**Overall verdict: FAIL** (`slot_mask.gates.passed: false`; `slotmask_calibration_record.json:verdict`). Under F10.6 and Board A
F10 condition 5, Part A is not decisive as registered. The scoring hold stays for every entry, including the ladders and the 19
originals, until a further prospective amendment is accepted. The chair's F11 decision (supplementary selection, 07:20 UTC in
`docs/PROGRESS.md`) is in progress and is not part of this report.

## 7. Exposure status

- **Nothing scored.** `manifest.json:scored` is `false` (and so is `manifest_a533.json`'s). The scorer was not run by the
  re-stage (`restage.evaluation_reads`, quoted in §4.5) or by this report.
- **No file under `data/evals` was opened** by this report. Neither did the re-stage, according to its own record. Board A's
  code review covered the `--rotation-only` route, whose guarded test forbids evaluation opens and listing.
- Path checks done for this report: `os.path.exists` only, under `data/evals/qwen3.5-0.8b/tf_v1/`. No directory was listed and
  no file was opened. 169 ids were checked:
  - **120 variant ids** (40 rotations including the 9 excluded, and 80 ladders): **0 exist**.
  - **19 missing originals**: **0 exist**.
  - 21 originals with records: all 21 exist, as expected, since their records were certified before the re-stage.
  - 9 predecessor ids, including `base`: all exist (reuse-proof references, carried, not opened).
- The F10.6 simulation reads only the simulation inputs listed in §6. Its docstring states "No project data discovery or access
  is performed."
- This report's author did not see any G3, rotation, ladder or reference score, or a summary of one. The inputs were manifests,
  audits, logs, staged bytes (hashes only), simulation outputs and governance documents.

## 8. Chair rulings carried

- **`replay_failed` for scorer round-trip failures.** Chair ruling 1 (`docs/PROGRESS.md`, 03:00 UTC entry) and Board A code
  re-review correction 4. The scorer chain convention bf16(base + (state − base)) cannot carry rotated values where the
  predecessor holds exact zeros and the base is non-zero. These are exclusions under F10.3(iii). Changing the chain storage or
  the apply convention would be a new post-audit amendment. The 6 chain exclusions (76 tensors) stand as implemented.
- **No retuning (F10a.3).** The 1e-5 construction tolerances, the one-fp32-spacing slack of F10.3(iv), the 0.5 guard and the
  replay settings were fixed before outcomes, not before audits. A failure beyond the fixed tolerance is an exclusion or a new
  disclosed amendment. **The 3 lora_sft `rounding_bound_failed` exclusions stand** (`docs/PROGRESS.md`, 06:30 UTC: "no retuning
  under F10a"). §2 notes that their recorded excess is 3,495–12,834.5 fp32 spacings at one entry each, not a one-spacing
  equality.
- **Inclusive equality.** Board A accepted equality at exactly one fp32 spacing under the registered `<=` (no F10c). In the
  re-stage, 93 passing tensors sit at exactly 1.0 spacing, and none of the 2,455 passing rows exceeds it.
- **Other carried items:**
  - no replacement edits, no new seeds, no ladder rebuild (§1.3, §4.2)
  - the rotation exclusion masks rotation slots only (§6)
  - F10.9 flags (§4.4) and the F10a.2 qualification of the eight non-root picks: their ε_D is unavailable
    (`eps_D_status: unavailable_non_root`, 8 pairs)

## 9. Items for the chair and Board A

1. **Freeze mismatch:** the Torch build-config SHA-256 (§1.4). The freeze has `d1b4e24c…`; the emitted value and the value I
   recomputed are `07f0372b…`, and the freeze's derivation is unrecorded. Every other identity field matches.
2. **Freeze gap:** `base_revision` is `{}` in the freeze. The substitute evidence in §1.3 matches the model pin.
3. **Slot-mask provenance:** the first mask (`a2ed6ae7…`) was rejected by the frozen schema. The run used the rewritten mask
   `86fa27a5…`, with the same mapping hash `b7724a69…`.
4. **NULL-SIZE block not equal to the registered null.** The difference is confined to the masked rotation contrast, and the size
   gate passes.
5. **Calibration verdict FAIL** (power 0.7514 < 0.80). The scoring hold remains.
6. **Wording of the three rounding-bound exclusions** in `docs/PROGRESS.md` against the recorded excesses (§2).
7. The A5.1.2 hash entries for this re-stage are in `part_a/RESTAGE_A512_NOTE.md`, for the chair to append. `PREREG_H1_TF.md`
   is not edited here.

## Appendix A. The 79 failing tensors

`rotation_audits.jsonl` rows with `passes: false`. `detail` (`fail_closed_detail`, `replay_detail`) is `null` on every row.

| edit | tensor | reason | failed check | scorer round-trip mismatches | bound violations | max excess (fp32 spacings) | replay bits equal | scorer delta bits equal |
|---|---|---|---|---|---|---|---|---|
| `0027efdf2eae7405` | `model.layers.10.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.11.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.14.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 3 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.15.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.15.self_attn.o_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.16.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.17.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.976562 | True | True |
| `0027efdf2eae7405` | `model.layers.18.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.20.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.22.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.4.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.5.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0 | True | True |
| `0027efdf2eae7405` | `model.layers.9.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0 | True | True |
| `05531ce1db4df63f` | `model.layers.0.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0.554688 | True | True |
| `05531ce1db4df63f` | `model.layers.10.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.9375 | True | True |
| `05531ce1db4df63f` | `model.layers.11.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.992188 | True | True |
| `05531ce1db4df63f` | `model.layers.11.self_attn.q_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0.640625 | True | True |
| `05531ce1db4df63f` | `model.layers.12.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.867188 | True | True |
| `05531ce1db4df63f` | `model.layers.13.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.761719 | True | True |
| `05531ce1db4df63f` | `model.layers.14.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 3 | 0 | 0.695312 | True | True |
| `05531ce1db4df63f` | `model.layers.16.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.859375 | True | True |
| `05531ce1db4df63f` | `model.layers.17.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.90625 | True | True |
| `05531ce1db4df63f` | `model.layers.17.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0.746094 | True | True |
| `05531ce1db4df63f` | `model.layers.18.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.945312 | True | True |
| `05531ce1db4df63f` | `model.layers.18.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0.832031 | True | True |
| `05531ce1db4df63f` | `model.layers.19.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.9375 | True | True |
| `05531ce1db4df63f` | `model.layers.19.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.75 | True | True |
| `05531ce1db4df63f` | `model.layers.2.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.691406 | True | True |
| `05531ce1db4df63f` | `model.layers.21.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.5 | True | True |
| `05531ce1db4df63f` | `model.layers.3.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.96875 | True | True |
| `05531ce1db4df63f` | `model.layers.3.self_attn.o_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.835938 | True | True |
| `05531ce1db4df63f` | `model.layers.3.self_attn.q_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.554688 | True | True |
| `05531ce1db4df63f` | `model.layers.4.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.914062 | True | True |
| `05531ce1db4df63f` | `model.layers.6.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.511719 | True | True |
| `05531ce1db4df63f` | `model.layers.7.self_attn.q_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.984375 | True | True |
| `05531ce1db4df63f` | `model.layers.8.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.5 | True | True |
| `05531ce1db4df63f` | `model.layers.9.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.726562 | True | True |
| `057dcfbedea3251f` | `model.layers.0.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.757812 | True | True |
| `057dcfbedea3251f` | `model.layers.0.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.710938 | True | True |
| `057dcfbedea3251f` | `model.layers.10.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0.980469 | True | True |
| `057dcfbedea3251f` | `model.layers.11.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.925781 | True | True |
| `057dcfbedea3251f` | `model.layers.11.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.667969 | True | True |
| `057dcfbedea3251f` | `model.layers.11.self_attn.q_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.820312 | True | True |
| `057dcfbedea3251f` | `model.layers.12.linear_attn.out_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.980469 | True | True |
| `057dcfbedea3251f` | `model.layers.14.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.886719 | True | True |
| `057dcfbedea3251f` | `model.layers.15.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.925781 | True | True |
| `057dcfbedea3251f` | `model.layers.15.self_attn.q_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.765625 | True | True |
| `057dcfbedea3251f` | `model.layers.16.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0.8125 | True | True |
| `057dcfbedea3251f` | `model.layers.19.self_attn.v_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.71875 | True | True |
| `057dcfbedea3251f` | `model.layers.20.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0.566406 | True | True |
| `057dcfbedea3251f` | `model.layers.20.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.449219 | True | True |
| `057dcfbedea3251f` | `model.layers.21.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.820312 | True | True |
| `057dcfbedea3251f` | `model.layers.22.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.996094 | True | True |
| `057dcfbedea3251f` | `model.layers.22.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0.878906 | True | True |
| `057dcfbedea3251f` | `model.layers.3.self_attn.q_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.839844 | True | True |
| `057dcfbedea3251f` | `model.layers.4.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.496094 | True | True |
| `057dcfbedea3251f` | `model.layers.6.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 1 | True | True |
| `057dcfbedea3251f` | `model.layers.7.self_attn.q_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.953125 | True | True |
| `057dcfbedea3251f` | `model.layers.8.linear_attn.out_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.832031 | True | True |
| `057dcfbedea3251f` | `model.layers.8.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.824219 | True | True |
| `057dcfbedea3251f` | `model.layers.9.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.921875 | True | True |
| `0bc0a0c4ad0520e0` | `model.layers.0.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.980469 | True | True |
| `0bc0a0c4ad0520e0` | `model.layers.11.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.875 | True | True |
| `0bc0a0c4ad0520e0` | `model.layers.18.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.617188 | True | True |
| `0bc0a0c4ad0520e0` | `model.layers.18.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.914062 | True | True |
| `0bc0a0c4ad0520e0` | `model.layers.23.self_attn.o_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.480469 | True | True |
| `0bc0a0c4ad0520e0` | `model.layers.5.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.820312 | True | True |
| `0bc0a0c4ad0520e0` | `model.layers.7.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.925781 | True | True |
| `0bc0a0c4ad0520e0` | `model.layers.8.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.992188 | True | True |
| `0bc0a0c4ad0520e0` | `model.layers.8.mlp.up_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.933594 | True | True |
| `0d055869527177b4` | `model.layers.21.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.828125 | True | True |
| `14cdaabee246493b` | `model.layers.15.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.890625 | True | True |
| `14cdaabee246493b` | `model.layers.15.self_attn.o_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.75 | True | True |
| `14cdaabee246493b` | `model.layers.21.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 3 | 0 | 0.589844 | True | True |
| `14cdaabee246493b` | `model.layers.8.linear_attn.out_proj.weight` | `replay_failed` | `scorer_roundtrip` | 1 | 0 | 0.5 | True | True |
| `14cdaabee246493b` | `model.layers.8.mlp.down_proj.weight` | `replay_failed` | `scorer_roundtrip` | 2 | 0 | 0.5 | True | True |
| `15a53586efc72a50` | `model.layers.11.self_attn.q_proj.weight` | `rounding_bound_failed` | `rounding_bound` | 0 | 1 | 3495 | True | True |
| `2233954b0cc3edc0` | `model.layers.12.mlp.up_proj.weight` | `rounding_bound_failed` | `rounding_bound` | 0 | 1 | 10970 | True | True |
| `72d3756e968e3209` | `model.layers.2.mlp.down_proj.weight` | `rounding_bound_failed` | `rounding_bound` | 0 | 1 | 12834.5 | True | True |

