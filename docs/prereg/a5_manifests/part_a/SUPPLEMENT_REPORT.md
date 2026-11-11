# Part A F11 supplementary staging — report (outcome-free)

Written 11 Nov 2026 (UTC), after the supplementary staging (08:06:02–09:26:07 UTC) and after the F11.4 calibration
(09:28:04–09:34:03 UTC). It is for the chair and Board A, under Board A's F11 staging review
(`docs/reviews/council/board-a-f11-stage-2026-11-11.md`, conditions 2–5 and its execution-record requirements), the F11
simulation review (`board-a-f11-sim-2026-11-11.md`, conditions 2–3) and the F11 amendment review (`board-a-f11-2026-11-11.md`,
conditions 2–4). It covers F11.3–F11.5, F11a and F11b in `docs/PREREG_H1_TF.md`. **Nothing has been scored. This report does
not release scoring.**

Every number below comes from a committed file, a log, or a re-hash or recomputation made for this report. Paths are relative to
the repository root unless absolute. The checks were run by read-only scripts outside the repository (`/tmp/f11report/pins.py`,
`analyze.py`, `rehash.py`, `tables.py`; not committed). They read `docs/prereg/a5_manifests/**`, git objects,
`data/runs/a5/part_a/qwen3.5-0.8b/<variant>/{*.safetensors,variant.json}` and that directory's `manifest.jsonl`,
`data/runs/a5/power/power_slotmask_f11.json`, `data/runs/a5/power/power_slotmask_f10.json` and `data/runs/a5/nullsize/inputs.json`.
Under `data/evals` they only tested whether paths exist (§8). Nothing under `data/` was written.

**Baseline.** This staging builds on the F10 re-stage record (`part_a/RESTAGE_REPORT.md`, sha256 `61294c86…`), which Board A
accepted with conditions (`board-a-f10-restage-2026-11-11.md`), together with the two documentary corrections that review required.
Both are in the chair's correction after the F10 rulings in `docs/PREREG_H1_TF.md` (about line 3902):

- the Torch build-config discrepancy of the re-stage freeze (`d1b4e24c…` frozen against `07f0372b…` emitted) is ruled a
  freeze-procedure error, and `07f0372b…` computed under the frozen environment is binding;
- the rounding count is corrected: 93 rows at exactly one fp32 spacing pass the rounding check, and 92 of them pass F10 overall.

The original re-stage report and its hash are kept. One observation bears on the first correction. The chair attributes
`d1b4e24c…` to a shell without the frozen thread variables. For this report I recomputed `sha256(torch.__config__.show())` with
only `CUDA_VISIBLE_DEVICES=` set and the thread variables **unset**: the result is still `07f0372b…` (with `torch_threads` 8).
Leaving the thread variables unset therefore does not reproduce `d1b4e24c…` here. This does not affect the present freeze,
which matches (§1.5).

## Summary

| item | result |
|---|---|
| Execution | `8211ecf4c079f32dd8cc59b77fdfa5bd8a99a7bc`, clean (0 dirty files). 08:06:02–09:26:07 UTC. Exit 0. CPU only, 4 threads |
| Command | Board A's frozen staging command, string-identical in the freeze, the launch script and the Board A review (§1.1) |
| Pins at execution SHA | 71/71 files in `code_hashes.json` (`46dd5234…`, re-pin (xx)) match. The 6 rows of the freeze's `pinned_source_hashes` match too. The F11 selection builder `scripts/a5_f11_supplement.py` and its test are **not** in `code_hashes.json`; their bytes at `8211ecf` equal Board A's reviewed hashes (§1.3) |
| Freeze vs emitted environment | **No mismatch.** All 10 compared fields are equal across the freeze, `supplement.replay_environment` and all 12 supplementary rotation `variant.json` records. Recomputed under the frozen environment variables: `07f0372b…` (§1.5) |
| main during the run | Advanced from `8211ecf` to `887a514` (8 commits; analysis code, docs and pins only). None touches a module the run imported or an input it read. The manifest's `code_sha`/`provenance` record the end-of-run HEAD `887a514`, not the execution SHA (§1.2) |
| Supplementary F10 verdicts | **12 / 12 pairs pass.** 535 / 535 tensors pass, with no reason codes and no failed checks |
| Yield | **43 / 52 rotation pairs available.** The 9 exclusions are the 9 original ones, unchanged. Ladders: 52 per strength |
| Preserved evidence | The forty's per-entry records are carried identically (8 keys). `manifest_pre_f11.json` and `rotation_audits_pre_f11.jsonl` are byte-identical to the pre-run committed files. `rotation_audits.jsonl` starts with exactly those bytes. On-disk re-hash: the forty's **249 files** (129 safetensors + 120 `variant.json`) equal both manifests (129/129 also equal the freeze inventory); the 72 supplementary files equal `manifest.json`. |
| Slot mask | `slot_mask_f11.json` sha256 `5f4e7701…`, schema-only. The accepted builder rebuilds it byte-identically from `manifest.json`. Rotation 43, ladder 0.25 52, ladder 0.5 52. Canonical mapping `171dfd20…` |
| F11.4 calibration | **PASS** (`slot_mask.gates.passed: true`). Ordered power **0.8722**, CP95 [0.8655, 0.8787]. Size: ordered go 0.0175; two-sided 0.0348 / 0.0378 / 0.0378 (all ≤ 0.07). Null alternative rate 0.2576 (disclosed, not gated) |
| Proofs and price | The 12 supplementary originals are unproven; their proofs are deferred to `score --manifest`. Priced 178 entries × 1.04 = **185.12 min** |
| Exposure | Nothing scored (`scored: false`). No evaluation record opened. 0 of 156 variant ids and 0 of 19 missing originals have an evaluation path (§8) |
| Open items | Ordering and record gaps against the Board A conditions (§10): the calibration ran before this report existed; no frozen `slot_mapping_f11.json`; the selection builder is not pinned in `code_hashes.json`; the launch record lacks the command, environment and output hashes; the launch wrapper was not frozen with a hash |

## 1. Execution record

### 1.1 Command, cwd, time, SHA

Sources: `docs/prereg/a5_manifests/part_a/supplement_launch.json` (sha256 `d17fab2f…fb38`), the agent-tmux log
`/home/ubuntu/agent-logs/a5-f11-supplement-20261111-080602.log`, and the launch script.

- agent-tmux job `a5-f11-supplement` (created 08:06:02 UTC). It ran the launch script `/tmp/a5_f11_supplement.sh` (outside the
  repository; 1,524 bytes, sha256 `1dcc133a4b1cfd8e356fd6b2ff7613054fc43033a0035c83cbd1c3d0eb6c92e4`, mtime
  08:06:02.28 UTC). The script:
  - records HEAD, the dirty count, the start time, the freeze hash, cwd, `nproc` and load into `supplement_launch.json`;
  - refuses on a dirty tree;
  - runs the command under `/usr/bin/time -v`;
  - appends `end_utc` and `exit_code` from inside the persistent job, and prints `SUPPLEMENT_EXIT`.
- cwd `/home/ubuntu/projects/wsp-lab` (launch record `cwd`; freeze `cwd`).
- The exact command is below. It is **string-identical** to the inner command of Board A's frozen staging command
  (`board-a-f11-stage-2026-11-11.md`, without the `agent-tmux a5-f11-supplement` prefix), to `supplement_freeze.json:command`, and
  to the command in the launch script (all three checked by string equality).

```
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m wsp.eval.response_intervention prepare --supplement docs/prereg/a5_manifests/selection_f11.json --no-reference-proofs --audit-rule f10 --staged-manifest docs/prereg/a5_manifests/part_a/manifest.json --zoo data/zoo/qwen3.5-0.8b --selection docs/prereg/a5_manifests/selection.json --references docs/prereg/a5_manifests/references.json --strengths 0.25 0.5 --out data/runs/a5/part_a --manifest-out docs/prereg/a5_manifests/part_a/manifest.json --config configs/eval_tf_v1.yaml
```

- UTC start `2026-11-11T08:06:02Z`, end `2026-11-11T09:26:07Z`, `exit_code` 0. The log ends `SUPPLEMENT_EXIT 0`. From
  `time -v`: wall 1:20:05; user 9,205.51 s; system 451.78 s; 200 % CPU; max RSS 9,537,176 kB; exit status 0. `nproc` 8; load at
  start 5.72 6.31 8.44.
- Execution SHA `8211ecf4c079f32dd8cc59b77fdfa5bd8a99a7bc`, `dirty_files` 0 (launch record). `8211ecf` is the re-pin (xx) merge
  commit (08:05:53 UTC).
- Order:
  - freeze written `2026-11-11T08:05:19.995078+00:00` (`supplement_freeze.json:frozen_at_utc`);
  - freeze committed in `8211ecf` (08:05:53);
  - launch 08:06:02;
  - outputs committed in `92b0338` (09:27:54).

  The freeze has a single version in git (`8211ecf`). Its sha256 `503e415e…a0fda4` equals `supplement_launch.json:freeze_sha256`.
- Log lines, one per supplement in edit-id order, each `rotation pass`. Seconds per edit: `15cdcf0b` 263.3, `17668df1` 369.7,
  `1ac30f3c` 62.4, `2410c595` 388.6, `342a1be0` 15.6, `388347cf` 22.0, `45cd0be9` 10.7, `8d29ac2a` 322.1, `aaf7a066` 320.8,
  `af6d608f` 318.8, `b5d0ac5d` 318.7, `c12f6ec6` 317.2. The final line is `merged 52 edits: rotation pairs 43 pass, 9 excluded;
  priced 178 entries (185.12 min); unproven originals 12`.
- `--evals` and `--reference-proofs` were not passed. Ordinary `prepare` and `--rotation-only` were not used. This is the single
  supplementary staging attempt: the `manifest.json` predecessor carried no `supplement` block, and there is no other
  `a5-f11-supplement` job or log.

### 1.2 main advanced during the run

The run's cwd is the canonical checkout. While it ran, main advanced from `8211ecf` to `887a514` (`git log 8211ecf..887a514`):

| commit | UTC | subject | files |
|---|---|---|---|
| `3aa9f62` | 08:06:22 | progress: timestamp correction note 2 | `docs/PROGRESS.md` |
| `afe24a9` | 08:06:57 | progress: analysis F11-path corrections committed | `docs/PROGRESS.md` |
| `62c2785` | 08:12:15 | Board A: analysis F11 path accepted with conditions | `docs/reviews/council/board-a-part-a-analysis-4-2026-11-11.md` |
| `f1565db` | 08:12:15 | merge Board A analysis review 4 | (merge) |
| `ade2850` | 08:12:15 | merge fix/part-a-analysis-manifest-binding (brings `f57b70a`, `aef0a83`) | `src/wsp/eval/part_a_analysis.py`, `scripts/part_a_analyse.py`, `tests/test_part_a_analysis.py` |
| `9add718` | 08:16:53 | merge Part A analysis F11 path; A5.1.2 re-pin (xxi) | `README.md`, `docs/PREREG_H1_TF.md`, `docs/PROGRESS.md`, `docs/prereg/a5_manifests/README.md`, `docs/prereg/a5_manifests/code_hashes.json` |
| `e9114c3` | 08:28:33 | A5 F11b | `docs/PREREG_H1_TF.md`, `docs/PROGRESS.md`, `docs/prereg/f10_attestations.md` |
| `887a514` | 08:28:51 | merge prereg/f11b | (merge) |

`git diff --stat 8211ecf 887a514` touches exactly 10 files: `README.md`, `docs/PREREG_H1_TF.md`, `docs/PROGRESS.md`,
`docs/prereg/a5_manifests/README.md`, `docs/prereg/a5_manifests/code_hashes.json`, `docs/prereg/f10_attestations.md`, the
Board A review, `scripts/part_a_analyse.py`, `src/wsp/eval/part_a_analysis.py` and `tests/test_part_a_analysis.py`.

**None of these files was imported or read by the run:**

- The only changed module under `src/` is `wsp/eval/part_a_analysis.py`. Nothing under `src/` at `8211ecf` imports it
  (`git grep part_a_analysis 8211ecf -- src` finds nothing). Importing the run's entry points (`wsp.eval.response_intervention`,
  `wsp.eval.batch`, `wsp.features.extract`, `wsp.zoo.apply`) loads 20 `wsp` modules and not `part_a_analysis`.
- `response_intervention.py` at `8211ecf` does not read `code_hashes.json`, `PREREG_H1_TF.md`, `PROGRESS.md` or the READMEs.
- Every input the run read is unchanged. This covers the five freeze inputs (§1.4) and `selection_f11.json`. It also covers the
  selection file's four further hashed inputs: `components.jsonl` `6b4eff8d…`, `power/fields_manifest.json` `19e9de84…`,
  `power/registered_donors.json` `6e77bca6…`, and the pre-supplement manifest, preserved as `manifest_pre_f11.json`
  `59d839e3…`. Each was compared at `8211ecf` and on disk.
- The three changed pins (`part_a_analysis.py`, `part_a_analyse.py`, `test_part_a_analysis.py`) are re-pin (xxi). Their old
  hashes are those pinned at `8211ecf`.

**Documentary consequence.** `prepare_supplement` calls `wsp.eval.batch.provenance()` after construction, just before writing the
manifest (`response_intervention.py` at `8211ecf`, line 2538). So `manifest.json:code_sha` and `provenance.git_sha` are
`887a5146b459e29bb42cda24482a2bcf458537a3`, the HEAD at write time, not the execution SHA. `provenance.source_hash`
`5775fc0a…6d139` reproduces exactly from the tracked `src/wsp/**/*.py` at `887a514`. The same derivation at `8211ecf` gives
`891d6faf…7627eb`, and the only difference is `part_a_analysis.py`. `provenance.dirty: false` refers to the tree at write time.
**The authoritative execution SHA is `8211ecf` (launch record).** The manifest's `code_sha` should not be read as the execution
SHA.

### 1.3 Pinned source hashes at the execution SHA

I re-hashed each file pinned in `git show 8211ecf:docs/prereg/a5_manifests/code_hashes.json` from `git show 8211ecf:<path>`. That
file has sha256 `46dd52341df19ad0588301633780d5934830d553000aa40abb5bfee5760b7387`, the re-pin (xx) value, with `git_sha`
`f4c8f05`. **All 71 of its pinned files match.** At `887a514` and `92b0338`, 68 of the 71 still match these pins, and the 3
that differ are the re-pin (xxi) rows. Against its own `code_hashes.json` (`34477308…`), `92b0338` matches 71/71.

| file | freeze `pinned_source_hashes` | `code_hashes.json` at `8211ecf` | re-hash at `8211ecf` | Board A reviewed |
|---|---|---|---|---|
| `src/wsp/eval/response_intervention.py` | `e52ea009740751a38c9445dfb6a63767a9535cd22ce343c240371269e55d4d79` | same | same | same |
| `src/wsp/features/rotation.py` | `8ac459934b0b145f62a3f0c81f68980c1f57d0ef7663d16e9ce1f693be376dba` | same | same | same |
| `tests/test_response_intervention.py` | `c7f202369469f56ac02403986b14c9ce25393c48df69a9741224a1c2397965bc` | same | same | same |
| `tests/test_rotation.py` | `7570224a0b4c0dd8e9f1b0aacdaa930e2170d97fa1d1e9c6864bd60d13869325` | same | same | – |
| `scripts/a5_simulation.py` | `4ce3faa51889c4aa27f4d9580e7aab9343351954146cbbd76e162ccd47a7947a` | same | same | same (sim review) |
| `tests/test_a5_simulation.py` | `d231c8b0264d48e7e75b991b5bf82ee2f71b846c0364063800bbea84920d2833` | same | same | same (sim review) |
| `src/wsp/eval/batch.py` (provenance; not in freeze) | – | `122b0fba…789c6` | same | – |
| `src/wsp/features/extract.py` (base load; not in freeze) | – | `11de545c…626906` | same | – |
| **`scripts/a5_f11_supplement.py`** | **absent** | **absent** | `90dcb312a9e81104dd24167c1e6243af7711a5d4e5ac610d3072cf65ae7764a3` | `90dcb312…` |
| **`tests/test_a5_f11_supplement.py`** | **absent** | **absent** | `6cf4f07c9281024ebaa529f206fc1bfd9153f9d5cddd4dcc7ee108888251e739` | `6cf4f07c…` |
| `docs/prereg/a5_manifests/selection_f11.json` | `ce746e71…793c` (`selection_f11.sha256`) | – | `ce746e7135dd94b087da86953beb02382e5706f8ff2dd17bd57c053aaf54793c` | same |

Board A F11 staging condition 1 asked for the selection builder to be pinned "as governed selection code", together with both
test files. Neither the builder nor `tests/test_a5_f11_supplement.py` is a row of `code_hashes.json` at `8211ecf`. Re-pin (xx)
lists only `response_intervention.py`, its test, `code_hashes.json` and the README. Their bytes at the execution SHA do equal the
reviewed hashes. The builder was not executed by this run: the loader recomputes the F11.2 rule in `response_intervention.py`
(`f11_rule`), and the manifest records `supplement.rule` "F11.2 recomputed from the selection's hashed inputs (f11_rule):
equal". The omission is documentary, and it is listed in §10. Environment pins at `8211ecf`: python 3.12.3, torch 2.15.0, numpy
2.5.4, safetensors 0.8.0, transformers 5.22.1. Model and tokenizer: `Qwen/Qwen3.5-0.8B` at
`2fc06364715b967f1860aea9cf38778875588b17`.

### 1.4 Frozen inputs

| input | freeze | re-hash at `8211ecf` | re-hash on disk now | match |
|---|---|---|---|---|
| `configs/eval_tf_v1.yaml` | `5c20cabe8e35b4116adeed4df650cf27461e7ab144ef66c0b60c397b13b00317` | same | same | yes |
| `docs/prereg/a5_manifests/control_seeds.json` | `d357ed51031877f817baeb4204cf93b38c66fdfcfd95bd41803d93d5160c12d5` | same | same | yes |
| `docs/prereg/a5_manifests/references.json` | `5b6991f299c1eee049de51bc1e4f00c71c1dfc58114cdf6549fb5d246850564e` | same | same | yes |
| `docs/prereg/a5_manifests/selection.json` | `393530365f87f981d5da3ecb2779c89b8e7bf98105b6c45d1824bf290e908c63` | same | same | yes |
| `docs/prereg/a5_manifests/text_groups.json` | `f769afcade5a80571daf3a9578b1c03584b46a5abbbba07b32d43a7e9035e7e1` | same | same | yes |
| `selection_f11.json` / sorted-id sha256 | `ce746e71…` / `1848fe4cacbaacc9f643a179835770fefd3027a48cef7cf9d83358f15b85fdd9` | same | same | yes. The manifest's `supplement.sha256` and `supplement.ids_sorted_sha256` are equal too |
| originals' manifest / audits | `59d839e3…0b06` / `4ec0928a…97f3` (`originals_before`; 40 edit ids, 111 delta-hash entries) | – | `manifest_pre_f11.json` / `rotation_audits_pre_f11.jsonl`: same | yes |
| staged byte inventory | 111 variants, 129 files | – | §4.2 | §4.2 |
| model revision | `2fc06364715b967f1860aea9cf38778875588b17` (`model_revision_source`: per-edit records of the F10 re-stage) | – | – | yes; every variant was built at this revision (the run refuses mixed or other revisions) |

Re-pin (xx) describes an "81-file staged byte inventory". The committed freeze holds 129 files (80 ladder deltas, 16 ladder
increments, 31 rotation deltas and 2 rotation increments) in 111 variants. The figure in the re-pin text is wrong; the frozen file
governs.

### 1.5 Numerical environment: freeze vs emitted

Emitted environment: `manifest.json:supplement.replay_environment`. It is identical to the `replay_environment` of each of the
12 supplementary rotation `variant.json` files on disk, and to the carried `restage.replay_environment` of the forty. All 535
supplementary tensor audits record `replay_threads` 4.

**Recomputation under the frozen environment.** I ran the pinned `rotation.numerical_environment()` (`rotation.py` `8ac45993…`)
in the same `.venv` under `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
PYTHONDONTWRITEBYTECODE=1`. It returns `torch_config_sha256` `07f0372b5746fc761d40701ab808447f3703337e82f04fcb4846c45e71ba008d`,
`torch_threads` 4, both thread variables `"4"`, Python 3.12.3, Torch 2.15.0+cu132, MKL available, CPU `Intel(R) Xeon(R) 6975P-C`
and x86_64. A direct `sha256(torch.__config__.show().encode())` gives the same digest. CUDA is unavailable in the process.

| field | `supplement_freeze.json` | emitted (`supplement.replay_environment` = 12 rotation `variant.json`) | recomputed (frozen env) | match |
|---|---|---|---|---|
| CPU only | `cpu_only: true`; `computed_under_env.CUDA_VISIBLE_DEVICES` `""` | `CUDA_VISIBLE_DEVICES=` in the command (not emitted) | CUDA unavailable | yes |
| `OMP_NUM_THREADS` | `"4"` (`computed_under_env`) | `"4"` (also `supplement.thread_env`) | `"4"` | yes |
| `MKL_NUM_THREADS` | `"4"` | `"4"` | `"4"` | yes |
| Torch intra-op threads, construction | 4 (`torch_intra_op_threads_required`; `torch_threads_now` 4) | `torch_threads` 4 | 4 | yes |
| Torch intra-op threads, replay | 4 | `replay_threads` 4; every row `replay_threads` 4 | – | yes |
| Python | 3.12.3 | 3.12.3 | 3.12.3 | yes |
| Torch | 2.15.0+cu132 | 2.15.0+cu132 | 2.15.0+cu132 | yes |
| Torch build-config SHA-256 | `torch_config_sha256` `07f0372b…008d` | `07f0372b…008d` | `07f0372b…008d` | **yes** |
| MKL available | true | true | true | yes |
| CPU model | Intel(R) Xeon(R) 6975P-C | Intel(R) Xeon(R) 6975P-C | same | yes |
| Machine | x86_64 | x86_64 | x86_64 | yes |
| `HF_HUB_OFFLINE` | `"1"` | `1` in the command (not emitted) | – | yes (command) |

**No freeze mismatch.** The freeze field name (`torch_config_sha256`) now equals the emitted name, and the value was computed under
the frozen environment, as the chair's ratification requires.

### 1.6 Execution-record gaps against Board A staging condition 2

- `supplement_launch.json` holds schema, start/end, execution SHA, dirty count, freeze hash, cwd, `nproc`, load and exit code. It
  does **not** hold the exact argv, environment, agent-tmux job/log identity or output hashes. These are supplied here from the
  launch script, the log (§1.1) and my re-hashes (§4, A5.1.2 note).
- The launch script `/tmp/a5_f11_supplement.sh` was written at the launch second (mtime 08:06:02.28). The freeze does not name
  it or carry its hash. Condition 2 asked for "any instrumentation wrapper" to be frozen verbatim with its hash before execution.
  Its hash is `1dcc133a…92e4`, recorded here after the fact. The script records start, SHA and freeze hash, refuses a dirty tree,
  wraps the frozen command in `time -v`, and records end and exit.

## 2. The twelve supplementary edits

Sources: `manifest.json` (`selections`, `edit_audits`, `variants`, `rotation_seeds`, `delta_sha256`, `variant_json_sha256`,
`supplement`), the 535 supplementary rows of `rotation_audits.jsonl` (rows 2,535–3,069, edit-id order), `selection_f11.json`
(`cells[].candidate_order`), and the 36 `variant.json` files on disk.

All twelve are roots (`predecessor_id` `base`, `delta_convention` `realised_increment_at_base`) and are flagged
`supplementary: true` on their selection row and their edit audit. Exactly 12 selection rows and 12 edit audits carry the flag;
none of the forty does. The tensor rows of `rotation_audits.jsonl` carry no `supplementary` key, either for the forty or for the
twelve; supplementary rows are identified by their position after the pre-F11 bytes and by edit id. Cells and slots are as in
`manifest.json:supplement.cells` and the calibration's slot table (§6): LoRA-high slots 40–43, chain-low slots 44–47, chain-high
slots 48–51. The position is the candidate's index in the sorted registered cell (`selection_f11.json`).

| slot | edit | cell | position in cell (status) | rotation variant | F10 pair verdict | tensors edited / rotated / excluded | per-tensor verdicts | reason codes | failed checks | `fail_closed_detail` / `replay_detail` non-null | max rounding excess (fp32 spacings) | bound violations | rows at exactly 1 spacing: rounding-check pass / F10 pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | `8d29ac2adb7fdda0` | lora_sft:2 | 10 (selected) | `9f4c38110febf1b7` | **pass** | 72 / 72 / 0 | pass 72 | – | – | 0 / 0 | 0.98828125 | 0 | 0 / 0 |
| 41 | `af6d608f8651fb46` | lora_sft:2 | 11 (selected) | `8f20341f47e95ec1` | **pass** | 72 / 72 / 0 | pass 72 | – | – | 0 / 0 | 1.0 | 0 | 2 / 2 |
| 42 | `b5d0ac5d76c3cad6` | lora_sft:2 | 12 (selected) | `448a6d0f338360f9` | **pass** | 72 / 72 / 0 | pass 72 | – | – | 0 / 0 | 1.0 | 0 | 1 / 1 |
| 43 | `c12f6ec6bad59ba3` | lora_sft:2 | 13 (selected) | `dcef7b692d17e808` | **pass** | 72 / 72 / 0 | pass 72 | – | – | 0 / 0 | 1.0 | 0 | 3 / 3 |
| 44 | `342a1be080049e8f` | chain:0 | 6 (selected) | `ee45ea9f284e7a2e` | **pass** | 2 / 2 / 0 | pass 2 | – | – | 0 / 0 | 0.94921875 | 0 | 0 / 0 |
| 45 | `388347cfb0b7fafa` | chain:0 | 7 (selected) | `2b0c71a449ddd099` | **pass** | 4 / 4 / 0 | pass 4 | – | – | 0 / 0 | 0.9609375 | 0 | 0 / 0 |
| 46 | `45cd0be9fea1c139` | chain:0 | 8 (selected) | `ef7d3ec65d35d6ce` | **pass** | 2 / 2 / 0 | pass 2 | – | – | 0 / 0 | 0.9921875 | 0 | 0 / 0 |
| 47 | `aaf7a0660c6d5a73` | chain:0 | 11 (selected) | `87b27885a3bbec6c` | **pass** | 72 / 72 / 0 | pass 72 | – | – | 0 / 0 | 1.0 | 0 | 3 / 3 |
| 48 | `15cdcf0bb126de17` | chain:2 | 3 (selected) | `42119ad1b0b27a5f` | **pass** | 11 / 11 / 0 | pass 11 | – | – | 0 / 0 | 0.984375 | 0 | 0 / 0 |
| 49 | `17668df17662b3f7` | chain:2 | 5 (selected) | `b5219d6a6978632c` | **pass** | 72 / 72 / 0 | pass 72 | – | – | 0 / 0 | 1.0 | 0 | 1 / 1 |
| 50 | `1ac30f3c7a209fe6` | chain:2 | 6 (selected) | `67c689ca2473e4f4` | **pass** | 12 / 12 / 0 | pass 12 | – | – | 0 / 0 | 0.9765625 | 0 | 0 / 0 |
| 51 | `2410c595817dc20b` | chain:2 | 7 (selected) | `74c12d9f47980688` | **pass** | 72 / 72 / 0 | pass 72 | – | – | 0 / 0 | 1.0 | 0 | 2 / 2 |

**F10 verdicts: 12 / 12 pairs pass; 535 / 535 tensors pass.**

- No row has a reason code, failed check, non-finite field, non-null `fail_closed_detail` or non-null `replay_detail`.
- Every row has `replay_bits_equal`, `prerounding_replay_bits_equal` and `scorer_delta_bits_equal` true, and no rounding-bound
  violation.
- The maximum rounding excess is 1.0 fp32 spacing (inclusive equality, as ruled). 12 supplementary rows sit at exactly 1.0
  spacing, and all 12 pass F10 overall. Across the whole merged audit file, 105 rows pass the rounding check at exactly one spacing
  and 104 pass F10 overall. The difference is the original `057dcfbedea3251f` row already identified in the baseline correction.
- No supplementary pair is excluded, so no supplementary ladder is excluded either. All 24 supplementary ladders are
  `excluded: false`, `realised_increment_ladder: false` and `scale_source: raw_stored_delta`.

**Diagnostics (F10.4).** These are `edit_audits.<id>.edit_audit` pooled fields, defined in its `aggregation` block; ε_D is
available for roots. The two subset columns are the supplements' values of the F10.5(c)/(e) predicates. Under F11.5 and F10a.4
they are shown beside the frozen subsets and do not enter them (§4.4).

| edit | cell | rank | effective norm | pooled norm mismatch (signed) | log energy ratio ln ρ | κ (pooled) | ε̂ (pooled) | ε_D (pooled) | noise-calibrated (tensors failing) | pooled-norm ≤ 2e-2 predicate | noise-calibrated predicate | a533 diagnostic failing / edited | replay bits / pre-rounding replay bits / scorer bits | replay threads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `8d29ac2adb7fdda0` | lora_sft:2 | 43 | 2.6595 | +4.3358e-03 | +8.6529e-03 | 3.038 | 0.09502 | 0.09448 (root_stored_delta) | True (0) | True | True | 17 / 72 | True / True / True | [4] |
| `af6d608f8651fb46` | lora_sft:2 | 44 | 3.0268 | +3.3801e-03 | +6.7489e-03 | 3.458 | 0.08349 | 0.08317 (root_stored_delta) | True (0) | True | True | 23 / 72 | True / True / True | [4] |
| `b5d0ac5d76c3cad6` | lora_sft:2 | 47 | 3.8990 | +2.0636e-03 | +4.1229e-03 | 4.454 | 0.06481 | 0.06463 (root_stored_delta) | True (0) | True | True | 38 / 72 | True / True / True | [4] |
| `c12f6ec6bad59ba3` | lora_sft:2 | 50 | 4.3304 | +1.6934e-03 | +3.3840e-03 | 4.947 | 0.05836 | 0.05824 (root_stored_delta) | True (0) | True | True | 18 / 72 | True / True / True | [4] |
| `342a1be080049e8f` | chain:0 | 0 | 0.2766 | +6.6690e-03 | +1.3294e-02 | 2.16 | 0.1337 | 0.0002355 (root_stored_delta) | True (0) | True | True | 2 / 2 | True / True / True | [4] |
| `388347cfb0b7fafa` | chain:0 | 1 | 0.3588 | +7.5939e-03 | +1.5131e-02 | 1.95 | 0.1481 | 0.0002108 (root_stored_delta) | True (0) | True | True | 4 / 4 | True / True / True | [4] |
| `45cd0be9fea1c139` | chain:0 | 2 | 0.3608 | +4.1569e-03 | +8.2966e-03 | 2.767 | 0.1043 | 0.0002676 (root_stored_delta) | True (0) | True | True | 2 / 2 | True / True / True | [4] |
| `aaf7a0660c6d5a73` | chain:0 | 6 | 0.6055 | +2.5911e-02 | +5.1162e-02 | 0.6915 | 0.4175 | 0.0001163 (root_stored_delta) | False (1) | False | False | 72 / 72 | True / True / True | [4] |
| `15cdcf0bb126de17` | chain:2 | 28 | 1.4596 | +1.6780e-03 | +3.3533e-03 | 4.437 | 0.06505 | 0.0003552 (root_stored_delta) | True (0) | True | True | 11 / 11 | True / True / True | [4] |
| `17668df17662b3f7` | chain:2 | 37 | 2.3409 | +5.5693e-03 | +1.1108e-02 | 2.674 | 0.108 | 0.0002184 (root_stored_delta) | True (0) | True | True | 37 / 72 | True / True / True | [4] |
| `1ac30f3c7a209fe6` | chain:2 | 36 | 2.3232 | +9.3237e-04 | +1.8639e-03 | 6.667 | 0.0433 | 0.0003601 (root_stored_delta) | True (0) | True | True | 12 / 12 | True / True / True | [4] |
| `2410c595817dc20b` | chain:2 | 34 | 1.9529 | +7.4838e-03 | +1.4912e-02 | 2.231 | 0.1294 | 0.0002081 (root_stored_delta) | True (0) | True | True | 46 / 72 | True / True / True | [4] |

The a533 diagnostic, logged only (F10.3), passes on 253 of the 535 supplementary tensors and on 0 of 12 pairs.

**Seeds.** The seed rule is `manifest.json:supplement.seed_rule`, the registered A5.3.3 rule of `control_seeds.json:rotation`
(domain `20261107:rotation:`, implementation `response_intervention.py:rotation_seed`): SHA256 of the UTF-8 bytes
`20261107:rotation:<id>`, the first eight bytes big-endian, mod 2^63. There is no new domain, counter or offset. One torch
generator per edit is shared across its tensors in sorted-name order, and the ladders carry the same seed. I recomputed each seed.
The recomputed value equals each of the following:

- `manifest.json:rotation_seeds`;
- `edit_audits.<id>.seed`;
- the `seed` of all three variants in `manifest.json:variants`;
- every tensor row's `seed`;
- the rotation `variant.json` `seed`.

`control_seeds.json` holds the rule and test vector, not per-id seeds.

| edit | seed (manifest `rotation_seeds`) | recomputed | all copies equal | rotation variant (delta sha256) | ladder s=0.25 (delta sha256) | ladder s=0.5 (delta sha256) |
|---|---|---|---|---|---|---|
| `8d29ac2adb7fdda0` | 3896599575158291721 | 3896599575158291721 | True | `9f4c38110febf1b7` (`5072771c74fced8ad74a1804a39eef086b2cf4fc56bf27c03580df93e1053367`) | `d627ee70fdf98765` (`54c3971d2c134541572b820fba7aa5cc357fbddd3636b397389957f4e796f2d8`) | `e04a0838ac4d050d` (`d7fffb3260df709cb211a4f435b67410e10177abeb391f02481e7d4e4769a69d`) |
| `af6d608f8651fb46` | 2807489662151630549 | 2807489662151630549 | True | `8f20341f47e95ec1` (`1365a4799857ed68d52dc9b47803d98359766c0ad9f6172c8dc5858e75c1d25f`) | `6149784b7e175235` (`069b0755d1f2b5837dae2c0470c26d45ee61633f3a6e11e8e675f89b74973772`) | `ae3438814695d10a` (`e23d3f3d5b85fc4ccda51067d0721e58da79d11fc25747df95756ada3a52264c`) |
| `b5d0ac5d76c3cad6` | 6837927833222583257 | 6837927833222583257 | True | `448a6d0f338360f9` (`b3adbe0725b999a0e96e4aef44e3bb93f3c24978c544f081941705a508f60afd`) | `82e6ef53bd3b5bd0` (`e1a4c51e1641b4139d08c3c906f3560c2b3a07a121b7e96169f03cc25243abaa`) | `3f0e14fc08521ff1` (`ceb6093e1f057c976784266f82fb818f623e75ebbdbe3c9de410557ea33fac77`) |
| `c12f6ec6bad59ba3` | 3104073855167916068 | 3104073855167916068 | True | `dcef7b692d17e808` (`37086303b4dd7a50c3b391fbbae3793ed3b96c5b511060a8c14bc9193c4796f1`) | `6b25cea0650c5b07` (`cce4bcc3156e3a28f80f2ebf3882cd5bdbbb570d5085722eabf05f1bb2720475`) | `ea199e4ddb21a294` (`c45f6f4cd0831e1043042cc2ebba9a97b1cb5a81873c4f2ebc51c3aed8218036`) |
| `342a1be080049e8f` | 1335364241872111394 | 1335364241872111394 | True | `ee45ea9f284e7a2e` (`07fd38c0135ebef645c7fe62270c75f874403f97a6998d239258a493018abaa8`) | `8decf208957f3989` (`15e267692abca5097645fca7d112591048bfe9124e5820a29361cb3a177f5967`) | `4368f8331d279091` (`527bab06ba217c9b1028328810bacd51da3c5fd182f179ae366b42657c5bcb89`) |
| `388347cfb0b7fafa` | 6540153900348937911 | 6540153900348937911 | True | `2b0c71a449ddd099` (`1a6de861b21695c6c902279226578f12de8c79fb4f6150daec0db1f8165aa28b`) | `b5e39024fd33d898` (`9a32641d3b229a8b9d4b14817fd5fbf202c7a190193445930a5a1bc9b2556943`) | `4b2306d72e767529` (`14828029746ebe8bbad292102c51725bba2d3d47e5a6d17077a2227766891ede`) |
| `45cd0be9fea1c139` | 1157222484757193273 | 1157222484757193273 | True | `ef7d3ec65d35d6ce` (`b8eda0b02a887a3881657ab574b8f811d3e19fd9bdaf6ed094c00c0c1db00e65`) | `956f31365297b6c2` (`a41566fed60dd559079afa10336ec558412463fa92aff19385ab431e5c3c2336`) | `8d6934de2d8367fc` (`998df8a07bd9dcbfbcb7109b48049abfafe0944b420c2b980bad6ca616abeb81`) |
| `aaf7a0660c6d5a73` | 4932963976599175416 | 4932963976599175416 | True | `87b27885a3bbec6c` (`00b6d7aa106dc0e4c4af1c44b330081d2843b10d023fc695d64da94efa6a2088`) | `24752f64cd3b4080` (`fe59b44ff94903bad53a6824678ae277ce77ad168dab0324e39eb2c8cb19dbeb`) | `41078f1c56148747` (`2d9f4bb900f851fc19e8c41eeb1b0b0161b834f1a7041270d329a3fb351cff26`) |
| `15cdcf0bb126de17` | 1324569809074110891 | 1324569809074110891 | True | `42119ad1b0b27a5f` (`7970cf154d165d08ccacf9f04537252642277d16cb3b7195aa51a64fb69c15ba`) | `46d309523cf70eac` (`101bae5b8fda6d3f87420e7ab44bacb5e7115ace831a233bca547c7f2996c5e9`) | `57185815ae8a2cf0` (`4a2ca400255d616e0f30dcf46ab02f75deacdf129d9b3cc24f5dcee2fe07da69`) |
| `17668df17662b3f7` | 4284138917104348434 | 4284138917104348434 | True | `b5219d6a6978632c` (`f3add79e1c7bc725baff351962fe3c8576a7072246f93610f8ade6c2301709f2`) | `f246f77918b32213` (`613846e4e76b869a308f060f26f485670cf6f80713884771c3892beba2ea67a5`) | `105202417885974f` (`a955cc6a055e777570f61c45d267b5afe20f624127f11d888dbdc3e352c50f27`) |
| `1ac30f3c7a209fe6` | 7772502519576634935 | 7772502519576634935 | True | `67c689ca2473e4f4` (`399d78d7b9bed79e63b0776980ea5cd7fc961ce19068ec07fbc9a867e4555d5e`) | `92089146a25f10d1` (`b593bc669568072e6d79eaeb57b55a18bbb30b421cd9bdeb2ec49215cae08996`) | `4204ea88f830d431` (`dea9c5cb51e3ddc281bc1e62dbda4feb3753bd300456da469afc71407d5d13b8`) |
| `2410c595817dc20b` | 6619342855621582922 | 6619342855621582922 | True | `74c12d9f47980688` (`47f725df9121008bc886ce080a88ab2229b5623db8ecfa053b3df3a39f233c91`) | `431f4d88693e6b9a` (`1d630ce4f01a14b42e76ec8233869fa5585bcbf86c38229856e55882c7975dea`) | `816953b82a434a9a` (`95eeecf331cfc0eedf62edc16553ca450f82b3d215bda3b52a1c059bf6689920`) |

Every supplementary `variant.json` on disk re-hashes to `manifest.json:variant_json_sha256` (36/36). The delta files are
re-hashed against the manifest in §4.3.

## 3. Yield

Sources: `manifest.json:coverage`, `edit_audits`, `excluded_variant_ids`, `rotation_audits`.

- **Rotation pairs: 43 pass, 9 excluded, of 52** (`coverage.n_rotation_pairs_staged` 43, `n_rotation_pairs_excluded` 9; log
  line `rotation pairs 43 pass, 9 excluded`).
- Tensors: 3,069 edited and 2,519 rotated (`coverage`; 2,534 + 535 and 1,984 + 535). `rotation_audits` has `n_tensors` 3,069 and
  `n_pass` 2,990, which equals my count.
- **The nine original exclusions are unchanged.**
  - The excluded pairs are `0027efdf2eae7405`, `05531ce1db4df63f`, `057dcfbedea3251f`, `0bc0a0c4ad0520e0`, `0d055869527177b4`
    and `14cdaabee246493b`, all `replay_failed`, and `15a53586efc72a50`, `2233954b0cc3edc0` and `72d3756e968e3209`, all
    `rounding_bound_failed`.
  - `excluded_variant_ids` (9) is identical to `manifest_pre_f11.json`.
  - All 40 original pair verdicts are identical to `manifest_pre_f11.json`.
  - The pre-F11 audit rows (2,455 pass, 76 `replay_failed`, 3 `rounding_bound_failed`) are byte-identical (§4.1).
- Ladders: 52 available per strength, the 40 original and the 12 supplementary.

| cell | registered: pairs / pass / excluded | supplementary: pairs / pass | rotation available | ladder 0.25 / 0.5 available |
|---|---|---|---|---|
| lora_sft:0 | 10 / 10 / 0 | – | 10 | 10 / 10 |
| lora_sft:1 | 10 / 10 / 0 | – | 10 | 10 / 10 |
| lora_sft:2 | 10 / 7 / 3 | 4 / 4 | 11 | 14 / 14 |
| chain:0 | 4 / 1 / 3 | 4 / 4 | 5 | 8 / 8 |
| chain:1 | 3 / 3 / 0 | – | 3 | 3 / 3 |
| chain:2 | 3 / 0 / 3 | 4 / 4 | 4 | 7 / 7 |
| **total** | **40 / 31 / 9** | **12 / 12** | **43** | **52 / 52** |

`coverage.by_family`: chain 18 edits, 6 pairs excluded; lora_sft 34 edits, 3 pairs excluded.

## 4. Preserved evidence

### 4.1 Pre-F11 files and audit bytes

| file | sha256 (re-hash) | equals |
|---|---|---|
| `part_a/manifest_pre_f11.json` | `59d839e351824fa154a665fcf105284ce5fdeed8423a5c0a2b7a60687b0d3b06` | byte-identical to `git show 8211ecf:…/part_a/manifest.json` and to `848d7a5`'s; freeze `originals_before.manifest_sha256`; `manifest.json:supplement.pre_f11.manifest`; Board A's required pre-supplement hash |
| `part_a/rotation_audits_pre_f11.jsonl` | `4ec0928a91b5523a400b911f00f8045f267386c2589aa72a58816d565f5c97f3` | byte-identical to `git show 8211ecf:…/rotation_audits.jsonl` and to `848d7a5`'s; freeze `originals_before.rotation_audits_sha256`; `supplement.pre_f11.rotation_audits` |
| `part_a/rotation_audits.jsonl` (merged) | `9412eb0d4d60f5587c4c5d19e9afcda86c6c45faa38a37b3ab937008d505c45a` | `manifest.json:rotation_audits.sha256`. Its first 2,534 lines are exactly the bytes of `rotation_audits_pre_f11.jsonl` (prefix check), followed by 535 supplementary rows |
| `part_a/manifest.json` (merged) | `71e77a3e1f703451864ce960129399be1579d8aa84631bebc0c8d6324b4790d6` | – |
| `part_a/manifest_a533.json`, `rotation_audits_a533.jsonl`, `README.md`, `slot_mask_f10.json` | `d7785b35…`, `f05fec42…`, `2fcbf517…`, `86fa27a5…` | unchanged (not modified by `92b0338`) |

`supplement.pre_f11` also records the pre-F11 `code_sha` `7bacec1…` and provenance `source_hash` `e1ca5194…`.

### 4.2 The forty's entries and staged bytes

**Per-entry records.** For each key in `supplement.carried.keys` I compared every entry of the forty in `manifest_pre_f11.json`
with `manifest.json` by JSON equality. All are equal:

- `edit_audits` (40), `rotation_seeds` (40), `selections` (40), `variants` (120), `delta_sha256` (111), `variant_json_sha256` (120);
- `restage` and `selection_ledger` (whole-object equality).

`excluded_variant_ids`, `realised_increment_ladder_variant_ids` (16) and `staged_zoo` are also equal. The record's own
`supplement.carried.entries_byte_identical` is true.

**Staged bytes.** I re-hashed every file in the 156 variant directories of `data/runs/a5/part_a/qwen3.5-0.8b/` (`rehash.py`, 4 processes, 1,021 s; 468 files including `edit.json`). For the forty:

| set | files | equal `manifest_pre_f11.json` | equal `manifest.json` | equal `supplement_freeze.json:staged_byte_inventory` |
|---|---|---|---|---|
| `delta.safetensors` / `increment.safetensors` of the 111 staged variants (80 ladders, 31 rotations; 18 with increments) | 129 | 129 | 129 | 129 (inventory has 129 files) |
| `variant.json` of all 120 variants (incl. the 9 excluded rotations) | 120 | 120 | 120 | – |
| **total** | **249** | **249** | **249** | – |

**The forty's staged bytes are byte-identical to their pre-F11 record, the merged manifest and the pre-execution freeze.**
Further checks:

- From my hashes I recomputed `rehash_staged`'s `by_variant_sha256` over the forty's 120 variants: `fab1b8a4…f952ef`. It equals
  the run's recorded `supplement.carried.rehash_before` and `rehash_after`, which are identical to each other (`n_files` 249,
  `all_equal` true).
- The staged-zoo `manifest.jsonl` (sha256 `23efdbdc…74dc`) has 147 rows. Every staged variant's `reuse_key` equals its delta
  sha256, and no excluded variant has a row. The forty's rows re-serialise to `313e0428…24c0`, the recorded
  `staged_rows_sha256`.
- The 9 excluded rotation directories hold only `variant.json`.
- No `.safetensors` file exists on disk outside the manifest's `delta_sha256`. The stage directory has 156 variant directories
  plus `manifest.jsonl` and `stage.json`.

### 4.3 The 36 supplementary variants on disk

For the 36 supplementary variants, all 36 `delta.safetensors` files and all 36 `variant.json` files re-hash to
`manifest.json:delta_sha256` and `variant_json_sha256` (72/72). The roots have no increment file. In total, all 321 manifest-listed
file hashes (165 safetensors, 156 `variant.json`) equal the files on disk. The delta hashes are in the §2 seeds table.

### 4.4 Subsets and the union summaries

`manifest.json:f10_subsets` is recomputed over the union. It has `noise_calibrated` 45 and `pooled_norm_le_2e-2` 41 ids, and
each list includes 11 supplementary ids (all except `aaf7a0660c6d5a73`). Restricted to the forty, both lists equal the pre-F11
lists (34 and 30). These union lists are staging summaries. The frozen F10.5(c)/(e) subsets stay the forty's (F11.5, F10a.4),
anchored to `manifest_a533.json` (`d7785b35…`) by the analysis (re-pin (xxi)). Board A noted that "union staging summaries do not
authorize redefining those frozen subsets". The supplements' predicate values are in the §2 diagnostics table.

## 5. The F11.4 slot mask

- `docs/prereg/a5_manifests/part_a/slot_mask_f11.json`, sha256 **`5f4e770185ec9cc6fa10461d3012196bbd3feb9ca4af9f3065510a634b5bbfc4`**,
  committed in `92b0338` (09:27:54 UTC), before the calibration launch (09:28:04; the launch log prints the same hash).
- **Schema-only:** exactly the keys `available` and `schema` (`a5-slot-mask-v1`).
- **Built by the accepted builder from the manifest.** I ran `response_intervention.slot_mask(manifest.json)` and
  `slot_mask_text` from the pinned `response_intervention.py` (`e52ea009…`) in memory. The rebuilt text is byte-identical to the
  committed file. The builder checks every pair verdict against its rotation variant's `excluded` flag.
- **Mapping:**
  - 52 ids. `rotation` is listed for 43; `ladder_0.25` and `ladder_0.5` are listed for all 52.
  - Masked rotation slots: the 9 original exclusions only. No ladder slot is masked.
  - The 40 original entries are identical to `slot_mask_f10.json`.
  - Per cell (ids / rotation / ladder 0.25 / ladder 0.5): lora_sft:0 10/10/10/10, lora_sft:1 10/10/10/10, lora_sft:2 14/11/14/14,
    chain:0 8/5/8/8, chain:1 3/3/3/3, chain:2 7/4/7/7.
- Canonical availability digest (`sha256(json.dumps(available, sort_keys=True))`, the simulator's `mapping_sha256` definition):
  **`171dfd20b47020dbc172d39eac87adf5c6efbe7864a0f3a784ef018fcb19741a`**. The calibration output reports the same
  `slot_mask.mapping_sha256`.
- **Ordered slot table.** The simulation review's condition 2 asked for the explicit slot table to be saved as
  `part_a/slot_mapping_f11.json` and hashed before the run. **That file does not exist.** The ordered table was first emitted
  inside the calibration output (`slot_mask.slots` and `supplement.slots`, §6), which is after the run. That table has slots 0–39
  registered in frozen A5.3.1 order and slots 40–51 the supplements in cell order. Each slot's `available` equals the frozen
  mask. The table is fully determined by the frozen `selection_f11.json`, the payload and the simulator code. It was nevertheless
  not frozen as a separate artifact (§10).

## 6. F11.4 calibration

Sources: the agent-tmux log `/home/ubuntu/agent-logs/a5-f11-calibration-20261111-092804.log`, the launch script
`/tmp/a5_f11_calibration.sh` (809 bytes, sha256 `00c3a8bb46acbd1e34f095deba6b8a5f58167060e1c4a8414b70ae23de91b19b`,
mtime 09:28:03.72), and the output `data/runs/a5/power/power_slotmask_f11.json`.

**Execution.**

- agent-tmux `a5-f11-calibration`. The log records `START 2026-11-11T09:28:04Z HEAD 92b03384d717065405c341d301bf6ba754783812
  dirty 0 mask 5f4e7701…`, `END 2026-11-11T09:34:03Z` and `CALIBRATION_EXIT 0` (wall 5 min 59 s). cwd
  `/home/ubuntu/projects/wsp-lab`.
- The command is string-identical to the inner command of Board A's frozen calibration command (`board-a-f11-sim-2026-11-11.md`;
  the `bash -lc 'cd … && …'` wrapper is replaced by the launch script, which does `cd` and records start, SHA, dirty count, mask
  hash, end and exit):

```
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/projects/wsp-lab/.venv/bin/python scripts/a5_simulation.py data/runs/a5/nullsize/inputs.json --fields data/runs/a5/power/fields.json --donor-ids docs/prereg/a5_manifests/power/registered_donors.json --supplement docs/prereg/a5_manifests/selection_f11.json --slot-mask docs/prereg/a5_manifests/part_a/slot_mask_f11.json --scenario registered --output data/runs/a5/power/power_slotmask_f11.json
```

- Execution HEAD `92b0338`, clean. `scripts/a5_simulation.py` at `92b0338` is `4ce3faa5…` and `tests/test_a5_simulation.py` is
  `d231c8b0…`, the simulation review's pins. All 71 pins of `92b0338`'s `code_hashes.json` match. Python 3.12.3, NumPy 2.5.4 (the
  `.venv`), threads 4 (command).
- Output: `data/runs/a5/power/power_slotmask_f11.json`, 241,039,486 bytes, sha256
  **`8a1be18f726d5b030dbaa5937dcc63ba9a3ca9f730d314f689a4fbb7631441fb`** (re-hashed).
- No other `a5-f11-calibration` job or output exists. This is the one registered-scale run. No seed, replicate or mask was changed.

**Input hashes.** My re-hashes equal the output's `input_provenance` and the simulation review's required values:

| input | sha256 |
|---|---|
| `data/runs/a5/nullsize/inputs.json` | `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de` |
| `data/runs/a5/power/fields.json` | `85843ed28d3fd3ba5dc71e2ba8943fd022924cbc4f0b62dddbfb001cb2bcf3b9` |
| `docs/prereg/a5_manifests/power/registered_donors.json` | `6e77bca615624a04cb43b3d11a035b5dc31e5a2f1cd918d42a35cc7786babbb5` (61 ids) |
| `docs/prereg/a5_manifests/selection_f11.json` | `ce746e7135dd94b087da86953beb02382e5706f8ff2dd17bd57c053aaf54793c` (output `supplement.sha256`; `ids_sorted_sha256` `1848fe4c…`, k 12, `n_slots` 52) |
| `docs/prereg/a5_manifests/part_a/slot_mask_f11.json` | `5f4e770185ec9cc6fa10461d3012196bbd3feb9ca4af9f3065510a634b5bbfc4` (output `slot_mask.sha256`; `mapping_sha256` `171dfd20…`) |

**Settings.** The output has label `slot-masked (F10.6); supplemented (F11.4)`, `scenario` `registered`, seed 20261107,
10,000 replicates, 4,000 resolution draws, minimum resolved 15 and registered alternative 0.75. The following fields are
**equal** to those of the F10.6 run (`power_slotmask_f10.json`): `stream_order` (panels, signs, resolution), `quotas`,
`frozen_cell_cutpoints`, `seed`, `resolution_draws`, `noise_scale`, `unavailable_audit`, `proxy`, `donor_cluster_members`,
`donor_pool` and `scenario`.

**Available slots per contrast** (`slot_mask.available_slots`): rotation **43**, ladder 0.25 **52**, ladder 0.5 **52**.

| cell | slots (registered + supplement) | rotation | ladder 0.25 | ladder 0.5 |
|---|---|---|---|---|
| LoRA:0 | 10 (10 + 0) | 10 | 10 | 10 |
| LoRA:1 | 10 (10 + 0) | 10 | 10 | 10 |
| LoRA:2 | 14 (10 + 4) | 11 | 14 | 14 |
| chain:0 | 8 (4 + 4) | 5 | 8 | 8 |
| chain:1 | 3 (3 + 0) | 3 | 3 | 3 |
| chain:2 | 7 (3 + 4) | 4 | 7 | 7 |

Masked ids: the 9 original rotation exclusions; no ladder masked.

**NULL-SIZE (full registered payload, F7.2 and F11a.2).** `donor_pool.applies_to` is `power_yield` only. The NULL-SIZE pass
draws from the payload `inputs.json`, which has 141 donors. The output's `supplement.donor_overlap` lists all 12 supplements as
present in the NULL-SIZE pool and in the POWER pool, with self-membership kept and no leave-one-out (policy text as in F11a.2(c)).
The full `reports.null_size` block without its traces:

| field | value |
|---|---|
| decisions go / alternative / unresolved | 175 / 2576 / 7249 |
| `ordered_go_rate` (CP95) | **0.0175** (0.015021–0.020265) |
| `two_sided_rejection_rates` rotation / ladder 0.25 / ladder 0.5 | **0.0348 / 0.0378 / 0.0378** |
| `two_sided_cp95` | [0.031294, 0.038580] / [0.034147, 0.041725] / [0.034147, 0.041725] |
| `contrast_rejection_rates` | 0.0175 / 0.019 / 0.019 |
| `positive_rejection_cp95` | [0.015021, 0.020265] / [0.016415, 0.021870] / [0.016415, 0.021870] |
| `yield_rates` (CP95) | 1.0 / 1.0 / 1.0 ([0.99963, 1.0] each) |
| `conditional_coverage` | 0.93621 / 0.93561 / 0.93561 |
| `audit_or_nonfinite_counts` | 90,000 / 0 / 0 (9 masked rotation slots × 10,000) |
| `accepted` | true |
| funding-only alternative rate | **0.2576** (`gates.size.alternative_rate`; not size-gated, F5.7) |

For comparison, the forty-slot F10.6 run had NULL-SIZE decisions 192 / 1939 / 7869 (alternative rate 0.1939), and the F11
review's 500-replicate planning value at 52 slots was 0.260. The 52-slot alternative rate 0.2576 is reported beside ordered go, as
that review requires. It is not a size failure and was not tuned.

**POWER-AND-YIELD (61-donor pool; `donor_pool.ids_sha256` `52366da3…107f1`).** The full `reports.power_yield` block without its
traces:

| field | value |
|---|---|
| decisions go / alternative / unresolved | 8722 / 0 / 1278 |
| **`ordered_go_rate`** (CP95) | **0.8722** (**0.865498–0.878685**) |
| `two_sided_rejection_rates` rotation / ladder 0.25 / ladder 0.5 | 0.8722 / 0.9272 / 0.9278 |
| `two_sided_cp95` | [0.865498, 0.878685] / [0.921932, 0.932217] / [0.922552, 0.932798] |
| `contrast_rejection_rates` | 0.8722 / 0.9272 / 0.9278 |
| `yield_rates` | 1.0 / 1.0 / 1.0 |
| `conditional_coverage` | 0.93458 / 0.93381 / 0.93392 |
| `audit_or_nonfinite_counts` | 90,000 / 0 / 0 |
| `accepted` | true |

The F10.6 forty-slot run, for reference, had ordered power 0.7514 [0.7428, 0.7598].

**Gates (F11.4, F11a.1, and the output's `slot_mask.gates`):**

| gate | threshold | value | result |
|---|---|---|---|
| ordered go power (point rate) | ≥ 0.80 | 0.8722 (CP95 lower 0.8655) | **pass** |
| size: ordered go | ≤ 0.07 | 0.0175 | **pass** |
| size: two-sided rotation / ladder 0.25 / ladder 0.5 | ≤ 0.07 each | 0.0348 / 0.0378 / 0.0378 | **pass** |
| funding-only alternative rate | not gated (F5.7) | 0.2576 | disclosed |

**Verdict: PASS** (`slot_mask.gates.passed: true`; `power.passed` and `size.passed` both true). Under F11a, neither failure branch
applies. The power-only screen route (F11a.1(a)) and the size hold (F11a.1(b)) are both moot. Subject to Board A's acceptance of
this report and the remaining gates, F11.5's decisive route applies: the registered Part A is the supplemented 52-component set,
with separate per-contrast availability. The registered-40 sensitivity was not run here; it has no decision authority (F11.5,
F11b.5).

Two points on dependence and support, as the F11 review requires. The supplements' own donor rows are drawable in both pools, so
the calibration is the registered empirical development proxy. It is not independent validation of these twelve edits or of
audit-selected roots. The proxy residuals are original-minus-predecessor. They are not true rotation- or strength-specific noise.

## 7. Proofs deferred and price

Source: `manifest.json:scoring` and `supplement.reference_proofs`.

- `supplement.reference_proofs`:
  - `computed: false`
  - `deferred_to` `score --manifest (score_reference_proofs)`
  - `paths_touched` `[]`
  - `identity` null
  - carried: `base (predecessor of every supplementary edit)`
- **The 12 supplementary originals are unproven** (`unproven_originals` = `originals_to_prove_or_score`): `15cdcf0bb126de17`,
  `17668df17662b3f7`, `1ac30f3c7a209fe6`, `2410c595817dc20b`, `342a1be080049e8f`, `388347cfb0b7fafa`, `45cd0be9fea1c139`,
  `8d29ac2adb7fdda0`, `aaf7a0660c6d5a73`, `af6d608f8651fb46`, `b5d0ac5d76c3cad6`, `c12f6ec6bad59ba3`.
  - None is among the 49 carried `reuse_proofs`.
  - `aaf7a0660c6d5a73`, the re-trained lineage that F11.8 names, is unproven like the others.
- `unproven_rule`: "each unproven original is priced as one entry to be proven or scored; score --manifest runs the reuse proofs
  first (proven: reused; failed: scored as a reference) and refuses to score any variant while its original stays unproven".
- **Priced count.**
  - `n_variants` 147 (111 + 36), `n_D` 19 (the original missing originals, unchanged), `n_P` 0, `priced_unproven_as_references` 12.
  - `n_priced` **178** = 147 + 19 + 12; `references_to_score` 19.
  - `minutes_per_entry` 1.04, so `minutes` **185.12**.
  - Pre-F11: 111 variants, n_D 19, 130 entries, 135.2 min.
  - If all twelve proofs succeed, the price would be 166 entries, 172.64 min. That is the F11.8 planning total, not a certified
    figure. `n_D` 19 does not prove the other twelve are reusable.
- The proof step is a later, separately gated scoring-phase action. Parsing `eval.json`/`records.jsonl` is exposure (Board A
  staging review). It runs only after the calibration-route acceptance, the analysis gates and the final exposure freeze, as the
  first authorized scoring-phase action. No proof read occurred in staging or for this report.

## 8. Exposure

- Nothing has been scored: `manifest.json:scored` is false, as are `manifest_pre_f11.json` and the staging `evaluation_reads` record
  ("none: no evaluation record opened, no scorer run").
- The calibration is a simulation on frozen development inputs. It reads no G3, rotation, ladder or reference outcome.
- Path checks for this report used `os.path.exists` only, under `data/evals/qwen3.5-0.8b/tf_v1` (`scoring.evals_dir`). Nothing
  under `data/evals` was opened, listed or read:
  - 156 variant ids (147 staged and 9 excluded): **0** have a path;
  - the 19 missing originals (`scoring.missing_originals`): **0** have a path;
  - the 12 supplementary originals: **12 of 12 have a path**. This is expected, because they are development originals with
    TF-v1 evaluations; that is why they are reuse candidates. Existence was tested; contents were not opened.
- The staging used `--no-reference-proofs` and did not pass `--evals` (§1.1).
- `scored: false` and absent outcome paths do not by themselves prove anyone's exposure history. The per-decision-maker
  attestations in `docs/prereg/f10_attestations.md` remain the record for that.

*Report author's exposure attestation.* Claude Opus 5.5, implementation sub-agent, worktree `report/f11-staging`
(`.worktrees/report/f11-staging`), 11 Nov 2026 from about 09:28 UTC to this report's commit. I read governance and review
documents, manifests, audits, freeze and launch records, logs, staged variant bytes and metadata, and the F11.4 simulation output
and its inputs. I re-hashed staged files on CPU. I did not open, compute, score or receive any G3, rotation, ladder,
missing-reference or sealed response outcome or summary. I opened nothing under `data/evals` (existence checks only), used no GPU,
and wrote nothing under `data/`. This covers this session only.

## 9. Stop-and-reprice status

- Board A staging review: "The code automatically adds failed proofs to reference commands; it does **not** implement F11.8's
  stop-and-reprice approval itself." The required sequence before any GPU scoring has not started. That sequence is: a proof-only
  phase (`score_reference_proofs(..., write=True)`) after all methodological holds, keeping its manifest-bound sidecar; then
  acceptance of the actual repriced plan; then the GPU launch, with any change in a recomputed proof stopping it.
- No proof has run. The conservative price stands at 178 entries, 185.12 min (§7).
- A main commit after this staging, `1b954a6` (09:29:04 UTC), adds a Part A scoring launch script described as "gated; re-hash,
  clean-tree refusal, proof step, timing record; not executed". It is outside this report's scope and is not reviewed here.
- No scoring release follows from the staging or the calibration pass. The open gates are:
  - Board A acceptance of this report and of the calibration result;
  - the analysis gates (F11b; Board A analysis review 4 conditions 2–3);
  - the final exposure freeze and attestations;
  - the proof-only, record and reprice sequence;
  - verification of the frozen manifest hash and the staged-zoo bytes at scoring launch.

## 10. Items for the chair and Board A

1. **Freeze mismatch: none** (§1.5). There is also no mismatch in inputs or pins.
2. **Manifest provenance records the end-of-run HEAD.** `manifest.json:code_sha` / `provenance.git_sha` is `887a514`, not the
   execution SHA `8211ecf` (§1.2). The changed files are analysis code and documents that the run neither imported nor read. The
   A5.1.2 record should name `8211ecf` as the execution SHA.
3. **Selection builder not pinned.** `scripts/a5_f11_supplement.py` and `tests/test_a5_f11_supplement.py` are absent from
   `code_hashes.json` at `8211ecf` (Board A staging condition 1). Their bytes equal the reviewed `90dcb312…` / `6cf4f07c…` (§1.3).
4. **Launch record and wrapper.** `supplement_launch.json` lacks argv, environment, job/log identity and output hashes. The wrapper
   `/tmp/a5_f11_supplement.sh` was not frozen with its hash before execution (staging condition 2; §1.6). This report supplies
   those items after the fact. Re-pin (xx)'s "81-file" inventory wording is wrong: the freeze holds 129 files (§1.4).
5. **Calibration ordering.** The F11.4 run was launched at 09:28:04, before this staging report existed and so before its
   acceptance. The simulation review's condition 2 and the F11 review's condition 3 place report acceptance before the calibration.
   No `slot_mapping_f11.json` was frozen; the mask and its canonical digest were (§5). The run used Board A's frozen command and
   inputs, unchanged, and is the single counted run. Whether to ratify the order is for the chair and Board A. Re-running is not
   permitted (no rerun-to-pass), and none is proposed.
6. **Calibration verdict: PASS** (0.8722; sizes ≤ 0.0378). It takes effect only on Board A's acceptance (§6).
7. **Baseline note.** Leaving the thread variables unset does not reproduce `d1b4e24c…` (introduction). This concerns only the
   inherited F10 freeze record.

## Hashes of this staging's artifacts

| file | sha256 |
|---|---|
| `docs/prereg/a5_manifests/part_a/manifest.json` | `71e77a3e1f703451864ce960129399be1579d8aa84631bebc0c8d6324b4790d6` |
| `docs/prereg/a5_manifests/part_a/rotation_audits.jsonl` | `9412eb0d4d60f5587c4c5d19e9afcda86c6c45faa38a37b3ab937008d505c45a` |
| `docs/prereg/a5_manifests/part_a/manifest_pre_f11.json` | `59d839e351824fa154a665fcf105284ce5fdeed8423a5c0a2b7a60687b0d3b06` |
| `docs/prereg/a5_manifests/part_a/rotation_audits_pre_f11.jsonl` | `4ec0928a91b5523a400b911f00f8045f267386c2589aa72a58816d565f5c97f3` |
| `docs/prereg/a5_manifests/part_a/supplement_freeze.json` | `503e415efda3801216cc57ba332ab643a333e4b73cb92ce6366add0682fda0a4` |
| `docs/prereg/a5_manifests/part_a/supplement_launch.json` | `d17fab2f7648a405045b85bcb92e309a5ea11fb19a6d864a852ccb72bd38fb38` |
| `docs/prereg/a5_manifests/part_a/slot_mask_f11.json` | `5f4e770185ec9cc6fa10461d3012196bbd3feb9ca4af9f3065510a634b5bbfc4` |
| `data/runs/a5/power/power_slotmask_f11.json` (not committed; under `data/`) | `8a1be18f726d5b030dbaa5937dcc63ba9a3ca9f730d314f689a4fbb7631441fb` |
| `docs/prereg/a5_manifests/selection_f11.json` | `ce746e7135dd94b087da86953beb02382e5706f8ff2dd17bd57c053aaf54793c` |

This report's own hash is in `SUPPLEMENT_A512_NOTE.md`.
