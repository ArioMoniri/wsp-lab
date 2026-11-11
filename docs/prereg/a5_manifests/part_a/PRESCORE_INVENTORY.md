# Part A pre-scoring inventory (11 Nov 2026; scoring still HELD)

This is the single committed inventory that every Part A scoring-release gate points to (F11b.6–F11b.7; Board A analysis-4
condition 3; Board A F11 condition 6; Board A F10 condition 7). It releases nothing. The machine-readable record is
`PRESCORE_INVENTORY.json`, next to this file. The JSON holds every hash, every check with its evidence, the full staged-byte list
(165 files) and the gate table. Every number below comes from a cited file or from a re-hash made in this session.

- **Built:** 2026-11-11T09:53:00Z, worktree `.worktrees/prereg/prescore-inventory`, branch `prereg/prescore-inventory`, HEAD
  `079e60b0d65da1f893d726c80acb9d2a0874acba` (main after the calibration-record commit).
- **How it was built:** CPU only. The data were read only. Under `data/evals` there were existence checks only.
- **Builder:** session-scratch scripts, not committed. `build.py` sha256 `8144f3ed0ff091618771a22c5d30a2de7c3cc8738080708bb3542629a1ad91aa`;
  `merge.py`, the gate table, attestation checklist and findings, hand-authored; `rehash_sample.py` sha256
  `19593831edb5fc7e15d0870ad1c8cd5895bf87f330a75e9856c0848269f61d59`.

## 1. Population (checks A1–A10)

| item | value | source / check |
|---|---|---|
| Execution manifest | `part_a/manifest.json` sha256 `71e77a3e1f703451864ce960129399be1579d8aa84631bebc0c8d6324b4790d6`; 52 edit ids, 52 selections, `coverage.n_edits` 52 | A1 |
| `supplementary: true` | on exactly 12 selection rows and the same 12 edit-audit rows; on no variant row | A2 |
| The 12 ids | `15cdcf0bb126de17 17668df17662b3f7 1ac30f3c7a209fe6 2410c595817dc20b 342a1be080049e8f 388347cfb0b7fafa 45cd0be9fea1c139 8d29ac2adb7fdda0 aaf7a0660c6d5a73 af6d608f8651fb46 b5d0ac5d76c3cad6 c12f6ec6bad59ba3` = `selection_f11.json:supplement_ids` = `manifest.supplement.ids` | A3 |
| Sorted-id sha256 of the 12 (`sha256(json.dumps(sorted(ids)))`) | recomputed `1848fe4cacbaacc9f643a179835770fefd3027a48cef7cf9d83358f15b85fdd9`; equals `selection_f11.json`, `manifest.supplement.ids_sorted_sha256` and `supplement_freeze.json` | A4 |
| Allocation | `lora_sft:2` 4, `chain:0` 4, `chain:2` 4; all predecessor `base` | A5, A6 |
| Components | 52 distinct `component_id`; the 12 supplements are disjoint from the forty by id and by component | A7, A9 |
| Registered 40 | sorted-id sha256 recomputed `e2bd6cb2956c0fbe64194b3f1ebbb481248ecec45b8a729a59a54b9a2bac9d6b`; equals `selection_ledger.edit_ids_sorted_sha256` in manifest, pre_f11 and a533, and `part_a_analysis.py:REGISTERED_EDIT_IDS_SHA256` | A8 |
| Supplementary variants | 36 (12 rotation, 24 `scale`); none excluded. All 12 supplementary rotations pass F10 | A10 |

## 2. Preservation of the forty (checks B1–B6)

- **No registered row was altered relative to `manifest_pre_f11.json`** (B1). For every pre-F11 key of `delta_sha256` (111),
  `variant_json_sha256` (120), `edit_audits` (40), `rotation_seeds` (40), `selections` (40) and `variants` (120), the current value
  is equal. Every addition (36 / 36 / 12 / 12) belongs to a supplement. `restage`, `selection_ledger`,
  `realised_increment_ladder_variant_ids` (16) and `excluded_variant_ids` (9) are equal.
  - *Disclosure:* `f10_subsets` is a union staging summary. Eleven supplements are appended to each list, and no registered
    member is removed. The registered members hash to the frozen S30 `3da5f44f…7267c` and S34 `fb1c34e6…aedfa0`
    (`part_a_analysis.py`). The analysis uses those frozen lists, and supplements never enter them (F11b.1(d)).
- **The forty's selection rows equal their a533 rows** (B2).
- **Audit file** `rotation_audits.jsonl` (`9412eb0d…5c45a`) begins with the exact bytes of `rotation_audits_pre_f11.jsonl`
  (2,534 rows) and adds 535 supplementary rows, for 3,069 in total (B3).
- **Nine original rotation exclusions, unchanged, with reasons** (B4; edit → rotation variant: reason × tensors):

  | edit | rotation variant | reason |
  |---|---|---|
  | `0027efdf2eae7405` | `8506c63c5b858637` | replay_failed ×13 |
  | `05531ce1db4df63f` | `ffe89cadb99d110c` | replay_failed ×24 |
  | `057dcfbedea3251f` | `0bcb2f48c5fcbcd2` | replay_failed ×24 |
  | `0bc0a0c4ad0520e0` | `7cb4ef4f618014cc` | replay_failed ×9 |
  | `0d055869527177b4` | `716996040438aed6` | replay_failed ×1 |
  | `14cdaabee246493b` | `0824ab1a336c0c19` | replay_failed ×5 |
  | `15a53586efc72a50` | `ed51a348b7fd834b` | rounding_bound_failed ×1 |
  | `2233954b0cc3edc0` | `417883272c745e7e` | rounding_bound_failed ×1 |
  | `72d3756e968e3209` | `9ebbe197b37dfdfd` | rounding_bound_failed ×1 |

- **Rotations and ladders of the forty** (B5): 31 rotations staged and 80 ladders (kind `scale`), none excluded. The ladders'
  delta hashes equal `manifest_a533.json`'s, so the ladders were not rebuilt. There are 16 `realised_increment_ladder` flags.
- **Staging re-hash of the forty** (B6): before = after, 249 files, 120 variants, `all_equal` true, by-variant sha256
  `fab1b8a4…f952ef` (`manifest.supplement.carried`).

## 3. Preserved manifests and audits; masks; calibration (checks C1–C4, D1–D3, E1)

| file | sha256 | note |
|---|---|---|
| `part_a/manifest_a533.json` | `d7785b35de1d7041edd2db139591562379db12e33c2532921c814735e8dde8d6` | = `FROZEN_STAGED_MANIFEST_SHA256` |
| `part_a/rotation_audits_a533.jsonl` | `f05fec4265ffd93c8c537100993d23262be95141369240fcad0a7e16ff474d35` | |
| `part_a/README.md` (a533) | `2fcbf517ccf26f2a1fe325ef31300d291bc0f254c7faaacd18154e327bceada1` | |
| `part_a/manifest_pre_f11.json` | `59d839e351824fa154a665fcf105284ce5fdeed8423a5c0a2b7a60687b0d3b06` | = `selection_f11.json:inputs.part_a_manifest` |
| `part_a/rotation_audits_pre_f11.jsonl` | `4ec0928a91b5523a400b911f00f8045f267386c2589aa72a58816d565f5c97f3` | |
| `part_a/RESTAGE_REPORT.md` | `61294c8666b4bbe4f938e76af7797631ef86613e77d431229bddcaf69d6d5f61` | accepted with conditions (`board-a-f10-restage-2026-11-11.md`) |
| `part_a/slot_mask_f10.json` | `86fa27a5c1838d0c0bd88ef530feba22eceafac698d0ec4506fbf05df83b698e` | mapping recomputed from pre_f11: `b7724a69315d8c8692a128504126251ac16945cf7b741ec60b3cdb6daa5cb329` = record (C2) |
| `part_a/slotmask_calibration_record.json` (F10.6) | `0c5fa8c38da42d63bd5762c307ec1c2254313501fbfd4763b3418eaa3eff6b51` | **FAIL**, power 0.7514; superseded by F11 (D3) |
| `data/runs/a5/power/power_slotmask_f10.json` | `b14756828bc27cbbe14e1e590226820371a65ef5f83ad9622b2e6ccf04a3450a` | re-hashed = record |
| `part_a/slot_mask_f11.json` | `5f4e770185ec9cc6fa10461d3012196bbd3feb9ca4af9f3065510a634b5bbfc4` | 52 ids; recomputed from the execution manifest: rotation 43, ladder_0.25 52, ladder_0.5 52; mapping `171dfd20b47020dbc172d39eac87adf5c6efbe7864a0f3a784ef018fcb19741a` = record = output (C1) |
| `part_a/f11_calibration_record.json` (F11.4) | `216bdc9c578e576cf67ea1dfc85a1a89850c69c6c9c39fbb4fc5b8fcda26f5a4` | **PASS** (D1, D2) |
| `data/runs/a5/power/power_slotmask_f11.json` | `8a1be18f726d5b030dbaa5937dcc63ba9a3ca9f730d314f689a4fbb7631441fb` | re-hashed = record; its `gates` block equals the record's |

**F11.4 result** (`f11_calibration_record.json`):
- Gates: ordered power **0.8722 ≥ 0.80**; ordered-go size **0.0175**; two-sided **0.0348 / 0.0378 / 0.0378 ≤ 0.07**. The
  alternative rate 0.2576 is reported and not gated.
- Available slots: rotation 43, ladder 52/52.
- Run: execution SHA `92b0338`, dirty 0, 09:28:04–09:34:03 UTC; supplement file `ce746e71…4793c`.

**Slot table.** The output's `slot_mask.slots` has 52 rows. The cells are LoRA 10/10/14 and chain 8/3/7, and slots 40–51 are the
12 supplements in selection order. Its sha256 (`json.dumps(slots, sort_keys=True)`) is
`68b8b01db16f9c267943e160fafbd8e6aecb15fa5d820f3eae62a5ecd6b4620e` (C3).
- **Missing:** `part_a/slot_mapping_f11.json`, which `board-a-f11-sim` condition 2 required to be saved and hashed before RNG
  execution, does not exist (C4). The slot table is evidenced only after the run, in the output.

**Inputs, all re-hashed and equal to their recorded identities** (E1; 28 compared). The recorded identities come from A5.1.2,
`selection_f11.json:inputs`, `supplement_freeze.json:inputs`, the board-a-f11-sim command table, `part_a_analysis.py` constants
and the calibration records. The five `selection_f11.json` inputs match their live files (E2).

| file | sha256 |
|---|---|
| `selection.json` | `393530365f87f981d5da3ecb2779c89b8e7bf98105b6c45d1824bf290e908c63` |
| `selection_f11.json` | `ce746e7135dd94b087da86953beb02382e5706f8ff2dd17bd57c053aaf54793c` |
| `references.json` | `5b6991f299c1eee049de51bc1e4f00c71c1dfc58114cdf6549fb5d246850564e` |
| `text_groups.json` | `f769afcade5a80571daf3a9578b1c03584b46a5abbbba07b32d43a7e9035e7e1` |
| `control_seeds.json` | `d357ed51031877f817baeb4204cf93b38c66fdfcfd95bd41803d93d5160c12d5` |
| `panels.json` | `c291b68516952c4f342bfc7f36843a83f0482df7d721a00e46287597bad108cd` (no prior recorded value compared) |
| `components.jsonl` | `6b4eff8d61a4b1c4d83f0c058cb4df5c6c5398e2c6656a8b58b38f8b46251dec` |
| `nullsize_inputs.json` | `6e58b589e32491b36fd70e679b20242538ea8e958b9b9072ce0e30b444508885` |
| `nullsize/report.json` | `1bed2c9531e4230864a1ebbe7e4023e0ffafcf704a59c40e93f9a75bd469dfa6` |
| `nullsize/inputs_provenance.json` | `ea160aa4088c56db0aa27046daf3e21eb5865b3950b98e70b4eff9242e1eb1a3` |
| `nullsize/validation.json` | `7a9f72876e73f5ffba2852f925a8aa3ae52a5c9261d55af942b322b1e0606136` |
| `data/runs/a5/nullsize/inputs.json` (NULL-SIZE payload) | `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de` |
| `data/runs/a5/power/fields.json` (F11.4 `--fields`) | `85843ed28d3fd3ba5dc71e2ba8943fd022924cbc4f0b62dddbfb001cb2bcf3b9` |
| `power/fields_manifest.json` | `19e9de8430ff2a135db848f7d0d5097b5b5724138e9ee815e807e5753e5f9a6b` |
| `power/registered_donors.json` (61) | `6e77bca615624a04cb43b3d11a035b5dc31e5a2f1cd918d42a35cc7786babbb5` |
| `power/report.json` | `ff46de30fa8e4ff97a4034e8bdef78cb19bc49662073523500a43c773b55845a` |
| `docs/design/F10_SENSITIVITY_SPEC.md` | `8624e06343bbb401ff3b6d783572b0c53e1172d0ea3f36aec3768a13701998d4` |
| `configs/eval_tf_v1.yaml` | `5c20cabe8e35b4116adeed4df650cf27461e7ab144ef66c0b60c397b13b00317` |
| `part_a/supplement_freeze.json` | `503e415efda3801216cc57ba332ab643a333e4b73cb92ce6366add0682fda0a4` (= `supplement_launch.json:freeze_sha256`) |
| `part_a/supplement_launch.json` | `d17fab2f7648a405045b85bcb92e309a5ea11fb19a6d864a852ccb72bd38fb38` |
| `part_a/restage_freeze.json` / `restage_launch.json` | `03dfaf26…48858` / `28dcba68…59a7e` |

## 4. Pins (check F1)

- `docs/prereg/a5_manifests/code_hashes.json`: sha256 `3447730883fe30306c9615418fcefb50b03dfbe78d05f6cc4d2ba5315e3b7bdb` (= re-pin
  (xxi)), 71 pinned files, `git_sha` `ade2850…`. Package README `bb6173b11aee2322…` (= (xxi)).
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/a5_build_manifests.py --verify-pins HEAD` at
  `079e60b0d65da1f893d726c80acb9d2a0874acba` printed **"all pins match"**, exit 0.
- This must be repeated at the actual scoring execution HEAD.

## 5. Attestations

`docs/prereg/f10_attestations.md` sha256 `e9a0cb0a30664f81a9aada9691cc1e57ca7ad0abcd841ad154634eed08ad2421`.

| entry | role | status |
|---|---|---|
| 1 | chair | present (to ~01:25 UTC) |
| 1-ext | chair | present to ~09:20 UTC; **does not cover** staging completion (09:26), the F11.4 run (09:28–09:34) or anything later |
| 2–7 | a533 stager, rotation-yield analyst, F10 drafter, F10 code builder, Board A (F10), F10a drafter | present |
| 8, 9 | F10.5 spec drafter, slot-mask path builder | present |
| 10–13 | metrology member, F11 drafter, chair's sizing runs, F11a drafter | present (Board A F11 incorporated by reference) |
| 14 | F11b drafter | present |

**Still pending in the file:**
- the simulation `--supplement` implementer (54f17fc);
- the `prepare --supplement` implementer (c914270/bef5396);
- the analysis F11-path implementer (f57b70a);
- the F11.3 stager and F11.4 operator (a chair extension beyond ~09:20 UTC);
- the scoring-launch-script author (1b954a6);
- this inventory session (proposed text in `PRESCORE_A512_NOTE.md`);
- the Board A re-stage, F11-stage, F11-sim and analysis-4 reviewers. Their lines are in their review files but are not entered or
  incorporated by reference in the attestation file.

**New attestations needed for the scoring phase:**
- **chair:** through the final methodological freeze and the release decision;
- **scorer session:** at launch. The proof step reads the twelve originals' TF-v1 `eval.json`/`records.jsonl`, which is exposure
  under the stage review, so it must follow the final freeze;
- **Board A:** with its acceptance of the staging and calibration reports.

## 6. Unproven originals and the proof step (check H1)

- `manifest.scoring.unproven_originals` = `originals_to_prove_or_score` = the 12 supplementary ids; `priced_unproven_as_references`
  is 12.
- `manifest.supplement.reference_proofs`: `computed` false, `paths_touched` [], `deferred_to` "score --manifest
  (score_reference_proofs)". So no proof read has happened.
- The 40's 40 reuse proofs are carried, `window_check` true.
- Rule (`unproven_rule`): score --manifest runs the proofs first. A proven original is reused, and a failed one is scored as a
  reference. It refuses to score any variant while its original stays unproven.
- Required sequence (board-a-f11-stage condition 5 and "Price and stop-and-reprice"): a proof-only call
  (`score_reference_proofs(..., write=True)`) after the final freeze → retain the record → accept the repriced plan → then GPU
  scoring. **`scripts/part_a_score_launch.sh` does not implement this stop (LS-1, §10).**

## 7. Priced count (check I1)

- `manifest.scoring`: n_variants **147**, references_to_score **19** (= n_D 19; n_P 0), unproven originals priced as references
  **12**. n_priced **178** = 147 + 19 + 12; minutes **185.12** = 178 × 1.04 (conservative).
- At all twelve proofs succeeding: 166 entries, 172.64 min (F11.8).
- Retries are **unpriced**.
- Stop-and-reprice: a failed proof adds that original at 1.04 min. It never changes the twelve ids. The repriced plan is accepted
  before GPU commands. Exceeding a cap stops and reprices and never silently cuts coverage (A5.3.4, F11.8, board-a-f11-stage).

## 8. Staged bytes (checks J1, J2)

- `manifest.delta_sha256` lists **165 files**: 147 `delta.safetensors` and 18 `increment.safetensors`, under
  `data/runs/a5/part_a/qwen3.5-0.8b`. All 165 are present, 116,715,970,712 bytes in total. Each path and hash is in the JSON
  (`staged_bytes.files`).
- **Re-hash now** (09:37:40–09:47:47 UTC, CPU, agent-tmux `prescore-rehash`): a deterministic sample of every 7th of the 165
  sorted (variant, file) pairs gives **24 files**, 4 of them supplementary and 4 increments, 17,475,728,968 bytes. **24 / 24
  equal.**
- All **156 / 156** `variant.json` files are equal to `variant_json_sha256`.
- `scripts/part_a_score_launch.sh` re-hashes every manifest delta/increment file before `score` and refuses on a mismatch. Its
  gaps are in §10 (LS-2, LS-3).

## 9. GPU identity; CPU-only preparation; exposure (check L1)

- **Expected GPU** (`nvidia-smi --query-gpu=name,driver_version,memory.total`): `NVIDIA RTX PRO 4500 Blackwell Server Edition,
  595.112.07, 32623 MiB`; UUID `GPU-3669f1b9-ff73-668a-ad0e-a3cb83c9be8c`. At 09:39 UTC no compute process was running and 0 MiB
  was in use.
- **CPU-only preparation:**
  - the re-stage command ran with `CUDA_VISIBLE_DEVICES=` (`restage_launch.json`);
  - the staging freeze has `cpu_only` true, computed under `CUDA_VISIBLE_DEVICES=""`, OMP/MKL 4 and `HF_HUB_OFFLINE=1`
    (`supplement_freeze.json`);
  - the F11.4 command ran with `CUDA_VISIBLE_DEVICES=` (`/tmp/a5_f11_calibration.sh`);
  - this session's verify-pins, re-hash and build were CPU only.
- **Exposure statement:** `manifest.scored` is false, and no evaluation record has been opened for any variant. For all 156
  variant ids (40×3 + 12×3), `os.path.lexists(data/evals/qwen3.5-0.8b/tf_v1/<id>)` is false for every id. A name walk of
  `data/evals` to depth 4 matches none of them. The `tf_v1/` directory mtime is 2026-11-07T22:09:24Z. Nothing under `data/evals`
  was opened in this session.

## 10. Findings on `scripts/part_a_score_launch.sh` (sha256 `69eae67c…8de41`; identical to `/tmp/a5_part_a_score.sh`)

- **LS-1.** There is no proof-only stop. `score --manifest … --text-groups …` proves and then scores in one call, so a failed proof
  and its new price do not halt before GPU work.
- **LS-2.** The re-hash loop `continue`s on a missing file instead of failing, and it does not re-hash `variant.json`.
- **LS-3.** The script checks neither the manifest sha256 against a frozen value (`71e77a3e…`) nor `--verify-pins HEAD`.
- **LS-4.** `score_launch.json` is written into the tracked tree after the clean-tree check. The GPU UUID is not recorded.

## 11. Disclosures

- **Manifest code SHA.** `manifest.code_sha` and `provenance.git_sha` are `887a5146…`, the canonical checkout's HEAD when the
  manifest was written. The launch was clean at `8211ecf` (`supplement_launch.json`), and main advanced in the same checkout
  during the run. `git diff 8211ecf..887a514` touches only the analysis module, its CLI and tests, the pins/README, PREREG,
  PROGRESS, attestations, the root README and the analysis-4 review. It touches no prepare or rotation code.
- **Calibration wrapper.** The F11.4 wrapper `/tmp/a5_f11_calibration.sh` (sha256
  `00c3a8bb46acbd1e34f095deba6b8a5f58167060e1c4a8414b70ae23de91b19b`) is not committed and was not frozen before the run. Its python
  command equals the Board A frozen command. The record lacks Python/NumPy/thread fields and input hashes; the inputs are
  re-hashed here.
- **Staging launch record.** `supplement_launch.json` lacks job/log identity, argv and output hashes.
- **No staging report yet.** The F11.3 staging report is not committed (PROGRESS 09:34 UTC: "in flight").
- **Field predictions.** No frozen 52-edit field-prediction file for the analysis `--fields` input was located.
  `data/runs/a5/power/fields.json` covers 33 of 52 edits; the missing 19 are the n_D originals.

## 12. Release gates

Status key:
- **MET:** evidence committed and verified here.
- **PARTIAL:** evidence present but short of the condition.
- **OPEN:** missing.
- **BOARD_A:** awaits Board A's final acceptance.
- **AT_SCORING:** can only be met in the authorised campaign.

| release gate | evidence | status |
|---|---|---|
| F10-1 F10a record + attestations | F10a; entries 1–9 | MET |
| F10-2 executable correctness | board-a-f10-code-2 (060cc60), merged | MET |
| F10-3 pins + frozen re-stage | (xviii); restage_freeze/launch | MET |
| F10-4 re-stage report accepted | board-a-f10-restage; chair correction 09:10 | MET |
| F10-5 calibration (yield < 40) | F10.6 FAIL superseded; F11.4 PASS (D1–D3) | BOARD_A |
| F10-6 F10.5 spec frozen | spec `8624e063…`, F10b, analysis-4 | MET |
| F10-7 release after 1–5 and 6 | rows above/below | OPEN |
| F10-7 attestations through final freeze | §5 pending list | OPEN |
| F10-7 / F10C2-4 re-hash staged bytes immediately before GPU | launch script; sample 24/24 now | AT_SCORING |
| F10-7 / F10C2-4 charge references, failures, retries | 178 / 185.12 min; retries unpriced | AT_SCORING |
| F10-7 / F10C2-4 seal ledger before the one analysis | no ledger yet | AT_SCORING |
| F10C2-1–3 merge, (xviii), execution record | (xviii); restage_launch | MET |
| F10C2-4 masked calibration before scoring | F11.4 PASS | BOARD_A |
| F10C2-4 scoring held | `scored` false; L1 | MET |
| F11-1 F11a, list frozen, forty preserved | F11a; A3–A5, B1–B6 | MET |
| F11-2 prepare gate | board-a-f11-stage; (xx) | MET |
| F11-3 staging report accepted | report not committed | OPEN |
| F11-3 sim option gate | board-a-f11-sim; (xix) | MET |
| F11-3 mask + mapping frozen before run | mask yes (92b0338); mapping file absent | PARTIAL |
| F11-3 original reuse proofs | deferred to proof step; 12 priced | AT_SCORING |
| F11-4 one registered-scale run, hashes | record; output re-hashed | MET |
| F11-4 calibration report + route accepted | none on file | BOARD_A |
| F11-5 analysis gate | analysis-4; (xxi); F11b | MET |
| F11-5 / AN4-3 tamper/binding tests on final manifest (synthetic) | not run | OPEN |
| F11-6 attestations through final freeze | §5 | OPEN |
| F11-6 all pins match | 079e60b; repeat at scoring HEAD | MET |
| F11-6 re-hash, charges, no advance scoring, seal | §6–§8 | AT_SCORING |
| F11-6 qualification/field/allocation/reference-reuse gates | 52-edit field-prediction file not located | OPEN |
| ST-1 merge, (xx), verify pins | (xx); launch dirty 0 | MET |
| ST-1 dated F11.8 clarification (deferred proofs, 178, stop-and-reprice) | only a re-pin (xx) mention located | OPEN |
| ST-1 implementer/stager/reviewer attestations | §5 | OPEN |
| ST-2 freeze + launch/end record | freeze OK; launch record lacks log id/argv/output hashes; code_sha 887a514 | PARTIAL |
| ST-3 staging evidence accepted, no replacement | A1–A10, B1–B6 verified; report/acceptance missing | OPEN |
| ST-4 calibration inputs frozen after staging | mask/manifest/audits in 92b0338; mapping absent | PARTIAL |
| ST-5 proof-phase hold; proof-only/reprice before GPU | no proof reads yet; LS-1 | OPEN |
| SIM-1 merge, re-pin | (xix); F1 | MET |
| SIM-2 mask from actual exclusions; `slot_mapping_f11.json` | C1, C3; C4 absent | PARTIAL |
| SIM-3 single frozen command, full record | wrapper uncommitted; env/input hashes missing in record | PARTIAL |
| SIM-3 report accepted | none on file | BOARD_A |
| SIM-4 holds, attestations | §5 | OPEN |
| AN4-1 conventions adopted | F11b | MET |
| AN4-2 merge, re-test, re-pin, verify pins | (xxi); F1 | MET |
| AN4-3 twelve ids, four per cell, uniqueness, forty preserved | A1–A10, B1–B6 | MET |
| AN4-3 a533 + execution manifest hashes | §3 | MET |
| AN4-3 ledger/field predictions, windows, producer/scoring identities | windows/config bound; ledger post-scoring; prediction file not located | OPEN |
| AN4-3 per-contrast masks/mapping, 52-slot report | C1–C3, D1–D2 | MET |
| AN4-3 calibration/route accepted | none on file | BOARD_A |

**Remaining for Board A's final acceptance (BOARD_A):** the F11.4 calibration report and its decisive route, and with them F10-5
and F10C2-4. The OPEN rows also need Board A: the staging report (F11-3, ST-3) and the attestation set.

**Scoring stays held.**
