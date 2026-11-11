# Part A pre-scoring inventory 2: release state after re-pin (xxiv) (11 Nov 2026; scoring still HELD)

This is the updated inventory that Board A asks for in staging-cal condition 7. It releases nothing. It supersedes the status
columns of the first inventory, [`PRESCORE_INVENTORY.md`](PRESCORE_INVENTORY.md) / [`PRESCORE_INVENTORY.json`](PRESCORE_INVENTORY.json)
(sha256 `419ee039de63ea75014b9271e3a2718e593bf9d17b5d83900533511c04100f0c` /
`2ab1caea1dce1f4ac2afb50a59e80d3f3cf3c19ae4a922144a61bbddd9cc2160`, built at `079e60b`). The first inventory stays byte-unchanged as
the historical snapshot. The machine-readable record is [`PRESCORE_INVENTORY_2.json`](PRESCORE_INVENTORY_2.json), which holds every
hash, check and gate row. Every number below comes from a cited file or from a re-hash made in this session.

- **Built:** 2026-11-11, 11:08–11:15 UTC (branch created 11:06:20 UTC). Worktree `.worktrees/prereg/prescore-inventory-2`, branch `prereg/prescore-inventory-2`.
  The branch was created from main `87f6895` and fast-forwarded to main **`c947cc9c7e48118573622507448d213b3648de2a`** before any
  build step. `c947cc9` adds only attestation entry 23. Every hash here is current at `c947cc9`.
- **How it was built:** CPU only (`CUDA_VISIBLE_DEVICES=` for every job). The data were read only. Under `data/evals` there were
  existence checks only.
- **Builders:** session-scratch scripts, not committed. Their sha256 are in the JSON (`builder`).

## 0. Summary

| status | of the earlier 52 gates | new rows (staging-cal (a), (b), 1–7) | total |
|---|---|---|---|
| **MET**: evidence on file and verified here | 31 (was 23) | 2 (SC-a, SC-3) | 33 |
| **BOARD_A**: evidence on file; awaits Board A's explicit acceptance or release | 9 | 5 | 14 |
| **AT_EXECUTION**: can only be discharged in the authorised campaign | 6 | 0 | 6 |
| **PARTIAL**: evidence short of the condition; Board A to rule | 3 | 1 (SC-1) | 4 |
| **OPEN**: missing | 3 | 1 (SC-6) | 4 |

**Awaiting Board A (14):** F10-5, F10-7a, F10C2-4b, F11-4b, F11-6e, ST-5, SIM-3b, AN4-3c, AN4-3e, SC-b, SC-2, SC-4, SC-5, SC-7.
Three things also need Board A's ruling: the PARTIAL rows (ST-1c, SIM-3a, SC-1, AN4-2) and the OPEN attestation rows (F10-7b,
F11-6a, SIM-4, SC-6), which require new attestation entries first.

## 1. What changed since inventory 1 (its gaps)

The first inventory's gaps are listed in its §11, §12 and its A5.1.2 note. They are numbered 1–9 here.

| # | gap in inventory 1 | now | evidence |
|---|---|---|---|
| 1 | F11.3 staging report and its acceptance | **closed** | `SUPPLEMENT_REPORT.md` `a88cfe7b6280169f68c301a0fdce4e98ac37ba6b3c9ec5a8ea3f913b1c2bc656` (ff81181); `board-a-f11-staging-cal-2026-11-11.md` (a) ACCEPT |
| 2 | Board A acceptance of the F11.4 report and route | **conditional** | (b) ACCEPT WITH CONDITIONS 1–3; F11d (759ef24); replay byte-identical (§4) |
| 3 | `slot_mapping_f11.json` | **closed, departure disclosed** | `1e2155adaf606b61ba7d325ff15a6970cee92ec0a5ef03ec7d48c3fdc7146862` (post-run, F11c.2; Board A reconstruction; F11d.1/.3) |
| 4 | analysis tamper/binding on the final manifest; the 40-edit simulation test | **evidence on file** | synthetic binding A–E (§6); `tests/test_a5_simulation.py` fixed in 77ee786 |
| 5 | frozen 52-edit field-prediction file | **evidence on file** | `predictions_f11.json` `50c378000b424346438de03bf2ad2881a1d10706dfc28b18d0f2130ab5e7a7bb` (§5) |
| 6 | dated F11.8 clarification | **closed** | A5 note F11c.1 (c319787); threshold 231.4 min |
| 7 | launch script LS-1 to LS-4 | **mostly fixed** | §8; new findings P2-1 to P2-3 |
| 8 | pending attestations | **partly** | entries 15–23 added; gaps in §9 |
| 9 | run-record gaps (launch record, calibration environment/inputs, uncommitted wrappers) | **partly** | `supplement_launch.json` enriched; calibration `environment` still `{}`; wrappers committed at 516d9c9 but not frozen before the runs (F11c.4) |

## 2. Population, pricing and holds (re-checked at `c947cc9`)

- **Execution manifest:** `part_a/manifest.json` `71e77a3e1f703451864ce960129399be1579d8aa84631bebc0c8d6324b4790d6`.
  - 52 selections: 40 registered + 12 supplementary.
  - Supplementary sorted-id sha256 `1848fe4cacbaacc9f643a179835770fefd3027a48cef7cf9d83358f15b85fdd9`. Registered-40 sorted-id
    sha256 `e2bd6cb2956c0fbe64194b3f1ebbb481248ecec45b8a729a59a54b9a2bac9d6b`.
  - Supplementary cells: `lora_sft:2` 4, `chain:0` 4, `chain:2` 4. There are 52 distinct components.
  - 156 variants, 9 excluded. The excluded variants are the rotations of the nine original edits.
  - 165 delta/increment files and 156 `variant.json` hashes. `scored` false. `code_sha` `887a5146…`, disclosed in F11c.3; the
    execution SHA is `8211ecf`.
- **Proofs:** `supplement.reference_proofs`: `computed` false, `paths_touched` [], `deferred_to` "score --manifest
  (score_reference_proofs)". `scoring.unproven_originals` equals the 12 supplementary ids, including `aaf7a0660c6d5a73`.
- **Priced count** (`manifest.scoring`: `n_variants` 147, `references_to_score` 19 = `n_D` 19, `n_P` 0, missing predecessors 0,
  `priced_unproven_as_references` 12, 1.04 min per entry):
  - **conservative: 178 entries, 185.12 min** (all 12 originals scored as references; `n_priced` 178, `minutes` 185.12);
  - **if all 12 prove: 166 entries, 172.64 min** (the lower bound, F11c.1(c)).
  - Retries are unpriced.
- **Stop-and-reprice threshold:** **231.4 min** = 1.25 × 185.12 (F11c.1(d)).
  - After phase 1, the price is recomputed from `reference_proofs_f11.json`. Above 231.4, scoring halts until a dated reprice note
    is appended before any GPU command. At or below 231.4, the repriced plan is recorded and accepted before GPU commands.
  - With the frozen manifest, the proof step can price at most 178 / 185.12 min (`priced_after_proofs`), so proof failures alone
    cannot cross 231.4. The threshold guards against an inconsistent record.
  - The script gates the price, not wall-clock time.
- **Exposure** (existence checks only):
  - For all 156 variant ids, `lexists(data/evals/qwen3.5-0.8b/tf_v1/<id>)` is false for each one. The `tf_v1/` mtime is
    2026-11-07T22:09:24Z.
  - `part_a/reference_proofs_f11.json` is absent, so no proof read has happened. There is no `reference_proofs_f11.prev-*` file
    and no `part_a/score_launch.json`.

## 3. Masks, mapping and calibration

| item | sha256 / value | check |
|---|---|---|
| `slot_mask_f11.json` | `5f4e770185ec9cc6fa10461d3012196bbd3feb9ca4af9f3065510a634b5bbfc4` | unchanged; = calibration and replay records |
| `slot_mapping_f11.json` | `1e2155adaf606b61ba7d325ff15a6970cee92ec0a5ef03ec7d48c3fdc7146862` | = Board A staging-cal and F11d.3 |
| ordered table `json.dumps(slots, sort_keys=True)` | `68b8b01db16f9c267943e160fafbd8e6aecb15fa5d820f3eae62a5ecd6b4620e` | recomputed from the mapping file; the mapping's `slots` equal the output's `slot_mask.slots` |
| availability digest | `171dfd20b47020dbc172d39eac87adf5c6efbe7864a0f3a784ef018fcb19741a` | = mapping, record, replay record |
| slot layout | 52 slots; rotation 43, ladder 52/52; slots 40–51 = the 12 in selection order; the 9 rotation-unavailable ids = the 9 exclusions | recomputed |
| `f11_calibration_record.json` (enriched, 516d9c9) | `3de71f022af7b8d94930a1d6f8c8b6ba428ba78f8637f921c71aba329f5548d7` | = Board A condition 3 / F11d.4; supersedes `216bdc9c…` (inventory 1) |
| output `data/runs/a5/power/power_slotmask_f11.json` | `8a1be18f726d5b030dbaa5937dcc63ba9a3ca9f730d314f689a4fbb7631441fb` | re-hashed |
| record `input_provenance` | 5 / 5 files re-hash equal | inputs.json `67da223c…`, fields.json `85843ed2…`, mask, registered_donors `6e77bca6…`, selection_f11 `ce746e71…` |
| calibration log / wrapper | `afbcd5b2…` / `00c3a8bb…` | re-hashed = record |
| `supplement_launch.json` (enriched) | `d20a4ab36faa47864bf5c5c1a724e2c4d7e096bbadb5631bc8c95bca6954fdbb` | = Board A / F11d.4; supersedes `d17fab2f…`. Its log `8e312647…` and wrapper `1dcc133a…` re-hash equal; outputs = manifest and audits `9412eb0d…` |

**Gates** (record): ordered power **0.8722 ≥ 0.80**; ordered-go size **0.0175**; two-sided **0.0348 / 0.0378 / 0.0378 ≤ 0.07**;
alternative rate 0.2576 reported, not gated. Run at `92b0338`, dirty 0, 09:28:04–09:34:03 UTC. The record's `environment` is still
`{}` (F11c.4).

## 4. The single replay (F11d.2)

- **Record:** `part_a/f11_calibration_replay_record.json` `d96036fa2087b196ec8c4a50a6a446e1bd828b6c5a10718a7895e0115ea77fd8`.
- **Output:** the replay output `data/runs/a5/f11_sim_dryrun/boarda/power_slotmask_f11_replay.json` re-hashes to
  **`8a1be18f…`, equal to the original output**: byte-identical, with no self-referential path.
- **Run:** execution SHA `759ef241c86e7c398ddcc2c9ab32cde932d76d25`, 10:08:53–10:16:02 UTC.
  - Log `/home/ubuntu/agent-logs/a5-f11-replay-20261111-100853.log` re-hashes to `7d0f11b3…`, equal to the record. Its marker
    lines: `START … HEAD 759ef241… dirty 1 mask 5f4e7701…`, `END 2026-11-11T10:16:02Z`, `REPLAY_EXIT 0`.
  - Wrapper `scripts/part_a_f11_calibration_replay.sh` `f41b72b25f06ad9c5ae3a2eef7fe486daa958b4c823fb60fcd17bb23beae9229`, equal to
    the record. Its argv equals the Board A command except `--output`.
- **F11d timing:** F11d (759ef24, 10:08:45 UTC) precedes the replay start.
- **Shortfalls against staging-cal conditions 1–2** (for Board A; SC-1, SIM-3a):
  - The wrapper was untracked at launch and was committed at 890f6e6 (10:09:03 UTC). It was not frozen by hash before execution,
    and the record's `dirty_at_launch` is 1.
  - F11d cites the review by path, not by hash. The review's sha256 is now `9f91596408dd…`.
  - The record has no exit-code field (the log has it) and no Python/NumPy/thread fields.
  - The record carries no input hashes other than the mask, and no mapping-file hash.

## 5. Frozen prediction rows (re-pin (xxiv))

- **File:** `part_a/predictions_f11.json` `50c378000b424346438de03bf2ad2881a1d10706dfc28b18d0f2130ab5e7a7bb`. It is pinned in
  `code_hashes.json` at the same value. Its status is "complete".
- **Hashes recomputed:**
  - rows sha256 `079f19c10b586c513026814d57c8709c2ddc54faa785c876fbcfe61e1cea0386`, equal to the recorded value;
  - predictions sha256 `cc0beda26c9a8c77ca1b5da9adf02f6e79db6b813762af361dbd40dd83a94c68`, equal to the recorded value.
- **Rows:**
  - 52 rows. The edit ids are exactly the manifest's 52, and 12 are flagged supplementary.
  - All 156 contrast variant ids equal the manifest's.
  - The staged-file hashes in the rows equal `manifest.delta_sha256` for **147 / 147** files and `variant_json_sha256` for
    **156 / 156** files.
  - There are 9 rotation abstentions (`rotation_pair_excluded_at_staging`), and they are the 9 exclusions. There are 0 q-partial
    rows.
  - Gradient source: base field for 42 edits, `chain_exact` for 10.
- **Comparison with `data/runs/a5/power/fields.json`** (`85843ed2…`): sign agreement 33/33 (ladder 0.25), 33/33 (ladder 0.5) and
  25/25 (rotation), with no disagreements.
- **Copies:** the run copy `data/runs/a5/part_a_predictions/predictions_f11.json` is equal to the committed file. The detail file
  `2eca1717…` equals the hash recorded in the file.
- **Provenance (finding P2-4):**
  - The file records `code.git_sha` `516d9c9` and `code_files_dirty` true, because the builder was uncommitted.
  - The builder's bytes, `d312967e…`, equal the pinned script.
  - `response_intervention.py` was at the re-pin (xx) bytes `e52ea009…`. The current pinned bytes are `47a1c6d1…`.
  - `outcomes_read`: "none: no evaluation record, TF-v1 score, rotation/ladder outcome or sealed outcome is opened …".

## 6. Analysis binding and tests

- **Synthetic input-binding run:** `data/runs/a5/part_a_predictions/synthetic_binding_summary.json`
  `e28161eb9cee01b6c00a4c1d7b8ccf2261c6a3569f0cdbf3556d09a0bebad830`, `all_passed` true, seed 20261111. The responses are random.
  - **A (intact):** exit 0. It bound the ledger, manifest, fields, spec and masked-calibration hashes.
  - **B–E (manifest byte flip; flip with its own sha; prediction row dropped; dropped row with the ledger re-bound):** each exited 2
    with the expected refusal.
  - **Inputs:** the bound inputs are the current files: manifest `71e77a3e…`, predictions `50c37800…`, output `8a1be18f…`, record
    `3de71f02…`, spec `8624e063…`, text groups `f769afca…`.
  - **Where it ran:** in `.worktrees/feat/part-a-predictions`, based on 516d9c9. `part_a_analysis.py` (`5a21e995…`) and
    `part_a_analyse.py` (`74bcff0f…`) are unchanged from 516d9c9 to `c947cc9`, and the analysis module does not import
    `response_intervention` (finding P2-5).
- **Targeted tests at `c947cc9`:**
  - Run: agent-tmux `prescore2-tests`, CPU only, 11:10:08–11:11:11 UTC; log `/home/ubuntu/agent-logs/prescore2-tests-20261111-111008.log`
    `74c8fdc4…`.
  - Files: `tests/test_part_a_analysis.py`, `test_part_a_predictions.py`, `test_a5_simulation.py`, `test_part_a_score_launch.py`
    and `test_response_intervention.py`.
  - Result: **356 passed, 2 skipped, 2 failed.** The two failures are the known order-dependent ones (§10).

## 7. Pins, staged bytes, GPU

- **Pins:** `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
  .venv/bin/python scripts/a5_build_manifests.py --verify-pins HEAD` at `c947cc9` printed **"all pins match"**, exit 0. (The same
  check printed "all pins match" on main at `87f6895` before the worktree was made.)
  - `code_hashes.json` `e0a2c7793c2554337c21449f67c0e1f690c91e00d00a3f909c0b528e056ba769` (77 files, `git_sha` `1e01a26`; re-pin
    (xxiv)).
  - Package README `26abc287dfe210fe34b0d5d534805d2578562ac03c4a40869100b4643c2e6934`.
- **Staged bytes re-hashed now** (agent-tmux `prescore2-rehash`, 11:08:57–11:11:37 UTC):
  - The sample is every 7th of the 165 sorted (variant, file) pairs from offset 3. It is disjoint from inventory 1's offset-0
    sample.
  - It covers 24 files (6 supplementary, 1 increment), 17,781,916,824 bytes, and **24 / 24 are equal**.
  - All **156 / 156** `variant.json` files are equal.
  - The two inventories together have re-hashed 48 distinct delta/increment files. The full re-hash is an execution obligation of
    each launch phase.
- **GPU identity** (11:11:38 UTC):
  - `NVIDIA RTX PRO 4500 Blackwell Server Edition, 595.112.07, 32623 MiB, GPU-3669f1b9-ff73-668a-ad0e-a3cb83c9be8c`, the same as
    inventory 1.
  - The GPU was **not idle**: a `pytest -q` process from another session (pid 2029310, started 11:09:23 UTC) held 308 MiB (finding
    P2-2).
  - This session ran nothing on the GPU.

## 8. The two-phase launcher and `score` options

**Files.** `scripts/part_a_score_launch.sh` `b1508ad0161840f2be62870508a64b49df6587cd14ee7da6c48a6a3296600aec` (77ee786; re-pin
(xxiii) text), not executed. `src/wsp/eval/response_intervention.py` `47a1c6d1…` (pinned (xxiii)).

**`score` options.**
- `--proofs-only` runs `score_reference_proofs(..., write=True)` and writes `reference_proofs_f11.json`, which carries the
  `manifest_sha256`, the `paths_touched`, the per-original verdicts and the `priced` block. It prints `PRICE …`, then returns
  without scoring. It refuses if any original stays without a proof.
- `--require-proofs` is the default. It loads the retained record and reads no evaluation record. It refuses if the record is
  absent, has another schema, belongs to another manifest, or is incomplete or inconsistent.
- `--max-minutes T` refuses before any scoring command if the price exceeds T.

**Inventory-1 findings.**
- **LS-1 fixed:** phase 1 runs proofs-only, then STOP. Phase 2 has a `score --dry-run --require-proofs --max-minutes T` gate (exit
  7) before GPU work.
- **LS-2 fixed:** a missing file fails, and all 156 `variant.json` files are re-hashed (exit 5).
- **LS-3 fixed:** argument 1 must equal the manifest sha256 (exit 4), and `--verify-pins HEAD` must print "all pins match" (exit 3).
- **LS-4 partly fixed:** the GPU UUID is recorded. The launch record is still written into the tracked tree after the clean-tree
  check, and phase 1 writes no launch record of its own.

**New findings.**
- **P2-1:** the launcher and `tests/test_part_a_score_launch.py` (`af58b943…`) are not rows of `code_hashes.json`, so
  `--verify-pins` does not cover the launcher.
- **P2-2:** the launcher checks GPU identity, not exclusivity.
- **P2-3:** phase 1 does not hide the GPU.

**Exact commands.** Run them from `/home/ubuntu/projects/wsp-lab` (canonical, clean tree), inside agent-tmux, only after Board A's
release and the chair's explicit release:

```
# phase 1 (proofs only; reads the twelve originals' TF-v1 records; then STOP)
scripts/part_a_score_launch.sh 71e77a3e1f703451864ce960129399be1579d8aa84631bebc0c8d6324b4790d6
# between phases: commit part_a/reference_proofs_f11.json; recompute the price; > 231.4 min -> halt and dated reprice note;
# <= 231.4 min -> record and accept the repriced plan; explicit GPU authorisation; confirm no other GPU compute process
# phase 2 (exclusive GPU)
scripts/part_a_score_launch.sh 71e77a3e1f703451864ce960129399be1579d8aa84631bebc0c8d6324b4790d6 --phase score --max-minutes 231.4
```

Each phase re-hashes 116,715,970,712 bytes (inventory 1 §8) and 156 metadata files. Inventory 1 read 17.48 GB in about 10 min.

## 9. Attestations (`docs/prereg/f10_attestations.md` `a5ab6708901cd879969e0aacd4d4db26131a87ab216b3a495b70621119478963`)

**Present:**
- entries 1–14, as in inventory 1;
- **15** inventory-1 session;
- **16–18** the simulator, `prepare --supplement` and analysis implementers (on record **only through the chair**);
- **19** chair extension 2;
- **20** Board A F11, F11-sim, F11-stage and analysis-4 (verbatim);
- **21** F11c drafter;
- **22** chair: F11d and the replay (to 10:16 UTC);
- **23** predictions builder (added by the chair at `c947cc9`; the user brief named entries 1–22).

**Not in the file** (needed through the final freeze; staging-cal condition 6):
- chair extension 3, from 10:16 UTC through the final freeze and release. It covers re-pins (xxiii)/(xxiv), the a36aeb9 commit and
  the hook bypass;
- the staging-report author (own line in `SUPPLEMENT_REPORT.md`, not entered);
- the Board A staging-cal reviewer (own line in its review; condition 6 asks for it to be included);
- the Board A re-stage reviewer (own line, not copied);
- the launch-hardening builder (77ee786);
- the test-order-fix builder, if that fix merges before the freeze;
- this inventory-2 session (proposed text in [`PRESCORE2_A512_NOTE.md`](PRESCORE2_A512_NOTE.md));
- any Board A reviewer of the launcher, the predictions and this inventory.

**Scoring-phase attestation.**
- **The chair as launcher.** If the chair launches phase 1 and phase 2 personally, the chair is the only new scoring-phase
  decision-maker. The chair writes one attestation through the final freeze and release **before phase 1**. Phase 1 reads the twelve
  originals' TF-v1 `eval.json`/`records.jsonl`, which counts as exposure under the stage review. The chair also adds a launch line to
  the campaign record.
- **No new session.** If the chair launches, no new session is involved. A delegated launcher would need its own attestation at
  launch.
- **After phase 1.** Accepting the repriced plan is mechanical under F11c.1(d), so no methodological choice follows the proof read.
- **Board A.** Board A attests in each review file it writes for the release.

## 10. The two order-dependent test failures (tests only, not the pipeline)

The two failures are:
- `test_f10_logs_the_complete_a533_verdict_without_using_it`: at line 789, `prerounding_norm_mismatch` is 2.9937e-09 against
  4.0984e-09 at rel 1e-6;
- `test_f10_pooled_diagnostics_are_independent_aggregates`: at line 1156, `pooled_rounding_error_rel` is 0.013612904055 against
  0.013612905618 at rel 1e-9.

**Why they do not affect the pipeline.**
- Each compares the test's own recomputation on synthetic `tmp_path` fixtures with a tight tolerance. Under the full suite (and in
  this 5-file run) the order/thread state changes the last digits.
- The staged values are frozen and hashed: the manifest, the audits and the 165 + 156 file hashes. No pipeline output depends on
  these tests.

**Status of the fix.**
- The fix is in flight on `fix/f10-diagnostic-test-order`, which had no commits at build time.
- It edits the pinned file `tests/test_response_intervention.py` (`00eed546…`), so it needs a re-pin (xxv). That re-pin changes
  `code_hashes.json` and the package README, but none of the manifest, audit, staged-byte, prediction or calibration identities
  here.
- AN4-2 asks for a green full suite at the execution commit. Until then AN4-2 is PARTIAL.

## 11. Discharged only at execution (and by whom)

| obligation | how | by |
|---|---|---|
| clean tree, pins at the execution HEAD, manifest hash | launcher checks (exit 2/3/4), both phases | launcher script, run by the chair |
| full pre-GPU re-hash (165 + 156) | launcher, phase 1 and again phase 2 (exit 5) | launcher script, run by the chair |
| reuse proofs of the 12 originals (incl. `aaf7a066`) | phase 1 `score --proofs-only` → `reference_proofs_f11.json` (commit it) | chair (launcher) |
| phase-1 price ≤ 231.4 min | `PRICE` line / record `priced`; above → halt + dated note; at or below → record and accept | chair; Board A condition 5 |
| GPU identity and exclusivity | phase 2 records name/driver/memory/UUID; exclusivity checked by hand (`--query-compute-apps` empty) | launcher + chair |
| launch record | phase 2 `score_launch.json` (start, SHA, dirty, manifest, pins, max_minutes, GPU, UUID, command; end, exit); commit after the campaign | launcher + chair |
| charges, failures, retries | campaign record; retries unpriced but recorded | chair |
| ledger sealed before the one analysis | manifest-bound sealed ledger, then `part_a_analyse.py` once | chair |
| scorer attestation at launch | the chair's own line if the chair launches | chair |

## 12. Gate table (61 rows; inventory-1 status → now)

Status key: MET, BOARD_A, AT_EXECUTION, PARTIAL and OPEN, as in §0. The full evidence text for each row is in the JSON (`gates`).

| gate | inv. 1 | now | evidence (short) |
|---|---|---|---|
| F10-1 F10a + attestations | MET | MET | unchanged |
| F10-2 executable correctness | MET | MET | unchanged |
| F10-3 pins + frozen re-stage | MET | MET | records re-hashed unchanged |
| F10-4 re-stage report accepted | MET | MET | unchanged |
| F10-5 calibration (yield < 40) | BOARD_A | BOARD_A | F11.4 PASS; (b) with conditions 1–3; F11d + replay on file |
| F10-6 F10.5 spec frozen | MET | MET | `8624e063…`; bound in synthetic run A |
| F10-7a release after 1–5, 6 | OPEN | BOARD_A | Board A, then the chair |
| F10-7b attestations through final freeze | OPEN | OPEN | §9 gaps |
| F10-7c re-hash before GPU | AT_SCORING | AT_EXECUTION | launcher; sample 24/24 + 156/156 |
| F10-7d charges | AT_SCORING | AT_EXECUTION | 178 / 185.12 or 166 / 172.64; retries unpriced |
| F10-7e seal ledger | AT_SCORING | AT_EXECUTION | post-campaign |
| F10C2-1–3 | MET | MET | unchanged |
| F10C2-4a re-stage report | MET | MET | unchanged |
| F10C2-4b frozen mask + passing calibration | BOARD_A | BOARD_A | as F10-5 |
| F10C2-4c scoring held | MET | MET | `scored` false; 0/156 paths; no proof record |
| F11-1 F11a, list, forty | MET | MET | population re-checked |
| F11-2 prepare gate | MET | MET | unchanged |
| F11-3a staging report accepted | OPEN | MET | `a88cfe7b…`; (a) ACCEPT |
| F11-3b sim option | MET | MET | unchanged |
| F11-3c mask + mapping before run | PARTIAL | MET | departure disclosed and accepted (F11c.2, F11d.1); mapping certified; replay identical |
| F11-3d reuse proofs | AT_SCORING | AT_EXECUTION | phase 1 (chair) |
| F11-4a one run, hashes | MET | MET | record `3de71f02…`; inputs 5/5 |
| F11-4b calibration report + route accepted | BOARD_A | BOARD_A | (b) with conditions; evidence on file |
| F11-5a analysis gate | MET | MET | analysis bytes unchanged |
| F11-5b tamper/binding tests | OPEN | MET | synthetic A–E passed; targeted tests |
| F11-6a attestations | OPEN | OPEN | §9 gaps |
| F11-6b all pins match | MET | MET | at `c947cc9`; repeated by launcher |
| F11-6c re-hash | AT_SCORING | AT_EXECUTION | launcher |
| F11-6d charges, no advance scoring, seal | AT_SCORING | AT_EXECUTION | chair |
| F11-6e qualification/field/allocation/reuse | OPEN | BOARD_A | predictions frozen and bound; Board A condition 4 |
| ST-1a merge, (xx), pins | MET | MET | unchanged |
| ST-1b F11.8 clarification | OPEN | MET | F11c.1 |
| ST-1c implementer/stager/reviewer attestations | OPEN | PARTIAL | 16–20 present; 16–18 relayed by the chair |
| ST-2 freeze + launch record | PARTIAL | MET | enriched record `d20a4ab3…`; departures accepted under (a) |
| ST-3 staging evidence accepted | OPEN | MET | (a) ACCEPT |
| ST-4 calibration inputs frozen after staging | PARTIAL | MET | as F11-3c |
| ST-5 proof-phase hold; proofs-only/reprice | OPEN | BOARD_A | launcher implements it; no Board A review of it on file |
| SIM-1 merge, re-pin | MET | MET | unchanged |
| SIM-2 mask; mapping before RNG | PARTIAL | MET | as F11-3c |
| SIM-3a single frozen command, full record | PARTIAL | PARTIAL | original `environment` `{}`; replay record gaps (§4) |
| SIM-3b report accepted | BOARD_A | BOARD_A | as F11-4b |
| SIM-4 holds, attestations | OPEN | OPEN | holds kept; §9 gaps |
| AN4-1 conventions | MET | MET | unchanged |
| AN4-2 merge, re-test, re-pin, pins | MET | PARTIAL | pins match; full suite not green (2 test-only failures; fix in flight) |
| AN4-3a twelve ids, cells, forty | MET | MET | re-checked |
| AN4-3b a533 + execution manifest hashes | MET | MET | re-hashed |
| AN4-3c ledger/fields/windows/identities | OPEN | BOARD_A | predictions bound; ledger/identities at execution |
| AN4-3d masks, mapping, 52-slot report | MET | MET | §3 |
| AN4-3e calibration/route accepted | BOARD_A | BOARD_A | as F11-4b |
| AN4-3f synthetic tamper checks | OPEN | MET | §6 |
| SC-a staging evidence accepted | — | MET | (a) ACCEPT |
| SC-b calibration accepted with conditions | — | BOARD_A | conditions 1–3 evidence on file |
| SC-1 prospective ordering-remedy freeze | — | PARTIAL | F11d before the replay; wrapper not hash-frozen; review not hash-cited |
| SC-2 one replay, byte identity, full record | — | BOARD_A | byte-identical; record gaps (§4) |
| SC-3 hash-history reconciliation | — | MET | F11d.4 + this inventory (§3, files table) |
| SC-4 predictions + final analysis gate | — | BOARD_A | §5, §6 |
| SC-5 two-phase launch reviewed | — | BOARD_A | §8; P2-1 to P2-3 |
| SC-6 F11c + attestations | — | OPEN | F11c met; §9 gaps |
| SC-7 updated inventory + explicit release | — | BOARD_A | this inventory; release not on file |

**Scoring stays held.**
