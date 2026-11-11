This file is not part of `docs/PREREG_H1_TF.md`. It holds the A5.1.2 hash entries for Part A pre-scoring inventory 2 and a
proposed attestation line. The chair appends both. The hashes below were computed on 11 Nov 2026, 11:08–11:15 UTC, on branch
`prereg/prescore-inventory-2` at main `c947cc9c7e48118573622507448d213b3648de2a`. Data files were hashed in place.

---

*A5.1.2 hash entries for Part A pre-scoring inventory 2 (11 Nov 2026, about 11:15 UTC; after re-pin (xxiv) and attestation 23;
before any G3 outcome; scoring held).*

**New files** under `docs/prereg/a5_manifests/part_a/`:
- `PRESCORE_INVENTORY_2.json`: `f8507c2dafdabd53a75ae09ca06b4a2172a692bf26c1d4ffdf53da186daf2ff0`
- `PRESCORE_INVENTORY_2.md`: `bed350432cc4c8654a791e33370827205740cf685af98b1ac72bfb4bb9a78245`

**Inventory 1.** It is kept unchanged as the historical snapshot at `079e60b`. Its status columns are superseded by inventory 2. Its
own A5.1.2 entries (`PRESCORE_A512_NOTE.md` `13001b8b0440ae0cccc459f9e0946932c7068592ecf311511ffaceb393611ebe`) were not appended
to the registration. They are recorded here:
- `PRESCORE_INVENTORY.json` `2ab1caea1dce1f4ac2afb50a59e80d3f3cf3c19ae4a922144a61bbddd9cc2160`
- `PRESCORE_INVENTORY.md` `419ee039de63ea75014b9271e3a2718e593bf9d17b5d83900533511c04100f0c`

**Current identities that supersede earlier entries** (Board A staging-cal condition 3; F11d.4):
- `supplement_launch.json` `d20a4ab36faa47864bf5c5c1a724e2c4d7e096bbadb5631bc8c95bca6954fdbb`. It was `d17fab2f…` in
  SUPPLEMENT_A512_NOTE and inventory 1, before enrichment at 516d9c9.
- `f11_calibration_record.json` `3de71f022af7b8d94930a1d6f8c8b6ba428ba78f8637f921c71aba329f5548d7`. It was `216bdc9c…` before
  enrichment.
- The enrichments are retrospective and are not prospective freeze evidence (F11c.4).

**Files added since inventory 1, re-hashed:**
- `SUPPLEMENT_REPORT.md` `a88cfe7b6280169f68c301a0fdce4e98ac37ba6b3c9ec5a8ea3f913b1c2bc656` (accepted, staging-cal (a))
- `SUPPLEMENT_A512_NOTE.md` `b8a5997a0e98243d7e451fdbe92850a96911ef8fed632101e6725746d66c9565`
- `slot_mapping_f11.json` `1e2155adaf606b61ba7d325ff15a6970cee92ec0a5ef03ec7d48c3fdc7146862`, written post-run and disclosed in
  F11c.2. Its ordered table is `68b8b01db16f9c267943e160fafbd8e6aecb15fa5d820f3eae62a5ecd6b4620e` and its availability digest is
  `171dfd20b47020dbc172d39eac87adf5c6efbe7864a0f3a784ef018fcb19741a`.
- `f11_calibration_replay_record.json` `d96036fa2087b196ec8c4a50a6a446e1bd828b6c5a10718a7895e0115ea77fd8`. The replay output
  `data/runs/a5/f11_sim_dryrun/boarda/power_slotmask_f11_replay.json` is `8a1be18f726d5b030dbaa5937dcc63ba9a3ca9f730d314f689a4fbb7631441fb`,
  byte-identical to the original output. The replay log is `7d0f11b3305c557e5506a08f437fe9ab38dd3be2d9e4da0d64d0edb8cbc5dc41`.
- `predictions_f11.json` `50c378000b424346438de03bf2ad2881a1d10706dfc28b18d0f2130ab5e7a7bb`. Its rows hash to
  `079f19c10b586c513026814d57c8709c2ddc54faa785c876fbcfe61e1cea0386` and its predictions to
  `cc0beda26c9a8c77ca1b5da9adf02f6e79db6b813762af361dbd40dd83a94c68`; both were recomputed.
- `data/runs/a5/part_a_predictions/synthetic_binding_summary.json`
  `e28161eb9cee01b6c00a4c1d7b8ccf2261c6a3569f0cdbf3556d09a0bebad830`: cases A–E passed.

**Scripts:**
- `part_a_score_launch.sh` `b1508ad0161840f2be62870508a64b49df6587cd14ee7da6c48a6a3296600aec` (77ee786; not executed; **not a row
  of `code_hashes.json`**)
- `part_a_f11_calibration_replay.sh` `f41b72b25f06ad9c5ae3a2eef7fe486daa958b4c823fb60fcd17bb23beae9229`
- `part_a_f11_calibration_launch.sh` `00c3a8bb46acbd1e34f095deba6b8a5f58167060e1c4a8414b70ae23de91b19b`
- `part_a_f11_supplement_launch.sh` `1dcc133a4b1cfd8e356fd6b2ff7613054fc43033a0035c83cbd1c3d0eb6c92e4`
- `part_a_f10_restage_launch.sh` `ce25c17c8484b006af2f0b5eac71695109899624ca8947673763725d5410afb2`

**Pins and records at `c947cc9`:**
- `code_hashes.json` `e0a2c7793c2554337c21449f67c0e1f690c91e00d00a3f909c0b528e056ba769` (re-pin (xxiv); 77 files)
- package README `26abc287dfe210fe34b0d5d534805d2578562ac03c4a40869100b4643c2e6934`
- `--verify-pins HEAD`: all pins match
- `docs/prereg/f10_attestations.md` `a5ab6708901cd879969e0aacd4d4db26131a87ab216b3a495b70621119478963` (entries 1–23)
- `docs/reviews/council/board-a-f11-staging-cal-2026-11-11.md` `9f91596408dd2f4f37745300ad0ab00889bd02d6f958428cbe215c6a43d0ef6e`

**Unchanged and verified:**
- `manifest.json` `71e77a3e…`
- `rotation_audits.jsonl` `9412eb0d…`
- `manifest_pre_f11.json` `59d839e3…`
- `rotation_audits_pre_f11.jsonl` `4ec0928a…`
- `manifest_a533.json` `d7785b35…`
- `rotation_audits_a533.jsonl` `f05fec42…`
- `README.md` `2fcbf517…`
- `supplement_freeze.json` `503e415e…`
- `slot_mask_f11.json` `5f4e7701…`
- `power_slotmask_f11.json` `8a1be18f…`
- the five calibration inputs
- `selection_f11.json` `ce746e71…`
- `F10_SENSITIVITY_SPEC.md` `8624e063…`
- `text_groups.json` `f769afca…`

**Staged bytes:** a disjoint sample of 24 of the 165 delta/increment files and all 156 `variant.json` files re-hash equal (11:08:57–11:11:37 UTC).

**Gate state:**
- 61 rows: 33 MET, 14 await Board A, 6 at execution, 4 partial, 4 open.
- Priced count: 178 entries / 185.12 min (conservative); 166 / 172.64 min if all 12 originals prove.
- Stop-and-reprice threshold: 231.4 min.

Nothing has been scored. The scoring hold remains.

---

*Proposed attestation entry for `docs/prereg/f10_attestations.md` (pre-score inventory 2 builder; the chair adds it).*

- **Role:** built Part A pre-scoring inventory 2 (`PRESCORE_INVENTORY_2.json`/`.md`) and this note.
- **Session identity:** Claude Opus 5.5 (claude_umut instance), branch `prereg/prescore-inventory-2` (worktree
  `.worktrees/prereg/prescore-inventory-2`), delegated by the chair. It was created from main `87f6895` on 11 Nov 2026 at 11:06:20 UTC
  and fast-forwarded to `c947cc9` at 11:08:34 UTC (git reflog).
- **Coverage:** from branch creation through the commit that adds these files.
- **Read and ran:**
  - documents: the F11c, F11d and re-pin (xxii)–(xxiv) text of `docs/PREREG_H1_TF.md`; the Board A reviews (F10, F10-code-2, F11,
    F11-stage, F11-sim, analysis-4, staging-cal); `f10_attestations.md`; `docs/PROGRESS.md`; inventory 1;
  - committed `part_a/` metadata, records, the predictions file, scripts and code;
  - the calibration and replay outputs (simulation outputs) and the synthetic binding summary and synthetic report (random
    responses).
  - It re-hashed staged bytes under `data/runs/a5/part_a/`, run outputs under `data/runs/a5/` and agent-tmux logs.
  - It ran `--verify-pins HEAD`, the targeted CPU test run (5 test files) and `nvidia-smi` identity queries.
- **Attestation:** Under `data/evals` I ran existence (`lexists`) checks only and opened nothing. I opened no zoo `eval.json`,
  `records.jsonl` or `summary.json`, and wrote nothing under `data/`. I used no GPU for computation and ran no scoring, proof step,
  staging, simulation or analysis. I have seen no G3, rotation, ladder, reference or sealed outcome, or a summary of one; none
  exists. I made no methodological choice. This attests this session only.
