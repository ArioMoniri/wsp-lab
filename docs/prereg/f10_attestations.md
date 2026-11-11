# F10 no-exposure attestations (Board A F10 condition 1; F8a.1(c))

**Status: 11 Nov 2026, before any G3 outcome.** One attestation per decision-maker who drafted, built, analysed, chaired or reviewed
A5 amendment F10 (`docs/PREREG_H1_TF.md`, F10 and note F10a). F8a.1(c) asks for role, session identity and the time through which
the attestation holds. Exposure here means having opened, computed, scored or received any G3, rotation, ladder, missing-reference
or sealed outcome, or a summary of one (any message, document or log line conveying one; F8a.1(c), F10.8). Part A has 120 variant
ids (40 rotation, 80 ladder; `docs/prereg/a5_manifests/part_a/manifest.json:variants`) and 19 missing originals (n_D = 19,
`manifest.json:scoring.reuse_proofs` entries failing `no_record`, the same ids as the staging README).

Sources of the record. Session identities are given as provider/model plus the git branch (worktree) each session worked on. No
session id is invented; only the chair's thread id is known, as given by the chair. Coverage times are UTC and come from git reflog
and commit times, which the timestamp note in `docs/PROGRESS.md` (10 Nov, 23:20 UTC) makes authoritative over the clock labels on
`PROGRESS.md` entries. Where an attestation rests only on a delegate's report to the chair, that is stated. An attestation by one
session does not prove another's access history (the review, "Pre-outcome legitimacy and attestation").

## Path check (this file's author, 11 Nov 2026, about 01:25 UTC)

Existence and name listing only; no file under `data/evals/` was opened or read.

1. For each of the 139 ids (120 variant ids from `manifest.json:variants[].variant_id`, 19 missing originals as above),
   `os.path.lexists("data/evals/qwen3.5-0.8b/tf_v1/<id>")`: **0 of 139 exist.** `data/evals/qwen3.5-0.8b/tf_v1/` itself exists
   (694 entries, last modified 7 Nov 2026 22:09 UTC).
2. A walk of directory and file names under `data/evals/` (to depth 4), matching names that contain any of the 139 ids: **no
   variant id appears anywhere.** The 19 missing originals appear only as `data/evals/qwen3.5-0.8b/<id>` (outside `tf_v1/`).
3. Those 19 names are symlinks, created 6 Nov 2026 08:10 UTC, into the zoo entries `/home/ubuntu/wsp-data/zoo/qwen3.5-0.8b/<id>`
   (389 of the 393 names in `data/evals/qwen3.5-0.8b/` are symlinks; none has a modification time after 7 Nov 2026 00:00 UTC). Listing the file names of one
   target (`1ab850a41051fe45`) shows `edit.json`, `eval.json`, `generator_edit.json`, `lora`, `manifest.json`, `records.jsonl`,
   `summary.json`, `train.json`: the zoo holds a pre-TF-v1 evaluation written at zoo generation. These files were not opened. They
   are not TF-v1 records, not the missing references A5.3.4 prices (those are TF-v1 records), and were not produced by Part A.
   Whether they count among the D3 development outcomes that A5.3 already treats as known is for the chair and Board A; they are
   listed here so that each attestation below can be read against them.

So no evaluation record of any Part A variant exists anywhere under `data/evals/`, and no TF-v1 record of any missing original
exists. A pre-TF-v1 record of each missing original exists in the zoo and is reachable through `data/evals/qwen3.5-0.8b/<id>`.

## 1. Chair

- **Role:** chair; orchestrating session. Took rulings R1 (remedy (d)), R2 (chain-ladder convention) and R3 (scoring hold).
- **Session:** Claude Fable 5.1, T3 thread `aef54e1a…`; merges to `main` and delegated the branches below.
- **Coverage:** 10 Nov 2026 18:00 UTC to the time of this note (11 Nov 2026, about 01:25 UTC).
- **Attestation (verbatim, as supplied by the chair):**

> Chair attestation. I (the orchestrating session, Claude Fable 5.1, thread aef54e1a…) coordinated staging, the rotation-yield
> analysis, the F10 drafting and the F10 code task between 10 Nov 2026 18:00 UTC and the time of this note. I read the staging
> README, manifest and rotation audits, the design note and the F10 text. I opened no file under data/evals for any Part A variant
> or missing reference, ran no scoring, and read no score, response or contrast. No Part A outcome exists. I took the F10 decisions
> (remedy d; chain-ladder convention; scoring hold) on the audit results alone.

## 2. Staging builder

- **Role:** builder of the Part A staging (120 variants, per-tensor rotation audits, reference plan) whose audit triggered F10.
- **Session:** Claude Opus 5.5, branch `feat/part-a-prepare` (worktree `.worktrees/feat/part-a-prepare`).
- **Coverage:** 10 Nov 2026 18:13–23:45 UTC for the runs (8-thread attempt from 18:13:46, stopped; `prepare` 18:46:52–23:45:24).
  Branch record: created 18:02:31, code commit `7f8b453` 18:09:26, staging commit `01e1326` 23:47:57 UTC.
- **Read and ran:** CPU only. `python -m wsp.eval.response_intervention prepare` on the registered selection, references and text
  groups (staging README, "What was run"); wrote only under `data/runs/a5/part_a/` and the committed package.
- **Attestation (quoted from its committed artefact, `docs/prereg/a5_manifests/part_a/README.md`, sha256 `2fcbf517…ceada1`):**

> **No outcome exists.** Nothing was scored: no staged variant and none of the 19 missing references has an evaluation record, and
> no response, score or contrast was read or computed.

  and "Nothing was written under `data/zoo` or `data/evals`." The chair's record (`docs/PROGRESS.md`, 10 Nov 23:50 UTC) adds:
  "Nothing has been scored and no outcome has been read." The manifest records `scored: false`.
- **Disclosure from the code it ran (not from a separate statement by the builder):** for the F3(i) reuse proofs the `prepare`
  process parsed `eval.json` and `records.jsonl` under `data/evals/qwen3.5-0.8b/tf_v1/` for the 30 references already on file
  (21 originals and 9 predecessors, including the base; `response_intervention.reuse_proof`). It extracted identity fields
  (status, config hash, model revision, benchmark counts, entry hashes, run id, WikiText window ids and token hashes) into
  `manifest.json:scoring.reuse_proofs`. These are pre-existing development records, not records of a Part A variant or missing
  reference. For the 19 missing originals it found no `tf_v1/<id>/eval.json` (`no_record`). In `response_intervention.py`,
  `eval.json` is read only from the `--evals` directory (`tf_v1/`), not from zoo entries (path check, item 3).

## 3. Rotation-yield analyst

- **Role:** author of the pre-outcome diagnosis and remedy yields that F10 adopts (`docs/design/ROTATION_YIELD_2026-11-11.md`,
  sha256 `fad409cc…f0e5719b`).
- **Session:** Claude Opus 5.5, branch `design/rotation-yield` (worktree `.worktrees/design/rotation-yield`).
- **Coverage:** branch created 10 Nov 2026 23:49:12 UTC; commit `ccf1e7b` 11 Nov 2026 01:01:05 UTC. Scratch outputs in
  `data/runs/a5/rotation_yield/` carry modification times from 23:50:39 UTC 10 Nov to 01:00:57 UTC 11 Nov.
- **Read and ran (the note §§0.1–0.2):** registration sections, the staging README, manifest and audit rows, the rotation and
  variant-build code, the Board A F7 review, the NULL-SIZE and POWER reports and run timings, and staged rotation `variant.json`
  files (logged moments only). CPU only, `OMP_NUM_THREADS=4`, inside `agent-tmux`: `a5_rotation_yield_population.py`,
  `a5_rotation_yield_check.py` (14 min 49 s), `a5_rotation_yield_recon64.py` (19 s), `a5_rotation_yield_rank.py` (6 min 10 s),
  `a5_rotation_yield_full.py` (36 min 45 s), `a5_rotation_yield_summarise.py`, `a5_rotation_yield_remedies.py`. Single write
  location `data/runs/a5/rotation_yield/`.
- **Attestation (quoted from its committed note, "Status"):**

> No variant, reference or outcome was scored, opened or read: nothing under `data/evals/` was touched, and no response, score,
> contrast or summary of one exists to be read (staging manifest `scored: false`).

  The chair's record (`docs/PROGRESS.md`, 11 Nov "Rotation yield: diagnosis and chair rulings") states "CPU only".

## 4. F10 drafter

- **Role:** wrote the F10 block (F10.1–F10.12), including the F10.12 implementability edits (near-cut rank window, replay
  environment, fp32 slack).
- **Session:** Claude Opus 5.5, branch `prereg/f10` (worktree `.worktrees/prereg/f10`).
- **Coverage:** branch created 11 Nov 2026 01:02:25 UTC; commits `87cb3f8` 01:08:21 and `0af5a66` 01:12:49 UTC (merged `4fe4379`).
- **Attestation (quoted from its committed artefact, F10.11 in `docs/PREREG_H1_TF.md` at `0af5a66`):**

> none of the 120 staged or excluded variant ids and none of the 19 missing originals has a directory under
> `data/evals/qwen3.5-0.8b/tf_v1/` (checked by path existence only, 11 Nov 2026; nothing was opened). […] This session read only
> `docs/`, `src/` and `scripts/`, wrote nothing under `data/`, and ran CPU-only synthetic checks of the F10.3(i) and (iv)
> arithmetic (F10.12).

  Its report to the chair states that F10.11 was checked by path existence only. That report is not committed; only the chair's
  record of it exists (this entry, on the chair's instruction). `docs/PROGRESS.md` does not quote it.

## 5. F10 code builder (completed by the chair from its report, 11 Nov 2026)

- **Role:** implementer of the F10.3 checks, F10.4 diagnostics, the `--audit-rule a533|f10` switch, the realised-increment ladder flag
  and `scripts/a5_f10_dryrun.py` (commit e177ce6, rejected by Board A under condition 2; corrected as commit 060cc60 on the same branch with the eight corrections and the rotation-only preparation mode; under Board A re-review; not merged).
- **Session identity:** delegated session, Claude Opus 5.5 (claude_umut instance), branch `feat/f10-rotation-audit`, worktree
  `.worktrees/feat/f10-rotation-audit`; coverage from 01:03:12 UTC to about 02:50 UTC, 11 Nov 2026.
- **Read and ran:** the rotation-yield note, the F10 text, the staging README/manifest, `rotation.py` and `response_intervention.py`;
  14 new tests (written first); a CPU dry run on exactly three staged edits (`00f8111cbad8218f`, `0a841d4082532e8b`,
  `0027efdf2eae7405`) writing only to `data/runs/a5/f10_dryrun/` (5.2 GB); rotation-only timing measurements.
- **Statement (from its report to the chair, quoted):** "Dry-run verdicts (scratch only; nothing scored)". It opened no file under
  `data/evals`; its inputs are stored weights and the staged variants, not outcomes. The dry-run verdicts (two passes, one
  `replay_failed` exclusion) are audit results, not outcomes.
- Second-round statement (from its report on 060cc60, quoted): "I opened nothing under `data/evals` and wrote only to the scratch directory" (`data/runs/a5/f10_dryrun/condition2/`, 1.3 GB; 4 threads). Coverage extended to about 05:20 UTC, 11 Nov 2026.
- Its reports to the chair are on record only through this entry and `docs/PROGRESS.md`.

## 6. Board A

- **Role:** ML methods/pre-registration reviewer; accepted F10 with conditions.
- **Session:** Codex GPT-6.1 Sol, worktree `review/board-a-f10` (branch created 11 Nov 2026 01:13:35 UTC; review commit `e9c6db7`
  01:17:25 UTC; review file sha256 `f1c494ab608c464d29f9dc261dee8711e553ab0fee90a136581b0efb432142b3`).
- **Attestation (verbatim from `docs/reviews/council/board-a-f10-2026-11-11.md`):**

> **Board A exposure attestation — role: ML methods/pre-registration reviewer; session/worktree: `review/board-a-f10` (this review
> turn), 11 November 2026, through completion of this report.** I have not opened, computed, scored or received a G3, rotation,
> ladder, missing-reference or sealed outcome or a summary of one. I read only the supplied task and repository
> methodological/code/staging/simulation documents for this ruling. Evaluation checks were path existence only for the 139
> identified IDs, all absent. No GPU or data write occurred. This is my own attestation; it does not substitute for the other
> decision-makers' attestations or prove their histories.

## 7. F10a drafter (this file's author)

- **Role:** wrote note F10a (adopting Board A's rank-budget and chain-ladder wording; threshold status; no-retuning rule) and this
  file.
- **Session:** Claude Opus 5.5, branch `prereg/f10a` (worktree `.worktrees/prereg/f10a`), delegated by the chair.
- **Coverage:** 11 Nov 2026 from branch creation 01:18:09 UTC through the commit that adds this file.
- **Attestation:** I read the Board A F10 review, the F10 block and F8a.1(c) of the registration, the A5.1.2 re-pin (xv) note, the
  design note, the staging README, `manifest.json` (ids, scoring plan and reuse-proof fields), `docs/council/2026-11-11/MEMBERS.md`,
  `docs/PROGRESS.md`, the `reuse_proof` code, git reflogs, file modification times under `data/runs/a5/rotation_yield/`, and the
  names (not contents) listed in the path check above. I opened no file under `data/evals/` and no zoo `eval.json`,
  `records.jsonl` or `summary.json`; I ran no scoring, GPU job or simulation; I wrote nothing under `data/`; and I read no score,
  response or contrast of any Part A variant or missing reference, or a summary of one. None exists.

## Not covered

Sessions that start after this file, for example the one on branch `design/f10-sensitivity` (created 11 Nov 2026 01:18:34 UTC),
are not covered. Any session that makes or approves a further F10 choice before scoring adds its own attestation here (review
condition 7), and the scoring record adds one for the interval up to the scoring campaign.

## 8. F10.5 sensitivity-specification drafter (added by the chair, 11 Nov 2026)

- **Role:** author of `docs/design/F10_SENSITIVITY_SPEC.md` (Board A F10 condition 6 specification).
- **Session identity:** delegated session, Claude Opus 5.5 (claude_umut instance), branch `design/f10-sensitivity`, worktree
  `.worktrees/design/f10-sensitivity`; created 01:18:34 UTC 11 Nov 2026 (git reflog); commit `e577c67` (merged to main as 2ee94c7,
  content identical except the root README status block).
- **Read and ran:** Board A's F10 review, the F10 block, the rotation-yield note, A5.4/A5.5/A5.3.2/F5.7/F7, the staged Part A
  manifest (`edit_audits`), the null-size report. CPU only; recomputed the 30-pair subset from `manifest.json`; took the 34-pair list
  as written in F10.5(e).
- **Statement (from its report to the chair, quoted):** "No GPU was used, nothing under `data/` was written, and nothing under
  `data/evals` was opened."
- **Coverage:** 01:18 UTC to the time of its report (about 01:55 UTC), 11 Nov 2026. Its report to the chair is on record only through
  this entry.

## 9. Slot-mask path builder (added by the chair, 11 Nov 2026)

- **Role:** implementer of `--slot-mask` in `scripts/a5_simulation.py` (Board A F10 condition 5), merged as b3bd276 with re-pin (xvi).
- **Session identity:** delegated session, Claude Opus 5.5 (claude_umut instance), branch `feat/a5-slot-mask`; commit a8a84af
  (content identical to the merged a03d2d2 except the root README status block).
- **Read and ran:** the F10 review condition 5, F10.6, F7 and the registered power run's command, the Part A manifest (edit ids and
  tertiles), the simulator tests; CPU only; one 200-replicate smoke run on the real inputs with a synthetic 39-slot mask written to
  `data/runs/a5/slot_mask_dryrun/`; the registered-scale 10,000-replicate slot-masked run was NOT executed (it is gated).
- **Statement (from its report to the chair, quoted):** "I did not run the 10,000-replicate registered run." It opened no file under
  `data/evals`; its inputs are the null-size payload and the sign predictions, not outcomes.
- **Coverage:** about 01:20 to 02:40 UTC, 11 Nov 2026. Its report to the chair is on record only through this entry.

## F11 entries (added 11 Nov 2026, after the F11 merge and Board A's F11 review; Board A F11 condition 1)

Entries 10–13 cover the sessions that have so far decided, drafted or sized amendment F11 (`docs/PREREG_H1_TF.md`, F11 and note
F11a). They are written by the F11a drafter (entry 13) from the committed record: the metrology report's provenance and exposure
lines, `selection_f11.json:outcomes_read`, the F11 block's own attestation, `docs/PROGRESS.md` (06:40, 07:20, 07:50, 08:15 and 08:50
UTC entries), git reflogs and the Board A F11 review. Coverage times are UTC, from the reflogs (branch creation and last commit).
Board A's F11 line is in `docs/reviews/council/board-a-f11-2026-11-11.md` (role ML methods/pre-registration reviewer; session
`review/board-a-f11`; 11 Nov 2026, through completion of that review) and is incorporated by reference (F11a.4). Not covered: the
stager, the `prepare --supplement` and 52-slot-simulation implementation sessions and any later reviewer; each adds its own entry
before it acts. The chair's coverage in entry 1 ended about 01:25 UTC; the chair's F11 decisions (07:20 UTC entry) fall after it, and
entry 12 covers only the sizing runs, so the chair extends entry 1 through the F11 decisions and the final methodological freeze
(review, condition 6).

## 10. Council metrology member for F11 (added by the F11a drafter, 11 Nov 2026)

- **Role:** council metrology member; author of `docs/council/2026-11-11/f11-metrology.md` (the resizing options, the k = 12
  recommendation and the Option B fallback wording).
- **Session identity:** Claude Opus 5.5, branch `council/f11-resize` (created from main `4c33fdb` 11 Nov 2026 06:37:44 UTC; commit
  `2151b2c` 07:08:49 UTC; merged as `92fff4e`).
- **Coverage:** 11 Nov 2026 06:37 UTC to the report commit at 07:08 UTC.
- **Read and ran:** CPU only. It read the registration sections A5.3 to A5.5, F2, F5.7, F7, F10 and F10a, the Board A F10 review,
  PROGRESS (06:30 and 06:40 UTC entries), the Part A staging metadata, the NULL-SIZE inputs and power fields, and the outputs
  `data/runs/a5/power/sizing_slots_{34,37,39}.json`. Its reduced-replicate emulator and grid runs (500 replicates) wrote only to
  `data/runs/a5/f11_sizing/` (`candidates.json`, `binomial_curve.json`, `donor_cell_resolvability.json`, `emulator_*.json`, 16 grid
  outputs, logs and `code/`), not committed. It also produced the sizing outputs reported in the metrology report. It used no GPU.
- **Attestation (quoted from `f11-metrology.md`):**

> Not opened: anything under `data/evals`.
>
> Exposure: this member has seen no G3, rotation, ladder, reference or sealed outcome, nor a summary of one. Simulation inputs are
> the pre-G3 NULL-SIZE payload and power fields.

  The report's header adds "No GPU. Data read only. No file under `data/evals` was opened." (its own header says "Data read only";
  the list above shows it wrote scratch files under `data/runs/a5/f11_sizing/`, which its provenance section also records). It
  opened nothing under `data/evals`. The audit and calibration figures it used (31/40, 0.7514) are construction and design evidence,
  not outcomes (Board A F11 review, "Legitimacy").

## 11. F11 drafter (added by the F11a drafter, 11 Nov 2026)

- **Role:** wrote the F11 block (F11.1–F11.9), the frozen supplementary list `selection_f11.json`, its builder
  `scripts/a5_f11_supplement.py` and `tests/test_a5_f11_supplement.py`.
- **Session identity:** Claude Opus 5.5, branch `prereg/f11` (created from main `04c64f0` 11 Nov 2026 07:07:47 UTC; commit `0a3049d`
  07:20:43 UTC; merged to main).
- **Coverage:** 11 Nov 2026 07:07 UTC to the commit at 07:20 UTC.
- **Read and ran:** `docs/` (the registration, PROGRESS, the metrology report, the Board A F10 and F10-code-2 reviews, the Part A and
  selection manifests); `scripts/` and `src/`; the zoo manifest `data/zoo/qwen3.5-0.8b/manifest.jsonl` and the `edit.json` files (and
  a directory listing) of four candidates; the outputs `data/runs/a5/power/sizing_slots_{34,37,39}.json`. It ran the builder and its
  11 tests on metadata (CPU). It made no GPU use, staged nothing, ran no simulation and wrote nothing under `data/`.
- **Attestation (quoted from `docs/prereg/a5_manifests/selection_f11.json`, field `outcomes_read`):**

> none: metadata manifests only (no TF-v1 record, evaluation file or G3 outcome)

  and from the F11 block (F11.6): "It opened nothing under `data/evals`, used no GPU, staged nothing, ran no simulation and wrote
  nothing under `data/`. It has seen no G3, rotation, ladder, reference or sealed outcome, or a summary of one." Disclosed: the
  not-alias-only filter was in the chair's brief to this drafter, and the drafter's own report states the order and filters are
  audit-informed and outcome-blind (F11.6).

## 12. Chair's sizing runs (added by the F11a drafter from the chair's PROGRESS entry, 11 Nov 2026)

- **Role:** the chair launched the three outcome-free sizing runs that supplied the 34, 37 and 39-slot anchors of the F11 sizing
  curve; the chair's F11 decision (Option A) is the 07:20 UTC PROGRESS entry.
- **Session identity:** the chair, Claude Fable 5.1, T3 thread `aef54e1a…` (as in entry 1). The runs ran under `agent-tmux` as
  `a5-sizing-34`, `a5-sizing-37` and `a5-sizing-39`.
- **Coverage:** 11 Nov 2026 from the 06:40 UTC PROGRESS entry (after the slot-masked calibration record was committed at 06:37 UTC)
  through the sizing outputs and the 07:20 UTC decision. The run end times are in the output files, not restated here.
- **Read and ran:** CPU only. The three runs used the same simulator, 61-donor pool, seeds and 10,000 replicates as the F10.6 run,
  with **synthetic masks** that unmask subsets of the 9 excluded rotation slots. They wrote
  `data/runs/a5/power/sizing_slots_34.json`, `sizing_slots_37.json` and `sizing_slots_39.json` (sha256 `e36a832b…`, `ed039ab5…`,
  `71de730f…`, F11.1(c)). The files carry the F10.6 label string; the runs are labelled sizing, not registered gates.
- **Attestation (quoted from the chair's record, `docs/PROGRESS.md`, 11 Nov 2026, 06:40 UTC):**

> Chair: outcome-free sizing runs launched (synthetic masks unmasking subsets of the excluded cells, labelled sizing, not
> registered) to size a supplementary-selection amendment (F11) before council metrology and Board A are asked.

  and the 07:20 UTC entry: "candidates exist in the three depleted cells [...] all base-exact roots so the scorer round-trip cannot
  fail". The chair's PROGRESS record states that nothing has been scored (10 Nov 23:50 UTC entry) and the Part A manifest records
  `scored: false`. The simulations read pre-G3 NULL-SIZE and power fields and audit-derived masks, not outcomes. This entry
  quotes the chair's record and is not a fresh chair statement; the chair's no-exposure statement for the F11 interval is for the
  chair to extend in entry 1 (see above).

## 13. F11a drafter (this drafting session; added 11 Nov 2026)

- **Role:** wrote note F11a (F11 clarifications adopting Board A F11 condition 1) and entries 10–13 of this file.
- **Session identity:** Claude Sonnet 5.5, branch `prereg/f11a` (worktree `.worktrees/prereg/f11a`), delegated by the chair; created
  from main `af82db4` at 11 Nov 2026 07:26:56 UTC.
- **Coverage:** from branch creation through the commit that adds this entry.
- **Read and ran:** the Board A F11 review (condition 1 in full and the body it refers to), the F11 and F10a blocks at the end of
  `docs/PREREG_H1_TF.md`, the format of this file, the metrology report (its provenance and exposure lines), `docs/PROGRESS.md`
  (06:40 to 08:50 UTC), `selection_f11.json` (the `outcomes_read` field), git reflogs and logs. It ran `scripts/secret_scan.sh` and
  git commands. It ran no scoring, GPU job, simulation, staging or test suite, and wrote only under `docs/`.
- **Attestation:** I opened nothing under `data/evals/`, opened no zoo `eval.json`, `records.jsonl` or `summary.json`, used no GPU,
  and wrote nothing under `data/`. I have seen no G3, rotation, ladder, reference or sealed outcome, or a summary of one. None
  exists. I made no selection or methodological choice: F11a copies Board A's wording. It rests on the chair's brief and the
  committed record; it does not prove other sessions' histories.
- **Note on the F11 drafter's commit record:** PROGRESS (08:15 UTC) records the F11 drafter's report as one commit made with
  `--no-verify` and the rest of the suite run under `agent-tmux`, and that nothing under `data/evals` was opened.

## 1 (extension). Chair — coverage through the F11 decisions (11 Nov 2026, about 09:20 UTC)

Extending entry 1: from 01:25 UTC to the time of this entry I coordinated the F10 code rounds and their Board A reviews, the re-pin
(xviii) freeze and the rotation-only re-stage launch (02:44 UTC, agent-tmux `a5-f10-restage`), the regenerated-manifest commit and
the F10.6 slot mask, the two slot-masked calibration launches (06:24 and 06:25 UTC), the three sizing runs (agent-tmux
`a5-sizing-{34,37,39}`, 06:37 UTC start; outputs `data/runs/a5/power/sizing_slots_*.json`; their start and end times are in the
agent-tmux logs `/home/ubuntu/agent-logs/a5-sizing-*.log`), the F11 decision (07:20 UTC) and the F11 and re-stage-report reviews. I
read the regenerated manifest, the audit files, the calibration and sizing outputs (simulation outputs, not evaluations) and the
reports. I opened no file under `data/evals` for any Part A variant, supplementary edit or missing reference, ran no scoring, and
read no score, response or contrast. No Part A outcome exists. The scoring launch script exists at `/tmp/a5_part_a_score.sh` and
has not been run.

## 14. F11b drafter (this drafting session; added 11 Nov 2026)

- **Role:** wrote note F11b (adoption of the Part A analysis F11-path conventions, Board A Part A analysis review 4, condition 1)
  and this entry.
- **Session identity:** Claude Sonnet 5.5, branch `prereg/f11b` (worktree `.worktrees/prereg/f11b`), delegated by the chair; created
  from main `9add718` at 11 Nov 2026, about 08:20 UTC.
- **Coverage:** from branch creation through the commit that adds this entry.
- **Read and ran:** `docs/reviews/council/board-a-part-a-analysis-4-2026-11-11.md`, the module docstring and two short passages (the
  supplement-row checks, the route-acceptance function, the size-hold check) of `src/wsp/eval/part_a_analysis.py`, the F10c, F11,
  F11a and re-pin (xxi) text of `docs/PREREG_H1_TF.md`, `docs/PROGRESS.md` (the 08:06 and 08:16 UTC entries) and this file. It ran
  `scripts/secret_scan.sh` and git commands. It ran no scoring, GPU job, simulation, staging or test suite, and wrote only under
  `docs/`.
- **Attestation:** I opened nothing under `data/evals/`, opened no zoo `eval.json`, `records.jsonl` or `summary.json`, used no GPU,
  and wrote nothing under `data/`. I have seen no G3, rotation, ladder, reference or sealed outcome, or a summary of one. None
  exists. I made no methodological choice: F11b copies Board A's rulings and the builder's recorded decisions. This attests this
  session only; it does not prove other sessions' histories.

## F11 execution entries (added by the F11c drafter, 11 Nov 2026, after the staging, the calibration and the pre-scoring inventory)

Entries 15 to 21 complete the pending list in `PRESCORE_INVENTORY.md` §5. Entries 16–18 rest only on the chair's PROGRESS record:
the three implementers' own reports are not in the repository, and the chair's entries carry no exposure line for them. They are
therefore on record **only through the chair**, and they quote what that record says. Where the record is silent, this file says so
and does not fill the gap. PROGRESS labels from "09:45 UTC" to "11:40 UTC" are estimates ("Timestamp note 2"); the commit times given
here are from git.

## 15. Pre-scoring inventory session (copied from `PRESCORE_A512_NOTE.md`; added 11 Nov 2026)

- **Role:** built the Part A pre-scoring inventory (`PRESCORE_INVENTORY.json`/`.md`) and its A5.1.2 note.
- **Session identity:** Claude Opus 5.5 (claude_umut instance), branch `prereg/prescore-inventory` (worktree
  `.worktrees/prereg/prescore-inventory`), delegated by the chair. It started from main `079e60b` on 11 Nov 2026 at 09:35:06 UTC
  (git reflog).
- **Coverage:** from branch creation through the commit that adds those files.
- **Read and ran:** the F10/F11/F11a/F11b text and re-pins (xviii)–(xxi) of `docs/PREREG_H1_TF.md`; the Board A reviews (F10,
  F10-code-2, re-stage, F11, F11-stage, F11-sim, analysis-4); `f10_attestations.md`; the `part_a/` manifests, audits, masks and
  records; `selection_f11.json`; `code_hashes.json`; `scripts/part_a_score_launch.sh`; PROGRESS entries. It re-hashed staged bytes
  under `data/runs/a5/part_a/` and the simulation outputs and inputs under `data/runs/a5/power/` and `data/runs/a5/nullsize/`. It
  ran `--verify-pins HEAD` and `nvidia-smi` queries (identity only), on CPU only.
- **Attestation (quoted from `PRESCORE_A512_NOTE.md`):**

> Under `data/evals` I ran existence (`lexists`) and name-walk checks only and opened nothing. I opened no zoo `eval.json`,
> `records.jsonl` or `summary.json`, and wrote nothing under `data/`. I used no GPU for computation and ran no scoring, staging,
> simulation or analysis. I have seen no G3, rotation, ladder, reference or sealed outcome, or a summary of one; none exists. I made
> no methodological choice. This attests this session only.

## 16. Simulator `--supplement` implementer (added by the F11c drafter from the chair's PROGRESS record; 11 Nov 2026)

- **Role:** built `--supplement`, `--registered-40-sensitivity` and the 52-id `--slot-mask` rules in `scripts/a5_simulation.py`
  (26 simulator tests).
- **Session identity:** branch `feat/a5-sim-supplement`, commit `54f17fc` (11 Nov 2026 07:34:59 UTC, git); merged after Board A's
  acceptance (`board-a-f11-sim-2026-11-11.md`) and pinned as re-pin (xix). The model and session id are not in the chair's record.
- **Coverage:** through commit `54f17fc`.
- **What the chair's record says** (PROGRESS, "11 Nov 2026, 09:45 UTC"):

> Simulator `--supplement` committed (54f17fc, `feat/a5-sim-supplement`; 26 simulator tests; byte-identical without the option,
> golden-hash regression; `--registered-40-sensitivity`; supplementary slots 40–51 appended in cell order; the supplementary edits'
> own donor rows stay in the pools per F11.5 — slots draw cell donors at random, never their own edit). Smoke run (200 replicates,
> synthetic 52-id mask with the 9 F10 exclusions masked): ordered power 0.86 [0.80, 0.91], null ordered go 0.02, null alternative
> 0.255 (reported, not gated); registered-40 sensitivity 0.73.

- **Exposure:** the record describes a smoke run on a synthetic mask over pre-G3 inputs. It states no exposure line for this
  implementer. Board A's review of the commit (entry 20) states that it read development-proxy simulation evidence and opened nothing
  under `data/evals`. This file does not prove the implementer's own history, and records that the statement is on file only through
  the chair.

## 17. `prepare --supplement` implementer (added by the F11c drafter from the chair's PROGRESS record; 11 Nov 2026)

- **Role:** built `prepare --supplement` in `src/wsp/eval/response_intervention.py` (loader bound to the frozen `selection_f11.json`,
  forty re-hashed before and after, merged 52-id manifest, `*_pre_f11.*` preserved, deferred reuse proofs) and the slot-mask builder.
- **Session identity:** branch `feat/f11-supplement-stage`, commits `c914270` (07:46:50 UTC) and `bef5396` (07:58:28 UTC, git); merged
  after Board A's acceptance (`board-a-f11-stage-2026-11-11.md`) and pinned as re-pin (xx). The model and session id are not in the
  chair's record.
- **Coverage:** through commit `bef5396`.
- **What the chair's record says** (PROGRESS, "11 Nov 2026, 10:50 UTC"):

> `prepare --supplement` committed (c914270, `feat/f11-supplement-stage`; 133 tests on the two files; dry run on one unselected
> LoRA root: rotation + two ladders 581 s under load 17–23; re-hash of the 40's 249 staged files 895 s, all matching; mask builder
> reproduces `slot_mask_f10.json` byte for byte; seeds derive from `control_seeds.json:rotation` by canonical id exactly as for the
> 40). [...] reference proofs for the 12 supplementary originals, which read TF-v1 records including score rows, are deferred to
> scoring time (`--no-reference-proofs` default; `score --manifest` proves first and refuses unproven supplementary variants), so
> staging touches no evaluation record.

  and (PROGRESS, "11 Nov 2026, 11:15 UTC"): "`prepare --supplement` fixes committed (bef5396): loader bound to the frozen
  `selection_f11.json` [...]; reuse proofs deferred to `score --manifest` [...]; default priced count ≤ 178 entries."
- **Exposure:** the record says staging touches no evaluation record. It does not say what the implementer opened while building. It
  ran a dry run on one unselected LoRA root and re-hashed the forty's staged files; where that wrote is not stated. Board A's staging
  review (entry 20) found no outcome read. The statement is on file only through the chair.

## 18. Part A analysis builder (added by the F11c drafter from the chair's PROGRESS record; 11 Nov 2026)

- **Role:** built the manifest-binding repair and the F11 path of `src/wsp/eval/part_a_analysis.py` and `scripts/part_a_analyse.py`
  (supplements in the F10.5 reports, registered-40 report, per-contrast masks, F11a screen route).
- **Session identity:** branch `fix/part-a-analysis-manifest-binding`, commits `aef0a83` (07:38:36 UTC) and `f57b70a` (08:05:49 UTC,
  git); merged after Board A's acceptance (`board-a-part-a-analysis-4-2026-11-11.md`) and pinned as re-pin (xxi). The model and
  session id are not in the chair's record.
- **Coverage:** through commit `f57b70a`.
- **What the chair's record says** (PROGRESS, "11 Nov 2026, 08:06 UTC"):

> Analysis F11-path corrections committed (f57b70a on `fix/part-a-analysis-manifest-binding`; 172 analysis tests; full suite 1110
> passed): supplements join the covariate, sign, comparator and tertile reports with the forty's frozen cells and S30/S34 unchanged;
> registered-40 descriptive report with availability-vs-resolved and root/non-root splits; per-contrast masks (supplementary ladder
> masks accepted); F11a screen route with size-failure priority and the F11.7 validity wording tested against the registration.
> Builder decisions for the adopting note: forty-only size failure now refuses; the screen route needs an `f11a_screen_route`
> acceptance input bound to the calibration-report hash; the 52-slot calibration is always required when supplements exist;
> supplement cell assignment and the acceptance-input format are conventions to adopt.

  and (PROGRESS, "11 Nov 2026, 10:20 UTC") that Board A "judged the analysis manifest-binding repair sound (aef0a83)".
- **Exposure:** the record states no exposure line. The tests used synthetic fixtures. Board A analysis-4 (entry 20) notes that
  temporary synthetic fixture trees named `data/evals` in the full suite are not repository evaluation artifacts. The statement is on
  file only through the chair. The builder's decisions were adopted before any outcome in F11b.

## 19. Chair — extension 2 (added by the F11c drafter from the chair's statement; 11 Nov 2026)

Quoted verbatim from the chair:

> Chair extension 2. From about 09:20 UTC to the time of this note I launched the supplementary staging (08:06 UTC by the system
> clock, execution SHA 8211ecf) and the F11.4 calibration (09:28 UTC, SHA 92b0338) through launch scripts I wrote (committed at
> 516d9c9), wrote the slot-mapping record after the calibration, enriched the run records, merged the inventory, and committed the
> gated scoring launch script. I opened no file under data/evals for any variant, supplementary edit or missing reference, ran no
> scoring, and read no score, response or contrast. No Part A outcome exists.

The statement extends entry 1 and its first extension. The staging launch time (08:06 UTC) precedes "about 09:20 UTC" because
PROGRESS labels before 08:12 UTC were estimates (Timestamp note 2); the system-clock times govern. The gated scoring launch script is
`scripts/part_a_score_launch.sh` (commit `1b954a6`, 09:29:04 UTC, not executed).

## 20. Board A reviewers, F11 chain (copied verbatim from their review files; added by the F11c drafter, 11 Nov 2026)

Each line attests only its own session. F11a.4 already incorporated the first by reference. Quotations are from the final lines of
the named files.

1. `docs/reviews/council/board-a-f11-2026-11-11.md` (session `review/board-a-f11`):

> **Board A exposure attestation:** role ML methods/pre-registration reviewer; session/worktree `review/board-a-f11`, this review turn on 11 November 2026, through completion of this document. I have not opened, computed, scored or received any G3, rotation, ladder, missing-reference or sealed response outcome, or a summary of one. I read the supplied task and repository methodological, code, committed metadata and simulation-report evidence described above. I opened nothing under `data/evals`, used no GPU and wrote nothing under `data/`. This line attests only this reviewer/session; it does not prove other decision-makers' histories.

2. `docs/reviews/council/board-a-f11-sim-2026-11-11.md` (session `review/board-a-f11-sim`):

> **Board A exposure attestation:** ML methods/pre-registration reviewer; session/worktree `review/board-a-f11-sim`, this review turn on 11 November 2026 through completion of this document. I received/read no G3, rotation, ladder, missing-reference or sealed response outcome or summary. I read methodological/code/metadata and development-proxy simulation evidence only, opened nothing under `data/evals`, and used no GPU. This attests only to this session; it does not certify anyone else's exposure history.

3. `docs/reviews/council/board-a-f11-stage-2026-11-11.md` (session `review/board-a-f11-stage`):

> **Board A exposure attestation:** ML methods/pre-registration reviewer, session/worktree `review/board-a-f11-stage`, 11 November 2026 through completion of this report. I read code, tests, registration/review documents, metadata and construction-only scratch timing/identity evidence. I received/read no affected G3, rotation, ladder, missing-reference or sealed response outcome or summary, opened nothing under `data/evals`, and used no GPU. Synthetic test outcomes are fixtures, not experiment outcomes. This attestation covers this reviewer session only.

4. `docs/reviews/council/board-a-part-a-analysis-4-2026-11-11.md` (session `review/board-a-part-a-analysis-4`):

> **Exposure attestation:** Board A ML methods/pre-registration reviewer, session/worktree `review/board-a-part-a-analysis-4`, 11 November 2026 through this report's completion. No new G3, rotation, ladder, missing-reference or sealed response outcome, or summary of one, was opened, computed, scored or received. Repository evidence inspected here was methodological text, code, tests and committed outcome-free metadata/calibration evidence. No GPU use, repository data writes or repository `data/evals` access. Temporary synthetic fixture trees named `data/evals` in the full suite are not repository evaluation artifacts. This attests only this reviewer/session, not other decision-makers.

The re-stage reviewer's line (`board-a-f10-restage-2026-11-11.md`) is not copied here; the inventory lists that reviewer as pending too,
and the chair did not name that file for this entry.

## 21. F11c drafter (this drafting session; added 11 Nov 2026)

- **Role:** wrote note F11c (F11.8 clarification and the slot-mapping, main-advance and record-enrichment disclosures) and entries
  15–21 of this file.
- **Session identity:** Claude Sonnet 5.5, branch `prereg/f11c` (worktree `.worktrees/prereg/f11c`), delegated by the chair; created
  from main `516d9c9` at 11 Nov 2026, about 10:00 UTC (system clock).
- **Coverage:** from branch creation through the commit that adds this entry.
- **Read and ran:** `PRESCORE_INVENTORY.md`, `PRESCORE_A512_NOTE.md`, the F11, F11a and F11b blocks and re-pins (xx) and (xxi) of
  `docs/PREREG_H1_TF.md`, this file, `docs/PROGRESS.md` (the 11 Nov entries from 08:06 UTC onward), the four Board A reviews named in
  entry 20 (their attestation lines and the passages quoted above), and `supplement_launch.json`, `f11_calibration_record.json` and
  `slot_mapping_f11.json` (committed metadata). It ran git commands and `scripts/secret_scan.sh`. It ran no scoring, GPU job,
  simulation, staging or test suite, and wrote only under `docs/`.
- **Attestation:** I opened nothing under `data/evals/`, opened no zoo `eval.json`, `records.jsonl` or `summary.json`, used no GPU,
  and wrote nothing under `data/`. I have seen no G3, rotation, ladder, reference or sealed outcome, or a summary of one. None
  exists. I made no methodological choice: F11c copies the inventory, Board A's rulings and the chair's threshold and statements.
  This attests this session only; it does not prove other sessions' histories.
