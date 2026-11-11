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
