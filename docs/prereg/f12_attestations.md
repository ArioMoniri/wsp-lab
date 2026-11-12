# F12 exposure attestations (Board A F12b review condition 1; F12a.1(c)-(d); F8a.1(c))

**Status: 11 Nov 2026, before any G3 outcome. Incomplete register: stubs marked PENDING must be completed by the session named,
not by anyone else.** The register lists every session known to this file's author that made, implemented or approved an F12
choice through F12c. It is needed through the final F12 freeze (C1, C2, the simulation report and the analysis). Board A's F12b
review (`docs/reviews/council/board-a-f12b-2026-11-11.md`, condition 1) asks for "all required decision-maker attestations through
the final freeze, including the disclosed prescore-audit qualification in original condition 1". An attestation by one session
does not establish another session's access history. A commit time earlier than an outcome is not an attestation (F12a.1(d)).

Each attestation makes two statements (F12a.1(c)):

- **(i) Old development outcomes.** Whether the session has seen the 123 existing TF-v1 root records, the G0/G1/G6
  exposure-ledger outcomes or D3 development outcomes. These are disclosed and do not disqualify.
- **(ii) New outcomes.** Whether the session has opened, computed, scored or received any Part A (G3, rotation, ladder,
  missing-reference, sealed), F8-E, F8-F or F9 outcome, or a summary of one. These disqualify.

It also gives **(iii)**, the prescore-audit qualification (original Board A F12 condition 1; F12a.1(a)): whether the session read
the disclosed pre-score audits. These are Part A's frozen predictions and simulated calibration as summarised in F12.1 and the
round-6 member reports. They are predictions and simulation outputs, not outcomes.

## Register

| # | session and role | identity (as known) | covers | (i) old dev. outcomes | (ii) new outcomes | (iii) prescore audits | source | status |
|---|---|---|---|---|---|---|---|---|
| 1 | F12 drafting (council round 6 implementation sub-agent) | branch `council/2026-11-11` | drafting of F12 | none opened; only commit metadata and record hashes | none | read the round-6 synthesis and member reports, which summarise Part A predictions and calibration | `docs/PREREG_H1_TF.md`, end of amendment F12 | given (its own session only) |
| 2 | Council round 6 chair and members (F12 choices) | `docs/council/2026-11-11/` | round 6 | — | — | — | — | **PENDING** |
| 3 | Board A, F12 / F12-I review | `docs/reviews/council/board-a-f12-2026-11-11.md` | that review | — | — | — | no attestation text found in the review | **PENDING** |
| 4 | F12a drafting (implementation sub-agent) | Claude Opus 5.5, branch `prereg/f12a`, about 12:00-12:50 UTC | F12a | none | none | read F12a.1's account of the audits | `docs/PREREG_H1_TF.md`, end of note F12a | given (its own session only) |
| 5 | F12 inputs implementer (`f12_population.py`, `f12_tokens.py`, `f12_additivity_prepare.py`; commit `0412799`) and the token-run sessions (`agent-tmux` jobs `f12-tokens2`, `f12-tokens3`) | unknown | pinned C1 code and its runs | — | — | — | — | **PENDING** |
| 6 | F12 simulation implementer (`f12_simulation.py` at `07e1dc2`; the synthetic smoke run that wrote the restricted payload copy) | branch `feat/f12-simulation` | pinned code; restricted-copy extractor | — | — | — | — | **PENDING** |
| 7 | F12b drafting (implementation sub-agent) | branch `prereg/f12b`, about 14:10-18:10 UTC | F12b, its code and synthetic diagnostics | none opened | none | not stated | `docs/PREREG_H1_TF.md`, end of amendment F12b | given (its own session only); (iii) **PENDING** |
| 8 | Board A methods reviewer, F12b review | `review/board-a-f12b` worktree | that review | not stated | none | not stated | the review's "Exposure attestation" | given (its own session only); (i), (iii) **PENDING** |
| 9 | F12c implementer (this file's author) | Claude Opus 5.5 implementation sub-agent, branch `prereg/f12c`, worktree `.worktrees/prereg/f12c`, 11 Nov 2026 about 18:10 UTC to the F12c commit | F12c note, code, tests, manifests, synthetic checks | see below | none | see below | below | given (this session only) |
| 10 | **Chair** (approves F12, F12a, F12b, F12c, the pilot projection and the CPU repricing) | orchestrating session of the programme, 11 Nov 2026 | through 21:50 UTC 11 Nov 2026; to be renewed at the C1 freeze | seen (council briefs, syntheses, earlier development results) | none | read and commissioned | entry 10 below | given to 21:50 UTC; **renewal PENDING** at C1 |
| 11 | Every later C1/C2 builder, analyst and Board A reviewer approving an F12 parameter or the simulation report | — | — | — | — | — | — | **PENDING** |
| 12 | F12d implementer and v2 restricted-payload extractor (chair-authorised for that step) | Claude Opus 5.5 implementation sub-agent, branch `prereg/f12d`, worktree `.worktrees/prereg/f12d`, 11 Nov 2026 about 21:45 UTC to the F12d note commit | F12d note, code, tests, v2 restricted copy and pin, C1 skeleton, resource plan | none seen; handled centred qualification noise (see below) | none | read F12a.1's account and entries 9-10, not the audits | entry 12 below | given (this session only) |
| 13 | F12e implementer (C1 draft fill, candidate identity freeze, real-token qualification) | Claude Opus 5.5 implementation sub-agent, branch `prereg/f12e`, worktree `.worktrees/prereg/f12e`, 11 Nov 2026 about 23:52 UTC to the F12e note commit (12 Nov) | F12e note, `scripts/f12_qualify.py`, the C1 draft fill, token facts, base tensor hashes, candidate identities, qualification report | none seen; read the real F12 token outputs (field forecasts, see below) | none | read F12a.1's account and entries 10 and 12, not the audits | entry 13 below | given (this session only) |
| 14 | F12f implementer (gauge-token builder and pricing, runner gauge audit, comparator-matrix builder, re-qualification code) | Claude Opus 5.5 implementation sub-agent, branch `prereg/f12f`, worktree `.worktrees/prereg/f12f`, 12 Nov 2026 about 00:33 UTC to the F12f note commit | F12f note, `scripts/f12_gauge_tokens.py`, `scripts/f12_matrix.py`, the gauge audit in `scripts/f12_simulation.py`, `scripts/f12_qualify.py` / `f12_c1.py` changes, priced-only gauge records | none seen; read the real F12 token outputs and the edit-derived v4 feature files (see below) | none | read F12a.1's account and entries 9, 10, 12 and 13, not the audits | entry 14 below | given (this session only) |

PENDING cells are blank on purpose. Nothing is inferred from a session's role, branch or timing.

## Entry 9: F12c implementer (given)

*Session.* An implementation sub-agent session (Claude Opus 5.5) working only in `.worktrees/prereg/f12c` on branch `prereg/f12c`,
from main `8df6c74`, on 11 Nov 2026 from about 18:10 UTC until the commit adding note F12c. It attests to this session only.

*What it read.*
- Committed files: Board A's F12b review; amendment F12, note F12a and amendment F12b of `docs/PREREG_H1_TF.md`;
  `docs/design/F12B_REPAIRS.md`; the F12 and F12b scripts, tests and configurations; `docs/prereg/f10_attestations.md` (format);
  `scripts/wt.sh` and `scripts/secret_scan.sh`.
- Synthetic diagnostics: the F12b reduced-run and timing files under `wsp-data/runs/a5/f12_sim_dryrun/f12b/`. It read their
  summaries and timing fields and computed their hashes.
- Logs: the first lines of `progress.log` from the F12 synthetic smoke run.
- The restricted noise payload copy `wsp-data/runs/a5/f12_sim_dryrun/noise_payload_restricted.json`: its provenance block,
  donor ids, components and families; it hashed the bytes and ran the noise-only checks.

*Disclosure.* Once, at about 18:11 UTC, the session ran the existing F12 test suite to get a baseline. One of those tests,
`test_real_payload_allowlist_and_panel_mean`, parsed the **raw** F8a.2 payload `data/runs/a5/nullsize/inputs.json`. It kept only
the allowlisted fields (`centred_noise`, `window_mask`, `window_ids`, donor id/component/family) and printed no value. The session
had been told to read only the restricted copy. The test now reads the pinned restricted copy instead (F12c.6). The session opened
no other file under `data/`.

*(i) Old development outcomes.* The session saw no TF-v1 record, no G0/G1/G6 ledger outcome and no D3 outcome. It handled
centred, donor-mean-free qualification noise derived from old development records: through the restricted copy and, once, through
the raw payload as just described.

*(ii) New outcomes.* None. The session opened nothing under `data/evals`. It saw no G3, rotation, ladder, missing-reference,
sealed, F8 or F9 outcome and no Part A score, and received no summary of one. It used no GPU, read no real F12 token and ran no
stage 1, stage 2, timing pilot or real-token job.

*(iii) Prescore audits.* The session read F12a.1's account of the disclosed pre-score audits, which names the Part A prediction and
calibration files and quotes F11.4's simulated 0.8722 ordered power. It did not open those files, the round-6 member reports or
`SYNTHESIS-r6.md`.

## Entry 10: chair (given to 21:50 UTC, 11 Nov 2026; renewal pending at the C1 freeze)

*Session.* The orchestrating chair session of the wsp-lab programme on 11 Nov 2026. It commissioned and merged the council
round 6 synthesis, amendment F12, note F12a, amendment F12b, note F12c and the chair ruling on F12c, and launched the F12 token
run (`agent-tmux` jobs `f12-tokens`, `f12-tokens2`, `f12-tokens3`). It attests to this session only, up to 21:50 UTC, and must
renew this entry at the C1 freeze.

*(i) Old development outcomes.* Seen. Across the programme the chair has read council briefs, syntheses and reviews that quote
old development results, including the 342-row predictor comparison (descriptor-only 0.51/0.61, StatNN 0.56/0.66), the M3 latent
dynamics result (rho 0.15/0.13 on 25 evaluated successor pairs), the D3 results and the TF-v1 qualification summaries used for the
F7 power donors and the F8a.2 noise floor. This is disclosed development exposure, which does not disqualify under F12a.1.

*(ii) New outcomes.* None. Part A is unscored and held by the programme owner. The single Part A phase-1 launch attempt (CPU
proofs-only, about 11:41 UTC) refused itself at its clean-tree check before reading anything, and no relaunch was made. The chair
opened nothing under `data/evals` for Part A variants or references, and received no G3, rotation, ladder, missing-reference,
sealed, F8 or F9 outcome or summary, because none exists. No F12 or F12-I purchase, scoring or GPU job was run.

*(iii) Prescore audits.* Read and commissioned. The chair commissioned the Part A prescore inventories (`PRESCORE_INVENTORY.md`,
`PRESCORE_INVENTORY_2.md` and their notes), merged the frozen predictions `predictions_f11.json` and the F11.4 calibration records,
and read the round-6 synthesis, which summarises those predictions (for example kappa median 0.023 at s = 1). These are the
disclosed prescore audits named in F12a.1.

*Disclosure: real F12 field values.* While monitoring the token run the chair read its log, whose per-row lines print the row id,
family, the field's first-order term a, the curvature term q and the token count. These are field forecasts computed from the
frozen G2 field, not outcomes. The chair also read the run's `pricing.json` and listed the token directory's file names; it opened
no token, detail or K-frame file. No F12 choice was made from those values: every F12 parameter was fixed in F12, F12a, F12b or
F12c before the token run produced its first row at 16:46 UTC.

*Decisions this entry covers.* The F12b repairs brief, the F12c brief, the 190 CPU-h all-in repricing (18:55 UTC) and the
ordering that keeps stage 1 held. None of them used a Part A, F8, F9 or F12 outcome.

## Entry 12: F12d implementer and restricted-payload extractor (given)

*Session.* An implementation sub-agent session (Claude Opus 5.5) working only in `.worktrees/prereg/f12d` on branch `prereg/f12d`,
from main `6e3fe0b`, on 11 Nov 2026 from about 21:45 UTC until the commit adding note F12d. The chair authorised it as the extractor
of the v2 restricted payload copy (Board A F12c review, condition 6). It attests to this session only and writes no other entry.

*What it read.*
- Committed files: Board A's F12c and F12b reviews; notes F12a, F12c, amendment F12b and the chair ruling on F12c in
  `docs/PREREG_H1_TF.md`; this register; `scripts/f12_simulation.py` and the F12 tests; the F12 manifests; the head and argument
  lists of `scripts/f12_population.py`, `scripts/f12_tokens.py` and `scripts/f12_additivity_prepare.py`; the `materialise` and
  `inputs_provenance` functions of `scripts/a5_nullsize_inputs.py` (to learn the payload layout) and
  `scripts/a5_simulation.py:donor_block`; `docs/prereg/f8_manifests/floor_payload/README.md`; the key names, counts and rules of
  `docs/prereg/a5_manifests/f12/population.json`; the committed metadata extract
  `docs/prereg/a5_manifests/nullsize/inputs_provenance.json` and the window order of
  `docs/prereg/a5_manifests/nullsize_inputs.json`. From the metadata extract it printed only the donor and inventory counts by
  eligibility flag and family, the component count, the number of donors with masked windows (0) and the window ids. It did not
  print norms or audit fields, and the file holds no noise values.
- `data/`: it listed the entries of `data/runs/a5/nullsize/` (file names, sizes, dates) and computed the sha256 of
  `data/runs/a5/nullsize/inputs.json` without parsing it, to confirm byte identity with the committed source. It opened no other
  file under `data/`.
- Noise payloads:
  - It ran the committed extractor once on the committed source. The extractor decoded only `window_ids`, `donors` (id,
    component, family, eligible) and the centred noise and masks; every other value, `donor_mean` included, was skipped
    undecoded. The output was the v2 copy and a report of counts, donor ids of exclusions (none) and key names.
  - It read the superseded v1 copy and the v2 copy (allowlisted fields only) for the membership comparison and the loader check.
    The comparison tested equality of arrays; no noise value was printed.
- Synthetic and dry-run evidence: it listed `wsp-data/runs/a5/f12_sim_dryrun/` and used only its own test outputs. It did not open
  the F12b or F12c synthetic reports.
- Test runs: it ran the F12 test files serially (OMP 2), and the repository's pre-commit hook ran the full test suite at its code
  commit. The suite's collection-time file access was not audited, as Board A noted for its own session; no test output showing an
  outcome was seen.

*Incident.* Its first implementation commit attempt ran the pre-commit hook, whose inherited `GIT_DIR` / `GIT_INDEX_FILE` sent the
C1 test fixtures' git commands to this worktree's repository. They made 58 toy commits on `prereg/f12d` and set `core.bare = true`
in the shared configuration for about four minutes. The session reset its branch and restored the setting (note F12d). No file
content of interest to F12 exposure was read or written by that incident.

*(i) Old development outcomes.* The session saw no TF-v1 record, no G0/G1/G6 ledger outcome and no D3 outcome. It handled centred,
donor-mean-free qualification noise derived from old development records only through the allowlisted extractor, the restricted
copies and tests, and printed none of it.

*(ii) New outcomes.* None. The session opened nothing under `data/evals`. It saw no G3, rotation, ladder, missing-reference,
sealed, F8 or F9 outcome and no Part A score, and received no summary of one. It did not list or read the real F12 token directory,
used no GPU, and ran no registered noise check, timing pilot, stage 1, coverage target, stage 2 or real-token job.

*(iii) Prescore audits.* The session read F12a.1's account of the disclosed pre-score audits and entries 9 and 10 of this register,
which describe them. It did not open the Part A prediction or calibration files, the prescore inventories, the round-6 member
reports or `SYNTHESIS-r6.md`.

## Entry 13: F12e implementer (given)

*Session.* An implementation sub-agent session (Claude Opus 5.5) working only in `.worktrees/prereg/f12e` on branch `prereg/f12e`,
from main `4565743`, from 11 Nov 2026 about 23:52 UTC until the commit adding note F12e on 12 Nov 2026. The chair's brief
authorised it to read the real F12 token outputs under `/home/ubuntu/wsp-data/runs/a5/f12_inputs/tokens/` after verifying them
against the committed manifest. It attests to this session only and writes no other entry.

*What it read.*
- Committed files: Board A's F12c review; notes F12a, F12c, F12d, amendment F12b and the chair ruling on F12c, and the A5.4,
  F5.1, "F5 schema pins" and F12.3 passages of `docs/PREREG_H1_TF.md`; this register (header, table, entries 10 and 12); the
  F12 scripts, tests and manifests, including `c1_draft/`; `docs/prereg/a5_manifests/f12/tokens_manifest.json` and
  `population.json`; `src/wsp/features/pool.py` (head) and `response_field._tensor_hash`; the repository hooks.
- **Real F12 token outputs (field forecasts and edit-derived inputs, not outcomes).** It verified the sha256 of all seven files
  against the committed manifest (all equal) and then read `tokens.safetensors`, `rows.jsonl`, `detail.jsonl`, `kframes.json`,
  `aq.json` and `pricing.json` through `scripts/f12_qualify.py`. Their values entered the qualification checks, the token facts
  and the omitted-energy summary. It printed key names, counts, hashes, flags, omitted-energy quantiles by family and k bin, and
  the 16 LoRA roots' tensor counts. It printed no field contraction a or q value and no token value.
- The pinned base model (`Qwen/Qwen3.5-0.8B` at `2fc06364…`) was loaded once on CPU by `a5_power_fields.load_base_state`, which
  re-verified its state hash, to hash 72 base tensors.
- `data/`: it listed `data/` and `data/hf/hub` (names), one zoo entry directory (`delta.safetensors`, `edit.json`; names and
  sizes, not opened), and tested the existence of `data/features/qwen3.5-0.8b/<id>/pooled.safetensors` for the 270 rows without
  opening any. A `find data/ -maxdepth 4 -name 'pooled*'` with the `data/evals` paths filtered out of its output still walked
  directory listings under `data/evals`; no file there was opened and none of its names was printed. A `grep -rl` over
  `docs/prereg` printed the names of Part A manifest files containing `down_proj.weight`; it opened none of them.
- It opened no noise payload (restricted or raw) and ran no noise check, pilot or simulation replicate.
- Test runs: the F12 test files serially (OMP 2), and the pre-commit hook's full suite at three commits; the suite's
  collection-time file access was not audited. No test reads the real token directory.

*(i) Old development outcomes.* None seen. It saw no TF-v1 record, no G0/G1/G6 ledger outcome and no D3 outcome. The token
outputs include field forecasts (a, q) for the 123 existing roots; these are forecasts, not their recorded outcomes.

*(ii) New outcomes.* None. The session opened nothing under `data/evals`. It saw no G3, rotation, ladder, missing-reference,
sealed, F8 or F9 outcome and no Part A score, received no summary of one and used no GPU.

*(iii) Prescore audits.* The session read F12a.1's account of the disclosed pre-score audits and entries 10 and 12 of this
register. It did not open the Part A prediction or calibration files, the prescore inventories, the round-6 member reports or
`SYNTHESIS-r6.md`.

## Entry 14: F12f implementer (given)

*Session.* An implementation sub-agent session (Claude Opus 5.5) working only in `.worktrees/prereg/f12f` on branch `prereg/f12f`,
from main `8eabc96`, from 12 Nov 2026 about 00:33 UTC until the commit adding note F12f. The chair's brief authorised it to read
the real F12 token outputs under `/home/ubuntu/wsp-data/runs/a5/f12_inputs/tokens/` after verifying them against the committed
manifest, and the stored zoo deltas and the existing pooled v4 files of the 270 rows after confirming that the v4 files carry no
outcome field. It attests to this session only and writes no other entry.

*What it read.*
- Committed files: the chair ruling on F12e and note F12e; the F5 schema pins ("Gauge audit", "Augmented comparators"), F12.3
  and F12a.2(h)–(i) of `docs/PREREG_H1_TF.md`; Board A's F12c review ("Ordered path to stage 1"); this register (header, table,
  entries 9, 10, 12 and 13); `f12e_qualification.json`; the F12 scripts, tests, configs and resource plan;
  `src/wsp/features/pool.py`, `extract.py` and `predictor/baselines.py` (`load_feature_table`); the snapshot's
  `content.feature_manifest` (hashes) and `tokens_manifest.json`; the repository hooks.
- **Real F12 token outputs (field forecasts and edit-derived inputs, not outcomes).**
  - The seven files were verified against the committed manifest by `f12_gauge_tokens.frozen_inputs` (all equal) before every
    read.
  - The gauge builder and a diagnostic read `tokens.safetensors`, `rows.jsonl`, `detail.jsonl`, `kframes.json` and
    `kframes.safetensors`.
  - For the diagnosis of the reproduction failure it printed, for row `0298985df48ec8f9` only, the row's family, rank, flags and
    omitted energy, two tensors' per-tensor records (including their field contractions a and q, which are forecasts) and those
    tokens' log σ columns.
  - It printed presence counts per row and K-block reproduction differences. It printed no other a, q or token value.
- **Stored zoo deltas and the base model.** Three rows' stored deltas were loaded and rebuilt into increments on CPU in each of
  four pricing attempts, and one row's in a ≤ 2-thread probe. The pinned base model was loaded (state hash re-verified) in each.
- **v4 feature files.** It verified the sha256 of `features.json`, `features.safetensors` and `pooled.safetensors` for the 270
  rows against the A5 snapshot (all equal). From `features.json` it printed key-name counts, the `eligibility` value counts, the
  `params` key names, the config, the edit ids' agreement with the row ids, and the agreement of changed supported tensors with
  the token presence. From `pooled.safetensors` it printed key names and shapes. It printed no feature value. Their values
  entered no output: the matrix was not built.
- `data/`:
  - It ran `ls data/` with the name `evals` filtered from the output (the listing itself read the top-level directory),
    `ls -d data/features/*`, and a name listing of `data/features/qwen3.5-0.8b` (first names and count).
  - It read `data/features/qwen3.5-0.8b/manifest.json`.
  - It opened nothing under `data/evals` and walked no directory below it.
- **Noise payload.** It computed the sha256 of the restricted payload copy `noise_payload_restricted_v2.json` (equal to
  `44d79403…6fb5`) and read the pin. It did not parse or print the payload and ran no noise check, pilot or simulation replicate
  on it. Tests use toy payloads.
- Test runs: the F12 test files (OMP 2, serial) and the pre-commit hook's full suite at each commit; the suite's collection-time
  file access was not audited. No test reads a real token, feature or payload file.
- Incidental: while checking machine load it read the tail of another agent's unrelated tmux job (an eICU benchmark log); nothing
  in it concerns this programme.

*(i) Old development outcomes.* None seen. It saw no TF-v1 record, no G0/G1/G6 ledger outcome and no D3 outcome. The token
outputs include field forecasts (a, q) for the 123 existing roots; these are forecasts, not their recorded outcomes.

*(ii) New outcomes.* None. The session opened nothing under `data/evals`. It saw no G3, rotation, ladder, missing-reference,
sealed, F8 or F9 outcome and no Part A score, received no summary of one and used no GPU.

*(iii) Prescore audits.* The session read F12a.1's account of the disclosed pre-score audits and entries 9, 10, 12 and 13 of
this register. It did not open the Part A prediction or calibration files, the prescore inventories, the round-6 member reports
or `SYNTHESIS-r6.md`.

*Entry 14, continuation (12 Nov 2026, about 01:17–01:43 UTC, same session, after merging main `9c94d62`).* In addition to the
above:
- It read the chair ruling on F12f.
- From the frozen token outputs, after verifying them, it printed per-rank token counts and per-family row counts by rank.
- It ran the gauge builder on four real rows in a scratch directory (`gauge_resume_check`), printing per-row ranks, token
  counts, changed-token counts and K-block changes only (all zero).
- It printed no a, q, token or feature value, opened no noise payload, ran no noise check, pilot or replicate, opened nothing
  under `data/evals` and used no GPU.
- Statements (i)–(iii) are unchanged.

*Entry 14, completion (12 Nov 2026, about 01:47–06:55 UTC, same session, after merging main `03bd2cb`).* In addition to the
above:
- It read the chair correction to the F12f ruling.
- It ran the full gauge-token build. Its log printed per-row ranks, token counts, changed-token counts and K-block changes only.
- It built the real comparator matrix from the verified token, gauge and v4 files. It printed counts, widths and hashes only.
- It ran the re-qualification (check verdicts and problem lists).
- **It opened the v2 restricted noise payload through the pinned loader** for the registered noise check and the timing pilot.
  It printed the noise check's summary statistics: panel-mean variance and within- and cross-component covariances against their
  plug-ins. These are properties of centred qualification noise, not outcomes.
- **The pilot's progress log printed each replicate's simulated decision code and simulated C_WM and C_WM,S.** These are
  outcome-free simulation outputs on synthetic Δy. They are not used, summarised or reported beyond the committed pilot report,
  and the note records timing, memory and validity only.
- No a, q, token or feature value was printed. Nothing under `data/evals` was opened. No GPU was used.
- Statements (i)–(iii) are unchanged.
