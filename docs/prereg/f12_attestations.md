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
| 10 | **Chair** (approves F12, F12a, F12b, F12c, the pilot projection and the CPU repricing) | — | through the final freeze | — | — | — | — | **PENDING**: the chair writes this entry |
| 11 | Every later C1/C2 builder, analyst and Board A reviewer approving an F12 parameter or the simulation report | — | — | — | — | — | — | **PENDING** |

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
