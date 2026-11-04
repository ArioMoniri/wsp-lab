# Board B re-review: paper/intro-related

Reviewed head b4433da against 0f9df52. Verdict: **APPROVE** for merging the current pre-results draft of these sections, not for submission.

Scores (1–5): claims-supported **4**, novelty **3**, clarity **4**, reproducibility **4**. Claims and reproducibility are scored against what this proposal asserts; no experimental results or completed protocol are claimed. Novelty is a proposed combination, not an established empirical contribution.

## Prior findings

1. Closed. The task-vector sentence now follows the stored excerpt: interference collapses under the training-format wrapper, not under the untrained chat template. The recorded claim was corrected too. Rechecked the remaining cited sentences against the stored claim records: takehi2026wsp, han2026survey, w2t2026, deltaactivations2025, fastprobing2026, tuneahead2026, predictability2026dynamics, leapverify2026, mamba3, jepawg2026physics, drivedecay2026, jepahmm2026, heretic2026, pksharma2026heretic, surgicalrefusal2026 and singleattack2026moe. Their stated claims remain within the recorded claims. This is a review of the stored records, not independent source verification.
2. Closed for anonymity. Page 1 says anonymous authors; title, author, subject and keywords in pdfinfo are empty. Technical metadata (creator, producer and timestamps) is present, so metadata is not literally empty. The switch in main.tex documents how to enable the identified preprint.
3. Closed. H1 includes both baselines and its falsification threshold; H2 includes trajectory length, relative MAE and within-noise falsification; H3 includes the transfer and falsification thresholds; H4 states both directions and the same held-out sets. These agree in meaning with HYPOTHESIS.md. H5 exists there and need not appear in this draft, but the introduction should mention it when that study is added; its present scope excludes optimization, so the paper's scope must be reconciled with H5 before submission.
4. Closed. The three flagged sentences have been rewritten, planned work is presented as proposed or future work, and the abstract's question is shorter. The title now signals a proposed study.
5. Closed under the clarified constraint: numerical prior results are attributed to verified citation records, and decision thresholds are pre-registered. Model identifiers and counts describe the proposed setting or cited studies. No invented experimental results or placeholder numbers found.

## Build and remaining work

Ran the requested latexmk command successfully and read all four PDF pages with pdftotext. The reference checker did not pass in this worktree: it reports missing source files and excerpt mismatches across entries. Restore/access the source corpus and obtain a passing `python scripts/verify_refs.py` before the protocol's paper compilation step; the stored-record audit above does not replace that check.

Before submission, write the methods and experimental setup, data and edit-zoo construction, splits and baselines, evaluation and statistical procedures, results (including negative results), discussion and specific limitations, reproducibility details including configurations, run IDs and artifacts, and conclusions. Reassess claims and citations against the final evidence. No manuscript files were edited for this review.
