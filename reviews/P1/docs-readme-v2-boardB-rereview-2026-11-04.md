# Board B — README v2 re-review

Verdict: **APPROVE**. Branch `docs/readme-v2`, head `2e642547daf60ce445cd1eeb141ff77706ede65d`; reviewed 4 November 2026.

Scores (1–5): claims-supported **4**, novelty **3** (editorial work), clarity **4**, reproducibility **4**.

Read the reviewer-journal role, WRITING_GUIDE, REVIEW_PROTOCOL, prior Board B review, the requested `43a5c08...2e64254` diff for README, PROGRESS, INFRA and precommit, and the whole README and PROGRESS. No reviewed-branch edits retained. The requested updater necessarily wrote README temporarily; its exact original bytes were restored in a finally block. Pre-existing untracked data symlink left alone.

## Four corrections

1. **D3: fixed.** README and PROGRESS now attach 9–12 minutes/entry and approximately 120 GPU-hours to the historical 700 entries, as docs/DECISIONS.md D3 does. README explicitly calls this a projection from smoke timing and keeps clean timing pending. D3's 60/100/60/30/50 allocation sums to 300, with approximately 50 hours as a projection. Live manifest had 1,136 unique IDs at the review snapshot, consistent with the revised “over 1,100” wording. The committed 1,131 count is timestamped earlier, not a claim of an unchanging total.
2. **Extraction eligibility: fixed.** `_manifest_ids` in src/wsp/features/extract.py validates IDs and row structure without filtering eligibility. `run_extract` applies optional explicit IDs/limit and artifact readiness, validates metadata, records failures, and resumes completed entries. README and generated status now correctly distinguish extraction from batch evaluation, whose default eligibility is production. “Skips only” is colloquial shorthand for eligibility-independent admission, not an exhaustive description of errors/resume/options; optional precision improvement below.
3. **Hooks: fixed.** .githooks/pre-commit runs secret_scan then scripts/precommit.sh; that script invokes ruff check, ruff format --check and pytest. .githooks/pre-push runs only secret_scan.sh --history. README and PROGRESS describe that division. INFRA accurately describes staging README after successful refresh; the code warns that unstaged README hand edits are included too. No claim that formatting is enforced before every push remains.
4. **Smoke evaluation: fixed.** Read all three legacy-smoke eval.json files and their manifests. Run 20261104-032002-1c2cc588 contains base plus adf416337f5a2277 and bccb9ab10b5661e0, four items per benchmark, max_new_tokens=32. Canonical zoo entry directories contained zero eval.json files; refreshed canonical indices had zero rows. README and PROGRESS now acknowledge the engineering scores and distinguish them from completed full fast/cheap targets. The earlier blanket “none exist yet” claim is gone. These tiny-budget scores do not support full-tier predictor conclusions.

## Validation

Ran `source scripts/env.sh`, `python scripts/update_readme_status.py`, then `pytest -q -k status`: **4 passed, 233 deselected**, 11.76 seconds, three dependency deprecation warnings. Restored README afterwards. Generated snapshot at 2026-11-04T06:30:03Z reported 1,136 entries, 689 extracted features (finished), zero canonical eval rows, one open Lambda ledger row, and pending CI pipeline 3025730218 on 64955bf. These are observed status, not new benchmark measurements; “finished” describes the extractor run, not coverage of the growing manifest.

The first README hero line is byte-for-byte equal to main's `assets/hero.svg` line. Neither README nor PROGRESS references `pipeline.gif`; the removed README line is absent at reviewed head.

## Prose and non-blocking precision

All four previously quoted README scaffolding/inventory sentences have been removed or rewritten. The family table and shorter next-step paragraph resolve the crowded inventories. The previous notebook fragments were expanded. On a full reread I found no remaining sentence I would block as machine-sounding. Some historical notebook fragments remain, but they fit dated lab notes rather than promotional prose.

Two optional precision edits remain (neither reinstates one of the four wrong claims):

> “Extraction reads every valid manifest id, including `smoke` and `unlabelled` entries, and skips only those whose files are not built yet; filtering by eligibility is left to whoever trains on the features.”

Drop “only” or say “does not filter by eligibility”: explicit ID/limit selection, metadata failures and resume checks also affect processing, as described above.

> “It says nothing about what the edits did to benchmark scores.”

Prefer “It does not establish performance at the full fast or cheap tier.” The smoke did record benchmark scores at n=4/32 tokens; the intended limitation is their evidentiary strength, not their existence. PROGRESS's “not benchmark performance” is similarly rhetorical shorthand.

PROGRESS's CI paragraph also has a mechanical capitalization error, “through; The formatting check”. Its Lambda paragraph says the ledger was empty “at the time of writing”, whereas the current ledger and generated README have a row starting 06:13:12Z; retain that sentence only as a historical note. Neither affects the corrected enforcement or evaluation claims.

Approval covers this documentation revision. It does not certify completed zoo evaluation, completed predictor training, or external human review.
