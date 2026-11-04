# Board B — README v2

Verdict: **REVISE**. Branch docs/readme-v2, head 43a5c08983cb20ca80ca81f5dc65ffd3c05c7a92; reviewed 4 November 2026.

Scores (1–5): claims-supported **3**, clarity **4**, honesty of limitations **4**, reproducibility **4**, novelty **3** (editorial work).

Read reviewer-journal, WRITING_GUIDE, AGENTS, RESULTS, DECISIONS, REVIEW_PROTOCOL and main...docs/readme-v2 diff. No reviewed-branch edits retained.

## Required corrections

1. README: “At the calibrated estimate of 9 to 12 minutes per entry that is on the order of 120 GPU-hours for the full zoo”. D3 and reviews/P2/feat-p2-batch-boardA-2026-11-04.md attach that estimate to **700** entries. The committed status already contains 1,110. Restore the historical denominator; do not present an extrapolation as a measurement. Retain the pending clean timing qualification.
2. Generated status: “`production` entries are the ones the eval and feature steps use” and “`unlabelled` rows predate the eligibility field and are excluded.” False for features: src/wsp/features/extract.py::_manifest_ids accepts all valid IDs; extract_zoo filters availability and explicit IDs, not eligibility. Batch evaluation does filter production by default. Correct the generator's prose.
3. PROGRESS: “`ruff format --check` now runs before every push.” .githooks/pre-push only runs scripts/secret_scan.sh --history. Formatting runs in scripts/precommit.sh through pre-commit. Describe the actual enforcement.
4. README “because none exist yet” (real scores) and PROGRESS “No zoo entry has an eval score” are overbroad. Real engineering smoke run 20261104-032002-1c2cc588 evaluated base plus two production edits, per RESULTS and sessions/2026-11-04-p2-batch.md. Say no completed full-tier zoo targets; distinguish engineering smoke from benchmark measurements. Empty canonical indices do not establish that no scores ever existed.

## Validation and evidence

`source scripts/env.sh && pytest -q -k status`: **4 passed, 236 deselected**, 58.48 seconds, three dependency deprecation warnings. Ran `python scripts/update_readme_status.py` twice; output identical after normalizing the timestamp. Restored original README bytes afterwards. Pre-existing untracked data symlink left alone.

Live snapshot 2026-11-04T06:19:38Z: **1,117** unique manifest IDs; production/unlabelled counts: random_dir 150/3, abliteration 200/5, heretic_ara 0/1, lora_sft 120/1, task_vector 80/0, chain 537/20. **664** feature files, complete_run=false. Canonical eval indices: **0 rows, 0 entries, 0 complete**. These match the generator. Committed 1,110 / 658 counts are earlier timestamped snapshots; refresh before merging because jobs are advancing.

Base summary data/runs/p2/base-final-4096/summary.json matches all seven displayed percentages: **15.5, 6.0, 42.0, 20.3, 51.3, 90.2, 99.0**, at 4096 tokens. Counts: GSM8K 200, MATH500/IFEval 100 each, MMLU-Pro/HellaSwag 300 each. Blog 0.016%, 7.9%, 48.1% claim matches verified takehi2026wsp and stored excerpts. BENCH_SSM supports batch 32, length 64, width 256 and 2.65/18.67 ms with shared-GPU caveat. INFRA records the eleven-minute build; D1 and the LoRA Board A review support approximately 87% rounding loss. lora_sft.py confirms Alpaca train[:5000].

README filesystem paths exist, including configs, scripts, source modules and paper sections. Run IDs and branch names are identifiers rather than missing filesystem paths. PROGRESS's paper/figures is a historical deletion. Main log confirms SSM 1494bd1, LoRA eval 9fb5e51/9e4a938, zoo 393fc18, batch 9146c5f, features bf99575, intro 2c16fec, generators b74c184, references 54184c2. The 588/697 growth snapshots are recorded in sessions/2026-11-04-p2-batch.md. Reviews corroborate rejection/fix history; those process records are not independent experimental verification.

Branch-local Lambda ledger is empty, as reported. Canonical main now has an open row starting 06:13:12Z; reconcile ledger/notebook at merge. Updated CI lookup returned pipeline 3025712780 on 5f276a8, running; the committed success is earlier history. Idempotence assumes unchanged artefacts and CI. No new benchmark measurements were made during this review.

## Prose

The rewrite is much plainer and reports absent predictors, the failed 32k run, IFEval uncertainty, rounding, correlated chains and model-only reviews. No overt marketing superlative found. These sentences still read as generated scaffolding or compressed inventories (editorial judgment, not proof of authorship):

> “The work so far is the data side: the zoo of edited models, the evaluation battery, the feature extractor, and the review process around them.”

Replace “data side” with concrete completed work and remaining evaluation.

> “Six families are implemented: random-direction ablation (the control), refusal-direction abliteration with Heretic's kernel parameters sampled instead of optimised, Heretic ARA runs, short LoRA fine-tunes on GSM8K training data or the first 5,000 Alpaca-cleaned examples, task-vector scaling and merges, and chains of two to eight edits drawn from the others.”

A table squeezed into a sentence. Use a family table or separate sentences about controls, trained edits and chains.

> “After that come the predictors (a stateless Transformer, the Mamba-2 model, a JEPA-style encoder, and a frontier-LLM reader), the family-held-out and seed-held-out tables for H1 to H4, a 27B transfer test on Lambda, and the forecast-then-verify loop in P6.”

Another roadmap inventory. State the next step and link ROADMAP.

> “We expect some of them to fail.”

Performative candor without a specific expectation; remove or identify which hypothesis and why.

PROGRESS fragments “Repo, uv venv, tmux job convention, worktrees.” and “Drafted by a cheaper model, humanizer pass, Board B REVISE …, fixed, then APPROVE.” should become sentences stating actions. Keep dated headings. Split the crowded limitations paragraph by model coverage, scoring and edit dependence.

## Non-blocking

- Precommit rewrites README after staging, leaving the committed status stale and working tree dirty. After successful refresh, stage with `git add -- README.md`; account for partially staged README edits and do not stage after updater failure.
- Drop the assets/pipeline.gif line; another branch replaces the hero.
- “Every branch goes through two reviews” is established as a protocol requirement, not a verified universal historical fact. Phrase it as the rule unless branch-by-branch compliance is audited.

No edits to the reviewed branch were retained.
