# Lessons (harnessing-level recursion)

| date | phase | lesson | component changed | evidence |
|---|---|---|---|---|
| 2026-11-03 | P2 | merge feat/p2-eval; follow-up: per-item seeding for batch invariance | eval runner | Board A review |
| 2026-11-03 | P1 | reference ledger must be mechanically verified (scripts/verify_refs.py) before any Board B pass; LLM-written excerpts were 8/12 fabricated on first attempt | ref-checker role + verifier gate | reviews/P1/docs-refs-venue-boardB-*.md |
| 2026-11-03 | P1 | demo/synthetic figures must carry a TOY DATA stamp and live outside paper/figures; private host ids are now blocked by secret_scan | demo_figures.py, secret_scan.sh | reviews/P1/feat-plots-design-boardA | 
| 2026-11-03 | P1 | user rule: no demo/mock/placeholder values anywhere in paper/docs/figures; removed demo figures | AGENTS.md, DOD | user instruction |
| 2026-11-03 | P0 | CI: image-specific before_script per job; bookkeeping commits must not trigger pipelines (concurrent=1 runner) | .gitlab-ci.yml | pipeline 3024878601 |
