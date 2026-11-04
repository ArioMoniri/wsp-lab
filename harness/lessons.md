# Lessons (harnessing-level recursion)

| date | phase | lesson | component changed | evidence |
|---|---|---|---|---|
| 2026-11-03 | P2 | merge feat/p2-eval; follow-up: per-item seeding for batch invariance | eval runner | Board A review |
| 2026-11-03 | P1 | reference ledger must be mechanically verified (scripts/verify_refs.py) before any Board B pass; LLM-written excerpts were 8/12 fabricated on first attempt | ref-checker role + verifier gate | reviews/P1/docs-refs-venue-boardB-*.md |
| 2026-11-03 | P1 | demo/synthetic figures must carry a TOY DATA stamp and live outside paper/figures; private host ids are now blocked by secret_scan | demo_figures.py, secret_scan.sh | reviews/P1/feat-plots-design-boardA | 
| 2026-11-03 | P1 | user rule: no demo/mock/placeholder values anywhere in paper/docs/figures; removed demo figures | AGENTS.md, DOD | user instruction |
| 2026-11-03 | P0 | CI: image-specific before_script per job; bookkeeping commits must not trigger pipelines (concurrent=1 runner) | .gitlab-ci.yml | pipeline 3024878601 |
| 2026-11-04 | P1 | paper sections merge only after Board B APPROVE + check_cites + verify_refs exit 0 (done for abstract/intro/related) | paper gate | reviews/P1/paper-intro-related-boardB-rereview |
| 2026-11-04 | P1 | worktrees feat/p1-zoo-build and feat/p2-eval have REAL data/ dirs (symlink replaced by agents); artefacts must be consolidated into canonical data/ after their jobs finish and wt.sh must guard the symlink | wt.sh, INFRA | ls -la .worktrees/*/data |
| 2026-11-04 | P1 | mamba-ssm 2.3.2 + causal-conv1d 1.7 build on Blackwell cc12.0/CUDA 13.2/torch 2.15 by patching -std=c++17→c++20 in both setup.py (scripts/build_mamba.sh on perf/ssm-kernels) | INFRA, build script | ~/agent-logs/wsp-mamba3b-*.log |
| 2026-11-04 | P1 | git add -A after consolidating data/ swept 600 MB of build artefacts into a commit; data/ is now ignored wholesale and pushes to GitHub rejected it | .gitignore | GH001 rejection |
| 2026-11-04 | P1 | heretic_ara at production budget ≈ 8 min/trial on the shared 32 GB GPU → run that family on Lambda H100; local build covers the other five families | configs/zoo_qwen3.5-0.8b_local.yaml | round-3 chain.log |
| 2026-11-04 | P0 | pre-commit hook was bypassed with core.hooksPath=/dev/null during merges, so ruff format drift reached CI; run `ruff format --check .` before every push | precommit | CI lint job 9e4a938 |
| 2026-11-04 | P5 | Lambda A100 hosts ship driver 570 / CUDA 12.8: torch cu132 wheels cannot init CUDA ("driver too old"); job script now picks the wheel index from nvidia-smi (cu128) | scripts/lambda_job_ara.sh | ~/ara_attempt1.log on the instance |
| 2026-11-04 | P5 | Qwen3.5 AutoProcessor needs torchvision (video processor); install torch AND torchvision from the same driver-matched index, never drop torchvision | scripts/lambda_job_ara.sh | ~/ara_attempt3.log on the instance |
| 2026-11-04 | P4 | parent_id semantics differ by family (chain=successor, task_vector=source); H1 target is per-step Δy; family holdout must drop ancestry | docs/DECISIONS.md D5, predictor/data.py | Board A review of feat/p4-predictors |
| 2026-11-04 | P5 | Heretic ARA trial cost is CPU/launch-bound (LBFGS per module); plan budgets per trial, not per GPU | D6 | py-spy dump on Lambda |
