# Docs accuracy check, 2026-11-04

Scope: README.md, docs/ROADMAP.md, docs/RESULTS.md, docs/INFRA.md, docs/MODELS.md, docs/DECISIONS.md, docs/ORCHESTRATOR.md. Method: read-only audit by the test sub-agent. No docs or code were edited. No GPU job was started; pytest ran in agent-tmux. Scripts that write or publish (`wt.sh new`, `session.sh start`, `hf_sync.sh push|pull`, `lambda.sh launch|terminate`, `build_mamba.sh`, `eval_p2.sh`, `gate_p0.sh`) were syntax-checked or read, not executed.

Revisions: the audit started at 7336e1f. `main` moved during the run (ce6b192, 0443220, 9c8259e) from other agents' commits. The in-scope docs changed in that window (DECISIONS D4 text, INFRA Lambda note, ORCHESTRATOR skills table). The new text was checked too. Zoo manifest row counts grew during the run (1111, then 1116, then 1125), so zoo counts are snapshots.

Verdicts: `ok` = matches the artefact or code; `wrong` = contradicted by the artefact; `unverifiable` = no artefact or cannot be run here. Issue numbers refer to GitHub issues in ArioMoniri/wsp-lab labelled board.

Totals: see the Counts section at the end.

## 1. Paths and commands

| file:line | claim | verdict | evidence |
|---|---|---|---|
| README.md:1-2, 7, 21-26 | Asset, doc and layout paths (assets/hero.svg, assets/pipeline.gif, docs/*.md, paper/, site/, src/wsp/, configs/, tests/, board/, reviews/, sessions/, harness/, assets/lottie/) | ok | All exist (checked with test -e). |
| README.md:7 | Animated overview link https://ariomoniri.github.io/wsp-lab/ | ok | HTTP 200. |
| README.md:30 | git clone https://github.com/ArioMoniri/wsp-lab | ok | git ls-remote --heads lists main. |
| README.md:31-32 | uv venv --python 3.12 .venv; uv pip install -e ".[dev]" | ok | uv 0.13.7. uv venv and uv pip install --dry-run into a fresh venv under /tmp exit 0. Warnings about missing extras huggingface_hub[cli] and mbstrdecoder[all]. The existing .venv was not recreated. |
| README.md:32 | pytest -q | ok | 228 passed, 8 deselected, 6 warnings, 29.63 s, exit 0. |
| README.md:33 | python -m wsp.zoo.build --config configs/zoo_qwen3.5-0.8b.yaml | ok | --help runs; config exists; full build not run (heavy). |
| README.md:34 | python -m wsp.eval.run --config configs/eval_cheap.yaml --zoo data/zoo/qwen3.5-0.8b | wrong | No --zoo flag (only --zoo-entry), and --model is required. Literal run: "error: the following arguments are required: --model", exit 2. Issue #15. |
| README.md:35 | python -m wsp.predictor.train --config configs/predictor_ssm.yaml | wrong | No wsp.predictor.train module (src/wsp/predictor has __init__, baselines, scan, splits, ssm_torch); configs/predictor_ssm.yaml absent. Literal run: "No module named wsp.predictor.train", exit 1. Issue #16. |
| README.md:37 | CI on a self-hosted GitLab runner; GitHub Actions not used | ok | .gitlab-ci.yml header; no .github directory. |
| README.md:38 | Nothing under data/ is tracked in git | ok | git ls-files data returns 0 files; .gitignore:43 ignores data/. |
| ROADMAP.md:10 | scripts/gate_p0.sh is the P0 gate | unverifiable | Exists; bash -n passes; not executed (runs pytest -m gpu on a busy GPU). |
| ROADMAP.md:61 | P3 numbers go in data/runs/p3/ | ok | Directory exists (currently smoke logs only). |
| ROADMAP.md:66 | wsp/predictor/ssm_torch.py | ok | src/wsp/predictor/ssm_torch.py exists. |
| ROADMAP.md:74 | docs/MODEL_CANDIDATES.md | ok | Exists. |
| ROADMAP.md:76 | LLM-reader prompt is in src/wsp/agents/reader_prompt.md | wrong | File absent (src/wsp/agents has only __init__.py). Issue #18. |
| RESULTS.md:20 | scripts/eval_p2.sh runs the 4096 battery, then the 32768 GSM8K/MATH-500 run, then builds the table | ok | Script read: run at 4096, report, run configs/eval_faithful.yaml at 32768. The 32768 manifest lists gsm8k and math500 only. Not executed. |
| RESULTS.md:27-28, 46 | configs/eval_subsets and data/runs/p2/base-final-4096/records.jsonl | ok | Both exist. |
| INFRA.md:4 | AWS eu-central-1, g7.2xlarge | ok | Matches ~/DR-RESTORE.md (outside repo). Instance id in INFRA.md is a placeholder. |
| INFRA.md:7 | Nightly DR tarballs and AMIs exist; ~/projects is backed up | unverifiable | Policy is documented in ~/DR-RESTORE.md. AWS was not queried from this host. |
| INFRA.md:11-15 | agent-tmux, scripts/wt.sh, scripts/session.sh, scripts/env.sh, sessions/<date>-<nnn>.md | ok | agent-tmux -l and -t run; wt.sh ls and check run; session.sh usage matches; "start" not run (it writes a session file). |
| INFRA.md:14 | ~/.config/wsp/env is chmod 600 | ok | Mode 600. Contents not read. |
| INFRA.md:21 | lambda.sh subcommands types, launch, ls, ssh, terminate | wrong | No ssh subcommand. launch also takes an ssh key name as third argument. Issue #23. |
| INFRA.md:21 | lambda.sh types lists instance prices | ok | Live read-only call. 1x H100 SXM 4.29, 1x H100 PCIe 3.29, 1x B200 6.99, 1x GH200 2.29, 1x A6000 1.09 USD/h, all match INFRA.md:19. |
| INFRA.md:22-23 | lambda_bootstrap.sh clones the repo, makes the venv, pulls models, rsyncs data/evals and data/features back | wrong | Script clones and creates the venv. It pulls no models (it prints a manual hf_sync pull line), has no rsync back, and says results go to HF. Issue #24. |
| INFRA.md:28 | build_mamba.sh pins causal-conv1d 1.7.0 and mamba-ssm 2.3.2.post1, patches C++17 to C++20, uses --no-build-isolation --no-deps, TORCH_CUDA_ARCH_LIST=12.0, MAX_JOBS=4 | ok | scripts/build_mamba.sh lines 5, 9, 18-19. Installed versions match. Not executed. |
| INFRA.md:28 | Build took about 11 minutes; log at ~/agent-logs/wsp-mamba3b-20261103-235407.log | ok | Log exists and reads "Prepared 1 package in 10m 52s". |
| INFRA.md:28 | Import and CUDA Mamba2 forward both succeeded | unverifiable | The log contains only install lines. |
| INFRA.md:35 | Lambda instances pull from HF, not from this server | ok | Consistent with lambda_bootstrap.sh. Contradicts INFRA.md:22-23 (issue #24). |
| INFRA.md:36 | On-demand A100 hosts run driver 570 (CUDA 12.8) | unverifiable | No A100 host is reachable from here. |
| INFRA.md:36 | torch is installed from the cu128 index automatically | ok | scripts/lambda_job_ara.sh:16 maps driver 12.8 to cu128. |
| MODELS.md:27 | mamba-ssm 2.3.2 (PyPI) | ok | Installed version is 2.3.2.post1 (doc omits .post1; minor). |
| MODELS.md:27 | Mamba-3 is arXiv 2603.15569, ICLR 2026 | unverifiable | Id and title match data/papers/2603.15569.txt. ICLR 2026 is not in the artefact. |
| MODELS.md:27 | Mamba-3 is not on PyPI yet | unverifiable | Not checked. |
| MODELS.md:30 | JEPAWG-style arXiv 2607.07127 and 2610.30847; W2T arXiv 2603.15990 | ok | Titles in data/papers match. 2607.07127 introduces JEPAWG; 2610.30847 is "JEPAWG on Parametric Dynamical Systems"; 2603.15990 is "W2T: LoRA Weights Already Know What They Can Do". |
| MODELS.md:17 | MoE abliteration needs expert-writer edits (arXiv 2609.09793) | unverifiable | Title matches. The paper discusses safety behaviour located in experts; the expert-writer requirement was not located in the text checked. |
| MODELS.md:32 | StatNN (Unterthiner 2020) | unverifiable | No artefact in repo. |
| MODELS.md:41 | bnb_4bit lets the 9B model run in 32 GB | unverifiable | Not measured. Weight arithmetic (about 4.5 GB) is plausible. |
| MODELS.md:7-9 | Qwen3.5-0.8B is prone to thinking loops (model card); gemma-3-270m-it is the sycophancy-pair base | ok | data/papers/takehi2026wsp.txt lines 931 and 819. |
| MODELS.md:10 | LiquidAI/LFM3-1.7B is #1 trending on 3 Nov | unverifiable | Trending snapshot; no artefact. |
| ORCHESTRATOR.md:10 | Ruflo keeps hybrid memory in data/memory | wrong | Path does not exist. Issue #20. |
| ORCHESTRATOR.md:10 | Ruflo v3.66 | unverifiable | No version string found in .claude-flow, .swarm, .mcp.json or CLAUDE.md. |
| ORCHESTRATOR.md:17-24 | Ten role files under .claude/agents/ | ok | All ten exist. |
| ORCHESTRATOR.md:26 | The same roles are mirrored in opencode.jsonc | wrong | infra-ops is absent; reviewer-ml, reviewer-journal and reviewer-arch are one "reviewer" entry; four model defaults differ from the table. Issue #21. |
| ORCHESTRATOR.md:30-35 | delegate_task recipe with role, isolation, title, target.providerInstanceId and model, task | ok | Matches the loaded tool schema: role enum includes implementation and review; isolation enum includes worktree. |
| ORCHESTRATOR.md:39 | Claude API workspace header set in ~/.claude-api/settings.json | unverifiable | A workspace key is present. The 23:46 UTC status is a historical snapshot. |
| ORCHESTRATOR.md:3-9, 38-50 | Provider health and routing policy as of 3 Nov | unverifiable | Historical snapshot; no source artefact. |
| ORCHESTRATOR.md (4 Nov skills table) | Skills listed per role | unverifiable | code-review, security-review, simplify, dataviz and run exist in this session. anthropic-skills:*, engineering:* and design:ux-copy are not in this session's skill list. |
| ORCHESTRATOR.md:46, 50 | Ledger board/usage.md | ok | File exists. Contents not audited. |
| DECISIONS.md:4 | reviews/P2/fix-eval-lora-boardA-2026-11-04.md (F2) | ok | Exists; F2 at lines 61-66. |
| DECISIONS.md:22 | reviews/P2/feat-p2-batch-boardA-2026-11-04.md | ok | Exists. |
| DECISIONS.md:32 | board/lambda_spend.md | ok | Exists. |
| DECISIONS.md:8-9 | Eval loader applies Δ_eff exactly and records the stats in its manifest | ok | src/wsp/eval/batch.py:234-271 records rounding_stats. |
| DECISIONS.md:6-7 | The zoo stores the effective delta together with rounding_stats | wrong | No rounding_stats in data/zoo artefacts. Δ_eff is computed at extraction (src/wsp/features/extract.py:13-16) and at eval. Already open as issue #12; not refiled. |
| DECISIONS.md:13-17 | D2 fast tier: gsm8k 50, math500 25, ifeval 25, mmlu_pro 100, hellaswag 100, refusal 50, kl 50; 1024 tokens; batch 32 | ok | configs/eval_fast.yaml matches. |
| DECISIONS.md:24-25 | D3 quotas random_dir 60, abliteration 100, lora_sft 60, task_vector 30, chain 50 | ok | configs/eval_subset_d3.yaml matches. |
| DECISIONS.md:29-30 | D4 cap of 150 USD and instance rules | ok | board/lambda_spend.md header; commit c3ff803. |
| DECISIONS.md:29 | D4 launch: 1x A100-SXM4 40 GB at 1.99/h | ok | Ledger row starting 2026-11-04T06:13:12Z, gpu_1x_a100_sxm4. |
| DECISIONS.md:29 | D4: "the only type with capacity" | unverifiable | A live list at about 06:19 UTC also showed gpu_1x_a10 with capacity in two regions. |

## 2. Numbers

| file:line | claim | verdict | evidence |
|---|---|---|---|
| RESULTS.md:33 | gsm8k 15.5 (n=200) | ok | data/runs/p2/base-final-4096/summary.json: gsm8k 0.155; counts gsm8k 200. |
| RESULTS.md:34 | math500 6.0 (n=100) | ok | math500 0.06; count 100. |
| RESULTS.md:35 | ifeval 42.0 (n=100) | ok | ifeval 0.42; count 100. |
| RESULTS.md:36 | mmlu_pro 20.3 (n=300) | ok | mmlu_pro 0.20333; count 300. |
| RESULTS.md:37 | hellaswag 51.3 (n=300) | ok | hellaswag 0.51333; count 300. |
| RESULTS.md:38 | Loop rate 90.2 | ok | loop_rate 0.9025. |
| RESULTS.md:39 | Refusal 99.0 (n=100) | ok | refusal_rate 0.99; count 100. |
| RESULTS.md:40 | KL to base 0.000 | ok | kl_to_base 0.0. |
| RESULTS.md:23-24, 33-38 | Blog reference: MATH-500 7.9, GSM8K 13.8, IFEval 21.0, loop 93.0 | ok | data/papers/takehi2026wsp.txt, results table row "Base 7.9 13.8 21.0 93.0" (line 958). |
| RESULTS.md:38 | Blog loop 93.0 is "MATH-500 only" | unverifiable | The paper table column is "Hit token limit (%)" with no benchmark qualifier; the text does not restrict it. |
| RESULTS.md:28 | Blog used 32,768 tokens and several samples per item | ok | Paper lines 928-931. |
| RESULTS.md:42 | 8x smaller token budget | ok | 32768 / 4096 = 8. |
| RESULTS.md:46 | Refusal 99/100 is consistent with Heretic's 97/100 | ok | data/papers/pksharma2026heretic.txt:273: "97 of 100" for Gemma 3 12B, per the project README. |
| RESULTS.md:29 | The 32k pass crashed (CUDA OOM); rerun pending | unverifiable | data/runs/p2/base-final-32768/manifest.json says status running and has no records. OOM mentions in board/P1.md, board/P2.md and sessions/2026-11-03-003.md were not read in detail. |
| RESULTS.md:54; ROADMAP.md:45 | Entry times 0.2997 and 0.3022 min; mean 0.301 | unverifiable | Values appear only in RESULTS and ROADMAP. No run directory under data/ (data/runs/p2-batch holds legacy-smoke only). No review or session quotes them. The mean arithmetic is correct. |
| ROADMAP.md:42-44 | Run 20261104-032002-1c2cc588; entries bccb9ab10b5661e0 and adf416337f5a2277 are production random_dir | ok | Both rows are in data/zoo/qwen3.5-0.8b/manifest.jsonl (random_dir, production). Run id appears in sessions/2026-11-04-p2-batch.md:8 and reviews/P2/feat-p2-batch-boardA-rereview-2026-11-04.md:35. |
| ROADMAP.md:46-48 | About 9-12 min per entry; about 120 GPU-h for 700 entries | ok | reviews/P2/feat-p2-batch-boardA-2026-11-04.md:173-176 (planning figure 120, range 100-150). |
| ROADMAP.md:46-47; DECISIONS.md:24-25 | D3 subset: about 300 entries, about 50 h | ok | Board A rereview R3 found 227 on an earlier manifest. On the current manifest, configs/eval_subset_d3.yaml resolves to 300 (60/100/60/30/50). 300 times 9-12 min is 45-60 GPU-h. Consistent now. |
| README.md:41 | P1 edit zoo has about 590 entries | wrong | 1125 rows at check time: 1087 production, 38 without eligibility; single-edit families 560, chain 557. Issue #17. |
| INFRA.md:6 | 8 vCPU, 30 GiB RAM | ok | nproc 8; 30 GiB total. |
| INFRA.md:6 | 922 GB free disk | wrong | df /: 991G size, 679G available. Issue #22. |
| INFRA.md:5 | GPU RTX PRO 4500 Blackwell Server Edition, 32,623 MiB, CC 12.0, driver 595.112, CUDA 13.2 | ok | nvidia-smi: 32623 MiB, compute_cap 12.0, driver 595.112.07, CUDA 13.2. |
| MODELS.md:14 | Qwen3.8-27B has 8.0M downloads | ok | 8,014,877 on the HF API at check time. |
| MODELS.md table (VRAM column) | bf16 VRAM is 2 bytes per parameter | ok | Arithmetic holds for every row except the ternary 27B row (unverifiable). |
| ROADMAP.md:60 | feature_version 3: 1098 stats + 120 spectrum dims | ok | src/wsp/features/extract.py:10, 47. |
| ROADMAP.md:23 | Builder has six families | ok | FAMILY_ORDER in src/wsp/zoo/build.py:112 lists six. |
| DECISIONS.md:4-9 (D1) | 87 % of non-zero LoRA delta entries are lost to bf16 rounding; about 40 % of the intended logit effect | ok | Review F2: 86.6 % absorbed; mean effect 0.009 against 0.023 intended (39 %). |

## 3. "merged" and "done" claims against git log

| file:line | claim | verdict | evidence |
|---|---|---|---|
| README.md:41 | P0 complete | ok | ci green 7c8f183; roadmap P0 gate 570d8c5; GitLab pipeline success (see next row). |
| ROADMAP.md:8 | GitLab CI fully green (pipeline 3025374435, 4 Nov) | ok | GitLab API: status success, ref main, created 2026-11-04T03:33:46Z. Jobs lint, test-cpu, test-gpu and paper are all success. |
| ROADMAP.md:23 | Builder merged 4 Nov 2026 | ok | 393fc18 merge feat/p1-zoo-build, 2026-11-04. |
| ROADMAP.md:23 | Local full build started (tmux p1-zoo-full) | ok | f7d3d4a "full local build launched"; tmux session p1-zoo-full exists. |
| ROADMAP.md:23 | Heretic ARA family deferred to Lambda H100 | wrong | Ledger shows the heretic_ara job on gpu_1x_a100_sxm4 40 GB (board/lambda_spend.md). Issue #19. |
| ROADMAP.md:60 | Extractor, pooling, splits and baselines merged 4 Nov (bf99575) | ok | bf99575 merge feat/p3-features, 2026-11-04. |
| ROADMAP.md:60 | Full-zoo extraction running (tmux p3-extract) | unverifiable | tmux session p3-extract exists (created 04:02). Its log was 0 bytes at check time, so progress is not confirmed. |
| README.md:41 | P2 eval battery and resumable batch evaluator merged | ok | 2b6904a merge feat/p2-eval (2026-11-03); 9146c5f merge feat/p2-batch (2026-11-04). |
| README.md:41 | Base-model reproduction recorded | ok | 1354891 results commit; base-final-4096 manifest status complete. |
| README.md:41 | P3 features and baselines in progress | ok | Consistent with bf99575 merged and extraction running. |
| RESULTS.md:14 | Implementation is on feat/p2-eval | wrong | feat/p2-eval was merged to main by 2b6904a on 2026-11-03. Issue #25. |
| DECISIONS.md:29-30 | D4: Lambda spend approved with a cap of 150 | ok | Commit c3ff803 "D4 Lambda spend approved (cap $150)". |

## 4. Lint, tests and citation checks

| file:line | claim | verdict | evidence |
|---|---|---|---|
| ruff check . | Lint clean | ok | "All checks passed!" (ruff 0.17.3). |
| ruff format --check . | Formatting clean | ok | "336 files already formatted". |
| pytest -q | Tests green | ok | 228 passed, 8 deselected (gpu, slow, net), 6 warnings, exit 0. |
| scripts/verify_refs.py | Excerpts and arXiv metadata verified | ok | Every entry reports "verified"; exit 0. |
| scripts/check_cites.py | Paper citations verified | ok | "citation check passed: 33 verified keys available; scanned 9 section files"; exit 0. |

## 5. Hugging Face model ids (anonymous HF API, checked 2026-11-04)

HTTP 200 means the repo is public. 401 means it exists but is not readable anonymously (gated or private). 404 would mean missing; none returned.

| file:line | claim | verdict | evidence |
|---|---|---|---|
| MODELS.md:7 | Qwen/Qwen3.5-0.8B | ok | 200, not gated. |
| MODELS.md:8 | Qwen/Qwen3.5-0.8B-Base | ok | 200, not gated. |
| MODELS.md:9 | google/gemma-3-270m-it | ok | 200, gated (manual). |
| MODELS.md:10 | LiquidAI/LFM3-1.7B | ok | 200, not gated. |
| MODELS.md:11 | Qwen/Qwen3.5-4B | ok | 200, not gated. |
| MODELS.md:11 | saidutta69/Qwen3.5-4B-heretic | ok | 200, not gated. |
| MODELS.md:12 | Qwen/Qwen3.5-9B | ok | 200, not gated. |
| MODELS.md:12 | huihui-ai/Huihui-Qwen3.5-9B-abliterated | ok | 200, not gated. |
| MODELS.md:13 | google/gemma-4-12b-it | ok | 307 redirect to google/gemma-4-12B-it (canonical casing differs); not gated. |
| MODELS.md:13 | culturerevolt/gemma-4-12b-heretic-abliterated | ok | 200, not gated. |
| MODELS.md:14 | Qwen/Qwen3.8-27B | ok | 200, not gated. |
| MODELS.md:14 | huihui-ai/Huihui-Qwen3.8-27B-abliterated | ok | 200, not gated. |
| MODELS.md:14 | 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored | ok | 200, not gated. |
| MODELS.md:14 | DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored | ok | 200, gated (manual). |
| MODELS.md:15 | Nanbeige/Nanbeige4.3-33B | ok | 200, not gated. |
| MODELS.md:15 | FRENCHFAKER/Nanbeige4.3-33B-cybersec-abliterated | ok | 401: exists, gated or private. |
| MODELS.md:16 | prism-ml/Ternary-Bonsai-2-27B | ok | 401: exists, gated or private. |
| MODELS.md:17 | Qwen/Qwen3.6-35B-A3B | ok | 200, not gated. |
| MODELS.md:17 | Qwen/Qwen3.8-Flash-Next (doc omits the org) | ok | 200, not gated. Org prefix Qwen/ is needed. |
| MODELS.md:18 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | ok | 200, not gated. |
| MODELS.md:12-14 | Elided names: DavidAU/...9B...Heretic, Vortecks/, orcarouter/ | unverifiable | Names are incomplete in the doc; cannot be checked. |
| INFRA.md:33 | Aralario/wsp-lab-zoo, wsp-lab-evals, wsp-lab-runs (private datasets) | ok | Each returns 401: exists, private. |

## 6. Secrets, private addresses and instance ids

Requested scan: `git grep -nE "secret_wsp|hf_[A-Za-z0-9]{30}|172\.31\.|i-0[0-9a-f]{15}"` returns four tracked lines. Broader search found one more.

| file:line | claim | verdict | evidence |
|---|---|---|---|
| scripts/secret_scan.sh:4 | Scanner host/instance pattern (HOSTPAT) | wrong | Contains a real EC2 instance id and a private hostname. Issue #26. Values not repeated here. |
| scripts/secret_scan.sh:5 | Secret-format regexes | ok | Patterns only. |
| scripts/board.sh:8 | Redaction regex | ok | Pattern only. |
| reviews/P2/feat-p2-batch-boardA-2026-11-04.md:146 | List of token prefixes checked | ok | Names only. |
| sessions/2026-11-03-002.md:4 | Session header "Host" line | wrong | Contains the private hostname. Not matched by the requested regex. Issue #26. |
| git log --all -G hf_[A-Za-z0-9]{30} | Hugging Face token-shaped strings in any commit | ok | 0 commits. |
| git log --all -G instance-id pattern | Instance id in history | wrong | 4 commits: 5688cce, 9690067, 75f5424, b422a3b. Issue #26. |
| git log --all -G private-range pattern | Private address in history | wrong | 2 commits: 9690067, 75f5424. Issue #26. |
| git log --all -G secret_wsp_ | Literal prefix in history | unverifiable | 4 commits (90ba1b2, 5688cce, 9690067, faa1c4a). Values not inspected. |
| data/hf/token | Token file under data/ | ok | Untracked and ignored (.gitignore:43). Not read. |
| docs/INFRA.md:4-5 | Instance id and host shown as placeholders | ok | Placeholders only. |

## 7. Prose against docs/WRITING_GUIDE.md

The guide is written for the paper, so it is applied loosely to the docs. No banned openers or fillers were found in README or docs ("Notably", "Importantly", "Interestingly", "delve", "leverage", "a testament to", "in the realm of"). Em-dash counts (not banned by the guide): ROADMAP 9, RESULTS 9, DECISIONS 7, MODELS 4, ORCHESTRATOR 1, README 0, INFRA 0.

The ten worst sentences, quoted:

1. docs/MODELS.md:20-21: "Public abliterated/heretic variants give *free* far-from-base data points; our own Heretic runs give the *dense* trajectory data." Parallel construction with italic emphasis; the claim is vague.
2. docs/RESULTS.md:42-43: "Reading: GSM8K and MATH-500 are within a few points of the blog at a 8× smaller token budget; the loop rate confirms the blog's central failure mode (≈ 90 % of generations hit the limit)." Three claims in one sentence; "a 8×" should read "an 8×".
3. README.md:41: "P1 edit zoo built locally (≈ 590 entries, Heretic ARA family pending GPU budget); P2 eval battery + resumable batch evaluator merged, base-model reproduction recorded; P3 features and baselines in progress." Four status claims in one run-on; the count is wrong (issue #17).
4. docs/INFRA.md:7: "Nightly DR tarballs + AMIs exist (see `~/DR-RESTORE.md`), so project data under `~/projects` is backed up." The inference from backups to "backed up" is not supported.
5. docs/DECISIONS.md:6-7: "Decision: the subject models are evaluated in bf16 as deployed. The zoo therefore stores the **effective** delta". "Therefore" presents an unimplemented storage step as following from the decision (issue #12).
6. docs/ROADMAP.md:90-92: "Forecast-then-verify: the predictor scores every proposed edit before evaluation; it then gates/ranks candidate edits (weight-space Bayesian optimisation) and acts as critic for `llm_optimizer` runs where the editor LLM replaces Adam/Muon steps." Three roles in one sentence, slash-joined.
7. README.md:13-16: "Backbones compared: a Mamba-2 state-space model over the edit sequence, a JEPA weight encoder (JEPAWG-style), a stateless Transformer over W2T-style canonical SVD tokens, LightGBM on weight statistics, and frontier LLMs reading the same deltas as text." A five-item list written as prose.
8. docs/MODELS.md:27: "The SSM the brief asks for." A fragment; "the brief" is never defined.
9. docs/INFRA.md:35: "They stay private because abliteration deltas are, in effect, uncensoring edits". "In effect" is a hedge that adds nothing.
10. docs/ORCHESTRATOR.md:3: "Two layers, both configured in this repo:" A setup frame that delays the content.

## 8. Issues filed (ArioMoniri/wsp-lab, label board)

- #15 README eval command does not run (no --zoo flag; --model required)
- #16 README training command references a missing module and config
- #17 README reports about 590 zoo entries; manifest has 1125 rows
- #18 ROADMAP cites src/wsp/agents/reader_prompt.md, which does not exist
- #19 ROADMAP says Heretic ARA is deferred to Lambda H100; ledger shows A100 40 GB
- #20 ORCHESTRATOR points hybrid memory at data/memory, which does not exist
- #21 ORCHESTRATOR says roles are mirrored in opencode.jsonc; four roles missing or differ
- #22 INFRA states 922 GB free disk; df shows 679 GB available
- #23 INFRA lambda.sh synopsis lists an ssh subcommand that does not exist
- #24 INFRA says lambda_bootstrap pulls models and rsyncs results back; the script does neither
- #25 RESULTS says P2 implementation is on feat/p2-eval; branch was merged
- #26 security: tracked files contain an instance id and a private hostname

Not filed: DECISIONS.md:6-7 (zoo stores Δ_eff) duplicates open issue #12. Existing open issues #1-14 do not otherwise overlap with these claims.

## Counts

Across sections 1-6 (135 rows): **ok 98, wrong 16, unverifiable 21**.

- The 16 wrong rows cover 12 issues. Issues #15-#25 are one row each (11 rows). Issue #26 covers four rows in section 6. The one remaining wrong row (DECISIONS.md:6-7) duplicates open issue #12 and was not refiled.
- Tests and checks: ruff check passed; ruff format --check passed (336 files); pytest 228 passed, 8 deselected; verify_refs and check_cites both passed.
- Unverifiable items that would settle them: an A100 host driver check, an AWS backup listing, the entry-time measurements (no run directory under data/), and the "ICLR 2026" venue for Mamba-3.
