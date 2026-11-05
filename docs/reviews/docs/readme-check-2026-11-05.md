# README accuracy check, 5 November 2026

Scope: every number, date, status and claim in the narrative sections of `README.md`, checked against the sources it cites or implies. Branch `docs/readme-check`, worktree `.worktrees/docs/readme-check`, started from `main` at `fe43498` (that commit touches only `board/usage.md`).

Result key: **OK** matches the source. **FIXED** changed in this branch. **FLAG** source disagreement or not verifiable from the repo; left as written and listed for the owner.

The status block was regenerated with `scripts/update_readme_status.py` and not hand-edited. Its only changes are the timestamp and the CI line, which moved from `running` to `failed` on a live `glab` query.

## Opening and "What we built"

| # | Claim | Source checked | Result |
|---|---|---|---|
| 1 | Blog: 0.016% of weights edited, MATH-500 7.9% to 48.1%, no gradients | `docs/RELATED_WORK_MAP.md` (takehi2026wsp row); the blog itself was not re-fetched | OK (secondary) |
| 2 | H1 to H6 thresholds and amendments are in `docs/HYPOTHESIS.md` | HYPOTHESIS H1 ρ ≥ 0.6; falsification ρ < 0.3 (line 44) | OK |
| 3 | Manifest lists 1,247 entries | Status block zoo total 1247 | OK |
| 4 | Six families and their descriptions | `configs/zoo_qwen3.5-0.8b.yaml`; status family rows | OK |
| 5 | LoRA data: GSM8K train or first 5,000 Alpaca-cleaned | Zoo config `gsm8k_train`, `alpaca_cleaned_5k` | OK |
| 6 | Chains: two to eight edits | Zoo config chain `length` min 2, max 8 | OK |
| 7 | "Heretic ARA is the slowest family, about eight minutes per trial" | PROGRESS: "about eight minutes per trial on the shared GPU, so the local config excludes that family". No "slowest" claim in source | FIXED (wording changed to the source's statement) |
| 8 | Lambda runs under a $150 cap | DECISIONS D4; PROGRESS | OK |
| 9 | "Board A asked for the 12 flagged ARA entries to be accounted for before use" | `docs/reviews/P1/heretic-ara-rebuild-forensic-2026-11-05.md` (REVISE: retain as diagnostic, flagged `ara_unconstrained_surrogate_scale`, excluded from Heretic-quality comparisons and production panels); DECISIONS D6 addendum 3 (`eligibility: smoke`) | FIXED (wording now matches the disposition; review linked) |
| 10 | Zoo has no production ARA entry | Status block: heretic_ara production 0, smoke 12, unlabelled 1 | OK |
| 11 | Corrected rebuild is running and not qualified | Ledger open row `ad0a0a09…` (gpu_1x_a10, since 2026-11-05T05:41Z); D6 addendum 3 ("corrected ARA only enters after … qualification") | FIXED (sentence added; the old text did not describe the rebuild) |
| 12 | About 87% of non-zero delta entries disappear for a 5-step adapter | DECISIONS D1 (~87%); PROGRESS (86.6%) | OK |
| 13 | Zoo stores the effective delta `bf16(W + Δ) − W` | PREREG §1 (effective bf16 norm); DECISIONS D1 | OK |
| 14 | Fast tier generates 1,024 tokens on GSM8K, MATH-500, IFEval, MMLU-Pro and HellaSwag | `configs/eval_fast.yaml` (`max_new_tokens: 1024`); `src/wsp/eval/run.py` lines 368–379: MMLU-Pro and HellaSwag are scored by `choice_scores`, not generated | FIXED (generation limited to GSM8K, MATH-500, IFEval; MMLU-Pro and HellaSwag described as choice-scored) |
| 15 | Fast-tier subsets of 25 to 100 items | `configs/eval_fast.yaml` (n 25 to 100) | OK |
| 16 | Cheap tier runs 100 to 300 items per benchmark | `configs/eval_cheap.yaml` (n 100 to 300) | OK |
| 17 | Cheap tier at 4,096 tokens | `configs/eval_cheap.yaml` `max_new_tokens: 4096` | FIXED (reworded as a generation cap; MMLU-Pro's `max_new_tokens: 1024` override is ignored because it is choice-scored) |
| 18 | Faithful tier uses 32,768 tokens | `configs/eval_faithful.yaml` | OK |
| 19 | Item lists fixed under `configs/eval_subsets/` | Directory exists | OK |
| 20 | Feature table covers 1,235 entries, finished | Status block: features 1235 (finished) | OK |
| 21 | Per-tensor statistics, canonical SVD tokens, hybrid linear-attention handling | Not checked in code | FLAG (not verified; no number) |
| 22 | Trajectory splits group by ancestry root | PREREG §2; Board A synthesis | OK (docs only; code not read) |
| 23 | Protocol v2 runs on the 300-entry D3 subset; M1/M2 matched variants; StatNN; descriptor-only and delta-norm controls | `docs/reviews/P4/d3v2-summary-2026-11-05.md`; Board A synthesis §A | OK |
| 24 | "Each configuration was run with three seeds" | v2 summary: "3 full seeds … baselines from seed 0"; Board A §A: seed-0 baselines are not three-seed means | FIXED (learned configs three seeds; StatNN and controls single seed-0 fits) |
| 25 | Board A approved the editor harness to merge with production disabled | `docs/reviews/H5/board-a-editor-final-2026-11-04.md` (APPROVE to merge; production disabled) | OK |
| 26 | Council parked paid and editor episodes; H4 parked | SYNTHESIS §3 (H4, H5) | OK |

## "What we found"

| # | Claim | Source checked | Result |
|---|---|---|---|
| 27 | No learned row reaches 0.6 on either target | v2 summary: highest learned mean 0.46 | OK |
| 28 | Highest HellaSwag three-seed mean 0.46, M1-matched, trajectory split | v2 summary (m1_matched-trajectory HellaSwag 0.46) | OK |
| 29 | Highest MMLU-Pro 0.45, shared by M1-matched (trajectory) and M2-matched (seed) | v2 summary (both 0.45) | OK |
| 30 | 25 complete rows | RESULTS line 9 ("25 complete runs") | OK |
| 31 | Falsification is ρ < 0.3 on all benchmarks; several rows exceed 0.3 | HYPOTHESIS line 44; v2 summary | OK |
| 32 | Seed-0 StatNN 0.57 and 0.65 on the trajectory split | v2 summary; SYNTHESIS §1 item 1 | OK |
| 33 | Matched M1 mean Spearman 0.45 and 0.46, trajectory split | v2 summary; abstract.tex | OK |
| 34 | Descriptor-only 0.52 and 0.61, same seed and split | v2 summary; SYNTHESIS §1 item 2 (0.523855, 0.613582) | OK |
| 35 | Gaps 0.042 (MMLU-Pro) and 0.035 (HellaSwag) | SYNTHESIS §1 item 2 (0.042050, 0.035019) | OK |
| 36 | Edit metadata ranks edits nearly as well as StatNN | `paper/sections/abstract.tex` | OK |
| 37 | Family split: StatNN −0.02 (MMLU-Pro), 0.11 (HellaSwag) | v2 summary; SYNTHESIS §1 item 1 | OK |
| 38 | Three-seed mean of every learned row below seed-0 StatNN on both targets, seed and trajectory splits | v2 summary: 7 learned configurations × 2 splits × 2 targets | OK |
| 39 | Residual θ for MMLU-Pro 0.146 [0.069, 0.214] | `round2_residual_estimand_output.txt` (theta 0.1462, CI [0.069, 0.214]) | OK |
| 40 | Residual θ for HellaSwag −0.034 [−0.118, 0.037] | Same output (−0.0342, CI [−0.118, 0.037]) | OK |
| 41 | 345 rows and 278 clusters | Same output (n 345, clusters 278) | OK |
| 42 | Negative θ means StatNN does worse | `discussion.tex` (θ = 1 − A/B) | OK |
| 43 | MMLU-Pro interval excludes zero; HellaSwag includes it | Same output | OK |
| 44 | MMLU-Pro result uses accuracy data already used for P4 | SYNTHESIS §1 item 3; discussion.tex | OK |
| 45 | 94 of 300 (31.3%) move KL by more than 0.01 | SYNTHESIS §1 item 4 (94/300; 94/300 = 31.3%) | OK |
| 46 | Counts by family: 56/60 LoRA, 30/50 chain, 6/30 task, 2/100 abliteration, 0/60 random | SYNTHESIS §1 item 4 (sum 94) | OK (values); citation changed from "§1.4" to "§1, item 4" (FIXED) |
| 47 | None of the 60 random-direction entries passes the threshold | SYNTHESIS §1 item 4 (0 of 60) | OK |
| 48 | Random-direction edits leave HellaSwag unchanged in every row | `metrology_budget_output.txt` per-family table: hellaswag random_dir sd_dy 0.000 | OK (checked at SD level, not row by row) |
| 49 | Random-direction MMLU-Pro SD 0.007 | Same table: mmlu_pro random_dir sd_dy 0.007 | OK |
| 50 | Abliteration SDs 0.013 (MMLU-Pro) and 0.005 (HellaSwag) | Same table gives MMLU-Pro abliteration sd_dy 0.012 (n 100, device-centred). The 0.013 matches `docs/council/2026-11-05/evidence-statnn-importances.csv` (family_distribution, sd 0.01292, n 98), which the paper audit uses. HellaSwag 0.005 matches | FLAG (two sources disagree; left as is; owner to choose one basis) |
| 51 | Each target moves in steps of one item out of 100 | Fast-tier MMLU-Pro and HellaSwag n = 100 (`eval_fast.yaml`) | OK |
| 52 | Binomial SE 0.045 and 0.049 "at the base values" | Recomputed at the fast-tier base values in RESULTS (MMLU-Pro 0.280, HellaSwag 0.390): sqrt(p(1−p)/100) = 0.0449 and 0.0488 | FIXED (names the fast-tier base values) |
| 53 | HellaSwag residual SD 0.0131 below item SE 0.0136 | `metrology_budget_output.txt` line 44 | OK |
| 54 | MMLU-Pro residual SD 0.0384 above item SE 0.0246; item sampling does not account for it | Same output line 42; remaining signal SD 0.0295 (computed) | OK |
| 55 | Base and ten controls matched exactly on re-evaluation on the same device | RESULTS, same-host test-retest (11 pairs, mean and max difference 0.000) | OK |
| 56 | H100 cross-host: MMLU-Pro 0.008 and HellaSwag 0.003 mean absolute difference on 11 edited entries | RESULTS, cross-host control panel | OK |
| 57 | Seed-1 SD 0.0000 for the two likelihood targets; 0.0124 to 0.0504 for the five sampled targets | `metrology_budget_output.txt` lines 19–27 (gsm8k 0.0504, math500 0.0268, ifeval 0.0248, refusal 0.0124, over-refusal 0.0356) | OK |
| 58 | HellaSwag accuracy retired at the second-round reliability gate; signal variance negative | `round2-metrology.md` §1.4 (HellaSwag accuracy fails and is retired); output table "signal variance < 0" | OK |
| 59 | MMLU log-prob ceiling 0.979 after family and device centring, a heuristic | SYNTHESIS §1 item 5; metrology `ceil_pop` 0.9790 | OK (citation changed from "§1.5" to "§1, item 5", FIXED) |

## "The base model"

| # | Claim | Source checked | Result |
|---|---|---|---|
| 60 | `p2/base-final-4096`: GSM8K 15.5, MATH-500 6.0, IFEval 42.0, MMLU-Pro 20.3, HellaSwag 51.3 | RESULTS P2 table; status block generated from `summary.json` | OK |
| 61 | Refused 99 of 100 harmful prompts | RESULTS refusal 99.0 | OK |
| 62 | 90.2% of generations on the three generative benchmarks hit the limit | RESULTS loop rate (gsm8k + math500 + ifeval) 90.2 | OK |
| 63 | Blog 13.8% GSM8K, 7.9% MATH-500, 21.0% IFEval | RESULTS table | OK |
| 64 | 4,096 tokens is one eighth of 32,768 | Arithmetic | OK |
| 65 | IFEval about twice the blog's 21.0% | 42.0 / 21.0 = 2.0 | OK |
| 66 | 32k pass crashed with CUDA OOM under GPU contention | RESULTS P2 section | OK |
| 67 | Mamba-2 kernels 2.65 ms vs 18.67 ms for a plain PyTorch loop | `docs/BENCH_SSM.md` (cuda mamba_ssm fwd+bwd 2.65; cuda loop fwd+bwd 18.67) | OK (backend naming not checked in code) |
| 68 | Timing taken while another job held the GPU | BENCH_SSM header | OK |
| 69 | `scripts/build_mamba.sh` is needed | File exists | OK |

## "What comes next"

| # | Claim | Source checked | Result |
|---|---|---|---|
| 70 | Council verdict quoted | SYNTHESIS §verdict, verbatim | OK |
| 71 | TF-v1: 1,000 paired items against the actual predecessor; sealed fresh source-closed components | PREREG §1, §2 | OK |
| 72 | Pass: θ ≥ 0.15; lower bound > 0; half-width ≤ θ/2; ≥ 100 fresh components; ≥ 80% power at 0.15 | PREREG §4, §5 | OK |
| 73 | Kill: upper 95% bound below 0.05 on a qualified, powered panel | PREREG §4 requires both component and joint intervals | FIXED (added "on both intervals") |
| 74 | "The council did not approve new Lambda spend or an additional history run" | PREREG status line: "User decisions adopted: … additional history no, new Lambda no"; SYNTHESIS §Board A revision | FIXED (attributed to the adopted user decisions) |
| 75 | Budget 30 local GPU-hours | PREREG status line; SYNTHESIS | OK |
| 76 | Status: "frozen prospective design, pending Board A acceptance and measured execution gates" | PREREG status line (verbatim) | OK (registration wording) |
| 77 | Reports no qualification, timing or power result | PREREG status line | OK |
| 78 | Board A's 5 November review said REVISE and listed four blocking items | `board-a-synthesis-and-v2-2026-11-05.md` line 55 lists five: baseline aggregation; frozen estimand and TF power; measured qualification; provenance, ARA and ancestry closure; budget and placebo equivalence | FIXED (four → five, budget and placebo added) |
| 79 | Board A has not accepted the design; no gate measured | PREREG status; SYNTHESIS (claims no measured gate closure) | OK |
| 80 | 32k rerun still pending | RESULTS P2 table ("32k rerun pending") | OK |
| 81 | "The second base-model reproduction that the P2 gate requires ([RESULTS])" | RESULTS does not mention a second reproduction. PROGRESS line 117 lists "the second reproduction run" as open. No P2 gate requirement found in RESULTS, ROADMAP or DOD | FIXED (cites PROGRESS; the gate requirement is removed) |

## Review record

| # | Claim | Source checked | Result |
|---|---|---|---|
| 82 | Board A, v2 and synthesis, 5 Nov: usable with qualifications; hypotheses unestablished; REVISE | `board-a-synthesis-and-v2-2026-11-05.md` | OK |
| 83 | Board B, v2 final check, 5 Nov: APPROVE with two wording items | `board-b-v2-results-final-2026-11-05.md` | OK |
| 84 | Board B, reliability re-review, 5 Nov: APPROVE for the scoped revision | `board-b-reliability-rereview-2026-11-05.md` | OK |
| 85 | Board A, D3 suite, 4 Nov: REVISE; H1 undetermined and not demonstrated | `board-a-d3-suite-2026-11-04.md` | OK |
| 86 | Board A, H5 harness, 4 Nov: APPROVE with production disabled; first pilot not yet permitted | `board-a-editor-final-2026-11-04.md` | OK |

The list is converted to a table (FIXED, style: the writing guide asks for tables, not bullets, in main text).

## Running it

| # | Claim | Source checked | Result |
|---|---|---|---|
| 87 | Clone URL `github.com/ArioMoniri/wsp-lab` | `git remote -v`: `origin` is that URL | OK |
| 88 | Commands and flags (`wsp.zoo.build --config`; `wsp.eval.run --config --model --out`; `wsp.eval.batch --zoo --subset d3`; `wsp.features.extract --zoo --out --workers`; `update_readme_status.py`) | Flags present in the source's argparse definitions. Not executed | OK (not executed) |
| 89 | Pre-push runs `secret_scan.sh --history`; pre-commit runs ruff and the CPU tests | PROGRESS CI note; `scripts/precommit.sh` (ruff check, format check, tests) | OK |
| 90 | Weights stay out of git; private HF datasets via `scripts/hf_sync.sh` | `scripts/hf_sync.sh` exists; HF dataset privacy not checked | OK (partial) |
| 91 | CI on a self-hosted GitLab runner with GPU passthrough; `.gitlab-ci.yml` | `.gitlab-ci.yml` exists; PROGRESS "self-hosted GPU runner" | OK |

## "What is not done" and "Limitations"

| # | Claim | Source checked | Result |
|---|---|---|---|
| 92 | Fast-tier scores exist for the D3 subset, and "the rest of the zoo is not scored" | The status block counts 389 scored entries: 300 D3 plus 89 chain predecessors (PROGRESS line 145; metrology "entries 389, D3 300"). "The rest is not scored" contradicted the 89 | FIXED (389 named as 300 + 89) |
| 93 | 858 manifest rows without a fast-tier result | 1,247 − 389 = 858 | OK (added) |
| 94 | Clean fast-tier run: 745 s (12.4 min) on an idle local GPU | RESULTS D2 timing section | OK |
| 95 | About 62 h for the 300-entry D3 subset | 300 × 745 s = 62.1 h | OK |
| 96 | "About 255 hours for the full zoo" | 1,247 × 745 s = 258.1 h. 255 h corresponds to 1,235 entries. RESULTS line 63 carried the same figure | FIXED (README and RESULTS now say 258 h for 1,247 rows) |
| 97 | "Timing is measured for one entry only" | Contradicted by RESULTS engine benchmark (HF 6.18 and vLLM 6.10 min on the same entry) and SYNTHESIS measured fast-tier means (8.63, 9.92, 23.66 min) | FIXED (now says the projections rest on one timed entry) |
| 98 | 27B transfer not started | RESULTS P5 "pending" | OK |
| 99 | Refusal rate is a keyword match on 100 prompts; base refuses 99 | `eval_cheap.yaml` (100 prompts); `eval_fast.yaml` (50 prompts) | FIXED (100 in cheap tier, 50 in fast tier) |
| 100 | Fast tier 25 to 100 items; cheap tier 100 to 300 | `eval_fast.yaml`, `eval_cheap.yaml` | OK |
| 101 | 260 matched D3 rows with inferred runtime metadata | Board A D3 suite §1 (260 usable; all rely on inferred runtime) | OK |
| 102 | Baselines are single seed-0 fits; paired intervals condition on those fits | v2 summary; Board A §A | OK |
| 103 | Abliteration parameters sampled from a hand-set range, not optimised | Zoo config abliteration min/max ranges; "hand-set" not checked in code | OK (partial) |
| 104 | Our kernel skips row normalisation that Heretic's default applies | `src/wsp/edits/kernels.py` lines 5–6 | OK |
| 105 | LoRA fine-tunes are short; bf16 rounding removes part of the change | DECISIONS D1 (5-step adapter, ~87%) | OK |
| 106 | Chains reuse parents from the same zoo | PREREG §2 (predecessor and chain edges) | OK |
| 107 | Every Board review was run by language models; no outside reviewer | Review and `board/` records are agent-authored. Not verifiable exhaustively from the repo | FLAG (not verified) |

## Layout, status block, sensitive content, style

| # | Claim | Source checked | Result |
|---|---|---|---|
| 108 | Layout table paths exist (src subpackages, configs, docs, docs/council, docs/reviews, board, sessions, harness, paper) | Directory listing | OK |
| 109 | Status block regenerated from artefacts | Ran `update_readme_status.py`. Timestamp 14:30:39Z; CI line `running` → `failed` (live `glab`). Other rows unchanged | OK |
| 110 | Open Lambda row: `ad0a0a09…`, gpu_1x_a10, us-east-1, since 2026-11-05T05:41Z | `board/lambda_spend.md` | OK |
| 111 | Closed Lambda spend $95.71 (SYNTHESIS; not in README) | Sum of the ledger's `$` column: 95.71 | OK |
| 112 | Codex statements match the ledger | README has no Codex statement (Codex appears only in PROGRESS and `board/usage.md`) | N/A |
| 113 | No hostnames, IPs or secrets | Sweep: no IP addresses or private hosts. Public hosts only (github.com, gitlab.com, rikiyat.github.io). The generated Lambda row includes an instance ID; it is an identifier, not a secret | OK; FLAG the instance ID for owner awareness (generator output, not hand-edited) |
| 114 | Prose per WRITING_GUIDE | Sweep for "Notably", "Importantly", "Interestingly", "delve", "leverage", "robust", "testament", "realm": none. Main-text bullet list converted to a table | FIXED (table) |

## Changes made on this branch

1. `README.md`: ARA paragraph (items 7, 9, 11); fast and cheap tier descriptions (14, 17); seed wording (24); citation labels (46, 59); base-value wording (52); kill criterion (73); user decisions (74); Board A five items (78); second reproduction citation (81); review record as a table (82–86); "not done" count and timing (92, 93, 96, 97); refusal prompts (99).
2. `docs/RESULTS.md` line 63: full-zoo projection 255 h → 258 h for 1,247 rows (item 96).
3. Status block regenerated (timestamp, CI state).

## Not verified or flagged for the owner

- Item 50: the abliteration MMLU-Pro SD is 0.013 in the paper audit's evidence CSV (n 98) and 0.012 in the metrology per-family budget (n 100). Pick one basis and say which.
- Item 107: whether every Board review was agent-written is not provable from the repo.
- Item 113: the Lambda instance ID in the generated status block is public. Decide whether the generator should drop it.
- Item 21: feature pipeline details were not checked in code.
- Item 1: the blog post itself was not re-fetched; figures come from `RELATED_WORK_MAP.md`.
- The README command lines were not executed.
