# Board B requirements: LLM weight-editor reproduction (H5)

- Date: 4 Nov 2026
- Reviewer: Board B (journal-reviewer role), read-only. No evaluation, GPU job or edit to `paper/` was run.
- Scope: what a TMLR reader needs in a future "Reproduction of an LLM weight editor" section; how the measurement-noise finding must be written; reference requirements; anonymity and attribution. No such section exists yet, so this file states requirements and marks where current text is wrong.
- Location: `docs/reviews/H5/` as requested. `docs/REVIEW_PROTOCOL.md` §Board records names `reviews/<phase>/`; move or copy this file when the H5 phase is opened.
- Read: `AGENTS.md`, `.claude/agents/reviewer-journal.md`, `docs/REVIEW_PROTOCOL.md`, `docs/WRITING_GUIDE.md`, `paper/CHECKLIST.md`, `paper/main.tex`, `paper/sections/experiments.tex`, `paper/sections/method.tex`, `docs/HYPOTHESIS.md` (H1, H5, H6), `docs/RESEARCH_NOTES.md`, `docs/RESULTS.md`, `docs/MODEL_CANDIDATES.md` (LLM-in-the-loop section and Meta^n note), `docs/FIGURE_STYLE.md`, `docs/H5_DESIGN.md`, `paper/refs/verified.yaml`, the stored blog capture `data/papers/takehi2026wsp.txt`, and the shard listings in `data/runs/d3_shards/`.
- Checks run today: arXiv titles via `curl -s https://arxiv.org/abs/<id> | grep -o '<title>[^<]*'`; tracked-file scans for anonymity patterns (counts and file names only, no values reproduced here); binomial standard errors recomputed from numbers in `docs/RESULTS.md`.

## Verdict

**REVISE (blocking list).**

Scores for the current state of the H5 text and plan. These are not scores for a finished result.

| Score | Value | Reason |
|---|---:|---|
| claims-supported | 2 | `experiments.tex` line 4 treats fast-tier scores as predictor targets. The noise data do not support that for GSM8K, MATH-500 or IFEval. |
| novelty | 3 | A forecast-gated LLM editor with equal-evaluation controls extends the blog. The blog is two days old and has no matched controls. |
| clarity | 3 | `docs/` protocol text is precise. The paper text omits hardware, sample counts and the host of each entry. |
| reproducibility | 2 | No frozen sealed split, no preregistration file, test-retest incomplete. |

Approval needs at least 4 on claims-supported and on reproducibility. Neither is met.

Blocking items:

- **B1 Anonymity.** `paper/main.tex` line 11 carries the author's name in the source, outside the arXiv guard. `README.md` and `site/index.html` link the repository under the author's account. Git history has one author name across 366 commits. No anonymised mirror exists. (Section 4.)
- **B2 No preregistration.** `docs/H5_DESIGN.md` Definition-of-done item 4 requires the sealed split, stopping rule, selection rule and Gemma decision rule to be frozen before the first episode. The requirements are written. No frozen split hash or preregistration file exists in `docs/`.
- **B3 Test-retest incomplete.** The `p2-retest` tmux session was still running at 21:39 UTC on 4 Nov (started 20:29). The H100 controls are not in `docs/RESULTS.md`. The reliability paragraph cannot be written with numbers yet.
- **B4 Invalidated sentence.** `paper/sections/experiments.tex` line 4, "Fast scores are the proposed predictor targets, and cheap scores will calibrate a subset." (Section 2.5.)
- **B5 Errors in `docs/RESULTS.md`.** The stated binomial SE "0.05–0.08" is too narrow; the correct planning range is 0.03–0.10, with IFEval at about 0.10. The statement that MMLU-Pro, HellaSwag and the refusal/KL probes are "usable targets at this tier" is not supported (Section 2.1). The "up to 0.16" gap is on four shared edited entries; the largest base-only gap is 0.12.
- **B6 Required controls absent.** Section 1.5.
- **B7 Citations not verified.** The Experiment 8 table and the Gemma results have no excerpt in `paper/refs/verified.yaml`. None of the arXiv candidates in Section 3 is in that file. The Meta^n best-of-N claim is unverified.
- **B8 Licence and consent.** No licence or consent statement exists for the editor providers, base models, blog-derived numbers or benchmarks (Section 1.6).

## 1. "Reproduction of an LLM weight editor": what a TMLR reader needs

### 1.1 Per-episode reporting table

One row per episode. Every field below is required. Failed and unselected proposals go in the table too, so that no survivor-only summary is possible.

| Field | Content | Source |
|---|---|---|
| Episode | `episode_id`, arm (primary editor, cheap editor, no critic, critic, random-direction, best-of-N, Gemma), goal id, seed | `data/runs/llm_edit/<episode_id>/` |
| Editor | Provider, pinned model id, version or snapshot date, access route with no account identifiers, prompt-template hash | `docs/H5_DESIGN.md` §Editor arms |
| Budget | Pre-registered diagnostic forwards (10, 25 or 50) and the charged count from the meter | `docs/H5_DESIGN.md` §Meter semantics |
| Sampling | Editor temperature, top-p, top-k, presence penalty; proposal seed | `docs/MODEL_CANDIDATES.md` controls |
| Attempts | Proposals attempted, valid, rejected by validator, timed out, failed evaluation, selected | `docs/H5_DESIGN.md` §Delta capture |
| Sealed before and after | Per benchmark: n items, samples per item, score, 95% CI | Section 1.2 |
| Sealed difference | Paired difference with CI, bootstrap over items. Loop rate reported under both definitions (Section 1.3) | — |
| Real evaluations | Fast-tier, cheap-tier, faithful 32k, and full-P2-equivalent counts | `docs/MODEL_CANDIDATES.md` §Smallest decisive run |
| Wall clock | Editor latency, sandbox time, diagnostic-forward GPU-seconds, evaluation GPU-seconds, load time; host and GPU model for each step | `docs/H5_DESIGN.md` §Cost |
| Tokens and cost | Editor input and output tokens per proposal and per episode; price-table date; cost in USD | `docs/MODEL_CANDIDATES.md` (log prompt and completion tokens) |
| Artefacts | `manifest.json` with seed, config hash, git SHA, resolved base revision, editor version; trajectory file hash | `AGENTS.md` |

### 1.2 Aggregate table and sample-size requirement

Report one row per arm and budget: episode count (goals × seeds), median sealed difference with CI, real evaluations needed to reach the target (median with CI), count of episodes that never qualified, and median tokens and GPU-minutes.

Use the decisive-run design already in `docs/MODEL_CANDIDATES.md`: at least 30 episodes per arm (at least 10 goals × 3 seeds), extended to 100 if the interval on the evaluation ratio overlaps 0.70.

Sample size must be stated. The equal-final-score criterion is non-inferiority within 1 percentage point. For a score near 0.4, the standard error of a between-arm difference must be well below 0.5 points. That needs roughly 10^4 independent item-samples per arm, since SE = sqrt(0.24/n) ≤ 0.005 gives n ≈ 9,600. The blog averages 4 samples on MATH-500; over 500 problems that is 2,000 item-samples, well short of this. Report the sealed-set size and samples per item, and show they reach the requirement, or state that the non-inferiority claim is not tested.

### 1.3 Blog row and definitional differences

The blog's Experiment 8 numbers, from the stored capture:

| Setting | MATH-500 | GSM8K | IFEval | MATH-500 answers at the 32,768-token limit |
|---|---:|---:|---:|---:|
| Base | 7.9 | 13.8 | 21.0 | 93.0 |
| Edited, 10 forward passes | 12.3 | 25.7 | 49.0 | 87.2 |
| Edited, 25 forward passes | 14.1 | 25.1 | 39.9 | 86.4 |
| Edited, 50 forward passes | 48.1 | 67.1 | 62.3 | 33.1 |

Requirements that follow:

1. **"Loop rate 93 → 33%" is a MATH-500 token-limit share at 32,768 tokens.** It is not the pooled loop rate defined in `paper/sections/experiments.tex` line 2 (GSM8K, MATH-500 and IFEval pooled, capped at 4,096 or, at the fast tier, 1,024). Report the blog's definition and ours side by side. Do not put the two in one column.
2. **Only the 50-pass budget changed the loop rate.** A reproduction that reports only the 50-pass arm reproduces only the favourable row. All three budgets are required, and the 10 and 25 rows must appear.
3. **The blog's IFEval column is not monotone** (49.0, 39.9, 62.3). The blog averages 2 samples per IFEval item and does not state the item count. Its small-sample noise looks like what Section 2 predicts. Our tables must give n for every cell.
4. **Sampling differs.** The blog averages 16, 4, 2 and 1 samples per item for different benchmarks. The repository does not state the number of samples per item at the fast tier. State it, and match or justify the difference.
5. **Token cap differs.** The blog uses 32,768 tokens; our fast tier caps at 1,024. A response that loops at 1,024 may finish at 32,768. Any loop-rate comparison must name the cap.
6. **Editor practice split.** The blog's editor saw 7,469 GSM8K and 7,275 MATH practice problems, not test sets. Our editor must see development items only, and the split must be named.
7. **Gemma sampling differs from Qwen.** The blog samples Gemma at temperature 1.0, top-k 64, top-p 0.95 and up to 8,192 tokens. If the Gemma control uses the Qwen settings, it is not the blog's negative control; say so, and report both if both are run.

### 1.4 Figures

Two figures are required. Both use real run artefacts with run ids in the caption (`AGENTS.md`).

**Figure A: at-a-glance (`docs/FIGURE_STYLE.md` item 1).**
- Left: schematic of edit → effective delta (D1 merge rule) → predictor as critic → gate → real evaluation only when uncertain.
- Right: predicted against measured Δscore on the held-out edit family, with the held-out family marked ★ and the one point to notice circled with a dashed red ellipse.
- Chips: "predicts before evaluating", "held-out family".
- Condition: the predictor must be trained without `llm_edit` for the star to mean anything. Otherwise the held-out marker is misleading.

**Figure B: build-step chain (`docs/FIGURE_STYLE.md` item 4).** One real episode.
- Vertical state boxes: checkpoint id and development score, one per accepted state.
- Colour: green or red by sign of the change only when the change exceeds the measured floor from test-retest. Otherwise grey. This is a requirement arising from Section 2, not part of the house style.
- Right-hand boxes: the edit record (proposal id, tensors touched, effective-delta norm, sparsity, forwards used), the predictor forecast with its interval, and the realised outcome.
- Rejected and failed proposals drawn as dashed boxes, so that the figure shows selection.
- The final sealed score goes in a separate box at the bottom, not in the chain. Sealed scores never appear inside the loop.
- Caption: n, seeds, run id, editor id, budget, and the source of the measurement floor.

Any aggregate plot uses bootstrap 95% bands over episodes (goals), not over items. Add the measurement floor as a shaded band. `docs/FIGURE_STYLE.md` item 4 asks for one real trajectory; for H5, use one real episode.

### 1.5 Required negative results

Each row is required for the H5 claim. Report results whether or not they are null.

| Control | Why it is required | What to report |
|---|---|---|
| Gemma-3-270M-it, same harness and pre-registered budgets | Blog negative: base MATH-500 6.8, edited 7.0–7.7 at 5, 10, 25 and 50 passes. `docs/H5_DESIGN.md` names it as the negative control. | Episodes run; count of episodes with a qualifying proposal (a zero count is a result); base and edited sealed scores with CI. Use Gemma's own tensor map. State the sampling settings (Section 1.3, item 7). |
| Cheap editor (Claude Haiku 5.5 in `docs/H5_DESIGN.md`), with and without critic | Tests whether any effect depends on the premium editor | Same table as Section 1.1. Report if there is no effect. |
| Random-direction control, matched per episode | Separates editor-specific edits from generic perturbation | Matched layer support, sparsity, effective-delta norm and diagnostic budget, with a check table showing the match. Paired comparison. |
| Single-draw best-of-N at equal real-evaluation budget, random-direction and cheap-editor draws labelled separately | The Meta^n note: archive loops act as best-of-N order statistics. Without this, a critic can look better because it keeps an archive. | Real evaluations equal to the critic arm; final sealed score. |
| Same editor without critic | H5 comparison (a) | As above. |
| Adam and Muon with equal diagnostic or forward-pass budget | H5 comparison (b) in `docs/HYPOTHESIS.md` | As above. |
| Predictor ranking replaced by random ranking | Critic-arm control in `docs/MODEL_CANDIDATES.md` | As above. |

### 1.6 Licence and consent statement

A paragraph of this form goes in the appendix, and the same facts go in the manifest. Items marked "to verify" were not checked in this review.

- **Editor providers.** Name the provider, the pinned model id and the date the terms were read. Confirm the terms permit publishing outputs and derived results. Confirm whether tensor data or summaries are sent to the provider, and whether the base-model licence and the provider terms allow that. If a premium editor is not available under our account, label the arm unavailable. Do not substitute (`docs/H5_DESIGN.md`). *To verify.*
- **Blog-derived numbers.** Quoted with attribution from the blog. The blog's logs repository and the third-party rebuild it mentions have licences that are not checked. *To verify.*
- **Base models.** Qwen3.5-0.8B and Gemma-3-270M-it. Gemma is gated, with licence accepted on 4 Nov per `docs/ROADMAP.md`. State which clauses bear on derivative weights and on publication. *To verify.*
- **Benchmarks.** GSM8K, MATH-500, IFEval, MMLU-Pro, HellaSwag and WikiText. *To verify.*
- **Data egress.** Sealed items never leave the evaluator. Development items sent to a provider follow the dataset terms. State which items went to which provider.
- **Consent.** No human participants and no personal data. Say so. Do not imply endorsement of any provider.
- **Spend.** Report provider spend per arm, with the date of the price table used.

## 2. How the measurement-noise finding must be written

### 2.1 What the numbers show

Recomputed from `docs/RESULTS.md`. Binomial values are planning approximations that assume independent items. They are not the final statistic. The final statistic is a per-item paired analysis from the records (`data/runs/p2/base-final-4096/records.jsonl` for the local base run; cross-host records still to be checked).

| Benchmark | n | Base score per host | Binomial SE per run | Base cross-host gap | SE of a gap if independent |
|---|---:|---|---:|---:|---:|
| GSM8K | 50 | 0.16 local, 0.06 A10 | 0.03–0.05 | 0.10 | 0.05–0.07 |
| MATH-500 | 25 | 0.16 local, 0.28 A10 | 0.07–0.09 | 0.12 | 0.10–0.13 |
| IFEval | 25 | 0.40 local, 0.44 A10 | 0.10 | 0.04 | 0.14 |
| MMLU-Pro | 100 | 0.28 both | 0.045 | 0.00 | 0.06 |
| HellaSwag | 100 | 0.39 both | 0.049 | 0.00 | 0.07 |
| WikiText perplexity | — | 22.90 local, 22.91 A10 | deterministic | 0.01 | — |

What this means:

- **The 0.16 figure is for four shared edited entries** (MATH-500). The base-only gaps are 0.10, 0.12 and 0.04. "Up to 0.16 per benchmark on 25–50-item sets" must say "on four shared edited entries".
- **The base cross-host gaps are within about 1.5 to 2 standard errors of a gap.** With one base pair per host and four shared edited entries, host variance and sampling noise cannot be separated. Do not state a host effect as established.
- **Seed matching does not give reproducible generation across GPUs at temperature 0.6.** Small numerical differences change the sampled trajectories after the first divergence. The determinism check covered eight items, not the fast tier.
- **The RESULTS.md statement "binomial SE of 0.05–0.08" is wrong.** Correct range: 0.03–0.10, with IFEval at about 0.10 (p ≈ 0.4 and n = 25).
- **Family means can be confounded by host.** Per-entry noise averages out over many entries. Host effects do not: a host-level bias shifts every entry on that host. The D3 family table in `docs/RESULTS.md` does not report host per entry, and the shard lists show entries moved between hosts (below). A family difference of 0.06–0.09 on GSM8K or MATH-500 cannot be read as an edit effect until families are host-balanced or compared against a same-host base.

### 2.2 What is missing before any claim

- **Same-host test-retest** (`p2-retest`, base plus 10 controls). Not in `docs/RESULTS.md`. Required per benchmark: mean absolute change between repeats, SD, number of discordant paired items, number of paired runs, host and GPU model.
- **H100 controls** (10 controls). Not in `docs/RESULTS.md`.
- **Host map per entry.** `data/runs/d3_shards/` contains a rebalance script that moved about 55% of the A10's remaining entries to the local GPU, and three lists of entries marked lost (35 on the A10, 20 on the H100, 15 on local). The meaning of these lists is not documented. The readout must state, for each entry, which host produced the kept record and whether it was re-run.
- **Hardware sentence.** `paper/sections/experiments.tex` names the local RTX PRO 4500 and one Lambda A100 for the ARA family. The cross-host controls and the D3 shards also use Lambda A10 and an H100 (RESULTS.md, `data/runs/d3_shards/`). The paper text omits both.

### 2.3 Claims inadmissible at the fast tier

| Claim | Status | Reason |
|---|---|---|
| Any single-edit gain or loss on GSM8K, MATH-500 or IFEval | Inadmissible | Per-run SE 0.03–0.10; base cross-host gaps up to 0.12 (paired SE up to 0.14). |
| Ranking edits on those three benchmarks | Inadmissible until reliability is reported | Ranking inherits the same noise. |
| Spearman ρ against observed Δy on those targets | Admissible only with the reliability ceiling reported | To first order, ρ against an observed change cannot exceed about √r, where r is the reliability of that change. The H1 bar ρ ≥ 0.6 requires r ≥ 0.36. State the measured r per benchmark. Do not change the pre-registered threshold. |
| Family-mean differences on those three benchmarks | Inadmissible until host-balanced or compared with a same-host base, using a paired analysis | Host effects shift whole families (Section 2.1). |
| Refusal-rate differences when the base is at 0.99–1.00 | Inadmissible | At n = 50, one item is 0.02. The metric sits at the ceiling. |
| Loop-rate comparison with the blog | Inadmissible | Different token cap (1,024 vs 32,768), different pooling, different sample counts. |
| GSM8K or MATH-500 "within a few points of the blog" at 4k | Inadmissible as stated | Base cross-host gap on GSM8K is 0.10 at fast tier; the 4k table has no CI. |
| Family-mean differences on MMLU-Pro or HellaSwag | Admissible with host-balanced family means, a same-host base and CIs | Single-entry SE is 0.045–0.05. |
| WikiText perplexity and KL | Admissible with the numerical floor stated | Deterministic; cross-host perplexity agrees to 0.01. |

Family-mean comparisons are a different claim from single-entry comparisons. A family mean over many entries can resolve differences that a single entry cannot, provided host and item structure are handled. The rows above say which kind of claim is admissible.

### 2.4 Reliability paragraph: template

Bracketed values are to be filled from run artefacts. Do not write numbers before they exist. The paragraph is written to be precise, not to disclaim everything.

> Fast-tier scores are noisy, and we report their reliability before using them. [Same-host test-retest, run `[retest id]`: the base model and [K] controls were each evaluated twice on [host, GPU] with the same seed and batch size. The mean absolute change between repeats was [a] points on GSM8K, [b] on MATH-500, [c] on IFEval, [d] on MMLU-Pro and [e] on HellaSwag. [f] of [N] paired items changed outcome on [benchmark].] [Cross-host, runs `[ids]`: on the base model, the same seed gave GSM8K [0.16 / 0.06] (n = 50), MATH-500 [0.16 / 0.28] (n = 25), IFEval [0.40 / 0.44] (n = 25), and identical MMLU-Pro and HellaSwag scores (n = 100 each). On four shared edited entries the largest gap was 0.16 on MATH-500.] Matching seeds does not make sampled generation reproducible across GPUs at temperature 0.6. A single fast-tier score on GSM8K, MATH-500 or IFEval has a standard error of about [value from test-retest] points, and we treat a paired difference smaller than [2 × paired SE] as zero. We therefore report those three benchmarks as descriptive at the fast tier and make no claim about single-edit effects on them. Fast-tier MMLU-Pro, HellaSwag, perplexity and KL become predictor targets only after the test-retest study fixes their measurement floors. Each predictive correlation is reported with the measured reliability of its target, because a correlation against an observed change cannot exceed about the square root of that change's reliability.

Why this is honest without being self-defeating: it names what is admissible, gives the number that governs each claim, and states the rule for the remaining benchmarks. It does not say the benchmarks are useless.

### 2.5 Sentences in `paper/sections/experiments.tex` affected by the noise finding

| Line | Sentence (abridged) | Status | Required change |
|---|---|---|---|
| 4 | "Fast scores are the proposed predictor targets, and cheap scores will calibrate a subset." | **Invalidated** | Replace with the Section 2.4 wording: GSM8K, MATH-500 and IFEval descriptive; MMLU-Pro, HellaSwag, perplexity and KL targets after their floors are measured. |
| 4 | "The proposed fast screening tier uses 50 GSM8K, 25 MATH-500, 25 IFEval, 100 MMLU-Pro and 100 HellaSwag items …" | Overstated | "Screening" implies ranking edits on the three small benchmarks. Remove the word and state the per-item standard error. Keep the item counts. |
| 6 | "Local work runs on an RTX PRO 4500 with 32 GB." and the Lambda A100 sentence after it | Incomplete | Cross-host and D3 work also ran on Lambda A10 and an H100. List every host and GPU used, or refer to a host map in the manifest. |
| 7–22 (Table 1) | Base reproduction values with no n and no CI | Unsupported | Add n, samples per item, host and CI (Wilson for binary metrics, as RESULTS.md states). The base cross-host GSM8K gap (0.10 at fast tier) is larger than the 15.5 vs 13.8 difference the table invites a reader to see. |
| 4 | "D3 samples production entries … these quotas resolve to 300 entries." | Needs host map | Add the host of each entry, or refer to the manifest. The 300-entry readout is not yet host-balanced. |
| 2, 6 | "Loop rate is the fraction …"; "The budgets and loop denominators differ …"; "Refusal keyword matching is a surface measure." | Not affected | Keep. The loop-rate sentence should name the token cap wherever loop rate appears. |

Count: **one sentence invalidated** (line 4); **one overstated** (line 4); **one incomplete** (line 6); **one table without n or CI** (Table 1).

Secondary, outside `experiments.tex`:

- `paper/sections/method.tex` line 6: "For each benchmark we will report Spearman correlation and mean absolute error, with bootstrap confidence intervals over edits." The CIs cover edit sampling only and leave out measurement noise. Add the measurement floor and the reliability ceiling. The target Δy = y_t − y_{t−1} also needs a same-host base (`docs/RESULTS.md`, "Δy will be computed against the same-host base").
- `docs/RESULTS.md`, "Interim D3 readout": correct the SE range (Section 2.1); make "usable targets" conditional on test-retest; list host per entry in the family table.

## 3. Reference requirements

### 3.1 The blog as a web citation

```bibtex
@misc{takehi2026wsp,
  author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  title        = {Sparks of Weight-Space Programming: {LLMs} can natively read \& write neural network weights},
  year         = {2026},
  howpublished = {Blog post, MIT CSAIL, 2 November 2026},
  url          = {https://rikiyat.github.io/blog/weight-space-programming/},
  note         = {Accessed 3 November 2026}
}
```

Access date basis: the stored capture `data/papers/takehi2026wsp.txt` was written at 23:49 UTC on 3 Nov 2026. That is the earliest date I can source. Re-fetch the page before submission and update the note to the date the cited text was actually read.

Excerpt status in `paper/refs/verified.yaml`: only the TL;DR sentences (MATH-500 7.9 → 48.1, IFEval raised about 3×) are present. The Experiment 8 table, the sampling footnote (16/4/2/1 samples) and the Gemma section are not. Add one excerpt per location before any of those numbers appear in the paper. The logs repository is a pointer only, with its licence to verify.

### 3.2 arXiv candidates: LLM as editor or proposer, and weight-edit precedents

Titles were checked today with the required command. None of these is in `paper/refs/verified.yaml`, so none may be cited until its excerpt is checked and added (`paper/CHECKLIST.md` line 4; `scripts/verify_refs.py` must exit 0 after the paper step). Status for every row: **to verify** (excerpt and claim).

| arXiv id | Title (checked 4 Nov 2026) | Role | Note |
|---|---|---|---|
| 2309.03409 | Large Language Models as Optimizers | LLM proposes candidates from scored history (OPRO) | Closest LLM-as-proposer precedent. |
| 2309.16797 | Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution | LLM mutates prompts in an outer loop | Outer-loop precedent. |
| 2402.11359 | Offline Training of Language Model Agents with Functions as Learnable Weights | LLM optimises functions, not weights | Precedent for an LLM as optimiser (arm B), not for weight edits. |
| 2407.08770 | Model Surgery: Modulating LLM's Behavior Via Simple Parameter Editing | Targeted parameter edits, no LLM | Non-LLM precedent. `docs/MODEL_CANDIDATES.md` lists it under LLM-assisted editing; reword that. |
| 2212.04089 | Editing Models with Task Arithmetic | Weight-space arithmetic, no LLM | Non-LLM baseline precedent. |
| 2608.24735 | Meta^n: Recursive Self-Improvement through Emergent Depth | Source of the best-of-N warning | Cite only the passage where the authors make that claim. Not checked. |

### 3.3 arXiv candidates: reliability of small-sample benchmark scores

| arXiv id | Title (checked 4 Nov 2026) | Relevance |
|---|---|---|
| 2411.00640 | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations | Standard errors and clustered intervals for benchmark means. Closest to Section 2.4. |
| 2406.10229 | Quantifying Variance in Evaluation Benchmarks | Variance across seeds and prompts; a test-retest reference. |
| 2408.04667 | Non-Determinism of "Deterministic" LLM Settings | Run-to-run and hardware variation at fixed settings; closest to the cross-host finding. |
| 2405.14782 | Lessons from the Trenches on Reproducible Evaluation of Language Models | Harness-level reproducibility. |
| 2308.11696 | Efficient Benchmarking of Language Models | Small-subset benchmarking; relevant to the fast tier's validity. |

## 4. Anonymity and attribution checklist

Checkboxes marked [x] were verified clean in this review. Items marked [ ] need action. File names are given; no matched values are reproduced here.

- [ ] **Author name in paper source (B1).** `paper/main.tex` line 11 sets the author name outside the arXiv guard. The anonymous build hides it, because `tmlr.sty` prints "Anonymous authors" in review mode, but the source uploaded for review carries the name. Wrap the line in `\ifarxivpreprint`, or give the anonymous build a placeholder.
- [ ] **Author-account links (B1).** `README.md` (two links: pipeline badge and clone command) and `site/index.html` (one source link) point to the author's account. Remove them from the anonymous copy, or point them at an anonymised mirror.
- [ ] **Git metadata (B1).** One author name and one e-mail address across 366 commits. An anonymised mirror needs a fresh history. Do not push this history.
- [ ] **Remote URL (B1).** `git remote -v` names the author's personal namespace. Keep it out of the paper and out of the mirror.
- [ ] **Home-directory paths and login-user prefixes.** Found in 14 tracked files: `AGENTS.md`, `CLAUDE.md`, `board/P1.md`, `board/P2.md`, `docs/ORCHESTRATOR.md`, three files in `reviews/P2/`, four files in `sessions/`, `tests/test_eval_lora.py`, `.claude-flow/mods/install.json`. Scrub in the mirror; fix going forward.
- [x] **Host addresses.** The only IPv4-shaped string and the only instance-hostname pattern in tracked files are detection regexes in `scripts/secret_scan.sh`. No instance address is present.
- [ ] **Board posts.** `scripts/board.sh` posts to a GitHub issue and redacts tokens only. Hostnames and paths can get through. Add redaction, or keep `board/` out of the mirror.
- [x] **AI attribution in paper and README.** No Claude, Anthropic, ruflo or claude-flow strings in `paper/main.tex`, `paper/sections/` or `README.md`.
- [x] **Commit trailers.** No `Co-Authored-By`, `Signed-off-by` or "Generated with" trailers in tracked files. `.githooks/commit-msg` rejects them. The one "generated with" hit is docstrip boilerplate in `paper/template/fancyhdr.sty`. Leave the TMLR template unmodified.
- [ ] **Commit subject.** Commit `1b56e27` ("orchestrator: Claude API instance online") names the AI provider in its subject. It is a subject, not a trailer. Reword it in the mirror.
- [ ] **Tool configuration.** `.claude/`, `.claude-flow/`, `.agents/`, the Ruflo block in `CLAUDE.md`, `docs/HARNESS.md` and `docs/ORCHESTRATOR.md` describe the agent tooling. `sessions/`, `reviews/` and `board/` record agent roles. This is not commit attribution, but "no AI attribution anywhere" applies to the public-facing copy. Decision for the author: exclude these from the mirror, or keep them and say so.
- [x] **Model names as study objects.** Claude Haiku 5.5, Claude Opus 5.5 and GPT-6 Astra appear as editors in `docs/`. That is content, not attribution. Keep them, with the terms in Section 1.6.
- [ ] **Anonymous code link (B1).** The paper contains no repository link. `paper/CHECKLIST.md` line 3 is unchecked. Create the anonymised mirror before submission.
- [x] **Author name in paper sections.** The name appears in `paper/main.tex` only, among paper files. Other occurrences are in `README.md`, `reviews/P1/`, `scripts/lambda_job_*.sh`, `scripts/update_readme_status.py`, `sessions/2026-11-03-001.md` and `site/index.html`.
- [x] **This file.** It names no host, address, login user, path or AI tool attribution. Re-check with the secret scan before committing.

## 5. Summary

1. Branch `h5-board-b` (worktree `.worktrees/h5-board-b`), file `docs/reviews/H5/board-b-requirements-2026-11-04.md`. Verdict REVISE.
2. Invalidated sentences in `paper/sections/experiments.tex`: one (line 4, "Fast scores are the proposed predictor targets …"); two more need rewording (line 4 "screening"; line 6 hardware); Table 1 lacks n and CI.
3. Required controls: Gemma-3-270M negative control; cheap editor with and without critic; per-episode matched random-direction; single-draw best-of-N at equal real-evaluation budget; Adam/Muon at equal forward budget; random-ranking arm.
4. Required figures: Figure A (at-a-glance, `docs/FIGURE_STYLE.md` item 1, held-out family starred) and Figure B (build-step chain, item 4, with predictor forecast intervals and failed proposals shown).
5. Top blocking item: no preregistered sealed split, no completed same-host test-retest, so no episode and no fast-tier claim on GSM8K, MATH-500 or IFEval is admissible yet; separately, `paper/main.tex` and the README and site identify the author, which blocks any anonymous submission.
