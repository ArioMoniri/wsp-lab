# Board B re-review: paper/method-zoo

Verdict: **APPROVE** — fit to merge as the current draft, not approval as a completed empirical study.

Scores (1–5): claims-supported **4**, novelty **4**, clarity **4**, reproducibility **4**.

Reviewed head `76eb3c7b377df8ec956c0a231d7d60a104dc2de4` in the paper worktree, including `git diff d1d0d6c...76eb3c7 -- paper`. Read reviewer-journal.md, REVIEW_PROTOCOL §Board B, WRITING_GUIDE, DECISIONS and the prior four-blocker Board B review. Ran `source scripts/env.sh && cd paper && latexmk -pdf -interaction=nonstopmode main.tex` successfully (targets up to date), converted main.pdf with pdftotext, and read the rendered text. No branch source edits.

## Prior blockers

1. **Closed: refusal direction.** The revised zoo paragraph matches `src/wsp/edits/refusal.py`: first 400 prompts per dataset by default, seeded subsampling for fewer, system/user chat template plus generation prompt and response prefix, system fallback into the user message, final prompt-token residuals at embedding and every layer. `directions_from_means` computes normalize(bad−good), removes its projection onto normalize(good), then normalizes again. The paragraph now states these operations. Layer interpolation and no-row-normalization classic kernel remain accurately distinguished. More implementation detail (fixed template date, prefix detection, float64 accumulation and dataset commits) is available in the extraction manifest/code; not a remaining blocker for this draft.

2. **Closed: feature kinds and pooling.** `extract.py` computes primary `svd` from effective bf16 dense deltas, including LoRA, and separately emits `lora` canonical tokens for the unrounded scaled factors. `svd.py` uses QR/core SVD for those factors, fixes paired signs using the largest absolute left component, and projects left/right vectors with fixed random matrices. Feature version is 3. `pool.py` has 15 types, four depth buckets, 18 statistics and eight singular values: 18×(1+15×4)=1098 and 8×15=120. Pooled spectra take column zero of the primary tokens; they are not vector projections. The revised method matches.

3. **Closed: loop rate and tiers.** `eval/run.py` truncates at first EOS and flags only sequences of max_new_tokens length whose last token is not EOS. Loop aggregation covers GSM8K, MATH-500 and IFEval. The revised definition says this. `eval_cheap.yaml` specifies 4096 with the 1024 MMLU-Pro override; `eval_fast.yaml` agrees with the listed item counts and 1024 cap. D2/D3 agree with fast predictor targets, cheap calibration and faithful 32768 reserved for the base and at most ten selected entries. These are budgets/plans, not claimed completed zoo evaluations.

4. **Closed: snapshot counts and D3 scope.** Unique IDs, last row wins, excluding headers reproduces every table cell on the manifest prefix identified below. The exact UTC time and counting rule are now explicit. D3 seed 0, sorted production IDs, sampling without replacement, shortfall rule and quotas match `configs/eval_subset_d3.yaml` and `eval/batch.py:subset_ids`. Resolution is 60+100+60+30+50=300 on this snapshot.

## Independent snapshot reconstruction

The manifest continued to grow. My read contained 1132 JSONL lines, 1124 unique IDs and 544 production chain steps, seven beyond the paper's snapshot. Its SHA256 was `ccb3aad8c22ce3dd4326cec868d76ba16287188a09b7d1ac0d9e7e20fea5e18a`. This later count does not contradict the time-qualified table.

The first **1125 lines / 575350 bytes** reproduce the table exactly. Prefix SHA256: `e145121898e16c17a63e4c556cea771fe9491b0e0fd232ed185cf774d51144c5`.

| Family | Production | Smoke | Unclassified | D3 selected |
|---|---:|---:|---:|---:|
| random_dir | 150 | 0 | 3 | 60 |
| abliteration | 200 | 0 | 5 | 100 |
| heretic_ara | 0 | 0 | 1 | 0 |
| lora_sft | 120 | 0 | 1 | 60 |
| task_vector | 80 | 0 | 0 | 30 |
| chain | 537 | 0 | 20 | 50 |

Reproduction: read the first 1125 lines as JSON; assign `by_id[row['id']] = row` for rows with IDs; count `(family, eligibility)` on by_id.values(), treating absent eligibility as unclassified. Apply the D3 quotas to sorted production IDs with one `random.Random(0)` in configured family order. All quotas are filled. The later live manifest also fills all quotas.

The earlier Board A P2 re-review used a **03:33:38** snapshot with only seven production task vectors and no eligible chain rows: 60+100+60+7+0=227. The 20 legacy chains still lack eligibility now; newly generated production chains, rather than relabelling those legacy rows, fill the current quota. Thus 227 and 300 describe different snapshots. The paper explicitly attaches 300 to Table 1 and states the shortfall rule, so the dependence is explained adequately. Optional archival improvement: store the prefix or its hash beside the paper. JSONL rows have no per-row wall-clock timestamp, so I can verify the contents of the claimed snapshot but cannot independently authenticate 06:20:20 from those rows alone.

## Claims, citations and writing

Checked all 17 cited keys against their stored supporting excerpts in `paper/refs/verified.yaml`; the claims retain their supported scope. `python scripts/verify_refs.py` exits 0. W2T numbers, fast probing and TuneAhead numbers, JEPAWG covariance omission, ResidualPred, HMM conditions, Heretic's default modifier, and refusal-study caveats agree with those excerpts. No new citation or unsupported performance result was introduced. The base table matches `data/runs/p2/base-final-4096/summary.json` at displayed precision; the text acknowledges unmatched budgets, loop denominators, the interrupted 32k run and unresolved IFEval checker differences. Predictor outcomes remain explicitly future work. There are tables but no new figures requiring caption review.

Nonblocking writing observations, quoting machine-sounding passages: “We will treat prior findings as evidence about their tested settings.” is generic and repeats nearby caveats; delete it. “We propose three components. First, we will build…” reads like a formulaic outline; the concrete actions could stand without the numbered prose. “Refusal keyword matching is a surface measure. It does not establish harmfulness.” remains choppy; combine into one sentence. The previous redundant “We plan these comparisons” is removed, and the manifest caveat is now combined as requested. None changes scientific meaning or prevents merging this draft.

Remaining blocking mismatches: **none**.
