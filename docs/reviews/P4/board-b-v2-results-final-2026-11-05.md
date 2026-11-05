# Board B final check: v2 results, 5 November 2026

Final check of branch `paper/v2-results` at `8390815` against my [re-review](board-b-v2-results-rereview-2026-11-05.md) (R1 to R5). I read the diff `76d2778..8390815 -- paper/` line by line, read the worktree without modifying it, and built from a `git archive` copy under `/tmp`.

Verdict: **APPROVE**, with two trivial wording items that I do not need to see again.

## Checks run

| Check | Result |
|---|---|
| `scripts/audit_paper_numbers.py` | PASS. 723 values, 90 assertions, 32 presence claims, 0 failures, 0 unclaimed tokens |
| `scripts/check_cites.py` | Passed. 35 verified keys, 10 section files |
| `scripts/verify_refs.py` | All 35 keys verified |
| Build (`latexmk -g -pdf`, clean export) | Exit 0, 15 pages, 0 overfull boxes |

## R1 to R5

**R1 (seed-1 panel): closed.** I recomputed the panel from the raw files: seed 1 from `data/runs/p2-batch/20261105-050543-aad6c310/<id>/repeats/seed1/eval.json` and seed 0 from `data/zoo/qwen3.5-0.8b/<id>/eval.json` (controls) and `data/evals/qwen3.5-0.8b/base/eval.json` (base), all device H0. Every number in the new paragraph reproduces:
- Mean absolute difference, maximum and Spearman ρ: GSM8K 0.055 / 0.12 / 0.15, MATH-500 0.036 / 0.12 / 0.46, IFEval 0.029 / 0.04 / 0.10, over-refusal 0.049 / 0.10 / 0.67.
- MMLU-Pro, HellaSwag and WikiText perplexity are identical. KL differs by at most 0.0003 nats (mean 0.0001). The scribe's correction of RESULTS.md's "identical" for KL is right.
- Seed SD of the paired change, n = 10 controls: 0.0504, 0.0268, 0.0248, 0.0356 (over-refusal), 0.0124 (refusal), and 0 for the two likelihood targets. These match `metrology_budget_output.txt` lines 19 to 27, and "between 0.0124 and 0.0504 for the five sampled targets" is correct.

The Reliability section no longer says that no seed repeat exists, and the Discussion uses the same panel.

**R2 (item SE): closed.** The text defines the item SE of a paired change (SD of item-level paired differences against the device-matched base, over √n, combined as a root mean square). That matches `metrology_budget.py` lines 120 to 180. The new table gives 0.0384 / 0.0246 / 0.0295 for MMLU-Pro and 0.0131 / 0.0136 / 0.0000 for HellaSwag, which match lines 42 and 44 of the output. The wording is now "consistent with" for HellaSwag and "does not account for" for MMLU-Pro.

**R3 (HellaSwag θ): closed.** θ = −0.034 [−0.118, 0.037] appears beside 0.146 [0.069, 0.214], on 345 rows and 278 clusters, both matching `round2_residual_estimand_output.txt`. The table caption says that the two column groups use different row sets. The "narrower claims" paragraph now says the advantage is "on MMLU-Pro and none on HellaSwag".

**R4 (hypotheses): closed.** The Introduction now says six hypotheses, H1 to H4 registered on 3 and 4 November, H5 and H6 added on 4 November, and the 5 November split into H1a/H1b, with the original-versus-prospective H1 distinction. It reports "H1 not demonstrated, H2 and H3 undetermined, H4 to H6 not tested", which is the status sentence in `docs/HYPOTHESIS.md` line 5 and agrees with the abstract and `experiments.tex`. Dates match the headings in HYPOTHESIS.md.

**R5: closed.**
- **260 and 85.** I ran `FeatureDataset` on the real data with torch stubbed. It gives 345 rows, 345 matched, 260 with inferred runtime and 44 missing. Reasons: `matched-inferred` is 98 abliteration, 59 LoRA, 60 random-direction, 30 task-vector and 13 chain, which sums to 260. `matched` is 85 chain rows. The 98 / 60 / 59 family counts equal the fold test sizes in the D3 audit (`audit.txt`, 260 matched, all inferred). So the claim that the 260 are the matched D3 edits and the other 85 are non-flagged chain rows holds. The 13 chain rows that sit inside the 260 do not break it.
- **Component.** Defined from PREREG section 2 as a union of whole ancestry clusters, and "both intervals" and "prospectively demonstrated" power are in the pass and kill rules.
- **Targets list, "recorded scores", forensic table.** The target list is fixed, "recorded scores" is replaced by the selection-trial refusal and KL, and the forensic table is cited.
- **Family split.** HellaSwag StatNN 0.11 and M1-matched 0.31 are stated, matching Table 6 and the family-held-out row.

## Residual items (wording only, no new data)

1. `reliability.tex`, last sentence of the seed-1 paragraph: "The sampled-generation differences are the same size as the cross-host gaps in Table~\ref{tab:devices}, so those gaps are consistent with sampling noise." That holds for the sampled benchmarks. It does not hold for the likelihood-scored targets in the same table: MMLU-Pro (0.008), HellaSwag (0.003), WikiText (0.013) and KL (0.0002) have nonzero cross-host gaps, while the seed panel shows exactly zero seed variation for them. Those gaps cannot be generation-seed noise, and they are the two targets the paper analyses. Restrict the sentence to the sampled benchmarks ("GSM8K, MATH-500, IFEval and over-refusal") and add that the likelihood-scored gaps are host or numerical differences.
2. `discussion.tex`, Table~\ref{tab:residual} caption: "centred" should be "centered" (the rest of the paper uses American spelling).

## Scores (1 to 5)

Claims supported 4. Novelty 3. Clarity 4. Reproducibility 4, with the audit (now 723 values) and clean build.

## Merge

**Yes**, once item 1 is fixed. It is one sentence and needs no rerun of the numeric audit.
