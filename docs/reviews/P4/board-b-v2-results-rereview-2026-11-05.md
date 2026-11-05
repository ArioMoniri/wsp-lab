# Board B re-review: v2 results, 5 November 2026

Re-review of branch `paper/v2-results` at `76d2778`, against my REVISE of `acd9673` ([first review](board-b-v2-results-2026-11-05.md)). I read the worktree without modifying it. I built and ran the audit in scratch copies under `/tmp`.

Verdict: **REVISE (narrow)**. All five blocking items are closed or closed with a small residual. The revision is a real improvement, and the audit is the strongest reproducibility asset the paper has had. My re-read of the changed sections found three residual problems of the same kind as the original item 1: a claim that outruns what its cited source shows. Each is a short edit. None needs new data. I would approve once they are fixed.

## Blocking items from the first review

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Causal and noise overreach in Discussion | Closed, with residuals R1 and R2 below | “We attribute…” is now “may contribute… names candidates, not measured causes.” The “ceiling” sentences are gone. Noise statements now cite the repeat data (cross-host MAD 0.008/0.003, ρ 0.75/0.76 at n = 11; A10 0.010/0.003 at n = 4; seed SD 0.0000). I checked these against reliability Table 3 and `metrology_budget_output.txt`. |
| 2 | Registration status | Closed | The string “frozen prospective design, pending Board A acceptance and measured execution gates” appears verbatim in `reliability.tex`, `experiments.tex` (base-repro caption) and `discussion.tex`. It matches the first sentence of the `PREREG_H1_TF.md` status line. “Registered design (pending acceptance of its gates)” is gone. |
| 3 | Title and abstract | Closed | The title is now “Exploratory Comparisons of Weight Statistics and Edit Metadata for Benchmark Changes in Qwen3.5-0.8B Edits.” The abstract states seed-0 StatNN and descriptor values, the unqualified accuracy endpoints, and the limit to “the specific models, seed-0 baselines and accuracy endpoints reported here.” No incremental-effect claim remains. |
| 4 | Twelve flagged ARA entries | Closed, minor residual R5 | `zoo.tex` gives the flag name, the D6 addenda 2 and 3, and the exclusion from every predictor analysis. It gives the denominators 1,247 = 1,235 + 12, then 389 evaluated, 345 matched in v2, 300 in D3 and 260 matched in D3. Table 1 has a Smoke column (12). I recounted the manifest: 1,255 lines, 12 flagged rows, and column sums of 1,235 unflagged and 1,247 total. |
| 5 | Independent audit of every table and numeric claim | Closed, with the caveat in “Audit” | See below. |

## Audit (item 5), run independently

- `python3 scripts/audit_paper_numbers.py` in the worktree: **PASS**. 673 values checked, 85 assertions, 28 presence claims, 0 failures, 0 unclaimed tokens. The 303 prose and 464 table tokens are all classified. 15 seed labels and 59 declared tokens are listed with reasons, which I found acceptable (hardware, cited literature values, plan parameters).
- Planted errors in a scratch copy (`git archive` plus a data symlink), each reverted afterwards:
  - Prose: discussion `0.008` changed to `0.009`. Result: **FAIL** (1 failure, 5 unclaimed tokens).
  - Table: reliability Table 3 MMLU-Pro Spearman `0.75` changed to `0.77`. Result: **FAIL**.
  - Count: zoo `1,235` changed to `1,236`. Result: **FAIL**.
  - Restored copy: PASS with 673. The audit does fail on wrong numbers.
- `python3 scripts/check_cites.py`: passed (35 verified keys, 10 section files). `python3 scripts/verify_refs.py`: all 35 keys verified.
- PDF build with `latexmk -g -pdf` on a clean export: exit 0, **14 pages**, no LaTeX warnings, 0 overfull boxes. Underfull vboxes remain, but page fill is even (45 to 60 lines per page, with a short last page), so I saw no layout problem.

The caveat is that the audit checks numbers against sources. It cannot check which numbers a section chooses to show, or whether prose contradicts a cited source. R1 to R3 below were all missed for that reason. “0 mismatches” is a statement about the 673 values, not about the claims built on them. The report in `docs/reviews/P4/paper-v2-audit-2026-11-05.md` should say so in one sentence.

Two scope notes. The audit covers eight sections and not `related.tex` or `appendix.tex`. `related.tex` quotes literature numbers (MAE 0.32, Spearman 95.35%, AUROC above 0.75, RMSE 1.47 pp, +12.5 points), which `verify_refs.py` does not check against the claims. Please state that scope in the audit report. Also, `docs/reviews/P4/paper-v2-audit-2026-11-05.md` is 814 lines of PASS entries. A summary table at the top (it has counts) is enough, but the per-token lines make it hard to review.

## Residual problems

**R1. The paper says no different-seed repeat exists, and then cites one.** `reliability.tex` line 6: “A repeat panel with seeds 1 and 2 would measure that noise, and we have not run one yet.” The same section opens with “None of the three estimates noise under a different sampling seed,” and says seeds 1 and 2 “must be estimated in separate repeat panels.” But `docs/RESULTS.md` (lines 170 to 182) records a seed-1 repeat panel on the ten controls plus base (run `p2-batch/20261105-050543-aad6c310`). Its generation-seed differences are nonzero on every sampled benchmark (mean absolute difference 0.055 on GSM8K, 0.036 on MATH-500, 0.029 on IFEval). `metrology_budget_output.txt` lines 18 to 26 report the corresponding seed SD of 0.0504, 0.0268 and 0.0248. The Discussion cites that file for “Seed-to-seed SD … is 0.0000 for both targets,” so the paper uses the seed panel for the two likelihood targets and denies it exists in the Reliability section. A reader sees a contradiction. The accurate statement is that one seed-1 panel exists (n = 10 controls, one host) and shows exactly zero seed variation for the likelihood-scored targets, as expected by construction, and sampling noise on generative targets of the same size as the cross-host gaps. That last finding also matters for Table 3: the paper currently suggests cross-host gaps may reflect host effects, while the project’s own results note reads them as sampling noise. Please reconcile the Reliability text with RESULTS.md and report the seed panel in Table 3 or beside it.

**R2. “Item sampling therefore accounts for the residual spread” rests on one target, and a quantity the paper never defines.** `discussion.tex` line 12. The comparison is HellaSwag residual SD 0.0131 against “the 0.0136 item standard error.” The preceding sentence gives binomial standard errors of 0.045 and 0.049 for the accuracy of a single 100-item score. The 0.0136 is a different quantity: it is the item-sampling SE of the paired change Δy (`sd_itemSE` in the budget table). The reader sees two “item standard errors” a factor of three or more apart, with no explanation. More importantly, the same table row for MMLU-Pro (within family and device) has residual SD 0.0384, item SE 0.0246 and an estimated population signal SD of 0.0295. For MMLU-Pro, item sampling does not account for the spread by that analysis. The Discussion reports only the case that supports the point. Please either show both targets or restrict the sentence to HellaSwag and say why MMLU-Pro behaves differently. The wording should also be “is consistent with” and not “accounts for,” since an SD at or below the item SE shows no resolvable signal there, which is not the same as the SE explaining the spread.

**R3. The residual analysis is reported for MMLU-Pro only.** `discussion.tex` lines 20 to 22 state that “the accuracy data also show an exploratory within-family signal in a residual analysis” and then give θ = 0.146 [0.069, 0.214], noting that the residual is built from MMLU-Pro. The same output file (`round2_residual_estimand_output.txt`) gives HellaSwag θ = −0.034 [−0.118, 0.037] on the same 345 rows and 278 clusters. That is no advantage for StatNN, and the interval spans zero. “The accuracy data” (plural) reads as support across both targets. The HellaSwag line must be reported next to the MMLU-Pro one. It strengthens the paper’s case for treating this as exploratory, and it explains why the next study uses an MMLU endpoint, which the paper should say if that was the reason. Omitting it is the kind of selective reporting the audit cannot detect.

**R4. Intro and experiments disagree on hypotheses H2 and H3, and on the number of hypotheses.** `introduction.tex` line 9: “the predictor-suite section reports outcomes for H1 and H2 and leaves H3 undetermined.” The abstract, `experiments.tex` and `PREREG_H1_TF.md` section 8 say H2 and H3 are undetermined, and the suite does not test H4 to H6. Line 7 also says “We registered four hypotheses” while the abstract and Discussion refer to H4 to H6. Please align to “H1 was not demonstrated; H2 and H3 are undetermined; H4 to H6 were not tested.” Say once that H5 and H6 were added on 4 November and that H1 became H1a/H1b on 5 November (`HYPOTHESIS.md` line 5). The paper’s “pre-registered H1 criterion” is the original one, and a reader following `PREREG_H1_TF.md` will find a different H1a.

**R5. Smaller items.**
- The 12 flagged entries are identified by flag, count and D6 addenda only. The IDs are in `docs/reviews/P1/heretic-ara-rebuild-forensic-2026-11-05.md` and in the PREREG table, but not in `DECISIONS.md`, which is the file the paper cites. Add the forensic table to the zoo citation, or list the IDs in the appendix.
- `zoo.tex` line 25 says the flagged entries “remain in the manifest with their identities, deltas and recorded scores,” then says none of them has an evaluation record. If “recorded scores” means the selection-trial KL and refusal values, say so.
- `reliability.tex` line 60 opens “The fast-tier targets are MATH exact-match, keyword refusal, keyword over-refusal, KL …, and the share hitting the token cap,” which omits MMLU-Pro and HellaSwag, the two targets the paper analyses. `experiments.tex` calls MMLU-Pro, HellaSwag, WikiText and KL the candidate targets. Say “the targets named in the protocol” or fix the list.
- The Discussion’s “On the family split, StatNN is near zero” (HellaSwag 0.11) leaves out that M1-matched reaches 0.31 on HellaSwag there, above StatNN. Mention it with the MMLU-Pro exceedances.
- Spelling mixes “programme” with “generalizes” and “neighboring”. Pick one.

## The two 260 counts and the terminology

**The two 260 counts.** They are distinguished, but only just. `experiments.tex` line 35 says the run files mark 260 targets as inferred-runtime, and that the 345 rows “are not the 300-entry D3 subset, and they are not the 260 matched D3 edits counted in the reliability section.” The sources show two different quantities: the v1 D3 audit (`board-a-d3-audit-20261104/audit.txt`: 300 entries, 260 matched, all 260 inferred) and the v2 run files (`targets_with_inferred_runtime` = 260 of 345 matched rows, which leaves 85 not inferred). The text distinguishes the 345 from the D3 260, and the audit asserts both values. It never says that the v2 count is 260 **of the 345**, or that 85 rows carry measured runtime. A reader cannot tell whether the two 260s are the same edits or a coincidence. Please write “260 of the 345 rows” and state what the other 85 are. The Discussion’s “The 260 targets with inferred runtime metadata block qualification (reliability section)” points to the D3 260 in the reliability section, but the analysis it qualifies is the v2 one. Point it at the experiments count, or name both.

**Components versus clusters.** The paper is consistent with `PREREG_H1_TF.md` in usage. “Clusters” means the ancestry clusters of the historical v2 runs (278 on 345 rows, in Table 5, the method section and the Discussion’s θ analysis). “Components” means source-closure components of the prospective design (at least 100 valid fresh components, matching PREREG sections 2 and 4). The paper never defines “component,” and the Discussion uses it as if it were known: “fresh items and components.” One sentence is enough: a component is a connected set of edits sharing ancestry or source (PREREG section 2), and is stricter than an ancestry cluster. The pass rule in the Discussion also omits that PREREG requires both the component-only and the joint component/item-block intervals, and a prospectively demonstrated power of 80% (it says “at least 80% power”). Add “both intervals” to the pass and kill rules.

## Writing

The Discussion is much better than the first version and reads like a person describing results. The new “narrower claims” paragraph and the readability study description are specific. Remaining prose problems, all small:
- “The repeat data are narrower than a noise estimate” (Discussion, line 14) is unclear. Say “The repeat data do not amount to a noise estimate.”
- The “not X, but Y” construction is frequent: “names candidates, not measured causes,” “not predictors,” “describe training and split variation, not confidence intervals,” “planning values, not the final statistic.” Each is defensible. Cut about half.
- The Discussion sentence on item sampling would read better as two sentences.

## Scores (1 to 5)

| Criterion | First review | Now | Reason |
|---|---|---|---|
| Claims supported | 2 | 3 | Causal and universal-noise claims are gone, scope is right, every number is traceable. R1 to R3 are the remaining claim-versus-source gaps. Fixing them reaches 4. |
| Novelty | 3 | 3 | Unchanged. The descriptor-versus-StatNN finding and the candid baseline analysis are the contribution, and the paper no longer overstates them. |
| Clarity | 3 | 3 | Better prose. The undefined “item standard error,” the 260 ambiguity and the H1/H2 inconsistency hold it back. |
| Reproducibility | 3 | 4 | A runnable audit that fails on planted errors, a clean build, and run ids in captions. One point withheld because the audit report and its scope are not explicit about what is unchecked, and R1’s seed panel is not in the paper. |

## Merge

**No**, pending R1 to R4. Estimated effort is under an hour of editing and one rerun of the audit script. I do not need a third full re-review; a check of the diff for R1 to R4 against this note is enough.
