# Board B re-review: reliability paper

Reviewed `paper/reliability` at `d50d4d1` against the 2026-11-04 Board B review. Verdict: **APPROVE** for this scoped revision. Scores (out of 5): claims-supported 4, novelty 3, clarity 4, reproducibility 4. These scores do not certify the full paper or qualify any predictor target.

## Prior blockers

1. **Closed.** The square-root statement is explicitly restricted to the classical independent additive-noise Pearson model, identified as a heuristic, and separated from Spearman and measured reliability. The 0.36 is orientation under that model, not an established requirement. The table no longer states a universal ceiling.
2. **Closed.** The text identifies the 10 controls, additional H100 entry, four A10 edits and base separately, run identifiers, seed, batch, record selection, statistics, and the mixed-source-hash exception with sensitivity to exclusion. RESULTS now agrees on 11 and four edits plus base. The ad hoc analysis is described but not retained as a script, so exact regeneration still requires implementing it; this limits reproducibility and warrants follow-up, not a blocker for the present descriptive claims.
3. **Closed.** The effect bound and paired-interval gates, rank correlation and ICC lower bounds, panel variation, and within- and cross-device ordering are stated. WikiText remains diagnostic pending expansion; the stated metric definitions and cap-hit limitation are appropriately narrow. No floor alone qualifies a target.
4. **Closed.** The table reports counts, available blog sampling rates and unknown denominators; it omits the pooled interval and removes the self-KL from the percentage table.

The three previously flagged phrases were replaced. The reviewed passages follow the writing guide's direct style and specific limitations; no prohibited filler or main-text bullet list was found. This is a scoped prose check, not a source-by-source audit of the entire paper.

## Checks

Independent spot checks of `data/evals/qwen3.5-0.8b/base*/eval.json` confirm local/H100/A10 GSM8K 0.16/0.06/0.06; MATH 0.16/0.08/0.28; IFEval local/A10 0.40/0.44; local/H100 over-refusal 0.10/0.20; and local/H100 perplexity 22.9041/22.9159, rounding to 22.90/22.92. For edit `0a841d4082532e8b`, local/A10 records yield absolute differences 0.06 GSM8K, 0.16 MATH and 0.06 over-refusal, consistent with the corresponding A10 maxima in the table and `docs/RESULTS.md`. These spot checks do not independently recompute every aggregate or correlation.

Both `scripts/verify_refs.py` and `scripts/check_cites.py` passed with the project virtual environment. `latexmk -g -pdf -interaction=nonstopmode -halt-on-error` succeeded (10 pages). I rendered page 7 with `pdftoppm` and visually inspected the cross-host reliability table: all columns and caption are readable and within the page, with no apparent clipping. The caption has loose spacing and some underfull warnings remain, but these do not obstruct reading.

Remaining work outside this approval: retain the aggregation script for exact reproduction, run the stated seed panels and qualification analysis before making qualified effect or ranking claims, and reconcile the IFEval checker as already disclosed.
