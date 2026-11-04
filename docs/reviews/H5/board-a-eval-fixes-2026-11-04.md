# Board A: eval fixes review

Date: 2026-11-04. Branch: fix/eval-board-a-h5, 54e0f4f. Verdict: **REVISE** (blocking list). This review does not qualify the H5 measurement panels.

## Blocking findings

1. **Matched labels are not enforced.** `src/wsp/predictor/data.py:195` and `:206`–`:219` still subtract a global base or predecessor without examining device, runtime, item, batch, or `matched_base_missing`. Thus unmatched observations silently produce targets, including successors. Wire exact matching into target construction, count missing references, and test cross-device and predecessor cases. `src/wsp/eval/repeats.py:94` also accepts matching null metadata as known; reject unknown values.
2. **Reliability calculation is incomplete and permits unverified pairs.** `src/wsp/eval/repeats.py:23` accepts two missing item hashes and IDs as matching, and `:29`–`:67` omits the preregistered RMSE, median/max absolute differences, SD and Bland–Altman limits, paired item discordances, matched-reference deltas, rank/ICC bootstrap intervals, and continuous-metric analyses. The returned binomial SE is zero at boundaries (`:58`), contrary to the protocol's requirement to never report zero uncertainty at 0 or 1. Require known identity and implement the specified analyses before claiming B3 complete.
3. **No base run for a new edit after resume.** `src/wsp/eval/batch.py:307`–`:318` loads a pre-existing base even if it lacks the new metadata. `matched_reference` then correctly rejects it, but no matched base is produced and the predictor still uses the old base. A resumed production run needs a compatible base or must leave targets missing and explicitly report that condition.
4. **B5 wording remains.** `docs/RESEARCH_NOTES.md:63` still describes an LLM-judged rate; the requested grep also finds historical uses at `docs/reviews/H5/board-a-prereg-2026-11-04.md:17` and an unrelated use at `docs/ORCHESTRATOR.md:48`. Correct the misleading metric claim and resolve the explicit wording requirement without erasing historical evidence.

## Evidence and limits

The per-benchmark rates in `src/wsp/schema.py:114` use benchmark-specific record counts; pooled loop rate remains weighted over the original three benchmarks. `src/wsp/eval/run.py:223`–`:232` excludes EOS at cap. CLI token overrides are rejected. Repeat destinations are separated and forced in `src/wsp/eval/batch.py:280`–`:326`; actual and nominal batches are recorded. The device ID is restricted to H followed by digits (`src/wsp/eval/metadata.py:14`–`:17`). No infrastructure identifier was found in the new metadata fields. Existing aliases retain their values.

For the three-edit panel (0,1), (.5,.5), (1,0), hand ANOVA gives MSR=0, MSC=0, MSE=.5 and ICC(2,1)=-3; the function returns -3, MAD 2/3 and Spearman -1. The 27 equally likely bootstrap samples of differences (1,0,-1) have discrete 2.5% and 97.5% quantiles -1 and 1; the seeded 10,000-replicate function returns [-1,1]. This checks one example, not all edge cases.

CPU suite: 408 passed, 8 deselected. A JSON parsing sweep of 119 existing eval.json files found no malformed files; this is not a full new-reader compatibility test. The predictor loader on the complete eval directory raised `ValueError: conflicting target records for base`; this may reflect existing mixed artifacts, so compatibility is not established. No GPU test was performed. The required faithfulness, repeat panels and qualification remain outstanding.
