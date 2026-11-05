# Board B review: v2 results, 5 November 2026

Reviewed branch `paper/v2-results` at `acd9673`. Verdict: **REVISE**. The descriptive result is useful and the baseline analysis is candid, but several statements overreach the evidence. This is not yet a full numerical audit; the checks below must not be interpreted as verification of every number in the manuscript.

## Blocking findings

1. Discussion, Reframing: “We attribute the gap to three design features that we did not test separately” claims an explanation without testing it. Present these as possible contributors. “as much a measure of noise as of edit effects” and “Rank agreement with a target this noisy is limited, whatever the predictor” assert a quantitative or universal noise limit not established by the repeat data. The reliability section itself says no reliability coefficient has been measured. Distinguish item-sampling uncertainty from demonstrated repeat noise and avoid asserting an unmeasured ceiling.
2. Discussion calls the prospective protocol “a registered design (pending Board A acceptance of its measured gates)”. Its status line says **frozen prospective design, pending Board A acceptance and measured execution gates**. Acceptance and measured gates are separate conditions; reproduce that status, and avoid implying completed registration acceptance.
3. The title's “Weight Statistics Add Little Beyond Edit Metadata” suggests a general incremental feature comparison. The displayed descriptor versus StatNN comparison concerns particular models, seed and endpoints, with no controlled feature ablation or general inference. Qualify it to the observed comparisons and exploratory, unqualified accuracy targets. The abstract should likewise locate its descriptor comparison explicitly at seed 0 and avoid suggesting an established incremental effect.
4. Explicitly identify the 12 flagged ARA entries and their exclusion from the predictor analysis, with the relevant denominator and provenance. The zoo table alone does not make this exclusion clear.
5. Complete an independent audit of every new table and numerical prose claim against source runs and underlying analyses, including baseline intervals, counts, residual analysis and family summaries. I checked ten cells below; that is a spot check, not the exhaustive verification requested.

## Evidence checked

Directly compared `seed_summary` Spearman mean and ranges in `data/runs/p4/d3v2-*/metrics.json` to the suite table (MMLU-Pro, HellaSwag respectively): `m1-family` −0.08 [−0.23, 0.14], 0.03 [−0.46, 0.31]; `m1_matched-trajectory` 0.45 [0.37, 0.52], 0.46 [0.43, 0.51]; `m2-seed` 0.36 [0.31, 0.44], 0.35 [0.25, 0.50]; `m6-trajectory` 0.14 [−0.10, 0.35], −0.03 [−0.16, 0.08]; `m2-holdout-task_vector` 0.06 [−0.13, 0.35], 0.39 [0.25, 0.51]. All ten agree at displayed precision. The suite table is also consistent with the displayed summary; this does not validate all underlying inputs or other tables.

The manuscript states H1 was not demonstrated, H2/H3 remain unresolved, H4–H6 untested, and targets unqualified, and reports the 260 inferred-runtime labels. The baseline section is a useful, non-defensive analysis, appropriately distinguishing its single-seed refits and exploratory residual result. The reframing is constructive once its causal and noise assertions are softened. The title needs the qualification above; the abstract otherwise accurately conveys the limited scope. Captions supply row counts and sources or run IDs, though the separate evidence analyses should identify executable run/configuration provenance as well as a generated document. Verify all remaining captions and numbers in the complete audit.

Writing: “The programme continues as a readability study” is vague about what is actually planned; say what will be measured. “We attribute the gap to three design features that we did not test separately” sounds formulaic as well as overstated. Prefer direct language specifying possible explanations and their uncertainty.

## Checks and scores

`latexmk -g -pdf -interaction=nonstopmode -halt-on-error` succeeded (13 pages). Log reports no overfull boxes, several underfull boxes including a vertical box of badness 10000; inspect the resulting pagination and tables visually before submission. `python3 scripts/check_cites.py` passed; `python3 scripts/verify_refs.py` reported 35 verified keys. The first attempt with `python` failed because that command is absent; both scripts succeeded with `python3`.

Claims supported: 2/5; novelty: 3/5; clarity: 3/5; reproducibility: 3/5. These scores reflect the current manuscript and the incomplete audit, not a judgment that its underlying analyses are invalid. Merge: no.
