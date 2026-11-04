# Board B review: reliability paper

Reviewed `paper/reliability` at 18de8b5 against H5 Board B §§2–4 and Board A's H5 and P4 wording requirements. Verdict: **REVISE**. Scores: claims-supported 2/5; novelty 3/5; clarity 3/5; reproducibility 2/5. This review concerns the changed paper text, not an H5 episode result.

## Blocking findings

1. The asserted universal ceiling “A correlation against an observed change cannot exceed about the square root of that change's reliability” and the resulting 0.36 requirement for H1 are not established. The P4 Board A review explicitly restricts the square-root relationship to a classical independent additive-noise Pearson model and says it is not an exact Spearman law or a measured ceiling. State assumptions and label any calculation a heuristic, or remove it, including the admissibility-table row. Fixed-subset predictability is not bounded by the binomial planning SE.
2. The cross-host table explicitly provides no run IDs or reproducible panel construction. Identify every per-host run and entry, seeds and analysis procedure, and attach the same provenance to the base and A10 comparisons. The paper says four **edited** entries shared with A10, whereas RESULTS describes four shared entries total, including base; reconcile the denominator and the maxima against the actual records. The cross-host numbers match the aggregate RESULTS transcription, but cannot all be independently certified from the paper's citations. The two base GSM8K values of 0.06 are correct: both `data/evals/qwen3.5-0.8b/base-lambda-*/eval.json` files report 0.06 with 50 items.
3. The qualification summary omits the effect-noise and paired-CI gates and ICC lower bound ≥0.75, and its implication that establishing measurement floors suffices to make targets eligible is weaker than the preregistration. WikiText is diagnostic until window expansion. State the complete criteria, and use Board A's required precise terminology: MATH exact-match, keyword refusal, KL on base-conditioned continuations and share hitting the token cap. In particular pooled hit-1024 is descriptive and is not semantic loop rate. The final sentence allowing effects several times larger than the floors is not a qualified effect-screening result.
4. The base table's blog values have no item/sample counts; the caption admits these are missing, so the requested n per reported number is unmet. Provide verified denominators and samples per item where available, or explain explicitly which are unknown. The pooled Wilson interval assumes independent Bernoulli responses across heterogeneous benchmarks; justify its estimand and assumptions or omit it. The KL self-check is placed under a percent heading despite being in different units.

## Other checks

The two P4 §5 template passages are verbatim, and their trajectory run and n are supplied. No affirmative H1 pass, hardware-free prediction, or loop-repair claim was found in the changed passages. Same-device repeat numbers and cross-host aggregates match RESULTS; the 0.03–0.10 binomial figures are planning approximations with stated independence assumptions. This is not a complete independent audit of every individual numeric cell or underlying run.

The new Atil and Miller verified excerpts support the narrow cited sentences; the blog base values and token-limit definition are supported by the stored excerpts, and the bibliography includes an access date. In the project virtual environment `python scripts/verify_refs.py` and `python scripts/check_cites.py` both passed. Mechanical verification does not establish missing denominators.

`latexmk -g -pdf -interaction=nonstopmode -halt-on-error` succeeded (9 pages). Extracted PDF text shows the three changed tables and captions; the admissibility table produces multiple severe underfull box warnings and should be visually inspected and typeset more cleanly. This review did not visually inspect rendered pages, so it cannot certify clipping or layout.

Machine-sounding or awkward sentences to revise: “Those floors set what a claim may say.”; “The two passages below say what this suite supports.” The phrase “We bounded it three ways” overstates what a fixed-seed repeat and small host panel establish. The rest is mostly direct and concrete.

No paper source was edited and no evaluation was run. Approval requires correction and a fresh Board B review.
