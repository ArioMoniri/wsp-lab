# Board B review: paper/intro-related

Reviewed head 0f9df52911cf77cb2f0153332e388c865cfea6f0; abstract, introduction, related work, and built PDF. Verdict: **REVISE**.

Scores (1–5): claims-supported **2**, novelty **3**, clarity **3**, reproducibility **2**. Approval requires at least 4 for claims and reproducibility. This is a study proposal, not yet a reproducible empirical paper: no runs, data, model configurations, evaluation protocol, or results are supplied in these sections. The novelty is a plausible combination of existing ideas, not an established empirical contribution.

## Blocking findings

1. The task-vector citation reverses the stored supporting excerpt. The sentence is: “Work on task-vector composition finds that interference depends on training-format serialization: a trained wrapper can mask interference that appears under an untrained chat template.” The `claim_we_make` repeats this assertion, but its own excerpt says there is a collapse under the training-format wrapper and no collapse under the untrained chat template. Correct the paper and the verified record after checking the source; the present wording is unsupported by the stored excerpt. This is one over-claim/inaccurate cited sentence, even though it matches the flawed `claim_we_make` verbatim.
2. The submission PDF shows “Ario Moniri” on page 1. `main.tex` uses `[preprint]`, which enables the named author in the local style. Use the anonymous review mode (the style's default, without `preprint`) for submission and inspect the resulting PDF and metadata. The checked-in `tmlr.sty` and `tmlr.bst` match the corresponding files in `paper/template` byte for byte; this comparison does not independently establish their provenance.
3. The instruction for this review forbids numbers, results, or placeholder values we do not have. The paper includes sourced prior-work results and planned numeric thresholds, not claimed new experimental results. The cited numbers are traceable to the verified entries, and hypothesis thresholds to HYPOTHESIS.md; no invented study results or numeric placeholders were identified. If the prohibition literally includes all numbers regardless of provenance, these numbers must be removed or the rule clarified.
4. H1 gives ρ ≥ 0.6 and baseline comparisons but does not say the predictor must beat both baselines, as required by the hypothesis. H1's falsification threshold (ρ < 0.3 on all benchmarks) and H3's (transfer ρ < 0.2) are omitted; H2's within-noise falsification is also omitted. H2's ≥15% relative MAE and length ≥4 and H3's ≥0.5 transfer threshold match. H4 preserves the two directional predictions but describes the second as a question. State all decision and falsification criteria explicitly and preserve H1's baseline requirement.

## Citation and literature audit

Compared each cited key with `claim_we_make` and its stored excerpt: `takehi2026wsp`, `han2026survey`, `w2t2026`, `deltaactivations2025`, `fastprobing2026`, `tuneahead2026`, `taskvectors2026interfere`, `predictability2026dynamics`, `leapverify2026`, `mamba3`, `jepawg2026physics`, `drivedecay2026`, `jepahmm2026`, `heretic2026`, `pksharma2026heretic`, `surgicalrefusal2026`, and `singleattack2026moe`. Apart from the task-vector discrepancy above, I found no cited sentence stronger than the corresponding recorded claim. This audit uses the stored passages; it is not a fresh independent validation of each source.

The six discarded assertions in the related-work map are not relied upon: JEPAWG optimizer attribution, demonstrated Leap+Verify speedup, unconditional JEPA/HMM equivalence, complete MoE refusal bypass, gradient-free contrastive steering, and reliable multi-agent evaluation. The physics-coupling-to-edit-descriptor adaptation remains a proposal, not a demonstrated equivalence. Takehi is framed as a motivating report without claiming advance predictability; W2T's benchmark and transfer limits are explicit. JEPAWG is correctly limited to its physics configuration, though its connection to language-model edits remains speculative. The text identifies the work as planned and makes no new outcome claims. Related-work numbers are attributed to their sources; no run IDs or tables exist for new results.

## Writing

Three sentences read mechanical or strained and should be rewritten:

- “The alignment-edit sub-study puts the prediction problem on concrete ground across model scales.”
- “Capability goes into the measurements; we do not assume it survives.”
- “We keep the boundaries of the source claims in view and do not assume that prediction can substitute for evaluation.”

The guide's ban on filler and main-text bullets is otherwise respected. Some sentences bundle claims, and present tense for planned parts can imply completed work; consistently use future or explicitly proposed language. The abstract's long four-clause question remains cumbersome. The title presents a state-space predictor rather than an untested comparison, and should reflect the study's prospective status.

## Build

Ran `source scripts/env.sh && cd paper && latexmk -pdf -interaction=nonstopmode main.tex`; latexmk reported the PDF up to date and successful. Read all four pages via `pdftotext -layout`. The build uses the preprint option and is not anonymous. No manuscript branch edits were made.
