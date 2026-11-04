# Board B review: paper/method-zoo

Reviewed d1d0d6c after merging main (merge 5466de8); PDF built with latexmk and read with pdftotext. Verdict: **REVISE**.

Scores (1–5): claims supported **3**, novelty **4**, clarity **4**, reproducibility **3**.

Blocking:

1. Refusal direction extraction is materially underspecified: `src/wsp/edits/refusal.py` normalizes the harmful-minus-harmless mean, projects out the normalized harmless mean, and normalizes again; activations are final prompt-token residuals. The draft describes only differences in means. Give the full operation, including prompt and layer conventions, so the projected variant can be reproduced.
2. The LoRA SVD description obscures D1: `extract.py` computes the main `svd` tokens from the rounded effective dense delta; canonical QR/core-SVD LoRA tokens separately describe the unrounded factors. Distinguish these features and identify the pooled baseline inputs: feature version 3, 1098 statistics and 120 singular-spectrum values. The singular-vector projections are not the 120-dimensional pooled baseline.
3. Loop rate is defined too broadly: `eval/run.py` counts a sequence only when it reaches the cap without EOS, and the reported aggregate pools GSM8K, MATH-500 and IFEval. Also the cheap tier has a 1024-token MMLU-Pro override (`configs/eval_cheap.yaml`), so the unqualified 4096-token statement is misleading. State the exception and scoring scope.
4. Provide an exact UTC snapshot time for the live manifest and regenerate counts at that time: at approximately 06:18 UTC the production chain count was 530, versus 523 in the draft. Counts can legitimately change during the build; a date alone cannot identify the snapshot. State D3's planned stratified family counts (60/100/60/30/50), and the faithful tier's at-most-10 selected entries, to make the proposed scope reproducible.

Other observations: the projected kernel's finite linear weight and no-row-normalization formula match `kernels.py`; ancestry grouping uses the full manifest in `splits.py`. The fast subset sizes agree with `configs/eval_fast.yaml`. The base table agrees with `data/runs/p2/base-final-4096/summary.json` and `docs/RESULTS.md` subject to displayed rounding; the stated A100 and $1.99/h agree with `board/lambda_spend.md`. The Heretic citation is supported at the narrower verified README strength in `paper/refs/verified.yaml`; the trials' local selection and constraint are supported by `heretic_ara.py`. Planned predictor results are appropriately presented as future work. No placeholder numbers found.

Writing: “The manifest is a build artefact. It does not show that every row has a completed evaluation.” is accurate but mechanically choppy; combine these sentences. “We propose six predictor families” followed by “We plan these comparisons” is repetitive. These are editorial suggestions, not separate blockers.
