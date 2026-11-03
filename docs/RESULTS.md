# Results

(empty — filled by P2+ runs; every number carries run id, n, CI method)

| Phase | Table | Run id | Status |
|---|---|---|---|
| P2 | Base model reproduction vs blog table | — | pending |
| P3 | Baselines by family / by seed | — | pending |
| P4 | H1–H4 | — | pending |
| P5 | 27B transfer | — | pending |

## P2 reproduction status

Implementation is on `feat/p2-eval`. CPU tests and the eight-item, two-run GPU
determinism smoke test pass. Full benchmark numbers and blog gaps remain pending;
no measurements are inferred from the smoke test. The provisional `p2-base` job
was superseded before any benchmark completed, so it supplies no quoted scores.

`scripts/eval_p2.sh` runs the full 4096-token cheap battery followed by the
32768-token GSM8K/MATH-500 comparison sequentially, then generates the table from
complete per-item records. It records Wilson 95% intervals for binary metrics and
explicitly identifies the different loop-rate denominators. The reference blog
numbers supplied in the task are MATH-500 7.9%, GSM8K 13.8%, IFEval 21.0%, and
loop rate 93% at 32k; they are not reproduced results yet.
