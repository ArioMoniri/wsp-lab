# Review protocol (strict, two-board)

Nothing merges to `main`, and nothing enters the paper, without passing **both** boards. Reviews are new
`delegate_task` calls with `role="review"`, read-only, never the author's model instance.

## Board A — ML professional (correctness)
Role file `.claude/agents/reviewer-ml.md`. Checks: data leakage across splits (by family, by seed, by base
model), seed handling, silent fallbacks, metric definitions, statistical tests (bootstrap CI, n reported),
GPU numerics (bf16 vs fp32 in SVD), reproducibility (`manifest.json` ↔ config hash ↔ git SHA).
Verdict: `APPROVE` | `REVISE (blocking list)` | `REJECT`. Blocking items must be fixed and re-reviewed.

## Board B — Journal reviewer (scientific claim)
Role file `.claude/agents/reviewer-journal.md`. Reviews in the style of a TMLR reviewer: are the claims
supported by the evidence, is the related work accurate (every citation checked against
`paper/refs/verified.yaml`), are negative results reported, is the writing precise, do figures follow
`docs/FIGURE_STYLE.md`. Scores: claims-supported 1–5, novelty 1–5, clarity 1–5, reproducibility 1–5.
Approval needs ≥ 4 on claims-supported and reproducibility.

## Competition step
When two agents produced alternative implementations or analyses for the same task, both boards score both;
the winner merges, the loser's useful parts are listed in the session note. This is the "compete" rule.

## Paper compilation step (after approval)
The scribe (cheap model) updates `paper/sections/*.tex` for that phase: what was done, how to use it,
what to use (models, configs), with run ids. The humanizer pass (`docs/WRITING_GUIDE.md`) runs after the
scribe; the reference checker runs after the humanizer. The board B reviewer re-reads the section diff.

## Board records
Every review is posted on the agent board (`scripts/board.sh review …`) and saved under
`reviews/<phase>/<task>-<board>-<date>.md`.

## Board C — Architecture assessor (every phase from P3 on)
Role file `.claude/agents/reviewer-arch.md`. For each predictor result, the assessor must state whether a
different model family would plausibly do better for this purpose and why, drawing on current literature:
JEPA variants (I-/V-JEPA 2, JEPAWG), latent world models (Dreamer-style RSSM, TD-MPC2-style latent dynamics,
LeCun-style hierarchical JEPA world models), SSMs (Mamba-2/3, S5), test-time-training / Titans layers, xLSTM,
neural functionals and graph metanetworks (UNF, GMN), set/permutation-equivariant encoders, diffusion or
flow models over weights, and plain Transformers. Output: a ranked shortlist with expected gain, cost, and
the smallest experiment that would decide it; kept current in `docs/MODEL_CANDIDATES.md`. The orchestrator
must schedule at least the top-ranked untested candidate as a competing run before a phase is closed.
