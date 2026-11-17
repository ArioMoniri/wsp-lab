# Council brief, round 10, 17 Nov 2026: fixing slope transport and benchmark headroom; Fugu-like systems

## The user's request (paraphrased)

Research and find things *like Sakana Fugu* that could improve performance, and solve two weak results:

1. **Slope transport.** It failed in pilot 1 (−4.8%). In pilot 2 only the oracle variant, which runs the edit,
   cuts the error (by 35%).
2. **Room to improve the model.** It failed in pilot 1 (1 of 24 candidates qualified). In pilot 2 it was partial:
   sycophancy had 10 of 32 qualifying candidates, but GSM8K had only 3 of 32 and MATH 4 of 32.

## Status of this round

This round is held **after** the pilot-2 results, which you may read. Because of that, every fix you propose is
**post hoc** and must:

- be declared before the data that tests it;
- be tested on **fresh** development data, i.e. new p3 steps and new dev panels;
- never be tested on pilot-2 data.

Earlier council round 9 already declared some contingencies (`docs/council/2026-11-16/SYNTHESIS-r9.md` C4, C5, C7).
Build on them. Do not repeat them without adding something new.

## Facts (cite files)

**Target.** Qwen/Qwen3.5-9B @ c202236…, text-only, bf16, run with `enable_thinking=False`. A Qwen3.8-27B
replication is planned. The design is in `docs/council/2026-11-16/SYNTHESIS-r8.md`; the code is in `src/wsp/wm/`.

**Access tiers.** The prediction and planning claims must work at:

- **R0:** base-only artefacts plus CPU arithmetic on the edit;
- **R1:** base passes that depend on the edit direction D.

**R2** means anything at θ≠θ₀, i.e. running the edit. It is not allowed for these claims.

**Slope results** (`docs/reviews/wme/wme-pilot2-2026-11-16.md` §4):

- The trapezoid uses the slope at the end of the step, which is only available at R2.
- Pooled over objectives, it cuts the error from 2.36 SE to 1.54 SE.
- Per objective, the cut is:

  | Objective | Cut |
  |---|---|
  | harmful refusal margin | 0.79 |
  | WikiText | 0.14 |
  | harmless refusal margin | 0.13 |
  | GSM8K gold log-prob | 0.04 |
  | MMLU-Pro gold log-prob | 0.02 |
  | sycophancy | **−0.26** |

  So the pooled pass is mostly the harmful refusal margin.

- The ladders never left the WikiText trust region: the maximum ΔNLL was 0.127.

**Headroom results** (`docs/reviews/wme/wme-pilot2-2026-11-16.md` §5):

- Every objective is a **log-probability proxy**, not benchmark accuracy.
- The qualifying candidates were late-layer steering actions plus small-λ LoRAs.
- Strong LoRAs gave large gains but **breached the harmless-refusal guard**, i.e. they over-refuse.
- Under value-scale guards, GSM8K would have only 1 qualifying candidate.
- Pilot 1 (`docs/reviews/wme/wme-v0-pilot-2026-11-16.md`) found no headroom on GSM8K or MMLU-Pro log-prob.

**Current plan.** The design for the enhancement claim and the next study is in
`docs/council/2026-11-17/RULING-wme-v2-design.md` and `RULING-wme-v2b.md`. Its primary enhancement claim (H-E1)
is sycophancy, with TriviaQA generation at N=500.

**Sakana Fugu**, a learned orchestrator, is summarised in `docs/council/2026-11-16/BRIEF-r9-addendum.md` and the
round-9 pro-engineer and challenger reports. We have no API key for Fugu.

## Rules

- **No invented numbers.** Every external claim must cite a URL you actually fetched. Record fetch failures.
- **Cite repository files** for every claim about our data.
- **Length and format:** at most 2,000 words. Open with a 10-line BLUF. End with the table:
  "proposal | mechanism | access tier | test on fresh data | cost (GPU-h, $) | kill criterion".
- **Write only** `docs/council/2026-11-17/round10-<seat>.md` in your worktree.
- **Do not:**
  - commit;
  - use a GPU;
  - download models;
  - write under `data/`;
  - open `data/evals`;
  - open any sealed test split;
  - edit `docs/PREREG_*`;
  - include secrets, IPs or hostnames.
