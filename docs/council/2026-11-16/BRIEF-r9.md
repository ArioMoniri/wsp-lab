# Council brief, 16 Nov 2026, round 9 (contingency research): if WM-E fails on 9B, what fixes it? Includes SSM (Mamba-style) memory structures

## The user's request (paraphrased)

If the 9B pilots fail:
- research papers and structures that would solve the problem;
- test memory-based state structures such as Mamba for the world model.

## Why this round runs now

**Timing.** This round runs before the pilot-2 result exists, so that any fix is designed without seeing it.

**Hard rules** (violating any of them invalidates your report):
- You must **not** open the `feat/wme-pilot2` branch or worktree, anything under `data/wme/` from pilot 2, or any `wme-pilot2*` report.
- You may read the pilot-1 results, which are on main.

## State (cite files)

- **Design.** `docs/council/2026-11-16/SYNTHESIS-r8.md` defines WM-E v0. It has:
  - an F3u-nested prediction;
  - a grounded latent h_t, trained to predict tap sketches, per-item losses and action derivatives;
  - a local residual step operator h_{t+1} = h_t + f(h_t, x_t, e_t);
  - exact bookkeeping supplied as input;
  - a 5-seed ensemble.

  It also defines the R0/R1/R2 "without running the edit" tiers and the claims E1/E2/E3. The code is in `src/wsp/wm/`.
- **Pilot 1 on Qwen3.5-9B** (`docs/reviews/wme/wme-v0-pilot-2026-11-16.md`, `.json`):

  | Gate | Result | Detail |
  |---|---|---|
  | Feasibility | Pass | 27.35 GB peak, 23.7 s per state, bitwise repeats |
  | Go-T | Fail, narrowly | 49.6% resolvable (58% excluding the control); static error at h = 3 of 1.91 SE (4–5 SE on WikiText and harmful refusal) |
  | Go-R | Fail | Left-point slope −4.8%; trapezoid −34.8%, descriptive only |
  | Go-E | Fail | 1/24 qualifying; no headroom on GSM8K or MMLU-Pro log-probability; 21/24 breached a guardrail |

  On the synthetic target, WM-E beat static only with a low-dimensional state, slope transport and per-step corrections.
- **Pilot 2** is pre-declared in `docs/council/2026-11-16/RULING-wme-pilot1.md`:
  - wider ladders;
  - a real sycophancy set, GSM8K, and MATH levels 4–5;
  - steering and rank-8 LoRA task vectors;
  - fallback to 4B.

  Its section 4 fixes the consequences of each outcome.
- **Earlier evidence.**
  - M3 (a JEPA latent) reached ρ 0.15/0.13; see `docs/council/2026-11-11/SYNTHESIS-r6.md`.
  - MSRM-1 (k2b fired): StatNN ≥ MSRM-1 on 0.8B; see `docs/reviews/msrm/msrm1-dev-v0-2026-11-16.md`.
- **Next step if it succeeds.** The user wants it scaled to Qwen3.8-27B on Lambda GPUs. Resources are authorised.

## Questions

1. **Failure modes and literature fixes.** For each of these, survey recent papers (2023–2026) and say what would fix it:
   - (a) the trajectory gate failing (static is good enough, or steps are too small);
   - (b) the slope-transport gate failing;
   - (c) the headroom gate failing.

   Fetch and cite every paper, and record fetch failures.
2. **SSM and memory structures for the dynamics.** This is the user's explicit ask. The candidates are:
   - Mamba, Mamba-2 and selective SSMs;
   - S4 or S5 world models (e.g. S4WM, "Facing Off World Model Backbones");
   - Mamba-based world models (e.g. Drama);
   - RWKV, xLSTM and linear-attention memories;
   - Dreamer-style RSSMs.

   For each, ask:
   - Does a recurrent or selective memory help here?
   - Here the transition is known exactly and only the response is unknown, and horizons are short (≤ 4–8 edits).
   - Where would memory actually matter: long edit sequences, order effects, saturation, normalisation interactions?

   Specify a concrete **SSM variant of WM-E**:
   - inputs and state size;
   - how the selective gating conditions on the edit;
   - nesting on F3u;
   - parameter count.

   Give a fair test against the current residual operator and against the static cumulative predictor, on identical folds, with a kill criterion. Design a synthetic task on which SSM memory **should** help and one on which it **should not**.

   Note that the target model itself is hybrid: Qwen3.5 uses Gated DeltaNet linear-attention layers. Does that suggest state features to read, such as DeltaNet state statistics?
3. **Weight-space learning architectures.** Consider:
   - permutation-equivariant metanetworks (DWSNets, NFN/NFT, graph metanetworks, Universal Neural Functionals);
   - hypernetwork and learned-optimizer forecasters;
   - "predict fine-tuning outcome from weights or gradients" papers;
   - influence functions and TRAK/datamodels for edit-effect prediction;
   - model-diff and task-arithmetic predictors.

   Which of these would give the world model better state or action encoders on a single 9B model, with a feasible compute cost?
4. **Headroom fixes.** If 9B has no headroom on its log-probability objectives, what objectives and edit generators do papers show can be *enhanced by weight editing* on modern instruct models? Consider:
   - loop repair (Sparks of WSP);
   - test-time/RL editing;
   - model merging and task arithmetic;
   - SVD or spectral edits (e.g. "LASER" rank reduction improving reasoning);
   - contrastive weight steering;
   - pruning or deletion that improves benchmarks.

   Which benchmarks have measured improvement held out?
5. **Contingency plan.** Give an ordered plan covering:
   - what to try first if (a), (b) or (c) fails;
   - the GPU-h on the local 32 GB card, and when Lambda (80 GB-class) is justified;
   - pre-declared gates for each fix.

   Every fix must be declared before the data that tests it, so that nothing becomes a post-hoc rescue.

## Rules

- **Evidence.** No invented numbers. Cite files for our data. Cite URLs you fetched for every external claim.
- **Length and format.** At most 2,200 words. Open with a 10-line BLUF. End with a table: "proposal | mechanism | test | cost | kill criterion".
- **Output.** Write only `docs/council/2026-11-16/round9-<role>.md` in this worktree.
- **Prohibited.**
  - commits;
  - GPU use;
  - model downloads;
  - writes under `data/`;
  - edits to `docs/PREREG_H1_TF.md`;
  - opening `data/evals`;
  - opening any pilot-2 material;
  - secrets, IPs or hostnames.
- **Roles.** `world-model` (Q2 primary, Q1b), `representation` (Q3 primary, Q1a), `editor` (Q4 primary, Q1c). Each answers Q5 from its own angle. The chair writes `SYNTHESIS-r9.md`.
