# Round 9 addendum (16 Nov 2026): pro-engineer and challenger-engineer seats; Sakana Fugu as a source of ideas

BRIEF-r9.md binds both new seats.

**Hard rules.**
- Never open the `feat/wme-pilot2` branch or worktree, the pilot-2 data under `data/wme`, or any `wme-pilot2` report.
- Every external claim must cite a URL you fetched.
- Every claim about our data must cite a repository file.
- Don't invent numbers.
- At most 2,000 words, a 10-line BLUF, and a closing table: "proposal | mechanism | test | cost | kill criterion".

## The user's request (paraphrased)

The user asked us to look at Sakana Fugu (https://sakana.ai/fugu/) and research it for three things:
- possible enhancements to the world model's predictions;
- orchestration and harness tips;
- "brilliant ideas" from pro and challenger engineers on the council.

## What the chair verified on Fugu (re-fetch and cite what you rely on)

**What Fugu is.** It is Sakana's learned orchestrator ("A Multi-Agent System, Delivered as One Model"). Technical report: arXiv 2606.21228 (https://arxiv.org/abs/2606.21228). There are two main variants:

- **Fugu.**
  - A pretrained LM backbone with a selection head that picks one worker LLM per input.
  - Only the singular values of selected matrices are fine-tuned; the orthogonal parts are fixed.
  - Stage 1 is SFT toward soft targets: each worker is run n times, and its mean rewards are passed through a softmax at temperature τ, trained with a KL loss.
  - Stage 2 is sep-CMA-ES evolution on end-to-end agentic reward (0/1).
- **Fugu-Ultra.**
  - Builds on the Conductor (https://arxiv.org/abs/2512.04388).
  - An RL-trained model (GRPO) writes a natural-language workflow of up to 5 steps. Each step specifies a subtask, a worker id and an *access list* of prior outputs.
  - Isolation through access lists prevents "orchestration collapse".
  - Workers share persistent memory.
  - Reward is 0 if the workflow fails to parse, 0.5 if it parses but the answer is wrong, and 1 if correct.

TRINITY (https://arxiv.org/abs/2512.04695) uses Thinker, Worker and Verifier roles. It reports sep-CMA-ES beating RL, imitation learning and random search under high dimensionality and tight budgets.

Fugu itself is an API product (OpenAI-compatible; `fugu-ultra-v2.0`; also on OpenRouter). It edits no weights. We currently have no key for it.

## Seats

### pro-engineer

You are a senior ML-systems and research engineer. Propose concrete, buildable improvements in three areas.

**(a) WM-E prediction quality** (`src/wsp/wm/`). Candidates to assess:
- soft and repeated-seed targets in the style of Fugu Stage 1;
- tuning singular values only (SVF) as a cheap state or action encoder, or as an edit family;
- sep-CMA-ES as the enhancement planner versus CEM, Thompson sampling or beam search under a tight real-evaluation budget;
- two-stage training (cheap proxy labels, then expensive end-to-end labels);
- active learning to choose which trajectories to label;
- ensembles and calibration.

**(b) The research harness and orchestration that runs this programme.** That is, a chair delegating to council members and implementers across providers. Consider:
- Conductor-style workflows with access lists;
- isolation between independent proposers;
- verifier roles;
- partial-credit gating;
- shared persistent memory;
- how to cut wall-clock time and wasted GPU, given the incidents in `docs/PREREG_H1_TF.md` notes F12-S1/F12-S2 and the pilot-1 cap breach in `docs/reviews/wme/wme-v0-pilot-2026-11-16.md`.

**(c) Engineering for the 27B scale-up on Lambda.**
- Multi-GPU sharding for gradient and K-FAC collection.
- Checkpoint and edit I/O.
- Spot/availability risk.
- Cost controls, including auto-terminate.

Every proposal comes with a test and a kill criterion.

### challenger-engineer

You are an adversarial red-team engineer. Your job is to break WM-E and the plan.

- Find the strongest reasons the trajectory, enhancement and WSR-comparison claims could be wrong, trivial or artefactual. Examples:
  - leakage between bank, dev and test;
  - the static baseline being too weak or too strong;
  - log-probability objectives that don't transfer to generation;
  - guardrail gaming;
  - selection effects in "fresh roots";
  - the world model winning only through bookkeeping features.
- Propose the cheapest decisive experiments that would expose each problem.
- Propose at least two radically different alternatives that a skeptic would expect to win at equal budget, for example:
  - no world model at all, with best-of-N direct evaluation;
  - an activation-steering-only planner;
  - an LLM orchestrator, Fugu-style, choosing edits by reading the weights.
- Say what result would convince you.

Address Fugu as well: is a Fugu-style orchestrator of LLM readers a stronger competitor to WM-E than a single WSR reader? How would it enter the E2 contrast fairly?

## Outputs and prohibitions

- **Write only:**
  - `docs/council/2026-11-16/round9-pro-engineer.md`
  - `docs/council/2026-11-16/round9-challenger.md`
- **Prohibited:**
  - commits;
  - GPU use;
  - model downloads;
  - writes under `data/`;
  - opening `data/evals`;
  - opening any pilot-2 material;
  - edits to `docs/PREREG_H1_TF.md`;
  - secrets, IPs or hostnames.
