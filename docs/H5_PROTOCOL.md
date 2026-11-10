# H5 protocol — reproducing the blog's Experiment 8 with our own editor agents (synthesis, 4 Nov 2026)

Inputs: docs/H5_RESEARCH.md (protocol extracted from the blog), docs/H5_DESIGN.md (Board C harness design),
docs/reviews/H5/board-a-prereg-2026-11-04.md (Board A: noise qualification + pre-registration, verdict REVISE,
blocking B1–B6), docs/reviews/H5/board-b-requirements-2026-11-04.md (Board B: reporting requirements, REVISE).
This file records what the orchestrator adopts from them, what is staged, and what still needs the user.

## What the blog did (stated items only; see H5_RESEARCH.md for the stated / inferred / not-stated labels)
GPT-6 Astra, in a CPU-only Python kernel holding Qwen3.5-0.8B's weights, chose practice prompts from the GSM8K and
MATH *training* sets, received model outputs through a forward-pass harness (one pass = up to 512 answered problems
or 1,024 read texts), with budgets of 10, 25 and 50 passes, no gradients. It subtracted a "loop" direction from 64
MLP output columns in each of layers 15 and 19 (131,072 weights). Sealed scoring used 32,768 tokens, T 0.6,
top-p 0.95, top-k 20, presence 1.5, several samples per item. Loop metric = share of MATH-500 answers hitting the
token limit. Qwen: 93 → 33 % limit hits, MATH-500 7.9 → 48.1, IFEval 21.0 → 62.3 at 50 passes; 10 and 25 passes
barely moved the limit share. Gemma-3-270M-it: no fix (MATH-500 6.8 → 7.0–7.7).

## Adopted decisions
1. **Harness** = Board C's design (docs/H5_DESIGN.md): subprocess Python REPL, allowlisted read-only tensors, metered
   probe API (batch of B sequences costs B), deltas validated and stored as Δ_eff zoo entries of family `llm_edit`,
   episodes under data/runs/llm_edit/. Implementation in progress on branch feat/h5-editor (CPU tests, scripted editor).
2. **Measurement first** (Board A B1–B3, B5): no editor episode is scored until the same-device repeat panel
   (10 controls + base, running) and the cross-device panel (H100, running) are analysed with the reliability
   statistics Board A specified; eval code gains per-benchmark truncation and validity fields, immutable repeat
   runs with actual batch size and an opaque device id, and honest keyword-refusal names (branch
   fix/eval-board-a-h5, in progress). Δy is always against the same-device base.
3. **Editors**: primary Codex `gpt-6-astra` (the blog's editor identity on our provider), cheap control Claude
   Haiku 5.5, non-LLM controls: norm- and support-matched random directions, single-draw best-of-N (Meta^n note),
   sham reload. Claude Opus 5.5 as a separately registered comparison if budget allows. No silent provider fallback.
4. **Data boundary**: visible pool = 200 GSM8K-train + 200 MATH-train problems (seed 0, hashed, deduplicated
   against test); sealed set frozen by item-id hash before any episode; the editor process never mounts the repo,
   data root, caches or results; the evaluator is a separate principal; no network in the sandbox.
5. **Labels** (Board A): "reproduced" requires a qualified 32k base in the looping regime and the magnitude criteria;
   anything at 4096 tokens, restricted practice pool or another editor is at most "partially reproduced". We will
   say so in the paper. A replay of the published layer edit is a positive-control reconstruction, not rediscovery.
6. **Reporting** (Board B): per-episode table (editor, budget, seed, sealed before/after with CIs, real evaluations,
   wall clock, tokens), Figure-style items 1 and 4, all three budgets reported, negative results (Gemma, cheap editor,
   random, best-of-N) in the main text, reliability paragraph with the measured statistics, anonymity checklist.

## Staged plan and cost (measured timings: fast tier 7.5 min/entry on this GPU, 3.1 min effective on an H100 with
three workers; 4096-token and 32k tiers to be measured on the frozen sealed set before Stage 1)
- **Stage 0 — qualification (no new spend). Status 4 Nov 23:00 UTC:** same-host retest bit-identical and three-device
  control panel recorded (docs/RESULTS.md); eval fixes merged (Board A APPROVE); editor harness merged with production
  disabled (Board A APPROVE, 482 tests); remaining before a pilot: GitHub issue "H5 pilot preconditions" (separate
  principal, 200+200 visible pool, kernel policy in manifest, validate_production check, N8/N13, provider adapters).
- **Stage 1 — pilot, budget 50 passes only:** 3 goal formulations × seeds 0–2 = 9 episodes for the primary editor,
  9 for the cheap editor, 9 matched-random, 9 best-of-N, 1 sham; sealed scoring at **4096 tokens** on the frozen
  set (MATH-500 all 500 ×1 sample, GSM8K 200, IFEval 100); 32k scoring only for the base and the single best
  candidate per arm. Output: effect sizes, episode variance and the empirical power calculation Board A requires,
  plus the "partially reproduced / not reproduced" label for the pilot. Estimated GPU: the 4096-tier evaluation time
  per candidate × 37 candidates plus editor forward passes; editor tokens ≤ 100k per episode. Exact figures are
  filled in after Stage 0 measurements; nothing is claimed before.
- **Stage 2 — confirmatory (needs the user's budget decision):** Board A's full design is 30 paired episodes per arm
  per budget (10/25/50) across eight arms with 32k sealed scoring of 500 MATH ×4 + 500 GSM8K ×2 + 300 IFEval ×2 per
  candidate. That is hundreds of GPU-hours and far above the $150 Lambda cap (D4). Options: (a) raise the cap and
  run the 50-pass budget only with four arms; (b) keep the cap and report Stage 1 as a pilot with explicit
  "partially reproduced" labelling; (c) stop after Stage 1 if the pilot shows no effect above the noise floor.

## Open items for the user
- Budget for Stage 2 (options a/b/c above). Default if no answer: (b).
- Consent to use commercial editor models and to publish their transcripts (Board B licence/consent statement).


## 10 Nov 2026 amendment — critic validation and matched search (before freeze)

Authority: PREREG A5 F5.4, [10 Nov synthesis](council/2026-11-10/SYNTHESIS.md).
This prospective section supplements the earlier design; it reports no execution.
G5 remains shadow feasibility, production disabled, and A5.7 controls any G8 trial.

Record four separate Boolean flags per treatment/attempt: `python_tool_access`,
`forward_access`, `hidden_state_access`, `feedback_access`. Pin them before the
run and hold them fixed across matched critic/no-critic arms. Log actual calls,
code/config/decode hashes, feedback exposure, sequences/tokens and all compute;
Python algebra alone does not imply model inference or hidden-state access.

The loop is forecast → schedule/abstain → measure on held-out support → rebuild
from pinned base → reload/verify → accept or roll back. Separate proposal-training,
support-validation, calibration and sealed items. Every candidate remains a
base-relative patch, never a compounded patch. Preserve actual incumbent hashes
and measure its relative improvement separately. If rebuilding changes bytes,
validate the rebuilt tensor on an untouched acceptance slice. Only measured
improvement satisfying the frozen support/feasibility/safety predicates authorises
acceptance; a critic prediction does not. Failure retains the incumbent, initially
the base. Test reversibility and deterministic reload; keep all invalid, failed,
rejected, no-op and reverted attempts. Compare architecture-compatible random
permutations of updates with matched support/count/norm and equal validation/tuning
budget, separately from feature permutations. Adaptive reused support is development;
sealed feedback cannot select edits or refit a frozen critic.

Retain the existing G5 arms and add matched random best-of-N: 32 independent
Gaussian draws per source block, effective support/norm matched, fixed scales,
seeds and draw order before outcomes. Use nested prefixes 1/8/32, select one winner
per prefix on development support, then evaluate on held-out items. Match editor
opportunities at each N; report equal-real-evaluation and equal-GPU comparisons,
charging all draws, diagnostics, failed attempts and selection. No editor advantage
claim is allowed without this comparator. Single-model selection (K=1) is separate
from top-K ensembles and their multi-pass inference costs; any ensemble is separately
registered. A5.7's ≥30% realised savings, noninferiority, missed-winner, net-cost and
prospective power conditions remain unchanged. This amendment authorises no episode.

## 11 Nov 2026 amendment — scripted-estimator arm, second editor, prior information (before any G3 outcome)

Authority: PREREG A5 F8.7, [11 Nov synthesis](council/2026-11-11/SYNTHESIS.md) §§1, 4.
This prospective section supplements the 10 Nov amendment; it reports no execution, authorises no episode, and
G5 stays shadow feasibility with production disabled. Inside the existing G5 ≤ 10 + ≤ 6 GPU-h caps; reprice before launch.

**Scripted-estimator arm.** A non-LLM proposal stream: a difference-of-means direction plus linear-programme column and
scale selection (the published replications' recipe: aiXiv 261103.000004; Barry 2026), run by a pinned script with no
model calls. It receives the same localisation (layers, matrix, column budget) that the editor receives, and nothing else
the editor does not. A second variant picks layers by field ranking (the base-anchored sensitivity field, not the editor's
choice). Both variants use the same forward-request meter, development/held-out split, rebuild-from-pinned-base and
rollback rules as every other arm, and are charged for every direction estimate, LP solve, forward request and failed attempt.

**Second editor.** GPT-6.1 Sol (medium reasoning) is registered as the practical editor beside Astra, which is retained
for reproduction of the blog protocol. Pin, before the first episode: provider, model identifier and version string,
reasoning-effort setting, system prompt, tool schema and decode settings, each hashed and logged per episode. No silent
substitution or provider fallback: an unavailable, renamed or changed model ends that arm and is reported, not replaced.
Voluntary early stops (an editor ending its episode before the forward-request budget) are logged with the stop reason and
the requests used, and are not discarded or imputed.

**Prior-information logging.** Whatever an arm is told beyond the weights and the goal is recorded per attempt as
prior information, not discovery: layers or matrices supplied, column budget, edit scale, direction source, and
whether it came from a published edit or from the field ranking. An arm's recovery is reported conditional on the
information supplied; the aiXiv ≈ 93 % recovery is conditional on Astra's layers, budget and scale and is not carried over.

**Per-stream savings reporting.** Evaluation savings are reported separately for each proposal stream: LLM editors (Astra,
Sol, cheap control), the scripted estimator (both variants) and random best-of-N, each with and without the critic, at
matched information and charged cost. The ≥ 30 % paired realised-savings criterion (A5.7, F5.4) concerns the critic and is
unchanged, as are the noninferiority, missed-winner, net-cost and power conditions. Any editor-advantage claim must beat both
the scripted and the random best-of-N baselines at matched information and charged cost.
