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
the editor does not. *[Superseded 11 Nov 2026 by F9a.5(a) — see S1 in the "11 Nov 2026 (F9a reconciliation)" section below.]* A second variant picks layers by field ranking (the base-anchored sensitivity field, not the editor's
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

## 11 Nov 2026 (council round 5) amendment — paired localisation episodes, shadow LoRA stream, re-anchoring (before any G3 outcome)

Source: [synthesis](council/2026-11-11/SYNTHESIS-r5.md) Q4, Q5; PREREG A5 amendment F9.4, F9.5, F9.9; the harness requirements are in
the [H5_DESIGN section of the same date](H5_DESIGN.md). Registered arms, caps, the A5.7 criterion and F5.5 are unchanged.

**Fifth access flag and logging order.** `field_localisation_access` is logged with the Python, forward, hidden-state and feedback
flags. In every localisation episode the editor's own verbal site ranking is requested and logged **before** it sees the field's
per-tensor ranking (g⊙W, g⊙D, ‖Dx‖ from the frozen G2 artefacts). Verbal localisation is a treatment flag, compared against the
field's ranking; it never gates acceptance.

**Paired episode design (inside the G5 ≤ 10 and ≤ 6 caps).** Four paired variants share the development/held-out split, the
rebuild-from-pinned-base and rollback rules, the forward-request meter and the charged-cost accounting of every other arm:

1. scripted estimator with field localisation (the field's layers and matrices);
2. scripted estimator on the LLM's chosen layers (taken from the logged verbal ranking);
3. LLM editor with the field ranking, and the same editor without it;
4. a shuffled-localisation control: a permuted field ranking at matched ranking length, information and opportunity.

The LLM is credited with localisation value only if the LLM-localised variant beats the field-localised variant on held-out gain per
real evaluation, paired, under A5.7 accounting. *[Superseded 11 Nov 2026 by F9a.5(a) — see S4 in the "11 Nov 2026 (F9a reconciliation)" section below.]* Beating the shuffled control shows only that real field localisation carries
information. *[Superseded 11 Nov 2026 by F9a.5(a) — see S4 in the "11 Nov 2026 (F9a reconciliation)" section below.]* Results are reported per proposal stream and conditional on the information supplied; an editor-advantage claim still
has to beat the scripted and random best-of-N baselines at matched information and charged cost. Attribution patching is not run.

**Test-time LoRA shadow sub-stream.** Scope: 32 frozen development queries, one isolated base-relative LoRA each (next-token loss on
the query context; no generation; no sealed item text). Access flags: python 1, forward 1, hidden-state 0, feedback 1, plus
`backward_enabled`, `gradient_exposed_to_editor`, and the optimiser, steps, rank and module allowlists and synthesis provenance.
The critic forecasts collateral change on the registered objectives only; each forecast is logged under F9.1 before scoring; all 32
are scored on TF-v1. Read-outs: E₀ and sign agreement; savings from evaluating only above-floor forecasts, with a missed-harm upper
bound ≤ 5 % (A5.7); forecast wall time against evaluation time. *[Superseded 11 Nov 2026 by F9a.5(a) — see S2 in the "11 Nov 2026 (F9a reconciliation)" section below.]* Training cost is paid in both arms and only verification
evaluations count as savings. The sub-stream may reject an uneconomic gate; it cannot show realised savings, which remain the
prospective paired G8 claim. It is not a G8 arm.

**Re-anchoring after an accepted edit.** After any accepted H5 edit the incumbent's exact gradient is recomputed (about 43 s per row,
F6 audit) and the factors are flagged `base-factor`; the forecast log records the anchor type on every row. *[Superseded 11 Nov 2026 by F9a.5(a) — see S3 in the "11 Nov 2026 (F9a reconciliation)" section below.]* No learned transition
from the base field to the incumbent is registered.

**Generation endpoint.** No H5 math-editor claim is made from teacher-forced results; it waits for the qualified F8-E MATH-500
generation endpoint (F9.6). H5's success criterion is unchanged.


## 11 Nov 2026 (F9a reconciliation) — H5 protocol: access meters, paired localisation contrast, shadow limits, re-anchoring (before any G3 outcome)

Source: [PREREG A5 amendment F9a](PREREG_H1_TF.md) F9a.5–F9a.6 (adopted 11 Nov 2026, before any G3 outcome). F9a.5(a) requires these two clauses to be copied into the H5 documents before the pilot and marks the superseded wording of the earlier 11 Nov sections of this file. Everything above stays; superseded sentences are marked in place.

**(a) F9a.5 and F9a.6, quoted verbatim from [PREREG A5 amendment F9a](PREREG_H1_TF.md)** (`docs/PREREG_H1_TF.md`, clauses F9a.5–F9a.6, main `e3920b1`):

> **F9a.5 — H5 reconciliation (condition 5).** (a) *Documents.* The governing H5 texts are `docs/H5_DESIGN.md` "11 Nov 2026 (council
> round 5) amendment — localisation flag, shadow LoRA sub-stream, re-anchoring" and `docs/H5_PROTOCOL.md` "11 Nov 2026 (council
> round 5) amendment — paired localisation episodes, shadow LoRA stream, re-anchoring" (both added by `2ab8fab`, after Board A reviewed
> `f181f0a`), `docs/HYPOTHESIS.md` "11 Nov 2026 (council round 5) prospective note — H5 and H1-OR", and the round-4 11 Nov sections
> (F8.7), each read as amended by F8a.6 and by F9a.5–F9a.6. Those two clauses are appended verbatim with F8a.6 to both H5 documents,
> and HYPOTHESIS is reconciled, before the H5 pilot. Superseded there: "with a missed-harm upper bound ≤ 5 % (A5.7)" (replaced by
> F9a.6), "run by a pinned script with no model calls" (F8a.6(a)) and "measured about 43 s per row" (F9a.5(e)). (b) *Access and
> training meters,* pinned per episode in the G5 manifest before the pilot: `python_tool_access`, `forward_access`,
> `hidden_state_access`, `feedback_access`, `field_localisation_access`, `backward_enabled`, `gradient_exposed_to_editor`. F9.4
> episodes: `backward_enabled = 0`; `gradient_exposed_to_editor = 1` exactly when `field_localisation_access = 1` (g⊙W and g⊙D derive
> from the frozen G2 gradient; no fresh backward pass). `lora_shadow`: python 1, forward 1, hidden-state 0, feedback 1 (F9.5), where
> feedback means the next-token training loss on the query context only (no TF-v1, held-out or sealed signal reaches training);
> `backward_enabled = 1`; `gradient_exposed_to_editor = 0` (no LLM editor; only the harness optimiser consumes gradients). This
> permission covers only the harness-run training of `lora_shadow`: no LLM editor arm gains backward passes, and no other arm's
> registered access changes (A5.7's gradient-based comparators keep theirs). Meters per arm: forward requests and tokens, backward
> passes and tokens, optimiser steps, GPU-seconds and wall time. Optimiser, learning rate, step, rank and target-module limits are
> pinned in the G5 manifest before the first adapter is trained, chosen without any shadow outcome; F9a sets no values. (c)
> *Localisation sequencing and controls.* Candidate-independent inputs (per-tensor g⊙W, cached down-projection input norms) may be
> released before the first candidate; candidate-specific g⊙D exists only after that candidate's realised delta and is returned as
> metered feedback on it, never guiding its own construction. The verbal site ranking is requested and logged with its hash before any
> field ranking is released (harness-enforced) and is never a critic input. The shuffled control permutes tensor labels over the same
> ranking positions, one permutation per episode drawn before the episode from `PCG64(domain_seed("20261111:f9-shuffle:", <episode
> id>))`, at matched length, support, opportunities and cost. (d) *Paired causal contrast.* The only localisation-value contrast is
> the scripted estimator on the LLM's logged layers (variant 2) against the scripted estimator with field localisation (variant 1):
> same editor, localisation alone differs. LLM with versus without the field ranking and real versus shuffled ranking are reported
> descriptively and support no claim. Endpoint: GPE_x = Σ_e gain_{x,e} / Σ_e E_{x,e}, with gain the change in A5.7's bounded held-out
> score (percentage points) of the final incumbent over the starting incumbent and E all real evaluator calls including failed calls
> and audits (A5.7); Δ_loc = GPE₂ − GPE₁ with cluster influence u_k = (g₂,k − GPE₂·E₂,k)/ΣE₂ − (g₁,k − GPE₁·E₁,k)/ΣE₁ in A5.7's paired
> episode-cluster bootstrap-t (4,000 common draws, PCG64 seed 20261107, its failure rules). A zero or nonfinite denominator makes
> Δ_loc undefined, never a success. Localisation value is credited only if the two-sided 95 % lower bound of Δ_loc is > 0; a point
> improvement alone is insufficient; there is one primary contrast and no multiplicity correction. Aggregate sensitivity is not causal
> localisation or faithful self-report, and comparisons at unequal access stay labelled (F8a.6(a)). (e) *Increments, gradients,
> re-anchoring.* Candidates stay stored base-relative (F5.4). A forecast for an acceptance decision is relative to the actual
> incumbent: the realised increment W_cand − W_inc (bf16 states, fp32 difference) contracted with an exact incumbent gradient
> (`gradient_anchor = incumbent-exact`, fresh gradient hash) and base factors (`factor_anchor = base-factor`), or, labelled
> `approx:difference_of_base_forecasts`, f_base(W_cand − W₀) − f_base(W_inc − W₀). Re-anchoring does not turn a submitted base patch
> into an incumbent increment, and an exact gradient contraction is not an exact finite-edit forecast. All re-anchoring work (gradient
> recomputation, loads, failures) is metered and charged to the critic arm's net GPU (A5.7). The historical timings are not future
> rates: 854.718742 s over 20 rows, mean 42.736 s, with different predecessor supports and including load and save/hash work
> (`docs/reviews/council/board-a-g2-2026-11-11.md`), and 13305.825513 s over 141 donors, 94.3676 s each including I/O (power
> `fields_manifest.json` summary, per Board A); per-objective re-anchoring and factored-LoRA costs are unmeasured, measured in the
> pilot and repriced under F8a.6(f). (f) *LoRA scope and groups.* The shadow forecasts completed adapters' collateral TF-v1 change
> only, not on-query gain or whether training is worthwhile; a TF surrogate cannot supply A5.7's bounded task-score endpoint.
> Query/training contexts, support, calibration, acceptance and sealed groups are disjoint by item id and source document; query
> contexts are disjoint from every TF-v1 scored item, the G2 gradient panels (including the 11 WikiText articles and the 500 reserved
> groups, A5.3.2(iii)) and any sealed panel; overlap fails closed. (g) *Contract preserved.* All of A5.7 (arms, primary comparator,
> ratio-of-totals savings, noninferiority, guardrails, the missed-proposal guardrail with its all-rejected denominator and all-zero
> unresolved case, net GPU, prospective joint power ≥ 0.80, unresolved versus kill), the F5.4 critic contract, F8.7's per-stream
> reporting and F8a.6's retained arms and matched-information rule stand. F8a.6(f)'s drop order gains, first, `lora_shadow` as one
> unit, then the F9.4 paired set, then spec §6.6's order. G5 stays shadow-only.
>
> **F9a.6 — shadow limits (condition 6).** (a) *Harm, defined separately.* Adapter j is harmful if its WikiText contrast Δy_j = y(W₀ +
> D_j) − y(W₀) under `configs/eval_tf_v1.yaml` is resolved (A5.3 paired block interval excludes 0) and Δy_j > 0; other TF-v1 endpoints
> are descriptive; a missing or nonfinite score is `unresolved`. This is not A5.7's missed qualifying beneficial proposal, which a
> collateral forecast cannot estimate; A5.7's guardrail, denominator and unresolved cases stay unchanged. (b) *Gate frozen before
> shadow outcomes* (in the G5 manifest, before the first adapter is trained): evaluate adapter j if its zero-fit forecast a_j + q_j
> (WikiText field, `base-exact`, `base-factor`) exceeds h\*, otherwise skip; a field abstention is evaluated. One gate, never tuned on
> the 32 outcomes. (c) *All 32 scored.* Every adapter is trained and scored on TF-v1 whatever the gate says; the gate is applied
> counterfactually. (d) *Read-outs.* Missed harm M_h = #(skipped ∧ harmful)/#skipped (undefined if nothing is skipped) with its exact
> one-sided 95 % Clopper–Pearson upper bound, and #(skipped ∧ harmful)/#harmful. The report states that even 32 independent skips with
> zero misses give an upper bound of 1 − 0.05^(1/32) = 0.08937 > 0.05, and fewer skips or dependence make it worse. Replay saving
> = #skipped/32 verification evaluations, descriptive, with training cost charged in both arms; E₀ and sign agreement follow F9a.2. (e)
> *Limits.* No ≤ 5 % certification, no realised-savings claim and no G8 gate or condition comes from this pilot. The shadow may report
> a gate as uneconomic, descriptively, when measured forecast cost per adapter is at least the measured verification cost per skipped
> adapter. Training, audits and failures stay in both arms' cost accounting; A5.7's unresolved cases are retained.

Clause guide (each item is in the quoted text above): access and training meters — F9a.5(b); localisation sequencing and the shuffled control — F9a.5(c); paired causal
contrast — F9a.5(d); base-relative versus incumbent increments, exact-gradient provenance and charged re-anchoring — F9a.5(e); LoRA scope and disjoint groups — F9a.5(f);
the full A5.7 / F5.4 / F8.7 contract preserved, drop order and shadow-only G5 — F9a.5(g); missed harm defined separately from missed beneficial proposals — F9a.6(a); collateral
gate frozen before shadow outcomes — F9a.6(b); all 32 adapters scored — F9a.6(c); read-outs and the 0.08937 bound — F9a.6(d); no ≤ 5 % certification, no realised-savings claim, no G8
gate, training and audits charged in both arms — F9a.6(e).

**(b) Superseded sentences in the earlier 11 Nov sections.** The earlier text is not deleted; each sentence below is marked in place with a bracketed note. Replacement sentences
are restatements of F9a.5–F9a.6 above; the quoted clauses govern.

- **S1 (access wording; F9a.5(a), F8a.6(a))** — `docs/H5_PROTOCOL.md`, "11 Nov 2026 amendment — scripted-estimator arm, second editor, prior information", **Scripted-estimator arm**. Superseded sentence (verbatim):
  > A non-LLM proposal stream … run by a pinned script with no model calls. It receives the same localisation (layers, matrix, column budget) that the editor receives, and nothing else the editor does not.

  Replaced under F9a by:
  > "makes no LLM editor or provider calls; it makes metered diagnostic model forwards (base generations and hidden activations, spec §6.1) through the same forward-request meter, charged to the arm; S-field additionally receives the frozen G2 field as setup access" (F8a.6(a)'s replacement text, which F9a.5(a) cites). The actual access flags of every arm are pinned in the G5 manifest before the pilot (F9a.5(b)); comparisons at unequal access are labelled and descriptive.

- **S2 (missed-harm bound and savings; F9a.5(a), F9a.6)** — `docs/H5_PROTOCOL.md`, round-5 section, **Test-time LoRA shadow sub-stream**. Superseded sentence (verbatim):
  > Read-outs: E₀ and sign agreement; savings from evaluating only above-floor forecasts, with a missed-harm upper bound ≤ 5 % (A5.7); forecast wall time against evaluation time.

  Replaced under F9a by:
  > Read-outs: E₀ and sign agreement (F9a.2); replay saving = #skipped/32 verification evaluations, descriptive; missed harm M_h with its exact one-sided 95 % Clopper–Pearson upper bound and #(skipped ∧ harmful)/#harmful, harm being a resolved positive WikiText change, defined separately from A5.7's missed qualifying beneficial proposals; no ≤ 5 % certification, realised-savings claim or G8 gate (F9a.6); forecast wall time against evaluation time.

- **S3 (re-anchoring timing; F9a.5(a), F9a.5(e))** — `docs/H5_PROTOCOL.md`, round-5 section, **Re-anchoring after an accepted edit**. Superseded sentence (verbatim):
  > After any accepted H5 edit the incumbent's exact gradient is recomputed (about 43 s per row, F6 audit) and the factors are flagged `base-factor`; the forecast log records the anchor type on every row.

  Replaced under F9a by:
  > After any accepted H5 edit, an acceptance forecast is made relative to the actual incumbent: the realised increment W_cand − W_inc contracted with an exact incumbent gradient (`gradient_anchor = incumbent-exact`, fresh gradient hash) and base factors (`factor_anchor = base-factor`), or, labelled `approx:difference_of_base_forecasts`, f_base(W_cand − W₀) − f_base(W_inc − W₀); candidates stay stored base-relative (F5.4). All re-anchoring work (gradient recomputation, loads, failures) is metered and charged to the critic arm's net GPU (A5.7). No re-anchoring rate is registered; per-objective re-anchoring and factored-LoRA costs are measured in the pilot and repriced under F8a.6(f) (F9a.5(e)).

- **S4 (localisation credit; F9a.5(d), read as amended by F9a.5)** — `docs/H5_PROTOCOL.md`, round-5 section, **Paired episode design** (two sentences). Superseded sentence (verbatim):
  > The LLM is credited with localisation value only if the LLM-localised variant beats the field-localised variant on held-out gain per real evaluation, paired, under A5.7 accounting.

  Replaced under F9a by:
  > Localisation value is credited only if the two-sided 95 % lower bound of Δ_loc = GPE₂ − GPE₁ is > 0 (scripted estimator on the LLM's logged layers, variant 2, against the scripted estimator with field localisation, variant 1; A5.7's paired episode-cluster bootstrap-t, 4,000 common draws, PCG64 seed 20261107); a point improvement alone is insufficient. LLM with versus without the field ranking, and real versus shuffled ranking, are reported descriptively and support no claim (F9a.5(d)).

  Second sentence (verbatim):
  > Beating the shuffled control shows only that real field localisation carries information.

  Replaced under F9a by:
  > Real versus shuffled ranking is reported descriptively and supports no claim (F9a.5(d)).


**(c) The ≈ 43 s per row re-anchoring timing is context, not a registered rate.** The figure quoted in the earlier 11 Nov sections ("about 43 s per row") is a historical F6-audit
measurement: 854.718742 s over 20 rows, mean 42.736 s, with different predecessor supports and including load and save/hash work
(`docs/reviews/council/board-a-g2-2026-11-11.md`); the other measured contraction is 13305.825513 s over 141 donors, 94.3676 s each including I/O (power
`fields_manifest.json` summary, per Board A). Under F9a.5(e) neither is a future rate. Per-objective re-anchoring and factored-LoRA costs are unmeasured; they are
measured in the pilot and repriced under F8a.6(f). Every second of re-anchoring is metered and charged to the critic arm's net GPU; nothing in this document registers a per-row cost.

**Consistency.** The same F9a.5–F9a.6 text, the same replacement sentences S1–S4 (each listed in the document where the superseded sentence occurs) and this timing statement are appended to `docs/H5_DESIGN.md`, `docs/H5_PROTOCOL.md` and
`docs/HYPOTHESIS.md`; if any wording elsewhere in the three documents differs from the quoted clauses, the quoted clauses govern. Nothing here registers a new arm, cap,
threshold or access value, authorises an episode, or changes A5.7, F5.4, F8.7 or the H5 success criterion. This section does not itself append F8a.6; F8a.6(a)'s replacement
sentence is quoted under S1 only because F9a.5(a) cites it. G5 stays shadow-only.

## 11 Nov 2026 (F8a.6 verbatim append)

F8a.6 (A5, `docs/PREREG_H1_TF.md`) requires its own text to be appended to the H5 documents. It follows verbatim; the F9a reconciliation section above supersedes nothing in it.

**F8a.6 — H5 reconciliation (condition 6).** Spec §6 (difference-of-means inputs, LP objective, deterministic selection and
ties, α grid, S-loc/S-field, treatments, Sol pin, per-stream contract, drop order) is adopted as the H5 scripted-arm contract and
must be appended verbatim with the overrides below to `docs/H5_DESIGN.md` and `docs/H5_PROTOCOL.md`, with `docs/HYPOTHESIS.md`
reconciled, before the H5 pilot. (a) *Access wording.* "run by a pinned script with no model calls … and nothing else the editor
does not" (`docs/H5_PROTOCOL.md` 11 Nov section; same text in `docs/H5_DESIGN.md`) is replaced by: "makes no LLM editor or provider
calls; it makes metered diagnostic model forwards (base generations and hidden activations, spec §6.1) through the same
forward-request meter, charged to the arm; S-field additionally receives the frozen G2 field as setup access". The actual access
flags of every arm (spec §6.3 table, plus F9.4's `field_localisation_access`) are pinned in the G5 manifest before the pilot.
Comparisons at unequal access are labelled and descriptive; they do not meet the matched-information requirement for an
editor-superiority claim. A T0 script given |L| and B is conditional on that prior, not unrestricted discovery. (b) *Retained arms.*
Astra stays the registered primary (A5.7 primary comparison: Astra with versus without the critic); Haiku, sham reload, random
ranking and the F5.4/FP5 nested 1/8/32 best-of-N are retained with every other A5.7/G5 comparator; arms are dropped only
prospectively for funding, never by observed performance. (c) *Complete A5.7 estimator per stream*, unchanged: ratio of total real
calls including failed calls and rejected-proposal audits; both totals finite and positive at the point and in every draw;
connected episodes clustered beyond the goal × seed × budget tuple when source-, direction- or prompt-connected; paired
episode-cluster bootstrap-t with 4,000 common draws, PCG64 seed 20261107, and its failure rules; bounded score endpoint with equal
episode weighting; missed-proposal denominator of all rejected candidates including failed attempts, unauditable rejected
attempts counted as missed, zero rejected attempts undefined; net GPU saving charging diagnostics, gradients, probes, failed
attempts and amortised preparation (including the G2 fields' 1.62 GPU-h, `g2_manifest_final.json` → `caps.fields.used_hours`);
prospective joint power ≥ 0.80; unresolved versus kill distinctions. Critic and no-critic share seeds and opportunity budgets
within a stream; realised calls are observed, not forced equal. A zero or nonfinite denominator makes a stream's savings undefined,
never a success. (d) *Frozen choices.* The α-selection metric is FP5's development-winner objective (mean paired improvement on
development support relative to the actual incumbent under A5.7's feasibility/safety predicates), ties within 1e-12 relative to
the smaller α; the proposal-training, support-validation, acceptance-slice, calibration and sealed splits (spec §6.1; F5.4; FP5;
A5.7) and the actual-incumbent comparisons are frozen in the G5 manifest before calibration. (e) *F9.4.* The F9.4 paired set
(scripted with field localisation, scripted on the LLM's layers, LLM with and without the field ranking, shuffled-localisation
control) uses the same estimator and access pinning; the LLM is credited with localisation value only under F9.4's paired rule.
(f) *Cost and drop order.* Every H5 stream here is unpriced (F8a.8). The pilot is repriced inside the existing G5 caps (≤ 10 GPU-h
plus ≤ 6 for best-of-N) before launch; if it does not fit, streams are dropped in this fixed order: the F9.4 paired set as one unit
(registered last), then spec §6.6's order (S-field; T1 as one unit; Sol T0). Astra, Haiku, sham and random best-of-N are never
dropped; dropped streams are reported as unfunded. G5 stays shadow-only, G8 keeps its existing eligibility (G5 feasibility plus
G3/G4 go), and every A5.7 rejection and acceptance safeguard is retained. No episode is authorised by this block.


## 11 Nov 2026 (F8 execution spec §6, verbatim, with F8a.6 overrides)

F8a.6 (A5, `docs/PREREG_H1_TF.md`) requires spec §6 to be appended verbatim, with the overrides it lists, to the H5 documents. Source: `docs/design/F8_EXECUTION_SPEC.md` §6.1–§6.6 (subsections copied unchanged below, headings included), read at main `67642ab`. The earlier "11 Nov 2026 (F9a reconciliation)" and "11 Nov 2026 (F8a.6 verbatim append)" sections are not repeated here. Nothing here authorises an episode.

### 6.1 Difference-of-means inputs

- **Items.** The episode's *proposal-training split* of the visible pool (200 GSM8K-train + 200 MATH-train, seed 0;
  `docs/H5_PROTOCOL.md` "Adopted decisions" 4). Never support-validation, calibration, acceptance-slice or sealed items
  (F5.4; FP5 "Acceptance slices").
- **Generations.** Base completions under the pinned G5 decode settings and token cap T, thinking on, metered through the same
  forward-request meter as the editor and charged to the arm.
- **Labels.** loop = the harness's truncation flag is true (the completion reached T without terminating). non-loop =
  terminated. Labels come from the evaluator's validity/truncation fields (`docs/H5_PROTOCOL.md` "Adopted decisions" 2),
  never from correctness.
- **Minimum counts.** ≥ 8 completions per class. Otherwise the attempt is recorded as `no_direction` (a failed, charged
  attempt), the incumbent is retained, and nothing is substituted.
- **Activations.** For each layer l in scope:
  - r_l = residual stream after block l, over the last 256 generated tokens of each completion (all tokens if shorter);
  - mean within each completion, then mean over completions (equal completion weight);
  - u_l = normalize(mean_loop r_l − mean_nonloop r_l), in fp32.
- **Neuron statistics.** x_{l,j} = post-gating input of `mlp.down_proj` at neuron j over the same tokens. For column j of
  W_l = `model.layers.l.mlp.down_proj.weight`, its contribution to the u-projection at a token is p_j = (u_lᵀ W_l[:, j])·x_{l,j}.
  The column score is s_{l,j} = E_loop[p_j] − E_nonloop[p_j].

### 6.2 LP: objective, constraints, tie-breaking, edit and scale

```text
maximise    Σ_{l∈L} Σ_j s_{l,j} z_{l,j}
subject to  Σ_j z_{l,j} ≤ B_l   for each layer l ∈ L        (column budget per layer)
            0 ≤ z_{l,j} ≤ 1
solver      scipy.optimize.linprog(c = −s, A_ub, b_ub, bounds=(0,1), method="highs"), version pinned
```

- **Selection, deterministic.** For each l, take columns sorted by (−s_{l,j}, j) and keep the first min(B_l, #{s_{l,j} > 0}).
  This is an optimal vertex of the LP. Assert that its objective equals the solver's optimum within 1e-9 relative. On a
  mismatch the attempt fails closed (`lp_mismatch`). Columns with s ≤ 0 are never selected.
- **Edit.** For selected columns S_l: W'_l[:, j] = W_l[:, j] − α (u_lᵀ W_l[:, j]) u_l. Equivalently D_l = −α u_l u_lᵀ W_l S_l,
  rank one per tensor. Rebuilt from the pinned base, stored as Δ_eff = bf16(W + Δ) − W (`docs/H5_DESIGN.md` "Delta capture").
- **Scale α.** Either supplied (when the treatment supplies it), or chosen from the fixed grid (0.5, 1.0, 1.5, 2.0):
  - each grid point is one development-support evaluation, charged;
  - the winner is the argmax of measured development improvement, with ties within 1e-12 relative going to the smaller α;
  - acceptance follows F5.4/FP5 (held-out support, rebuild, untouched acceptance slice if the bytes changed).
- **Variants.**
  - **S-loc (supplied localisation):** L, matrix, B_l and α (if supplied) are exactly what the paired LLM treatment
    receives, logged as prior information.
  - **S-field (field-localised):** compute the candidate D_l for every l ∈ 0..23 at budget B and α = 1. Rank layers with the
    frozen G2 fields by r_l = a_l^{mmlu} − q_l^{mmlu}: the first-order predicted gain of the `mmlu_gold_log_prob` field
    (higher is better) minus its own curvature magnitude, both in nats per item. Take the top |L| by (−r_l, l). This ranking
    rule is a design default, chosen because MMLU is the only G2 objective tied to the reasoning task. The amendment may
    replace it before any episode, never after. |L| and B are supplied counts, logged as prior information.

### 6.3 Treatments, pairing and access flags

Each connected episode cluster k = (goal formulation × seed × budget) runs every funded arm from the same start state (the
base), the same proposal/validation/acceptance split and the same sealed panel, with frozen seeds. Critic and no-critic are
paired **within each stream**: same proposals, seeds and budget; the critic only schedules or abstains (A5.7, F5.4).

| Treatment | Arm (stream) | Information supplied (prior information) | python / forward / hidden / feedback |
|---|---|---|---|
| T0 discovery (blog protocol) | **Astra** (registered primary, reproduction; protocol unchanged) | goal + weights only | as pinned in the G5 manifest for LLM editors; identical across critic/no-critic |
| T0 | **Sol** (GPT-6.1 Sol, medium) | same prompt, tool schema and budgets as Astra | same as Astra |
| T0 | **Haiku** (cheap control, retained) | same as Astra | same as Astra |
| T0 | **S-field** | \|L\|, B (counts only) + G2 field artefact | 1 / 1 / 1 / 1 if α is tuned, else 1 / 1 / 1 / 0 |
| T0 | **Random best-of-N** (retained; FP5, nested N = 1/8/32, matched to each LLM editor's realised proposal blocks) | support/norm of the paired editor proposal | 1 / 0 / 0 / 1 (development-winner selection) |
| T0 | **Sham reload** (retained) | — | 0 / 0 / 0 / 0 |
| T1 supplied localisation | **Sol + loc** | layers 15, 19; `mlp.down_proj`; 64 columns per layer; source = published Astra edit. Exact names only if §4 PASS; otherwise the blog prose (`docs/H5_RESEARCH.md` line 10), labelled an inference | as Sol |
| T1 | **S-loc** | identical to Sol + loc | 1 / 1 / 1 / 1 (α tuned; α is not given in the prose) |
| T1 | **Random best-of-N** on the supplied support | same support | 1 / 0 / 0 / 1 |

- **Astra's reproduction treatment stays unchanged.** Astra does not run in T1 unless separately funded.
- **When access differs, the comparison is labelled.** If the LLM editors' pinned `hidden_state_access` is 0, every
  editor-versus-scripted comparison is labelled "scripted arm has hidden-state access". An editor advantage over a
  better-informed script remains reportable. An editor deficit is not attributed to the editor.
- **The G2 field is setup access.** It is logged with its measured G2 cost (fields 1.62 GPU-h,
  `g2_manifest_final.json` → `caps.fields.used_hours`) as amortised preparation, and is not charged per episode as forward
  access.

### 6.4 Sol pin

Recorded in the G5 manifest before the first episode, hashed and logged per episode:
- provider;
- model identifier exactly as the provider reports it;
- version or snapshot string;
- reasoning effort = "medium";
- decode settings;
- system prompt sha256;
- tool schema sha256;
- the provider's per-call reported model field.

If the provider offers no immutable snapshot id, the pin is the alias plus the pin date, and any per-call model field that
differs from the pinned one ends the Sol arm. The arm is reported as stopped and not replaced (`docs/H5_PROTOCOL.md` 11 Nov
"Second editor"). Voluntary early stops are logged with their reason and the requests used.

### 6.5 Per-stream savings and claims

For each stream x ∈ {Astra, Sol, Haiku, S-field, S-loc, Sol + loc, random best-of-N}, compute A5.7's contract on that stream's
own critic/no-critic pairs:
- S_x = 1 − Σ_e E_c,e / Σ_e E_0,e, with episode-cluster bootstrap-t;
- the ≥ 30 % point estimate and paired lower bound;
- the score noninferiority lower bound ≥ −1 point;
- the missed-qualifying-proposal upper bound ≤ 5 %;
- net GPU saving > 0;
- prospective joint power ≥ 0.80.

Further rules:
- **Zero denominators:** a zero or nonfinite denominator (e.g. S-loc with α supplied makes no search evaluations) makes S_x
  **undefined**, never a success. Absolute real-evaluation counts and GPU-h are reported for every stream.
- **Primary stream:** the A5.7 primary comparison stays **Astra with versus without the critic**. Other streams are
  reported. A stream's result is only called powered if its own prospective power simulation met the 0.80 bound. There is no
  retrospective best-stream selection: every funded stream is reported.
- **Editor-advantage claims** (F8.7; Board A condition 6) need, at matched treatment (T0 or T1) and labelled access: an
  episode-paired final-score difference with lower bound > 0 points against **both** the scripted stream and random
  best-of-N, with total charged cost (GPU-h, editor tokens and provider charge, reported separately) not higher.

**Retained unchanged:** G5 shadow-only status (production rejection disabled), G8 eligibility (G5 feasibility plus G3/G4
go), and all A5.7 rejection and acceptance safeguards.

### 6.6 Price

The H5 sealed scoring tier (4096 tokens) and the diagnostic-forward latency are **unmeasured**. `docs/H5_PROTOCOL.md`
"Staged plan" says they are measured before Stage 1. So every H5 item here is **unpriced**. The only measured reference rates
are the fast tier at 12.4 min/entry and the engine trial at 6.1 min/entry (`docs/RESULTS.md` "P2 — Clean fast-tier timing").
These are not the H5 tier.

The streams added here multiply the arm count. The pilot must be repriced inside the existing G5 caps (≤ 10 GPU-h + ≤ 6 for
best-of-N; `docs/council/2026-11-10/SYNTHESIS.md` §3) before launch. If the repriced plan exceeds the caps, streams are
dropped in this fixed order until it fits:
1. S-field;
2. T1 (Sol + loc together with S-loc and its random best-of-N, so T1 stays matched);
3. Sol T0.

Astra, Haiku, sham and random best-of-N are never dropped. Dropped streams are reported as unfunded.

**F8a.6 overrides of the spec §6 text above.** Quoted verbatim from F8a.6 (`docs/PREREG_H1_TF.md`); where an override and the spec text differ, the override governs.

| F8a.6 override | replaces spec text |
|---|---|
| (a) "makes no LLM editor or provider calls; it makes metered diagnostic model forwards (base generations and hidden activations, spec §6.1) through the same forward-request meter, charged to the arm; S-field additionally receives the frozen G2 field as setup access" | the earlier H5 wording "run by a pinned script with no model calls … and nothing else the editor does not" (not spec §6 text; §6.1, §6.3 already describe the metered forwards and the G2 field as setup access) |
| (a) "The actual access flags of every arm (spec §6.3 table, plus F9.4's `field_localisation_access`) are pinned in the G5 manifest before the pilot." | the fixed python / forward / hidden / feedback values of the §6.3 table, which are defaults until pinned and gain the `field_localisation_access` flag |
| (a) "Comparisons at unequal access are labelled and descriptive; they do not meet the matched-information requirement for an editor-superiority claim. A T0 script given \|L\| and B is conditional on that prior, not unrestricted discovery." | §6.3 "An editor advantage over a better-informed script remains reportable" and the §6.5 editor-advantage condition, as to matched information |
| (d) "The α-selection metric is FP5's development-winner objective (mean paired improvement on development support relative to the actual incumbent under A5.7's feasibility/safety predicates), ties within 1e-12 relative to the smaller α" | §6.2 "the winner is the argmax of measured development improvement, with ties within 1e-12 relative going to the smaller α" |
| (d) "the proposal-training, support-validation, acceptance-slice, calibration and sealed splits (spec §6.1; F5.4; FP5; A5.7) and the actual-incumbent comparisons are frozen in the G5 manifest before calibration" | §6.1 and §6.3 split statements, which are now frozen in the G5 manifest before calibration |
| (e) "The F9.4 paired set (scripted with field localisation, scripted on the LLM's layers, LLM with and without the field ranking, shuffled-localisation control) uses the same estimator and access pinning; the LLM is credited with localisation value only under F9.4's paired rule." | adds the F9.4 streams to §6.5's per-stream list; localisation value is credited only under F9.4's paired rule |
| (f) "the F9.4 paired set as one unit (registered last), then spec §6.6's order (S-field; T1 as one unit; Sol T0). Astra, Haiku, sham and random best-of-N are never dropped; dropped streams are reported as unfunded." | §6.6 "dropped in this fixed order until it fits: 1. S-field; 2. T1 …; 3. Sol T0": the F9.4 paired set goes first, then the §6.6 order |

**F9a precedence.** Where F9a.5–F9a.6 and the "11 Nov 2026 (F9a reconciliation)" section supersede a spec §6 sentence, F9a governs. One found: the §6.6 sentence "streams are dropped in this fixed order until it fits: 1. S-field; 2. T1 …; 3. Sol T0" is superseded as to order by F9a.5(g) ("F8a.6(f)'s drop order gains, first, `lora_shadow` as one unit, then the F9.4 paired set, then spec §6.6's order"); no other §6 sentence is superseded.
