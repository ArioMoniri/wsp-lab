1. Reuse one receiver-anchored forecasting interface, with endpoint-specific contracts rather than one universal predictor.
2. Preserve Part A and F5.5; proposed F9 additions are descriptive and authorise no execution.
3. Extend forecast hashing beyond F8; report family-wise calibration, abstentions and unresolved outcomes.
4. Treat substantial pruning as an extrapolation test, not evidence of smaller deployed models.
5. Supply measured localisation to editors; test verbal localisation as an incremental treatment.
6. Register an isolated LoRA shadow arm, explicitly permitting training gradients.
7. Gate verification of completed adapters first; predicting whether to train requires another contract.
8. Compare identical proposal streams and charge training, probes, audits and critic overhead.
9. Generation endpoints are necessary for math-editor claims; TF endpoints cannot certify termination repair.
10. Preserve H5’s conjunctive savings criterion and require separate evidence of LLM-editor advantage.

# Editor, round 5 — 11 November 2026

Prospective F9 recommendations only. Authority: `docs/PREREG_H1_TF.md` A5.7/F5.4/F8.7; context: `docs/council/2026-11-11/{BRIEF-r5,SYNTHESIS,round4-editor}.md`. No new result or spending authority is asserted.

## Q1 — One interface, several contracts

Use `(receiver hash, incumbent hash, effective candidate delta, objective, support provenance, access flags)` → signed response, uncertainty, evaluation cost, schedule/abstain. Reuse frozen G2 gradients/Fisher, exact coordinates/chart and augmented StatNN; retain TF-v1 as its own measurement head (`docs/PREREG_H1_TF.md`, A5.8). Freeze receiver, predictor, calibration, endpoint and policy during evaluation; only a sandboxed proposal adapter learns online. Library or critic learning belongs to separate development episodes.

H5 already accommodates math-edit proposals, localisation-assisted construction, and validated weight deltas, including deletion. It does **not** yet establish generation accuracy, faithful self-report, compression efficiency, or live adaptation (`docs/H5_DESIGN.md`; `docs/HYPOTHESIS.md`, H5). Training-enabled adapters need explicit access/meter amendments; query-visible adaptation needs a new split contract; structured shrinking needs architecture maps and memory/latency endpoints. Unstructured zeros alone do not reduce dense parameter allocation.

JEPAWG-style latent dynamics could model nonlocal interactions and receiver drift; that is a proposed benefit, requiring held-out residual improvement over field-only and augmented StatNN, not justification for replacing them. [RSI-Env](https://arxiv.org/abs/2611.01736) supports persistent model–harness improvement; [Evo²](https://arxiv.org/abs/2610.13025) co-evolves weights and harnesses; [Exo](https://arxiv.org/abs/2611.00463) edits runtime harness policy. Their changing system state exceeds a frozen-receiver delta contract.

## Q2 — Forecast-before-modify ledger

Extend F8.3 to future F8/H5/pruning proposals, excluding Part A variants. Before scoring or deployment, hash candidate effective bytes, receiver/incumbent, objective, G2 versions, a, q, κ, first-order and quadratic signs, uncertainty, abstention and timestamp. Record undefined κ when a=0; never silently regularise it. Training completion and forecast timestamps must distinguish “before evaluation” from “before training.”

Beyond F8.3 register population, split, audit sampling, cost accounting and calibration summaries: per-family signed bias and MAE for a and a+q, calibration intercept/slope, interval coverage/width if intervals exist, resolved-sign agreement and abstention coverage. Cluster by connected source/episode; retain unresolved outcomes and report their fraction. A deterministic sign is not a probability: no invented Brier calibration. No fitting on audit outcomes. [RLHF forecasts](https://arxiv.org/abs/2611.03244) motivate freezing forecasts, but their reported horizon-dependent ranking reversal warns against treating local predictions as long-run guarantees.

## Q3 — Pruning ladder

Recommend a separately bounded descriptive extension: proposed 10/30/50% sparsity, Wanda/SparseGPT/random, frozen calibration support and mask seeds. First check tensor-factor coverage and CPU feasibility. Nine candidates plus base cost approximately **0.173 GPU-h** for TF scoring at the registered 1.04 min/entry planning rate; construction, loading and repeats are extra (`docs/PREREG_H1_TF.md`, A5.3; `docs/RESULTS.md`, TF-v1 pilot). This is a proposed panel size, not measured throughput at high sparsity.

For pure deletion, a=−Σ_M gᵢⱼWᵢⱼ; q must retain the full supported quadratic contraction. Reconstruction/compensation edits require the actual delta, not merely the mask. Rank masks by frozen a and a+q separately; compare within sparsity to avoid ranking only damage severity. Chart coordinates could add orientation/interactions, tested against identical-input StatNN and field-only. Missing factors require abstention, not substituted matrices. [PPF](https://arxiv.org/abs/2508.02381) predicts structured pruning-policy performance; it does not establish that our local field ranks large unstructured deletions. The existing tiny-mask panel is materially narrower (`docs/PREREG_H1_TF.md`, F8.2).

## Q4 — Localisation before eloquence

Require a versioned measured-localisation input for the proposed assisted-editor comparison, preserving a separately labelled unaided control. Prefer cached G2 objective saliency first; attribution patching requires additional forwards and separate costing. Averaged sensitivity is not query-specific causal localisation, and intervention detection is not identification of useful edit sites.

[OWMI](https://arxiv.org/abs/2608.20569) finds failed verbal intervention discrimination despite successful activation probes; this motivates validation, not a universal impossibility theorem. The exploratory [weight-reading audit](https://arxiv.org/abs/2611.08070) likewise cautions against treating plausible explanations as verified tracing.

Register paired episodes: identical measured localisation with a scripted estimator, with an LLM, and with that same LLM explicitly producing structured verbal site rankings before editing. Add a shuffled-localisation control at matched information and opportunities. Freeze prompts, support/count/norm limits, seeds and development selection. Primary verbal contrast: held-out final-score gain of LLM-with-report over LLM-without-report at equal total cost; editor contrast: beat both scripted and random best-of-N. Use paired episode intervals, prospective multiplicity/power planning and a predeclared meaningful effect; insufficient precision means unresolved. Site overlap is diagnostic, not success.

Log `verbal_localisation_enabled`, site rankings/confidence and supplied priors. Rationale remains audit-only for the critic. Cached-gradient provenance must disclose gradient-derived information even when editor-time backward access is off (`docs/PREREG_H1_TF.md`, A5.7/F8.7; `docs/H5_PROTOCOL.md`, 11 Nov).

## Q5 — Adapter gate: cheapest decisive path

Yes: propose an isolated, base-relative `test_time_lora` shadow stream alongside gradient-free editing. [Agentic-TTT](https://arxiv.org/abs/2610.12002) learns whether/how to adapt or reuse skills; its [execution contract](https://arxiv.org/html/2610.12002v1) creates isolated LoRAs over a frozen backbone. It already makes prospective policy decisions: do not claim it merely evaluates afterward. Our added claim is calibrated **candidate-specific response prediction**, saving verification cost.

[QueST](https://arxiv.org/abs/2605.13369) constructs query-conditioned problem–solution pairs and trains a [per-query LoRA](https://arxiv.org/html/2605.13369v2). That maps to proposal generation, not automatically to safe acceptance. Query-visible test-time adaptation conflicts with today's editor exclusion of sealed item text (`docs/H5_DESIGN.md`). Start with development-query adaptations evaluated on disjoint support; call this a QueST-inspired transfer pilot, not faithful per-query replication. A faithful arm must separately register query-text visibility, hidden answers, isolated resets, grouping and no sealed-score feedback.

Pin existing flags `(python_tool_access, forward_access, hidden_state_access, feedback_access)` as `(1,1,0,1)` for a minimal trainer with development-loss feedback and no exposed hidden-state summaries. Add `backward_enabled`, `gradient_exposed_to_editor`, optimizer/steps/rank/module allowlists, training-data access and synthesis provenance. Internal backpropagation is allowed; raw gradients need not reach the LLM. Meter training forwards/backwards separately: the no-gradient 10/25/50 probe budgets cannot silently pay for training (`docs/H5_DESIGN.md`).

The critic sees validated effective bf16 deltas and predicts endpoint change relative to the **actual incumbent**, with uncertainty and feasibility constraints. A base-gradient contraction gives a TF surrogate; it is not a math-accuracy forecast without separate calibration. Candidate/incumbent base-relative forecasts must be differenced consistently. Train privately, freeze/hash delta, forecast, schedule support evaluation, then measure acceptance; rebuild/reload from pinned base and roll back failures under F5.4. Forecasting before adapter training instead needs `(query, recipe, state)` inputs and an uncertain future-delta model: defer that separate contract.

Cheapest sequence: shadow-replay a frozen development LoRA proposal stream with full outcome audits, comparing field gate, augmented StatNN, random ranking and evaluate-all under identical proposal order and measured acceptance rules. Fit/calibrate only on disjoint episodes. Retain scripted and best-of-N controls: [self-edit strategy search](https://arxiv.org/abs/2601.14532) approaches but does not surpass its strongest human template.

Shadow replay can reject an uneconomic gate but cannot demonstrate realised savings. Advance only a feasible gate to fresh paired G8 episodes, with prespecified rejected-proposal audits charged. Training cost T is paid in both arms; if E is verification cost, net saving is `(N−K)E−critic overhead−extra audits−amortised setup`. Never count saved training after it occurred. Use measured H0 TF rate only for surrogate feasibility; price generation and adaptation independently. No supported numerical full-trial price exists: G5 remains within ≤10+≤6 GPU-h, repriced before launch (`docs/H5_PROTOCOL.md`, 11 Nov).

## Q6 — Judge the intended behaviour

Register generation-side diagnostics for F8-E now; require qualified generation scoring before any H5 math-editor success claim. Retain TF-v1 for its registered question. Natural completion alone can reward premature answers: pair completion/truncation with checked accuracy, base/edited thinking-on and thinking-off, fixed token budget, items and sampling seeds. “Every math gain is termination repair” exceeds the evidence; the local record specifically identifies that endpoint mismatch (`docs/council/2026-11-11/SYNTHESIS.md`; `docs/H5_RESEARCH.md`).

Cost is `(candidate/mode/repeat cells × measured seconds per cell)/3600`; 32k and 4096-token timings remain to be measured, not extrapolated from TF or fast-tier rates (`docs/H5_PROTOCOL.md`). Qualify same-seed execution separately from independent-sampling variance, paired shared-base errors and item uncertainty. Apply A3-style stochastic rank/ICC and effect-resolution gates on a variable control panel (`docs/reviews/H5/board-a-prereg-2026-11-04.md`); freeze completion's own meaningful-effect margin before sampling.

## Q7 — Exact sentences for adoption

“F5.5 remains unchanged; these descriptive extensions neither satisfy nor invalidate its predicates.”

“H5 success remains A5.7's paired savings lower bound ≥30%, score-difference lower bound ≥−1 percentage point, missed-qualifying-proposal upper bound ≤5%, positive fully charged net GPU saving, guardrails and prospective joint power.” (`docs/PREREG_H1_TF.md`, A5.7.)

“An editor-only gain is not critic success; verbal localisation must improve on matched probe-guided controls.”

“Prediction before modification is not unique to weights: [IDM](https://arxiv.org/abs/2610.13991) forecasts task-level outcomes of agent-program modifications; our claim concerns calibrated effective-weight-delta forecasts and verified evaluation savings.”

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---|---|
| F9 ledger | Frozen response forecasts | Proposed per-family calibration | 0 additional scoring; CPU accounting | Missing provenance blocks claim |
| Pruning ladder | Receiver-sensitive deletion | Proposed nine edits plus base | ≈0.173 scoring; construction/repeats extra | Coverage failure or unaffordable construction stops panel |
| Localisation | Probe plus verbal treatment | Proposed paired incremental contrasts | Reprice within G5 caps | Powered absence of meaningful gain rejects added value |
| LoRA gate | Forecast completed adapter | Shadow first; A5.7 live comparison | G5 ≤10+≤6; G8 unpriced | Powered savings upper bound <30% or demonstrated inferiority; otherwise unresolved |
| Generation | Completion plus correctness | Proposed F8-E; qualify for H5 | Meter before purchase | Failed qualification blocks behavioural claim |
