1. **One structure, not four:** frozen G2 field, exact per-family algebra, an online readout of ≤ 3 coefficients per family, F5.4 as policy.
2. Isolated adapters (thread 2) and deletions (thread 4) are base-root edits, so the field is exact there. Thread 3 is a's per-tensor breakdown; thread 1 lies outside.
3. JEPAWG has no latent dynamics (static map z_g → z_θ). Fit a latent model only if the forecast log shows structured residuals.
4. The transition we need is exact re-anchoring at each accepted incumbent, at a measured ≈ 43 s per gradient row.
5. F9: an append-only forecast log for every edit outside Part A, hashed before scoring, with selection and anchor flags.
6. Calibration statistic: zero-parameter E₀ = 1 − Σ(Δy − a)²/ΣΔy², because a is already in nats. Reported with the bias/CMSE split and resolved-sign agreement, per family and κ-stratum.
7. Sparsity ladder: {10, 20, 50 %}, testing magnitude and ranking. q_A (G = I) is pruning's own objective, so the test is τ(a+q) against τ(q_A).
8. Test-time LoRA: a 32-adapter G5 shadow stream in which the critic forecasts collateral WikiText change. Forecast cost is unmeasured; the only measured contraction runs at ≈ 94 s CPU per donor.
9. The field ranking is the editor's localisation input, and verbal localisation is a logged flag. The generation endpoint is descriptive, F8-E only, and invisible to the field.
10. F5.5 and the H5 criterion unchanged; four paper sentences.

# World-model — round 5, 11 Nov 2026

No experiment was run. The **field** is the G2 base gradient g plus the K-FAC factors G and A (A5.8.1). In `src/wsp/features/response_field.py`, `contract` returns the signed ⟨g,D⟩ and q = ½tr(GDADᵀ) per tensor.

Sources read on 11 Nov (HTML, truncated at 39–79k characters; I use the read text only):
[2610.13991](https://arxiv.org/html/2610.13991), [2508.02381](https://arxiv.org/html/2508.02381),
[2611.03244](https://arxiv.org/html/2611.03244), [2610.12002](https://arxiv.org/html/2610.12002),
[2608.20569](https://arxiv.org/html/2608.20569), [2604.25098](https://arxiv.org/html/2604.25098),
[2611.10866](https://arxiv.org/html/2611.10866), [2605.13369](https://arxiv.org/html/2605.13369),
[2607.07127](https://arxiv.org/html/2607.07127), [2610.30847](https://arxiv.org/html/2610.30847),
[2609.17890](https://arxiv.org/html/2609.17890).

**Brief corrections.** PPF predicts JS divergence for structured channel pruning, trained on 775 evaluated masks. In 2611.03244 κ is fitted on the same runs: 0.82 is in-sample, the real upgrades give 0.66, and the price predictor also scores 20/20 signs. IDM is an editable program calling a frozen LLM, calibrated by an online bias b plus CMSE. 2604.25098 tests unstructured pruning only at 10 % and 20 %, without SparseGPT. Agentic-TTT's stream (0.256 → 0.510) was "deliberately enrich[ed]" for TTT benefit.

## Q1 — One structure

| thread | D | field reading | anchor |
|---|---|---|---|
| 1 math | −u vᵀ | a = −E[(uᵀδ)(vᵀx)], q = ½(uᵀGu)(vᵀAv) | exact, but teacher-forced: blind to termination |
| 2 runtime LoRA | BA on frozen base | a = ⟨g, BA⟩ | exact: each invocation is "an isolated LoRA" (2610.12002); QueST resets per query |
| 3 "which part" | — | per-tensor shares of a and q | localisation, not a forecast |
| 4 deletion | −W⊙M | a = −Σ_M g_ij W_ij, full mask q | exact; bf16-exact |

**Frozen.** Base `2fc0636…`; G2 fields (five objectives, 186 modules, 372 factors); projectors (dimension 4); probe cache (24 hooks, 64 prompts; `docs/reviews/council/board-a-g2-2026-11-11.md`); TF-v1 labels.

**What is computed per edit, on CPU.** a_t, q_t, κ, chart tokens, D·x.

**Learns online.** Only A5.8.3's readout (intercept + Σa + Σq, ridge 1) and abstention quantiles, fitted on scored log entries; IDM's swap rule (held-out loss falls, CMSE does not rise).

**Comparator and policy.** Augmented StatNN; F5.4. Working forecasters are frozen first-order terms (2611.03244) or surrogates over low-dimensional spaces (PPF, IDM). Ours is the first with a thin layer of the second; four predictors would need four small label sets.

**The transition is re-anchoring.** After an accepted edit the next edit is not a root edit, and F6 withdrew the cross term after one audit row with error 0.336. Exact predecessor gradients cost 854.7 s for 20 rows, ≈ 43 s per row including I/O (Board A G2 review), below one 1.04-min TF-v1 entry (`docs/RESULTS.md`). So the state is (W_inc, g(W_inc)), factors flagged `base-factor`; no learned transition is needed.

**JEPAWG.** 2607.07127's predictor maps a coupling embedding to a weight embedding: no dynamics over the parameter, no edits. 2610.30847 reads known bifurcations post hoc from ‖∂θ̂/∂p‖. Our analytic counterparts are s, κ and s* = |a|/q. A latent model adds only where a and q cannot represent the response (non-quadratic in s; interactions, cf. a joint loss "2.4 times the sum" in 2611.03244), so it is a trigger on F9 residuals, not a registration.

## Q2 — F9 forecast-before-modify log

What needs registering beyond F8.3:

1. **Scope.** Every edit outside Part A: F8 entries, pruning masks, H5 proposals and test-time adapters. Each record is written before the critic decides, so rejected attempts carry forecasts too.

2. **Record.** Incumbent, realised bf16 D and recipe hashes; **anchor type** (base-exact, incumbent-exact, base-factor q); **selection status** (outcome-blind, field-selected, externally selected; selecting on a is prediction-conditioned sampling, Board A F8 review); a_t, q_t, sums, κ, s* (only for a < 0 < q); top-k tensor shares; chart output or abstention; G2 hashes.

3. **Commitment.** Append-only JSONL; each batch's SHA-256 committed before its first scoring job. Precedent: `docs/prereg/a5_manifests/power/fields_manifest.json` (141 donors, `outcomes_read: none`).

4. **Statistic, per family, clustered by component.**
   - **Primary.** E₀ = 1 − Σ(Δy − s·a)²/ΣΔy². This is 2611.03244's uncentred explained share with κ fixed at 1, which is legitimate because a is in nats. E₀ < 0 means the forecast is worse than predicting no change.
   - **Decomposition.** MSE = b² + CMSE (IDM's pair).
   - **Secondary.** E₀ for s·a + s²·q (descriptive: q is a Gauss–Newton magnitude, A5.8.1); resolved-sign agreement (A5.3); a predicted-resolvable × resolved 2×2 on Board A condition 2's floor.
   - **Out-of-family test.** A fitted κ̂ is only applied, frozen, to the next family.
   - **Inference.** Component bootstrap, 4,000 draws, seed 20261107. No A5.5/F5.5 authority.

## Q3 — Sparsity ladder and mask ranking

**Algebra.** For D = −W⊙M and A = E[xxᵀ], Wanda's W²_ij‖X_j‖² is the diagonal of tr(DADᵀ), and SparseGPT minimises n·tr(DADᵀ) with compensation. So **q_A = ½tr(DADᵀ) is pruning's own objective**; the field adds objective-specific output curvature G and the first-order a. OBC-Prune shows objective-anchored weighting of X·diag(w)·Xᵀ matters (Wanda, 7B, 40 %: MATH500 0.862 vs 0.702 with C4).

**Ladder.** Wanda, SparseGPT and random at {10, 20, 50 %}, unstructured, each with its A5.3 rotation: 18 entries. At 10–20 % 2604.25098 finds 7–8B reasoning preserved; at 50 % OBC-Prune's 1.5B Wanda/C4 MATH500 falls from 0.802 to 0.378. Signs are trivial, so the tests are E₀ and its κ-dependence. My labelled forecast: E₀ for ŷ = a falls as sparsity rises.

**Ranking panel (the PPF question).** 24 outcome-blind Wanda masks at 20 % global sparsity, per-layer ratios from a registered seeded Dirichlet. Kendall τ of realised WikiText change against three zero-fit scores: a + q, q_A, and the chart's ⟨D,W⟩ = −‖W⊙M‖² (magnitude). Descriptive, conditional on one base; PPF needed 775 evaluations, we need none.

**Construction.** Wanda and random masks on CPU from the G2 WikiText A factors (development windows; check A5.8.1's 99 %-energy truncation first). SparseGPT solves unmeasured.

**Price.** 42 × 1.04 min ≈ 0.73 GPU-h, plus construction capped at ≤ 0.5 GPU-h.

**Chart.** Adds only the magnitude baseline: E = 0 at a base root and no fitted chart head exists (Board A F8 review). **Freeze** before G3 exposure (Board A condition 1), else exploratory.

## Q4 — Self-localisation

OWMI measures introspection (AUROC 0.5007 pooled over 11,216 dose pairs, against 75–95.8 % for a layer-16 probe). Our editor is a different model reading Qwen's weights (`docs/H5_DESIGN.md`): a reading claim, not introspection. The field's per-tensor shares apply OWMI's lesson.

Proposal: supply the field ranking under a fifth flag, `field_localisation_access`, logging the editor's own localisation before it sees the ranking. Add an LLM-localised scripted variant (F8.7's difference-of-means + LP on the LLM's chosen layers). The LLM adds something only if it beats the field-localised variant on held-out gain per real evaluation, paired, per stream, A5.7 accounting; inside G5 caps.

## Q5 — Critic-gated test-time LoRA

Neither paper forecasts: Agentic-TTT trains on ΔU measured after rollouts; QueST trains on every query, ungated. The field cannot forecast on-query gain. It can forecast **collateral change** on registered objectives, which is the input a keep/reuse decision needs.

**G5 shadow sub-stream, not a G8 arm.**
- 32 frozen development queries, one isolated LoRA each, trained with next-token loss on the query context (no generation).
- Each forecast is logged under F9 before scoring, then all 32 are scored on TF-v1.
- **Accuracy:** E₀ and resolved-sign agreement.
- **Decision:** savings from evaluating only above-floor forecasts, at a missed-harm upper bound ≤ 5 % (A5.7's guardrail).
- **Cost:** measure forecast wall time. The only measured contraction is 13,305.8 s CPU for 141 donors (`fields_manifest.json` summary), ≈ 94 s each with I/O, slower than a 62 s TF-v1 entry; a factored ⟨g,BA⟩ should be far cheaper, unmeasured.
- **Price:** 32 × 1.04 min ≈ 0.55 GPU-h for scoring, plus training capped at ≤ 0.5 GPU-h.

## Q6 — Generation endpoint

**What the field cannot see.** Termination repairs change p(end-of-thinking) on the model's own rollouts; the teacher-forced field has no gradient for it. A rollout-anchored field is new, unpriced G2 work (the WikiText field alone took 0.458 GPU-h, `g2_manifest_final.json`).

**What I support.** A descriptive F8-E-only endpoint (natural-completion rate at a fixed budget, thinking on, paired with the thinking-off base), qualified by a two-seed repeat. Pricing goes to metrology: the only measured generation timing, 745 s per fast-tier entry at 1,024 tokens, had loop rate 100 (`docs/RESULTS.md`), so that budget cannot observe completion.

## Q7 — Line-stop and paper

F5.5 and the ≥ 30 % H5 criterion unchanged; no thread tests a registered predicate. Sentences:

1. "Pre-modification forecasts exist for RLHF reward edits (first-order rule fixed before training, arXiv 2611.03244), structured pruning masks (surrogate trained on evaluated masks, arXiv 2508.02381) and agent-program changes (editable forecaster with online bias correction, arXiv 2610.13991). Our field is a first-order forecast for weight edits of one receiver, in the endpoint's units, with no fitted coefficient in its sign test."
2. "Models do not verbally report interventions on their own computation, although linear probes recover them (arXiv 2608.20569). Our localisation is a gradient–Fisher measurement, not a self-report."
3. "Runtime self-editing systems train isolated per-query adapters and score them after application (arXiv 2610.12002, 2605.13369). We test no runtime editing."
4. "Magnitude and Wanda pruning at 10–20 % sparsity preserved reasoning on 7–8B models (arXiv 2604.25098). These saliencies are special cases of our curvature term, tested only as a descriptive family."

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---|---|
| One structure | Exact family algebra at a root anchor; ≤ 3-coefficient online readout | E₀ per family on F9 | 0 | E₀ ≤ 0 in every family with ≥ 2 components |
| Re-anchoring | Exact incumbent gradient ≈ 43 s per row | E₀, base- vs incumbent-anchored | ≈ 0.012 per edit per objective | No E₀ gain → keep the base anchor |
| F9 log | Hash before scoring | E₀, b²/CMSE, signs, 2×2, κ strata | 0 (CPU) | descriptive |
| Sparsity ladder + ranking | q ⊃ q_A; g and G add objective anchoring | E₀ by sparsity; τ(a+q) vs τ(q_A) vs magnitude | ≈ 0.73 + ≤ 0.5 | τ(a+q) ≤ τ(q_A) → drop deletion from the critic |
| JEPAWG residual | Latent fit on field residuals | trigger only | 0 | Residuals unstructured → never build |
| LLM-localised variant | Probe vs verbal localisation | Gain per evaluation vs field-localised | G5 caps | Not better → stays a flag |
| Test-time LoRA shadow | Root edits; collateral forecast | Missed-harm UB ≤ 5 %, savings, forecast cost | ≈ 0.55 + ≤ 0.5 | UB > 5 % at ≥ 30 % savings, or forecast cost ≥ evaluation cost |
| Generation endpoint | TF field blind to termination | Two-seed repeat | metrology | Repeat fails |
| Paper sentences | Scope | Above | 0 | — |
