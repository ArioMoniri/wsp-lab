# Council decision — 11 November 2026, round 5 (chair synthesis)

Branch `council/2026-11-11`. Inputs: [BRIEF-r5](BRIEF-r5.md); [world-model](round5-world-model.md),
[representation](round5-representation.md), [metrology](round5-metrology.md), [editor](round5-editor.md); the round-4
[SYNTHESIS](SYNTHESIS.md); appendix A5 with amendments F1–F8; Board A's F8 and F8-spec reviews
(`docs/reviews/council/board-a-f8-2026-11-11.md`, `board-a-f8-spec-2026-11-11.md`); `docs/design/F8_EXECUTION_SPEC.md`. The
registration is frozen (a638efa). This synthesis changes it only through the dated amendment F9 below, written while Part A staging
stood at 27 of 40 edits and before any G3 outcome existed. Thresholds named here are chair choices unless a file is cited.

## Verdict

The user's five threads (math-improving edits, models editing themselves while working, models reporting which parts they use,
fewer parameters at equal performance, and predicting all of it) reduce, in the council's unanimous reading, to **one structure we
already have and three things we do not yet measure**. The structure is a receiver-anchored first-order forecast: the frozen G2
gradient and Fisher factors of the pinned base, contracted with the exact algebra of each edit family (rank-one, LoRA, deletion mask,
chain increment), read out by at most three coefficients that may learn online from scored entries only, and used by the registered
H5 critic contract as the policy that decides whether an edit is worth evaluating. Isolated test-time adapters (thread 2) and deletion
masks (thread 4) are root edits of the base, so the field is exact for them without the predecessor problem F6 withdrew. "Which part
am I using" (thread 3) is the per-tensor breakdown of the same forecast, not a self-report; the strongest new evidence (OWMI) says
verbal self-reports sit at chance while probes on the same activations do not. The three things we do not measure are: whether the
forecast is **calibrated** on edits outside the zoo (no log exists yet), whether it holds at **pruning sparsities** where the
literature says performance survives, and anything on the **generation side**, where every published small-model math gain lives
and where teacher-forced endpoints are blind. F9 registers all three as descriptive extensions with no decision authority, after
Part A and the F8 panels, under F8's freeze and gate rules. Nothing in the five threads touches Part A, F5.5 or the H5 success
criterion, and no member asked for a JEPAWG-style latent model now: it would be built only if the forecast log shows structured
residuals the first-order terms cannot represent.

## 1. Decisions by question

| Question | Decision | Dissent and handling |
|---|---|---|
| Q1 One structure or four | **One interface, endpoint-specific heads.** Frozen: pinned base, G2 fields (5 objectives), projectors, probe cache, TF-v1 labels. Per edit on CPU: a = ⟨g, D⟩, q = ½tr(GDADᵀ) per tensor, κ = q/\|a\|, chart tokens. Learns online: only the A5.8.3 readout (intercept + Σa + Σq, ridge) plus abstention quantiles, fitted on scored log entries only, with IDM's swap rule (accept a recalibration only if held-out loss falls and conditional MSE does not rise). Policy: F5.4 critic contract. **Re-anchoring** replaces any learned transition: after an accepted edit the incumbent's exact gradient is recomputed (≈ 43 s per row measured in the F6 audit, `docs/reviews/council/board-a-g2-2026-11-11.md`), factors flagged `base-factor`. Thread 1 (termination repair) lies outside the teacher-forced structure. JEPAWG: a trigger, not a build; its 2026 instances map couplings to weights with no edit dynamics. | None. Representation adds that the interface must keep realised increments, predecessor displacement, tensor identity and missingness, and that the 4-dimensional chart and pooled StatNN are lossy controls; adopted. |
| Q2 Forecast-before-modify log | **F9-L.** An append-only, hash-committed JSONL row before scoring for every edit outside Part A (F8 entries, F9 panels, H5 proposals including rejected ones, test-time adapters): incumbent, realised bf16 delta and recipe hashes; anchor type (base-exact / incumbent-exact / base-factor); selection status (outcome-blind / field-selected / externally selected); a, q, κ, s* = \|a\|/q when a < 0 < q, signs at each s, quadratic-sign forecast, top-k tensor shares, chart output or abstention, G2 hashes, timestamp, decision. κ undefined at a = 0 is recorded, never regularised. **Statistics, per family and endpoint, clustered by independent source component:** primary signed-magnitude E₀ = 1 − Σ(Δy − s·a)² / ΣΔy² with the b²/CMSE split; resolved-sign agreement with exact Clopper–Pearson over clusters; calibration intercept and slope of Δy on a; coverage (fraction of non-abstaining forecasts) and the predicted-resolvable × resolved 2×2 on the F8a floor. Component bootstrap 4,000 draws, seed 20261107. **Claim thresholds:** below 30 independent clusters per family, entries only; 30–99, exploratory with exact intervals; a precision statement needs ≥ 100. A fitted κ̂ is applied, frozen, only to the next family. | Members proposed three different primaries (E₀, sign agreement, α/β). Chair: E₀ primary because a is already in nats and the log must calibrate magnitude, not only direction; the other two are mandatory secondaries. |
| Q3 Pruning as an edit family | **F9-P, descriptive, after F8.** Sparsity ladder at **10, 20 and 50 %** unstructured (10 and 20 are where arXiv 2604.25098 reports reasoning preserved; 50 where OBC-Prune reports collapse), three mask sources (Wanda from the G2 WikiText A factors on development windows; SparseGPT mask **transplanted** onto the original weights; count-matched random), each with its A5.3 rotation: 18 entries. Three native SparseGPT entries with compensation are added as a separately labelled family (D = −W⊙M + C; the contraction uses the complete D). A **ranking panel** of 24 outcome-blind Wanda masks at 20 % global sparsity with per-layer ratios from a registered seeded Dirichlet answers the PPF question with zero fitted parameters: Kendall τ of realised WikiText change against a + q, against q_A (G = I, which is pruning's own objective), and against the magnitude ⟨D, W⟩ = −‖W⊙M‖². Comparisons are **within sparsity only**; eligible-weight and total-model sparsity reported separately; unsupported tensors preserved; A5.8.1's 99 %-energy truncation checked before construction. Price: 45 entries × 1.04 min ≈ 0.78 GPU-h scoring (`docs/RESULTS.md`), construction capped at 0.5 GPU-h under stop-and-reprice; SparseGPT solve cost unmeasured. | Three members proposed 10/30/50, one 10/20/50 with the literature citation; chair takes 10/20/50 so that two rungs sit where performance is reported to hold. Metrology's 64-window resolution figures (panel half-width 0.00487 nats, per-entry maximum 0.01282, `docs/reviews/P5/tf-v1-requalification-2026-11-06.md`) are recorded as the floor for what the ladder can resolve. |
| Q4 Self-localisation for H5 | **Measured localisation in, verbal localisation as a logged treatment.** The field's per-tensor ranking (gradient saliency g⊙W, candidate-specific g⊙D, cached Dx norms) is supplied under a fifth access flag `field_localisation_access`; the editor's own verbal site ranking is logged **before** it sees the field ranking. Paired episodes: scripted estimator with field localisation; scripted estimator on the LLM's chosen layers; LLM editor with and without the field ranking; a shuffled-localisation control at matched information. The LLM adds something only if the LLM-localised variant beats the field-localised variant on held-out gain per real evaluation, paired, under A5.7 accounting. Attribution patching is a new metered artefact (clean/corrupted pairs, intermediate activations, patch sites) and is not registered now. Inside the G5 caps. | None. |
| Q5 Critic-gated test-time LoRA | **G5 shadow sub-stream, not a G8 arm.** 32 frozen development queries, one isolated base-relative LoRA each trained with next-token loss on the query context (no generation, no sealed item text), access flags (python 1, forward 1, hidden-state 0, feedback 1) plus `backward_enabled`, `gradient_exposed_to_editor`, optimiser/steps/rank/module allowlists and synthesis provenance. The critic forecasts **collateral** change on the registered objectives (it cannot forecast on-query gain); each forecast is logged under F9-L before scoring; all 32 are scored on TF-v1. Read-outs: E₀ and sign agreement; savings from evaluating only above-floor forecasts at a missed-harm upper bound ≤ 5 % (A5.7); forecast wall time measured (the only measured contraction is ≈ 94 s CPU per donor, slower than a TF-v1 entry; a factored ⟨g, BA⟩ is expected far cheaper but unmeasured). Shadow replay can **reject** an uneconomic gate; it cannot demonstrate realised savings, which stay with A5.7's prospective paired G8 design. Price ≈ 0.55 GPU-h scoring plus training capped at 0.5. | None. Editor and metrology both insist training cost is paid in both arms and only verification evaluations count as savings; adopted. |
| Q6 Generation endpoint | **F8-E-only descriptive endpoint, required before any H5 math-editor claim.** MATH-500 (all 500 items), three arms (unedited thinking-on, edited thinking-on, unedited thinking-off), fixed 8,192-token cap and decoding settings, natural-completion rate and strict correctness reported together. Screen: five paired seeds per item per arm. Qualification: two independent five-seed batches per item and arm plus a same-device same-seed replay, A3-style margin pre-declared. **Unpriced**: no comparable generation-throughput measurement exists (`docs/RESULTS.md`); the ceiling is 122.88 M generated tokens for the full qualification panel and a timed same-device pilot must precede purchase. Never alters F5.5. | None. World-model notes a rollout-anchored field (a gradient for p(end-of-thinking)) would be new, unpriced G2 work; recorded as a later hypothesis. |
| Q7 Line-stop, H5 criterion, paper | **F5.5 and A5.7 unchanged** (all four). Related-work sentences in section 3. | None. |
| Brief corrections | PPF predicts a divergence to the dense model, covers structured channel pruning, and trained on 775 evaluated masks; arXiv 2611.03244's 82 % is in-sample, 66 % out of sample; IDM is an editable program calling a frozen LLM, calibrated by online bias plus conditional MSE; arXiv 2604.25098 tests unstructured pruning at 10 % and 20 % only, no SparseGPT; Agentic-TTT's evaluation stream was deliberately enriched for TTT benefit. | — |

## 2. Spend (H0 unless stated; all under the stop-and-reprice rule; nothing waits on F9)

| Item | GPU-h | Order |
|---|---|---|
| F9-L log | 0 (CPU) | from the first F8 entry onward |
| F9-P ladder (18) + native SparseGPT (3) + ranking panel (24) | ≈ 0.78 scoring + ≤ 0.5 construction | after F8-F |
| Test-time LoRA shadow (32) | ≈ 0.55 scoring + ≤ 0.5 training | inside G5 |
| Localisation variants | inside G5 ≤ 10 + ≤ 6 caps | inside G5 |
| F8-E generation endpoint | unpriced until a timed pilot | after F8-E TF scoring |

## 3. Sentences for the measurement paper

1. "Pre-modification forecasts exist for RLHF reward edits (a first-order rule fixed before training, arXiv 2611.03244), structured
   pruning masks (a surrogate trained on evaluated masks, arXiv 2508.02381) and agent-program changes (an editable forecaster with
   online bias correction, arXiv 2610.13991). Our field is a first-order forecast of the same kind; it has no learned surrogate and
   needs no evaluated masks."
2. "Models do not verbally report interventions on their own computation, although linear probes recover them (arXiv 2608.20569);
   our localisation is a gradient–Fisher measurement, not a self-report, and verbal localisation is tested as an incremental treatment."
3. "Runtime self-editing systems train isolated per-query adapters and score them after application (arXiv 2610.12002, 2605.13369);
   they are root edits for which our field is exact, and we test the critic on them only in shadow."
4. "Magnitude and Wanda pruning at 10–20 % sparsity preserved reasoning on 7–8B models (arXiv 2604.25098); mask-only and compensated
   pruning are distinct edit families, Wanda's score is the diagonal of our curvature term, and Fisher magnitude is not signed curvature."
5. "An independent replication of the Qwen3.5-0.8B anti-loop edit measured completion separately from correctness at an 8,192-token
   budget; its paired result did not survive its pre-registered correction, and thinking-off on the unedited model scored higher than
   the edited model with thinking on."
6. "F5.5 remains unchanged; these descriptive extensions neither satisfy nor invalidate its predicates. An editor-only gain is not critic
   success."

## 4. Amendment F9 (appended to A5 today, before any G3 outcome)

F9.1 forecast-before-modify log (F9-L); F9.2 calibration statistics and claim thresholds; F9.3 sparsity ladder, native SparseGPT family
and ranking panel (F9-P); F9.4 H5 localisation flag and paired variants; F9.5 test-time LoRA shadow sub-stream; F9.6 F8-E generation
endpoint; F9.7 no decision authority; F9.8 brief corrections; F9.9 ordering and governance (F8a freeze, gate and scoring-config rules
apply). Text in `docs/PREREG_H1_TF.md`.

## 5. What this changes in the record

`docs/PREREG_H1_TF.md`: F9 appended. `docs/HYPOTHESIS.md`: H5 gains the localisation flag, the test-time LoRA shadow sub-stream and
the generation prerequisite; H1-OR gains the forecast log. `docs/H5_DESIGN.md` / `H5_PROTOCOL.md`: flags, variants, shadow stream.
`docs/RESEARCH_NOTES.md`: the round-5 items. `paper/refs/verified.yaml`: 2610.13991, 2508.02381, 2611.03244, 2610.12002, 2605.13369,
2608.20569, 2604.25098, 2609.17890 if cited. Board A reviews F9 together with the F8a adopting amendment; Part A does not wait.
