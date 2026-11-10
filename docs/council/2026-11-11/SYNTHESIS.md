# Council decision — 11 November 2026, round 4 (chair synthesis)

Branch `council/2026-11-11`. Inputs: [BRIEF](BRIEF.md); [world-model](round4-world-model.md),
[representation](round4-representation.md), [metrology](round4-metrology.md), [editor](round4-editor.md); the round-3
[SYNTHESIS](../2026-11-10/SYNTHESIS.md); appendix A5 of `docs/PREREG_H1_TF.md` with amendments F1–F7; Board A's F7 acceptance
(`docs/reviews/council/board-a-f7-2026-11-11.md`). The registration is frozen (a638efa). This synthesis changes it only through the
dated amendment F8 below, written before any G3 outcome exists (Part A staging at 20 of 40 edits, nothing scored). Thresholds named
here are chair choices unless a file is cited.

## Verdict

The four members agree, without dissent on substance, that **nothing published since 10 November touches the registered programme**:
the base-anchored field remains the primary arm, Part A (rotation and ladder) remains the decision experiment, its spectrum-matched
rotation control is at least as strong as the norm-matched orthogonal control the best new replication used, and the joint line-stop
rule F5.5 stays verbatim. The news adds two **descriptive panels** that the programme did not have and could not have built a week
ago: real, outcome-validated weight edits of our exact base by other groups, and three cheap edit families the zoo lacks. Both enter
as amendment F8, outside A5.5 and F5.5, with every prediction frozen before scoring and with a CPU pre-check that refuses to buy GPU
time for edits the field itself predicts are unresolvable. The news also settles two H5 questions: the pilot gains a scripted
estimator arm (difference of means plus a linear programme, no LLM) and a second editor model, and keeps its success criterion.

## 1. What the news changes

| Item | Decision | Dissent and handling |
|---|---|---|
| WSP-Scope (2611.09382) and the Astra edit as a **sign test on our base** (Q1) | Register an **external-edit panel** (F8-E), descriptive. Entries: the released Astra two-tensor patch (exact replay, hash-pinned against the pinned base revision; fails closed if any other tensor differs); Barry's released LP edits if their bytes or a complete recipe are published; the aiXiv difference-of-means recipe re-estimated on frozen development inputs, labelled as a reconstruction. **WSP-Scope is excluded** until its direction u is released: the recipe (token-embedding word sets weighted by final-norm gains, per the world-model's reading) does not determine u. All loop-family entries count as **one component** (aiXiv's direction has cosine ≈ 0.78 to Astra's; Barry reuses Astra's direction), so the panel has n ≈ 2–3 independent components, not 4–8. | None on registration. Metrology and world-model both show that even 8/8 agreeing signs give a CP lower bound of only 0.63, so the panel **cannot** be an inference and is not one. Editor asks that "outcome-blind" be stated as blind to *our* endpoints, not to the published gains; adopted. |
| What a loop edit means on TF endpoints | Expected outcome **unresolved**: the known effect is a thinking-on generation effect (loop termination) and TF-v1 scores fixed sequences with thinking off (`configs/eval_tf_v1_wikitext.yaml`). A resolved sign hit is a transfer precondition for the H5 critic, not Part A evidence; a resolved miss marks a locality or K-FAC failure; unresolved says nothing. None of the three touches A5.5 or F5.5. | None. |
| Rane (2611.07610), weights vs activations (Q2) | **Prior only, no effect-size revision.** Rane's white-box arm is a 48-dimensional spectral summary (block norms, depth profile, eight singular values), the same kind of readout as our descriptor-only control; its clustered CI for weights (0.194–0.639) includes chance; its target is domain identity, not signed response. Our field ⟨g, D⟩ is a loss-weighted linear activation probe of the edit, which sits between Rane's white-box and gray-box arms. The access-ladder report (chart-only, field-only, hybrid, activation-assisted) states this prior in one sentence (section 4). | None. Representation and editor add that the field must be described as *weights plus frozen receiver sensitivity*, never as data-free; adopted in the paper sentence. |
| Deletion Tuning (2611.10866), SWEET (2611.10974), TFTV (2609.09054) as **new families** (Q3) | Register a **family-transfer panel** (F8-F), descriptive: Wanda-initialised gate deletion at < 0.05 % and a count-matched random gate deletion; TFTV rank-one on o_proj; a Concept-Country-style down_proj rank-one from a non-scored split; each at ladder s ∈ {0.25, 0.5, 1} plus its registered rotation. **SWEET deferred** (needs generated counterfactuals; unpriced). The field's algebra is exact for each family (rank-one: a = −E[(uᵀδ)(vᵀx)], q = ½(uᵀGu)(vᵀAv); deletion: a = −Σ_M g_ij W_ij, q a full quadratic form over the mask with no rank-one shortcut). A learned Deletion-Tuning mask is a separate unpriced purchase and is not in F8. | World-model requires a **CPU pre-check**: compute a, q and κ = q/\|a\| for every candidate first; buy scoring only if at least four components have \|a\| + q above the floor (median paired-interval half-width of the registered null-size payload). Adopted as the F8 gate: it is what makes the panel beat the closed 4B diagnostic per GPU-hour. Representation warns that at a base root the chart's displacement products vanish and that direction can fall in the projector's null space; the chart prediction is therefore reported but does not gate. |
| Behavioural rate (2611.06598), receiver-relative features (Q4) | **No new arm.** Base anchoring already carries the receiver (g, G, A are the receiver's; q is the edit's energy in its curvature). The paper's statistic is constant across edits for one base and one objective, so it has no within-receiver variance; a per-edit source-data Fisher needs training-data access that chain and external edits lack. G2 stores averaged gradients and model-sampled Fisher factors, not per-item gradients, so the correction covariance cannot be recovered from existing artefacts (representation). The "compression recovers a corrupted adapter" result is a field prediction (spectral ladder ⟨g, D_k − D⟩ + q), testable after the G3 ledger closes on non-pick LoRA rows at ≤ 0.5 GPU-h; recorded as a post-G3 option, not registered now. | None. |
| Concept Country control; Part A controls (Q5) | **No change, no flaw found** (all four). The norm-matched orthogonal control is the published analogue of A5.3's Haar rotation, which also preserves spectrum, rank and support; at 1,024 dimensions a Haar direction's cosine to the original has SD ≈ 1/32. Nothing from its pre-registration practice needs to enter A5 (A5 already freezes hashes, selection, scoring, controls and inference). | Two members could not fetch concept.country or nathan.rs (HTTP 403) and treat those numbers as brief-reported; the chair read both pages on 11 Nov. The reliance is on the control's *design*, which the brief states, not on its numbers. |
| IWIBench (2611.08647), scripted replications; **H5** (Q5) | H5 pilot adds a **scripted-estimator arm**: difference-of-means direction plus LP column/scale selection, no LLM, given the same localisation the editor receives, and a second variant that picks layers by field ranking; information supplied to any arm (layers, budget, scale) is logged as prior information, not discovery. Editor-model comparison: **GPT-6.1 Sol (medium)** registered as the practical editor next to Astra, which is retained for reproduction; no silent substitution; voluntary early stops logged. The ≥ 30 % paired evaluation-savings criterion (A5.7, F5.4) concerns the critic and **stays**; savings are reported per proposal stream (LLM, scripted, random best-of-N), and any editor-advantage claim must beat both baselines at matched information and charged cost. | None. |
| Line-stop rule and the paper (Q6) | **F5.5 unchanged.** F8 states that its panels cannot satisfy, fail or unresolve any F5.5 predicate or A5.5 step. Related-work sentences in section 3. | None. |
| Brief corrections | WSP-Scope names "long, unresolved reasoning", not loops, and derives u from token embeddings, not activations; TFTV's edit is s̄ vᵀ with v built from W's singular vectors and the expected input, rank one and norm-matched; Deletion Tuning tests no Qwen3.5 backbone; the aiXiv 93 % figure is conditional on Astra's layers, budget and scale. | — |

## 2. Spend

| Item | GPU-h (H0) | Status |
|---|---|---|
| F8-E external panel scoring (≤ 3 components × ≤ 4 entries + base) | ≤ 0.3 at the measured 1.04 min per TF-v1 entry (`docs/RESULTS.md`) | after the CPU pre-check; construction CPU only |
| F8-F family panel scoring (≤ 6 components × 4 entries) | ≤ 0.6 | after the CPU pre-check; construction capped at 0.5 GPU-h, stop-and-reprice |
| H5 scripted arm, Sol comparison | inside the existing G5 ≤ 10 + ≤ 6 caps; reprice before launch | G5 unchanged otherwise |
| Everything registered (Part A 2.1 + 0.33, Part B CPU) | unchanged | Part A `score` after staging; nothing waits on F8 |

F8 panels never delay Part A: they are scored after the Part A ledger is sealed, in the order Part A → F8-E → F8-F, so that no
external outcome is read before the registered one.

## 3. Sentences for the measurement paper (related work and limitations)

1. "Concurrent work edits the same base, Qwen3.5-0.8B, with column-sparse or rank-one changes that raise generation accuracy
   substantially (Hu et al. 2026, arXiv 2611.09382; Takehi et al. 2026); scripted difference-of-means and linear-programming
   estimators recover most of that gain once layers and budget are supplied (aiXiv 261103.000004; Barry 2026). We predict the
   teacher-forced response of a fixed zoo; we do not ask whether useful edits exist."
2. "Published weight-space edits demonstrate endpoint-specific behavioural interventions, including loop removal, but do not by
   themselves establish held-out prediction of benchmark change from a weight edit."
3. "On the same base, spectral weight-delta statistics identified a LoRA's training domain less well than activation or output
   features (Rane 2026, arXiv 2611.07610); that ordering is context, not an effect-size prior for our field, which contracts the
   delta with frozen base gradients and Fisher factors and is therefore weights plus receiver sensitivity, not a data-free readout."
4. "Deletion-only post-training (Lu et al. 2026, arXiv 2611.10866), activation-derived suppression edits (Fu et al. 2026,
   arXiv 2611.10974) and forward-statistic rank-one task vectors (Perin et al. 2026, arXiv 2609.09054) broaden the space of edits;
   they evaluate different targets and do not validate our registered predictions, which is why we test them only as a descriptive
   family-transfer panel."
5. "A pre-registered norm-matched orthogonal control on Qwen3.8-27B (Concept Country 2026) is the published analogue of our
   spectrum-matched rotation; it is a complementary control, not a calibration of ours."
6. "Useful fine-tuning information is receiver-dependent (David et al. 2026, arXiv 2611.06598); our field is anchored to one receiver
   and makes no cross-base claim."
7. "Our line-stop rule concerns the tested 0.8B representations and endpoints, not the possibility of weight editing in general."

## 4. Amendment F8 (appended to A5 today, before any G3 outcome)

F8.1 **External-edit panel (F8-E), descriptive.** Entries: the released Astra patch (exact replay; two dense deltas as one base-root
entry; hash-pinned; fails closed on any other tensor difference), published LP edits with complete bytes or recipe, and the aiXiv
difference-of-means reconstruction on frozen development inputs (labelled reconstruction). Source clusters are declared before
scoring; the loop family is one component. WSP-Scope enters only if u is released. F8.2 **Family-transfer panel (F8-F), descriptive.**
Wanda gate deletion (< 0.05 %), count-matched random gate deletion, TFTV rank-one on o_proj, Concept-Country-style down_proj rank-one
from a non-scored split; each with the registered rotation and ladder s ∈ {0.25, 0.5, 1}; SWEET deferred; learned masks excluded.
F8.3 **Frozen predictions.** Before any scoring: realised bf16 delta hashes, the A5.3.3 effective-rank audit, sign(a) at each s,
sign⟨g, D − R⟩, q, κ = q/|a|, the quadratic-sign forecast (flip at s* = |a|/q, descriptive), the chart prediction, abstentions, and a
minimum surviving-entry fraction after bf16 rounding (zero realised change counts as disagreement, A5.3). F8.4 **CPU pre-check gate.**
Scoring is bought only if at least four components have |a| + q above the floor (median paired-interval half-width of the registered
null-size payload); otherwise the panel is reported unscored. F8.5 **Endpoints and order.** H0 TF-v1 with the common base reference;
WikiText decides for the panel, MMLU descriptive; scored after the Part A ledger is sealed, Part A → F8-E → F8-F; every entry
reported, including unresolved. F8.6 **No decision authority.** The panels cannot satisfy, fail or unresolve any F5.5 predicate or
A5.5 step and fund nothing. F8.7 **H5.** Scripted-estimator arm (difference of means + LP, no LLM; a second variant chooses layers by
field ranking), supplied information logged as prior; GPT-6.1 Sol (medium) as the practical editor beside Astra; ≥ 30 % savings
criterion unchanged, reported per proposal stream; editor-advantage claims must beat the scripted and random best-of-N baselines.
F8.8 **Access-ladder prior.** The access-ladder report states Rane's ordering as a prior in one sentence. F8.9 **Brief corrections**
recorded (section 1, last row). F8.10 **Post-G3 option, not registered:** compression-recovery truncation test on non-pick LoRA rows,
≤ 0.5 GPU-h.

## 5. What this changes in the record

`docs/PREREG_H1_TF.md`: amendment F8 (section 4) appended under a dated heading; no earlier text rewritten. `docs/HYPOTHESIS.md`:
H5 gains the scripted-estimator arm and the second editor. `docs/H5_DESIGN.md` / `H5_PROTOCOL.md`: the scripted arm, Sol, per-stream
savings. `docs/RESEARCH_NOTES.md`: the eleven new items with URLs. `paper/refs/verified.yaml`: 2611.09382, 2611.07610, 2611.10866,
2611.06598, 2611.08647, 2611.10974, 2609.09054 if cited. Board A reviews F8 before any F8 entry is scored; Part A does not wait.
