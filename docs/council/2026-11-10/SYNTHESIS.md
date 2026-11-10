# Council decision — 10 November 2026, round 3 (chair synthesis)

Branch `council/2026-11-10`. Inputs: [BRIEF](BRIEF.md); [world-model](round3-world-model.md), [representation](round3-representation.md),
[metrology](round3-metrology.md), [editor](round3-editor.md); the 7 Nov [SYNTHESIS](../2026-11-07/SYNTHESIS.md) v3.1; appendix A5 of
`docs/PREREG_H1_TF.md` with its freeze-prep amendments F1–F4 and the pre-G2 evidence merged on 10 Nov (`docs/prereg/a5_manifests/`).
This is a decision proposal for Board A; it changes the registration only through a dated amendment (F5) written before the
freeze and before any G2 or G3 outcome exists. Thresholds named here are chair choices unless a file is cited.

## Verdict

The four members agree on the shape of the next step and disagree only on ordering and labels. **The registered programme stands:**
the base-anchored sensitivity field is the primary arm, Part A (rotation and ladder on the forty registered edits) is the decision
experiment, and WikiText is the mechanism endpoint. The news adds three things, none of which displaces the field: a published
precedent for predicting LLM accuracy from LoRA factors (LoL), which becomes a **registered secondary family of arms** with no
power to fund anything; a scale prior (Neural Thickets) that justifies a **capped 4B diagnostic**, run on Lambda in parallel so it
costs no local GPU time and no gate; and WeightLoop's validated, reversible, rebuilt-from-base update loop, which tightens the
**H5 critic contract** without changing its success criterion. The council also writes down, for the first time, the **joint rule
under which the weight-space world-model line at 0.8B is stopped** and the measurement paper becomes the final word.

## 1. What the news changes (unanimous unless stated)

| Item | Decision | Who dissents, and how it is handled |
|---|---|---|
| LoL (arXiv 2410.04207) | Register a LoL-style family as **secondary, exploratory arms** before the freeze (A5.4.3): learned-frame GL-set head, random-frame control, K-FAC-anchored frame, and a **singular-values-only control**; inputs are factors of the realised increment (truncated SVD at the recorded rank), raw adapters as a sensitivity check; LoRA-only folds and the 386-row folds; paired ratio intervals; CPU only; no funding power. Reason (representation): on LoL's own LLM table a singular-value MLP reaches R² 0.983 against 0.987 for the equivariant net, so the published result is largely a spectrum readout that StatNN already has; GL-invariant functions of (B, A) are functions of BA, which the field and chart already consume with frozen frames; a learned frame is unidentifiable on 60 labelled lora_sft rows. Metrology adds that 122 adapter rows cannot clear the 0.80 power gate (the 120-component H1w design had 0.728 / 0.600). | None. World-model and editor independently reach "secondary before freeze". |
| Neural Thickets (arXiv 2603.12228) | Low density of beneficial random edits at 0.8B does **not** imply low predictability of edit effects (world-model); it does change the scale prior. Fund a **capped scale diagnostic**: 20 outcome-blind Qwen3.5-4B Gaussian edits plus the shared 4B base, scored on the 64-window WikiText log-PPL endpoint (metrology's design, ≈ 1.8 TF-v1 GPU-h by linear size extrapolation, unmeasured). It is a diagnostic of beneficial-edit density and effect resolution at 4B, **not a gate** and not part of G7. | Ordering: metrology wants it before Part A, representation last, world-model and editor as a capped pilot. **Chair:** run it on a Lambda H100 in parallel with local G2/G3 (the user allows Lambda for parallelism), so it neither delays nor conditions Part A; TF-v1 on an H100 must be timed and device-qualified first (A3), otherwise the diagnostic reports fast-tier WikiText only. |
| WeightLoop (arXiv 2611.07319) | Tighten the H5 critic contract (editor): every accepted editor update is **rebuilt from the pinned base**, validated on held-out support items, compared with random-permutation controls, and reversible; the critic **schedules** evaluation and abstains, measured improvement **authorises** acceptance. Python, forward, hidden-state and feedback access are four logged treatment flags. The H5 success criterion (≥ 30 % realised evaluation savings, section 4.4 of the 7 Nov synthesis) is unchanged. Correction to the brief: WeightLoop's targeted edits beat random ones in 8 of 8 settings and the original predictor in 7 of 8. | None. |
| Blog critiques | A **matched random best-of-N** arm is mandatory for any editor claim at 0.8B: 32 draws per source block with nested 1 / 8 / 32 curves, single-model selection kept separate from RandOpt's top-K ensemble (editor). The unpaired 6× comparison and the Gemma failure are recorded as reasons the blog's Experiment 8 is motivation, not evidence. | None. |
| Neural-field LoRA representation (arXiv 2512.01759) | No change. | None. |
| Brief corrections | The "50,000 times faster" phrase is not in the LoL text fetched and is withdrawn. No chain LoRA factors are stored (chain steps exist only as dense snapshots); the 122 adapters on disk are 121 lora_sft plus 1 heretic_ara. | — |

## 2. The best predictor we can build now

All four members converge: **the registered tree on pooled statistics plus field summaries (augmented StatNN with field tokens) is
the best primary predictor available**, and the sharpest test of whether orientation carries information is Part A, because a
spectrum-matched rotation blinds every spectrum readout by construction. The best secondary is representation's **K-FAC-anchored
GL-set head** (about 1.6 k parameters, zero output at zero edit, L1 loss, early stopping on inner folds) taking field tokens plus
whitened curvature blocks; its expected gain over augmented StatNN is, by its proposer's own prior, inside the screen band
(0.05–0.15) on WikiText, because 0.713 of the residual error mass sits in chain rows and 0.112 in lora_sft. No member proposes a
new large encoder; the editor's critic is a small support-conditioned residual model over the same tokens.

## 3. Programme and spend (local H0 unless stated; caps under the stop-and-reprice rule)

| Order | Gate | What changes against the 7 Nov table | GPU-h |
|---|---|---|---|
| 1 | Freeze | Board A item 7 on the merged evidence plus amendment F5 (section 4) in one pass; hash list in A5.1 | 0 |
| 2 | G2 | unchanged: fields, chart, probe cache, 20-predecessor audit; measurements recorded | ≤ 3 cap |
| 3 | Power/yield simulation | unchanged (post-G2, pre-G3) | CPU |
| 4 | G3 Part A | unchanged: 120 variants plus 19 references | 2.1 + 0.33 |
| 5 | G4 Part B | unchanged primary; **adds** the LoL secondary family and the anchored GL-set head as descriptive arms | CPU |
| parallel | 4B scale diagnostic | new, Lambda H100: 20 Gaussian edits + base, WikiText 64 windows; device qualification first | ≈ 2 (Lambda) |
| 6 | G5 editor pilot | **adds** the 32-draw random best-of-N arm per block and the four access flags; WeightLoop loop in the adapter | ≤ 10 cap (+ ≤ 6 for best-of-N) |
| 7 | G6 | done (census closed 8 Nov) | — |
| 8 | G7 | conditional, unchanged | 12.5–33 |

Local total of new allocations before any G7: ≤ 5.4 (G2 + Part A) + ≤ 16 (G5 with best-of-N); the 4B diagnostic is Lambda.

## 4. Amendment F5 (to be appended to A5 before the freeze)

F5.1 Secondary LoL family on realised-increment factors: four arms (learned, random, anchored, singular-values-only); LoRA-only and
386-row folds; paired ratio intervals; descriptive; no funding; kill of this family only: learned vs singular-values-only paired upper
bound ≤ 0. F5.2 Anchored GL-set head (field tokens plus whitened K-FAC blocks) as a registered secondary with its field-only and
anchored-only comparators. F5.3 4B scale diagnostic: 20 outcome-blind Gaussian edits at a fixed relative radius plus the shared base,
WikiText 64-window log-PPL, device H100 qualified by A3 or reported as fast-tier only; a diagnostic, never a gate; stop scale
expansion if inconclusive. F5.4 H5 critic contract: rebuilt-from-base, held-out support validation, permutation controls,
reversibility; four access flags; matched random best-of-N (32 draws per block, nested 1/8/32) mandatory. F5.5 **Line-stop rule at
0.8B** (joint, all three): (a) Part A resolves ≥ 15 pairs with agreement upper bound ≤ 0.6; (b) every registered orientation
contrast in Part B has upper bound < 0.05 at simulated power ≥ 0.80; (c) the learned LoL arm does not beat the singular-values-only
arm. Any one failing to hold leaves the line **unresolved**, not killed; funding stops on a registered no-go regardless. F5.6 Brief
corrections recorded (LoL speed claim withdrawn; WeightLoop 8/8 and 7/8; no stored chain LoRA factors).

## 5. Null-size evidence: what Board A must weigh at the freeze

The registered run passes the ≤ 0.07 gate (two-sided size 0.037, ordered go rate 0.018, yield 1.0), but the pre-G2 evidence report
itself lists four caveats the chair forwards without softening: the three contrasts are rescalings of one donor block, so this is one
test, not three; the null reaches the "alternative" branch in 19.4 % of replicates, a route the gate does not cover; the LoRA low
tertile has a single donor reused by all ten of its slots; and donor means dominate the within-pair noise (SD ≈ 0.004 nats), which is
why yield is 1.0. The chair's proposal: keep the gate as registered, add to F5 that the alternative branch's null rate is reported next to
the go rate in every Part A report, and that a Part A "alternative" outcome is read as a funding stop for the field, never as evidence
of wrong signs (already the rule of 4.0 step 2).

## 6. What this changes in the record

`docs/PREREG_H1_TF.md`: appendix A5 amendment F5 (sections 4–5 above) before the freeze; `docs/HYPOTHESIS.md`: H1-OR gains the
secondary family and the line-stop rule; H5 gains the tightened critic contract; `docs/H5_DESIGN.md` / `H5_PROTOCOL.md`: the four
access flags, the rebuilt-from-base loop, the best-of-N arm; `docs/ROADMAP.md`: the 4B diagnostic as a parallel Lambda item;
`docs/RESEARCH_NOTES.md`: the six news items with URLs; `paper/refs/verified.yaml`: WeightLoop, Neural Thickets, LoL if cited.
