# Board B review, second pass — council SYNTHESIS v2 of 7 Nov 2026

Reviewer: Board B (journal / TMLR-style), read-only. Scope: `docs/council/2026-11-07/SYNTHESIS.md` (version 2; line numbers refer to it), my first pass
`board-b-synthesis-2026-11-07.md` (E1–E13), the four round-2 reports, `docs/PREREG_H1_TF.md` A4.4/A4.6, `docs/WRITING_GUIDE.md`. Nothing was run; no `data/` read;
no file other than this one was touched.

**Verdict: ACCEPT WITH REQUIRED EDITS.** Ten of my thirteen edits are resolved, three are partly resolved, none is unresolved. Version 2 fixes the structural problems
of version 1 (false agreement, reused outcomes under pass/kill language, outcome-contingent paper policy). What remains is mechanical: one missing pass criterion in the
Part B go rule (N5), one sequencing error in the A5 freeze (N3), one stale threshold list (N2), and wording. None needs a new analysis. After N1–N8 I would score
claims 4 / reproducibility 4 and recommend ACCEPT.

Scores (current text): claims-supported **4** / novelty **3** / clarity **3** / reproducibility **3** (4 once N3, N5 and N8 are applied). Approval needs ≥ 4 on claims and reproducibility.

## Resolution of E1–E13

| Edit | Status | v2 passage that answers it |
|---|---|---|
| E1 verdict claims shared diagnosis | **resolved** (one residual, N1) | L20–25: "The representation member argues … the other three treat this as a mechanism worth testing, not an established cause." The overreach left is "not another encoder" (L22–23): world-model and editor both propose a set encoder or residual head, and §6 builds one. |
| E2 access claim | **resolved** | L27–30: "A success will be reported as 'weights plus base sensitivity' or 'weights plus predecessor', never as weight reading alone. The project's original claim … stays untested." Repeated in §5 (HYPOTHESIS row) and §7. |
| E3 "one decisive purchase" vs 28 GPU-h; 0.9 extraction | **resolved** (arithmetic nit, N6) | L32–35 separates the decision experiment (Part A about 2 GPU-h, Part B CPU) from infrastructure "not conditioned on Part A"; G5 row labels the 0.9 "CPU in practice". |
| E4 §1.1 relabel and result strengths | **resolved** | Heading "What each negative tested"; row 1 "randomly framed directions … does not test base- or predecessor-anchored directions"; row 5 names the clue (checked: `metrology_budget_output.txt` l.249/280, about 94 of 300 non-null); row 7 "verified by three members in round 1", Experiment 8 nuance. |
| E5 field/chart "same object" | **resolved** | L77–81: both positions stated, world-model's "primary test" and field-only comparator kept, Board C's "competing representations" adopted. |
| E6 "corrections accepted by proposers" | **resolved** | L83–86: heading "raised by the world-model and editor members; the representation member's round 2 does not address them"; sign from ⟨g_k, ΔW⟩ alone, q_t a trust-region feature; chain cross term is stateless. |
| E7 endpoint dissent, power verdict | **resolved** | Dissent row 1 now reads "WikiText first (representation; metrology r2 …) vs MMLU decisive (world-model r2; metrology r1)"; last row records metrology's "not powered as written" and "D3 outcomes already examined". |
| E8 unsourced "eight or forty-eight runs", within-M2 ablation | **resolved** | Deleted; row 5 now "Any history ablation needs its own amendment and is not authorised here." |
| E9 gate design | **partly** | (1) reuse of seen outcomes: L119, L187–192 "exploratory development evidence … frozen by hash". (2) no kill, power condition: L186–188. (3) one estimand, 0.15/0.05 provenance: L172–182. (4) multiplicity: one primary arm, L172–174. (5) binomial 12/15 and 27/40, yield simulation, constant-sign comparator, sign predictor ⟨g, D−R⟩ only: §4.1. **Gaps:** go rule never states the Part A pass criterion (N5); the chart fallback reopens arm choice and gives the chart no destruction controls (N4); ladder rule is a de facto gate (N8); A5 freeze timing (N3). |
| E10 record table | **resolved** | §5 table: H1-OR new, H1a/H1w verbatim, A5 items (i)–(iv) incl. sealed partition `ebf0379a15d9…` kept sealed, ROADMAP appended, RESULTS, DECISIONS, H5 docs, MODEL_CANDIDATES, `verified.yaml`. Trivial omission: `docs/reviews/council/` itself. |
| E11 paper policy | **resolved** | `paper/` row: split "by evidence level, never by sign"; Parts A/B as an exploratory section with the A5 disclosure; second paper only if G7 runs and passes its own registration; limitations paragraph discloses post-hoc design. |
| E12 risks | **partly** | §7 adds reused D3 outcomes, underpowered no-go, access claim, MMLU never qualifying. The "chair choices, not measurements" statement (L9–10) lists thresholds that no longer exist in §§3–4 and contradicts §4.2 (N2). |
| E13 plain language, numbers | **partly** | Definitions block L11–16 (StatNN, descriptor-only, TF-v1, bf16, effective delta, support, θ, half-width, tripwire, census) and the two evidence tables are real improvements. Still open: undefined terms and long sentences (N7). |

## Fidelity re-check

**Round-2 reports.** Counts and quotations match: 0.1137/0.2264, 0.01282/0.9923, θ̂ 0.0071, 0.728/0.600, 242/99 vs 162, 296 = 358 − 62, 36 rows/16 roots, G5 split 6/2/2 and cap
1.0+0.9+2+2+2+2.1 = 10.0, 120 rows = 40 + 80, G7 12.5–33.3 GPU-h from 720–1,920 rows at 1.04 min, binomial 12 of 15 (tail 0.0176) and 27 of 40, P(≥15 of 40 | p = 0.375) ≈ 0.56.
Representation's four kill rules are replaced by go / screen / no-go / unresolved, and the replacement is justified (A4.4 forbids kill language on this endpoint). Two world-model
positions are not carried and not recorded: the registered-H1b MMLU Spearman ≥ 0.6 retained as an extra condition, and "superior to original *and* augmented StatNN and descriptor-only"
(v2 uses augmented StatNN only as primary). Both are defensible chair choices; both should be named as such (N2). Representation's rebuttal that the editor's ΔW·x critic "is half of the field"
is absent from the dissent table (row 4 lists only "editor r1" as the position); add it to the positions cell.

**Dissent table.** Seven rows, each with positions, decision and a "reverses it" condition. Sound. Row 4's positions cell should read "editor r1; representation r2 (critic = half the field); world-model r2 (local, not downstream)".

**Paper policy.** Now split by evidence level and independent of sign. Meets the TMLR concern from E11. One sentence is missing: where a *failed* G7 is reported (implied: paper 1, since the policy says "whatever the outcome"; say so).

**Record-change table.** Complete for the files that carry claims. A5 item (ii) discloses both θ̂ values and matches PREREG A4.6, which disclosed only MMLU 0.129 and said WikiText was not yet computed.

**Writing guide.** Evidence now in tables; no bullet lists in the decision body. Definitions at first use are mostly met; sentence length is not (N7).

## New required edits

**N1 — L22–23, "not another encoder".** World-model r2 §3 keeps "a small within-layer set encoder"; editor r2 keeps "a regularized residual head"; v2 §6 builds one. The shared position is a same-input tree first. Replace the sentence with:
> "The four members agree that the next step is a controlled intervention, and that a new encoder is not worth testing until a tree on the same inputs shows that the directions carry signal."

**N2 — L9–10, threshold list.** The list "(0.15, 0.05, 0.5, 0.6, 0.7, 15 pairs, ≥ 80 % power)" contains 0.6 and 0.7, which appear nowhere in §§3–4, omits thresholds that do (5e-3, 2e-2, 0.1 on 16 of 20, 0.30, −1 point, 5 %), and calls 0.15 and 0.05 chair choices although §4.2 says they are reused from PREREG §§4 and 7. Replace with:
> "Every threshold in sections 3 and 4 is a design choice made before any new outcome exists, not a measurement. Two are inherited from `docs/PREREG_H1_TF.md` (0.15 from §4, 0.05 from §7), here applied to a different contrast; the rest (the 12-of-15 and 27-of-40 agreement counts, the 15-pair floor, the 5e-3 and 2e-2 Frobenius tolerances, the 16-of-20 audit rule, the 0.80 simulated power, and the H5 margins) are chair choices. The world-model's extra conditions (MMLU Spearman ≥ 0.6, superiority over the original StatNN) are not inherited until MMLU qualifies."

**N3 — §4 heading and G2.** Part A's predictor ⟨g, D−R⟩ and every Part B field arm need g_k and K-FAC factors, which G2 computes *before* G3; the heading says A5 is frozen "before G3 runs", the disclosure says "before the first field is computed". Make one rule:
> "## 4. Registered decision rules (frozen by hash as PREREG appendix A5 before G2 computes any field, chart feature or probe cache; G3 and G4 do not start until the hash is committed)"

**N4 — L188–189, chart fallback and unresolved outcome.** "If the chart arm passes the same contrast and the field arm does not, the chart becomes the frozen G7 arm" lets the better of two arms through, the multiplicity problem E9.4 removed; Part A tests only the field's sign, so it says nothing about the chart; and the chart has no destruction control in the arm list (L175–178). Replace with:
> "If the field arm does not meet the go rule but the chart arm meets the same contrast (≥ 0.15 over augmented StatNN, lower bound > 0) and beats a norm-matched random-frame chart and a layer-permuted chart with lower bound > 0, the programme may buy G7 for the chart only. The chart route is reported as a second, exploratory selection among two arms; Part A is still run and reported but does not gate it. The choice is written into A5 before G7. An *unresolved* Part B buys no sealed rows and funds no extension. All six intervals are reported, and 'distinguishable from zero' is used only if all six lower bounds exceed 0 (PREREG A4.4)."
Add "random-frame chart" and "layer-permuted chart" to the descriptive arm list.

**N5 — L184–186, go rule lacks the Part A criterion.** The go rule requires "Part A is not unresolved and not sent to the alternative", so a resolved Part A with 9 of 15 sign agreements would pass. The pass condition appears only as a side remark under "Unit and interval". Replace the last clause of the go rule with:
> "…and Part A has at least 15 resolved pairs and the lower 95 % Clopper–Pearson bound of the rotation sign agreement at s = 1 exceeds 0.5 (at least 12 of 15, 27 of 40), and the ladder rule below does not send the programme to the alternative."

**N6 — L125–127, GPU sum.** 3 + 6.55 + 3 + 2.1 + 10 + 4.9 = 29.55; removing the 0.9 reserve gives 28.65, not "≈ 28". Write "≈ 29 GPU-h (29.55 less the 0.9 extraction reserve, which runs on CPU)". Also state that the 26.6 run-wall-h P4 rerun is excluded from the sum and whether it uses the GPU; if it does, the sum is wrong.

**N7 — definitions and sentence length (closes E13).** Define at first use, one clause each: K-FAC ("a Kronecker-factored estimate of the loss curvature of one linear layer", L68); Fisher (same sentence, "the curvature estimated from model-sampled gradients"); H1a / H1w (L44–45, "the MMLU and WikiText versions of the PREREG H1 contrast"); ladder (L32, "the same edit applied at 0.25 and 0.5 of its strength"); destruction control ("a copy of the feature with its direction information scrambled", L100); A1-lite ("a KL vector from TF-v1 scoring used as a non-weight comparator", L177); residual-MAE (L172, "mean absolute error left after the nuisance model"). Split the sentences over 30 words, in this order of severity: the "Terms used below" sentence (L11–16, about 90 words, make it a three-row glossary table); the Rotation paragraph (L154–162, about 80); the primary-contrast paragraph (L172–178, about 70); the G0 and G2 cells (L115, L117, about 100 each; one rule per sentence); Verdict L20–22 (38). Drop the bold on "unresolved", "Go", "Screen", "No-go" in running text or use it only in the table.

**N8 — L150–152, ladder rule.** "A lower bound < 0.5 at s = 0.25 on ≥ 15 resolved pairs … stops field purchases" fires unless at least 12 of 15 agree, so a rule called descriptive is the strictest gate in the design, and it stops purchases on an *absence of support*. Representation's own stopping rule is an *upper* bound. Replace with:
> "…reported as descriptive. If at least 15 pairs are resolved at s = 0.25 and the upper 95 % bound of the sign agreement is ≤ 0.6, the programme moves to the probe-response alternative (section 4.2) and stops field purchases; a lower bound below 0.5 alone is reported as 'not supported' and changes nothing."

## Remaining minor points (not blocking)

- G7 (L196–198) says "the registered contrast" without a number; state "≥ 0.15 over augmented StatNN, lower bound > 0" or defer it explicitly to A5.
- "Ties and missing scores count as disagreements" (L138–139) conflicts with "agreement among resolved pairs" (L142–144): a pair with a missing score cannot be resolved. Say which.
- §3 intro: say once that the 316-entry G1 figure subtracts aliases.
- `docs/reviews/council/` (these reviews) belongs in the §5 table.

## Items checked and found sound

Part A rotation construction and audit rule, the resolution rule using a paired block bootstrap over 32 two-window blocks, the exposure statement for the 40 edits, the no-kill language for Part B, the H2 ceiling and census wording, G5 and G8 arms and the editor's structural-kill condition (field-only and augmented StatNN), the 5 Nov 262-cluster figure used as the reason against a 100-block purchase. Per `docs/REVIEW_PROTOCOL.md` this review does not replace Board A on the statistical items (N5, N8, the simulated-power conditions).
