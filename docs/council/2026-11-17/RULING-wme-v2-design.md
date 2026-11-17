# Chair ruling on WM-E: design amendments after the outcome-blind simulation (17 Nov 2026, pre-freeze)

**Inputs.**
- `docs/reviews/wme/wme-pipeline-2026-11-17.md` (merged at 57bdd2a).
- PREREG_WME DRAFT v1.

**Status.** Only synthetic and outcome-blind simulation results exist. No bank, trajectory, enhancement or confirmatory outcome has been generated or read. These amendments go into DRAFT v2, which then requires Board A re-review.

**What the simulation showed.** Under v1, no claim was powered:
1. **H-T.** The C2-I inert control voided every alternative replicate. Null coverage was 0.81.
2. **H-E2.** Size was 0.034–0.052, against a limit of 0.035.
3. **H-E1.** The generation transfer test is limited to ≤ 250 TriviaQA groups, giving power of 0.50.

## A1. H-T: path consistency enforced by construction (fixes the C2-I failure in the model, not the gate)

**Why the gate stays.** Outcomes depend only on the realised final weights θ_T, so a round trip with realised E_T = 0 has true effect exactly 0. A predictor that outputs a non-zero value there is wrong. C2-I was right to void it, and the gate is **not** loosened.

**The new primary H-T model, WM-CP (canonical-path world model).**
- The learned residual step operator of D3 is integrated along a fixed canonical path from θ₀ to the realised cumulative edit: K(E_T) equal sub-steps of the realised oriented delta.
- K grows with path length, using a frozen rule written into the manifest.
- The prediction is therefore a function of E_T only:
  - an exact zero edit gives exactly zero;
  - alternative decompositions give the same prediction;
  - round trips give exactly zero.

**Horizon and the trajectory claim.**
- Horizon is now path length:
  - the training states are those with h ≤ 2;
  - the held-out states are h = 3–4, built from larger and composed cumulative edits.
- The trajectory claim becomes: "a learned local response field, integrated along the edit path, predicts larger and composed edits of Qwen3.5-9B better than the matched direct static map of the same cumulative edit."

**Controls.**
- C2-S and C2-D hold by construction. They are reported as such and cannot be counted as evidence.
- C2-I stays a hard gate. On never-emitted-row edits it remains a real ghost test, because E_T ≠ 0 there.

**The history-dependent WM-E v0 is kept as a descriptive arm only.**

## A2. H-T inference

The ratio bootstrap-t is replaced. The primary test is a one-sided **component-level sign-flip randomisation test**:
- The data are the component means d_c of SE-scaled paired error differences (|e_static| − |e_WM|), over held-out (state, objective) pairs.
- There are 10,000 sign flips, using a manifest seed.
- The interval is formed by inverting the test.

C_T is reported descriptively, with the same interval method.

The simulation must show size ≤ 0.035 and coverage ≥ 0.965 under the null, and power ≥ 0.80 under the alternative scenario it already uses. **The scenario is not re-tuned**, and the achieved C_T is reported.

## A3. H-E2 inference

The two-way percentile bootstrap is replaced by a **family-stratified sign-flip randomisation test** on episode-level paired differences: WM-guided against StatNN-guided test score, with 10,000 flips. It is exact under the null that paired differences are symmetric about zero.

## A4. H-E1 generation items

**Source.** The generation audit uses the registered sycophancy_real template (the answer-suggestion format of `sycophancy-eval/answer.jsonl`) applied to questions from a **pinned TriviaQA revision**:
- the rc.nocontext validation and train question texts;
- sha256-pinned;
- deduplicated by the frozen near-duplicate policy against every pilot item, every wme-c1 item and every 27B exposure.

**Size.** The generation N is set by the simulation from this pool. The template is unchanged. Only the question pool grows.

## A5. Open items carried from the track-A report (required before the freeze)

- Bring the wme-c1 panel builder up to the v1 specification.
- Re-run the exposure audit under the v1 near-duplicate policy.
- Switch the generation audit to the v1 specification.
- Make `wm.model` and WM-CP use the shared representation R(E) on 9B.
- Change the guard bounds from normal-z to the registered bootstrap-t.

## A6. Re-simulation and the outcome rule

- Re-run the cheap grid and the full-refit validation, as in ruling §4(b), under A1–A4.
- Any claim that still fails is registered **"unresolved — not powered"**.
- The E2 library stays declared as pilot-informed, because it is a near-duplicate of the pilot-2 directions (max |cos| 0.986). E2 tests that declared library.
- Caps are unchanged (RULING-wme-resources §1).
