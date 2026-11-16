# World-model — round 7, 16 Nov 2026

No experiment, GPU, `data/` access or outcome read. *Computed* = arithmetic from cited figures.

## BLUF

1. WSR's +0.152 (27B self, 18 labels) is exploratory: single runs, uncorrected, and bootstrap ignores shared dose parents. In 3 of 4 subtle cells the weights-minus-text interval includes zero.
2. Part of the gain is informational. Only the weights arm could decode token IDs. The authors also find that whether an edit acts is "largely predictable from its description".
3. An LLM weight reader is **not** a model-specific world model. Nothing in it is fitted to the target's outcomes, an outside model read every target best, and no reader noticed Muse's post-norm cancellation.
4. For our NLL endpoint, a WSR reader without our tools has *less* model-specific information than the field, which uses this base's gradient.
5. F12 failed by construction. The expected C_WM,S (0.045) sat below its own 0.05 point threshold. Ridge bias makes the stress gate worse as N grows.
6. **Develop MSRM-1**, a field-nested learned response model: (c) with (b)'s statistics as inputs. Develop it on the 270 non-Part-A roots, freeze it, then test it once on fresh independent roots against an unpenalised field readout.
7. Alternative: **R-audit**, a descriptive WSR-style LLM reader against a scripted reader, decoded-text and name-blind arms.
8. The simulation must pass before any purchase. A crude √K extrapolation (*computed*) puts 0.80 power for one learned-vs-field contrast near 590 components.
9. Cost: ≤ 2.55 GPU-h of optional development labels, and ≤ 10.4 GPU-h for a 600-root confirmatory panel. Confirmatory runs only after Part A, by the user's decision.
10. No change to F8, F9, H5 or F12-I. Flag: A5.7's "StatNN **and** field-only" clause repeats F12's conjunction.

**Fetched (16 Nov).**
- https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models: abstract and overview. Full text came from that page's PDF asset; the alphaXiv PDF viewer returned only a loading shell.
- https://github.com/suayptalha/wsr, cloned at `b5251bf`.
- https://www.alphaxiv.org/abs/2611.08070, with full text from https://pdfs.assets.alphaxiv.org/2611.08070v1.pdf.

**Brief correction.** The count is 689 TF-v1 records, not 694 (F12.10(a)).

## Q1 — What WSR establishes

**From the code:**
- The weights arm gets the tensors and `tokenizer.json` but no description (`wsr/read.py:session_prompt`).
- The text arm gets templates such as "Row(s) [248069] of `lm_head` multiplied by …" (`wsr/ops.py:apply_ops`).
- The authors concede: "only those readers can decode token IDs … Part of the weight-access gain … may therefore come from knowing which token was edited" (§4.4).
- Banks are outcome-selected (`wsr/select.py:select_ids`).
- The only non-LLM baselines are constant or random answers (`harness.run_baseline`). **There is no scripted-statistics reader.**

**Priors:**
- Change-detection AUROC has a median of 0.76 with weights and 0.73 from text. In the authors' words: "what weight access adds is mainly the direction of the change" (§8).
- On Muse, all 40 edits were inert. Readers still predicted change for 85%, 88% and 53% of them with weights, against 72%, 68% and 47% from text, so weight access *raised* false alarms. No reader checked how a scaled tensor enters the next norm (§6.1).
- This matches Ahmed's "local arithmetic plus dataset priors". After label permutation, the reader found the right split at the new indices, then named all nine class pairs wrongly (Table 2, §6.2).

**Labelling vs our floor:**
- WSR labels a change by item-level Welch p < 0.01 on single sampled runs, which the authors call "optimistic" (§8), plus a minimum effect.
- At base rate 0.5, p < 0.01 alone needs a shift of about 0.141 on 100 MMLU-Pro items and 0.176 on 64 math items against 5-seed pooled bases (*computed*: 2.576·√(.25/n + .25/5n)). The 0.05 minimum does not bind.
- A "same" label can therefore hide a 0.10 shift. Some specificity loss charged to weight readers may be real sub-threshold change.
- Our TF-v1 endpoint is deterministic, continuous NLL. Its panel half-width is 0.00487 nats (`docs/reviews/P5/tf-v1-requalification-2026-11-06.md`) and its floor h* is 0.0049397 nats/token (F8a.2; `floor_recompute_2026-11-11.json`). That is far finer than WSR's labels and not three-way.

## Q2 — Relation to the thesis

Round 6 defined the model-specific world model as f_θ trained on this base's edit–outcome pairs (`SYNTHESIS-r6.md` Q2). A WSR reader is a **generic transformer prior, instantiated per target by tool calls**. Like the field, it is model-specific in its inputs, not its parameters.

WSR's own evidence:
- "a more capable outside reader was worth more than being the target's own checkpoint" (§6.2).
- Self-framing had no effect (−0.026 [−0.100, +0.049]).
- The crossed self contrast includes zero in both rounds (Table 3).

**Information ordering for our endpoint:**
1. Family or text prior.
2. Scripted weight statistics, run either by an LLM or by a fixed script.
3. The field (a, q): zero fitting, but it uses this base's gradient and K-FAC, which no forward-free reader can compute.
4. Field plus a learned residual: F12's WM-R, and MSRM-1.
5. Forward evaluation.

StatNN is level 2 plus supervision. The H5 critic (A5.7) consumes a predictor and judges it by realised savings.

The missing scripted baseline is decisive: without it, "reasons over weights" is indistinguishable from "computed norms and decoded a token".

## Q3 — Transfer

Technically trivial: the 0.8B base and deltas fit a CPU sandbox.

**Labels.** Up, down or same at h*, plus sign and a magnitude bin. Report the label marginal first, with constant-answer and family-only floors.

**Data.**
- Only exposed non-Part-A development roots, accessed through the F12 input path with F12a.1(c) attestations.
- No Part A component, no `confirmatory` or `fresh_unselected` entry, and nothing under `data/evals` for Part A.

**Fixes to WSR's design:**
- Both arms get decoded tokens and relative norms.
- Add a name-blind arm with opaque tensor IDs. This is Ahmed's permutation analogue for role and layer priors.
- Run every cell twice.

## Q4 — Why F12 was underpowered

Figures are from `board-a-f12-sim-report-2026-11-14.md`.

1. **Threshold.**
   - Under the alternative, C_WM,S was 0.045, because the tree already captured C_S:F3 = 0.132 of the oracle's 0.15. h* was a random function of WM-R's own class, which a tree learns easily.
   - Detection needs a point C_WM,S ≥ 0.05. As N grows, the estimate concentrates below that, so power tends to **zero**. No N rescues this conjunction.
2. **Ridge bias.**
   - Under stress, a linear signal with no correction still gives a target C_WM of 0.0188 > 0: learned heads undo F3's ridge-1 shrinkage.
   - Not a world-model effect, yet the size gate worsens with N (stress marginal 156/1000).
   - Fix: three unpenalised coefficients.
3. **Coverage.**
   - Conditional bootstrap-t covered the target in 570/1000 runs, against 905/1000 for the joint percentile interval.
   - A frozen predictor on fresh rows keeps refits out of the resampling.
4. **Size.**
   - The single-contrast WM:F3 power was 424/1000. With μ = 1.96 + Φ⁻¹(0.424) = 1.768 and μ ∝ √K, power 0.80 needs about 587 components and 0.90 about 786 (*computed*; not simulated; ignores the stress and coverage gates).
   - The 689 records sit in 126 components (`SYNTHESIS-r6.md` Q3), and chain steps are below the floor. The extra units must therefore be new roots.
5. **Sign targets.** Roots are predicted far above the floor: Part A LoRA median predicted |a| is 0.228 nats (F12.1, predictions only). Sign labels would mostly reward family priors.

## Q5 — Recommended: MSRM-1

**Model.** ŷ(D) = F3u(a, q) + h_θ(x(D)) − h_θ(x(0)).
- **F3u** is the A5.8.3 readout (β₀ + β₁Σa + β₂Σq) fitted by unpenalised least squares.
- **x(D)** combines:
  - F12.3's 42-column per-tensor tokens (a_t, q_t, K frames, role, probe D·x columns 39–41);
  - the v4 pooled statistics StatNN uses;
  - three scripted WSR-style columns per tensor: ‖D_t‖/‖W_t‖, top-1 singular-energy share of D_t, and |cos| of D_t's leading output vector with the norm-scaled unembedding mean.

  All are computed on CPU.
- **h_θ** is chosen once in development, by grouped inner CV, from: the F5-pinned K+F head (1,185 parameters); boosted trees ({100, 300, 600}); or Gaussian kernel ridge. Architecture is a development choice, not a confirmatory rival.
- **Training:** L1 on F3u residuals (F5.1 recipe, seeds 0/1/2).
- **Target:** continuous TF-v1 Δy (nats). A three-way label at h* is reported for WSR comparison.

**Development (exploratory, A4.4 wording):**
- **Rows:** the 123 existing non-Part-A roots (89 components, sha `f96e4e…`). Optionally add the 147 development roots (sha `d31b96…`; 2.55 GPU-h at 1.04 min each, `part_a/manifest.json`), bought as development rows under a new amendment by the user's decision. F12's no-purchase ruling binds only F12.
- **Tokens:** the F12 token matrix for these 270 rows already exists.
- **Folds:** `20261111:f12:` component folds.
- **Exclusions:** as F12.2, plus all non-root chain steps.
- **Baselines (same folds):**
  1. F3u.
  2. Zero-fit a.
  3. StatNN on identical inputs.
  4. A **description-only** learned model (family, dose, role, layer, rank; no weight statistics). This is the text-only analogue.
  5. A **scripted reader** (scripted columns and norms only, same h class).
  6. Name-blind MSRM.

**Confirmatory.** Only by a new dated amendment, after the Part A ledger and the F8/F9 order (F5.5), and only by the user's decision.
- Freeze MSRM-1 and F3u fitted on all development rows.
- Score a fresh panel of **independent training-free roots**, one component each, generated on CPU in development-covered families: abliteration-type and random-direction roots, plus dose ladders crossing s* and 2–9-tensor combinations, which are the field's blind spots.
- One contrast: C = 1 − Σ|e_MSRM| / Σ|e_F3u|, with a component bootstrap of paired errors and no refits.
- "Detected" requires C ≥ 0.05 and a lower bound > 0. StatNN, the description-only model and the scripted reader are descriptive.

**Simulation before purchase.** Adapt `scripts/f12_simulation.py`:
- **Inputs:** real development and fresh-panel tokens. Fresh-panel tokens cost about 94 s per entry (13,305.8 s / 141, `fields_manifest`) and need no outcomes.
- **Data-generating process:** Δy = F3u-true + γh* + ε, with ε from the F8a.2 payload. h* is drawn from both a random head and a random tree.
- **Stress scenarios:** purely linear (must give C ≈ 0); and family offset plus linear (measures prior-only gain).
- **Gates:** size ≤ 0.07; power ≥ 0.80 at oracle C = 0.10; lower-tail coverage within 0.07.
- **Panel size:** N is the smallest of {150, 300, 450, 600} that passes. If none passes, nothing is bought.
- **Pricing:** a 10-replicate pilot. F12's full run cost 102.01 CPU-h.

The grid, the C value and the cap are chair choices.

**Cost.**
- Development: CPU only (+ ≤ 2.55 GPU-h optional).
- Simulation: CPU, pilot-priced.
- Confirmatory at N = 600: 10.4 GPU-h scoring plus 15.7 CPU-h fields (*computed*).
- LLM tokens: zero.

**Kill criteria.**
- k1: development C(MSRM:F3u) < 0.05, or upper bound < 0.10.
- k2: no development margin over the description-only model, meaning the gain is prior.
- k3: the simulation fails at N ≤ 600.
- k4: confirmatory upper bound < 0.05 at power ≥ 0.80 (bounds this class and population only).

**Alternative — R-audit (descriptive, development-only).**
- **Rows:** 40 exposed non-Part-A roots, stratified by family, in 4 batches of 10.
- **LLM arms:** decoded text with norms; weights + tools; weights + tools + field tool (a_t, q_t); name-blind weights.
- **Scripted reader:** trained on the other 230 rows.
- **Runs:** two repeats per arm, 32 sessions of ≤ 1.5 h each.
- **Tokens:** unmeasured. A pilot session measures them and the user caps them.
- **Kill:** neither weights arm beats the scripted reader by ≥ 0.05 balanced accuracy. This makes (d) conditional on (a) beating (b).

## Q6 — Registered plans

- **No change** to F8, F9, F12-I or H5.
- **Flag only.** A5.7's rejection clause ("upper bound < 0.05 against augmented StatNN **and** field-only") has F12's conjunctive shape. It should be re-simulated, and amended if needed, before any G8 episode count is fixed.

| proposal | mechanism | registered test | cost | kill criterion |
|---|---|---|---|---|
| MSRM-1 (recommended) | Unpenalised field readout plus a learned residual over field tokens, StatNN statistics and scripted WSR-style columns | New post-Part-A amendment: frozen predictor vs F3u on fresh independent roots; one contrast; C ≥ 0.05 and LB > 0 | Dev CPU (+ ≤ 2.55 GPU-h); sim CPU (pilot-priced); ≤ 10.4 GPU-h + 15.7 CPU-h at N = 600 | k1 dev C < 0.05 or UB < 0.10; k2 no margin over description-only; k3 sim fails at N ≤ 600; k4 UB < 0.05 at power ≥ 0.80 |
| R-audit (alternative) | LLM agent over 0.8B deltas, with and without field tools, vs scripted, decoded-text and name-blind arms | None (descriptive) | 0 GPU; 32 sessions ≤ 1.5 h; tokens pilot-measured | Weights arms < scripted reader + 0.05 balanced accuracy |
| F12 (status) | Cross-fit two-comparator design | Unresolved: not powered or qualified | Closed, 102.01 CPU-h | Killed by its gates |
| F8/F9/H5/F12-I | Unchanged | As registered | As registered | As registered; A5.7 conjunction flagged |
