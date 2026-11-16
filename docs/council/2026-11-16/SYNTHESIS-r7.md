# Council decision, 16 November 2026, round 7 (chair synthesis): Weight-Space Reading, and the next model to develop

- **Branch:** `council/2026-11-16`.
- **Inputs:** [BRIEF-r7](BRIEF-r7.md) and four member reports:
  - [world-model](round7-world-model.md): Claude Opus 5.5, high effort.
  - [representation](round7-representation.md): Codex GPT-6.1 Sol, high reasoning.
  - [metrology](round7-metrology.md): GitHub Copilot, default model.
  - [editor](round7-editor.md): Codex GPT-6 Astra, high reasoning.
- **Constraints:** No outcome was read and no GPU was used. Part A remains unscored and held by the user. Nothing in this synthesis changes appendix A5.
- **Thresholds:** any threshold not cited to a file is a chair choice.

## Verdict on the paper

**What WSR (Kocabay, alphaXiv, 13 Nov 2026) shows.** It is exploratory, positive evidence that a capable LLM given read-only weights plus a CPU sandbox predicts the direction of behaviour changes better than the same LLM given a text description. It does not show that LLMs "read" weights in any sense a fixed script could not.

All four members independently reached the same five points:

1. **Thin evidence.** The headline +0.152 balanced accuracy (0.730 vs 0.578) rests on 18 changed labels from 10 interventions. Each cell ran once, contrasts are uncorrected, and dose siblings share parents. World-model notes that in 3 of the 4 subtle cells the weights-minus-text interval includes zero.
2. **Unequal information.**
   - Only the weights arm can decode token IDs from `tokenizer.json`; the authors concede part of the gain may come from knowing which token was edited (§4.4).
   - The authors say whether an edit acts is "largely predictable from its description".
   - **There is no scripted-statistics baseline.** The only non-LLM baselines are constant and random (`wsr/harness.py`). The paper therefore cannot separate "reasons over weights" from "computed norms and decoded a token".
3. **Over-prediction.** On Muse-30B, all 40 edits were inert, yet readers predicted change on 53–88% of them. Weight access *raised* false alarms relative to text. No reader checked how a scaled tensor passes through the following normalisation.

   Ahmed's audit (alphaXiv 2611.08070) finds the same pattern in small CNNs: local arithmetic survives label permutation, but semantic naming fails. That supports dataset-prior caution, not a refutation of WSR.
4. **Labels.** WSR's labels are not our metrology.
   - They use item-level Welch tests with a normal reference (`wsr/labels.py` uses `erfc`, not finite-df t) on single sampled runs.
   - World-model computes that a "same" label can hide a shift of about 0.10–0.14 (*computed*).
   - Our TF-v1 endpoint is deterministic continuous NLL, with a panel half-width of 0.00487 nats and a floor h* = 0.0049397 nats/token (F8a.2, F9a). WSR thresholds cannot be imported.
5. **Relation to the thesis.** An LLM weight reader is **not** a model-specific world model. It is a general transformer prior, instantiated per target by tool calls, with nothing fitted to that target's outcomes.

   WSR's own evidence agrees:
   - A stronger outside reader read every target best.
   - Self-framing had no effect.
   - The crossed self contrast includes zero.

   For our NLL endpoint, a forward-free reader has *less* model-specific information than the field. The field uses this base's own gradient and K-FAC factors, which no forward-free reader can compute.

**Consequence for the user's thesis.** WSR is consistent with "more than a single neuron is needed": it uses a readout far richer than three coefficients. But it does not supply the model-specific learned response model the thesis calls for. That model is what this round recommends developing.

## Why F12 failed: the design lesson

Members agree. The figures are from `docs/reviews/council/board-a-f12-sim-report-2026-11-14.md`.

- **(i) A conjunction built to fail.** The second comparator had an expected margin C_WM,S = 0.045 under the alternative. That is below the 0.05 point threshold, so power tends to zero as N grows (world-model). StatNN already captured C_S:F3 = 0.132 of the oracle's 0.15.
- **(ii) Ridge bias.** F3's ridge-1 shrinkage lets any learned head "win" on a purely linear signal. The stress-scenario target C_WM was 0.021, and detection fired in 110/1000 replicates. The fix is an **unpenalised** three-coefficient field readout.
- **(iii) Coverage.** Coverage failed because refits sat inside the resampling. The fix is a predictor frozen before a fresh panel, so that no refit enters the interval.
- **(iv) Sample size.** Rows are not independent units. Only new independent root components add power, and chain steps sit below the floor.

World-model's crude √K extrapolation (*computed*, not simulated) puts 0.80 power for a single learned-vs-field contrast near **590 independent components**. Metrology, representation and editor decline to state any N without a new simulation. **Adopted:** no N is registered here. The extrapolation is recorded only as an order-of-magnitude warning that a confirmatory panel is likely to need several hundred fresh roots.

## Decision 1: develop MSRM-1, a model-specific response model (exploratory development)

All four members recommend the same class: a scripted-statistics residual model nested on the field.

- Three members (metrology, editor, representation) recommend the scripted residual directly.
- World-model's MSRM-1 is the same model with a wider input set.

The chair adopts the merged specification below as **MSRM-1 development v0**.

- **Prediction.** ŷ(D) = F3u(a, q) + h_θ(x(D)) − h_θ(x(0)).
  - F3u is β₀ + β₁Σa + β₂Σq fitted by **unpenalised** least squares inside each training split.
  - The correction is exactly zero at the zero edit.
  - The target is continuous TF-v1 WikiText Δy in nats/token.
  - Sign and a three-way label at h* are reported descriptively, with "same" read as *unresolved*, not as equivalence.
- **Inputs x(D).** All are CPU-only, from the realised bf16 delta and cached fields, with no new base-field collection.
  - (1) Representation's 48-vector:
    - 18 global means of the `src/wsp/features/stats.py` delta statistics, with the top-8 stable-rank truncation labelled;
    - a and q overall and per depth bucket (10 values);
    - 17 role shares of squared delta norm, in `pool.py:TYPES` order;
    - maximum tensor share, rounding-loss fraction, and fraction of tensors changed.
  - (2) Editor's scripted WSR-style columns where they are not already present:
    - relative Frobenius displacement;
    - cosine(D, W);
    - realised/intended norm ratio;
    - delta-energy-weighted depth;
    - world-model's top-1 singular-energy share, and |cos| of the leading output vector with the unembedding mean.

  Every zero denominator has a pinned value and a test. A missing required field excludes the row from **every** arm.
- **Head class.** This is a development choice made once by grouped inner CV, not a confirmatory race. Candidates:
  - a width-8 GELU residual head (401 parameters at 48 inputs);
  - a ridge residual;
  - a boosted tree over {100, 300, 600}.

  The head is trained with L1 on F3u residuals, using the F5.1/F12.2 inner folds, seeds 0/1/2 and stopping recipe as defaults.
- **Baselines on identical folds and rows.**
  1. F3u.
  2. Zero-fit a.
  3. Same-input StatNN.
  4. A **scripted reader**: the same statistics with the same head class but no field-derived inputs.
  5. A **description-only** learned model (family, dose, role, layer, rank; no weight statistics). This is the analogue of WSR's text-only arm.
  6. **Name-blind** MSRM-1, with role and layer identities replaced by opaque IDs.
  7. A field-ablated MSRM-1.
  8. A shuffled-outcome control, which must show no learnable gain.
- **Rows.**
  - The development rows are the **123 existing non-Part-A root records / 89 components** (F12.2; sorted-id sha256 `f96e4ea5…ca2160`). They are all already on file and all treated as exposed, so A4.4 exploratory wording applies.
  - The 44 roots in 37 old-seal `confirmatory` components are G6-exposed and barred from G7 already. Using them in development adds no new exposure.
  - The status-literal subset of **79 roots / 52 components** is reported as a sensitivity analysis.
  - The family mix is chain 105, lora_sft 10, abliteration 4 and random_dir 4. Family results are therefore descriptive only, with a leave-one-family-out diagnostic.
  - Corrected census: 689 TF-v1 records, not the 694 the brief stated (F12.10(a); `docs/PROGRESS.md`). These are not independent training rows.
- **Exclusions.**
  - Every Part A component.
  - Any `fresh_unselected` entry.
  - Non-root chain steps.
  - Task vectors.
  - Rows lacking provenance.
  - Any row whose component closure (shared sources, directions, ancestry, duplicate delta bytes) is unresolved.

  Outcomes are read only through an allowlist built from the F12 population ids, enforced by the `scripts/f12_inputs_guard.py` read boundary. **Nothing under `data/evals` for Part A is opened.**
- **Test-first contract** (union of the members' lists):
  - realised-bf16 no-ops give exactly zero;
  - signed-a reverses under ±D;
  - q is never negative;
  - LoRA gauge equivalence;
  - invariance to tensor order;
  - norm shares sum to one;
  - the residual is zero at the zero edit;
  - preprocessing is isolated inside each training fold;
  - folds are component-disjoint, with parent, dose and alias groups never crossing a fold;
  - prohibited paths are rejected;
  - the shuffled-outcome control shows no gain;
  - all arms receive identical eligible rows.
- **Budget.** At most **10 CPU-h**, **0 GPU-h**, **0 LLM tokens**, and **≤ 4 GB per job**. Jobs run via agent-tmux with script files. Stop and reprice if cached inputs are insufficient.
- **Report.** Pooled out-of-fold MAE, and the ratio-of-sums contrast C = 1 − Σ|e| / Σ|e_F3u| for every arm, with component-bootstrap intervals. Results are split by family and by the 123 vs 79 populations. All of this is labelled **exploratory development on exposed outcomes, not evidence for or against the thesis**.
- **Development kill criteria** (development decisions, not scientific nulls):
  - k1: development C(MSRM-1 : F3u) < 0.05, or its upper bound < 0.10.
  - k2: no margin over the description-only model, meaning the gain is a prior.
  - k2b: no margin over the scripted reader, meaning the field is adding nothing beyond statistics.
  - Provenance or calibration failure.
  - The CPU cap is exhausted.

## Decision 2: the LLM reader audit (R-audit) is deferred

A WSR-style reader on our deltas has four arms:
- decoded text with norms;
- weights plus tools;
- weights plus tools plus a field tool;
- name-blind weights.

It would run against the scripted reader, with two repeats per arm. Members agree it is worth doing only as a **descriptive** test of whether tool use adds information beyond matched scripted statistics.

The proposals differ on scale:
- editor: at most 20 cases, about 200k tokens;
- world-model: 40 roots, 32 sessions, with tokens measured by a pilot.

**Deferred** until MSRM-1 development reports. Running it needs a token cap that only the user can set. Its kill rule is adopted now: stop if neither weights arm beats the scripted reader by ≥ 0.05 balanced accuracy. A hybrid that feeds LLM features into MSRM-1 is not developed unless the audit shows reproducible extra information.

## Decision 3: confirmation is a later, separate registration, and the user's call

- No confirmatory claim can come from the 123 exposed rows.
- A confirmatory test of MSRM-1 would need a **new dated amendment** with the following design:
  - MSRM-1 and F3u frozen on all development rows;
  - a fresh panel of independent training-free roots, one component each, in families the development rows cover, frozen before any outcome;
  - per the editor and world-model, the panel should include dose ladders crossing s*, multi-tensor combinations, WSR-style scaling, direction-removal and noise edits, and explicit identity controls kept separate from rounded-away edits;
  - **one** primary contrast, MSRM-1 vs F3u: "detected" requires C ≥ 0.05 and a lower bound > 0;
  - StatNN, the scripted reader and the description-only model as descriptive secondaries.
- **Before any purchase**, it must pass an outcome-blind simulation adapted from `scripts/f12_simulation.py`. The simulation must:
  - use real development and fresh-panel tokens;
  - draw noise from the F8a.2 payload;
  - include a purely linear stress scenario (which must give C ≈ 0) and a family-offset stress scenario;
  - test both coverage tails;
  - refit the full pipeline on every replicate;
  - include the Meter.end and `units_by_kind` repairs owed at C2.

  Gates: size ≤ 0.07, power lower bound ≥ 0.80, and coverage ≤ 0.07. N is the smallest panel size that passes. If none passes, nothing is bought.
- **Order.** The amendment sits after Part A and the F8/F9 order (F5.5). Its GPU cost is priced only once N is set. World-model's estimate is 10.4 GPU-h plus 15.7 CPU-h at N = 600 (*computed*).
- **Optional development rows.** Buying the 147 purchasable development roots (about 2.55 GPU-h) is a separate option for the user. F12's no-purchase ruling binds only F12. It is not exercised here.

## Decision 4: registered plans

- **No change** to F8, F9, H5, F12-I or F5.5. F12 stays closed.
- **Flag for Board A, not an amendment.** World-model notes that A5.7's H5 rejection clause ("upper bound < 0.05 against augmented StatNN **and** field-only") has F12's conjunctive shape. It should be re-simulated, and amended by a dated amendment if needed, before any G8 episode count is fixed.

## Summary table

| proposal | mechanism | registered test | cost | kill criterion |
|---|---|---|---|---|
| **MSRM-1 dev v0 (adopted)** | Unpenalised field readout plus a small learned residual over signed field terms and scripted weight statistics | None (exploratory, 123 exposed roots; 79-root sensitivity) | ≤ 10 CPU-h, 0 GPU-h, 0 tokens, ≤ 4 GB | k1, k2, k2b; provenance or calibration failure; cap exhausted |
| R-audit (deferred) | LLM reader with weights and tools vs scripted reader, decoded-text and name-blind arms | None (descriptive) | 0 GPU-h; token cap set by the user | Weights arms fail to beat the scripted reader by ≥ 0.05 balanced accuracy |
| MSRM-1 confirmation (future) | Frozen MSRM-1 vs F3u on fresh independent roots; one contrast | New amendment after Part A, by the user's decision; simulation first | Simulation CPU (pilot-priced); GPU priced at the passing N | Simulation fails at the affordable N; confirmatory upper bound < 0.05 |
| F8/F9/H5/F12-I | Unchanged | As registered | As registered | As registered (A5.7 conjunction flagged) |
