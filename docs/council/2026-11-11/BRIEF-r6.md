# Council brief — 11 Nov 2026, round 6 (read first): is a first-order field a "single neuron", and does weight-edit prediction need a model-specific world model?

## The user's thesis (verbatim intent)

"State weight editing and prediction should be more complicated than predicting with a single neuron and can be done with world
models. Check the new articles on how they edit the weights and check these with the council: not a single neuron can be able to
do this; at worst there should be model-specific world models, at least to predict about that model."

The council is asked to take this thesis seriously as a challenge to its rounds 4–5 position (one receiver-anchored first-order
forecast plus a ≤ 3-coefficient online readout as the best predictor now), and either refute it with evidence or turn it into a
registered arm.

## State (unchanged facts; numbers live in the files named)

| Item | Result | File |
|---|---|---|
| Registered primary | base-anchored sensitivity field: a = ⟨g, D_eff⟩ (first-order), q = ½tr(GDADᵀ) (K-FAC magnitude), chart secondary; Part A rotation + ladder on 52 edits (40 registered + 12 supplementary); power 0.8722 | A5, F11, `part_a/f11_calibration_record.json` |
| Part A status | staged, predictions frozen for all 52 edits, released with conditions, **not scored** (phase 1 refused on a dirty tree; held pending the user) | `part_a/PRESCORE_INVENTORY_2.md`, `board-a-release-2026-11-11.md` |
| Learned model-specific predictors already measured on this model | 342-row fast-tier suite: pooled-statistics tree (StatNN) 0.56 / 0.66; descriptor-only 0.51 / 0.61; best neural encoders (W2T-style tokens, JEPA-style) 0.45 / 0.50 and 0.42 / 0.56; family split within [−0.29, 0.23] | `docs/reviews/P4/d3v2f-summary-2026-11-08.md` |
| Registered learned arms (Part B, after Part A) | augmented StatNN (field tokens + pooled stats), K-FAC-anchored GL-set head (~1.6k params), LoL secondary family (learned / random / anchored frame, spectrum-only control), field-absent and random/permuted-field controls, chart arms | A5.4, F5.1–F5.2 |
| H5 critic | ≤ 3-coefficient online readout over a, q (A5.8.3) with abstention; scripted estimator and random best-of-N baselines | A5.7, F5.4, F8.7, F9.4–F9.5 |
| Zoo | 1,259 entries (618 chain entries in 99 lineages; 121 lora_sft; abliteration; random_dir; heretic); TF-v1 records for 694 | `docs/PROGRESS.md` 8 Nov |
| Frozen predictions, this morning | sign of a per contrast for 52 edits; rotation predictions 40 negative / 3 positive; ladders 47 / 5 and 45 / 7 negative / positive | `part_a/predictions_f11.json` |

## How the new papers edit the weights (verified 10–11 Nov; cite the URL you read)

| Paper | How the edit is made | What predicted the effect before it was applied |
|---|---|---|
| Sparks of WSP (Takehi et al., 2 Nov) | LLM reads activations on practice prompts, finds a loop direction u, subtracts u-components from 64 down-projection columns in two layers (rank-one per column) | nothing: 50 forward passes of trial and error on practice problems |
| Exploring WSP (arXiv 2611.09382) | Astra edits a single down-projection column (1,023 values) or dense rank-one updates; 16 trajectories; 6 kept the original | development-set feedback only; development gains often vanished held-out |
| WeightLoop (arXiv 2611.07319) | an LLM-written editing *program* changes 16 of 128 output weights of a 3.6M-parameter predictor as assay results arrive; validated on held-out support, permutation controls, rebuilt from base | validation after the fact; no forecast |
| aiXiv replication; Barry LP | difference-of-means direction; LP over column choice and scale | none; recovers 93 % of the gain given the layers and budget |
| Concept Country (27B) | rank-one additive term in nine down-projections from residual activations; norm-matched orthogonal control | none; pre-registered confirmation |
| SWEET (2611.10974), TFTV (2609.09054) | activation contrasts → suppress contributions along a direction (neurons/heads); forward statistics → rank-one weight edit with arithmetic | none |
| Deletion Tuning (2611.10866) | learned binary mask on gate projections (< 0.05 %) via GRPO/SFT/OPD | the training loss itself |
| RL for neural model editing (arXiv 2606.13461) | a policy observes weights and emits weight modifications, trained by RL | the policy's value function, implicitly |
| WSP of policies with model checking (arXiv 2611.11604) | the LLM writes all layers' weights with notes on what neurons compute; model checking verifies the policy | formal verification after writing |
| Predict Before You Modify / IDM (2610.13991) | agent-program modifications | an editable improvement-dynamics model, learned online from outcomes |
| PPF (2508.02381) | structured channel pruning masks | a surrogate trained on 775 evaluated masks (second-level prediction) |
| RLHF reward-edit forecast (2611.03244) | reward-model edits | a first-order covariance rule fixed before training (82 % in-sample, 66 % out of sample) |

Reading: every weight-editing paper on LLMs either predicts nothing (trial and error with feedback) or uses a first-order /
statistical rule; the only *learned* forecasters are model-specific surrogates trained on hundreds of evaluated modifications in a
low-dimensional modification space (masks, programs). None learns a predictor over arbitrary dense deltas for one LLM from that
LLM's own edit outcomes, which is exactly what the zoo plus TF-v1 records could train.

## The questions for round 6

1. **Is the registered field a "single neuron"?** State precisely what the primary arm is as a function class (a fixed linear
   functional of the realised delta plus a fixed quadratic form, with at most three learned coefficients), what it can and cannot
   represent (interactions across tensors, non-local effects, sign changes along the ladder, trajectory dependence), and what the
   rounds 4–5 evidence (sign agreement 33/33 with the power predictions; Part A unscored) does and does not establish. Be
   explicit about the user's claim: can a single first-order readout in principle predict weight-edit effects, or only their local
   direction at small scale?
2. **What is a model-specific world model here, concretely?** Define the candidate class: a predictor f_θ(state, edit) trained
   only on THIS base model's data (the 1,259-entry zoo, 694 TF-v1 records, chain trajectories, probe activations), predicting the
   endpoint change for an unseen edit; options: (a) the already-registered augmented StatNN and anchored GL-set head (are these
   the model-specific world model the user asks for? say exactly why or why not); (b) a JEPAWG-style latent dynamics model over
   chain trajectories (state z_t → z_{t+1} given the edit), using the 618 chain entries as supervision; (c) a learned readout over
   per-tensor field tokens, probe activations and chart coordinates with more than three coefficients (a small MLP or kernel
   model) with proper nested cross-validation; (d) an activation-assisted model (forward passes on probe prompts after the edit)
   as the upper-cost reference. For each: inputs, parameters, training rows available, expected gain over the field and over
   StatNN with a stated prior, cost, and the registered test it must pass (Part B contrast rules, A5.4.2 power ≥ 0.80).
3. **Why did the learned encoders lose to pooled statistics on 342 rows** (`d3v2f-summary`), and what would change that:
   more rows (the 1,259-entry zoo at TF-v1 is 694 records; what is the marginal GPU cost to close it: 565 × 1.04 min ≈ 9.8 GPU-h),
   better inputs (field tokens, probe activations), or a different target (trajectory steps rather than roots)? Give the sample-size
   argument with the measured noise (0.00487 nats panel half-width).
4. **Model-specific by construction.** The field is already anchored to this model (g, G, A are its gradients and Fisher factors).
   Is "model-specific world model" satisfied by anchoring plus a learned readout, or does the user's thesis require the predictor
   to learn this model's *response function* from outcomes? Where does the registered programme already test the difference
   (field-only vs augmented StatNN vs activation-assisted ladder), and what is missing?
5. **A registered arm (F12) if warranted.** Propose the smallest decisive, pre-outcome registrable arm that would test "a learned
   model-specific world model beats the first-order field on this model's held-out edits", with: training set (which rows, which
   folds, source-component clustering), model class and parameter count, the contrast and its decision rule, the power simulation
   it needs, GPU-h, the kill criterion, and how it sits relative to Part A/Part B ordering and F5.5 (it cannot change them).
   Say explicitly whether it should wait for the Part A outcome (it must not read it) or can be registered now.
6. **The editing side.** Do the papers' edit mechanisms (column edits, rank-one from activations, learned masks, LLM-written
   programs, RL policies) change what the predictor must represent? Which of these edit families would most embarrass a
   first-order predictor, and is that family already in F8/F9 (deletion, SWEET, TFTV, external edits)?

Rules as before: no invented numbers; every claim about our data cites a file; every external claim cites a URL you fetched;
≤ 2,000 words; 10-line BLUF; final table "proposal | mechanism | registered test | GPU-h | kill criterion". Write
`round6-<role>.md` in this directory only; no commits, no GPU, no data/ writes, no edits to A5; do not open anything under
data/evals (Part A is unscored and held). Roles: `world-model`, `representation`, `metrology`, `editor`. The chair writes
`SYNTHESIS-r6.md`; any amendment goes to A5 as F12 before any G3 outcome is read.
