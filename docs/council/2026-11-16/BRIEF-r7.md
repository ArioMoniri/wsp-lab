# Council brief: 16 Nov 2026, round 7 (read this first)

**Topic:** what "Weight-Space Reading" (WSR) means for this programme, and which model we should develop next.

## The user's request

"check this in council n model develop": https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models

The council is asked to do two things:

1. Assess the paper against this programme.
2. Specify one concrete model to develop next, with a design an implementer can build test-first.

The user's standing thesis comes from round 6 (`docs/council/2026-11-11/BRIEF-r6.md`). It holds that predicting weight edits needs more than a single-neuron readout, and that at minimum a model-specific world model is required.

## The paper, as fetched by the chair on 16 Nov

Re-fetch it yourself and cite what you read.

**Reference**
- Kocabay, "Weight-Space Reading", Poindexter Labs, alphaXiv v1, submitted 13 Nov 2026.
- Code (Apache-2.0): https://github.com/suayptalha/wsr. The repo was created 11 Nov and last pushed 13 Nov.

**Method**
- The reader is an LLM agent with read-only access to a target model's weights and a CPU Python sandbox. It runs no forward passes.
- For each surgical edit, it predicts the direction of five behaviours: up, down or same. The edits are block or row scaling, direction removal, and noise. The five behaviours are thinking length, math accuracy, multiple-choice accuracy, sycophancy and refusal.
- Ground truth comes from running the edited model. A change is labelled when Welch's test gives p < 0.01 and the effect meets a minimum: 10% of base tokens for thinking length, or 0.05 for rates.
- The paper reports three pre-registered rounds. Models are Qwen3.8-27B, Qwen3.5-9B and Muse-Glimmer-30B, each serving as both reader and target.

**Headline result**
On subtle, non-destructive edits, 27B reading its own weights scored a balanced accuracy of 0.730. The baseline, a text description of the same edits, scored 0.578. The difference is +0.152, with 95% CI [+0.056, +0.256].
- Change recall rose from 0.278 to 0.667.
- Specificity fell from 0.878 to 0.793.
- The sample was 10 changed interventions, giving 18 changed labels.

**Other results**

| Comparison | Result |
|---|---|
| 9B→27B, weights vs text | 0.541 vs 0.451 |
| 9B→9B, weights vs text | 0.554 vs 0.492 |
| 27B→9B, weights vs text | 0.628 vs 0.648 |
| Self-reading advantage | Crossed CIs include zero |
| Destructive edits | About 0.59 for every reader; the authors call this damage detection |

**Muse-30B:** none of its 40 edits changed measured behaviour, yet readers predicted a change on 53–88% of them.

**Limitations the authors state**
- Each cell ran once.
- No multiplicity correction.
- A single model pair.
- Unexplained inert edits.

**Related**
- Ahmed's audit of zero-shot weight reading (https://www.alphaxiv.org/abs/2611.08070). It reports that LLM readers lean on dataset priors.
- Takehi et al., Sparks of WSP (round 6 brief).

## Our state (cite files, not this brief, for anything you rely on)

| Item | State | File |
|---|---|---|
| Base model, zoo | Qwen3.5-0.8B. 1,259 zoo entries, of which 694 have TF-v1 records. | `docs/PROGRESS.md` |
| Part A (registered primary) | Base-anchored field (a, q) with a readout of at most 3 coefficients. 52 edits, predictions frozen, **unscored and held by the user**. | A5/F11 in `docs/PREREG_H1_TF.md`, `docs/prereg/a5_manifests/part_a/` |
| Round 6 verdict | The readout is a single neuron, but the information is model-specific. No registered contrast pitted a learned model-specific predictor against the field, so F12 was added. | `docs/council/2026-11-11/SYNTHESIS-r6.md` |
| F12 (learned K+F head vs field and StatNN, 270 roots) | The outcome-blind power simulation **failed all 14 gates**. Power was 218/1000 (L = 0.180), and both stress and coverage gates exceeded 0.07. Result: "unresolved — not powered / not qualified". No purchase, repeat or extension. Descriptive figures: C_WM 0.172 vs F3, but only 0.045 vs StatNN. | `docs/reviews/council/board-a-f12-sim-report-2026-11-14.md`, PREREG notes F12-S2 and the chair acknowledgement at bd7ae62 |
| Earlier learned predictors (342 rows) | StatNN 0.56/0.66 beat neural encoders (0.45/0.50, 0.42/0.56). | `docs/reviews/P4/d3v2f-summary-2026-11-08.md` |
| Noise floor | Panel half-width 0.00487 nats; F8 floor h* = 0.0049397 nats/token. | A5, F8 |
| Already registered for later | F8/F9 external-edit and family-transfer panels, H5 critic (A5.7), and the F12-I additivity panel. All come after Part A. | A5 F8–F9, F12-I |

## Questions

1. **What WSR establishes.**
   - Is 0.730 vs 0.578 on 18 labels, single runs, uncorrected, evidence that an LLM can read edit effects from weights?
   - Or could it be explained by edit-type priors, since the text baseline omits weight statistics an agent can compute in a sandbox? Address the Muse over-prediction and the Ahmed audit.
   - Compare its labelling (Welch p < 0.01 plus a minimum effect) with our panel floor.
2. **Relation to the thesis.**
   - Is an LLM weight reader a "model-specific world model", a general prior, or a scripted-statistics estimator in disguise?
   - Where does it sit relative to the field (a, q), StatNN and the H5 critic?
   - Does WSR's tool-use reader beat a scripted baseline that computes the same statistics? The paper does not test this; say whether that matters.
3. **Transfer to our setting.**
   - Can a WSR-style reader (an LLM agent with read-only access to our 0.8B base and delta, plus our field tooling) be run on our edits?
   - Which edits and labels would it use? Our targets are TF-v1 endpoint changes in nats, not up/down/same behaviours.
   - It must not touch Part A edits or anything under `data/evals` for Part A.
4. **Lesson from F12's failure.**
   - Why was the 270-root two-comparator design underpowered? Consider the small edge over StatNN (0.045), ridge-bias stress, and narrow intervals.
   - What design change would be needed for any learned or reader arm to reach power ≥ 0.80? Options include more independent roots (the zoo's 694 TF-v1 records), a single comparator, sign/direction targets as in WSR, or lower-noise targets.
   - Use the cheap outcome-blind F12 simulator as the pricing tool, and give numbers only if you compute or cite them.
5. **The model to develop (the main deliverable).** Specify one model, plus at most one alternative. Include:
   - inputs, architecture or agent loop, parameter count or token cost;
   - training and development data (which existing zoo rows; source-component clustering; exclusions);
   - what it predicts (sign, magnitude, or a three-way label) and its baselines: field, StatNN, a scripted-statistics reader, and text-only;
   - the exploratory development protocol on already-observed non-Part-A rows, kept strictly separate from any later confirmatory test. A confirmatory test can only run on a new prospectively registered panel, after Part A per F5.5 ordering, and as the user's decision;
   - the outcome-blind power simulation it must pass before any purchase;
   - CPU-h, GPU-h and LLM-token cost, and a kill criterion.

   Candidates to weigh include:
   - (a) a WSR-style LLM reader agent over our deltas;
   - (b) a scripted "reader" that computes the same weight statistics, then a small learned head;
   - (c) a learned model-specific response model trained on all 694 TF-v1 rows with field tokens;
   - (d) a hybrid in which the LLM reader outputs features consumed by (c).

   Recommend one.
6. **Anything the paper changes in registered plans** (F8/F9/H5/F12-I). These can be changed only by a new dated amendment before any outcome. Default: no change.

## Rules (as before)

- **Evidence.** No invented numbers. Every claim about our data cites a repository file. Every external claim cites a URL you fetched; record fetch failures.
- **Length and format.** At most 2,000 words. Open with a 10-line BLUF. End with a table: "proposal | mechanism | registered test | cost | kill criterion".
- **Output.** Write only `docs/council/2026-11-16/round7-<role>.md` in this worktree.
- **Prohibited.** No commits, no GPU, no writes to `data/`, no edits to `docs/PREREG_H1_TF.md`, and do not open anything under `data/evals`.
- **Secrets.** Write no secrets, IPs or hostnames.
- **Roles.** `world-model`, `representation`, `metrology`, `editor`. The chair writes `SYNTHESIS-r7.md`.
