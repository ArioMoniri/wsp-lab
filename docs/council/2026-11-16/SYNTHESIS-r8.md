# Council decision, 16 Nov 2026, round 8 (chair synthesis): WM-E, a single-LLM edit world model for trajectory prediction and enhancement

**Inputs.**
- [BRIEF-r8](BRIEF-r8.md), revised scope.
- Member reports:
  - [world-model](round8-world-model.md): Claude Opus 5.5, high.
  - [representation](round8-representation.md): Codex GPT-6.1 Sol, high.
  - [metrology](round8-metrology.md): GitHub Copilot, auto.
  - [editor](round8-editor.md): Codex GPT-6 Astra, high.
- `round8-representation-crossmodel-superseded.md` answered the cancelled cross-model brief. It is archived but not used.

**Status.**
- No GPU was used and no outcome was read.
- Part A remains held and is untouched.
- A5, F5.5, F8, F9 and F12-I are unchanged.
- Thresholds not cited to a file are chair choices.

**User direction (16 Nov).**
- Use one popular, current open-weight LLM.
- Show (1) **trajectory prediction** and (2) **enhancement of the LLM using world-model predictions**.
- Use the resources needed.

## Decisions

### D1 Target model: Qwen/Qwen3.5-9B, text-only, non-thinking, frozen revision

- **Choice.** Qwen/Qwen3.5-9B, chosen unanimously, conditional on a measured pilot.
- **Frozen setup.**
  - Revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, downloaded 16 Nov.
  - The tokenizer and chat template are frozen.
  - `enable_thinking=False`.
  - The vision encoder and MTP tensors are frozen and never edited.
- **Reasons.**
  - It is a WSR target: 9B→9B weights 0.554 vs text 0.492, from the alphaXiv landing page.
  - It is a current, popular model.
  - It is in our model family.
  - The Qwen-Scope SAEs are optional and are not a reason for the choice.
- **Architecture.** The model is hybrid: 8 × (3 Gated DeltaNet + 1 Gated Attention), 32 layers, width 4096, FFN 12288. Edits are restricted to audited text writers.
- **Fallback.** Qwen3.5-4B (revision `851bf6e8…`, downloaded). It is used only by a documented pre-bank decision if 9B fails the resource or cost gates. It is never pooled with 9B and is never a second confirmatory target.
- **Headroom risk.** Recorded by world-model and editor. Vendor MMLU-Pro is 82.5 and IFEval 91.5. Exploring WSP (arXiv 2611.09382, Table 2) reports a 9B math development gain of +29.69 pp that is −0.35 held-out. **The enhancement objective is therefore chosen by a headroom pilot, not assumed.** MATH-500 is descriptive only.

### D2 What "without running the edit" means: tiers R0/R1/R2 (adopted from world-model; consistent with representation and editor)

- **R0** is world-model inference and planning. It allows any artefact computed at θ₀ independently of the candidate, plus CPU arithmetic on D.
- **R1** is base passes whose inputs depend on D (JVP/HVP). It is a separately charged arm.
- **R2** is any pass at θ ≠ θ₀, including re-linearisation, edited-model activations and evaluation. This counts as running the edit. It is allowed only:
  - as training targets on bank trajectories;
  - as a charged oracle comparator;
  - through the observation tap on states already evaluated for real.
- **Enforcement.** The tiers are enforced in code by `wm.access`. Every artefact carries its θ-hash, and `predict`/`plan` raise an error on non-θ₀ artefacts.
- **Cost of re-linearising.** On 0.8B, re-linearising costs about 15.2× a full TF-v1 evaluation (*computed*: 1,648.6 s vs 1.807 min). It is an oracle, never a planning tool.

### D3 World-model structure: WM-E v0

- **Prediction.** ŷ_t = F3u(cumulative D) + g_ψ(h_t, x_t), which is zero at the zero edit.
- **Exact bookkeeping.** The exact bookkeeping x_t (cumulative E_t = W_t − W₀, realised bf16 D_t, ordered action tokens, signed overlaps with previous edits) is an **input, never predicted**. This removes M3's failure mode, in which M3 predicted φ(θ_t − θ₀).
- **Latent state.**
  - h_{t+1} = h_t + f_θ(h_t, x_t, e_t).
  - h_t is **grounded**: during training it predicts external observations at θ_t. These are tap residual sketches on fixed probe prompts, per-item losses, and action derivatives A_t[j,k] = ⟨∇L_k(θ_t), E_j⟩ for a primitive dictionary.
  - Because the targets are fixed, collapse is not an optimum.
- **Optional slope transport.** Δŷ = ⟨Â_t, e⟩ + ½q(e) + r. It is kept only if the R2 oracle A_t cuts per-step error by at least 20% (gate Go-R).
- **Training and uncertainty.** Training uses whole-trajectory unrolls with scheduled sampling and multi-horizon loss. A 5-seed ensemble supplies σ.
- **Tokens (representation).**
  - Oriented E/D coordinates from `response_chart.py`.
  - Role taxonomy from `pool.py`.
  - A fresh 32-layer schema; MSRM's 24-layer and 186-slot constants are not copied.
  - Statistics from `stats.py`, with the stable rank labelled top-8.
- **SAEs.** One layer at a time, as an ablation against random and PCA projections of equal width. They are dropped if there is no gain.
- **The trajectory claim (what the rollout must beat).** Y is path-independent, so a static predictor given the **full cumulative edit** is a valid trajectory predictor. The world model must beat that same-input static predictor (StatNN and MSRM-style) on held-out whole trajectories. The test is **trained on horizon ≤ 2 and tested on 3–4**, which is the horizon-extrapolation claim. Trajectories must leave the field's trust region: past s*, saturation, normalisation effects.

### D4 Trajectory bank (editor's plan, doses set by the pilot)

- **Roots.** 96 root recipes, 16 in each of six families:
  - block attenuation and amplification on audited writers;
  - vocabulary-row scaling, with tied-weight checks;
  - projection removal from unembedding or benign activation contrasts, plus matched random directions;
  - norm- and support-matched Gaussian noise;
  - benign difference-of-means steering compiled into writer columns or rank-one updates;
  - small LoRA SFT task deltas on construction-only examples.
- **Dose ladders.** Signed ladders at cumulative ±{0.25, 0.5, 1, 2} × a pilot-set nominal magnitude, for 768 states.
  - Each endpoint is built from W₀.
  - Scaling uses W(s) = [1 + s(f − 1)]W₀.
  - Additive edits use W₀ + sD.
- **Compositions.** 24 paired-order composition bundles (AB vs BA, extended to four steps), at most 192 visits. The total is ≤ 960 non-base visits before deduplication.
- **Signal.** Structured directions only. Doses are chosen so each step's paired change is at least 2× its paired SE. The 4B Gaussian ρ = 0.005 diagnostic resolved 0 of 20 (`docs/reviews/scale/scale4b-diagnostic-2026-11-10.md`).
- **Independent units.** These are **source-closure components** (`src/wsp/msrm/closure.py`), never rows, seeds or doses. The 96 roots are a workload, not a powered N.

### D5 Targets and metrology

- **Primary target.** Continuous paired teacher-forced log-likelihood on fixed items, with item pairing kept:
  - WikiText ΔNLL;
  - task gold-answer log-probability;
  - harmful and harmless refusal margins;
  - a sycophancy contrast.
- **"Same" labels.** "Same" means unresolved and never equivalence. The 0.8B floor is **not** reused; the 9B noise is measured.
- **Generation.** Generation is a separate endpoint. It uses fresh prompts and replicated decoding, on a frozen stratified subset of whole trajectories and in the final enhancement confirmation.
- **Primary contrasts.** One primary endpoint and horizon aggregate per claim. Secondaries use Holm adjustment.

### D6 Enhancement experiment: claims E1, E2, E3

- **Search.**
  - A frozen benign direction and task-vector library with layers and doses, explored by beam or CEM search over up to 4 actions.
  - Refusal removal and vocabulary shortcuts are excluded from the primary search.
  - "Base" is always an available choice.
- **Planner.**
  - Candidates are R0-scored, maximising predicted gain minus a calibrated uncertainty penalty, under LCB guardrails.
  - A batch is evaluated for real on **development items only**.
  - The residual head refits online.
- **Budget.** Arms receive equal real-evaluation bundles. Proposed: 32 per arm per episode, with incumbents reported at 1, 4, 8, 16 and 32. The final number is set by simulation.
- **Arms.**
  - world model;
  - world-model greedy one-step, which isolates planning;
  - random;
  - field;
  - StatNN or static residual;
  - scripted reader;
  - WSR-style LLM reader;
  - `wsr program`;
  - best-of-N directly evaluated steering;
  - LoRA SFT with ≤ 32 checkpoint evaluations, same construction data, training compute charged.
- **Splits.** Disjoint hashed and deduplicated construction, predictor-validation, search-development and final-test item groups. Test items are unreadable by every planner object (guard pattern of `scripts/f12_inputs_guard.py`). Each arm selects one winner on development items, which is then tested once.
- **E1 "enhanced".** The paired 95% lower bound of the primary capability gain on test is above 0. Every guardrail is non-inferior at registered margins. The proposed margins (WikiText ΔNLL UCB ≤ 0.01 nats/token; capability, IFEval and appropriate-refusal LCB ≥ −1 pp) are fixed only after pilot precision. The result is confirmed by generation accuracy.
- **E2 "the world model helped".** There is **one primary contrast: world model vs StatNN-ranked** at equal budget. It is paired across fresh episodes (objective × family × seed), with an episode-cluster bootstrap and power simulated first. This avoids A5.7's flagged conjunction. The other arms are secondary (Holm).
- **E3 "competitive".** Results against LoRA SFT and best-of-N steering are always reported, even if unfavourable. Total compute and bank cost are reported both amortised and unamortised.

### D7 WSR head-to-head on 9B (after WM-E v0 exists)

- **Three tracks** (representation):
  1. Replication on WSR's own edits and labels.
  2. Matched-weights, where every arm receives identical descriptions, tokens and arithmetic.
  3. Auxiliary-information:
     - reader without a tool;
     - reader with a frozen `wm_predict` tool;
     - tool-free reader given the same prediction JSON;
     - world model alone;
     - scripted reader.
- **Disclosure.** World-model training exposure is disclosed. Track 3 is not WSR's zero-shot regime.
- **Pilot token cap.** The user authorised the resources needed, so the chair sets **≤ 1.92M tokens** for the pilot: 20 edits × 3 arms × 2 seeds × 16k. Truncations count as failures. Serving is sequential with evaluation on the single GPU.

### D8 First build and pilot (next actions)

**Stage A: CPU, test-first, ≤ 15 CPU-h.** Modules:

- `wm.actions`:
  - families and compositions;
  - bf16 realisation;
  - zero action is bitwise identity;
  - restore from saved original bytes back to the base hash (bf16 add-then-subtract is inexact);
  - role map read from config.
- `wm.access`: implements the R0 guard.
- `wm.tap`: seeded sketches and per-item losses; evaluation outputs must stay bitwise unchanged.
- `wm.model`, on a **synthetic target** (a small network with RMSNorm and saturating units). It must:
  - beat the static predictor at h ≥ 3 on non-quadratic dynamics;
  - show **no** gain on quadratic dynamics or shuffled outcomes;
  - be zero at the zero edit.
- `wm.plan`:
  - equal-budget ledger;
  - test items unreachable;
  - identical budget B per arm.
- **Grouped and streamed gradient / K-FAC collector** (representation's memory contract):
  - load shards directly to the GPU;
  - gradients per layer group only;
  - no vocabulary-size covariance (it would be about 247 GB);
  - support-only field;
  - equivalence validated on CPU toy models first.

**Stage B: GPU pilot, ≤ 8 GPU-h**, through agent-tmux on the one RTX PRO 4500. Part A is held, so the GPU is free.

- **P1:** load time, peak GPU and host memory, lean-panel timing, tap overhead, bitwise base repeat.
- **P2:** support-only gradient and K-FAC timing and memory on 4 layers.
- **P3:** 3 families × 2 directions × 5 doses, plus six 2-step and six 3-step compositions. Measures paired SE per step and static rollout error.
- **P4:** R2 oracle A_t on 6 trajectories.
- **P5:** 24 random candidates from the search space on development items, to measure headroom and choose the objective.

**Gates before the bank (chair-adopted thresholds).**

| Gate | Passes when | If it fails |
|---|---|---|
| Go-M | GPU ≤ 28 GB, host RSS ≤ 24 GB, no swap | Switch to 4B before the bank |
| Go-N | Deterministic repeat within tolerance | Stop |
| Go-C | Lean panel ≤ 4 min per state | Reprice or switch to 4B |
| Go-T | ≥ 50% of steps resolvable, and static error at h ≥ 3 > 2× noise | Report "static prediction suffices" and drop the rollout-vs-static contrast |
| Go-R | Oracle A_t cuts per-step error ≥ 20% | Drop slope transport |
| Go-E | ≥ 2 of 24 candidates have a development-gain LCB > 0 with no guardrail breach | Change objective or action space |

After the gates pass, the bank is priced as H = (visits·t_ll + N_gen·t_gen + T_features + T_train) / 3600 from measured seconds. An outcome-blind power simulation (size and coverage ≤ 0.07, power LB ≥ 0.80) sets the number of independent components for confirmation.

### D9 Governance

- **Separate study.** WM-E is a new study. Its own preregistration (`docs/PREREG_WME.md`) is written and dated before any confirmatory outcome. Development is exploratory.
- **Part A.** It does not release or inform held Part A. It does not supply A5 predicates, F5.5, F8, F9 or H5 success. Manifests and panels are independent, and no held outcome is reused.
- **Operations.**
  - Long jobs run via agent-tmux.
  - 4 GB per job does not apply to the target-model pilot. This chair ruling authorises up to 28 GB GPU and 24 GB host RSS for WM-E pilot jobs, and only one heavy job at a time.
  - No `git add -A`.
  - Secret-scan before every commit.

## Summary table

| proposal | mechanism | test | cost | kill criterion |
|---|---|---|---|---|
| Target Qwen3.5-9B | Frozen revision, text-only, non-thinking | Pilot gates Go-M, Go-N, Go-C | Within the ≤ 8 GPU-h pilot | Fail → 4B before the bank |
| R0/R1/R2 tiers | θ-hash-tagged artefacts; R0 = base-only | `wm.access` raises on non-θ₀ | 0 | Definition |
| WM-E v0 | F3u nest + grounded latent + local step operator + ensemble | Held-out trajectories; train h ≤ 2 → test 3–4 against same-input static, StatNN, description-only, oracle | ≤ 15 CPU-h build; bank pilot-priced | No margin over the static cumulative predictor (LB ≤ 0); Go-T fails |
| Trajectory bank | 96 roots × signed ladders + 24 AB/BA compositions (≤ 960 visits) | Noise and closure qualification | Pilot-priced | Mostly unresolved steps or closure leakage |
| Enhancement E1/E2/E3 | R0-ranked constrained planning; equal real-eval budget; sealed test | E1: test LB > 0 plus guardrails; E2: WM vs StatNN; E3: vs LoRA and steering | 32 bundles/arm/episode (sim-set) | Go-E fails; power < 0.80; test LB ≤ 0 or a guardrail breach |
| WSR head-to-head | Three tracks; reader ± world-model tool | Paired grouped metrics | ≤ 1.92M tokens (pilot) | Gain disappears after information matching |
