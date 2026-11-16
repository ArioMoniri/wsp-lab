# World-model — round 8, 16 Nov 2026 (single-LLM edit world model: trajectories and enhancement)

No experiment, GPU use, model download, `data/` access or outcome read. *Computed* means arithmetic on the cited figures, not a measurement.

## BLUF

1. The weight transition is **known exactly** (θ_{t+1} = θ_t + D_t) and outcomes depend only on θ_T. A world model must never learn this bookkeeping; M3 did (it predicted φ(θ_t − θ₀)).
2. Rollouts beat static prediction only via (i) a **functional state** observed at intermediate θ_t during training, (ii) a local step operator that **extrapolates in horizon**, (iii) trajectories leaving the field's trust region. Inside it, the static field on the cumulative delta is already exact to second order.
3. **"Without running the edit"** = no evaluation of the network or its derivatives at any θ ≠ θ₀ (tier R0). Re-linearising at θ_t is tier R2, i.e. running the edit; the world model replaces it at inference but may use R2 observations as **training targets**.
4. On 0.8B, re-linearising cost 15× a full TF-v1 evaluation (*computed*); it cannot be a cheap planning tool.
5. **WM-E v0:** field-nested ŷ; latent h_t trained to predict observation-tap sketches and action derivatives at θ_t; ensemble; multi-horizon training.
6. **Planner:** batch Thompson/CEM over a parametric action space, R0-scored, refit online from real dev evaluations; A5.7 equal-budget accounting.
7. (E1) selected edit gains on held-out test items, lower bound > 0, no guardrail breach; (E2) separately, the WM arm beats StatNN at equal budget. E1 alone does not show the WM helped.
8. Qwen3.5-9B is acceptable here if evaluated in non-thinking mode by log-likelihood, with objectives chosen by a headroom pilot (MMLU-Pro 82.5, IFEval 91.5 leave little room).
9. **First build:** five test-first modules plus a ≤ 4 GPU-h pilot with four go/no-go gates.
10. MSRM-1 dev v0 binds: StatNN ≥ MSRM-1 and k2b fired, so every claim needs scripted and description-only arms.

**Fetched (16 Nov; no failures; WSR source files not opened beyond README and tree).**
- https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models: reader "cannot run a forward pass"; 9B→9B text 0.492, weights 0.554; no compositions; predictions not used to improve a model.
- https://github.com/suayptalha/wsr: CPU-only container, read-only weights, no GPU or network; `wsr program` accepts edits by *measured* held-out scores with guards (`passes: 10` forward passes).
- https://huggingface.co/Qwen/Qwen3.5-9B: 32 layers, hidden 4096, FFN 12288, 8 × (3 × Gated DeltaNet + 1 × Gated Attention), vision encoder, thinking on by default, MMLU-Pro 82.5, IFEval 91.5, no GSM8K row; vLLM main branch required.

## Q2 — World-model structure

### What a rollout can add

Y is path-independent, so a static P(θ₀, Σ_{i≤t} D_i) is a valid trajectory predictor, and the field on the cumulative delta (q carries within-tensor cross terms) is exact for locally quadratic Y. A world model adds something only where Y is **not** quadratic along the path: past s*, saturation or behaviour flips, and normalisation rescaling (WSR's readers missed Muse's post-norm cancellation). The 0.8B zoo had median κ 0.023 at s = 1 (SYNTHESIS-r6 Q1), mostly inside the trust region. Trajectories must leave it or the world-model-vs-static contrast is vacuous.

### Three mechanisms that can beat static prediction

1. **Grounded functional state.** h_t predicts quantities observed at θ_t: residual sketches at K layers on fixed probe prompts, per-item losses, and action derivatives A_t[j,k] = ⟨∇L_k(θ_t), E_j⟩ for a dictionary of primitive edits E_j (hook-computable as Σ(uᵀδ_out)(vᵀx_in) without materialising g). Targets are external and fixed, so unlike M3's VICReg self-prediction, collapse is not an optimum.
2. **Local step operator.** h_{t+1} = h_t + f_θ(h_t, x_t, e_t), with the exact bookkeeping x_t as input, never predicted. Trained on horizons ≤ 2, a state-conditioned operator should stay nearer its training distribution at horizons 3–5 than a static model facing an unseen cumulative delta. That **horizon extrapolation** is the claim to test.
3. **Slope transport.** Δŷ_t = ⟨Â_t, e_t⟩ + ½q(e_t) + r_ψ(h_t, e_t); the static field uses A₀ throughout. The real A_t (R2) is a training-time **oracle**; if it does not cut per-step error, drop this mechanism (Go-R).

### Outputs and training

ŷ_t = F3u(cumulative D) + g_ψ(h_t, x_t), zero at the zero edit as in MSRM-1. Training unrolls whole trajectories with scheduled sampling; the loss covers outcomes at every horizon plus observation targets. A 5-seed ensemble supplies σ.

### Targets with signal above noise

Random directions are too small: 0.8B chain steps had predicted median |a| 0.00054 nats against the 0.00487 floor (SYNTHESIS-r6 Q1); 4B ρ = 0.005 Gaussian edits resolved 0 of 20 improvements, intervals about ±3e-4 nats (`docs/reviews/scale/scale4b-diagnostic-2026-11-10.md`). Use structured directions only, with doses set in the pilot so each step's paired change exceeds 2× its paired SE. Targets are continuous and teacher-forced: WikiText ΔNLL, harmful and harmless refusal margins, MMLU-Pro and math final-answer gold log-probability, a sycophancy log-likelihood contrast; accuracies only at the final test. Report error versus horizon, raw and as MAE(h)/SE_paired(h), for cumulative Δy (primary) and per-step increments.

### How this avoids M3's failure

M3 (`src/wsp/predictor/models/__init__.py` JEPA; round2-jepawg; SYNTHESIS-r6 Q2–Q3) had 25 evaluated pairs → hundreds of trajectories, independence counted by `src/wsp/msrm/closure.py`; sub-floor targets → doses resolvable per step; a latent predicting known features → bookkeeping as input, latent predicts functional observations only; a frozen encoder plus small head → joint outcome and observation loss; no analytic prior → nested on F3u; no control parameter → a parametric action space (layer, direction, α), the analogue of JEPAWG's p.

### Defining "without running the edit"

| Tier | Allowed computation | Status |
|---|---|---|
| **R0** | Any artefact computed at θ₀ independently of the candidate (fields, K-FAC, base activations, SAEs, any number of base passes), plus CPU arithmetic on D | World-model inference and planning. WSR-comparable: the WSR reader is CPU-only, has no forward pass, and reads weights read-only. |
| **R1** | Network passes at θ₀ whose inputs depend on D (JVP or HVP along D) | Not "running the edit", but GPU time per candidate. A separate, charged arm. |
| **R2** | Any forward or backward pass at θ ≠ θ₀: re-linearisation, probe activations of the edited model, evaluation | Running the edit. Allowed only (a) as training targets on bank trajectories, (b) as a charged oracle arm, and (c) through the **observation tap** on states already evaluated for real. |

**Re-linearising at plan time is uneconomic.** The 0.8B WikiText field took 1,648.6 s (`docs/reviews/council/board-a-g2-2026-11-11.md`); a warm TF-v1 entry 1.807 min (`docs/reviews/P5/board-a-tf-v1-rereview-2026-11-05.md`): **15.2×** a full evaluation (*computed*). Even gradient-only costs at least one evaluation.

## Q4 — Enhancement planner

**Search space** (≤ 3 composed actions; exact families are the editor's): rank-1 writes u·wᵀ into o_proj/down_proj (u a contrastive difference of means; layer, α); WSR `orth` removal (α ∈ [0, 1]); per-layer-group scaling λ_g ∈ [−0.5, 2] of small LoRA task vectors; WSR `scale`/`scale_rows` as controls.

**Loop per round:** CEM/Thompson proposes about 10⁴ candidates; the world model scores them at R0; a batch b is chosen by objective UCB under pessimistic (LCB) guardrails; the batch is evaluated for real on **dev items**; the residual head refits online. Budget: B real evaluations per episode (e.g. 4 × 6), set by simulation.

**Arms at equal B, identical dev panels:** world model, random, F3u-, StatNN-, scripted- and LLM-reader-ranked, WSR's `wsr program`, plus best-of-N runtime steering and a small LoRA SFT whose selection uses B dev evaluations (training compute reported).

**Item splits:** disjoint hashed bank, dev and test pools. Test items are unreadable by planner objects (guarded like `scripts/f12_inputs_guard.py`); each arm's single final edit is scored on test once.

**Claims:**
- **E1, "enhanced":** target test gain over base with paired 95% lower bound > 0; every guardrail (WikiText ΔNLL, MMLU-Pro, harmless refusal) non-inferior at preregistered margins; confirmed once by **generation** accuracy in non-thinking mode (`enable_thinking=False`, per the card) in a separate vLLM venv.
- **E2, "guidance helped":** paired across episodes (objective × family × seed) with A5.7 accounting: best-found test-score difference, savings S = 1 − ΣE_WM/ΣE_random as a ratio of totals, episode-cluster bootstrap-t (seed 20261107), power simulated first. **One primary contrast, WM vs StatNN**, not A5.7's flagged "and" conjunction.
- **E3, competitiveness:** against LoRA SFT and steering, descriptive, with the bank's GPU-h reported as amortised cost.

## Q1/Q3 notes from this angle

**Memory** (*computed* from the card): K-FAC FFN factors alone are 3 × (4096² + 12288²) × 4 B = 2.01 GB per layer, **64.4 GB** fp32 over 32 layers (0.8B: 11.3 GB per objective). That exceeds RAM, so restrict the field to action-support tensors. Base gradients for down_proj and the output projections are 32 × (201 + 67) MB = **8.6 GB**. Load language-model-only weights; edit in place and restore from saved original bytes, since bf16 add-then-subtract is inexact.

**Evaluation cost:** scaling the 0.8B timings ×11.25 gives about 20 min per full TF-v1 entry and 48 s for WikiText (*computed*, unmeasured). The lean panel is priced in the pilot; at an assumed 3 min per state, an 800-state bank is about 40 GPU-h.

## Q5 (brief)

Arms on identical WSR operations on 9B: text; weights; weights + a `wm_predict(edit) → (ŷ, σ)` tool; world model alone; scripted. Report continuous and WSR three-way labels. The user sets the token cap after a 10-session pilot.

## Q6 — Smallest first build (this week)

### Test-first modules, CPU only, ≤ 15 CPU-h

1. **`wm.actions`**: families, compositions, bf16 realisation. Tests: zero action is bitwise identity; restore returns the base hash; tensor-order invariance; role map read from config, not hard-coded to 0.8B.
2. **`wm.access`**: every artefact carries its θ-hash; `predict`/`plan` **raise** on non-θ₀ artefacts. This is the executable definition of R0.
3. **`wm.tap`**: seeded residual sketches and per-item losses during evaluation; test that evaluation outputs stay bitwise unchanged.
4. **`wm.model`** on a **synthetic target** (small network with RMSNorm and saturating units). Tests: beats the static field at h ≥ 3 on non-quadratic dynamics; **no gain** on quadratic dynamics or shuffled outcomes; zero at the zero edit.
5. **`wm.plan`**: equal-budget ledger; tests that test items are unreachable and every arm gets identical B.

### GPU pilot (≤ 4 GPU-h, through agent-tmux)

- **P1:** load time, peak memory, lean-panel timings, tap overhead, bitwise base repeat.
- **P2:** support-only gradient time and memory; K-FAC on 4 layers to extrapolate.
- **P3:** 3 families × 2 directions × 5 doses, plus 6 two-step and 6 three-step compositions: per-step paired SE and static-field rollout error.
- **P4:** R2 oracle A_t on 6 trajectories.
- **P5:** 24 random search-space candidates on dev items (headroom).

### Go/no-go before the bank (thresholds are my choices)

- **Go-T:** ≥ 50% of steps resolvable *and* static-field error at h ≥ 3 above 2× noise; otherwise report "the field predicts trajectories" and drop the world-model-vs-static contrast.
- **Go-R:** oracle A_t cuts per-step error ≥ 20% relative to A₀.
- **Go-E:** ≥ 2 of 24 candidates have a dev-gain lower bound > 0 with no guardrail breach; otherwise change objective or action space.
- **Go-C:** lean panel ≤ 4 min per state.

| proposal | mechanism | test | cost | kill criterion |
|---|---|---|---|---|
| R0/R1/R2 access tiers | Every artefact hash-tagged by θ; R0 = base-only artefacts plus CPU work on D | `wm.access` raises on non-θ₀ artefacts | 0 GPU | None (definition) |
| WM-E v0 | F3u nest + grounded latent h_t (tap sketches, A_t) + local step operator + ensemble | Held-out trajectories and families; error versus horizon, train h ≤ 2 → test 3–5; against static F3u, StatNN, scripted, description-only, re-linearised oracle | ≤ 15 CPU-h to build; bank ≈ 40 GPU-h at the priced panel | No margin over static StatNN at h ≥ 3 (lower bound ≤ 0); Go-T or Go-R fails |
| Slope transport | Predict A_t; Δŷ = ⟨Â_t, e⟩ + ½q | P4 oracle | ≈ 0.5 GPU-h (in the pilot) | Oracle gain < 20% |
| Planner (E1/E2) | R0-ranked batch Thompson/CEM, online residual refit, LCB guardrails | E1: test lower bound > 0, guardrails non-inferior, generation confirmation. E2: WM vs StatNN at equal B, episode bootstrap | B × episodes × panel cost, priced after the pilot | Go-E fails; simulated power < 0.80 at an affordable episode count |
| Strong alternatives (E3) | LoRA SFT, best-of-N steering, `wsr program` at equal B | Descriptive head-to-head | Priced in the pilot | None: always reported |
| Pilot P1–P5 | Measured timings, noise, dose response, headroom | Gates Go-T/R/E/C | ≤ 4 GPU-h | Any gate fails → reprice or redesign before the bank |
