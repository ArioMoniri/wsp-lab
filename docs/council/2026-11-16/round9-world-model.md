# World-model, round 9, 16 Nov 2026: contingency fixes and SSM memory for WM-E

No GPU, download, `data/` access or pilot-2 material. *Computed* = arithmetic on cited figures. "P1 report" = `docs/reviews/wme/wme-v0-pilot-2026-11-16.md`.

## BLUF

1. Weights are bookkept exactly and Y depends on θ_T only. So **memory over the edit history carries no information beyond the cumulative edit E_T**. Any SSM gain is an inductive-bias gain, and the test needs a static control with the same functional form.
2. At T ≤ 8, what selective SSMs are good at (long-range recall, parallel scan) does not apply. On non-memory tasks SSM world models only match RNNs (S4WM, R2I).
3. It is still worth one CPU arm: a **dose-time selective SSM (WM-S)**. The edit dose is Mamba's Δ, and zero-order hold gives an exact zero-edit identity, exact ladder split invariance and exponential saturation.
4. Memory genuinely matters in **planning**: a posterior that takes in real tap observations from states already evaluated. Beyond that, only for edit sequences of hundreds of steps.
5. **Q1(b):** trapezoid − left-point = ½(A_t − A_{t−1})·e − ½eᵀH₀e. The −4.8% vs 34.8% gap points at K-FAC curvature. Fixes in order: an exact R1 HVP curvature, then a learned end slope (Heun/hypersolver).
6. **Gated DeltaNet:** 24 linear-attention layers in 9B with Mamba-2-style gates. Gate-saturation margins and state norms are candidate trust-region sensors, run as an ablation against random sketches.
7. Synthetic gates come first: WM-S must win the should-help task and show no gain on three should-not-help tasks, or it never reaches GPU.
8. Every fix is declared now, on fresh `wme-p3` states that do not exist yet.
9. My contingencies cost ≤ 3 local GPU-h (*computed*). Lambda waits until T2 and E2 pass on 9B and the claim is preregistered.
10. If Go-R2 and (b1) both fail, the synthetic leaves WM-E no mechanism on 9B. Accept "static suffices" and fund only the planning filter.

## Q2: SSM and memory structures

### Literature, mapped to our regime

- **Mamba** makes Δ, B, C "functions of the input". Δ balances "how much to focus or ignore the current input". On selective copying, S6 scores 97.0% against S4's 18.3% (https://arxiv.org/abs/2312.00752, https://arxiv.org/html/2312.00752v2). The gain is content selection over long sequences.
- **Mamba-2/SSD** ("scalar times identity" A, "2-8X faster"; https://arxiv.org/abs/2405.21060) and **S5** (MIMO diagonal, parallel scan; https://arxiv.org/abs/2208.04933) offer speed that is irrelevant at T ≤ 8.
- **S4WM / "Facing Off World Model Backbones"** (https://arxiv.org/abs/2307.02064): S4WM wins on long-term memory (Four Rooms MSE at 500 steps 44.0 vs RSSM-TBTT 219.4), a Transformer wins short-context recall, and "RNNs exhibit significantly higher imagination throughput."
- **R2I** puts SSMs into DreamerV3. It solves Memory Maze but is only "comparable … in classic RL tasks, such as Atari and DMC" (https://arxiv.org/abs/2403.04253).
- **Drama** (7M-parameter Mamba-2 world model): Atari100k mean/median 105/27 vs DreamerV3 125/49 (https://arxiv.org/abs/2410.08893); efficiency, not superiority.
- **DreamerV3 RSSM** has a prior p(ẑ_t | h_t), a posterior q(z_t | h_t, x_t), free bits "below the value of 1 nat" and symlog targets (https://arxiv.org/abs/2301.04104). The prior/posterior split is the part we need.
- **State tracking.** Diagonal SSMs stay in TC⁰ and cannot compose permutations (https://arxiv.org/abs/2404.08819); negative eigenvalues (https://arxiv.org/abs/2411.12537), RWKV-7 (https://arxiv.org/abs/2503.14456) and xLSTM's matrix memory (https://arxiv.org/abs/2405.04517) extend this. That expressivity serves *inferring* state; ours is bookkept, so it buys nothing for trajectories.

### Does memory help here, and where?

Not as memory. Our families are additive or W₀-anchored, and E_T is a sufficient statistic. Even current-weight actions, where AB ≠ BA, are captured by the bookkept E_T. What remains:

1. **Planning (E2), the strongest case.** Real evaluations of batches already paid for yield tap sketches and per-item losses at no extra cost. A DreamerV3-style posterior re-anchors the world model at the incumbent, and the R0 prior scores the children. The fair comparator is **anchor-shift static**, ŷ(θ_t + D) = y(θ_t) + F3u(E_t + D) − F3u(E_t). It sees the same observations but has no state.
2. **Saturation and normalisation on long same-direction ladders.** This is the inductive-bias case for WM-S.
3. **Hundreds of edits**, where ROME/MEMIT forgetting turns catastrophic "as early as 100 edits, or as late as 1000" (https://arxiv.org/abs/2401.07453; AlphaEdit's fix: https://arxiv.org/abs/2410.02355). Outside the WM-E bank; not proposed.
4. **Order effects:** with E_T supplied, static should tie (a should-not-help control).

### WM-S: a dose-time selective SSM (drop-in arm for `wm.model`)

**Edit descriptor u_t** (m = 24, CPU arithmetic on the recipe): family one-hot (6), layer/32, role one-hot (4), dose sign, and the field a_k, q_k of D_t for K = 6 objectives (12).

**Selection, conditioned on the edit:** emb = SiLU(W_u u_t); Δ_t = |dose_t|·softplus(w_Δ·emb); A_t = −softplus(W_A emb), diagonal, N = 32; B_t = sign(dose_t)·W_B emb; MLP readout.

**Recurrence (zero-order hold in dose-time):**

  h_t = exp(Δ_t A_t) ⊙ h_{t−1} + (exp(Δ_t A_t) − 1)/A_t ⊙ B_t

By construction: Δ = 0 leaves h exactly unchanged (zero edit); same-direction, same-sign steps s₁ then s₂ equal one step s₁ + s₂ (semigroup, so ladder split invariance); relaxation towards −B/A is exponential (saturation); diagonal A commutes, so order sensitivity enters only through the affine terms and is measurable.

**Shared with WM-E** (`src/wsp/wm/model.py`): h₀ = enc(o₀) at R0; the grounding decoder (training targets only); scheduled sampling; trapezoid slope transport Â_t = A₀ + H₀E_t + a_s[C(h_t) − C(h₀)] with gated correction ρ; the raw-field nest, `SHRINK_GRID`/`select_shrink`, 5 seeds.

If slope transport is dropped, WM-S uses the readout form F3u(E_t) + λ[g(h_t, xl_t) − g(h₀, 0)].

**Parameters** (*computed*: N = 32, width 64, d_xl = 10, K = 6):

| Component | Parameters |
|---|---|
| emb | 1,600 |
| Δ, A, B heads | 6,240 |
| readout | 9,350 |
| **SSM core total** | **17,190** |
| WM-E step + correction at the same widths | 21,414 |

Widths are matched to ±10% of total parameters. Encoder, decoder and slope heads are shared.

**Controls separating structure from memory:** **WM-S-static** (the same flows applied once, in canonical order, to E_T's per-direction coordinates, so a function of E_T only) and **WM-GRU** (a DreamerV3-style deterministic path).

**Fair test:** identical closure-component folds (`src/wsp/msrm/closure.py`), features, seeds, epochs and shrink rule; train h ≤ 2, test h = 3–4 (5–8 where available); C = 1 − Σ|e_arm|/Σ|e_ref| at h ≥ 3 with a 95% component bootstrap (4,000 draws, seed 20261107, as in `docs/reviews/msrm/msrm1-dev-v0-2026-11-16.md`). **Primary:** WM-S vs same-input static. **Secondary (Holm):** vs WM-E, WM-S-static, WM-GRU.

**Kill criteria:**
- C(WM-S : static) lower bound ≤ 0 → WM-S is dead.
- C(WM-S : WM-E) upper bound < 0.05 → WM-S is not preferred.
- On additive-only families, C(WM-S : WM-S-static) lower bound > 0 → a leak (θ_T is identical, so the "gain" is overfitting the path) → WM-S is withdrawn.

### Synthetic gates (CPU, ≤ 2 CPU-h, extend `src/wsp/wm/synthetic.py`, 5 world seeds)

**S+ (should help): long saturating ladders.**
- World: `SynthWorld("saturating", input_rank=2)`.
- Train on cumulative |dose| ≤ 2 units (h ≤ 2). Test on h = 3–8, up to |dose| 8, plus two-direction compositions.
- **Pass:** WM-S/static ≤ 0.85 in ≥ 4 of 5 seeds, and WM-S ≤ WM-E in ≥ 3 of 5.

**Should not help.** A gain/outcome-scale above 0.05 in any seed is a fail.
- **S−1, quadratic world.** WM-E gave exactly 1.00 here (P1 report).
- **S−2, sign-flip null.** Uses `shuffle_outcomes`.
- **S−3, non-commuting current-weight row scaling.** Every arm receives E_T. WM-S must not beat static or WM-S-static.

**S+filter (planning).**
- Set-up: 8 sequential real evaluations per episode; predict the children from the incumbent.
- **Pass:** posterior WM-S beats anchor-shift static with C ≥ 0.10 and lower bound > 0. Otherwise the filter is not built.

### Gated DeltaNet state features

The pinned snapshot `config.json` places full attention every 4th layer, so the 32-layer model has 24 linear-attention layers, each with 32 value heads of key/value dimension 128.

- **Recurrence:** S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ (https://arxiv.org/abs/2412.06464).
- **Gates in the HF code:** β = sigmoid(b) and g = −exp(A_log)·softplus(a + dt_bias), which is Mamba-2's −exp(A)·Δ. `torch_chunk_gated_delta_rule` returns the final state on request (installed `transformers/models/qwen3_5/modeling_qwen3_5.py`).
- **State size:** 32 × 128 × 128 = 524,288 floats per layer per prompt (*computed*).

**Tap** (24 × 32 × 5 = 3,840 scalars, seeded-projected to the sketch width): per head on 8 probe prompts, mean log-decay (memory half-life), mean β, the fraction of β/α pre-activations beyond ±4, ‖S_T‖_F, and the effective rank of S_T. **Why:** gate saturation is where the quadratic field fails, and A_log, dt_bias and conv1d are `not_linear` in the field collector (P1 report, P2). R0 uses base margins; R1 adds a JVP of gate pre-activations along D (charged). **Gate:** the D3 SAE rule, an ablation against equal-width random and PCA sketches; dropped if no gain above noise.

## Q1(b): the slope-transport gate fails

**Diagnosis (P1 report, P4).** Per-step oracle error was 1.54 SE (trapezoid), 2.47 (left-point) and 2.36 (static), so the raw, undamped K-FAC term ½eᵀH₀e predicts the within-step slope change *worse than omitting it*. The left-point cut is also negative on GSM8K (−0.154) and sycophancy (−0.745), both near noise (resolvable 0.33/0.40), with noisy 24-item A estimates.

**Literature.**
- **Hypersolvers** fit a learned term ε^{p+1}g_ω to a base step's local truncation error; HyperEuler is "roughly 88 times faster than dopri5 and with comparable accuracy" (https://arxiv.org/abs/2007.09601). That is WM-E's ρ; the learned end slope makes WM-E a Heun-type predictor–corrector.
- **Integrated gradients:** Σ IG = F(x) − F(x′), with 20–300 Riemann steps "within 5%" (https://arxiv.org/abs/1703.01365). Intermediate slopes are R2, so they are training targets only.
- **Tangent-space fine-tuning** "amplifies weight disentanglement" (https://arxiv.org/abs/2305.12827). It makes the field accurate, which helps planning but removes the trajectory contrast.
- **Steering vectors** give "the opposite behaviour for almost 50% of inputs" on some datasets (https://arxiv.org/abs/2407.12404), so per-item slope noise is real.

**Pre-declared fixes:**
- **(b1) R1 exact curvature.**
  - Replace ½eᵀH₀e with an exact HVP at θ₀ along e_t.
  - **Gate:** a left-point cut ≥ 20% on 12 fresh `wme-p3` steps.
  - **Cost:** ≈ 0.5 GPU-h (*computed*: 12 × 2 × the 61 s per state in P4).
  - Better curvature also improves the static field. Both are reported.
- **(b2) Learned end slope.**
  - **Gate:** the R0-predicted Â_t recovers ≥ 50% of the oracle trapezoid cut on held-out trajectories.
  - **Kill:** below 25%, drop slope transport.
- **(b3) Future Go-R counts resolvable step × objective pairs only** (|Δy| ≥ 2 SE). Declared now; not retroactive.

## Q5: contingency plan (adds to `RULING-wme-pilot1.md` §4; does not replace it)

| Order | Trigger | Action | Cost |
|---|---|---|---|
| 0 | Now, blind to outcomes | Build WM-S, WM-S-static, WM-GRU and the S+/S−1–3/S+filter synthetics. **All synthetic gates must pass**, or the SSM arms are withdrawn. | ≤ 2 CPU-h |
| 1 | Go-R2 fails | Run (b1) | 0.5 GPU-h |
| 1 | (b1) passes | Run (b2) on CPU | CPU |
| 1 | (b1) fails | Drop slope transport; retire the rollout-vs-static claim on 9B | — |
| 2 | T2 fails, E2 passes | Do not rescue T. Run 8 dev episodes × 8 evaluations on 9B: posterior WM-S vs anchor-shift static. **Gate:** C ≥ 0.10 and lower bound > 0. | ≈ 0.45 GPU-h (*computed*: 64 × 24.8 s) |
| 3 | T2 passes | WM-S joins the bank as a secondary arm | Inside the bank's 7.8 GPU-h (P1 report) |
| 4 | Any bank or planning run | Gated DeltaNet tap: an overhead run, then the ablation | ≈ 0.2 GPU-h |
| 5 | E2 fails on both 9B and 4B | No world-model work; return to the council | 0 |

**Total:** ≤ 3 GPU-h (*computed*: 0.5 + 0.45 + 0.2, plus 50% for reruns, plus ≤ 1 h of bank delta).

**Lambda.** 27B bf16 weights are about 54 GB (*computed*), so an 80 GB-class card is needed, but only after T2/E2 pass on 9B and the claim is preregistered. Nothing here needs it.

**Fetched (16 Nov):** all URLs cited above (arxiv.org/abs/<id>, HTML text for most; 2410.02355 abstract only), plus 1806.07366 (Neural ODEs, background). No failures; long HTML pages were truncated and partly summarised.

| proposal | mechanism | test | cost | kill criterion |
|---|---|---|---|---|
| WM-S dose-time selective SSM | Edit-conditioned Δ, A, B; zero-order hold in dose; exact zero-edit and split invariance; F3u nest | Train h ≤ 2, test 3–8 vs static, WM-E, WM-S-static, WM-GRU on identical closure folds | ≤ 2 CPU-h + bank share | Synthetic S+ fails or any S− gains; C(WM-S : static) LB ≤ 0 |
| Planning filter (RSSM posterior) | Take in real tap observations of evaluated states; prior scores children | S+filter; then 9B dev episodes vs anchor-shift static | ≈ 0.45 GPU-h | C < 0.10 or LB ≤ 0 |
| (b1) R1 exact curvature | HVP at θ₀ replaces K-FAC ½eᵀH₀e | Left-point cut on fresh `wme-p3` steps | ≈ 0.5 GPU-h | Cut < 20% |
| (b2) Learned end slope | Heun/hypersolver correction | Share of the oracle trapezoid cut recovered at R0 | CPU | < 25% |
| Gated DeltaNet tap | Gate-margin and state-norm sensors; R1 JVP margin crossing | Ablation vs equal-width random/PCA sketch | ≈ 0.2 GPU-h | No gain above noise |
