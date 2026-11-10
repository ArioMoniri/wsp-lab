# Round 3: representation (10 Nov 2026)

Data cite repo files; external claims cite URLs fetched 10 Nov. *Prior* = judgement.

## BLUF (10 lines)

1. LoL's LLM result (2,000 rank-4 K/V LoRAs, one Qwen2-1.5B sweep): ARC-C R² 0.987 for GL-net vs 0.983 from **singular values alone** (Table 5). A spectrum readout, which StatNN already has.
2. GL(r)-invariant functions of (B, A) are functions of BA. Field ⟨g, BA⟩ and chart P_outᵀ·BA·P_in are already GL-invariant with frozen frames; LoL adds only a **learned** frame.
3. That frame is unidentifiable here: a width-32 first layer has 9.2 M parameters on our adapters, against 60 labelled lora_sft rows.
4. Register a LoL family as **secondary** arms (A5.4.3 amendment before freeze): learned, random, K-FAC-anchored frames, plus a spectrum-only control. The primary is unchanged; no G7 funding.
5. Inputs: factors of the *realised* increment (truncated SVD at the recorded rank), with raw adapters as a sensitivity check. This covers 386 rows; chain LoRA steps exist only as dense snapshots.
6. On WikiText lora_sft holds 0.112 of the isotonic residual mass, chain 0.713: a LoRA-only arm cannot move the panel.
7. Best predictor: the registered tree on v4 plus field summaries. Secondary: an anchored GL-set head of about 1.6k parameters (field tokens plus whitened K-FAC blocks). *Prior:* screen band at best.
8. Part A is the sharpest LoL test: rotation preserves singular values, so the spectrum arm is blind by construction.
9. WeightLoop's δf = Σ δw_j h_j is our first-order contraction. Thickets concerns isotropic perturbations, as our random_dir nulls do. Neither changes A5, nor does 2512.01759.
10. GPU-h order: Part A (2.1 plus references), G2 (≤ 3), LoL family (≈ 0, CPU), G6 (4.9). Kill at 0.8B if Part A resolves with agreement upper bound ≤ 0.6 and every orientation contrast has upper bound < 0.05 at power ≥ 0.80.

## 1. What the news changes

| Item | Fetched fact | Change |
|---|---|---|
| WeightLoop https://arxiv.org/abs/2611.07319 | "Only the 128 entries of w"; θ_t = θ₀ + Δ_t; permutation controls; Edited beats Random 8/8, Original 7/8 (Table 1). δf(g) = s_y Σ δw_j h_j(x_g) needs cached activations; no predictor from Δ | None: it is our probe D·x / first-order field on a linear head |
| Neural Thickets https://arxiv.org/abs/2603.12228 | Improving-perturbation density rises with scale; "in small models such expert solutions occupy a negligible fraction of the volume" | None. For isotropic ε, ⟨g, ε⟩ ≈ 0 and only q ≥ 0 remains: the field predicts null or harmful, as our random_dir rows are. Silent on trained edits |
| LoL https://arxiv.org/abs/2410.04207 | §2 | Secondary arms by amendment (§3) |
| Critiques; item 6 | Doubt weights-only reading | None; our claim is "weights plus base sensitivity" (SYNTHESIS v3.1) |
| https://arxiv.org/abs/2512.01759 | GL(r) removed by a training-time asymmetric frozen mask (§3.4); reconstruction, generation, clustering | None. Our adapters were trained without the mask; `svd.py::canonical_lora_tokens` already canonicalises |

## 2. LoL against our data

**What LoL is** (https://arxiv.org/html/2410.04207 and the PDF):
- *Symmetry.* (U_i, V_i) → (U_iR_i, V_iR_i^{-T}).
- *Layers.* Equivariant linear maps (Φ_iU_i, Ψ_iV_i), learned **per layer** across the long axis (Prop. 1). The product-gated nonlinearity is "often" omitted.
- *Head.* An MLP on the U_iV_iᵀ, factors first reduced to "about 32 × r".
- *LLM data* (App. D.2, Table 8). 2,000 Qwen2-1.5B LoRAs, rank 4, "key and value projection matrices" only. Learning rate, weight decay, epochs, batch size, dropout and ARC source subsets are randomised.
- *Table 5.* ARC-C R²: MLP 0.107, O-Align 0.965, SVD 0.983, Dense 0.981, GL-net 0.987.
- *Not found:* split sizes, optimiser, early stopping, meaning of "±", and the "50,000 times faster" phrase (not used).

**Consequences.**
1. **The precedent is a spectrum result inside one hyperparameter sweep.**
   - Analogue of our descriptor-only ≈ StatNN (0.51/0.61 vs 0.56/0.66, `docs/reviews/P4/d3v2f-summary-2026-11-08.md`); StatNN has sv1–sv8 per tensor (`src/wsp/features/stats.py`).
   - GL-net wins clearly only on diffusion attributes (Table 3: 91.3 vs 77.3), with 2,046–3,900 LoRAs.
2. **The symmetry is already handled.**
   - GL(r) invariants of (B, A) are functions of BA. LoL's universality proof reduces GL-net to MLP + Dense.
   - Field: 1-D frozen-frame invariant; chart (`response_chart.py`, k = 4): k×k; M1: random frame (`svd.py::_proj`). LoL adds the **learned frame**.
3. **That frame is unidentifiable here.**
   - A stored rank-32 adapter has 144 factor tensors on 72 matrices: up/down in 24 layers, q/k/v/o in the 6 full-attention layers. The long dimensions sum to 288,768 (read from `data/zoo/qwen3.5-0.8b/*/lora/adapter_model.safetensors`).
   - Stored: 121 lora_sft (ranks 4/8/16/32: 35/26/34/26) plus 1 heretic_ara.
   - So a width-32 first layer has 9,240,576 parameters.
   - Labels: 60 lora_sft rows in 60 components on D3 TF-v1 (`docs/reviews/P5/h1w-gate-2026-11-06.md`). M1, given exact layer IDs, reached 0.45/0.50 against the tree's 0.56/0.66 (`d3v2f`).

## 3. Registered LoL family (secondary; A5.4.3 amendment before freeze)

**Inputs.**
- Per supported matrix: factors of the realised D_t = bf16(W_t + Δ_t) − W_t (A5.8.2), by truncated SVD at the recorded rank:
  - r for LoRA and chain LoRA steps;
  - 1 for random_dir/abliteration steps;
  - 32 plus a tail-energy token for task vectors.
- Raw α/r-scaled adapters (`src/wsp/zoo/apply.py`) are a sensitivity arm only: they are the nominal, not the scored, edit.
- Rows: all 386 G1 rows on the registered H1w folds, plus A5.4.3's LoRA-only subset. Only the 122 `zoo/*/lora/` adapters store factors; no chain factors exist.

**Frames (the only difference between arms).**
- **L, learned.** GL-net with Φ, Ψ shared within role across layers, width 4, plus layer/role embeddings. Up/down alone carry 36,864 frame parameters.
- **R, random.** Frozen Haar frames, k = 8, seed domain `20261110:lol-random:`. This is M1's frame in GL form.
- **K, anchored.** Frozen top-8 eigenvectors of the G2 WikiText K-FAC factors, whitened: M_t = (Λ_G^{1/2}Q_Gᵀ B_t)(A_t Q_A Λ_A^{1/2}). This is the top block of G^{1/2}DA^{1/2}, whose squared norm is 2q_t (A5.8.1).
- **S, spectrum.** log σ_1..8 and tail energy only; LoL's MLP + SVD.

**Head (shared).**
- Token: vec(M_t), log σ, tail, layer/24, full-attention flag.
- MLP 75→16→16 plus a role embedding; masked sum and mean over tensors; linear head per endpoint.
- Output f(D) − f(0), zero at zero edit as in A5.8.2.
- 1,617 parameters for one endpoint.

**Training.**
- L1 loss on the train-only nuisance residual, matching A5.4.2's absolute-residual sum.
- Heads: WikiText, with MMLU as a descriptive head.
- AdamW: learning rate 1e-3, weight decay 1e-2, batch 16, ≤ 300 epochs. Seeds 0/1/2 averaged before the loss.
- **Early stopping on development folds only.**
  - Four inner folds by ancestry component inside each outer training fold, patience 30.
  - The stopping epoch is the median of the inner best epochs; then refit on the outer training fold.
  - Outer held-out rows are never seen.
- A5.4.1 forbids early stopping for primary pipelines, so this applies to these secondary arms only. Their comparator, LoRA-only augmented StatNN with the vec(M_t) columns, gets the same inner choice of tree count from {100, 300, 600}.

**Fusion K + F.** Field tokens (⟨g_k, D_t⟩, q_t, c_t) join each K token before pooling.

## 4. What each arm identifies

| Arm / contrast (same rows, folds, seeds) | Identifies |
|---|---|
| Field only vs descriptor-only / no-field | First-order loss-anchored sign at θ₀ |
| L vs S and descriptor-only | Weights-only readability **beyond spectrum** (the original claim); L ≈ S is LoL's mechanism, not orientation |
| L vs R | Whether ~300 labels learn a relevance frame. M1 predicts not |
| K vs R | Base anchoring without linearisation (round-1 A2) |
| K + F vs field only | A nonlinear or second-order remainder beyond ⟨g, D⟩ |
| K + F vs K | Value of the gradient direction (absent from the eigenframe) |
| Every arm on Part A pairs (leave-component-out) | Orientation. S predicts 0 by construction, which counts as disagreement (A5 §4.1) |

Descriptive. If L beats S in Part B but not on the Part A sign, it learned hyperparameter regularities, not orientation.

## 5. Best predictor now; expected gain

**Primary (unchanged).** LightGBM on v4 plus within-role field summaries, against augmented StatNN (A5.4, A5.8.4). **Best secondary:** K + F.

**On main already.**
- g_k, K-FAC, `contract`, `cross_term` (`response_field.py`);
- frozen-frame blocks and slots (`response_chart.py`);
- canonical tokens (`svd.py`);
- the masked set encoder (`TensorEncoder`).

**To build.**
- An eigenframe projector source in `block_projectors`;
- a realised-increment SVD loader;
- the 1.6k-parameter head;
- nested early stopping;
- the S/L/R manifests.

**Expected gain.**
- Augmented StatNN gets every candidate scalar (A5.4.1), so gain can only come from combination, where a 1.6k-parameter head on 386 rows has no edge over 300 trees. *Prior:* C(K + F) in or below the 0.05–0.15 screen band on WikiText.
- WikiText residual mass is 0.713 chain in 41 components (`h1w-gate`), where the cross term is untested (G2 audit).
- On MMLU, lora_sft holds 0.339 of the mass and the tree reaches θ 0.230 [0.014, 0.400] (`docs/reviews/P5/h1a-companions-2026-11-06.md`): the likeliest cell for a signed field to add; descriptive while MMLU is unqualified.
- *Prior:* L does not beat S.

## 6. Next 30 GPU-hours (representation ranking)

1. **Part A as registered** (2.1 plus references). It is the only test separating orientation from spectrum, and it tests every arm here for free.
2. **G2** fields, factors and audit (≤ 3); field, K and K + F need them.
3. **LoL amendment and fits.** 0 GPU (CPU SVD).
4. **G6 closure** (4.9). Chain is where WikiText error sits. The rows may enter only the *training* folds of secondary arms, by its own amendment line, never the A_x rows.
5. **New matched-norm LoRA edits: G7 sealed blocks only**, each with its rotation. Per-adapter training cost is unmeasured in our files, so price it first.
6. **4B scale check last** (Thickets measures isotropic perturbations). If Part A is unresolved for yield, price a 4B field ladder on about 10 LoRA edits.

## 7. Kill criteria (0.8B line)

Stop and publish the measurement paper as final if **all** of these hold:
- (a) Part A has ≥ 15 resolved pairs with agreement upper bound ≤ 0.6, or the ladder alternative fires (A5 §4.0 step 2).
- (b) The field C_abl, K-vs-R and L-vs-S contrasts all have upper 95 % bounds < 0.05 at simulated power ≥ 0.80. Otherwise the result is unresolved.
- (c) L does not beat S on LoRA rows (paired upper bound ≤ 0).

Partial outcomes:
- (a) alone stops the field arm (A5). Part A unresolved for yield is not a kill; it moves the scale check up.

## Summary table

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---|---|
| A5 primary and Part A unchanged, Part A first | Rotation keeps spectrum, changes orientation | A5 §4.1: ≥ 15 resolved pairs, lower bound > 0.5 | 2.1 + refs; G2 ≤ 3 | Upper bound ≤ 0.6 on ≥ 15 pairs |
| LoL family L/R/K/S on realised-increment factors | GL-invariance ⇒ function of BA; the arms differ only in frame | A5.4.3 secondary, LoRA-only and 386-row folds, ratio intervals, no funding | ≈ 0 (CPU) | L vs S paired upper bound ≤ 0 |
| K + F anchored GL-set head (1.6k parameters, zero-at-zero, nested early stopping) | First-order field plus whitened curvature blocks | vs field-only and vs K, same intervals | ≈ 0 | Upper bound < 0.05 at power ≥ 0.80 |
| Part A sign read-out for all arms | S is blind by construction | Descriptive agreement on the 30 LoRA pairs | 0 | Arm ≤ S: not reading orientation |
| G6 rows in secondary training folds | Chain holds 0.713 of WikiText mass | Amendment; A_x rows unchanged | 4.9 | No change in secondary contrasts |
| New LoRA edits as G7 sealed blocks only | Fresh orientation contrasts | A5.6 | price first | A5.6 |
| Line kill at 0.8B | Orientation neither sets sign nor adds prediction | §7 (a)–(c) jointly | — | Underpowered = unresolved |
