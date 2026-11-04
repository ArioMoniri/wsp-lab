# Model candidates — Board C architecture assessment (P4)

## Cross-base-model transfer (H6)

### Honest assessment

Cross-base-model transfer is a materially harder problem than the existing held-out-edit-family H3 result. Fixed-width features are not by themselves architecture-agnostic: their coordinates can encode tensor shape, module naming, layer count, and architecture-specific parameterization. A random row split, or even a family split within one base checkpoint, cannot establish H6. The current representation can support an initial test, but zero-shot transfer across Qwen, LFM, and Gemma is a hypothesis rather than an expected consequence of pooling.

The current inputs divide into three classes:

- **Mostly transferable after dimensionless normalization:** scalar summaries such as log Frobenius norm, relative Frobenius change, mean/std normalized by an appropriate scale, sign fraction, nonzero fraction, normalized spectral entropy, stable rank, and singular-value energy fractions. They describe a tensor without requiring its coordinates to align. Current `delta_stats` also emits raw norms, raw singular values, and cosine-to-predecessor; these can encode parameter count, base scale, or model-specific history and should be logged separately or normalized. The current function computes only the top eight singular values, so its stable-rank estimate is a truncated-spectrum statistic, not full stable rank.
- **Fixed-width but silently model-specific:** `truncated_delta_tokens` uses seeded Gaussian projections whose input dimension is the tensor row/column size. Thus a `q_proj` of one shape and a `q_proj` of another shape use different projection matrices and their 64 coordinates have no shared basis, despite identical token width. The tokens are useful within compatible shapes, but are not directly comparable across widths. Canonical SVD removes LoRA factorization ambiguity; it does not align bases across unrelated matrices.
- **Pooling semantics are architecture-dependent:** `pool.py` hard-codes 15 types and name regexes, then sends any unmatched parameter to `other`. Qwen3.5 full-attention/MLP names, LFM3's GatedDeltaNet (`lin_qkv`, `lin_z`, `lin_b`, `lin_a`, `lin_state`), and Gemma-3's attention/MLP and possibly multimodal/norm vocabulary will therefore produce different occupancy and meaning per type. The same vector slot can mean a populated projection on one model and a zero/other aggregate on another. Four depth buckets are relative to `n_layers`, which is better than absolute layer indices but still assumes comparable functional depth and can smear different hybrid layer schedules together. This must be audited per model; do not treat tensor type as a universal semantic label.

The sequence-level relative-change feature `cos_prev_new` also needs care: it is defined inside each tensor's own coordinate space and can be meaningful for successive edits of that same base tensor, but it cannot align different bases. Prefer dimensionless change summaries computed per tensor and pooled by a shared, coarse functional role, with architecture-specific occupancy/missingness exposed explicitly. No activations are required by the current weight-statistics route; using deltas relative to each model's own base is the appropriate activation-free input, but it does not remove architecture mismatch.

### Candidate representations

1. **Dimensionless spectral and change descriptors (lowest-risk control).** Per changed matrix: log-normalized singular-value histogram/quantiles, singular-value energy fractions, spectral entropy, stable rank and a fitted heavy-tail exponent where the spectrum has enough support; plus relative Frobenius change, cosine to the predecessor delta, mean/std standardized by base-weight RMS, sign and sparsity fractions. Aggregate robust summaries (median/quantiles and changed-tensor mass) over broad roles such as attention, MLP, embedding, norm, recurrent/state-space, and unknown. Keep raw dimension and occupancy as separate nuisance covariates/diagnostics, not an implicit coordinate system. Heavy-tailed weight spectra motivate spectral summaries, but a power-law fit to a rank-8 spectrum is not credible: compute a longer spectrum or omit the exponent. This is activation-free and should be the first cross-model baseline.
2. **Activation-free delta embeddings.** Encode each `ΔW` through matrix-invariant sketches (e.g. normalized singular spectrum plus randomized trace/Frobenius probes whose outputs are defined by the sketch distribution rather than a fixed shape-specific basis), then pool across layers/roles with permutation-invariant set statistics. For task-vector/LoRA inputs, canonical low-rank factors can be used where shape-compatible; do not concatenate shape-seeded projections as if aligned. This tests whether the delta itself carries portable signal without requiring forward passes or base activations.
3. **Hypernetwork/metanetwork encoders.** GMNs represent networks as typed computation graphs and respect neuron permutation symmetries; recent transformer-specific GMN work extends the graph vocabulary to LM components. LoL demonstrates prediction from LoRA weights and reports transfer across models within a family, which is encouraging but does not establish transfer across Qwen/LFM/Gemma families. A metanetwork that consumes the architecture graph and weight deltas is the principled longer-term option, but its graph schema must cover hybrid recurrent layers and token embeddings and is likely too costly to make the first decisive P4b run.
4. **JEPA latents with per-model adapters.** Train a shared JEPA encoder/predictor on masked or paired dimensionless descriptors, with a small model-ID or architecture adapter and an explicit held-out-model objective. This can learn family-level edit semantics while adapters absorb type/depth calibration. A zero-shot claim must evaluate the shared encoder with the held-out adapter untrained or with a fixed metadata-only adapter; learned target labels belong only in the ≤20-edit calibration arm. JEPA cannot repair non-aligned random-projection coordinates unless the front end is changed first.

These are competing hypotheses, not established cross-family solutions. References checked against arXiv abstract-page titles with `curl -s https://arxiv.org/abs/<id> | grep -o '<title>[^<]*'`: 2610.13123, *Why Heavy-Tailed Weight Spectra Predict Model Quality*; 2410.04207, *Learning on LoRAs: GL-Equivariant Processing of Low-Rank Weight Spaces for Large Finetuned Models* (LoL); 2312.04501, *Graph Metanetworks for Processing Diverse Neural Architectures*; 2610.18471, *Graph Metanetworks for Transformer Based Language Models*; 2603.10090, *A Survey of Weight Space Learning: Understanding, Representation, and Generation*; 2512.01759, *Weight Space Representation Learning via Neural Field Adaptation*; 2603.15990, *W2T: LoRA Weights Already Know What They Can Do*; 2410.05016, *T-JEPA: Augmentation-Free Self-Supervised Learning for Tabular Data*; 2607.07127, *Weight-Space Physics: Interpretable Hypernetworks for Lattice Quantum Field Theories*; and 2610.30847, *Bifurcations from Weight Space: JEPAWG on Parametric Dynamical Systems*. The cross-architecture potential of GMNs is architectural motivation, not evidence that they already solve transfer for large hybrid LMs. The heavy-tail paper is likewise motivation for descriptors, not proof that delta spectra predict benchmark changes.

### Evaluation protocol and smallest decisive experiment

Use **leave-one-base-model-out** (LOMO): for each of three target bases, train and tune on labeled edits from the other two only; hold out every target-model edit and all its descendants from normalization, feature selection, pretraining, early stopping, calibration, and hyperparameter decisions. Freeze the exact edit-family mixture and model revisions before splitting. Report per held-out model, benchmark, and edit family; primary metric is Spearman ρ for predicted versus measured predecessor-relative `Δy`, with MAE as a secondary metric. Bootstrap by trajectory/edit source, not individual trajectory positions. Report zero-shot results separately from a calibration arm that may use at most 20 labeled edits from the target model; choose these calibration edits from training-visible target data, then score only disjoint target edits. Include an explicit `other`/unknown occupancy audit and feature normalization fit on training models only. The three-fold LOMO answers transfer among these three bases; it does not justify a universal architecture-transfer claim.

**Smallest decisive screening run:** Qwen3.5-0.8B, LFM3-1.7B, and Gemma-3-270M-it, one held out at a time. Use **n=100 labeled edits per model** (300 total; matched as closely as possible by edit family and evaluation protocol), reserving at least 40 held-out-model edits for zero-shot evaluation and a separate 20-edit calibration subset; remaining target rows are unused in that fold. If 100/model cannot be collected, report the actual n and treat the result as inconclusive rather than lowering the threshold. On at least six common benchmarks, zero-shot pass requires median per-benchmark Spearman ρ ≥ 0.40 across the three held-out models and ρ ≥ 0.20 on at least six benchmarks for each model; falsification for this screening design is ρ < 0.20 on all common benchmarks for every held-out model. The calibration arm passes if ≤20 edits raises the held-out model's median ρ by ≥0.10 without test leakage. These gates make the test intentionally harder than an average dominated by one friendly Qwen size; with small samples, report bootstrap intervals and call overlapping/unstable outcomes inconclusive.

Expected failure modes: (a) random projections mismatch shape and create arbitrary cross-model coordinates; (b) type-name vocabulary shifts push a model's useful tensors into `other`; (c) depth buckets conflate unlike layer schedules; (d) parameter count/base scale leaks through raw norms or spectra; (e) edit-family, tokenizer, benchmark, or instruction-tuning differences dominate the architecture effect; (f) hybrid recurrent state tensors have no analogue in dense attention models; (g) small n and noisy benchmark deltas produce rank instability; (h) calibration overfits if edits are not disjoint by source/family. Per-model adapters may improve calibration but do not count as zero-shot transfer.

### Board C recommendation

Add two competing P4b representation runs: **(A) dimensionless spectral/change set descriptors** as the low-cost, shape-agnostic baseline, with broad role pools, relative depth, explicit occupancy, and training-only scaling; **(B) the same descriptors into a shared Tabular-JEPA/JEPAWG encoder with per-model adapters**, evaluating the shared zero-shot representation and the ≤20-edit adapted arm separately. Defer raw W2T random-projection tokens as a cross-model representation until projections are demonstrably aligned, and defer GMN as a follow-up if A/B fail: it has the best architectural path to genuine heterogeneity, but data coverage and implementation cost make it a less decisive first comparison on this zoo. Current prior is modest: A should beat unnormalized pooled stats if leakage/mismatch matters; B may help only with sufficient cross-model edits. No claim that either will meet H6 is warranted before LOMO results.

Smallest decisive comparison is these two runs under the identical three-fold split above, against current pooled stats as an ablation. Keep all transforms train-fold-only, report ρ/MAE and 20-edit calibration gain, and include per-model confidence intervals. A representation wins only if it clears the zero-shot gate on each held-out model; improvement on average alone is not sufficient.

Reviewer-arch assessment of model families for predicting the performance vector y_t ∈ ℝ^K of an
LLM after each edit, from the sequence of weight deltas (Δθ_1..Δθ_t). Scope: about 600 edits per
base model, trajectories of length 1–8, and one 32 GB GPU. Written for P4 (3 Nov 2026),
against `docs/HYPOTHESIS.md` (H1–H4), `docs/ROADMAP.md` §P4, the M1–M4 specs, and the literature in
`docs/RESEARCH_NOTES.md`. Every arXiv id below was verified on 3 Nov 2026 by opening the abstract
page using the requested `curl -s https://arxiv.org/abs/<id> | grep -o '<title>[^<]*'` check.
The required records resolve to: 2607.07127 *Weight-Space Physics: Interpretable Hypernetworks for
Lattice Quantum Field Theories*; 2610.30847 *Bifurcations from Weight Space: JEPAWG on Parametric
Dynamical Systems*; 2608.13621 *Your Probabilistic JEPA Is Secretly a Hidden Markov Model*;
2610.02344 *Drive vs. Decay: On the Training Dynamics of Joint-Embedding Predictive Architectures*;
2603.15569 *Mamba-3: Improved Sequence Modeling using State Space Principles*; 2603.15990 *W2T:
LoRA Weights Already Know What They Can Do*; and 2603.10090 *A Survey of Weight Space Learning:
Understanding, Representation, and Generation*. Non-arXiv sources (LeCun 2022 OpenReview; LightGBM
NeurIPS 2017; Rasmussen & Williams 2006)
are cited without ids.

## What a candidate must fit

- **Data budget**: about 600 labeled edits and trajectories of length 1–8 per base model (0.8B first;
  4B/9B/27B later). K ≈ 10 targets (GSM8K, MATH-500, IFEval, MMLU-Pro, HellaSwag, WikiText-2 PPL,
  refusal, over-refusal, KL-to-base, loop rate). Splits 5-fold **by edit family and by seed** (P3
  protocol). This n is small by weight-space-learning standards: LoL finetuned *thousands* of LoRAs
  (2410.04207), Transformer-NFN released 125k checkpoints (2410.04209). Candidates that need >10⁴
  samples are dead on arrival.
- **Features already computed (P3)**: φ_stats (~40 spectral/norm floats × n_tensors) and φ_svd
  (W2T-style rank-8 canonical tokens, 129-d per rank component, random-projected to 64-d per side;
  GL(r) resolved at data level by QR→SVD, sign fixed by largest-|·| rule — `src/wsp/features/svd.py`).
- **Symmetries that matter**: GL(r) reparameterisation of LoRA factors (ΔW = BA ~ (BG, G⁻¹A));
  neuron and attention-head permutations; (u,v) ~ (−u,−v); positive rescalings (ScaleGMN's scaling
  symmetries). Canonicalisation (W2T, JEPAWG) vs architectural equivariance (GMN, NFN) is the live
  design choice.
- **Sequence structure**: T ≤ 8, so *attention is not yet expensive* (64-token attention is trivial)
  and constant-memory recurrences have no cost advantage. The H2 claim is that y_t depends on
  accumulated drift and interference: task-vector composition is non-additive and order/format
  dependent (2608.09490), which is exactly the "rotational"/state-tracking dynamics that real-valued
  selective SSMs cannot represent (Mamba-3's TC0 argument, 2603.15569 §2.5). Short-horizon weight
  motion is predictable only in pockets (2608.15483), so models must tolerate noisy labels.
- **Compute**: one RTX PRO 4500 (32 GB, CUDA 13.2). All candidates below are ≤ 20M parameters on
  precomputed features; a training run is 10–60 min. `mamba-ssm` 2.3.2 kernels may not compile on
  cc 12.0; `ssm_torch.py` already carries a pure-torch selective-scan fallback.

## Summary table

| # | Family | Models what | Bias match | Gain vs M1–M4 | Cost | Risk |
|---|---|---|---|---|---|---|
| 1 | Trajectory Transformer (W2T tokens + cross-edit attention) | joint set-of-tensors × sequence | exact: GL(r) by canonicalisation, interference by attention | H2 without state bottleneck; ≥ M2 | ~1 h/run | low (overfit at T≤8, n_traj=150) |
| 2 | Tabular-JEPA + ResidualPred pretrain | latent of φ_stats/φ_svd, no augmentations | exact: features are tabular; anti-collapse init | H3 transfer +2–5 ρ pts vs M3 | < 1 h | low |
| 3 | Graph metanetworks / GL(r)-equivariant (GMN, LoL) | deltas as symmetry-structured graphs | exact by construction (GL(r), head perm, scale) | +10–20 % MAE *if* rank-8 truncation is the bottleneck; P5 transfer bet | 2–4 h + engineering | medium (n=600, W2T beat GLNet on ARC-LoRA) |
| 4 | Mamba-3 block swap (M2 upgrade) | complex-valued selective state | tracks rotational interference Mamba-2 cannot | +5–10 % MAE on merge/interference trajectories | 1–2 h | medium (kernel on cc 12.0) |
| 5 | Stochastic latent dynamics (RSSM-lite / Var-JEPA) | distribution p(y_t \| history) | multimodality + calibration | CRPS/calibration, point parity with M2 | ~2 h | medium |
| 6 | xLSTM (sLSTM/mLSTM) | gated scalar/matrix memory | parity-solvable gating (per 2603.15569 TC0 note) | parity with M2 expected | < 1 h | low, low ceiling |
| 7 | TTT layers / Titans | test-time-adapting memory | mismatch at T≤8 | none expected now | 2–6 h | high (inner-loop learning at n=600) |
| 8 | Diffusion / flow over weights | distribution of weight populations | generation, not prediction | none; infeasible at n≈600 | — | fatal (data scale) |
| 9 | DWSNets / DeepSets / Set Transformer | rank components and tensors as sets | permutation-invariance at token level | subsumed by W2T tokens; degenerate-σ robustness only | < 1 h | low, low ceiling |
| 10 | Non-neural: LightGBM (StatNN) / GP on delta norms | direct feature→y map | none learned; spectral stats justified by 2610.13123 | sets the floor + calibrated CI | minutes, CPU | none |

M1 (stateless per-tensor token Transformer) and M4 (LLM reader) are unaffected by this ranking: M1
is family 1 with the sequence axis ablated; M4 is a different hypothesis (H4).

## Family assessments

### 1. Plain Transformers over canonical tokens, extended to trajectories (W2T-style)

**What it models.** Each Δθ_t (or cumulative θ_t − θ₀) becomes rank-wise tokens [σ_k, proj(u_k),
proj(v_k)] + layer/module-type embeddings; a position-level Transformer attends across tokens and
tensors; attention pooling yields a checkpoint embedding (W2T, 2603.15990). The natural P4
extension — and our M1 → M5 candidate — is one more attention axis over the edit sequence
(cross-edit attention), i.e. a small (T·|tokens|)-length attention model with causal masking over
edits and full attention within an edit.

**Inductive-bias match.** Exact where it counts: GL(r) and sign symmetries are removed *at the data
level* by QR→SVD canonicalisation (provably identical tokens for all factorisations), head
permutations are broken by SVD re-sorting, and full attention within T ≤ 8 represents arbitrary
pairwise interference — including the non-commuting compositions (2608.09490) and second-pass
effects H2 is about. No state bottleneck: an SSM must compress the past into d_state; attention
need not.

**Expected gain vs M1–M4.** W2T's own numbers set the single-edit bar: ARC-LoRA test MAE 0.32,
Spearman 95.35 (vs GLNet 0.43 / 94.16 — canonicalisation+Transformer beat the equivariant
metanetwork on their benchmark). M1 already inherits this. The untested part is the sequence axis:
expected +1–3 ρ points and the H2 gate (≥ 15 % relative MAE reduction on T ≥ 4 vs stateless) with
equal or better numbers than M2, because interference at T ≤ 8 is more than a fixed-size state can
hold. Risk of overfitting is real: 150 trajectories × T points; mitigate with ≤ 2 layers, family-split
early stopping.

**Compute.** Same as M1 (tiny); ~1 h per run including 5-fold by-family and by-seed splits.

**Risk.** Low. Everything except the cross-edit mask is already implemented (φ_svd, M1).

**References (verified).** W2T 2603.15990 (Mar 2026); survey 2603.10090 §WSR; finite-probe
identifiability (probes can be small) 2609.33901 (Sep 2026); heavy-tailed spectra justify σ-stats
in tokens 2610.13123 (NeurIPS 2026).

**Smallest decisive experiment.** n ≈ 150 trajectories (the available ~600 edits, with length 1–8;
evaluate the T ≥ 4 subset and report its actual count), 5-fold by family; metric: Δy MAE (points) on held-out T ≥ 4 trajectories + per-
benchmark Spearman ρ; thresholds: (a) ≥ 15 % relative MAE reduction vs stateless M1 (H2 gate),
(b) ≥ M2 everywhere (wins the M2/M5 competition), (c) if within 2 σ bootstrap of M2, the SSM state
is sufficient and attention is unnecessary — either outcome is decisive for the paper.

### 2. JEPA variants: JEPAWG two-stage (M3), Tabular-JEPA + ResidualPred, I-JEPA/V-JEPA 2, Var-JEPA

**What they model.** A latent space shaped by predicting embeddings of related inputs, never
reconstructing inputs. JEPAWG (2607.07127): weight encoder E_θ (MLP, width 128, dz = 16), coupling
encoder, predictor, VICReg *without* the covariance term, trained two-stage — freeze (E_θ, E_g, P),
then train the decoder; canonicalises basis/permutation/sign symmetries before encoding. On
dynamical-systems zoos (2610.30847) JEPAWG is competitive with an end-to-end hypernetwork on
forecasting despite never seeing the forecast loss, and its decoded-weight derivative locates
bifurcations — qualitative regime changes read from weights alone. T-JEPA (2410.05016, ICLR 2025)
makes JEPA work on tabular data with *no augmentations*: predict the latent of one feature subset
from the complement. Drive-vs-Decay (2610.02344, NeurIPS 2026) gives the collapse theory (per-mode
stability ratio µ = γ/σ; > 800 Tabular-JEPA configurations mapped) and the fix: **ResidualPred**,
an identity-biased predictor (W_p = I + ΔW_p, diagonal attention bias β ≈ 10 at init), which raised
linear-probe accuracy on tabular (Helena 8.6 → 21.3, Jannis 43.4 → 46.9) and I-JEPA image
pretraining (CIFAR-100 36.83 → 40.85, IN-100 37.09 → 42.73).

**Inductive-bias match.** Strong, and stronger than M3's current spec admits: our φ_stats is
literally a tabular feature matrix and weight-space has no crop/flip augmentations, so T-JEPA's
subset-mask views are the *right* views; ResidualPred is the missing stabiliser for M3's small-n
regime (dz = 16 latent, ~600 samples is exactly the brittle regime where standard predictors
collapse); JEPAWG's two-stage recipe (shape the latent with the JEPA loss only, then read it out)
is already M3's design and its bifurcation result is the direct analog of "detect the edit family
and capability cliff from Δθ". I-JEPA (2301.08243) / V-JEPA 2 (2506.09985) contribute masking
strategy (large semantic targets, distributed context) and, via V-JEPA 2-AC, evidence that latent
rollouts support planning — the P6 surrogate argument. Var-JEPA (2603.20111, ICML 2026) derives a
JEPA from an explicit ELBO, giving principled latent uncertainty without ad-hoc anti-collapse terms.

**Expected gain vs M1–M4.** In-distribution, JEPAs will at best match W2T-style supervised heads
(labels are available; JEPA is a *pretraining* bet). The expected gain is H3: +2–5 ρ points
transfer to a held-out edit family (e.g. train on abliteration + random-direction, test on LoRA),
because the latent is shaped by Δθ↔Δθ predictability rather than by benchmark labels. ResidualPred
additionally de-risks M3 (keeps effective rank up; every-seed-above-every-seed in their ImageNet
pilot).

**Compute.** < 1 h per run on 32 GB (MLPs on precomputed features; JEPAWG trained "in minutes" on
19,689-d weights on one A100).

**Risk.** Low. Collapse is the historical failure mode; ResidualPred + VICReg-variance + ERank
monitoring (2608.13621-style diagnostics) make it detectable and preventable.

**References (verified).** 2607.07127; 2610.30847 (NeurIPS 2026 workshop); 2610.02344;
2410.05016; 2603.20111; 2301.08243; 2506.09985; JEPA-as-HMM/MC-JEPA 2608.13621 (see family 4);
LeCun 2022 (OpenReview, no id).

**Smallest decisive experiment.** n ≈ 480 of the available 600 edits from 4 families, hold out 1
family per fold (rotate across 5 folds; report all family counts);
metric: transfer Spearman ρ per benchmark on the held-out family + latent effective rank (ERank)
during training; thresholds: ρ ≥ 0.5 on ≥ 6 of K benchmarks (H3 gate), ERank(ResPred) ≥ 2×
ERank(standard predictor) with zero collapsed seeds; ablate two-stage vs joint decoder (JEPAWG
Table 1 says two-stage wins by 0.88 vs 0.61 ESS analog).

### 3. Latent world models: RSSM (DreamerV3), TD-MPC2 latent dynamics, H-JEPA, MC-JEPA

**What they model.** A recurrent/stochastic latent state summarising history, plus (usually) a
decoder or reward heads. DreamerV3 (2301.04104): RSSM with deterministic + stochastic state, KL
balancing, one config across 150+ domains. TD-MPC2 (2310.16828): decoder-free implicit latent
model, planning inside the latent, 104 tasks / one hyperparameter set. LeCun 2022's H-JEPA:
hierarchical latent predictors, uncertainty handled per level. MC-JEPA (2608.13621) proves the
bridge relevant to us: temporal JEPA ≡ HMM (filter = context encoder, transition = predictor,
emission = decoder/inverse encoder); a *learned transition matrix* on latents gives exact
Chapman–Kolmogorov multi-horizon consistency.

**Inductive-bias match.** Partial. These are built for T ≈ 100–10⁹ environment steps; our T ≤ 8
makes the recurrence itself trivial (a linear map suffices — MC-JEPA's point). What they add that
M1–M4 lack: a *stochastic* state, i.e. a distribution over y_t rather than a point estimate —
edits have multimodal outcomes (an abliteration pass either removes the direction or cripples the
model), and KL-balanced posteriors give calibrated intervals that the surrogate search in P6
needs. TD-MPC2's decoder-free planning is the honest architecture for P6 (choose next edit by
imagining in latent space) without ever decoding 0.8B weights.

**Expected gain vs M1–M4.** Point predictions: parity with M2/M5 expected (the deterministic part
of an RSSM at T ≤ 8 is a fancy GRU). The gain is calibration: 90 % interval coverage / CRPS better
than a GP on Δ-norms, at equal MAE — which is what makes the P6 Bayesian-surrogate claim credible.

**Compute.** ~2 h per run (DreamerV3-lite on precomputed features; the full DreamerV3 machinery —
world-model, actor, critic — is unnecessary, keep only the RSSM + observation heads).

**Risk.** Medium: stochastic latents at n = 150 trajectories can collapse to the prior (free bits
mitigate); more moving parts than families 1–2 for no point-prediction gain.

**References (verified).** 2301.04104; 2310.16828; 2608.13621; LeCun 2022 (OpenReview); V-JEPA 2
action-conditioned rollout 2506.09985; short-horizon predictability pockets 2608.15483.

**Smallest decisive experiment.** n = 150 trajectories, 5-fold by family; metrics: CRPS and 90 %
empirical coverage of y_t + MAE; thresholds: ≥ 10 % CRPS improvement vs GP-on-delta-norms while
point MAE within 2 % of M2; if coverage < 80 %, stochastic latents are not earning their cost.

### 4. SSMs: Mamba-2 (M2), Mamba-3, S5

**What they model.** Input-dependent (selective) linear recurrences; M2 stacks Mamba-2 blocks
(2405.21060) over the edit sequence — our current M2. Mamba-3 (2603.15569, ICLR 2026) upgrades the
recurrence three ways: exponential-trapezoidal discretisation, a **complex-valued state update**
(equivalent to data-dependent rotary), and MIMO formulation; it solves state-tracking tasks
(parity-class) that real-valued SSMs provably cannot, at +1.8 downstream points vs Gated DeltaNet
at 1.5B and matching Mamba-2 perplexity at half the state. S5 (2208.04933) is the minimal MIMO +
parallel-scan baseline.

**Inductive-bias match.** The selective decay is a good story for H2 (accumulated drift decays or
persists per feature channel), *but* the interference we care about — non-additive, order-dependent
task-vector composition (2608.09490) — is rotational dynamics, and Mamba-2's real-valued state
cannot represent it (2603.15569 §2.5: parity ∈ TC0 needs complex/rotational state; a one-layer
LSTM also suffices). Mamba-3's complex state fixes exactly this. S5's non-selective recurrence is
the wrong bias (edits are not time-translation invariant) and exists here only as a control.

**Expected gain vs M1–M4.** M2 → Mamba-3: +5–10 % relative MAE on the merge/interference-heavy
trajectory subsets (families 5–6 of the zoo), parity elsewhere; the honest caveat is that at T ≤ 8
full attention (family 1) already represents everything an SSM can and more, so Mamba-3's edge only
survives if M5 fails experiment (a) above. Mamba-3's real long-term case is P6 long-horizon rollouts
(T ≫ 8: repeated abliteration passes, heretic trial chains of 50+).

**Compute.** 1–2 h per run. Kernel risk: `mamba-ssm` 2.3.2 ships Mamba-2 kernels; Mamba-3 kernels
may not exist for cc 12.0 / CUDA 13.2 — a pure-torch complex selective scan at T ≤ 8 is ~50 lines
on top of `ssm_torch.py` (the per-step loop is already there).

**Risk.** Medium: availability/compile risk on Blackwell; complexity of a second sequence model
competing for the same 150 trajectories.

**References (verified).** 2603.15569; 2405.21060; 2208.04933; task-vector interference 2608.09490.

**Smallest decisive experiment.** n = 150 trajectories; metrics: H2 relative MAE (T ≥ 4) + an
order-swap probe (predict Δy(A∘B) vs Δy(B∘A) where the two differ by > 1σ; report swap-detection
accuracy); thresholds: ≥ 15 % relative MAE reduction vs stateless M1 *and* ≥ 5 % over Mamba-2 M2
*and* ≥ 90 % swap-detection — else keep M2 and spend the effort on family 1.

### 5. Neural functionals and graph metanetworks: GMN, ScaleGMN, GMN-for-LMs, LoL/GLNet, NFN, UNF, NFT, Transformer-NFN

**What they model.** Networks that eat other networks' weights with the symmetries built into the
architecture. NFN (2302.14040, NeurIPS 2023): permutation-equivariant NF-layers for MLP/CNN weight
spaces (predicts generalisation from weights). UNF (2402.05232): *automatically constructed*
equivariant functionals for any architecture incl. residual/recurrent. NFT (2305.13546): attention
inside the equivariant layers. Transformer-NFN (2410.04209, ICLR 2025): the maximal symmetry group
of multi-head attention weights + a 125k-transformer-checkpoint benchmark. GMN (2312.04501): cast
the input net as a graph, message-pass; handles attention, norms, ResNet blocks; ScaleGMN
(2406.10685, NeurIPS 2024) adds positive-rescaling symmetries and can simulate the input net's
forward/backward pass. **GMN for transformer LMs (2610.18471, Oct 2026)** formalises the full
symmetry group of decoder-only LMs (GL of attention matrices, head permutations, token embeddings)
and beats baselines on fine-tuning detection and training-corpus prediction — the closest 2026
prior work to "audit an LM from weights alone". Transferable GMNs (2610.00420, Sep 2026) make
predictions transfer across input widths (up to 42× under μP) — the P5 transfer story
(0.8B → 27B). LoL (2410.04207) is the low-rank specialisation: GL(r)-equivariant processing of
LoRA factors via canonicalisation, invariant featurisation, or equivariant layers, predicting
downstream accuracy of finetuned models.

**Inductive-bias match.** Architecturally exact for every symmetry we listed. The catch is what
our current pipeline throws away before any model sees the data: φ_svd truncates at rank 8 and
random-projects directions to 64-d. An equivariant per-tensor encoder consumes the full factors
losslessly. That is the one concrete, testable information-loss hypothesis in this family.

**Expected gain vs M1–M4.** Bounded by the W2T-vs-GLNet comparison (2603.15990 Table 2: MAE 0.32
vs 0.43) — equivariance did not beat canonicalisation+Transformer at LoRA scale, so we should
*not* expect an in-distribution win over M1/M5. The two places GMNs can still win: (a) if rank-8
truncation is the bottleneck (gain +10–20 % MAE), (b) cross-width transfer for P5 (2610.00420), and
(c) raw merged deltas (not LoRA-factorable) where φ_svd's canonicalisation is a lossy truncation
but a graph sees the whole Δθ.

**Compute.** 2–4 h per run + real engineering: graph construction for Qwen3.5-0.8B (~200 tensors;
per-neuron graphs are too big, per-tensor/per-head graphs are the sane granularity). Memory fine
on 32 GB.

**Risk.** Medium-high: at n = 600, equivariant nets overfit less than MLPs but still need the
augmentation tricks (2311.08851); W2T already won this comparison once.

**References (verified).** 2312.04501; 2406.10685; 2610.18471; 2610.00420; 2410.04207; 2302.14040;
2402.05232; 2305.13546; 2410.04209; data augmentation for weight spaces 2311.08851; survey
2603.10090.

**Smallest decisive experiment.** Step 1 (1 day, no training): measure the information ceiling of
truncation — regress y on rank-r tokens for r ∈ {4, 8, 16, 32} with the M1 head; if MAE(r=32) <
MAE(r=8) by ≥ 5 %, truncation is a real bottleneck and the GMN run is justified. Step 2 (if
justified): n = 600 single edits, 5-fold by seed, per-tensor GL(r)-equivariant encoder (LoL recipe)
vs φ_svd+M1; thresholds: ≥ 10 % relative MAE reduction on ≥ 6 of K benchmarks + invariance unit
test (predictions change < 1e-4 under (B,A) → (BG, G⁻¹A) and head permutation).

### 6. Permutation/scale-equivariant set encoders: DWSNets, DeepSets, Set Transformer

**What they model.** Sets of tokens with pooling/attention that is permutation-invariant by
construction (1703.06114; 1810.00825) or affine-equivariant layers built from pooling +
broadcasting + FC (DWSNets 2301.12780, ICML 2023).

**Inductive-bias match.** Good at the wrong level: after canonicalisation our rank components are
already ordered by σ, so positional encodings are meaningful; the one residual symmetry is
near-degenerate singular values (unordered), where a set encoder is strictly more correct than a
position-level Transformer, and head/tensor sets are unordered.

**Expected gain vs M1–M4.** Marginal: W2T's attention pooling already implements a Set-Transformer
readout. Expected gain ≤ 1 ρ point; keep as a robustness ablation, not a run.

**Compute / risk.** < 1 h; low.

**References (verified).** 2301.12780; 1703.06114; 1810.00825; ScaleGMN 2406.10685 for scale.

**Smallest decisive experiment.** n = 600 single edits; inject σ-degenerate pairs (shuffle token
order where σ_k ≈ σ_{k+1}); metric: prediction variance under shuffle; threshold: set-pooling
reduces shuffle-induced variance ≥ 5× at ≤ 1 % MAE cost, else keep positional W2T.

### 7. Test-time-training layers / Titans

**What they models.** TTT (2407.04620): the hidden state is itself a learner updated by
self-supervised gradient steps on each test sequence. Titans (2501.00663): a neural long-term
memory with surprise-based writes, attention as short-term memory, scaling to > 2M context.

**Inductive-bias match.** Poor for now. Both are long-context machines; our context is 8 steps.
The TTT inner loop must learn "how this trajectory's edits behave" from ~8 examples of
(Δθ, Δy) at test time — there is not enough in-trajectory signal at K ≈ 10 targets.

**Expected gain vs M1–M4.** None expected at T ≤ 8; revisit at P6 if trajectories extend to
50+ heretic-trial chains, where memorising surprising edits (capability cliffs) becomes the right
operation.

**Compute / risk.** 2–6 h to wire in; high risk of a null result.

**References (verified).** 2407.04620; 2501.00663.

**Smallest decisive experiment.** Only if P6 lengthens trajectories: n = 50 long trajectories
(T ≥ 32); metric: MAE on the last 25 % of steps; threshold: ≥ 10 % over M5 with the same budget —
else drop permanently.

### 8. xLSTM

**What it models.** sLSTM: scalar memory with exponential gating and new memory mixing; mLSTM:
parallelisable matrix memory with covariance update rule (2405.04517).

**Inductive-bias match.** The Mamba-3 TC0 note says a one-layer LSTM already solves parity-class
rotational dynamics — so sLSTM's gating is, in principle, the cheap fix for the same interference
representational gap as Mamba-3, and "forget after a merge/reset edit" is literally a gate.

**Expected gain vs M1–M4.** Parity with M2 expected (± 3 % MAE); it is a drop-in control that
separates "gating helps" from "complex state helps" if Mamba-3 wins.

**Compute / risk.** < 1 h (xlstm repo); low.

**References (verified).** 2405.04517; 2603.15569 §2.5.

**Smallest decisive experiment.** Same harness/splits as M2; threshold: adopt as a reported
baseline only if within 2 % of M2's MAE (it is a control, not a candidate).

### 9. Diffusion / flow models over weights

**What they model.** Generative distributions over weight space: HyperDiffusion (2303.17015)
diffuses over INR-MLP weights; the 2025–2026 wave (survey 2603.10090 §WSG: FLoWN, NeuroGen, T2W,
ORAL, D2NWG, GeoSANE, BEND, p-diff) generates weights or adapters, generally from ≫ 10³ checkpoints.

**Inductive-bias match.** Mismatch: we need conditional *prediction* of a 10-d y from a sequence,
not unconditional generation of 10⁹-d weights. A conditional diffusion head over y_t given history
is possible but a GP does that with calibrated uncertainty at n = 600 for free.

**Expected gain vs M1–M4.** None for P4. The only live variant: generative *editing* (sample the
next Δθ) — that is P6/non-goal territory.

**Compute / risk.** Fatal at n ≈ 600 (HyperDiffusion needed hundreds of INRs for 2k-d weights;
ours would be 10⁶-d weights at 3× fewer samples).

**References (verified).** 2303.17015; 2603.10090; position paper 2605.18632.

**Smallest decisive experiment.** None scheduled; revisit only if the zoo grows to ≥ 10⁴ edits per
family (not in this roadmap).

### 10. Non-neural baselines: LightGBM on spectral stats (StatNN), GP on delta norms

**What they model.** LightGBM (Ke et al., NeurIPS 2017, no arXiv) on φ_stats: boosted trees over
~40 × n_tensors spectral/norm features — the H1 "weight-statistics baseline". GP (Rasmussen &
Williams 2006, no arXiv) with RBF + ARD kernel over ~10 delta-norm features per step: a calibrated
regressor with n = 600 well inside its regime, and the honest uncertainty floor that H2's "within
noise" falsification line must be measured against. A trajectory GP (kernel over concatenated
per-step norm vectors) is the non-neural H2 control.

**Inductive-bias match.** None learned — and that is their value: heavy-tailed weight spectra
carry quality signal (2610.13123), trees capture threshold effects (a small LoRA merge does
nothing until rank > r*, then breaks the model), the GP gives per-benchmark CIs the neural models
must beat to claim anything.

**Expected gain vs M1–M4.** Negative by design (floor-setters), except two genuine possibilities:
LightGBM beats M1 on the smallest families (trees at n < 200 are hard to beat); the GP's
calibration beats the deterministic nets' intervals (they have none) — both are paper-relevant
sentences, not P4 deliverables.

**Compute / risk.** Minutes on CPU; none.

**References (verified).** 2610.13123; survey 2603.10090; TuneAhead 2606.17660 (dataset+probe
outcome prediction, RMSE 1.47 pt — the "you don't need weights" foil); probing 2604.01025.

**Smallest decisive experiment.** Already P3's gate; add only the trajectory GP for the H2 control:
n = 150 trajectories, metric: MAE(T ≥ 4) vs its own T = 1 fit; threshold: if the trajectory GP
captures ≥ 10 % of the M2 gain, part of H2 is cheap statistics, not sequence modelling — a
confound we must report.

## Ranked top-5

1. **M5 — trajectory Transformer (W2T canonical tokens + cross-edit attention).** Cheapest
   decisive test of H2; the only family whose bias covers *arbitrary* interference at T ≤ 8;
   W2T's numbers say the token side is already competitive. Risk: overfit at 150 trajectories.
2. **M6 — Tabular-JEPA pretraining with ResidualPred (T-JEPA recipe on φ_stats/φ_svd, linear/MLP
   head → Δy).** The JEPA variant actually matched to tabular weight features; ResidualPred is the
   2026 answer to the collapse that M3's spec is exposed to; the H3 transfer bet.
3. **Per-tensor GL(r)-equivariant / GMN encoder (LoL 2410.04207 recipe; GMN-for-LMs 2610.18471
   symmetry group).** The only candidate that removes φ_svd's rank-8 truncation and 64-d random
   projection information loss; conditional on the rank-r ceiling probe showing truncation matters;
   also the P5 cross-width transfer bet (2610.00420).
4. **Mamba-3 block swap for M2.** Complex state tracks the rotational interference dynamics that
   Mamba-2 provably cannot; worth one run if (and only if) M5 does not already take H2.
5. **RSSM-lite / Var-JEPA stochastic latent over the edit sequence.** For calibrated p(y_t |
   history): the P6 surrogate needs intervals; no point-prediction gain expected.

Per `docs/REVIEW_PROTOCOL.md` §Board C, the top-ranked untested candidate (M5) must be scheduled as
a competing P4 run before the phase closes; M6 competes on the H1/H3 axis.

## Two picks to add as competing P4 runs

**M5 (compete vs M2, hypothesis H2).** Per-tensor-token Transformer with a second, causally-masked
attention axis over edits; input tokens = φ_svd(Δθ_t) ⊕ edit-descriptor embedding; output signed Δy_t ∈
ℝ^K at every step (linear head). ≤ 2 layers, d_model 256,
dropout 0.2, family-split early stopping. Acceptance: ≥ 15 % relative MAE reduction vs stateless
M1 on held-out T ≥ 4, and ≥ M2 on all splits (bootstrap over edits, 5 seeds). Either M5 or M2
winning is a publishable H2 result; a tie means the SSM state is sufficient at T ≤ 8.

**M6 (compete vs M3, hypotheses H1/H3).** Stage 1 (unsupervised): T-JEPA subset-mask views over
φ_stats columns (mask ≈ 50 % of the ~40 × n_tensors features, predict the latent of the rest) with
ResidualPred predictor (W_p = I + ΔW_p, diagonal attention bias β = 10 at init), VICReg variance
term, dz = 32; monitor ERank. Stage 2 (frozen encoder): linear + 2-layer MLP head → Δy. Acceptance:
H3 transfer ρ ≥ 0.5 on ≥ 6 of K benchmarks only for the explicit `--holdout lora_sft` test family; ERank ≥ 2× the
standard-predictor ablation with zero collapsed seeds. Same 5-fold family splits as M3.

## What would change this ranking

- Rank-r ceiling probe (family 5, step 1) shows truncation is *not* the bottleneck → GMN drops
  below xLSTM; M5/M6 stand.
- M5 ties with M2 → Mamba-3 rises to #2 for P6 long-horizon rollouts; complex state becomes the
  only in-family upgrade path.
- Trajectories extended beyond T ≈ 32 (P6) → Titans/TTT and Mamba-3 rise; attention stays viable
  to T ≈ 10³.
- Zoo scaled past ~10⁴ edits per family → diffusion over weight deltas becomes testable; not in
  this roadmap.
- 27B transfer (P5) becomes the headline → Transferable-GMN-style width invariance (2610.00420)
  moves to #1.

## LLM-in-the-loop editing and optimisation

This assessment answers H5 before any P6 run. The three proposed roles are distinct: **A** an LLM
直接 proposes a weight delta; **B** an LLM proposes optimizer updates or learning-rate-like controls during
short fine-tuning; **C** the existing predictor forecasts outcomes and ranks or rejects proposals before
the expensive benchmark evaluation. C is a selection layer, not an edit generator. The relevant
question is whether C saves *real model evaluations at equal final score*, not whether its predictions
are numerically attractive.

### Relevant work (2024–2026, plus optimizer foundations)

- **LLMs as optimizers.** OPRO (2309.03409) has an LLM propose candidates from a prompt containing
earlier candidates and their measured scores. Promptbreeder (2309.16797) extends evolutionary prompt
search with self-referential prompt mutation. Both establish a useful outer-loop pattern, but their
search variables are short text strings; generating a valid, useful delta with 10⁹ parameters is a much
harder representation and credit-assignment problem. SkillOpt (2605.23904) is a more controlled text-
space successor: bounded artifact edits are accepted against held-out validation. This supports bounded
proposals and validation, not assuming an LLM has learned a stable gradient substitute.
- **Learned and discovered optimizers.** VeLO (2211.09760) is a learned optimizer trained across many
tasks, with meta-training reported at roughly 4,000 TPU-months; it is evidence for learned update rules
but not a recipe to meta-train on this project's small zoo. Lion (2302.06675) is a compact symbolic
optimizer discovered by program search. Muon's LLM-scale feasibility is studied in *Muon is Scalable
for LLM Training* (2502.16982). These are essential equal-budget controls: replacement is not an
advance unless it beats standard update rules under the same data, optimizer steps, and model-forward
budget.
- **LLM control of training and hyperparameters.** *Offline Training of Language Model Agents with
Functions as Learnable Weights* (2402.11359) lets an LLM optimize agent functions without modifying
the underlying model. ATTUNE (2610.17528) uses an agent to tune training throughput, with correctness
and convergence gates outside the agent. These are closer to B than OPRO: the LLM reasons over run-
level feedback and bounded controls, while an external gate protects the objective. For our short
fine-tunes, per-step free-form parameter deltas are much less constrained than changing a scalar
learning rate, clipping threshold, or update scale; start with the latter if B is attempted.
- **LLM-assisted model editing.** *Model Surgery* (2407.08770) demonstrates behavior changes through
targeted parameter edits, while task arithmetic (2212.04089) gives a non-LLM basis for controlled
weight-space edits. The Takehi et al. *Sparks of Weight-Space Programming* blog cited in
`docs/RESEARCH_NOTES.md` is the direct motivation for A: Experiment 8 edits Qwen3.5-0.8B weights in a
Python kernel with GPT-6 Astra, compares 10/25/50 forward-pass budgets, and uses no gradients. Treat
that as a promising report, not evidence that arbitrary editors are repeatable or generalize beyond a
localized loop-related defect.
- **Surrogate/critic-gated search.** Bayesian optimization uses a surrogate plus an acquisition rule to
allocate expensive evaluations; TuRBO (1910.01739) is a trust-region BO method for difficult
high-dimensional objectives. Model-based RL provides a related warning: learned dynamics support
planning, but model error compounds when the planner exploits the learned model. A here is an expensive
black-box proposal problem; C resembles surrogate-assisted BO and model-based planning. The transferable
lesson is uncertainty-aware ranking, short horizons, and frequent real checks. A predictor trained on
observational zoo edits is not automatically a calibrated acquisition model for adversarially chosen
edits.

All arXiv identifiers in this section were checked against their abstract-page titles using the
project's required command form (`curl -s https://arxiv.org/abs/<id> | grep -o '<title>[^<]*'`).
Verified IDs/titles: 2309.03409 *Large Language Models as Optimizers*; 2309.16797 *Promptbreeder:
Self-Referential Self-Improvement Via Prompt Evolution*; 2211.09760 *VeLO: Training Versatile Learned
Optimizers by Scaling Up*; 2302.06675 *Symbolic Discovery of Optimization Algorithms*; 2502.16982
*Muon is Scalable for LLM Training*; 2402.11359 *Offline Training of Language Model Agents with
Functions as Learnable Weights*; 2610.17528 *ATTUNE: Agentic Throughput Tuning for LLM Training*;
2407.08770 *Model Surgery: Modulating LLM's Behavior Via Simple Parameter Editing*; 2212.04089
*Editing Models with Task Arithmetic*; 2605.23904 *SkillOpt: Executive Strategy for Self-Evolving
Agent Skills*; and 1910.01739 *Scalable Global Optimization via Local Bayesian Optimization*. The
Takehi et al. blog is not an arXiv paper. Muon and Lion are optimizer baselines, not LLM agents; the
distinction matters when interpreting B.

### A–C assessment and practical costs

| Design | Expected gain | Episode cost on Qwen3.5-0.8B / one 32 GB card | Main risks |
|---|---|---|---|
| **A. LLM writes a delta** | Potentially high for sparse, interpretable defects with a clear diagnostic signal; low and unknown for diffuse benchmark gains. The blog gives a useful target. | 10/25/50 model forward passes per editor attempt, plus editor tokens and Python-kernel orchestration. At the P2 ceiling of ≤2 min for the full eval battery, 10–50 diagnostic forwards should generally be seconds to a few minutes; budget **~1–5 GPU-min per proposal**, then ≤2 min for full verification. A 50-forward episode can approach 7 GPU-min before retries. Track actual GPU time and input/output tokens; hosted-model spend depends on provider pricing. | Editor variance and prompt sensitivity; malformed deltas; exploiting benchmark artifacts; sparse success does not establish broad gain; repeated attempts inflate compute and selection bias. |
| **B. LLM steers/replaces optimizer** | Moderate for choosing among a small set of hyperparameters or diagnosing loss curves; low prior for raw parameter-update generation beating Adam/Muon on ordinary differentiable fine-tuning. | Each proposed training step incurs editor latency/tokens in addition to training and usually gradient forward/backward. Use 10–50 optimizer decisions per short run; expect **minutes to tens of minutes** even if the fine-tune itself is short. A local Qwen editor shares the GPU and competes for memory/throughput; a hosted editor adds API latency/cost. | Confounded by unequal gradient/data budgets; unstable update scales; delayed feedback; high variance; training loss may improve while held-out benchmarks degrade. |
| **C. Predictor critic/ranker** | Potentially the most direct route to H5: save full battery evaluations by rejecting weak proposals, while uncertainty-based exploration limits missed winners. | Predictor inference on cached φ features is negligible (seconds or less); feature extraction is one-time per delta. The costs are proposal generation and predictor training, plus real evaluations that pass the gate. | Reward hacking / distribution shift when editors adapt to the critic; miscalibrated uncertainty; selection bias; contamination if tuning sees final test items. |

These are planning ranges, not measured timings. P2's ≤2 min refers to the full battery and must be
profiled separately from diagnostic forward passes. Log editor prompt/completion tokens, GPU-seconds,
peak memory, diagnostic forwards, full evaluations, and rejected proposals. Use a cheap editor model as
the first A/B control; reserve Astra/Claude for a small matched comparison, not every episode.

### Controls and smallest decisive experiment

Use the same fixed goal set, base checkpoint, random seeds, and stopping rule across arms. Keep a sealed
per-item test set: expose only development items to the editor and critic; score the final selected
checkpoint once on the sealed set. Include (1) cheap LLM editor + C, (2) the same editor without C,
(3) random edits matched for layer, sparsity, norm, and diagnostic-forward budget, (4) Adam and Muon
short fine-tunes with the same real forward-pass budget as B, and (5) predictor ranking against random
ranking. Report paired outcomes and bootstrap confidence intervals by goal, not benchmark item. Use
replicated episodes per goal/seed so one lucky prompt cannot dominate.

**Smallest decisive run:** after the prerequisites below, run 30 held-out goal-seed episodes per arm
(at least 10 distinct goals × 3 fixed seeds), with one editor proposal batch capped at 10, 25, or 50
diagnostic forwards according to a pre-registered budget. Define each goal's target gain before
running. The metric is the number of *full P2-equivalent real evaluations* required to reach it, with
final sealed-set score non-inferior within 1 percentage point of the no-critic editor's mean gain. H5
passes only if C uses **≥30% fewer** full real evaluations than both the same editor without C and
equal-forward-budget Adam/Muon, at that equal final-score criterion; report paired bootstrap 95%
intervals and require the upper interval for the evaluation ratio to be <0.70 for a decisive pass.
Thirty episodes is a screening experiment; if the interval overlaps 0.70, expand to 100 episodes before
claiming or falsifying the threshold. Compare at equal actual GPU-seconds as a sensitivity analysis,
since forward-pass counts omit editor latency and optimizer backward cost.

### Recommendation and prerequisites

Try **C → A → B**. First validate C in shadow mode: predict and rank a held-out set of already logged P1
edits without letting the predictor alter proposal generation. Then run A with C as an evaluation gate,
because this is closest to the blog and does not entangle the critic with a changing optimizer. Attempt
B only after A shows repeatable wins; start B by letting an LLM choose a small discrete set of learning
rates/update scales around Adam/Muon, not emit raw per-parameter updates. Apply C to B only after its
proposal distribution has been checked for critic exploitation.

Do not start the decisive run until P1 has at least the roadmap target of **600 single edits and 150
trajectories**, with an explicit `llm_edit` stratum and per-item eval results. Require H1 first: predictor
must reach **ρ ≥ 0.6 per benchmark** on held-out edits and beat StatNN and delta-norm baselines, with
calibration checked on the relevant edit family. If H1 is weak, C has no credible basis for pruning
proposals; collect edits and improve the predictor instead. H1 accuracy alone is insufficient for
critic use: add a held-out calibration/coverage check and an uncertainty-triggered real-evaluation
fallback. Freeze the predictor during each episode, cap proposals, randomize editor arm order, log all
prompts and deltas, and prohibit access to sealed evaluation examples. The top Board C run is C's
shadow-ranker on existing P1 proposals; schedule A+C only after that gate passes.

## P4 implementation protocol (Board A revision, 4 November 2026)

The default supervised target is Δy_t = y_t − y_predecessor, computed from measured
benchmark rates. Successor snapshots use the immediate predecessor evaluation;
single-step edits (including task vectors) use base/eval.json. An entry with any
observed benchmark lacking its predecessor rate is excluded and counted. `--target level`
is the separate absolute-rate baseline. Heads use linear outputs in both modes;
all losses, validation, baselines and reported metrics use the selected target.

Edges are classified from the zoo manifest, supplemented with edit.json step metadata.
Only `chain` parent links are `successor`; task-vector parents are `source`, never
history or M3 transition pairs. Task vectors are base + scale·Δparent, with parent
stats available as a separate side-information block in pooled features (consumed
by M2 and by M1/M5's source projection). A chain `lora_vs_base` step is a flagged
successor: although its increment is optimized against base, it is added to the
accumulated state and its stored snapshot follows that state. This permits history
and parent-relative targets without claiming sequential LoRA optimization. Legacy
chain links without step metadata retain cumulative-successor semantics; their
missing flag is visible in the run manifest.

Family/seed training folds purge every descendant/source whose full manifest
ancestry (including itself and unlabeled steps) contains a held-out family/seed.
For family holdouts, `params.inner_family` also counts as that family. Purge counts
and exact split ids are recorded before normalization, pretraining or fitting.
Trajectory splits still group full source ancestry conservatively, even though
source edges never become sequence positions. Chain `length` and `inner_seed` are
excluded from descriptors.

M5 factorizes attention into **one tensor layer plus one causal edit layer**, d=256,
dropout=0.2. This bounds total attention depth at two. `--model m1_matched` uses two
stateless tensor layers at the same dropout and widens the final feedforward layer
by four units: its trainable parameter count differs from M5 by four parameters
(8d+4 versus 8d extra parameters). Compare M2/M5 against this control as well as M1.
The factorization remains a deviation from joint attention over all tensor/edit
positions. H2 tests history **beyond the latest cumulative-from-base snapshot**;
it does not isolate access to accumulated drift from no access to that drift.

M3 predicts the latent of cumulative-from-base delta features, φ(θ_t−θ₀), rather
than absolute weight features φ(θ_t), which P3 does not provide. This is an explicit
H3 design deviation; no absolute-weight JEPA claim follows from these experiments.

M6 trains a same-initial-encoder ablation with a frozen standard random linear
predictor (ResidualPred off), identical train rows, minibatches, complementary masks,
optimizer and epoch budget. ERank is computed after pretraining on the unmasked
training representations. Operational collapse threshold: ERank < 2. The collapse
gate requires zero collapsed fold seeds and ERank(ResidualPred) ≥ 2× ablation in
every fold; the separate overall acceptance metric also requires explicit `--holdout lora_sft`
and transfer ρ ≥ 0.5 on at least six benchmarks on that sole test fold. Runs of different
outer seeds must all pass before claiming zero collapsed seeds across a five-seed
experiment. The current parameter-free Q=K latent attention in ResidualPred is a
project implementation, not a verified exact reproduction of 2610.02344.

`python -m wsp.predictor.train --model m2 --selftest --device cpu` (or m5) creates a
four-step synthetic toy zoo and a two-epoch smoke run under `/tmp/wsp-p4-synthetic-*`.
Both feature and run manifests mark `synthetic: true`; these are never real results.
CUDA M2 must report `mamba_ssm`; CPU reports `loop`. Run CUDA only after the
`p1-zoo-full` agent-tmux job disappears. No synthetic run satisfies H1/H2/H3 gates.

Named transfer evaluation uses `--holdout lora_sft`: all labeled LoRA rows form the sole test fold. Other rows are ancestry-purged before inner validation. `--holdout-variant chain-free` additionally excludes chain rows and their descendants from training. K-fold runs serve H1/H2 and do not evaluate H3. H2 metrics are stratified by explicit step semantics, with unflagged legacy chain rows reported as unknown. Runs freeze the zoo manifest bytes before splitting and record their SHA256 and line count.
