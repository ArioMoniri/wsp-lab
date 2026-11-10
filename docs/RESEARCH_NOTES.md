# Research notes (state of the art as of 3 Nov 2026)

## The trigger
- Takehi, Teoh, Zhang, Khattab. *Sparks of Weight-Space Programming*, MIT CSAIL, 2 Nov 2026.
  https://rikiyat.github.io/blog/weight-space-programming/
  Key numbers: Qwen3.5-0.8B MATH-500 7.9 → 48.1 by editing 131,072 weights (0.016 %) in MLP output columns of
  layers 15 and 19 (subtracting a "loop" direction), 50 forward passes, no gradients. IFEval 21.0 → 62.3,
  loop rate 93 → 33 %. Gemma-3-270M could *not* be improved (no single component behind errors).
  Astra reads game/shape/language from raw weights; finds induction heads via QK/OV composition (Elhage 2021)
  but *fails* to find sycophancy from weights alone (needs attention patterns).
  → Our takeaway: weight-only reading works when the behaviour is a low-rank direction; sequence/trajectory
  context may be what is missing for diffuse behaviours. That is H2/H3.

## Weight-space learning (predicting performance from weights)
- Han et al. *A Survey of Weight Space Learning: Understanding, Representation, Generation*, arXiv 2603.10090 (Mar 2026).
- **W2T: LoRA Weights Already Know What They Can Do**, arXiv 2603.15990 (Mar 2026). QR→SVD canonical LoRA
  tokens + 2-level Transformer. ARC-LoRA performance prediction MAE 0.32, Spearman 95.4. Our φ_svd front-end.
- *Finite Probes Suffice: Identifiability and Universality for Weight-Space Learning*, arXiv 2609.33901 (Sep 2026).
- *Why Heavy-Tailed Weight Spectra Predict Model Quality*, arXiv 2610.13123 (NeurIPS 2026) — spectral stats
  belong in φ_stats.
- *Learning Model Representations Using Publicly Available Model Hubs*, arXiv 2510.02096 — the "free zoo" idea.
- *The Impact of Model Zoo Size and Composition on WSL*, arXiv 2504.10141.
- *Position: Weight Space Should Be a First-Class Generative AI Modality*, arXiv 2605.18632 (May 2026).
- *Block-Level Weight-Space Structure Persists Under Post-Training*, arXiv 2609.26147 — post-training keeps
  block cosine > 0.99, rel. Frobenius < 0.13; independent specialisations drop to cos ≈ 0.64. Good sanity
  ranges for our deltas.
- *Learning to Interpret Weight Differences in Language Models* (Diff Interpretation Tuning), ICLR 2026.
- *Delta Activations*, arXiv 2509.04442 — activation-shift embeddings of fine-tunes; a baseline that *does*
  run the model (we treat it as an upper-cost reference).

## Sequence / JEPA over weights
- **JEPAWG**: *Weight-Space Physics* arXiv 2607.07127 (Jul 2026) and *Bifurcations from Weight Space*
  arXiv 2610.30847 (Oct 2026). JEPA (VICReg, no covariance term) between task parameter and weights, decoder
  trained after freezing. MLP encoders, latent 16, Adam 3e-4. Template for M3.
- *Measuring Structured Predictability in Neural Training Dynamics*, arXiv 2608.15483 — short-horizon weight
  motion is predictable in pockets; motivates trajectory features.
- *Leap+Verify*, arXiv 2602.19580 — speculative weight prediction during training.
- *Your Probabilistic JEPA Is Secretly a Hidden Markov Model*, arXiv 2608.13621 — JEPA ↔ state-space link
  (justifies "JEPA or Mamba" in the brief as the same family).
- *Drive vs. Decay: training dynamics of JEPAs*, arXiv 2610.02344 — ResidualPred predictor init; use it.
- Mamba-3, arXiv 2603.15569 (ICLR 2026). `mamba-ssm` 2.3.2 on PyPI (Mamba-2 kernels).

## Performance prediction without full eval (adjacent)
- *Fast and Accurate Probing of In-Training LLMs' Downstream Performances*, arXiv 2604.01025 — probes on
  hidden states, AUROC > 0.75, 1 h → 3 min. Activation-based, needs forward passes; our method should not.
- *TuneAhead*, arXiv 2606.17660 — predicts fine-tune outcome from dataset + short probe; RMSE 1.47 pt.
- *Forecasting Downstream Performance of LLMs With Proxy Metrics*, arXiv 2605.18607.
- *When Do Task Vectors Interfere?*, arXiv 2608.09490 — non-additivity is input-format conditioned;
  evaluate with and without the chat template.

## Abliteration / Heretic
- Heretic 2.0.0 released 1 Nov 2026 (p-e-w/heretic, AGPL, PyPI `heretic-llm`). Default modifier is now
  **ARA (Arbitrary-Rank Ablation)**; classic directional abliteration (projected + MPOA norm-preserving) kept.
  Optuna TPE, 200 trials, 60 startup, objectives (KL on 100 harmless, refusals on 100 harmful).
  Since 1.2 can emit a LoRA adapter instead of merged weights. Research features moved to "Lophius".
  Benchmark: Gemma-3-12B refusals 97→3 at KL 0.16 vs 0.45 (huihui) / 1.04 (mlabonne).
- *Comparative Analysis of LLM Abliteration Methods*, arXiv 2512.13655. *Surgical Refusal Ablation*,
  arXiv 2601.08489. *Refusal Beyond a Single Direction*, arXiv 2606.13720. *Single-direction attack on a 320B MoE*,
  arXiv 2609.09793. Defences: *Decoy Direction Optimization* 2609.16204, *Refusal Aliases* 2608.18093.
- *Steering Language Models with Weight Arithmetic*, arXiv 2511.05408 — contrastive weight steering; another
  edit family we can add cheaply.
- Critique (pk-sharma, Sep 2026): the 97→3 headline counts refusal-phrase matches, not harm. We report the
  keyword refusal rate; an independent LLM-judged rate is not executed by this metric.

## Hugging Face, trending 3 Nov 2026 (relevant)
LFM3-1.7B (#1), DeepSeek-V4.1-Flash-1017, Qwen3.8-27B (8.0M downloads), Nanbeige4.3-33B, and a dense
ecosystem of `heretic`/`abliterated` variants (huihui-ai, 0bserverx, DavidAU, SC117, orcarouter).
See `docs/MODELS.md`.


## 10 Nov 2026 news amendment — six items and limits

Source: council round-3 reports and [chair synthesis](council/2026-11-10/SYNTHESIS.md).
The earlier notes remain archival; the following corrections govern this brief.

1. [WeightLoop](https://arxiv.org/abs/2611.07319): feedback-validated,
   rebuilt-from-base predictor updates motivate H5's reversible critic contract;
   8/8 versus random and 7/8 versus original are assay-ranking results, not
   held-out LLM delta-to-benchmark prediction (verified entry `weightloop2026`).
2. [Neural Thickets](https://arxiv.org/abs/2603.12228): scale-dependent random-edit
   expert density and select/top-K-ensemble search motivate a capped 4B diagnostic;
   they do not establish predictability of Qwen3.5-0.8B trained edit effects
   (`neuralthickets2026`).
3. [LoL](https://arxiv.org/abs/2410.04207): property prediction from low-rank
   factors warrants descriptive L/R/K/S secondaries; ARC-C R² .987 versus .983
   from singular values alone does not isolate orientation information, and the
   brief's “50,000 times faster” claim is withdrawn (`lol2024`).
4. Blog debate: [MindStudio](https://www.mindstudio.ai/blog/weight-space-programming-overhyped-debate),
   [csinva](https://github.com/csinva/weight-space-programming-autoresearch),
   [MaceTenth](https://github.com/MaceTenth/weight-space-programming-replication),
   [Wikipedia](https://en.wikipedia.org/wiki/Weight-space_programming), plus
   [the original blog](https://rikiyat.github.io/blog/weight-space-programming/),
   motivate explicit tool-access flags and matched random best-of-N; an unpaired
   6× comparison, Gemma failure and small-network replication do not establish
   general LLM weight readability. Evolving repository descriptions are not
   universal impossibility results; the González/Polaris attribution remains
   unverified (round3-editor §1).
5. [Neural Field Adaptation](https://arxiv.org/abs/2512.01759): low-rank
   representations support reconstruction/generation, not our behavioral-response
   endpoint; no primary-programme change (round3-world-model §1).
6. [Proxy-metric forecasting](https://arxiv.org/abs/2605.18607) and
   [weight forecasting](https://arxiv.org/abs/2508.05059) use output statistics or
   predict weights; neither supplies our held-out delta-to-benchmark test, and
   this limited search cannot establish absence of relevant 2026 work.

Inventory correction (round3-representation §2): no chain LoRA factors are stored;
chain steps are dense snapshots. Stored adapters number 121 lora_sft + 1 heretic_ara.
The compact head is about 1.6k parameters; learned frames add parameters, not labels.
No empirical new programme result is claimed here.
