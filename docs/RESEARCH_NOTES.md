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


## 11 Nov 2026 (council round 4)

Source: round-4 [brief](council/2026-11-11/BRIEF.md) ("What appeared since round 3") and the
[chair synthesis](council/2026-11-11/SYNTHESIS.md) §1; the decisions are PREREG A5 amendment F8. Earlier notes stay archival.
Headline: nothing published since 10 Nov touches the registered programme; the news adds two descriptive panels (F8-E external
edits, F8-F family transfer, no decision authority) and a scripted arm plus second editor for the H5 pilot. Items 1–5, 8 and 9
were fetched from arXiv on 11 Nov for this record and are in `paper/refs/verified.yaml`; items 6 and 7 are the brief's reading, and the
blogs and working papers (11–16) are the council's reading (the Concept Country and nathan.rs pages returned HTTP 403 to two members;
the chair read both on 11 Nov).

**Papers**

1. [WSP-Scope, "Exploring Weight-Space Programming"](https://arxiv.org/abs/2611.09382) (Hu et al.; site <https://wsp-scope.github.io/>):
   Astra edits of Qwen3.5-0.8B/9B incl. a 1,023-value column edit; direction u not released. Takeaway: excluded from F8-E until u
   is released; the recipe does not determine u; not an effect-size input. Development numbers are adaptively selected.
2. [Rane, LoRA fingerprinting](https://arxiv.org/abs/2611.07610): Qwen3.5-0.8B, weights < activations < outputs for domain recovery.
   Takeaway: prior only, no effect-size revision (48-dimensional spectral summary, CI includes chance, target is domain identity);
   stated in one sentence in the access-ladder report (F8.8).
3. [Deletion Tuning](https://arxiv.org/abs/2611.10866) (Lu et al.): < 0.05 % gate-weight mask. Takeaway: new family for the F8-F
   panel as Wanda-initialised gate deletion plus count-matched random deletion; a learned mask is a separate unpriced purchase.
4. [Behavioral rate of fine-tuning](https://arxiv.org/abs/2611.06598) (David et al.): rate is receiver-dependent. Takeaway: no new
   arm (base anchoring already carries the receiver); the compression-recovers-corrupted-adapter result is a post-G3 option
   (F8.10), not registered.
5. [IWIBench](https://arxiv.org/abs/2611.08647) (von Csefalvay et al.): controlled benchmark of gradient-free editing by an
   editor model on Gemma 3 4B IT. Takeaway: GPT-6.1 Sol registered as the practical editor beside Astra in the H5 pilot.
6. [Can an LLM reason and program about a network's weights](https://arxiv.org/abs/2611.08070) (Ahmed): audit of zero-shot weight
   reading on small CNNs. Takeaway: context for the tool-access flags and audit discipline; no programme change (not itemised in
   synthesis §1).
7. [Can LLMs debug robots by editing weights](https://arxiv.org/abs/2611.09376): 65 embedding rows; random-direction edits do not
   help. Takeaway: context only; no programme change (not itemised in synthesis §1).
8. [SWEET](https://arxiv.org/abs/2611.10974) (Fu et al.): activation-contrast weight edits. Takeaway: **deferred** from F8-F
   (needs generated counterfactuals; unpriced).
9. [Training-Free Task Vectors](https://arxiv.org/abs/2609.09054) (Perin et al.): rank-one edits from forward statistics.
   Takeaway: TFTV rank-one on `o_proj` enters F8-F; the edit is s̄ vᵀ with v from W's singular vectors and the expected input.
10. Minor, no action: [2611.04268](https://arxiv.org/abs/2611.04268) (WSP for image-restoration diffusion, conjectures only),
    [2611.09559](https://arxiv.org/abs/2611.09559) (LLM-written MILP branching scorers, matched by a pseudocost rule),
    [2611.06185](https://arxiv.org/abs/2611.06185) (geometry-aware regularised editing).

**Replications, controls and critiques (working papers and blogs, 4–10 Nov)**

11. [Concept Country working paper](https://concept.country/research/weight-space-programming) (Qwen3.8-27B, pre-registered,
    norm-matched orthogonal control). Takeaway: no change to Part A controls and no flaw found; the control is the published
    analogue of A5.3's Haar rotation (which also preserves spectrum, rank and support); the down_proj rank-one form feeds F8-F;
    reliance is on the control's design, not its numbers.
12. [aiXiv 261103.000004](https://aixiv.science/abs/aixiv.261103.000004): a difference-of-means script given Astra's layers,
    budget and scale recovers ≈ 93 % of the gain (cosine ≈ 0.78 to Astra's direction). Takeaway: motivates the scripted arm; its
    reconstruction is an F8-E entry; the 93 % figure is conditional on the supplied information; counts as one component with Astra.
13. [Barry, "Can a Linear Program Do Weight-Space Programming?"](https://nathan.rs/posts/weight-space-lp/): Astra's column/scale
    choice is an LP. Takeaway: LP selection is part of the scripted arm; published LP edits enter F8-E only with bytes or a
    complete recipe; reuses Astra's direction, so the same component.
14. [Biswas, "Editing a Model's Weights for $10"](https://glitch401.github.io/writing/editing-a-models-weights/): scripted edits cut
    answer length as much or more than the LLM edit. Takeaway: supports the scripted baseline (grouped under the H5 row; no
    separate synthesis line).
15. [Achintya K J audit](https://achintyakj.tech/blog/weight-space-audit): released logs reproduce; most results single runs with
    partly undisclosed selection. Takeaway: context for the held-out, selection-logged design (not itemised in synthesis §1).
16. Blogs: [brainwagon](https://brainwagon.org/blog/2026-11-08-a-skeptical-look-at-weight-space-programming),
    [AutoWeight](https://midhunxavier.com/blog/auto-weight/better-weights-tougher-tests/),
    [GEPA over weights](https://devichand.com/blog/2026/gepa-wsp/),
    [Khalfallah](https://dev.to/iyadh_khalfallah/i-tried-weight-space-programming-on-6-kinds-of-models-heres-where-it-works-and-where-it-breaks-58e3),
    [Jevlar](https://opper.ai/blog/jevlar-weight-space-programming). Takeaway: skeptical or negative small-scale replications; no
    programme change (not itemised in synthesis §1).
17. Public artefacts on our base: Hugging Face `RikiyaT/Qwen3.5-0.8B-weight-space-programming` and a two-tensor diff. Takeaway: the
    released Astra patch is the exact-replay entry of F8-E (hash-pinned against the pinned base revision; fails closed on any other
    tensor difference).
18. Not found: any paper since 10 Nov predicting benchmark change from a weight delta on a held-out split with a metanetwork.
    This limited search does not establish absence.

**Brief corrections (F8.9).** WSP-Scope names "long, unresolved reasoning", not loops, and derives u from token embeddings, not
activations. TFTV's edit is s̄ vᵀ with v built from W's singular vectors and the expected input, rank one and norm-matched.
Deletion Tuning tests no Qwen3.5 backbone. The aiXiv 93 % figure is conditional on Astra's layers, budget and scale. Fetch-time
check, 11 Nov: arXiv lists IWIBench (2611.08647) as submitted 5 Nov, not 4 Nov as in the brief.

## 11 Nov 2026 (council round 5)

Source: round-5 [brief](council/2026-11-11/BRIEF-r5.md) (five user threads) and the [chair synthesis](council/2026-11-11/SYNTHESIS-r5.md) §1;
the decisions are PREREG A5 amendment F9. Earlier notes stay archival. Headline: nothing found in the sweep changes Part A, F5.5 or the
H5 success criterion; the round adds descriptive extensions with no decision authority (F9.7): a forecast-before-modify log, a pruning
ladder, H5 field-localisation and test-time-LoRA shadow items, and a generation endpoint for F8-E. Items marked **(fetched)** were
read in full from arXiv on 11 Nov for this record and are in `paper/refs/verified.yaml`; every other item is the brief's reading of the
cited page and has not been re-fetched for this record. "Not itemised" means the synthesis draws no programme consequence from it.

**Thread 1: edits that improve mathematics.** Reading: every small-model math gain so far is a termination or robustness repair
(loops, overthinking, distractor reliance), generation-side and invisible to teacher-forced endpoints.

1. [ThinkEdit](https://arxiv.org/abs/2503.22048): weight edits against overly short thinking. Takeaway: not itemised.
2. [CoT Vectors](https://arxiv.org/abs/2510.00579): reasoning mechanisms transferred as vectors. Takeaway: not itemised.
3. [Weight-of-Thought](https://arxiv.org/abs/2504.10646). Takeaway: not itemised.
4. [Das replication](https://dev.to/krishanu_das/before-trusting-a-weight-edit-i-measured-the-bug-it-fixes-401f) (dev.to, 7 Nov; blog,
   not peer-reviewed): the published Astra edit makes thinking finish about five times as often at an 8,192-token budget (10/30 vs
   2/30), yet thinking-off on the unedited model (16/30) beats the edited model with thinking on (9/30); McNemar p = 0.021 raw, 0.107
   after his pre-committed Holm correction. Takeaway: completion must be measured separately from correctness; this motivates the
   F8-E generation endpoint (F9.6; MATH-500, three arms, 8,192-token cap, unpriced until a timed pilot) and paper sentence 5.
5. Carried from round 4, not repeated: Astra / WSP-Scope [2611.09382](https://arxiv.org/abs/2611.09382), SWEET
   [2611.10974](https://arxiv.org/abs/2611.10974).

**Thread 2: models editing their own weights while working.** Reading: runtime self-editing in 2026 is gradient-based and
adapter-isolated, governed by a policy that decides whether to edit; none of these systems predicts the edit's effect first.

6. [Agentic-TTT](https://arxiv.org/abs/2610.12002) (Lu, Kankanhalli; 8 Oct) **(fetched)**: a learned policy decides when to invoke
   test-time training, which method, or whether to reuse a stored LoRA skill; each invocation yields an isolated LoRA over the frozen
   backbone; the evaluation stream was deliberately enriched for TTT benefit. Takeaway: isolated adapters are root edits of the base,
   so the field is exact for them; the critic is tested on them only in shadow (F9.5, 32 queries; paper sentence 3).
7. [QueST](https://arxiv.org/abs/2605.13369) (Song et al.) **(fetched)**: query-conditioned test-time self-training, a LoRA per query,
   gains on seven math benchmarks and GPQA-Diamond. Takeaway: same as item 6; evaluated after application, never forecast.
8. [Search over Self-Edit Strategies](https://arxiv.org/abs/2601.14532): model-generated SEAL templates; the archive variant does not
   beat the best human template. Takeaway: not itemised.
9. [RSI-Env](https://arxiv.org/abs/2611.01736) (1 Nov): persistent environment for model-harness self-improvement from Qwen3.8-27B;
   gains across capabilities are hard to preserve. Takeaway: not itemised.
10. Harness-side or taxonomy, not itemised: [Recursa 2611.04035](https://arxiv.org/abs/2611.04035),
    [Evo2 2610.13025](https://arxiv.org/abs/2610.13025), [Exo 2611.00463](https://arxiv.org/abs/2611.00463),
    [MetaRSI 2609.06396](https://arxiv.org/abs/2609.06396), [SpecRSI 2610.16491](https://arxiv.org/abs/2610.16491),
    [RSI-Opt 2610.32026](https://arxiv.org/abs/2610.32026), [Harness RSI survey 2610.18813](https://arxiv.org/abs/2610.18813),
    [RSI taxonomy 2610.14625](https://arxiv.org/abs/2610.14625),
    [test-time self-improvement survey 2609.01679](https://arxiv.org/abs/2609.01679). Round-4 WeightLoop
    ([2611.07319](https://arxiv.org/abs/2611.07319)) is carried, not repeated.

**Thread 3: models reporting which parts they use.** Reading: a model cannot verbally report which weights it uses; probes and
attribution patching can. "Which part am I using" is the per-tensor breakdown of the forecast, not a self-report.

11. [OWMI](https://arxiv.org/abs/2608.20569) (Ferrara, Aug) **(fetched)**: eight open-weight models asked whether their computation was
    altered sit at chance (AUROC about 0.5007 over 78,000 measurements), while a linear probe on the same activations recovers the
    intervention at 75 % to 95.8 % accuracy. Takeaway: the strongest new evidence for supplying measured localisation (fifth access
    flag, F9.4) and logging verbal localisation as a treatment; paper sentence 2.
12. [Mechanistic study of LM introspection](https://arxiv.org/abs/2609.35108): gate and router heads decide whether and where a change
    is reported. Takeaway: not itemised.
13. [Mechanisms of Introspective Awareness](https://arxiv.org/abs/2603.21396); [Identifying Introspection From the
    Inside](https://arxiv.org/abs/2610.07186) (COLM 2026); [Training LMs to Explain Their Own
    Computations](https://arxiv.org/abs/2511.08579); [Introspect-Bench](https://arxiv.org/abs/2603.20276); [Multi-component causal
    tracing](https://arxiv.org/abs/2606.03085); [self-interpretation adapters](https://arxiv.org/abs/2602.10352). Takeaway: not itemised.
    Attribution patching would be a new metered artefact and is not registered (F9.4).
14. Round-4 reading audit [2611.08070](https://arxiv.org/abs/2611.08070): zero-shot weight reading is local arithmetic plus priors.
    Carried, not repeated.

**Thread 4: fewer parameters, same performance.** Reading: the active frontier is which weights to delete, decided by gradients,
router sensitivity or causal importance, the same object as the field term a = ⟨g, D⟩ for D = −W⊙M.

15. [Revisiting LLM pruning for test-time scaling](https://arxiv.org/abs/2604.25098) (Monjur et al.) **(fetched)**: unstructured
    Magnitude and Wanda pruning of s1.1-7B and Qwen3-8B at 10 % and 20 % sparsity matches or at times beats the unpruned model;
    structured pruning degrades. Takeaway: the 10 and 20 % rungs of the F9-P ladder sit where performance is reported to hold;
    paper sentence 4. No SparseGPT in this paper.
16. [OBC-Prune](https://arxiv.org/abs/2609.17890) (Nguyen et al.) **(fetched)**: outcome-based calibration weights rescale the
    calibration activations of SparseGPT, Wanda and ALPS on R1-Distill-Qwen 1.5B–14B at 40–50 % sparsity. Takeaway: the 50 % rung
    sits where this line works on calibrated methods; a rollout-anchored field (a gradient for p(end-of-thinking)) would be new,
    unpriced G2 work, recorded as a later hypothesis.
17. [Deletion Tuning](https://arxiv.org/abs/2611.10866) (round 4): under 0.05 % zeroed matches full fine-tuning. Carried; gate
    deletion is F8-F.
18. Not itemised: [Router sensitivity 2608.07890](https://arxiv.org/abs/2608.07890),
    [Pruning laws 2504.04342](https://arxiv.org/abs/2504.04342) (collapse past about 70–80 %),
    [LLMs can compress LLMs 2601.09694](https://arxiv.org/abs/2601.09694),
    [multi-objective structured pruning 2607.22583](https://arxiv.org/abs/2607.22583),
    [pruning vs training from scratch 2606.14150](https://arxiv.org/abs/2606.14150),
    [GradPruner 2601.19503](https://arxiv.org/abs/2601.19503), and ModHiFi (NeurIPS 2025; no arXiv id in the brief).

**Thread 5: predicting the effect of a modification before making it.** Reading: published modification world models are first-order
forecasts fixed before the run, which work, and learned surrogates over low-dimensional modification spaces, which work while they keep
learning; none predicts benchmark change from an arbitrary dense weight delta on a held-out split.

19. [Predict Before You Modify / IDM](https://arxiv.org/abs/2610.13991) (Xie et al.) **(fetched)**: an editable program that invokes a
    frozen LLM forecasts task-level outcomes of a proposed agent modification before evaluation, with online bias calibration.
    Takeaway: agent programs, not weights; its swap rule (accept a recalibration only if held-out loss falls and conditional MSE does
    not rise) is adopted for the F9.2 readout; paper sentence 1.
20. [PPF, predictive pruning](https://arxiv.org/abs/2508.02381) (Ma et al.) **(fetched)**: a predictor evaluates a structured pruning
    policy in 1.52 s against 1 min 38 s; built from 775 collected mask samples. Takeaway: a learned surrogate over a structured space;
    the F9-P ranking panel asks the same question with zero fitted parameters; paper sentence 1.
21. [Forecasting how RLHF responds to reward-model edits](https://arxiv.org/abs/2611.03244) (Wang et al., 2 Nov) **(fetched)**: a
    first-order covariance forecast fixed before 54 RL runs explains 82 % of standardised change in the factorial design and 66 % out
    of sample on real reward-model upgrades. Takeaway: the closest published analogue of F9-L; paper sentence 1.
22. RankAdaptor (NAACL 2025 findings; the brief gives no URL or arXiv id): an MLP over per-layer LoRA
    ranks for pruned LLMs. [TuneAhead 2606.17660](https://arxiv.org/abs/2606.17660) (round 3) and the risk-decomposition theory of
    pre-hoc prediction (PMLR v306, Luo et al.; no URL in the brief). Takeaway: not itemised.

**Not found.** No paper in which a model decides which of its own weights to change from a self-report of which parts it uses,
applies the change during inference without gradients, and predicts the effect first. This limited search does not establish absence.

**Brief corrections (F9.8).** PPF predicts a divergence to the dense model, covers structured channel pruning, and was built from 775
evaluated masks; arXiv 2611.03244's 82 % is in-sample and 66 % out of sample; IDM is an editable program calling a frozen LLM,
calibrated by online bias plus conditional MSE; arXiv 2604.25098 tests unstructured pruning at 10 % and 20 % only, without SparseGPT;
Agentic-TTT's evaluation stream was deliberately enriched for TTT benefit. Fetch-time checks, 11 Nov: arXiv lists 2610.13991 as
submitted 26 Sep 2026 (the brief says 15 Oct); the PPF source describes 775 collected mask samples with 20 % held out for testing, so
"trained on 775 masks" in the brief and F9.8 is read as "built from 775 collected samples".
