# Related Work Map

This document systematically organizes all foundational literature and state-of-the-art references (as of November 2026) supporting our investigation of weight-space performance prediction.

The references are grouped into four functional areas:
1. **Weight-Space Learning / Performance Prediction**
2. **JEPA / SSM over Weights**
3. **Abliteration & Defences**
4. **Agent Harnesses**

Each entry specifies the citation key, full title, its precise relationship to our hypotheses (**H1–H4**), and exactly what claim or component in the paper we cite it for.

---

## 1. Weight-Space Learning & Performance Prediction

| Citation Key | Reference & Venue | Relation to H1–H4 | What We May Cite It For |
| :--- | :--- | :--- | :--- |
| `takehi2026wsp` | **Takehi et al. (2026)**<br>*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*, Blog (MIT CSAIL), 2 Nov 2026 | Direct empirical trigger for **H1** (gradient-free weight reading/editing) and benchmark for **H4** (frontier LLM native reading). | Editing 0.016% of Qwen3.5-0.8B weights raised MATH-500 from 7.9% to 48.1% without gradients. |
| `han2026survey` | **Han et al. (2026)**<br>*A Survey of Weight Space Learning: Understanding, Representation, and Generation*, arXiv:2603.10090 | Foundational taxonomic framework supporting the motivation of **H1** and **H3**. | Categorizes weight-space learning into three primary axes: understanding weight geometries, learning representations, and generating parameters. |
| `w2t2026` | **Han et al. (2026)**<br>*W2T: LoRA Weights Already Know What They Can Do*, arXiv:2603.15990 | Canonical LoRA tokenization reference for H1. | Canonical QR-SVD LoRA tokens feed rank-level and position-level Transformers; Table 2 reports ARC-LoRA test MAE 0.32 and Spearman 95.35 (%). |
| `sarangi2026finite` | **Sarangi et al. (2026)**<br>*Finite Probes Suffice: Identifiability and Universality for Weight-Space Learning*, arXiv:2609.33901 | Theory of finite probing relevant to H1. | Finite input queries/probes provide provable identifiability and universality guarantees for learning neural functionals of network behavior. |
| `wilson2026spectra` | **Wilson et al. (2026)**<br>*Why Heavy-Tailed Weight Spectra Predict Model Quality*, NeurIPS 2026 / arXiv:2610.13123 | Directly informs the statistical feature extractor $\phi_{\text{stats}}$ baseline in **H1**. | Heavy-tailed spectral distribution statistics in DNN weight matrices correlate strongly with and predict model quality. |
| `falk2025modelhubs` | **Falk et al. (2025)**<br>*Learning Model Representations Using Publicly Available Model Hubs*, arXiv:2510.02096 | Motivates our open model zoo curation strategy for training predictors across **H1–H3**. | Weight-space representation learners can be trained on heterogeneous public model repositories (like Hugging Face) rather than costly curated laboratory zoos. |
| `falk2025zoosize` | **Falk et al. (2025)**<br>*The Impact of Model Zoo Size and Composition on Weight Space Learning*, arXiv:2504.10141 | Model population diversity relevant to H1 and H3. | Model zoo size and dataset diversity affect weight-space model generation and generalization; increasing single-zoo training sample size significantly improves generation performance. |
| `position2026weightspace` | **Wang et al. (2026)**<br>*Position: Weight Space Should Be a First-Class Generative AI Modality*, arXiv:2605.18632 | Conceptual justification for treating model weight parameters and deltas as native data modalities in **H1–H3**. | The position paper argues for treating checkpoints as a first-class data modality and standardizing generative modeling in weight space. |
| `blockstructure2026` | **Wang (2026)**<br>*Block-Level Weight-Space Structure Persists Under Post-Training: An Empirical Study Across LLM Families*, arXiv:2609.26147 | Establishes our empirical boundary and sanity-check thresholds for post-training weight deltas in **H1**. | Post-training modifies every tensor but preserves block-level cosine > 0.99 and relative Frobenius < 0.13, while independent specializations drop to cosine ~ 0.64. |
| `goel2025diffinterpretation` | **Goel et al. (2025)**<br>*Learning to Interpret Weight Differences in Language Models*, ICLR 2026 / arXiv:2510.05092 | Directly contrasts with **H1** and provides a conceptual bridge to **H4** (interpreting weight diffs via natural language). | Diff Interpretation Tuning uses synthetic labeled weight diffs to train an adapter that makes compatible fine-tuned models describe their modifications. |
| `deltaactivations2025` | **Xu et al. (2025)**<br>*Delta Activations: A Representation for Finetuned Large Language Models*, arXiv:2509.04442 | Provides an activation-based upper-cost comparison baseline against our weight-only predictor in **H1**. | Delta Activations embeds fine-tuned models by measuring internal activation shifts relative to a base model. |
| `fastprobing2026` | **Liu et al. (2026)**<br>*Fast and Accurate Probing of In-Training LLMs' Downstream Performances*, arXiv:2604.01025 | Adjacent performance prediction baseline contrasted with **H1**'s zero-inference weight prediction. | Probing intermediate hidden activations predicts downstream performance with AUROC > 0.75 while cutting evaluation latency from 1 hr to 3 min. |
| `tuneahead2026` | **Luo et al. (2026)**<br>*TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins*, arXiv:2606.17660 | Related predictive benchmark for downstream performance estimation relevant to **H1**. | TuneAhead combines dataset descriptors with short-probe features; on 370 held-out Qwen2.5-7B-Instruct runs it reports 1.47 percentage-point RMSE. |
| `proxymetrics2026` | **Patel et al. (2026)**<br>*Forecasting Downstream Performance of LLMs With Proxy Metrics*, arXiv:2605.18607 | Evaluates how forward-pass predictive statistics correlate with downstream benchmark metrics in **H1**. | Proxy metrics aggregate token-level statistics from next-token distributions over expert-written solutions to forecast model capabilities. |
| `taskvectors2026interfere` | **Zhu et al. (2026)**<br>*When Do Task Vectors Interfere? Mapping the Validity Boundaries of Weight-Space Composition*, arXiv:2608.09490 | Critical methodological caution for our evaluation harness across **H1–H3**. | Task-vector interference depends on training-format serialization: the trained wrapper masks interference, while an untrained chat template does not. |
| `elhage2021mathematical` | **Elhage et al. (2021)**<br>*A Mathematical Framework for Transformer Circuits*, Transformer Circuits Thread | Mechanistic circuit grounding referenced by Astra in the weight-space programming trigger for **H1**. | In small attention-only transformers, induction heads explain in-context learning and develop only with at least two attention layers. |

---

## 2. Sequence & JEPA over Weights

| Citation Key | Reference & Venue | Relation to H1–H4 | What We May Cite It For |
| :--- | :--- | :--- | :--- |
| `jepawg2026physics` | **Göbel et al. (2026)**<br>*Weight-Space Physics: Interpretable Hypernetworks for Lattice Quantum Field Theories*, arXiv:2607.07127 | Covariance ablation option relevant to H3. | The JEPAWG main-text configuration omits the VICReg covariance term (gamma = 0), allowing a lower-dimensional latent representation. |
| `jepawg2026bifurcations` | **Göbel et al. (2026)**<br>*Bifurcations from Weight Space: JEPAWG on Parametric Dynamical Systems*, arXiv:2610.30847 | Extends **H3** validation to non-linear physical regimes and phase transitions. | On parametric dynamical systems, derivatives of JEPAWG decoded weights locate Hopf bifurcations, track period doubling, and detect a global bifurcation. |
| `predictability2026dynamics` | **Wang et al. (2026)**<br>*Measuring Structured Predictability in Neural Training Dynamics: A Cross-Regime Study*, arXiv:2608.15483 | Directly motivates trajectory-level sequential modeling in **H2**. | Across CIFAR training and Pythia checkpoints, bulk parameters have localized, time-varying pockets of predictable short-horizon dynamics, unlike simpler auxiliary parameters. |
| `leapverify2026` | **McEntire (2026)**<br>*Leap+Verify: Regime-Adaptive Speculative Weight Prediction for Accelerating Neural Network Training*, arXiv:2602.19580 | Connects sequential weight motion to speculative trajectory estimation under **H2**. | Leap+Verify proposes speculative future-weight prediction; predictions are accepted only after validation against a held-out loss criterion. |
| `jepahmm2026` | **Huang (2026)**<br>*Your Probabilistic JEPA Is Secretly a Hidden Markov Model: A State-Space Interpretation of Joint-Embedding Predictive Learning*, arXiv:2608.13621 | Conditional state-space interpretation relevant to H2 and H3. | The paper distinguishes four levels of JEPA/HMM correspondence and gives sufficient conditions for exact sequence-level HMM equivalence. |
| `drivedecay2026` | **Costa et al. (2026)**<br>*Drive vs. Decay: On the Training Dynamics of Joint-Embedding Predictive Architectures*, arXiv:2610.02344 | Predictor initialization reference for H3. | ResidualPred biases transformer predictor attention toward identity at initialization and improves effective rank and downstream accuracy in the reported experiments. |
| `mamba3` | **Lahoti et al. (2026)**<br>*Mamba-3: Improved Sequence Modeling using State Space Principles*, ICLR 2026 / arXiv:2603.15569 | Directly provides the backbone architecture for our sequential trajectory predictor in **H2**. | Mamba-3 combines a more expressive SSM recurrence, complex-valued state updates, and a multi-input multi-output formulation. |

---

## 3. Abliteration & Defences

| Citation Key | Reference & Venue | Relation to H1–H4 | What We May Cite It For |
| :--- | :--- | :--- | :--- |
| `heretic2026` | **Weidmann (2026)**<br>*Heretic*, GitHub: p-e-w/heretic (README accessed 3 November 2026) | Defines our primary edit family for the safety sub-study and benchmarks under **H1–H3**. | Heretic combines a configurable modifier with Optuna TPE optimization; Arbitrary-Rank Ablation is the default modifier. |
| `pksharma2026heretic` | **Sharma (2026)**<br>*Heretic automates stripping refusals from open-weight models. Its headline score counts 33 words, not harm*, P.K. Sharma briefing, 21 September 2026 | Motivates our dual-evaluation metric (keyword matching vs. judge-model) for alignment edits in **H1**. | Sharma describes the default Heretic refusal score as case-insensitive matching of 33 strings in the first 100 tokens, rather than a measure of harmfulness. |
| `abliteration2025comparative` | **Young (2025)**<br>*Comparative Analysis of LLM Abliteration Methods: A Cross-Architecture Evaluation*, arXiv:2512.13655 | Guides our benchmark selection for evaluating capability degradation after weight edits in **H1**. | Evaluates abliteration frameworks across 16 models showing mathematical reasoning (GSM8K) is most sensitive to refusal ablation interventions. |
| `surgicalrefusal2026` | **Cristofano (2026)**<br>*Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning*, arXiv:2601.08489 | Provides a concept-guided edit family with measured collateral-disruption proxies, relevant to **H1** and **H3**. | SRA orthogonalizes refusal against protected concept atoms; across five tested models it reports 0–2% refusal, mean perplexity change about +0.02 and mean first-token KL about 0.025, supporting limited measured collateral disruption rather than universal capability preservation. |
| `refusalbeyond2026` | **Rocchetti et al. (2026)**<br>*Refusal Beyond a Single Direction: A Preliminary Comparison of Diff-in-Means and INLP*, arXiv:2606.13720 | Extends single-direction ablation to subspace interventions, enriching our edit zoo diversity for **H3**. | Compares Difference-in-Means with Iterative Nullspace Projection (INLP) for refusal representation removal beyond single-direction assumptions. |
| `singleattack2026moe` | **Shi et al. (2026)**<br>*How Fragile Is Safety Alignment at Frontier Scale? A Single-Direction Attack on a 320B MoE*, arXiv:2609.09793 | Architecture-dependent limits on refusal interventions relevant to H1. | Directional ablation is applied to GLM-5.3-Flash (320B), but measurable refusal survives every tested edit, including subspaces of ranks 1–12. |
| `decoy2026direction` | **Muhamed et al. (2026)**<br>*Decoy Direction Optimization: A Post-Hoc Defense Against LLM Abliteration*, arXiv:2609.16204 | Represents an adversarial weight-editing defence family in our zoo for **H1** and **H3**. | DDO is a post-hoc weight-editing defense requiring no base-model fine-tuning; it injects a nonlinear decoy signal to corrupt refusal-direction estimators. |
| `refusalaliases2026` | **Truong (2026)**<br>*Abliteration Mitigation via Refusal Aliases*, arXiv:2608.18093 | Represents a defensive weight transformation family in our zoo for **H1** and **H3**. | Mitigates abliteration by applying low-rank weight updates that replace refusal-inducing activations with randomized alias vectors. |
| `weightarithmetic2025steering` | **Fierro et al. (2025)**<br>*Steering Language Models with Weight Arithmetic*, arXiv:2511.05408 | Edit family requiring two fine-tunes, incorporated into our multi-task edit zoo for **H1** and **H3**. | Contrastive weight steering subtracts deltas from two small fine-tunes with opposite behaviors, then adds or removes the resulting direction. |

---

## 4. Agent Harnesses

| Citation Key | Reference & Venue | Relation to H1–H4 | What We May Cite It For |
| :--- | :--- | :--- | :--- |
| `harness2026recursive` | **Xu et al. (2026)**<br>*Harness^2: Recursive Agent Harnessing for an Open World*, arXiv:2610.14224 | Recursive harness improvement relevant to experiment orchestration. | Harness2 improves task harnesses with Propose–Probe–Compose and uses harnessing-level recursion to transfer experience to future tasks. |

## Round 3 hypothesis dependencies and limits

H1 uses W2T as a performance-prediction precedent; its Table 2 benchmark is specific to ARC-LoRA, not our edit zoo. H1/H3 use the zoo-size result to motivate sample size and diversity, without assuming monotonic scaling. SRA motivates measured refusal/capability trade-offs in the alignment sub-study; perplexity and KL do not certify preserved task accuracy.

H3 borrows JEPAWG's covariance omission and coupling-to-weight embedding setup (latent dimension 16). The Adam 3e-4 passage describes conditional-flow data generation, not the JEPAWG optimizer; the paper must not attribute that optimizer to JEPAWG or treat physical couplings as edit descriptors without adaptation. H2 uses Leap+Verify as a speculative prediction proposal, not evidence of demonstrated training speedup. H2/H3 use JEPA/HMM correspondence only conditionally: Markov dynamics, valid emissions, consistent marginals and a filtering-consistent encoder are additional assumptions, not consequences of standard JEPA training.

H1's alignment sub-study uses the 320B MoE work as evidence of persistent refusal after tested edits, not a complete single-direction bypass. H1/H3 may use contrastive weight steering only as arithmetic after two gradient-based fine-tunes; HYPOTHESIS.md explicitly excludes gradient-free optimization claims. Harness² motivates recursive orchestration only: it supplies no established reliable multi-agent evaluation result for H1–H4. H4 uses the native-reader comparison and does not depend on these discarded assertions.
