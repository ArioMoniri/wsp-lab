# Round 15, challenger seat: attacking the premises of world-model pruning and unlearning (23 Nov 2026)

## BLUF

1. "World-model-predicted pruning" is Taylor saliency in disguise; Wanda equals our field score q_A up to a scalar factor (Board A F9a.8).
2. The mandatory baseline is pooled curve + dose (sparsity ratio) + Wanda; in Round 13, no learned model beat pooled + dose, with inner CV switching sequence models off in 30/30 folds.
3. The R13 identifiability finding dooms learned pruning prediction: norm tracks recipe dose at r = 0.996, leaving no beyond-dose variation that a world model can identify without retraining.
4. Knowledge localisation does not inform editing: Hase et al. proved causal tracing attribution gives zero insight into which MLP layers to edit, with layer depth dominating attribution.
5. SOTA unlearning (RMU, NPO, task vectors) operates globally across representations or parameter directions; localized weight deletion damages coherence while leaving semantic circuits intact.
6. Forget-set QA loss and refusal metrics measure shallow behavioural suppression: Hu et al. show benign relearning instantly restores knowledge, while Patil et al. show linear probes decode latent facts.
7. Study M′ proved behavioural refusal heuristics fail against blind judges (agreement 0.66–0.74), while Gemma post-norms cancel uniform rescales, invalidating naive localized zeroing.
8. Decisive $0 CPU/local-GPU gate: if a learned predictor cannot beat Wanda on 24 masks by Δτ ≥ 0.15, or if localized unlearning fails a 10-step relearning audit, both projects must be killed immediately.

## 1. Attack on Premise 1: World-Model Pruning is Saliency in Disguise

Proposing a world model to predict deletion damage mischaracterises the literature and ignores our repo's empirical constraints.

### Classical Saliency is Already the First-Order World Model
Pruning without retraining is the domain of Taylor saliency (OBD, 1990; OBS, 1993), modeling damage as $\Delta \mathcal{L} \approx g^T \Delta w + \frac{1}{2} \Delta w^T H \Delta w$. In LLMs, Wanda ([Sun et al., 2023](https://arxiv.org/abs/2306.11695)) scores weights via $|W_{ij}| \cdot \|X_j\|_2$, while SparseGPT ([Frantar & Alistarh, 2023](https://arxiv.org/abs/2301.00774)) computes second-order reconstructions in one shot.

Per Board A review F9a.8 (`docs/reviews/council/board-a-f9a-2026-11-11.md`), Wanda is algebraically identical to our field term $q_A = \frac{1}{2}(W \odot X)^2$ up to a factor. A world model predicting deletion damage merely re-approximates what first-order fields and second-order Hessians compute in closed form. Surrogates like PPF ([Ma et al., 2025](https://arxiv.org/abs/2508.02381)) require 775 masks to search coarse layer/channel ratios, not unstructured parameter masks.

### The R13 Identifiability Barrier
In Round 13 (`docs/council/2026-11-20/SYNTHESIS-r13.md`), trajectory prediction beyond dose was registered negative. In the 9B exploratory model search (`docs/reviews/wme/wme-stagew2-modelsearch-2026-11-20.md`), five candidates (J1 ridge, GPs, per-endpoint heads, GRUs, selective SSMs) were evaluated under leave-one-run-out CV against **pooled curve + dose** ($E = 0.678$). None beat the baseline; inner CV disabled GRUs and SSMs in 30/30 and 29/30 folds.

Weight norm tracks learning rate and recipe dose at $r = 0.996$. In pruning, sparsity ($k\%$) is the direct analogue of dose, explaining across-mask loss variance, while within-sparsity variation is dominated by local activations ($q_A$). Without executing forward-backward passes on masked networks, a world model cannot overcome the identifiability barrier from Stage W2. The mandatory baseline is:
$$\text{Baseline} = \text{Pooled Curve}(\text{sparsity } k) + \text{Wanda } q_A$$
A learned model has zero demonstrated headroom above this baseline.

## 2. Attack on Premise 2: Knowledge Localisation Does Not Inform Editing

The proposal assumes that finding where a fact resides in weight space reveals which weights to delete or modify to unlearn it. This premise is empirically refuted.

### Hase et al. and the Failure of Localization to Direct Interventions
ROME ([Meng et al., 2022](https://arxiv.org/abs/2202.05262)) and MEMIT ([Meng et al., 2022](https://arxiv.org/abs/2210.07229)) assume causal tracing localises factual recall to mid-layer MLPs. However, Hase et al. (NeurIPS 2023, [arXiv:2301.04213](https://arxiv.org/abs/2301.04213)) tested whether causality-based localization informs editing. Their finding was definitive: **causal tracing provides no insight into which MLP layers are best to edit to override or erase facts.**

Edits in layers with zero causal attribution succeed just as well as in localized layers. Coarse layer depth (early vs late) predicts editing efficacy far better than localized causal attribution scores. Activation during generation does not imply that deleting those parameters achieves clean unlearning.

### SOTA Unlearning is Global and Directional, Not Localised Deletion
Effective unlearning on TOFU ([Maini et al., 2024](https://arxiv.org/abs/2401.06121)) and WMDP ([Li et al., 2024](https://arxiv.org/abs/2403.03218)) does not perform discrete weight zeroing. RMU ([Li et al., 2024](https://arxiv.org/abs/2403.03218)) misdirects representations toward random steer vectors. NPO ([Zhang et al., 2024](https://arxiv.org/abs/2404.05868)) optimizes negative preferences to avoid collapse, while task-vector negation ([Ilharco et al., 2022](https://arxiv.org/abs/2212.04089)) subtracts coherent parameter directions. Localized weight deletion causes rank collapse and activation shocks while leaving the distributed semantic circuit intact.

## 3. Attack on Premise 3: Forget Metrics Measure Suppression, Not Erasure

The proposal relies on forget-set QA accuracy, loss, and retain benchmarks. Literature confirms these measure superficial suppression rather than genuine unlearning.

### Relearning Attacks, Probing, and Internal Guard Failures
1. **Benign Relearning Vulnerability:** Hu et al. (2024, [arXiv:2406.13356](https://arxiv.org/abs/2406.13356)) showed that approximate unlearning on TOFU and WMDP merely suppresses output logits. Minimal fine-tuning on loosely related auxiliary data instantly "jogs" memory and recovers supposedly forgotten knowledge.
2. **Linear Representation Probing:** Patil et al. (2024, [arXiv:2402.16835](https://arxiv.org/abs/2402.16835)) showed that even when an unlearned model achieves 0% extraction and perfect refusal on target QA, linear probes trained on intermediate residual activations decode target facts with high accuracy.
3. **MIA Leakage:** In MUSE ([Shi et al., 2024](https://arxiv.org/abs/2407.06460)) and Membership Inference audits ([Duan et al., 2024](https://arxiv.org/abs/2402.07841)), aggressive unlearning causes loss overshooting on the forget split. This creates two-sided MIA separability: an adversary easily detects that Person X was an unlearning target, compromising privacy.
4. **Study M′ Audit Failures:** In Study M′ (`docs/reviews/wmm/wmm-mprime-final-2026-11-23.md`), frozen refusal rules failed E4 against blind Opus judges (agreement 0.66–0.74), missing >20 pp of refusals. Surface heuristics cannot establish erasure. Furthermore, Gemma post-norms cancel uniform rescales (`docs/reviews/wmm/wmm-gate0-2026-11-19.md`), invalidating naive localized modifications.

## 4. Decisive Experiments, Power, and Kill Rules

We must not commit GPU budget or launch Lambda runs for speculative claims. Two low-cost gates run locally at $0.

### Experiment 1: The F9-P Pruning Prediction Gate (CPU / Local GPU, $0)
- **Model & Masks:** Qwen3.5-0.8B; 24 masks at 20% and 50% sparsity (8 Wanda, 8 SparseGPT, 8 random).
- **Predictors:** Baseline Wanda ($q_A$), mask norm ($\|W \odot M\|_2^2$), and world-model field forecast.
- **Metric & Kill Rule:** Within-sparsity Kendall $\tau$ on validation perplexity. If the world model does not achieve $\tau \ge \tau_{\text{Wanda}} + 0.15$ ($p < 0.01$), **kill world-model pruning permanently**.

### Experiment 2: The Localised Unlearning Relearning Gate (Local GPU, ~2 GPU-h, $0)
- **Model & Data:** Qwen3.5-0.8B on TOFU 1% fictitious author split (synthetic only; zero real PII).
- **Interventions at Matched Edit Norm:** (a) Predicted localized deletion; (b) Random deletion; (c) Task-vector negation; (d) NPO baseline.
- **Audit & Kill Rule:** 10-step benign relearning (Hu et al.) and Layer 12 linear probing (Patil et al.). If predicted localized deletion exhibits $> 50\%$ knowledge recovery or probe accuracy exceeds chance by $> 15\%$, **kill localized weight unlearning permanently**.

## 5. Evidence Base and Fetch Log

All external citations were fetched and verified via HTTP:
- Hase et al. (2023): [arXiv:2301.04213](https://arxiv.org/abs/2301.04213)
- Hu et al. (2024): [arXiv:2406.13356](https://arxiv.org/abs/2406.13356)
- Maini et al. (2024, TOFU): [arXiv:2401.06121](https://arxiv.org/abs/2401.06121)
- Shi et al. (2024, MUSE): [arXiv:2407.06460](https://arxiv.org/abs/2407.06460)
- Li et al. (2024, WMDP / RMU): [arXiv:2403.03218](https://arxiv.org/abs/2403.03218)
- Meng et al. (2022, ROME): [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
- Meng et al. (2022, MEMIT): [arXiv:2210.07229](https://arxiv.org/abs/2210.07229)
- Sun et al. (2023, Wanda): [arXiv:2306.11695](https://arxiv.org/abs/2306.11695)
- Frantar & Alistarh (2023, SparseGPT): [arXiv:2301.00774](https://arxiv.org/abs/2301.00774)
- Ma et al. (2025, PPF): [arXiv:2508.02381](https://arxiv.org/abs/2508.02381)
- Zhang et al. (2024, NPO): [arXiv:2404.05868](https://arxiv.org/abs/2404.05868)
- Ilharco et al. (2022, Task Vectors): [arXiv:2212.04089](https://arxiv.org/abs/2212.04089)
- Patil et al. (2024, Eight Methods): [arXiv:2402.16835](https://arxiv.org/abs/2402.16835)
- Duan et al. (2024, MIA in LLMs): [arXiv:2402.07841](https://arxiv.org/abs/2402.07841)

**Recorded Fetch Failure:** Verification probe `https://arxiv.org/abs/2402.9999999` returned HTTP 404. No unverified sources are cited.

## Decision table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Reject unconstrained WM pruning | Saliency (Wanda/SparseGPT) is already first-order prediction; r13 proves beyond-dose failure | F9-P 24-mask ranking panel on Qwen-0.8B | < 1 GPU-h, $0 | Δτ < 0.15 over Wanda: kill WM pruning |
| Reject locate-then-delete unlearning | Hase et al. refutes localization-informed editing; localized deletion leaves circuits intact | TOFU 1% deletion vs random/NPO on Qwen-0.8B | ~2 GPU-h, $0 | Relearning recovery > 50%: kill localized deletion |
| Enforce relearning & probe audits | Forget QA metrics measure suppression; Hu et al. and Patil et al. show prompt/probe leakage | 10-step benign relearning + linear probe | 0 (bundled in gate) | Probe accuracy > 15% above chance: reject unlearning claim |
| Restrict to synthetic-only data | Binding privacy boundary; TOFU author format; zero real patient/personal data | Audit of synthetic corpus before fine-tuning | 0, $0 | Any non-synthetic data: immediate abort |
| Freeze Lambda spend for exploratory tests | C1 has priority; local 32GB GPU is sufficient for 0.8B gating | Local GPU gating before any external pre-reg | 0, $0 | Paid compute proposed: chair veto |
