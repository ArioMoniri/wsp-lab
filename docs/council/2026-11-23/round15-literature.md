# Round 15, literature seat: saliency, localisation and unlearning (23 Nov 2026)

## BLUF

1. Saliency (Wanda, SparseGPT) predicts pruning damage well on average, but papers report perplexity, not per-mask rank agreement. I found no LLM paper reporting τ against realised loss change, so F9-P's ranking panel would be new.
2. Perplexity hides knowledge loss: knowledge tasks drop at 25–30 % sparsity while perplexity holds to 50–60 % (LLM-KICK). Ground truth needs a knowledge endpoint.
3. Localisation does not reliably say where to edit. Causal tracing does not predict the best layer (Hase et al.), and pruning-based localisers lead but are not item-specific (Chang et al.).
4. Mechanism-level localisation is the exception: the gain from choosing weights shows up as robustness to relearning (Guo et al.), not as forget accuracy.
5. Tested approximate unlearning (GA, NPO, RMU families) mostly suppresses. Relearning, 4-bit quantisation (about 21 % → 83 % retained), probing and pre/post logit differences all recover "forgotten" content.
6. TOFU (MIT, 200 GPT-4-made fictitious authors) fits our boundary but has no medical data. Synthetic records must be ours, as in arXiv 2505.24379.
7. A WM must beat Wanda/q_A, ‖W⊙M‖² and sparsity (pruning), and forget–retain saliency, random and layer-only sets at equal size (unlearning). Our r13 result predicts it will not.
8. No fetched repository lists Qwen3.5 (hybrid Gated-DeltaNet); OpenUnlearning lists Qwen2.5 and Gemma, Wanda lists LLaMA/OPT. Ports are needed.

## Q1. State of the literature

**Pruning saliency.**
- Wanda scores weights by magnitude × input-activation norm per output, with no update. It "significantly outperforms" magnitude pruning and is competitive with SparseGPT on LLaMA/LLaMA-2 [W]. Board A F9a.8: Wanda equals our q_A up to a factor; it is not task Fisher q.
- SparseGPT (OBS reconstruction) reaches 50 % one-shot "at minimal loss of accuracy", and 60 % with negligible perplexity increase [S].
- PPF, the one learned predictor, scores a structured policy in 1.52 s instead of 1 min 38 s, with error below 0.0011, on Llama2-7B and Llama3-8B [P]. RESEARCH_NOTES item 20: built from 775 masks; predicts divergence to the dense model.
- Predictive validity is thin. TraceNAS's gradient-trace proxy over Llama/Qwen gives no correlation figures in its abstract [T]. Within-sparsity ranking, as in F9-P, is the untested object.
- Knowledge vs general ability: LLM-KICK reports large knowledge-task drops "at trivial sparsity ratios (e.g., 25-30%)", while perplexity is flat to 50–60 %. N:M fails there; retrieval and summarisation stay robust at 50 % or more [K]. Our 10/20 % rungs (RESEARCH_NOTES item 15) are shown safe on perplexity, not on facts.

**Localisation and editing.**
- ROME places factual recall in mid-layer MLPs at subject tokens; a rank-one edit keeps specificity and generalisation [R]. MEMIT scales this to thousands of edits [M].
- Hase et al.: tracing gives no "insight into which model MLP layer would be best to edit"; the layer choice predicts success far better [H].
- Chang et al. (INJ/DEL benchmarks): pruning-based localisers rank best, but the neurons they find are not specific to one sequence [C].
- Guo et al.: localising to the fact-lookup mechanism is more robust to relearning and has fewer side effects than output-tracing localisation (no figures in the abstract) [G].
- Weight-choosing unlearning: Selective Pruning (neuron importance per capability) [SP]; WAGLE (forget/retain weight attribution inside GradDiff/NPO, Zephyr-7B, Llama2-7B) [WA]; SAU (unlearning degrades on already-sparse models) [SAU]. None reports a calibrated forecast of the edit's effect. **Fetch failure:** the EMNLP 2026 Forget-Retain Pruning PDF was unreadable; it is not cited.

**Unlearning methods and benchmarks.**
- GA collapses. NPO drifts towards collapse exponentially slower and is the first method with "reasonable" TOFU results when forgetting 50 % or more [N].
- RMU lowers WMDP accuracy (3,668 MCQs) while keeping general biology and CS ability [WM]. Task-vector negation lowers target performance with little control change [TA].
- TOFU: none of its baselines unlearn effectively [TO]. MUSE (six criteria, 8 methods, 7B models): only one method avoids severe privacy leakage, and most fail sequential or large requests [MU].
- OpenUnlearning (MIT) bundles GA, GradDiff, NPO, SimNPO, RMU and more over TOFU, MUSE, WMDP and DUET [OU].

**Forgetting vs suppression.**
- Benign relearning on loosely related data restores knowledge; public medical articles revived hazardous bio content [HU].
- The WHP model keeps extractable latent knowledge [LY].
- 4-bit quantisation raises retained knowledge from about 21 % to about 83 % [Q].
- Even exact retraining leaks when both pre- and post-unlearning logits are open. In some cases this about doubles extraction, and it was shown on a simulated medical-diagnosis set [WU].
- Medical/PII: the only fetched evidence is [WU] and [HU]. Other search hits (AMNESIA, REMEDI/MIMIC-III, MedForget, UnPII, K-Bench) were not fetched and are not cited; the ones built on real clinical data are outside our boundary.
- **Not found:** any test of whether a *predicted* weight set beats random and saliency-matched sets at equal size.

## Q2. Is either idea a world-model problem?

- **Pruning.** Saliency *is* the literature's predictor. PPF needed 775 evaluated masks over a low-dimensional space [P]. The baseline is the best of {a + q, q_A ≡ Wanda, −‖W⊙M‖²} within sparsity, plus sparsity level (r13's "pooled + dose" analogue). A learned WM is warranted only if residuals are structured (F9.7). The likeliest such place is knowledge-endpoint damage, where perplexity-based saliency may mislead [K].
- **Unlearning.** "Where the fact is" and "where editing works" differ [H][C]; the value of choosing weights shows up as robustness [G]. The WM must beat random, forget–retain gradient/Fisher saliency (WAGLE-style) and layer-only (Hase) at equal edit size, on relearning-resistant forget quality.

## Q4. Literature input to the unlearning design

- **Data:** TOFU authors (MIT, GPT-4-generated) [TF], plus newly synthesised medical records attached to fictitious persons, never derived from AMNESIA, REMEDI or MIMIC. Insert them into a local Qwen3.5 copy (Apache-2.0 [QW]).
- **Architecture:** Qwen3.5-0.8B is 24 layers, 3×Gated-DeltaNet + 1×Gated-Attention, each with a dense FFN [QW]. MLP edits are well defined, but causal-tracing code needs porting.
- **Forget battery:** forget-set QA and paraphrases; MUSE-style leakage; benign relearning [HU]; 4/8-bit quantisation [Q]; probes or logit lens [LY]; pre/post logit-difference extraction [WU].
- **Retain battery:** neighbour persons, TOFU real_authors/world_facts, and one general benchmark.

## Q6. Ethics and claims

- Cooper et al. (38 authors) argue that unlearning is "not a general-purpose solution" for privacy or copyright [CO]. [WU] shows that even exact unlearning can increase leakage when both models are exposed.
- Our reports must therefore claim only "suppression under the listed attacks, at the listed budgets". They must never claim erasure, a privacy guarantee, or GDPR compliance, and must never release both the pre- and post-unlearning weights.

## Sources (all fetched 23 Nov 2026 unless marked)

[W] https://arxiv.org/abs/2306.11695 · Wanda repository https://github.com/locuslab/wanda (MIT; LLaMA/OPT; magnitude, wanda, sparsegpt) · [S] https://arxiv.org/abs/2301.00774 · [P] https://arxiv.org/abs/2508.02381 · [T] https://arxiv.org/abs/2602.02891 · [K] https://arxiv.org/abs/2310.01382 · [R] https://arxiv.org/abs/2202.05262 · [M] https://arxiv.org/abs/2210.07229 · [H] https://arxiv.org/abs/2301.04213 · [C] https://arxiv.org/abs/2311.09060 · [G] https://arxiv.org/abs/2410.12949 · [SP] https://arxiv.org/abs/2403.01267 · [WA] https://arxiv.org/abs/2410.17509 · [SAU] https://arxiv.org/abs/2602.00577 · [N] https://arxiv.org/abs/2404.05868 · [WM] https://arxiv.org/abs/2403.03218 · [TA] https://arxiv.org/abs/2212.04089 · [TO] https://arxiv.org/abs/2401.06121 · [TF] https://huggingface.co/datasets/locuslab/TOFU · [MU] https://arxiv.org/abs/2407.06460 · [OU] https://github.com/locuslab/open-unlearning · [HU] https://arxiv.org/abs/2406.13356 · [LY] https://arxiv.org/abs/2402.16835 · [Q] https://arxiv.org/abs/2410.16454 · [WU] https://arxiv.org/abs/2505.24379 · [CO] https://arxiv.org/abs/2412.06966 · [QW] https://huggingface.co/Qwen/Qwen3.5-0.8B. **Fetch failure:** https://aclanthology.org/2026.emnlp-main.1956.pdf (unreadable PDF). Abstract-level reads only.

## Decisions

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Prunability WM must beat max(Wanda/q_A, a+q, −‖W⊙M‖²) + sparsity, within sparsity | Saliency is the literature's predictor [W][S]; r13 "pooled + dose" analogue | Kendall τ on held-out masks, permutation null within sparsity | CPU for the predictor; mask evaluation on local GPU, $0 | No τ gain over the best baseline: report saliency only, no WM |
| Add a knowledge endpoint (TOFU world_facts / factual QA) next to WikiText | Knowledge drops at 25–30 % while perplexity holds [K] | Separate τ per endpoint, declared before scoring | Local, $0 | Endpoints agree: keep WikiText primary |
| Unlearning arm: synthetic persons + synthetic medical records on a local Qwen3.5-0.8B; TOFU authors as the second set | Boundary, Apache/MIT licences [QW][TF] | Insertion verified (forget QA high) before any unlearning | Local, $0 | Insertion fails: use TOFU only |
| Main contrast: predicted weight set vs random vs forget–retain saliency vs layer-only, at equal edit size | Open niche; Hase/Chang/Guo [H][C][G] | Relearning-resistant forget score at matched retain loss | Local, $0 | Predicted ≤ saliency: close, report the negative |
| Forget battery must include relearning, 4-bit quantisation, probe and pre/post-diff | Suppression evidence [HU][Q][LY][WU] | All four run on every arm | Local, $0 | Item missing: label arm "suppression untested" |
| Port the tooling (Wanda, OpenUnlearning) to the Qwen3.5 hybrid; no MedGemma insertion | Repositories list LLaMA/OPT/Qwen2.5/Gemma only [OU][W] | Reproduce one Qwen2.5 TOFU NPO run first | CPU/local, $0 | Port fails: Qwen2.5 local fallback (licence to be checked) |
| Wording: "suppression under listed attacks", no guarantees, never release paired weights | [CO][WU] | Chair wording check | $0 | — |
