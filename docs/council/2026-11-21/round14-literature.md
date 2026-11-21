# Round 14, literature seat: weight watermarking, fingerprinting and lineage (Q1, Q2, Q5)

**BLUF**
1. Weight-level watermarks come in two kinds. Function-preserving ones hide bits in symmetries (Fernandez et al.). Function-altering ones perturb chosen weights (EmMark). Both need an owner key (often the original weights) to verify.
2. White-box lineage fingerprints are mature. LeaFBench gives HuRef and PDF an AUC of 0.98–1.00 under fine-tuning, PEFT, quantisation, merging and distillation, while black-box methods mostly sit at 0.5–0.74.
3. Code exists for HuRef, REEF, LeaFBench and Scalable Fingerprinting. None of these papers reports results on Qwen3.5, Gemma 3 or MedGemma.
4. I found no public statement that Qwen3.5 or MedGemma carry a weight watermark or fingerprint. SynthID Text is an opt-in logits processor, not a weight mark.
5. All four anomalies have documented standard causes, so none needs a watermark to explain it.
6. A function-preserving mark cannot change outputs; a function-altering one is fixed and cannot depend on batch composition.
7. Qwen3.5 is Apache-2.0 with no use restrictions. Any modified MedGemma is a HAI-DEF "Model Derivative", so keep such copies internal.
8. Recommend a CPU-only lineage pilot (MedGemma vs Gemma 3 positive, Qwen vs Gemma negative); class-level robustness reporting only.

## Q1. Literature

### Weight-embedded watermarks (white-box, keyed)

**Functional invariants** ([2310.11446](https://arxiv.org/html/2310.11446v2))
- **Embeds:** an identifier, written with three function-preserving operations: permutations, RMSNorm scaling/unscaling with compensation in the adjacent linear layers, and QK rotations.
- **Access:** non-blind white-box (needs original weights).
- **Robustness (LLaMA-7B, Table 1, byte accuracy %):**
  - Scaling invariant: 100 under noise, 98.1 under 3-bit quantisation, 100 under 50 % pruning, 100 under fine-tuning.
  - p-values are "always bellow 10^-60".
- **Relevance:** same symmetry family as our rescale finding; not evidence of a mark.

**EmMark** ([2402.17938](https://arxiv.org/html/2402.17938))
- **Embeds:** ±1 changes on INT8/INT4 weights at saliency-scored, seeded positions: 300 bits per INT8 layer, 40 per INT4 layer.
- **Access:** verification needs the seed, the original quantised weights and full-precision activations.
- **Robustness:**
  - WER stays "over 99%" under parameter overwriting and "over 95%" under re-watermarking.
  - Forgery probability is 9.09×10⁻¹³ⁿ.
  - Fine-tuning, pruning and re-quantisation were not tested quantitatively.
- OPT and LLaMA-2; no code link.

**Newer schemes, abstract only, no numbers on the fetched pages**
- SEAL ([2511.11356](https://arxiv.org/abs/2511.11356)): latent-space multi-bit signatures, verifiable white- or black-box.
- MergePrint ([2410.08604](https://arxiv.org/abs/2410.08604)): a black-box fingerprint optimised to survive merging.
- LineageMark (arxiv.org/html/2606.17123): **fetch failed (HTTP 404)**, so I have not used it.

### Invariant and representation fingerprints (non-invasive, white-box)

**HuRef** ([2312.04828](https://arxiv.org/html/2312.04828); [code](https://github.com/LUMIA-Group/HuRef))
- **Measures:** three invariant terms built from attention and FFN weights, designed to resist permutation and rotation. They are published as an image together with a zero-knowledge proof (ZKP).
- **Results:**
  - Offspring vs LLaMA-7B: ICS 91.35–99.95. Independent models: 0.03–3.16, and 0.38 on average across 28 models.
  - Correct base identified for 51 of 51 offspring.
- **Not tested:** pruning and merging.

**REEF** ([2410.14273](https://arxiv.org/html/2410.14273); [code](https://github.com/tmylla/REEF))
- **Measures:** CKA between hidden representations on shared inputs, so it needs activation access.
- **Results vs Llama-2-7b:** fine-tunes (5M–700B tokens) 0.9947–0.9991; structured pruning ≥0.9278; merging 0.9374–0.9635; permutation and ×0.8 scaling 1.0000.

**LeaFBench SoK** ([2508.19843](https://arxiv.org/html/2508.19843v2); [code](https://github.com/shaoshuo-ss/LeaFBench))
- 149 models, 13 edit techniques; no pruning.
- **Table 4 AUC:**

| Method | FT | PEFT | Quant | Merge | Distil |
|---|---|---|---|---|---|
| HuRef | .984 | 1.0 | 1.0 | .985 | 1.0 |
| PDF | .981 | 1.0 | 1.0 | 1.0 | 1.0 |
| REEF | .843 | .986 | .851 | .936 | .843 |
| Gradient | .731 | .772 | .749 | .818 | .886 |
| TRAP | .619 | .661 | .743 | .670 | .567 |

- LLMmap, MET, SEF: 0.505–0.589; parameter-altering edits make black-box fingerprints "collapse".
- **Q3 prior:** "Gradient", the closest analogue to our J1, is the weakest white-box method.

**SeedPrints** ([2509.26404](https://arxiv.org/abs/2509.26404))
- Initialisation-seed biases; LLaMA- and Qwen-style models; no numbers on the abstract page.

### Behavioural trigger fingerprints (invasive, black-box verification)

- **Instructional Fingerprinting** ([2401.12255](https://arxiv.org/abs/2401.12255)): plants a private-key instruction backdoor. Tested on 11 LLMs, and claims robustness to PEFT. Code is on the [project page](https://cnut1648.github.io/Model-Fingerprint/).
- **Scalable Fingerprinting** ([2502.07760](https://arxiv.org/abs/2502.07760); [code](https://github.com/SewoongLab/scalable-fingerprinting-of-llms)): 24,576 fingerprints in Llama-3.1-8B that "persist" after SFT. No numbers on merging.

### Testability on our models

None of the papers tests Qwen3.5, Gemma 3 or MedGemma.
- **REEF** is activation-only, so architecture-agnostic.
- **HuRef**, by my inference, would cover only the 8 gated-attention layers of Qwen3.5-9B, not its 24 Gated DeltaNet layers ([card](https://huggingface.co/Qwen/Qwen3.5-9B)). It is also unverified with Gemma's (1+w) norms.
- **MedGemma** is "built based on Gemma 3" ([card](https://huggingface.co/google/medgemma-1.5-4b-it)), which gives a public known-positive lineage pair.

## Q2. Are our anomalies watermark-related?

**Public statements.** None found.
- The Qwen3.5-9B card and the [Qwen3.5 blog](https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894) say nothing about watermarks, fingerprints or provenance. The tech report is still "upcoming".
- Nor do the MedGemma 1.5 card or the [Gemma 3 report](https://arxiv.org/html/2503.19786) (no watermarking or SynthID).
- [SynthID Text](https://huggingface.co/blog/synthid-text) is active only when `watermarking_config=` is passed to `generate()`.
- Absence of a statement is not proof of absence (defensive test in the table).

**1. Post-norm cancellation.** Fully explained by the architecture.
- In [modeling_gemma3.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma3/modeling_gemma3.py), `post_attention_layernorm` and `post_feedforward_layernorm` are applied to each sublayer's output before the residual add. Gemma3RMSNorm returns `output * (1.0 + self.weight)`.
- RMSNorm is scale-invariant, so a ×2 rescale of `o_proj` or `down_proj` cancels up to ε. That matches our measured ΔNLL of +0.0103 for the rescale against 12.3 for noise of the same norm (`docs/reviews/wmm/wmm-gate0-2026-11-19.md`).
- A watermark could exploit this symmetry (Fernandez scaling) but cannot cause it.

**2. Batch non-determinism.** Explained by known kernel behaviour.
- [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) traces it to kernels that are not batch-invariant: RMSNorm, split-K matmul and split-KV attention. Qwen3-235B at temperature 0 gave 80 unique completions out of 1000, and all 1000 were identical once batch-invariant kernels were used.
- A maintainer in [transformers#23017](https://github.com/huggingface/transformers/issues/23017): batching reorders floating-point operations, flipping near-tied tokens.
- Our data fit this explanation:
  - The teacher-forced path and vLLM are bitwise reproducible, but HF `generate` at batch 128 is not (`docs/reviews/wme/wme-c1-launch-2026-11-20.md`).
  - Changing batch composition alone flips 13.5 % of MedQA and 14.3 % of MedMCQA answers (`docs/reviews/wmm/wmm-gate1-2026-11-20.md`).
- Fixed weights cannot produce dependence on batch composition.

**3. Headroom ceilings.** Explained by a strong base model.
- Replicates across two families (`docs/council/2026-11-21/RULING-wmm-gate1.md`).
- The schemes above all claim to be harmless to quality, so none would cap edit gains.

**4. (a) non-identifiability.**
- R13 attributes this to a dose-dominated tie and the noise floor (`docs/council/2026-11-20/SYNTHESIS-r13.md`, lines 34–35 and 127). That explanation is sufficient.
- Caveat: weight symmetries make *parameters* non-identifiable, so features should be invariant to them.

## Q5. Licence, ethics, disclosure

**Qwen3.5, Apache-2.0** ([LICENSE](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE))
- Derivative works allowed; no use restrictions.
- If we redistribute, modified files need "prominent notices" (§4(b)), and trademarks are excluded (§6).

**MedGemma, HAI-DEF terms** ([terms](https://developers.google.com/health-ai-developer-foundations/terms))
- §1.1(h) makes "all modifications" Model Derivatives. That covers every bank edit and any fingerprinted copy.
- Distribution must pass on §3.2 restrictions, the agreement and modified-file notices.
- Patient-facing research counts as Clinical Use.
- No clause on watermarks, provenance or reverse engineering.

**HAI-DEF prohibited-use policy** ([policy](https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy))
- It bars "attempts to override or circumvent safety filters" and misleading health-expertise claims.
- It is silent on watermark removal; the brief rules that out anyway.

**Implications**
- Keep MedGemma derivatives internal.
- Passive lineage only on MedGemma; trigger insertion on Qwen only.
- Report degradation as "edit class X degrades scheme Y", no recipes; notify authors first.

**Fetch log**
- Failed: LineageMark (HTTP 404).
- Abstract-only, no numbers: SEAL, MergePrint, SeedPrints, IF, Scalable Fingerprinting.

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Close the hypothesis that watermarks cause our anomalies | Every anomaly has a standard cause, and no public mark statement exists | Note it in the synthesis | 0, $0 | Reopen if the lineage pilot shows unexplained keyed structure |
| CPU lineage pilot: HuRef plus weight cosine | Needs static weights only, and code exists. Known +/− pairs give ground truth | Separation of positive from negative pairs. Use approved local weights, run by an authorised seat | 0, $0 (CPU) | If HuRef fails on Gemma norms or DeltaNet, use plain per-block cosine |
| REEF as the activation baseline for Q3 | Architecture-agnostic; LeaFBench AUC .84–.99 | Same pairs, existing caches only | ≤1 local, $0 | Drop if there are no caches. No new generation without a ruling |
| LeaFBench "Gradient" as the bar for J1 | It is the closest analogue, at AUC .73–.89 | J1 must beat Gradient on the same pairs | Within Q3 | If J1 does not beat Gradient, J1 is not a lineage tool |
| Trigger-fingerprint robustness study on Qwen only | Apache-2.0 allows it; HAI-DEF derivatives carry duties | Class-level reporting and author notice | Per the Q3 costing | Descriptive side result only. No removal recipes |
| Keep MedGemma edits internal | HAI-DEF §1.1(h) and §3.1 | Licence check before any release | 0, $0 | Share metrics only |
