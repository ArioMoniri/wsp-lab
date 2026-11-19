# Round 12 Challenger Report: The Case Against Adding MedGemma

## BLUF
1. Adding MedGemma as a second target model is strongly opposed: it brings fatal legal risk, severe contamination, and zero scientific rescue for trajectory prediction.
2. Licence terms (HAI-DEF §1.1h, §3.1, §4.5) impose sweeping indemnification for modified weights and forbid circumventing safety filters, creating direct liability.
3. Benchmark headroom is an illusion: MedGemma trained on the train splits of MedQA, MedMCQA, PubMedQA, and AfriMed-QA; on unseen OOD MedXpertQA, 4B collapses to 14.2%.
4. Trajectory prediction (a) will not be rescued: Stage W and W2 proved on Qwen3.5-9B that edit-informed WMs fail to beat pooled path-type curves (W2 zero-shot d = 0.003, few-shot d = 0.000).
5. Enhancement (b) risks repeating Pilot 3 truncation confounding, where apparent reasoning gains evaporated once length was matched (-0.024 on MMLU-Pro).
6. Medical safety guards are undefinable and numerically chaotic: Pilot 3 Stage S proved refusal Hessian-vector products disagree by 54–109% and killed the harmful-refusal margin.
7. Porting cost is prohibitive: `src/wsp/wm/` hardcodes Qwen hybrid attention kernels and 152k vocabularies; MedGemma requires multimodal token handling and Gemma 3 GQA rewrites.
8. Hardware allocation forbids it: Lambda has zero H100 capacity, stalling the frozen 9B confirmatory run ($360 / 66.9 GPU-h); 27B cannot fit locally and 4B risks local OOM.

## 1. What is MedGemma Now? Variants, Licence, and Published Metrics

MedGemma is a suite of medical models built on Gemma 3 (Sellergren et al., [arXiv:2507.05201](https://arxiv.org/abs/2507.05201); [Hugging Face](https://huggingface.co/google/medgemma-4b-it)):
- **MedGemma 4B** (`google/medgemma-4b-it` / `-pt`): Multimodal model pairing a 4B Gemma 3 decoder with a medically pre-trained SigLIP vision encoder (128k context, GQA). MedGemma 1.5 4B ([Google Developers](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)) updated 3D imaging and EHR comprehension.
- **MedGemma 27B**: Instruction-tuned text-only and multimodal variants.

### Licence Terms and Legal Hazards
MedGemma is governed by the Health AI Developer Foundations (HAI-DEF) Terms of Use ([Terms](https://developers.google.com/health-ai-developer-foundations/terms)) and Prohibited Use Policy ([Policy](https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy)):
1. **Model Derivatives (§1.1h, §3.1):** Weight edits and adapters are legally "Model Derivatives." Downstream distribution requires enforceable pass-through of use restrictions, modification notices, and regulatory notifications.
2. **Aggressive Indemnification (§4.5):** Licensees must defend and indemnify Google against *all* liabilities, damages, and fees arising from "modification, Distribution, reproduction, performance, display, or adaptation of HAI-DEF." Evaluating or publishing edited weights that generate erroneous clinical outputs creates immediate organizational liability.
3. **Safety Circumvention Prohibition (§12 Policy):** Strictly bans "Attempts to override or circumvent safety filters or intentionally drive HAI-DEF or Model Derivatives to act in a manner that contravenes this Policy." Steering or ablating refusal representations directly risks breaching this clause.
4. **Gated Distribution:** Access requires authenticated Hugging Face agreement clicks, blocking unattended CI/CD.

### Published Benchmark Numbers
In the technical report ([arXiv:2507.05201](https://arxiv.org/html/2507.05201), Tables 3 & 4; [Model Card v1](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card-v1)):
- **MedGemma 4B:** MedQA: 64.4%; MedMCQA: 55.7%; PubMedQA: 73.4%; MMLU Anatomy: 59.3%; MMLU Clin. Knowledge: 71.3%; MMLU Prof. Medicine: 76.8%; AfriMed: 52.0%.
- **MedGemma 27B (test-time scaled):** MedQA: 87.7%; MedMCQA: 74.2%; PubMedQA: 76.8%; MMLU Coll. Biology: 96.5%; MMLU Med. Genetics: 97.0%; MMLU Prof. Medicine: 93.4%; AfriMed: 84.0%.
- **Out-of-Distribution (MedXpertQA, Table 4):** MedGemma 4B collapses to **14.2%** text-only (base Gemma 3 4B: 11.6%). MedGemma 27B achieves **25.7%** (base Gemma 3 27B: 15.7%; o3: 54.6%).
*(URL audit: 7 URLs fetched successfully; 0 fetch failures).*

## 2. Scientific Value: The Illusions of Domain Specialisation

### Contamination and Saturation Overstate Headroom
Reported scores on MedQA (87.7% on 27B) and PubMedQA (76.8%) do not provide viable headroom:
1. **Explicit Train Leakage:** Sellergren et al. (2025, §3.1) state that instruction tuning used the train splits of MedQA, MedMCQA, PubMedQA, MedExpQA, and AfriMed-QA. The model card explicitly warns that contamination risks overestimating true generalization.
2. **Ceiling Effects:** PubMedQA is saturated at ~74–77% due to annotator noise. MedQA 4-option on 27B (87.7%) leaves narrow margin.
3. **Catastrophic OOD Floor:** On MedXpertQA (Table 4), where leakage was absent, MedGemma 4B drops to 14.2%—near random guessing. There is no robust medical reasoning capability to enhance.

### Repeating the Negative Result on Part (a)
MedGemma cannot rescue trajectory prediction (a):
- In Stage W (`docs/reviews/wme/wme-stagew-2026-11-18.md`), the un-nested WM lost to the simple pooled path-type curve across all horizons (zero-shot $E_{WM} = 2.336$ vs $E_{pool} = 0.624$, $C = -2.740$, $p = 1.000$).
- In Stage W2 (`docs/reviews/wme/wme-stagew2-2026-11-18.md`), the nested WM shrank to the pooled path-type baseline with zero gain (zero-shot $d = 0.003$, Holm LB $-0.053$, $p = 0.450$; few-shot $k=2$, $d = 0.000$, $p = 1.000$).
- Trajectory dynamics are governed by the macro path-family dose curve, not low-dimensional linear encodings. A 4B or 27B Gemma model shares these spectral weight properties. Running (a) on MedGemma will merely re-verify the negative result already frozen in `docs/PREREG_WME.md` §II.23.1.

### Repeating the Truncation Artifact on Part (b)
In Pilot 3 (`docs/reviews/wme/wme-pilot3-2026-11-17.md`, `docs/council/2026-11-17/SYNTHESIS-r11.md`), apparent H1 gains on MMLU-Pro (0.682 to 0.726) vanished once length was matched: untruncated items lost accuracy ($-0.024$, SE $0.009$). The gain was pure budget relief. In MedGemma, where base OOD accuracy is 14.2%, search over weight edits will merely exploit brevity bias rather than clinical reasoning.

## 3. Minimal Design Hazards: Safety Guards and Architecture Porting

### Medical-Safety Guard Invalidity
In WM-E, primary claim H-E1 targets sycophancy on TriviaQA, guarded by surrogate harmless-refusal margin and neutral preservation (`docs/PREREG_WME.md` §II.2).
- In medicine, refusal vs helpfulness carries immediate safety risks. Reducing refusal risks unlicensed medical diagnosis, breaching HAI-DEF Prohibited Use Policy §8 and §22.
- Pilot 3 Stage S (`docs/reviews/wme/wme-pilot3-2026-11-17.md`) established that the exact Hessian-vector product (S1) is numerically unstable on refusal objectives, disagreeing with S2 by 54–109%, while harmful-refusal margin was killed (recovery $-0.05$). First-order transport cannot safely guard clinical outputs against contraindications or toxic dosages.

### Architecture Porting Costs
The codebase `src/wsp/wm/` is tightly bound to Qwen3.5:
1. **Reference Kernels & Mixers:** `src/wsp/wm/hf.py` loads Qwen hybrid linear attention kernels (`torch_chunk_gated_delta_rule`). `src/wsp/wm/actions.py` maps `MIXER_OUT` to `linear_attn.out_proj` and `self_attn.o_proj`. Gemma 3 uses standard GQA with sliding window attention.
2. **Multimodal Weight Pollution:** MedGemma 4B loads as `AutoModelForMultimodalLM` with a SigLIP vision backbone. `src/wsp/wm/actions.py` regexes (`FROZEN_PATTERNS`) do not cover MedGemma vision towers, risking silent weight corruption.
3. **Vocabulary Size:** Gemma 3 uses a 256k vocabulary (`src/wsp/wm/vocabfield.py` and `src/wsp/wm/rootsampler.py` hardcode 152k Qwen token tables).
Porting and certifying theta-hash invariance (`src/wsp/wm/actions.py`) would require 2–3 weeks of unbudgeted engineering.

## 4. Compute, Sequencing, and Isolation

### Hardware Realities and Lambda Starvation
1. **Lambda Capacity Failure:** The 9B confirmatory run (track C1, `run/wme-c1-9b`) is registered for 66.93 GPU-h-eq (~$360, leaving 23.07 under the 90 GPU-h cap; `docs/PREREG_WME.md` §II.23.1). Lambda currently has zero H100 or B200 instances. Track C1 is completely stalled.
2. **Local GPU Limitations:** The local RTX PRO 4500 Blackwell has 32 GB VRAM and is shared across concurrent workspace sessions. MedGemma 27B (54 GB bf16) cannot fit locally. MedGemma 4B (~9 GB) fits for raw inference, but evaluation harnesses with observation hooks (`src/wsp/wm/tap.py`) and candidate batching risk OOM and session eviction.
3. **Protocol Isolation:** Track C1 must remain untouched. Configuring MedGemma via `scripts/venv_track.sh` diverts developer bandwidth while C1 awaits execution.

## 5. What Evidence Would Change My Mind?

To reconsider MedGemma, four strict gates must be met:
1. **Legal Clearance:** Formal institutional legal approval confirming that weight-edited derivatives of MedGemma do not trigger HAI-DEF §4.5 indemnification or violate §12 policy against safety filter alteration.
2. **Verified OOD Headroom:** A sealed, contamination-free medical benchmark (post-dating July 2025) where MedGemma base accuracy is in the 40–60% range.
3. **Stable Safety Guard:** Empirical pilot proof that clinical safety and contraindication guards achieve oracle recovery >= 0.70 under weight edits, resolving the Stage S refusal failure.
4. **Pre-Pilot Trajectory Predictability:** Proof on an exploratory pilot that an edit-specific WM beats the pooled path-type baseline (Holm lower bound > 0), refuting Stage W/W2 findings.

## Decision and Gate Table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| **REJECT MedGemma 27B** | Cannot run on local 32 GB GPU (needs 54 GB); Lambda has 0 H100s; repeats negative (a). | Hardware fit check & Lambda capacity audit | 0 GPU-h, $0 | **KILL permanently.** Retain planned Qwen3.8-27B replication scope only. |
| **REJECT MedGemma 4B** | Gated licence with severe indemnification (§4.5); MedQA/PubMedQA contaminated (§3.1); OOD MedXpertQA at 14.2% floor; porting cost high. | Gate 1 (legal clearance) & Gate 2 (uncontaminated headroom >= 40%) | 0 GPU-h, $0 | **KILL.** Fall back to completing frozen 9B confirmatory run (track C1). |
| **HOLD any medical domain study** | Refusal guards unstable (Pilot 3 S recovery -0.05); risk of clinical harm / malpractice policy violation (§8 Policy). | Gate 3 (oracle guard recovery >= 0.70 on medical safety panel) | 0 GPU-h, $0 | **KILL.** Preserve TriviaQA sycophancy (H-E1) as sole primary enhancement claim. |
| **ENFORCE Stage W/W2 negative reporting** | Trajectory prediction (a) failed against pooled curves (d = 0.000, p = 1.0); MedGemma will merely replicate the negative. | Protocol compliance with PREREG_WME §II.23.1 | 0 GPU-h, $0 | **BINDING.** Report negative result for (a); zero budget allocation for second-model (a) runs. |
