# Round 12, medical-domain seat: MedGemma

## BLUF

1. MedGemma is a set of Gemma 3 models: 4B (pt and it, multimodal), 27B text, 27B multimodal, and 1.5 4B-it (13 Jan 2026, multimodal only). There is no 1.5 27B [1–4].
2. The HAI-DEF terms allow modification, derivatives and redistribution, with notices. They are silent on publishing benchmarks. They ban circumventing safety filters [5, 6].
3. Q2: choose **(ii), a narrow domain pilot**, not (i). The frozen claims are bound to the Qwen bank and panels.
4. It does not rescue (a). The W2 pooled curve tied the world model, and smooth λ-paths make the pooled curve even stronger.
5. The 4B has paper headroom: MedQA 64.4/69.1 vs 87.7 for the 27B. It is confounded by leakage. MedXpertQA is near the floor, and AfriMed-QA has 25 items.
6. Edits: the Gemma 3 → MedGemma task vector scaled by λ, the 1 ↔ 1.5 vector, and the H1-LoRA. Endpoint: generated accuracy at a frozen budget, untruncated-in-both, on option-permuted items.
7. It runs locally (4B, 32 GB) under its own PREREG, track and panels. C1 and the Qwen 27B run are untouched.
8. Kill it if the base misses the card, or no gain survives truncation matching and permutation, or the guards fail.

## Q1. What MedGemma is

**Variants.**
- **4B:** released 20 May 2025; v1.0.1 on 9 Jul 2025 fixed an image-token bug. SigLIP encoder; context ≥ 128K; output ≤ 8,192 tokens [1].
- **27B text-only:** v1.0.0, base gemma-3-27b-pt [2].
- **27B multimodal:** 9 Jul 2025 [3].
- **1.5 4B-it:** v1.5.0, 13 Jan 2026. Greedy decoding became the default on 23 Jan [4].
- The Hub lists exactly these five Google repositories [7].

**Architecture.** The cards say "same as Gemma 3", with GQA [1–3]. Gemma 3 has [8]:
- a 5:1 ratio of local to global layers, with a 1,024-token window;
- QK-norm;
- a 262k vocabulary (the report also says 256k);
- in the 4B, 417M vision, 675M embedding and 3,209M non-embedding parameters.

There is no linear attention, so `src/wsp/wm/actions.py` (`linear_attn`) and the GatedDeltaNet shims in `hf.py` do not apply.

**Training.** Teacher distillation plus RL [9].

**Licence** [5] (15 Nov 2024).
- "Model Derivatives" covers modifications and works based on the models.
- Use, modification and distribution are allowed "only in accordance" with the terms. Modified files need notices. Recipients get the Agreement and the §3.2 restrictions.
- Clinical use (diagnosis or treatment) needs regulatory authorisation.
- Outputs are ours.
- **No clause covers benchmarking or publication.**

The Prohibited Use Policy [6] bans "Attempts to override or circumvent safety filters" and automated decisions affecting well-being.

**Consequences.**
- No edit may aim to reduce harmful-request refusals, so harm is a non-inferiority guard.
- Do not release edited weights.
- The repositories are gated, and **the user** must accept the terms [1].
- The Gemma 3 base terms were not fetched. Check them before using Gemma 3 as an anchor.

**Published numbers** (zero-shot CoT, "Final Answer: (X)", temperature 0, so generated answers, not log-probs [9]).

| Benchmark | G3 4B | MG 4B | MG1.5 4B | G3 27B | MG 27B text | MG 27B MM |
|---|---|---|---|---|---|---|
| MedQA 4-op | 50.7 | 64.4 | 69.1 | 74.9 | 87.7 (89.8 bo5) | 85.3 |
| MedMCQA | 45.4 | 55.7 | 59.8 | 62.6 | 74.2 | 70.2 |
| PubMedQA | 68.4 | 73.4 | 68.2 | 73.4 | 76.8 | 77.2 |
| MMLU Med | 67.2 | 70.0 | 69.6 | 83.3 | 87.0 | 86.2 |
| MedXpertQA text | 11.6 | 14.2 | 16.4 | 15.7 | 25.7/26.7* | 23.7 |
| AfriMed-QA (25) | 48.0 | 52.0 | 56.0 | 72.0 | 84.0 | 72.0 |

Sources: [1–4, 9].

- \*The 27B-text card [2] says 26.7; the 27B-it card [3] and the report [9] say 25.7.
- The 27B numbers use undefined "test-time scaling" [2, 3].

**Contamination.**
- The cards advise validating on non-public data [1, 2].
- The report has no contamination analysis. It concedes "the possibility of test data leakage" [9].
- MedXpertQA (4,460 questions) used data synthesis against leakage [10].

**Safety.** The cards report "safe levels" but **no refusal rates** [3].

## Q2. Value

- **(i) Replication: no.** H-E1 and H-E2 depend on the Qwen bank, StatNN and the sealed panels. Qwen3.8-27B already provides the second model.
- **(ii) Domain study: yes, narrowly.**
  - The medical recipe moved MedQA by +13.7 pp over Gemma 3 4B [1], so a known accuracy direction exists in weight space, as a task vector.
- **Does a domain model help (b)?** Moderately, but only via task vectors. Pilot 3 showed H1 is budget relief (`docs/reviews/wme/wme-pilot3-2026-11-17.md` §2). MedGemma's gain came from distillation and RL [9]; our LoRA families add no knowledge. This is a prior, not a number.
- **Endpoints:**
  - **MedQA and MedMCQA:** primaries, with a permutation control.
  - **PubMedQA:** a format probe (1.5 dropped 5.2 pp [4]).
  - **MMLU Med:** a guard.
  - **MedXpertQA:** descriptive (floor).
  - **AfriMed-QA:** unusable (SE ≈ 10 pp).
- **Does it rescue (a)? No.** W2 found nested and pooled E of 0.768 vs 0.771 (`wme-stagew2-2026-11-18.md`). Report trajectories descriptively.

## Q3. Design (new PREREG_MED)

**Target.** `medgemma-1.5-4b-it`, text only, pinned revision.

**Edit families.**
- **TV:** λ·(θ_MG − θ_G3), applied to language-model weights only, λ ∈ {0.5…1.5}.
- **TV2:** MedGemma 1 ↔ 1.5.
- **H1-LoRA:** self-distillation on training splits.

**Primary.**
- MedQA and MedMCQA generated accuracy at a frozen budget B (base p95, ≤ 8,192).
- Holm at α 0.025.
- SYNTHESIS-r11 R2 wording applies.
- The lower bound must also exceed 0 on pre-built option-permuted copies.

**Guards** (non-inferiority, frozen UCB margins).
1. **Harm:** MedSafetyBench-style harmful requests [11], judged blind.
2. **Over-refusal:** the OR-Bench-hard panel (audit agreement 0.986, `wme-pilot3` §1(c)) plus benign medical questions. Re-audit both on Gemma output.
3. **Calibration:** ECE and Brier of the answer-letter probability; margin +2 pp.
4. **General:** MMLU-Pro non-health.
5. **HealthBench** [12]: descriptive.

**Panels.** Hash-split dev and sealed in a new namespace before any read. Permuted copies are made at split time.

**Power.** Illustrative only: discordance 0.13 (`wme-pilot3` §2) and N = 1,000 give SE ≈ 1.1 pp, so the minimum detectable effect is about 3.5 pp at α 0.0125 and power 0.8. Check pool sizes before the freeze.

**Reuse:** `signflip`, `confirm_stats`, `powersim*`, `guards_v2`, `runguard`, `scorers`.

**Port:**
- `actions` and `lora` paths, for local and global attention; add the layer type as a feature.
- `hf` pins.
- `controls` inert checks: QK-norm exists [8], so remeasure the `k_proj` control.
- The `p3gen` chat template.
- The multimodal-wrapper prefix.

## Q4. Sequencing

- **Order:** C1 on Lambda, untouched → user accepts HAI-DEF → local pilot → PREREG_MED freeze → local confirmatory → Qwen3.8-27B after the 9B is sealed.
- **Fit:** 4.3B parameters [8], about 8.6 GB in bf16. It fits 32 GB.
- **Isolation:** own worktree, `venv_track.sh` track, caches and panels. No reads of `data/evals`. Use agent-tmux; the GPU is shared.
- **Caps:** port ≤ 3, pilot ≤ 12, confirmatory ≤ 20 GPU-h, all local, $0.
- **No 27B.**

## Q5. Kill criteria

- **K0:** terms not accepted (if only the Gemma base terms block, drop TV).
- **K1:** port fails the null or inert checks within 3 GPU-h.
- **K2:** base misses the card [4] by more than 5 pp after one prompt fix, or more than 5 % truncates at B.
- **K3:** no gain with lower bound > 0 on both untruncated-in-both and permuted items.
- **K4:** the permuted-minus-original gap has an upper bound < 0.
- **K5:** every qualifier fails a guard.
- **K6:** (a) is never claimed.

## Fetch log

- All URLs below were fetched. [9]: test-time scaling and the safety section were not found.
- [7]: the 1.5 date shows no year.
- The Gemma terms were not fetched.

[1] https://huggingface.co/google/medgemma-4b-it
[2] https://huggingface.co/google/medgemma-27b-text-it
[3] https://huggingface.co/google/medgemma-27b-it
[4] https://huggingface.co/google/medgemma-1.5-4b-it
[5] https://developers.google.com/health-ai-developer-foundations/terms
[6] https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy
[7] https://huggingface.co/models?search=google/medgemma
[8] https://arxiv.org/html/2503.19786
[9] https://arxiv.org/abs/2507.05201, https://arxiv.org/html/2507.05201
[10] https://arxiv.org/abs/2501.18362
[11] https://arxiv.org/abs/2403.03744
[12] https://arxiv.org/abs/2505.08775

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| (ii), not (i) | Frozen claims are bound to the Qwen bank | Own PREREG_MED | 0 | (iii) if K0 |
| 1.5 4B-it, text only | Current; local; 20 pp gap to the 27B | K1, K2 | ≤ 3, $0 | Drop |
| TV, TV2, H1-LoRA | TV has a knowledge channel | Untruncated-in-both and permuted lower bound > 0 | ≤ 12, $0 | K3 or K4 → headroom negative |
| Harm, over-refusal and calibration guards | Licence bans safety circumvention | Frozen UCB margins, re-audited | in the pilot | K5 → no claim |
| (a) descriptive | W2 tie with the pooled curve | Pooled comparator | 0 | No claim |
| Local, parallel to C1 | Lambda has zero capacity | Isolated track | ≤ 20, $0 | Pause if contended |
