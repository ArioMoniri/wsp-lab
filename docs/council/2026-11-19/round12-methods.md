# Round 12, methods seat: MedGemma as a second target model (19 Nov 2026)

## BLUF

1. MedGemma is not a replication of the frozen 9B claims. It has a different base (Gemma 3), it is domain-tuned, and it has no 9B bank or planner. Register it as a separate **study M**: its own pre-registration and α, never pooled with 9B (PREREG §10, §11.2).
2. Worth paying for: **can a domain-tuned model be enhanced at matched truncation?** "Predict its edits" only replicates a negative.
3. Claim (a) will not be rescued. In W2 the nested WM tied the pooled curve (d 0.003, Holm LB −0.053, 15 runs). The cause is structural. If run, (a) is pre-declared as a **replication of the negative** (equivalence-style).
4. **Primary M-E1:** generated accuracy on MedMCQA and MedXpertQA at a budget with base truncation ≤ 2 %. Conjunctive: untruncated-in-both LB > −1 pp.
5. **M-SYC:** medical-fact sycophancy, as its own family. Brier calibration, harmful compliance and benign over-refusal are conjunctive guards.
6. The 9B discordances do not transfer. A dev pilot measures d, and N follows. Planning range: 1,047–2,631 items per benchmark.
7. Target: MedGemma 1.5 4B on the local GPU. The medical H-E2 is deferred unless 9B H-E2 is supported.
8. Kill gates: licence, headroom, truncation, scorer audit, pool and sycophancy floor. All are decided before any sealed read.

## 1. External facts (fetched URLs only)

- **Variants and architecture.** Model card (https://developers.google.com/health-ai-developer-foundations/medgemma/model-card):
  - MedGemma 1.5 4B (multimodal), MedGemma 1 4B and MedGemma 1 27B;
  - built on Gemma 3, with GQA and a SigLIP encoder.
- **27B-text** is instruction-tuned only, and its results use "test-time scaling" (https://huggingface.co/google/medgemma-27b-text-it).
- **Gemma 3 attention.** It raises the local-to-global attention ratio (https://arxiv.org/abs/2503.19786). That is not Qwen3.5's hybrid attention, so `src/wsp/wm` needs porting.
- **Model-card text scores** (Gemma 3 4B / MedGemma 1 4B / MedGemma 1.5 4B):

  | Benchmark | Gemma 3 4B | MedGemma 1 4B | MedGemma 1.5 4B |
  |---|---|---|---|
  | MedQA | 50.7 | 64.4 | 69.1 |
  | MedMCQA | 45.4 | 55.7 | 59.8 |
  | PubMedQA | 68.4 | 73.4 | 68.2 |
  | MMLU-Med | 67.2 | 70.0 | 69.6 |
  | MedXpertQA | 11.6 | 14.2 | 16.4 |

  Vendor decoding; not our endpoints.
- **Contamination.** The card itself warns of "a risk of data contamination".
- **Licence.** The HAI-DEF terms define "Model Derivatives" and do not prohibit them; modified files need notices (https://developers.google.com/health-ai-developer-foundations/terms).
  - The terms **do not address publishing results**.
  - The prohibited-use policy bars attempts to "override or circumvent safety filters" (https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy). An edit that lowers harmful refusal is therefore both a guard failure and a licence risk.
- **Technical report.** Fine-tuning cut EHR-retrieval errors by 50 % (https://arxiv.org/abs/2507.05201). That is not evidence of QA headroom.
- **Fetches.** None failed. Pool sizes were not found, and none are stated.

## 2. Value

**(i) Replication of the 9B claims: no.**
- H-E2's power (0.828) is model-based on the 9B library (§II.2), and S-BoN power is 0.006 (`wme-stagew` §5).
- A new model needs a new bank and field. Calling that a "replication" invites pooling.

**(ii) A domain study: yes, narrowly.**
- On 9B, the untruncated-in-both differences were −2.4 / −2.1 pp at 2,048 tokens and −3.3 / −2.7 pp at 4,096 (§II.8; SYNTHESIS-r11 R1).
- The same matched-truncation test on a domain-tuned model is publishable in either direction.

**P(b) is not raised.**
- Domain tuning has already harvested the gain: +18.4 pp MedQA, Gemma 3 4B → MedGemma 1.5 4B.
- MedXpertQA (16.4) has the most nominal headroom and the highest truncation risk.
- PubMedQA regressed between versions, which makes it a poor primary.
- MMLU-Med is excluded. MMLU-Pro derives from MMLU and is a 9B sealed source.

**(a).**
- In W, the pooled curve beat the WM in every setting (0.624 vs 2.336 at k = 0).
- In W2, the nested model equalled the pooled curve (C 0.004 at k = 0; exactly 0 at k = 2, where s = 0 in all 15 runs).
- The cause, small edit-specific variance against the path-type means, is not model-specific.
- **Registration as M-T**, at an equivalence margin C_eq = 0.10, with three outcomes:
  - **"Negative replicated"** if the one-sided 97.5 % UB of C (nested vs pooled, k = 0) is < 0.10.
  - **"Separate MedGemma finding; does not rescue 9B"** if the LB is > 0.
  - **"Inconclusive"** otherwise.
- **Transfer check.** Back-solving W2's bound gives SE(C) ≈ 0.032, so the UB is ≈ 0.067 at 15 runs. This is approximate, and **it is not known to transfer**.
- **Priority:** lowest. Run M-T on idle GPU only.

## 3. Minimal powered design

**Edits.** No selection on any data (as §II.6.1).
- **E-acc:** one H1-type LoRA (rank 8, α 16, lr 1e-4), self-distilled from MedGemma's gold-verified answers on construction items. Final checkpoint, λ 1.
- **E-syc:** one LoRA on medical suggested-wrong vs neutral pairs.
- Expect the over-refusal guard to bind: all three 9B sycophancy LoRAs breached it (`wme-pilot2`).

**Endpoints.**
- **M-E1:** paired Δ accuracy on MedMCQA and MedXpertQA-text.
  - Greedy, frozen batch, one instance.
  - Budget B* is set on dev as the smallest budget with base truncation ≤ 2 %.
  - Tango interval, Holm over 2 at α 0.025.
  - **Conjunctive:** untruncated-in-both LB > −0.01.
  - Wording: "accuracy at budget B*".
- **M-SYC** (α 0.025): ΔAcc under suggested-wrong prompts > 0, with a frozen within-type wrong-answer rule. **Conjunctive:** neutral ΔAcc LB ≥ −0.03.
- **Guards**, conjunctive, each at α 0.025:
  - Brier score: UB ≤ +0.01.
  - Harmful-medical compliance: UB ≤ +0.02.
  - Benign over-refusal: UB ≤ +0.02.
  - The scorer gates only after ≥ 0.9 agreement with a two-judge audit. The 9B audit (0.986) does not transfer.

**Power.** N = ⌈(z₁₋α + z₁₋β)²(d − δ²)/δ²⌉ at power 0.8 (round11-methods).
- The 9B discordances do **not transfer**: edits 0.129 / 0.191 (Wilson upper 0.167 / 0.235), and null 0.046–0.092 (`wme-stagew`, `wme-stagew2`).
- The pilot's Wilson-upper d and half the de-biased pilot gain set N.

| d | δ 0.02 | δ 0.03 | δ 0.05 |
|---|---|---|---|
| 0.10 (M-E1, α 0.0125) | 2,367 | 1,047 | 371 |
| 0.15 (M-E1) | 3,555 | 1,575 | 561 |
| 0.25 (M-E1) | 5,932 | 2,631 | 941 |
| 0.15 (M-SYC, α 0.025) | 2,936 | 1,301 | 464 |

- **Shortfall:** use the largest fresh pool, and declare the achieved power before the freeze.
- The 9B ΔAcc_S planning input (0.025) is not reused.

**Panels.**
- Panels `wmm-*` use only medical sources that no 9B panel uses, one item per near-duplicate group.
- Splits: construction, dev, sealed_test, and guard calibration/sealed.

**Multiplicity.** M-E1, M-SYC and M-T are separate families. No study-level "works" statement; nothing from study M supports any 9B or 27B claim.

## 4. Isolation

- **Registration.** A new `docs/PREREG_WMM.md` (PREREG_WME untouched), with its own manifest, `wmm-` salt, seeds and freeze tag.
- **Exposure registry.** A separate, append-only `wmm` list.
  - No 9B source (MMLU-Pro, SuperGPQA, TriviaQA, sycophancy-eval, AdvBench, HarmBench, WikiText-2) enters it.
  - Cross-checks use manifest hashes only. Nothing sealed is opened.
- **Code.** Own branch and venv (`scripts/venv_track.sh`).
- **Timing.** Freeze before any 9B confirmatory outcome. A later freeze is labelled "informed by 9B" (§11.4).

## 5. Kill gates (dev or exposed data only)

| Gate | Rule |
|---|---|
| K0 licence | If publishing edited-model results is not confirmed allowed, stop. |
| K1 headroom | If the dev untruncated-in-both LB ≤ 0 on both benchmarks, drop M-E1 and report "no headroom". |
| K2 truncation | If no budget ≤ 8,192 reaches ≤ 2 % truncation, use untruncated-in-both as primary or drop that benchmark. |
| K3 sycophancy | If the base drops < 5 pp under suggestion, drop M-SYC. |
| K4 scorer | If audit agreement is < 0.9, guards are descriptive and there is no "enhanced" claim. |
| K5 numerics | Canary bitwise repeat; report null discordance. |
| K6 pool | If achieved power is < 0.5, do not run that family. |

## Decision table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Separate study M, not a 9B replication | Different base and domain; §11.2 | Own PREREG_WMM, manifest, salt | 0 | Not registered → not run |
| MedGemma 1.5 4B, local GPU | Lambda has no capacity; 27B uses test-time scaling | Timing pilot, 1.5× re-price | $0 marginal; GPU-h from the timing pilot (unmeasured) | Queue behind shared use |
| M-E1 at matched truncation | 9B gain was budget relief | Holm(2) Tango + untruncated-in-both LB > −1 pp | ≈ 2 models × ΣN_b reads; priced after the pilot | K1/K2: "no headroom" |
| M-SYC on medical facts | H-E1's objective in a new domain | α 0.025; neutral LB ≥ −3 pp | Same order as M-E1 | K3: drop |
| Brier, harm, over-refusal guards | Licence item 2.4; 9B LoRAs over-refused | Conjunctive UBs; audit ≥ 0.9 | In the same read | K4: descriptive |
| (a) as M-T, negative replication | W/W2 mechanism is structural | C UB < 0.10, ≥ 15 runs | W2 took 8.58 H100-h on 9B; local cost unmeasured | Idle GPU only; never rescues 9B |
| Defer the medical H-E2 | New bank needed; S-BoN power 0.006 | Only if 9B H-E2 is supported | 0 | Drop |
| Pilot now; freeze before C1 outcomes; C1 and 27B keep priority | Isolation, no adaptation | Freeze tag first | 0 | Otherwise label "informed by 9B" |
