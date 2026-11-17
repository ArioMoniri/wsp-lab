# Council Round 11: Challenger-Engineer Report
## Adversarial Red-Team Audit of Pilot 3 Interim Results and the Post-v2b Research Plan

**Seat:** Challenger-Engineer (Adversarial Red Team, Independent Vendor Perspective)  
**Worktree:** `/home/ubuntu/projects/wsp-lab/.worktrees/council/2026-11-17-r11/`  
**Binding Brief:** `docs/council/2026-11-17/BRIEF-r11.md`  
**Literature Status:** No external URLs fetched; all cited numbers verified directly from repository files.

---

### 10-Line BLUF
1. H1 "gains" (+6.2 pp MMLU-Pro, +9.3 pp SuperGPQA) are artefacts of selection, budget recovery, and regex formatting.
2. Truncation explains H1 lift: base non-thinking truncates on 20% of MMLU-Pro and 38% of SuperGPQA; H1 simply stops rambling.
3. The extractor is brittle regex (`_MC`); H1 mimics the teacher template, converting unparsed base answers into scores without reasoning.
4. The fresh harmless guard is broken: base refusal rate is 0.917 (91.7% false-positive refusal on OR-Bench), mistaking keyword avoidance for safety.
5. In enhancement (H-E2), direct Multi-Fidelity Best-of-N (MF-BoN) will beat R0/R1 world models once R1 field costs are amortized.
6. StatNN is a strawman; beating weight-space nearest-neighbors proves nothing if direct search achieves superior Pareto efficiency.
7. Refitting H-T to 1D checkpoint/λ curves is curve-fitting; it must beat simple 2-parameter power-law learning curves to claim validity.
8. Stage S slope transport failed across pilots (failing 5 of 6 objectives in p2; stalling in p3); drop it to preserve project runway.
9. An honest paper under the $1,000 cap drops S-stage, demotes H-T to descriptive, drops thinking ceilings, and prunes secondaries.
10. Conviction requires: untruncated gain on sealed splits, beating flexible parsers, beating MF-BoN at equal compute, and beating learning curves.

---

### 1. H1 Gain: Real Enhancement or Compound Measurement Artefact?

Interim dev results (`docs/council/2026-11-17/r11-interim-pilot3-dev.md`) show base accuracy of 0.682 on MMLU-Pro and 0.354 on SuperGPQA, rising under H1 to 0.744 (step 52, λ 1.0) and 0.447 (step 208, λ 1.0). Five confounding artefacts invalidate this apparent headroom:

#### 1.1 Checkpoint Selection Bias on Shared Dev Panel
Thirty-two candidates (16 checkpoints × 2 λ values) were scored on the identical 1,000-item dev panel (`wme-p3h`). In any family of 32 correlated draws, the maximum is biased upward ($E[\max X_i] \gg \mu$). In `data/wme/pilot3/run-20261117/h/sizing.json`, the SuperGPQA peak (`step: 208, lam: 1.0`) was picked as sizing probe to lock in $N=1000$. Scoring 32 checkpoints on one panel is in-sample optimization.  
*Cheapest Check:* Split-half cross-validation on `evals.jsonl` (select best on split A of 500 items; score on split B). Cost: 0 GPU-h, CPU ($0.00).

#### 1.2 Truncation and Budget Recovery
Base Qwen3.5-9B (`enable_thinking=False`) is token-starved at budget $B=2048$: base truncation is 20.0% on MMLU-Pro and 37.9% on SuperGPQA (`r11-interim-pilot3-dev.md`). Truncated outputs score strictly `correct: False`. As H1 trains, truncation drops monotonically: MMLU-Pro 0.20 → 0.06; SuperGPQA 0.38 → 0.15. Observed accuracy gain (+6.2 pp on MMLU-Pro, +9.3 pp on SuperGPQA) mirrors truncation reductions (−14 pp and −23 pp). H1 learned to stop rambling before the 2,048-token cutoff rather than acquiring reasoning.  
*Cheapest Check:* Score `evals.jsonl` on items untruncated in both base and candidate (`untruncated_both`). If paired gain LCB $\le 0$, the effect is budget recovery. Cost: 0 GPU-h, CPU ($0.00).

#### 1.3 Answer-Format Extraction Rigidity
`src/wsp/wm/p3gen.py` requires `_MC = re.compile(r"answer is:?\s*\(?\s*([A-P])\s*\)?", re.IGNORECASE)`. Base outputs (e.g. "Therefore, option C is correct") return `None` and fail. H1 trained on `after_think()` from thinking mode, which systematically outputs `"The answer is (A)"`. H1 merely mimics the token pattern that satisfies the regex parser. The length-matched control showed zero gain (0.681–0.703 on MMLU-Pro, 0.353–0.373 on SuperGPQA).  
*Cheapest Check:* Re-parse base generations with flexible regex `(?i)(?:option|choice|answer is|is)\s*[:\*\s]*\(?([A-P])\)?`. Measure residual gap. Cost: 0 GPU-h, CPU ($0.00).

#### 1.4 Contamination and Teacher Data Leakage
H1 trains on 1,000 construction items. Deduplication in `scripts/wme_p3_build_panels.py` uses word-5-gram MinHash at Jaccard 0.7 (`wm.exposure`), missing paraphrases and shared question stems. If construction items share problem stems with dev items, H1 memorized domain solutions.  
*Cheapest Check:* Compute embedding cosine similarity between construction and dev prompts; test if H1 gains concentrate on dev items with similarity $> 0.85$. Cost: 0 GPU-h, CPU ($0.00).

#### 1.5 The Broken Guard Scorer
In `r11-interim-pilot3-dev.md`, base `guard_harmless` refusal rate is **0.917** (378 of 412 items flagged refused); base `guard_harmful` is **0.990**. The brief claims "Guards: they look unchanged." This is an alarming diagnostic blindspot: `GA.detect_refusal` searches for strings like `"i cannot"` or `"sorry"` in the first 80 characters. On OR-Bench Hard, benign disclaimers trip keyword detectors on 91.7% of items. At step 208, H1 harmless refusal drops to 0.854 (a 6.3 pp drop). A detector with a 92% false-positive rate cannot audit safety.  
*Cheapest Check:* Audit 50 base harmless generations flagged as refused to measure true precision. Cost: 0 GPU-h, text audit ($0.00).

---

### 2. Enhancement: World Models vs. Direct Multi-Fidelity Best-of-N

In H-E2, the council proposes using the world model (WM-CP) at R0/R1 to select edits from an H1 library under budget $B=32$ or $48$.

#### 2.1 The Multi-Fidelity Best-of-N (MF-BoN) Reality Check
Computing R1 derivatives at $	heta_0$ required Amendment 1 to raise stage S compute to 6.0 H100 GPU-h. In contrast, direct Multi-Fidelity Best-of-N screens candidates via vLLM batching:
- **Screen:** Evaluate $M=64$ candidates on $N_1=32$ items ($2,048$ gens).
- **Refine:** Advance top 16 candidates to $N_2=128$ items ($1,536$ gens).
- **Final:** Advance top 4 candidates to $N_3=1000$ items ($3,488$ gens).
- **Total Compute:** 7,072 generations $pprox$ **3.9 GPU-h** at batch 128 (P-H100b pricing, `docs/reviews/wme/wme-v2b-2026-11-17.md`). Direct search finishes in less time than computing S-stage R1 derivatives.

#### 2.2 StatNN is a Strawman; Contest Must Be Fair
In v2b, H-E2 nominal power (0.828) came from beating StatNN (weight-space nearest neighbors). Beating StatNN proves nothing if direct search achieves higher accuracy at lower total compute.  
Requirements for a fair contest:
1. **Co-Primary Baseline:** Elevate MF-BoN from descriptive E3 (SYNTHESIS-r10 D7) to co-primary in H-E2.
2. **Total-Compute Parity:** Charge R1 field derivative compute (6 GPU-h) to the world model ledger.
3. **Combinatorial Library ($M \ge 256$):** Test over cross-layer LoRA merges and block-wise scales where $M \gg B$. If WM cannot beat MF-BoN here, enhancement claims fail.

---

### 3. Trajectory Prediction: Genuine Dynamics or Curve-Fitting?

In v2b (`docs/reviews/wme/wme-v2b-2026-11-17.md`), H-T failed: $C_T = -0.327$, 0/156 cells powered, and quadratic null rejection was 0.41 (Flag 1).

#### 3.1 The "Training/λ Trajectory" Fallacy
Pivoting H-T to 1D training steps ($t \in [13, 208]$) or doses ($\lambda$) is learning-curve fitting. Monotonic 2-parameter functions (Power Law $a - b \cdot t^{-lpha}$, Exponential $a - b e^{-kt}$) on early checkpoints ($t \le 39$) extrapolate subsequent points with high precision. Labeling a 1D learning curve a "world-model trajectory" is re-branding.

#### 3.2 Mandatory Baselines and Flag 1
H-T must be preregistered against:
1. **Power-Law Extrapolation:** $acc(t) = a - b \cdot t^{-lpha}$ fitted on early steps.
2. **Monotone Spline:** Monotonic cubic interpolation on early evaluations.
3. **Static First-Order Taylor ($A_0$):** Static gradient projection from $	heta_0$.  
*Decisive Gate:* WM must achieve $\ge 25\%$ lower MATE over power-law extrapolation on held-out steps ($t > 52$). Otherwise, kill H-T.  
*Flag 1 Resolution:* Enforce symmetric shrinkage: both static and WM models must share identical cross-validation regularization penalties, with S1 taking precedence on ties.

---

### 4. What Must Be Dropped for an Honest, Finishable Paper

To deliver an honest paper within caps ($\le 90$ GPU-h, $\le \$1,000$):
1. **Drop Stage S Slope Transport:** Failed in p1 (−4.8%), passed in p2 via refusal distortion (5 of 6 objectives failed; sycophancy worsened 25.6%), and stalled in p3. Demote to appendix.
2. **Drop H-T from Confirmatory Claims:** H-T is unpowered (0/156 cells). Refitting to 1D curves creates an indefensible curve-fitting target. Relegate H-T to descriptive status.
3. **Drop "Thinking-Mode Ceiling":** At budget $B=8192$, thinking mode truncates on 31% to 63% of items (`r11-interim-pilot3-dev.md`). An invalid ceiling must not frame the narrative.
4. **Prune Secondary Objectives:** GSM8K, MATH, and IFEval dev pools are only 300 items ($n_{	ext{rep\_sec}} = 0$ in v2b). Restrict confirmatory testing to primary benchmarks ($N=1000$).
5. **Drop 80-Character Keyword Refusal Scorer:** A guard with 91.7% false-positive rate on harmless prompts cannot gate scientific claims. Replace with calibrated classification.

---

### 5. What Would Convince the Challenger

To convert this red team to a paper advocate, the council must establish:
1. **Untruncated Accuracy Gain on Sealed Splits:** Frozen H1 candidate shows paired gain LCB $> +3.0$ pp ($p < 0.01$) on items untruncated in both base and edit, verified by flexible regex.
2. **Beating Direct MF-BoN at Equal FLOPs:** In a combinatorial library ($M \ge 256$), world model finds higher-performing edits than direct MF-BoN at equal total GPU-hours (including R1 features).
3. **Trajectory Dynamics Beating Learning-Curve Fits:** Trajectory predictions on non-trivial multi-step edit compositions achieve $\ge 25\%$ error reduction over power-law extrapolation ($p < 0.01$).
4. **Audited Safety Verification:** Enhancement does not increase harmful compliance or harmless refusal under a classifier with false-positive rate $< 5\%$.

---

### Decision Table

| Decision | Rationale | Test or Gate | Cost (GPU-h, $) | Kill or Fallback |
|---|---|---|---|---|
| **Gate H1 on Untruncated Items** | Truncation reduction (13–23 pp) masks whether reasoning improved or verbosity dropped. | Paired gain LCB $> 0$ on untruncated items (`untruncated_both`). | 0 GPU-h, $0.00 (CPU) | Kill H1 accuracy claim; fallback to sycophancy H-E1. |
| **Normalize Answer Extraction** | Base conversational outputs fail rigid `_MC` regex, artificially depressing base scores. | Re-evaluate base with flexible regex. Measure residual gap $\Delta	ext{acc}_{	ext{norm}}$. | 0 GPU-h, $0.00 (CPU) | If $\Delta	ext{acc}_{	ext{norm}} \le 1.0$ pp, declare gain formatting artefact. |
| **Audit Harmless Guard Scorer** | Base harmless refusal of 0.917 reveals severe false-positive saturation on OR-Bench. | Calibrated audit of 50 base harmless generations; replace 80-char keyword check. | 0 GPU-h, $0.00 (CPU) | If false-positive rate $> 20\%$, invalidate guard gate. |
| **MF-BoN Co-Primary in H-E2** | Beating StatNN is a strawman; world model must beat direct search at equal compute. | WM-CP vs MF-BoN on combinatorial library ($M=256$) at equal compute. | 8.0 GPU-h, ~$40.00 | If WM $\le$ MF-BoN, report negative result for amortized WM. |
| **Preregister Learning Curves for H-T** | 1D training trajectories are trivial curve-fitting unless they beat standard learning curves. | WM beats 2-parameter power law by $\ge 25\%$ lower MATE on held-out steps ($t > 52$). | 0.5 GPU-h, ~$2.50 | Demote H-T permanently to descriptive appendix. |
| **Drop Stage S Confirmatory Claims** | Curvature transport failed in p1/p2, stalled in p3, requires eager attention, burns cap. | S2 achieves $\ge 50\%$ oracle cut on $\ge 3$ non-refusal objectives. | 6.0 H100 GPU-h, ~$30.00 (capped) | If $< 50\%$ or sycophancy cut $\le 0$, drop S entirely. |
| **Focus Confirmatory on A8 Rank 3** | {E1, E2} is powered (0.884, 0.828) at 65.7 GPU-h-eq ($354), safely within $1,000 cap. | Board A signoff on frozen PREREG v2 manifest. | 65.7 GPU-h-eq, ~$354.00 | Fallback to rank 4 {E1} (17.5 GPU-h-eq, $95). |
