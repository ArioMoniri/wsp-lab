# Chair ruling, 18 Nov 2026: consequences of pilot 3 (applies SYNTHESIS-r10 D2/D4 and SYNTHESIS-r11 R1/R2/R8)

**Input.** `docs/reviews/wme/wme-pilot3-2026-11-17.md` (EXPLORATORY), merged at e9646bd. Every consequence below was
**pre-declared** in R8 or D4. This ruling only applies those rules; it loosens nothing.

## P1. Slope transport is dropped permanently

- R8 says: "S partial or fail → transport dropped permanently; no further S spending."
- The rerun stopped at its cap with 43 of 54 steps measured, which makes it **partial**.
- S2 passed on GSM8K gold log-prob and on the harmless-refusal margin, but those passes cover only the measured
  steps. They are reported as **descriptive**. That curvature along the edit can be computed at R1 (forward-mode
  jets at θ₀) is a documented engineering result.
- S1 disagrees with S2 by 54–109 % on the 9B refusal objectives, against 1e-6 on the toy model. It is recorded as
  an open numerical defect. It will not be investigated unless a later claim needs S1.
- The bank therefore carries no per-state oracle gradients.

## P2. No accuracy-enhancement claim from H1–H4 (D4)

| Family | Result |
|---|---|
| H1 | Fails the D4 over-budget rule |
| H2 | Fails |
| H3 | Fails: it does not beat the format-only control. Its 13 unevaluated candidates count as non-qualifying, by the plan |
| H4 | Not measured |

- Under D4, if H1–H4 all fail, there is **no accuracy-enhancement claim**, and headroom is reported as the limiting
  finding.
- At 4,096 tokens, H1 versus the base is −0.9 pp on MMLU-Pro and +4.1 pp on SuperGPQA. On items untruncated in
  both it is −3.3 and −2.7 pp. Under R2, the paper **may not** say "improves accuracy".
- The parser re-parse is identical to the strict parse, so the result is not a parser artefact.
- Caveat on the D4 over-budget rule: its operational form was written after interim H1 values were visible. The
  report gives both the gate-only count and the rule-applied count. This ruling applies the stricter one. The
  4,096-token check independently supports it.

## P3. Claim set, by the R1 ordering with H-E1g removed

| Claim | Status |
|---|---|
| **H-E1**, sycophancy (TriviaQA generation, N = 1,000) | The registered enhancement claim. Already powered in v2b (0.884) |
| **H-E2**, WM-guided vs StatNN-guided search, on the sycophancy objective | Primary contrast. Already powered (0.828) |
| H-E2, WM vs direct multi-fidelity best-of-N at total-compute parity | Co-primary only if stage W powers it, otherwise a registered secondary |
| **H-T2**, trajectory: accuracy and truncation along held-out edit paths | Defined in R4. It does not need an accuracy gain: H1/H3-type paths move accuracy and truncation by large, measured amounts. Registered only if stage W shows power ≥ 0.8 with ≤ 40 paths |
| Token efficiency | Registered **secondary**, explicitly pilot-informed and descriptive in wording: accuracy at the frozen 2,048-token budget for one H1 recipe retrained with fresh seeds, with the 4,096-token and untruncated-in-both analyses reported beside it. Never worded as "improves accuracy". Read cost ≈ 0.45 GPU-h |
| Retired | The D3 proxy (killed: Spearman −0.56 and −0.58); H-E1g; the old H-T (descriptive on the bank) |

## P4. Guards

- The blind audit agreed with the refusal rule on 0.986 of the 143 items the two judges agreed on (143/150), so the
  guards keep gating.
- The base genuinely refuses about 0.9 of OR-Bench-hard prompts. That is a property of the model, and it is
  reported.

## P5. Next steps

1. **Stage W** (≤ 4 GPU-h plus timing pilots ≤ $11, Lambda H100 SXM5, instances named `wsp-wme-w-*`):
   - fresh H1/H3-type training paths with fresh seeds, across checkpoints × λ × layer-group masks;
   - measure accuracy and truncation on fresh dev panels;
   - fit the R5 budget-aware state against the R4 baselines (persistence, power-law and parametric learning curves,
     LC-PFN, StatNN, static map) and the three nulls;
   - simulate power for H-T2, and for the E2 best-of-N co-primary on the sycophancy library.

   Before any stage-W data exists:
   - register `wme-p3s` and `wme-p3h` in `wm.exposure.EXPOSURES` (R7);
   - commit a pre-declared stage-W plan.
2. **PREREG v2**: the author fills in A1–A11, R1–R8 and P1–P4.
3. **Board A** re-reviews, then the freeze.
4. **9B confirmatory runs**, then **27B**, as in R9.

**Spend to date (Lambda):** track A $20.12, pilot 3 $110.13, 27B calibration $5.02. No `wsp-wme-*` instance is live.
