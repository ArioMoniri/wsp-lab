# Round 11, world-model seat (17 Nov 2026)

EXPLORATORY. **[seat calc]** marks my own read-only arithmetic on the dev file `data/wme/pilot3/run-20261117/h/evals.jsonl` (`wme-p3h`, selection-exposed, not a gate). Cost in $ uses the pilot-3 ruling's "4.5 GPU-h ≈ $22", i.e. ≈ $4.9 per H100-h.

## BLUF

1. **H1's dev gain is budget relief.** Δacc tracks the fall in truncation (Spearman 0.94 on MMLU-Pro and on SuperGPQA, 54 candidates). On items neither model truncated, H1 at step 208, λ 1 scores *below* the base: 0.829 vs 0.853 and 0.545 vs 0.566 **[seat calc]**.
2. **The registered R0 proxy is likely dead.** Δ log-prob of the base's own verified answers is anti-correlated with Δacc: ρ −0.81 and −0.88 **[seat calc]**. The final D3 bootstrap decides.
3. **New WM state:** acc_B = P(complete ≤ B) · P(correct | complete), per item.
4. **Tiers.** A teacher-forced forward of the candidate is R2 (r8 D2) and is only a charged low fidelity. The R1 equivalent is jets along ΔW at θ₀.
5. **Q2.** WM = prior of a shared MF-BO over a combinatorial H-library, 8 full-dev evaluations per arm; primary WM vs StatNN; must also beat direct MF best-of-N and "last checkpoint, λ 1".
6. **Q3: H-T2.** On fresh training×λ×mask paths: nested WM (early points + edit features) vs per-path learning-curve extrapolation; zero-shot WM vs the static map.
7. **Flag 1.** Symmetric shrinkage for every arm. H-T (WM-CP vs static) stays descriptive.
8. **Claims.** {H-T2, H-E1acc, H-E2} first. Sycophancy H-E1 becomes a Holm secondary.
9. **Budget.** Stage W (≤ 4 GPU-h) gates the new state; new confirmatory work ≈ 45 GPU-h, inside caps.
10. **Every pending H3, H4 and S outcome has a fixed consequence** (§5).

## 1. Pilot-3 facts the world model must respect

- **Costs** (`.../h/ledger.jsonl`): full dev evaluation 10,791 s / 44 = 245 s; one 208-step LoRA run ≈ 650 s. MC budget B = 2,048 (`budgets.json`); base truncation 0.20 / 0.38 (interim).
- **Targets** (`sft_stats.json`): mean length 525 (H1), 919 (H2), 686 (length-matched control, which does not cut truncation, ≈ 0.18).
- **Reading.** H1 teaches brevity from thinking mode; pilot-3 §6 read its fail rule narrowly. **Ruling needed:** the gain counts only as **accuracy at a frozen non-thinking budget**, reported with untruncated-item accuracy and a 2×B sensitivity, never as a capability gain.
- **Noise.** Discordance 0.129 / 0.191 (`sizing.json`) gives paired SE ≈ 1.1–1.4 pp at N = 1,000. The H1 λ 1 MMLU-Pro curve stays in 0.711–0.744 after step 13, mostly noise; truncation is the high-signal coordinate.
- `guard_harmless` drifts 0.917 → 0.854 on late λ 1 checkpoints (sign per `refused` semantics). `STATUS` says FAILED (12:26) but evals ran to 19:34; looks stale, untouched.

## 2. Q2: where the world model adds value to enhancement

### State and action

Member m = (run r [seed, data subset, lr], checkpoint t, λ ∈ {0.25…1.5}, layer-group mask G ⊆ 4 groups, convex merge w of two checkpoints/runs, H4 flag, SVF α if H3 passes). ΔW(m) is exact on CPU; outcomes depend on ΔW only (r8 D3). Hundreds to thousands of members: where prediction pays.

### Features

- **R0** (θ₀ artefacts plus CPU arithmetic on D): per-group ‖ΔW‖; cosine with refusal direction r̂; overlap with each base writer's top singular subspace; ⟨g₀, ΔW⟩ and K-FAC q for base gradients of objectives (a)–(c); the run's free training-loss curve.
- **R1** (jets at θ₀ along D, `wm/jets.py`). Per holdout item they give the dose curve f(λ) ≈ λJ1 + ½λ²J2 for three objectives:
  - (a) the `<|im_end|>` hazard over the base's own response positions, which yields P(complete ≤ B);
  - (b) the log-prob of the thinking-distilled concise target;
  - (c) the answer-letter margin (correct letter against the best wrong one).
- **Low fidelity (R2, charged pro rata, D6):** teacher-forced (a)–(c) under the candidate. One prefill pass replaces up to 2,048 decode steps.

### Model

- Per-item p̂_complete and p̂_correct|complete feed a monotone link recalibrated on real evaluations bought so far; an ensemble gives uncertainty; exactly zero at ΔW = 0, a function of ΔW only.
- **Lineage:** two-step "intermediate → accuracy" ladders (https://arxiv.org/abs/2412.04403). **StatNN:** an MLP on the same features predicting Δacc, no structure.

### Planning

- **Search.** A shared D6 P3 MF-BO with the ladder teacher-forced → 32 items → 300 → full dev. B = 8 full-dev equivalents per arm per episode (≈ 0.54 GPU-h).
- **Primary contrast.** WM vs StatNN (A3 sign-flip).
- **Descriptive arms:** direct MF best-of-N (successive halving, no prior); "last checkpoint, λ 1"; "best training loss"; the readers (D7).
- **Endpoint.** Sealed-test accuracy of each frozen winner, as regret against the library best estimated on a separate selection panel.
- **Secondary.** WM at B = 8 is no worse than direct search at B = 32.

## 3. Q3: H-T2, a physically valid and powerable trajectory claim

### Paths and claim

A path is an ordered family of realised edits, θ₀ + λΔW_r(t). It is ordered by training step t, by λ at fixed t, or by mask growth. Outcomes are functions of θ, so A7 holds by construction.

> **Claim.** Given k ∈ {0, 2, 4} real points of a held-out fresh path, plus R0/R1 features at every point, the WM predicts accuracy and truncation at the remaining points better than matched baselines with the same real points.

### Baselines

| Arm | Description |
|---|---|
| B1 | Per-path learning-curve extrapolation: a parametric ensemble (Domhan-style), plus LC-PFN, which reports better posterior accuracy than MCMC at over 10,000× the speed (https://arxiv.org/abs/2310.20447) |
| B2 | The static map: StatNN on R(E_T), trained on other paths. This is the k = 0 comparator. |
| B3 | The pooled cross-path curve offset by the early points (freeze-thaw kernel, https://arxiv.org/abs/1406.3896) |
| B4 | The last value carried forward. It is the cheapest and strongest under a plateau. |

### Test

- **Contrasts:** WM vs the best of {B1, B3, B4} at k = 2 and 4; WM vs B2 at k = 0.
- **Statistic:** the A2 sign-flip test over paths on SE-scaled |error| differences, with Holm correction over k.

### Why this is a world model and not a curve fit

- B1, B3 and B4 see only outcomes; the WM also sees edit content, must predict members no curve reaches (unseen λ, masks, merges at k = 0), and is nested over B1, so its margin is exactly the features' value.
- **Prior art.** TuneAhead (https://arxiv.org/abs/2606.17660) is the closest. It uses static descriptors plus a 100-step probe, reports an RMSE of 1.47 pp on 370 Qwen2.5-7B-Instruct runs and beats early-stop extrapolation. It is cited as the probe-feature competitor.

### Synthetic tests (CPU, pre-freeze)

- **Should-help.** The late plateau or downturn is set by a feature (J2 or the termination hazard) that the early points do not reveal. The WM must win.
- **Must-not-beat nulls (size ≤ 0.035, coverage ≥ 0.965):**
  - N1: the plateau is fixed by the early curve;
  - N2: features are permuted across paths;
  - N3: the static map is exact (the analogue of `null_quadratic`).
- Noise from `sizing.json`; between-path spread pilot-informed.

### Power and cost

- **Simulation grid.** Simulate P ∈ {16, 24, 32, 40} paths at 8 points and N = 500.
- **Cost per path:** ≈ 0.18 GPU-h of training plus 8 × 0.034 GPU-h of evaluation, i.e. ≈ 0.45 GPU-h. For P = 24 that is ≈ 11 GPU-h plus ≤ 4 GPU-h of jets.
- **Fallback.** If power stays below 0.80 for every P ≤ 40, H-T2 is registered "unresolved — not powered" (A8).
- **S-stage curvature transport** is not a separate claim. If S passes for an objective, its jets become WM features (D2).

## 4. Other questions (brief)

- **Q1: claims.**
  - Add **H-E1acc**: generated accuracy at frozen B on MMLU-Pro and SuperGPQA, guarded, on a sealed test.
  - Keep sycophancy H-E1, powered at 0.884 (v2b review), as a Holm secondary.
  - A8 order: {H-T2, H-E1acc, H-E2} > {H-E1acc, H-E2} > {H-T2, H-E1acc} > {H-E1acc}.
- **Flag 1.** Static members select shrinkage by WM-CP's own inner-fold rule, with ties going to the exact nest (v2b review §7, option 2). Re-run `null_quadratic`.
- **Q4: fresh data.**
  - Register `wme-p3h` and `wme-p3s` in `wm.exposure.EXPOSURES`.
  - Draw new sealed MC panels with a new salt, sized N = ⌈7.84·d/0.03²⌉ = 1,123 (MMLU-Pro) and 1,668 (SuperGPQA). Check the pools after exclusion.
  - The pilot-3 best values enter only as a pilot-informed library declaration, never as effect sizes.
  - §1 is post hoc, so the new state is tested only on fresh runs (stage W).
- **Q5: order.** Pilot 3 (untouched) → stage W → CPU simulations (H-T2, E2, D6 regret) → PREREG v2 → Board A → freeze → 9B bank and confirmatory → 27B (H-E1acc only).

## 5. Pre-declared consequences

| Pending result | Consequence |
|---|---|
| D3 proxy killed / survives | Leaves the state, replaced by (a)–(c) if stage W passes / kept as a feature |
| H3 passes / fails | SVF α paths join library and H-T2 as a second family / H1-type library only |
| H4 guarded with gain / fails | Projection becomes a library action / dropped |
| S passes for k / fails | Jets allowed for k / transport dropped (D2); (a)–(c) jets still need stage W |
| H1 gain vanishes at 2×B on dev | H-E1acc is worded "at budget B" only. |
| Stage W fails | No WM-guided accuracy claim. E2 falls back to the declared sycophancy library (v2b rank 3). H-E1acc runs as direct-search enhancement with no WM credit. |

**Stage W** (exploratory):
- 4 fresh H1-type runs × 8 members on a fresh dev panel, at N = 500.
- The WM is fitted on construction runs only.
- **Gate:** the Spearman lower bound between WM-predicted and measured Δacc is > 0, *and* the WM beats StatNN in |error|.

**Other literature (fetched):** observational scaling laws (https://arxiv.org/abs/2405.10938); fine-tuning scaling law (https://arxiv.org/abs/2402.17193); NTK spectrum vs LoRA (https://arxiv.org/abs/2602.08239); AdaMerging (https://arxiv.org/abs/2310.02575); gradient-alignment mergeability (https://arxiv.org/abs/2601.22285); SimMerge (https://arxiv.org/abs/2601.09473). They support R0/R1 prediction of merge and λ members.

**Fetch failure.** The Domhan et al. IJCAI 2015 PDF (https://www.ijcai.org/Proceedings/15/Papers/487.pdf) was unparseable. It is cited by name only and is unverified.

## Decision table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| WM state = termination × conditional correctness | Δacc tracks truncation | Stage W on fresh runs | ≤ 4, ≈ $20 | No WM-guided accuracy claim |
| Teacher-forced passes charged as R2 fidelity | r8 D2 | `wm.plan` ledger audit | 0 | Uncharged use voids E2 |
| H-E2 on the H-library, MF-BO, B = 8 | Large library, small budget | A3 sign-flip on sealed regret | ≈ 30, ≈ $150 | Sycophancy E2 fallback |
| H-T2, nested vs LC extrapolation and static map | Physical; isolates edit-content value | A2 over paths; N1–N3 size ≤ 0.035 | ≈ 15, ≈ $75 | "Unresolved — not powered" |
| H-E1acc at frozen B, plus 2×B sensitivity | Budget relief is real but narrow | LCB > 0 on sealed 1,123 / 1,668, guarded | in bank | Restricted wording |
| Symmetric shrinkage (flag 1) | Size 0.41 is a selection artefact | Re-simulate size ≤ 0.035 | CPU | H-T descriptive only |
| Sycophancy H-E1 kept as secondary | Powered at 0.884 | As registered | as v2b | — |
| 27B: H-E1acc replication only | WM claims tested on 9B first | Same gates | within $1,000 | Skip if 9B fails |
