# Round 13, world-model architect seat (Q1, Q3, Q5), 20 Nov 2026

## BLUF

1. P1 shows LLM readers beating a text-only reader at the ternary *direction* of edit effects in one pair (27B self-read: 0.730 vs 0.578). It has no magnitudes, trajectories, calibration or statistical baseline.
2. P2 is agentic forward-pass editing with no predictor.
3. The premise "trajectory prediction with world models is possible" is **partly supported, by our own W2 only**: trajectories are predictable *from path type*. Neither paper shows *edit-specific* prediction beating a path-type baseline.
4. Pooled E is already 0.771 (k = 0) and 0.881 (k = 2). A perfect predictor under Gaussian noise scores √(2/π) ≈ 0.80. Little residual is left for any architecture.
5. **Reject a Mamba/SSM memory WM for (a).** There are 14 training runs per fold, 6 points per path and composition depth h ≤ 2. LC-PFN, an in-context sequence learner, lost (2.64 vs 0.881). WM-CP is history-free by construction, enforced by C2-D.
6. **Endorse (CPU, $0):** measure the noise floor; build per-layer and per-item R1 from the stored `j1`; make pooled + learning rate (LR) the mandatory baseline.
7. **Reject** an LLM reader as the WM, and P1 outputs as features, for (a).
8. For (b): H-E2 stays frozen; it already has a WSR-style arm. Study M gets a descriptive WSR-style reader scored on norm-cancelled edits, P1's Muse failure mode.

## 1. What P1 and P2 show (Q1)

**P1** (https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models, fetched)

- **Task.** Sandboxed agents read the weights without forward passes and predict up/down/same for five behaviours. A label counts as changed only when Welch p < 0.01 and a minimum effect is met.
- **Baseline.** A text-only reader. The page describes no majority-class or statistical baseline.
- **27B self-read.** Balanced accuracy 0.730 vs 0.578, +0.152 [+0.056, +0.256]. Recall rose 0.278 → 0.667 and specificity fell 0.878 → 0.793.
- **Other pairs** (text only → with weights): 9B→9B 0.492 → 0.554; 27B→9B 0.648 → 0.628.
- **Sample.** 18 changed labels; each cell ran once.
- **Muse.** No edit (0 of 40) changed behaviour, yet readers predicted a change 53–88 % of the time.

**Repo** (https://github.com/suayptalha/wsr, fetched): an evaluation harness with fixed same/change/random baselines. There is no trained predictor and no paper results.

**P2** (https://arxiv.org/html/2611.09382, fetched): GPT-6 Astra with 50 forward requests per search.

Held-out: 0.8B MATH-500 8.35 → 37.80 and IFEval 21.40 → 58.12; 9B MATH-500 97.70 → 97.35 (dev +29.69). 6 of 16 trajectories kept the original weights. There is no learned predictor and no random-search baseline across trajectories.

**Against our (a).** Our (a) is calibrated multi-point regression, scored as |error|/SE, against the pooled same-path-type curve (`docs/reviews/wme/wme-stagew2-2026-11-18.md`). In our bank the direction of effects is largely fixed by path type (more steps, more dose, more layers). A path-type-majority rule would therefore do well on P1's metric while carrying no edit-specific information.

**Verdict.** The premise holds at path-type level: every generic model lost to the pooled curve by d ≥ 1.7 at k = 2. It is unsupported at edit level.

## 2. The Mamba/SSM memory layer (Q3)

The memory could range over three things. None helps here.

- **Edit histories.**
  - WM-CP predicts from R(E_T) only, and C2-D requires every valid decomposition to give the same prediction (`src/wsp/wm/wmcp.py`; PREREG_WME §6.3.2).
  - A history-sensitive memory therefore breaks a frozen control. If it is history-invariant, it adds nothing.
  - The bank's histories are h ≤ 2.
- **Points within a path.** There are 6 points and k ∈ {0, 2} (`stagew2.GRID`, `KS`). Context length is not the bottleneck.
- **Many trajectories in context.** This is LC-PFN's design: a transformer pretrained on 10 M synthetic curves (https://neurips.cc/virtual/2023/poster/69992, fetched).
  - It scored E 2.638.
  - Mamba roughly matches transformers at in-context learning (https://arxiv.org/html/2402.03170v2, fetched), so an SSM changes efficiency, not the prior.
  - LC-PFN lost because its prior did not match our curves, not because it lacked capacity.

**Published evidence.**

- I found no paper on SSM- or Mamba-based learning-curve extrapolation, nor on weight-space trajectory world models. The search returned nothing relevant.
- CTX-LC-PFN (https://icml.cc/virtual/2026/71467, fetched) credits gains to *context conditioning*, but its page gives no numbers or ablation.
- Unterthiner et al. (https://arxiv.org/abs/2002.11448, fetched) reach R² > 0.98 from weight statistics, but over 120k CNNs. That data regime is about 10⁴ times ours.

**Capacity is not the binding constraint.**

- **Shrinkage.** At k = 2, shrinkage chose s = 0 in every fold, so the model *is* the pooled curve.
- **Effective variation.** The runs vary only shard (3 levels) and LR (5 levels) (`stagew2.RUNS`). That leaves about two effective edit dimensions per type, learned from 14 runs.
- **Noise floor.** Pooled E is about 0.8, near the √(2/π) reference.
  - Caveat: `se_of` is a binomial SE on a fixed panel, so the real repeat-noise floor may be lower. D2 measures it.

**Endorsement conditions** (all must hold; the 9B bank meets none): (a) within-type residual variance exceeds the measured noise floor; (b) ridge on richer features first beats pooled + LR; (c) hundreds of trajectories per type with long, non-commuting histories (h ≫ 2); (d) pretraining on a bank-matched simulated prior (`wm.synthetic`).

**Spec, if they ever hold:** input the sequence of (ΔR(E_k), observed points) plus run descriptors; output a per-endpoint mean and variance; baseline pooled + LR; metric W2's E with run-cluster sign-flip.

## 3. Alternatives (Q3)

- **P1-style reader outputs as WM features: reject for (a).**
  - They add at most one bit per behaviour, which path type already largely determines.
  - A direction call cannot resolve residuals smaller than one SE.
- **An LLM reader as the WM: reject.**
  - It is uncalibrated: P1 shows 53–88 % false-change calls on inert edits.
  - It gives no variance, so it cannot be scored on E.
- **Richer R1 features: endorse the CPU half.**
  - `data/wme/stagew2/run-20261118/{a,b,c}/w2/features.npz` stores `j1` as [3 objectives, 64 items, 30 checkpoints, 32 layers].
  - W2 used only its layer-summed, item-mean value (`stagew.member_features`).
  - Per-layer gradient alignment and per-item dispersion are therefore free. Fit them by ridge or PLS inside the identical leave-one-run-out pipeline, with nulls N1–N3.
  - Fisher-weighted norms need a new θ₀ pass, so defer them.
- **A different target: endorse as descriptive analyses.**
  - The within-type rank of endpoint residuals (Spearman ρ vs a random-rank null).
  - The residual sign.
- **Mandatory baseline: pooled + LR.** LR is a recipe scalar. Beating the pooled curve alone with LR-correlated features is not weight-space information.

## 4. P1/P2 for (b) and study M (Q5)

- **H-E2.**
  - It already has a zero-shot WSR-style reader and a reader team (PREREG_WME §6.8.3; 140,000-token cap).
  - It is frozen, and P1's single positive pair does not argue for change. P1's 27B→9B null is a weak prior against the reader arm on 9B.
- **Study M.**
  - Gemma-3 post-norms cancel uniform writer rescales (`docs/council/2026-11-19/SYNTHESIS-r12.md`). This is P1's Muse mechanism, where readers were badly over-confident.
  - Add a **descriptive** WSR-style reader that scores two things:
    - its false-change rate on norm-cancelled and inert edits;
    - its direction balanced accuracy against a family-majority baseline.
  - It reuses Gate 1 labels and runs only after Gate 0b and Gate 1 pass (`RULING-wmm-gate0.md`). It carries no α.
- **P2.**
  - The 0.8B gains start from very low bases.
  - On 9B MATH, a dev gain of +29.69 became −0.35 held-out, at a 97.70 base. That matches our headroom finding: adaptive dev selection overfits when headroom is small.
  - It supports sealed tests, not an agentic editor on 9B.
  - P2's editing is R2 in our tiers: at most a charged oracle arm in PREREG_WMM, never a WM feature.

**Fetch log.** All seven cited URLs were fetched; P2 was read in its HTML version. There were no failures. The DPL and hyper-representations papers were not fetched and are not cited.

## Decisions

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| D1. Premise "partly supported" | P1 is direction vs text; P2 has no predictor; W2 is path-type only | none | 0, $0 | none |
| D2. Measure the repeat-noise floor of E | Pooled E is near √(2/π) | Run E for an oracle-mean predictor on the `base` / `base_rerun` records in `evals.jsonl` | 0, $0 | If the floor ≥ pooled E − 0.05, close (a) on 9B |
| D3. Pooled + LR baseline | LR is not weight-space information | Same leave-one-run-out pipeline and nulls N1–N3 | 0, $0 | Wins are reported as LR-explained |
| D4. Per-layer and per-item R1 from `j1` | The only unexploited signal on disk | Ridge/PLS; d vs D3; Holm LB at k = 0, 2 | 0, $0 | d ≤ 0 → stop |
| D5. Fisher-weighted norms | Needs a θ₀ pass | Only if D4's LB > 0 | Priced after D4, local, $0 | Not run |
| D6. Mamba/SSM WM | Blocked by C2-D; 14 runs; LC-PFN lost on its prior; no published evidence found | §2 conditions (a)–(d) | 0, $0 | Rejected |
| D7. LLM reader as the WM, or P1 outputs as features | Ternary and uncalibrated | none | 0, $0 | Rejected |
| D8. Rank and sign targets | Re-scoped, needs no calibration | Spearman ρ, run-cluster sign-flip | 0, $0 | CI includes 0 → drop |
| D9. Descriptive WSR reader in study M | Tests the norm-cancellation failure | After Gate 1; specificity on inert edits | 0 GPU-h; tokens priced before the run | Skip if Gates 0b/1 fail |
| D10. H-E2 unchanged; no P2 agent on 9B | Frozen; −0.35 held-out | none | 0, $0 | none |
