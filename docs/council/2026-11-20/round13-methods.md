# Round 13, methods / diagnosis seat: the W2 tie and a powerable (a)

> EXPLORATORY, post hoc. Uses only the exposed W2/W data, on CPU, with nothing written under `data/`. Scripts and commands are in §5.

## BLUF

1. **It is not a capacity problem.** W2 does have edit-specific signal, but it is one-dimensional: **dose**, i.e. learning rate (corr(log ‖ΔW‖, log lr) = 0.996). The nested WM's shared ridge misses it.
2. **A post-hoc weights-only dose model beats the pooled curve on W2.** The model is pooled + log R0 norm, fitted per (type, point, endpoint).
   - k = 0: E 0.678 vs 0.771, run sign-flip p 0.021.
   - k = 2: E 0.766 vs 0.881, p 0.003.
3. **Variance decomposition** (SE² units, mean over cells): between type 1.85; within type 0.93; item-level noise 0.34.
   - Accuracy signal is 0.12–0.15 SE² (reliability 0.29–0.30); truncation signal is 0.83–1.25.
4. **Ceiling.** A noise-free oracle reaches E ≈ 0.505 (max C ≈ 0.35). An oracle with SE-level noise reaches only 0.690 (0.946 unshrunk).
5. **The bank cannot identify anything beyond dose.** Shard adds nothing (0.709 vs 0.682). The dose model loses on stage W (non-evidential).
6. **I reject Mamba/SSM for (a).** There is no history to exploit, the signal is a run-level scalar, and k = 2 already hurts.
7. **Re-scoped (a): beyond-dose ranking.** Baseline: pooled + dose. Edits are ranked within type at matched norm. It needs ≥ 30 direction units, truncation first. Direction-only scoring has no headroom: the majority rule already scores 0.887.
8. **Where to run it.** In study M after Gate 0b/1 (local, $0). New 9B runs only if the user explicitly overrides Q4 ("No further spending on (a) for the 9B model").

## 1. Diagnosis (Q2)

### Data and variance decomposition

Data: `data/wme/stagew2/run-20261118/{a,b,c}/w2`, loaded via `wm.stagew2.load_runs`/`arrays`: 45 paths × 6 points × 4 endpoints. SE is the binomial SE at n = 500.

I estimated noise as the paired (McNemar) variance between same-type, same-point members, halved. Values are in SE² units.

| endpoint | within-type var | noise σₑ² | signal σₐ² | reliability |
|---|---|---|---|---|
| acc MMLU-Pro | 0.389 | 0.271 | 0.117 | 0.29 |
| acc SuperGPQA | 0.484 | 0.334 | 0.151 | 0.30 |
| trunc MMLU-Pro | 1.240 | 0.413 | 0.827 | 0.65 |
| trunc SuperGPQA | 1.591 | 0.343 | 1.248 | 0.77 |

- **Between type.** Mean 1.85 SE², up to 10.0 for truncation at j = 6. This is why every generic curve model loses to path type.
- **Run-to-run numerics.** The base re-run alone gives 0.13–0.29 SE², so about half of the "noise" is vLLM non-invariance.
- **The signal sits at run level.** Residuals correlate across a run's S/L/M paths (0.34–0.88) and along a path (lag 1: 0.41).

### Ceiling

These are Gaussian-model figures; the model is checked by predicted pooled E 0.757 vs observed 0.771. Values are k = 0 E.

| predictor | E |
|---|---|
| pooled curve (measured) | 0.771 |
| oracle, true effect (τ² = 0) | 0.505 |
| oracle, τ² = 0.5 SE², best s 0.50 | 0.649 |
| oracle, τ² = 1 SE², best s 0.33 | 0.690 |
| oracle, τ² = 1 SE², unshrunk | 0.946 |

**Per-endpoint floor vs pooled:**
- accuracy: 0.44 / 0.48 vs 0.51 / 0.57;
- truncation: 0.57 / 0.54 vs 0.91 / 1.03.

**Even a perfect oracle gains only about 0.08 E on accuracy.** Any edit-specific win must come from truncation.

### What the nested WM did

Out-of-fold g at s = 1: corr(g, residual) is 0.02 / 0.18 on accuracy and 0.40 / 0.34 on truncation. sd(g)/sd(residual) is 0.32–0.46.

- **k = 0.** Forced s gives E 0.771 (s 0), 0.749 (s 0.5) and 0.758 (s 1). The inner CV picked a mean s of 0.45 (`wme-stagew2-2026-11-18.json`).
- **k = 2.** s = 0 in all 15 folds. The edit component is **shrunk exactly to zero**, hence d = 0.000. The k-point offset injects noise from two early points while the signal sits late; pooled E rises from 0.771 to 0.881.

### Specification tests

All rows are LORO, compared by exact sign-flip over 15 runs.

| residual model | E k = 0 | E k = 2 |
|---|---|---|
| log R0 norm, one slope per endpoint | 0.700 | 0.868 |
| log R0 norm, slope per type | 0.691 | 0.861 |
| log R0 norm, per (type, point, endpoint) | **0.678** (p 0.021) | **0.766** (p 0.003) |
| log lr per cell (design) | 0.682 (p 0.024) | 0.765 (p 0.003) |
| log lr + shard per cell | 0.709 | — |
| J1 features alone per cell | 0.763 (p 0.44) | 0.907 |
| pooled | 0.771 | 0.881 |

The k = 0 gain falls mostly on S paths (d 0.199 S / 0.08 L / −0.001 M).

**Diagnosis.** The tie comes from **specification**: no log transform, no type × point interaction, and dilution by J1. On top of that, **data identifiability** limits the bank to dose as its only within-type axis. Model capacity is not the cause.

**Caveats.**
- I tried six specifications after seeing the data.
- On stage W, thinned to 4 runs with lr ∈ {1e-4, 2e-4}, the dose model *loses*: 0.817 vs 0.639 at k = 0 and 1.126 vs 0.893 at k = 2. It is overfitting a 2-parameter fit to 3 runs.
- This is a hypothesis.

## 2. Mamba/SSM (Q3)

A memory layer adds history, cross-path transfer and in-context learning. None of these is the bottleneck here:

- **History.** Paths are only 6 points long, and k = 2 already loses to k = 0.
- **Cross-path transfer.** The pooled curve already captures it.
- **In-context learning.** LC-PFN, the in-context learner, lost (E 2.64 at k = 2).
- **The residual itself.** It is a run-level scalar, with accuracy reliability of 0.29.

A memory model can only beat pooled + dose if both of these hold:
- the bank has **≥ 20–30 directions** whose effects differ at matched norm;
- the inputs carry **direction information**: R1/R2 projections, P1-style reader calls, or training-data identity.

A per-cell ridge on those features should be tried first; an SSM is justified only if it beats that ridge.

**Verdict: reject for now.**

## 3. Re-scoped (a) (Q4)

### A1: dose confirmation

**Claim.** Pooled + per-cell log ‖ΔW‖ beats the pooled curve.

**Test.**
- Metric and rule: the W2 E metric and rule (Holm over k, run-cluster sign-flip).
- Nulls: N1–N3, plus **N4** (norm permuted within type).

**Power.** From a run bootstrap of the W2 run-level d, using MC sign-flip at α 0.0125:

| runs | observed effect, k = 0 | observed effect, k = 2 | half effect, k = 2 |
|---|---|---|---|
| 15 | 0.41 | 0.84 | 0.07 |
| 20 | 0.62 | 0.96 | — |
| 30 | 0.88 | ≈ 1.00 | 0.27 |
| 45 | — | — | 0.53 |

**Value.** Scientifically low ("bigger LoRAs move more"). Its main use is as **the new mandatory baseline**.

### A2: beyond-dose within-type ranking (the real (a))

**Claim.** Direction features predict each run's residual after removing the dose baseline, at matched log-norm. Candidate features: J1/R2, a P1-style reader, data identity.

**Unit: an edit run aggregated over its members.** Per-point within-type Spearman of the dose predictor is only 0.037 (accuracy) and 0.235 (truncation). Run-level Spearman is 0.44 / 0.73 (accuracy) and 0.80 / 0.61 (truncation).

**Test.**
- Statistic: Spearman, one-sided α 0.025.
- Baseline: pooled + dose.

**Nulls.**
- direction features permuted within dose stratum;
- sham directions (random orthogonal deltas at matched norm);
- N3.

**Power.** Fisher z, 80 %:

| true ρ | runs needed |
|---|---|
| 0.4 | 47 |
| 0.5 | 29 |
| 0.6 | 19 |
| 0.7 | 13 |

Attenuation by run-level reliability is unestimable without replicate runs. **Plan for ≥ 30 direction units, truncation first.**

### A3: P1-style direction

Not powerable here.

- **Label mix.** Ternary labels (|Δ| ≥ 1.96 paired SE vs base) split 31 % up, 28 % same, 41 % down.
- **Baseline already strong.** The type × point majority scores 0.887 in sample (accuracy 0.83 / 0.82). 42 of 72 cells are mixed, mostly near the threshold.
- **Use.** Keep it as a secondary inside A2.

### Where

- **9B exposed data.** It can freeze the specifications, as done here, but cannot test A2: shard adds nothing.
- **Study M.**
  - **Why it fits.** Its families (task vector, interpolation, two LoRAs with non-monotone dose; `RULING-wmm-gate0.md`) are a natural matched-norm, different-direction bed.
  - **Scale needed.** Four families give n = 4, so it needs about 30 LoRAs on disjoint shards or objectives at 2–3 matched norms.
  - **Conditions.** Only after Gate 0b/1 pass, under M4 (local, $0).
  - **Cost.** Local GPU-h are **unmeasured**; price them from Gate 1's per-member timing.
- **New 9B runs.** W2 cost 8.58 instance-h and $41.95 for 15 runs. Thirty runs scale to about 17 h and $84. This conflicts with Q4.

## 4. Decision table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Record the W2 tie as specification + identifiability, not capacity | Signal is 1-D dose; g shrunk to 0 at k = 2; dose model 0.678/0.766 vs 0.771/0.881 | Second seat reruns §5 scripts | 0, $0 | Withdraw §1 if not reproduced |
| Pooled + per-cell log-norm dose is the mandatory baseline for future (a) | It beats path type on W2 | Frozen now, before new data | 0, $0 | Revert to pooled if A1 fails fresh |
| Reject Mamba/SSM WM for (a) now | No history; scalar run-level signal; LC-PFN lost; k = 2 hurts | Revisit only if A2 ridge gives ρ_LB > 0 | 0, $0 | none |
| No direction-only (a) on 9B | Majority baseline 0.887 | none | 0, $0 | Secondary inside A2 |
| A2 in study M, ≥ 30 direction units, truncation first | Only design with within-type direction spread | After Gate 0b/1; pre-declared plan; Spearman LB > 0, nulls ≤ 0.035 | Local, unmeasured, $0 | (a) stays negative; dose result descriptive |
| A1 fresh 9B, 30 runs | Confirms dose; low value | Only on explicit user override of Q4 | ≈ 17 h, ≈ $84 | Default: not run |

## 5. Commands

From the worktree:

```
. scripts/env.sh
python /tmp/r13-methods/load.py
python /tmp/r13-methods/decomp.py
python /tmp/r13-methods/oracle.py
python /tmp/r13-methods/targets.py
python /tmp/r13-methods/power.py
python /tmp/r13-methods/stagew_check.py
```

- **Inputs** (read only): `data/wme/stagew2/run-20261118/{a,b,c}/w2` and `data/wme/stagew/run-20261118`.
- **Seed:** 0.
