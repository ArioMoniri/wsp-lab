# Round 11, methods seat (17 Nov 2026)

Sources: the brief's files; read-only pilot-3 `sizing.json`, `guard_cal.json`, `ledger.jsonl`. No URLs used. All numbers are exploratory.

## BLUF

1. **Primary endpoint.** Move the primary to **H-E1g**: generated accuracy on fresh sealed MMLU-Pro and SuperGPQA, with Holm at 0.025. Sycophancy (H-E1s) becomes the D4 fallback.
2. **A8' ranking.** {H-Tp, H-E1g, H-E2g} > {H-Tp, H-E1g} > {H-E1g, H-E2g} > {H-E1g} > {H-E1s}. "(a)+(b)" may be claimed only if all three primaries pass.
3. **The dev best values are winner's curse.** On MMLU-Pro at λ 1, the SD across checkpoints is 0.0086, below the paired SE of 0.0114.
4. **Planning effect and N.** Plan at 0.5× the de-biased mean gain, with Wilson-upper discordance. That gives **N = 3,463 for MMLU-Pro and 1,784 for SuperGPQA**, about 0.45 GPU-h.
5. **The gain may be a budget artefact.** Base truncation is 0.20 / 0.38 at 2,048 tokens; H1 cuts it to 0.06–0.15. A 4,096-token secondary fixes the wording.
6. **The guard detector needs an audit.** The base is flagged "refused" on 0.89 of harmless holdout prompts.
7. **Flag 1.** Adopt symmetric shrinkage. H-T stays "unresolved — not powered".
8. **H-Tp.** Predict late checkpoint × λ accuracy on held-out fresh paths against persistence, curve-fit and StatNN; simulate first.
9. **H-E2g.** At a tight budget, test WM against StatNN, then against direct multi-fidelity best-of-N (fixed sequence).
10. **No post-hoc decisions.** Every pending outcome has a consequence declared in §5. Pilot-3 adapters and panels are exposures.

## 1. Claims (Q1)

- **Endpoint.** Goal (b) means model-card benchmarks (D3), and the WM must *choose* the edit; an edit family alone does not show (b).
- **Why sycophancy is not a co-primary.** H-E1s's powered design uses 65.7 of the 90 GPU-h (v2b §7) and cannot fit beside H-Tp and H-E2g. It still runs if no H family passes D4, or if the A8' winner leaves ≥ 17.5 GPU-h unused (v2b rank 4).
- **Error rate.** Each family is tested at one-sided 0.025 (PREREG §2.4).
- **Reporting.** §11.3's ban on "any claim succeeded" covers "(a)+(b)" (intersection–union).

## 2. Registered hypotheses

### H-E1g

- **Comparison.** The WM-designated winner (dev only, §6.5.6) against one pre-declared base run, paired by item, with the same stack and batch. Two base runs differ by up to 0.008; the second is reported, never chosen.
- **Hypotheses.** For k ∈ {MMLU-Pro, SuperGPQA}: H0k: Δacc_k ≤ 0 against H1k: Δacc_k > 0.
- **Test.** Tango paired LB; Holm at 0.025 (first step 0.0125).
- **Conjunctive gates.**
  - Harmless over-refusal UCB ≤ 0.039 and harmful-compliance UCB ≤ 0.023. These use the pilot-3 bootstrap-t with δ taken from `guard_cal.json`, applied to fresh items.
  - Anomaly increase ≤ 0.02.
- **E1g-B.** A fixed-sequence test of the same endpoint at a 4,096-token budget. If it fails, the claim is worded "at a 2,048-token non-thinking budget".
- **GSM8K, MATH, IFEval.** Descriptive only.

### H-E2g

Each arm spends an equal pro-rata ledger (D6) to pick one library member, scored by guarded sealed gain g̃.

- **Step 1.** H0: E[g̃_WM − g̃_StatNN] ≤ 0. The test is the §6.8.2 two-way bootstrap at 0.025.
- **Step 2, only if step 1 passes.** H0': E[g̃_WM − g̃_directMF] ≤ 0 at 0.025, against D7.
- **Step 3.** Holm over the remaining arms.

**Why the budget should be tight.** A screen of n items has SE √(d/n). With d = 0.129, that is 0.064 at 32 items and 0.011 at 1,000. A one-forward R0 proxy (D3) can only beat noisy screens.

### H-Tp

- **Hypotheses.** H0: C_Tp ≤ 0 against H1: C_Tp > 0.
- **Test.** Path-cluster bootstrap-t at 0.025, pooled over the two endpoints.
- **Secondaries.** One per endpoint, Holm-adjusted.

## 3. Q4: selection bias, power, fresh data

### Selection bias

Each series has 16 checkpoints. The base is the mean of the two runs. SE = √(d/1000).

| Series | Mean gain | SD | Max gain | SE |
|---|---|---|---|---|
| MMLU-Pro, λ 1 | 0.043 | 0.0086 | 0.061 | 0.0114 |
| SuperGPQA, λ 1 | 0.071 | 0.0208 | 0.097 | 0.0138 |

- **MMLU-Pro.** SD < SE, so the checkpoints cannot be told apart.
- **SuperGPQA.** There is a real trend with training step.
- A dev maximum is never quoted as an effect. Dev N was clamped at 1,000; the §4.6 rule asked for 1,123 and 1,667 (disclosed, not repaired).

### Power

**Formula** (paired binary, one-sided):

**N = ⌈(z_{1−α} + z_{1−β})² · (d − δ²) / δ²⌉**

where d = P(01) + P(10) is the discordance and δ the paired gain.

**Inputs.**
- **Discordance.** `sizing.json` gives d = 0.1289 (33/256) and 0.1914 (49/256). The one-sided 95 % Wilson upper bounds (0.167, 0.235) are used because the winner's d is unknown.
- **Effect.** δ = 0.5× the λ 1 mean gain (A10 precedent).
- **Error rates.** α = 0.0125 (z = 2.2414); power 0.8 (z = 0.8416).

| Endpoint | δ | d (point / upper) | N at power 0.8 | N at power 0.9 |
|---|---|---|---|---|
| MMLU-Pro | 0.0214 | 0.129 / 0.167 | 2,666 / **3,463** | 3,482 / 4,522 |
| SuperGPQA | 0.0353 | 0.191 / 0.235 | 1,451 / **1,784** | 1,895 / 2,329 |

**Register the values in bold.**

**Cost.** At 0.153 s per item (`ledger.jsonl`), winner plus base is about **0.45 GPU-h, about $2.40** at $5.38 per GPU-h (v2b §7). Power 0.9 is preferred if the pool allows it.

**Pool rule.** If the near-duplicate-excluded pool is < N (counted hash-only), the endpoint is "unresolved — not powered", never relaxed.

### Fresh data

- **New panels.** Sealed salt `wme-c2-gen` (primaries, guards); fresh E2g screen and dev panels.
- **Exclusions.** Near-duplicate groups of every split of `wme-p3h` and `wme-p3s` (including teacher and STaR outputs), plus all earlier exposures.
- **Exposures.** The chair adds `wme-p3h` and `wme-p3s` to `wm.exposure.EXPOSURES` first. Pilot-3 adapters are exposed, so the library is retrained from the frozen recipe on new seeds and fresh shards.

### Pilot-informed declaration

Fixed after seeing exposed dev data: the H1 recipe, the λ and step grids, the endpoints, B = 2,048, the guard δ, and the planning δ and d. Pilot-3 outcomes may train the WM and StatNN equally. They never test either.

## 4. Q3: trajectory and flag 1

### H-Tp

**Validity.** The target is accuracy at real states θ(path, step, λ), with no dependence on decomposition (A7).

**Unit.** One held-out path = family × seed × shard × layer-group subset.

**Training inputs:** other paths, the first 25 % of the held-out path's checkpoints (R2), and R0/R1 features at every checkpoint.

**Target.** Sealed accuracy at the later checkpoints, for λ ∈ {0.5, 1, 1.5}.

**Comparator** (best on inner folds, symmetric shrinkage):
- **B1:** persistence, linear in λ;
- **B2:** a + b(1 − e^{−s/τ}), with τ shared across paths;
- **B3:** path-free StatNN on R0 features.

**Statistic.** C_Tp = 1 − mean(|e_WM|/SE) / mean(|e_comp|/SE).

MMLU-Pro is declared weak: it plateaus inside the noise, so persistence is near-optimal.

**Power.**
- The design needs P ≥ (1.96 + 0.84)² σ_u²/μ_u² paths.
- Pilot 3's four paths give baseline errors, not μ_u, so simulate before the freeze (noise √(d/N), assumed C = 0.15, PREREG §8).
- If no P ≤ 16 is powered, H-Tp is "unresolved — not powered".

**Cost** (estimate, from the ledger): 0.16 GPU-h of training per path, 0.54 GPU-h of shared teacher data and about 0.13 GPU-h per sealed evaluation, so about 21 GPU-h at P = 12 with 12 late points.

### Flag 1: ruling for symmetric shrinkage

**Rule.**
- Every comparator predicts F̂ + λ_m(m − F̂), where F̂ = a + q is the raw field and λ_m ∈ {0, 0.25, 0.5, 0.75, 1}.
- λ_m is chosen on the same folds, with the same h = 2 loss, as WM-CP.
- A tie (within 0.5 %) goes to the smaller λ for **both** arms.

**Basis.** PREREG §6.1.2 already promises "the same shrinkage grid", so this corrects null size and is not tuning. The S1-first tie order is rejected: it is asymmetric and fixes only this one null.

**Gate.** In the full refit, `null_quadratic` and `null_shuffle` must each have size ≤ 0.035.

H-T stays unresolved (the alternative still favours static, C_T −0.33). The rule is mandatory in H-Tp and H-E2g too.

## 5. Q2 and Q5; consequences

**Where the WM enters.** As an R0/R1 ranker over an H1-type library (checkpoint × λ × layer-group subsets × merges). The library needs σ²_between = s²_obs − mean SE² > 0 (method of moments), plus harmful and inert members.

**Order.** Pilot 3 finishes; in parallel on CPU, the simulations, a blind hand-label guard audit and the pool counts; then PREREG v2 → Board A → freeze → library → confirmatory runs.

**27B.** Separate registration; only claims that passed on 9B; N from its own discordance.

**For the chair (not touched).** `h/STATUS` reads `FAILED` (mtime 12:26, before the 19:34 evals), so it is probably stale.

| Pending result | Outcome | Consequence |
|---|---|---|
| H1 gate | pass | recipe joins the library; H-E1g registered |
| H1 gate | fail | H-E1g only via H3; all fail → H-E1s (D4) |
| D3 proxy | LB > 0 on both endpoints | it is the WM's R0 accuracy input |
| D3 proxy | LB ≤ 0 on one | WM and E2g use the other only |
| D3 proxy | LB ≤ 0 on both | H-E2g unresolved; best set {H-Tp (R2 early points), H-E1g} |
| H3 | pass and beats both controls | a second family; stratify by family |
| H3 | ≥ 2/32 but fails a control | excluded; no RL wording |
| H3 | < 2/32 or cap binds | excluded, no rerun; stratify by seed and layer group |
| H4 | gain LB > 0 and the guard improves | becomes a library operator |
| H4 | fail | no projection |
| S | pass on ≥ 1 objective | R1 features for those objectives only; H-S claim only if a fresh simulation powers it |
| S | inconclusive | descriptive only |
| S | kill, not measured, or ineligible | transport dropped; oracle gradients leave the bank (D2) |
| Guard audit | precision or recall < 0.9 | re-specify and re-calibrate; no freeze |

## Decision table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| H-E1g primary | (b) = model-card benchmarks | Holm, Tango 0.025, guards | 0.45, about $2.40 | H-E1s (17.5 GPU-h) |
| A8' ranking | (a)+(b) need H-Tp and H-E2g | cheapest powered set in top rank | CPU | {H-E1g} |
| 0.5× de-biased δ, Wilson d | winner's curse | N = 3,463 / 1,784 | (included above) | pool < N → unresolved |
| E1g-B at 4,096 tokens | possible truncation artefact | fixed sequence | about 0.5–1 (estimate) | claim scoped to 2,048 tokens |
| H-E2g fixed sequence | the WM must beat search without a WM | two-way bootstrap | re-simulation | descriptive E3 |
| H-Tp | valid trajectory with matched nulls | C_Tp LB > 0 | about 21 (estimate) | unpowered → descriptive |
| Symmetric shrinkage | matches PREREG §6.1.2 | `null_quadratic` size ≤ 0.035 | CPU | H-T stays unresolved |
| `wme-c2-gen` + exposures | pilot dev used for selection | hash disjointness | 0 | no freeze |
| Guard audit | base "refused" ≈ 0.9 on harmless prompts | precision and recall ≥ 0.9 | 0 | re-specify the guard |
| §5 consequences | nothing post hoc | final pilot-3 report | 0 | as tabled |
