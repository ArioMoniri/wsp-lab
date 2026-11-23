# Study M′ (EXPLORATORY): phase reports 2–4 and final report (23 Nov 2026)

**Plan:** `docs/PREREG_WMM.md`, tag `prereg-wmm-exploratory` (`1904bbe`). **Driver:** `run/wmm-mprime`.
**Run:** `data/wmm/mprime-2026-11-21` (not committed). **Machine-readable:** `wmm-mprime-final-2026-11-23.json`.
**Base report:** `wmm-mprime-base-2026-11-22.md`.

Every outcome was read through the FC1 read-gate. Every result is **exploratory**. Intervals are two-sided 95 %
(Tango, paired). Claim tests use the registered one-sided levels (α_f = 0.025/3 for endpoints, 0.025 for guards and
non-inferiority).

**E4 status: done (23 Nov 2026; §5).** The frozen refusal rule **failed** the audit on all three harmful panels
(rule–judge agreement 0.66–0.74), and passed on all three benign panels (≥ 0.986). As registered, the judges'
consensus labels replace the rule for harmful compliance. Rule-based harmful-compliance figures in §2–§3 are kept
for the record but are **superseded**.

## Ledger and cuts (FC2)

| Phase | GPU-h | Status |
|---|---|---|
| Job start / end (canaries, byte hashes) | 0.03 | done |
| Base | 20.55 | done |
| M-SYC (pool, training, 7 sets) | 16.13 | done |
| M-TE without ARC (pool 512, rewrite, training, 5 sets) | 5.55 | done |
| **Total** | **42.26** of 42.0 | hard stop reached after `mte:harm` (started at 41.81) |

**Cuts applied, in the registered order:**
1. **M-A minimum bank (20 units) and the M-A base/pool sets:** not measured (cap).
2. **M-TE ARC guard:** not measured (cap). The M-TE claim is therefore **not made**.
3. **M-A remaining 12 units:** not measured (cap).
4. TE-I: not run.

The schedule placed the M-A minimum before the M-TE ARC guard, so both were lost together once the ledger passed 42.0.
Cause: throughput ran at about 1.5–2.4× the Gate 1 estimates on the shared host (CPU contention; see the base report's
engineering note).

**Integrity.** Job-start and job-end weight sketches identical; batch-1 behavioural canary identical at both ends;
base sha256 from bytes `8ef19ef6…`; one hash-chained provenance record per realised edit (M-SYC, M-TE); every restore
returned the base bytes. No OOM; every set ran at batch 32.

## Phase 2: M-SYC (sycophancy LoRA, λ 1; 98 training rows, 26 steps)

**Exploratory wording.** On MedGemma 1.5 4B, the sycophancy LoRA reduced wrong-suggestion adoption on MedQA from
24.8 % to 9.9 %, **−14.9 pp (95 % CI −17.0 to −12.9)**. **The claim is not made: two guards failed** — G-CAL (as
expected) and harmful compliance (not expected).

| Condition | Estimate [95 % CI] | Registered test | Result |
|---|---|---|---|
| Adoption change (primary, n 1,544) | −14.9 pp [−17.0, −12.9] | UB < 0 at α_f | pass |
| Adoption change, parsed in both (n 1,452) | −15.3 pp [−17.4, −13.3] | UB < 0 at α_f | pass |
| Neutral accuracy, MedQA + MedMCQA (n 3,544) | −0.06 pp [−1.5, +1.4] | LB ≥ −3 pp | **pass** (expected fail; did not fail) |
| **G-CAL ΔBrier** (n 3,544) | **+0.043 [+0.039, +0.047]** | UB ≤ +0.01 | **FAIL** (expected) |
| Harmful compliance, frozen rule (n 1,700) | +1.7 pp [+0.4, +3.0] | — | superseded: rule failed E4 |
| **Harmful compliance, judge consensus (E4; n 73 + 73, unpaired)** | **+1.4 pp [−11.0, +13.7]** | UB ≤ +2 pp | **FAIL** (bound too wide; no increase shown) |
| Benign over-refusal (n 1,500) | −0.7 pp [−1.5, −0.1] | UB ≤ +2 pp | pass (rule confirmed by E4) |
| ARC general accuracy (n 2,161) | +0.5 pp [−1.4, +2.4] | LB ≥ −3 pp | pass |

Descriptive:
- Parse rate rose 1.9 pp [0.8, 3.1]; accuracy under a wrong suggestion rose 11.1 pp [8.8, 13.4].
- Generated-answer Brier change (≡ −Δaccuracy, F1): −0.004 [−0.018, +0.010].

**The trade-off (addendum).** Adoption fell about 15 pp at no measurable neutral-accuracy cost, but the edit made the
model's direct-answer letter probabilities worse calibrated (+0.043 Brier, four times the margin, matching dev +0.040).
The rule-based +1.7 pp rise in harmful compliance is **not supported**: the rule failed E4. Under the judges' labels
the change is +1.4 pp with a wide interval, so the harm guard still fails, because safety could not be shown, not
because an increase was found. Harmful outputs stay internal.

## Phase 3: M-TE (short-form self-rewrite LoRA, λ 1; 388 correct pool answers, 98 valid rewrites, 26 steps)

**Exploratory wording.** The token-efficiency LoRA cut mean response length on MedQA + MedMCQA from 1,170 to 294
tokens (Δ log-tokens −1.71 [−1.74, −1.69]) and raised accuracy at a 1,024-token budget by +23.4 pp [+21.6, +25.2],
but **lowered full-budget accuracy by 2.7 pp [−4.4, −1.1]**, failing its −2 pp non-inferiority limit. **The claim is
not made** (non-inferiority failed, and the ARC guard was cut by the cap).

| Condition | Estimate [95 % CI] | Registered test | Result |
|---|---|---|---|
| Δ log-tokens (n 3,544) | −1.71 [−1.74, −1.69] | UB < 0 at α_f | pass |
| Accuracy at 512 | +54.3 pp [+52.6, +55.9] (3.5 % → 57.8 %) | LB > 0, Holm within α_f | pass |
| Accuracy at 1,024 | +23.4 pp [+21.6, +25.2] (35.9 % → 59.2 %) | LB > 0, Holm within α_f | pass |
| **Full-budget accuracy, untruncated in both** (n 3,405) | **−2.7 pp [−4.4, −1.1]** | LB ≥ −2 pp | **FAIL** |
| G-CAL ΔBrier | −0.011 [−0.014, −0.007] | UB ≤ +0.01 | pass |
| Harmful compliance, frozen rule | −0.7 pp [−1.9, +0.6] | — | superseded: rule failed E4 |
| **Harmful compliance, judge consensus (E4; n 74 + 73, unpaired)** | **+11.9 pp [−1.6, +25.0]** | UB ≤ +2 pp | **FAIL** |
| Benign over-refusal | −0.4 pp [−1.2, +0.3] | UB ≤ +2 pp | pass (rule confirmed by E4) |
| ARC general accuracy | — | LB ≥ −3 pp | **not measured (cap)** |

Descriptive: full-budget accuracy on all items (truncated = wrong) −2.7 pp [−4.3, −1.0]; generated-answer Brier
change +0.025 [+0.009, +0.041] (≡ −Δaccuracy, F1). The low-budget gains are mostly the base model running out of
budget (base accuracy at 512 is 3.5 %), not better reasoning.

## Phase 4: M-A

**Not measured.** The hard stop removed the whole M-A phase, including the declared 20-unit minimum. Registered
wording: "beyond-dose prediction not established on study M". No M-A data exist; the sealed `ma` panel and `pool_ma`
were not read.

## 5. E4 guard-scorer re-audit (registered §3.1)

**Setup.** No judge was pinned in the plan, so per the chair both judges were `claude-opus-5-5`, run as separate agents
with different prompts and the same refuse/comply definition. Each judge labelled all 450 items: 75 harmful and 75
benign per state (base, M-SYC, M-TE), shuffled together, with neutral ids, no state shown, and no access to the rule
labels, the keymap or the other judge's labels. Judges read every response in full. Packets, outputs and labels stay
internal under `data/wmm/`; this report gives aggregates only.

**Deviations, disclosed.**
- The packet builder first truncated long responses. It was fixed to give full text before any labelling, and the
  item set and ids were unchanged.
- One Judge B shard (75 items) reported labelling 54 items from excerpts after a safety filter cut off its read. Those
  labels were discarded unused, and the shard was re-run as three single-chunk agents that read every item in full.
- **Unlabelled:** Judge A 0, Judge B 1 (a benign M-SYC item); that item is excluded from consensus.

| Panel | Judge–judge agreement | Rule–judge agreement (≥ 0.9) | Consensus refusal | Rule refusal (same sample) | Rule misses a refusal / rule false refusal |
|---|---|---|---|---|---|
| base harmful | 0.973 | **0.740 FAIL** | 83.6 % | 56.0 % | 19 / 0 |
| M-SYC harmful | 0.973 | **0.658 FAIL** | 82.2 % | 52.0 % | 23 / 2 |
| M-TE harmful | 0.987 | **0.730 FAIL** | 71.6 % | 46.7 % | 19 / 1 |
| base benign | 1.000 | 1.000 pass | 1.3 % | 1.3 % | 0 / 0 |
| M-SYC benign | 0.986 | 0.986 pass | 1.4 % | 0.0 % | 1 / 0 |
| M-TE benign | 1.000 | 1.000 pass | 2.7 % | 2.7 % | 0 / 0 |

**Registered consequence.**
- **Benign over-refusal:** the rule stands, and the rule-based results stand (M-SYC and M-TE pass).
- **Harmful compliance:** the judges agree ≥ 0.9, so their consensus labels replace the rule. The rule mostly
  misses refusals that come later in the reply or are not phrased in the first person, so it overstates compliance
  (rule-based base harmful compliance is about 41 %; consensus base compliance is 16.4 % [9.7, 26.6]).

**Harmful compliance under consensus labels (one-sided 97.5 % UB ≤ +2 pp; Newcombe interval for independent
samples).**

| State | Compliance [95 % Wilson CI] | n | Change vs base [95 % CI] | Guard |
|---|---|---|---|---|
| base | 16.4 % [9.7, 26.6] | 73 | — | — |
| M-SYC | 17.8 % [10.7, 28.1] | 73 | +1.4 pp [−11.0, +13.7] | **FAIL** |
| M-TE | 28.4 % [19.4, 39.5] | 74 | +11.9 pp [−1.6, +25.0] | **FAIL** |

The registered audit samples were drawn separately for each state. They share only 6 (M-SYC) and 1 (M-TE) items with
the base sample, so the consensus guard cannot be paired, and it has no power to bound a change at 2 pp. This is a
design weakness of the registered audit, recorded here, not repaired after the fact.

**Does the M-SYC harmful-compliance failure stand?** As a guard outcome, **yes**: it fails under the registered
consensus labels, so the claim stays unmade (G-CAL also failed). **Its basis changes.** The rule-based finding of a
+1.7 pp increase is withdrawn, because the rule failed the audit. Under the judges' labels the point estimate is
+1.4 pp with a 25-point-wide interval. On study M, the data neither show that the sycophancy edit increases harmful
compliance nor show that it does not.

## Final exploratory summary

1. **M-SYC:** large adoption reduction (−14.9 pp), neutral accuracy unchanged; claim not made because G-CAL failed
   (expected, disclosed in the plan) and the harmful-compliance guard failed under E4 consensus labels, by a
   too-wide bound (+1.4 pp [−11.0, +13.7]), not by a shown increase.
2. **M-TE:** a ~4× token cut with large low-budget gains, at a −2.7 pp full-budget accuracy cost; claim not made
   (non-inferiority failed; ARC guard cut).
3. **M-A, TE-I:** not run (cap).
4. **E4:** the frozen refusal rule is valid for benign over-refusal and **invalid for harmful compliance** on this
   model (agreement 0.66–0.74), so every rule-based harmful-compliance figure on study M is superseded. Measuring the
   harm guard properly would need judge labels on the paired 1,700-item panel. That would be a new, unregistered
   measurement and needs a chair decision.
