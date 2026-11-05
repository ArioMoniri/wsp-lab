# TF-v1 qualification (section 6) and TF-specific power (section 5), 5 November 2026

Author: eval/metrology. Development panel only; no fresh component has been scored. Branch `feat/tf-v1-prepanel`; inputs: the timing pilot (10 D3 controls + base), G1 (`repeats/g1`), G2 (bridge), the sealed partition manifest (components, metadata only). Source `source_hash` 12ee875a68c9, device H0 only. Every number below is produced by `scripts/tf_v1_qualification.py` and `scripts/tf_v1_power.py` (4,000 draws, seed 20261105) and rendered by `scripts/tf_v1_qualification_doc.py`. This document qualifies nothing by itself: it applies the registered rule (PREREG_H1_TF §6, amendment A1) and reports where it passes or fails.

## 1. Verdict

| Endpoint | δ | Measured provenance | Paired repeat error (max / p95 upper) | Item half-width: every entry / panel-mean upper bound vs δ/2 | Reliability lower 95% bound ≥ .90 | Result |
|---|---:|---|---|---|---|---|
| MMLU-Pro gold log-prob (predecessor-relative) | 0.100 | native-measured, H0, clean, matched | 0.0e+00 / 0.0e+00 (pass) | **FAIL** (max 0.0893; 2 over) / pass (0.0312 vs 0.0500) | pass (lower 0.938, point 0.991) | **NOT QUALIFIED** |
| KL(base‖edited) | 0.010 | native-measured, H0, clean, matched | 0.0e+00 / 0.0e+00 (pass) | **FAIL** (max 0.0103; 1 over) / pass (0.0031 vs 0.0050) | pass (lower 1.000, point 1.000) | **NOT QUALIFIED** (KL > 0.10 excluded: 5/20 = 25%; tripwire 20%: TRIPPED) |
| WikiText log-PPL (predecessor-relative) | 0.049 | native-measured, H0, clean, matched | 0.0e+00 / 0.0e+00 (pass) | pass (max 0.0110; 0 over) / pass (0.0059 vs 0.0244) | pass (lower 0.945, point 0.984) | **QUALIFIED** |

Reading. WikiText log-PPL passes every registered component (diagnostic; it cannot pass H1a). The H1a primary endpoint (MMLU gold log-prob) passes provenance, the repeat bound (exactly zero), the panel-mean δ/2 bound and the reliability bound on the 20-entry panel, but **fails the registered per-entry δ/2 condition** (A1.3): two large-effect LoRA entries have item half-widths of 0.0893 (`b5d0ac5d76c3cad6`), 0.0608 (`c3ae3e8c8edca6b8`) against δ/2 = 0.050. KL is **not qualified** under A1.2 (see §4). **The H1a primary endpoint is therefore not qualified by the rule as registered.** The failure is a property of the larger LoRA edits, which move individual items' log-probs a lot (§4), not of noise in repeats.

## 2. Panel and provenance

Composition (PREREG §6 rule, `docs/prereg/qualification_panel.json`): LoRA nearest median legacy KL `2f61a4086395b719`, `b5d0ac5d76c3cad6`; LoRA largest `d7f933d3b4e5d25e`, `c3ae3e8c8edca6b8`; chains (step ≥ 3, distinct lineages) `0027efdf2eae7405`, `00f8111cbad8218f`, `057dcfbedea3251f` with predecessors `b7edbb9f8b339f52`, `34ee195e55227908`, `86ea1fe0ec777cf8`; task vectors `023b384a7799a787`, `02a1b5e21971290f`; largest-KL abliteration `5c766ee2fa8bf915`. The ten-entry panel is **expanded to 20 y-bearing entries** with the ten already-measured D3 pilot controls (3 random_dir, 3 abliteration, 2 LoRA, 2 task-vector; same source, same base run, same device). That is a deviation from the prespecified hash-order expansion, made because those entries were already measured; both panels are reported. Chain entries use their actual scored predecessor as reference; every other entry uses the base. Provenance of every entry, predecessor and the base: native-measured, device H0, batch 1, clean tree, complete, same `source_hash`, base matched (`matched_base_missing: false`): **all pass**.

| Entry | Family | MMLU y | MMLU half-width | KL | KL half-width | WikiText y | WikiText half-width |
|---|---|---:|---:|---:|---:|---:|---:|
| `2f61a4086395b719` | lora_sft | 0.0430 | 0.0386 | 0.1525 | 0.0102 | -0.0841 | 0.0091 |
| `b5d0ac5d76c3cad6` | lora_sft | -0.8017 | 0.0893 | 0.0699 | 0.0103 | -0.0858 | 0.0101 |
| `d7f933d3b4e5d25e` | lora_sft | 0.0323 | 0.0420 | 0.5692 | 0.0284 | -0.0503 | 0.0110 |
| `c3ae3e8c8edca6b8` | lora_sft | -0.1681 | 0.0608 | 0.5868 | 0.0335 | -0.0384 | 0.0109 |
| `0027efdf2eae7405` | chain | -0.0043 | 0.0037 | 0.0285 | 0.0022 | 0.0008 | 0.0004 |
| `00f8111cbad8218f` | chain | 0.0005 | 0.0032 | 0.0029 | 0.0002 | 0.0002 | 0.0004 |
| `057dcfbedea3251f` | chain | -0.0037 | 0.0033 | 0.6916 | 0.0312 | -0.0001 | 0.0003 |
| `023b384a7799a787` | task_vector | -0.0525 | 0.0105 | 0.0104 | 0.0018 | 0.0489 | 0.0035 |
| `02a1b5e21971290f` | task_vector | -0.0718 | 0.0229 | 0.0091 | 0.0007 | -0.0483 | 0.0043 |
| `5c766ee2fa8bf915` | abliteration | 0.0141 | 0.0075 | 0.0139 | 0.0030 | 0.0068 | 0.0012 |
| `b6fe7f59016cdda2` | random_dir | -0.0138 | 0.0039 | 0.0011 | 0.0000 | -0.0009 | 0.0004 |
| `3ddc3b4bf47ed994` | abliteration | -0.0085 | 0.0143 | 0.0225 | 0.0018 | 0.0625 | 0.0053 |
| `0a841d4082532e8b` | lora_sft | -0.1274 | 0.0323 | 0.0205 | 0.0016 | -0.0964 | 0.0073 |
| `3db97b535c996265` | task_vector | 0.0013 | 0.0021 | 0.0004 | 0.0000 | -0.0003 | 0.0003 |
| `7b8247d8d7ab7db9` | random_dir | 0.0017 | 0.0022 | 0.0005 | 0.0000 | -0.0004 | 0.0003 |
| `439dbfa62a14d55f` | abliteration | -0.0182 | 0.0054 | 0.0033 | 0.0002 | 0.0090 | 0.0012 |
| `af6d608f8651fb46` | lora_sft | 0.0412 | 0.0434 | 0.3230 | 0.0173 | -0.1035 | 0.0106 |
| `dd90d99e2ddc0e77` | task_vector | 0.0005 | 0.0026 | 0.0007 | 0.0000 | -0.0004 | 0.0003 |
| `db5c3a7bfc19178c` | random_dir | -0.0097 | 0.0041 | 0.0019 | 0.0001 | -0.0013 | 0.0005 |
| `1caef8eb4c640c3c` | abliteration | -0.0070 | 0.0036 | 0.0013 | 0.0001 | -0.0005 | 0.0003 |

## 3. Rule components, per endpoint (20-entry panel; 10-entry composition in brackets where it differs)

### MMLU-Pro gold log-prob (predecessor-relative), δ = 0.1000

- **Paired repeat error (G1: the 10 composition entries, their 3 chain predecessors and the base, each executed twice; the 10 pilot controls have no g1 repeat):** 10,000 item values; max |repeat − primary| = 0.0e+00 (limit 1e-04), 95th percentile 0.0e+00, upper 97.5% component-bootstrap bound 0.0e+00 (limit δ/2 = 0.0500); bitwise identical: True. The repeat-error variance entering reliability is therefore 0; a bound of 0 reflects deterministic same-device execution, not a measurement of device error.
- **Item uncertainty (joint item draws across edits and the shared reference; blocks of 2 windows for WikiText):** per-entry half-width median 0.0065, max 0.0893 (limit δ/2 0.0500); entries over the limit: `b5d0ac5d76c3cad6`, `c3ae3e8c8edca6b8`; panel-mean half-width 0.0198, upper 97.5% component-bootstrap bound 0.0312.
- **Signal reliability** (family + log-norm OLS residuals, signal = between-entry residual variance − mean item-error variance − repeat variance; untruncated): n = 20, design rank 6, residual variance 2.784e-02, mean item-error variance 2.48e-04, signal variance 2.759e-02 (positive: True); reliability 0.9911, 95% component-bootstrap interval [0.938, 0.994] (4000 valid of 4,000 draws). [10-entry composition: reliability 0.992, lower bound 0.770, **FAIL**; item half-width bound 0.0463.]

### KL(base‖edited), δ = 0.0100

- **Paired repeat error (G1: the 10 composition entries, their 3 chain predecessors and the base, each executed twice; the 10 pilot controls have no g1 repeat):** 2,000 item values; max |repeat − primary| = 0.0e+00 (limit 1e-05), 95th percentile 0.0e+00, upper 97.5% component-bootstrap bound 0.0e+00 (limit δ/2 = 0.0050); bitwise identical: True. The repeat-error variance entering reliability is therefore 0; a bound of 0 reflects deterministic same-device execution, not a measurement of device error.
- **Item uncertainty (joint item draws across edits and the shared reference; blocks of 2 windows for WikiText):** per-entry half-width median 0.0002, max 0.0103 (limit δ/2 0.0050); entries over the limit: `b5d0ac5d76c3cad6`; panel-mean half-width 0.0015, upper 97.5% component-bootstrap bound 0.0031.
- **Signal reliability** (family + log-norm OLS residuals, signal = between-entry residual variance − mean item-error variance − repeat variance; untruncated): n = 15, design rank 6, residual variance 1.197e-02, mean item-error variance 2.29e-06, signal variance 1.197e-02 (positive: True); reliability 0.9998, 95% component-bootstrap interval [1.000, 1.000] (4000 valid of 4,000 draws). [10-entry composition: reliability 1.000, lower bound 0.995, pass; item half-width bound 0.0063.]

### WikiText log-PPL (predecessor-relative), δ = 0.0488

- **Paired repeat error (G1: the 10 composition entries, their 3 chain predecessors and the base, each executed twice; the 10 pilot controls have no g1 repeat):** 640 item values; max |repeat − primary| = 0.0e+00 (limit 1e-04), 95th percentile 0.0e+00, upper 97.5% component-bootstrap bound 0.0e+00 (limit δ/2 = 0.0244); bitwise identical: True. The repeat-error variance entering reliability is therefore 0; a bound of 0 reflects deterministic same-device execution, not a measurement of device error.
- **Item uncertainty (joint item draws across edits and the shared reference; blocks of 2 windows for WikiText):** per-entry half-width median 0.0012, max 0.0110 (limit δ/2 0.0244); entries over the limit: none; panel-mean half-width 0.0039, upper 97.5% component-bootstrap bound 0.0059.
- **Signal reliability** (family + log-norm OLS residuals, signal = between-entry residual variance − mean item-error variance − repeat variance; untruncated): n = 20, design rank 6, residual variance 5.360e-04, mean item-error variance 8.51e-06, signal variance 5.275e-04 (positive: True); reliability 0.9841, 95% component-bootstrap interval [0.945, 0.994] (4000 valid of 4,000 draws). [10-entry composition: reliability 0.983, lower bound 0.918, pass; item half-width bound 0.0079.]

## 4. Why the result is what it is, and what it does not show

- **MMLU per-entry gate.** The ten smallest item half-widths are ≤ 0.0054, but 6 entries exceed 0.03 (`b5d0ac5d76c3cad6` (lora_sft, y -0.802, half-width 0.089), `c3ae3e8c8edca6b8` (lora_sft, y -0.168, half-width 0.061), `af6d608f8651fb46` (lora_sft, y 0.041, half-width 0.043), `d7f933d3b4e5d25e` (lora_sft, y 0.032, half-width 0.042), `2f61a4086395b719` (lora_sft, y 0.043, half-width 0.039), `0a841d4082532e8b` (lora_sft, y -0.127, half-width 0.032)); 2 exceed δ/2 = 0.050. The half-width follows how much an edit moves individual items' log-probs, not the size of the mean change (several entries with |y| ≈ 0.04 have half-widths near 0.04), so it is a property of the larger LoRA edits, not of repeat noise. The panel-mean bound (0.0312) passes. The panel includes the two largest-KL LoRAs by the §6 rule. I do not relax the rule after seeing it fail; options for the owner are in §7.
- **KL.** Under A1.2 entries above 0.10 nats are excluded from the gate: 5 of 20 here (`2f61a4086395b719`, `d7f933d3b4e5d25e`, `c3ae3e8c8edca6b8`, `057dcfbedea3251f`, `af6d608f8651fb46`), above the 20% tripwire, so KL is unqualified for this panel regardless of the other components. Among the 15 in-gate entries, `b5d0ac5d76c3cad6` still has half-width 0.0103 > δ/2 = 0.005: its KL is below 0.10 and the half-width is 0.15 of its KL (measured relative half-widths of the entries above 0.03 range 0.05 to 0.15), so a threshold of 10δ is too high for an absolute δ/2 bound. That is a defect of the A1.2 threshold, reported here and not repaired retroactively.
- **Reliability is panel-dependent.** It is high on this deliberately dispersed panel (LoRA, abliteration, chain, task) because between-entry variance is large relative to the item error. On a fresh panel of mostly small edits the between-entry variance is smaller and the same error would give a lower ratio; the estimate cannot be carried over without recomputing on the fresh panel's variance. The nuisance (family, log-norm) is fitted on these 20 entries because no D3 TF-v1 training rows exist yet (the registered nuisance is training-only), and the norm is the stored cumulative Frobenius norm, not the incremental effective norm against the predecessor; both are approximations that matter little at 6 parameters but are not the registered computation.
- **Not shown:** device error (single device, DECISIONS D2 addendum), cross-batch/engine agreement, signal or effect resolution for HellaSwag and the phrase margins (descriptive, unqualified; only their bitwise repeat is established), and anything about fresh components.

## 5. TF-specific power for θ (planning computation, not the §5 gate)

**Achievable count (sealed partition, `docs/prereg/panel_manifest.json`, partition hash `df4689983f1bfe01…`).** 483 source-closed components; development 252 (143 non-null), excluded 19, fresh-unselected 112, **confirmatory 100** (all non-null): 71 fresh components containing any of the 110 lora_sft/task_vector candidates (80 candidate entries) plus 29 chain components appended in hash order to reach 100. Only 6 more fresh chain components exist, so the achievable maximum is 106 and any post-freeze exclusion leaves almost no margin above the ≥100 floor. Rows: 241 stratum rows (chain 161, lora_sft 55, task_vector 25); 278 closure entries (255 unique deltas, aliases scored once). Component sizes (rows): 1: 60, 10: 1, 17: 1, 2: 15, 3: 2, 4: 7, 5: 4, 6: 1, 7: 8, 8: 1. Most components are 1 or 2 rows; a few chain lineages carry up to 17 rows.

**What is missing for the §5 gate.** The gate needs the D3 out-of-fold TF-v1 residuals and the StatNN/isotonic predictions, item/reference covariance and device support. None exists: no D3 entry has been scored under TF-v1 except the 23 here, and no predictor has been fit on TF-v1 labels. The computation below is therefore a planning calculation. It uses the measured TF-v1 pilot residual scale (|family/log-norm residual| of the 20 entries, CV 1.97) for the comparator loss and a grid of assumptions for how the StatNN/isotonic loss ratio varies across rows. The alternative is built so the **row-summed ratio of expectations is exactly θ = .15** (and 0 for the null), losses nonnegative, row ratio lognormal with within-component correlation ρ. Panels use the actual component size distribution; each of 4,000 simulated panels is analysed with the registered conditional procedure (4,000 component-bootstrap draws of the row-summed ratio, 2.5th percentile > 0). The joint component/item-block procedure is **not run**: it needs item-level paired predictions that do not exist; I report only the conditional procedure and a variance-inflated sensitivity (scenarios D and F).

| Scenario | Loss shape | Row heterogeneity s | ρ within component | Component scale sd | Power at θ=.15 (lower binomial bound) | Precision gate attained (half-width ≤ θ̂/2) | Decision pass rate (θ̂ ≥ .15 ∧ lower > 0 ∧ half-width ≤ θ̂/2) | θ̂ mean (sd) | Null rejection rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | pilot_abs_residual | 0.3 | 0.3 | 0.5 | 0.880 (0.870) | 0.483 | 0.416 | 0.150 (0.046) | 0.046 |
| B | pilot_abs_residual | 0.6 | 0.3 | 0.5 | 0.485 (0.469) | 0.147 | 0.147 | 0.150 (0.096) | 0.070 |
| C | pilot_abs_residual | 0.6 | 0.6 | 0.5 | 0.449 (0.433) | 0.124 | 0.124 | 0.150 (0.106) | 0.076 |
| D | pilot_abs_residual | 1.0 | 0.6 | 0.5 | 0.307 (0.293) | 0.097 | 0.097 | 0.142 (0.208) | 0.117 |
| E | lognormal_cv1 | 0.6 | 0.6 | 0.5 | 0.540 (0.525) | 0.165 | 0.165 | 0.151 (0.082) | 0.067 |
| F | lognormal_cv1 | 1.0 | 0.6 | 1.0 | 0.345 (0.331) | 0.109 | 0.109 | 0.151 (0.210) | 0.116 |

Component-count sweep, scenario C (actual size distribution resampled to n components):

| Components | Power (lower bound) | Precision gate | θ̂ sd |
|---:|---:|---:|---:|
| 60 | 0.391 (0.376) | 0.115 | 0.128 |
| 80 | 0.424 (0.409) | 0.124 | 0.122 |
| 100 | 0.449 (0.433) | 0.126 | 0.109 |
| 120 | 0.475 (0.459) | 0.138 | 0.098 |
| 160 | 0.544 (0.529) | 0.176 | 0.083 |
| 200 | 0.587 (0.571) | 0.194 | 0.076 |

**Reading.** At the achievable 100 components (241 rows), scenario A, whose θ̂ spread (sd 0.046, a cluster SD near 0.46) is the closest to the metrology planning value of 0.47 behind the "79 clusters" figure, has conditional power 0.880 (lower binomial bound 0.870), but attains the registered precision gate (half-width ≤ θ̂/2) in only 48% of panels and the full decision rule in 42%. Scenarios with lower-bound power ≥ .80: A; below: B, C, D, E, F. §5 requires the lower binomial bound ≥ .80 for both the conditional and the joint procedure and a separate precision report; the joint procedure was not run, so **the §5 gate is not satisfied**, and where the conditional bound is met it depends on assumptions that only the D3 OOF losses can confirm. The decision rule also needs half-width ≤ θ̂/2, which is not met in most simulated panels even when the rejection test succeeds: per §4 an imprecise result is inconclusive, not negative. The old 79-cluster figure is not carried over. **The registered percentile bootstrap is anti-conservative here:** under the null (θ = 0) it rejects in 0.046 to 0.117 of panels against the nominal one-sided 0.025 (heavy-tailed row losses, unequal component sizes, a ratio statistic and only 100 components). The power figures above are therefore optimistic: they include false rejections, and a calibrated test would have less power. This is a planning finding for the registered procedure, to be re-examined on the real OOF losses.

## 6. Ancestry-cluster bootstrap plan (registered procedures, restated as executable steps)

1. Unit: source-closed component from the sealed manifest (`component_id`); all rows, both predictions and shared reference links of a component move together. Selected components are drawn with replacement at the original count (100), seed 20261105, 4,000 draws, one stream shared by all contrasts.
2. Statistic: θ = 1 − ΣA/ΣB over the drawn rows (row sums, not a mean of component ratios), A = |r − mean of 3 StatNN seed predictions|, B = |r − isotonic|; B/n ≤ 1e-12 nats in any replicate counts as a denominator failure (not clamped, reported; any failure leaves the precision gate unmet).
3. Report 2.5/97.5 percentiles, valid-draw count, component-size distribution; decision needs θ̂ ≥ .15, lower bound > 0, half-width ≤ θ̂/2.
4. Item claim: additional 4,000-draw joint bootstrap of components and sealed independent problem blocks with one common item draw across every edit, predecessor and the shared base; fixed D3 fit. Both intervals must satisfy the decision.
5. Qualification bootstraps (this document) use the same unit (the dev-panel components from the manifest), the same seed and the same 4,000 draws; shared references are drawn once per replicate.
6. Power simulation (§5) reuses steps 1 to 3 on simulated losses, 4,000 panels per scenario, with the real D3 OOF losses substituted for the proxy before sealed execution.

## 7. Decisions and blockers for the owner (none is taken here)

1. **MMLU per-entry δ/2.** The registered rule fails on 2 of 20 entries (large effects). Options, each needing a dated amendment before fresh scoring: keep the rule and accept the endpoint unqualified; scope the per-entry check to entries with |y| below a stated multiple of δ (as A1.2 does for KL) and report the rest; or switch the per-entry condition to the panel-level bound plus a relative condition. This choice should be made on grounds other than this pilot's outcome (for example the H1a stratum's expected effect sizes).
2. **KL threshold.** A1.2's 10δ is too high by the measured relative width; a lower threshold (and its tripwire) needs an amendment, and KL cannot pass H1a anyway.
3. **Budget.** The as-built source does not fit the 19.0 h cap at the actual closure count (PREREG A1.4); the batching remedy (A1.5) would, but needs a source change, a new pilot, G1 and G2 on the final source, and invalidates the artefacts used here (they would become development evidence).
4. **Sealed item panel** (TF-v1:item domain, text-group dedupe, disjointness from the development rankings and legacy items) is not built; no sealed item may be scored before it is committed.
5. **Power.** Re-run `scripts/tf_v1_power.py` with the D3 OOF TF-v1 residuals and predictions once D3 is scored, and add the joint component/item-block procedure; the 100-component floor is exactly met and only 6 spare fresh chain components exist, so expansion within the cap is limited to 106.
