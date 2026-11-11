# F12b — repairs to the F12 simulation design

Design note for PREREG_H1_TF A5 amendment **F12b** (11 Nov 2026, before any G3 outcome). Branch `prereg/f12b`, main
`2255f5e`. **Every number in this note is synthetic-token evidence.** It comes from the deterministic synthetic token matrix of
`scripts/f12_simulation.py` (270 rows / 234 components with the F12.2 counts; not zoo tokens) and from the restricted copy of
the F8a.2 noise payload (`noise_payload_restricted.json`; fields `centred_noise`, `window_mask`, `window_ids`,
`donors[j].{id, component, family}`; provenance sha256 `67da223c…3de`). No real token, outcome, `donor_mean`, `data/evals` file
or Part A payload was read. Replicate counts are ≤ 20 per scenario. The γ panels are noise-only calibration panels (no model is
fitted). Outputs are in `/home/ubuntu/wsp-data/runs/a5/f12_sim_dryrun/f12b/` (not committed; sha256s in
`f12b_reduced_summary.json` there). CPU seconds are process CPU on a shared 8-core machine (a 6-thread token job was running).

## 1. What failed: the F12-as-drafted smoke run

`scripts/f12_simulation.py` at `07e1dc2`; synthetic tokens; k = 145; 100-tree StatNN; 7 null + 7 alternative replicates before
the chair stopped the run (`progress.log`). Four replicates were re-run bit-for-bit with `f12b_diagnostics.py smoke`: C_WM was
−33.84, −12.41, −9.42 and −17.08, with 18, 6, 0 and 109 non-positive joint draws, exactly as in the log.

| finding | evidence |
|---|---|
| (a) head mis-scaled | default-initialised K+F head output ≈ 10 nats (sum pool over 48–132 tokens) vs F3 residual ≈ 0.0015 nats; C_WM −5.8 to −34 in all 14 replicates; one probe fold: held-out WM-R error 37.0 × F3's |
| (b) joint studentiser | V_comp + V_item − V_both ≤ 0 in 5–109 of 4,000 draws in 13 of 14 replicates → "unresolved (validity)" |
| (c) γ tolerance | the validation C on 100 panels was 0.119–0.175 (SE ≈ 0.027); ±0.005 is out of reach at that panel count |
| (d) cost | ≈ 570 s per replicate × 1,000 × 6 k × 2 scenarios ≈ 1,900 CPU-h |

## 2. Repair 1 — head scaling (F12b.1)

**Rule.** (i) The output weights w (32) and the unused output bias are set to exactly 0 after the unchanged PyTorch default
construction under `torch.manual_seed(s)`. (ii) For each fit (every inner-training split and the outer refit), the residual
target r = Δy − ŷ_F3 is divided by s = SD(r) over **that fit's training rows** (ddof 0). The head trains on r/s and its
prediction is multiplied by s. If s ≤ 1e-12 or s is nonfinite, the fit fails (closed). (iii) Token standardisation is unchanged
(present-token mean and SD over the fit's training rows; absent slots masked to 0). Nothing else is standardised. The untrained
head is h ≡ 0, so WM-R = F3, and h = 0 at the zero edit for any parameters. Tests:
`test_zero_initialised_head_is_f3_until_trained_and_zero_at_zero_edit` and
`test_target_standardisation_is_training_only_and_inverted`. The second test checks exact 2¹⁰ scale equivariance and that
held-out outcomes never enter the fit.

**Evidence** (`head.json`, k = 145, WM-R against F3 only, the same 6 + 6 replicate panels as the reduced run; C_WM as ratio of
sums, with the median and range in brackets):

| variant | null C_WM | alternative C_WM | CPU-s / replicate (WM-R) |
|---|---|---|---|
| drafted (smoke run) | −5.8 to −34 | −7.4 to −17 | ≈ 185 |
| zero-init only | −0.090 (−0.066; −0.201 to −0.018) | 0.118 (0.157) | 62 |
| target-standardised only | −0.031 (−0.022; −0.092 to 0.004) | 0.145 (0.175) | 58 |
| **zero-init + standardised (registered)** | **−0.005** (−0.008; −0.043 to 0.051) | **0.152** (0.190) | **48** |

The two parts are complementary. With the zero init alone, the first Adam steps still move a raw-scale prediction far past a
0.0015-nat residual. With standardisation alone, the head starts ≈ 10 SD off. Refit epochs fell from about 70–230 (drafted) to
a median of 22 (reduced run), which is the main source of the WM-R saving.

## 3. Repair 2 — joint studentiser (F12b.2)

**Rule (registered `floor_vcomp`).** V\* = max(V_comp + V_item − V_both, V_comp) in every joint draw and at the point. V_comp is
the same draw's component-only term; at the point it equals the conditional linearised variance
(`test_joint_parts_reproduce_the_a542_two_way_studentiser`). No draw is dropped. The conditional procedure and all other
guards are unchanged. Justification: the joint design resamples components and items. The cross-component common-item
correlation of the noise is positive (0.475 in the reduced run's noise check), so V_item − V_both ≥ 0 in expectation, and a
two-way value below V_comp is sampling noise of the subtraction. Each replicate also reports the strict rule and the
one-way-max floor max(·, V_comp, V_item) descriptively.

**Effect on the smoke-run draws** (drafted head, 4 replicates re-run; counts of 4,000 draws; joint bootstrap-t interval on C):

| replicate | contrast | raw ≤ 0 | strict | floor_vcomp: binds / interval | floor_one_way: binds |
|---|---|---|---|---|---|
| null 0 | C_WM | 18 | invalid | 35 / [−1910, −35.3] | 2,552 |
| null 1 | C_WM | 6 | invalid | 32 / [−227, −3.44] | 1,888 |
| alt 0 | C_WM | 0 | [−26.15, −5.89] | 127 / [−26.03, −6.02] | 347 |
| alt 1 | C_WM | 109 | invalid | 184 / [−23.07, −17.05] | 2,397 |
| alt 1 | C_WM,S | 37 | invalid | 53 / [−3151, −35.3] | 2,594 |

On the drafted draws `floor_vcomp` binds in ≤ 4.6 % of draws and turns every replicate valid. Where the strict rule was valid,
it moves the bounds slightly. The one-way-max floor binds in 9–65 % of draws, so it is not registered.

**Effect after the head repair** (reduced run, 20 + 20 replicates): raw non-positive draws occur in **every** replicate (C_WM:
null 72–678, alternative 46–326 of 4,000). The strict rule would leave **0 of 40** replicates valid. `floor_vcomp` binds in
221–1,185 (null) and 99–1,535 (alternative) draws of C_WM, and at the point in 3 + 1 replicates. It is therefore not a
rare-event patch once WM-R ≈ F3: the two-way subtraction is noisy whenever the two predictors are close. Its size consequences
can be certified only by the registered run's size gate. This is the main point for Board A's review of F12b.2.

## 4. Repair 3 — oracle calibration (F12b.3)

**Rule.** The grid {0} ∪ τ·2^(−8…12) and the bracket are unchanged. Bisection runs to |C̄_cal − 0.15| ≤ 0.001 (≤ 50 halvings) on
**4,000** calibration panels (`cal:<k>:<r>:<d>`). Validation uses **4,000** independent panels (`calval:<k>:<r>:<d>`), and
**|C̄_val − 0.15| > 0.02 fails qualification**. The uniqueness argument replaces F12a.4(h)'s non-decreasing check, which is
still reported: C̄(γ) = 1 − N/Σ|u + γv| is an increasing transform of a convex function of γ. It can dip below C̄(0), but
C̄ = 0.15 > C̄(0) has exactly one root (`test_f12b_calibration_grid_bisection_and_validation_tolerance` checks a case with a dip
on a 4,001-point grid).

**Monte Carlo error of the calibrated oracle C by panel count** (`gamma_k0.json`, `gamma_k145.json`; for each h\*, calibrate on N
panels, then measure the achieved C̄ on 20,000 independent reference panels, whose own SE is ≈ 0.0016):

| k | panels N | h\* qualified | achieved C̄ (20,000 ref panels): mean | SD across h\* | max abs dev | validation within ±0.005 / ±0.02 | F12a.4(h) check would fail |
|---|---|---|---|---|---|---|---|
| 0 | 100 | 20/20 | 0.1508 | 0.0173 | 0.0431 | 4 / 13 | 9 |
| 0 | 400 | 20/20 | 0.1477 | 0.0103 | 0.0198 | 5 / 17 | 5 |
| 0 | 1,000 | 20/20 | 0.1509 | 0.0058 | 0.0135 | 4 / 20 | 9 |
| 0 | 4,000 | 20/20 | 0.1495 | 0.0035 | 0.0066 | 17 / 20 | 10 |
| 0 | 10,000 | 20/20 | 0.1500 | 0.0025 | 0.0053 | 18 / 20 | 11 |
| 145 | 100 | 10/10 | 0.1566 | 0.0224 | 0.0494 | 1 / 6 | 3 |
| 145 | 400 | 10/10 | 0.1498 | 0.0113 | 0.0229 | 1 / 9 | 3 |
| 145 | 1,000 | 10/10 | 0.1488 | 0.0108 | 0.0168 | 3 / 8 | 1 |
| 145 | 4,000 | 10/10 | 0.1509 | 0.0044 | 0.0092 | 6 / 10 | 2 |

k = 0: 20 h\*; k = 145: 10 h\* (`gamma_k145.json`). "Validation within" counts h\* whose validation C̄ on N independent panels lies within the tolerance of 0.15.

The ratio-of-sums estimator is heavy-tailed because the common panel-mean term dominates some panels. With 100 panels the
achieved C misses 0.15 with an SD of 0.017–0.022 and by up to 0.049, so ±0.005 was unattainable. At 4,000 panels the SD is
0.0035 (k = 0) and 0.0044 (k = 145), the worst deviation is 0.009, and validation sits within ±0.02 for all 30 h\*; ±0.005
would still fail 7 of 30. F12a.4(h)'s monotonicity check fails for 10 of 20 h\* at k = 0 and 2 of 10 at k = 145 (at 4,000
panels), each with a unique root; those would have counted as power failures. Cost: 1.3 ms per panel at k = 145, ≈ 11 CPU-s per alternative
replicate. In the reduced run all 20 alternatives qualified, with validation C 0.134–0.163 (mean 0.152, SE ≈ 0.004).

## 5. Reduced run after the repairs (registered stage-2 configuration)

`f12_simulation.py --synthetic --config docs/prereg/f12_manifests/f12b_stage2_config.json --k-grid 145 --replicates 20
--scenarios null,alt --workers 2` → `reduced_k145/report.json` (`config_registered: f12b_stage2`, 4,000 draws, tree grid
{100, 300, 600}, seeds 0/1/2). It is not a gate result: 20 replicates at one k.

| k = 145, synthetic tokens | null (γ = 0), n = 20 | alternative (oracle C = 0.15), n = 20 |
|---|---|---|
| C_WM ratio of sums | +0.025 | +0.166 |
| C_WM median (min, max) | +0.040 (−0.043, +0.177) | +0.180 (−0.058, +0.589) |
| C_WM,S ratio of sums / median | +0.025 / +0.024 | +0.025 / +0.033 |
| C_S:F3 ratio of sums (tree vs field) | −0.000 | +0.145 |
| C_WM lower bound > 0: conditional / joint | 7 / 3 | 16 / 7 |
| C_WM,S lower bound > 0: conditional / joint | 5 / 2 | 5 / 4 |
| "correction detected" | 1 | 4 |
| valid replicates | 20 | 20 |
| refit epochs, median | 22 | — |
| CPU-s per replicate (mean) | 202 | 219 |

**Reading.** WM-R no longer loses to F3 by construction: the null C_WM moved from −5.8…−34 to a median of +0.04. The
alternative now produces positive C_WM (16 of 20 > 0). Power is far from 0.80 here because StatNN learns h\* almost as well as
WM-R (C_WM,S ≈ 0.025). That is a property of these synthetic tokens and h\*, not a gate verdict.

**The comparator absorbs most of the synthetic signal.** Under the alternative, mean C_WM is 0.224 (ratio 0.166) but mean
C_WM,S is only 0.057 (ratio 0.025), because the tree on the same 42-column tokens learns h\* too (C_S:F3 ratio 0.145). The
oracle is calibrated to C = 0.15 **against F3 only**, while "correction detected" also requires C_WM,S ≥ 0.05 with both lower
bounds > 0. The conjunction's power is therefore governed by the WM-R-over-tree margin, which this alternative does not
control: C_WM,S ≥ 0.05 held in 4–5 of 20 replicates, and all 4 detections came from those replicates.
**Recommendation for Board A (wording; not adopted as a design change here).** Keep the intersection–union conjunction (it is
the scientific claim). State explicitly in F12.5/F12a.5 that the registered alternative is a token-function alternative
calibrated against F3, in which the same-input tree is a full competitor. Its "ordered power" is power to beat **both**
comparators when the tree can also learn the signal, so it is not the power to detect a K+F-specific correction. A failed
power gate then reads "not powered against a same-input tree", not "no correction". If Board A wants power for a
WM-specific correction, a second alternative has to be defined with its effect calibrated on C_WM,S, or h\* restricted to
structure the tree cannot represent. That needs StatNN fits inside the calibration (or a fixed, pre-registered surrogate) and
a new costing; F12b does not register it.

**Open item for Board A (not repaired by F12b).** The null C_WM is slightly **positive**, and C_WM's conditional lower bound is
> 0 in 7 of 20 null replicates. Diagnosis on 400 noise-only null panels at k = 145: Σa has SD 0.152, ≈ 45 × the noise SD
(0.0034). Ridge-1's estimation shrinkage (slope 0.995 on the population; ≈ 0.5 % more out of fold) leaves a deterministic misfit.
Replacing ridge-1 by OLS improves on F3 by **C = +0.036**, under Δy = a + ε and equally under Δy = (population ridge fit) + ε. A
training-median location shift does not help (−0.005). WM-R learns part of this ridge bias (+0.025). So γ = 0 is not a
no-correction null for C_WM. The size gate as registered would then measure detection of the ridge-1 bias, and the same
"correction" would exist in the real analysis if real Δy tracks Σa this closely. Options for Board A, none adopted here: a
penalty scaled to n (or λ → 0) for F3; a null DGP built from F3's own out-of-fold operator; or accepting that the size gate
covers this bias. F12b changes neither F3 nor the null.

## 6. Two-stage k selection, cheaper fits and CPU budget (F12b.4–F12b.6)

**Timing** (process CPU per replicate, registered stage-2 configuration, synthetic tokens; `reduced_k145`, `timing_k0_k87`):

| k (rows) | null | alternative |
|---|---|---|
| 0 (123) | 121 s | 136 s |
| 87 (≈ 211) | 190 s | 208 s |
| 145 (270) | 202 s | 219 s |

k = 29, 58 and 116 are interpolated linearly. Before the repairs it was ≈ 570 s wall per replicate. The components: WM-R ≈ 48–60
CPU-s (was ≈ 185); StatNN ≈ 130 CPU-s at k = 145 after `n_jobs` 1 (bit-identical predictions, −22 % CPU) and seed-0 tree-count
selection (4 instead of 12 inner fits of 600 trees per outer fold; one 600-tree fit ≈ 6.5 CPU-s); intervals 0.2 CPU-s per
contrast (so the pilot keeps all 4,000 draws: the permitted reduction would save < 0.5 %); calibration ≈ 11 CPU-s. Rejected:
dropping training-constant columns (identical predictions, no saving), `max_bin` 63/31 (≤ 20 % saving, different trees) and
column-wise histograms (6 % saving).

**Budget.**

| item | replicates | CPU-h |
|---|---|---|
| F12a.5(f) timing pilot (10 alternatives, k = 145) | 10 | 0.6 |
| stage 1: 40 null + 40 alternative at each of six k | 480 | 23.8 |
| **stage 2 (registered), worst case k\* = 145**: 1,000 null + 1,000 alternative + 200 variance-×2 | 2,200 | **128.2** |
| stage 2 if k\* = 87 / k\* = 0 | 2,200 | 121.1 / 78.1 |
| total, worst case | 2,690 | 152.6 |

The registered run (stage 2) stays ≤ 150 CPU-h at every k. Wall time with 4 single-thread workers is ≈ 32 h for the worst-case
stage 2. Peak RSS is ≈ 0.9 GB per worker plus a 0.7 GB parent. Real tokens may cost differently: the timing pilot re-measures
before stage 1, and a stage-2 projection above 150 CPU-h stops the run for prospective repricing.

**Stage-1 rule** (`select_stage2_k`, `test_stage2_k_selection_rule`): k\* is the smallest k with pilot power ≥ 36/40 and every
size rate ≤ 2/40; stage 2 runs at k\* only, or at 145 if no k qualifies. With 40 replicates the pilot's power SE is ≈ 0.05 at
p = 0.9. The ≥ 0.90 threshold aims at k values whose true power clears the n = 1,000 requirement (observed ≥ 0.838). Stage 1 uses
replicate indices 1,000–1,039, so its h\*, γ and noise are disjoint from stage 2's.

## 7. Conformance fixes (no design change)

The registered configurations also bring the code into line with F12a where `07e1dc2` departed from it:
- the F12a.4(b) noise draw order (`test_noise_draw_order_follows_f12a`, against a literal reference implementation);
- the F12a.5(b)–(d) six gating rates, with Bonferroni 0.05/36 and invalid replicates counted as size rejections
  (`test_f12a_gates_six_rates_bonferroni_36_invalid_counted`: 838/837 and 46/47 at n = 1,000);
- the restricted-payload loader, which refuses `donor_mean` or any other key (`test_restricted_payload_round_trip_and_pin`);
- committed, sha256-pinned configurations (`test_registered_configs_are_committed_and_pinned`; the pilot differs from stage 2
  only in replicates, seed tag, offset and stage);
- binding release only for stage 2 at the selected k (`test_release_is_binding_only_for_the_registered_stage2_run`).

The drafted defaults are kept, so the 14 F12 tests still pass. F12a.4(d)'s 10,000-panel noise check is not implemented yet.

## 8. Reproduction

```
OUT=/home/ubuntu/wsp-data/runs/a5/f12_sim_dryrun/f12b; PAY=$OUT/../noise_payload_restricted.json
python scripts/f12_simulation.py --synthetic --restricted-payload $PAY --out $OUT/reduced_k145 \
  --config docs/prereg/f12_manifests/f12b_stage2_config.json --k-grid 145 --replicates 20 --scenarios null,alt --workers 2
python scripts/f12_simulation.py ... --out $OUT/timing_k0_k87 --k-grid 0,87 --replicates 2 ...
python scripts/f12b_diagnostics.py smoke --name smoke_draws.json --k 145 --replicates 2
python scripts/f12b_diagnostics.py head  --name head.json --k 145 --replicates 6
python scripts/f12b_diagnostics.py gamma --name gamma.json --hstar 20 --cal-counts 100,400,1000,4000,10000 --k-grid 0
python scripts/f12b_diagnostics.py gamma --name gamma_k145.json --hstar 10 --cal-counts 100,400,1000,4000 --k-grid 145
```

Run with `OMP_NUM_THREADS=2` under `agent-tmux`, one job at a time. The k = 0 gamma output was saved as `gamma_k0.json`; the
k = 145 part of that first invocation was stopped and re-run separately with 10 h\*.
