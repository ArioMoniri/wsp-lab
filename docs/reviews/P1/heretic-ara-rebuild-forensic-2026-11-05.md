# Board A: Heretic ARA rebuild forensic — 5 November 2026

**Verdict: REVISE (blocking for production/Heretic-quality comparisons).** These are real, finite, exported extreme edits with measured large first-token KL, not a recurrence of the old adapter-export failure. They do not establish a capability-preserving Heretic treatment. Recommend a corrected, explicitly constrained rebuild; retain the present files only as a diagnostic stratum flagged `ara_unconstrained_surrogate_scale`, excluded from Heretic-quality comparisons and default production training panels until explicitly curated.

## Scope and evidence

Read-only ML review of all 12 IDs in `/tmp/ara_new_ids.json`, all four requested files per ID, the canonical manifest, generator/config, D6 and both addenda, and installed Heretic 2.0.0 source. The requested earlier forensic is actually at `reviews/P1/heretic-ara-densify-check-2026-11-04.md`, not under `docs/`; also read `reviews/P1/fix-heretic-ara-export-boardA-2026-11-04.md`. This report and session note are the only intended tracked edits; direct commit on main is explicitly requested. No model execution, training, GPU allocation, or remote rental was performed.

Audit run: `data/runs/ara-rebuild-forensic-20261105/` contains `audit.py`, `audit.json` (all per-module measurements), and `manifest.json` with source and input SHA256s, base revision and review snapshot. CPU only, one Torch/BLAS thread, CUDA hidden. For each safetensor, read Δ as float64 and compute sqrt(sum of squared entries), per-module norm, finite check, cosine to base, and row norms of bf16-base + Δ. Leading-direction energy is a **30-step deterministic power-iteration estimate**, not an exact SVD or rank certificate. No confidence interval applies to this exhaustive finite-file audit; it makes no population estimate.

All 12 studies have status `passed`, reload difference 0.0 and verified keyword refusals 0.0. All saved delta tensors are finite. The 12 selected records exactly match `min(trials, key=(refusals, kl_divergence))`. Recomputed norms agree with edit and generator descriptors to displayed precision. Note that scalar KL agreement verifies the stated reload metric, not equality of all logits or a proof of equal weights on every input.

| Zoo ID | Seed / selected trial (1-based) | Verified KL (nats) | Saved Δ Frobenius norm | Lower-KL alternative: trial / refusals / KL |
| --- | ---: | ---: | ---: | --- |
| `bbf6af9653df23c2` | 0 / 8 | 10.270516 | 1.94342e+09 | 1 / 0.06 / 0.3633 |
| `770d16bb686c8226` | 1 / 3 | 12.710419 | 623.327 | 5 / 0.02 / 0.2861 |
| `39b0bc05af270314` | 2 / 4 | 14.042039 | 4.67346e+08 | 7 / 0.03 / 0.5708 |
| `cb43afcdcf968a92` | 3 / 7 | 12.366482 | 5.67455e+08 | 1 / 0.06 / 0.4395 |
| `eeda0ec4d3dc118b` | 4 / 6 | 4.939729 | 30.1021 | 8 / 0.03 / 0.5462 |
| `2b3c19e6158e0788` | 5 / 1 | 12.818939 | 399.904 | 3 / 0.05 / 0.3223 |
| `0b5a6ef591cee209` | 6 / 5 | 2.216703 | 42.3755 | 8 / 0.04 / 0.2182 |
| `5c1c957dab65753f` | 7 / 1 | 4.192908 | 35.3165 | 8 / 0.03 / 0.1529 |
| `d9da6daecd7c8d6d` | 8 / 3 | 0.765164 | 19.3279 | 10 / 0.07 / 0.4269 |
| `cec362b7c5eaf0f1` | 9 / 10 | 13.736275 | 1.34536e+10 | 4 / 0.06 / 0.3653 |
| `cbc856f1b3ed9f7b` | 10 / 9 | 13.050547 | 1184.91 | 6 / 0.04 / 0.4981 |
| `69c7390859177261` | 11 / 10 | 9.920191 | 1.70019e+10 | 7 / 0.02 / 0.5352 |

Alternatives shown minimize (refusals, KL) among recorded trials with KL ≤ 0.6; these are **trial scores**, not newly verified exports. Every study also has at least two recorded candidates at KL ≤ 0.5. The alternatives are evidence of a selection tradeoff, not evidence that their saved weights or capabilities have passed evaluation.

## B1 — No KL target was used; final selection prioritizes refusals (blocking)

`configs/zoo_qwen3.5-0.8b.yaml:48–56` specifies n=12, preservation weights {0.5,1,2,4}, trials=10, and **no `kl_divergence_target`**. All 12 `study.json.params` likewise omit it. `src/wsp/zoo/heretic_ara.py:138` reads:

```python
kl_target = params.get("kl_divergence_target")
```

Thus the actual target is **None / unconstrained**, not 0.1, 1, 10, or the preservation weight. A preservation weight is a module-output MSE coefficient, not a bound in nats. The installed Heretic 2.0.0 scorer/config has no `kl_divergence_target` setting at all (searched the installed package): `heretic/config.py:374–383` defaults to two minimization objectives, keyword rate and KL. `heretic/scorers/kl_divergence.py:13–21` supplies prompts only. No upstream default KL ceiling is inherited. WSP's smoke YAML's target 1.0 is unrelated to this production config.

The objective returned to Optuna is the pair of evaluator scores (`heretic_ara.py:175–189`, `197–200`; installed `heretic/evaluator.py:211–230`). However **final export does not use `study.best_trials` or select a balanced Pareto point**. It calls the wrapper's selector at `heretic_ara.py:205`; absent a target, `:51–54` says:

```python
return min(
    trials,
    key=lambda t: (t.get("refusals", float("inf")), t.get("kl_divergence", float("inf"))),
)
```

This is lexicographic refusals-first selection: any 0.00 beats 0.01 or 0.20 regardless of KL. It then chooses the smallest KL among equally low refusal rates. Such a point is itself nondominated; merely saying “use Pareto” would not fix the problem. A hard feasibility bound and a declared choice from the feasible frontier are required.

Concrete reproduction: `bbf6af9653df23c2` selects trial 8, refusals 0.00/KL 10.270283, over trial 1 (0.06/0.363314) and trial 10 (0.20/0.027762). Its re-applied merged model verifies at KL 10.270516. This is exactly the implemented ordering, not a scorer-label swap.

At `heretic_ara.py:225`, the metadata test is:

```python
"kl_constraint_met": kl_target is None or verified_kl <= kl_target,
```

**There is no wrong numeric threshold: the left branch is vacuously true.** The subsequent rejection gate (`:265–266`) therefore accepts all finite verified KLs when no target exists. The boolean is misleading as a quality label. Record explicit effective target, whether a constraint was enabled, and selection rule; use null/“not applied” for the satisfaction status when disabled. For production ARA, require an explicit finite positive target rather than silently dropping the constraint. Failure scenario: downstream reads `passed` + `kl_constraint_met: true` + `production` as low drift, accepting KL 10–14.

## B2 — Huge norms reflect scale growth under a mismatched surrogate (blocking for intended ARA semantics)

Saved Δ is a dense fp32 **effective deployed difference**, not A or B. The exporter takes bf16 merged weights minus the base (`heretic_ara.py:90–109`, `212–217`) and reloads that same file (`249–266`). There is no new α/r multiplier: installed `heretic/model.py:274–277` sets `r=lora_rank`, `lora_alpha=lora_rank`, hence scale=1. Nonquantized merge is plain `merge_and_unload()` (`:363–370`). These findings and the prior densifier's B@A/PEFT comparison rule out a densification orientation/scale error as the explanation.

The five entries above 4e8 (three between 4e8 and 1e10, **two additional above 1e10**) concentrate in a few modules and largely one direction per dominant module. Largest module below; “row ratio” is median row norm of deployed W+Δ divided by base W, and “direction energy” is the estimated top singular-value squared / module Frobenius norm squared.

| ID | Dominant module (under model.language_model.layers) | Module Δ norm | Fraction of total squared norm | Base module norm | Median row ratio | Leading direction energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `bbf6af9653df23c2` | 14.mlp.down_proj.weight | 1.9333e+09 | 98.96% | 15.142 | 8.884e+07 | 99.997% |
| `39b0bc05af270314` | 22.linear_attn.out_proj.weight | 3.24261e+08 | 48.14% | 17.702 | 1.195e+07 | 99.955% |
| `cb43afcdcf968a92` | 17.mlp.down_proj.weight | 4.56567e+08 | 64.74% | 16.387 | 1.864e+07 | 99.982% |
| `cec362b7c5eaf0f1` | 4.mlp.down_proj.weight | 1.2846e+10 | 91.17% | 15.584 | 5.136e+08 | 98.864% |
| `69c7390859177261` | 20.mlp.down_proj.weight | 1.00161e+10 | 34.71% | 17.506 | 4.093e+08 | 94.549% |

For `bbf6af9653df23c2`, layer 14 down_proj alone carries 98.96% of squared delta norm. This is a huge change in a learned low-dimensional direction, not uniform scaling of every weight. The retained dense effective delta does not identify separate A/B factors, optimizer trajectory, or a semantic “refusal direction”; those cannot be recovered from the study parameters alone.

Selected parameters of the five extreme runs (end layer exclusive):

| ID | Layer interval | Preserve-good coefficient | Steer-bad coefficient | Overcorrect coefficient | Neighbors |
| --- | --- | ---: | ---: | ---: | ---: |
| `bbf6af9653df23c2` | 8:20 | 4 | 0.0293586 | 1.22687 | 11 |
| `39b0bc05af270314` | 10:23 | 0.5 | 0.00948303 | 1.10053 | 2 |
| `cb43afcdcf968a92` | 12:21 | 1 | 0.408636 | 1.09948 | 6 |
| `cec362b7c5eaf0f1` | 2:21 | 4 | 0.0651804 | 1.20276 | 1 |
| `69c7390859177261` | 3:22 | 4 | 0.418409 | 1.03041 | 9 |

Mechanism from installed `heretic/modifiers/ara.py`:

- `:71–78`: `preserve_row_magnitudes=True` by default. These studies supply no `ara_settings` override, so the installed default applies (effective upstream settings were not serialized; source/default inference, not a separately logged runtime configuration).
- `:341–351` computes outputs under W+B@A, then, **only inside the loss closure**, applies:

```python
eff_row_norms = LA.vector_norm(W_base + (B @ A), dim=1)
scale = W_row_norms.squeeze(1) / eff_row_norms
new_good_output = new_good_output * scale
new_bad_output = new_bad_output * scale
```

- `:133–159` penalizes good-output MSE plus steering distances, with a **negative** distance from bad outputs multiplied by the overcorrection coefficient. None of these terms directly penalizes raw parameter norm. All five extreme selections have overcorrection >1; this is an association consistent with aggressive steering, not proof of an independently unbounded loss (row normalization bounds the effective output scale).
- `:365–380` optimizes A and B using LBFGS with strong-Wolfe search; settings `:90–107` default to five outer steps, lr=1, max_iter=20, history=10. There is no norm cap, weight decay or saved convergence trace here.

When the low-rank product dominates W, enlarging its scale can leave the **normalized** surrogate output almost unchanged, creating a poorly identified radial direction. Plain PEFT scoring and merged export omit that normalization, so radial growth becomes a real enormous deployed weight change. Observed row ratios up to hundreds of millions in the median are incompatible with a row-magnitude-preserving export. A truly row-norm-preserving W' would obey ‖W'−W‖F ≤ 2‖W‖F for each module; e.g. base norm ≈15 cannot yield a normalized Δ of 1.93e9.

**Attribution:** actual factor-product/optimizer scale pathology enabled by the surrogate/deployment mismatch is strongly supported. A specific LBFGS divergence or line-search failure is **not established** without losses, gradient/parameter norms and step traces. The raw saved norms alone cannot distinguish a failing solver from convergence on a poorly scaled flat surrogate. The exporter reproduces this pathology faithfully; the reload gate does not fix upstream loss semantics.

Nor is this proved “normal ARA behavior at tiny budgets.” The code sets `n_startup_trials=30` (`heretic_ara.py:191–195`), while D6 funds ten. Every trial here is in the startup sampling phase; adaptive TPE never starts. Ten trials can fail to find a good tradeoff, but trial count does not itself multiply Δ by 1e9. Low-KL candidates already exist in every study. Increasing budget alone while retaining this selector and surrogate is not a scientifically justified repair.

Single-step comparison: the live manifest has 205 unique root `abliteration` entries with edit descriptors; 200 are `production`. Their descriptor Δ norms range **0.240868–3.143225**, with production median **1.111158**. All 205, including five non-production/missing-label entries, share that same min/max (lower-middle median 1.142292). These are descriptor/manifest measurements, not a fresh full-tensor audit of 205 files. The smallest rebuilt ARA norm (19.3279) already exceeds the classic maximum by ≈6.15×; 1.7002e10 exceeds it by ≈5.41e9×. Algorithms and touched-layer counts differ, so the ratio is a scale warning, not a matched efficacy comparison.

## Disposition and concrete correction

**Recommendation: rebuild with a corrected explicit KL target and a feasible-Pareto rule at the D6 budget, 12 entries × 10 trials.** Do not silently rebrand these 12 as Heretic-quality. Preserve their artifacts and study records under the diagnostic flag `ara_unconstrained_surrogate_scale`; set their production eligibility aside through the normal curation process. This report does not rewrite metadata or delete data. They are legitimate examples of extreme edits when accurately labelled; they are invalid evidence for a KL-controlled, row-norm-preserving ARA treatment. Their current `production` tag and vacuous satisfaction label are insufficient.

Proposed exploratory replacement protocol (a **new planned criterion**, not a previously preregistered threshold or a claim that KL guarantees capability): explicitly set `kl_divergence_target: 0.5` nats. Reject nonfinite metrics; form the nondominated frontier in (refusals, KL) among trials with KL ≤0.5; select lexicographically by (refusals, KL, trial number) within that feasible frontier. Fail the entry if empty, without relaxing the target. Reapply/merge and require **verified** KL ≤0.5 and reload difference ≤0.01; an infeasible verification fails publication rather than silently falling back. Keep the full frontier and discarded candidates. The new rule eliminates high-KL points even when their keyword refusals are lower. Explicit KL tie-breaking also corrects the constrained branch's current refusal-only `min` (`heretic_ara.py:45–49`).

Before calling that rebuild a row-preserving Heretic treatment, resolve the upstream semantic mismatch: apply the same row normalization to the scored/exported effective weights as to the ARA loss, with a separately versioned kernel and provenance, or deliberately disable it in both optimization and deployment and label that variant. Verify row norms if the former is chosen. Record effective upstream settings/version, final tensor equality on reload, and per-module optimizer traces/norms. Do not retroactively rescale existing deltas and retain their old scores. Independently evaluate capabilities; zero keyword matches do not establish useful non-refusing responses.

Changes needed (locations refer to reviewed source):

1. `configs/zoo_qwen3.5-0.8b.yaml:48–56`: add explicit target, retain 12×10 unless D6 is amended. At this budget describe it as a ten-trial startup search; raising to ≥31 for adaptive TPE is a separate budget decision, not authorized by this review.
2. `src/wsp/zoo/heretic_ara.py:28–54,138,205,220–229`: implement feasible-frontier selection with deterministic tie-breaks, reject missing production target, record effective target/selection/constraint-enabled and full effective settings. Preserve the post-merge target and reload gates (`:263–266`).
3. Installed `heretic/modifiers/ara.py:344–351` versus `heretic/model.py:363–370`: implement the chosen consistent semantics in a pinned upstream patch or repo-controlled wrapper, not an untracked site-packages edit. `heretic_ara.py:212–215` is the integration point before final evaluation/export.
4. Metadata concern, independent of norms: `edit.json.params.delta_format` still says `lora`, whereas `provenance.delta_format` and `generator_edit.json.params.delta_format` say `dense` (all 12). `src/wsp/zoo/build.py:523` unconditionally injects `lora`. Fix format identity before rebuilding; do not reuse the old IDs for a changed semantic kernel/target. `generator_edit.json` also records a dirty source tree, so nominal SHA alone is not complete provenance; preserve its recorded tree.

D6 addendum 2 and the closed spend ledger report **18.8 A100-hours, $37.41** for this 12×10 rebuild. A fresh same-budget search is approximately **$37–40 plus verification/independent capability evaluation and patch-validation overhead**, conditional on similar timing and price; ten trials do not guarantee twelve feasible useful entries. Re-selecting/reapplying existing low-KL parameter records might avoid repeating the search, but is not a free metadata repair: only selected deltas were saved, so each replacement requires optimization, merged evaluation, new artifact identity and reload verification. Changing normalization changes the scored treatment and warrants fresh search. No rebuild is launched here.

## Paper language allowed now

“We generated twelve ten-trial ARA-wrapper edits (seeds 0–11), selecting first for minimum keyword-refusal rate and then for KL without a KL ceiling. All twelve exported deltas reproduce their in-process first-token KL under the reload check; selected KL ranges from 0.765 to 14.042 nats, with zero keyword-refusal rate. Several edits have extreme weight norms. We exclude this unconstrained surrogate-scale stratum from capability-preserving Heretic comparisons.”

Explain that Heretic's KL is baseline-to-edited **first-token** KL over default harmless prompts (`heretic/scorers/kl_divergence.py:14–20,50–64`), not sequence-level KL or a benchmark capability score. Do not describe these runs as optimized Heretic-quality, evidence of intelligence retention, successful harmless/harmful response compliance, or a negative result about Heretic as a method. The source mismatch and ten startup trials limit that inference. Any diagnostic predictor use must be separately declared, with inclusion/exclusion sensitivity analysis, so a norm-separated pathological stratum cannot masquerade as general edit-effect forecasting. Independent held-out capability scores and both-board review remain necessary for paper claims.

## Validation

Full CPU suite: **498 passed, 1 skipped, 8 deselected**, 7 warnings, 61.10 seconds (slightly over the repository's 60-second target while other work was running). Ruff check passed; all 100 src/test files pass format checks. Run via `agent-tmux boardA-ara-checks bash -lc ...`; log: `/home/ubuntu/agent-logs/boardA-ara-checks-20261105-052130.log`. No new tests or implementation changes were needed for this documentation-only review. Artifact audit exhaustively reads all 12 saved deltas and metadata; no fresh GPU forward-pass claim is made.

## Five-line summary
Cause of KL-10 selections: missing target → refusals-first lexicographic selection; `kl_constraint_met` is vacuously true, with no numeric threshold applied.
Cause of huge norms: actual low-dimensional scale growth under row-normalized LBFGS surrogate but unnormalized deployment; precise solver divergence unproven, export scaling error unsupported.
Disposition: REVISE; preserve as `ara_unconstrained_surrogate_scale` diagnostics, exclude from production/Heretic-quality comparisons pending curation and correction.
Config/code required: `configs/zoo_qwen3.5-0.8b.yaml:48`; `src/wsp/zoo/heretic_ara.py:28,138,212,225`; `src/wsp/zoo/build.py:523`; align upstream `ara.py:344` with merge semantics.
Rebuild: needed for a corrected Heretic-quality cohort, 12×10 trials ≈18.8 A100-hours / $37–40 plus validation; no new spend authorized or incurred by this review.
