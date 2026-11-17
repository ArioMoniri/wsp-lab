# WM-E confirmatory-stage engineering prerequisites, 17 Nov 2026

> **ENGINEERING ONLY.** Code and tests for `docs/PREREG_WME.md` (DRAFT v0) §3–§9. They are built to the contract
> in Board A's review (`docs/reviews/council/board-a-prereg-wme-2026-11-17.md`, ACCEPT WITH CONDITIONS).
>
> **What did not happen:**
> - No bank, trajectory, enhancement or confirmatory outcome was generated.
> - No sealed test split was opened. Pilot sealed sets were read only as hash lists from manifests.
> - The `wme-c1` panel was **not built**.
> - The full power simulation was **not run**. It waits for Board A approval.
>
> **What did run:** GPU use was four smoke tests on Qwen3.5-0.8B using wme-p2 *construction* items (well under 0.01
> GPU-h). The only other runs were the three CPU jobs below.

- **Branch:** `feat/wme-confirm-eng`, from main at `1913911`. Main was merged in to pick up the Board A review.
- **Commits:** `fa9ccdd`, plus the commit that adds this report.
- **Machine-readable outputs:**
  - `docs/prereg/wme_manifest_draft.json`: the executable manifest draft for the PREREG v1 author.
  - `docs/prereg/wme_bank_recipe_draft.json`: the bank recipe.
  - `docs/reviews/wme/wme-c1-exposure-audit-2026-11-17.json`
  - `docs/reviews/wme/wme-planner-synthetic-2026-11-17.json`
  - `docs/reviews/wme/wme-powersim-pricing-2026-11-17.json`

## BLUF

- **C1 σ fixes are built and tested:**
  - a field-residual σ floor, so σ > 0 even at shrink 0;
  - `StaticPredictor` returns (mean, σ) with the same treatment;
  - selection and member bootstraps are by closure component;
  - split-conformal calibration is fitted on h ≤ 2, with the coverage check at h = 3–4. A miss beyond 10 pp switches **every** arm to ranks only.
- **C2 controls are library functions, rebuilt to Board A conditions 5, 6 and 10:**
  - a *fixed-endpoint* bookkeeping control over consistent decompositions, under which the static prediction is shown to be invariant;
  - a separate, descriptive corruption diagnostic;
  - inert edits;
  - frozen trust-region strata;
  - two-stage best-of-N;
  - full cost accounting.
- **Smoke finding: the "post-norm-cancelled" k_proj × 2 edit is *not* inert in bf16.** On 0.8B it moved sycophancy values by up to 0.11 nats (mean). It is excluded, and replaced by **power-of-two compensated pairs**, which are **bitwise inert on the real 0.8B model**, plus never-produced embedding rows, which are mathematically inert on untied 9B.
- **The exposure audit is hash-only.** After excluding every pilot split and the 27B calibration exposures:
  - **sycophancy has 467 groups left**, against the Board's bound of ≤ 469;
  - **harmful refusal has only 127**, the binding guardrail constraint.

  All proposed sizes are feasible. The panel itself was not built.
- **Bank recipe:** 96 roots and 1,128 planned states.
  - **44 closure components under the registered "source" policy**, and 30 under the conservative "concept" policy. Under "concept", one 460-state component unbalances the folds.
  - Five folds are feasible under both policies.
- **Planner gate (C6), 200 synthetic worlds at B = 32:**

  | Inner optimiser | Mean regret |
  |---|---|
  | sep-CMA-ES | 0.861 |
  | CEM | 0.857 |

  The sep − CEM difference is +0.004, 95% CI [−0.021, 0.029]. **The kill rule applies: the planner uses CEM.**
- **Power simulator is built and priced, not run.**
  - Error-level grid at 1,000 replicates: **0.14 CPU-h**.
  - A fully nested refit pipeline: about **160 s per replicate**, so 90 CPU-h for one cell's null + alternative, or 903 CPU-h for the whole H-T grid.
- **GPU feasibility is the binding risk:**

  | Item | GPU-h | Cap |
  |---|---|---|
  | Bank at the pilot-2 per-state rate | **14.5** | 12 |
  | Bank under the pilot-1 formula | 10.6 | 12 |
  | Bank **with per-state oracle grounding** | **28.4** | 12 |
  | Confirmatory primary, 12 episodes, B = 32 | 10.3 | 10 |
  | Confirmatory, all arms | 40.5 | 10 |

  As designed, the study would be "unresolved — not powered" unless the design or the budget changes.

## 1. Modules and tests

All 262 `wm` tests pass, plus 4 GPU smoke tests. The new tests are listed below. The full suite on the exact committed tree fails only the three pinned tripwires (`test_f12d_prereqs` ×2 and `test_f12e_qualify` ×1); 1,863 passed for `fa9ccdd`.

Tests were written before each module, except `wm.wmclosure`, whose tests (in `test_wm_bank.py`) were written right after its first draft.

| Module | Content | Tests |
|---|---|---|
| `wm.uncertainty` (new) and `wm.model` (C1) | σ floor σ = gate(E_t)·√(σ_F² + (λσ_ens)²); component selection split and bootstraps; `ConformalScaler` (finite-sample quantile); `calibrate` (h ≤ 2 only); `check_coverage` (h = 3–4); `CalibratedPredictor` (raises `RanksOnlyError`) | 15 |
| `wm.model` (conditions 3 and 4) | `MatchedStatic`, the static comparator matched to WM (see below); `aux_ablation`; `AnchorShift`, the planning comparator; `stage_audit` (every learned stage ≤ h 2); `refit_f3u` (refuses h > 2) | 7 |
| `wm.controls` (C2; conditions 5, 6 and 10) | `consistent_decompositions` and `fixed_endpoint_control`; `corruption_diagnostic` (descriptive); `dead_vocab_rows`, `compensated_pairs`, `post_norm_cancelled_scales` (audit only) and `verify_inert`; `inert_check` (2σ **and** an absolute tolerance, coverage first); `assign_stratum` and `stratum_report`; `two_stage_best_of_n`; `cost_report` | 25 |
| `wm.confirm_stats` (new) | C_T and its component-cluster bootstrap-t lower bound; paired lower bounds; Bonferroni guards; Holm; Clopper–Pearson | 5 |
| `wm.plan` (condition 10) | `BUNDLE_DEFINITION` and bundle kinds; fractional screen and oracle units; cache at 0 units; `close` and `verify_parity`. `report()` now separates `budget_identical` from `equal_usage` | 4 new (12 in total) |
| `wm.exposure` (new; condition 1) | Split roles; question grouping; near-duplicate policy; manifest-only hash reader (O_NOFOLLOW); exposure registry; group-level exclusion; intersection audit; feasibility | 7 |
| `wm.panel_c1` (new; item 3) | The `wme-c1` builder: salt `wme-c1-panel-20261117`, near-duplicate groups, group exclusion, four disjoint splits, an in-memory audit, refusal to overwrite. `SealedTestReader`: a single read with frozen winners, a manifest hash check and a symlink refusal | 9 |
| `wm.bank` and `wm.wmclosure` (new; item 4; conditions 2 and 12) | Six families × 16 roots, frozen nominals, caps, duplicate collapse, side order, stop rule, horizon rule; 24 AB/BA four-step bundles; WM-specific identities, union-find, folds by signature, inner allocation, direction-similarity audit | 15 |
| `wm.planner` (new; C6) | sep-CMA-ES (Ros & Hansen) and CEM inner optimisers; R0 beam; Thompson slots; LCB guardrails; support penalty; symmetric `penalty_mode`; `run_episode` on the ledger; synthetic regret harness | 9 |
| `wm.harness` (new; H4 and H5) | `worst_batches` and `preflight` (27 GB device cap); `unit_key`, atomic `UnitStore`, `run_units`; `ResidentServer` (apply → evaluate → restore, raising `ThetaDrift`); **a real SIGKILL/resume test** | 7 |
| `wm.genaudit` (new; C2 and condition 7) | The frozen `SPEC` (greedy, non-thinking, token limits); loop, refusal and truncation detectors; answer parsers; paired audit with gain lower bound, neutral preservation and anomaly bounds; `generate_greedy` | 6 CPU + 1 GPU |
| `wm.powersim` (new; item 7; condition 11) | Pilot-2 inputs; the bank component graph; H-T, E1 and E2 simulators; grid; scenarios; Monte Carlo intervals; requirement check; pricing; nested pricing; GPU repricing; `run_full` gated | 11 |
| GPU smoke (`test_wm_confirm_gpu.py`) | Compensated pairs bitwise inert on 0.8B; dead rows refused on the tied 0.8B embedding; k_proj × 2 measurably not inert; H4 preflight on real batches | 3 |

Two existing assertions changed, each because a condition required it:
- `test_wm_model`: the static predictor's zero edit now returns (mean, σ) (C1.2).
- `test_wm_plan`: `identical_budget` became `budget_identical` + `equal_usage` (Board A condition 10).

Scripts:
- `wme_c1_exposure_audit.py`: was run; writes counts and hashes only.
- `wme_c1_build_panel.py`: **refuses** unless `WME_C1_FREEZE` is set.
- `wme_bank_recipe.py`: was run; emits the recipe.
- `wme_planner_synthetic.py`: was run.
- `wme_powersim.py`: `price` was run; `run` is gated.
- `wme_manifest_draft.py`: was run.

### Details of the matched static comparator (condition 4)

`MatchedStatic` is WM-E's own network with the recurrence removed:
- h_t = h_0 + gate · step(h_0, xs_t, E_t);
- y_t = F(E_t) + gate · readout(h_0, h_t, xs_t, E_t).

The input slots have the same widths, so its **parameter count is identical**. It also shares:
- the base observation, normalisation, labels and auxiliary observation/slope targets;
- the shrink grid, seeds and component bootstraps.

Its prediction at T is a function of the realised final edit only. This is tested over consistent decompositions to within 1e-6, the float32 batch-shape tolerance; the measured deviation is about 1e-10.

## 2. Board A conditions → where they are addressed

| # | Condition | Code / artefact | Status |
|---|---|---|---|
| 1 | Exposure contract | `wm.exposure`, `wm.panel_c1`, the exposure audit JSON | Built. Audit run (hash-only). Sizes proposed |
| 2 | WM closure mapping | `wm.wmclosure`; recipe identities; component graph and folds in `wme_bank_recipe_draft.json` | Built. Graph published before outcomes |
| 3 | h ≤ 2 for every learned stage | `stage_audit`, `refit_f3u`, `uncertainty.calibrate`; the horizon rule in `wm.bank` | Built |
| 4 | Matched static comparator, auxiliary ablation, anchor-shift | `MatchedStatic`, `aux_ablation`, `AnchorShift` | Built |
| 5 | Fixed-endpoint bookkeeping control | `fixed_endpoint_control`; `corruption_diagnostic` kept separate | Built. The pass rule is a proposal for v1 |
| 6 | Inert edits and strata | `INERT_SPEC`, `verify_inert`, `inert_check(abs_tol, coverage_passed)`, `stratum_report` | Built. k_proj excluded on smoke evidence |
| 7 | E1 selection and generation audit | `wm.genaudit.SPEC` and `audit` (lower bound, neutral preservation, anomaly upper bound); `run_episode` winner rule | Generation built. The E1-versus-E2 winner relation is registration text |
| 8 | Multiplicity | `confirm_stats.holm`, Bonferroni guards, the E2 secondary-family FWER in the simulator | Tools only. The family definitions are registration text |
| 9 | Guardrail scales | Unchanged `panel2.pp_value` and `p2design.GUARDS`, listed in the manifest | Justifying the margins is registration text |
| 10 | Bundle, parity, penalties, costs | `plan.BUNDLE_DEFINITION` and `verify_parity`, `planner.penalty_mode`, `controls.cost_report` (10 categories) | Built |
| 11 | Power simulator scope | `wm.powersim` and its pricing | Built and priced. Gaps in §6 |
| 12 | Six-family nominals and the ×8 freeze | `bank.NOMINALS`, `effective`, `ladder_sides`, `walk_side`, `SIDE_ORDER`, `COMPOSITION_DOSE` | Built (proposed values) |
| 13 | 27B separation | Only the exposure registry entry `calib27-20261116` | Not built (see §6) |
| 14 | Null-to-success language | n/a (registration text) | n/a |

## 3. Results of the runs

### 3.1 Exposure audit (CPU, 17 s, 1.8 GB)

**Method.**
- Exposures were read from manifest hash lists: pilot 1 (1,440 hashes including 288 sealed) and pilot 2 (5,232 hashes including 624 sealed; its unread sealed test is held as a reserve set).
- The 27B calibration read pilot-1 construction, validation and dev items, so it is covered by the pilot-1 manifest and registered separately.
- Near-duplicates: word 5-gram Jaccard ≥ 0.7, via MinHash LSH with fixed seeds. They are grouped before exclusion, so a near-duplicate of an exposed item is excluded with it.

**Results.**

| Objective | Pool items | Groups | Excluded groups | Available | Proposed C/V/D/T | Slack |
|---|---|---|---|---|---|---|
| sycophancy_real | 1813 | 1809 | 1341 | **467** | 64/96/128/160 | 19 |
| refusal_margin_harmful | 520 | 509 | 382 | **127** | 20/40/32/32 | 3 |
| wikitext_nll | 714 | 714 | 411 | 303 | 64/64/64/48 | 63 |
| gsm8k_gold_lp | 8792 | 8791 | 1552 | 7239 | 1024/64/128/128 | 5895 |
| mathcomp_gold_lp | 6269 | 6226 | 1343 | 4883 | 1024/64/128/128 | 3539 |
| mmlupro / refusal_harmless / syc_gap | ≥ 6051 | | | ≥ 5415 | 64/64/64/48 | large |

Consequences:
- The 64-item sycophancy construction split cannot support sycophancy LoRA roots. The bank's LoRA roots train on GSM8K and competition math instead.
- The harmful-refusal guardrail has 32 test items. That affects guardrail power (§5).

### 3.2 Bank recipe (not realised)

- **Recipe hash:** `39b3c100…`.
- **Pilot disjointness:**
  - no pilot scale tensor, contrast pair or seed namespace is reused;
  - one concept overlap is declared: the LoRA objectives GSM8K and math were also pilot-2 LoRA objectives (on different items and layers 12–15 only). A build-time direction-similarity audit (|cos| < 0.5) is required.
- **Planned states:** 1,056 ladder + 72 composition = 1,128.
- **Components:**

  | Policy | Components | Largest component (states) |
  |---|---|---|
  | "source" (registered) | 44 | 74 |
  | "concept" (sensitivity) | 30 | 460 |

  Multi-family composition components each have a unique signature, so most strata have fewer than five components; `fold_report` flags them.
- **Horizon:** steps on the registered trajectory, so ±1 is h = 3 and ±8 is h = 6. Training uses h ≤ 2.
- **Proposed nominals for the three new families:**
  - row_scale f = 1.5 on 32 rows;
  - noise ρ = 0.02;
  - LoRA λ = 1.
- **Caps:**
  - block and row multipliers m(s) = max(0, 1 + s(f − 1)), so attenuation sides collapse after s = −2;
  - projection α ≤ 1 on the positive side, so +4 and +8 duplicate +2.

### 3.3 Planner synthetic gate (C6; CPU, 4 processes, 416 s, 0.46 CPU-h)

The world is synthetic: a saturating gain, a quadratic damage guardrail, and a biased five-member ensemble surrogate with online kernel correction. Budget B = 32, Thompson batches of 4, beam 3, 48 R0 evaluations per node. 200 worlds.

| Method | Mean regret |
|---|---|
| sep-CMA-ES | 0.861 |
| CEM | 0.857 |
| Random | 0.892 |

- The paired sep − CEM difference is +0.0043, with 95% bootstrap CI [−0.021, 0.029]; sep-CMA-ES wins 95 worlds and CEM 105.
- **Decision (r9 C6 kill rule): sep-CMA-ES is not better than CEM, so the shared planner uses CEM.** This is frozen in the manifest before any real search.

### 3.4 Power simulator: pricing only (not run)

**Inputs** (pilot-2 design quantities):
- paired validation SE per objective, e.g. sycophancy 0.0668;
- dev gain SE 0.0169;
- the between-candidate SD under a **frozen robust estimator**, 1.4826 × MAD over all 32 candidates = 0.117. The raw SD of 0.819 is dominated by three LoRA candidates and is not used;
- guard SEs on the pp scale;
- bank component sizes at h = 3–4.

**Pricing (CPU pilot):** 20 replicates per priced cell, 2,000 bootstrap draws.

| Claim | Cells × scenarios | s / replicate | CPU-h at 1,000 replicates |
|---|---|---|---|
| H-T | 20 | 0.0186 | 0.103 |
| H-E1 | 144 | 0.00013 | 0.005 |
| H-E2 | 270 | 0.00038 | 0.029 |
| **Total (error-level)** | 434 | | **0.14** |
| Nested refits (WM 16.8 s + MatchedStatic 13.6 s + StatNN 1.7 s per fold, 5 folds) | per cell × scenario | 160 | 45 per cell; **90** for one cell's null + alternative; 903 for the H-T grid |

The fit times are single-threaded at synthetic bank scale; the real bank has more observation dimensions.

**GPU repricing (condition 11)** uses validation 520 items and dev 672 items per bundle:

| Item | GPU-h |
|---|---|
| Bank, 1,128 states at the pilot-2 T2 rate (46.4 s per state) | 14.5 |
| Bank, pilot-1 formula | 10.6 |
| Per-state oracle gradients for slope grounding (44.2 s per state) | +13.8 → **28.4** |
| Confirmatory primary (WM and StatNN) at 12 episodes and B = 32, with test scoring and generation | 10.3 |
| Same with all 11 arms | 40.5 |
| 6 episodes at B = 16, primary | 3.5 (all arms 11.1) |

The stop rule can only reduce bank states, and it is unknown in advance. As written, the 12 + 10 GPU-h envelope does not hold with slope grounding retained.

## 4. Smoke findings (0.8B, construction items)

- **Compensated pairs are bitwise inert on the real model.** MLP up × 2 with down × 0.5 at layers 12 and 15, and attention v × 2 with o × 0.5 on the last full-attention layer, gave **bitwise-equal** teacher-forced values. Restore was bitwise.
- **k_proj × 2 (k_norm-cancelled) is not inert:**

  | Layer | sycophancy_real (mean change, nats) | Max single-item change |
  |---|---|---|
  | 3 | −0.108 | 0.285 |
  | 7 | +0.043 | 0.172 |
  | 23 | −0.0004 | 0.004 |

  WikiText moved by ≤ 0.0021 (mean) at every layer.

  Two problems follow. First, the measured change is larger than the 9B sycophancy paired SE (0.067) at most layers, so the edit cannot serve as an inert control. Second, k_proj is outside the model's input representation, which would make the ghost test vacuous. It is kept only as an audit generator.
- **The dead-row generator refuses the tied 0.8B embedding, as designed.** The 9B embedding is untied, and rows 248077–248319 are never produced by the tokenizer.
- **The H4 preflight ran each objective's worst batch on the real model** under the 27 GB device cap.

## 5. Implementation choices the PREREG v1 author or Board A must ratify

1. **Fixed-endpoint pass rule:** the static prediction is invariant, **and** LB(C_T on consistent alternates) > 0. Consistent alternates are reorderings and step splits, kept only when the realised final edit is identical.
2. **Inert controls:**
   - dead embedding rows: 3 × 32 rows;
   - compensated pairs: MLP layers 12–15 and attention layer 15;
   - absolute tolerance |ŷ| ≤ 0.5 × paired SE, together with the 2σ rule;
   - the coverage check must pass first.
3. **Two-stage best-of-N:**
   - a screen costs screen_items / full_items of a bundle;
   - survivors need screen gain > 0 and ΔNLL ≤ 0.01;
   - up to 16 full evaluations.
4. **Bundle definition** (`plan.BUNDLE_DEFINITION`): repeats served from the cache cost 0 units; generation is outside B.
5. **Generation audit** (`genaudit.SPEC`):
   - greedy decoding, departing from r8's replicated sampling (justified as a paired deterministic comparison);
   - max new tokens 96 for sycophancy, 512 for GSM8K and math;
   - anomalies must meet incidence ≤ 2% **and** a paired-increase upper bound ≤ 2%;
   - neutral non-inferiority at −0.02.
6. **Six-family nominals and caps** (§3.2), and the horizon rule.
7. **Closure policy:** "source" is registered and "concept" is a sensitivity analysis. Under "concept", the 460-state component would dominate one fold.
8. **Proposed panel sizes** (§3.1).
9. **Robust E2 SD estimator** (§3.4). The alternative "0.5 × SD" becomes 0.058 nats.

## 6. Left unbuilt or partial

**Power simulator: what it does not model.**
- It works at the error and decision level. It models:
  - the static selection between StatNN and F3u on inner folds;
  - the fixed-endpoint and inert gates;
  - Bonferroni guards;
  - generation, neutral and anomaly rules;
  - the Holm secondary family;
  - the symmetric ranks fallback;
  - episode versus family clustering.
- It does **not** refit WM, StatNN or the conformal scaler per replicate. Coverage failure is the probability `p_ranks`. The nested variant is priced (§3.4), not implemented.
- The **weak trust-region strata** scenario was removed because its DGP was not implemented.
- Search reach (b0 = 12), generation transfer dp, base rates and 1.5 s per generated answer are **assumptions**.

**Real-model integration.**
- Not written:
  - the adapter from WM-E or StatNN trajectories to the planner interface (`draw`/`predict`/`embed`/`update`);
  - the bank build and confirmatory run scripts on 9B;
  - E2 episode-root definitions;
  - the Fugu-style reader, LoRA-SFT and steering best-of-N competitor arms.
- The harness pieces (preflight, units, resident server) are ready for them.
- **Field support:** row_scale roots (lm_head and embed rows) lie outside the layers 12–15 field support. The field collector must add vocabulary-row factors, or those roots have no field nest.

**27B (condition 13).** Only the exposure registry entry exists. The `wme-c27` panel, the `wme-bank27` namespace and the 27B gates are not built.

**Audits and fresh-data checks not run.**
- The direction-similarity audit needs realised directions; it runs at bank build.
- `verify_inert` on 9B must run on construction items before the inert controls are used.

## 7. Resources and reproduction

| Job | Wall | CPU-h | Peak RSS | GPU |
|---|---|---|---|---|
| Exposure audit | 17 s | < 0.01 | 1.8 GB | none |
| Planner gate | 416 s | 0.46 | < 0.1 GB per process | none |
| Pricing, including nested fit timing | 40 s | 0.01 | 0.94 GB | none |
| GPU smoke tests (0.8B) | about 25 s per run | — | — | < 0.01 GPU-h |

- Full test-suite verification ran twice, at about 5 minutes each.
- Every job stayed under 4 GB RSS. Long jobs ran under `agent-tmux`.

**Commands:**

```
python scripts/wme_c1_exposure_audit.py
python scripts/wme_bank_recipe.py
python scripts/wme_planner_synthetic.py --n-worlds 200 --procs 4
python scripts/wme_powersim.py price
python scripts/wme_manifest_draft.py
```

The manifest draft `docs/prereg/wme_manifest_draft.json` collects every constant from module code and carries the sha256 of each referenced artefact and source file. It is not frozen.
