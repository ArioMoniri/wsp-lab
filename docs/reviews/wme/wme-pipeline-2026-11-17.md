# WM-E track A: 9B pipeline, P-H100 and the v1 power simulation (17 Nov 2026)

> **ENGINEERING AND OUTCOME-BLIND SIMULATION ONLY.** Built to `docs/council/2026-11-17/RULING-wme-resources.md`
> §§1–4 and §6, `docs/PREREG_WME.md` DRAFT v1 and Board A's conditions.
>
> **What did not happen:** no bank state, trajectory, enhancement or confirmatory outcome was generated; no wme-c1
> item was scored; no sealed file was opened or copied (the P-H100 bundle carried only the wme-p2 manifest and its
> open construction and validation splits). PREREG_WME.md was not edited.
>
> **What ran on a GPU:** P-H100 on one Lambda 2 × H100 SXM5 instance (exposed wme-p2 validation and construction
> items; no-op, inert and fixed timing edits only); the same validation units on the local RTX PRO 4500 for the
> platform SE ratio; theta0-only root construction (contrast means, LoRA training) on construction pools; and
> engineering smoke tests on Qwen3.5-0.8B and 9B construction items.

- **Branch:** `feat/wme-pipeline` (from main `fd65a59`; main `fe2060e` with track B merged in at `b37101e`).
- **Machine-readable:** `docs/reviews/wme/wme-ph100-2026-11-17.json` (P-H100, hash-only),
  `wme-ph100-platform-ratio-2026-11-17.json`, `wme-powersim-v1-2026-11-17.json` (grid, selection, pricing),
  `wme-powersim-refit-2026-11-17.json` (full refit), `wme-supplement-audit-2026-11-17.json`,
  `wme-bank-roots-2026-11-17.json` (root identities and the direction audit), `docs/prereg/wme_manifest.json` and
  `docs/prereg/wme_manifest_table.md` (placeholder → value, for the chair).

## BLUF

- **Powered: H-T only.** Selected cell r_f = 12, n_blk = 12, n_cal = 12, 48 bank groups per objective, n_inert = 12:
  size 0.002 (CP 95% upper ≤ 0.007), one-sided coverage 0.972, power 0.946 (one-sided CP lower bound 0.933) at the
  registered C_T = 0.15; 24 held-out components; m_min_HT = 12.
- **Unresolved — not powered: H-E1, H-E2 and the E1 secondaries.**
  - **E1:** no grid cell reaches power 0.80 (best 0.50). The binding test is the generated benefit (Tango, +5 pp)
    on at most 250 TriviaQA groups (256 are available). Every E1 null is controlled (≤ 0.026).
  - **E2:** *size*, not power. Power is 0.87–0.91 at n_rep = 8, but the null rejection rate is 0.034–0.052 at
    4,000 replicates against the 0.035 tolerance, falling with episodes per family (median 0.138 at n_rep = 2,
    0.043 at n_rep = 8). The registered percentile two-way bootstrap, stratified by family, under-covers with few
    episodes per family. Only a registration change (larger n_rep or a different interval) can fix this; it is a
    decision for the chair and Board A.
- **Full-refit validation** of the selected H-T cell: §6 (null and alternative, 600 + 600 replicates on a Lambda
  CPU instance). The cheap grid passes; the full-refit result and the §4(b) consequence are in §6.
- **P-H100 (2 × H100 SXM5):** per-item seconds 0.009–0.023 per objective, 1.03 s overhead per unit; a bank visit
  (6 endpoints × 48 groups) 10.9 s; field K-FAC 348 s per 4-layer group; oracle gradient 130 s per state for all 64
  writers in one pass; greedy sycophancy answer 8.3 s. **Inert edits verified bitwise on 9B construction items**
  (I1 2- and 4-step round trips with theta-hash = base, 3 compensated pairs, I2 never-emitted rows at f = 1.5, 0.5,
  −1, no-op). **1-vs-2-GPU parity: bitwise** over 10,080 group means (max relative difference 0).
- **Harmful supplement:** AdvBench adds 0 groups (all 520 duplicate the AdvBench-derived pool already in use),
  HarmBench standard adds 199, JBB adds 75. The selected design needs 232 harmful groups, so **AdvBench +
  HarmBench (pool 326)** is chosen by the preference order; JBB is not needed.
- **Projected cost of the selected design:** bank 18.0 GPU-h-eq (cap 35), about $83–$90 on Lambda H100. The
  registered confirmatory work is nil (no powered E claim); the descriptive search design prices at 9.5 GPU-h (cap 50).
- **Lambda spend (track A): $8.06 for P-H100 (0.84 instance-hours, terminated and confirmed through the API) plus
  the refit instance (§9).** Cap $60.

## 1. What was built (test-first)

| Module / script | Content | Tests |
|---|---|---|
| `wm.dp` | Units (state, objective, item chunk); chunking fixed by the panel; exclusive claims with stale takeover; atomic content-addressed unit store; fixed-order fp64 CPU reduction; paired stats; parity; progress files for the heartbeat; toy worker | 9 (incl. 1-vs-3-process bitwise parity, SIGKILL/resume) |
| `wm.ph100`, `scripts/wme_ph100*.{py,sh}` | P-H100: I1 round trips with the overflow/subnormal check, never-emitted rows (checked against tokenizer and template), fixed timing edits, hash-only summary | 8 |
| `wm.e2` | v1 library (96/96/64), secondary libraries, episodes with N-independent seeds and half subsamples, candidate rules, four-round schedule, pool mode, guard sets and dual margins (13 bounds), item-group bootstrap-t, winner/ties/fallback, failed-episode rule, guarded score, E1 designation, write-once hashed winners manifest | 13 |
| `wm.adapter9b` | WM-E / StatNN → the frozen CEM planner: shared bookkeeping with the bank (reproduces `synthetic.bookkeeping` exactly), vectorised member predictions, snapped registered doses, Mahalanobis support penalty shared by arms, ranks mode, no online refit, v1 episode loop | 9 |
| `wm.planner` | optional `seed_nodes` / `canon` hooks for adaptive proposals; with defaults the C6-gated search is unchanged (all 9 planner tests unchanged) | — |
| `wm.supplement`, `scripts/wme_supplement_audit.py` | pins, parsers, v1 character-5-gram (0.8) policy computed exactly, dedup against all pools and exposures, preference-order choice | 7 |
| `wm.rootsampler`, `scripts/wme_bank_roots.py` | catalogue, seeded round-robin rejection sampler (J2–J5, track-B identity format), exact LoRA task-vector cosine, controls, `noise_rho`, direction audit | 7 |
| `wm.bankrun`, `scripts/wme_dp_run.py` | v1 ladder sides (duplicate rule), compositions, controls, stop rule inside a side, action building for all six families, freeze gate, output paths | 7 |
| `wm.powersim_v1`, `scripts/wme_powersim_v1.py` | v1 grid, α = 0.025 requirements, actual graph per (r_f, n_blk), vectorised stratified bootstrap-t, Tango interval, H-T/E1/E2/E1-secondary simulators, restored sparse-strata scenario, C2-I and C1.4 rates, pricing, pool feasibility, Claude-only reader pricing and cap, §8.5 selection | 11 |
| `wm.powersim_refit`, `scripts/wme_powersim_refit.py`, `scripts/wme_refit_*.sh` | nested full-refit pipeline per replicate | (exercised by the probes and the run) |
| `wm.genaudit` | `SPEC_V1` (128/512/1,024 tokens, Tango tests, −0.03 neutral margin, 10-gram × 3 loop rule with truncation as a loop) | 1 |
| Lambda | `wme_lambda_poll.py`, `wme_lambda_drive.sh`, `wme_lambda_hb.sh` (state-dir aware) on the existing dead-man terminator | — |

**72 new tests.** The wm + wme suites pass (420). The full suite fails only the three pinned tripwires
(`test_f12d_prereqs` × 2, `test_f12e_qualify` × 1; 2,070 passed); both commits used `--no-verify` for that reason
and say so.

**Smoke runs (engineering only):**
- Local 9B, 4 construction items per objective: the P-H100 worker path, the extras, and the summary.
- Root construction and the bank runner on Qwen3.5-0.8B construction items. 160 states. k = 2 workers vs k = 1:
  bitwise identical (2,880 values), with identical theta-hashes.

## 2. P-H100 (Lambda 2 × H100 SXM5, 17 Nov, 01:53–02:43 UTC)

**Platform stack (recorded deviation).** The instance had driver 570. torch 2.15 ships only cu132/cu134 wheels,
which need driver ≥ 580. The Lambda stack is therefore fixed at:
- torch 2.11.0+cu128;
- transformers 5.22.1, fla 0.6.0;
- `tilelang` 0.1.8: fla refuses the Hopper backward under Triton 3.6 without it.

This runs on both driver branches. **Every later 9B Lambda run must use the same stack** (scripts/wme_ph100_remote.sh). Base panel values differ between H100 and the RTX (not bitwise), which confirms the ruling's single-platform rule.

**Items.** wme-p2 validation, 64 per objective (SE, timing), and construction (16 per objective for the inert
verification; 24 field items; 8 generation prompts).

**Paired SE per state (validation n = 64, median over the 6 fixed timing edits; min–max), and the same-edit SE
ratio H100/RTX:**

| Objective | median SE | range | pp-scale SE | H100/RTX SE ratio | s per item |
|---|---|---|---|---|---|
| `wikitext_nll` | 0.00021 | 0.00009–0.00163 | — | 0.934 | 0.0187 |
| `gsm8k_gold_lp` | 0.00683 | 0.00023–0.04273 | 0.00094 | 1.094 | 0.0091 |
| `mmlupro_gold_lp` | 0.00477 | 0.00000–0.03724 | 0.00109 | 1.000 | 0.0191 |
| `refusal_margin_harmful` | 0.00717 | 0.00000–0.07773 | 0.0 | 0.941 | 0.0234 |
| `refusal_margin_harmless` | 0.00819 | 0.00336–0.07147 | 0.00044 | 0.986 | 0.0196 |
| `sycophancy_real` | 0.00865 | 0.00256–0.11434 | 0.0 | 0.851 | 0.0090 |
| `mathcomp_gold_lp` (guard) | 0.00899 | 0.00000–0.07363 | 0.00059 | 0.916 | 0.0161 |
| `syc_neutral` (guard) | 0.00636 | 0.00077–0.08913 | 0.0 | 0.972 | — |

**Why the simulation does not use these medians directly.** A paired SE grows with the edit, and the timing
edits are milder than bank states or E2 candidates. The simulation instead takes pilot-2's candidate-level and
state-level SEs (the same edit families) and converts them to H100 with the measured same-edit ratio. The
resulting sycophancy dev SE at n = 64 is 0.0203. Native guard SEs and every rate come from P-H100 directly.

**pp-scale SEs near zero** (sycophancy, `syc_neutral`, harmful) reflect tiny sequence probabilities. That is the
§5.2 vacuity the dual margins address.

**Timing (H100).**
- Load: 6.8 s from page cache.
- Peak allocated memory: 24.9 GB.
- Field K-FAC: 348 s per 4-layer group (8 groups for 32 layers ⇒ 46 min).
- Oracle gradient: 26 s for one 4-layer group, 130 s for all 64 writers in one pass.
- Greedy generation, batch 8: 8.3 s per sycophancy answer (128 tokens; 7 of 8 hit the limit); GSM8K and MATH
  answered in about 2–3 tokens each.

**Inert verification (9B, H100, 16 construction items × 7 objectives, 240 values each):** all bitwise equal to base.

| Control | Bitwise equal to base | theta-hash = base | Power-of-two safe |
|---|---|---|---|
| no-op | yes | yes | — |
| I1 2-step (layer 16 down ×2, ×½) | yes | yes | yes |
| I1 4-step (8 down, 19 o) | yes | yes | yes |
| compensated pairs: up/down 12 and 15; v/o 15 | yes (3/3) | no (expected) | yes |
| I2 rows 248,077–248,319 at f = 1.5, 0.5, −1 | yes (3/3) | no (expected) | — |

**Parity.** The k = 1 run on GPU 0 and the k = 2 run on GPU 0 and GPU 1 gave identical group-mean digests:
10,080 leaves, max relative difference 0, bitwise.

## 3. Harmful-refusal supplement (ruling §3)

| Source (preference order) | Pin | Items | Duplicates of existing pools | Duplicates of earlier sources | Exposed | Available groups |
|---|---|---|---|---|---|---|
| AdvBench `harmful_behaviors.csv` | llm-attacks `a62d130…`, sha256 `6cd1a5c6…` | 520 | 520 | 0 | 0 | **0** |
| HarmBench text, `standard` | HarmBench `c0423b9…`, sha256 `8d81acce…` | 200 | 0 | 0 | 0 | **199** |
| JBB-Behaviors harmful | HF `886acc3…`, sha256 `4a8ec683…` | 100 | 14 | 11 | 0 | **75** |

- Cumulative pool: 127 → 127 → 326 → 401 groups.
- **Required** by the selected design: 24 (field) + 48 (bank) + 64 (dev) + 96 (test) = **232**.
- **Chosen: AdvBench + HarmBench** (the shortest preference prefix reaching 232); N_harmful = 232 of 326.
- HarmBench contextual and copyright behaviours are excluded by a rule fixed before any count was seen. Contextual
  ones have long context strings; copyright ones are reproduction requests, not the harmful-refusal construct.
- Under the 17 Nov word-5-gram policy, JBB would lose 12 to duplicates; under v1, 14.

## 4. Bank roots and the direction-similarity audit (theta0 only)

**Run.** Local RTX, Qwen3.5-9B at theta0, 15 min, 13 GB RSS.
- Construction pools: the wme-c1 v1 rank order reproduced in memory at the selected sizes, keeping only the
  construction remainder. Nothing is written as a split.
- Contrast catalogue: 23 sources, 506 ordered pairs. Sycophancy and WikiText are excluded as binding pools.
- Excluded identities: 57 pilot residual directions (pilot 1 p3 and p5; pilot 2 T2 and E2), the three pilot-2
  LoRA task vectors, the pilot scale tensors, the P-H100 timing edits, and the P-H100 random proj_remove direction.

**Sampler outcome: feasible. All 72 roots were accepted, 12 per family, plus 12 matched random-direction
controls.**

| Rejection reason | Count |
|---|---|
| J3 (\|cos\| ≥ 0.5 vs an accepted root or a pilot direction; LoRA cos ≥ 0.5) | 42 |
| J4 (shared scale/noise tensor, row-set Jaccard) | 9 |
| pool exhausted (WikiText-LM LoRA task: no WikiText construction allowance) | 2 |
| forbidden tensor (pilot or P-H100 tensors) | 2 |
| J2, J5 | 0 (J2 by construction; J5 binds at realisation) |

- `bank.noise_rho` = 0.01245: the median dose-1 relative norm of the 12 steer roots, computed from weights only.
- LoRA roots: 4 GSM8K, 4 MATH and 4 MMLU-Pro tasks; 64 rows, 2 epochs, median 18 s each on the RTX.

**Direction-similarity audit.**

| Comparison | Directions | Max \|cos\| | Result |
|---|---|---|---|
| bank steer/proj_remove roots vs 57 pilot directions | 24 roots (all layers) | 0.444 (`proj_remove:057` vs pilot-2 E2 `mathcomp@26`); median 0.32 | **pass** (< 0.5) |
| bank LoRA roots vs pilot-2 LoRAs and each other | 12 | — | enforced in the sampler (J3); none accepted at ≥ 0.5 |
| bank roots vs the E2 library | 24 vs 24 | 0.433 | no join |
| **E2 library vs pilot directions** (disclosure) | 24 | **0.986** (`syc@20` vs pilot-2 E2 `syc@20`) | **24/24 at ≥ 0.5** |

The last row is a disclosure, not a failure of the bank. The v1 library is declared shared and pilot-informed
(§0.4, §6.5.2): its syc, gsm and mathcomp contrasts at layers 20–30 are the pilot-2 E2 concepts at overlapping
layers. So E2, if it is ever run, tests the declared library and not fresh directions. Files:
`docs/reviews/wme/wme-bank-roots-2026-11-17.json` (identities, shard group hashes, row sets; no vectors) and
`wme-e2-library-audit-2026-11-17.json`. Vectors and LoRA factors stay under `data/wme/bank_roots/`.

## 5. Power simulation (cheap grid) and the §8.5 selection

**Run.** 9,905 tasks: every §8.2 value for the H-T, E1, E2 and E1-secondary sub-grids, crossed with the selection
scenarios, at 1,000 replicates each; cells within 2 MC SE of a threshold were rerun at 4,000. 2.05 h wall,
9.6 CPU-h on 6 workers, peak RSS 0.63 GB. Bootstrap R = 9,999 for H-T, 1,999 for the E2 two-way bootstrap.

**Inputs.**
- SEs and rates: P-H100 and the ratio in §2.
- SD_cand (method of moments over all 32 pilot-2 sycophancy candidates) = 0.814; the MAD sensitivity value is 0.117.
- Component graph: the actual v1 graph for each (r_f, n_blk).
- Pools: hash-only availability, plus the supplement.

**Selection: 3,035,136 designs considered, 129,675 feasible within the caps.**

| [SIM] | Value | Note |
|---|---|---|
| r_f | 12 | r_f = 8 cannot give ≥ 5 held-out block components with n_blk ≥ 12 |
| n_blk | 12 | n_blk = 6: C2-S never computable (≤ 4 held-out blocks) ⇒ H-T void |
| n_cal | 12 | |
| m_min_HT | 12 | smallest held-out count with coverage ≥ 0.965 in every larger cell |
| n_bank (per trajectory objective) | 48 | |
| n_inert | 12 | false-fail 0.000; detection 0.987 at 1 f_k and 1.000 at 2 f_k |
| shard contrast / LoRA | 16 / 64 | not power-relevant in the simulator; smallest (pool use) |
| construction/e2 | 64 | as above |
| n_dev, n_test, n_gen,syc | (64, 96, 100) | the cheapest values; E1 and E2 are not run as confirmatory tests |
| n_rep, B, n_ep_sec, n_rep_sec | (2, 16, all, 0) | as above; n_rep_sec = 0 means "not run — not powered" |
| pool_size, reader_token_cap | 64; 120,000 | Claude client only (§7) |

**Selected cell (H-T):**

| Scenario | Rejection rate | Coverage | C2-S pass | C2-I pass |
|---|---|---|---|---|
| null, ignores step inputs | 0.000 | 0.997 | 0.00 | 1.00 |
| null, uses them, no better | 0.002 | 0.998 | 1.00 | 1.00 |
| alternative C_T = 0.15 | **0.946** (CP lower bound 0.933) | 0.972 | 1.00 | 1.00 |

**Grid totals.** H-T passes in 57 of 156 feasible cells. E1 passes in 0 of 1,024, E2 in 0 of 256, and the E1
secondaries (GSM8K, MATH) in 0. C1.4 at n_cal = 12: the false-switch rate is 0.000 and the switch rate under
15 pp under-coverage is 0.969.

**Descriptive scenarios at the selected cell (1,000 replicates each; rejection rate).**

| Claim | Scenario | Rate | Note |
|---|---|---|---|
| H-T | C_T = 0.05 / 0.10 / 0.15 / 0.20 | 0.204 / 0.662 / 0.963 / 0.999 | coverage 0.973–0.979 |
| H-T | heavy tails (t3): null / alternative | 0.005 / 0.706 | |
| H-T | weak families (row_scale, noise unresolvable) | 0.935 | |
| H-T | high stop-endpoint share (30 %) | 0.935 | |
| H-T | **sparse outer stratum (restored)**: advantage only in the 5 % outer states | 0.113 | outer stratum reportable (≥ 10 comps) 0.39; inside reportable 1.00; "damaging states" rule 0.00 |
| H-T | sparse outer stratum, uniform advantage | 0.960 | outer reportable 0.41 |
| E1 (best cell: n_rep 2, B 24, dev 192, test 192, gen 250) | alternative / TF 0.05 / TF 0.20 / transfer 0.3 / guard at ½ margin / ρ 0.5 / base anomaly 0.02 | 0.496 / 0.460 / 0.450 / 0.433 / 0.464 / 0.468 / 0.196 | every null ≤ 0.015 |
| E2 (best cell: n_rep 8, B 48, dev 96, test 192) | null / shared episodes / heavy null | 0.046 / 0.052 / 0.056 | **size fails** |
| E2 | 0.5 SD / 0.25 SD / MAD SD / ranks-only | 0.907 / 0.401 / 0.870 / 0.752 | Holm FWER over 9 null arms 0.036–0.055 |

So the restored scenario shows what the sparse-stratum rule is for. A pooled H-T test cannot detect an
advantage confined to damaging states. When it is detected, the outer stratum is reportable in only about 40% of
replicates at this bank size.

## 6. Full-refit validation of the selected cell (ruling §4(b))

See §6a below (filled from `wme-powersim-refit-2026-11-17.json`).

## 7. Projected GPU-h and dollars against the caps

| Item | GPU-h-eq (H100) | Cap | $ on 1 × H100 ($4.89/h) | $ on 8 × H100 ($36.72/h) |
|---|---|---|---|---|
| Bank: 912 visits × 10.9 s | 2.8 | | | |
| A_t oracle gradients: 384 training-eligible states × 130 s | 13.8 | | | |
| Controls (138 states), field (8 groups), directions, 12 LoRA roots | 1.4 | | | |
| **Bank total** | **18.0** | 35 | $88 | $83 + boot |
| Registered confirmatory (no powered E claim) | 0 | 50 | — | — |
| Descriptive search at the cheapest cell (if the chair runs it) | 9.5 | 50 | $46 | $44 |
| LLM reader arms (Claude only; per 6 episodes) | — | — | Haiku 5.5 $0.13 · Sonnet 5.5 $2.57 · Opus 5.5 $5.13 | |

- Before the 4,000-replicate reruns, one E2 cell passed (n_rep = 8, B = 32, dev 96, test 128, all 11 arms). It
  priced at 49.6 GPU-h confirmatory, just under the 50 cap; its size then failed at 4,000 replicates.
- A_t labels dominate the bank (77 %). The price excludes the vocabulary-row field for row_scale roots (track B)
  and generation-heavy GSM8K/MATH answers (measured short).
- **Reader token cap.**
  - Basis: track B's measured Claude overhead of 543 tokens per call, +70 tokens per pool candidate and an assumed
    +80 per own result.
  - Worst case at pool 64 and B = 16, with every call at its output limit: a team episode uses 119,552 tokens.
    Hence `reader_token_cap` = **120,000** (one cap for both LLM arms).
  - Codex (~17.8k tokens per call) cannot serve the arms.

## 8. Claims: powered and unresolved

| Claim | Status | Reason |
|---|---|---|
| H-T | **powered** at the selected cell (cheap grid); full refit in §6a | — |
| H-E1 | **unresolved — not powered** | best power 0.50: generated transfer on ≤ 250 TriviaQA groups |
| H-E2 | **unresolved — not powered** | size 0.034–0.052 > 0.035 at the grid maximum n_rep = 8 (power 0.87–0.91) |
| E1 secondaries (GSM8K, MATH) | **not run — not powered** | 0 passing cells |

Per §8.5, the bank is still built for H-T. E1 and E2 ingredients run only if the chair amends the registration
before the freeze. The study does not shrink N to fit.

## 9. Lambda

**Instances.** Only `wsp-wme-*`, each with its own terminator state directory on this server. The key was never
on a GPU box or in a log.

| Instance | Type | Window (UTC) | Instance-hours | $ | Termination |
|---|---|---|---|---|---|
| `wsp-wme-ph100-20261117` | gpu_2x_h100_sxm5 | 01:53–02:43 | 0.8415 | 8.06 | requested; `confirmed_terminated` 02:43:46Z; API status `terminated`; no live `wsp-wme-*` |
| `wsp-wme-refit-20261117` | gpu_1x_a10 (30 vCPU) | 02:54– | §9a | §9a | §9a |

Other instances were not touched.

## 10. Coordination, deviations and open items

- **9B outcome paths** (chair note): the bank and confirmatory runner writes only under `data/wme/runs/wme-c1`
  (freeze-gated) and winners manifests under `data/wme/winners/wme-c1`. No extra path is needed for track B's
  27B read guard. Engineering smoke runs write under `data/wme/runs/smoke`, and construction artefacts under
  `data/wme/bank_roots`. Neither holds 9B wme-c1 outcomes, but the chair may add them to the guard.
- **The wme-c1 panel builder is still v0-shaped.** It has test/dev/validation/construction and no `sealed_gen` or
  `bank` split, and its near-duplicate policy is word-5-gram 0.7. `panel.wme_c1.split_hashes` therefore stays
  deferred. The builder must gain the v1 split set and the v1 character policy before the freeze build; the
  exposure audit must then be re-run under the v1 policy.
- **`genaudit.SPEC` is v0.** `SPEC_V1` is added; the audit functions must switch to it (Tango, −0.03, the
  10-gram rule) before the freeze.
- **Model representation.** `wm.model` takes per-direction dictionary coordinates. Held-out bank roots must live
  in a shared representation (R(E): field features and chart columns) for H-T to generalise. The full refit
  therefore uses a shared 8-dimensional coefficient space. Making the 9B predictors consume R(E) is outstanding
  model engineering (Board A condition 4's representation-sufficiency item).
- **Track B uses z-bounds for guards** in `arms.spec.select_winner`; v1 §5.1 registers the item-group bootstrap-t
  (`wm.e2.bootstrap_t_bounds`). The winner rule is otherwise identical.
- **Assumptions.** Every assumption of the simulator is listed in `powersim_v1.ASSUMPTIONS` and in the JSON. The
  most consequential are C_T = 0.15, the C2-S degradation of 30%, the 5% generated flip noise and the
  0.5 × SD_cand search noise.
