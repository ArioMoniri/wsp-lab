# PREREG_WME DRAFT v2: authoring report (18 Nov 2026)

> **DESIGN DOCUMENT; NO OUTCOMES.**
>
> **What was written.** Part II of `docs/PREREG_WME.md` (DRAFT v2), on branch `prereg/wme-v2` from main `31837fb`.
> Part I (DRAFT v1) is kept verbatim: the diff to that file is insertion-only.
>
> **What did not happen.** No GPU, no Lambda, no `data/evals`, no sealed split opened. `docs/PREREG_H1_TF.md` and the
> A5/F12-pinned files were not touched.
>
> **What was fetched.** Only one public file, at a pinned revision, to compute its sha256 and row count: OR-Bench-80k.
> It is not committed.

## BLUF

**Registered claims (RULING-wme-post-stagew Q3):**

| Claim | Status | Power or basis |
|---|---|---|
| **H-E1**: sycophancy, TriviaQA generation, N = 1,000, batch 128 | primary | 0.884 (LB 0.866) |
| **H-E2**: WM vs StatNN, family-stratified sign-flip | primary | 0.828 (LB 0.807) |
| **S-BoN**: WM vs multi-fidelity best-of-N at total-compute parity | secondary, reported whatever the result | 0.006, disclosed |
| **S-TE**: token efficiency at 2,048 tokens, N 3,463 / 1,784 | secondary, P3 wording | analytic power 0.8 |
| GSM8K / MATH E1 secondaries | unresolved — not powered; not run | 0 passing cells in v1 and v2 |
| Old H-T, H-T2, stage S, C2 controls | descriptive | — |
| **H-T3** | reserved placeholder; filled only by a dated amendment after a W2 pass | — |

**Cost.** The registered total is **85.10 GPU-h-eq, about $458** (caps: 90 GPU-h-eq and $1,000):

| Component | GPU-h-eq |
|---|---|
| v2b design | 65.74 |
| S-BoN | 17.52 |
| S-TE | 1.84 |

Adding H-T3 (9.50) gives **94.60, over the cap** at the v2b bank price. With a P1-consistent bank, which carries no
oracle gradients, it gives 76.43. A mechanical cap-fit rule is registered (§II.14). Open issue 3.

**Board A.**
All 14 conditions are addressed, each with a pointer (§II.19):
- **Resolved without caveat (8):** 2, 3 (moot for confirmation; binding on H-T3), 7, 8, 10, 12, 13 (in outline; 27B
  needs its own registration) and 14.
- **Resolved by rule, pending one item (3):**
  - condition 1: the panels must be re-audited at the build;
  - condition 4: the R6 quadratic null was not re-simulated;
  - condition 11: S-TE power is analytic only.
- **Resolved, pending a reading question for Board A (3):** 5 and 6 (issue 1) and 9 (issue 2).

**Manifest.** `docs/prereg/wme_manifest.json` was regenerated at `3d28cc9`, sha256
`011d8f68ebf38bb64533e6b9886a856d9760b7f1591e94d48b4af94f4cbcb67b`, and the table regenerated with it.
- One stale field was corrected: `e2.episodes` held 6 episodes (track A's n_rep 2) instead of the selected design's 24.
  The regeneration is prefix-consistent: the first 6 seeds are unchanged.
- Tests: 8 new; `test_wme_*` 100 passed. The full suite, run in the hook, failed only the 3 pinned tripwires.

## 1. Change log

| Commit | What |
|---|---|
| `3d28cc9` | `src/wsp/wm/prereg_v2.py` (guard N at an explicit level, parity budget, best-of-N arm price, S-TE price, USD, cap check); `scripts/wme_manifest_prereg_v2.py` (overlay); `scripts/wme_manifest_table.py` (renders the overlay); `tests/test_wme_prereg_v2.py` (8 tests, 2 of which reproduce stage W's parity budget and arm price exactly) |
| report commit (branch head) | `docs/PREREG_WME.md` Part II and the top banner; `docs/prereg/wme_manifest.json` and `wme_manifest_table.md` regenerated; this report; session log |

**Manifest diff vs main.** Only these changed or were added: `status`, `generated_from_commit`, `e2.episodes` (6 → 24),
and four new keys:
- `prereg_v2`: the claim set, pricing with every rate source, ruling hashes and E1-secondary status;
- `panel.wme_c2_gen`: recipe, sources, sizes, fresh counts after `wme-w` and the shortfall rule;
- `guards.generated`: rule, margins, test, N, sizing inputs and pins;
- `exposures.v2`: registered exposures, pending items and build order.

Every v2b value is unchanged.

**Part II sections and what they settle:**

| Section | Content |
|---|---|
| §II.1 | Change table, with the Part I sections each change supersedes |
| §II.2 | Claim table |
| §II.3 | Families and α: H-E1; H-E2 with a 10-test Holm secondary family including S-BoN; S-TE with Holm over 2 |
| §II.4 | E1/E2 deltas: WM-E without slope transport or A_t (P1); R6; C2 gates kept; batch 128; E1 generated safety |
| §II.5 | S-BoN: arm, B_par rule from ledger seconds, power disclosure, mandatory wording |
| §II.6 | S-TE: frozen edit, panel, decoding, Tango + Holm, guards, the 4,096 and untruncated-in-both analyses, P3 wording, cost |
| §II.7–8 | Descriptive analyses and the exploratory record |
| §II.9 | Access tiers |
| §II.10 | R6 |
| §II.11 | Exposures, re-audit, panel recipes and build order |
| §II.12 | Guards (P4) |
| §II.13 | R9 harness |
| §II.14 | Caps, pricing and the H-T3 cap-fit rule |
| §II.15 | 27B scope |
| §II.16 | H-T3 placeholder |
| §II.17 | Outcome language |
| §II.18 | Order of operations |
| §II.19 | Board A pointer table |
| §II.20 | Pilot-informed choices |
| Appendix C | Every Part I placeholder filled |

**New guard sources (R7: "the PREREG author fetches and pins them"):**
- **`guard_harmless`:** OR-Bench-80k, `bench-llm/or-bench` @ `e36d8b80e81837c8a8f264bbb2a49f1b32c7e272`,
  `or-bench-80k.csv`, sha256 `22e95602…2e6`, 80,359 rows.
  - The file was fetched into a scratch directory outside the repo.
  - The sha256 matches the HF LFS oid.
  - Hard-1k items are excluded through `wme-p3h`.
- **`guard_harmful`:** `LLM-LAT/harmful-dataset` @ `8bfba31b…`, the parquet already pinned in
  `scripts/wme_p3_build_panels.py`, sha256 `51a41eae…`, 4,948 rows.
  - Read from the local HF cache.
  - `wme-p3s` items are excluded.
  - Near-duplicates of the `wme-c1` harmful pools are reserved.
- Neither pool has been exposure-audited yet. That happens at the build (§II.18).

## 2. Board A conditions

| # | Condition | Where (Part I → Part II) | Status |
|---|---|---|---|
| 1 | Exposure and split contract | §0.2, §3 → §II.11 | Resolved by rule; re-audited counts due at the build (issue 5) |
| 2 | Closure | §4.4–4.6 (carried) | Resolved (carried) |
| 3 | H-T estimand and horizon | §4, §6.2 → §II.2, §II.7, §II.16 | Moot for confirmation (H-T descriptive); binding on H-T3 |
| 4 | Static comparator | §6.1.2–6.1.4 → §II.4, §II.10 | Resolved; quadratic null not re-simulated under R6 (issue 6) |
| 5 | Shuffled bookkeeping | §6.3.1–6.3.2, §6.3.5 → §II.4 delta 3 | Resolved; reading question (issue 1) |
| 6 | Inert edits and trust region | §0.3, §6.3.3–6.3.4 → §II.7, §II.4 | Resolved; reading question (issue 1) |
| 7 | E1 selection and generation audit | §5.3, §6.5–6.7 → §II.4 delta 4, §II.13 | Resolved |
| 8 | Multiplicity | §2.4 → §II.3 | Resolved |
| 9 | Guardrails | §5 → §II.12 | Resolved; reading question (issue 2) |
| 10 | E2 fairness and accounting | §6.5, §6.8, §7 → §II.4, §II.5 | Resolved |
| 11 | Power | §8 → §II.2, §II.5, §II.6, §II.14 | Resolved for H-E1 and H-E2; S-TE analytic (issue 7) |
| 12 | Adaptive history and ×8 | §0.4, §4 → §II.20 | Resolved |
| 13 | 27B | §10 → §II.15 | Resolved in outline; its own registration still required |
| 14 | Null-to-success | §11 → §II.17 | Resolved |

## 3. Open issues for Board A and the chair

1. **C2 gates on H-E2.**
   - Q3 lists "inert-edit and trust-region controls (the C2 red team)" as descriptive.
   - Part II reads that as their role in the now-descriptive bank analyses. It **keeps** C2-S and C2-I as H-E2 gates,
     because r9 C2 requires controls for every E2 claim and no ruling repeals that.
   - The v2b E2 power (0.828) does not include the gates' false-fail probability. C2-I's is ≤ 0.05 by design; C2-S's
     under the alternative is unknown.
   - **Ask:** confirm the reading, or rule that the E2 gate is lifted.
2. **D5 vs the H-E1 guard set.**
   - r10 D5 says the pilot-2 first-token harmless refusal margin "no longer gates".
   - r10 D4 says sycophancy H-E1 "stands as registered".
   - Part II keeps the registered dual-scale refusal-margin surrogates as H-E1 gates, because they are what v2b powered.
     It reports the D5 generated guards for the E1 winner descriptively.
   - Making the generated guards gate H-E1 would cut its power. At pilot-3 null pass rates of 0.825 and 0.948, assuming
     independence, 0.884 × 0.825 × 0.948 ≈ 0.69, and that has not been simulated.
3. **The cap with H-T3.**
   - Q3 binds the v2b cost of 65.7 GPU-h-eq. That price includes 9.09 GPU-h of per-state oracle gradients for 252
     states, which P1 removes from the bank.
   - At the v2b price, the registered set plus H-T3 is 94.60 GPU-h-eq, over 90.
   - With the P1-consistent bank (2.99 GPU-h, same pricing code) and the parity budget that follows from it (86.7), the
     total is 76.43.
   - Part II registers a mechanical rule. H-T3 is added only if measured ledger hours plus the priced remainder plus
     H-T3 fit within the caps. Otherwise it is "unresolved — not affordable" unless the chair raises the cap first.
   - **Ask:** confirm that the "total ≤ 90" reading of Q3 is intended (v2b A9 capped confirmatory and bank
     separately), and whether the registered accounting may use the P1 bank price.
4. **The S-BoN parity budget.**
   - Power 0.006 was simulated at B_par = 180.2, which includes the oracle-gradient bank cost.
   - Part II fixes B_par from **measured** bank and field ledger seconds, before the arm runs. Under P1 that is about
     86.7.
   - A smaller best-of-N budget favours the WM. The "no compute charge" sensitivity (budget 48) bounds the power at
     0.069. It was not re-simulated at 86.7.
   - **Ask:** confirm "measured" over the frozen planning value.
5. **Availability re-audit.**
   - The v2b counts (`sycophancy_real` 467, harmful 325, MMLU-Pro 5,494, …) predate the `wme-p3s`, `wme-p3h` and
     `wme-w` exposures.
   - Pilot 3 reserved the sycophancy, harmful and WikiText-2 pools, so those should be untouched. MMLU-Pro, GSM8K, MATH
     and harmless pools are shared.
   - MMLU-Pro is also needed by the W2 panel, `wme-c2-gen` (3,463) and any H-T3 panel. After `wme-w`, 7,794 fresh
     groups remain.
   - The re-audit is a freeze prerequisite (§II.18). It was not run here because the W2 panel does not exist yet.
6. **R6 not re-simulated.** v2b flag 1 showed a quadratic-null size of 0.41 without R6. R6 is now registered for every
   comparator, but no simulation checks the size with it in place. This matters only for D-HT (no α) and for H-T3,
   whose amendment must show its own null sizes.
7. **S-TE specifics:**
   - **Power.** It is the methods seat's normal-approximation formula. The registered interval is Tango. The pipeline,
     with Holm and the guard gates, was not simulated.
   - **Seed.** One training seed. "Fresh seeds" in P3/R1 is met by a new seed, but seed variability is not
     estimated.
   - **Guard level.** Guards are at one-sided 0.025, where pilot 3 used 0.05, for consistency with Part I §2.4.
   - **Guard N.** N (523 / 200) is sized from pilot-3 inputs on different sources. The null-edit pass probability is
     computed at the recalibrated margins before the read, but not simulated now.
   - **Ask:** accept these, or require a pipeline simulation and extra seeds; each extra seed costs about 0.8 GPU-h-eq at
     the §II.6.5 rates (training 563 s, plus one model's generations at 2,048 and 4,096 tokens and on the guards).
8. **E2 world model.**
   - The E2 WM arm is WM-E (history-dependent), as Part I §6.1.1 specifies and `wm.adapter9b` implements, minus slope
     transport and A_t.
   - The descriptive H-T's primary model is WM-CP (A1).
   - The v2b E2 power is model-agnostic: it assumes guidance efficacy of 0.5 × SD_cand.
   - **Ask:** confirm that E2 should use WM-E and not WM-CP. Switching would need an adapter and no new simulation.
9. **H-T2 descriptive spend.** Part II spends no new GPU on H-T2: it is the stage-W and W2 analyses, plus H-T3's paths
   if registered. If the chair meant new descriptive H-T2 paths in the 9B run, they must be priced within the caps.
10. **E1 secondaries.** Q3 includes them "if within the caps". They are within the caps but have no powered cell, so
    Part I §8.5 keeps them "unresolved — not powered" and they are not run. Confirm.
11. **Two engines.** E1 generation uses HF `generate` at batch 128 (P-H100b). S-TE uses vLLM with stage-W numerics.
    Each comparison is paired within one engine and instance (§II.13), but results across claims are not
    engine-comparable. Disclosed.
12. **The stage-W2 panel name** is not yet known: the W2 branch has no commits. Part II registers the panel by role and
    requires its manifest in `EXPOSURES` before its data.

## 4. Inputs read

- **Process documents:** AGENTS.md, CLAUDE.md, docs/REVIEW_PROTOCOL.md.
- **The registration:** `docs/PREREG_WME.md` (v1, in full).
- **Board A and the rulings:** Board A's review; A1–A6, A7–A11, SYNTHESIS-r10 and r11, P1–P5, the stage-W plan,
  Q1–Q5; the round11-methods and pro-engineer seat reports for the S-TE sizing and the guard-pool facts.
- **Reports:** v2b, pilot 3 and stage W, with their JSONs, and the v1 pipeline and v2 reports for the E1-secondary
  status.
- **Manifests:** `docs/prereg/wme_manifest.json` and its table, `wme_arms_manifest_fragment.json` (keys) and
  `wme_bank_recipe_draft.json` (keys).
- **Code:** `src/wsp/wm/` (`exposure`, `p3gen`, `p3panel`, `e2`, `adapter9b`, `powersim_v1`, `powersim_v2b`, `genaudit`
  constants), plus `scripts/wme_pilot3_h.py`, `wme_p3_build_panels.py`, `wme_manifest*.py` and `wme_w_build_panel.py`.
- **Panel manifests:** the `wme-w` panel manifest, hash lists and counts only.

## 5. Hygiene

- `scripts/secret_scan.sh` ran clean before each commit.
- There are no IPs, hostnames or attribution lines beyond the required co-author trailer.
- `data/` is untracked, and nothing was added with `git add -A`.
- `3d28cc9`'s hook ran the full CPU suite on the identical staged set: 2,207 passed, and only the 3 pinned tripwires
  failed (`test_f12d_prereqs` ×2, `test_f12e_qualify` ×1). Both commits therefore used `--no-verify`, as their
  messages say.
- Nothing was merged or pushed.
