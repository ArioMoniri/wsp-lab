# Board A: conditions check of PREREG_WMM as an EXPLORATORY plan (study M′), head 6664f40

**Verdict: ACCEPT WITH CONDITIONS to freeze as `prereg-wmm-exploratory`.**

- All of C1–C6 are met, and so is every exploratory resubmission condition I set (E1–E12).
- All four RULING-wmm-prereg addendum items are met.
- Two items are specified and unit-tested but not yet wired into a run driver, because no M′ driver exists: the E9 read-gate and the hard-stop cut order. They must be wired in before the first M′ generation (FC1, FC2).
- Three text and code nits must be fixed before the tag (FC3–FC5). None of them changes the plan.

**Scope.**
- Branch `prereg/wmm` at **6664f40**, in its worktree.
- I read:
  - `docs/PREREG_WMM.md` in full;
  - `docs/prereg/{wmm_manifest,wmm_powersim,wmm_gcal_calibration}.json`;
  - `src/wsp/wm/wmm/prereg.py`;
  - `scripts/{wmm_gcal_calibration,wmm_powersim,wmm_prereg_build}.py`;
  - the new tests;
  - `git diff 0988b3c..6664f40` and `f0b8f7e..6664f40`;
  - my REJECT report;
  - `RULING-wmm-prereg.md` with its addendum.

**Handling.**
- I wrote only this file. I made no commits and used no GPU or Lambda.
- I read no sealed outcome and nothing under `data/evals`. No M′ outcome exists.
- I opened the sealed panel files only to:
  - hash them;
  - read the pool ids (for `ids_sha256`);
  - count `options` per item (all 4: `mq` 1,544, `mm` 2,000).

  I printed no item text.

## 1. Verification

| Check | Result |
|---|---|
| Tests: `. scripts/env.sh && CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_wmm_*.py` | **132 passed**, 1 warning (scipy spearmanr), 36.6 s, exit 0. That is +12 on the 120 at my REJECT. |
| Panels unchanged vs the draft I reviewed | `manifest.panels`, `training_pools` and `exposure_registry` are **equal** between e502aed and 6664f40 (and 0988b3c). On-disk sha256 of all 6 panels = manifest `file_sha256`; all 4 pool `ids_sha256` recompute exactly. **Byte-identical.** |
| Code hashes | All 11 `manifest.code_sha256` entries match the files at 6664f40. `git_sha_at_build` = f0b8f7e (built before the addendum commit). The hashes still match, so this is harmless. |
| History | e502aed, 0a4cabe and 0988b3c have identical prereg, code and `docs/prereg` content. Only 0988b3c is an ancestor of 6664f40 (the branch was rebased). See FC3. |
| G-CAL s_U | 0.10414 · √(499/χ²₀.₀₂₅,₄₉₉) = 0.1110. The χ² bound is correct (`wmm_gcal_calibration.py:160-162`). |
| Compute arithmetic | 12.0 + 12.0 + 0.3 + 1.2 + 4.5 + 0.15 + 0.3 + 10.7 = **41.15** (≤ 42). With the M-A minimum (6.6) it is 37.05. ARC 2,161 × 4.0 s = 2.40 h, as stated. |
| Margins unchanged by the expected failures | `MARGINS` at 6664f40 equals 0988b3c except `brier` → `gcal` (both +0.01). `GUARD_ALPHA` is 0.025 in both. The addendum commit (f0b8f7e → 6664f40) touches no margin, α or `GCAL_GATES`. |

## 2. Condition table

| Condition | Met? | file:line evidence |
|---|---|---|
| **C1** G-CAL exactly as F2: prefix `Final Answer: (`, no reasoning, variants pooled, renormalised over options, ½Σ(p−y)², all 3,544 items, UB ≤ +0.01, α 0.025 | **Yes** | `PREREG_WMM.md:198`; `prereg.py:295-321`; `wmm_gcal_calibration.py:45,63-74`; test `test_wmm_prereg.py:286-304` |
| **C2** SD run on exposed Gate 1 only, ≤ 0.3 GPU-h, committed JSON with input hashes | **Yes**: s = **0.1041**, s_U 0.1110, 187 s (0.05 GPU-h), 500 exposed items (`items.json` medqa + medmcqa); dev ΔBS +0.0404 disclosed | `wmm_gcal_calibration.json:4-5,10-17,47-48`; `wmm_gcal_calibration.py:96-97`; `PREREG_WMM.md:120-137` |
| **C3(i)** non-vacuity: max-p ≤ 0.95, BS in [0.05, 0.60] | **Yes**: 0.849 and 0.392 | `wmm_gcal_calibration.json:19-20,24-25`; `prereg.py:340-370` |
| **C3(ii)** positive control: ×2 sharpening gives ΔBS ≥ +0.02 | **Yes**: **+0.069** | `wmm_gcal_calibration.json:23,26`; `prereg.py:324-326,354`; test `:327-330` |
| C3: G-CAL not reversed by the dev ΔBS | **Yes** | `PREREG_WMM.md:53-54,108` |
| **C4** power re-run: n 3,544, sd = s_U, null, dev and +0.01 scenarios, 2,000 reps; pass ≥ 0.95 at null, fail ≥ 0.97 at +0.01 | **Yes**: pass **1.00**, fail **0.9705**; M-SYC joint 0.282 / 0.330, within my 0.29–0.34 expectation (α now 0.025/3) | `wmm_powersim.json:166-171`; `wmm_powersim.py:112-141,217-219`; test `test_wmm_powersim.py:83-98` |
| **C5** old Brier and ECE reported, labelled ≡ −Δacc; F1 table in §0 | **Yes** | `PREREG_WMM.md:95-106,201` |
| **C6** dated chair note on r12 R4 | **Yes** | `RULING-wmm-prereg.md:60-67`; `PREREG_WMM.md:141-145`; manifest `gcal.c6_note` |
| **E1** joint-pass power declared; < 0.8 → exploratory | **Yes** | `PREREG_WMM.md:326-355` |
| **Every claim EXPLORATORY** | **Yes** (nit FC4) | `PREREG_WMM.md:4-7,210,237,252,257,279`; `prereg.py:19-21,263,283`; manifest `:237`; test `:279-283`. `mte_claim` output has no `status` key (`prereg.py:418-429`). |
| **E2** no margin or α change; no enlargement | **Yes** | `prereg.py:122-130`; test `:17-24`; `PREREG_WMM.md:182,200` |
| **E3** parsed-in-both companion; Δ parse rate reported | **Yes** | `prereg.py:252-271`; `PREREG_WMM.md:214-221`; test `:332-351` (a format-drift edit fails) |
| **E4** re-audit ≥ 0.9 per panel, ≥ 75 + 75 per state, every gating edit (M-SYC, M-TE) | **Yes** | `PREREG_WMM.md:203-208` |
| **E5** TE-I descriptive, never Holm-combined | **Yes** | `PREREG_WMM.md:252-255`; `prereg.py:63-67`; schedule `:98` |
| **E6** M-KD: α_f = 0.025/3 fixed; all-items (truncated = wrong) ∧ untruncated-in-both; power before run | **Yes** | `prereg.py:25-26,133-136,274-289`; `PREREG_WMM.md:186,265-273`; tests `:35-38,354-369` |
| **E7 / ruling** M-A expected-null and descriptive with F9 wording; SE floor + test; sham-null size check | **Yes**. The size ≤ 0.035 check is procedural, before any M-A read. | `PREREG_WMM.md:279-304`; `prereg.py:103-110,571-577,594-596`; test `:372-383` (finite stat, lb, p and all three nulls) |
| **Ruling** M-A restored; minimum of 20 units declared, all types, matched norms | **Yes**: 8 τ-slices, 4 seeds, 4 random, 4 sham | `prereg.py:107-109,504-512`; manifest `:275`; test `:417-426` |
| **E8 / ruling** re-priced at prompt-length order; 42 GPU-h cap; 41.2 ≤ 42 | **Yes** (thin 0.85 h slack; see note) | `PREREG_WMM.md:357-399`; `prereg.py:70-102`; manifest `:240`; test `:400-414` |
| **Ruling** cut order: M-A to its minimum, then M-TE ARC guard; "enforced in code" | **Partly**. It is declared as data and its order is asserted by a test (`i_min < i_arc < i_rest`). There is no runtime ledger or hard stop yet, because no driver exists (FC2). | `prereg.py:87-101`; `PREREG_WMM.md:386-397`; test `:400-414` |
| **E9** analysis reads only hash-logged states | **Partly**. `assert_provenanced` is implemented and tested (accepts a logged sha256; rejects an unknown or missing one). It is not yet called by any analysis code (FC1). | `prereg.py:376-393`; test `:386-397`; `PREREG_WMM.md:322-324` |
| **E10** descriptive WM-E/H-E1 wording; harmful outputs internal | **Yes** | `PREREG_WMM.md:156,235,277` |
| **E11** stem check of the 32 Gate 0 MedMCQA-train items in the manifest | **Yes**: 0 hits; panels still byte-identical | `wmm_prereg_build.py:195-200,322-332`; manifest `:225-228`; `PREREG_WMM.md:116-118` |
| **E12** L1 status stated | **Yes**: descriptive appendix, own ≤ 2 GPU-h, re-realised Gate 1 states | `PREREG_WMM.md:401-417` |
| **Ruling** expected M-SYC failures disclosed (G-CAL +0.040; neutral pass probability 0.28), guards NOT changed | **Yes** | `PREREG_WMM.md:69-75,223-233,335-336`; `prereg.py:41-49`; manifest `:301`; test `:429-436`; §1 margin diff |
| Exposure clean; panels byte-identical | **Yes** | §1 |
| Append-only vs draft 1 (optional) | **No** (an in-place rewrite, 755 lines changed). This is acceptable because nothing is frozen. After the tag, changes must be dated amendments only. | `git diff --stat 0988b3c 6664f40` |

**Notes. These are not conditions.**
- **G-CAL is priced optimistically.** The C2 run took 187 s for 1,000 batch-1 forwards, including load and canaries, which is ≤ 0.19 s per forward. Three states × 3,544 items at that rate is ≤ 0.55 GPU-h, against the 0.15 h budgeted. That still fits inside the 0.85 h slack.
- **The larger risk is M-TE's neutral pass**, priced at 1.0 s/item on the assumption that the edit shortens answers. If it does not, the hard stop removes the M-A remainder, then the ARC guard, as the ruling intends.

## 3. Final conditions

| # | Condition | When |
|---|---|---|
| **FC1** | The M′ driver or analysis entry point calls `prereg.assert_provenanced` on every state before it reads any outcome. Add one integration test showing that an unlogged state aborts the analysis. | Before the first M′ generation |
| **FC2** | The driver runs `COMPUTE["schedule"]` in order with a GPU-h ledger. No new set starts once the ledger reaches 42.0. An unfinished guard is written as `"not measured"`, and its claim is `pass False`. Add a test with a mocked ledger: the M-A remainder is lost first, then the M-TE ARC guard. | Before the first M′ generation |
| **FC3** | `PREREG_WMM.md:93`: cite the draft by an on-branch commit, **0988b3c** (content-identical to the reviewed e502aed), not `0a4cabe`, which is not an ancestor. | Before the tag |
| **FC4** | Add `"status": STATUS` to the `mte_claim` output, as `msyc_claim` and `mkd_claim` have it, and assert it in a test. | Before the tag |
| **FC5** | Correct the stale comment at `prereg.py:56` ("halved under the 30 GPU-h cap"). The cap is 42. Rebuild the manifest so `code_sha256` and `git_sha_at_build` match the tagged commit. | Before the tag |

FC3–FC5 are text or labelling changes only. Board A does not need to re-review them. The chair or drafter confirms them, with the panel hashes unchanged.

## BLUF

1. **ACCEPT WITH CONDITIONS:** freeze PREREG_WMM as `prereg-wmm-exploratory` once FC3–FC5 are applied.
2. G-CAL meets C1–C6: SD **0.104** (s_U 0.111) in 0.05 GPU-h on exposed items; max-p 0.849; BS 0.392; positive control **+0.069**.
3. Size and sensitivity: pass **1.00** with no true change; fail **0.9705** at +0.01. The C6 chair note is in the addendum.
4. All claims are EXPLORATORY. TE-I is descriptive. M-KD has α 0.025/3 and the all-items ∧ untruncated superiority test. M-A is expected-null, and the SE=0→NaN fix is tested.
5. The expected M-SYC failures (G-CAL +0.040; neutral 0.28) are disclosed, and no margin, α or gate was changed because of them.
6. Compute is 41.15 of 42 GPU-h. The M-A minimum of 20 units is declared and tested. The cut order is declared and its order tested, but not yet enforced at runtime (FC2).
7. E3, E4, E10, E11 and E12 are met. The E9 read-gate is implemented and tested but not yet wired into a driver (FC1). FC1 and FC2 are due before the first generation.
8. 132 tests pass. Panels, pools and exposure are byte-identical to the reviewed draft, with 0 exposure overlap. The prereg is a rewrite, not append-only, which is acceptable before the freeze.
