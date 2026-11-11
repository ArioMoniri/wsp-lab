# Part A rotation-audit yield failure: diagnosis, remedies and F10 proposal (pre-outcome)

**Status: design document, 11 Nov 2026 (server date).** It does not amend the registration. `docs/PREREG_H1_TF.md` and
everything under `docs/prereg/` were not edited. No variant, reference or outcome was scored, opened or read: nothing under
`data/evals/` was touched, and no response, score, contrast or summary of one exists to be read (staging manifest
`scored: false`). CPU only, no GPU. Data were read only. The single write location is the scratch directory
`data/runs/a5/rotation_yield/`, whose outputs are not committed; §0.3 lists their sha256.

**Notation.** W is the stored bf16 state of the actual predecessor. Δ is the stored raw delta (root edits only). D = D_eff =
fp32(bf16(W + Δ)) − fp32(W) is the realised increment of a tensor (for chain steps, successor minus predecessor bytes).
R = Q_out D Q_inᵀ is the registered rotation of the changed block before rounding (`rotation.rotate_block`). R_eff =
bf16(W + R) − W is the scored rotation. ulp(w) is the bf16 spacing at w (2^(e−8) for w = m·2^e, m ∈ [½, 1)). Per tensor:
κ = ‖D_block‖ / √(Σ_block ulp(W)²), the rms of D in units of the rms ulp; ε_R = ‖R_eff − R‖ / ‖D‖ (the staged
`rounding_error_rel`); ε_D = ‖D − Δ‖ / ‖D‖ (roots); ε̂ = min(1, 1/(√12·κ)), the uniform-rounding prediction of ε_R;
τ = √(mean ulp²/12)·(√m + √n), the predicted spectral norm of one bf16 rounding of an m × n block.

---

## BLUF

1. Under the registered A5.3.3 realised audit, 39 of 40 rotation pairs are excluded, so the rotation contrast is unresolved by yield (< 15) whatever the scores. No outcome was read.
2. Cause: the realised mismatch is the bf16 rounding of W at the rotation's scale. In all 2,534 tensors, ε_R ≤ 1.0024·min(1, 1/(√12κ)) (§1.2).
3. The edit carries the same rounding. On 2,188 root-edit tensors the median ε_R/ε_D is 1.0003 (5–95 %: 0.996–1.021). For LoRA, ε_R·‖D‖ is 0.22–0.25 against a noise floor of 0.253 for the 72 base tensors.
4. Failures therefore follow effective norm. At low κ, norm and spectrum fail: LoRA's R_eff is 2–4 % larger (added quantisation noise, not removal). Only chain residue tensors (κ ≤ 0.09, < 0.9 % of ‖D‖²) are rounded away. At high κ, rank fails: 1e-3·s₁ falls inside the bf16 noise bulk (rank-16/32 LoRA edits have 1e-3 rank 437–1,024), and R_eff carries two roundings of noise.
5. No norm window passes reliably. The single pass (`00f8111cbad8218f`, κ ≈ 4, rank saturated at 1,023–1,024) is incidental. The five recon failures are fp32 artefacts (fp64 residual ≤ 4.0e-15).
6. Remedies, computed on CPU. (a1) rounding-aware iteration: 2/14 probed tensors converge, and those 2 are no longer rotations. (a3) rotate Δ: 16/32 roots, none of the 8 non-root chain picks. (b) noise-calibrated tolerances: 34/40, but the count swings 17–39 with its constants, and its 6 exclusions are the one registered pass plus 5 low-tertile pairs. (c) partial coverage: 1/40. (e) accept: 1/40.
7. Recommendation, F10 = (d). Keep the registered construction. Exclude on exact fp64 construction checks, support, bitwise replay and the deterministic round-to-nearest bound. Report realised norm, spectrum and rank and use them as a covariate. Computed yield: 40/40.
8. No simulation rerun is needed: 40/40 equals the audit assumption of the registered NULL-SIZE and POWER runs. If the re-stage passes fewer than 40, a slot-masked POWER-AND-YIELD run (≈ 22 min CPU) must clear 0.80 before scoring.
9. Order: F10 text → Board A → code and re-pin → re-stage the rotations (CPU, ≈ 1–5 h; +35.8 GB) → re-stage report → score 139 entries (144.56 GPU-min) → close the ledger → one analysis.
10. Do not score the 80 ladders (or the 19 missing originals) before F10 is accepted. Their outcomes decide step 2, which overrides a rotation go, and reveal per-edit response size, which correlates with any audit rule. F10 would then not be pre-outcome (F8a.1(c)).

---

## 0. Sources, runs and outputs

### 0.1 Read

| Source | Path (sha256 where it matters) |
|---|---|
| Registration A5.3, A5.3.1–A5.3.4, A5.5, F2, F3, F5.7, F7, F8a.1 | `docs/PREREG_H1_TF.md` (`dcfb5c6d…6bfc3919`, main `c41fe5a`) |
| Staging README, manifest, per-tensor audits | branch `feat/part-a-prepare` (`01e1326`), `docs/prereg/a5_manifests/part_a/README.md` (`2fcbf517…ceada1`), `manifest.json` (`d7785b35…dde8d6`), `rotation_audits.jsonl` (`f05fec42…474d35`) |
| Rotation construction and audit | `src/wsp/features/rotation.py` (`f3ff1242…1897e3`; identical on main) |
| Variant build and audit wiring | `src/wsp/eval/response_intervention.py` on `feat/part-a-prepare` (`b511d3d1…e353a`): `build_variants`, `_rotation_diagnostics`, `prepare` |
| Board A F7 review, condition 4 | `docs/reviews/council/board-a-f7-2026-11-11.md` (`817dd454…5b19`) |
| NULL-SIZE report | `docs/prereg/a5_manifests/nullsize/report.json` (`1bed2c95…dfa6`) |
| POWER report, run timings | `docs/prereg/a5_manifests/power/report.json`; `data/runs/a5/{nullsize,power}/*.time` |
| Staged rotation `variant.json` (logged moments only; no scores exist) | `data/runs/a5/part_a/qwen3.5-0.8b/<rotation variant id>/variant.json` |

### 0.2 What was run (CPU, `OMP_NUM_THREADS=4`, inside `agent-tmux`, pinned code from `feat/part-a-prepare/src`)

| Script (committed with this note) | Purpose | Wall |
|---|---|---|
| `scripts/a5_rotation_yield_population.py` | Classes and per-edit norm shares from the 2,534 staged audit rows and the logged moments | seconds |
| `scripts/a5_rotation_yield_check.py --light --deep …` | Rebuild D and R as `build_variants` does for the 5 deep edits, with ulp maps, ranks, ε_D and the probes for (a1) and (a3). Light pass (κ, ε_D) over all 40 edits | 14 min 49 s |
| `scripts/a5_rotation_yield_recon64.py` | fp64 recheck of the 5 tensors whose fp32 reconstruction residual was > 1e-5 | 19 s |
| `scripts/a5_rotation_yield_rank.py` | Noise-referenced rank of D, R_eff and (roots) R′_eff on the 5 deep edits; SVD timing | 6 min 10 s |
| `scripts/a5_rotation_yield_full.py` | All 2,534 tensors: the registered realised audit recomputed, remedy (b) and remedy (a3) | 36 min 45 s |
| `scripts/a5_rotation_yield_summarise.py`, `scripts/a5_rotation_yield_remedies.py` | Tables in §§1–2 | seconds |

Deep edits. One passing chain edit: `00f8111cbad8218f` (predecessor `34ee195e55227908`). Two excluded lora_sft edits:
`0a841d4082532e8b` (low tertile) and `15a53586efc72a50` (high tertile, rank-only failures). Two excluded chain edits with a
non-base predecessor: `0027efdf2eae7405` (high, predecessor `b7edbb9f8b339f52`) and `057dcfbedea3251f` (low, re-trained
lineage, predecessor `86ea1fe0ec777cf8`).

**Reproduction.** Every rebuilt tensor reproduces its staged audit row. In the deep check that is 258/258 tensors (ε_R to
1e-4 relative, and rank_d and rank_r exactly). In the all-tensor run it is 2,534/2,534. The recomputed registered
verdicts give 535 passing tensors, against 535 staged. My numbers therefore describe the staged bytes.

### 0.3 Scratch outputs (not committed; `data/runs/a5/rotation_yield/`)

| file | sha256 |
|---|---|
| `check.json` | `26215b315b800a764bc5a27c8e8c7590b61abb1669cf2fca105a83b67f4c5243` |
| `full.json` | `49b09d6f8992dc338d01c322338cd22ad528b79386ebd4c07d6084864fd0053b` |
| `population.json` | `92eebd0acdab8cb043a02b6b51bc43d30780a4785f7f44726cb7e2c56df0b4a0` |
| `rank.json` | `89474d41d885c3b50fe19248693f2aa74d3e9afa4fca464c502f9626acdf5032` |
| `recon64.json` | `c69aa9e12fd7048259abd31ce5d9a39cddbb7f0cb5c7fc35c56cdfb6f7d7eb88` |
| `remedies.json` | `4e0cd5a7fbb4cfd4c8f8738bc9886ad9755ca75bd4a7493ff6486d961bb93833` |
| `summary.json` | `650e16d8f328df5206fd1f96388c15eead49b649adae4aafd07eec42c4c27886` |
| `test_00f8.json` | `582ba93a336b81465053ed5377fdb1931ca9dddd76ae62e76560f5bd967512ca` |

---

## 1. Diagnosis

### 1.1 Population: the 2,534 staged audit rows by failure class

Classes are disjoint and assigned in order. N: norm mismatch > 2e-2. S: norm passes, spectrum > 5e-2. K: norm and spectrum
pass, |rank diff| > 1. ε_R is the staged `rounding_error_rel`. Source: `rotation_audits.jsonl` via
`a5_rotation_yield_population.py`.

| family | class | tensors | median ε_R (p10–p90) | median norm | median spectrum | median rank diff (min, max) | median D density | median nonzero_r / nonzero_d |
|---|---|---|---|---|---|---|---|---|
| lora_sft | pass | 462 | 0.075 (0.044–0.111) | 0.0028 | 0.030 | 0 (0, 1) | 0.92 | 0.999 |
| lora_sft | N | 796 | 0.295 (0.237–0.416) | 0.0270 | 0.116 | 0 (0, 7) | 0.66 | 1.012 |
| lora_sft | S | 625 | 0.167 (0.132–0.217) | 0.0128 | 0.068 | 0 (0, 26) | 0.80 | 0.999 |
| lora_sft | K | 277 | 0.048 (0.034–0.081) | 0.0011 | 0.020 | +48 (+2, +340) | 0.94 | 0.999 |
| chain | pass | 73 | 0.121 (0.073–0.156) | 0.0069 | 0.035 | 0 (0, 1) | 0.85 | 0.997 |
| chain | N | 225 | 0.442 (0.265–0.935) | 0.0355 | 0.153 | 0 (−179, +88) | 0.53 | 1.027 |
| chain | S | 40 | 0.176 (0.135–0.458) | 0.0123 | 0.064 | +26.5 (0, +142) | 0.68 | 1.053 |
| chain | K | 36 | 0.071 (0.055–0.123) | 0.0025 | 0.025 | +39 (+3, +222) | 0.85 | 1.015 |

Rank direction among the |diff| > 1 failures: lora_sft 306 up and 0 down; chain 66 up and 12 down. The README's criterion
counts (norm 796/225, spectrum 1,421/265, rank 306/78, recon 3/2) are reproduced.

Two invariants hold on every row. By Mirsky's inequality and the triangle inequality, norm mismatch ≤ ε_R and spectrum
mismatch ≤ ε_R, since σ(R) = σ(D) and ‖R‖ = ‖D‖. Both hold in 2,534/2,534 rows. **Any realised norm or spectrum tolerance
is therefore a bound on the rounding error ε_R and nothing else.**

### 1.2 Mechanism: double rounding at a scale set by W's grid

1. D is already rounded once: D = Δ + E_D, with E_D the rounding of W + Δ. Its entries are whole multiples of the local
   ulp, and coordinates where |Δ| < ulp/2 are zero.
2. R = Q_out D Q_inᵀ rotates D, including its rounding noise E_D, and spreads the energy evenly over the block.
3. R_eff = R + E_R adds a second, independent rounding. When most entries are a few ulps, E_R behaves like uniform noise
   with variance ulp²/12 per entry, so ε_R ≈ 1/(√12·κ). Below κ ≈ 0.3 the entries fall under half an ulp, are rounded to
   zero, and ε_R → 1.

The uniform-rounding prediction ε̂ matches the staged ε_R across all 2,534 tensors (light pass; ratio
ε_R/ε̂ ≤ 1.0024 everywhere, with 44 tensors between 1.000 and 1.0024):

| κ bin | tensors | median ε_R (measured) | median ε̂ = 1/(√12κ) |
|---|---|---|---|
| [0, 0.5) | 150 | 0.858 | 2.547 (capped at 1 in ε̂) |
| [0.5, 1) | 569 | 0.337 | 0.373 |
| [1, 2) | 857 | 0.208 | 0.215 |
| [2, 4) | 489 | 0.100 | 0.101 |
| [4, 8) | 397 | 0.052 | 0.052 |
| ≥ 8 | 72 | 0.034 | 0.034 |

The edit's own rounding has the same size. For the 32 root edits the raw Δ is stored. Across their 2,188 tensors the median
ε_R/ε_D is **1.0003** (5th–95th percentile 0.996–1.021). The two chain roots have ε_D = 0, because their stored delta is
already realised. At the edit level, the pooled ε_R of every lora_sft edit is 0.253/‖D‖ for κ ≳ 1, and somewhat below it for κ < 1.
Here 0.253 = √(Σ ulp²/12) summed over the 72 targeted base tensors. Examples: 0.037 at ‖D‖ = 6.78 (`15a53586`,
predicted 0.037) and 0.335 at ‖D‖ = 0.657 (`0a841d40`, predicted 0.385). The full table is in §1.4. The realised mismatch is therefore a deterministic function of the predecessor's
grid and the edit's effective norm. It is not a property of the particular rotation.

### 1.3 CPU check of the five deep edits, per failure class

Columns: κ, the median of med|D|/ulp over the block, D density, and fraction of R entries below ulp/2 and fraction rounded
to zero. Then ε_R measured against ε̂, signed norm mismatch (R_eff − D), spectrum mismatch, and ranks at 1e-3 of D and R_eff.
Then ranks above ‖E_R‖₂ (the realised noise edge), ε_D and rank(Δ) at 1e-3 for roots, and the two remedy probes: (a3)
passes under the registered audit, and (a1) passes after ≤ 12 error-feedback iterations. Medians are over the class's
tensors. Source: `check.json` via `a5_rotation_yield_summarise.py`.

| edit | family/tertile | pred | class | n | κ | med abs D/ulp | D density | R < ulp/2 | R → 0 | ε_R / ε̂ | norm (signed) | spectrum | rank D / R @1e-3 | rank above ‖E_R‖₂ D / R | ε_D | rank Δ | a3 | a1 (iters) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `00f8111cbad8218f` | chain/1 | 34ee195e | pass | 6 | 3.96 | 4.00 | 0.91 | 0.09 | 0.09 | 0.073 / 0.073 | +0.0026 | 0.019 | 1024 / 1024 | 183 / 276 | – | – | – | – |
| `0a841d4082532e8b` | lora_sft/0 | base | N | 70 | 0.71 | 1.00 | 0.60 | 0.39 | 0.39 | 0.359 / 0.404 | +0.0303 | 0.136 | 1024 / 1024 | 16 / 16 | 0.355 | 16 | 70/70 | 0/2 (12, 12) |
| `0a841d4082532e8b` | lora_sft/0 | base | S | 2 | 0.38 | 0.00 | 0.43 | 0.55 | 0.55 | 0.524 / 0.768 | +0.0113 | 0.181 | 768 / 768 | 6 / 5 | 0.506 | 16 | 2/2 | 0/2 (12, 12) |
| `15a53586efc72a50` | lora_sft/2 | base | pass | 28 | 6.36 | 7.00 | 0.95 | 0.05 | 0.05 | 0.045 / 0.045 | +0.0010 | 0.019 | 1024 / 1024 | 32 / 60 | 0.045 | 32 | 28/28 | – |
| `15a53586efc72a50` | lora_sft/2 | base | K | 44 | 7.66 | 8.50 | 0.96 | 0.04 | 0.04 | 0.038 / 0.038 | +0.0007 | 0.015 | 954 / 1024 | 32 / 70 | 0.038 | 32 | 28/44 | 0/2 (12, 12) |
| `0027efdf2eae7405` | chain/2 | b7edbb9f | K | 10 | 5.13 | 3.00 | 0.84 | 0.15 | 0.15 | 0.055 / 0.056 | +0.0014 | 0.023 | 756 / 963 | 5 / 11 | – | – | – | 0/2 (12, 12) |
| `0027efdf2eae7405` | chain/2 | b7edbb9f | N | 22 | 0.0023 | 0.00 | 0.0019 | 1.00 | 1.00 | 0.995 / 1 | −0.8651 | 0.930 | 658 / 638 | 1 / 0 | – | – | – | 0/2 (12, 12) |
| `057dcfbedea3251f` | chain/0 | 86ea1fe0 | pass | 7 | 2.18 | 2.00 | 0.79 | 0.19 | 0.19 | 0.129 / 0.132 | +0.0076 | 0.045 | 1024 / 1024 | 20 / 115 | – | – | – | – |
| `057dcfbedea3251f` | chain/0 | 86ea1fe0 | K | 5 | 2.17 | 2.00 | 0.80 | 0.19 | 0.19 | 0.130 / 0.133 | +0.0076 | 0.046 | 1018 / 1024 | 21 / 101 | – | – | – | 0/2 (12, 12) |
| `057dcfbedea3251f` | chain/0 | 86ea1fe0 | N | 64 | 0.09 | 0.00 | 0.05 | 0.86 | 0.86 | 0.882 / 1 | −0.3466 | 0.367 | 1024 / 1024 | 1 / 0 | – | – | – | 2/2 (2, 3) |

Signed norm ranges: `0a841d40` N +0.0222 to +0.0381; `15a53586` K +0.0003 to +0.0024; `0027efdf` N −0.9365 to −0.7704;
`057dcfbe` N −0.5005 to −0.1703.

**Noise-referenced rank** (`rank.json`) counts singular values above max(1e-3·s₁, c·τ):

| edit | c = 1: tensors with \|rank D − rank R\| ≤ 1 | c = 1.5 | c = 2 | c = 3 | median rank D / R at c = 3 | (a3) at c = 3 |
|---|---|---|---|---|---|---|
| `00f8111cbad8218f` | 0/6 | 0/6 | 0/6 | 5/6 | 7 / 7.5 | – |
| `0a841d4082532e8b` | 1/72 | 71/72 | 71/72 | 69/72 | 16 / 15 | 72/72 |
| `15a53586efc72a50` | 0/72 | 47/72 | 66/72 | 72/72 | 32 / 32 | 72/72 |
| `0027efdf2eae7405` | 22/32 | 22/32 | 32/32 | 32/32 | 0 / 0 | – |
| `057dcfbedea3251f` | 63/76 | 64/76 | 64/76 | 76/76 | 0 / 0 | – |

**fp64 reconstruction recheck** (`recon64.json`). The five tensors with fp32 residuals of 1.05e-5 to 1.40e-5
(`0027efdf…` L19 down_proj, `13b02fe2…` L17 down_proj, `1b2b671e…` L0 down_proj, `34e407b9…` L14 down_proj, `5f80bd48…`
L8 down_proj) have fp64 residuals of 1.4e-15 to 4.0e-15. These are fp32 SVD artefacts on 1024 × 3580–3584 blocks; the
SVD serves only the audit and plays no part in the construction.

### 1.4 The README's two patterns: confirmed, corrected, and one added

| README statement | Verdict | Evidence |
|---|---|---|
| "Where D_eff is small relative to W's bf16 spacing … rounding then removes most of R (`rounding_error_rel` near 1)" | **Holds only for chain residue tensors.** These tensors have κ ≤ 0.09, and D is a sparse set of ±1-ulp flips (density 0.19 % in `0027efdf`, 5 % in `057dcfbe`). 86–100 % of R falls below half an ulp and rounds to zero, with ε_R 0.88–0.995 and norm −17 % to −94 %. Across the 40 edits, tensors with κ < 0.25 hold at most 0.89 % of an edit's ‖D‖² (`057dcfbe`); in the other 39 edits they hold < 0.005 %. | §1.3 rows N of `0027efdf`/`057dcfbe`; light pass |
| (same, as applied to lora_sft, the main failure) | **Corrected.** The LoRA norm and spectrum failures are additive quantisation noise, not removal. For `0a841d40` (κ 0.71), 39 % of R falls below half an ulp, but R_eff has as many nonzeros as D (population ratio 1.012) and is **larger** (norm +2.2 % to +3.8 %). ε_R = 0.359, the same as the edit's own ε_D = 0.355. | §1.1 lora_sft N; §1.3 |
| "bf16 rounding noise in R_eff often lifts its numerical rank … by more than ±1 … sometimes by hundreds" | **Confirmed and sharpened.** At high κ, 1e-3·s₁ lies inside the bf16 noise bulk, so the 1e-3 rank of D counts its own rounding noise: rank-16 and rank-32 LoRA edits have rank_d 437–1,024 (512 or 1,024 is the full block rank). R_eff carries two roundings (rotated E_D plus E_R), its bulk edge is ≈ √2 higher, and its rank rises (lora_sft 306 up, 0 down; up to +340). | §1.1 K; §1.3 (rank Δ 16/32 against rank D 437–1,024) |
| (not in README) | **No norm window passes reliably.** Low κ fails norm or spectrum: lora_sft passing tensors have ε_R ≤ 0.111 (p90), and spectrum failures start at ε_R ≈ 0.13 (S-class p10), i.e. κ ≲ 2. High κ fails rank once s₁ clears the noise bulk. The one passing pair has κ 3.9–4.0 in all six tensors (ε_R 0.073), with rank_d 1,023–1,024 saturated. lora_sft passes 0/10 in every tertile; chain passes 1/10 (middle). | §1.3 row `00f8111c`; manifest `edit_audits` |

Per-edit rounding error, effective norm and κ (light pass, all 40 edits; pooled ε_R = √(Σ_t ε_t²‖D_t‖²)/‖D‖):

| edit | family/tertile | root | ‖D‖ | noise floor √(Σulp²/12) | pooled ε_R | pooled ε̂ (noise/‖D‖) | pooled ε_D | median κ | staged passing tensors | pooled realised norm (signed) | ‖D‖² share in κ < 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `057dcfbedea3251f` | chain/0 | no | 0.769 | 0.259 | 0.152 | 0.336 | – | 0.09 | 7/76 | +0.0051 | 0.0089 |
| `0bc0a0c4ad0520e0` | chain/0 | no | 0.645 | 0.253 | 0.341 | 0.392 | – | 0.73 | 0/72 | +0.0296 | 0.1283 |
| `0d055869527177b4` | chain/0 | no | 0.654 | 0.253 | 0.331 | 0.386 | – | 0.79 | 0/72 | +0.0270 | 0.1195 |
| `13b02fe2673cc296` | chain/0 | no | 0.419 | 0.072 | 0.160 | 0.172 | – | 1.67 | 0/6 | +0.0098 | 0.0000 |
| `00f8111cbad8218f` | chain/1 | no | 1.032 | 0.075 | 0.073 | 0.073 | – | 3.96 | 6/6 | +0.0026 | 0.0000 |
| `014f8a9ec78db2be` | chain/1 | yes | 0.813 | 0.122 | 0.140 | 0.151 | 0.000 | 1.99 | 0/18 | +0.0075 | 0.0000 |
| `0559051354691a70` | chain/1 | yes | 1.293 | 0.093 | 0.072 | 0.072 | 0.000 | 4.02 | 1/10 | +0.0025 | 0.0000 |
| `0027efdf2eae7405` | chain/2 | no | 1.618 | 0.108 | 0.055 | 0.066 | – | 0.00 | 0/32 | +0.0014 | 0.0000 |
| `05531ce1db4df63f` | chain/2 | no | 1.957 | 0.253 | 0.127 | 0.129 | – | 2.12 | 57/72 | +0.0072 | 0.0000 |
| `14cdaabee246493b` | chain/2 | no | 1.305 | 0.093 | 0.071 | 0.072 | – | 4.04 | 2/10 | +0.0024 | 0.0000 |
| `0a841d4082532e8b` | lora_sft/0 | yes | 0.657 | 0.253 | 0.335 | 0.385 | 0.331 | 0.71 | 0/72 | +0.0292 | 0.1048 |
| `1ab850a41051fe45` | lora_sft/0 | yes | 0.712 | 0.253 | 0.313 | 0.355 | 0.309 | 0.77 | 0/72 | +0.0273 | 0.0652 |
| `1b2b671e5da5d625` | lora_sft/0 | yes | 0.976 | 0.253 | 0.239 | 0.259 | 0.238 | 1.03 | 0/72 | +0.0199 | 0.0000 |
| `1fc51f7164052d50` | lora_sft/0 | yes | 0.673 | 0.253 | 0.319 | 0.376 | 0.317 | 0.77 | 0/72 | +0.0255 | 0.0806 |
| `22659cefc904e29f` | lora_sft/0 | yes | 1.031 | 0.253 | 0.226 | 0.245 | 0.225 | 1.19 | 0/72 | +0.0177 | 0.0042 |
| `2d1de332e36d8f00` | lora_sft/0 | yes | 1.028 | 0.253 | 0.232 | 0.246 | 0.231 | 1.12 | 0/72 | +0.0201 | 0.0000 |
| `453d8c08966438e0` | lora_sft/0 | yes | 0.900 | 0.253 | 0.252 | 0.281 | 0.250 | 0.99 | 0/72 | +0.0201 | 0.0066 |
| `455b881d5de4e6e0` | lora_sft/0 | yes | 0.820 | 0.253 | 0.275 | 0.308 | 0.273 | 0.87 | 0/72 | +0.0230 | 0.0069 |
| `61d1f61e25212068` | lora_sft/0 | yes | 0.590 | 0.253 | 0.365 | 0.428 | 0.360 | 0.63 | 0/72 | +0.0309 | 0.1547 |
| `649bd035e79488e5` | lora_sft/0 | yes | 0.728 | 0.253 | 0.308 | 0.347 | 0.305 | 0.83 | 0/72 | +0.0271 | 0.0168 |
| `19d9a98cfb2db6e2` | lora_sft/1 | yes | 1.309 | 0.253 | 0.183 | 0.193 | 0.183 | 1.34 | 1/72 | +0.0130 | 0.0000 |
| `2d7f5f78fac4a9ae` | lora_sft/1 | yes | 1.436 | 0.253 | 0.172 | 0.176 | 0.172 | 1.50 | 14/72 | +0.0131 | 0.0000 |
| `2f61a4086395b719` | lora_sft/1 | yes | 1.305 | 0.253 | 0.187 | 0.194 | 0.187 | 1.42 | 0/72 | +0.0145 | 0.0000 |
| `2fb8d03d2547f357` | lora_sft/1 | yes | 1.146 | 0.253 | 0.211 | 0.221 | 0.210 | 1.18 | 0/72 | +0.0175 | 0.0000 |
| `3499f7dcc5968d96` | lora_sft/1 | yes | 1.137 | 0.253 | 0.213 | 0.222 | 0.212 | 1.23 | 0/72 | +0.0178 | 0.0000 |
| `3996bea032c24200` | lora_sft/1 | yes | 1.776 | 0.253 | 0.139 | 0.142 | 0.139 | 1.93 | 23/72 | +0.0087 | 0.0000 |
| `4f918f6bbd264b34` | lora_sft/1 | yes | 1.142 | 0.253 | 0.209 | 0.221 | 0.208 | 1.17 | 0/72 | +0.0166 | 0.0000 |
| `5f80bd482d092938` | lora_sft/1 | yes | 1.940 | 0.253 | 0.127 | 0.130 | 0.127 | 2.02 | 25/72 | +0.0071 | 0.0000 |
| `7d03d89c0b0299e6` | lora_sft/1 | yes | 1.881 | 0.253 | 0.133 | 0.134 | 0.133 | 1.96 | 26/72 | +0.0083 | 0.0000 |
| `983c5e13c59e0022` | lora_sft/1 | yes | 1.125 | 0.253 | 0.215 | 0.225 | 0.214 | 1.15 | 0/72 | +0.0181 | 0.0000 |
| `0712caa1d248514a` | lora_sft/2 | yes | 2.473 | 0.253 | 0.102 | 0.102 | 0.102 | 2.45 | 45/72 | +0.0050 | 0.0000 |
| `07f9a50b29a7ef5c` | lora_sft/2 | yes | 3.236 | 0.253 | 0.078 | 0.078 | 0.078 | 3.56 | 49/72 | +0.0030 | 0.0000 |
| `15a53586efc72a50` | lora_sft/2 | yes | 6.780 | 0.253 | 0.037 | 0.037 | 0.037 | 7.39 | 28/72 | +0.0007 | 0.0000 |
| `2233954b0cc3edc0` | lora_sft/2 | yes | 4.696 | 0.253 | 0.054 | 0.054 | 0.054 | 4.94 | 50/72 | +0.0014 | 0.0000 |
| `2971f30e5276edd9` | lora_sft/2 | yes | 4.202 | 0.253 | 0.060 | 0.060 | 0.060 | 4.53 | 49/72 | +0.0018 | 0.0000 |
| `34e407b92a902282` | lora_sft/2 | yes | 5.268 | 0.253 | 0.048 | 0.048 | 0.048 | 5.83 | 12/72 | +0.0011 | 0.0000 |
| `5abd76a640cc2b9d` | lora_sft/2 | yes | 3.668 | 0.253 | 0.069 | 0.069 | 0.069 | 3.77 | 62/72 | +0.0023 | 0.0000 |
| `72d3756e968e3209` | lora_sft/2 | yes | 5.225 | 0.253 | 0.048 | 0.048 | 0.048 | 6.00 | 15/72 | +0.0011 | 0.0000 |
| `85366c308158ea48` | lora_sft/2 | yes | 2.336 | 0.253 | 0.108 | 0.108 | 0.108 | 2.34 | 34/72 | +0.0056 | 0.0000 |
| `8cfc1393b97300ab` | lora_sft/2 | yes | 5.174 | 0.253 | 0.049 | 0.049 | 0.049 | 5.57 | 29/72 | +0.0012 | 0.0000 |

**Consequence.** Within an edit, every tensor fails or passes by κ, and the registered rule excludes a pair when any tensor
fails. Exclusion is therefore a function of effective norm, predecessor grid and noise-bulk geometry. It is unrelated to
whether a given Haar draw is a faithful rotation: all 2,534 drawn rotations are exact to fp32 before rounding
(pre-rounding norm mismatch ≤ 1.43e-7 in the staged rows; orthogonality error ≤ 8.7e-7).

---

## 2. Remedies that keep the registered hypothesis

All of them keep the same question: does the sign of ⟨g, D_eff − R_eff⟩ predict the sign of y(W + D) − y(W + R), with R a
Haar rotation of the realised block? None reads an outcome. Each is a legitimate **pre-outcome** amendment if adopted before
any G3 score exists (A5.3 "Exposure"; F8a.1(c) exposure definition), because its inputs are bytes and staging metadata only.
The trigger is a yield computed from metadata, which must be disclosed (F10.1 below does so).

### 2.1 Remedy table

| remedy | rule changed | yield (computed unless marked) | simulations affected | cost | bias risk |
|---|---|---|---|---|---|
| **(e) accept**, score ladders only, rotation unresolved (the registered reading) | none | 1/40 (staged) | None to rerun. The registered go branch is unreachable (step 3 needs ≥ 15), so the registered power 0.854 is moot: Part A can only end alternative (step 2) or unresolved | Scoring 81 staged + 19 refs = 104.00 GPU-min (README) | None added. The main contrast is lost |
| **(a1) rounding-aware construction**: error feedback X ← X + (P(Y) − Y), with Y = bf16(W + X) − W and P(Y) = Y's singular vectors with D's singular values | A5.3 construction (R no longer Q D Qᵀ); F3(ii) algorithm; audit unchanged | ≈ 0 (estimated from probes). 2/14 failing tensors pass in ≤ 12 iterations, both chain residue tensors whose output is 0.45–0.53‖D‖ from R (0.91–0.98 realised). The other 12 do not pass: on LoRA the norm drifts to −6 % to −10 %. K-class tensors keep a rank excess (`15a53586` 682 against 437 and 946 against 802; `0027efdf` 803 against 681 and 886 against 794). `0027efdf`'s residue tensors reach a norm and spectrum match only by abandoning R (0.97–1.00‖D‖ away) | None if it worked | ≥ 12 full SVDs per tensor ≈ 2,534 × 12 × 0.60 s ≈ 5 h on top of staging, and non-convergent | **High.** Where it converges it is no longer a rotation; orientation is set by W's grid |
| (a2) round R to W's grid before the audit | none: R_eff = bf16(W + R) − W *is* R on W's grid | = staged, 1/40 | – | – | – |
| **(a3) rotate the raw Δ** (roots), round once, registered audit | A5.3 rotation object; overrides A5.3's raw-factor condition (≤ 1e-3 reproduction; actual ε_D is 0.04–0.36); family-asymmetric | 16/32 roots under the registered audit (low 7/10, middle 8/10, high 1/10: the 1e-3 rank still fails at high κ), and 30/32 if rank is also moved to c = 3 τ (two rule changes). The 8 non-root chain picks have no stored Δ (cumulative snapshot only) and stay excluded, so ≤ 30/40 | Yield < 40, so POWER-AND-YIELD must be rerun with slot masks (≈ 22 min CPU, code change) | Re-stage 32 rotations (CPU, within the §3.4 range) | **Moderate.** The pre-rounding spectrum becomes Δ's, not D_eff's. Noise is symmetric (one rounding each) |
| **(b) noise-calibrated realised tolerances**: norm ≤ 2e-2 + ε̂, spectrum ≤ 5e-2 + ε̂, rank counted above max(1e-3 s₁, c τ) within ±1 | A5.3/A5.3.3 tolerances and rank threshold | 34/40 at c = 3; 17/40 at c = 2; 39/40 at c = 3 with ±2. Norm and spectrum alone: 40/40. The 6 excluded at c = 3 are `00f8111cbad8218f` (the only registered pass, 1 tensor), `0a841d4082532e8b` (3), `0bc0a0c4ad0520e0` (3), `1ab850a41051fe45` (1), `61d1f61e25212068` (20) and `649bd035e79488e5` (1): 5 of them low tertile | Rerun with slot masks if < 40 | Re-audit all 2,534 tensors (CPU, within the §3.4 range) | **Moderate.** Norm and spectrum become vacuous (§1.1 invariant), so everything rests on a realised rank whose pass/fail flips by 2 near the noise edge; c and ±k would be chosen with yields visible |
| **(c) partial rotation**: rotate tensors with κ ≥ κ\*, carry the others bitwise; keep the pair if rotated tensors cover ≥ x of ‖D‖² and pass the registered audit | A5.3.3 whole-pair exclusion → partial; contrast and prediction cover rotated tensors only | 1/40 for every κ\* ∈ {0.25, 0.5, 1, 2} and x ∈ {0.9, 0.99}. Coverage alone, without the registered audit: 40, 36, 24, 14 at x = 0.9 | Rerun if used with (d) | Small | Moderate: shrinks the contrast. It does not address the cause, because κ is nearly uniform within an edit |
| **(d) construction audit + registered rounding bound** (recommended; F10) | A5.3 post-rounding *exclusion* → fp64 construction checks, support, bitwise replay, round-to-nearest bound and a sub-resolution guard. Realised norm, spectrum and rank become reported diagnostics and a covariate | **40/40** (staged rows: orth ≤ 8.7e-7, pre-rounding norm ≤ 1.43e-7, support 2,534/2,534; recon fp64 ≤ 4.0e-15; pooled \|norm\| ≤ 0.031 against the 0.5 guard) | None if 40/40 (equals the F2/F7 assumption); otherwise a slot-masked rerun, gate ≥ 0.80 | Re-stage 39 rotations: ≈ 1–5 h CPU, +35.8 GB, code + tests + re-pin (xv). Scoring 139 entries = 144.56 GPU-min (the registered price) | **Low–moderate.** R_eff carries extra rounding energy (pooled norm +0.0007 to +0.0309). The first-order part is in the prediction; the curvature part is handled by covariate and sensitivities (F10.5) |
| (d′) (d) plus a cap on pooled ε_R | as (d), plus a cap | 12/40 at 0.10, 19 at 0.15, 24 at 0.20, 31 at 0.25, 39 at 0.35; all low-tertile pairs excluded below 0.20 | Rerun required (composition moves to the high tertile) | as (d) + 22 min | **High.** A cap is selection on effective norm, set with yields visible, and it undoes the registered tertile design |

Yield for (d) is computed, not assumed. Every staged tensor already passes every construction quantity F10 keeps
(`rotation_audits.jsonl`: orthogonality ≤ 8.7e-7, pre-rounding norm ≤ 1.43e-7, support identical 2,534/2,534). The only
fp32 failures are the five reconstruction residuals, which are ≤ 4.0e-15 in fp64. The round-to-nearest bound holds by
construction for correct code, and bitwise reproduction (§0.2) confirms the staged bytes are the registered rounding. The
largest pooled realised norm mismatch is 0.0309, far inside the 0.5 guard.

### 2.2 Why (d) and not (b) or (a3)

* Over all 2,534 rows, a realised norm or spectrum tolerance only bounds ε_R (§1.1), and ε_R is fixed by κ (§1.2). It
  is a function of effective norm, so a tolerance that bites selects on norm, and one that does not bite is vacuous. With
  ε̂ added, norm and spectrum pass 40/40, and (b) reduces to a realised-rank rule. That rule is tolerable at c = 3
  (34/40), but it is not stable: 17/40 at c = 2 and 39/40 at c = 3 with ±2. In five of the six excluded pairs, 1–3
  tensors are off by exactly 2 (all five pass at ±2); only `61d1f61e` (20 tensors) is further off. The six include
  `00f8111cbad8218f`, the only pair the registered audit passes. The constants would be chosen with these yields
  visible, and five of the six exclusions are low-tertile. (b) is the fallback if Board A wants a realised criterion with bite; F10.5(e) reports its 34-pair
  subset descriptively.
* (a3) is the cleanest symmetric object: one rounding each. It passes the registered audit on `0a841d40` (72/72) and
  on 16/32 root edits, but only 1/10 high-tertile edits, because the 1e-3 rank still fails at high κ. It does not exist
  for the 8 non-root chain picks, which store only cumulative snapshots. It also reverses A5.3's explicit choice of the
  realised D as the rotated object. It is worth having as a descriptive sensitivity, but not as the primary rule.
* (d) keeps everything the registration says "spectrum-matched" means ("refers to the pre-rounding construction", A5.3;
  "Spectrum matching is pre-rounding unless the realised audit supports it", A5.3.3). It makes the exact parts of the
  audit the exclusion rule. It reports the realised part with a registered, interpretable bound instead of using it as
  an accidental norm filter, and it leaves the 40 registered pairs and their tertile allocation intact.

### 2.3 Bias analysis of (d)

R_eff = R + E_R. E_R depends on W's grid and on R, never on g or on any outcome.

* *First order.* ⟨g, E_R⟩ is part of the registered prediction ⟨g, D_eff − R_eff⟩ (A5.3; restated in F9 as "first-order
  forecast ⟨g, D_eff − R_eff⟩ (A5.3)").
* *Second order.* R_eff carries about ε_R² of extra energy relative to D_eff. D_eff already contains its own E_D; the
  rotated copy of E_D plus the fresh E_R is the asymmetry. Under positive curvature this raises y(W + R) by an amount
  growing with ε_R², which is largest in the low tertile (pooled ε_R up to 0.365). It pushes d_i towards "edit less
  harmful than its rotation" in every pair. That is a constant-sign nuisance; it does not depend on the predicted sign.
  It helps agreement where the prediction is negative and hurts it where positive, so it inflates agreement only if
  predictions are sign-imbalanced. Hence F10.5: the energy ratio as a covariate in the registered nuisance sensitivity,
  a tertile-stratified report, a pre-listed subset (the 30 pairs with pooled |norm mismatch| ≤ 2e-2), and the registered
  constant-sign comparator reported next to it. The F9 quadratic forecast (a_R + q_R) also prices this energy through
  K-FAC.
* *Composition.* Under (d) all 40 registered pairs stay in, so the 10/10/10 and 4/3/3 tertile design is untouched. Under
  (e), (d′) and (c) the design is lost or skewed.

### 2.4 Effect on NULL-SIZE and POWER-AND-YIELD

The registered NULL-SIZE run and the F7 POWER-AND-YIELD runs count unavailable audit evidence as a pass (F2: 423
unavailable entries, `nullsize/report.json` validation counts). Their stated yield of 1.0 "rests on" that assumption
(F7 results). Board A F7 condition 4 forbids converting it into audit evidence. The frozen `unavailable_audit_fails`
scenario resolves no pair in 10,000 replicates.

* Under **(d)** with 40/40 actual audits passing, the actual audit state matches the simulated assumption, and **no rerun
  is needed**. That is not a conversion: the evidence is the F10 re-stage audit, and the simulations stay conditional on
  the F2 proxy (which already lists rounding among the effects it misses).
* If the re-stage passes **fewer than 40**, rerun POWER-AND-YIELD (and its embedded NULL-SIZE pass) before scoring,
  labelled, with the failing selected slots masked in every replicate. The ≥ 0.80 ordered-power gate and the ≤ 0.07 size
  gates apply (A5.3.2). Cost: the F7 runs took 21 min 39 s and 22 min 27 s (`power/*.time`); NULL-SIZE scenarios took
  4 min 54 s to 16 min 27 s. It needs a small `--slot-mask` addition to `scripts/a5_simulation.py` plus tests and a
  re-pin.
* (e) needs no rerun because the outcome is fixed. (a3), (b), (c) and (d′) each need the masked rerun, and (d′) and (c)
  would likely fall below 0.80 (not run).

---

## 3. Recommendation: amendment F10 (draft)

### 3.1 Recommended remedy

**(d)**: keep the registered rotation construction. Make the exact construction checks, support, bitwise replay and a
deterministic round-to-nearest bound the exclusion rule. Make the realised norm, spectrum and rank reported diagnostics, and
add the realised energy ratio to the registered nuisance sensitivity. Computed yield 40/40 (§2.1). Alternative if Board A
rejects F10: (e), the registered reading.

### 3.2 Draft text (to append to `docs/PREREG_H1_TF.md` after Board A review; not appended here)

> #### A5 amendment F10 (DATE, before any G3 outcome): bf16-resolution rotation audit
>
> This block is appended; earlier text is not rewritten. It records chair ruling F10, taken after the Part A staging
> (`docs/prereg/a5_manifests/part_a/`, branch `feat/part-a-prepare`, staging code `7f8b453`; audit file sha256
> `f05fec4265ffd93c8c537100993d23262be95141369240fcad0a7e16ff474d35`) and before any G3, rotation, ladder or sealed
> outcome existed. No variant or reference has been scored. The trigger is the staged audit yield (1 of 40 rotation
> pairs), a function of stored bytes only. The analysis is `docs/design/ROTATION_YIELD_2026-11-11.md`.
>
> **F10.1 — Facts.** Under the A5.3/A5.3.3 realised audit, 39 of 40 pairs are excluded (2,534 tensors, 535 pass). Per
> tensor, the realised discrepancy ε_R = ‖R_eff − R‖/‖D‖ is the bf16 rounding of the predecessor W at the rotation's scale.
> It never exceeds min(1, 1/(√12κ)) by more than 0.24 %, with κ = ‖D_block‖/√(Σ ulp(W)²). On the 2,188 tensors of the 32
> root edits it equals the edit's own rounding ε_D = ‖D − Δ‖/‖D‖ (median ratio 1.0003, 5–95 % 0.996–1.021). Realised norm
> and spectrum mismatches are bounded by ε_R (Mirsky). The 1e-3 rank counts bf16 noise: rank-16/32 LoRA edits have D rank
> 437–1,024. The registered realised tolerances therefore exclude by effective norm and predecessor grid, not by
> rotation quality.
>
> **F10.2 — Construction unchanged.** R = Q_out D_block Q_inᵀ, with registered seeds, the cloned generator state, the QR
> sign convention and rounding at the actual predecessor (A5.3, A5.3.3). The scored object (bf16(W + R) − W), the
> prediction ⟨g, D_eff − R_eff⟩, the "spectrum-matched (pre-rounding)" wording, the selection, the 40 pairs, the seeds,
> the yield rule (≥ 15 resolved), A5.5 and the ladders are unchanged.
>
> **F10.3 — Exclusion audit (replaces the exclusion role of A5.3's post-rounding audit and of A5.3.3's "all tensors must
> meet S31's norm/spectrum/rank/support criteria", and F3(ii)'s per-tensor pass rule).** Each edited tensor passes if and
> only if all of the following hold. A failure of any tensor excludes the pair with its reason, without replacement
> (A5.3.1).
> (i) *Construction*, computed in fp64 from the stored bytes: orthogonality error ≤ 1e-5; full-SVD reconstruction residual
> of the realised block ≤ 1e-5; ‖R‖ and the sorted singular values of R equal those of D within 1e-5 relative; the
> numerical rank of R (1e-3 rule, A5.3.3) equals that of D. (Staged fp32 maxima: orthogonality 8.7e-7, pre-rounding norm
> 1.43e-7. The five fp32 reconstruction residuals above 1e-5 are ≤ 4.0e-15 in fp64.)
> (ii) *Support*: the realised change of R_eff lies inside the block (unchanged rule).
> (iii) *Replay*: the stored R_eff equals bf16(W + R) − W bitwise when recomputed from the registered seed with the pinned
> code, and round-trips through the scorer's arithmetic (as staged).
> (iv) *Rounding bound*: |R_eff − R| ≤ min(|R|, ½·ulp(W + R_eff)) entrywise. This is the round-to-nearest guarantee; a
> failure means a wrong grid, predecessor or dtype.
> (v) *Sub-resolution guard*, per edit: the pooled realised norm mismatch |‖R_eff‖ − ‖D‖|/‖D‖ over all edited tensors is
> ≤ 0.5 (the rotation keeps at least half the edit's norm after rounding). This is fixed a priori; the staged maximum is
> 0.031.
> Existing pre-audit reasons (`non_2d`, `non_finite`, `zero_block`, `rank_degenerate`, `not_single_matrix`) are
> unchanged. New reason codes: `construction_failed`, `support_failed`, `replay_failed`, `rounding_bound_failed`,
> `sub_resolution`.
>
> **F10.4 — Realised diagnostics (reported, not exclusion criteria).** Per tensor and pooled per edit: the registered
> realised norm mismatch (signed), spectrum mismatch and rank at 1e-3; the rank above max(1e-3 s₁, 3τ) for D and R_eff,
> with τ = √(mean ulp²/12)(√m + √n); ε_R, ε̂ = min(1, 1/(√12κ)), κ, ε_D (roots) and the energy ratio
> ρ = ‖R_eff‖²/‖D‖². "Spectrum-matched" refers to the pre-rounding construction. The realised rotation equals the exact
> rotation up to bf16 rounding of W, of the same size as the edit's own rounding. No realised spectrum match is claimed.
>
> **F10.5 — Sensitivities (descriptive; the decision uses F10.3's set).** (a) The A5.3.3 nuisance-controlled sensitivity
> adds log ρ (pooled per edit) as a covariate, with train-only standardisation and penalty 1 as registered. (b) Agreement
> and its CP interval are reported by norm tertile. (c) They are also reported on the 30 pairs whose staged pooled
> realised norm mismatch is ≤ 2e-2: all except `0a841d4082532e8b`, `0bc0a0c4ad0520e0`, `0d055869527177b4`,
> `1ab850a41051fe45`, `1fc51f7164052d50`, `2d1de332e36d8f00`, `453d8c08966438e0`, `455b881d5de4e6e0`,
> `61d1f61e25212068`, `649bd035e79488e5` (all low tertile). (d) The registered constant-sign comparator is shown next to
> each. (e) They are also reported on the 34 pairs that pass the noise-referenced realised audit (norm ≤ 2e-2 + ε̂,
> spectrum ≤ 5e-2 + ε̂, rank above max(1e-3 s₁, 3τ) within ±1): all except `00f8111cbad8218f`, `0a841d4082532e8b`,
> `0bc0a0c4ad0520e0`, `1ab850a41051fe45`, `61d1f61e25212068`, `649bd035e79488e5`. None of these can change A5.5's
> outcome.
>
> **F10.6 — Simulations.** If all 40 pairs pass F10.3, the actual audit state equals the audit assumption of the
> registered NULL-SIZE and POWER-AND-YIELD runs (F2, F7). They stand without a rerun and remain conditional on the F2
> proxy (Board A F7 conditions 2 and 4). If any pair is excluded, then before scoring, POWER-AND-YIELD (with its NULL-SIZE
> pass) is rerun as a labelled run with those selected slots masked in every replicate. The ≥ 0.80 ordered-power and
> ≤ 0.07 size gates apply to that run (A5.3.2). A failure makes Part A not decisive and needs a further amendment before
> scoring.
>
> **F10.7 — Execution.** The F10 audit is implemented in `src/wsp/eval/response_intervention.py` and
> `src/wsp/features/rotation.py`, with tests, and pinned by a re-pin (xv) before the re-stage. The rotation variants of
> all 40 edits are rebuilt under that code. The R bytes of the staged passing pair must be bitwise unchanged; the 80
> ladder variants and the reuse proofs are re-hashed, not rebuilt. The Part A manifest and audit file are regenerated,
> and their hashes enter A5.1.2. No variant is scored before Board A accepts F10 and the re-stage report.
>
> **F10.8 — Exposure.** F10 is pre-outcome only if no decision-maker who drafts or approves it has been exposed to any G3,
> rotation, ladder or reference score, or to a summary of one (F8a.1(c) definition). Part A scoring, including the 80
> ladders and the 19 missing originals, starts only after F10 is accepted or rejected.

### 3.3 Order of operations

| step | what | who | cost | gate |
|---|---|---|---|---|
| 1 | This note and the F10 draft | design | done | – |
| 2 | Chair adopts the F10 text (append block, attestation of no exposure) | chair | – | attestation that no G3 score exists (staging manifest `scored: false`; the scorer has not been run) |
| 3 | **Board A review of F10** | Board A | – | accept, or reject (then (e)) |
| 4 | Implement the F10 audit and tests; `--slot-mask` in `a5_simulation.py` only if needed; re-pin (xv) | builder | hours (dev) | tests pass; pins verified |
| 5 | Re-stage the 40 rotations under F10 (CPU, `agent-tmux`, 4 threads) | builder | **≈ 1–5 h** wall (§3.4); +35.8 GB disk (404 GB free) | F10.3 yield; bitwise check of `00f8111c` |
| 6 | Re-stage report; if yield < 40, slot-masked POWER-AND-YIELD (≈ 22 min CPU) | builder → Board A/chair | ≈ 22 min if needed | ≥ 0.80 ordered power |
| 7 | Chair rules on the chain-ladder convention flagged in the staging README (s·D_t for the 8 non-root chain picks) | chair | – | needed before scoring regardless of F10 |
| 8 | **Score** 40 rotations + 80 ladders + 19 originals = 139 entries (exclusive GPU, `agent-tmux`) | builder | 139 × 1.04 = 144.56 GPU-min | staged bytes re-hashed |
| 9 | Close the ledger, then the single analysis (A5.3 "Analysis time") | – | – | – |

### 3.4 Re-stage price

* **Upper bound: 4 h 58 min 33 s.** The full staging of 40 edits × 3 variants, with SVD audits of all 2,534 rotated
  tensors, writing 73.0 GB, at load 10–26 (README). The F10 re-stage repeats only the rotation third, plus fp64 audit SVDs
  (0.74 s against 0.60 s fp32 per 1024 × 3584 SVD here, ≈ 1.2×).
* **Measured-rate estimate: ≈ 1 h.** My all-tensor run rebuilt R, rounded it and took 3–4 SVDs per tensor for all
  2,534 tensors in 36 min 45 s wall (2,196 s of per-edit compute) at load 9–10, with frames cached per shape. The re-stage
  adds two QRs per tensor (staging redraws the frames, up to 3584 × 3584), fp64 audit SVDs, and writes about 35.8 GB for
  the 39 new rotation deltas (the sum of their s = 0.5 ladder files).
* Disk: +35.8 GB under `data/runs/a5/part_a/` (404 GB free on `/`).

### 3.5 May the 80 ladder variants be scored before F10 is accepted?

**No, not without compromising F10's pre-outcome status.**

1. *Exposure.* F8a.1(c), Board A's operative definition, counts as exposure any message, document or log line conveying
   a G3, rotation or ladder outcome. An amendment is pre-outcome only if no person, session or process that makes or
   approves the choice has been exposed. A Part A scoring run writes TF-v1 records and run logs. Under F8a.1(c), any of them reaching the chair or
   Board A, who must approve F10, is exposure. A5.3 also restricts variant-outcome access until ledger closure.
2. *Information channel.* Ladder outcomes are not ancillary to the rotation contrast. Step 2 of A5.5 (s = 0.25,
   ≥ 15 resolved, upper bound ≤ 0.6) **overrides a rotation go**. Seeing the ladders would show whether F10 can matter
   at all.
3. *Correlation with the audit choice.* y(W + 0.25Δ) and y(W + 0.5Δ) reveal each edit's response size and therefore which
   rotation pairs are likely to resolve. Every candidate rule here (κ, norm caps, coverage) is a function of effective
   norm, so a rule chosen after the ladders could select pairs by expected resolution.
4. *Mechanics.* The staged `score --manifest` command also scores the 19 missing originals y(W + D), the left side of every
   rotation contrast d_i. They must not be scored first either.
5. *Cost of waiting.* None to the science. The ladders and references cost 104 GPU-min whenever they run, and the
   rotations must be scored in the same campaign in any case (one analysis after all 120 scores).

The only exposure-safe variant would be a sealed run under an F8a-style protocol after F10's text is frozen (committed and
hashed), with attestations that no decision-maker reads outputs until Board A rules. Part A has no such registered
protocol, and the saving is under two GPU-hours. I recommend against it.

---

## 4. Limitations

* The (a1) probe and the c ∈ {1, 1.5} rank probe cover 5 edits (258 tensors). The (a3) and (b) yields in §2.1 cover all
  2,534 tensors (`full.json`, `remedies.json`). The (a1) yield is an estimate from 14 probed failing tensors.
* ε̂ assumes rounding error uniform over ±ulp/2. It overpredicts for κ < 1 (0.337 measured against 0.373 predicted, bin
  [0.5, 1)), because sub-half-ulp entries carry less error. F10 does not use ε̂ as a pass criterion, only as a reported
  diagnostic.
* The second-order bias in §2.3 is argued, not measured: measuring it needs outcomes. F10.5 makes it visible without
  making it decisive.
* Pricing assumes the shared machine's load stays similar to the staging run (load average 10–26 then, 5–16 during these
  checks).
