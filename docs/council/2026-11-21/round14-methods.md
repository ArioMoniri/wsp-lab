# Round 14, methods and tools seat (Q3, Q4)

**BLUF**
1. The proposed study is **L1, "edited or only re-encoded?"**. It runs on the M′ bank's states, with no extra models and no downloads.
2. Arm A (primary) asks whether a detector can tell real functional edits from function-preserving re-encodings. sha256 cannot, by design.
3. Arm B (secondary) asks which of three Gemma-3-4B siblings is the parent, with merges included. A cross-family Qwen3.5 sanity check runs alongside.
4. Detectors: sha256, raw-parameter cosine (PCS), HuRef invariant cosine (ICS), REEF CKA, and our per-tensor K-FAC/J1 features.
5. Prediction, recorded in advance: product invariants beat our R1 features in Arm A; PCS saturates Arm B. A negative is cheap.
6. Cost: ≤ 2 local GPU-h, $0, run after the M′ freeze. Metrics: AUROC and TPR at 1 % FPR, with a cluster bootstrap.
7. Our inert checks and canaries prove that the *editor and runtime* are exact. Given a trusted reference hash, they add nothing to weight provenance over sha256.
8. Q4: a hash-chained per-edit provenance record, a seeded weight sketch, batch-1 canaries, and a watermark note gated on open code and author disclosure.

## Q3. Study L1

**Why not "derived vs unrelated".** That question is already answered by large published margins:
- HuRef: offspring ICS 82.56–99.97 against an average of 0.38 over 28 independent LLMs, with 51 of 51 offspring matched (https://arxiv.org/html/2312.04828);
- REEF: mean CKA 0.9585 for derived models against 0.2361 for unrelated ones (https://arxiv.org/html/2410.14273);
- black-box provenance testing: 90–95 % precision and 80–90 % recall over more than 600 models (https://arxiv.org/abs/2502.00706).

We also lack an independent same-architecture model.

**The question sha256 cannot answer** is whether a checkpoint was *functionally* edited or only re-encoded. Our inert controls give exact negatives: compensated (up×2, down×½) and (v×2, o×½) pairs give ΔNLL exactly 0 and bitwise logits (`docs/reviews/wmm/wmm-gate0-2026-11-19.md` §3; `src/wsp/wm/wmm/controls.py`). HuRef designs its invariants against invertible (W_Q, W_K) and (W_V, W_O) maps and against permutations. REEF reports CKA 1.0000 under permutation and under 0.8 scaling. Arm A tests these defensive claims on our models, with the transform set fixed from existing controls; we search for no transform that defeats a detector.

**Arm A (primary).** The reference is MedGemma 1.5 4B at its pinned revision (`src/wsp/wm/gemma3.py` TARGETS).
- **Negatives:**
  - 200 seeded compensated-pair combinations on random layer subsets;
  - one bf16 → int8 per-channel round trip (plain torch);
  - the uniform `o_proj`/`down_proj` ×2 rescale, which changes NLL by +0.0103 against 12.3 for noise of the same norm (Gate 0 report).
- **Positives:**
  - the ≥ 30 M-A matched-norm direction units (`RULING-wmm-gate1.md` item 2);
  - the M-SYC LoRA at λ = 1;
  - the Gate 0 ladders (tv, interp, h1/syc LoRA), rebuilt from recipes. `gemma3.WeightLine` reproduces endpoints bitwise.
- **Labels** come from recipes only, never from M′ outcomes.

**Arm B (secondary).** The candidates are the same positives. The references are MedGemma 1.5 (the true parent), MedGemma 1 and Gemma 3 4B it. These are siblings: ‖θ₁ − θ₁.₅‖ = 389.7 and ‖τ‖ = 435.5 (Gate 0 report). The interp states at α ∈ {0.25, 0.5} have two true parents, which tests REEF's merge claim; Evollm-jp scored 0.9374–0.9635 there. Sanity arm: cached Qwen3.5-0.8B and 4B, compared by D2 and D3 only (HuRef's K×K terms are size-independent).

**Detectors.**
- **D0, sha256** (`actions.state_hash`). It must fail Arm A, and serves as the floor.
- **D1, PCS.**
- **D2, HuRef ICS**, from the paper's Eq. 8: X̂W_QW_KᵀX̂ᵀ, X̂W_VW_OX̂ᵀ and X̂W₁W₂X̂ᵀ, over the last 2 layers and the K least-frequent tokens, with frequencies counted on the cached WikiText.
- **D3, REEF linear CKA** at one mid layer on 200 cached-WikiText prompts. The paper says the score is stable after 200–300 samples.
- **D4, ours:** K-FAC-weighted ‖Δ‖ per support writer from M-A's base field at θ0 (`src/wsp/wm/collect.py`), plus J1. These are R1 under `src/wsp/wm/access.py`.

**Mechanism.** For a compensated pair, Δo·v + o·Δv = ov/2 ≠ 0 to first order, so per-tensor linearised scores should flag exactly inert pairs; HuRef's products are invariant.

**Metrics and nulls.**
- AUROC, with TPR at 1 % FPR co-primary (2 false positives of 200), plus TPR at 5 % with Clopper–Pearson intervals. Primary contrast: D2 vs D4, paired. Arm B: top-1 parent accuracy.
- Nulls: label permutation (10⁴); replicate reloads must give identical scores.
- REEF is trivially 1 on exactly inert negatives (bitwise logits), so its quantisation and rescale negatives are reported separately.
- Cluster bootstrap over bank components (`wm.wmclosure`) and negative seeds.

**Power** (my Hanley–McNeil arithmetic at assumed AUROCs, not data; independence assumed, clustering widens it): 60 positives and 200 negatives give SE 0.036 at AUROC 0.8 and 0.028 at 0.9; 30 positives give 0.050 and 0.038. Fix the minimum detectable paired ΔAUROC by CPU simulation (`powersim*.py` pattern) before the freeze. Kill L1 if the 95 % half-width at 30 positives exceeds 0.10.

**Cost.** D0–D2 are weight arithmetic; D3 is one 200-prompt forward pass per state; D4 reuses M-A's R1 passes. Cap ≤ 2 local GPU-h, ≤ 16 GB, $0, priced on a 4-state smoke. States are realised in memory from M′/Gate 0 recipes; L1 reads weights and WikiText only, never M′ labels or sealed panels.

**A WSR-style reader** is deferred: with full weights, the `tensor_hash` diff already localises edits exactly.

**Tamper-evidence, honestly.**
- `text_state_hash`, theta-hash = base at null, and bitwise restore prove the editor is exact.
- The logits canary adds *runtime* evidence a weight hash cannot (same weights, different kernels or batches). This matters: batch composition flips 13.5 % of answers (`docs/reviews/wmm/wmm-gate1-2026-11-20.md`).
- For weight integrity against a trusted reference, nothing beats sha256. For lineage after any change, sha256 gives nothing; L1 measures that gap.
- Gap: `gemma3.base_hash` hashes blob names and the revision pin, not recomputed bytes (fixed in Q4a).

## Q4. Process specs

**(a) Per-edit provenance record**, one per realised state, in a hash-chained log. Copy the `runguard.StudyLog` pattern into `wm/wmm/` (`runguard.py` is pinned in `docs/prereg/wme_manifest.json`). Fields:
- unit id, `Action.key()`, bank `recipe_hash`;
- base revision and **recomputed** full-state sha256;
- theta-hash and changed-tensor `tensor_hash`es;
- LoRA adapter sha256, training-rows sha256, seed;
- semantic fingerprint (D1, D2, per-layer ‖Δ‖);
- code SHA (`c1/bundling.py` discipline), library versions, batch-order hash, canary hashes, previous-record sha.

Tests (CPU toy): replay reproduces the record; a one-byte tamper breaks the chain.

**(b) Weight-signature canaries.** Store 64 seeded fp32 Rademacher projections per tensor per reference; check before each unit and after restore. Added value over sha256 is modest: a magnitude estimate for accidental changes (leaked adapter merge, wrong revision, failed restore) that also works across bf16/fp32 loads. Run the behavioural canary at batch 1 in fixed order, recorded separately, so runtime drift is not misread as a weight edit.

**(c) Watermark robustness under benign edits (descriptive, defensive).** No public weight watermark is stated for our models (see Q2), so measure only watermarks **we embed in local copies**. Candidates:
- the invariant-based weights watermark, which was tested on Gemma-2B. Its own reported results include detection as low as 48 % after fine-tuning at a 1 % ranking threshold, and 0.84 for 4-bit multi-user (https://arxiv.org/html/2507.08288v1);
- EmMark (https://arxiv.org/abs/2402.17938).

Neither fetched page showed a code release. Rules: run only with author-released code under a permitting licence, else drop; the edits are the frozen M′ bank, never chosen adaptively against the detector; report per edit class only ("rank-8 LoRA at λ = 1 degrades scheme X").

Disclosure norms: notify the scheme's authors before any public mention; release no watermarked or derived MedGemma weights (HAI-DEF terms, seat 5); publish no parameter-level degradation recipe; log the disclosure date.

## Decisions

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Run L1 Arm A (edit vs re-encode) | sha256 cannot answer it; inert controls give exact negatives; tests HuRef/REEF claims | separate exploratory pre-declaration before M′ data; 4-state smoke; CPU power sim | ≤ 2, $0 | half-width > 0.10 or smoke > 2 GPU-h → D0–D2 on CPU only |
| L1 Arm B (sibling parent, merges) | ground truth is ours; PCS expected to saturate | prediction recorded in advance | ~0, $0 | PCS AUROC ≥ 0.99 → close "R1 helps lineage" as a negative |
| Cross-family sanity (cached Qwen3.5) | confirms the pipeline | D2/D3 | < 0.2, $0 | drop if > 16 GB |
| Defer the WSR reader arm | full-weight diff localises exactly | revisit for no-reference use | 0 | — |
| Hash-chained provenance record | base hash not recomputed; no per-edit lineage | replay identity; tamper breaks chain | 0 | per-unit JSON with sha manifest |
| Weight sketch + batch-1 canary | separates weight leaks from runtime drift | before and after every unit | negligible | sha256 only, disclosed |
| Watermark robustness note | defensive, descriptive, edits fixed in advance | author code + licence; authors told first | ≤ 1, $0 | no open code → drop |
| C1 untouched | priority, freeze | no edits to wme-registered files | 0 | — |
