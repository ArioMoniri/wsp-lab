# Council round 14 synthesis, 21 Nov 2026: weight watermarking, fingerprinting and our tools

| Seat | Model | Report |
|---|---|---|
| literature | Claude | `round14-literature.md` |
| methods / tools | Claude | `round14-methods.md` |
| challenger | Gemini 3.8 Flash | `round14-challenger.md` |

**Safety boundary, binding:** only defensive detection, verification and provenance. Nothing that removes or evades a
watermark is designed or run.

## BLUF

1. **Our anomalies are not watermarks.** All three seats agree. Each has a standard cause:
   - post-norm rescale cancellation comes from RMSNorm before the residual add (transformers Gemma3 source);
   - batch-dependent output is the known kernel batch-invariance or floating-point-order effect;
   - the headroom ceiling replicates across two model families;
   - (a) fails on dose collinearity (r = 0.996).
   A watermark that preserves function cannot change outputs, and a fixed one cannot depend on batch composition. No
   public statement says Qwen3.5 or MedGemma carry weight watermarks. SynthID Text is applied at generation time, not
   stored in the weights.
2. **Process improvements are adopted now, in study M′, at $0.** They are specified in R2.
3. **A lineage study (L1) is approved as low priority.** It is CPU-first, runs after the PREREG_WMM freeze, uses
   ≤ 2 local GPU-h, costs $0, and yields to C1 and M′.
4. **Watermark-robustness measurement is parked.** It is dual-use, and no fetched scheme had a code release. It may run
   only on watermarks we embed ourselves, with class-level reporting and the authors notified first.
5. **Licence rule.** MedGemma derivatives stay internal and MedGemma is measured passively. Any trigger-insertion
   fingerprint work runs on Qwen (Apache-2.0) only.
6. **A correction to the challenger.** It describes C1 as blocked. That is stale. Phase L passed on 21 Nov (main
   `5060e32`) and the confirmatory run is in progress.

## Rulings

**R1. The watermark hypothesis is closed.**
- Evidence: `round14-literature.md` §2 and §6, and `round14-challenger.md` premise 1.
- We make no claim that any studied model is watermarked.

**R2. Process improvements for study M′.** These are added to PREREG_WMM. They are engineering, not claims.

- **(a) Provenance record.** One hash-chained record per realised edit, kept in `src/wsp/wm/wmm/` and following the
  `runguard.StudyLog` pattern. `runguard.py` is pinned by C1 and must not be edited. Each record holds:
  - the recipe;
  - a base sha256 **recomputed from the loaded bytes**;
  - hashes of the changed tensors;
  - the edit sha256.

  This fixes the gap the methods seat found: `gemma3.base_hash` hashed the blob names and revision, not the bytes.
- **(b) Weight-sketch canary.** 64 seeded random projections per tensor, checked at the start and end of every job.
  This estimates the size of any accidental change.
- **(c) Behavioural canary.** It runs at batch 1 in a fixed order, so that it never fires on the 13.5 %
  batch-composition floor.

**R3. Lineage pilot L1** (methods seat), **conditional and low priority.**
- **Arm A.** Real edits against function-preserving byte changes:
  - 200 seeded inert compensated-pair combinations;
  - an int8 round trip;
  - a uniform rescale.
- **Arm B.** Pick the true parent among MedGemma 1.5, MedGemma 1 and Gemma 3 it, with the interpolation states treated
  as merges.
- **Detectors:**
  - sha256;
  - raw cosine;
  - HuRef invariants;
  - REEF CKA, but only on cached activations;
  - the data-free SVD spectral and subspace baseline the challenger cites (https://arxiv.org/abs/2608.07786). It is
    to be fetched and verified before use;
  - our K-FAC/J1 features.
- **Pre-written prediction.** Our per-tensor features false-flag the inert pairs, while HuRef is exactly invariant to
  them.
- **Metrics:** AUROC, and TPR at 1 % FPR, with cluster bootstrap and permutation nulls.
- **Kill:** CI half-width above 0.10.
- **Reuse:** it reuses M′ bank states, so it needs no new models or downloads.
- **Claim limit:** none beyond a descriptive comparison.

**R4. Watermark robustness under benign edits is parked.** It may be reopened only if all of these hold:
- (i) the scheme's authors have released code;
- (ii) the watermark is embedded by us in a local copy;
- (iii) the edits are fixed in advance;
- (iv) reporting is per edit class, with no degradation recipe;
- (v) the authors are notified before publication;
- (vi) no derived MedGemma weights are released.

## Decision table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Close the watermark hypothesis | Standard causes explain every anomaly; no vendor disclosure | — | 0 | — |
| Provenance record, weight sketch, batch-1 canary in M′ | Fixes the base-hash gap; real tamper evidence | Tests in `wm/wmm` | ~0, $0 | — |
| L1 lineage pilot (conditional) | Descriptive tool comparison on states we own | After PREREG_WMM freeze | ≤ 2 local, $0 | CI half-width > 0.10 |
| Park watermark robustness | Dual-use; no code releases found | R4 (i)–(vi) | 0 | — |
| MedGemma derivatives internal; triggers on Qwen only | HAI-DEF terms; Apache-2.0 for Qwen | — | 0 | — |
