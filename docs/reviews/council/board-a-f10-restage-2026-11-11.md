# Board A — F10 re-stage report review — 11 November 2026

**Verdict: ACCEPT WITH CONDITIONS for the outcome-free re-stage record. Scoring remains held for every Part A entry.** The report supplies the substantive all-forty evidence required by Board A F10 condition 4 and F10 code re-review condition 4. Two documentary corrections below remain before F11 supplementary staging may build on this state. Acceptance of this record neither passes F10.6 nor accepts F11, which is under separate review.

## Scope and integrity

First ran `cd /home/ubuntu/projects/wsp-lab && scripts/wt.sh new review/board-a-f10-restage`; this review's only written artifact is this file in the printed worktree `/home/ubuntu/projects/wsp-lab/.worktrees/review/board-a-f10-restage`. No commit, GPU, scoring, simulation, construction rerun or write under `data/`. Python checks used `PYTHONDONTWRITEBYTECODE=1`. Nothing under `data/evals` was opened; checks there used path existence only.

`git log --oneline -3` at review HEAD:

```
6514d61 A5.1.2 entries after the F10 re-stage; chair rulings on the report's three findings; progress
98c22a3 merge report/f10-restage (outcome-free re-stage report)
dd31847 Part A F10 re-stage report and A5.1.2 hash note (outcome-free)
```

Read the governing Board A F10 and corrected-code conditions, F10/F10a/F10.6 and the re-pin (xviii), F11's staging/hold context, the report and A512 note, chair additions and progress rulings, both manifests and both audit streams, freeze/launch, mask/provenance, calibration record and allowed simulation outputs. This is a report/integrity review, not another implementation or F11 review.

Independently SHA256-hashed all 71 pinned source/configuration files using their Git bytes at **both `7bacec1` and HEAD `6514d61`: 71/71 match at each**. Selection, references, control seeds and text groups match the freeze at both revisions. The launch records clean execution `7bacec12604f1090168a79a6c6d55d4f548af4ec`, zero dirty files, rotation-only/F10 CPU command, four threads, 02:44:00–06:22:28 UTC, exit 0. The committed freeze precedes launch. The registration's approximately 05:45 entry time is distinct from the actual 02:40:50 freeze timestamp; it is not evidence that execution began after 05:45.

My re-hashes reproduce the full hashes printed in RESTAGE_A512_NOTE and entered by the chair: manifest `59d839e3…`, audits `4ec0928a…`, preserved manifest `d7785b35…`, preserved audits `f05fec42…`, old README `2fcbf517…`, report `61294c86…`, freeze `03dfaf26…`, launch `28dcba68…`, mask `86fa27a5…`, provenance `0bb56b4b…`, calibration record `0c5fa8c3…`. All eleven full hashes occur in the chair's A5.1.2 entries. The regenerated report serves as the F10.7 README; preserving the a533 README is appropriate.

## Independent staged-byte checks

I sampled twelve ladder files, selecting the six smallest files in each flag stratum (unflagged and realised-increment flagged), then hashed the prior passing rotation's two files. This deliberately covers chain increments and ordinary ladder deltas; it is an integrity sample, not random statistical evidence. Each full digest agrees with **both manifests and the pre-execution freeze**:

| Variant | File | SHA256 |
|---|---|---|
| `21e27d70aacb7804` | delta | `ff2a1b9a60c73d4a018b157e19a7fe3fd352140ebaf1ee54fd4cef8b3bb4e780` |
| `bf0e67a749a6f4a0` | delta | `8809e6d97101c97c9e1bbd8df1da1912707301787dbed09cce59756ed38edc47` |
| `12db4caa66fe8ca4` | delta | `56e2452022c39089941b7d466c013b9dd9ac3ff0bdebf8c6ec3e3740fcfdfa4f` |
| `8d5a95f8c96beee0` | delta | `6b038205a739e67e49b6b256a2c84931f90afff4bcda2f1346874f54997391f4` |
| `05a14e9476819235` | delta | `30066b57296b2df49dde73c9f98b9039cbc000cd23fe6849ea4d05a17b9ddff1` |
| `065ec848e52de490` | delta | `2118a10974d01bf3918d77a316d96d3e7c90f8193589e17c8756a3dd1c879e8d` |
| `3177a34add5494f5` | increment | `8c72e43d4d2908718ae41bb041a5b68d49802df73407cf43ceee2f7073361294` |
| `4c96c43a37f1ddb6` | increment | `f421ca9121e3136c509dbc9db33d5e9fda8f72ab417d5a245ea2dc8d2b9d99c8` |
| `69440966f51b6863` | increment | `f397c993d7209c1b82a110bb4f2e555188aeeb4dd3d876f6571c4b31a2f87e8a` |
| `8e307a2ad1f7eaf6` | increment | `dee9d35b92d309f39f8bbacfc73897ebce60e375eff724f43be1b6464f4792ad` |
| `3701979a52879081` | increment | `66f1e21d5a84948baf02dd90c81ee659c0624122e1c7abca8850878cb91c663b` |
| `a27ba8a3c75802ce` | increment | `b7e3d80e83c1fc1aa794dab1f7102af3821058b687a970cc92b65641a5f86ecf` |
| `21c6044695fed38e` | delta | `951e8621f34b7fce8a9fc48884b91bd90806acfd15802afa5c27786a5b4d197d` |
| `21c6044695fed38e` | increment | `78ec4689a5110b7054df3e431cb096df613fb523dc8ab0c9d41766628f2fbe94` |

File names above have suffix `.safetensors`; paths are `data/runs/a5/part_a/qwen3.5-0.8b/<variant>/<file>`. I additionally checked equality of all 96 ladder-file hash entries across both manifests and freeze, and independently re-hashed all eighty ladder `variant.json` files against both manifests. I did not independently re-hash the remaining 84 ladder tensor files; the report records the complete 96-file re-hash and the manifest records its all-equal result.

All eighty ladder entries differ from a533 only by the added flag. Exactly sixteen are flagged, eight at each strength, from exactly eight non-root chain sources, agreeing with metadata's `scale_source: realised_increment`. Rotation exclusion does not discard these ladders. All thirteen carried scoring fields, including 49 reuse proofs, missing-reference lists and pricing inputs, equal a533; selection identities, seeds, ledger and staged zoo also agree. Carrying proofs is not renewed certification of evaluation contents.

## Yield, diagnostics, exposure and price

Independent audit aggregation reproduces **31/40 passing pairs; six chain replay exclusions and three LoRA rounding-bound exclusions; 2,534 edited tensors and 1,984 tensors in passing pairs**. There are 2,455 individually passing rows, 76 replay-failing rows and three bound-failing rows. Whole-pair exclusion also removes 471 individually passing tensors. The 76 replay failures are scorer round-trip failures (95 mismatching entries), despite successful fresh replay bit comparisons. All 2,534 fresh pre-rounding and written-increment replay comparisons are true at four threads. All forty emitted rotation environments agree; excluded rotation directories contain no safetensors.

I checked all forty per-pair table rows against audit/manifest verdicts and edited/rotated/failing counts. I also checked all forty a533-versus-F10 diagnostics rows against both manifests, including displayed numeric precision for signed norm, log energy, kappa, epsilon-hat and available epsilon-D, and all subset/Boolean columns. Preserved and recomputed a533 tensor verdicts agree on all 2,534 keys; the original 1/40 pair and 535/2,534 tensor yield stays a diagnostic. Frozen subsets remain 30 and 34, intersecting passing pairs in 23 and 26. No replacement denominator is warranted.

The price is correct for the original forty: 31 rotations + 80 ladders + 19 missing originals = **130 entries, 135.2 minutes** at 1.04 minutes each. It is not F11's prospective supplemented price.

Both manifests say `scored: false`; this review's existence-only checks found no evaluation path for any of the **120** variant IDs or the nineteen missing originals. This supports the recorded exposure statement, but does not prove historical non-access. The report's author attestation and the guarded rotation-only route provide the additional evidence. Existing per-decision-maker attestation obligations remain in force.

There is one small report error: exactly **93** rows have maximum excess equal to one spacing and pass the rounding check, but only **92** pass F10 overall. The other is `057dcfbedea3251f`, `model.layers.6.mlp.up_proj.weight`, excluded for replay. Thus §2/§8's description of 93 *passing tensors*, including the claim about 2,455 passing rows, needs the distinction in condition 2. Inclusive equality itself remains valid.

## Chair rulings

1. **Freeze mismatch and empty base revision: accept the disclosed documentary exception, with a correction to the hash explanation.** The pinned `numerical_environment()` explicitly uses `hashlib.sha256(torch.__config__.show().encode()).hexdigest()`. My CPU recomputation is `07f0372b5746fc761d40701ab808447f3703337e82f04fcb4846c45e71ba008d`, matching emitted records. This is precisely the UTF-8 derivation the chair attributes to frozen `d1b4e24c…`; the claim that the code derives it differently is therefore unsupported as written. Matching other identity fields, replay and prior-pair bytes support accepting the execution, but do not mathematically prove identical build configuration or recover the unexplained frozen digest. Treat that digest as an unresolved erroneous freeze field unless its actual derivation is supplied. No audit retuning or automatic re-stage is required by the observed evidence. For the base field, I independently read all forty selected `edit.json` records: every revision is `2fc06364715b967f1860aea9cf38778875588b17`, matching the model pin. Accept the chair's wrong-key explanation as a disclosed freeze omission supported by this substitute evidence, not as a populated pre-execution base field.

2. **NULL-SIZE equality never a gate: sound.** F10.6 requires the exact masked-slot simulation and applicable size/power gates, not equality with an unmasked null. Masking may change rotation and ordered decisions; it need not change them in every possible run. Here null go/alternative/unresolved changes from 180/1939/7881 to 192/1939/7869. I compared all 10,000 traces in each null/power report: donor representatives and component signs are unchanged from the unmasked run, as are ladder resolved/agreement counts and ladder rejection rates. Withdrawing the equality claim is correct.

3. **Gross bound failures, not inclusive equality: sound.** The three rows record one violation each, excesses 3,495, 10,970 and 12,834.5 fp32 spacings. Their exclusions stand. They cannot be described as exact one-spacing equality or repaired by invoking that ruling. The later correction also supersedes the report-identified 06:30 wording, irrespective of the chair addendum's reference to 05:20.

## Calibration and required conditions

The F10.6 failure is correctly recorded. I re-hashed the calibration output to `b14756828bc27cbbe14e1e590226820371a65ef5f83ad9622b2e6ccf04a3450a`, verified its four input-provenance hashes and agreement with the calibration record's gates. The schema repair preserves the original `848d7a5` availability object exactly; canonical mapping SHA256 is `b7724a69315d8c8692a128504126251ac16945cf7b741ec60b3cdb6daa5cb329`. Only the nine excluded rotation slots are masked; both ladder contrasts retain forty slots. The failed validation attempt is not a second stochastic calibration or a seed search.

The counted run has seed 20261107, 10,000 replicates, 4,000 resolution draws, 61 power donors and unchanged registered design/stream fields. Ordered power **0.7514**, CP95 **[0.7428065, 0.7598468]**, fails 0.80 even at the interval's upper endpoint. Ordered null size **0.0192** and individual sizes **0.0377/0.0372/0.0372** pass 0.07. The funding-only alternative rate 0.1939 is disclosed, not a size gate. Passing size does not discharge failed power. Under F10.6 and Board A condition 5, the hold is correctly maintained for rotations, ladders and references. F11 is the merged prospective amendment proposed to address this failure; its separate acceptance and execution gates are required before its use. Merge alone supplies no scoring release.

Numbered conditions:

1. **Before F11 supplementary staging freezes this baseline, append a chair clarification of the Torch discrepancy.** Correct the unsupported “different derivation” assertion: either document a reproducible derivation of `d1b4e24c…`, or explicitly record its provenance as unresolved/erroneous and ratify the disclosed freeze exception on the corroborating identity/replay evidence. State the authoritative exact expression and emitted digest for future freezes. Retain the original freeze and hash. Preserve the empty-base omission with its independently corroborated model revision; do not backfill the original file or call every frozen field matched.

2. **Append a report erratum before this record is adopted as F11's baseline:** 93 tensors pass the rounding check at inclusive equality, of which 92 pass F10 overall; identify the replay-failing row above. Preserve the hashed original report and append the clarification with an A5.1.2 reference, or version and re-hash an explicit correction. Do not silently alter the accepted artifact behind `61294c86…`. This is a count/wording correction with no eligibility or calibration change.

3. **Carry this state prospectively under the separately reviewed F11 gates.** Preserve the original forty, nine exclusions, a533 evidence, exact mask/calibration failure, 80 ladders, sixteen flags, reference proofs and no-exposure record. Supplementary staging requires accepted F11 text/code/pins, a new explicit freeze/launch and required attestations; its report must identify this baseline and its corrected documentary exceptions. No retuning, replacement of failed original edits, advance ladder/reference scoring or calibration rerun to rescue F10 is authorized. F11's subsequent calibration/report acceptance and all existing scoring-release conditions remain outstanding.

Conditions 1–2 correct the inherited record; condition 3 is the continuing execution gate. No tensor-byte correction is indicated by this review. The report is accepted conditionally as an honest outcome-free record of an unsuccessful decisive-power calibration, not a certificate that Part A is ready to score.

**Exposure attestation:** Board A ML methods/pre-registration reviewer, session/worktree `review/board-a-f10-restage`, 11 November 2026, from worktree creation through completion of this report. Read only governance/source documents, construction manifests/audits, permitted edit metadata and simulation records; hashed selected staged bytes on CPU. No affected G3/rotation/ladder/missing-reference response outcome or summary was opened, computed or received. No evaluation content was opened and no file under `data/` was written. This attestation covers this review session only.
