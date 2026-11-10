# Board A — A5 freeze review — 10 November 2026

**Verdict: REVISE.** The closed selection/reference package is substantially reconciled; the registered numerical null-size gate passes, conditionally. Item 7 is not complete. Required edits below must precede acceptance of the evidence freeze; this review authorises no G2/G3, GPU job or purchase.

Scope: Board A package review §8 items 3–7; appendix A5, freeze-prep F1–F4 and the original 7 Nov procedural F5, the separate 10 Nov F5.1–F5.7, all named manifests and five scripts, four round-3 reports, and chair SYNTHESIS §§4–5. Requested baseline is `e3c577f`; checkout HEAD is `6f3d5715987b897ed74ea4f71b2f22743502c39f`, whose only difference from that baseline is five lines in `docs/PROGRESS.md`. All reviewed inputs equal the requested baseline. Work was read-only except this report: no builders, full simulation, GPU, data writes or commit. Checks reconstructed selection independently from components, compared pinned code and snapshot evidence hashes, decompressed/count-checked all three traces, revalidated the existing payload and compared the two actual primary/g1b record files. Full model tensors, cached datasets and deterministic end-to-end builds were not replayed.

## Checklist items 3–6

**3 — accept metadata closure, with required record reconciliation (edit 1).** Snapshot closed `2026-11-08T00:59:56Z`, base `19f5cbf781cb0955fdfe482c12ec523cc2873dc6`. G0 records 629 replayed steps, 351 installations, seven rejected/lost states and their reasons; this is closure of an inventory, not successful bitwise recovery of every historical chain. G1 has 14 unique entries including base, with complete bitwise-repeat reports. The regenerated predecessor IDs `b7edbb9f8b339f52` and `86ea1fe0ec777cf8` have primaries from census run `20261107-105825-56ef1597`, not a later separate predecessor-script run; their `repeats/g1b/records.jsonl` equal their primary files byte for byte. The empty predecessor-script logs cannot establish a different chronology. G6 accounts for all 618 census IDs, with zero missing; the larger chain inventory separately accounts for four alias-held records, sixteen legacy-revision drops and seven lost states.

Current closure reports no merges of registered components and forty distinct selected components; conservative historical components remain authoritative despite current-byte splits. The twelve added heretic_ara entries do not join selected components. Forty edits and their actual predecessors are available; two predecessors require alias holders (`983bb1066d0fd006`, `bc6edb1e3c913a10`). Three re-trained selected successors and their predecessors remain flagged, development-exposed, and are not historical bitwise replays. Eight G3 reuse proofs and realised tensor/rotation audits remain correctly staged under F3; pre-G2 donor reuse still requires the actual matching record/holder identity, not an assumed future proof.

Exposure rows enumerate 680 G0 replay uses, 14 G1 scoring entries and 618 G6 census uses, with role/time/hash accounting. Critically, G6 scored **187 entries in 37 old-confirmatory components**. Their old status label cannot restore freshness. Costs disclose replay/shard wall time, CPU charges, failures and approximate GPU time; they are not all additive GPU-hours. The 0.05 h predecessor-primary line is already inside the 11.2 h census. The earlier G0/G1 history is not retrospectively dated after the procedural freeze.

**4 — accept.** Independent metadata reconstruction gives the same forty IDs and forty component IDs, LoRA 10/10/10 and chain 4/3/3. Representatives remain 60 LoRA and 39 chain, with precisely the four registered `(norm,id)` cutpoints. Sorted-ID hash is `e2bd6cb2956c0fbe64194b3f1ebbb481248ecec45b8a729a59a54b9a2bac9d6b`. The positive lost-predecessor proof replaces error-string inference. References give **n_D=19, n_P=0**, 19.76 additional minutes, 144.56 total minutes for 120 variants plus references, excluding preprocessing/retries. Missing old records do not mean unavailable current edit bytes. All presently pinned code hashes match disk; every manifest/report/raw-run hash in the snapshot evidence block matches disk. Full feature/record content indices are now recorded, rather than feature IDs alone.

**5 — accept within the certified scope.** All six panels report zero intersections with their enumerated checked sets. MMLU/HellaSwag reserve all 500 groups (512/589 member IDs), field prefix exactly 64. Harmful refusal is 20, harmless 64, probe 64, with extraction fill rejected. WikiText reserves eleven whole articles including tails, excludes twelve development articles and supplies 64 windows; the remaining 37 articles are only candidates. Article grouping remains relevant to any population inference. Future-seal statements are explicitly obligations, not present certificates, and F4 requires an actual exclusion-consuming builder/certificate before G7 qualification. Seed vectors are algorithm checks, not measured destruction eligibility.

**6 — numerical registered gate passes; evidence is limited and incomplete (edit 2).** Payload validation independently returns valid: 796 inventory rows, 179 eligible, 141 donor pairs from 63 components, zero missing windows and **423 unavailable audit entries**. Registered individual two-sided size is .0372 and ordered-go rate .0180, both ≤.07. Trace counts reproduce 180 go / 1,939 alternative / 7,881 unresolved. Alternative rate .1939 has exact two-sided 95% CP interval **[.18619,.20179]**. Noise-doubled counts reproduce 162 / 1,884 / 7,954. Audit-fails has 10,000 unresolved and no resolved pair; its zero rejection rate is vacuous, not validation of feasible execution. Coverage around .938 is a reported conditional-coverage diagnostic, not a newly invented passing coverage threshold.

All contrasts rescale the same donor block, so their identical sizes are **one check, not three independent validations**. LoRA-low has one donor component for ten simulated slots; independently signed copies implement the registered DGP but do not add empirical donor diversity. Means dominate roughly .004-nat within-pair noise, explaining yield 1.0 even after doubling noise. None establishes observed rotation covariance, actual ladder-specific resolution, real audit yield or post-G2 power. The constant-positive-mean and cell-specific mask/noise sensitivities explicitly retained by F2 were not implemented/run; an honest `not_run` field does not complete that requirement.

**Alternative-branch gate — explicit decision.** No separate ≤.07 chance-null gate is required for the registered **funding-only** alternative branch or for positive-go error control. Under chance agreement p=.5, a CP upper bound ≤.6 is not a wrong-sign rejection: .5 lies inside the region this resource rule is intended to stop funding. Calling .1939 an inflated type-I wrong-sign error confuses two nulls. However, **before calling the alternative branch a decisive statistical exclusion of useful agreement, it needs its own calibration at the boundary null p=.60, with false-alternative rate ≤.07 at nominal .05**, including resolution/masks and branch order, plus prospective power. The current p=.5 run does not supply that gate. A decisive wrong-sign claim is not registered at all. Preserve F5.7's funding-only interpretation; neither .1939 nor go+alternative=.2119 is the registered scientific rejection rate. Clarify the older “whole-decision error” phrase accordingly.

## F5 clause decisions

| Clause | Board A decision | Reason / required completion |
|---|---|---|
| F5.1 | **Accept with caveat** | Correct descriptive/exploratory fence, authentic adapters versus realised-SVD distinction, source-grouped nested selection, same-row comparator, paired conditional/joint ratios and no purchase/promotion. Eligibility is not 386 independent examples. Complete dimensions/frame conventions before fitting (edit 3). Family-only upper-bound rule is bounded descriptive retirement, not fresh scientific rejection. |
| F5.2 | **Accept with caveat** | K+F:F and K+F:K remove the respective entire information channels and give augmented StatNN the same scalars. No funding fallback; retain this fence. Fusion dimensions/counts must be pinned with F5.1 (edit 3). |
| F5.3 | **Accept with caveat** | Fixed support/radius, twenty seeds, shared base, complete ledger and cap are sensible diagnostics, never a gate or causal scale comparison. A 0.8B qualification does not qualify 4B. “Under A3” does not itself instantiate the required 4B paired qualification panel/precision procedure (edit 4). |
| F5.4 | **Accept with caveat** | Four separate access flags and 32 iid draws with nested 1/8/32 opportunities are operationally meaningful, as are actual-incumbent acceptance and base-relative rebuilds. Frozen seed order is requested but not actually defined; seed keys, support/norm matching, winner/tie/failure policy and acceptance-slice exhaustion need executable pins before G5/G8 (edit 5). |
| F5.5 | **Accept with caveat** | Conjunction, validity first, both interval procedures, invalid-control exclusion and unresolved default are well posed. Rotation upper ≤.6 is explicitly required; ladder alternative alone cannot kill this line. Finite contrast enumeration and a power calculation for the actual stopping claim are still incomplete (edit 6). Retrospective Part B prevents “final word” from meaning scientific falsification. |
| F5.6 | **Accept** | Corrections are consistent with the four reports: withdrawn unsupported speed claim, 8/8 random versus 7/8 original, 121+1 authentic adapters, dense chains. This review checks internal attribution, not a new external-literature audit. |
| F5.7 | **Accept with caveat** | Accurately discloses the single-block proxy, sparse donor cell and high null alternative frequency; funding-only interpretation is correct. Add exact alternative CP intervals for every scenario and the null/claim distinction above, without changing branch order after outcomes (edit 2). |

## Numbered required edits

1. **A5.9/A5.2, A5.1; package README.** Append a dated closure reconciliation: 37 old-seal components/187 census-scored members are exposed, so “unspent/stays sealed” is no longer a current certificate. Preserve historical wording as historical. State the actual two-predecessor primary/g1b chronology, forty-byte availability and alias staging, and that cost rows overlap and mix units (in particular predecessor primaries are a subset of census). Do not silently turn approximate wall/CPU figures into additive GPU charges or call costs fully reconciled cash spend. Existing snapshot content can remain immutable; link its evidence.
2. **A5.3.2/F2/F5.7; `scripts/a5_simulation.py`, `scripts/a5_nullsize_inputs.py`, `scripts/a5_build_panels.py` and their affected tests/spec/reports.** Define outcome-independent numerical constant-positive-mean and cell-specific mask/noise sensitivity scenarios, run/report them at registered scale before acceptance, or obtain an explicit dated prospective amendment withdrawing/defering them with its consequence for decisiveness; a missing implementation is not a run. Add alternative counts/rates/CP intervals for every scenario in report JSON/README. Disambiguate positive-go size from the funding-only alternative; retain ≤.07 registered gates. If an inferentially decisive negative alternative is desired, register the p=.60 boundary calibration with ≤.07 false-alternative rate and its power, before outcomes. No such negative inference is presently approved. Refresh dependent code/report/snapshot hashes in dependency order, preserving raw original runs.
3. **F5.1–F5.2/A5.1.** Freeze per-arm token schemas, masks, rank/energy conventions, zero flags, pooling and complete parameter counts; resolve width-4 learned blocks versus width-8 anchored/random blocks in the nominal 75-input head and extra fusion inputs. Pin eigenvalue ties/sign/basis and well-conditioned gauge-audit Q/seeds/tolerance. Do not claim identical representation capacity when widths differ; L:R is a bundled frame/capacity comparison unless prospectively matched. Pin fold assignments/eligibility, inner stopping/tie/refit rules and genuine per-arm augmented-comparator inputs. Code/settings must exist and be hashed before affected fitting; explicitly stage unbuilt secondaries rather than call them implemented at the G2 freeze.
4. **F5.3/A5.1.** Specify the representative 4B edit/reference qualification panel, independent repeats and immutable run IDs, exact actual device/dtype/batch/order/config/model/tokenizer and window identity; no transfer from the 0.8B pilot. Apply paired shared-base/item-block §6/A2 precision/reliability criteria (WikiText δ=log1.05, paired uncertainty bounds ≤δ/2, repeat discrepancy ≤1e-4 nats, reliability lower bound ≥.90, 4,000 draws), with expansion/undefined fail-closed within cap. Specify whether qualification is H100-only or transport: missing second device means single-device scope; claimed transport requires matched centred-paired device error SD ≤.1 between-edit SD. A3 source-change checks are additional, not a substitute. Pin separately defined fast-tier identity if used. Failure cannot become qualified TF-v1 by relabeling; abort/reprice under the 2 h cap before exceeding it.
5. **F5.4 and its pre-G5/G8 adapter manifest.** Freeze per-block draw-seed derivation, effective support/norm tolerance after rounding, permutations and compatibility constraints, development winner objective/ties, invalid-draw accounting (no hidden replacement), and equal evaluation/GPU opportunity accounting for all three prefixes. Log actual capability use as well as four Boolean permissions, with episode feedback history, code/decode identity and calls/costs. Specify untouched acceptance-slice allocation after changed-byte rebuilds and fail-closed exhaustion; reused validation is adaptive development. These are later-gate pins, not missing G2 outcomes or permission to run G5 production rejection.
6. **F5.5 and its pre-fitting power specification.** Enumerate each chart contrast by name/denominator/control eligibility instead of “plus ... contrasts”; chart seeds are currently explicitly unregistered in `control_seeds.json`. Freeze both procedures' covariance/ratio/guard/failure simulation and decision/power thresholds. Power to reject C=0 at true C=.15 is not power to exclude C≥.05 when C is small: retain the registered .15 superiority-design power, but specify separately the upper-bound exclusion operating characteristic at boundary C=.05 and a prospective small-effect alternative (e.g. C=0), requiring ≥.80 power for both procedures if this is to be a powered line-stop claim. Report uncertainty; missing power remains unresolved. Explicitly include validity/power treatment of (a) and (c) if a scientific negative claim is contemplated; otherwise label the entire conjunction a bounded programme/resource retirement on examined data. No .5 ladder substitution, secondary funding fallback or universal/fresh falsifier.
7. **A5.1/item 7 — final inventory and acceptance.** Follow the exact order and inventory below. Remove stale present-tense “no inventory/not frozen” claims only when replaced by actual evidence; retain historical dates. Supply deterministic regeneration evidence from the fixed closed inputs without writing new governed outcomes. Board A must check the resulting commit before execution; this REVISE is not that acceptance.

## Item 7: ordered freeze inventory and post-commit audit

Order: (i) resolve edits 1–6 prospectively and freeze resolved settings/source identities; (ii) finish any explicitly required sensitivities without G2/G3 outcomes, preserving old raw evidence; (iii) pin final source bytes in `code_hashes.json`; (iv) regenerate inventory/selection/references/text groups, then panels/seeds/null-input spec, asserting unchanged forty IDs/components/cutpoints; (v) validate payload and reports, then write snapshot evidence hashes; (vi) refresh snapshot-dependent selection and both READMEs without altering the historical procedural block; (vii) compute all final SHA256s, write the following explicit path inventory into A5.1 and commit. Do not rerun an entire builder against newly exposed live data just to refresh labels. Avoid hash cycles: payload pins snapshot content indices; snapshot omits files that contain its own final hash; A5.1 is the external final inventory.

The tables below give **current reviewed bytes**, not pre-authorised replacement hashes. Any changed file receives its new literal full SHA256 in A5.1, with old/new identity and actual amendment date. List every package file, including both READMEs and all traces; pin raw payload/outputs by path/hash without committing data files. Expand the source map by literal path/hash or incorporate its separately hashed complete table explicitly; do not pin only five scripts while ignoring imports/settings. This report's completed bytes must also be hashed in A5.1. Do not put a self-hash of `docs/PREREG_H1_TF.md` inside itself: its final identity is the freeze commit's blob (and, if desired, an external detached registration digest).

### Complete committed companion package

| Path | Current SHA256 |
|---|---|
| `docs/prereg/a5_manifests/README.md` | `3bc5132b717a35b4cde20fa723a9a9d68f9e4f04c01dbcd603eb884295d8a5bc` |
| `docs/prereg/a5_manifests/code_hashes.json` | `05a2d244a0d15bb8ef70cb62e89611c783447737b724f755ad98ed9b08ffeabe` |
| `docs/prereg/a5_manifests/components.jsonl` | `6b4eff8d61a4b1c4d83f0c058cb4df5c6c5398e2c6656a8b58b38f8b46251dec` |
| `docs/prereg/a5_manifests/control_seeds.json` | `8e2cbae160e8e8448b60545b0f79c0680f1fdd9ccf005f30d1c04907fe4bd28b` |
| `docs/prereg/a5_manifests/nullsize/README.md` | `a63534e6b16ae7c9167959113594e68266e02f56027c5e4a20ece952d0025174` |
| `docs/prereg/a5_manifests/nullsize/inputs_provenance.json` | `ea160aa4088c56db0aa27046daf3e21eb5865b3950b98e70b4eff9242e1eb1a3` |
| `docs/prereg/a5_manifests/nullsize/registered_traces.jsonl.gz` | `34079bb88397386542261bb311720a1509bca76bb53ca3e93f71cabd05582aab` |
| `docs/prereg/a5_manifests/nullsize/report.json` | `b6ea5d134790090afeec9ad080531965d3f1b4682df3eb88bc6875d589bf9228` |
| `docs/prereg/a5_manifests/nullsize/unavailable_audit_fails_traces.jsonl.gz` | `105ec8aab49b7bb5e9dba5d8034d4c724b62c870252efcd3d9e4a2af2849d95b` |
| `docs/prereg/a5_manifests/nullsize/validation.json` | `7a9f72876e73f5ffba2852f925a8aa3ae52a5c9261d55af942b322b1e0606136` |
| `docs/prereg/a5_manifests/nullsize/variance_x2_traces.jsonl.gz` | `2c7e497ff72223e86e1c24b1e1691cb1af8a6d2e223d54adc131e1573cdfe582` |
| `docs/prereg/a5_manifests/nullsize_inputs.json` | `6e58b589e32491b36fd70e679b20242538ea8e958b9b9072ce0e30b444508885` |
| `docs/prereg/a5_manifests/panels.json` | `c291b68516952c4f342bfc7f36843a83f0482df7d721a00e46287597bad108cd` |
| `docs/prereg/a5_manifests/references.json` | `5b6991f299c1eee049de51bc1e4f00c71c1dfc58114cdf6549fb5d246850564e` |
| `docs/prereg/a5_manifests/selection.json` | `13e5fc6ab78ba1fa08a6a043691605759c1a027c279fc2efd795943053540bb9` |
| `docs/prereg/a5_manifests/snapshot.json` | `030517947267287875ee55847d24732c0e30f0140a83bfd3fdbd9a58264a71c1` |
| `docs/prereg/a5_manifests/text_groups.json` | `f769afcade5a80571daf3a9578b1c03584b46a5abbbba07b32d43a7e9035e7e1` |

### Raw evidence (hash pins only; no data commit)

| Path | Current SHA256 |
|---|---|
| `data/runs/a5/nullsize/inputs.json` | `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de` |
| `data/runs/a5/nullsize/registered.json` | `452c078aae520dfedbcf25a44a9390305612d32ade3793840f3118043e4239e8` |
| `data/runs/a5/nullsize/unavailable_audit_fails.json` | `279c31f8eda4ffef535802befab261ec018bcf363ce9c4dd128e5fb38305790d` |
| `data/runs/a5/nullsize/validation.json` | `7a9f72876e73f5ffba2852f925a8aa3ae52a5c9261d55af942b322b1e0606136` |
| `data/runs/a5/nullsize/variance_x2.json` | `c8d9b0783127d7cba31edeccd3befaaf56df0ad92ee3e776e9191870b8e8745b` |

Additional sensitivity outputs/traces must have explicit names and hashes added before the amended package can be accepted; this review cannot invent hashes of unbuilt evidence. If those scenarios are formally withdrawn instead, do not fabricate output entries.

### Source/config identities (all existing code-map members)

| Path | Current SHA256 |
|---|---|
| `configs/editor_qwen3.5-0.8b.yaml` | `cd53d67beb2cb231351b3e2dcc8bdc76ad2278d0da0cd6f720e08ae749f41ed4` |
| `configs/eval_tf_v1.yaml` | `5c20cabe8e35b4116adeed4df650cf27461e7ab144ef66c0b60c397b13b00317` |
| `scripts/a5_build_manifests.py` | `2f994d7bf2f32110f909732b3cdb430abeaae89be411e636cfb1c99b8a3fbcab` |
| `scripts/a5_build_panels.py` | `f218b72e927a620fa4b1b1a5bc72e6543bb43846a96c5be12377eaae8a2b29f4` |
| `scripts/a5_incremental_norms.py` | `fbea8c8e9ee147e18f6ca0d6f8d8232866ebc417dd1b5acabda7f2105bc26c3b` |
| `scripts/a5_nullsize_inputs.py` | `b09c07a15d9965874e22228477c948a29c359e7b09da560432d663e485229169` |
| `scripts/a5_simulation.py` | `ecde1dd7cb99e2b6dc450da5bcbc229e417e21c75b2b94a2a9db61f6b3870668` |
| `scripts/a5_snapshot.py` | `a7518b416ca21259cef584db28492298de8aa2146ee3c58e0ea165664e6653b3` |
| `src/wsp/eval/batch.py` | `122b0fbac62e3b94d7a5c356bd2f45692a18f96ea1a7811c45763b0863e789c6` |
| `src/wsp/eval/response_intervention.py` | `7cf453f263b92b3dc056109e06766b7dbd3dc606ad463f11aa4ccdec1b73f903` |
| `src/wsp/eval/run.py` | `24d01a9e7cb846da34e63bf157f8a49bde817570ce474e3120999f267e39c69a` |
| `src/wsp/eval/subsets.py` | `5f8b55ad555aa910c3ca2024cc830d5bda479021a0a01de268d5f27a5c26f24f` |
| `src/wsp/eval/tf_v1.py` | `29125dac45ddf7ffa7a490c0daead1ea06236835f09a0f34f7b38579043caef9` |
| `src/wsp/features/response_chart.py` | `f23397779c13d2d63cb9db79f126886612eb3deb67d1d97b433ead7ec5bcc0c6` |
| `src/wsp/features/response_field.py` | `94bbe9b89ee51b8a444ca9859cb3415182c38dca61ce90133df8beef5bb3bd4d` |
| `src/wsp/features/rotation.py` | `f3ff12426ad89dac69073b2d06ca034f83de7257edd248d2a8063a194b1897e3` |
| `tests/test_a5_build_manifests.py` | `ad5ef44bb3549500b0b2588f1ca3497fdc9a5052f5a4a1557a2af76379547683` |
| `tests/test_a5_build_panels.py` | `84f734f3dcdfb97d10087d76ff9787e6a8064119f02843166cbd186d66f800f4` |
| `tests/test_a5_nullsize_inputs.py` | `dadddb13bebaa28ac1c50a63e6e59e425a53c992fea500e123f4d833e0940125` |
| `tests/test_a5_simulation.py` | `c1bae99ae8565143c688bc5dabbee8a979904ae4a8ff7b14318d5128948a4709` |
| `tests/test_a5_snapshot.py` | `85d27f488d489934e703bc77336614ef7882120f39c0de6ec346264a99ac5dcc` |

### Additional fixed selection/panel/qualification inputs

These are transitive immutable input pins, not a request to rebuild or modify them.

| Path | Current SHA256 |
|---|---|
| `configs/eval_subsets/harmless_alpaca_tf_v1.txt` | `026ce60174fb8f227c6dc433d85c0574c06a1caf62e5b7228ad153f4de4f169c` |
| `configs/eval_subsets/hellaswag.txt` | `938f0b691e5d90e705da6ac88586a766c1f5ab00ffa77fe129a0ce546c8ab507` |
| `configs/eval_subsets/hellaswag_tf_v1.txt` | `a48427f88a2ff5dcb02d0f3c02ccb8fc472455ebb336fbe40b48206c45cc706e` |
| `configs/eval_subsets/mmlu_pro.txt` | `45aa5c72467f0cbbee7df5deb5c104fde3a4c880d3a60eea533e709dabce62ac` |
| `configs/eval_subsets/mmlu_pro_tf_v1.txt` | `a7bef4f45d9c888673868999a5a611a8d31d9c8e956347bc43cf4e0ebfaddbea` |
| `configs/p4_predictors.json` | `c217b5a5b012689475609782bfb68d862cbfb4864b1233093be7d355a7e04cfe` |
| `data/runs/d8/a5_incremental_norms.jsonl` | `6d8954f1aefde5b0af63944c1286f73f86d7f5b35dbe2ec2d8d1f470edc7a11a` |
| `data/zoo/qwen3.5-0.8b/artefact_status.jsonl` | `376a829c3d5839604c61319f41123ef3d4e544b05586988f9012c7c7b02c1878` |
| `data/zoo/qwen3.5-0.8b/manifest.jsonl` | `71a396e5bab2e53f8f74b0a8a0c646e1a791ef92bb3cf3f3df5c51c8010140e7` |
| `docs/prereg/panel_manifest.json` | `7c72bfd3280954a82de1161e640a870da5f3a876c03b22247656701ed6af2c67` |
| `docs/reviews/P5/d8-regeneration-2026-11-06/chain_replay_install.json` | `b5f5b713690142b4c700f28886006c870ffdbed86baeb499c70125fee976a6dd` |
| `docs/reviews/P5/d8-regeneration-2026-11-06/chain_replay_pilot.json` | `30e672b4b81c7b52b96fe78d9bdb10d47d38e1065339551a87777a235a5f4621` |
| `docs/reviews/P5/d8-regeneration-2026-11-06/chain_replay_restore.json` | `78f59f514be22fa0315953ea26814610ce2a5301ed04cda7b7063965f41264ea` |
| `docs/reviews/P5/d8-regeneration-2026-11-06/g1_report.json` | `356f4fa15ef961a6ab001d1ef284dc8ba2d1b13b1a62786d16e2cf839f480f2e` |
| `docs/reviews/P5/d8-regeneration-2026-11-06/g1b_report.json` | `45f152b2614348d4038a49f8aaaccc6cce46840202fe1ec90bc0e18f97bd30ff` |
| `src/wsp/predictor/baselines.py` | `c1602d20db9b5e3fdb6558a769b98b495cdb0e256228f745751ebe197c9aa6bc` |

Pin the primary fitting code/config transitively as well: the two last predictor paths above are currently absent from the code map despite A5.8.4 depending on them. Freeze the complete resolved fitting settings in A5.1, not a claim that the generic P4 JSON supplies absent tree settings. Future LoL/H5/4B implementations cannot inherit a fictitious current digest: their explicit pre-use gate must add literal implementation/settings hashes and qualification identities before affected execution. This is staging, not a waiver of the primary G2 code/schema freeze.

Content inventory pins in `selection.json:inputs` must be copied/referenced with their complete paths and literal values: feature content index `af893deab180568858487f4aac9481f249eda1f9acc86b9d8652d563d95ee3fc` and TF-v1 record content index `124e9b9245b78ff480f145a8dd89086e7dae21822e5d46bad06a5d46a8c66064`; snapshot supplies individual file identities, current zoo bytes and exposure/cost/source history. Retain the separately pinned effective-config hash `79d13d456efee587281566168ebd598bcffe136d3c3c7f8f13a05692350014f6`, model/tokenizer revision `2fc06364715b967f1860aea9cf38778875588b17`, dependency versions, qualification-run identities and the post-G2/pre-G3 staging obligations.

**Post-commit Board A checks:**

- Compare literal A5.1 SHA256s with `git show <freeze>:<path>` bytes for every committed path; compare raw-payload/output hashes, trace hashes/decompressed trace identities and transitive file/content indices with the closed evidence. A historical `working_tree_matches` flag is not evidence. Check code-map bytes, resolved settings and source-change qualification; reconstruct selection and certificates and verify reports remain reproducible.
- Preserve the **7 Nov procedural** ID/cutpoint/code identity as historical: its code-map digest `2b93ec10b728c981aaee126d9f4d5e5ded70210b08f842bf8906a454ac900a95` is deliberately different from current evidence code-map digest `05a2d244a0d15bb8ef70cb62e89611c783447737b724f755ad98ed9b08ffeabe`. The later snapshot closed 8 Nov; evidence assembly/acceptance and 10 Nov F5 adoption retain their actual dates. Do not regenerate the historical freeze block with current source hashes or claim it froze later feature/donor/evidence bytes.
- Verify all forty IDs and components/cutpoints, n_D/n_P, current predecessor identity, qualified reference reuse, exposure and old-seal reconciliation, declared null-size scope, sensitivity completion/amendment and gate staging. Field/cache values and power/yield are post-G2 outputs; actual G3 audits/proofs precede affected scoring/reuse. No absent G2/G3 result is certified here.
- Audit the freeze diff against its declared change allowlist: `docs/PREREG_H1_TF.md`, this review, the complete companion-package paths above, and only source/config/test paths explicitly enumerated above when necessary for the required repairs. Raw-data paths are **pins only, never permitted writes in this review/freeze-doc operation**; later scenario execution needs its normal separately authorised output stage. No fixed panel/ranking/qualification input may change under the guise of repinning. Any newly required implementation, settings or sensitivity-artifact path must first be enumerated in the prospective freeze inventory and approved; it cannot be silently absorbed into a directory wildcard. Compare against the actual pre-freeze parent, and separately acknowledge the existing `docs/PROGRESS.md` baseline-to-HEAD difference. **No file outside that explicit list may change.** Hashing a new file does not by itself authorise a scope change.

Acceptance after that audit is a separate Board A act. Current numerical PASS is acknowledged; the missing sensitivities, qualification/execution specification and final inventory cannot be replaced by a blanket freeze approval.
