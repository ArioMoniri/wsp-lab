# Board A — Part A analysis re-review — 11 November 2026

**Verdict: ACCEPT WITH CONDITIONS as the analysis-code implementation of F10 condition 6 / F10a part B (pin before scoring).** The five implementation corrections from the first review are addressed for the explicit-input CLI and its documented producer/evidence contract. The sixth correction, adoption/merge/re-test/pinning, remains a pre-scoring condition. This review does not release scoring or close the other F10 gates.

## Reviewed object and verification

Board A, ML methods and pre-registration, acting as the requested review sub-agent. Reviewed commit **`2b03bda571bc46bce00b4b8aaf6b9c9c07424139`**, branch `feat/f10-sensitivity-reports`, in `/home/ubuntu/projects/wsp-lab/.worktrees/feat/f10-sensitivity-reports`. Its HEAD matched that commit; the reviewed implementation/test/spec paths had no diff against it. The implementation is not treated as merged.

Read the six conditions in `docs/reviews/council/board-a-part-a-analysis-2026-11-11.md`, the corrected module, CLI wrapper and tests, the frozen sensitivity specification and relevant registration/adoption/input-pin material. Ran in the feature worktree:

- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_part_a_analysis.py`: **138 passed in 8.75 s**.
- `.venv/bin/ruff check --no-cache src/wsp/eval/part_a_analysis.py scripts/part_a_analyse.py tests/test_part_a_analysis.py`: **All checks passed**.
- `.venv/bin/ruff format --check --no-cache src/wsp/eval/part_a_analysis.py scripts/part_a_analyse.py tests/test_part_a_analysis.py`: **3 files already formatted**.

The builder's **968 passed** full-suite result is a builder report, not independently rerun or certified here. No GPU/full-suite test was run by this reviewer. Unit-test outcomes are synthetic verification only, not scientific results.

Created the requested review worktree with `scripts/wt.sh new review/board-a-part-a-analysis-2`; authored only this report in the printed worktree. No implementation changes or commits. No writes under `data/`; no file or symlink target under `data/evals` opened. No new G3, rotation, ladder, missing-TF-v1-reference or sealed outcome, or summary of one, was opened, scored, computed or received. The targeted tests read committed metadata and synthetic temporary inputs. This attestation covers this reviewer's session through report completion and no other actor.

## Disposition of the six prior corrections

1. **Validity first and retained Part B report — addressed.** `decision` checks external evidence, internal predicates and `check_part_b` before inspecting the ladder/rotation decision summaries. Missing or invalid Part B leaves step 1 and every downstream decision unresolved. The supplied Part B object, validity and evidence remain in JSON on every branch, explicitly marked as supplied and not computed by this tool; Markdown reports it too. The tests deliberately exercise invalid Part B against a ladder alternative, rotation go and otherwise unresolved Part A; missing, NaN, infinite, reversed and malformed intervals/points despite true flags; missing diagnostics/evidence; failed draws, insufficient distinct components and ineligible controls. They also check required evidence hashes, report retention, the G2 hard-stop/pending-amendment consequence, exact C = 0.15 and power = 0.80, and independent conditional/joint control failures. Existing strict interval/equality rules are preserved. F10.5 sensitivity outputs are not validity predicates.

2. **Runtime specification verification — addressed.** `verify_spec` reads the file bytes and binds their SHA256 to F10b.1, the module constant and the required CLI `--spec-sha256`. `main` does this before reading the other analysis inputs; a default direct `analyse` call also verifies it. Tests change/remove the spec, change/remove the adoption hash, supply a wrong hash and omit the required flag; the CLI refuses with no JSON/Markdown created. The injectable `spec_check`/`text_groups` objects and reduced-draw API parameters are trusted internal/testing interfaces, not independent certification of operator-created objects. The accepted execution path is the pinned CLI with registered draw counts.

3. **Window identity and re-stage evidence — addressed within the producer/evidence contract; operational binding remains required below.** Schema v2 requires string item IDs `"0"`–`"63"` in registered order, exactly 32 adjacent two-window blocks, aligned per-window token hashes and the pinned `text_groups.json` digest. Reversed, substituted, missing, duplicate, numeric and swapped IDs, wrong/missing blocks and token hashes, and changed text-group bytes are deliberately refused. Re-stage entries require explicit verdicts, exclusion lists, a boolean registered-realised verdict, nonempty diagnostics and a variant-metadata SHA256, with three provenance hashes. Missing/empty/partial/malformed evidence becomes `audit_missing`; contradictions and unknown reason codes refuse. Tests verify these cases, provenance omission and missing validity-evidence links. Missing covariates still affect complete-case models only, without imputation. The code checks evidence structure and hash links; it does not independently rerun the accepted audit, validate every F10.4 diagnostic, or prove a producer aligned its numerical arrays correctly. The sealed-ledger producer and trusted pre-score input inventory must certify those facts. Repeating a hash in the ledger and validity file alone does not establish freezing.

4. **JSON/Markdown contents and fit failure diagnostics — addressed.** Covariate output now carries selected/resolved/complete-case IDs, dropped rows, complete-case agreement/ledger and zero-prediction accounting; subsets carry full ledgers and agreement/disagreement/zero-prediction IDs. Markdown includes the required 3 × 2 tables with per-prediction agree/disagree splits, covariate rules/uncertainty/units, frozen energy member IDs and cut keys with re-stage values/drift flags, and subset accounting. LOO/bootstrap diagnostics retain degenerate-column events and fit failures. Deliberate tests cover non-convergence, forced fit failure, one-minority-row LOO failure and degenerate columns; a deterministic held-out-value test independently computes training moments, transformed held-out values and prediction. Report tests check the 24 required sign tables, complete ID accounting, fixed descriptive authority and frozen interpretation texts. No estimator, subset or tertile gains decision authority.

5. **Conditional F10.6 disclosure — addressed.** Below full audit yield, a hashed, separately labelled masked-calibration report is mandatory. Its rotation-mask IDs and canonical mapping must match the audit exclusions; its hash must appear in validity evidence. The embedded NULL-SIZE row is appended to the five original rows without replacement. Missing report/hash, wrong IDs/mapping/label and inconsistent gates are deliberately rejected; either failed calibration gate leaves step 1 unresolved while preserving disclosure. At full yield, the registered simulations stand and a supplied masked report is refused. This is report ingestion and provenance checking, not an independently performed calibration run; the accepted masked-run producer and gates remain governed by F10.6.

6. **Merge, re-test and pin — still open operationally.** The four reviewed files below are still absent from the inspected `code_hashes.json`; passing its existing-file verification would not prove their inclusion. No accepted execution-commit pin check can be certified before those rows exist and the implementation is merged. The imported `scripts/a5_simulation.py` is already pinned at the current `c4f0c55f…` bytes, rather than the older pre-slot-mask `04f283d8…` bytes from the first review. No new runtime helper file is introduced by these corrections. Complete condition 2 below before scoring.

## Rulings for the adopting note

Board A supports prospectively adopting all of the following, with the scope stated here:

- **Missing Part B:** A5.5 step 1 is unresolved. There is no Part A decision without the mandatory Part B inference/control validity information. This tool retains supplied Part B; it does not fit it. Absence is not eligibility and must not be represented as a computed Part B report.
- **Masked report at 40/40:** refuse, rather than silently ignore it. This prevents an inapplicable run from being mistaken for operative calibration. Below 40/40, require the correctly bound masked report and preserve its gates and extra NULL-SIZE row.
- **Sealed ledger v2:** accept the exact ordered-window/block/token-hash and re-stage provenance contract described above. An outcome-free producer must certify scoring identity and alignment; adoption does not make self-asserted hashes independent evidence.
- **Resolution stream:** accept PCG64(20261107) with one shared `[4000, 32]` index matrix across all analysed pairs/arms/contrasts, using paired differences and linear percentile endpoints. `resolve` calls `sim.resolution` once on the stacked columns. The existing exact-degenerate fast path returns identical endpoints without generating indices when every block column is constant; this is mathematically equivalent and involves no alternative seed or retuning.
- **Flat-step solver tie-break:** accept the existing numerical refinement: normal objective decrease is accepted; a floating-point-flat step within `8 * eps * max(1, abs(f))` is accepted only if maximum absolute gradient decreases. Fixed objective, lambda, tolerance, iteration limit and failure counting remain unchanged. The objective comparison to sklearn and the new failure tests support this acceptance; this is not permission to tune a solver after outcomes.
- **`audit_missing`:** accept as the fail-closed F10.3 rotation exclusion reason for absent/incomplete audit evidence. It remains visible in pair/subset ledgers and contributes to the yield requiring masked calibration. It neither supplies missing covariates nor silently repairs an audit.
- **Wrong subset hash:** stop the whole analysis before summarising outcomes or writing reports. Frozen S30/S34 membership must not be replaced or silently omitted. The S30 mutation test deliberately verifies refusal; the same guard covers S34. S34 remains acceptance of a frozen list, not independent reproduction of its historical audit.

These rulings support the dated adopting note; this review does not itself edit the registration.

## Confirmed pin list and inventory placement

Independently computed SHA256 values match all four builder prefixes:

| File | Exact SHA256 | Pin role |
|---|---|---|
| `src/wsp/eval/part_a_analysis.py` | `419dc07486279a4fe16cdefd90973042596ca12f65028e466172aa7d7d1d9bc9` | Part A analysis implementation |
| `scripts/part_a_analyse.py` | `a47a72f8b6485363b4c9ba8fc600b8564c31dfcb02e0c1a1106ffe734f07395d` | Part A CLI |
| `tests/test_part_a_analysis.py` | `5adaeaf8ea66d8b64ef1e9cfd6964006eaea363ba58d1214d1d03befc6a48b4f` | Analysis/fail-closed tests |
| `docs/design/F10_SENSITIVITY_SPEC.md` | `8624e06343bbb401ff3b6d783572b0c53e1172d0ea3f36aec3768a13701998d4` | Adopted analysis specification |
| `scripts/a5_simulation.py` (retain existing pin) | `c4f0c55fd8e8c9bc5bb6521539f58db996f23e1a3cb37d99a19403e134d2fdf2` | Imported resolution/CP helpers and simulation |

**Yes:** `text_groups.json`, the frozen Part A manifest and the NULL-SIZE report belong in **A5.1.2's input inventory**, rather than being added as analysis-code rows to `code_hashes.json`. This refines the first review's inventory wording; it does not remove their hash-binding requirement. Their bytes independently match:

| Input | SHA256 |
|---|---|
| `docs/prereg/a5_manifests/text_groups.json` | `f769afcade5a80571daf3a9578b1c03584b46a5abbbba07b32d43a7e9035e7e1` |
| `docs/prereg/a5_manifests/part_a/manifest.json` | `d7785b35de1d7041edd2db139591562379db12e33c2532921c814735e8dde8d6` |
| `docs/prereg/a5_manifests/nullsize/report.json` | `1bed2c9531e4230864a1ebbe7e4023e0ffafcf704a59c40e93f9a75bd469dfa6` |

The accepted re-stage report/manifest/README, per-edit variant metadata, frozen prediction file, scoring/panel/config identity and applicable mask/calibration inputs also need trusted pre-score inventory binding. Keeping data-input hashes outside the code-file map does not permit them to remain unfrozen.

## Numbered conditions of acceptance

1. **Adopt the operational choices before scoring.** Append a dated A5 note explicitly adopting the three requested behaviours, sealed-ledger v2 and producer/evidence contract, resolution stream, exact flat-step tie-break, `audit_missing`, whole-run subset-hash refusal, and validity/Part B/masked-report interfaces. Carry the supplied-versus-computed Part B distinction and all existing F10b interpretation restrictions. No retrospective choice from outcomes.
2. **Merge, re-test and pin the accepted execution bytes before scoring.** Add the four exact file/role rows above to the code inventory (and its builder's pin list so regeneration retains them); retain/verify simulation, package/import dependencies and applicable test/environment pins. After merging, rerun the targeted CPU suite and Ruff and satisfy applicable repository checks; then run `scripts/a5_build_manifests.py --verify-pins HEAD` at the actual accepted execution commit and record its successful result plus explicit inclusion of the new rows in A5.1.2. If implementation/spec/test bytes change, recompute pins and obtain review of any material change. Bind the trusted input inventory and producer evidence as described above; copied input hashes alone are insufficient.
3. **Preserve the other release conditions.** Complete the required reviewer/builder/decision-maker attestations in the governed record, outcome-free re-stage acceptance, applicable F10.6 calibration and every remaining F10/F10a release gate. This conditional acceptance closes no other gate and authorises no Part A scoring, including ladders and missing originals, until all applicable conditions are satisfied.
