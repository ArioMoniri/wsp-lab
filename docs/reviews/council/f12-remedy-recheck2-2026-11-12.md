# Independent second F12 remedy recheck — 12 November 2026

**A10: ADOPT. A19: ADOPT.** This supplies the remaining independent unexposed-role adoption for session A under F12a.1(c)–(d). It does not supply the deceased implementer's personal history, verify C1, authorise scoring, or discharge the separate package re-freeze and Board A review requirements.

Fresh review session, no prior participation in F12. Reviewed `prereg/f12-remedy` read-only at `df0265e75b16388112f62800d3f0e225a47d1faa`, including the `7acc6b8..df0265e` changes, previous recheck, original adoption A10/A19, Board A condition 1, entry 24 and second addendum. References below are remedy-head line numbers. This record alone is committed on `review/f12-remedy-recheck2`, created from main with `bash scripts/wt.sh new review/f12-remedy-recheck2`. No remedy file was changed; its tracked status remained clean.

## Decisions and independent negative checks

| Item | Decision | Reasons and references |
|---|---|---|
| A10 | **ADOPT** | `scripts/f12_base_loader.py:143–183` resolves every pinned cache leaf, applies the model/shared-blob boundary and the caller's prohibited-path boundary, opens the resolved leaf with O_NOFOLLOW, copies to exclusively created regular files, and checks both SHA256 and size. Lines 207–249 authenticate the committed pin, require its expected base-state identity, create private temporary staging, load only that directory, require the loaded-state hash and clean up. The production library call at 187–195 uses the local staged path and `local_files_only=True`. Tokens 740–744 and additivity 556–560 both route exclusively through this loader. This remedies the previous directory-only check. |
| A19 | **ADOPT** | Tokens 702–720 and 864–875 refuse any true resume request as the first action of `run`, before input/output checks, reading or directory creation. `main` 1206–1226 only parses arguments then calls `run`. The old checkpoint reader/writer/binding/schema machinery is removed; the retained checkpoint filename only prevents overwriting prior output. This selects the original adoption's expressly permitted governed-resume exclusion. Guard 271–280 rejects overflow floats as well as nonfinite constants. Guard 435–457 guarantees the breach exit even if writing the breach record fails. |

I independently built temporary synthetic fixtures, separate from the submitted tests:

* A synthetic `config.json` symlink pointed from a toy model snapshot into a prohibited temporary directory containing `{}`. Its pin matched those bytes. Both `f12_tokens.load_base` and `f12_additivity_prepare.load_base` raised BoundaryError; the substituted library loader was never called. Thus matching content does not bypass prohibited-leaf refusal. No actual cache or prohibited programme file was used.
* I reconstructed the previous overflow case with every row key, finite 42-column token vectors, matching row/detail IDs and increment, and a checkpoint containing matching line and prefix hashes, but `"a": 1e999`. The new CLI refused it. It also refused a nonexistent output directory. A Python audit hook recorded **zero open or mkdir events during each `main` invocation**, after imports and fixture preparation. There is no surviving `load_checkpoint` to exercise. This is rejection by explicit path exclusion, not a claim that the old row validator was repaired.
* Strict JSON rejected positive and negative overflow, nested overflow, NaN and Infinity. A simulated RSS breach with a deliberately failing breach-record writer still invoked exit code 75. This verifies the stated fail-safe for reporting failure; it is not an OS memory guarantee or proof against every possible monitor/runtime failure.

The submitted tests additionally exercise other-model and dataset-tree rejection, shared-blob symlink chains, mismatched pin/content/state, unpinned and outside Python reads, cleanup and leaf records, and a tiny synthetic model through actual Transformers.

## Confinement judgment

The three-file pin is `docs/prereg/a5_manifests/f12/f12r/base_model_leaves.json:1–21`, digest `6d99ad84c23d451be6800328bb80740ed75abf1eaf8bbd99c19cd3c256ce50f6`: configuration, safetensors index and weight shard. Only authenticated copies enter the loader's directory. The loaded base must equal `18f07d70a89aa63125e7e95535ff2730eb54bae4ee846d72f183e004a740d982`.

**Verified staging is an adequate answer to the disclosed native-open limitation for this fixed model and trusted library path.** Native safetensors reads receive staged regular files, not mutable cache symlinks. The authenticated config/index and successful load-only evidence support completeness of this model's leaf set. This is not a claim that Python's audit hook confines arbitrary native code: it cannot. Nor does the hook make code/system trees inaccessible (`f12_base_loader.py:86–137`). No general adversarial-process sandbox, concurrent hostile directory-renaming protection, or retrospective read telemetry is adopted. The static leaf escape demonstrated by the previous review is closed.

**The shared blob allowance is acceptable** (`f12_base_loader.py:68–72`): it permits storage sharing, not arbitrary new model inputs. Every requested file still needs its exact pinned size and digest and must pass the caller's prohibited boundary; other model and dataset trees are refused even with matching bytes. Hash verification necessarily reads a candidate before accepting it, but the candidate is first confined to these explicitly allowed storage roots. The allowance does not permit following a shared-store symlink outside those roots.

## Numerical identity and proportionate requalification

I independently parsed current files and `git show e7283e2:<file>` with `ast`, removed function docstrings and compared `ast.dump(..., include_attributes=False)`. Confirmed identical population selection, strata, folds, ID/fold hashing, alias resolution, exposure membership and subsets; token `k_frame`, `tensor_frames`, `recorded_rank`, `spectrum_columns`, `balanced_factors`, `k_block`, `probe_panel`, `probe_columns`, `tensor_token`, `row_tokens`, `field_contraction`, `assemble`, `projection`, `save_frames`; additivity `part_id`, `tensor_split_supports`, `column_split_supports`, `restrict`, `check_disjoint_cover`, `construct`, `support_record`, `cross_term`, `split_forecast`, `forecast_entry`, `gradient`, `factors`. Supporting hash/canonicalisation functions also match. This confirms numerical-kernel identity, not whole-script AST identity. Loader, guard, orchestration, manifest and validation functions differ as disclosed; the second remedy's token-run diff removes resume bookkeeping while preserving fresh-run computation.

I independently verified the plan/reference file digests against `f12r/plan_requalification2.json`, and decoded equality of **all 18 entries** (16 LoRA plus two status entries). The entire documents differ only in code, floor binding, intent labels and summary, as reported; whole-document byte identity is not claimed. The run's plan digest matches the current requalified plan, and its reused-code hashes match current files. It records no failed roots, two threads and peak 3,106,230,272 bytes under the 4,000,000,000-byte cap.

I checked `base_load_check.json` against the pin: identical base-state hash, all three recorded leaf hashes and sizes, matching pin digest, 321 tensors, and peak 3,107,717,120 bytes under 4 GB. Current loader/token/additivity/guard hashes match the second addendum's identities (PREREG 7134–7207). These are independently checked committed evidence, not a fresh real-model load in this review. The load record is not exhaustive native-open telemetry.

These cheap checks are adequate for the second remedy's guard-only changes: the plan directly exercises the changed loader and preserves construction entries; the load-only check preserves the exact base state consumed by unchanged token kernels. Together with the prior review's accepted full no-resume token reconciliation, they justify not repeating the heavy token build. I did not open real token arrays. Entry 24's disclosures of full-suite pilot decoding remain the implementer's own history; they are not imported into my exposure statement or used to assert historical blindness. The disclosed re-freeze remains necessary.

## Validation and commit hygiene

Ran in the remedy worktree, serially, with OMP/OPENBLAS/MKL thread limits 2, bytecode disabled and pytest cache disabled:

```
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_f12r_base_loader.py tests/test_f12r_tokens.py
```

**23 passed (11 base-loader, 12 token), 0 failed, 0 skipped, 0 deselected; 2.79 seconds.** Inspected both test files and conftest before execution. Tests use synthetic fixtures and committed manifest metadata, not real token arrays. Four additional independent check groups above passed and are outside the pytest count. No full suite, parallel pytest, real-base load, scoring, simulation or job over 4 GB was launched. No job expected to exceed two minutes was needed.

Only this review file is staged. The doc-only commit bypasses the full-suite hook with `--no-verify`, as expressly authorised; `bash scripts/secret_scan.sh` passed on the staged record before committing. No verification record or remedy change is committed.

## Overall statement for session A

**Yes: every retained choice for session A is now registered or adopted by an unexposed role**, within the itemised scope of the original adoption and previous recheck. A10 and A19 are adopted here; the other six formerly flagged items were adopted by the previous recheck. This completes the session-A adoption component of Board A condition 1. It is not automatic Board A C1 verification, does not rewrite either deceased session's exposure history, and leaves the separate package binding and later review gates intact.

## Exposure attestation — this session only

(i) **Old development outcomes:** I read old-development/qualification summaries and exposure accounts in the prescribed reviews, registration, entry 24 and source constants, including Board A's aggregate qualification/noise/pilot summaries. I did not open underlying TF-v1 root, G0/G1/G6 outcome or D3 outcome records. Synthetic tests computed toy results only.

(ii) **none**. I did not open, compute, score or receive any Part A (G3, rotation, ladder, missing-reference, sealed), F8-E, F8-F or F9 outcome or summary. I opened and walked nothing under `data/evals`. I read no Part A staging, prediction or calibration file, no F12f pilot per-replicate record or decision code, and no real token array. Synthetic prohibited-path tests are not programme outcomes. This fresh session qualifies as an unexposed role under F12a.1(c)–(d).

(iii) **Prescore audits:** I read accounts and quoted summaries in the prescribed review/registration/register material, not underlying Part A prescore inventories, predictions, calibration or audit payloads. This attestation covers this session only.
