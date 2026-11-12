# Independent static access audit of F12 inputs — 12 November 2026

Audit target: `04127991845e246c2fab77293654b773674f01a2` (`0412799`). All source/test line references below refer to that commit, not current main. This is an independent audit for Board A F12g condition 5 and register entry 22 / implementer row 5. I did not implement or previously review this code. This report is evidence for Board A's adjudication; it does not attest for the killed implementer's session or discharge the C1 condition by itself.

## Findings

**Overall: PARTLY SUPPORTED** for the unqualified statement “the committed F12 inputs code reads no outcome values.” The intended input paths contain no reachable read of a TF-v1 score, per-item evaluation result, new held outcome, or Part A prediction payload. In particular, the 123 existing-root selection uses record membership, not metric values. However, additivity embeds a numerical F8a floor in its output, accepts a qualification disposition that controls construction, and none of these programs establishes an input filesystem sandbox. A narrower claim is **SUPPORTED**: with authentic intended inputs, these entry points do not load evaluation metric values from outcome files. This is a static finding, not a run-history certification.

| Script and tests | Finding | Evidence and qualification |
|---|---|---|
| `scripts/f12_population.py`, `tests/test_f12_population.py` | **SUPPORTED** for no outcome metric-value use on the committed manifests | Whole snapshot bytes are read, hashed and parsed (537–565); primary-record dictionary keys select existing roots (327, 150–168). Ledger `gate`, `id`, `aliases` are used (242–253). No record payload is followed. |
| `scripts/f12_tokens.py`, `tests/test_f12_tokens.py` | **SUPPORTED**, conditional on genuine inputs and resume files | Reads population, stored edit bytes/configuration, pinned base, G2 field/probe cache and its own resume files (646–719). Contracts fields rather than reading measured edited-model performance. `outcomes_read` is a literal (964–965), not access telemetry. No input path containment guarantee. |
| `scripts/f12_additivity_prepare.py`, `tests/test_f12_additivity.py` | **PARTLY SUPPORTED** for the broad claim; **SUPPORTED** for no runtime evaluation metric-file load on intended inputs | G2/base/deltas supply constructions and forecasts. The F8a floor literal (74–76) reaches output (606–607). Optional Astra `qualified` controls acceptance (405–419). No TF-v1 metric field is extracted; no outcome file is required by the default path. |

A copied protocol floor is not a runtime read of its source file. Nor is a Boolean qualification disposition a continuous performance score. Both nevertheless prevent interpreting “no outcome values” as “no numerical or decision information from upstream audits whatsoever.” This report does not establish the floor's underlying provenance or whether Astra qualification itself used outcomes.

## Method, restrictions and validation

`scripts/secret_scan.sh` ran first in the canonical checkout and returned `secret_scan: clean`. The pinned source was inspected in the detached worktree created by `git worktree add --detach .../.worktrees/review/f12-inputs-audit 0412799`. No checkout-over-current-tree command was used. Repository source was read statically; Python standard-library AST/JSON inspection did not import or execute the target modules. JSON inspection of the committed snapshot printed schema/field types and index shape, not outcome metric values. The current token manifest was read through git, restricted to its claim and code pins; it is absent at the audit target and was produced later.

No tests, model loading, evaluation, simulation, scoring, GPU work, installation or heavy job ran. No `pytest` or parallel test worker was started. Nothing under `data/evals` was opened or listed. No file under `/home/ubuntu/wsp-data` was opened, and no listing there was needed. The detached audit checkout's data contents were not examined. The only tracked change is this report in a separate worktree made from main with `bash scripts/wt.sh new review/f12-inputs-audit-report`, whose branch was renamed to the requested `review/f12-inputs-audit` before committing.

The documentation-only commit uses `git commit --no-verify` to bypass the hook that runs the full suite, as explicitly authorized. The secret scan is also run on the staged report before committing. This is not a passing-test claim. No new session file is committed because the request restricts the commit to this file.

## Reachable path inventory

Notation: `R` is the script's resolved repository root; `M` defaults to `R/docs/prereg/a5_manifests`; `Z` defaults to `R/data/zoo/qwen3.5-0.8b`; `G` defaults to `R/data/runs/g2`; `O` is the selected F12 output directory. Read includes hashing, parsing, tensor/header loading, and existence checks; writes are identified separately. There is no filesystem glob or recursive directory enumeration in the three F12 entry points. Safetensors key enumeration is tensor-header enumeration, not a directory listing.

| Path or path expression | Class; operation | Pinned references |
|---|---|---|
| `M/components.jsonl` | Population/component manifest; entire bytes/JSONL, SHA256. Metadata used: `id`, `component_id`, `family`, `parent_id`, `component_status`, `bytes_present`, `bytes_via_alias`, `eligibility`, `exactness`. No norm or metric-based selection. | population 143–168, 213–233, 325–379, 537–565 |
| `M/snapshot.json` | Outcome index/snapshot and embedded exposure ledger; entire bytes hashed/parsed; semantic projections below. | population 242–253, 327–338, 537–565 |
| `M/selection.json`, `M/selection_f11.json` | Population/Part A exclusion manifests; entire bytes hashed/parsed. Uses `registered.component_ids`, `supplement_ids`, and `components_jsonl_sha256` pin; no staged edit or prediction payload. | population 143–147, 537–565 |
| `--out`, default `M/f12/population.json` | Population/fold manifest, write only (parent mkdir). | population 568–583; defaults 43–45 |
| `--population`, default `M/f12/population.json` | Population/fold manifest, full bytes/JSON; token rereads for row count. Population/fold hashes, row IDs and byte holders affect computation; not an outcome table. | tokens 652–675, 943; additivity 547–548, 360–368 |
| `Z/<row.id>/edit.json` | Stored edit descriptor; full JSON then family/rank fields. Rank affects row eligibility and tokens. | tokens 233–263, 709–712 |
| `Z/<bytes_holder>/delta.safetensors` or `lora/adapter_model.safetensors`, `lora/adapter_config.json` | Stored edit bytes; existence/header-key count, tensor load, LoRA rank/scaling/config. `edit.json` of holder also existence/hash checked for additivity. | tokens 463–470, 708–714; additivity 346–375, 381–394; apply 58–110, 113–126 |
| `G/g2_manifest.json` | G2 field/probe metadata; bytes hash-checked before JSON use. `fields[wikitext_nll]` status/identity/files; tokens additionally `probe_cache`. No chain-exact outcome path followed. | tokens 539–605; additivity 252–286 |
| `G/fields/wikitext_nll/{manifest.json,tensors.pt}` | G2 field: file hashes, full identity/content hashes, gradients and K-FAC factors. These numerical field inputs reach a/q/tokens/forecasts and eligibility; they are not evaluation outcome records. Additivity plan reads only the manifest's `identity.state_hash`; build/dry-run maps tensor payload. | tokens 549–575; power-fields 420–427; response-field 299–311; additivity 252–306, 551–559 |
| `G/probe_cache/probe_cache.json`; `G/probe_cache/<row.file>` | G2 probe cache; full metadata, hashes, tensor payload. Metadata `index`, `file`, `content_hashes`, `n_tokens`; tensor `layer.<n>` values feed probe columns. `row.file` is unconstrained path text. | tokens 473–508, 682 |
| Hugging Face `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17` | Base model; `AutoModelForCausalLM.from_pretrained` reads library-resolved configuration/checkpoint/cache files (may download); state hash checked. Exact cache filenames depend on external libraries/environment, not fixed here. | power-fields 76, 430–443; tokens 683; additivity 559 |
| `O/rows.jsonl`, `O/detail.jsonl` | F12's own prediction/token staging/resume outputs; read in `--resume`, otherwise truncate/write; whole rows retained and used, not merely IDs/hashes. No resume provenance authentication is implemented. | tokens 688–705, 725–761, 852–892 |
| `O/{kframes.safetensors,kframes.json,tokens.safetensors,aq.json,pricing.json}` | F12 feature/prediction outputs; writes then hashes/existence checks. Also hashes rows/detail files. No Part A staging/prediction input. | tokens 609–640, 786, 850–891 |
| `--manifest`, default `M/f12/tokens_manifest.json` | Run metadata output, write; unrestricted destination. Claim is a constant, not an instrumentation result. | tokens 964–979 |
| Optional `--astra-record`; its JSON `entry_dir` | Other: upstream qualification/construction record (could itself contain outcomes; full JSON parsed); selected fields `qualified`, `entry_dir`, `entry_id`. Full source delta and holder hashes read from supplied directory. Qualification Boolean gates construction; metric fields are not selected or copied. | additivity 395–422, 346–375, 641–646 |
| Optional `--c4-delta` with `--c4-id` | Stored edit bytes from arbitrary supplied safetensors path; values and SHA256 read, tensor name checked; feeds split/forecast. | additivity 424–440, 560–563, 641–650 |
| `--target/<part_id>/{delta.safetensors,variant.json}` | F12-I construction staging; write and reread for file hash. `--out` or default plan/target `manifest.json` is output only. Targets restricted to `/home/ubuntu/wsp-data/runs/a5/{f12_inputs,f12_additivity}` in non-plan modes. | additivity 64–70, 333–343, 502–537, 625–628 |
| `R/<CODE_FILES>` in tokens 119–128 and additivity 79–86 | Other: explicit source-file SHA256 reads; tokens includes population, Part A/power helpers, response-field/chart, zoo apply/arch; additivity includes itself, Part A/power, response-field/rotation, apply. No path recursion. | tokens 511–523; additivity 540–541; power-fields 90–95 |
| Git metadata / current worktree source status | Other: `git rev-parse HEAD`, `git status --porcelain -- <CODE_FILES>`; subprocesses do not run scoring or list eval directories. | tokens 514–519 |

## Existing TF-v1 roots, snapshot and ledger dataflow

`load_inputs` reads the entire committed snapshot, not a separately opened minimal allowlisted index (population 544, 561). The semantic read at 327 is `set(snapshot["content"]["tf_v1_records"]["primary"])`: dictionary iteration returns IDs, discarding its values. The committed primary index maps each ID to two 64-character hash strings; the schema inspection confirmed that shape for all primary entries without opening the named records. Existing membership then follows `parent_id is None`, outside Part A components, and `id in primary_ids` (150–168). It does not read a score, rank on a score, or choose a score threshold. The 123 is a frozen expected count (56–61), compared with the selected count (384–389), not evidence of a fresh scoring run in this audit.

This membership information does affect decisions: a primary ID makes a root existing; any primary entry in a component blocks that component's purchasable roots (153–164). Component/family metadata then determines strata, folds and purchase subsets (171–207, 286–313). This is dependence on *which outcomes exist*, not on their metric values. Index values are still present in parsed memory and the whole snapshot SHA256 reaches `inputs` (552–556, 460); changing any snapshot byte can change that provenance hash. A hash dependency must not be described as zero byte access or complete information independence.

The other selected snapshot fields are `content.zoo_bytes.entry_delta_hash` (328, 213–233), and `exposure_ledger.rows[*].{gate,id,aliases}` for G0/G1/G6 (242–253). Delta hashes resolve aliases and duplicate groups; gate/ID/alias values reach row exposure annotations and reproduction counts (338, 370–373, 417–421, 476–481). Those are categorical exposure metadata, not metric values. Other ledger fields are parsed but not selected. No ledger file path or `snapshot.content.tf_v1_records.path` is opened; embedded record/hash/path strings are not dereferenced. The committed ledger row schema inspected here has activity/identity/provenance fields, not score fields. Thus “snapshot contains an outcome index/ledger” does not imply “selector reads outcome scores,” while “reads membership only” is only a semantic-use description, not a byte-level parsing description.

## Imports and helper reachability

The direct repository dependencies are population (stdlib only); tokens (power-fields, Part A predictions, response-chart, response-field, runtime population and zoo apply); additivity (power-fields, Part A predictions, response-field, rotation, runtime zoo apply). Package initialization also imports feature stats/SVD and zoo random-direction utilities. Power-fields imports only the `ROTATION_DOMAIN` and pure `rotation_seed` function from `wsp.eval.response_intervention` (power-fields 68; intervention 252–255). **Importing that module does not call its preparation, proof, or scoring routines.** Its batch/extraction imports are inside other functions, not module-top-level (1121, 1498–1500, 1669–1671, 2221, 2405–2406, 2739).

Reachable I/O helpers are power-fields `sha256_file`, `load_field`, `load_base_state`; response-field `FieldArtifact.load`; zoo apply `load_entry_delta` → LoRA config/tensors or dense tensors; common `map_delta_keys` is in-memory. Part A `FactorCache`, `contract`, `contraction_row`, `field_row` perform tensor arithmetic; neither Part A `run` nor `load_staged` is called. Response-chart slot layout/architecture routines do not open datasets. Rotation `apply_and_round` is tensor arithmetic. Package initialization and the schemas define functions/types without opening outcome files.

Dormant helpers must not be mistaken for reachable reads. Part A `load_staged` can read `<stage>/<variant_id>/{variant.json,delta.safetensors,increment.safetensors}` (214–245); its own `run` opens Part A manifests, selections, components and G2/chain-exact fields (663–727), and optional prediction comparison files (855–860). Power-fields' own `run` reads payload/selection/components/G2/chain-exact fields (496–558). None is called by the F12 entry points/tests. Intervention proof helpers can open `<evals_dir>/<ref_id>/eval.json` and `records.jsonl` (1041–1101): selected fields there are status/config/revision/counts/entry hashes, benchmark/item ID/token hash, repeat/run metadata, not score values; they are not called here. Its scoring path can invoke the evaluation machinery. Those functions' existence is a latent capability, not evidence that F12 called them.

For completeness the appendix inventories the lexical import closure, including function-local imports and test fixture imports, and their filesystem/library loading sites. Sites marked by a containing function are capabilities, not proof of invocation. The dormant evaluation modules can read/write metrics when explicitly run; they are outside the reachable F12 call path. The local IFEval utility has an NLTK resource lookup/download on its own import, but that module's import is downstream of a dormant evaluation function, not the top-level F12 imports. Third-party implementations are not audited as repository modules.

## Tests

`test_f12_population.py` imports population (12); its real-manifest reproduction test (169–185) calls the same four-manifest loader and build, including whole-snapshot parsing. Its other tests use in-memory fixtures, including ledger IDs/gates/aliases. Token tests import power-fields/tokens/Part A/response-field (12–15), use synthetic tensors, and create/read/hash temporary `probe_cache.json`, `000.safetensors`, `001.safetensors` (286–318), `d/delta.safetensors`, `l/lora/adapter_model.safetensors` (328–340). Additivity tests use synthetic tensors and write/read/hash `<tmp>/<part_id>/{delta.safetensors,variant.json}` (163–192). Guard tests resolve intended output paths but do not read their contents (tokens test 321–325; additivity test 154–160). None invokes F12 `run`, real models or outcome scoring.

`tests/conftest.py` has autouse thread-restoration and pinned-base-revision fixtures; the latter imports `wsp.zoo.build` and replaces its revision resolver with a deterministic fake (19–42). Importing build imports edits kernels/refusal, schema, zoo arch/apply/common/random_dir, but does not call zoo construction, Hub loading or refusal-data extraction. Its optional family dynamic imports occur only in its dormant generator. This audit ran no pytest, so no fixture or collection side effect occurred here. The tests do not prove input path containment, safe symlink handling, resume provenance or the killed session's access history.

## Could it read data/evals? Is the token claim supported?

**Yes, as a capability; no direct intended F12 outcome path was found.** All three trust input paths. Population's `--manifests` can point anywhere; four fixed filenames can be symlinks. Tokens' `--population`, `--g2`, `--zoo`, population IDs/holders and probe `row.file` are not containment-checked. Additivity likewise accepts population/G2/zoo/Astra/C4 paths and an Astra-controlled directory. Absolute path components or `..`, and ordinary OS symlink following, can redirect reads under `data/evals`. Even hashing reads bytes. G2 hash pins often reject substitutions, but only **after reading** the bytes, so they are integrity checks, not access controls. Resume files beneath an allowed output root can themselves be symlinks; path resolution guards constrain the directory, not every leaf opened later. No filesystem race protection or no-follow open is present. I did not test any such redirection or inspect actual data symlink targets.

The token manifest's `outcomes_read: none...` is **SUPPORTED for its intended metric-free extraction algorithm**, but **not supported as an unconditional runtime/session attestation**. It omits own-output resume reads and source hashing from its prose input list, and cannot detect redirected or contaminated inputs. The same constant is written on runs regardless of optional resume behavior (tokens 964–965). The current committed manifest reports the target commit and clean code; I compared its listed source hashes against git blobs of `0412799`. This authenticates consistency of those code pins only, not the input filesystem or actual historical execution. No real output hashes were verified against `/home/ubuntu/wsp-data`.

## Limits and this session's exposure attestation

Static source does not show what the dead session itself opened interactively while developing, outside the committed code. It cannot establish shell/editor/notebook history, unseen prompts, earlier uncommitted code, actual launch arguments, data symlink targets, runtime library behavior, environmental tampering, or which files the token process really opened. It does not prove outcome blindness of role assignment or cure a missing personal attestation. Board A must separately adjudicate row 5 / entry 22 and any remedy before C1 verification.

This session only:

1. **Old development outcomes:** I read the repository-required README/method/governance material and attestation/review text. These expose published old-development numerical summaries (including earlier predictor/base-model results) and exposure accounts; I cannot attest to having seen no old numerical results. I did not open underlying TF-v1 root records, D3 records or an external G0/G1/G6 outcome ledger. I parsed the committed A5 snapshot and printed keys/types/index shape, including its embedded exposure ledger metadata; no underlying metric payload was opened. I saw the embedded F8a floor constant in source.
2. **New outcomes and data/evals:** I opened and listed nothing under `data/evals`, opened nothing under `/home/ubuntu/wsp-data`, and received no new Part A, F8-E/F8-F, F9, F12 or F12-I evaluation outcome. I ran no target code, pytest, scoring, simulation or model loading, and opened no real token array, prediction, construction or noise payload.
3. **Prescore audits:** I read source/tests containing forecast machinery and synthetic numerical examples, the F8a floor literal, and governance/register/review descriptions of prescore audits. I did not open underlying Part A prediction/calibration or prescore audit payloads. The source's floor reference was not followed to its source JSON.

These statements cover only this audit session and do not attest for any implementer, chair, reviewer or prior session.

## Appendix: repository import closure and filesystem capability index

Inventory generated by static AST inspection of the pinned source, with paths left as source expressions. It includes imports inside functions, even when unreachable from F12. Classifications for the reachable sites are in the path table above. In the dormant index below: zoo deltas/configs are **stored edit bytes**; model/cache checkpoint configuration is **base model**; field/probe artifacts are **G2 field/probe**; selection/components/panels/subset IDs are **population/fold or panel manifests**; `eval.json`, `records.jsonl`, eval ledgers and repeat files are **evaluation/outcome**; staged variants, features and prediction tables are **staging/prediction**; source/config/system/resource files are **other**. A generic path parameter has no statically guaranteed class; classify by the supplied file. None of the dormant evaluation I/O sites is invoked by the F12 scripts or their three tests.

### `scripts/a5_power_fields.py`

Repository imports (lexical): `src/wsp/eval/response_intervention.py`, `src/wsp/features/__init__.py`, `src/wsp/features/response_field.py`, `src/wsp/features/rotation.py`, `src/wsp/zoo/apply.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 92 | `sha256_file` | `open(path, 'rb')` |
| 426 | `load_field` | `rf.FieldArtifact.load(directory)` |
| 434 | `load_base_state` | `AutoModelForCausalLM.from_pretrained(MODEL[0], revision=MODEL[1], dtype=torch.bfloat16, device_map='cpu')` |
| 468 | `write_json` | `path.parent.mkdir(parents=True, exist_ok=True)` |
| 470 | `write_json` | `path.write_bytes(data)` |
| 496 | `run` | `args.inputs.read_bytes()` |
| 498 | `run` | `args.selection.read_bytes()` |
| 501 | `run` | `comp_path.read_bytes()` |
| 513 | `run` | `g2_manifest_path.read_text()` |
| 541 | `run` | `progress.parent.mkdir(parents=True, exist_ok=True)` |
| 542 | `run` | `progress.write_text('')` |
| 546 | `run.record` | `progress.open('a')` |

### `scripts/f12_additivity_prepare.py`

Repository imports (lexical): `scripts/a5_power_fields.py`, `scripts/part_a_predictions.py`, `src/wsp/features/__init__.py`, `src/wsp/features/response_field.py`, `src/wsp/features/rotation.py`, `src/wsp/zoo/apply.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 253 | `FieldView.__init__` | `(g2 / 'g2_manifest.json').read_bytes()` |
| 267 | `FieldView.__init__` | `(d / 'manifest.json').read_text()` |
| 272 | `FieldView.__init__` | `torch.load(d / 'tensors.pt', map_location='cpu', weights_only=True, mmap=True)` |
| 340 | `write_json` | `path.parent.mkdir(parents=True, exist_ok=True)` |
| 342 | `write_json` | `path.write_bytes(data)` |
| 355 | `holder_files` | `p.exists()` |
| 405 | `plan_constructions` | `Path(astra).read_text()` |
| 428 | `plan_constructions` | `load_file(str(path))` |
| 506 | `write_part` | `d.mkdir(parents=True, exist_ok=True)` |
| 507 | `write_part` | `save_file({k: inc[k].contiguous() for k in sorted(inc)}, str(d / 'delta.safetensors'))` |
| 547 | `run` | `args.population.read_bytes()` |
| 553 | `run` | `(Path(args.g2) / 'fields' / OBJECTIVE / 'manifest.json').read_text()` |

### `scripts/f12_population.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 544 | `load_inputs` | `p.read_bytes()` |
| 569 | `write_json` | `path.parent.mkdir(parents=True, exist_ok=True)` |
| 571 | `write_json` | `path.write_bytes(data)` |

### `scripts/f12_tokens.py`

Repository imports (lexical): `scripts/a5_power_fields.py`, `scripts/part_a_predictions.py`, `src/wsp/features/__init__.py`, `src/wsp/features/response_chart.py`, `src/wsp/features/response_field.py`, `scripts/f12_population.py`, `src/wsp/zoo/apply.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 466 | `stored_tensor_count` | `(entry_dir / 'delta.safetensors').exists()` |
| 467 | `stored_tensor_count` | `safe_open(str(entry_dir / 'delta.safetensors'), 'pt')` |
| 469 | `stored_tensor_count` | `safe_open(str(entry_dir / 'lora/adapter_model.safetensors'), 'pt')` |
| 481 | `load_probe` | `(probe_dir / 'probe_cache.json').read_text()` |
| 489 | `load_probe` | `load_file(str(p))` |
| 533 | `write_json` | `path.parent.mkdir(parents=True, exist_ok=True)` |
| 535 | `write_json` | `path.write_bytes(data)` |
| 541 | `load_field_setup` | `(g2 / 'g2_manifest.json').read_bytes()` |
| 629 | `save_frames` | `save_file(tensors, str(out_dir / 'kframes.safetensors'))` |
| 651 | `run` | `out.mkdir(parents=True, exist_ok=True)` |
| 652 | `run` | `args.population.read_bytes()` |
| 690 | `run` | `detail_path.exists()` |
| 691 | `run` | `detail_path.read_text()` |
| 695 | `run` | `rows_path.read_text()` |
| 700 | `run` | `rows_path.write_text('')` |
| 701 | `run` | `detail_path.write_text('')` |
| 709 | `run` | `(zoo / r['id'] / 'edit.json').read_text()` |
| 711 | `run` | `cfg_path.exists()` |
| 711 | `run` | `cfg_path.read_text()` |
| 756 | `run` | `rows_path.open('a')` |
| 758 | `run` | `detail_path.open('a')` |
| 863 | `write_manifest` | `save_file({'X': torch.from_numpy(X), 'presence': torch.from_numpy(P), 'row_eligible': torch.from_numpy(E)}, str(out / 'tokens.safetensors'), metadata={'row_ids': json.dumps(ids), 'columns': json.dumps(COLUMNS)})` |
| 890 | `write_manifest` | `(out / name).exists()` |
| 943 | `write_manifest` | `args.population.read_text()` |

### `scripts/part_a_predictions.py`

Repository imports (lexical): `scripts/a5_power_fields.py`, `src/wsp/features/__init__.py`, `src/wsp/features/response_field.py`, `src/wsp/features/rotation.py`, `src/wsp/zoo/apply.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 222 | `load_staged` | `(d / 'variant.json').read_bytes()` |
| 230 | `load_staged` | `(d / name).read_bytes()` |
| 241 | `variant_json_only` | `(d / 'variant.json').read_bytes()` |
| 648 | `write_json` | `path.parent.mkdir(parents=True, exist_ok=True)` |
| 650 | `write_json` | `path.write_bytes(data)` |
| 667 | `run` | `args.manifest.read_bytes()` |
| 669 | `run` | `args.selection.read_bytes()` |
| 672 | `run` | `comp_path.read_bytes()` |
| 682 | `run` | `g2_manifest_path.read_bytes()` |
| 731 | `run` | `out_rows.parent.mkdir(parents=True, exist_ok=True)` |
| 732 | `run` | `out_rows.write_text('')` |
| 733 | `run` | `out_detail.write_text('')` |
| 828 | `run` | `out_rows.open('a')` |
| 830 | `run` | `out_detail.open('a')` |
| 856 | `run` | `Path(args.compare_fields).read_text()` |
| 860 | `run` | `prow_path.read_text()` |

### `src/wsp/__init__.py`

Repository imports (lexical): none.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/editor/api.py`

Repository imports (lexical): `src/wsp/eval/run.py`, `src/wsp/features/extract.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 68 | `_probe_worker` | `model.load()` |

### `src/wsp/editor/denials.py`

Repository imports (lexical): none.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/editor/driver.py`

Repository imports (lexical): `src/wsp/editor/api.py`, `src/wsp/editor/episode.py`, `src/wsp/editor/sandbox.py`, `src/wsp/editor/zoo.py`, `src/wsp/editor/providers.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 59 | `qualify_pools` | `path.write_text(json.dumps(manifest, sort_keys=True))` |
| 106 | `load_config` | `open(path)` |
| 128 | `run_episode` | `manifest_path.exists()` |
| 128 | `run_episode` | `manifest_path.read_text()` |
| 138 | `run_episode` | `manifest_path.write_text(json.dumps(manifest, sort_keys=True))` |

### `src/wsp/editor/episode.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 38 | `Episode.__init__` | `self.path.parent.mkdir(parents=True, exist_ok=True, mode=448)` |
| 41 | `Episode.append` | `os.open(self.path, os.O_WRONLY &#124; os.O_CREAT &#124; os.O_APPEND, 384)` |

### `src/wsp/editor/providers.py`

Repository imports (lexical): `src/wsp/editor/driver.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/editor/sandbox.py`

Repository imports (lexical): `src/wsp/editor/episode.py`, `src/wsp/editor/denials.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/editor/zoo.py`

Repository imports (lexical): `src/wsp/editor/episode.py`, `src/wsp/features/extract.py`, `src/wsp/schema.py`, `src/wsp/zoo/common.py`, `src/wsp/zoo/build.py`, `src/wsp/editor/driver.py`, `src/wsp/editor/providers.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 50 | `register` | `entry.exists()` |
| 94 | `build` | `(entry / 'edit.json').read_text()` |
| 95 | `build` | `load_file(entry / 'delta.safetensors')` |

### `src/wsp/edits/__init__.py`

Repository imports (lexical): none.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/edits/kernels.py`

Repository imports (lexical): `src/wsp/edits/refusal.py`, `src/wsp/zoo/random_dir.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/edits/refusal.py`

Repository imports (lexical): `src/wsp/zoo/arch.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 209 | `resolve_commit` | `ref.exists()` |
| 210 | `resolve_commit` | `ref.read_text()` |
| 239 | `_sha256_file` | `path.read_bytes()` |
| 265 | `_compute` | `AutoModelForCausalLM.from_pretrained(config['base_model'], revision=key['model_commit'], torch_dtype=getattr(_torch, key['dtype']), device_map='cuda' if _torch.cuda.is_available() else 'cpu', **remote_code_kwargs(config['base_model'], config))` |
| 292 | `get_refusal_directions` | `AutoTokenizer.from_pretrained(config['base_model'], revision=commits['model'], **remote_code_kwargs(config['base_model'], config))` |
| 326 | `get_refusal_directions` | `manifest_path.exists()` |
| 326 | `get_refusal_directions` | `path.exists()` |
| 327 | `get_refusal_directions` | `manifest_path.read_text()` |
| 333 | `get_refusal_directions` | `torch.load(path, map_location='cpu', weights_only=True)` |
| 342 | `get_refusal_directions` | `load_dataset(dataset, split=SPLIT, revision=commit)` |
| 352 | `get_refusal_directions` | `zoo_dir.mkdir(parents=True, exist_ok=True)` |
| 353 | `get_refusal_directions` | `torch.save(dirs, path)` |
| 370 | `get_refusal_directions` | `manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))` |

### `src/wsp/eval/__init__.py`

Repository imports (lexical): none.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/eval/batch.py`

Repository imports (lexical): `src/wsp/__init__.py`, `src/wsp/eval/metadata.py`, `src/wsp/eval/run.py`, `src/wsp/eval/zoo_compat.py`, `src/wsp/eval/repeats.py`, `src/wsp/zoo/common.py`, `src/wsp/schema.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 31 | `provenance` | `(REPO / 'src/wsp').rglob('*.py')` |
| 33 | `provenance` | `path.read_bytes()` |
| 83 | `upsert_index` | `path.parent.mkdir(parents=True, exist_ok=True)` |
| 84 | `upsert_index` | `path.with_suffix('.lock').open('w')` |
| 86 | `upsert_index` | `path.exists()` |
| 86 | `upsert_index` | `path.read_text()` |
| 97 | `upsert_index` | `tmp.write_text(''.join((json.dumps(r) + '\n' for r in keyed.values())))` |
| 98 | `upsert_index` | `tmp.replace(path)` |
| 183 | `select_entries` | `result.exists()` |
| 184 | `select_entries` | `result.read_text()` |
| 222 | `atomic_json` | `tmp.write_text(json.dumps(value, indent=2))` |
| 223 | `atomic_json` | `tmp.replace(path)` |
| 231 | `base_destination` | `root.glob('devices/*/*/eval.json')` |
| 233 | `base_destination` | `file.exists()` |
| 235 | `base_destination` | `file.read_text()` |
| 246 | `base_destination` | `(root / 'eval.json').exists()` |
| 266 | `run_batch` | `config.read_bytes()` |
| 279 | `run_batch` | `(zoo / 'manifest.jsonl').read_text()` |
| 302 | `run_batch` | `(REPO / 'configs/eval_subset_d3.yaml').read_text()` |
| 357 | `run_batch` | `out.mkdir(parents=True, exist_ok=True)` |
| 365 | `run_batch` | `dest.mkdir(parents=True, exist_ok=False)` |
| 376 | `run_batch` | `(source / 'edit.json').read_text()` |
| 443 | `run_batch` | `dest.mkdir(parents=True, exist_ok=True)` |
| 467 | `run_batch` | `p.read_bytes()` |
| 470 | `run_batch` | `(source / 'delta.safetensors').exists()` |
| 471 | `run_batch` | `(source / 'lora').glob('*')` |
| 475 | `run_batch` | `p.is_file()` |

### `src/wsp/eval/ifeval/__init__.py`

Repository imports (lexical): none.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/eval/ifeval/instructions.py`

Repository imports (lexical): `src/wsp/eval/ifeval/__init__.py`, `src/wsp/eval/ifeval/instructions_util.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/eval/ifeval/instructions_registry.py`

Repository imports (lexical): `src/wsp/eval/ifeval/__init__.py`, `src/wsp/eval/ifeval/instructions.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/eval/ifeval/instructions_util.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 42 | `download_nltk_resources` | `nltk.data.find('tokenizers/punkt_tab')` |
| 45 | `download_nltk_resources` | `nltk.download('punkt_tab')` |
| 1686 | `_get_sentence_tokenizer` | `nltk.data.load('nltk:tokenizers/punkt/english.pickle')` |

### `src/wsp/eval/ifeval/utils.py`

Repository imports (lexical): `src/wsp/eval/ifeval/__init__.py`, `src/wsp/eval/ifeval/instructions_registry.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/eval/metadata.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 14 | `runtime_metadata` | `path.exists()` |
| 17 | `runtime_metadata` | `path.read_text()` |

### `src/wsp/eval/refusal.py`

Repository imports (lexical): `src/wsp/schema.py`, `src/wsp/eval/run.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 48 | `prompts` | `load_dataset(spec.dataset, getattr(spec, 'config', None), split=spec.split)` |
| 128 | `export_judge` | `out.parent.mkdir(parents=True, exist_ok=True)` |
| 140 | `export_judge` | `out.write_text(''.join((json.dumps(r) + '\n' for r in rows)))` |
| 150 | `main` | `Path(args.records).read_text()` |

### `src/wsp/eval/repeats.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 120 | `item_records` | `path.is_file()` |
| 123 | `item_records` | `file.read_text()` |
| 124 | `item_records` | `file.exists()` |
| 262 | `compare_pairs.read` | `Path(x).read_text()` |

### `src/wsp/eval/response_intervention.py`

Repository imports (lexical): `src/wsp/features/__init__.py`, `src/wsp/features/rotation.py`, `src/wsp/zoo/apply.py`, `src/wsp/eval/batch.py`, `src/wsp/features/extract.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 266 | `_sha256` | `open(path, 'rb')` |
| 383 | `_load_written` | `Path(path).exists()` |
| 383 | `_load_written` | `load_file(str(path))` |
| 581 | `_stage_header` | `manifest.exists()` |
| 582 | `_stage_header` | `manifest.read_text()` |
| 595 | `_upsert_stage_rows` | `path.exists()` |
| 595 | `_upsert_stage_rows` | `path.read_text()` |
| 605 | `_upsert_stage_rows` | `tmp.write_text(''.join((json.dumps(r, sort_keys=True) + '\n' for r in lines)))` |
| 606 | `_upsert_stage_rows` | `tmp.replace(path)` |
| 659 | `build_variants` | `(entry_dir / 'edit.json').read_text()` |
| 681 | `build_variants` | `(entry_dir / 'delta.safetensors').exists()` |
| 681 | `build_variants` | `(entry_dir / 'lora').exists()` |
| 701 | `build_variants` | `stage_root.mkdir(parents=True, exist_ok=True)` |
| 703 | `build_variants` | `stage_meta.exists()` |
| 704 | `build_variants` | `stage_meta.read_text()` |
| 707 | `build_variants` | `stage_meta.write_text(json.dumps({'source_zoo': str(canonical)}, indent=1, sort_keys=True))` |
| 739 | `build_variants.write` | `d.mkdir(exist_ok=True)` |
| 741 | `build_variants.write` | `(d / name).unlink(missing_ok=True)` |
| 750 | `build_variants.write` | `save_file(dict(sorted(increment.items())), str(d / 'increment.safetensors'))` |
| 758 | `build_variants.write` | `save_file({k: delta[k].contiguous() for k in sorted(delta)}, str(d / 'delta.safetensors'))` |
| 761 | `build_variants.write` | `shutil.copyfile(entry_dir / 'edit.json', d / 'edit.json')` |
| 767 | `build_variants.write` | `(d / name).unlink(missing_ok=True)` |
| 797 | `build_variants.write` | `(d / 'variant.json').write_text(json.dumps(info, indent=1, sort_keys=True, allow_nan=not strict))` |
| 949 | `_entry_files` | `(d / n).exists()` |
| 977 | `_f10_subsets` | `(v.entry_dir / 'variant.json').read_text()` |
| 1027 | `_load_config` | `Path(config).read_bytes()` |
| 1032 | `_source_hashes` | `(source / 'delta.safetensors').exists()` |
| 1036 | `_source_hashes` | `(source / 'lora').glob('*')` |
| 1037 | `_source_hashes` | `p.is_file()` |
| 1044 | `_reference_on_file` | `path.exists()` |
| 1046 | `_reference_on_file` | `path.read_text()` |
| 1055 | `_reference_on_file` | `source.exists()` |
| 1062 | `record_windows` | `Path(records_path).read_text()` |
| 1073 | `registered_windows` | `Path(text_groups).read_text()` |
| 1085 | `reuse_proof` | `(d / 'eval.json').exists()` |
| 1087 | `reuse_proof` | `(d / 'eval.json').read_text()` |
| 1088 | `reuse_proof` | `source.exists()` |
| 1099 | `reuse_proof` | `(d / 'records.jsonl').exists()` |
| 1125 | `_stage` | `(staged_zoo / 'stage.json').read_text()` |
| 1127 | `_stage` | `(staged_zoo / 'manifest.jsonl').read_text()` |
| 1172 | `scoring_summary` | `p.read_text()` |
| 1173 | `scoring_summary` | `Path(staged_zoo).glob('*/variant.json')` |
| 1174 | `scoring_summary` | `p.read_text()` |
| 1217 | `score_commands` | `(staged_zoo / 'manifest.jsonl').read_text()` |
| 1225 | `score_commands` | `(staged_zoo / 'stage.json').read_text()` |
| 1229 | `score_commands` | `(source_zoo / i / 'delta.safetensors').exists()` |
| 1230 | `score_commands` | `(source_zoo / i / 'lora').is_dir()` |
| 1246 | `load_feature_norms` | `path.is_file()` |
| 1247 | `load_feature_norms` | `path.read_text()` |
| 1260 | `load_feature_norms` | `f.exists()` |
| 1262 | `load_feature_norms` | `f.read_text()` |
| 1278 | `load_registered_selection` | `path.read_text()` |
| 1334 | `load_bytes_holders` | `Path(references).read_text()` |
| 1435 | `audit_rows` | `(v.entry_dir / 'variant.json').read_text()` |
| 1462 | `edit_audit_summary` | `(v.entry_dir / 'variant.json').read_text()` |
| 1482 | `_du` | `Path(path).rglob('*')` |
| 1482 | `_du` | `p.is_file()` |
| 1503 | `prepare` | `(zoo / 'manifest.jsonl').read_text()` |
| 1513 | `prepare` | `Path(args.sidecar).read_text()` |
| 1526 | `prepare` | `(zoo / s.edit_id / 'edit.json').read_text()` |
| 1561 | `prepare` | `out.parent.mkdir(parents=True, exist_ok=True)` |
| 1563 | `prepare` | `audit_path.write_text(''.join((json.dumps(r, sort_keys=True) + '\n' for r in audits)))` |
| 1591 | `prepare` | `out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')` |
| 1631 | `_ladder_hashes` | `(d / 'variant.json').exists()` |
| 1685 | `prepare_rotation_only` | `staged_path.read_bytes()` |
| 1692 | `prepare_rotation_only` | `p.exists()` |
| 1692 | `prepare_rotation_only` | `p.read_bytes()` |
| 1712 | `prepare_rotation_only` | `out.parent.mkdir(parents=True, exist_ok=True)` |
| 1717 | `prepare_rotation_only` | `p.exists()` |
| 1717 | `prepare_rotation_only` | `p.read_bytes()` |
| 1719 | `prepare_rotation_only` | `p.write_bytes(data)` |
| 1722 | `prepare_rotation_only` | `(zoo / 'manifest.jsonl').read_text()` |
| 1726 | `prepare_rotation_only` | `(zoo / s.edit_id / 'edit.json').read_text()` |
| 1775 | `prepare_rotation_only` | `audit_path.write_text(''.join((json.dumps(r, sort_keys=True) + '\n' for r in audits)))` |
| 1824 | `prepare_rotation_only` | `out.write_text(json.dumps(rotation.json_safe(manifest), indent=1, sort_keys=True, allow_nan=False) + '\n')` |
| 2017 | `load_supplement_selection` | `Path(path).read_text()` |
| 2020 | `load_supplement_selection` | `Path(registered_path).read_bytes()` |
| 2071 | `_check_supplement` | `p.is_file()` |
| 2071 | `_check_supplement` | `p.read_bytes()` |
| 2166 | `_stage_rows` | `path.exists()` |
| 2166 | `_stage_rows` | `path.read_text()` |
| 2181 | `rehash_staged` | `(d / 'variant.json').exists()` |
| 2420 | `prepare_supplement` | `staged_path.read_bytes()` |
| 2436 | `prepare_supplement` | `p.exists()` |
| 2436 | `prepare_supplement` | `p.read_bytes()` |
| 2449 | `prepare_supplement` | `(zoo / 'manifest.jsonl').read_text()` |
| 2460 | `prepare_supplement` | `(zoo / i / 'delta.safetensors').exists()` |
| 2461 | `prepare_supplement` | `(zoo / i / 'lora').is_dir()` |
| 2486 | `prepare_supplement` | `out.parent.mkdir(parents=True, exist_ok=True)` |
| 2491 | `prepare_supplement` | `p.exists()` |
| 2491 | `prepare_supplement` | `p.read_bytes()` |
| 2493 | `prepare_supplement` | `p.write_bytes(data)` |
| 2497 | `prepare_supplement` | `(zoo / i / 'edit.json').read_text()` |
| 2612 | `prepare_supplement` | `audit_path.write_bytes(audit_bytes)` |
| 2613 | `prepare_supplement` | `out.write_text(json.dumps(rotation.json_safe(merged), indent=1, sort_keys=True, allow_nan=False) + '\n')` |
| 2653 | `write_slot_mask` | `Path(args.manifest).read_text()` |
| 2655 | `write_slot_mask` | `Path(args.out).write_text(text)` |
| 2712 | `score_reference_proofs` | `manifest_path.read_bytes()` |
| 2731 | `score_reference_proofs` | `(Path(stage) / 'stage.json').read_text()` |
| 2733 | `score_reference_proofs` | `(source_zoo / i / 'edit.json').read_text()` |
| 2771 | `score_reference_proofs` | `out.exists()` |
| 2771 | `score_reference_proofs` | `out.read_text()` |
| 2772 | `score_reference_proofs` | `out.read_bytes()` |
| 2778 | `score_reference_proofs` | `out.write_text(text)` |
| 2811 | `load_reference_proofs` | `manifest_path.read_bytes()` |
| 2816 | `load_reference_proofs` | `path.exists()` |
| 2821 | `load_reference_proofs` | `path.read_text()` |
| 2852 | `score` | `Path(args.manifest).read_text()` |

### `src/wsp/eval/run.py`

Repository imports (lexical): `src/wsp/eval/subsets.py`, `src/wsp/schema.py`, `src/wsp/zoo/arch.py`, `src/wsp/eval/ifeval/utils.py`, `src/wsp/eval/zoo_compat.py`, `src/wsp/eval/metadata.py`, `src/wsp/zoo/common.py`, `src/wsp/eval/tf_v1.py`, `src/wsp/eval/refusal.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 161 | `HFRunner.__init__` | `(Path(zoo_entry) / 'edit.json').read_text()` |
| 167 | `HFRunner.__init__` | `(Path(model_id) / 'config.json').read_bytes()` |
| 168 | `HFRunner.__init__` | `Path(model_id).is_dir()` |
| 173 | `HFRunner.__init__` | `AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side='left', **remote)` |
| 178 | `HFRunner.__init__` | `AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map=device or 'auto', attn_implementation='sdpa', **remote)` |
| 194 | `HFRunner.__init__` | `(entry / 'edit.json').read_text()` |
| 205 | `HFRunner.__init__` | `(entry / 'delta.safetensors').exists()` |
| 216 | `HFRunner.__init__` | `p.read_bytes()` |
| 285 | `write_results` | `out.mkdir(parents=True, exist_ok=True)` |
| 286 | `write_results` | `(out / 'records.jsonl').write_text(''.join((json.dumps(asdict(r)) + '\n' for r in records)))` |
| 330 | `write_results` | `(out / 'summary.json').write_text(json.dumps(summary, indent=2))` |
| 418 | `perplexity` | `load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test', **{'revision': cfg['revision']} if 'revision' in cfg else {})` |
| 471 | `main` | `Path(args.config).read_bytes()` |
| 491 | `main` | `Path(__file__).read_bytes()` |
| 502 | `main` | `out.mkdir(parents=True, exist_ok=True)` |

### `src/wsp/eval/subsets.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 36 | `load_items` | `load_dataset(repo, config, split=split, revision=REVISIONS[name])` |
| 39 | `load_items` | `(ROOT / f'{name}.txt').read_text()` |
| 54 | `main` | `ROOT.mkdir(parents=True, exist_ok=True)` |
| 56 | `main` | `(ROOT / f'{name}.txt').write_text('\n'.join(subset_ids(size, n)) + '\n')` |

### `src/wsp/eval/tf_v1.py`

Repository imports (lexical): `src/wsp/eval/metadata.py`, `src/wsp/eval/subsets.py`, `src/wsp/schema.py`, `src/wsp/eval/refusal.py`, `src/wsp/eval/run.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 63 | `atomic_write_json` | `path.parent.mkdir(parents=True, exist_ok=True)` |
| 65 | `atomic_write_json` | `tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))` |
| 66 | `atomic_write_json` | `tmp.replace(path)` |
| 80 | `eligible_ids` | `(ROOT / f'{name}.txt').read_text()` |
| 89 | `read_panel` | `panel_path(name).read_text()` |
| 123 | `write_panels` | `panel_path(name).write_text(''.join((f'# {h}\n' for h in header)) + '\n'.join(rank_ids(pool, salt)[:size]) + '\n')` |
| 136 | `load_items` | `load_dataset(repo, config, split=split, revision=REVISIONS[name])` |
| 157 | `load_prompts` | `load_dataset(repo, split='test', revision=revision)` |
| 435 | `load_verified` | `root.glob('*.pt')` |
| 440 | `load_verified` | `torch.load(root / name, weights_only=True)` |
| 458 | `build_cache` | `torch.save(entry, tmp)` |
| 461 | `build_cache` | `tmp.unlink()` |
| 463 | `build_cache` | `tmp.replace(file)` |
| 502 | `cache_kl` | `root.mkdir(parents=True, exist_ok=True)` |
| 505 | `cache_kl` | `manifest.exists()` |
| 506 | `cache_kl` | `manifest.read_text()` |
| 671 | `main` | `Path(args.config).read_text()` |

### `src/wsp/eval/zoo_compat.py`

Repository imports (lexical): `src/wsp/zoo/common.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 54 | `lora_dense_delta` | `(lora_dir / 'adapter_config.json').read_text()` |
| 64 | `lora_dense_delta` | `load_file(lora_dir / 'adapter_model.safetensors')` |
| 109 | `load_entry_delta` | `descriptor.exists()` |
| 109 | `load_entry_delta` | `descriptor.read_text()` |
| 112 | `load_entry_delta` | `(d / 'delta.safetensors').exists()` |
| 114 | `load_entry_delta` | `load_file(d / 'delta.safetensors')` |
| 117 | `load_entry_delta` | `(d / 'delta.safetensors').exists()` |

### `src/wsp/features/__init__.py`

Repository imports (lexical): `src/wsp/features/stats.py`, `src/wsp/features/svd.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/features/extract.py`

Repository imports (lexical): `src/wsp/features/pool.py`, `src/wsp/features/stats.py`, `src/wsp/features/svd.py`, `src/wsp/zoo/apply.py`, `src/wsp/zoo/arch.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 93 | `read_edit` | `path.read_text()` |
| 139 | `_atomic_save` | `save_file({k: v.float().contiguous() for k, v in tensors.items()}, tmp)` |
| 140 | `_atomic_save` | `os.replace(tmp, path)` |
| 158 | `extract_entry` | `(entry_dir / 'delta.safetensors').exists()` |
| 271 | `extract_entry` | `out_dir.mkdir(parents=True, exist_ok=True)` |
| 275 | `extract_entry` | `tmp.write_text(json.dumps(meta, indent=1, sort_keys=True))` |
| 276 | `extract_entry` | `os.replace(tmp, out_dir / 'features.json')` |
| 285 | `resolve_base_commit` | `snapshot_download(base_model, revision=revision, allow_patterns=['config.json'], local_files_only=True)` |
| 297 | `load_base_state` | `transformers.AutoModelForCausalLM.from_pretrained(base_model, revision=revision, dtype=torch.bfloat16, device_map='cpu', local_files_only=True, **remote_code_kwargs(base_model))` |
| 322 | `file_sha256` | `open(path, 'rb')` |
| 358 | `_manifest_ids` | `path.exists()` |
| 362 | `_manifest_ids` | `path.read_text()` |
| 391 | `_is_done` | `(out_entry / 'pooled.safetensors').exists()` |
| 391 | `_is_done` | `f.exists()` |
| 394 | `_is_done` | `f.read_text()` |
| 427 | `_write_json` | `tmp.write_text(json.dumps(data, indent=1, sort_keys=True))` |
| 428 | `_write_json` | `os.replace(tmp, path)` |
| 460 | `run_extract` | `manifest_path.exists()` |
| 462 | `run_extract` | `manifest_path.read_text()` |
| 540 | `run_extract` | `out.mkdir(parents=True, exist_ok=True)` |
| 543 | `run_extract.write_manifest` | `(out / i / 'features.json').exists()` |
| 628 | `main` | `load_file(args.base_file)` |

### `src/wsp/features/pool.py`

Repository imports (lexical): none.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/features/response_chart.py`

Repository imports (lexical): `src/wsp/zoo/apply.py`, `src/wsp/zoo/arch.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/features/response_field.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 285 | `FieldArtifact.save` | `root.mkdir(parents=True, exist_ok=True)` |
| 286 | `FieldArtifact.save` | `torch.save({'gradients': self.gradients, 'factors': self.factors}, root / 'tensors.pt')` |
| 287 | `FieldArtifact.save` | `(root / 'manifest.json').write_text(json.dumps({'identity': self.identity, 'content_hashes': self.content_hashes}, indent=2, sort_keys=True))` |
| 301 | `FieldArtifact.load` | `(root / 'manifest.json').read_text()` |
| 307 | `FieldArtifact.load` | `torch.load(root / 'tensors.pt', map_location='cpu', weights_only=True)` |
| 454 | `collect_field` | `Path(__file__).read_bytes()` |

### `src/wsp/features/rotation.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 229 | `numerical_environment` | `open('/proc/cpuinfo')` |

### `src/wsp/features/stats.py`

Repository imports (lexical): none.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/features/svd.py`

Repository imports (lexical): none.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/schema.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 133 | `write_manifest` | `path.parent.mkdir(parents=True, exist_ok=True)` |
| 134 | `write_manifest` | `path.write_text(json.dumps(fields, indent=2, sort_keys=True, default=str))` |

### `src/wsp/zoo/__init__.py`

Repository imports (lexical): `src/wsp/zoo/random_dir.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `src/wsp/zoo/apply.py`

Repository imports (lexical): `src/wsp/schema.py`, `src/wsp/zoo/common.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 22 | `entry_has_artifacts` | `(d / 'edit.json').exists()` |
| 23 | `entry_has_artifacts` | `(d / 'delta.safetensors').exists()` |
| 23 | `entry_has_artifacts` | `(d / LORA_ADAPTER_FILE).exists()` |
| 61 | `lora_factors` | `(lora_dir / 'adapter_config.json').read_text()` |
| 71 | `lora_factors` | `load_file(lora_dir / 'adapter_model.safetensors')` |
| 123 | `load_entry_delta` | `(d / 'delta.safetensors').exists()` |
| 124 | `load_entry_delta` | `load_file(d / 'delta.safetensors')` |
| 144 | `load_edited_model` | `(entry_dir / 'edit.json').read_text()` |
| 147 | `load_edited_model` | `load_file(base)` |

### `src/wsp/zoo/arch.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 221 | `read_text_config` | `AutoConfig.from_pretrained(config['base_model'], revision=config.get('base_revision', 'main'), **remote_code_kwargs(config['base_model'], config))` |

### `src/wsp/zoo/build.py`

Repository imports (lexical): `src/wsp/edits/kernels.py`, `src/wsp/edits/refusal.py`, `src/wsp/schema.py`, `src/wsp/zoo/__init__.py`, `src/wsp/zoo/arch.py`, `src/wsp/zoo/apply.py`, `src/wsp/zoo/common.py`, `src/wsp/zoo/random_dir.py`, `src/wsp/editor/zoo.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 39 | `load_config` | `path.open()` |
| 46 | `config_hash` | `path.open('rb')` |
| 186 | `_load_manifest` | `path.exists()` |
| 188 | `_load_manifest` | `path.read_text()` |
| 200 | `_load_manifest` | `path.write_text(''.join((json.dumps(r, sort_keys=True) + '\n' for r in [*headers, *rows])))` |
| 237 | `_write_entry` | `entry_dir.mkdir(parents=True, exist_ok=True)` |
| 240 | `_write_entry` | `(entry_dir / 'edit.json').write_text(desc.to_json())` |
| 241 | `_write_entry` | `save_file({k: v.to(dtype).contiguous() for k, v in delta.items()}, entry_dir / 'delta.safetensors')` |
| 247 | `_append_manifest` | `manifest_path.parent.mkdir(parents=True, exist_ok=True)` |
| 248 | `_append_manifest` | `manifest_path.open('a')` |
| 371 | `check_reusable` | `(entry / 'edit.json').read_text()` |
| 612 | `build_lora_entry` | `(final_dir / 'edit.json').read_text()` |
| 615 | `build_lora_entry` | `shutil.rmtree(stage, ignore_errors=True)` |
| 616 | `build_lora_entry` | `stage.parent.mkdir(parents=True, exist_ok=True)` |
| 622 | `build_lora_entry` | `(stage / 'edit.json').exists()` |
| 624 | `build_lora_entry` | `(stage / 'edit.json').write_text(desc.to_json())` |
| 625 | `build_lora_entry` | `shutil.rmtree(final_dir, ignore_errors=True)` |
| 649 | `_build_lora_in_memory` | `staging.mkdir(parents=True, exist_ok=True)` |
| 876 | `_load_base_state` | `transformers.AutoModelForCausalLM.from_pretrained(config['base_model'], revision=config.get('base_revision', 'main'), torch_dtype=dtype, device_map='cpu', **_arch.remote_code_kwargs(config['base_model'], config))` |
| 939 | `build_zoo` | `zoo_dir.mkdir(parents=True, exist_ok=True)` |

### `src/wsp/zoo/common.py`

Repository imports (lexical): none.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 16 | `resolve_base_revision` | `snapshot_download(repo_id=base_model, revision=revision or 'main', allow_patterns=['config.json'])` |

### `src/wsp/zoo/random_dir.py`

Repository imports (lexical): none.

No filesystem/loading site; in-memory definitions/imports only.

### `tests/conftest.py`

Repository imports (lexical): `src/wsp/zoo/__init__.py`, `src/wsp/zoo/build.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `tests/test_f12_additivity.py`

Repository imports (lexical): `scripts/f12_additivity_prepare.py`, `scripts/part_a_predictions.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 180 | `test_write_part_round_trip` | `load_file(str(d / 'delta.safetensors'))` |
| 184 | `test_write_part_round_trip` | `(d / 'variant.json').read_text()` |

### `tests/test_f12_population.py`

Repository imports (lexical): `scripts/f12_population.py`.

No filesystem/loading site; in-memory definitions/imports only.

### `tests/test_f12_tokens.py`

Repository imports (lexical): `scripts/a5_power_fields.py`, `scripts/f12_tokens.py`, `scripts/part_a_predictions.py`, `src/wsp/features/__init__.py`, `src/wsp/features/response_field.py`.

| Line | Containing definition | File/path operation (source expression) |
|---|---|---|
| 293 | `test_load_probe_hash_checked` | `save_file(t, str(tmp_path / f'{i:03d}.safetensors'))` |
| 303 | `test_load_probe_hash_checked` | `(tmp_path / 'probe_cache.json').write_text(json.dumps(rows))` |
| 331 | `test_stored_tensor_count` | `(tmp_path / 'd').mkdir()` |
| 332 | `test_stored_tensor_count` | `save_file({'a': torch.zeros(2), 'b': torch.zeros(3)}, str(tmp_path / 'd/delta.safetensors'))` |
| 334 | `test_stored_tensor_count` | `(tmp_path / 'l/lora').mkdir(parents=True)` |
| 335 | `test_stored_tensor_count` | `save_file({'x.lora_A.weight': torch.zeros(2, 3), 'x.lora_B.weight': torch.zeros(3, 2)}, str(tmp_path / 'l/lora/adapter_model.safetensors'))` |


Dynamic-import boundary: `wsp.zoo.build` can import optional `wsp.zoo.heretic_ara` / `wsp.zoo.lora_sft` inside its generator (542–544). The autouse test fixture imports build and patches its revision resolver only; it never calls that generator. Those optional generator modules and arbitrary third-party dynamic imports are not part of the actually imported F12/test dependency closure. The index deliberately follows the named repository imports even inside dormant functions; it does not claim that all such modules are imported at runtime.
