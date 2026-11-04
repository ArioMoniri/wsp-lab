# Board A final re-review: feat/p1-zoo-build

Reviewer: reviewer-ml (Board A). Head: d4da0295049322335a5420315ad9cc271098bc32.
Reviewed diff: `03384a9...d4da029`. Verdict: **REVISE**. Blocking: **yes — one remaining eligibility-policy defect**.
No implementation edits or GPU jobs.

## Findings

1. **High, blocking yes — changed eligibility thresholds do not quarantine existing production parents.** `src/wsp/zoo/build.py:556-557,605-620,965-967`. Eligibility thresholds are absent from reuse identity; LoRA reuse returns the old descriptor, registration preserves the old row, and parent filtering trusts the historical production label without applying the current policy. Reproduced independently using the existing fake-generator/toy-base fixtures in a temporary directory: build two lora_sft entries (steps=800), change YAML `eligibility.min_params.lora_sft.steps` to 10000, rebuild. Both returned rows and edit.json files still say `production`, while `classify_eligibility` under that config returns `smoke` with reason `steps=800 below production minimum 10000`. `parent_exclusion_reason` admits those rows by its production branch. Thus changing the configurable production policy silently leaves below-budget parents in the default research pool. Reclassify/reconcile metadata and parent eligibility under the effective policy, or explicitly version/fingerprint the policy and reject incompatible rows; cover reuse and historical parent selection with a regression. Preserve smoke inheritance and explicit opt-in behavior.

2. **Low, blocking no — manifest rows omit eligibility reasons.** `src/wsp/zoo/build.py:945-964`. edit.json contains reasons, but the row only carries eligibility. Copying `eligibility_reasons` into rows is cheap and makes the design note's promise accurate.

3. **Medium, blocking no — deterministic training device policy recommended.** `src/wsp/zoo/lora_sft.py:75,93,144`. Automatic dispatch depends on concurrent GPU pressure, while inputs target CUDA directly. Historical chain logs show allocation failures under heavy competing usage; offloaded training can also encounter device mismatch. Prefer explicit supported single-device placement (or explicit CPU mode) with a clear capacity failure and recorded placement. These observed failures abort generation rather than establish wrong completed artifacts; standalone generation is staged and chain generation finishes before chain entries are written. No evidence here makes this an artifact-correctness blocker. Successful GPU training/dispatch correctness is not certified by this CPU review.

4. **Low, blocking no — add top-level delta_format for evaluator interoperability.** `src/wsp/schema.py:43`, main `src/wsp/eval/run.py:163`. The evaluator reads top-level `delta_format`, then infers dense versus LoRA from files. A cheap explicit serialization field (`dense` for cumulative dense snapshots, `lora` for adapters) avoids inference; keep `stored_delta_format=dense_cumulative` as the separate semantic distinction. The inspected main evaluator has the inference fallback, without a warning at this call site. Current files remain interpretable.

## Disposition of prior three blockers

- **Identity/reuse: closed for the prior artifact-identity failure.** Base resolution occurs before descriptors; pinned revision reaches IDs, generator and manifest. reuse_identity is SHA-256 hashed into reuse_key and checked against existing descriptor and manifest before standalone reuse or dense overwrite. Tests exercise base-ref drift/new IDs, generator revision disagreement, dtype/global override/refusal fingerprint drift and rejection without overwriting artifacts, and dry-run/real ID equality. Applicable generator extras are included in sampled params. Eligibility policy consistency is the separate remaining finding above.
- **Eligibility: partially closed; remains blocking via finding 1.** Production/smoke labels and reasons exist, thresholds are configurable, smoke propagates through chains/task vectors, and tests exercise default smoke exclusion, explicit include-smoke, missing-artifact filtering, single-family filtering, legacy exclusion and incompatible pinned revision. Historical production labels bypass a subsequently stricter policy.
- **Chain semantics: closed.** Tests cover both lora_vs_base and sequential in params/provenance/manifest, prefix semantics, smoke inheritance, dense cumulative storage and distinct source format, and chain dry-run/real ID equality. docs/ZOO_IDENTITY.md correctly states H2 uses consecutive cumulative-snapshot differences and LoRA optimizes against the original base, not the accumulated state.

## Validation and artifact inspection

- Read reviewer role, review protocol, both prior reviews and docs/ZOO_IDENTITY.md; inspected requested diff and relevant generator/loader/evaluator code.
- `source scripts/env.sh && CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD/src pytest -q`: **144 passed, 6 deselected**, 15.36 seconds, three dependency deprecation warnings.
- `source scripts/env.sh && CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD/src python -m wsp.zoo.build --config configs/zoo_qwen3.5-0.8b.yaml --dry-run --n 3`: **exit 0**. CUDA hidden throughout; no GPU jobs.
- Independent threshold-change reproduction exited 0 and exposed the failure above; no repository test/code files were changed. An initial fixture invocation was corrected (the fixture returns a list, not a generator).
- Inspected worktree data/zoo/qwen3.5-0.8b: 30 legacy edit.json files, none with reuse_key. These remain rejected-run evidence and are excluded by the new legacy guard, including the former trials=2 ARA entry.
- Inspected all three real data/zoo/p1-zoo-r3-smoke edit.json files and manifest rows: e64b0cd99ab79585 -> 2e7ba46653bd43b9 -> 35faea127fe0117d. All have delta.safetensors, pinned base 2fc06364715b967f1860aea9cf38778875588b17, matching reuse-key hashes (independently recomputed), smoke status and edit.json reasons for steps=5/override. Semantics are lora_vs_base, sequential, lora_vs_base; stored format is dense_cumulative throughout, source formats lora, dense, lora. Manifest rows match these semantics but omit reasons. Artifacts record precommit SHA 03384a9, so they are wiring evidence, not clean-head production certification.
- data/runs/p1-zoo-r3/attempts.txt records attempt 1 rc=0 with 25853 MiB already used. Historical r3b/r3d and integration-chain logs show competing-memory OOMs. No GPU retry was performed.
- No new statistical claims, split metrics or bootstrap estimates are supplied by this change; production evaluation and full P1 completion remain outside this approval.

**REVISE — one remaining blocker: eligibility policy changes leave historical production parents admissible.**
