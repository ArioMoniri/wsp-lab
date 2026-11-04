# Zoo entry identity, eligibility and chain semantics

Decisions from Board A re-review of `feat/p1-zoo-build` (P1).

## 1. Identity and reuse

* **Base is pinned first.** `build_zoo` resolves `base_revision` (e.g. `main`) to a commit with
  `resolve_base_revision` before any id is built. `EditDescriptor.base_revision` (part of `id`),
  the refusal cache, `generate()` and the CPU base copy all use that commit; the requested ref is
  kept in `provenance.requested_base_revision` and the manifest header. Upstream drift therefore
  yields *new ids* instead of aliasing an old entry. (Breaking: pre-pin ids used `main`.)
  `generate()` returning another base commit raises `EntryIdentityMismatch`.
* **`reuse_key` rather than a bigger id.** The id stays the planned, dry-run-computable
  (family, sampled params, seed, pinned base, parent) hash, so dry-run ids equal real ids. What
  the id cannot see (refusal fingerprint, kernel variant, dtype, stored format, step semantics,
  parent's reuse_key, config overrides applicable to the family or global, and for chains the
  whole prefix of per-step identities) goes into `provenance.reuse_identity`; its SHA-256 is
  `provenance.reuse_key` in edit.json and `reuse_key` in the manifest row.
* **Check before skip/overwrite.** For every entry (LoRA before generation; dense before writing)
  an existing entry dir or manifest row with the same id must have the same `reuse_key`; equal =
  reused/rewritten deterministically; different or missing (legacy) = `EntryIdentityMismatch`.
  It is never reused or overwritten silently. To rebuild, move the old entry directory aside
  (artefact-less manifest rows are dropped on the next build).

## 2. Parent eligibility

* `eligibility` (`production|smoke`, `unclassified` for old files) plus `eligibility_reasons`
  are fields of edit.json and the manifest row. `smoke` = a sampled param below
  `eligibility.min_params.<family>.<param>` (defaults: `heretic_ara.trials` 20, `lora_sft.steps`
  50; set in the config) or any applicable `--override` (global, or scoped to the family; `n`
  excluded). Chain entries are smoke from the first smoke step onwards; a task vector is smoke
  if its parent is.
* `task_vector` parents must be single-family, have artefacts, a `reuse_key` and `eligibility`
  (else `legacy_unprovenanced`), the same pinned base (`incompatible_base`) and be `production`.
  `--include-smoke` (`build_zoo(include_smoke=True)`) also admits `smoke`; legacy and other-base
  rows are never admitted. Exclusion counts are recorded in `provenance.parent_pool_excluded`.
  Legacy artefacts (e.g. the 30 pre-policy rows incl. heretic_ara trials=2) stay on disk as rejected-run
  evidence but are not sampled; rebuild to get provenanced entries.

## 3. Chain semantics

Chain entries are **cumulative snapshots** (entry k = state after step k minus base, stored dense).
H2 per-step deltas = differences of consecutive snapshots; never apply a snapshot onto its parent.

* `params.step_semantics` (id-bearing), `provenance.step_semantics`, `provenance.chain_step_semantics`
  (steps so far) and the manifest row carry `lora_vs_base` (LoRA family: optimised against the
  *base checkpoint*, delta added to the accumulated state) or `sequential` (random_dir /
  abliteration edit the accumulated weights).
* `provenance.source_delta_format` (`lora|dense`) is the inner step's format;
  `provenance.stored_delta_format = dense_cumulative`; `delta_format: lora` is no longer copied
  onto chain entries.
* **Not done:** a *sequential* LoRA variant (train/optimise against the accumulated state) needs
  `heretic_ara.generate`/`lora_sft.generate` to start from a supplied state dict instead of the
  Hub checkpoint; deferred. Its entries would use `step_semantics: "lora_sequential"`.
