# Part A staging (G3 preparation): rotation and ladder variants, audits, reference plan

**No outcome exists.** Nothing was scored: no staged variant and none of the 19 missing references has an evaluation record, and no response, score or contrast was read or computed. This package is preparation only (Board A G2 acceptance, `docs/reviews/council/board-a-g2-2026-11-11.md`, "Exactly what downstream work may consume" item 5: Part A preparation is authorised, not G3 scoring or Part A inference).

## What was run

CPU only, no GPU. Data were read only, except the staging area `data/runs/a5/part_a/`. Nothing was written under `data/zoo` or `data/evals`.

```
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 PYTHONPATH=src HF_HUB_OFFLINE=1 \
  .venv/bin/python -m wsp.eval.response_intervention prepare --zoo data/zoo/qwen3.5-0.8b \
  --selection docs/prereg/a5_manifests/selection.json --references docs/prereg/a5_manifests/references.json \
  --text-groups docs/prereg/a5_manifests/text_groups.json --evals data/evals/qwen3.5-0.8b/tf_v1 \
  --config configs/eval_tf_v1.yaml --out data/runs/a5/part_a \
  --manifest-out docs/prereg/a5_manifests/part_a/manifest.json
```

- Code: commit `7f8b453c8549` (clean tree, `dirty: false`; `src/wsp` source hash `e59ed4941c1e2cb42f9285059ffda9dd0d8d2e4da95ff57591c86c07e1ca28cd`). The `response_intervention.py` bytes it ran are pinned in `../code_hashes.json` (re-pin (xiv), PREREG A5.1.2).
- Inputs: `selection.json` sha256 `393530365f87f981d5da3ecb2779c89b8e7bf98105b6c45d1824bf290e908c63` (registered picks loaded and checked, not re-selected; sorted-id sha256 `e2bd6cb2956c0fbe64194b3f1ebbb481248ecec45b8a729a59a54b9a2bac9d6b` equals the procedural freeze), `references.json` (bytes holders of alias predecessors), `text_groups.json` (64 registered windows for the reuse proofs), `configs/eval_tf_v1.yaml`.
- Wall time 4 h 58 min 33 s (2026-11-10 18:46:52 to 23:45:24 UTC), peak RSS 9.1 GB. The machine was shared (load average 10 to 26 from another job).
- An earlier attempt with 8 threads (started 18:13:46 UTC) was stopped by hand after 33 min, before any edit finished, because of thread oversubscription. It wrote only `stage.json`, with identical content. The configuration was the same apart from the thread settings.

## What was staged

Forty registered edits (30 lora_sft, 10 chain). Each got its three registered variants: the rotation R at s = 1 and the ladder at s = 0.25 and 0.5, all built at the actual predecessor and rounded to bf16 (A5.3.3). That is 120 variants built (40 rotation, 80 ladder). Rotation seeds follow `control_seeds.json:rotation` (`rotation_seed`, listed per edit in `manifest.json:rotation_seeds`). There is one generator per edit, and its initial state is cloned for each tensor.

- **Staged for scoring: 81 variants** (1 rotation, 80 ladder). Each is a zoo entry `data/runs/a5/part_a/qwen3.5-0.8b/<variant id>/` with `edit.json` (copy of the source edit's), `delta.safetensors`, `variant.json`, and for chain steps with a non-base predecessor also `increment.safetensors`. There are 98 delta files (81 `delta`, 17 `increment`), and each sha256 is in `manifest.json:delta_sha256`. `manifest.jsonl` (staged zoo rows) and `stage.json` sit next to them.
- **Excluded (whole rotation pair): 39 rotation variants.** Each holds only `variant.json`, with its per-tensor audits and exclusion reasons, and is not in the staged `manifest.jsonl`. The pair stays in the forty-ID ledger, and nothing is replaced (A5.3.1, F3(ii)).
- Convention (module docstring): root edits store the realised increment bf16(W + X) − W. Chain steps store the realised cumulative state minus base, because the scorer applies deltas to the base. Every tensor was checked to reproduce the intended bf16 state bit for bit under the scorer's arithmetic. Root ladders scale the raw stored delta Δ. Chain ladders scale the step's realised increment D, which is the only per-step object stored (`scale_source` in `variant.json`).
- **Flag (chain ladders).** A5.3 says scaled edits are built "by applying s·Δ_t at W_t and rounding, not by scaling D_t". For the 8 chain picks with a non-base predecessor, no per-step Δ is stored, only the cumulative bf16 snapshot. The pinned builder therefore applies s·D_t at W_t and rounds: bf16(W + s·D) − W, realised, as Board A's G2 item 2 requires for the prediction. The 30 lora_sft and the 2 root chain picks scale the stored Δ. This is the code's documented convention, used here unchanged. Whether it counts as the registered ladder direction for chain steps is for the chair to confirm before scoring.
- Disk use: 72,979,914,676 bytes (73.0 GB, 68.0 GiB) under `data/runs/a5/part_a/`. Delta files are dense fp32.

## Rotation audits (A5.3.3, F3(ii))

`rotation_audits.jsonl` (sha256 `f05fec4265ffd93c8c537100993d23262be95141369240fcad0a7e16ff474d35`) has one row per edited tensor (2534 rows). Each row gives the edit, predecessor, seed, tensor, shape, block shape and rank, rank_d/rank_r, norm and spectrum mismatch, support identity, orthogonality error, reconstruction residual, pre-rounding and rounding errors, nonzero counts and the verdict. Full per-tensor audits, including the logged moments, are in each rotation `variant.json`, whose sha256 is in `manifest.json:variant_json_sha256`. Per-edit summaries are in `manifest.json:edit_audits`.

No tensor had a pre-audit exclusion reason (`non_2d`, `non_finite`, `zero_block`, `rank_degenerate`, `not_single_matrix`), so all exclusions are `audit_failed`.

| family | tensors | pass | norm > 2e-2 | spectrum > 5e-2 | abs(rank diff) > 1 | recon > 1e-5 | orth > 1e-5 | support differs |
|---|---|---|---|---|---|---|---|---|
| lora_sft | 2160 | 462 | 796 | 1421 | 306 | 3 | 0 | 0 |
| chain | 374 | 73 | 225 | 265 | 78 | 2 | 0 | 0 |

**Coverage:** 1 of 40 rotation pairs pass. 6 of 2534 edited tensors are rotated in a passing pair. By family: lora_sft 30/30 excluded, chain 9/10 excluded. The only passing pair is `00f8111cbad8218f` (chain, predecessor `34ee195e55227908`, 6 tensors).

Descriptive reading of the failures (no rule is changed here). The registered audit compares the realised R_eff = bf16(W + R) − W with D_eff. Two patterns account for the failures:

- **Norm and spectrum.** Where D_eff is small relative to W's bf16 spacing, spreading the same spectrum over the whole block puts most entries below half a unit in the last place of W. Rounding then removes most of R (`rounding_error_rel` near 1). An example is the sparse rounding-residue increments of chain `0027efdf2eae7405`, about 1,800 nonzeros in blocks of roughly 700 × 1,300.
- **Rank.** Where norm and spectrum agree (for example, norm mismatch about 1e-3), bf16 rounding noise in R_eff often lifts its numerical rank at the 1e-3 relative tolerance by more than the ±1 allowed, sometimes by hundreds.

Consequence under the registered rules, stated and not decided here: at most one rotation pair can be resolved, which is fewer than the 15 that A5.3 requires. The rotation contrast is therefore **unresolved by its yield rule whatever the scores** (no replacement, no resampling). The ladder contrasts (s = 0.25 and 0.5, separate denominators, 40 edits each) are not affected. Whether to score, amend A5.3.3 prospectively, or stop is for the chair and Board A, not this staging step.

### Per-edit verdicts

| edit | family | tertile | predecessor (bytes holder) | tensors | failing | failing criteria (tensor counts) | rotation |
|---|---|---|---|---|---|---|---|
| `0027efdf2eae7405` | chain | high | `b7edbb9f8b339f52` | 32 | 32 | norm 22, spectrum 22, rank 32, recon 1 | excluded |
| `00f8111cbad8218f` | chain | middle | `34ee195e55227908` | 6 | 0 | none | pass |
| `014f8a9ec78db2be` | chain | middle | `base` | 18 | 18 | spectrum 14, rank 18 | excluded |
| `05531ce1db4df63f` | chain | high | `8ea1a0633ad30d22` | 72 | 15 | norm 3, spectrum 15 | excluded |
| `0559051354691a70` | chain | middle | `base` | 10 | 9 | rank 9 | excluded |
| `057dcfbedea3251f` | chain | low | `86ea1fe0ec777cf8` | 76 | 69 | norm 64, spectrum 64, rank 5 | excluded |
| `0712caa1d248514a` | lora_sft | high | `base` | 72 | 27 | spectrum 27, rank 2 | excluded |
| `07f9a50b29a7ef5c` | lora_sft | high | `base` | 72 | 23 | spectrum 4, rank 20 | excluded |
| `0a841d4082532e8b` | lora_sft | low | `base` | 72 | 72 | norm 70, spectrum 72 | excluded |
| `0bc0a0c4ad0520e0` | chain | low | `fe7f8926879f91db` | 72 | 72 | norm 69, spectrum 72 | excluded |
| `0d055869527177b4` | chain | low | `2114e313c73f15a6` | 72 | 72 | norm 67, spectrum 72 | excluded |
| `13b02fe2673cc296` | chain | low | `bc6edb1e3c913a10 (4b84f551adaf52ba)` | 6 | 6 | spectrum 6, rank 6, recon 1 | excluded |
| `14cdaabee246493b` | chain | high | `983bb1066d0fd006 (345a7db356689c69)` | 10 | 8 | rank 8 | excluded |
| `15a53586efc72a50` | lora_sft | high | `base` | 72 | 44 | rank 44 | excluded |
| `19d9a98cfb2db6e2` | lora_sft | middle | `base` | 72 | 71 | norm 17, spectrum 71, rank 3 | excluded |
| `1ab850a41051fe45` | lora_sft | low | `base` | 72 | 72 | norm 71, spectrum 72 | excluded |
| `1b2b671e5da5d625` | lora_sft | low | `base` | 72 | 72 | norm 46, spectrum 72, recon 1 | excluded |
| `1fc51f7164052d50` | lora_sft | low | `base` | 72 | 72 | norm 67, spectrum 72, rank 4 | excluded |
| `2233954b0cc3edc0` | lora_sft | high | `base` | 72 | 22 | rank 22 | excluded |
| `22659cefc904e29f` | lora_sft | low | `base` | 72 | 72 | norm 31, spectrum 72, rank 1 | excluded |
| `2971f30e5276edd9` | lora_sft | high | `base` | 72 | 23 | rank 23 | excluded |
| `2d1de332e36d8f00` | lora_sft | low | `base` | 72 | 72 | norm 46, spectrum 72 | excluded |
| `2d7f5f78fac4a9ae` | lora_sft | middle | `base` | 72 | 58 | norm 20, spectrum 58 | excluded |
| `2f61a4086395b719` | lora_sft | middle | `base` | 72 | 72 | norm 25, spectrum 72, rank 1 | excluded |
| `2fb8d03d2547f357` | lora_sft | middle | `base` | 72 | 72 | norm 43, spectrum 72 | excluded |
| `3499f7dcc5968d96` | lora_sft | middle | `base` | 72 | 72 | norm 42, spectrum 72 | excluded |
| `34e407b92a902282` | lora_sft | high | `base` | 72 | 60 | rank 60, recon 1 | excluded |
| `3996bea032c24200` | lora_sft | middle | `base` | 72 | 49 | norm 3, spectrum 49, rank 2 | excluded |
| `453d8c08966438e0` | lora_sft | low | `base` | 72 | 72 | norm 43, spectrum 72, rank 5 | excluded |
| `455b881d5de4e6e0` | lora_sft | low | `base` | 72 | 72 | norm 52, spectrum 72 | excluded |
| `4f918f6bbd264b34` | lora_sft | middle | `base` | 72 | 72 | norm 42, spectrum 72 | excluded |
| `5abd76a640cc2b9d` | lora_sft | high | `base` | 72 | 10 | spectrum 1, rank 9 | excluded |
| `5f80bd482d092938` | lora_sft | middle | `base` | 72 | 47 | norm 1, spectrum 47, rank 5, recon 1 | excluded |
| `61d1f61e25212068` | lora_sft | low | `base` | 72 | 72 | norm 68, spectrum 72 | excluded |
| `649bd035e79488e5` | lora_sft | low | `base` | 72 | 72 | norm 65, spectrum 72 | excluded |
| `72d3756e968e3209` | lora_sft | high | `base` | 72 | 57 | rank 57 | excluded |
| `7d03d89c0b0299e6` | lora_sft | middle | `base` | 72 | 46 | norm 1, spectrum 46, rank 2 | excluded |
| `85366c308158ea48` | lora_sft | high | `base` | 72 | 38 | spectrum 38, rank 3 | excluded |
| `8cfc1393b97300ab` | lora_sft | high | `base` | 72 | 43 | rank 43 | excluded |
| `983c5e13c59e0022` | lora_sft | middle | `base` | 72 | 72 | norm 43, spectrum 72 | excluded |

## Predecessors and alias holders

Predecessors: `2114e313c73f15a6`, `34ee195e55227908`, `86ea1fe0ec777cf8`, `8ea1a0633ad30d22`, `983bb1066d0fd006`, `b7edbb9f8b339f52`, `base`, `bc6edb1e3c913a10`, `fe7f8926879f91db`. Two predecessors have no bytes of their own and were loaded from their alias holders, as `references.json` lists:

- `983bb1066d0fd006` via `345a7db356689c69`: the holder's current `delta.safetensors` sha256 `f318c4dbae97f27cfa5d70d66067c2b94fe420363bde24d9afffe9abe027af26` equals the `entry_hashes` of the alias's record (run `20261107-105825-56ef1597`). The record is complete and primary, and its scoring identity and 64 WikiText windows match. F3(i) verdict: **pass**.
- `bc6edb1e3c913a10` via `4b84f551adaf52ba`: the holder's current `delta.safetensors` sha256 `b8f4a8d28e14314dd0e302268c1f11bb04f377e5141be5f594a6a90651f2983a` equals the `entry_hashes` of the alias's record (run `20261107-105825-56ef1597`). The record is complete and primary, and its scoring identity and 64 WikiText windows match. F3(i) verdict: **pass**.

The staged rows of these chain variants carry `predecessor_bytes_holder`, so the scoring plan checks the holder's bytes.

## Reference plan (A5.3.4)

Each reference was checked with the F3(i) reuse proof: complete and primary record, effective config hash, model revision, benchmark counts, entry bytes (the holder's bytes for an alias) and the 64 registered windows. Of the 49 references (40 originals and 9 predecessors, including the base), 30 pass and are reused. The full proofs are in `manifest.json:scoring.reuse_proofs`.

- n_P = 0: every predecessor reference is on file.
- n_D = 19: originals with no record (`no_record`), all lora_sft. They must be scored because all forty original records are required (BC2 edit 3): `1ab850a41051fe45`, `1b2b671e5da5d625`, `1fc51f7164052d50`, `22659cefc904e29f`, `2971f30e5276edd9`, `2d1de332e36d8f00`, `2d7f5f78fac4a9ae`, `2fb8d03d2547f357`, `3499f7dcc5968d96`, `3996bea032c24200`, `453d8c08966438e0`, `455b881d5de4e6e0`, `4f918f6bbd264b34`, `61d1f61e25212068`, `649bd035e79488e5`, `72d3756e968e3209`, `7d03d89c0b0299e6`, `85366c308158ea48`, `983c5e13c59e0022`.
- These agree with `references.json` (n_D = 19, n_P = 0, same ids).

**Priced scoring count: 100 entries = 81 staged variants + 19 references, × 1.04 min = 104.00 min.** The registered launch price of 120 + 19 = 139 entries (144.56 min) assumed no exclusions. The difference is the 39 audit-excluded rotations only. Preprocessing and retries are not priced (A5.3.4 stop-and-reprice).

## Scoring later (not run; not authorised by this commit)

On the exclusive GPU, through agent-tmux, from the repository root, after G3 scoring is authorised:

```
agent-tmux a5-part-a-score env PYTHONPATH=src .venv/bin/python -m wsp.eval.response_intervention score \
  --stage data/runs/a5/part_a/qwen3.5-0.8b \
  --manifest docs/prereg/a5_manifests/part_a/manifest.json --config configs/eval_tf_v1.yaml
```

This runs `wsp.eval.batch` twice: on the staged zoo for the 81 variants, then on `data/zoo/qwen3.5-0.8b` for the 19 missing originals. Both write TF-v1 records to `data/evals/qwen3.5-0.8b/tf_v1/<id>/`. Add `--dry-run` to print the two commands without running them. Before it runs, the staged bytes should be re-hashed against `manifest.json:delta_sha256`.

## Files

| file | content |
|---|---|
| `manifest.json` | A5.11 Part A launch manifest (schema `a5-part-a-launch-v1`): selection and ledger source, variant ids and kinds, excluded ids and reasons, delta and `variant.json` sha256, rotation seeds, predecessor ids and bytes holders, per-edit audits and coverage, reference plan with reuse proofs, priced count, disk bytes, provenance, `scored: false` |
| `rotation_audits.jsonl` | per-tensor rotation audit rows (F3(ii)) |
| `README.md` | this file |
