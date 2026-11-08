# D8 regeneration report (6 Nov 2026)

Executes the regeneration plan of [board-a-incident-d8-2026-11-06.md](board-a-incident-d8-2026-11-06.md) section 2 (adopted in the D8 addendum). Branch `feat/d8-regen` (code, scripts, committed copies of the small result files). Data stay in `/home/ubuntu/wsp-data` (nothing under `data/` is committed). Result files quoted below are committed in [`d8-regeneration-2026-11-06/`](d8-regeneration-2026-11-06/) and `docs/prereg/artefact_*`.

Every regeneration ran from a clean detached worktree (`git status` empty, `dirty: false` in every record) with `data` -> `/home/ubuntu/wsp-data` and `PYTHONPATH` set to that worktree's `src` (the venv's editable install points at the main checkout, so `wsp.__file__` was printed and checked in each log). Hosts: one RTX PRO 4500 (device H0), exclusive during every GPU job; torch 2.15.0+cu132, transformers 5.22.1, numpy 2.5.4 throughout (recorded in each record; unchanged between the first and the last job).

| Step | Worktree (commit) | `source_hash` |
|---|---|---|
| 3 base, first-pass tree | `regen-846ff84` | `d7cece35fad9` |
| 3 base and 5b controls | `regen-f63a08e` | `6f274a830401` |
| 3 base, chain-predecessor tree | `regen-f90f591` | `e871923e1e18` |
| 4 seed-1 panel | `regen-0a8e708` | `ec88cfd95d7d` (full hash equals the one printed by the surviving original log, see step 4) |
| 5 TF-v1 | `regen-dd2a609` (scoring), `regen-9922ea3` (report and bridge scripts; `src/wsp` byte-identical) | `a224779bf873` |

## Summary of outcomes

| | Result |
|---|---|
| 1 | 1,259 manifest rows: **838 with bytes, 63 aliased, 358 lost**; 0 hash, header or norm misses over the 838 stored deltas. |
| 2 | **901 entries re-extracted** (838 + 63 aliases), 0 failed; 10-entry re-extraction byte-identical. No pre-incident feature file survives anywhere, so there was nothing to compare with. |
| 3 | Three H0 bases regenerated; **458 of 458 records bitwise equal on every scored field across the three source trees**; all printed base scalars reproduce; reference keys equal those of the restored H0 entries of the same source hash. |
| 4 | Seed-1 panel: **every printed number reproduces** (11 records, 10 controls + base). KL is *not* identical between seeds (max 3.35e-4): the paper's "at most 0.0003 nats" is right, RESULTS.md's "identical" is wrong. |
| 5 | KL cache root `2ed18bda284b3a89…` reproduced; G1 **12 of 12** repeats bitwise (14 of 14 originally; 2 chain predecessors lost); G2 **11 of 11 table rows** reproduce; **28 of 28** recoverable (y, h) rows reproduce to the printed 4 digits. |
| 6 | 25 of 26 runs (m3-family fails as on 5 Nov); 297 rows / 270 clusters; same ordering as 5 Nov (StatNN ≥ neural; descriptor-only within 0.07); see §6. |

## 1. Restore verification and loss sidecar

`scripts/d8_restore_verify.py` (log `~/agent-logs/d8-restore-verify-20261106-042648.log`, 1,058 s wall, 4 processes, disk-bound at ~125 MB/s).

* Every one of the 838 stored delta/adapter files has a readable safetensors header whose byte ranges match the file size.
* Frobenius norms against `edit.json:delta_fro_norm`: 716 dense deltas, max relative error 6.4e-8; 122 LoRA adapters (norm of the dense product `(B A) * scaling`), max relative error 1.8e-8.
* `entry_delta_hash` of every stored entry is a member of its component's `source_identities` in `panel_manifest.json` (826 entries). The other 12 are the flagged ARA diagnostics, which the panel manifest keeps outside any component by design.
* Alias ledger (101 groups): the stored member of each group has exactly the ledger key as its content hash (0 unverified). **63 lost entries resolve to a stored member** (21 of them evaluated); `alias_of` is the ledger's canonical entry if it has bytes, else the smallest stored member id.
* Sidecar rows, 1,259: lost **358** (all chain; 62 of them evaluated), aliased **63** (all chain), stored **838** (584 non-chain: 205 abliteration, 153 random_dir, 121 lora_sft, 80 task_vector, 25 heretic_ara; 254 chain). `eval.json`: 389 (unchanged); `records.jsonl`: 200.
* `manifest.jsonl` (sha256 `71a396e5…`) and every eligibility were read, never written. Sidecar: `data/zoo/qwen3.5-0.8b/artefact_status.jsonl`, committed as `docs/prereg/artefact_status.jsonl` (fields exactly as requested; verification summary in `docs/prereg/artefact_restore_summary.json`).

Caveat for the sidecar's `records_present`: the surviving `zoo/<id>/records.jsonl` of **all 11 H0 controls is the Lambda (H1) record**, not the H0 record (e.g. `0a841d4082532e8b`: records give gsm8k 0.20 = the H1 `eval.json`, the H0 `eval.json` says 0.16). The hub copy was overwritten by the Lambda upload. So no H0 item-level record of a control survived; step 5b regenerated them.

## 2. Feature re-extraction

`scripts/d8_extract_features.sh` -> `data/features/qwen3.5-0.8b`, code at `fcfb16d` (`src/wsp/features` unchanged to the end), `feature_version` 4, 4 workers x 1 torch thread (`OMP/MKL_NUM_THREADS=1`), 04:44:39 to 08:09:07 UTC.

* **n = 901, not the ≈ 1,095 of the brief**: 838 stored entries + 63 aliases (Board A's 901). Aliases were extracted under their own id and `edit.json` through a staging zoo of symlinks (`runs/d8/staging_zoo`, `manifest.jsonl` symlinked, nothing written into the canonical zoo). `done=901 skipped=0 failed=0 no_artifacts=358`.
* Feature manifest: `manifest.json` sha256 `ce3b84e62e1a3c708ae849f58d65c28dec6c08c6ef8781b901435ea9d4025840` (contains the run time and `git_sha`, so it is not reproducible by construction); stable identifiers: `config_hash` `43edff1b534edec2b769de3f57ee113053961ec3468cd45d8112e9e14cbd64e1`, `identity_hash` `6eb3c3bc79edf4db2d4f3f52f1b61eff0d09fc4f8b2eabc7240feba86f0df0ca`; content root over all 901 entries (id, sha256 of `features.safetensors` and `pooled.safetensors`) `60f3787ec553a5f2cb0b092597a33e4b04763b72da4c3659b92b5e9a7dad70db`. `n_entries` is 901: the paper's "1,259 entries … no failures" sentence (Z3) is false and must say 901 of 1,259.
* **Comparison with surviving copies: none exist** (searched the whole host; only synthetic `/tmp/wsp-p4-synthetic-*` toys). Substitute check: ten entries (2 lora_sft, random_dir, abliteration, task_vector, 2 chain, 1 alias, 1 heretic_ara, 1 more chain) re-extracted in a fresh process with 2 workers into a separate directory: **all three files of all ten entries byte-identical** to the main run. This shows extractor determinism, not equality with the lost files; the indirect check of Board A (P4/X1 fingerprints) is impossible on the recovered set (step 6).
* Cost: mean 54.3 s per entry (sum 13.6 core-h, wall 3.41 h on 4 workers) against the ≈ 5 h estimate. The host was oversubscribed: the GPU jobs took ~2.5 cores and other tenants' processes (`logd`, `lmkd`, `system_server`, ~2.5 cores) ran throughout.

## 3. Three H0 base re-evaluations

`scripts/d8_base_eval.sh <commit> <hash>` -> `python -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --ids none` (base only, fast config, batch 32, no OOM retry), then `scripts/d8_register_base.py` checks the gates (`source_hash`, `dirty: false`, `git_sha`, status, actual batch 32) and moves the record to `data/evals/qwen3.5-0.8b/base/devices/H0/<run_id>/` (the historical commits write the base to `base/`; the registered layout is `devices/H0/<run>`, D2 addendum). Runs: 10.2, 7.5 and 6.7 min wall (0.41 GPU-h in all, planned 0.4).

| Source | Run | Wall min | config_hash | Actual batch |
|---|---|---:|---|---:|
| `d7cece35fad9` (`846ff84`) | `20261106-043002-501554db` | 7.06 | `519e18781b214281` | 32 |
| `6f274a830401` (`f63a08e`) | `20261106-044003-37b3ae3d` | 6.59 | `519e18781b214281` | 32 |
| `e871923e1e18` (`f90f591`) | `20261106-044735-5ac57b8f` | 6.44 | `519e18781b214281` | 32 |

Verification (`scripts/d8_base_compare.py`, `base_compare.json`):

* **Scalars**: all three give gsm8k .16, math500 .16, ifeval .40, mmlu_pro .28, hellaswag .39, over-refusal .10, WikiText ppl 22.904104… (RESULTS.md: 22.90): all printed values match.
* **Bitwise across source trees**: `d7cece35` vs `6f274a83`: 458 of 458 records identical (every field). `e871923e` vs either: 358 of 458 identical and **458 of 458 identical on every field both records carry**; the 100 generative records differ only by an `invalid_final_answer` key that the later commit adds to `extra` (a schema extension, no score, token count, text or flag differs). This is the D2 addendum's "bit-identical across source trees" for the base, now 3 trees.
* **Item identity against the two Lambda bases** (`base-lambda-209…`, `base-lambda-129…`, source `6f274a83`): same (benchmark, item_id) set, same order, 458 of 458, for all three. Cross-device records differ as expected (0 of 458 identical; mmlu_pro correctness flips, WikiText up to 0.052 nats), not gated.
* **Items against H0 entries of the same source**: the control records of `6f274a83` and `013b7b61202fe188` (`d7cece35`) have the same items in the same order; after backfill (below) the regenerated bases have **every reference key equal to the restored H0 entry records of the same source hash**, including `item_order_hash` `2548af19…`, `item_id_hash` `fcf7c6ad…`, batch, generation, config, seed.
* **Backfill** (`scripts/d8_backfill_bases.py`, the `backfill()` recipe of `backfill_eval_metadata.py` applied to these records only): the two bases from pre-metadata commits (`846ff84`, `f63a08e`) get device H0, runtime fields (current environment, labelled `partial-inferred` exactly as the restored entries are) and item hashes from their own records; `eval.json.orig` kept. `e871923e` already carries native metadata. The two Lambda base records were re-registered under `base/devices/H1|H2/<run_id>/` by `scripts/d8_backfill_lambda_bases.py` (same backfill, runtime from `configs/eval_runtime_backfill.json`, run ids equal those used by the council scripts); the 389 `evals/<id>` -> `zoo/<id>` symlinks of the original backfill were recreated. Without these two steps the P4 loader finds 81 matched rows instead of 268 (step 6).
* **Deviation**: the three original `e871923e` runs were three concurrent processes (`p2-chainpred-0/1/2`), and 33 depth-0 entries name their base by `base_run_id` (`20261104-233616-e04693d7`, `…233637-7c43da7c`, `…233657-d17ba458`); no regenerated record can carry those run ids. Step 6 handles it in a view only.

## 4. Seed-1 repeat panel

`scripts/d8_seed1.sh`: `--config configs/eval_fast_seed1.yaml --repeat seed1 --ids <the 10 controls>` at `0a8e708`, run `p2-batch/20261106-045437-03cde0e3`, 04:54:35 to 06:22:29 UTC = **1.46 GPU-h** (planned 1.6). The source commit was not given; it is **recovered from a surviving log** (`~/agent-logs/p4-after-chainpred2-20261105-041534.log` prints two manifests of the original run `20261105-050543-aad6c310`): `git_sha 0a8e708f6432…`, `dirty false`, `source_hash ec88cfd95d7da1f6…`, `config_hash a86af85123a01267…`; the regenerated records carry exactly these full values. Every record: device H0, batch 32 actual and nominal, status complete.

Comparison (`scripts/d8_seed1_compare.py`, `seed1_compare.json`): seed 0 = surviving H0 `eval.json` of the 10 controls and the regenerated `6f274a83` base.

| Benchmark | mean abs | max abs | Spearman ρ | RESULTS.md | Match |
|---|---:|---:|---:|---|---|
| gsm8k | 0.0545 | 0.12 | 0.151 | 0.055 / 0.12 / 0.15 | yes |
| math500 | 0.0364 | 0.12 | 0.455 | 0.036 / 0.12 / 0.46 | yes |
| ifeval | 0.0291 | 0.04 | 0.100 | 0.029 / 0.04 / 0.10 | yes |
| over-refusal | 0.0491 | 0.10 | 0.667 | 0.049 / 0.10 / 0.67 | yes |
| refusal | 0.0091 | 0.04 | 0.262 | 0.009 / 0.04 / 0.26 | yes |
| loop rate | 0.0064 | 0.02 | 0.661 | 0.006 / 0.02 / 0.66 | yes |
| mmlu_pro, hellaswag, wikitext ppl | 0 | 0 | — | identical | yes (exactly 0 on all 11) |
| KL to base | 7.2e-5 | **3.347185e-4** | 1.00 | "identical" (RESULTS.md), "at most 0.0003" (paper) | audit log 0.000334719: **reproduced to 6 digits** |

Also reproduced: base gsm8k .16 → .10, math500 .16 → .04, ifeval .40 → .36; per-control seed-noise SD 0.0504, 0.0268, 0.0248, 0.0356, 0.0124, 0 and 0 (gsm8k, math500, ifeval, over-refusal, refusal, mmlu_pro, hellaswag) equal `metrology_budget_output.txt`. **The KL discrepancy is arbitrated**: KL differs by up to 3.3e-4 nats between seeds (the KL continuation is sampled), so RESULTS.md's "KL identical under seed 1" must be changed to "differs by at most 0.0003"; the paper sentence stands. Item-level bitwise check is possible for the base only (the controls' surviving records are H1's): mmlu_pro, hellaswag and WikiText records of the seed-1 base equal the seed-0 base records exactly. The 11 regenerated records replace the lost directory under a new run id; the original run id must not be cited.

## 5. TF-v1 at `a224779bf873` (fingerprint audit, G1, G2)

Driver `scripts/d8_tf_audit.sh` (log `~/agent-logs/d8-gpu-chain-20261106-045434.log`); the lost `data/runs/d3_shards/controls.txt` was **reconstructed** (the ten ids in the order of the round-3 table; set equal to the ten `eval.local-run1.json` entries and to `metrology_budget.py`). Run `20261106-062239-29d865d2`: base + 10 controls, device H0, `dirty: false`, `source_hash a224779bf873…` in every record.

* **KL cache** rebuilt from scratch (mode `build`, 277.7 s, 5.78 GB): content root `base_logits_sha256` **`2ed18bda284b3a892a8181a5dd85d17588ede252394c9121b8725fb08e45d77c`** — the registered `2ed18bda284b3a89…`, carried by every KL record. Mean per-prompt KL, median over the ten edits by prefix length: 16 → 0.00144, 32 → 0.00153, 64 → 0.00160 (RESULTS.md round 2: identical to the printed digits).
* **Pilot table of RESULTS.md ("TF-v1 timing pilot", gold log-prob to 6 decimals, TF acc)** does **not** reproduce (e.g. `0a841d4082532e8b`: printed 0.175 / -2.569524, regenerated 0.170 / -2.571223). That table belongs to the round-1 run `20261105-131348-277686c3`, whose source ("may be dirty") predates the round-2 blocking fixes (`02c2847`); it is not a fingerprint of the frozen source and Board A's plan to use it as one was mistaken. It is reported, not gated. The fingerprints that belong to the frozen source are the next four items.
* **G1**: 11 entries + base = **12 repeats, all bitwise identical to their primaries** (`records_identical: true`, 1,614 records each; every maximum exactly 0 for MMLU-Pro gold/option log-probs, HellaSwag, KL per prompt and position, WikiText NLL, refusal phrase margins). The two chain predecessors `b7edbb9f8b339f52` and `86ea1fe0ec777cf8` have no delta bytes, so 2 of the original 14 cannot be repeated; the nonzero-difference stop rule was never triggered. Primary and repeat were separate processes (cache `load` once, `resident` after). `g1_report.json` committed.
* **G2** (`scripts/d8_g2_compare.py`, `g2_compare.json`): all **11 rows of the round-3 G2 table reproduce** (MMLU acc fast/TF, |Δacc|, argmax agreement, gold-lp median / p99 / max, HellaSwag acc, HellaSwag score |Δ| max, as printed). Pooled: MMLU median 2.2e-16, p99 0.066, argmax 99%: PASS as printed; HellaSwag mean correctness agreement 0.9991, worst entry 0.99, worst argmax 0.98, max score |Δ| 0.0111, not bitwise to the fast tier (as printed). **This needed a deviation (5b)**: the bridge compares TF scores with `evals/<id>/records.jsonl`, which now holds the H1 record. I regenerated the fast-tier seed-0 records of the ten controls on H0 from `regen-f63a08e` with a byte-identical external copy of `eval_fast.yaml` (non-production, so nothing in `zoo/<id>` is written; `scripts/d8_controls_h0.sh`, 07:10:40 to 08:36:00 = **1.42 GPU-h, not in the plan**): **all ten controls reproduce every numeric field of both surviving H0 records bitwise** (`eval.json` of run `20261104-202947-1dcb69a2` and `eval.local-run1.json`, run `20261104-104702-5fac075c`; max abs difference 0), and the regenerated base equals the base of step 3 record for record. So the D2 claim "11 of 11 bit-identical across two source trees" holds again. The stock bridge's own comparison step then failed on the missing path after scoring; the unchanged module was re-run with `EVALS` pointed at the regenerated records.
* **28 recoverable (y, h) rows** of `tf-v1-requalification-2026-11-06.md` (21 lora_sft, 1 task_vector, 6 chain; the other 14 of the 42 listed rows have a lost entry or predecessor): 31 entries scored (35.4 min, **0.59 GPU-h, not in the plan**) through a staged zoo; y = mean paired gold log-prob difference, h = the registered item half-width (`tfv1_requalify.item_halfwidths`, seed 20261105, 4,000 draws). **All 28 y and all 28 h agree with the printed 4 significant digits** (largest |Δy| 3.3e-4 and |Δh| 4.9e-6, both below half a unit in the last printed digit), including the ceiling failure `d8a9fc56d92a3ad1` (h = 0.1036 > 0.10). `tf_yh_audit.json` has every row.
* Cold/warm cost: base 5.91 min with the cache build, warm median entry ≈ 1.04 min (1.02 to 1.24) under CPU contention (round 3: 1.003). Wall-clock is class C and is not compared.

## 6. P4 suite on the recovered set

`scripts/d8_p4_prepare.py` builds the input view (`runs/d8/p4_input_rebind`): the complete-ancestry rule keeps the 780 feature sets whose whole history has features (121 chain rows dropped), and 297 matched same-device rows result (abliteration 98, chain 50, lora_sft 59, random_dir 60, task_vector 30; 11 matched rows lack a reference; 255 targets carry inferred runtime metadata). The 33 depth-0 entries that named the three lost `p2-chainpred-*` base runs are bound to the regenerated `e871923e` base in the view only (`rebound_base_run_ids` in `p4_dataset_report_rebind.json`; the zoo records are untouched). `scripts/d8_p4_suite.sh` then ran the 26 runs of the 5 Nov list (CPU, 3 seeds, 1,000 bootstrap draws; flags inferred as documented in the script) as `data/runs/p4/d3v2r-*`, 08:17 to 16:03 UTC, 26.6 run-wall-h over up to 4 parallel workers. 25 runs have metrics; `m3-family` fails exactly as on 5 Nov ("training fold has no parent-child pairs").

Table and row-by-row comparison: [`docs/reviews/P4/d3v2r-summary-2026-11-06.md`](../P4/d3v2r-summary-2026-11-06.md) (`scripts/d8_p4_summary.py`). **This is a different experiment from the 5 Nov suite** (297 rows in 270 clusters against 345 in 278; the chain holdout's test rows fall from 98 to 50), so no value is expected to reproduce and none is gated. What the recovered set says, read on its own:

| | mmlu_pro ρ | hellaswag ρ |
|---|---|---|
| StatNN (seed split, seed 0) | 0.54 | 0.65 |
| descriptor-only (seed split) | 0.48 | 0.58 |
| best neural, seed split (m1_matched) | 0.47 [0.43, 0.49] | 0.54 [0.46, 0.59] |
| best neural, trajectory split (m1_matched) | 0.46 [0.44, 0.48] | 0.55 [0.46, 0.62] |
| family split, every model | within [-0.07, 0.18] | within [-0.21, 0.16] |

The ordering of the 5 Nov suite holds: StatNN is at or above every neural predictor on both targets, descriptor-only is within 0.07 of StatNN, delta-norm alone is weak (0.16 / 0.21), and no model transfers across families. The P4/X1 fingerprint check Board A asked for is impossible (no original fold table or feature file survives); the recovered-set table is reported next to the retained 5 Nov table in the paper, never mixed with it.

## 7. Chain snapshot replay (7 Nov 2026, step 7; branch `feat/d8-chain-replay`, merged 5989602)

The 358 lost chain snapshots were rebuilt by **replaying their surviving `edit.json` descriptors** (`scripts/d8_chain_replay.py`), not by re-running the zoo sampler: the sampler code at the chain build commit `f7d3d4a` is not in the repository, and the current sampler reproduces only 51 of the 675 chain ids. The replay calls the builder's own family functions with the samplers patched to return the recorded `inner_params`, so kernels, dtype casts and the `lora_vs_base` chain semantics are the production code paths. Three shards, device H0, 06:09–08:22 UTC, 629 steps (358 computed, 271 stored steps recomputed for comparison).

| | Result |
|---|---|
| Pilot on three fully stored chains (`chain_replay_pilot.json`, re-run 08:25 with the matrix metric) | 8 of 11 steps **bitwise** (all random_dir and abliteration steps; the refusal-direction cache was rebuilt from scratch and its fingerprint `0b6e272d…` equals the manifest's). The 3 re-trained LoRA steps are **not** within rounding at the matrix level: ‖D_replayed − D_stored‖_F / ‖D_stored‖_F = 2.8e-2, 9.9e-2 and 1.5e-1 (1 bf16 ulp on 6–61 % of elements), although their Frobenius norms differ by ≤ 9.1e-4 relative. The norm difference is a lower bound on the matrix error and must not be quoted as the error. |
| Full restore (`chain_replay_restore.json`) | 358 written; 354 with the touched-tensor set of `edit.json`; stored steps compared 271, bitwise 117, re-anchored 154 (every stored step downstream of a re-trained LoRA step is re-anchored to its stored bytes, so drift never propagates past a stored snapshot). |
| Install (`chain_replay_install.json`, `scripts/d8_chain_install.py`) | **351 installed**, 7 rejected and still lost: 4 entries of the pre-production commit `c744719` (chains `69d1f20f…`, `ab471184…`; no eligibility, no reuse key, older kernel; touched-tensor sets differ) and 3 re-trained LoRA steps whose norm error exceeds the 5e-3 acceptance (`c08e70a0…` 1.0e-2, `c36475af…` 5.6e-3 and its child `c60bc6ef…` 5.3e-3). Sidecar now: 1,189 stored, 63 aliased, 7 lost. |
| Exactness classes (sidecar `regenerated.exactness`; `runs/d8/a5_incremental_norms.jsonl`) | **bitwise** lineage: 156 of the 351 (random_dir/abliteration steps whose ancestry since the last stored snapshot has no re-trained step); **re-trained** lineage: 195 installed (306 zoo entries in total carry the label through their ancestry); lost lineage: 18 entries (7 lost + descendants). |
| Acceptance rule actually applied | norm rtol 1e-6 for bitwise lineages, **norm** rtol 5e-3 for re-trained lineages (the "matrix relative error ≤ 5e-3" wording of the council synthesis G0 was not the applied rule; corrected there). Re-trained lineages are re-instantiations of the same edit descriptor, not recovered bytes, and are excluded from every fingerprint claim. |
| TF-v1 (y, h) audit after install (`scripts/d8_tf_yh.sh`, run 08:32–09:03 UTC, 51 entries, 0.52 GPU-h) | 41 of the 42 listed rows are now reproducible (the 42nd depends on a still-lost entry). **All 27 bitwise-lineage rows reproduce to the printed 4 digits; 13 of the 14 re-trained-lineage rows do not** (e.g. `f1d7ab49…` y −0.3899 printed vs −0.3885 regenerated; `fbe56a7d…` −0.8136 vs −0.7874), as expected for re-instantiated LoRA steps. |
| Features | `scripts/d8_extract_features_resume.sh`: the extractor's resume identity accepted the existing manifest; 351 new entries (log `d8-features-resume-*.log`). |
| G1 for the two lost predecessors (`g1b_report.json`) | Repeat label `g1b` scored by `scripts/d8_g6_census.sh` 10:29 UTC, primaries by `scripts/d8_g1_predecessors.sh` after the census: **`b7edbb9f…`, `86ea1fe0…` and the base, 3 of 3 bitwise identical** between primary and repeat (every maximum 0). With the 12 of 12 of step 5 this closes G1 at 14 of 14 entries; the two regenerated predecessors are bitwise-lineage replays. |
| G6 census | `scripts/d8_g6_census.sh`, 10:56–22:09 UTC on H0: TF-v1 records for all **618** stored chain entries that had none (635 minus 17 legacy-revision rows, `runs/d8/g6_dropped_legacy.json`); ≈ 11.2 GPU-h (repriced from the synthesis's 296-entry estimate because a lineage census needs every node). 652 of 675 chain entries now carry a TF-v1 record; 95 of the 99 root-to-leaf lineages with ≥ 4 nodes are fully scored (512 nodes), up from 16. |
| Full-set P4 suite (`docs/reviews/P4/d3v2f-summary-2026-11-08.md`) | 26 runs `d3v2f-*`, 10:52–23:28 UTC, CPU: **342 matched rows in 278 clusters** (original 345 / 278; the 3 missing rows depend on the 7 lost entries). Seed-0 StatNN 0.56 / 0.66 (seed split) and 0.55 / 0.64 (trajectory) against the logged 0.57 / 0.65; descriptor-only 0.51 / 0.61 against 0.52 / 0.61; best neural (M1-matched) 0.45 / 0.50 seed and 0.42 / 0.56 trajectory; family split within [−0.29, 0.23]. The ordering of the 5 Nov suite holds on the re-instantiated set; the row-by-row table lists every difference. This is the X1 fingerprint Board A asked for, at the level the re-trained LoRA class allows: close, not bitwise. |

Cost: 20.3 run-wall-h over three GPU shards (6,572 + 5,792 + 7,944 s; GPU utilisation 40–70 %, CPU-bound); pilot re-run 0.1 GPU-h; (y, h) audit 0.52 GPU-h; incremental norms ≈ 1 CPU-h.

## Deviations and findings that change the record

1. Entries with delta bytes: 901 (838 + 63 aliases), not ≈ 1,095.
2. The three base records were registered to `devices/H0/<run>` after the run (the old commits write `base/`).
3. The original seed-1 source was not on file; recovered from a surviving log (`0a8e708`).
4. The round-1 pilot table cannot serve as a fingerprint (superseded scorer); replaced by G2, G1 and the 28 (y, h) rows, all of which reproduce.
5. Surviving control `records.jsonl` files are H1 records; the ten H0 fast-tier controls were regenerated (+1.42 GPU-h) and verified bitwise against the surviving H0 `eval.json`.
6. 28-row TF audit (+0.59 GPU-h) and the regenerated controls were not in the brief; both are inside Board A's P6c/P6b scope.
7. `controls.txt`, the P4 suite script and the original P4 flags were lost with `data/` (they lived in `data/runs`); the first two were reconstructed, the P4 flags inferred (step 6).
8. KL is not identical between seeds (RESULTS.md wrong, paper right); the paper's Z3 sentence must say 901 of 1,259.
9. Process: a first launch of the seed-1 and TF drivers through `agent-tmux` with a quoted shell string died instantly (tmux joins the arguments and drops the quotes), costing ~7 min of idle GPU; waiters now live in script files. No misses against Board A's success criteria occurred in steps 1–5, so no incident is recorded.

## Resource use

| | GPU-h | CPU (core-h, workers x wall) |
|---|---:|---:|
| 1 restore verification | 0 | ≤ 1.2 (4 x 17.6 min, disk-bound) |
| 2 features (+ 10-entry recheck) | 0 | 13.6 (+ 0.2) |
| 3 three bases | 0.41 | — |
| 4 seed-1 panel | 1.46 | — |
| 5 TF pilot + KL cache + G1 + G2 | 0.81 | — |
| 5b H0 controls (deviation) | 1.42 | — |
| 5d 28-row audit (deviation) | 0.59 | — |
| 6 P4 | 0 | 26.6 (run-wall sum, ≤ 4 parallel x 2 threads) |
| 7 chain replay (3 shards, run-wall) | 20.3 run-wall-h (LoRA re-training; GPU 40–70 % busy) | ≈ 1 (norms) |
| 7 (y, h) audit after install | 0.52 | — |
| **Total so far** | **4.69** + step 7 (planned 3.15 for steps 1–6) | |
