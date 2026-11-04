# P4 Board A revision

Merged main as requested, preserving both sides of append-only board/session conflicts.
Implementation stays on feat/p4-predictors; no integration into main or PR created.

1. Manifest-derived relations distinguish successors from sources. lora_vs_base is
   a flagged successor because the cumulative stored state includes its predecessor;
   it does not imply optimizing LoRA against that predecessor. Task-vector sources
   are single-step base edits with parent stats as side information, never history.
2. Family/seed folds purge full ancestry, including unlabeled chain inner_family
   steps and source dependencies, before normalization or JEPA pretraining. Audit
   captured the real manifest at 1219 rows (the zoo is still growing):

   | held out | candidate train | dropped ancestry rows | kept | violations |
   |---|---:|---:|---:|---:|
   | lora_sft | 1098 | 459 | 639 | 0 |
   | abliteration | 1014 | 443 | 571 | 0 |
   | random_dir | 1066 | 468 | 598 | 0 |

   These are structural manifest audits, not training runs: real eval targets remain
   absent. Counts include the current step if its inner_family is held out.
3. Default target is delta: measured rate minus immediate successor predecessor
   rate, or measured base rate for single-step edits. Missing predecessor targets
   cause exclusion with counts. --target level is explicit; all heads are linear.
4. M5 has two total attention layers and dropout 0.2; m1_matched widens one stateless
   feedforward layer, differing from M5 by four parameters. M6 reports frozen-standard-
   predictor ERank, collapse gates and H3 acceptance. Descriptor excludes length and
   inner_seed. MODEL_CANDIDATES/HYPOTHESIS explain cumulative-feature qualifications.
5. --selftest produces synthetic toy trajectories and target rates under /tmp,
   synthetic=true in manifests; no results paths. CUDA was not attempted because
   agent-tmux -l still lists p1-zoo-full. CPU selftests passed:
   - M2 loop: /tmp/wsp-p4-synthetic-gw0lydz8/run
   - M5 attention: /tmp/wsp-p4-synthetic-yv_fw8dc/run

Validation: 319 passed, 8 deselected in 21.10 seconds; 21 P4 tests (8 added).
Ruff lint and format clean. Real-data CLI refusal: zero entries have targets with
required predecessor targets; missing_predecessor_targets=0; no output dir created.
No real H1/H2/H3 scores or GPU/Mamba validation claimed.

## Board A re-review implementation (2026-11-04)

Merged main into the feature branch, preserving both board-log entries. B1: explicit named-family holdout uses all labeled rows of that family as its sole test fold, with ancestry purge before inner validation. H3 is evaluated only for explicit lora_sft holdout; neither pooled k-fold results nor other named families report an H3 gate. Regression tests force all six benchmark correlations above threshold to check this distinction.

N1: purge/chain-free variants are recorded in config and manifest; chain-free excludes chain steps and descendants (including task-vector children). N2: H2 metrics and report rows split explicit sequential/lora_vs_base semantics; legacy unflagged chain rows stay unknown. N3: corrected M5 linear-head spec; recursively flatten inner_params numeric fields for structural hashing; scope outer/inner split audits; read manifest bytes once before splitting, persist the snapshot and record SHA256 plus nonblank JSONL row count. Added a synthetic selftest banner to results.md.

CPU selftests m2 (loop), m5 (torch_attention), m6 (torch_jepa) all passed, three folds each. No real hypothesis scores or GPU claims.

Final validation: CPU pytest 325 passed, 8 deselected in 25.26 s (< 60 s local and < 90 s task limits); Ruff clean after import ordering fix; real-data named-holdout run refuses zero target entries before creating output. Toy-zoo tests cover k-fold, named LoRA/other-family holdouts, both sensitivity variants, six-benchmark gate isolation, structural descriptor ordering, snapshot digest/count and audit scopes.
- 09:05 Merged feat/p4-predictors (9f1d3da, 325 tests). All P1–P4 code on main; critical path = evaluation targets (post-zoo D2 timing → eval throughput work → D3 subset batch). Lambda ARA 12×10 running.
- 07:16 Zoo build finished (exit 0): 1,235 unique entries; production random_dir 150 / abliteration 200 / lora_sft 120 / task_vector 80 / chain 655. post-zoo chain proceeding (HF sync → extraction top-up → D2 timing). NOTE: earlier hand-typed times in this note were ~1.5 h ahead of UTC; from here on timestamps come from date -u.
- 07:31 Board C H6 merged (7d70888): LOMO screening on 3 models with n=100 labelled edits each; two competing representation runs. Multi-model zoo configs in progress. Post-zoo chain at HF sync; Lambda ARA entry 1.
- 07:48 Zoo is 250 GB (chains 337 MB/entry); first HF upload failed → D7: metadata to HF (done), non-chain deltas uploading (tmux hf-deltas), chain deltas stay local. Extraction top-up running (700/1,235), then D2 timing. Lambda ARA entry 1 (12×10).
- 07:58 Multi-model zoo branch delivered (f1325ba; LFM3-1.7B/Qwen3.5-4B/Gemma-270M configs; feature_version 4 → 0.8B features must be re-extracted after merge; fold_post_norm variant; Gemma via unsloth mirror because google repo access is not granted). Board A review running.
- 08:03 multi-model-zoo: Board A REVISE (trust_remote_code override) fixed by orchestrator (0a1f6f0, 395 tests); merge deferred until the 0.8B extraction top-up ends (feature_version 4). HF non-chain delta upload resumed (hf-deltas2). perf-benchmark tmux gated behind post-zoo.
- 08:05 Merged feat/multi-model-zoo (e35de29). v4 feature extraction restarted from scratch (tmux p3-extract-v4, 4 workers; v3 dir kept as qwen3.5-0.8b_v3). D2 timing running under post-zoo on the GPU; perf-benchmark gated behind it. HF deltas: 128 commits/h limit hit → batched single-commit uploader scheduled (hf-deltas3, after 1 h).
