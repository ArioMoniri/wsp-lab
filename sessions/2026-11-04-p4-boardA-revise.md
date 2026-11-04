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
- 08:19 Clean D2 timing: 745 s (12.4 min) per entry on the idle GPU; first edited-entry eval (abliteration 4463696a4c62e8f3): GSM8K 14, MATH 8, IFEval 44, MMLU-Pro 28, HellaSwag 39, refusal 100, KL 0.001, loop 100. D3 subset ≈ 62 h locally → waiting on the throughput benchmark (running now); Lambda A100 fallback after ARA. v4 extraction at ~125/1,235.
- 08:50 Throughput benchmark: HF b32 6.18 / HF best 6.15 / vLLM 6.10 min per entry (no engine gain; decode-bound). Launched D3 subset eval locally (tmux p2-d3-batch, ≈300 entries, ≈31 h). perf branch under Board A.
- 09:03 Merged fix/eval-skip-key (4b501f3; resume key no longer includes the code hash; legacy D3 files migrate). perf/eval-throughput left unmerged (no engine gain; docs corrected on main). D3 batch: 2 entries done. Queue after D3: HF batch-1 greedy control, 32k base rerun, baselines on first ~100 targets.
- 09:48 Lambda ARA: first two production entries landed (10 trials each): bbf6af96 refusals 0.20 / KL 0.028; 770d16bb refusals 0.96 / KL 0.0. ~80 min/entry → 12 entries ≈ 16 h ≈ $32. Anomaly filed: delta_fro_norm 1.9e9 on the strong entry. HF private storage limit reached → deltas stay local (D7 addendum). D3 evals 3 done; v4 extraction 434/1,235.
- 09:53 Lambda ARA stopped and instance terminated (3.66 h, $7.29): Board A found the exported adapters do not match the evaluated model (quarantined 2 entries). Fix delegated; family rebuild later. HF private storage limit → deltas local only.
- 09:56 fix/heretic-ara-export: tests-first dense export from merged bf16 weights, selected parameters applied once after study, verified/refreshed scores and saved-artifact KL gate; targeted CPU tests 11 passed. Full CPU and GPU smoke pending.
- 09:59 fix/heretic-ara-export validation: full CPU pytest 401 passed / 8 deselected in 30.29s (commit hook 28.45s); ruff check src tests and ruff format --check src tests clean. agent-tmux ara-fix-smoke log /home/ubuntu/agent-logs/ara-fix-smoke-20261104-095732.log: GPU/slow Qwen3.5-0.8B trials=2 smoke passed in 40.70s, batch=1, rank=4, n_optimization_steps=1/max_iter=2, 4 train prompts per class and 2 evaluation prompts per scorer, configs/p1_generator_smoke.yaml. Artifact data/runs/p1-generator-smoke/heretic_ara; verified KL=0.001249964232556522, reloaded KL=0.001249964232556522 (difference 0), verified/reloaded refusals=1.0 (2/2). This is plumbing validation, not production ARA quality evidence; upstream row-normalization semantics remain unchanged. No weights committed; no merge.
- 10:11 Merged fix/heretic-ara-export (evaluated dense export + reload KL gate); fixed a self-inflicted 3-test regression (format label). Relaunching the ARA family on Lambda with the fixed exporter (D6 budget).
- 21:42 Board A H5 review: wrote prospective noise qualification and Experiment 8/H5 arm A protocol in docs/reviews/H5/board-a-prereg-2026-11-04.md; REVISE B1-B6, no GPU runs or merge. Reviewed current controls; independent retest provenance and 32k power remain open.
- 21:48 Implemented Board A B2/B3/B5 eval metadata, immutable repeats, keyword names, matched-reference helper; CPU/GPU validation in agent-tmux board-a-checks.
- 21:49 Regression tests red first (2 failed); targeted suite now 17 passed, additional provenance/hash tests 5 passed. First corrected CPU suite 406 passed; final CPU/GPU CI validation pending.
- 21:50 Final validation: CPU 407 passed; CI GPU subset 4 passed; additional hash/reference tests 5 passed; ruff check and format clean. No benchmark/reliability panels or predictor training launched.

## Board A eval fixes re-review (2026-11-04)

Implemented the four blocking findings from board-a-eval-fixes-2026-11-04.md
(read from the canonical checkout because that historical review is absent here).
Tests first: seven new regressions failed before implementation. Added subsequent
end-to-end checks for successor matching, delta reliability and both resume paths.

Predictor labels now require known, equal device/runtime/config/source/model,
actual batch, generation/seed and item/order hashes; legacy unknowns and explicit
missing-base flags cannot produce base deltas. All base snapshots and device/run
base directories remain reference candidates, avoiding conflicting-base errors.
Metrics and training manifests report missing_matched_reference and matched counts
in edit-row units. Toy fixtures/selftest now supply explicit synthetic identity.

Reliability requires known aligned edit/item/order identity, retains negative ICC
and undefined constant ranks, separates base, adds difference summaries and
Bland–Altman limits, edit-cluster bootstrap rank/ICC intervals, matched-reference
level deltas, centered-effect tail diagnostics, available-item discordance and
paired intervals. KL remains continuous; WikiText uses log-ppl/window diagnostics;
boundary binomial uncertainty uses labelled Wilson half-width divided by 1.96.

Resume rejects legacy/foreign bases and preserves them, writing new evaluations
under base/devices/<opaque-device-id>/<run-id>/eval.json. --no-base prints the
missing-reference condition and leaves unmatched effects missing. Corrected only
the live RESEARCH_NOTES keyword-metric claim; historical reviews/ORCHESTRATOR untouched.

Validation: 417 CPU tests passed, 8 deselected, 24.30 seconds; Ruff lint and format
check passed. Exact requested real-data m1/family/CPU/one-epoch/n-boot-20 command
reads all artifacts without conflicting-base errors, then exits 1 explicitly:
matched_reference_targets=0, missing_matched_reference=108 (observed edit rows).
No training output or hypothesis result is claimed: real legacy artifacts lack
required matched-reference metadata. New measured matched runs are needed; metadata
was not inferred or backfilled. No GPU runs, repeat panels or qualification claims.

Metadata backfill implementation: scripts/backfill_eval_metadata.py is dry-run by
 default, labels full/partial records, reconstructs item hashes, updates index
metadata, preserves scores and links device bases and evaluated zoo directories.
Applied to shared real artefacts; rerun reports zero changed records. Private
mapping created outside git. Lambda runtime evidence remains incomplete; only
torch is known. Control copies lacking corresponding item records stay partial.
Before CPU suite: 417 passed / 8 deselected. M1 before: 0 matched / 109 missing.
After dataset: 53 matched / 225 missing; family training exits on empty fold after
ancestry purge. No runtime values were guessed to increase matching. Existing
script executable bit corrected for repository-wide ruff EXE001.
Final CPU suite: 420 passed / 8 deselected (24.42 s); repository-wide ruff clean.
