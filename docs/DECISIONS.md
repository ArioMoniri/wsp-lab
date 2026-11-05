# Decisions

## D1 (4 Nov 2026) — bf16 rounding of small deltas
Finding (reviews/P2/fix-eval-lora-boardA-2026-11-04.md F2): merging a LoRA delta into bf16 weights rounds away
~87 % of non-zero delta entries for a 5-step LoRA; rounding alone moves logits by ~40 % of the intended effect.
Decision: the subject models are evaluated in bf16 as deployed. The zoo therefore stores the **effective** delta
Δ_eff = bf16(W + Δ) − W (what actually changed), together with `rounding_stats` (fraction of entries lost,
‖Δ‖ vs ‖Δ_eff‖). Predictor features use Δ_eff. The eval loader must apply Δ_eff exactly (upcast W to fp32, add,
cast to bf16) and record the stats in its manifest. Any fp32-merge variant is a separate, labelled experiment.

## D2 (4 Nov 2026) — eval tiers and token budgets
The base model loops in ~90 % of generations, so generative benchmarks cost max_new_tokens × n tokens per
entry regardless of quality. Tiers: **fast** (per-entry zoo screening): gsm8k 50 / math500 25 / ifeval 25 /
mmlu_pro 100 / hellaswag 100 / refusal 50 / kl 50, max_new_tokens **1024**, batch 32 — loop rate is still
measured as hit-1024; **cheap** (base and selected entries): the P2 battery at 4096; **faithful** (headline
comparisons with the blog): 32,768 tokens, few entries. Throughput work (vLLM or HF static-cache batching) is a
perf-challenger task; the predictor targets are the fast-tier scores, with cheap-tier scores used to
calibrate on a subset.

### D2 addendum — labelled metadata backfill (4 Nov 2026)

Backfilled metadata is labelled; new runs record it natively. Run
`python scripts/backfill_eval_metadata.py` to inspect changes and add `--apply`
to write them. The private device map supplies opaque IDs; historical host labels
are removed from updated payloads. Original base records are retained, with
copies under `base/devices/<device>/<run>/`. Evaluated zoo directories are linked
into the eval directory so directory readers see all completed entries.

Item identity and order are reconstructed only from the corresponding item records;
control copies without their own records (or a matching main run ID) retain null
hashes. Local runtime fields describe the current environment and the runner's
bf16/SDPA settings, not a contemporaneous measurement. Remote runtime evidence is recorded in
`configs/eval_runtime_backfill.json`, with per-field transcribed/inferred provenance.
H1 SDPA is inferred from the identical model load path. H2 engine, transformers
and SDPA are inferred from the identical bootstrap/lock; other supplied fields are
transcribed from the orchestrator's device evidence. This is not a new measurement.
`metadata_backfill=full` requires every reference key to be known and transcribed;
`partial-inferred` lists inferred keys, and incomplete records list missing keys.
Strict matching compares every reference key, including inferred runtime values.
Predictor manifests disclose `targets_with_inferred_runtime` (edit targets, not
individual benchmark cells). Local backfills retain the existing current-runtime
convention described above; they are not contemporaneous historical measurements.

Run `python -m wsp.predictor.data --report` for coverage by family/device/reason.
The supplied remote evidence increases coverage from 53 matched / 226 missing to
221 matched / 58 missing, with 167 targets depending on inferred remote runtime.
The M1 compatibility run completes five family folds. Empty ancestry-purged folds
are logged and skipped; if no validation plans survive, training reports an error.
Scores and actual batch sizes are preserved. These inferred matches do not qualify
measurement repeatability panels as independently verified runtime equivalence.

## D3 (4 Nov 2026) — evaluation budget and order
Board A's calibrated estimate (reviews/P2/feat-p2-batch-boardA-2026-11-04.md): ~9–12 min per entry under D2 on
this GPU, ≈ 120 GPU-hours for 700 entries. Order of work: (1) measure one clean D2 entry when the GPU is free;
(2) evaluate a stratified subset first — random_dir 60, abliteration 100, lora_sft 60, task_vector 30, chain
steps 50 (≈ 300 entries, ≈ 50 h) — enough for H1/H2 baselines; (3) pursue vLLM (if Qwen3.5 is supported) or a
Lambda H100 to finish the remainder; (4) refusal/KL prompts stay at 50/50 because they are cheap (short
continuations). The 32k faithful tier is reserved for the base model and ≤ 10 selected entries.

## D4 (4 Nov 2026) — Lambda GPU spend approved
User approved Lambda Cloud usage with a cap of **$150**: (1) Heretic ARA zoo family on one GPU (launched 4 Nov 2026 as a 1×A100-SXM4 40 GB instance at $1.99/h, the only type with capacity), (2) faster
evaluation of the D3 subset if local timing is slow. Rules: instances named `wsp-lab-*`, launched with a
per-hour price cap, results pushed to the private HF datasets, instance terminated as soon as its job ends;
spend ledger in `board/lambda_spend.md`.

## D5 (4 Nov 2026) — trajectory relations and per-step targets
Board A (reviews/P4/feat-p4-predictors-boardA-2026-11-04.md) showed that `parent_id` means different things per
family: for chain steps the parent is the predecessor state (`successor` edge); for task vectors the parent is
the *source* of a scaled delta applied to the base (`source` edge), not a predecessor. Predictors that model
history (M2, M5) and JEPA transitions (M3) use `successor` edges only. H1's target is the per-step change
Δy_t = y_t − y_{t−1} (predecessor = base for single-step edits); absolute level is an explicit secondary target.
A family holdout drops every entry whose ancestry contains the held-out family.

## D6 (4 Nov 2026) — Heretic ARA budget
Profiling on the Lambda A100 (py-spy): each ARA trial runs an LBFGS fit per layer and module with strong-Wolfe
line search, so a trial is launch-bound (~10 min, GPU mostly idle) regardless of GPU class; 30 trials × 30 entries
would be ≈ 150 GPU-hours, over the D4 cap. Decision: heretic_ara n = 12 entries at 10 trials (≈ 20 h, ≈ $40);
production eligibility threshold for this family lowered to trials ≥ 10 (each trial is a full optimisation, unlike
the cheap classic-abliteration trials). Recorded in configs/zoo_qwen3.5-0.8b.yaml.

## D7 (4 Nov 2026) — what goes to the Hugging Face store
The Qwen3.5-0.8B zoo is 250 GB on disk: chain steps store dense cumulative deltas (337 MB/entry, 227 GB for
675 steps); random-direction 38 MB, abliteration 85 MB, task-vector 191 MB, LoRA 17 MB per entry. The first
full upload failed. Decision: HF receives all metadata (manifest, edit.json, provenance, eval.json, features,
runs) plus the deltas of the non-chain families (≈ 40 GB); chain deltas stay on this server (and are
reproducible from their recorded step parameters and seeds). `scripts/hf_sync.py` implements the selection.

## D3 addendum (4 Nov 2026) — measured timing
Clean measurement: 12.4 min per entry (fast tier, HF path, batch 32). D3 subset ≈ 62 h locally. Decision pending the
throughput benchmark: if vLLM gives ≥ 3×, run D3 locally; otherwise split D3 across the local GPU and the Lambda A100
after the ARA family finishes (A100 ≈ 2× this GPU at ~$2/h, i.e. ≈ 30 h ≈ $60, within the D4 cap).

## D3 addendum 2 (4 Nov 2026) — engine decision
Throughput benchmark on the same entry (perf/eval-throughput 73f1419): HF batch 32 = 6.18 min and vLLM = 6.10 min per
fast-tier entry, single runs while four CPU extraction workers were busy (Board A: the "hf-best" 6.15 figure was the same
config as hf-b32, i.e. timing noise; static KV cache changed scores (0/32 greedy agreement) and torch.compile crashed, so
neither is used). vLLM also changed scores (gsm8k 0.14→0.06, ifeval 0.44→0.36 at the stochastic settings), so it is not a
drop-in replacement without a batch-1 HF control. No engine advantage; the floor is ~6.1 min (decode-step bound).
Decision: run the D3 subset (300 entries, ≈ 31 h) locally with the HF path now; use the Lambda A100 for the second
half once the ARA family finishes if the local run is still behind.

## D7 addendum (4 Nov 2026) — private storage limit
The batched delta upload stopped with "Private storage limit reached for user Aralario". HF keeps the metadata,
features, runs and evals (small); zoo deltas stay on this server (nightly DR backup) unless the account is
upgraded. Chain deltas were never going to HF (D7). Public release of deltas is not planned (abliteration deltas).

## D6 addendum (4 Nov 2026) — ARA adapters invalid as exported
Board A forensic check (reviews/P1/heretic-ara-densify-check-2026-11-04.md): the saved PEFT adapters do not equal the
model Heretic evaluated (row normalisation lives in the objective, not the export; the generator re-runs the selected
trial before saving), and the saved factors are enormous (delta norm 1.9e9 for a KL-0.028 trial). The two Lambda
entries are quarantined under data/runs/lambda-ara-quarantine/ and the instance was terminated ($7.29 total). Fix:
export the exact evaluated model (merged delta computed in-process from Heretic's merged model) and verify by a
forward-pass KL on the saved artefact before the family is rebuilt.

## D3 addendum 3 (4 Nov 2026) — parallel evaluation shards
User asked for faster turnaround. The D3 subset (282 entries remaining at 10:45 UTC) is split by host: 60 entries on
this server (all 50 chain entries, whose deltas are local only, plus 10 controls) and 121 on each of two Lambda
A100-SXM4-40GB instances (`scripts/lambda_job_eval.sh`, family-balanced round-robin split, self-terminating; a third
instance had no capacity). Cross-host control: the same 10 non-chain entries are evaluated on every host, and every
Lambda host also evaluates the base model (stored as evals/qwen3.5-0.8b/base-<host>). The paper reports the
host-to-host score spread from these controls before pooling; pulled eval.json files carry `eval_host`
(`scripts/d3_pull_shards.py`). If the spread is not small relative to edit effects, Δy is computed against the
same-host base instead of pooling.

## D2 addendum (Board A B2/B3/B5)

Stopping budgets come from the YAML config; caller token-cap overrides are rejected.
`--repeat LABEL` forces independent execution in immutable `repeats/LABEL` directories;
reuse a new label after a failed run. Primary resume identity is unchanged, including the
reporting-only keyword configuration rename. Repeat indexes are separate from primary indexes.

Eval.json field list (all additions optional when reading legacy results):
- Existing benchmark fields retain their score semantics and counts.
- `loop_rate` remains pooled GSM8K/MATH/IFEval truncation, fast denominator 50/25/25.
  EOS-at-cap is not a hit. `loop_rate_math500` uses only MATH (the blog denominator);
  `loop_rate_gsm8k` and `loop_rate_ifeval` expose the other components.
- `invalid_final_answer_rate_<benchmark>` measures empty extracted final answers; no
  new mathematical validity checker is claimed. Accuracy remains `<benchmark>`.
- `refusal_keyword_rate` and `over_refusal_keyword_rate` explicitly name keyword metrics;
  `refusal_rate` / `over_refusal` are deprecated aliases for one release. Judge prompt
  export is separate (`wsp.eval.refusal --export-judge`), off by default, and executes no judge.
- `batch_size` is actual after OOM retry; `batch_size_nominal` records the requested size.
  `item_order_hash` hashes benchmark/item pairs in execution order, `item_id_hash` their sorted list.
- `repeat_id`, `eval_device_id`, `gpu_class`, `engine`, `torch_version`, `cuda_version`,
  `transformers_version`, `dtype`, `attention_implementation` describe execution conditions.

`wsp.eval.repeats.compare_pairs(primary, repeat, bootstrap=10000, seed=0)` accepts aligned
payloads or eval.json paths, returning per-benchmark MAD, signed paired bootstrap CI,
Spearman, ICC(2,1), binomial SE and mixed-actual-batch flags. Constant panels have undefined
rank/ICC; negative ICC is retained. Analyse base separately; mixed-batch pairs cannot qualify
repeatability. Match reference device/runtime/config/batch/items/order/seed before deriving
single-step labels; missing matched references remain missing, never silently substituted.

`wsp.eval.repeats.matched_reference(observation, candidates)` returns an exact metadata
match or `None`, including for legacy unknowns. Batch results expose `base_run_id`,
`matched_base_missing` and `entry_hashes`; an OOM batch mismatch leaves the base missing.
Successor callers must supply predecessor candidates rather than base candidates.


## D2 addendum (4 Nov 2026) — Board A runtime and reference corrections

The orchestrator reports that the 20:30 UTC `--force` retest produced bit-identical
scores on base plus ten controls across the old and new source trees (11 records).
This is supporting evidence, **not** a reference matching rule. Source hashes remain
unchanged and exact matching is required. The original base run
`20261104-084939-f912077b` is registered under `base/devices/H0/<run>/`; control
copies are registered under `<edit>/devices/<device>/<run>/` without replacing primary scores.

The original base lacked item hashes and a preserved item sidecar. Its clean git
revision `846ff8423b9c67a9d9a9bfa3c1a46ec8f3feb2e5` reproduces source hash
`d7cece35fad962816421132974855a3d8b127f8aa596db485f334750a8a837b7` exactly.
`reconstruct_historical_items.py` recovers the ordered 458 identities from that
archived tree's subset files, positional-ID evaluator paths, effective config and
complete counts. These hashes are explicitly inferred; no retest records or
scores are used. Evidence is stored in `data/evals/qwen3.5-0.8b/historical-item-identities.json`.
This resolves the 18 random-direction source mismatches. Contrary to the initial
19-entry diagnosis, abliteration `4463696a4c62e8f3`, run `20261104-080338-4d5632a1`,
uses a third source tree: `49c83214bd08cb76783cd9539ea88868c5bd074c`, independently
hashed to `560abb41c7b1f2ce3b690532666d36c8389a28e85577f78274bee7437bdc595a`.
No matching base exists for that tree; retain it as missing.

Local fields reconstructed from today's runtime are inferred with source
"current environment, not contemporaneous". H1 transformers is inferred too;
there is no new contemporaneous evidence establishing its installed version.
`partial-inferred` records list inferred keys and separately list missing keys.
The post-correction feature dataset has 240 matched, 40 missing and 240
`targets_with_inferred_runtime`. All matched targets currently depend on inference;
excluding inferred pairs leaves no qualified target panel in this snapshot.

| Family | Matched | Missing |
| --- | ---: | ---: |
| abliteration | 84 | 2 |
| chain | 13 | 37 |
| lora_sft | 59 | 1 |
| random_dir | 54 | 0 |
| task_vector | 30 | 0 |

Missing: 37 unevaluated predecessors, two H2 actual-batch mismatches and the
third-tree abliteration. Counts describe this live snapshot, not frozen cohort sizes.

Reliability pairs now reject incompatible known runtime/config/source/seed and
actual batch (the explicit `--allow-batch-mismatch` diagnostic override records the
mismatch). Host IDs may differ; other compatibility fields remain strict. Item
uncertainty resamples edit clusters then paired items; WikiText uses contiguous
nonoverlapping length-two window blocks on log-ppl, with exponentiated ratio
intervals for display. Matched-reference delta uncertainty resamples shared bases
once per replicate, preserving their covariance across edits. The preregistered
host-by-edit qualification analysis remains blocked, and no panel qualification
is claimed.

Before this revision's backfill each existing record was archived as `.orig` if
absent. `verify_backfill_scores.py` checks every non-metadata key (including counts
and nested scores), missing archives and missing current records. It found zero
score differences across 337 unique archives. These archives establish preservation
for this revision, not retroactively for backfills performed before archives existed.
Empty ancestry folds emit WARNING on stderr through Python logging and appear in
`skipped_folds` in metrics and manifest; the CPU test captures that warning and
checks the persisted metrics.

## P4 addendum — 2026-11-05: selection, refit and audit protocol

These old → new rules deliberately change numerical results; round-two metrics
are not numerically interchangeable with the earlier protocol.

- Selection/refit: first surviving inner split and its trained model → all
  surviving inner folds select the median best epoch (rounded up), then a fresh
  model fits all outer-training rows for that fixed epoch budget. Outer-test
  labels never select epochs. Predictions and losses therefore change.
- Preprocessing: one inner-train normalizer → separately fitted inner-training
  normalizers and a fresh outer-training refit normalizer. Signed log1p precedes
  scaling; train-constant coordinates are zeroed, including at test time.
  Optional robust scaling uses training medians/IQR rather than mean/std.
  Different fitted contexts and constant handling change transformed inputs.
- Baselines: first inner-training subset → the same outer-training refit rows
  as the predictor. StatNN retains the original identical seed in every fold,
  not seed + fold; LightGBM uses deterministic row-wise training. Fewer than
  two finite training labels → explicitly skip that target/fold, log the reason
  and record skipped_reason; never replace StatNN with a training mean.
  This preserves the legacy fit_eval skip rule and removes the round-two
  silent fallback. Audit controls also skip that unsupported target/fold.
  Legacy StatNN model, n, n_folds and n_features remain model name, total input
  rows, supplied fold count and input stats width, respectively.
- Fold reporting: legacy train_ids, normalization_context_ids and history now
  describe the outer refit; validation_ids is empty and refit validation_loss
  is null. Validation IDs, losses and epoch selection live in inner_selection.
  Thus membership, ordering and training-loss fields change interpretation.
- Splits: family/seed label exclusion alone → full-ancestry cluster exclusion
  plus ancestor/source metadata purging in affected holdouts. This can remove
  training rows, including clusters joined through unlabeled ancestors.
  Trajectory splits already group full ancestry; the compatibility fixture
  uses trajectory splits and does not demonstrate row loss in these holdouts.
- Intervals: marginal metrics retain seeded edit percentile bootstrap;
  paired predictor-minus-control differences use full-ancestry cluster
  bootstrap with paired predictions retained, conditional on fitted models.
  Each interval identifies its own method. Changed predictions change marginal
  endpoints; the paired estimand has a different sampling unit.
- Centring: family centring after pooling folds → family × outer-test-fold
  centring and fold-n weighted finite fold correlations. Pooled centring is
  separately labelled family_centered_pooled. Removing fold offsets changes
  the within-fold diagnostic and can leave correlations undefined.

## D6 addendum 2 (5 Nov 2026) — ARA rebuild landed, selection suspect
The 12 rebuilt heretic_ara entries (Lambda A100, 18.8 h, $37.41; sync zoo 690134e) pass the reload-KL gate
(kl_reload_difference 0.0 on all 12: the saved Δ is the evaluated model), but the selected trials have verified KL
to base of 0.77–14.0 nats with refusals 0.0, and delta Frobenius norms from 19 to 1.7e10. Lower-KL trials existed
and were not selected. Pending Board A's forensic (docs/reviews/P1/heretic-ara-rebuild-forensic-2026-11-05.md), these
entries stay in the zoo but are NOT used in any Heretic-quality comparison; eligibility disposition follows the review.
The instance's `hf_sync.sh push zoo` also overwrote the hub manifest with its 458-row copy; the canonical local
manifest (1,247 rows) is re-synced.

## D6 addendum 3 (5 Nov 2026) — diagnostic exclusion and corrected ARA protocol

Board A's REVISE disposition in
`docs/reviews/P1/heretic-ara-rebuild-forensic-2026-11-05.md` is adopted.
`scripts/flag_ara_unconstrained.py` was run on all 12 IDs in `/tmp/ara_new_ids.json`:
`edit.json` provenance and canonical manifest rows now carry
`diagnostic_flag: ara_unconstrained_surrogate_scale`, with `eligibility: smoke`
(the schema's excluded, non-production class) and an explicit diagnostic reason.
The script is idempotent. Identities, deltas, generator descriptors, studies and
measured scores are retained; these entries are excluded from default production
panels and Heretic-quality comparisons. Diagnostic inclusion requires explicit
curation and inclusion/exclusion sensitivity analysis. No old delta was rescaled.

The replacement remains **12 entries × 10 trials**, a ten-trial startup search
(TPE startup remains 30). The new planned exploratory ceiling is explicitly
`kl_divergence_target: 0.5` nats of baseline-to-edited first-token KL; it is not a
capability guarantee or a previously preregistered criterion. Missing, nonfinite
or nonpositive targets are rejected. Reject nonfinite metrics; among KL-feasible
trials form the nondominated (refusals, KL) frontier and select lexicographically
by (refusals, KL, trial number). An empty frontier fails the entry, without
relaxation. Preserve every scored candidate, the complete frontier, target,
selection rule, effective settings and upstream version in `study.json`.
Selected parameters are reapplied once; merged and reloaded KL must each meet
the ceiling, reload KL difference must be ≤0.01, and reloaded tensors must exactly
equal the scored merged weights. Failures are recorded and no edit is published.

For this round use the review's explicit alternative: **disable row normalization
in both optimization and deployment**, labelled
`wsp_ara_no_row_normalization_v1`. A repo-controlled wrapper forces
`preserve_row_magnitudes=False` and rejects attempts to enable it. This is a
labelled unnormalized ARA variant, not a row-preserving Heretic treatment.
Heretic is pinned to 2.0.0; effective settings and per-module LBFGS outer-step
last-closure losses, factor/product/effective-weight norms and gradient norms
are recorded. These are diagnostic traces, not proof of solver convergence.
The kernel version and explicit target enter artifact identity; builder params
and provenance now agree on dense format, so corrected artifacts receive new IDs.
The Lambda script keeps its CLI and inherits the explicit target from config;
its default budget now matches D6.

**Corrected rebuild pending.** No GPU work, rental or new spend was authorized
or launched here. Fresh search, independent held-out capability evaluation and
both-board review remain necessary before Heretic-quality or paper claims.

## D2 addendum — TF-v1 teacher-forced tier (5 Nov 2026)

`configs/eval_tf_v1.yaml` adds a versioned, deterministic tier through the existing
`wsp.eval.batch`/`HFRunner` execution path. Run it with `python -m wsp.eval.batch
--zoo data/zoo/qwen3.5-0.8b --config configs/eval_tf_v1.yaml --ids ...`.
Primary results live under `data/evals/<zoo>/tf_v1/<entry>/`; fast-tier labels and
indexes retain their original locations. `--repeat` remains an explicitly separate,
immutable execution, and is not used for the timing pilot.

The fixed seed is 20261105. Committed MMLU-Pro/HellaSwag positional ID lists exclude
all legacy subset IDs. MMLU gold-letter log probabilities are normalized over the
available answer letters, with raw likelihoods retained; HellaSwag uses character
length normalization, matching the fast scorer. WikiText reports token-weighted
log-PPL. KL is base-to-edited over the full vocabulary on greedy base continuations
of up to 128 tokens. Base logits are hashed and cached per measured device, runtime,
config and source identity under `base/tf_cache/`. Hash failures abort; another
device's base is never substituted. The two prompt datasets and WikiText are pinned.

The requested timing configuration uses 50 harmful and 50 benign refusal prompts.
First-token sets are frozen lexical prefixes in the config; token IDs and masses'
log-probability difference are saved per prompt. These are exploratory margins,
not keyword refusal rates or a calibrated safety measure. No TF-v1 endpoint is
qualified by this implementation or timing pilot. Board A's prospective paired
repeat, device, uncertainty and reliability requirements still apply. Fresh
source-closed components, predecessor-relative references and a frozen pipeline
are additionally required for the approved H1a estimand. Existing D3 results and
the legacy bridge are descriptive; no confirmatory readability claim follows.

TF-v1 timing-pilot note (5 Nov 2026): the base-logit cache key now uses the source content
hash only (not git sha/dirty flag), because a mid-pilot commit otherwise orphaned and rebuilt an
identical cache. Pilot results: see "TF-v1 timing pilot" in `docs/RESULTS.md`; its warm timing
fails the council gate and the legacy bridge is between different item sets (descriptive).

### TF-v1 panel status and cache (5 Nov 2026, Board A round 2 fixes)

The TF-v1 MMLU-Pro (1,000), HellaSwag (500 ranking, 250 used) and KL/over-refusal prompt panels in `configs/eval_subsets/*_tf_v1.txt` are **development/pilot** panels (hash-ranked by sha256(item_id + salt), salt recorded in each file and in `configs/eval_tf_v1.yaml`). They were scored on D3 controls and are barred from the sealed confirmatory panel (PREREG_H1_TF §2), which is built later under the distinct `TF-v1:item:` domain. KL and over-refusal prompts are disjoint. Refusal margins score continuation phrases and remain non-reportable as a refusal measure until calibrated on a disjoint set. The base-logit cache is keyed on the scorer's identity (not the full source tree) and every KL record carries its content hash.

### D2 addendum — H0 is the sole scoring device for the H1a panel (5 Nov 2026)

**Decision.** Every TF-v1 number used for H1a is scored on one device, **H0** (NVIDIA RTX PRO 4500 Blackwell, this server; batch 1, bf16, sdpa, the pinned torch/CUDA/transformers recorded in each `eval.json`). That covers the D3 training rows, the qualification repeats, the fresh-panel bases and edits, and every matched predecessor reference. **G3 (second-device transport) is not possible**: there is no second device and no new spend is authorised (PREREG_H1_TF §9: new Lambda = no). This is the "missing second device is single-device scope, not a transport pass" case of PREREG §3 and §6, declared before any fresh scoring and before the sealed partition is used.

**Consequences for claims.**
- The registered H1a claim (θ, its component and joint CIs, the kill rule) is a claim about the **H0 scoring pipeline only**. It says nothing about the same predictor's agreement with labels scored on another GPU, another batch size or engine (vLLM changed scores in the D3 engine benchmark), another dtype, or a different torch/CUDA/transformers stack. A pass is "readable on H0 under qualified single-device measurement", not "device-independent".
- The cross-device criteria of PREREG §6 (paired centred-difference SD ≤ .2× / .1× between-entry SD) are **not evaluated and not passed**. The device-variance term in the reliability calculation is zero by construction, not measured; only same-device repeat error (G1) is bounded.
- Training, qualification and confirmation use H0 rows only; unseen device levels fail closed (PREREG §3). Historical fast-tier labels from H1/H2 (Lambda hosts, inferred runtime) never enter training or qualification (D2 quarantine).
- A change of device, driver or software stack mid-programme changes the runtime metadata, so matched references and the base-logit cache key no longer match and scoring must stop; it is not silently continued. `~/.config/wsp/device_id` stays fail-closed and the cache stays under `tf_cache/H0/`.
- A cross-device claim needs hardware, an independent qualification on it, and a new dated amendment; none is planned.
