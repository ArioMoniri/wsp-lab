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
