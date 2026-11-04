# Board A: measurement qualification and H5 arm A pre-registration

Date: 2026-11-04. Reviewer: strict ML correctness board. **Verdict: REVISE (blocking list below).**
This is a prospective protocol, not a report of H5 success. All numerical design thresholds, margins,
counts and budgets below are **planned criteria**; measured values are explicitly identified. No model
runs were launched for this review. Scope includes H1–H6: unreliable labels undermine H1/H4 ranking,
H2 history effects, H3 transfer and H6 cross-base transfer as well as H5 optimisation.

## Blocking findings

| ID / source | Failure scenario | Required resolution before confirmatory work |
|---|---|---|
| B1 — `configs/eval_fast.yaml:2`, `:11`; `docs/DECISIONS.md:83` | Sampling and host effects masquerade as edit effects; family and host are partly confounded. | Complete independent base + ten-entry repeat panels on each device class; qualify each endpoint below; rebuild matched-reference labels. |
| B2 — `src/wsp/schema.py:111`; `src/wsp/eval/run.py:220` | Pooled hit-1024 is described as the blog's MATH-only hit-32768; forced EOS appears to fix looping. | Persist MATH-only truncation, answer validity and accuracy; freeze stopping settings outside sandbox; faithful final evaluation. |
| B3 — `src/wsp/eval/batch.py:48`, `:197`, `:319` | Resume key omits host/repeat identity; OOM changes actual batch while the nominal config hash stays fixed. A supposed retest may be skipped, reused or use a different RNG schedule. | Separate immutable repeat directories, force execution, record actual batch/order and runtime; reject mixed-batch pairs or rerun both sides at the same batch. |
| B4 — `docs/MODEL_CANDIDATES.md:558`, `:579` | An editor-only win is called H5, or an uncalibrated critic exploits noisy labels. | Board C shadow-ranking gate, zoo/H1 prerequisites, critic calibration and the matched efficiency comparison below. |
| B5 — `src/wsp/eval/refusal.py:121`; `configs/eval_fast.yaml:32` | YAML lists an LLM judge but runtime only exports judge prompts; keyword refusal is called safety or judged refusal. | Identify keyword targets explicitly; separately execute and validate any independent judge. |
| B6 — `docs/DECISIONS.md:25`; `docs/MODEL_CANDIDATES.md:561` | Full faithful scoring of every episode conflicts with the existing ≤10 selected-entry allowance; 1-point noninferiority is asserted from tiny samples. | Approve and freeze a costed evaluation allocation and power check before running. This review does not authorize extra GPU spend or relax D3. |

Non-blocking: WikiText is a useful continuous diagnostic but eight adjacent windows do not establish
corpus-wide precision. Correct the stale ≤2-minute cost assumptions in the H5 assessment using measured
D3 timings. Neither issue licenses changing endpoints after observing editor results.

## Task 1: measurement noise floor

### What the current artifacts establish

Snapshot inspected approximately 21:39 UTC. Base run `20261104-202947-1dcb69a2` has GSM8K .16,
MATH-500 .16, IFEval .40; cross-device base run `20261104-112739-7cbc5bbc` has .06, .28, .44.
Both have MMLU-Pro .28 and HellaSwag .39. Source: `data/evals/qwen3.5-0.8b/` base-result artifacts;
paths with device identifiers must be mapped to opaque IDs in released tables.

Four shared edited entries (`dd90d99e2ddc0e77`, `0a841d4082532e8b`, `3db97b535c996265`,
`af6d608f8651fb46`) have paired original-local and imported cross-device results. Original run
`20261104-104702-5fac075c`; imported runs `20261104-180717-7eb22e74`,
`20261104-180717-b0b1206d`, `20261104-180717-febfc204`. Their nominal config and source hashes match.
For each listed edit, pair `eval.local-run1.json` with `eval.lambda-*.json`, subtract
local from imported and aggregate absolute differences. Descriptive only, n=4 edits; no inferential CI claimed:

| Benchmark | Mean absolute cross-device difference | Maximum absolute difference |
|---|---:|---:|
| GSM8K | .025 | .06 |
| MATH-500 | .080 | .16 |
| IFEval | .050 | .12 |
| MMLU-Pro | .010 | .02 |
| HellaSwag | .0025 | .01 |
| Keyword refusal | .000 | .00 |
| Keyword over-refusal | .025 | .06 |
| KL(base \|\| edited), sequence mean | .001118 | .003650 |
| WikiText perplexity | .006405 | .011546 |
| Pooled hit-1024 | .0125 | .02 |

Nine original/current snapshot pairs with different run IDs (base plus eight edits) currently have
identical reported scalars. They are not a completed ten-edit independent retest panel: config/source
metadata are not uniformly identical and preserved per-item repeat records must establish fresh
execution rather than copying/reuse. Do not infer zero noise from these snapshots. No complete H100
panel was present in the inspected matching files. Pending measurements remain pending.

### Endpoint definitions and denominators

`run.py:273` scores final-text GSM8K/MATH answers by normalized exact match, not a mathematical
proof checker; IFEval is strict prompt-level all-instructions success. Nominal fast n are 50/25/25;
MMLU-Pro and HellaSwag each use 100 likelihood-scored questions (no sampled generation; HellaSwag
normalizes summed log likelihood by choice character length). WikiText uses eight 1024-token windows,
aggregated as token-weighted log perplexity, not an arithmetic mean of window perplexities.
Refusal and over-refusal each use 50 generated prompts, first-n dataset selection, keyword checks on
final text; empty final text counts as refusal. KL uses 50 harmless prompts and up to 128 base-generated
continuation tokens, teacher-forced identical contexts: mean vocabulary KL(base || edited), separately
first-token and full-continuation means. It is not a binary accuracy and has no binomial SE.

`schema.py:111` pools only GSM8K/MATH/IFEval cap hits: fast denominator 100, weighted 50/25/25.
EOS at the cap is not a hit; refusal, KL and likelihood records are excluded. Report pooled and each
component's denominator; the blog comparison uses MATH only. A cap hit is a truncation proxy, not a
validated detector of semantic repetition. `run.py:195` reseeds once per generate call; batch, prompt
ordering and length alter RNG consumption. A fixed seed at temperature .6 does not imply invariance
across GPU kernels, hardware, runtime or batch changes.

### Required reliability calculation (separately for every benchmark and tier)

1. Freeze ten production edit IDs spanning effect sizes and families plus base, before comparing
   reruns; evaluate twice independently on the original device with seed 0. Repeat the same panel on
   each other device, including two executions there. Preserve records, manifests and hashes under
   distinct run IDs. Add generation seeds 1 and 2 as a separate stochastic-variance panel; identical-seed
   repeatability cannot estimate deployment sampling variance. Never mix these two estimands.
2. For each pair of executions and each benchmark, let d_i = y_i(second) − y_i(first).
   Report n matched edits, base separately, mean signed difference, MAD, RMSE, median and maximum
   |d_i|, SD(d), and Bland–Altman bias ±1.96 SD(d). Report pairwise item disagreement and McNemar's
   discordant counts for binary metrics. Compute all of these on levels **and matched-reference Δy**.
   Do not treat the base as an extra exchangeable edited entry in rank statistics.
3. Report Spearman with ties and two-way absolute-agreement single-measure ICC(A,1) across edits,
   separately same-device and each cross-device pair; cross-device hosts are fixed to those measured.
   Use 10,000 edit-cluster bootstrap replicates for 95% CIs. Declare rank/ICC undefined for constant
   panels, never replace with 1. Negative ICC is retained. Ten edits yield wide intervals, not certification
   of universal transportability. Include a host×edit interaction analysis on centered Δy; base subtraction
   removes an offset, not an interaction. Keep replicated rows of an edit together in every split/bootstrap.
4. Binary item uncertainty: report Wilson 95% intervals and SE sqrt(p(1−p)/n) as a descriptive
   approximation; never zero uncertainty at p=0 or 1. Maximal SEs are .0707 at n=50, .1000 at n=25,
   .0500 at n=100. Observed base SEs: GSM8K .0518/.0336; MATH .0733/.0898; IFEval .0980/.0993.
   These are finite-item uncertainty, not measured hardware noise. For paired edit-minus-reference
   effects use SD of item differences / sqrt(n), retaining discordance/covariance; use paired item
   bootstrap CIs. Pooling independent SEs is only a conservative planning approximation.
5. KL: bootstrap prompts, report mean/median/quantiles and outliers, first-token and full-sequence
   results separately. WikiText: work in mean NLL/log-ppl, block-bootstrap windows, exponentiate CIs;
   eight contiguous windows give weak population inference. For loop rate, cluster multiple samples by
   problem and stratify benchmark; 100 mixed questions are not 100 independent MATH observations.
6. Define R_b as the larger upper 95% bootstrap bound on the empirical 95th percentile of |d_i|
   for same-device or cross-device centered effects. With ten edits use maximum |d_i| as an additional
   conservative bound; report that tail estimation is weak. Include seed-panel variation separately.
   Archive a per-benchmark qualification table with R_b, CI widths, ranks, counts and missingness.

### Exact fast-tier qualification rule

For binary targets pre-register a smallest scientifically relevant change δ_b = .10 (10 percentage
points). For log-ppl use δ_b = log(1.05); for KL use δ_b = .01 nats. These are planned screening
thresholds, not observed MDEs. A target qualifies for **effect screening** only if R_b ≤ δ_b/2 and
its paired effect CI half-width ≤ δ_b/2. Qualification for **ranking/predictor evidence** additionally
requires Spearman point ≥ .8 with lower 95% bound ≥ .6 and ICC lower bound ≥ .75 on a sufficiently
variable panel. Undefined/saturated rank panels fail ranking qualification; expand the control panel
before claiming rank reliability. Qualify within-device first; mixed-device use requires cross-device
criteria too. No universal fast-tier qualification follows from agreement of two base scores.

Current provisional targets: MMLU-Pro/HellaSwag for large-effect screening; refusal/over-refusal and
KL as narrow, conditional probes after panel qualification. Saturated refusal cannot rank small edits;
a five-point change is not resolved merely because keywords are cheap. WikiText is diagnostic only
until window expansion. GSM8K/MATH/IFEval and pooled hit-1024 are descriptive secondary screening
signals, excluded from confirmatory small-effect/ranking targets at present.

On failure, increase items until paired CI criterion passes (rough independent worst-case planning:
about 769 items per side for .05 difference half-width; use pairing to refine), and rerun the panel.
For generative truncation also run 4096 on a frozen calibration subset; headline loop/capability
comparisons require 32768. More tokens do not reduce finite-item SE automatically; more items do not
make hit-1024 into hit-32768. Unresolved host interactions require single-device inference, not averaging.

### Reference labels and host encoding

**Yes: same-device, same runtime/config/batch/item set/seed base for every single-step Δy.** Store
`eval_device_id` (opaque H0/H1/H2), GPU class, engine, library/CUDA/dtype/attention settings, actual batch,
item/order hashes, generation config, run/repeat/seed IDs, base-run ID and delta/checkpoint hashes.
Keep the raw infrastructure identifier mapping private. No silent fallback to a different-device base;
missing matched base means target missing and counted. For successor chains D5 still applies:
Δy_t = y_t − y_predecessor, both measured under the same conditions. Task-vector source is not predecessor.
Repeated base uncertainty is shared covariance, never fresh independent noise for each row.

Include opaque device as categorical nuisance metadata in the observation table; compare a device-only
baseline and an optional device-conditioned predictor against the weights-only predictor. Device must
not enter weight features or substitute for family. Group splits by edit/ancestry/seed as already required,
then report within-device and leave-device-out sensitivity. Family-balanced shards do not cure chain/device
confounding. Do not claim hardware-free weight prediction from a device-conditioned model.

### Paper wording constraints (mandatory)

Allowed wording: “At the 1,024-token screening tier, under the recorded sampling/runtime and fixed
subsets, we observed [effect, paired CI, n items, n edits/episodes]. Cross-device calibration is reported
separately; labels use matched-device references.” Say “MATH exact-match”, “keyword refusal”,
“KL on base-conditioned continuations” and “share hitting the token cap”, with the correct denominator.
Use “no effect resolved at this precision” when intervals span meaningful effects.

Forbidden from this tier alone: “reproduced Experiment 8”, “fixed reasoning loops”, “improved math
ability generally”, “safer/less harmful”, “deterministic across devices”, “no effect”, or “critic saves
30% evaluations at equal quality”. Do not compare 1k pooled truncation to the blog's 32k MATH rate,
rank families from partial confounded D3 means, claim H1/H3/H6 on disqualified labels, or count repeated
samples as independent benchmark items. State every tier change and report null/failed edits.

## Task 2: prospective Experiment 8 reproduction and H5 arm A test

### Source and distinct claims

[Experiment 8, checked 4 November](https://rikiyat.github.io/blog/weight-space-programming/) reports
Qwen3.5-0.8B, a CPU Python editing sandbox, no gradients, and 10/25/50 remote forward-call budgets.
Calls answer up to 512 problems or read up to 1024 texts. Evaluation uses 32768 tokens and sampled
answers (four per MATH problem, two per GSM8K/IFEval). At 50 calls MATH improves 7.9→48.1%, IFEval
21→62.3%, MATH cap hits 93→33.1%. Its Gemma-3-270M comparison finds no established gain at 8192
with four MATH samples. These are source reports, not our observations.

Three separate questions: (i) can an editor independently find a useful weight-only intervention;
(ii) does it replicate the reported magnitude/settings; (iii) does a frozen predictor critic save real
evaluations? A positive (i) does not establish (iii). This protocol covers A and A+C, not optimizer arm B.
No decisive A+C run before the existing ≥600 single edits/150 trajectories, H1, held-out `llm_edit`
calibration and Board C shadow-ranker gates. Editor-only harness pilots cannot count toward confirmation.

### Frozen experimental units, seeds, budgets and data boundary

Plan **30 independent paired goal–seed episodes per arm per budget**: ten frozen goal formulations
of “improve math by editing weights” × editor/search seeds 0,1,2; budget cells 10,25,50 run independently
from base, not one continuing search selectively reported three times. Thus 90 episodes per arm.
Goal formulations and editor prompts are sealed by hash before execution; no mention of the known
layer/direction solution. Fresh editor context, sandbox and base each time; randomized arm order within
blocks; no cross-episode archive. Pin editor provider/model revision, sampling, tool policy and token
ceiling (planned 100,000 output tokens per episode); pin a named cheap editor before any outcomes.
Unavailable seed support is logged and independent session IDs retained; do not substitute providers.
Generation evaluation seeds 101–104 for MATH, 101–102 for GSM8K/IFEval, common across paired arms/base.
Search/practice generation seeds derive from episode, call and candidate, independently of final seeds.

Visible data: **200 GSM8K training + 200 MATH training problems**, fixed by seed 0 and hashed, deduplicated
against test by normalized text and near-duplicate/manual audit. At most these 400 distinct visible
problems may be queried, including rewritten equivalents; no IFEval test prompts visible. Repeated
queries consume budget. This restricted practice pool is an explicit adaptation of the source setup.
At most 64 generated problem sequences per diagnostic call and 1024 read-only texts per diagnostic
call; total generated-sequence ceiling 640/1600/3200 for 10/25/50 calls, diagnostic max 32768 tokens.
A call generates a batch of complete answers, not one decoder token. Charge retries, failed and
base queries; disallow combining read and generation modes in a call. Log actual token-level model
invocations, sequences, tokens and GPU-seconds too. Larger source batch ceilings cannot be described
as compute-matched to this design without another registered arm.

Sealed final set: **all 500 MATH-500, 500 GSM8K test, 300 IFEval prompts** sampled seed 0 without
replacement for the latter two, disjoint from visible material. MATH gets four samples/problem;
GSM8K/IFEval two. Audit benchmark availability/counts and freeze item IDs/fingerprints/checkers before
running. Final scoring at 32768 on one qualified device class; thinking, temperature .6, top-p .95,
top-k 20, presence 1.5 and immutable stopping configuration. Also evaluate 4096 on the same frozen
items for calibration, and likelihood/refusal/KL guardrails on frozen sets. Base scored under identical
settings. Final tests run once per committed candidate (the prescribed multiple samples are part of
that single evaluation), outside the editor process, after all arms' candidates are committed.

The evaluator runs as a separate OS principal/container with no shared writable filesystem. Editor
mount contains only base weights/tokenizer, allowlisted visible data and tools, never the repo/data
root, test caches, results, critic training labels, source blog or downloaded edited weights. Deny
network, subprocess escapes, evaluator credentials and test oracle access. Editor only exports delta
and candidate hash. Final evaluator returns nothing until the full experiment is locked. Public test
memorization cannot be ruled out by isolation: audit editor traces and label that residual limitation;
new private isomorphic problems are a registered exploratory contamination check.

### Arms and fair accounting

All arms receive identical goal/visible data, episode block and budget. Selection uses a fixed visible
utility U = mean(GSM8K accuracy, MATH accuracy); ties prefer lower MATH cap-hit rate then earlier
candidate. No final-set feedback or best-sealed-candidate selection. A+C uses at most 20 candidate
proposals, frozen calibrated critic, and uncertainty fallback; A without critic has the same cap.
Each real visible validation battery scores a fixed 50+50 visible items, counted separately from
arbitrary diagnostic calls; its forwards consume the 10/25/50 ceiling. Compare both call totals and
sequence/token totals. Define one P2-equivalent evaluation by the frozen fast battery workload;
convert actual evaluated records/tokens to equivalents and also report unconverted calls/batteries.
No rejected/failed proposal or initial diagnostic work disappears from costs.

| Arm | Required contrast / safeguards |
|---|---|
| Frontier editor, weight access, no critic | Experiment 8 arm A; autonomous weight proposal, no gradients. |
| Same editor + frozen predictor critic | H5(A+C) against A; critic model/pretraining cost separately disclosed; device covariates treated consistently. |
| Same editor without weight access | Same visible queries; may propose prompt strategies but cannot modify weights; prompts frozen before final scoring. Score both standard-template base and its selected prompt strategy; never call a prompt gain a weight gain. |
| Cheap editor with and without critic | Same interfaces, budgets and selection; identifies whether expensive editor reasoning is needed. |
| Random-direction edits | Match each A candidate's tensor support/layer/sparsity and **effective bf16** Frobenius norm, not just intended fp32 norm; matched diagnostic and selection budget; independent direction seeds. |
| Single-draw best-of-N | Draw independent proposals in one batch from base, no adaptive transcript/archive; pick by real visible evaluation. Run both random and LLM proposal variants. N equals the adaptive comparator's real validated candidate count (capped 20); match total diagnostic/evaluation budget, include rejected proposals. This implements the Meta^n selection control. |
| Adam and Muon, if feasible | Train only on visible items, same effective target-parameter support, forward/sequence budgets; all tuning charged. Freeze LR and other settings on separate pilot data. Gradients allowed only here, backward/FLOPs/time disclosed. If an optimizer cannot execute validly within a cell, record infeasible before outcomes, not a loss. No full H5 superiority claim against an omitted optimizer. |
| Untouched base / sham reload | Checks restore, export and evaluation drift; exact tensor hashes and logits check before/after round trip. |

### Endpoints, effect sizes and power gate

**Primary reproduction endpoint:** paired reduction in MATH-only hit-32768 rate, per episode, at
50-call budget, standard prompt. Planned minimum meaningful/detectable reduction **20 percentage
points**, with positive MATH accuracy improvement as a mandatory guardrail (adjusted lower CI >0;
point gain ≥10 points) and no increase in empty/invalid-final-answer rate >5 points. Independently
report valid-answer rates, token distributions and a blinded repetition audit; a truncation win alone
cannot pass. Primary has 500 independent problem clusters, not 2000 independent items.

Secondary: MATH exact-match gain (planned detection target 10 points), sealed IFEval strict gain
(10 points), GSM8K exact-match gain (10 points), budget response at 10/25 calls, guardrail benchmarks,
sparsity/norm, critic selection accuracy, wall/GPU time, editor tokens and total cost. Gemma negative
control: repeat 30 paired 50-call episodes with the same editor, its own pinned revision and
8192-token/model-specific sampling protocol; planned equivalence margin ±5 points MATH gain,
not “cannot improve” from a nonsignificant test. No transfer claim from one Qwen/Gemma pair.

These are **design targets, not a guaranteed power result**. At fast n=25 the worst-case independent
binary-difference SE is .1414 and a single-pair 80%-power/two-sided-5% normal MDE is about .396;
at n=50 .280, at n=100 .198. Thus the existing fast tier cannot resolve a 10-point math gain.
At 500 items the analogous conservative MDE is .089, at 300 .114; pairing and extra samples may help,
but repeated samples share problems and episodes share the final set. Multiplicity increases these
requirements. For episode-level effects, MDE ≈ (z_(1−α*/2)+z_.8) × SD(paired episode differences)/sqrt(E),
using E=30 (or the ten-goal cluster sensitivity), plus a separately estimated finite-problem component.

Before running editors, use independent controls at the actual 32k tier to compute R_b and episode/
problem variance. Require primary MDE ≤.20, and ≤.10 for claimed capability secondary endpoints,
with multiplicity accounted for. Use an empirical two-way episode/problem bootstrap power calculation
including base covariance; freeze its code/config. If achieved MDE exceeds target, expand sealed items
where possible or pre-register 100 episodes per cell; otherwise classify the cell underpowered. Do not
lower effect thresholds after seeing edits. A 1-point final-score noninferiority margin needs a separate
power demonstration; 30 episodes and 500 questions do not automatically resolve it.

### Success, falsification and reproduction labels

Arm A efficacy passes only if the primary adjusted lower CI >0, mean reduction ≥.20, both answer
validity and MATH accuracy guardrails pass, and the improvement exceeds the sham/no-weight and
norm-matched random controls with adjusted lower contrast CI >0. A CI whose upper bound is below
.20 rules out the registered meaningful loop effect for this editor/budget; otherwise failure to pass
is inconclusive, not proof that no editor can work. Report frequency of successful episodes, not best run.

**Reproduced (substantive Experiment 8 pattern):** qualified 32k base in the looping regime
(MATH cap-hit ≥.80), registered A efficacy passes, and at 50 calls mean loop reduction ≥.50,
MATH gain ≥.30 and IFEval gain ≥.30 with all three simultaneous lower CIs above zero. These planned
magnitude criteria approximate the reported pattern; do not claim numerical equivalence. Numerical
replication additionally needs all source settings/batches/items/checkers reconciled and simultaneous
CI precision supporting pre-registered ±.10 equivalence margins around the source changes.

**Partially reproduced:** a sealed, statistically supported weight-edit improvement passes relevant
validity checks but misses the magnitude/transfer criteria, or uses 4096 tokens/restricted practice
settings/another editor. A simple replay of the published layer edit is a positive-control reconstruction,
not independent editor rediscovery. Failed baselines/settings and the restricted practice adaptation
must accompany every reproduction label. Gemma equivalence supports only a bounded null for this
protocol; a positive Gemma result contradicts the negative-control pattern, not the Qwen observation.

**H5 arm A success is a different endpoint:** critic-gated editor reaches the pre-set visible goal
(U gain .10, MATH cap-hit reduction .20) using ≥30% fewer real P2-equivalent evaluations than both
A without critic and feasible Adam/Muon, while final sealed MATH gain is noninferior within .01.
Require upper simultaneous CI of each mean evaluation-cost ratio <.70 and lower CI of each final-score
difference >−.01; additionally retain the reproduction validity guardrails. Include all diagnostic
costs in sensitivity analyses. Time-to-target is capped at the cell budget, with failures assigned
budget+one evaluation for restricted-cost analysis; report success rates and censoring separately.
An arm that never reaches target cannot demonstrate efficiency by making zero evaluations.

If 30-episode ratios overlap .70, follow the existing escalation rule to 100, but first commit the
amended analysis and reserve α=.025 for each stage (Bonferroni across two looks); use α=.025 at the
initial look from the outset. At the final stage a lower simultaneous ratio bound >.70 rules out the
registered 30% saving; a final-quality upper bound <−.01 rejects noninferiority. Intervals crossing
thresholds are unresolved. If every arm fails to reach target, H5 is untested at that goal, not passed
or universally falsified. Missing optimizer controls limit the conclusion to the editor comparison.

### Analysis, multiplicity and failure accounting

Intent-to-treat: all registered episodes included. Invalid delta, timeout, budget violation or crashed
editor retains zero gain/base outcome and consumed cost; report counts and a complete-case sensitivity.
Infrastructure failure before editor exposure can be rerun once with the same seed and linked log;
never selectively rerun a poor scientific outcome. Freeze one final checkpoint per episode by visible U.

Use 10,000 paired bootstrap resamples of episode blocks, preserving all arms/budgets and the shared
base. Report mean effects, medians, 95% CIs and episode success fractions. Cluster by goal in a
sensitivity analysis (only ten independent goal formulations); use a crossed problem/episode bootstrap
for population-of-items inference, keeping all samples of a problem together. Bootstrap shared bases
once per replicate; never pseudo-replicate them as separate controls. Fixed-set conditional inference
and item-population inference are reported separately. Report rank robustness across devices before pooling.

Primary 50-call truncation test first at family α=.05 (or .025 per escalation look). Gate the capability
and control superiority family with Holm correction across all registered confirmatory secondary
contrasts, including budgets, editors and benchmarks; no selection of a favorable budget. H5 uses an
intersection-union requirement across all comparator/noninferiority conditions, with simultaneous
Bonferroni bootstrap bounds across the fixed list of contrasts. Gemma equivalence uses two one-sided
tests with its own Holm-adjusted family. Publish the full contrast list and exclusions before execution;
all unregistered subgroup/mechanism analyses are explicitly exploratory. Preserve negative results,
raw item records, prompts, proposed/rejected deltas, effective bf16 norms and manifests; SVD-derived
features must be computed in fp32 with version/hash recorded.

### Editor cheating and confounding threat register

| Route | Required preventative check / falsifier |
|---|---|
| Read sealed files, caches, scorer answers or repo history | Separate principal, minimal read-only mount, no network; deny/audit filesystem and tool calls; overlap scan; any access invalidates that arm's confirmatory run. |
| Test oracle via metrics, exceptions, timing, critic training or repeated final scoring | No live final feedback; generic errors; fixed release time; critic trained only on disjoint training ancestry/goals; final artifact commitment before scoring. |
| Memorized public questions or published fix; download known edited model | No browsing/artifact imports; blind task prompt; audit traces; private problem sensitivity and source-edit positive control; disclose irreducible pretraining contamination. |
| Overfit the 400 visible problems, select lucky samples or exploit visible scorer | Fixed query cap, log every attempt; sealed transfer endpoints; frozen seeds; compare visible/sealed gap; no critic tuning on final episodes. |
| Truncate, force EOS, emit empty/refusal text, skip thinking | Immutable generation/tokenizer/chat/stopping code; score weight-only artifact in clean process; report EOS/length, valid final answers, correctness and blinded repetition audit. Early EOS through weights still fails guardrails. |
| Hack answer parser/IFEval formatting, print gold numbers or repeated candidate answers | Freeze/check scorer versions, blinded manual audit of all parser-disagreement/invalid classes and a seeded sample of successes, disclose exact-match limitations; no post-hoc parser changes without rerunning all arms. |
| Alter tokenizer, model code, config, hooks, sampler or install a prompt wrapper | Export tensors only with names/shapes validated; evaluator reloads trusted base architecture; hash all nonweight inputs, canonical final prompt and buffers. Prompt-control arm labeled separately. |
| Hidden gradients, autograd, finite-difference search or unmetered model execution | No model executable in sandbox; all forwards through metered broker; reject backward/gradient requests, monitor calls; numerical directional probes count fully. Inspect exported code and logs. |
| Hide retries/batched queries/read-mode work, exceed editor tokens/time, use outside models | Broker enforces call/sequence/token caps; deny network/unapproved executables; log editor API usage; include all attempts and tool time. |
| Archive/best-of-N advantage or pick winner after sealed scores | Registered independent single-draw controls, equal real budgets, fixed visible selector, checkpoint hashes committed first; report every episode. |
| Critic exploitation, host-ID shortcut or family leakage | Freeze predictor per episode, uncertainty fallback, holdout `llm_edit` and ancestry, device-only baseline and device-held-out analysis; audit false negatives by shadow evaluation. |
| Norm mismatch, bf16 erased edits, NaNs, corrupted export or state carryover | Match Δ_eff support/norm, finite tensor check, clean reload/sham/logit comparison; hash untouched tensors and reset model/cache/RNG each episode. |
| Hardware/batch/seed cherry-picking; dropping failed episodes | Block randomization, actual-runtime manifests, matched-device predecessor/base, immutable registered ledger and intent-to-treat analysis. |
| Claim gradient-free speed from omitting backward/editor/training costs | Separate forward-call, sequence/token, full-evaluation, FLOP/GPU-time and editor spend ledgers; disclose critic training amortization. |

Board disposition: **REVISE**. Release B1–B6 with artifact-backed checks and a new independent review;
this document alone neither qualifies a target nor satisfies the H5 gates.

Branch: h5-board-a
Usable fast-tier targets: provisional MMLU-Pro/HellaSwag; keyword refusal/over-refusal and KL conditional on qualification.
Same-host-base decision: required; successor edits use matched-host predecessor, never an unmatched base.
Primary endpoint + MDE: 50-call sealed MATH hit-32768 reduction; planned 20-point MDE, pending power qualification.
Number of episodes: 30 paired per arm per budget (90 per arm), possible 100 per cell; Gemma 30 at 50 calls.
Top blocking item: independent matched-device repeat panels and qualified 32k endpoint precision are incomplete.
