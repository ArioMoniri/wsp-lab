# Lab notebook

Short dated notes, newest at the bottom, compiled from `sessions/`, `reviews/` and `board/`. Times are UTC. Numbers
are quoted only where a run id, review file or log is named; anything else is described, not measured. Board A is the
ML-correctness review, Board B the journal-style review (`docs/REVIEW_PROTOCOL.md`). Keep adding to this file by
hand when something happens; the README status block (`scripts/update_readme_status.py`) is the automatic part.

## 2026-11-03 (evening)

**P0, setup (23:00–23:10).** The server came up (RTX PRO 4500 Blackwell, 32 GB, CUDA 13.2). I created the repo and a uv venv, set the rule that long jobs run under tmux, and moved each branch into its own worktree. Read the Takehi et al. blog and wrote H1–H4 plus the alignment sub-study
(`docs/HYPOTHESIS.md`) and a P0–P6 roadmap before touching data. Scope grew during the first hour to include two-board
reviews, a paper draft per phase, a reference checker, a GitLab runner for CI and a message board. Six branches were
delegated at once, each in its own worktree.

**P2, eval battery (23:10–23:25).** First CPU tests failed on a missing module, as intended. 9 CPU tests, then 31,
and an eight-item GPU determinism test passed. A provisional full run was killed before it produced any score and
no number from it was kept. The durable two-budget job (`scripts/eval_p2.sh`, 4,096 and 32,768 tokens) was
started instead. Board A approved `feat/p2-eval` with two non-blocking notes: results depend on batch size
(`run.py`) and the model and WikiText revisions were not pinned. Both came back later as the batch work.

**P1, reference audit (23:27–23:50).** The first reference ledger was written by a model and Board B found author
lists and quoted excerpts wrong. A mechanical check (`scripts/verify_refs.py`) fetched the sources and compared:
23 entries needed correction, and the first model-written excerpts were mostly fabricated (8 of 12 in the first
sample, noted in `harness/lessons.md`). After three rounds, 33 references verify. The rule now is that nothing
reaches a paper section unless the verifier exits 0 first.

**P1, zoo builder (23:35 onward).** Board A rejected the first builder twice that evening. Round one: the
abliteration kernel was not what Heretic computes (support, per-layer mode, row-normalisation), dry runs wrote into
the real manifest, a missing family silently fell back, and the refusal-direction cache had no provenance. Round two
asked for a parity test against Heretic's own `Abliteration.modify_model`; the kernel is now stamped
`heretic2-classic-projected-rownorm-none` because row normalisation is still not implemented. One bug found on the
way: `direction_index` max was 24, which indexes out of range for 24 layers; it is 23.

**P1, Heretic ARA and LoRA generators (23:34–00:00).** Board A sent `feat/p1-heretic-lora` back with four blockers:
the GPU smoke tests had run on uncommitted code (manifest SHA a000d79 against head ac85fc6), touched tensor names kept
the PEFT prefix, the base revision was pinned to `main`, and refusal/KL metrics were read by list position. All four
were fixed with regression tests and the re-review approved it (49 CPU tests). Left open: Heretic 2.0.0 optimises a
row-scaled objective but exports `W + BA`, so our artefact matches the export, not the objective; an upstream report
has not been filed.

**P1, plots and animation (23:36–23:50).** Board A rejected `feat/plots-design`: the demo PNGs in `paper/figures/`
carried invented benchmark numbers with no label, and a session note contained the host name in a public repo. The
user then ruled that no demo, mock or placeholder values may appear anywhere in `paper/`, `docs/`, figures or tables
(`AGENTS.md`). Demo figures were deleted and the secret scan now blocks host identifiers.

**P1, SSM kernels (23:12–00:00).** `perf/ssm-kernels` came back from Board A: the scan block always used the
chunked path with no device dispatch (about 5× slower on CPU, and no GPU gain in the reviewer's rerun), and the
benchmark was a single shot taken on a busy GPU. Rewritten with a loop path, a chunked path and warmups plus medians.

**P1, paper intro and related work (23:50–00:09).** A cheaper model drafted both sections and a humanizer pass edited them. Board B
returned REVISE (a reversed task-vector claim, a submission that was not anonymised, hypothesis criteria stated
loosely); after the fixes it approved. Board B also noted that source verification and the study sections are required before submission.

**Board C (00:10).** An assessment of LLM-in-the-loop designs (`docs/MODEL_CANDIDATES.md`) ranked a critic-only
shadow mode first, editor plus critic second, LLM-as-optimiser last, and required a zoo of at least 600 edits and 150
trajectories and H1 ρ ≥ 0.6 before any critic-gated run. This became the P6 order in the roadmap.

## 2026-11-04

**P2, base model (night).** The 4,096-token pass of `p2/base-final-4096` completed: GSM8K 15.5%, MATH-500 6.0%,
IFEval 42.0%, MMLU-Pro 20.3%, HellaSwag 51.3%, loop rate 90.2%, refusal 99.0% (`docs/RESULTS.md`). We expected
IFEval nearer the blog's 21% and did not get it; cause unknown, so the number is not used. The 32,768-token pass died
with CUDA out of memory at about 03:25 while the zoo build shared the GPU. It has to be rerun with batch 16.

**Merges (01:30).** `perf/ssm-kernels` (Mamba-2 CUDA kernels; `mamba-ssm` 2.3.2 and `causal-conv1d` 1.7 compile on
cc 12.0 only after changing `-std=c++17` to `c++20` in both `setup.py` files) and `fix/eval-lora`. Board A's
re-review of the SSM branch was happy except that Ruff had to exclude `data/`.

**P2, LoRA evaluation and decision D1 (01:10).** Board A's first review of `fix/eval-lora` found that its real-adapter test skipped in every
place it could run, and that merging a 5-step adapter into bf16 weights rounds away 86.6% of the non-zero delta
entries. Rounding alone moved logits by about 40% of the intended effect. Decision D1: store and evaluate the
effective delta `bf16(W + Δ) − W` and keep rounding statistics with each entry.

**P1, zoo identity (01:00–02:00).** Third and fourth Board A passes on the builder: entries are reused only if
base revision, family, kernel variant, overrides, parent and refusal fingerprint all match (`reuse_key`);
smoke and production entries are labelled and kept apart; chain steps record whether they are sequential or relative
to the base. The last blocker was that stricter eligibility thresholds left already-built parents admissible.
`p1-zoo-full` started at 02:02. An earlier `git add -A` had swept 600 MB of build artefacts into a commit and GitHub
rejected the push, so `data/` is now ignored wholesale.

**P2, batch evaluator (03:18–04:05).** Board A rejected the first version for five reasons: smoke and limit runs
overwrote canonical zoo entry directories, git SHA provenance was wrong, `eval.json` lacked eligibility and index
appends were not idempotent, `eval_fast.yaml` did not match the decided tier, and the roadmap's 5,617 GPU-hour figure
was wrong. The corrected estimate is 9 to 12 minutes per entry, about 120 GPU-hours for the 700 entries that existed then. The first smoke
(`20261104-032002-1c2cc588`, two entries, n=4, 32 tokens) took 0.30 minutes per entry; that timing is engineering
only and says nothing about the real tier. Merged after re-review (9146c5f). D2 (three tiers) and D3 (a stratified
subset first) were written down at about 03:50.

**P3, features and baselines (03:17–05:05).** The first pass failed Board A on leakage: trajectory folds were built
from rows that had targets, so removing an intermediate entry disconnected a chain and let a descendant land across a
fold. Also the hybrid linear-attention tensors were missing from the pooled spectrum, a malformed edit aborted the
whole batch, and resume did not check enough identity. Fixed with tests: full-ancestry grouping, fifteen tensor
types (`feature_version` 3), per-entry failure records, resume identity including thread count and library versions.
One more blocker remained at 04:00 (a malformed `resolved_base_revision` still aborted the batch); the branch
merged afterwards as bf99575. Full extraction then ran on four CPU workers (`p3-extract`).

**CI (04:30).** GitLab pipeline fully green on the self-hosted GPU runner. An earlier red run came from bypassing the
pre-commit hook during merges, which let formatter drift through; The formatting check lives in `scripts/precommit.sh` (ruff check, ruff format --check, tests), which the pre-commit hook runs;
the pre-push hook only runs `scripts/secret_scan.sh --history`.

**Zoo growth (05:00–06:00).** Zoo went from 588 entries at 04:05 to 697 at 05:05 as chains finished.
Heretic ARA at production budget costs about eight minutes per trial on the shared GPU, so the local config
excludes that family.

**Lambda (06:00).** The user approved up to $150 (D4). A job for the Heretic ARA family was launched on a rented
A100, with a per-hour price cap and automatic termination; results are pushed to the private HF dataset. At the time
of writing `board/lambda_spend.md` has no rows, so spend is untracked there until the ledger is filled.

**README and this notebook (06:10).** User feedback was that the README read like generated text and did not
match what was done. Rewritten from the artefacts, with a status block that a script regenerates. Items the old
README claimed that do not exist yet: a predictor training script, a predictor config, and an LLM-reader prompt file.

## Open as of the last entry

No zoo entry has completed a fast- or cheap-tier evaluation. The only per-entry results are the engineering smoke run
`20261104-032002-1c2cc588` (base plus two production edits, n=4, 32 tokens), which measured pipeline timing and not
benchmark performance. Still open are the base 32k rerun, the second reproduction run, a clean timing of one entry, the D3 subset,
Heretic ARA on Lambda, and all of the predictors. Board A's non-blocking items from the Heretic and P3 reviews
(selection prompts disjoint from benchmark prompts, unpinned dataset revisions, GatedDeltaNet coverage in the LoRA
smoke tests) are still on the list.

**P3 v4 features complete; D3 evaluation parallelised (4 Nov, 10:20–14:15 UTC).** Feature extraction v4 finished over the
full zoo (1,235 entries, 0 failed, mean 66 s/entry, `data/features/qwen3.5-0.8b/manifest.json`). The HF zoo store went
public at the user's request (D7 addendum 2) and all non-chain deltas are on the hub. The D3 subset is evaluated on
three hosts (D3 addendum 3): this server (chain entries + 10 controls), an H100 PCIe with three evaluator processes
(generation at the canonical batch 32 is launch-bound, so processes share the GPU) and an A10 with two; every host also
evaluates the base model. First Lambda attempt lost its log by self-terminating on failure (lesson in
harness/lessons.md); the job now keeps failed instances and uploads the log. The P4 training path was exercised on real
v4 features with the 48 targets available so far (`data/runs/p4/smoke-m1-realdata`, M1, family split, 5 folds): a
data-path check only, no claims from it. Gemma 3 270M (both variants) downloaded with the official licence.

**H5 council, P4 first suite, reliability section (4 Nov 14:15 – 5 Nov 00:20 UTC).** User asked whether the blog's
Experiment 8 (an editor LLM improving Qwen3.5-0.8B by hand-editing weights) can be reproduced by our agents. Council
wave: researcher (protocol extracted, docs/H5_RESEARCH.md), Board C design (docs/H5_DESIGN.md), Board A
pre-registration + noise qualification, Board B reporting requirements; synthesis in docs/H5_PROTOCOL.md. Editor
harness implemented and merged after three Board A rounds (482 tests; production disabled; pilot preconditions in
GitHub issue 31). Eval reliability fixes merged after five rounds (same-device references, immutable repeats,
reliability statistics, labelled backfill). D3 subset completed on three hosts (300/300; H100 $38.33, A10 $11.48).
Same-host retest bit-identical; three-device control panel in docs/RESULTS.md. First P4 suite: Board A REVISE, H1 not
demonstrated (M2 ρ .54/.49 vs StatNN .61/.67 on the trajectory split; family-split negatives are family offsets).
P4 round 2 (nested selection, fold-mean/delta-norm/descriptor/device baselines, per-family centred metrics,
ancestry-cluster paired bootstrap, multi-seed) merged after three rounds with the 102 intended numeric changes
documented in DECISIONS. Paper: reliability subsection and experiments.tex corrections merged after scribe →
humanizer → Board B (APPROVE, claims 4/5; 10 pages). ESM protein-LM idea assessed by Boards A and C: do later
(P4c). Queued (tmux p4-after-chainpred): 89 chain predecessors (running, 3 workers) → seed-1 repeat panel →
P4 suite v2 (d3v2-*). ARA rebuild on Lambda at 8/12.

**Council deliberation and reframing (5 Nov, 12:00–14:30 UTC).** Five-member expert council (weight-reading
researcher, JEPAWG researcher, features CTO, metrology CTO, strategy CTO), two rounds with mutual challenge, chair
synthesis, Board A critique (docs/council/2026-11-05/). Agreed: descriptor-only (family + edit parameters) reaches
Spearman 0.52/0.61 vs StatNN 0.57/0.65 on mmlu_pro/hellaswag (weight features add ≈ 0.04; 0.20 on KL); only 94 of 300
evaluated edits are non-null; sampled benchmarks are at or below noise while teacher-forced fields are near-perfectly
reliable; the blog never showed Δ-only benchmark prediction. Protocol-v2 suite finished (26 runs, 3 seeds): best
neural ρ 0.45 vs StatNN 0.57/0.65; held-out families ≈ 0. Board A: H1 not demonstrated, H2/H3 undetermined, synthesis
REVISE → frozen prospective design docs/PREREG_H1_TF.md (H1a residual TF readability; H1b; H2 history ablation parked),
HYPOTHESIS/ROADMAP amended. User defaults adopted: TF-v1 programme yes; 30 local GPU-h; no history budget; no new
Lambda. Paper: protocol-v2 results, "What the baseline exploits", "Reframing", new scoped title; Board B APPROVE after
three rounds with a scripted audit of 723 numbers (15 pages). Codex usage limit reached 13:35 UTC (until 6 Nov 21:10);
TF-v1 tier build rerouted to the Claude API (Sonnet). ARA corrected rebuild at 6/12 on the A10.

**TF-v1 tier accepted; gate 1 passed (5 Nov, 14:30–15:45 UTC).** Teacher-forced tier built (Codex handoff to Claude
Sonnet after the Codex usage limit), two Board A rounds (letter-scoring stop, phrase-set refusal margins, hash-ranked
nested panels, required device id, content-verified resident cache; KL 200 × 64, HellaSwag 250 nested, MMLU-Pro
1,000, WikiText 64, refusal 50 + 50). Round-2 pilot on the committed source: warm median 1.805 min/entry (gate line
2.0), cache 5.38 GiB, 2.36 MiB/entry; Board A ACCEPT, merged e4103ec (542 tests; CI green after fonts-liberation +
pytest pythonpath fixes). Before the fresh non-null panel: G1 bitwise repeat, G2 same-item bridge, single-device
scope, PREREG amendment, sealed-panel manifest, §6 qualification + §5 power (running). README rewritten for the
reframed study and accuracy-checked (114 claims, 18 fixes). ARA corrected rebuild 7/12 on the A10.

**Re-qualification result (6 Nov, 00:50–01:37 UTC; docs/reviews/P5/tf-v1-requalification-2026-11-06.md).** D3 stratum
scored under one source (378/381; 3 legacy rows excluded). MMLU-Pro gold log-prob NOT qualified even under the amended
rule (max half-width 0.114 > 0.10; tripwire 0.226 > 0.20); WikiText log-ppl qualified. §5 gate: exploratory
(power-limited); no interval procedure passed all simulation checks. Empirical out-of-fold θ̂ = 0.129 (n_eff 40) with
every interval crossing zero; leaving out high-item-variance rows gives θ̂ = 0.007. Board A adjudicating whether to
switch the primary to WikiText, run the sealed panel as exploratory, or stop and report. Corrected ARA family landed
(12 entries) and featurised after the delta-key mapping fix (D6 addenda 4–5, PREREG A3). Lambda closed at $116.22.

**Board A decision on the re-qualification (6 Nov 02:30 UTC; docs/reviews/P5/board-a-requalification-adjudication-2026-11-06.md).**
Decision (c): the sealed MMLU-Pro panel is not scored (endpoint unqualified; power lower bound 0.43–0.47; one-shot
asset). H1a is untested, not falsified. θ̂ = 0.129 is borderline (all six intervals within 0.01 of zero) and carried by
the large-effect rows; the item-split replicates (0.105/0.117) argue against pure item-variance noise, so neither
"real" nor "artefact" is established; the §5 step-1 companions (outcome-free family × norm strata, flexible nuisance,
weighted/restricted θ̂) settle it on CPU. WikiText log-PPL is qualified and may carry a separately named exploratory
claim H1w by amendment A4 (CPU gate ≈ 3 h; fresh scoring ≈ 1.9 GPU-h only if it passes). Paper gets the permitted
sentences. Delegated: A4 + companions + WikiText gate (Sonnet); paper outcome section (Haiku).

**Data-loss incident and recovery (6 Nov 03:18–04:30 UTC; D8).** A committed `data` symlink (ea78234) destroyed the
canonical data directory on merge. Restored from the public HF dataset (137 GB): 1,259 entry dirs, 716 dense + 122
LoRA deltas, 389 eval.json, 200 records.jsonl, both Lambda base evaluations. Board A plan (d64e410): 12 result groups
re-derivable, 21 regenerable, 4 irrecoverable (to be disclosed); 358 chain deltas lost → H2 no longer testable;
regeneration ≈ 3.15 GPU-h + 8–10 CPU-h from detached worktrees at verified commits (running). Canonical data now at
/home/ubuntu/wsp-data outside the checkout; pre-commit, ignore and wt.sh merge guards added; 25 stale worktrees
removed. Backup policy (S3 versioned copy + EBS DLM repair) awaits the user's approval. Earlier the same night: H1w
WikiText gate FAILED (θ̂ 0.007), companions keep the MMLU residual signal positive in point estimate but on zero;
A4 recorded; paper outcome section merged (17 pages).

**D8 regeneration complete; paper updated (6 Nov 16:03–17:30 UTC).** Steps 1–6 of the Board A plan executed from detached
worktrees (docs/reviews/P5/d8-regeneration-2026-11-06.md): 838 stored / 63 aliased / 358 lost; 901 feature sets re-extracted;
three base re-evaluations bitwise across source trees; seed-1 panel reproduces every printed value (KL differs by ≤ 3.4e-4
between seeds; RESULTS.md corrected); TF-v1 cache root, G1 12/12, G2 11/11 and 28/28 (y, h) rows reproduce. Step 6: the
protocol-v2 suite on the recovered set (d3v2r-*, 297 rows / 270 clusters, 25 of 26 runs) keeps the 5 Nov ordering (StatNN
0.54/0.65 at or above every neural model; descriptor-only 0.48/0.58). GPU 4.69 h against the 3.15 h plan (two deviations
inside Board A's scope). Merged ab20983. Paper: appendix states the completed regeneration and labels, TF-v1 and seed-1
cite regenerated run ids, v2 table keeps original ids marked lost, recovered-set paragraph added; audit passes in both
modes. CI test-gpu was red because the runner's read-only HF cache lived under data/; repopulated, green. The regeneration
and scribe delegates on the Claude API instance both died on repeated API errors; the orchestrator finished both.

**Board A close review and Board B whole-manuscript read (6 Nov 18:00–20:30 UTC).** Four providers failed in a row (Claude API
instance: repeated API errors; Antigravity: quota; Grok: no credits; OpenCode: blocked response); both reviews ran on Copilot.
Board A: ACCEPT WITH REQUIRED EDITS, eight paper edits applied (KL 3.3472e-4, 901-entry extraction, ordering scoped to the
seed and trajectory splits, G1 14/14 and the A2.1 results as reported from logs, base hashes with the shared-field qualifier,
class C meaning, TF note scope), D8 addendum 2 adopted, audit reads the regeneration JSON files; D8 stays open until the
backup controls are verified. Board B: MAJOR REVISION (1 blocking: author name outside the anonymity switch; 2 major:
ordering claim, access links; 4 minor; 1 nit), all addressed: author behind the switch, access paragraph naming the
recovered-rerun commit, broader-impact paragraph, compute statement from the spend ledger, contribution stated once,
StatNN defined. Paper 20 pages; audit PASS in both modes.


**7 Nov 2026, 06:00–09:30 UTC.** User decisions: GPU time approved (local and Lambda beyond the earlier cap); recreate the data
lost in D8; run the agentic council on every negative and neutral result to find a world-model structure. Chain regeneration
(branch `feat/d8-chain-replay`): the zoo sampler code of the chain build is not in the repository, so the 358 lost snapshots
are replayed from their surviving edit descriptors; pilot on three intact chains reproduced random_dir and abliteration steps
bitwise (refusal fingerprint `0b6e272d…` rebuilt from scratch) and re-trained LoRA steps within bf16 rounding (norm differences
≤ 9.1e-4 relative; the matrix-level error metric was added afterwards and is still to be measured). Full restore running as three
GPU shards. Council 7 Nov (`docs/council/2026-11-07/`): four members, two rounds, chair synthesis in three versions reviewed by
Boards A, B and C (`docs/reviews/council/`); final state ACCEPT WITH REQUIRED EDITS by all three boards as a programme proposal,
edits applied in v3.1 and merged (9f59e3d). Programme: oriented local response (base-anchored sensitivity field as the primary
arm; predecessor chart as registered secondary), WikiText-first, Part A rotation-and-ladder intervention on 40 existing edits
and Part B exploratory retrospective decide whether sealed rows are bought; gates G0–G8 with ≈ 29 GPU-h of allocations on H0;
the registration (PREREG appendix A5) is the next deliverable and must be frozen before any field is computed.

**7 Nov 2026, 09:30–11:00 UTC.** PREREG appendix A5 drafted and reviewed by Board A in three passes (ACCEPT as a draft pending the
A5.11 manifests; merged 62d73bf). G2 code merged: `src/wsp/features/rotation.py` (rotation controls), `response_field.py`
(gradients, K-FAC, contractions, predecessor audit), `response_chart.py` (chart tokens and augmented-StatNN columns),
`src/wsp/eval/response_intervention.py` (Part A selection, variants, manifests, scoring plan), `scripts/a5_simulation.py`.
Open design item for the A5 freeze: at the chair-default projector dimension k = 32 the augmented-StatNN layout has 318 slots and
571,836 columns, far too wide for a tree on a few hundred rows; k must be reduced (4 gives 48 contraction columns per matrix slot)
or the contractions pooled before freeze. Codex gpt-6.1-terra stopped early on three implementation tasks today; rerouted to Claude Opus.

**7 Nov 2026, 11:00–13:00 UTC.** Chain replay finished and installed: 351 of 358 lost snapshots back in the zoo (sidecar 1,189
stored / 63 aliased / 7 lost); re-trained LoRA steps measured at 3–15 % matrix relative error (norms within 1e-3), recorded as the
"re-trained" exactness class (regeneration report §7, D9). TF-v1 (y, h) audit: all 27 bitwise-lineage rows reproduce, 13 of 14
re-trained-lineage rows differ as expected. Features re-extracted for the 351 (0 failed; 99 s/entry under contention). Full-set P4
view: 1,241 feature sets with complete ancestry, **342 matched rows** (original 345; 3 rows lost with the 7 entries); suite
`d3v2f-*` running (tmux `d8-p4-full`). G6 census running on H0 (tmux `d8-g6-census`): G1 repeats of the two regenerated
predecessors, then TF-v1 on the 635 stored chain entries without a record (repriced from 296, ≈ 11 GPU-h). A5.11 manifest package
being built on branch `prereg/a5-manifests`.

**7 Nov 2026, 13:00–14:00 UTC.** A5.11 manifest package merged (`docs/prereg/a5_manifests/`: components, selection, references,
text groups, code hashes; builder `scripts/a5_build_manifests.py`). Chair ruling recorded in A5.3.1: an edit whose actual
predecessor state is lost is ineligible, not a selection block; under it the Part A selection is 30 lora_sft (10/10/10 by
norm tertile) and 10 chain (4/3/3), 40 distinct components, 3 chain picks with re-trained lineage flagged. References to
score before G3: 24 originals and 6 predecessors (31 min). Gradient/probe panels, control seeds and null-size inputs are
being built on `prereg/a5-panels`. G6 census restarted on 618 entries after excluding 17 legacy-revision rows
(`base_revision: main`, pre-production commits; list in `runs/d8/g6_dropped_legacy.json`).

**7 Nov 2026, 14:00–16:00 UTC.** Board A reviewed the A5.11 package and the three chair rulings (ACCEPT WITH REQUIRED EDITS,
`docs/reviews/council/board-a-a5-package-2026-11-07.md`, seven-step freeze checklist). Checklist items 1–2 merged (1974b65):
freeze-prep amendments appended to A5 (registered forty picks, 20-prompt harmful exception, rejected extraction fill, 500-ranked
first-64 prefix, whole-article WikiText rule, executable null-size proxy mapping, G3 staging outputs frozen by schema, future
sealed-builder exclusion obligation); `scripts/a5_simulation.py --inputs-spec` accepts the edit-minus-predecessor donor; positive
lost-predecessor check in the manifest builder; procedural pins and cutpoints in `code_hashes.json` / `selection.json`;
`response_field.registered_field_items` enforces the registered panel sizes. Items 3–7 (snapshot closure, manifest and panel
rebuild, null-size run, hash freeze) wait for the G6 census to close.

**7 Nov 2026, 22:10 UTC.** G6 census closed: TF-v1 records for all 618 stored chain entries without one (10:56–22:09 UTC on H0,
≈ 11.2 GPU-h as repriced); 652 of 675 chain entries now carry a TF-v1 record (the rest are the 7 lost, their descendants and
the 17 legacy-revision rows). H2 census: 95 of the 99 root-to-leaf lineages with ≥ 4 nodes are now fully TF-v1-scored (512
nodes), up from 16 lineages; still below the historical 162-lineage stand-in, so H2 stays parked per the synthesis. Full-set P4
suite: 24 of 26 runs done, no errors. G1 primaries for the two regenerated predecessors queued behind the census.

**8 Nov 2026, 00:30 UTC.** Full-set P4 suite finished (26 runs, `docs/reviews/P4/d3v2f-summary-2026-11-08.md`): 342 rows / 278
clusters; StatNN 0.56/0.66 and 0.55/0.64 against the logged 0.57/0.65, descriptor-only 0.51/0.61 against 0.52/0.61; ordering of
the 5 Nov suite holds. G1 closed at 14 of 14 (the two regenerated predecessors and the base bitwise, label g1b). Regeneration
report §7 updated. Next: Board A checklist items 3–6 (snapshot closure, manifest and panel rebuild, null-size run), then the hash
freeze and G2.

**8 Nov 2026, 01:00 UTC.** Paper updated for the chain replay (merged d6209ec, 21 pages, number audit PASS in both modes; Board B
accept-with-edits applied, `docs/reviews/docs/board-b-d8-step7-2026-11-08.md`): full-set rerun paragraph and table (seed-0 baselines
within 0.04 of the logged values, neural means drifted by a median of 0.09, so the rerun is a re-instantiation and not a close numerical
reproduction), G1 closed at 14 of 14, the (y, h) audit split by exactness class, H2 "stays parked" wording. Waiting on Board A
checklist items 3–6 (branch `prereg/a5-freeze-evidence`).

**10 Nov 2026, 11:00–11:30 UTC.** User approval ("all you need") received. Board A checklist items 3–6 merged (944febc): closed
snapshot, rebuilt manifests (forty ids and cutpoints unchanged; 19 references to score), rebuilt panels (82 certificate checks
pass), null-size simulation PASS for the registered run (two-sided size 0.037, ordered go rate 0.018 at nominal 0.05, yield 1.0) with
caveats for Board A: the three contrasts are rescalings of one donor block, the null reaches the "alternative" branch in 19.4 % of
replicates, one donor in the LoRA low tertile. Backup controls live (D8 addendum 3): nightly versioned S3 copy (initial 270 GB sync
running), EBS snapshot `snap-0e72f033cb0002355`. Literature check done (WeightLoop 2611.07319, Neural Thickets 2603.12228, LoL
2410.04207, blog critiques); council round 3 convened (`docs/council/2026-11-10/BRIEF.md`) on the best predictor now, a LoL-style
arm, a 4B scale check and kill criteria. Item 7 (freeze) waits for the round-3 synthesis so one Board A pass covers both.

**10 Nov 2026, 12:00 UTC.** Council round 3 merged (d83529e, `docs/council/2026-11-10/SYNTHESIS.md`): programme stands (field primary,
Part A decision experiment, WikiText first); news adds a LoL-style secondary family (learned / random / anchored frames plus a
singular-values-only control, descriptive, no funding), a capped Qwen3.5-4B scale diagnostic on Lambda in parallel (20 outcome-blind
Gaussian edits, WikiText, not a gate), a tightened H5 critic contract after WeightLoop (rebuilt-from-base, held-out support validation,
permutation controls, four logged access flags, matched random best-of-N 32 draws per block) and a joint line-stop rule at 0.8B.
Brief corrections: LoL's speed claim withdrawn; WeightLoop 8 of 8 vs random and 7 of 8 vs original; no stored chain LoRA factors.
Three delegates running: A5 amendment F5 with record changes (`prereg/a5-f5`), the G2 driver code (`feat/g2-driver`), the 4B
Lambda job code (`feat/scale-4b-diagnostic`). Board A's single freeze pass follows F5.

**10 Nov 2026, 13:00 UTC.** A5 amendment F5 merged (e3c577f; append-only; F5.1–F5.7 with HYPOTHESIS, H5_DESIGN, H5_PROTOCOL, ROADMAP,
RESEARCH_NOTES and verified refs updated; WeightLoop, Neural Thickets and LoL added to `paper/refs`). Board A's single freeze pass is
under review (`docs/reviews/council/board-a-freeze-2026-11-10.md` when it lands): checklist item 7, the null-size caveats and F5.
G2 driver and the 4B Lambda job are being written as code only; neither runs before the freeze.

**10 Nov 2026, 14:30 UTC.** G2 driver merged (0788e53): `scripts/g2_fields.py` / `g2_fields.sh` (panel hash verification, five
objectives with the scorer's tokenisation, K-FAC coverage of 186 linear modules, probe cache on the 24 MLP writers, chart projectors,
20-predecessor audit with HARD_STOP, wall-clock caps, manifests; 26 CPU tests). Finding from its CPU dry run: the A5.8.1 eigenvalue
tolerance (1e-6 x trace/dim) fails on rank-deficient factors (MMLU: 64 rows against 1,024–6,144 dims; refusal) because fp32 rounding
leaves eigenvalues near −1e-7 x the largest; WikiText and HellaSwag pass. Chair decision for the freeze follow-up: fp64 eigendecomposition
of the factors and a negativity tolerance relative to the largest eigenvalue, recorded as a dated A5.8.1 amendment with the field module
re-pinned, before Board A's post-commit audit. Board A's freeze pass (4fd1e2a) returned REVISE with seven edits, being applied on
`prereg/a5-freeze`. The 4B Lambda job code is in its commit hook.

**10 Nov 2026, 16:00 UTC.** Board A freeze edits 1–7 merged (b7f10a1, branch `prereg/a5-freeze`): closure reconciliation (37 old-seal
components / 187 census-scored members exposed), the two missing sensitivity scenarios implemented and run (constant positive mean:
size 0.041, alternative branch 0.218 [0.210, 0.226]; cell mask/noise: size 0.031, alternative 0.154 [0.147, 0.162]; all five scenarios
under the 0.07 gate), LoL/head schema pins (all frames width 4; 39-input head, 1,137 parameters per endpoint), 4B paired qualification
spec, best-of-N executable rules, chart contrasts enumerated and chart seeds registered, upper-bound exclusion power spec, A5.1.2 literal
inventory (90 hashes). 4B diagnostic code merged (f252d68). Pre-audit follow-up running on `prereg/a5-eig`: A5.8.1 eigenvalue
tolerance amendment (fp64, relative to the largest eigenvalue) with the field module, G2 driver and 4B scripts re-pinned. Then Board A's
post-commit audit, then G2.

**10 Nov 2026, 17:00 UTC.** A5.8.1 eigenvalue amendment merged (43283ec): fp64 eigendecomposition, negativity tolerance relative to the
largest eigenvalue with clamp counts recorded (toy MMLU factor: 541 of 1,948 eigenvalues clamped, worst ratio −3.8e-7; all five toy
objectives now pass), field-module schema v2, G2 driver and 4B scripts pinned (64 files), fitting-library versions recorded. Chair ruling
on the three 4B discrepancies the amendment surfaced: the registered F5.3/FP4 text governs (ρ = 0.005, SHA-derived seeds, pairing
indices 0/4/8/12/16); the builder is being aligned and re-pinned on `prereg/a5-4b-align`. Backup corrected: symlinks no longer followed,
duplicates removed, lifecycle rule for non-current versions (30 days) set.

**10 Nov 2026, 17:30 UTC.** 4B builder aligned to the registered F5.3/FP4 text and re-pinned (7ecd186; `--verify-pins HEAD`: all pins
match). Board A's post-commit freeze audit requested on main at 7ecd186 (`docs/reviews/council/board-a-freeze-audit-2026-11-10.md`).
G2 launches only on FREEZE ACCEPTED, with `PANELS_SHA256` equal to the A5.1.2 hash and the registered caps.

**10 Nov 2026, 18:30 UTC.** Board A post-commit freeze audit (a0f352b): REVISE, G2 not authorised yet. Edits 1–6 resolved
prospectively; the eigenvalue amendment and the 4B alignment accepted prospectively; all 97 inventoried hashes and the five
null-size scenarios verified (103 tests). Two blockers: the pinned G2 driver audits predecessors for WikiText only, while A5.8.1 and
A5.5.1 require a separate ≥ 16/20 verdict per objective; and the post-freeze commit range mixes governed amendments with
independently motivated infrastructure and progress commits, which needs an explicit scope reconciliation and three missing
old/new test hashes. Both repairs are in progress on `prereg/a5-audit-repairs`; Board A re-audits after the merge.

**10 Nov 2026, 19:30 UTC.** Audit repairs merged (a638efa): the G2 driver now audits the twenty predecessors separately for each of the
five objectives (independent ≥ 16/20 verdicts, short or failed objective = HARD_STOP, per-objective resume, cost estimate checked against
the 1 GPU-h cap; 36 driver tests), and A5.1.2 carries the scope reconciliation of the post-freeze commit range plus the missing test-file
hashes; `--verify-pins HEAD` passes on main. Board A's re-audit requested (`docs/reviews/council/board-a-freeze-reaudit-2026-11-10.md`).

**10 Nov 2026, 20:00 UTC. FREEZE ACCEPTED** by Board A at a638efa (`docs/reviews/council/board-a-freeze-reaudit-2026-11-10.md`): both
blocking repairs resolved, all pins and hashes verified, no further conditions. G2 launched on the exclusive local GPU under the frozen
bytes (`PANELS_SHA256` = the A5.1.2 panels hash; caps 2 GPU-h fields and cache, 1 GPU-h audit, 28 GiB). Outcome-access order holds:
G2 reads no rotation, ladder or sealed outcome.

**10 Nov 2026, 20:30 UTC.** F5.3 scale diagnostic launched on Lambda (H100 PCIe, us-west-3, instance `a6508259…`, device id H3, repo
revision 736ca25, cap 2.0 GPU-h, self-terminating): builds 20 Qwen3.5-4B Gaussian edits at ρ = 0.005, scores the FP4 pairing panel
first (base + draws 0/4/8/12/16, two fresh-process repeats), then the remaining 15 entries on WikiText; uploads to the HF zoo dataset.
Diagnostic, not a gate; the H0 half of the pairing runs after G2. G2 running locally since 20:00 UTC.

**10 Nov 2026, 21:30 UTC.** G2 first run (data/runs/g2, started 20:49 UTC): all five sensitivity fields computed (1.45 of the 2 GPU-h
cap; wall 525–1,649 s per objective, CPU K-FAC accumulation dominant), projectors and probe cache complete. The predecessor audit
stage stopped before any row: the driver's cost estimate (0.5 × field wall per row, 100 rows → 11.7 GPU-h) exceeds the 1 GPU-h cap,
although a gradient-only pass should take a small fraction of the field time. Under the stop-and-reprice rule the driver is being
amended to measure the first registered row per objective before checking the cap (`feat/g2-audit-reprice`, with an A5.8.1 note and
re-pin); the audit stage is then resumed. No registered quantity changes. 4B diagnostic running on Lambda.

**10 Nov 2026, 22:30 UTC.** 4B scale diagnostic done on Lambda (0.16 GPU-h, $0.54; instance self-terminated). H100 half of the report
(`docs/reviews/scale/scale4b-diagnostic-2026-11-10.md`, status "not counted: A3 pairing pending"): base WikiText log-PPL 2.379 nats;
3 of 20 Gaussian edits at ρ = 0.005 improve the base (0.15, Clopper–Pearson [0.03, 0.38]), none resolved; 2 resolved worsening; the
rest unresolved. Realised radius 5.19e-3 per tensor, 68 % of elements changed. Reading: at this radius a 4B Gaussian edit is at the
64-window resolution floor, so the diagnostic cannot yet separate density from noise; the H0 pairing half runs after the G2 audit.
Diagnostic, not a gate; exposure recorded. Entries under `data/zoo/qwen3.5-4b-scale/`, records under `data/evals/qwen3.5-4b-scale/lambda-H3/`.

**10 Nov 2026, 23:30 UTC.** G2 audit reprice committed on `feat/g2-audit-reprice` (8f81ab0; 46 driver tests): the cap check now uses the
measured first registered row per objective, a per-row guard stops an overrun as incomplete, every audit row and stage records wall,
GPU seconds and peak memory (the fields of the first run already had peak memory 4.8–10.9 GB and GPU hours), and the audit stage
resumes alone after hash-checking the completed stages. Follow-up in progress: objectives processed in ascending estimated row cost and
cost-guarded objectives re-checked at the end of the same run, so one resume can complete all five; then merge, re-verify pins and
resume the audit on the GPU.

**11 Nov 2026, 00:00 UTC.** Audit reprice and ordering fix merged (1a238ee; 46 driver tests; pins verified on main). G2 audit stage
resumed on the exclusive GPU (`--resume --stages audit`; completed fields, projectors and probe cache hash-checked first).

**11 Nov 2026, 00:30 UTC.** G2 audit stage, resumed with the measured cost check: the first registered row (refusal_margin_harmful,
cheapest objective) took 394 s wall (0.17 GPU-h) and its relative error against the base-plus-cross-term approximation was 0.336
(rule: ≤ 0.1); the measured projection for the registered 100 rows is about 11 GPU-h against the 1 GPU-h cap, so the run stopped
incomplete with 0.17 GPU-h spent and no further row. Two findings for Board A: the registered audit is infeasible at its cap, and the
first measurement fails the approximation rule. Per A5.8.1's own fallback the base approximation is not used for chains; chair
amendment F6 (exact predecessor gradients for the ten chain picks on WikiText and MMLU, ≈ 20 rows, new 2.2 GPU-h cap; HellaSwag and
refusal tokens masked for those rows; a Part B sensitivity arm excluding unvalidated chain rows) is being drafted for Board A.
G2 run-1 manifest and audit file copied with hashes to `docs/prereg/a5_manifests/g2/`. The H0 half of the 4B pairing is running.

**11 Nov 2026, 01:30 UTC.** A5 amendment F6 merged (918e656; pins verified): the registered predecessor audit is recorded as run and
infeasible at its cap with its single row failing the rule; the base approximation is not used for chains; the ten chain picks get exact
predecessor gradients for WikiText and MMLU through the new `chain_exact` stage (20 rows, 2.2 GPU-h cap), HellaSwag and refusal tokens
masked for those rows, a Part B sensitivity arm excludes unvalidated chain rows. Board A reviews F6 before the rows run. The H0 half of
the 4B pairing was relaunched with the H100 job's revision and device id.

**11 Nov 2026, 02:15 UTC.** Board A accepted F6 and the reuse of the G2 fields stage (`docs/reviews/council/board-a-f6-2026-11-11.md`;
conditions: frozen identities and outcome-access order preserved, all predecessor, alias-holder, support and gradient identities recorded,
objective masks enforced, base-factor q_t flagged, failed rows retained, every attempt charged against the 2.2 GPU-h cap, the accepted
exact WikiText chain predictions used for the power/yield simulation). The `chain_exact` stage is queued on the local GPU behind the
4B H0 pairing job.

**11 Nov 2026, 02:45 UTC.** `chain_exact` complete: exact WikiText and MMLU gradients at the actual predecessor for the ten chain picks,
20 rows in 15 minutes (0.24 of the 2.2 GPU-h cap; rows 10–88 s each). Final G2 manifest copied with hashes (`docs/prereg/a5_manifests/g2/`).
Board A is checking the completed G2 run; the registered sign predictions per donor for the power/yield simulation are being built from the
WikiText field (base for LoRA donors, chain_exact gradients for the chain picks). The 4B H0 pairing runs next on the GPU.

**11 Nov 2026, 03:00 UTC. 4B scale diagnostic complete** (`docs/reviews/scale/scale4b-diagnostic-2026-11-10.md`, both halves). Same-device
repeats are bitwise on the H100 and on H0 (12 of 12 each), but the A3 cross-device pairing fails: the spread between devices is 0.662 of
the spread between edits (rule ≤ 0.1), so the H100 records stay "not counted" and the diagnostic is read on the H0 half only. Substance:
at ρ = 0.005, 3 of 20 Gaussian edits of Qwen3.5-4B improve WikiText (none resolved; 2 resolved worsening); the edits sit at the
64-window resolution floor, so beneficial-edit density at 4B is not measurable at this radius. Diagnostic, not a gate; scale expansion
stops here unless a larger-radius or larger-panel diagnostic is registered. Cost $0.54 Lambda plus 0.2 local GPU-h.

**11 Nov 2026, 03:30 UTC.** Board A accepted the completed G2 run (`docs/reviews/council/board-a-g2-2026-11-11.md`; 117 artefact hashes
verified; no new conditions). Downstream may consume: the base WikiText field for LoRA donors, the chain_exact WikiText gradient at the
actual predecessor for the ten chain picks, and nothing from any other objective in the registered Part A predictor. In flight: the
per-donor sign predictions for the power/yield simulation (`feat/a5-power-fields`) and Part A staging without scoring
(`feat/part-a-prepare`: 40 edits × rotation + two ladder variants, audits, reference plan).

**11 Nov 2026, 06:30 UTC. Power/yield gate PASSED; Board A accepted F7.** Sign predictions merged (`data/runs/a5/power/fields.json`,
sha256 85843ed2…; 141 donors: 10 chain_exact, 51 base-exact roots, 80 flagged base-approximation-unvalidated). Chair ruling F7 (A5
amendment, before any G3 outcome): the registered POWER-AND-YIELD run uses only the 61 exact-gradient donors; the 141-donor run is a
labelled sensitivity run; the null-size pass is unchanged and reproduces the registered null (go 180 / alternative 1939 / unresolved
7881 of 10,000). Registered ordered power 0.8540 [0.8469, 0.8609] (gate ≥ 0.80), alternative 0, unresolved 0.146, yield 1.0 on every
contrast; sensitivity 0.8501. Part A is decisive under A5.3.2 (`docs/prereg/a5_manifests/power/report.json`, README). Board A: ACCEPT WITH
CONDITIONS (`docs/reviews/council/board-a-f7-2026-11-11.md`): keep the 61-donor run as the registered reading; qualify decisiveness by the
F2 proxy, the .75 alternative and the single-donor LoRA low tertile; carry the shared-block dependence of the three contrasts and the
funding-only reading of the alternative branch; yield 1.0 assumes unavailable audits pass, so the actual audits remain required; all F6
and pre-G3 requirements stand. Part A staging (`feat/part-a-prepare`) at 19 of 40 edits; `score` starts only after staging completes.

**11 Nov 2026, 06:30 UTC. Council round 4 (branch `council/2026-11-11`)** on the weight-space editing papers and replications that
appeared since round 3 (arXiv 2611.09382, 2611.07610, 2611.10866, 2611.06598, 2611.08647, 2611.08070, 2611.09376, 2611.10974,
2609.09054; the Concept Country, aiXiv, LP and scripted-baseline replications). Brief `docs/council/2026-11-11/BRIEF.md`; member
reports in progress; chair synthesis and any amendment F8 to follow before any G3 outcome is read.

**11 Nov 2026, 08:00 UTC. Council round 4 closed; amendment F8 appended.** All four members (`docs/council/2026-11-11/round4-*.md`)
and the chair (`SYNTHESIS.md`) find that nothing published since 10 Nov touches the registered arms, Part A controls or F5.5. F8
(end of `docs/PREREG_H1_TF.md`) adds two descriptive, decision-free panels scored only after the Part A ledger is sealed: F8-E
external edits of the pinned base (Astra patch exact replay; LP and difference-of-means reconstructions; the loop family counts as
one component; arXiv 2611.09382 excluded until its direction is released) and F8-F family transfer (Wanda and random gate deletions,
TFTV rank-one, Concept-Country-style rank-one; SWEET deferred), each with frozen field/chart predictions, κ = q/|a|, a CPU pre-check
floor from the null-size payload, and ≤ 0.9 GPU-h scoring. H5 gains a scripted-estimator arm and GPT-6.1 Sol as practical editor;
the ≥ 30 % savings criterion stands. Board A review of F8 requested; Part A does not wait on it. Main at b627cc8; pins match.

**11 Nov 2026, 09:30 UTC.** Board A accepted F8 with conditions (`docs/reviews/council/board-a-f8-2026-11-11.md`): conditions 1–5 and 7
gate F8 scoring (freeze both panels before any G3 exposure; make the F8.4 gate deterministic with a named floor computation; complete
the F8.3 prediction contract; qualify the Astra replay against the pinned base; define cluster-level summaries and correct the
"cannot reject chance" wording; retain citation evidence), condition 6 gates the H5 pilot (reconcile H5_DESIGN/H5_PROTOCOL/HYPOTHESIS,
pin Sol, specify the LP). Part A does not wait. Condition 7 discharged for the two blocked pages by retained excerpts
(`docs/council/2026-11-11/evidence/README.md`); the H5 reconciliation is in flight (`docs/council-r4-record`); the F8 execution
manifest spec (conditions 1–5) is scheduled after the Part A ledger is sealed and before any F8 construction.

**11 Nov 2026, 10:30 UTC.** Round 4 recorded (`docs/council-r4-record` merged): H5 scripted-estimator arm, second editor and per-stream
savings in `docs/HYPOTHESIS.md`, `docs/H5_DESIGN.md`, `docs/H5_PROTOCOL.md` (Board A F8 condition 6, document part); the eighteen
round-4 items in `docs/RESEARCH_NOTES.md`; seven new verified references (2609.09054, 2611.06598, 2611.07610, 2611.08647,
2611.09382, 2611.10866, 2611.10974) with source texts under `paper/refs/sources/`. Known pre-existing failure: `scripts/verify_refs.py`
exits 1 on `fastprobing2026` (YAML/BibTeX author mismatch), present on main before this merge; it blocks paper review and is queued
for repair. Part A staging at 21 of 40.

**11 Nov 2026, 11:30 UTC.** F8 execution specification merged (`docs/design/F8_EXECUTION_SPEC.md`, design document; adopts nothing by
itself). Discharges Board A F8 conditions 1–6 on paper: one combined gate across a single frozen E+F candidate set (per panel, F8-E
could never reach four components); design-time floor preview h* = 0.00494 nats/token (component-weighted median of null-size
half-widths; binding value must come from committed code); explicit chart abstention for every F8 entry (no fitted chart predictor
exists before Part B); corrected binomial wording (chance rejected only at 6/6 resolved clusters); scoring config must be the Part A
default `configs/eval_tf_v1.yaml`, not the WikiText-only file named in F8.5 (to be fixed by the adopting amendment); aiXiv
reconstruction needs thinking-on generations and is unfunded as priced; H5 scripted-arm costs unpriced until the 4096-token tier is
measured. Remaining for F8: manifest build and freeze after the Part A ledger is sealed, Astra replay checks, builder script, two new
edit families. Council round 5 (self-editing, self-localisation, pruning, predict-before-modify) in progress; Part A staging 26/40.

**11 Nov 2026, 13:00 UTC. Council round 5 closed; amendment F9 appended.** Five user threads (math-improving edits, runtime
self-editing, self-localisation, parameter reduction, predict-before-modify) reduce to one receiver-anchored first-order forecast the
programme already has, plus three things it does not yet measure. F9 (end of `docs/PREREG_H1_TF.md`; `docs/council/2026-11-11/SYNTHESIS-r5.md`)
registers, descriptively and after Part A and F8: a hash-before-score forecast log for every non-Part-A edit with E₀ calibration and
30/100-cluster claim thresholds (F9.1–F9.2); a 10/20/50 % Wanda / SparseGPT-transplant / random pruning ladder with rotations, three
native SparseGPT entries and a 24-mask zero-parameter ranking panel (≈ 0.78 GPU-h scoring, construction cap 0.5); a fifth H5 access
flag for measured localisation with verbal localisation as a logged treatment (F9.4); a 32-query test-time LoRA shadow sub-stream
inside G5 (F9.5); an F8-E-only MATH-500 generation endpoint, unpriced until a throughput pilot, prerequisite for any math-editor claim
(F9.6). F5.5 and A5.7 unchanged. Board A's F8-spec review accepted the execution spec with eight conditions; F8a (adopting amendment)
and the F9 review are in flight. Part A staging 27/40. Main at f181f0a; pins match.

**11 Nov 2026, 14:00 UTC.** Board A accepted F9 with eight conditions (`docs/reviews/council/board-a-f9-2026-11-11.md`): authority
and provenance (archive member identities, logging-vs-panel order, F8a overrides, F9's own gate population); executable forecast
and statistics conventions (realised scale, rotation reference, κ notation, degenerate values, MSE decomposition, cluster
representatives; P5 half-widths are not a new floor); leakage control (frozen train / swap-validation / evaluation splits, no
retrospective refits; 30/100 are reporting conventions, not precision evidence); pruning execution and overhead-inclusive funding;
H5 reconciliation with F8a overrides and incumbent-relative forecasts; shadow-stream limits (no ≤ 5 % safety certification or
realised-savings claim from 32 adapters); generation qualification and bounded pilot funding; evidence wording. Member identities
archived in `docs/council/2026-11-11/MEMBERS.md`. F9a (adopting amendment) follows F8a, which is being drafted. Part A staging 29/40.

**11 Nov 2026, 15:00 UTC.** F8a (adopting amendment for the F8 execution rules) merged at 097b106; pins match. The F8a floor payload
(byte-identical copy of the registered null-size inputs, sha256 67da223c…) is committed under `docs/prereg/f8_manifests/floor_payload/`
so the floor can be recomputed independently from a clone. Board A review of F8a and the F9a draft (Board A's eight F9 conditions)
are in flight. Round-5 record merged (e46c825; 53 references verified). Part A staging 30/40.

**11 Nov 2026, 15:30 UTC.** Board A accepted F8a with six conditions (`docs/reviews/council/board-a-f8a-2026-11-11.md`): before the F8
freeze commits (C1/C2) commit the floor payload, code and output and reproduce the floor from a fresh clone; freeze representatives
and counting rules for merged dependency groups; supply the CPU identity verifier, separate shared scoring identity from edit-specific
hashes and record the effective config hash; enforce cumulative caps alongside the per-entry watchdog; scoring requires verified
C1/C2, a passing gate, qualified identities, a funded purchase and a sealed Part A ledger; H5-only items (document reconciliation,
Sol pin, bounded timing measurements, repricing) before the pilot. The floor payload is already committed (11543d3); the verifier,
floor code and freeze manifests are built after the Part A ledger is sealed. F9a draft in flight. Part A staging 30/40.

**11 Nov 2026, 16:00 UTC.** F9a (adopting amendment for the F9 execution rules; Board A's eight F9 conditions) merged; pins match.
Board A review of F9a requested. F8 tooling (floor script, replay identity verifier) in flight. Part A staging 31/40.

**11 Nov 2026, 17:00 UTC.** F8 tooling merged (`scripts/f8_floor.py`, `scripts/f8_replay_verify.py`, 33 tests). Floor recomputed
from a fresh clone by the chair: h* = 0.0049397 nats/token, equal to the spec preview (`docs/prereg/f8_manifests/floor_payload/`).
Round-5 member reports archived in git (e3920b1). H5 documents being reconciled with F9a.5–F9a.6; Board A review of F9a in flight.

**Timestamp note (added at 23:20 UTC, 10 Nov 2026 system clock).** The clock labels on the entries above from "11 Nov 2026,
06:30 UTC" through "17:00 UTC" were written as estimates and are wrong; the work they describe happened between about 21:30 UTC on
10 Nov and 23:20 UTC on 10 Nov 2026. The git commit timestamps (`git log --format='%h %ci %s'`) are authoritative for ordering and
time; the "before any G3 outcome" status of F7–F9a is unaffected (Part A staging was at 35 of 40 and nothing had been scored).

**10 Nov 2026, 23:17 UTC.** Board A accepted F9a with five conditions (`docs/reviews/council/board-a-f9a-2026-11-11.md`): archive
the round-5 reports before F9-L logging starts (done, e3920b1); correct F9-P's connected components before C1 and scoring (the fixed
three-component declaration is rejected; components must come from the dependency graph); freeze cross-population source separation
before any learned readout update; reconcile H5 documents and freeze/reprice G5 prerequisites before H5 execution (reconciliation in
flight); confirm generation thresholds before sampling and reprice before qualification. All 15 drafter choices dispositioned in the
review. Applied at the F8/F9 freeze, after the Part A ledger is sealed.

**10 Nov 2026, 23:50 UTC. Part A staging complete; rotation yield failure found before any scoring.** The `prepare` run
(branch `feat/part-a-prepare`, 4 h 58 min CPU, 73 GB under `data/runs/a5/part_a/`) staged all 80 ladder variants but the registered
A5.3.3 rotation audit excludes 39 of 40 rotation pairs (30/30 lora_sft, 9/10 chain; 535 of 2,534 edited tensors pass): where D_eff
is small relative to the bf16 spacing of W, the spectrum-matched rotation's entries fall below half an ulp and are rounded away, and
bf16 noise lifts the numerical rank beyond the ±1 tolerance. Under A5.3 the rotation contrast needs ≥ 15 resolved pairs, so it is
**unresolved by yield** before any outcome exists; the null-size and power simulations assumed yield 1.0 with audits unavailable
(Board A F7 condition 4). Nothing has been scored and no outcome has been read. Decision: hold all Part A scoring, including the
ladder, until a pre-outcome analysis and amendment proposal (F10, `docs/design/ROTATION_YIELD_2026-11-11.md`, in flight) and
Board A's review settle the rotation construction and whether ladder scoring may precede F10. GPU idle.

**10 Nov 2026, 23:58 UTC.** `feat/part-a-prepare` merged into main (staging commit 01e1326; CLI `--selection/--references/--text-groups`,
per-tensor audit file, `score --manifest`). Conflicts were append-only (A5 appendix union; pins rebuilt on the merged tree as A5.1.2
re-pin (xv)). Priced Part A scoring after exclusions: 81 variants + 19 references = 100 entries ≈ 1.73 GPU-h. Open chair decision
flagged by the staging: for the 8 chain picks with a non-base predecessor the ladder scales the realised increment D because no
per-step Δ is stored (A5.3 says "not by scaling D_t"); referred to the F10 analysis. Scoring remains on hold.

**11 Nov 2026, 00:20 UTC. Rotation yield: diagnosis and chair rulings.** `docs/design/ROTATION_YIELD_2026-11-11.md` (merged ca7ea4b;
CPU only; all 2,534 tensors rebuilt and the 535 registered passes reproduced) shows the realised mismatch is bf16 rounding of W at
the rotation's scale: the rotation's rounding error equals the edit's own (median ratio 1.0003 on 2,188 root tensors), so the
registered realised audit selects edits by effective norm, not by construction quality; the 1e-3 rank tolerance sits inside the bf16
noise floor; the five reconstruction failures are fp32 artefacts (fp64 residual ≤ 4e-15). Remedy yields computed on CPU: iterative
rounding-aware construction 2/14 tensors (and no longer rotations); rotating the raw Δ 16/32 root edits; noise-calibrated tolerances
34/40 but unstable (17–39) with constants; partial rotation 1/40; registered reading 1/40; remedy (d) 40/40. Chair rulings: adopt
(d) as F10 (construction unchanged; exclusion on exact fp64 construction checks, support, bitwise replay, the round-to-nearest bound
and a half-norm guard; realised norm/spectrum/rank reported as diagnostics and a nuisance covariate; two pre-listed subsets reported,
none decisive; no simulation rerun needed at 40/40, slot-masked power run if fewer); accept the realised-increment ladder for the 8
non-root chain picks as the registered ladder with Δ_t substituted by its realised form, flagged; **no Part A entry is scored before
Board A accepts F10** (ladder outcomes would decide A5.5 step 2 and reveal response sizes, making F10 post-outcome under F8a.1(c)).
F10 text in flight.

**11 Nov 2026, 01:00 UTC.** F10 merged (4fe4379). Board A accepted F10 with seven conditions (`docs/reviews/council/board-a-f10-2026-11-11.md`):
before the code merge, freeze methodological clarifications and per-decision-maker exposure attestations, and provide executable
arithmetic, rank-boundary and on-disk replay tests; before the re-stage, freeze execution settings and complete re-pin (xvi) with
pins verified; before scoring, Board A accepts the outcome-free re-stage report; if yield < 40, pass the slot-masked calibration
(failed rotation slots masked, valid ladder slots retained); freeze all F10.5 sensitivities and interpretation limits before scoring;
**all scoring, including ladders and references, is held until the applicable gates pass.** F10 code (tests, dry run on three edits)
in flight.

**11 Nov 2026, 01:40 UTC.** F10a merged (d2a4660): Board A's rank-budget and chain-ladder wording adopted verbatim; F10 thresholds
recorded as fixed before outcomes, not before audits; no-retuning rule; `docs/prereg/f10_attestations.md` with six attestations (code
builder's pending its report; the sensitivity-spec session to be added). Disclosure from the F10a drafter: the 19 "missing" lora_sft
originals have no TF-v1 record, but each has an older fast-tier evaluation reachable through `data/evals/qwen3.5-0.8b/<id>` symlinks
(6 Nov, D3 suite). Chair ruling: those are development records already known to the registration (the D3 fast-tier suite that the
A5 selection was built on; A5.3 treats development outcomes as known); they are not G3/TF-v1 outcomes and do not affect the
pre-outcome status of F10/F10a. Board A to confirm at the re-stage report acceptance.

**11 Nov 2026, 02:10 UTC.** Board A accepted F10a (wording; the D3 development-record ruling confirmed; condition 1 stays open only
for the code builder's attestation) and the F10.5 sensitivity specification (`docs/design/F10_SENSITIVITY_SPEC.md`, commit e577c67,
subset hashes 3da5f44f… (30 pairs) and fb1c34e6… (34 pairs)) with conditions: adopt the spec by dated A5 note and exact hash
confirming its operational choices (v4 differences = the five logged moments; penalty 1 = λ = 1 with unpenalised intercept; two
tertile sets); implement, test and pin it before scoring; report the sign table as 3×2 and units as nats per valid next token
(`docs/reviews/council/board-a-f10a-2026-11-11.md`). Scoring remains held. Waiting on the F10 code (condition 2) and the slot-mask
path (condition 5).

**11 Nov 2026, 02:45 UTC.** Merged: `feat/a5-slot-mask` (`--slot-mask` for the F10.6 slot-masked calibration; 18 simulator tests;
A5.1.2 re-pin (xvi), pins match) and `prereg/f10b` (F10.5 specification adopted by hash with Board A's operational confirmations and
reporting corrections). The F10 exclusion-rule implementation (commit e177ce6 on `feat/f10-rotation-audit`: F10.3 checks, F10.4
diagnostics, a533/f10 audit-rule switch, realised-increment ladder flag, dry-run script) is under Board A review for condition 2
before merge; the code builder's attestation and the Part A analysis code (A5.4/A5.5 outputs plus F10.5 reports, synthetic-tested)
are in flight. Scoring held.

**11 Nov 2026, 03:00 UTC. F10 code dry run: two rulings.** Commit e177ce6 (`feat/f10-rotation-audit`; 14 new tests; full suite 839
passed) implements F10.3–F10.4. Dry run on three staged edits (scratch only, nothing scored): `00f8111c` (chain) and `0a841d40`
(lora_sft) pass F10 with construction checks far inside 1e-5 and byte-identical rebuilt deltas; `0027efdf` (chain, predecessor
`b7edbb9f`) is **excluded** under F10.3(iii): the predecessor holds exact zeros where the base is about ±1e-4, the rotated state there
is about 1e-10, and the scorer's chain convention bf16(base + (state − base)) cannot carry it in fp32 (19 entries, 13 of 32 tensors;
reason `replay_failed`, check `scorer_roundtrip`). Chair ruling 1: the exclusion stands as implemented; changing the chain storage or
apply convention is outside F10.3 and would be a new post-audit amendment; all 8 non-root chain picks are at risk, so the F10
yield may fall below 40 and the F10.6 slot-masked calibration (implemented, re-pin xvi) then applies. The rotation-yield note's 40/40
covered only the measured items (F10 table), so this is not a contradiction of the registration. Chair ruling 2: check (iv) must
match the F10/F10a text (½ ulp of the rounded value W + R_eff, which the draft states, with a slack of one fp32 spacing at W + R);
the builder used a slack of 2·ulp_fp32(|W| + |R|) and reports the literal half-ulp(W) count separately; Board A's code review (in
flight) rules whether the factor-two slack is accepted; if so it enters as a disclosed F10c note with the measured justification
before the re-stage, otherwise the code is corrected before merge. Re-stage price measured: about 2 h 11 min to 3 h 32 min CPU for
the 40 rotations (4 threads), +35.8 GB; `prepare` needs a rotation-only mode that preserves the staged ladders and the original
manifest (Board A condition 3), to be added with Board A's code conditions. Attestation 5 completed from the builder's report.

**11 Nov 2026, 03:20 UTC.** Board A REJECTED the F10 implementation e177ce6 under condition 2 with eight corrections
(`docs/reviews/council/board-a-f10-code-2026-11-11.md`): F10a near-cut rank budget; exactly the registered one-fp32-spacing slack;
pooled half-norm guard; bit-pattern replay against reloaded files with four threads; fail-closed missing/non-finite audits and D
support; the complete a533 verdict as a diagnostic; pooled F10.4 diagnostics with deliberate-failure tests; an audit-only
(rotation-only) preparation path with no evaluation reads. Corrections in flight on the same branch (merging main first). Nothing
merged; scoring held.

**11 Nov 2026, 03:45 UTC.** Part A analysis code committed (`feat/f10-sensitivity-reports`, 91d976b; `src/wsp/eval/part_a_analysis.py`,
`scripts/part_a_analyse.py`, 53 synthetic tests; subset and tertile hashes reproduce from the manifest; fail-closed CLI). Under Board A
review. Items the adopting note must rule on, from the builder's report: Part B contrasts are supplied as precomputed intervals via
`--part-b` (funding unresolved if absent); seven A5.5 step-1 validity checks come from a `--validity` evidence file (gate unresolved if
absent); resolution seed PCG64(20261107) with one shared [4000, 32] index array (not the simulation's SeedSequence stream); a
flat-step tie-break in the λ = 1 solver (accepted only if it reduces the gradient norm; matches sklearn C=1 to 1e-6); the sealed-ledger
and fields formats (`a5-part-a-sealed-ledger-v1`); `audit_missing` as a fail-closed exclusion code; a wrong subset hash stops the run
rather than voiding the subset; frozen staged correlations printed, not recomputed; edit `057dcfbedea3251f` sits on ln ρ cut 1 and
will be flagged `cell_drift` on any downward re-stage drift (never moved).

**11 Nov 2026, 04:00 UTC.** Board A REJECTED the Part A analysis code 91d976b under condition 6 with six corrections
(`docs/reviews/council/board-a-part-a-analysis-2026-11-11.md`): validity-first A5.5 ordering with Part B reporting preserved;
runtime verification of the adopted specification hash; frozen window-order validation and complete re-stage audit evidence;
complete ledgers, 3×2 sign tables and fit diagnostics; conditional masked-calibration disclosure; re-test and pin before scoring.
Corrections in flight on the same branch. Both code branches (F10 rule, analysis) are now in their second round; scoring held.

**11 Nov 2026, 04:30 UTC.** Corrected Part A analysis code committed (2b03bda; 138 tests, full suite 968 passed): validity-first A5.5
ordering with Part B always reported; runtime spec-hash check (`--spec-sha256`); window order and per-window token hashes validated
against `text_groups.json`; complete re-stage evidence per pair (`audit_missing` fail-closed); complete ledgers, 3×2 sign tables,
fit diagnostics; F10.6 masked-calibration disclosure (refused at 40/40, required below). Under Board A re-review. Pin list proposed:
`src/wsp/eval/part_a_analysis.py`, `scripts/part_a_analyse.py`, `tests/test_part_a_analysis.py`, `docs/design/F10_SENSITIVITY_SPEC.md`.
