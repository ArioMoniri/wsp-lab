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
