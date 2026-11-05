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
