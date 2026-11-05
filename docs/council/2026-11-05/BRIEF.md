# Council brief — 5 Nov 2026 (read first; every council member starts here)

## Goal of the project
Test whether a dedicated weight-space "world model" (SSM / JEPA / trajectory transformer) can predict, from the weights
and the edit alone, how an LLM's benchmark scores change after each weight edit (single edits and edit trajectories),
on small models (Qwen3.5-0.8B now; 4B/27B later), including normal vs abliterated vs Heretic edits, and later act as a
critic for an LLM editor (H5). Hypotheses: docs/HYPOTHESIS.md (H1–H6). Trigger: the MIT CSAIL blog "Sparks of
Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, 2 Nov 2026, https://rikiyat.github.io/blog/weight-space-programming/;
our extraction in docs/H5_RESEARCH.md and docs/RESEARCH_NOTES.md). Method inspiration: JEPAWG — *Weight-Space Physics*
(arXiv 2607.07127) and *Bifurcations from Weight Space* (arXiv 2610.30847); W2T canonical LoRA tokens (arXiv 2603.15990).

## What exists (all in this repo; read the files, do not guess)
- Zoo: 1,235 edits of Qwen3.5-0.8B in six families (random_dir, abliteration = projected refusal-direction ablation,
  heretic_ara, lora_sft, task_vector, chain = multi-step trajectories); provenance per entry (docs/ZOO_IDENTITY.md,
  src/wsp/zoo/). Deltas are stored as effective bf16 differences (DECISIONS D1).
- Features v4 (src/wsp/features/): per-tensor statistics (1,242 dims) + singular spectrum summaries (136 dims),
  pooled per architecture role (pool.py); W2T-style canonicalisation for LoRA.
- Evaluation: fast tier = gsm8k 50 / math500 25 / ifeval 25 items sampled at T 0.6 with 1,024 tokens, mmlu_pro 100 and
  hellaswag 100 likelihood-scored, keyword refusal 50, over-refusal 50, KL-to-base on 50 teacher-forced continuations,
  wikitext ppl 8 windows; 300-entry D3 subset fully evaluated + 89 chain predecessors; same-device references.
- Measurement facts (docs/RESULTS.md): same-host same-seed repeat is bit-identical; a different sampling seed moves
  gsm8k/math500/ifeval by 0.03–0.06 mean (rank ρ 0.1–0.5 against themselves); mmlu_pro, hellaswag, KL, ppl are
  exactly repeatable. Cross-device gaps equal seed noise. Only the likelihood/teacher-forced metrics are admissible
  targets at this tier (Board A rule: docs/reviews/H5/board-a-prereg-2026-11-04.md).
- Predictors (src/wsp/predictor/): M1 pooled-feature MLP/attention, M1-matched control, M2 Mamba-2 over the edit
  sequence, M2-matched stateless control, M3 JEPA on cumulative-delta features, M5 trajectory transformer, M6
  (see docs/MODEL_CANDIDATES.md and docs/ARCHITECTURES.md); baselines StatNN (gradient-boosted trees on the same
  pooled stats), delta-norm, fold-mean, descriptor-only, device-only. Protocol v2: nested selection, train-only
  preprocessing, per-family centred metrics, ancestry-cluster paired bootstrap, 3 full seeds (docs/DECISIONS.md P4 addendum).
- Results so far (docs/reviews/P4/*, data/runs/p4/d3v2-*): Spearman on mmlu_pro / hellaswag, 3-seed mean —
  trajectory split: StatNN 0.57 / 0.65, M1-matched 0.45 / 0.46, M2 0.35 / 0.32, M3 0.15 / 0.11; seed split similar;
  held-out-family split: everything in [−0.2, 0.3] including StatNN (−0.02 / 0.11); delta-norm alone ≈ 0.2.
  Pre-registered H1 bar (ρ ≥ 0.6 per benchmark and beating the baselines): not met by any model. H2 (history helps)
  and H3 (held-out family transfer): no evidence.
- Heretic ARA family: first rebuild selected trials with no KL target (destroyed models; flagged); corrected rebuild
  (KL target 0.5, feasible-Pareto selection, non-row-preserving variant) running on Lambda.
- H5 editor harness (src/wsp/editor/): merged, production disabled; pilot preconditions in GitHub issue 31;
  protocol docs/H5_PROTOCOL.md, design docs/H5_DESIGN.md, pre-registration by Board A.
- Reviews so far: docs/reviews/** (Boards A, B, C). Lessons: harness/lessons.md.

## The question for this council
Given these results, what is wrong with the approach, and what would make the goal achievable? Challenge the premise
(weight-only prediction of noisy downstream scores), the zoo (are the edit families informative?), the features (do
pooled statistics discard the signal the blog exploited: specific low-rank directions in specific layers?), the
targets (are we predicting noise?), the models (is an SSM over edit sequences the right inductive bias when
StatNN wins?), and the evaluation protocol. Read the primary sources yourself (blog page; arXiv abs/HTML of
2607.07127, 2610.30847, 2603.15990; any public code they link) and cite exactly what you verified.
Output rules: no invented numbers; every claim about our data points to a file; every external claim to a URL or
arXiv id you fetched; proposals must be concrete (what to build, what to measure, expected effect, cost in GPU-hours
from our measured timings: fast-tier evaluation 7.5 min/entry on this GPU, 3.1 min effective on an H100 with three
workers; feature extraction 66 s/entry; predictor run 1–10 min).
