# Council brief, round 15, 23 Nov 2026: world-model-predicted pruning, and predicted targeted unlearning

## The user's request (paraphrased)

Two test ideas for the world model:

1. **Prunability prediction.** Use a world model to predict, for each model, which weights can be **deleted** with
   little loss of overall performance.
2. **Targeted unlearning.** Predict which weights to **change or delete** so a model forgets specific data, for example
   facts about a person X and that person's medical data, while overall performance barely changes.

## Scope and safety boundary (binding)

- **No real personal data, ever.** Any "person X" is a **fictitious, synthetic** individual (TOFU-style), with
  synthetic medical records. Nothing scraped; no real patients; no attempt to extract real people's data from any
  model.
- **Defensive framing.** Unlearning here serves privacy (right-to-be-forgotten). Report extraction or relearning
  attacks only as *evaluation* of forgetting, at the level needed for defence.
- **HAI-DEF (MedGemma).** Edited MedGemma weights stay internal; no harmful-refusal targeting; outputs that could be
  harmful stay under `data/wmm/`; reports carry aggregates. Insertion of synthetic people by fine-tuning is allowed
  only on local copies. Qwen (Apache-2.0) is preferred for anything that needs inserting facts.
- Frozen studies are untouched: C1 (Qwen3.5-9B, running on Lambda) and `PREREG_WME` / `PREREG_WMM`. Any new work is a
  **separate exploratory pre-registration**.

## What we have (cite files)

**Our evidence on "world model" prediction of edit effects.**
- (a) trajectory prediction is a registered negative. The WM ties the pooled path-type curve
  (`docs/reviews/wme/wme-stagew2-2026-11-18.md`).
- The diagnosis is identifiability, not capacity. Pooled curve + dose is the mandatory baseline
  (`docs/council/2026-11-20/SYNTHESIS-r13.md`).
- Study M′ on MedGemma (`docs/reviews/wmm/wmm-mprime-final-2026-11-23.md`): edits work but are not predicted beyond
  dose; the edit-prediction test (M-A) was not run because of the cap.

**Prior pruning plan, never executed.** The F9-P ladder (`docs/HYPOTHESIS.md` around line 231; Board A reviews
`docs/reviews/council/board-a-f9-2026-11-11.md`, `board-a-f9a-2026-11-11.md`; `docs/RESEARCH_NOTES.md` items 15–20):
- Wanda, transplanted SparseGPT and count-matched random masks at 10/20/50 %;
- a 24-mask ranking panel scored by Kendall τ of realised WikiText change against a + q, q_A and −‖W⊙M‖².
- Note: Wanda's score equals our q_A up to a factor (Board A F9a.8).

**Tooling.**
- Fisher / K-FAC sensitivity fields with five objectives (`src/wsp/features/response_field.py`, `src/wsp/wm/c1/field.py`).
- Per-layer gradient alignment J1, R0/R1 access tiers (`src/wsp/wm/access.py`).
- Exact inert and null checks, canaries, provenance records (`src/wsp/wm/wmm/provenance.py`, `canary.py`).
- Gemma-3 post-norms cancel uniform writer rescales (`docs/reviews/wmm/wmm-gate0-2026-11-19.md`), so "delete" and
  "rescale" are not the same operation there.

**Compute.** Local 32 GB GPU at $0 (shared, ≤ 16 GB per job), Lambda for paid work only after the user approves a
price. C1 has priority.

## Questions

1. **Literature.** What is the state of:
   - pruning-saliency prediction (magnitude, Wanda, SparseGPT/OBS, Fisher, learned predictors such as PPF) and how
     well saliency scores *predict* realised damage;
   - knowledge localisation and editing for unlearning (ROME/MEMIT, causal tracing, "does localisation inform
     editing?"), gradient-based unlearning (GA, NPO, RMU, task-vector negation), and benchmarks (TOFU, MUSE, WMDP);
   - evaluation of *real* forgetting versus suppression (relearning attacks, probing, membership inference).

   Cite fetched URLs only; record fetch failures.
2. **Is either idea a world-model problem at all?** Saliency scores are first-order predictors of deletion damage.
   What does a learned world model add over Wanda/SparseGPT/Fisher, given our r13 finding that edit-specific
   prediction beyond dose did not identify? State the baseline a WM must beat.
3. **Prunability study.** Propose one concrete, powerable study: model(s), mask families, ground truth (realised
   loss/accuracy change), predictors (Wanda, OBS, Fisher, our fields, a learned ranker), metric (rank correlation
   within sparsity, top-k precision), nulls, power, cost. Which model: Qwen3.5-0.8B, 9B, MedGemma?
4. **Unlearning study.** Propose one concrete design:
   - synthetic persons inserted into a model (or TOFU's fictitious authors, if licence and model fit allow);
   - **predicted localisation** (which layers/units/weights) versus methods that need no localisation (GA/NPO/RMU,
     task-vector negation);
   - forget quality (QA accuracy on the forget set, paraphrases, relearning resistance, probe-based recovery) and
     retain quality (neighbour persons, general benchmarks);
   - the key question: does a *predicted* weight set beat random and saliency-matched sets at equal edit size?
5. **Order, cost and kill rules.** C1 comes first. What can run on CPU or the local GPU at $0? What would need
   Lambda? Give kill rules.
6. **Ethics and licence.** Synthetic-only data, HAI-DEF, Qwen licence, and whether unlearning claims risk overstating
   privacy guarantees.

## Rules

- **Evidence:** fetched URLs only for external claims; repo files for our data. No invented numbers.
- **Format:** at most 1,500 words; an 8-line BLUF; end with the table
  "decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback".
- **Output:** write ONLY `docs/council/2026-11-23/round15-<seat>.md` in
  `/home/ubuntu/projects/wsp-lab/.worktrees/council/2026-11-23-r15`.
- **Prohibited:** commits; GPU use; model downloads; writes under `data/`; opening `data/evals` or sealed splits;
  editing `docs/PREREG_*`; touching running jobs, tmux sessions or Lambda; secrets, IPs or hostnames; any real
  personal data.
