# Council round 13 synthesis, 20 Nov 2026: can (a) be fixed? Weight-Space Reading, WSP, and a memory/SSM world model

**Seats and reports.**

| Seat | Model | Report |
|---|---|---|
| methods / diagnosis | Claude | `round13-methods.md` (CPU analysis of exposed W2 data; scripts outside the repo) |
| world-model architect | Claude | `round13-world-model.md` |
| challenger | Gemini 3.8 Flash | `round13-challenger.md` |

## BLUF

1. **The papers do not show that (a) works.**
   - **P1** (Kocabay, *Weight-Space Reading*) shows that LLM readers with weight access predict the ternary
     *direction* of behaviour change. The only solid cell is 27B reading its own weights on subtle edits: 0.730 vs
     0.578 for the text-only reader.
   - P1 has no magnitudes, no trajectories, no statistical baseline, and few labels.
   - **P2** (Hu et al., *Exploring WSP*) is agentic forward search, not prediction.
   - Its 9B held-out MATH result (−0.35 after +29.69 on dev) matches our headroom findings.
2. **The user's premise is partly right, from our own data.** Trajectories *are* predictable from path type. The WM
   and the pooled curve beat every generic forecaster by a wide margin (`wme-stagew2-2026-11-18.md`). What fails is
   the *edit-specific* part.
3. **Diagnosis: identifiability and data, not capacity.** All three seats agree.
   - The 9B W2 bank varies along essentially one axis, LoRA dose. The log weight-change norm tracks learning rate at
     r = 0.996 (methods).
   - At k = 2 the WM's edit term was shrunk to 0 in all 15 folds.
   - Even an oracle with SE-level noise beats the pooled curve only narrowly.
4. **Mamba/SSM memory layer: rejected for now** (unanimous).
   - Paths have 6 points, so k-shot context is 0–2.
   - The in-context learner LC-PFN already lost (E 2.638 vs 0.881).
   - History memory would break the frozen C2-D decomposition invariance.
   - No published SSM evidence for this task was found.
   - The re-entry conditions are below.
5. **New mandatory baseline.** Any future (a) claim must beat **pooled curve + dose**, not the pooled curve alone.
6. **Re-scoped (a).** The new target is *beyond-dose* within-type prediction: ranking and sign at matched norm.
   - It runs on a bank built to vary in **direction**, in study M (MedGemma, local, $0), after Gates 0b and 1.
   - It needs at least 30 direction units, under a pre-registered PREREG_WMM claim.
7. **No new 9B spending on (a).** Ruling Q4 stands; the chair recommends against the $84 dose confirmation (R3).
8. **(b).** H-E2 stays frozen. Study M adds a descriptive WSR-style reader arm, scored on false "change" calls for
   edits that the post-norm cancels.

## Rulings

**R1. Reading of P1/P2.** Both are recorded as related work.
- **P1** supports "weight access carries some directional signal for a strong self-reader". It does not support
  calibrated trajectory prediction.
- **P2** supports "agentic search can find dev gains on weak models". Its 9B held-out reversal is independent
  corroboration of our headroom result.
- Neither changes the frozen claims.

**R2. Diagnosis of the W2 tie: identifiability, not capacity.**

Methods variance decomposition, in SE² units:

| Component | Variance |
|---|---|
| between path type | 1.85 |
| within type | 0.93 |
| item-level noise | 0.34 |

- The accuracy signal is small (reliability about 0.3).
- Truncation carries most of the within-type spread.
- Oracle E is about 0.505 without noise and 0.690 with SE-level noise.
- The world-model seat notes that E for a perfect predictor under pure Gaussian noise is about √(2/π) ≈ 0.80, close
  to the pooled curve's E. This is a theoretical bound only.
- **Action:** measure the empirical noise floor from the `base`/`base_rerun` records (CPU, $0). Record it as a
  descriptive addendum to the W2 report.

**R3. The post-hoc dose model.** Pooled curve plus a per-type log-norm slope gives E 0.678 vs 0.771 (p 0.021) at
k = 0 and 0.766 vs 0.881 (p 0.003) at k = 2. These are **post hoc** results:
- six specifications were tried after seeing the data;
- the model loses on stage W.

The weight norm tracks learning rate at r = 0.996, and learning rate is a recipe setting, so this mostly recovers
recipe information, not weight-space understanding (world-model seat).

*Ruling:*
- This is **not evidence for (a)**.
- It becomes the mandatory baseline (BLUF item 5).
- Confirming it on fresh 9B runs (about 17 GPU-h, about $84, by the methods estimate) is offered to the user. The chair
  recommends **no**: it would confirm a recipe-dose effect, not edit-specific prediction, and it reverses Q4.

**R4. Mamba/SSM memory.** Rejected for (a) now. Re-entry needs all of these on development data:
- (i) paths long enough for context to matter: at least 10 observed points, or edit histories deeper than 2;
- (ii) a bank with beyond-dose variance that a ridge on direction features detects first;
- (iii) a design that keeps C2-D invariance, or a dated amendment that justifies dropping it;
- (iv) a matched-capacity non-SSM learner (ridge/GP, or an attention ICL) as comparator.

**R5. The re-scoped (a), as an exploratory claim in PREREG_WMM.** It is pre-declared before any study M Gate 1 data
on directions.

- **Bank:** at least 30 direction units at **matched norm** within type. Candidate families:
  - task-vector subspaces (per-layer-group slices of τ);
  - random and sham directions at matched norm;
  - LoRA seeds and shards.
- **Targets:**
  - within-type Spearman rank of effect;
  - sign of effect, scored at run level;
  - truncation-adjusted generated accuracy first.
- **Baselines:** pooled + dose (mandatory), and a type-majority sign.
- **Nulls:**
  - within-dose-stratum permutation;
  - sham directions;
  - shuffled features.
- **Power** (methods, Fisher-z): ρ 0.5 needs about 29 units; ρ 0.6 needs about 19.
- **Features, at R0/R1 only:**
  - per-layer J1 (already on disk for 9B, used there only to freeze specs);
  - alignment of the direction with τ;
  - Fisher-weighted norm, if a GPU pass is affordable under M4.
- **Kill rule:** if the ridge on direction features shows no beyond-dose signal on study M dev, (a) closes on study M
  as well.

The direction-only P1-style target is **not** powerable on the 9B bank, because the type-majority already scores 0.887
(methods).

**R6. P1/P2 for (b).**
- H-E2 is unchanged; its WSR-style reader arm already exists.
- Study M gets a **descriptive** WSR-style reader after Gates 0b and 1. Its main metric is the false-change rate on
  post-norm-cancelled edits, which is P1's Muse failure mode. Its LLM token cost is priced before it runs.
- P2-style forward search counts only as a charged R2 oracle arm, never as a WM feature.

## Decision table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| P1/P2 = related work; no claim change | Direction-only, small N; search not prediction | — | 0 | — |
| Tie = identifiability; measure noise floor | 1-axis bank; shrink to 0; oracle ceiling | base/base_rerun noise floor (CPU) | 0, $0 | If floor ≥ pooled E − 0.05: close (a) on 9B for good |
| Pooled + dose = mandatory baseline | Dose = recipe information | applied to every future (a) | 0 | — |
| No 9B dose confirmation (recommended) | Confirms a recipe effect; reverses Q4 | user decision | would be ~17 GPU-h, ~$84 | — |
| Mamba/SSM rejected now | Short paths; LC-PFN lost; C2-D | re-entry conditions R4 (i)–(iv) | 0 | Revisit only if all four hold |
| Beyond-dose (a) in study M | Direction-varied bank at matched norm | PREREG_WMM, ≥ 30 units, nulls | local, $0 (priced from Gate 1 timing) | No ridge signal on dev: close (a) |
| WSR-style reader, descriptive, study M | Tests P1's Muse failure on Gemma post-norms | after Gates 0b/1 | LLM tokens, priced first | — |
