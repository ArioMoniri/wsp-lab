# Council brief, round 13, 20 Nov 2026: can (a) be fixed? Kocabay's Weight-Space Reading, WSP, and a memory/SSM world model

## The user's request (paraphrased)

Check the weight-space editing research by Şuayp Talha Kocabay with the council, and work out how to solve our (a)
failure. The user's premises:
- trajectory prediction with world models is known to be possible;
- the world model may need something like a Mamba memory layer.

## The papers (chair's fetch, 20 Nov; verify and cite yourself)

**The arXiv ID the user gave is a different paper.** `arxiv.org/abs/2609.25541` is "A JEPA Recipe for Tabular
Foundation Models" (Jeon, Cho, Suh). It has nothing to do with Kocabay or with weight editing.

**P1. Weight-Space Reading** (Kocabay, sole author, Poindexter Labs; alphaXiv, submitted 13 Nov 2026; arXiv pending).
- **Sources:**
  - https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models
  - code: https://github.com/suayptalha/wsr (the paper's results are not in the repo)
- **Task:** LLM reader agents, given read-only weights and a CPU sandbox but no forward passes, predict a ternary
  **direction** (up, down or same) for five behaviours under surgical edits.
  - Behaviours: thinking length, MATH, MMLU-Pro, sycophancy, refusal.
  - Edits: block and vocabulary-row scaling, orthogonal removal, norm-matched noise, norm-gain, and no-ops.
  - Models: Qwen3.8-27B, Qwen3.5-9B, Muse-30B.
- **Headline:** 27B reading its own weights on subtle edits scores balanced accuracy 0.730, against 0.578 for the
  same reader given text only (+0.152, CI [+0.056, +0.256]). Other pairs are weaker or null (9B→9B 0.554, 27B→9B
  0.628 vs 0.648).
- **Muse:** none of the 40 Muse edits changed behaviour, attributed partly to post-sublayer RMSNorm, yet readers
  predicted a change 53–88 % of the time.
- **What it is not:** no learned world model, no memory or SSM layer, no magnitude or trajectory prediction, no
  pooled or statistical baseline, and no use of predictions to choose edits.

**P2. Exploring Weight-Space Programming** (Hu et al., Shanghai AI Lab, Tongji, Fudan; arXiv 2611.09382, 6 Nov 2026;
**Kocabay is not an author**).
- **Setup:**
  - Editor: an LLM agent (GPT-6 Astra) with 50 forward-tool calls per search and no gradients.
  - Targets: Qwen3.5-0.8B and 9B, with dev and held-out splits.
- **Results:**
  - 0.8B held-out: MATH-500 8.35 → 37.80, IFEval 21.40 → 58.12, from very low base scores.
  - 9B MATH held-out: 97.70 → 97.35, despite +29.69 on dev.
  - 6 of 16 trajectories kept the original weights.
  - Dev selection is adaptive and uncorrected.

## Our evidence on (a) (read the files)

**Stage W2** (`docs/reviews/wme/wme-stagew2-2026-11-18.md`). The nested WM **ties** the pooled same-path-type curve:

| Setting | WM E | Pooled E |
|---|---|---|
| zero-shot | 0.768 | 0.771 |
| k = 2 | 0.881 | 0.881 (d exactly 0) |

The WM, and the pooled curve, **beat by a wide margin** every generic trajectory model:

| Comparator | E (zero-shot) | E (k = 2) |
|---|---|---|
| persistence | 4.43 | 3.17 |
| LC-PFN | — | 2.64 |
| lc_ensemble | — | 3.18 |
| static | 1.38 | 1.63 |
| statnn | 0.845 | 1.019 |
| unnested WM | 0.983 | 1.370 |

So trajectories **are** predictable from path type. What fails is extracting **edit-specific** information beyond path
type.

**Stage W** (`docs/reviews/wme/wme-stagew-2026-11-18.md`). Also read the pilot reports and
`docs/council/2026-11-17/SYNTHESIS-r11.md`.

**WM-E design** (`docs/PREREG_WME.md`, FROZEN):
- R0/R1 access tiers;
- the WM-CP canonical-path model;
- the frozen claims: H-E1 sycophancy and H-E2 WM vs StatNN search.

(a) is registered as a negative result. The frozen 9B study must not be changed. Any new (a) work is a **separate
exploratory stage** with its own plan, and confirmatory data must be fresh.

**Related live work: study M** (`docs/council/2026-11-19/SYNTHESIS-r12.md`, `docs/council/2026-11-20/RULING-wmm-gate0.md`).
- Gemma-3 post-norms cancel uniform writer rescales. This is the same mechanism P1 reports for Muse.

## Questions

1. **Do P1 and P2 show (a) is possible?** What exactly do they demonstrate? Compare it with our (a): direction
   classification versus calibrated trajectory regression against a path-type baseline. Is the user's premise
   correct?
2. **Diagnose the tie.** Is the W2 tie a model-capacity problem, which more memory would fix, or an **identifiability
   or data** problem?
   - Data side: low within-path-type variance in the bank, too few runs per type, or features that carry no
     edit-specific signal.
   - Propose a cheap CPU-only analysis on existing exposed W/W2 data (no new GPU) that decides this, e.g. a variance
     decomposition and the irreducible-error ceiling of any edit-specific predictor.
3. **The Mamba/SSM memory idea.**
   - What would a memory layer add over the k-shot nested WM? Options: long edit histories, cross-path transfer, an
     in-context learner over many trajectories (LC-PFN-style, which already lost).
   - Under what data conditions could it beat the pooled curve? Specify it concretely, or reject it.
   - Alternatives: P1-style weight readers as features; an LLM reader as the WM; richer R1 features; a different
     target quantity.
4. **A re-scoped, powerable (a).** For example:
   - predict the *direction* of edit effects, P1-style, against a path-type-majority baseline;
   - predict *within-type ranking*;
   - predict on edit families with real within-type spread.

   Give the baseline that must be beaten, nulls, power, and cost. Say whether to run it on 9B exposed data, on
   MedGemma (study M), or on new runs.
5. **P1/P2 for (b).**
   - Should a WSR-style reader become an arm or feature? We already have WSR-style reader arms in H-E2.
   - Does P2's agentic editing on 0.8B transfer?
   - P2's 9B held-out −0.35 matches our headroom findings. What does that imply?

## Rules

- **Evidence.** Cite fetched URLs for external claims and repo files for our data. No invented numbers. Record fetch
  failures.
- **Format.** At most 1,500 words. Open with an 8-line BLUF. End with the table
  "decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback".
- **Output.** Write only `docs/council/2026-11-20/round13-<seat>.md` in the worktree
  `/home/ubuntu/projects/wsp-lab/.worktrees/council/2026-11-20-r13`.
- **Prohibited.**
  - commits;
  - GPU use;
  - downloads of models;
  - writes under `data/` (reading exposed W/W2 artefacts under `data/wme/` is allowed);
  - opening `data/evals` or sealed splits;
  - editing `docs/PREREG_*`;
  - touching running jobs or Lambda;
  - secrets, IPs or hostnames.
