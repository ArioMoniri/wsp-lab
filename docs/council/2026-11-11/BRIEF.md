# Council brief — 11 Nov 2026, round 4 (read first)

## Why this round sits

The user asked on 11 November to check the weight-space editing papers and articles that appeared since round 3
(`docs/council/2026-11-10/BRIEF.md`, verified 10 Nov) and to put them to the council. The registered programme is now
**frozen** (Board A accepted the hash freeze at a638efa on 10 Nov; F6 and the completed G2 fields were accepted on
11 Nov; `docs/reviews/council/board-a-*.md`). Nothing from G3 (Part A) has been scored: the Part A variants are being
staged (`feat/part-a-prepare`), and the power/yield simulation with the registered sign predictions
(`data/runs/a5/power/fields.json`, amendment F7 on the donor pool) is running. Any change this round proposes therefore
enters only as a **dated amendment to appendix A5 before any G3 outcome is read**, and must not touch the frozen
registered arms, their decision rules or the Part A selection.

## State of results (unchanged since round 3 unless noted; numbers live in the files named)

| Item | Result | File |
|---|---|---|
| Full-set predictor suite (342 rows / 278 clusters) | StatNN 0.56 / 0.66 (seed split), 0.55 / 0.64 (trajectory); descriptor-only 0.51 / 0.61; best neural 0.45 / 0.50 | `docs/reviews/P4/d3v2f-summary-2026-11-08.md` |
| Teacher-forced tier | G1 14/14 bitwise; MMLU gold log-prob unqualified; H1w WikiText inconclusive (θ̂ 0.007, power 0.73 / 0.60) | `docs/reviews/P5/tf-v1-requalification-2026-11-06.md`, `h1w-gate-2026-11-06.md` |
| Registered programme (frozen) | base-anchored sensitivity field vs augmented StatNN; chart secondary; Part A rotation + ladder on 40 edits (30 lora_sft + 10 chain); WikiText first; go / screen / no-go / unresolved | `docs/PREREG_H1_TF.md` appendix A5, amendments F1–F7 |
| G2 (new) | fields for 5 objectives on the base, projectors, probe cache; registered predecessor audit infeasible at 394 s/row (error 0.336) → F6 `chain_exact`: exact predecessor gradients for the 10 chain picks; accepted by Board A | `docs/reviews/council/board-a-f6-2026-11-11.md`, `board-a-g2-2026-11-11.md`, `data/runs/g2/` |
| Power/yield inputs (new) | 141 donors with field sign predictions: 10 chain_exact, 51 base-exact roots, 80 flagged base-approximation-unvalidated; rotation signs 43+ / 98−; F7: registered run on the 61 exact donors, 141-donor run as sensitivity | `docs/prereg/a5_manifests/power/fields_manifest.json` |
| 4B scale diagnostic (new, F5.3) | 20 Gaussian edits ρ=0.005 on Qwen3.5-4B: effects at the resolution floor; H100↔H0 pairing FAIL (A3 0.662 > 0.1); diagnostic closed, no gate | `docs/reviews/scale/scale4b-diagnostic-2026-11-10.md` |
| Null-size simulation | PASS (ordered go rate 0.018; sensitivities 0.016, 0.0); one sparse cell (LoRA low tertile, 1 donor) | `docs/prereg/a5_manifests/nullsize/` |

## What appeared since round 3 (verified 11 Nov; cite the URL you read, not this digest)

### Papers (arXiv ids not in `docs/RESEARCH_NOTES.md` before today)

1. **Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations** (arXiv 2611.09382, Hu et al.,
   Shanghai AI Lab / Tongji / Fudan, 6 Nov). 16 Astra editing trajectories on **Qwen3.5-0.8B and 9B**, seven benchmarks,
   50 forward requests per search, edits confined to transformer blocks, held-out evaluation separate from development.
   0.8B held-out: MATH-500 8.35 → 37.80 (+29.45) from **1,023 changed bf16 values** (layer-23 MLP down-projection,
   column 507, W' = W − u e₅₀₇ᵀ); IFEval +36.72; SciBench +17.11; GPQA +4.42; SciCode 0 → 0. Three independent MATH
   searches on the same model: +29.45 (1,023 values), +5.00 (1,037,572), +27.30 (1,981,083). 9B: IFEval 89.30 → 91.14,
   SciBench 72.46 → 75.49, MATH candidate +29.69 on development landed at 97.35 vs 97.70 held-out. MedXpertQA +17.19
   on development → +0.24 held-out. **6 of 16 trajectories kept the original weights.** Development results are
   adaptively selected with no multiplicity correction. The GitHub repository holds only the PDF; the direction u is not
   released. https://arxiv.org/abs/2611.09382 , https://wsp-scope.github.io/
2. **Embedding-Space Fingerprinting of LoRA Fine-Tunes** (arXiv 2611.07610, Rane, 5 Nov). **Qwen3.5-0.8B**, 36 LoRA
   fine-tunes over 18 public datasets in five domains; leave-one-dataset-out protocol frozen before any result.
   Domain recovery: white-box weight-delta statistics 42 %, gray-box activations on a fixed probe set 61 %, black-box
   outputs 64 % (chance 20 %). The pre-registered "signal decays from weights to outputs" hypothesis was **refuted**.
   Code and per-run feature vectors released. https://arxiv.org/abs/2611.07610
3. **The Surprising Effectiveness of Deleting Weights in LLM Post-Training** (arXiv 2611.10866, Lu et al., NYU /
   Calgary, 7 Nov). Deletion Tuning: a learned binary mask over gate-projection weights (< 0.05 % zeroed, no new values)
   matches full fine-tuning at scale under GRPO, SFT and on-policy distillation; eight backbones 0.5B–30B; forgets far
   less (prior-task accuracy near base; FFT/LoRA/random-sparse lose up to 5.4 points). Geometry of the learned change
   (stable rank, spectrum drift, principal-weight overlap) is closest to FFT. https://arxiv.org/abs/2611.10866
4. **The Behavioral Rate of Fine-Tuning: Compression as a Measure of Useful Information** (arXiv 2611.06598, David,
   Caillon, Allauzen, 4 Nov). Operational rate R* = smallest compressed adapter that retains a fixed fraction of the
   held-out gain; **receiver-dependent** (same datasets reorder across base models); correction spectra measured on the
   data rank R* within a receiver; compression of a corrupted adapter recovers 23.7 → 82.9 on GSM8K (base 77.6); damage
   concentrated in leading singular directions. https://arxiv.org/abs/2611.06598
5. **IWIBench** (arXiv 2611.08647, von Csefalvay et al., HCLTech, 4 Nov). First controlled benchmark of gradient-free
   behavioural weight editing by another model (target Gemma 3 4B IT, ten task families, four tools, fixed budget,
   validator in a clean process). Macro reward: GPT-6.1 Sol 0.460 / 0.455, Astra 0.436, GPT-6.1 Terra 0.293, GPT-5.6
   0.21–0.24, open models ≤ 0.055; structured-response tasks unsolved by all; alias acquisition and recoding easiest;
   success across families correlates poorly. https://arxiv.org/abs/2611.08647
6. **Can an LLM Actually Reason and Program about Neural Network's Weights …** (arXiv 2611.08070, Ahmed, 5 Nov).
   Audit of zero-shot weight reading (Claude Opus 5.5, ~125k-parameter CNNs, weights as text): local arithmetic at the
   edges plus memorised dataset priors; label permutation breaks every semantic claim; mostly single runs; an audit
   protocol for future claims. https://arxiv.org/abs/2611.08070
7. **Can LLMs Debug Robots by Editing Their Weights?** (arXiv 2611.09376, 6 Nov). 65 token-embedding rows repair
   lexical grounding (86.3 → 92.0 on 765 held-out paraphrases, best of three sessions); matched random-direction edits
   do not help. https://arxiv.org/abs/2611.09376
8. **SWEET** (arXiv 2611.10974, Fu et al., 7 Nov). Training-free weight edits from activation contrasts on 20
   original/counterfactual pairs; localises MLP neurons and heads and removes their contribution along the extracted
   directions; +11.8 points on perturbed inputs across five open reasoning models. https://arxiv.org/abs/2611.10974
9. **Training-Free Task Vectors** (arXiv 2609.09054, Perin et al., revised 22 Oct). Maps activation steering vectors to
   rank-one weight edits from forward-pass statistics with task-vector arithmetic (add / subtract / compose); code at
   tftv-llm.github.io. https://arxiv.org/abs/2609.09054
10. Minor: WSP for image-restoration diffusion, conjectures only (arXiv 2611.04268); LLM-written scoring networks for
    MILP branching, matched by a plain pseudocost rule (arXiv 2611.09559); geometry-aware regularised editing
    (arXiv 2611.06185).

### Replications, controls and critiques (working papers and blogs, 4–10 Nov)

11. **Concept Country working paper** (Bennington et al., 8 Nov): pre-registered replication on **Qwen3.8-27B**.
    Directions read off residual activations on a train split, compiled as a rank-one additive term into the
    down-projections of nine alternating layers. Held-out: GPQA Diamond +4.3 (95 % CI +1.8 to +7.1, −30 % thinking
    tokens), LiveCodeBench v6 +13.0 (+6.1 to +20.6), IFBench +3.3 at −54 % tokens. **Norm-matched random control**
    (same magnitude, same nine matrices, per-layer random direction orthogonal to the fitted one) changes no benchmark
    and saves no tokens. Weights and graded artefacts published; every statistic recomputed from artefacts.
    https://concept.country/research/weight-space-programming
12. **aiXiv replication** (4 Nov): the released Astra edit raises 64 practice problems from 10.2 to 64.8 %; a
    **difference-of-means script** given Astra's layers, column budget and scale reaches 60.9 % (93 % of the gain) with
    directions at cosine ≈ 0.78 to Astra's; decoding changes alone reach 47.7 %; Opus 5.5 / Sonnet 5.5 blinded runs
    stop after 18–23 forward requests. https://aixiv.science/abs/aixiv.261103.000004
13. **Nathan Barry, "Can a Linear Program Do Weight-Space Programming?"** (9 Nov): Astra's column selection and scale is
    an LP; two LP solves (≈ 1 min each) reproduce the fix with 8× fewer changed weights; the heuristic direction gets
    most of the way without an LLM. https://nathan.rs/posts/weight-space-lp/
14. **Biswas, "Editing a Model's Weights for $10"** (8 Nov): Claude Fable 5.1 edits on Qwen3.5-9B and Qwen3.8-27B cut
    answer length 10–15 % with no measurable accuracy change (random edit 1.2 %); scripted edits cut as much or more;
    on Kinetix the Fable edit replaced the trained controller rather than repairing it; with thinking off the unedited
    0.8B already scores 44–47 % on MATH-500. https://glitch401.github.io/writing/editing-a-models-weights/
15. **Achintya K J audit** (4 Nov): every traced number appears in the released logs; independent rerun of the edited
    0.8B scores 51 % MATH-500 vs 9 % base; most results are single runs with partly undisclosed selection.
    https://achintyakj.tech/blog/weight-space-audit
16. **brainwagon, "A Skeptical Look"** (8 Nov) and **AutoWeight** (Xavier, 7 Nov; CartPole 164.6 → 465.1 held-out,
    SmolLM2 and LunarLander negative, numerical search matches Astra), **GEPA over weights** (6 Nov; mutation search on
    small nets reaches Adam-level), **Iyadh Khalfallah** (6 Nov; diagnosis transfers, LLM edits on a 0.8B model move the
    stop probability 1.2×, "effectively nothing"), **Jevlar** (opper.ai, 6 Nov; computed edits to 0.69 % of
    Qwen3.5-4B for a Pac-Man benchmark). URLs: https://brainwagon.org/blog/2026-11-08-a-skeptical-look-at-weight-space-programming ,
    https://midhunxavier.com/blog/auto-weight/better-weights-tougher-tests/ , https://devichand.com/blog/2026/gepa-wsp/ ,
    https://dev.to/iyadh_khalfallah/i-tried-weight-space-programming-on-6-kinds-of-models-heres-where-it-works-and-where-it-breaks-58e3 ,
    https://opper.ai/blog/jevlar-weight-space-programming
17. Public artefacts on our base: the Astra-edited checkpoint `RikiyaT/Qwen3.5-0.8B-weight-space-programming` (HF) and
    an edit diff of two tensors; Hugging Face lists dozens of new Qwen3.5-0.8B LoRA/SFT derivatives per day (free zoo).
18. Not found: any paper since 10 Nov that predicts benchmark change from a weight delta on a held-out split with a
    metanetwork.

## The questions for round 4

1. **External edits as a sign test.** Four real, held-out-validated or publicly checkable edits of our exact base now
   exist (Astra's two-tensor edit; the WSP-Scope 1,023-value column edit — fully specified except the direction u; the
   aiXiv difference-of-means edit; Barry's LP edit; plus SWEET/TFTV recipes that can be run on the base). Should the
   programme register an **outcome-blind external-edit panel** (scored on the TF-v1 WikiText and MMLU endpoints like any
   zoo entry; the field's sign and the chart's prediction recorded before scoring) as a dated A5 amendment, separate
   from Part A and Part B? What does it cost (TF-v1 ≈ 0.02 GPU-h per entry on H0), what is the decision value of n ≈ 4–8
   edits, and what would a fail mean given that these edits remove a looping direction rather than add capability?
2. **Weights vs activations on our base (item 2).** Rane's frozen LODO result (weights 42 %, activations 61 %, outputs
   64 %) is the closest published measurement to H1 on Qwen3.5-0.8B. Does it change the expected effect size of the
   field (which is weights-only with base gradients) against the activation-assisted arm, and should the registered
   access ladder (chart-only, field-only, hybrid, activation-assisted) be reported with this prior stated?
3. **New edit families (items 3, 8, 9).** Deletion masks on gate projections, SWEET suppressions and TFTV rank-one
   edits are cheap, outcome-blind-able and absent from the zoo (which has lora_sft, chain, abliteration, random_dir,
   heretic). Is a small registered **family-transfer panel** worth more information per GPU-hour than the 4B diagnostic
   was, and how would the field be expected to behave on a sparse deletion (the K-FAC contraction of a mask)?
4. **Receiver-relative information (item 4).** The behavioural rate is a property of the dataset–receiver pair, and
   correction spectra on the data rank it within a receiver. Our field is base-anchored (receiver-relative) by design.
   Is there a feature we are missing — e.g. the correction-gradient spectrum per objective — that the field could add
   without a new arm, and is "compression recovers a corrupted adapter" a prediction our chart could make?
5. **Controls and the H5 critic (items 5, 11–14).** Concept Country's norm-matched orthogonal control and
   pre-registration are the published analogue of Part A's spectrum-matched rotation. Does it suggest any change to the
   registered Part A controls (no, unless a flaw is found — say so explicitly)? For H5: IWIBench ranks GPT-6.1 Sol
   above Astra as an editor; the replications show a scripted difference-of-means / LP estimator recovers ≥ 90 % of the
   Astra gain once the layers and budget are supplied. Should the H5 pilot add a **scripted-estimator arm** (diff-of-means
   + LP, no LLM) next to the registered random best-of-N, and does that change the H5 success criterion (≥ 30 % realised
   evaluation savings)?
6. **Line-stop rule and the measurement paper.** Do any of items 1–16 change F5.5 (the joint line-stop rule) or what the
   measurement paper must say in related work? Name the exact sentences to add.

Rules as before: no invented numbers; every claim about our data cites a file; every external claim cites a URL you
fetched; ≤ 2,000 words; 10-line BLUF; final table "proposal | mechanism | registered test | GPU-h | kill criterion".
Write `round4-<role>.md` in this directory only; no commits, no GPU, no data/ writes, no edits to A5. Roles:
`world-model`, `representation`, `metrology`, `editor` (same angles as 7 Nov, `docs/council/2026-11-07/BRIEF.md`).
The chair then writes `SYNTHESIS.md`; any amendment goes to A5 as F8 before any G3 outcome is read.
