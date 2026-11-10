# Council brief — 10 Nov 2026, round 3 (read first)

## Why this round sits

The user asked for two things on 10 November: check the latest work on weight editing, and ask the council again what the
**best world-model predictor we can build now** is, given everything measured so far. GPU time is approved. The 7 Nov programme
(`docs/council/2026-11-07/SYNTHESIS.md` v3.1, PREREG appendix A5) is registered but **not yet frozen**: Board A's checklist items 1–6
are done (null-size simulation PASS), item 7 (hash freeze) is pending. This round may therefore still change the programme by a
dated amendment, as long as no new outcome has been read. Nothing from G3 has been scored.

## Results so far (all numbers are in the files named)

| Item | Result | File |
|---|---|---|
| Data recovery (D8) | 351 of 358 lost chain snapshots replayed and installed; 7 remain lost; exactness classes bitwise / re-trained / lost; re-trained LoRA steps have 3–15 % matrix relative error | `docs/reviews/P5/d8-regeneration-2026-11-06.md` §7, `docs/DECISIONS.md` D9 |
| Full-set predictor suite (342 rows / 278 clusters, protocol v2, 3 seeds) | seed-0 StatNN 0.56 / 0.66 (seed split), 0.55 / 0.64 (trajectory) against logged 0.57 / 0.65; descriptor-only 0.51 / 0.61; best neural (M1-matched) 0.45 / 0.50 and 0.42 / 0.56; family split within [−0.29, 0.23]; neural means drifted by a median of 0.09 between the two instantiations | `docs/reviews/P4/d3v2f-summary-2026-11-08.md` |
| Recovered-set suite (297 rows) | StatNN 0.54 / 0.65; descriptor-only 0.48 / 0.58; delta-norm 0.16 / 0.21; M1-matched 0.47 / 0.54 | `docs/reviews/P4/d3v2r-summary-2026-11-06.md` |
| Teacher-forced tier | G1 repeat 14 of 14 bitwise; MMLU gold log-prob unqualified (H1a untested); WikiText H1w inconclusive (θ̂ 0.007, power 0.73 / 0.60) | `docs/reviews/P5/tf-v1-requalification-2026-11-06.md`, `h1w-gate-2026-11-06.md` |
| TF-v1 census | records for 694 entries incl. 652 of 675 chain entries; 95 of 99 root-to-leaf lineages with ≥ 4 nodes fully scored (H2 still parked: below the 162-lineage stand-in) | `docs/PROGRESS.md` 8 Nov |
| Registered programme | oriented local response: base-anchored sensitivity field (gradients + K-FAC contractions of the effective delta) as primary arm against augmented StatNN, field-absent ablation, random/permuted-field controls; chart as secondary; Part A rotation + ladder on 40 selected D3 edits (120 TF-v1 scores); WikiText first; go / screen / no-go / unresolved rules | `docs/PREREG_H1_TF.md` appendix A5, `docs/council/2026-11-07/SYNTHESIS.md` |
| A5 evidence | registered selection 30 lora_sft + 10 chain (40 components); references to score 19; null-size simulation PASS (ordered go rate 0.018 at nominal 0.05; sensitivities 0.016 and 0.0); one sparse cell (LoRA low tertile, 1 donor) | `docs/prereg/a5_manifests/` (branch `prereg/a5-freeze-evidence`) |
| Code on main | `features/rotation.py`, `response_field.py`, `response_chart.py` (projector dim 4; 10,812 augmented columns), `eval/response_intervention.py`, `scripts/a5_simulation.py` | main |

## What changed outside since 2 November (verified 10 Nov; cite the URL you read, not this digest)

1. **WeightLoop** (arXiv 2611.07319, 5 Nov 2026, Li et al.): weight-space programming with experimental feedback. A fixed editor
   written once by GPT-6 Astra revises a 3.6 M-parameter MLP's 128 output weights as new assay results arrive; each accepted update is
   rebuilt from θ₀ (patches never compound), kept only if it improves held-out support validation, with random-permutation controls;
   targeted edits beat random ones in 7 of 8 ranking settings. No LLM benchmarks; no held-out predictor of an edit's effect from the delta.
   https://arxiv.org/abs/2611.07319
2. **Neural Thickets** (arXiv 2603.12228, Gan & Isola, Mar 2026): the fraction of random Gaussian perturbations that beat the base
   rises with scale (Qwen2.5 0.5B → 32B on GSM8K, Countdown, MATH-500 and others); RandOpt samples hundreds of perturbations, keeps the top-K
   on a training split and ensembles them. No predictor; but it says that at 0.5–0.8B almost all random edits are null or harmful, which is
   exactly our zoo's picture (0 of 60 random_dir and 2 of 100 abliteration edits beyond KL 0.01). https://arxiv.org/abs/2603.12228
3. **Learning on LoRAs (LoL)** (arXiv 2410.04207, ICLR 2025): GL-equivariant networks over raw LoRA factors predict properties of the
   fine-tuned model including ARC-C accuracy of LLMs, "over 50,000 times faster than standard evaluation". This is the closest published
   precedent for our H1 and it was **not** among the arms we trained (M1 used W2T-style canonical tokens in a random frame).
   https://arxiv.org/abs/2410.04207
4. **Critiques and replications of the blog** (all from the last week, none peer-reviewed): the raw-weights-only setting was used only on
   networks of 17k–172k weights, every LM experiment used Python tooling (Honnibal; Willison); the Gemma-3-270M repeat produced no gain
   (Wikipedia summary of the blog's own addendum); the "6×" MATH-500 comparison is against the looping thinking-on arm and is not paired
   (González / "Polaris" thread); a community autoresearch repo (csinva) concludes that reading all weights at once without running the
   network is not a plausible way to predict what an edit does; a Claude-driven replication (MaceTenth) reproduces the small-network
   results, not the Qwen edit. https://www.mindstudio.ai/blog/weight-space-programming-overhyped-debate ,
   https://github.com/csinva/weight-space-programming-autoresearch , https://github.com/MaceTenth/weight-space-programming-replication ,
   https://en.wikipedia.org/wiki/Weight-space_programming
5. **Weight Space Representation Learning via Neural Field Adaptation** (arXiv 2512.01759, CVPR 2026): multiplicative LoRA weights as
   representations; structure induced by a pretrained base plus low rank; reconstruction and generation, not capability prediction.
   https://arxiv.org/abs/2512.01759
6. No 2026 paper was found that predicts downstream benchmark change from a weight delta on a held-out split with a metanetwork; the
   nearest are proxy-metric forecasting from outputs (arXiv 2605.18607) and hypernetwork weight prediction (arXiv 2508.05059).

## The question for round 3

**Design the best predictor we can build now**, for the effect of a weight edit on Qwen3.5-0.8B's teacher-forced and likelihood
benchmarks, with the data we have (1,252 feature sets, 694 TF-v1 records, 342 fast-tier matched rows, 40 registered Part A edits) and
up to 30 further GPU-hours on the local card (Lambda allowed for parallelism). Each member answers from their angle:

1. **What the news changes.** For each of items 1–6: does it change the registered programme, and how (an amendment before freeze, a
   new registered secondary arm, nothing)? In particular: should a **LoL-style GL-equivariant encoder over raw LoRA factors** become a
   registered arm next to the field (it is the only published method with a held-out LLM-accuracy result from weights); does Neural
   Thickets' density-vs-scale result mean a 0.8B zoo is the wrong place to find signal (and what the cheapest scale check is); does
   WeightLoop's validated-reversible-update loop change the H5 critic contract.
2. **The best predictor now.** One concrete design: inputs (field tokens, chart contractions, raw LoRA factors, probe activations, descriptors),
   architecture, training recipe on our row counts, the expected gain over augmented StatNN and why, and the registered test it would
   face. Be specific about what is cheap because the code exists (section "Code on main") and what must be built.
3. **The decisive next 30 GPU-hours**, ranked: Part A as registered (≈ 2.1 GPU-h + 19 references) first or not; which additional arms or
   data purchases (new LoRA edits at matched norms? a 4B scale check? the 296-entry fast-tier closure?) buy the most information per hour.
4. **Kill criteria** for the whole "weight-space world model" line at this scale: what result, if it comes out of Part A/B, should make us
   stop and publish the measurement paper as the final word.

Rules as before: no invented numbers; every claim about our data cites a file; every external claim cites a URL you fetched; ≤ 2,000 words;
10-line BLUF; final table "proposal | mechanism | registered test | GPU-h | kill criterion". Write `round3-<role>.md` in this directory
only; no commits, no GPU, no data/ writes. Roles: `world-model`, `representation`, `metrology`, `editor` (same angles as 7 Nov,
`docs/council/2026-11-07/BRIEF.md`). The chair then writes `SYNTHESIS.md`; any amendment goes to A5 before the freeze.
