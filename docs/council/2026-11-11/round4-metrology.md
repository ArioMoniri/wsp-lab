# Round 4 — Metrology

## BLUF
1. Register an outcome-blind external-edit panel as a separate, descriptive A5 amendment; it must not alter Part A/B or become a gate.
2. Prefer n=8 only if eight edits represent distinct, fixed sources; otherwise count source clusters, not near-duplicate edits.
3. Exact signs are informative but weak at n=4–8: under independent fair-coin signs, 4/4 is not a two-sided 5% result; even 8/8 only narrowly clears 0.60.
4. Score on TF-v1's paired WikiText and MMLU endpoints, but only WikiText is qualified; MMLU remains descriptive.
5. Astra's loop-removal result is not directly measured by current TF-v1: its known gain is thinking-on generation/termination, while our fixed-sequence endpoints use thinking off.
6. Rane changes the access-level prior, not the predicted effect size of our base-gradient field; report it as context, not calibration.
7. Concept Country exposes no flaw in the stronger spectrum-matched rotation or ladder; import no new practice into A5.
8. A six-edit deletion/SWEET/TFTV transfer panel buys more targeted information per scoring hour than the closed 4B Gaussian diagnostic, but remains exploratory.
9. Keep F5.5 bounded and unchanged; its Part A branch is a funding/precision rule, not evidence of wrong signs (A5 F5.7).
10. Add related work that distinguishes editing success from held-out edit-effect prediction and states these endpoint and control limits.

## 1. External-edit sign panel

**Recommendation: yes, as a non-gating external audit.** Freeze the four described edit entries and their source lineage, field/chart signs, scoring configuration and endpoint hierarchy before viewing TF-v1 outcomes. Expand toward n=8 only with fixed, source-distinct edits; Astra, aiXiv and Barry reuse the anti-loop target/support to varying degrees, so those entries are not independent Bernoulli trials. Report edit-level `k/n`, with exact binomial inference only at the independent source-cluster level. Publicly selected edits test these fixed artifacts, not a random sample of all edits.

Operationally, start with those four described entries; include the WSP-Scope edit only if its exact delta is publicly recoverable and hash-pinned, since the paper does not release direction `u` ([arXiv:2611.09382](https://arxiv.org/abs/2611.09382)). Astra, aiXiv and Barry are correlated anti-loop lineages, so these four entries are fewer than four independent trials. Add distinct sources only to reach n=8; otherwise report the small clustered panel descriptively.

For a predeclared one-sided sign test against p=.5, the exact tail is `2^-n Σ[j=k..n] C(n,j)`. Illustrative results: 4/4 gives p=.0625 (two-sided .125); 6/6 gives p=.015625 (5/6 gives .1094); 7/8 gives .0352 and 8/8 gives .0039. Equal-tail 95% Clopper–Pearson lower bounds for perfect agreement are only .398, .541 and .631 at n=4, 6 and 8. Thus n=4 cannot reject fair-coin agreement even if perfect; n=8 needs 8/8 to put the two-sided lower bound above .60. These calculations assume independent edits and a fixed directional hypothesis. `k/n` cannot establish effect magnitude, causal benefit, calibration outside the panel, or that a looping fix adds capability. A mismatch on WikiText/MMLU means only that the registered weight-only sign failed on those endpoints.

Use the registered TF-v1 scorer and the common base reference; record signs before scoring. The config has 64 WikiText windows, 1,000 MMLU items, and `thinking: false` (`configs/eval_tf_v1_wikitext.yaml`; MMLU qualification remains failed in `docs/reviews/P5/h1w-gate-2026-11-06.md`). Warm TF-v1 entries measured 1.003 min median and 1.040 min maximum on H0 (`docs/RESULTS.md`, TF-v1 G2 timing table); at 1.04 min/entry, n=4–8 plus one newly scored base costs 0.087–0.156 GPU-h, excluding edit construction, any cold load and repeats. If the identical base record is reusable, subtract 0.017 GPU-h. MMLU signs are descriptive, not a second confirmatory test.

The known Astra result is a generation effect: Qwen3.5-0.8B MATH-500 rises 7.9→48.1 with thinking on, while the authors describe repairing reasoning loops ([original post](https://rikiyat.github.io/blog/weight-space-programming/); [arXiv:2611.09382](https://arxiv.org/abs/2611.09382), HTML read at https://arxiv.org/html/2611.09382v1). The aiXiv replication and Barry's LP analysis report that scripted difference-of-means/LP methods recover much of the same anti-loop gain once support and related choices are supplied ([aiXiv](https://aixiv.science/abs/aixiv.261103.000004); [Barry](https://nathan.rs/posts/weight-space-lp/)). TF-v1 scores fixed next-token likelihoods, not generated loop frequency, answer completion, or token-limit rates. Astra's behavioral effect is therefore **not directly measurable on these TF-v1 endpoints**. Its incidental log-probability change could be measured, but a null or non-null NLL shift would not establish loop removal; that needs a separately registered teacher-forced loop-trace panel or generation endpoint.

## 2. Weights versus activations

Rane's frozen LODO study on the same model family reports 42% domain classification from weight statistics, 61% from activations and 64% from outputs; its target is the training-data domain, not the direction or size of an edit's benchmark change ([arXiv:2611.07610](https://arxiv.org/abs/2611.07610); HTML read at https://arxiv.org/html/2611.07610v1). It does not supply an effect-size prior for the base-gradient field versus the activation-assisted arm. It does support reporting access as an explicit axis: show chart-only, field-only, hybrid and activation-assisted results separately, alongside the paired contrasts and the preregistered field-absent control (`docs/PREREG_H1_TF.md`, A5.4 and A5.8). Do not pool the accuracy numbers with H1 or treat the activation arm as a fairer estimate of field signal; it has additional information.

## 3. New-family transfer versus the closed 4B diagnostic

Register a six-edit, 0.8B family-transfer screen (two each: deletion masks, SWEET-style suppressions, TFTVs), fixing each recipe/support from development-only inputs and keeping TF-v1 labels sealed. Deletion produces sparse `ΔW=−M⊙W`; its field contraction is exactly the registered signed `⟨g,ΔW⟩` over selected entries, with the K-FAC term evaluated on that support. Sparsity alone gives no sign prior: cancellations and receiver-specific gradients can make the effect small or reverse it. Deletion Tuning's published results concern learned gate masks during post-training, SWEET derives edits from activation contrasts, and TFTVs map activation directions to rank-one edits; none is a validation of our TF-v1 field ([Deletion Tuning](https://arxiv.org/abs/2611.10866); [SWEET](https://arxiv.org/abs/2611.10974); [TFTV](https://arxiv.org/abs/2609.09054)).

This is more decision-relevant per scoring hour than repeating a random-Gaussian scale screen: it tests mechanism/family transfer on the target base rather than beneficial-edit density. Six edits plus base cost about 0.121 GPU-h at the measured 1.04 min/entry (or 0.104 h with a reusable base); construction and any activation-probe work are additional and must be separately capped. Keep WikiText as the only qualified primary and MMLU descriptive. The 4B diagnostic is already closed: only 3/20 improved, none resolved as improving, and H100↔H0 A3 pairing failed (0.662 > 0.1); it was explicitly a diagnostic, not a gate (`docs/reviews/scale/scale4b-diagnostic-2026-11-10.md`). The new panel cannot retroactively repair or replace it.

## 4. Receiver-relative information

The Behavioral Rate paper defines compression rate relative to a dataset–receiver pair and reports that correction spectra rank rates within a receiver; its “compression recovers a corrupted adapter” result is about compressed-adapter held-out task performance, not a predictor of one edit's TF-v1 delta ([arXiv:2611.06598](https://arxiv.org/abs/2611.06598)). A5 already stores objective-specific gradients, K-FAC factors, spectra/tails and signed edit contractions (`docs/PREREG_H1_TF.md`, A5.8.1). A deterministic spectrum-of-correction summary could be reported descriptively from those artifacts, but feeding it into a predictor changes the frozen feature schema and needs a pre-outcome amendment. Do not add it to this run. The chart cannot predict recovery from adapter corruption without corruption/compression variants and corresponding outcomes; no such target is registered.

## 5. Concept Country control and preregistration

Concept Country's Qwen3.8-27B replication uses a same-magnitude random direction orthogonal to the fitted direction on the same nine matrices; its control is null while edited results improve held-out benchmarks ([Concept Country, “Gradient-Free Weight Editing Reduces Overthinking in Qwen3.8-27B”](https://concept.country/research/weight-space-programming)). This is a useful control, but it does not expose a flaw in Part A: it tests a random orthogonal direction, while A5 rotates each realised edit while preserving its singular spectrum and support, then tests the registered strength ladder against the same predecessor. These are complementary nulls, and Concept Country has no corresponding ladder. Different model, endpoints and intervention also prevent using its effect as A5 calibration (`docs/PREREG_H1_TF.md`, A5.3.2–A5.3.4).

**Nothing from its pre-registration practice needs to enter A5.** Its artifact publication and recomputation of statistics from graded files are good reporting practice, but A5 already freezes hashes, selection, scoring, controls and inference before outcomes and requires provenance and audit traces (`docs/PREREG_H1_TF.md`, A5.1–A5.3). No control, ladder or decision-rule amendment is warranted.

## 6. Line stop and related-work text

The new external work does not change F5.5's scope: retain its joint, bounded stop rule. A5 F5.7 and the null-size report explicitly note that the “alternative” branch occurred in 1,939/10,000 fair-coin replicates (0.1939), is a funding stop, and is not evidence of wrong signs; preserve that distinction in reporting (`docs/prereg/a5_manifests/nullsize/report.json`; `docs/PREREG_H1_TF.md`, F5.5/F5.7). Do not promote this branch to a stand-alone scientific falsifier.

Add these exact sentences to related work/limitations:

> “Published weight-space edits demonstrate endpoint-specific behavioral interventions, including loop removal, but do not by themselves establish held-out prediction of benchmark deltas from a weight edit ([Takehi et al.](https://rikiyat.github.io/blog/weight-space-programming/)).”
>
> “For the Qwen3.5-0.8B anti-loop edit, independent analyses report that scripted difference-of-means and linear-program estimators recover much of the reported gain once direction, support or budget choices are supplied; this motivates separating edit discovery from effect prediction ([aiXiv](https://aixiv.science/abs/aixiv.261103.000004); [Barry](https://nathan.rs/posts/weight-space-lp/)).”
>
> “Recent work also studies deletion-only post-training, activation-derived suppression edits and forward-statistic rank-one task vectors, but evaluates different intervention targets and does not validate our registered TF-v1 field predictions ([Deletion Tuning](https://arxiv.org/abs/2611.10866); [SWEET](https://arxiv.org/abs/2611.10974); [TFTV](https://arxiv.org/abs/2609.09054)).”
>
> “Rane et al. compare weight, activation and output access for fine-tune-domain classification, not edit-effect prediction; their access ordering is context rather than an effect-size prior for our field ([arXiv:2611.07610](https://arxiv.org/abs/2611.07610)).”
>
> “A preregistered norm-matched orthogonal control on Qwen3.8-27B is a complementary intervention control, not a calibration of our spectrum-matched rotation and ladder on Qwen3.5-0.8B ([Concept Country](https://concept.country/research/weight-space-programming)).”

| proposal | mechanism | registered test | GPU-h | kill criterion |
|---|---|---|---:|---|
| External-edit panel, target n=8 | Tests frozen field/chart signs on public edits without feedback | Date/hash edits, lineage clusters and predictions before TF-v1; WikiText primary, MMLU descriptive; exact CP/binomial by independent source | 0.087–0.156 including one base (H0 warm estimate) | No standalone kill; panel failure is endpoint-specific. No structural conclusion from clustered/underpowered signs. |
| Family-transfer panel, n=6 | Sparse deletion, counterfactual suppression and rank-one edits test transfer beyond zoo families | Two per family, development-only construction, frozen TF-v1 scoring | ≈0.121 including base; construction excluded | Stop expansion if no resolved WikiText effects or source/outcome separation fails; do not infer universal failure. |
| Preserve F5.5 | Joint evidence bounds one 0.8B line, not all weight-space editing | Run existing A5 rules; report the 0.1939 null alternative-branch rate and all three predicates | 0 new | Stop only when all three valid predicates hold; otherwise unresolved. |
| No new correction-spectrum predictor | Reuses existing field factors as descriptive diagnostics | Any new predictor input requires a dated pre-outcome amendment and fresh test | 0 now | Do not add to frozen run; compression recovery remains untested. |
