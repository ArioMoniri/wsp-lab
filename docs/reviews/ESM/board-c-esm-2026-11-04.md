# Board C review: protein LMs as an H5/H6 editing domain

Date: 4 Nov 2026  
Role: architecture assessor  
Branch: `esm-board-c`

## Verdict

**Do later, as a bounded H6 sub-study (provisional H7 if it changes the question).** It is a strong low-noise test of whether an editor can improve a small model's *contact head*, and a cleaner measurement setting for H1/H3 than sampled generation. It does not by itself test whether an editor can identify useful changes to a protein language model's representation weights, nor whether the small model can mimic a larger model. The standard ESM2 contact predictor has a very small, explicit supervised head over attention maps, making head-only fitting the most likely trivial source of gains.

## Fit to the hypotheses

| Hypothesis | Assessment |
|---|---|
| **H1: Δweights predict Δmetric** | Cleaner and likely stronger measurement than another generative LLM: precision@L on a fixed set is deterministic under fixed weights, software, and data; there is no decoding temperature or sampled output. This removes the cross-device sampled-generation effect documented in `docs/RESULTS.md`. It does not remove dataset/split uncertainty or implementation differences. One must use fixed model versions, precision, contact definition, and per-protein predictions. H1's present thresholds were framed around the existing edit zoo; preregister a protein-domain adaptation rather than silently calling it the same test. |
| **H3: held-out edit families** | Potentially stronger and more interpretable if edit families are meaningfully distinct: head-only, attention-map mixer, selected backbone layers, and low-rank adapters. Hold out a complete family and protein families/sequence clusters. An editor that only changes the fitted contact head could make H3 uninteresting; isolate head edits from backbone edits. |
| **H5: LLM editor + critic** | Stronger measurement of whether the critic ranks candidates at equal final score / evaluation count, because contact evaluation is cheap and deterministic. Weaker test of the headline H5 claim about savings: the outcome evaluation may itself be so cheap that saved evaluation calls have little practical value. Editors still need a well-defined interface, bounded edit space, sealed proteins, and matched non-LLM controls. It is a useful methods pilot, not evidence that the LLM replaces an optimizer. |
| **H6: across bases/architectures** | ESM2 size variants form a useful within-family scaling transfer ladder. ESM2 8M→35M→150M→650M→3B shares a family and contact-head mechanism, so success is weaker evidence for architecture generalization than the current Qwen/LFM/Gemma H6 design. ESMC 300M/600M offers a second family, but its contact-head availability/definition must be verified; it is not safe to assume ESM2's attention contact head exists there. Cross-size raw task vectors are impossible when tensor shapes differ; shared edit *rules* and shape-agnostic pooled descriptors are testable. |

The ESM2 released head is **not just an arbitrary classifier on sequence embeddings**. The ESM code's `ContactPredictionHead` symmetrizes attention maps, applies average-product correction (APC), flattens all layer×head maps into pairwise features, then applies a learned `nn.Linear(in_features, 1)` followed by sigmoid. The ESM repository describes the unsupervised contact predictor as a sparse linear combination of attention heads fit with logistic regression on 20 structures. Thus plausible editor actions include changing this head's coefficients/bias, changing which attention heads feed it, or modifying backbone attention weights so the maps change. The known direct mechanism is the linear contact head over transformed attention maps; a claimed backbone-level discovery needs to beat refitting that head alone and should not confuse head optimization with making the protein LM itself more capable. [ESM repository implementation](https://github.com/facebookresearch/esm) [ESM contact prediction notebook](https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb)

## Feasibility

### Checkpoints and terms

Hub API checks on 4 Nov 2026 found all five named ESM2 repositories public and ungated: `facebook/esm2_t6_8M_UR50D`, `esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D`, and `esm2_t36_3B_UR50D`. The model API did not expose a `license` field for these pages. The upstream ESM GitHub repo carries an MIT `LICENSE`, but that alone does not establish the terms attached to every downloaded checkpoint: **verify each model card and applicable weight terms before any use/publication**.

The EvolutionaryScale 300M and 600M ESMC checkpoints are also public, ungated Hub repos under `EvolutionaryScale/esmc-300m-2024-12` and `EvolutionaryScale/esmc-600m-2024-12` (the `biohub/` aliases resolve to the same revisions). Hub card metadata lists `mit` and `other`, not a single unambiguous license. The ESMC model card and linked Cambrian terms must be inspected; treat use as **to verify**, especially for commercial use. Do not infer that a Hub API `license` value of null means unrestricted.

### Data and protocol

Standard contact prediction reports **long-range precision at L** (often also L/5 and L/2), with contacts defined by a Cβ–Cβ distance threshold (glycine uses Cα); the exact threshold and minimum sequence separation vary by protocol and must be pinned. ProteinNet provides historical CASP-derived train/validation/test splits and is a defensible controlled source. CASP12 and later CASP targets and CAMEO are standard blind evaluation families, but raw structures, labels, access/redistribution terms, and exact split construction need verifying before adoption. The ESM repository includes an `ESMStructuralSplitDataset` notebook/tutorial and a contact-prediction evaluation path; confirm current file availability and split provenance before treating its subset as independent. Do not train/refit the contact head on any held-out evaluation proteins. Prefer a train split from ProteinNet and one sealed, sequence-clustered historical CASP subset; reserve CAMEO only if its data and scoring protocol can be frozen and obtained freely.

Verified reference pointers: [ProteinNet repository](https://github.com/aqlaboratory/proteinnet), [ProteinNet paper](https://arxiv.org/abs/1902.00249), [CASP prediction center](https://predictioncenter.org/), and [ESM repository](https://github.com/facebookresearch/esm). CASP/CAMEO download rights, ESM structural split details, and the precise contact threshold remain **to verify**. A current public benchmark catalog reports L/5 precision and ProteinNet/CASP12 test practice, but it is secondary and is not used as a protocol authority here.

### Cost, reusable code, and gaps

Parameter counts alone suggest ESM2-8M is a feasible local pilot and 35M/150M are practical follow-ons; 650M and 3B bring substantially higher memory/forward cost. For a rough lower-bound compute estimate, a standard dense Transformer forward is on the order of `2 × parameters × tokens` FLOPs, before attention's sequence-length-squared term, padding, data movement, and implementation overhead. If the pilot is 100 proteins, average length 256 residues, and 10 candidate evaluations per protein, the 8M backbone workload is only about 4.1×10¹² dense-linear FLOPs (100×256×10×2×8M), plus attention and overhead. This arithmetic is not a measured GPU-hour estimate. **Profile before scheduling; 8M pilot GPU time is to verify, expected well under one GPU-hour on the project GPU under these assumptions.** At 650M and 3B the same workload scales roughly with parameter count, but contact maps scale with L² and actual runtime depends on batching and attention implementation. No concrete timing claim for ESMC is justified until an inference smoke test.

Reusable: the project already has a zoo/edit manifest pattern, per-entry delta extraction and pooled features, predictors, split logic, and eval result manifests. `wsp.features.pool` is architecture-dependent and currently pools transformer and hybrid LLM module names; it must gain protein attention/head tensor roles, an ESM feature version, and tested handling for shared or packed Q/K/V and contact-head parameters. `wsp.zoo.arch` needs ESM model-family definitions and parameter roles. The existing eval harness is causal-LM/generation oriented; implement residue tokenization/loading, structure labels, pairwise masks, contact metric and per-protein records. Add sequence-cluster-aware splits to prevent homology leakage. Tokenizer/data loaders and exact ESM checkpoint loading are new; do not assume current HF `AutoModelForCausalLM` paths apply.

Edit families that can transfer: seeded random directions or low-rank perturbations within same-shaped tensors; same-model task vectors only (e.g. two fine-tunes derived from the same exact checkpoint and architecture); contact-head refits; head sparsification/ablation; and layer-local low-rank updates. A task vector between ESM sizes is shape-incompatible and must not be attempted. Cross-size transfer should use role-conditioned edit recipes (relative layer depth, normalized magnitude, target head or attention role), then instantiate them separately at each model size. The distinction is essential: transfer of an edit *rule* is not transfer of a parameter delta.

## “Mimic the larger model” claim

Improving contact precision on one benchmark is not evidence that an 8M model mimics a 300M/600M/3B model. At minimum compare, on identical held-out proteins:

1. **Unedited small-model baseline** and **larger-model teacher** with identical input sequences, contact definition, and evaluation split.
2. **Head-only supervised refit** on the small model's frozen attention maps, using training proteins only. This is the mandatory simple baseline and may explain most gain.
3. **Head-only editor** constrained to the same training examples and parameter budget.
4. **LoRA/backbone distillation**: train small-model adapters to match the large model's pairwise contact logits/probabilities (and optionally hidden/attention targets), with an equal teacher-forward and student-forward budget; evaluate the resulting student contacts on sealed proteins. Also include ordinary supervised fine-tuning on contact labels if available.
5. **Budget controls**: same number of small-model/teacher forwards and same optimizer/evaluation allowance for LoRA distillation, head fitting, and editor proposals. Report teacher inference cost separately from student training and test cost.

“Fraction of the cost” must name the cost axis and include teacher generation/training amortization, student training and inference FLOPs or measured GPU-hours, peak memory, repeated candidate evaluations, editor API/token cost and wall time, and any data/label acquisition cost. Compare equal quality on the same held-out proteins, not just relative gain from different baselines. If the teacher is evaluated once to produce labels, report that cost and the number of downstream student uses needed to amortize it. Without these controls, say only “contact score improved under this edit,” not “mimics a larger model” or “at a fraction of the cost.”

## Minimal pilot and recommendation

**Recommendation: do later**, after the protein data/license and contact protocol are frozen and a no-editor head-only baseline exists. Then run this preregistered pilot as a sub-study of H6 (or label **H7: protein-domain editor generalization** if the council wants a separable claim):

- Model: `facebook/esm2_t6_8M_UR50D`, exact pinned revision, with released contact head.
- Data: ProteinNet historical train/validation for fitting; a frozen, sequence-clustered CASP12-derived test set for final scoring. Exact files, split hashes, contacts and exclusion rules are **to verify before run**.
- Metric: long-range precision@L, per protein then macro-average; report paired bootstrap CI across proteins, and L/5 as secondary. Freeze sequence-separation and distance thresholds before viewing test results.
- Arms: unchanged model; training-set head-only logistic regression; Claude editor allowed to propose edits only to the contact head; and one constrained attention/backbone edit family. Use 20 proposals per editable arm, with 5 fixed prompt/seed replications (100 episodes each) only if the first 20-proposal screen has usable variance. Do not expose sealed structures or per-protein test outcomes.
- H1/H3 add-on: extract shared features for the 20+ labeled edits; test delta-norm and StatNN baselines, then one held-out edit-family split. This is exploratory until data volume supports the existing H1/H3 thresholds.
- Cost: target **<1 GPU-hour for the initial 20-proposal screen**, derived from the explicit 8M/100-protein/256-residue/10-candidate FLOP assumption above; measure actual GPU time and stop expanding if the full evaluation exceeds the cap. This is a budget target, not an observed timing. Claude calls/tokens are separately metered and not included in GPU-hours.

The test resolves whether Claude can improve a small ESM contact predictor beyond ordinary head refitting, whether a backbone edit adds anything, and whether one cheap deterministic domain can support H1/H3/H5 methods. It cannot establish scaling equivalence, broad protein understanding, or ESM2-to-ESMC architecture generalization. If contact-head refitting closes the gap, stop the editor claim. If a backbone edit beats the head-only control on sealed proteins and transfers to another edit family, then expand to ESM2-35M and ESMC-300M as a deliberate cross-family H6 follow-up.

## Verification notes and unresolved facts

- Verified by `curl` against Hub model APIs: named ESM2 repos exist, are public and ungated; ESMC repos exist, are public and ungated; ESMC metadata reports `mit` plus `other`; ESM2 API omitted license metadata.
- Verified against upstream ESM source: `ContactPredictionHead` symmetrizes and APC-corrects attention, then linear regression and sigmoid; source LICENSE is MIT.
- Verified reference: ProteinNet arXiv identifier `1902.00249` resolves to *ProteinNet: a standardized data set for machine learning of protein structure*.
- To verify: ESM2 checkpoint-specific terms; ESMC applicable license clauses; current official contact evaluation download links/splits; CASP/CAMEO reuse terms; contact thresholds and separation; inference throughput on this GPU; whether ESMC exposes a comparable contact head.

## Five-line summary

Branch: `esm-board-c`  
Verdict: **do later**  
Strongest reason: deterministic scores improve measurement, but the known ESM2 sparse logistic attention-map head is an obvious confound to any “weight discovery” claim.  
Pilot cost: target under 1 GPU-hour for ESM2-8M / 100 proteins / 20 editor proposals; estimate only, profile to verify.  
Biggest risk: apparent mimicry is just contact-head refitting or leakage from homologous proteins, not a better protein language model.
