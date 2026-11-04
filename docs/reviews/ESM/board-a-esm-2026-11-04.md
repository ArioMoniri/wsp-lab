# Board A review: ESM contact-map weight editing (measurement and claims)

Date: 2026-11-04. Reviewer: Board A (strict ML correctness). Scope: metric, noise, leakage, admissible claims, cheating routes, and value to H1/H3/H5. Architecture and feasibility belong to Board C and are not assessed here.

**Verdict: REVISE.** This reviews a proposal. Nothing has been run, no run ids exist, and no numbers below are measurements. Every threshold is a proposed pre-registration criterion. Arithmetic on verified config values is labelled as arithmetic.

---

## 0. Verified facts and reference status

Status key: **V** = checked against a primary source in this review. **S** = taken from a search summary or secondary source, not opened directly. **TV** = to verify before any paper use.

| # | Fact used below | Status | Source |
|---|---|---|---|
| R1 | Lin et al., *Science* 379(6637):1123–1130, 2023, DOI 10.1126/science.ade2574 (ESM-2; P@L reported across scales) | S | Search summary; GitHub README bibtex |
| R2 | ESM-2 8M: 6 layers, 20 heads, hidden 320. 650M: 33 layers, 20 heads, hidden 1280. 3B: 36 layers, 40 heads, hidden 2560 | V (values extracted from `config.json` by fetch summariser; re-read the file at the pinned revision) | huggingface.co/facebook/esm2_t6_8M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D |
| R3 | `ContactPredictionHead`: `nn.Linear(layers·heads, 1, bias=True)`, sigmoid, symmetrise then APC | V (fetch summariser; line numbers not captured; pin a commit and cite lines) | github.com/facebookresearch/esm, `esm/modules.py` |
| R4 | Contact-head parameter counts: 8M = 6·20+1 = **121**; 650M = 33·20+1 = **661**; 3B = 36·40+1 = **1441** | Arithmetic from R2 and R3, not measured | this review |
| R5 | ESMC-300M and ESMC-600M exist; 300M card says not fine-tuned for a task, so a contact head must be trained downstream; 600M card lists a custom non-commercial licence | S | HF cards: multimolecule/esmc-600m, Biohub ESMC-300M-hf, EvolutionaryScale/esmc-600m-2024-12 |
| R6 | ESMC pretraining: UniRef, MGnify, JGI, clustered at 70% identity | S (secondary mirror) | Biohub card via search |
| R7 | Contact bands: short 6–11, medium 12–23, long ≥24; contact = Cβ–Cβ < 8 Å (Cα for Gly); L/k denominators; denominator substitution when a bin has no native contacts in the top L/k | S (secondary papers; official CASP assessment document not located) | arXiv 1609.00680 (Ultra-deep); CASP13 AFiser slides |
| R8 | ProteinNet: CASP targets as test, identity-level training splits (train_30, train_50, …) | S for concept; the 30% clustering procedure is **TV** | PMC6560865 (AlQuraishi 2019) |
| R9 | ESM-2 pretrained on UniRef50 2021_04 | S (one preprint; labels differ between sources) → **TV** against the Lin et al. supplement | Search |
| R10 | Whether the ESM-2 contact head was trained on PDB structures | **TV** (not found; do not rely on it) | — |
| R11 | MMseqs2 for identity clustering (Steinegger & Söding 2017) | **TV** (not searched) | — |
| R12 | Repo facts: Board A prereg (`docs/reviews/H5/board-a-prereg-2026-11-04.md`), `docs/RESULTS.md`, `docs/MODEL_CANDIDATES.md` §LLM-in-the-loop, `docs/H5_PROTOCOL.md` | V (read directly) | repo |

No ESM code, data, or ESM-related docs exist in the repo; `grep` found only a board usage-log line for this idea.

---

## 1. Metric and noise

### 1.1 Is the metric deterministic? Only conditionally

Given fixed weights, a fixed protein set, and a fixed evaluation pipeline, the P@k number is a single value. That removes *evaluation sampling* noise. It does not make the number portable. Determinism holds only conditional on:

- **Device class and kernel path.** Cross-device gaps appear even at fixed seed on sampled LLM generations (RESULTS.md: mean cross-device |Δ| on gsm8k 0.053, max 0.12; the same-host retest was bit-identical). A forward pass without sampling is not immune: fp16/bf16 kernels can change logits, and therefore top-k membership when scores are near-tied.
- **Dtype.** If attention or the sigmoid output is computed in bf16/fp16, logits are quantised and **exact ties at the k-th rank become common**. Ties are then broken by index order, which is arbitrary and can favour particular residue pairs. Score in fp32 and record tie counts at the cut.
- **Batch composition and padding.** Padding and batch size can change numerics in attention. Check batch=1 against batched outputs on the same proteins.

**Required check (pre-registered):** same-device repeat (bit-identical expected, same as RESULTS.md); batch-invariance; cross-device panel with per-protein top-k set hashes; report fraction of proteins whose top-k set is identical across devices and mean/max |ΔP@k|. Expect near-identity; **measure it, do not assume it.**

### 1.2 The scientifically relevant variance is not the metric's

The editor is stochastic. Each episode produces a different edit, and the claim is about the edit process, not one edit. So uncertainty has two layers:

1. **Protein-set sampling** (finite test proteins; clusters are the unit).
2. **Editor-run variation** (independent episodes, seeds, and prompts).

The metric's determinism removes neither. It does make layer 2 cheap to sample: with a deterministic metric, hundreds of episodes are feasible. That is the real advantage (section 4).

### 1.3 Noise sources that remain

| Source | Why it matters here | Design response |
|---|---|---|
| Protein-set sampling | Dominant. Few clusters give wide CIs. | Cluster bootstrap; report n clusters per cell. |
| Length distribution | P@L/5 is over k = L/5 predictions. Short proteins give small k, so per-protein P@k moves in steps of 1/k (e.g. 0.1 at k=10). Long proteins may lack eligible long-range pairs. | Stratify by length bin; pre-register minimum k and minimum eligible pairs per cell; exclude below threshold; report exclusions. |
| MSA depth / pretraining-perplexity confound | Contact accuracy in masked LMs is widely reported to track pretraining perplexity and homolog depth (**TV**, Lin et al. 2023). An edit that lowers perplexity on low-depth families looks like a contact gain. | Record per-protein masked-LM perplexity and a pre-registered homolog-count covariate; stratify; test whether the edit gain is explained by the covariate. |
| Sequence-identity leakage to the pretraining corpus | ESM-2 pretraining used UniRef (R9, TV); PDB sequences are largely in UniRef. A "held-out" PDB protein may be in the pretraining set of one model and not another. Asymmetric leakage favours the larger model. | Cluster the test set at a stated identity (30% is the ProteinNet naming convention, R8; procedure TV) and check each protein's maximum identity to **each model's** pretraining corpus. Use a post-cutoff subset as a robustness arm. |
| Contact definition | Cβ/Cα rule, distance cutoff, resolved vs full length L, chains, missing residues, homo-oligomer inter-chain pairs. Each choice moves P@k. | Freeze the definition in the pre-reg before scoring. Report the sensitivity to the two most likely alternatives (Cα-only, resolved-length L). |
| Denominator convention | Empty bins substitute L/k in some papers (R7). Changes the per-cell means. | Pre-register and state the convention; report counts of empty bins. |
| Head-protocol mismatch across models | ESM-2 ships a trained head (R3). ESMC is not shipped with one (R5). "Matching 650M" compares protocols if the ESMC head is trained differently. | One head protocol for all models: same training split, same budget, same parameters. Head-free metrics as a secondary check if available. |
| Process noise (editor) | Dominant for the claim (1.2). | Episode replicates with a registered count; crossed episode × protein bootstrap. |

### 1.4 Reliability design

- **Resampling unit:** identity cluster of proteins. Count clusters, not proteins, in n.
- **Bootstrap:** 10,000 cluster-bootstrap replicates for paired differences, with the pre-registered CI method (percentile, or BCa if the cluster count is large enough to estimate acceleration). Crossed bootstrap over episode × cluster when editor runs are in the claim.
- **Pairing:** each protein is scored under every arm; the comparison is the paired difference per protein, aggregated over clusters.
- **Multiplicity:** 3 ranges × 3 cutoffs = 9 cells. Pre-register **one primary cell** (proposed: long-range P@L/5, the classic CASP-style cell, R7). The other eight are secondary, Holm-adjusted.

**Minimum set size for a 1-point paired difference.** Let σ_d be the SD of the per-protein paired difference in P@k. For two-sided α = 0.05 and power 0.8, the paired-mean sample size is:

n ≈ ((z₀.₉₇₅ + z₀.₈) · σ_d / δ)² = 7.85 · σ_d² / δ²

With δ = 0.01 (one point, absolute): **n ≈ 78,500 · σ_d²**. σ_d is **not measured**; the table shows the arithmetic at hypothetical values only:

| Hypothetical σ_d (not measured) | n proteins (or clusters) at 80% power |
|---|---|
| 0.05 | ≈ 197 |
| 0.10 | ≈ 785 |
| 0.20 | ≈ 3,140 |

Clustering inflates the variance (design effect 1 + (m−1)ρ, with m proteins per cluster), so the count must be clusters after identity clustering. **Required pilot:** measure σ_d on untouched 8M vs 650M (and the head-only control) on a pilot set, then size the test set from the measured σ_d. Until then, the set size is undetermined and any "1-point" claim is unpowered.

---

## 2. What claim would be admissible

### 2.1 Claims by tier

| Claim | Admissible? | Required |
|---|---|---|
| "Edited 8M improves long-range P@L/5 over untouched 8M" | Only after M1, M8, M9 below | Paired protein comparison, cluster CI, one locked test evaluation per candidate. |
| "Editor-found edit beats a head-only tuned 8M" | Only with M2 | Same data and budget; head-only is the real baseline (121 parameters). |
| "LLM editor beats non-LLM search over the same weights" | Only with M5 and M6 | Same parameter support, evaluation budget, and selection rule. |
| "Edited 8M matches ESM-2 650M" | Only with an equivalence test on the same protocol | TOST with pre-registered margin (proposed δ = 0.01 absolute, to be confirmed after σ_d pilot), 650M scored under the identical head protocol. |
| "At a fraction of the cost" | Only with a full cost ledger | Editor tokens, editor wall time, forward passes, GPU-seconds for validation, head or fine-tuning budget, and 650M inference cost on the same set. |

### 2.2 Mandatory before any "LLM editor" claim

- **M1 Reference points.** Untouched 8M and 650M (3B optional, expensive), same protocol, same protein set. Gives the scale of the gap being closed.
- **M2 Head-only tuning control.** Fit only the contact-head parameters (121 for 8M) on the training split under the same data and budget. If this closes most of the gap, the "weight editing" claim is a head-calibration claim.
- **M3 Equal-budget fine-tuning and distillation.** LoRA (or full) fine-tuning of 8M on the training split, and distillation from 650M contact logits on the training split. Budget equal in GPU-seconds and FLOPs, both disclosed. Distillation from 650M is the teacher ceiling: an editor-only edit that reaches it is a strong result; one that does not, is not.
- **M4 Random-direction edits** matched to the editor's edit in tensor support, per-tensor count, and norm in the **stored dtype**, under a pre-registered number of seeds. Include a head-only random control.
- **M5 Single-draw best-of-N.** Independent LLM proposals from the untouched model, no adaptive transcript, N equal to the number of validated editor candidates, selected on train/val only. Random-proposal variant too.
- **M6 Non-LLM search over the same support and budget.** *Added by Board A; not in the list the user gave.* Without it, "LLM editor" is not separable from "any search over these weights". Gradient-based search is allowed here only as a control, with its gradient budget disclosed.
- **M7 Sham reload and round trip.** Export/reload leaves tensor hashes and logits unchanged; a zero-edit edit gives zero metric change.
- **M8 Episode replicates.** A pre-registered number of independent editor runs per arm, intent-to-treat. Failed or invalid runs stay in the denominator.
- **M9 Equal-protein paired comparison** with cluster-bootstrap CIs, and one locked test evaluation per committed candidate. No selection on test results.

Recommended but not mandatory: editor without weight access (prompt-only proposals); a second family (ESMC) under its own pre-registered head protocol; a post-cutoff test subset.

---

## 3. Cheating and confounding routes specific to this setting

| Route | Specific mechanism here | Check |
|---|---|---|
| Editor sees held-out structures, contact maps, or 650M outputs on held-out proteins | Editor could read the test set, or query 650M/3B on test proteins and copy the result into an edit. | Mount audit (editor cannot read test path, caches, or large-model weights); deny network; log every file read and tool call; any access invalidates that arm. Grep editor transcripts and deltas for test-set identifiers and test-sequence fragments. |
| Editing the contact head only, called "weight editing" | The contact head has 121 (8M) parameters. Refitting it is a logistic regression over attention features. Refitting it on held-out structures is direct label leakage. | Parameter ledger by module. Any edit confined to `contact_head` is labelled "head calibration", and M2 is mandatory. Head fit only on the training split. |
| Memorising PDB IDs or sequences | An edit could encode per-protein corrections keyed to test identity (e.g. by sequence hash), or the editor could recall published contact maps. | Gain versus maximum identity to the training set, stratified by cluster; test on a post-cutoff subset; blinded prompt that does not name the test proteins; audit deltas for constant patterns that match test-protein properties. |
| Exploiting P@k definitions | (a) Denominator substitution in empty bins (R7). (b) Resolved-length L instead of full L. (c) Contact threshold or Cβ/Cα choice. (d) Saturated probabilities create ties at the cut, and index-order tie-breaks fill top-k with favoured pairs. (e) Edits that change APC/symmetrisation behaviour. (f) Choosing the best of 9 cells. (g) Short-range helix-like periodicity inflating short-range cells. | Freeze the definitions and tie-break rule in the pre-reg. Report tie counts at the cut; randomised tie-break sensitivity. Sequence-separation-only baseline per cell and a secondary-structure-free baseline. Report all 9 cells with Holm. |
| Selection on the test set | Best-of-N chosen by test P@k; editor retried until the test passes. | Visible/val selection only; log all attempts; ITT over episodes. |
| Asymmetric pretraining leakage | 650M/3B see more of the test-set distribution than 8M (R9, TV), biasing the "matches 650M" comparison. | Per-protein identity to each model's pretraining corpus; cutoff-matched subset; report the comparison on the subset with no pretraining identity above the pre-registered threshold. |
| Editor prior knowledge | The editor LLM may recall ESM contact literature, which the Qwen/Gemma blog setting does not offer in this form. | Blinded task prompt; record whether the editor names ESM-2 internals; compare to a no-name prompt. |
| Protocol mismatch across families | ESMC has no shipped contact head (R5); a head trained differently makes the comparison a protocol comparison. | One head protocol for every model; report the head protocol as a factor. |

---

## 4. Value to our hypotheses

### 4.1 What the cheap metric buys

- **Evaluation noise is removed.** Board A's qualification of the LLM fast tier rests on cross-device spread on sampled generations (RESULTS.md) and item-count SE of 0.03–0.10 at 25–100 items. A deterministic contact metric does not have that sampling problem.
- **The H5 power problem is mostly a cost problem.** Board A's pre-reg plans 30 paired episodes per arm per budget (90 per arm) and 32k-token sealed scoring, which is hundreds of GPU-hours. A contact metric over a few hundred clusters takes far less. More episodes are affordable, so episode variance can be estimated directly.

### 4.2 What it does not buy

- **Editor variance is the bottleneck, and it is untouched.** The metric's cheapness does not reduce between-episode variance. That variance must be measured, and it is likely to be the largest term.
- **The H5 efficiency endpoint breaks.** The pre-reg's endpoint is "≥30% fewer real evaluations". When evaluation is cheap, that endpoint is insensitive to the editor's real cost. An ESM sidecar would expose this. The efficiency endpoint must be redefined as cost-weighted (editor tokens + evaluation GPU-seconds + head or fine-tuning budget). **This redefinition matters for the LLM paper too, not only the sidecar.**
- **H1 and H3/H6 are not tested by one protein family.** H1 needs a predictor that reads weight deltas across families; an ESM zoo is one encoder family on a different modality. The predictor's inputs (including bf16 rounding statistics and shape-agnostic features in H6) would need redefinition for ESM checkpoints; the checkpoint dtype must be verified (**TV**). An ESM zoo can add a cheap labelled edit set, but it cannot establish cross-family transfer for LLM weights.
- **Focus.** The paper's claims are about LLM benchmarks (Qwen 0.8B, abliteration, Heretic, the blog's Experiment 8). A protein sidecar changes the paper's subject. It would weaken the focus unless it is framed as a separate, pre-registered secondary study.

### 4.3 Verdict

**Conditionally admissible as a clearly labelled secondary sidecar, not as the main H1/H3/H5 test bed, and not as evidence for the LLM-editor claim.** Its real value is testing the H5 accounting: power, episode variance, and cost-weighted efficiency. It should be run only after the pre-registration below is committed.

### 4.4 Minimal pre-registration (if run)

- **P1 Question and claim tier.** Which of the section 2.1 claims is being made; "matches 650M" only with P-equivalence below.
- **P2 Protein set.** Source, date cutoff, identity clustering method and threshold (TV for convention), n clusters per length and range bin, minimum eligible pairs and minimum k per cell, exclusion rules.
- **P3 Metric.** Contact definition, L definition, k rule, denominator convention, fp32 scoring, tie-break rule, 9 cells, **primary = long-range P@L/5**, others Holm.
- **P4 Models and head protocol.** Pinned checkpoints and revisions; one head protocol for all models; parameter counts (121/661/1441 for ESM-2 8M/650M/3B, R4).
- **P5 Leakage checks.** Identity of each test protein to each model's pretraining corpus; post-cutoff subset; the threshold above which a protein is excluded.
- **P6 Editor.** Pinned model id, token ceiling, tool policy, sealed goal prompt (hashed before running), no test access, log all reads and calls.
- **P7 Arms.** M1–M9 with budgets; random-direction seeds; N for best-of-N.
- **P8 Seeds and replicates.** Registered editor-run count per arm; registered identity clusters.
- **P9 Statistics.** 10,000 cluster-bootstrap replicates; crossed episode × cluster bootstrap; Holm across the 9 cells; TOST margin for "matches" fixed before running, set from the pilot σ_d.
- **P10 Pilot.** Measured σ_d on controls, then the set size from section 1.4; if the set cannot reach the target MDE, the cell is "underpowered", not "no difference".
- **P11 Cost ledger.** Editor tokens and time, forward passes, GPU-seconds, head/fine-tuning budget, 650M cost on the same set.
- **P12 Reporting.** ITT, all arms and failed episodes, null edits, the cross-device panel, the tie-count table, and the wording rule: "no effect resolved at this precision" where intervals span the margin.

---

## 5. Cited statements requiring verification before use in the paper

- R1, R9: ESM-2 pretraining corpus and version (UniRef50 2021_04 vs September 2021 labelled differently). TV against the Lin et al. supplement.
- R10: whether the contact head was trained on PDB structures. TV; not found.
- R5, R6: ESMC cards (no shipped contact head; 70% clustering; custom non-commercial licence for 600M). TV against the HF cards directly.
- R7: CASP bands and denominator substitution. TV against the official CASP assessment and arXiv 1609.00680.
- R8: ProteinNet 30% clustering procedure. TV against PMC6560865.
- R11: MMseqs2 citation. TV, not searched.
- Perplexity and MSA-depth relationship (section 1.3). TV against Lin et al.

R2–R4 (configs and head code) are the only parameter-level facts used in the arithmetic; re-read them at the pinned revision before the pre-reg is frozen.

---

## 6. Summary

1. **Branch:** `esm-board-a` (worktree `.worktrees/esm-board-a`). Doc: `docs/reviews/ESM/board-a-esm-2026-11-04.md`.
2. **Verdict:** REVISE. Conditionally admissible as a secondary, pre-registered sidecar; not the main H1/H3/H5 test bed; no "LLM editor" claim admissible yet.
3. **Mandatory controls:** untouched 8M and 650M reference; head-only tuning (121 params); equal-budget LoRA/fine-tuning and 650M distillation; norm/support-matched random edits; single-draw best-of-N; non-LLM search (added); sham reload; episode replicates; equal-protein paired CIs.
4. **Noise design:** deterministic metric (fp32, tie counts, cross-device top-k panel); cluster bootstrap over identity clusters; editor-run replicates as a crossed random effect; n ≈ 78,500·σ_d² for a 1-point paired difference, with σ_d to be measured in a pilot.
5. **Biggest cheating risk:** head-only refit (or any edit confined to the 121-parameter contact head) reported as weight editing, combined with test-set access through the editor's sandbox; the next largest is P@k definition choices (ties, denominators, L).
