# Round 1: representation (7 Nov 2026)

## BLUF (10 lines)

1. Pooled v4 statistics are near sign-blind and base-agnostic. Their benchmark rank is mostly family plus norm: descriptor-only trails StatNN by 0.042/0.035 (`evidence-statnn-importances.md`), and H1w θ̂ = 0.007 (`docs/reviews/P5/h1w-gate-2026-11-06.md`).
2. What Δ carries that pooling drops: its **orientation relative to what the base is sensitive to**. To first order that is ⟨∇L_k(θ₀), Δ⟩ plus a curvature term.
3. M1 never saw that orientation. Its direction tokens use a random JL frame (`src/wsp/features/svd.py::_proj`), so relevance had to be learned from ≤345 labels. I read M1 < tree as a failure of the **frame**, not of the network.
4. **Primary structure: a base-anchored sensitivity field.** Label-free and computed once on the base: per-objective gradients on disjoint dev panels, plus K-FAC curvature. Per-tensor tokens [⟨g_k,ΔW_t⟩, quadratic term, norms, layer, role] are used zero-shot and in StatNN+field.
5. Alternatives: (A1) a probe-response encoder (edited model on 64 fixed prompts), which is also the mandatory probe baseline; (A2) W2T/LoL canonical factors in a base frame instead of a random one.
6. The one clean **structure** negative is H1w, and it is specific to pooled statistics. Result 1 is mostly **data composition** plus 0.01-step labels. Result 2 is mostly an **estimand** problem (between-family offsets).
7. **Decisive test** on the D3 TF-v1 development panel, with WikiText (qualified) as primary. Pass: θ(StatNN+field) − θ(StatNN) ≥ 0.15, lower 95% > 0, and the real field beats norm-matched random and layer-permuted fields.
8. A 120-evaluation linearity ladder (scales 0.25/0.5/2) measures the field's validity radius. It separates "the Taylor structure is wrong" from "our edits are too large".
9. Cost **≤9.5 GPU-h**: 5.5 rescoring (already priced as P6d), ≤2 fields, 2 ladder. Contraction runs on CPU.
10. For H5, a field critic hands the loop the gradient information the blog's Experiment 8 forbade. It must beat a same-support gradient step and single-draw best-of-N at equal real-evaluation budget.

## 1. What the raw delta carries that v4 discards

`stats.py` emits 18 numbers per tensor: norms, cosine, mean/std/max, sign and nonzero fractions, the top-8 singular values, and entropy and stable rank computed **from the top 8 only**. `pool.py` averages these over 17 roles × 4 depth buckets, and the spectrum vector keeps only singular values. What is lost, by expected relevance:

1. **Sign and orientation relative to θ₀.** Benchmark Δ is signed, but the statistics are nearly sign-free (only `mean`, `frac_pos`). Whether an edit raises or lowers gold log-prob depends on the pair (θ₀, Δ). The pooled statistics consult θ₀ only through `rel_fro` and `cos_prev_new`.
2. **Curvature-weighted magnitude.** Within-family ρ(‖Δ‖, ΔMMLU) is −0.55 for chain, −0.51 for lora_sft, and 0.06 for random_dir (`docs/council/2026-11-05/metrology_budget_output.txt` l.240). Norm informs only where direction is not random.
3. **Cross-layer coherence.** random_dir and abliteration are exactly rank-one per tensor, ΔW = −w·d(dᵀW) (`src/wsp/zoo/random_dir.py`, `src/wsp/edits/kernels.py`). A global-index abliteration writes the *same* d into every supported layer, and pooling cannot tell one direction in ten layers from ten unrelated directions.
4. **Exact layer and spectral tail.** A bucket spans six layers. Stable rank ignores energy beyond rank 8, but LoRA rank goes up to 32 (`configs/zoo_qwen3.5-0.8b.yaml`).
5. **Incremental vs cumulative.** Chain features are base-relative (`extract.py` docstring), while labels are predecessor-relative (`docs/HYPOTHESIS.md`, P4 qualification).

**What the tree uses.** Top MMLU coordinates are bucket-0 (layers 0–5) down/v stable rank and o entropy (`evidence-statnn-importances.md`); random_dir edits only layers {8,12,15,17,19,21} (zoo config), so its bucket-0 slots are always zero. Inference: these act largely as family/support indicators, consistent with descriptor-only ≈ StatNN.

**Why M1 did not fix it.** M1 gets exact layer/type IDs plus [σ, UᵀP, VᵀP] tokens (`predictor/models/__init__.py::TensorEncoder`) with P a fixed Gaussian matrix: consistent across edits, silent on which residual directions matter to the base, so relevance must be learned from 297–345 labels. The 5 Nov council conceded pooling alone is not the cause (`round2-features.md`); my refinement is that **frame anchoring** is missing, and the base supplies it for zero labels.

## 2. Structure versus measurement

| # | Result | Tests | Why; separating measurement |
|---|---|---|---|
| 1 | Tree ≈ descriptors; no neural win | **Data composition + label resolution**. The neural part tests only a base-agnostic frame | η²_family = 0.345 on MMLU accuracy (`metrology_budget_output.txt` l.240). random_dir HellaSwag Δ ≡ 0, MMLU SD 0.007 at 0.01 resolution (`evidence-statnn-importances.md`). **Separator:** frame swap inside M1 (random-JL vs anchored tokens, same capacity and folds). |
| 2 | Family holdout ∈ [−0.21, 0.18] | Mostly **estimand** (offsets, `docs/RESULTS.md` P4) | A learned head cannot know a held-out family's mean. **Separator:** within-held-out-family Spearman after centring. A zero-shot field is the only proposal here that can predict offsets without labels from that family. |
| 3 | H2 | **Data** | 6 of 99 long lineages keep four leading deltas (`board-a-incident-d8-2026-11-06.md` §3). |
| 4 | H1a untested; H1w FAIL | H1a: **measurement** (max h 0.1137 > 0.10, `docs/HYPOTHESIS.md`). H1w: **structure, for pooled statistics only** | WikiText qualified (max h 0.0128 vs δ 0.0488), StatNN θ̂ 0.007, flexible −0.049 (`h1w-gate`). Silent on directional representations. |
| 5 | 94/300 non-null | **Measurement range**, and a structural prediction | Random d should carry near-zero Fisher weight. A field should predict the nulls zero-shot, which is falsifiable. |

## 3. Primary structure: a base-anchored sensitivity field

For a fixed base θ₀ and objective L_k (MMLU gold-letter log-prob, HellaSwag likelihood, WikiText NLL, refusal margin):

Δy_k ≈ ⟨g_k, Δ⟩ + ½ΔᵀH_kΔ,  g_k = ∇L_k(θ₀),

with H approximated per tensor by K-FAC. K-FAC approximates layer Fisher blocks "as being the Kronecker product of two much smaller matrices" (arXiv 1503.05671, abstract). I take the factor identities, input covariance A_t and output-gradient covariance G_t, from the method; the abstract does not name them.

**Consumes.** Raw effective Δ: dense, or canonical factors for LoRA and rank-one kernels, where contraction costs O(r(d_in+d_out)). It also consumes base-only fields frozen before any label is seen:
- g_k on **disjoint dev panels**: 512 MMLU-Pro items outside TF-v1's 1,000, the fast-tier 100 and the sealed partition; 250 HellaSwag items; 64 WikiText windows; 50+50 refusal prompts. Hashes are committed first.
- K-FAC (A_t, G_t) with model-sampled labels on the WikiText dev windows. Factors above 1,024 dimensions keep their top-256 eigenpairs.

**Per-tensor token.**
τ_t = [⟨g_1..4,ΔW_t⟩, q_t = ½tr(G_tΔW_tA_tΔW_tᵀ), ‖ΔW_t‖_F, ‖ΔW_tA_t^{1/2}‖_F, layer, role].
For chains, add the K-FAC cross term c_t = tr(G_tΔW_t^{prev}A_tΔW_t^{incᵀ}). This is the second-order route by which history enters, and a concrete H2 feature for the regenerated lineages.

**Predicts.** Predecessor-relative TF-v1 endpoints, in three readouts:
(a) **zero-shot** Σ_t⟨g_k,ΔW_t⟩ + Σ_t q_t; (b) StatNN on [v4 ‖ summed tokens ‖ per-role/bucket sums] (a tree win counts as representation evidence, H1b rule); (c) a ≤10k-parameter DeepSets head, only if (b) passes.

**Inductive bias.** Physics-style sensitivity: additive at first order, quadratic correction, explicit validity radius; a parameter-indexed field as in JEPAWG, derived from the base's loss geometry rather than learned.

**Why it should beat StatNN + descriptors here.**
1. Labels calibrate 2–3 coefficients instead of learning a 64-d relevance frame from 345 rows.
2. It predicts the **sign** of within-family effects at matched norm, exactly where isotonic norm and pooled statistics tie (the θ estimand).
3. A lora_sft edit is roughly the negative accumulated SFT gradient on gsm8k_train or alpaca (zoo config), so ⟨g_MMLU, Δ⟩ is a gradient-alignment transfer score. LoRA is where the D3 MMLU signal already sits: θ = 0.230 [0.014, 0.400] (`h1a-companions-2026-11-06.md`).
4. It predicts family offsets from Δ alone, the only route out of result 2.

**Failure mode.** Large edits leave the linear regime. Ortiz-Jimenez et al. (arXiv 2305.12827) treat linearisation as an intervention that "amplifies weight disentanglement"; their abstract does not claim ordinary fine-tuning is linear. Hence the ladder.

## 4. Alternatives

**A1: probe-response encoder (mandatory probe baseline).**
- *What it does:* runs the edited model on 64 fixed disjoint prompts; records Δlogits on the base top-64 tokens at the last 16 positions and Δ residual RMS at layers {6, 12, 18, 24}; encodes with a set model or tree.
- *Mechanism:* it observes the nonlinear function, so it has no validity radius. This is what the blog's Experiment 7 needed: after weights-only failure it succeeded with "only the attention patterns … when both models are run on texts that Astra writes itself" (https://rikiyat.github.io/blog/weight-space-programming/).
- *Cost anchor:* the TF-v1 WikiText stage scores 64 × 1,024 tokens in 4.2 s warm (`docs/RESULTS.md`, round 3).
- *A1-lite, free:* the 200-prompt TF-v1 per-prompt KL vector as features.
- *Reading:* A1 ≫ field: behaviour must be sampled; field ≈ A1: weights plus base suffice.

**A2: base-frame canonical factors.**
- *What it does:* keeps W2T's QR-then-SVD canonical (u, v, σ) tokens with rank- and position-level Transformers (arXiv 2603.15990 §3–4). It replaces our random `_proj` with base frames: C_t = U₀ᵀΔW_tV₀ (top-16, the 5 Nov proposal in `round2-features.md`), refusal-direction projections (`src/wsp/edits/refusal.py`), and unembedding top directions.
- *My addition:* diag(C_t) = δσ_i, the first-order shift of the **full** matrix's singular values. Full-matrix spectra are what the heavy-tail literature ties to quality. Wilson et al. report a data-free spectral free energy with ρ = −0.77 against Qwen3.5 MMLU-Pro across six sizes (arXiv 2610.13123 §3.3). It is derived for final-layer weights and not tested on fine-tuning, so for edits this is a hypothesis.
- *Mechanism:* signed and anchored, with no linearisation, so it may win where the field breaks.
- *Data caution:* W2T trains on about 10k labelled LoRAs per task. LoL's SVD-feature MLP is competitive on some tasks and far behind on others (arXiv 2410.04207, Tables 3–4 per fetch summary). Neither covers dense deltas. At n ≈ 300, use A2 as tree features, not as a learned encoder.

## 5. Decisive experiment

**Data.** The D3 TF-v1 development panel: 229 stratum rows (139 chain, 60 lora_sft, 30 task_vector) + 157 null-family controls, 120 components (`h1w-gate`).
- The records must be rescored after D8: "239 non-chain + 77 chain = 316 … ≈ 5.5 GPU-h" (`board-a-incident-d8-2026-11-06.md`, P6d).
- The other 62 chain rows come back via `scripts/d8_chain_replay.py`. If the replay is incomplete, run on the recovered panel and label it.
- No sealed/fresh component touched.

**Labels.**
- *Primary:* predecessor-relative WikiText log-PPL (qualified). g_wiki comes from 64 different windows.
- *Descriptive:* MMLU-Pro gold log-prob (1,000 items, unqualified) and HellaSwag (250).
- *Validity only:* KL, against the Fisher term.

**New rows: linearity ladder.**
- 20 lora_sft + 10 abliteration + 10 random_dir D3 edits, picked by sorted ID within norm tertiles before outcomes, at s ∈ {0.25, 0.5, 2}.
- 120 TF-v1 evaluations at about 1.0 min each (`docs/RESULTS.md`) ≈ 2 GPU-h.
- Prediction: Δy(s) ≈ s⟨g,Δ⟩ + s²q, measured per family.

**Split and estimand.** Registered H1w machinery: 5 inner folds by ancestry component, train-only family + device + log-norm nuisance, θ = 1 − Σ|r − r̂|/Σ|r − r̂_iso|, 4,000-draw component bootstrap with the six interval variants.

**Arms** (same rows, folds, seeds 0/1/2 and tuning budget):
1. StatNN
2. descriptor-only
3. delta-norm
4. **field zero-shot**
5. **StatNN + field**
6. StatNN + norm-matched random field
7. StatNN + layer-permuted field
8. A1-lite (KL vector + StatNN)

**Pass** (all required):
- WikiText θ(5) − θ(1) ≥ 0.15, paired lower 95% > 0.
- Arm 5 beats arms 6 and 7 (paired lower > 0).
- Arm 4 within-family residual Spearman lower 95% > 0 in lora_sft and chain.

**Power.** Power at θ = 0.15 on this panel is 0.73 conditional and 0.60 joint (`h1w-gate` §1). So 0.15 is the pass line and 0.05–0.15 is inconclusive, not a near-miss.

**Kill.**
- Upper 95% gain < 0.05; **or**
- no win over the random field (anchoring is just extra norm features); **or**
- ladder sign agreement < 0.7 at s = 0.25 in lora_sft (the field itself is wrong).
- Holds at s = 0.25 but fails at s ≥ 1: structure right, edits too large; move to A2/A1, not bigger encoders.

**Cost.** P6d rescoring 5.5 GPU-h; fields ≤2 GPU-h (cap, to be timed); ladder 2 GPU-h; contraction on CPU, capped at the 54 s/entry extraction budget (brief). **Total ≤9.5 GPU-h.**

## 6. Models editing models (H5 `llm_edit`)

**Shape of an LLM edit.** Like Experiment 8: "one direction written by the MLPs of layers 15 and 19 … subtracted … from 64 output columns in each (131,072 weights)" (blog). Sparse low-rank edits are the best case for contraction (cost ∝ support) and the worst for linearity (large coefficients); repeat the ladder on the first 20 `llm_edit` proposals before trusting the critic.

**Gradient leak.** The post says "no training or gradient information is allowed". A field critic gives the loop exactly that first-order gradient. Any H5 claim must beat:
- (i) a **same-support projected gradient step** along g at matched norm; and
- (ii) single-draw best-of-N at equal real-evaluation budget (`docs/MODEL_CANDIDATES.md`, Meta^n note).
If (i) matches editor+critic, the finding is "gradients help", not "a world model critiques the editor".

**Rationale.** Keep it out of the primary critic. Descriptors already carry most of our rank through family identity (result 1). A rationale is a richer descriptor that encodes the editor's own belief, so reading it partly grades the editor by itself. Run rationale-only as a separate arm, and adopt rationale+field only if it beats field alone.

**Order statistic.** Picking the arg-max of N proposals by critic score creates a winner's curse whenever critic error correlates with the editor's search direction. That is guaranteed for a linear field, because big steps along g exit the trust region. Mitigations:
- Compute fields on panels disjoint from the scoring items.
- Make the critic abstain when q/|⟨g,Δ⟩| exceeds the ladder's validity ratio.
- Report calibration on selected proposals separately.

## 7. Blog honesty

Verified at the URL above: Experiment 7 failed "from the weights alone" in "all four attempts"; Experiment 8's direction-finding is not in the post (the activation-contrast account is from release logs, `docs/council/2026-11-05/round1-weight-reading.md`); "Auditing without evaluation" is a "What Comes Next?" bullet; no experiment predicts held-out benchmark deltas from weights. Because my primary structure uses base gradients, success must be reported as "weights + base sensitivity", not weight reading.

## Summary table

| structure | mechanism | decisive test | cost | kill criterion |
|---|---|---|---|---|
| **Base-anchored sensitivity field** (⟨g_k,ΔW_t⟩ + K-FAC quadratic + chain cross term; zero-shot, then StatNN+field) | Base supplies signed relevance geometry; labels only calibrate; predicts sign at matched norm and family offsets | D3 TF-v1 panel (229+157), WikiText primary: θ(StatNN+field) − θ(StatNN) ≥ 0.15, lower > 0; beats random and layer-permuted fields; 120-eval ladder | ≤9.5 GPU-h | Upper 95% gain < 0.05; or no win over the random field; or ladder sign agreement < 0.7 at s = 0.25 |
| A1 probe-response encoder | Observes the nonlinear function; no validity radius | Same panel and arms; A1-lite from TF-v1 KL vectors | ≈0 (lite); <0.5 GPU-h full (estimate) | No θ gain over StatNN |
| A2 base-frame canonical factors (U₀ᵀΔWV₀, δσ_i, refusal/unembedding projections) | Signed, anchored, no linearisation; links to full-matrix spectral quality | Anchored vs random-JL frame at equal capacity, in M1 and the tree | ≤30 s/entry CPU (estimate) | Anchored frame ties random-JL (paired lower ≤ 0) |
