# Round 2 — weight-structure readability of teacher-forced effects

Branch: `council-r2-reading`. This is a research recommendation, not an experiment result or authorization to spend. No model training, new labels, GPU jobs or cloud operations were run. All proposed effect sizes, counts and caps below are prospective criteria; none is a measured gain. Provenance: [round-2 manifest](round2-weight-reading/manifest.json).

The **weight-space world model framing must become “weight-structure readability of teacher-forced effects.”** The current evidence supports neither a learned dynamical law nor capability forecasting. Preserve historical H1/H2/H3 failures; register this narrower question prospectively. A useful positive outcome is a small structural feature improving a tree's within-family prediction of fixed-item likelihood effects. It does not require JEPA, an SSM, or an editor.

## What the new evidence changes

I read every file in this council directory, including the three analysis scripts, source manifest, CSV and metrology output. I checked `src/wsp/predictor/data.py` (descriptor and tensor-token paths) and `src/wsp/features/{pool,stats}.py`. The [StatNN report](evidence-statnn-importances.md) and [CSV](evidence-statnn-importances.csv) reproduce seed-0 trajectory predictions on **345 matched rows**: descriptor-only versus StatNN Spearman is 0.523855 versus 0.565906 on MMLU-Pro and 0.613582 versus 0.648602 on HellaSwag. The gaps are **0.042050 / 0.035019**, without paired uncertainty. Descriptor-only actually has higher pooled R² on both endpoints. These are comparisons of different estimators, not an additive or causal attribution of exactly 4% to weights. An equal-capacity descriptor-plus-weight ablation is missing.

On KL the corresponding Spearman gap is 0.204828 (0.493122 versus 0.697950), but descriptor-only again has higher R² (0.526138 versus 0.475709). On log-PPL the Spearman gap is negative, −0.014405. Ranking and calibrated error answer different questions. Gain importance is also not causal: `down.bucket0.stable_rank` leads MMLU importance, but that code uses top-eight energy, not full stable rank; `o.bucket0.spectral_entropy` leads HellaSwag, where a family fingerprint can suffice. No importance analysis establishes directional readability.

The [metrology output](metrology_budget_output.txt) concerns **300 base-relative D3 entries**, plus a separate 89-predecessor analysis. Its pooled population correlation ceilings are 0.9839/0.9891 for MMLU gold log-prob/margin, 0.9996 for KL and 0.9885 for log-PPL. These are heuristic square-root reliability bounds, not measured achievable Spearman. HellaSwag log-prob is lower (0.9655 pooled, 0.9099 family × device residual); do not extend “0.98–1.00” to every teacher-forced field. HellaSwag accuracy's residual SD 0.0131 is below item SE 0.0136: no aggregate within-family population signal clears this test. Sampled GSM8K/MATH/IFEval fail qualification at this tier.

Only **94/300 have KL > 0.01**, an operational non-null definition, not proof the other 206 have no effect. The informative subset is mostly LoRA (56) and chains (30). The predictor panel has random-direction HellaSwag identically zero and abliteration range −0.01 to +0.02; the latter is near-null, but not literally bounded by ±0.01 in every predecessor-relative row ([CSV](evidence-statnn-importances.csv), family distributions). The metrology table gives abliteration p90 absolute change 0.01. These distinctions matter: the 345-row panel includes predecessor-relative chain effects; the metrology ceiling cannot simply be attached to every predictor row. Family-level exceptions (e.g. task-vector HellaSwag) can coexist with the pooled residual failing qualification.

## Answers to every question addressed to weight reading

### JEPAWG researcher

> “can matched layer- and direction-sensitive features reveal behaviorally meaningful information that pooling destroys, and what is the cheapest decisive control?”

They can distinguish operators that pooling maps together; predictive value remains unproved. Cheapest first: recover exact-layer scalar features from existing files, compare with the same features shuffled **within each depth bucket**, retaining role, support and norm controls. This requires CPU cache reads and no new extraction. Head identity and base-relative direction need a second contrast: the compact reader/writer feature below versus the same computation with reader-neuron correspondence permuted. A positive geometry/reconstruction result alone is insufficient; held-out family-plus-norm residual error must improve. Judge task likelihood rather than constant HellaSwag accuracy. The missing evidence is a powered, paired comparison on qualified residual targets, not another representation's ability to identify edit family.

### Features CTO

> “which directional representation best preserves functionally meaningful coordinates while remaining invariant to permissible weight symmetries, and what falsifiable matched-norm test should we run first?”

For this fixed base, use contractions of the effective delta with the corresponding frozen base readers/writers, retaining exact layers. They avoid SVD sign and degenerate-basis ambiguity and are invariant to an equivalent LoRA factorization. Summing matched neuron contractions is invariant to simultaneous MLP neuron permutation of down columns and gate/up rows. This is not invariance to arbitrary residual rotations: RMSNorm gains, elementwise gates, RoPE and head layout restrict permissible transformations. Canonical factors alone do not align different base networks.

First run the cached layer-scalar control, then test the explicitly defined overlaps below against a reader-permuted control and a norm/support-only baseline. A directional claim requires the real pairing to beat the destroyed pairing on held-out teacher-forced residuals. Norm matching alone does not control support, task, ancestor or bf16 rounding; include them. There is no measured reason to promise the representation will predict sampled reasoning scores.

### Metrology CTO

> “Within family, ‖Δ‖_F alone gives ρ = −0.51 on chain and −0.43 on lora for mmlu gold log-prob (`metrology_budget_output.txt`). Name the per-layer statistic you will commit to before seeing labels, and state the within-family ρ you expect it to add over ‖Δ‖_F on mmlu margin. Chain has n = 50, so the 95% CI on a ρ near 0.5 is about ±0.2 (Fisher z); what result would you accept as evidence?”

Commit to signed down-update/up-reader overlap, `c_up,l = <ΔD_l, U_0,l^T>_F / (‖ΔD_l‖_F ‖U_0,l‖_F)`, with the auxiliary features and masks frozen below. My planning expectation is **+0.03–0.08 within-family Spearman on MMLU margin**, central continuation target +0.05, with zero entirely plausible. This is an uncalibrated hypothesis, not an extrapolation of the 0.04 accuracy gap. Compare full predictions to family-specific norm curves and measure residual association separately; the two correlations are not interchangeable.

Accept provisional evidence only with ≥5% paired residual-MAE reduction, positive paired ancestry-cluster 95% interval, positive residual rank association, and loss of the advantage under reader permutation. Require a fresh item panel and independent ancestry replication for confirmation. The existing n=50 chain panel cannot establish a modest +0.05 gain through a favorable point estimate. An interval spanning zero means inconclusive and no scaling at the cap, not proof of impossibility. Metrology's independent-correlation power approximation is not power for the paired improvement between two predictors; estimate the latter with observed paired cluster losses before choosing any additional sample count.

### Strategy CTO (joint question with features)

> “can we hold layer/support and singular directions fixed while changing their orientation relative to base weights, producing matched pooled norms/spectra but different qualified task effects? What single direction-preserving feature comparison would distinguish representation loss from insufficient zoo signal within the two-week cap?”

Not literally with singular directions fixed in base coordinates: changing orientation changes those directions. We can fix layers, support pattern and singular **values**, while permuting residual coordinates of a writer delta relative to the frozen base. A row permutation within supported rows preserves Frobenius norm, singular values, column energies and scalar delta marginals. It generally changes `cos_prev_new`, relative alignment and functional effects; do not call the full pooled vector matched. Constrained supports may permit too few nontrivial permutations; record that and reject the pair rather than inventing a control. Recompute all effective bf16 norms/spectra after applying and rounding, and reject pairs beyond a preregistered numerical tolerance.

The single comparison is **same small residual learner with true versus reader-permuted base contractions**, conditioning on exact-layer scalar features, descriptors and norm. The cache-level control costs no new labels; it tests whether pair identity adds information. A later paired edit pilot would test whether changes in actual orientation cause qualified TF effects. If the real features beat destroyed pairings on a reliable target, the pooling explanation gains support. If all variants fail despite resolvable residual variation, these features fail; if residual variation itself is below item uncertainty, the zoo/measurement question remains unresolved. Neither failure proves that full weights are unreadable.

## Specific challenges and concessions

**Features CTO — token expansion is not the missing control.**

> “the neural models consume pooled vectors (M1/M2), or cumulative pooled vectors (M3); they cannot recover directions or tensor identities already averaged away.”

> “Unpooled per-layer role tokens with positional encoding. … **First add.**”

`FeatureDataset.entry` already reshapes all eight projected SVD components into tokens and supplies `types` and exact `layers`; my [round-1 audit](round1-weight-reading.md), section 2, identifies the consuming M1/M5 encoder path. The matched pooled comparator and token model must be distinguished by actual run/config, not lumped together. Some controlled pooled models do lack this information; saying all M1 failures do is wrong. The missing control is same-capacity, same-label, true-versus-destroyed base alignment, after descriptors/norm and exact-layer scalars. I agree pooling loses directions and counts and canonical LoRA factors do not solve dense-delta truncation. I retract my own round-1 demand for gains on **both binary benchmarks**: HellaSwag supplies no suitable residual success target here.

**JEPAWG researcher — reconstructing geometry can preserve only nuisance variation.**

> “Add a compact encoder trained on masked pooled features or genuine predecessor→successor pairs with VICReg variance guard, then freeze and fit a small supervised head”

> “Prediction: at least +0.05 paired ρ on both deterministic targets”

Variance preservation and held-out reconstruction can reward family, support and dose, exactly the signals that descriptors already capture. The new accuracy gaps are only 0.042/0.035, and HellaSwag accuracy has zero residual ceiling under metrology's test. A +0.05 pooled success could still be a better family classifier. Condition geometry tests on family **and norm**, require teacher-forced residual prediction beyond scalar/tree controls, and retain ancestor exclusion during unsupervised training. I concede their correct distinction between physical parameter-indexed populations and our arbitrary edit sequences; no observed smooth control chart in this zoo warrants my borrowing the world-model rhetoric. Keep their order-shuffle idea parked until a qualified residual and enough independent long runs exist.

**Metrology CTO — KL should be a structural target, with restricted interpretation.**

> “KL is a validity probe, not a target. … Scoring a predictor on KL tests whether it can read ‖Δ‖.”

The first sentence is a reasonable rule for *capability* claims; the second is too strong. Within-family norm–KL ρ is 0.58–0.91, not unity, and the StatNN/descriptor KL ranking gap is 0.204828. Neither establishes structure beyond norm, but neither rules it out. Keep KL as a secondary TF-v1 structural endpoint **after flexible family × norm conditioning**. Predicting that residual would show anisotropic sensitivity, not competence. A difference of two base-referenced KL values on a chain is also not KL(edited || predecessor); label these quantities separately. I concede the CTO is right to demote KL-only wins from H1 capability evidence and to prioritize continuous scores. My round-1 proposal overstated what improving a sensitivity proxy could establish.

**Strategy CTO — genuine trajectories need independent-run power, not just better provenance.**

> “eight datasets × three seeds × two schedules = 48 independent runs, five checkpoints per run = 240 endpoints.”

This is better data generation than random controls. It is not automatically 48 independent units for task transfer: the task-holdout question has eight dataset groups, and schedules/seeds share task/source ancestry. With only 94 operationally non-null existing entries, buying correlated checkpoints before the residual audit may increase row count rather than power. Their two-week pilot and timing gate are sensible; I endorse those conditions, not the six-week expansion as a default. I concede their point that history benefit can only compensate for compressed state: equal full current weights and next edit determine deterministic behavior.

## Revised feature and decisive protocol

### A small, weight-only reader/writer test

Freeze the base revision and exact layer mapping first. For each MLP layer let `D_0 ∈ R^(d×m)` be base down projection, `U_0,G_0 ∈ R^(m×d)` base up/gate projections, and `ΔD` the **effective** bf16 writer delta converted to float32. For each reader `R ∈ {U_0,G_0}` define:

- `c_R = Σ_j <ΔD[:,j], R[j,:]> / (‖ΔD‖_F ‖R‖_F)` — signed aggregate alignment.
- `q_R = Σ_j <ΔD[:,j], R[j,:]>² / (‖R[j,:]‖² + ε) / (‖ΔD‖_F² + ε)` — aligned energy without cancellation.
- `c_base = <ΔD,D_0> / (‖ΔD‖_F ‖D_0‖_F)` — signed base-writer overlap.

Add a presence bit and layer fraction of total delta energy: **seven scalars/layer, 168 for 24 layers**, a proposed schema. Normalize with explicit zero handling; masks distinguish absent from zero. The preregistered leading statistic is `c_up`, not whichever coordinate later ranks highest. Compute on effective deltas, not unrounded adapter factors. Contractions are O(dm) reductions; no SVD, learned encoder, activation cache, or task examples are needed. Pairing is invariant to shared neuron permutations and factorization choices, while permuting reader rows alone destroys it. If up/gate weights change substantially, flag that limitation; this first feature isolates writer changes rather than approximating the entire nonlinear MLP.

This tests whether an update writes into coordinates read by the same base neuron and whether signed cancellation matters. It is a structural hypothesis, not a functional Jacobian: SiLU gating, activation frequency and downstream propagation are absent. MMLU log-prob/margin are the primary task endpoints; KL residual is secondary. HellaSwag gold log-prob is secondary only in families passing their own residual-variance gate. No expected gain is claimed for random-direction binary accuracy or across bases. The refetched [Experiment 3 protocol](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp3_language/README.md) permits matrix products and reports weight-only head identification; it motivates examining relationships between matrices, not a quantitative forecast for this feature. The refetched [Experiment 7 protocol](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp7_sycophancy/README.md) records weights-only failure followed by attention-assisted success; input conditioning remains a plausible missing ingredient.

### Estimand, controls and stopping

1. Build a frozen join of entry, predecessor, family, device, item panel, ancestry and base revision. For chains use **incremental** delta features alongside predecessor cumulative state; do not regress predecessor-relative labels on an unlabeled mixture of cumulative quantities. Existing base-relative metrology labels support an exploratory screen; new TF-v1 letter scoring (1,000 MMLU items), HellaSwag log-prob (500) and cached-base KL (200 prompts) are new prospective endpoints. Gold-solution NLL and refusal margins stay pilot-only as the CTO specifies.
2. Within each outer training fold, fit `m0 = family × spline(log(ε+‖Δ‖_F)) + device/reference terms`, with train-only tuning and shrinkage for small cells. Use consistent incremental/cumulative norms for each estimand. Generate **inner out-of-fold** training residuals `r=y−m0`; fit the small residual learner there, then refit m0 on all outer training data. On held-out rows evaluate `r_test=y_test−m0_train` and `m0_train+g_train`. Never subtract test-family means estimated from held-out labels.
3. Compare identical learners/tuning budgets for descriptors-plus-norm, those plus original pooled stats, those plus exact-layer scalars, and those plus the five contractions. Include reader-permuted features, within-bucket layer shuffles and within-family/norm-bin label permutations. Add LoRA task/rank/steps and support metadata as stronger nuisance controls; the existing 32-bin hashed descriptor with Ridge may underfit those interactions. A gain only against that weak comparator is not structure evidence.
4. Use ancestry-cluster outer CV, grouping complete chains and all shared source adapters; hold out both parents and descendants. Three fixed full seeds (0/1/2), train-only preprocessing, matched folds for every arm. Headline within-family out-of-fold residual MAE and residual Spearman; also report each family's full-prediction Δρ over m0, not a pooled correlation dominated by offsets. Bootstrap paired losses by ancestry cluster and assess item uncertainty jointly, preserving shared base/reference draws; use block resampling for text windows. Constant ranks are undefined, never zero successes.
5. Expected unmeasured margin gain is +0.03–0.08 Δρ and roughly 3–5% residual-MAE reduction; set continuation at **≥5%** MAE reduction with positive cluster interval and direction-control specificity. KL planning target is **5–10%** residual-MAE reduction, not +0.20 incremental correlation. These are modest bets with substantial failure probability. Adequate family × norm residual reliability has not yet been measured; re-estimate it before judging the feature. If intervals remain wide, stop scaling and report an inconclusive screen. Confirmation requires an untouched item/ancestry panel, with sample size set from paired pilot losses.

The literal no-forward claim survives only for the feature arm (base weights plus delta); producing labels still needs evaluation. Cached-input replay is a separate assisted comparator. If it succeeds where the weight feature fails, the supported claim is cheap activation-assisted prediction, not zero-forward auditing.

## Ranked actions for the chair (five maximum)

Costs are **local occupied-GPU-hour caps or explicitly marked estimates**, incremental cloud dollars **$0**; CPU engineering and local electricity are unpriced. No rental is proposed. Anchors: [brief](BRIEF.md) 66 s/extraction and 1–10 min/predictor run; [metrology](round1-metrology.md) TF-v1 1–2 min/entry is unmeasured. Existing work/caches are reused; no full-zoo extraction. These actions are sequential gates, not a funded batch.

| Rank | Action and owner | GPU-h / new $ | Falsifiable prediction | Kill criterion |
|---|---|---|---|---|
| 1 | Metrology CTO + features CTO: reconstruct cached continuous labels and fit the family × norm residual audit, equal-capacity descriptors/pooled/exact-layer controls. | CPU label recovery; predictor cap **0.5 h / $0**. | MMLU gold log-prob/margin retain residual variation above paired item uncertainty in LoRA or chains after dose conditioning. | No qualified task residual, or broken predecessor/item/reference join: stop predictive feature claims and repair measurement; no new encoder. |
| 2 | Metrology CTO: time and qualify TF-v1 on **10 preregistered entries** spanning null/moderate/drift, including chains; independent item panel and cross-device item records. | **0.17–0.33 h estimate**, **0.5 h hard cap / $0**, including reference/timing reserve. | MMLU continuous effects retain direction on the independent item panel and paired item uncertainty stays below the within-family residual spread. | New letter-scoring residual unresolvable, provenance fails, or timing exceeds cap: freeze revised scope before collection; do not extrapolate pooled ceilings. |
| 3 | Weight-reading researcher + features CTO: frozen 168-feature reader/writer comparison and reader-permutation controls on qualified cached/TF-v1 labels. | **33 s/entry extraction cap** (unmeasured), **2.75 h/300** plus **0.5 h** predictor cap = **3.25 h / $0** maximum. Ten-entry timing gate included, not additional. | ≥5% MMLU margin residual-MAE reduction and about +0.05 within-family Δρ over norm baseline; real pairing beats permuted pairing. KL secondary improvement 5–10%. | Cap exceeded, positive paired interval absent, or shuffled pairing equally good: stop this feature; no large token model or circuit-composition expansion. |
| 4 | Features + metrology CTOs: only after action 3 passes, generate **10 orientation-matched edit pairs** within declared supports; sealed TF-v1 scoring, bf16 matching checks. | 20×66 s = **0.37 h** extraction anchor; TF labels **0.33–0.67 h estimate**, generation/matching **0.25 h cap**, total **1.5 h cap / $0**. | At matched effective norm/spectrum/support, predicted orientation contrasts agree in sign with independently resolved task effects in at least **8/10 pairs**. | Fewer than 10 valid independent pairs within cap, fewer than 8 concordant pairs, or contrasts below item uncertainty: no causal structural claim. This is a pilot gate, not a powered population proof. |
| 5 | Weight-reading researcher: if qualified residual exists but weight controls fail, run one separate cached-activation-assisted writer-replay comparator on the same panel. | Base-cache **0.25 h cap** + 300×66 s replay cap **5.5 h** + predictor **0.5 h** = **6.25 h / $0**. | ≥5% task residual-MAE improvement beyond the best weight-only baseline, positive cluster interval. | No task gain, only KL gain, or cap breach: stop candidate-screening claim; publish measurement/negative result. Never describe this arm as weight-only. |

Any future job expected to exceed two minutes must use `agent-tmux`; actual timings replace caps before expansion. No 700-entry purchase, world-model scaling, paid editor, cross-base transfer or six-week trajectory programme is endorsed before these gates. The proposed actions are recommendations only; this commit changes documentation and provenance.

Branch: `council-r2-reading`.
Strongest challenges: features CTO—M1 already has direction/layer tokens; metrology CTO—KL residual structure must be tested before declaring it only norm.
Revised top action: recover continuous labels and audit family × norm residual signal with ancestry-cluster controls before buying features or labels at scale.
Changed my mind: binary HellaSwag gains cannot be a success condition; the world-model framing must narrow to teacher-forced readability.
Chair adjudication: does a KL-only structural win merit a separate sensitivity result, or must every continuation require an independently qualified task-likelihood gain?
