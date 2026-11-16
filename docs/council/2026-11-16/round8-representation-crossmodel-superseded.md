1. Use functional tensor roles and dimensionless contractions, not aligned neuron coordinates.
2. Architecture-agnostic means a common interface; unsupported operations must remain explicit.
3. Preserve signed task derivatives alongside normalized spectral and displacement summaries.
4. Cache base-only activations, objective gradients and model-sampled Fisher factors separately.
5. Never describe Fisher magnitude as signed Hessian curvature or a behavior-rate derivative.
6. Recompute exact edit arithmetic on CPU; predict changes to functional state without edited inference.
7. Compare dynamics against a cumulative static predictor and a quadratic interaction baseline.
8. Match information in separate forward-free and base-calibrated reader comparisons.
9. Begin with size transfer within one family; reserve family transfer as a separate challenge.
10. Run a representation contract first; current evidence does not justify buying trajectories.

## Evidence and decision

The [round-7 synthesis](SYNTHESIS-r7.md) and [representation report](round7-representation.md) motivate preserving signed field terms. The subsequent [MSRM development report](../../reviews/msrm/msrm1-dev-v0-2026-11-16.md) reports C=+0.378 versus field, StatNN +0.430, and k2b firing: C(MSRM:scripted)=+0.258 [−0.094,+0.440]. This is unresolved added value, not proof that fields are useless. There are 29 closure groups behind 123 rows; the reported family holdout is explicitly not closure-independent. These are exploratory single-base results, not cross-model evidence.

Fetched 16 November: [required abstract](https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models), [paper](https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models.pdf), [repository](https://github.com/suayptalha/wsr), and linked source below. Initial PDF open returned “Internal Error”; following the abstract's PDF link succeeded. The paper motivates architecture-aware controls: post-sublayer normalization can cancel projection scaling, and reader false alarms on inert edits were substantial (paper §6). The repository explicitly excludes paper results; no reusable outcome dataset is established by that release. These sources do not establish learned cross-family transfer.

## Q2: a shared interface, not universal coordinates

The current implementation needs a new schema. `src/wsp/msrm/features.py` fixes 24 layers and 186 slots. `features/pool.py` merges mixer outputs into `o`, embedding and unembedding into `embed`, and places non-layer tensors in bucket zero. `features/response_chart.py` flattens architecture-specific slots and uses name-seeded random projectors. `features/svd.py` projects singular vectors into dimension-dependent random frames: equal projected width does not align neurons across models. `features/stats.py` reports top-eight truncated stable rank. Retain these as legacy baselines, not transferable semantics.

**Proposed state token:** one per logical operator, with a shared scalar schema and variable token count. Include functional role (residual writer, query/key/value, gate, expansion/contraction, norm, embedding, unembedding, router, recurrent-state parameter), mixer subtype, normalization placement/gain convention, tied-parameter flag, and connectivity. Split fused projections only through a verified layout adapter. Record relative depth i/max(L−1,1), log dimensions, aspect ratio and head/KV ratios. Embed/final-norm tokens have a global-position flag, not early-layer depth. Pool expert summaries symmetrically with usage weights and retain router identity; initially abstain on unsupported expert/fused operations.

For each matrix store RMS, normalized leading singular energies, spectral tail mass, and full stable rank ‖W‖²_F/σ₁². For repeated singular values use subspace projector overlaps, not individual vectors. Include base activation RMS, normalized covariance-spectrum summaries, and objective-gradient/Fisher summaries with masks. Keep model size and baseline score as explicit covariates; ablate them to expose size shortcuts. A set encoder must be invariant to token enumeration while retaining connectivity and depth. Shared output width alone guarantees neither transfer nor information sufficiency.

**Proposed action token:** join to the state operator; use realised stored-state displacement D, not raw LoRA factors. Include relative norm, cosine(D,W), rank/min(shape), stable rank/min(shape), singular-energy fractions, changed-row/column fractions, delta-energy share and realised/intended norm. Pin zero denominators and distinguish missing intended delta from zero. Include signed aₜ=〈gₜ,Dₜ〉 and qₜ=½tr(GₜDₜAₜDₜᵀ), both raw and normalized by their norm products. Preserve raw sums for endpoint scale; use train-model-only transforms. Zero-gradient cases get validity flags.

Use within-model subspace overlaps for alignment; never compare singular-vector coordinates across bases. Unembedding alignment applies only in the corresponding residual space, with final normalization declared. Prefer task-defined token-group directions over the vocabulary mean. Token IDs do not transfer: map frozen strings or semantic groups separately through each tokenizer, retaining missing/multi-token flags. Give those identities to every comparison arm. Add probe-space action energy tr(DADᵀ)/tr(WAWᵀ), with support flags; optionally signed probe contractions δᵀDx. These use cached base activations/sensitivities, not edited activations.

This offers permutation-respecting summaries and task-anchored coordinates, not invariance to arbitrary invertible reparameterizations. Test valid function-preserving permutations on toy networks; do not impose symmetries broken by nonlinearities, normalization, or rounding.

## What base-only passes compute

Freeze disjoint calibration/probe text groups, objective definitions, weighting, chat templates, precision, tokenizer and model revisions before labels. Same text transfers across tokenizers; identical token IDs do not. Preserve TF-v1 ΔNLL within each model, but report models separately because nats/token scales differ with tokenization. Add byte-normalized NLL as a separately named transfer diagnostic, not a replacement endpoint.

1. **Forward collection:** fixed teacher-forced prompts, valid-position masks, base objective values, input/output activation norms and second moments at supported operators, normalization statistics, and token-group logits. Store counts and content hashes. Base generation length may be an optional separately priced feature; teacher forcing alone does not measure thinking length.
2. **Objective backward collection:** gradients of exactly normalized NLL, gold-answer log probability, or separately named refusal/sycophancy margin surrogates. Keep harmful/harmless objectives distinct. These are auxiliary differentiable objectives, not derivatives of sampled accuracy, a judge's refusal label, or reasoning length.
3. **Fisher collection:** independently sample categorical targets from base logits; backpropagate summed sampled-token NLL and accumulate A=E[xxᵀ], G=E[δδᵀ] on valid positions. Retain traces, damping/truncation audits and gradient/factor coverage. q is a nonnegative block approximation, not a Hessian or a guaranteed upper bound.

`features/response_field.py` already separates objective gradients from a second model-sampled Fisher forward/backward, validates panel disjointness, and rejects missing Linear hooks. Its contraction represents unsupported q as None and omits cross-tensor blocks. Preserve those distinctions; dense gradients may exist where factors do not. Do not convert missing factors to zero as the v0 aggregate defaults could permit. Recurrent/fused modules require validated adapters before claiming coverage.

“Once per model” means one frozen collection campaign, not one forward/backward total. The current collector repeats two scorer passes per objective/item. Full covariance storage scales quadratically in width; price collection and factorization before choosing ranks. Any diagonal/sketch replacement gets a separate schema and matched ablation, not silent reuse of existing q.

## Dynamics must beat exact arithmetic

For each rollout retain exact cumulative stored-weight displacement E and realised next increment D, computed without network inference. Scaling/removal actions must operate on the current weights; fixed additive actions have different semantics. Cached g,A,G stay anchored at the base. A latent prediction of their changes is a prediction, never a newly measured field.

Include the same-tensor quadratic cross-term tr(GEA Dᵀ): q(E+D)−q(E)−q(D). Thus a frozen quadratic field already predicts some interactions. It misses cross-tensor curvature and gradient drift; compare against both this baseline and a static residual evaluated on cumulative E.

Require endpoint consistency for histories ending in identical stored weights. Additive edits commute in exact arithmetic; stored rounding may change endpoints. Scaling and direction removal can be noncommutative. Test these separately, plus inverse-return and no-op trajectories. Learn dynamics only if it improves endpoint and incremental errors beyond the best cumulative baseline, without introducing artificial order dependence. Token summaries are lossy; they are not automatically a sufficient Markov state.

## Q4: information-matched head-to-head

The fetched [reader prompt](https://raw.githubusercontent.com/suayptalha/wsr/main/wsr/read.py) exposes config/tokenizer and replacement tensors to its weights arm, prohibits target inference and fitting, and permits tensor arithmetic. [Its helper](https://raw.githubusercontent.com/suayptalha/wsr/main/wsr/sandbox_lib.py) exposes CPU weights and basic statistics. A field-equipped predictor versus that unaugmented reader is a system comparison with unequal information.

Use identical opaque edit IDs, realised bytes, endpoints, source closures, train/test models and label rules. Freeze training outcome exposure, tuning opportunities and reader budget. Supply the reader the same allowed development examples or a frozen retrieval interface; report its pretrained exposure as uncontrolled. Exclude intended-effect prose. Freeze prospective panels before outcomes rather than selecting changed cases afterward.

| Track | Learned/scripted predictor | LLM reader |
|---|---|---|
| Forward-free matched table | Weight/action summary table only | Identical table, metadata and endpoint contract |
| Base-calibrated matched table | Same table plus cached field/probe columns | Identical augmented table, no new target passes |
| Raw-weight system audit | Frozen extraction plus fitted predictor | Same base/edit bytes and allowed caches, logged CPU tools |
| Hybrid | Frozen best predictor | Identical reader plus predictor distribution, support flags and uncertainty tool |

The table tracks isolate readout capability conditional on compression. The raw-weight audit tests practical systems; extra reader computations break strict summary matching. Replay useful computations into development features, then freeze a fresh comparison. The hybrid's gain establishes complementarity; it is not a standalone world-model victory.

Require numeric ΔNLL and behavior probabilities, with abstention distinct from “same.” Report MAE, three-way macro scores, change sensitivity/specificity, proper probability scores and risk versus coverage; no free removal of abstentions. Calibrate thresholds on training models only. A world-model advantage requires a prospectively meaningful improvement over the matched reader on held-out models with calibrated uncertainty; the reverse result favors the reader. A win only against forward-free access establishes value of base calibration, not superior dynamics.

## Smallest decision and separation

First propose a synthetic CPU representation contract, then within-family size transfer, then a held-out family. Three to five models support case studies, not a population-level family claim. Hold out architecture lineages together; close shared directions, parents and dose siblings across splits. Report per-model results and uncertainty over roots within models; root bootstrap cannot estimate uncertainty over unseen families.

Proposed contract budget: ≤2 CPU-h, ≤4 GB, zero GPU and reader tokens; synthetic tensors only. Test tensor-order invariance, LoRA gauge equivalence including repeated spectra, relative-depth boundaries, fused/split parity, missing-versus-zero coverage, signed-a reversal before rounding, q nonnegativity, and endpoint consistency. Reject forbidden paths before access. Any failed invariant stops extraction. No behavioral go decision is possible from synthetic tests alone.

A subsequent collection pilot needs separately measured calibration, TF-v1, generation and judge costs; the brief's TF-v1 timing cannot price WSR behaviors. Keep new schemas, manifests and outcomes outside Part A; authenticate reused base artifacts read-only. The [round-7 synthesis](SYNTHESIS-r7.md) leaves A5/F8/F9/F12-I/F5.5 unchanged and places confirmation after Part A. This report authorizes no collection or purchase.

| proposal | mechanism | test | cost | kill criterion |
|---|---|---|---|---|
| Shared token contract | Functional roles, invariants, coverage masks | Synthetic width/depth/layout checks | Proposed ≤2 CPU-h; 0 GPU/tokens | Any contract failure |
| Base calibration | Frozen gradients, Fisher and probes | Coverage, provenance, matched ablations | Unmeasured; pilot-price separately | Unsupported edited mass or cap breach |
| Size/family transfer | Variable operator sets, training-only transforms | Held-out model/lineage and source closures | Separate outcome purchase | No gain over pooled static baseline |
| Rollout candidate | Exact E plus learned functional drift | Cumulative static/quadratic baselines | Trajectory cost unpriced | Artificial path dependence or no predictive gain |
| Reader comparison | Matched table tracks and hybrid | Frozen paired endpoint predictions | User-set token cap; cache cost shared | No meaningful matched advantage |
