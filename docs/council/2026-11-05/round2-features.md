# Round 2 — Features CTO

Branch: `council-r2-features` (worktree `.worktrees/council-r2-features`).

## Evidence update and diagnosis

The new importance refit answers the open question from my round 1: top StatNN coordinates are role- and bucket-specific stable rank, spectral entropy and leading singular values, with Frobenius norm also prominent for HellaSwag. For example, the MMLU top five are down/bucket0 stable_rank, o/bucket0 spectral_entropy, v/bucket0 stable_rank, up/bucket3 sv1 and up/bucket1 stable_rank; HellaSwag's top five include o/bucket0 spectral_entropy, q/bucket0 sv2 and q/bucket0 fro. These are gain allocations among correlated coordinates, not causal importance. Source: `evidence-statnn-importances.md/.csv`.

The descriptor-only model is already strong (Spearman 0.524/0.614 for MMLU-Pro/HellaSwag); StatNN reaches 0.566/0.649 at n=345. The extra pooled weight features therefore add only about .04 on MMLU and .035 on HellaSwag; they add about .205 in KL (0.698 vs .493). This makes the headline weight signal less impressive than a pooled benchmark correlation suggests. Within-family ranges 0.13–0.71 in StatNN, and random_dir has exactly zero HellaSwag movement; aggregate prediction partly learns family/descriptor structure and null controls. Teacher-forced targets are highly repeatable, so measurement noise is not the main explanation for these deterministic endpoints. Sources: `evidence-statnn-importances.md`, `metrology_budget_output.txt`, `round1-metrology.md`.

I concede the weight-reading researcher's key correction: pooling alone cannot explain M1's failure. M1 already retains exact layer IDs and projected directions in tensor tokens; the missing information must be more specific (projection truncation/frame ambiguity, base-relative orientation, module composition), or the supervised sample/predictor is inadequate. The new importances reinforce that the tree gets useful low-order spectral shape, but do not show why any one coordinate is causal.

## Replies to council questions

### Weight-reading researcher

> “Which directional representation best preserves functionally meaningful coordinates while remaining invariant to permissible weight symmetries, and what falsifiable matched-norm test should we run first?”

Use base-anchored signed projections of ΔW, not singular vectors of ΔW alone: for each selected matrix, project ΔW onto fixed top left/right singular frames of the base tensor and fixed unembedding-row directions, retaining layer and role. Under exact parameter symmetries, transform both base and delta consistently; do not demand invariance to arbitrary rotations that change the function. The first test should create paired edits with identical support, Frobenius norm, singular values and family/operator, but rotate the edit relative to the base frame. Fit on ancestry clusters and test held-out clusters. A directional feature should predict TF residual differences where pooled norms/spectra tie; a permutation of base-frame coordinates is the negative control. The existing M1 result means this comparison must isolate base anchoring and matched orientation, not merely add tokens.

### JEPAWG researcher

> “Which edit-family metadata and predecessor-aligned features are available without leakage, and can we cache fold-safe views for the manifold audit?”

Available without target leakage: family/operator label, ancestry and predecessor IDs, exact changed tensor role/layer/shape, edit norm, and cumulative predecessor-relative ΔW. These metadata are in zoo provenance and feature tensor records. They are not automatically safe as predictors: family/operator and ancestry can make the task trivial, so report metadata-only, family-only and family+norm controls. Cache raw per-entry per-tensor rows keyed by entry hash and extractor version; construct scalers, PCA/manifold coordinates and neighbor structures inside each training fold only. Do not put descendants of held-out ancestry into unsupervised fitting. This answers availability; I did not run the manifold audit, so no geometry claim is made.

### Metrology CTO

> “Within family, ‖Δ‖F alone gives ρ = −0.51 on chain and −0.43 on lora for mmlu gold; state the within-family ρ you expect it to add over ‖Δ‖F on mmlu margin. … Will you train and score on family-residual targets … and make that the headline?”

Yes: use fold-local family (and device where applicable) residualization, then measure incremental performance over family+log‖Δ‖F with ancestry-cluster paired intervals. For margin I predict a small initial gain, +0.05 Spearman over that comparator, not a new high ceiling; the present evidence does not justify a more precise estimate. This is a prospective prediction, not a measured result. KL remains a calibration/validity endpoint: its strong norm relation makes it a poor primary behavioral target. TF gold log-prob and margin are preferable; HellaSwag accuracy is not a target for within-family claims absent within-family movement.

### Strategy CTO

> “Can we hold layer/support and singular directions fixed while changing their orientation relative to base weights, producing matched pooled norms/spectra but different qualified task effects? What single direction-preserving feature comparison would distinguish representation loss from insufficient zoo signal within the two-week cap?”

Yes, algebraically: conjugate/rotate a constructed edit within an equal-shaped matrix while preserving singular values and norm, keeping layer/support fixed, and use a paired synthetic edit control to verify invariants. For natural edits, generate matched-norm edits with the same base-frame projection energy but shuffled signed coefficients as the control. The decisive comparison is fixed-base-frame signed coefficients versus shuffled coefficients, against the identical family+norm StatNN baseline, on ancestry-held-out chain/LoRA edits and TF targets. This separates representation from a near-null family only if paired edits have measurable TF effects; random_dir's exact-zero HellaSwag change cannot establish that. Do not expand the zoo until this gated screen shows residual signal.

## Challenges to other proposals

1. **Weight-reading researcher — “M1 already keeps exact layers and projected directions so pooling alone cannot explain the failure.”** Agreed, and this is the right constraint on my prior diagnosis. The flaw in a generic “unpool all tokens” proposal is that it repeats information already delivered to M1 and does not isolate the residual cause. Evidence: round1-features’ extractor/data/model audit, plus the new importance report. Missing control: compare base-frame signed coordinates against same-dimensional shuffled-frame coordinates at matched support/norm and identical training budget.

2. **JEPAWG researcher — “Screen at 1,000–3,000 independent ancestry clusters” before expecting a sequence-model crossover.** This is a sensible scale warning, but insufficient as the next action: the current 345-row refit shows descriptor-only already at 0.52/0.61 and feature lift small on the two behavioral benchmarks, while some family signal is zero/near-zero. A much larger correlated chain zoo could strengthen provenance/size shortcuts without revealing a transferable mechanism. Missing control: first establish positive family+norm residual signal and a learning curve over independent ancestry clusters; only buy more labels if the directional feature beats that comparator. Evidence: `evidence-statnn-importances.md`, `metrology_budget_output.txt`.

3. **Metrology CTO — “KL is a validity probe, not a target.”** Correct: KL tests perturbation magnitude well (within-family norm ρ 0.58–0.91), so a KL gain cannot validate task understanding. The missing qualification is that near-perfect repeatability does not imply adequate *within-family effect range*: target repeat panels should report both reliability and non-null/range coverage. Evidence: `round1-metrology.md`, `metrology_budget_output.txt`.

## Revised feature experiment

**Choose base-anchored directional projections** as the single next feature family. Per-layer unpooled spectra are cheaper but overlap more with M1's exact layer/tensor tokens; cached-input direct effects are plausible for activation-mediated behavior but add probe-set overlap and inference costs before the weight-only residual is established.

For each eligible matrix ΔW∈R^(m×n), construct frozen base-only orthonormal frames U_k and V_k from the top k singular vectors of W₀, with k=16 each, plus the 32 highest-frequency unembedding rows projected into the output space where dimensions match. Emit C=U₁₆ᵀ ΔW V₁₆ (256 signed scalars), row/column energy in those frames (32), and signed projection of ΔW's output action onto unembedding directions (32); include layer, role, and normalized frame-energy coverage (3). Total 323 values per matrix, variable-length across changed matrices; predictor uses shared projection encoder plus exact role/layer embeddings. Use only base-derived frames, never target/evaluation-fitted directions. Extraction is one-time base SVD/calibration plus per-entry small projected products; estimate 5–30 s/entry after the existing 66 s feature pass, to be timed on 20 entries before scaling. Cost estimate is not measured. The top-16 frames capture orientation but not all null/low-rank subspaces; compare a larger-k sensitivity only after the first gate.

Leakage risks: benchmark vocabulary or prompts used to select unembedding rows; base/model identity and layer fingerprint; ancestry leakage across descendants; and unstable top singular frames under near-degeneracy. Freeze generic tokenizer-frequency selection before outcomes, publish frame hashes, align signs deterministically, and cluster-split by ancestry. Report a frame-shuffled control and a role/layer-only control.

**Falsifiable prediction:** on TF-v1 gold log-prob and margin, after family+log‖Δ‖F residualization, the feature adds at least +0.05 ancestry-cluster held-out Spearman versus that comparator for chain/LoRA combined, with a paired 95% cluster-bootstrap lower bound above zero. Report KL as a secondary diagnostic; do not count it as success. **Kill criterion:** stop this feature family if the upper 95% cluster interval is below +0.05, or if base-frame coefficients fail to beat same-dimensional shuffled-frame controls; also stop any benchmark-specific claim if the selected family has fewer than 30 independent non-null ancestry clusters.

## Ranked actions for the chair

1. **Run the matched orientation screen** — Owner: Features + Weight-reading; cost: 20-entry extraction timing plus 1–2 predictor hours; prediction: ≥.05 residual Spearman on TF-v1 gold log-prob/margin over family+norm; kill: upper interval <.05 or shuffled frame ties.
2. **Make residualized TF endpoints the primary score** — Owner: Metrology; cost: CPU analysis, no new evaluations; prediction: conclusions differ from pooled ρ, with clear family-specific non-null subsets; kill: no endpoint has enough independent non-null clusters for interval estimation.
3. **Add fold-safe family/norm and descriptor controls to every comparison** — Owner: Predictor; cost: 1–10 min/run; prediction: expose how much apparent lift is descriptor/family structure; kill: no meaningful incremental comparison if metadata-only matches all candidates.
4. **Expand only the informative zoo families** — Owner: Zoo + Strategy; cost: defer budget until action 1 passes; prediction: independent chain/LoRA/operator variants produce diverse TF residuals; kill: projected learning curve remains flat at 2× independent clusters.
5. **Defer activation-cache direct effects to a second gate** — Owner: Weight-reading + Metrology; cost: estimate 30–120 s/entry plus prompt curation; prediction: if directional weights fail but activation-mediated residual persists, cached-input effects may recover it; kill: probe-disjoint held-out TF residuals do not improve over direction features.

## Five-line handoff

Branch: `council-r2-features`.
Two strongest challenges: unpooling repeats M1's layer/direction inputs; scaling the zoo before residual-signal gating risks amplifying shortcuts.
Revised top action: matched-norm base-anchored signed projections on TF-v1 residuals.
Changed my mind: pooling alone is not the cause; importance evidence and M1 already falsify that simple story.
Open disagreement for chair: whether to fund more independent chain/LoRA clusters before the directional screen clears its gate.
