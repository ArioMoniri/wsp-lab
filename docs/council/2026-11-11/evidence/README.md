# Retained primary-source excerpts (Board A F8 condition 7)

Two sources cited in round 4 returned HTTP 403 to some council members. The chair fetched both on 11 Nov 2026 through a
content-extraction service and retains the excerpts below verbatim so that the claims used in `SYNTHESIS.md` can be audited without
a live fetch. Only the control design and the construction details are used as premises; no number from either page is a registered
threshold or funding predicate. Numbers quoted elsewhere in the brief come from these excerpts.

## Concept Country, "Gradient-Free Weight Editing Reduces Overthinking in Qwen3.8-27B" (working paper v1, 8 Nov 2026)

URL: https://concept.country/research/weight-space-programming — fetched 11 Nov 2026. Authors listed on the page: Justin Bennington,
Nikki Kaelar, Spencer Lee, Mitchell Edwards, Kojin Glick, Cilvia Danks (Concept Country, New York). PDF SHA-256 stated on the page:
`8003956b1f9a4a461dbed067af7066a83389f44200ff9a6757a710dee6fa0f80`. Weights: `conceptcountry/qwen-concept-27b` (bf16 and quantised
variants; "every number in this paper comes from the bf16 weights").

Abstract (verbatim): "A reasoning model can be made to think less and score more by editing nine of its weight matrices, with no
gradient step, no training data and no fine-tuning. We replicate weight-space programming on Qwen3.8-27B: directions are read off
residual activations captured in forward passes on a train split, then compiled into the down-projections of nine alternating layers
as a rank-one additive term. On a held-out confirmation suite the edit improves GPQA Diamond by 4.3 points (95% CI +1.8 to +7.1) while
cutting thinking tokens 30.0%, improves LiveCodeBench v6 by 13.0 points (+6.1 to +20.6) at 29.2% fewer tokens, and improves IFBench by
3.3 points at 54.1% fewer tokens. The result that makes this a finding rather than folklore is the control: a norm-matched random edit,
identical in perturbation magnitude and applied to the same nine matrices with a per-layer random direction orthogonal to the fitted
one, does nothing. Every base-to-control accuracy interval includes zero and it saves no tokens on any benchmark."

Status (verbatim): "A pre-registered confirmation study with a norm-matched random control. The criterion passes on GPQA Diamond and
LiveCodeBench v6 against both base and control, and on IFBench against base on the token clause. The IFBench control clause is
reported as borderline and is not claimed."

Data and code (verbatim): "The edit and control specifications, every graded result file with its digest, the generating script with
its per-arm sample census, and the seed-stability sweep are published with this paper. emit_results.py recomputes every statistic from
the graded artifacts and emit_numbers.py regenerates the figures quoted in the text, so no number in the paper is transcribed by hand.
Discovery used the MATH train split only; the confirmation suite was never used for selection."

## Nathan Barry, "Can a Linear Program Do Weight-Space Programming?" (blog, 9 Nov 2026)

URL: https://nathan.rs/posts/weight-space-lp/ — fetched 11 Nov 2026.

Excerpts (verbatim): "Looking at Astra's edit, I realized most of its search was solving a linear program by trial and error. It spent
about 8 hours and 28 forward passes choosing which weights to change and by how much. A couple of LP solves, about a minute each, do
the same job with 8× fewer changed weights."

"Astra wrote four pairs of short texts, each a piece of productive reasoning and a looping version of the same reasoning. It ran all
eight through the model and recorded each token's residual stream vector at one layer. Averaging over the looping texts gives one
vector, and averaging over the productive texts gives another. Their difference, normalized, is the loop direction u: the way the
residual stream moves when the model loops."

"Each column points partly along u. That amount, uᵀw_j, is how much column j pushes the model toward looping whenever h_j is active.
Astra's edit subtracts a multiple α of it: w_j → w_j − α (uᵀw_j) u. At α = 1 the component along u is gone, so the column no longer
pushes toward looping. Doing this to every column in layers 15 and 19 took accuracy from 9% to 33% on GSM8K and MATH training
problems."

"This is one model. The LP still relies on Astra for the layers, and the best results use Astra's direction too. The heuristic
direction gets most of the way without an LLM, but not all of it."

Reading used in the synthesis: the LP edits reuse Astra's layers and, in the best case, Astra's direction, which is why the loop family
is declared one source cluster (F8.1). The "8×" figure is reported, not used.
