# Council brief — 7 Nov 2026 (read first; every council member starts here)

## Why this council sits

The user has asked the agentic council to **challenge every negative and neutral result of the project** and to
**find a world-model structure** (JEPAWG-like, or deliberately different) that can predict the effect of a weight edit
on a model, including the setting where one model edits another (H5, the blog's Experiment 8). The question is not
"is the current result right" (Boards A and B have reviewed that; see below) but "what structure would make the
prediction work, what is the cheapest decisive test of it, and what in our negative results was a test of the
structure rather than of the measurement".

GPU time is approved by the user for the next round (local RTX PRO 4500 32 GB; Lambda H100/A100 beyond the earlier
$150 cap is now allowed). Lost chain deltas (358 of 1,259 zoo rows) are being regenerated in parallel
(`scripts/d8_chain_replay.py`, branch `feat/d8-chain-replay`); plan for the full 1,259-row zoo being back.

## Goal of the project (unchanged)

Test whether a dedicated weight-space "world model" can predict, from the weights and the edit alone, how an LLM's
benchmark scores change after each weight edit (single edits and edit trajectories), on small models
(Qwen3.5-0.8B now; 4B later), including normal vs abliterated vs Heretic edits, and later act as a critic for an
LLM editor (H5). Hypotheses: `docs/HYPOTHESIS.md` (H1–H6, with the 5–6 Nov amendments). Trigger: MIT CSAIL blog
"Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, 2 Nov 2026,
https://rikiyat.github.io/blog/weight-space-programming/; our extraction `docs/H5_RESEARCH.md`,
`docs/RESEARCH_NOTES.md`). Method inspiration: JEPAWG — *Weight-Space Physics* (arXiv 2607.07127), *Bifurcations from
Weight Space* (arXiv 2610.30847); W2T canonical LoRA tokens (arXiv 2603.15990). Previous council (5 Nov):
`docs/council/2026-11-05/SYNTHESIS.md` and the ten round reports next to it. Do not repeat its findings; build on them.

## The results to challenge (all numbers are in the files named; quote the file, never the memory of it)

1. **Single-edit prediction, protocol v2, recovered set** (`docs/reviews/P4/d3v2r-summary-2026-11-06.md`, runs
   `data/runs/p4/d3v2r-*`, 297 rows / 270 ancestry clusters, Spearman on mmlu_pro / hellaswag):
   StatNN (LightGBM on pooled weight statistics) 0.54 / 0.65 seed split, 0.51 / 0.66 trajectory split;
   descriptor-only 0.48 / 0.58; delta-norm 0.16 / 0.21; best neural predictor M1-matched 0.47 / 0.54 (seed),
   0.46 / 0.55 (trajectory); M2 Mamba 0.39 / 0.47; M3 JEPA 0.25 / 0.34; M5 0.23 / 0.15; M6 0.25 / 0.25.
   **Neutral/negative**: no neural model beats the tree; the descriptor-only control is within 0.07 of the tree.
2. **Held-out edit family** (same file, `*-family` rows and `m2-holdout-*`): everything in [-0.21, 0.18]; StatNN
   -0.18 / 0.06; neural 0.08–0.18 / -0.21–0.16. **Negative**: no model transfers across families. Caveat already on
   record: pooled family-holdout ρ is confounded by between-family calibration offsets (`docs/RESULTS.md` P4 section).
3. **History (H2)**: M2 vs M2-matched contrast confounded; only 18 matched rows / 10 clusters at length ≥ 4
   (`docs/council/2026-11-05/SYNTHESIS.md` finding 8). **Not testable** on the recovered set; the lost chain deltas
   are being regenerated (this matters: it is the only place where "trajectory" means anything).
4. **Teacher-forced tier (TF-v1)**: H1a untested (MMLU gold log-prob did not qualify: θ̂ 0.129, CI crosses 0;
   `docs/reviews/P5/tf-v1-requalification-2026-11-06.md`, `docs/reviews/P5/h1a-companions-2026-11-06.md`);
   H1w FAIL (θ̂ 0.007, `docs/reviews/P5/h1w-gate-2026-11-06.md`). **Negative**: beyond family + device + log-norm,
   the weight statistics do not read the teacher-forced effect.
5. **Measurement facts** (`docs/RESULTS.md`): same-seed repeat bit-identical; different sampling seed moves
   gsm8k/math500/ifeval by 0.03–0.06 (ρ 0.10–0.46 against themselves); mmlu_pro, hellaswag, ppl exactly
   repeatable; KL differs ≤ 3.3e-4 nats between seeds. Only 94/300 D3 edits have KL-to-base > 0.01
   (LoRA 56/60, chain 30/50, task 6/30, abliteration 2/100, random 0/60). Most zoo edits are behaviourally null.
6. **What the tree exploits** (`docs/council/2026-11-05/evidence-statnn-importances.md`): down/bucket0 stable rank,
   o/bucket0 entropy, v/bucket0 stable rank; descriptor-only is almost as good, so most of the rank is family + norm.
7. **Blog facts verified by the 5 Nov council** (`docs/council/2026-11-05/round1-weight-reading.md`): weights-only
   recognition of game/shape/circuit; Experiment 7 needed attention/activation assistance after four weights-only
   failures; Experiment 8 used activation/behavioural feedback for localised repair. **The blog never demonstrates a
   held-out Δ-only benchmark predictor**; our H1 was stronger than the blog's claim.

Paper state: `paper/main.pdf` (21 pages, TMLR format), Board A close review `docs/reviews/P5/board-a-d8-close-2026-11-06.md`,
Board B `docs/reviews/docs/board-b-whole-manuscript-2026-11-06.md`. Infrastructure: `docs/ARCHITECTURES.md`
(Qwen3.5-0.8B, Qwen3.5-4B, LFM3-1.7B, Gemma-3-270M tensor maps), `docs/MODEL_CANDIDATES.md` (Board C's earlier
representation assessment, incl. GMN/LoL/W2T/T-JEPA references), `src/wsp/features/` (features v4: 1,242 stats +
136 spectrum, pooled by role and depth bucket), `src/wsp/predictor/models/` (M1, M2 Mamba-2, M3 JEPA, M5, M6),
`src/wsp/editor/` (H5 sandbox harness, merged, production disabled; `docs/H5_DESIGN.md`, `docs/H5_PROTOCOL.md`,
`docs/reviews/H5/*`). Measured costs: fast-tier evaluation 7.5 min/entry on this GPU (3.1 min effective on an H100
with three workers); TF-v1 scoring ≈ 1.0 min/entry warm; feature extraction 54 s/entry; predictor run 1–10 min CPU;
LoRA SFT step 50–800 steps ≈ 1–5 min (to be confirmed by the chain pilot).

## The question for this council

**Find the structure.** Given everything above, propose the world-model structure most likely to predict the effect
of weight edits — in particular edits produced by another model (H5 `llm_edit`) — and the smallest experiment that
would show it working or kill it. Each member answers from their own angle (below) and must:

1. Say which of results 1–7 is a **test of structure** and which is a **test of measurement or data** (and why);
   where a negative result cannot be attributed, say what single measurement would separate the two.
2. Propose **one primary structure** (what the model consumes: raw Δ? canonicalised low-rank factors? activations
   on a probe set? the edit *program* (code/descriptor)? the editor's own transcript?; what it predicts: Δ-benchmark,
   Δ-logits on probes, latent state; what inductive bias: JEPAWG parameter-indexed populations, GMN permutation
   symmetry, set/sequence over layers, physics-style conservation/sensitivity structure) and **two alternatives**,
   each with the mechanism by which it would beat StatNN + descriptors on *our* data.
3. Give the **decisive experiment**: data needed (which zoo rows, which new rows and how many; whether the 358
   regenerated chain rows suffice for H2), labels (which TF-v1 or fast-tier metrics; item counts), split (ancestry/
   family/base), baselines (StatNN + descriptor-only + delta-norm are mandatory; add a probe-activation baseline if
   you use activations), pass/fail numbers with a power argument, GPU-hours from the measured costs above.
4. Address **models editing models** explicitly: what changes when the edit is an LLM's proposal (sparse,
   localised, semantically motivated, with a transcript) rather than a sampled kernel; whether the predictor should
   read the proposal's *rationale*; how to avoid the Meta^n order-statistic trap (`docs/H5_DESIGN.md`).
5. Keep the **blog's own evidence** honest: cite the exact experiment and URL you verified for anything about it.

Output rules (unchanged from 5 Nov): no invented numbers; every claim about our data points to a file path; every
external claim to a URL or arXiv id you actually fetched; proposals must be concrete and priced; write ≤ 2,500 words;
put a 10-line BLUF at the top; end with a table "structure | mechanism | decisive test | cost | kill criterion".
Write your report to `docs/council/2026-11-07/round1-<role>.md` in this worktree only; do not edit any other file,
do not commit, do not touch `data/`.

## Members and angles (round 1, parallel)

| role | file | angle |
|---|---|---|
| `world-model` | `round1-world-model.md` | JEPAWG / weight-space-physics structure vs alternatives (GMN, LoL/W2T, set transformers over layers, hypernetwork decoders); what a *dynamical* chart of edits would look like for a 0.8B LM |
| `representation` | `round1-representation.md` | what the raw delta carries that pooled statistics discard; low-rank directions × layer, alignment with refusal/task directions, activation probes; a concrete feature or encoder spec |
| `metrology` | `round1-metrology.md` | which negatives are measurement artefacts; label reliability, null edits, power; the exact TF-v1 metric and item count a structure test must use; how to make H2 testable with the regenerated chains |
| `editor` | `round1-editor.md` | models editing models: what an `llm_edit` delta looks like, what a critic must predict for the editor loop to save evaluations, how to read rationale/transcripts, controls against order-statistic gains |

Round 2 (after round 1 is in): each member reads the other three reports and writes `round2-<role>.md` (≤ 1,200 words):
concessions, rebuttals, and a revised single proposal. The chair then writes `SYNTHESIS.md`; Boards A, B and C review
the synthesis before anything is scheduled.
