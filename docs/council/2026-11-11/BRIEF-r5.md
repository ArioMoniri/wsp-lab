# Council brief — 11 Nov 2026, round 5 (read first)

## Why this round sits

After round 4 the user asked for a wider sweep on five threads and for the council to say what they mean for a weight-space world
model: (1) edits that make a model better at mathematics; (2) models that edit their own weights surgically **while working**;
(3) models that can tell **which parts of themselves they are using**; (4) **shrinking parameter count at equal performance**; and
(5) **predicting all four with a world model**. The registration is frozen (a638efa; F6, F7 accepted; F8 accepted with conditions,
`docs/reviews/council/board-a-f8-2026-11-11.md`). Part A staging is at 25 of 40 edits and nothing from G3 has been scored, so any
change still enters only as a dated amendment (F9) before any G3 outcome is read. Round-4 state and results: `BRIEF.md` and
`SYNTHESIS.md` in this directory; the round-4 items are not repeated here.

## What the sweep found (verified 11 Nov; cite the URL you read, not this digest)

### Thread 1 — edits that improve mathematics

- Round 4 covered the Astra loop edit, arXiv 2611.09382 (column edit, +29.45 MATH-500 held-out on our base), SWEET (2611.10974)
  and the replications. New: **ThinkEdit** (arXiv 2503.22048) edits weights to mitigate overly short thinking in reasoning models;
  **CoT Vectors** (arXiv 2510.00579) transfer reasoning mechanisms as vectors; **Weight-of-Thought** (arXiv 2504.10646). A further
  replication (Krishanu Das, dev.to, 7 Nov) finds the published edit makes thinking finish about five times as often at an 8,192-token
  budget (10/30 vs 2/30) but thinking-off on the unedited model (16/30) still beats the edited model with thinking on (9/30); paired
  McNemar p = 0.021 raw, 0.107 after his pre-committed Holm correction. https://dev.to/krishanu_das/before-trusting-a-weight-edit-i-measured-the-bug-it-fixes-401f
- Reading: every math gain on small models so far is a **termination or robustness repair** (loops, overthinking, distractor
  reliance), not new capability; the gains are generation-side and largely invisible to teacher-forced endpoints.

### Thread 2 — models that edit their own weights while working

- **Agentic-TTT** (arXiv 2610.12002, 8 Oct): a learned test-time policy decides when to invoke test-time training, which method, or
  whether to reuse a stored LoRA skill; nearly doubles Qwen2.5-7B utility on a curated heterogeneous stream; each invocation
  produces an isolated LoRA over the frozen backbone. https://arxiv.org/abs/2610.12002
- **QueST** (arXiv 2605.13369): query-conditioned test-time self-training; builds problem–solution pairs from the query and
  fine-tunes a PEFT adapter per query; beats test-time-optimisation baselines on seven math benchmarks and GPQA Diamond.
  https://arxiv.org/abs/2605.13369
- **Search over Self-Edit Strategies** (arXiv 2601.14532): SEAL with model-generated self-edit templates; archive variant approaches
  but does not beat the best human template; exploration collapses without novelty pressure. https://arxiv.org/abs/2601.14532
- **RSI-Env** (arXiv 2611.01736, 1 Nov): persistent environment for coordinated model–harness self-improvement starting from
  Qwen3.8-27B; without benchmark feedback GPQA Diamond 65.2 → 73.2, AIME 2026 53.3 → 56.7; gains across capabilities are hard to
  preserve. https://arxiv.org/abs/2611.01736
- Harness-side or taxonomy: **Recursa** (2611.04035), **Evo2** weight–harness co-evolution (2610.13025), **Exo** runtime
  self-editing harness (2611.00463), **MetaRSI** (2609.06396), **SpecRSI** (2610.16491), **RSI-Opt** (2610.32026), **Harness RSI**
  66-page survey (2610.18813), RSI taxonomy (2610.14625), test-time self-improvement survey (2609.01679). Round 4 already has
  WeightLoop (2611.07319).
- Reading: runtime self-editing in 2026 is **gradient-based and adapter-isolated** (LoRA skills, per-query PEFT), governed by a
  policy that decides *whether* to edit. None of these systems predicts the edit's effect before applying it; they evaluate after.

### Thread 3 — models reporting which parts they use

- **OWMI** (arXiv 2608.20569, Aug): eight open-weight models asked whether their own computation was altered (residual sites, heads,
  SAE features) score at chance (AUROC 0.5007 over 78,000 measurements, equivalence bound 0.15 pp), while a linear probe on the same
  activations recovers the intervention at 75–96 % and a fine-tuned reporter reaches AUROC ≈ 1.0: the information is present, the
  verbal path is missing. https://arxiv.org/abs/2608.20569
- **A mechanistic study of LM introspection** (arXiv 2609.35108): gate heads (middle layers) decide whether a change is reported,
  router heads (one later layer) pick the position; three families up to 12B. https://arxiv.org/abs/2609.35108
- **Mechanisms of Introspective Awareness** (arXiv 2603.21396): detection is distributed MLP computation (evidence-carrier and gate
  features), emerges from post-training, improves ~53 pp with refusal-direction ablation. **Identifying Introspection From the
  Inside** (arXiv 2610.07186, COLM 2026): faithful self-report shows higher attribution similarity between task and testimony.
  **Training LMs to Explain Their Own Computations** (arXiv 2511.08579): self-explanation beats cross-model explanation.
  **Introspect-Bench** (arXiv 2603.20276); **Multi-component causal tracing** (arXiv 2606.03085); self-interpretation adapters
  (arXiv 2602.10352). Round 4's reading audit (2611.08070) says zero-shot weight reading is local arithmetic plus priors.
- Reading: a model cannot verbally report which weights it uses; probes and attribution patching can. "Which part am I using" is an
  interpretability measurement, not a self-report, and it is exactly what our base-anchored gradients and Fisher factors encode for a
  given objective.

### Thread 4 — fewer parameters, same performance

- Round 4 has **Deletion Tuning** (2611.10866: < 0.05 % zeroed matches full fine-tuning). New: **Revisiting LLM pruning for
  test-time scaling** (arXiv 2604.25098): unstructured pruning of s1.1-7B and Qwen3-8B matches or beats the unpruned model on
  reasoning benchmarks, structured pruning degrades; **OBC-Prune** (2609.17890): outcome-based calibration preserving causally
  important reasoning circuits beats SparseGPT/Wanda/ALPS calibration on R1-Distill-Qwen 1.5B–14B at 40–50 % sparsity; **Router
  sensitivity** (2608.07890): experts whose router weights move least under a 0.002 %-parameter LoRA can be pruned with
  quasi-linear accuracy decay; **Pruning laws** (2504.04342): collapse past ~70–80 %; **LLMs can compress LLMs** (2601.09694);
  multi-objective structured pruning (2607.22583); pruning vs training from scratch (2606.14150); GradPruner (2601.19503);
  **ModHiFi** (NeurIPS 2025): subset fidelity without gradients or loss.
- Reading: the active frontier is *which* weights to delete, decided from gradients, router sensitivity, or causal importance on the
  model's own rollouts, i.e. receiver-anchored saliency, the same object as our first-order field term a = ⟨g, D⟩ for D = −W⊙M.

### Thread 5 — predicting the effect of a modification before making it

- **Predict Before You Modify / IDM** (arXiv 2610.13991, 15 Oct): an editable Improvement Dynamics Model predicts task-level
  outcomes of a proposed agent modification before evaluation and guides revision; up to 46 % higher performance over generations;
  forecast accuracy improves as the IDM evolves; transfers to a new domain. Targets agent programs, not weights.
  https://arxiv.org/abs/2610.13991
- **PPF, predictive pruning** (arXiv 2508.02381): a lightweight predictor maps a pruning policy (mask) to perplexity in 1.5 s with
  prediction error < 0.0011, 64× faster than evaluation; an agent uses it for real-time pruning decisions on Llama2-7B/Llama3-8B.
  https://arxiv.org/abs/2508.02381
- **Forecasting how RLHF responds to reward-model edits** (arXiv 2611.03244, 2 Nov): first-order theory of KL-regularised RL; each
  feature moves in proportion to its within-prompt covariance with the reward change; predicts the direction of all 20 consistent
  changes and 82 % of standardised change before training; 54 RL runs, forecasts fixed before runs. https://arxiv.org/abs/2611.03244
- **RankAdaptor** (NAACL 2025 findings): an MLP performance model over per-layer LoRA ranks for pruned LLMs. **TuneAhead**
  (2606.17660, round 3) and the risk-decomposition theory of pre-hoc prediction (PMLR v306, Luo et al.) for datasets/hyperparameters.
- Reading: published "world models" for modifications are (i) first-order covariance/gradient forecasts fixed before the run, which
  work, and (ii) learned surrogates over structured modification spaces (masks, ranks, agent programs) trained on past
  evaluations, which work when the modification space is low-dimensional and the surrogate keeps learning. No published work
  predicts benchmark change from an arbitrary dense weight delta on a held-out split; that remains our H1.

## Not found

No paper in which a model (a) decides which of its own weights to change from a self-report of which parts it uses, (b) applies the
change during inference without gradients, and (c) predicts the effect first. The pieces exist separately: policy-governed test-time
training (thread 2), probe-based localisation (thread 3), saliency-chosen deletion (thread 4), first-order forecasts (thread 5).

## The questions for round 5

1. **One system or four?** Is there a single world-model structure that covers threads 1–4 (a predictor over a receiver-anchored
   edit representation, used by a policy that decides whether and where to edit), or are these four problems with four predictors?
   Be concrete: inputs, what is frozen, what learns online, and which of our existing artefacts (G2 fields, chart, StatNN, TF-v1
   endpoints, H5 critic contract) it reuses. Say what JEPAWG-style latent dynamics would add over the first-order field, if anything.
2. **First-order forecasts as the baseline world model.** The RLHF-edit paper and our field both forecast with a frozen first-order
   term computed before the run. Register (as F9, descriptive) a **forecast-before-modify log** for every future edit the programme
   makes (Part A variants excluded; F8 panels, H5 proposals, any pruning entries): the field's a, q, κ and sign recorded with a
   hash before scoring, with calibration reported per family. Does anything beyond F8.3 need registering, and what is the
   calibration statistic?
3. **Pruning as an edit family (thread 4).** F8-F already has Wanda and random gate deletions at < 0.05 %. Should the family panel
   extend to a **sparsity ladder** (e.g. 10 / 30 / 50 % unstructured, Wanda vs SparseGPT vs random, outcome-blind) so that the field
   is tested where the pruning literature says performance holds, and is the first-order term a = −Σ_M g_ij W_ij plus q enough to
   rank masks (the PPF question) on our endpoints? Price it from measured rates; say what the chart adds.
4. **Self-localisation (thread 3).** OWMI shows the information is in the activations but not in the verbal report. For the H5
   editor: should the critic contract require **probe-based localisation** (attribution patching or gradient saliency from the
   frozen G2 artefacts) as the editor's "which part am I using" input, with the LLM's own verbal localisation logged as a treatment
   flag and compared against it? What is the registered test that the LLM adds anything over the probe?
5. **Runtime self-editing (thread 2).** Agentic-TTT and QueST edit at deployment with gradients through isolated adapters, gated by a
   policy. Does the programme need a registered arm in which the H5 critic gates a **test-time LoRA** rather than a gradient-free
   edit (the critic as the "whether to edit" policy, with a forecast-before-modify log), and what is the cheapest decisive test of
   "the critic's forecast beats the policy's own after-the-fact evaluation on cost" on our hardware?
6. **Math (thread 1).** Given that every small-model math gain so far is a termination repair that teacher-forced endpoints cannot see,
   should the programme register a **generation-side endpoint** (e.g. natural-completion rate or MATH-500 with thinking on at a fixed
   token budget, paired, with the thinking-off baseline) as a descriptive endpoint for F8-E only, or stay with TF-v1? Price it and
   state the qualification it would need (A3-style repeatability under sampling).
7. **Line-stop and the paper.** Does anything in threads 1–5 change F5.5, the H5 success criterion, or the measurement paper's
   related work? Name exact sentences.

Rules as before: no invented numbers; every claim about our data cites a file; every external claim cites a URL you fetched;
≤ 2,000 words; 10-line BLUF; final table "proposal | mechanism | registered test | GPU-h | kill criterion". Write
`round5-<role>.md` in this directory only; no commits, no GPU, no data/ writes, no edits to A5. Roles: `world-model`,
`representation`, `metrology`, `editor`. The chair then writes `SYNTHESIS-r5.md`; any amendment goes to A5 as F9 before any G3
outcome is read.
