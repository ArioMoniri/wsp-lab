# Council brief, 16 Nov 2026, round 8 (revised scope): a single-model edit world model for trajectory prediction and enhancement

## The user's request (latest, lightly paraphrased)

The user has asked for three things:

- Test on popular, available, newest open-weight models that are likely to give the best results or have the most room for enhancement.
- The paper must show two things:
  - (1) that we predicted the **trajectory** of edits;
  - (2) that we **enhanced an LLM** using predictions from the world model.
- For now, use **one LLM, not a generalised model**, and use whatever resources are needed.

The context is WSR (https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models). In that work, LLM readers predict the effects of edits on other LLMs without running the edit. Our answer is a world model of one target LLM's response to edits.

That world model must:

- predict single edits;
- roll out multi-step edit trajectories;
- be used to **plan** edits that improve the model, with the improvement verified by real evaluation.

This supersedes the cross-model draft. The earlier cross-model session was cancelled before writing anything.

## Prior facts (cite files)

- **Round 7** (`docs/council/2026-11-16/SYNTHESIS-r7.md`, `docs/reviews/msrm/msrm1-dev-v0-2026-11-16.md`). These results are on our 0.8B zoo:
  - Learned model-specific residuals beat the field readout by C ≈ 0.31–0.43.
  - The description-only baseline reached ≈ 0.04.
  - Kill criterion k2b fired, because the field did not separate from scripted statistics.
  - The 123 rows formed only 29 independent groups.
- **M3**, the JEPAWG-style latent dynamics model, reached ρ 0.15/0.13 on 25 pairs (`docs/council/2026-11-11/SYNTHESIS-r6.md` Q2). Chain steps fall below the noise floor.
- **Code we have:**
  - field (a, q) and K-FAC: `src/wsp/features/response_field.py`;
  - statistics: `src/wsp/features/stats.py`, `pool.py`;
  - MSRM-1: `src/wsp/msrm/`;
  - TF-v1 evaluation;
  - the A5.7/H5 critic design, which covers prediction-guided search and verification savings.
- **WSR code:** https://github.com/suayptalha/wsr. It includes edit operations (block/row scaling, direction removal, noise, no-op), a label harness and a reader.
- **Machine:**
  - GPU: one RTX PRO 4500 Blackwell, 32 GB.
  - RAM: 30 GB.
  - CPU: 8 cores.
  - Disk: about 295 GB free.
  - Software: torch 2.15 and transformers 5.22 in `.venv`. vLLM is not installed; a separate venv is allowed.

## Chair's draft choice of target model (attack it)

The draft target is **Qwen/Qwen3.5-9B**.

**Reasons for this choice:**

- It is one of WSR's own targets, which makes a direct head-to-head possible: WSR reports 9B→9B weights 0.554 vs text 0.492.
- It is a popular, current dense model.
- Its bf16 weights (≈ 18 GB) fit the GPU for inference.
- It is the same family as our 0.8B zoo, so our tooling carries over.
- The official Qwen-Scope SAEs exist for Qwen3.5-9B-Base (`Qwen/SAE-Res-Qwen3.5-9B-Base-*`). These could serve as interpretable state features.
- The newest dense flagship, Qwen3.8-27B, does not fit for gradient or Fisher work on 32 GB.

**Fallbacks and alternatives:**

- Qwen3.5-4B, if throughput or 30 GB of host RAM makes 9B impractical.
- Other new options seen on the Hub: LiquidAI/LFM3-1.7B (hybrid, custom code), granite-4.2-8b derivatives, and Nemotron-3.5 E4B.

**Required judgement:** decide which single model maximises the likelihood of a demonstrable result. Weigh:

- headroom for edits to improve it;
- evaluation throughput;
- gradient feasibility;
- WSR comparability.

## Chair's draft study design (WM-E; attack, fix or replace)

1. **World model.**
   - **State** = per-tensor tokens of the current weights, computed from base-model passes only:
     - field a, q;
     - spectra;
     - probe activations;
     - optionally SAE feature alignments.
   - **Action** = edit tokens relative to the current state.
   - **Dynamics:** z_{t+1} = z_t + f_θ(z_t, e_t), nested on the field prior.
   - **Decoders** predict:
     - ΔNLL;
     - benchmark rates on math, multiple choice, IFEval and others;
     - WSR behaviours.
   - It runs on **sequences** of edits.
2. **Trajectory bank (training data).**
   - Edit families: dose ladders, 2–4-step compositions, and steering directions written into weights (difference of means, abliteration-type, task vectors from small LoRA SFTs). WSR operations are included.
   - Each state along a trajectory is evaluated once.
   - Independent units are trajectories with distinct directions and seeds. Count closure groups honestly.
3. **Trajectory prediction claim.**
   - Hold out whole trajectories and families.
   - Roll out from the base.
   - Report error as a function of horizon against these baselines:
     - field-only, both re-linearised and not;
     - per-step MSRM-style static residual;
     - StatNN;
     - description-only;
     - WSR-style LLM reader.
4. **Enhancement claim (the key new element).**
   - The world model is the critic or planner in a search over candidate edits, e.g. scale, direction and layer for steering-into-weights, and compositions.
   - Its objective is to improve target capabilities (e.g. math or MMLU-Pro) under guardrails: perplexity, refusal and other benchmarks must not degrade.
   - **Equal real-evaluation budget** for each arm: world-model-ranked candidates vs random, field-ranked, StatNN-ranked and LLM-reader-ranked.
   - Success means the best edit found per evaluation is better and the result holds up.
   - Selection uses development items only. The final improvement is reported on **held-out test items never used in any selection**, with paired confidence intervals.
5. **Gates and governance.**
   - This is a new study, separate from A5 (Part A is held and untouched).
   - It gets its own preregistration file, written before any confirmatory outcome.
   - Exploratory development first.
   - Confirmatory trajectory and enhancement tests run on fresh frozen panels.
   - Outcome-blind power simulation before the confirmatory spend.

## Questions

1. **Target model.** Make the choice (see above). Give the evidence: fetch model cards. Cover the memory plan for gradients and K-FAC (layerwise, offload) on 32 GB GPU and 30 GB RAM.
2. **World-model structure.**
   - What makes rollouts actually better than per-step static prediction?
   - Which trajectory targets carry signal above the noise?
   - How do we avoid M3's failure?
   - Is re-linearising the field at intermediate states (a new gradient per state, which needs a base-model pass of the edited weights; that is allowed only if not "running the edit") permitted, or must the world model replace it?
   - Define precisely what "without running the edit" allows.
3. **Data plan and cost.**
   - Which edit families, how many trajectories and steps, which evaluations?
   - Prefer cheap log-likelihood evaluations over generation where valid.
   - Give GPU-h estimates from cited timings or explicit measured-pilot plans.
   - Is a vLLM venv needed for generation evaluations?
4. **Enhancement protocol.**
   - Search space, budget, arms, objective and guardrails.
   - Dev/test item split.
   - What counts as "enhanced"? For example: a test-set gain with lower bound > 0 and no guardrail breach.
   - Compare against the cheapest strong alternative: a small LoRA SFT or best-of-N steering with the same evaluation budget. The paper must report whether world-model-guided weight editing is competitive.
5. **Head-to-head with WSR on the chosen model.** Use the same edits and labels, matched information, and an LLM reader with and without the world model as a tool. Token budget: propose one.
6. **Smallest first build.**
   - What to build test-first this week?
   - Pilot measurements (timings, noise).
   - Go/no-go criteria before the large bank.
   - Budgets.

## Rules

- **Evidence:** no invented numbers. Cite files for our data. Cite URLs you fetched for external claims, and record fetch failures.
- **Length and format:** at most 2,000 words. Open with a 10-line BLUF. End with a table: "proposal | mechanism | test | cost | kill criterion".
- **Output:** write only `docs/council/2026-11-16/round8-<role>.md` in this worktree.
- **Prohibited:**
  - commits;
  - GPU use;
  - downloading models;
  - writes under `data/`;
  - edits to `docs/PREREG_H1_TF.md`;
  - opening anything under `data/evals`;
  - secrets, IPs or hostnames.
- **Roles:** `world-model`, `representation`, `metrology`, `editor`. The chair writes `SYNTHESIS-r8.md`.
