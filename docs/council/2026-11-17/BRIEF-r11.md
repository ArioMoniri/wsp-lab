# Council brief, round 11, 17 Nov 2026: decide the plan after v2b and pilot 3

## The user's request (paraphrased)

Continue with the council: decide what to do next and how to continue.

## Standing goal

The user wants a paper that, on a single popular new open-weight LLM, shows two things:

- **(a) Trajectory prediction:** a world model predicts how the LLM's benchmark behaviour evolves under weight edits,
  without running the edit.
- **(b) Enhancement:** using the world model's predictions to choose edits measurably improves the LLM on benchmarks
  it has been evaluated on.

The primary model is Qwen/Qwen3.5-9B. After that comes a replication on Qwen3.8-27B on Lambda. Resources are
authorised: the caps are ≤ 90 GPU-h-eq confirmatory and ≤ $1,000 Lambda for the 9B run (RULING-wme-v2b).

## State (cite files; read them)

**Design history.**
- `docs/council/2026-11-16/SYNTHESIS-r8.md` defines the WM-E design, the R0/R1/R2 access tiers and the claims.
- `docs/council/2026-11-17/RULING-wme-v2-design.md` and `RULING-wme-v2b.md` (A1–A11) define the WM-CP
  canonical-path world model and the claim priority A8.

**v2b outcome-blind power simulation** (`docs/reviews/wme/wme-v2b-2026-11-17.md`; manifest at
`docs/prereg/wme_manifest.json`):

| Claim | Status |
|---|---|
| H-E1 (sycophancy, TriviaQA generation, N = 1000) | Powered, 0.884 |
| H-E2 (WM-guided vs StatNN-guided search) | Powered, 0.828 |
| H-T (trajectory: WM-CP vs static family) | Not powered (C_T −0.33) |

- In the realistic saturating alternative, the static least-squares refit absorbs the saturation.
- The quadratic-world null fails at 0.41 (flag 1), because the static member's model selection overfits.
- Selected design: A8 rank 3, {E1, E2}, 65.7 GPU-h-eq, ≈ $354.

**Round 10** (`docs/council/2026-11-17/SYNTHESIS-r10.md`, D1–D10) diagnosed two problems:
- the pooled slope cut was mostly harmful refusal;
- benchmark headroom had been mis-measured: gold log-prob proxies, "final answer only" prompts, thinking off, and
  an underpowered guard.

It declared two stages:
- **S:** R1 curvature at θ₀ (S1 exact HVP, S2 jets), a router, and a learned end slope.
- **H:** generated accuracy, with edit families H1 (thinking→non-thinking LoRA), H2 (STaR), H3 (SVF policy
  gradient) and H4 (guard-orthogonal), plus a new generated-refusal guard.

**Pilot 3 is still running.**
- Plan: `docs/council/2026-11-17/RULING-wme-pilot3.md` on branch `feat/wme-pilot3`. Read it with
  `git show feat/wme-pilot3:docs/council/2026-11-17/RULING-wme-pilot3.md`; it includes amendment 1, which raised
  the S cap to 6 GPU-h.
- Interim dev numbers: `docs/council/2026-11-17/r11-interim-pilot3-dev.md`. In summary:
  - **Base:** MMLU-Pro 0.682 and SuperGPQA 0.354.
  - **H1, best checkpoints:** about 0.73–0.74 on MMLU-Pro and 0.43–0.45 on SuperGPQA, rising with training step
    and λ.
  - **H1 length-matched control:** about 0.68–0.70 and 0.35–0.37.
  - **H2 (STaR):** about no gain.
  - **H1 secondaries:** GSM8K 0.917 → 0.92–0.93, MATH 0.707 → 0.73–0.77, IFEval 0.913 → 0.93.
  - **Guards:** they look unchanged.
  - **Thinking-mode ceiling:** not valid at this budget, because it is heavily truncated.
  - **Still pending:** H3, H4, and S (the slope rerun on its own H100).

  Many checkpoints were scored on the same dev panel, so the best values are selection-biased.

## Questions the council must answer

1. **Claims.**
   - What is the registered claim set now?
   - Should H-E1 move from sycophancy to generated accuracy on model-card benchmarks (MMLU-Pro/SuperGPQA), or
     should both be kept?
   - Re-rank A8 in light of the user's goal (a)+(b).
2. **Where the world model enters enhancement.**
   - H1 is an edit, not a world-model result. Define how the world model must add value. For example:
     - predict the accuracy of unseen members of an H1-type library (checkpoint × λ × layer-group subsets ×
       merges) at R0/R1;
     - choose among them under a tight real-evaluation budget, against StatNN, direct multi-fidelity best-of-N and
       the reader competitors (SYNTHESIS-r10 D7).
3. **The trajectory claim.**
   - H-T as designed is not powered. Propose a physically valid, powerable trajectory claim. Candidates:
     - predicting held-out *training/λ trajectories* of accuracy along H1/H3 edit paths from early points plus
       R0/R1 features;
     - order-free dose curves;
     - the S-stage curvature transport, if it passes.
   - Give the matched baseline, the null that must not be beaten, and the power plan.
   - Rule on flag 1 (symmetric shrinkage).
4. **Fresh data and selection.**
   - How do pilot-3 dev results feed the design without contaminating confirmation? Cover new sealed panels,
     exposure registration of `wme-p3h`/`wme-p3s`, and the pilot-informed declaration.
   - Power for the new endpoints: use the measured discordance in `data/wme/pilot3/run-20261117/h/sizing.json`
     (read-only).
5. **Execution.**
   - Order, parallelism, GPU-h and $ for:
     - finishing pilot 3;
     - PREREG v2;
     - Board A;
     - freeze;
     - the 9B bank and confirmatory runs;
     - the 27B replication.
   - Kill rules at each gate.
   - Pre-declared consequences for each possible outcome of the pending H3/H4/S results, so that nothing becomes
     post hoc.

## Rules

- **Evidence.** No invented numbers. Cite repository files for our data and fetched URLs for any external claim;
  record fetch failures.
- **Length and format.** At most 1,800 words. Open with a 10-line BLUF. End with the table "decision | rationale |
  test or gate | cost (GPU-h, $) | kill or fallback".
- **Output.** Write only `docs/council/2026-11-17/round11-<seat>.md` in the worktree
  `/home/ubuntu/projects/wsp-lab/.worktrees/council/2026-11-17-r11`.
- **Prohibited.**
  - commits;
  - GPU use;
  - downloads;
  - writes under `data/`;
  - opening `data/evals` or any sealed split;
  - editing `docs/PREREG_*`;
  - touching running jobs or Lambda instances;
  - secrets, IPs or hostnames.
