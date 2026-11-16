# Chair ruling on the WM-E pilot-1 gates, and pre-declaration of pilot 2 (16 Nov 2026)

**Inputs**

- Pilot report: `docs/reviews/wme/wme-v0-pilot-2026-11-16.md` and `.json`, merged on main at d66ef2e.
- Spec: `SYNTHESIS-r8.md`, D8.

**Timing.** This ruling is written and committed **before any pilot-2 data exists**. Every pilot-2 rule below is fixed now.

**Scope.** WM-E is exploratory and separate from A5. Part A remains held and untouched.

## 1. Pilot-1 gates stand as declared

| Gate | Result | Ruling |
|---|---|---|
| Go-M, Go-N, Go-C | PASS | 9B stays the target. The 4B fallback is not triggered. |
| Go-T | FAIL (0.496 vs 0.50; 1.91 SE vs > 2) | The pre-declared consequence is recorded: **for the pilot-1 design, static cumulative prediction suffices.** The bank is not built on that design. |
| Go-R | FAIL (left-point −4.8%) | Left-point slope transport is dropped. The trapezoid result (−34.8%) is descriptive only, because it was not pre-declared. |
| Go-E | FAIL (1 of 24) | No enhancement search on the pilot-1 action space. |

The P3 cap breach (31.6 GB on the first P2 run, stopped by the monitor) is recorded. The fix (scored-position logits, a 27 GB allocator cap, a ≤ 1024-token field filter) is accepted.

## 2. Design defects found, fixed prospectively for pilot 2 only

- **(a) The negative control was counted in the resolvability gate.** Resolvability included the random-direction projection, which resolved 0.07 by design. Excluding it, pooled resolvability was 0.58. That figure is descriptive and does not change the pilot-1 result. From pilot 2 on, negative controls are reported separately and excluded from the Go-T resolvability count.
- **(b) Doses did not leave the trust region in a controlled way.**
  - Steering at 0.25 × residual norm had already left it by dose 1.
  - Scale at 1.25 resolved only from |dose| 0.5.

  Pilot 2 uses wider ladders built from the pilot-1 nominals, so trajectories deliberately cross from inside the trust region to outside it.
- **(c) The sycophancy endpoint was a proxy built from MMLU-Pro.** Pilot 2 uses a real sycophancy dataset. The proxy is kept as a descriptive endpoint.

## 3. Pilot 2 (pre-declared; at most 6 GPU-h; same 28 GB GPU / 24 GB RSS caps; one heavy job at a time; agent-tmux)

### T2: trajectory gate

**Roots**
- Fresh roots only, from the seed namespace `wme-p2`. No pilot-1 root, direction or seed is reused.
- Families: scale, steer (rank-one), and proj_remove (contrast directions), with **two roots each**. Random-direction proj_remove is included as a negative control.

**Doses**
- Signed ladders at cumulative dose ×{0.25, 0.5, 1, 2, 4} of these nominals:
  - scale: factor 1.5;
  - steer: shift 0.125 × residual norm;
  - proj_remove: α = 0.5, capped at α ≤ 1.
- A state whose WikiText ΔNLL exceeds 1.0 nats/token is kept as a labelled endpoint. Its ladder stops there.

**Compositions**
- Eight 3-step and four 4-step compositions across families, each with its reverse order.

**Go-T2**
- On the non-control steps, the fraction resolvable is ≥ 0.50.
- **The primary endpoint is fixed now:** the median over all six objectives of the refitted-F3u static error at h ≥ 3, in paired-SE units. It must exceed 2. This is the same aggregate as pilot 1, so no single objective is chosen after the fact.

**Go-R2**
- Uses the trapezoid oracle slope rule, pre-declared now.
- It is evaluated on the pilot-2 trajectories and must cut per-step error by ≥ 20% relative to A₀.

### E2: headroom gate

**Objectives** (all three pre-declared, each judged separately):
- (i) A real sycophancy evaluation: an open sycophancy dataset downloaded and hashed before scoring. The MMLU-Pro proxy is descriptive.
- (ii) GSM8K gold-answer log-probability.
- (iii) Competition-math gold-answer log-probability at levels 4–5, from an open MATH copy disjoint from MATH-500 by problem hash.

**Action space**
- Rank-one steering at layers ≥ 20 with |shift| ≤ 0.0625 × residual norm, sampled without replacement.
- Plus **scaled small-LoRA task vectors**, λ ∈ {0.25, 0.5, 1}, per objective. Each is a rank-8 LoRA SFT trained on construction-split items only, at most 0.3 GPU-h per objective, with training compute charged.
- 32 candidates per objective, scored on **dev items only**.

**Guardrails** (as in D6): WikiText ΔNLL UCB ≤ 0.01 nats/token, and the other objectives' LCB ≥ −1 pp equivalent on their log-probability scale.

**Go-E2 and choice of objective**
- An objective passes when ≥ 2 of its 32 candidates have a dev-gain LCB > 0 with no guardrail breach.
- The passing objective with the most qualifying candidates becomes **the primary enhancement objective**. Ties go to the order (iii) > (ii) > (i).
- This selection uses dev items only. Test items remain sealed.

**If Go-E2 fails for all three objectives on 9B:** run the identical E2 on Qwen3.5-4B (revision `851bf6e8…`) before any bank. This extends the D1 fallback rule, and the extension is declared now. A 4B result is never pooled with 9B.

## 4. Consequences, fixed now

| Outcome | What happens |
|---|---|
| T2 passes and E2 passes | Build the bank (priced at 7.8 GPU-h without per-state oracle gradients). Write `docs/PREREG_WME.md`. Run the outcome-blind power simulation, then the confirmatory tests. |
| T2 fails, E2 passes | Report the trajectory finding as **"static cumulative-edit prediction suffices on Qwen3.5-9B for these families"**, a result in its own right. The world model is then tested on the enhancement claims E1 and E2 only. |
| E2 fails on both 9B and 4B | No enhancement claim. Report headroom as the limiting finding, then return to the council. |
