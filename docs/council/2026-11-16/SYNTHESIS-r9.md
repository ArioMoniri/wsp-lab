# Council decision, 16 Nov 2026, round 9 (chair synthesis): contingency fixes, SSM memory, Fugu-derived upgrades and red-team controls for WM-E

## Inputs and status

**Member reports:**
- [world-model](round9-world-model.md): Claude Opus 5.5.
- [representation](round9-representation.md): Codex GPT-6.1 Sol.
- [editor](round9-editor.md): Codex GPT-6 Astra.
- [pro-engineer](round9-pro-engineer.md): Claude Opus 5.5, Claude API instance.
- [challenger](round9-challenger.md): Gemini 3.8 Flash. The first challenger session, on Grok 4.7, failed on a usage limit and wrote nothing. The user chose Gemini.

**Briefs:** [BRIEF-r9](BRIEF-r9.md) and [addendum](BRIEF-r9-addendum.md).

**Chronology:**
- This synthesis is written and committed **before the pilot-2 result exists**.
- No member and not the chair opened any pilot-2 material.
- Every gate below is therefore pre-declared for the data that will test it.

**Scope:**
- `RULING-wme-pilot1.md` §4 (the consequences of pilot 2) stays binding.
- This document is the "return to the council" that §4 names, written in advance.
- WM-E is exploratory and separate from A5. Part A remains held.

## Verdicts on the central questions

1. **Does memory help in the dynamics?** Not as memory. Weights are bookkept exactly and outcomes depend on θ_T only, so the cumulative edit E_T is a sufficient statistic. The SSM world-model literature agrees: SSMs only *match* RNNs on short, non-memory tasks (S4WM, R2I, Drama; URLs in the world-model report).
   - A selective SSM can still add **inductive bias**: saturation, and invariance to splitting a ladder step.
   - Memory genuinely helps in **planning**, by absorbing real observations from states already evaluated.
   - Both are tested below as pre-declared arms.
2. **Weight-space metanetworks** (DWSNets, NFN, NFT, UNF, GMN): deferred.
   - On a single 9B checkpoint, oriented response features come first.
   - A coarse 64-writer graph is the second ablation.
   - Full parameter-level networks are infeasible here; a dense 16-channel fp32 representation alone would be 576 GB (*computed*, representation).
3. **Fugu.** Its transferable lessons apply mostly to the **harness** and the **planner**, not to the predictor (pro-engineer).
   - sep-CMA-ES cannot run on real evaluations: one TRINITY iteration ≈ 512 evaluations, against our 32 per arm. It runs inside R0.
   - A Fugu-style multi-agent reader is a legitimate *competitor* arm (challenger).
4. **Red team.** The challenger's claims that alternatives "would decisively outperform" WM-E are treated as **hypotheses to test**, not findings. Its proposed controls are adopted (C2).

## Decisions

### C1. Mandatory code fixes before any E2 or confirmatory use (pro-engineer W4; test-first)

1. `WMEnsemble.predict` returns σ = shrink × spread, so σ is exactly 0 when shrink is 0. **Fix:** add a field-residual error floor σ_F.
2. `StaticPredictor` returns no σ. **Fix:** give it the same uncertainty treatment.
3. Ensembles are resampled by source-closure component.
4. Calibration is conformal: fit on h ≤ 2 and check coverage at h = 3–4.
   - Kill: miss > 10 pp. In that case, E2 uses ranks only and no LCB.

These fixes are applied **after pilot 2 finishes**, so the running code is not disturbed. Pilot-2 gates use real evaluations, not model σ.

### C2. Red-team controls, required in every trajectory or E2 claim (challenger, adopted)

- **Shuffled-bookkeeping null.** Permute the step-local inputs and the step order while holding E_T fixed. If WM-E's held-out error does not degrade, its "dynamics" are bookkeeping artefacts, and the trajectory claim is void.
- **Inert-edit control.** Apply edits to dead rows and to scales cancelled after normalisation. Predicted |ŷ| must stay within 2σ. This is a ghost-prediction check, the WSR Muse failure mode.
- **Trust-region stratification.** Report trajectory results inside the trust region and outside it (WikiText ΔNLL ≤ 0.1 vs > 0.1 nats/token), so that damage detection is never counted as dynamics. This answers the challenger's selection-bias objection to the wide ladders. The ladders were pre-declared, but results must also be shown by region.
- **Generation audit.** Any E1 winner must improve **generated** accuracy on held-out prompts, with loop and refusal anomalies ≤ 2%. A gain in log-probability alone never counts as enhancement.
- **Direct best-of-N arm.** A two-stage screen (quick 4-item filter → 32-item panel) with no world model, at the same real-evaluation budget, added to E2 as a secondary arm.
- **Unamortised accounting.** Report net gain per total GPU-h including bank construction (7.84–24.07 GPU-h). This is reported, not gated, because E2's primary contrast stays at equal real-evaluation budget, per SYNTHESIS-r8 D6.

### C3. If Go-T2 fails

Per the ruling, "static cumulative-edit prediction suffices" is recorded as the trajectory result. One contingency test is allowed, on **fresh `wme-p3` states**:

- **(a) Measurement first.** Neyman item allocation across objectives from pilot-1 SE and timing (pro-engineer W5), within ≤ 240 s per state.
- **(b) Oriented response encoder.** Task-delta geometry, base-gradient contractions and construction-probe sketches (representation).
- **(c) WM-S, a dose-time selective SSM** (world-model):
  - edit dose as Mamba's Δ, with zero-order hold;
  - exact zero-edit identity and exact split invariance;
  - about 17k parameters;
  - nested in F3u.
  - It must first pass the synthetic gates: **win** on long saturating ladders, and show **no gain** on the quadratic world, the sign-flip null and non-commuting scaling.
  - Its controls are WM-S-static (cumulative edit only) and WM-GRU.

**Primary contrast for (b) and (c):** each arm against the same-input static predictor on identical closure folds, trained on h ≤ 2 and tested on 3–4.

**Kill:** lower bound ≤ 0, any gain on a should-not-help task, or a WM-S win over WM-S-static on additive-only families, which can only be a leak.

**Cost:** ≤ 2 GPU-h for p3 measurement; CPU fitting.

### C4. If Go-R2 fails

1. **(b1)** Exact R1 Hessian-vector-product curvature at θ₀ replaces K-FAC ½eᵀH₀e. It must give a left-point cut ≥ 20% on fresh p3 steps. Cost ≈ 0.5 GPU-h.
2. **(b2)** A learned end slope (Heun/hypersolver) at R0 must recover ≥ 50% of the oracle cut, and is dropped below 25%.

If both fail, slope transport is dropped permanently.

### C5. If Go-E2 fails on both 9B and 4B (ordered headroom contingency; editor)

Each stage runs on fresh development panels with the same guardrails, and passes if **≥ 2 of 32 guarded development candidates** qualify. Stages run in order:

| Stage | Action | Cap |
|---|---|---|
| (1) | Generation loop diagnosis. Proceed to loop repair only if pathological looping limits completion. | ≤ 1 GPU-h |
| (2) | TFTV / contrastive weight steering on **truthfulness and factual robustness under misleading cues** (TruthfulQA-style, open generation) | ≤ 4 GPU-h |
| (3) | Task deltas and spectral (LASER-style) edits, IFEval primary, SciBench secondary, compared against the strongest donor | ≤ 4 GPU-h |
| (4) | Construction-only RL adapter on verifiable rewards, checkpoints frozen on development data | ≤ 8 GPU-h |

If all stages fail: **no enhancement claim**, and headroom is reported as the limiting finding.

### C6. Planner (pro-engineer W3)

- A beam search over sep-CMA-ES runs inside R0, with Thompson batch selection, an LCB plus support penalty, **shared identically across all ranking arms**.
- It is compared with CEM on a synthetic test of regret at B = 32.
- Kill: sep-CMA-ES ≤ CEM, in which case CEM is used.

### C7. Optional upgrades

Each is used only after its own pre-declared test passes.

| Upgrade | Source | Kill rule |
|---|---|---|
| W1 listwise soft targets (τ = paired SE; 1/SE² weights; 2-seed LoRA mean) | pro-engineer | No synthetic top-k regret gain at k = 4 |
| W2 SVF action family ΔW = U diag(σ⊙z)Vᵀ, with R0 a/q contractions | pro-engineer | Equivalence mismatch, or it resolves less than rank-one |
| Planning filter: RSSM-style posterior over real tap observations, against **anchor-shift static** (same observations, no state) | world-model | Needs C ≥ 0.10 with LB > 0 |
| Gated DeltaNet tap (gate-saturation margins and state norms from the 24 linear-attention layers), against equal-width random/PCA sketches | world-model | No gain above noise |
| Coarse 64-writer graph metanetwork, against a shared token MLP | representation | No incremental LB > 0 |

### C8. Harness (pro-engineer H1–H5)

Adopted for all future rounds and jobs:

| Item | Change | Test |
|---|---|---|
| H1 | Each council seat gets **its own worktree**, with an explicit access list and a read audit. Round 9 seats shared one directory; the challenger was told not to read other reports. | A planted-file read test |
| H2 | A shared fetch-status ledger, recording URL status and mirrors, never conclusions | — |
| H3 | A format lint plus a **verifier seat** that checks every number against its source, before chair review | — |
| H4 | A longest-item-first preflight before GPU loops | — |
| H5 | Resumable, content-addressed units of work, and a resident apply/evaluate/restore server | Kill, resume and compare hashes |

H1 and H3 apply from round 10. H4 and H5 are built test-first before the bank.

### C9. 27B scale-up on Lambda (user-authorised; gated)

**Target:** Qwen/Qwen3.8-27B.

**Start conditions:** only after **both** hold:
- T2 and E2 pass on 9B, or their contingencies pass under C3/C5;
- `docs/PREREG_WME.md` is written.

**Engineering (L1–L4):**
- one bf16 model per 80 GB GPU (≈ 54 GB);
- data-parallel over states;
- fixed-order fp64 CPU reduction, with a test that 1 GPU and k GPUs agree to rtol 1e-6;
- contract-and-discard fields;
- an **external dead-man terminator** that holds the API key, enforces idle and T_max limits, and alerts if an instance is alive more than 15 min past T_max.

**Calibration:** ≤ 2 Lambda GPU-h first, with new memory and headroom gates for 27B. Kill: peak > 72 GB, or no scientific gate passed.

The acudx instance is never touched.

### C10. Competitor arm in the Fugu style

- **Design:** a TRINITY/Conductor-style Thinker/Worker/Verifier team of LLM readers with access lists. It ranks candidates under the same budget ledger (32 real-evaluation bundles), with LLM tokens charged.
- **Implementation:** built with our own providers. **Fugu itself needs a Sakana or OpenRouter key, which we do not have.** The user can add one; it is then a sub-arm.
- **Status:** secondary in E2. The primary contrast stays WM vs StatNN (SYNTHESIS-r8 D6).

## Summary table

| proposal | mechanism | test | cost | kill criterion |
|---|---|---|---|---|
| C1 σ fixes | σ floor, StatNN σ, closure bootstrap, conformal | Coverage at h = 3–4 | < 1 CPU-h | Miss > 10 pp → ranks only |
| C2 red-team controls | Shuffled bookkeeping, inert edits, trust-region strata, generation audit, best-of-N, unamortised cost | Required in every claim | ≈ 0.6 GPU-h + CPU | Any control fails → that claim is void |
| C3 trajectory contingency | Neyman items; oriented encoder; WM-S dose-time SSM | Synthetic gates, then fresh p3, h ≤ 2 → 3–4, against same-input static | ≤ 2 GPU-h | LB ≤ 0; should-not-help gain; additive-only leak |
| C4 slope contingency | R1 exact HVP; learned end slope | Fresh p3 steps | ≈ 0.5 GPU-h | < 20% / < 25% |
| C5 headroom contingency | Loop diagnosis → TFTV/CWS truthfulness → task deltas/LASER (IFEval) → construction-only RL | ≥ 2 of 32 guarded development candidates per stage | ≤ 1 / 4 / 4 / 8 GPU-h | All fail → no enhancement claim |
| C6 planner | sep-CMA-ES inside R0, shared across arms | Synthetic regret at B = 32 | ≈ 2 CPU-h | ≤ CEM |
| C7 upgrades | Soft targets, SVF actions, planning filter, DeltaNet tap, coarse graph | Each its own ablation | CPU + ≤ 0.65 GPU-h | As listed |
| C8 harness | Isolation, fetch ledger, verifier, preflight, resumable server | Planted-read and kill-resume tests | Minutes to 4 CPU-h | Plant missed / hash drift |
| C9 27B on Lambda | Data-parallel 80 GB, dead-man terminator | 1-vs-k parity; new 27B gates | ≤ 2 Lambda GPU-h calibration | > 72 GB or no gate passes |
| C10 Fugu-style competitor | Thinker/Worker/Verifier readers with access lists | Secondary E2 arm, equal ledger | LLM tokens (charged) | Reported either way |
