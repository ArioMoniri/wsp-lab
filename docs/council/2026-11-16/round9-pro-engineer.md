# Round 9, pro-engineer seat (16 Nov 2026): buildable improvements to WM-E, the harness and the 27B scale-up, from Sakana Fugu

**Inputs.** Both briefs, `SYNTHESIS-r8.md`, `RULING-wme-pilot1.md`, the pilot-1 report, `src/wsp/wm/`, F12-S1/S2 notes (read only). Not opened: pilot-2 material, `data/`, other round-9 reports. No GPU.

## BLUF

1. Fugu's lessons transfer to the **harness** more than to the predictor: isolate proposers, share tool memory not conclusions, gate format before substance (H1–H3).
2. **sep-CMA-ES cannot run on real evaluations.** One TRINITY iteration is 16 × 32 = 512 evaluations (computed); our budget is 32 per arm. Use it inside R0, under a calibrated LCB (W3).
3. **Code finding 1.** `WMEnsemble.predict` returns σ = shrink × spread, so σ is exactly 0 when the selected shrink is 0.
4. **Code finding 2.** `StaticPredictor` returns no σ. Unfixed, E2 compares uncertainty machinery, not dynamics (W4).
5. **SVF edits** ΔW = U diag(σ⊙z)Vᵀ give a fixed low-dimensional action basis, which the synthetic gain required. They also make a/q cheap R0 contractions (W2).
6. **Fugu's Stage-1 analogue** is a listwise soft-target loss with τ = paired SE. Seeds are the wrong analogue, because our panel is bitwise deterministic (W1).
7. **Measurement power is an unused knob.** |change|/SE runs from 0.86 (GSM8K) to 4.9 (WikiText); Neyman item allocation targets the Go-T miss (W5).
8. **Failed runs cost 0.60 of 2.25 pilot GPU-h,** which a longest-first preflight targets (H4). F12-S2 lost 11.56 CPU-h because the work was not resumable (H5).
9. **27B:** data-parallel over states, fixed-order CPU reduction, contract-and-discard fields, external dead-man terminator (L1–L4).
10. Each proposal has a CPU-first test and kill criterion.

## (a) WM-E prediction quality

### W1. Soft targets, translated correctly

**What Fugu does.** Fugu's Stage 1 runs each worker *n* times and trains by KL toward softmax(mean reward/τ). n, τ and a soft-vs-hard ablation are unreported (§3.1.2, https://arxiv.org/html/2606.21228v2).

**Why seeds don't transfer.** Our panel repeats bitwise (Go-N), so noise is item sampling, measured as paired SE. Translations:

- **(i) Listwise term.** Add KL(softmax(y/τ)‖softmax(ŷ/τ)) to `Member.loss_by_horizon` over an episode's candidates, τ = median paired SE: E2 ranks, and sub-noise differences become ties.
- **(ii) Weight by 1/SE².** The loss currently scales by residual std (`model.py`, `fit`). SE differs 34× across objectives (WikiText 4.12e-4 vs GSM8K 0.014; computed from the pilot report).
- **(iii) LoRA seeds.** The rank-8 LoRA task vectors (`RULING-wme-pilot1.md` §3) do have seed variance. Train 2 seeds; use their mean as the action and their spread as label noise.

**Test and kill.** Synthetic top-k regret over 5 world seeds. Kill if there is no gain at k = 4.

### W2. SVF as action family and encoder

**Evidence.** Transformer² tunes only singular-value scales and reports beating LoRA with fewer parameters (https://arxiv.org/abs/2501.06252). In TRINITY, removing SVF drops the average from 70.44 to 67.85 (Table 2, https://arxiv.org/html/2512.04695).

**For writer W = UΣVᵀ, define the edit ΔW = U diag(σ⊙z)Vᵀ.**
- **Bookkeeping.** z lives in fixed per-writer coordinates. That is the "low-dimensional functional state reached by many directions" the synthetic gain needed (pilot report §1).
- **Field.** a_i = σᵢuᵢᵀGvᵢ uses the gradient G already collected. Under K-FAC, q = Σᵢⱼ zᵢzⱼσᵢσⱼ(vᵢᵀAvⱼ)(uᵢᵀGuⱼ). Both are computed once at θ₀, so they are R0, and raw factors become discardable (L3).
- **Realisation.** Via `actions.additive`/`realise`, never rebuilt in bf16, so z = 0 stays bitwise identity.

**Test/kill.** Contractions match stored factors to rtol 1e-6 (CPU toys); kill on mismatch, or if SVF steps resolve less than rank-one at matched norm (9B dev, exploratory).

### W3. Planner: sep-CMA-ES inside R0

**sep-CMA-ES.** It uses a diagonal covariance with linear cost (Ros & Hansen, abstract https://hal.archives-ouvertes.fr/inria-00287367; I reached it through search only, because direct fetch hit an Anubis block and CORE returned 403).

**What TRINITY shows.**
It beats REINFORCE, random search and SFT (MATH500 0.880 vs 0.459/0.794/0.786, Table 4) but with λ = 32, 16 replicates and "1.5k–40k evaluations" (A.1.2). Our budget: 32 per arm per episode (`SYNTHESIS-r8.md` D6).

**Design.**
- **Inner loop (R0, free).** A beam over action identities (≤ 4 actions). sep-CMA-ES runs over doses and SVF z at each beam node. The objective is calibrated LCB minus a support penalty. MOPO's uncertainty-penalised reward is the precedent against model exploitation (https://arxiv.org/abs/2005.13239).
- **Outer loop.** Each batch slot optimises one posterior draw (Thompson-style). Evaluate for real, then refit.

**E2 fairness.** The planner code is identical for the WM and StatNN arms; only the predictor differs.

**Test/kill.** Synthetic real regret at B = 32 vs CEM, random, greedy; keep CEM unless sep-CMA beats it.

### W4. Calibrated σ (code fixes)

- **(i) Field-error floor.** Use σ_total = √(σ_F² + (λσ_ens)²), where σ_F is the field-only residual std on the selection split.
- **(ii) Bootstrap by closure component.** Members currently differ only by initialisation (`Member(cfg, s)`). Independent units are source-closure components (D4), and the bootstrap should resample those (deep-ensembles rationale, https://arxiv.org/abs/1612.01474).
- **(iii) Same σ for `StaticPredictor`.**
- **(iv) Conformal scaling.** Split-conformal scaling of |err|/σ (https://arxiv.org/abs/2107.07511), calibrated on h ≤ 2 and tested at h = 3–4.

**Kill.** 80% coverage at h = 3–4 is missed by more than 10 pp. Then the planner uses ranks only.

### W5. Allocate measurement power

**Pilot-1 ladder medians of |change|/SE** (computed from the pilot report): WikiText 4.9, harmful 3.7, harmless 3.9, sycophancy 1.4, MMLU-Pro 0.92, GSM8K 0.86. Go-T missed at 0.496 against 0.50.

**Arithmetic.** SE ∝ 1/√n: doubling a ratio needs 4× items. Post-pilot-2 designs set item counts by Neyman allocation from pilot-1 SEs and seconds only (5–19 s per 64 items), within the 240 s Go-C cap. Four times the items on two objectives adds ≤ 2 × 3 × 19 ≈ 114 s per state (computed).

**Kill.** The projected resolvability stays below 0.50 at the cap.

### W6. Two-stage labels and active learning, under guard

**PPI.** Prediction-powered inference gives valid intervals from many predictions plus few gold labels (https://arxiv.org/abs/2301.09633). Use it for the generation endpoint: teacher-forced log-probability is the predictor, and replicated decoding on a subset is the gold label.

**Active learning.** BatchBALD-style diverse acquisition (https://arxiv.org/abs/1906.08158), under three constraints:
- it chooses **training** roots only;
- test trajectories stay on the pre-declared random design;
- acquisition is arm-neutral, using field-only residuals and never WM-vs-StatNN disagreement.

**Kill.** No gain over random at equal visits (synthetic).

## (b) Harness and orchestration

### H1. Isolate proposers

**Sources.**
- Fugu-Ultra lets agents see others' outputs "only through the access list", to avoid "orchestration collapse", where the first actor fixes the trajectory (§3.2.2).
- Conductor's fine-grained topology gave no "significant performance gains" over the binary scheme (Appendix B.8, https://arxiv.org/html/2512.04388v5).

**Our leak.** Other seats' round-9 reports sit in this worktree (unopened by me).

**Fix.** One worktree per seat, a read access list in each brief, and a read-path audit of the session logs.

**Kill.** The audit misses a planted read.

### H2. Share tool memory, not conclusions

**Source.** Fugu shares earlier workflows' tool calls while keeping reasoning isolated (§3.2.2).

**Our waste.** Retrieval failures repeat across rounds (`MEMBERS.md`); I hit HAL/CORE blocks.

**Fix.** A shared fetch ledger: URL → status, hash and working mirror. No summaries.

**Kill.** No fall in failed fetches over two rounds.

### H3. Format gate, then verifier

**Gate.** Conductor scores parse failure 0, wrong 0.5, correct 1 (§3.1). Analogue: a lint for word count, BLUF, table columns, a citation per number, and no forbidden reads.

**Verifier.** Then a seat like TRINITY's ACCEPT/REVISE Verifier (ablated only jointly, Table 2), reading only the report, repository and fetch ledger, checking each number against its source.

**Caution.** Frontier models as conductor were "over-reliant on an initial pool" (70.05/71.59 vs trained 75.65, Table 11); audit the chair's routing too.

### H4. Fail fast on the worst case

**What pilot 1 cost.**
- The 31.6 GB device breach (allocator reading 27.6 GB) surfaced at 419 s.
- The long-row OOM surfaced at 384 s.
- Failed runs cost 0.60 GPU-h (pilot report §5).

**Preflight.** Longest-first worst batch per (objective, group), reading device memory, not just the allocator; abort before the main loop. Hash at write time (P4 re-hashed ~109 GB at ~100 MB/s).

**Kill.** The preflight misses a later breach.

### H5. Resumable units and a resident state server

**What happened.** An unattended-upgrade restart of `t3-agent-tmux` killed F12-S2 part 1 at 400 of 1,200 units, and the full rerun cost 34.54 CPU-h (`PREREG_H1_TF.md`).

**Fix.**
- **Atomic per-unit results** keyed by sha256(recipe ‖ θ₀ ‖ panel); a kill loses ~25 s.
- **Resident server** holding the model: apply → evaluate → restore (`EditSession.restore` is bitwise, P3), removing the 137–159 s load per job.
- **Apt timers.** Pre-launch check; the owner sets policy.

**Kill.** The θ-hash drifts after a restore.

## (c) The 27B scale-up on Lambda

### L1. Data-parallel evaluation

27B bf16 ≈ 54 GB (computed) fits one 80 GB GPU: one model per GPU, states as queue items, no communication.

**Fallback.** FSDP2 `fully_shard` (dim-0 per-parameter sharding with CPU offload; https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html), only if Go-M fails per GPU.

### L2. Deterministic statistics

**Design.** Shard items across GPUs. Write fp32 partial sums per (objective, group). Reduce on CPU in a fixed order, in fp64.

NCCL all-reduce order can depend on topology; fixed order keeps Go-N bitwise.

### L3. Contract and discard

At 9B, four layers wrote 25 GB of factors (P2), ~200 GB for 32 layers (computed). Contract to dictionary a/q (W2) per (group, objective), persist megabytes, discard factors, sync only these.

### L4. Availability and cost

**What the Lambda docs say** (https://docs.lambda.com/api/cloud): launch is limited to one request per 12 s, and `provider/` errors can mean "quota limitations or resource exhaustion". The section I read does not mention spot instances, so assume on-demand with capacity risk.

**Controls.**
- **Launcher.** Backoff, keyed on `code`.
- **External controller.** It holds the API key (never on the GPU box) and terminates the instance on any of:
  - queue drained and sync verified;
  - 15 minutes idle;
  - T_max, unconditionally.
- **Budget ledger.** It projects GPU-h × rate and stops spending at a projected cap, like F12.
- **Parity first.** Go-M/N/C on one 27B GPU before multi-GPU.

## Closing table

| proposal | mechanism | test | cost | kill criterion |
|---|---|---|---|---|
| W1 soft targets | listwise KL, τ = SE; 1/SE² weights; 2-seed LoRA mean | synthetic top-k regret | <1 CPU-h | no gain at k = 4 |
| W2 SVF actions | additive U diag(σ⊙z)Vᵀ; R0 a/q contractions | CPU equivalence; synthetic WM vs static | ~1 CPU-h | mismatch; resolves less than rank-one |
| W3 R0 planner | beam × sep-CMA-ES, Thompson batch, LCB + support penalty, shared across arms | synthetic regret at B = 32 | ~2 CPU-h | sep-CMA ≤ CEM |
| W4 calibrated σ | σ_F floor; closure bootstrap; StatNN σ; conformal | coverage at h = 3–4 | <1 CPU-h | miss > 10 pp → ranks only |
| W5 Neyman items | allocation from pilot-1 SE and seconds | projected resolvability ≤ 240 s per state | arithmetic | < 0.50 |
| W6 PPI + AL | PPI for generation; arm-neutral AL on training roots | synthetic error; PPI coverage simulation | ~1 CPU-h | no gain; coverage miss |
| H1 isolation | per-seat worktree, access list, audit | planted read | minutes | plant missed |
| H2 fetch ledger | shared fetch status, no conclusions | failed fetches per report | minutes | no fall in 2 rounds |
| H3 lint + verifier | format gate, numeric verifier | seeded false number | one seat-run | plant missed |
| H4 preflight | longest-first worst-batch probe | replay pilot-1 configs on toys | minutes | later breach |
| H5 resumable server | content-addressed units; resident apply/eval/restore | kill, resume, compare hashes | ~4 CPU-h | θ-hash drift |
| L1–L2 27B DP | one model per GPU; fixed-order fp64 reduction | 1 vs k GPU rtol 1e-6; bitwise repeats | ~1 GPU-h | mismatch → FSDP2 |
| L3 contract-discard | persist dictionary a/q only | vs stored factors, one 9B group | ~0.2 GPU-h | mismatch |
| L4 cost controls | external dead-man terminator, budget barrier | kill controller on a 1-GPU instance | <1 GPU-h | alive > T_max + 15 min |
