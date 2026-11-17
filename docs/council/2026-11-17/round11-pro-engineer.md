# Round 11, pro-engineer seat: execution plan (17 Nov 2026)

Paths are relative to `data/wme/pilot3/run-20261117/` (read-only). "est." = scaled from a measurement.

## BLUF

1. **Measured unit.** One candidate eval takes **226–312 s on one H100**, or 12–16 evals per GPU-h (`h/ledger.jsonl`). An eval is MMLU-Pro + SuperGPQA at 1,000 items each, with a 2,048-token budget, plus 612 guard items and the proxy.
2. **Pilot 3 needs nothing from us.** It ends by 05:25 UTC on 18 Nov (`h/deadline`). Worst case it uses ≤ 24.97 instance-h, about **$121**, against a $150 cap. H4 will probably be starved by the deadline.
3. **E1-acc** (one pre-declared H1 recipe on a fresh sealed panel) costs **≈ 1 GPU-h**. Register it alongside sycophancy E1.
4. **Where the WM enters:** E2-acc, a contest of WM vs StatNN vs direct multi-fidelity (MF) best-of-N on an H1 library at equal ledger. It costs **≈ 22 GPU-h** (est.).
5. **Trajectory:** T-acc predicts held-out H1 training paths. It reuses the library machinery and costs **≈ 4.8 GPU-h** (est.).
6. **9B with everything kept:**

   | Item | GPU-h | Cap |
   |---|---|---|
   | Bank and library | 21.9 | 35 |
   | Confirmatory | 81.5 | 90 |
   | Lambda total, including pilot 3 | ≈ $700 | $1,000 |

7. **At most 4 concurrent 1× H100 SXM5 instances**, each with its own dead-man state directory and no PCIe fallback. The 9B result is about 2.5 days away (est.).
8. **Blockers before the bank:**
   - the heartbeat measures liveness, not progress;
   - both guard pools are used up;
   - there is no cross-instance base canary.
9. **27B needs a 2× H100 worker.** At $600, E1-acc and T-acc fit. E2-acc needs re-pricing.
10. **H3, H4 and S outcomes map to the plan by the fixed table in §4.**

## 1. Measured inputs

| Quantity | Value | Source |
|---|---|---|
| One eval (vLLM 0.34.0 at 0.45 utilisation, max_num_seqs 128) | 226–312 s: about 234 s for H1, about 307 s for base and H2 | `h/ledger.jsonl` |
| Teacher data: 2,000 items in thinking mode | about 1,957 s | same |
| H1 LoRA training (208 steps) | 576 s | same |
| Peak device memory, vLLM and the HF copy in one process | 58.7 GB | same |
| Paired discordance at the probe | MMLU-Pro 0.129, SuperGPQA 0.191 | `h/sizing.json` |
| Base re-run with reversed item order | moves accuracy 0.3–0.8 pp | interim |
| S cost check | jet = 6.02 forwards (stop at 8); HVP = 5.18 gradient passes (stop at 6); projected 4.66 GPU-h | `s/cost_check.json` |
| H100 SXM5 prices | 1× $4.89, 2× $9.58, 4× $18.76; capacity binds | `wme-27b-calib-2026-11-16.md` §6 |
| 27B model | 53.79 GB bf16; 16 attention layers; HF generation 30.5 s vs 5.45 s on 9B | `wme-27b-calib` §1, §3 |

**Sample sizes.** N = 7.84·d/δ² (RULING-wme-pilot3 §4.6).

| Effect | MMLU-Pro | SuperGPQA |
|---|---|---|
| 3 pp | 1,123 | 1,668 |
| 2 pp | 2,527 | 3,752 |

**Fresh pools** (`wme-p3h/manifest.json`, usable items minus those used, est.):
- MMLU-Pro ≈ 8,300 left.
- SuperGPQA ≈ 24,000 left.
- **The guard pools are exhausted.** All 1,319 or-bench-hard items and all 653 or-bench-toxic items are used. New sources must be pinned before the freeze. I name none, because I made no fetch.

## 2. Plan

$ = GPU-h × $4.89 × 1.10, v2b's convention.

| # | Phase | After | Instances | Wall | GPU-h | $ |
|---|---|---|---|---|---|---|
| P0 | Pilot 3 (H2 rest → H3 → H4; S2) | running | 2 | ≤ 05:25 UTC | H ≤ 9.7 more, S2 ≤ 6.5 | total ≤ $121 |
| P0′ | CPU engineering: §3 fixes, panel builder v2, library generator, eval server, queue driver | now | — | ≈ 10 h | 0 | 0 |
| P0″ | P-9B-eval (0.9 utilisation; rates at 250, 1,000 and 2,600 items; R0 proxy cost) and P-27B-gen (vLLM TP1 eval, training memory). Construction items only. | now | 2 | ≤ 1.5 h | ≤ 2 | ≤ $11 |
| P0‴ | Flag-1 quadratic-null refit (600 replicates) | now | 2 A10 (CPU) | ≈ 3 h | CPU | ≤ $8 (est.) |
| P1 | Pilot-3 report; apply §4 | P0 | — | 2 h | 0 | 0 |
| P2 | PREREG v2 → Board A → freeze. Sealed panels built as hashes before the freeze. | P1 | — | ≈ 12 h | 0 | 0 |
| P3a | Library-acc: 6 H1 runs, every step saved. Members are checkpoint × λ × layer group. 96 labels on a fresh bank panel; R0 proxy on 384 members. Teacher data reused. | P2 | 2 | ≈ 5 h | 9.8 (est.) | $53 |
| P3b | Bank-syc (v2b design) | P2 | 2 | ≈ 6.5 h | 12.1 | $65 |
| P4a | E1-acc: base, base re-run, candidate, length-matched control, fresh-seed retrain. Sealed test: MMLU-Pro 2,600, SuperGPQA 3,800. | P3a | 1 | ≈ 1.5 h | ≈ 1 (est.) | $5 |
| P4b | E2-acc: 4 arms × 12 episodes. Screens at 250 items with B = 16; the pick is scored on sealed test. | P3a | 2 | ≈ 11 h | ≤ 22 (est.) | $119 |
| P4c | T-acc: 4 held-out runs × 16 checkpoints | P3a | shared with P4b | ≈ 2.5 h | 4.8 (est.) | $26 |
| P4d | Sycophancy E1/E2 (v2b) | P3b | 2 | ≈ 14 h | 53.7 | $289 |

**Pricing rule.** If a P0″ rate is more than 1.5× its estimate, the item is re-priced by a dated amendment before any of its data exist. The ≈ 8.5 GPU-h of confirmatory headroom is held for crash reruns only.

## 3. Harness rules

1. **Dead-man per instance.**
   - Each instance has its own state directory, `t_max` = cap + setup, and an idle limit of 1,200 s.
   - On `DONE`, the driver syncs, verifies sha256 hashes, issues `request-terminate`, confirms `terminated` through the API, and writes `HB_STOP`.
2. **Heartbeat fix (blocking).**
   - `wme_p3_remote.sh` touches `logs/alive` every 60 s.
   - `wme_lambda_hb.sh` counts any change in `logs/` as progress.
   - As a result, a hung vLLM or NCCL process stays alive until `t_max`.
   - The fix: send the heartbeat only on `ledger.jsonl`/`evals.jsonl` growth, with a stall window of at least 30 min. The longest measured unit is 1,255 s.
3. **Naming.** `wsp-wme-<track>-<phase>-<yyyymmdd>[-k]`, for example `wsp-wme-c2-e2acc-…` or `wsp-wme-27b-p-…`.
4. **Platform.**
   - The bank and confirmatory phases run on SXM5 only. If there is no capacity, wait.
   - Pilots may use PCIe.
5. **Isolation.**
   - Each Lambda instance builds its own uv venv: vllm 0.34.0, torch 2.14.0+cu130, driver ≥ 580 or cuda-compat.
   - Locally, use `scripts/venv_track.sh` and `scripts/cuda_env.sh`, never the shared `.venv`. A pin mismatch stops the job.
6. **Numerics.**
   - Freeze the batch, max_num_seqs, utilisation, item order and budget (v2b flag 3).
   - Each instance first scores a 64-item base canary. On a non-bitwise match, it pairs only with a base run on that instance.
   - A base/candidate pair always shares one instance.
7. **Sync and recovery.**
   - Save adapters at every step.
   - `evals.jsonl` is append-only, holds whole candidates only, and is keyed by `spec_key`.
   - Pull `out/` every 10 min.
   - A resume skips completed keys and discards partial units.
   - After the freeze, a resume must use the identical commit. Pilot 3's three new-code resumes (`h/remote/logs/remote.log`) would not be allowed.
8. **Status file.** `h/STATUS` reads a stale `FAILED` (12:26); the report must read `remote/STATUS`.

## 4. Pre-declared consequences

**Expected path, by the cap arithmetic.**
- H2 has 4,928 s of cap left, which covers about 16 of its 18 remaining evals.
- H3 then has up to 8 h.
- H4 gets only what is left before 05:25.

| Outcome | Plan | Budget |
|---|---|---|
| H3 passes | SVF-PG becomes a second library family (2 runs) and gives T-acc its RL paths | ≤ +6 GPU-h (est.), ≤ $32 |
| H3 fails or is undecidable | Undecidable counts as fail (§0). The library is H1 only, with H2 as near-null members. No pilot rerun. | 0 |
| H4 passes | The projection becomes an R0 library operator. The E1-acc candidate is projected if the unprojected candidate fails the dev guard. | 0 |
| H4 fails, not run, or partial | Recorded as "not measured"; it carries no claim | 0 |
| S passes | Transport is kept only for the passing objectives, in bank-syc, with jets priced from the measured S2 cost | within the 35 cap, or for passing objectives only |
| S fails, is inconclusive, or not measured | Transport dropped; oracle gradients removed | 0 |
| H1 gain only on truncation-affected items | The untruncated subset becomes co-primary, or E1-acc becomes descriptive, as the chair rules before the freeze | 0 |
| Proxy killed | The WM drops the proxy feature; E2-acc still runs | −1.6 GPU-h |

## 5. 27B

**Memory.**
- vLLM TP1 holds the 53.79 GB of weights, leaving about 18 GB of KV cache (est.).
- An HF trainer needs about 66 GB (est.).
- Both cannot fit on one 80 GB GPU. Use a `gpu_2x_h100` worker ($9.58/h): GPU0 trains, GPU1 serves.

**Throughput.** It is 3× (parameters) to 5.6× (the HF ratio) the 9B unit, i.e. 680–1,750 s per eval (est.). P-27B-gen measures it.

**Scope at $600.**

| Item | GPU-h (est.) | $ (est.) |
|---|---|---|
| E1-acc | 5–10 | — |
| T-acc | 14–27 | — |
| Both, with ×1.5 two-GPU overhead | — | ≈ $155–300 |
| E2-acc | 66–123 | $355–660 |

E2-acc fits only if the measured ratio is ≲ 2.6 (est.). Otherwise it needs a user re-pricing, or runs descriptively with 6 episodes.

**Start gate:**
- 9B E1-acc is confirmed;
- the P-27B peak is ≤ 72 GB per GPU.

## 6. Other questions (engineering)

**Q1.**
- E1-acc costs ≈ 1 GPU-h; sycophancy E1 costs 17.5. Keep both.
- Under pressure, drop sycophancy E2 first (≈ 48 GPU-h, $259).
- Power E1-acc at 2 pp, which is about 0.5× the selection-biased dev gains (+4.4/+9.3 pp), as in A10.

**Q2.**

| Arm | Cost per arm-episode |
|---|---|
| Direct best-of-N, B = 32 full evals | 2.0–2.8 GPU-h |
| MF ladder, B = 16 (screens est. ≈ 70 s, including the test eval) | ≈ 0.46 GPU-h |

- The WM pays extra for its R0 features (≈ 1.6 GPU-h, unamortised, D7) plus CPU training.
- The WM can win only if the library ≫ B (≥ 384 vs 16).

**Q3.**
- Baseline: a per-run saturating fit on the first 4 checkpoints.
- Null not to beat: the pooled mean curve.
- Power by outcome-blind simulation, with √(d/N) SEs: 0.011 and 0.014 at 1,000 items.
- Flag 1: give the static family WM-CP's λ rule, then re-check (P0‴).

**Q4.**
- Register every `wme-p3h`/`wme-p3s` hash as exposed, including unscored dev pool items, and assert it in the builder.
- Pilot numbers enter only as declared design inputs.

## Decision table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Let pilot 3 finish untouched | caps and deadline bind | `h/deadline`, stage caps | ≤ 16.2 more; total ≤ $121 | unevaluated candidates = non-qualifying |
| Fix heartbeat, add canary, re-source guards | hung jobs; drift; empty pools | unit test; bitwise canary; null-pass ≥ 0.8 | 0 | no bank launch until fixed |
| P-9B-eval, P-27B-gen | prices are scaled | rate ≤ 1.5× estimate | ≤ 2, ≤ $11 | re-price by amendment |
| E1-acc plus sycophancy E1 | cheap, decisive | sealed N at 2 pp | ≈ 1, $5 | none qualify → H-E1 stays sycophancy |
| E2-acc (WM vs StatNN vs MF best-of-N) | where the WM adds value | equal ledger; power ≥ 0.8 | ≤ 22, $119 | unpowered → descriptive |
| T-acc on held-out H1 paths | physically valid, cheap | power sim; quadratic-null size | 4.8, $26 | H-T stays unresolved |
| Sycophancy E2 only if it fits | largest item | confirmatory ≤ 90 | ≈ 48, $259 | drop first |
| ≤ 4 × 1× SXM5 | capacity, isolation | per-instance dead-man | — | wait; no PCIe |
| Apply the §4 table | no post hoc choices | pilot-3 report | 0 to +6 | as tabled |
| 27B: E1-acc + T-acc on 2× H100 | memory; cap | 9B E1-acc confirmed; ≤ 72 GB | 19–37, ≈ $155–300 | E2-acc descriptive unless re-priced |
