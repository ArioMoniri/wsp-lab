# wme-c1 9B launch preconditions (PREREG_WME §II.28.3), 2026-11-20

**LAUNCH BLOCKED.** Precondition 1 failed on the pinned HF-generate engine (and with it precondition 3). Per §II.28.3 and the run brief, the run stops before any confirmatory unit; both check instances are terminated and API-confirmed. **No confirmatory outcome exists; nothing sealed was read.**

| Precondition | Met | Evidence |
|---|---|---|
| 1 Pinned-engine inert identity | **NO** | teacher-forced: 8/8 constructions bitwise equal to base, I1 theta-hash = base; vLLM: 5/5 bitwise; **HF generate: 0/5** — base HF canary itself not reproducible (see row 3) |
| 2 Real 1-vs-2-GPU parity | yes | teacher-forced chunks k=1 vs k=2 bitwise True; GPU0 vs GPU1 every chunk True; vLLM GPU0 = GPU1 True (b34cc04e97b4d243…, equal to the registered stage-W canary b34cc04e…); HF generate GPU0 = GPU1 False |
| 3 Committed per-engine canary references | **NO** | not reached on a confirmatory instance; the HF-generate canary is not reproducible: three runs on one instance gave c9bb1a34d0a1… (GPU0), a8173098ef52… (GPU1), 68c050840bfe… (GPU0, later); teacher-forced canary stable 86a8c5594676… |
| 4 Evidence and fallback binding | not reached | not reached (no confirmatory instance, no study log) |
| 5 Unique physical-instance identities | not reached | not reached for a confirmatory instance; the check instance recorded GPU-UUID and machine-id digests |
| 6 Deployed persistence and repair | yes | CPU deployment drill: all 7 scenarios pass (wme-c1-drill-2026-11-19.json) |
| 7 Authoritative time and dead-man reconciliation | not reached | not reached (no study log to reconcile into); both check instances terminated and API-confirmed in the dead-man ledger |
| 8 ProductionGuard only | not reached | no confirmatory unit run; the c1 path uses only ProductionGuard |

## Finding

On the pinned stack (torch 2.14.0+cu130, transformers 5.22.2, vLLM 0.34.0; driver 580.105.08; 2 × H100 80GB HBM3), `genaudit.generate_greedy` at the registered batch 128 (left padding, greedy, 128 new tokens, non-thinking) on the 64 exposed canary items is **not bitwise reproducible**, not even run to run on one GPU. The teacher-forced path (bank/dev/test) and vLLM (S-TE) are bitwise reproducible across runs and GPUs, and vLLM reproduces the registered stage-W canary hash. Consequences for the frozen design: the HF canary cannot have a committed reference (every bracket would raise `CanaryMismatch`); the HF inert-identity check fails; and Part I §6.7.1(a)'s premise that greedy E1 generation is deterministic does not hold on this stack.

## Options for the chair (none taken)

1. Outcome-blind technical repair, then re-run Phase L: run HF generation under `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` (and, if needed, a deterministic attention backend), recorded as an engine-config change before any data. Untested; it may cost speed or not suffice.
2. A dated amendment moving E1 generation (and its canary) onto the vLLM engine, which is deterministic here. This is a design change to a registered engine (§II.4 delta 4, §II.13 item 6) and needs a chair ruling and Board A.
3. Keep the HF engine and accept within-instance pairing without bitwise canaries. This conflicts with §II.22.6 eligibility and is not recommended.

## Ledger (combined charge to be reconciled into the study ledger at any later launch, §II.30 / precondition 7)

| Instance | Type | Outcome | Hours | GPU-h-eq | $ (provider) | Terminated |
|---|---|---|---|---|---|---|
| d10681e8… | gpu_2x_h100_sxm5 | never left booting; provider 'unhealthy', ~60 min in 'terminating' | 1.511 | 3.023 | 14.48 | yes (API-confirmed) |
| c1fb8fba… | gpu_2x_h100_sxm5 | parity + inert identity checks | 0.394 | 0.789 | 3.78 | yes (API-confirmed) |

Total: 3.812 GPU-h-eq, $18.26 at the provider price ($20.51 under the 4.89 × 1.10 convention). Evidence: `docs/reviews/wme/wme-c1-launch-parity-2026-11-19.json` sha256 `1e7bf367845767a19ab4bf26acc59e030b86c57fbf9b4bce92977fb154826844`.
