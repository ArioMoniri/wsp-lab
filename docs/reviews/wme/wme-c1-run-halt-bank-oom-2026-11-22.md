# wme-c1 run7: halt in the bank stage (closing-canary CUDA OOM in the HF child), 22 Nov 2026

This was an engineering halt. No sealed file was read and no claim outcome exists. One incident is open: `worker_error` (seq 569).

## Sequence (clean checkout of `53e97bc`; bundle sha256 `a44e909f…`; the worktree's `.venv` link was in place)

| step | result |
|---|---|
| Dead-man pre-checks l7, l7m | pass (4 clean ticks each, no error event) |
| Preflight | pass: venv interpreter, a dry heartbeat run, dead-man ticks (also run by `up` before each poll) |
| Parity, 2×H100 (driver 580.105.08) | PASS: every pass-rule check true, including `hf_gen_eq_committed`; hf_gen `7299ac4f…` on GPU0, GPU1 and 3/3 repeats; code identity pass; gate 265 passed; projection 98.0; terminated and confirmed |
| Main, 1×H100 (driver 580.105.08) | launched 04:00Z; progress heartbeats landing (2-heartbeat gate met) |
| Resume (generation 1) | PASS: canaries equal the committed references; start canary passed |
| `resolve-incident` | seq 338 resolved (record 369), seq 340 resolved (record 371), both citing addendum 5 |
| Bank (generation 1) | the field was collected deterministically (g0–g7, vocab); `bank:base` and two chunks of bank sides ran. One closing canary passed (seq 456). The next closing canary failed: **`worker_error` incident seq 569** |
| Instance | terminated by the operator and confirmed through the API; reconciled at 1.399 GPU-h-eq |

## Cause: `Runtime.release()` frees the CUDA cache before it drops the bank tensors (code defect)

The failure is a CUDA out-of-memory error inside the HF child (K1) during the teacher-forced / hf_gen canary, in SDPA attention:
- the child tried to allocate 2.36 GiB with 1.55 GiB free of 79.18 GiB;
- the child held 53.9 GiB;
- the parent worker (pid 3757) still held 23.7 GiB.

After the halt, with the worker idle, `nvidia-smi` showed the parent process still at 24.2 GiB.

In `serve.Runtime.release()`, the order of operations is:
1. `self.m.release()`, which calls `gc.collect()` and `torch.cuda.empty_cache()`;
2. only then are `_rc` and `_bank` set to None and the field's kept and gradient caches cleared.

The memory of the R cache, the bank world and the field tensors therefore stays reserved by the parent's caching allocator. That reservation grows with bank work: earlier canaries in the same run passed, and the parent's memory had risen to about 63 GiB during units. Eventually the child no longer fits.

The fix is a code change: release every GPU reference first, then run `gc.collect()` and `torch.cuda.empty_cache()`. The parent's free memory should also be checked and recorded before the child is spawned. Under P2 this needs a new commit and a new instance.

## State

- Contract generation 1 (`8f38d82c…`) is current.
- Generation-1 results:
  - **10 eligible**: 8 field groups, vocab and `bank:base`. Their per-file hashes are recorded, so the field can be re-materialised on a new instance by deterministic re-collection plus bitwise K3, which C6 supports.
  - **16 retained-ineligible**: bank sides whose closing canary failed. They were disposed at the authoritative termination. Each used 1 of its 3 generation-1 attempts.
- One incident is open (`worker_error` seq 569) and needs a resolution citing the chair's ruling.
- Ledger: study = dead-man = 13.447 GPU-h-eq ($64.95); cap 110; no R8 raise.
