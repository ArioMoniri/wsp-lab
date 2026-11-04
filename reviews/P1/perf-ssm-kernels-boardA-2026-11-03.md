# Board A review: `perf/ssm-kernels` @ 2808e52

- Reviewer: reviewer-ml (Board A), 2026-11-03
- Scope: `git diff main...perf/ssm-kernels` (merge-base 1113a2f). 5 files, +124/−8: `src/wsp/predictor/scan.py` (new), `src/wsp/predictor/ssm_torch.py`, `tests/test_predictor.py`, `scripts/bench_ssm.py` (new), `docs/INFRA.md`.
- Verdict: **REVISE**
- Read-only review. No repo code was edited. Scratch scripts are in `/tmp/boardA-scratch/`, logs in `~/agent-logs/boardA-*-perf-ssm-kernels.log`.

## Blocking

1. **SelectiveSSMBlock always uses the chunked scan; there is no backend dispatch. The default CPU path regresses.** `src/wsp/predictor/ssm_torch.py:57` calls `selective_scan` unconditionally. The "loop on CPU, chunked on GPU" choice the branch describes is not in the code. Measured on CPU at B=32, T=64, d=512, N=16:
   - Author's log: loop 140 ms, chunked 2473 ms.
   - My mean-of-5 forward: loop 113 ms, chunked 609 ms.
   - Chunked is also not faster on GPU in my runs (median of 20, see item 2): forward 5.9 ms loop vs 20.7 ms chunked; forward+backward 24.3 ms vs 69.3 ms; peak training memory 220 MB vs 1176 MB.
   - Failure scenario: after merge, every predictor forward/training on CPU (the path in use, since `mamba_ssm` cannot build per `docs/INFRA.md`) runs several times slower, and GPU training runs slower and uses about 5× the memory.
   - Fix: keep the loop as the default. Use the chunked scan only where an idle-machine sweep shows a win, with an explicit flag and a backend log line (TDD §7 "no silent fallbacks"). Otherwise drop the chunked path from `SelectiveSSMBlock`.

2. **The benchmark cannot support a backend decision.** `scripts/bench_ssm.py:24-42` does one timed call after one warmup, with threads unpinned and no repeats. The machine was shared during my runs (load average about 20 on 8 cores; GPU at 99% utilisation and 24 GB used by other jobs). Repeat runs contradicted each other:
   - Committed script, run 1: CPU loop 805 ms, chunked 555 ms (chunked faster).
   - Profile script, mean of 5: loop 113 ms, chunked 609 ms.
   - The author's GPU result (18 vs 12 ms) was not reproduced; in my median-of-20 run chunked was slower (item 1).
   - Fix: median and IQR over at least 20 repeats, for forward and forward+backward, with peak memory. Record `torch.get_num_threads()`, `nvidia-smi` utilisation, and load. Run on an idle host, or re-run `bench_ssm.py` with `OMP_NUM_THREADS` set and the result recorded in `docs/INFRA.md`.

## Non-blocking

3. **Why the CPU path is slow (answer to the question asked).** `src/wsp/predictor/scan.py:19-28` runs a Hillis–Steele doubling scan with log2(C)=6 levels per chunk. Each level does two `torch.cat`, an `exp`, a `mul` and an `add` on full `(B, C, d, N)` tensors (16.7M fp32 elements, 67 MB each). That is O(C log C) work with about 30 large allocations per chunk. The profile (`profile_scan.py`):
   - Chunked self-CPU 440 ms: `aten::cat` 36% (13 calls × 12 ms), `mul` 24%, `add` 23%, `exp` 13%.
   - Loop self-CPU 308 ms: 384 small `mul` calls on 262k-element tensors, which stay in cache.
   - The Python loop is not the bottleneck: it is 6 iterations per chunk. The cost is memory bandwidth and allocation, not interpreter overhead. The `log/exp` overhead is 13% of the total.
   - The author's 17× figure was not reproduced. The mechanism supports roughly 3–6× on a memory-bound CPU, and my number is from a loaded machine.
   - A chunk-size sweep (8, 16, 32, 64) is a cheap experiment: smaller chunks reduce levels and per-level tensor size, at the cost of more Python iterations.

4. **torch.compile is not applied.** `scan.py:34-36` defines `compiled_selective_scan`, which nothing calls (`grep` finds no call site in src, tests or scripts). My measurement, same run: compiled chunked steady state 183 ms vs eager chunked 609 ms, but first call 45.8 s. Compiled chunked is still about 1.6× slower than the loop on CPU, so compile does not rescue the chunked path. If it is wired in, use `dynamic=True` or bucketed T, since trajectory lengths vary and each new length would otherwise recompile. Otherwise delete the dead function.

5. **Test gaps in `tests/test_predictor.py:46-66`.** The new test is forward-only, float32, one shape (B=2, T=130, D=8, N=4). Also missing:
   - Gradient equivalence. Training goes through `backward`. In my float64 check (`/tmp/boardA-scratch/check_scan.py`), gradient error versus the loop was at most 1.4e-13 for T=64, 130 and 300. Cheap to add.
   - Block-level equivalence against the old per-step loop. My check gave 4.4e-16 in float64.
   - Causality across chunk boundaries. The repo's causality test (`test_outputs_bounded_and_causal`) uses T=5, a single chunk. My check showed past outputs bit-identical when the future is perturbed at cut points 1, 63, 64, 65, 130 and 199.
   - Lengths T ≤ chunk and T = chunk, which I checked (T=1, 2, 63, 64, 65) but the repo does not.
   These are not blocking because my checks pass; they are the minimum for a merge that changes the training path.

6. **INFRA note documents the failure but gives no fix.** `docs/INFRA.md:26-41` records the C++17/C++20 mismatch. I traced the cause:
   - `mamba-ssm` `setup.py:213,216,226,230` and `causal-conv1d` `setup.py:209,212` hard-code `-std=c++17` for both the C++ and nvcc flags (sdists under `~/.cache/uv/sdists-v10/pypi/`).
   - torch 2.15 `utils/cpp_extension.py:2136,3082` adds `-std=c++20`, but `extra_compile_args` come after it, so the `-std=c++17` wins.
   - Environment variables such as `CXXFLAGS` do not override it, because the flags are placed after them.
   - Actionable fix: patch both `setup.py` files, replacing `-std=c++17` with `-std=c++20` in the cxx and nvcc lists, then build from the patched source (`uv pip install <patched-dir>`). Pin the patch in the repo (a patch file or fork), since the uv cache is local.
   - Not verified here: I did not rerun the build, which is long and loads the host.
   - This matters for the GPU argument: if `mamba_ssm` imports, `TrajectoryPredictor` uses `Mamba2` and never calls `scan.py`. The chunked scan's GPU justification therefore depends on whether this build is fixed.

7. **Low-precision risk (latent).** No autocast is used today, so the scan runs in fp32 and is fine. If bf16 autocast is enabled (the perf-challenger remit), the cumulative log-decay sums in `scan.py:12-26` must stay in fp32. Add a dtype assertion or a comment.

8. **Hygiene.** The worktree `perf/ssm-kernels` has uncommitted edits to `board/P1.md` and `sessions/2026-11-03-001.md`, and an untracked `.venv/` that is not in `.gitignore`. Do not commit these with the branch. An earlier test log (`wsp-ssm-tests`) failed with `ModuleNotFoundError` because `scripts/env.sh` puts the canonical `src` first on `PYTHONPATH`; run with `PYTHONPATH=$PWD/src`.

9. **Board A checks not affected.** The diff touches no data splits, seeds, labels, metrics, bootstrap CIs, SVD dtype or manifests. Numerical equivalence is at float64 round-off (≤1e-14) and float32 drift of 8.8e-6 at T=2048 against a float64 reference (max |ref| 57). Existing predictor results should therefore not change beyond float noise. If the chunked path is ever used for a reported run, the run manifest should record the scan implementation.

## Checks run

- `pytest` on CPU, with `PYTHONPATH=$PWD/src`, `CUDA_VISIBLE_DEVICES=` and `OMP_NUM_THREADS=4`: **16 passed, 1 skipped** (the CUDA test), 2.0 s. Without the thread cap the same suite ran past 170 s and was killed, because torch threads oversubscribed the shared host.
- The GPU test was not run on CUDA. The predictor smoke test on GPU is not part of this review.
- Scratch equivalence checks (forward, gradient, causality, float32 drift, block-level): all pass, numbers in item 5.
- Benchmarks: committed `scripts/bench_ssm.py` run twice, plus profile and GPU scripts; numbers and their noise in items 1–4.

## Summary for the author

The chunked scan is numerically correct and causal. The performance case for it is not established, and the branch makes the default CPU path slower with no dispatch. Fix items 1 and 2 before merge. Items 3–8 can follow.
