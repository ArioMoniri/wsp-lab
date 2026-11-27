# wme-c1 run17: may a passing parity on the identical code commit be reused? Question and ruling (27 Nov 2026)

**Context.** The run17 parity poller had waited about 8 h for `gpu_2x_h100_sxm5` capacity (from 07:17:30Z). Run16's parity (l16) passed on `41a9a9f`. Nothing under `src/` or `scripts/` has changed since; `7e2ae32` adds only docs, and the operator files live outside the repository.

**Question (chair).** Does the registered rule require a fresh 2-GPU parity for every launch, or does a passing parity on the identical commit cover a launch from a fresh checkout of that commit at the run10 path?

**Findings (registered text and code only).**
- §II.22.6: inert identity, 1-vs-2-GPU parity and canaries "are launch preconditions. They are re-run on the confirmatory stack at launch." §II.22 (v2d): "After the bank, the run checks the validity preconditions of §II.22.6 instead: inert identity, 1-vs-2-GPU parity and canaries." Closure table, row 6: "the launch checks re-run at launch".
- §II.31.9 N4 (b): "`phasel-main` accepts parity evidence only if it was produced by its own commit (P2)." P2: "`phasel-main` runs from the same commit."
- Code: `phasel-main --resume` requires parity evidence bound to the engine config, with `controller_commit == HEAD` (exact), a pass including `hf_gen_eq_committed`, and the main's canaries equal to the parity GPU-0 hashes (N3). It has no freshness or per-instance check. `up` does not check parity.
- Reuse would need the l16 parity file, which is not in `41a9a9f`'s tree, supplied as an out-of-tree, untracked file.
- Every GO so far has run parity at each launch.

**Ruling (chair).** No parity reuse. §II.22.6 says the launch preconditions are "re-run on the confirmatory stack at launch", every GO so far has done so, and the reuse path needs an out-of-tree evidence file. On a launch-validity precondition, an ambiguity is resolved the strict way. Waiting costs only time.

Polling continues for `gpu_2x_h100_sxm5` as registered (H100 SXM5, no PCIe fallback, §II.13 item 1; no other 2-GPU configuration is registered). The operator reports when parity starts. If no capacity has appeared after 48 h of polling in total, the operator reports once, with the polling log. No other changes.

**Poller coverage.**
- Instance type: only `gpu_2x_h100_sxm5` (`TYPES["parity"]`).
- Regions: no filter. It launches in any region the provider reports with capacity, preferring us-southeast-1.
