# Board A re-review: perf/ssm-kernels @ 9a57a4f

Reviewer: reviewer-ml, 2026-11-04. Verdict: **REVISE**. Blocking: **yes**, one item.

Read reviewer-ml.md, REVIEW_PROTOCOL.md and the prior REVISE review. Reviewed `git diff main...perf/ssm-kernels -- src tests scripts docs`, tracked data and Ruff configuration. No code edited.

## Blocking finding

1. **Ruff includes downloaded third-party sources.** `pyproject.toml:41-42` only sets line length, with no data exclusion or separate Ruff config. `ruff check --show-files` actually enumerates data/build/causal_conv1d-1.7.0 source/build copies and mamba sources. After running the recipe, routine repository lint scans upstream files and produces unrelated failures. Add an explicit data/ exclusion (prefer extend-exclude) and verify enumeration excludes these files. Blocking: **yes**, outstanding requested hygiene requirement.

## Prior blockers closed

- `src/wsp/predictor/ssm_torch.py:27-28,90-107`: mamba_ssm is the default when importable plus CUDA, loop otherwise. Chunked is explicit opt-in. Backend is logged and invalid values raise. The CPU regression from unconditional chunked scanning is removed.
- `scripts/bench_ssm.py:61-83`, `docs/BENCH_SSM.md:3-42`: three warmups, 20 repeats, synchronization, medians/IQR, CUDA peak allocated memory, threads, host load and GPU utilization are recorded. CPU peak memory is honestly N/A. Shared GPU contention is disclosed. These microbenchmarks support leaving chunked opt-in; they do not establish uncontended or end-to-end Mamba2 predictor speedups. Backward differentiates u/dt only, not all model parameters.

## Wrapper correctness

`ssm_torch.py:68-76`: headdim divides d_model*expand and is at most 64; Mamba2 receives d_model, d_state, expand and headdim. Both block implementations have pre-LayerNorm and exactly one residual addition. `tests/test_mamba_gpu.py:9-23` really perturbs future inputs and checks earlier predictions, repeated forward and bounded outputs. Additional live CUDA check reconstructing two models with seed 42 verified exact seeded equality, unchanged causal prefix, changed future predictions, finite values and [0,1] bounds. Passed. Chunked forward/input-gradient equivalence and causality at the length-65 boundary are now tested. No data splits, labels, metrics, bootstrap CIs, SVD or manifests changed.

## Non-blocking findings

2. **Clean-venv prerequisites are implicit.** `docs/INFRA.md:28-32`, `scripts/build_mamba.sh:4-19`: list uv, canonical venv setup/activation, compatible PyTorch, setuptools, packaging, wheel, ninja, runtime dependencies including einops, C++ compiler, and CUDA toolkit/nvcc (CUDA_HOME/PATH as needed). No-build-isolation/no-deps requires these beforehand. TORCH_CUDA_ARCH_LIST=12.0 and MAX_JOBS=4 are already exported; architecture is host-specific. Both version-pinned sdists get the C++20 patch. Recipe is workable in principle with prerequisites; no expensive rebuild performed. Optionally set CAUSAL_CONV1D_FORCE_BUILD=TRUE and MAMBA_FORCE_BUILD=TRUE to avoid upstream cached-wheel selection. Blocking: **no**.

3. **Build artifacts can be staged accidentally.** `scripts/build_mamba.sh:6` creates data/build, which is not ignored. Both `git ls-files data` and `git ls-tree -r perf/ssm-kernels data` are empty: no committed sdists. Worktree has untracked data/ and an existing modified session file; data is a real directory rather than the AGENTS-required canonical symlink. Ignore build artifacts and restore canonical worktree layout in follow-up; do not stage them. Blocking: **no**, no committed artifacts.

4. `ssm_torch.py:18-24` catches all import exceptions without logging the cause. Log extension import failures to diagnose fallback; backend selection itself is logged. Blocking: **no**, inherited behavior.

No credential-like secrets or IP literals found in reviewed source/tests/scripts/docs diff.

## Validation

Target worktree, source scripts/env.sh, OMP_NUM_THREADS=4:

- CPU, CUDA_VISIBLE_DEVICES= pytest -q: **79 passed, 5 deselected**, 13.05 s.
- Shared GPU, pytest -q -m gpu -k "mamba or predictor": **2 passed, 82 deselected**, 22.05 s.
- Additional seeded CUDA/causality/sensitivity/finite-bounds check: **passed**.
- Log: ~/agent-logs/boardA-ssm-rereview-tests-20261104-011610.log.

**REVISE**: explicitly exclude data/ from Ruff. The two original performance blockers are closed.
