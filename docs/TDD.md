# Test-Driven Development policy

1. **Red first.** Every module in `src/wsp/` gets a failing test in `tests/` before implementation.
   Name tests after the behaviour: `test_svd_tokens_are_invariant_to_lora_refactorisation`.
2. **Three tiers.**
   - `pytest -q` — CPU, < 60 s total, runs on every commit (pre-commit hook `scripts/precommit.sh`).
   - `pytest -q -m gpu` — needs CUDA, < 10 min, run before merging a worktree.
   - `pytest -q -m "slow or net"` — downloads models / runs evals; run inside `agent-tmux`.
3. **Fixtures.** `tests/conftest.py` builds a 2-layer toy transformer (≈ 50k params) and a toy zoo of 8 edits
   so feature/predictor tests never need a real model.
4. **Determinism.** Every zoo entry and eval has a seed; tests assert that re-running gives identical hashes.
5. **Invariants worth testing** (from the literature):
   - LoRA (B,A) and (B G, G⁻¹ A) must give identical φ_svd tokens (W2T symmetry).
   - Ablating a direction d and then evaluating `delta @ d` must be ≈ 0 on edited rows (abliteration contract).
   - SSM predictor with sequence length 1 must equal the stateless predictor given identical init.
   - Predictor outputs are bounded to [0,1] for accuracy-type targets.
6. **Eval contracts.** `wsp.eval` returns per-item records; aggregate numbers are computed by tests from items,
   never trusted from the runner.
7. **No silent fallbacks.** If `mamba-ssm` fails to import, the SSM predictor must log which backend is used and
   tests must cover both backends.
