# Board A review: feat/multi-model-zoo

Head reviewed: f1325ba73ee0e19c1fceb02539e98b52bc7c80a7. Verdict: **REVISE**. Blocking: **yes**.

## Finding

- **B1 — remote code trust is not limited to LFM3** ([src/wsp/zoo/arch.py:186](../../src/wsp/zoo/arch.py)). `remote_code_kwargs` accepts `trust_remote_code: true` for any base model, bypassing the LFM3 allow-list. A config for another repository can therefore execute its remote modeling code. This violates the requested LFM3-only scope. Remove or constrain the override and test rejection for another model. **Blocking: yes.**

## Evidence and assessment

- CPU suite: `source scripts/env.sh && pytest -q`: 394 passed, 8 deselected, 6 warnings, 30.96s. All three requested `--dry-run --n 3` commands exited 0. No GPU jobs were run.
- Independently read local safetensors headers: the fixture language-model names and shapes match all 488 Qwen 0.8B, 738 Qwen 4B, and 286 LFM3 checkpoint headers (excluding tied `lm_head`); zero fixture shape mismatches. Writer selection matches the specified attention/linear/conv and MLP roles, one per layer with residual-width rows. Width and depth derive from text config; refusal direction width and layer count are validated. Gemma's model-class fixture is mirror-config-derived, not verified against official checkpoint headers.
- The 0.8B config has no branch diff; independently ran `--n 5` on branch and main: all 1,326 ordered entry IDs match.
- Pool tests enumerate all four model fixtures, including Qwen checkpoint vision/MTP as `aux` and LFM3 attention `gate_proj` as `attn_gate`. Gemma checkpoint names remain unaudited because official weights are gated. Feature version 4 participates in resume identity, which rejects mismatches. **Merging invalidates v3 features; the 0.8B zoo requires the scheduled re-extraction** into a compatible v4 output.
- Post-norm folding projects along normalized `g ⊙ d`: after RMS normalization and channel gain the dot product with `d` is zero at full projection weight, up to numerical rounding. The orthogonality test passes. Plain projection is preferable as the default when Heretic parity is the experimental objective; the new sandwich-model configs intentionally opt into folding and the variant is recorded in the reuse key. This is nonblocking because 0.8B retains the plain default and IDs.
- Gemma mirror-derived architecture with `verified: false` is acceptable for planning and dry runs with the documented provenance and limitation; official checkpoint verification and real-model smoke tests remain prerequisites for interpreting production results.
- No weight or secret files appear in the branch diff. No Gemma stray download was found in `~/.cache/huggingface/hub`; the remaining Qwen cache was not removed. No claim is made about historical commits or other cache locations.
