# Base-model architectures for the multi-model zoo (P4b / H6)

Everything here was read from model configs and safetensors metadata on CPU: no weights were loaded.
Parameter names and shapes are frozen in `tests/fixtures/arch/*.json` (written by
`scripts/arch_inspect.py`: the text config, the `AutoModelForCausalLM` state-dict names built on the meta
device, and the checkpoint's safetensors names where we have the file). `tests/test_arch.py` and
`tests/test_arch_pool.py` assert every statement below that is about tensor names.
For Qwen3.5-0.8B, Qwen3.5-4B and LFM3 every name and shape was also checked against the local safetensors headers
(`shapes_verified_against_checkpoint_headers` in the fixtures); Gemma has no local weights, so its shapes come from
the config only.

| | Qwen3.5-0.8B (reference) | Qwen3.5-4B | LFM3-1.7B | Gemma-3-270M-it |
|---|---|---|---|---|
| repo | `Qwen/Qwen3.5-0.8B` | `Qwen/Qwen3.5-4B` | `LiquidAI/LFM3-1.7B` | `google/gemma-3-270m-it` |
| config revision | `2fc06364…` | `851bf6e8…` | `0dc4268b…` | official `ac82b4e8…` (see note) |
| HF class (zoo namespace) | `Qwen3_5ForCausalLM` | `Qwen3_5ForCausalLM` | `Lfm3ForCausalLM` (remote code) | `Gemma3ForCausalLM` |
| hidden size (residual width) | 1024 | 2560 | 2048 | 640 |
| layers | 24 | 32 | 24 | 18 |
| layer mix | 18 GatedDeltaNet + 6 full attn | 24 GatedDeltaNet + 8 full attn | 13 short-conv + 6 full + 5 sliding attn | 15 sliding + 3 full attn |
| attention heads / KV heads / head dim | 8 / 2 / 256 | 16 / 4 / 256 | 16 / 4 / 128 | 4 / 1 / 256 |
| MLP intermediate | 3584 | 9216 | 7680 | 2048 |
| vocab | 248 320 | 248 320 | 128 000 | 262 144 |
| tied embeddings | yes | yes | yes | yes |
| text parameters (no vision/MTP, `lm_head` tied) | 752.4 M | 4 205.8 M | 1 728.8 M | 268.1 M |
| norm style | pre-norm | pre-norm | **sandwich** (`use_sandwich_norm`) | **sandwich**, gain `1 + w` |

Note on Gemma: `google/gemma-3-270m-it` is gated with manual approval. The account behind `HF_TOKEN` gets
HTTP 403 on file contents (4 Nov 2026), so no official file was downloaded. Architecture facts come from the
config of the ungated mirror `unsloth/gemma-3-270m-it` (revision `23cf460f…`), whose safetensors has the
same byte size (536 223 056) and parameter count (268 098 176, bf16) as the official repo according to Hub
metadata. The zoo config declares `hidden_size`/`num_layers` so `--dry-run` works without file access
(recorded as `architecture.verified: false` in the manifest header); the official revision is taken from Hub
commit metadata, which is readable. Once access is granted, re-run `scripts/arch_inspect.py --only gemma-3-270m`
and `pytest tests/test_arch.py tests/test_arch_pool.py`, and a real build will verify the declared values.

## Tensor types per model

Module paths are relative to `model.layers.<i>.` (zoo namespace; the checkpoint names of Qwen3.5 carry
`model.language_model.` in front, which `wsp.zoo.arch` and `wsp.zoo.apply` both handle). Shapes are
`[out, in]`. "Pool" is the per-type pool of `wsp.features.pool` (feature_version 4).

| Role | Pool | Qwen3.5-4B (0.8B same names) | LFM3-1.7B | Gemma-3-270M |
|---|---|---|---|---|
| attention q | `q` | `self_attn.q_proj` [8192, 2560] ×8 (width 2× heads·dim: the output gate is fused into q) | `self_attn.q_proj` [2048, 2048] ×11 | `self_attn.q_proj` [1024, 640] ×18 |
| attention k / v | `k` / `v` | `self_attn.k_proj`, `v_proj` [1024, 2560] ×8 | `[512, 2048]` ×11 | `[256, 640]` ×18 |
| attention o | `o` | `self_attn.o_proj` [2560, 4096] ×8 | `self_attn.o_proj` [2048, 2048] ×11 | `self_attn.o_proj` [640, 1024] ×18 |
| q/k norm | `norm` | `self_attn.q_norm`, `k_norm` [256] ×8 | `[128]` ×11 | `[256]` ×18 |
| attention head gate | `attn_gate` | (fused into `q_proj`) | `self_attn.gate_proj` [16, 2048] ×11 (sigmoid per head; **not** the MLP gate) | none |
| MLP gate / up / down | `gate` / `up` / `down` | `mlp.gate_proj`, `up_proj` [9216, 2560], `mlp.down_proj` [2560, 9216] ×32 | `feed_forward.w1` (gate), `w3` (up) [7680, 2048], `w2` (down) [2048, 7680] ×24 | `mlp.gate_proj`, `up_proj` [2048, 640], `mlp.down_proj` [640, 2048] ×18 |
| linear mixer input | `lin_qkv` | `linear_attn.in_proj_qkv` [8192, 2560] ×24 | `conv.in_proj` [6144, 2048] ×13 (B, C, x gates) | none |
| linear mixer gate / beta / decay | `lin_z` / `lin_b` / `lin_a` | `linear_attn.in_proj_z` [4096, 2560], `in_proj_b`, `in_proj_a` [32, 2560] ×24 | none | none |
| linear mixer state | `lin_state` | `linear_attn.A_log`, `dt_bias` [32], `conv1d` [8192, 1, 4], `norm` [128] ×24 | `conv.conv` [2048, 1, 7] ×13 (depthwise) | none |
| linear mixer output | `o` | `linear_attn.out_proj` [2560, 4096] ×24 | `conv.out_proj` [2048, 2048] ×13 | none |
| pre-operator / pre-MLP norm | `norm` | `input_layernorm`, `post_attention_layernorm` (this is the *pre*-MLP norm in Qwen) ×32 | `operator_norm`, `ffn_norm` ×24 | `input_layernorm`, `pre_feedforward_layernorm` ×18 |
| post-writer norm (sandwich) | `norm` | none | `post_operator_norm`, `post_ffn_norm` ×24 | `post_attention_layernorm`, `post_feedforward_layernorm` ×18 |
| final norm | `norm` | `model.norm` | `model.embedding_norm` (applied after the last layer) | `model.norm` |
| embeddings | `embed` | `model.embed_tokens` [248320, 2560], `lm_head` (tied) | `[128000, 2048]` | `[262144, 640]` |
| vision tower, MTP head (checkpoint only) | `aux` | `model.visual.*`, `mtp.*` (312 of 738 checkpoint tensors; not in the CausalLM state dict) | none | none |

The Qwen3.5-0.8B column of counts is the same with 24 layers (18 linear / 6 full), hidden 1024, MLP 3584.

## Which tensors write into the residual stream

A tensor is a *writer* if its output (its rows) is added to the residual stream. Candidates for direction
ablation (`W' = W − w·d dᵀ W`, rows live in the `hidden_size` space):

| Component (`wsp.zoo.arch`) | Qwen3.5-4B | LFM3-1.7B | Gemma-3-270M | Layers covered |
|---|---|---|---|---|
| `attn_out` | `self_attn.o_proj` (8 layers) or `linear_attn.out_proj` (24 layers) | `self_attn.o_proj` (11) or `conv.out_proj` (13) | `self_attn.o_proj` (18) | every layer has exactly one |
| `mlp_out` | `mlp.down_proj` | `feed_forward.w2` | `mlp.down_proj` | every layer has exactly one |
| not edited: `embed_tokens` | writes row 0 of the direction stack (the embedding residual) | same | same | tied with `lm_head`, so an edit would also change the output side |

Everything else either reads the residual (q/k/v, gate/up, `in_proj_*`, `conv.in_proj`, norms) or lives
inside the mixer (state, conv, head gate). The test `test_every_layer_has_exactly_one_writer_per_component_writing_the_residual`
checks one writer per component per layer and that its row count equals `hidden_size` for all four models.
Note that LFM3's `q_proj` is also `[2048, 2048]`, so a shape rule alone cannot find writers: the registry is
by name.

**Sandwich norm.** In LFM3 and Gemma-3 the writer output passes through an RMSNorm with a per-channel gain
`g` before the residual add (`post_operator_norm`/`post_ffn_norm`; `post_attention_layernorm`/
`post_feedforward_layernorm`; effective gain `g = w` for LFM3, `1 + w` for Gemma). Projecting `d` out of
`W`'s output then leaves a component along `d` after the norm, because `diag(g)` rotates the subspace.
`wsp.edits.kernels.fold_direction` projects `normalize(g ⊙ d)` instead, which makes the vector written to
the residual exactly orthogonal to `d` (`test_fold_post_norm_makes_the_written_vector_orthogonal_after_the_post_norm`).
The LFM3 and Gemma configs set `families.abliteration.fold_post_norm: true`; edits record
`kernel_variant = heretic2-classic-projected-rownorm-none+postnorm-fold`, which enters the reuse key, so
folded and plain edits never alias. The flag is an error on a model without post-writer norms. Public Heretic
variants of these models (and Heretic's own `heretic_ara` modifier, which is optimisation-based) use the plain
projection; set the flag to `false` to reproduce them.

## How the zoo builder is model-agnostic now

* `wsp.zoo.arch` holds the registry (`qwen3_5`, `lfm3`, `gemma3`, generic `dense`) detected from the text
  config's `model_type`. Each zoo config also carries an `architecture:` block (family, `hidden_size`,
  `num_layers`, writer regexes, post-norms, gain), checked against the model's own config when readable.
* `tensors:` entries of `random_dir` are writer components (`attn_out`, `mlp_out`), the legacy 0.8B names
  (`mlp.down_proj`, `self_attn.o_proj`, kept as *role aliases*, so the same family descriptor means the same
  role on every model and 0.8B edit ids are unchanged) or regexes fully matched against the in-layer module
  path. A spec that matches nothing in a chosen layer raises; a matched tensor whose rows are not the residual
  width raises.
* Residual width and depth come from the config (`hidden_size`, `num_hidden_layers`). Layer indices, the
  abliteration `direction_index` / `max_weight_position` ranges and the shape of the refusal-direction stack
  `(num_layers + 1, hidden_size)` are validated before anything is planned or applied.
* Refusal-direction extraction itself is generic (last prompt token of `hidden_states`, embedding row + every
  layer). Two model-specific inputs: LFM3 needs `trust_remote_code` (allow-listed in `wsp.zoo.arch`, passed to
  every `from_pretrained` we own), and the response prefix: Qwen3.5-0.8B's template emits a closed
  `<think>\n\n</think>\n\n` block (prefix `""`), the Qwen3.5-4B and LFM3 templates open `<think>` (Heretic's
  detection gives prefix `</think>`), Gemma's template has no CoT block so its config pins
  `refusal_response_prefix: ""`.
* LoRA targets: shorthand `q k v o gate up down` expand per architecture (`Arch.lora_targets`). Qwen3.5 and
  Gemma keep the legacy `*_proj` names (existing 0.8B adapters stay reproducible; in Qwen3.5 this means
  `o` covers only the full-attention layers, as before). LFM3 maps `gate/up/down` to `w1/w3/w2`, because the
  bare name `gate_proj` would hit the attention head gate. `heretic_ara` needs no mapping: installed Heretic
  discovers `self_attn.o_proj`, `linear_attn.out_proj`, `conv.out_proj`, `mlp.down_proj` and
  `feed_forward.w2` itself.
* Dry runs for a gated repo take the base revision from Hub commit metadata (`base_revision_source` in the
  header); a real build still requires file access.

## Pooling audit (feature_version 4)

Every tensor of each model, in both namespaces, lands in a typed pool; none falls into `other` and none is
global-only (`tests/test_arch_pool.py`, which enumerates names from the fixtures). Counts (zoo namespace):

| Pool | Qwen3.5-0.8B | Qwen3.5-4B | LFM3-1.7B | Gemma-3-270M |
|---|---|---|---|---|
| q / k / v | 6 / 6 / 6 | 8 / 8 / 8 | 11 / 11 / 11 | 18 / 18 / 18 |
| o | 24 | 32 | 24 | 18 |
| gate / up / down | 24 / 24 / 24 | 32 / 32 / 32 | 24 / 24 / 24 | 18 / 18 / 18 |
| lin_qkv / lin_z / lin_b / lin_a | 18 each | 24 each | 13 / 0 / 0 / 0 | 0 |
| lin_state | 72 | 96 | 13 | 0 |
| attn_gate | 0 | 0 | 11 | 0 |
| norm | 61 | 81 | 119 | 109 |
| embed | 2 | 2 | 2 | 2 |
| aux (checkpoint names only) | 168 | 312 | 0 | n/a |
| other | 0 | 0 | 0 | 0 |
| total tensors | 321 | 427 | 287 | 237 |

Gemma's checkpoint listing is not available (gated), so its `aux`/checkpoint row is not audited; its zoo
namespace comes from the model class built from the mirror config. Qwen3.5 checkpoint names (738 for 4B,
488 for 0.8B) also pass: 312 / 168 of them are vision-tower and MTP tensors, pooled as `aux`.

What changed in `wsp.features.pool`, and why `feature_version` is now **4**:

| Name | Before (v3) | Now (v4) |
|---|---|---|
| LFM3 `feed_forward.w1/w3/w2` | `other` | `gate` / `up` / `down` |
| LFM3 `conv.in_proj` | `other` | `lin_qkv` (fused input projection of a linear mixer) |
| LFM3 `conv.out_proj` | `other` | `o` (as `linear_attn.out_proj`) |
| LFM3 `conv.conv` | `other` | `lin_state` (as `conv1d`) |
| LFM3 `self_attn.gate_proj` | **`gate`** (wrong: attention head gate pooled with the MLP gate) | new type `attn_gate` |
| `model.visual.*`, `mtp.*` | `other`, with `mtp.layers.0` placed in layer bucket 0 | new type `aux`, never bucketed |
| Gemma-3 `pre/post_feedforward_layernorm`, LFM3 `operator/ffn/post_*_norm`, `embedding_norm` | `norm` | `norm` (unchanged) |

The two new types are inserted before `other` (`other` stays the last, catch-all pool), so the pooled
layout changes: stats `18·(1 + 17·buckets)` = 1242 (was 1098), spectrum `8·17` = 136 (was 120). A v3 vector
is **not** a prefix of a v4 vector (for entries without LFM3/aux tensors the new blocks are zero and `other`
moves), so features extracted with v3 are rejected by the resume identity and must be re-extracted for any
cross-model use. The analogue mapping (LFM3 conv ≈ linear mixer) is deliberate for H6: the predictor's
inputs are per-type statistics, so tensors with the same role in different architectures must share a pool.
Whether the analogy hurts transfer is an empirical question for the P4b leave-one-model-out runs.

## Zoo configs and dry-run results

`configs/zoo_qwen3.5-4b.yaml`, `configs/zoo_lfm3-1.7b.yaml`, `configs/zoo_gemma-3-270m.yaml` have the same
families, counts (`random_dir` 150, `abliteration` 200, `heretic_ara` 12 at 10 trials per D6, `lora_sft` 120,
`task_vector` 80, `chain` 150), sampling ranges, seed and eligibility thresholds as
`configs/zoo_qwen3.5-0.8b.yaml`. Only depth-dependent values are rescaled to the model's layer count
(`random_dir.layers` at the same relative depths; `direction_index` max = layers − 1,
`max_weight_position` max = layers, `min_weight_distance` max = layers // 2).
`tests/test_zoo_multimodel.py` enforces the equality. The 0.8B config is untouched, and dry-run entry ids
of the 0.8B config are identical before and after this change (compared for all families at `--n 5`).

`python -m wsp.zoo.build --config <cfg> --dry-run --n 3` (4 Nov 2026, real Hub metadata, exit 0 for all):

| Config | base revision | architecture check | random_dir | abliteration | heretic_ara | lora_sft | task_vector | chain (steps) | rows |
|---|---|---|---|---|---|---|---|---|---|
| `zoo_lfm3-1.7b.yaml` | `0dc4268b…` (snapshot) | verified (lfm3, 2048 × 24) | 3 | 3 | 3 | 3 | 3 | 16 | 31 |
| `zoo_qwen3.5-4b.yaml` | `851bf6e8…` (snapshot) | verified (qwen3_5, 2560 × 32) | 3 | 3 | 3 | 3 | 3 | 16 | 31 |
| `zoo_gemma-3-270m.yaml` | `ac82b4e8…` (Hub metadata, gated) | declared only (gemma3, 640 × 18) | 3 | 3 | 3 | 3 | 3 | 16 | 31 |

The `chain` column counts manifest rows (cumulative snapshots; 3 chains of 2 to 8 steps), not chains. Dry runs
write `manifest.dryrun.jsonl` under each `out_dir` and never touch `manifest.jsonl`.

## Needs the GPU (not done here)

* Real builds for all three zoos, and a one-entry smoke per family per model (`random_dir`, `abliteration`,
  `lora_sft`, `heretic_ara` with `--override heretic_ara.trials=2`) to confirm key names against the loaded
  models: the `AutoModelForCausalLM` state-dict names used here come from the meta device, and LFM3 needs the
  remote modelling code.
* Refusal-direction extraction per model (400 + 400 prompts); inspect `example_formatted_prompt` in each
  `refusal_dirs.*.manifest.json`, in particular the response prefix for Qwen3.5-4B / LFM3 (`</think>`) versus the
  0.8B's closed block, and Gemma's empty prefix.
* Whether `fold_post_norm` changes refusal/KL outcomes enough to matter (compare folded and plain
  abliteration on one LFM3 and one Gemma entry).
* `heretic_ara` cost: D6's 10 trials were profiled on 0.8B; the 4B and LFM3 trials will be slower.
* Gemma-3-270M: download and `scripts/arch_inspect.py --only gemma-3-270m` once the HF account is approved.
