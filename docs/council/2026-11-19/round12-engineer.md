# Round 12, pro-engineer seat: running WM-E on MedGemma (19 Nov 2026)

## BLUF

1. MedGemma is Gemma 3: a standard decoder with sliding and global attention and no linear-attention layers. The port touches about 12 places in WM-E.
2. Effort: about 6–9 engineer-days plus about 1 RTX GPU-h of smokes. The work goes on a new branch and venv, with no C1 changes.
3. Risk: every Gemma 3 writer (`o_proj`, `down_proj`) feeds an RMSNorm before the residual add. Frozen bank doses (scale, noise, LoRA λ, steer) are partly normalised away, so the bank needs a redesign.
4. MedGemma 4B (text use) fits on the shared 32 GB RTX: about 7.8 GB of text weights, against the 9B's 27.35 GB measured peak. Pilots are feasible under a ≤ 16 GB cap.
5. A C1-sized 4B study is about 29 H100 GPU-h (≈ $140) by parameter scaling, or roughly 100 RTX GPU-h. Run only pilots locally.
6. The 27B text model matches the calibrated Qwen3.8-27B footprint (54.0 vs 53.79 GB). Its bank is about 8.9 GPU-h (≈ $44); a full design is about 200 GPU-h (≈ $980), which is unfunded.
7. The repos are gated by an immediate click-through. The MCP account reads 4B-it and 27B-text-it but is denied 1.5-4B-it and 27B-it. `HF_TOKEN` exists in the env file; its grants are unverified.
8. Do next: port plus a local G0 inert and dose pilot on 4B under its own pre-registration. No Lambda use until C1 has finished.

## 1. Architecture (fetched)

**4B-it** (config read through the HF MCP from huggingface.co/google/medgemma-4b-it, `config.json`):
- `Gemma3ForConditionalGeneration`.
- Text: 34 layers, hidden 2560, FFN 10240, 8 Q and 4 KV heads, head_dim 256, GELU-tanh, vocabulary 262,208.
- `layer_types`: 5 sliding layers to 1 full layer, window 1024.
- Vision: SigLIP, 27 layers, hidden 1152, 896 px, 256 tokens per image.
- Size: 8.60 GB. Names: `language_model.model.layers.L.*`.

**27B-text-it** (`config.json` of google/medgemma-27b-text-it):
- `Gemma3ForCausalLM`: 62 layers, hidden 5376, FFN 21504, 32 Q and 16 KV heads, head_dim 128.
- `sliding_window_pattern` 6, and no `layer_types` key.
- Size: 54.02 GB. Names: `model.layers.L.*`.

**Layer structure.** Each layer has `q_norm` and `k_norm`, plus pre- and post-norms around both sublayers. In the source, `post_attention_layernorm` and `post_feedforward_layernorm` normalise each sublayer's output before the residual add. RMSNorm uses (1 + w), embeddings are scaled by √hidden, and `lm_head` is tied to `embed_tokens` (https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma3/modeling_gemma3.py).

**Template and tokens.** Turns are `<start_of_turn>user|model … <end_of_turn>`, and a system message is folded into the first user turn. There is no thinking switch. `eos_token` is `<eos>`, while the generation EOS ids are [1, 106] (`chat_template.jinja`, `special_tokens_map.json`, `generation_config.json`).

**Card text scores (4B / Gemma 3 4B):**

| Benchmark | MedGemma 4B | Gemma 3 4B |
|---|---|---|
| MedQA | 64.4 | 50.7 |
| MedMCQA | 55.7 | 45.4 |
| PubMedQA | 73.4 | 68.4 |
| MMLU-Med | 70.0 | 67.2 |
| MedXpertQA | 14.2 | 11.6 |

MedQA is listed among the training data sources (`README.md`).

**Serving.** vLLM supports both classes, with LoRA and PP (https://docs.vllm.ai/en/latest/models/supported_models.html).

**Licence.** HAI-DEF §2.2 permits modification, and §3.1 conditions distribution only. Neither section restricts publishing results (https://developers.google.com/health-ai-developer-foundations/terms). So: publish recipes and metrics, not edited weights.

**Fetch failures:** `medgemma-1.5-4b-it` and `medgemma-27b-it` returned `HF_FS_ACCESS_DENIED`.

## 2. Qwen3.5 assumptions (branch `run/wme-c1-9b`, read-only)

| # | Module | Assumption | Gemma 3 effect | Port |
|---|---|---|---|---|
| 1 | `wm/actions.py` `MIXER_OUT`, `RoleMap.from_config` | Layer types {full, linear}; `tie_word_embeddings` defaults to False | 4B: `sliding_attention` raises. 27B: "all full". The key is absent, so both models read as **untied** | Add the sliding type and derive types from the pattern; take the tie default from the model class |
| 2 | `wm/hf.py` | Qwen `TARGETS`; `AutoModelForCausalLM`; `.visual.` guard; Qwen GatedDeltaNet toy kernels | The 4B vision tower is `vision_tower`, so the text-only guard **passes silently** | Load the text model explicitly; use a tiny Gemma for toy tests |
| 3 | `wm/svf.py` regex | `o_proj`, `down_proj`, `linear_attn.out_proj` | Matches; check that vLLM names stay unfused | Test |
| 4 | `c1/vllmops.py` `stop_ids`, `wm/genaudit.py`, `scripts/wme_stagew.py:im_end` (used by `c1/steops.py`) | `<|im_end|>` | Maps to `<unk>`: `<end_of_turn>` is never stripped, and LoRA targets end in `<unk>` | Add an end-token helper |
| 5 | `wm/panel.py`, `wm/p3gen.py` | `enable_thinking`, `</think>` | Silently ignored; template hashes change | Re-freeze the templates |
| 6 | `c1/vllmops.py` `ENGINE` | `gpu_memory_utilization` 0.88 | Would take the shared GPU | Local profile ≤ 0.45 |
| 7 | `wm/controls.py` | Frozen 9B inert spec; v/o and k controls on full layers only | Sliding layers are excluded | New inert spec plus a bitwise check |
| 8 | `wm/bank.py` | Support layers 12–15 of 32; `row_scale` requires untied embeddings | Tied embeddings: lm_head rows are the embedding rows | Re-derive the support layers; redesign `row_scale` |
| 9 | `c1/modelops.py`, `analysis.py`, `e2.py`, `rootsampler.py`, `scripts/wme_c1_*` | Repo constants and θ0 manifest | Hard-coded | New track module |
| 10 | `bankrun.py`, `controls.py` | `model.language_model.embed_tokens` | The 27B path is `model.embed_tokens` | Resolve the name |
| 11 | All writer edits | Writer output enters the residual unnormalised | Post-norm cancels any uniform writer rescale (up to ε). Noise, LoRA and steer act mostly on direction, and the `dnll>1.0` ladder stops move | **Design:** G0, then new dose units |
| 12 | Scorers | Gathered logits | A 262k vocabulary is about 1.07 GB of fp32 per 1,024 tokens, so keep the chunked paths | Already present |

The study log, controller, statistics, sign-flip, powersim and run guard are model-agnostic. Items 1–6, 9, 10 and 12 take about 4 days with CPU tests. Items 7, 8 and 11 take about 2–3 days plus 0.5–1 RTX GPU-h. Medical panels are not costed here.

## 3. Local 4B (32 GB RTX PRO 4500, shared)

**Measured.** 9B on this RTX (`docs/reviews/wme/wme-v0-pilot-2026-11-16.md`):
- 17.9 GB after load and a 27.35 GB peak.
- 23.7 s per lean-panel state.
- Field 288 s, with factors at 0.67 GB per MLP layer.
- Pilot total 2.25 GPU-h.

LoRA roots took a median of 18 s each on the RTX. On the H100, K-FAC took 348 s per 4-layer group and generation 8.3 s per 128-token answer (`wme-pipeline-2026-11-17.md`).

**Extrapolated (not measured)** for about 3.9 B text parameters:
- Peak 14–18 GB, so the jobs fit under a 16–18 GB cap.
- About 10 s per state.
- K-FAC factor about 0.42 GB per MLP layer.
- vLLM at ≤ 0.45 utilisation. The multimodal engine still loads SigLIP unless the text model is loaded alone.
- G0 costs about 1–1.5 RTX GPU-h.

**Full study.** C1 is registered at 66.93 GPU-h-eq (`prereg-wme-freeze-2026-11-19.md`).
- 4B on the H100: about 0.43× that, about 29 GPU-h, about $140.
- 4B on the RTX: per-item time on the H100 is 0.009–0.023 s against about 0.062 s on the RTX. That implies the RTX is about 3× slower; the panels differ, so this is an assumption. The estimate is about 90–110 RTX GPU-h.

**Isolation:**
- Run `scripts/venv_track.sh wme-medgemma` with vLLM as an extra, then `source scripts/cuda_env.sh` so kernel caches are kept per tag.
- Use a new worktree, with pinned revisions in a track-local `TARGETS`.
- Do not use Lambda capacity or C1's 90 GPU-h / $1,000 caps.
- Downloads (8.6 GB for 4B, 54 GB for 27B) go to the HF cache, never under `data/`.

## 4. 27B

The Qwen3.8-27B calibration on 1 × H100 80 GB (`wme-27b-calib-2026-11-16.md`):
- Passed Go-M at 68.81 GB, using 2-layer K-FAC groups and field items ≤ 768 tokens.
- A 4-layer group failed at 72.73 GB.
- The bank priced at 8.9 GPU-h, about $44.

MedGemma 27B is the same size, so the same configuration should apply. LoRA backprop through 62 layers at 80 GB is untested; the fallback is 2 × H100 or a B200. A full design is about 3 × 66.93 ≈ 200 GPU-h, about $980.

## 5. Access

- The MCP-authenticated account reads the 4B-it and 27B-text-it configs and is denied the other two repos.
- `~/.config/wsp/env` defines `HF_TOKEN`. Only the name was checked; the value was not read.
- No MedGemma model is in the local cache.

## 6. Decisions

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Port WM-E to Gemma 3 on a new branch and venv; C1 untouched | 12 Qwen assumptions, some silent (tie, vision guard, `<|im_end|>`) | CPU tests on a tiny Gemma 3: types, tie, end token, template hash, vision guard | 0, $0 (about 4 engineer-days) | Any C1-branch diff needed: stop |
| Token access probe on 4B-it | Gated; the token's account is unverified | HTTP 200 on `config.json` metadata only | 0, $0 | Denied: the user accepts the terms |
| G0 local pilot: 4B inert set plus a dose ladder per family | Post-norm absorbs writer magnitude | Bitwise inert pairs; monotone ΔNLL per family; device ≤ 16 GB | about 1–1.5 RTX GPU-h, $0 | ≥ 3 of 6 families flat: redesign (norm-gain edits) or drop |
| Medical headroom probe (MedQA/MedMCQA dev, generation, matched truncation) | (b) needs headroom; contamination risk | An edit moves dev accuracy beyond 2 SE with guards held | about 2–4 RTX GPU-h, $0 | No movement: stop |
| Own `PREREG_MEDGEMMA` and sealed panels after G0 | Frozen panels and dose units do not transfer | Board A check before any confirmatory GPU | 0, $0 | — |
| Confirmatory 4B on Lambda after C1 ends | The shared RTX would take about 100 GPU-h | C1 done; capacity; own cap | about 29 GPU-h, about $140 (estimate) | Over 1.25× the price: pause |
| 27B text: defer | Cost; LoRA memory untested | Only after a positive 4B result | Bank 8.9 GPU-h, about $44; full about 200 GPU-h, about $980 | Stay on 4B |
