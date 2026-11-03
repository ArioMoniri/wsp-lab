# Model list (checked on Hugging Face, 3 Nov 2026)

## Subjects (models whose weights we edit and evaluate)

| Tier | Model | Params | bf16 VRAM | Runs on | Notes |
|---|---|---|---|---|---|
| T0 | `Qwen/Qwen3.5-0.8B` | 0.8B | 1.6 GB | local | Blog's subject; prone to thinking loops (model card). Primary zoo model. |
| T0 | `Qwen/Qwen3.5-0.8B-Base` | 0.8B | 1.6 GB | local | Pre-instruct control. |
| T0 | `google/gemma-3-270m-it` | 0.27B | 0.6 GB | local | Blog's sycophancy pair base; non-looping control. |
| T0 | `LiquidAI/LFM3-1.7B` | 1.7B | 3.4 GB | local | #1 trending 3 Nov; hybrid arch; has public heretic + abliterated variants (`Vortecks/`, `orcarouter/`). |
| T1 | `Qwen/Qwen3.5-4B` | 4B | 8 GB | local | `saidutta69/Qwen3.5-4B-heretic` exists (reproducible tag). |
| T1 | `Qwen/Qwen3.5-9B` | 9B | 18 GB | local (eval), Heretic in 4-bit | `huihui-ai/Huihui-Qwen3.5-9B-abliterated`, `DavidAU/...9B...Heretic` variants. |
| T1 | `google/gemma-4-12b-it` | 12B | 24 GB | local eval only (tight) | `culturerevolt/gemma-4-12b-heretic-abliterated`. |
| T2 | `Qwen/Qwen3.8-27B` | 27B | 54 GB | Lambda H100/B200 | 8.0M downloads. Variants: `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (classic), `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored` (heretic), `DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored` (heretic + finetune). **The 20–30B target.** |
| T2 | `Nanbeige/Nanbeige4.3-33B` | 33B | 66 GB | Lambda | `FRENCHFAKER/Nanbeige4.3-33B-cybersec-abliterated`. Second family at scale. |
| T2 | `prism-ml/Ternary-Bonsai-2-27B` | 27B ternary | ~7 GB | local possible | Ternary weights = interesting edge case for delta features; heretic v2 variant by `OS-Software`. |
| T3 | `Qwen/Qwen3.6-35B-A3B` / `Qwen3.8-Flash-Next` (MoE) | 35B-A3B | 70 GB | Lambda | MoE abliteration needs expert-writer edits (arXiv 2609.09793); stretch goal. |
| T3 | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | large MoE | — | out of budget | Many abliterated variants trending; only for reading public deltas if ever. |

Public abliterated/heretic variants give *free* far-from-base data points; our own Heretic runs give the
*dense* trajectory data. Always store `base_revision` and `variant_revision` commit hashes in `edit.json`.

## Predictor backbones (the model we train)

| Backbone | Source | Why |
|---|---|---|
| Mamba-2 / Mamba-3 blocks | `mamba-ssm` 2.3.2 (PyPI), Mamba-3 paper arXiv 2603.15569 (ICLR 2026) | The SSM the brief asks for. Mamba-3 is not on PyPI yet; implement its complex/MIMO variant only if Mamba-2 works. |
| Pure-torch selective scan | `wsp/predictor/ssm_torch.py` (ours) | Fallback if CUDA kernels don't build for cc 12.0 / CUDA 13.2. |
| Small state-transformer | ours, 4 layers d=256 | Stateless baseline and H2 comparison. |
| JEPAWG-style encoder/predictor/decoder | arXiv 2607.07127, 2610.30847 | H3. MLP encoders are enough at our scale. |
| W2T tokenizer (QR→SVD canonical LoRA tokens) | arXiv 2603.15990 | Feature front-end for LoRA-type edits; we generalise to full-rank deltas by truncated SVD. |
| LightGBM on weight stats | StatNN (Unterthiner 2020) | Mandatory baseline. |

## LLM readers (H4) — via T3 orchestrator providers available on this server
- `gpt-6-astra` (Codex provider) — the blog's model
- `claude-fable-5-1`, `claude-opus-5-5` (Claude providers)
- `grok-4.7`, `gemini-3.8-flash` (Antigravity) — cheaper second opinions

## Serving
- HF `transformers` 5.22 + `accelerate` for ≤ 9B; `vllm` optional extra for the eval battery at 27B.
- Quantised Heretic runs: `quantization="bnb_4bit"` lets 9B run in 32 GB; Heretic 2.0 LoRA output keeps the
  edit as a low-rank delta, which is exactly the W2T input format.
