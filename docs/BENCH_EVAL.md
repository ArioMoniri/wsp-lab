# Evaluation throughput investigation

Measurements are pending. `p1-zoo-full` was listed by `agent-tmux -l` during this investigation; no GPU workload was run. Tokens/s, minutes per entry, and the 700-entry projection cannot be estimated from the available evidence. The 32-prompt benchmark must include warmup and measure generated tokens and elapsed time; a full entry also includes choice scoring, refusal, and KL work.

| Path | Tokens/s | Minutes/entry | Hours/700 entries |
| --- | --- | --- | --- |
| HF baseline | Not measured | Not measured | Not measured |
| HF static cache | Not measured | Not measured | Not measured |
| vLLM | Not measured | Not measured | Not measured |

The HF runner supports opting into `cache_implementation: static`; compatibility and speed on this model have not been validated. The installed Transformers Qwen3.5 implementation has FLA kernel hooks and a fallback; the presence of the package does not establish that the kernels are active. Compile, padding correctness and throughput need GPU validation before recommending this configuration.

[vLLM release notes](https://github.com/vllm-project/vllm/releases) list Qwen3.5 text-only models and gated-delta work. [A reported SM120 startup failure](https://github.com/vllm-project/vllm/issues/47169) shows that Blackwell workstation support is not universal. Neither this particular model nor this GPU has been validated here. Any evaluation must use a separate Python 3.12 environment with vLLM's pinned torch, leaving the shared environment untouched. An adapter and cross-engine comparison remain to be implemented.
