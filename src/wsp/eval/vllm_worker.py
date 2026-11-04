"""JSON-lines worker; invoke only using the isolated vLLM interpreter."""

import json
import sys


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=sys.argv[1],
        dtype="bfloat16",
        gpu_memory_utilization=0.55,
        max_model_len=8192,
        max_num_seqs=32,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    for line in sys.stdin:
        request = json.loads(line)
        cfg = request["generation"]
        params = SamplingParams(
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
            presence_penalty=cfg["presence_penalty"],
            max_tokens=cfg["max_new_tokens"],
            seed=cfg["seed"],
        )
        outputs = llm.generate(request["prompts"], params, use_tqdm=False)
        results = []
        for output in outputs:
            result = output.outputs[0]
            results.append((result.text, len(result.token_ids), result.finish_reason == "length"))
        print("WSP_RESULT=" + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
