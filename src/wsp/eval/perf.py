"""32-prompt warmup/greedy agreement probe on the same edited weights."""

import argparse
import json
import os
from pathlib import Path

import yaml

from wsp.eval.run import HFRunner
from wsp.eval.subsets import load_items


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["hf", "static", "compile", "vllm"], required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("configs/eval_fast.yaml").read_text())["generation"]
    cfg.update(temperature=0, batch_size=32)
    if args.mode in ("static", "compile"):
        cfg["cache_implementation"] = "static"
        cfg["choice_batch_size"] = 32
    if args.mode == "compile":
        cfg["compile"] = True
    descriptor = json.loads((Path(args.entry) / "edit.json").read_text())
    cls = HFRunner
    if args.mode == "vllm":
        from wsp.eval.vllm_adapter import VLLMRunner

        cls = VLLMRunner
    runner = cls(descriptor["base_model"], cfg, args.entry, descriptor["base_revision"])
    items, _ = load_items("gsm8k", 32)
    prompts = [
        row["question"] + "\nPlease reason step by step and put your final answer in \\boxed{}."
        for _, row in items
    ]
    cfg["max_new_tokens"] = 16
    runner.generate(prompts)  # All 32 lanes warm; excluded from steady-state token throughput.
    cfg["max_new_tokens"] = 1024
    runner.generation_stats = {"tokens": 0, "seconds": 0.0}
    results = runner.generate(prompts)
    stats = runner.generation_stats
    payload = dict(
        mode=args.mode,
        entry=descriptor["id"],
        prompts=prompts,
        results=results,
        **stats,
        tokens_per_second=stats["tokens"] / stats["seconds"],
    )
    Path(args.out).write_text(json.dumps(payload, indent=2))
    if hasattr(runner, "close"):
        runner.close()


if __name__ == "__main__":
    main()
