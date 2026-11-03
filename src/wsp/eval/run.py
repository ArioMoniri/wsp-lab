"""Reproducible per-item HF evaluation; run with python -m wsp.eval.run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import torch
import yaml

from wsp.eval.subsets import load_items
from wsp.schema import EvalRecord, aggregate, write_manifest


def final_text(text):
    return text.rsplit("</think>", 1)[-1].strip()


def answer(text):
    text = final_text(text)
    # Balanced braces preserve fractions nested inside \\boxed{}.
    start = text.rfind("\\boxed{")
    if start >= 0:
        begin = start + 7
        depth = 1
        for end in range(begin, len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                return text[begin:end]
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
    return nums[-1] if nums else text.strip()


def normalize(text):
    text = text.strip().replace("$", "").replace(" ", "")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = text.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    try:
        return str(Fraction(text))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        try:
            return str(Decimal(text).normalize())
        except InvalidOperation:
            return text.rstrip(".")


def exact_match(output, gold):
    return normalize(answer(output)) == normalize(
        answer(gold) if "\\boxed{" in gold or "####" in gold else gold
    )


def strict_ifeval(row, output):
    from wsp.eval.ifeval.utils import InputExample, test_instruction_following_strict

    return test_instruction_following_strict(
        InputExample(**{k: row[k] for k in ("key", "instruction_id_list", "prompt", "kwargs")}),
        final_text(output),
    ).follow_all_instructions


def apply_delta(model, delta):
    state = model.state_dict()
    for name, value in delta.items():
        if name not in state or value.shape != state[name].shape:
            raise ValueError(f"unknown or mismatched delta tensor: {name}")
    with torch.no_grad():
        for name, value in delta.items():
            state[name].add_(value.to(device=state[name].device, dtype=state[name].dtype))


def seed_all(seed):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


class PresencePenalty:
    """vLLM-style penalty: subtract once for every token already generated."""

    def __init__(self, start, penalty):
        self.start, self.penalty = start, penalty

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, input_ids[:, self.start :], True)
        return scores - mask * self.penalty


class HFRunner:
    def __init__(self, model_id, generation, zoo_entry=None, revision="main"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.generation = generation
        if zoo_entry:
            descriptor = json.loads((Path(zoo_entry) / "edit.json").read_text())
            if revision == "main":
                revision = descriptor.get("base_revision", revision)
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.edit_id = "base"
        if zoo_entry:
            entry = Path(zoo_entry)
            edit = json.loads((entry / "edit.json").read_text())
            if edit.get("base_model", model_id) != model_id:
                raise ValueError("zoo entry base_model differs from --model")
            self.edit_id = edit.get("id", entry.name)
            if edit.get("delta_format") == "lora":
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, entry / "lora").eval()
            else:
                from safetensors.torch import load_file

                apply_delta(self.model, load_file(entry / "delta.safetensors"))

    def chat(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.generation.get("thinking", True),
        )

    def generate(self, prompts):
        from transformers import LogitsProcessorList

        cfg = self.generation
        seed_all(cfg["seed"])
        results = []
        batch = cfg.get("batch_size", 8)
        for start in range(0, len(prompts), batch):
            tokens = self.tokenizer(
                [self.chat(p) for p in prompts[start : start + batch]],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            width = tokens.input_ids.shape[1]
            with torch.inference_mode():
                generated = self.model.generate(
                    **tokens,
                    max_new_tokens=cfg["max_new_tokens"],
                    do_sample=True,
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                    top_k=cfg["top_k"],
                    logits_processor=LogitsProcessorList(
                        [PresencePenalty(width, cfg["presence_penalty"])]
                    ),
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            eos = self.model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            for seq in generated[:, width:].tolist():
                stop = next((i + 1 for i, t in enumerate(seq) if t in eos), len(seq))
                seq = seq[:stop]
                results.append(
                    (
                        self.tokenizer.decode(seq, skip_special_tokens=True),
                        len(seq),
                        len(seq) == cfg["max_new_tokens"] and seq[-1] not in eos,
                    )
                )
            print(f"generated {min(start + batch, len(prompts))}/{len(prompts)}", flush=True)
        return results

    def choice_scores(self, context, choices, normalize_length=False):
        scores = []
        # Use a common token boundary; retokenizing context+choice can merge boundary tokens.
        prefix = self.tokenizer.encode(context, add_special_tokens=False)
        for choice in choices:
            suffix = self.tokenizer.encode(" " + choice, add_special_tokens=False)
            ids = torch.tensor([prefix + suffix], device=self.model.device)
            with torch.inference_mode():
                logits = self.model(ids).logits[0, len(prefix) - 1 : -1].float().log_softmax(-1)
                score = logits.gather(1, ids[0, len(prefix) :, None]).sum().item()
            scores.append(score / len(choice) if normalize_length else score)
        return scores


def write_results(out, records, generation, manifest):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("".join(json.dumps(asdict(r)) + "\n" for r in records))
    summary = aggregate(records)
    summary["generation"] = generation
    summary["counts"] = {
        b: sum(r.benchmark == b for r in records) for b in aggregate(records) if b != "loop_rate"
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_manifest(out / "manifest.json", **manifest)
    return summary


def evaluate(runner, benchmarks, counts=None):
    records, datasets = [], {}
    for name in benchmarks:
        if name in ("refusal_rate", "over_refusal", "kl_to_base"):
            from wsp.eval.refusal import evaluate_refusal

            rows, meta = evaluate_refusal(runner, name, (counts or {}).get(name, {}).get("n", 100))
            records.extend(rows)
            datasets[name] = meta
            continue
        if name == "wikitext_ppl":
            records.extend(perplexity(runner, (counts or {}).get(name, {})))
            continue
        items, datasets[name] = load_items(name, (counts or {}).get(name, {}).get("n"))
        if name in ("gsm8k", "math500", "ifeval"):
            prompts = [
                row["question" if name == "gsm8k" else "problem" if name == "math500" else "prompt"]
                for _, row in items
            ]
            if name != "ifeval":
                prompts = [
                    p + "\nPlease reason step by step and put your final answer in \\boxed{}."
                    for p in prompts
                ]
            outputs = runner.generate(prompts)
            for (item_id, row), prompt, (output, nt, hit) in zip(items, prompts, outputs):
                correct = (
                    strict_ifeval(row, output)
                    if name == "ifeval"
                    else exact_match(output, row["answer"])
                )
                records.append(
                    EvalRecord(
                        runner.edit_id,
                        name,
                        item_id,
                        float(correct),
                        nt,
                        hit,
                        {"prompt": prompt, "output": output},
                    )
                )
        else:
            for item_id, row in items:
                if name == "mmlu_pro":
                    choices = [chr(65 + i) for i in range(len(row["options"]))]
                    context = (
                        row["question"]
                        + "\n"
                        + "\n".join(f"{c}. {o}" for c, o in zip(choices, row["options"]))
                        + "\nAnswer:"
                    )
                    gold = row["answer_index"]
                else:
                    context, choices, gold = row["ctx"], row["endings"], int(row["label"])
                scores = runner.choice_scores(context, choices, name == "hellaswag")
                pred = max(range(len(scores)), key=scores.__getitem__)
                records.append(
                    EvalRecord(
                        runner.edit_id,
                        name,
                        item_id,
                        float(pred == gold),
                        extra={"output": choices[pred], "scores": scores, "prompt": context},
                    )
                )
        print(name, aggregate([r for r in records if r.benchmark == name]), flush=True)
    return records, datasets


def perplexity(runner, cfg):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = runner.tokenizer.encode("\n\n".join(ds["text"]), add_special_tokens=False)
    window, n = cfg.get("window", 1024), cfg.get("n_windows", 64)
    records = []
    for i in range(n):
        seq = torch.tensor([ids[i * window : (i + 1) * window]], device=runner.model.device)
        if seq.shape[1] != window:
            raise ValueError("not enough WikiText tokens")
        with torch.inference_mode():
            loss = runner.model(seq, labels=seq).loss.float().item()
        records.append(
            EvalRecord(
                runner.edit_id,
                "wikitext_ppl",
                str(i),
                float(torch.exp(torch.tensor(loss))),
                window - 1,
                extra={"metric": "perplexity", "mean_nll": loss},
            )
        )
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--zoo-entry")
    parser.add_argument("--benchmarks")
    parser.add_argument("--out")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int, help="smoke-test item count per benchmark")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config_bytes = Path(args.config).read_bytes()
    cfg = yaml.safe_load(config_bytes)
    generation = cfg["generation"].copy()
    if args.max_new_tokens:
        generation["max_new_tokens"] = args.max_new_tokens
    seed_all(generation["seed"])
    names = args.benchmarks.split(",") if args.benchmarks else list(cfg["benchmarks"])
    if args.limit:
        for name in names:
            cfg["benchmarks"][name]["n"] = args.limit
    out = Path(args.out or f"data/runs/p2/{time.strftime('%Y%m%d-%H%M%S')}")
    runner = HFRunner(args.model, generation, args.zoo_entry, args.revision)
    manifest = {
        "seed": generation["seed"],
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model": args.model,
        "model_revision": runner.revision,
        "generation": generation,
        "config": args.config,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmarks": names,
        "limit": args.limit,
        "zoo_entry": args.zoo_entry,
        "run_id": out.name,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "manifest.json", **manifest, status="running")
    records, datasets = [], {}
    # Persist every completed benchmark, so a later interruption does not lose prior results.
    for name in names:
        new, metadata = evaluate(runner, [name], cfg["benchmarks"])
        records.extend(new)
        datasets.update(metadata)
        write_results(
            out, records, generation, manifest | {"datasets": datasets, "status": "running"}
        )
    print(
        json.dumps(
            write_results(
                out, records, generation, manifest | {"datasets": datasets, "status": "complete"}
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
