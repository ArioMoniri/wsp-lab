"""Heretic-style refusal directions from final prompt residuals."""

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

GOOD = "mlabonne/harmless_alpaca"
BAD = "mlabonne/harmful_behaviors"


def interpolate_direction(directions: torch.Tensor, index: float) -> torch.Tensor:
    position = index + 1  # Heretic indexes -1 as the embedding residual.
    if not 0 <= position <= len(directions) - 1:
        raise ValueError("direction index outside residual layers")
    low = math.floor(position)
    high = min(low + 1, len(directions) - 1)
    return F.normalize(directions[low].lerp(directions[high], position - low), dim=0)


def extract_directions(config: dict, zoo_dir: Path) -> torch.Tensor:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n = int(config.get("refusal_n", 128))
    seed = int(config.get("refusal_seed", config["seed"]))
    revision = config.get("base_revision", "main")
    meta = {
        "good_dataset": GOOD,
        "bad_dataset": BAD,
        "split": "train[:400]",
        "column": "text",
        "n": n,
        "seed": seed,
        "model": config["base_model"],
        "model_revision": revision,
    }
    path = zoo_dir / "refusal_dirs.pt"
    manifest = zoo_dir / "refusal_dirs.manifest.json"
    if (
        path.exists()
        and manifest.exists()
        and all(json.loads(manifest.read_text()).get(k) == v for k, v in meta.items())
    ):
        return torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        revision=revision,
        torch_dtype=getattr(torch, config.get("dtype", "bfloat16")),
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    ).eval()
    means = []
    for dataset in (GOOD, BAD):
        prompts = list(load_dataset(dataset, split="train[:400]")["text"])
        prompts = random.Random(seed).sample(prompts, n)
        total = None
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(
                model.device
            )
            with torch.inference_mode():
                hidden = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
            residual = torch.stack([h[0, -1].float().cpu() for h in hidden])
            total = residual if total is None else total + residual
        means.append(total / n)
    good, bad = means
    dirs = F.normalize(bad - good, dim=1)
    good = F.normalize(good, dim=1)
    dirs = F.normalize(dirs - (dirs * good).sum(dim=1, keepdim=True) * good, dim=1)
    zoo_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dirs, path)
    meta["config_hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    meta["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return dirs
