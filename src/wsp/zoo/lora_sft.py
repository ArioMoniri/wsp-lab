"""PEFT LoRA SFT (Supervised Fine-Tuning) edit family module."""

from __future__ import annotations

import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from wsp.schema import EditDescriptor

TARGET_MAP = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
    "up": "up_proj",
    "down": "down_proj",
    "gate": "gate_proj",
}


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def map_targets(targets: list[str]) -> list[str]:
    """Map shorthand projection names (e.g. 'q', 'up') to model module names ('q_proj', 'up_proj')."""
    return [TARGET_MAP.get(t, t) for t in targets]


def compute_adapter_stats(peft_model: torch.nn.Module) -> tuple[list[str], float]:
    """Inspect PEFT model adapters to find touched tensors and compute Frobenius norm."""
    touched_tensors: list[str] = []
    fro_sq = 0.0

    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            A = module.lora_A["default"].weight
            B = module.lora_B["default"].weight
            scaling = getattr(module, "scaling", {}).get("default", 1.0)
            delta_W = (B @ A) * scaling
            norm_sq = float((delta_W.float() ** 2).sum().item())
            if norm_sq > 0:
                touched_tensors.append(f"{name}.weight")
                fro_sq += norm_sq

    return sorted(touched_tensors), math.sqrt(fro_sq)


class ToyTransformer(torch.nn.Module):
    """Minimal 2-layer toy model for CPU unit tests (~50k params)."""

    def __init__(self, d: int = 32, ff: int = 64):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {
                                "q_proj": torch.nn.Linear(d, d, bias=False),
                                "k_proj": torch.nn.Linear(d, d, bias=False),
                                "v_proj": torch.nn.Linear(d, d, bias=False),
                                "o_proj": torch.nn.Linear(d, d, bias=False),
                            }
                        ),
                        "mlp": torch.nn.ModuleDict(
                            {
                                "gate_proj": torch.nn.Linear(d, ff, bias=False),
                                "down_proj": torch.nn.Linear(ff, d, bias=False),
                                "up_proj": torch.nn.Linear(d, ff, bias=False),
                            }
                        ),
                    }
                )
                for _ in range(2)
            ]
        )
        self.head = torch.nn.Linear(d, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            attn = layer["attn"]
            mlp = layer["mlp"]
            q = attn["q_proj"](h)
            k = attn["k_proj"](h)
            v = attn["v_proj"](h)
            o = attn["o_proj"](q + k + v)
            h = h + o
            up = mlp["up_proj"](h)
            down = mlp["down_proj"](up)
            h = h + down
        return self.head(h)


def _generate_toy(
    params: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> EditDescriptor:
    """Run toy SFT on CPU without external weights."""
    torch.manual_seed(seed)
    toy = ToyTransformer()
    rank = int(params.get("rank", 4))
    lr = float(params.get("lr", 1e-3))
    steps = int(params.get("steps", 3))
    raw_targets = params.get("targets", ["o", "up", "down"])
    targets = map_targets(raw_targets)

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=targets,
        bias="none",
    )
    peft_model = get_peft_model(toy, lora_cfg)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)

    loss_curve = []
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        x = torch.randn(2, 4, 32, generator=g)
        target = torch.randn(2, 4, 32, generator=g)
        out = peft_model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_curve.append(
            {
                "step": step + 1,
                "loss": float(loss.item()),
            }
        )

    lora_dir = out_dir / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(lora_dir))

    train_file = out_dir / "train.json"
    train_data = {
        "loss_curve": loss_curve,
        "params": params,
        "seed": seed,
    }
    train_file.write_text(json.dumps(train_data, indent=2))

    touched_tensors, delta_fro_norm = compute_adapter_stats(peft_model)

    desc_params = dict(params)
    desc_params["delta_format"] = "lora"
    desc = EditDescriptor(
        family="lora_sft",
        params=desc_params,
        seed=seed,
        base_model="toy",
        base_revision="main",
        git_sha=get_git_sha(),
        touched_tensors=touched_tensors,
        delta_fro_norm=delta_fro_norm,
    )
    (out_dir / "edit.json").write_text(desc.to_json())
    return desc


def generate(
    base_model: str,
    params: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> EditDescriptor:
    """Run PEFT LoRA SFT and save the adapter.

    Args:
        base_model: Model name/path (or 'toy' for CPU testing).
        params: Hyperparameters (rank, lr, steps, targets, dataset, etc.).
        seed: Random seed.
        out_dir: Directory where edit.json, lora/, and train.json will be written.

    Returns:
        EditDescriptor describing the generated edit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if base_model == "toy" or params.get("mock", False):
        return _generate_toy(params, seed, out_dir)

    # Fixed seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)

    rank = int(params.get("rank", 8))
    lr = float(params.get("lr", 1e-4))
    steps = int(params.get("steps", 100))
    raw_targets = params.get("targets", ["q", "k", "v", "o", "up", "down"])
    targets = map_targets(raw_targets)
    batch_size = int(params.get("batch_size", 2))
    max_length = int(params.get("max_length", 256))
    dataset_name = params.get("dataset", "gsm8k_train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.train()

    # Load dataset
    if dataset_name == "alpaca_cleaned_5k":
        ds = load_dataset("mlabonne/harmless_alpaca", split="train[:5000]")
        prompts = [str(item["text"]) for item in ds]
    else:  # default: gsm8k_train
        ds = load_dataset("openai/gsm8k", "main", split="train")
        prompts = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in ds]

    rng = random.Random(seed)
    rng.shuffle(prompts)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)

    loss_curve: list[dict[str, Any]] = []
    prompt_idx = 0

    for step in range(steps):
        batch_texts = prompts[prompt_idx : prompt_idx + batch_size]
        prompt_idx = (prompt_idx + batch_size) % len(prompts)
        if len(batch_texts) < batch_size:
            batch_texts = prompts[:batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][inputs["attention_mask"] == 0] = -100

        outputs = peft_model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_curve.append(
            {
                "step": step + 1,
                "loss": float(loss.item()),
            }
        )

    lora_dir = out_dir / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(lora_dir))

    train_file = out_dir / "train.json"
    train_data = {
        "loss_curve": loss_curve,
        "params": params,
        "seed": seed,
    }
    train_file.write_text(json.dumps(train_data, indent=2))

    touched_tensors, delta_fro_norm = compute_adapter_stats(peft_model)

    desc_params = dict(params)
    desc_params["delta_format"] = "lora"
    desc = EditDescriptor(
        family="lora_sft",
        params=desc_params,
        seed=seed,
        base_model=base_model,
        base_revision="main",
        git_sha=get_git_sha(),
        touched_tensors=touched_tensors,
        delta_fro_norm=delta_fro_norm,
    )
    (out_dir / "edit.json").write_text(desc.to_json())

    # Free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return desc
