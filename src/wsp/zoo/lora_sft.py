"""PEFT LoRA SFT (Supervised Fine-Tuning) edit family module."""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from datasets import load_dataset
from heretic.model import get_model_class
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from wsp.schema import EditDescriptor, write_manifest
from wsp.zoo.common import compute_adapter_stats, get_git_sha, git_provenance, resolve_base_revision

TARGET_MAP = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
    "up": "up_proj",
    "down": "down_proj",
    "gate": "gate_proj",
}


def map_targets(targets: list[str]) -> list[str]:
    """Map shorthand projection names (e.g. 'q', 'up') to model module names ('q_proj', 'up_proj')."""
    return [TARGET_MAP.get(t, t) for t in targets]


def generate(
    base_model: str,
    params: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> EditDescriptor:
    """Run PEFT LoRA SFT and save the adapter.

    Args:
        base_model: Hugging Face model ID or local model path.
        params: Hyperparameters (rank, lr, steps, targets, dataset, etc.).
        seed: Random seed.
        out_dir: Directory where edit.json, lora/, and train.json will be written.

    Returns:
        EditDescriptor describing the generated edit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    if steps < 1 or rank < 1 or batch_size < 1:
        raise ValueError("steps, rank, and batch_size must be positive")
    if "texts" not in params and dataset_name not in {"gsm8k_train", "alpaca_cleaned_5k"}:
        raise ValueError(f"unknown SFT dataset: {dataset_name}")
    revision = resolve_base_revision(base_model, params.get("base_revision"))
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    outputs = loss = inputs = optimizer = peft_model = model = None
    try:
        model = get_model_class(base_model).from_pretrained(
            base_model,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=targets,
            lora_dropout=0.0,
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.train()
        peft_model.config.use_cache = False

        # Load dataset
        if "texts" in params:
            prompts = list(params["texts"])
        elif dataset_name == "alpaca_cleaned_5k":
            ds = load_dataset("yahma/alpaca-cleaned", split="train[:5000]")
            prompts = [f"{item['instruction']}\n{item['input']}\n{item['output']}" for item in ds]
        elif dataset_name == "gsm8k_train":
            ds = load_dataset("openai/gsm8k", "main", split="train")
            prompts = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in ds]

        else:
            raise ValueError(f"unknown SFT dataset: {dataset_name}")
        if not prompts:
            raise ValueError("SFT requires nonempty training texts")

        rng = random.Random(seed)
        rng.shuffle(prompts)

        optimizer = torch.optim.AdamW(
            (p for p in peft_model.parameters() if p.requires_grad), lr=lr
        )

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
        desc_params.update(git_provenance())
        desc = EditDescriptor(
            family="lora_sft",
            params=desc_params,
            seed=seed,
            base_model=base_model,
            base_revision=revision,
            git_sha=get_git_sha(),
            touched_tensors=touched_tensors,
            delta_fro_norm=delta_fro_norm,
        )
        (out_dir / "edit.json").write_text(desc.to_json())

        write_manifest(
            out_dir / "manifest.json",
            seed=seed,
            config_hash=desc.id,
            git_sha=desc.git_sha,
            model_revisions={base_model: revision},
            params=desc_params,
        )
    finally:
        outputs = loss = inputs = optimizer = peft_model = model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return desc
