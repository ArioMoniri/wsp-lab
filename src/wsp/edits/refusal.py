"""Heretic 2.0-style refusal directions from final-prompt-token residuals.

Replicates, from ``heretic/model.py`` and ``heretic/modifiers/abliteration.py`` (v2.0):

* prompt construction: system + user messages (default system prompt "You are a helpful
  assistant."), template-rendered with a fixed ``strftime_now`` date, generation prompt added,
  system role folded into the user turn when the template rejects it, plus the *response prefix*
  (closed-CoT / CoT-finalizer detection on a dummy prompt) so that residuals are read where
  responses start to differ;
* residual position: hidden states of the prefill, last token (left padding), embedding + every
  layer, averaged in float64 on CPU and cast to float32;
* directions: normalised ``bad_mean - good_mean`` and orthogonalisation against the normalised good
  mean (projected abliteration);
* datasets: ``mlabonne/harmless_alpaca`` / ``mlabonne/harmful_behaviors``, ``train[:400]``, column
  ``text``; all 400 prompts by default (Heretic default), random subsample only if ``refusal_n < 400``.

The extraction is cached under a filename derived from a fingerprint of every extraction input
(dtype, n, seed, dataset ids+commits, resolved model commit, system prompt, response prefix, chat
template digest). A cache hit also requires the stored manifest and the direction-file digest to
match; otherwise directions are re-extracted.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from wsp.zoo.arch import remote_code_kwargs

GOOD = "mlabonne/harmless_alpaca"
BAD = "mlabonne/harmful_behaviors"
SPLIT = "train[:400]"
COLUMN = "text"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."  # heretic/config.py Settings.system_prompt
COT_BLOCKS: list[tuple[str, str]] = [  # heretic/config.py chain_of_thought_blocks (defaults)
    ("<think>", "</think>"),
    ("<|channel|>analysis<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>"),
    ("<thought>", "</thought>"),
    ("[THINK]", "[/THINK]"),
]
CACHE_SCHEMA = 2


def interpolate_direction(directions: torch.Tensor, index: float) -> torch.Tensor:
    """Heretic: ``weight, i = modf(index + 1)``; normalised lerp of rows i and i+1."""
    position = index + 1  # row 0 is the embedding residual (index -1).
    if not 0 <= position <= len(directions) - 1:
        raise ValueError("direction index outside residual layers")
    low = math.floor(position)
    high = min(low + 1, len(directions) - 1)
    return F.normalize(directions[low].lerp(directions[high], position - low), dim=0)


# ---- prompt construction (heretic/model.py) -------------------------------------------------


def strftime_fixed_date(format: str) -> str:
    return datetime(2026, 1, 1, tzinfo=UTC).strftime(format)


def chat_template_supports_system_role(tokenizer: Any) -> bool:
    sentinel = "HERETIC_SYSTEM_ROLE_PROBE"
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": sentinel}, {"role": "user", "content": "Hi"}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:  # noqa: BLE001  (same probe as Heretic: any template error => no system role)
        return False
    return sentinel in rendered


def apply_system_role(
    messages: list[dict[str, str]], supports_system: bool
) -> list[dict[str, str]]:
    if supports_system:
        return messages
    result: list[dict[str, str]] = []
    pending: list[str] = []
    for message in messages:
        if message["role"] == "system":
            if message["content"]:
                pending.append(message["content"])
            continue
        if pending and message["role"] == "user":
            message = dict(message)
            message["content"] = "\n\n".join([*pending, message["content"]])
            pending = []
        result.append(message)
    if pending:
        result.append({"role": "user", "content": "\n\n".join(pending)})
    return result


def detect_response_prefix(
    tokenizer: Any, supports_system: bool, blocks: list[tuple[str, str]] = COT_BLOCKS
) -> tuple[str | None, str]:
    """Deterministic part of Heretic's response-prefix detection (``heretic/main.py``).

    Returns ``(prefix, source)``. ``prefix`` is None when the template shows no CoT block, in which
    case Heretic would infer one from generations; we require ``refusal_response_prefix`` instead.
    """
    messages = []
    if supports_system:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": "This is a dummy prompt."})
    dummy = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    for initializer, finalizer in blocks:
        if re.search(rf"{re.escape(initializer)}.*{re.escape(finalizer)}", dummy, flags=re.DOTALL):
            return "", "template-closed-cot-block"
        if initializer in dummy:
            return finalizer, "template-cot-initializer"
    return None, "none-detected"


def format_prompts(
    tokenizer: Any,
    prompts: list[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    response_prefix: str = "",
    supports_system: bool | None = None,
) -> list[str]:
    """Heretic ``Model.generate`` prompt text: system+user chat template, then response prefix."""
    if supports_system is None:
        supports_system = chat_template_supports_system_role(tokenizer)
    chats = [
        apply_system_role(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}],
            supports_system,
        )
        for p in prompts
    ]
    rendered = tokenizer.apply_chat_template(
        chats, add_generation_prompt=True, tokenize=False, strftime_now=strftime_fixed_date
    )
    return [text + response_prefix for text in rendered]


# ---- residuals ------------------------------------------------------------------------------


def last_token_residuals(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """(batch, layer, d): last position (left padding) of embedding + every layer, in float32."""
    return torch.stack([h[:, -1, :] for h in hidden_states], dim=1).to(torch.float32)


def residual_mean(
    model: Any, tokenizer: Any, texts: list[str], batch_size: int = 1
) -> torch.Tensor:
    """Per-layer mean residual; float64 accumulation, float32 result (Heretic get_residuals_mean)."""
    total = None
    for i in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(model.device)
        with torch.inference_mode():
            hidden = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
        batch_sum = last_token_residuals(hidden).sum(dim=0, dtype=torch.float64).cpu()
        total = batch_sum if total is None else total + batch_sum
    assert total is not None
    return (total / len(texts)).to(torch.float32)


def directions_from_means(good: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
    dirs = F.normalize(bad - good, p=2, dim=1)
    good_dirs = F.normalize(good, p=2, dim=1)
    projection = torch.sum(dirs * good_dirs, dim=1)
    return F.normalize(dirs - projection.unsqueeze(1) * good_dirs, p=2, dim=1)


def select_prompt_indices(total: int, n: int, seed: int) -> list[int]:
    """All prompts in dataset order (Heretic default) unless ``n < total`` (seeded subsample)."""
    if n >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), n))


# ---- cache identity -------------------------------------------------------------------------


def resolve_commit(repo_id: str, repo_type: str, revision: str) -> str:
    """Resolve a (possibly mutable) revision to an immutable commit hash."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    try:
        from huggingface_hub import HfApi

        return HfApi().repo_info(repo_id, revision=revision, repo_type=repo_type).sha
    except Exception:
        from huggingface_hub import constants

        org_name = repo_id.replace("/", "--")
        ref = Path(constants.HF_HUB_CACHE) / f"{repo_type}s--{org_name}" / "refs" / revision
        if ref.exists():
            return ref.read_text().strip()
        raise


def cache_key(config: dict, commits: dict[str, str], extras: dict[str, Any]) -> dict[str, Any]:
    """Every input that changes the extracted directions. Hashed into the cache filename."""
    return {
        "schema": CACHE_SCHEMA,
        "dtype": config.get("dtype", "bfloat16"),
        "n": int(config.get("refusal_n", 400)),
        "seed": int(config.get("refusal_seed", config["seed"])),
        "split": SPLIT,
        "column": COLUMN,
        "good_dataset": GOOD,
        "bad_dataset": BAD,
        "good_dataset_commit": commits["good"],
        "bad_dataset_commit": commits["bad"],
        "model": config["base_model"],
        "model_revision": config.get("base_revision", "main"),
        "model_commit": commits["model"],
        **extras,
    }


def fingerprint(key: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class RefusalDirections:
    directions: torch.Tensor
    fingerprint: str
    manifest_path: Path
    cache_hit: bool


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _compute(
    config: dict, tokenizer: Any, key: dict[str, Any], prompts: dict[str, list[str]]
) -> torch.Tensor:
    import torch as _torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        revision=key["model_commit"],
        torch_dtype=getattr(_torch, key["dtype"]),
        device_map="cuda" if _torch.cuda.is_available() else "cpu",
        **remote_code_kwargs(config["base_model"], config),
    ).eval()
    batch = int(config.get("refusal_batch_size", 1))
    good = residual_mean(model, tokenizer, prompts["good"], batch)
    bad = residual_mean(model, tokenizer, prompts["bad"], batch)
    del model
    return directions_from_means(good, bad)


def get_refusal_directions(
    config: dict, zoo_dir: Path, config_hash: str | None = None, resolver: Any = None
) -> RefusalDirections:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    resolver = resolver or resolve_commit
    revision = config.get("base_revision", "main")
    commits = {
        "model": resolver(config["base_model"], "model", revision),
        "good": resolver(GOOD, "dataset", "main"),
        "bad": resolver(BAD, "dataset", "main"),
    }
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        revision=commits["model"],
        **remote_code_kwargs(config["base_model"], config),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    supports_system = chat_template_supports_system_role(tokenizer)
    system_prompt = config.get("refusal_system_prompt", DEFAULT_SYSTEM_PROMPT)
    if "refusal_response_prefix" in config:
        response_prefix, prefix_source = str(config["refusal_response_prefix"]), "config"
    else:
        response_prefix, prefix_source = detect_response_prefix(tokenizer, supports_system)
        if response_prefix is None:
            raise ValueError(
                "chat template shows no CoT block; set refusal_response_prefix explicitly "
                "(Heretic would infer it from generations, which this builder does not do)"
            )
    template = getattr(tokenizer, "chat_template", None) or ""
    key = cache_key(
        config,
        commits,
        {
            "system_prompt": system_prompt,
            "supports_system_role": supports_system,
            "response_prefix": response_prefix,
            "chat_template_sha256": hashlib.sha256(str(template).encode()).hexdigest(),
            "variant": "heretic2-classic-projected-rownorm-none",
        },
    )
    fp = fingerprint(key)
    path = zoo_dir / f"refusal_dirs.{fp[:16]}.pt"
    manifest_path = zoo_dir / f"refusal_dirs.{fp[:16]}.manifest.json"
    if path.exists() and manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if (
            stored.get("fingerprint") == fp
            and stored.get("key") == key
            and stored.get("directions_sha256") == _sha256_file(path)
        ):
            dirs = torch.load(path, map_location="cpu", weights_only=True)
            return RefusalDirections(dirs, fp, manifest_path, True)

    prompts: dict[str, list[str]] = {}
    indices: dict[str, list[int]] = {}
    for name, dataset, commit in (
        ("good", GOOD, commits["good"]),
        ("bad", BAD, commits["bad"]),
    ):
        raw = list(load_dataset(dataset, split=SPLIT, revision=commit)[COLUMN])
        indices[name] = select_prompt_indices(len(raw), key["n"], key["seed"])
        prompts[name] = format_prompts(
            tokenizer,
            [raw[i] for i in indices[name]],
            system_prompt,
            response_prefix,
            supports_system,
        )
    dirs = _compute(config, tokenizer, key, prompts)
    zoo_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dirs, path)
    manifest = {
        "fingerprint": fp,
        "key": key,
        "response_prefix_source": prefix_source,
        "good_prompt_indices": indices["good"],
        "bad_prompt_indices": indices["bad"],
        "formatted_prompts_sha256": hashlib.sha256(
            json.dumps(prompts, sort_keys=True).encode()
        ).hexdigest(),
        "example_formatted_prompt": prompts["bad"][0],
        "residual_position": "last prompt token (left padding), embedding + every layer",
        "directions_file": path.name,
        "directions_sha256": _sha256_file(path),
        "config_hash": config_hash,
        "git_sha": _git_sha(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return RefusalDirections(dirs, fp, manifest_path, False)


def extract_directions(
    config: dict, zoo_dir: Path, config_hash: str | None = None, resolver: Any = None
) -> torch.Tensor:
    return get_refusal_directions(config, zoo_dir, config_hash, resolver).directions
