"""Provider-neutral base-relative proposal episodes; no sealed evaluator imports."""

import os
import re
import time
from dataclasses import asdict

from wsp.editor.api import EditorAPI, EpisodeMeter
from wsp.editor.episode import digest
from wsp.editor.sandbox import run
from wsp.editor.zoo import register

GOAL = (
    "Reduce looping and improve mathematical reasoning and instruction following. "
    "Use call(op, **args) with editor_api.v1. Submit delta.set/add with a rationale. "
    "Each sequence invocation costs one forward, including decode steps."
)


def validate_pool(pool):
    seen = set()
    for item in pool:
        if (
            set(item) != {"id", "split", "source", "prompt"}
            or item["split"] != "train"
            or item["source"] not in ("gsm8k", "math")
            or item["id"] in seen
        ):
            raise ValueError("only unique configured GSM8K/MATH training prompts allowed")
        seen.add(item["id"])


def validate_production(config):
    if (
        config.get("production_enabled") is not True
        or not re.fullmatch("[0-9a-f]{40}", config.get("base_revision") or "")
        or not re.fullmatch("[0-9a-f]{64}", config.get("development_pool_hash") or "")
    ):
        raise ValueError("production requires enable flag, pinned revision and pool hash")
    if config.get("require_separate_principal", False):
        uid = config.get("sandbox", {}).get("uid")
        if uid is None or uid <= 0 or uid == os.getuid() or os.geteuid() != 0:
            raise ValueError("separate principal unavailable")


def validate_config(config, weights, pool):
    validate_pool(pool)
    allowed = config.get("editable_tensors")
    if (
        not isinstance(allowed, list)
        or not allowed
        or len(set(allowed)) != len(allowed)
        or any(k not in weights for k in allowed)
    ):
        raise ValueError("explicit nonempty editable_tensors allowlist required")
    if not re.fullmatch("[0-9a-f]{40}", config.get("base_revision") or ""):
        raise ValueError("resolved base revision required")
    if config.get("budget") not in (10, 25, 50):
        raise ValueError("choose one budget from 10/25/50; budgets is a treatment menu")
    if (
        not re.fullmatch("[0-9a-f]{64}", config.get("development_pool_hash") or "")
        or digest(pool) != config["development_pool_hash"]
    ):
        raise ValueError("pinned development pool hash required and must match")
    if config.get("production_enabled"):
        validate_production(config)


def load_config(path, weights, pool):
    """Load and qualify a single treatment before constructing any provider."""
    import yaml

    with open(path) as stream:
        config = yaml.safe_load(stream)
    validate_config(config, weights, pool)
    return config


def run_episode(provider, config, weights, model, pool, episode, zoo_dir, *, proposals=1):
    validate_config(config, weights, pool)
    from wsp.editor.providers import ScriptedEditor

    if type(provider) is not ScriptedEditor:
        validate_production(config)
    if not provider.model or not provider.version:
        raise ValueError("provider model/version required")
    limits = config.get("episode_limits", {})
    meter = EpisodeMeter(**{"max_" + k: v for k, v in limits.items()})
    results = []
    for index in range(proposals):
        allowed = config["editable_tensors"]
        api = EditorAPI(
            {k: weights[k] for k in allowed},
            model,
            config["budget"],
            hidden_layers=config.get("hidden_layers", ()),
            max_element=config.get("max_element", 1.0),
            max_norm=config.get("max_norm", 10.0),
            meter=meter,
            **config.get("probe", {}),
        )
        prompt = GOAL + "\n" + str(api.list_tensors()) + "\n" + str(pool)
        start = time.monotonic()
        episode.append("prompt", proposal=index, prompt=prompt, budget=api.limit)
        try:
            completion = provider.complete(prompt)
            episode.append(
                "completion",
                proposal=index,
                **asdict(completion),
                wall_seconds=time.monotonic() - start,
            )
            sandbox = dict(config.get("sandbox", {}))
            sandbox["require_separate_principal"] = config.get("require_separate_principal", False)
            result = run(completion.code, api, **sandbox)
            episode.append(
                "sandbox",
                proposal=index,
                **asdict(result),
                calls=api.calls,
                meter=meter.totals(),
                budget={"limit": api.limit, "used": api.used, "remaining": api.remaining},
            )
            if not result.valid or not api.deltas:
                raise ValueError(result.error or "empty or failed proposal")
            provenance = {
                "provider": type(provider).__name__,
                "model": provider.model,
                "version": provider.version,
                "prompt_template_hash": digest(GOAL),
                "episode_id": episode.path.parent.name,
                "proposal_id": index,
                "budget": api.limit,
                "meter_totals": api.used,
                "api_version": "editor_api.v1",
                "sandbox_version": "seccomp-v1",
                "code_hash": digest(completion.code),
                "transcript_hash": digest(result.transcript),
                "wall_seconds": time.monotonic() - start,
                "tokens": {"input": completion.input_tokens, "output": completion.output_tokens},
            }
            entry = register(api, config, provenance, zoo_dir)
            episode.append("registered", proposal=index, entry_id=entry)
            results.append(entry)
        except Exception as exc:  # noqa: BLE001 - retain every failed attempt
            episode.append(
                "failed",
                proposal=index,
                error=type(exc).__name__ + ": " + str(exc),
                used=api.used,
                wall_seconds=time.monotonic() - start,
            )
            results.append(None)
    return results
