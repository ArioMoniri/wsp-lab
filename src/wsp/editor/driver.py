"""Provider-neutral base-relative proposal episodes; no sealed evaluator imports."""

import time
from dataclasses import asdict

from wsp.editor.api import EditorAPI
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


def run_episode(provider, config, weights, model, pool, episode, zoo_dir, *, proposals=1):
    validate_pool(pool)
    if not provider.model or not provider.version:
        raise ValueError("provider model/version required")
    results = []
    for index in range(proposals):
        allowed = config.get("editable_tensors", list(weights))
        api = EditorAPI(
            {k: weights[k] for k in allowed},
            model,
            config["budget"],
            hidden_layers=config.get("hidden_layers", ()),
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
            result = run(completion.code, api, **config.get("sandbox", {}))
            episode.append(
                "sandbox",
                proposal=index,
                **asdict(result),
                calls=api.calls,
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
                error=type(exc).__name__,
                used=api.used,
                wall_seconds=time.monotonic() - start,
            )
            results.append(None)
    return results
