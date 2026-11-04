#!/usr/bin/env python3
"""Explicit integration runs, excluded from pytest. Source scripts/env.sh first."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

from wsp.editor.api import EditorAPI, EpisodeMeter, ModelSpec
from wsp.editor.driver import qualify_pools
from wsp.editor.episode import digest


def prepare_pools():
    from datasets import load_dataset

    from wsp.eval.subsets import REVISIONS, SPECS, load_items

    train = load_dataset("openai/gsm8k", "main", split="train", revision=REVISIONS["gsm8k"])
    development = [
        {"id": f"gsm8k:train:{i}", "source": "gsm8k", "split": "train", "prompt": x["question"]}
        for i, x in enumerate(train.select(range(400)))
    ]
    sealed = []
    for name in SPECS:
        items, _ = load_items(name)
        for i, item in items:
            prompt = next(item[k] for k in ("question", "problem", "prompt", "ctx") if k in item)
            sealed.append({"id": f"{name}:{SPECS[name][-1]}:{i}", "prompt": prompt})
    hashes = qualify_pools(development, sealed)
    root = Path("data/editor/pools")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, pool in [("development", development), ("sealed", sealed)]:
        path = root / f"{name}.json"
        path.write_text(json.dumps(pool, sort_keys=True))
        path.chmod(0o600)
    (root / "qualification.json").write_text(json.dumps(hashes, sort_keys=True))
    manifest = {
        "seed": 0,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_revisions": REVISIONS,
        "pools": hashes,
        "config_hash": digest({"development": "gsm8k-train-first-400", "sealed": SPECS}),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    print(json.dumps(hashes, sort_keys=True))


def smoke(config, device="cuda"):
    jobs = subprocess.check_output(["agent-tmux", "-l"], text=True)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    print(jobs, flush=True)
    if device == "cuda" and gpu.strip():
        raise SystemExit("GPU busy; real-model smoke refused")
    spec = ModelSpec(
        config["base_model"], config["base_revision"], device, config["probe"]["kernels"]
    )
    api = EditorAPI(
        {},
        spec,
        10,
        meter=EpisodeMeter(max_wall_seconds=600),
        **{**config["probe"], "device": device, "call_seconds": 600},
    )
    start = time.monotonic()
    result = api.forward([[1, 2, 3, 4]] * 4, top_k=4)
    elapsed = time.monotonic() - start
    assert len(result["token_ids"]) == 4 and api.used == 4
    print(
        json.dumps(
            {
                "device": device,
                "kernels": spec.kernels,
                "sequences": 4,
                "load_and_probe_seconds": elapsed,
                "meter": api.meter.totals(),
                "worker_timing": api.calls[-1],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-pools", action="store_true")
    group.add_argument("--smoke-real-model", action="store_true")
    parser.add_argument("--config", default="configs/editor_qwen3.5-0.8b.yaml")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.prepare_pools:
        prepare_pools()
    else:
        smoke(yaml.safe_load(Path(args.config).read_text()), args.device)


if __name__ == "__main__":
    main()
