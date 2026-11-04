#!/usr/bin/env python3
"""Selective sync of zoo artefacts to the private HF dataset (D7).

python scripts/hf_sync.py metadata   # manifest + every *.json / *.jsonl under data/zoo (small)
python scripts/hf_sync.py deltas     # delta.safetensors / lora/ for non-chain production families
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
REPO = "Aralario/wsp-lab-zoo"


def families() -> dict[str, str]:
    out = {}
    for line in (ZOO / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "id" in r and "family" in r:
            out[r["id"]] = r["family"]
    return out


def main(kind: str) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if kind == "metadata":
        patterns = ["manifest.jsonl", "*/*.json", "*/lora/*.json", "*/**/*.jsonl"]
    elif kind == "deltas":
        ids = [i for i, f in families().items() if f != "chain"]
        patterns = [f"{i}/delta.safetensors" for i in ids] + [f"{i}/lora/*" for i in ids]
        print(f"{len(ids)} non-chain entries")
    else:
        raise SystemExit("kind must be metadata|deltas")
    api.upload_large_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=str(ZOO),
        allow_patterns=patterns,
        print_report=False,
    ) if kind == "deltas" else api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=str(ZOO),
        path_in_repo="zoo/qwen3.5-0.8b",
        allow_patterns=patterns,
        commit_message=f"sync {kind}",
    )
    print("done", kind)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "metadata")
