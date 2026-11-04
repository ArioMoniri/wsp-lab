#!/usr/bin/env python3
"""Selective sync of zoo artefacts to the HF dataset (D7; public since 4 Nov 2026, D7 addendum 2).

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
        # One commit per batch (HF allows 128 commits/hour); each batch ≈ 40 entries ≤ 8 GB.
        ids = [i for i, f in families().items() if f != "chain"]
        print(f"{len(ids)} non-chain entries")
        done_file = ZOO / ".hf_deltas_done"
        done = set(done_file.read_text().split()) if done_file.exists() else set()
        todo = [i for i in ids if i not in done]
        for k in range(0, len(todo), 40):
            batch = todo[k : k + 40]
            patterns = [f"{i}/delta.safetensors" for i in batch] + [f"{i}/lora/*" for i in batch]
            api.upload_folder(
                repo_id=REPO,
                repo_type="dataset",
                folder_path=str(ZOO),
                path_in_repo="zoo/qwen3.5-0.8b",
                allow_patterns=patterns,
                commit_message=f"deltas batch {k // 40}",
            )
            done.update(batch)
            done_file.write_text("\n".join(sorted(done)))
            print(f"batch {k // 40}: {len(batch)} entries committed ({len(done)}/{len(ids)})")
        print("done", kind)
        return
    else:
        raise SystemExit("kind must be metadata|deltas")
    api.upload_folder(
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
