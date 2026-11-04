#!/usr/bin/env python3
"""Pull Lambda eval shards back from the public HF zoo store into the local zoo.

python scripts/d3_pull_shards.py data/runs/d3_shards/lambda0.txt [more id files]

For each id: downloads zoo/<id>/{eval.json,summary.json,records.jsonl} and places them in the local entry dir only
when the local entry has no eval.json yet; entries already evaluated locally keep their local result (cross-host
controls are stored next to it as eval.<host>.json so the two hosts can be compared). Every pulled eval.json gets an
`eval_host` field (the base evaluation of that host lands in data/evals/qwen3.5-0.8b/base-<host>/).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
REPO = "Aralario/wsp-lab-zoo"


def main(id_files: list[str]) -> None:
    ids = [l.strip() for f in id_files for l in Path(f).read_text().splitlines() if l.strip()]
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    hosts = sorted(
        {
            s.rfilename.split("/")[2]
            for s in api.repo_info(REPO, repo_type="dataset").siblings
            if s.rfilename.startswith("evals/qwen3.5-0.8b/base-lambda")
        }
    )
    local = snapshot_download(
        REPO,
        repo_type="dataset",
        token=token,
        allow_patterns=[f"zoo/qwen3.5-0.8b/{i}/eval.json" for i in ids]
        + [f"zoo/qwen3.5-0.8b/{i}/summary.json" for i in ids]
        + [f"zoo/qwen3.5-0.8b/{i}/records.jsonl" for i in ids]
        + [f"evals/qwen3.5-0.8b/{h}/*" for h in hosts],
    )
    local = Path(local)
    for h in hosts:
        dest = ROOT / "data/evals/qwen3.5-0.8b" / h
        if (local / "evals/qwen3.5-0.8b" / h).exists():
            shutil.copytree(local / "evals/qwen3.5-0.8b" / h, dest, dirs_exist_ok=True)
    placed = kept = missing = 0
    for i in ids:
        src = local / "zoo/qwen3.5-0.8b" / i
        if not (src / "eval.json").exists():
            missing += 1
            continue
        payload = json.loads((src / "eval.json").read_text())
        host = payload.get("eval_host") or _host_from_commit(api, f"zoo/qwen3.5-0.8b/{i}/eval.json")
        payload["eval_host"] = host
        dst = ZOO / i
        if (dst / "eval.json").exists():
            (dst / f"eval.{host}.json").write_text(json.dumps(payload, indent=1))
            kept += 1
        else:
            for f in ("summary.json", "records.jsonl"):
                if (src / f).exists():
                    shutil.copy2(src / f, dst / f)
            (dst / "eval.json").write_text(json.dumps(payload, indent=1))
            placed += 1
    print(
        f"placed={placed} kept_local(control copies)={kept} not_yet_on_hub={missing} hosts={hosts}"
    )


def _host_from_commit(api: HfApi, path: str) -> str:
    for c in api.list_repo_commits(REPO, repo_type="dataset"):
        if c.title.startswith("eval shard "):
            return re.sub(r" \(part \d+\)$", "", c.title.split("eval shard ", 1)[1].strip())
    return "lambda-unknown"


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/runs/d3_shards/lambda0.txt", "data/runs/d3_shards/lambda1.txt"])
