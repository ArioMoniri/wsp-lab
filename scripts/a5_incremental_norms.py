"""A5.3.1 input: predecessor-relative (incremental) effective-delta norms for every zoo entry.

For a root entry the increment is its effective delta; for a chain step it is the realised difference between the two
stored states, ``bf16(base + delta_succ) - bf16(base + delta_pred)``, subtracted in fp32 (the realised-increment
convention of SYNTHESIS 7 Nov §4.1 and A5.3.3). Base-relative norms (``edit.json:delta_fro_norm``) are reported next to
them. Output: one JSON line per entry with ``id, family, parent_id, stored, inc_fro, base_fro, n_tensors, exactness``
where ``exactness`` is ``bitwise``, ``re-trained`` (the entry or an ancestor is a re-trained LoRA replay) or ``lost``.

    a5_incremental_norms.py --zoo data/zoo/qwen3.5-0.8b --sidecar <artefact_status.jsonl> --out <jsonl> [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from wsp.zoo.apply import load_entry_delta


def _inc_norm(args: tuple[str, str | None, str]) -> dict:
    entry_id, parent_id, zoo_s = args
    zoo = Path(zoo_s)
    edit = json.loads((zoo / entry_id / "edit.json").read_text())
    out = {
        "id": entry_id,
        "family": edit["family"],
        "parent_id": parent_id,
        "base_fro": edit.get("delta_fro_norm"),
    }
    try:
        d_succ = load_entry_delta(zoo / entry_id)
    except Exception as e:  # noqa: BLE001  # lost bytes or unreadable adapter: reported, never fabricated
        out.update({"stored": False, "inc_fro": None, "n_tensors": 0, "error": str(e)[:120]})
        return out
    d_succ = dict(d_succ)
    if parent_id is None:
        sq = sum(float(v.float().pow(2).sum()) for v in d_succ.values())
        out.update({"stored": True, "inc_fro": math.sqrt(sq), "n_tensors": len(d_succ)})
        return out
    try:
        d_pred = load_entry_delta(zoo / parent_id)
    except Exception as e:  # noqa: BLE001
        out.update(
            {
                "stored": True,
                "inc_fro": None,
                "n_tensors": len(d_succ),
                "error": f"predecessor: {str(e)[:100]}",
            }
        )
        return out
    d_pred = dict(d_pred)
    # realised increment: both stored states are base + delta in bf16; their difference in fp32 equals
    # bf16(delta_succ) - bf16(delta_pred) because the base cancels exactly for tensors present in both.
    sq = 0.0
    n = 0
    for k in set(d_succ) | set(d_pred):
        a = d_succ.get(k)
        b = d_pred.get(k)
        if a is None:
            diff = -b.to(torch.bfloat16).float()
        elif b is None:
            diff = a.to(torch.bfloat16).float()
        else:
            diff = a.to(torch.bfloat16).float() - b.to(torch.bfloat16).float()
        s = float(diff.pow(2).sum())
        if s > 0:
            n += 1
        sq += s
    out.update({"stored": True, "inc_fro": math.sqrt(sq), "n_tensors": n})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--zoo", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    side = {
        json.loads(line)["id"]: json.loads(line)
        for line in a.sidecar.read_text().splitlines()
        if line.strip()
    }
    rows = [
        json.loads(line)
        for line in (a.zoo / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r.get("type") != "header"]
    parent = {r["id"]: r.get("parent_id") for r in rows}
    exact: dict[str, str] = {}

    def exactness(i: str) -> str:
        if i in exact:
            return exact[i]
        s = side.get(i, {})
        if not s.get("delta_present") and not s.get("alias_of"):
            e = "lost"
        else:
            own = (
                "re-trained"
                if (s.get("regenerated") or {}).get("exactness", "").startswith("re-trained")
                else "bitwise"
            )
            p = parent.get(i)
            up = exactness(p) if p else "bitwise"
            e = (
                "lost"
                if up == "lost"
                else ("re-trained" if "re-trained" in (own, up) else "bitwise")
            )
        exact[i] = e
        return e

    def resolved(i: str) -> str:
        s = side.get(i, {})
        return s["alias_of"] if (not s.get("delta_present") and s.get("alias_of")) else i

    jobs = [
        (resolved(r["id"]), resolved(parent[r["id"]]) if parent[r["id"]] else None, str(a.zoo))
        for r in rows
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with ProcessPoolExecutor(a.workers) as ex, a.out.open("w") as f:
        for r, res in zip(rows, ex.map(_inc_norm, jobs, chunksize=8)):
            res["id"] = r["id"]
            res["parent_id"] = parent[r["id"]]
            res["exactness"] = exactness(r["id"])
            f.write(json.dumps(res, sort_keys=True) + "\n")
    n = sum(1 for _ in a.out.open())
    print(f"wrote {n} rows to {a.out}")


if __name__ == "__main__":
    main()
