"""Backfill existing evals without changing scores (dry run unless --apply).

--runtime-json accepts an opaque-device keyed object of runtime fields transcribed
from job logs. Unknown remote fields remain null; no local runtime is borrowed.
Control copies require records.<copy-label>.jsonl, or the same run as eval.json.
"""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from wsp.eval.metadata import item_hashes, runtime_metadata

RUNTIME_KEYS = (
    "gpu_class",
    "engine",
    "torch_version",
    "cuda_version",
    "transformers_version",
    "dtype",
    "attention_implementation",
)


def default_map():
    # Infrastructure labels stay in the private mapping, never in output metadata.
    return {
        "local": "H0",
        **{
            "lambda-" + "-".join(map(str, address)): device
            for address, device in [((209, 20, 158, 168), "H1"), ((129, 153, 20, 232), "H2")]
        },
    }


def backfill(path, payload, devices, local, remote, now, host=None):
    out = payload.copy()
    label = out.pop("eval_host", host or "local")
    device = (
        payload.get("eval_device_id") or devices[label]
    )  # Unknown hosts must be mapped explicitly, never guessed.
    out["eval_device_id"] = device
    runtime = local if device == devices.get("local") else remote.get(device, {})
    for key in RUNTIME_KEYS:
        if out.get(key) is None:
            out[key] = runtime.get(key)
    out.setdefault(
        "batch_size_nominal",
        payload.get("effective_config", {})
        .get("generation", payload.get("generation", {}))
        .get("batch_size", 32),
    )
    records = path.parent / "records.jsonl"
    if path.name != "eval.json":
        records = path.with_name(path.name.replace("eval.", "records.").replace(".json", ".jsonl"))
        main = path.with_name("eval.json")
        if not records.exists() and main.exists():
            other = json.loads(main.read_text())
            if payload.get("run_id") and payload["run_id"] == other.get("run_id"):
                records = path.parent / "records.jsonl"
    hashes = {"item_id_hash": None, "item_order_hash": None}
    if records.exists():
        rows = [
            SimpleNamespace(**json.loads(line))
            for line in records.read_text().splitlines()
            if line.strip()
        ]
        if rows:
            hashes = item_hashes(rows)
    out.update(hashes)
    missing = [key for key in (*RUNTIME_KEYS, *hashes) if out.get(key) is None]
    out["metadata_backfill"] = "partial" if missing else "full"
    out["metadata_backfill_missing"] = missing
    out["metadata_backfill_utc"] = payload.get("metadata_backfill_utc", now)
    return out


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".backfill-tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evals", type=Path, default=Path("data/evals/qwen3.5-0.8b"))
    parser.add_argument("--zoo", type=Path, default=Path("data/zoo/qwen3.5-0.8b"))
    parser.add_argument("--device-map", type=Path, default=Path.home() / ".config/wsp/device_map")
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    devices = json.loads(args.device_map.read_text()) if args.device_map.exists() else default_map()
    if any(not re.fullmatch(r"H\d+", d) for d in devices.values()):
        raise ValueError("device map values must be opaque H<number> IDs")
    if args.apply and not args.device_map.exists():
        write(args.device_map, devices)
    local = runtime_metadata(
        SimpleNamespace(
            model=SimpleNamespace(
                dtype=torch.bfloat16, config=SimpleNamespace(_attn_implementation="sdpa")
            )
        )
    )
    remote = (
        json.loads(args.runtime_json.read_text())
        if args.runtime_json
        else {d: {"torch_version": "2.11.0+cu128"} for h, d in devices.items() if h != "local"}
    )
    now = datetime.now(UTC).isoformat()
    files = sorted(set(args.evals.rglob("eval*.json")) | set(args.zoo.glob("*/eval*.json")))
    changed = partial = 0
    results = {}
    for path in files:
        old = json.loads(path.read_text())
        host = (
            path.parent.name.removeprefix("base-") if path.parent.name.startswith("base-") else None
        )
        new = backfill(path, old, devices, local, remote, now, host)
        results[path.resolve()] = new
        partial += new.get("metadata_backfill") == "partial"
        if new != old:
            changed += 1
            if args.apply:
                write(path, new)
        if host:
            destination = args.evals / "base/devices" / new["eval_device_id"] / new["run_id"]
            if args.apply:
                write(destination / "eval.json", new)
                for name in ("records.jsonl", "manifest.json"):
                    source = path.parent / name
                    target = destination / name
                    if source.exists() and not target.exists():
                        target.symlink_to(source.resolve())
    # Expose evaluated zoo entries to the directory reader without copying data.
    for source in sorted(args.zoo.glob("*/eval.json")):
        target = args.evals / source.parent.name
        if args.apply and not target.exists():
            target.symlink_to(source.parent.resolve(), target_is_directory=True)
    # Keep index-only consumers consistent without altering ordering or score fields.
    index = args.evals / "index.jsonl"
    if index.exists():
        rows = []
        for line in index.read_text().splitlines():
            row = json.loads(line)
            path = Path(row.get("path", "")) / "eval.json"
            updated = results.get(path.resolve())
            if updated and updated.get("run_id") == row.get("run_id"):
                row = row | {
                    k: v
                    for k, v in updated.items()
                    if k
                    in {
                        *RUNTIME_KEYS,
                        "eval_device_id",
                        "batch_size_nominal",
                        "item_id_hash",
                        "item_order_hash",
                        "metadata_backfill",
                        "metadata_backfill_utc",
                        "metadata_backfill_missing",
                    }
                }
                row.pop("eval_host", None)
            rows.append(row)
        if args.apply:
            tmp = index.with_suffix(".backfill-tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
            tmp.replace(index)
    print(
        json.dumps(
            {"files": len(files), "changed": changed, "partial": partial, "applied": args.apply}
        )
    )


if __name__ == "__main__":
    main()
