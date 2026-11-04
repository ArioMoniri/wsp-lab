"""Backfill existing evals without changing scores (dry run unless --apply).

--runtime-json accepts an opaque-device keyed object of runtime fields transcribed
from job logs. Unknown remote fields remain null; no local runtime is borrowed.
Control copies require records.<copy-label>.jsonl, or the same run as eval.json.
"""

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from wsp.eval.metadata import item_hashes, runtime_metadata
from wsp.eval.repeats import REFERENCE_KEYS, known

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


def backfill(path, payload, devices, local, remote, now, host=None, historical_items=None):
    out = payload.copy()
    label = out.pop("eval_host", host or "local")
    device = (
        payload.get("eval_device_id") or devices[label]
    )  # Unknown hosts must be mapped explicitly, never guessed.
    out["eval_device_id"] = device
    runtime = local if device == devices.get("local") else remote.get(device, {})
    local_inferred = (
        set(payload.get("metadata_backfill_inferred", []))
        if device == devices.get("local")
        else set()
    )
    if device == devices.get("local"):
        local_inferred.update(
            k
            for k in RUNTIME_KEYS
            if payload.get(k) is None
            or ("metadata_backfill" in payload and payload.get("metadata_backfill_schema", 1) < 2)
        )
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
    hashes = {key: payload.get(key) for key in ("item_id_hash", "item_order_hash")}
    if records.exists():
        rows = [
            SimpleNamespace(**json.loads(line))
            for line in records.read_text().splitlines()
            if line.strip()
        ]
        if rows:
            hashes = item_hashes(rows)
    evidence = (historical_items or {}).get(payload.get("run_id"))
    reconstructed = False
    if not records.exists() and evidence:
        from collections import Counter

        if evidence["source_hash"] != payload.get("source_hash"):
            raise ValueError("historical item evidence source mismatch")
        if dict(Counter(b for b, _ in evidence["items"])) != payload.get("counts"):
            raise ValueError("historical item evidence count mismatch")
        hashes = item_hashes(
            [SimpleNamespace(benchmark=b, item_id=i) for b, i in evidence["items"]]
        )
        out["metadata_backfill_item_source"] = evidence["source"]
        reconstructed = True
    out.update(hashes)
    provenance = dict(payload.get("metadata_backfill_provenance", {}))
    for key in REFERENCE_KEYS:
        if key in runtime and (payload.get(key) is None or key in runtime.get("provenance", {})):
            provenance[key] = runtime.get("provenance", {}).get(key, "transcribed")
        elif known(out.get(key)):
            provenance.setdefault(key, "transcribed")
    if reconstructed:
        for key in hashes:
            provenance[key] = "inferred"
    for key in local_inferred:
        provenance[key] = "inferred"
    if local_inferred:
        out["metadata_backfill_source"] = "current environment, not contemporaneous"
    missing = [key for key in REFERENCE_KEYS if not known(out.get(key))]
    inferred = sorted(k for k in REFERENCE_KEYS if provenance.get(k) == "inferred")
    out["metadata_backfill_schema"] = 2
    out["metadata_backfill_provenance"] = provenance
    out["metadata_backfill_inferred"] = inferred
    out["metadata_backfill"] = "partial-inferred" if inferred else "partial" if missing else "full"
    out["metadata_backfill_missing"] = missing
    out["metadata_backfill_utc"] = payload.get("metadata_backfill_utc", now)
    return out


def archive(path):
    original = path.with_name(path.name + ".orig")
    if path.exists() and not original.exists():
        shutil.copy2(path, original)


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
    parser.add_argument("--historical-items-json", type=Path)
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
    historical = (
        json.loads(args.historical_items_json.read_text()) if args.historical_items_json else {}
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
        new = backfill(path, old, devices, local, remote, now, host, historical)
        results[path.resolve()] = new
        partial += new.get("metadata_backfill") != "full"
        if args.apply:
            archive(path)
        if new != old:
            changed += 1
            if args.apply:
                write(path, new)
        if host or (path.name != "eval.json" and "devices" not in path.parts):
            destination = (
                args.evals
                / ("base" if host else path.parent.name)
                / "devices"
                / new["eval_device_id"]
                / new["run_id"]
            )
            if args.apply:
                target_eval = destination / "eval.json"
                archive(target_eval)
                if not target_eval.exists():
                    write(target_eval.with_name("eval.json.orig"), old)
                write(target_eval, new)
                for name in ("records.jsonl", "manifest.json"):
                    source = path.parent / name
                    if name == "records.jsonl" and path.name != "eval.json":
                        source = path.with_name(
                            path.name.replace("eval.", "records.").replace(".json", ".jsonl")
                        )
                        # Same run ID is the only safe fallback for item records.
                        if not source.exists() and (path.parent / "eval.json").exists():
                            main = json.loads((path.parent / "eval.json").read_text())
                            if main.get("run_id") == new.get("run_id"):
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
                        "metadata_backfill_inferred",
                        "metadata_backfill_provenance",
                        "metadata_backfill_source",
                        "metadata_backfill_schema",
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
