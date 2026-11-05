"""One-load, resumable zoo evaluation: python -m wsp.eval.batch."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml

import wsp
from wsp.eval.metadata import item_hashes, runtime_metadata
from wsp.eval.run import HFRunner, apply_zoo_entry, evaluate, write_results
from wsp.eval.zoo_compat import load_entry_delta

REPO = Path(wsp.__file__).resolve().parents[2]


def provenance():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()

    digest = hashlib.sha256()
    for path in sorted((REPO / "src/wsp").rglob("*.py")):
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "dirty": bool(
            git("status", "--porcelain", "--", "src", "configs", "tests", "pyproject.toml")
        ),
        "source_hash": digest.hexdigest(),
    }


def effective_config_hash(cfg):
    # Match HFRunner's default, including configs that omit batch_size.
    cfg = cfg | {"generation": {"batch_size": 8} | cfg["generation"]}
    # Reporting-only rename preserves the historical production resume identity.
    cfg["benchmarks"] = {k: dict(v) for k, v in cfg["benchmarks"].items()}
    for name, metric, judges in (
        ("refusal_rate", "refusal_keyword_rate", ["keyword", "llm"]),
        ("over_refusal", "over_refusal_keyword_rate", ["keyword"]),
    ):
        spec = cfg["benchmarks"].get(name, {})
        if spec.get("metric") == metric and spec.get("export_judge") is False:
            spec.pop("metric")
            spec.pop("export_judge")
            spec["judge"] = judges
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def resume_key(edit_id, config_hash, model_revision, reuse_key=None):
    return {
        "edit_id": edit_id,
        "config_hash": config_hash,
        "model_revision": model_revision,
        "reuse_key": reuse_key,
    }


def payload_key(payload):
    if "skip_key" in payload:
        return payload["skip_key"]
    return resume_key(
        payload.get("edit_id"),
        effective_config_hash(payload["effective_config"])
        if "effective_config" in payload
        else payload.get("config_hash"),
        payload.get("model_revision"),
        payload.get("reuse_key"),
    )


def upsert_index(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        keyed = {json.dumps(payload_key(r), sort_keys=True): r for r in rows}
        # Replace the pre-migration row when repairing an index from eval.json.
        rows_key = json.dumps(payload_key(payload), sort_keys=True)
        for key, row in list(keyed.items()):
            if "skip_key" not in row and payload_key(row) | {
                "reuse_key": payload_key(payload)["reuse_key"]
            } == payload_key(payload):
                del keyed[key]
        keyed[rows_key] = payload
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in keyed.values()))
        tmp.replace(path)


def complete(payload, signature, benchmarks):
    if isinstance(signature, dict):
        actual = payload_key(payload)
        # Old batch results predate reuse_key recording. Bind them to the current
        # zoo entry on migration; subsequent results require an exact key match.
        if "skip_key" not in payload and "signature" in payload and "reuse_key" not in payload:
            actual["reuse_key"] = signature["reuse_key"]
        matches = actual == signature
    else:
        matches = payload.get("signature") == signature
    return (
        payload.get("eval_status") == "complete"
        and matches
        and all(
            payload.get("counts", {}).get(k) == v.get("n", v.get("n_windows"))
            for k, v in benchmarks.items()
        )
    )


def subset_ids(rows, spec):
    rng = random.Random(spec["seed"])
    selected = []
    for family, n in spec["families"].items():
        pool = sorted(
            r["id"]
            for r in rows
            if r.get("family") == family and r.get("eligibility") == "production"
        )
        selected.extend(rng.sample(pool, min(n, len(pool))))
    return selected


class WeightRestore:
    def __init__(self, model):
        self.model = model
        self.base = {}

    def capture(self, keys):
        state = self.model.state_dict()
        for key in keys:
            if key not in self.base:
                self.base[key] = state[key].detach().cpu().clone()

    def restore(self):
        with torch.no_grad():
            state = self.model.state_dict()
            for key, value in self.base.items():
                state[key].copy_(value)

    @contextmanager
    def base_context(self):
        state = self.model.state_dict()
        edited = {k: state[k].detach().cpu().clone() for k in self.base}
        self.restore()
        try:
            yield
        finally:
            with torch.no_grad():
                for key, value in edited.items():
                    state[key].copy_(value)


def select_entries(
    rows,
    zoo,
    ids=None,
    families=None,
    eligibility="production",
    signature=None,
    benchmarks=None,
    index=None,
    force=False,
):
    for row in rows:
        if "id" not in row or row.get("eligibility") != eligibility:
            continue
        if ids is not None and row["id"] not in ids:
            continue
        if families and row["family"] not in families:
            continue
        result = zoo / row["id"] / "eval.json"
        if result.exists():
            previous = json.loads(result.read_text())
            key = (
                resume_key(
                    row["id"],
                    signature["config_hash"],
                    signature["model_revision"],
                    row.get("reuse_key"),
                )
                if isinstance(signature, dict)
                else signature
            )
            done = (
                complete(previous, key, benchmarks or {})
                if key is not None
                else previous.get("eval_status") == "complete"
            )
            if done and isinstance(key, dict):
                previous = previous | {"skip_key": key}
            if index is not None:
                upsert_index(index, previous | {"path": str(result.parent)})
            if done and not force:
                continue
        yield row


def retry_oom(run, batch):
    for attempt in range(3):
        try:
            return run(batch), batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt < 2:
                batch = max(1, batch // 2)
    return None, batch


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def base_destination(out, expected, run_id, no_base=False, force=False):
    """Reuse only known compatible bases; preserve every prior base artifact."""
    from wsp.eval.repeats import REFERENCE_KEYS, known

    root = Path(out) / "base"
    candidates = [root / "eval.json", *sorted(root.glob("devices/*/*/eval.json"))]
    for file in reversed(candidates):
        if not file.exists():
            continue
        payload = json.loads(file.read_text())
        if (
            not force
            and all(known(payload.get(k)) for k in REFERENCE_KEYS)
            and all(payload.get(k) == v for k, v in expected.items())
            and payload.get("eval_status") == "complete"
        ):
            return file.parent, payload
    if no_base:
        print("missing matched reference: --no-base; unmatched targets remain missing", flush=True)
        return None, None
    if (root / "eval.json").exists():
        return root / "devices" / expected["eval_device_id"] / run_id, None
    return root, None


def run_batch(
    zoo,
    config,
    ids=None,
    families=None,
    eligibility="production",
    max_new_tokens=None,
    batch_size=None,
    limit=None,
    subset=None,
    force=False,
    repeat=None,
    no_base=False,
):
    zoo, config = Path(zoo), Path(config)
    cfg = yaml.safe_load(config.read_bytes())
    generation = {"batch_size": 8} | cfg["generation"]
    if max_new_tokens is not None:
        raise ValueError("stopping settings must come from the config")
    if batch_size is not None:
        generation["batch_size"] = batch_size
    if min(generation["batch_size"], generation["max_new_tokens"]) < 1:
        raise ValueError("batch size and token budget must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if limit:
        for spec in cfg["benchmarks"].values():
            spec["n"] = spec["n_windows"] = limit
    rows = [json.loads(line) for line in (zoo / "manifest.jsonl").read_text().splitlines()]
    cfg["generation"] = generation.copy()
    config_hash = effective_config_hash(cfg)
    origin = provenance()
    header = next(r for r in reversed(rows) if r.get("type") == "header")
    revision = header["base_revision"]
    if len(revision) != 40:
        from wsp.zoo.common import resolve_base_revision

        revision = resolve_base_revision(header["base_model"], revision)
    signature = resume_key("base", config_hash, revision)
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    production = (
        limit is None
        and max_new_tokens is None
        and batch_size is None
        and eligibility == "production"
        and config.resolve() == (REPO / "configs/eval_fast.yaml").resolve()
    )
    out = REPO / "data/evals" / zoo.name if production else REPO / "data/runs/p2-batch" / run_id
    if generation.get("tier") == "tf_v1":
        out = REPO / "data/evals" / zoo.name / "tf_v1"
    if subset:
        spec = yaml.safe_load((REPO / "configs/eval_subset_d3.yaml").read_text())
        chosen = subset_ids(rows, spec)
        ids = [i for i in chosen if not ids or i in ids]
    print(
        f"excluded legacy rows without eligibility: {sum('id' in r and 'eligibility' not in r for r in rows)}"
    )
    if repeat is not None:
        if (
            not repeat
            or repeat in (".", "..")
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in repeat
            )
        ):
            raise ValueError("repeat label must be a simple directory name")
        force = True
    selected = list(
        select_entries(
            rows,
            zoo if production else out,
            ids,
            families,
            eligibility,
            signature,
            cfg["benchmarks"],
            None if repeat else out / "index.jsonl",
            force=force,
        )
    )
    load_started = time.monotonic()
    runner = HFRunner(header["base_model"], generation.copy(), revision=revision)
    load_seconds = time.monotonic() - load_started
    if generation.get("tier") == "tf_v1":
        runner.tf_config = cfg
        runner.tf_cache_root = REPO / "data/evals" / zoo.name / "base/tf_cache"
        runner.tf_stop_dir = out / "stops"
    expected = runtime_metadata(runner) | {
        "config_hash": config_hash,
        "model_revision": revision,
        "source_hash": origin["source_hash"],
        "generation": generation,
        "batch_size": generation["batch_size"],
    }
    base_out, reference = base_destination(out, expected, run_id, no_base, force)
    if reference is not None and repeat is None:
        upsert_index(out / "index.jsonl", reference | {"path": str(base_out)})
    base_done = reference is not None or no_base
    if repeat is not None and not no_base:
        base_out = out / "base" / "repeats" / repeat
        base_done = False
    if not selected and base_done:
        return
    restore = WeightRestore(runner.model)
    runner.base_context = restore.base_context
    out.mkdir(parents=True, exist_ok=True)
    jobs = ([] if base_done else [(None, base_out)]) + [
        (r, zoo / r["id"] if production else out / r["id"]) for r in selected
    ]
    if repeat is not None:
        jobs = [(row, dest / "repeats" / repeat if row else dest) for row, dest in jobs]
        # Reserve every destination before execution; an existing label is immutable.
        for _, dest in jobs:
            dest.mkdir(parents=True, exist_ok=False)
    for row, dest in jobs:
        entry_started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        restore.restore()
        runner.edit_id = row["id"] if row else "base"
        runner.rounding_stats = {}
        source = zoo / row["id"] if row else None
        if row:
            edit = json.loads((source / "edit.json").read_text())
            if edit["base_revision"] != runner.revision:
                raise ValueError("entry base revision differs from loaded snapshot")
            delta = load_entry_delta(source, runner.model.state_dict().keys())
            restore.capture(delta)
            del delta
            runner.rounding_stats = apply_zoo_entry(runner.model, source)
        print(
            f"entry={runner.edit_id} gpu_allocated={torch.cuda.memory_allocated()} "
            f"gpu_reserved={torch.cuda.memory_reserved()}",
            flush=True,
        )
        started = time.monotonic()

        def execute(batch):
            runner.generation["batch_size"] = batch
            return evaluate(runner, list(cfg["benchmarks"]), cfg["benchmarks"])

        result, used_batch = retry_oom(execute, generation["batch_size"])
        key = resume_key(
            runner.edit_id, config_hash, runner.revision, row.get("reuse_key") if row else None
        )
        manifest = {
            "skip_key": key,
            "signature": hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest(),
            "reuse_key": key["reuse_key"],
            "seed": generation["seed"],
            "config_hash": config_hash,
            **origin,
            "eligibility": row.get("eligibility") if row else eligibility,
            "family": row.get("family") if row else "base",
            "parent_id": row.get("parent_id") if row else None,
            "step_semantics": row.get("step_semantics") if row else None,
            "effective_config": cfg,
            "model": runner.model_id,
            "model_revision": runner.revision,
            "run_id": run_id,
            "edit_id": runner.edit_id,
            "max_new_tokens": generation["max_new_tokens"],
            "rounding_stats": runner.rounding_stats,
            "generation": dict(runner.generation),
            "batch_size": used_batch,
            "batch_size_nominal": generation["batch_size"],
            "repeat_id": repeat,
            **runtime_metadata(runner),
            "limit": limit,
            "config": str(config),
            "minutes": (time.monotonic() - started) / 60,
            "wall_seconds": time.monotonic() - entry_started,
            "model_load_seconds": load_seconds,
            "stage_seconds": getattr(runner, "tf_stage_seconds", {}),
            **(
                {
                    "tier": "tf_v1",
                    "panel": cfg["tf_v1"]["panel"],
                    "tf_cache": getattr(runner, "tf_cache_info", None),
                }
                if generation.get("tier") == "tf_v1"
                else {}
            ),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "metadata_provenance": "native-measured",
            "eval_status": "complete" if result is not None else "oom",
        }
        dest.mkdir(parents=True, exist_ok=True)
        summary = {}
        if result is not None:
            records, datasets = result
            summary = write_results(
                dest, records, dict(runner.generation), manifest | {"datasets": datasets}
            )
        else:
            from wsp.schema import write_manifest

            write_manifest(dest / "manifest.json", **manifest)
        if result is not None and not complete(summary | manifest, key, cfg["benchmarks"]):
            manifest["eval_status"] = "incomplete"
        payload = summary | manifest
        if result is not None:
            payload.update(item_hashes(records))
        from wsp.eval.repeats import matched_reference

        if row is None and result is not None:
            reference = payload.copy()
        matched = matched_reference(payload, [reference] if reference else [])
        payload["base_run_id"] = matched.get("run_id") if matched else None
        payload["matched_base_missing"] = matched is None
        payload["entry_hashes"] = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                [source / "delta.safetensors"]
                if source and (source / "delta.safetensors").exists()
                else sorted((source / "lora").glob("*"))
                if source
                else []
            )
            if p.is_file()
        }
        atomic_json(dest / "eval.json", payload)
        index_path = out / "repeats" / repeat / "index.jsonl" if repeat else out / "index.jsonl"
        upsert_index(index_path, payload | {"path": str(dest)})
        torch.cuda.empty_cache()
        print(json.dumps(manifest), flush=True)
    restore.restore()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--no-base", action="store_true", help="leave unmatched targets missing")
    parser.add_argument("--repeat", help="immutable independent repeat label")
    parser.add_argument("--zoo", required=True)
    parser.add_argument("--config", default="configs/eval_fast.yaml")
    parser.add_argument("--subset", choices=["d3"])
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--eligibility", default="production", choices=["production", "smoke"])
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, help="tiny GPU smoke items per benchmark")
    parser.add_argument(
        "--resume", action="store_true", help="completed entries are always skipped"
    )
    parser.add_argument("--force", action="store_true", help="re-evaluate completed entries")
    args = vars(parser.parse_args())
    args.pop("resume")
    for key in ("ids", "families"):
        if args[key]:
            args[key] = [v for group in args[key] for v in group.split(",")]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run_batch(**args)


if __name__ == "__main__":
    main()
