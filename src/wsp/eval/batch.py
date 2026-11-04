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


def upsert_index(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        keyed = {(r["edit_id"], r["config_hash"]): r for r in rows}
        keyed[payload["edit_id"], payload["config_hash"]] = payload
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in keyed.values()))
        tmp.replace(path)


def complete(payload, signature, benchmarks):
    return (
        payload.get("eval_status") == "complete"
        and payload.get("signature") == signature
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
            if index is not None:
                upsert_index(index, previous | {"path": str(result.parent)})
            if benchmarks is not None and not complete(previous, signature, benchmarks):
                yield row
                continue
            if previous.get("eval_status") == "complete" and (
                signature is None or previous.get("signature") == signature
            ):
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
    out=None,
    skip_base=False,
    backend="hf",
):
    zoo, config = Path(zoo), Path(config)
    cfg = yaml.safe_load(config.read_bytes())
    generation = cfg["generation"].copy()
    if max_new_tokens is not None:
        generation["max_new_tokens"] = max_new_tokens
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
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    origin = provenance()
    signature = hashlib.sha256((config_hash + origin["source_hash"]).encode()).hexdigest()
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    production = (
        out is None
        and backend == "hf"
        and limit is None
        and max_new_tokens is None
        and batch_size is None
        and eligibility == "production"
        and config.resolve() == (REPO / "configs/eval_fast.yaml").resolve()
    )
    out = (
        Path(out)
        if out
        else (
            REPO / "data/evals" / zoo.name if production else REPO / "data/runs/p2-batch" / run_id
        )
    )
    signature = hashlib.sha256((signature + backend).encode()).hexdigest()
    if subset:
        spec = yaml.safe_load((REPO / "configs/eval_subset_d3.yaml").read_text())
        chosen = subset_ids(rows, spec)
        ids = [i for i in chosen if not ids or i in ids]
    print(
        f"excluded legacy rows without eligibility: {sum('id' in r and 'eligibility' not in r for r in rows)}"
    )
    selected = list(
        select_entries(
            rows,
            zoo if production else out,
            ids,
            families,
            eligibility,
            signature,
            cfg["benchmarks"],
            out / "index.jsonl",
        )
    )
    header = next(r for r in reversed(rows) if r.get("type") == "header")
    base_out = out / "base"
    base_done = False
    if (base_out / "eval.json").exists():
        previous = json.loads((base_out / "eval.json").read_text())
        upsert_index(out / "index.jsonl", previous | {"path": str(base_out)})
        base_done = complete(previous, signature, cfg["benchmarks"])
    if not selected and base_done:
        return
    runner_class = HFRunner
    if backend == "vllm":
        from wsp.eval.vllm_adapter import VLLMRunner

        runner_class = VLLMRunner
    runner = runner_class(header["base_model"], generation.copy(), revision=header["base_revision"])
    restore = WeightRestore(runner.model)
    runner.base_context = restore.base_context
    if hasattr(runner, "wrap_base_context"):
        runner.wrap_base_context()
    out.mkdir(parents=True, exist_ok=True)
    jobs = ([] if base_done or skip_base else [(None, base_out)]) + [
        (r, zoo / r["id"] if production else out / r["id"]) for r in selected
    ]
    for row, dest in jobs:
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
        runner.generation_stats = {"tokens": 0, "seconds": 0.0}
        started = time.monotonic()

        def execute(batch):
            runner.generation["batch_size"] = batch
            return evaluate(runner, list(cfg["benchmarks"]), cfg["benchmarks"])

        result, used_batch = retry_oom(execute, generation["batch_size"])
        manifest = {
            "backend": backend,
            "generation_stats": getattr(runner, "generation_stats", {}),
            "signature": signature,
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
            "limit": limit,
            "config": str(config),
            "minutes": (time.monotonic() - started) / 60,
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
        if result is not None and not complete(summary | manifest, signature, cfg["benchmarks"]):
            manifest["eval_status"] = "incomplete"
        payload = summary | manifest
        atomic_json(dest / "eval.json", payload)
        upsert_index(out / "index.jsonl", payload | {"path": str(dest)})
        if hasattr(runner, "close"):
            runner.close()
        torch.cuda.empty_cache()
        print(json.dumps(manifest), flush=True)
    restore.restore()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--zoo", required=True)
    parser.add_argument("--config", default="configs/eval_fast.yaml")
    parser.add_argument("--subset", choices=["d3"])
    parser.add_argument(
        "--out", help="isolated output directory; never writes canonical zoo results"
    )
    parser.add_argument("--skip-base", action="store_true", help="time selected entries only")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--eligibility", default="production", choices=["production", "smoke"])
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, help="tiny GPU smoke items per benchmark")
    parser.add_argument(
        "--resume", action="store_true", help="completed entries are always skipped"
    )
    args = vars(parser.parse_args())
    args.pop("resume")
    for key in ("ids", "families"):
        if args[key]:
            args[key] = [v for group in args[key] for v in group.split(",")]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run_batch(**args)


if __name__ == "__main__":
    main()
