"""One-load, resumable zoo evaluation: python -m wsp.eval.batch."""

import argparse
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml

from wsp.eval.run import HFRunner, apply_zoo_entry, evaluate, write_results
from wsp.eval.zoo_compat import load_entry_delta
from wsp.zoo.common import get_git_sha


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


def select_entries(rows, zoo, ids=None, families=None, eligibility="production", signature=None):
    for row in rows:
        if "id" not in row or row.get("eligibility") != eligibility:
            continue
        if ids and row["id"] not in ids:
            continue
        if families and row["family"] not in families:
            continue
        result = zoo / row["id"] / "eval.json"
        if result.exists():
            previous = json.loads(result.read_text())
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
    if limit:
        for spec in cfg["benchmarks"].values():
            spec["n"] = spec["n_windows"] = limit
    rows = [json.loads(line) for line in (zoo / "manifest.jsonl").read_text().splitlines()]
    signature = hashlib.sha256(
        json.dumps([cfg, generation, limit], sort_keys=True).encode()
    ).hexdigest()
    selected = list(select_entries(rows, zoo, ids, families, eligibility, signature))
    header = next(r for r in reversed(rows) if r.get("type") == "header")
    out = Path("data/evals") / zoo.name
    base_out = out / "base"
    base_done = (base_out / "eval.json").exists() and json.loads(
        (base_out / "eval.json").read_text()
    ).get("eval_status") == "complete"
    base_done = (
        base_done and json.loads((base_out / "eval.json").read_text()).get("signature") == signature
    )
    if not selected and base_done:
        return
    runner = HFRunner(header["base_model"], generation, revision=header["base_revision"])
    restore = WeightRestore(runner.model)
    runner.base_context = restore.base_context
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    out.mkdir(parents=True, exist_ok=True)
    jobs = ([] if base_done else [(None, base_out)]) + [(r, zoo / r["id"]) for r in selected]
    for row, dest in jobs:
        restore.restore()
        runner.edit_id = row["id"] if row else "base"
        runner.rounding_stats = {}
        if row:
            edit = json.loads((dest / "edit.json").read_text())
            if edit["base_revision"] != runner.revision:
                raise ValueError("entry base revision differs from loaded snapshot")
            delta = load_entry_delta(dest, runner.model.state_dict().keys())
            restore.capture(delta)
            del delta
            runner.rounding_stats = apply_zoo_entry(runner.model, dest)
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
        manifest = {
            "signature": signature,
            "seed": generation["seed"],
            "config_hash": hashlib.sha256(config.read_bytes()).hexdigest(),
            "git_sha": get_git_sha(),
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
        payload = summary | manifest
        atomic_json(dest / "eval.json", payload)
        with (out / "index.jsonl").open("a") as stream:
            stream.write(json.dumps(payload | {"path": str(dest)}) + "\n")
        print(json.dumps(manifest), flush=True)
    restore.restore()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--zoo", required=True)
    parser.add_argument("--config", default="configs/eval_fast.yaml")
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
