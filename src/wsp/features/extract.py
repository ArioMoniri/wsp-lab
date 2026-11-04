"""P3 feature extraction: per zoo entry, φ_stats / φ_svd of the *effective* delta (docs/DECISIONS.md D1).

    python -m wsp.features.extract --zoo data/zoo/qwen3.5-0.8b --out data/features/qwen3.5-0.8b \
        [--ids a,b] [--workers N] [--limit N] [--base-file base.safetensors]

Per entry ``<out>/<id>/``:
  features.safetensors  ``stats`` (n, 18), ``svd`` (n, rank, 1 + 2*proj_dim), and for LoRA entries
                        ``lora`` (m, rmax, 1 + 2*proj_dim) canonical (B, A) tokens. Rows follow
                        ``features.json["tensors"]`` (``row`` / ``lora_row``).
  pooled.safetensors    ``stats`` (522,) and ``svd`` (56,) from :func:`wsp.features.pool.pool_features`.
  features.json         metadata; written last, so its presence marks the entry as complete.

Δ_eff = bf16(W + Δ) − W is computed exactly as the eval loader applies deltas (W upcast to fp32, add,
cast to bf16). Tensors whose Δ_eff is identically zero are listed under ``unchanged_tensors`` and are
not featurised. Chain entries store cumulative deltas, so their features are relative to the base.
LoRA canonical tokens describe the un-rounded factorisation (B, A); ``svd`` always describes Δ_eff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from wsp.features.pool import pool_features
from wsp.features.stats import DELTA_STAT_NAMES, delta_stats
from wsp.features.svd import canonical_lora_tokens, truncated_delta_tokens
from wsp.zoo.apply import entry_has_artifacts, load_entry_delta, lora_factors

log = logging.getLogger("wsp.features.extract")

DEFAULT_CONFIG: dict[str, Any] = {
    "feature_version": 2,  # 2: out_proj pools with o_proj
    "rank": 8,
    "proj_dim": 64,
    "stats_k": 8,
    "buckets": 4,
    "seed": 0,
    "delta": "effective_bf16",
}


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


@torch.no_grad()
def effective_delta(base: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Δ_eff = bf16-style round(W + Δ) − W in fp32 (rounded to ``base.dtype``) plus rounding stats."""
    w = base.float()
    d = delta.to(torch.float32)
    effective = (w + d).to(base.dtype).float() - w
    nonzero = d != 0
    lost = nonzero & (effective == 0)
    n_nonzero, n_lost = int(nonzero.sum()), int(lost.sum())
    stats = {
        "fraction_nonzero_lost": n_lost / max(n_nonzero, 1),
        "delta_norm": torch.linalg.vector_norm(d).item(),
        "effective_delta_norm": torch.linalg.vector_norm(effective).item(),
        "max_abs_lost": (d - effective).abs().max().item() if d.numel() else 0.0,
        "n_nonzero": n_nonzero,
        "n_lost": n_lost,
    }
    return effective, stats


def _n_layers(state_keys: Sequence[str]) -> int:
    idx = [int(m.group(1)) for k in state_keys if (m := re.search(r"\.layers\.(\d+)\.", k))]
    if not idx:
        raise ValueError("cannot infer n_layers: no '.layers.<i>.' keys in the base state")
    return max(idx) + 1


def _atomic_save(tensors: dict[str, torch.Tensor], path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    save_file({k: v.float().contiguous() for k, v in tensors.items()}, tmp)
    os.replace(tmp, path)


def extract_entry(
    entry_dir: str | Path,
    base_state: dict[str, torch.Tensor],
    out_dir: str | Path,
    *,
    base_commit: str | None = None,
    n_layers: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Featurise one zoo entry into ``out_dir``; returns (and writes) the features.json dict."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    entry_dir, out_dir = Path(entry_dir), Path(out_dir)
    edit = json.loads((entry_dir / "edit.json").read_text())
    n_layers = n_layers or _n_layers(list(base_state))
    lora = not (entry_dir / "delta.safetensors").exists()
    delta = load_entry_delta(entry_dir, base_state.keys())
    factors = lora_factors(entry_dir / "lora", base_state.keys()) if lora else {}
    for name, d in delta.items():
        if name not in base_state:
            raise KeyError(f"delta tensor {name} not in the base state")
        if tuple(d.shape) != tuple(base_state[name].shape):
            raise ValueError(
                f"delta {name} has shape {tuple(d.shape)}, base {tuple(base_state[name].shape)}"
            )

    records, rows, unchanged = [], [], []
    lora_tokens: list[torch.Tensor] = []
    totals = {
        "n_nonzero": 0,
        "n_lost": 0,
        "sq": 0.0,
        "sq_eff": 0.0,
        "max_abs_lost": 0.0,
        "sq_base": 0.0,
    }
    for name in sorted(delta):
        w = base_state[name]
        eff, rs = effective_delta(w, delta[name])
        totals["n_nonzero"] += rs["n_nonzero"]
        totals["n_lost"] += rs["n_lost"]
        totals["sq"] += rs["delta_norm"] ** 2
        totals["sq_eff"] += rs["effective_delta_norm"] ** 2
        totals["max_abs_lost"] = max(totals["max_abs_lost"], rs["max_abs_lost"])
        if not bool(eff.any()):
            unchanged.append(name)
            continue
        totals["sq_base"] += torch.linalg.vector_norm(w.float()).item() ** 2
        stats = delta_stats(w, w.float() + eff, k=cfg["stats_k"])
        svd = truncated_delta_tokens(
            eff, rank=cfg["rank"], proj_dim=cfg["proj_dim"], seed=cfg["seed"]
        )
        lora_row = None
        if name in factors:
            B, A, scaling = factors[name]
            lora_row = len(lora_tokens)
            lora_tokens.append(
                canonical_lora_tokens(B * scaling, A, proj_dim=cfg["proj_dim"], seed=cfg["seed"])
            )
        records.append({"name": name, "stats": stats, "svd": svd})
        rows.append(
            {
                "name": name,
                "shape": list(w.shape),
                "row": len(rows),
                "lora_row": lora_row,
                "fro_stored": rs["delta_norm"],
                "fro_effective": rs["effective_delta_norm"],
                "fraction_nonzero_lost": rs["fraction_nonzero_lost"],
            }
        )

    width = 1 + 2 * cfg["proj_dim"]
    feats = {
        "stats": torch.stack([r["stats"] for r in records])
        if records
        else torch.zeros(0, len(DELTA_STAT_NAMES)),
        "svd": torch.stack([r["svd"] for r in records])
        if records
        else torch.zeros(0, cfg["rank"], width),
    }
    if lora_tokens:
        rmax = max(t.shape[0] for t in lora_tokens)
        feats["lora"] = torch.stack(
            [torch.nn.functional.pad(t, (0, 0, 0, rmax - t.shape[0])) for t in lora_tokens]
        )
    pooled_stats, pooled_svd = pool_features(records, n_layers, cfg["buckets"])

    fro_eff = totals["sq_eff"] ** 0.5
    prov = edit.get("provenance") or {}
    meta: dict[str, Any] = {
        "edit_id": edit.get("id", entry_dir.name),
        "family": edit["family"],
        "parent_id": edit.get("parent_id"),
        "seed": edit.get("seed"),
        "params": edit.get("params"),
        "step_semantics": prov.get("step_semantics"),
        "eligibility": edit.get("eligibility", "unclassified"),
        "base_model": edit.get("base_model"),
        "base_revision": edit.get("base_revision"),
        "base_commit": base_commit,
        "stored_delta_format": "lora" if lora else "dense",
        "delta_relative_to": "base",
        "n_layers": n_layers,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "stat_names": DELTA_STAT_NAMES,
        "rounding_stats": {
            "fraction_nonzero_lost": totals["n_lost"] / max(totals["n_nonzero"], 1),
            "n_nonzero": totals["n_nonzero"],
            "n_lost": totals["n_lost"],
            "delta_norm": totals["sq"] ** 0.5,
            "effective_delta_norm": fro_eff,
            "max_abs_lost": totals["max_abs_lost"],
        },
        "norms": {
            "fro_stored": totals["sq"] ** 0.5,
            "fro_effective": fro_eff,
            "fro_base_changed": totals["sq_base"] ** 0.5,
            "rel_fro_effective": fro_eff / (totals["sq_base"] ** 0.5 + 1e-12),
            "n_changed_tensors": len(rows),
        },
        "tensors": rows,
        "unchanged_tensors": unchanged,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(feats, out_dir / "features.safetensors")
    _atomic_save({"stats": pooled_stats, "svd": pooled_svd}, out_dir / "pooled.safetensors")
    tmp = out_dir / ".features.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1, sort_keys=True))
    os.replace(tmp, out_dir / "features.json")  # completion marker
    return meta


def load_base_state(base_model: str, revision: str) -> tuple[dict[str, torch.Tensor], str | None]:
    """bf16 CPU state dict of the cached base checkpoint and its resolved commit hash."""
    import transformers
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            base_model, revision=revision, allow_patterns=["config.json"], local_files_only=True
        )
    )
    commit = snapshot.name if re.fullmatch(r"[0-9a-f]{40}", snapshot.name) else None
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
    )
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    return state, commit


def _manifest_ids(zoo: Path) -> list[str]:
    path = zoo / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    seen: dict[str, None] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if "id" in row:  # build runs also append header rows (config, families) without an id
            seen.setdefault(row["id"])
    return list(seen)


def _is_done(out_entry: Path, chash: str) -> bool:
    f = out_entry / "features.json"
    if not (f.exists() and (out_entry / "pooled.safetensors").exists()):
        return False
    return json.loads(f.read_text()).get("config_hash") == chash


_STATE: dict[str, Any] = {}


def _work(entry_id: str) -> tuple[str, float, str | None]:
    t0 = time.perf_counter()
    try:
        extract_entry(
            _STATE["zoo"] / entry_id,
            _STATE["base"],
            _STATE["out"] / entry_id,
            base_commit=_STATE["commit"],
            n_layers=_STATE["n_layers"],
            config=_STATE["cfg"],
        )
        return entry_id, time.perf_counter() - t0, None
    except Exception as e:  # noqa: BLE001
        # recorded; one bad entry (e.g. mid-write by a running build) must not abort the run
        return entry_id, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def _init_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def run_extract(
    zoo: str | Path,
    out: str | Path,
    *,
    base_state: dict[str, torch.Tensor] | None = None,
    base_commit: str | None = None,
    ids: Sequence[str] | None = None,
    limit: int | None = None,
    workers: int = 1,
    threads: int = 1,
    n_layers: int | None = None,
    **config: Any,
) -> dict[str, Any]:
    """Extract all (selected) manifest entries of ``zoo`` into ``out``; resumable and deterministic.

    Every entry is computed with exactly ``threads`` torch threads (default 1) whatever ``workers``
    is, so the output bytes do not depend on the degree of parallelism (LAPACK results vary with the
    thread count at the 1e-7 level). Scale with ``workers``, not ``threads``.
    """
    zoo, out = Path(zoo), Path(out)
    cfg = {**DEFAULT_CONFIG, **config}
    chash = config_hash(cfg)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text()).get("config_hash")
        if old != chash:
            raise ValueError(f"{out} was extracted with another config ({old}); use a new --out")
    all_ids = _manifest_ids(zoo)
    if ids is not None:
        unknown = sorted(set(ids) - set(all_ids))
        if unknown:
            raise ValueError(f"ids not in zoo manifest: {unknown}")
        all_ids = [i for i in all_ids if i in set(ids)]
    if limit is not None:
        all_ids = all_ids[:limit]
    ready = [i for i in all_ids if entry_has_artifacts(zoo / i)]
    no_artifacts = sorted(set(all_ids) - set(ready))
    todo = [i for i in ready if not _is_done(out / i, chash)]
    skipped = len(ready) - len(todo)
    if no_artifacts:
        log.warning("%d manifest entries have no artefacts yet; skipped", len(no_artifacts))

    edit0 = json.loads((zoo / ready[0] / "edit.json").read_text()) if ready else {}
    if todo and base_state is None:
        revisions = {
            (json.loads((zoo / i / "edit.json").read_text()).get("provenance") or {}).get(
                "resolved_base_revision"
            )
            for i in todo
        } - {None}
        if len(revisions) > 1:
            raise ValueError(f"entries pin different base revisions: {sorted(revisions)}")
        revision = next(iter(revisions), edit0.get("base_revision", "main"))
        base_state, base_commit = load_base_state(edit0["base_model"], revision)
        if revisions and base_commit not in revisions:
            raise ValueError(f"loaded base commit {base_commit} != pinned {sorted(revisions)}")
    n_layers = n_layers or (_n_layers(list(base_state)) if base_state else None)

    out.mkdir(parents=True, exist_ok=True)
    _STATE.update(zoo=zoo, out=out, base=base_state, commit=base_commit, n_layers=n_layers, cfg=cfg)
    seconds: dict[str, float] = {}
    failed: dict[str, str] = {}
    if workers > 1 and todo:
        ctx = mp.get_context("fork")  # shares the base state copy-on-write
        with ctx.Pool(workers, _init_worker, (threads,)) as pool:
            results = list(pool.imap_unordered(_work, todo))
    else:
        previous = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            results = [_work(i) for i in todo]
        finally:
            torch.set_num_threads(previous)
    for entry_id, secs, err in results:
        if err:
            failed[entry_id] = err
            log.error("extract %s failed: %s", entry_id, err)
        else:
            seconds[entry_id] = secs
    summary = {
        "done": len(seconds),
        "skipped": skipped,
        "failed": failed,
        "no_artifacts": no_artifacts,
        "seconds": seconds,
    }
    complete = sum(1 for i in all_ids if (out / i / "features.json").exists())
    manifest = {
        "zoo": str(zoo),
        "git_sha": _git_sha(),
        "config": cfg,
        "config_hash": chash,
        "base_model": edit0.get("base_model"),
        "base_revision": edit0.get("base_revision"),
        "base_commit": base_commit,
        "n_layers": n_layers,
        "n_entries": complete,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "last_run": {k: summary[k] for k in ("done", "skipped")}
        | {"failed": sorted(failed), "no_artifacts": len(no_artifacts)},
    }
    tmp = out / ".manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(tmp, manifest_path)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--zoo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", help="comma-separated entry ids")
    p.add_argument("--workers", type=int, default=1, help="parallel entries (processes)")
    p.add_argument(
        "--threads", type=int, default=1, help="torch threads per entry (fixed for determinism)"
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--base-file", help="safetensors base state instead of the cached HF checkpoint")
    p.add_argument("--rank", type=int, default=DEFAULT_CONFIG["rank"])
    p.add_argument("--proj-dim", type=int, default=DEFAULT_CONFIG["proj_dim"])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_state = load_file(args.base_file) if args.base_file else None
    t0 = time.perf_counter()
    summary = run_extract(
        args.zoo,
        args.out,
        base_state=base_state,
        base_commit=f"file:{Path(args.base_file).name}" if args.base_file else None,
        ids=args.ids.split(",") if args.ids else None,
        limit=args.limit,
        workers=args.workers,
        threads=args.threads,
        rank=args.rank,
        proj_dim=args.proj_dim,
    )
    secs = list(summary["seconds"].values())
    for entry_id, t in sorted(summary["seconds"].items()):
        log.info("entry %s %.1fs", entry_id, t)
    log.info(
        "done=%d skipped=%d failed=%d no_artifacts=%d wall=%.1fs mean=%.1fs/entry",
        summary["done"],
        summary["skipped"],
        len(summary["failed"]),
        len(summary["no_artifacts"]),
        time.perf_counter() - t0,
        sum(secs) / max(len(secs), 1),
    )
    if summary["failed"]:
        raise SystemExit(f"{len(summary['failed'])} entries failed: {sorted(summary['failed'])}")


if __name__ == "__main__":
    main()
