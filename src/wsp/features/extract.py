"""P3 feature extraction: per zoo entry, φ_stats / φ_svd of the *effective* delta (docs/DECISIONS.md D1).

    python -m wsp.features.extract --zoo data/zoo/qwen3.5-0.8b --out data/features/qwen3.5-0.8b \
        [--ids a,b] [--workers N] [--limit N] [--base-file base.safetensors]

Per entry ``<out>/<id>/``:
  features.safetensors  ``stats`` (n, 18), ``svd`` (n, rank, 1 + 2*proj_dim), and for LoRA entries
                        ``lora`` (m, rmax, 1 + 2*proj_dim) canonical (B, A) tokens. Rows follow
                        ``features.json["tensors"]`` (``row`` / ``lora_row``).
  pooled.safetensors    ``stats`` (1098,) and ``svd`` (120,) from :func:`wsp.features.pool.pool_features`.
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

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from wsp.features.pool import pool_features
from wsp.features.stats import DELTA_STAT_NAMES, delta_stats
from wsp.features.svd import canonical_lora_tokens, truncated_delta_tokens
from wsp.zoo.apply import entry_has_artifacts, load_entry_delta, lora_factors

log = logging.getLogger("wsp.features.extract")

DEFAULT_CONFIG: dict[str, Any] = {
    "feature_version": 3,  # 2: out_proj pools with o_proj; 3: per-type pools for hybrid/norm/embed
    "rank": 8,
    "proj_dim": 64,
    "stats_k": 8,
    "buckets": 4,
    "seed": 0,
    "delta": "effective_bf16",
}


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


class MetadataError(ValueError):
    """Malformed edit.json or manifest row for one entry (recorded as that entry's failure)."""


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _check_edit(edit: Any, where: str) -> dict[str, Any]:
    if not isinstance(edit, dict):
        raise MetadataError(f"{where}: expected a JSON object, got {type(edit).__name__}")
    for key in ("family", "base_model"):
        if not isinstance(edit.get(key), str) or not edit[key]:
            raise MetadataError(f"{where}: missing or invalid {key!r}: {edit.get(key)!r}")
    if edit.get("parent_id") is not None and not isinstance(edit["parent_id"], str):
        raise MetadataError(f"{where}: invalid parent_id {edit['parent_id']!r}")
    seed = edit.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise MetadataError(f"{where}: invalid seed {seed!r}")
    for key in ("params", "provenance"):
        if edit.get(key) is not None and not isinstance(edit[key], dict):
            raise MetadataError(f"{where}: invalid {key!r}: expected an object")
    for key in ("eligibility", "base_revision"):
        if edit.get(key) is not None and not isinstance(edit[key], str):
            raise MetadataError(f"{where}: invalid {key!r}: {edit[key]!r}")
    return edit


def read_edit(entry_dir: str | Path) -> dict[str, Any]:
    """Parse and validate ``<entry>/edit.json``; raises :class:`MetadataError` with the reason."""
    path = Path(entry_dir) / "edit.json"
    try:
        edit = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise MetadataError(f"{path}: unreadable edit.json ({type(e).__name__}: {e})") from e
    return _check_edit(edit, str(path))


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
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Featurise one zoo entry into ``out_dir``; returns (and writes) the features.json dict."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    entry_dir, out_dir = Path(entry_dir), Path(out_dir)
    edit = read_edit(entry_dir)
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
        "identity_hash": config_hash(identity) if identity else None,
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


def resolve_base_commit(base_model: str, revision: str) -> str | None:
    """Commit hash of the cached base checkpoint snapshot (None if the cache is not hash-named)."""
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            base_model, revision=revision, allow_patterns=["config.json"], local_files_only=True
        )
    )
    return snapshot.name if re.fullmatch(r"[0-9a-f]{40}", snapshot.name) else None


def load_base_state(base_model: str, revision: str) -> tuple[dict[str, torch.Tensor], str | None]:
    """bf16 CPU state dict of the cached base checkpoint and its resolved commit hash."""
    import transformers

    commit = resolve_base_commit(base_model, revision)
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


def base_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Content hash of a base state (names, dtypes, shapes, bytes) for bases without a commit."""
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def resume_identity(cfg: dict[str, Any], threads: int, base: str | None) -> dict[str, Any]:
    """Everything that can change the output bytes of an entry (``workers`` cannot, so it is absent)."""
    return {
        "config": cfg,  # feature_version, rank, proj_dim, projection seed, ...
        "threads": threads,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "base": base,
    }


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in d.items():
        flat.update(_flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return flat


def identity_diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    a, b = _flatten(old), _flatten(new)
    return [
        f"{k}: {a.get(k)!r} -> {b.get(k)!r}"
        for k in sorted(set(a) | set(b))
        if a.get(k) != b.get(k)
    ]


def _manifest_ids(zoo: Path) -> tuple[list[str], dict[str, str]]:
    """Entry ids of the zoo manifest plus ``{"manifest.jsonl:<line>": reason}`` for malformed rows."""
    path = zoo / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    seen: dict[str, None] = {}
    errors: dict[str, str] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        key = f"manifest.jsonl:{n}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors[key] = f"MetadataError: invalid JSON ({e})"
            continue
        if not isinstance(row, dict):
            errors[key] = f"MetadataError: row is {type(row).__name__}, expected an object"
            continue
        header = "dry_run" in row or "families_requested" in row  # build-run header rows, no id
        if header or not ("id" in row or any(k in row for k in ("family", "seed", "parent_id"))):
            continue
        entry_id = row.get("id")
        if not isinstance(entry_id, str) or not _SAFE_ID.fullmatch(entry_id):
            errors[key] = f"MetadataError: missing or invalid id {entry_id!r}"
        elif not isinstance(row.get("family"), str) or not row["family"]:
            errors[key] = f"MetadataError: row {entry_id} missing or invalid family"
        elif row.get("parent_id") is not None and not isinstance(row["parent_id"], str):
            errors[key] = f"MetadataError: row {entry_id} invalid parent_id {row['parent_id']!r}"
        else:
            seen.setdefault(entry_id)
    return list(seen), errors


def _is_done(out_entry: Path, chash: str, ihash: str) -> bool:
    f = out_entry / "features.json"
    if not (f.exists() and (out_entry / "pooled.safetensors").exists()):
        return False
    try:
        meta = json.loads(f.read_text())
    except json.JSONDecodeError:
        return False
    return meta.get("config_hash") == chash and meta.get("identity_hash") == ihash


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
            identity=_STATE["identity"],
        )
        return entry_id, time.perf_counter() - t0, None
    except Exception as e:  # noqa: BLE001
        # recorded; one bad entry (e.g. mid-write by a running build) must not abort the run
        return entry_id, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def _init_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, path)


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

    Resume identity (stored in ``<out>/manifest.json`` and per entry): the feature config
    (feature_version, rank, proj_dim, projection seed, ...), ``threads``, torch and numpy versions and
    the base (commit, or a content hash for an injected base). A run with any difference is refused.
    Malformed manifest rows or edit.json files are recorded under ``failed`` and do not abort the run.
    """
    zoo, out = Path(zoo), Path(out)
    cfg = {**DEFAULT_CONFIG, **config}
    chash = config_hash(cfg)
    manifest_path = out / "manifest.json"
    old: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{manifest_path} is unreadable ({e}); use a new --out") from e

    all_ids, failed = _manifest_ids(zoo)
    if ids is not None:
        unknown = sorted(set(ids) - set(all_ids))
        if unknown:
            raise ValueError(f"ids not in zoo manifest: {unknown}")
        all_ids = [i for i in all_ids if i in set(ids)]
    if limit is not None:
        all_ids = all_ids[:limit]
    ready = [i for i in all_ids if entry_has_artifacts(zoo / i)]
    no_artifacts = sorted(set(all_ids) - set(ready))
    if no_artifacts:
        log.warning("%d manifest entries have no artefacts yet; skipped", len(no_artifacts))
    edits: dict[str, dict[str, Any]] = {}
    for i in ready:
        try:
            edits[i] = read_edit(zoo / i)
        except MetadataError as e:
            failed[i] = f"MetadataError: {e}"
            log.error("entry %s has malformed metadata: %s", i, e)
    valid = [i for i in ready if i in edits]
    edit0 = edits[valid[0]] if valid else {}

    # base identity (global rejection of incompatible bases stays deliberate)
    revisions = {
        (edits[i].get("provenance") or {}).get("resolved_base_revision") for i in valid
    } - {None}
    revision = next(iter(revisions), edit0.get("base_revision") or "main")
    if base_state is not None:
        base_id = base_commit or f"state:{base_fingerprint(base_state)}"
    else:
        if len({edits[i]["base_model"] for i in valid}) > 1:
            raise ValueError("entries use different base models")
        if len(revisions) > 1:
            raise ValueError(f"entries pin different base revisions: {sorted(revisions)}")
        base_id = None
        if valid:
            try:
                base_id = resolve_base_commit(edit0["base_model"], revision)
            except Exception as e:  # noqa: BLE001  # offline cache miss; load below fails loudly
                log.warning("cannot resolve base commit: %s", e)
            if base_id is not None and revisions and base_id not in revisions:
                raise ValueError(f"cached base commit {base_id} != pinned {sorted(revisions)}")
            base_id = base_id or next(iter(revisions), None)
    identity = resume_identity(cfg, threads, base_id)
    ihash = config_hash(identity)
    if old:
        old_identity = old.get("resume_identity")
        if old_identity is None:
            raise ValueError(
                f"{out} was written by an extractor without resume identity; use a new --out"
            )
        if diffs := identity_diff(old_identity, identity):
            raise ValueError(
                f"refusing to resume {out}: extraction identity differs ({'; '.join(diffs)}); "
                "use a new --out"
            )

    todo = [i for i in valid if not _is_done(out / i, chash, ihash)]
    skipped = len(valid) - len(todo)
    if todo and base_state is None:
        base_state, commit = load_base_state(edit0["base_model"], revision)
        if commit is not None and commit != base_id:
            raise ValueError(f"loaded base commit {commit} != resolved {base_id}")
        base_commit = commit
    elif base_state is None:
        base_commit = base_id
    n_layers = n_layers or (_n_layers(list(base_state)) if base_state else old.get("n_layers"))

    out.mkdir(parents=True, exist_ok=True)

    def write_manifest(final: bool, seconds: dict[str, float]) -> None:
        complete = sum(1 for i in all_ids if (out / i / "features.json").exists())
        stale = {  # failures of earlier runs for entries outside this run's selection
            k: v
            for k, v in (old.get("failed") or {}).items()
            if k not in failed and k not in all_ids and not k.startswith("manifest.jsonl:")
        }
        _write_json(
            manifest_path,
            {
                "zoo": str(zoo),
                "git_sha": _git_sha(),
                "config": cfg,
                "config_hash": chash,
                "resume_identity": identity,
                "identity_hash": ihash,
                "base_model": edit0.get("base_model") or old.get("base_model"),
                "base_revision": edit0.get("base_revision") or old.get("base_revision"),
                "base_commit": base_id,
                "n_layers": n_layers,
                "n_entries": complete,
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "complete_run": final,
                "failed": {**stale, **failed},
                "last_run": {
                    "done": len(seconds),
                    "skipped": skipped,
                    "failed": sorted(failed),
                    "no_artifacts": len(no_artifacts),
                },
            },
        )

    write_manifest(False, {})  # identity is bound to the out dir before any entry is written
    _STATE.update(
        zoo=zoo,
        out=out,
        base=base_state,
        commit=base_commit,
        n_layers=n_layers,
        cfg=cfg,
        identity=identity,
    )
    seconds: dict[str, float] = {}
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
    write_manifest(True, seconds)
    return {
        "done": len(seconds),
        "skipped": skipped,
        "failed": failed,
        "no_artifacts": no_artifacts,
        "seconds": seconds,
    }


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
        base_commit=f"file:{Path(args.base_file).name}:{file_sha256(args.base_file)}"
        if args.base_file
        else None,
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
