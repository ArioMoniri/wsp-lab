#!/usr/bin/env python3
"""Confined, leaf-verified base-model loader for the F12 inputs code (PREREG_H1_TF A5 note F12-R, second addendum;
A10 as re-checked by the independent F12 remedy re-check of 12 Nov 2026).

The previous loaders resolved only the Hugging Face cache *directory* and then let ``from_pretrained`` open whatever
the cache's symlinks pointed to; the state hash was checked only after those reads. This loader instead:

1. reads the committed pin file (``docs/prereg/a5_manifests/f12/f12r/base_model_leaves.json``, sha256 ``PIN_SHA256``),
   which names the repository, the revision, the base-state hash and the sha256/size of every file the library reads
   (config, weight index, weight shard), and requires its base-state hash to equal the one the caller expects;
2. resolves each pinned leaf of ``<hub cache>/models--<repo>/snapshots/<revision>/`` individually (every symlink, to the
   leaf), confines it to the allowed cache root (that model's own cache tree plus the hub's shared content-addressed
   blob store) and to the job's read boundary (``data/evals`` and the other
   prohibited paths refused before opening), and streams it with ``O_NOFOLLOW`` into a private staging directory as a
   regular file, checking its sha256 and size against the pin; nothing unpinned is staged;
3. loads from the staging directory only (``local_files_only``), under a Python audit hook that refuses every
   ``open``/``listdir``/``scandir`` outside the staging directory and the interpreter's code and system trees
   (``sys.prefix``/``base_prefix``, the repository's ``src``/``scripts``, ``/proc``, ``/sys``, ``/dev``): a library read
   of anything else fails closed; reads by native code (safetensors, torch) are not seen by the hook, but every file
   in the staging directory is a verified regular copy;
4. hashes the loaded state (G2 convention) and requires the pinned base-state hash; the staging directory is removed.

    python scripts/f12_base_loader.py pin --out <path>       # write the pin from the cache (hashes only)
    python scripts/f12_base_loader.py check --out <record>   # load-only check under the 4 GB cap
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import f12_inputs_guard as fg

MODEL = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")  # = a5_power_fields.MODEL
# The pinned base-state hash: the G2 WikiText field's identity.state_hash, recorded as base.state_hash in the frozen
# tokens_manifest.json and the committed additivity plan.
BASE_STATE_HASH = "18f07d70a89aa63125e7e95535ff2730eb54bae4ee846d72f183e004a740d982"
BASE_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model.safetensors-00001-of-00001.safetensors",
)
PIN_SCHEMA = "f12r-base-model-leaves-v1"
PIN_PATH = ROOT / "docs/prereg/a5_manifests/f12/f12r/base_model_leaves.json"
PIN_SHA256 = "6d99ad84c23d451be6800328bb80740ed75abf1eaf8bbd99c19cd3c256ce50f6"  # docs/prereg/a5_manifests/f12/f12r/base_model_leaves.json
CHECK_SCHEMA = "f12r-base-load-check-v1"


def hub_cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def model_cache_root(hub_cache, repo) -> Path:
    return Path(hub_cache) / ("models--" + repo.replace("/", "--"))


def allowed_cache_roots(hub_cache, repo) -> list:
    """The allowed cache root of a leaf: this model's own cache tree and the hub's shared content-addressed blob store
    (``<hub>/blobs``, into which this cache's weight blob links); no other hub tree (other models, datasets)."""
    return [model_cache_root(hub_cache, repo), Path(hub_cache) / "blobs"]


def state_hash(state) -> str:
    from wsp.features import response_field as rf

    return rf._hash({n: rf._tensor_hash(t) for n, t in state.items()})


# ----------------------------------------------------------------------------------------- confinement

_CONFINED = {"allowed": None}
_HOOK = {"installed": False}


def _audit(event, args):
    allowed = _CONFINED["allowed"]
    if allowed is None or event not in ("open", "os.listdir", "os.scandir"):
        return
    path = args[0] if args else "."
    if path is None or isinstance(path, int):
        return
    try:
        p = os.fsdecode(path)
    except TypeError:
        return
    for q in (os.path.abspath(p), os.path.realpath(p)):
        if not any(q == a or q.startswith(a.rstrip(os.sep) + os.sep) for a in allowed):
            raise PermissionError(f"confined base load: refused {p}")


def code_roots() -> list:
    roots = {
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        str(ROOT / "src"),
        str(ROOT / "scripts"),
        "/proc",
        "/sys",
        "/dev",
    }
    return sorted({os.path.abspath(r) for r in roots} | {os.path.realpath(r) for r in roots})


class confined_reads:
    """Context in which Python-level opens/listings outside ``allowed`` (plus the code roots) raise."""

    def __init__(self, allowed):
        self.allowed = (
            sorted({os.path.abspath(a) for a in allowed} | {os.path.realpath(a) for a in allowed})
            + code_roots()
        )

    def __enter__(self):
        if not _HOOK["installed"]:
            sys.addaudithook(_audit)
            _HOOK["installed"] = True
        _CONFINED["allowed"] = self.allowed
        return self

    def __exit__(self, *exc):
        _CONFINED["allowed"] = None
        return False


# --------------------------------------------------------------------------------------------- staging


def stage_leaves(pin: dict, hub_cache, dest: Path, boundary) -> list:
    """Resolve, confine, verify and copy every pinned leaf into ``dest`` (regular files); records."""
    root = model_cache_root(hub_cache, pin["repo"])
    snap = root / "snapshots" / pin["revision"]
    leaf_boundary = fg.ReadBoundary(
        allowed=allowed_cache_roots(hub_cache, pin["repo"]), prohibited=boundary.prohibited
    )
    records = []
    for name, want in sorted(pin["files"].items()):
        fg.safe_name(name)
        p = snap / name
        real = leaf_boundary.resolve(
            p
        )  # every symlink resolved; confined to this model's cache root
        boundary.resolve(p)  # and to the job's boundary
        if not real.is_file():
            raise fg.AuthenticationError(f"{p}: not a regular file leaf")
        import hashlib

        h, size = hashlib.sha256(), 0
        fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW)
        out_fd = os.open(dest / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "rb") as src, os.fdopen(out_fd, "wb") as dst:
            for chunk in iter(lambda: src.read(1 << 24), b""):
                h.update(chunk)
                size += len(chunk)
                dst.write(chunk)
        got = h.hexdigest()
        if got != want["sha256"] or size != want["bytes"]:
            raise fg.AuthenticationError(
                f"{p}: sha256 {got} / {size} bytes differ from the pinned {want['sha256']} / {want['bytes']}"
            )
        rec = {
            "path": str(p),
            "resolved": str(real),
            "sha256": got,
            "bytes": size,
            "authenticated": True,
            "note": "base-model leaf: resolved, confined to the model cache root, pinned, staged as a regular copy",
        }
        boundary.reads.append(rec)
        records.append(rec)
    return records


def transformers_load(dest: Path):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(dest), dtype=torch.bfloat16, device_map="cpu", local_files_only=True
    )
    state = {n: t.detach() for n, t in model.state_dict().items()}
    del model
    return state


def _prepare_transformers():
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM  # noqa: F401


transformers_load.prepare = _prepare_transformers


def load_verified_base(
    boundary,
    expected_hash,
    pin_path=None,
    pin_sha256=None,
    hub_cache=None,
    loader=None,
    state_hash_fn=None,
):
    """(state, state hash) of the base, loaded only from its verified, staged leaves (see the module docstring)."""
    pin_path = Path(pin_path or PIN_PATH)
    pin_sha256 = pin_sha256 or PIN_SHA256
    if pin_sha256 is None:
        raise fg.AuthenticationError("no base-model pin is committed (PIN_SHA256 is None)")
    pin = fg.strict_json_loads(boundary.read_bytes(pin_path, sha256=pin_sha256).decode())
    if pin.get("schema") != PIN_SCHEMA:
        raise ValueError("base-model pin schema differs")
    if pin.get("base_state_hash") != expected_hash:
        raise ValueError(
            f"pinned base state {pin.get('base_state_hash')} differs from the expected {expected_hash}"
        )
    loader = loader or transformers_load
    state_hash_fn = state_hash_fn or state_hash
    dest = Path(tempfile.mkdtemp(prefix="f12r-base-"))
    try:
        stage_leaves(pin, hub_cache or hub_cache_dir(), dest, boundary)
        prep = getattr(loader, "prepare", None)
        if prep is not None:
            prep()
        with confined_reads([dest]):
            state = loader(dest)
        got = state_hash_fn(state)
        if got != expected_hash:
            raise ValueError(
                f"loaded base state hash {got} differs from the pinned {expected_hash}"
            )
        return state, got
    finally:
        shutil.rmtree(dest, ignore_errors=True)


# ------------------------------------------------------------------------------------------------- CLI


def write_pin(out: Path, hub_cache=None) -> dict:
    """Hash the pinned files of the cache snapshot (no load). The base-state hash is the committed one."""
    hub_cache = hub_cache or hub_cache_dir()
    root = model_cache_root(hub_cache, MODEL[0])
    snap = root / "snapshots" / MODEL[1]
    b = fg.ReadBoundary(allowed=allowed_cache_roots(hub_cache, MODEL[0]))
    files = {}
    for name in BASE_FILES:
        real = b.resolve(snap / name)
        files[name] = {"sha256": b.sha256_file(snap / name), "bytes": real.stat().st_size}
    pin = {
        "schema": PIN_SCHEMA,
        "repo": MODEL[0],
        "revision": MODEL[1],
        "base_state_hash": BASE_STATE_HASH,
        "files": files,
        "rule": "every file the base loader reads; each leaf resolved, confined to the model cache root and checked "
        "against these hashes before staging; the loaded state must hash to base_state_hash",
    }
    if Path(out).exists():
        raise FileExistsError(f"{out} exists")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(pin, indent=1, sort_keys=True) + "\n")
    return pin


def load_check(out: Path, threads: int) -> dict:
    """Load-only check within the 4 GB cap: the hardened loader reproduces the pinned base-state hash."""
    import torch

    torch.set_num_threads(threads)
    if Path(out).exists():
        raise FileExistsError(f"{out} exists")
    t0 = time.time()
    guard = fg.ResourceGuard(build="scripts/f12_base_loader.py", threads=threads)
    guard.launch(3_114_868_736, breach_path=Path(out).with_name(Path(out).stem + "_breach.json"))
    boundary = fg.ReadBoundary(allowed=[ROOT / "docs/prereg/a5_manifests/f12", hub_cache_dir()])
    state, h = load_verified_base(boundary, BASE_STATE_HASH)
    guard.check()
    rec = {
        "schema": CHECK_SCHEMA,
        "base_state_hash": h,
        "matches_pin": h == BASE_STATE_HASH,
        "n_tensors": len(state),
        "pin": {"path": str(PIN_PATH.relative_to(ROOT)), "sha256": PIN_SHA256},
        "wall_seconds": time.time() - t0,
        "resources": guard.stop(),
        "read_boundary": boundary.record(),
    }
    Path(out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("pin", "check"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    a = p.parse_args(argv)
    if a.command == "pin":
        pin = write_pin(a.out)
        print(
            json.dumps(
                {"files": pin["files"], "sha256": fg.sha256_bytes(a.out.read_bytes())}, indent=1
            )
        )
    else:
        rec = load_check(a.out, a.threads)
        print(
            json.dumps(
                {k: rec[k] for k in ("base_state_hash", "matches_pin", "n_tensors", "wall_seconds")}
            )
        )
        print(
            json.dumps(
                {k: rec["resources"][k] for k in ("peak_rss_bytes", "cap_bytes", "within_cap")}
            )
        )
        return 0 if rec["matches_pin"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
