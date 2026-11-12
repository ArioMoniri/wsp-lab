#!/usr/bin/env python3
"""F12-R token requalification comparison (PREREG_H1_TF A5 chair ruling on the F12 token requalification rerun, 12 Nov
2026, 09:19 UTC, (c); A5 note F12-R).

Compares the hardened builder's full rerun with the frozen token run on every deterministic field:

* ``tokens.safetensors``: header metadata as a mapping (row ids, columns), tensor dtypes/shapes/offsets, the data
  section byte for byte, ``X`` bitwise (int32 view: -0.0 and NaN payloads count), ``presence`` and ``row_eligible``
  exactly; rows whose ``X`` slice differs are named by id. The whole-file sha256 is reported but not compared:
  safetensors serialises ``__metadata__`` keys in a nondeterministic order (observed: five writes of identical tensors
  gave two header orders), so the file hash is not a deterministic field;
* ``rows.jsonl``: same ids in the same order, each record equal as canonical JSON with ``seconds`` removed;
* ``detail.jsonl``, ``aq.json``, ``kframes.json``: byte-identical (differing detail rows named by id);
* ``kframes.safetensors``: byte-identical, else every tensor compared bitwise and differing names listed;
* ``pricing.json``: its deterministic keys only;
* the manifests: every deterministic field (``manifest_differences``).

Excluded, and listed in the report: ``EXCLUDED``. Before comparing, every file of both runs is authenticated against its
own manifest's ``files`` hashes, and every read goes through the F12 read boundary. The report holds equality results,
counts, ids, tensor names and hashes only: no token, a/q or K-frame value is printed or written.

    python scripts/f12_tokens_compare.py --new-manifest <m.json> --new-dir <dir> --out <report.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import f12_inputs_guard as fg
from scripts import f12_tokens as ft

SCHEMA = "f12r-token-comparison-v1"
FROZEN_MANIFEST = ROOT / "docs/prereg/a5_manifests/f12/tokens_manifest.json"
FROZEN_MANIFEST_SHA256 = (
    "424333d9b6b2e19a0586b34096b1968d1de054e6ec625da9bceb4507b567001f"  # C1 code_hashes pin
)
FROZEN_DIR = fg.DATA_ROOT / "runs/a5/f12_inputs/tokens"
ROW_KEYS = ft.ROW_KEYS
BYTE_FILES = ("detail.jsonl", "aq.json", "kframes.json")
AUTH_FILES = (
    "tokens.safetensors",
    "rows.jsonl",
    "detail.jsonl",
    "aq.json",
    "kframes.json",
    "kframes.safetensors",
    "pricing.json",
)
PRICING_KEYS = ("n_rows", "stored_tensors_total", "threads")
RESUME_HISTORY = (
    "the frozen run (relaunch 3, run_tokens_resume.sh) resumed with row 1 from the capped run 2, so its pricing records "
    "its first new row; a fresh rerun's first row differs by construction"
)
# Manifest fields that depend on time, memory, paths, the code identity (changed by design in F12-R), resume or the
# relabel; everything else is compared. Paths are dotted; "*" matches any key at that level.
MANIFEST_EXCLUDED = (
    "ordering_note",
    "out_dir",
    "rows_sha256",
    "code",
    "outcomes_read",
    "outcomes_read_intent",
    "read_boundary",
    "resources",
    "resume",
    "population_frozen.verified_at",
    "inputs.timing",
    "inputs.zoo",
    "inputs.*.path",
    "inputs.*.dir",
    "tokens.path",
    "kframes.path",
    "files.rows.jsonl",
    "files.tokens.safetensors",
    "tokens.sha256",
    "files.pricing.json",
    "files.checkpoint.json",
    "summary.timing",
    "summary.pricing",
    "summary.max_rss_bytes",
)
EXCLUDED = (
    (
        "tokens.safetensors whole-file sha256 (and manifest files.tokens.safetensors, tokens.sha256): safetensors "
        "writes __metadata__ keys in a nondeterministic order; the metadata mapping, tensor headers and data section "
        "are compared"
    ),
    "rows.jsonl: seconds",
    f"pricing.json: first_row_id ({RESUME_HISTORY})",
    f"pricing.json: first_row_stored_tensors ({RESUME_HISTORY})",
    "pricing.json: setup_seconds",
    "pricing.json: first_row_seconds",
    "pricing.json: projected_wall_seconds_naive",
    "pricing.json: projected_wall_seconds_tensor_weighted",
    "pricing.json: projected_wall_seconds",
    "pricing.json: projected_wall_hours",
    "pricing.json: within_cap",
    "pricing.json: cap_wall_hours",
    "checkpoint.json (new in the hardened builder; no frozen counterpart)",
    *(f"manifest: {k}" for k in MANIFEST_EXCLUDED),
)


def read_boundary(new_manifest: Path) -> fg.ReadBoundary:
    """The frozen and new token directories, the committed F12 manifests and the new manifest's directory."""
    return fg.ReadBoundary(
        allowed=[ft.ALLOWED_OUT, ROOT / "docs/prereg/a5_manifests/f12", Path(new_manifest).parent]
    )


def _excluded(path: str) -> bool:
    parts = path.split(".")
    for pat in MANIFEST_EXCLUDED:
        if path == pat or path.startswith(pat + "."):
            return True
        p = pat.split(".")
        if "*" in p and len(parts) >= len(p) and all(a == "*" or a == b for a, b in zip(p, parts)):
            return True
    return False


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                out |= _flatten(v, key)
            else:
                out[key] = v
    return out


def manifest_differences(frozen: dict, new: dict) -> list:
    """Dotted paths of deterministic manifest fields that differ (excluded fields ignored)."""
    a, b = _flatten(frozen), _flatten(new)
    return sorted(
        k
        for k in set(a) | set(b)
        if not _excluded(k)
        and json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True)
    )


def _authenticate(boundary, d: Path, man: dict, label: str) -> dict:
    out = {}
    for name in AUTH_FILES:
        want = man["files"].get(name)
        if want is None:
            raise fg.AuthenticationError(f"{label}: {name} has no hash in its manifest")
        out[name] = boundary.sha256_file(Path(d) / name, sha256=want)
    return out


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t


def _load_tensors(boundary, path: Path):
    from safetensors import safe_open
    from safetensors.torch import load_file

    real = boundary.resolve(path)
    with safe_open(str(real), "pt") as f:
        meta = f.metadata() or {}
    return load_file(str(real)), meta


def split_safetensors(data: bytes):
    """(header JSON object, data section bytes) of a safetensors file."""
    import struct

    (n,) = struct.unpack("<Q", data[:8])
    return json.loads(data[8 : 8 + n]), data[8 + n :]


def compare_tokens(boundary, fz: Path, nw: Path, shas) -> dict:
    a, ma = _load_tensors(boundary, fz / "tokens.safetensors")
    b, mb = _load_tensors(boundary, nw / "tokens.safetensors")
    ha, da = split_safetensors(boundary.read_bytes(fz / "tokens.safetensors"))
    hb, db = split_safetensors(boundary.read_bytes(nw / "tokens.safetensors"))
    ids = json.loads(ma.get("row_ids", "[]"))
    rec = {
        "frozen_sha256": shas[0]["tokens.safetensors"],
        "new_sha256": shas[1]["tokens.safetensors"],
        "file_identical_informational": shas[0]["tokens.safetensors"]
        == shas[1]["tokens.safetensors"],
        "header_metadata_identical": ha.pop("__metadata__", None) == hb.pop("__metadata__", None),
        "tensor_headers_identical": ha == hb,
        "data_section_sha256": fg.sha256_bytes(da),
        "data_section_sha256_new": fg.sha256_bytes(db),
        "data_section_identical": da == db,
        "row_ids_identical": ma.get("row_ids") == mb.get("row_ids"),
        "columns_identical": ma.get("columns") == mb.get("columns"),
        "shapes": {k: list(v.shape) for k, v in a.items()},
        "shapes_identical": {k: list(v.shape) for k, v in a.items()}
        == {k: list(v.shape) for k, v in b.items()},
        "tensor_sha256": {k: ft.rf._tensor_hash(v) for k, v in a.items()},
        "tensor_sha256_new": {k: ft.rf._tensor_hash(v) for k, v in b.items()},
    }
    X_rows = []
    if rec["shapes_identical"] and rec["row_ids_identical"]:
        diff = (_bits(a["X"]) != _bits(b["X"])).reshape(a["X"].shape[0], -1).any(1)
        X_rows = [ids[i] for i in torch.nonzero(diff).flatten().tolist()]
        rec["presence_identical"] = bool(torch.equal(a["presence"], b["presence"]))
        rec["row_eligible_identical"] = bool(torch.equal(a["row_eligible"], b["row_eligible"]))
    else:
        rec["presence_identical"] = rec["row_eligible_identical"] = False
    rec["X_rows_differing"] = X_rows
    rec["n_rows"] = len(ids)
    rec["identical"] = bool(
        rec["header_metadata_identical"]
        and rec["tensor_headers_identical"]
        and rec["data_section_identical"]
        and rec["row_ids_identical"]
        and rec["columns_identical"]
        and rec["shapes_identical"]
        and not X_rows
        and rec["presence_identical"]
        and rec["row_eligible_identical"]
    )
    return rec


def _lines(boundary, path: Path) -> list:
    return boundary.read_bytes(path).decode().splitlines(keepends=True)


def compare_rows(boundary, fz: Path, nw: Path) -> dict:
    def load(d):
        out = []
        for line in _lines(boundary, d / "rows.jsonl"):
            r = fg.strict_json_loads(line)
            out.append((r["id"], ft.canonical_json({k: v for k, v in r.items() if k != "seconds"})))
        return out

    a, b = load(fz), load(nw)
    ida, idb = [i for i, _ in a], [i for i, _ in b]
    differing = (
        sorted(i for (i, x), (_, y) in zip(a, b) if x != y)
        if ida == idb
        else sorted(set(ida) ^ set(idb))
    )
    return {
        "n_rows": len(a),
        "ids_identical_in_order": ida == idb,
        "keys_compared": sorted(set(ROW_KEYS) - {"seconds"}),
        "rows_differing": differing,
        "deterministic_sha256": fg.sha256_bytes(b"".join(x for _, x in a)),
        "deterministic_sha256_new": fg.sha256_bytes(b"".join(x for _, x in b)),
        "identical": ida == idb and not differing,
    }


def compare_bytes(name, shas, fz=None, nw=None, boundary=None) -> dict:
    rec = {"frozen_sha256": shas[0][name], "new_sha256": shas[1][name]}
    rec["identical"] = rec["frozen_sha256"] == rec["new_sha256"]
    if name == "detail.jsonl" and boundary is not None:
        la, lb = _lines(boundary, fz / name), _lines(boundary, nw / name)

        def lid(x):
            return fg.strict_json_loads(x)["id"]

        rec["n_rows"] = len(la)
        rec["rows_differing"] = (
            sorted(lid(x) for x, y in zip(la, lb) if x != y)
            if len(la) == len(lb)
            else ["<row count differs>"]
        )
    return rec


def compare_kframes(boundary, fz: Path, nw: Path, shas) -> dict:
    rec = compare_bytes("kframes.safetensors", shas)
    a, _ = _load_tensors(boundary, fz / "kframes.safetensors")
    b, _ = _load_tensors(boundary, nw / "kframes.safetensors")
    names = sorted(set(a) | set(b))
    rec["n_tensors"] = len(names)
    rec["tensors_differing"] = [
        n
        for n in names
        if n not in a
        or n not in b
        or a[n].shape != b[n].shape
        or a[n].dtype != b[n].dtype
        or not torch.equal(a[n].contiguous().view(torch.uint8), b[n].contiguous().view(torch.uint8))
    ]
    rec["identical"] = rec["identical"] and not rec["tensors_differing"]
    return rec


def compare_pricing(boundary, fz: Path, nw: Path) -> dict:
    a = fg.strict_json_loads(boundary.read_bytes(fz / "pricing.json").decode())
    b = fg.strict_json_loads(boundary.read_bytes(nw / "pricing.json").decode())
    differing = [k for k in PRICING_KEYS if a.get(k) != b.get(k)]
    return {
        "keys_compared": list(PRICING_KEYS),
        "keys_differing": differing,
        "identical": not differing,
    }


def compare_runs(
    frozen_manifest: Path, frozen_dir: Path, new_manifest: Path, new_dir: Path, boundary
) -> dict:
    fm_bytes = boundary.read_bytes(frozen_manifest)
    nm_bytes = boundary.read_bytes(new_manifest)
    fm, nm = fg.strict_json_loads(fm_bytes.decode()), fg.strict_json_loads(nm_bytes.decode())
    fz, nw = Path(frozen_dir), Path(new_dir)
    shas = (_authenticate(boundary, fz, fm, "frozen"), _authenticate(boundary, nw, nm, "new"))
    files = {
        "tokens.safetensors": compare_tokens(boundary, fz, nw, shas),
        "rows.jsonl": compare_rows(boundary, fz, nw),
        "detail.jsonl": compare_bytes("detail.jsonl", shas, fz, nw, boundary),
        "aq.json": compare_bytes("aq.json", shas),
        "kframes.json": compare_bytes("kframes.json", shas),
        "kframes.safetensors": compare_kframes(boundary, fz, nw, shas),
        "pricing.json": compare_pricing(boundary, fz, nw),
    }
    diffs = manifest_differences(fm, nm)
    files["manifest"] = {
        "frozen_sha256": fg.sha256_bytes(fm_bytes),
        "new_sha256": fg.sha256_bytes(nm_bytes),
        "fields_differing": diffs,
        "identical": not diffs,
    }
    return {
        "schema": SCHEMA,
        "basis": "PREREG_H1_TF A5 chair ruling on the F12 token requalification rerun (12 Nov 2026, 09:19 UTC), (c)",
        "frozen": {"manifest": str(frozen_manifest), "dir": str(fz)},
        "new": {"manifest": str(new_manifest), "dir": str(nw)},
        "authenticated": "every compared file of both runs equals its own manifest's sha256 (checked before comparing)",
        "files": files,
        "excluded": list(EXCLUDED),
        "identical": all(f["identical"] for f in files.values()),
        "values_printed": "none: equality results, counts, ids, tensor names and hashes only",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--frozen-manifest", type=Path, default=FROZEN_MANIFEST)
    p.add_argument("--frozen-dir", type=Path, default=FROZEN_DIR)
    p.add_argument("--new-manifest", type=Path, required=True)
    p.add_argument("--new-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists")
    boundary = read_boundary(a.new_manifest)
    if a.frozen_manifest == FROZEN_MANIFEST:
        boundary.sha256_file(FROZEN_MANIFEST, sha256=FROZEN_MANIFEST_SHA256)
    rep = compare_runs(a.frozen_manifest, a.frozen_dir, a.new_manifest, a.new_dir, boundary)
    rep["read_boundary"] = boundary.record()
    ft.write_json(a.out, rep)
    summary = {k: v["identical"] for k, v in rep["files"].items()}
    print(json.dumps({"identical": rep["identical"], "files": summary}, indent=1))
    return 0 if rep["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
