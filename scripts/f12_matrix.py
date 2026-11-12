#!/usr/bin/env python3
"""F12f: the pinned tokens-to-matrix builder of the real ``f12-token-matrix-v1`` matrix (PREREG_H1_TF chair ruling on F12e
(b); F12.3 comparator; F5 schema pins "Augmented comparators"; F12a.2(h) zero-edit input; A5.8.3 order).

Inputs, each verified before it is read:

* the frozen token output (``docs/prereg/a5_manifests/f12/tokens_manifest.json``, sha256 pinned; its seven files must
  equal the manifest's sha256): tokens X [270, 186, 42] fp32, presence, row eligibility, and per row a, q
  (``rows.jsonl``);
* the gauge-transformed tokens (``docs/prereg/a5_manifests/f12/gauge_tokens_manifest.json``, written by
  ``scripts/f12_gauge_tokens.py``; status complete, built on the same token manifest; every file's sha256 equals the
  manifest's); its presence and columns 16-41 must equal the frozen tokens';
* the existing pooled v4 files ``data/features/qwen3.5-0.8b/<row id>/pooled.safetensors`` (``stats`` [1242] then ``svd``
  [136], computed by the pinned pooling code ``wsp.features.pool`` (feature_version 4, 4 buckets) from the effective
  edit delta relative to the base, ``wsp.features.extract``). Every row's ``features.json``, ``features.safetensors``
  and ``pooled.safetensors`` must equal the sha256 pinned in the A5 snapshot (``docs/prereg/a5_manifests/snapshot.json``
  ``content.feature_manifest``, itself sha256 pinned), and ``features.json`` must be edit-derived only (the
  extractor's key set; edit id = row id; feature_version 4, 4 buckets, 24 layers, delta relative to the base, effective
  bf16) with changed supported tensors equal to the frozen token presence.

Matrix (``scripts/f12_simulation.py`` schema, ``synthetic`` false): rows in population id order; component, family,
origin (purchasable / existing) and eligibility from population.json and the token output; ``sum_a`` / ``sum_q`` the
row's a and q (A5.8.3 Sum a, Sum q); ``v4`` [rows, 1378] = pooled stats then spectrum; one token per present (row,
slot) in row then A5.8.3 slot order (the frozen 42 columns, fp32 values held in fp64); ``tokens_gauge`` aligned with
``tokens``; ``x0_v4`` [1378] = the pinned pooling code on the zero delta (no changed tensor). The StatNN comparator
input (``f12_simulation.statnn_features``) is v4 followed by 186 x (42 token columns + presence), zero padding,
flattened slot-major: 1,378 + 7,998 = 9,376 columns. Any non-finite value fails closed (nothing is imputed).

    python scripts/f12_matrix.py   # writes the matrix under /home/ubuntu/wsp-data/runs/a5/f12_inputs/matrix and the
                                   # committed docs/prereg/a5_manifests/f12/matrix_manifest.json

CPU only, seconds. No evaluation record, TF-v1 score, noise payload or outcome is opened; nothing under data/evals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import f12_qualify as fq
from scripts import f12_simulation as fs

SCHEMA = "f12f-token-matrix-manifest-v1"
SNAPSHOT = ROOT / "docs/prereg/a5_manifests/snapshot.json"
SNAPSHOT_SHA256 = "1b2af25cde1bd7ad0427fec2d0ee22d34f75651ba3fb979176464f8fad701da3"
FEATURES_DIR = ROOT / "data/features/qwen3.5-0.8b"
FEATURE_FILES = ("features.json", "features.safetensors", "pooled.safetensors")
GAUGE_MANIFEST = ROOT / "docs/prereg/a5_manifests/f12/gauge_tokens_manifest.json"
GAUGE_FILES = ("gauge_tokens.safetensors", "rows.jsonl", "detail.jsonl", "pricing.json")
MATRIX_MANIFEST = fs.MATRIX_MANIFEST
DEFAULT_OUT = Path("/home/ubuntu/wsp-data/runs/a5/f12_inputs/matrix/f12_token_matrix.npz")
N_LAYERS, BUCKETS = 24, 4
FEATURES_META_KEYS = frozenset({
    "base_commit", "base_model", "base_revision", "config", "config_hash", "delta_key_mapping", "delta_relative_to",
    "edit_id", "eligibility", "family", "identity_hash", "n_layers", "norms", "params", "parent_id",
    "rounding_stats", "seed", "stat_names", "step_semantics", "stored_delta_format", "tensors", "unchanged_tensors",
})  # fmt: skip
CODE_FILES = (
    "scripts/f12_matrix.py",
    "scripts/f12_simulation.py",
    "scripts/f12_qualify.py",
    "scripts/f12_gauge_tokens.py",
    "src/wsp/features/pool.py",
    "src/wsp/features/extract.py",
)


class SourceMismatch(ValueError):
    """A source differs from its pinned identity, or a value is non-finite (fail closed)."""


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------------------------------- v4


def v4_widths():
    from wsp.features import pool

    return pool.pooled_dims(BUCKETS)


def v4_vector(path) -> np.ndarray:
    """[1378] fp64: the pooled ``stats`` (1,242) then ``svd`` (136) of one ``pooled.safetensors``."""
    from safetensors import safe_open

    n_stats, n_spec = v4_widths()
    with safe_open(str(path), framework="np") as f:
        if set(f.keys()) != {"stats", "svd"}:
            raise SourceMismatch(f"{path}: keys {sorted(f.keys())} are not the pooled stats / svd")
        st, sv = f.get_tensor("stats"), f.get_tensor("svd")
    if st.shape != (n_stats,) or sv.shape != (n_spec,):
        raise SourceMismatch(f"{path}: width {st.shape} + {sv.shape} is not the pinned pooling code's "
                             f"({n_stats}, {n_spec})")  # fmt: skip
    v = np.concatenate([st, sv]).astype(np.float64)
    if not np.isfinite(v).all():
        raise SourceMismatch(f"{path}: non-finite v4 statistics")
    return v


def x0_v4() -> np.ndarray:
    """F12a.2(h): the v4 statistics of the zero delta by the pinned pooling code (no changed tensor is pooled)."""
    from wsp.features import pool

    s, p = pool.pool_features([], N_LAYERS, BUCKETS)
    v = np.concatenate([s.numpy(), p.numpy()]).astype(np.float64)
    if v.shape != (sum(v4_widths()),):
        raise SourceMismatch("zero-edit v4 width differs from the pinned pooling code")
    if not np.isfinite(v).all():
        raise SourceMismatch("non-finite zero-edit v4 statistics x0")
    return v


def snapshot_feature_content(path=SNAPSHOT, expected=SNAPSHOT_SHA256) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SourceMismatch(f"{path}: sha256 is not the pinned A5 snapshot {expected}")
    fmn = json.loads(raw)["content"]["feature_manifest"]
    if tuple(fmn["files"]) != FEATURE_FILES:
        raise SourceMismatch("snapshot feature manifest lists other files")
    return fmn["content"]


def check_features_meta(meta, row_id, present, slot_names):
    extra = set(meta) - FEATURES_META_KEYS
    if extra:
        raise SourceMismatch(f"{row_id}: features.json has unexpected keys {sorted(extra)} (not edit-derived)")
    cfg = meta.get("config") or {}
    checks = {
        "edit_id": meta.get("edit_id") == row_id,
        "feature_version": cfg.get("feature_version") == 4,
        "buckets": cfg.get("buckets") == BUCKETS,
        "delta": cfg.get("delta") == "effective_bf16",
        "n_layers": meta.get("n_layers") == N_LAYERS,
        "delta_relative_to": meta.get("delta_relative_to") == "base",
    }
    for k, ok in checks.items():
        if not ok:
            raise SourceMismatch(f"{row_id}: features.json {k} is not the registered v4 extraction")
    slot_set = set(slot_names)
    changed = {t["name"] for t in meta.get("tensors", []) if t["name"] in slot_set}
    if changed != set(present):
        raise SourceMismatch(f"{row_id}: features.json changed supported tensors differ from the token presence")


def verify_v4_sources(row_ids, presence, features_dir, content):
    """{row id: v4 vector} and {row id: [sha256 of the three feature files]} after every check passes."""
    slot_names = [s[2] for s in fs.SLOTS]
    v4, shas = {}, {}
    for i, rid in enumerate(row_ids):
        if rid not in content:
            raise SourceMismatch(f"{rid}: not in the A5 snapshot feature manifest")
        d = Path(features_dir) / rid
        got = [sha256_file(d / n) for n in FEATURE_FILES]
        for n, g, e in zip(FEATURE_FILES, got, content[rid]):
            if g != e:
                raise SourceMismatch(f"{rid}/{n}: sha256 {g} differs from the A5 snapshot's {e}")
        meta = json.loads((d / "features.json").read_text())
        check_features_meta(meta, rid, [slot_names[s] for s in np.flatnonzero(presence[i])], slot_names)
        v4[rid] = v4_vector(d / "pooled.safetensors")
        shas[rid] = got
    return v4, shas


# ------------------------------------------------------------------------------------------------ gauge


def load_gauge(gman, token_manifest_sha256):
    from safetensors import safe_open

    if gman.get("status") != "complete":
        raise SourceMismatch("the gauge-token manifest is not complete")
    if (gman.get("inputs") or {}).get("token_manifest", {}).get("sha256") != token_manifest_sha256:
        raise SourceMismatch("the gauge tokens were built on another token manifest")
    out = Path(gman["out_dir"])
    files = gman.get("files") or {}
    got = {}
    for n in GAUGE_FILES:
        if n not in files or not (out / n).exists():
            raise SourceMismatch(f"{n}: missing from the gauge-token output or manifest")
        got[n] = sha256_file(out / n)
        if got[n] != files[n]:
            raise SourceMismatch(f"{n}: sha256 {got[n]} differs from the gauge-token manifest's {files[n]}")
    with safe_open(str(out / "gauge_tokens.safetensors"), framework="np") as f:
        meta = f.metadata() or {}
        Xg, P = f.get_tensor("X_gauge"), f.get_tensor("presence")
    return {"X_gauge": Xg, "presence": P, "row_ids": json.loads(meta.get("row_ids", "[]")), "files": got}


# ----------------------------------------------------------------------------------------------- matrix


def matrix_from_tokens(tok, gauge, population, v4_by_id, x0):
    """The real ``f12-token-matrix-v1`` matrix from verified inputs (see the module docstring)."""
    X, P = np.asarray(tok["X"]), np.asarray(tok["presence"]).astype(bool)
    ids = list(tok["row_ids"])
    if ids != sorted(r["id"] for r in population["rows"]):
        raise SourceMismatch("token rows are not the population in id order")
    if list(gauge["row_ids"]) != ids:
        raise SourceMismatch("gauge-token rows differ from the token rows")
    Xg, Pg = np.asarray(gauge["X_gauge"]), np.asarray(gauge["presence"]).astype(bool)
    if Xg.shape != X.shape or not np.array_equal(Pg, P):
        raise SourceMismatch("gauge-token presence differs from the frozen token presence")
    if not np.array_equal(Xg[..., 16:], X[..., 16:]):
        raise SourceMismatch("gauge tokens differ from the frozen tokens in columns 16-41")
    if X[~P].any() or Xg[~P].any():
        raise SourceMismatch("absent slots are not zero padding")
    if not (np.isfinite(X).all() and np.isfinite(Xg).all()):
        raise SourceMismatch("non-finite token values")
    by_id = {r["id"]: r for r in population["rows"]}
    sum_a, sum_q = [], []
    for rid in ids:
        r = tok["rows"][rid]
        a, q = r.get("a"), r.get("q")
        if a is None or q is None or not (np.isfinite(a) and np.isfinite(q)):
            raise SourceMismatch(f"{rid}: non-finite a or q")
        sum_a.append(float(a))
        sum_q.append(float(q))
    rr, ss = np.nonzero(P)
    m = {
        "schema": fs.TOKEN_SCHEMA,
        "synthetic": False,
        "row_ids": np.asarray(ids),
        "component": np.asarray([by_id[i]["component_id"] for i in ids]),
        "family": np.asarray([by_id[i]["family"] for i in ids]),
        "origin": np.asarray(["purchasable" if by_id[i]["source"] == "purchasable" else "existing" for i in ids]),
        "eligible": np.asarray(tok["row_eligible"]).astype(bool),
        "sum_a": np.asarray(sum_a, dtype=np.float64),
        "sum_q": np.asarray(sum_q, dtype=np.float64),
        "v4": np.stack([v4_by_id[i] for i in ids]).astype(np.float64),
        "token_row": rr.astype(np.int64),
        "token_slot": ss.astype(np.int64),
        "token_role": np.asarray([fs.SLOTS[s][1] for s in ss], dtype=np.int64),
        "tokens": X[rr, ss].astype(np.float64),
        "tokens_gauge": Xg[rr, ss].astype(np.float64),
        "x0_v4": np.asarray(x0, dtype=np.float64),
    }
    try:
        fs.validate_matrix(m)
    except ValueError as exc:
        raise SourceMismatch(str(exc)) from exc
    return m


def widths():
    n_stats, n_spec = v4_widths()
    tb = fs.N_SLOTS * (fs.N_COLS + 1)
    return {"v4": n_stats + n_spec, "v4_stats": n_stats, "v4_spectrum": n_spec, "token_block": tb,
            "statnn_total": n_stats + n_spec + tb, "token_columns": fs.N_COLS, "slots": fs.N_SLOTS}  # fmt: skip


def assemble(*, token_manifest, token_manifest_sha256, gauge_manifest, gauge_manifest_sha256, population,
             population_sha256, feature_content, features_dir):  # fmt: skip
    """Verify every source and build the matrix; returns (matrix, facts for the manifest)."""
    if (token_manifest.get("population_frozen") or {}).get("population_json_sha256") != population_sha256:
        raise SourceMismatch("population.json differs from the frozen token run's")
    out_dir = Path(token_manifest["out_dir"])
    tok_files = fq.verify_token_files(token_manifest, out_dir)
    tok = fq.load_output(out_dir)
    slot_sha = hashlib.sha256(json.dumps([s[2] for s in fs.SLOTS]).encode()).hexdigest()
    if (token_manifest.get("slots") or {}).get("names_sha256") != slot_sha:
        raise SourceMismatch("the token run's slot names differ from the A5.8.3 slot table")
    gauge = load_gauge(gauge_manifest, token_manifest_sha256)
    v4, shas = verify_v4_sources(tok["row_ids"], tok["presence"], features_dir, feature_content)
    x0 = x0_v4()
    m = matrix_from_tokens(tok, gauge, population, v4, x0)
    index = sorted([rid, *shas[rid]] for rid in tok["row_ids"])
    facts = {
        "widths": widths(),
        "counts": {"rows": len(tok["row_ids"]), "tokens": len(m["tokens"]),
                   "eligible": int(m["eligible"].sum()), "components": len(set(m["component"].tolist()))},  # fmt: skip
        "sources": {
            "token_manifest": {"sha256": token_manifest_sha256, "files": tok_files},
            "gauge_manifest": {"sha256": gauge_manifest_sha256, "files": gauge["files"]},
            "population_sha256": population_sha256,
            "features": {
                "dir": str(features_dir),
                "files": list(FEATURE_FILES),
                "rows_content_index_sha256": _sha_json(index),
                "rows_content_index_convention": "sha256 of json.dumps(sorted [[row id, sha256 of features.json, "
                "features.safetensors, pooled.safetensors]], sort_keys=True) over the 270 rows",
                "pooled_sha256": {rid: shas[rid][2] for rid in tok["row_ids"]},
            },
        },
        "v4_gauge_tokens_change": {
            "k_block_max_abs_change": float(np.max(np.abs(m["tokens_gauge"][:, :16] - m["tokens"][:, :16])))
            if len(m["tokens"]) else 0.0,
            "tokens_changed": int(np.any(m["tokens_gauge"][:, :16] != m["tokens"][:, :16], axis=1).sum()),
        },
        "x0": {"nonzero": int(np.count_nonzero(x0)), "all_zero": not x0.any()},
    }
    return m, facts


def code_identity():
    out = {f: sha256_file(ROOT / f) for f in CODE_FILES}
    try:
        out["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *CODE_FILES], text=True, cwd=ROOT)
        out["code_files_dirty"] = bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        out["git_sha"] = None
    return out


def write_outputs(m, facts, npz_path, manifest_path, sources=None):
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    fs.save_matrix(npz_path, m)
    back = fs.load_matrix(npz_path)
    msha = fs.matrix_sha256(back)
    if msha != fs.matrix_sha256(m):
        raise SourceMismatch("the saved matrix does not reload to the built matrix")
    fsha = sha256_file(npz_path)
    doc = {
        "schema": SCHEMA,
        "basis": "PREREG_H1_TF chair ruling on F12e (b); F12.3 comparator; F5 schema pins 'Augmented comparators'; "
        "F12a.2(h) zero-edit input; A5.8.3 slot order",
        "matrix_sha256": msha,
        "file_sha256": fsha,
        "matrix": {
            "path": str(npz_path),
            "schema": fs.TOKEN_SCHEMA,
            "synthetic": False,
            "file_sha256": fsha,
            "matrix_sha256": msha,
            "matrix_sha256_rule": "scripts/f12_simulation.py matrix_sha256 (keys, dtypes, shapes and bytes of every "
            "matrix array including tokens_gauge and x0_v4)",
            "x0_v4_sha256": hashlib.sha256(np.ascontiguousarray(m["x0_v4"]).tobytes()).hexdigest(),
            "v4_sha256": hashlib.sha256(np.ascontiguousarray(m["v4"]).tobytes()).hexdigest(),
            "tokens_gauge_sha256": hashlib.sha256(np.ascontiguousarray(m["tokens_gauge"]).tobytes()).hexdigest(),
        },
        "widths": facts["widths"],
        "layout": {
            "v4": "pooled.safetensors stats (1,242: global mean then 17 types x 4 layer buckets x 18 stats) then svd "
            "(136: 17 types x mean top-8 singular values), wsp.features.pool feature_version 4",
            "statnn": "v4 then, per A5.8.3 slot (186, layer then sorted role), the 42 masked WM-R token columns and a "
            "presence column; absent slots zero (f12_simulation.statnn_features)",
            "x0": "pinned pooling code on the zero delta: all slots absent, token columns 0, presence 0 (F12a.2(h))",
            "synthetic_stand_in": f"the {fs.V4_COLS}-column v4_stats stand-in is for synthetic runs only; governed "
            "runs refuse it and accept only the matrix named here",
        },
        **{k: v for k, v in facts.items() if k != "widths"},
        "sources_extra": sources or {},
        "outcomes_read": "none: the frozen token outputs, the gauge-token outputs, population.json, the A5 snapshot "
        "and the edit-derived v4 feature files only; no evaluation record, outcome, noise payload or data/evals file",
        "code": code_identity(),
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--manifest", type=Path, default=MATRIX_MANIFEST)
    p.add_argument("--gauge-manifest", type=Path, default=GAUGE_MANIFEST)
    a = p.parse_args(argv)
    if not str(a.out.resolve()).startswith("/home/ubuntu/wsp-data/"):
        raise SystemExit("the matrix is written under /home/ubuntu/wsp-data only")
    man, msha = fq._manifest()
    graw = a.gauge_manifest.read_bytes()
    pop_raw = fq.POPULATION.read_bytes()
    m, facts = assemble(
        token_manifest=man, token_manifest_sha256=msha, gauge_manifest=json.loads(graw),
        gauge_manifest_sha256=hashlib.sha256(graw).hexdigest(), population=json.loads(pop_raw),
        population_sha256=hashlib.sha256(pop_raw).hexdigest(), feature_content=snapshot_feature_content(),
        features_dir=FEATURES_DIR,
    )  # fmt: skip
    doc = write_outputs(m, facts, a.out, a.manifest, sources={
        "token_manifest_path": str(fq.TOKEN_MANIFEST.relative_to(ROOT)),
        "gauge_manifest_path": str(Path(a.gauge_manifest).resolve().relative_to(ROOT)),
        "snapshot": {"path": str(SNAPSHOT.relative_to(ROOT)), "sha256": SNAPSHOT_SHA256},
        "population_path": str(fq.POPULATION.relative_to(ROOT)),
    })  # fmt: skip
    print(json.dumps({"matrix_sha256": doc["matrix_sha256"], "file_sha256": doc["file_sha256"],
                      "widths": doc["widths"], "manifest": str(a.manifest)}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
