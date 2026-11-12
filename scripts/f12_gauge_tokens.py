#!/usr/bin/env python3
"""F12f: the registered gauge-transformed WM-R tokens (PREREG_H1_TF chair ruling on F12e (a); F12.3; F12a.2(i);
F5 schema pins "Gauge audit"). A pinned extension of the F12 token builder; the frozen run's code path
(``scripts/f12_tokens.py``) is not changed and its outputs are not rewritten.

For every population row and every present token (supported matrix tensor t with D_t != 0; each has balanced low-rank
factors (B, A) at the recorded rank, F5 pins "Increment, rank and energy"):

* the realised increment D is rebuilt from the stored zoo bytes exactly as the frozen run did
  (``wsp.zoo.apply.load_entry_delta`` + ``a5_power_fields.edit_increments``), and its hash must equal the frozen
  ``rows.jsonl`` ``increment_sha256``; the set of nonzero supported tensors must equal the frozen presence;
* the full thin fp32 SVD and the balanced factors B = U_r diag(sigma_r^1/2), A = diag(sigma_r^1/2) V_r^T are recomputed
  with the frozen builder's own functions (``f12_tokens.balanced_factors``), and the K block M = (Lambda_G^1/2 U_G^T B)
  (A U_A Lambda_A^1/2) from the frozen K frames (``kframes.safetensors``) must reproduce the frozen token's columns 0-15
  bitwise (``REPRO_RTOL`` = 0; the frozen run's six threads are required, since near-tied truncations make the K block
  sensitive to thread-order rounding), so the gauge acts on exactly the factors behind the frozen tokens;
* the gauge (F5 pins): Q = H diag(d), H the Haar r x r matrix from seed('20261110:lol-gauge:', '<edit id>:<exact
  tensor name>') via ``torch.Generator('cpu').manual_seed``, ``randn(r, r)`` in fp64 and positive-diagonal QR (a zero
  diagonal sign counts as +1); d_j = 2^((j mod 3) - 1) for j = 0..r-1 (chair ruling on F12f (b); d = (1/2, 1, 2, ...)), so
  cond(Q) <= 4; (B, A) -> (BQ, Q^-1 A) in fp64 (Q^-1 = diag(1/d) H^T), cast to fp32. ``<edit id>`` is the population
  row id (the canonical zoo edit id; an alias row keeps its own id); r is the recorded rank capped at min(out, in)
  (the frozen ``r_eff``);
* the gauge-transformed token is the frozen token with columns 0-15 replaced by the K block of (BQ, Q^-1 A) (fp32);
  columns 16-41 are functions of D_t alone (spectrum, layer, field and probe columns) and are copied. A K block masked
  by ``k_cutoff_tie`` stays zero.

Outputs (``--out-dir`` under ``/home/ubuntu/wsp-data/runs/a5/f12_inputs``): ``gauge_tokens.safetensors`` (``X_gauge``
[rows, 186, 42] fp32 and ``presence`` [rows, 186] uint8, population id order, the frozen layout), ``rows.jsonl`` and
``detail.jsonl`` (per row and per tensor: seed, Q sha256, cond(Q), r, change of the K block), ``pricing.json``.
``--manifest`` (``docs/prereg/a5_manifests/f12/gauge_tokens_manifest.json``) records hashes and source identities.
Pricing (as f12_tokens, from the first ``--price-rows`` rows): naive (rows x mean row seconds) and tensor-weighted
(seconds per present tensor x all present tensors) projections plus setup; the run stops (``price_exceeds_cap``) if
the larger exceeds ``--max-wall-hours``.

    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python scripts/f12_gauge_tokens.py --threads 6 --max-wall-hours 10

CPU only. Reads: the committed frozen token manifest (sha256 pinned) and, after their sha256 equal it, the frozen
token outputs; population.json; the stored zoo deltas and ``edit.json``; the pinned base model. No evaluation
record, TF-v1 score, noise payload or outcome is opened; nothing under ``data/evals`` is read.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import a5_power_fields as pf
from scripts import f12_tokens as ft
from scripts.a5_build_panels import domain_seed

SCHEMA = "f12-gauge-tokens-v1"
GAUGE_DOMAIN = "20261110:lol-gauge:"
# The recomputed K block of the balanced factors must reproduce the frozen columns 0-15 bitwise (REPRO_RTOL = 0). A
# tolerance is not used: rank-r truncations next to a near-tie (e.g. random_dir rows at rank 1 with sigma_1 / sigma_2 =
# 1.02) move the K block by ~1e-5 relative under thread-order rounding of the fp32 SVD, which is the order of the
# audit's own 1e-4 tolerance; the gauge must act on exactly the factors behind the frozen tokens. The frozen run's
# numerical environment (six threads) is therefore required (``--threads 6``).
REPRO_RTOL = 0.0
FROZEN_THREADS = 6  # the builder refuses any other thread count (chair ruling on F12f (a))
GAUGE_INDEX_BASE = 0  # chair ruling on F12f (b): j = 0, ..., r - 1 (supersedes the 1-based F12f pricing variant)
DEFAULT_MAX_WALL_HOURS = 24.0  # chair ruling on F12f (a): the self-pricing stop at 24 h wall
GAUGE_ATOL, GAUGE_RTOL = 1e-5, 1e-4  # F5 pins: |delta f| <= 1e-5 + 1e-4 |f| nats per row (the runner applies it)
ALLOWED_OUT = ft.ALLOWED_OUT
DEFAULT_OUT = ALLOWED_OUT / "gauge_tokens"
DEFAULT_MANIFEST = ROOT / "docs/prereg/a5_manifests/f12/gauge_tokens_manifest.json"
POPULATION = ft.DEFAULT_POPULATION
OUTPUT_FILES = ("gauge_tokens.safetensors", "rows.jsonl", "detail.jsonl", "pricing.json")
CODE_FILES = (
    "scripts/f12_gauge_tokens.py",
    "scripts/f12_tokens.py",
    "scripts/f12_qualify.py",
    "scripts/a5_power_fields.py",
    "scripts/a5_build_panels.py",
    "src/wsp/zoo/apply.py",
    "src/wsp/features/response_field.py",
)


class SourceMismatch(ValueError):
    """A recomputed source differs from the frozen token run (fail closed; nothing is repaired)."""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------------------------- gauge


def gauge_seed(edit_id: str, tensor: str) -> int:
    return domain_seed(GAUGE_DOMAIN, f"{edit_id}:{tensor}")


def gauge_d(r: int) -> torch.Tensor:
    """d_j = 2^((j mod 3) - 1), j = 0..r-1 (chair ruling on F12f (b)): d = (1/2, 1, 2, 1/2, ...) (fp64)."""
    return torch.tensor([2.0 ** ((j % 3) - 1) for j in range(GAUGE_INDEX_BASE, GAUGE_INDEX_BASE + r)],
                        dtype=torch.float64)  # fmt: skip


def gauge_matrix(edit_id: str, tensor: str, r: int) -> dict:
    """Q = H diag(d) with H Haar r x r (positive-diagonal QR of fp64 randn) and its exact-form inverse."""
    seed = gauge_seed(edit_id, tensor)
    gen = torch.Generator("cpu").manual_seed(seed)
    Z = torch.randn(r, r, generator=gen, dtype=torch.float64)
    H, R = torch.linalg.qr(Z)
    s = torch.sign(torch.diagonal(R))
    s[s == 0] = 1.0
    H = H * s[None, :]
    d = gauge_d(r)
    Q = H * d[None, :]
    Q_inv = (1.0 / d)[:, None] * H.T
    sv = torch.linalg.svdvals(Q)
    return {
        "seed": seed,
        "H": H,
        "Q": Q,
        "Q_inv": Q_inv,
        "cond": float(sv.max() / sv.min()),
        "sha256": sha256_bytes(Q.contiguous().numpy().tobytes()),
    }


def gauge_factors(B, A, g):
    """(BQ, Q^-1 A) in fp64, cast to fp32."""
    return (B.double() @ g["Q"]).float(), (g["Q_inv"] @ A.double()).float()


def _masked(frames):
    return any("k_cutoff_tie" in f["flags"] for f in frames)


def tensor_gauge(D, frozen_token, r, frames, edit_id, name):
    """Gauge-transformed token [42] fp32 of one present tensor and its per-tensor record.

    Recomputes the frozen builder's SVD and balanced factors, requires the frozen K block (columns 0-15) to reproduce
    bitwise, then replaces columns 0-15 with the K block of (BQ, Q^-1 A); columns 16-41 are copied."""
    frozen = np.asarray(frozen_token, dtype=np.float32)
    U, S, Vh = torch.linalg.svd(D.float(), full_matrices=False)
    if not (torch.isfinite(U).all() and torch.isfinite(S).all() and torch.isfinite(Vh).all()):
        raise SourceMismatch(f"{edit_id}:{name}: nonfinite SVD")
    B, A, binfo = ft.balanced_factors(U, S, Vh, r)
    r_eff = binfo["r_eff"]
    g = gauge_matrix(edit_id, name, r_eff)
    out = frozen.copy()
    info = {"r_eff": r_eff, "seed": g["seed"], "Q_sha256": g["sha256"], "cond": g["cond"]}
    if _masked(frames):
        if frozen[:16].any():
            raise SourceMismatch(f"{edit_id}:{name}: K block masked by k_cutoff_tie but the frozen block is nonzero")
        return out, info | {"masked": True, "k_reproduced_exact": True, "k_reproduction_max_abs_diff": 0.0,
                            "k_block_max_abs_change": 0.0}
    fG, fA = frames
    orig = ft.k_block(B, A, fG, fA).reshape(-1).numpy().astype(np.float32)
    diff = float(np.max(np.abs(orig.astype(np.float64) - frozen[:16].astype(np.float64))))
    scale = float(np.max(np.abs(frozen[:16].astype(np.float64))))
    if not np.isfinite(diff) or diff > REPRO_RTOL * scale:
        raise SourceMismatch(
            f"{edit_id}:{name}: recomputed K block differs from the frozen token (max {diff:.3e} > "
            f"{REPRO_RTOL:g} x {scale:.3e})"
        )
    Bg, Ag = gauge_factors(B, A, g)
    new = ft.k_block(Bg, Ag, fG, fA).reshape(-1).numpy().astype(np.float32)
    if not np.isfinite(new).all():
        raise SourceMismatch(f"{edit_id}:{name}: nonfinite gauge-transformed K block")
    out[:16] = new
    change = np.abs(new.astype(np.float64) - frozen[:16].astype(np.float64))
    return out, info | {
        "masked": False,
        "k_reproduced_exact": diff == 0.0,
        "k_reproduction_max_abs_diff": diff,
        "k_reproduction_scale": scale,
        "k_block_max_abs_change": float(change.max()),
        "k_block_max_rel_change": float(np.max(change / np.maximum(np.abs(frozen[:16]), 1e-30))),
    }


def row_gauge(D, row_id, X_row, P_row, slot_names, r, frames, per_tensor):
    """Gauge-transformed tokens [186, 42] of one row. ``D``: realised increments of the edited tensors;
    ``per_tensor``: the frozen detail record (``r_eff`` per tensor name)."""
    present = {slot_names[s] for s in np.flatnonzero(P_row)}
    slot_of = {n: i for i, n in enumerate(slot_names)}
    changed = {n for n, d in D.items() if n in slot_of and bool(torch.count_nonzero(d))}
    if changed != present:
        raise SourceMismatch(
            f"{row_id}: presence differs from the recomputed increment "
            f"(only frozen {sorted(present - changed)[:3]}, only recomputed {sorted(changed - present)[:3]})"
        )
    if set(per_tensor) != present:
        raise SourceMismatch(f"{row_id}: frozen per-tensor detail differs from the presence")
    Xg = np.zeros_like(np.asarray(X_row, dtype=np.float32))
    info = {}
    for name in sorted(present, key=lambda n: slot_of[n]):
        s = slot_of[name]
        d = D[name]
        r_eff = min(int(r), min(d.shape))
        if int(per_tensor[name]["r_eff"]) != r_eff:
            raise SourceMismatch(f"{row_id}:{name}: frozen r_eff {per_tensor[name]['r_eff']} != {r_eff}")
        if name not in frames:
            raise SourceMismatch(f"{row_id}:{name}: no K frame")
        Xg[s], info[name] = tensor_gauge(d, X_row[s], r, frames[name], row_id, name)
        info[name]["slot"] = s
    return Xg, info


def check_increment(row_id, D_edit, expected):
    got = pf.tensors_hash(D_edit)
    if got != expected:
        raise SourceMismatch(f"{row_id}: increment sha256 {got} differs from the frozen rows.jsonl {expected}")
    return got


# ---------------------------------------------------------------------------------------------- pricing


def pricing(*, setup_seconds, first_seconds, first_tensors, tensor_counts, cap_hours, prior_seconds=0.0):
    """Projected total wall: setup + rows already finished by an earlier attempt (``prior_seconds``) + the rows of
    this invocation (``tensor_counts``) at the first rows' rate (max of per-row and per-tensor projections)."""
    n = len(first_seconds)
    naive = setup_seconds + prior_seconds + (sum(first_seconds) / n) * len(tensor_counts)
    weighted = setup_seconds + prior_seconds + sum(first_seconds) / max(1, sum(first_tensors)) * sum(tensor_counts)
    projected = max(naive, weighted)
    return {
        "setup_seconds": setup_seconds,
        "prior_rows_seconds": prior_seconds,
        "first_rows": n,
        "first_row_seconds": list(first_seconds),
        "first_row_present_tensors": list(first_tensors),
        "n_rows": len(tensor_counts),
        "present_tensors_total": int(sum(tensor_counts)),
        "projected_wall_seconds_naive": naive,
        "projected_wall_seconds_tensor_weighted": weighted,
        "projected_wall_seconds": projected,
        "projected_wall_hours": projected / 3600,
        "cap_wall_hours": cap_hours,
        "within_cap": projected <= cap_hours * 3600,
    }


# --------------------------------------------------------------------------------------------------- IO


def check_out_dir(path):
    p = Path(path).resolve()
    if p != ALLOWED_OUT and ALLOWED_OUT not in p.parents:
        raise ValueError(f"{path}: outputs must stay under {ALLOWED_OUT}")


def load_frames(path, kdoc):
    """{tensor name: (G frame, A frame)} from the frozen ``kframes.safetensors`` with ``kframes.json`` flags."""
    from safetensors import safe_open

    frames = {}
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
        for name, rec in kdoc["tensors"].items():
            pair = []
            for side in ("G", "A"):
                ku, ks = f"{name}.{side}.U", f"{name}.{side}.sqrt_lambda"
                if ku not in keys or ks not in keys:
                    raise SourceMismatch(f"{name}: K frame {side} missing from kframes.safetensors")
                pair.append({"U": f.get_tensor(ku), "sqrt_lambda": f.get_tensor(ks),
                             "flags": list(rec[side].get("flags") or [])})  # fmt: skip
            frames[name] = tuple(pair)
    return frames


def output_hashes(out: Path) -> dict:
    return {n: pf.sha256_file(out / n) for n in OUTPUT_FILES if (out / n).exists()}


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return sha256_bytes(data)


def code_identity():
    out = {f: pf.sha256_file(ROOT / f) for f in CODE_FILES}
    try:
        out["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *CODE_FILES], text=True, cwd=ROOT)
        out["code_files_dirty"] = bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        out["git_sha"] = None
    return out


def frozen_inputs(manifest_path=None):
    """The committed frozen token manifest (sha256 pinned) and its outputs, verified file by file."""
    from scripts import f12_qualify as fq

    path = Path(manifest_path or fq.TOKEN_MANIFEST)
    raw = path.read_bytes()
    if sha256_bytes(raw) != fq.TOKEN_MANIFEST_SHA256:
        raise SourceMismatch(f"{path}: sha256 is not the frozen token manifest's {fq.TOKEN_MANIFEST_SHA256}")
    man = json.loads(raw)
    out_dir = Path(man["out_dir"])
    files = fq.verify_token_files(man, out_dir)
    return man, sha256_bytes(raw), out_dir, files


# -------------------------------------------------------------------------------------------------- run


class PriceStop(Exception):
    def __init__(self, price):
        super().__init__("priced")
        self.price = price


def _complete_lines(path):
    """{id: record} of the complete JSON lines of a progress file (a torn final line is dropped)."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and "id" in d:
            out[d["id"]] = d
    return out


def process_rows(ids, compute, out, identity, *, resume, on_new_row=None):
    """Run ``compute(row id, index) -> (gauge tokens [186, 42], presence [186], row record, per-tensor info)`` over
    ``ids``, appending each finished row to ``detail.jsonl`` (its present gauge tokens, exact fp32 values) and then
    ``rows.jsonl``. A row is finished only when both hold it. With ``resume`` the finished rows of an interrupted run
    with the same ``run_identity.json`` are reloaded and never recomputed; without it an existing output is refused.
    Returns (gauge tokens [len(ids), 186, 42] fp32, row records in ``ids`` order)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ident_path, rows_path, detail_path = out / "run_identity.json", out / "rows.jsonl", out / "detail.jsonl"
    recs, dets = {}, {}
    if resume:
        if ident_path.exists() and json.loads(ident_path.read_text()) != identity:
            raise SourceMismatch(f"{out}: run identity differs from the interrupted run's; refusing to resume")
        recs, dets = _complete_lines(rows_path), _complete_lines(detail_path)
        done = [i for i in ids if i in recs and i in dets]
        recs = {i: recs[i] for i in done}
        dets = {i: dets[i] for i in done}
        rows_path.write_text("".join(json.dumps(recs[i], sort_keys=True) + "\n" for i in done))
        detail_path.write_text("".join(json.dumps(dets[i], sort_keys=True) + "\n" for i in done))
    elif any(p.exists() and p.read_text().strip() for p in (rows_path, detail_path)):
        raise SystemExit(f"{out} already holds rows; use --resume to continue it (finished rows are kept)")
    else:
        rows_path.write_text("")
        detail_path.write_text("")
    ident_path.write_text(json.dumps(identity, sort_keys=True, indent=1) + "\n")
    Xg = np.zeros((len(ids), len(_slot_names()), ft.N_COLS), dtype=np.float32)
    for i, rid in enumerate(ids):
        for slot, tok in (dets.get(rid) or {}).get("gauge_tokens", {}).items():
            Xg[i, int(slot)] = np.asarray(tok, dtype=np.float32)
    if resume:
        log(f"resume: {len(recs)} finished rows kept, {len(ids) - len(recs)} to do")
    for i, rid in enumerate(ids):
        if rid in recs:
            continue
        row, pres, rec, info = compute(rid, i)
        Xg[i] = row
        toks = {str(s): [float(x) for x in row[s]] for s in np.flatnonzero(pres)}
        with detail_path.open("a") as f:
            f.write(json.dumps({"id": rid, "gauge_tokens": toks, "per_tensor": info}, sort_keys=True,
                               allow_nan=False) + "\n")  # fmt: skip
        with rows_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")
        recs[rid] = rec
        if on_new_row is not None:
            on_new_row(rec)
    return Xg, [recs[r] for r in ids]


def _slot_names():
    from scripts import f12_simulation as fs

    return [s[2] for s in fs.SLOTS]


def run(args):
    from safetensors import safe_open
    from safetensors.torch import save_file

    from wsp.zoo.apply import load_entry_delta

    t_start = time.time()
    torch.set_num_threads(args.threads)
    check_out_dir(args.out_dir)
    out = Path(args.out_dir)
    man, man_sha, tok_dir, tok_files = frozen_inputs()
    pop_bytes = POPULATION.read_bytes()
    if sha256_bytes(pop_bytes) != man["population_frozen"]["population_json_sha256"]:
        raise SourceMismatch("population.json differs from the frozen token run's")
    with safe_open(str(tok_dir / "tokens.safetensors"), framework="np") as f:
        meta = f.metadata() or {}
        X, P = f.get_tensor("X"), f.get_tensor("presence")
    row_ids = json.loads(meta["row_ids"])
    rows = {r["id"]: r for r in (json.loads(x) for x in (tok_dir / "rows.jsonl").read_text().splitlines() if x.strip())}
    detail = {}
    for line in (tok_dir / "detail.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            detail[d["id"]] = d["per_tensor"]
    pop = json.loads(pop_bytes)
    if row_ids != sorted(r["id"] for r in pop["rows"]) or set(rows) != set(row_ids):
        raise SourceMismatch("frozen token rows differ from the population")
    slot_names = _slot_names()
    if sha256_bytes(json.dumps(slot_names).encode()) != man["slots"]["names_sha256"]:
        raise SourceMismatch("slot names differ from the frozen token run's")
    kdoc = json.loads((tok_dir / "kframes.json").read_text())
    frames = load_frames(tok_dir / "kframes.safetensors", kdoc)
    base, base_hash = pf.load_base_state(man["base"]["state_hash"])
    zoo = Path(args.zoo)
    todo = row_ids if args.limit is None else row_ids[: args.limit]
    setup = time.time() - t_start
    identity = {
        "schema": SCHEMA,
        "token_manifest_sha256": man_sha,
        "gauge_index_base": GAUGE_INDEX_BASE,
        "threads": args.threads,
        "builder_sha256": pf.sha256_file(ROOT / "scripts/f12_gauge_tokens.py"),
        "base_state_hash": base_hash,
    }
    prior = {r["id"]: r for r in _complete_lines(out / "rows.jsonl").values()} if args.resume else {}
    prior = {k: v for k, v in prior.items() if k in _complete_lines(out / "detail.jsonl")}
    remaining = [i for i in todo if i not in prior]
    counts = [int(P[row_ids.index(i)].sum()) for i in remaining]
    log(f"setup {setup:.0f}s; {len(todo)} rows ({len(remaining)} to compute), {sum(counts)} present tensors to do")
    state = {"firsts": [], "price": None}

    def compute(rid, _n):
        i = row_ids.index(rid)
        t0, c0 = time.time(), time.process_time()
        frow = rows[rid]
        stored = load_entry_delta(zoo / frow["bytes_holder"], base)
        _W, D, _raw, edited = pf.edit_increments(base, stored, None)
        del stored, _W, _raw
        d_edit = {k: D[k] for k in edited}
        inc = check_increment(rid, d_edit, frow["increment_sha256"])
        row, info = row_gauge(d_edit, rid, X[i], P[i], slot_names, frow["rank"], frames, detail[rid])
        if not np.array_equal(row[:, 16:], X[i][:, 16:]) or not np.isfinite(row).all():
            raise SourceMismatch(f"{rid}: gauge tokens changed columns 16-41 or are nonfinite")
        del D, d_edit
        gc.collect()
        sec, cpu = time.time() - t0, time.process_time() - c0
        changes = [v["k_block_max_abs_change"] for v in info.values()]
        rec = {
            "id": rid,
            "rank": frow["rank"],
            "increment_sha256": inc,
            "n_tokens": len(info),
            "k_reproduced_exact": all(v["k_reproduced_exact"] for v in info.values()),
            "tokens_reproduced_bitwise": sum(v["k_reproduced_exact"] for v in info.values()),
            "k_reproduction_max_abs_diff": max((v["k_reproduction_max_abs_diff"] for v in info.values()), default=0.0),
            "k_block_max_abs_change": max(changes) if changes else 0.0,
            "tokens_k_block_changed": sum(c > 0 for c in changes),
            "cond_max": max((v["cond"] for v in info.values()), default=None),
            "masked": sum(v["masked"] for v in info.values()),
            "seconds": sec,
            "cpu_seconds": cpu,
        }
        log(f"{rid} r={frow['rank']} tokens={rec['n_tokens']} changed={rec['tokens_k_block_changed']} "
            f"max_change={rec['k_block_max_abs_change']:.3e} ({sec:.1f}s)")  # fmt: skip
        return row, P[i], rec, info

    def on_new_row(rec):
        if state["price"] is not None:
            return
        state["firsts"].append((rec["seconds"], rec["n_tokens"]))
        if len(state["firsts"]) == min(args.price_rows, len(remaining)):
            f = state["firsts"]
            price = pricing(setup_seconds=setup, first_seconds=[x for x, _ in f], first_tensors=[t for _, t in f],
                            tensor_counts=counts, cap_hours=args.max_wall_hours,
                            prior_seconds=sum(r["seconds"] for r in prior.values())) | {"threads": args.threads}  # fmt: skip
            state["price"] = price
            write_json(out / "pricing.json", price)
            log(f"pricing: {json.dumps(price)}")
            if not price["within_cap"] or args.price_only:
                raise PriceStop(price)

    try:
        Xg_todo, recs = process_rows(todo, compute, out, identity, resume=args.resume, on_new_row=on_new_row)
    except PriceStop as stop:
        recs = list(_complete_lines(out / "rows.jsonl").values())
        write_manifest(args, out, "priced_only", man_sha, tok_files, base_hash, stop.price, recs, None, t_start)
        if not stop.price["within_cap"]:
            raise SystemExit(f"price_exceeds_cap: projected {stop.price['projected_wall_hours']:.2f} h > "
                             f"{args.max_wall_hours} h") from None  # fmt: skip
        return 0
    complete = args.limit is None and len(recs) == len(row_ids)
    Xg = None
    if complete:
        Xg = Xg_todo
        if not np.array_equal(Xg[..., 16:], X[..., 16:]) or Xg[P == 0].any():
            raise SourceMismatch("assembled gauge tokens differ from the frozen tokens outside columns 0-15")
        save_file({"X_gauge": torch.from_numpy(Xg), "presence": torch.from_numpy(P)},
                  str(out / "gauge_tokens.safetensors"),
                  metadata={"row_ids": json.dumps(row_ids), "columns": json.dumps(ft.COLUMNS)})  # fmt: skip
    status = "complete" if complete else f"partial ({len(recs)}/{len(row_ids)})"
    write_manifest(args, out, status, man_sha, tok_files, base_hash, state["price"], recs, Xg, t_start)
    return 0


def write_manifest(args, out, status, man_sha, tok_files, base_hash, price, recs, Xg, t_start):
    from wsp.features import response_field as rf

    ru = resource.getrusage(resource.RUSAGE_SELF)
    files = output_hashes(out)
    gauge = None
    if Xg is not None:
        gauge = {
            "path": str(out / "gauge_tokens.safetensors"),
            "sha256": files["gauge_tokens.safetensors"],
            "shape": list(Xg.shape),
            "X_gauge_sha256": rf._tensor_hash(torch.from_numpy(Xg)),
        }
    changes = [r["k_block_max_abs_change"] for r in recs]
    doc = {
        "schema": SCHEMA,
        "status": status,
        "basis": "PREREG_H1_TF chair ruling on F12e (a); F12.3; F12a.2(i); F5 schema pins 'Gauge audit'",
        "construction": {
            "Q": "H diag(d); H Haar r x r: torch.Generator('cpu').manual_seed(seed('20261110:lol-gauge:', "
            "'<row id>:<exact tensor name>')), randn(r, r) fp64, QR with positive diagonal (zero sign -> +1); "
            "d_j = 2^((j mod 3) - 1), j = 0..r-1 (chair ruling on F12f (b)); r = recorded rank capped at min(out, "
            "in) (frozen r_eff); at r = 1, Q = 1/2 exactly, so (B/2, 2A) is exact in floating point and the token is "
            "bitwise unchanged",
            "factors": "balanced B = U_r diag(sigma_r^1/2), A = diag(sigma_r^1/2) V_r^T of the full thin fp32 SVD of "
            "D_t (f12_tokens.balanced_factors); (B, A) -> (BQ, Q^-1 A) in fp64 (Q^-1 = diag(1/d) H^T), cast to fp32",
            "token": "frozen token with columns 0-15 replaced by the K block of the transformed factors (frozen K "
            "frames); columns 16-41 copied (functions of D_t alone); masked K blocks stay zero",
            "edit_id": "the population row id (the canonical zoo edit id); alias rows keep their own id",
            "scope": "every population row and every present token (each present token has balanced low-rank "
            "factors); absent slots stay zero",
            "source_checks": "token manifest sha256 pinned; token files equal its sha256; population sha256; slot "
            "names; base state hash; per-row increment sha256 equals the frozen rows.jsonl; presence equals the "
            "nonzero supported tensors; r_eff equals the frozen detail; the recomputed K block reproduces the "
            "frozen columns 0-15 bitwise (frozen run's six threads)",
        },
        "inputs": {
            "token_manifest": {"path": "docs/prereg/a5_manifests/f12/tokens_manifest.json", "sha256": man_sha},
            "token_files": tok_files,
            "population_sha256": pf.sha256_file(POPULATION),
            "zoo": str(args.zoo),
            "base_state_hash": base_hash,
        },
        "gauge_tokens": gauge,
        "files": files,
        "out_dir": str(out),
        "rows_sha256": sha256_bytes(json.dumps(sorted(recs, key=lambda r: r["id"]), sort_keys=True).encode()),
        "summary": {
            "n_rows_done": len(recs),
            "n_tokens": sum(r["n_tokens"] for r in recs),
            "k_reproduced_exact_rows": sum(r["k_reproduced_exact"] for r in recs),
            "tokens_reproduced_bitwise": sum(r["tokens_reproduced_bitwise"] for r in recs),
            "k_reproduction_max_abs_diff": max((r["k_reproduction_max_abs_diff"] for r in recs), default=None),
            "tokens_k_block_changed": sum(r.get("tokens_k_block_changed", 0) for r in recs),
            "tokens_k_block_changed_by_rank": {
                str(k): [sum(r.get("tokens_k_block_changed", 0) for r in recs if r["rank"] == k),
                         sum(r["n_tokens"] for r in recs if r["rank"] == k)]
                for k in sorted({r["rank"] for r in recs})
            },  # fmt: skip
            "gauge_index_base": GAUGE_INDEX_BASE,
            "k_reproduction_rule": "recomputed K block bitwise equal to the frozen columns 0-15 (REPRO_RTOL = 0; "
            "the frozen run's six threads), else fail closed",
            "masked_tokens": sum(r["masked"] for r in recs),
            "k_block_max_abs_change": max(changes) if changes else None,
            "cond_max": max((r["cond_max"] for r in recs if r["cond_max"] is not None), default=None),
            "pricing": price,
            "timing": {
                "wall_seconds": time.time() - t_start,
                "row_seconds_total": sum(r["seconds"] for r in recs),
                "row_cpu_seconds_total": sum(r["cpu_seconds"] for r in recs),
                "process_cpu_seconds": ru.ru_utime + ru.ru_stime,
                "process_cpu_hours": (ru.ru_utime + ru.ru_stime) / 3600,
                "threads": args.threads,
            },
            "max_rss_bytes": ru.ru_maxrss * 1024,
            "charged_to": "F12 inputs (chair ruling on F12e (c)), not the 190 CPU-h simulation ceiling",
        },
        "outcomes_read": "none: the frozen token outputs (verified), population.json, stored edit bytes and the base "
        "model only; no evaluation record, TF-v1 score, noise payload or data/evals file",
        "code": code_identity(),
    }
    sha = write_json(Path(args.manifest), doc)
    log(f"wrote {args.manifest} ({status}) sha256 {sha}")
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--zoo", default=str(ROOT / "data/zoo/qwen3.5-0.8b"))
    p.add_argument("--limit", type=int)
    p.add_argument("--threads", type=int, default=FROZEN_THREADS)
    p.add_argument("--max-wall-hours", type=float, default=DEFAULT_MAX_WALL_HOURS)
    p.add_argument("--resume", action="store_true", help="continue an interrupted run; finished rows are kept")
    p.add_argument("--price-rows", type=int, default=3)
    p.add_argument("--price-only", action="store_true", help="stop after pricing the first rows")
    a = p.parse_args(argv)
    if a.threads != FROZEN_THREADS:
        raise SystemExit(f"the gauge-token builder runs only at {FROZEN_THREADS} threads (bitwise reproduction of the "
                         "frozen tokens depends on the thread count; chair ruling on F12f (a))")  # fmt: skip
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
