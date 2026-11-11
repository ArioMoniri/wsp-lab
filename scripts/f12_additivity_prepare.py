#!/usr/bin/env python3
"""F12-I additivity constructions: split supports, part deltas and field forecasts (PREREG_H1_TF F12.6; Board A F12
review conditions 8 and 9).

Each part is a root edit whose stored delta is the full entry's stored delta restricted to a disjoint support, applied
to the base and rounded in bf16 (``rotation.apply_and_round``, the F10/Part A root convention; no rotation). Rounding is
elementwise, so the realised parts must sum bitwise to the full realised increment; that is verified at construction
and fails closed. A part with zero realised increment is a no-op: written nowhere, its I undefined and reported.

Planned full edits (F12.6):

* (a) the 16 eligible purchasable lora_sft roots (``population.json`` -> ``f12i_lora_roots``), split into layers 0-11
  (``early``) and 12-23 (``late``): 32 parts. A stored tensor outside a decoder layer fails closed.
* (b) the F8-E Astra two-tensor patch split by tensor: 2 parts, **only** from a qualified F8a.4 replay record
  (``--astra-record``); without one it is listed ``awaiting_qualification`` and nothing is built.
* (c) F8a.2's C4 own-definition rank-one ``model.layers.12.mlp.down_proj`` edit, split by input-column index (weight
  [out, in], columns j < in//2 versus j >= in//2): 2 parts, from ``--c4-delta`` (its stored delta at the base). Without
  it, it is listed ``awaiting_construction`` (F8 C1 construction inputs do not exist yet).

Field forecasts (zero-fit, zero-edit-centred: no intercept enters; Board A condition 9): per entry a and q of the full
edit and of each part from the frozen G2 WikiText field (``part_a_predictions.contract``, per tensor, base spectral
factors), I_a = a(full) - a(p1) - a(p2) (the linear forecast: 0 for every split; the numerical residual is recorded),
I_q = q(full) - q(p1) - q(p2) (0 for disjoint-tensor splits, by construction additive; for the column split the
within-tensor cross term), the explicit fp64 cross term tr(G D1 A D2^T) beside it, and the quadratic zero-fit forecast
I_a + I_q. A resolved C4 I contradicts linear additivity only (condition 9).

Modes:

    plan     no field, no contraction: build every available construction in memory, verify the bitwise sum, record
             supports, part ids and hashes -> ``docs/prereg/a5_manifests/f12/additivity_plan.json``
    build    write parts (``<target>/<part id>/{delta.safetensors,variant.json}``) and ``<target>/manifest.json`` with
             forecasts for ``--entries`` (default: every available construction)
    dry-run  exactly two constructions into the scratch dir: the first F12-I LoRA root's layer split and the column
             split (the C4 edit if ``--c4-delta`` is given, else a labelled stand-in: the same LoRA root's layer-12
             down_proj tensor, used only to exercise the column-split path)

Targets must lie under ``/home/ubuntu/wsp-data/runs/a5/f12_additivity`` (staging, chair-authorised) or
``/home/ubuntu/wsp-data/runs/a5/f12_inputs`` (scratch). CPU only; no evaluation record or outcome is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import a5_power_fields as pf
from scripts import part_a_predictions as pap
from wsp.features import response_field as rf
from wsp.features import rotation

SCHEMA = "f12-additivity-v1"
OBJECTIVE = pap.OBJECTIVE
DATA_ROOT = pf.DATA_ROOT
ALLOWED_TARGETS = (DATA_ROOT / "runs/a5/f12_additivity", DATA_ROOT / "runs/a5/f12_inputs")
DEFAULT_POPULATION = ROOT / "docs/prereg/a5_manifests/f12/population.json"
DEFAULT_PLAN = ROOT / "docs/prereg/a5_manifests/f12/additivity_plan.json"
DEFAULT_DRYRUN = DATA_ROOT / "runs/a5/f12_inputs/additivity_dryrun"
LAYER_BOUNDARY = 12  # early: layers 0-11, late: layers 12-23
C4_TENSOR = "model.layers.12.mlp.down_proj.weight"
PART_DOMAIN = "20261111:f12-i:part:"
F8A_FLOOR_H_STAR = (
    0.00493967211805284  # docs/prereg/f8_manifests/floor_payload/floor_recompute_2026-11-11.json
)
STATUS_AWAIT_Q = "awaiting_qualification"
STATUS_AWAIT_C = "awaiting_construction"
CODE_FILES = (
    "scripts/f12_additivity_prepare.py",
    "scripts/part_a_predictions.py",
    "scripts/a5_power_fields.py",
    "src/wsp/features/response_field.py",
    "src/wsp/features/rotation.py",
    "src/wsp/zoo/apply.py",
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def part_id(full_id: str, label: str) -> str:
    return sha256_bytes(f"{PART_DOMAIN}{full_id}:{label}".encode())[:16]


# ------------------------------------------------------------------------------------------ supports


def layer_of(name: str):
    m = re.match(r"model\.(?:language_model\.)?layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def layer_split_supports(names, boundary=LAYER_BOUNDARY):
    """[('early', {name: None}), ('late', {name: None})]: tensors of layers < boundary versus >= boundary."""
    early, late = {}, {}
    for n in sorted(names):
        layer = layer_of(n)
        if layer is None:
            raise ValueError(f"{n}: not a decoder-layer tensor; the layer split is undefined")
        (early if layer < boundary else late)[n] = None
    return [("early", early), ("late", late)]


def tensor_split_supports(names):
    """One part per tensor (the Astra two-tensor patch), sorted by name."""
    names = sorted(names)
    if len(names) != 2:
        raise ValueError(f"tensor split expects exactly two tensors, got {len(names)}")
    return [(f"tensor:{n}", {n: None}) for n in names]


def column_split_supports(name, shape):
    """Input-column halves of one [out, in] weight: columns [0, in//2) and [in//2, in)."""
    if len(shape) != 2:
        raise ValueError(f"{name}: column split needs a matrix")
    n_in = int(shape[1])
    half = n_in // 2
    return [("cols_lo", {name: (0, half)}), ("cols_hi", {name: (half, n_in)})]


def restrict(raw, support):
    """The stored delta restricted to a support ({name: None (whole tensor) | (col_start, col_stop)})."""
    out = {}
    for n, cols in support.items():
        x = raw[n].float()
        if cols is None:
            out[n] = x.clone()
        else:
            m = torch.zeros_like(x)
            m[:, cols[0] : cols[1]] = x[:, cols[0] : cols[1]]
            out[n] = m
    return out


def check_disjoint_cover(raw, supports):
    """Parts are disjoint and cover every stored tensor/column exactly once."""
    cover = {
        n: torch.zeros(raw[n].shape[-1] if raw[n].ndim == 2 else 1, dtype=torch.int64) for n in raw
    }
    for _, sup in supports:
        for n, cols in sup.items():
            if n not in raw:
                raise ValueError(f"{n}: support tensor not in the stored delta")
            if cols is None:
                cover[n] += 1
            else:
                cover[n][cols[0] : cols[1]] += 1
    bad = sorted(n for n, c in cover.items() if not torch.all(c == 1))
    if bad:
        raise ValueError(f"supports are not a disjoint cover: {bad[:5]}")


def construct(W, raw, supports):
    """Realised full and part increments at the base; bitwise sum verified (fails closed).

    Returns (full {name: fp32}, [(label, support, part increments {name: fp32}, noop)])."""
    check_disjoint_cover(raw, supports)
    full = {n: rotation.apply_and_round(W[n], raw[n].float())[1] for n in sorted(raw)}
    parts = []
    for label, sup in supports:
        r = restrict(raw, sup)
        inc = {n: rotation.apply_and_round(W[n], r[n])[1] for n in sorted(r)}
        noop = not any(bool(torch.count_nonzero(v)) for v in inc.values())
        parts.append((label, sup, inc, noop))
    for n, fn in full.items():
        total = torch.zeros_like(fn)
        for _, _, inc, _ in parts:
            if n in inc:
                total = total + inc[n]
        if not rotation.bits_equal(total, fn):
            raise ValueError(
                f"{n}: realised parts do not sum bitwise to the full realised increment"
            )
        for _, _, inc, _ in parts:
            if n in inc and torch.any((inc[n] != 0) & (fn == 0)):
                raise ValueError(f"{n}: a part is nonzero where the full increment is zero")
    return full, parts


def support_record(sup):
    return {
        "tensors": sorted(sup),
        "columns": {n: list(c) for n, c in sorted(sup.items()) if c is not None},
    }


# ------------------------------------------------------------------------------------------ forecasts


def cross_term(G, A, D1, D2):
    """tr(G D1 A D2^T) in fp64: q(D1 + D2) - q(D1) - q(D2) before the q >= 0 clamp."""
    return float(torch.trace(G.double() @ D1.double() @ A.double() @ D2.double().T))


def split_forecast(full_per, part_pers, disjoint_tensors, cross=None):
    """Forecasts of I from per-tensor contractions (``part_a_predictions.contract`` rows)."""
    full = pap.contraction_row(full_per, {})
    prow = [pap.contraction_row(p, {}) for p in part_pers]
    a_parts = [p["a"] for p in prow]
    q_parts = [p["q"] for p in prow]
    i_a = full["a"] - math.fsum(a_parts)
    i_a64 = full["a_fp64"] - math.fsum(p["a_fp64"] for p in prow)
    i_q = full["q"] - math.fsum(q_parts)
    out = {
        "full": {k: full[k] for k in ("a", "a_fp64", "q", "q_partial")},
        "parts": [{k: p[k] for k in ("a", "a_fp64", "q", "q_partial")} for p in prow],
        "I_a_numerical": i_a,
        "I_a_numerical_fp64": i_a64,
        "I_q_numerical": i_q,
        "forecast_linear_I": 0.0,
        "forecast_linear_rule": "a is linear in D: a(full) - a(p1) - a(p2) = 0 exactly for every split",
    }
    if disjoint_tensors:
        out["forecast_quadratic_I"] = 0.0
        out["forecast_quadratic_rule"] = (
            "disjoint tensor sets: q is additive across tensors, so I_q = 0 exactly"
        )
    else:
        out["forecast_quadratic_I"] = i_q
        out["forecast_quadratic_rule"] = (
            "within-tensor split: q(full) - q(p1) - q(p2) = tr(G D1 A D2^T) cross term"
        )
        out["cross_term_fp64"] = cross
    out["forecast_zero_fit_quadratic_I"] = out["forecast_linear_I"] + out["forecast_quadratic_I"]
    return out


class FieldView:
    """The frozen G2 WikiText field, memory-mapped: file sha256s and identity checked once, per-tensor content hashes
    checked for every tensor used, spectral factors computed per module on demand and released."""

    def __init__(self, g2: Path):
        g2_bytes = (g2 / "g2_manifest.json").read_bytes()
        if sha256_bytes(g2_bytes) != pap.G2_MANIFEST_SHA256:
            raise ValueError("g2_manifest.json differs from g2_manifest_final.json")
        man = json.loads(g2_bytes)
        frec = man["fields"][OBJECTIVE]
        if (
            frec.get("status") != "complete"
            or frec["identity_sha256"] != pap.WIKITEXT_IDENTITY_SHA256
        ):
            raise ValueError("base WikiText field is not the frozen complete G2 field")
        d = g2 / "fields" / OBJECTIVE
        files = {f: pf.sha256_file(d / f) for f in ("manifest.json", "tensors.pt")}
        if files != frec["files"]:
            raise ValueError("field file hashes differ from the G2 manifest")
        fman = json.loads((d / "manifest.json").read_text())
        self.identity = fman["identity"]
        if rf._hash(self.identity) != pap.WIKITEXT_IDENTITY_SHA256:
            raise ValueError("field identity differs from the frozen identity")
        self.content_hashes = fman["content_hashes"]
        self.payload = torch.load(
            d / "tensors.pt", map_location="cpu", weights_only=True, mmap=True
        )
        self.coverage = self.identity["coverage"][OBJECTIVE]
        self.records = {
            "g2_manifest_sha256": sha256_bytes(g2_bytes),
            "field_files": files,
            "identity_sha256": frec["identity_sha256"],
            "state_hash": self.identity["state_hash"],
            "loading": "torch.load(mmap=True); file sha256 = G2 manifest; per-tensor content hash checked on use",
        }
        self.state_hash = self.identity["state_hash"]

    def gradient(self, name):
        g = self.payload["gradients"][OBJECTIVE][name]
        if rf._tensor_hash(g) != self.content_hashes[f"gradient/{OBJECTIVE}/{name}"]:
            raise ValueError(f"{name}: gradient content hash differs")
        return g.clone()

    def factors(self, name):
        """A ``part_a_predictions.FactorCache`` over the module of ``name`` only (None without factors)."""
        row = self.coverage[name]
        if not row["factors"]:
            return None
        m = row["module"]
        pair = self.payload["factors"][OBJECTIVE][m]
        for key in ("A", "G"):
            if rf._tensor_hash(pair[key]) != self.content_hashes[f"factor/{OBJECTIVE}/{m}/{key}"]:
                raise ValueError(f"{m}: factor {key} content hash differs")
        raw = {m: {"A": pair["A"].clone(), "G": pair["G"].clone(), "bias": pair["bias"]}}
        return pap.FactorCache(self.coverage, raw, drop_raw=True)


def forecast_entry(view, full, parts, disjoint_tensors):
    """Per-tensor contraction of the full and every part increment (one spectral factor per module)."""
    full_per, part_pers, cross = {}, [{} for _ in parts], 0.0
    for name in sorted(full):
        grads = {name: view.gradient(name)}
        fc = view.factors(name)
        per, unc = pap.contract(grads, view.coverage, fc, {name: full[name]})
        if unc:
            raise ValueError(f"{name}: no gradient")
        full_per[name] = per[name]
        incs = []
        for k, (_, _, inc, _) in enumerate(parts):
            if name in inc:
                p, _ = pap.contract(grads, view.coverage, fc, {name: inc[name]})
                part_pers[k][name] = p[name]
                incs.append(inc[name])
        if not disjoint_tensors and fc is not None and len(incs) == 2:
            a, g = fc(name)
            cross += cross_term(g, a, incs[0], incs[1])
        del grads, fc
    return split_forecast(
        full_per, part_pers, disjoint_tensors, cross if not disjoint_tensors else None
    )


# --------------------------------------------------------------------------------------------- IO


def check_target(path: Path):
    p = Path(path).resolve()
    if not any(p == t or t in p.parents for t in ALLOWED_TARGETS):
        raise ValueError(f"{path}: targets must lie under {[str(t) for t in ALLOWED_TARGETS]}")


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return sha256_bytes(data)


def holder_files(entry_dir: Path) -> dict:
    out = {}
    for rel in (
        "delta.safetensors",
        "lora/adapter_model.safetensors",
        "lora/adapter_config.json",
        "edit.json",
    ):
        p = entry_dir / rel
        if p.exists():
            out[rel] = pf.sha256_file(p)
    return out


def lora_entries(pop, zoo: Path):
    rows = {r["id"]: r for r in pop["rows"]}
    out = []
    for i in pop["f12i_lora_roots"]:
        r = rows[i]
        if r["family"] != "lora_sft" or r["source"] != "purchasable":
            raise ValueError(f"{i}: not a purchasable lora_sft root")
        out.append({"full_id": i, "kind": "lora_layer_split", "entry_dir": zoo / r["bytes_holder"]})
    return out


def load_raw(entry_dir: Path, base):
    from wsp.zoo.apply import load_entry_delta

    stored = load_entry_delta(entry_dir, base)
    return {k: v.float() for k, v in stored.items()}


def plan_constructions(pop, zoo, base, c4=None, astra=None, stand_in=False):
    """Every planned full edit: (record, W, raw, supports, disjoint) or a status-only record."""
    plan = []
    for e in lora_entries(pop, zoo):
        raw = load_raw(e["entry_dir"], base)
        plan.append(
            {
                "full_id": e["full_id"],
                "kind": e["kind"],
                "component": "F12.6(a)",
                "source": {"entry_dir": str(e["entry_dir"]), "files": holder_files(e["entry_dir"])},
                "raw": raw,
                "supports": layer_split_supports(raw),
                "disjoint_tensors": True,
                "split_rule": f"layers 0-{LAYER_BOUNDARY - 1} (early) versus {LAYER_BOUNDARY}-23 (late)",
            }
        )
    if astra is None:
        plan.append(
            {
                "full_id": "astra",
                "kind": "astra_tensor_split",
                "component": "F12.6(b)",
                "status": STATUS_AWAIT_Q,
            }
        )
    else:
        rec = json.loads(Path(astra).read_text())
        if rec.get("qualified") is not True or not rec.get("entry_dir"):
            raise ValueError(
                "Astra record is not a qualified F8a.4 replay record with an entry_dir"
            )
        d = Path(rec["entry_dir"])
        raw = load_raw(d, base)
        plan.append(
            {
                "full_id": rec.get("entry_id", d.name),
                "kind": "astra_tensor_split",
                "component": "F12.6(b)",
                "source": {"entry_dir": str(d), "files": holder_files(d), "record": str(astra)},
                "raw": raw,
                "supports": tensor_split_supports(raw),
                "disjoint_tensors": True,
                "split_rule": "one part per changed tensor",
            }
        )
    if c4 is not None:
        path, cid = c4
        from safetensors.torch import load_file

        raw = {k: v.float() for k, v in load_file(str(path)).items()}
        if sorted(raw) != [C4_TENSOR]:
            raise ValueError(f"C4 delta must hold exactly {C4_TENSOR}")
        plan.append(
            {
                "full_id": cid,
                "kind": "c4_column_split",
                "component": "F12.6(c)",
                "source": {"delta": str(path), "sha256": pf.sha256_file(path)},
                "raw": raw,
                "supports": column_split_supports(C4_TENSOR, raw[C4_TENSOR].shape),
                "disjoint_tensors": False,
                "split_rule": "weight [out, in]; input columns j < in//2 versus j >= in//2",
            }
        )
    elif stand_in:
        src = plan[0]
        raw = {C4_TENSOR: src["raw"][C4_TENSOR]}
        plan.append(
            {
                "full_id": f"stand-in:{src['full_id']}:{C4_TENSOR}",
                "kind": "column_split_stand_in",
                "component": "F12.6(c) mechanics only (C4 not constructed)",
                "stand_in_for": "C4 (awaiting F8 C1 construction)",
                "source": src["source"] | {"restricted_to": C4_TENSOR},
                "raw": raw,
                "supports": column_split_supports(C4_TENSOR, raw[C4_TENSOR].shape),
                "disjoint_tensors": False,
                "split_rule": "weight [out, in]; input columns j < in//2 versus j >= in//2",
            }
        )
    else:
        plan.append(
            {
                "full_id": "c4",
                "kind": "c4_column_split",
                "component": "F12.6(c)",
                "status": STATUS_AWAIT_C,
            }
        )
    return plan


def entry_record(p, full, parts):
    rec = {
        k: p[k]
        for k in ("full_id", "kind", "component", "source", "split_rule", "disjoint_tensors")
    }
    if "stand_in_for" in p:
        rec["stand_in_for"] = p["stand_in_for"]
    rec["full_increment_sha256"] = pf.tensors_hash(full)
    rec["n_full_tensors"] = len(full)
    rec["bitwise_sum_verified"] = True
    rec["parts"] = []
    for label, sup, inc, noop in parts:
        rec["parts"].append(
            {
                "label": label,
                "part_id": part_id(p["full_id"], label),
                "support": support_record(sup),
                "n_tensors": len(inc),
                "increment_sha256": pf.tensors_hash(inc),
                "noop": noop,
                "I_status": "undefined:noop_part" if noop else "defined",
            }
        )
    rec["I_defined"] = not any(x["noop"] for x in rec["parts"])
    return rec


def status_record(p):
    return {k: p[k] for k in ("full_id", "kind", "component", "status")}


def write_part(target: Path, rec, part_rec, inc, base_hash):
    from safetensors.torch import save_file

    d = target / part_rec["part_id"]
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: inc[k].contiguous() for k in sorted(inc)}, str(d / "delta.safetensors"))
    sibling = [x["part_id"] for x in rec["parts"] if x["part_id"] != part_rec["part_id"]]
    info = {
        "schema": SCHEMA,
        "variant_id": part_rec["part_id"],
        "kind": "f12i_part",
        "full_entry_id": rec["full_id"],
        "full_entry_kind": rec["kind"],
        "sibling_part_ids": sibling,
        "construction": "full stored delta restricted to the support, applied to the base and rounded in bf16 "
        "(rotation.apply_and_round); delta.safetensors holds the realised increment at the base (fp32)",
        "delta_convention": "realised_increment_at_base",
        "predecessor_id": "base",
        "support": part_rec["support"],
        "split_rule": rec["split_rule"],
        "edited_tensors": sorted(inc),
        "increment_sha256": part_rec["increment_sha256"],
        "full_increment_sha256": rec["full_increment_sha256"],
        "bitwise_sum_verified": True,
        "noop": part_rec["noop"],
        "source": rec["source"],
        "base_state_hash": base_hash,
        "delta_sha256": pf.sha256_file(d / "delta.safetensors"),
    }
    if "stand_in_for" in rec:
        info["stand_in_for"] = rec["stand_in_for"]
    write_json(d / "variant.json", info)
    return {
        "delta.safetensors": info["delta_sha256"],
        "variant.json": pf.sha256_file(d / "variant.json"),
    }


def code_identity():
    return {f: pf.sha256_file(ROOT / f) for f in CODE_FILES}


def run(args):
    t0 = time.time()
    torch.set_num_threads(args.threads)
    pop_bytes = args.population.read_bytes()
    pop = json.loads(pop_bytes)
    zoo = Path(args.zoo)
    view = None
    if args.mode == "plan":
        state_hash = json.loads(
            (Path(args.g2) / "fields" / OBJECTIVE / "manifest.json").read_text()
        )["identity"]["state_hash"]
    else:
        check_target(args.target)
        view = FieldView(Path(args.g2))
        state_hash = view.state_hash
    base, base_hash = pf.load_base_state(state_hash)
    c4 = (args.c4_delta, args.c4_id) if args.c4_delta else None
    plan = plan_constructions(
        pop, zoo, base, c4=c4, astra=args.astra_record, stand_in=args.mode == "dry-run"
    )
    if args.mode == "dry-run":
        plan = [plan[0], plan[-1]]
    elif args.entries:
        plan = [p for p in plan if p["full_id"] in set(args.entries) or "status" in p]
    entries = []
    for p in plan:
        if "status" in p:
            entries.append(status_record(p))
            continue
        W = {n: base[n] for n in p["raw"]}
        full, parts = construct(W, p["raw"], p["supports"])
        rec = entry_record(p, full, parts)
        if view is not None:
            t = time.time()
            rec["forecast"] = forecast_entry(
                view, full, [x for x in parts if not x[3]], p["disjoint_tensors"]
            )
            rec["forecast"]["seconds"] = time.time() - t
            rec["files"] = {}
            for part_rec, (_, _, inc, noop) in zip(rec["parts"], parts):
                if not noop:
                    rec["files"][part_rec["part_id"]] = write_part(
                        Path(args.target), rec, part_rec, inc, base_hash
                    )
            log(
                f"{rec['full_id']} {rec['kind']}: I_a={rec['forecast']['I_a_numerical']:.3e} "
                f"I_q={rec['forecast']['I_q_numerical']:.3e} ({rec['forecast']['seconds']:.0f}s)"
            )
        entries.append(rec)
    doc = {
        "schema": SCHEMA,
        "mode": args.mode,
        "basis": "PREREG_H1_TF F12.6 (F12-I constructions); Board A F12 review conditions 8 (supports, bitwise sum, "
        "no-op halves, Astra only if qualified) and 9 (linear versus quadratic forecasts; zero-edit-centred)",
        "rules": {
            "part": "full stored delta restricted to a disjoint support, applied to the base and rounded in bf16; "
            "realised parts sum bitwise to the full realised increment (verified, fails closed)",
            "part_id": f"sha256(UTF-8 '{PART_DOMAIN}' + full id + ':' + label)[:16]",
            "noop": "a part with zero realised increment is not written or scored; its entry's I is undefined",
            "statistic": "I = Delta y(full) - Delta y(part1) - Delta y(part2) on the common base reference (F12.6)",
            "forecasts": "zero-fit, zero-edit-centred (no intercept): linear I = 0 for every split; quadratic I = 0 "
            "for disjoint-tensor splits and q(full) - q(p1) - q(p2) for the within-tensor column split",
            "floor": f"F8a floor h* = {F8A_FLOOR_H_STAR} nats/token (floor_recompute_2026-11-11.json); above floor "
            "iff |I| > h* strictly; reported beside I, not a significance threshold",
        },
        "population_json_sha256": sha256_bytes(pop_bytes),
        "base": {"model": {"repo": pf.MODEL[0], "revision": pf.MODEL[1]}, "state_hash": base_hash},
        "field": view.records if view is not None else None,
        "entries": entries,
        "summary": {
            "n_planned_full": 18,
            "n_built": sum("parts" in e for e in entries),
            "n_parts": sum(len(e.get("parts", [])) for e in entries),
            "n_noop_parts": sum(x["noop"] for e in entries for x in e.get("parts", [])),
            "awaiting": {e["full_id"]: e["status"] for e in entries if "status" in e},
            "wall_seconds": time.time() - t0,
            "threads": args.threads,
        },
        "outcomes_read": "none",
        "code": code_identity(),
    }
    out = args.out or (DEFAULT_PLAN if args.mode == "plan" else Path(args.target) / "manifest.json")
    if args.mode != "plan":
        check_target(Path(out).parent)
    sha = write_json(Path(out), doc)
    log(f"wrote {out} sha256 {sha}")
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=("plan", "build", "dry-run"))
    p.add_argument("--target", type=Path, default=DEFAULT_DRYRUN)
    p.add_argument("--out", type=Path)
    p.add_argument("--entries", nargs="*")
    p.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    p.add_argument("--g2", type=Path, default=ROOT / "data/runs/g2")
    p.add_argument("--zoo", default=str(ROOT / "data/zoo/qwen3.5-0.8b"))
    p.add_argument("--c4-delta", type=Path)
    p.add_argument("--c4-id")
    p.add_argument("--astra-record", type=Path)
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    args = p.parse_args(argv)
    if bool(args.c4_delta) != bool(args.c4_id):
        p.error("--c4-delta and --c4-id go together")
    run(args)


if __name__ == "__main__":
    main()
