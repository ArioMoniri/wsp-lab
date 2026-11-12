#!/usr/bin/env python3
"""Streaming F12-I LoRA plan (PREREG_H1_TF F12g as accepted by Board A, condition 2; chair instruction of 12 Nov 2026).

``scripts/f12_additivity_prepare.py plan`` materialises the dense fp32 delta of all 16 LoRA roots before building any
of them (about 0.88 GB per root, about 19 GB peak), above the shared machine's per-job limit. This driver writes the
same ``f12-additivity-v1`` plan document for the 16 registered LoRA roots, but one root at a time and, within a root,
one tensor at a time. It reuses the pinned functions of ``f12_additivity_prepare.py`` (supports, ``construct`` with
its bitwise realised-part check, part ids, support records, holder hashes, status records, code identity, JSON
writer), which it does not change:

* per tensor: the dense stored delta ``(B @ A) * scaling`` (the ``wsp.zoo.apply.lora_dense_delta`` expression, from
  ``lora_factors``), then ``construct`` on that one tensor with the root's layer-split support restricted to it;
* per root: the full and part increment hashes are the G2 convention ``rf._hash`` over per-tensor
  ``rf._tensor_hash`` (as ``a5_power_fields.tensors_hash``), a part is a no-op iff all its realised tensors are zero;
* a root whose bitwise check (or anything else) fails is recorded with ``bitwise_sum_verified: false`` and the
  failure, and nothing is substituted;
* on success its output is byte-identical to ``plan`` on the same inputs (tested); a sidecar ``<out>_run.json``
  records the driver identity, the governed population sha256, the threads and the process's peak RSS. With
  ``--rss-limit-gb`` the run aborts once the peak RSS exceeds the limit.

F12-R (PREREG_H1_TF A5 note F12-R; unexposed adoption A10, A11, A21, A26, A30), no numerical change: the population,
G2 manifests and every root's stored bytes are read through ``f12_inputs_guard.ReadBoundary`` and authenticated
(``f12_additivity_prepare`` pins; each holder against its row's ``resolved_delta_hash``); a tensor of a layer outside
0-23 or with a non-canonical name fails its root closed (recorded, never split as ``late``); the F8a floor literal is
bound to its committed recompute; the 4 GB cap of the resource authority (``--rss-limit-gb``, default 4; above 4 only
with ``--chair-ruling``) is enforced at launch and by a process-tree monitor, and the sidecar records the cap, the peak
and the exact reads; an existing output is never overwritten; ``outcomes_read`` is relabelled as algorithmic intent.

    python scripts/f12_additivity_stream.py --threads 2 --out <new path>
    python scripts/f12_additivity_stream.py --roots 64b10bbaef2b726d --out <scratch>/x.json   # cross-check
    python scripts/f12_additivity_stream.py cross-check <plan.json> <reference manifest.json> <root id>
    python scripts/f12_additivity_stream.py requalify <new plan.json> <committed plan.json> [--out <report.json>]

No field, forecast or evaluation record is opened by the intended algorithm. CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import f12_additivity_prepare as fa
from scripts import f12_inputs_guard as fg
from wsp.features import response_field as rf

DRIVER = "scripts/f12_additivity_stream.py"
SIDECAR_SCHEMA = "f12-additivity-stream-run-v1"
CROSS_FIELDS = ("full_increment_sha256", "n_full_tensors", "bitwise_sum_verified", "I_defined")
PART_FIELDS = ("label", "part_id", "support", "n_tensors", "increment_sha256", "noop", "I_status")
REQUALIFY_SCHEMA = "f12-additivity-requalification-v1"
# A21 launch projection: the measured peak of the committed streaming run (additivity_plan_run.json), dominated by the
# bf16 base state; per-tensor work adds one dense tensor and its realised parts at a time.
STREAM_ESTIMATE_BYTES = 3_114_868_736


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def lora_factors_of(entry_dir: Path, base):
    from wsp.zoo.apply import lora_factors

    if (entry_dir / "delta.safetensors").exists():
        raise ValueError(f"{entry_dir}: not a LoRA holder (dense delta.safetensors); not streamed")
    return lora_factors(entry_dir / "lora", base)


def stream_root(e, base, guard, boundary=None):
    """One LoRA root, tensor by tensor: the plan's entry record (or a recorded failure)."""
    d = Path(e["entry_dir"])
    p = {
        "full_id": e["full_id"],
        "kind": e["kind"],
        "component": "F12.6(a)",
        "source": {"entry_dir": str(d), "files": fa.holder_files(d, boundary)},
        "split_rule": f"layers 0-{fa.LAYER_BOUNDARY - 1} (early) versus {fa.LAYER_BOUNDARY}-23 (late)",
        "disjoint_tensors": True,
    }
    factors = lora_factors_of(d, base)
    names = sorted(factors)
    try:
        supports = fa.layer_split_supports(names, canonical=set(base))
        full_h, part_h = {}, [{} for _ in supports]
        nonzero = [False for _ in supports]
        for n in names:
            B, A, s = factors[n]
            raw_n = ((B @ A) * s).float()
            sub = [(label, ({n: sup[n]} if n in sup else {})) for label, sup in supports]
            full, parts = fa.construct({n: base[n]}, {n: raw_n}, sub)
            full_h[n] = rf._tensor_hash(full[n])
            for k, (_, _, inc, _) in enumerate(parts):
                if n in inc:
                    part_h[k][n] = rf._tensor_hash(inc[n])
                    nonzero[k] = nonzero[k] or bool(torch.count_nonzero(inc[n]))
            del raw_n, full, parts
            guard.check()
    except (ValueError, KeyError) as exc:
        return p | {"bitwise_sum_verified": False, "failure": str(exc), "I_defined": False}
    rec = dict(p)
    rec["full_increment_sha256"] = rf._hash(full_h)
    rec["n_full_tensors"] = len(full_h)
    rec["bitwise_sum_verified"] = True
    rec["parts"] = []
    for (label, sup), h, nz in zip(supports, part_h, nonzero):
        rec["parts"].append(
            {
                "label": label,
                "part_id": fa.part_id(p["full_id"], label),
                "support": fa.support_record(sup),
                "n_tensors": len(sup),
                "increment_sha256": rf._hash(h),
                "noop": not nz,
                "I_status": "defined" if nz else "undefined:noop_part",
            }
        )
    rec["I_defined"] = not any(x["noop"] for x in rec["parts"])
    return rec


def plan_doc(args, boundary, guard, floor):
    t0 = time.time()
    torch.set_num_threads(args.threads)
    pop_bytes = boundary.read_bytes(args.population, sha256=fa.GOVERNED_POPULATION_SHA256)
    pop = json.loads(pop_bytes)
    entries_in = fa.lora_entries(pop, Path(args.zoo))
    if args.roots:
        known = {e["full_id"] for e in entries_in}
        unknown = sorted(set(args.roots) - known)
        if unknown:
            raise ValueError(f"{unknown}: not an F12-I LoRA root of the population")
        entries_in = [e for e in entries_in if e["full_id"] in set(args.roots)]
    for e in entries_in:
        if (Path(e["entry_dir"]) / "delta.safetensors").exists():
            raise ValueError(
                f"{e['entry_dir']}: not a LoRA holder (dense delta.safetensors); not streamed"
            )
        fg.authenticate_entry(boundary, e["entry_dir"], e["resolved_delta_hash"])
    state_hash = fa.g2_state_hash(boundary, Path(args.g2))
    base, base_hash = fa.load_base(boundary, state_hash)
    guard.check()
    entries = []
    for e in entries_in:
        rec = stream_root(e, base, guard, boundary)
        fa.log(
            f"{rec['full_id']} {rec['kind']}: bitwise {rec['bitwise_sum_verified']}"
            + (
                f" noop {[x['noop'] for x in rec['parts']]}"
                if "parts" in rec
                else f" FAILED {rec['failure']}"
            )
            + f" peak RSS {peak_rss_bytes() / 1e9:.2f} GB"
        )
        entries.append(rec)
    for full_id, kind, comp, status in (
        ("astra", "astra_tensor_split", "F12.6(b)", fa.STATUS_AWAIT_Q),
        ("c4", "c4_column_split", "F12.6(c)", fa.STATUS_AWAIT_C),
    ):
        entries.append(
            fa.status_record(
                {"full_id": full_id, "kind": kind, "component": comp, "status": status}
            )
        )
    doc = {
        "schema": fa.SCHEMA,
        "mode": "plan",
        "basis": "PREREG_H1_TF F12.6 (F12-I constructions); Board A F12 review conditions 8 (supports, bitwise sum, "
        "no-op halves, Astra only if qualified) and 9 (linear versus quadratic forecasts; zero-edit-centred)",
        "rules": {
            "part": "full stored delta restricted to a disjoint support, applied to the base and rounded in bf16; "
            "realised parts sum bitwise to the full realised increment (verified, fails closed)",
            "part_id": f"sha256(UTF-8 '{fa.PART_DOMAIN}' + full id + ':' + label)[:16]",
            "noop": "a part with zero realised increment is not written or scored; its entry's I is undefined",
            "statistic": "I = Delta y(full) - Delta y(part1) - Delta y(part2) on the common base reference (F12.6)",
            "forecasts": "zero-fit, zero-edit-centred (no intercept): linear I = 0 for every split; quadratic I = 0 "
            "for disjoint-tensor splits and q(full) - q(p1) - q(p2) for the within-tensor column split",
            "floor": f"F8a floor h* = {fa.F8A_FLOOR_H_STAR} nats/token (floor_recompute_2026-11-11.json); above floor "
            "iff |I| > h* strictly; reported beside I, not a significance threshold",
        },
        "population_json_sha256": fa.sha256_bytes(pop_bytes),
        "base": {
            "model": {"repo": fa.pf.MODEL[0], "revision": fa.pf.MODEL[1]},
            "state_hash": base_hash,
        },
        "field": None,
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
        "outcomes_read_intent": fa.outcomes_read_intent(),
        "floor_binding": floor,
        "code": fa.code_identity(),
    }
    return doc, pop_bytes


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _git_head():
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    r = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return r.stdout.strip() or None


def run(args):
    out = Path(args.out)
    side_path = out.with_name(out.stem + "_run.json")
    for p in (out, side_path):
        if p.exists():
            raise FileExistsError(f"{p} exists; a governed output is never overwritten")
    boundary = fa.read_boundary()
    floor = fa.floor_binding(boundary)
    guard = fa.resource_guard(
        DRIVER, args.threads, round(args.rss_limit_gb * 1e9), args.chair_ruling
    )
    guard.launch(STREAM_ESTIMATE_BYTES, breach_path=out.with_name(out.stem + "_breach.json"))
    doc, pop_bytes = plan_doc(args, boundary, guard, floor)
    sha = fa.write_json(out, doc)
    fa.log(f"wrote {out} sha256 {sha}")
    lora = [e for e in doc["entries"] if e["kind"] == "lora_layer_split"]
    side = {
        "schema": SIDECAR_SCHEMA,
        "plan": _rel(out),
        "plan_sha256": sha,
        "population": {"path": _rel(args.population), "sha256": fa.sha256_bytes(pop_bytes)},
        "driver": {"path": DRIVER, "sha256": fa.pf.sha256_file(ROOT / DRIVER)},
        "reused_code": fa.code_identity(),
        "git_head": _git_head(),
        "threads": args.threads,
        "rss_limit_gb": args.rss_limit_gb,
        "roots": [e["full_id"] for e in lora],
        "roots_failed": [e["full_id"] for e in lora if not e["bitwise_sum_verified"]],
        "noop_parts": doc["summary"]["n_noop_parts"],
        "floor_binding": floor,
        "read_boundary": boundary.record(),
        "outcomes_read_intent": fa.outcomes_read_intent(),
    }
    resources = guard.stop()
    side |= {"peak_rss_bytes": resources["peak_rss_bytes"], "resources": resources}
    fa.write_json(side_path, side)
    return doc


def cross_check(doc, ref, root_id) -> list:
    """Differences between ``doc``'s and ``ref``'s entries for ``root_id`` in supports, part ids, increment hashes,
    no-op status and bitwise flags (empty list: identical)."""

    def find(d, label):
        for e in d.get("entries", []):
            if e.get("full_id") == root_id:
                return e
        raise ValueError(f"{root_id} absent from the {label}")

    a, b = find(doc, "plan"), find(ref, "reference")
    diffs = [f"{k}: {a.get(k)!r} != {b.get(k)!r}" for k in CROSS_FIELDS if a.get(k) != b.get(k)]
    pa, pb = a.get("parts", []), b.get("parts", [])
    if len(pa) != len(pb):
        diffs.append(f"parts: {len(pa)} != {len(pb)}")
    for i, (x, y) in enumerate(zip(pa, pb)):
        diffs += [
            f"parts[{i}].{k}: {x.get(k)!r} != {y.get(k)!r}"
            for k in PART_FIELDS
            if x.get(k) != y.get(k)
        ]
    return diffs


def requalify(new, ref) -> dict:
    """F12-R plan requalification: every LoRA entry of ``new`` against ``ref``, byte for byte (canonical JSON and the
    plan writer's serialisation), plus the non-entry top-level keys that differ."""

    def lora(d):
        return {
            e["full_id"]: e for e in d.get("entries", []) if e.get("kind") == "lora_layer_split"
        }

    a, b = lora(new), lora(ref)
    diffs = sorted(
        i
        for i in set(a) | set(b)
        if i not in a
        or i not in b
        or fa.canonical_json(a[i]) != fa.canonical_json(b[i])
        or json.dumps(a[i], indent=1, sort_keys=True) != json.dumps(b[i], indent=1, sort_keys=True)
    )
    other = [e for e in new.get("entries", []) if e.get("kind") != "lora_layer_split"]
    other_ref = [e for e in ref.get("entries", []) if e.get("kind") != "lora_layer_split"]
    return {
        "schema": REQUALIFY_SCHEMA,
        "n_lora_entries": len(a),
        "n_lora_entries_reference": len(b),
        "identical": not diffs and len(a) == len(b),
        "differences": diffs,
        "lora_entries_sha256": fa.sha256_bytes(fa.canonical_json([a[i] for i in sorted(a)])),
        "lora_entries_sha256_reference": fa.sha256_bytes(
            fa.canonical_json([b[i] for i in sorted(b)])
        ),
        "status_entries_identical": fa.canonical_json(other) == fa.canonical_json(other_ref),
        "top_level_keys_differing": sorted(
            k for k in set(new) | set(ref) if k != "entries" and new.get(k) != ref.get(k)
        ),
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["requalify"]:
        p = argparse.ArgumentParser(
            description="F12-R: LoRA entries of a new plan against the committed plan"
        )
        p.add_argument("command")
        p.add_argument("plan", type=Path)
        p.add_argument("reference", type=Path)
        p.add_argument("--out", type=Path)
        a = p.parse_args(argv)
        new_bytes, ref_bytes = a.plan.read_bytes(), a.reference.read_bytes()
        rep = requalify(json.loads(new_bytes), json.loads(ref_bytes)) | {
            "plan": {"path": _rel(a.plan), "sha256": fa.sha256_bytes(new_bytes)},
            "reference": {"path": _rel(a.reference), "sha256": fa.sha256_bytes(ref_bytes)},
        }
        print(json.dumps(rep, indent=1))
        if a.out:
            if a.out.exists():
                raise FileExistsError(f"{a.out} exists")
            fa.write_json(a.out, rep)
        return 0 if rep["identical"] else 1
    if argv[:1] == ["cross-check"]:
        p = argparse.ArgumentParser(description="cross-check one root against a reference manifest")
        p.add_argument("command")
        p.add_argument("plan", type=Path)
        p.add_argument("reference", type=Path)
        p.add_argument("root")
        a = p.parse_args(argv)
        diffs = cross_check(
            json.loads(a.plan.read_text()), json.loads(a.reference.read_text()), a.root
        )
        print(json.dumps({"root": a.root, "identical": not diffs, "differences": diffs}, indent=1))
        return 0 if not diffs else 1
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=fa.DEFAULT_PLAN)
    p.add_argument("--roots", nargs="*")
    p.add_argument("--population", type=Path, default=fa.DEFAULT_POPULATION)
    p.add_argument("--g2", type=Path, default=ROOT / "data/runs/g2")
    p.add_argument("--zoo", default=str(ROOT / "data/zoo/qwen3.5-0.8b"))
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    p.add_argument("--rss-limit-gb", type=float, default=fg.DEFAULT_CAP_BYTES / 1e9)
    p.add_argument(
        "--chair-ruling", help="exact heading of the dated chair ruling for a cap above 4 GB"
    )
    run(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
