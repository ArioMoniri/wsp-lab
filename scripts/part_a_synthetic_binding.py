#!/usr/bin/env python3
"""Synthetic end-to-end input-binding run of the Part A analysis CLI on the final manifest and predictions.

Builds a SYNTHETIC sealed ledger (schema ``a5-part-a-sealed-ledger-v2``) from the real execution manifest
(``docs/prereg/a5_manifests/part_a/manifest.json``), the real frozen predictions (``predictions_f11.json``) and
seeded random window responses, then runs ``scripts/part_a_analyse.py`` with ``--manifest-sha256``, ``--spec-sha256``
and the F11.4 supplemented calibration output named by ``f11_calibration_record.json`` on copies of the inputs:

  A  intact inputs                                        -> exit 0 (reports written, inputs bound)
  B  manifest copy with one byte flipped, original sha256   -> refused (file hash != --manifest-sha256)
  C  same flipped copy, its own sha256 supplied             -> refused (ledger binds the original manifest)
  D  predictions copy with one row dropped, ledger as is    -> refused (fields sha256 != ledger's)
  E  same dropped copy, ledger re-bound to its hash         -> refused (missing frozen prediction row)

Everything is written under ``--scratch`` (outside data/); the window values are random (``numpy`` PCG64, ``--seed``),
not outcomes. The real inputs read are committed metadata (manifest, predictions, text groups, specification,
registration, a533 manifest, NULL-SIZE report, re-stage audit file and README), the staged variant.json files (logged
moments for the covariate block; hash-checked against the manifest), the base field's coverage block (tensor shapes)
and the F11.4 calibration output (a simulation, hash-checked against the calibration record). No evaluation record or
TF-v1 score is opened.

    python scripts/part_a_synthetic_binding.py --scratch /tmp/part_a_synthetic_binding \
        [--summary-out /home/ubuntu/wsp-data/runs/a5/part_a_predictions/synthetic_binding_summary.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from wsp.eval import part_a_analysis as pa

PART_A = ROOT / "docs/prereg/a5_manifests/part_a"
DEFAULTS = {
    "manifest": PART_A / "manifest.json",
    "predictions": PART_A / "predictions_f11.json",
    "calibration_record": PART_A / "f11_calibration_record.json",
    "audit_file": PART_A / "rotation_audits.jsonl",
    "readme": PART_A / "README.md",
    "coverage": ROOT / "data/runs/g2/fields/wikitext_nll/manifest.json",
}
SEED = 20261111
MOMENTS = ("mean", "std", "frac_pos", "frac_nonzero", "cos_to_predecessor")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    return sha256_bytes(Path(p).read_bytes())


def contrast_of(v) -> str:
    return "rotation" if v["kind"] == "rotation" else f"ladder_{float(v['strength'])}"


def restage_entry(manifest, i, stage, shapes):
    """The ledger's re-stage entry of edit ``i`` from the execution manifest (verdict, codes, F10.4 diagnostics) and
    the staged rotation variant.json (logged moments; sha256 checked against the manifest)."""
    ea = manifest["edit_audits"][i]
    codes = sorted({x["reason"] for x in ea["exclusions"]})
    vid = ea["rotation_variant_id"]
    raw = (Path(stage) / vid / "variant.json").read_bytes()
    if sha256_bytes(raw) != manifest["variant_json_sha256"][vid]:
        raise ValueError(f"{vid}: variant.json differs from the manifest")
    vj = json.loads(raw)
    tensors = {}
    for name, t in sorted((vj.get("tensor_audits") or {}).items()):
        if isinstance(t, dict) and all(
            isinstance(t.get(k), dict) and all(m in t[k] for m in MOMENTS)
            for k in ("logged_d", "logged_r")
        ):
            tensors[name] = {
                "shape": shapes[name],
                "logged_d": {m: t["logged_d"][m] for m in MOMENTS},
                "logged_r": {m: t["logged_r"][m] for m in MOMENTS},
            }
    rho = ea["edit_audit"].get("pooled_energy_ratio")
    return {
        "f10_3_verdict": ea["pair_verdict"],
        "pre_audit_exclusions": [c for c in codes if c in pa.PRE_AUDIT_CODES],
        "f10_3_exclusions": [c for c in codes if c in pa.F10_3_CODES],
        "registered_realised_verdict": bool(ea["edit_audit"].get("realised_s31_passes")),
        "diagnostics": ea["edit_audit"],
        "variant_json_sha256": manifest["variant_json_sha256"][vid],
        "pooled_rho": rho if pa._finite(rho) else None,
        "tensors": tensors,
    }


def build_ledger(manifest, manifest_sha, fields_sha, text_groups, provenance, stage, shapes, seed):
    """Synthetic sealed ledger over every selected edit of the real execution manifest.

    Window values: reference ~ 3 + N(0, .05); original = reference + .02 + N(0, 1e-3); rotation = original -
    U(-.01, .01) + N(0, 1e-3); ladders = predecessor + s * U(-.01, .01) + N(0, 1e-3). Random, seeded; not outcomes.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    sel = {s["edit_id"]: s for s in manifest["selections"]}
    var = {(v["source_id"], contrast_of(v)): v for v in manifest["variants"]}
    refs, variants, restage = {}, {}, {}
    for i in sorted(sel):
        pred = sel[i]["predecessor_id"]
        if pred not in refs:
            refs[pred] = {"mean_nll": (3.0 + rng.normal(0, 0.05, 64)).tolist()}
        base = np.asarray(refs[pred]["mean_nll"])
        orig = base + 0.02 + rng.normal(0, 1e-3, 64)
        refs[i] = {"mean_nll": orig.tolist()}
        variants[i] = {}
        for c in pa.CONTRASTS:
            v = var[(i, c)]
            entry = {
                "variant_id": v["variant_id"],
                "excluded": bool(v["excluded"]),
                "exclusions": sorted({x["reason"] for x in v.get("exclusions") or []}),
            }
            if not v["excluded"]:
                if c == "rotation":
                    y = orig - rng.uniform(-0.01, 0.01) + rng.normal(0, 1e-3, 64)
                else:
                    s = pa.LADDER_STRENGTH[c]
                    y = base + s * rng.uniform(-0.01, 0.01) + rng.normal(0, 1e-3, 64)
                entry["mean_nll"] = y.tolist()
            variants[i][c] = entry
        restage[i] = restage_entry(manifest, i, stage, shapes)
    return {
        "schema": pa.LEDGER_SCHEMA,
        "sealed": True,
        "synthetic": "random seeded window values; not outcomes (part_a_synthetic_binding.py)",
        "manifest_sha256": manifest_sha,
        "fields_sha256": fields_sha,
        "text_groups_sha256": text_groups["sha256"],
        "window_ids": list(text_groups["window_ids"]),
        "two_window_blocks": [list(b) for b in text_groups["two_window_blocks"]],
        "window_token_ids_hashes": list(text_groups["token_ids_hashes"]),
        "restage_provenance": dict(provenance, manifest_sha256=manifest_sha),
        "references": refs,
        "variants": variants,
        "restage": restage,
    }


def flip_byte_in_value(data: bytes, key: bytes = b'"code_sha": "') -> tuple[bytes, int]:
    """Flip one hex digit inside the value of ``key`` (the JSON stays valid); returns (bytes, offset)."""
    at = data.index(key) + len(key)
    b = bytearray(data)
    b[at] = ord("0") if b[at] != ord("0") else ord("1")
    return bytes(b), at


def drop_row(doc, edit_id):
    out = json.loads(json.dumps(doc))
    out["predictions"].pop(edit_id)
    return out


def run_cli(name, scratch, ledger, fields, manifest, manifest_sha, masked, spec_sha, expect):
    out = scratch / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/part_a_analyse.py"),
        "--ledger", str(ledger),
        "--fields", str(fields),
        "--spec-sha256", spec_sha,
        "--manifest", str(manifest),
        "--manifest-sha256", manifest_sha,
        "--masked-calibration", str(masked),
        "--output", str(out / "report.json"),
        "--markdown", str(out / "report.md"),
    ]  # fmt: skip
    t0 = time.time()
    p = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=ROOT)
    res = {
        "case": name,
        "command": [str(x) for x in cmd],
        "exit_code": p.returncode,
        "stderr_tail": p.stderr.strip().splitlines()[-3:],
        "stdout_tail": p.stdout.strip().splitlines()[-3:],
        "seconds": round(time.time() - t0, 1),
        "expect": expect,
    }
    if expect["exit_code"] == 0:
        res["passed"] = p.returncode == 0
    else:
        res["passed"] = (
            p.returncode == expect["exit_code"] and expect["refusal_contains"] in p.stderr
        )
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=Path, default=v)
    ap.add_argument("--stage", type=Path, default=ROOT / "data/runs/a5/part_a/qwen3.5-0.8b")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--summary-out", type=Path)
    args = ap.parse_args(argv)
    scratch = args.scratch
    if (Path("/home/ubuntu/wsp-data") in scratch.resolve().parents) or scratch.resolve() == ROOT:
        raise SystemExit("scratch must be outside data/ and the repository root")
    scratch.mkdir(parents=True, exist_ok=True)

    record = json.loads(args.calibration_record.read_text())
    masked_src = ROOT / record["output_path"]
    masked_sha = sha256_file(masked_src)
    if masked_sha != record["output_sha256"]:
        raise SystemExit(f"F11.4 calibration output {masked_src} differs from the record's sha256")
    inputs = scratch / "inputs"
    inputs.mkdir(exist_ok=True)
    manifest_copy = inputs / "manifest.json"
    fields_copy = inputs / "predictions_f11.json"
    masked_copy = inputs / "power_slotmask_f11.json"
    for src, dst in (
        (args.manifest, manifest_copy),
        (args.predictions, fields_copy),
        (masked_src, masked_copy),
    ):
        shutil.copyfile(src, dst)
    manifest_bytes = manifest_copy.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    if manifest_sha != sha256_file(args.manifest):
        raise SystemExit("manifest copy differs")
    manifest = json.loads(manifest_bytes)
    fields_sha = sha256_file(fields_copy)
    predictions = json.loads(fields_copy.read_text())
    text_groups = pa.load_text_groups()
    spec_sha = pa.SPEC_SHA256
    shapes = {
        k: v["shape"]
        for k, v in json.loads(args.coverage.read_text())["identity"]["coverage"][
            "wikitext_nll"
        ].items()
    }
    provenance = {
        "audit_file_sha256": sha256_file(args.audit_file),
        "readme_sha256": sha256_file(args.readme),
    }
    ledger = build_ledger(
        manifest,
        manifest_sha,
        fields_sha,
        text_groups,
        provenance,
        args.stage,
        shapes,
        args.seed,
    )
    ledger_path = scratch / "ledger.json"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True))

    results = []
    a = run_cli(
        "A_intact",
        scratch,
        ledger_path,
        fields_copy,
        manifest_copy,
        manifest_sha,
        masked_copy,
        spec_sha,
        {"exit_code": 0},
    )
    rep_path = scratch / "A_intact" / "report.json"
    if a["exit_code"] == 0:
        rep = json.loads(rep_path.read_text())
        inp = rep["inputs"]
        # every pair's prediction in the report is the frozen file's
        a["bound"] = {
            "fields_sha256": inp["fields"]["sha256"] == fields_sha,
            "manifest_sha256": inp["manifest"]["sha256"] == manifest_sha,
            "spec_sha256": inp["spec"]["sha256"] == spec_sha,
            "masked_calibration_sha256": inp["masked_calibration"]["sha256"] == masked_sha,
            "ledger_sha256": inp["ledger"]["sha256"] == sha256_file(ledger_path),
        }
        a["passed"] = a["passed"] and all(a["bound"].values())
        a["report_sha256"] = sha256_file(rep_path)
        a["report_top_keys"] = sorted(rep)
        a["decision"] = _decision(rep)
        a["contrast_summary"] = {
            c: {
                k: v for k, v in row.items() if isinstance(v, (int, float, str)) and k != "contrast"
            }
            for c, row in (rep["registered"].get("contrasts") or {}).items()
            if isinstance(row, dict)
        }
        a["n_selected"] = len(manifest["selections"])
        a["predictions_used"] = _predictions_used(rep, predictions["predictions"])
        a["passed"] = a["passed"] and a["predictions_used"]["all_equal"]
    results.append(a)

    flipped, offset = flip_byte_in_value(manifest_bytes)
    flipped_path = inputs / "manifest_flipped.json"
    flipped_path.write_bytes(flipped)
    json.loads(flipped)  # still valid JSON: the refusal is the hash binding, not a parse error
    flipped_sha = sha256_bytes(flipped)
    common = (masked_copy, spec_sha)
    b = run_cli(
        "B_manifest_byte_flipped",
        scratch,
        ledger_path,
        fields_copy,
        flipped_path,
        manifest_sha,
        *common,
        {"exit_code": 2, "refusal_contains": "does not match the supplied --manifest-sha256"},
    )
    b["flip"] = {"offset": offset, "sha256_flipped": flipped_sha, "sha256_original": manifest_sha}
    results.append(b)
    c = run_cli(
        "C_manifest_byte_flipped_own_sha",
        scratch,
        ledger_path,
        fields_copy,
        flipped_path,
        flipped_sha,
        *common,
        {"exit_code": 2, "refusal_contains": "does not match the execution manifest"},
    )
    results.append(c)

    dropped_id = min(predictions["predictions"])
    dropped = drop_row(predictions, dropped_id)
    dropped_path = inputs / "predictions_dropped_row.json"
    dropped_path.write_text(json.dumps(dropped, indent=1, sort_keys=True) + "\n")
    d = run_cli(
        "D_prediction_row_dropped",
        scratch,
        ledger_path,
        dropped_path,
        manifest_copy,
        manifest_sha,
        *common,
        {"exit_code": 2, "refusal_contains": "does not match the ledger's fields_sha256"},
    )
    d["dropped_edit_id"] = dropped_id
    results.append(d)
    rebound = dict(ledger, fields_sha256=sha256_file(dropped_path))
    rebound_path = scratch / "ledger_rebound_to_dropped.json"
    rebound_path.write_text(json.dumps(rebound, sort_keys=True))
    e = run_cli(
        "E_prediction_row_dropped_ledger_rebound",
        scratch,
        rebound_path,
        dropped_path,
        manifest_copy,
        manifest_sha,
        *common,
        {
            "exit_code": 2,
            "refusal_contains": f"no frozen prediction row (3 contrasts) for selected edit {dropped_id}",
        },
    )
    e["dropped_edit_id"] = dropped_id
    results.append(e)

    summary = {
        "schema": "a5-part-a-synthetic-binding-v1",
        "synthetic": True,
        "seed": args.seed,
        "inputs": {
            "manifest": {"path": str(args.manifest), "sha256": manifest_sha},
            "predictions": {"path": str(args.predictions), "sha256": fields_sha},
            "calibration_record": {
                "path": str(args.calibration_record),
                "sha256": sha256_file(args.calibration_record),
            },
            "masked_calibration": {"path": str(masked_src), "sha256": masked_sha},
            "spec_sha256": spec_sha,
            "text_groups_sha256": text_groups["sha256"],
            "restage_provenance": provenance,
        },
        "ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
        "cases": results,
        "all_passed": all(r["passed"] for r in results),
        "outcomes_read": "none: window values are random and seeded; no evaluation record or score is opened",
    }
    (scratch / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    for r in results:
        print(
            f"{r['case']}: exit {r['exit_code']} passed={r['passed']} {r['stderr_tail'][-1:] or r['stdout_tail'][-1:]}"
        )
    print(f"all_passed={summary['all_passed']}")
    return 0 if summary["all_passed"] else 1


def _decision(rep):
    """The registered decision's outcome fields (whatever the report names them)."""
    dec = (rep.get("registered") or {}).get("decision") or {}
    return {k: v for k, v in dec.items() if not isinstance(v, (dict, list))}


def _predictions_used(rep, preds):
    """Every per-pair prediction the report carries (registered.pairs) equals the frozen file's row."""
    pairs = (rep.get("registered") or {}).get("pairs") or {}
    seen, bad = 0, []
    for i, by_c in sorted(pairs.items()):
        for k, c in enumerate(pa.CONTRASTS):
            r = (by_c or {}).get(c)
            if isinstance(r, dict) and "prediction" in r:
                seen += 1
                if r["prediction"] != preds[i][k]:
                    bad.append([i, c])
    return {
        "n_pairs": len(pairs),
        "n_compared": seen,
        "mismatches": bad,
        "all_equal": not bad and seen == 3 * len(preds) and len(pairs) == len(preds),
    }


if __name__ == "__main__":
    raise SystemExit(main())
