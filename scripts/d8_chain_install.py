"""D8 step 7b: install regenerated chain snapshots into the canonical zoo and update the artefact sidecar.

Reads the restore report written by ``scripts/d8_chain_replay.py --mode restore`` and, for every regenerated entry
that passes the acceptance checks, moves ``delta.safetensors`` + ``regen.json`` from the staging directory into
``<zoo>/<id>/`` (which already holds the surviving ``edit.json``). ``manifest.jsonl`` is never touched. The sidecar
``artefact_status.jsonl`` gets ``delta_present: true``, ``artefact_lost: false`` and a ``regenerated`` block per
installed row; the committed copy under ``docs/prereg/`` is refreshed from it.

Acceptance (per entry): touched tensor set equals ``edit.json``; Frobenius norm within ``--fro-rtol`` of
``edit.json:delta_fro_norm``; the chain's stored steps up to this entry were bitwise or re-anchored (never
silently drifted); the file exists and loads. Entries that fail stay in staging and are listed in the report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--zoo", type=Path, required=True)
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument(
        "--report", type=Path, required=True, nargs="+", help="restore report.json (one per shard)"
    )
    ap.add_argument(
        "--sidecar", type=Path, required=True, help="canonical artefact_status.jsonl (rewritten)"
    )
    ap.add_argument("--sidecar-copy", type=Path, default=None, help="committed copy to refresh")
    ap.add_argument(
        "--fro-rtol", type=float, default=1e-6, help="tolerance for exact lineages (no LoRA step)"
    )
    ap.add_argument(
        "--fro-rtol-approx",
        type=float,
        default=5e-3,
        help="tolerance for lineages with a re-trained LoRA step (pilot: |Δ‖F‖|/‖F‖ ≤ 1e-3)",
    )
    ap.add_argument("--out", type=Path, required=True, help="install report json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reports = [json.loads(p.read_text()) for p in args.report]
    report = {
        "environment": reports[0]["environment"],
        "chains": [c for r in reports for c in r["chains"]],
    }
    rows = [json.loads(line) for line in args.sidecar.read_text().splitlines() if line.strip()]
    by_id = {r["id"]: r for r in rows}
    installed, rejected = [], []
    for chain in report["chains"]:
        drifted = False
        for step in chain["steps"]:
            if step["stored"]:
                cmp = step.get("compare") or {}
                if not cmp.get("bitwise") and not step.get("anchored"):
                    drifted = True  # a stored step differed and the state was carried on anyway
                continue
            entry = step["id"]
            reasons = []
            if not step.get("touched_equal_edit_json"):
                reasons.append("touched tensors differ from edit.json")
            fro_e, fro_r = step.get("fro_edit_json"), step.get("fro_regenerated")
            exact = bool(step.get("exact_lineage", False))
            rtol = args.fro_rtol if exact else args.fro_rtol_approx
            if fro_e is None or fro_r is None or abs(fro_r - fro_e) > rtol * max(abs(fro_e), 1e-12):
                reasons.append(
                    f"fro norm {fro_r} vs edit.json {fro_e} (rtol {rtol}, exact={exact})"
                )
            if drifted:
                reasons.append("an earlier stored step drifted without re-anchoring")
            src = args.staging / entry / "delta.safetensors"
            if not src.exists():
                reasons.append("no regenerated file")
            else:
                try:
                    load_file(src)
                except (OSError, RuntimeError, ValueError) as e:  # pragma: no cover
                    reasons.append(f"unreadable: {e}")
            if entry not in by_id:
                reasons.append("not in sidecar")
            elif by_id[entry].get("delta_present"):
                reasons.append("sidecar says bytes already present")
            if reasons:
                rejected.append({"id": entry, "reasons": reasons})
                continue
            dest = args.zoo / entry
            if not (dest / "edit.json").exists():
                rejected.append({"id": entry, "reasons": ["edit.json missing in zoo"]})
                continue
            if not args.dry_run:
                shutil.move(str(src), str(dest / "delta.safetensors"))
                regen_src = args.staging / entry / "regen.json"
                if regen_src.exists():
                    shutil.move(str(regen_src), str(dest / "regen.json"))
                r = by_id[entry]
                r["delta_present"] = True
                r["artefact_lost"] = False
                r["regenerated"] = {
                    "method": "d8_chain_replay",
                    "exactness": "bitwise-reproducible kernels"
                    if exact
                    else "re-trained LoRA step (bf16 rounding)",
                    "fro_regenerated": fro_r,
                    "fro_edit_json": fro_e,
                    "anchored_from": step.get("regenerated_from_anchor"),
                    "git_sha": report["environment"].get("git_sha"),
                    "written_utc": None,
                }
                regen_path = dest / "regen.json"
                if regen_path.exists():
                    r["regenerated"]["written_utc"] = json.loads(regen_path.read_text()).get(
                        "written_utc"
                    )
            installed.append(entry)
    if not args.dry_run and installed:
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        args.sidecar.write_text(text)
        if args.sidecar_copy:
            args.sidecar_copy.write_text(text)
    summary = {
        "dry_run": args.dry_run,
        "installed": len(installed),
        "rejected": rejected,
        "installed_ids": installed,
        "sidecar": {
            "stored": sum(1 for r in rows if r.get("delta_present")),
            "aliased": sum(1 for r in rows if not r.get("delta_present") and r.get("alias_of")),
            "lost": sum(1 for r in rows if not r.get("delta_present") and not r.get("alias_of")),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "installed_ids"}
            | {"n_rejected": len(rejected)}
        )
    )


if __name__ == "__main__":
    sys.exit(main())
