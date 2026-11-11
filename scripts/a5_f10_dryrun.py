"""F10 dry run: re-stage a few registered rotations under the F10 audit into a scratch directory.

Not the re-stage. Builds the rotation variant (s = 1) of the named registered edits with the pinned
construction and ``audit_rule="f10"`` (phase 1, timed per edit, used to price the re-stage of the 40
rotations), then optionally the full variant set of ``--ladder-edits`` into a separate stage (phase 2:
chain-ladder flag, ladder bytes against the staged sha256, rotation determinism). Compares every rebuilt
tensor with its staged audit row and every written delta with ``manifest.json:delta_sha256``.

Reads only zoo bytes, the committed Part A manifest and audit rows, and the registered selection and
references. Never reads ``data/evals`` or any score. Writes only under ``--out``. CPU only.

    OMP_NUM_THREADS=4 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python scripts/a5_f10_dryrun.py \\
        --out data/runs/a5/f10_dryrun --edits 00f8111cbad8218f 0a841d4082532e8b 0027efdf2eae7405 \\
        --ladder-edits 00f8111cbad8218f 0027efdf2eae7405
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

PKG = Path("docs/prereg/a5_manifests")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def staged_rows():
    out = {}
    for line in (PKG / "part_a/rotation_audits.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[(r["edit_id"], r["tensor"])] = r
    return out


def compare_rows(eid, meta, staged):
    """Reproduction of the staged realised audit by the F10 diagnostics (same fp32 realised audit)."""
    n, exact, eps_ok = 0, 0, 0
    worst = 0.0
    for t, a in meta["tensor_audits"].items():
        s = staged[(eid, t)]
        n += 1
        exact += (a["rank_d"], a["rank_r"], a["norm_mismatch"], a["spectrum_mismatch"]) == (
            s["rank_d"],
            s["rank_r"],
            s["norm_mismatch"],
            s["spectrum_mismatch"],
        )
        rel = abs(a["rounding_error_rel"] - s["rounding_error_rel"]) / max(
            s["rounding_error_rel"], 1e-30
        )
        worst = max(worst, rel)
        eps_ok += rel <= 1e-4
    return {
        "n": n,
        "realised_audit_identical": exact,
        "eps_R_within_1e-4": eps_ok,
        "max_rel_diff_eps_R": worst,
    }


def summarise(eid, sel, v, meta, staged, seconds):
    e = meta["edit_audit"]
    ta = meta["tensor_audits"]
    failed = {}
    for a in ta.values():
        for c in a["failed_checks"]:
            failed[c] = failed.get(c, 0) + 1
    return {
        "edit_id": eid,
        "family": sel.family,
        "tertile": sel.tertile,
        "predecessor_id": sel.predecessor_id,
        "variant_id": v.variant_id,
        "audit_rule": meta["audit_rule"],
        "f10_verdict": "excluded" if v.excluded else "pass",
        "exclusions": v.exclusions,
        "n_tensors": e["n_tensors"],
        "failed_checks": failed,
        "energy_guard_ratio": e["energy_guard"]["ratio"],
        "log_energy_ratio": e["log_energy_ratio"],
        "pooled_norm_mismatch_signed": e["pooled_norm_mismatch_signed"],
        "pooled_rounding_error_rel": e["pooled_rounding_error_rel"],
        "noise_calibrated_passes": e["noise_calibrated_passes"],
        "n_tensors_failing_noise_calibrated": sum(
            not a["noise_calibrated_passes"] for a in ta.values()
        ),
        "in_subset_pooled_norm_2e-2": e["in_subset_pooled_norm_2e-2"],
        "in_subset_noise_calibrated": e["in_subset_noise_calibrated"],
        "realised_s31_passes": e["realised_s31_passes"],
        "n_tensors_failing_realised_s31": sum(not a["s31_passes"] for a in ta.values()),
        "max_orthogonality_error": e["max_orthogonality_error"],
        "max_reconstruction_residual": e["max_reconstruction_residual"],
        "max_prerounding_norm_mismatch": e["max_prerounding_norm_mismatch"],
        "max_prerounding_spectrum_mismatch": e["max_prerounding_spectrum_mismatch"],
        "prerounding_rank_equal": all(
            a["prerounding_rank_d"] == a["prerounding_rank_r"] for a in ta.values()
        ),
        "max_construction_residual": max(a["construction_residual"] for a in ta.values()),
        "all_replay": e["all_replay"],
        "all_support_identical": e["all_support_identical"],
        "max_rounding_bound_ratio": e["max_rounding_bound_ratio"],
        "n_entries_exceed_half_ulp_w": sum(a["n_exceed_half_ulp_w"] for a in ta.values()),
        "min_energy_ratio_tensor": min(a["energy_ratio"] for a in ta.values()),
        "kappa_range": [min(a["kappa"] for a in ta.values()), max(a["kappa"] for a in ta.values())],
        "reproduces_staged_realised_audit": compare_rows(eid, meta, staged),
        "seconds": seconds,
    }


def main():
    from wsp.eval import response_intervention as ri
    from wsp.features.extract import load_base_state
    from wsp.zoo.apply import load_edited_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoo", default="data/zoo/qwen3.5-0.8b")
    ap.add_argument("--edits", nargs="+", required=True)
    ap.add_argument("--ladder-edits", nargs="*", default=[])
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out, zoo = Path(args.out), Path(args.zoo)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((PKG / "part_a/manifest.json").read_text())
    selections, _ = ri.load_registered_selection(PKG / "selection.json")
    holders = ri.load_bytes_holders(PKG / "references.json", selections)
    sel = {s.edit_id: s for s in selections}
    rows = [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines() if x.strip()]
    header = next(r for r in reversed(rows) if r.get("type") == "header")
    rev = json.loads((zoo / args.edits[0] / "edit.json").read_text())["base_revision"]
    t0 = time.time()
    base, _ = load_base_state(header["base_model"], rev)
    base_seconds = time.time() - t0
    shapes = {k: tuple(v.shape) for k, v in base.items()}
    staged = staged_rows()
    code_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = ["git", "status", "--porcelain", "src"]
    dirty = bool(subprocess.run(status, capture_output=True, text=True, check=True).stdout)
    src_sha = {
        p: sha256(p)
        for p in ("src/wsp/features/rotation.py", "src/wsp/eval/response_intervention.py")
    }

    def build(eid, stage, strengths):
        s = sel[eid]
        kw = {"stage_root": stage, "predecessor_id": s.predecessor_id, "audit_rule": "f10"}
        if s.predecessor_id == "base":
            pred = base
        else:
            holder = holders.get(s.predecessor_id)
            pred, kw["base_state"] = (
                load_edited_model(base, zoo / (holder or s.predecessor_id)),
                base,
            )
            kw["predecessor_holder"] = holder
        return ri.build_variants(zoo / eid, pred, ri.rotation_seed(eid), strengths, **kw)

    report = {
        "code_sha": code_sha,
        "src_dirty": dirty,
        "src_sha256": src_sha,
        "base_load_seconds": base_seconds,
        "threads": torch.get_num_threads(),
        "edits": {},
    }
    variants = []
    for eid in args.edits:  # phase 1: rotations only, timed
        t = time.time()
        (v,) = build(eid, out / "rotations" / zoo.name, ())
        seconds = time.time() - t
        variants.append(v)
        meta = json.loads((v.entry_dir / "variant.json").read_text())
        row = summarise(eid, sel[eid], v, meta, staged, seconds)
        staged_hash = man["delta_sha256"].get(v.variant_id)
        row["staged_delta_sha256"] = staged_hash
        if not v.excluded:
            row["delta_sha256"] = ri._entry_files(v.entry_dir)
            row["bytes_equal_staged"] = (
                row["delta_sha256"] == staged_hash if staged_hash is not None else None
            )
        report["edits"][eid] = row
        (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
        print(eid, row["f10_verdict"], f"{seconds:.0f}s", flush=True)

    picked = [sel[e] for e in args.edits]
    m = ri.manifest_for(picked, variants, code_sha)
    m["edit_audits"] = ri.edit_audit_summary(variants)
    (out / "manifest_dryrun.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    audits = ri.audit_rows(variants, shapes)
    (out / "rotation_audits_dryrun.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in audits)
    )

    report["ladders"] = {}
    for eid in args.ladder_edits:  # phase 2: full variant set (flag, ladder bytes, determinism)
        t = time.time()
        vs = build(eid, out / "full" / zoo.name, (0.25, 0.5))
        rot = report["edits"].get(eid, {})
        for v in vs:
            files = ri._entry_files(v.entry_dir) if not v.excluded else None
            meta = json.loads((v.entry_dir / "variant.json").read_text())
            report["ladders"][v.variant_id] = {
                "edit_id": eid,
                "kind": v.kind,
                "strength": v.strength,
                "realised_increment_ladder": meta.get("realised_increment_ladder"),
                "bytes_equal_staged": files == man["delta_sha256"].get(v.variant_id)
                if files
                else None,
                "rotation_bytes_equal_phase1": files == rot.get("delta_sha256")
                if v.kind == "rotation" and files
                else None,
            }
        report["ladders"][f"{eid}:seconds"] = time.time() - t
        (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
        print(eid, "full set", f"{time.time() - t:.0f}s", flush=True)
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
