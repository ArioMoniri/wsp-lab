#!/usr/bin/env python3
"""NULL-SIZE payload and reports (Board A A5 package review, checklist item 6).

``materialise`` writes the registered edit-minus-predecessor proxy payload exactly as
``docs/prereg/a5_manifests/nullsize_inputs.json`` (schema ``a5-nullsize-inputs-spec-v2``) specifies, from
the closed snapshot's TF-v1 primary records of the development pairs. Per eligible inventory row whose
edit and actual predecessor (``base`` for roots) both have a primary record with the same scoring
identity (config hash, model revision, source hash, device), current entry bytes (``entry_hashes``) and
the 64 registered windows:

    U[w] = mean_nll(edit)[w] - mean_nll(predecessor)[w]    (wikitext_ppl ``extra.mean_nll``)
    window_mask[w] = both values on file and finite;  donor_mean = mean of valid U;
    centred_noise[w] = U[w] - donor_mean (NaN/null where masked)

Rows lacking either record are not donors and are listed with their reason in ``provenance.missing``.
Audit evidence: the A5.3.3 rotation audit and the scaled-edit construction are G3 staging outputs (A5
freeze-prep F3), so every donor carries ``audit_evidence`` 'unavailable' and an explicit ``audit`` of
``false`` (no audit ran; nothing is recorded as a pass). The registered run counts unavailable evidence
as a pass, the frozen ``unavailable_audit_fails`` scenario as a failure (F2).

Only window-level NLLs of the WikiText development windows of development records are read; no G3
outcome exists or is read, and no field is computed.

``report`` turns the ``scripts/a5_simulation.py`` outputs of the registered run and its two frozen
sensitivity scenarios into the committed reports (summary JSON + markdown, gzip traces with sha256).

    python scripts/a5_nullsize_inputs.py materialise [--out data/runs/a5/nullsize/inputs.json]
    python scripts/a5_nullsize_inputs.py report --runs data/runs/a5/nullsize \
        --dest docs/prereg/a5_manifests/nullsize
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import a5_build_manifests as bm
import a5_simulation as sim

DATA_ROOT = Path("/home/ubuntu/wsp-data")
ALLOWED_OUT = DATA_ROOT / "runs/a5"  # the one writable directory under data/ for this work
SCENARIOS = ("registered", "variance_x2", "unavailable_audit_fails")
REGISTERED_DONOR_FLOOR = (
    40  # the forty registered pairs (A5.3.1); fewer donors leaves calibration unresolved
)
IDENTITY = ("config_hash", "model_revision", "source_hash", "eval_device_id")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def window_nll(records_path: Path):
    """{item_id: (mean_nll, token_ids_hash)} of the wikitext_ppl records (no other benchmark is parsed)."""
    out = {}
    for line in records_path.read_text().splitlines():
        if '"wikitext_ppl"' not in line:
            continue
        r = json.loads(line)
        if r.get("benchmark") == "wikitext_ppl":
            extra = r.get("extra") or {}
            out[r["item_id"]] = (extra.get("mean_nll"), extra.get("token_ids_hash"))
    return out


def record_check(evals: Path, rid: str, source: Path | None, identity, windows):
    """Primary record of ``rid`` usable as a proxy operand; returns (meta, reason-or-None)."""
    d = evals / rid
    if not (d / "eval.json").exists() or not (d / "records.jsonl").exists():
        return None, "no_primary_record"
    e = json.loads((d / "eval.json").read_text())
    if e.get("eval_status") != "complete" or e.get("repeat_id") is not None:
        return None, "record_not_complete_primary"
    if e.get("config_hash") != identity["config_hash"]:
        return None, "config_hash_differs"
    if e.get("model_revision") != identity["model_revision"]:
        return None, "model_revision_differs"
    if not all(e.get("counts", {}).get(k) == n for k, n in identity["counts"].items()):
        return None, "counts_differ"
    if source is not None:
        current = bm.ri._source_hashes(source) if source.exists() else None
        if not e.get("entry_hashes") or e.get("entry_hashes") != current:
            return None, "entry_hashes_differ_from_current_bytes"
    nll = window_nll(d / "records.jsonl")
    if [(k, nll[k][1]) for k in sorted(nll, key=int)] != windows:
        return None, "windows_differ_from_text_groups"
    meta = {k: e.get(k) for k in IDENTITY} | {
        "run_id": e.get("run_id"),
        "records_sha256": bm.sha256_file(d / "records.jsonl"),
        "eval_sha256": bm.sha256_file(d / "eval.json"),
    }
    return (meta, nll), None


def materialise(paths):
    import yaml

    out_dir = Path(paths["manifests"])
    evals, zoo = Path(paths["evals"]), Path(paths["zoo"])
    spec = json.loads((out_dir / "nullsize_inputs.json").read_text())
    text_groups = json.loads((out_dir / "text_groups.json").read_text())
    selection = json.loads((out_dir / "selection.json").read_text())
    comp_bytes = (out_dir / "components.jsonl").read_bytes()
    components = [json.loads(x) for x in comp_bytes.decode().splitlines() if x.strip()]
    cfg = yaml.safe_load((ROOT / "configs/eval_tf_v1.yaml").read_bytes())
    fmap = spec["payload_schema"]["family_map"]
    window_ids = spec["payload_schema"]["window_item_ids"]
    windows = [(w["item_id"], w["token_ids_hash"]) for w in text_groups["windows"]]
    if [w[0] for w in windows] != window_ids:
        raise ValueError("text_groups windows differ from the registered window order")
    counts = {k: v.get("n", v.get("n_windows")) for k, v in cfg["benchmarks"].items()}
    chash = bm.effective_config_hash(cfg)
    manifest = {r["id"]: r for r in bm.read_jsonl(zoo / "manifest.jsonl") if "id" in r}
    snap_path = out_dir / "snapshot.json"
    snapshot = json.loads(snap_path.read_text()) if snap_path.exists() else None
    pinned = snapshot["content"]["tf_v1_records"]["primary"] if snapshot else None

    inventory, donors, noise, masks, means = [], [], [], [], []
    prov_donors, missing, cache = {}, {}, {}
    by_id = {r["id"]: r for r in components}
    for r in sorted((r for r in components if r["family"] in fmap), key=lambda r: r["id"]):
        row = {
            "id": r["id"],
            "component": r["component_id"],
            "family": fmap[r["family"]],
            "norm": r["effective_norm"] if bm.finite(r["effective_norm"]) else None,
            "eligible": r["eligibility"] == "eligible",
        }
        inventory.append(row)
        if not row["eligible"]:
            continue
        rev = manifest[r["id"]].get("base_revision")
        identity = {"config_hash": chash, "model_revision": rev, "counts": counts}

        def check(rid, identity=identity):
            if rid not in cache:
                if rid == "base":
                    source = None
                else:
                    inv = by_id[rid]
                    holder = rid if inv["bytes_present"] else inv["bytes_via_alias"]
                    source = zoo / holder if holder else zoo / rid
                res = record_check(evals, rid, source, identity, windows)
                if res[0] is not None and pinned is not None:
                    # the closed snapshot pins [sha256(eval.json), sha256(records.jsonl)] per record
                    want = pinned.get(rid)
                    got = [res[0][0]["eval_sha256"], res[0][0]["records_sha256"]]
                    if want != got:
                        raise AssertionError(f"record {rid} differs from the closed snapshot pin")
                cache[rid] = res
            return cache[rid]

        ref = r["parent_id"] or "base"
        edit_res, edit_why = check(r["id"])
        ref_res, ref_why = check(ref)
        if edit_res is None or ref_res is None:
            missing[r["id"]] = (
                f"edit:{edit_why}" if edit_res is None else f"reference {ref}:{ref_why}"
            )
            continue
        (em, enll), (rm, rnll) = edit_res, ref_res
        if any(em[k] != rm[k] for k in IDENTITY):
            missing[r["id"]] = f"scoring identity differs from reference {ref}"
            continue
        u, mask = [], []
        for w in window_ids:
            a, b = enll[w][0], rnll[w][0]
            ok = bm.finite(a) and bm.finite(b)
            u.append(a - b if ok else None)
            mask.append(ok)
        valid = [x for x in u if x is not None]
        m = math.fsum(valid) / len(valid) if valid else None
        noise.append([None if x is None else x - m for x in u])
        masks.append(mask)
        means.append(m)
        donors.append(
            row
            | {
                "audit": [False, False, False],
                "audit_evidence": ["unavailable", "unavailable", "unavailable"],
            }
        )
        prov_donors[r["id"]] = {
            "edit_record_sha256": em["records_sha256"],
            "edit_eval_sha256": em["eval_sha256"],
            "reference_id": ref,
            "reference_record_sha256": rm["records_sha256"],
            "reference_eval_sha256": rm["eval_sha256"],
            "run_ids": {"edit": em["run_id"], "reference": rm["run_id"]},
            "scoring_identity": {k: em[k] for k in IDENTITY},
            "masked_windows": [w for w, ok in zip(window_ids, mask) if not ok],
        }
    _, boundaries = sim.selection_cells(inventory)
    frozen = selection["procedural_freeze"]["cutpoints"]
    fam_back = {v: k for k, v in fmap.items()}
    got = {fam_back[f]: [[n, i] for n, i in b] for f, b in boundaries.items()}
    if got != {f: c["keys"] for f, c in frozen.items()}:
        raise AssertionError(f"inventory cutpoints {got} differ from the procedural freeze")
    # Pins are content-level (no sha256 of snapshot.json or selection.json bytes): snapshot.json records
    # this payload's sha256 in its content block, so a file-level pin here would be circular.
    snap_pins = (
        {
            "closed_at": snapshot["closed_at"],
            "base_ref": snapshot["base_ref"],
            "tf_v1_records_content_index_sha256": snapshot["content"]["tf_v1_records"][
                "content_index_sha256"
            ],
            "zoo_bytes_content_index_sha256": snapshot["content"]["zoo_bytes"][
                "content_index_sha256"
            ],
            "records_checked_against_snapshot": True,
        }
        if snapshot
        else None
    )
    payload = {
        "schema": "a5-nullsize-inputs-v2",
        "spec": {
            "path": "docs/prereg/a5_manifests/nullsize_inputs.json",
            "sha256": bm.sha256_file(out_dir / "nullsize_inputs.json"),
            "schema": spec["schema"],
        },
        "donor_source": sim.PROXY_SOURCE,
        "window_ids": window_ids,
        "inventory": inventory,
        "donors": donors,
        "centred_noise": noise,
        "window_mask": masks,
        "donor_mean": means,
        "provenance": {
            "donors": prov_donors,
            "missing": dict(sorted(missing.items())),
            "inputs": {
                "components_jsonl_sha256": sha_bytes(comp_bytes),
                "text_groups_sha256": bm.sha256_file(out_dir / "text_groups.json"),
                "procedural_freeze_ids_sorted_sha256": selection["procedural_freeze"][
                    "registered_edit_ids_sorted_sha256"
                ],
                "snapshot": snap_pins,
                "config_effective_hash": chash,
            },
            "audit_value_semantics": "audit booleans are explicit false because no audit ran pre-G2; "
            "audit_evidence 'unavailable' governs (registered run: assume pass; scenario "
            "unavailable_audit_fails: assume fail). Never a recorded pass",
            "frozen_cutpoints_match": True,
            "outcomes_read": "wikitext_ppl extra.mean_nll of the development records only; no G3 outcome "
            "exists or is read; no field is computed",
        },
    }
    return payload


def write_json(path: Path, obj):
    path.write_bytes((json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode())


def gzip_bytes(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as f:
        f.write(data)
    return buf.getvalue()


def reason_class(reason: str) -> str:
    """'reference 86ea…:no_primary_record' -> 'reference:no_primary_record' (ids dropped for counting)."""
    if reason.startswith("reference "):
        return "reference:" + reason.split(":", 1)[1]
    if reason.startswith("scoring identity"):
        return "scoring_identity_differs"
    return reason


def sufficiency(payload):
    """Registered-input sufficiency: donor count, donor cells, masked windows (A5.3.2, F2)."""
    donors = payload["donors"]
    _, boundaries = sim.selection_cells(payload["inventory"])
    cells = {f"{f}:{t}": 0 for f in boundaries for t in range(3)}
    reps = {}
    for r in sorted(donors, key=lambda r: r["id"]):
        t = sum((r["norm"], r["id"]) >= edge for edge in boundaries[r["family"]])
        reps.setdefault((r["family"], t, r["component"]), r["id"])
    for f, t, _c in reps:
        cells[f"{f}:{t}"] += 1
    incomplete = sum(not all(m) for m in payload["window_mask"])
    problems = []
    if len(donors) < REGISTERED_DONOR_FLOOR:
        problems.append(f"{len(donors)} donor pairs < the registered {REGISTERED_DONOR_FLOOR}")
    if any(n == 0 for n in cells.values()):
        problems.append("empty donor cell")
    if incomplete:
        problems.append(f"{incomplete} donor pairs with missing windows")
    return {
        "n_donor_pairs": len(donors),
        "n_donor_components": len({r["component"] for r in donors}),
        "donor_cluster_representatives_per_cell": cells,
        "pairs_with_missing_windows": incomplete,
        "missing_rows": len(payload["provenance"]["missing"]),
        "missing_reasons": dict(
            sorted(
                Counter(reason_class(v) for v in payload["provenance"]["missing"].values()).items()
            )
        ),
        "problems": problems,
        "sufficient": not problems,
    }


def report(runs: Path, dest: Path, validation: Path):
    inputs_path = runs / "inputs.json"
    payload = json.loads(inputs_path.read_text())
    val = json.loads(validation.read_text())
    suff = sufficiency(payload)
    dest.mkdir(parents=True, exist_ok=True)
    files, summary, trace_stats = {}, {}, {}
    for name in SCENARIOS:
        raw = (runs / f"{name}.json").read_bytes()
        res = json.loads(raw)
        rep = res["reports"]["null_size"]
        traces = rep.pop("traces")
        trace_stats[name] = trace_statistics(traces)
        trace_bytes = "".join(json.dumps(t, sort_keys=True) + "\n" for t in traces).encode()
        files[f"{name}_traces.jsonl.gz"] = gzip_bytes(trace_bytes)
        summary[name] = {
            "scenario": res["scenario"],
            "noise_scale": res["noise_scale"],
            "unavailable_audit": res["unavailable_audit"],
            "replicates": res["replicates"],
            "resolution_draws": res["resolution_draws"],
            "seed": res["seed"],
            "stream_order": res["stream_order"],
            "frozen_cell_cutpoints": res["frozen_cell_cutpoints"],
            "contrast_order": ["rotation", "ladder_0.25", "ladder_0.5"],
            "nominal": 0.05,
            "size_two_sided": rep["two_sided_rejection_rates"],
            "size_two_sided_cp95": rep["two_sided_cp95"],
            "positive_rejection": rep["contrast_rejection_rates"],
            "positive_rejection_cp95": rep["positive_rejection_cp95"],
            "ordered_go_rate": rep["ordered_go_rate"],
            "ordered_go_cp95": rep["ordered_go_cp95"],
            "decisions": rep["decisions"],
            "yield_rates_ge15": rep["yield_rates"],
            "yield_cp95": rep["yield_cp95"],
            "conditional_coverage": rep["conditional_coverage"],
            "audit_or_nonfinite_counts": rep["audit_or_nonfinite_counts"],
            "gate_le_0_07": rep["accepted"],
            "raw_output_sha256": sha_bytes(raw),
            "traces_jsonl_sha256": sha_bytes(trace_bytes),
            "traces_gz": f"{name}_traces.jsonl.gz",
            "input_provenance": res["input_provenance"],
        }
    reg = summary["registered"]
    if not suff["sufficient"] or val["status"] != "valid":
        verdict = "UNRESOLVED"
        why = "inputs insufficient or invalid: " + "; ".join(
            suff["problems"] + ([f"validation {val['status']}"] if val["status"] != "valid" else [])
        )
    elif reg["gate_le_0_07"]:
        verdict = "PASS"
        why = "registered run: ordered go rate and every individual two-sided size <= 0.07"
    else:
        verdict = "FAIL"
        why = "registered run exceeds 0.07: stop and amend before G2 (A5.3.2)"
    doc = {
        "schema": "a5-nullsize-report-v1",
        "gate": "A5.3.2 / A5.5.1 step 1: development null size <= 0.07 at nominal 0.05 for the ordered "
        "Part A decision and each individual two-sided test; applied to the registered run. Sensitivity "
        "scenarios are separately labelled and are not substitutes",
        "verdict": verdict,
        "verdict_reason": why,
        "scope": "conditional on the registered edit-minus-predecessor proxy and the sign-flip DGP; not an "
        "empirical certification of the rotation null (F2). No G3 outcome was read; no field was computed",
        "inputs": {
            "path": "data/runs/a5/nullsize/inputs.json",
            "sha256": bm.sha256_file(inputs_path),
            "validation": val["status"],
            "validation_counts": val["counts"],
            "validation_sha256": bm.sha256_file(validation),
        },
        "sufficiency": suff,
        "not_run": "A5.3.2's constant-positive-mean and cell-specific mask/noise sensitivities have no "
        "frozen scenario in scripts/a5_simulation.py and were not run",
        "scenarios": summary,
        "observations": observations(summary, trace_stats, suff),
    }
    files["validation.json"] = validation.read_bytes()
    files["inputs_provenance.json"] = (
        json.dumps(
            inputs_provenance(payload, bm.sha256_file(inputs_path)), indent=1, sort_keys=True
        )
        + "\n"
    ).encode()
    files["report.json"] = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    files["README.md"] = report_md(doc, {k: sha_bytes(v) for k, v in files.items()}).encode()
    for k, v in files.items():
        (dest / k).write_bytes(v)
    return doc


def trace_statistics(traces):
    """Per-scenario facts read from the traces: identical contrast tests, resolved counts."""
    resolved = [t["resolved_counts"] for t in traces]
    return {
        "contrasts_identical": all(
            len(set(map(tuple, zip(t["resolved_counts"], t["agreement_counts"])))) == 1
            for t in traces
        ),
        "max_resolved": max((max(r) for r in resolved), default=0),
        "replicates": len(traces),
    }


def observations(summary, trace_stats, suff):
    """Computed caveats on the NULL-SIZE runs; descriptive only, never a change to the registered gate."""
    out = []
    if all(s["contrasts_identical"] for s in trace_stats.values()):
        out.append(
            "in every replicate of every scenario the three contrasts have identical resolved and agreement "
            "counts: under the proxy they are rescalings (a = 1, 0.25, 0.5) of one donor block, so the three "
            "individual sizes are one test, not three separate checks"
        )
    for name, st in trace_stats.items():
        if st["max_resolved"] == 0:
            out.append(
                f"`{name}`: no pair resolved in any of {st['replicates']} replicates (every audit evidence is "
                "'unavailable' and counted as a failure), so its size 0 and gate value are vacuous: under "
                "that assumption Part A cannot resolve at all"
            )
    for name, s in summary.items():
        alt = s["decisions"]["alternative"]
        if alt:
            out.append(
                f"`{name}`: under the null the ordered decision returned 'alternative' (ladder 0.25 CP upper "
                f"<= 0.6) in {alt} of {s['replicates']} replicates ({alt / s['replicates']:.4f}); the "
                "registered <= 0.07 gate covers 'go' and the individual two-sided tests, not this route"
            )
    single = sorted(c for c, n in suff["donor_cluster_representatives_per_cell"].items() if n == 1)
    if single:
        out.append(
            f"donor cell(s) {', '.join(single)} hold a single donor component: every simulated slot of "
            "that cell redraws the same donor block (with independent signs); the registered rule requires "
            "a non-empty cell only"
        )
    return out


def inputs_provenance(payload, payload_sha256):
    """Committed extract of the payload (spec ``input_file.commit``): everything but the numeric arrays."""
    keys = ("schema", "spec", "donor_source", "window_ids", "inventory", "donors", "provenance")
    return {
        "payload_path": "data/runs/a5/nullsize/inputs.json",
        "payload_sha256": payload_sha256,
    } | {k: payload[k] for k in keys}


def _fmt(x):
    return f"{x:.4f}"


def report_md(doc, shas):
    lines = [
        "# A5 NULL-SIZE reports (Board A checklist item 6)",
        "",
        f"**Verdict: {doc['verdict']}** — {doc['verdict_reason']}.",
        "",
        doc["scope"] + ".",
        "",
        (
            f"Input `{doc['inputs']['path']}` sha256 `{doc['inputs']['sha256']}`; `--inputs-spec` validation: "
            f"**{doc['inputs']['validation']}** ({doc['inputs']['validation_counts']})."
        ),
        "",
        (
            f"Donor pairs: {doc['sufficiency']['n_donor_pairs']} in {doc['sufficiency']['n_donor_components']} "
            f"components; cluster representatives per (family:tertile) cell "
            f"{doc['sufficiency']['donor_cluster_representatives_per_cell']}; pairs with missing windows "
            f"{doc['sufficiency']['pairs_with_missing_windows']}; eligible rows without a matched pair "
            f"{doc['sufficiency']['missing_rows']} ({doc['sufficiency']['missing_reasons']})."
        ),
        "",
        (
            "Contrasts in order (rotation, ladder 0.25, ladder 0.5); 10,000 replicates, 4,000 resolution draws, "
            "SeedSequence(20261107).spawn(3) [panels, signs, resolution]. Exact CP 95% intervals in brackets."
        ),
        "",
        "| scenario | two-sided size per contrast | ordered go rate | yield (>= 15 resolved) | decisions | gate <= 0.07 |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in doc["scenarios"].items():
        size = "; ".join(
            f"{_fmt(v)} [{_fmt(c[0])}, {_fmt(c[1])}]"
            for v, c in zip(s["size_two_sided"], s["size_two_sided_cp95"])
        )
        yld = "; ".join(
            f"{_fmt(v)} [{_fmt(c[0])}, {_fmt(c[1])}]"
            for v, c in zip(s["yield_rates_ge15"], s["yield_cp95"])
        )
        go = f"{_fmt(s['ordered_go_rate'])} [{_fmt(s['ordered_go_cp95'][0])}, {_fmt(s['ordered_go_cp95'][1])}]"
        lines.append(
            f"| `{name}`{' (registered)' if name == 'registered' else ' (sensitivity)'} | {size} | {go} | "
            f"{yld} | {s['decisions']} | {s['gate_le_0_07']} |"
        )
    lines += [
        "",
        doc["not_run"] + ".",
        "",
        "Observations (computed from the runs and traces; they do not change the registered verdict):",
        "",
        *(f"- {o}" for o in doc["observations"]),
        "",
        "| file | sha256 |",
        "|---|---|",
        *(f"| `{k}` | `{v}` |" for k, v in sorted(shas.items())),
        "",
        (
            "Traces are gzip (mtime 0) JSONL, one line per replicate: donor representatives per slot, component "
            "signs, audit/finite and resolved masks, resolved and agreement counts, decision. `report.json` holds "
            "the sha256 of each scenario's raw simulation output (kept at `data/runs/a5/nullsize/<scenario>.json`)."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("materialise")
    m.add_argument("--manifests", default=ROOT / "docs/prereg/a5_manifests", type=Path)
    m.add_argument("--evals", default=ROOT / "data/evals/qwen3.5-0.8b/tf_v1", type=Path)
    m.add_argument("--zoo", default=ROOT / "data/zoo/qwen3.5-0.8b", type=Path)
    m.add_argument("--out", default=ALLOWED_OUT / "nullsize/inputs.json", type=Path)
    r = sub.add_parser("report")
    r.add_argument("--runs", default=ALLOWED_OUT / "nullsize", type=Path)
    r.add_argument(
        "--validation", type=Path, help="--inputs-spec report (default RUNS/validation.json)"
    )
    r.add_argument("--dest", default=ROOT / "docs/prereg/a5_manifests/nullsize", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "materialise":
        out = a.out.resolve()
        if out.is_relative_to(DATA_ROOT) and not out.is_relative_to(ALLOWED_OUT):
            p.error(f"under data/ only {ALLOWED_OUT} is writable")
        payload = materialise({"manifests": a.manifests, "evals": a.evals, "zoo": a.zoo})
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, payload)
        suff = sufficiency(payload)
        print(json.dumps({"path": str(out), "sha256": bm.sha256_file(out)} | suff, indent=1))
        return 0
    doc = report(a.runs, a.dest, a.validation or a.runs / "validation.json")
    print(json.dumps({"verdict": doc["verdict"], "reason": doc["verdict_reason"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
