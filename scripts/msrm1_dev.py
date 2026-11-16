#!/usr/bin/env python3
"""MSRM-1 development v0 (council round 7, SYNTHESIS-r7 Decision 1). EXPLORATORY development on already-exposed
outcomes (PREREG_H1_TF A4.4 wording): not a registered test, not evidence for or against any hypothesis.

Two stages, each a separate job (run through agent-tmux):

* ``inputs`` (opens no outcome): population and allowlist checks, component-closure audit, F12 folds, and the x(D)
  inputs of the 123 existing non-Part-A roots, built from the pinned F12 caches (per-tensor a_t, q_t, sigma_1 of
  ``runs/a5/f12_inputs/tokens``; the v4 stats.py statistics of ``data/features``; the F12 token matrix) plus a CPU pass
  over the stored edit bytes and the pinned base weights (realised bf16 increments, authenticated against the token
  build's ``increment_sha256``) for <D, W>, ||W||, the intended delta and the leading output vectors. No base field is
  collected and no forward pass is run.
* ``fit`` (reads outcomes through ``wsp.msrm.outcomes.OutcomeReader`` only): TF-v1 WikiText Δy of the 123 allowlisted
  roots against the common base record; head class chosen once by grouped inner CV; every arm on identical rows and
  folds; contrasts, component-bootstrap intervals, family and leave-one-family-out diagnostics, the 79-root
  status-literal sensitivity, kill criteria k1/k2/k2b; JSON results and the markdown report.

    OMP_NUM_THREADS=2 python scripts/msrm1_dev.py inputs
    OMP_NUM_THREADS=2 python scripts/msrm1_dev.py fit --date 2026-11-16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import f12_inputs_guard as fg
from scripts import f12_population as fp

DATA_ROOT = fg.DATA_ROOT
MANIFESTS = ROOT / "docs/prereg/a5_manifests"
POPULATION = MANIFESTS / "f12/population.json"
POPULATION_SHA256 = "07ab85d31cf8595b9b8dcd095a1f272aceb004cfc79d3296c55373db2e233da5"
SNAPSHOT_SHA256 = fp.INPUT_SHA256["snapshot.json"]
TOKENS_DIR = DATA_ROOT / "runs/a5/f12_inputs/tokens"
TOKENS_MANIFEST = MANIFESTS / "f12/tokens_manifest.json"
TOKENS_MANIFEST_SHA256 = "424333d9b6b2e19a0586b34096b1968d1de054e6ec625da9bceb4507b567001f"
MATRIX = DATA_ROOT / "runs/a5/f12_inputs/matrix/f12_token_matrix.npz"
MATRIX_FILE_SHA256 = "26b2fb63d16dc6272cdbecca6cac3e52bc532d66a0f02a70ee4aa26f0ac7ddfa"
FEATURES = DATA_ROOT / "features/qwen3.5-0.8b"
ZOO = DATA_ROOT / "zoo/qwen3.5-0.8b"
EVALS = ROOT / "data/evals/qwen3.5-0.8b/tf_v1"
OUT = DATA_ROOT / "runs/msrm1_dev"
REPORT_DIR = ROOT / "docs/reviews/msrm"
LEDGER = OUT / "cpu_ledger.jsonl"
CPU_CAP_HOURS = 10.0
BUILD = "scripts/msrm1_dev.py"
LITERAL_79 = {"rows": 79, "components": 52}
LABEL = (
    "EXPLORATORY development on already-exposed outcomes (PREREG_H1_TF A4.4 wording); not a registered test; not "
    "evidence for or against any hypothesis"
)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_seconds() -> float:
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def ledger_hours() -> float:
    if not LEDGER.exists():
        return 0.0
    return (
        sum(json.loads(x)["cpu_seconds"] for x in LEDGER.read_text().splitlines() if x.strip())
        / 3600
    )


def append_ledger(stage, seconds, wall, peak):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {
        "stage": stage,
        "cpu_seconds": seconds,
        "wall_seconds": wall,
        "peak_rss_bytes": peak,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _finite(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, np.generic):
        return _finite(o.item())
    return o


def write_json(path: Path, obj) -> str:
    obj = _finite(obj)
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    return fg.atomic_write_bytes(path, data)


def boundary(extra=()):
    from huggingface_hub import constants

    return fg.ReadBoundary(
        allowed=[
            MANIFESTS / "components.jsonl",
            MANIFESTS / "snapshot.json",
            MANIFESTS / "selection.json",
            MANIFESTS / "selection_f11.json",
            MANIFESTS / "references.json",
            MANIFESTS / "f12",
            TOKENS_DIR,
            MATRIX.parent,
            FEATURES,
            ZOO,
            OUT,
            Path(constants.HF_HUB_CACHE),
            *extra,
        ]
    )


def jload(b, path, sha=None):
    return json.loads(b.read_bytes(path, sha256=sha))


# ============================================================================================ governance


def load_governance(b):
    """Population, Part A ids, snapshot projections; every exclusion of the synthesis asserted."""
    from wsp.msrm import outcomes as oc

    pop = jload(b, POPULATION, POPULATION_SHA256)
    comps_rows = [
        json.loads(x)
        for x in b.read_bytes(
            MANIFESTS / "components.jsonl", sha256=fp.INPUT_SHA256["components.jsonl"]
        )
        .decode()
        .splitlines()
        if x.strip()
    ]
    by_id = {r["id"]: r for r in comps_rows}
    sel = jload(b, MANIFESTS / "selection.json", fp.INPUT_SHA256["selection.json"])
    sel11 = jload(b, MANIFESTS / "selection_f11.json", fp.INPUT_SHA256["selection_f11.json"])
    part_a = fp.part_a_components(sel, sel11, by_id)
    if len(part_a) != 52:
        raise ValueError(f"{len(part_a)} Part A components, 52 expected")
    refs = jload(b, MANIFESTS / "references.json")
    ref_ids = {r.get("edit_id") for r in refs["rows"]} | {
        (r.get("closure_proof") or {}).get("bytes_holder") for r in refs["rows"]
    }
    forbidden = {i for i, r in by_id.items() if r["component_id"] in part_a} | (ref_ids - {None})
    snap = jload(b, MANIFESTS / "snapshot.json", SNAPSHOT_SHA256)
    allow = oc.allowlist_from_population(pop)
    rows = {r["id"]: r for r in pop["rows"] if r["id"] in set(allow)}
    for i, r in rows.items():
        z = by_id[i]
        checks = {
            "root": z.get("parent_id") is None,
            "outside_part_a": z["component_id"] not in part_a and i not in forbidden,
            "not_fresh_unselected": z.get("component_status") != "fresh_unselected",
            "not_task_vector": r["family"] != "task_vector",
            "status_present": z.get("component_status") in ("development", "confirmatory"),
            "has_primary_record": i in snap["content"]["tf_v1_records"]["primary"],
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            raise ValueError(f"{i}: population exclusion violated: {bad}")
    lit = [i for i in allow if not rows[i]["g6_exposed_confirmatory"]]
    n_lit_c = len({rows[i]["component_id"] for i in lit})
    if (len(lit), n_lit_c) != (LITERAL_79["rows"], LITERAL_79["components"]):
        raise ValueError(f"status-literal subset {len(lit)} / {n_lit_c}, expected 79 / 52")
    return {
        "pop": pop,
        "by_id": by_id,
        "part_a": part_a,
        "forbidden": forbidden,
        "snap": snap,
        "allow": allow,
        "rows": rows,
        "literal": lit,
    }


# ================================================================================================ inputs


def unembedding_mean(base):
    """ubar = mean over vocabulary rows of the unembedding, times the final norm's effective gain (1 + w)."""
    import torch

    E = base.get("lm_head.weight")
    src = "lm_head.weight"
    if E is None:
        cands = [k for k in base if k.endswith("embed_tokens.weight")]
        if len(cands) != 1:
            raise ValueError(f"unembedding not identifiable: {cands}")
        src, E = cands[0], base[cands[0]]
    norms = [
        k for k in base if k.endswith("norm.weight") and ".layers." not in k and "visual" not in k
    ]
    norms = [k for k in norms if k.split(".")[-2] == "norm"]
    if len(norms) != 1:
        raise ValueError(f"final norm not identifiable: {norms}")
    acc = torch.zeros(E.shape[1], dtype=torch.float64)
    for s in range(0, E.shape[0], 16384):
        acc += E[s : s + 16384].double().sum(0)
    ubar = acc / E.shape[0] * (1.0 + base[norms[0]].double())
    return ubar, {
        "unembedding": src,
        "final_norm": norms[0],
        "norm_gain": "1 + weight (Qwen3_5RMSNorm)",
    }


def stage_inputs(args):
    import torch
    from safetensors.torch import load_file

    from scripts import a5_power_fields as pf
    from scripts import f12_base_loader as fbl
    from wsp.features import response_field as rf
    from wsp.features import rotation
    from wsp.msrm import closure as cl
    from wsp.msrm import features as fx
    from wsp.msrm import folds as fo
    from wsp.zoo.apply import load_entry_delta

    t0, c0 = time.time(), cpu_seconds()
    if ledger_hours() >= CPU_CAP_HOURS:
        raise SystemExit("CPU cap exhausted")
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    guard = fg.ResourceGuard(build=BUILD, threads=args.threads)
    guard.launch(estimate_bytes=3_200_000_000, breach_path=OUT / "inputs_breach.json")
    b = boundary()
    # ---- base weights first (pinned, confined loader; its load peaks near 3.1 GB, base_load_check.json), then the
    # unembedding mean; everything else is read afterwards so the peak stays under the 4 GB cap
    import gc

    base, base_hash = fbl.load_verified_base(b, fbl.BASE_STATE_HASH)
    gc.collect()
    guard.check()
    ubar, ubar_rec = unembedding_mean(base)
    g = load_governance(b)
    allow, rows, by_id, part_a = g["allow"], g["rows"], g["by_id"], g["part_a"]

    # ---- closure audit (edit.json opened for non-Part-A entries only)
    edits = {}
    for i, z in sorted(by_id.items()):
        if z["component_id"] in part_a or i in g["forbidden"]:
            continue
        p = ZOO / i / "edit.json"
        if p.exists():
            edits[i] = json.loads(b.read_bytes(p))
    view = cl.EditView(
        edits, forbidden={i for i, z in by_id.items() if z["component_id"] in part_a}
    )
    dh = g["snap"]["content"]["zoo_bytes"]["entry_delta_hash"]
    resolved = {}
    for i, z in by_id.items():
        h = dh.get(i) or (dh.get(z.get("bytes_via_alias")) if z.get("bytes_via_alias") else None)
        if h:
            resolved[i] = h
    audit = cl.closure_audit(allow, by_id, view, resolved, part_a)
    kept = [i for i in allow if i not in audit["excluded"]]

    # ---- folds (F12.2 rule on the closure-merged components of the kept rows)
    comp = fo.merge_components([rows[i]["component_id"] for i in kept], audit["merge"])
    fams = [rows[i]["family"] for i in kept]
    fold_map = fo.outer_folds(comp, fams)
    fold = [fold_map[c] for c in comp]
    fo.assert_component_disjoint(comp, fold)
    # registered components (no closure merges): the frozen F12 S0 folds, kept as a sensitivity
    comp_reg = [rows[i]["component_id"] for i in kept]
    freg = fo.outer_folds(comp_reg, fams)
    fold_reg = [freg[c] for c in comp_reg]
    s0 = next(s for s in g["pop"]["s_k"] if s["k"] == 0)

    def s0_equal(fl):
        return all(
            fo.ids_sha256([i for i, f in zip(kept, fl) if f == fs_["fold"]])
            == fs_["row_ids_sha256"]
            for fs_ in s0["fold_summary"]
        )

    s0_match, s0_match_reg = s0_equal(fold), s0_equal(fold_reg)

    # ---- cached F12 inputs (pinned)
    tm = jload(b, TOKENS_MANIFEST, TOKENS_MANIFEST_SHA256)
    for f in ("rows.jsonl", "detail.jsonl"):
        b.sha256_file(TOKENS_DIR / f, sha256=tm["files"][f])
    trows = {
        json.loads(x)["id"]: json.loads(x)
        for x in (TOKENS_DIR / "rows.jsonl").read_text().splitlines()
        if x.strip()
    }
    detail = {}
    with open(TOKENS_DIR / "detail.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            if d["id"] in rows:
                detail[d["id"]] = d["per_tensor"]
    b.sha256_file(MATRIX, sha256=MATRIX_FILE_SHA256)
    M = dict(np.load(MATRIX, allow_pickle=False))
    m_ids = [str(x) for x in M["row_ids"]]
    fm = g["snap"]["content"]["feature_manifest"]["content"]

    X, DESC, STAT, recs_out, checks = [], [], [], {}, []
    for n, i in enumerate(kept, 1):
        r = rows[i]
        fdir = FEATURES / i
        pins = dict(zip(g["snap"]["content"]["feature_manifest"]["files"], fm[i]))
        fj = json.loads(b.read_bytes(fdir / "features.json", sha256=pins["features.json"]))
        b.sha256_file(fdir / "features.safetensors", sha256=pins["features.safetensors"])
        st = load_file(str(b.check(fdir / "features.safetensors")))["stats"]
        stats_by = {t["name"]: st[t["row"]].tolist() for t in fj["tensors"]}
        holder = ZOO / r["bytes_holder"]
        fg.authenticate_entry(b, holder, r["resolved_delta_hash"])
        stored = load_entry_delta(holder, base)
        # a root's edited tensors are its stored tensors (a5_power_fields.edit_increments, root branch)
        edited = sorted(stored)
        stored_names = list(edited)
        per = detail[i]
        recs, s1_err, th, changed = [], 0.0, {}, []
        for k in edited:
            delta = stored.pop(k)
            # the root path of a5_power_fields.edit_increments, one tensor at a time
            D = rotation.realised_increment(base[k], pf.stored_state(base[k], delta))
            th[k] = rf._tensor_hash(D)
            if torch.count_nonzero(D):
                changed.append(k)
            rec = fx.tensor_record(
                k,
                base[k],
                D,
                intended=delta.float(),
                a=per.get(k, {}).get("a"),
                q=per.get(k, {}).get("q"),
                stats=stats_by.get(k),
                sigma1=per.get(k, {}).get("sigma1"),
                ubar=ubar,
            )
            if rec["sigma1_computed"] is not None:
                s1 = rec["sigma1_computed"]
                s1_err = max(s1_err, abs(s1 - per[k]["sigma1"]) / max(per[k]["sigma1"], 1e-30))
            recs.append(rec)
            del D, delta
        if rf._hash(th) != trows[i]["increment_sha256"]:
            raise fg.AuthenticationError(
                f"{i}: realised increment differs from the F12 token build"
            )
        if set(changed) != set(per) or set(changed) != set(stats_by):
            raise ValueError(f"{i}: changed tensors differ between bytes, tokens and features")
        x = fx.aggregate(recs)
        X.append(x)
        mi = m_ids.index(i)
        v4 = M["v4"][mi]
        sel = M["token_row"] == mi
        T = np.zeros((186, 43))
        T[M["token_slot"][sel], :42] = M["tokens"][sel]
        T[M["token_slot"][sel], 42] = 1.0
        STAT.append(np.concatenate([x, v4, T.reshape(-1)]))
        edit = edits[i]
        DESC.append(fx.description_vector(edit, stored_names, trows[i].get("rank")))
        xd = dict(zip(fx.COLUMNS, x))
        energy = sum(rr["sq_norm"] for rr in recs)
        checks.append(
            {
                "id": i,
                "a_rel_err": abs(xd["a"] - trows[i]["a"]) / max(abs(trows[i]["a"]), 1e-30),
                "q_rel_err": abs(xd["q"] - trows[i]["q"]) / max(abs(trows[i]["q"]), 1e-30),
                "stats18_max_abs_err_vs_v4": float(np.max(np.abs(x[:18] - v4[:18]))),
                "fro_rel_err": abs(np.sqrt(energy) - fj["norms"]["fro_effective"])
                / max(fj["norms"]["fro_effective"], 1e-30),
                "lost_vs_features": [
                    sum(rr["n_lost"] for rr in recs),
                    fj["rounding_stats"]["n_lost"],
                ],
                "sigma1_writer_max_rel_err": s1_err,
            }
        )
        recs_out[i] = {"n_changed": len(changed), "n_stored": len(edited)}
        del stored
        guard.check()
        if n % 10 == 0:
            print(f"[inputs] {n}/{len(kept)} rows, {time.time() - t0:.0f}s", flush=True)

    X = np.array(X)
    x0 = fx.x_zero()
    nb = fx.name_blind(X)
    nofield = X[:, [fx.COLUMNS.index(c) for c in fx.NO_FIELD_COLUMNS]]
    st0 = np.concatenate([x0, M["x0_v4"], np.zeros(186 * 43)])
    arrays = {
        "row_ids": np.array(kept),
        "comps": np.array(comp),
        "families": np.array(fams),
        "fold": np.array(fold),
        "comps_registered": np.array(comp_reg),
        "fold_registered": np.array(fold_reg),
        "literal79": np.array([not rows[i]["g6_exposed_confirmatory"] for i in kept]),
        "a": X[:, fx.COLUMNS.index("a")],
        "q": X[:, fx.COLUMNS.index("q")],
        "x": X,
        "x0": x0,
        "x_nofield": nofield,
        "x0_nofield": x0[[fx.COLUMNS.index(c) for c in fx.NO_FIELD_COLUMNS]],
        "x_nameblind": nb,
        "x0_nameblind": fx.name_blind(x0[None, :])[0],
        "desc": np.array(DESC),
        "desc0": fx.desc_zero(),
        "statnn": np.array(STAT),
        "statnn0": st0,
    }
    npz = OUT / "inputs.npz"
    np.savez(npz, **arrays)
    peak = guard.stop()
    cpu = cpu_seconds() - c0
    append_ledger("inputs", cpu, time.time() - t0, peak["peak_rss_bytes"])
    worst = {
        k: max(c[k] for c in checks)
        for k in (
            "a_rel_err",
            "q_rel_err",
            "stats18_max_abs_err_vs_v4",
            "fro_rel_err",
            "sigma1_writer_max_rel_err",
        )
    }
    rec = {
        "schema": "msrm1-dev-v0-inputs",
        "label": LABEL,
        "outcomes_read": "none (this stage opens no evaluation record)",
        "inputs_npz_sha256": sha256_file(npz),
        "columns": {
            "x": fx.COLUMNS,
            "x_nofield": fx.NO_FIELD_COLUMNS,
            "x_nameblind": fx.NAME_BLIND_COLUMNS,
            "desc": fx.DESC_COLUMNS,
            "statnn": "x then v4 (1,378) then 186 x (42 token columns + presence), A5.8.3 slot order",
        },
        "zero_conventions": fx.ZERO_CONVENTIONS,
        "population": {
            "allowlist_n": len(allow),
            "allowlist_sha256": fo.ids_sha256(allow),
            "kept_n": len(kept),
            "kept_components": len(set(comp)),
            "literal79_kept": int(arrays["literal79"].sum()),
            "families": {f: fams.count(f) for f in sorted(set(fams))},
            "part_a_components": len(part_a),
            "forbidden_ids": len(g["forbidden"]),
        },
        "closure_audit": audit,
        "folds": {
            "rule": "F12.2 (20261111:f12:, strata abliteration/chain/lora_sft/mixed/random_dir, round-robin to 5)",
            "unit": "closure group (registered components merged by the closure audit)",
            "equal_to_population_s0": s0_match,
            "fold_sizes": {str(f): fold.count(f) for f in sorted(set(fold))},
            "registered_components_folds_equal_to_population_s0": s0_match_reg,
            "registered_fold_sizes": {str(f): fold_reg.count(f) for f in sorted(set(fold_reg))},
            "closure_groups": len(set(comp)),
            "closure_groups_literal79": len(
                {c for c, i in zip(comp, kept) if not rows[i]["g6_exposed_confirmatory"]}
            ),
        },
        "base": {"state_hash": base_hash, **ubar_rec},
        "cross_checks_worst": worst,
        "cross_checks": checks,
        "rows": recs_out,
        "resource": peak,
        "cpu_seconds": cpu,
        "wall_seconds": time.time() - t0,
        "read_boundary": b.record(),
    }
    write_json(OUT / "inputs.json", rec)
    print(json.dumps({"kept": len(kept), "worst": worst, "cpu_h": cpu / 3600}, indent=1))


# =================================================================================================== fit


def load_inputs(path=OUT / "inputs.npz", rec_path=OUT / "inputs.json"):
    rec = json.loads(Path(rec_path).read_text())
    if sha256_file(path) != rec["inputs_npz_sha256"]:
        raise fg.AuthenticationError("inputs.npz differs from its recorded sha256")
    return dict(np.load(path, allow_pickle=False)), rec


def arms_for(head):
    from wsp.msrm.models import ArmSpec

    return [
        ArmSpec("f3u", base="f3u"),
        ArmSpec("zero_a", base="zero_a"),
        ArmSpec("const", base="const"),
        ArmSpec("msrm1", base="f3u", inputs="x", head=head),
        ArmSpec("statnn", base="f3u", inputs="statnn", head="statnn"),
        ArmSpec("scripted", base="const", inputs="x_nofield", head=head),
        ArmSpec("desc_only", base="const", inputs="desc", head=head),
        ArmSpec("name_blind", base="f3u", inputs="x_nameblind", head=head),
        ArmSpec("field_ablated", base="f3u", inputs="x_nofield", head=head),
        ArmSpec("f3u_plus_desc", base="f3u", inputs="desc", head=head),
    ]


PAIRS = {  # (arm, reference) contrasts reported with bootstrap intervals
    "msrm_vs_f3u": ("msrm1", "f3u"),
    "msrm_vs_desc": ("msrm1", "desc_only"),
    "msrm_vs_scripted": ("msrm1", "scripted"),
    "msrm_vs_statnn": ("msrm1", "statnn"),
    "msrm_vs_field_ablated": ("msrm1", "field_ablated"),
    "msrm_vs_name_blind": ("msrm1", "name_blind"),
    "msrm_vs_f3u_plus_desc": ("msrm1", "f3u_plus_desc"),
}


def analyse(data, fold, families, oof, draws):
    from wsp.msrm import evaluate as ev

    y, comps = data.y, data.comps
    ok = np.all([np.isfinite(v) for v in oof.values()], axis=0)
    if not np.all(ok):
        raise ValueError("arms do not cover identical rows")
    vs_f3u = ev.component_bootstrap(y, oof, comps, "f3u", draws=draws)
    per_arm = {
        k: {
            "mae": float(np.mean(np.abs(y - v))),
            "sum_abs_err": float(np.sum(np.abs(y - v))),
            **({"C_vs_f3u": vs_f3u[k]} if k != "f3u" else {}),
        }
        for k, v in oof.items()
    }
    pairs = {}
    for key, (arm, ref) in PAIRS.items():
        pairs[key] = ev.component_bootstrap(
            y, {arm: oof[arm], ref: oof[ref]}, comps, ref, draws=draws
        )[arm]
    fam = {}
    for f in sorted(set(families)):
        m = families == f
        fam[f] = {
            "rows": int(m.sum()),
            "components": len(set(comps[m])),
            "mae": {k: float(np.mean(np.abs(y[m] - v[m]))) for k, v in oof.items()},
            "C_vs_f3u": {
                k: ev.contrast(y[m], v[m], oof["f3u"][m]) for k, v in oof.items() if k != "f3u"
            },
        }
    cal = ev.mz_calibration(y, oof["msrm1"], comps, draws=draws)
    return {
        "n_rows": len(y),
        "n_components": len(set(comps)),
        "per_arm": per_arm,
        "pairs": pairs,
        "by_family": fam,
        "calibration_msrm1": dict(cal, failure=ev.calibration_failure(cal)),
        "calibration_f3u": ev.mz_calibration(y, oof["f3u"], comps, draws=draws),
        "labels_msrm1": ev.label_table(y, oof["msrm1"]),
        "labels_f3u": ev.label_table(y, oof["f3u"]),
    }


def stage_fit(args):
    import torch

    from wsp.msrm import evaluate as ev
    from wsp.msrm import folds as fo
    from wsp.msrm import models as md
    from wsp.msrm import outcomes as oc

    t0, c0 = time.time(), cpu_seconds()
    if ledger_hours() >= CPU_CAP_HOURS:
        raise SystemExit("CPU cap exhausted")
    torch.set_num_threads(args.threads)
    guard = fg.ResourceGuard(build=BUILD, threads=args.threads)
    guard.launch(estimate_bytes=1_500_000_000, breach_path=OUT / "fit_breach.json")
    A, irec = load_inputs()
    b = boundary()
    g = load_governance(b)
    kept = [str(i) for i in A["row_ids"]]
    if not set(kept) <= set(g["allow"]):
        raise fg.BoundaryError("inputs carry ids outside the allowlist")
    pins = {
        i: g["snap"]["content"]["tf_v1_records"]["primary"][i] for i in [*g["allow"], oc.REFERENCE]
    }
    reader = oc.OutcomeReader(
        EVALS,
        g["allow"],
        oc.EXISTING_ROOTS_SHA256,
        pins,
        g["forbidden"],
        data_root=DATA_ROOT,
    )
    if args.synthetic_outcomes:  # dry run of the whole fit path: no outcome is opened
        rng = np.random.default_rng(12345)
        xa = A["a"]
        y = 0.9 * xa + 0.3 * np.tanh(A["x"][:, 20]) * np.abs(xa) + 1e-3 * rng.normal(size=len(xa))
        reader.reads.clear()
    else:
        y = np.array([reader.delta_y(i)[0] for i in kept])
    data = md.Dataset(
        row_ids=np.array(kept),
        comps=A["comps"],
        y=y,
        a=A["a"],
        q=A["q"],
        inputs={
            "x": (A["x"], A["x0"]),
            "x_nofield": (A["x_nofield"], A["x0_nofield"]),
            "x_nameblind": (A["x_nameblind"], A["x0_nameblind"]),
            "desc": (A["desc"], A["desc0"]),
            "statnn": (A["statnn"], A["statnn0"]),
        },
    )
    elig = md.eligible_rows(data)
    if not elig.all():
        raise ValueError(f"{(~elig).sum()} rows lack a required input (excluded from every arm)")
    cfg = md.FitConfig()
    if args.fast:
        if not args.synthetic_outcomes:
            raise SystemExit("--fast is for the synthetic dry run only")
        import dataclasses

        cfg = dataclasses.replace(cfg, max_epochs=10, patience=3, tree_grid=(20, 40))
    fams = A["families"]
    fold = A["fold"]
    draws = args.draws
    results = {"label": LABEL, "schema": "msrm1-dev-v0-results"}

    # ---- head class, chosen once by grouped inner CV on all development rows
    head, table = md.select_head_class(data, np.arange(len(y)), "x", cfg)
    results["head_class"] = {
        "chosen": head,
        "rule": "grouped CV over the four 20261110:lol-inner: component folds of all 123 rows; lowest pooled MAE "
        "of F3u + head; ties keep mlp, ridge, tree order; each candidate's own hyperparameters nested",
        "pooled_mae": table,
        "mlp_params_at_inputs": md.mlp_param_count(A["x"].shape[1]),
        "mlp_params_at_48": md.mlp_param_count(48),
    }
    print(f"[fit] head class {head} {table} ({time.time() - t0:.0f}s)", flush=True)
    guard.check()

    # ---- primary: all arms on identical rows / folds
    rec = {}
    oof = md.run_cv(arms_for(head), data, fold, cfg, record=rec)
    results["population_123"] = analyse(data, fold, fams, oof, draws)
    results["fit_records_123"] = {
        k: {
            str(f): {kk: vv for kk, vv in v.items() if kk != "inner_best_epochs"}
            for f, v in r.items()
        }
        for k, r in rec.items()
    }
    results["zero_fit_a_E0"] = 1 - float(np.sum((y - data.a) ** 2) / np.sum(y**2))
    print(f"[fit] arms done ({time.time() - t0:.0f}s)", flush=True)

    # ---- nested per-fold class selection (honesty check of the once-chosen class)
    nested = np.full(len(y), np.nan)
    per_fold = {}
    for f in sorted(set(fold.tolist())):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        c, _ = md.select_head_class(data, tr, "x", cfg)
        per_fold[str(f)] = c
        nested[te], _ = md.fit_predict_arm(md.ArmSpec("m", "f3u", "x", c), data, tr, te, cfg)
    results["nested_class_selection"] = {
        "per_fold_class": per_fold,
        "C_vs_f3u": ev.component_bootstrap(
            y, {"nested": nested, "f3u": oof["f3u"]}, data.comps, "f3u", draws=draws
        )["nested"],
    }
    guard.check()

    # ---- shuffled-outcome control (three permutations)
    shuf = {}
    for s in (0, 1, 2):
        ds = md.shuffle_outcomes(data, seed=s)
        o = md.run_cv(
            [md.ArmSpec("f3u", "f3u"), md.ArmSpec("msrm1", "f3u", "x", head)], ds, fold, cfg
        )
        shuf[str(s)] = ev.component_bootstrap(ds.y, o, ds.comps, "f3u", draws=draws)["msrm1"]
    results["shuffled_outcome_control"] = {
        "rule": "outcomes permuted across rows (numpy PCG64 seeds 0/1/2); F3u and MSRM-1 refitted; C(MSRM-1:F3u)",
        "C": shuf,
        "shows_no_gain": all(v["C"] < 0.05 for v in shuf.values()),
    }
    print(f"[fit] shuffled done ({time.time() - t0:.0f}s)", flush=True)

    # ---- leave-one-family-out (descriptive)
    lofo = {}
    lofo_arms = [a for a in arms_for(head) if a.name in ("f3u", "msrm1", "scripted", "desc_only")]
    for fam in ("abliteration", "chain", "lora_sft", "random_dir"):
        # registered components (F12a.1(i): the one mixed component always trains); closure merging would make
        # most groups "mixed" and leave nothing to hold out
        tr, te = fo.lofo_split(list(A["comps_registered"]), list(fams), fam)
        tr, te = np.array(tr, dtype=int), np.array(te, dtype=int)
        if len(te) == 0:
            lofo[fam] = {"test_rows": 0}
            continue
        p = {a.name: md.fit_predict_arm(a, data, tr, te, cfg)[0] for a in lofo_arms}
        lofo[fam] = {
            "test_rows": len(te),
            "test_components": len(set(data.comps[te])),
            "mae": {k: float(np.mean(np.abs(y[te] - v))) for k, v in p.items()},
            "C_vs_f3u": {k: ev.contrast(y[te], v, p["f3u"]) for k, v in p.items() if k != "f3u"},
        }
    results["leave_one_family_out"] = lofo
    guard.check()

    # ---- status-literal 79 / 52 sensitivity (folds recomputed by the F12.2 rule on the subset)
    lit = np.flatnonzero(A["literal79"])
    sub = md.subset(data, lit)
    fmap = fo.outer_folds(list(sub.comps), list(fams[lit]))
    sfold = np.array([fmap[c] for c in sub.comps])
    soof = md.run_cv(arms_for(head), sub, sfold, cfg)
    results["population_79_sensitivity"] = analyse(sub, sfold, fams[lit], soof, draws)
    print(f"[fit] subset done ({time.time() - t0:.0f}s)", flush=True)

    # ---- registered-component sensitivity: the frozen F12 S0 folds, components not merged by the closure audit
    reg = md.Dataset(
        row_ids=data.row_ids,
        comps=A["comps_registered"],
        y=y,
        a=data.a,
        q=data.q,
        inputs=data.inputs,
    )
    roof = md.run_cv(arms_for(head), reg, A["fold_registered"], cfg)
    results["registered_components_sensitivity"] = analyse(
        reg, A["fold_registered"], fams, roof, draws
    )
    print(f"[fit] registered-components sensitivity done ({time.time() - t0:.0f}s)", flush=True)

    # ---- kill criteria (on the 123-row population, closure groups as units)
    pairs = results["population_123"]["pairs"]
    kc = ev.kill_criteria(pairs)
    kc["provenance_or_calibration"] = {
        "rule": "any allowlist / closure / authentication failure (the run stops) or MSRM-1 calibration failure "
        "(Mincer-Zarnowitz slope 95% interval entirely outside [0.5, 2.0])",
        "fires": bool(results["population_123"]["calibration_msrm1"]["failure"]),
        "meaning": ev.MEANING,
    }
    peak = guard.stop()
    cpu = cpu_seconds() - c0
    append_ledger(
        "fit-dryrun" if args.synthetic_outcomes else "fit",
        cpu,
        time.time() - t0,
        peak["peak_rss_bytes"],
    )
    total = ledger_hours()
    kc["cpu_cap"] = {
        "rule": f"total CPU-h of the development jobs > {CPU_CAP_HOURS}",
        "fires": bool(total > CPU_CAP_HOURS),
        "meaning": ev.MEANING,
    }
    results["kill_criteria"] = kc
    results["resources"] = {
        "fit_cpu_hours": cpu / 3600,
        "fit_wall_seconds": time.time() - t0,
        "ledger_total_cpu_hours": total,
        "ledger": [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()],
        "fit_peak_rss_bytes": peak["peak_rss_bytes"],
        "inputs_peak_rss_bytes": irec["resource"]["peak_rss_bytes"],
        "gpu_hours": 0.0,
        "llm_tokens": 0,
    }
    results["config"] = {
        "fit": cfg.as_dict(),
        "bootstrap": {
            "draws": draws,
            "seed": ev.BOOT_SEED,
            "generator": "PCG64",
            "unit": "component",
        },
        "h_star": ev.H_STAR,
        "inputs_npz_sha256": irec["inputs_npz_sha256"],
        "code_commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
    }
    results["outcome_reads"] = {
        "allowlist_sha256": oc.EXISTING_ROOTS_SHA256,
        "n_ids_read": len({r["id"] for r in reader.reads}),
        "reference": oc.REFERENCE,
        "files_read": len(reader.reads),
        "reads_sha256": hashlib.sha256(
            json.dumps(reader.reads, sort_keys=True).encode()
        ).hexdigest(),
    }
    per_row = {
        "row_ids": kept,
        "y": y.tolist(),
        "oof": {k: v.tolist() for k, v in oof.items()},
        "reads": reader.reads,
    }
    results["per_row_file_sha256"] = write_json(
        OUT / ("dryrun_per_row_oof.json" if args.synthetic_outcomes else "per_row_oof.json"),
        per_row,
    )
    out_dir = OUT / "dryrun" if args.synthetic_outcomes else REPORT_DIR
    if args.synthetic_outcomes:
        results["label"] = "SYNTHETIC DRY RUN (no outcome opened): " + LABEL
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"msrm1-dev-v0-{args.date}.json"
    write_json(out_json, results)
    print(f"[fit] wrote {out_json}; CPU {cpu / 3600:.3f} h; total {total:.3f} h", flush=True)


# ================================================================================================ report

ARM_LABELS = {
    "f3u": "F3u (unpenalised β₀ + β₁Σa + β₂Σq)",
    "zero_a": "zero-fit a (ŷ = a)",
    "const": "intercept only (reference)",
    "msrm1": "**MSRM-1** (F3u + head over x(D))",
    "statnn": "same-input StatNN (F3u + LightGBM over x ∪ v4 ∪ 186×43 tokens)",
    "scripted": "scripted reader (intercept + head over x without field columns)",
    "desc_only": "description-only (intercept + head over family/dose/role/layer/rank)",
    "name_blind": "name-blind MSRM-1",
    "field_ablated": "field-ablated MSRM-1 (F3u + head over x without field columns)",
    "f3u_plus_desc": "F3u + description head (descriptive)",
}


def _f(v, nd=3):
    return "n/a" if v is None else f"{v:+.{nd}f}" if isinstance(v, float) else str(v)


def _ci(c):
    lo, hi = c["ci95"]
    return f"[{_f(lo)}, {_f(hi)}]"


def _arm_table(pop):
    lines = [
        "| arm | pooled OOF MAE (nats/token) | Σ\\|e\\| | C vs F3u | 95% component-bootstrap interval |",
        "|---|---|---|---|---|",
    ]
    for k, v in pop["per_arm"].items():
        c = v.get("C_vs_f3u")
        lines.append(
            f"| {ARM_LABELS.get(k, k)} | {v['mae']:.5f} | {v['sum_abs_err']:.4f} | "
            f"{_f(c['C']) if c else '—'} | {_ci(c) if c else '—'} |"
        )
    return "\n".join(lines)


def _pairs_table(pop):
    lines = ["| contrast | C | 95% interval |", "|---|---|---|"]
    for k, v in pop["pairs"].items():
        arm, ref = PAIRS[k]
        lines.append(f"| C({arm} : {ref}) | {_f(v['C'])} | {_ci(v)} |")
    return "\n".join(lines)


def _family_table(pop, arms=("msrm1", "statnn", "scripted", "desc_only", "zero_a")):
    head = (
        "| family | rows / components | MAE F3u | " + " | ".join(f"C({a}:F3u)" for a in arms) + " |"
    )
    lines = [head, "|" + "---|" * (3 + len(arms))]
    for f, v in pop["by_family"].items():
        cells = " | ".join(_f(v["C_vs_f3u"][a]) for a in arms)
        lines.append(f"| {f} | {v['rows']} / {v['components']} | {v['mae']['f3u']:.5f} | {cells} |")
    return "\n".join(lines)


def stage_report(args):
    from collections import Counter

    rdir = OUT / "dryrun" if args.synthetic_outcomes else REPORT_DIR
    res = json.loads((rdir / f"msrm1-dev-v0-{args.date}.json").read_text())
    inp = json.loads((OUT / "inputs.json").read_text())
    p, s = res["population_123"], res["population_79_sensitivity"]
    kc, hc = res["kill_criteria"], res["head_class"]
    m = p["pairs"]["msrm_vs_f3u"]
    fires = [k for k, v in kc.items() if v["fires"]]
    au = inp["closure_audit"]
    rs = res["resources"]
    md = []
    w = md.append
    w(f"# MSRM-1 development v0 — {args.date}")
    w("")
    w(
        f"> **{LABEL}.** Every number below comes from out-of-fold predictions on 123 root edits whose TF-v1 "
        "outcomes were already on file and treated as exposed. Kill-criterion outcomes are development decisions, "
        "not scientific nulls. Nothing here changes F8, F9, H5, F12-I or F5.5, and Part A is untouched."
    )
    w("")
    w(
        "Spec: Decision 1 of `docs/council/2026-11-16/SYNTHESIS-r7.md` (main `741d88f`). Machine-readable results: "
        f"`docs/reviews/msrm/msrm1-dev-v0-{args.date}.json`. Code: `src/wsp/msrm/`, `scripts/msrm1_dev.py`, "
        f"commit `{res['config']['code_commit'][:12]}`."
    )
    w("")
    w("## BLUF")
    w("")
    w(
        f"- Head class (chosen once by grouped inner CV): **{hc['chosen']}** "
        f"(pooled MAE {', '.join(f'{k} {v:.5f}' for k, v in hc['pooled_mae'].items())})."
    )
    w(
        f"- C(MSRM-1 : F3u) = **{_f(m['C'])}**, 95% component-bootstrap interval {_ci(m)} "
        f"({p['n_rows']} rows / {p['n_components']} closure groups as resampling units)."
    )
    w(
        f"- Development kill criteria firing: **{', '.join(fires) if fires else 'none'}** "
        "(development decisions, not scientific nulls)."
    )
    w(
        f"- Shuffled-outcome control: C = {', '.join(_f(v['C']) for v in res['shuffled_outcome_control']['C'].values())}"
        f" (shows no gain: {res['shuffled_outcome_control']['shows_no_gain']})."
    )
    w(
        f"- Status-literal 79 / 52 sensitivity ({s['n_components']} closure groups): C(MSRM-1 : F3u) = "
        f"{_f(s['pairs']['msrm_vs_f3u']['C'])} {_ci(s['pairs']['msrm_vs_f3u'])}. Registered-component units "
        f"(89, frozen F12 S₀ folds): {_f(res['registered_components_sensitivity']['pairs']['msrm_vs_f3u']['C'])} "
        f"{_ci(res['registered_components_sensitivity']['pairs']['msrm_vs_f3u'])}."
    )
    w(
        f"- Cost: {rs['ledger_total_cpu_hours']:.3f} CPU-h in total (cap 10), 0 GPU-h, 0 LLM tokens; peak RSS "
        f"{max(rs['fit_peak_rss_bytes'], rs['inputs_peak_rss_bytes']) / 1e9:.2f} GB (cap 4 GB)."
    )
    w("")
    w(
        f"## Per-arm results, {p['n_rows']} rows / {p['n_components']} closure groups "
        "(all arms on identical rows and folds)"
    )
    w("")
    w(_arm_table(p))
    w("")
    w(
        "C = 1 − Σ|e_arm| / Σ|e_F3u| over pooled out-of-fold errors; seeds 0/1/2 averaged before the loss. "
        f"Intervals resample whole components, {res['config']['bootstrap']['draws']} draws, PCG64 seed "
        f"{res['config']['bootstrap']['seed']}, percentile. Zero-fit E₀ = 1 − Σ(Δy − a)²/ΣΔy² = "
        f"{_f(res['zero_fit_a_E0'])}."
    )
    w("")
    w("### MSRM-1 against the controls")
    w("")
    w(_pairs_table(p))
    w("")
    w("## Head class")
    w("")
    w(
        f"Rule: {hc['rule']}. MLP head: {hc['mlp_params_at_inputs']} parameters at {len(inp['columns']['x'])} "
        f"inputs ({hc['mlp_params_at_48']} at 48). Nested per-outer-fold selection, as an honesty check of the "
        f"once-chosen class: classes {res['nested_class_selection']['per_fold_class']}, C vs F3u "
        f"{_f(res['nested_class_selection']['C_vs_f3u']['C'])} {_ci(res['nested_class_selection']['C_vs_f3u'])}."
    )
    w("")
    w("## Family split (descriptive; 105 chain rows dominate)")
    w("")
    w(_family_table(p))
    w("")
    w("### Leave-one-family-out (descriptive; the mixed component always trains)")
    w("")
    w(
        "| held-out family | rows / components | MAE F3u | C(MSRM-1:F3u) | C(scripted:F3u) | C(desc:F3u) |"
    )
    w("|---|---|---|---|---|---|")
    for f, v in res["leave_one_family_out"].items():
        if not v.get("test_rows"):
            w(f"| {f} | 0 | — | — | — | — |")
            continue
        c = v["C_vs_f3u"]
        w(
            f"| {f} | {v['test_rows']} / {v['test_components']} | {v['mae']['f3u']:.5f} | {_f(c['msrm1'])} | "
            f"{_f(c['scripted'])} | {_f(c['desc_only'])} |"
        )
    w("")
    w(
        f"## Sensitivity: status-literal subset, 79 rows / 52 registered components = {s['n_components']} "
        "closure groups (folds recomputed by the F12.2 rule)"
    )
    w("")
    w(_arm_table(s))
    w("")
    w(_pairs_table(s))
    w("")
    r = res["registered_components_sensitivity"]
    w(
        "## Sensitivity: the 89 registered components as units (frozen F12 S₀ folds, no closure merges)"
    )
    w("")
    w(_arm_table(r))
    w("")
    w(_pairs_table(r))
    w("")
    w("## Calibration and descriptive three-way labels")
    w("")
    c1, c0 = p["calibration_msrm1"], p["calibration_f3u"]
    w(
        f"Mincer–Zarnowitz slope of Δy on the OOF prediction: MSRM-1 {c1['slope']:.3f} "
        f"[{c1['slope_ci95'][0]:.3f}, {c1['slope_ci95'][1]:.3f}], F3u {c0['slope']:.3f} "
        f"[{c0['slope_ci95'][0]:.3f}, {c0['slope_ci95'][1]:.3f}]."
    )
    lm, lf = p["labels_msrm1"], p["labels_f3u"]
    w(
        f"Sign agreement where both truth and prediction resolve beyond h* = 0.0049397 nats/token: MSRM-1 "
        f"{_f(lm['sign_agreement_when_both_resolved'])} on {lm['n_both_resolved']} rows; F3u "
        f'{_f(lf["sign_agreement_when_both_resolved"])} on {lf["n_both_resolved"]} rows. "0" labels are '
        "unresolved, never equivalence."
    )
    w("")
    w("## Development kill criteria")
    w("")
    w("| criterion | rule | fires |")
    w("|---|---|---|")
    for k, v in kc.items():
        w(f"| {k} | {v['rule']} | **{v['fires']}** |")
    w("")
    w(
        "All are development decisions (SYNTHESIS-r7), not scientific nulls. The margin rule of k2 and k2b (point "
        "C ≤ 0 or lower bound ≤ 0) and the calibration rule are the implementer's readings, committed before any "
        "outcome was read."
    )
    w("")
    w("## Rows, closure and provenance")
    w("")
    pop_ = inp["population"]
    w(
        f"- Allowlist: {pop_['allowlist_n']} ids, sorted-id sha256 `{pop_['allowlist_sha256']}` (= F12.2). Kept "
        f"after the closure audit: {pop_['kept_n']} rows in {pop_['kept_components']} closure groups; families "
        f"{pop_['families']}."
    )
    w(
        f"- Closure audit joins: {', '.join(au['joins'])}. Edge counts {au['edge_counts']}. Cross-component joins: "
        f"{len(au['cross_component_joins'])} ({dict(Counter(e['kind'] for e in au['cross_component_joins']))}); "
        f"{len(au['merge'])} registered components merged into others; rows excluded: "
        f"{au['excluded'] or 'none'}."
    )
    w(
        f"- Folds: {inp['folds']['rule']}, applied to the {inp['folds']['closure_groups']} closure groups "
        f"({inp['folds']['closure_groups_literal79']} in the 79-row subset); sizes {inp['folds']['fold_sizes']}. On "
        "the 89 registered components the same rule reproduces the frozen F12 S₀ manifest: "
        f"{inp['folds']['registered_components_folds_equal_to_population_s0']} "
        f"(sizes {inp['folds']['registered_fold_sizes']})."
    )
    cw = inp["cross_checks_worst"]
    w(
        f"- Inputs cross-checks (worst over rows): Σa relative error vs the F12 token build {cw['a_rel_err']:.2e}, "
        f"Σq {cw['q_rel_err']:.2e}, stats18 vs pinned v4 {cw['stats18_max_abs_err_vs_v4']:.2e}, ||D|| vs features "
        f"{cw['fro_rel_err']:.2e}, writer σ₁ vs tokens {cw['sigma1_writer_max_rel_err']:.2e}. Every realised "
        "increment matched the token build's `increment_sha256`."
    )
    o = res["outcome_reads"]
    w(
        f"- Outcome reads: {o['files_read']} files of {o['n_ids_read']} ids (the allowlisted roots plus the common "
        f"`{o['reference']}` reference), each sha256-authenticated against the A5 snapshot pins, no directory "
        f"listed; read log sha256 `{o['reads_sha256'][:16]}…`. No Part A file, variant, reference or component was "
        "opened; no outcome of the 147 purchasable roots exists or was scored."
    )
    w("")
    w("## Resources")
    w("")
    w("| stage | CPU-h | wall (s) | peak RSS (GB) |")
    w("|---|---|---|---|")
    for r in rs["ledger"]:
        w(
            f"| {r['stage']} | {r['cpu_seconds'] / 3600:.3f} | {r['wall_seconds']:.0f} | "
            f"{r['peak_rss_bytes'] / 1e9:.2f} |"
        )
    w(f"| **total** | **{rs['ledger_total_cpu_hours']:.3f}** | | |")
    w("")
    w("## Implementer choices (fixed before outcomes were read)")
    w("")
    for line in IMPLEMENTER_CHOICES:
        w(f"- {line}")
    w("")
    out = rdir / f"msrm1-dev-v0-{args.date}.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")


IMPLEMENTER_CHOICES = [
    (
        "x(D) = the 48-vector plus six scripted columns (54). a_t, q_t and σ₁ come from the pinned F12 token build; the "
        "stats.py statistics from the pinned v4 features; ⟨D, W⟩, ‖W‖, the intended delta and the leading output vectors "
        "from a CPU pass over the stored bytes and the pinned base weights, authenticated against `increment_sha256`."
    ),
    (
        "Unembedding mean ū = mean over vocabulary rows of the unembedding × (1 + final-norm weight); the |cos| column "
        "covers residual writers (o, down) only, energy-weighted."
    ),
    (
        "Fraction changed = changed tensors / 186 supported matrix slots; rounding loss = intended nonzeros realised as 0 "
        "/ intended nonzeros."
    ),
    (
        "Input transform: signed asinh by the training median |x| of nonzero entries, then standardisation; all statistics "
        "from training rows only."
    ),
    (
        "MLP head follows F5.1 / F12b.1 (zero-initialised output, residual target scaled by its training SD); the tree "
        "head uses LightGBM with the F12 settings and an L1 objective; ridge λ from {1e-2, …, 1e3}."
    ),
    (
        "Scripted reader and description-only arms have an intercept-only base (no field anywhere); field-ablated MSRM-1 "
        "keeps F3u and drops the ten field columns from the head; StatNN is the F12.3 augmented comparator with F3u in "
        "place of ridge F3."
    ),
    (
        "Name-blind: depth buckets reordered by |a_b| and role shares sorted (identity-free order statistics); "
        "delta-energy-weighted depth dropped."
    ),
    (
        "Every residual head is fitted as the function it is evaluated as, h(x) − h(x(0)): the MLP is trained on that "
        "difference, ridge on the anchored design x − x(0) without intercept, and the tree head with an anchor row "
        "(x(0), 0) weighted by the training-row count (found in the synthetic dry run, before any outcome was read)."
    ),
    (
        "Leave-one-family-out uses the 89 registered components (F12a.1(i)); its rows are not closure-independent of "
        "the training rows, so it is descriptive only."
    ),
    (
        "Head class chosen once on all 123 rows (literal reading of 'chosen once'); a nested per-fold re-selection is "
        "reported as an honesty check. The 79-row sensitivity reuses the chosen class."
    ),
    (
        "Description-only uses edit.json parameters and the names of stored tensors; 'F3u + description' is reported "
        "descriptively to separate family priors from the field."
    ),
    (
        "Closure audit: ancestry, generator, per-direction random seeds (seed + 1000·layer + 100·di, as wsp.zoo.build "
        "draws them), exact abliteration directions (refusal fingerprint + direction_index) and duplicate delta bytes. "
        "The registered A5 closure does not join directions; joining them merges the 89 registered components into the "
        "closure groups used as fold and bootstrap units (registered components reported as a sensitivity). "
        "provenance.reuse_key is not a join (an environment hash without parameters or seed). Part A edit records are "
        "never opened, so a direction shared with a Part A entry is not detectable."
    ),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inputs", "fit", "report"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d", time.gmtime()))
    ap.add_argument("--synthetic-outcomes", action="store_true", help="dry run; opens no outcome")
    ap.add_argument("--fast", action="store_true", help="dry run only: short training budgets")
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    {"inputs": stage_inputs, "fit": stage_fit, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
