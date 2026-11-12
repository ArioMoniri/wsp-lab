#!/usr/bin/env python3
"""F12e: real-token qualification and the C1 token facts (Board A F12c review, "Ordered path to stage 1", steps 2-3;
PREREG_H1_TF F12a.1(e)-(k), F12a.2(c), (e), (f), (i), F12a.6(c), F12a.8).

Reads the committed frozen token manifest ``docs/prereg/a5_manifests/f12/tokens_manifest.json`` (sha256 pinned) and,
only after every file's sha256 equals the manifest's, the token run's outputs (``tokens.safetensors``, ``rows.jsonl``,
``detail.jsonl``, ``kframes.json``). These are field forecasts and edit-derived inputs, not outcomes. Nothing under
``data/evals``, no evaluation record, no noise payload and no Part A file is read.

* ``qualify``: outcome-free builder/verifier checks of the frozen tokens against the committed population and the
  F12a requirements: token files, candidate code identities, population identity, 270 rows / 234 components, the
  F12a.2(e) exclusions, the finite token schema, the S_k fold assignments, fold nesting (inner folds inside outer
  training; leave-one-family-out with the mixed component always training), the F5 gauge audit's inputs
  (F12a.2(i), F12a.5(d)) and the inputs of the simulation's ``f12-token-matrix-v1`` matrix. Each check passes or
  fails; ``first_failure`` names the first failure in that order. The omitted-energy diagnostic is reported
  descriptively and changes no rule.
* ``token_facts``: what C1 needs from the tokens (row eligibility, eligible-row folds by hash, token input hashes,
  G2 / probe / field / base identities, the 24 ``down_proj`` base tensor hashes and the F12-I LoRA parent / support /
  base tensor hashes). ``scripts/f12_c1.py build_draft`` fills the C1 draft from the committed facts file.
* ``candidate_identities``: the frozen candidate code, configuration, payload and dependency identities used by steps
  3-5; ``identity_drift`` names every file whose bytes changed since.

    python scripts/f12_qualify.py verify
    python scripts/f12_qualify.py base-hashes --out docs/prereg/f12_manifests/f12e_base_tensor_hashes.json
    python scripts/f12_qualify.py facts --base-hashes docs/prereg/f12_manifests/f12e_base_tensor_hashes.json
    python scripts/f12_qualify.py identities
    python scripts/f12_qualify.py qualify --identities docs/prereg/f12_manifests/f12e_candidate_identities.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import f12_population as fpop
from scripts import f12_simulation as fs
from scripts import f12_tokens as ft

TOKEN_MANIFEST = ROOT / "docs/prereg/a5_manifests/f12/tokens_manifest.json"
TOKEN_MANIFEST_SHA256 = "424333d9b6b2e19a0586b34096b1968d1de054e6ec625da9bceb4507b567001f"
POPULATION = ROOT / "docs/prereg/a5_manifests/f12/population.json"
MANIFEST_DIR = ROOT / "docs/prereg/f12_manifests"
FACTS_PATH = MANIFEST_DIR / "f12e_token_facts.json"
BASE_HASHES_PATH = MANIFEST_DIR / "f12e_base_tensor_hashes.json"
F12E_QUALIFICATION_PATH = MANIFEST_DIR / "f12e_qualification.json"
# F12f: re-pinned identities and the re-run qualification (the F12e records stay as committed)
IDENTITIES_PATH = MANIFEST_DIR / "f12f_candidate_identities.json"
QUALIFICATION_PATH = MANIFEST_DIR / "f12f_qualification.json"
GAUGE_MANIFEST = ROOT / "docs/prereg/a5_manifests/f12/gauge_tokens_manifest.json"
MATRIX_MANIFEST = ROOT / "docs/prereg/a5_manifests/f12/matrix_manifest.json"
SNAPSHOT = ROOT / "docs/prereg/a5_manifests/snapshot.json"
RESTRICTED_SHA256 = "44d794030daba753ddcef5f066d92ed2bcb0271d0e73b5441de1de427d086fb5"
REQUIRED_FILES = (
    "aq.json",
    "detail.jsonl",
    "kframes.json",
    "kframes.safetensors",
    "pricing.json",
    "rows.jsonl",
    "tokens.safetensors",
)
EXPECTED_COUNTS = {"rows": 270, "components": 234}
EXPECTED_STRATA = {"abliteration": 83, "chain": 71, "lora_sft": 29, "mixed": 1, "random_dir": 50}
EXPECTED_SOURCES = {"existing-TF-v1": 123, "purchasable": 147}
LOFO_FAMILIES = ("abliteration", "chain", "lora_sft", "random_dir")
LOFO_TRAIN_RULE = (
    "every S_k component outside eval_components; the mixed component always trains and is never evaluated "
    "(F12a.1(i))"
)
OMITTED_THRESHOLD = 0.01
CANDIDATE_CODE = (
    "scripts/f12_simulation.py",
    "scripts/f12_budget.py",
    "scripts/f12_c1.py",
    "scripts/f12_qualify.py",
    "scripts/f12_payload_extract.py",
    "scripts/f12_population.py",
    "scripts/f12_tokens.py",
    "scripts/a5_build_panels.py",
    "scripts/a5_simulation.py",
    "scripts/tfv1_requalify.py",
    "scripts/f12_gauge_tokens.py",
    "scripts/f12_matrix.py",
    "src/wsp/features/pool.py",
    "src/wsp/features/extract.py",
)
SCHEMA_FACTS = "f12e-token-facts-v1"
SCHEMA_QUAL = "f12f-qualification-v1"
SCHEMA_IDENT = "f12f-candidate-identities-v1"
SCHEMA_BASE = "f12e-base-tensor-hashes-v1"
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class TokenFileMismatch(ValueError):
    pass


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def layer_of(name):
    m = _LAYER.search(name)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------------------------------------- loading


def verify_token_files(manifest, out_dir) -> dict:
    """{file: sha256} for every required token file; raises ``TokenFileMismatch`` on a missing manifest entry,
    a missing file or a digest differing from the manifest's."""
    out_dir = Path(out_dir)
    files = manifest.get("files") or {}
    got = {}
    for name in REQUIRED_FILES:
        if name not in files:
            raise TokenFileMismatch(f"{name}: not listed in the token manifest")
        p = out_dir / name
        if not p.exists():
            raise TokenFileMismatch(f"{name}: missing from {out_dir}")
        sha = sha256_file(p)
        if sha != files[name]:
            raise TokenFileMismatch(
                f"{name}: sha256 {sha} differs from the manifest's {files[name]}"
            )
        got[name] = sha
    return got


def load_output(out_dir) -> dict:
    """The token run's outputs (call ``verify_token_files`` first)."""
    from safetensors import safe_open

    out_dir = Path(out_dir)
    with safe_open(str(out_dir / "tokens.safetensors"), framework="np") as f:
        meta = f.metadata() or {}
        X, P, E = f.get_tensor("X"), f.get_tensor("presence"), f.get_tensor("row_eligible")
    rows_list = [
        json.loads(x) for x in (out_dir / "rows.jsonl").read_text().splitlines() if x.strip()
    ]
    detail = {}
    for line in (out_dir / "detail.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            detail[d["id"]] = {"tokens": d["tokens"], "per_tensor": d["per_tensor"]}
    return {
        "X": X,
        "presence": P,
        "row_eligible": E,
        "row_ids": json.loads(meta.get("row_ids", "[]")),
        "columns": json.loads(meta.get("columns", "[]")),
        "rows_list": rows_list,
        "rows": {r["id"]: r for r in rows_list},
        "detail": detail,
        "kframes": json.loads((out_dir / "kframes.json").read_text()),
    }


# -------------------------------------------------------------------------------------- folds and k bins


def k_bins(population) -> dict:
    """Row id -> the smallest registered k whose S_k contains it (0 for existing rows)."""
    rank = {p["component_id"]: p["rank"] for p in population["purchase_order"]}
    out = {}
    for r in population["rows"]:
        if r["source"] != "purchasable":
            out[r["id"]] = 0
        else:
            out[r["id"]] = min(k for k in fs.K_GRID if k > rank[r["component_id"]])
    return out


def eligible_row_folds(population, eligible_ids) -> list:
    """F12a.1(i), F12a.2(e): for every registered k, S_k's eligible rows (excluded rows not replaced, no top-up),
    outer folds recomputed on them by the F12.2 rule, the four inner folds of every outer fold's training
    components, and the leave-one-family-out evaluation sets."""
    order = [
        p["component_id"] for p in sorted(population["purchase_order"], key=lambda p: p["rank"])
    ]
    existing = [r for r in population["rows"] if r["source"] != "purchasable"]
    purch = [r for r in population["rows"] if r["source"] == "purchasable"]
    out = []
    for k in fs.K_GRID:
        chosen = set(order[:k])
        rows = [
            r
            for r in existing + [r for r in purch if r["component_id"] in chosen]
            if r["id"] in eligible_ids
        ]
        st = fpop.strata(rows)
        outer, _ = fpop.assign_folds(st)
        outer = dict(sorted(outer.items()))
        inner = {}
        for o in range(fs.N_OUTER):
            train = sorted(c for c, g in outer.items() if g != o)
            inner[str(o)] = dict(sorted(fs.inner_folds(train).items()))
        lofo = {}
        for fam in LOFO_FAMILIES:
            ev = sorted(c for c, s in st.items() if s == fam)
            lofo[fam] = {
                "eval_components": ev,
                "eval_rows": sum(1 for r in rows if r["component_id"] in set(ev)),
                "train": LOFO_TRAIN_RULE,
            }
        out.append(
            {
                "k": k,
                "rows": len(rows),
                "components": len(st),
                "row_ids_sha256": fpop.ids_sha256([r["id"] for r in rows]),
                "outer": outer,
                "outer_sha256": fpop.fold_manifest_sha256(outer),
                "inner": inner,
                "inner_sha256": _sha_json(inner),
                "lofo": lofo,
            }
        )
    return out


def folds_sha256(folds) -> str:
    return _sha_json(folds)


def omitted_energy_summary(rows_by_id, population, threshold=OMITTED_THRESHOLD) -> dict:
    """Descriptive: rows whose largest per-tensor omitted energy exceeds ``threshold``, by family and k bin."""
    bins = k_bins(population)
    fam = {r["id"]: r["family"] for r in population["rows"]}

    def block(ids):
        vals = [rows_by_id[i].get("omitted_energy_max") or 0.0 for i in ids]
        q = np.quantile(vals, [0, 0.25, 0.5, 0.75, 1.0]).tolist() if vals else []
        return {
            "rows": len(ids),
            "gt_threshold": sum(v > threshold for v in vals),
            "omitted_energy_max_quantiles": dict(zip(("min", "q25", "median", "q75", "max"), q)),
        }

    ids = sorted(rows_by_id)
    by_family = {f: block([i for i in ids if fam[i] == f]) for f in sorted(set(fam.values()))}
    by_bin = {str(k): block([i for i in ids if bins[i] == k]) for k in fs.K_GRID}
    cross = {
        f: {
            str(k): sum(
                (rows_by_id[i].get("omitted_energy_max") or 0) > threshold
                for i in ids
                if fam[i] == f and bins[i] == k
            )
            for k in fs.K_GRID
        }
        for f in sorted(set(fam.values()))
    }
    rank_src = Counter(
        rows_by_id[i].get("rank_source")
        for i in ids
        if (rows_by_id[i].get("omitted_energy_max") or 0) > threshold
    )
    return {
        "statistic": "per row, omitted_energy_max = the largest over its present tokens of sum_{j>r} sigma_j^2 / "
        "sum_j sigma_j^2 at the recorded rank r (f12_tokens.balanced_factors)",
        "threshold": threshold,
        "rows_gt_threshold": sum(v["gt_threshold"] for v in by_family.values()),
        "n_rows": len(ids),
        "by_family": by_family,
        "by_k_bin": by_bin,
        "gt_threshold_by_family_and_k_bin": cross,
        "gt_threshold_by_rank_source": dict(sorted(rank_src.items())),
        "k_bin_rule": "0 = existing rows; k = the smallest registered k > the component's purchase rank",
        "registered_status": "F5.1 and the F5 schema pins (PREREG_H1_TF): omitted energy is recorded as a "
        "diagnostic, not an input; the K block uses the rank-r truncation BA. F12, F12a, F12b, F12c and F12d set "
        "no rule on it. Reported descriptively; no rule is changed; flagged for Board A.",
    }


# ------------------------------------------------------------------------------------------------ checks


def _check(name, problems, detail=None):
    return {
        "check": name,
        "pass": not problems,
        "problems": problems[:50],
        "n_problems": len(problems),
        "detail": detail or {},
    }


def check_population_identity(manifest, population, population_sha, tok):
    P = []
    pf = manifest.get("population_frozen") or {}
    if manifest.get("status") != "complete":
        P.append(f"manifest status {manifest.get('status')!r} is not 'complete'")
    if pf.get("population_json_sha256") != population_sha:
        P.append("manifest population_json_sha256 differs from the committed population.json")
    if pf.get("population_ids_sha256") != population["population_ids_sha256"]:
        P.append("manifest population_ids_sha256 differs from the population's")
    if pf.get("fold_manifest_sha256") != population["folds"]["fold_manifest_sha256"]:
        P.append("manifest fold_manifest_sha256 differs from the population's")
    if (population.get("reproduction") or {}).get("status") != "reproduced":
        P.append("population.json does not record a reproduced population")
    pop_ids = sorted(r["id"] for r in population["rows"])
    if fpop.ids_sha256(pop_ids) != population["population_ids_sha256"]:
        P.append("population ids do not hash to population_ids_sha256")
    if tok["row_ids"] != pop_ids:
        P.append("token row ids are not the population ids in sorted order")
    if hashlib.sha256(json.dumps(tok["row_ids"]).encode()).hexdigest() != (
        manifest.get("tokens") or {}
    ).get("row_ids_sha256"):
        P.append("token row ids do not hash to the manifest's row_ids_sha256")
    by_id = {r["id"]: r for r in population["rows"]}
    if sorted(tok["rows"]) != pop_ids or len(tok["rows_list"]) != len(pop_ids):
        P.append("rows.jsonl does not hold exactly one record per population row")
    if sorted(tok["detail"]) != pop_ids:
        P.append("detail.jsonl does not hold exactly one record per population row")
    for rid, rec in sorted(tok["rows"].items()):
        r = by_id.get(rid)
        if r is None:
            continue
        for key in ("family", "source", "bytes_holder"):
            if rec.get(key) != r[key]:
                P.append(
                    f"{rid}: rows.jsonl {key} {rec.get(key)!r} differs from the population's {r[key]!r}"
                )
    s = manifest.get("summary") or {}
    if not (s.get("n_rows_done") == s.get("n_rows_population") == len(pop_ids)):
        P.append("manifest row counts differ from the population")
    recs = sorted(tok["rows_list"], key=lambda r: r["id"])
    if hashlib.sha256(ft.canonical_json(recs)).hexdigest() != manifest.get("rows_sha256"):
        P.append("rows.jsonl records do not hash to the manifest's rows_sha256")
    return _check(
        "population_identity", P, {"rows": len(pop_ids), "population_json_sha256": population_sha}
    )


def check_counts(population):
    rows = population["rows"]
    st = fpop.strata(rows)
    got = {"rows": len(rows), "components": len(st)}
    strata = dict(sorted(Counter(st.values()).items()))
    sources = dict(sorted(Counter(r["source"] for r in rows).items()))
    P = []
    if got != EXPECTED_COUNTS:
        P.append(f"population has {got}, registered {EXPECTED_COUNTS}")
    if strata != EXPECTED_STRATA:
        P.append(f"component strata {strata}, registered {EXPECTED_STRATA}")
    if sources != EXPECTED_SOURCES:
        P.append(f"row sources {sources}, registered {EXPECTED_SOURCES}")
    fams = dict(sorted(Counter(r["family"] for r in rows).items()))
    return _check(
        "counts",
        P,
        got | {"component_strata": strata, "row_sources": sources, "row_families": fams},
    )


def check_exclusions(manifest, tok):
    P = []
    excluded = {}
    for i, rid in enumerate(tok["row_ids"]):
        rec = tok["rows"].get(rid, {})
        flag = int(tok["row_eligible"][i])
        elig = rec.get("eligibility") == "eligible"
        if flag != int(elig):
            P.append(
                f"{rid}: row_eligible flag {flag} but rows.jsonl eligibility {rec.get('eligibility')!r}"
            )
        if not elig:
            excluded[rid] = rec.get("eligibility")
        elif rec.get("rank") is None:
            P.append(f"{rid}: eligible without a recorded rank")
    s = manifest.get("summary") or {}
    if (s.get("excluded") or {}) != excluded:
        P.append("manifest summary 'excluded' differs from rows.jsonl")
    n_elig = sum(1 for r in tok["rows"].values() if r.get("eligibility") == "eligible")
    if s.get("eligible") != n_elig:
        P.append(f"manifest summary eligible {s.get('eligible')} differs from rows.jsonl {n_elig}")
    unsupported = sorted(
        r["id"] for r in tok["rows"].values() if r.get("unsupported_changed_tensors")
    )
    detail = {
        "rule": "F12a.2(e) / F5 pins: a row with any supported changed tensor lacking r, a finite SVD or a K frame "
        "is excluded from every arm and listed; changed tensors outside the 186 slots get no token and are listed",
        "eligible": n_elig,
        "excluded": excluded,
        "rows_with_unsupported_changed_tensors": unsupported,
        "rank_sources": dict(
            sorted(Counter(r.get("rank_source") for r in tok["rows"].values()).items())
        ),
        "field_status": dict(
            sorted(Counter(r.get("field_status") for r in tok["rows"].values()).items())
        ),
        "q_partial_rows": sum(bool(r.get("q_partial")) for r in tok["rows"].values()),
    }
    return _check("exclusions", P, detail)


def check_finite_schema(manifest, tok):
    import torch

    from wsp.features import response_field as rf

    P = []
    X, Pr, E = tok["X"], tok["presence"], tok["row_eligible"]
    n = len(tok["row_ids"])
    if list(tok["columns"]) != list(ft.COLUMNS) or list(manifest.get("columns") or []) != list(
        ft.COLUMNS
    ):
        P.append("token columns differ from the f12-tokens-v1 schema")
    if len(ft.COLUMNS) != fs.N_COLS:
        P.append(f"token schema has {len(ft.COLUMNS)} columns, the simulation {fs.N_COLS}")
    names = [s[2] for s in fs.SLOTS]
    sl = manifest.get("slots") or {}
    if hashlib.sha256(json.dumps(names).encode()).hexdigest() != sl.get("names_sha256") or sl.get(
        "n"
    ) != len(names):
        P.append("the simulation's slot table differs from the token run's slots")
    lt = sl.get("layer_types") or []
    if len(lt) != fs.N_LAYERS or any(
        full != (lt[layer] == "full_attention") for layer, _, _, full in fs.SLOTS
    ):
        P.append("slot full-attention flags differ from the token run's layer types")
    if X.dtype != np.float32 or X.shape != (n, fs.N_SLOTS, fs.N_COLS):
        P.append(f"X is {X.dtype} {X.shape}, expected float32 {(n, fs.N_SLOTS, fs.N_COLS)}")
        return _check("finite_schema", P)
    if Pr.dtype != np.uint8 or Pr.shape != (n, fs.N_SLOTS) or not np.isin(Pr, (0, 1)).all():
        P.append("presence is not a 0/1 uint8 [rows, 186] array")
    if E.shape != (n,) or not np.isin(E, (0, 1)).all():
        P.append("row_eligible is not a 0/1 [rows] array")
    if not np.isfinite(X).all():
        bad = sorted({tok["row_ids"][i] for i in np.argwhere(~np.isfinite(X))[:, 0]})
        P.append(f"non-finite token values in rows {bad[:10]}")
    present = Pr.astype(bool)
    if np.any(X[~present] != 0):
        P.append(f"{int(np.any(X[~present] != 0, axis=1).sum())} absent slots carry nonzero values")
    role = np.array([fs.ROLES[s[1]] for s in fs.SLOTS])
    layer = (np.array([s[0] for s in fs.SLOTS], dtype=np.float64) / fs.N_LAYERS).astype(np.float32)
    full = np.array([float(s[3]) for s in fs.SLOTS], dtype=np.float32)
    down = role == "mlp.down_proj"
    for i, rid in enumerate(tok["row_ids"]):
        rec, det = tok["rows"].get(rid, {}), tok["detail"].get(rid, {"tokens": {}})
        slots = sorted(int(s) for s in det["tokens"])
        if slots != np.flatnonzero(Pr[i]).tolist():
            P.append(f"{rid}: presence differs from detail.jsonl token slots")
        if rec.get("n_tokens") != int(Pr[i].sum()):
            P.append(
                f"{rid}: presence count {int(Pr[i].sum())} differs from n_tokens {rec.get('n_tokens')}"
            )
        for s in slots:
            if not np.array_equal(np.asarray(det["tokens"][str(s)], dtype=np.float32), X[i, s]):
                P.append(f"{rid}: slot {s} differs between X and detail.jsonl")
                break
    T = X[present]
    S = np.argwhere(present)[:, 1]
    if not np.isin(T[:, 24:32], (0.0, 1.0)).all():
        P.append("rank-presence columns 24-31 are not 0/1")
    if np.any(T[:, 33] != layer[S]):
        P.append("column 33 differs from float32(layer / 24)")
    if np.any(T[:, 34] != full[S]):
        P.append("column 34 differs from the slot's full-attention flag")
    if np.any(T[:, 37] != 0):
        P.append("column 37 (cross term) is nonzero for a root")
    if not np.isin(T[:, 38], (0.0, 1.0)).all() or np.any(T[T[:, 38] == 0][:, 35:38] != 0):
        P.append("field presence column 38 is not 0/1 or an absent field carries values")
    if not np.isin(T[:, 41], (0.0, 1.0)).all() or np.any(T[T[:, 41] == 0][:, 39:41] != 0):
        P.append("probe presence column 41 is not 0/1 or an absent probe carries values")
    if np.any(T[~down[S]][:, 39:42] != 0):
        P.append("probe columns 39-41 nonzero off mlp.down_proj")
    tk = manifest.get("tokens") or {}
    if rf._tensor_hash(torch.from_numpy(X)) != tk.get("X_sha256"):
        P.append("X does not hash to the manifest's X_sha256")
    if rf._tensor_hash(torch.from_numpy(Pr)) != tk.get("presence_sha256"):
        P.append("presence does not hash to the manifest's presence_sha256")
    if int(Pr.sum()) != tk.get("n_present_tokens"):
        P.append("present-token count differs from the manifest")
    a = np.array([tok["rows"][rid].get("a") or 0.0 for rid in tok["row_ids"]])
    sum35 = X[:, :, 35].astype(np.float64).sum(axis=1)
    detail = {
        "shape": list(X.shape),
        "present_tokens": int(Pr.sum()),
        "down_proj_tokens": int(down[S].sum()),
        "down_proj_tokens_with_probe": int((T[down[S]][:, 41] == 1).sum()),
        "field_present_tokens": int((T[:, 38] == 1).sum()),
        "tokens_per_row_quantiles": np.quantile(Pr.sum(axis=1), [0, 0.5, 1]).tolist(),
        "descriptive_max_abs_sum_col35_minus_a": float(np.max(np.abs(sum35 - a)))
        if len(a)
        else None,
    }
    return _check("finite_schema", P, detail)


def _minimal_matrix(population, tok):
    by_id = {r["id"]: r for r in population["rows"]}
    ids = tok["row_ids"]
    return {
        "row_ids": np.asarray(ids),
        "component": np.asarray([by_id[i]["component_id"] for i in ids]),
        "family": np.asarray([by_id[i]["family"] for i in ids]),
        "origin": np.asarray(
            ["purchasable" if by_id[i]["source"] == "purchasable" else "existing" for i in ids]
        ),
        "eligible": np.asarray(tok["row_eligible"]).astype(bool),
    }


def check_folds(population, tok, folds):
    P = []
    m = _minimal_matrix(population, tok)
    if fs.purchase_order(m) != [
        p["component_id"] for p in sorted(population["purchase_order"], key=lambda p: p["rank"])
    ]:
        P.append("the simulation's purchase order differs from population.json")
    all_eligible = bool(m["eligible"].all())
    st145 = fpop.strata(population["rows"])
    for r in population["rows"]:
        if r["stratum"] != st145[r["component_id"]]:
            P.append(
                f"{r['id']}: stratum {r['stratum']} differs from the recomputed {st145[r['component_id']]}"
            )
    k145 = population["s_k"][-1]["folds"]
    for r in population["rows"]:
        if r["fold"] != k145.get(r["component_id"]):
            P.append(f"{r['id']}: row fold {r['fold']} differs from S_145's component fold")
    for f, s in zip(folds, population["s_k"]):
        k = f["k"]
        if s["k"] != k:
            P.append(f"population s_k order differs at k = {k}")
            continue
        idx = fs.s_k_rows(m, k)
        rows = [str(x) for x in m["row_ids"][idx]]
        if fpop.ids_sha256(rows) != f["row_ids_sha256"]:
            P.append(f"k = {k}: the simulation's S_k rows differ from the eligible-row folds")
        sim_outer = fs.outer_folds(m["component"][idx].tolist(), m["family"][idx].tolist())
        if dict(sorted(sim_outer.items())) != f["outer"]:
            P.append(
                f"k = {k}: the simulation's outer folds differ from the F12.2 rule (f12_population)"
            )
        if all_eligible:
            if f["outer"] != s["folds"] or f["outer_sha256"] != s["fold_manifest_sha256"]:
                P.append(f"k = {k}: outer folds differ from population.json's committed S_k folds")
            if f["row_ids_sha256"] != s["row_ids_sha256"] or f["rows"] != s["rows"]:
                P.append(f"k = {k}: S_k rows differ from population.json's committed S_k")
    detail = {
        "per_k": [
            {
                "k": f["k"],
                "rows": f["rows"],
                "components": f["components"],
                "outer_sha256": f["outer_sha256"],
                "fold_sizes": dict(sorted(Counter(f["outer"].values()).items())),
            }
            for f in folds
        ],
        "eligible_rows_equal_population": all_eligible,
    }
    return _check("folds", P, detail)


def check_nesting(population, folds):
    P = []
    mixed = {m["component_id"] for m in population.get("mixed_components", [])}
    comp_of = {r["id"]: r["component_id"] for r in population["rows"]}
    for g in population.get("shared_delta_groups", []):
        if len({comp_of[i] for i in g["ids"]}) != 1:
            P.append(f"shared-delta group {g['ids']} spans components")
    for f in folds:
        k, outer = f["k"], f["outer"]
        if sorted(set(outer.values())) != list(range(fs.N_OUTER)):
            P.append(f"k = {k}: not every outer fold is populated")
        for o in range(fs.N_OUTER):
            inner = f["inner"][str(o)]
            train = {c for c, g in outer.items() if g != o}
            held = {c for c, g in outer.items() if g == o}
            if set(inner) != train:
                P.append(
                    f"k = {k}, outer {o}: inner folds do not partition the outer-training components"
                )
            if set(inner) & held:
                P.append(f"k = {k}, outer {o}: a held-out component is in an inner fold")
            if sorted(set(inner.values())) != list(range(fs.N_INNER)):
                P.append(f"k = {k}, outer {o}: an inner fold is empty")
            if inner != dict(sorted(fs.inner_folds(sorted(train)).items())):
                P.append(f"k = {k}, outer {o}: inner folds differ from the F5 rule")
        for fam, split in f["lofo"].items():
            ev = set(split["eval_components"])
            if ev & mixed:
                P.append(f"k = {k}, {fam}: the mixed component is evaluated")
            if not ev <= set(outer):
                P.append(f"k = {k}, {fam}: an evaluation component is outside S_k")
            if not ev:
                P.append(f"k = {k}, {fam}: empty held-family evaluation set")
            if (mixed & set(outer)) and (mixed & ev):
                P.append(f"k = {k}, {fam}: the mixed component does not train")
    detail = {
        "rule": "F12a.2(f): F3, standardisation, bandwidth and epoch / lambda / tree-count selection are fitted "
        "inside each inner-training split (outer training minus inner fold j), then refitted on outer training; "
        "checked here at the fold level (inner folds partition outer training, disjoint from held-out); the "
        "model-level nesting is code (fit_wmr_fold, fit_statnn_fold; F12c.1, accepted by Board A) and is not "
        "re-tested here",
        "lofo_train": LOFO_TRAIN_RULE,
        "shared_delta_groups": len(population.get("shared_delta_groups", [])),
    }
    return _check("nesting", P, detail)


def gauge_inputs(gman, gsha, token_manifest_sha256):
    """The gauge-token manifest and its verified outputs, or {"error": ...} (F12f)."""
    from scripts import f12_matrix as fm

    try:
        data = fm.load_gauge(gman, token_manifest_sha256)
    except (OSError, ValueError, KeyError) as exc:
        return {"manifest": gman, "sha256": gsha, "error": str(exc)}
    return {"manifest": gman, "sha256": gsha, "data": data}


def check_gauge(manifest, tok, gauge=None):
    frames = (tok["kframes"] or {}).get("tensors") or {}
    missing_frame, missing_rank, cutoff = [], [], 0
    for rid, det in sorted(tok["detail"].items()):
        if det["per_tensor"] and tok["rows"].get(rid, {}).get("rank") is None:
            missing_rank.append(rid)
        for name, info in det["per_tensor"].items():
            fr = frames.get(name)
            if not fr or "G" not in fr or "A" not in fr:
                missing_frame.append(f"{rid}:{name}")
            if "k_cutoff_tie" in (info.get("flags") or []):
                cutoff += 1
    prereq = not missing_frame and not missing_rank
    inputs = any("gauge" in n for n in (manifest.get("files") or {})) or "gauge" in manifest
    runner = callable(getattr(fs, "gauge_audit", None)) and "tokens_gauge" in getattr(fs, "OPTIONAL_MATRIX_KEYS", ())
    P = []
    if missing_frame:
        P.append(f"{len(missing_frame)} present tokens without a K frame, e.g. {missing_frame[:3]}")
    if missing_rank:
        P.append(f"{len(missing_rank)} rows with tokens but no recorded rank")
    audit = None
    if gauge is None and not inputs:
        P.append(
            "the frozen token output holds no gauge-transformed tokens and no (B, A) factors, so the F5 gauge "
            "audit cannot be evaluated on these inputs"
        )
    elif gauge is not None:
        audit = _gauge_audit_inputs(gauge, tok, P)
    if not runner:
        P.append(
            "scripts/f12_simulation.py implements no gauge audit for real tokens (its rules text: 'not run on "
            "synthetic tokens (no factors); the real token builder supplies gauge-transformed tokens')"
        )
    detail = {
        "requirement": "F12a.2(i): the F5 gauge audit applies to WM-R; F12a.5(d): a replicate failing a gauge audit "
        "counts as a failure. F5 pins: for every held-out row and supported tensor with factors, (B, A) -> (BQ, "
        "Q^-1 A) with Q = H diag(d) from seed('20261110:lol-gauge:', '<canonical edit id>:<exact tensor name>'), "
        "fp64 then fp32; every fitted model evaluated on both inputs; |delta f| <= 1e-5 + 1e-4 |f| nats per row",
        "prerequisites_pass": prereq,
        "k_cutoff_tie_tokens": cutoff,
        "kframe_flag_counts": (manifest.get("kframes") or {}).get("flag_counts"),
        "audit_inputs_in_frozen_output": inputs,
        "audit_inputs": audit,
        "runner_implements_real_token_audit": runner,
        "runner_audit": "scripts/f12_simulation.py gauge_audit / gauge_record: every refitted WM-R head (each seed, "
        "each outer fold) on the held-out rows' frozen and gauge-transformed tokens; violations make C_WM and "
        "C_WM,S undefined and the replicate unresolved (validity)" if runner else None,
        "analytic_note": "columns 16-41 are functions of D_t alone; columns 0-15 depend on (B, A) only through BA, "
        "M = (Lambda_G^1/2 U_G^T)(BA)(U_A Lambda_A^1/2), so a gauge change alters tokens only by fp32 rounding. "
        "This is context (chair ruling on F12e (a)); the registered numerical audit is what the runner applies.",
    }
    return _check("gauge", P, detail)


def _gauge_audit_inputs(gauge, tok, P):
    """F12f: the gauge-transformed tokens are complete, verified and consistent with the frozen tokens."""
    gman = gauge.get("manifest") or {}
    rec = {"gauge_manifest_sha256": gauge.get("sha256"), "status": gman.get("status")}
    if "error" in gauge:
        P.append(f"gauge-token output not verified: {gauge['error']}")
        return rec
    if gman.get("status") != "complete":
        P.append("the gauge-token manifest is not complete")
    data = gauge["data"]
    X, Pr = np.asarray(tok["X"]), np.asarray(tok["presence"]).astype(bool)
    Xg, Pg = np.asarray(data["X_gauge"]), np.asarray(data["presence"]).astype(bool)
    summ = gman.get("summary") or {}
    n_rows, n_tok = len(tok["row_ids"]), int(Pr.sum())
    if list(data["row_ids"]) != list(tok["row_ids"]) or Xg.shape != X.shape or not np.array_equal(Pg, Pr):
        P.append("gauge-token rows or presence differ from the frozen tokens")
    else:
        if not np.array_equal(Xg[..., 16:], X[..., 16:]):
            P.append("gauge tokens differ from the frozen tokens in columns 16-41")
        if Xg[~Pr].any() or not np.isfinite(Xg).all():
            P.append("gauge tokens are non-finite or nonzero in absent slots")
    if summ.get("n_rows_done") != n_rows or summ.get("n_tokens") != n_tok:
        P.append(f"gauge-token manifest token count {summ.get('n_tokens')} / rows {summ.get('n_rows_done')} differ "
                 f"from the frozen {n_tok} / {n_rows}")  # fmt: skip
    if summ.get("k_reproduced_exact_rows") != n_rows:
        P.append("the gauge builder did not reproduce every frozen K block bitwise "
                 f"({summ.get('k_reproduced_exact_rows')} of {n_rows} rows)")  # fmt: skip
    changed = int(np.any(Xg[..., :16][Pr] != X[..., :16][Pr], axis=1).sum()) if Xg.shape == X.shape else None
    rec |= {"rows": n_rows, "tokens": n_tok, "tokens_with_changed_k_block": changed,
            "k_reproduced_exact_rows": summ.get("k_reproduced_exact_rows"),
            "tokens_reproduced_bitwise": summ.get("tokens_reproduced_bitwise"),
            "k_block_max_abs_change": summ.get("k_block_max_abs_change"), "cond_max": summ.get("cond_max"),
            "masked_tokens": summ.get("masked_tokens"), "files": data.get("files")}  # fmt: skip
    return rec


def matrix_inputs(mdoc, msha):
    """The committed matrix manifest and its matrix, loaded (F12f), or {"error": ...}."""
    try:
        m = fs.load_matrix(mdoc["matrix"]["path"])
        fsha = sha256_file(mdoc["matrix"]["path"])
    except (OSError, ValueError, KeyError) as exc:
        return {"manifest": mdoc, "sha256": msha, "error": str(exc)}
    return {"manifest": mdoc, "sha256": msha, "matrix": m, "file_sha256": fsha}


def check_matrix_inputs(manifest, tok=None, gauge=None, matrix=None):
    from wsp.features import pool

    stats, spec = pool.pooled_dims(4)
    has_v4 = any("v4" in n for n in (manifest.get("files") or {})) or "v4" in manifest or matrix is not None
    try:
        from scripts import f12_matrix as fm

        has_builder = callable(getattr(fm, "matrix_from_tokens", None))
    except ImportError:
        fm, has_builder = None, False
    P = []
    if not has_v4:
        P.append(
            "the frozen token output holds no v4 pooled statistics; the registered StatNN comparator takes v4 "
            "pooled statistics plus the 186 x (42 + presence) token block (F12.3, F5 pins, A5.4.1)"
        )
    if not has_builder:
        P.append(
            "no builder from the f12-tokens-v1 output to the simulation's f12-token-matrix-v1 matrix exists"
        )
    detail = {
        "simulation_v4_columns_synthetic": fs.V4_COLS,
        "pinned_pooling_code_v4_width": {
            "stats": stats,
            "spectrum": spec,
            "total": stats + spec,
            "source": "src/wsp/features/pool.py pooled_dims(buckets=4)",
        },
    }
    if matrix is None:
        if has_builder:
            P.append("no committed real token matrix was supplied to the qualification")
    elif "error" in matrix:
        P.append(f"real token matrix not loadable: {matrix['error']}")
    else:
        detail |= _matrix_detail(matrix, tok, gauge, stats + spec, P)
    return _check("simulation_matrix_inputs", P, detail)


def _matrix_detail(matrix, tok, gauge, v4_width, P):
    from scripts import f12_matrix as fm

    mdoc, m = matrix["manifest"], matrix["matrix"]
    msha = fs.matrix_sha256(m)
    if mdoc.get("matrix_sha256") != msha or mdoc.get("file_sha256") != matrix["file_sha256"]:
        P.append("the real token matrix differs from its matrix manifest")
    if m["synthetic"] or m["v4"].shape[1] != v4_width:
        P.append(f"the matrix's v4 block is not the pinned pooling code's {v4_width} columns")
    widths = fm.widths()
    F_width = m["v4"].shape[1] + fs.N_SLOTS * (fs.N_COLS + 1)
    if F_width != widths["statnn_total"] or mdoc.get("widths") != widths:
        P.append("the comparator width differs from the matrix manifest")
    if tok is not None:
        X, Pr = np.asarray(tok["X"]), np.asarray(tok["presence"]).astype(bool)
        rr, ss = np.nonzero(Pr)
        same = (list(m["row_ids"]) == list(tok["row_ids"]) and m["token_row"].tolist() == rr.tolist()
                and m["token_slot"].tolist() == ss.tolist()
                and np.array_equal(m["tokens"], X[rr, ss].astype(np.float64)))  # fmt: skip
        if not same:
            P.append("matrix tokens differ from the frozen tokens")
        a = [tok["rows"][i]["a"] for i in tok["row_ids"]]
        q = [tok["rows"][i]["q"] for i in tok["row_ids"]]
        if list(m["sum_a"]) != a or list(m["sum_q"]) != q:
            P.append("matrix Sum a / Sum q differ from the frozen rows")
        if not np.array_equal(m["eligible"], np.asarray(tok["row_eligible"]).astype(bool)):
            P.append("matrix eligibility differs from the frozen rows")
        if gauge is not None and "data" in gauge and same:
            if not np.array_equal(m["tokens_gauge"], np.asarray(gauge["data"]["X_gauge"])[rr, ss].astype(np.float64)):
                P.append("matrix gauge tokens differ from the gauge-token output")
    if not np.array_equal(m["x0_v4"], fm.x0_v4()):
        P.append("the matrix's zero-edit x0 differs from the pinned pooling code on the zero delta")
    return {
        "matrix_manifest_sha256": matrix["sha256"],
        "matrix_sha256": msha,
        "matrix_file_sha256": matrix["file_sha256"],
        "widths": widths,
        "rows": len(m["row_ids"]),
        "tokens": len(m["tokens"]),
        "x0_all_zero": not m["x0_v4"].any(),
        "v4_sources": (mdoc.get("sources") or {}).get("features"),
        "builder": "scripts/f12_matrix.py matrix_from_tokens",
        "synthetic_stand_in_refused": "f12_simulation.validate_matrix refuses the 18-column stand-in on a "
        "non-synthetic matrix; governed runs refuse synthetic matrices and any matrix but the committed manifest's",
    }


def omitted_energy_unchanged(report, prior) -> bool:
    """The omitted-energy flag carried to Board A unchanged (chair ruling on F12e (c))."""
    return json.dumps(report.get("omitted_energy"), sort_keys=True) == json.dumps(
        (prior or {}).get("omitted_energy"), sort_keys=True
    )


QUALIFICATION_ORDER = (
    "token_files",
    "code_identity",
    "population_identity",
    "counts",
    "exclusions",
    "finite_schema",
    "folds",
    "nesting",
    "gauge",
    "simulation_matrix_inputs",
)


def qualify(manifest, population, population_sha, out_dir, tok=None, identities=None, gauge=None,
            matrix=None) -> dict:
    files = verify_token_files(manifest, out_dir)
    tok = load_output(out_dir) if tok is None else tok
    checks = [_check("token_files", [], {"verified": files})]
    if identities is not None:
        drift = identity_drift(identities)
        checks.append(
            _check("code_identity", [f"{x}: bytes differ from the frozen identity" for x in drift])
        )
    elig = {rid for rid, r in tok["rows"].items() if r.get("eligibility") == "eligible"}
    folds = eligible_row_folds(population, elig)
    checks += [
        check_population_identity(manifest, population, population_sha, tok),
        check_counts(population),
        check_exclusions(manifest, tok),
        check_finite_schema(manifest, tok),
        check_folds(population, tok, folds),
        check_nesting(population, folds),
        check_gauge(manifest, tok, gauge),
        check_matrix_inputs(manifest, tok, gauge, matrix),
    ]
    first = next((c["check"] for c in checks if not c["pass"]), None)
    return {
        "schema": SCHEMA_QUAL,
        "counts_expected": dict(EXPECTED_COUNTS),
        "order": list(QUALIFICATION_ORDER),
        "checks": checks,
        "pass": first is None,
        "first_failure": first,
        "omitted_energy": omitted_energy_summary(tok["rows"], population),
        "eligible_row_folds_sha256": folds_sha256(folds),
        "outcomes_read": "none: the token run's outputs (field forecasts and edit-derived inputs), the gauge-token "
        "outputs, the real token matrix (edit-derived v4 statistics), the committed manifests and population only; "
        "no evaluation record, outcome, noise payload or data/evals file",
    }


# ---------------------------------------------------------------------------------------------- C1 facts


def down_proj_names():
    return [s[2] for s in fs.SLOTS if fs.ROLES[s[1]] == "mlp.down_proj"]


def base_tensor_names(tok, population):
    names = set(down_proj_names())
    for rid in population["f12i_lora_roots"]:
        names |= set(tok["detail"][rid]["per_tensor"])
    return sorted(names)


def base_tensor_hashes(state, names, expected_state_hash) -> dict:
    from wsp.features import response_field as rf

    got = rf._hash({n: rf._tensor_hash(t) for n, t in state.items()})
    if got != expected_state_hash:
        raise ValueError(
            f"base state hash {got} differs from the token manifest's {expected_state_hash}"
        )
    return {"state_hash": got, "tensors": {n: rf._tensor_hash(state[n]) for n in sorted(names)}}


def token_facts(manifest, manifest_sha, population, population_sha, tok, base_hashes) -> dict:
    if base_hashes.get("state_hash") != (manifest.get("base") or {}).get("state_hash"):
        raise ValueError("base tensor hashes were not computed on the token manifest's base state")
    rows = tok["rows"]
    elig = sorted(rid for rid, r in rows.items() if r.get("eligibility") == "eligible")
    folds = eligible_row_folds(population, set(elig))
    fam = {r["id"]: r["family"] for r in population["rows"]}
    tk = manifest["tokens"]
    incr = {rid: rows[rid]["increment_sha256"] for rid in sorted(rows)}
    bt = base_hashes["tensors"]
    lora = {}
    for rid in sorted(population["f12i_lora_roots"]):
        names = sorted(tok["detail"][rid]["per_tensor"])
        p1 = [n for n in names if layer_of(n) is not None and layer_of(n) <= 11]
        p2 = [n for n in names if layer_of(n) is not None and layer_of(n) >= 12]
        lora[rid] = {
            "family": fam[rid],
            "eligibility": rows[rid]["eligibility"],
            "increment_sha256": rows[rid]["increment_sha256"],
            "n_edited_tensors": rows[rid]["n_edited_tensors"],
            "n_nonzero_tensors": len(names),
            "part1": p1,
            "part2": p2,
            "undefined_split_tensors": [n for n in names if layer_of(n) is None],
            "support_base_tensor_hashes": {n: bt[n] for n in names},
        }
    return {
        "schema": SCHEMA_FACTS,
        "status": "outcome-free facts of the frozen token run for the C1 draft (F12e); not a C1 file",
        "token_manifest": {
            "path": _rel(TOKEN_MANIFEST) if TOKEN_MANIFEST.exists() else None,
            "sha256": manifest_sha,
        },
        "population": {"path": _rel(POPULATION), "sha256": population_sha},
        "files_verified": dict(sorted(manifest["files"].items())),
        "eligibility": {
            "rule": "F12a.2(e): a row with any supported changed tensor lacking r, a finite SVD or a K frame is "
            "excluded from every arm before any fit; excluded rows are not replaced and S_k is never topped up",
            "eligible_ids": elig,
            "eligible_ids_sha256": fpop.ids_sha256(elig),
            "n_eligible": len(elig),
            "excluded": {
                rid: r["eligibility"]
                for rid, r in sorted(rows.items())
                if r["eligibility"] != "eligible"
            },
            "by_family": dict(sorted(Counter(fam[i] for i in elig).items())),
            "rank_sources": dict(
                sorted(Counter(r.get("rank_source") for r in rows.values()).items())
            ),
            "field_status": dict(
                sorted(Counter(r.get("field_status") for r in rows.values()).items())
            ),
            "rows_with_unsupported_changed_tensors": sorted(
                rid for rid, r in rows.items() if r.get("unsupported_changed_tensors")
            ),
        },
        "token_inputs": {
            "tokens_safetensors_sha256": tk["sha256"],
            "X_sha256": tk["X_sha256"],
            "presence_sha256": tk["presence_sha256"],
            "row_ids_sha256": tk["row_ids_sha256"],
            "shape": tk["shape"],
            "n_present_tokens": tk["n_present_tokens"],
            "rows_sha256": manifest["rows_sha256"],
            "columns_sha256": hashlib.sha256(
                json.dumps(list(manifest["columns"])).encode()
            ).hexdigest(),
            "slots_names_sha256": manifest["slots"]["names_sha256"],
            "realised_increment_sha256_by_row": incr,
            "realised_increments_sha256": _sha_json(incr),
            "kframes": {
                k: manifest["kframes"].get(k)
                for k in ("sha256", "kframes_safetensors_sha256", "n_tensors", "flag_counts")
            },
            "token_code": manifest.get("code"),
        },
        "identities": {
            "g2_manifest": manifest["inputs"].get("g2_manifest"),
            "probe_cache": manifest["inputs"].get("probe_cache"),
            "wikitext_field": manifest["inputs"].get("wikitext_field"),
            "base": manifest.get("base"),
        },
        "down_proj_base_tensor_hashes": {n: bt[n] for n in down_proj_names()},
        "base_state_hash": base_hashes["state_hash"],
        "f12i_lora_parents": lora,
        "folds_sha256": folds_sha256(folds),
        "folds_summary": [
            {
                k: f[k]
                for k in (
                    "k",
                    "rows",
                    "components",
                    "row_ids_sha256",
                    "outer_sha256",
                    "inner_sha256",
                )
            }
            for f in folds
        ],
    }


# ---------------------------------------------------------------------------------------- identity freeze


def _versions():
    import scipy
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "lightgbm": fs._lgbm_version(),
        "scipy": scipy.__version__,
    }


def candidate_identities() -> dict:
    pin = fs.RESTRICTED_PIN
    pin_obj = json.loads(pin.read_text())
    if pin_obj["restricted_copy"]["sha256"] != RESTRICTED_SHA256:
        raise ValueError("the restricted-payload pin does not name the v2 bytes")
    cfg_files = {_rel(p): sha256_file(p) for p in sorted(MANIFEST_DIR.glob("f12c_*_config.json"))}
    plan = MANIFEST_DIR / "f12c_resource_plan.json"
    return {
        "schema": SCHEMA_IDENT,
        "status": "candidate identities re-frozen (F12f) before the re-run qualification, noise check and timing pilot "
        "use them; not C1 and not a C1 verification",
        "code": {rel: sha256_file(ROOT / rel) for rel in CANDIDATE_CODE},
        "simulation_code_identity": fs.current_code_identity(),
        "configs": dict(fs.F12C_CONFIG_SHA256),
        "config_files": cfg_files,
        "restricted_payload": {
            "sha256": RESTRICTED_SHA256,
            "pin": _rel(pin),
            "pin_sha256": sha256_file(pin),
            "path": pin_obj["restricted_copy"].get("path"),
        },
        "token_manifest": {"path": _rel(TOKEN_MANIFEST), "sha256": TOKEN_MANIFEST_SHA256},
        "gauge_manifest": {"path": _rel(GAUGE_MANIFEST), "sha256": sha256_file(GAUGE_MANIFEST)}
        if GAUGE_MANIFEST.exists() else {"path": _rel(GAUGE_MANIFEST), "sha256": None},
        "matrix_manifest": {"path": _rel(MATRIX_MANIFEST), "sha256": sha256_file(MATRIX_MANIFEST)}
        if MATRIX_MANIFEST.exists() else {"path": _rel(MATRIX_MANIFEST), "sha256": None},
        "snapshot": {"path": _rel(SNAPSHOT), "sha256": sha256_file(SNAPSHOT)},
        "population": {"path": _rel(POPULATION), "sha256": sha256_file(POPULATION)},
        "resource_plan": {"path": _rel(plan), "sha256": sha256_file(plan)},
        "code_hashes_c1_draft": {
            "path": _rel(fs.CODE_HASHES_C1),
            "sha256": sha256_file(fs.CODE_HASHES_C1),
        },
        "versions": _versions(),
    }


def identity_drift(ident) -> list:
    out = [rel for rel, sha in sorted(ident["code"].items()) if sha256_file(ROOT / rel) != sha]
    out += [
        rel
        for rel, sha in sorted(ident.get("config_files", {}).items())
        if sha256_file(ROOT / rel) != sha
    ]
    for key in ("restricted_payload",):
        rp = ident[key]
        if sha256_file(ROOT / rp["pin"]) != rp["pin_sha256"]:
            out.append(rp["pin"])
    for key in ("token_manifest", "population", "resource_plan", "gauge_manifest", "matrix_manifest", "snapshot"):
        if key not in ident:
            continue
        p = ROOT / ident[key]["path"]
        got = sha256_file(p) if p.exists() else None
        if got != ident[key]["sha256"]:
            out.append(ident[key]["path"])
    return out


# ------------------------------------------------------------------------------------------------------ CLI


def _write(path: Path, obj):
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return sha256_file(path)


def _manifest():
    raw = TOKEN_MANIFEST.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != TOKEN_MANIFEST_SHA256:
        raise SystemExit(f"token manifest sha256 {sha} is not the frozen {TOKEN_MANIFEST_SHA256}")
    return json.loads(raw), sha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    b = sub.add_parser("base-hashes")
    b.add_argument("--out", type=Path, default=BASE_HASHES_PATH)
    f = sub.add_parser("facts")
    f.add_argument("--base-hashes", type=Path, default=BASE_HASHES_PATH)
    f.add_argument("--out", type=Path, default=FACTS_PATH)
    i = sub.add_parser("identities")
    i.add_argument("--out", type=Path, default=IDENTITIES_PATH)
    q = sub.add_parser("qualify")
    q.add_argument("--identities", type=Path, default=IDENTITIES_PATH)
    q.add_argument("--out", type=Path, default=QUALIFICATION_PATH)
    a = p.parse_args(argv)
    if a.cmd == "identities":
        print(json.dumps({"written": str(a.out), "sha256": _write(a.out, candidate_identities())}))
        return 0
    man, msha = _manifest()
    out_dir = Path(man["out_dir"])
    files = verify_token_files(man, out_dir)
    if a.cmd == "verify":
        print(json.dumps({"verified": files, "manifest_sha256": msha}, indent=1))
        return 0
    population = json.loads(POPULATION.read_text())
    psha = sha256_file(POPULATION)
    tok = load_output(out_dir)
    if a.cmd == "base-hashes":
        from scripts import a5_power_fields as pf

        state, _ = pf.load_base_state(man["base"]["state_hash"])
        obj = base_tensor_hashes(
            state, base_tensor_names(tok, population), man["base"]["state_hash"]
        )
        obj = {
            "schema": SCHEMA_BASE,
            "model": man["base"]["model"],
            "token_manifest_sha256": msha,
            "names_rule": "the 24 mlp.down_proj slots and every nonzero tensor of the 16 F12-I LoRA roots",
            "hash": "wsp.features.response_field._tensor_hash of the bf16 base tensor",
        } | obj
        print(
            json.dumps(
                {
                    "written": str(a.out),
                    "sha256": _write(a.out, obj),
                    "tensors": len(obj["tensors"]),
                }
            )
        )
        return 0
    if a.cmd == "facts":
        base = json.loads(a.base_hashes.read_text())
        if base.get("token_manifest_sha256") != msha:
            raise SystemExit("base tensor hashes were computed for another token manifest")
        obj = token_facts(man, msha, population, psha, tok, base)
        obj["base_tensor_hashes_file"] = {
            "path": _rel(a.base_hashes),
            "sha256": sha256_file(a.base_hashes),
        }
        print(json.dumps({"written": str(a.out), "sha256": _write(a.out, obj)}))
        return 0
    ident = json.loads(a.identities.read_text())
    rep = qualify(man, population, psha, out_dir, tok=tok, identities=ident)
    rep["token_manifest"] = {"path": _rel(TOKEN_MANIFEST), "sha256": msha}
    rep["population"] = {"path": _rel(POPULATION), "sha256": psha}
    rep["candidate_identities"] = {"path": _rel(a.identities), "sha256": sha256_file(a.identities)}
    rep["code"] = {"scripts/f12_qualify.py": sha256_file(ROOT / "scripts/f12_qualify.py")}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    rep["run_at_head"] = head
    sha = _write(a.out, rep)
    print(
        json.dumps(
            {
                "written": str(a.out),
                "sha256": sha,
                "pass": rep["pass"],
                "first_failure": rep["first_failure"],
                "checks": {c["check"]: c["pass"] for c in rep["checks"]},
            },
            indent=1,
        )
    )
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
