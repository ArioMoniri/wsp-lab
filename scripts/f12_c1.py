#!/usr/bin/env python3
"""F12 C1 package: manifest chain, verifier and draft skeleton (PREREG_H1_TF F12a.6(b)-(c); Board A F12c review
condition 6 and step 2 of its "Ordered path to stage 1").

* ``write_manifest(root)`` writes ``docs/prereg/f12_manifests/MANIFEST_C1.sha256`` holding H0-H11:
  H0 = sha256(UTF-8 ``f12-manifest-v1:`` + S), S = ``49fec4a3c885e7f611084c0e56dac77053ba2ec8``;
  H_k = sha256(H_{k-1} || sha256(file_k)) over raw 32-byte digests, files in the F12a.6(c) order ``C1_FILES``.
* ``verify_c1(root, code_identity)`` returns the failures that keep C1 from being a verified freeze (empty list only
  for a complete, committed, Board A-verified C1). It requires: the manifest committed and unmodified, exactly the
  required file set H0-H11 with a chain that recomputes; every C1 file committed, unmodified, equal to its manifest
  digest and neither a draft nor awaiting inputs; ``code_hashes_c1.json`` not a draft, every listed file at its pinned
  bytes and the run's code identity among them; and a Board A verification record
  (``c1_board_a_verification.json``) naming the manifest's sha256 and a committed Board A review under
  ``docs/reviews/`` whose bytes match, which cites that sha256 and states "C1 VERIFIED", committed in a later commit
  descending from the manifest's commit. Status strings or flags inside C1 files never authenticate themselves: code
  can check binding and chronology, not authorship, so the record is checked only by its citation of the committed
  review.
* ``build_draft(root)`` builds the C1 package skeleton (exposure, population, folds, schemas, models, payload, simulation
  configs, intervals, F12-I, resources, code hashes) as far as it can be built before the token manifest exists; every
  file is labelled a draft skeleton (not a C1 file) and every token-dependent part says ``AWAITING``. It is written to
  ``docs/prereg/f12_manifests/c1_draft/`` (never the governed C1 paths).

    python scripts/f12_c1.py build-draft     # rewrites docs/prereg/f12_manifests/c1_draft/
    python scripts/f12_c1.py verify          # prints the failures; exit 1 unless C1 is verified
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S_HEAD = "49fec4a3c885e7f611084c0e56dac77053ba2ec8"
H0 = hashlib.sha256(f"f12-manifest-v1:{S_HEAD}".encode()).hexdigest()
H0_F12I = hashlib.sha256(f"f12i-manifest-v1:{S_HEAD}".encode()).hexdigest()
C1_FILES = (
    "exposure_c1.json",
    "population.json",
    "folds.json",
    "schemas.json",
    "models.json",
    "noise_payload/restricted.json",
    "simulation_config.json",
    "intervals.json",
    "f12i_c1.json",
    "resources.json",
    "code_hashes_c1.json",
)
MANIFEST_DIR = "docs/prereg/f12_manifests"
MANIFEST_PATH = f"{MANIFEST_DIR}/MANIFEST_C1.sha256"
VERIFICATION_PATH = f"{MANIFEST_DIR}/c1_board_a_verification.json"
VERIFICATION_SCHEMA = "f12d-c1-board-a-verification-v1"
VERDICT = "C1 VERIFIED"
DRAFT_DIR = f"{MANIFEST_DIR}/c1_draft"
AWAITING = "awaiting the token manifest"
TOKEN_MANIFEST = "docs/prereg/a5_manifests/f12/tokens_manifest.json"
DRAFT_STATUS = (
    "draft skeleton, not a C1 file: built before the token manifest exists (F12d); the governed C1 file is written "
    "at the C1 freeze and verified by Board A"
)
C1_CODE_SCHEMA = "f12-code-hashes-c1-v1"
FACTS_REL = f"{MANIFEST_DIR}/f12e_token_facts.json"
QUALIFICATION_REL = f"{MANIFEST_DIR}/f12e_qualification.json"
F12F_QUALIFICATION_REL = f"{MANIFEST_DIR}/f12f_qualification.json"
GAUGE_MANIFEST_REL = "docs/prereg/a5_manifests/f12/gauge_tokens_manifest.json"
MATRIX_MANIFEST_REL = "docs/prereg/a5_manifests/f12/matrix_manifest.json"
FILLED_STATUS = (
    "draft, not a C1 file: token-dependent parts filled from the frozen token manifest and its verified outputs "
    "(F12e); the governed C1 file is written at the C1 freeze and verified by Board A"
)
F12I_AWAITING = (
    "awaiting the F12-I constructions (scripts/f12_additivity_prepare.py): the Astra and C4 parent, support and base "
    "tensor hashes and the bitwise part checks are not in the F12 token output"
)


def _sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def chain(digests) -> list:
    hs = [H0]
    for d in digests:
        hs.append(hashlib.sha256(bytes.fromhex(hs[-1]) + bytes.fromhex(d)).hexdigest())
    return hs


def write_manifest(root=ROOT) -> Path:
    root = Path(root)
    digests = [_sha_file(root / MANIFEST_DIR / n) for n in C1_FILES]
    hs = chain(digests)
    lines = [
        (
            "# F12 C1 manifest (PREREG_H1_TF F12a.6(b)): H0 = sha256('f12-manifest-v1:' + S); "
            "H_k = sha256(H_{k-1} || sha256(file_k)), raw 32-byte digests"
        ),
        f"H0 {hs[0]} anchor f12-manifest-v1:{S_HEAD}",
    ]
    lines += [f"H{i} {hs[i]} {d} {n}" for i, (d, n) in enumerate(zip(digests, C1_FILES), 1)]
    out = root / MANIFEST_PATH
    out.write_text("\n".join(lines) + "\n")
    return out


def git_env() -> dict:
    """The environment without inherited GIT_* variables. A pre-commit hook exports GIT_DIR / GIT_INDEX_FILE;
    with them, ``git -C <root>`` would address the hook's repository instead of ``root``."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(root, *args):
    r = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False, env=git_env()
    )
    return r.returncode, r.stdout.strip()


def _committed_clean(root, rel) -> bool:
    rc, _ = _git(root, "ls-files", "--error-unmatch", "--", rel)
    if rc != 0:
        return False
    rc, out = _git(root, "status", "--porcelain", "--", rel)
    return rc == 0 and out == ""


def _last_commit(root, rel):
    rc, out = _git(root, "log", "--format=%H", "-1", "--", rel)
    return out if rc == 0 and out else None


def parse_manifest(text):
    rows = {}
    for ln in text.splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = ln.split()
        if not parts[0].startswith("H") or not parts[0][1:].isdigit():
            raise ValueError(f"malformed manifest line {ln!r}")
        rows[int(parts[0][1:])] = parts[1:]
    return rows


def verify_c1(root=ROOT, code_identity=None) -> list:
    root = Path(root)
    F = []
    man = root / MANIFEST_PATH
    if not man.exists():
        return [f"C1 manifest {MANIFEST_PATH.rsplit('/', 1)[1]} missing: C1 is not frozen"]
    if not _committed_clean(root, MANIFEST_PATH):
        F.append("C1 manifest is not committed or has uncommitted changes")
    try:
        rows = parse_manifest(man.read_text())
    except ValueError as exc:
        return F + [f"C1 manifest unreadable: {exc}"]
    if sorted(rows) != list(range(len(C1_FILES) + 1)):
        F.append(f"C1 manifest incomplete: holds H{sorted(rows)}, required H0-H{len(C1_FILES)}")
    if rows.get(0, [None])[0] != H0:
        F.append("C1 manifest H0 is not the registered anchor")
    digests = []
    for i, name in enumerate(C1_FILES, 1):
        row = rows.get(i)
        rel = f"{MANIFEST_DIR}/{name}"
        if row is None or len(row) < 3 or row[2] != name:
            F.append(f"C1 manifest incomplete: H{i} does not list {name}")
            digests.append(None)
            continue
        digests.append(row[1])
        p = root / rel
        if not p.exists():
            F.append(f"C1 file {name} missing")
            continue
        if not _committed_clean(root, rel):
            F.append(f"C1 file {name} is not committed or has uncommitted changes")
        if _sha_file(p) != row[1]:
            F.append(f"C1 file {name} differs from the manifest digest")
        text = p.read_text()
        status = ""
        try:
            status = str(json.loads(text).get("status", ""))
        except (ValueError, AttributeError):
            pass
        if "draft" in status.lower() or "awaiting" in text.lower():
            F.append(f"C1 file {name} is a draft or awaiting inputs (status {status!r})")
    if None not in digests and len(digests) == len(C1_FILES):
        hs = chain(digests)
        for i in range(1, len(C1_FILES) + 1):
            if rows.get(i, [None])[0] != hs[i]:
                F.append(f"C1 manifest chain mismatch at H{i}")
                break
    ch = root / MANIFEST_DIR / "code_hashes_c1.json"
    if ch.exists():
        try:
            obj = json.loads(ch.read_text())
        except ValueError:
            obj = {}
        if obj.get("schema") != C1_CODE_SCHEMA:
            F.append("C1 code hashes: schema is not " + C1_CODE_SCHEMA)
        if "draft" in str(obj.get("status", "draft")).lower():
            F.append(f"C1 code hashes: status {obj.get('status')!r} is a draft")
        files = obj.get("files") or {}
        bad = [
            rel for rel, sha in sorted(files.items())
            if not (root / rel).exists() or _sha_file(root / rel) != sha
        ]  # fmt: skip
        if bad:
            F.append(f"C1 code hashes: current bytes differ for {bad}")
        if code_identity is not None and (
            not code_identity or any(files.get(rel) != sha for rel, sha in code_identity.items())
        ):
            F.append("C1 code hashes: the run's code identity is not the C1-pinned code")
    F += _verify_board_a(root)
    return F


def _verify_board_a(root):
    rec_path = root / VERIFICATION_PATH
    if not rec_path.exists():
        return [
            "C1: no Board A C1 verification record (fields inside C1 files cannot verify themselves)"
        ]
    F = []
    if not _committed_clean(root, VERIFICATION_PATH):
        F.append("C1: the Board A verification record is not committed")
    try:
        rec = json.loads(rec_path.read_text())
    except ValueError:
        return F + ["C1: Board A verification record unreadable"]
    man = root / MANIFEST_PATH
    msha = _sha_file(man) if man.exists() else None
    if rec.get("schema") != VERIFICATION_SCHEMA:
        F.append("C1: Board A verification record schema is not " + VERIFICATION_SCHEMA)
    if rec.get("manifest_sha256") != msha:
        F.append("C1: Board A record names a manifest sha256 that is not the committed manifest's")
    rel = str(rec.get("review") or "")
    if not rel.startswith("docs/reviews/") or not (root / rel).exists():
        return F + ["C1: Board A review named by the record is missing or not under docs/reviews/"]
    text = (root / rel).read_text()
    if _sha_file(root / rel) != rec.get("review_sha256"):
        F.append("C1: Board A review bytes differ from the record's review_sha256")
    if not _committed_clean(root, rel):
        F.append("C1: Board A review is not committed")
    if msha is None or msha not in text:
        F.append("C1: Board A review does not cite the manifest sha256")
    if VERDICT not in text:
        F.append(f"C1: Board A review lacks the verdict {VERDICT!r}")
    mc, rc = _last_commit(root, MANIFEST_PATH), _last_commit(root, rel)
    if not mc or not rc or mc == rc:
        F.append("C1 chronology: the Board A review must be committed after the manifest commit")
    else:
        code, _ = _git(root, "merge-base", "--is-ancestor", mc, rc)
        if code != 0:
            F.append("C1 chronology: the Board A review commit does not follow the manifest commit")
    return F


# ------------------------------------------------------------------------ draft skeleton


def _rel_sha(root, rel):
    return {"path": rel, "sha256": _sha_file(root / rel)}


def build_draft(root=ROOT, use_facts=True) -> dict:
    """The C1 package draft (deterministic; no timestamps). Without the committed token facts
    (``f12e_token_facts.json``) or with ``use_facts=False`` it is the F12d skeleton whose token-dependent parts say
    ``AWAITING``; with them those parts are filled (F12e)."""
    root = Path(root)
    facts = None
    if use_facts and (root / FACTS_REL).exists():
        facts = json.loads((root / FACTS_REL).read_text())
    pop = json.loads((root / "docs/prereg/a5_manifests/f12/population.json").read_text())
    pin_rel = f"{MANIFEST_DIR}/noise_payload_restricted_pin.json"
    pin = json.loads((root / pin_rel).read_text())
    plan_rel = f"{MANIFEST_DIR}/f12c_resource_plan.json"
    plan = json.loads((root / plan_rel).read_text())
    cfg_rel = {
        n: f"{MANIFEST_DIR}/f12c_{n}_config.json" for n in ("stage1", "stage2", "target", "pilot")
    }
    s2 = json.loads((root / cfg_rel["stage2"]).read_text())["config"]
    st = {"status": DRAFT_STATUS}
    files = {}
    files["exposure_c1.json"] = st | {
        "requirement": "F12a.1(c)-(d): statements (i) old development outcomes, (ii) new outcomes, (iii) prescore "
        "audits for every decision-maker, code/input builder, extractor, token-run session, chair and approving "
        "reviewer through the freeze; one session never attests for another",
        "register": "docs/prereg/f12_attestations.md",
        "at_freeze": "the register's rows are copied here with their statuses; every PENDING row must be given by "
        "its own session before C1 (Board A F12c condition 1)",
    }
    files["population.json"] = st | {
        "selector": _rel_sha(root, "scripts/f12_population.py"),
        "population": _rel_sha(root, "docs/prereg/a5_manifests/f12/population.json"),
        "population_ids_sha256": pop["population_ids_sha256"],
        "rows_sha256": pop["rows_sha256"],
        "rows": len(pop["rows"]),
        "components": len({r["component_id"] for r in pop["rows"]}),
        "eligibility": f"{AWAITING} ({TOKEN_MANIFEST}): F12a.2(e) rank / finite-SVD / K-frame audit of all 270 "
        "rows from the frozen tokens; each S_k is the eligible rows of its components, never topped up",
    }
    files["folds.json"] = st | {
        "rules": {k: pop["rules"][k] for k in ("folds", "s_k", "purchase_order", "leave_one_family_out")},
        "outer_fold_manifest_sha256": pop["folds"]["fold_manifest_sha256"],
        "s_k_fold_manifest_sha256": {str(k): s["fold_manifest_sha256"] for k, s in zip((0, 29, 58, 87, 116, 145), pop["s_k"])},
        "purchase_order_sha256": hashlib.sha256(json.dumps(pop["purchase_order"]).encode()).hexdigest(),
        "inner_folds": "outer-training components sorted by sha256('20261110:lol-inner:' + id), round-robin to four "
        "folds (F5 pins; scripts/f12_simulation.py inner_folds)",
        "eligible_row_folds": f"{AWAITING}: outer, inner and leave-one-family-out fold ids over the eligible rows "
        "of every S_k are frozen after the eligibility audit",
    }  # fmt: skip
    files["schemas.json"] = st | {
        "endpoint_config": _rel_sha(root, "configs/eval_tf_v1.yaml"),
        "token_schema": "f12-token-matrix-v1",
        "token_columns": 42,
        "token_slots": 186,
        "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "token_builder": _rel_sha(root, "scripts/f12_tokens.py"),
        "token_input_hashes": f"{AWAITING}: realised-delta, SVD, K-frame and probe input hashes",
        "g2_manifest_and_probe_cache": f"{AWAITING}: full G2 manifest sha256 and probe-cache identity as recorded",
        "down_proj_tensor_hashes": f"{AWAITING}: the 24 down_proj tensor names and base tensor hashes",
    }
    files["models.json"] = st | {
        "config": _rel_sha(root, cfg_rel["stage2"]),
        "training_and_selection": {k: s2[k] for k in sorted(s2) if k in (
            "head_zero_init", "target_standardise", "statnn_select_seeds", "lgbm", "seeds", "max_epochs",
            "patience", "tree_grid", "lr", "weight_decay", "batch", "ridge_lambda", "joint_studentiser")},
        "WM-R_head_parameters": 1185,
        "F3": "ridge-1 intercept / a / q readout on training rows only (A5.8.3)",
        "descriptive_arms": "K (Gaussian-kernel ridge) and M (width-8 GELU MLP), F12a.2(g)",
        "gauge_and_control_audits": f"{AWAITING}: F5 gauge audit and permuted-field eligibility on the real tokens",
    }  # fmt: skip
    files["noise_payload/restricted.json"] = st | {
        "pin": pin_rel,
        "pin_sha256": _sha_file(root / pin_rel),
        "restricted_sha256": pin["restricted_copy"]["sha256"],
        "source_sha256": pin["source"]["sha256"],
        "extractor": pin["extractor"],
        "at_freeze": "the pinned v2 restricted bytes are committed at this path with their provenance "
        "(F12a.6(c) item 6)",
    }
    files["simulation_config.json"] = st | {
        "configs": {n: _rel_sha(root, r) for n, r in cfg_rel.items()},
        "superseded": {
            n: _rel_sha(root, f"{MANIFEST_DIR}/f12b_{n}_config.json") for n in ("stage1", "stage2")
        },
    }
    files["intervals.json"] = st | {
        "procedures": ["conditional_bootstrap_t", "joint_bootstrap_t", "conditional_bca", "joint_bca",
                       "conditional_percentile", "joint_percentile"],
        "joint_studentiser": "max(V_comp + V_item - V_both, V_comp) for C_WM and C_WM,S only (F12b.2, F12c.2)",
        "code": [_rel_sha(root, "scripts/f12_simulation.py"), _rel_sha(root, "scripts/tfv1_requalify.py")],
        "functions": "scripts/f12_simulation.py: a542_contrast, joint_variance, joint_parts; "
        "scripts/tfv1_requalify.py: cluster_intervals, theta_interval, jackknife_cluster, linearised_se",
    }  # fmt: skip
    files["f12i_c1.json"] = st | {
        "own_chain_H0": H0_F12I,
        "construction": _rel_sha(root, "scripts/f12_additivity_prepare.py"),
        "lora_roots": len(pop["f12i_lora_roots"]),
        "lora_root_ids_sha256": hashlib.sha256(json.dumps(sorted(pop["f12i_lora_roots"])).encode()).hexdigest(),
        "parts": "LoRA 16 x 2 (layers 0-11 / 12-23), Astra 2 (by tensor), C4 2 (input columns < 1792 / >= 1792)",
        "parent_support_tensor_hashes": f"{AWAITING} and the F12-I constructions: parent, support and base tensor "
        "hashes, bitwise part checks",
        "ceiling_gpu_min": 56.16,
    }  # fmt: skip
    files["resources.json"] = st | {
        "ceiling_cpu_h": 190.0,
        "ceiling_source": "PREREG_H1_TF A5 chair ruling on F12c (11 Nov 2026, 18:55 UTC)",
        "resource_plan": _rel_sha(root, plan_rel),
        "budget_barrier": plan.get("budget_barrier"),
        "workers_max": 2,
        "omp_num_threads": 2,
        "rss_limit_gb": 4,
        "gpu_caps": {"F12_scoring_gpu_h": 3.0, "F12I_gpu_min": 56.16},
        "pilot_acceptance": "written by the chair only after the measured real-token pilot (schema "
        "f12d-pilot-acceptance-v2)",
    }
    files["code_hashes_c1.json"] = st | {
        "draft_pin": _rel_sha(root, f"{MANIFEST_DIR}/code_hashes_c1.json"),
        "at_freeze": "regenerated over the frozen code and promoted only through MANIFEST_C1.sha256 and Board A "
        "verification, never by changing its status string",
    }
    if facts is not None:
        _fill_from_facts(root, files, facts, pop)
    return {n: files[n] for n in C1_FILES}


def _fill_from_facts(root, files, facts, pop):
    """F12e: replace every AWAITING token-dependent part with the committed token facts."""
    from scripts import f12_qualify as fq

    src = _rel_sha(root, FACTS_REL)
    folds = fq.eligible_row_folds(pop, set(facts["eligibility"]["eligible_ids"]))
    if fq.folds_sha256(folds) != facts["folds_sha256"]:
        raise ValueError("eligible-row folds recomputed from population.json differ from the token facts")
    qual = root / QUALIFICATION_REL
    qref = _rel_sha(root, QUALIFICATION_REL) if qual.exists() else "f12e qualification report not yet written"
    for obj in files.values():
        obj["status"] = FILLED_STATUS
    files["population.json"]["eligibility"] = {
        "source": src,
        "token_manifest": facts["token_manifest"],
        **{k: facts["eligibility"][k] for k in ("rule", "n_eligible", "eligible_ids_sha256", "excluded",
                                                 "by_family", "rank_sources", "field_status",
                                                 "rows_with_unsupported_changed_tensors")},
    }  # fmt: skip
    files["folds.json"].pop("eligible_row_folds", None)
    files["folds.json"] |= {
        "eligible_row_folds_source": src,
        "eligible_row_folds_sha256": facts["folds_sha256"],
        "eligible_row_folds": folds,
    }
    files["schemas.json"] |= {
        "token_input_hashes": {"source": src, "token_manifest": facts["token_manifest"]} | facts["token_inputs"],
        "g2_manifest_and_probe_cache": facts["identities"],
        "down_proj_tensor_hashes": {
            "base_state_hash": facts["base_state_hash"],
            "base_model": (facts["identities"].get("base") or {}).get("model"),
            "hash": "wsp.features.response_field._tensor_hash of the bf16 base tensor (state hash re-verified)",
            "source": facts.get("base_tensor_hashes_file"),
            "tensors": facts["down_proj_base_tensor_hashes"],
        },
    }
    files["models.json"]["gauge_and_control_audits"] = {
        "F5_gauge_audit": {
            "requirement": "F12a.2(i): the F5 gauge audit applies to WM-R; F12a.5(d): a gauge-audit failure counts "
            "as a replicate failure",
            "satisfied_by_frozen_output": False,
            "qualification": qref,
            "reason": "the frozen token output holds no gauge-transformed tokens or (B, A) factors and "
            "scripts/f12_simulation.py has no real-token gauge audit; open for the chair and Board A (F12e)",
        },
        "permuted_field_eligibility": "A5.8.3 permuted-field eligibility of the descriptive WM-R-perm control "
        "applies at analysis; the token builder does not compute it",
    }
    g = files["models.json"]["gauge_and_control_audits"]
    g["F5_gauge_audit"]["f12f"] = {
        "ruling": "PREREG_H1_TF A5 chair ruling on F12e (a): no waiver; registered gauge-transformed tokens and the "
        "runner audit",
        "gauge_tokens_manifest": _rel_or_none(root, GAUGE_MANIFEST_REL),
        "builder": _rel_sha(root, "scripts/f12_gauge_tokens.py"),
        "runner": "scripts/f12_simulation.py gauge_audit / gauge_record: every refitted WM-R head, each seed and each "
        "outer fold, on the held-out rows' frozen and gauge-transformed tokens; |delta f| <= 1e-5 + 1e-4 |f| nats "
        "per row; all nonzero discrepancies retained; a violation makes C_WM and C_WM,S undefined and the replicate "
        "unresolved (validity), a failure (F12a.5(d))",
        "qualification": _rel_or_none(root, F12F_QUALIFICATION_REL),
    }
    from scripts import f12_matrix as fm

    files["schemas.json"]["comparator_matrix"] = {
        "ruling": "PREREG_H1_TF A5 chair ruling on F12e (b)",
        "manifest": _rel_or_none(root, MATRIX_MANIFEST_REL),
        "builder": _rel_sha(root, "scripts/f12_matrix.py"),
        "pooling_code": _rel_sha(root, "src/wsp/features/pool.py"),
        "widths": fm.widths(),
        "zero_edit_input": "x0_v4: the pinned pooling code on the zero delta (F12a.2(h))",
    }
    files["f12i_c1.json"]["parent_support_tensor_hashes"] = {
        "source": src,
        "base_state_hash": facts["base_state_hash"],
        "lora": facts["f12i_lora_parents"],
        "astra_and_c4": F12I_AWAITING,
    }
    files["code_hashes_c1.json"]["token_facts"] = src


def _rel_or_none(root, rel):
    p = Path(root) / rel
    return {"path": rel, "sha256": _sha_file(p) if p.exists() else None}


def write_draft(root=ROOT) -> list:
    root = Path(root)
    out = []
    for name, obj in build_draft(root).items():
        p = root / DRAFT_DIR / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
        out.append(p)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("build-draft", "verify"))
    a = p.parse_args(argv)
    if a.command == "build-draft":
        print(json.dumps({"written": [str(x.relative_to(ROOT)) for x in write_draft()]}, indent=1))
        return 0
    fails = verify_c1(ROOT)
    print(json.dumps({"verified": not fails, "failures": fails}, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
