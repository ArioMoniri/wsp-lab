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

* F12g (Board A F12g review, conditions 2 and 4; chair acceptance of F12g): item 9 ``f12i_c1.json`` is the scoped F12
  plan record of F12-I (``scoped_f12i``): Board A's two scope paragraphs verbatim, the 16 LoRA roots and 32 part ids
  from the committed streaming plan evidence, the registered Astra and C4 rules, H0-I, prices and ceiling. The Astra and
  C4 construction hashes belong wholly to F12-I's own C1. ``C1_FILES`` and ``verify_c1`` are unchanged.
* ``build_c1(root)`` / ``write_c1(root)`` build and write the governed C1 files (status ``FROZEN_STATUS``): the
  exposure file from the committed attestation register (``build_exposure``), the pinned restricted payload bytes,
  the governed code hashes (``build_code_hashes``) and a static prerequisite binding check (``prerequisite_binding``);
  ``write_c1`` refuses once ``MANIFEST_C1.sha256`` exists.

    python scripts/f12_c1.py build-draft     # rewrites docs/prereg/f12_manifests/c1_draft/
    python scripts/f12_c1.py freeze          # writes the governed C1 files and regenerates the draft
    python scripts/f12_c1.py manifest        # writes MANIFEST_C1.sha256 (write_manifest)
    python scripts/f12_c1.py verify          # prints the failures; exit 1 unless C1 is verified
"""

from __future__ import annotations

import argparse
import copy
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
        raise ValueError(
            "eligible-row folds recomputed from population.json differ from the token facts"
        )
    qual = root / QUALIFICATION_REL
    qref = (
        _rel_sha(root, QUALIFICATION_REL)
        if qual.exists()
        else "f12e qualification report not yet written"
    )
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
        "token_input_hashes": {"source": src, "token_manifest": facts["token_manifest"]}
        | facts["token_inputs"],
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
    # F12g as accepted by Board A: item 9 is the scoped F12 plan record of F12-I (no Astra/C4 hashes, which belong
    # wholly to F12-I's own C1); it needs the committed LoRA construction evidence and fails closed without it
    files["f12i_c1.json"] = {"status": FILLED_STATUS} | scoped_f12i(root, facts, pop)
    files["code_hashes_c1.json"]["token_facts"] = src


def _rel_or_none(root, rel):
    p = Path(root) / rel
    return {"path": rel, "sha256": _sha_file(p) if p.exists() else None}


# ------------------------------------------------------------------------ F12g scoped item 9

ADDITIVITY_PLAN_REL = "docs/prereg/a5_manifests/f12/additivity_plan.json"
ADDITIVITY_RUN_REL = "docs/prereg/a5_manifests/f12/additivity_plan_run.json"
ADDITIVITY_XCHECK_REL = "docs/prereg/a5_manifests/f12/additivity_plan_crosscheck.json"
ADDITIVITY_DRIVER = "scripts/f12_additivity_stream.py"
F12R_DIR = "docs/prereg/a5_manifests/f12/f12r"
PLAN_REQUALIFICATIONS = (
    f"{F12R_DIR}/plan_requalification.json",
    f"{F12R_DIR}/plan_requalification2.json",
)
BASE_LEAVES_REL = f"{F12R_DIR}/base_model_leaves.json"
BASE_LOAD_CHECK_REL = f"{F12R_DIR}/base_load_check.json"
TOKEN_RERUN_REL = f"{F12R_DIR}/token_rerun_run.json"
TOKEN_REQUAL_REL = f"{F12R_DIR}/token_requalification.json"
TOKEN_MANIFEST_REQUAL_REL = f"{F12R_DIR}/tokens_manifest_requal.json"
DISPOSITION_HEADING = "#### A5 chair resource disposition on F12 input builds"
RERUN_RULING_HEADING = "#### A5 chair ruling on the F12 token requalification rerun"
ADDITIVITY_SCRIPT = "scripts/f12_additivity_prepare.py"
POPULATION_REL = "docs/prereg/a5_manifests/f12/population.json"
F12G_REVIEW_REL = "docs/reviews/council/board-a-f12g-2026-11-12.md"
F12G_SCOPE = (
    (
        "Scope: F12 plan record under F12g, complete for this scope only. This file is not F12-I's own C1 and does not "
        "authorise F12-I construction, contraction, scheduling or scoring. It records the 16 registered LoRA roots and "
        "their verified layer-split evidence, the registered Astra and C4 split definitions and source requirements, the "
        "F12-I chain anchor, and the registered prices and ceiling."
    ),
    (
        "The Astra and C4 construction hashes, eligibility and dependencies are not part of F12's C1. They are recorded "
        "wholly in F12-I's own C1 under docs/prereg/f12_manifests/f12i/, after F8a's C1 and the F8a.4 Astra "
        "qualification disposition. Failed Astra qualification drops that pair and is recorded as such; no replacement or "
        "purchase of the full bypasses it. These exclusions from this file do not exclude any requirement from F12-I's "
        "own C1/C2 or change F12-I conditions 8–10."
    ),
)
F12G_INDEPENDENCE = (
    "F12's numerical simulation consumes no F12-I construction, forecast or outcome data; governance verification "
    "reads the frozen F12 plan in item 9 and verifies pinned source/configuration bytes."
)
C4_TENSOR = "model.layers.12.mlp.down_proj.weight"


def _sha_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


def _ref(root, p):
    p = Path(p)
    try:
        rel = str(p.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        rel = str(p)
    return {"path": rel, "sha256": _sha_file(p)}


def _plan_requalification(root, plan_bytes) -> dict:
    """The F12-R requalifications of the committed plan with the hardened driver and confined base loader (fail
    closed unless each is identical to the committed plan), and the base-model leaves pin and load check."""
    root = Path(root)
    recs = []
    for rel in PLAN_REQUALIFICATIONS:
        p = root / rel
        if not p.exists():
            raise ValueError(f"F12g: plan requalification record {rel} is missing")
        r = json.loads(p.read_text())
        ok = (
            r.get("identical") is True
            and r.get("status_entries_identical") is True
            and not r.get("differences")
            and (r.get("reference") or {}).get("sha256") == _sha_bytes(plan_bytes)
        )
        if not ok:
            raise ValueError(
                f"F12g: plan requalification {rel} is not identical to the committed plan"
            )
        recs.append(
            _ref(root, p)
            | {
                k: r[k]
                for k in ("identical", "status_entries_identical", "lora_entries_sha256", "plan")
            }
        )
    chk = json.loads((root / BASE_LOAD_CHECK_REL).read_text())
    leaves = _ref(root, root / BASE_LEAVES_REL)
    if (
        chk.get("matches_pin") is not True
        or (chk.get("pin") or {}).get("sha256") != leaves["sha256"]
    ):
        raise ValueError("F12g: the base-model load check does not reproduce the pinned leaves")
    return {
        "basis": "note F12-R and its second addendum: the hardened streaming driver (A26 checks, authenticated "
        "inputs, 4 GB cap, confined base loader) reproduces every LoRA entry and status entry of the committed plan",
        "records": recs,
        "base_model_leaves": leaves,
        "base_load_check": _ref(root, root / BASE_LOAD_CHECK_REL)
        | {"matches_pin": True, "base_state_hash": chk.get("base_state_hash")},
    }


def scoped_f12i(root, facts, pop, plan_path=None, run_path=None) -> dict:
    """Item 9 under F12g as accepted by Board A: the F12 plan record of F12-I, complete for its scope (no status).

    Fails closed on missing or inconsistent LoRA construction evidence and on any claim of a completed Astra or C4
    construction; a root that failed its bitwise check or has a no-op part is recorded as such, never substituted."""
    from scripts import f12_additivity_prepare as fa

    root = Path(root)
    plan_path = Path(plan_path or root / ADDITIVITY_PLAN_REL)
    run_path = Path(run_path or root / ADDITIVITY_RUN_REL)
    xcheck_path = root / ADDITIVITY_XCHECK_REL
    if not plan_path.exists() or not run_path.exists() or not xcheck_path.exists():
        raise ValueError(
            "F12g: the committed LoRA construction evidence (streaming plan, its run record and the dry-run "
            "cross-check) is missing"
        )
    plan_bytes = plan_path.read_bytes()
    plan, run = json.loads(plan_bytes), json.loads(run_path.read_text())
    xcheck = json.loads(xcheck_path.read_text())
    if run.get("plan_sha256") != _sha_bytes(plan_bytes):
        raise ValueError("F12g: the plan's run record names other plan bytes")
    if xcheck.get("identical") is not True:
        raise ValueError("F12g: the streaming driver's dry-run cross-check is not identical")
    if plan.get("schema") != fa.SCHEMA or plan.get("mode") != "plan":
        raise ValueError("F12g: the LoRA construction evidence is not an f12-additivity-v1 plan")
    pop_sha = _sha_file(root / POPULATION_REL)
    if plan.get("population_json_sha256") != pop_sha or run["population"]["sha256"] != pop_sha:
        raise ValueError("F12g: the plan was not built from the governed population.json")
    if plan["base"]["state_hash"] != facts["base_state_hash"]:
        raise ValueError("F12g: the plan's base state differs from the token facts' base state")
    for e in plan["entries"]:
        if e.get("kind") != "lora_layer_split" and ("parts" in e or "status" not in e):
            raise ValueError(
                f"F12g: the plan claims an Astra/C4 construction ({e.get('full_id')}, {e.get('kind')}); those "
                "belong wholly to F12-I's own C1"
            )
    lora = [e for e in plan["entries"] if e.get("kind") == "lora_layer_split"]
    if sorted(e["full_id"] for e in lora) != sorted(pop["f12i_lora_roots"]) or len(lora) != 16:
        raise ValueError("F12g: the plan's LoRA roots are not the 16 registered F12-I LoRA roots")
    parents = facts["f12i_lora_parents"]
    by_id = {e["full_id"]: e for e in lora}
    out = {}
    for rid in pop["f12i_lora_roots"]:
        e, par = by_id[rid], parents[rid]
        rec = {
            "family": par["family"],
            "eligibility": par["eligibility"],
            "parent_increment_sha256": par["increment_sha256"],
            "n_edited_tensors": par["n_edited_tensors"],
            "n_nonzero_tensors": par["n_nonzero_tensors"],
            "source_files": e["source"]["files"],
            "split_rule": e["split_rule"],
            "bitwise_sum_verified": e["bitwise_sum_verified"],
            "I_defined": e["I_defined"],
        }
        if not e["bitwise_sum_verified"]:
            out[rid] = rec | {"failure": e["failure"], "parts": []}
            continue
        if len(e.get("parts") or []) != 2:
            raise ValueError(f"F12g: {rid}: a verified entry without its two parts")
        if e["full_increment_sha256"] != par["increment_sha256"]:
            raise ValueError(
                f"F12g: {rid}: plan full increment differs from the token facts' parent increment"
            )
        want = {"early": sorted(par["part1"]), "late": sorted(par["part2"])}
        parts = []
        for p in e["parts"]:
            if p["part_id"] != fa.part_id(rid, p["label"]):
                raise ValueError(
                    f"F12g: {rid}: part id {p['part_id']} is not the registered part id"
                )
            if sorted(p["support"]["tensors"]) != want.get(p["label"]):
                raise ValueError(
                    f"F12g: {rid}: {p['label']} support differs from the token facts' layer split"
                )
            parts.append(
                {
                    k: p[k]
                    for k in (
                        "label",
                        "part_id",
                        "support",
                        "n_tensors",
                        "increment_sha256",
                        "noop",
                        "I_status",
                    )
                }
                | {
                    "support_base_tensor_hashes": {
                        n: par["support_base_tensor_hashes"][n] for n in p["support"]["tensors"]
                    }
                }
            )
        out[rid] = rec | {
            "full_increment_sha256": e["full_increment_sha256"],
            "n_full_tensors": e["n_full_tensors"],
            "parts": parts,
        }
    failed = [r for r, x in out.items() if not x["bitwise_sum_verified"]]
    return {
        "scope": list(F12G_SCOPE),
        "authority": {
            "amendment": "PREREG_H1_TF A5 amendment F12g (12 Nov 2026, 07:14 UTC) and the A5 chair acceptance of "
            "F12g as amended by Board A (12 Nov 2026, 07:19 UTC)",
            "board_a_review": _ref(root, root / F12G_REVIEW_REL),
            "registered_item": "F12a.6(c) item 9 at H9 of MANIFEST_C1.sha256; C1_FILES and verify_c1 unchanged",
        },
        "independence": F12G_INDEPENDENCE,
        "own_chain_H0": H0_F12I,
        "own_chain_anchor": f"sha256(UTF-8 'f12i-manifest-v1:' + S), S = {S_HEAD} (F12a.10(a))",
        "own_c1_location": "docs/prereg/f12_manifests/f12i/",
        "construction": _ref(root, root / ADDITIVITY_SCRIPT),
        "construction_evidence": {
            "plan": _ref(root, plan_path),
            "run_record": _ref(root, run_path),
            "dry_run_cross_check": _ref(root, xcheck_path)
            | {"identical": True, "root": xcheck.get("root")},
            "driver": _ref(root, root / ADDITIVITY_DRIVER),
            "population_json_sha256": pop_sha,
            "base_state_hash": plan["base"]["state_hash"],
            "threads": run.get("threads"),
            "peak_rss_bytes": run.get("peak_rss_bytes"),
            "roots_built": len(out) - len(failed),
            "roots_failed": failed,
            "noop_parts": sum(p["noop"] for x in out.values() for p in x["parts"]),
            "outcomes_read_intent": fa.outcomes_read_intent(),
            "evidence_driver_sha256": run["driver"]["sha256"],
            "requalification": _plan_requalification(
                root, (root / ADDITIVITY_PLAN_REL).read_bytes()
            ),
            "what_it_proves": "outcome-free CPU construction in memory: each realised part (stored delta restricted "
            "to its support, applied to the bf16 base and rounded) sums bitwise to the full realised increment, "
            "tensor by tensor; no part file is written and no field, forecast or outcome is read",
        },
        "lora_roots": len(out),
        "lora_root_ids_sha256": hashlib.sha256(json.dumps(sorted(out)).encode()).hexdigest(),
        "lora": out,
        "lora_rule": {
            "split": "layers 0-11 (early) versus 12-23 (late): two parts per root, 32 parts (F12.6(a); F12a.8(a))",
            "part_id": f"sha256(UTF-8 '{fa.PART_DOMAIN}' + full id + ':' + label)[:16]",
            "noop": "a part with zero realised increment is not written or scored; its entry's I is undefined",
            "failure": "a root failing the bitwise check is recorded with the failure and never substituted",
            "shared_sources": "each LoRA full is its F12 population row's stored adapter bytes (source_files), the "
            "same bytes the F12 token builder read; no new input",
        },
        "astra_rule": {
            "split": "one part per changed tensor of the qualified two-tensor patch (F12a.8(b); F8a.4 A6): 2 parts",
            "source_requirement": "a qualified F8a.4 replay record and its entry bytes, authenticated in F12-I's own "
            "C1; the construction helper's qualified/entry_dir check alone does not authenticate upstream authority",
            "failed_qualification": "a failed F8a.4 qualification drops that pair and is recorded as such; no "
            "replacement patch or purchase of the full bypasses it",
        },
        "c4_rule": {
            "tensor": C4_TENSOR,
            "stored_shape_out_in": [1024, 3584],
            "boundary": 1792,
            "split": "input columns j < 1792 (cols_lo) versus j >= 1792 (cols_hi) of the stored [out, in] weight: 2 "
            "parts (F12.6(c); F12a.8(c))",
            "shape_requirement": "a C4 delta that is not exactly this one tensor at stored shape [1024, 3584] fails "
            "closed; nothing is reshaped, padded or substituted",
            "status": "own-definition contrast (F8a.2 C4), not a certified reproduction",
            "source_requirement": "F8a's C1-frozen construction, neither construction_degenerate nor a no-op, "
            "passing the identity verifier; never the dry-run stand-in",
            "forecast": "F12a.9: signed quadratic cross-term forecast and descriptive J discrepancy",
        },
        "full_eligibility_rule": "F12a.10(c): a full already scored by F12 or F8 on the identity-matched common base "
        "reference is reused; a LoRA full is eligible iff its row passes the F12 C1 population and identity audits; "
        "Astra only after a passed F8a.4 qualification; C4 only from F8a's C1-frozen construction; a panel-level "
        "non-purchase is not a refusal, an upstream validity refusal is never bypassed",
        "planned": {"parts": 36, "fulls": 18, "lora_parts": 32, "astra_parts": 2, "c4_parts": 2},
        "planned_counts_note": "planned counts are not claims of completed Astra/C4 construction or eligibility",
        "prices_gpu_min": {"parts": 37.44, "additional_fulls_max": 18.72, "all_in_ceiling": 56.16},
        "pricing_rule": "F12a.10(d)-(e): 36 parts x 1.04 min = 37.44 GPU-min; at most 18 additional fulls x 1.04 min "
        "= 18.72 GPU-min; 56.16 GPU-min is the separately metered all-in stop-and-reprice ceiling including "
        "base/reference overhead, failures and retries; after ten F12-I entries the rate is reprojected",
        "ceiling_gpu_min": 56.16,
    }


# ------------------------------------------------------------------------ C1 freeze (governed files)

FROZEN_STATUS = (
    "C1 file (PREREG_H1_TF F12a.6(c); item 9 scoped by F12g as accepted by Board A): frozen through "
    "MANIFEST_C1.sha256; not verified until Board A's separate C1 verification record and review"
)
REGISTER_REL = "docs/prereg/f12_attestations.md"
ACCESS_AUDIT_REL = "docs/reviews/council/f12-inputs-access-audit-2026-11-12.md"
ADOPTION_REVIEWS = {
    "adoption": "docs/reviews/council/f12-unexposed-adoption-2026-11-12.md",
    "recheck": "docs/reviews/council/f12-remedy-recheck-2026-11-12.md",
    "recheck2": "docs/reviews/council/f12-remedy-recheck2-2026-11-12.md",
}
PIN_REL = f"{MANIFEST_DIR}/noise_payload_restricted_pin.json"
NOISE_CHECK_REL = f"{MANIFEST_DIR}/f12f_noise_check.json"
PILOT_REPORT_REL = f"{MANIFEST_DIR}/f12f_pilot_report.json"
PILOT_ACCEPTANCE_REL = f"{MANIFEST_DIR}/f12_pilot_acceptance.json"
C1_INPUTS = (
    TOKEN_MANIFEST,
    GAUGE_MANIFEST_REL,
    MATRIX_MANIFEST_REL,
    POPULATION_REL,
    ADDITIVITY_PLAN_REL,
    ADDITIVITY_RUN_REL,
    ADDITIVITY_XCHECK_REL,
    "configs/eval_tf_v1.yaml",
    FACTS_REL,
    QUALIFICATION_REL,
    F12F_QUALIFICATION_REL,
    f"{MANIFEST_DIR}/f12e_base_tensor_hashes.json",
    f"{MANIFEST_DIR}/f12e_candidate_identities.json",
    f"{MANIFEST_DIR}/f12f_candidate_identities.json",
    NOISE_CHECK_REL,
    PILOT_REPORT_REL,
    f"{MANIFEST_DIR}/f12f_pilot_timing.json",
    f"{MANIFEST_DIR}/f12f_pilot_projection.json",
    f"{MANIFEST_DIR}/f12f_budget_ledger_snapshot.jsonl",
    PILOT_ACCEPTANCE_REL,
    "pyproject.toml",
    "uv.lock",
)
ENTRY_NOTES = {
    "5": {
        "status": "UNATTESTABLE",
        "session": "F12 inputs implementer (scripts/f12_population.py, scripts/f12_tokens.py, "
        "scripts/f12_additivity_prepare.py; commit 0412799); its delegate session was killed out of memory on 11 Nov "
        "2026 before reporting and cannot be asked",
        "sources": ["row 5", "22"],
        "static_access_audit": "commissioned by the chair (A5 chair acceptance of F12g (d)): an independent static "
        "access audit of the code at 0412799 (f12_population.py, f12_tokens.py, f12_additivity_prepare.py), listing "
        "every file path each opens and every field each reads; committed as {path} (sha256 {sha256}), cited in "
        "access_audit; evidence for Board A's adjudication under F12g condition 5, not a substitute attestation; not "
        "adjudicated here",
    },
    "22": {
        "status": "UNATTESTABLE",
        "sessions": [
            (
                "F12 inputs implementer (row 5; commit 0412799), killed out of memory on 11 Nov 2026 before reporting; "
                "its token run's manifest records outcomes_read: none, a code-written record, not a session attestation"
            ),
            (
                "first F12b session, killed out of memory on 11 Nov 2026 before committing anything; its uncommitted "
                "diagnostics script was rewritten by the F12b relaunch session (entry 18), which does not know its access"
            ),
        ],
        "for_board_a": "the chair records both as unattestable rather than inferring their exposure and asks Board A "
        "to rule whether F12a.1(d)'s unexposed-role or exploratory-amendment rule applies to the code they produced",
    },
    "21": {
        "status": "qualified",
        "session": "Board A methods reviewer, F12c review (row 15)",
        "qualification": "the commit hook launched repository-wide pytest, terminated after three progress dots; "
        "whether its collection-time filesystem access reached data/evals is not certain; the reviewer cannot give "
        "an unqualified no-access attestation",
    },
    "15b": {
        "session": "council round 6, representation member",
        "prescore_audits": "read directly",
        "detail": "portions of PRESCORE_INVENTORY_2.md, the Board A release review, and the top-level keys and "
        "summary of predictions_f11.json (its first 55 lines were requested; displayed coverage not certain); direct "
        "exposure to F11 calibration-record contents not certain",
    },
    "15c": {
        "session": "council round 6, world-model member",
        "prescore_audits": "read directly",
        "detail": "predictions_f11.json (summary, predictor block, power-field comparison, one full row and "
        "statistics computed over all 52 rows) and f11_calibration_record.json",
    },
    "15d": {
        "session": "council round 6, metrology member",
        "prescore_audits": "read directly",
        "detail": "PRESCORE_INVENTORY_2.md and the F11 calibration record; not predictions_f11.json",
    },
    "17": {
        "session": "F12 simulation implementer (restricted-copy extractor, v1)",
        "disclosure": "raw F8a.2 payload parsed: runs/a5/nullsize/inputs.json was read and json.loads parsed the "
        "whole object (donor_mean parsed into memory, never accessed, printed or written); during exploration the "
        "session printed per-donor means and SDs of centred_noise and the full donors[0] record including its audit "
        "and norm fields",
    },
    "20": {
        "session": "F12c implementer, confirmation of the raw-loader account",
        "disclosure": "raw F8a.2 payload parsed once (about 18:11 UTC, 11 Nov 2026) by "
        "test_real_payload_allowlist_and_panel_mean through the worktree's data symlink; json.loads parsed the whole "
        "file and only the allowlisted fields were kept; no value printed, logged or written; whether the file holds "
        "donor_mean is not certain to that session",
    },
    "10": {
        "session": "chair",
        "status": "row 10's status cell, verbatim (filled from the register)",
    },
    "11": {
        "session": "every later C1/C2 builder, analyst and Board A reviewer",
        "status": "row 11's status cell, verbatim (filled from the register)",
    },
}


def _register_rows(text):
    rows, header = [], None
    for ln in text.splitlines():
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def _audit_findings(text):
    """The access audit's overall finding paragraph and its findings table, verbatim."""
    sec = text.split("## Findings", 1)[1].split("\n## ", 1)[0]
    overall = next(p.strip() for p in sec.split("\n\n") if p.strip().startswith("**Overall:"))
    table = "\n".join(ln for ln in sec.splitlines() if ln.startswith("|"))
    return overall, _register_rows(table)


def build_exposure(root=ROOT, register_root=None) -> dict:
    """Item 1 from the attestation register as committed: its rows verbatim, its sha256 and the entries Board A must
    adjudicate, cited by number, plus the committed static access audit of the F12 inputs code. Nothing is resolved
    here. ``register_root`` (default ``root``) is the repository holding the committed register and audit report."""
    src = Path(register_root or root)
    for rel in (REGISTER_REL, ACCESS_AUDIT_REL, *ADOPTION_REVIEWS.values()):
        if not _committed_clean(src, rel):
            raise ValueError(f"{rel} is not committed or has uncommitted changes")
    text = (src / REGISTER_REL).read_text()
    rows = _register_rows(text.split("## Register", 1)[1].split("\n## ", 1)[0])
    by_row = {r["#"]: r for r in rows}
    audit_text = (src / ACCESS_AUDIT_REL).read_text()
    overall, findings = _audit_findings(audit_text)
    audit = {"path": ACCESS_AUDIT_REL, "sha256": _sha_file(src / ACCESS_AUDIT_REL)}
    entries = copy.deepcopy(ENTRY_NOTES)
    entries["5"]["static_access_audit"] = entries["5"]["static_access_audit"].format(**audit)
    for k in ("10", "11"):
        entries[k]["status"] = by_row[k]["status"]
    after = (
        "still unattestable for their own interactive history: the unexposed adoption review, the F12-R remedy and "
        "the two re-checks register or adopt retained choices; they do not supply either session's personal "
        "statement (F12a.1(c)); not adjudicated here"
    )
    for k in ("5", "22"):
        entries[k]["after_remedy"] = after
    reviews = {}
    for key, rel in ADOPTION_REVIEWS.items():
        text = (src / rel).read_text()
        disp = next(p.strip() for p in text.split("\n\n") if p.strip().startswith("**"))
        reviews[key] = {"path": rel, "sha256": _sha_file(src / rel), "disposition": disp}
    sec = (
        (src / ADOPTION_REVIEWS["adoption"])
        .read_text()
        .split("## (c) Overall session statements", 1)[1]
    )
    sec = sec.split("\n## ", 1)[0]
    statements = {
        tag: next(p.strip() for p in sec.split("\n\n") if p.strip().startswith(f"**{tag}:**"))
        for tag in ("A", "B", "Combined")
    }
    return {
        "requirement": "F12a.1(c)-(d): statements (i) old development outcomes, (ii) new outcomes, (iii) prescore "
        "audits for every decision-maker, code/input builder, extractor, token-run session, chair and approving "
        "reviewer through the freeze; one session never attests for another; a commit time is not an attestation",
        "register": {"path": REGISTER_REL, "sha256": _sha_file(src / REGISTER_REL)},
        "rows": rows,
        "entries": entries,
        "access_audit": audit
        | {
            "target": "commit 04127991845e246c2fab77293654b773674f01a2 (0412799)",
            "role": "independent static audit commissioned by the chair for Board A F12g condition 5 and register "
            "entry 22 / row 5; evidence for Board A, not an attestation for the killed implementer",
            "overall": overall,
            "findings": findings,
        },
        "unexposed_adoption": {
            "reviews": reviews,
            "adoption_review_session_statements": statements,
            "remedy_entry_24": {"row": by_row["24"], "register_entry": "Entry 24"},
            "session_A": "retained choices registered or adopted by unexposed roles: the adoption review registered or "
            "verified 12 and adopted 13 of 33 itemised choices and flagged eight (A10, A11, A19, A21, A24, A25, A26, "
            "A30); the unexposed F12-R implementer (register entry 24) remedied them; re-check 1 adopted A11, A21, "
            "A24, A25, A26 and A30, and re-check 2 adopted A10 and A19",
            "session_B": "covered by the adoption review: within its conservative envelope of every potentially "
            "inherited contribution, 13 retained items, five registered and eight adopted, zero flagged",
        },
        "pending": [f"row {r['#']}: {r['status']}" for r in rows if "PENDING" in r["status"]],
        "adjudication": "none in this file: the unattestable sessions (row 5, entry 22), the qualified Board A F12c "
        "attestation (entry 21), the direct prescore-audit readers (entries 15b, 15c, 15d), the raw-payload parses "
        "(entries 17 and 20), the static access audit's findings and the unexposed adoption review, remedy and "
        "re-checks are recorded for Board A to adjudicate at C1 verification (F12g condition 5; Board A F12-C1 "
        "condition 1)",
    }


def code_pin_failures(root=ROOT) -> list:
    """The code-pin check the hook runs: ``code_hashes_c1.json`` is the draft pin before the freeze and the governed
    item 11 once ``MANIFEST_C1.sha256`` exists, and every listed file is at its pinned bytes. It catches
    ``python scripts/f12_simulation.py c1-draft`` overwriting the governed item 11 after the freeze (that command is
    not changed to refuse, because f12_simulation.py is bound to the noise check and pilot)."""
    root = Path(root)
    obj = json.loads((root / MANIFEST_DIR / "code_hashes_c1.json").read_text())
    frozen = (root / MANIFEST_PATH).exists()
    want = FROZEN_STATUS if frozen else "draft, pending Board A C1 verification"
    F = []
    if obj.get("status") != want:
        F.append(
            f"code_hashes_c1.json status {obj.get('status')!r} is not {want!r} (frozen={frozen})"
        )
    bad = [
        rel
        for rel, sha in sorted((obj.get("files") or {}).items())
        if not (root / rel).exists() or _sha_file(root / rel) != sha
    ]
    if bad:
        F.append(f"code_hashes_c1.json: current bytes differ for {bad}")
    return F


def _prereg_block(text, heading) -> str:
    """The appended PREREG block whose heading line equals (or uniquely starts with) ``heading``: its text through the
    line before the next ``#### `` heading."""
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln == heading] or [
        i for i, ln in enumerate(lines) if ln.startswith(heading)
    ]
    if len(hits) != 1:
        raise ValueError(f"PREREG block {heading!r}: {len(hits)} matching headings")
    i = hits[0]
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("#### ")), len(lines))
    return "\n".join(lines[i:j]).rstrip("\n") + "\n"


def resource_bindings(root=ROOT) -> dict:
    """Item 10 at the second freeze (Board A F12-C1 conditions 2 and 3): the chair resource disposition and the token
    requalification ruling, each bound by heading and text sha256, and the rerun and comparison records."""
    root = Path(root)
    prereg = (root / "docs/PREREG_H1_TF.md").read_text()
    out = {}
    for key, prefix, commit in (
        ("disposition", DISPOSITION_HEADING, "159af3e"),
        ("requalification_ruling", RERUN_RULING_HEADING, "53976db"),
    ):
        block = _prereg_block(prereg, prefix)
        out[key] = {
            "heading": block.split("\n", 1)[0],
            "commit": commit,
            "text_sha256": _sha_bytes(block.encode()),
        }
    disp = _prereg_block(prereg, DISPOSITION_HEADING)
    out["disposition"]["future_boundary"] = " ".join(
        disp[disp.index("(d) *Future boundary") :].split()
    )
    ruling = _prereg_block(prereg, RERUN_RULING_HEADING)
    out["requalification_ruling"]["resource_cap"] = next(
        ln.strip() for ln in ruling.splitlines() if ln.startswith("resource-cap:")
    )
    run = json.loads((root / TOKEN_RERUN_REL).read_text())
    if run.get("status") != "complete" or run.get("within_cap") is not True:
        raise ValueError("the token requalification rerun record is not complete within its cap")
    out["token_rerun"] = {
        "record": _ref(root, root / TOKEN_RERUN_REL),
        "manifest": _ref(root, root / TOKEN_MANIFEST_REQUAL_REL),
        "status": run["status"],
        "exit_status": run.get("exit_status"),
        "git_head_at_launch": run.get("git_head_at_launch"),
        "cap_bytes": run["cap_bytes"],
        "stop_threshold_bytes": run.get("stop_threshold_bytes"),
        "within_cap": run["within_cap"],
        "peak_rss_bytes": run["peak_rss_bytes"]["process_ru_maxrss"],
        "peak_rss_sampled_tree_bytes": run["peak_rss_bytes"].get("max_sampled_tree"),
        "wall_seconds": run["wall_seconds_time"],
        "cpu_seconds": run["cpu_seconds"]["total"],
        "cpu_hours": run.get("cpu_hours"),
        "mem_available_at_launch_bytes": run.get("mem_available_at_launch_bytes"),
    }
    req = json.loads((root / TOKEN_REQUAL_REL).read_text())
    if req.get("identical") is not True:
        raise ValueError("the token requalification comparison is not identical")
    out["token_requalification"] = {
        "record": _ref(root, root / TOKEN_REQUAL_REL),
        "identical": True,
        "basis": req.get("basis"),
    }
    out["accounting"] = (
        "input-build CPU and memory are charged to the F12 inputs, not to the 190 CPU-h simulation ceiling; no "
        "retroactive memory compliance is claimed (chair resource disposition (c))"
    )
    return out


def restricted_payload_bytes(root=ROOT) -> bytes:
    """Item 6: the pinned v2 restricted copy, byte-exact (its own provenance block; the pin and extractor are in
    code_hashes_c1.json)."""
    root = Path(root)
    pin = json.loads((root / PIN_REL).read_text())
    want = pin["restricted_copy"]["sha256"]
    for p in (
        root / MANIFEST_DIR / "noise_payload/restricted.json",
        Path(pin["restricted_copy"]["path"]),
    ):
        if p.exists():
            data = p.read_bytes()
            if _sha_bytes(data) == want:
                return data
    raise ValueError(f"no restricted payload with the pinned sha256 {want}")


def _module_paths(root, mod):
    parts = mod.split(".")
    if parts[0] == "scripts":
        base = root / "scripts" / Path(*parts[1:])
        cands = [base.with_suffix(".py")]
    elif parts[0] == "wsp":
        base = root / "src" / Path(*parts)
        cands = [base.with_suffix(".py"), base / "__init__.py"]
        cands += [root / "src" / Path(*parts[:i]) / "__init__.py" for i in range(1, len(parts))]
    else:
        return []
    return [p for p in cands if p.exists()]


def local_import_closure(root, seeds) -> set:
    """Repository-local modules (scripts.*, wsp.*) imported anywhere, transitively, by the seed files."""
    import ast

    root = Path(root)
    todo, seen = [root / s for s in seeds], set()
    while todo:
        p = todo.pop()
        rel = str(p.relative_to(root))
        if rel in seen or p.suffix != ".py":
            continue
        seen.add(rel)
        for node in ast.walk(ast.parse(p.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for m in mods:
                todo += _module_paths(root, m)
    return seen


def code_unchanged(root, mapping) -> bool:
    return bool(mapping) and all(
        (Path(root) / rel).exists() and _sha_file(Path(root) / rel) == sha
        for rel, sha in mapping.items()
    )


def prerequisite_binding(root=ROOT) -> dict:
    """Static check (from the committed records, nothing re-run) that the registered noise check, the timing pilot
    and the chair's pilot acceptance stay bound to the frozen bytes."""
    root = Path(root)
    nc = json.loads((root / NOISE_CHECK_REL).read_text())
    pr = json.loads((root / PILOT_REPORT_REL).read_text())
    pa = json.loads((root / PILOT_ACCEPTANCE_REL).read_text())
    pin = json.loads((root / PIN_REL).read_text())
    mm = json.loads((root / MATRIX_MANIFEST_REL).read_text())
    ids = {
        "noise_check": nc["identities"],
        "pilot_report": pr["identities"],
        "pilot_acceptance": pa["identities"],
    }
    codes = {"noise_check": nc["code_sha256"]} | {
        k: ids[k]["code_sha256"] for k in ("pilot_report", "pilot_acceptance")
    }
    out = {
        "basis": "static: identities read from the committed records and compared with the frozen bytes; nothing "
        "was re-run",
        "noise_check": _ref(root, root / NOISE_CHECK_REL),
        "pilot_report": _ref(root, root / PILOT_REPORT_REL),
        "pilot_acceptance": _ref(root, root / PILOT_ACCEPTANCE_REL),
    }
    for k in codes:
        out[k] |= {"code_files": sorted(codes[k]), "code_unchanged": code_unchanged(root, codes[k])}
    out["payload_matches_item6"] = all(
        i["payload_restricted_sha256"] == pin["restricted_copy"]["sha256"] for i in ids.values()
    )
    out["matrix_matches_manifest"] = all(
        i["tokens_matrix_sha256"] == mm["matrix_sha256"] for i in ids.values()
    )
    out["acceptance_cites_committed_records"] = (
        pa["noise_check_sha256"] == out["noise_check"]["sha256"]
        and pa["pilot_report_sha256"] == out["pilot_report"]["sha256"]
    )
    return out


def candidate_identity_drift(root=ROOT) -> dict:
    """Pins of f12f_candidate_identities.json whose current bytes differ (disclosed, not repaired)."""
    root = Path(root)
    ci = json.loads((root / f"{MANIFEST_DIR}/f12f_candidate_identities.json").read_text())
    out = {}
    for rel, sha in sorted((ci.get("code") or {}).items()):
        cur = _sha_file(root / rel) if (root / rel).exists() else None
        if cur != sha:
            out[rel] = {"pinned": sha, "current": cur}
    d = ci.get("code_hashes_c1_draft") or {}
    if (
        d.get("path")
        and (root / d["path"]).exists()
        and _sha_file(root / d["path"]) != d.get("sha256")
    ):
        out[d["path"]] = {"pinned": d.get("sha256"), "current": _sha_file(root / d["path"])}
    return out


def build_code_hashes(root=ROOT) -> dict:
    """Item 11: every F12 script, test and config, their repository-local import closure, the frozen inputs and
    pins, with runtime versions."""
    from scripts import f12_simulation as fs

    root = Path(root)
    if root.resolve() != fs.ROOT.resolve():
        raise ValueError(
            "code hashes are built only for the repository the simulation module lives in"
        )
    inv = fs.c1_draft_inventory()
    files = set(inv["files"])
    files |= local_import_closure(root, sorted(f for f in files if f.endswith(".py")))
    files |= {rel for rel in C1_INPUTS if (root / rel).exists()}
    files |= {str(p.relative_to(root)) for p in sorted((root / F12R_DIR).glob("*.json"))}
    files -= {f"{MANIFEST_DIR}/{n}" for n in C1_FILES}
    return {
        "schema": C1_CODE_SCHEMA,
        "status": FROZEN_STATUS,
        "note": "F12a.6(c) item 11: sha256 of every F12 script and test, the configurations, the repository-local "
        "modules they import (transitively), the frozen inputs and pins (token, gauge-token and matrix manifests, "
        "population, F12-I LoRA construction evidence, token facts, qualification, candidate identities, noise check, "
        "pilot and pilot acceptance) and the dependency lock, with runtime versions.",
        "files": {rel: _sha_file(root / rel) for rel in sorted(files)},
        "restricted_payload": inv["restricted_payload"],
        "versions": inv["versions"],
        "candidate_identity_drift": candidate_identity_drift(root),
    }


def build_c1(root=ROOT, exposure_root=None) -> dict:
    """The governed C1 package: {name: bytes} in chain order. Fails closed if the prerequisite binding broke. The
    exposure file is built from the committed register in ``exposure_root`` (default ``root``)."""
    root = Path(root)
    draft = build_draft(root)
    binding = prerequisite_binding(root)
    broken = [
        k
        for k in ("noise_check", "pilot_report", "pilot_acceptance")
        if not binding[k]["code_unchanged"]
    ]
    if broken or not binding["payload_matches_item6"] or not binding["matrix_matches_manifest"]:
        raise ValueError(
            f"prerequisite identity binding broken ({broken}); restart rules apply, freeze refused"
        )
    objs = {}
    for name, obj in draft.items():
        objs[name] = {
            k: v
            for k, v in obj.items()
            if k not in ("status", "at_freeze", "draft_pin", "token_facts")
        }
    objs["exposure_c1.json"] = build_exposure(root, register_root=exposure_root)
    objs["resources.json"]["pilot_acceptance"] = _ref(root, root / PILOT_ACCEPTANCE_REL) | {
        "all_in_cpu_h": json.loads((root / PILOT_ACCEPTANCE_REL).read_text())["projection"][
            "all_in_cpu_h"
        ]
    }
    objs["resources.json"]["prerequisite_binding"] = binding
    objs["resources.json"]["resource_bindings"] = resource_bindings(root)
    objs["simulation_config.json"]["noise_check_and_pilot"] = {
        rel.rsplit("/", 1)[1]: _ref(root, root / rel)
        for rel in (
            NOISE_CHECK_REL,
            PILOT_REPORT_REL,
            f"{MANIFEST_DIR}/f12f_pilot_timing.json",
            f"{MANIFEST_DIR}/f12f_pilot_projection.json",
            f"{MANIFEST_DIR}/f12f_budget_ledger_snapshot.jsonl",
        )
    }
    objs["noise_payload/restricted.json"] = None
    objs["code_hashes_c1.json"] = build_code_hashes(root)
    out = {}
    for name in C1_FILES:
        if name == "noise_payload/restricted.json":
            out[name] = restricted_payload_bytes(root)
            continue
        obj = {"status": FROZEN_STATUS} | objs[name]
        out[name] = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    return out


def write_c1(root=ROOT) -> list:
    """Write the governed C1 files; refuses once MANIFEST_C1.sha256 exists (no C1 file is rewritten)."""
    root = Path(root)
    if (root / MANIFEST_PATH).exists():
        raise ValueError("MANIFEST_C1.sha256 exists: C1 files are never rewritten")
    written = []
    for name, data in build_c1(root).items():
        p = root / MANIFEST_DIR / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        written.append(p)
    return written


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
    p.add_argument("command", choices=("build-draft", "freeze", "manifest", "verify"))
    a = p.parse_args(argv)
    if a.command == "build-draft":
        print(json.dumps({"written": [str(x.relative_to(ROOT)) for x in write_draft()]}, indent=1))
        return 0
    if a.command == "freeze":
        gov = [str(x.relative_to(ROOT)) for x in write_c1()]
        drafts = [str(x.relative_to(ROOT)) for x in write_draft()]
        print(json.dumps({"governed": gov, "draft_regenerated": drafts}, indent=1))
        return 0
    if a.command == "manifest":
        out = write_manifest()
        print(
            json.dumps({"manifest": str(out.relative_to(ROOT)), "sha256": _sha_file(out)}, indent=1)
        )
        return 0
    fails = verify_c1(ROOT)
    print(json.dumps({"verified": not fails, "failures": fails}, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
