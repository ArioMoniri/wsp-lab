"""A5.11 manifest package builder on a synthetic toy zoo (no data/, CPU, no git)."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import a5_build_manifests as bm
from wsp.eval import response_intervention as ri
from wsp.eval.batch import effective_config_hash

REV = "r" * 40
CFG = {
    "generation": {"tier": "tf_v1", "max_new_tokens": 4, "batch_size": 1, "seed": 1},
    "benchmarks": {
        "mmlu_pro": {"n": 2},
        "wikitext_ppl": {"revision": "w" * 40, "n_windows": 4, "window": 8},
    },
}
COUNTS = {"mmlu_pro": 2, "wikitext_ppl": 4}
WINDOWS = [(str(i), f"tok{i}") for i in range(4)]

# id, family, parent, component, status, dev member, norm, features, bytes, extra
ROWS = [
    # LoRA: six D3 components; L6 shares CL1 with L1 (represented by L1)
    ("L1", "lora_sft", None, "CL1", "development", True, 0.1, True, True, {}),
    ("L2", "lora_sft", None, "CL2", "development", True, 0.2, True, True, {}),
    ("L3", "lora_sft", None, "CL3", "development", True, 0.3, True, True, {}),
    ("L4", "lora_sft", None, "CL4", "development", True, 0.4, True, True, {}),
    ("L5", "lora_sft", None, "CL5", "development", True, 0.5, True, True, {}),
    ("L6", "lora_sft", None, "CL1", "development", True, 0.6, True, True, {}),
    ("L7", "lora_sft", None, "CL7", "development", True, 0.7, True, True, {}),
    # exclusions
    ("La", "lora_sft", None, "CL2", "development", True, 0.2, True, False, {"alias_of": "L2"}),
    ("Ln", "lora_sft", None, "CL8", "development", True, 0.8, False, True, {}),
    ("Lc", "lora_sft", None, "CC1", "confirmatory", False, 0.9, True, True, {}),
    ("Lu", "lora_sft", None, "CL9", "development", False, 0.9, True, True, {}),
    ("Ls", "lora_sft", None, "CL10", "excluded", False, 0.9, True, True, {"eligibility": "smoke"}),
    ("Lm", "lora_sft", None, "CL11", "excluded", False, 0.9, True, True, {"eligibility": None}),
    ("T1", "task_vector", None, "CT1", "development", True, 1.0, True, True, {}),
    # chain lineages K1 (k1a <- k1b, only k1b in D3), K2 (k2a <- k2b, re-trained), K3 (k3a <- k3b lost)
    ("k1a", "chain", None, "K1", "development", False, 1.0, True, True, {}),
    ("k1b", "chain", "k1a", "K1", "development", True, 0.4, True, True, {}),
    ("k2a", "chain", None, "K2", "development", False, 2.0, True, True, {}),
    ("k2b", "chain", "k2a", "K2", "development", True, 0.3, True, True, {"exact": "re-trained"}),
    ("k3a", "chain", None, "K3", "development", True, 0.9, True, True, {}),
    ("k3b", "chain", "k3a", "K3", "development", True, None, False, False, {"exact": "lost"}),
]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def _record(evals, ref_id, entry_hashes, *, windows=WINDOWS, **over):
    d = evals / ref_id
    d.mkdir(parents=True)
    payload = {
        "eval_status": "complete",
        "repeat_id": None,
        "config_hash": effective_config_hash(CFG),
        "model_revision": REV,
        "counts": COUNTS,
        "entry_hashes": entry_hashes,
        "run_id": f"run-{ref_id}",
    } | over
    (d / "eval.json").write_text(json.dumps(payload))
    _write_jsonl(
        d / "records.jsonl",
        [{"benchmark": "mmlu_pro", "item_id": "0"}]
        + [
            {"benchmark": "wikitext_ppl", "item_id": i, "extra": {"token_ids_hash": h}}
            for i, h in windows
        ],
    )


def toy(tmp_path, rows=ROWS, extra_norms=()):
    root = tmp_path / "repo"
    zoo, features, evals = root / "zoo", root / "features", root / "evals"
    for d in (zoo, features, evals):
        d.mkdir(parents=True)
    manifest = [{"type": "header", "base_revision": REV}]
    status, norms, components = [], [], {}
    ledger = {}
    for i, fam, parent, comp, cstatus, dev, norm, feat, has_bytes, extra in rows:
        row = {"id": i, "family": fam, "parent_id": parent, "base_revision": REV}
        if extra.get("eligibility", "production") is not None:
            row["eligibility"] = extra.get("eligibility", "production")
        manifest.append(row)
        (zoo / i).mkdir()
        if has_bytes:
            (zoo / i / "delta.safetensors").write_bytes(f"bytes-{i}".encode())
        if feat:  # roots carry the effective norm; their norms-file inc_fro is a decoy (raw)
            (features / i).mkdir()
            meta = {"delta_relative_to": "base", "norms": {"fro_effective": norm}}
            (features / i / "features.json").write_text(json.dumps(meta if parent is None else {}))
        status.append(
            {"id": i, "family": fam, "delta_present": has_bytes, "alias_of": extra.get("alias_of")}
        )
        raw = norm if parent or norm is None else 10 - norm  # reversed order for roots
        n = {"id": i, "family": fam, "parent_id": parent, "inc_fro": raw, "base_fro": norm}
        n["exactness"] = extra.get("exact", "bitwise")
        n["stored"] = has_bytes or bool(extra.get("alias_of"))
        norms.append(n)
        c = components.setdefault(
            comp,
            {
                "component_id": comp,
                "status": cstatus,
                "members": [],
                "development_members": [],
                "source_identities": [],
            },
        )
        c["members"].append(i)
        if dev:
            c["development_members"].append(i)
        if has_bytes:
            h = bm.entry_delta_hash(bm.artefact_files(zoo / i))
            c["source_identities"].append(f"delta:{h}")
        if extra.get("alias_of"):
            ledger[f"key-{i}"] = {"canonical": extra["alias_of"], "aliases": [i]}
    norms += list(extra_norms)
    _write_jsonl(zoo / "manifest.jsonl", manifest)
    _write_jsonl(zoo / "artefact_status.jsonl", status)
    _write_jsonl(root / "norms.jsonl", norms)
    panel = {"components": list(components.values()), "alias_ledger": ledger}
    (root / "panel.json").write_text(json.dumps(panel))
    (root / "eval_tf_v1.yaml").write_text(yaml.safe_dump(CFG))
    _record(evals, "base", {})
    sha = hashlib.sha256((zoo / "L1" / "delta.safetensors").read_bytes()).hexdigest()
    _record(evals, "L1", {"delta.safetensors": sha})
    _record(evals, "L4", {"delta.safetensors": "0" * 64})  # bytes differ from the zoo
    _record(evals, "k1b", {"delta.safetensors": "0" * 64}, repeat_id="g1")  # not a primary
    return {
        "root": root,
        "zoo": zoo,
        "norms": root / "norms.jsonl",
        "features": features,
        "evals": evals,
        "panel": root / "panel.json",
        "config": root / "eval_tf_v1.yaml",
    }


def build(paths, n_lora=3, n_chain=3):
    return bm.build(paths, n_lora, n_chain, git_sha="0" * 40, read_blob=lambda p: p.encode())


def test_build_is_byte_deterministic(tmp_path):
    paths = toy(tmp_path)
    a, b = build(paths), build(paths)
    assert a == b
    assert set(a) == {
        "components.jsonl",
        "selection.json",
        "references.json",
        "text_groups.json",
        "code_hashes.json",
        "README.md",
    }
    assert str(tmp_path) not in b"".join(a.values()).decode()  # no absolute paths


def test_readme_lists_every_file_sha(tmp_path):
    files = build(toy(tmp_path))
    readme = files["README.md"].decode()
    for name, blob in files.items():
        if name != "README.md":
            assert f"`{name}` | `{hashlib.sha256(blob).hexdigest()}`" in readme
    sel = json.loads(files["selection.json"])
    assert sel["components_jsonl_sha256"] == hashlib.sha256(files["components.jsonl"]).hexdigest()


def test_inventory_eligibility_reasons(tmp_path):
    files = build(toy(tmp_path))
    inv = {r["id"]: r for r in map(json.loads, files["components.jsonl"].decode().splitlines())}
    assert len(inv) == len(ROWS)
    assert {k: inv[k]["eligibility"] for k in ("La", "Ln", "Lc", "Lu", "Ls", "Lm", "T1")} == {
        "La": "excluded:alias_of:L2",
        "Ln": "excluded:no_feature_manifest",
        "Lc": "excluded:not_d3_development_member:confirmatory",
        "Lu": "excluded:not_d3_development_member:development",
        "Ls": "excluded:eligibility:smoke",
        "Lm": "excluded:eligibility:missing",
        "T1": "excluded:unsupported_family:task_vector",
    }
    assert inv["k3b"]["eligibility"] == "excluded:artefact_lost"
    assert inv["k2b"]["eligibility"] == "eligible"  # re-trained lineage is flagged, not excluded
    assert inv["k2b"]["exactness"] == "re-trained"
    assert inv["La"]["canonical_id"] == "L2" and inv["La"]["bytes_via_alias"] == "L2"
    # roots: effective norm from the feature manifest, never the raw adapter norm of the norms file
    assert inv["L1"]["norm_source"] == "features.fro_effective"
    assert (inv["L1"]["effective_norm"], inv["L1"]["inc_fro"]) == (0.1, 9.9)
    assert inv["k1b"]["norm_source"] == "a5_incremental_norms.inc_fro"
    assert inv["k1b"]["effective_norm"] == 0.4


def test_selection_reproduces_select_edits_from_components(tmp_path):
    files = build(toy(tmp_path))
    sel = json.loads(files["selection.json"])["registered"]
    assert sel["status"] == "selected"
    assert [(s["family"], s["tertile_label"], s["edit_id"]) for s in sel["selected"]] == [
        ("lora_sft", "low", "L1"),
        ("lora_sft", "middle", "L3"),
        ("lora_sft", "high", "L5"),
        ("chain", "low", "k2b"),
        ("chain", "middle", "k1b"),
        ("chain", "high", "k3a"),
    ]
    # components.jsonl is the select_edits sidecar; rerunning the tool on it gives the same selection
    inv = [json.loads(x) for x in files["components.jsonl"].decode().splitlines()]
    manifest = bm.read_jsonl(tmp_path / "repo" / "zoo" / "manifest.jsonl")
    norms = {r["id"]: r["effective_norm"] for r in inv if r["feature_manifest"]}
    again = ri.select_edits(manifest, norms, inv, n_lora=3, n_chain=3)
    assert [s.edit_id for s in again] == [s["edit_id"] for s in sel["selected"]]
    assert sel["counts"]["lora_sft"] == {"low": 1, "middle": 1, "high": 1}


def test_references_identity_and_pricing(tmp_path):
    paths = toy(tmp_path)
    refs = json.loads(build(paths)["references.json"])
    rows = {r["edit_id"]: r for r in refs["rows"]}
    assert rows["L1"]["original"]["on_file"] and rows["L1"]["original"]["run_id"] == "run-L1"
    assert rows["L1"]["predecessor_id"] == "base" and rows["L1"]["predecessor"]["on_file"]
    assert rows["L3"]["original"]["failed_checks"] == ["no_record"]
    assert rows["k1b"]["predecessor_id"] == "k1a"
    assert rows["k1b"]["original"]["failed_checks"] == ["entry_hashes", "primary"]
    assert rows["k1b"]["closure_proof"]["hash_in_source_identities"]
    assert refs["missing_originals"] == ["L3", "L5", "k1b", "k2b", "k3a"]
    assert refs["missing_predecessors"] == ["k1a", "k2a"]  # base on file; k3a is a root
    assert (refs["n_D"], refs["n_P"], refs["n_variants"]) == (5, 2, 18)
    assert refs["additional_minutes"] == round(7 * 1.04, 4)
    assert refs["total_minutes"] == round(25 * 1.04, 4)


def test_reference_entry_hash_mismatch_is_not_on_file(tmp_path):
    paths = toy(tmp_path)
    identity = {"config_hash": effective_config_hash(CFG), "model_revision": REV, "counts": COUNTS}
    st = bm.reference_status(paths["evals"], "L4", paths["zoo"] / "L4", identity, WINDOWS)
    assert not st["on_file"] and st["failed_checks"] == ["entry_hashes"]
    st = bm.reference_status(paths["evals"], "L1", paths["zoo"] / "L1", identity, WINDOWS[:3])
    assert st["failed_checks"] == ["windows"]


LOST_ROWS = [*ROWS, ("k3c", "chain", "k3b", "K3", "development", True, None, True, True, {})]


def _set_norm_error(paths, row_id, error):
    norms = bm.read_jsonl(paths["norms"])
    for r in norms:
        if r["id"] == row_id:
            r["error"] = error
    paths["norms"].write_text("".join(json.dumps(r) + "\n" for r in norms))


def _set_status(paths, row_id, **fields):
    rows = bm.read_jsonl(paths["zoo"] / "artefact_status.jsonl")
    for r in rows:
        if r["id"] == row_id:
            r.update(fields)
    (paths["zoo"] / "artefact_status.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_lost_predecessor_is_a_registered_exclusion_with_positive_proof(tmp_path):
    paths = toy(tmp_path, LOST_ROWS)
    # No error string is inspected: a non-'predecessor:' message still gets the positive proof.
    _set_norm_error(paths, "k3c", "something else entirely")
    files = build(paths)
    sel = json.loads(files["selection.json"])
    reg = sel["registered"]
    assert reg["status"] == "selected" and reg["ruling"].startswith("REGISTERED")
    excl = reg["registered_exclusions"]["k3c"]
    assert excl["reason"] == "actual_predecessor_state_lost:k3b"
    assert excl["proof"]["sidecar"] == {
        "delta_present": False,
        "alias_of": None,
        "artefact_lost": None,
    }
    assert excl["proof"]["predecessor_bytes_present"] is False
    assert sel["pre_ruling_literal"]["status"] == "blocked"
    assert sel["pre_ruling_literal"]["status_label"].startswith("SUPERSEDED")
    assert "conditional" not in sel
    assert json.loads(files["references.json"])["selection_basis"].startswith("registered")
    readme = files["README.md"].decode()
    assert "Status: **selected**" in readme and "NOT REGISTERED" not in readme
    assert "`excluded:actual_predecessor_state_lost` | 1 | REGISTERED" in readme


def test_lost_predecessor_check_is_positive_not_an_error_prefix(tmp_path):
    # The old prefix test: 'predecessor:' error, but the predecessor has aliased bytes -> still blocks.
    paths = toy(tmp_path / "a", LOST_ROWS)
    _set_norm_error(paths, "k3c", "predecessor: [Errno 2] No such file")
    _set_status(paths, "k3b", alias_of="k3a")
    sel = json.loads(build(paths)["selection.json"])
    assert (
        sel["registered"]["status"] == "blocked" and not sel["registered"]["registered_exclusions"]
    )
    assert sel["registered"]["blocking_ids"] == ["k3c"] and sel["pre_ruling_literal"] is None
    # delta_present not explicitly false (unknown) is not a proof either.
    paths = toy(tmp_path / "b", LOST_ROWS)
    _set_status(paths, "k3b", delta_present=None)
    assert json.loads(build(paths)["selection.json"])["registered"]["status"] == "blocked"
    # No sidecar row for the predecessor: no proof.
    inv = {
        r["id"]: r
        for r in map(
            json.loads,
            build(toy(tmp_path / "c", LOST_ROWS))["components.jsonl"].decode().splitlines(),
        )
    }
    assert bm.lost_predecessor_proof(inv["k3c"], inv, {}) is None
    status = {"k3b": {"id": "k3b", "delta_present": False, "alias_of": None}}
    assert bm.lost_predecessor_proof(inv["k3c"], inv, status)["predecessor_id"] == "k3b"
    # An alias-ledger holder with bytes is not a lost predecessor.
    holder = dict(inv["k3b"], canonical_id="k3a")
    assert bm.lost_predecessor_proof(inv["k3c"], inv | {"k3b": holder}, status) is None
    # Rows that do not block (eligible or excluded) never get an exclusion.
    assert bm.lost_predecessor_proof(inv["k1b"], inv, status) is None


def _write_package(files, out):
    out.mkdir(parents=True)
    for name, blob in files.items():
        (out / name).write_bytes(blob)


def test_procedural_freeze_from_committed_package(tmp_path):
    paths = toy(tmp_path, LOST_ROWS)
    (paths["root"] / "configs").mkdir()
    (paths["root"] / "configs/eval_tf_v1.yaml").write_bytes(paths["config"].read_bytes())
    files = build(paths)
    out = tmp_path / "pkg"
    _write_package(files, out)
    status = paths["zoo"] / "artefact_status.jsonl"
    env = {"python": "3.x", "numpy": "1"}
    frozen = bm.build_procedural_freeze(out, paths["root"], status, environment=env)
    assert set(frozen) == {"selection.json", "references.json", "code_hashes.json", "README.md"}
    sel = json.loads(frozen["selection.json"])
    freeze = sel.pop("procedural_freeze")
    assert sel == json.loads(files["selection.json"])  # same registered selection as a full build
    assert freeze["label"] == "procedural freeze" and freeze["date"] == "2026-11-07"
    assert "evidence" in freeze["separate_from"]
    assert freeze["registered_edit_ids"] == sel["registered"]["edit_ids"]
    assert (
        freeze["registered_edit_ids_sorted_sha256"]
        == hashlib.sha256(json.dumps(sel["registered"]["edit_ids"]).encode()).hexdigest()
    )
    for fam, c in freeze["cutpoints"].items():
        assert c["keys"] == sel["registered"]["ledger"]["families"][fam]["cutpoints"]
    assert freeze["inventory"]["registered_exclusions"] == ["k3c"]
    assert freeze["retrained_lineage_picks"] == ["k2b"]
    assert freeze["board_a_cutpoints_match"] is False  # toy inventory, not the registered one
    code_bytes = frozen["code_hashes.json"]
    assert freeze["code_hashes_json_sha256"] == hashlib.sha256(code_bytes).hexdigest()
    code = json.loads(code_bytes)
    assert code["pinned_from"] == "working tree" and code["environment"] == env
    assert code["resolved_config"]["records_match"]
    assert code["resolved_config"]["effective_config_hash"] == effective_config_hash(CFG)
    for path in (
        "scripts/a5_build_manifests.py",
        "scripts/a5_build_panels.py",
        "scripts/a5_simulation.py",
        "tests/test_a5_simulation.py",
        "tests/test_a5_build_manifests.py",
        "tests/test_a5_build_panels.py",
        "src/wsp/eval/response_intervention.py",
        "src/wsp/eval/batch.py",
        "configs/eval_tf_v1.yaml",
    ):
        assert path in code["files"]
    assert (
        code["files"]["configs/eval_tf_v1.yaml"]["sha256"]
        == hashlib.sha256(paths["config"].read_bytes()).hexdigest()
    )
    readme = frozen["README.md"].decode()
    assert "## Procedural freeze (2026-11-07)" in readme and "superseded" in readme
    # The a5-selection-v1 layout (ruling run labelled 'conditional') converts to the same result.
    v1 = json.loads(files["selection.json"])
    literal = {k: v for k, v in v1.pop("pre_ruling_literal").items() if k != "status_label"}
    reg = v1.pop("registered")
    body = {k: v for k, v in reg.items() if k not in ("ruling", "registered_exclusions")}
    v1["registered"] = literal
    v1["conditional"] = {
        "ruling": "CONDITIONAL, NOT REGISTERED",
        "extra_exclusions": {"k3c": "actual_predecessor_state_lost:k3b"},
    } | body
    (out / "selection.json").write_bytes(bm.dumps(v1))
    again = bm.build_procedural_freeze(out, paths["root"], status, environment=env)
    assert again["selection.json"] == frozen["selection.json"]
    # A later full build must reproduce the frozen ids and cutpoints, or it stops.
    full = build_frozen(paths, freeze)
    assert json.loads(full["selection.json"])["procedural_freeze"] == freeze
    tampered = dict(freeze, registered_edit_ids=freeze["registered_edit_ids"][1:])
    with pytest.raises(AssertionError, match="procedural freeze"):
        build_frozen(paths, tampered)
    # The sidecar must be the committed input; anything else stops before relabelling.
    status.write_text(status.read_text() + "\n")
    with pytest.raises(ValueError, match="sidecar"):
        bm.build_procedural_freeze(out, paths["root"], status, environment=env)


def build_frozen(paths, freeze):
    return bm.build(
        paths, 3, 3, git_sha="0" * 40, read_blob=lambda p: p.encode(), procedural_freeze=freeze
    )


def test_shortage_is_recorded_not_filled(tmp_path):
    files = build(toy(tmp_path), n_lora=9)
    sel = json.loads(files["selection.json"])["registered"]
    assert sel["status"] == "unresolved_shortage"
    assert "no cross-cell replacement" in sel["action"]
    assert "lora_sft tertile 0" in sel["error"]
    refs = json.loads(files["references.json"])
    assert refs["selection_status"] == "unresolved_shortage" and "rows" not in refs


def test_text_groups_hash_and_consistency(tmp_path):
    paths = toy(tmp_path)
    tg = json.loads(build(paths)["text_groups.json"])
    assert tg["records_consistent"] and len(tg["windows"]) == 4
    ids = sorted(w["window_id"] for w in tg["windows"])
    assert tg["window_ids_sorted_sha256"] == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    assert tg["two_window_blocks"] == [[0, 1], [2, 3]]
    _record(paths["evals"], "L2", {}, windows=[(str(i), f"other{i}") for i in range(4)])
    tg = json.loads(build(paths)["text_groups.json"])
    assert not tg["records_consistent"]


def test_code_hashes_pin_blob_bytes(tmp_path):
    code = json.loads(build(toy(tmp_path))["code_hashes.json"])
    assert code["git_sha"] == "0" * 40 and code["schema"] == "a5-code-hashes-v2"
    path = "src/wsp/eval/response_intervention.py"
    assert code["files"][path]["sha256"] == hashlib.sha256(path.encode()).hexdigest()
    assert set(code["files"]) == set(bm.CODE_FILES)
    assert {"python", "numpy", "torch", "transformers"} <= set(code["environment"])
    assert code["resolved_config"]["effective_config_hash"] == effective_config_hash(CFG)
    assert bm.verify_code_hashes(code, lambda p: p.encode()) == []
    assert bm.verify_code_hashes(code, lambda p: b"x" if p == path else p.encode()) == [path]


def test_refuses_to_write_under_data():
    with pytest.raises(SystemExit):
        bm.main(["--out", str(Path(bm.ROOT) / "data" / "a5_manifests")])


def _snapshot_for(files):
    """A closed-snapshot stub whose pins equal the build's live inputs."""
    inputs = json.loads(files["selection.json"])["inputs"]
    content = {sk: {f: inputs[k]["sha256"]} for k, (sk, f) in bm.SNAPSHOT_PINS.items()}
    content["tf_v1_records"] = {
        "index_sha256": inputs["tf_v1_records"]["index_sha256"],
        "content_index_sha256": inputs["tf_v1_records"]["content_index_sha256"],
    }
    content["feature_manifest"] = {
        "sha256": inputs["feature_manifests"]["manifest_sha256"],
        "content_index_sha256": inputs["feature_manifests"]["content_index_sha256"],
    }
    return {"content": content, "closed_at": "2026-11-08T00:00:00Z", "base_ref": "0" * 40}


def test_closed_snapshot_pins_contents_and_must_match(tmp_path):
    paths = toy(tmp_path)
    files = build(paths)
    inputs = json.loads(files["selection.json"])["inputs"]
    assert "content_index_sha256" in inputs["feature_manifests"]
    assert "content_index_sha256" in inputs["tf_v1_records"]
    snap = bm.dumps(_snapshot_for(files))
    pinned = bm.build(paths, 3, 3, git_sha="0" * 40, read_blob=lambda p: p.encode(), snapshot=snap)
    sel = json.loads(pinned["selection.json"])
    assert sel["inputs"]["snapshot"]["sha256"] == hashlib.sha256(snap).hexdigest()
    assert sel["inputs_scope"].startswith("closed pre-G2 snapshot")
    assert "snapshot.json" in pinned["README.md"].decode()
    # Feature-file CONTENT (not only the id set) is pinned: changing bytes breaks the snapshot.
    (paths["features"] / "L1" / "features.json").write_text("{}")
    with pytest.raises(ValueError, match="feature_manifests.content_index_sha256"):
        bm.build(paths, 3, 3, git_sha="0" * 40, read_blob=lambda p: p.encode(), snapshot=snap)
    # Record contents too.
    paths = toy(tmp_path / "b")
    snap = bm.dumps(_snapshot_for(build(paths)))
    with open(paths["evals"] / "L1" / "records.jsonl", "a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="tf_v1_records.content_index_sha256"):
        bm.build(paths, 3, 3, git_sha="0" * 40, read_blob=lambda p: p.encode(), snapshot=snap)


def test_selection_diff_reports_changed_ids_and_cutpoints(tmp_path):
    paths = toy(tmp_path, LOST_ROWS)
    sel = json.loads(build(paths)["selection.json"])
    reg = sel["registered"]
    freeze = {
        "registered_edit_ids": [*reg["edit_ids"][1:], "Zz"],
        "cutpoints": {f: {"keys": [[0.0, "x"], [1.0, "y"]]} for f in reg["ledger"]["families"]},
    }
    diff = bm.selection_diff(freeze, sel)
    assert diff["ids_added"] == [reg["edit_ids"][0]] and diff["ids_removed"] == ["Zz"]
    assert diff["cutpoints_rebuilt"]["chain"] == reg["ledger"]["families"]["chain"]["cutpoints"]
    assert diff["n_distinct_components"] == len(reg["edit_ids"])
