"""F12g (Board A F12g review, conditions 2 and 4; chair acceptance of F12g) and the F12 C1 freeze.

* the scoped item 9 builder: Board A's two scope paragraphs verbatim, the 16 LoRA roots and 32 part ids from the
  committed streaming plan evidence, the registered Astra and C4 rules, the F12-I anchor and prices; it rejects missing
  LoRA evidence and any claim of a completed Astra/C4 construction, and records failed or no-op roots unsubstituted;
* the unchanged verifier barrier on temporary repositories: a complete scoped item 9 passes; "awaiting" fails even
  with recomputed hashes; omission, stale bytes, missing authority and bad chronology fail;
* the freeze builders: governed files without draft or awaiting markers, the exposure file from the register, the
  restricted payload bytes, the governed code hashes and the static prerequisite binding check.

Synthetic or committed metadata only; no outcome, nothing under data/evals."""

import copy
import hashlib
import json
import re
import shutil

import pytest

from scripts import f12_additivity_prepare as fa
from scripts import f12_c1 as c1
from scripts import f12_simulation as fs
from tests.test_f12c_governance import IDS
from tests.test_f12d_prereqs import _git, add_board_a_record

ROOT = fs.ROOT
REVIEW = ROOT / "docs/reviews/council/board-a-f12g-2026-11-12.md"
H0I = "7ba2957b30755704b670156131cdfe6393fc56051851cb4d094d6ea1d82b961e"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _facts_pop():
    facts = json.loads((ROOT / c1.FACTS_REL).read_text())
    pop = json.loads((ROOT / "docs/prereg/a5_manifests/f12/population.json").read_text())
    return facts, pop


def _review_paragraphs():
    """Board A's quoted scope paragraphs, as they stand in the committed review."""
    text = REVIEW.read_text()
    block, paras, cur = False, [], []
    for ln in text.splitlines():
        if ln.startswith("> Scope: F12 plan record"):
            block = True
        if block:
            if not ln.startswith(">"):
                break
            body = ln[1:].strip()
            if body:
                cur.append(body)
            elif cur:
                paras.append(" ".join(cur))
                cur = []
    if cur:
        paras.append(" ".join(cur))
    return paras


# ------------------------------------------------------------------ scoped item 9


def test_scope_paragraphs_are_board_a_verbatim():
    assert list(c1.F12G_SCOPE) == _review_paragraphs()
    assert len(c1.F12G_SCOPE) == 2
    prereg = " ".join((ROOT / "docs/PREREG_H1_TF.md").read_text().split())
    assert c1.F12G_INDEPENDENCE in prereg


def test_scoped_item9_from_committed_evidence():
    obj = c1.build_draft(ROOT)["f12i_c1.json"]
    assert obj["scope"] == list(c1.F12G_SCOPE)
    assert obj["independence"] == c1.F12G_INDEPENDENCE
    assert obj["own_chain_H0"] == H0I == c1.H0_F12I
    assert "awaiting" not in json.dumps(obj).lower()
    _, pop = _facts_pop()
    assert sorted(obj["lora"]) == sorted(pop["f12i_lora_roots"]) and len(obj["lora"]) == 16
    pids = [p["part_id"] for r in obj["lora"].values() for p in r["parts"]]
    assert len(pids) == 32 == len(set(pids))
    for rid, r in obj["lora"].items():
        assert [p["label"] for p in r["parts"]] == ["early", "late"]
        for p in r["parts"]:
            assert p["part_id"] == fa.part_id(rid, p["label"])
            assert re.fullmatch(r"[0-9a-f]{64}", p["increment_sha256"])
            assert set(p["support_base_tensor_hashes"]) == set(p["support"]["tensors"])
        assert r["bitwise_sum_verified"] is True
        assert r["parent_increment_sha256"] == r["full_increment_sha256"]
        assert set(r["source_files"]) >= {
            "lora/adapter_model.safetensors",
            "lora/adapter_config.json",
        }
    ev = obj["construction_evidence"]
    for key in ("plan", "run_record"):
        assert ev[key]["sha256"] == _sha((ROOT / ev[key]["path"]).read_bytes())
    assert ev["driver"]["path"] == "scripts/f12_additivity_stream.py"
    astra, c4 = obj["astra_rule"], obj["c4_rule"]
    assert "one part per changed tensor" in astra["split"]
    assert "drops that pair" in astra["failed_qualification"]
    assert c4["tensor"] == "model.layers.12.mlp.down_proj.weight"
    assert c4["stored_shape_out_in"] == [1024, 3584] and c4["boundary"] == 1792
    assert "fail" in c4["shape_requirement"] and "own-definition" in c4["status"]
    assert obj["planned"] == {
        "parts": 36,
        "fulls": 18,
        "lora_parts": 32,
        "astra_parts": 2,
        "c4_parts": 2,
    }
    assert obj["prices_gpu_min"] == {
        "parts": 37.44,
        "additional_fulls_max": 18.72,
        "all_in_ceiling": 56.16,
    }
    assert obj["ceiling_gpu_min"] == 56.16
    for forbidden in ("astra_and_c4", "parent_support_tensor_hashes"):
        assert forbidden not in obj


def _evidence(tmp_path, mutate=None, mutate_run=None):
    plan = json.loads((ROOT / c1.ADDITIVITY_PLAN_REL).read_text())
    run = json.loads((ROOT / c1.ADDITIVITY_RUN_REL).read_text())
    if mutate:
        mutate(plan)
    tmp_path.mkdir(parents=True, exist_ok=True)
    pp = tmp_path / "additivity_plan.json"
    pp.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    run["plan_sha256"] = _sha(pp.read_bytes())
    if mutate_run:
        mutate_run(run)
    rp = tmp_path / "additivity_plan_run.json"
    rp.write_text(json.dumps(run, indent=1, sort_keys=True) + "\n")
    return pp, rp


def _scoped(tmp_path, mutate=None, mutate_run=None):
    facts, pop = _facts_pop()
    pp, rp = _evidence(tmp_path, mutate, mutate_run)
    return c1.scoped_f12i(ROOT, facts, pop, plan_path=pp, run_path=rp)


def _lora(plan):
    return [e for e in plan["entries"] if e["kind"] == "lora_layer_split"]


def test_builder_accepts_unmodified_evidence_copy(tmp_path):
    assert c1.build_draft(ROOT)["f12i_c1.json"]["lora"] == _scoped(tmp_path)["lora"]


def test_builder_rejects_missing_lora_evidence(tmp_path):
    facts, pop = _facts_pop()
    with pytest.raises(ValueError, match="LoRA construction evidence"):
        c1.scoped_f12i(ROOT, facts, pop, plan_path=tmp_path / "none.json", run_path=tmp_path / "x")

    def drop(plan):
        plan["entries"] = [e for e in plan["entries"] if e["full_id"] != pop["f12i_lora_roots"][3]]

    with pytest.raises(ValueError, match="LoRA roots"):
        _scoped(tmp_path / "a", drop)

    def strip_parts(plan):
        _lora(plan)[0].pop("parts")

    with pytest.raises(ValueError, match="parts"):
        _scoped(tmp_path / "b", strip_parts)

    def wrong_parent(plan):
        _lora(plan)[1]["full_increment_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="parent increment"):
        _scoped(tmp_path / "c", wrong_parent)

    def wrong_pop(plan):
        plan["population_json_sha256"] = "1" * 64

    with pytest.raises(ValueError, match="population"):
        _scoped(tmp_path / "d", wrong_pop)

    def wrong_part_id(plan):
        _lora(plan)[2]["parts"][1]["part_id"] = "ffffffffffffffff"

    with pytest.raises(ValueError, match="part id"):
        _scoped(tmp_path / "e", wrong_part_id)

    with pytest.raises(ValueError, match="run record"):
        _scoped(tmp_path / "f", None, lambda r: r.update(plan_sha256="2" * 64))


def test_builder_rejects_any_completed_astra_or_c4_construction(tmp_path):
    def built(kind):
        def m(plan):
            e = next(x for x in plan["entries"] if x["kind"] == kind)
            e.pop("status", None)
            e["parts"] = [{"label": "tensor:a", "part_id": "x"}]

        return m

    with pytest.raises(ValueError, match="Astra/C4"):
        _scoped(tmp_path / "a", built("astra_tensor_split"))
    with pytest.raises(ValueError, match="Astra/C4"):
        _scoped(tmp_path / "b", built("c4_column_split"))

    def stand_in(plan):
        plan["entries"].append(
            {"full_id": "stand-in:x", "kind": "column_split_stand_in", "parts": []}
        )

    with pytest.raises(ValueError, match="Astra/C4"):
        _scoped(tmp_path / "c", stand_in)


def test_builder_records_failed_and_noop_roots_unsubstituted(tmp_path):
    _, pop = _facts_pop()
    rid_fail, rid_noop = pop["f12i_lora_roots"][4], pop["f12i_lora_roots"][5]

    def m(plan):
        e = next(x for x in _lora(plan) if x["full_id"] == rid_fail)
        for k in ("parts", "full_increment_sha256", "n_full_tensors"):
            e.pop(k)
        e.update(
            bitwise_sum_verified=False,
            failure="x: realised parts do not sum bitwise",
            I_defined=False,
        )
        n = next(x for x in _lora(plan) if x["full_id"] == rid_noop)
        n["parts"][0].update(noop=True, I_status="undefined:noop_part")
        n["I_defined"] = False

    obj = _scoped(tmp_path, m)
    assert sorted(obj["lora"]) == sorted(pop["f12i_lora_roots"])
    f = obj["lora"][rid_fail]
    assert (
        f["bitwise_sum_verified"] is False
        and "do not sum bitwise" in f["failure"]
        and f["parts"] == []
    )
    assert (
        obj["lora"][rid_noop]["parts"][0]["noop"] is True
        and obj["lora"][rid_noop]["I_defined"] is False
    )
    assert obj["construction_evidence"]["roots_failed"] == [rid_fail]
    assert obj["construction_evidence"]["noop_parts"] == 1


# ------------------------------------------------------------------ verifier barrier (unchanged verify_c1)


def _repo(tmp_path, item9, *, verify=True, same_commit=False):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/f12_simulation.py", root / "scripts/f12_simulation.py")
    man = root / c1.MANIFEST_DIR
    for name in c1.C1_FILES:
        p = man / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name == "code_hashes_c1.json":
            obj = {
                "schema": fs.C1_SCHEMA,
                "status": c1.FROZEN_STATUS,
                "files": dict(IDS["code_sha256"]),
            }
        elif name == "f12i_c1.json":
            obj = item9
        else:
            obj = {"schema": f"toy-{name}", "status": "C1 frozen"}
        p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    _git(root, "init", "-q")
    c1.write_manifest(root)
    _git(root, "add", "-A")
    if same_commit:
        add_board_a_record(root)  # review and manifest land in one commit
        return root
    _git(root, "commit", "-qm", "C1 freeze")
    if verify:
        add_board_a_record(root)
    return root


def _item9():
    return {"status": c1.FROZEN_STATUS} | {
        k: v for k, v in c1.build_draft(ROOT)["f12i_c1.json"].items() if k != "status"
    }


def _fails(root):
    return c1.verify_c1(root, code_identity=IDS["code_sha256"])


def test_complete_scoped_item9_passes_the_unchanged_verifier(tmp_path):
    assert _fails(_repo(tmp_path, _item9())) == []


def test_item9_with_awaiting_fails_even_with_recomputed_hashes(tmp_path):
    for i, word in enumerate(("awaiting", "AWAITING", "Awaiting")):
        obj = _item9()
        obj["astra_rule"] = dict(obj["astra_rule"], note=f"Astra hashes {word} F8")
        fails = _fails(_repo(tmp_path / str(i), obj))
        assert any("f12i_c1.json is a draft or awaiting" in f for f in fails), fails
        assert not any("differs from the manifest" in f or "chain mismatch" in f for f in fails)


def test_omission_stale_bytes_authority_and_chronology_still_fail(tmp_path):
    item9 = _item9()
    root = _repo(tmp_path / "omit", item9)
    (root / c1.MANIFEST_DIR / "f12i_c1.json").unlink()
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "drop item 9")
    assert any("f12i_c1.json missing" in f for f in _fails(root))
    root = _repo(tmp_path / "h9", item9)
    m = root / c1.MANIFEST_PATH
    m.write_text("".join(ln for ln in m.read_text().splitlines(True) if not ln.startswith("H9 ")))
    assert any("incomplete" in f for f in _fails(root))
    root = _repo(tmp_path / "stale", item9)
    p = root / c1.MANIFEST_DIR / "f12i_c1.json"
    p.write_text(p.read_text().replace('"ceiling_gpu_min": 56.16', '"ceiling_gpu_min": 60.0'))
    fails = _fails(root)
    assert any("f12i_c1.json differs from the manifest digest" in f for f in fails)
    assert any("f12i_c1.json is not committed" in f for f in fails)
    _git(root, "commit", "-qam", "edit item 9")
    assert any("f12i_c1.json differs from the manifest digest" in f for f in _fails(root))
    root = _repo(tmp_path / "auth", item9, verify=False)
    assert any("no Board A C1 verification record" in f for f in _fails(root))
    root = _repo(tmp_path / "chrono", item9, same_commit=True)
    assert any("chronology" in f for f in _fails(root))


# ------------------------------------------------------------------ freeze builders


def _register_repo(tmp_path):
    """A temporary repository holding committed copies of the attestation register and the access audit report. The
    exposure builders are tested on it, never on the live register, so editing the register never fails the hook."""
    root = tmp_path / "register_repo"
    for rel in (c1.REGISTER_REL, c1.ACCESS_AUDIT_REL):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "register copy")
    return root


def test_governed_package_has_no_draft_or_awaiting_markers(tmp_path):
    files = c1.build_c1(ROOT, exposure_root=_register_repo(tmp_path))
    assert list(files) == list(c1.C1_FILES)
    for name, data in files.items():
        text = data.decode()
        assert "awaiting" not in text.lower(), name
        if name != "noise_payload/restricted.json":
            obj = json.loads(text)
            assert obj["status"] == c1.FROZEN_STATUS, name
            # the exposure file quotes the register's rows verbatim, whose role descriptions name earlier sessions'
            # work ("F12 drafting", "C1 draft fill"); a draft marker anywhere else is the word "draft"
            rest = {k: v for k, v in obj.items() if k not in ("rows", "access_audit")}
            assert not re.search(r"\bdraft\b", json.dumps(rest), re.IGNORECASE), name
            assert "at_freeze" not in obj and "draft_pin" not in obj, name
    pin = json.loads((ROOT / c1.MANIFEST_DIR / "noise_payload_restricted_pin.json").read_text())
    assert _sha(files["noise_payload/restricted.json"]) == pin["restricted_copy"]["sha256"]


def test_exposure_file_cites_the_register_and_the_access_audit_faithfully(tmp_path):
    src = _register_repo(tmp_path)
    exp = c1.build_exposure(ROOT, register_root=src)
    reg = src / c1.REGISTER_REL
    assert exp["register"] == {"path": c1.REGISTER_REL, "sha256": _sha(reg.read_bytes())}
    rows = c1._register_rows(reg.read_text().split("## Register", 1)[1].split("\n## ", 1)[0])
    assert exp["rows"] == rows
    assert [r["#"] for r in exp["rows"]][:5] == ["1", "2", "3", "4", "5"]
    assert "UNATTESTABLE" in exp["rows"][4]["status"]
    e = exp["entries"]
    assert e["5"]["status"] == "UNATTESTABLE" and "22" in e["5"]["sources"]
    audit = exp["access_audit"]
    rep = src / c1.ACCESS_AUDIT_REL
    assert audit["path"] == c1.ACCESS_AUDIT_REL and audit["sha256"] == _sha(rep.read_bytes())
    assert audit["overall"].startswith("**Overall: PARTLY SUPPORTED**")
    assert audit["overall"] in rep.read_text()
    assert [f["Script and tests"].split(",")[0] for f in audit["findings"]] == [
        "`scripts/f12_population.py`",
        "`scripts/f12_tokens.py`",
        "`scripts/f12_additivity_prepare.py`",
    ]
    for f in audit["findings"]:
        assert "| " + " | ".join(f.values()) + " |" in rep.read_text()
    for needle in (
        "0412799",
        c1.ACCESS_AUDIT_REL,
        "not a substitute attestation",
        "not adjudicated",
    ):
        assert needle in e["5"]["static_access_audit"], needle
    assert e["22"]["status"] == "UNATTESTABLE" and len(e["22"]["sessions"]) == 2
    assert e["21"]["status"] == "qualified"
    for k in ("15b", "15c", "15d"):
        assert e[k]["prescore_audits"] == "read directly", k
    for k in ("17", "20"):
        assert "raw" in e[k]["disclosure"], k
    by_row = {r["#"]: r for r in rows}
    assert (
        e["10"]["status"] == by_row["10"]["status"] and e["11"]["status"] == by_row["11"]["status"]
    )
    assert exp["pending"] == [
        f"row {r['#']}: {r['status']}" for r in rows if "PENDING" in r["status"]
    ]
    assert "Board A" in exp["adjudication"]


def test_exposure_refuses_an_uncommitted_register_or_audit(tmp_path):
    src = _register_repo(tmp_path)
    reg = src / c1.REGISTER_REL
    reg.write_text(reg.read_text() + "\nedit\n")
    with pytest.raises(ValueError, match="not committed"):
        c1.build_exposure(ROOT, register_root=src)
    _git(src, "commit", "-qam", "edit")
    c1.build_exposure(ROOT, register_root=src)
    (src / c1.ACCESS_AUDIT_REL).unlink()
    with pytest.raises(ValueError, match="not committed"):
        c1.build_exposure(ROOT, register_root=src)


def test_c1_draft_overwrite_after_the_freeze_is_caught_by_both_checks(tmp_path):
    """``python scripts/f12_simulation.py c1-draft`` is not changed to refuse (f12_simulation.py is bound to the noise
    check and pilot); writing its draft pin over the governed item 11 is caught by verify_c1's digest check and by the
    code-pin check the hook runs (``c1.code_pin_failures``)."""
    root = _repo(tmp_path, _item9())
    assert _fails(root) == [] and c1.code_pin_failures(root) == []
    p = root / c1.MANIFEST_DIR / "code_hashes_c1.json"
    draft = {
        "schema": fs.C1_SCHEMA,
        "status": "draft, pending Board A C1 verification",
        "files": dict(IDS["code_sha256"]),
    }
    p.write_text(json.dumps(draft, indent=1, sort_keys=True) + "\n")  # what c1-draft writes
    fails = _fails(root)
    assert any("code_hashes_c1.json differs from the manifest digest" in f for f in fails), fails
    pins = c1.code_pin_failures(root)
    assert any("status" in f for f in pins), pins
    _git(root, "commit", "-qam", "c1-draft after the freeze")
    assert any("code_hashes_c1.json differs from the manifest digest" in f for f in _fails(root))
    assert c1.code_pin_failures(root)


def test_code_pin_failures_before_the_freeze_expect_the_draft_pin(tmp_path):
    root = tmp_path / "r"
    (root / c1.MANIFEST_DIR).mkdir(parents=True)
    shutil.copy(ROOT / "scripts/f12_simulation.py", tmp_path / "x.py")
    obj = {"schema": fs.C1_SCHEMA, "status": "draft, pending Board A C1 verification", "files": {}}
    (root / c1.MANIFEST_DIR / "code_hashes_c1.json").write_text(json.dumps(obj))
    assert c1.code_pin_failures(root) == []
    obj["status"] = c1.FROZEN_STATUS
    (root / c1.MANIFEST_DIR / "code_hashes_c1.json").write_text(json.dumps(obj))
    assert any("status" in f for f in c1.code_pin_failures(root))
    obj["files"] = {"scripts/missing.py": "0" * 64}
    obj["status"] = "draft, pending Board A C1 verification"
    (root / c1.MANIFEST_DIR / "code_hashes_c1.json").write_text(json.dumps(obj))
    assert any("scripts/missing.py" in f for f in c1.code_pin_failures(root))


def test_governed_code_hashes_pin_the_frozen_code_and_inputs():
    obj = c1.build_code_hashes(ROOT)
    assert obj["schema"] == c1.C1_CODE_SCHEMA and obj["status"] == c1.FROZEN_STATUS
    files = obj["files"]
    for rel in (
        "scripts/f12_gauge_tokens.py",
        "scripts/f12_matrix.py",
        "scripts/f12_qualify.py",
        "scripts/f12_budget.py",
        "scripts/f12_c1.py",
        "scripts/f12_payload_extract.py",
        "scripts/f12_additivity_prepare.py",
        "scripts/f12_additivity_stream.py",
        "tests/test_f12_additivity_stream.py",
        "tests/test_f12g_c1_freeze.py",
        "src/wsp/zoo/apply.py",
        "src/wsp/features/response_field.py",
        c1.ADDITIVITY_PLAN_REL,
        c1.ADDITIVITY_RUN_REL,
        c1.TOKEN_MANIFEST,
        c1.GAUGE_MANIFEST_REL,
        c1.MATRIX_MANIFEST_REL,
        "docs/prereg/f12_manifests/f12_pilot_acceptance.json",
    ):
        assert rel in files, rel
    for rel in fs.CODE_IDENTITY_FILES:
        assert rel in files
    for rel, sha in files.items():
        assert _sha((ROOT / rel).read_bytes()) == sha, rel
    for name in c1.C1_FILES:
        assert f"{c1.MANIFEST_DIR}/{name}" not in files
    assert set(obj["versions"]) >= {"python", "numpy", "torch", "scipy", "lightgbm"}


def test_prerequisite_binding_is_checked_against_current_bytes(tmp_path):
    b = c1.prerequisite_binding(ROOT)
    assert b["noise_check"]["code_unchanged"] and b["pilot_acceptance"]["code_unchanged"]
    assert b["payload_matches_item6"] and b["matrix_matches_manifest"]
    rec = json.loads((ROOT / c1.MANIFEST_DIR / "f12f_noise_check.json").read_text())
    bad = copy.deepcopy(rec)
    k = next(iter(bad["code_sha256"]))
    bad["code_sha256"][k] = "0" * 64
    assert c1.code_unchanged(ROOT, bad["code_sha256"]) is False
    assert c1.code_unchanged(ROOT, rec["code_sha256"]) is True


def test_write_c1_refuses_once_the_manifest_exists(tmp_path):
    m = tmp_path / c1.MANIFEST_PATH
    m.parent.mkdir(parents=True)
    m.write_text("H0 x\n")
    with pytest.raises(ValueError, match="never rewritten"):
        c1.write_c1(tmp_path)
