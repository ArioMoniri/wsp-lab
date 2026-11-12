"""F12f step 4: re-qualification with the gauge-transformed tokens and the real comparator matrix (chair ruling on F12e
(c)). Fabricated token, gauge and v4 inputs over the committed population only; nothing under data/ is opened."""

import copy
import hashlib
import json

import numpy as np
import pytest

from scripts import f12_matrix as fm
from scripts import f12_qualify as fq
from scripts import f12_simulation as fs
from tests.test_f12e_qualify import POP, POP_SHA, _fake_output
from tests.test_f12f_matrix import _fake_features, _fake_gauge


def _sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture()
def f12f(tmp_path):
    out, man = _fake_output(tmp_path)
    tok = fq.load_output(out)
    tsha = _sha(json.dumps(man).encode())
    gman, gsha = _fake_gauge(tmp_path, tok, tsha)
    froot, content = _fake_features(tmp_path, tok)
    m, facts = fm.assemble(token_manifest=man, token_manifest_sha256=tsha, gauge_manifest=gman,
                           gauge_manifest_sha256=gsha, population=POP, population_sha256=POP_SHA,
                           feature_content=content, features_dir=froot)  # fmt: skip
    mdoc = fm.write_outputs(m, facts, tmp_path / "matrix.npz", tmp_path / "matrix_manifest.json")
    gauge = fq.gauge_inputs(gman, gsha, tsha)
    matrix = fq.matrix_inputs(mdoc, _sha((tmp_path / "matrix_manifest.json").read_bytes()))
    return {"out": out, "man": man, "tok": tok, "tsha": tsha, "gman": gman, "gsha": gsha, "gauge": gauge,
            "matrix": matrix, "mdoc": mdoc, "tmp": tmp_path}  # fmt: skip


def _checks(rep):
    return {c["check"]: c for c in rep["checks"]}


def test_gauge_and_matrix_checks_pass_on_verified_f12f_inputs(f12f):
    x = f12f
    rep = fq.qualify(
        x["man"], POP, POP_SHA, x["out"], tok=x["tok"], gauge=x["gauge"], matrix=x["matrix"]
    )
    ch = _checks(rep)
    assert ch["gauge"]["pass"], ch["gauge"]
    assert ch["simulation_matrix_inputs"]["pass"], ch["simulation_matrix_inputs"]
    assert rep["pass"] is True and rep["first_failure"] is None
    g = ch["gauge"]["detail"]
    assert g["audit_inputs"]["gauge_manifest_sha256"] == x["gsha"]
    assert g["runner_implements_real_token_audit"] is True
    mx = ch["simulation_matrix_inputs"]["detail"]
    assert mx["matrix_sha256"] == x["mdoc"]["matrix_sha256"]
    assert mx["widths"]["statnn_total"] == 9376 and mx["widths"]["v4"] == 1378
    assert rep["order"][-2:] == ["gauge", "simulation_matrix_inputs"]


def test_without_f12f_inputs_the_checks_still_fail(f12f):
    x = f12f
    ch = _checks(fq.qualify(x["man"], POP, POP_SHA, x["out"], tok=x["tok"]))
    assert ch["gauge"]["pass"] is False and ch["simulation_matrix_inputs"]["pass"] is False


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda g: g["manifest"].update(status="priced_only"), "complete"),
        (lambda g: g["manifest"]["summary"].update(k_reproduced_exact_rows=269), "bitwise"),
        (lambda g: g["manifest"]["summary"].update(n_tokens=1), "token count"),
        (lambda g: g["data"]["X_gauge"].__setitem__((0, slice(None), 30), 9.0), "16-41"),
    ],
)
def test_gauge_check_fails_closed(f12f, mutate, match):
    x = f12f
    gauge = copy.deepcopy(x["gauge"])
    mutate(gauge)
    ch = _checks(
        fq.qualify(x["man"], POP, POP_SHA, x["out"], tok=x["tok"], gauge=gauge, matrix=x["matrix"])
    )
    assert ch["gauge"]["pass"] is False
    assert any(match in p for p in ch["gauge"]["problems"]), ch["gauge"]["problems"]


def test_gauge_inputs_verify_the_gauge_files(f12f):
    x = f12f
    (fq.Path(x["gman"]["out_dir"]) / "rows.jsonl").write_text("tampered\n")
    g = fq.gauge_inputs(x["gman"], x["gsha"], x["tsha"])
    assert "error" in g and "rows.jsonl" in g["error"]
    ch = _checks(
        fq.qualify(x["man"], POP, POP_SHA, x["out"], tok=x["tok"], gauge=g, matrix=x["matrix"])
    )
    assert ch["gauge"]["pass"] is False


def test_matrix_check_fails_on_drift_from_the_frozen_tokens_or_the_manifest(f12f, tmp_path):
    x = f12f
    m = fs.load_matrix(x["mdoc"]["matrix"]["path"])
    m2 = dict(m, tokens=m["tokens"].copy())
    m2["tokens"][0, 3] += 1.0
    m2["tokens_gauge"] = m["tokens_gauge"].copy()
    m2["tokens_gauge"][0, 3] += 1.0
    doc = fm.write_outputs(m2, {"widths": fm.widths()}, tmp_path / "m2.npz", tmp_path / "m2.json")
    bad = fq.matrix_inputs(doc, "x")
    ch = _checks(
        fq.qualify(x["man"], POP, POP_SHA, x["out"], tok=x["tok"], gauge=x["gauge"], matrix=bad)
    )
    assert ch["simulation_matrix_inputs"]["pass"] is False
    assert any("frozen tokens" in p for p in ch["simulation_matrix_inputs"]["problems"])
    stale = copy.deepcopy(x["mdoc"])
    stale["matrix_sha256"] = "0" * 64
    ch = _checks(fq.qualify(x["man"], POP, POP_SHA, x["out"], tok=x["tok"], gauge=x["gauge"],
                            matrix=fq.matrix_inputs(stale, "y")))  # fmt: skip
    assert ch["simulation_matrix_inputs"]["pass"] is False
    assert any("matrix manifest" in p for p in ch["simulation_matrix_inputs"]["problems"])


def test_candidate_identities_pin_the_f12f_code_and_inputs():
    ident = fq.candidate_identities()
    for rel in ("scripts/f12_gauge_tokens.py", "scripts/f12_matrix.py", "src/wsp/features/pool.py",
                "src/wsp/features/extract.py"):  # fmt: skip
        assert ident["code"][rel] == _sha((fs.ROOT / rel).read_bytes())
    assert ident["schema"] == "f12f-candidate-identities-v1"
    assert ident["snapshot"]["sha256"] == fm.SNAPSHOT_SHA256
    for key in ("gauge_manifest", "matrix_manifest"):
        assert ident[key]["path"].startswith("docs/prereg/a5_manifests/f12/")
    assert fq.identity_drift(ident) == []
    stale = copy.deepcopy(ident)
    stale["code"]["scripts/f12_matrix.py"] = "0" * 64
    assert fq.identity_drift(stale) == ["scripts/f12_matrix.py"]


def test_omitted_energy_is_reported_unchanged(f12f):
    x = f12f
    rep = fq.qualify(
        x["man"], POP, POP_SHA, x["out"], tok=x["tok"], gauge=x["gauge"], matrix=x["matrix"]
    )
    prior = {"omitted_energy": copy.deepcopy(rep["omitted_energy"])}
    assert fq.omitted_energy_unchanged(rep, prior) is True
    prior["omitted_energy"]["rows_above_threshold"] = -1
    assert fq.omitted_energy_unchanged(rep, prior) is False
    assert np.isfinite(rep["omitted_energy"]["threshold"])


def test_c1_draft_cites_the_f12f_gauge_tokens_runner_audit_and_matrix():
    from scripts import f12_c1 as c1

    files = c1.build_draft(fs.ROOT)
    g = files["models.json"]["gauge_and_control_audits"]["F5_gauge_audit"]
    assert (
        g["satisfied_by_frozen_output"] is False
    )  # the frozen run itself still holds no gauge tokens
    f = g["f12f"]
    assert (
        f["gauge_tokens_manifest"]["path"]
        == "docs/prereg/a5_manifests/f12/gauge_tokens_manifest.json"
    )
    assert "gauge_audit" in f["runner"] and "1e-5 + 1e-4 |f|" in f["runner"]
    assert f["qualification"]["path"] == "docs/prereg/f12_manifests/f12f_qualification.json"
    mx = files["schemas.json"]["comparator_matrix"]
    assert mx["manifest"]["path"] == "docs/prereg/a5_manifests/f12/matrix_manifest.json"
    assert mx["builder"]["path"] == "scripts/f12_matrix.py" and mx["widths"]["statnn_total"] == 9376
    for obj in (f["gauge_tokens_manifest"], f["qualification"], mx["manifest"]):
        p = fs.ROOT / obj["path"]
        assert obj["sha256"] == (_sha(p.read_bytes()) if p.exists() else None)
