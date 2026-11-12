"""F12d: Board A F12c condition 6, residual provenance gaps: prerequisite identity binding.

The noise prerequisite binds the noise check's code_sha256; pilot_acceptance binds the pilot report and timing hashes,
token / payload / code identities, the recomputed all-in projection, the 190 CPU-h ceiling and chronology; the C1
prerequisite needs a complete committed MANIFEST_C1.sha256 with the required file set and a Board A verification record
in a later commit (temporary git repositories). Fabricated records only; no outcome, no data/evals, no real token."""

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import f12_c1 as c1
from scripts import f12_simulation as fs
from tests.test_f12c_governance import (
    IDS,
    PI,
    PI_SHA,
    S1,
    S1_SHA,
    S2,
    S2_SHA,
    TG,
    TG_SHA,
    _report,
    _stage1_records,
    _stage2_records,
    _write,
    rec,
)

NOISE_FINISHED = "2026-11-12T10:00:00Z"
PILOT_STARTED = "2026-11-12T10:05:00Z"
PILOT_FINISHED = "2026-11-12T11:00:00Z"
ACCEPTED = "2026-11-12T12:00:00Z"


# --------------------------------------------------------------------- C1 repository


def _git(root, *args):
    # never inherit GIT_DIR / GIT_INDEX_FILE (exported by commit hooks): they would redirect these commands from the
    # temporary repository to the hook's repository
    return subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
        check=True, capture_output=True, text=True, env=c1.git_env(),
    ).stdout.strip()  # fmt: skip


def c1_repo(tmp_path, *, code_status="C1 frozen", draft_file=None, verify=True):
    """A temporary repository holding a complete C1 freeze (11 files + MANIFEST_C1.sha256, one commit) and,
    in a later commit, a Board A review citing the manifest hash plus the verification record."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(fs.ROOT / "scripts/f12_simulation.py", root / "scripts/f12_simulation.py")
    man = root / c1.MANIFEST_DIR
    for name in c1.C1_FILES:
        p = man / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name == "code_hashes_c1.json":
            obj = {"schema": fs.C1_SCHEMA, "status": code_status, "files": dict(IDS["code_sha256"])}
        else:
            obj = {"schema": f"toy-{name}", "status": "C1 frozen"}
        if name == draft_file:
            obj["status"] = "draft skeleton; awaiting the token manifest"
        p.write_text(json.dumps(obj, sort_keys=True) + "\n")
    _git(root, "init", "-q")
    assert (
        Path(_git(root, "rev-parse", "--absolute-git-dir")).resolve() == (root / ".git").resolve()
    )
    c1.write_manifest(root)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "C1 freeze")
    if verify:
        add_board_a_record(root)
    return root


def add_board_a_record(root, *, manifest_sha=None, cite=True, verdict="C1 VERIFIED"):
    msha = manifest_sha or hashlib.sha256((root / c1.MANIFEST_PATH).read_bytes()).hexdigest()
    review = root / "docs/reviews/council/board-a-c1-2026-11-12.md"
    review.parent.mkdir(parents=True, exist_ok=True)
    body = f"# Board A C1 verification\n\n**Verdict: {verdict}.**\n"
    if cite:
        body += f"\nManifest `MANIFEST_C1.sha256` sha256 `{msha}`.\n"
    review.write_text(body)
    rec_obj = {
        "schema": c1.VERIFICATION_SCHEMA,
        "manifest_sha256": msha,
        "review": str(review.relative_to(root)),
        "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    }
    (root / c1.VERIFICATION_PATH).write_text(json.dumps(rec_obj, sort_keys=True) + "\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "Board A C1 verification")
    return rec_obj


def test_complete_committed_c1_with_board_a_record_passes(tmp_path):
    root = c1_repo(tmp_path)
    assert c1.verify_c1(root, code_identity=IDS["code_sha256"]) == []
    lines = (root / c1.MANIFEST_PATH).read_text().splitlines()
    assert [ln.split()[0] for ln in lines if ln and not ln.startswith("#")] == [
        f"H{i}" for i in range(12)
    ]
    assert c1.H0 == "e0619f6a29af18c80301b42b7ed2a43f12fdd43ec5e8e98462846140289b5301"


def _fails(root, **kw):
    return c1.verify_c1(root, code_identity=kw.pop("code", IDS["code_sha256"]))


def test_c1_rejects_absent_or_draft_manifests_and_files(tmp_path):
    root = c1_repo(tmp_path / "a", verify=False)
    assert any("Board A" in f for f in _fails(root))
    root = c1_repo(tmp_path / "b", code_status="draft, pending Board A C1 verification")
    assert any("draft" in f for f in _fails(root))
    root = c1_repo(tmp_path / "c", draft_file="population.json")
    assert any("population.json" in f and "awaiting" in f for f in _fails(root))
    empty = tmp_path / "d"
    empty.mkdir()
    _git(empty, "init", "-q")
    assert any("MANIFEST_C1.sha256" in f for f in _fails(empty))


def test_c1_rejects_incomplete_or_altered_manifest(tmp_path):
    root = c1_repo(tmp_path / "a")
    p = root / c1.MANIFEST_PATH
    lines = p.read_text().splitlines()
    p.write_text("\n".join(x for x in lines if " folds.json" not in x) + "\n")
    _git(root, "commit", "-qam", "drop a line")
    f = _fails(root)
    assert any("incomplete" in x or "folds.json" in x for x in f), f
    root = c1_repo(tmp_path / "b")
    (root / c1.MANIFEST_DIR / "models.json").write_text('{"status": "C1 frozen", "x": 1}\n')
    _git(root, "commit", "-qam", "rewrite a C1 file")
    assert any("models.json" in x for x in _fails(root))
    root = c1_repo(tmp_path / "c")
    (root / c1.MANIFEST_DIR / "schemas.json").write_text('{"status": "C1 frozen", "y": 2}\n')
    assert any("schemas.json" in x for x in _fails(root))  # uncommitted change


def test_c1_rejects_stale_code_and_uncovered_code_identity(tmp_path):
    root = c1_repo(tmp_path)
    with open(root / "scripts/f12_simulation.py", "a") as fh:
        fh.write("\n# changed after C1\n")
    assert any("current bytes" in x for x in _fails(root))
    root = c1_repo(tmp_path / "b")
    other = {"scripts/f12_simulation.py": "f" * 64}
    assert any("code identity" in x for x in _fails(root, code=other))


def test_c1_self_asserted_verification_never_passes(tmp_path):
    root = c1_repo(tmp_path / "a", verify=False)
    ch = root / c1.MANIFEST_DIR / "code_hashes_c1.json"
    obj = json.loads(ch.read_text())
    obj.update(status="verified by Board A", board_a_verified=True)
    ch.write_text(json.dumps(obj))
    (root / c1.MANIFEST_DIR / "board_a_verified.flag").write_text("yes")
    assert any("Board A" in x for x in _fails(root))
    root = c1_repo(tmp_path / "b", verify=False)
    add_board_a_record(root, cite=False)  # review does not cite the manifest hash
    assert any("cite" in x for x in _fails(root))
    root = c1_repo(tmp_path / "c", verify=False)
    add_board_a_record(root, verdict="REJECT")
    assert any("C1 VERIFIED" in x for x in _fails(root))
    root = c1_repo(tmp_path / "d", verify=False)
    add_board_a_record(root, manifest_sha="0" * 64)
    assert any("manifest" in x for x in _fails(root))


def test_c1_verification_must_follow_the_manifest_commit(tmp_path):
    root = c1_repo(tmp_path, verify=False)
    # record and review committed in the same commit as a rewritten manifest: no chronology
    add_board_a_record(root)
    p = root / c1.MANIFEST_PATH
    p.write_text(p.read_text() + "# touched\n")
    _git(root, "commit", "-qam", "manifest touched after the review")
    f = _fails(root)
    assert any("manifest" in x for x in f), f


# ------------------------------------------------------------------ pilot acceptance


def pilot_bundle(tmp_path, ids=IDS, alt_cpu=219.0, tree_cpu=2250.0):
    recs = [rec(PI, s, k, r) for s, k, r in fs.planned_ids(PI, [145])]
    pr = _write(tmp_path / "pilot_report.json", _report(PI, PI_SHA, recs, [145], ids=ids))
    timing = {
        "schema": fs.TIMING_SCHEMA,
        "report_sha256": hashlib.sha256(pr.read_bytes()).hexdigest(),
        "started_utc": PILOT_STARTED,
        "finished_utc": PILOT_FINISHED,
        "workers": 2,
        "cpu_seconds_total_process_tree": tree_cpu,
        "cpu_seconds_overhead": 10.0,
        "per_replicate_cpu_by_k_and_scenario": {"145": {"alt": {"n": 10, "cpu_mean": alt_cpu}}},
    }
    pt = _write(tmp_path / "pilot_timing.json", timing)
    nc = {"schema": fs.NOISE_CHECK_SCHEMA, "status": "pass", "f12a4d_check": True, "panels": 10000,
          "k": 145, "cpu_seconds": 14.0, "finished_utc": NOISE_FINISHED, "code_sha256": ids["code_sha256"],
          "identities": {k: ids[k] for k in ("tokens_matrix_sha256", "tokens_synthetic",
                                             "payload_restricted_sha256")}}  # fmt: skip
    ncp = _write(tmp_path / "noise_check.json", nc)
    return pr, pt, ncp


def test_all_in_projection_from_an_alternative_only_pilot():
    timing = {"per_replicate_cpu_by_k_and_scenario": {"145": {"alt": {"n": 10, "cpu_mean": 219.0}}},
              "cpu_seconds_total_process_tree": 2250.0, "cpu_seconds_overhead": 10.0}  # fmt: skip
    pr = fs.all_in_projection(timing, {"cpu_seconds": 14.0})
    want = (480 * 219 + 2200 * 219 + 200 * 219 + 2250 + 14 + fs.PLANNED_INVOCATIONS * 10) / 3600
    assert pr["all_in_cpu_h"] == pytest.approx(want)
    assert pr["stress_timing"].startswith("extrapolated")
    assert "not measured stress timing" in pr["label"]


def test_pilot_acceptance_record_binds_hashes_identities_projection_ceiling_and_time(tmp_path):
    pr, pt, ncp = pilot_bundle(tmp_path)
    pa = fs.make_pilot_acceptance(pilot_report=pr, pilot_timing=pt, noise_check=ncp, signed_by="chair",
                                  accepted_utc=ACCEPTED)  # fmt: skip
    assert pa["schema"] == fs.PILOT_ACCEPTANCE_SCHEMA and pa["chair_accepted"] is True
    assert pa["all_in_ceiling_cpu_h"] == fs.CHAIR_CEILING_CPU_H == 190.0
    assert pa["pilot_report_sha256"] == hashlib.sha256(pr.read_bytes()).hexdigest()
    assert pa["pilot_timing_sha256"] == hashlib.sha256(pt.read_bytes()).hexdigest()
    assert pa["identities"] == json.loads(json.dumps(IDS))
    assert pa["projection"]["all_in_cpu_h"] <= 190.0
    with pytest.raises(ValueError, match="before"):
        fs.make_pilot_acceptance(pilot_report=pr, pilot_timing=pt, noise_check=ncp, signed_by="chair",
                                 accepted_utc="2026-11-12T10:30:00Z")  # fmt: skip
    (tmp_path / "over").mkdir()
    pr2, pt2, nc2 = pilot_bundle(tmp_path / "over", alt_cpu=260.0)
    with pytest.raises(ValueError, match="190"):
        fs.make_pilot_acceptance(pilot_report=pr2, pilot_timing=pt2, noise_check=nc2, signed_by="chair",
                                 accepted_utc=ACCEPTED)  # fmt: skip


# ------------------------------------------------------------- the execution validator


def governed(tmp_path, k=58):
    """A complete governed stage-2 bundle: report, selection, prereqs, their file hashes and a C1 root."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    s1 = _write(tmp_path / "s1.json", _report(S1, S1_SHA, _stage1_records(), list(fs.K_GRID)))
    sel = fs.verify_stage1_report(s1, identities=IDS)
    tg_recs = [rec(TG, s, kk, r, A={"F3": 1.0, "WM": 0.98, "S": 0.99, "oracle": 0.9})
               for s, kk, r in fs.planned_ids(TG, [k])]  # fmt: skip
    target_report = _report(TG, TG_SHA, tg_recs, [k], selection=sel)
    target = fs.expected_row_sum_target(tg_recs, fs.planned_ids(TG, [k]), TG)
    report = _report(S2, S2_SHA, _stage2_records(k), [k], selection=sel, target=target)
    pr, pt, ncp = pilot_bundle(tmp_path)
    pa = fs.make_pilot_acceptance(pilot_report=pr, pilot_timing=pt, noise_check=ncp, signed_by="chair",
                                  accepted_utc=ACCEPTED)  # fmt: skip
    pap = _write(tmp_path / "pilot_acceptance.json", pa)
    tgp = _write(
        tmp_path / "target.json", json.loads(json.dumps(target_report, default=fs._json_default))
    )
    prereqs, shas = fs.load_prereqs({"noise_check": ncp, "pilot_report": pr, "pilot_timing": pt,
                                     "pilot_acceptance": pap, "coverage_target": tgp})  # fmt: skip
    root = c1_repo(tmp_path / "c1")
    kw = {"prereq_sha256": shas, "c1_root": root}
    return json.loads(json.dumps(report, default=fs._json_default)), sel, prereqs, kw


def test_validator_passes_with_bound_prerequisites(tmp_path):
    report, sel, prereqs, kw = governed(tmp_path)
    v = fs.validate_execution(report, selection=sel, prereqs=prereqs, **kw)
    assert v["passed"], v["failures"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        # noise check bound to the current code
        (lambda p, s: p["noise_check"].update(code_sha256={"scripts/f12_simulation.py": "e" * 64}), "noise_check: code"),
        (lambda p, s: p["noise_check"].pop("code_sha256"), "noise_check: code"),
        # pilot acceptance: superseded schema, bare chair flag
        (lambda p, s: p.update(pilot_acceptance={"schema": "f12c-pilot-acceptance-v1", "chair_accepted": True,
                                                 "all_in_ceiling_cpu_h": 190}), "pilot_acceptance"),
        (lambda p, s: p["pilot_acceptance"].update(pilot_report_sha256="0" * 64), "pilot report"),
        (lambda p, s: s.update(pilot_report="1" * 64), "pilot report"),
        (lambda p, s: s.update(pilot_timing="1" * 64), "pilot timing"),
        (lambda p, s: p["pilot_acceptance"].update(all_in_ceiling_cpu_h=200.0), "190"),
        (lambda p, s: p["pilot_acceptance"]["projection"].update(all_in_cpu_h=150.0), "projection"),
        (lambda p, s: p["pilot_timing"]["per_replicate_cpu_by_k_and_scenario"]["145"]["alt"].update(cpu_mean=260.0),
         "pilot"),
        (lambda p, s: p["pilot_acceptance"].update(accepted_utc="2026-11-12T10:30:00Z"), "chronology"),
        (lambda p, s: p["noise_check"].update(finished_utc="2026-11-12T10:10:00Z"), "chronology"),
        (lambda p, s: p["pilot_acceptance"]["identities"].update(tokens_matrix_sha256="b" * 64), "identit"),
        (lambda p, s: p["pilot_report"]["identities"].update(tokens_matrix_sha256="b" * 64), "identit"),
        (lambda p, s: p["pilot_report"]["records"].pop(), "pilot report"),
        (lambda p, s: p["pilot_report"]["config_file"].update(sha256="0" * 64), "pilot report"),
        (lambda p, s: p.pop("pilot_report"), "pilot report"),
        (lambda p, s: p["pilot_acceptance"].update(signed_by=""), "signed"),
    ],
)  # fmt: skip
def test_validator_rejects_unbound_prerequisites(tmp_path, mutate, match):
    report, sel, prereqs, kw = governed(tmp_path)
    mutate(prereqs, kw["prereq_sha256"])
    v = fs.validate_execution(report, selection=sel, prereqs=prereqs, **kw)
    assert not v["passed"] and any(match in f for f in v["failures"]), v["failures"]
    assert fs.release_decision(report["summary"], S2, v)["eligible"] is False


def test_validator_rejects_an_unverified_or_stale_c1(tmp_path):
    report, sel, prereqs, kw = governed(tmp_path)
    for root in (c1_repo(tmp_path / "x", verify=False),
                 c1_repo(tmp_path / "y", code_status="draft, pending Board A C1 verification")):  # fmt: skip
        v = fs.validate_execution(report, selection=sel, prereqs=prereqs, **{**kw, "c1_root": root})
        assert not v["passed"] and any(f.startswith("C1") for f in v["failures"]), v["failures"]
    # the real repository's C1 is a draft skeleton: never passes
    v = fs.validate_execution(report, selection=sel, prereqs=prereqs, **{**kw, "c1_root": fs.ROOT})
    assert not v["passed"] and any(f.startswith("C1") for f in v["failures"])


def test_prereq_files_are_hashed_as_committed_bytes(tmp_path):
    p = _write(tmp_path / "x.json", {"a": 1})
    objs, shas = fs.load_prereqs({"noise_check": p})
    assert shas["noise_check"] == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="KEY"):
        fs._parse_prereqs(["c1_code_hashes=" + str(p)])
    assert copy.deepcopy(objs)["noise_check"] == {"a": 1}


# ------------------------------------------------------------------ C1 package skeleton


def test_c1_draft_skeleton_marks_token_dependent_parts_and_never_verifies(tmp_path):
    # without token facts (F12e fills them; tests/test_f12e_qualify.py checks the committed filled draft)
    files = c1.build_draft(fs.ROOT, use_facts=False)
    assert list(files) == list(c1.C1_FILES)
    for name, obj in files.items():
        assert "draft" in obj["status"] and "not a C1 file" in obj["status"], name
    text = json.dumps(files)
    assert text.count(c1.AWAITING) >= 4
    for name in ("population.json", "folds.json", "schemas.json"):
        assert c1.AWAITING in json.dumps(files[name]), name
    assert (
        files["noise_payload/restricted.json"]["pin_sha256"]
        == hashlib.sha256(fs.RESTRICTED_PIN.read_bytes()).hexdigest()
    )
    assert files["resources.json"]["ceiling_cpu_h"] == 190.0
    assert c1.verify_c1(fs.ROOT, code_identity=fs.current_code_identity()) != []


def test_temporary_repositories_ignore_an_inherited_hook_environment(tmp_path, monkeypatch):
    """Regression (F12d incident): inside a commit hook GIT_DIR / GIT_INDEX_FILE point at the real repository; the
    fixtures and the verifier must still address only the temporary repository."""
    victim = tmp_path / "victim"
    victim.mkdir()
    _git(victim, "init", "-q")
    (victim / "f.txt").write_text("x\n")
    _git(victim, "add", "f.txt")
    _git(victim, "commit", "-qm", "victim")
    head = _git(victim, "rev-parse", "HEAD")
    cfg = (victim / ".git/config").read_text()
    monkeypatch.setenv("GIT_DIR", str(victim / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(victim / ".git/index"))
    root = c1_repo(tmp_path / "c1")
    assert c1.verify_c1(root, code_identity=IDS["code_sha256"]) == []
    monkeypatch.delenv("GIT_DIR")
    monkeypatch.delenv("GIT_INDEX_FILE")
    assert _git(victim, "rev-parse", "HEAD") == head
    assert (victim / ".git/config").read_text() == cfg
    assert _git(victim, "status", "--porcelain") == ""
