"""Board A confirm of v2o (docs/reviews/council/board-a-prereg-wme-v2o-2026-11-20.md), condition N7: the instance
bundle is built from the committed tree (git archive HEAD), never from a dirty or untracked working tree. CPU only."""

import hashlib
import json
import subprocess
import tarfile

import pytest

from wsp.wm.c1 import bundling as BD


def run(repo, *a):
    return subprocess.run(
        ["git", *a], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src/pkg").mkdir(parents=True)
    (r / "scripts").mkdir()
    (r / "data/panels").mkdir(parents=True)
    (r / ".gitignore").write_text("data/\n__pycache__/\n")
    (r / "src/pkg/mod.py").write_text("X = 1\n")
    (r / "scripts/setup.sh").write_text("echo setup\n")
    (r / "data/panels/open.jsonl").write_text('{"id": 1}\n')
    run(r, "init", "-q")
    run(r, "config", "user.email", "x@y")
    run(r, "config", "user.name", "x")
    run(r, "add", ".gitignore", "src/pkg/mod.py", "scripts/setup.sh")
    run(r, "commit", "-q", "-m", "c")
    return r


FILES = ["scripts/setup.sh", "data/panels/open.jsonl"]


def members(out):
    with tarfile.open(out) as tf:
        return {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}


def test_n7_bundle_is_git_archive_head_plus_hashed_data_inputs(repo, tmp_path):
    out = tmp_path / "b.tgz"
    rec = BD.build(out, root=repo, files=FILES)
    head = run(repo, "rev-parse", "HEAD")
    m = members(out)
    assert m["wmec1/CODE_SHA"].decode() == head == rec["code_sha"]
    assert m["wmec1/src/pkg/mod.py"] == b"X = 1\n"
    assert m["wmec1/scripts/setup.sh"] == b"echo setup\n"
    inputs = json.loads(m["wmec1/BUNDLE_INPUTS.json"])
    assert inputs["data/panels/open.jsonl"] == hashlib.sha256(b'{"id": 1}\n').hexdigest()
    assert rec["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert not any("__pycache__" in k for k in m)


def test_n7_bundle_refuses_a_dirty_or_untracked_source_tree(repo, tmp_path):
    (repo / "src/pkg/mod.py").write_text("X = 2\n")  # dirty tracked source
    with pytest.raises(SystemExit, match="dirty or untracked"):
        BD.build(tmp_path / "a.tgz", root=repo, files=FILES)
    run(repo, "checkout", "--", "src/pkg/mod.py")
    (repo / "src/pkg/new.py").write_text("Y = 1\n")  # untracked source
    with pytest.raises(SystemExit, match="dirty or untracked"):
        BD.build(tmp_path / "b.tgz", root=repo, files=FILES)
    (repo / "src/pkg/new.py").unlink()
    (repo / "scripts/setup.sh").write_text("echo changed\n")  # dirty tracked bundle file
    with pytest.raises(SystemExit, match="dirty or untracked"):
        BD.build(tmp_path / "c.tgz", root=repo, files=FILES)
    run(repo, "checkout", "--", "scripts/setup.sh")
    (repo / "src/pkg/__pycache__").mkdir()
    (repo / "src/pkg/__pycache__/mod.cpython-312.pyc").write_bytes(
        b"x"
    )  # ignored: never bundled, not refused
    assert BD.build(tmp_path / "d.tgz", root=repo, files=FILES)["code_sha"] == run(
        repo, "rev-parse", "HEAD"
    )
    assert not tmp_path.joinpath("a.tgz").exists()  # a refused build writes nothing


def test_n7_bundle_refuses_untracked_code_outside_data_and_sealed_paths(repo, tmp_path):
    (repo / "scripts/extra.py").write_text("print(1)\n")  # untracked, not a data input
    with pytest.raises(SystemExit, match="not tracked"):
        BD.build(tmp_path / "a.tgz", root=repo, files=[*FILES, "scripts/extra.py"])
    (repo / "data/panels/sealed_test.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="sealed"):
        BD.build(tmp_path / "b.tgz", root=repo, files=[*FILES, "data/panels/sealed_test.jsonl"])


def test_n7_ctl_bundle_uses_the_git_archive_builder():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts/wme_c1_ctl.py").read_text()
    body = src[
        src.index("def cmd_bundle(") : src.index("\ndef ", src.index("def cmd_bundle(") + 10)
    ]
    assert "BD.build(" in body and "copytree" not in body
