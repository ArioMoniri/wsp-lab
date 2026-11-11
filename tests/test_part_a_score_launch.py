"""scripts/part_a_score_launch.sh, tested at the bash level (PRESCORE_INVENTORY §10, LS-1 to LS-4).

The real script is copied into a temporary git repository with stubs for the parts these tests are not
about: ``.venv/bin/python`` execs this interpreter; ``scripts/a5_build_manifests.py`` prints a controllable
pin verdict; ``src/wsp/eval/response_intervention.py`` logs its argv, prints a price and, only when asked to
score for real, writes ``data/evals/scored`` as the GPU scorer would; ``nvidia-smi`` prints a fixed
identity. The staged bytes and the manifest are real files hashed by the script itself. The behaviour of
``score --proofs-only`` / ``--require-proofs`` / ``--max-minutes`` is tested in test_response_intervention.

Each refusal must exit non-zero before any scoring call, write nothing under ``data/evals`` and write no
launch record.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "part_a_score_launch.sh"
MANIFEST = "docs/prereg/a5_manifests/part_a/manifest.json"
RECORD = "docs/prereg/a5_manifests/part_a/score_launch.json"
STAGE = "data/runs/a5/part_a/qwen3.5-0.8b"
GPU = "NVIDIA TEST GPU, 1.0, 1 MiB, GPU-0000-test"

STUB_RI = r"""
import json, os, sys
argv = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps(argv) + "\n")
if "--proofs-only" in argv:
    print("PRICE entries=166 minutes=172.64 (variants 147, references 19, failed proofs 0)")
    sys.exit(int(os.environ.get("STUB_PROOFS_RC", "0")))
if "--dry-run" in argv:
    sys.exit(int(os.environ.get("STUB_GATE_RC", "0")))
os.makedirs("data/evals/scored", exist_ok=True)
sys.exit(int(os.environ.get("STUB_SCORE_RC", "0")))
"""

STUB_PINS = r"""
import os, sys
print(os.environ.get("STUB_PINS_OUT", "all pins match"))
sys.exit(int(os.environ.get("STUB_PINS_RC", "0")))
"""


def _w(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data)
    return path


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _clean_env():
    """os.environ without STUB_* and without every GIT_* variable. Git exports GIT_DIR, GIT_INDEX_FILE and
    friends to hooks; inherited by these tests (the pre-commit hook runs pytest), they would point the
    temporary repository's git commands at the real one."""
    return {k: v for k, v in os.environ.items() if not k.startswith(("STUB_", "GIT_"))}


def _git(repo, *args):
    env = _clean_env() | {"GIT_CEILING_DIRECTORIES": str(Path(repo).parent)}
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    ).stdout


def _assert_relocatable(text):
    """The copy under test must act on the temporary repository only: a script that changes to a fixed
    checkout would run its checks, proof step and GPU scoring against the real one."""
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        assert "/home/" not in code and not code.strip().startswith("cd /"), line


@pytest.fixture
def launch(tmp_path):
    return _make_launch(tmp_path)


def _make_launch(tmp_path):
    _assert_relocatable(SCRIPT.read_text())
    repo = tmp_path / "repo"
    _w(repo / "scripts" / "part_a_score_launch.sh", SCRIPT.read_text()).chmod(0o755)
    _w(repo / "scripts" / "a5_build_manifests.py", STUB_PINS)
    _w(repo / ".venv" / "bin" / "python", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n').chmod(0o755)
    _w(repo / "src" / "wsp" / "__init__.py", "")
    _w(repo / "src" / "wsp" / "eval" / "__init__.py", "")
    _w(repo / "src" / "wsp" / "eval" / "response_intervention.py", STUB_RI)
    _w(repo / ".gitignore", "data/\n.venv/\n__pycache__/\n")
    files = {
        ("v1", "delta.safetensors"): b"delta-1",
        ("v2", "delta.safetensors"): b"delta-2",
        ("v2", "increment.safetensors"): b"increment-2",
        ("v1", "variant.json"): b'{"id": "v1"}',
        ("v2", "variant.json"): b'{"id": "v2"}',
        ("x9", "variant.json"): b'{"id": "x9", "excluded": true}',
    }
    for (vid, fn), b in files.items():
        _w(repo / STAGE / vid / fn, b)
    man = {
        "staged_zoo": STAGE,
        "delta_sha256": {
            "v1": {"delta.safetensors": _sha(files["v1", "delta.safetensors"])},
            "v2": {
                "delta.safetensors": _sha(files["v2", "delta.safetensors"]),
                "increment.safetensors": _sha(files["v2", "increment.safetensors"]),
            },
        },
        "variant_json_sha256": {v: _sha(files[v, "variant.json"]) for v in ("v1", "v2", "x9")},
    }
    mbytes = json.dumps(man, indent=1).encode()
    _w(repo / MANIFEST, mbytes)
    bin_dir = tmp_path / "bin"
    _w(bin_dir / "nvidia-smi", f'#!/bin/sh\necho "{GPU}"\n').chmod(0o755)
    _git(repo, "init", "-q")
    top = _git(repo, "rev-parse", "--show-toplevel").strip()
    assert Path(top).resolve() == repo.resolve(), top  # never touch any other repository
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "toy")
    log = tmp_path / "calls.jsonl"
    env = _clean_env() | {"GIT_CEILING_DIRECTORIES": str(tmp_path)}
    env |= {"PATH": f"{bin_dir}:{env['PATH']}", "STUB_LOG": str(log)}

    def run(*args, **stub):
        r = subprocess.run(
            ["bash", str(repo / "scripts" / "part_a_score_launch.sh"), *args],
            check=False,
            cwd=tmp_path,  # the script must find its own repository root
            env=env | {f"STUB_{k.upper()}": str(v) for k, v in stub.items()},
            capture_output=True,
            text=True,
            timeout=60,
        )
        calls = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        log.unlink(missing_ok=True)
        return r, calls

    return {"repo": repo, "run": run, "sha": _sha(mbytes), "bin": bin_dir}


def _refused(launch, r, calls, code, text):
    out = r.stdout + r.stderr
    assert r.returncode == code, out
    assert text in out, out
    assert calls == []  # no proof step, no gate, no scoring
    assert not (launch["repo"] / "data" / "evals").exists()
    assert not (launch["repo"] / RECORD).exists()


def test_the_fixture_ignores_git_variables_inherited_from_a_hook(tmp_path, monkeypatch):
    sentinel = tmp_path / "sentinel"
    _git(sentinel.parent, "init", "-q", str(sentinel))
    _w(sentinel / "f.txt", "x")
    _git(sentinel, "add", "f.txt")
    _git(sentinel, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "sentinel")
    before = (_git(sentinel, "rev-parse", "HEAD"), (sentinel / ".git" / "config").read_bytes(),
              (sentinel / ".git" / "index").read_bytes())  # fmt: skip
    monkeypatch.setenv("GIT_DIR", str(sentinel / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(sentinel / ".git" / "index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(sentinel))
    launch = _make_launch(tmp_path / "t")
    r, calls = launch["run"](launch["sha"])
    assert r.returncode == 0 and len(calls) == 1, r.stdout + r.stderr
    monkeypatch.delenv("GIT_DIR")
    monkeypatch.delenv("GIT_INDEX_FILE")
    monkeypatch.delenv("GIT_WORK_TREE")
    after = (_git(sentinel, "rev-parse", "HEAD"), (sentinel / ".git" / "config").read_bytes(),
             (sentinel / ".git" / "index").read_bytes())  # fmt: skip
    assert after == before


def test_the_relocatability_guard_rejects_a_fixed_checkout():
    with pytest.raises(AssertionError):
        _assert_relocatable("set -u\ncd /home/ubuntu/projects/wsp-lab || exit 1\n")


def test_phase_1_checks_then_runs_the_proofs_only_step_and_stops(launch):
    r, calls = launch["run"](launch["sha"])
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "all pins match" in out
    assert "rehashed 6 files (3 delta/increment, 3 variant.json), 0 missing, 0 mismatches" in out
    assert "PRICE entries=166 minutes=172.64" in out
    assert "STOP" in out and "--phase score" in out
    assert len(calls) == 1 and calls[0][0] == "score" and "--proofs-only" in calls[0]
    assert calls[0][calls[0].index("--manifest") + 1] == MANIFEST
    assert calls[0][calls[0].index("--stage") + 1] == STAGE
    assert not (launch["repo"] / "data" / "evals").exists()
    assert not (launch["repo"] / RECORD).exists()


def test_phase_2_rechecks_gates_the_price_then_scores_with_a_launch_record(launch):
    r, calls = launch["run"](launch["sha"], "--phase", "score", "--max-minutes", "180")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "all pins match" in out and "0 missing, 0 mismatches" in out
    assert "SCORE_EXIT 0" in out
    assert [c[0] for c in calls] == ["score", "score"]
    gate, real = calls
    assert "--dry-run" in gate and "--dry-run" not in real
    for c in calls:
        assert "--proofs-only" not in c and "--require-proofs" in c
        assert c[c.index("--max-minutes") + 1] == "180"
    assert (launch["repo"] / "data" / "evals" / "scored").is_dir()  # the (stub) scorer ran
    rec = json.loads((launch["repo"] / RECORD).read_text())
    assert rec["schema"] == "a5-part-a-score-launch-v2"
    assert rec["gpu"] == GPU and rec["gpu_uuid"] == "GPU-0000-test"
    assert rec["manifest_sha256"] == launch["sha"] and rec["max_minutes"] == 180.0
    assert rec["dirty_files"] == 0 and rec["exit_code"] == 0
    assert rec["start_utc"] <= rec["end_utc"]
    assert len(rec["execution_sha"]) == 40
    assert "--max-minutes 180" in rec["command"]


def test_phase_2_records_a_failed_scoring_exit(launch):
    r, _ = launch["run"](launch["sha"], "--phase", "score", "--max-minutes", "180", score_rc=3)
    assert r.returncode == 3 and "SCORE_EXIT 3" in r.stdout
    assert json.loads((launch["repo"] / RECORD).read_text())["exit_code"] == 3


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["nothex"],
        ["--phase", "score"],
        ["SHA", "--phase", "score"],  # no threshold
        ["SHA", "--phase", "score", "--max-minutes", "lots"],
        ["SHA", "--max-minutes", "180"],  # threshold belongs to phase 2
        ["SHA", "--phase", "gpu", "--max-minutes", "180"],
        ["SHA", "--phase"],
        ["SHA", "extra"],
    ],
)
def test_bad_arguments_are_refused(launch, args):
    args = [launch["sha"] if a == "SHA" else a for a in args]
    r, calls = launch["run"](*args)
    _refused(launch, r, calls, 64, "usage")


PHASES = [[], ["--phase", "score", "--max-minutes", "180"]]


@pytest.mark.parametrize("phase", PHASES)
def test_a_dirty_tree_is_refused(launch, phase):
    _w(launch["repo"] / "stray.txt", "x")
    _refused(launch, *launch["run"](launch["sha"], *phase), 2, "TREE NOT CLEAN")


PIN_FAILURES = [
    {"pins_out": "differs: src/x.py", "pins_rc": 1},
    {"pins_out": "differs: a", "pins_rc": 0},
    {"pins_out": "all pins match", "pins_rc": 1},
]


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("stub", PIN_FAILURES)
def test_a_pin_mismatch_is_refused(launch, phase, stub):
    _refused(launch, *launch["run"](launch["sha"], *phase, **stub), 3, "PIN CHECK FAILED")


@pytest.mark.parametrize("phase", PHASES)
def test_another_manifest_hash_is_refused(launch, phase):
    other = "0" * 64
    _refused(launch, *launch["run"](other, *phase), 4, "MANIFEST HASH")


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize(
    "damage",
    [
        "missing_delta",
        "missing_increment",
        "missing_variant_json",
        "flipped_byte",
        "flipped_vjson",
        "missing_stage",
    ],
)
def test_the_strict_rehash_refuses_missing_or_changed_files(launch, phase, damage):
    stage = launch["repo"] / STAGE
    {
        "missing_delta": lambda: (stage / "v1" / "delta.safetensors").unlink(),
        "missing_increment": lambda: (stage / "v2" / "increment.safetensors").unlink(),
        "missing_variant_json": lambda: (stage / "x9" / "variant.json").unlink(),
        "flipped_byte": lambda: (stage / "v2" / "delta.safetensors").write_bytes(b"delta-3"),
        "flipped_vjson": lambda: (stage / "v1" / "variant.json").write_bytes(b"{}"),
        "missing_stage": lambda: shutil.rmtree(stage),
    }[damage]()
    _refused(launch, *launch["run"](launch["sha"], *phase), 5, "STAGED HASH RECHECK FAILED")


def test_a_failed_proof_step_stops_phase_1(launch):
    r, calls = launch["run"](launch["sha"], proofs_rc=1)
    assert r.returncode == 6 and "PROOF STEP FAILED" in r.stdout + r.stderr
    assert len(calls) == 1 and "--proofs-only" in calls[0]
    assert not (launch["repo"] / "data" / "evals").exists()
    assert not (launch["repo"] / RECORD).exists()


def test_phase_2_refuses_when_the_proof_and_price_gate_fails(launch):
    r, calls = launch["run"](launch["sha"], "--phase", "score", "--max-minutes", "100", gate_rc=1)
    assert r.returncode == 7 and "PROOF/PRICE GATE" in r.stdout + r.stderr
    assert len(calls) == 1 and "--dry-run" in calls[0]  # the gate only; no scoring call
    assert not (launch["repo"] / "data" / "evals").exists()
    assert not (launch["repo"] / RECORD).exists()


def test_phase_2_refuses_an_existing_launch_record(launch):
    _w(launch["repo"] / RECORD, "{}")
    _git(launch["repo"], "add", RECORD)
    _git(launch["repo"], "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "rec")
    r, calls = launch["run"](launch["sha"], "--phase", "score", "--max-minutes", "180")
    assert r.returncode == 8 and "LAUNCH RECORD EXISTS" in r.stdout + r.stderr
    assert calls == [] and not (launch["repo"] / "data" / "evals").exists()
    assert (launch["repo"] / RECORD).read_text() == "{}"


def test_phase_2_refuses_without_a_gpu_identity(launch):
    _w(launch["bin"] / "nvidia-smi", "#!/bin/sh\nexit 9\n").chmod(0o755)
    r, calls = launch["run"](launch["sha"], "--phase", "score", "--max-minutes", "180")
    assert r.returncode == 9 and "NO GPU IDENTITY" in r.stdout + r.stderr
    assert len(calls) == 1 and "--dry-run" in calls[0]
    assert not (launch["repo"] / "data" / "evals").exists()
    assert not (launch["repo"] / RECORD).exists()
