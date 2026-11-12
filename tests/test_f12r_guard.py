"""F12-R inputs guard (scripts/f12_inputs_guard.py; unexposed adoption A10, A19, A21): read boundary resolved to the
leaf, refusal of data/evals and the other prohibited paths, pinned-hash authentication, atomic checkpoints, and the
resource authority's caps at launch. Synthetic paths only; nothing under data/ is opened or listed."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import f12_inputs_guard as g

# ------------------------------------------------------------------------------------- read boundary


@pytest.fixture
def tree(tmp_path):
    """allowed/ (inputs), evals/ (a stand-in prohibited root) and a few symlinks between them."""
    allowed = tmp_path / "allowed"
    evals = tmp_path / "evals"
    allowed.mkdir()
    evals.mkdir()
    (allowed / "a.json").write_bytes(b'{"x": 1}\n')
    (evals / "secret.json").write_bytes(b'{"score": 0.5}\n')
    (allowed / "link_to_evals.json").symlink_to(evals / "secret.json")
    (allowed / "dir_link").symlink_to(evals)
    (tmp_path / "outside.json").write_bytes(b"{}\n")
    b = g.ReadBoundary(allowed=[allowed], prohibited=[evals])
    return {"tmp": tmp_path, "allowed": allowed, "evals": evals, "b": b}


def test_boundary_reads_and_records_allowed(tree):
    b = tree["b"]
    data = b.read_bytes(tree["allowed"] / "a.json")
    assert data == b'{"x": 1}\n'
    rec = b.record()
    assert rec["reads"][0]["sha256"] == g.sha256_bytes(data)
    assert rec["reads"][0]["resolved"] == str((tree["allowed"] / "a.json").resolve())
    assert rec["allowed"] == [str(tree["allowed"].resolve())]
    assert str(tree["evals"].resolve()) in rec["prohibited"]


@pytest.mark.parametrize(
    "rel",
    ["link_to_evals.json", "dir_link/secret.json", "../evals/secret.json"],
)
def test_boundary_refuses_symlinks_and_dotdot_into_prohibited(tree, rel):
    with pytest.raises(g.BoundaryError, match="prohibited"):
        tree["b"].read_bytes(tree["allowed"] / rel)
    assert tree["b"].record()["reads"] == []


def test_boundary_refuses_paths_outside_allowed(tree):
    with pytest.raises(g.BoundaryError, match="outside the allowed"):
        tree["b"].read_bytes(tree["tmp"] / "outside.json")


def test_boundary_refuses_before_opening(tree, monkeypatch):
    opened = []
    real = os.open

    def spy(path, *a, **k):
        opened.append(str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(g.os, "open", spy)
    with pytest.raises(g.BoundaryError):
        tree["b"].read_bytes(tree["allowed"] / "link_to_evals.json")
    assert not any("secret" in p for p in opened)


def test_boundary_authenticates_hash(tree):
    b = tree["b"]
    p = tree["allowed"] / "a.json"
    b.read_bytes(p, sha256=g.sha256_bytes(p.read_bytes()))
    with pytest.raises(g.AuthenticationError, match="sha256"):
        b.read_bytes(p, sha256="0" * 64)
    assert b.record()["reads"][0]["authenticated"] is True


def test_default_prohibited_covers_data_evals_and_part_a():
    roots = {str(p) for p in g.prohibited_roots()}
    assert str(g.ROOT / "data/evals") in roots
    assert str(g.DATA_ROOT / "evals") in roots
    assert str(g.ROOT / "docs/prereg/a5_manifests/part_a") in roots
    assert str(g.ROOT / "docs/prereg/f12_manifests/f12f_pilot_report.json") in roots
    b = g.ReadBoundary(allowed=[g.DATA_ROOT])
    for bad in (
        g.DATA_ROOT / "evals/x/eval.json",
        g.DATA_ROOT / "runs/a5/part_a_predictions/rows.jsonl",
        g.DATA_ROOT / "runs/a5/part_a_stage/v/variant.json",
        g.DATA_ROOT / "runs/a5/sealed/ledger.json",
    ):
        with pytest.raises(g.BoundaryError, match="prohibited"):
            b.resolve(bad)  # refused lexically: nothing is stat'ed or opened


def test_entry_delta_hash_matches_a5_convention(tmp_path):
    from scripts import a5_build_manifests as bm

    d = tmp_path / "e"
    (d / "lora").mkdir(parents=True)
    (d / "lora" / "adapter_model.safetensors").write_bytes(b"abc")
    (d / "lora" / "adapter_config.json").write_bytes(b'{"r": 2}')
    (d / "edit.json").write_bytes(b"{}")
    b = g.ReadBoundary(allowed=[tmp_path], prohibited=[])
    assert g.entry_delta_hash(b, d) == bm.entry_delta_hash(bm.artefact_files(d))
    g.authenticate_entry(b, d, bm.entry_delta_hash(bm.artefact_files(d)))
    with pytest.raises(g.AuthenticationError):
        g.authenticate_entry(b, d, "f" * 64)
    dense = tmp_path / "dense"
    dense.mkdir()
    (dense / "delta.safetensors").write_bytes(b"xyz")
    assert g.entry_delta_hash(b, dense) == bm.entry_delta_hash(bm.artefact_files(dense))
    with pytest.raises(g.AuthenticationError, match="no stored bytes"):
        g.authenticate_entry(b, tmp_path / "missing", "0" * 64)


def test_entry_delta_hash_refuses_symlinked_adapter(tmp_path):
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "x").write_bytes(b"secret")
    d = tmp_path / "zoo" / "e"
    (d / "lora").mkdir(parents=True)
    (d / "lora" / "adapter_model.safetensors").symlink_to(evals / "x")
    b = g.ReadBoundary(allowed=[tmp_path / "zoo"], prohibited=[evals])
    with pytest.raises(g.BoundaryError):
        g.entry_delta_hash(b, d)


def test_safe_name():
    assert g.safe_name("000.safetensors") == "000.safetensors"
    for bad in ("../x", "a/b", "/etc/passwd", "", ".", ".."):
        with pytest.raises(g.BoundaryError):
            g.safe_name(bad)


# ------------------------------------------------------------------------------ atomic checkpoints


def test_atomic_write_and_prefix_hash(tmp_path):
    p = tmp_path / "c.json"
    g.atomic_write_bytes(p, b"one")
    g.atomic_write_bytes(p, b"two")
    assert p.read_bytes() == b"two"
    assert not [x for x in tmp_path.iterdir() if x.name != "c.json"]
    q = tmp_path / "log"
    q.write_bytes(b"abcdef")
    assert g.prefix_sha256(q, 3) == g.sha256_bytes(b"abc")
    with pytest.raises(ValueError):
        g.prefix_sha256(q, 7)


def test_strict_json_rejects_nonfinite():
    assert g.strict_json_loads('{"a": 1.5}') == {"a": 1.5}
    for bad in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
        with pytest.raises(ValueError, match="nonfinite"):
            g.strict_json_loads(bad)


# ------------------------------------------------------------------------------ resource authority


RULING = (
    "#### A5 chair ruling on the F12-R token rerun (13 Nov 2026, 09:00 UTC, before any G3 outcome)"
)


def _prereg(line):
    return f"intro\n\n{RULING}\n\nText.\n`{line}`\n\n#### A5 note next\nresource-cap: build=x cap_bytes=1 threads=1\n"


def test_ruling_verified_from_its_own_block():
    text = _prereg("resource-cap: build=scripts/f12_tokens.py cap_bytes=18000000000 threads=6")
    rec = g.verify_ruling(text, RULING, "scripts/f12_tokens.py", 18_000_000_000, 6)
    assert rec["cap_bytes"] == 18_000_000_000 and rec["threads"] == 6
    with pytest.raises(g.ResourceRefused, match="no resource-cap line"):
        g.verify_ruling(text, RULING, "scripts/f12_tokens.py", 18_000_000_000, 8)
    with pytest.raises(g.ResourceRefused, match="no resource-cap line"):
        g.verify_ruling(text, RULING, "scripts/f12_tokens.py", 19_000_000_000, 6)
    with pytest.raises(g.ResourceRefused, match="heading"):
        g.verify_ruling(text, "#### A5 chair ruling (13 Nov 2026) absent", "x", 1, 1)
    undated = "#### A5 chair ruling without a date"
    with pytest.raises(g.ResourceRefused, match="dated"):
        g.verify_ruling(
            undated + "\nresource-cap: build=x cap_bytes=5 threads=1\n", undated, "x", 5, 1
        )
    # a line in a later block does not count
    with pytest.raises(g.ResourceRefused):
        g.verify_ruling(_prereg("nothing"), RULING, "x", 1, 1)


def _guard(**kw):
    state = {"exit": None}
    kw.setdefault("rss_fn", lambda: 10**8)
    kw.setdefault("mem_fn", lambda: 20 * 10**9)
    kw.setdefault("peak_fn", lambda: 10**8)
    kw.setdefault("interval", 0.01)
    guard = g.ResourceGuard(
        build="scripts/x.py", threads=2, exit_fn=lambda c: state.update(exit=c), **kw
    )
    return guard, state


def test_launch_refused_above_cap():
    guard, _ = _guard()
    with pytest.raises(g.ResourceRefused, match="above the cap"):
        guard.launch(estimate_bytes=g.DEFAULT_CAP_BYTES + 1)


def test_launch_refused_without_free_memory():
    guard, _ = _guard(mem_fn=lambda: 3 * 10**9)
    with pytest.raises(g.ResourceRefused, match="available"):
        guard.launch(estimate_bytes=10**6)


def test_cap_above_default_needs_committed_ruling():
    guard, _ = _guard(cap_bytes=18_000_000_000)
    with pytest.raises(g.ResourceRefused, match="chair ruling"):
        guard.launch(estimate_bytes=10**6)
    text = _prereg("resource-cap: build=scripts/x.py cap_bytes=18000000000 threads=2")
    guard, _ = _guard(cap_bytes=18_000_000_000, ruling=RULING, prereg_text=text,
                      mem_fn=lambda: 21 * 10**9)  # fmt: skip
    with pytest.raises(g.ResourceRefused, match="available"):  # needs cap + 4 GB free
        guard.launch(estimate_bytes=10**6)
    guard, _ = _guard(cap_bytes=18_000_000_000, ruling=RULING, prereg_text=text,
                      mem_fn=lambda: 23 * 10**9)  # fmt: skip
    guard.launch(estimate_bytes=17 * 10**9)
    rec = guard.stop()
    assert (
        rec["ruling"]["cap_bytes"] == 18_000_000_000
        and rec["required_available_bytes"] == 22 * 10**9
    )


def test_monitor_stops_before_breach_and_records(tmp_path):
    seq = iter([10**8, 10**8, 39 * 10**8, 39 * 10**8])
    guard, state = _guard(rss_fn=lambda: next(seq, 39 * 10**8))
    breach = tmp_path / "breach.json"
    guard.launch(estimate_bytes=10**6, breach_path=breach)
    guard.join(timeout=5)
    assert state["exit"] == g.BREACH_EXIT_CODE
    rec = json.loads(breach.read_text())
    assert rec["sampled_tree_rss_bytes"] == 39 * 10**8
    assert rec["stop_threshold_bytes"] == int(0.95 * g.DEFAULT_CAP_BYTES)


def test_check_and_record_peak():
    guard, _ = _guard(peak_fn=lambda: 5 * 10**9)
    guard.launch(estimate_bytes=10**6)
    with pytest.raises(MemoryError, match="peak RSS"):
        guard.check()
    rec = guard.stop()
    assert rec["peak_rss_bytes"] == 5 * 10**9 and rec["within_cap"] is False
    guard, _ = _guard()
    guard.launch(estimate_bytes=10**6)
    rec = guard.stop()
    assert rec["within_cap"] is True and rec["cap_bytes"] == g.DEFAULT_CAP_BYTES
    assert rec["authority"].startswith("PREREG_H1_TF A5 chair resource disposition")


def test_meminfo_and_tree_rss_read_proc():
    assert g.mem_available_bytes() > 0
    assert g.tree_rss_bytes() > 0
    assert g.peak_rss_bytes() > 0


def test_committed_prereg_text_reads_head_not_worktree(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(k)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, env=env)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    p = tmp_path / "docs" / "PREREG_H1_TF.md"
    p.parent.mkdir()
    p.write_text("committed\n")
    git("add", "docs/PREREG_H1_TF.md")
    git("commit", "-q", "-m", "x")
    p.write_text("committed\nuncommitted ruling\n")
    assert g.committed_prereg_text(tmp_path) == "committed\n"
    out = subprocess.run(
        ["git", "-C", str(tmp_path), "config", "--get", "core.bare"],
        check=False, capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert out.stdout.strip() == "false"


def test_root_paths():
    assert (g.ROOT / "scripts" / "f12_inputs_guard.py").exists()
    assert isinstance(g.DATA_ROOT, Path)
