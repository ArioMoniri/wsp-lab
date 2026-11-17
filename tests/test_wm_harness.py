"""wm.harness: H4 longest-first preflight and H5 resumable content-addressed units + resident server
(SYNTHESIS-r9 C8; PREREG_WME §9). Includes the kill/resume test on a real subprocess (SIGKILL mid-run)."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from wsp.wm import actions as A
from wsp.wm import harness as H

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ H4


def test_worst_batches_are_longest_first_per_objective():
    lens = {"a": [5, 50, 20, 49], "b": [3, 4]}
    plan = H.worst_batches(lens, max_batch_tokens=100)
    assert plan[0]["objective"] == "a" and plan[0]["rows"] == [
        1,
        3,
    ]  # 50 + 49 (longest pair under 100 tokens)
    assert plan[0]["padded_tokens"] == 100
    assert {p["objective"] for p in plan} == {"a", "b"}
    assert plan == sorted(plan, key=lambda p: -p["max_len"])


def test_preflight_aborts_before_the_main_loop_above_the_cap():
    plan = H.worst_batches({"a": [10, 90], "b": [10]}, max_batch_tokens=100)
    seen = []
    meter = iter([10e9, 27.5e9])
    with pytest.raises(H.PreflightError) as e:
        H.preflight(
            plan, run=lambda p: seen.append(p["objective"]), device_bytes=lambda: next(meter)
        )
    assert seen == ["a", "b"][: len(seen)] and "27.5" in str(e.value)
    ok = H.preflight(plan, run=lambda p: None, device_bytes=lambda: 20e9)
    assert (
        ok["pass"]
        and ok["peak_bytes"] == 20e9
        and ok["cap_bytes"] == H.DEVICE_CAP_BYTES == 27_000_000_000
    )


# ------------------------------------------------------------------ H5 units


def test_unit_key_is_content_addressed():
    k = H.unit_key({"root": "r", "dose": 1.0}, "t" * 64, "p" * 64)
    assert k == H.unit_key({"dose": 1.0, "root": "r"}, "t" * 64, "p" * 64)
    assert k != H.unit_key({"root": "r", "dose": 2.0}, "t" * 64, "p" * 64)
    assert k != H.unit_key({"root": "r", "dose": 1.0}, "u" * 64, "p" * 64)


def test_unit_store_atomic_write_verify_and_tamper_detection(tmp_path):
    st = H.UnitStore(tmp_path)
    k = H.unit_key({"x": 1}, "t" * 64, "p" * 64)
    assert not st.done(k)
    st.write(k, {"value": 1.5}, theta_after=("t" * 64))
    assert st.done(k) and st.read(k)["value"] == 1.5
    p = st.path(k)
    rec = json.loads(p.read_text())
    rec["payload"]["value"] = 2.0
    p.write_text(json.dumps(rec))
    with pytest.raises(H.CorruptUnit):
        st.read(k)
    assert not st.done(k)  # a corrupt unit is recomputed, never trusted
    assert not list(tmp_path.glob("**/*.tmp"))


def _toy():
    rm = A.RoleMap(2, 4, 8, 10, ("full_attention",) * 2, tied=False)
    g = torch.Generator().manual_seed(0)
    params = {
        rm.name(L, "down"): torch.randn(4, 8, generator=g).to(torch.bfloat16) for L in range(2)
    }
    return rm, params


def test_resident_server_restores_bitwise_and_detects_drift():
    rm, params = _toy()
    base = A.state_hash(params)
    sess = A.EditSession(params, base_hash=base, rolemap=rm)
    srv = H.ResidentServer(
        sess, evaluate=lambda: {"s": float(sum(p.float().sum() for p in params.values()))}
    )
    acts = [A.scale(rm.name(0, "down"), factor=1.5, dose=1.0)]
    r1 = srv.visit(acts)
    assert r1["theta_hash"] != base and sess.theta_hash() == base
    assert A.state_hash(params) == base
    r2 = srv.visit(acts)
    assert r1 == r2  # bitwise repeat

    class Leaky(H.ResidentServer):
        def _restore(self):
            pass  # forgets to restore

    bad = Leaky(A.EditSession(params, base_hash=base, rolemap=rm), evaluate=dict)
    with pytest.raises(H.ThetaDrift):
        bad.visit(acts)


def test_run_units_skips_verified_units(tmp_path):
    st = H.UnitStore(tmp_path)
    calls = []
    units = [({"i": i}, f"r{i}") for i in range(4)]
    fn = lambda recipe: calls.append(recipe["i"]) or {"v": recipe["i"] * 2}
    H.run_units(st, units, fn, theta0="t" * 64, panel="p" * 64)
    H.run_units(st, units, fn, theta0="t" * 64, panel="p" * 64)
    assert calls == [0, 1, 2, 3]


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_kill_and_resume_gives_identical_hashes(tmp_path):
    """SIGKILL the resident worker mid-run, resume, and compare every unit with an uninterrupted run."""
    script = [sys.executable, "-m", "wsp.wm.harness", "toy-worker"]
    full = tmp_path / "full"
    subprocess.run([*script, str(full), "8", "0", "full"], check=True, env=_env(), timeout=120)
    part = tmp_path / "part"
    p = subprocess.Popen([*script, str(part), "8", "0.4", "run1"], env=_env())
    deadline = time.time() + 60
    while time.time() < deadline and len(list((part / "units").glob("*/*.json"))) < 3:
        time.sleep(0.05)
    p.send_signal(signal.SIGKILL)
    p.wait(timeout=30)
    before = set(H.UnitStore(part / "units").digest())
    assert 3 <= len(before) < 8
    subprocess.run([*script, str(part), "8", "0", "run2"], check=True, env=_env(), timeout=120)
    log = [json.loads(x) for x in (part / "computed.jsonl").read_text().splitlines()]
    run2 = {r["key"] for r in log if r["run"] == "run2"}
    assert not run2 & before  # verified units are never recomputed
    assert (
        len({r["key"] for r in log}) == 8 and len(log) <= 9
    )  # a kill loses at most the unit in flight
    a = H.UnitStore(full / "units").digest()
    b = H.UnitStore(part / "units").digest()
    assert a == b and len(a) == 8
    assert json.loads((part / "final.json").read_text())["theta_restored"] is True
