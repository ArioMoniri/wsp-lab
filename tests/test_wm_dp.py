"""Data-parallel runner for the bank and confirmatory work (RULING-wme-resources §2; pro-engineer L1-L2).

Units are (state, objective, fixed item chunk); the chunking is a function of the panel only, so k never changes a
batch. Workers claim units through exclusive lock files and write content-addressed results; the reduction is on
CPU, in fp64 and in a fixed (objective, group, item) order, so 1-vs-k results agree (rtol 1e-6; bitwise on CPU).
"""

import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from wsp.wm import dp

ROOT = Path(__file__).resolve().parents[1]


def _items(n=10, objs=("a", "b")):
    return {
        o: [{"id": f"{o}{i:03d}", "group": f"g{i // 2}", "hash": f"h{o}{i}"} for i in range(n)]
        for o in objs
    }


def test_chunks_depend_on_panel_only():
    its = _items(11)
    c1 = dp.chunk_items(its, chunk_size=4)
    shuffled = {o: random.Random(1).sample(v, len(v)) for o, v in its.items()}
    c2 = dp.chunk_items(shuffled, chunk_size=4)
    assert [(c["objective"], c["chunk"], c["item_ids"]) for c in c1] == [
        (c["objective"], c["chunk"], c["item_ids"]) for c in c2
    ]
    assert [c["chunk"] for c in c1 if c["objective"] == "a"] == [0, 1, 2]
    assert sum(len(c["item_ids"]) for c in c1) == 22


def test_units_content_addressed_and_unique():
    states = {"base": [], "s1": [{"family": "scale", "tensor": "t", "dose": 1.0}]}
    units = dp.make_units(states, dp.chunk_items(_items(6), 4), theta0="th0", panel="p")
    keys = [u["key"] for u in units]
    assert len(keys) == len(set(keys)) == 2 * 2 * 2
    again = dp.make_units(states, dp.chunk_items(_items(6), 4), theta0="th0", panel="p")
    assert keys == [u["key"] for u in again]
    other = dp.make_units(states, dp.chunk_items(_items(6), 4), theta0="th1", panel="p")
    assert set(keys).isdisjoint(u["key"] for u in other)


def test_claim_is_exclusive_and_stale_claims_expire(tmp_path):
    q = dp.ClaimDir(tmp_path / "claims", ttl_s=0.2)
    assert q.claim("k1", "w0")
    assert not q.claim("k1", "w1")
    time.sleep(0.3)
    assert q.claim("k1", "w1")  # stale claim of a dead worker is taken over
    q.release("k1")
    assert q.claim("k1", "w2")


def test_fixed_order_reduction_is_order_invariant():
    rng = np.random.default_rng(0)
    recs = [
        {"objective": o, "group": f"g{i % 7}", "item_id": f"{o}{i}", "value": float(np.float32(v))}
        for o in ("x", "y")
        for i, v in enumerate(rng.standard_normal(200) * 1e3)
    ]
    a = dp.reduce_fixed_order(recs)
    for s in range(5):
        b = dp.reduce_fixed_order(random.Random(s).sample(recs, len(recs)))
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_paired_stats_group_means_and_se():
    base = {"o": {"g0": 1.0, "g1": 2.0, "g2": 3.0}}
    edit = {"o": {"g0": 1.5, "g1": 2.0, "g2": 4.0}}
    st = dp.paired_stats(base, edit)["o"]
    d = np.array([0.5, 0.0, 1.0])
    assert st["n"] == 3
    assert st["mean"] == pytest.approx(d.mean())
    assert st["se"] == pytest.approx(d.std(ddof=1) / np.sqrt(3))
    with pytest.raises(ValueError):
        dp.paired_stats(base, {"o": {"g0": 1.0}})


def test_parity_check():
    a = {"o": {"g0": 1.0, "g1": 2.0}}
    assert dp.parity(a, {"o": {"g0": 1.0 + 1e-9, "g1": 2.0}})["pass"]
    r = dp.parity(a, {"o": {"g0": 1.0 + 1e-3, "g1": 2.0}})
    assert not r["pass"] and r["max_rel"] > 1e-6
    assert not dp.parity(a, {"o": {"g0": 1.0}})["pass"]


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _run_toy(out, k, n_states=5, sleep=0.0, extra=()):
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "wsp.wm.dp",
                "toy-worker",
                str(out),
                str(n_states),
                str(sleep),
                f"w{i}",
                *extra,
            ],
            env=_env(),
        )
        for i in range(k)
    ]
    return procs


def test_one_vs_k_parity_and_resume(tmp_path):
    one, many = tmp_path / "one", tmp_path / "many"
    for p in _run_toy(one, 1):
        assert p.wait(60) == 0
    for p in _run_toy(many, 3):
        assert p.wait(60) == 0
    r1 = dp.toy_reduce(one, 5)
    rk = dp.toy_reduce(many, 5)
    assert json.dumps(r1, sort_keys=True) == json.dumps(rk, sort_keys=True)  # bitwise on CPU
    assert dp.parity_states(r1, rk)["pass"]
    # every unit was computed exactly once across the k workers
    log = [json.loads(x) for x in (many / "computed.jsonl").read_text().splitlines()]
    assert len(log) == len({x["key"] for x in log})
    assert len({x["worker"] for x in log}) >= 2


def test_kill_and_resume(tmp_path):
    out = tmp_path / "kr"
    (p,) = _run_toy(out, 1, n_states=6, sleep=0.15)
    deadline = time.time() + 30
    while time.time() < deadline:
        f = out / "computed.jsonl"
        if f.exists() and len(f.read_text().splitlines()) >= 3:
            break
        time.sleep(0.05)
    p.send_signal(signal.SIGKILL)
    p.wait(10)
    done_before = len(dp.UnitStore(out / "units").digest())
    assert 0 < done_before < 6 * 4
    for q in _run_toy(out, 2, n_states=6, extra=("--ttl", "0.5")):
        assert q.wait(60) == 0
    ref = tmp_path / "ref"
    for q in _run_toy(ref, 1, n_states=6):
        assert q.wait(60) == 0
    assert json.dumps(dp.toy_reduce(out, 6), sort_keys=True) == json.dumps(
        dp.toy_reduce(ref, 6), sort_keys=True
    )


def test_heartbeat_file_tracks_progress(tmp_path):
    out = tmp_path / "hb"
    for p in _run_toy(out, 1, n_states=2):
        assert p.wait(60) == 0
    prog = json.loads((out / "progress" / "w0.json").read_text())
    assert prog["done"] == 2 * 4 and prog["state"] == "finished"
