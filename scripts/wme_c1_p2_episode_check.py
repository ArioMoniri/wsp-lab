#!/usr/bin/env python3
"""P2 check on one whole arm-episode (RULING-wme-c1-bank-halt addenda 9-10). Engineering only, $0, CPU: no GPU, no
model weights, no dev or sealed item. ``e2arms.run_planner`` runs with and without the P2 memos on the production
planner inputs (the prep ``fit.pkl``, the real ``library.pt`` with the pinned 9B role map, episode, ``PLANNER_CONFIG``,
seeds); the per-tensor field contributions and the dev evaluations are deterministic stand-ins (both need the GPU).
Reports the wall time of each run and whether the timing-free payloads are equal (``fieldrepair.timing_free_sha``).

    wme_c1_p2_episode_check.py --fit FIT.pkl --library LIB.pt --config CONFIG.json --arm wm_e --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


class StandInWorld:
    def __init__(self):
        self.n = 0

    def evaluate(self, pids):
        self.n += 1
        h = hashlib.sha256(json.dumps(list(pids)).encode()).digest()
        g = np.random.default_rng(int.from_bytes(h[:8], "little"))
        return {
            "gain": float(g.normal(0.0, 0.1)),
            "gain_se": 0.05,
            "guards_pass": bool(g.random() > 0.2),
            "rel_norm": float(g.random()),
            "theta_hash": "s" * 64,
            "seconds": 0.0,
            "guard_means": {},
            "guard_bounds": {},
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, choices=("wm_e", "statnn", "field_ranked"))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from wsp.wm import actions as A
    from wsp.wm.c1 import e2arms as EA
    from wsp.wm.c1 import e2ops as EO
    from wsp.wm.c1 import field as F
    from wsp.wm.c1 import fieldrepair as FR

    sp = importlib.util.spec_from_file_location("rofb", Path(__file__).with_name("wme_c1_rof_bench.py"))
    rb = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(rb)
    fit = pickle.loads(Path(a.fit).read_bytes())
    rm = A.RoleMap.from_config(json.loads(Path(a.config).read_text()), prefix="model.layers.")
    lib = EO.Library(a.library, rm)

    def env():
        m = types.SimpleNamespace(sess=rb.StandInSession(), embed="model.embed_tokens.weight", at_base=lambda: None)
        e = EA.Env(m, StandInWorld(), lib, F.RCache(m, rb.StandInField()), fit["fitted"], fit["support"],
                   fit["q_hat"], fit["mode"])  # fmt: skip
        e.ticks = 0

        def tick(ev, **kw):
            e.ticks += 1

        e.tick = tick
        return e

    rep = {"arm": a.arm, "episode": a.episode}
    for label, memo in (("memo", True), ("plain", False)):
        e = env()
        t0 = time.time()
        out = EA.run_planner(e, a.arm, EA.episode(a.episode), 48, memo=memo)
        rep[label] = {
            "wall_s": time.time() - t0,
            "timing_free_sha256": FR.timing_free_sha(out),
            "r_cache_misses": out["r_cache"]["misses"],
            "ticks": e.ticks,
            "evaluated": len(out["evaluated"]),
        }
        Path(a.out).write_text(json.dumps(rep, indent=1))
        print(label, json.dumps(rep[label]), flush=True)
    m_, p_ = rep["memo"], rep["plain"]
    rep["timing_free_equal"] = m_["timing_free_sha256"] == p_["timing_free_sha256"]
    rep["misses_equal"] = m_["r_cache_misses"] == p_["r_cache_misses"]
    rep["ticks_equal"] = m_["ticks"] == p_["ticks"]
    rep["wall_ratio_memo_over_plain"] = m_["wall_s"] / p_["wall_s"]
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
