#!/usr/bin/env python3
"""Planner benchmark for RULING-wme-c1-bank-halt addendum 9 (P1/P2). Engineering only, $0, CPU: no GPU, no model
weights, no dev or sealed item. It rebuilds ``e2arms.run_planner``'s scorer (``adapter9b.V0RArm`` over the fitted
WM-E / StatNN model from the prep ``fit.pkl``, episode 0, ``PLANNER_CONFIG``, CEM) and runs one fixed planner workload,
``Planner.beam`` for slot 0 of round 0 (seed as ``run_episode_v1`` derives it). ``r_of`` is a deterministic stand-in
(R(E) needs the field on the GPU), so the explored candidates differ from production; the per-call costs and the
redundancy structure are what it measures:

* the wall time of the beam and of the model's batched prediction (``adapter9b.predict_batch``);
* every scored batch, keyed by its exact input bytes: how often the identical batch was already scored (a
  whole-batch memo hit) and how often a row's trajectory was already scored (a row memo hit);
* the beam's result (values and candidates) as a sha256, for a bitwise comparison across processes;
* batch invariance: for a sample of batches, whether each row predicted alone equals its row in the batch bitwise.

Run it once per torch thread count (separate processes):

    wme_c1_planner_bench.py --fit FIT.pkl --arm wm_e|statnn --threads N --out OUT.json [--memo]
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pickle
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


def _rng(*parts) -> np.random.Generator:
    h = hashlib.sha256(json.dumps(parts, default=str).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def batch_key(trajs) -> bytes:
    h = hashlib.sha256()
    for t in trajs:
        for k in ("x", "e", "y", "obs"):
            a = np.ascontiguousarray(t[k])
            h.update(str(a.shape).encode() + a.dtype.str.encode() + a.tobytes())
    return h.digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--arm", required=True, choices=("wm_e", "statnn"))
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument(
        "--memo",
        action="store_true",
        help="the P2 candidate on: r_of memo by primitive ids, whole-batch memo",
    )
    ap.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="stop the slice after N scored batches (0: whole beam)",
    )
    ap.add_argument(
        "--real-rof",
        nargs=2,
        metavar=("LIBRARY_PT", "CONFIG_JSON"),
        help="r_of through the production RCache.r_of(Library.bundles(pids)) (stand-in field contributions)",
    )
    ap.add_argument(
        "--no-profile",
        action="store_true",
        help="time without cProfile (its overhead inflates Python)",
    )
    ap.add_argument("--invariance-sample", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    from wsp.wm import adapter9b as AD
    from wsp.wm import e2 as E2
    from wsp.wm import planner as PN
    from wsp.wm.c1 import e2arms as EA
    from wsp.wm.c1 import field as F

    fit = pickle.loads(Path(a.fit).read_bytes())
    fitted, support, q_hats, mode = fit["fitted"], fit["support"], fit["q_hat"], fit["mode"]
    spec = F.spec_9b()
    lib_index = E2.library_index(EA.OBJ)
    ep = EA.episode(a.episode)
    acts, doses = AD.episode_actions(ep, lib_index)
    prim = lambda x, d: AD._primitive_id(lib_index, ep, x, d)
    rof = {"calls": 0, "keys": set()}
    rmemo = {}

    def r_of_direct(pids):
        return _rng("r", pids).normal(0.0, 0.01, spec.dim)

    if a.real_rof:  # e2arms.Env.r_of_pids: RCache.r_of(Library.bundles(pids))[0]
        import importlib.util
        import types

        from wsp.wm import actions as A
        from wsp.wm.c1 import e2ops as EO

        sp = importlib.util.spec_from_file_location(
            "rofb", Path(__file__).with_name("wme_c1_rof_bench.py")
        )
        rb = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(rb)
        lib = EO.Library(a.real_rof[0], A.RoleMap.from_config(json.loads(Path(a.real_rof[1]).read_text()),
                                                              prefix="model.layers."))  # fmt: skip
        m = types.SimpleNamespace(
            sess=rb.StandInSession(), embed="model.embed_tokens.weight", at_base=lambda: None
        )
        rc = F.RCache(m, rb.StandInField())
        r_of_direct = lambda pids: rc.r_of(lib.bundles(pids))[0]

    def r_of(st):
        rof["calls"] += 1
        pids = tuple(prim(x, d) for x, d in st)
        rof["keys"].add(pids)
        if not a.memo:
            return r_of_direct(list(pids))
        v = rmemo.get(pids)
        if v is None:
            v = rmemo[pids] = r_of_direct(list(pids))
        return v.copy()

    model = fitted["wm"] if a.arm == "wm_e" else fitted["static"]
    q_hat = q_hats["wm"] if a.arm == "wm_e" else q_hats["static"]
    pa = AD.V0RArm(
        model, fitted["base_obs"], spec, r_of, acts, doses, EA.PRIMARY_INDEX, E2.ORIENT[EA.OBJ], q_hat,
        EA.GUARD_SPECS, support=support, psi=AD.SUPPORT_PSI, mode=mode, primitive_of=prim,
    )  # fmt: skip
    st = {
        "batches": 0,
        "rows": 0,
        "seconds": 0.0,
        "batch_hits": 0,
        "row_hits": 0,
        "row_dups_in_batch": 0,
    }
    seen_b, seen_r, sample, memo, out_h = set(), set(), [], {}, []
    orig_run = pa._run

    class StopSlice(Exception):
        pass

    def ohash(out):
        h = hashlib.sha256()
        for k in ("mean", "sigma", "members"):
            h.update(np.ascontiguousarray(out[k]).tobytes())
        return h.hexdigest()

    def run(trajs):
        if a.max_batches and st["batches"] >= a.max_batches:
            raise StopSlice
        kb = batch_key(trajs)
        st["batches"] += 1
        st["rows"] += len(trajs)
        st["batch_hits"] += kb in seen_b
        seen_b.add(kb)
        ks = [batch_key([t]) for t in trajs]
        st["row_hits"] += sum(k in seen_r for k in ks)
        st["row_dups_in_batch"] += len(ks) - len(set(ks))
        seen_r.update(ks)
        if a.memo and kb in memo:
            out = {k: v.copy() for k, v in memo[kb].items()}
            out_h.append(ohash(out))
            return out
        t0 = time.time()
        out = orig_run(trajs)
        st["seconds"] += time.time() - t0
        if a.memo:
            memo[kb] = {k: v.copy() for k, v in out.items()}
        out_h.append(ohash(out))
        if len(sample) < a.invariance_sample and len(trajs) > 1 and st["batches"] % 7 == 0:
            sample.append((trajs, out))
        return out

    pa._run = run
    planner = PN.Planner(len(pa.actions), AD.DOSE_BOUNDS, "cem", AD.PLANNER_CONFIG)
    seed = int(ep["seeds"]["planner"])  # round 0, attempt 0; slot 0 of propose
    rng = np.random.default_rng(seed)
    draw = pa.draw(rng)
    prof = cProfile.Profile()
    t0 = time.time()
    if not a.no_profile:
        prof.enable()
    try:
        res = planner.beam(pa, draw, pa.mode, None, seed, set(), None, pa.canon)
    except StopSlice:
        res = []
    prof.disable()
    wall = time.time() - t0
    s_ = io.StringIO()
    if not a.no_profile:
        pstats.Stats(prof, stream=s_).sort_stats("tottime").print_stats(25)
    res_h = hashlib.sha256()
    for v, c in res:
        res_h.update(
            np.float64(v).tobytes() + json.dumps([(int(x), float(z).hex()) for x, z in c]).encode()
        )
    inv, inv_diff = [], 0.0
    for trajs, out in sample:
        for i, t in enumerate(trajs):
            o1 = orig_run([t])
            pairs = [(o1[k][:, 0], out[k][:, i]) if k == "members" else (o1[k][0], out[k][i])
                     for k in ("mean", "sigma", "members")]  # fmt: skip
            inv.append(all(np.array_equal(x, y) for x, y in pairs))
            inv_diff = max(inv_diff, max(float(np.max(np.abs(x - y))) for x, y in pairs))
    rep = {
        "arm": a.arm,
        "episode": a.episode,
        "mode": mode,
        "torch_threads": torch.get_num_threads(),
        "memo": a.memo,
        "cprofile": not a.no_profile,
        "real_rof": bool(a.real_rof),
        "max_batches": a.max_batches,
        "slice_complete_beam": bool(res),
        "beam_wall_s": wall,
        "predict_seconds": st["seconds"],
        "predict_batches": st["batches"],
        "predict_rows": st["rows"],
        "whole_batch_repeat_fraction": st["batch_hits"] / max(1, st["batches"]),
        "row_repeat_fraction": st["row_hits"] / max(1, st["rows"]),
        "row_duplicate_within_batch_fraction": st["row_dups_in_batch"] / max(1, st["rows"]),
        "distinct_batches": len(seen_b),
        "distinct_rows": len(seen_r),
        "r_of_calls": rof["calls"],
        "r_of_distinct_primitive_tuples": len(rof["keys"]),
        "beam_result_sha256": res_h.hexdigest(),
        "batch_outputs_sha256": hashlib.sha256("".join(out_h).encode()).hexdigest(),
        "batch_output_hashes_first": out_h[:50],
        "batch_invariance_rows_checked": len(inv),
        "batch_invariance_rows_bitwise": int(sum(inv)),
        "batch_invariance_max_abs_diff": inv_diff,
        "cprofile_tottime_top25": s_.getvalue(),
    }
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("batch_output_hashes_first", "cprofile_tottime_top25")},
                     indent=1))  # fmt: skip
    print(rep["cprofile_tottime_top25"])


if __name__ == "__main__":
    main()
