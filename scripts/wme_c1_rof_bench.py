#!/usr/bin/env python3
"""CPU benchmark of the planner's R(E) path (RULING-wme-c1-bank-halt addendum 9, P1/P2). Engineering only, $0: no
GPU, no model weights, no dev or sealed item. It rebuilds the production call ``e2arms.Env.r_of_pids(pids)`` =
``RCache.r_of(Library.bundles(pids))[0]`` with the real library file, the real role map (the pinned Qwen3.5-9B
``config.json``) and the real ``wmcp.r_from_contractions`` builder; only the per-tensor field contributions are
deterministic stand-ins (the field lives on the GPU instance). After the cache is warm, it times cache-hit calls over
the planner's prefix pattern (every prefix of a candidate's step list), then a memoised ``r_of_pids`` keyed by the
primitive-id tuple, and checks the memoised R vectors are bitwise equal to the direct ones.

    wme_c1_rof_bench.py --library LIB.pt --config CONFIG.json --out OUT.json [--calls 2000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


class StandInField:
    """Every touched writer is in the support; contributions are deterministic pseudo-random numbers."""

    def __init__(self):
        self.param_group = _AllNames()

    @staticmethod
    def _vals(name, D, objs):
        h = int.from_bytes(
            hashlib.sha256((name + repr(D.tolist())).encode()).digest()[:8], "little"
        )
        g = np.random.default_rng(h)
        return {o: (float(g.normal()), float(abs(g.normal()))) for o in objs}

    def contract_tensor(self, name, D, objs):
        return self._vals(name, D, objs)

    def contract_embed(self, name, D, objs):
        return self._vals(name, D, objs)


class _AllNames(dict):
    def __contains__(self, k):
        return True


class StandInSession:
    """Records the ordered action keys per tensor; the displacement is a deterministic function of them (so R depends
    on the actions and their doses, as the realised displacement does)."""

    def __init__(self):
        self.d = {}

    def apply(self, a):
        self.d.setdefault(a.tensor, []).append(repr(a.key()))

    def displacement(self, name):
        import torch

        h = hashlib.sha256("|".join(self.d.get(name, [])).encode()).digest()
        return torch.tensor([int.from_bytes(h[:4], "little") / 2**32, int.from_bytes(h[4:8], "little") / 2**32],
                            dtype=torch.float64)  # fmt: skip

    def restore(self):
        self.d = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=2000)
    ap.add_argument("--episode", type=int, default=0)
    a = ap.parse_args()
    import types

    from wsp.wm import actions as A
    from wsp.wm import adapter9b as AD
    from wsp.wm import e2 as E2
    from wsp.wm.c1 import e2arms as EA
    from wsp.wm.c1 import e2ops as EO
    from wsp.wm.c1 import field as F

    cfg = json.loads(Path(a.config).read_text())
    rm = A.RoleMap.from_config(cfg, prefix="model.layers.")
    lib = EO.Library(a.library, rm)
    ep = EA.episode(a.episode)
    lib_index = E2.library_index(EA.OBJ)
    acts, doses = AD.episode_actions(ep, lib_index)
    m = types.SimpleNamespace(
        sess=StandInSession(), embed="model.embed_tokens.weight", at_base=lambda: None
    )
    rc = F.RCache(m, StandInField())
    direct = lambda pids: rc.r_of(lib.bundles(pids))[0]

    # candidate step lists as the planner forms them: up to MAX_PRIMITIVES (action, registered dose) steps
    rng = np.random.default_rng(20261123)
    pool = [(act, d) for act in acts for d in doses[act]]
    cands = []
    for _ in range(a.calls):
        k = int(rng.integers(1, E2.MAX_PRIMITIVES + 1))
        idx = rng.choice(len(pool), size=k, replace=False)
        cands.append([pool[i] for i in idx])
    prim = lambda st: [AD._primitive_id(lib_index, ep, x, d) for x, d in st]
    calls = [
        prim(c[:t]) for c in cands for t in range(1, len(c) + 1)
    ]  # every prefix, as v0_trajectory_r asks
    t0 = time.time()
    for pids in (
        calls
    ):  # warm: every per-tensor contribution in the cache (misses), as the episode's first visits
        direct(pids)
    warm_s = time.time() - t0
    t0 = time.time()
    ref = [direct(pids) for pids in calls]  # cache hits: the planner's steady state
    hit_s = time.time() - t0
    memo = {}

    def memoised(pids):
        k = tuple(pids)
        v = memo.get(k)
        if v is None:
            v = direct(pids)
            memo[k] = v
        return v.copy()

    t0 = time.time()
    got = [memoised(pids) for pids in calls]
    memo_first_s = time.time() - t0
    t0 = time.time()
    got2 = [memoised(pids) for pids in calls]
    memo_hit_s = time.time() - t0
    bitwise = all(
        np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(ref, got + got2[: len(ref)])
    )
    bitwise = bitwise and all(np.array_equal(x, y) for x, y in zip(ref, got2))
    n = len(calls)
    rep = {
        "episode": a.episode,
        "r_of_calls": n,
        "distinct_primitive_tuples": len(memo),
        "tensors_per_call_median": float(
            np.median([len({x.tensor for s in lib.bundles(p) for x in s}) for p in calls[:200]])
        ),
        "warm_s": warm_s,
        "direct_hit_ms_per_call": 1000 * hit_s / n,
        "memoised_first_pass_ms_per_call": 1000 * memo_first_s / n,
        "memoised_hit_ms_per_call": 1000 * memo_hit_s / n,
        "memoised_bitwise_equal_to_direct": bool(bitwise),
        "note": "stand-in field contributions; the R assembly, the library bundles and the cache are the production code",
    }
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
