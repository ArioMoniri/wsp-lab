"""Full-refit validation under RULING-wme-v2 (A1, A2, A6). Synthetic worlds only; no real outcome.

Identical to ``wm.powersim_refit`` (v1) in world, bank structure, allocation, labels and **the alternative, which is
not re-tuned** (gamma = 3: a decomposition-dependent path effect; null: residuals shuffled beyond the field). The
pipeline under test is v2's:

* predictors on the shared representation R(E_T) (``wm.wmcp``): WM-CP (canonical path) against the static family
  S1 F3u(a, q), S2 StaticR, S3 StaticR-aux sized to WM-CP (+-10 %), chosen per outer fold on 4 inner component folds
  at h = 2 (0.5 % relative tie -> lower capacity);
* inference: the component-level sign-flip test (``wm.signflip.ht_test``, 10,000 flips, one-sided 0.025) with the
  inverted lower bound; C_T and its bound reported descriptively;
* gates: C2-S and C2-D hold by construction (asserted: alternative decompositions give identical WM-CP predictions);
  C2-I is a gate: I1 round trips (realised E_T = 0, predicted exactly 0) and I2-style controls, a representation
  controls: the embedding-chart column (seen non-zero in training through row_scale roots, as on 9B) set with
  a = q = 0 and no coordinate content (E_T != 0 for the representation, zero field), checked by the v1 rule.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from wsp.wm import powersim_refit as R1
from wsp.wm import powersim_v1 as PV
from wsp.wm import signflip as SF
from wsp.wm import synthetic as S
from wsp.wm import wmcp as CP

K_OUT, K_IN, TIE_REL = 5, 4, 0.005


def build_bank_v2(world, g, seed):
    """``powersim_refit.build_bank`` (same seeds, roots, ladders, blocks and orders) with each step's family
    recorded, so the embedding-chart column below can be formed."""
    rng = np.random.default_rng(seed)
    fam_roots = {f: [] for f in PV.FAMILIES}
    for f in PV.FAMILIES:
        for _ in range(g["r_f"]):
            u = rng.standard_normal(world.J)
            fam_roots[f].append(u / np.linalg.norm(u))
    trajs = []
    for ci, c in enumerate(g["components"]):
        roots = [(f, fam_roots[f].pop()) for f in c["families"]]
        for f, r in roots:
            for sgn in (1.0, -1.0):
                incs = [sgn * d for d in R1._increments(R1.LADDER_CUM)]
                t = R1.traj_vec(world, [(r, d) for d in incs])
                trajs.append(
                    dict(
                        t,
                        comp=ci,
                        stratum=c["stratum"],
                        kind="ladder",
                        step_fams=[f] * 4,
                        step_doses=incs,
                    )
                )
        if c["stratum"] == "block":
            order = list(range(len(roots)))
            rng.shuffle(order)
            for seq in (order, order[::-1]):
                t = R1.traj_vec(world, [(roots[i][1], 1.0) for i in seq])
                trajs.append(
                    dict(
                        t,
                        comp=ci,
                        stratum="block",
                        kind="composition",
                        step_fams=[roots[i][0] for i in seq],
                        step_doses=[1.0] * len(seq),
                    )
                )
    return trajs


def spec_with_dead(world) -> CP.RSpec:
    s = CP.spec_synthetic(world)
    names = s.names[:-1] + ("dead",) + s.names[-1:]
    deg = s.degree[:-1] + (1,) + s.degree[-1:]
    return CP.RSpec(names, deg, s.n_obj, len(names) - 1)


def _r(world, E, dead=0.0):
    r = CP.r_synthetic(world, E)
    return np.concatenate([r[:-1], [dead], r[-1:]])


def states_of(world, trajs):
    """The "dead" column is the embedding-chart analogue: the signed dose content of row_scale roots (degree 1),
    non-zero in training whenever a row_scale root is in the history, with non-zero field features there."""
    out = []
    for t in trajs:
        for h in range(1, t["horizon"] + 1):
            E = t["x"][h, : world.J]
            fams, doses = t.get("step_fams", []), t.get("step_doses", [])
            chart = world.step * sum(
                d for f, d in zip(fams[:h], doses[:h], strict=True) if f == "row_scale"
            )
            out.append(
                {
                    "r": _r(world, E, chart),
                    "y": np.asarray(t["y"][h], float),
                    "obs": np.asarray(t["obs"][h], float),
                    "h": h,
                    "comp": t["comp"],
                    "kind": t["kind"],
                }
            )
    return out


def _err(model, states, se):
    P = (
        model.predict_batch(np.stack([s["r"] for s in states]))["mean"]
        if hasattr(model, "predict_batch")
        else np.stack([model.predict(s["r"])[0] for s in states])
    )
    Y = np.stack([s["y"] for s in states])
    return np.abs(P - Y) / se


def _members(cfg, spec, bo, ref_params=None):
    out = {
        "S1": lambda: CP.F3uR(spec),
        "S2": lambda: CP.StaticR(cfg, spec, bo, n_seeds=5, aux=False),
        "S3": lambda: _s3(cfg, spec, bo, ref_params),
    }
    return out


def _s3(cfg, spec, bo, ref_params):
    m = CP.StaticR(cfg, spec, bo, n_seeds=5, aux=True)
    if ref_params:
        m.size_to(ref_params, len(bo))
    return m


def choose_static(cfg, spec, bo, train, se, seed, ref_params):
    comps = sorted({s["comp"] for s in train})
    order = list(np.random.default_rng(seed).permutation(comps))
    inner = {c: i % K_IN for i, c in enumerate(order)}
    loss = {}
    for name, ctor in _members(cfg, spec, bo, ref_params).items():
        tot = 0.0
        for k in range(K_IN):
            fit = [s for s in train if inner[s["comp"]] != k]
            val = [s for s in train if inner[s["comp"]] == k and s["h"] == 2]
            if not fit or not val:
                continue
            m = ctor().fit(fit, [s["comp"] for s in fit])
            tot += float(_err(m, val, se).sum())
        loss[name] = tot
    best = min(loss.values())
    for name in ("S1", "S2", "S3"):
        if loss[name] <= best * (1 + TIE_REL):
            return name, loss
    return min(loss, key=loss.get), loss


def replicate(rep, cell, null, gamma=3.0, cfg=None, n_flip=SF.N_FLIP, seed=PV.SEED, n_inert=12):
    t0 = time.time()
    torch.set_num_threads(1)
    cfg = cfg or CP.CPConfig()
    g = PV.graph(cell["r_f"], cell["n_blk"])
    world = S.SynthWorld(kind="saturating", seed=10_000 + rep, n_dirs=R1.J_SHARED, n_obj=6)
    bo = world.base_observation().payload["obs"]
    raw = build_bank_v2(world, g, seed + rep)
    trajs, se = R1.label(raw, world, 1.0, seed + 7 * rep, null, gamma)
    spec = spec_with_dead(world)
    _cal, fold, rest = R1._alloc(g["components"], cell["n_cal"], seed + 13 * rep)
    allst = states_of(world, trajs)
    e_w, e_s, comps, chosen, inert_ok, ct_inv = [], [], [], [], [], []
    for k in range(K_OUT):
        held = {c for c in rest if fold[c] == k}
        trn = {c for c in rest if fold[c] != k}
        train = [s for s in allst if s["comp"] in trn and s["h"] <= 2]
        cp = CP.CanonicalPathWM(cfg, spec, bo, n_seeds=5).fit(train, [s["comp"] for s in train])
        name, _ = choose_static(cfg, spec, bo, train, se, seed + 31 * rep + k, CP.param_count(cp))
        chosen.append(name)
        st = _members(cfg, spec, bo, CP.param_count(cp))[name]().fit(
            train, [s["comp"] for s in train]
        )
        test = [s for s in allst if s["comp"] in held and s["h"] in (3, 4)]
        if test:
            ew, es = _err(cp, test, se), _err(st, test, se)
            e_w.append(ew.ravel())
            e_s.append(es.ravel())
            comps.append(np.repeat([s["comp"] for s in test], ew.shape[1]))
        # C2-S / C2-D by construction: a held-out ladder state re-expressed as one step gives the same R
        lad = [t for t in trajs if t["comp"] in held and t["kind"] == "ladder"][:1]
        for t in lad:
            u = t["e"][0] / np.linalg.norm(t["e"][0])
            one = R1.traj_vec(
                world, [(u, float(np.linalg.norm(t["x"][3, : world.J]) / world.step))]
            )
            ct_inv.append(
                float(
                    np.abs(
                        cp.predict(_r(world, one["x"][1, : world.J]))[0]
                        - cp.predict(_r(world, t["x"][3, : world.J]))[0]
                    ).max()
                )
            )
        # C2-I: I1 round trips (exact zero) and I2-style dead-column controls
        s0 = cp.k_rule["s0"]
        ys, sg = [], []
        for j in range(n_inert // 2):
            m0, s_0 = cp.predict(np.zeros(spec.dim))
            ys.append(m0 / se)
            sg.append(np.maximum(s_0 / se, 1.0))
        for j in range(n_inert - n_inert // 2):
            r = np.zeros(spec.dim)
            mag = s0 * (0.5, 1.0, -0.5, -1.0, 2.0, -2.0)[j % 6]
            r[spec.names.index("dead")] = mag
            r[spec.norm_index] = abs(mag)
            m0, s_0 = cp.predict(r)
            ys.append(m0 / se)
            sg.append(np.maximum(s_0 / se, 1.0))
        ys, sg = np.array(ys), np.array(sg)
        inert_ok.append(
            bool(np.mean(np.abs(ys) > 2 * sg) <= 0.10 and np.all(np.abs(ys).mean(0) <= 1.0))
        )
    e_w, e_s, comps = np.concatenate(e_w), np.concatenate(e_s), np.concatenate(comps)
    r = SF.ht_test(e_w, e_s, comps, n_flip=n_flip, seed=seed + rep)
    c2i = all(inert_ok)
    return {
        "rep": rep,
        "null": null,
        "gamma": gamma,
        "stat": r["stat"],
        "p": r["p"],
        "lb": r["lb"],
        "ct": r["ct"],
        "ct_lb": r["ct_lb"],
        "c2i": c2i,
        "decomposition_gap_max": max(ct_inv or [0.0]),
        "reject": bool(r["reject"] and c2i),
        "static_chosen": chosen,
        "m": r["n_components"],
        "seconds": time.time() - t0,
    }


def summarise(rows) -> dict:
    out = {}
    for null in (True, False):
        rs = [r for r in rows if r["null"] == null]
        if not rs:
            continue
        k, n = sum(r["reject"] for r in rs), len(rs)
        d = {
            "reps": n,
            "rate": k / n,
            "ci95": PV.clopper_pearson(k, n),
            "power_lb": PV.cp_lower_one_sided(k, n),
            "mean_ct": float(np.mean([r["ct"] for r in rs])),
            "mean_stat": float(np.mean([r["stat"] for r in rs])),
            "c2i_pass": float(np.mean([r["c2i"] for r in rs])),
            "decomposition_gap_max": float(max(r["decomposition_gap_max"] for r in rs)),
            "static_chosen": {
                m: sum(c == m for r in rs for c in r["static_chosen"]) for m in ("S1", "S2", "S3")
            },
            "mean_seconds": float(np.mean([r["seconds"] for r in rs])),
        }
        if null:
            d["coverage_at_zero"] = float(np.mean([r["lb"] <= 0 for r in rs]))
            d["pass"] = bool(
                d["rate"] <= PV.REQ["size_max"] and d["coverage_at_zero"] >= PV.REQ["coverage_min"]
            )
        else:
            d["pass"] = bool(d["power_lb"] >= PV.REQ["power_lb_min"])
        out["null" if null else "alt"] = d
    return out


if __name__ == "__main__":
    import json
    import sys

    cell = json.loads(sys.argv[1])
    print(
        json.dumps(
            replicate(
                int(sys.argv[2]),
                cell,
                sys.argv[3] == "null",
                float(sys.argv[4]),
                cfg=CP.CPConfig(epochs=int(sys.argv[5])) if len(sys.argv) > 5 else None,
            )
        )
    )
