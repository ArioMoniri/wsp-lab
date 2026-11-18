"""Full-refit validation under RULING-wme-v2b A7: the physically possible H-T alternative. Synthetic only.

**World.** ``SynthWorld(kind="quadratic")``, so the R0 field F(E) = a + q = <g0, E> + 1/2 E^T H0 E is exact. The
bank, its roots, ladders, blocks, allocation, folds, predictors (WM-CP vs the R-space static family), inference
(component-level sign-flip, 10,000 flips) and C2-I gate are those of ``wm.powersim_refit_v2``.

**Registered alternative (A7)**, a function of the realised cumulative edit E_T and the component c only (path
independent), second order near zero, with a saturating normalisation-type departure along each of the
component's dose directions u_r and an independent random strength per component:

    y(E) = F(E) + kappa * s_c * sum_r F(P_r E) * (rho(z_r) - 1),
    z_r = <u_r, E> / step (dose units),  P_r E = <u_r, E> u_r,  rho(z) = (1 + (z / Z0)^2)^(-1/2),
    s_c ~ LogNormal(0, STRENGTH_SIGMA) independently per component.

For one root at kappa = s_c = 1 this is the normalised field F * rho(z): exact to second order at zero (the
departure is O(|E|^3)), growing with |E_T| and saturating the linear part. The magnitude ``kappa`` is the only
calibrated quantity: the selected static member's median held-out SE-scaled error at h = 3-4 equals
:data:`CAL_TARGET` (2.17 paired SE, the exposed exploratory pilot-2 value), on calibration seeds disjoint from the
validation seeds (:func:`static_errors`). Nothing is set from WM-CP's errors.

**Scenarios** (:data:`SCENARIOS`): ``alt``; ``null_shuffle`` (the alternative's residuals beyond the field permuted
across trajectories with random signs, as v1/v2); ``null_quadratic`` (A7's paired null: the quadratic world itself,
where WM-CP must not beat static). Both nulls enter the size and coverage gates. One SE vector (from the quadratic
h <= 2 outcomes) and one noise draw serve every scenario. Observations (tap losses and slopes) are made consistent
with the scenario's outcomes.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from wsp.wm import powersim_refit as R1
from wsp.wm import powersim_refit_v2 as R2
from wsp.wm import powersim_v1 as PV
from wsp.wm import signflip as SF
from wsp.wm import synthetic as S
from wsp.wm import wmcp as CP

Z0 = 1.0
STRENGTH_SIGMA = 0.5
CAL_TARGET = 2.17
SCENARIOS = ("alt", "null_shuffle", "null_quadratic")
NULLS = ("null_shuffle", "null_quadratic")
CAL_SEED0 = 70_000  # calibration replicates (kappa); validation uses rep = 0, 1, ...
# SYNTHESIS-r11 R6 / PREREG v2 §II.10 (flag 1): every arm shrunk toward the raw field on the same grid, split and loss,
# ties within 0.5 % (relative) to the smaller shrinkage, i.e. toward the exact field; applied to WM-CP and S1-S3 alike.
R6_TIE_REL = 0.005


def members_r6(cfg, spec, bo, ref_params=None):
    """The static family under R6: S1 is the F3u refit shrunk toward the field; S2/S3 keep their shrinkage with the R6
    tie rule. Same constructors otherwise (``powersim_refit_v2._members``)."""

    def s2():
        m = CP.StaticR(cfg, spec, bo, n_seeds=5, aux=False)
        m.shrink_tie_rel = R6_TIE_REL
        return m

    def s3():
        m = R2._s3(cfg, spec, bo, ref_params)
        m.shrink_tie_rel = R6_TIE_REL
        return m

    return {"S1": lambda: CP.ShrunkF3uR(spec, R6_TIE_REL), "S2": s2, "S3": s3}


def rho(z):
    return 1.0 / np.sqrt(1.0 + (np.asarray(z, float) / Z0) ** 2)


def strengths(n_comp: int, seed: int) -> np.ndarray:
    return np.exp(STRENGTH_SIGMA * np.random.default_rng([seed, 2027]).standard_normal(n_comp))


def departure(world, E, roots, s_c: float) -> np.ndarray:
    E = np.asarray(E, float)
    out = np.zeros(world.K)
    for u in roots:
        u = np.asarray(u, float)
        c = float(u @ E)
        if c == 0.0:
            continue
        out += S.raw_field(world, c * u) * (rho(c / world.step) - 1.0)
    return s_c * out


def alt_outcome(world, E, roots, s_c: float, kappa: float) -> np.ndarray:
    return S.raw_field(world, np.asarray(E, float)) + kappa * departure(world, E, roots, s_c)


def build_bank_v2b(world, g, seed):
    """``powersim_refit_v2.build_bank_v2`` with each trajectory's component roots recorded."""
    rng = np.random.default_rng(seed)
    fam_roots = {f: [] for f in PV.FAMILIES}
    for f in PV.FAMILIES:
        for _ in range(g["r_f"]):
            u = rng.standard_normal(world.J)
            fam_roots[f].append(u / np.linalg.norm(u))
    trajs = []
    for ci, c in enumerate(g["components"]):
        roots = [(f, fam_roots[f].pop()) for f in c["families"]]
        rs = [r for _f, r in roots]
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
                        roots=rs,
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
                        roots=rs,
                    )
                )
    return trajs


def _obs_delta(world, E, roots, s_c, kappa, eps=1e-5) -> np.ndarray:
    """Change of the tap observation implied by the departure: per-item losses (each item's loss moves with its
    objective's outcome in the quadratic world) and the action slopes (central differences)."""
    n_sk, n_it = len(world.probes) * world.P.shape[1], len(world.obs_items)
    d = np.zeros(n_sk + world.K * n_it + world.J * world.K)
    dep = kappa * departure(world, E, roots, s_c)
    d[n_sk : n_sk + world.K * n_it] = np.repeat(dep, n_it)
    G = np.zeros((world.J, world.K))
    for j in range(world.J):
        e = np.zeros(world.J)
        e[j] = eps
        G[j] = (
            kappa
            * (departure(world, E + e, roots, s_c) - departure(world, E - e, roots, s_c))
            / (2 * eps)
        )
    d[n_sk + world.K * n_it :] = G.reshape(-1)
    return d


def label(trajs, world, scenario, kappa, seed):
    """(labelled trajectories, se) for one scenario; ``trajs`` from :func:`build_bank_v2b` on a quadratic world."""
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    se = np.concatenate([t["y"][1:3] for t in trajs]).std(0) / 4.0 + 1e-12
    st = strengths(1 + max(t["comp"] for t in trajs), seed)
    alt = []
    for t in trajs:
        y, obs = t["y"].copy(), t["obs"].copy()
        if scenario != "null_quadratic":
            for h in range(1, t["horizon"] + 1):
                E = t["x"][h, : world.J]
                y[h] = y[h] + kappa * departure(world, E, t["roots"], st[t["comp"]])
                obs[h] = obs[h] + _obs_delta(world, E, t["roots"], st[t["comp"]], kappa)
        alt.append(dict(t, y=y, obs=obs))
    if scenario == "null_shuffle":
        alt = S.shuffle_outcomes(alt, seed, world)
    out = []
    for t in alt:
        noise = np.vstack(
            [np.zeros((1, len(se))), se * rng.standard_normal((t["horizon"], len(se)))]
        )
        out.append(dict(t, y=t["y"] + noise))
    return out, se


def _setup(rep, cell, scenario, kappa, seed):
    g = PV.graph(cell["r_f"], cell["n_blk"])
    world = S.SynthWorld(kind="quadratic", seed=10_000 + rep, n_dirs=R1.J_SHARED, n_obj=6)
    bo = world.base_observation().payload["obs"]
    raw = build_bank_v2b(world, g, seed + rep)
    trajs, se = label(raw, world, scenario, kappa, seed + 7 * rep)
    spec = R2.spec_with_dead(world)
    _cal, fold, rest = R1._alloc(g["components"], cell["n_cal"], seed + 13 * rep)
    return g, world, bo, trajs, se, spec, fold, rest, R2.states_of(world, trajs)


def static_errors(rep, cell, kappa, cfg=None, seed=PV.SEED) -> np.ndarray:
    """Calibration quantity: the selected static member's held-out SE-scaled |error| at h = 3-4 (all objectives),
    under the alternative. WM-CP enters only through its (untrained) parameter count for sizing S3."""
    torch.set_num_threads(1)
    cfg = cfg or CP.CPConfig()
    _g, _world, bo, _trajs, se, spec, fold, rest, allst = _setup(rep, cell, "alt", kappa, seed)
    out = []
    for k in range(R2.K_OUT):
        held = {c for c in rest if fold[c] == k}
        train = [s for s in allst if s["comp"] in rest and fold[s["comp"]] != k and s["h"] <= 2]
        ref = CP._count(
            CP._CPNet(spec.dim, len(bo), spec.n_obj, cfg)
        )  # WM-CP's size (architecture only)
        name, _ = R2.choose_static(cfg, spec, bo, train, se, seed + 31 * rep + k, ref)
        st = R2._members(cfg, spec, bo, ref)[name]().fit(train, [s["comp"] for s in train])
        test = [s for s in allst if s["comp"] in held and s["h"] in (3, 4)]
        if test:
            out.append(R2._err(st, test, se).ravel())
    return np.concatenate(out)


def cp_errors(rep, cell, cfg, seed=PV.SEED, world_kind="saturating") -> np.ndarray:
    """WM-CP config selection quantity (A7: synthetic data only, never the registered alternative): WM-CP's own
    held-out SE-scaled |error| at h = 3-4 on the native synthetic world (``SynthWorld(kind="saturating")``
    outcomes + noise, the v1/v2 refit world without any path term)."""
    torch.set_num_threads(1)
    g = PV.graph(cell["r_f"], cell["n_blk"])
    world = S.SynthWorld(kind=world_kind, seed=10_000 + rep, n_dirs=R1.J_SHARED, n_obj=6)
    bo = world.base_observation().payload["obs"]
    raw = R2.build_bank_v2(world, g, seed + rep)
    trajs, se = R1.label(raw, world, 1.0, seed + 7 * rep, False, 0.0)
    spec = R2.spec_with_dead(world)
    _cal, fold, rest = R1._alloc(g["components"], cell["n_cal"], seed + 13 * rep)
    allst = R2.states_of(world, trajs)
    out = []
    for k in range(R2.K_OUT):
        held = {c for c in rest if fold[c] == k}
        train = [s for s in allst if s["comp"] in rest and fold[s["comp"]] != k and s["h"] <= 2]
        cp = CP.CanonicalPathWM(cfg, spec, bo, n_seeds=5).fit(train, [s["comp"] for s in train])
        test = [s for s in allst if s["comp"] in held and s["h"] in (3, 4)]
        if test:
            out.append(R2._err(cp, test, se).ravel())
    return np.concatenate(out)


def replicate(
    rep, cell, scenario, kappa, cfg=None, n_flip=SF.N_FLIP, seed=PV.SEED, n_inert=12, r6=False
):
    """One full-refit replicate. ``r6=True`` applies R6 symmetric shrinkage (:data:`R6_TIE_REL`, :func:`members_r6`);
    the default reproduces the v2b pipeline exactly."""
    t0 = time.time()
    torch.set_num_threads(1)
    cfg = cfg or CP.CPConfig()
    _g, world, bo, trajs, se, spec, fold, rest, allst = _setup(rep, cell, scenario, kappa, seed)
    e_w, e_s, comps, chosen, inert_ok, gap = [], [], [], [], [], []
    med_static, shr_w, shr_s = [], [], []
    for k in range(R2.K_OUT):
        held = {c for c in rest if fold[c] == k}
        train = [s for s in allst if s["comp"] in rest and fold[s["comp"]] != k and s["h"] <= 2]
        cp = CP.CanonicalPathWM(cfg, spec, bo, n_seeds=5)
        if r6:
            cp.shrink_tie_rel = R6_TIE_REL
        cp.fit(train, [s["comp"] for s in train])
        ref = CP.param_count(cp)
        members = members_r6 if r6 else R2._members
        name, _ = R2.choose_static(
            cfg, spec, bo, train, se, seed + 31 * rep + k, ref, members=members
        )
        chosen.append(name)
        st = members(cfg, spec, bo, ref)[name]().fit(train, [s["comp"] for s in train])
        shr_w.append(float(cp.shrink))
        shr_s.append(float(getattr(st, "shrink", float("nan"))))
        test = [s for s in allst if s["comp"] in held and s["h"] in (3, 4)]
        if test:
            ew, es = R2._err(cp, test, se), R2._err(st, test, se)
            e_w.append(ew.ravel())
            e_s.append(es.ravel())
            med_static.append(es.ravel())
            comps.append(np.repeat([s["comp"] for s in test], ew.shape[1]))
        for t in [t for t in trajs if t["comp"] in held and t["kind"] == "ladder"][:1]:
            u = t["e"][0] / np.linalg.norm(t["e"][0])
            one = R1.traj_vec(
                world, [(u, float(np.linalg.norm(t["x"][3, : world.J]) / world.step))]
            )
            gap.append(
                float(
                    np.abs(
                        cp.predict(R2._r(world, one["x"][1, : world.J]))[0]
                        - cp.predict(R2._r(world, t["x"][3, : world.J]))[0]
                    ).max()
                )
            )
        s0 = cp.k_rule["s0"]
        ys, sg = [], []
        for _j in range(n_inert // 2):
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
        "scenario": scenario,
        "kappa": kappa,
        "r6": bool(r6),
        "shrink_wm": shr_w,
        "shrink_static": shr_s,
        "stat": r["stat"],
        "p": r["p"],
        "lb": r["lb"],
        "ct": r["ct"],
        "ct_lb": r["ct_lb"],
        "c2i": c2i,
        "decomposition_gap_max": max(gap or [0.0]),
        "reject": bool(r["reject"] and c2i),
        "static_chosen": chosen,
        "median_static_err": float(np.median(np.concatenate(med_static))),
        "median_wm_err": float(np.median(e_w)),
        "m": r["n_components"],
        "seconds": time.time() - t0,
    }


def summarise(rows) -> dict:
    out = {}
    for sc in SCENARIOS:
        rs = [r for r in rows if r["scenario"] == sc]
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
        if "median_static_err" in rs[0]:
            d["median_static_err"] = float(np.median([r["median_static_err"] for r in rs]))
        if sc in NULLS:
            d["coverage_at_zero"] = float(np.mean([r["lb"] <= 0 for r in rs]))
            d["pass"] = bool(
                d["rate"] <= PV.REQ["size_max"] and d["coverage_at_zero"] >= PV.REQ["coverage_min"]
            )
        else:
            d["pass"] = bool(d["power_lb"] >= PV.REQ["power_lb_min"])
        out[sc] = d
    return out
