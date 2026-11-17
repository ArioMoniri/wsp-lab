"""Full-refit validation of the selected simulation cell (RULING-wme-resources §4(b); Board A condition 11:
"simulate the actual nested pipeline"). CPU only; synthetic worlds; no real outcome.

Per replicate, under the H-T null and alternative:

1. **World and bank.** A fresh ``SynthWorld`` (saturating, 6 objectives, J = 6 r_f dictionary directions, seed =
   replicate). Roots are directions; families label them (r_f each). The registered v1 graph (``powersim_v1.graph``):
   ``n_blk`` blocks of 4 roots from 4 families, the rest singletons. Ladders: both sides, cumulative doses
   0.25, 0.5, 1, 2 (h = 1..4; the far horizon is not needed for H-T). Compositions: each block's order and its
   reverse at dose 1 (h = 1..4).
2. **Labels.** One measured label per state: y + N(0, se_k^2), se_k = sd_k(y at h <= 2) / 4 (assumption). **Null**:
   residuals over the shared R0 field are permuted across trajectories of equal horizon with random signs
   (``synthetic.shuffle_outcomes``), so neither arm can learn beyond the field. **Alternative**: residual =
   kappa x true + sqrt(1 - kappa^2) x shuffled, kappa calibrated so the mean C_T is near 0.15 (``calibrate``).
3. **Allocation.** ``n_cal`` calibration components drawn first (stratified) and excluded; five outer folds by
   component, stratified by stratum; four inner folds by component.
4. **Per outer fold.** Training: h <= 2 prefixes of training components only. Static family S1 (F3u least squares),
   S2 (StatNN), S3 (MatchedStatic: WM-E's network without recurrence, the same auxiliary targets); chosen on the 4
   inner folds by pooled scaled error at h = 2 (0.5 % relative tie -> lower capacity S1 < S2 < S3), refitted on all
   outer-training components. WM-E (``WMEnsemble``, 5 members, component bootstrap; its shrinkage uses the code's
   component selection split, a disclosed difference from the registered inner folds).
5. **Statistic.** Scaled errors |y_hat - y_obs| / se_k at h in {3, 4} on held-out components; C_T with the
   stratified component bootstrap-t (R = 9,999, one-sided 0.025); C2-S (all h! - 1 permutations of the step-local
   inputs on held-out compositions, cumulative inputs fixed; component bootstrap LB > 0); C2-I (I1 round trips
   +d, -d on a training direction: mean |y_hat_k| <= se_k and the 2-sigma part). Reject iff LB > 0, C2-S and C2-I.
"""

from __future__ import annotations

import itertools
import json
import math
import time

import numpy as np
import torch

from wsp.wm import model as M
from wsp.wm import powersim_v1 as PV
from wsp.wm import synthetic as S

LADDER_CUM = (0.25, 0.5, 1.0, 2.0)
K_OUT, K_IN = 5, 4
TIE_REL = 0.005


def _increments(cum):
    out, prev = [], 0.0
    for c in cum:
        out.append(c - prev)
        prev = c
    return out


def prefix(t, h):
    return dict(
        t, horizon=h, e=t["e"][:h], x=t["x"][: h + 1], y=t["y"][: h + 1], obs=t["obs"][: h + 1]
    )


J_SHARED = 8


def traj_vec(world, actions):
    """Trajectory for actions [(unit coefficient vector u, dose)] in the world's shared J-dim coefficient space:
    the same layout as ``SynthWorld.trajectory_from_actions`` (exact bookkeeping, outcomes, observations)."""
    J = world.J
    Ed = world.E.reshape(J, -1).numpy()
    coef = torch.zeros(J, dtype=torch.float64)
    ys, obs = [world.outcomes(coef).numpy()], [world.observe(coef)]
    T = len(actions)
    x = np.zeros((T + 1, 2 * J + 3))
    e = np.zeros((T, J))
    E = np.zeros(J)
    for t, (u, dose) in enumerate(actions, start=1):
        D = dose * world.step * np.asarray(u, float)
        prevE, Dm = Ed.T @ E, Ed.T @ D
        den = np.linalg.norm(prevE) * np.linalg.norm(Dm)
        E = E + D
        x[t, :J], x[t, J : 2 * J], x[t, -3] = E, D, t
        x[t, -2] = float(prevE @ Dm / den) if den > 0 else 0.0
        x[t, -1] = np.linalg.norm(Ed.T @ E)
        e[t - 1] = D
        coef = torch.as_tensor(E, dtype=torch.float64)
        ys.append(world.outcomes(coef).numpy())
        obs.append(world.observe(coef))
    y = np.stack(ys)
    return {"e": e, "x": x, "y": y - y[0], "obs": np.stack(obs), "horizon": T}


def build_bank(world, g, seed):
    """Trajectories (full length 4) with component ids and strata. Each root is a random unit direction in the
    world's shared J-dim coefficient space (the analogue of the registered shared representation R(E): held-out
    roots are new directions in a space the predictors were trained on)."""
    rng = np.random.default_rng(seed)
    comps = g["components"]
    fam_roots = {f: [] for f in PV.FAMILIES}
    for f in PV.FAMILIES:
        for _ in range(g["r_f"]):
            u = rng.standard_normal(world.J)
            fam_roots[f].append(u / np.linalg.norm(u))
    trajs = []
    for ci, c in enumerate(comps):
        roots = [fam_roots[f].pop() for f in c["families"]]
        for r in roots:
            for s in (1.0, -1.0):
                acts = [(r, s * d) for d in _increments(LADDER_CUM)]
                t = traj_vec(world, acts)
                trajs.append(dict(t, comp=ci, stratum=c["stratum"], kind="ladder"))
        if c["stratum"] == "block":
            order = list(range(len(roots)))
            rng.shuffle(order)
            for seq in (order, order[::-1]):
                t = traj_vec(world, [(roots[i], 1.0) for i in seq])
                trajs.append(dict(t, comp=ci, stratum="block", kind="composition"))
    return trajs


def path_term(world, trajs, seed):
    """Decomposition-dependent effect sum_s tanh(<w_k, D_s> / tau_k) (unit sd per objective at h <= 2): a function of
    the step sequence, not of the cumulative edit, so a static predictor of E_T cannot represent it."""
    rng = np.random.default_rng([seed, 99])
    W = rng.standard_normal((world.J, world.K))
    raw = [
        np.vstack([np.zeros((1, world.K)), np.cumsum(np.tanh(t["e"] @ W / 0.3), 0)]) for t in trajs
    ]
    sd = np.concatenate([r[1:3] for r in raw]).std(0) + 1e-12
    return [r / sd for r in raw]


def label(trajs, world, kappa, seed, null, gamma=0.0):
    """``gamma``: size of the path effect under the alternative, in units of the per-state SE."""
    rng = np.random.default_rng(seed)
    y2 = np.concatenate([t["y"][1:3] for t in trajs])
    se = y2.std(0) / 4.0 + 1e-12
    if gamma and not null:
        pt = path_term(world, trajs, seed)
        trajs = [dict(t, y=t["y"] + gamma * se * p) for t, p in zip(trajs, pt, strict=True)]
    if null or kappa < 1.0:
        shuf = S.shuffle_outcomes(trajs, seed, world)
    out = []
    for i, t in enumerate(trajs):
        if null:
            y = shuf[i]["y"].copy()
            obs = shuf[i]["obs"]
        elif kappa < 1.0:
            f = S.raw_field(world, t["x"][:, : world.J])
            y = f + kappa * (t["y"] - f) + math.sqrt(1 - kappa**2) * (shuf[i]["y"] - f)
            obs = t["obs"]
        else:
            y, obs = t["y"].copy(), t["obs"]
        y = y + np.vstack([np.zeros((1, len(se))), se * rng.standard_normal((len(y) - 1, len(se)))])
        out.append(dict(t, y=y, obs=obs))
    return out, se


def _alloc(comps, n_cal, seed):
    rng = np.random.default_rng(seed)
    strata = [c["stratum"] for c in comps]
    by = {}
    for i in rng.permutation(len(comps)):
        by.setdefault(strata[i], []).append(int(i))
    cal = []
    while len(cal) < n_cal:
        for s in sorted(by):
            if by[s] and len(cal) < n_cal:
                cal.append(by[s].pop(0))
    rest = [i for s in sorted(by) for i in by[s]]
    fold = {}
    k = 0
    for s in sorted(by):
        for i in by[s]:
            fold[i] = k % K_OUT
            k += 1
    return set(cal), fold, rest


def _train_set(trajs, comps):
    out, cids = [], []
    for t in trajs:
        if t["comp"] in comps:
            for h in (1, 2):
                out.append(prefix(t, h))
                cids.append(t["comp"])
    return out, cids


def _static_members(cfg, world):
    f = world.field()
    bo = world.base_observation()
    return {
        "S1": lambda: _F3uMember(f),
        "S2": lambda: M.StaticPredictor(cfg, f, n_seeds=5),
        "S3": lambda: M.MatchedStatic(cfg, f, bo, n_seeds=5),
    }


class _F3uMember:
    def __init__(self, field):
        self.f3u = M.F3u(field.payload)

    def fit(self, trajs, components=None):
        M.refit_f3u(self.f3u, trajs)
        return self

    def predict(self, base_obs, traj):
        return self.f3u(M.batch([traj])["E"][0]), None


def _pt(model, bo, t):
    p = model.predict(bo, t)
    return p[0] if isinstance(p, tuple) else p


def _err(model, bo, units, se):
    return np.array(
        [np.abs(_pt(model, bo, t)[t["horizon"]] - t["y"][t["horizon"]]) / se for t in units]
    )


def choose_static(cfg, world, train, cids, se, seed):
    comps = sorted(set(cids))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(comps))
    inner = {c: i % K_IN for i, c in enumerate(order)}
    mk = _static_members(cfg, world)
    bo = world.base_observation()
    loss = {}
    for name, ctor in mk.items():
        tot = 0.0
        for k in range(K_IN):
            fit = [t for t, c in zip(train, cids, strict=True) if inner[c] != k]
            fc = [c for c in cids if inner[c] != k]
            val = [
                t for t, c in zip(train, cids, strict=True) if inner[c] == k and t["horizon"] == 2
            ]
            if not val or not fit:
                continue
            m = ctor().fit(fit, fc)
            tot += float(_err(m, bo, val, se).sum())
        loss[name] = tot
    best = min(loss.values())
    for name in ("S1", "S2", "S3"):
        if loss[name] <= best * (1 + TIE_REL):
            return name, loss
    return min(loss, key=loss.get), loss


def _permuted(t, perm, J):
    """Step-local inputs (D_t block, overlap, e_t) reordered by ``perm``; cumulative E, norm, position fixed."""
    x = t["x"].copy()
    e = t["e"].copy()
    h = t["horizon"]
    src = [p + 1 for p in perm]
    x[1 : h + 1, J : 2 * J] = t["x"][src, J : 2 * J]
    x[1 : h + 1, -2] = t["x"][src, -2]
    e[:h] = t["e"][[p for p in perm]]
    return dict(t, x=x, e=e)


def replicate(
    rep, cell, null, kappa=1.0, cfg=None, n_boot=PV.N_BOOT, seed=PV.SEED, n_inert=12, gamma=0.0
):
    t0 = time.time()
    torch.set_num_threads(1)
    cfg = cfg or M.WMConfig()
    g = PV.graph(cell["r_f"], cell["n_blk"])
    world = S.SynthWorld(kind="saturating", seed=10_000 + rep, n_dirs=J_SHARED, n_obj=6)
    bo = world.base_observation()
    raw = build_bank(world, g, seed + rep)
    trajs, se = label(raw, world, kappa, seed + 7 * rep, null, gamma)
    _cal, fold, rest = _alloc(g["components"], cell["n_cal"], seed + 13 * rep)
    A = np.zeros(len(g["components"]))
    Sx = np.zeros(len(g["components"]))
    perm_rows = []
    chosen = []
    inert_y = []
    J = world.J
    for k in range(K_OUT):
        held = {c for c in rest if fold[c] == k}
        trn = {c for c in rest if fold[c] != k}
        train, cids = _train_set(trajs, trn)
        name, _ = choose_static(cfg, world, train, cids, se, seed + 31 * rep + k)
        chosen.append(name)
        st = _static_members(cfg, world)[name]().fit(train, cids)
        wm = M.WMEnsemble(cfg, world.field(), bo, n_seeds=5).fit(train, cids)
        for t in trajs:
            if t["comp"] not in held:
                continue
            for h in (3, 4):
                u = prefix(t, h)
                ew = np.abs(_pt(wm, bo, u)[h] - u["y"][h]) / se
                es = np.abs(_pt(st, bo, u)[h] - u["y"][h]) / se
                A[t["comp"]] += ew.sum()
                Sx[t["comp"]] += es.sum()
                if t["kind"] == "composition":
                    ps = [p for p in itertools.permutations(range(h)) if list(p) != list(range(h))]
                    ep = np.mean(
                        [np.abs(_pt(wm, bo, _permuted(u, p, J))[h] - u["y"][h]) / se for p in ps], 0
                    )
                    perm_rows.append((t["comp"], float(ep.sum() - ew.sum())))
        # C2-I: I1 round trips on training directions (+d then -d: E_T = 0, history nonzero)
        dirs = [
            t["e"][0] / np.linalg.norm(t["e"][0])
            for t in trajs
            if t["comp"] in trn and t["kind"] == "ladder"
        ]
        for d in dirs[: max(1, n_inert // (2 * K_OUT))]:
            tr = traj_vec(world, [(d, 1.0), (d, -1.0)])
            m, s = wm.predict(bo, tr)
            inert_y.append((m[2] / se, s[2] / se))
    keep = [c for c in rest]
    strata = [g["components"][c]["stratum"] for c in keep]
    res = PV.ct_bootstrap_t(A[keep], Sx[keep], strata, n_boot=n_boot, seed=seed + rep)
    if perm_rows:
        cc = sorted({c for c, _ in perm_rows})
        tot = np.array([sum(v for c2, v in perm_rows if c2 == c) for c in cc])
        se_d = tot.std(ddof=1) / math.sqrt(len(tot)) if len(tot) > 1 else 0.0
        c2s = bool(len(tot) >= 5 and se_d > 0 and tot.mean() - PV.Z * se_d > 0)
        c2s_mean = float(tot.mean())
    else:
        c2s, c2s_mean = False, float("nan")
    yy = np.array([y for y, _ in inert_y])
    ss = np.array([s for _, s in inert_y])
    part1 = float(np.mean(np.abs(yy) > 2 * np.maximum(ss, 1.0))) <= 0.10
    part2 = bool(np.all(np.abs(yy).mean(0) <= 1.0))
    c2i = bool(part1 and part2)
    return {
        "rep": rep,
        "null": null,
        "kappa": kappa,
        "gamma": gamma,
        "ct": res["ct"],
        "lb": res["lb"],
        "estimable": res["estimable"],
        "c2s": c2s,
        "c2s_mean": c2s_mean,
        "c2i": c2i,
        "reject": bool(res["estimable"] and res["lb"] > 0 and c2s and c2i),
        "static_chosen": chosen,
        "m": len(keep),
        "seconds": time.time() - t0,
    }


def summarise(rows) -> dict:
    out = {}
    for null in (True, False):
        rs = [r for r in rows if r["null"] == null]
        if not rs:
            continue
        k = sum(r["reject"] for r in rs)
        n = len(rs)
        d = {
            "reps": n,
            "rate": k / n,
            "ci95": PV.clopper_pearson(k, n),
            "power_lb": PV.cp_lower_one_sided(k, n),
            "mean_ct": float(np.nanmean([r["ct"] for r in rs])),
            "c2s_pass": float(np.mean([r["c2s"] for r in rs])),
            "c2i_pass": float(np.mean([r["c2i"] for r in rs])),
            "mean_seconds": float(np.mean([r["seconds"] for r in rs])),
        }
        if null:
            miss = sum(0.0 < r["lb"] for r in rs if r["estimable"])
            d["coverage_at_zero"] = 1 - miss / max(1, sum(r["estimable"] for r in rs))
            d["pass"] = (
                d["rate"] <= PV.REQ["size_max"] and d["coverage_at_zero"] >= PV.REQ["coverage_min"]
            )
        else:
            d["pass"] = d["power_lb"] >= PV.REQ["power_lb_min"]
        out["null" if null else "alt"] = d
    return out


if __name__ == "__main__":
    import sys

    cell = json.loads(sys.argv[1])
    g = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    print(
        json.dumps(
            replicate(int(sys.argv[2]), cell, sys.argv[3] == "null", float(sys.argv[4]), gamma=g)
        )
    )
