"""Pilot-3 stage S rules and gates (SYNTHESIS-r10 D1, D2; RULING-wme-pilot3 §2). EXPLORATORY; CPU only.

A *step* record holds, per objective, the observed paired change ``obs`` with its SE ``se`` (validation items) and
the predictions of every rule: ``A0`` (static field), ``oracle`` (R2 trapezoid), ``S1`` (exact HVP), ``S2``
(per-item jets, with item SE ``s_hat``), ``S2a`` and the jet means ``J1``, ``J2``, ``X``; the oracle end slope
``R``; the step's ``h``, ``family``, ``dnorm``, ``control`` flag, ``split`` (construction / test / other) and
``cluster`` (sequence-pair id for the bootstrap).

* Cut of a rule = 1 - mean(|ŷ - obs|/se) / mean(|A0 - obs|/se); recovery = cut / oracle cut.
* D2 gate per objective with fresh oracle cut >= 0.20: pass if S1 or S2 recovers >= 0.50; kill if both < 0.25.
* S3 shrinkage and switch, P1 router, the static + residual rule and S5 are fitted on construction steps only.
"""

from __future__ import annotations

import numpy as np

ELIGIBLE_CUT = 0.20
PASS_RECOVERY = 0.50
KILL_RECOVERY = 0.25
SINGLE_RULES = ("A0", "S1", "S2", "RES")
Z95 = 1.96


def _err(st, obj, rule):
    r = st["obj"][obj]
    return abs(r[rule] - r["obs"]) / r["se"]


def cut(steps, obj, rule) -> float | None:
    steps = [s for s in steps if obj in s["obj"]]
    if not steps:
        return None
    a0 = np.mean([_err(s, obj, "A0") for s in steps])
    x = np.mean([_err(s, obj, rule) for s in steps])
    return float(1 - x / a0) if a0 > 0 else None


def pooled_cut(steps, objs, rule) -> float | None:
    e = [_err(s, o, rule) for s in steps for o in objs if o in s["obj"]]
    a = [_err(s, o, "A0") for s in steps for o in objs if o in s["obj"]]
    return float(1 - np.mean(e) / np.mean(a)) if a else None


def gate_steps(steps):
    return [s for s in steps if s["h"] >= 2 and not s["control"]]


def resolvable(steps, obj):
    return [s for s in steps if abs(s["obj"][obj]["obs"]) > Z95 * s["obj"][obj]["se"]]


def s_gate(steps, objectives) -> dict:
    """D2 gate per objective on fresh h >= 2 non-control steps."""
    g = gate_steps(steps)
    per = {}
    for o in objectives:
        oc = cut(g, o, "oracle")
        row = {"n_steps": len([s for s in g if o in s["obj"]]), "oracle_cut": oc}
        for r in ("S1", "S2", "S2a"):
            c = cut(g, o, r)
            row[f"{r}_cut"] = c
            row[f"{r}_recovery"] = (c / oc) if (c is not None and oc) else None
        row["eligible"] = bool(oc is not None and oc >= ELIGIBLE_CUT)
        rec = [row["S1_recovery"], row["S2_recovery"]]
        if not row["eligible"]:
            row["verdict"] = "not eligible (oracle cut < 0.20)"
        elif any(x is not None and x >= PASS_RECOVERY for x in rec):
            row["verdict"] = "PASS"
        elif all(x is None or x < KILL_RECOVERY for x in rec):
            row["verdict"] = "KILL"
        else:
            row["verdict"] = "inconclusive"
        rs = resolvable(g, o)
        row["resolvable"] = {
            "n": len(rs),
            **{f"{r}_cut": cut(rs, o, r) for r in ("oracle", "S1", "S2")},
        }
        per[o] = row
    passed = [o for o, r in per.items() if r["verdict"] == "PASS"]
    eligible = [o for o, r in per.items() if r["eligible"]]
    if not eligible:
        cons = "no objective reaches an oracle cut >= 0.20: no objective can pass; slope transport is dropped"
    elif not passed:
        killed = [o for o in eligible if per[o]["verdict"] == "KILL"]
        inconc = [o for o in eligible if per[o]["verdict"] == "inconclusive"]
        cons = (
            f"S1 and S2 pass on no eligible objective (killed: {killed}; inconclusive, 25-50 % recovery: {inconc}): "
            "slope transport is dropped (SYNTHESIS-r9 C4); the per-state oracle gradients leave the bank"
        )
    else:
        cons = f"transport kept only for {passed}; every other objective uses static plus a learned residual on WM-CP"
    pooled = {
        r: {
            "pooled": pooled_cut(g, objectives, r),
            "pooled_without_harmful": pooled_cut(
                g, [o for o in objectives if o != "refusal_margin_harmful"], r
            ),
        }
        for r in ("oracle", "S1", "S2", "S2a")
    }
    return {
        "per_objective": per,
        "passed": passed,
        "eligible": eligible,
        "consequence": cons,
        "pooled": pooled,
    }


# ------------------------------------------------------------------ fitted rules (construction steps only)


def fit_residual(cons, objectives) -> dict:
    """Static + learned residual: obs = c0 + c1 * A0 per objective (least squares on construction steps)."""
    out = {}
    for o in objectives:
        x = np.array([s["obj"][o]["A0"] for s in cons if o in s["obj"]])
        y = np.array([s["obj"][o]["obs"] for s in cons if o in s["obj"]])
        X = np.c_[np.ones_like(x), x]
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[o] = [float(c[0]), float(c[1])]
    return out


def add_residual(steps, coef):
    for s in steps:
        for o, r in s["obj"].items():
            c0, c1 = coef[o]
            r["RES"] = c0 + c1 * r["A0"]


def s3_weight(r) -> float:
    d = r["S2"] - r["A0"]
    s2 = r["s_hat"] ** 2
    return max(0.0, 1.0 - s2 / (d * d + s2)) if (d * d + s2) > 0 else 0.0


def fit_s3(cons, objectives) -> dict:
    """Per-objective switch: on only if the construction-step cut of the shrunk S2 rule vs static is > 0."""
    for s in cons:
        for r in s["obj"].values():
            r["S3on"] = r["A0"] + s3_weight(r) * (r["S2"] - r["A0"])
    return {o: bool((cut(cons, o, "S3on") or 0) > 0) for o in objectives}


def add_s3(steps, switch):
    for s in steps:
        for o, r in s["obj"].items():
            r["S3on"] = r["A0"] + s3_weight(r) * (r["S2"] - r["A0"])
            r["S3"] = r["S3on"] if switch.get(o) else r["A0"]


def fit_p1(cons, objectives) -> dict:
    """Router: per (objective, family) the single rule with the lowest construction mean |err|/SE (ties -> A0)."""
    table = {}
    fams = sorted({s["family"] for s in cons})
    for o in objectives:
        for f in fams:
            ss = [s for s in cons if s["family"] == f and o in s["obj"]]
            if not ss:
                continue
            errs = {r: float(np.mean([_err(s, o, r) for s in ss])) for r in SINGLE_RULES}
            best = min(SINGLE_RULES, key=lambda r: (errs[r], SINGLE_RULES.index(r)))
            table[f"{o}|{f}"] = best
    return table


def add_p1(steps, table):
    for s in steps:
        for o, r in s["obj"].items():
            r["P1"] = r[table.get(f"{o}|{s['family']}", "A0")]


S5_FAMILIES = ("scale", "steer", "proj_remove")


def _s5_x(s, r):
    fam = [1.0 if s["family"] == f else 0.0 for f in S5_FAMILIES]
    return [1.0, r["J1"], r["X"], r["J2"], s["dnorm"], float(s["h"]), *fam, r["s_hat"]]


def fit_s5(cons, objectives, lam=1.0) -> dict:
    """Ridge (λ = 1, intercept unpenalised) of r = R - (J1 + X + J2) on S5 features, weights 1/SE²."""
    out = {}
    for o in objectives:
        ss = [s for s in cons if o in s["obj"]]
        X = np.array([_s5_x(s, s["obj"][o]) for s in ss])
        y = np.array(
            [
                s["obj"][o]["R"] - (s["obj"][o]["J1"] + s["obj"][o]["X"] + s["obj"][o]["J2"])
                for s in ss
            ]
        )
        w = np.array([1.0 / s["obj"][o]["se"] ** 2 for s in ss])
        mu = X[:, 1:].mean(0)
        sd = X[:, 1:].std(0) + 1e-12
        Xs = np.c_[np.ones(len(X)), (X[:, 1:] - mu) / sd]
        P = np.eye(Xs.shape[1]) * lam
        P[0, 0] = 0.0
        W = np.diag(w / w.mean())
        beta = np.linalg.solve(Xs.T @ W @ Xs + P, Xs.T @ W @ y)
        out[o] = {"beta": beta.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}
    return out


def add_s5(steps, model):
    for s in steps:
        for o, r in s["obj"].items():
            m = model[o]
            x = np.array(_s5_x(s, r))
            xs = np.r_[1.0, (x[1:] - np.array(m["mu"])) / np.array(m["sd"])]
            rhat = float(xs @ np.array(m["beta"]))
            L = r["J1"] + r["X"]
            R1 = r["J1"] + r["X"] + r["J2"]
            r["S5"] = 0.5 * (L + R1 + rhat)


def cluster_lb(steps, objs, better, worse, n_boot=4000, seed=0) -> dict:
    """Mean of (|err_worse| - |err_better|)/SE over (step, objective), cluster bootstrap 2.5 % lower bound."""
    rows = [
        (s["cluster"], _err(s, o, worse) - _err(s, o, better))
        for s in steps
        for o in objs
        if o in s["obj"]
    ]
    if not rows:
        return {"n": 0, "mean": None, "lb": None, "beats": False}
    cl = sorted({c for c, _ in rows})
    by = {c: [d for k, d in rows if k == c] for c in cl}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(cl), len(cl))
        vals = [d for i in pick for d in by[cl[i]]]
        bs.append(np.mean(vals))
    mean = float(np.mean([d for _, d in rows]))
    lb = float(np.quantile(bs, 0.025))
    return {"n": len(rows), "mean": mean, "lb": lb, "beats": bool(lb > 0)}


def fitted_gates(steps, objectives, eligible) -> dict:
    """Fit on construction steps, evaluate on test h >= 2 non-control steps (P1, S3, S5; RULING §2.3)."""
    cons = [s for s in steps if s["split"] == "construction" and not s["control"]]
    test = [s for s in gate_steps(steps) if s["split"] == "test"]
    res = fit_residual(cons, objectives)
    add_residual(steps, res)
    sw = fit_s3(cons, objectives)
    add_s3(steps, sw)
    table = fit_p1(cons, objectives)
    add_p1(steps, table)
    s5 = fit_s5(cons, objectives)
    add_s5(steps, s5)
    out = {
        "n_construction": len(cons),
        "n_test": len(test),
        "residual": res,
        "s3_switch": sw,
        "p1_table": table,
    }
    # P1 must beat every single global rule (so the best one) with lower bound > 0
    vs = {r: cluster_lb(test, objectives, "P1", r) for r in SINGLE_RULES}
    out["P1"] = {"vs": vs, "pass": bool(test) and all(v["beats"] for v in vs.values())}
    # S3: no objective worse than static by > 0.05 in cut; pooled cut >= S2's
    per = {o: cut(test, o, "S3") for o in objectives}
    worst_ok = all(c is None or c >= -0.05 for c in per.values())
    p3, p2 = pooled_cut(test, objectives, "S3"), pooled_cut(test, objectives, "S2")
    out["S3"] = {
        "per_objective_cut": per,
        "pooled_cut": p3,
        "S2_pooled_cut": p2,
        "pass": bool(test) and worst_ok and (p3 is not None and p2 is not None and p3 >= p2),
        "kill": not worst_ok,
    }
    # S5: beats S2 (LB > 0) and >= 25 % of the oracle cut, per eligible objective
    s5 = {}
    for o in eligible:
        oc = cut(test, o, "oracle")
        c5 = cut(test, o, "S5")
        b = cluster_lb(test, [o], "S5", "S2")
        s5[o] = {
            "oracle_cut": oc,
            "S5_cut": c5,
            "vs_S2": b,
            "pass": bool(b["beats"] and oc is not None and c5 is not None and c5 >= 0.25 * oc),
        }
    out["S5"] = {
        "per_eligible_objective": s5,
        "pooled_cut": pooled_cut(test, objectives, "S5"),
        "pass": bool(s5) and all(v["pass"] for v in s5.values()),
    }
    out["test_cuts"] = {
        o: {r: cut(test, o, r) for r in ("oracle", "S1", "S2", "RES", "S3", "P1", "S5")}
        for o in objectives
    }
    return out
