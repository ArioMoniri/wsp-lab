"""WM-E pilot-2 design rules and gates, fixed by ``docs/council/2026-11-16/RULING-wme-pilot1.md``. EXPLORATORY.

Implementation choices where the ruling's text needs one (each written here before any pilot-2 data existed):

* **Ladders.** Signed cumulative doses ±{0.25, 0.5, 1, 2, 4} × nominal, each built from W0. Scale: factor 1.5,
  W(s) = [1 + s(f - 1)] W0. Steer: signed shift s × 0.125 × mean residual norm. proj_remove: alpha_eff = s × 0.5,
  capped at alpha_eff <= 1 (the ruling's "alpha <= 1"; negative alpha is not capped). A capped dose that equals an
  earlier state's alpha (+4 -> alpha 2 -> 1 = the +2 state) is the same weights and is not visited again.
* **Stop rule.** Each signed side is walked outward; a state whose paired mean WikiText ΔNLL (vs base) exceeds
  1.0 nats/token is kept, labelled ``dnll>1.0``, and is that side's last state.
* **Compositions.** Over the six non-control roots (two per family): eight 3-step sequences with one root per
  family, no sequence the reverse of another, then each reverse (16); four 4-step sequences of four distinct roots
  covering all three families, then each reverse (8). Seeded from the ``wme-p2`` namespace. Each step is the
  root's nominal dose (1), applied sequentially; the negative control is never composed.
* **Steps.** Ladder steps chain consecutive visited doses from the base on each side; composition steps go from
  the h-1 prefix (h = 1 is the root's +1 ladder state) to the h prefix. A step is a **control** step if any root
  in its target state is a control root. Controls are reported separately and excluded from the Go-T2 count.
* **Go-T2.** (i) fraction of (non-control step, T2 objective) pairs with |paired mean| >= 2 paired SE is >= 0.50;
  (ii) the pilot-1 aggregate: F3u (y = c1 a + c2 q, least squares per objective over all ladder states, as pilot 1)
  static error |observed - F3u| / paired SE, median pooled over every (composition state with h >= 3, objective)
  pair of the six objectives, > 2.
* **Go-R2.** Over composition steps with h >= 2: mean over (step, objective) of |observed - trapezoid| / step SE
  vs the same for the static base-slope step F(E_t) - F(E_{t-1}) (A0); pass when the cut is >= 20 %. Trapezoid
  = ½(<g(θ_{t-1}), D_t> + <g(θ_t), D_t>), the pilot-1 oracle_trapezoid rule.
* **Go-E2.** Per objective, 32 candidates: its own rank-8 LoRA task vector at λ ∈ {0.25, 0.5, 1} plus 29 single
  rank-one steering actions sampled without replacement from the late-layer primitive set (layers >= 20, |shift| <=
  0.0625 × residual norm). A candidate qualifies when its dev gain 95 % LCB (mean - 1.96 SE, oriented) > 0 and no
  guardrail is breached: WikiText ΔNLL UCB <= 0.01; every other guarded objective's paired change on the
  probability ("pp equivalent", ``wm.panel2.pp_value``) scale has LCB >= -0.01 when higher is better, UCB <= +0.01
  when lower is better. The objective itself and the descriptive MMLU-Pro proxy are not guardrails. An objective
  passes with >= 2 qualifying of 32; the primary is the passing objective with most qualifying, ties (iii) > (ii) > (i).
"""

from __future__ import annotations

import hashlib
import itertools
import random

import numpy as np

from wsp.wm.p2data import seed_for

LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)
NOMINAL = {"scale": 1.5, "steer": 0.125, "proj_remove": 0.5}
ALPHA_CAP = 1.0
DNLL_STOP = 1.0
Z95 = 1.96
E2_N = 32
E2_LAMBDAS = (0.25, 0.5, 1.0)
E2_MAX_SHIFT = 0.0625
E2_MIN_LAYER = 20
GO_T2_RESOLVABLE = 0.50
GO_T2_STATIC = 2.0
GO_R2_CUT = 0.20
GO_E2_MIN = 2
OBJECTIVE_TIE_ORDER = ("mathcomp_gold_lp", "gsm8k_gold_lp", "sycophancy_real")  # (iii) > (ii) > (i)
# guardrails: (scale, bound, limit); "value" = the objective's own scale, "pp" = wm.panel2.pp_value scale
GUARDS = {
    "wikitext_nll": ("value", "ucb", 0.01),
    "gsm8k_gold_lp": ("pp", "lcb", -0.01),
    "mathcomp_gold_lp": ("pp", "lcb", -0.01),
    "mmlupro_gold_lp": ("pp", "lcb", -0.01),
    "refusal_margin_harmful": ("pp", "lcb", -0.01),
    "refusal_margin_harmless": ("pp", "ucb", 0.01),
    "sycophancy_real": ("pp", "ucb", 0.01),
}
NOT_GUARDS = ("sycophancy_gap",)  # descriptive proxy
ORIENT = {
    "sycophancy_real": -1.0,
    "gsm8k_gold_lp": 1.0,
    "mathcomp_gold_lp": 1.0,
}

__all__ = ["seed_for"]


# ------------------------------------------------------------------ ladders


def effective_param(family: str, dose: float) -> tuple[float, bool]:
    """(parameter, capped): scale -> factor f (W(s) uses s, f); steer -> signed residual-norm fraction;
    proj_remove -> alpha_eff = dose × 0.5 capped at <= 1."""
    if family == "scale":
        return NOMINAL["scale"], False
    if family == "steer":
        return dose * NOMINAL["steer"], False
    if family in ("proj_remove", "control"):
        a = dose * NOMINAL["proj_remove"]
        return (ALPHA_CAP, True) if a > ALPHA_CAP else (a, False)
    raise KeyError(family)


def ladder_sides(family: str) -> tuple[list[float], list[float]]:
    """Doses to visit on the positive and negative side, dropping capped duplicates of an earlier state."""
    sides = []
    for sign in (1.0, -1.0):
        seen, out = set(), []
        for m in LADDER:
            d = sign * m
            p, _ = effective_param(family, d)
            key = (p, d) if family == "scale" else p
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        sides.append(out)
    return sides[0], sides[1]


def walk_ladder(doses, evaluate) -> list[dict]:
    """Visit ``doses`` in order; ``evaluate(dose)`` returns the paired mean WikiText ΔNLL vs base. The first state
    above ``DNLL_STOP`` is kept as a labelled endpoint and the side stops there."""
    out = []
    for d in doses:
        dn = float(evaluate(d))
        stop = dn > DNLL_STOP
        out.append({"dose": d, "dnll": dn, "endpoint": stop, "label": "dnll>1.0" if stop else None})
        if stop:
            break
    return out


# ------------------------------------------------------------------ compositions


def make_compositions(families: dict, seed: int):
    """(sixteen 3-step, eight 4-step) sequences of non-control roots; each followed by its reverse."""
    by_fam = {}
    for r, f in sorted(families.items()):
        if f != "control":
            by_fam.setdefault(f, []).append(r)
    fams = sorted(by_fam)
    if fams != ["proj_remove", "scale", "steer"]:
        raise ValueError(f"need scale, steer and proj_remove roots, got {fams}")
    roots = sorted(r for f in fams for r in by_fam[f])
    rng = random.Random(seed)
    triples = [
        p
        for combo in itertools.product(*(by_fam[f] for f in fams))
        for p in itertools.permutations(combo)
    ]
    rng.shuffle(triples)
    c3 = []
    for t in triples:
        if t in c3 or tuple(reversed(t)) in c3:
            continue
        c3.append(t)
        if len(c3) == 8:
            break
    quads = [
        p
        for combo in itertools.combinations(roots, 4)
        if {families[r] for r in combo} == set(fams)
        for p in itertools.permutations(combo)
    ]
    rng.shuffle(quads)
    c4 = []
    for q in quads:
        if q in c4 or tuple(reversed(q)) in c4:
            continue
        c4.append(q)
        if len(c4) == 4:
            break
    with_rev = lambda cs: [x for c in cs for x in (c, tuple(reversed(c)))]
    return with_rev(c3), with_rev(c4)


def composition_states(seqs) -> list[dict]:
    out, seen = [], set()
    for s in seqs:
        for h in range(2, len(s) + 1):
            sid = "comp:" + ">".join(s[:h])
            if sid in seen:
                continue
            seen.add(sid)
            out.append(
                {"id": sid, "kind": "composition", "h": h, "steps": [(r, 1.0) for r in s[:h]]}
            )
    return out


def ladder_id(root: str, dose: float) -> str:
    return "base" if dose == 0 else f"ladder:{root}:{dose:+g}"


def comp_prev(sid: str) -> str:
    parts = sid.split(":", 1)[1].split(">")
    return ladder_id(parts[0], 1.0) if len(parts) == 2 else "comp:" + ">".join(parts[:-1])


def state_roots(sid: str) -> list[str]:
    if sid == "base":
        return []
    if sid.startswith("ladder:"):
        return [sid.split(":")[1]]
    return sid.split(":", 1)[1].split(">")


def steps_list(visited: dict, comp_ids, families: dict) -> list[dict]:
    """``visited[root] = {"pos": [doses...], "neg": [...]}`` (visited, in order)."""
    steps = []
    for root in sorted(visited):
        for side in ("pos", "neg"):
            prev = "base"
            for d in visited[root][side]:
                cur = ladder_id(root, d)
                steps.append({"kind": "ladder", "from": prev, "to": cur})
                prev = cur
    for sid in comp_ids:
        steps.append({"kind": "composition", "from": comp_prev(sid), "to": sid})
    for s in steps:
        s["control"] = any(families.get(r) == "control" for r in state_roots(s["to"]))
    return steps


# ------------------------------------------------------------------ Go-T2


def is_resolvable(mean: float, se: float) -> bool:
    return abs(mean) >= 2 * se


def resolvability(rows) -> dict:
    """rows: {control, objective, mean, se} per (step, objective)."""
    nc = [is_resolvable(r["mean"], r["se"]) for r in rows if not r["control"]]
    c = [is_resolvable(r["mean"], r["se"]) for r in rows if r["control"]]
    allr = nc + c
    return {
        "noncontrol": float(np.mean(nc)) if nc else float("nan"),
        "n_noncontrol": len(nc),
        "control": float(np.mean(c)) if c else float("nan"),
        "n_control": len(c),
        "pooled_with_control_descriptive": float(np.mean(allr)) if allr else float("nan"),
    }


def fit_f3u(ladder_rows, objectives) -> dict:
    coef = {}
    for o in objectives:
        rs = [r for r in ladder_rows if r["objective"] == o]
        X = np.array([[r["a"], r["q"]] for r in rs], dtype=float)
        y = np.array([r["observed"] for r in rs], dtype=float)
        coef[o] = np.linalg.lstsq(X, y, rcond=None)[0].tolist()
    return coef


def static_endpoint(ladder_rows, comp_rows, objectives) -> dict:
    coef = fit_f3u(ladder_rows, objectives)
    ratios, rows = [], []
    for r in comp_rows:
        if r["objective"] not in objectives:
            continue
        c = coef[r["objective"]]
        f = c[0] * r["a"] + c[1] * r["q"]
        e = abs(r["observed"] - f) / r["se"]
        rows.append({**r, "F3u_fit": f, "err_over_se": e})
        if r["h"] >= 3:
            ratios.append(e)
    per_obj = {
        o: float(np.median([x["err_over_se"] for x in rows if x["objective"] == o and x["h"] >= 3]))
        for o in objectives
        if any(x["objective"] == o and x["h"] >= 3 for x in rows)
    }
    return {
        "coef": coef,
        "median_err_over_se_h3plus": float(np.median(ratios)),
        "n_pairs": len(ratios),
        "per_objective_median_descriptive": per_obj,
        "rows": rows,
    }


def go_t2(resolvable_noncontrol: float, static_median: float) -> dict:
    a = resolvable_noncontrol >= GO_T2_RESOLVABLE
    b = static_median > GO_T2_STATIC
    return {"pass_resolvable": a, "pass_static": b, "pass": bool(a and b)}


# ------------------------------------------------------------------ Go-R2


def go_r2(rows) -> dict:
    later = [r for r in rows if r["h"] >= 2]
    e_s = float(np.mean([abs(r["observed"] - r["static"]) / r["se"] for r in later]))
    e_t = float(np.mean([abs(r["observed"] - r["trap"]) / r["se"] for r in later]))
    cut = 1.0 - e_t / e_s
    return {
        "n": len(later),
        "static_err_over_se": e_s,
        "trap_err_over_se": e_t,
        "cut": cut,
        "pass": cut >= GO_R2_CUT,
    }


# ------------------------------------------------------------------ Go-E2


def gain_lcb(objective: str, mean: float, se: float) -> float:
    return ORIENT[objective] * mean - Z95 * se


def guard_breaches(stats: dict, objective: str) -> list[str]:
    """``stats[g] = {mean, se}`` on the guard's scale (WikiText: ΔNLL; others: pp-equivalent change)."""
    out = []
    for g, s in stats.items():
        if g == objective or g in NOT_GUARDS:
            continue
        if g not in GUARDS:
            raise KeyError(g)
        _scale, kind, lim = GUARDS[g]
        if kind == "ucb" and s["mean"] + Z95 * s["se"] > lim:
            out.append(g)
        if kind == "lcb" and s["mean"] - Z95 * s["se"] < lim:
            out.append(g)
    return out


def qualifies(objective, mean, se, breaches) -> bool:
    return gain_lcb(objective, mean, se) > 0 and not breaches


def go_e2(counts: dict, n_candidates: dict) -> dict:
    for o, n in n_candidates.items():
        if n != E2_N:
            raise ValueError(f"{o}: {n} candidates scored, the ruling fixes {E2_N}")
    passed = {o: counts[o] >= GO_E2_MIN for o in counts}
    ok = [o for o in counts if passed[o]]
    primary = None
    if ok:
        best = max(counts[o] for o in ok)
        tied = [o for o in ok if counts[o] == best]
        primary = min(tied, key=OBJECTIVE_TIE_ORDER.index)
    return {"counts": dict(counts), "pass": passed, "any_pass": bool(ok), "primary": primary}


def e2_candidates(objective: str, primitives, seed: int) -> list[dict]:
    prims = sorted(set(map(tuple, primitives)), key=repr)
    if len(prims) != len(list(primitives)):
        raise ValueError("duplicate primitives")
    for L, _d, s in prims:
        if L < E2_MIN_LAYER or abs(s) > E2_MAX_SHIFT:
            raise ValueError(f"primitive {(L, _d, s)} outside layers >= 20, |shift| <= 0.0625")
    k = E2_N - len(E2_LAMBDAS)
    if len(prims) < k:
        raise ValueError(f"{len(prims)} primitives for {k} draws without replacement")
    rng = random.Random(seed)
    pick = rng.sample(prims, k)
    lora = [{"kind": "lora", "objective": objective, "lam": lam} for lam in E2_LAMBDAS]
    return lora + [{"kind": "steer", "primitive": list(p)} for p in pick]


def candidate_key(c: dict) -> str:
    if c["kind"] == "lora":
        return f"lora:{c['objective']}:{c['lam']:g}"
    L, d, s = c["primitive"]
    return f"steer:{d}@{L}:{s:+g}"


# ------------------------------------------------------------------ consequences (ruling section 4)

CONSEQUENCES = {
    "T2 passes and E2 passes": "Build the bank (priced at 7.8 GPU-h without per-state oracle gradients). Write "
    "docs/PREREG_WME.md. Run the outcome-blind power simulation, then the confirmatory tests.",
    "T2 fails, E2 passes": 'Report the trajectory finding as "static cumulative-edit prediction suffices on '
    'Qwen3.5-9B for these families", a result in its own right. The world model is then tested on the enhancement '
    "claims E1 and E2 only.",
    "E2 fails on both 9B and 4B": "No enhancement claim. Report headroom as the limiting finding, then return to "
    "the council.",
}


def consequence(t2_pass: bool, e2_9b_pass: bool, e2_4b_pass) -> dict:
    if not e2_9b_pass and e2_4b_pass is None:
        raise ValueError(
            "E2 failed on 9B: the ruling requires the identical E2 on 4B before any consequence"
        )
    if e2_9b_pass:
        row = "T2 passes and E2 passes" if t2_pass else "T2 fails, E2 passes"
    elif not e2_4b_pass:
        row = "E2 fails on both 9B and 4B"
    else:
        row = None
    return {
        "row": row,
        "text": CONSEQUENCES.get(
            row, "Not covered by section 4 (E2 fails on 9B, passes on 4B): return to the chair."
        ),
    }


def digest(obj) -> str:
    import json

    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
