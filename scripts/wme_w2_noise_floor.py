#!/usr/bin/env python3
"""Empirical noise floor of the stage-W2 path score E (SYNTHESIS-r13 R2). DESCRIPTIVE addendum to W2; CPU only.

E (wsp.wm.stagew2.score) = mean |pred - y| / SE(y) over the points after k and the 4 endpoints (accuracy and
truncation on MMLU-Pro and SuperGPQA), with SE(y) = sqrt(max(y(1-y), 1/N)/N) (wsp.wm.stagew.se_of), N = 500 items
per benchmark. A perfect predictor (it knows the expected value of a state's measurement on the fixed panel) still
scores the measurement noise of the target. That noise is measured from replicate generations of one state: the
base model's `base` and `base_rerun` evaluations (same items, same engine, reversed item order).

Per endpoint e and replicate pair (1, 2), with item-level differences d_i = x1_i - x2_i:

* sigma_e = sqrt(var(d) / (2 N)): the single-measurement noise SD (Var(y1 - y2) = var(d) / N, two noisy copies);
* floor_e = sqrt(2/pi) * sigma_e * mean(1/SE(y1), 1/SE(y2)): the expected E of a perfect predictor (Gaussian |.|);
* e_replicate_e = mean over both directions of |y1 - y2| / SE(target): the E of predicting one measurement by its
  replicate (one draw); floor_plugin_e = e_replicate_e / sqrt(2) rescales it to a single measurement's noise.

The floor is the mean over the 4 endpoints (every endpoint carries equal weight in E). CIs: item bootstrap (items
resampled in pairs within each benchmark). The pooled curve's E is recomputed per endpoint (leave-one-run-out, s = 0,
exactly wsp.wm.stagew2.fit/predict/score) for the per-endpoint comparison. Rule (SYNTHESIS-r13 decision table): floor
>= pooled E - 0.05 closes (a) on 9B for good; otherwise the headroom is pooled E - floor.

    wme_w2_noise_floor.py --w2 data/wme/stagew2/run-20261118 --stagew data/wme/stagew/run-20261118/remote/w \
        --json docs/reviews/wme/wme-stagew2-noisefloor-2026-11-20.json

Reads only exposed W2 (and, as a secondary pair, stage-W) run artefacts; writes only the JSON review file (the
markdown review is written from it).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from wsp.wm import stagew as W
from wsp.wm import stagew2 as W2

ENDPOINTS = tuple(f"acc:{p}" for p in W2.PANELS) + tuple(f"trunc:{p}" for p in W2.PANELS)
MARGIN = 0.05
POOLED_REPORTED = {0: 0.771, 2: 0.881}
GAUSS = math.sqrt(2 / math.pi)
N_BOOT = 10_000
SEED = 20261120


def item_endpoints(records) -> tuple:
    """(acc_i, trunc_i) as float arrays; a truncated response is never correct (wsp.wm.stagew.cell_counts)."""
    tr = np.array([bool(r["trunc"]) for r in records], float)
    acc = np.array([bool(r["correct"]) and not bool(r["trunc"]) for r in records], float)
    return acc, tr


def _matrix(panels) -> np.ndarray:
    """{panel: records} -> [N, 4] item-level endpoint indicators in ENDPOINTS order."""
    cols = {}
    for p in W2.PANELS:
        cols[f"acc:{p}"], cols[f"trunc:{p}"] = item_endpoints(panels[p])
    return np.stack([cols[e] for e in ENDPOINTS], 1)


def _align(a, b) -> tuple:
    """Item-aligned matrices; records are matched by id when ids exist, else by position."""
    for p in W2.PANELS:
        if len(a[p]) != len(b[p]):
            raise ValueError(f"{p}: replicate sizes differ")
        if "id" in a[p][0] and [r["id"] for r in a[p]] != [r["id"] for r in b[p]]:
            idx = {r["id"]: r for r in b[p]}
            b = dict(b)
            b[p] = [idx[r["id"]] for r in a[p]]
    return _matrix(a), _matrix(b)


def _floor_from(X1, X2) -> dict:
    n = X1.shape[0]
    y1, y2 = X1.mean(0), X2.mean(0)
    se1, se2 = W.se_of(y1, n), W.se_of(y2, n)
    d = X1 - X2
    sigma = np.sqrt(d.var(0) / (2 * n))
    inv = 0.5 * (1 / se1 + 1 / se2)
    floor = GAUSS * sigma * inv
    e_rep = np.abs(y1 - y2) * inv
    return {
        "n": n,
        "y1": y1,
        "y2": y2,
        "se1": se1,
        "se2": se2,
        "discordance": (d != 0).mean(0),
        "sigma": sigma,
        "sigma_over_se": sigma * inv,
        "floor": floor,
        "e_replicate": e_rep,
        "floor_plugin": e_rep / math.sqrt(2),
    }


def pair_floor(a, b) -> dict:
    """Floor of one replicate pair ({panel: records} each). Note N = items per benchmark (the W2 SE's N)."""
    X1, X2 = _align(a, b)
    f = _floor_from(X1, X2)
    eps = {
        e: {
            k: float(f[k][j])
            for k in (
                "y1",
                "y2",
                "se1",
                "se2",
                "discordance",
                "sigma",
                "sigma_over_se",
                "floor",
                "e_replicate",
                "floor_plugin",
            )
        }
        for j, e in enumerate(ENDPOINTS)
    }
    return {
        "n_items": f["n"],
        "endpoints": eps,
        "floor": float(f["floor"].mean()),
        "floor_plugin": float(f["floor_plugin"].mean()),
        "e_replicate": float(f["e_replicate"].mean()),
        "sigma_over_se": float(f["sigma_over_se"].mean()),
    }


def bootstrap(pairs, n_boot: int = N_BOOT, seed: int = SEED, conf: float = 0.95) -> dict:
    """Item bootstrap (paired, within benchmark) of the floor averaged over pairs; percentile CIs."""
    rng = np.random.default_rng(seed)
    mats = [_align(a, b) for a, b in pairs]
    # columns j and j + 2 belong to the same benchmark (acc, trunc): resample items per benchmark
    stats = {"floor": [], "floor_plugin": [], "sigma_over_se": []}
    per_ep = {e: [] for e in ENDPOINTS}
    for _ in range(n_boot):
        fl, fp, so = [], [], []
        ep = np.zeros(len(ENDPOINTS))
        for X1, X2 in mats:
            n = X1.shape[0]
            B1, B2 = np.empty_like(X1), np.empty_like(X2)
            for j in range(len(W2.PANELS)):
                ix = rng.integers(0, n, n)
                cols = [j, j + len(W2.PANELS)]
                B1[:, cols] = X1[ix][:, cols]
                B2[:, cols] = X2[ix][:, cols]
            f = _floor_from(B1, B2)
            fl.append(f["floor"].mean())
            fp.append(f["floor_plugin"].mean())
            so.append(f["sigma_over_se"].mean())
            ep += f["floor"]
        stats["floor"].append(np.mean(fl))
        stats["floor_plugin"].append(np.mean(fp))
        stats["sigma_over_se"].append(np.mean(so))
        for j, e in enumerate(ENDPOINTS):
            per_ep[e].append(ep[j] / len(mats))
    q = ((1 - conf) / 2, 1 - (1 - conf) / 2)

    def ci(v):
        lo, hi = np.quantile(np.asarray(v), q)
        return {"lo": float(lo), "hi": float(hi)}

    out = {k: ci(v) for k, v in stats.items()}
    out["endpoints"] = {e: ci(v) for e, v in per_ep.items()}
    out["n_boot"] = n_boot
    out["seed"] = seed
    out["conf"] = conf
    return out


def rule(floor: float, pooled: float, margin: float = MARGIN) -> dict:
    closed = bool(floor >= pooled - margin - 1e-12)
    return {
        "floor": floor,
        "pooled": pooled,
        "threshold": pooled - margin,
        "closed": closed,
        "headroom": float(pooled - floor),
    }


def pooled_by_endpoint(A, ks=W2.KS) -> dict:
    """Leave-one-run-out pooled curve (s = 0), path score split by endpoint; mean over endpoints = W2's pooled E."""
    runs = A["run_ids"]
    P = A["Y"].shape[0]
    per = {k: np.zeros((P, 4)) for k in ks}
    for r in runs:
        f = W2.fit(A, [q for q in runs if q != r])
        for i in np.flatnonzero(A["runs"] == r):
            for k in ks:
                pred = W2.predict(f, A, i, 0.0, k)
                per[k][i] = np.mean(np.abs(pred[k:] - A["Y"][i, k:]) / A["SE"][i, k:], 0)
    out = {}
    for k in ks:
        m = per[k].mean(0)
        out[k] = {
            "E": float(per[k].mean()),
            "by_endpoint": {e: float(m[j]) for j, e in enumerate(ENDPOINTS)},
            "by_type": {
                t: float(per[k][A["types"] == t].mean()) for t in "SLM" if (A["types"] == t).any()
            },
        }
    return out


def _bases(evals: Path) -> dict:
    out = {}
    for line in evals.read_text().splitlines():
        r = json.loads(line)
        if r["tag"] != "member":
            out[r["tag"]] = r
    return out


def _bitwise(a, b) -> bool:
    keys = ("correct", "trunc", "len", "parsed")
    return all(
        [tuple(r[k] for k in keys) for r in a["panels"][p]]
        == [tuple(r[k] for k in keys) for r in b["panels"][p]]
        for p in W2.PANELS
    )


def run(w2_dir: Path, stagew_dir: Path | None, n_boot: int) -> dict:
    t0 = time.time()
    slots = {s: w2_dir / s / "w2" for s in sorted(W2.SLOTS) if (w2_dir / s / "w2").is_dir()}
    bases = {s: _bases(d / "evals.jsonl") for s, d in slots.items()}
    pairs = {}
    for s, b in bases.items():
        if "base" in b and "base_rerun" in b:
            pairs[f"w2:{s}"] = (b["base"]["panels"], b["base_rerun"]["panels"])
    # other repeated measurements of the base state: the other slots' `base` (same order, other hosts)
    ref = next(s for s in slots if "base" in bases[s])
    same_order = {
        f"w2:{ref}~w2:{s}": _bitwise(bases[ref]["base"], bases[s]["base"])
        for s in slots
        if s != ref and "base" in bases[s]
    }
    same_order_floor = {
        f"w2:{ref}~w2:{s}": pair_floor(bases[ref]["base"]["panels"], bases[s]["base"]["panels"])[
            "floor"
        ]
        for s in slots
        if s != ref and "base" in bases[s]
    }
    if not pairs:
        raise SystemExit("no base/base_rerun pair in the W2 run")
    primary = {k: pair_floor(*v) for k, v in pairs.items()}
    boot = bootstrap(list(pairs.values()), n_boot=n_boot)
    floor = float(np.mean([v["floor"] for v in primary.values()]))
    floor_plugin = float(np.mean([v["floor_plugin"] for v in primary.values()]))

    secondary = None
    if stagew_dir is not None and (stagew_dir / "evals.jsonl").is_file():
        sb = _bases(stagew_dir / "evals.jsonl")
        if "base" in sb and "base_rerun" in sb:
            sp = (sb["base"]["panels"], sb["base_rerun"]["panels"])
            both = list(pairs.values()) + [sp]
            secondary = {
                "stagew": pair_floor(*sp),
                "stagew_ci": bootstrap([sp], n_boot=n_boot),
                "combined_floor": float(np.mean([pair_floor(*p)["floor"] for p in both])),
                "combined_ci": bootstrap(both, n_boot=n_boot),
                "n_pairs": len(both),
            }

    world = W2.load_runs(slots)
    A = W2.arrays(world)
    pooled = pooled_by_endpoint(A)
    k2_offset = math.sqrt(
        1 + 1 / 2
    )  # a perfect predictor offset by its mean residual on 2 noisy points
    verdict = {}
    for k in W2.KS:
        pk = pooled[k]["E"]
        verdict[k] = {
            "primary": rule(floor, pk),
            "ci_lo": rule(boot["floor"]["lo"], pk),
            "ci_hi": rule(boot["floor"]["hi"], pk),
            "plugin": rule(floor_plugin, pk),
        }
        if k == 2:
            verdict[k]["k2_offset_protocol"] = rule(floor * k2_offset, pk)
        if secondary:
            verdict[k]["combined_with_stagew"] = rule(secondary["combined_floor"], pk)
    return {
        "label": "DESCRIPTIVE addendum to W2 (wme-stagew2-2026-11-18); SYNTHESIS-r13 R2",
        "definition": {
            "E": "mean |pred - y| / SE(y) over the points after k and the 4 endpoints (wsp.wm.stagew2.score)",
            "SE": "sqrt(max(y(1-y), 1/N)/N), N = items per benchmark (wsp.wm.stagew.se_of)",
            "floor": "sqrt(2/pi) * sqrt(var(d)/(2N)) * mean(1/SE(y1), 1/SE(y2)), d = item-level replicate "
            "difference; mean over the 4 endpoints",
            "e_replicate": "mean over directions of |y1 - y2| / SE(target): E of predicting a measurement by its "
            "replicate",
            "floor_plugin": "e_replicate / sqrt(2)",
            "rule": "floor >= pooled E - 0.05 -> (a) closed on 9B for good; else headroom = pooled E - floor",
        },
        "endpoints": list(ENDPOINTS),
        "n_replicate_pairs": len(pairs),
        "pairs": list(pairs),
        "n_items_per_benchmark": next(iter(primary.values()))["n_items"],
        "primary": primary,
        "floor": floor,
        "floor_plugin": floor_plugin,
        "bootstrap": boot,
        "same_order_bases_bitwise": same_order,
        "same_order_bases_floor": same_order_floor,
        "secondary_stagew": secondary,
        "pooled": {str(k): v for k, v in pooled.items()},
        "pooled_reported": {str(k): v for k, v in POOLED_REPORTED.items()},
        "gauss_theory": GAUSS,
        "k2_offset_factor": k2_offset,
        "verdict": {str(k): v for k, v in verdict.items()},
        "n_paths": int(A["Y"].shape[0]),
        "cpu_seconds": round(time.time() - t0, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--w2", required=True, type=Path)
    ap.add_argument("--stagew", type=Path, default=None)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    for p in (a.json,):
        if "data" in p.resolve().parts and "docs" not in p.resolve().parts:
            sys.exit("refusing to write under data/")
    res = run(a.w2, a.stagew, a.n_boot)
    res["command"] = " ".join(["scripts/wme_w2_noise_floor.py"] + (argv or sys.argv[1:]))
    a.json.write_text(json.dumps(res, indent=2, sort_keys=False) + "\n")
    print(json.dumps({k: res[k] for k in ("floor", "floor_plugin", "n_replicate_pairs")}, indent=2))
    print(json.dumps(res["bootstrap"]["floor"]), json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
