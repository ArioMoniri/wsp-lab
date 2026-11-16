#!/usr/bin/env python3
"""WM-E v0 synthetic validation record (EXPLORATORY; CPU): world model vs same-input static on the synthetic target.

For world seeds 0-4: saturating dynamics (the positive test), quadratic dynamics and the field-residual null
(the controls), the slope-transport-off ablation, and the plain-permutation diagnostic. Train on horizons <= 2
(320 trajectories), test on 120 held-out horizon-4 trajectories. Writes ``data/wme/synthetic/<run>.json``.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wsp.wm import model as M
from wsp.wm import synthetic as S

CFG = M.WMConfig(epochs=250, hidden=32, latent=16)


def run_one(seed, kind, shuffle=None, slope=True):
    cfg = M.WMConfig(epochs=CFG.epochs, hidden=CFG.hidden, latent=CFG.latent, slope_transport=slope)
    w = S.SynthWorld(kind=kind, seed=seed)
    tr = w.trajectories(n=320, horizon=2, seed=seed + 1)
    te = w.trajectories(n=120, horizon=4, seed=seed + 2)
    if shuffle == "field_null":
        tr, te = (
            S.shuffle_outcomes(tr, seed + 3, world=w),
            S.shuffle_outcomes(te, seed + 4, world=w),
        )
    elif shuffle == "plain":
        tr, te = S.shuffle_outcomes(tr, seed + 3), S.shuffle_outcomes(te, seed + 4)
    wm = M.WMEnsemble(cfg, w.field(), w.base_observation(), n_seeds=5).fit(tr)
    st = M.StaticPredictor(cfg, w.field()).fit(tr)
    bo = w.base_observation()
    e = {
        "wm": M.horizon_errors(wm, bo, te),
        "static": M.horizon_errors(st, bo, te),
        "field": M.horizon_errors(M.FieldOnly(w.field()), bo, te),
    }
    h34 = {k: float(np.mean([v[3], v[4]])) for k, v in e.items()}
    scale = M.outcome_scale(te)
    return {
        "seed": seed,
        "kind": kind,
        "shuffle": shuffle,
        "slope_transport": slope,
        "errors_by_horizon": e,
        "h34": h34,
        "wm_over_static_h34": h34["wm"] / h34["static"] if h34["static"] > 0 else None,
        "gain_over_scale_h34": (h34["static"] - h34["wm"]) / scale,
        "outcome_scale_h34": scale,
        "shrink": {"wm": wm.shrink, "static": st.shrink},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="synth-20261116")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(2)
    t0, c0 = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    rows = []
    for seed in range(args.seeds):
        for kind, shuffle, slope in (
            ("saturating", None, True),
            ("quadratic", None, True),
            ("saturating", "field_null", True),
            ("saturating", None, False),
            ("saturating", "plain", True),
        ):
            rows.append(run_one(seed, kind, shuffle, slope))
            print(
                json.dumps(
                    {
                        k: rows[-1][k]
                        for k in (
                            "seed",
                            "kind",
                            "shuffle",
                            "slope_transport",
                            "wm_over_static_h34",
                            "gain_over_scale_h34",
                        )
                    }
                ),
                flush=True,
            )
    r = resource.getrusage(resource.RUSAGE_SELF)
    out = {
        "label": "EXPLORATORY (synthetic validation of the WM-E v0 machinery; not evidence about any LLM)",
        "config": {
            "epochs": CFG.epochs,
            "hidden": CFG.hidden,
            "latent": CFG.latent,
            "train_n": 320,
            "test_n": 120,
            "train_horizon": 2,
            "test_horizon": 4,
            "ensemble": 5,
        },
        "rows": rows,
        "wall_seconds": time.time() - t0,
        "cpu_seconds": (r.ru_utime - c0.ru_utime) + (r.ru_stime - c0.ru_stime),
    }
    d = ROOT / "data/wme/synthetic"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.run}.json").write_text(json.dumps(out, indent=1, default=float))
    print("wall", out["wall_seconds"], "cpu", out["cpu_seconds"])


if __name__ == "__main__":
    main()
