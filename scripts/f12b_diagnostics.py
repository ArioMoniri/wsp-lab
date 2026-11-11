#!/usr/bin/env python3
"""F12b design diagnostics (docs/design/F12B_REPAIRS.md). CPU only; synthetic tokens; outcome-free.

Reduced experiments behind the F12b repairs of the F12 simulation design. Inputs: the synthetic token
generator of ``scripts/f12_simulation.py`` and the restricted, allowlisted copy of the F8a.2
qualification noise payload (``noise_payload_restricted.json``: centred_noise, window_mask,
window_ids, donors{id, component, family}). No outcome, no ``donor_mean``, no ``data/evals`` file and
no Part A payload is read. Every number written is synthetic-token evidence.

Subcommands (each writes one JSON under --out):

* ``gamma``  noise-only oracle calibration under the F12b rule (F12a.4(h) grid, convexity bracket):
             Monte Carlo error of the calibrated oracle C by calibration-panel count, validation pass
             rates at +-0.005 and +-0.02, and how often F12a.4(h)'s monotonicity check would fail.
* ``head``   WM-R held-out error against F3 (WM-R only, no StatNN) under head variants
             (drafted / zero-init / target-standardised / both), null and alternative.
* ``smoke``  re-runs replicates of the drafted F12 smoke run (same seeds, drafted config, 100-tree
             StatNN as in the smoke run) and applies the strict and the F12b joint studentiser rule
             to the same 4,000 draws.

Replicate counts are capped at 20 per scenario (F12b instruction); ``gamma`` panels are noise-only
calibration panels, not replicates (no model is fitted).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import dataclasses
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import f12_simulation as fs  # noqa: E402

OUT = Path("/home/ubuntu/wsp-data/runs/a5/f12_sim_dryrun/f12b")
PAYLOAD = Path("/home/ubuntu/wsp-data/runs/a5/f12_sim_dryrun/noise_payload_restricted.json")
MAX_REPLICATES = 20
DOMAIN = "20261111:f12b:"
LABEL = "synthetic-token evidence (outcome-free; no real token, no outcome)"


def cpu():
    return time.process_time()


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def stage2():
    return fs.load_config(fs.F12B_CONFIGS["stage2"])[0]


def head_variants():
    drafted = fs.Config()
    return {
        "drafted": drafted,
        "zero_init": dataclasses.replace(drafted, head_zero_init=True),
        "std_target": dataclasses.replace(drafted, target_standardise=True),
        "zero_init+std_target": dataclasses.replace(
            drafted, head_zero_init=True, target_standardise=True
        ),
    }


def panel(ctx, scenario, r):
    """(D, y, gamma, h, calibration info) for replicate r with the context's config and seeds."""
    alt = scenario == "alt"
    cfg = ctx.cfg
    gamma, h, cal = 0.0, np.zeros(len(ctx.a)), None
    if alt:
        h = ctx.hstar(r)
        gamma, cal = ctx.calibrate(h, r)
        if gamma is None:
            return None, cal
    seed = fs.domain_seed(fs.FOLD_DOMAIN, f"{cfg.seed_tag}:{scenario}:{ctx.k}:{r}")
    rng = np.random.default_rng(seed)
    E, _, _ = fs.draw_panel_noise(ctx.payload, ctx.comp, rng, order=cfg.noise_draw, row_ids=ctx.ids)
    D = (ctx.a + gamma * h)[:, None] + E
    return (D, D.mean(1), gamma, h), cal


def wmr_oof(ctx, y, cfg):
    wm = np.full(len(y), np.nan)
    info = []
    for f in range(fs.N_OUTER):
        test, train = np.flatnonzero(ctx.fold == f), np.flatnonzero(ctx.fold != f)
        wm[test], i = fs.fit_wmr_fold(ctx.data, ctx.a, ctx.q, y, ctx.m["component"], train, test, cfg)
        info.append(i)
    return wm, info


# --------------------------------------------------------------------------------- gamma


def ref_panels(ctx, key, count, cfg):
    out = []
    for d in range(count):
        rng = np.random.default_rng(fs.domain_seed(DOMAIN, f"{key}:{d}"))
        E = fs.draw_panel_noise(ctx.payload, ctx.comp, rng, order=cfg.noise_draw, row_ids=ctx.ids)[0]
        out.append(E.mean(1))
    return np.stack(out)


def run_gamma(args, payload):
    base = stage2()
    counts = [int(x) for x in args.cal_counts.split(",")]
    nmax = max(counts)
    m = fs.synthetic_token_matrix()
    out = {"label": LABEL, "panel_counts": counts, "reference_panels": args.reference, "k": {}}
    for k in [int(x) for x in args.k_grid.split(",")]:
        t0 = cpu()
        ctx = fs.Context(m, payload, k, base)
        ref = ref_panels(ctx, f"ref:{k}", args.reference, base)
        rows = []
        for r in range(args.hstar):
            h = ctx.hstar(r)
            args_p = (payload, ctx.comp, ctx.ids)
            cal_all = fs.calibration_panels(*args_p, nmax, "cal", k, r, base)
            val_all = fs.calibration_panels(*args_p, nmax, "calval", k, r, base)
            rec = {"r": r}
            for N in counts:
                cfg = dataclasses.replace(base, cal_draws=N, val_draws=N, cal_tol=1.0)
                gamma, info = fs.calibrate_gamma_f12b(
                    ctx.P, ctx.a, h, cal_all[:N], val_all[:N], ctx.tau, cfg
                )
                f12a = fs.calibrate_gamma_f12b(
                    ctx.P, ctx.a, h, cal_all[:N], val_all[:N], ctx.tau,
                    dataclasses.replace(cfg, cal_monotone="f12a"),
                )[1]["status"]  # fmt: skip
                if gamma is None:
                    rec[str(N)] = {"status": info["status"], "f12a_status": f12a}
                    continue
                c_ref, se_ref = fs.ratio_c_with_se(ctx.P, ctx.a, h, ref, gamma)
                rec[str(N)] = {
                    "status": info["status"],
                    "f12a_status": f12a,
                    "dip_below_C0": info["dip_below_C0"],
                    "gamma": gamma,
                    "halvings": info["halvings"],
                    "C_cal": info["C_calibration"],
                    "C_val": info["C_validation"],
                    "C_val_se": info["C_validation_se"],
                    "C_ref": c_ref,
                    "C_ref_se": se_ref,
                }
            rows.append(rec)
            print(
                f"[gamma] k={k} r={r} "
                + " ".join(
                    f"N{N}: ref {rec[str(N)].get('C_ref', float('nan')):.4f} "
                    f"val {rec[str(N)].get('C_val', float('nan')):.4f} {rec[str(N)]['f12a_status']}"
                    for N in counts
                ),
                file=sys.stderr,
                flush=True,
            )
        summ = {}
        for N in counts:
            ok = [x[str(N)] for x in rows if x[str(N)]["status"] == "ok"]
            if not ok:
                summ[str(N)] = {"ok": 0}
                continue
            cref = np.array([x["C_ref"] for x in ok])
            cval = np.array([x["C_val"] for x in ok])
            summ[str(N)] = {
                "ok": len(ok),
                "n": len(rows),
                "f12a_check_fails": sum(x[str(N)]["f12a_status"] != "ok" for x in rows),
                "dip_below_C0": sum(x["dip_below_C0"] for x in ok),
                "C_ref_mean": float(cref.mean()),
                "C_ref_sd_across_hstar": float(cref.std(ddof=1)) if len(cref) > 1 else None,
                "C_ref_max_abs_dev": float(np.abs(cref - fs.TARGET_C).max()),
                "C_val_minus_target_sd": float((cval - fs.TARGET_C).std(ddof=1))
                if len(cval) > 1
                else None,
                "C_val_se_median": float(np.median([x["C_val_se"] for x in ok])),
                "C_ref_se_median": float(np.median([x["C_ref_se"] for x in ok])),
                "validation_within_0.005": int(np.sum(np.abs(cval - fs.TARGET_C) <= 0.005)),
                "validation_within_0.02": int(np.sum(np.abs(cval - fs.TARGET_C) <= 0.02)),
            }
        out["k"][str(k)] = {
            "records": rows,
            "summary": summ,
            "cpu_s": cpu() - t0,
            "n_rows": len(ctx.a),
            "tau": ctx.tau,
        }
        write(args, out)
    return out


# ---------------------------------------------------------------------------------- head


def run_head(args, payload):
    variants = head_variants()
    names = args.variants.split(",")
    base = dataclasses.replace(stage2(), seeds=fs.Config().seeds)
    ctx = fs.Context(fs.synthetic_token_matrix(), payload, args.k, base)
    out = {"label": LABEL, "k": args.k, "variants": names, "records": [], "n_rows": len(ctx.a)}
    for scenario in args.scenarios.split(","):
        for r in range(min(args.replicates, MAX_REPLICATES)):
            p, cal = panel(ctx, scenario, r)
            if p is None:
                out["records"].append({"scenario": scenario, "r": r, "failed": cal["status"]})
                continue
            D, y, gamma, h = p
            f3 = ctx.P @ y
            rec = {
                "scenario": scenario,
                "r": r,
                "gamma": gamma,
                "A_F3": float(np.abs(y - f3).sum()),
                "A_oracle": float(np.abs(y - ctx.a - gamma * h).sum()),
                "variants": {},
            }
            rec["C_oracle"] = 1 - rec["A_oracle"] / rec["A_F3"]
            for name in names:
                t0 = cpu()
                try:
                    wm, info = wmr_oof(ctx, y, variants[name])
                except ValueError as exc:
                    rec["variants"][name] = {"failed": str(exc)}
                    continue
                A = float(np.abs(y - wm).sum())
                rec["variants"][name] = {
                    "A_WM": A,
                    "C_WM": 1 - A / rec["A_F3"],
                    "head_mean_abs": float(np.abs(wm - f3).mean()),
                    "refit_epochs": [i["refit_epochs"] for i in info],
                    "inner_best_median": float(np.median([i["inner_best"] for i in info])),
                    "cpu_s": cpu() - t0,
                }
                print(
                    f"[head] {scenario} r={r} {name}: C_WM {1 - A / rec['A_F3']:.4f} "
                    f"(oracle {rec['C_oracle']:.4f}) cpu {cpu() - t0:.0f}s rss {rss_mb():.0f}MB",
                    file=sys.stderr,
                    flush=True,
                )
            out["records"].append(rec)
            write(args, out)
    summ = {}
    for scenario in args.scenarios.split(","):
        recs = [x for x in out["records"] if x["scenario"] == scenario and "variants" in x]
        for name in names:
            ok = [x for x in recs if "A_WM" in x["variants"].get(name, {})]
            if not ok:
                continue
            c = np.array([x["variants"][name]["C_WM"] for x in ok])
            summ[f"{scenario}:{name}"] = {
                "n": len(ok),
                "C_WM_ratio_of_sums": float(
                    1 - sum(x["variants"][name]["A_WM"] for x in ok) / sum(x["A_F3"] for x in ok)
                ),
                "C_WM_median": float(np.median(c)),
                "C_WM_min": float(c.min()),
                "C_WM_max": float(c.max()),
                "cpu_s_mean": float(np.mean([x["variants"][name]["cpu_s"] for x in ok])),
            }
        if recs and recs[0]["gamma"]:
            summ[f"{scenario}:oracle"] = {
                "C_oracle_ratio_of_sums": float(
                    1 - sum(x["A_oracle"] for x in recs) / sum(x["A_F3"] for x in recs)
                )
            }
    out["summary"] = summ
    return out


# --------------------------------------------------------------------------------- smoke


def run_smoke(args, payload):
    """Drafted-config replicates (the smoke run's seeds and settings) with both studentiser rules."""
    cfg = dataclasses.replace(fs.Config(), tree_grid=(100,))  # the smoke run's --tree-grid 100
    m = fs.synthetic_token_matrix()
    ctx = fs.Context(m, payload, args.k, cfg)
    out = {"label": LABEL, "k": args.k, "config": "F12 as drafted, tree grid {100}", "records": []}
    for scenario in args.scenarios.split(","):
        for r in range(min(args.replicates, MAX_REPLICATES)):
            t0 = cpu()
            p, cal = panel(ctx, scenario, r)
            if p is None:
                out["records"].append({"scenario": scenario, "r": r, "failed": cal["status"]})
                continue
            D, y, gamma, h = p
            f3, wm, st, _ = ctx.fit_all(y)
            rec = {"scenario": scenario, "r": r, "gamma": gamma}
            for cname, qb in (("C_WM", f3), ("C_WM_S", st)):
                parts = fs.joint_parts(D, wm, qb, ctx.comp, ctx.mults[1])
                raw = parts["v_comp"] + parts["v_item"] - parts["v_both"]
                res = {}
                for rule in fs.STUDENTISER_RULES:
                    c = fs.a542_contrast(D, wm, qb, ctx.comp, ctx.mults, rule=rule)
                    res[rule] = {
                        "C": c["C"],
                        "valid": c["valid"],
                        "failures": c.get("failures", []),
                        "studentiser": c.get("studentiser"),
                        "joint_bootstrap_t": c["intervals"].get("joint_bootstrap_t"),
                        "conditional_bootstrap_t": c["intervals"].get("conditional_bootstrap_t"),
                    }
                ok = raw > 0
                res["raw_over_vcomp_quantiles"] = {
                    str(q): float(np.quantile(raw / parts["v_comp"], q))
                    for q in (0.0, 0.01, 0.05, 0.5)
                }
                res["n_raw_nonpositive"] = int((~ok).sum())
                rec[cname] = res
            rec["cpu_s"] = cpu() - t0
            print(
                f"[smoke] {scenario} r={r} C_WM={rec['C_WM']['strict']['C']:.4g} "
                f"nonpos {rec['C_WM']['n_raw_nonpositive']}/{rec['C_WM_S']['n_raw_nonpositive']} "
                f"floor valid {rec['C_WM']['floor_one_way']['valid']}/"
                f"{rec['C_WM_S']['floor_one_way']['valid']} cpu {rec['cpu_s']:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            out["records"].append(rec)
            write(args, out)
    return out


def write(args, obj):
    path = Path(args.out) / args.name
    path.write_text(json.dumps(obj, indent=1, default=fs._json_default))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("gamma", "head", "smoke"))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--name", required=True)
    p.add_argument("--payload", type=Path, default=PAYLOAD)
    p.add_argument("--k", type=int, default=145)
    p.add_argument("--scenarios", default="null,alt")
    p.add_argument("--replicates", type=int, default=4)
    p.add_argument("--variants", default="zero_init,std_target,zero_init+std_target")
    p.add_argument("--k-grid", default="0,145")
    p.add_argument("--hstar", type=int, default=20)
    p.add_argument("--cal-counts", default="100,400,1000,2000")
    p.add_argument("--reference", type=int, default=8000)
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    fs.guard_out(Path(a.out))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    payload = fs.NoisePayload.load_restricted(a.payload, fs.RESTRICTED_PIN)
    t0, w0 = cpu(), time.time()
    res = {"gamma": run_gamma, "head": run_head, "smoke": run_smoke}[a.command](a, payload)
    res["cpu_seconds_total"] = cpu() - t0
    res["wall_seconds_total"] = time.time() - w0
    res["max_rss_mb"] = rss_mb()
    res["payload"] = payload.provenance
    write(a, res)
    print(json.dumps({k: res[k] for k in ("cpu_seconds_total", "wall_seconds_total", "max_rss_mb")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
