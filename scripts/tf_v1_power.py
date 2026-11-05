#!/usr/bin/env python3
"""TF-specific planning power for the H1a theta estimand (PREREG_H1_TF section 5), development only.

theta = 1 - sum_i |r_i - StatNN_i| / sum_i |r_i - isotonic_i| over ALL rows of the selected
components. This script is a PLANNING computation, not the section 5 gate: the gate needs the D3
out-of-fold TF-v1 residuals and StatNN / isotonic predictions, which do not exist until the D3
entries are scored under TF-v1. It therefore uses the measured TF-v1 pilot residual scale as a
proxy for the comparator loss and a grid of loss-heterogeneity assumptions, and takes component
sizes (rows per component) from the sealed partition manifest (metadata only).

Construction (never a result row): row comparator loss B_i = c_g * b_i with b_i from the empirical
|family/log-norm residual| of the TF pilot (or a lognormal of matching CV) and a component scale
c_g (lognormal sd sigma_c); alternative loss A_i = B_i * m_i, m_i = (1 - theta) * exp(s z_i - s^2/2),
z_i = sqrt(rho) g_c + sqrt(1 - rho) e_i. E[m_i] = 1 - theta, so the row-summed ratio of expectations
is exactly theta (theta = .15 alternative, 0 null); losses stay nonnegative. Each simulated panel
is analysed with the registered conditional procedure: a component percentile bootstrap (4,000
draws) of the row-summed ratio, reject when the lower 95% bound is above 0 (two-sided 5%).

Usage: tf_v1_power.py [--panels 4000] [--draws 4000] [--quick]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/runs/h1a/panel_manifest.json"
QUAL = ROOT / "data/runs/h1a/qualification_report.json"
OUT = ROOT / "data/runs/h1a/power_report.json"
SEED = 20261105


def wilson_lower(k, n):
    p, z = k / n, 1.96
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n)


def ratio_theta(a, b):
    return 1 - a / b


def simulate(sizes, shape, theta, s, rho, sigma_c, rng, panels, draws, size_inflation=1.0):
    """Rejection rate, precision gate and theta-hat spread for one scenario."""
    n_comp = len(sizes)
    reject = precision = passed = 0
    thetas = []
    for _ in range(panels):
        g = rng.standard_normal(n_comp)
        scale = np.exp(sigma_c * size_inflation * rng.standard_normal(n_comp))
        a_c = np.zeros(n_comp)
        b_c = np.zeros(n_comp)
        for c, n in enumerate(sizes):
            b = shape(n, rng) * scale[c]
            z = np.sqrt(rho) * g[c] + np.sqrt(1 - rho) * rng.standard_normal(n)
            m = (1 - theta) * np.exp(s * z - s * s / 2)
            b_c[c], a_c[c] = b.sum(), (b * m).sum()
        counts = rng.multinomial(n_comp, np.full(n_comp, 1 / n_comp), size=draws)
        a_b, b_b = counts @ a_c, counts @ b_c
        ok = b_b > 0
        boot = 1 - a_b[ok] / b_b[ok]
        lo, hi = np.quantile(boot, [0.025, 0.975])
        hat = ratio_theta(a_c.sum(), b_c.sum())
        thetas.append(hat)
        reject += lo > 0
        half = (hi - lo) / 2
        precision += half <= abs(hat) / 2
        passed += (hat >= 0.15) and (lo > 0) and (half <= hat / 2)
    return {
        "panels": panels,
        "reject_rate": reject / panels,
        "reject_binomial_lower95": float(wilson_lower(reject, panels)),
        "precision_gate_rate": precision / panels,
        "decision_pass_rate": passed / panels,
        "theta_hat_mean": float(np.mean(thetas)),
        "theta_hat_sd": float(np.std(thetas, ddof=1)),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--panels", type=int, default=4000)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--quick", action="store_true", help="500 panels x 500 draws for testing")
    args = parser.parse_args()
    if args.quick:
        args.panels = args.draws = 500
    manifest = json.loads(PANEL.read_text())
    comps = [c for c in manifest["components"] if c["status"] == "confirmatory"]
    sizes = np.array([len(c["stratum_rows"]) for c in comps])
    qual = json.loads(QUAL.read_text())
    resid = np.abs(
        np.array(
            list(
                qual["panels"]["expanded20"]["mmlu_gold_log_prob"]["family_norm_residual"].values()
            )
        )
    )
    empirical = resid / resid.mean()
    cv = float(empirical.std(ddof=1) / empirical.mean())
    shapes = {
        "pilot_abs_residual": lambda n, rng: rng.choice(empirical, size=n) + 1e-9,
        "lognormal_cv1": lambda n, rng: rng.lognormal(-0.5 * np.log(2), np.sqrt(np.log(2)), size=n),
    }
    scenarios = [
        ("A", "pilot_abs_residual", 0.3, 0.3, 0.5),
        ("B", "pilot_abs_residual", 0.6, 0.3, 0.5),
        ("C", "pilot_abs_residual", 0.6, 0.6, 0.5),
        ("D", "pilot_abs_residual", 1.0, 0.6, 0.5),
        ("E", "lognormal_cv1", 0.6, 0.6, 0.5),
        ("F", "lognormal_cv1", 1.0, 0.6, 1.0),
    ]
    rng = np.random.default_rng(SEED)
    report = {
        "seed": SEED,
        "panels_per_scenario": args.panels,
        "bootstrap_draws": args.draws,
        "n_components": len(sizes),
        "rows_per_component": {
            "sum": int(sizes.sum()),
            "mean": float(sizes.mean()),
            "max": int(sizes.max()),
            "size_counts": {str(k): int((sizes == k).sum()) for k in sorted(set(sizes))},
        },
        "proxy_residual": {"n": int(resid.size), "cv_abs_residual": cv},
        "scenarios": {},
    }
    for name, shape, s, rho, sigma_c in scenarios:
        spec = {
            "shape": shape,
            "row_heterogeneity_s": s,
            "within_component_rho": rho,
            "component_scale_sd": sigma_c,
        }
        report["scenarios"][name] = {
            **spec,
            "alternative_theta_0.15": simulate(
                sizes, shapes[shape], 0.15, s, rho, sigma_c, rng, args.panels, args.draws
            ),
            "null_theta_0": simulate(
                sizes, shapes[shape], 0.0, s, rho, sigma_c, rng, args.panels, args.draws
            ),
        }
        print(name, report["scenarios"][name]["alternative_theta_0.15"], flush=True)
    # Component-count sweep for the middle scenario C: resample the actual size distribution.
    sweep = {}
    for n in (60, 80, 100, 120, 160, 200):
        pool = rng.choice(sizes, size=n)
        sweep[str(n)] = simulate(
            pool, shapes["pilot_abs_residual"], 0.15, 0.6, 0.6, 0.5, rng, args.panels, args.draws
        )
    report["component_count_sweep_scenario_C"] = sweep
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
