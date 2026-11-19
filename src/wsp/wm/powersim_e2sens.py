"""Outcome-blind sensitivity of the H-E2 sign-flip test (PREREG_WME §II.22.4; Board A v2 required change 4).
Synthetic, CPU only; the selected design (24 episodes, 8 per family; B 48; dev 64; test 96) and the v2 inputs.

**What the registered test assumes.** ``signflip.e2_test`` flips the signs of the 24 episode paired differences
independently. It is exact for the **conditional null** that, given the realised sealed test items (and the declared
library and predictors), the vector of paired differences is jointly sign-symmetric (each episode's difference is
symmetric about 0 and independent across episodes given the items). It is not by itself a test of equal means.

**Scenarios** (extending ``powersim_v2.sim_e2``):
* ``v2_null`` / ``v2_alt``: the v2 model (shared item shock identical across winners; it cancels in differences).
* ``item_ix_*``: arm-by-item interaction, a per-arm item effect u_a[i] ~ N(0, (iota * item_sd)^2) shared by all of an
  arm's winners (population mean difference 0; conditional on the realised items it is a fixed shift).
* ``fam_ix_*``: arm-by-family (library-specific) contrast v[a, f] ~ N(0, (tau * sd_cand)^2) (population mean 0).
* ``asym_*`` / ``sym_fallback_heavy``: arm-specific fallback rates q_a (winner present with probability q_a) and
  winner gains (m / q_a) * Exp(1), so both arms have the same expected score m exactly (guard breaches off in these
  scenarios so that the population mean difference is exactly 0); the differences are skewed (asymmetric) and many
  are ties at 0 (fallback-heavy).
* ``alt_item_ix_025``: the v2 alternative with the 0.25 interaction.

**Pre-declared calibration rule and consequences (frozen before any run; §II.22.4):**
1. The interaction scenarios measure how often the conditional claim is supported when the population mean
   difference is 0. Whatever they show, H-E2 is worded conditionally on the realised test items and the declared
   library (no generalisation over the item population). No test changes.
2. Mean-null calibration: if the sign-flip rejection rate is <= 0.035 in every ``asym_*`` and
   ``sym_fallback_heavy`` scenario, H-E2 stands as registered.
3. If any exceeds 0.035, a mean advantage requires both the sign-flip test and RB-E2
   (``signflip.rb_e2_test``) to reject (intersection-union); if RB-E2 also exceeds 0.035 in any of those
   scenarios, the mean-advantage wording is withdrawn and a sign-flip rejection is reported only as "the paired
   differences are not sign-symmetric about zero, favouring WM, conditional on these items".
4. The 0.828 power is model-based planning power under an imposed guidance effect, not a guarantee for learned
   search; it is restated with the consequence in force (IU power under ``v2_alt``).
"""

from __future__ import annotations

import math

import numpy as np

from wsp.wm import powersim_v1 as PV
from wsp.wm import signflip as SF

DESIGN = {"n_rep": 8, "B": 48, "n_dev": 64, "n_test": 96}
SCENARIOS = {
    "v2_null": {"delta_sd": 0.0, "null": True},
    "v2_alt": {"delta_sd": 0.5},
    "item_ix_010": {"delta_sd": 0.0, "iota": 0.10, "null": True},
    "item_ix_025": {"delta_sd": 0.0, "iota": 0.25, "null": True},
    "item_ix_050": {"delta_sd": 0.0, "iota": 0.50, "null": True},
    "fam_ix_025": {"delta_sd": 0.0, "tau": 0.25, "null": True},
    "fam_ix_050": {"delta_sd": 0.0, "tau": 0.50, "null": True},
    "asym_moderate": {"asym": (0.3, 0.7), "null": True},
    "asym_fallback_heavy": {"asym": (0.1, 0.5), "null": True},
    "asym_extreme": {"asym": (0.05, 0.95), "null": True},
    "asym_reversed": {"asym": (0.7, 0.3), "null": True},
    "sym_fallback_heavy": {"asym": (0.15, 0.15), "null": True},
    "alt_item_ix_025": {"delta_sd": 0.5, "iota": 0.25},
}
MEAN_NULL_GATE = (
    "asym_moderate",
    "asym_fallback_heavy",
    "asym_extreme",
    "asym_reversed",
    "sym_fallback_heavy",
)
SIZE_MAX = 0.035


def replicate(rng, sc, inp, design=DESIGN, n_flip=SF.N_FLIP, n_boot=1999) -> dict:
    n_rep, B, n_dev, n_test = design["n_rep"], design["B"], design["n_dev"], design["n_test"]
    sd_c = inp["sd_cand"]
    n = 3 * n_rep
    fam = np.arange(n) % 3
    reach = (1 - math.exp(-B / 12.0)) / (1 - math.exp(-32 / 12.0))
    item_sd = inp["syc_se_64"] * 8.0
    se_dev = inp["syc_se_64"] * math.sqrt(64 / n_dev)
    if "asym" in sc:
        m = 0.5 * sd_c * reach
        q = np.array(sc["asym"])[:, None]
        win = rng.random((2, n)) < q
        G = (m / q) * rng.exponential(1.0, (2, n))
        is_base = ~win
        guards_on = False
    else:
        delta = sc["delta_sd"] * sd_c * reach
        ep = PV._noise(rng, n, "normal")
        base = 0.5 * sd_c * reach + 0.5 * sd_c * ep
        G = np.stack(
            [
                base + delta + 0.5 * sd_c * PV._noise(rng, n, "normal"),
                base + 0.5 * sd_c * PV._noise(rng, n, "normal"),
            ]
        )
        if sc.get("tau"):
            v = sc["tau"] * sd_c * rng.standard_normal((2, 3))
            G = G + v[:, fam]
        is_base = (G + se_dev * rng.standard_normal(G.shape) - PV.Z * se_dev) <= 0
        guards_on = True
    shared = math.sqrt(0.5) * item_sd * rng.standard_normal(n_test)
    X = (
        G[:, :, None]
        + shared[None, None, :]
        + math.sqrt(0.5) * item_sd * rng.standard_normal((2, n, n_test))
    )
    if sc.get("iota"):
        u = sc["iota"] * item_sd * rng.standard_normal((2, 1, n_test))
        X = X + u
    if guards_on:
        gmu = np.where(rng.random(G.shape) < 0.05, -0.02, 0.0)
        Gi = gmu[:, :, None] + inp["guard_point_sd_item"] * rng.standard_normal((2, n, n_test))
        ok = Gi.mean(-1) >= -0.01
    else:
        ok = np.ones(G.shape, bool)
    xm = X.mean(-1)
    g = np.where(is_base, 0.0, np.where(ok, xm, np.minimum(xm, 0.0)))
    d = g[0] - g[1]
    seed = int(rng.integers(2**31))
    sf = SF.e2_test(d, fam, n_flip=n_flip, seed=seed)
    rb = SF.rb_e2_test(d, fam, n_boot=n_boot, seed=seed + 1)
    return {
        "sf": bool(sf["reject"]),
        "rb": bool(rb["reject"]),
        "iu": bool(sf["reject"] and rb["reject"]),
        "d_mean": float(d.mean()),
        "zeros": int((d == 0).sum()),
    }


def run(name, reps, inp, seed=20261119, n_flip=SF.N_FLIP, n_boot=1999) -> dict:
    sc = SCENARIOS[name]
    rng = np.random.default_rng([seed, PV._tag("e2sens", (), name)])
    rows = [replicate(rng, sc, inp, n_flip=n_flip, n_boot=n_boot) for _ in range(reps)]
    out = {"scenario": name, "spec": sc, "reps": reps}
    for k in ("sf", "rb", "iu"):
        c = sum(r[k] for r in rows)
        out[k] = {
            "rate": c / reps,
            "ci95": PV.clopper_pearson(c, reps),
            "power_lb": PV.cp_lower_one_sided(c, reps),
        }
    out["mean_d"] = float(np.mean([r["d_mean"] for r in rows]))
    out["mean_d_se"] = float(np.std([r["d_mean"] for r in rows], ddof=1) / math.sqrt(reps))
    out["mean_zero_differences"] = float(np.mean([r["zeros"] for r in rows]))
    return out


def decide(results: dict) -> dict:
    sf_fail = [s for s in MEAN_NULL_GATE if results[s]["sf"]["rate"] > SIZE_MAX]
    rb_fail = [s for s in MEAN_NULL_GATE if results[s]["rb"]["rate"] > SIZE_MAX]
    if not sf_fail:
        verdict = "sign-flip calibrated on every mean-null scenario: H-E2 stands as registered"
    elif not rb_fail:
        verdict = "sign-flip exceeds 0.035 on a mean-null scenario: a mean advantage requires sign-flip AND RB-E2 (IU)"
    else:
        verdict = (
            "sign-flip and RB-E2 both exceed 0.035 on a mean-null scenario: mean-advantage wording withdrawn; "
            "sign-flip rejection reported only as non-symmetry favouring WM, conditional on the items"
        )
    return {"sign_flip_failures": sf_fail, "rb_e2_failures": rb_fail, "verdict": verdict}
