"""wme-c1 E2 planning-predictor stage on a synthetic bank (CPU): history assembly, training restriction, inner
folds, conformal q_hat, coverage switch, support and inert predictions."""

import dataclasses

import numpy as np
import pytest

from wsp.wm import e2run as ER
from wsp.wm.c1 import field as F
from wsp.wm.c1 import fitops as FO
from wsp.wm.c1.schema import OBJ6

THETA0 = "0" * 64


def synth_bank(n_comp=12, seed=0, n_items=4, k_sk=4):
    rng = np.random.default_rng(seed)
    spec = F.spec_9b()
    base = {
        "values": {o: rng.normal(size=n_items).tolist() for o in OBJ6},
        "sketch": {o: rng.normal(size=k_sk).tolist() for o in OBJ6},
        "theta_hash": THETA0,
    }
    states, comps, strata = [], [], {}
    g = rng.normal(size=(spec.dim, len(OBJ6))) * 0.1
    for c in range(n_comp):
        cid = f"single:r{c}"
        comps.append(cid)
        strata[cid] = ["scale", "steer", "noise"][c % 3]
        u = rng.normal(size=spec.dim)
        for side, sgn in (("pos", 1), ("neg", -1)):
            for h, dose in enumerate([0.25, 0.5, 1.0, 2.0], 1):
                R = sgn * dose * u
                y = R @ g
                vals = {
                    o: (
                        np.asarray(base["values"][o]) + y[k] + rng.normal(scale=0.01, size=n_items)
                    ).tolist()
                    for k, o in enumerate(OBJ6)
                }
                states.append(
                    {
                        "state_id": f"{cid}:{side}:{h}",
                        "kind": "ladder",
                        "component": cid,
                        "roots": [f"r{c}"],
                        "side": side,
                        "h": h,
                        "dose": sgn * dose,
                        "R": R.tolist(),
                        "values": vals,
                        "sketch": {
                            o: (
                                np.asarray(base["sketch"][o])
                                + 0.1 * np.resize(R, len(base["sketch"][o]))
                            ).tolist()
                            for o in OBJ6
                        },
                    }
                )
    bank = {
        "base": base,
        "states": states,
        "inert": [{"spec": {"kind": "i1"}, "states": [{"R": [0.0] * spec.dim}]}],
    }
    return bank, comps, strata, spec


def fast_fit(spec, base_obs):
    cfg = dataclasses.replace(ER.E2_WM_CONFIG, epochs=5)
    return lambda trajs, comps: ER.fit_planning_predictors(
        trajs, comps, spec, base_obs, cfg=cfg, n_members=2
    )


def test_histories_respect_the_training_restriction_and_calibration():
    bank, _comps, strata, spec = synth_bank()
    H = FO.histories(bank)
    assert {h["h"] for h in H} == {1, 2, 3, 4}
    assert all(h["R"].shape == (h["h"] + 1, spec.dim) for h in H)
    assert all(np.all(h["y"][0] == 0) for h in H)
    folds = FO.inner_folds([h["component"] for h in H], strata, 1051842779)
    assert set(folds.values()) == {0, 1, 2, 3}
    assert FO.inner_folds([h["component"] for h in H], strata, 1051842779) == folds


def test_fit_all_q_hat_coverage_mode_and_inert():
    bank, comps, strata, spec = synth_bank()
    base_obs = FO.base_artefact(bank, THETA0)
    cal = comps[:3]
    seen = []

    def fit_fn(trajs, cs):
        seen.append(set(cs))
        assert all(t["horizon"] <= 2 for t in trajs)
        return fast_fit(spec, base_obs)(trajs, cs)

    out = FO.fit_all(bank, cal, strata, 1051842779, base_obs, spec, fit_fn=fit_fn)
    assert len(seen) == 5  # 4 inner folds + the final fit
    assert all(not (s & set(cal)) for s in seen)
    assert out["q_hat"]["wm"] > 0 and out["q_hat"]["static"] > 0
    assert out["mode"] in ("lcb", "ranks")
    assert set(out["coverage"]) == {"wm", "static"} and out["n_cov_units"] > 0
    if not all(v["pass"] for v in out["coverage"].values()):
        assert out["mode"] == "ranks"
    ip = FO.inert_predictions(out, bank, spec)
    assert len(ip["wm"]) == 1 and len(ip["wm"][0]["pred"]) == 6


def test_conformal_quantile_rule():
    assert FO.conformal_q(list(range(1, 10))) == 9  # ceil(10 * 0.9) = 9th of 9
    assert FO.conformal_q(list(range(1, 101))) == pytest.approx(91)
