"""wm.p3slope: D2 per-objective gate, fitted S3/P1/S5 rules on construction steps, cluster-bootstrap bounds."""

import numpy as np

from wsp.wm import p3slope as S


def _steps(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        h = 1 + i % 4
        obs_a = rng.normal(0, 5)
        obs_b = rng.normal(0, 5)
        noise = lambda: rng.normal(0, 0.1)
        a = {  # transport works: oracle and S2 near obs, A0 far
            "obs": obs_a,
            "se": 1.0,
            "A0": obs_a + rng.choice([-3, 3]),
            "oracle": obs_a + noise(),
            "S1": obs_a + noise(),
            "S2": obs_a + noise(),
            "S2a": obs_a + 1.0,
            "s_hat": 0.1,
            "J1": obs_a / 2,
            "J2": obs_a / 4,
            "X": obs_a / 4,
            "R": obs_a,
        }
        b = {  # nothing to recover: oracle = A0
            "obs": obs_b,
            "se": 1.0,
            "A0": obs_b + 1,
            "oracle": obs_b + 1,
            "S1": obs_b + 1,
            "S2": obs_b + 2,
            "S2a": obs_b + 2,
            "s_hat": 5.0,
            "J1": 0.0,
            "J2": 0.0,
            "X": 0.0,
            "R": 0.0,
        }
        out.append(
            {
                "h": h,
                "family": ["scale", "steer", "proj_remove"][i % 3],
                "control": False,
                "split": "construction" if i % 2 == 0 or h == 1 else "test",
                "cluster": i // 4,
                "dnorm": 1.0,
                "obj": {"a": a, "b": b},
            }
        )
    out.append(dict(out[-1], control=True, cluster=99))
    return out


def test_cut_definition():
    st = [{"obj": {"o": {"obs": 0.0, "se": 1.0, "A0": 2.0, "X": 1.0}}}]
    assert S.cut(st, "o", "X") == 0.5
    assert S.cut(st, "missing", "X") is None


def test_d2_gate_pass_kill_and_eligibility():
    st = _steps()
    g = S.s_gate(st, ["a", "b"])
    assert g["per_objective"]["a"]["eligible"] and g["per_objective"]["a"]["verdict"] == "PASS"
    assert not g["per_objective"]["b"]["eligible"]
    assert g["passed"] == ["a"] and "kept only for ['a']" in g["consequence"]
    # kill: S1 and S2 recover < 25 %
    for s in st:
        s["obj"]["a"]["S1"] = s["obj"]["a"]["A0"]
        s["obj"]["a"]["S2"] = s["obj"]["a"]["A0"]
    g = S.s_gate(st, ["a", "b"])
    assert g["per_objective"]["a"]["verdict"] == "KILL" and "dropped" in g["consequence"]
    # controls and h = 1 steps never enter the gate
    assert all(s["h"] >= 2 and not s["control"] for s in S.gate_steps(st))


def test_no_eligible_objective_drops_transport():
    st = _steps()
    for s in st:
        s["obj"]["a"]["oracle"] = s["obj"]["a"]["A0"]
    g = S.s_gate(st, ["a", "b"])
    assert g["eligible"] == [] and "no objective can pass" in g["consequence"]


def test_fitted_rules_use_construction_steps_and_p1_routes_per_objective():
    st = _steps()
    out = S.fitted_gates(st, ["a", "b"], eligible=["a"])
    assert out["n_test"] > 0 and out["n_construction"] > 0
    assert all(out["p1_table"][k] in S.SINGLE_RULES for k in out["p1_table"])
    assert out["p1_table"]["a|scale"] in ("S1", "S2")
    assert out["s3_switch"]["a"] is True
    # S5 reproduces the end slope here, so it is close to the trapezoid and beats nothing it should not
    assert out["S5"]["per_eligible_objective"]["a"]["S5_cut"] is not None
    # fitting must not read test outcomes: perturbing test obs leaves the fitted rules unchanged
    st2 = _steps()
    for s in st2:
        if s["split"] == "test":
            s["obj"]["a"]["obs"] += 100.0
    out2 = S.fitted_gates(st2, ["a", "b"], eligible=["a"])
    assert out2["p1_table"] == out["p1_table"] and out2["s3_switch"] == out["s3_switch"]
    assert out2["residual"] == out["residual"]


def test_cluster_lb():
    st = _steps()
    r = S.cluster_lb(S.gate_steps(st), ["a"], "S2", "A0")
    assert r["beats"] and r["mean"] > 2
    r = S.cluster_lb(S.gate_steps(st), ["a"], "A0", "S2")
    assert not r["beats"]
