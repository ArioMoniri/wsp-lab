"""scripts/wme_w2_noise_floor.py: the empirical noise floor of the stage-W2 path score E from replicate measurements
of one state (SYNTHESIS-r13 R2). Synthetic records and synthetic tensors only."""

import importlib.util
import math
from pathlib import Path

import numpy as np

from wsp.wm import stagew2 as W2

P = Path(__file__).resolve().parents[1] / "scripts" / "wme_w2_noise_floor.py"
spec = importlib.util.spec_from_file_location("wme_w2_noise_floor", P)
NF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NF)


def _recs(correct, trunc):
    return [{"correct": bool(c), "trunc": bool(t)} for c, t in zip(correct, trunc)]


def _panels(rng, n, p_acc=0.5, p_tr=0.3):
    tr = rng.random(n) < p_tr
    return {p: _recs(rng.random(n) < p_acc, tr) for p in W2.PANELS}


def test_item_endpoints_follow_the_w2_definition():
    # a truncated response is never correct (W.cell_counts); the endpoints are acc then trunc
    rec = _recs([1, 1, 0, 1], [0, 1, 0, 0])
    acc, tr = NF.item_endpoints(rec)
    assert acc.tolist() == [1, 0, 0, 1] and tr.tolist() == [0, 1, 0, 0]
    assert NF.ENDPOINTS == tuple(f"acc:{p}" for p in W2.PANELS) + tuple(
        f"trunc:{p}" for p in W2.PANELS
    )


def test_identical_replicates_give_a_zero_floor():
    rng = np.random.default_rng(0)
    pan = _panels(rng, 400)
    out = NF.pair_floor(pan, pan)
    assert out["n_items"] == 400
    for e in NF.ENDPOINTS:
        assert out["endpoints"][e]["floor"] == 0.0 and out["endpoints"][e]["e_replicate"] == 0.0
    assert out["floor"] == 0.0


def test_independent_bernoulli_replicates_recover_sqrt_2_over_pi():
    # if every item is redrawn independently, the replicate noise equals the binomial SE, so the floor is ~0.80
    rng = np.random.default_rng(1)
    n = 20000
    a = {p: _recs(rng.random(n) < 0.4, np.zeros(n)) for p in W2.PANELS}
    b = {p: _recs(rng.random(n) < 0.4, np.zeros(n)) for p in W2.PANELS}
    out = NF.pair_floor(a, b)
    for p in W2.PANELS:
        assert abs(out["endpoints"][f"acc:{p}"]["floor"] - math.sqrt(2 / math.pi)) < 0.03


def test_floor_matches_the_flip_formula():
    # 500 items, 20 flip 1 -> 0 and 10 flip 0 -> 1 on accuracy: var(d) = 30/500 - (10/500)^2
    n = 500
    c1 = np.zeros(n, bool)
    c1[:250] = True
    c2 = c1.copy()
    c2[:20] = False
    c2[300:310] = True
    z = np.zeros(n, bool)
    a = {p: _recs(c1, z) for p in W2.PANELS}
    b = {p: _recs(c2, z) for p in W2.PANELS}
    out = NF.pair_floor(a, b)
    e = out["endpoints"][f"acc:{W2.PANELS[0]}"]
    vd = 30 / n - (10 / n) ** 2
    sigma = math.sqrt(vd / (2 * n))
    y1, y2 = 0.5, 0.48
    se = [math.sqrt(y * (1 - y) / n) for y in (y1, y2)]
    want = math.sqrt(2 / math.pi) * sigma * 0.5 * (1 / se[0] + 1 / se[1])
    assert abs(e["floor"] - want) < 1e-12
    want_rep = 0.5 * (0.02 / se[0] + 0.02 / se[1])
    assert abs(e["e_replicate"] - want_rep) < 1e-12
    assert abs(e["floor_plugin"] - want_rep / math.sqrt(2)) < 1e-12
    # truncation is identical in both runs
    assert out["endpoints"][f"trunc:{W2.PANELS[0]}"]["floor"] == 0.0


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(2)
    n = 300
    a = _panels(rng, n)
    b = {p: [dict(r) for r in v] for p, v in a.items()}
    for p in W2.PANELS:
        for i in rng.choice(n, 30, replace=False):
            b[p][i]["correct"] = not b[p][i]["correct"]
    est = NF.pair_floor(a, b)["floor"]
    ci1 = NF.bootstrap([(a, b)], n_boot=300, seed=5)
    ci2 = NF.bootstrap([(a, b)], n_boot=300, seed=5)
    assert ci1 == ci2
    assert ci1["floor"]["lo"] <= est <= ci1["floor"]["hi"]


def test_rule_closes_only_when_floor_is_within_005_of_pooled():
    assert NF.rule(0.75, 0.771)["closed"] is True
    assert NF.rule(0.721, 0.771)["closed"] is True
    r = NF.rule(0.30, 0.771)
    assert r["closed"] is False and abs(r["headroom"] - 0.471) < 1e-12


def test_pooled_by_endpoint_averages_to_the_w2_path_score():
    rng = np.random.default_rng(3)
    runs = [f"r{i:02d}" for i in range(4)]
    types = np.array([t for _ in runs for t in "SLM"])
    rr = np.array([r for r in runs for _ in "SLM"])
    Y = np.clip(0.4 + 0.05 * rng.standard_normal((12, 6, 4)), 0.01, 0.99)
    A = {
        "Y": Y,
        "PHI": rng.standard_normal((12, 6, 3)),
        "SE": W2.W.se_of(Y, 500),
        "types": types,
        "runs": rr,
        "run_ids": runs,
        "y0": np.full(4, 0.4),
    }
    out = NF.pooled_by_endpoint(A)
    for k in W2.KS:
        ref = []
        for r in runs:
            f = W2.fit(A, [q for q in runs if q != r])
            for i in np.flatnonzero(rr == r):
                ref.append(W2.score(W2.predict(f, A, i, 0.0, k), A, i, k))
        assert abs(out[k]["E"] - float(np.mean(ref))) < 1e-12
        assert abs(np.mean(list(out[k]["by_endpoint"].values())) - out[k]["E"]) < 1e-12
