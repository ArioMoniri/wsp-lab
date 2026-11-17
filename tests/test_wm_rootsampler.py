"""v1 bank root sampler and direction-similarity audit (PREREG_WME v1 §4.1, §4.4; Board A condition 2)."""

import numpy as np
import pytest

from wsp.wm import rootsampler as RS


def _pools(n=400):
    return {s: [f"{s}:{i:04d}" for i in range(n)] for s in RS.SOURCES}


def _dirfn(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    cache = {}

    def f(c):
        key = (
            (tuple(c["pair"]), c["layers"][0])
            if c["family"] in ("steer", "proj_remove")
            else c["id"]
        )
        if key not in cache:
            cache[key] = rng.standard_normal(64)
        return cache[key]

    return f


def test_catalogue_excludes_scarce_sources():
    cat = RS.catalogue()
    srcs = {s for p in cat for s in p}
    assert not any(s.startswith(("syc", "wiki")) for s in srcs)
    assert len(cat) == len(set(cat)) and all(a != b for a, b in cat)


def test_sampler_fills_families_with_disjoint_shards():
    res = RS.sample_roots(
        r_f=4,
        pools=_pools(),
        direction_fn=_dirfn(),
        excluded=[],
        shard_contrast=16,
        shard_lora=64,
        seed=1,
    )
    acc = res["accepted"]
    fams = [r["family"] for r in acc]
    assert all(fams.count(f) == 4 for f in RS.FAMILIES)
    groups = [
        g
        for r in acc
        for g in (r.get("contrast_source_groups") or []) + (r.get("adapter_source_groups") or [])
    ]
    assert len(groups) == len(set(groups))  # J2 holds by construction
    assert all(r["namespace"] == RS.NAMESPACE and r["model"] == RS.MODEL for r in acc)
    scale_noise_tensors = [tuple(r["tensor"]) for r in acc if r["family"] in ("scale", "noise")]
    assert len(scale_noise_tensors) == len(set(scale_noise_tensors))  # J4
    assert not set(scale_noise_tensors) & RS.FORBIDDEN_TENSORS
    # controls: one matched random direction per proj_remove root, same tensors and alpha
    assert len(res["controls"]) == 4


def test_sampler_is_deterministic_and_rejects_j3():
    a = RS.sample_roots(4, _pools(), _dirfn(), [], 16, 64, seed=1)
    b = RS.sample_roots(4, _pools(), _dirfn(), [], 16, 64, seed=1)
    assert [r["id"] for r in a["accepted"]] == [r["id"] for r in b["accepted"]]
    v = np.ones(64)
    same = lambda c: v + 0.01 * np.random.default_rng(len(c["id"])).standard_normal(64)
    r = RS.sample_roots(2, _pools(), same, [], 16, 64, seed=1)
    resid = [x for x in r["accepted"] if x["family"] in ("steer", "proj_remove")]
    assert len(resid) <= 1  # every later residual direction is a J3 duplicate
    assert any(any(s.startswith("J3") for s in x["reasons"]) for x in r["rejected"])
    assert r["feasible"] is False


def test_pilot_and_timing_exclusions():
    pilot = [{"id": "p2:syc@12", "model": RS.MODEL, "space": "residual", "direction": np.ones(64)}]
    same = lambda c: np.ones(64)
    r = RS.sample_roots(1, _pools(), same, pilot, 16, 64, seed=2)
    assert not [x for x in r["accepted"] if x["family"] in ("steer", "proj_remove")]
    timing = [{"id": "t", "model": RS.MODEL, "family": "scale", "tensor": [5, "down"]}]
    r2 = RS.sample_roots(4, _pools(), _dirfn(), timing, 16, 64, seed=3)
    assert [5, "down"] not in [
        x["tensor"] for x in r2["accepted"] if x["family"] in ("scale", "noise")
    ]


def test_direction_audit_matrix():
    rng = np.random.default_rng(0)
    bank = {"b0": rng.standard_normal(32), "b1": rng.standard_normal(32)}
    pilot = {"p0": bank["b0"] * -2.0, "p1": rng.standard_normal(32)}
    a = RS.direction_audit(bank, pilot)
    assert a["flagged"] == ["b0"] and a["worst"]["b0"]["cos"] == pytest.approx(1.0)
    assert a["pass"] is False


def test_noise_rho_is_median_steer_relative_norm():
    roots = [{"family": "steer", "rel_norm_dose1": x} for x in (0.01, 0.03, 0.02)] + [
        {"family": "scale"}
    ]
    assert RS.noise_rho(roots) == pytest.approx(0.02)


def test_lora_task_vector_cosine_is_exact():
    import torch

    g = torch.Generator().manual_seed(0)
    f1 = {"w": (torch.randn(6, 2, generator=g), torch.randn(2, 5, generator=g), 2.0)}
    f2 = {"w": (torch.randn(6, 2, generator=g), torch.randn(2, 5, generator=g), 2.0)}
    d1 = (f1["w"][2] * f1["w"][0] @ f1["w"][1]).ravel()
    d2 = (f2["w"][2] * f2["w"][0] @ f2["w"][1]).ravel()
    ref = float(d1 @ d2 / (d1.norm() * d2.norm()))
    assert RS.lora_cos(f1, f2) == pytest.approx(ref, rel=1e-5)
    c = {"id": "a", "family": "lora_tv", "space": "lora", "direction": {"lora": f1}}
    e = {"id": "b", "family": "lora_tv", "space": "lora", "direction": {"lora": f1}}
    assert RS.joins(c, e) == "J3"
