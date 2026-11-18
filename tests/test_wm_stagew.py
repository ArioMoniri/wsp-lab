"""wm.stagew: stage-W design, R0/R1 member features, the R5 state model, the R4 baselines, nulls and the H-T2
statistic (docs/council/2026-11-18/RULING-wme-stagew.md). Synthetic tensors and synthetic worlds only."""

import math

import numpy as np
import pytest

from wsp.wm import stagew as W

# ------------------------------------------------------------------ design


def test_identities_and_grids_follow_the_plan():
    assert W.SALT == "wme-w-panel-20261118" and W.NAMESPACE == "wme-w"
    assert W.N_DEV == 500 and W.BUDGET == 2048
    assert W.LAMS == (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    assert set(W.RUNS) == {"r1", "r2", "r3", "r4"}
    assert W.RUNS["r1"] == ("A", 1e-4) and W.RUNS["r4"] == ("B", 2e-4)
    assert W.KS == (0, 2, 4)
    assert W.seed_for("lora:r1") != W.seed_for("lora:r2")
    # fresh namespace: never the pilot-3 seed for the same label
    import hashlib

    p3 = int(hashlib.sha256(b"wme-p3:lora:r1").hexdigest()[:8], 16)
    assert W.seed_for("lora:r1") != p3


def test_teacher_split_holds_out_feature_items_and_two_disjoint_shards():
    rows = [
        {
            "panel": p,
            "id": f"{p}:{i}",
            "correct": i % 5 != 0,
            "answer": "x" if i % 7 else "",
            "n_tokens": 10,
        }
        for p in W.PANELS
        for i in range(200)
    ]
    rows.append(
        {"panel": W.PANELS[0], "id": "long", "correct": True, "answer": "x", "n_tokens": 5000}
    )
    sp = W.split_teacher(rows, n_feature=32)
    ids = {k: {(r["panel"], r["id"]) for r in v} for k, v in sp.items()}
    assert set(sp) == {"feature", "A", "B"}
    assert (
        not ids["feature"] & ids["A"] and not ids["feature"] & ids["B"] and not ids["A"] & ids["B"]
    )
    for p in W.PANELS:
        assert sum(r["panel"] == p for r in sp["feature"]) == 32
    allowed = {
        (r["panel"], r["id"])
        for r in rows
        if r["correct"] and r["answer"] and r["n_tokens"] <= W.BUDGET
    }
    assert ids["feature"] | ids["A"] | ids["B"] == allowed
    assert abs(len(sp["A"]) - len(sp["B"])) <= 2
    assert W.split_teacher(rows, n_feature=32) == sp  # deterministic


def test_checkpoints_are_eight_increasing_fractions():
    assert W.ckpt_steps(96) == [12, 24, 36, 48, 60, 72, 84, 96]
    s = W.ckpt_steps(95)
    assert len(s) == 8 and s == sorted(set(s)) and s[-1] == 95
    with pytest.raises(ValueError):
        W.ckpt_steps(7)


def _steps():
    return {r: 96 for r in W.RUNS}


def test_paths_share_one_final_point_per_run_and_the_queue_is_in_priority_order():
    ps = W.make_paths(_steps())
    smx = [p for p in ps if p["type"] in "SLM"]
    assert len(smx) == 12 and all(len(p["keys"]) == 8 for p in ps)
    final = W.key(W.member("r1", 96))
    assert sum(final in p["keys"] for p in smx) == 3  # S, L and M of r1
    s = next(p for p in ps if p["id"] == "S:r1")
    assert [W.unkey(k)["step"] for k in s["keys"]] == W.ckpt_steps(96)
    lp = next(p for p in ps if p["id"] == "L:r2")
    assert [W.unkey(k)["lam"] for k in lp["keys"]] == list(W.LAMS)
    m = next(p for p in ps if p["id"] == "M:r3")
    assert [W.unkey(k)["groups"] for k in m["keys"]] == list(range(1, 9))
    x = next(p for p in ps if p["id"] == "X:r1-r2")
    assert x["keys"][0] == final and x["keys"][-1] == W.key(W.member("r2", 96))
    q = W.queue(ps)
    assert len(q) == len(set(q)) == 4 * 22 + 2 * 6
    first_l = min(i for i, k in enumerate(q) if W.unkey(k)["lam"] != 1.0)
    last_s = max(i for i, k in enumerate(q) if any(k in p["keys"] for p in ps if p["type"] == "S"))
    assert last_s < first_l  # every S point before any L point
    assert all(W.unkey(k).get("merge") for k in q[-12:])


def test_member_canonical_forms():
    assert W.member("r1", 96, other=("r2", 96), w=1.0) == W.member("r1", 96)
    assert W.member("r1", 96, other=("r2", 90), w=0.0) == W.member("r2", 90)
    m = W.member("r1", 96, other=("r2", 90), w=0.5)
    assert m["merge"] == {"run": "r2", "step": 90, "w": 0.5}
    assert W.unkey(W.key(m)) == m


# ------------------------------------------------------------------ features


def _feature_inputs(rng, runs=("r1", "r2"), steps=(48, 96), n_items=4):
    sq, j1 = {}, {}
    for r in runs:
        for t in steps:
            sq[(r, t)] = rng.random(W.N_LAYERS)
            j1[(r, t)] = rng.standard_normal((3, n_items, W.N_LAYERS))
    cross = {("r1", 96, "r2", 96): 0.1 * rng.random(W.N_LAYERS)}
    return sq, cross, j1


def test_member_features_are_exact_linear_algebra_on_layer_quantities():
    rng = np.random.default_rng(0)
    sq, cross, j1 = _feature_inputs(rng)
    f = W.member_features(W.member("r1", 96), sq, cross, j1)
    assert f.shape == (len(W.FEATURES),)
    for g in range(4):
        assert math.isclose(f[g], math.sqrt(sq[("r1", 96)][8 * g : 8 * g + 8].sum()))
    assert np.allclose(f[4:], j1[("r1", 96)].sum(-1).mean(-1))
    # dose: norms and J1 scale with lambda
    f2 = W.member_features(W.member("r1", 96, lam=2.0), sq, cross, j1)
    assert np.allclose(f2, 2 * f)
    # mask: groups=1 keeps layers 0-3 only
    fm = W.member_features(W.member("r1", 96, groups=1), sq, cross, j1)
    assert math.isclose(fm[0], math.sqrt(sq[("r1", 96)][:4].sum())) and np.all(fm[1:4] == 0)
    assert np.allclose(fm[4:], j1[("r1", 96)][..., :4].sum(-1).mean(-1))
    # merge: |w a + (1-w) b|^2 = w^2|a|^2 + (1-w)^2|b|^2 + 2w(1-w)<a,b>
    m = W.member("r1", 96, other=("r2", 96), w=0.25)
    fx = W.member_features(m, sq, cross, j1)
    a, b, c = sq[("r1", 96)], sq[("r2", 96)], cross[("r1", 96, "r2", 96)]
    want = math.sqrt((0.0625 * a + 0.5625 * b + 0.375 * c)[:8].sum())
    assert math.isclose(fx[0], want)
    jj = 0.25 * j1[("r1", 96)] + 0.75 * j1[("r2", 96)]
    assert np.allclose(fx[4:], jj.sum(-1).mean(-1))
    # base: exactly zero
    assert np.all(W.member_features(W.BASE, sq, cross, j1) == 0)


def test_item_cells_and_counts():
    rng = np.random.default_rng(1)
    n = 300
    base = [
        {
            "trunc": bool(t),
            "correct": bool(c) and not t,
            "len": int(2048 if t else rng.integers(50, 2000)),
        }
        for t, c in zip(rng.random(n) < 0.2, rng.random(n) < 0.6)
    ]
    cells = W.item_cells(base)
    assert len(cells["cell"]) == n and cells["n"].sum() == n and len(cells["n"]) == W.N_CELLS
    assert all(c == 0 for c, b in zip(cells["cell"], base) if b["trunc"])
    assert cells["T"][0] == 1 and np.all(cells["T"][1:] == 0) and cells["l"][0] == 1.0
    recs = [{"trunc": b["trunc"], "correct": b["correct"]} for b in base]
    cnt = W.cell_counts(recs, cells)
    acc, tr = W.endpoints_from_counts(cnt)
    assert math.isclose(acc, np.mean([b["correct"] for b in base]))
    assert math.isclose(tr, np.mean([b["trunc"] for b in base]))
    assert np.all(cnt[:, 1] <= cnt[:, 0]) and np.all(cnt[:, 2] <= cnt[:, 1])


# ------------------------------------------------------------------ synthetic world


def _world(seed=0, n_runs=4, edit_effect=True, **kw):
    return W.synthetic_world(seed=seed, n_runs=n_runs, edit_effect=edit_effect, **kw)


def test_synthetic_world_has_the_stage_w_shape():
    w = _world()
    assert {p["type"] for p in w["paths"]} == {"S", "L", "M"} and len(w["paths"]) == 12
    for p in w["paths"]:
        for k in p["keys"]:
            m = w["members"][k]
            assert m["phi"].shape == (len(W.FEATURES),) and m["y"].shape == (4,)
            assert set(m["counts"]) == set(W.PANELS)
    assert np.all(w["base"]["phi"] == 0)


def test_wm_recovers_a_well_specified_world_and_is_exact_at_zero():
    w = _world(seed=3)
    keys = sorted(w["members"])
    wm = W.WM().fit([w["members"][k] for k in keys] + [w["base"]], w["cells"])
    pred = wm.predict(np.stack([w["members"][k]["phi"] for k in keys]))
    y = np.stack([w["members"][k]["y"] for k in keys])
    se = W.se_of(y, W.N_DEV)
    assert np.median(np.abs(pred - y) / se) < 1.2
    p0 = wm.predict(np.zeros((1, len(W.FEATURES))))[0]
    assert np.all(np.abs(p0 - w["base"]["y"]) / W.se_of(w["base"]["y"][None], W.N_DEV)[0] < 1.5)


def test_wm_path_offsets_absorb_a_shift():
    w = _world(seed=4)
    keys = sorted(w["members"])
    wm = W.WM().fit([w["members"][k] for k in keys] + [w["base"]], w["cells"])
    p = w["paths"][0]
    pts = [w["members"][k] for k in p["keys"]]
    shifted = [W.shift_member(m, w["cells"], df=1.0, dc=-0.5) for m in pts]
    off = wm.fit_offsets(shifted[:4])
    for panel in W.PANELS:
        assert off[panel][0] > 0.4 and off[panel][1] < -0.1


# ------------------------------------------------------------------ baselines


def test_curve_baselines_recover_their_own_family():
    x = np.arange(1, 9, dtype=float)
    y0 = 0.3
    y = y0 + 0.2 * (x / x[0]) ** 0.5
    pred = W.curve_predict("pow2", y0, x[:2], y[:2], x[2:])
    assert np.allclose(pred, np.clip(y[2:], 0, 1), atol=1e-4)
    ens = W.curve_predict("lc_ensemble", y0, x[:4], y[:4], x[4:])
    assert ens.shape == (4,) and np.all(np.isfinite(ens))
    per = W.curve_predict("persistence", y0, x[:2], y[:2], x[2:])
    assert np.all(per == y[1])
    assert np.all(W.curve_predict("persistence", y0, x[:0], y[:0], x) == y0)


def test_pooled_curve_and_offsets():
    tr = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]])  # changes from base, two training paths
    pooled = W.pooled_curve(tr)
    assert np.allclose(pooled, [0.2, 0.3, 0.4])
    y0, y = 0.5, np.array([0.8, 0.9, 1.0])
    pred = W.offset_predict(y0 + pooled, y, k=1)
    assert np.allclose(pred, y0 + pooled + 0.1)
    assert np.allclose(W.offset_predict(y0 + pooled, y, k=0), y0 + pooled)


def test_shrinkage_choice_ties_go_to_the_smaller_s():
    losses = {0.0: 1.0, 0.25: 0.9, 0.5: 0.8, 0.75: 0.799, 1.0: 0.9}
    assert W.choose_shrink(losses) == 0.5
    assert W.choose_shrink({0.0: 1.0, 0.25: 1.0, 0.5: 1.0, 0.75: 1.0, 1.0: 1.0}) == 0.0


# ------------------------------------------------------------------ the statistic


def test_iu_and_holm_rules():
    p = {0: {"a": 0.001, "b": 0.02}, 2: {"a": 0.001, "b": 0.03}, 4: {"a": 0.0001, "b": 0.0001}}
    res = W.decide(p, alpha=0.025)
    assert res[4]["reject"] and res[0]["reject"] is False and res[2]["reject"] is False
    assert res[0]["p_iu"] == 0.02 and res[2]["p_iu"] == 0.03
    # Holm: smallest p at alpha/3, then alpha/2, then alpha
    p2 = {0: {"a": 0.008}, 2: {"a": 0.012}, 4: {"a": 0.02}}
    r2 = W.decide(p2, alpha=0.025)
    assert r2[0]["reject"] and r2[2]["reject"] and r2[4]["reject"]
    p3 = {0: {"a": 0.009}, 2: {"a": 0.012}, 4: {"a": 0.02}}
    r3 = W.decide(p3, alpha=0.025)
    assert not any(r3[k]["reject"] for k in r3)


def test_full_pipeline_on_a_world_with_edit_information():
    w = _world(seed=7, n_runs=4)
    res = W.analyse(w, n_flip=None, lcpfn=None, inner=True)
    assert set(res["settings"]) == {0, 2, 4}
    s0 = res["settings"][0]
    assert set(s0["baselines"]) == {"persistence", "pooled", "statnn", "static"}
    s2 = res["settings"][2]
    assert {"pow2", "lc_ensemble", "persistence", "pooled", "statnn", "static"} <= set(
        s2["baselines"]
    )
    assert len(s0["E"]["wm"]) == 12
    # in a world whose curves are driven by the edit features the WM beats the feature-free zero-shot baselines
    assert s0["C"]["persistence"] > 0 and s0["C"]["pooled"] > 0


def test_nulls_break_the_feature_outcome_link():
    w = _world(seed=8)
    n1 = W.null_shuffle_features(w, seed=1)
    n2 = W.null_permute_paths(w, seed=1)
    # same members and outcomes, different feature/outcome pairing
    k = w["paths"][0]["keys"][3]
    assert np.all(n1["members"][k]["y"] == w["members"][k]["y"])
    assert any(
        not np.allclose(n1["members"][kk]["phi"], w["members"][kk]["phi"]) for kk in w["members"]
    )
    ys = {p["id"]: np.stack([n2["members"][kk]["y"] for kk in p["keys"]]) for p in n2["paths"]}
    yo = {p["id"]: np.stack([w["members"][kk]["y"] for kk in p["keys"]]) for p in w["paths"]}
    assert any(not np.allclose(ys[i], yo[i]) for i in ys)
    for p in n2["paths"]:  # curves move only within a path type
        assert p["type"] == next(q["type"] for q in w["paths"] if q["id"] == p["id"])


def test_lcpfn_wrapper_batches_curves_when_available():
    pytest.importorskip("lcpfn")
    m = W.LCPFN()
    xk = np.array([0.0, 1.0, 2.0])
    Y = np.array([[0.30, 0.50], [0.40, 0.55], [0.45, 0.58]])
    P = m.predict(xk, Y, np.arange(3, 9, dtype=float))
    assert P.shape == (6, 2) and np.all(np.isfinite(P)) and np.all((P >= 0) & (P <= 1))
    assert P[:, 0].mean() > 0.4  # an improving curve stays above its early values


# ------------------------------------------------------------------ fresh panel


def _mc_items(prefix, n, shared=()):
    out = []
    for i in range(n):
        q = (
            shared[i]
            if i < len(shared)
            else f"{prefix} question number {i} about topic {i * 7} with words"
        )
        out.append({"question": q, "options": ["a", "b", "c"], "answer_index": 0})
    return out


def test_panel_selection_excludes_every_exposure_and_near_duplicates():
    from wsp.wm.panel import text_hash

    mm = _mc_items("mmlu", 60)
    sg = _mc_items("sgpqa", 60)
    pools = W.panel_pools(mm, sg)
    exposed = {pools["mmlupro_gen"][0]["hash"], pools["supergpqa_gen"][3]["hash"]}
    # a near-duplicate of an exposed MMLU-Pro question appears in SuperGPQA: it must go too
    sg2 = _mc_items("sgpqa", 60, shared=[mm[0]["question"] + " extra"])
    pools2 = W.panel_pools(mm, sg2)
    res = W.panel_select(pools2, exposed, n=20)
    used = {r["hash"] for p in res["splits"].values() for rows in p.values() for r in rows}
    assert not used & exposed
    assert pools2["supergpqa_gen"][0]["hash"] not in used
    assert {p: len(v["dev"]) for p, v in res["splits"].items()} == {
        "mmlupro_gen": 20,
        "supergpqa_gen": 20,
    }
    assert res["audit"]["clean"]
    assert res == W.panel_select(pools2, exposed, n=20)  # deterministic salted order
    with pytest.raises(ValueError):
        W.panel_select(pools2, exposed, n=60)  # never relaxed
    assert all(
        r["hash"] == text_hash(r["question"] + "||" + "||".join(r["options"]))
        for r in pools["mmlupro_gen"]
    )


def _write_fake_run(d, rng):
    import json as js

    steps = {r: 16 for r in W.RUN_ORDER}
    paths = W.make_paths(steps)
    ck = [(r, s) for r in W.RUN_ORDER for s in W.ckpt_steps(16)]
    L = 32
    (d / "train_meta.json").write_text(
        js.dumps(
            {
                "final_steps": steps,
                "ckpts": {r: W.ckpt_steps(16) for r in W.RUN_ORDER},
                "n_layers": L,
            }
        )
    )
    np.savez(
        d / "features.npz",
        ckpts=np.array([f"{r}:{s}" for r, s in ck]),
        sq=rng.random((len(ck), L)),
        cross_pairs=np.array(["r1:16|r2:16", "r3:16|r4:16"]),
        cross=0.1 * rng.random((2, L)),
        j1=rng.standard_normal((3, 4, len(ck), L)),
        feature_items=np.array(["a|b"] * 4),
    )

    def panels():
        return {
            p: [
                {
                    "id": f"{p}{i}",
                    "correct": bool(c),
                    "trunc": bool(t),
                    "len": 2048 if t else int(rng.integers(50, 2000)),
                }
                for i, (c, t) in enumerate(zip(rng.random(40) < 0.5, rng.random(40) < 0.2))
            ]
            for p in W.PANELS
        }

    lines = [{"key": "base", "tag": "base", "spec": W.BASE, "panels": panels()}]
    lines.append({"key": "base_rerun", "tag": "base_rerun", "spec": W.BASE, "panels": panels()})
    q = W.queue(paths)
    for k in q[:-3]:  # the last merge path stays incomplete
        lines.append({"key": k, "tag": "member", "spec": W.unkey(k), "panels": panels()})
    for r in lines:
        for v in r["panels"].values():
            for x in v:
                x["correct"] = x["correct"] and not x["trunc"]
    (d / "evals.jsonl").write_text("\n".join(js.dumps(r) for r in lines) + "\n")
    return paths


def test_load_run_builds_the_analysis_world(tmp_path):
    rng = np.random.default_rng(5)
    _write_fake_run(tmp_path, rng)
    w = W.load_run(tmp_path)
    assert len(w["paths"]) == 13 and w["incomplete"] == ["X:r3-r4"]
    assert w["N"] == 40 and set(w["base_rerun_discordance"]) == set(W.PANELS)
    k = w["paths"][0]["keys"][2]
    m = w["members"][k]
    assert m["phi"].shape == (7,) and np.all(m["phi"][:4] > 0)
    assert m["counts"]["mmlupro_gen"][:, 0].sum() == 40
    assert np.all(w["base"]["phi"] == 0)
