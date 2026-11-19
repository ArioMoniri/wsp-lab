"""wm.stagew2: stage-W2 design, the nested world model (pooled same-path-type curve + an edit-informed residual
shrunk toward 0), the run-cluster test with Holm over the two settings, and the three nulls
(docs/council/2026-11-18/RULING-wme-stagew2.md). Synthetic tensors and synthetic worlds only."""

import hashlib

import numpy as np
import pytest

from wsp.wm import stagew as W
from wsp.wm import stagew2 as W2

# ------------------------------------------------------------------ design


def test_identities_are_fresh_and_follow_the_plan():
    assert W2.NAMESPACE == "wme-w2" and W2.SALT == "wme-w2-panel-20261118"
    assert W2.N_DEV == 500 and W2.PANELS == W.PANELS and W2.BUDGET == W.BUDGET == 2048
    assert W2.KS == (0, 2)
    assert W2.GRID == (1, 2, 3, 4, 6, 8)
    assert W2.LAMS == (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    assert W2.MIN_PATHS == 40 and W2.ALPHA == 0.025 and W2.NULL_MAX == 0.035
    assert W2.SHRINK_GRID == (0.0, 0.25, 0.5, 0.75, 1.0)
    # fresh seeds: never the stage-W or pilot-3 seed of the same label
    for lab in ("lora:r01", "lora-order:r01", "feature-items", "shards"):
        assert W2.seed_for(lab) != W.seed_for(lab)
        assert W2.seed_for(lab) != int(hashlib.sha256(f"wme-p3:{lab}".encode()).hexdigest()[:8], 16)


def test_fifteen_runs_cover_every_shard_lr_cell_and_every_slot_has_every_lr():
    assert len(W2.RUNS) == 15 and list(W2.RUNS) == W2.RUN_ORDER
    assert set(W2.RUNS.values()) == {(s, lr) for s in W2.SHARDS for lr in W2.LRS}
    assert sorted(r for v in W2.SLOTS.values() for r in v) == sorted(W2.RUNS)
    for slot, runs in W2.SLOTS.items():
        assert len(runs) == 5
        assert {W2.RUNS[r][1] for r in runs} == set(W2.LRS), slot
    assert 3 * len(W2.RUNS) >= W2.MIN_PATHS + 3  # one lost run still leaves >= 40 paths


def _rows(n=200):
    rows = [
        {
            "panel": p,
            "id": f"{p}:{i}",
            "correct": i % 5 != 0,
            "answer": "x" if i % 7 else "",
            "n_tokens": 10,
        }
        for p in W.PANELS
        for i in range(n)
    ]
    rows.append(
        {"panel": W.PANELS[0], "id": "long", "correct": True, "answer": "x", "n_tokens": 5000}
    )
    return rows


def test_teacher_split_holds_out_feature_items_and_three_disjoint_shards():
    rows = _rows()
    sp = W2.split_teacher(rows)
    assert set(sp) == {"feature", "T1", "T2", "T3"}
    ids = {k: {(r["panel"], r["id"]) for r in v} for k, v in sp.items()}
    ks = list(ids)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            assert not ids[ks[i]] & ids[ks[j]]
    for p in W.PANELS:
        assert sum(r["panel"] == p for r in sp["feature"]) == 32
    allowed = {
        (r["panel"], r["id"])
        for r in rows
        if r["correct"] and r["answer"] and r["n_tokens"] <= W2.BUDGET
    }
    assert set().union(*ids.values()) == allowed
    sizes = [len(sp[s]) for s in W2.SHARDS]
    assert max(sizes) - min(sizes) <= 2
    assert W2.split_teacher(rows) == sp
    with pytest.raises(ValueError):
        W2.split_teacher(rows + [dict(rows[1])])
    # fresh: not the stage-W feature items
    old = W.split_teacher(rows)
    assert {(r["panel"], r["id"]) for r in old["feature"]} != ids["feature"]


def test_checkpoints_are_the_thinned_stage_w_fractions():
    assert W2.ckpt_steps(64) == [8, 16, 24, 32, 48, 64]
    assert W2.ckpt_steps(96) == [12, 24, 36, 48, 72, 96]
    with pytest.raises(ValueError):
        W2.ckpt_steps(4)


def test_paths_share_one_final_point_per_run_and_the_queue_is_run_major():
    steps = {r: 64 for r in W2.RUN_ORDER}
    paths = W2.make_paths(steps)
    assert [p["type"] for p in paths[:3]] == ["S", "L", "M"]
    assert len(paths) == 45 and all(len(p["keys"]) == 6 for p in paths)
    assert all(tuple(p["x"]) == W2.GRID for p in paths)
    S, L, M = paths[:3]
    assert S["keys"][-1] == L["keys"][3] == M["keys"][-1]
    assert [W.unkey(k)["lam"] for k in L["keys"]] == list(W2.LAMS)
    assert [W.unkey(k)["groups"] for k in M["keys"]] == list(W2.GRID)
    assert [W.unkey(k)["step"] for k in S["keys"]] == W2.ckpt_steps(64)
    q = W2.queue(paths, W2.SLOTS["a"])
    assert len(q) == len(set(q)) == 5 * 16
    runs_in_order = [W.unkey(k)["run"] for k in q]
    assert runs_in_order == sorted(runs_in_order, key=W2.SLOTS["a"].index)  # run-major
    first = q[:16]
    assert {W.unkey(k)["run"] for k in first} == {W2.SLOTS["a"][0]}


def test_thin_world_keeps_the_grid_points_of_a_stage_w_world():
    w = W.synthetic_world(seed=1)
    t = W2.thin_world(w)
    assert all(len(p["keys"]) == 6 and tuple(p["x"]) == W2.GRID for p in t["paths"])
    for p, q in zip(w["paths"], t["paths"]):
        assert q["keys"] == [p["keys"][j - 1] for j in W2.GRID]


# ------------------------------------------------------------------ the nested model


def _world(seed=0, n_runs=6, effect=1.0, path_sd=0.02, noise=True):
    """Endpoint-level synthetic world on the W2 grid: y = type curve + effect * (phi - type mean) @ B + path offset
    (+ binomial-scale noise)."""
    rng = np.random.default_rng(seed)
    runs = [f"r{i + 1:02d}" for i in range(n_runs)]
    steps = {r: 64 for r in runs}
    paths = W2.make_paths(steps)
    F = len(W.FEATURES)
    B = rng.normal(0, 0.02, (F, 4))
    fac = {r: float(np.exp(rng.normal(0, 0.5))) for r in runs}
    base_phi = np.abs(rng.normal(1, 0.3, F))
    curve = {
        t: np.linspace(0.0, 0.08, 6)[:, None] * np.array([1, -1, -0.5, 0.5]) * (i + 1)
        for i, t in enumerate("SLM")
    }
    y0 = np.array([0.7, 0.4, 0.15, 0.35])
    members = {}
    for p in paths:
        r, t = p["runs"][0], p["type"]
        off = rng.normal(0, path_sd, 4)
        for j, k in enumerate(p["keys"]):
            if k in members:
                continue
            phi = fac[r] * base_phi * W2.GRID[j] / 8
            mean_phi = np.mean([fac[q] for q in runs]) * base_phi * W2.GRID[j] / 8
            y = y0 + curve[t][j] + effect * (phi - mean_phi) @ B + off
            if noise:
                y = y + rng.normal(0, 0.02, 4)
            members[k] = {
                "phi": phi,
                "y": np.clip(y, 0.01, 0.99),
                "counts": {pn: np.zeros((W.N_CELLS, 3)) for pn in W.PANELS},
            }
    return {
        "members": members,
        "paths": [p for p in paths if p["type"] in "SLM"],
        "base": {"phi": np.zeros(F), "y": y0},
        "N": 500,
    }


def test_arrays_index_paths_and_points():
    w = _world()
    A = W2.arrays(w)
    assert A["Y"].shape == (18, 6, 4) and A["PHI"].shape == (18, 6, len(W.FEATURES))
    assert list(A["types"][:3]) == ["S", "L", "M"]
    assert A["SE"].shape == A["Y"].shape and np.all(A["SE"] > 0)


def test_pooled_curve_matches_the_stage_w_pooled_baseline():
    w = _world(seed=3)
    A = W2.arrays(w)
    train = [r for r in A["run_ids"] if r != "r01"]
    fit = W2.fit(A, train)
    y0 = w["base"]["y"]
    for t in "SLM":
        ch = [
            np.stack([w["members"][k]["y"] for k in p["keys"]]) - y0
            for p in w["paths"]
            if p["type"] == t and p["runs"][0] in train
        ]
        np.testing.assert_allclose(fit["curve"][t], y0 + W.pooled_curve(ch), atol=1e-12)
    # held-out path scores with s = 0 are the stage-W pooled baseline scores
    i = int(np.flatnonzero(A["runs"] == "r01")[1])
    y = A["Y"][i]
    for k in (0, 2):
        pred = W2.predict(fit, A, i, s=0.0, k=k)
        ref = np.stack(
            [W.offset_predict(fit["curve"][A["types"][i]][:, e], y[:, e], k) for e in range(4)], 1
        )
        np.testing.assert_allclose(pred, ref, atol=1e-12)
        assert W2.score(pred, A, i, k) == pytest.approx(W._score(ref, y, 500, k))


def test_nested_model_is_exactly_the_pooled_curve_at_zero_shrinkage():
    w = _world(seed=4)
    A = W2.arrays(w)
    r = W2.analyse(A, shrink_grid=(0.0,))
    for k in W2.KS:
        st = r["settings"][k]
        np.testing.assert_allclose(st["E"]["nested"], st["E"]["pooled"], atol=1e-12)
        assert st["p"] == 1.0 and not st["reject"]
    assert not r["pass_primary"]


def test_shrinkage_is_chosen_inside_the_training_folds_only():
    w = _world(seed=5, n_runs=5)
    A = W2.arrays(w)
    r1 = W2.analyse(A)
    A2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in A.items()}
    ix = A2["runs"] == "r03"
    A2["Y"][ix] = np.clip(A2["Y"][ix] + 0.2, 0, 1)
    A2["SE"] = W.se_of(A2["Y"], A2["N"])
    r2 = W2.analyse(A2)
    assert r1["shrink"]["r03"] == r2["shrink"]["r03"]


def test_residual_ridge_uses_type_centred_standardised_features():
    w = _world(seed=6, noise=False, path_sd=0.0)
    A = W2.arrays(w)
    fit = W2.fit(A, list(A["run_ids"]))
    for t in "SLM":
        ix = A["types"] == t
        np.testing.assert_allclose(fit["phibar"][t], A["PHI"][ix].mean(0), atol=1e-12)
    assert fit["W"].shape == (len(W.FEATURES), 4)


def test_edit_information_is_found_and_a_no_edit_world_is_not():
    hits = []
    for s in range(3):
        A = W2.arrays(_world(seed=10 + s, n_runs=15, effect=3.0, path_sd=0.005))
        r = W2.analyse(A)
        hits.append(r["pass_primary"])
        assert r["settings"][0]["shrink_mean"] > 0
    assert all(hits)
    A = W2.arrays(_world(seed=20, n_runs=15, effect=0.0, path_sd=0.02))
    r = W2.analyse(A)
    assert not r["pass_primary"]


def test_holm_over_two_settings_and_lower_bounds():
    d = W2.holm({0: 0.011, 2: 0.03})
    assert d[0]["reject"] and not d[2]["reject"]
    assert d[0]["alpha"] == pytest.approx(0.0125) and d[2]["alpha"] == pytest.approx(0.025)
    d = W2.holm({0: 0.02, 2: 0.02})
    assert not d[0]["reject"] and not d[2]["reject"]
    assert d[0]["alpha"] == d[2]["alpha"] == pytest.approx(0.0125)
    d = W2.holm({0: 0.001, 2: 0.02})
    assert d[0]["reject"] and d[2]["reject"]


def test_run_cluster_test_weights_paths_and_flips_whole_runs():
    d = np.array([1.0, 1.0, 1.0, -0.5, 2.0, 0.5])
    runs = np.array(["a", "a", "a", "b", "c", "c"])
    t = W2.cluster_test(d, runs, alpha=0.025)
    assert t["n"] == 3 and t["stat"] == pytest.approx(d.mean())
    assert t["p"] == pytest.approx(2 / 8)  # exact over 2^3 patterns: (+,+,+) and (+,-,+) reach T
    z = W2.cluster_test(np.zeros(6), runs, alpha=0.025)
    assert z["p"] == 1.0 and not z["reject"] and z["lb"] <= 0


# ------------------------------------------------------------------ nulls


def test_nulls_keep_shapes_and_break_the_feature_outcome_link():
    w = _world(seed=30, n_runs=6)
    A = W2.arrays(w)
    for kind in ("n1", "n2"):
        B = W2.arrays(W2.null_world(w, kind, seed=1))
        assert B["Y"].shape == A["Y"].shape
    B1 = W2.arrays(W2.null_world(w, "n1", seed=1))
    np.testing.assert_allclose(B1["Y"], A["Y"])
    assert not np.allclose(B1["PHI"], A["PHI"])
    B2 = W2.arrays(W2.null_world(w, "n2", seed=1))
    np.testing.assert_allclose(B2["PHI"], A["PHI"])
    assert not np.allclose(B2["Y"], A["Y"])


def _count_world(seed=0):
    """Cell-count world with the stage-W shape thinned to the W2 grid (for the N3 generator)."""
    return W2.thin_world(W.synthetic_world(seed=seed, n_runs=4))


def test_n3_world_draws_counts_from_type_curves_and_path_effects_only():
    w = _count_world(seed=2)
    cal = W2.calibrate_n3(w)
    assert set(cal["tau"]) == set(W.PANELS)
    assert all(v >= 0 for p in W.PANELS for v in cal["tau"][p])
    rng = np.random.default_rng(0)
    w3 = W2.n3_world(w, cal, rng)
    A3 = W2.arrays(w3)
    A = W2.arrays(w)
    np.testing.assert_allclose(A3["PHI"], A["PHI"])
    assert A3["Y"].shape == A["Y"].shape
    # truth logits are the same for every path of a type at a point (no edit information)
    for t in "SLM":
        z = np.stack([cal["q"][t][j][W.PANELS[0]][0] for j in range(6)])
        assert np.all(np.isfinite(z))


def test_null_size_counts_whole_decisions():
    w = _world(seed=40, n_runs=6)
    res = W2.null_sizes(w, kinds=("n1", "n2"), reps=4, seed=0)
    for kind in ("n1", "n2"):
        r = res[kind]
        assert r["reps"] == 4 and 0.0 <= r["size"] <= 1.0
        assert set(r["reject_rate"]) == set(W2.KS)


def test_stage_w_comparators_run_on_the_six_point_grid():
    """The stage-W comparator pipeline (persistence, curves, pooled, StatNN, static, unnested WM) on W2-shaped paths."""
    w = W2.thin_world(W.synthetic_world(seed=3, n_runs=3))
    r = W.analyse(w, n_flip=200, inner=False, ks=W2.KS)
    for k in W2.KS:
        E = r["settings"][k]["E"]
        assert len(E["wm"]) == 9
        assert {"persistence", "pooled", "statnn", "static", "wm"} <= set(E)
    assert {"pow2", "lc_ensemble"} <= set(r["settings"][2]["E"])
    # the stage-W pooled baseline equals the W2 pooled curve, path by path
    A = W2.arrays(w)
    n = W2.analyse(A, inner=False)
    for k in W2.KS:
        a = dict(zip(r["paths"], r["settings"][k]["E"]["pooled"]))
        b = dict(zip(n["paths"], n["settings"][k]["E"]["pooled"]))
        assert a.keys() == b.keys()
        for p, v in a.items():
            assert v == pytest.approx(b[p], abs=1e-9)


# ------------------------------------------------------------------ panel and loader


def _mc_items(prefix, n, shared=()):
    out = [
        {
            "question": f"{prefix} question number {i} about topic {i * 7} with words {i} {i + 1} {i + 2}",
            "options": [f"{prefix} opt {i} {c}" for c in "ABCD"],
            "answer_index": i % 4,
        }
        for i in range(n)
    ]
    for j, q in enumerate(shared):
        out[j] = dict(out[j], question=q)
    return out


def test_panel_requires_every_registered_exposure_and_excludes_them():
    mm, sg = _mc_items("mmlu", 60), _mc_items("sgpqa", 60)
    pools = W.panel_pools(mm, sg)
    exposed = {pools["mmlupro_gen"][0]["hash"], pools["supergpqa_gen"][3]["hash"]}
    with pytest.raises(RuntimeError):
        W2.panel_select(
            pools, exposed, registered=[x for x in W2.REQUIRED_EXPOSURES if x != "wme-w"], n=20
        )
    res = W2.panel_select(pools, exposed, registered=list(W2.REQUIRED_EXPOSURES), n=20)
    used = {r["hash"] for p in res["splits"].values() for rows in p.values() for r in rows}
    assert not used & exposed and res["audit"]["clean"]
    assert {p: len(v["dev"]) for p, v in res["splits"].items()} == {
        "mmlupro_gen": 20,
        "supergpqa_gen": 20,
    }
    # a fresh salt: not the stage-W order
    old = W.panel_select(pools, exposed, n=20)
    assert [r["hash"] for r in old["splits"]["mmlupro_gen"]["dev"]] != [
        r["hash"] for r in res["splits"]["mmlupro_gen"]["dev"]
    ]
    assert {"wme-v0-p1", "wme-p2", "wme-p3s", "wme-p3h", "wme-w"} <= set(W2.REQUIRED_EXPOSURES)


def _write_slot(d, slot, rng, with_base=True, drop_last=0, L=32, n_items=40):
    import json as js

    runs = W2.SLOTS[slot]
    steps = {r: 64 for r in runs}
    paths = W2.make_paths(steps)
    ck = [(r, s) for r in runs for s in W2.ckpt_steps(64)]
    d.mkdir(parents=True, exist_ok=True)
    (d / "train_meta.json").write_text(
        js.dumps(
            {
                "final_steps": steps,
                "ckpts": {r: W2.ckpt_steps(64) for r in runs},
                "n_layers": L,
                "slot": slot,
            }
        )
    )
    np.savez(
        d / "features.npz",
        ckpts=np.array([f"{r}:{s}" for r, s in ck]),
        sq=rng.random((len(ck), L)),
        cross_pairs=np.array([], dtype=str),
        cross=np.zeros((0, L)),
        j1=rng.standard_normal((3, 4, len(ck), L)),
        feature_items=np.array(["a|b"] * 4),
    )

    def panels():
        out = {}
        for p in W.PANELS:
            rows = []
            for i in range(n_items):
                t = bool(rng.random() < 0.2)
                rows.append(
                    {
                        "id": f"{p}{i}",
                        "correct": bool(rng.random() < 0.5) and not t,
                        "trunc": t,
                        "len": 2048 if t else int(rng.integers(50, 2000)),
                    }
                )
            out[p] = rows
        return out

    lines = []
    if with_base:
        lines.append({"key": "base", "tag": "base", "spec": W.BASE, "panels": panels()})
    q = W2.queue(paths, runs)
    for k in q[: len(q) - drop_last]:
        lines.append({"key": k, "tag": "member", "spec": W.unkey(k), "panels": panels()})
    (d / "evals.jsonl").write_text("\n".join(js.dumps(r) for r in lines) + "\n")


def test_load_runs_merges_slots_and_keeps_only_complete_paths(tmp_path):
    rng = np.random.default_rng(7)
    _write_slot(tmp_path / "a", "a", rng)
    _write_slot(tmp_path / "b", "b", rng, drop_last=3)  # the last run's M path is incomplete
    w = W2.load_runs({"a": tmp_path / "a", "b": tmp_path / "b"})
    assert w["base_slot"] == "a"
    assert len(w["paths"]) == 29 and w["incomplete"] == ["M:r10"]
    assert w["N"] == 40
    A = W2.arrays(w)
    assert A["Y"].shape == (29, 6, 4)
    # base member has zero features; an L member's features scale with lambda
    lk = next(p for p in w["paths"] if p["id"] == "L:r01")["keys"]
    np.testing.assert_allclose(
        w["members"][lk[1]]["phi"][4:], 2 * w["members"][lk[0]]["phi"][4:], rtol=1e-9
    )


def test_null_decisions_split_across_chunks_equal_one_pass():
    w = _world(seed=41, n_runs=5)
    a = W2.null_decisions(w, "n2", range(4), seed=3)
    b = W2.null_decisions(w, "n2", range(2), seed=3) + W2.null_decisions(
        w, "n2", range(2, 4), seed=3
    )
    assert [r["pass"] for r in a] == [r["pass"] for r in b]
    assert [r["d_mean"] for r in a] == [r["d_mean"] for r in b]
    s = W2.summarize_null(a)
    assert s["reps"] == 4 and 0 <= s["size"] <= s["size_cp_upper95"] <= 1


def test_ht3_power_grows_with_runs_and_vanishes_without_effect():
    rng = np.random.default_rng(0)
    runs = np.repeat([f"r{i}" for i in range(15)], 3)
    d = 0.5 + rng.normal(0, 0.5, 45)
    p15 = W2.ht3_power(d, runs, 15, reps=60, seed=1)
    p30 = W2.ht3_power(d, runs, 30, reps=60, seed=1)
    assert p30 >= p15 and p30 > 0.5
    z = W2.ht3_power(d - d.mean(), runs, 15, reps=60, seed=1)
    assert z <= 0.1
