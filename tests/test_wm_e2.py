"""E2 episode definitions (PREREG_WME v1 §5.2-5.3, §6.5, §6.8.1, §7.1; Board A condition 10)."""

import json

import numpy as np
import pytest

from wsp.wm import e2 as E


def test_library_sizes_and_ids_match_registration():
    lib = E.library("sycophancy_real")
    assert {f: len(v) for f, v in lib.items()} == {"steer": 96, "proj_remove": 96, "lora_tv": 64}
    ids = [p["id"] for v in lib.values() for p in v]
    assert len(ids) == len(set(ids))
    steer = lib["steer"]
    assert {p["layer"] for p in steer} == {20, 22, 24, 26, 28, 30}
    assert {p["contrast"] for p in steer} == {"syc", "gsm", "mathcomp", "trivia"}
    assert sorted({p["dose"] for p in steer}) == [-0.0625, -0.03125, 0.03125, 0.0625]
    assert sorted({p["dose"] for p in lib["proj_remove"]}) == [-0.25, -0.125, 0.125, 0.25]
    lt = lib["lora_tv"]
    assert {tuple(p["block"]) for p in lt} == {(0, 7), (8, 15), (16, 23), (24, 31)}
    assert (
        sorted({p["dose"] for p in lt}) == [0.25, 0.5, 1.0, 2.0]
        and len({p["lora"] for p in lt}) == 4
    )
    assert E.CONTRASTS["syc"] == ("syc_suggested", "syc_neutral")
    assert E.CONTRASTS["trivia"] == ("syc_neutral", "harmless")


def test_secondary_libraries_use_their_own_contrast_and_lora_task():
    g = E.library("gsm8k_gold_lp")
    assert {p["contrast"] for p in g["steer"]} == {"gsm"}
    assert {p["task"] for p in g["lora_tv"]} == {"gsm8k_gold_lp"}
    m = E.library("mathcomp_gold_lp")
    assert {p["contrast"] for p in m["proj_remove"]} == {"mathcomp"}


def test_episodes_seeds_and_subsamples_are_fixed_and_fresh():
    eps = E.episodes("sycophancy_real", n_rep=4)
    assert len(eps) == 12 and [e["index"] for e in eps] == list(range(12))
    assert [e["family"] for e in eps[:3]] == ["steer", "proj_remove", "lora_tv"]
    again = E.episodes("sycophancy_real", n_rep=4)
    assert json.dumps(eps, sort_keys=True) == json.dumps(again, sort_keys=True)
    seeds = [e["seeds"]["planner"] for e in eps] + [e["seeds"]["subsample"] for e in eps]
    assert len(set(seeds)) == len(seeds)
    for e in eps:
        k = 48 if e["family"] in ("steer", "proj_remove") else 32
        assert len(e["subsample"]) == k == len(set(e["subsample"]))
    # n_rep = 2 episodes are a prefix of n_rep = 4 (seeds do not depend on the grid value)
    assert E.episodes("sycophancy_real", n_rep=2) == [e for e in eps if e["rep"] < 2]
    sec = E.episodes("gsm8k_gold_lp", n_rep=1)
    assert {e["seeds"]["planner"] for e in sec}.isdisjoint(seeds)


def test_candidate_validation():
    ep = E.episodes("sycophancy_real", n_rep=1)[0]
    s = ep["subsample"]
    assert E.validate_candidate((), ep) == ()
    assert E.validate_candidate((s[0], s[3]), ep) == (s[0], s[3])
    with pytest.raises(ValueError):
        E.validate_candidate((s[0], s[0]), ep)  # distinct primitives
    with pytest.raises(ValueError):
        E.validate_candidate(tuple(s[:5]), ep)  # at most 4
    outside = next(p["id"] for p in E.library("sycophancy_real")["steer"] if p["id"] not in s)
    with pytest.raises(ValueError):
        E.validate_candidate((outside,), ep)
    assert E.candidate_key((s[1], s[0])) != E.candidate_key(
        (s[0], s[1])
    )  # order is part of the edit


def test_schedule_four_rounds():
    assert E.schedule(16) == [4, 4, 4, 4] and E.schedule(48) == [12] * 4
    with pytest.raises(ValueError):
        E.schedule(18)


def test_pool_mode_is_seeded_and_within_subsample():
    ep = E.episodes("sycophancy_real", n_rep=1)[1]
    inc = [(ep["subsample"][0], ep["subsample"][1])]
    p1 = E.pool(ep, round_index=1, incumbents=inc, size=64, exclude=set())
    p2 = E.pool(ep, round_index=1, incumbents=inc, size=64, exclude=set())
    assert p1 == p2 and len(p1) == 64 and len(set(map(E.candidate_key, p1))) == 64
    for c in p1:
        E.validate_candidate(c, ep)
    assert any(set(inc[0]) & set(c) for c in p1)  # mutations of incumbents are present
    p0 = E.pool(ep, round_index=0, incumbents=[], size=64, exclude=set())
    assert p0 != p1


def _ev(i, lcb, ok=True, norm=0.1):
    return {
        "candidate": (f"c{i}",),
        "order": i,
        "gain": lcb + 0.196,
        "gain_se": 0.1,
        "guards_pass": ok,
        "rel_norm": norm,
    }


def test_winner_rule_ties_and_fallback():
    evs = [_ev(0, 0.05), _ev(1, 0.20, ok=False), _ev(2, 0.10, norm=0.3), _ev(3, 0.10, norm=0.2)]
    w = E.winner(evs)
    assert w["candidate"] == ("c3",) and not w["fallback"]
    evs2 = [_ev(0, 0.10, norm=0.2), _ev(1, 0.10, norm=0.2)]
    assert E.winner(evs2)["candidate"] == ("c0",)  # then earlier evaluation
    none = [_ev(0, -0.01), _ev(1, 0.3, ok=False)]
    w0 = E.winner(none)
    assert w0["candidate"] == () and w0["fallback"] and w0["gain"] == 0.0


def test_failed_episode_rule():
    assert E.episode_outcome(attempts=[{"complete": False}] * 3)["winner"] == "base"
    assert E.episode_outcome(attempts=[{"complete": False}] * 3)["sensitivity_exclude"]
    out = E.episode_outcome(attempts=[{"complete": False}, {"complete": True, "winner": ("a",)}])
    assert out["winner"] == ("a",) and not out["sensitivity_exclude"]
    with pytest.raises(ValueError):
        E.episode_outcome(attempts=[{"complete": False}] * 4)


def test_guard_sets_and_dual_margins():
    g = E.GUARD_SETS["sycophancy_real"]
    assert "sycophancy_real" not in g and "syc_neutral" in g and len(g) == 7
    assert "gsm8k_gold_lp" not in E.GUARD_SETS["gsm8k_gold_lp"]
    assert "mathcomp_gold_lp" not in E.GUARD_SETS["mathcomp_gold_lp"]
    assert E.n_bounds("sycophancy_real") == 13  # wikitext native only; six dual guards
    ok = {g: {"pp": {"lcb": 0.0, "ucb": 0.0}, "native": {"lcb": 0.0, "ucb": 0.0}} for g in g}
    assert E.guards_pass("sycophancy_real", ok)["pass"]
    bad = json.loads(json.dumps(ok))
    bad["gsm8k_gold_lp"]["native"]["lcb"] = -0.06  # pp fine, native breach -> fail (both required)
    r = E.guards_pass("sycophancy_real", bad)
    assert not r["pass"] and r["breaches"] == ["gsm8k_gold_lp:native"]
    bad2 = json.loads(json.dumps(ok))
    bad2["wikitext_nll"]["native"]["ucb"] = 0.011
    assert not E.guards_pass("sycophancy_real", bad2)["pass"]


def test_bootstrap_t_bounds_one_sided():
    rng = np.random.default_rng(0)
    d = rng.normal(0.1, 1.0, 400)
    groups = np.arange(400) // 2
    b = E.bootstrap_t_bounds(d, groups, n_boot=999, seed=1)
    se = d.reshape(200, 2).mean(1).std(ddof=1) / np.sqrt(200)
    assert b["lcb"] < d.mean() < b["ucb"]
    assert abs((d.mean() - b["lcb"]) / se - 1.96) < 0.35


def test_guarded_score():
    ok = {"pass_points": True}
    assert E.guarded_score(0.3, guard_points_ok=True, is_base=False) == 0.3
    assert E.guarded_score(0.3, guard_points_ok=False, is_base=False) == 0.0
    assert E.guarded_score(-0.2, guard_points_ok=False, is_base=False) == -0.2
    assert E.guarded_score(0.5, guard_points_ok=True, is_base=True) == 0.0
    assert ok


def test_e1_designation():
    w = [
        {"episode": 0, "winner": (), "dev_lcb": 0.0, "rel_norm": 0.0},
        {"episode": 1, "winner": ("a",), "dev_lcb": 0.2, "rel_norm": 0.3},
        {"episode": 2, "winner": ("b",), "dev_lcb": 0.2, "rel_norm": 0.1},
    ]
    assert E.designate_e1(w)["episode"] == 2
    assert E.designate_e1([w[0]])["status"] == "not established — no feasible WM winner"


def test_winners_manifest_hash_is_canonical(tmp_path):
    rows = [
        {"arm": "wm", "episode": 0, "winner": ["a"]},
        {"arm": "statnn", "episode": 0, "winner": []},
    ]
    h1 = E.winners_manifest(rows, e1={"episode": 0}, path=tmp_path / "w.json")
    h2 = E.winners_manifest(list(reversed(rows)), e1={"episode": 0}, path=tmp_path / "w2.json")
    assert h1 == h2 and len(h1) == 64
    with pytest.raises(FileExistsError):
        E.winners_manifest(rows, e1={"episode": 0}, path=tmp_path / "w.json")
