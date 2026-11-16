"""MSRM-1 development v0: folds, closure audit, contrasts, bootstrap and development kill criteria."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import f12_simulation as fs
from wsp.msrm import closure as cl
from wsp.msrm import evaluate as ev
from wsp.msrm import folds as fo

ROOT = Path(__file__).resolve().parents[1]
POPULATION = ROOT / "docs/prereg/a5_manifests/f12/population.json"


# ------------------------------------------------------------------------------------------- folds


def test_outer_folds_follow_the_f12_rule():
    rng = np.random.default_rng(0)
    comps = [f"{i:064x}" for i in rng.integers(0, 2**62, size=40)]
    fams = ["chain"] * 20 + ["lora_sft"] * 10 + ["abliteration"] * 5 + ["random_dir"] * 5
    assert fo.outer_folds(comps, fams) == fs.outer_folds(comps, fams)
    assert fo.FOLD_DOMAIN == "20261111:f12:" and fo.INNER_DOMAIN == "20261110:lol-inner:"


def test_existing_population_folds_equal_the_frozen_s0_manifest():
    pop = json.loads(POPULATION.read_text())
    rows = [r for r in pop["rows"] if r["source"] == "existing-TF-v1"]
    fold = fo.outer_folds([r["component_id"] for r in rows], [r["family"] for r in rows])
    s0 = next(s for s in pop["s_k"] if s["k"] == 0)
    for f in s0["fold_summary"]:
        ids = sorted(r["id"] for r in rows if fold[r["component_id"]] == f["fold"])
        assert fo.ids_sha256(ids) == f["row_ids_sha256"]


def test_closure_groups_never_cross_a_fold():
    comps = ["A", "B", "C", "D", "E", "F", "G"]
    fams = ["chain"] * 7
    merged = fo.merge_components(comps, {"B": "A", "E": "D"})
    fold = fo.outer_folds(merged, fams)
    assert fold[merged[0]] == fold[merged[1]]  # A and B share a closure group
    assert merged[3] == merged[4]
    fo.assert_component_disjoint(merged, [fold[c] for c in merged])
    with pytest.raises(ValueError):
        fo.assert_component_disjoint(["A", "A"], [0, 1])


def test_leave_one_family_out_keeps_mixed_in_training():
    rows = [
        ("r1", "c1", "chain"),
        ("r2", "c2", "lora_sft"),
        ("r3", "cm", "chain"),
        ("r4", "cm", "random_dir"),
        ("r5", "c5", "random_dir"),
    ]
    strata = fo.component_strata([c for _, c, _ in rows], [f for _, _, f in rows])
    assert strata["cm"] == "mixed"
    train, test = fo.lofo_split([c for _, c, _ in rows], [f for _, _, f in rows], "random_dir")
    assert set(test) == {4} and {2, 3} <= set(train)
    train, test = fo.lofo_split([c for _, c, _ in rows], [f for _, _, f in rows], "chain")
    assert set(test) == {0} and {2, 3} <= set(train)


# ---------------------------------------------------------------------------------- closure audit


def _zoo():
    return {
        "p1": {"component_id": "C1", "parent_id": None, "family": "chain"},
        "p1s1": {"component_id": "C1", "parent_id": "p1", "family": "chain"},
        "x": {"component_id": "C2", "parent_id": None, "family": "random_dir"},
        "y": {"component_id": "C3", "parent_id": None, "family": "random_dir"},
        "z": {"component_id": "C4", "parent_id": None, "family": "lora_sft"},
        "w": {"component_id": "C5", "parent_id": None, "family": "lora_sft"},
        "pa": {"component_id": "PA", "parent_id": None, "family": "lora_sft"},
        "v": {"component_id": "C6", "parent_id": None, "family": "abliteration"},
    }


def test_closure_audit_joins_and_flags():
    zoo = _zoo()
    edits = {
        "p1": {"provenance": {"generator_edit_id": "g1"}},
        "p1s1": {"provenance": {"generator_edit_id": "g1"}},
        "x": {
            "family": "random_dir",
            "seed": 5,
            "params": {"layers": [3], "n_directions": 1},
            "provenance": {"reuse_key": "env"},
        },
        "y": {  # chain step whose inner random_dir draws direction seed 5 + 3*1000 like x
            "family": "chain",
            "params": {
                "inner_family": "random_dir",
                "inner_seed": 5,
                "inner_params": {"layers": [3, 7], "n_directions": 2},
            },
            "provenance": {"reuse_key": "env"},  # a shared environment key is not a join
        },
        "z": {"provenance": {"generator_edit_id": "g9"}},
        "w": {"provenance": {"generator_edit_id": "g9"}},  # shared source with z
        "v": {"family": "abliteration", "params": {"direction_index": 3.0}, "provenance": {}},
    }
    delta_hash = {"x": "h1", "y": "h2", "z": "h3", "w": "h4", "v": "h5", "p1": "h0"}
    audit = cl.closure_audit(
        population=["p1", "x", "y", "z", "w", "v"],
        zoo=zoo,
        edits=edits,
        delta_hash=delta_hash,
        part_a_components={"PA"},
    )
    assert audit["merge"] == {"C3": "C2", "C5": "C4"}
    assert "reuse_key" not in audit["edge_counts"]
    assert audit["excluded"] == {}
    kinds = {e["kind"] for e in audit["cross_component_joins"]}
    assert kinds == {"direction_seed", "generator"}
    # a join into a Part A component excludes the row; missing provenance excludes the row
    delta_hash["v"] = "hpa"
    delta_hash["pa"] = "hpa"
    edits2 = dict(edits)
    del edits2["z"]
    audit2 = cl.closure_audit(
        population=["p1", "x", "y", "z", "w", "v"],
        zoo=zoo,
        edits=edits2,
        delta_hash=delta_hash,
        part_a_components={"PA"},
    )
    assert audit2["excluded"]["v"].startswith("joins_part_a")
    assert audit2["excluded"]["z"] == "missing_provenance"


def test_random_direction_keys_are_per_direction_seeds():
    e = {"family": "random_dir", "seed": 113, "params": {"layers": [21], "n_directions": 2}}
    assert cl.direction_keys(e) == [
        ("random_direction", 113 + 21000),
        ("random_direction", 113 + 21100),
    ]
    same_seed_other_layer = {
        "family": "random_dir",
        "seed": 113,
        "params": {"layers": [4], "n_directions": 1},
    }
    assert not set(cl.direction_keys(e)) & set(cl.direction_keys(same_seed_other_layer))
    abl = {
        "family": "chain",
        "params": {"inner_family": "abliteration", "inner_params": {"direction_index": 18.7}},
        "provenance": {"refusal_fingerprint": "fp"},
    }
    assert cl.direction_keys(abl) == [("abliteration_direction", "fp", 18.7)]


def test_closure_audit_never_needs_part_a_edit_files():
    zoo = _zoo()
    edits = cl.EditView({k: {"provenance": {}} for k in zoo if k != "pa"}, forbidden={"pa"})
    with pytest.raises(PermissionError):
        edits["pa"]
    audit = cl.closure_audit(["x"], zoo, edits, {}, part_a_components={"PA"})
    assert audit["excluded"] == {}


# ------------------------------------------------------------------------------ contrasts/bootstrap


def test_contrast_is_ratio_of_sums():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.5, 2.0, 2.0])
    r = np.array([0.0, 0.0, 0.0])
    assert ev.contrast(y, p, r) == pytest.approx(1 - 1.5 / 6.0)
    assert ev.contrast(y, r, r) == 0.0


def test_component_bootstrap_resamples_whole_components():
    rng = np.random.default_rng(0)
    comps = np.repeat(np.arange(20), 3).astype(str)
    y = rng.normal(size=60)
    pa, pb = y + 0.1 * rng.normal(size=60), y + rng.normal(size=60)
    b1 = ev.component_bootstrap(y, {"m": pa, "ref": pb}, comps, "ref", draws=500, seed=20261107)
    b2 = ev.component_bootstrap(y, {"m": pa, "ref": pb}, comps, "ref", draws=500, seed=20261107)
    assert b1 == b2
    lo, hi = b1["m"]["ci95"]
    assert lo < b1["m"]["C"] < hi and lo > 0.5
    assert b1["m"]["draws"] == 500 and b1["m"]["valid_draws"] == 500


def test_development_kill_criteria():
    ok = {
        "msrm_vs_f3u": {"C": 0.12, "ci95": (0.02, 0.2)},
        "msrm_vs_desc": {"C": 0.10, "ci95": (0.01, 0.2)},
        "msrm_vs_scripted": {"C": 0.08, "ci95": (0.005, 0.15)},
    }
    k = ev.kill_criteria(ok)
    assert not k["k1"]["fires"] and not k["k2"]["fires"] and not k["k2b"]["fires"]
    k = ev.kill_criteria(dict(ok, msrm_vs_f3u={"C": 0.04, "ci95": (-0.1, 0.3)}))
    assert k["k1"]["fires"]
    k = ev.kill_criteria(dict(ok, msrm_vs_f3u={"C": 0.06, "ci95": (-0.1, 0.09)}))
    assert k["k1"]["fires"]  # upper bound < 0.10
    k = ev.kill_criteria(dict(ok, msrm_vs_desc={"C": 0.03, "ci95": (-0.02, 0.1)}))
    assert k["k2"]["fires"]
    k = ev.kill_criteria(dict(ok, msrm_vs_scripted={"C": -0.01, "ci95": (-0.1, 0.1)}))
    assert k["k2b"]["fires"]
    assert all("development decision" in v["meaning"] for v in k.values())
