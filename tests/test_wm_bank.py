"""wm.bank + wm.wmclosure: the trajectory-bank recipe (PREREG_WME §4; Board A conditions 2, 3, 12). Recipes only;
nothing is realised or evaluated.

* 96 roots, 16 per family, namespace ``wme-bank``, disjoint from every pilot root, tensor, contrast pair, token and
  seed namespace; immutable identities per root;
* frozen nominals, caps, signed realisation and duplicate-state collapse for all six families; ladders
  ±{0.25, 0.5, 1, 2, 4, 8}; side order positive then negative, outward, stop after the first ΔNLL > 1.0 state;
* horizon = number of steps on the registered trajectory; training eligibility h <= 2;
* 24 AB/BA 4-step bundles; closure components merged transitively; five outer folds by component signature.
"""

import torch

from wsp.wm import bank as B
from wsp.wm import wmclosure as W


def test_namespace_and_seeds_are_disjoint_from_pilot_namespaces():
    from wsp.wm.p2data import seed_for as p2_seed

    assert B.NAMESPACE == "wme-bank"
    assert B.seed_for("roots") != p2_seed("roots")
    assert B.seed_for("x") == B.seed_for("x")


def test_ladder_and_frozen_nominals_cover_all_six_families():
    assert B.LADDER == (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert set(B.NOMINALS) == set(B.FAMILIES) and len(B.FAMILIES) == 6
    assert B.DNLL_STOP == 1.0 and B.SIDE_ORDER == ("pos", "neg")


def test_multiplier_cap_and_duplicate_collapse():
    # block / row scale: m(s) = max(0, 1 + s (f - 1)), f = 1.5: s <= -2 zeroes the block, later doses collapse
    assert B.effective("scale", -2.0)["multiplier"] == 0.0
    assert B.effective("scale", -8.0)["capped"]
    pos, neg = B.ladder_sides("scale")
    assert pos == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] and neg == [-0.25, -0.5, -1.0, -2.0]
    # projection removal: alpha = 0.5 s, capped at 1 on the positive side; +4 and +8 duplicate +2
    pos, neg = B.ladder_sides("proj_remove")
    assert pos == [0.25, 0.5, 1.0, 2.0] and len(neg) == 6
    assert B.effective("proj_remove", 8.0)["alpha"] == 1.0
    for fam in ("noise", "steer", "lora"):
        pos, neg = B.ladder_sides(fam)
        assert len(pos) == len(neg) == 6
    assert B.effective("noise", -1.0)["sign"] == -1  # signed realisation: -G on the negative side


def test_walk_side_stops_after_first_dnll_endpoint():
    out = B.walk_side([0.25, 0.5, 1.0, 2.0], lambda d: {0.25: 0.01, 0.5: 0.2, 1.0: 1.5, 2.0: 3}[d])
    assert [s["dose"] for s in out] == [0.25, 0.5, 1.0] and out[-1]["endpoint"]
    assert [s["h"] for s in out] == [1, 2, 3]


def test_ninety_six_roots_sixteen_per_family_with_identities():
    roots = B.make_roots()
    assert len(roots) == 96
    for fam in B.FAMILIES:
        assert sum(r["family"] == fam for r in roots.values()) == 16
    hashes = {r["identity"]["recipe"] for r in roots.values()}
    assert len(hashes) == 96
    for r in roots.values():
        assert r["identity"]["direction"] is not None
        assert r["layer_support"] in ("support", "vocab")


def test_roots_are_disjoint_from_every_pilot_root():
    roots = B.make_roots()
    audit = B.pilot_overlap(roots)
    assert audit["tensor_overlap"] == [] and audit["pair_overlap"] == []
    assert audit["seed_namespace"] == "wme-bank"
    assert set(audit["declared_concept_overlap"]) == {"lora:gsm8k_gold_lp", "lora:mathcomp_gold_lp"}
    for r in roots.values():
        if r["family"] == "scale":
            assert tuple(r["tensor"]) not in B.PILOT_SCALE_TENSORS
        if r["family"] == "steer":
            assert frozenset(r["pair"]) not in B.PILOT_PAIRS


def test_disjoint_construction_subsets_and_token_sets():
    roots = B.make_roots()
    used = {}
    for rid, r in roots.items():
        for s in r["identity"].get("contrast_source") or []:
            used.setdefault(s, set()).add(r["identity"]["concept"])
        if r["identity"].get("adapter_source"):
            assert r["identity"]["adapter_source"] not in used
            used[r["identity"]["adapter_source"]] = {rid}
    assert all(len(v) == 1 for v in used.values())  # a subset never feeds two concepts
    sets = [r["rows_rule"]["set"] for r in roots.values() if r["family"] == "row_scale"]
    ranks = [r["token_rule"]["rank"] for r in roots.values() if r.get("token_rule")]
    assert len(sets) == len(set(sets)) == 16 and len(ranks) == len(set(ranks)) == 8


def test_compositions_are_24_ab_ba_four_step_bundles():
    roots = B.make_roots()
    comps = B.make_compositions(roots)
    assert len(comps) == 24
    for i in range(0, 24, 2):
        assert comps[i + 1] == tuple(reversed(comps[i]))
        fams = [roots[r]["family"] for r in comps[i]]
        assert len(comps[i]) == 4 and len(set(fams)) == 4
    base_roots = [r for c in comps[::2] for r in c]
    assert len(base_roots) == len(set(base_roots))  # each root in at most one AB/BA bundle


def test_components_merge_transitively_through_compositions():
    roots = {
        "a": {"family": "x", "identity": {"recipe": "1", "direction": "da"}},
        "b": {"family": "y", "identity": {"recipe": "2", "direction": "db", "concept": "k"}},
        "c": {"family": "y", "identity": {"recipe": "3", "direction": "dc", "concept": "k"}},
        "d": {"family": "z", "identity": {"recipe": "4", "direction": "dd"}},
        "e": {"family": "z", "identity": {"recipe": "5", "direction": "dd"}},  # shared direction
    }
    comp = W.components(roots, compositions=[("a", "b")])
    assert comp["a"] == comp["b"] and comp["c"] != comp["a"] and comp["d"] == comp["e"]
    comp2 = W.components(roots, compositions=[("a", "b")], policy="concept")
    assert comp2["c"] == comp2["a"]
    merged = W.join_realised(comp, {"a": ["h1"], "d": ["h1"]})
    assert merged["a"] == merged["d"] == merged["e"]


def test_bank_component_graph_and_folds_are_published():
    rec = B.bank_recipe()
    for policy in ("source", "concept"):
        g = rec["components"][policy]
        assert g["n_components"] >= 5
        assert g["folds"]["five_folds_feasible"]
        assert sum(v["components"] for v in g["folds"]["per_fold"].values()) == g["n_components"]
        sigs = {e["signature"] for e in g["table"].values()}
        assert any("+" in s for s in sigs)  # multi-family components keep their full signature
    assert (
        rec["components"]["concept"]["n_components"] <= rec["components"]["source"]["n_components"]
    )
    # every state of a root and every composition prefix is in its root's component
    comp = rec["components"]["source"]["assignment"]
    for seq in rec["compositions"]:
        assert len({comp[r] for r in seq}) == 1


def test_horizon_rule_and_training_eligibility():
    rec = B.bank_recipe()
    hr = rec["horizon_rule"]
    assert hr["train_max_h"] == 2 and hr["test_h"] == [3, 4]
    tr = B.trajectories(rec)
    lad = [t for t in tr if t["kind"] == "ladder"]
    assert all(s["h"] == i + 1 for t in lad for i, s in enumerate(t["states"]))
    comp = [t for t in tr if t["kind"] == "composition"]
    assert all(len(t["states"]) == 4 for t in comp)
    elig = B.training_states(tr)
    assert all(s["h"] <= 2 for s in elig)
    assert not any(abs(s["dose"]) > 0.5 for s in elig if s.get("kind") == "ladder")


def test_inner_allocation_and_state_counts():
    rec = B.bank_recipe()
    assert (
        rec["n_states_max"]
        == sum(rec["ladder_states_per_root"].values()) + rec["composition_states"]
    )
    assert rec["composition_states"] == 72
    for alloc in rec["inner_allocation"]["source"].values():
        assert set(alloc.values()) <= {"fit", "selection", "calibration"}
        assert "calibration" in alloc.values()


def test_direction_similarity_audit():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(64, generator=g)
    new = {"n1": a, "n2": torch.randn(64, generator=g)}
    pilot = {"p": a * 3}
    out = W.direction_similarity_audit(new, pilot, max_cos=0.5)
    assert out["flagged"] == ["n1"] and not out["pass"]


def test_recipe_manifest_is_deterministic():
    assert B.bank_recipe()["recipe_hash"] == B.bank_recipe()["recipe_hash"]


def test_composite_identities_do_not_join_on_shared_elements():
    roots = {
        "a": {"family": "x", "identity": {"recipe": "1", "direction": ["scale", [12, "down"], 0]}},
        "b": {"family": "x", "identity": {"recipe": "2", "direction": ["scale", [12, "down"], 1]}},
        "c": {
            "family": "y",
            "identity": {"recipe": "3", "direction": "dc", "contrast_source": ["s1", "s2"]},
        },
        "d": {
            "family": "y",
            "identity": {"recipe": "4", "direction": "dd", "contrast_source": ["s2", "s3"]},
        },
    }
    comp = W.components(roots)
    assert comp["a"] != comp["b"] and comp["c"] == comp["d"]
