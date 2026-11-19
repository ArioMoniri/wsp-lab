"""Joint MMLU-Pro allocation, builder consumption, W2 precondition, shortfall and the hash-only audit
(wsp.wm.panels_v2d; Board A v2 required change 3). Synthetic fixtures only."""

import hashlib
import itertools

import pytest

from wsp.wm import exposure as X
from wsp.wm import neardup as ND
from wsp.wm import panels_v2d as V
from wsp.wm.panel import text_hash


def _words(i, tag, n=8):
    h = hashlib.sha256(f"{tag}{i}".encode()).hexdigest() * 2
    return " ".join(h[j : j + 6] for j in range(0, 6 * n, 6))


def _mmlu(n, tag="m"):
    out = []
    for i in range(n):
        q = "Question " + _words(i, tag)
        opts = [f"opt{k} {_words(i, tag + str(k), 2)}" for k in range(4)]
        it = {"question": q, "options": opts, "answer": "A"}
        it["hash"] = text_hash(X.group_text("mmlupro_gold_lp", it))
        out.append(it)
    # a near-duplicate of item 0 under both policies (one changed character at the end)
    dup = dict(out[0], question=out[0]["question"] + "!")
    dup["hash"] = text_hash(X.group_text("mmlupro_gold_lp", dup))
    return out + [dup]


def _pool(items):
    return {r["hash"]: X.group_text("mmlupro_gold_lp", r) for r in items}


ALL_EXP = {e: {"test": set()} for e in V.REQUIRED_EXPOSURES}


def test_joint_groups_is_at_least_as_coarse_as_each_policy():
    items = _mmlu(25)
    texts = _pool(items)
    j = V.joint_groups(texts)
    for rep in (ND.groups(texts), X.near_duplicate_groups(texts)):
        for a, b in itertools.combinations(texts, 2):
            if rep[a] == rep[b]:
                assert j[a] == j[b]
    assert j[items[0]["hash"]] == j[items[-1]["hash"]]


def test_allocation_is_deterministic_salted_and_group_whole():
    items = _mmlu(40)
    exposed = {items[5]["hash"]}
    cons = (("c1", 6, "raise"), ("c2", 10, "reduce"))
    a = V.allocate(_pool(items), exposed, cons)
    b = V.allocate(_pool(items), exposed, cons)
    assert a == b and len(a["sha256"]) == 64
    assert items[5]["hash"] not in a["assignment"] and a["excluded_groups"] == 1
    assert a["assignment"][items[0]["hash"]] == a["assignment"][items[-1]["hash"]]
    got = {c: sum(v == c for v in a["assignment"].values()) for c in ("c1", "c2", "reserve")}
    assert got["c1"] >= 6 and got["c2"] >= 10 and sum(got.values()) == len(items) - 1
    other = V.allocate(_pool(items), exposed, cons, salt="another-salt")
    assert other["assignment"] != a["assignment"]
    V.verify_allocation(a)
    bad = dict(a, salt="tampered")
    with pytest.raises(ValueError):
        V.verify_allocation(bad)


def test_shortfall_reduces_c2_and_raises_for_c1():
    items = _mmlu(12)
    a = V.allocate(_pool(items), set(), (("c1", 4, "raise"), ("c2", 50, "reduce")))
    c2 = a["consumers"][1]
    assert c2["shortfall"] and c2["restate_power"] and c2["frozen_n"] == a["fresh_groups"] - 4
    with pytest.raises(ValueError):
        V.allocate(_pool(items), set(), (("c1", 40, "raise"),))


def test_builders_require_wme_w2(tmp_path):
    exp = {e: {"test": set()} for e in V.REQUIRED_EXPOSURES if e != "wme-w2"}
    alloc = V.allocate(_pool(_mmlu(5)), set(), (("x", 1, "raise"),))
    with pytest.raises(V.MissingExposure):
        V.build_c1_panel_v2d({}, {}, tmp_path / "c1", exp, alloc)
    with pytest.raises(V.MissingExposure):
        V.build_c2gen_panel({"mmlupro_gen": []}, tmp_path / "c2", exp, alloc, {})


def _gen(n, tag):
    out = []
    for i in range(n):
        t = f"{tag} " + _words(i, tag)
        out.append({"hash": text_hash(t), "gtext": t, "prompt": t})
    return out


def test_both_builders_consume_the_one_allocation(tmp_path):
    mm = _mmlu(60)
    alloc = V.allocate(
        _pool(mm),
        set(),
        (("wme-c1:mmlupro_gold_lp", 8, "raise"), ("wme-c2-gen:mmlupro_gen", 20, "reduce")),
    )
    a = alloc["assignment"]
    sizes = {
        "mmlupro_gold_lp": {
            "sealed_gen": 0,
            "sealed_test": 3,
            "dev": 2,
            "bank": 2,
            "construction": 1,
        }
    }
    m1 = V.build_c1_panel_v2d({"mmlupro_gold_lp": mm}, sizes, tmp_path / "c1", ALL_EXP, alloc)
    used1 = {h for s in m1["objectives"]["mmlupro_gold_lp"].values() for h in s["hashes"]}
    assert used1 and all(a[h] == "wme-c1:mmlupro_gold_lp" for h in used1)
    assert m1["mmlu_allocation_sha256"] == alloc["sha256"]
    mm_c2 = [dict(r, gtext=X.group_text("mmlupro_gold_lp", r)) for r in mm]
    pools = {
        "mmlupro_gen": mm_c2,
        "supergpqa_gen": _gen(30, "sg"),
        "guard_harmless": _gen(30, "gh"),
        "guard_harmful": _gen(30, "gx"),
    }
    sizes2 = {
        "mmlupro_gen": {
            "sealed_test": 50
        },  # more than the allocation's share: shortfall -> reduced
        "supergpqa_gen": {"sealed_test": 10},
        "guard_harmless": {"calibration": 4, "sealed_test": 8},
        "guard_harmful": {"calibration": 4, "sealed_test": 8},
    }
    m2 = V.build_c2gen_panel(pools, tmp_path / "c2", ALL_EXP, alloc, {}, sizes=sizes2)
    used2 = set(m2["objectives"]["mmlupro_gen"]["sealed_test"]["hashes"])
    assert used2 and all(a[h] == "wme-c2-gen:mmlupro_gen" for h in used2)
    assert not used1 & used2
    sf = m2["shortfall"]["mmlupro_gen"]
    assert sf["shortfall"] and sf["restate_power"] and sf["frozen_n"] == len(used2) < 50
    assert not m2["shortfall"]["supergpqa_gen"]["shortfall"]
    assert (tmp_path / "c2/sealed_test/sealed_test.jsonl").exists()
    assert (tmp_path / "c2/open/calibration.jsonl").exists()
    # cross-panel hash-only audit
    texts = {r["hash"]: X.group_text("mmlupro_gold_lp", r) for r in mm}
    for p in ("supergpqa_gen", "guard_harmless", "guard_harmful"):
        texts.update({r["hash"]: r["gtext"] for r in pools[p]})
    panels = {
        "wme-c1": {s: v["hashes"] for s, v in m1["objectives"]["mmlupro_gold_lp"].items()},
        **{
            f"wme-c2-gen/{p}": {s: v["hashes"] for s, v in m2["objectives"][p].items()}
            for p in m2["objectives"]
        },
    }
    rep = V.audit(panels, texts, ALL_EXP)
    assert rep["clean"] and len(rep["sha256"]) == 64
    # a duplicated item across panels is caught by hash and by joint group
    panels["wme-c2-gen/mmlupro_gen"]["sealed_test"] = list(used2) + [next(iter(used1))]
    assert not V.audit(panels, texts, ALL_EXP)["clean"]


def test_preassigned_groups_go_whole_to_their_consumer_even_if_exposed():
    items = _mmlu(30)
    pre = {items[0]["hash"]: "roots", items[3]["hash"]: "roots"}
    exposed = {items[3]["hash"], items[7]["hash"]}
    a = V.allocate(
        _pool(items), exposed, (("c1", 4, "raise"), ("c2", 5, "reduce")), preassigned=pre
    )
    assert a["assignment"][items[0]["hash"]] == a["assignment"][items[-1]["hash"]] == "roots"
    assert a["assignment"][items[3]["hash"]] == "roots"  # exposed later, still committed
    assert items[7]["hash"] not in a["assignment"]
    assert a["preassigned_groups"] == 2 and a["preassigned_exposed_groups"] == 1


def test_c1_v2e_forces_root_shards_into_construction(tmp_path):
    mm = _mmlu(60)
    reserved = {mm[1]["hash"], mm[2]["hash"]}
    exp = dict(
        ALL_EXP, later={"dev": {mm[2]["hash"]}}
    )  # one root-shard item exposed by a later panel
    alloc = V.allocate(
        _pool(mm),
        X.all_exposed(exp),
        (("wme-c1:mmlupro_gold_lp", 10, "raise"), ("wme-c2-gen:mmlupro_gen", 20, "reduce")),
        preassigned={h: V.C1_ROOT_CONSUMER for h in reserved},
    )
    sizes = {
        "mmlupro_gold_lp": {
            "sealed_gen": 0,
            "sealed_test": 4,
            "dev": 3,
            "bank": 2,
            "construction": "rest",
        }
    }
    man, _splits = V.build_c1_panel_v2e(
        {"mmlupro_gold_lp": mm},
        sizes,
        tmp_path / "c1",
        exp,
        alloc,
        reserved,
        construction_min={"mmlupro_gold_lp": 1},
    )
    o = man["objectives"]["mmlupro_gold_lp"]
    assert set(reserved) <= set(o["construction"]["hashes"])
    for s in ("sealed_test", "dev", "bank"):
        assert not set(reserved) & set(o[s]["hashes"])
    assert o["sealed_test"]["n"] == 4 and o["dev"]["n"] == 3 and o["bank"]["n"] == 2
    assert man["construction_exposure_hits_reserved_roots"] == 1
    assert man["exclusion"]["mmlupro_gold_lp"]["reserved_groups_exposed_later"] == 1
    # only the allocation's c1 shares are used
    a = alloc["assignment"]
    used = {h for s in o.values() for h in s["hashes"]}
    assert all(a[h] in ("wme-c1:mmlupro_gold_lp", V.C1_ROOT_CONSUMER) for h in used)


def test_registry_before_orders_the_confirmatory_panels():
    reg = X.registry_before("wme-c1")
    assert "wme-w2" in reg and "wme-c1" not in reg and "wme-c2-gen" not in reg
    reg2 = X.registry_before("wme-c2-gen")
    assert "wme-c1" in reg2 and "wme-c2-gen" not in reg2


def test_consumers_are_dealt_only_groups_with_an_item_in_their_own_pool():
    items = _mmlu(40)
    c1_pool = {r["hash"] for r in items[:20]}
    a = V.allocate(
        _pool(items), set(), (("c1", 8, "raise"), ("c2", 10, "reduce")), eligible={"c1": c1_pool}
    )
    rep = V.joint_groups(_pool(items))
    groups = {rep[h] for h, c in a["assignment"].items() if c == "c1"}
    assert groups and all(any(rep[h] == g for h in c1_pool) for g in groups)
    assert a["consumers"][0]["assigned_groups"] == 8
    with pytest.raises(ValueError):
        V.allocate(_pool(items), set(), (("c1", 25, "raise"),), eligible={"c1": c1_pool})


def test_c2_groups_mmlu_and_supergpqa_jointly_and_reserves_c1_texts(tmp_path):
    mm = [dict(r, gtext=X.group_text("mmlupro_gold_lp", r)) for r in _mmlu(30)]
    alloc = V.allocate(
        _pool(mm),
        set(),
        (("wme-c1:mmlupro_gold_lp", 2, "raise"), ("wme-c2-gen:mmlupro_gen", 25, "reduce")),
    )
    share = [r for r in mm if alloc["assignment"].get(r["hash"]) == "wme-c2-gen:mmlupro_gen"]
    twin_text = (
        share[0]["gtext"] + "?"
    )  # a SuperGPQA near-duplicate of an MMLU-Pro item in c2's share
    sg = _gen(12, "sg") + [{"hash": text_hash(twin_text), "gtext": twin_text, "prompt": twin_text}]
    reserved_hit = _gen(1, "gh")[0]["gtext"]  # a guard prompt already in a wme-c1 pool
    pools = {
        "mmlupro_gen": mm,
        "supergpqa_gen": sg,
        "guard_harmless": _gen(20, "gh"),
        "guard_harmful": _gen(20, "gx"),
    }
    sizes = {
        "mmlupro_gen": {"sealed_test": 40},
        "supergpqa_gen": {"sealed_test": 40},
        "guard_harmless": {"calibration": 2, "sealed_test": 40},
        "guard_harmful": {"calibration": 2, "sealed_test": 40},
    }
    m = V.build_c2gen_panel(
        pools, tmp_path / "c2", ALL_EXP, alloc, {"c1": [reserved_hit]}, sizes=sizes
    )
    sg_used = set(m["objectives"]["supergpqa_gen"]["sealed_test"]["hashes"])
    assert text_hash(twin_text) not in sg_used  # owned by MMLU-Pro (first in the family)
    gh_used = {h for s in m["objectives"]["guard_harmless"].values() for h in s["hashes"]}
    assert text_hash(reserved_hit) not in gh_used and len(gh_used) == 19
    assert (
        m["shortfall"]["supergpqa_gen"]["shortfall"]
        and m["shortfall"]["supergpqa_gen"]["restate_power"]
    )


def test_c1_keeps_the_root_item_as_its_groups_representative(tmp_path):
    mm = _mmlu(40)
    root_item = mm[-1]["hash"]  # the near-duplicate of item 0; item 0 has the other hash
    alloc = V.allocate(
        _pool(mm),
        set(),
        (("wme-c1:mmlupro_gold_lp", 10, "raise"), ("wme-c2-gen:mmlupro_gen", 5, "reduce")),
        preassigned={root_item: V.C1_ROOT_CONSUMER},
    )
    sizes = {
        "mmlupro_gold_lp": {
            "sealed_gen": 0,
            "sealed_test": 3,
            "dev": 2,
            "bank": 2,
            "construction": "rest",
        }
    }
    man, _ = V.build_c1_panel_v2e(
        {"mmlupro_gold_lp": mm}, sizes, tmp_path / "c1", ALL_EXP, alloc, {root_item}
    )
    con = man["objectives"]["mmlupro_gold_lp"]["construction"]["hashes"]
    assert root_item in con and mm[0]["hash"] not in con


def test_c1_field_minimum_counts_fresh_construction_only(tmp_path):
    mm = _mmlu(40)
    reserved = {r["hash"] for r in mm[20:30]}
    alloc = V.allocate(
        _pool(mm),
        set(),
        (("wme-c1:mmlupro_gold_lp", 9, "raise"), ("wme-c2-gen:mmlupro_gen", 5, "reduce")),
        preassigned={h: V.C1_ROOT_CONSUMER for h in reserved},
    )
    sizes = {
        "mmlupro_gold_lp": {
            "sealed_gen": 0,
            "sealed_test": 3,
            "dev": 2,
            "bank": 2,
            "construction": "rest",
        }
    }
    fresh = V.build_c1_panel_v2e(
        {"mmlupro_gold_lp": mm}, sizes, tmp_path / "x", ALL_EXP, alloc, reserved, dry_run=True
    )
    assert fresh["mmlupro_gold_lp"] >= 7 and not (tmp_path / "x").exists()
    with pytest.raises(
        ValueError
    ):  # 10 root-shard groups do not count toward a fresh field minimum
        V.build_c1_panel_v2e(
            {"mmlupro_gold_lp": mm},
            sizes,
            tmp_path / "y",
            ALL_EXP,
            alloc,
            reserved,
            construction_min={"mmlupro_gold_lp": fresh["mmlupro_gold_lp"] - 7 + 1},
        )
