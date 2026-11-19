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
