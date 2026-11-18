"""wm.p3panel: fresh pilot-3 panels: exposure/reserved exclusion, near-duplicate groups, disjoint splits."""

import json

import pytest

from wsp.wm import p3panel as PP
from wsp.wm.panel import text_hash


def _items(texts, prefix):
    return [
        {"gtext": t, "hash": text_hash(t), "id": f"{prefix}:{text_hash(t)[:8]}", "q": t}
        for t in texts
    ]


def _fam():
    a = [f"question number {i} about topic alpha beta gamma delta" for i in range(40)]
    b = [f"another prompt {i} with words one two three four five" for i in range(30)]
    return a, b


def test_select_excludes_exposed_reserved_and_avoided_groups_and_is_disjoint():
    a, b = _fam()
    exposed = {text_hash(a[0])}
    fams = {
        "f": {
            "pools": {"p1": _items(a, "p1"), "p2": _items(a[:20] + b, "p2")},
            "reserved": {"res": [a[1]]},
            "avoid_texts": {"alp": [b[0]]},
            "avoid": {"p2": ["alp"]},
        }
    }
    sizes = {"p1": {"dev": 10, "construction": 10}, "p2": {"dev": 5, "construction": 5}}
    order = {"p1": ("dev", "construction"), "p2": ("dev", "construction")}
    res = PP.select(fams, ["p1", "p2"], sizes, "salt", exposed, order)
    sp = res["splits"]
    got = {r["q"] for ss in sp.values() for rows in ss.values() for r in rows}
    assert a[0] not in got and a[1] not in got and b[0] not in got
    assert res["stats"]["p1"]["dropped"]["exposed"] == 1
    assert res["stats"]["p1"]["dropped"]["reserved"] == 1
    assert res["stats"]["p2"]["dropped"]["avoided"] == 1
    assert PP.audit(sp, exposed)["clean"]
    # p2 never reuses a p1 group
    g1 = {r["group"] for rows in sp["p1"].values() for r in rows}
    assert not g1 & {r["group"] for rows in sp["p2"].values() for r in rows}
    again = PP.select(fams, ["p1", "p2"], sizes, "salt", exposed, order)
    assert again["splits"] == sp
    other = PP.select(fams, ["p1", "p2"], sizes, "other-salt", exposed, order)
    assert other["splits"] != sp


def test_near_duplicates_share_a_group_and_count_once():
    t = "the quick brown fox jumps over the lazy dog near the river bank today"
    texts = [t, t + "!", "completely different words for a separate question entirely here"]
    fams = {"f": {"pools": {"p": _items(texts, "p")}}}
    res = PP.select(fams, ["p"], {"p": {"dev": 2}}, "s", set(), {"p": ("dev",)})
    assert res["stats"]["p"]["usable_groups"] == 2
    with pytest.raises(ValueError):
        PP.select(fams, ["p"], {"p": {"dev": 3}}, "s", set(), {"p": ("dev",)})


def test_audit_flags_overlap():
    r = {"hash": "h", "group": "g"}
    assert not PP.audit({"a": {"dev": [r]}, "b": {"dev": [r]}}, set())["clean"]
    assert not PP.audit({"a": {"dev": [r]}}, {"h"})["clean"]


def test_write_and_read_panel(tmp_path):
    a, _ = _fam()
    fams = {"f": {"pools": {"p": _items(a, "p")}}}
    res = PP.select(
        fams, ["p"], {"p": {"dev": 4, "holdout": 3}}, "s", set(), {"p": ("dev", "holdout")}
    )
    man = PP.write_panel(tmp_path / "x", res["splits"], "s", {"label": "test"})
    assert man["objectives"]["p"]["dev"]["n"] == 4 and man["sealed_hashes"] == []
    assert not (tmp_path / "x" / "sealed_test").exists()
    back = PP.read_split(tmp_path / "x", "dev")
    assert [r["hash"] for r in back["p"]] == [r["hash"] for r in res["splits"]["p"]["dev"]]
    assert "gtext" not in back["p"][0]
    with pytest.raises(FileExistsError):
        PP.write_panel(tmp_path / "x", res["splits"], "s", {})
    with pytest.raises(PermissionError):
        PP.read_split(tmp_path / "x", "sealed_test/test")
    # the manifest is readable by the exposure registry reader
    from wsp.wm import exposure as X

    hs = X.manifest_hashes(tmp_path / "x" / "manifest.json")
    assert hs["dev"] == set(man["objectives"]["p"]["dev"]["hashes"])
    assert json.loads((tmp_path / "x" / "manifest.json").read_text())["label"] == "test"


def test_used_groups_from_an_earlier_panel_set_are_never_reused():
    a, _ = _fam()
    fams = {"f": {"pools": {"s": _items(a, "s"), "h": _items(a, "h")}}}
    g = PP.group_all(fams)
    rs = PP.select(fams, ["s"], {"s": {"dev": 10}}, "salt-s", set(), {"s": ("dev",)}, grouped=g)
    used = {r["group"] for r in rs["splits"]["s"]["dev"]}
    rh = PP.select(
        fams,
        ["h"],
        {"h": {"dev": 30}},
        "salt-h",
        set(),
        {"h": ("dev",)},
        used_groups=used,
        grouped=g,
    )
    assert not used & {r["group"] for r in rh["splits"]["h"]["dev"]}
    assert PP.audit({"s": rs["splits"]["s"], "h": rh["splits"]["h"]}, set())["clean"]
