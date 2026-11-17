"""wm.panel_c1: the wme-c1 confirmatory panel builder (PREREG_WME §3; Board A condition 1). Synthetic pools only:
the panel itself is not built here.

* salt ``wme-c1-panel-20261117``; ranking by sha256(salt:group);
* group-level exclusion of every exposure (pilot 1, pilot 2, 27B calibration, smoke) by manifest hash;
* near-duplicate groups stay in one split; a group's items never straddle splits or objectives;
* four disjoint splits, test written only under sealed_test/; disjointness asserted in memory on every split;
* the sealed-test reader: one read, only after every arm's winner is frozen, content hash checked against the
  manifest, realpath + O_NOFOLLOW.
"""

import json

import pytest

from wsp.wm import panel_c1 as C1
from wsp.wm.plan import PlannerItemStore, SealedError

SIZES = {"syc": {"construction": 2, "validation": 2, "dev": 2, "test": 2}}


def _pool(n=30, extra=()):
    rows = []
    for i in range(n):
        q = f"question number {i} about a topic that is unique to item {i} with words"
        rows.append({"question": q, "hash": f"h{i:03d}", "id": f"sya:{i}"})
    rows.extend(extra)
    return {"syc": rows}


def _gtext(obj, r):
    return r["question"]


def _build(tmp_path, pool=None, exposures=None, sizes=SIZES):
    return C1.build_c1_panel(
        pool or _pool(), sizes, tmp_path / "wme-c1", exposures or {}, group_text=_gtext
    )


def test_salt_and_splits():
    assert C1.SALT == "wme-c1-panel-20261117"
    assert C1.SPLITS == ("test", "dev", "validation", "construction")


def test_four_disjoint_splits_and_sealed_layout(tmp_path):
    man = _build(tmp_path)
    root = tmp_path / "wme-c1"
    assert (root / "sealed_test" / "test.jsonl").exists() and not (
        root / "open" / "test.jsonl"
    ).exists()
    hs = {s: set(man["objectives"]["syc"][s]["hashes"]) for s in C1.SPLITS}
    for a in C1.SPLITS:
        for b in C1.SPLITS:
            if a < b:
                assert not hs[a] & hs[b]
    assert man["salt"] == C1.SALT and man["content_hash"]
    assert set(man["split_roles"]) == {"construction", "validation", "dev", "test"}


def test_exposed_groups_and_their_near_duplicates_are_excluded(tmp_path):
    dup = {
        "question": "question number 3 about a topic that is unique to item 3 with words too",
        "hash": "hdup",
        "id": "sya:dup",
    }
    exposures = {"wme-p2": {"test": {"h003"}}, "wme-v0-p1": {"construction": {"h005"}}}
    man = _build(tmp_path, pool=_pool(extra=[dup]), exposures=exposures)
    written = {h for s in man["objectives"]["syc"].values() for h in s["hashes"]}
    assert not written & {"h003", "hdup", "h005"}
    assert man["exclusion"]["syc"]["excluded_items"] == 3
    assert man["audit_clean"] is True
    assert man["audit"]["wme-p2"]["test"]["test"] == 0


def test_groups_never_straddle_splits(tmp_path):
    twin = {
        "question": "QUESTION number 7 about a topic that is unique to item 7 with words",
        "hash": "htwin",
        "id": "sya:twin",
    }
    man = _build(
        tmp_path,
        pool=_pool(extra=[twin]),
        sizes={"syc": {"construction": 10, "validation": 5, "dev": 5, "test": 5}},
    )
    where = {h: s for s in C1.SPLITS for h in man["objectives"]["syc"][s]["hashes"]}
    # one item per near-duplicate group: the twin never enters any split beside its original
    assert "htwin" not in where
    assert man["exclusion"]["syc"]["duplicates_dropped"] == 1


def test_infeasible_sizes_raise_rather_than_relax(tmp_path):
    with pytest.raises(ValueError):
        _build(tmp_path, sizes={"syc": {"construction": 20, "validation": 5, "dev": 5, "test": 5}})


def test_existing_panel_is_never_overwritten(tmp_path):
    _build(tmp_path)
    with pytest.raises(FileExistsError):
        _build(tmp_path)


def test_planner_store_cannot_read_c1_test(tmp_path):
    _build(tmp_path)
    store = PlannerItemStore(tmp_path / "wme-c1")
    assert store.read("dev")
    with pytest.raises(SealedError):
        store.read("test")
    with pytest.raises(SealedError):
        store.read_path(tmp_path / "wme-c1" / "sealed_test" / "test.jsonl")


def test_sealed_reader_requires_frozen_winners_checks_hash_and_reads_once(tmp_path):
    man = _build(tmp_path)
    root = tmp_path / "wme-c1"
    reader = C1.SealedTestReader(root, receipt_dir=tmp_path / "receipts")
    with pytest.raises(SealedError):
        reader.read({"wm": "abc", "statnn": None})
    winners = {"wm": "a" * 64, "statnn": "b" * 64}
    rows = reader.read(winners)
    assert len(rows) == 2 and all("hash" in r for r in rows)
    with pytest.raises(SealedError):
        C1.SealedTestReader(root, receipt_dir=tmp_path / "receipts").read(winners)  # one read only
    receipt = json.loads((tmp_path / "receipts" / "sealed_read.json").read_text())
    assert receipt["winners_digest"] and receipt["sha256"] == man["files"]["sealed_test/test.jsonl"]


def test_sealed_reader_detects_tampering_and_symlinks(tmp_path):
    _build(tmp_path)
    root = tmp_path / "wme-c1"
    p = root / "sealed_test" / "test.jsonl"
    p.write_text(p.read_text() + "{}\n")
    with pytest.raises(SealedError):
        C1.SealedTestReader(root, receipt_dir=tmp_path / "r1").read({"wm": "a" * 64})
    p.unlink()
    (tmp_path / "elsewhere.jsonl").write_text("{}\n")
    p.symlink_to(tmp_path / "elsewhere.jsonl")
    with pytest.raises((SealedError, OSError)):
        C1.SealedTestReader(root, receipt_dir=tmp_path / "r2").read({"wm": "a" * 64})
