"""F12 population selector, strata, folds and purchase order (scripts/f12_population.py).

Synthetic component rows for the rules; one check reproduces the registered counts from the committed manifests
under docs/prereg/a5_manifests (no data/ file is opened).
"""

import hashlib
import json

import pytest

from scripts import f12_population as fp


def _row(
    i,
    comp,
    family="chain",
    parent=None,
    status="development",
    present=True,
    alias=None,
    elig="eligible",
):
    return {
        "id": i,
        "component_id": comp,
        "family": family,
        "parent_id": parent,
        "component_status": status,
        "bytes_present": present,
        "bytes_via_alias": alias,
        "eligibility": elig,
        "exactness": "bitwise",
    }


def _toy():
    rows = [
        # existing (primary) roots
        _row("e1", "cA", "chain"),
        _row("e2", "cA", "chain", present=False, alias="e1"),
        _row("e3", "cB", "chain", status="confirmatory"),
        _row("e4", "cM", "chain"),
        _row("e5", "cM", "random_dir"),
        _row("e6", "cB", "chain", parent="e3", status="confirmatory"),  # nonroot: never a row
        # purchasable roots
        _row("p1", "cP1", "abliteration"),
        _row("p2", "cP2", "lora_sft"),
        _row("p3", "cP3", "random_dir"),
        _row("p4", "cP4", "lora_sft", present=False),  # no bytes
        _row("p5", "cP5", "abliteration", status="confirmatory"),  # not development
        _row("p6", "cA", "chain"),  # component holds a primary record
        _row("p7", "cPA", "lora_sft"),  # Part A component
        _row("x1", "cPA", "lora_sft"),
    ]
    primary = {"e1", "e2", "e3", "e4", "e5", "e6", "x1", "base"}
    return rows, primary


def test_select_rules():
    rows, primary = _toy()
    ex, pu = fp.select(rows, primary, {"cPA"})
    assert [r["id"] for r in ex] == ["e1", "e2", "e3", "e4", "e5"]
    assert [r["id"] for r in pu] == ["p1", "p2", "p3"]
    _, pu_nf = fp.select(rows, primary, {"cPA"}, component_filter=False)
    assert [r["id"] for r in pu_nf] == ["p1", "p2", "p3", "p6"]


def test_strata_mixed_and_unknown():
    rows, primary = _toy()
    ex, pu = fp.select(rows, primary, {"cPA"})
    st = fp.strata(ex + pu)
    assert st["cM"] == "mixed" and st["cA"] == "chain" and st["cP2"] == "lora_sft"
    with pytest.raises(ValueError, match="registered order"):
        fp.strata([_row("z", "cZ", "task_vector")])


def test_fold_assignment_rule():
    st = {
        f"c{i}": s
        for i, s in enumerate(
            ["random_dir", "chain", "abliteration", "mixed", "chain", "lora_sft", "chain"]
        )
    }
    fold_of, listing = fp.assign_folds(st)
    # strata concatenated in the registered order, hash-sorted within stratum, round-robin by position
    want = []
    for s in fp.STRATUM_ORDER:
        cs = [c for c in st if st[c] == s]
        want += sorted(cs, key=lambda c: hashlib.sha256(f"20261111:f12:{c}".encode()).hexdigest())
    assert [x["component_id"] for x in listing] == want
    assert all(fold_of[c] == pos % 5 for pos, c in enumerate(want))
    assert [x["stratum"] for x in listing] == [st[c] for c in want]
    # order-independent of dict insertion
    fold2, _ = fp.assign_folds(dict(reversed(list(st.items()))))
    assert fold2 == fold_of
    assert (
        fp.fold_manifest_sha256(fold_of)
        == hashlib.sha256(
            json.dumps(
                dict(sorted(fold_of.items())), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )


def test_alias_resolution_and_shared_pairs():
    rows, _primary = _toy()
    by = {r["id"]: r for r in rows}
    edh = {"e1": "h1", "e3": "h3", "e4": "hM", "e5": "hM", "p1": "hp1"}
    res = fp.resolve_delta_hashes([by["e1"], by["e2"], by["e4"], by["e5"]], by, edh)
    assert res["e2"] == ("e1", "h1")  # null own hash resolves through its alias
    assert fp.shared_deltas(res) == [["e1", "e2"], ["e4", "e5"]]
    with pytest.raises(ValueError, match="no stored or aliased bytes"):
        fp.resolve_delta_hashes([_row("q", "cQ", present=False)], by, edh)
    with pytest.raises(ValueError, match="another component"):
        fp.resolve_delta_hashes([_row("q", "cQ", present=False, alias="e1")], by, edh)
    with pytest.raises(ValueError, match="no delta hash"):
        fp.resolve_delta_hashes([by["p2"]], by, edh)


def test_ids_sha256_rule():
    assert fp.ids_sha256(["b", "a"]) == hashlib.sha256(b'["a", "b"]').hexdigest()


def test_subsets_recompute_folds():
    rows, primary = _toy()
    ex, pu = fp.select(rows, primary, {"cPA"})
    order, s_k = fp.subsets(ex, pu, k_grid=(0, 1, 3))
    keys = [o["order_key"] for o in order]
    assert keys == sorted(keys)
    assert [s["purchased_rows"] for s in s_k] == [0, 1, 3]
    assert s_k[0]["rows"] == len(ex)
    first = order[0]["component_id"]
    assert s_k[1]["purchased_components"] == [first]
    for s in s_k:
        st = fp.strata(ex + [r for r in pu if r["component_id"] in s["purchased_components"]])
        assert s["folds"] == dict(sorted(fp.assign_folds(st)[0].items()))
    with pytest.raises(ValueError, match="exceeds"):
        fp.subsets(ex, pu, k_grid=(4,))


def test_exposure_by_own_id_and_alias():
    snap = {
        "exposure_ledger": {
            "rows": [
                {"gate": "G0", "id": "a", "aliases": ["b"]},
                {"gate": "G6", "id": "a", "aliases": []},
                {"gate": "G9", "id": "c", "aliases": []},
            ]
        }
    }
    own, alias = fp.exposure_ids(snap)
    assert own == {"a": ["G0", "G6"]} and alias == {"b": ["G0:a"]}


def test_reproduces_registered_population():
    components, snapshot, selection, selection_f11, inputs = fp.load_inputs(fp.MANIFESTS)
    doc = fp.build(components, snapshot, selection, selection_f11, inputs)
    assert doc["reproduction"]["discrepancies"] == []
    assert doc["population_ids_sha256"] == fp.EXPECTED["population"]["sha256"]
    assert len(doc["rows"]) == 270
    assert {
        r["stratum"] for r in doc["rows"] if r["id"] in ("4b67b0aa12cce3e5", "db5c3a7bfc19178c")
    } == {"mixed"}
    assert sum(r["g6_exposed_confirmatory"] for r in doc["rows"]) == 44
    assert [s["rows"] for s in doc["s_k"]] == [123, 152, 182, 211, 241, 270]
    assert doc["s_k"][-1]["folds"] == doc["folds"]["component_fold"]
