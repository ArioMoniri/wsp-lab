"""F11 supplementary procedural selection (scripts/a5_f11_supplement.py) on a synthetic pool (CPU, no data/)."""

import hashlib
import json
import random

import pytest

from scripts import a5_f11_supplement as sup

# Synthetic pool. Per cell: the registered quota of picks (sorted canonical-ID prefix), then unselected
# candidates whose eligibility for the F11 rule varies. Depleted cells: lora_sft:2, chain:0, chain:2.
LORA2 = {
    "picks": ["a00", "a01", "a02"],
    "rest": [
        ("a10", {}),  # root, exact, in pool -> pick 1
        (
            "a11",
            {"pool": False, "fm": False},
        ),  # not a NULL-SIZE donor: no exact-gradient record -> skip
        ("a12", {}),  # pick 2
        ("a13", {}),  # pick 3
        ("a14", {}),  # pick 4
        ("a15", {}),  # reserve (never reached)
    ],
}
CHAIN0 = {
    "picks": ["c00", "c01"],
    "rest": [
        ("c10", {"parent": "p1"}),  # non-root chain step -> skip
        ("c11", {}),  # pick 1
        ("c12", {"alias": "zz9"}),  # alias-only bytes -> skip
        ("c13", {}),  # pick 2
        ("c14", {}),  # pick 3
        ("c15", {"parent": "p2"}),  # non-root -> skip
        ("c16", {}),  # pick 4
        ("c17", {}),  # reserve
    ],
}
CHAIN2 = {
    "picks": ["e00", "e01"],
    "rest": [("e10", {}), ("e11", {}), ("e12", {}), ("e13", {}), ("e14", {})],
}
LORA1 = {"picks": ["b00", "b01", "b02"], "rest": [("b10", {}), ("b11", {})]}
LORA0 = {"picks": ["l00", "l01", "l02"], "rest": [("l10", {})]}
CHAIN1 = {"picks": ["d00", "d01"], "rest": [("d10", {})]}

CELLS = {
    "lora_sft": [LORA0, LORA1, LORA2],
    "chain": [CHAIN0, CHAIN1, CHAIN2],
}
EXCLUDED_ROTATIONS = ["a01", "c00", "c01", "e00", "e01"]
EXPECTED = {
    "lora_sft:2": ["a10", "a12", "a13", "a14"],
    "chain:0": ["c11", "c13", "c14", "c16"],
    "chain:2": ["e10", "e11", "e12", "e13"],
}
TEST_CELLS = (("lora_sft", 2), ("chain", 0), ("chain", 2))


def _write_pool(tmp_path, shuffle_seed=None, overrides=None, excluded=EXCLUDED_ROTATIONS):
    overrides = overrides or {}
    rows, donors, pool, selected, variants = [], {}, [], [], []
    ledger = {}
    for fam, cells in CELLS.items():
        fam_cells, quotas = [], []
        for t, cell in enumerate(cells):
            members = list(cell["picks"]) + [i for i, _ in cell["rest"]]
            if shuffle_seed is not None:
                random.Random(shuffle_seed + t).shuffle(members)
            fam_cells.append(members)
            quotas.append(len(cell["picks"]))
            flags = {i: dict(f) for i, f in cell["rest"]}
            flags.update({i: {} for i in cell["picks"]})
            for i in members:
                f = {**flags[i], **overrides.get(i, {})}
                parent = f.get("parent")
                alias = f.get("alias")
                rows.append(
                    {
                        "id": i,
                        "canonical_id": i,
                        "component_id": "comp-" + i,
                        "family": fam,
                        "parent_id": parent,
                        "bytes_present": alias is None,
                        "bytes_via_alias": alias,
                        "eligibility": f.get("eligibility", "eligible"),
                        "closure_certified": True,
                        "component_status": "development",
                        "exactness": "bitwise",
                        "effective_norm": 1.0,
                    }
                )
                if f.get("fm", True):
                    flagged = parent is not None
                    donors[i] = {
                        "canonical_id": i,
                        "family": fam,
                        "flags": ["base-approximation-unvalidated"] if flagged else [],
                        "gradient_source": f.get("source", "base_field"),
                        "predecessor_is_base": parent is None,
                        "donor_via_alias": alias is not None,
                        "donor_bytes": alias or i,
                    }
                    if not flagged and f.get("pool", True):
                        pool.append(i)
            for i in cell["picks"]:
                selected.append({"edit_id": i, "family": fam, "tertile": t})
                variants.append({"kind": "rotation", "source_id": i, "excluded": i in excluded})
                for s in (0.25, 0.5):
                    variants.append(
                        {"kind": "scale", "source_id": i, "strength": s, "excluded": False}
                    )
        ledger[fam] = {"cells": fam_cells, "quotas": quotas}
    reg_ids = sorted(s["edit_id"] for s in selected)
    selection = {
        "registered": {"ledger": {"families": ledger}, "edit_ids": reg_ids},
        "procedural_freeze": {"registered_edit_ids": reg_ids},
    }
    paths = {
        "selection": tmp_path / "selection.json",
        "components": tmp_path / "components.jsonl",
        "fields_manifest": tmp_path / "fields_manifest.json",
        "registered_donors": tmp_path / "registered_donors.json",
        "part_a_manifest": tmp_path / "manifest.json",
    }
    paths["selection"].write_text(json.dumps(selection))
    paths["components"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    paths["fields_manifest"].write_text(json.dumps({"donors": donors}))
    paths["registered_donors"].write_text(json.dumps({"donor_ids": sorted(pool)}))
    paths["part_a_manifest"].write_text(json.dumps({"selections": selected, "variants": variants}))
    return paths


def _build(paths, **kw):
    kw.setdefault("expected_cells", TEST_CELLS)
    return sup.build(**{k: v for k, v in paths.items()}, **kw)


def test_picks_first_four_eligible_in_canonical_order(tmp_path):
    out = _build(_write_pool(tmp_path))
    got = {c["cell"]: c["selected"] for c in out["cells"]}
    assert got == EXPECTED
    assert out["k"] == 12 and out["per_cell"] == 4
    assert len(out["supplement_ids"]) == 12


def test_skip_reasons_are_recorded(tmp_path):
    out = _build(_write_pool(tmp_path))
    by_cell = {c["cell"]: c for c in out["cells"]}
    reasons = {r["id"]: r.get("reasons") for r in by_cell["chain:0"]["candidate_order"]}
    assert "not_root" in reasons["c10"]
    assert "alias_only" in reasons["c12"]
    assert reasons["c11"] == []
    lora = {r["id"]: r.get("reasons") for r in by_cell["lora_sft:2"]["candidate_order"]}
    assert "not_exact_gradient" in lora["a11"]
    # Candidates after the fourth pick are listed as unreached reserve, not evaluated for selection.
    tail = by_cell["chain:0"]["candidate_order"][-1]
    assert tail["id"] == "c17" and tail["status"] == "reserve_unreached"


def test_order_is_canonical_id_not_ledger_order(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _build(_write_pool(tmp_path / "a"))
    b = _build(_write_pool(tmp_path / "b", shuffle_seed=7))
    assert a["supplement_ids"] == b["supplement_ids"]
    for ca, cb in zip(a["cells"], b["cells"]):
        assert [r["id"] for r in ca["candidate_order"]] == [r["id"] for r in cb["candidate_order"]]


def test_registered_picks_must_be_the_sorted_prefix(tmp_path):
    # A registered pick that is not in the cell's sorted quota prefix breaks the frozen order: stop.
    paths = _write_pool(tmp_path)
    sel = json.loads(paths["selection"].read_text())
    sel["registered"]["ledger"]["families"]["chain"]["quotas"][0] = 3
    paths["selection"].write_text(json.dumps(sel))
    with pytest.raises(sup.SupplementError, match="registered picks"):
        _build(paths)


def test_shortage_stops_without_cross_cell_replacement(tmp_path):
    over = {i: {"parent": "px"} for i in ("e10", "e11")}
    paths = _write_pool(tmp_path, overrides=over)
    with pytest.raises(sup.SupplementError, match="chain:2"):
        _build(paths)


def test_depleted_cells_come_from_rotation_exclusions_only(tmp_path):
    paths = _write_pool(tmp_path)
    m = json.loads(paths["part_a_manifest"].read_text())
    # A ladder exclusion does not deplete a cell.
    for v in m["variants"]:
        if v["source_id"] == "b00" and v["kind"] == "scale":
            v["excluded"] = True
    paths["part_a_manifest"].write_text(json.dumps(m))
    out = _build(paths)
    assert [c["cell"] for c in out["cells"]] == ["lora_sft:2", "chain:0", "chain:2"]
    assert out["registered_rotation_exclusions"]["n"] == 5


def test_depleted_cells_must_match_the_frozen_set(tmp_path):
    paths = _write_pool(tmp_path, excluded=["a01", "c00", "d00"])
    with pytest.raises(sup.SupplementError, match="depleted cells"):
        _build(paths)


def test_inconsistent_root_flags_stop(tmp_path):
    paths = _write_pool(tmp_path)
    fm = json.loads(paths["fields_manifest"].read_text())
    fm["donors"]["c11"]["predecessor_is_base"] = False
    paths["fields_manifest"].write_text(json.dumps(fm))
    with pytest.raises(sup.SupplementError, match="c11"):
        _build(paths)


def test_ineligible_or_registered_rows_are_skipped(tmp_path):
    over = {"a10": {"eligibility": "excluded:foo"}}
    out = _build(_write_pool(tmp_path, overrides=over))
    got = {c["cell"]: c["selected"] for c in out["cells"]}
    assert got["lora_sft:2"] == ["a12", "a13", "a14", "a15"]
    reasons = {r["id"]: r.get("reasons") for r in out["cells"][0]["candidate_order"]}
    assert "not_eligible" in reasons["a10"]


def test_hash_rule_and_input_hashes(tmp_path):
    paths = _write_pool(tmp_path)
    out = _build(paths)
    ids = sorted(i for v in EXPECTED.values() for i in v)
    assert out["supplement_ids_sorted"] == ids
    assert (
        out["supplement_ids_sorted_sha256"] == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    )
    for name, p in paths.items():
        assert out["inputs"][name]["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    # Disjoint from the forty and component-distinct.
    assert not set(ids) & set(out["registered_edit_ids_sorted"])
    comps = [r["component_id"] for r in out["supplements"]]
    assert len(set(comps)) == 12


def test_cli_writes_deterministic_bytes(tmp_path):
    paths = _write_pool(tmp_path)
    args = [f"--{k.replace('_', '-')}={v}" for k, v in paths.items()]
    o1, o2 = tmp_path / "o1.json", tmp_path / "o2.json"
    cells = ["--expected-cells", "lora_sft:2,chain:0,chain:2"]
    assert sup.main(args + cells + ["--out", str(o1)]) == 0
    assert sup.main(args + cells + ["--out", str(o2)]) == 0
    assert o1.read_bytes() == o2.read_bytes()
    assert o1.read_bytes().endswith(b"\n")
    d = json.loads(o1.read_text())
    assert d["schema"] == sup.SCHEMA
    assert {c["cell"]: c["selected"] for c in d["cells"]} == EXPECTED
