"""Closed pre-G2 snapshot helpers (checklist item 3): closure reconciliation and ledger rows."""

from scripts import a5_snapshot as sn


def _panel():
    return {
        "components": [
            {"component_id": "A", "status": "development", "members": ["a1", "a2"]},
            {"component_id": "B", "status": "development", "members": ["b1"]},
            {"component_id": "C", "status": "confirmatory", "members": ["c1", "c2"]},
            {"component_id": "D", "status": "development", "members": ["d1"]},
        ]
    }


def test_reconcile_detects_merges_splits_and_new_entries():
    rows = {
        "a1": {"family": "lora_sft"},
        "a2": {"family": "chain", "parent_id": "a1"},
        "b1": {"family": "lora_sft"},
        "c1": {"family": "chain"},
        "c2": {"family": "chain"},  # registered with c1 but no current join -> split
        "d1": {"family": "chain"},
        "h1": {"family": "heretic_ara"},
    }
    edits = {"h1": {"provenance": {"generator_edit_id": "d1"}}, "d1": {"provenance": {}}}
    hashes = {"a1": "x", "a2": "y", "b1": "z", "c1": "u", "c2": "v", "d1": "z", "h1": "w"}
    out = sn.reconcile_closure(rows, edits, hashes, _panel(), ["a2", "b1"], ["c1", "a2"])
    # b1 and d1 now share stored bytes: a merge of two registered components, one of them selected
    assert out["registered_components_merged"] == [["B", "D"]]
    assert out["selected_components_merged_with_any_other_registered_component"] == ["B"]
    assert out["registered_components_split"] == ["C"]
    assert out["selected_components_distinct_in_current_closure"] is True
    # generator joins: h1 (unregistered) joins d1's class, which holds B and D
    assert out["entries_without_registered_component"]["h1"] == {
        "family": "heretic_ara",
        "joined_registered_components": ["B", "D"],
        "joins_a_selected_component": True,
    }
    assert out["census_rows_by_component_status"] == {"confirmatory": 1, "development": 1}
    assert out["old_seal_components_scored_by_census"]["n_components"] == 1
    assert out["edge_counts"] == {"delta_hash": 1, "generator_entry": 1, "parent": 1}


def test_selected_components_merging_is_detected():
    rows = {"a1": {}, "b1": {"parent_id": "a1"}}
    panel = {
        "components": [
            {"component_id": "A", "status": "development", "members": ["a1"]},
            {"component_id": "B", "status": "development", "members": ["b1"]},
        ]
    }
    out = sn.reconcile_closure(rows, {}, {}, panel, ["a1", "b1"], [])
    assert out["selected_components_distinct_in_current_closure"] is False


def test_run_time_and_ledger_row():
    assert sn.run_time("20261107-105825-56ef1597") == "2026-11-07T10:58:25Z"
    assert sn.run_time(None) is None
    row = sn.ledger_row(
        "e",
        gate="G6",
        role="census",
        activity="x",
        time="t",
        inv={"e": {"parent_id": "p"}},
        comp={"e": ("C", "development")},
        canonical={},
        aliases={"e": ["e2"]},
        hashes={"h": 1},
    )
    assert row["predecessor"] == "p" and row["strength"] == 1.0 and row["fold"] is None
    assert row["component_id"] == "C" and row["aliases"] == ["e2"] and row["canonical_id"] == "e"


def test_evidence_block_lists_final_hashes_without_a_cycle(tmp_path):
    out, runs, reports = tmp_path / "m", tmp_path / "runs", tmp_path / "rep"
    for d in (out, runs, reports):
        d.mkdir()
    for n in sn.EVIDENCE_MANIFESTS:
        (out / n).write_text(n)
    (runs / "inputs.json").write_text("{}")
    (runs / "materialise.log").write_text("log")
    (reports / "report.json").write_text("{}")
    block = sn.evidence_block(out, runs, reports)
    assert set(block["manifests"]) == set(sn.EVIDENCE_MANIFESTS)
    assert "selection.json" not in block["manifests"] and "README.md" not in block["manifests"]
    assert list(block["nullsize_runs"]) == ["data/runs/a5/nullsize/inputs.json"]
    assert block["nullsize_reports"] == {
        "docs/prereg/a5_manifests/nullsize/report.json": sn.sha_file(reports / "report.json")
    }
