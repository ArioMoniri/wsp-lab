"""Pure-CPU tests for the TF-v1 pre-panel scripts (G2 bridge comparison, qualification composition)."""

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair(scores_fast, scores_tf, gold, item="1", gold_lp=None):
    from wsp.eval.tf_v1 import choice_record

    argmax = lambda s: max(range(len(s)), key=s.__getitem__)
    fast = {
        "item_id": item,
        "correct": float(argmax(scores_fast) == gold),
        "extra": {"scores": scores_fast, "gold_index": gold},
    }
    record = choice_record("e", "mmlu_pro", item, scores_tf, gold)
    return fast, {
        "item_id": item,
        "correct": record.correct,
        "extra": record.extra,
    }


def test_g2_identical_scores_agree_exactly():
    bridge = load("tf_v1_g2_bridge")
    fast, tf = pair([-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0], 0)
    result = bridge.compare([fast], [tf], "mmlu_pro")
    assert result["correctness_agreement"] == 1.0 and result["argmax_agreement"] == 1.0
    assert result["bitwise_equal_scores"] and result["gold_score_abs_diff_max"] == 0.0
    assert result["accuracy_abs_diff"] == 0.0


def test_g2_flip_and_logprob_difference_are_hand_computed():
    bridge = load("tf_v1_g2_bridge")
    # Fast tier picks index 1 (wrong), TF-v1 picks index 0 (gold): one correctness flip.
    fast, tf = pair([-2.0, -1.0], [-1.0, -2.0], 0)
    result = bridge.compare([fast], [tf], "mmlu_pro")
    assert result["n_correctness_flips"] == 1 and result["argmax_agreement"] == 0.0
    assert result["accuracy_abs_diff"] == 1.0
    # normalized gold log-prob: log_softmax([-2,-1])[0] vs log_softmax([-1,-2])[0]
    expected = abs(-math.log1p(math.e) - (-math.log1p(math.exp(-1))))
    assert result["gold_score_abs_diff_max"] == pytest.approx(expected)


def test_g2_item_mismatch_is_an_error_not_a_drop():
    bridge = load("tf_v1_g2_bridge")
    a, b = pair([-1.0, -2.0], [-1.0, -2.0], 0, item="1")
    b["item_id"] = "2"
    with pytest.raises(ValueError, match="item ids differ"):
        bridge.compare([a], [b], "mmlu_pro")


def test_g2_pooled_criteria_use_worst_entry():
    bridge = load("tf_v1_g2_bridge")
    fast, tf = pair([-1.0, -2.0], [-1.0, -2.0], 0)
    entry = {name: bridge.compare([fast], [tf], "mmlu_pro") for name in ("mmlu_pro", "hellaswag")}
    pooled = bridge.pooled({"a": entry, "b": entry})
    assert pooled["prereg_mmlu_criteria"]["pass"] is True
    entry["mmlu_pro"]["gold_score_abs_diff_median"] = 0.5
    assert bridge.pooled({"a": entry})["prereg_mmlu_criteria"]["pass"] is False


def test_composition_rule_is_deterministic_and_has_ten_entries():
    panel = load("tf_v1_qualification_panel")
    rows, edits, kl = {}, {}, {}
    for k in range(9):
        i = f"l{k}"
        rows[i] = {"id": i, "family": "lora_sft", "parent_id": None}
        kl[i] = k / 100
    for k in range(3):
        for family, key in (("task_vector", "t"), ("abliteration", "a")):
            i = f"{key}{k}"
            rows[i] = {"id": i, "family": family, "parent_id": None}
            kl[i] = k / 10
    for lineage in range(4):
        root = f"r{lineage}"
        rows[root] = {"id": root, "family": "chain", "parent_id": None}
        edits[root] = {"params": {"step": 0}}
        parent = root
        for step in range(1, 5):
            i = f"c{lineage}s{step}"
            rows[i] = {"id": i, "family": "chain", "parent_id": parent}
            edits[i] = {"params": {"step": step}}
            parent = i
    for i in rows:
        if rows[i]["family"] == "chain":
            kl[i] = 0.0
    chosen, median = panel.select(rows, edits, kl)
    assert median == 0.04
    assert chosen["lora_nearest_median"] == ["l4", "l3"] and chosen["lora_largest"] == ["l8", "l7"]
    assert chosen["task_vector"] == ["t0", "t1"] and chosen["abliteration_largest"] == ["a2"]
    assert chosen["chain"] == ["c0s3", "c1s3", "c2s3"]  # distinct lineages, step >= 3, sorted id
    assert chosen == panel.select(rows, edits, kl)[0]


def synthetic_zoo():
    rows, edits = {}, {}

    def add(i, family, parent=None, **extra):
        rows[i] = {"id": i, "family": family, "parent_id": parent}
        edits[i] = {
            "id": i,
            "family": family,
            "parent_id": parent,
            "eligibility": "production",
            "params": extra.pop("params", {}),
            "provenance": extra,
        }

    add("L1", "lora_sft", generator_edit_id="g1")
    add("L2", "lora_sft", generator_edit_id="g2")  # unevaluated candidate, fresh
    add("L3", "lora_sft", generator_edit_id="g3")  # shares nothing, but evaluated -> development
    add("T1", "task_vector", parent="A1", parent_reuse_key="k", reuse_key="t")
    add("A1", "abliteration", reuse_key="k")
    add("T2", "task_vector", parent="L3", parent_reuse_key="k", reuse_key="t")
    # chain lineage sharing a trained adapter with L1 (generator join) -> one component
    add("C0", "chain", generator_edit_id="g1", params={"step": 0, "length": 2})
    add("C1", "chain", parent="C0", params={"step": 1, "length": 2})
    add("D0", "chain", params={"step": 0, "length": 2})  # exact copy of A1's delta (hash join)
    add("D1", "chain", parent="D0", params={"step": 1, "length": 2})
    add("F0", "lora_sft", generator_edit_id="gf")  # flagged ARA id -> excluded
    add("E0", "chain", params={"step": 0, "length": 1}, generator_edit_id="ge")
    edits["E0"]["eligibility"] = "smoke"
    return rows, edits


def test_partition_joins_closes_and_seals(monkeypatch):
    build = load("build_h1a_panel")
    rows, edits = synthetic_zoo()
    monkeypatch.setattr(build, "artefact_files", lambda path: [path])
    monkeypatch.setattr(build, "FLAGGED_ARA", ["F0"])
    hashes = {i: f"h-{i}" for i in rows} | {"A1": "same", "D0": "same"}
    evaluated, pilots = {"L3", "A1"}, set()
    components, aliases, edge_kinds, candidates, first = build.build(
        rows, edits, hashes, evaluated, pilots
    )
    where = {m: c for c in components for m in c["members"]}
    # parent, generator and delta-hash joins; a task vector joins its source parent
    assert where["T1"] is where["A1"] is where["D0"] is where["D1"]
    assert where["C0"] is where["C1"] is where["L1"]
    assert where["T2"] is where["L3"] and where["T2"]["status"] == "development"
    # an evaluated A1 puts T1 and the duplicate chain D0/D1 in development: no fresh leak
    assert where["T1"]["status"] == "development" and where["D1"]["status"] == "development"
    assert (
        where["F0"]["status"] == "excluded"
        and "flagged_ara_diagnostic" in where["F0"]["exclusion_reasons"]
    )
    assert where["E0"]["status"] == "excluded"
    # fresh candidates: L1 (with the generator-sharing chain) and L2 are confirmatory
    assert where["L1"]["status"] == "confirmatory" and where["L2"]["status"] == "confirmatory"
    # T2 is an unevaluated candidate whose source L3 is evaluated: candidate, but development
    assert candidates == {"L1", "L2", "T1", "T2", "F0"}
    assert aliases == {"same": ["A1", "D0"]}
    assert edge_kinds["delta_hash"] == 1 and edge_kinds["generator"] == 1
    summary = build.summarize(
        components, rows, candidates, first, aliases, edge_kinds, evaluated, hashes
    )
    assert summary["confirmatory_components"] == 2 and not summary["met_target"]
    ids = [c["component_id"] for c in components]
    assert ids == sorted(
        ids, key=lambda i: next((c["rank"], i) for c in components if c["component_id"] == i)
    )
    # component ids and ranks depend only on membership, never on dict order
    again = build.build(dict(reversed(list(rows.items()))), edits, hashes, evaluated, pilots)[0]
    assert {c["component_id"] for c in again} == set(ids)
