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
