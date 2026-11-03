import json

import pytest

from wsp.schema import EditDescriptor, EvalRecord, aggregate


def test_edit_id_is_deterministic_and_roundtrips():
    e = EditDescriptor(
        family="random_dir",
        params={"layer": 3, "weight": 1.0},
        seed=7,
        base_model="Qwen/Qwen3.5-0.8B",
    )
    e2 = EditDescriptor.from_json(e.to_json())
    assert e.id == e2.id and len(e.id) == 16
    assert json.loads(e.to_json())["id"] == e.id


def test_unknown_family_rejected():
    with pytest.raises(ValueError):
        EditDescriptor(family="magic", params={}, seed=0, base_model="x")


def test_aggregate_from_items_and_loop_rate():
    recs = [
        EvalRecord("e", "gsm8k", str(i), float(i % 2), hit_max_tokens=(i == 0)) for i in range(4)
    ]
    recs += [EvalRecord("e", "hellaswag", "a", 1.0)]
    a = aggregate(recs)
    assert a["gsm8k"] == 0.5 and a["hellaswag"] == 1.0 and a["loop_rate"] == 0.25
