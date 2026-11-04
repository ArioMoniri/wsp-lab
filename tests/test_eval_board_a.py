import math

from wsp.schema import EvalRecord, aggregate


def test_component_rates_and_keyword_aliases():
    rows = [
        EvalRecord(
            "x", "math500", "1", 0, hit_max_tokens=True, extra={"invalid_final_answer": True}
        ),
        EvalRecord("x", "gsm8k", "2", 1),
        EvalRecord("x", "refusal_rate", "3", 1),
    ]
    result = aggregate(rows)
    assert result["loop_rate"] == 0.5
    assert result["loop_rate_math500"] == 1
    assert result["invalid_final_answer_rate_math500"] == 1
    assert result["refusal_keyword_rate"] == result["refusal_rate"]


def test_repeat_statistics_keep_negative_icc_and_constant_undefined():
    from wsp.eval.repeats import compare_pairs

    left = [{"math500": x, "counts": {"math500": 25}, "batch_size": 32} for x in (0.0, 0.5, 1.0)]
    right = [{"math500": x, "counts": {"math500": 25}, "batch_size": 16} for x in (1.0, 0.5, 0.0)]
    for panel in (left, right):
        for i, row in enumerate(panel):
            row.update(edit_id=str(i), item_id_hash="items", item_order_hash="order")
    result = compare_pairs(left, right, bootstrap=100)
    assert result["math500"]["spearman"] == -1
    assert result["math500"]["icc_2_1"] < 0
    assert result["math500"]["batch_size_mismatch"]
    assert math.isnan(compare_pairs(left[:1], right[:1])["math500"]["spearman"])


def test_metadata_hashes_order_and_private_device(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from wsp.eval.metadata import item_hashes, runtime_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    runner = SimpleNamespace(model=SimpleNamespace())
    assert runtime_metadata(runner)["eval_device_id"] == "H0"
    rows = [EvalRecord("x", "math500", str(i), 0) for i in range(2)]
    assert item_hashes(rows)["item_id_hash"] == item_hashes(rows[::-1])["item_id_hash"]
    assert item_hashes(rows)["item_order_hash"] != item_hashes(rows[::-1])["item_order_hash"]


def test_reference_never_falls_back_across_device_or_missing_metadata():
    from wsp.eval.repeats import matched_reference

    assert matched_reference({}, [{}]) is None
    row = dict.fromkeys(
        (
            "eval_device_id",
            "gpu_class",
            "engine",
            "torch_version",
            "cuda_version",
            "transformers_version",
            "dtype",
            "attention_implementation",
            "batch_size",
            "item_order_hash",
            "item_id_hash",
            "generation",
            "model_revision",
            "source_hash",
            "config_hash",
        ),
        "same",
    )
    base = row | {"run_id": "base-run"}
    assert matched_reference(row, [base]) == base
    assert matched_reference(row, [base | {"eval_device_id": "H9"}]) is None


def test_reporting_rename_preserves_production_resume_hash():
    import yaml

    from wsp.eval.batch import REPO, effective_config_hash

    cfg = yaml.safe_load((REPO / "configs/eval_fast.yaml").read_text())
    new_hash = effective_config_hash(cfg)
    for name, judges in [("refusal_rate", ["keyword", "llm"]), ("over_refusal", ["keyword"])]:
        cfg["benchmarks"][name].pop("metric")
        cfg["benchmarks"][name].pop("export_judge")
        cfg["benchmarks"][name]["judge"] = judges
    assert effective_config_hash(cfg) == new_hash
