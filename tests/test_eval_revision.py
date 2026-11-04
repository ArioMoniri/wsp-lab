import json

import numpy as np
import pytest

from wsp.eval.repeats import compare_pairs, matched_reference

KEYS = [
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
]


def known(**kwargs):
    return dict.fromkeys(KEYS, "known") | kwargs


def test_unknown_reference_values_are_not_matched():
    row = known(item_id_hash=None)
    assert matched_reference(row, [row]) is None


@pytest.mark.parametrize("key", ["edit_id", "item_id_hash", "item_order_hash"])
def test_reliability_rejects_unknown_identity(key):
    row = known(edit_id="x", math500=0, counts={"math500": 25})
    row.pop(key)
    with pytest.raises(ValueError):
        compare_pairs([row], [row], bootstrap=20)


def test_reliability_statistics_continuous_and_item_discordance(tmp_path):
    panels = []
    for side in range(2):
        panel = []
        for i, score in enumerate(([0, 0.5, 1], [1, 0.5, 0])[side]):
            dest = tmp_path / str(side) / str(i)
            dest.mkdir(parents=True)
            (dest / "records.jsonl").write_text(
                json.dumps({"benchmark": "math500", "item_id": "a", "correct": side}) + "\n"
            )
            panel.append(
                known(
                    edit_id=str(i),
                    math500=score,
                    kl_to_base=0.1 + score,
                    wikitext_ppl=np.exp(1 + score),
                    counts={"math500": 25, "kl_to_base": 2, "wikitext_ppl": 2},
                    path=str(dest),
                )
            )
        panels.append(panel)
    result = compare_pairs(*panels, bootstrap=100)
    r = result["math500"]
    assert r["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert r["median_absolute_difference"] == 1
    assert r["max_absolute_difference"] == 1
    assert r["sd_difference"] == 1
    assert r["bland_altman_limits"] == pytest.approx([-1.96, 1.96])
    assert r["item_discordance"]["n01"] == 3
    assert r["item_discordance"]["n10"] == 0
    assert r["binomial_se"][0][0] > 0
    assert len(r["spearman_ci95"]) == len(r["icc_2_1_ci95"]) == 2
    assert result["kl_to_base"]["binomial_se"][0] == [None, None]
    assert result["wikitext_ppl"]["scale"] == "log-ppl"


def test_loader_matches_multiple_device_bases_and_predecessors(tmp_path):
    from wsp.predictor.data import read_targets

    for name, row in {
        "base": known(edit_id="base", metrics={"acc": 0.2}),
        "base-other": known(edit_id="base", eval_device_id="H1", metrics={"acc": 0.7}),
        "edit": known(edit_id="edit", eval_device_id="H1", metrics={"acc": 0.8}),
    }.items():
        dest = tmp_path / name
        dest.mkdir()
        (dest / "eval.json").write_text(json.dumps(row))
    raw = read_targets(tmp_path)
    assert raw.reference("edit", "base")["acc"] == 0.7
    assert raw.reference("edit", "edit")["acc"] == 0.8
    raw.records["edit"][-1]["eval_device_id"] = "H2"
    assert raw.reference("edit", "base") == {}
    assert raw.reference("base", "edit") == {}


def test_base_resume_preserves_legacy_and_no_base_reports_missing(tmp_path, capsys):
    from wsp.eval.batch import base_destination

    dest = tmp_path / "base"
    dest.mkdir()
    old = dest / "eval.json"
    old.write_text(json.dumps({"edit_id": "base"}))
    path, reference = base_destination(tmp_path, known(), "run", no_base=False)
    assert path == dest / "devices" / "known" / "run"
    assert reference is None and json.loads(old.read_text()) == {"edit_id": "base"}
    path, reference = base_destination(tmp_path, known(), "run", no_base=True)
    assert path is None and reference is None
    assert "missing matched reference" in capsys.readouterr().out


def test_reliability_matched_delta_panel_and_base_separation():
    left = [
        known(edit_id=str(i), math500=v, counts={"math500": 25})
        for i, v in enumerate((0.2, 0.4, 0.8))
    ]
    right = [r | {"math500": r["math500"] + 0.1} for r in left]
    base = known(edit_id="base", math500=0.1, counts={"math500": 25})
    result = compare_pairs(
        left + [base],
        right + [base | {"math500": 0.2}],
        bootstrap=30,
        primary_references=[base],
        repeat_references=[base | {"math500": 0.2}],
    )
    r = result["math500"]
    assert r["n"] == 3 and len(r["base"]) == 1
    assert r["matched_reference_deltas"]["rmse"] == pytest.approx(0, abs=1e-15)
    assert r["missing_matched_reference"] == 0
    right[0]["item_order_hash"] = "different"
    with pytest.raises(ValueError, match="item_order_hash"):
        compare_pairs(left, right, bootstrap=20)
