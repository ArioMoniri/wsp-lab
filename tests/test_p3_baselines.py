import json
import logging

import numpy as np
import pytest
from p3_toy import build_toy_zoo, toy_base_state

from wsp.features.extract import run_extract
from wsp.predictor.baselines import (
    fit_eval,
    load_targets,
    main,
    metrics_with_ci,
    run_baselines,
)
from wsp.predictor.splits import make_splits


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("p3")
    base = toy_base_state()
    ids = build_toy_zoo(root / "zoo", base)
    run_extract(root / "zoo", root / "features", base_state=base, n_layers=4)
    rows = [json.loads(line) for line in (root / "zoo" / "manifest.jsonl").read_text().splitlines()]
    feats = {i: json.loads((root / "features" / i / "features.json").read_text()) for i in ids}
    # synthetic targets: a monotone function of the effective delta norm plus family offset
    rng = np.random.default_rng(0)
    for i in ids:
        fro = feats[i]["norms"]["fro_effective"]
        y = {"acc": 0.9 - 2.0 * fro + 0.001 * rng.normal(), "ppl": 10 + 30 * fro}
        d = root / "evals" / i
        d.mkdir(parents=True)
        (d / "eval.json").write_text(json.dumps({"metrics": y}))
    return root, ids, rows


def test_fit_eval_ridge_recovers_linear_targets_with_seeded_ci():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    Y = np.stack([X @ [1.0, -2.0, 0.5], X[:, 0] ** 2], axis=1) + 0.01 * rng.normal(size=(60, 2))
    folds = make_splits([{"id": str(i), "seed": i % 5} for i in range(60)], "seed", n_splits=5)
    r1 = fit_eval(X, Y, folds, model="ridge", target_names=["lin", "quad"], seed=3, n_boot=200)
    r2 = fit_eval(X, Y, folds, model="ridge", target_names=["lin", "quad"], seed=3, n_boot=200)
    assert r1 == r2
    lin = r1["targets"]["lin"]
    assert lin["spearman"]["value"] > 0.99 and lin["mae"]["value"] < 0.05
    lo, hi = lin["mae"]["ci95"]
    assert lo <= lin["mae"]["value"] <= hi
    assert r1["targets"]["quad"]["spearman"]["value"] < 0.5  # ridge cannot fit x^2
    assert r1["n"] == 60 and r1["targets"]["lin"]["n"] == 60
    r3 = fit_eval(X, Y, folds, model="ridge", target_names=["lin", "quad"], seed=4, n_boot=200)
    assert r3["targets"]["lin"]["mae"]["ci95"] != lin["mae"]["ci95"]


def test_fit_eval_masks_missing_target_values():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    Y = (X @ [1.0, 1.0])[:, None].copy()
    Y[::4] = np.nan
    folds = make_splits([{"id": str(i), "seed": i % 4} for i in range(40)], "seed", n_splits=4)
    r = fit_eval(X, Y, folds, model="ridge", n_boot=50)
    assert r["targets"]["y0"]["n"] == 30


@pytest.mark.parametrize("model", ["statnn", "mlp"])
def test_other_models_run_and_beat_chance_on_monotone_signal(model):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 6))
    Y = (3 * X[:, 0] + X[:, 1])[:, None] + 0.05 * rng.normal(size=(80, 1))
    folds = make_splits([{"id": str(i), "seed": i % 4} for i in range(80)], "seed", n_splits=4)
    r = fit_eval(X, Y, folds, model=model, n_boot=50, seed=0)
    assert r["targets"]["y0"]["spearman"]["value"] > 0.6


def test_fit_eval_rejects_unknown_model_and_bad_shapes():
    X, Y = np.zeros((10, 2)), np.zeros((10, 1))
    folds = make_splits([{"id": str(i), "seed": i % 2} for i in range(10)], "seed", n_splits=2)
    with pytest.raises(ValueError, match="model"):
        fit_eval(X, Y, folds, model="nope")
    with pytest.raises(ValueError, match="rows"):
        fit_eval(X, Y[:5], folds, model="ridge")


def test_metrics_with_ci_handles_constant_prediction():
    m = metrics_with_ci(np.arange(10.0), np.ones(10), seed=0, n_boot=50)
    assert np.isnan(m["spearman"]["value"]) and m["mae"]["value"] == pytest.approx(3.7)


def test_load_targets_skips_entries_without_eval_and_logs_counts(toy, caplog):
    root, ids, _rows = toy
    (root / "evals" / ids[0] / "eval.json").rename(root / "evals" / ids[0] / "x.json")
    try:
        with caplog.at_level(logging.INFO):
            targets, names = load_targets(root / "evals", ids)
    finally:
        (root / "evals" / ids[0] / "x.json").rename(root / "evals" / ids[0] / "eval.json")
    assert ids[0] not in targets and len(targets) == len(ids) - 1
    assert names == ["acc", "ppl"]
    assert f"{len(ids) - 1} of {len(ids)}" in caplog.text


def test_load_targets_empty_dir_returns_nothing(tmp_path):
    targets, names = load_targets(tmp_path / "none", ["a", "b"])
    assert targets == {} and names == []


@pytest.mark.parametrize("split", ["family", "seed", "trajectory"])
def test_run_baselines_on_toy_zoo_all_splits(toy, tmp_path, split):
    root, ids, _rows = toy
    res = run_baselines(
        root / "features", root / "evals", split, tmp_path / split, n_splits=3, n_boot=100
    )
    assert set(res["models"]) == {"delta_norm_ridge", "statnn", "mlp_svd"}
    assert res["n_edits"] == len(ids) and res["split"] == split
    for r in res["models"].values():
        assert set(r["targets"]) == {"acc", "ppl"}
    ridge = res["models"]["delta_norm_ridge"]["targets"]["acc"]
    if split != "family":  # held-out family changes the norm range; others interpolate
        assert ridge["spearman"]["value"] > 0.8
    assert (tmp_path / split / "metrics.json").exists()
    md = (tmp_path / split / "results.md").read_text()
    assert "| model |" in md and "delta_norm_ridge" in md and "acc" in md
    stored = json.loads((tmp_path / split / "metrics.json").read_text())
    assert stored["feature_dims"] == {"delta_norm": 2, "statnn": 1098, "mlp_svd": 120}
    assert (tmp_path / split / "manifest.json").exists()


def test_cli_writes_outputs_and_reports_no_targets(toy, tmp_path, capsys):
    root, ids, _rows = toy
    main(
        [
            "--features", str(root / "features"), "--evals", str(root / "evals"),
            "--split", "seed", "--out", str(tmp_path / "r"), "--n-splits", "3", "--n-boot", "50",
        ]
    )  # fmt: skip
    assert json.loads((tmp_path / "r" / "metrics.json").read_text())["n_edits"] == len(ids)
    with pytest.raises(SystemExit):
        main(
            [
                "--features", str(root / "features"), "--evals", str(tmp_path / "missing"),
                "--split", "seed", "--out", str(tmp_path / "r2"),
            ]
        )  # fmt: skip


def test_trajectory_folds_keep_chain_together_when_middle_entry_has_no_target(toy, tmp_path):
    root, ids, rows = toy
    chain = [r["id"] for r in rows if r["family"] == "chain"]
    assert len(chain) == 3
    middle = root / "evals" / chain[1] / "eval.json"
    hidden = middle.with_name("hidden.json")
    middle.rename(hidden)
    try:
        res = run_baselines(
            root / "features",
            root / "evals",
            "trajectory",
            tmp_path / "t",
            n_splits=3,
            n_boot=20,
            models=["delta_norm_ridge"],
        )
    finally:
        hidden.rename(middle)
    assert res["n_edits"] == len(ids) - 1
    flat = [i for fold in res["folds"] for i in fold]
    assert chain[1] not in flat and sorted(flat) == sorted(set(ids) - {chain[1]})
    holders = [k for k, fold in enumerate(res["folds"]) if chain[0] in fold or chain[2] in fold]
    assert len(holders) == 1 and {chain[0], chain[2]} <= set(res["folds"][holders[0]])


def test_trajectory_split_needs_the_zoo_manifest(toy, tmp_path):
    root, _ids, _rows = toy
    import shutil

    feats = tmp_path / "features"
    shutil.copytree(root / "features", feats)
    manifest = json.loads((feats / "manifest.json").read_text())
    manifest["zoo"] = str(tmp_path / "gone")
    (feats / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="zoo manifest"):
        run_baselines(feats, root / "evals", "trajectory", tmp_path / "o", n_boot=10)
    res = run_baselines(
        feats,
        root / "evals",
        "trajectory",
        tmp_path / "o2",
        n_splits=3,
        n_boot=10,
        zoo=root / "zoo",
        models=["delta_norm_ridge"],
    )
    assert res["split"] == "trajectory"


def test_delta_from_base_rejects_targets_missing_from_base(toy, tmp_path):
    root, _ids, _rows = toy
    evals = tmp_path / "evals"
    import shutil

    shutil.copytree(root / "evals", evals)
    (evals / "base").mkdir()
    (evals / "base" / "eval.json").write_text(json.dumps({"metrics": {"acc": 0.9}}))
    with pytest.raises(ValueError, match="ppl"):
        run_baselines(root / "features", evals, "seed", tmp_path / "o", delta_from_base=True)
    (evals / "base" / "eval.json").write_text(json.dumps({"metrics": {"acc": 0.9, "ppl": 10.0}}))
    res = run_baselines(
        root / "features",
        evals,
        "seed",
        tmp_path / "o2",
        n_splits=3,
        n_boot=10,
        delta_from_base=True,
        models=["delta_norm_ridge"],
    )
    assert res["delta_from_base"] and res["target_names"] == ["acc", "ppl"]
