"""P4 contracts; all labels here are synthetic and confined to temporary toy zoos."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from p3_toy import build_toy_zoo, toy_base_state

from wsp.features.extract import run_extract
from wsp.predictor.data import FeatureDataset, collate_trajectories
from wsp.predictor.models import build_model, effective_rank
from wsp.predictor.train import main, run_training


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("p4")
    base = toy_base_state()
    ids = build_toy_zoo(root / "zoo", base)
    run_extract(root / "zoo", root / "features", base_state=base, n_layers=4)
    for i in ids:
        meta = json.loads((root / "features" / i / "features.json").read_text())
        score = 0.2 + 0.6 / (1 + meta["norms"]["fro_effective"])
        dest = root / "evals" / i
        dest.mkdir(parents=True)
        (dest / "eval.json").write_text(json.dumps({"metrics": {"acc": score}}))
    (root / "evals" / "base").mkdir()
    (root / "evals" / "base" / "eval.json").write_text('{"metrics":{"acc":0.9}}')
    return root


def test_dataset_layout_chains_padding_and_no_targets(toy, tmp_path):
    ds = FeatureDataset(toy / "features", toy / "evals")
    assert len(ds) == 12 and ds.stats_dim == 1098 and ds.pooled_dim == 2316
    batch = collate_trajectories([ds[i] for i in range(len(ds))])
    assert batch["mask"].sum(1).max() == 3
    assert batch["tokens"].shape[-1] == 1032
    assert batch["target"].shape == (12, 1)
    assert not batch["token_mask"][~batch["mask"]].any()
    with pytest.raises(ValueError, match="zero entries have targets"):
        FeatureDataset(toy / "features", tmp_path)


@pytest.mark.parametrize("name", ["m1", "m2", "m3", "m5", "m6"])
def test_models_overfit_toy_and_jepa_does_not_collapse(toy, name):
    ds = FeatureDataset(toy / "features", toy / "evals")
    batch = collate_trajectories([ds[i] for i in range(len(ds))])
    model = build_model(name, ds, seed=7, d_model=32)
    if name in {"m3", "m6"}:
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
        for _ in range(8):
            opt.zero_grad()
            loss, z = model.pretrain_loss(batch)
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
        assert effective_rank(z) > 1.2
        model.freeze_encoder()
        before = {n: p.clone() for n, p in model.encoder.named_parameters()}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    losses = []
    for _ in range(16):
        opt.zero_grad()
        pred = model(batch)
        assert pred.shape == (12, 3, 1) and torch.isfinite(pred).all()
        last = pred[torch.arange(12), batch["mask"].sum(1) - 1]
        loss = (last - batch["target"]).square().mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    assert min(losses[-4:]) < losses[0] * 0.8
    if name in {"m3", "m6"}:
        assert all(torch.equal(before[n], p) for n, p in model.encoder.named_parameters())


@pytest.mark.parametrize("name", ["m2", "m5"])
def test_future_edits_cannot_change_past_predictions(toy, name):
    ds = FeatureDataset(toy / "features", toy / "evals")
    b = collate_trajectories([ds[i] for i in range(len(ds))])
    model = build_model(name, ds, seed=2, d_model=32).eval()
    p = model(b).detach()
    changed = {k: v.clone() for k, v in b.items()}
    for key in ("tokens", "pooled", "descriptor", "stats"):
        changed[key][:, 2] += 12
    torch.testing.assert_close(model(changed)[:, :2], p[:, :2])


def test_training_fold_outputs_provenance_and_deltas(toy, tmp_path):
    out = tmp_path / "run"
    r = run_training(
        toy / "features",
        toy / "evals",
        "seed",
        out,
        model="m6",
        epochs=2,
        pretrain_epochs=2,
        n_splits=3,
        d_model=32,
        n_boot=20,
    )
    assert r["targets"]["acc"]["n"] == 12
    rows = pd.concat([pd.read_parquet(p) for p in out.glob("fold-*.parquet")])
    assert rows.id.nunique() == 12
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["git_sha"] and manifest["config_hash"] and manifest["backend"]
    for fold in r["folds"]:
        assert not (set(fold["train_ids"]) & set(fold["validation_ids"]))
        assert not (set(fold["test_ids"]) & set(fold["validation_ids"]))
    assert (out / "results.md").exists()
    assert r["m6_acceptance"]["accepted"] is False  # one synthetic benchmark cannot meet H3
    for fold in r["folds"]:
        assert fold["collapse_check"]["frozen_predictor_erank"] >= 0
        assert isinstance(fold["collapse_check"]["acceptance_gate"], bool)
    (toy / "evals" / "base").mkdir(exist_ok=True)
    (toy / "evals" / "base" / "eval.json").write_text('{"metrics":{"acc":0.9}}')
    ds = FeatureDataset(toy / "features", toy / "evals", target="delta")
    assert (ds.targets.numpy() < 0).any()
    np.testing.assert_allclose(ds.base_scores, [0.9])
    with pytest.raises(SystemExit, match="zero entries have targets"):
        main(
            [
                "--model",
                "m1",
                "--features",
                str(toy / "features"),
                "--evals",
                str(tmp_path / "empty"),
                "--split",
                "seed",
                "--out",
                str(out),
            ]
        )


def test_residual_predictor_identity_bias_and_rank():
    from wsp.predictor.models import ResidualPred

    pred = ResidualPred(32)
    torch.testing.assert_close(pred.weight, torch.eye(32))
    assert pred.beta.item() == 10
    z = torch.randn(16, 32) * 0.1
    torch.testing.assert_close(pred(z), z, atol=0.001, rtol=0.01)
    assert effective_rank(torch.ones(10, 32)) == 0
    assert effective_rank(z) > 5


def test_seeds_padding_and_stateless_contract(toy):
    ds = FeatureDataset(toy / "features", toy / "evals")
    b = collate_trajectories([ds[i] for i in range(len(ds))])
    for name in ("m1", "m2", "m3", "m5", "m6"):
        a = build_model(name, ds, seed=123, d_model=32).eval()
        other = build_model(name, ds, seed=123, d_model=32).eval()
        torch.testing.assert_close(a(b), other(b))
        changed = {k: v.clone() for k, v in b.items()}
        changed["tokens"][~b["token_mask"]] = 100
        changed["pooled"][~b["mask"]] = 100
        torch.testing.assert_close(a(b)[b["mask"]], a(changed)[b["mask"]])
    a = build_model("m1", ds, seed=0, d_model=32).eval()
    changed = {k: v.clone() for k, v in b.items()}
    changed["tokens"][:, 0] += 1
    torch.testing.assert_close(a(b)[:, 1:], a(changed)[:, 1:])


def test_missing_middle_target_keeps_ancestry_and_folds(toy, tmp_path):
    import shutil

    evals = tmp_path / "evals"
    shutil.copytree(toy / "evals", evals)
    ds = FeatureDataset(toy / "features", evals)
    chain = max((ds.chain(r["id"]) for r in ds.rows), key=len)
    (evals / chain[1] / "eval.json").unlink()
    ds = FeatureDataset(toy / "features", evals)
    assert ds.chain(chain[2]) == chain
    assert ds.missing_predecessor_targets == 1
    for train, test in ds.folds("trajectory", 3):
        a, b = {ds.rows[i]["id"] for i in train}, {ds.rows[i]["id"] for i in test}
        assert not (chain[0] in a and chain[2] in b)
        assert not (chain[0] in b and chain[2] in a)


def test_index_payloads_are_real_targets_and_incomplete_are_excluded(toy, tmp_path):
    ds = FeatureDataset(toy / "features", toy / "evals")
    rows = [
        {"edit_id": r["id"], "eval_status": "complete", "metrics": {"acc": 0.4}} for r in ds.rows
    ]
    rows[0]["eval_status"] = "oom"
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = FeatureDataset(toy / "features", index, target="level")
    assert len(ds) == 11


def test_relations_sources_and_flagged_lora(toy, tmp_path):
    import shutil

    from wsp.predictor.data import read_relations

    root = tmp_path / "copy"
    shutil.copytree(toy, root)
    rows = read_relations(root / "zoo")
    chain = [r for r in rows.values() if r["family"] == "chain"]
    parent = next(r for r in rows.values() if r["family"] == "lora_sft")
    child = chain[-1]
    child.update(family="task_vector", parent_id=parent["id"])
    # Manifest is authoritative for edge kind; features must agree on parent id.
    meta_path = root / "features" / child["id"] / "features.json"
    meta = json.loads(meta_path.read_text())
    meta.update(family="task_vector", parent_id=parent["id"])
    meta_path.write_text(json.dumps(meta))
    (root / "zoo" / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows.values())
    )
    ds = FeatureDataset(root / "features", root / "evals", zoo=root / "zoo")
    assert ds.chain(child["id"]) == [child["id"]]
    assert ds.metadata[child["id"]]["relation"] == "source"
    torch.testing.assert_close(
        ds.entry(child["id"])["pooled"][-ds.stats_dim :], ds.entry(parent["id"])["stats"]
    )
    j = next(j for j, r in enumerate(ds.rows) if r["id"] == child["id"])
    raw = json.loads((root / "evals" / child["id"] / "eval.json").read_text())["metrics"]["acc"]
    assert ds.targets[j, 0] == pytest.approx(raw - 0.9)
    flagged = next(r for r in chain if r["parent_id"] and r["id"] != child["id"])
    edit_path = root / "zoo" / flagged["id"] / "edit.json"
    edit = json.loads(edit_path.read_text())
    edit["params"]["step_semantics"] = "lora_vs_base"
    edit_path.write_text(json.dumps(edit))
    assert read_relations(root / "zoo")[flagged["id"]]["relation"] == "successor"


def test_mixed_family_unlabeled_ancestor_purge():
    from wsp.predictor.data import ancestor_ids, clean_training

    rows = [
        {"id": "held", "family": "lora_sft", "seed": 0},
        {"id": "later", "family": "chain", "seed": 1},
        {"id": "source", "family": "task_vector", "seed": 2},
        {"id": "safe", "family": "random_dir", "seed": 3},
    ]
    metadata = {r["id"]: r for r in rows}
    metadata["middle"] = {"family": "chain", "seed": 1, "params": {"inner_family": "lora_sft"}}
    ancestry = {"held": None, "middle": None, "later": "middle", "source": "held", "safe": None}
    train, dropped = clean_training(rows, [1, 2, 3], [0], metadata, ancestry, "family")
    assert train.tolist() == [3] and dropped == 2
    assert not ancestor_ids(rows[train[0]]["id"], ancestry) & {"held", "middle"}


def test_step_delta_level_and_missing_predecessor(toy, tmp_path):
    import shutil

    from wsp.predictor.data import read_targets

    ds = FeatureDataset(toy / "features", toy / "evals")
    raw = read_targets(toy / "evals")
    for i, row in enumerate(ds.rows):
        previous = ds.successors[row["id"]] or "base"
        assert ds.targets[i, 0] == pytest.approx(raw[row["id"]]["acc"] - raw[previous]["acc"])
    level = FeatureDataset(toy / "features", toy / "evals", target="level")
    assert level.targets[0, 0] == pytest.approx(raw[level.rows[0]["id"]]["acc"])
    shutil.copytree(toy / "evals", tmp_path / "evals")
    (tmp_path / "evals" / "base" / "eval.json").unlink()
    missing = FeatureDataset(toy / "features", tmp_path / "evals")
    assert missing.missing_predecessor_targets == 10 and len(missing) == 2


def test_m5_layers_dropout_matched_control_and_descriptor(toy):
    from wsp.predictor.data import descriptor

    ds = FeatureDataset(toy / "features", toy / "evals")
    m5 = build_model("m5", ds, d_model=32)
    matched = build_model("m1_matched", ds, d_model=32)
    assert sum(isinstance(m, torch.nn.MultiheadAttention) for m in m5.modules()) == 2
    assert m5.edits.layers[0].dropout.p == 0.2
    assert (
        abs(sum(p.numel() for p in m5.parameters()) - sum(p.numel() for p in matched.parameters()))
        == 4
    )
    torch.testing.assert_close(
        descriptor({"family": "chain", "params": {"length": 8, "inner_seed": 99}}),
        descriptor({"family": "chain"}),
    )


def test_mixed_target_configs_refused(tmp_path):
    from wsp.predictor.data import read_targets

    p = tmp_path / "index.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"id": str(i), "metrics": {"acc": 0.5}, "config_hash": str(i)})
            for i in range(2)
        )
    )
    with pytest.raises(ValueError, match="mixed evaluation"):
        read_targets(p)


@pytest.mark.parametrize("name", ["m3", "m6"])
def test_jepa_alignment_target_is_detached(toy, monkeypatch, name):
    from wsp.predictor import models

    ds = FeatureDataset(toy / "features", toy / "evals")
    b = collate_trajectories([ds[i] for i in range(len(ds))])
    net = build_model(name, ds, d_model=32)
    monkeypatch.setattr(models, "variance_loss", lambda z: z.sum() * 0)
    outputs = []

    def capture(module, args, output):
        output.retain_grad()
        outputs.append(output)

    handle = (net.encoder if name == "m6" else net.latent).register_forward_hook(capture)
    loss, _ = net.pretrain_loss(b)
    loss.backward()
    handle.remove()
    if name == "m6":
        assert outputs[-1].grad.abs().sum() == 0
        assert outputs[-2].grad.abs().sum() > 0
    else:
        # Last state of every trajectory is a target only.
        z = outputs[0]
        assert (z.grad[torch.arange(len(ds)), b["mask"].sum(1) - 1] == 0).all()
        assert z.grad.abs().sum() > 0


def test_chunked_scan_matches_loop_and_gradients():
    from wsp.predictor.scan import loop_scan, selective_scan

    torch.manual_seed(10)
    inputs = [
        torch.randn(2, 9, 4),
        torch.rand(2, 9, 4),
        -torch.rand(4, 3),
        torch.randn(2, 9, 3),
        torch.randn(2, 9, 3),
        torch.randn(4),
    ]
    for x in inputs:
        x.requires_grad_()
    a = selective_scan(*inputs, chunk_size=4)
    b = loop_scan(*inputs)
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(a.sum(), inputs)
    gb = torch.autograd.grad(b.sum(), inputs)
    for x, y in zip(ga, gb, strict=True):
        torch.testing.assert_close(x, y)
