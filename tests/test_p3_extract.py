import json

import pytest
import torch
from p3_toy import build_toy_zoo, toy_base_state, write_lora
from safetensors.torch import load_file

from wsp.features import DELTA_STAT_NAMES
from wsp.features.extract import effective_delta, extract_entry, main, run_extract
from wsp.zoo.apply import lora_dense_delta


@pytest.fixture(scope="module")
def base():
    return toy_base_state()


@pytest.fixture
def zoo(tmp_path, base):
    z = tmp_path / "zoo"
    ids = build_toy_zoo(z, base)
    return z, ids


def test_effective_delta_is_bf16_rounded_and_reports_losses():
    W = torch.randn(32, 32).bfloat16()
    tiny = torch.randn(32, 32) * 1e-5  # far below bf16 resolution of W
    eff, stats = effective_delta(W, tiny)
    expected = (W.float() + tiny.float()).bfloat16().float() - W.float()
    assert torch.equal(eff, expected)
    assert 0.8 < stats["fraction_nonzero_lost"] <= 1.0
    assert stats["effective_delta_norm"] <= stats["delta_norm"] * 50
    _big, stats = effective_delta(W, torch.randn(32, 32))
    assert stats["fraction_nonzero_lost"] < 0.01
    assert abs(stats["effective_delta_norm"] / stats["delta_norm"] - 1) < 0.05


def test_extract_dense_entry_writes_features_pooled_and_metadata(tmp_path, zoo, base):
    z, ids = zoo
    out = tmp_path / "out" / ids[0]
    info = extract_entry(z / ids[0], base, out, base_commit="abc", n_layers=4)
    assert (out / "features.safetensors").exists() and (out / "pooled.safetensors").exists()
    meta = json.loads((out / "features.json").read_text())
    assert meta == info
    assert meta["edit_id"] == ids[0] and meta["family"] == "random_dir"
    assert meta["base_commit"] == "abc" and meta["parent_id"] is None
    assert meta["eligibility"] == "unclassified" and "step_semantics" in meta
    names = [t["name"] for t in meta["tensors"]]
    assert names == sorted(names) and len(names) == 4
    assert all(t["shape"] and t["row"] == i for i, t in enumerate(meta["tensors"]))
    feats = load_file(out / "features.safetensors")
    assert feats["stats"].shape == (4, len(DELTA_STAT_NAMES))
    assert feats["svd"].shape == (4, 8, 129)
    assert "lora" not in feats
    assert meta["rounding_stats"]["fraction_nonzero_lost"] >= 0
    assert meta["norms"]["fro_effective"] > 0
    # stats row 0 col 0 is ||Δ_eff||_F of the first tensor
    t0 = meta["tensors"][0]
    assert feats["stats"][0, 0].item() == pytest.approx(t0["fro_effective"], rel=1e-4)
    pooled = load_file(out / "pooled.safetensors")
    assert pooled["stats"].shape == (522,) and pooled["svd"].shape == (56,)
    assert pooled["stats"][0].item() > 0  # global group fro mean


def test_unchanged_effective_tensors_are_listed_not_featurised(tmp_path, zoo, base):
    z, _ids = zoo
    from p3_toy import write_dense

    tiny_id = write_dense(z, base, "random_dir", 9, 1e-9)
    info = extract_entry(z / tiny_id, base, tmp_path / "o", n_layers=4)
    assert info["tensors"] == []
    assert len(info["unchanged_tensors"]) == 4
    assert load_file(tmp_path / "o" / "pooled.safetensors")["stats"].abs().sum() == 0


def test_extract_lora_entry_has_canonical_tokens_matching_dense_delta(tmp_path, zoo, base):
    z, ids = zoo
    lora_id = ids[6]
    info = extract_entry(z / lora_id, base, tmp_path / "l", n_layers=4)
    feats = load_file(tmp_path / "l" / "features.safetensors")
    assert feats["lora"].shape == (2, 4, 129)
    assert [t["lora_row"] for t in info["tensors"]] == [0, 1]
    dense = lora_dense_delta(z / lora_id / "lora", base.keys())
    for t in info["tensors"]:
        assert t["name"] in dense
        s = torch.linalg.svdvals(dense[t["name"]])[:4]
        got = feats["lora"][t["lora_row"], :, 0]
        assert torch.allclose(got, s, rtol=1e-4)
        eff = feats["svd"][t["row"], :4, 0]
        assert torch.allclose(eff, s, rtol=0.15)  # rounding to bf16 perturbs but does not erase


def test_lora_tokens_invariant_to_refactorisation(tmp_path, base):
    G = torch.tensor([[2.0, 1, 0, 0], [0, 1, 0.5, 0], [0, 0, 1, 1], [1, 0, 0, 3]])
    zoo = tmp_path / "zoo"
    zoo.mkdir()
    a = write_lora(zoo, base, 5, 0.1)
    b_zoo = tmp_path / "zoo2"
    b_zoo.mkdir()
    b = write_lora(b_zoo, base, 5, 0.1, refactor=G)
    ia = extract_entry(zoo / a, base, tmp_path / "a", n_layers=4)
    ib = extract_entry(b_zoo / b, base, tmp_path / "b", n_layers=4)
    assert ia["tensors"][0]["name"] == ib["tensors"][0]["name"]
    fa = load_file(tmp_path / "a" / "features.safetensors")["lora"]
    fb = load_file(tmp_path / "b" / "features.safetensors")["lora"]
    assert torch.allclose(fa, fb, atol=2e-3, rtol=1e-3)


def test_extraction_is_deterministic_and_resumable(tmp_path, zoo, base):
    z, ids = zoo
    r1 = run_extract(z, tmp_path / "a", base_state=base, ids=ids[:2], n_layers=4)
    r2 = run_extract(z, tmp_path / "b", base_state=base, ids=ids[:2], n_layers=4)
    for i in ids[:2]:
        for f in ("features.safetensors", "pooled.safetensors"):
            assert (tmp_path / "a" / i / f).read_bytes() == (tmp_path / "b" / i / f).read_bytes()
    assert r1["done"] == 2 and r1["skipped"] == 0
    mtime = (tmp_path / "a" / ids[0] / "features.json").stat().st_mtime_ns
    r3 = run_extract(z, tmp_path / "a", base_state=base, ids=ids[:3], n_layers=4)
    assert r3["done"] == 1 and r3["skipped"] == 2
    assert (tmp_path / "a" / ids[0] / "features.json").stat().st_mtime_ns == mtime
    assert r2["done"] == 2


def test_out_manifest_has_git_sha_config_hash_and_rejects_config_change(tmp_path, zoo, base):
    z, _ids = zoo
    run_extract(z, tmp_path / "o", base_state=base, limit=2, n_layers=4)
    m = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert m["git_sha"] and len(m["config_hash"]) == 64
    assert m["config"]["rank"] == 8 and m["config"]["proj_dim"] == 64
    assert m["n_entries"] == 2 and m["zoo"] == str(z)
    with pytest.raises(ValueError, match="config"):
        run_extract(z, tmp_path / "o", base_state=base, limit=2, n_layers=4, rank=4)


def test_failed_entry_is_recorded_and_does_not_abort(tmp_path, zoo, base):
    z, ids = zoo
    (z / ids[1] / "delta.safetensors").write_bytes(b"corrupt")
    r = run_extract(z, tmp_path / "o", base_state=base, ids=ids[:3], n_layers=4)
    assert r["done"] == 2 and list(r["failed"]) == [ids[1]]
    assert not (tmp_path / "o" / ids[1] / "features.json").exists()


def test_entries_without_manifest_row_or_artifacts_are_not_processed(tmp_path, zoo, base):
    z, ids = zoo
    (z / "ffff").mkdir()
    (z / "ffff" / "edit.json").write_text("{}")  # half-written build entry, not in manifest
    r = run_extract(z, tmp_path / "o", base_state=base, n_layers=4)
    assert r["done"] == len(ids) and r["failed"] == {}


def test_cli_main_with_toy_base_file(tmp_path, zoo, base):
    from safetensors.torch import save_file

    z, ids = zoo
    save_file(base, tmp_path / "base.safetensors")
    out = tmp_path / "cli"
    main(
        [
            "--zoo",
            str(z),
            "--out",
            str(out),
            "--limit",
            "1",
            "--base-file",
            str(tmp_path / "base.safetensors"),
        ]
    )
    assert (out / ids[0] / "pooled.safetensors").exists()


def test_manifest_header_rows_without_id_are_ignored(tmp_path, zoo, base):
    z, _ids = zoo
    with (z / "manifest.jsonl").open("a") as f:
        f.write(json.dumps({"config_hash": "x", "dry_run": False, "families_requested": []}) + "\n")
        f.write("\n")
    r = run_extract(z, tmp_path / "o", base_state=base, limit=2, n_layers=4)
    assert r["done"] == 2


def test_output_bytes_do_not_depend_on_worker_count(tmp_path, zoo, base):
    z, ids = zoo
    run_extract(z, tmp_path / "a", base_state=base, ids=ids[:4], n_layers=4, workers=1)
    run_extract(z, tmp_path / "b", base_state=base, ids=ids[:4], n_layers=4, workers=2)
    for i in ids[:4]:
        for f in ("features.safetensors", "pooled.safetensors", "features.json"):
            assert (tmp_path / "a" / i / f).read_bytes() == (tmp_path / "b" / i / f).read_bytes()
