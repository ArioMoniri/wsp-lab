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
    assert pooled["stats"].shape == (1098,) and pooled["svd"].shape == (120,)
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


def test_hybrid_entry_edit_on_every_qwen35_tensor_type_reaches_per_type_pools(tmp_path):
    from p3_toy import _descriptor, tiny_qwen35_shapes
    from safetensors.torch import save_file

    from wsp.features.pool import TYPES, classify

    shapes = tiny_qwen35_shapes()
    g = torch.Generator().manual_seed(0)
    base = {k: torch.randn(*s, generator=g).bfloat16() for k, s in shapes.items()}
    delta = {k: (torch.randn(*s, generator=g) * 0.1).bfloat16() for k, s in shapes.items()}
    desc = _descriptor("random_dir", 1)
    d = tmp_path / "zoo" / desc.id
    d.mkdir(parents=True)
    (d / "edit.json").write_text(desc.to_json())
    save_file(delta, d / "delta.safetensors")
    info = extract_entry(d, base, tmp_path / "o")
    assert {t["name"] for t in info["tensors"]} == set(shapes)
    pooled = load_file(tmp_path / "o" / "pooled.safetensors")
    spectrum = pooled["svd"].reshape(len(TYPES), 8)
    used = {classify(n)[0] for n in shapes}
    assert "other" not in used and used == {t for t in TYPES if t != "other"}
    assert all(spectrum[TYPES.index(t)].sum() > 0 for t in used)


def _manifest_failed(out):
    return json.loads((out / "manifest.json").read_text())["failed"]


@pytest.mark.parametrize(
    "bad",
    [
        "{",  # invalid JSON
        "[]",  # not an object
        json.dumps({"base_model": "toy/model"}),  # family missing
        json.dumps({"family": "x", "base_model": "toy/model", "seed": "zero"}),
        json.dumps({"family": "x", "base_model": "toy/model", "params": [1]}),
        json.dumps({"family": 5, "base_model": "toy/model"}),
    ],
)
def test_malformed_edit_json_is_recorded_and_healthy_entries_continue(tmp_path, zoo, base, bad):
    z, ids = zoo
    (z / ids[0] / "edit.json").write_text(bad)
    r = run_extract(z, tmp_path / "o", base_state=base, ids=ids[:3], n_layers=4)
    assert r["done"] == 2 and list(r["failed"]) == [ids[0]]
    recorded = _manifest_failed(tmp_path / "o")
    assert set(recorded) == {ids[0]} and "MetadataError" in recorded[ids[0]]
    assert (tmp_path / "o" / ids[1] / "features.json").exists()
    assert not (tmp_path / "o" / ids[0] / "features.json").exists()
    # still recorded (and still non-fatal) when the base has to be discovered from the entries
    r = run_extract(z, tmp_path / "o", base_state=base, ids=ids[:3], n_layers=4)
    assert r["done"] == 0 and r["skipped"] == 2 and list(r["failed"]) == [ids[0]]


def test_malformed_edit_json_does_not_break_base_discovery(tmp_path, zoo, base, monkeypatch):
    import wsp.features.extract as ex

    z, ids = zoo
    (z / ids[0] / "edit.json").write_text("{")
    monkeypatch.setattr(ex, "resolve_base_commit", lambda model, revision: "c" * 40)
    monkeypatch.setattr(ex, "load_base_state", lambda model, revision: (base, "c" * 40))
    r = run_extract(z, tmp_path / "o", ids=ids[:3], n_layers=4)
    assert r["done"] == 2 and list(r["failed"]) == [ids[0]]
    assert json.loads((tmp_path / "o" / "manifest.json").read_text())["base_commit"] == "c" * 40


def test_malformed_manifest_rows_are_recorded_not_fatal(tmp_path, zoo, base):
    z, ids = zoo
    with (z / "manifest.jsonl").open("a") as f:
        f.write("not json\n")
        f.write(json.dumps({"id": "../escape", "family": "x"}) + "\n")
        f.write(json.dumps({"id": "abcd", "seed": 1}) + "\n")  # family missing
        f.write(json.dumps({"family": "x", "parent_id": None}) + "\n")  # id missing
        f.write(json.dumps({"id": "ef01", "family": "x", "parent_id": 7}) + "\n")
        f.write(json.dumps([1]) + "\n")
    r = run_extract(z, tmp_path / "o", base_state=base, n_layers=4)
    assert r["done"] == len(ids)
    failed = _manifest_failed(tmp_path / "o")
    assert len(failed) == 6 and all(k.startswith("manifest.jsonl:") for k in failed)
    assert all("MetadataError" in v for v in failed.values())
    assert "invalid id" in failed[f"manifest.jsonl:{len(ids) + 2}"]


def test_cli_exits_nonzero_on_metadata_failure(tmp_path, zoo, base):
    from safetensors.torch import save_file

    z, ids = zoo
    (z / ids[0] / "edit.json").write_text("{")
    save_file(base, tmp_path / "base.safetensors")
    with pytest.raises(SystemExit, match="failed"):
        main(
            ["--zoo", str(z), "--out", str(tmp_path / "o"), "--limit", "2"]
            + ["--base-file", str(tmp_path / "base.safetensors")]
        )
    assert (tmp_path / "o" / ids[1] / "features.json").exists()


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"threads": 2}, "threads"),
        ({"rank": 4}, "config.rank"),
        ({"proj_dim": 32}, "config.proj_dim"),
        ({"seed": 7}, "config.seed"),
        ({"feature_version": 99}, "config.feature_version"),
        ({"base_commit": "other"}, "base"),
    ],
)
def test_resume_refused_when_numerical_identity_changes(tmp_path, zoo, base, change, field):
    z, ids = zoo
    kw = {"base_state": base, "ids": ids[:2], "n_layers": 4, "base_commit": "abc"}
    run_extract(z, tmp_path / "o", **kw)
    with pytest.raises(ValueError, match="refusing to resume") as e:
        run_extract(z, tmp_path / "o", **{**kw, **change})
    assert field in str(e.value) and "new --out" in str(e.value)
    assert not (tmp_path / "o" / ids[2] / "features.json").exists()
    run_extract(z, tmp_path / "o", **kw, workers=2)  # same identity, other worker count: fine


@pytest.mark.parametrize("package", ["torch", "numpy"])
def test_resume_refused_when_library_version_changes(tmp_path, zoo, base, monkeypatch, package):
    import numpy

    z, ids = zoo
    kw = {"base_state": base, "ids": ids[:2], "n_layers": 4}
    run_extract(z, tmp_path / "o", **kw)
    monkeypatch.setattr({"torch": torch, "numpy": numpy}[package], "__version__", "0.0.1")
    with pytest.raises(ValueError, match=f"refusing to resume.*{package}"):
        run_extract(z, tmp_path / "o", **kw)


def test_resume_refused_for_a_different_base_without_commit(tmp_path, zoo, base):
    z, ids = zoo
    kw = {"ids": ids[:2], "n_layers": 4}
    run_extract(z, tmp_path / "o", base_state=base, **kw)
    other = {k: v.clone() for k, v in base.items()}
    key = next(iter(other))
    other[key][0, 0] = other[key][0, 0] + 1
    with pytest.raises(ValueError, match="refusing to resume.*base"):
        run_extract(z, tmp_path / "o", base_state=other, **kw)
    run_extract(z, tmp_path / "o", base_state=dict(base), **kw)  # identical content resumes


def test_entries_from_another_identity_are_recomputed_when_manifest_is_missing(tmp_path, zoo, base):
    z, ids = zoo
    kw = {"base_state": base, "ids": ids[:2], "n_layers": 4}
    run_extract(z, tmp_path / "o", **kw)  # threads=1
    (tmp_path / "o" / "manifest.json").unlink()  # e.g. a crash before the manifest was written
    r = run_extract(z, tmp_path / "o", threads=2, **kw)
    assert r["done"] == 2 and r["skipped"] == 0
    meta = json.loads((tmp_path / "o" / ids[0] / "features.json").read_text())
    assert meta["identity_hash"]


def test_out_manifest_records_identity(tmp_path, zoo, base):
    z, ids = zoo
    run_extract(z, tmp_path / "o", base_state=base, ids=ids[:1], n_layers=4, threads=1)
    m = json.loads((tmp_path / "o" / "manifest.json").read_text())
    ident = m["resume_identity"]
    assert ident["threads"] == 1 and ident["torch"] == torch.__version__
    assert ident["config"]["feature_version"] == 3 and ident["base"].startswith("state:")
    assert m["complete_run"] is True


@pytest.mark.parametrize(
    "cfg_patch", [{"fan_in_fan_out": True}, {"bias": "all"}, {"modules_to_save": ["lm_head"]}]
)
def test_feature_path_rejects_the_lora_configs_the_eval_path_rejects(
    tmp_path, zoo, base, cfg_patch
):
    z, ids = zoo
    lora_id = ids[6]
    cfg_path = z / lora_id / "lora" / "adapter_config.json"
    cfg_path.write_text(json.dumps({**json.loads(cfg_path.read_text()), **cfg_patch}))
    r = run_extract(z, tmp_path / "o", base_state=base, ids=[lora_id], n_layers=4)
    assert "NotImplementedError" in r["failed"][lora_id]


def test_real_build_header_rows_are_not_malformed_entries(tmp_path, zoo, base):
    z, _ids = zoo
    header = {
        "base_model": "Qwen/Qwen3.5-0.8B",
        "config_hash": "x",
        "dry_run": False,
        "families_requested": ["random_dir"],
        "family": "ignored",
        "seed": 0,
    }
    with (z / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(header) + "\n")
    r = run_extract(z, tmp_path / "o", base_state=base, limit=2, n_layers=4)
    assert r["done"] == 2 and r["failed"] == {}
