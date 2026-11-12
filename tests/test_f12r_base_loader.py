"""F12-R second remedy of A10 (scripts/f12_base_loader.py; independent re-check of 12 Nov 2026): the base model is
loaded only from individually resolved, confined and pinned cache leaves, staged as regular files, under an audit hook
that refuses any other read; both F12 loaders route through it. Synthetic caches and a tiny synthetic transformers model
only; the committed pin file is read, the real cache is not."""

import json
import os

import pytest
import torch

from scripts import f12_additivity_prepare as fa
from scripts import f12_base_loader as fbl
from scripts import f12_inputs_guard as fg
from scripts import f12_tokens as ft
from wsp.features import response_field as rf

REPO, REV = "Org/Toy-Model", "0123456789abcdef0123456789abcdef01234567"


def _cache(tmp_path, files):
    """A hub cache with blobs and snapshot symlinks, as huggingface_hub lays it out."""
    root = tmp_path / "hub" / ("models--" + REPO.replace("/", "--"))
    snap = root / "snapshots" / REV
    (root / "blobs").mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, data in files.items():
        blob = root / "blobs" / fg.sha256_bytes(data)
        blob.write_bytes(data)
        (snap / name).symlink_to(os.path.relpath(blob, snap))
    return tmp_path / "hub", root, snap


def _pin(tmp_path, files, state_hash="toy-state"):
    pin = {
        "schema": fbl.PIN_SCHEMA,
        "repo": REPO,
        "revision": REV,
        "base_state_hash": state_hash,
        "files": {n: {"sha256": fg.sha256_bytes(d), "bytes": len(d)} for n, d in files.items()},
    }
    p = tmp_path / "pin.json"
    p.write_text(json.dumps(pin))
    return p, fg.sha256_bytes(p.read_bytes())


FILES = {"config.json": b'{"a": 1}', "model.safetensors": b"weights"}


def _load(tmp_path, hub, pin_path, pin_sha, loader, expected="toy-state", prohibited=()):
    b = fg.ReadBoundary(allowed=[tmp_path], prohibited=list(prohibited))
    return fbl.load_verified_base(
        b, expected, pin_path=pin_path, pin_sha256=pin_sha, hub_cache=hub, loader=loader,
        state_hash_fn=lambda s: "toy-state",
    )  # fmt: skip


def test_stages_only_pinned_leaves_as_regular_files(tmp_path):
    hub, _root, _snap = _cache(tmp_path, FILES | {"tokenizer.json": b"tok"})
    pin_path, pin_sha = _pin(tmp_path, FILES)
    seen = {}

    def loader(dest):
        seen["names"] = sorted(p.name for p in dest.iterdir())
        seen["links"] = [p.is_symlink() for p in dest.iterdir()]
        seen["bytes"] = (dest / "model.safetensors").read_bytes()
        return {"w": torch.zeros(1)}

    _state, h = _load(tmp_path, hub, pin_path, pin_sha, loader)
    assert seen["names"] == [
        "config.json",
        "model.safetensors",
    ]  # tokenizer.json not pinned, not staged
    assert seen["links"] == [False, False] and seen["bytes"] == b"weights"
    assert h == "toy-state"


def test_recheck_demonstration_static_cache_leaf_symlink_fails_closed(tmp_path):
    """The re-check: cache config.json linked into a prohibited directory holding {}. Now refused per leaf, before
    any library read (the substituted loader is never called)."""
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "config.json").write_bytes(b"{}")
    hub, _root, snap = _cache(tmp_path, {"model.safetensors": b"weights"})
    (snap / "config.json").symlink_to(evals / "config.json")
    pin_path, pin_sha = _pin(tmp_path, {"config.json": b"{}", "model.safetensors": b"weights"})
    called = []
    with pytest.raises(fg.BoundaryError):
        _load(tmp_path, hub, pin_path, pin_sha, lambda d: called.append(d), prohibited=[evals])
    assert called == []


def test_recheck_demonstration_through_the_token_and_additivity_loaders(tmp_path, monkeypatch):
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "config.json").write_bytes(b"{}")
    hub, _root, snap = _cache(tmp_path, {"model.safetensors": b"weights"})
    (snap / "config.json").symlink_to(evals / "config.json")
    pin_path, pin_sha = _pin(tmp_path, {"config.json": b"{}", "model.safetensors": b"weights"})
    monkeypatch.setattr(fbl, "PIN_PATH", pin_path)
    monkeypatch.setattr(fbl, "PIN_SHA256", pin_sha)
    monkeypatch.setattr(fbl, "hub_cache_dir", lambda: hub)
    called = []
    monkeypatch.setattr(fbl, "transformers_load", lambda d: called.append(d))
    monkeypatch.setattr(fa.pf, "load_base_state", lambda h: called.append("unconfined"))
    for mod in (ft, fa):
        b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[evals])
        with pytest.raises(fg.BoundaryError):
            mod.load_base(b, "toy-state")
    assert called == []


def test_leaf_outside_the_model_cache_root_refused(tmp_path):
    hub, _root, snap = _cache(tmp_path, {"model.safetensors": b"weights"})
    other = tmp_path / "hub" / "models--Other--Model" / "blobs"
    other.mkdir(parents=True)
    (other / "x").write_bytes(b'{"a": 1}')
    (snap / "config.json").symlink_to(other / "x")
    pin_path, pin_sha = _pin(tmp_path, FILES)
    with pytest.raises(fg.BoundaryError, match="outside the allowed"):
        _load(tmp_path, hub, pin_path, pin_sha, lambda d: {})


def test_leaf_hash_and_pin_checks(tmp_path):
    hub, _root, _snap = _cache(tmp_path, FILES)
    pin_path, pin_sha = _pin(tmp_path, FILES | {"config.json": b'{"a": 2}'})
    with pytest.raises(fg.AuthenticationError):
        _load(tmp_path, hub, pin_path, pin_sha, lambda d: {})
    pin_path, pin_sha = _pin(tmp_path, FILES)
    with pytest.raises(fg.AuthenticationError):
        _load(tmp_path, hub, pin_path, "0" * 64, lambda d: {})
    with pytest.raises(ValueError, match="base state"):
        _load(tmp_path, hub, pin_path, pin_sha, lambda d: {}, expected="other-state")
    with pytest.raises(ValueError, match="base state hash"):
        b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[])
        fbl.load_verified_base(
            b, "toy-state", pin_path=pin_path, pin_sha256=pin_sha, hub_cache=hub,
            loader=lambda d: {"w": torch.zeros(1)}, state_hash_fn=lambda s: "loaded-other",
        )  # fmt: skip


def test_library_reads_outside_the_verified_set_fail_closed(tmp_path):
    hub, _root, _snap = _cache(tmp_path, FILES)
    pin_path, pin_sha = _pin(tmp_path, FILES)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")

    def nosy(dest):
        return open(outside).read()

    with pytest.raises(PermissionError, match="confined base load"):
        _load(tmp_path, hub, pin_path, pin_sha, nosy)

    def unpinned(dest):
        return open(dest / "tokenizer.json").read()

    with pytest.raises(FileNotFoundError):
        _load(tmp_path, hub, pin_path, pin_sha, unpinned)
    open(outside).read()  # noqa: SIM115  (the confinement ends with the load)


def test_staging_removed_and_reads_recorded(tmp_path):
    hub, root, _snap = _cache(tmp_path, FILES)
    pin_path, pin_sha = _pin(tmp_path, FILES)
    kept = {}
    b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[])
    fbl.load_verified_base(b, "toy-state", pin_path=pin_path, pin_sha256=pin_sha, hub_cache=hub,
                           loader=lambda d: kept.setdefault("d", d) and {}, state_hash_fn=lambda s: "toy-state")  # fmt: skip
    assert not kept["d"].exists()
    leaves = [r for r in b.reads if r.get("note", "").startswith("base-model leaf")]
    assert sorted(r["path"].rsplit("/", 1)[-1] for r in leaves) == [
        "config.json",
        "model.safetensors",
    ]
    assert all(r["authenticated"] and r["resolved"].startswith(str(root.resolve())) for r in leaves)


def test_real_transformers_load_under_confinement(tmp_path):
    """A tiny synthetic Llama saved by transformers loads through the confined path with the same state hash."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2)  # fmt: skip
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    src = tmp_path / "saved"
    model.save_pretrained(src)
    files = {p.name: p.read_bytes() for p in src.iterdir() if p.is_file()}
    hub, _root, _snap = _cache(tmp_path, files)
    want = fbl.state_hash(model.state_dict())
    pin_path, pin_sha = _pin(tmp_path, files, state_hash=want)
    b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[])
    state, h = fbl.load_verified_base(b, want, pin_path=pin_path, pin_sha256=pin_sha, hub_cache=hub,
                                      loader=fbl.transformers_load)  # fmt: skip
    assert h == want and set(state) == set(model.state_dict())


def test_committed_pin_binds_the_base_state():
    pin = json.loads(fbl.PIN_PATH.read_text())
    assert fg.sha256_bytes(fbl.PIN_PATH.read_bytes()) == fbl.PIN_SHA256
    assert (pin["repo"], pin["revision"]) == fbl.MODEL
    tm = json.loads((fbl.ROOT / "docs/prereg/a5_manifests/f12/tokens_manifest.json").read_text())
    assert pin["base_state_hash"] == tm["base"]["state_hash"]
    assert set(pin["files"]) == set(fbl.BASE_FILES)
    assert all(len(v["sha256"]) == 64 and v["bytes"] > 0 for v in pin["files"].values())


def test_state_hash_is_the_g2_convention():
    s = {"a": torch.ones(2), "b": torch.zeros(3)}
    assert fbl.state_hash(s) == rf._hash({n: rf._tensor_hash(t) for n, t in s.items()})


def test_shared_hub_blob_store_chain_accepted_other_hub_trees_refused(tmp_path):
    """The real cache layout: snapshot -> models--<repo>/blobs/<sha> -> hub/blobs/<xx>/<sha> (shared store). The
    allowed cache root is the model's own tree plus the hub's shared blob store; other hub trees stay refused."""
    hub, root, snap = _cache(tmp_path, {"config.json": b'{"a": 1}'})
    shared = hub / "blobs" / "ab"
    shared.mkdir(parents=True)
    (shared / "abcd").write_bytes(b"weights")
    blob = root / "blobs" / "w"
    blob.symlink_to(os.path.relpath(shared / "abcd", blob.parent))
    (snap / "model.safetensors").symlink_to(os.path.relpath(blob, snap))
    pin_path, pin_sha = _pin(tmp_path, FILES)
    _state, h = _load(tmp_path, hub, pin_path, pin_sha, lambda d: {})
    assert h == "toy-state"
    ds = hub / "datasets--Org--Data" / "blobs"
    ds.mkdir(parents=True)
    (ds / "x").write_bytes(b"weights")
    (snap / "model.safetensors").unlink()
    (snap / "model.safetensors").symlink_to(ds / "x")
    with pytest.raises(fg.BoundaryError, match="outside the allowed"):
        _load(tmp_path, hub, pin_path, pin_sha, lambda d: {})
