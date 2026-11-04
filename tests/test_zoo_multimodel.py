"""Multi-base-model zoo (P4b / H6): configs, dry runs, writer selection and kernels (CPU, no weights)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import wsp.zoo.build as B
from wsp.zoo import arch as A

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "arch"
CONFIGS = {
    "qwen3.5-0.8b": ROOT / "configs" / "zoo_qwen3.5-0.8b.yaml",
    "qwen3.5-4b": ROOT / "configs" / "zoo_qwen3.5-4b.yaml",
    "lfm3-1.7b": ROOT / "configs" / "zoo_lfm3-1.7b.yaml",
    "gemma-3-270m": ROOT / "configs" / "zoo_gemma-3-270m.yaml",
}
NEW = ["qwen3.5-4b", "lfm3-1.7b", "gemma-3-270m"]


def fixture(slug: str) -> dict:
    return json.loads((FIXTURES / f"{slug}.json").read_text())


def load(slug: str) -> dict:
    return yaml.safe_load(CONFIGS[slug].read_text())


@pytest.fixture
def readable_config(monkeypatch):
    """Serve each model's text config from the fixtures instead of the Hub."""
    fixtures = {fixture(s)["repo"]: fixture(s)["text_config"] for s in CONFIGS}
    monkeypatch.setattr(A, "read_text_config", lambda cfg: dict(fixtures[cfg["base_model"]]))


# ---- configs ----------------------------------------------------------------------------------


@pytest.mark.parametrize("slug", NEW)
def test_new_configs_have_the_same_families_and_sizes_as_the_08b_config(slug):
    ref, cfg = load("qwen3.5-0.8b"), load(slug)
    assert list(cfg["families"]) == list(ref["families"])
    for family, spec in ref["families"].items():
        got = cfg["families"][family]
        assert got["n"] == spec["n"], family
        depth_dependent = {
            "layers",
            "direction_index",
            "max_weight_position",
            "min_weight_distance",
        }
        for key in set(spec) - depth_dependent:
            assert got[key] == spec[key], (family, key)
    # D6: heretic_ara n=12, trials=10, production threshold trials >= 10.
    assert (
        cfg["families"]["heretic_ara"]["n"] == 12 and cfg["families"]["heretic_ara"]["trials"] == 10
    )
    assert cfg["eligibility"] == ref["eligibility"] and cfg["seed"] == ref["seed"]
    assert cfg["out_dir"] == f"data/zoo/{slug}" and cfg["base_revision"] == "main"
    assert cfg["eval"] == ref["eval"]


@pytest.mark.parametrize("slug", [*NEW, "qwen3.5-0.8b"])
def test_layer_indices_fit_the_model_and_scale_with_depth(slug):
    cfg, fx = load(slug), fixture(slug)
    n = fx["text_config"]["num_hidden_layers"]
    info = A.arch_info_from_dict(fx["text_config"], cfg.get("architecture"))
    A.validate_family_config(cfg, info)
    layers = cfg["families"]["random_dir"]["layers"]
    ref = load("qwen3.5-0.8b")["families"]["random_dir"]["layers"]
    assert layers == sorted(set(layers)) and len(layers) == len(ref)
    assert layers == sorted({round(x * n / 24) for x in ref})  # same relative depths as the 0.8B
    abl = cfg["families"]["abliteration"]
    assert abl["direction_index"]["max"] == n - 1 and abl["max_weight_position"]["max"] == n
    A.check_declared(cfg.get("architecture") or {}, info)  # declared hidden size / depth are right


def test_sandwich_norm_models_fold_and_others_do_not():
    for slug in NEW + ["qwen3.5-0.8b"]:
        cfg = load(slug)
        fold = bool(cfg["families"]["abliteration"].get("fold_post_norm"))
        info = A.arch_info_from_dict(fixture(slug)["text_config"], cfg.get("architecture"))
        assert fold == bool(info.post_norms), slug


def test_config_writers_match_the_registry_defaults():
    for slug in NEW:
        cfg, fx = load(slug), fixture(slug)
        declared = A.arch_info_from_dict(fx["text_config"], cfg["architecture"])
        default = A.arch_info_from_dict(fx["text_config"])
        assert {k: tuple(v) for k, v in declared.arch.writers.items()} == {
            k: tuple(v) for k, v in default.arch.writers.items()
        }
        assert dict(declared.post_norms) == dict(default.post_norms)
        assert declared.norm_gain == default.norm_gain


# ---- dry runs ---------------------------------------------------------------------------------


def _dry(tmp_path, slug, **kw):
    cfg = load(slug)
    cfg["out_dir"] = str(tmp_path / slug)
    path = tmp_path / f"{slug}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return B.build_zoo(path, dry_run=True, git_sha="t", **kw), Path(cfg["out_dir"])


@pytest.mark.parametrize("slug", NEW)
def test_dry_run_n3_succeeds_for_every_family(tmp_path, readable_config, slug):
    manifest, out = _dry(tmp_path, slug, n=3)
    rows = [m for m in manifest if m.get("type") != "header"]
    counts = Counter(r["family"] for r in rows)
    for family in ("random_dir", "abliteration", "heretic_ara", "lora_sft"):
        assert counts[family] == 3, (slug, counts)
    assert counts["task_vector"] > 0 and counts["chain"] > 0
    assert len({r["id"] for r in rows}) == len(rows)
    assert (out / "manifest.dryrun.jsonl").exists() and not (out / "manifest.jsonl").exists()
    lines = [json.loads(x) for x in (out / "manifest.dryrun.jsonl").read_text().splitlines()]
    head = next(x for x in lines if x.get("type") == "header")
    assert head["dry_run"] and head["architecture"]["verified"] is True
    assert head["architecture"]["hidden_size"] == fixture(slug)["text_config"]["hidden_size"]


def test_dry_run_ids_differ_between_models(tmp_path, readable_config):
    ids = {}
    for slug in NEW:
        manifest, _ = _dry(tmp_path, slug, n=3, family="random_dir")
        ids[slug] = {m["id"] for m in manifest if m.get("type") != "header"}
    assert all(len(v) == 3 for v in ids.values())
    assert len(set().union(*ids.values())) == 9


def test_layer_out_of_range_for_the_model_fails_before_anything_is_planned(
    tmp_path, readable_config
):
    cfg = load("gemma-3-270m")
    cfg["families"]["random_dir"]["layers"] = [6, 21]  # a 24-layer layout on an 18-layer model
    cfg["out_dir"] = str(tmp_path / "z")
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="layer 21"):
        B.build_zoo(p, dry_run=True, n=1)
    assert not (tmp_path / "z" / "manifest.dryrun.jsonl").exists()


def test_declared_architecture_is_the_fallback_for_a_gated_unreadable_model(tmp_path, monkeypatch):
    def unreadable(cfg):
        raise OSError("403 gated")

    monkeypatch.setattr(A, "read_text_config", unreadable)
    _, out = _dry(tmp_path, "gemma-3-270m", n=1)
    head = json.loads((out / "manifest.dryrun.jsonl").read_text().splitlines()[0])
    assert head["architecture"] == {
        "family": "gemma3",
        "hidden_size": 640,
        "num_layers": 18,
        "verified": False,
    }
    cfg = load("gemma-3-270m")
    del cfg["architecture"]["hidden_size"]
    cfg["out_dir"] = str(tmp_path / "nodecl")
    p = tmp_path / "nodecl.yaml"
    p.write_text(yaml.safe_dump(cfg))
    B.build_zoo(p, dry_run=True, n=1)  # no declaration and unreadable: unvalidated, never a crash


def test_declared_size_that_contradicts_the_readable_config_is_an_error(tmp_path, readable_config):
    cfg = load("qwen3.5-4b")
    cfg["architecture"]["hidden_size"] = 1024
    cfg["out_dir"] = str(tmp_path / "z")
    p = tmp_path / "wrong.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="hidden_size"):
        B.build_zoo(p, dry_run=True, n=1)


def test_gated_repo_resolves_its_revision_from_hub_metadata_in_dry_run_only(
    tmp_path, readable_config, monkeypatch
):
    import httpx
    from huggingface_hub.errors import GatedRepoError

    def gated(model, revision=None):
        request = httpx.Request("GET", "https://huggingface.co/x/resolve/main/config.json")
        raise GatedRepoError("403 manual approval", response=httpx.Response(403, request=request))

    monkeypatch.setattr(B, "resolve_base_revision", gated)
    monkeypatch.setattr(B, "metadata_revision", lambda model, revision=None: "a" * 40)
    _, out = _dry(tmp_path, "gemma-3-270m", n=1)
    head = json.loads((out / "manifest.dryrun.jsonl").read_text().splitlines()[0])
    assert head["base_revision"] == "a" * 40
    assert "hub metadata" in head["base_revision_source"]
    cfg = load("gemma-3-270m")
    cfg["out_dir"] = str(tmp_path / "real")
    p = tmp_path / "real.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(GatedRepoError):
        B.build_zoo(p, dry_run=False, n=1)  # a real build never proceeds without file access


# ---- writer selection and kernels on each architecture's tensors ---------------------------------

H, N_LAYERS = 8, 6


def tiny_state(slug: str, seed: int = 0) -> tuple[dict[str, torch.Tensor], A.ArchInfo]:
    """First 6 layers of the model's real tensor layout at hidden size 8 (rows of writers = 8)."""
    fx = fixture(slug)
    hid = fx["text_config"]["hidden_size"]
    g = torch.Generator().manual_seed(seed)

    def shrink(shape):
        return [H if s == hid else min(s, 12) for s in shape]

    state = {}
    for name, shape in fx["params"].items():
        parsed = A.split_key(name)
        if parsed and parsed[0] >= N_LAYERS:
            continue
        state[name] = torch.randn(*shrink(shape), generator=g) * 0.2
    cfg = load(slug)
    info = A.arch_info_from_dict(
        {**fx["text_config"], "hidden_size": H, "num_hidden_layers": N_LAYERS},
        cfg.get("architecture"),
    )
    return state, info


def _config(slug, info, **fam):
    return {
        "base_model": fixture(slug)["repo"],
        "base_revision": "r",
        "seed": 0,
        "_arch": info,
        "_config_hash": "h",
        "families": {},
        **fam,
    }


@pytest.mark.parametrize("slug", [*NEW, "qwen3.5-0.8b"])
def test_random_dir_edits_exactly_the_residual_writers_of_each_architecture(slug):
    state, info = tiny_state(slug)
    fam = {"layers": [1, 3, 5], "tensors": ["mlp.down_proj", "self_attn.o_proj"]} | {
        "weight": {"min": 1.0, "max": 1.0},
        "n_directions": {"choices": [1]},
    }
    config = _config(slug, info)
    ((desc, new_state, touched),) = B.build_random_dir(config, fam, 3, state)
    layers = desc.params["layers"]
    assert len(touched) == 2 * len(layers)
    for key in touched:
        assert A.component_of(key, info.arch) in ("attn_out", "mlp_out"), key
        assert state[key].shape[0] == H
        assert not torch.equal(new_state[key], state[key])
    changed = {k for k in state if not torch.equal(state[k], new_state[k])}
    assert changed == set(touched)  # nothing else in the model moves


def test_random_dir_refuses_a_tensor_that_does_not_write_the_residual():
    state, info = tiny_state("lfm3-1.7b")
    fam = {
        "layers": [2],
        "tensors": [r"feed_forward\.w1"],  # (12, 8): an MLP up-side matrix, rows are not residual
        "weight": {"min": 1.0, "max": 1.0},
        "n_directions": {"choices": [1]},
    }
    with pytest.raises(ValueError, match="residual"):
        B.build_random_dir(_config("lfm3-1.7b", info), fam, 0, state)


def _abl_family(**extra):
    return {
        "n": 1,
        "direction_index": {"min": 2.0, "max": 2.0},
        "max_weight": {"min": 1.0, "max": 1.0},
        "max_weight_position": {"min": 2, "max": 2},
        "min_weight": {"min": 0.0, "max": 0.0},
        "min_weight_distance": {"min": 6, "max": 6},
        "components": {"choices": ["both"]},
        **extra,
    }


def _fake_directions(monkeypatch, width=H, rows=N_LAYERS + 1, seed=5):
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(torch.randn(rows, width, generator=g), dim=1)
    monkeypatch.setattr(
        B,
        "get_refusal_directions",
        lambda *a, **k: SimpleNamespace(
            directions=dirs, fingerprint="fp", manifest_path=Path("m.json")
        ),
    )
    return dirs


@pytest.mark.parametrize("slug", [*NEW, "qwen3.5-0.8b"])
def test_abliteration_edits_attn_and_mlp_writers_of_every_layer(slug, monkeypatch):
    state, info = tiny_state(slug)
    dirs = _fake_directions(monkeypatch)
    fold = bool(load(slug)["families"].get("abliteration", {}).get("fold_post_norm"))
    config = _config(slug, info)
    ((desc, new_state, touched),) = B.build_abliteration(
        config, _abl_family(fold_post_norm=fold), 0, state, Path("."), dry_run=False
    )
    comps = Counter(A.component_of(k, info.arch) for k in touched)
    assert set(comps) == {"attn_out", "mlp_out"}
    assert comps["attn_out"] == comps["mlp_out"] == N_LAYERS  # exactly one writer per layer each
    assert all(state[k].shape[0] == H for k in touched)
    expected_variant = "heretic2-classic-projected-rownorm-none" + (
        "+postnorm-fold" if fold else ""
    )
    assert desc.provenance["kernel_variant"] == expected_variant
    # The peak layer (2) is ablated with weight 1 along the interpolated direction.
    key = A.select_keys(["mlp_out"], 2, A.index_state_keys(state), info.arch)[0]
    dir2 = torch.nn.functional.normalize(dirs[3], dim=0)  # direction_index 2.0 -> row 3
    if fold:
        gain = A.effective_gain(
            state[A.post_norm_key(key, info.post_norms["mlp_out"])], info.norm_gain
        )
        dir2 = torch.nn.functional.normalize(gain * dir2, dim=0)
    assert (dir2 @ new_state[key]).abs().max() < 1e-5


def test_fold_without_a_post_norm_is_an_error(monkeypatch):
    state, info = tiny_state("qwen3.5-4b")
    _fake_directions(monkeypatch)
    with pytest.raises(ValueError, match="post-writer norm"):
        B.build_abliteration(
            _config("qwen3.5-4b", info),
            _abl_family(fold_post_norm=True),
            0,
            state,
            Path("."),
            dry_run=False,
        )


def test_refusal_directions_of_the_wrong_width_or_depth_are_rejected(monkeypatch):
    state, info = tiny_state("gemma-3-270m")
    for kw in ({"width": 16}, {"rows": N_LAYERS + 3}):
        _fake_directions(monkeypatch, **kw)
        with pytest.raises(ValueError, match="refusal directions"):
            B.build_abliteration(
                _config("gemma-3-270m", info), _abl_family(), 0, state, Path("."), dry_run=False
            )


def test_08b_legacy_selection_is_unchanged_without_architecture_info():
    """No ``_arch`` (old tests, unreadable model): Qwen3.5 naming, layers as given."""
    state, _ = tiny_state("qwen3.5-0.8b")
    fam = {
        "layers": [0, 3],
        "tensors": ["mlp.down_proj", "self_attn.o_proj"],
        "weight": {"min": 0.5, "max": 0.5},
        "n_directions": {"choices": [1]},
    }
    config = {"base_model": "Qwen/Qwen3.5-0.8B", "base_revision": "r", "seed": 0, "families": {}}
    ((desc, _, touched),) = B.build_random_dir(config, fam, 1, state)
    legacy = []
    for layer in desc.params["layers"]:
        for t in ("mlp.down_proj", "self_attn.o_proj"):
            key = f"model.layers.{layer}.{t}.weight"
            legacy.append(
                key if key in state else key.replace("self_attn.o_proj", "linear_attn.out_proj")
            )
    assert touched == legacy


def test_remote_code_kwargs_reach_the_base_state_loader(monkeypatch):
    import transformers

    seen = {}

    class Model:
        def state_dict(self):
            return {"w": torch.zeros(1)}

    def fake(model, **kw):
        seen.update(kw)
        return Model()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake)
    B._load_base_state({"base_model": "LiquidAI/LFM3-1.7B", "base_revision": "r"})
    assert seen["trust_remote_code"] is True
    seen.clear()
    B._load_base_state({"base_model": "Qwen/Qwen3.5-4B", "base_revision": "r"})
    assert "trust_remote_code" not in seen
