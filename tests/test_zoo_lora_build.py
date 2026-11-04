"""Builder wiring + LoRA loading for the heretic_ara / lora_sft families (CPU, fake generators)."""

from __future__ import annotations

import importlib.machinery
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml
from peft import LoraConfig, get_peft_model

from wsp.schema import EditDescriptor
from wsp.zoo.apply import load_edited_model, lora_dense_delta
from wsp.zoo.build import (
    apply_overrides,
    build_zoo,
    config_hash,
    family_policy,
    sample_param,
)


class _Inner(torch.nn.Module):
    def __init__(self, prefix: str):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.ModuleDict({"mlp": torch.nn.ModuleDict({"down_proj": torch.nn.Linear(8, 8)})})
            for _ in range(2)
        )
        self.prefix = prefix


class Toy(torch.nn.Module):
    """Module tree whose names mimic ``model.layers.N.mlp.down_proj`` (+ optional language_model)."""

    def __init__(self, language_model: bool = False):
        super().__init__()
        self.model = torch.nn.Module()
        holder = self.model
        if language_model:
            self.model.language_model = torch.nn.Module()
            holder = self.model.language_model
        holder.layers = _Inner("").layers


def write_toy_adapter(lora_dir: Path, seed: int, language_model: bool = False, rank: int = 2):
    """Save a real PEFT adapter with non-zero B; return the merged dense deltas keyed by name."""
    torch.manual_seed(seed)
    model = Toy(language_model)
    base = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=["down_proj"], bias="none")
    peft = get_peft_model(model, cfg)
    for n, p in peft.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.1)
    peft.save_pretrained(str(lora_dir))
    merged = peft.merge_and_unload().state_dict()
    return {k: merged[k] - base[k] for k in base if k.endswith("down_proj.weight")}


def test_load_edited_model_applies_lora_entry_like_peft_merge(tmp_path):
    entry = tmp_path / "entry"
    expected = write_toy_adapter(entry / "lora", seed=0)
    (entry / "edit.json").write_text(
        EditDescriptor(
            family="lora_sft", params={"delta_format": "lora"}, seed=0, base_model="toy"
        ).to_json()
    )
    base = {f"model.layers.{i}.mlp.down_proj.weight": torch.randn(8, 8) for i in range(2)}
    out = load_edited_model(base, entry)
    assert set(out) == set(base)
    for k, d in expected.items():
        assert torch.allclose(out[k], base[k] + d, atol=1e-6)
        assert not torch.equal(out[k], base[k])


def test_lora_keys_map_across_language_model_prefix(tmp_path):
    """Heretic loads Qwen3.5 as image-text (model.language_model.*); base state is model.*."""
    expected = write_toy_adapter(tmp_path / "lora", seed=1, language_model=True)
    assert all("language_model" in k for k in expected)
    keys = [f"model.layers.{i}.mlp.down_proj.weight" for i in range(2)]
    delta = lora_dense_delta(tmp_path / "lora", keys)
    assert set(delta) == set(keys)
    for k, d in expected.items():
        assert torch.allclose(delta[k.replace("language_model.", "")], d, atol=1e-6)
    with pytest.raises(KeyError, match="no matching tensor"):
        lora_dense_delta(tmp_path / "lora", ["unrelated.weight"])


# ---- builder wiring with fake generate() modules ----------------------------------------------


CALLS: list[dict] = []


def _fake_family_module(family: str) -> types.ModuleType:
    def generate(base_model, params, seed, out_dir):
        CALLS.append({"family": family, "base_model": base_model, "params": params, "seed": seed})
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_toy_adapter(out_dir / "lora", seed)
        (out_dir / "study.json").write_text("{}")
        desc = EditDescriptor(
            family=family,
            params={**params, "delta_format": "lora", "git_dirty": False, "git_tree": "t"},
            seed=seed,
            base_model=base_model,
            base_revision="a" * 40,
            git_sha="gen",
            touched_tensors=["model.language_model.layers.0.mlp.down_proj.weight"],
            delta_fro_norm=1.5,
        )
        (out_dir / "edit.json").write_text(desc.to_json())
        return desc

    mod = types.ModuleType(f"wsp.zoo.{family}")
    mod.__spec__ = importlib.machinery.ModuleSpec(mod.__name__, None)
    mod.generate = generate
    return mod


@pytest.fixture
def fake_generators(monkeypatch):
    CALLS.clear()
    for fam in ("heretic_ara", "lora_sft"):
        monkeypatch.setitem(sys.modules, f"wsp.zoo.{fam}", _fake_family_module(fam))
    return CALLS


@pytest.fixture
def toy_base(monkeypatch):
    import wsp.zoo.build as B

    g = torch.Generator().manual_seed(0)
    state = {
        f"model.layers.{i}.mlp.down_proj.weight": torch.randn(8, 8, generator=g) for i in range(2)
    }
    loads = []

    def load(config):
        loads.append(1)
        return {k: v.clone() for k, v in state.items()}

    monkeypatch.setattr(B, "_load_base_state", load)
    return loads


FAMILIES = {
    "heretic_ara": {
        "n": 2,
        "preserve_good_behavior_weight": {"choices": [0.5, 1, 2, 4]},
        "trials": 40,
    },
    "lora_sft": {
        "n": 2,
        "datasets": ["gsm8k_train", "alpaca_cleaned_5k"],
        "rank": {"choices": [4, 8, 16, 32]},
        "lr": {"log_min": 2.0e-5, "log_max": 5.0e-4},
        "steps": {"choices": [50, 150, 400, 800]},
    },
    "random_dir": {
        "n": 1,
        "layers": [0, 1],
        "tensors": ["mlp.down_proj"],
        "weight": {"min": 0.25, "max": 1.0},
        "n_directions": {"choices": [1]},
    },
    "task_vector": {"n": 4, "scale": {"min": 0.5, "max": 1.0}},
    "chain": {"n": 4, "length": {"min": 2, "max": 4}},
}


def _cfg(tmp_path, families):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "base_model": "Qwen/Qwen3.5-0.8B",
                "dtype": "float32",
                "out_dir": str(tmp_path / "zoo"),
                "seed": 3,
                "families": families,
            }
        )
    )
    return p


def _rows(zoo: Path):
    lines = [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines()]
    return [x for x in lines if x.get("type") != "header"]


def test_real_config_has_every_family_available():
    cfg = yaml.safe_load(Path("configs/zoo_qwen3.5-0.8b.yaml").read_text())
    assert family_policy(cfg)["skipped_families"] == []


def test_lora_family_entries_are_generated_with_sampled_params(tmp_path, fake_generators, toy_base):
    cfg = _cfg(tmp_path, {k: FAMILIES[k] for k in ("heretic_ara", "lora_sft")})
    build_zoo(cfg, git_sha="s", overrides=["heretic_ara.trials=2"])
    assert not toy_base  # LoRA-only builds never load the CPU base copy
    zoo = tmp_path / "zoo"
    rows = _rows(zoo)
    assert len(rows) == 4 and all(r["delta_format"] == "lora" for r in rows)
    by_family = {f: [c for c in fake_generators if c["family"] == f] for f in FAMILIES}
    assert len(by_family["heretic_ara"]) == 2 and len(by_family["lora_sft"]) == 2
    for c in by_family["heretic_ara"]:
        assert c["params"]["trials"] == 2  # override, not the 40 in the file
        assert c["params"]["preserve_good_behavior_weight"] in (0.5, 1, 2, 4)
        assert c["base_model"] == "Qwen/Qwen3.5-0.8B" and c["params"]["base_revision"] == "main"
    for c in by_family["lora_sft"]:
        p = c["params"]
        assert p["dataset"] in ("gsm8k_train", "alpaca_cleaned_5k")
        assert p["rank"] in (4, 8, 16, 32) and p["steps"] in (50, 150, 400, 800)
        assert 2.0e-5 <= p["lr"] <= 5.0e-4
    for r in rows:
        d = zoo / r["id"]
        edit = json.loads((d / "edit.json").read_text())
        assert edit["id"] == r["id"] and edit["git_sha"] == "s"
        assert (d / "lora" / "adapter_model.safetensors").exists()
        assert (d / "generator_edit.json").exists() and not (d / "delta.safetensors").exists()
        assert edit["params"]["delta_format"] == "lora"
        assert edit["provenance"]["config_overrides"] == ["heretic_ara.trials=2"]
        assert edit["provenance"]["resolved_base_revision"] == "a" * 40
        assert edit["delta_fro_norm"] == 1.5 and edit["touched_tensors"]
    assert not list((zoo / ".staging").iterdir())

    # Idempotent: a second build reuses finished entries and does not regenerate them.
    n_calls = len(fake_generators)
    assert [r["id"] for r in build_zoo(cfg, git_sha="s", overrides=["heretic_ara.trials=2"])] == [
        r["id"] for r in rows
    ]
    assert len(fake_generators) == n_calls


def test_dry_run_ids_match_real_ids_and_overrides_change_identity(
    tmp_path, fake_generators, toy_base
):
    cfg = _cfg(tmp_path, {k: FAMILIES[k] for k in ("heretic_ara", "lora_sft")})
    dry = {r["id"] for r in build_zoo(cfg, dry_run=True, git_sha="s")}
    assert not fake_generators
    real = {r["id"] for r in build_zoo(cfg, git_sha="s")}
    assert dry == real
    assert config_hash(cfg) != config_hash(cfg, ["heretic_ara.trials=2"])


def test_family_flag_runs_one_lora_family(tmp_path, fake_generators, toy_base):
    cfg = _cfg(tmp_path, FAMILIES)
    rows = build_zoo(cfg, family="lora_sft", n=1, overrides=["lora_sft.steps=5"], git_sha="s")
    assert [r["family"] for r in rows] == ["lora_sft"]
    assert fake_generators[0]["params"]["steps"] == 5


def test_task_vector_and_chain_use_lora_entries(tmp_path, fake_generators, toy_base):
    cfg = _cfg(tmp_path, FAMILIES)
    build_zoo(cfg, git_sha="s")
    zoo = tmp_path / "zoo"
    rows = _rows(zoo)
    fams = {r["family"] for r in rows}
    assert {"heretic_ara", "lora_sft", "task_vector", "chain"} <= fams
    lora_ids = {r["id"] for r in rows if r.get("delta_format") == "lora"}
    # Chains sample the LoRA families too, and every dense entry is well-formed.
    inner = {
        json.loads((zoo / r["id"] / "edit.json").read_text())["params"]["inner_family"]
        for r in rows
        if r["family"] == "chain"
    }
    assert inner & {"heretic_ara", "lora_sft"}
    for r in rows:
        if r["family"] in ("task_vector", "chain"):
            assert (zoo / r["id"] / "delta.safetensors").exists()
    tv_parents = {r["parent_id"] for r in rows if r["family"] == "task_vector"}
    assert tv_parents & lora_ids  # at least one task vector densified a LoRA parent


def test_apply_overrides_and_scalar_sample():
    cfg = {"families": {"lora_sft": {"steps": {"choices": [1]}}}}
    apply_overrides(cfg, ["lora_sft.steps=5", "families.lora_sft.batch_size=1"])
    assert cfg["families"]["lora_sft"] == {"steps": 5, "batch_size": 1}
    assert sample_param(__import__("random").Random(0), 5) == 5
    for bad in ("nokey", "heretic_ara.trials=2", "seed.x=1"):
        with pytest.raises((ValueError, TypeError)):
            apply_overrides(cfg, [bad])
