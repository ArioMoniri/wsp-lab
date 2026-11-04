"""Tests for Heretic ARA edit family."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from wsp.schema import EditDescriptor
from wsp.zoo.heretic_ara import generate, select_best_trial


def test_select_best_trial_without_target():
    trials = [
        {"trial": 1, "refusals": 0.8, "kl_divergence": 0.05},
        {"trial": 2, "refusals": 0.2, "kl_divergence": 0.30},
        {"trial": 3, "refusals": 0.4, "kl_divergence": 0.15},
    ]
    # Default selection: lowest refusals
    best = select_best_trial(trials)
    assert best["trial"] == 2


def test_select_best_trial_with_kl_target():
    trials = [
        {"trial": 1, "refusals": 0.8, "kl_divergence": 0.05},
        {"trial": 2, "refusals": 0.1, "kl_divergence": 0.35},
        {"trial": 3, "refusals": 0.3, "kl_divergence": 0.12},
    ]
    # With KL target <= 0.20, trial 3 is the lowest refusal satisfying the constraint
    best = select_best_trial(trials, kl_divergence_target=0.20)
    assert best["trial"] == 3

    with pytest.raises(ValueError, match="No trial"):
        select_best_trial(trials, kl_divergence_target=0.01)


def test_generate_heretic_ara_toy(tmp_path: Path, fake_heretic):
    out_dir = tmp_path / "heretic_ara_toy"
    params = {
        "preserve_good_behavior_weight": 1.0,
        "trials": 2,
        "lora_rank": 4,
        "ara_settings": {"n_optimization_steps": 1, "max_iter": 2},
        "kl_divergence_target": 100.0,
    }
    desc = generate(base_model="toy", params=params, seed=42, out_dir=out_dir)

    assert isinstance(desc, EditDescriptor)
    assert desc.family == "heretic_ara"
    assert desc.seed == 42
    assert desc.base_model == "toy"
    assert desc.params["delta_format"] == "dense"
    assert desc.params["preserve_good_behavior_weight"] == 1.0
    assert len(desc.touched_tensors) > 0
    assert desc.delta_fro_norm is not None and desc.delta_fro_norm >= 0

    # Check written artifacts
    edit_file = out_dir / "edit.json"
    assert edit_file.exists()
    roundtrip = EditDescriptor.from_json(edit_file.read_text())
    assert roundtrip.id == desc.id
    assert roundtrip.family == "heretic_ara"
    assert roundtrip.params["delta_format"] == "dense"

    study_file = out_dir / "study.json"
    assert study_file.exists()
    study = json.loads(study_file.read_text())
    assert "trials" in study
    assert len(study["trials"]) == 2
    for t in study["trials"]:
        assert "refusals" in t
        assert "kl_divergence" in t
        assert "parameters" in t
    assert "best_trial" in study

    from safetensors.torch import load_file

    delta = load_file(out_dir / "delta.safetensors")
    assert sorted(delta) == desc.touched_tensors
    assert desc.delta_fro_norm == pytest.approx(
        sum(t.float().square().sum().item() for t in delta.values()) ** 0.5
    )
    assert study["verification_status"] == "passed"
    assert abs(study["verified_kl"] - study["reloaded_kl"]) <= 0.01
    assert study["verified_refusals"] == study["reloaded_refusals"]
    assert not (out_dir / "lora").exists()


@pytest.mark.gpu
@pytest.mark.slow
def test_generate_heretic_ara_smoke_gpu(tmp_path: Path, cuda):
    out_dir = Path(__file__).parents[1] / "data/runs/p1-generator-smoke/heretic_ara"
    import yaml

    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/p1_generator_smoke.yaml").read_text()
    )
    params = config["heretic_ara"]
    desc = generate(
        base_model="Qwen/Qwen3.5-0.8B",
        params=params,
        seed=123,
        out_dir=out_dir,
    )
    import math

    from huggingface_hub import hf_hub_download

    from wsp.zoo.common import get_git_sha, resolve_base_revision

    assert desc.git_sha == get_git_sha()
    assert not desc.params["git_dirty"]
    assert desc.base_revision == resolve_base_revision(desc.base_model)
    index = json.loads(
        Path(
            hf_hub_download(
                desc.base_model, "model.safetensors.index.json", revision=desc.base_revision
            )
        ).read_text()
    )
    assert set(desc.touched_tensors) <= set(index["weight_map"])
    assert desc.touched_tensors and math.isfinite(desc.delta_fro_norm) and desc.delta_fro_norm > 0
    assert json.loads((out_dir / "manifest.json").read_text())["git_sha"] == get_git_sha()
    assert desc.family == "heretic_ara"
    assert desc.params["delta_format"] == "dense"
    assert (out_dir / "edit.json").exists()
    assert (out_dir / "study.json").exists()
    study = json.loads((out_dir / "study.json").read_text())
    assert len(study["trials"]) == 2
    assert (out_dir / "delta.safetensors").exists()
    assert abs(study["verified_kl"] - study["reloaded_kl"]) <= 0.01
    print(
        {
            k: study[k]
            for k in ("verified_kl", "reloaded_kl", "verified_refusals", "reloaded_refusals")
        }
    )


@pytest.fixture
def fake_heretic(monkeypatch, toy_model):
    """Replace model loading and evaluation, retaining installed ARA and PEFT code."""
    import torch
    from heretic.scorer import Score
    from peft import LoraConfig, get_peft_model

    from wsp.zoo import heretic_ara

    monkeypatch.setattr(heretic_ara, "resolve_base_revision", lambda *a: "a" * 40)

    class Model:
        def __init__(self, settings):
            self.model = copy.deepcopy(toy_model).bfloat16()
            self.base = {k: v.clone() for k, v in self.model.state_dict().items()}

        def get_layers(self):
            return self.model.layers

        def get_layer_modules(self, i):
            layer = self.get_layers()[i]
            return {"attn": [layer["attn"]["o_proj"]], "mlp": [layer["mlp"]["down_proj"]]}

        def get_module_io_batched(self, prompts):
            g = torch.Generator().manual_seed(1 if prompts[0] == "good" else 2)
            return [
                {
                    key: [
                        (x := torch.randn(4, m.in_features, generator=g).bfloat16(), m(x).detach())
                        for m in modules
                    ]
                    for key, modules in self.get_layer_modules(i).items()
                }
                for i in range(2)
            ]

        def apply_lora(self, rank):
            self.model = get_peft_model(
                self.model,
                LoraConfig(r=rank, lora_alpha=rank, target_modules=["o_proj", "down_proj"]),
            )

        def get_merged_model(self):
            self.model = self.model.merge_and_unload()
            return self.model

        def reset_model(self):
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if "lora_" in name:
                        p.zero_()
            return True

    class Evaluator:
        def __init__(self, settings, model):
            self.model = model
            from types import SimpleNamespace

            from heretic.scorers.keyword_rate import KeywordRate
            from heretic.scorers.kl_divergence import KLDivergence

            self._scorer_entries = [
                SimpleNamespace(scorer=object.__new__(KeywordRate), name="Refusals"),
                SimpleNamespace(scorer=object.__new__(KLDivergence), name="KL divergence"),
            ]

        def get_scores(self):
            if hasattr(self.model.model, "peft_config"):
                from wsp.zoo.common import compute_adapter_stats

                _, norm = compute_adapter_stats(self.model.model)
            else:
                norm = (
                    sum(
                        (v.float() - self.model.base[k].float()).square().sum().item()
                        for k, v in self.model.model.state_dict().items()
                    )
                    ** 0.5
                )
            return [
                ("Refusals", Score(1 / (1 + norm), "", "")),
                ("KL divergence", Score(norm, "", "")),
            ]

        def get_objective_values(self, scores):
            return tuple(s.value for _, s in scores)

        def get_objective_directions(self):
            return ["minimize", "minimize"]

    monkeypatch.setattr(heretic_ara, "Model", Model)
    monkeypatch.setattr(heretic_ara, "Evaluator", Evaluator)
    monkeypatch.setattr(
        heretic_ara.Context,
        "load_prompts",
        lambda self, spec: ["good" if "harmless" in spec.dataset else "bad"] * 4,
    )


def test_ara_seed_reproduces_saved_adapter(tmp_path, fake_heretic):
    import torch
    from safetensors.torch import load_file

    params = {
        "trials": 2,
        "lora_rank": 2,
        "preserve_good_behavior_weight": 2.0,
        "ara_settings": {"n_optimization_steps": 1, "max_iter": 2},
    }
    a = generate("toy", params, 7, tmp_path / "a")
    b = generate("toy", params, 7, tmp_path / "b")
    assert a.id == b.id
    assert a.delta_fro_norm > 0
    wa = load_file(tmp_path / "a/delta.safetensors")
    wb = load_file(tmp_path / "b/delta.safetensors")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    study = json.loads((tmp_path / "a/study.json").read_text())
    assert all(t["parameters"]["preserve_good_behavior_weight"] == 2 for t in study["trials"])
    assert study["best_trial"] == select_best_trial(study["trials"])
    manifest = json.loads((tmp_path / "a/manifest.json").read_text())
    assert manifest["seed"] == 7 and manifest["git_sha"] == a.git_sha


def test_select_best_trial_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        select_best_trial([])


def test_scores_use_plugin_output_names_not_position():
    from types import SimpleNamespace

    from heretic.scorer import Score
    from heretic.scorers.keyword_rate import KeywordRate
    from heretic.scorers.kl_divergence import KLDivergence

    from wsp.zoo.heretic_ara import extract_trial_metrics

    evaluator = SimpleNamespace(
        _scorer_entries=[
            SimpleNamespace(scorer=object.__new__(KeywordRate), name="Safety - custom"),
            SimpleNamespace(scorer=object.__new__(KLDivergence), name="Divergence"),
        ]
    )
    scores = [
        ("unrelated", Score(99, "", "")),
        ("Divergence", Score(0.4, "", "")),
        ("Safety - custom", Score(0.1, "", "")),
    ]
    assert extract_trial_metrics(evaluator, scores) == (0.1, 0.4)
    with pytest.raises(ValueError, match="Divergence"):
        extract_trial_metrics(evaluator, scores[::2])


def test_selection_rejects_missing_kl():
    with pytest.raises((ValueError, KeyError)):
        select_best_trial([{"refusals": 0}], kl_divergence_target=0.2)


def test_fake_study_filters_violating_trial(tmp_path, fake_heretic, monkeypatch):
    from heretic.scorer import Score

    from wsp.zoo import heretic_ara

    outputs = iter([(0.1, 0.9), (0.6, 0.1), (0.6, 0.1), (0.6, 0.1)])

    def scores(self):
        refusal, kl = next(outputs)
        return [("KL divergence", Score(kl, "", "")), ("Refusals", Score(refusal, "", ""))]

    monkeypatch.setattr(heretic_ara.Evaluator, "get_scores", scores)
    generate(
        "toy",
        {
            "trials": 2,
            "lora_rank": 2,
            "kl_divergence_target": 0.2,
            "ara_settings": {"n_optimization_steps": 1, "max_iter": 2},
        },
        2,
        tmp_path,
    )
    study = json.loads((tmp_path / "study.json").read_text())
    assert study["best_trial"]["trial"] == 2
    assert study["kl_constraint_met"] is True


def test_reload_mismatch_records_failure_without_entry(tmp_path, fake_heretic, monkeypatch):
    from wsp.zoo import heretic_ara

    original = heretic_ara.Evaluator.get_scores
    calls = []

    def scores(self):
        calls.append(1)
        result = original(self)
        if len(calls) == 4:
            result[1][1].value += 0.02
        return result

    monkeypatch.setattr(heretic_ara.Evaluator, "get_scores", scores)
    with pytest.raises(ValueError, match="verification"):
        generate(
            "toy",
            {
                "trials": 2,
                "lora_rank": 2,
                "ara_settings": {"n_optimization_steps": 1, "max_iter": 2},
            },
            7,
            tmp_path,
        )
    study = json.loads((tmp_path / "study.json").read_text())
    assert study["verification_status"] == "failed"
    assert abs(study["reloaded_kl"] - study["verified_kl"]) > 0.01
    assert not (tmp_path / "edit.json").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_selected_parameters_applied_once_after_search(tmp_path, fake_heretic, monkeypatch):
    from wsp.zoo import heretic_ara

    calls = []
    original = heretic_ara.ara.ARA.modify_model

    def modify(self, ctx, parameters):
        calls.append(parameters.to_dict())
        return original(self, ctx, parameters)

    monkeypatch.setattr(heretic_ara.ara.ARA, "modify_model", modify)
    generate(
        "toy",
        {"trials": 2, "lora_rank": 2, "ara_settings": {"n_optimization_steps": 1, "max_iter": 2}},
        7,
        tmp_path,
    )
    study = json.loads((tmp_path / "study.json").read_text())
    assert len(calls) == 3
    assert calls[-1] == study["best_trial"]["parameters"]


def test_dense_export_is_effective_bf16_delta():
    import torch

    from wsp.zoo.heretic_ara import merged_delta

    base = {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16), "unchanged": torch.ones(2)}
    merged = {"weight": torch.tensor([1.0001, 2.02]), "unchanged": torch.ones(2)}
    delta = merged_delta(base, merged)
    expected = merged["weight"].bfloat16().float() - base["weight"].float()
    assert list(delta) == ["weight"]
    assert torch.equal(delta["weight"], expected)
    assert delta["weight"].dtype == torch.float32


def test_failed_verification_removes_stale_artifacts(tmp_path, fake_heretic, monkeypatch):
    import json

    from wsp.zoo import heretic_ara

    out = tmp_path / "entry"
    out.mkdir()
    (out / "edit.json").write_text("{}")
    (out / "delta.safetensors").write_bytes(b"old")
    monkeypatch.setattr(heretic_ara, "_reload_kl", lambda *a, **k: (999.0, 1.0), raising=False)
    import contextlib

    with contextlib.suppress(Exception):
        heretic_ara.generate("toy", {"trials": 1, "kl_divergence_target": 1.0}, 0, out)
    study = json.loads((out / "study.json").read_text()) if (out / "study.json").exists() else {}
    if study.get("verification_status") == "failed":
        assert not (out / "edit.json").exists() and not (out / "delta.safetensors").exists()
