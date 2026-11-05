import copy
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model

from wsp.zoo.common import compute_adapter_stats, resolve_base_revision


def test_touched_tensors_match_unwrapped_state_dict(toy_model):
    keys = set(toy_model.state_dict())
    wrapped = get_peft_model(copy.deepcopy(toy_model), LoraConfig(r=2, target_modules=["o_proj"]))
    for name, p in wrapped.named_parameters():
        if "lora_B" in name:
            torch.nn.init.ones_(p)
    touched, norm = compute_adapter_stats(wrapped)
    assert touched and set(touched) <= keys
    assert norm > 0


def test_revision_is_snapshot_commit_not_requested_ref(monkeypatch, tmp_path):
    from wsp.zoo import common

    sha = "a" * 40
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshots" / sha)

    monkeypatch.setattr(common, "snapshot_download", download)
    assert resolve_base_revision("repo", "main") == sha
    assert calls[0]["revision"] == "main"
    monkeypatch.setattr(common, "snapshot_download", lambda **kw: str(Path("/cache/main")))
    with pytest.raises(ValueError, match="snapshot"):
        resolve_base_revision("repo")


def test_sft_releases_cache_on_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from wsp.zoo import lora_sft

    monkeypatch.setattr(lora_sft, "resolve_base_revision", lambda *a: "a" * 40)
    monkeypatch.setattr(
        lora_sft.AutoTokenizer, "from_pretrained", lambda *a, **kw: SimpleNamespace(pad_token="pad")
    )

    def fail(*a, **kw):
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        lora_sft, "get_model_class", lambda _: SimpleNamespace(from_pretrained=fail)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    released = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(True))
    with pytest.raises(RuntimeError, match="load failed"):
        lora_sft.generate("repo", {}, 0, tmp_path)
    assert released


def test_smoke_provenance_check_rejects_stale_sha(tmp_path):
    import json
    import runpy
    import subprocess

    check = runpy.run_path("scripts/check_generator_provenance.py")["check"]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for family in ("heretic_ara", "lora_sft"):
        entry = tmp_path / family
        entry.mkdir()
        (entry / "edit.json").write_text(
            json.dumps({"git_sha": sha, "params": {"git_dirty": False}, "base_revision": "a" * 40})
        )
        (entry / "manifest.json").write_text(json.dumps({"git_sha": sha}))
    check(tmp_path)
    (tmp_path / "lora_sft/manifest.json").write_text(json.dumps({"git_sha": "stale"}))
    with pytest.raises(AssertionError):
        check(tmp_path)


# --- delta key normalisation (D6 addendum 5) ---------------------------------------------------

WRAPPED = "model.language_model.layers.0.mlp.down_proj.weight"
TEXT = "model.layers.0.mlp.down_proj.weight"


def _state(keys, shape=(2, 3)):
    return {k: torch.zeros(*shape) for k in keys}


def test_delta_keys_saved_under_multimodal_prefix_map_onto_text_state():
    from wsp.zoo.common import DELTA_KEY_MAPPING, map_delta_keys

    delta = {WRAPPED: torch.ones(2, 3)}
    mapped = map_delta_keys(delta, _state([TEXT, "lm_head.weight"]))
    assert list(mapped) == [TEXT]
    assert torch.equal(mapped[TEXT], delta[WRAPPED])
    assert mapped.key_mapping == DELTA_KEY_MAPPING == "language_model_prefix_v1"


def test_delta_keys_in_text_naming_map_onto_multimodal_state():
    from wsp.zoo.common import map_delta_keys

    mapped = map_delta_keys({TEXT: torch.ones(2, 3)}, _state([WRAPPED]))
    assert list(mapped) == [WRAPPED] and mapped.key_mapping == "language_model_prefix_v1"


def test_delta_keys_that_already_match_are_untouched_and_unlabelled():
    from wsp.zoo.common import map_delta_keys

    delta = {TEXT: torch.ones(2, 3), "lm_head.weight": torch.ones(2, 3)}
    mapped = map_delta_keys(delta, _state([TEXT, "lm_head.weight"]))
    assert list(mapped) == list(delta) and mapped.key_mapping is None


def test_delta_key_mapping_is_deterministic_and_keeps_order():
    from wsp.zoo.common import map_delta_keys

    names = [f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in (3, 1, 2)]
    state = _state([n.replace("language_model.", "") for n in names])
    a = map_delta_keys({n: torch.ones(2, 3) for n in names}, state)
    b = map_delta_keys({n: torch.ones(2, 3) for n in names}, state)
    assert list(a) == list(b) == [n.replace("language_model.", "") for n in names]


def test_unmatched_delta_tensor_fails_loudly_naming_it():
    from wsp.zoo.common import map_delta_keys

    delta = {WRAPPED: torch.ones(2, 3), "model.language_model.layers.9.nope.weight": torch.ones(1)}
    with pytest.raises(KeyError, match=r"layers\.9\.nope"):
        map_delta_keys(delta, _state([TEXT]))
    with pytest.raises(KeyError, match="not in the base state"):
        map_delta_keys({"model.visual.x.weight": torch.ones(1)}, _state([TEXT]))


def test_shape_mismatch_after_mapping_fails_loudly():
    from wsp.zoo.common import map_delta_keys

    with pytest.raises(ValueError, match="shape"):
        map_delta_keys({WRAPPED: torch.ones(3, 2)}, _state([TEXT], shape=(2, 3)))


def test_two_saved_names_collapsing_onto_one_base_key_fail_loudly():
    from wsp.zoo.common import map_delta_keys

    delta = {WRAPPED: torch.ones(2, 3), TEXT: torch.ones(2, 3)}
    with pytest.raises(ValueError, match="same base tensor"):
        map_delta_keys(delta, _state([TEXT]))


def test_plain_key_iterable_skips_the_shape_check_but_still_maps():
    from wsp.zoo.common import map_delta_keys

    mapped = map_delta_keys({WRAPPED: torch.ones(9)}, [TEXT])
    assert list(mapped) == [TEXT]


def test_export_names_strip_the_multimodal_wrapper_and_reject_collisions():
    from wsp.zoo.common import to_text_model_keys

    out = to_text_model_keys({WRAPPED: torch.ones(1), "lm_head.weight": torch.zeros(1)})
    assert list(out) == [TEXT, "lm_head.weight"]
    with pytest.raises(ValueError, match="collide"):
        to_text_model_keys({WRAPPED: torch.ones(1), TEXT: torch.ones(1)})
