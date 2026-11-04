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
