"""Diagnostic curation preserves identities, scores and deltas."""

import importlib.util
import json
from pathlib import Path


def test_flagging_is_idempotent_and_preserves_artifacts(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "flag_ara", Path("scripts/flag_ara_unconstrained.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = tmp_path / "abc"
    entry.mkdir()
    edit = {
        "id": "abc",
        "family": "heretic_ara",
        "eligibility": "production",
        "params": {"old": True},
    }
    (entry / "edit.json").write_text(json.dumps(edit))
    (entry / "delta.safetensors").write_bytes(b"preserve")
    (entry / "study.json").write_text('{"verified_kl": 10}')
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"type": "header"}) + "\n" + json.dumps(edit) + "\n")
    module.flag_entries(tmp_path, ["abc"])
    first = (manifest.read_bytes(), (entry / "edit.json").read_bytes())
    module.flag_entries(tmp_path, ["abc"])
    assert first == (manifest.read_bytes(), (entry / "edit.json").read_bytes())
    for updated in [json.loads(first[1]), json.loads(manifest.read_text().splitlines()[1])]:
        assert updated["eligibility"] == "smoke"
        assert updated["provenance"]["diagnostic_flag"] == "ara_unconstrained_surrogate_scale"
        assert updated["id"] == "abc" and updated["params"] == edit["params"]
    assert (entry / "delta.safetensors").read_bytes() == b"preserve"
    assert (entry / "study.json").read_text() == '{"verified_kl": 10}'
