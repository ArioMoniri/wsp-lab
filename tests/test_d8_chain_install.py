"""scripts/d8_chain_install.py: accepts only verified regenerated entries and updates the sidecar."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "d8_chain_install.py"


def _setup(tmp_path: Path):
    zoo, staging = tmp_path / "zoo", tmp_path / "staging"
    for i in ("good", "badfro", "stored"):
        (zoo / i).mkdir(parents=True)
        (zoo / i / "edit.json").write_text("{}")
    for i in ("good", "badfro"):
        (staging / i).mkdir(parents=True)
        save_file({"w": torch.zeros(2, 2, dtype=torch.bfloat16)}, staging / i / "delta.safetensors")
        (staging / i / "regen.json").write_text(json.dumps({"written_utc": "t"}))
    sidecar = tmp_path / "artefact_status.jsonl"
    sidecar.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in (
                {"id": "good", "delta_present": False, "artefact_lost": True, "alias_of": None},
                {"id": "badfro", "delta_present": False, "artefact_lost": True, "alias_of": None},
                {"id": "stored", "delta_present": True, "artefact_lost": False, "alias_of": None},
            )
        )
    )
    report = {
        "environment": {"git_sha": "abc"},
        "chains": [
            {
                "root": "stored",
                "steps": [
                    {"id": "stored", "stored": True, "compare": {"bitwise": True}},
                    {
                        "id": "good",
                        "stored": False,
                        "touched_equal_edit_json": True,
                        "fro_edit_json": 1.0,
                        "fro_regenerated": 1.0,
                        "exact_lineage": True,
                        "regenerated_from_anchor": None,
                    },
                    {
                        "id": "badfro",
                        "stored": False,
                        "touched_equal_edit_json": True,
                        "fro_edit_json": 1.0,
                        "fro_regenerated": 1.5,
                        "exact_lineage": False,
                    },
                ],
            }
        ],
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    return zoo, staging, sidecar


def test_install_moves_verified_entries_and_updates_sidecar(tmp_path):
    zoo, staging, sidecar = _setup(tmp_path)
    copy = tmp_path / "copy.jsonl"
    out = tmp_path / "install.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--zoo",
            str(zoo),
            "--staging",
            str(staging),
            "--report",
            str(tmp_path / "report.json"),
            "--sidecar",
            str(sidecar),
            "--sidecar-copy",
            str(copy),
            "--out",
            str(out),
        ],
        check=True,
    )
    summary = json.loads(out.read_text())
    assert summary["installed"] == 1 and summary["installed_ids"] == ["good"]
    assert [r["id"] for r in summary["rejected"]] == ["badfro"]
    assert (zoo / "good" / "delta.safetensors").exists() and (zoo / "good" / "regen.json").exists()
    assert not (zoo / "badfro" / "delta.safetensors").exists()
    rows = {json.loads(l)["id"]: json.loads(l) for l in sidecar.read_text().splitlines()}
    assert rows["good"]["delta_present"] and not rows["good"]["artefact_lost"]
    assert rows["good"]["regenerated"]["method"] == "d8_chain_replay"
    assert rows["badfro"]["artefact_lost"]
    assert copy.read_text() == sidecar.read_text()
    assert summary["sidecar"] == {"stored": 2, "aliased": 0, "lost": 1}
