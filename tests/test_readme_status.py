"""Tests for scripts/update_readme_status.py using a tmp fixture tree (synthetic inputs, never written
outside tmp_path)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_readme_status.py"


def _load():
    spec = importlib.util.spec_from_file_location("update_readme_status", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _fixture(root: Path) -> None:
    _jl(
        root / "data/zoo/qwen3.5-0.8b/manifest.jsonl",
        [
            {"type": "header", "seed": 0, "config_hash": "x"},
            {"id": "a", "family": "random_dir", "eligibility": "production"},
            {"id": "b", "family": "random_dir", "eligibility": "production"},
            {"id": "b", "family": "random_dir", "eligibility": "production"},  # duplicate row
            {"id": "c", "family": "lora_sft", "eligibility": "smoke"},
            {"id": "d", "family": "chain"},  # legacy, no eligibility
        ],
    )
    feat = root / "data/features/qwen3.5-0.8b"
    for name in ("a", "b"):
        (feat / name).mkdir(parents=True)
        (feat / name / "features.json").write_text("{}")
    (feat / "c").mkdir()  # directory without features.json does not count
    (feat / "manifest.json").write_text(json.dumps({"complete_run": False}))
    _jl(
        root / "data/evals/qwen3.5-0.8b/index.jsonl",
        [
            {"edit_id": "a", "config_hash": "h1", "eval_status": "complete"},
            {"edit_id": "b", "config_hash": "h1", "eval_status": "oom"},
            {"edit_id": "a", "config_hash": "h2", "eval_status": "complete"},
        ],
    )
    run = root / "data/runs/p2/base-final-4096"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "gsm8k": 0.25,
                "math500": 0.5,
                "ifeval": 0.75,
                "mmlu_pro": 0.125,
                "hellaswag": 0.5,
                "loop_rate": 0.5,
                "refusal_rate": 1.0,
                "kl_to_base": 0.0,
                "generation": {"max_new_tokens": 4096},
                "counts": {"gsm8k": 8},
            }
        )
    )
    (root / "board").mkdir()
    (root / "board/lambda_spend.md").write_text(
        "# ledger\n| start (UTC) | end | instance | type | $/h | hours | $ | job |\n|---|---|---|---|---|---|---|---|\n"
        "| 2026-11-04T06:00Z | | wsp-lab-1 | a100 | 1.99 | | | ara |\n"
        "| 2026-11-04T01:00Z | 2026-11-04T02:00Z | wsp-lab-0 | a100 | 1.99 | 1 | 1.99 | old |\n"
    )
    (root / "README.md").write_text(
        "intro\n<!-- status:start -->\nSTALE\n<!-- status:end -->\noutro\n"
    )


def test_collect_counts(tmp_path):
    mod = _load()
    _fixture(tmp_path)
    s = mod.collect(tmp_path, ci=None)
    assert s["zoo"]["total"] == 4
    assert s["zoo"]["by_family"]["random_dir"] == {"production": 2}
    assert s["zoo"]["by_family"]["lora_sft"] == {"smoke": 1}
    assert s["zoo"]["by_family"]["chain"] == {"unlabelled": 1}
    assert s["features"]["entries"] == 2
    assert s["features"]["complete_run"] is False
    assert s["evals"]["rows"] == 3 and s["evals"]["entries"] == 2
    assert s["evals"]["complete_entries"] == 1
    assert s["base"]["gsm8k"] == 0.25
    assert len(s["lambda_open"]) == 1 and s["lambda_open"][0]["instance"] == "wsp-lab-1"


def test_render_and_update_idempotent(tmp_path):
    mod = _load()
    _fixture(tmp_path)
    fixed = "2026-11-04T07:00:00Z"
    mod.update(tmp_path, ci=None, now=fixed)
    first = (tmp_path / "README.md").read_text()
    assert "STALE" not in first
    assert first.startswith("intro\n") and first.endswith("outro\n")
    assert "25.0" in first and "random_dir" in first and fixed in first
    assert "CI" not in first  # omitted without a status
    mod.update(tmp_path, ci=None, now=fixed)
    assert (tmp_path / "README.md").read_text() == first
    assert first.count("<!-- status:start -->") == 1


def test_missing_artefacts_do_not_crash(tmp_path):
    mod = _load()
    (tmp_path / "README.md").write_text("<!-- status:start -->\n<!-- status:end -->\n")
    mod.update(tmp_path, ci=None, now="t")
    text = (tmp_path / "README.md").read_text()
    assert "not found" in text or "none" in text.lower()


def test_ci_line_when_present(tmp_path):
    mod = _load()
    _fixture(tmp_path)
    mod.update(tmp_path, ci={"status": "success", "id": 7, "sha": "abcdef1234"}, now="t")
    assert "success" in (tmp_path / "README.md").read_text()
