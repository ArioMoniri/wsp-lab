import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("backfill", Path("scripts/backfill_eval_metadata.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_backfill_preserves_scores_and_is_idempotent(tmp_path):
    p = tmp_path / "eval.json"
    rows = [{"benchmark": "a", "item_id": "2"}, {"benchmark": "a", "item_id": "1"}]
    (tmp_path / "records.jsonl").write_text("\n".join(map(json.dumps, rows)))
    original = {
        **dict.fromkeys(m.REFERENCE_KEYS, "known"),
        "gsm8k": 0.25,
        "generation": {"batch_size": 32},
        "batch_size": 16,
    }
    runtime = dict.fromkeys(m.RUNTIME_KEYS, "known")
    result = m.backfill(p, original, {"local": "H0"}, runtime, {}, "now")
    assert result["gsm8k"] == original["gsm8k"]
    assert result["batch_size"] == 16 and result["batch_size_nominal"] == 32
    assert result["metadata_backfill"] == "full"
    assert result["item_order_hash"] != result["item_id_hash"]
    assert m.backfill(p, result, {"local": "H0"}, runtime, {}, "later") == result


def test_remote_unknown_runtime_and_control_records_are_partial(tmp_path):
    p = tmp_path / "eval.remote.json"
    (tmp_path / "records.jsonl").write_text("{}")
    result = m.backfill(
        p,
        {"eval_host": "remote", "gsm8k": 0.5},
        {"remote": "H1"},
        {},
        {"H1": {"torch_version": "logged"}},
        "now",
    )
    assert result["eval_device_id"] == "H1"
    assert "eval_host" not in result
    assert result["torch_version"] == "logged"
    assert result["cuda_version"] is None
    assert result["item_id_hash"] is None
    assert result["metadata_backfill"] == "partial"


def test_cli_dry_run_and_apply_preserve_scores_and_link_zoo(tmp_path, monkeypatch):
    import sys

    zoo, evals = tmp_path / "zoo", tmp_path / "evals"
    entry = zoo / "entry"
    entry.mkdir(parents=True)
    evals.mkdir()
    p = entry / "eval.json"
    payload = {"gsm8k": 0.125, "run_id": "run", "eval_host": "local"}
    p.write_text(json.dumps(payload))
    mapping = tmp_path / "device_map"
    argv = ["backfill", "--zoo", str(zoo), "--evals", str(evals), "--device-map", str(mapping)]
    monkeypatch.setattr(sys, "argv", argv)
    m.main()
    assert json.loads(p.read_text()) == payload and not mapping.exists()
    monkeypatch.setattr(sys, "argv", argv + ["--apply"])
    m.main()
    assert (evals / "entry").is_symlink()
    assert json.loads(p.read_text())["gsm8k"] == 0.125
    before = p.read_bytes()
    m.main()
    assert p.read_bytes() == before
    assert json.loads(mapping.read_text()) == m.default_map()


def test_inferred_runtime_remains_labelled_and_requires_equal_values(tmp_path):
    from wsp.eval.repeats import matched_reference

    path = tmp_path / "eval.json"
    (tmp_path / "records.jsonl").write_text('{"benchmark":"a","item_id":"1"}')
    payload = dict.fromkeys(m.REFERENCE_KEYS, "known")
    payload["generation"] = {"batch_size": 32}
    payload["eval_device_id"] = "H1"
    payload["attention_implementation"] = None
    remote = {
        "H1": {
            "attention_implementation": "sdpa",
            "provenance": {"attention_implementation": "inferred"},
        }
    }
    result = m.backfill(path, payload, {}, {}, remote, "now")
    assert result["metadata_backfill"] == "partial-inferred"
    assert result["metadata_backfill_inferred"] == ["attention_implementation"]
    assert matched_reference(result, [result]) == result
    assert matched_reference(result, [result | {"attention_implementation": "eager"}]) is None
    assert m.backfill(path, result, {}, {}, remote, "later") == result
