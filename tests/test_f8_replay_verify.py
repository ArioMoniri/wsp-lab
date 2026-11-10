"""F8a.4 CPU identity verifier on synthetic tiny checkpoints in tmp_path (no download, no data/)."""

import json
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from scripts import f8_replay_verify as rv

REV = "ab" * 20
DOWN = "model.language_model.layers.1.mlp.down_proj.weight"
UP = "model.language_model.layers.0.mlp.down_proj.weight"
ALLOW = [UP, DOWN]


def base_tensors():
    g = torch.Generator().manual_seed(0)
    t = {
        UP: torch.randn(8, 16, generator=g).to(torch.bfloat16),
        DOWN: torch.randn(8, 16, generator=g).to(torch.bfloat16),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8, generator=g).to(
            torch.bfloat16
        ),
        "model.language_model.layers.0.linear_attn.A_log": torch.randn(4, generator=g),
        "model.language_model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    return t


def edited(tensors):
    out = {k: v.clone() for k, v in tensors.items()}
    for name, cols in ((UP, [1, 3]), (DOWN, [2])):
        w = out[name].float()
        w[:, cols] += 0.5
        out[name] = w.to(torch.bfloat16)
    return out


TOKENIZER = {
    "tokenizer.json": b'{"model": "toy"}',
    "tokenizer_config.json": b'{"x": 1}',
    "chat_template.jinja": b"{{ messages }}",
}
CONFIG = {
    "architectures": ["Toy"],
    "hidden_size": 8,
    "_name_or_path": "base",
    "transformers_version": "5.0.0",
}


def write_dir(path, tensors, *, config=None, tokenizer=None, shards=1):
    path.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    if shards == 1:
        save_file(tensors, str(path / "model.safetensors"))
    else:
        weight_map = {}
        for i in range(shards):
            part = names[i::shards]
            fname = f"model-{i + 1:05}-of-{shards:05}.safetensors"
            save_file({k: tensors[k] for k in part}, str(path / fname))
            weight_map |= dict.fromkeys(part, fname)
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
    (path / "config.json").write_text(json.dumps(config or CONFIG))
    for name, data in (tokenizer or TOKENIZER).items():
        (path / name).write_bytes(data)
    return path


@pytest.fixture
def dirs(tmp_path):
    base = write_dir(tmp_path / "hub/snapshots" / REV, base_tensors())
    cand = write_dir(
        tmp_path / "cand",
        edited(base_tensors()),
        config=CONFIG | {"_name_or_path": "astra", "transformers_version": "5.1.0"},
        shards=2,
    )
    return base, cand


def run(base, cand, allow=ALLOW, **kw):
    kw.setdefault("reload", "inline")
    return rv.verify(cand, base, allow, expected_base_revision=REV, **kw)


def failed(record):
    return {s for s, r in record["steps"].items() if r["status"] == "fail"}


def test_exact_two_tensor_edit_passes_and_lists_coordinates(dirs):
    base, cand = dirs
    rec = run(base, cand)
    assert rec["verdict"] == "PASS", failed(rec)
    diff = rec["changed_tensors"]
    assert sorted(diff) == sorted(ALLOW)
    assert diff[UP]["changed_coordinates"] == 16 and diff[UP]["changed_cols"]["count"] == 2
    assert diff[DOWN]["changed_coordinates"] == 8 and diff[DOWN]["changed_rows"]["count"] == 8
    assert diff[DOWN]["changed_cols"]["indices"] == [2]
    assert rec["steps"]["A7_unchanged_complement"]["status"] == "pass"
    assert rec["hashes"]["complement"]["base"] == rec["hashes"]["complement"]["candidate"]
    json.dumps(rec, allow_nan=False)


def test_unexpected_changed_tensor_fails_closed(dirs, tmp_path):
    base, _ = dirs
    t = edited(base_tensors())
    t["model.language_model.norm.weight"][3] = 2.0
    cand = write_dir(tmp_path / "c2", t)
    rec = run(base, cand)
    assert rec["verdict"] == "FAIL"
    assert {"A6_changed_set", "A7_unchanged_complement"} <= failed(rec)
    assert rec["steps"]["A6_changed_set"]["unexpected_changed"] == [
        "model.language_model.norm.weight"
    ]
    assert rec["changed_tensors"]["model.language_model.norm.weight"]["changed_coordinates"] == 1


def test_allowlisted_tensor_unchanged_fails(dirs, tmp_path):
    base, _ = dirs
    t = base_tensors()
    w = t[UP].float()
    w[0, 0] += 1
    t[UP] = w.to(torch.bfloat16)
    rec = run(base, write_dir(tmp_path / "c", t))
    assert rec["verdict"] == "FAIL"
    assert rec["steps"]["A6_changed_set"]["allowlisted_unchanged"] == [DOWN]


@pytest.mark.parametrize("change", ["missing", "extra", "dtype", "shape"])
def test_tensor_set_shape_dtype_violations_fail(dirs, tmp_path, change):
    base, _ = dirs
    t = edited(base_tensors())
    key = "model.language_model.layers.0.self_attn.o_proj.weight"
    if change == "missing":
        del t[key]
    elif change == "extra":
        t["lm_head.weight"] = torch.zeros(2, 2)
    elif change == "dtype":
        t[key] = t[key].float()
    else:
        t[key] = t[key].reshape(4, 16)
    rec = run(base, write_dir(tmp_path / "c", t))
    assert rec["verdict"] == "FAIL"
    step = "A4_tensor_names" if change in ("missing", "extra") else "A5_shapes_dtypes"
    assert step in failed(rec)


def test_bitwise_comparison_nan_and_signed_zero(tmp_path):
    t = base_tensors()
    t["model.language_model.layers.0.linear_attn.A_log"][0] = float("nan")
    t["model.language_model.norm.weight"][1] = 0.0
    base = write_dir(tmp_path / "hub/snapshots" / REV, t)
    c = edited(t)
    rec = run(base, write_dir(tmp_path / "same", c))
    assert rec["verdict"] == "PASS", failed(rec)  # identical NaN bytes are unchanged
    c["model.language_model.norm.weight"][1] = -0.0
    rec = run(base, write_dir(tmp_path / "negzero", c))
    assert rec["verdict"] == "FAIL"
    assert rec["changed_tensors"]["model.language_model.norm.weight"]["changed_coordinates"] == 1


def test_tokenizer_byte_difference_and_missing_file_fail(dirs, tmp_path):
    base, _ = dirs
    tok = dict(TOKENIZER) | {"tokenizer.json": b'{"model": "toy2"}'}
    rec = run(base, write_dir(tmp_path / "c", edited(base_tensors()), tokenizer=tok))
    assert "A3_tokenizer" in failed(rec) and rec["verdict"] == "FAIL"
    tok = {k: v for k, v in TOKENIZER.items() if k != "chat_template.jinja"}
    rec = run(base, write_dir(tmp_path / "c2", edited(base_tensors()), tokenizer=tok))
    assert "A3_tokenizer" in failed(rec)


def test_config_allowlist(dirs, tmp_path):
    base, cand = dirs
    assert run(base, cand)["steps"]["A3_config"]["status"] == "pass"
    rec = run(
        base, write_dir(tmp_path / "c", edited(base_tensors()), config=CONFIG | {"hidden_size": 9})
    )
    assert "A3_config" in failed(rec)
    assert rec["steps"]["A3_config"]["differing_keys"] == ["hidden_size"]


def test_base_revision_mismatch_fails(dirs):
    base, cand = dirs
    rec = rv.verify(cand, base, ALLOW, expected_base_revision="cd" * 20, reload="inline")
    assert "base_revision" in failed(rec) and rec["verdict"] == "FAIL"


def test_nondeterministic_reload_fails(dirs, monkeypatch):
    base, cand = dirs
    real = rv.state_digest
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        out = real(path)
        if calls["n"] == 2:  # second of the two reloads
            out = out | {"state_hash": "0" * 64}
        return out

    monkeypatch.setattr(rv, "state_digest", flaky)
    rec = run(base, cand)
    assert "A11_reload_determinism" in failed(rec) and rec["verdict"] == "FAIL"


def test_fresh_process_reload(dirs):
    base, cand = dirs
    rec = run(base, cand, reload="process")
    step = rec["steps"]["A11_reload_determinism"]
    assert step["status"] == "pass" and step["mode"] == "process" and step["loads"] == 3


def test_claimed_columns(dirs):
    base, cand = dirs
    ok = run(base, cand, claimed_columns={UP: [1, 3, 5], DOWN: [2]})
    assert ok["steps"]["A8_claimed_columns"]["status"] == "pass"
    bad = run(base, cand, claimed_columns={UP: [1], DOWN: [2]})
    assert "A8_claimed_columns" in failed(bad)
    assert bad["steps"]["A8_claimed_columns"]["outside"] == {UP: [3]}
    assert run(base, cand)["steps"]["A8_claimed_columns"]["status"] == "not_run"


def test_roundtrip_delta_reconstructs_candidate(dirs):
    base, cand = dirs
    rec = run(base, cand)
    assert rec["steps"]["A9_roundtrip"]["status"] == "pass"
    assert rec["hashes"]["state"]["roundtrip"] == rec["hashes"]["state"]["candidate"]


def test_scoring_identity_compares_shared_fields_only(dirs):
    base, cand = dirs
    ref = rv.scoring_identity(base, REV, rv.DEFAULT_CONFIG)
    ref = ref | {"entry_hashes": {}, "state_hash": "base-state"}  # a base reference record
    rec = run(base, cand, reference_identity=ref)
    assert rec["steps"]["scoring_identity"]["status"] == "pass"
    rec = run(base, cand, reference_identity=ref | {"config_hash": "x" * 64})
    assert rec["steps"]["scoring_identity"]["mismatched"] == ["config_hash"]
    assert rec["verdict"] == "FAIL"


def test_incomplete_qualification_is_flagged(dirs):
    base, cand = dirs
    rec = run(base, cand)
    assert rec["complete_for_f8a4"] is False
    assert {"A1_provenance", "A2_form", "A10_apply_path"} <= set(rec["steps_not_run"])


def test_dry_run_cli(tmp_path):
    out = tmp_path / "dry.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(rv.ROOT / "scripts/f8_replay_verify.py"),
            "--dry-run",
            "--output",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    rec = json.loads(out.read_text())
    assert rec["dry_run"] is True
    assert rec["positive"]["verdict"] == "PASS"
    assert rec["negative"]["verdict"] == "FAIL"
    assert "PASS" in proc.stdout
