"""F12-R remedies in the F12 token builder (scripts/f12_tokens.py; unexposed adoption A10, A11, A19, A21): authenticated
inputs (population, per-row stored bytes, edit.json through the snapshot's pinned index, probe files confined to bare
names), resume bound to code/population/base/G2/probes/threads and per-row sources with exact id equality and atomic
checkpoints, completeness by exact ids, launch refusal above the 4 GB cap, intent relabel, no overwrite. Synthetic
files only; nothing under data/ is opened."""

import argparse
import json
import os

import pytest

from scripts import a5_build_manifests as bm
from scripts import f12_inputs_guard as fg
from scripts import f12_tokens as ft

# ------------------------------------------------------------------------------------------- A10


def _zoo(tmp_path, ids=("e1", "e2", "e3")):
    zoo = tmp_path / "zoo"
    comps = []
    for i in ids:
        d = zoo / i
        d.mkdir(parents=True)
        (d / "edit.json").write_text(json.dumps({"id": i, "family": "random_dir"}))
        (d / "delta.safetensors").write_bytes(f"bytes-{i}".encode())
        comps.append(json.dumps({"id": i}))
    cj = tmp_path / "components.jsonl"
    cj.write_text("\n".join(comps) + "\n")
    index = fg.sha256_bytes(
        json.dumps(
            sorted([i, fg.sha256_bytes((zoo / i / "edit.json").read_bytes())] for i in ids)
        ).encode()
    )
    return zoo, cj, index


def test_edit_json_index_authenticates_every_edit(tmp_path, monkeypatch):
    zoo, cj, index = _zoo(tmp_path)
    monkeypatch.setattr(ft, "COMPONENTS_SHA256", fg.sha256_bytes(cj.read_bytes()))
    monkeypatch.setattr(ft, "EDIT_JSON_INDEX_SHA256", index)
    b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[])
    edit_sha = ft.authenticate_edit_index(b, zoo, cj)
    assert set(edit_sha) == {"e1", "e2", "e3"}
    assert ft.read_edit(b, zoo, "e2", edit_sha)["id"] == "e2"
    (zoo / "e2" / "edit.json").write_text(json.dumps({"id": "e2", "family": "lora_sft"}))
    with pytest.raises(fg.AuthenticationError):
        ft.read_edit(b, zoo, "e2", edit_sha)
    with pytest.raises(fg.AuthenticationError, match="edit.json index"):
        ft.authenticate_edit_index(b, zoo, cj)


def test_governed_pins():
    assert (
        ft.EDIT_JSON_INDEX_SHA256
        == "1c4ba09da3d7821fea8aabcf3561a4cec003297ae3f6adfb621f33663e0a0945"
    )
    assert (
        ft.COMPONENTS_SHA256 == "6b4eff8d61a4b1c4d83f0c058cb4df5c6c5398e2c6656a8b58b38f8b46251dec"
    )
    assert (
        ft.GOVERNED_POPULATION_SHA256
        == "07ab85d31cf8595b9b8dcd095a1f272aceb004cfc79d3296c55373db2e233da5"
    )
    b = ft.read_boundary()
    with pytest.raises(fg.BoundaryError, match="prohibited"):
        b.resolve(ft.ROOT / "data/evals/x/records.jsonl")
    with pytest.raises(fg.BoundaryError):
        b.resolve(ft.ROOT / "docs/prereg/a5_manifests/part_a/predictions_f11.json")


def test_row_source_authenticates_holder_bytes(tmp_path):
    zoo, _, _ = _zoo(tmp_path)
    b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[])
    h = bm.entry_delta_hash(bm.artefact_files(zoo / "e1"))
    r = {"id": "e1", "bytes_holder": "e1", "resolved_delta_hash": h}
    src = ft.row_source(b, zoo, r, {"e1": "sha-of-edit"})
    assert src == {"entry_delta_hash": h, "edit_json_sha256": "sha-of-edit"}
    with pytest.raises(fg.AuthenticationError):
        ft.row_source(b, zoo, r | {"resolved_delta_hash": "0" * 64}, {"e1": "x"})


def test_probe_files_are_bare_names_inside_the_cache(tmp_path):
    probe = tmp_path / "probe_cache"
    probe.mkdir()
    meta = [{"index": 0, "file": "../outside.safetensors", "content_hashes": {}, "n_tokens": 1}]
    (probe / "probe_cache.json").write_text(json.dumps(meta))
    man = {
        "probe_cache": {
            "status": "complete",
            "files": {
                "probe_cache.json": fg.sha256_bytes((probe / "probe_cache.json").read_bytes())
            },
        }
    }
    b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[])
    with pytest.raises(fg.BoundaryError, match="bare file name"):
        ft.load_probe(man, probe, range(1), boundary=b)


# ------------------------------------------------------------------------------------------- A19
# F12-R second remedy (independent re-check): resume is disabled for governed runs. Every run of this builder is
# governed (its outputs must lie under wsp-data/runs/a5 and it reads the real token inputs), so --resume is refused before
# any read; the checkpoint/resume code was removed.


def _recheck_counterexample(out):
    """The re-check's resume directory: one self-consistent row whose a is the overflow literal 1e999."""
    out.mkdir(parents=True)
    rec = {k: None for k in ft.ROW_KEYS} | {"id": "a", "increment_sha256": "inc-a", "seconds": 1.0}
    line = json.dumps(rec, sort_keys=True).replace('"a": null', '"a": 1e999') + "\n"
    (out / "rows.jsonl").write_text(line)
    det = {"id": "a", "tokens": {"3": [0.5] * ft.N_COLS}, "per_tensor": {}, "field_per_tensor": {}}
    dl = json.dumps(det, sort_keys=True) + "\n"
    (out / "detail.jsonl").write_text(dl)
    cp = {"schema": "f12-tokens-checkpoint-v1", "binding": {}, "rows": [{"id": "a"}]}
    (out / "checkpoint.json").write_text(json.dumps(cp))
    return line


def test_recheck_resume_counterexample_refused_before_any_read(tmp_path, monkeypatch):
    out = ft.ALLOWED_OUT / "f12r_resume_counterexample_never_created"
    assert not out.exists()

    def no_read(*a, **k):
        raise AssertionError("an input or resume file was read")

    monkeypatch.setattr(ft, "read_boundary", no_read)
    monkeypatch.setattr(ft, "check_fresh_out", no_read)
    monkeypatch.setattr(
        ft, "check_out_dir", no_read
    )  # a regression fails here, before any directory is made
    with pytest.raises(ValueError, match="resume is disabled for governed runs"):
        ft.main(["--out-dir", str(out), "--manifest", str(tmp_path / "m.json"), "--resume"])
    assert not out.exists()
    # the same directory as the re-check built it: refused too, and nothing in it is opened
    d = tmp_path / "rc"
    line = _recheck_counterexample(d)
    assert '"a": 1e999' in line
    with pytest.raises(ValueError, match="resume is disabled for governed runs"):
        ft.main(["--out-dir", str(d), "--manifest", str(tmp_path / "m2.json"), "--resume"])


def test_resume_code_removed():
    for name in ("load_checkpoint", "write_checkpoint", "checkpoint_entry", "checkpoint_entries", "_check_row",
                 "resume_binding"):  # fmt: skip
        assert not hasattr(ft, name), name
    assert ft.governed_reasons(ft.argparse.Namespace(out_dir=ft.ALLOWED_OUT / "x"))


def test_strict_json_rejects_overflow_literals():
    for bad in ('{"a": 1e999}', '{"a": -1e999}', "[1e400]"):
        with pytest.raises(ValueError, match="nonfinite"):
            fg.strict_json_loads(bad)
    assert fg.strict_json_loads('{"a": 1e300, "b": 0.5}') == {"a": 1e300, "b": 0.5}


def test_completeness_by_exact_ids():
    rows = [{"id": "a"}, {"id": "b"}]
    assert ft.is_complete({"a": 1, "b": 1}, rows, None)
    assert not ft.is_complete({"a": 1, "c": 1}, rows, None)  # same length, other ids
    assert not ft.is_complete({"a": 1, "b": 1}, rows, 2)


def test_fresh_run_refuses_prior_outputs(tmp_path):
    ft.check_fresh_out(tmp_path)
    (tmp_path / "tokens.safetensors").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        ft.check_fresh_out(tmp_path)


# ------------------------------------------------------------------------------------------- A21


def test_launch_refused_when_field_exceeds_cap(tmp_path):
    t = tmp_path / "g2" / "fields" / ft.OBJECTIVE / "tensors.pt"
    t.parent.mkdir(parents=True)
    with open(t, "wb") as f:  # sparse: no disk blocks
        f.truncate(fg.DEFAULT_CAP_BYTES + 1)
    b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[])
    est = ft.launch_estimate(b, tmp_path / "g2")
    assert est > fg.DEFAULT_CAP_BYTES
    guard = fg.ResourceGuard("scripts/f12_tokens.py", 6, mem_fn=lambda: 10**12, rss_fn=lambda: 1)
    with pytest.raises(fg.ResourceRefused, match="above the cap"):
        guard.launch(est)
    os.remove(t)


# --------------------------------------------------------------------------------------- A11, A22


def _manifest(tmp_path, man):
    pop = tmp_path / "population.json"
    pop.write_text(json.dumps({"rows": []}))
    args = argparse.Namespace(manifest=man, population=pop, zoo="z", threads=2)
    kdoc = {"kframes_safetensors_sha256": "k", "tensors": {}, "counts": {}}
    guard = fg.ResourceGuard("scripts/f12_tokens.py", 2, rss_fn=lambda: 1, mem_fn=lambda: 10**12,
                             peak_fn=lambda: 1)  # fmt: skip
    guard.launch(1)
    return ft.write_manifest(
        args, tmp_path, "partial (0/0)", {}, {"timing": {}}, {}, kdoc, "ks", None, [], None, 0.0, 0.0,
        "b", [], [], boundary=fg.ReadBoundary(allowed=[tmp_path]), guard=guard,
    )  # fmt: skip


def test_manifest_relabels_intent_records_reads_and_resources(tmp_path):
    doc = _manifest(tmp_path, tmp_path / "m.json")
    assert "outcomes_read" not in doc
    assert doc["outcomes_read_intent"]["statement"].startswith("none: stored edit bytes")
    assert "not access telemetry" in doc["outcomes_read_intent"]["label"]
    assert doc["read_boundary"]["allowed"] == [str(tmp_path.resolve())]
    assert doc["resources"]["cap_bytes"] == fg.DEFAULT_CAP_BYTES


def test_manifest_never_overwritten(tmp_path):
    m = tmp_path / "m.json"
    m.write_text("governed")
    with pytest.raises(FileExistsError):
        _manifest(tmp_path, m)
    assert m.read_text() == "governed"
