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

BINDING = {"code": {"scripts/f12_tokens.py": "c0"}, "population_json_sha256": "p", "threads": 6}


def _rec(i):
    return {k: None for k in ft.ROW_KEYS} | {
        "id": i,
        "increment_sha256": f"inc-{i}",
        "seconds": 1.0,
    }


def _detail(i, value=0.5):
    return {"id": i, "tokens": {"3": [value] * ft.N_COLS}, "per_tensor": {}, "field_per_tensor": {}}


def _write_rows(out, ids, sources):
    rows, detail = out / "rows.jsonl", out / "detail.jsonl"
    rows.write_text("")
    detail.write_text("")
    entries = []
    ft.write_checkpoint(out, BINDING, entries)
    for i in ids:
        rl, dl = ft.append_row(rows, detail, _rec(i), _detail(i))
        entries.append(ft.checkpoint_entry(i, sources[i], f"inc-{i}", rl, dl))
        ft.write_checkpoint(out, BINDING, entries)
    return entries


@pytest.fixture
def resumable(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    sources = {i: {"entry_delta_hash": f"h-{i}", "edit_json_sha256": f"s-{i}"} for i in "abcd"}
    _write_rows(out, ["a", "b"], sources)
    return {"out": out, "sources": sources, "todo": list("abcd")}


def _load(r, binding=BINDING, todo=None, sources=None):
    src = sources or r["sources"]
    return ft.load_checkpoint(r["out"], binding, todo or r["todo"], lambda i: src[i])


def test_resume_from_bound_checkpoint(resumable):
    done, recs = _load(resumable)
    assert list(done) == ["a", "b"] and [x["id"] for x in recs] == ["a", "b"]


def test_resume_refused_without_checkpoint(resumable):
    (resumable["out"] / ft.CHECKPOINT).unlink()
    with pytest.raises(ValueError, match="no bound checkpoint"):
        _load(resumable)


@pytest.mark.parametrize("key", ["code", "population_json_sha256", "threads"])
def test_resume_refused_on_binding_change(resumable, key):
    other = json.loads(json.dumps(BINDING))
    other[key] = {"scripts/f12_tokens.py": "c1"} if key == "code" else "changed"
    with pytest.raises(ValueError, match=f"binding differs.*{key}"):
        _load(resumable, binding=other)


def test_resume_refused_when_ids_not_the_todo_prefix(resumable):
    with pytest.raises(ValueError, match="exact prefix"):
        _load(resumable, todo=list("bacd"))


def test_resume_refused_on_changed_row_source(resumable):
    src = dict(resumable["sources"]) | {
        "b": {"entry_delta_hash": "other", "edit_json_sha256": "s-b"}
    }
    with pytest.raises(ValueError, match="source"):
        _load(resumable, sources=src)


def test_resume_truncates_uncommitted_tail(resumable):
    out = resumable["out"]
    with (out / "rows.jsonl").open("a") as f:
        f.write(json.dumps(_rec("c")) + "\n")  # appended, crashed before the checkpoint
    with (out / "detail.jsonl").open("a") as f:
        f.write('{"id": "c", "tok')
    done, _ = _load(resumable)
    assert list(done) == ["a", "b"]
    assert len((out / "rows.jsonl").read_text().splitlines()) == 2
    assert (out / "detail.jsonl").read_text().endswith("}\n")


def test_resume_refused_on_tampered_committed_bytes(resumable):
    p = resumable["out"] / "detail.jsonl"
    p.write_text(p.read_text().replace("0.5", "0.6", 1))
    with pytest.raises(ValueError, match="prefix"):
        _load(resumable)


def test_resume_refused_on_duplicates_nonfinite_and_schema(tmp_path):
    sources = {i: {"entry_delta_hash": i, "edit_json_sha256": i} for i in "ab"}
    out = tmp_path / "dup"
    out.mkdir()
    entries = _write_rows(out, ["a"], sources)
    ft.write_checkpoint(out, BINDING, entries + entries)
    with pytest.raises(ValueError, match="duplicate"):
        ft.load_checkpoint(out, BINDING, ["a", "a", "b"], lambda i: sources[i])
    out = tmp_path / "nan"
    out.mkdir()
    (out / "rows.jsonl").write_text("")
    (out / "detail.jsonl").write_text("")
    rl, dl = ft.append_row(out / "rows.jsonl", out / "detail.jsonl", _rec("a"),
                           _detail("a") | {"tokens": {"3": [float("nan")] * ft.N_COLS}}, strict=False)  # fmt: skip
    ft.write_checkpoint(out, BINDING, [ft.checkpoint_entry("a", sources["a"], "inc-a", rl, dl)])
    with pytest.raises(ValueError, match="nonfinite"):
        ft.load_checkpoint(out, BINDING, ["a", "b"], lambda i: sources[i])
    out = tmp_path / "short"
    out.mkdir()
    (out / "rows.jsonl").write_text("")
    (out / "detail.jsonl").write_text("")
    rl, dl = ft.append_row(out / "rows.jsonl", out / "detail.jsonl", _rec("a"),
                           _detail("a") | {"tokens": {"3": [0.0] * 5}})  # fmt: skip
    ft.write_checkpoint(out, BINDING, [ft.checkpoint_entry("a", sources["a"], "inc-a", rl, dl)])
    with pytest.raises(ValueError, match="token"):
        ft.load_checkpoint(out, BINDING, ["a", "b"], lambda i: sources[i])


def test_checkpoint_written_atomically(resumable, monkeypatch):
    calls = []
    real = fg.atomic_write_bytes
    monkeypatch.setattr(ft.fg, "atomic_write_bytes", lambda p, d: calls.append(p) or real(p, d))
    ft.write_checkpoint(resumable["out"], BINDING, [])
    assert calls == [resumable["out"] / ft.CHECKPOINT]


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
