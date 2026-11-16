"""MSRM-1 development v0: the outcome allowlist and its read boundary (hard rule 1).

Only fake evaluation trees under tmp_path are opened here; the real population manifest is read for ids only.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import f12_inputs_guard as fg
from wsp.msrm import outcomes as oc

ROOT = Path(__file__).resolve().parents[1]
POPULATION = ROOT / "docs/prereg/a5_manifests/f12/population.json"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _records(nll, hashes=None, n_tokens=1023):
    lines = [json.dumps({"benchmark": "mmlu_pro", "item_id": "0", "correct": 1.0, "extra": {}})]
    for w, v in enumerate(nll):
        extra = {"mean_nll": v, "token_ids_hash": (hashes or {}).get(w, f"t{w}")}
        lines.append(
            json.dumps(
                {
                    "benchmark": "wikitext_ppl",
                    "item_id": str(w),
                    "n_tokens": n_tokens,
                    "extra": extra,
                }
            )
        )
    return ("\n".join(lines) + "\n").encode()


def _fake_tree(tmp_path, values):
    """evals/<id>/{eval.json, records.jsonl} plus pins, as the snapshot pins them."""
    evals = tmp_path / "data_root" / "evals" / "qwen3.5-0.8b" / "tf_v1"
    pins = {}
    for rid, nll in values.items():
        d = evals / rid
        d.mkdir(parents=True)
        e = json.dumps(
            {
                "eval_status": "complete",
                "repeat_id": None,
                "config_hash": "cfg",
                "model_revision": "rev",
            }
        ).encode()
        r = _records(nll)
        (d / "eval.json").write_bytes(e)
        (d / "records.jsonl").write_bytes(r)
        pins[rid] = [_sha(e), _sha(r)]
    return evals, pins


def _reader(tmp_path, values, allow=None, forbidden=()):
    evals, pins = _fake_tree(tmp_path, values)
    allow = sorted(allow if allow is not None else [k for k in values if k != "base"])
    return (
        oc.OutcomeReader(
            evals_root=evals,
            allowlist=allow,
            allowlist_sha256=oc.ids_sha256(allow),
            pins=pins,
            forbidden_ids=set(forbidden),
            data_root=tmp_path / "data_root",
        ),
        evals,
    )


def test_real_allowlist_is_the_123_existing_roots():
    pop = json.loads(POPULATION.read_text())
    ids = oc.allowlist_from_population(pop)
    assert len(ids) == 123
    assert oc.ids_sha256(ids) == oc.EXISTING_ROOTS_SHA256
    assert oc.EXISTING_ROOTS_SHA256 == (
        "f96e4ea55f4e887abdd713e97b27a06eec2b80fba82a397fbe5d6b9a36ca2160"
    )
    bad = dict(pop, rows=[r for r in pop["rows"] if r["id"] != ids[0]])
    with pytest.raises(fg.AuthenticationError):
        oc.allowlist_from_population(bad)


def test_reader_refuses_a_wrong_allowlist_hash(tmp_path):
    evals, pins = _fake_tree(tmp_path, {"aaaa": [1.0], "base": [1.0]})
    with pytest.raises(fg.AuthenticationError):
        oc.OutcomeReader(evals, ["aaaa"], "0" * 64, pins, set(), data_root=tmp_path)


def test_delta_y_is_token_weighted_difference_from_base(tmp_path):
    reader, _ = _reader(tmp_path, {"aaaa": [2.0, 4.0], "base": [1.0, 1.0]})
    dy, meta = reader.delta_y("aaaa")
    assert dy == pytest.approx(2.0)
    assert meta["n_windows"] == 2 and meta["eval_sha256"] and meta["records_sha256"]
    assert {r["id"] for r in reader.reads} == {"aaaa", "base"}


@pytest.mark.parametrize("rid", ["bbbb", "../aaaa", "aaaa/../bbbb", "", "part_a", "base"])
def test_ids_outside_the_allowlist_are_refused_before_opening(tmp_path, rid, monkeypatch):
    reader, _ = _reader(tmp_path, {"aaaa": [1.0], "bbbb": [1.0], "base": [1.0]}, allow=["aaaa"])
    opened = []
    real_open = os.open
    monkeypatch.setattr(os, "open", lambda p, *a, **k: opened.append(p) or real_open(p, *a, **k))
    with pytest.raises(fg.BoundaryError):
        reader.delta_y(rid)
    assert opened == []


def test_part_a_ids_are_refused_even_if_allowlisted(tmp_path):
    with pytest.raises(fg.BoundaryError):
        _reader(tmp_path, {"aaaa": [1.0], "base": [1.0]}, forbidden={"aaaa"})


def test_symlink_escape_into_a_prohibited_tree_is_refused(tmp_path):
    reader, evals = _reader(tmp_path, {"aaaa": [1.0], "base": [1.0]})
    target = tmp_path / "data_root" / "part_a_staging" / "x"
    target.mkdir(parents=True)
    (target / "eval.json").write_bytes((evals / "aaaa" / "eval.json").read_bytes())
    (target / "records.jsonl").write_bytes((evals / "aaaa" / "records.jsonl").read_bytes())
    for f in ("eval.json", "records.jsonl"):
        (evals / "aaaa" / f).unlink()
    (evals / "aaaa").rmdir()
    (evals / "aaaa").symlink_to(target)
    with pytest.raises(fg.BoundaryError):
        reader.delta_y("aaaa")


def test_pinned_bytes_are_authenticated(tmp_path):
    reader, evals = _reader(tmp_path, {"aaaa": [1.0], "base": [1.0]})
    (evals / "aaaa" / "records.jsonl").write_bytes(_records([9.0]))
    with pytest.raises(fg.AuthenticationError):
        reader.delta_y("aaaa")


def test_reader_never_lists_a_directory(tmp_path, monkeypatch):
    reader, _ = _reader(tmp_path, {"aaaa": [1.0, 2.0], "base": [1.0, 1.0]})

    def refuse(*a, **k):
        raise AssertionError("directory listing attempted")

    monkeypatch.setattr(os, "listdir", refuse)
    monkeypatch.setattr(os, "scandir", refuse)
    monkeypatch.setattr(Path, "iterdir", refuse)
    monkeypatch.setattr(Path, "glob", refuse)
    reader.delta_y("aaaa")


def test_window_mismatch_with_base_fails_closed(tmp_path):
    reader, _ = _reader(tmp_path, {"aaaa": [1.0, 2.0], "base": [1.0]})
    with pytest.raises(ValueError):
        reader.delta_y("aaaa")


def test_incomplete_or_repeat_record_fails_closed(tmp_path):
    evals, pins = _fake_tree(tmp_path, {"aaaa": [1.0], "base": [1.0]})
    e = json.dumps({"eval_status": "complete", "repeat_id": "g1", "config_hash": "cfg"}).encode()
    (evals / "aaaa" / "eval.json").write_bytes(e)
    pins["aaaa"][0] = _sha(e)
    reader = oc.OutcomeReader(
        evals, ["aaaa"], oc.ids_sha256(["aaaa"]), pins, set(), data_root=tmp_path / "data_root"
    )
    with pytest.raises(ValueError):
        reader.delta_y("aaaa")
