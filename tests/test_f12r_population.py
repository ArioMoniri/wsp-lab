"""F12-R remedies in the F12 population selector (scripts/f12_population.py; unexposed adoption A10, A11): the four
committed manifests are read through the read boundary and authenticated against their pins, the snapshot reaches the
selector only as its minimal projection, ``outcomes_read`` is relabelled as algorithmic intent, and the selection is
unchanged: the committed population.json is reproduced byte for byte once the relabel is mapped back."""

import json
import shutil

import pytest

from scripts import f12_inputs_guard as fg
from scripts import f12_population as fp

NAMES = ("components.jsonl", "snapshot.json", "selection.json", "selection_f11.json")


def test_input_pins_are_the_committed_population_inputs():
    pop = json.loads((fp.MANIFESTS / "f12/population.json").read_text())
    assert {k: v["sha256"] for k, v in pop["inputs"].items()} == fp.INPUT_SHA256


def test_manifests_outside_the_committed_directory_refused(tmp_path):
    for n in NAMES:
        shutil.copyfile(fp.MANIFESTS / n, tmp_path / n)
    with pytest.raises(fg.BoundaryError, match="outside the allowed"):
        fp.load_inputs(tmp_path)


def test_symlinked_manifest_into_prohibited_refused(tmp_path):
    evals = tmp_path / "evals"
    evals.mkdir()
    man = tmp_path / "m"
    man.mkdir()
    for n in NAMES:
        (man / n).write_text("{}")
    (evals / "x.json").write_text("{}")
    (man / "snapshot.json").unlink()
    (man / "snapshot.json").symlink_to(evals / "x.json")
    b = fg.ReadBoundary(allowed=[man], prohibited=[evals])
    with pytest.raises(fg.BoundaryError, match="prohibited"):
        fp.load_inputs(man, boundary=b, pins={n: None for n in NAMES})


def test_tampered_input_refused(tmp_path):
    for n in NAMES:
        (tmp_path / n).write_text("{}\n")
    b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[])
    with pytest.raises(fg.AuthenticationError):
        fp.load_inputs(tmp_path, boundary=b)  # governed pins


def test_snapshot_reaches_selector_as_minimal_projection():
    snap = {
        "content": {
            "tf_v1_records": {"primary": {"a": ["h1", "h2"], "b": ["h3", "h4"]}, "path": "x"},
            "zoo_bytes": {"entry_delta_hash": {"a": "d1"}, "hash": "rule"},
            "feature_manifest": {"anything": 1},
        },
        "exposure_ledger": {"rows": [{"gate": "G0", "id": "a", "aliases": ["b"], "note": "n"}]},
        "g1_scoring": {"x": 1},
    }
    m = fp.minimal_snapshot(snap)
    assert m == {
        "content": {
            "tf_v1_records": {"primary": ["a", "b"]},
            "zoo_bytes": {"entry_delta_hash": {"a": "d1"}},
        },
        "exposure_ledger": {"rows": [{"gate": "G0", "id": "a", "aliases": ["b"]}]},
    }


def test_intent_relabel_and_committed_population_reproduced():
    components, snapshot, selection, selection_f11, inputs = fp.load_inputs(fp.MANIFESTS)
    doc = fp.build(components, snapshot, selection, selection_f11, inputs)
    assert "outcomes_read" not in doc
    intent = doc["outcomes_read_intent"]
    assert "not access telemetry" in intent["label"]
    legacy = {k: v for k, v in doc.items() if k != "outcomes_read_intent"}
    legacy["outcomes_read"] = intent["statement"]
    data = (json.dumps(legacy, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    assert data == (fp.MANIFESTS / "f12/population.json").read_bytes()


def test_main_writes_read_boundary_record_and_refuses_overwrite(tmp_path, monkeypatch):
    def guard(build, threads, cap_bytes, ruling):  # the pytest process's own peak is not this job's
        return fg.ResourceGuard(
            build, threads, cap_bytes, ruling,
            rss_fn=lambda: 10**6, mem_fn=lambda: 10**12, peak_fn=lambda: 10**6,
        )  # fmt: skip

    monkeypatch.setattr(fp, "resource_guard", guard)
    out = tmp_path / "population.json"
    fp.main(["--out", str(out)])
    doc = json.loads(out.read_text())
    reads = doc["read_boundary"]["reads"]
    assert [r["path"].rsplit("/", 1)[-1] for r in reads] == list(NAMES)
    assert all(r["authenticated"] for r in reads)
    assert doc["resources"]["cap_bytes"] == fg.DEFAULT_CAP_BYTES and doc["resources"]["within_cap"]
    with pytest.raises(FileExistsError):
        fp.main(["--out", str(out)])
