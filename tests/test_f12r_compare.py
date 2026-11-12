"""F12-R token requalification comparison (scripts/f12_tokens_compare.py; chair ruling of 12 Nov 2026, 09:19 UTC (c)):
the rerun's token array, per-row records, K frames and manifest against the frozen run on every deterministic field,
timing fields excluded and listed, frozen and new files authenticated against their manifests, no token value printed.
Synthetic runs only; nothing under data/ is opened."""

import copy
import json

import pytest
import torch
from safetensors.torch import save_file

from scripts import f12_inputs_guard as fg
from scripts import f12_tokens_compare as tc

IDS = ["r1", "r2", "r3"]
SECRET = 0.123456789  # a token value that must never appear in the report


def _write(path, data: bytes) -> str:
    path.write_bytes(data)
    return fg.sha256_bytes(data)


def make_run(
    d, seconds=1.0, x_bump=None, rec_bump=None, kf_bump=False, detail_bump=None, pricing_secs=10.0
):
    d.mkdir(parents=True)
    X = torch.zeros(3, 4, 42)
    X[:, 1, :] = SECRET
    if x_bump is not None:
        X[x_bump, 1, 0] = torch.nextafter(torch.tensor(SECRET), torch.tensor(1.0))
    P = torch.zeros(3, 4, dtype=torch.uint8)
    P[:, 1] = 1
    E = torch.ones(3, dtype=torch.uint8)
    save_file({"X": X, "presence": P, "row_eligible": E}, str(d / "tokens.safetensors"),
              metadata={"row_ids": json.dumps(IDS), "columns": json.dumps(["c"] * 42)})  # fmt: skip
    rows, details = [], []
    for k, i in enumerate(IDS):
        rec = {k2: None for k2 in tc.ROW_KEYS} | {
            "id": i,
            "a": 0.5 + k,
            "q": 0.25,
            "seconds": seconds + k,
        }
        if rec_bump == i:
            rec["q"] = 0.2500001
        rows.append(json.dumps(rec, sort_keys=True) + "\n")
        det = {"id": i, "tokens": {"1": [SECRET] * 42}, "per_tensor": {}, "field_per_tensor": {}}
        if detail_bump == i:
            det["per_tensor"] = {"t": {"sigma1": 2.0}}
        details.append(json.dumps(det, sort_keys=True) + "\n")
    files = {
        "tokens.safetensors": fg.sha256_bytes((d / "tokens.safetensors").read_bytes()),
        "rows.jsonl": _write(d / "rows.jsonl", "".join(rows).encode()),
        "detail.jsonl": _write(d / "detail.jsonl", "".join(details).encode()),
        "aq.json": _write(d / "aq.json", json.dumps({i: {"a": 1.0} for i in IDS}).encode()),
        "pricing.json": _write(
            d / "pricing.json",
            json.dumps(
                {
                    "first_row_id": "r1",
                    "n_rows": 3,
                    "first_row_stored_tensors": 2,
                    "stored_tensors_total": 6,
                    "threads": 6,
                    "setup_seconds": pricing_secs,
                    "first_row_seconds": pricing_secs,
                    "projected_wall_seconds": pricing_secs,
                    "cap_wall_hours": 8.0,
                }
            ).encode(),
        ),
    }
    kf = {"m.G.U": torch.eye(3, dtype=torch.float64)}
    if kf_bump:
        kf["m.G.U"][0, 0] = 1.0000000000000002
    save_file(kf, str(d / "kframes.safetensors"))
    files["kframes.safetensors"] = fg.sha256_bytes((d / "kframes.safetensors").read_bytes())
    files["kframes.json"] = _write(
        d / "kframes.json",
        json.dumps({"kframes_safetensors_sha256": files["kframes.safetensors"]}).encode(),
    )
    man = {
        "schema": "f12-tokens-v1",
        "status": "complete",
        "basis": "b",
        "columns": ["c"] * 42,
        "slots": {"n": 4},
        "base": {"state_hash": "s"},
        "population_frozen": {"population_json_sha256": "p", "verified_at": f"t{seconds}"},
        "tokens": {
            "sha256": files["tokens.safetensors"],
            "shape": [3, 4, 42],
            "path": str(d / "tokens.safetensors"),
        },
        "kframes": {"sha256": files["kframes.json"], "path": str(d / "kframes.json")},
        "inputs": {
            "g2_manifest": {"path": str(d), "sha256": "g"},
            "timing": {"field_load_seconds": seconds},
        },
        "files": files,
        "out_dir": str(d),
        "rows_sha256": fg.sha256_bytes(f"{seconds}".encode()),
        "summary": {
            "n_rows_done": 3,
            "eligible": 3,
            "timing": {"wall_seconds": seconds},
            "max_rss_bytes": int(seconds * 1000),
            "pricing": {"setup_seconds": pricing_secs},
        },
        "code": {"scripts/f12_tokens.py": f"c{seconds}"},
    }
    return man


@pytest.fixture
def runs(tmp_path):
    def build(**new_kw):
        fz, nw = tmp_path / "frozen", tmp_path / "new"
        for p in (fz, nw):
            if p.exists():
                import shutil

                shutil.rmtree(p)
        fm = make_run(fz)
        nm = make_run(nw, seconds=7.0, pricing_secs=99.0, **new_kw)
        fmp, nmp = tmp_path / "frozen_manifest.json", tmp_path / "new_manifest.json"
        fmp.write_text(json.dumps(fm))
        nmp.write_text(json.dumps(nm))
        b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[])
        return tc.compare_runs(fmp, fz, nmp, nw, b)

    return build


def test_identical_apart_from_timing(runs):
    rep = runs()
    assert rep["identical"] is True
    for name in ("tokens.safetensors", "rows.jsonl", "detail.jsonl", "aq.json", "kframes.json",
                 "kframes.safetensors", "pricing.json", "manifest"):  # fmt: skip
        assert rep["files"][name]["identical"] is True, name
    assert rep["files"]["rows.jsonl"]["n_rows"] == 3
    excl = rep["excluded"]
    assert "rows.jsonl: seconds" in excl and "manifest: summary.timing" in excl
    assert "manifest: summary.max_rss_bytes" in excl and "pricing.json: setup_seconds" in excl


def test_token_bit_difference_named_by_row(runs):
    rep = runs(x_bump=1)
    t = rep["files"]["tokens.safetensors"]
    assert rep["identical"] is False and t["identical"] is False
    assert t["X_rows_differing"] == ["r2"] and t["presence_identical"] is True


def test_row_record_difference(runs):
    rep = runs(rec_bump="r3")
    assert rep["files"]["rows.jsonl"]["rows_differing"] == ["r3"] and rep["identical"] is False


def test_detail_difference(runs):
    rep = runs(detail_bump="r1")
    assert rep["files"]["detail.jsonl"]["rows_differing"] == ["r1"]


def test_kframe_difference_named_by_tensor(runs):
    rep = runs(kf_bump=True)
    assert rep["files"]["kframes.safetensors"]["tensors_differing"] == ["m.G.U"]
    assert rep["identical"] is False


def test_report_prints_no_token_values(runs):
    for kw in ({}, {"x_bump": 0}, {"detail_bump": "r2"}):
        text = json.dumps(runs(**kw))
        assert repr(SECRET)[:8] not in text and "0.1234" not in text


def test_files_authenticated_against_their_manifests(tmp_path):
    fz, nw = tmp_path / "frozen", tmp_path / "new"
    fm, nm = make_run(fz), make_run(nw, seconds=7.0)
    (fz / "detail.jsonl").write_text("{}\n")
    fmp, nmp = tmp_path / "fm.json", tmp_path / "nm.json"
    fmp.write_text(json.dumps(fm))
    nmp.write_text(json.dumps(nm))
    b = fg.ReadBoundary(allowed=[tmp_path], prohibited=[])
    with pytest.raises(fg.AuthenticationError):
        tc.compare_runs(fmp, fz, nmp, nw, b)


def test_manifest_deterministic_fields_compared(tmp_path):
    fm = make_run(tmp_path / "a")
    nm = copy.deepcopy(fm)
    nm["base"]["state_hash"] = "other"
    nm["summary"]["timing"]["wall_seconds"] = 123.0
    nm["code"] = {"x": "y"}
    diffs = tc.manifest_differences(fm, nm)
    assert diffs == ["base.state_hash"]


def test_governed_boundary_and_paths():
    b = tc.read_boundary(fg.ROOT / "docs/prereg/a5_manifests/f12/f12r/x.json")
    b.resolve(tc.FROZEN_DIR / "tokens.safetensors")  # inside the boundary (nothing is opened)
    b.resolve(fg.ROOT / "docs/prereg/a5_manifests/f12/tokens_manifest.json")
    with pytest.raises(fg.BoundaryError):
        b.resolve(fg.ROOT / "data/evals/x")


def _swap_metadata_order(path):
    """Rewrite a safetensors header with its __metadata__ keys in the other order (same length, same data)."""
    import struct

    b = path.read_bytes()
    (n,) = struct.unpack("<Q", b[:8])
    head = json.loads(b[8 : 8 + n])
    meta = head.pop("__metadata__")
    new = {"__metadata__": dict(reversed(list(meta.items())))} | head
    text = json.dumps(new, separators=(",", ":")).encode()
    assert len(text) <= n
    path.write_bytes(b[:8] + text + b" " * (n - len(text)) + b[8 + n :])


def test_metadata_key_order_is_not_a_deterministic_field(tmp_path):
    """safetensors writes __metadata__ keys in a nondeterministic order: the whole-file sha256 is excluded, the
    metadata mapping and the data section are compared."""
    fz, nw = tmp_path / "frozen", tmp_path / "new"
    fm, nm = make_run(fz), make_run(nw, seconds=7.0)
    _swap_metadata_order(nw / "tokens.safetensors")
    sha = fg.sha256_bytes((nw / "tokens.safetensors").read_bytes())
    nm["files"]["tokens.safetensors"] = nm["tokens"]["sha256"] = sha
    fmp, nmp = tmp_path / "fm.json", tmp_path / "nm.json"
    fmp.write_text(json.dumps(fm))
    nmp.write_text(json.dumps(nm))
    rep = tc.compare_runs(fmp, fz, nmp, nw, fg.ReadBoundary(allowed=[tmp_path], prohibited=[]))
    t = rep["files"]["tokens.safetensors"]
    assert t["header_metadata_identical"] and t["data_section_identical"] and t["identical"]
    assert rep["identical"] is True
    assert any("tokens.safetensors whole-file sha256" in e for e in rep["excluded"])


def test_data_section_difference_detected(tmp_path):
    fz, nw = tmp_path / "frozen", tmp_path / "new"
    fm, nm = make_run(fz), make_run(nw, seconds=7.0)
    b = bytearray((nw / "tokens.safetensors").read_bytes())
    b[-1] ^= 1  # last byte: row_eligible of the last row
    (nw / "tokens.safetensors").write_bytes(bytes(b))
    nm["files"]["tokens.safetensors"] = nm["tokens"]["sha256"] = fg.sha256_bytes(bytes(b))
    fmp, nmp = tmp_path / "fm.json", tmp_path / "nm.json"
    fmp.write_text(json.dumps(fm))
    nmp.write_text(json.dumps(nm))
    rep = tc.compare_runs(fmp, fz, nmp, nw, fg.ReadBoundary(allowed=[tmp_path], prohibited=[]))
    t = rep["files"]["tokens.safetensors"]
    assert (
        not t["data_section_identical"] and not t["row_eligible_identical"] and not rep["identical"]
    )


def test_pricing_first_row_fields_are_resume_history(tmp_path):
    """The frozen run resumed with row 1 from an earlier capped run, so its pricing.json first-row fields record resume
    history; a fresh rerun's differ by construction. Excluded before the rerun, and listed."""
    assert tc.PRICING_KEYS == ("n_rows", "stored_tensors_total", "threads")
    rep_excl = tc.EXCLUDED
    assert any(e.startswith("pricing.json: first_row_id") for e in rep_excl)
    assert any(e.startswith("pricing.json: first_row_stored_tensors") for e in rep_excl)
