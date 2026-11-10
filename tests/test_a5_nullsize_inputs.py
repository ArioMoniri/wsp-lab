"""NULL-SIZE payload materialisation (checklist item 6) on a synthetic toy package (no data/, CPU)."""

import hashlib
import json
import math

import numpy as np
import pytest
import yaml

from scripts import a5_build_panels as bp
from scripts import a5_nullsize_inputs as ni
from scripts import a5_simulation as sim
from wsp.eval.batch import effective_config_hash

ROOT = ni.ROOT
CFG = yaml.safe_load((ROOT / "configs/eval_tf_v1.yaml").read_bytes())
COUNTS = {k: v.get("n", v.get("n_windows")) for k, v in CFG["benchmarks"].items()}
REV = "r" * 40
WINDOWS = [(str(i), f"tok{i}") for i in range(64)]


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def _record(evals, rid, nll, *, entry_hashes=None, windows=WINDOWS, **over):
    d = evals / rid
    d.mkdir(parents=True)
    e = {
        "eval_status": "complete",
        "repeat_id": None,
        "config_hash": effective_config_hash(CFG),
        "model_revision": REV,
        "source_hash": "s" * 64,
        "eval_device_id": "H0",
        "counts": COUNTS,
        "entry_hashes": entry_hashes or {},
        "run_id": f"run-{rid}",
    } | over
    (d / "eval.json").write_text(json.dumps(e))
    _jsonl(
        d / "records.jsonl",
        [{"benchmark": "mmlu_pro", "item_id": "0", "correct": 1.0}]
        + [
            {
                "benchmark": "wikitext_ppl",
                "item_id": i,
                "extra": {"mean_nll": v, "token_ids_hash": h},
            }
            for (i, h), v in zip(windows, nll)
        ],
    )


def toy(tmp_path, n_lora=33, n_chain=12):
    man, evals, zoo = tmp_path / "man", tmp_path / "evals", tmp_path / "zoo"
    for d in (man, evals, zoo):
        d.mkdir()
    rng = np.random.default_rng(0)
    base = 2.5 + 0.1 * rng.standard_normal(64)
    _record(evals, "base", base.tolist())
    comps, manifest = [], [{"type": "header"}]

    def entry(i, fam, parent, norm, eligible=True):
        (zoo / i).mkdir()
        (zoo / i / "delta.safetensors").write_bytes(f"bytes-{i}".encode())
        sha = hashlib.sha256(f"bytes-{i}".encode()).hexdigest()
        manifest.append({"id": i, "family": fam, "parent_id": parent, "base_revision": REV})
        comps.append(
            {
                "id": i,
                "family": fam,
                "parent_id": parent,
                "component_id": f"C-{parent or i}",
                "effective_norm": norm,
                "eligibility": "eligible" if eligible else "excluded:eligibility:missing",
                "bytes_present": True,
                "bytes_via_alias": None,
            }
        )
        return {"delta.safetensors": sha}

    shift = {}
    for k in range(n_lora):
        i = f"L{k:02d}"
        h = entry(i, "lora_sft", None, 0.1 * (k + 1))
        shift[i] = 0.01 * (k + 1)
        _record(
            evals, i, (base + shift[i] + 0.01 * rng.standard_normal(64)).tolist(), entry_hashes=h
        )
    for k in range(n_chain):
        root, step = f"R{k:02d}", f"S{k:02d}"
        hr = entry(root, "chain", None, 5.0, eligible=False)
        hs = entry(step, "chain", root, 0.2 * (k + 1))
        pred = base + 0.05 + 0.01 * rng.standard_normal(64)
        _record(evals, root, pred.tolist(), entry_hashes=hr)
        _record(evals, step, (pred + 0.02 * (k + 1)).tolist(), entry_hashes=hs)
    _jsonl(zoo / "manifest.jsonl", manifest)
    _jsonl(man / "components.jsonl", comps)
    text_groups = {
        "windows": [{"window_id": f"w:{i}", "item_id": i, "token_ids_hash": h} for i, h in WINDOWS],
        "window_ids_sorted_sha256": hashlib.sha256(
            json.dumps(sorted(f"w:{i}" for i, _ in WINDOWS)).encode()
        ).hexdigest(),
    }
    (man / "text_groups.json").write_text(json.dumps(text_groups))
    (man / "nullsize_inputs.json").write_text(json.dumps(bp.build_nullsize_spec(text_groups)))
    inv = [
        {
            "id": r["id"],
            "component": r["component_id"],
            "family": {"lora_sft": "LoRA", "chain": "chain"}[r["family"]],
            "norm": r["effective_norm"],
            "eligible": r["eligibility"] == "eligible",
        }
        for r in comps
    ]
    _, b = sim.selection_cells(inv)
    cut = {
        {"LoRA": "lora_sft", "chain": "chain"}[f]: {"keys": [[n, i] for n, i in e]}
        for f, e in b.items()
    }
    (man / "selection.json").write_text(
        json.dumps(
            {"procedural_freeze": {"cutpoints": cut, "registered_edit_ids_sorted_sha256": "0" * 64}}
        )
    )
    return {"manifests": man, "evals": evals, "zoo": zoo}, base


def test_payload_matches_the_registered_proxy_and_validates(tmp_path):
    paths, _base = toy(tmp_path)
    payload = ni.materialise(paths)
    spec = json.loads((paths["manifests"] / "nullsize_inputs.json").read_text())
    report = sim.validate_inputs(json.loads(json.dumps(payload)), spec)
    assert report["status"] == "valid", report["errors"]
    ids = [d["id"] for d in payload["donors"]]
    assert len(ids) == 33 + 12 and "R00" not in ids  # ineligible chain roots are not donors
    j = ids.index("S03")
    edit = [
        json.loads(x) for x in (paths["evals"] / "S03" / "records.jsonl").read_text().splitlines()
    ]
    pred = [
        json.loads(x) for x in (paths["evals"] / "R03" / "records.jsonl").read_text().splitlines()
    ]
    u = np.array(
        [a["extra"]["mean_nll"] - b["extra"]["mean_nll"] for a, b in zip(edit[1:], pred[1:])]
    )
    assert math.isclose(payload["donor_mean"][j], u.mean(), abs_tol=1e-12)
    assert np.allclose(payload["centred_noise"][j], u - u.mean())
    assert payload["provenance"]["donors"]["S03"]["reference_id"] == "R03"
    assert payload["provenance"]["donors"]["L00"]["reference_id"] == "base"
    for d in payload["donors"]:  # audits are never a recorded pass pre-G2
        assert d["audit"] == [False] * 3 and d["audit_evidence"] == ["unavailable"] * 3
    suff = ni.sufficiency(payload)
    assert suff["sufficient"] and suff["n_donor_pairs"] == 45


def test_missing_records_windows_and_bytes_are_listed_not_imputed(tmp_path):
    paths, _base = toy(tmp_path)
    # L05: no record; L06: bytes changed since scoring; S01: one window nonfinite (masked)
    for f in (paths["evals"] / "L05").iterdir():
        f.unlink()
    (paths["evals"] / "L05").rmdir()
    (paths["zoo"] / "L06" / "delta.safetensors").write_bytes(b"replayed")
    rec = paths["evals"] / "S01" / "records.jsonl"
    rows = [json.loads(x) for x in rec.read_text().splitlines()]
    rows[5]["extra"]["mean_nll"] = None
    _jsonl(rec, rows)
    payload = ni.materialise(paths)
    miss = payload["provenance"]["missing"]
    assert miss["L05"] == "edit:no_primary_record"
    assert miss["L06"] == "edit:entry_hashes_differ_from_current_bytes"
    j = [d["id"] for d in payload["donors"]].index("S01")
    assert payload["window_mask"][j][4] is False and payload["centred_noise"][j][4] is None
    valid = [x for x in payload["centred_noise"][j] if x is not None]
    assert abs(sum(valid)) < 1e-9
    spec = json.loads((paths["manifests"] / "nullsize_inputs.json").read_text())
    assert sim.validate_inputs(json.loads(json.dumps(payload)), spec)["status"] == "valid"
    suff = ni.sufficiency(payload)
    assert not suff["sufficient"] and suff["pairs_with_missing_windows"] == 1
    assert suff["missing_reasons"] == {
        "edit:entry_hashes_differ_from_current_bytes": 1,
        "edit:no_primary_record": 1,
    }


def test_too_few_donors_is_insufficient_and_cutpoints_must_match(tmp_path):
    paths, _ = toy(tmp_path)
    payload = ni.materialise(paths)
    payload["donors"] = payload["donors"][:39]
    payload["window_mask"] = payload["window_mask"][:39]
    assert "39 donor pairs < the registered 40" in ni.sufficiency(payload)["problems"][0]
    sel = json.loads((paths["manifests"] / "selection.json").read_text())
    sel["procedural_freeze"]["cutpoints"]["chain"]["keys"][0][1] = "other"
    (paths["manifests"] / "selection.json").write_text(json.dumps(sel))
    with pytest.raises(AssertionError, match="procedural freeze"):
        ni.materialise(paths)


def test_gzip_is_deterministic():
    assert ni.gzip_bytes(b"abc") == ni.gzip_bytes(b"abc")
    assert ni.reason_class("reference 86ea:no_primary_record") == "reference:no_primary_record"


def test_committed_provenance_extract_has_no_numeric_arrays(tmp_path):
    paths, _ = toy(tmp_path)
    payload = ni.materialise(paths)
    ext = ni.inputs_provenance(payload, "0" * 64)
    assert ext["payload_sha256"] == "0" * 64
    assert ext["inventory"] == payload["inventory"] and ext["donors"] == payload["donors"]
    assert ext["provenance"] == payload["provenance"]
    assert not {"centred_noise", "window_mask", "donor_mean"} & set(ext)


def test_records_must_equal_the_closed_snapshot_pins(tmp_path):
    paths, _ = toy(tmp_path)
    evals = paths["evals"]
    primary = {
        d.name: [ni.sha_bytes((d / f).read_bytes()) for f in ("eval.json", "records.jsonl")]
        for d in evals.iterdir()
        if (d / "eval.json").exists()
    }
    snap = {
        "closed_at": "2026-11-08T00:00:00Z",
        "base_ref": "x",
        "content": {
            "tf_v1_records": {"primary": primary, "content_index_sha256": "a" * 64},
            "zoo_bytes": {"content_index_sha256": "b" * 64},
        },
    }
    (paths["manifests"] / "snapshot.json").write_text(json.dumps(snap))
    payload = ni.materialise(paths)
    pins = payload["provenance"]["inputs"]["snapshot"]
    assert pins["records_checked_against_snapshot"] and pins["closed_at"] == snap["closed_at"]
    primary["S03"][1] = "c" * 64  # S03's records changed after closure
    (paths["manifests"] / "snapshot.json").write_text(json.dumps(snap))
    with pytest.raises(AssertionError, match="closed snapshot pin"):
        ni.materialise(paths)


def test_observations_flag_identical_contrasts_vacuous_scenarios_and_single_cells():
    tr = [{"resolved_counts": [3, 3, 3], "agreement_counts": [2, 2, 2]}]
    zero = [{"resolved_counts": [0, 0, 0], "agreement_counts": [0, 0, 0]}]
    stats = {
        "registered": ni.trace_statistics(tr),
        "unavailable_audit_fails": ni.trace_statistics(zero),
    }
    assert stats["registered"]["contrasts_identical"] and stats["registered"]["max_resolved"] == 3
    summary = {
        "registered": {"decisions": {"go": 0, "alternative": 2, "unresolved": 8}, "replicates": 10},
        "unavailable_audit_fails": {"decisions": {"alternative": 0}, "replicates": 1},
    }
    suff = {"donor_cluster_representatives_per_cell": {"LoRA:0": 1, "LoRA:1": 4}}
    obs = ni.observations(summary, stats, suff)
    assert any("one test" in o for o in obs)
    assert any("`unavailable_audit_fails`: no pair resolved" in o for o in obs)
    assert any("'alternative'" in o and "2 of 10" in o for o in obs)
    assert any("LoRA:0 hold a single donor" in o for o in obs)
    stats["registered"] = ni.trace_statistics(
        [{"resolved_counts": [3, 2, 3], "agreement_counts": [2, 2, 2]}]
    )
    assert not any("one test" in o for o in ni.observations(summary, stats, suff))
