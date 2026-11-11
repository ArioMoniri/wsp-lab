"""F12d: Board A F12c condition 6 ("Missing donor flag"): restricted-payload re-extraction with the original
``eligible`` flag, the eligibility-requiring loader and the raw-load barrier for configured execution.

Toy payloads written to tmp_path only (synthetic numbers with sentinel non-allowlisted values); no outcome, no
data/evals, no raw F8a.2 payload. The committed-pin test reads only the pinned restricted copy and skips when absent."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import f12_payload_extract as px
from scripts import f12_simulation as fs

SENTINEL = 987654.321  # a donor_mean value that must never reach the restricted copy


def _raw_payload(n_comp=3, per=2, *, eligible=None, mask_holes=(), extra_donor=True, seed=0):
    """A toy raw payload with the F8a.2 layout: allowlisted fields plus donor_mean, inventory,
    provenance and non-allowlisted donor keys (norm, audit)."""
    rng = np.random.default_rng(seed)
    donors, noise, masks = [], [], []
    for c in range(n_comp):
        for j in range(per):
            i = len(donors)
            d = {"id": f"e{c}{j}", "component": f"pc{c}", "family": "chain", "norm": 1.5 + i,
                 "audit": [False, False, False], "audit_evidence": ["unavailable"] * 3}  # fmt: skip
            d["eligible"] = True if eligible is None else eligible[i]
            if not extra_donor:
                d = {k: d[k] for k in ("id", "component", "family", "eligible")}
            donors.append(d)
            v = rng.normal(size=64) * 0.01
            v = (v - v.mean()).tolist()
            m = [True] * 64
            if i in mask_holes:
                v[5], m[5] = None, False
            noise.append(v)
            masks.append(m)
    return {
        "schema": "a5-nullsize-inputs-v2",
        "spec": {"path": "toy", "sha256": "0" * 64, "schema": "toy"},
        "donor_source": "tf_v1_edit_minus_actual_predecessor",
        "window_ids": [str(w) for w in range(64)],
        "inventory": [{k: d[k] for k in ("id", "component", "family", "eligible")} for d in donors],
        "donors": donors,
        "centred_noise": noise,
        "window_mask": masks,
        "donor_mean": [SENTINEL] * len(donors),
        "provenance": {"donors": {d["id"]: {"reference_id": "base"} for d in donors}},
    }


def _write_raw(tmp_path, obj, name="raw.json"):
    p = tmp_path / name
    p.write_bytes(json.dumps(obj).encode())
    return p


def _extract(path, **kw):
    import hashlib

    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return px.extract(path, expected_sha256=sha, **kw)


# ------------------------------------------------------------------- extraction


def test_extract_keeps_only_allowlisted_fields_with_original_eligible_flag(tmp_path):
    raw = _raw_payload()
    path = _write_raw(tmp_path, raw)
    restricted, report = _extract(path)
    assert set(restricted) == {"schema", "provenance", "window_ids", "donors", "centred_noise",
                               "window_mask"}  # fmt: skip
    assert restricted["schema"] == px.SCHEMA == fs.NoisePayload.RESTRICTED_SCHEMA
    for d in restricted["donors"]:
        assert set(d) == {"id", "component", "family", "eligible"} and d["eligible"] is True
    blob = json.dumps(restricted)
    assert str(SENTINEL) not in blob and 'donor_mean": [' not in blob
    assert "norm" not in blob and "audit_evidence" not in blob and "inventory" not in blob
    assert restricted["centred_noise"] == raw["centred_noise"]
    assert restricted["window_ids"] == raw["window_ids"]
    # the report names what was not selected, by key only (no values)
    assert set(report["not_selected_top_level_keys"]) == {"schema", "spec", "donor_source",
                                                          "inventory", "donor_mean", "provenance"}  # fmt: skip
    assert str(SENTINEL) not in json.dumps(report)
    assert report["counts"] == {"source_donors": 6, "flag_true": 6, "flag_false": 0,
                                "incomplete_windows": 0, "used": 6, "used_components": 3}  # fmt: skip


def test_non_allowlisted_values_are_skipped_without_decoding(tmp_path):
    raw = json.dumps(_raw_payload())
    # an undecodable token inside donor_mean proves the scanner never decodes that value
    bad = raw.replace(f"[{SENTINEL}, ", "[@@not-json@@, ", 1)
    assert bad != raw
    path = tmp_path / "raw.json"
    path.write_text(bad)
    restricted, _ = _extract(path)
    assert len(restricted["donors"]) == 6
    with pytest.raises(json.JSONDecodeError):
        json.loads(bad)


def test_source_digest_is_checked_before_any_parse(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text("{not json at all")
    with pytest.raises(px.ExtractionError, match="digest"):
        px.extract(path, expected_sha256="0" * 64)


def test_missing_or_non_boolean_eligible_flag_fails_closed(tmp_path):
    raw = _raw_payload()
    del raw["donors"][2]["eligible"]
    with pytest.raises(px.ExtractionError, match="eligible"):
        _extract(_write_raw(tmp_path, raw))
    raw = _raw_payload()
    raw["donors"][1]["eligible"] = 1  # truthy but not the boolean flag: never coerced
    with pytest.raises(px.ExtractionError, match="eligible"):
        _extract(_write_raw(tmp_path, raw, "r2.json"))


def test_registered_rule_applied_explicitly_and_exclusions_disclosed(tmp_path):
    elig = [True, False, True, True, True, True]
    raw = _raw_payload(eligible=elig, mask_holes=(4,))
    restricted, report = _extract(_write_raw(tmp_path, raw))
    ids = [d["id"] for d in restricted["donors"]]
    assert ids == ["e00", "e10", "e11", "e21"]  # e01 flag false; e20 has a masked window
    assert report["excluded"] == {"flag_false": ["e01"], "incomplete_windows": ["e20"]}
    assert report["counts"]["used"] == 4 and report["counts"]["flag_false"] == 1
    assert report["rule"] == px.ELIGIBILITY_RULE and "never derived" in px.ELIGIBILITY_RULE
    # a flag-false donor with finite complete windows is still excluded: finiteness never grants eligibility
    assert all(v is not None for v in raw["centred_noise"][1])


def test_mask_noise_inconsistency_and_window_alignment_fail_closed(tmp_path):
    raw = _raw_payload()
    raw["window_mask"][0][3] = False  # masked but a finite value on file
    with pytest.raises(px.ExtractionError, match="mask"):
        _extract(_write_raw(tmp_path, raw))
    raw = _raw_payload()
    raw["window_ids"][7] = "6"
    with pytest.raises(px.ExtractionError, match="window"):
        _extract(_write_raw(tmp_path, raw, "w.json"))
    raw = _raw_payload()
    raw["centred_noise"][0] = raw["centred_noise"][0][:63]
    with pytest.raises(px.ExtractionError, match="64"):
        _extract(_write_raw(tmp_path, raw, "s.json"))


def test_reference_cross_check_of_donor_identities_components_and_windows(tmp_path):
    raw = _raw_payload()
    path = _write_raw(tmp_path, raw)
    ref = {"donors": [{k: d[k] for k in ("id", "component", "family", "eligible")} for d in raw["donors"]],
           "window_ids": raw["window_ids"]}  # fmt: skip
    _, report = _extract(path, reference=ref)
    assert (
        report["reference_check"]
        == "donor ids, components, families, eligible flags and window ids equal"
    )
    bad = copy.deepcopy(ref)
    bad["donors"][3]["component"] = "pc9"
    with pytest.raises(px.ExtractionError, match="reference"):
        _extract(path, reference=bad)


def test_membership_diff_against_the_superseded_copy(tmp_path):
    raw = _raw_payload()
    new, _ = _extract(_write_raw(tmp_path, raw))
    old = copy.deepcopy(new)
    old["schema"] = "f12-noise-payload-restricted-v1"
    for d in old["donors"]:
        del d["eligible"]
    diff = px.membership_diff(old, new)
    assert diff["n_donors_old"] == diff["n_donors_new"] == 6
    assert diff["donors_added"] == [] and diff["donors_removed"] == []
    assert diff["components_changed"] == [] and diff["unchanged"] is True
    assert diff["centred_noise_identical"] and diff["window_mask_identical"]
    old["donors"].pop()
    old["centred_noise"].pop()
    old["window_mask"].pop()
    diff = px.membership_diff(old, new)
    assert diff["donors_added"] == ["e21"] and diff["unchanged"] is False


# ------------------------------------------------------------------------ loader


def _v2_copy(tmp_path, raw=None):
    raw = raw or _raw_payload()
    restricted, _ = _extract(_write_raw(tmp_path, raw))
    path = tmp_path / "restricted_v2.json"
    path.write_bytes(px.serialise(restricted))
    pin = fs.restricted_pin(path, extractor={"script": "scripts/f12_payload_extract.py", "commit": "0" * 40,
                                             "sha256": "1" * 64, "functions": "extract"})  # fmt: skip
    return path, pin


def test_loader_requires_the_eligible_flag(tmp_path):
    path, pin = _v2_copy(tmp_path)
    pay = fs.NoisePayload.load_restricted(path, pin)
    assert all(d["eligible"] is True for d in pay.donors) and len(pay.donors) == 6
    assert pin["schema"] == fs.RESTRICTED_PIN_SCHEMA and pin["contents"]["eligible_flags"] == {
        "true": 6
    }
    obj = json.loads(path.read_bytes())
    for mutate, match in (
        (lambda o: o["donors"][0].pop("eligible"), "eligible"),
        (lambda o: o["donors"][0].update(eligible=False), "eligible"),
        (lambda o: o["donors"][0].update(eligible="true"), "eligible"),
    ):
        bad = copy.deepcopy(obj)
        mutate(bad)
        p2 = tmp_path / "bad.json"
        p2.write_bytes(json.dumps(bad, sort_keys=True).encode())
        pin2 = fs.restricted_pin(p2, extractor=pin["extractor"])
        with pytest.raises(ValueError, match=match):
            fs.NoisePayload.load_restricted(p2, pin2)


def test_loader_refuses_the_superseded_v1_copy_and_pin(tmp_path):
    path, pin = _v2_copy(tmp_path)
    obj = json.loads(path.read_bytes())
    obj["schema"] = "f12-noise-payload-restricted-v1"
    for d in obj["donors"]:
        del d["eligible"]
    p1 = tmp_path / "v1.json"
    p1.write_bytes(json.dumps(obj, sort_keys=True).encode())
    pin1 = fs.restricted_pin(p1, extractor=pin["extractor"])
    with pytest.raises(ValueError, match="superseded|eligible"):
        fs.NoisePayload.load_restricted(p1, pin1)
    old_pin = copy.deepcopy(pin)
    old_pin["schema"] = "f12c-restricted-payload-pin-v1"
    with pytest.raises(ValueError, match="pin schema"):
        fs.NoisePayload.load_restricted(path, old_pin)


def test_constructor_refuses_a_donor_flagged_ineligible():
    noise = np.zeros((2, 64))
    noise[:, ::2], noise[:, 1::2] = 0.01, -0.01
    donors = [{"id": "a", "component": "c", "family": "chain", "eligible": True},
              {"id": "b", "component": "c", "family": "chain", "eligible": False}]  # fmt: skip
    with pytest.raises(ValueError, match="eligible"):
        fs.NoisePayload(noise, np.ones((2, 64), bool), donors)


# ------------------------------------------------------------- raw-load barrier


def test_raw_loader_refuses_without_explicit_ungoverned_opt_in(tmp_path):
    p = _write_raw(tmp_path, _raw_payload())
    with pytest.raises(ValueError, match="raw"):
        fs.NoisePayload.load(p, expected_sha256=None)


class _Stop(Exception):
    pass


@pytest.mark.parametrize("name", sorted(fs.F12C_CONFIGS) + ["f12b_stage1", "f12b_stage2"])
def test_configured_execution_can_never_load_the_raw_payload(tmp_path, monkeypatch, name):
    path, pin = _v2_copy(tmp_path)
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin))
    monkeypatch.setattr(fs, "RESTRICTED_PIN", pin_path)
    reads = []
    real_read_bytes = Path.read_bytes

    def spy(self):
        reads.append(Path(self).resolve())
        return real_read_bytes(self)

    def no_raw(*a, **k):
        raise AssertionError("raw payload loader called in configured execution")

    def stop(*a, **k):
        raise _Stop()

    monkeypatch.setattr(Path, "read_bytes", spy)
    monkeypatch.setattr(fs.NoisePayload, "load", classmethod(no_raw))
    monkeypatch.setattr(fs, "Context", stop)
    monkeypatch.setattr(fs, "noise_check", stop)
    cfg = fs.F12C_CONFIGS.get(name) or fs.F12B_CONFIGS[name.split("_", 1)[1]]
    base = ["--synthetic", "--out", str(tmp_path / "o"), "--restricted-payload", str(path)]
    with pytest.raises((SystemExit, _Stop)):
        fs.main(["run", *base, "--config", str(cfg)])
    with pytest.raises(SystemExit, match="raw"):  # an explicit raw payload is refused with a config
        fs.main(["run", *base, "--config", str(cfg), "--payload", str(tmp_path / "raw.json")])
    with pytest.raises((SystemExit, _Stop)):
        fs.main(
            [
                "noisecheck",
                "--synthetic",
                "--restricted-payload",
                str(path),
                "--out",
                str(tmp_path / "n"),
            ]
        )
    raw_paths = {fs.DEFAULT_PAYLOAD.resolve(), (fs.ROOT / px.SOURCE_REL).resolve()}
    assert not raw_paths & set(reads)


# ------------------------------------------------------------------ committed pin


def test_committed_v2_pin_supersedes_the_old_record():
    pin = json.loads(fs.RESTRICTED_PIN.read_text())
    assert pin["schema"] == fs.RESTRICTED_PIN_SCHEMA
    assert pin["status"] == "draft, pending Board A C1 verification"
    assert pin["source"]["sha256"] == fs.PAYLOAD_SHA256 == px.SOURCE_SHA256
    assert pin["source"]["path"] == px.SOURCE_REL
    assert pin["extractor"]["script"] == "scripts/f12_payload_extract.py"
    assert len(pin["extractor"]["commit"]) == 40 and pin["extractor"]["functions"]
    assert pin["eligibility"]["rule"] == px.ELIGIBILITY_RULE
    old = pin["superseded"][0]
    assert old["record"]["restricted_copy"]["sha256"] == (
        "a21f426f0f8aef616d4bedca19603ed4cf8b7acd4823383acf749ba3d8f4a972"
    )
    assert "diagnostics of the old bytes" in old["status"]
    assert set(pin["membership_vs_superseded"]) >= {"n_donors_old", "n_donors_new", "donors_added",
                                                    "donors_removed", "components_changed", "unchanged"}  # fmt: skip
    path = Path(pin["restricted_copy"]["path"])
    if not path.exists():
        pytest.skip("restricted copy absent")
    pay = fs.NoisePayload.load_restricted(path, pin)
    assert len(pay.donors) == pin["contents"]["n_donors"]
    assert len(pay.members) == pin["contents"]["n_components"]
    assert all(d["eligible"] is True for d in pay.donors)
