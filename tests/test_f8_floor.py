"""F8a floor (spec §2.3, PREREG F8a.2) on synthetic payloads, plus the committed payload copy.

Synthetic payloads are generated in tmp_path only; nothing is written under data/ or docs/.
"""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from scripts import a5_simulation as sim
from scripts import f8_floor as ff

ROOT = ff.ROOT
COMMITTED = ROOT / "docs/prereg/f8_manifests/floor_payload/nullsize_inputs.json"


def payload(components, *, seed=0, sd=0.01, mask_out=(), nonfinite=(), ineligible=()):
    """Synthetic NULL-SIZE-shaped payload; ``components`` lists the component of each donor row."""
    rng = np.random.default_rng(seed)
    n = len(components)
    raw = rng.normal(size=(n, 64)) * sd * (1 + np.arange(n)[:, None])
    noise = raw - raw.mean(axis=1, keepdims=True)
    mask = np.ones((n, 64), dtype=bool)
    noise_json = noise.tolist()
    for j in mask_out:
        mask[j, 5] = False
        noise_json[j][5] = None
    for j in nonfinite:
        noise_json[j][7] = None  # mask stays true, value missing: nonfinite
    donors = [
        {
            "id": f"d{j:03}",
            "component": c,
            "family": "LoRA" if j % 2 else "chain",
            "eligible": j not in ineligible,
            "norm": 1.0,
            "audit": [False] * 3,
            "audit_evidence": ["unavailable"] * 3,
        }
        for j, c in enumerate(components)
    ]
    return {
        "centred_noise": noise_json,
        "window_mask": mask.tolist(),
        "donors": donors,
        "donor_mean": [0.0] * n,
        "donor_source": sim.PROXY_SOURCE,
        "schema": "a5-nullsize-inputs-v2",
    }


def test_seed_is_the_registered_domain_seed():
    assert ff.domain_seed("20261111:f8-floor:", "resolution") == 1731840362851662234
    assert ff.FLOOR_SEED == 1731840362851662234


def test_per_pair_half_widths_are_one_common_resolution_call():
    p = payload(["a", "a", "b"])
    out = ff.compute_floor(p)
    noise = np.asarray(p["centred_noise"], dtype=float)
    blocks = np.stack([sim.donor_block(noise[j], np.ones(64, bool), 1.0) for j in range(3)], 1)
    rng = np.random.Generator(np.random.PCG64(ff.FLOOR_SEED))
    lo, hi = sim.resolution(blocks, rng, draws=4000)
    expected = (hi - lo) / 2
    got = [r["h"] for r in out["pairs"]]
    assert got == pytest.approx(expected.tolist(), rel=0, abs=0)
    assert [r["id"] for r in out["pairs"]] == ["d000", "d001", "d002"]


def test_component_median_of_medians_and_even_count():
    p = payload(["a", "a", "a", "b", "c", "c"])
    out = ff.compute_floor(p)
    h = {r["id"]: r["h"] for r in out["pairs"]}
    hc = {
        "a": float(np.median([h["d000"], h["d001"], h["d002"]])),
        "b": h["d003"],
        "c": float(np.median([h["d004"], h["d005"]])),
    }
    assert out["components"] == {k: pytest.approx(v, abs=0) for k, v in hc.items()}
    assert out["h_star"] == float(np.median(list(hc.values())))
    # even number of components: mean of the two middle values
    p4 = payload(["a", "b", "c", "d"], seed=3)
    out4 = ff.compute_floor(p4)
    vals = sorted(out4["components"].values())
    assert out4["h_star"] == pytest.approx((vals[1] + vals[2]) / 2, abs=0)


def test_pair_weighted_alternative_is_labelled_non_binding():
    p = payload(["a", "a", "a", "b"])
    out = ff.compute_floor(p)
    alt = out["sensitivities"]["pair_weighted_median"]
    assert alt["binding"] is False
    assert alt["value"] == float(np.median([r["h"] for r in out["pairs"]]))
    for key, sens in out["sensitivities"].items():
        if isinstance(sens, dict) and "binding" in sens:
            assert sens["binding"] is False, key


def test_missingness_excludes_and_counts_never_imputes():
    comps = ["a", "b", "c", "d", "e"]
    out = ff.compute_floor(payload(comps, mask_out=[1], nonfinite=[2], ineligible=[3]))
    assert [r["id"] for r in out["pairs"]] == ["d000", "d004"]
    assert out["counts"]["excluded_masked_or_nonfinite"] == 2
    assert out["counts"]["excluded_ineligible"] == 1
    assert sorted(out["counts"]["excluded_ids"]) == ["d001", "d002", "d003"]
    assert out["counts"]["pairs"] == 2 and out["counts"]["components"] == 2


def test_exclusion_does_not_change_the_kept_donors_draws():
    """Common index array: kept donors are resolved together in payload row order."""
    full = ff.compute_floor(payload(["a", "b", "c"]))
    dropped = ff.compute_floor(payload(["a", "b", "c"], mask_out=[1]))
    assert dropped["pairs"][0]["h"] == full["pairs"][0]["h"]
    assert dropped["pairs"][1]["h"] == full["pairs"][2]["h"]


def test_deterministic():
    p = payload(["a", "b", "b", "c"], seed=7)
    assert ff.compute_floor(p) == ff.compute_floor(p)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(centred_noise=[r[:63] for r in p["centred_noise"]]),
        lambda p: p.update(window_mask=p["window_mask"][:-1]),
        lambda p: p["donors"].pop(),
        lambda p: p.update(donor_source="something_else"),
    ],
)
def test_schema_violations_fail_closed(mutate):
    p = payload(["a", "b", "c"])
    mutate(p)
    with pytest.raises(ValueError):
        ff.compute_floor(p)


def test_no_eligible_donor_fails_closed():
    with pytest.raises(ValueError):
        ff.compute_floor(payload(["a", "b"], ineligible=[0, 1]))


def test_cli_refuses_unpinned_payload(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(payload(["a", "b"])))
    out = tmp_path / "floor.json"
    with pytest.raises(SystemExit):
        ff.main(["--payload", str(path), "--output", str(out)])
    assert not out.exists()


def test_cli_report_carries_provenance(tmp_path, capsys):
    path = tmp_path / "p.json"
    path.write_bytes(json.dumps(payload(["a", "b", "b"])).encode())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    out = tmp_path / "floor.json"
    ff.main(["--payload", str(path), "--expected-sha256", sha, "--output", str(out)])
    report = json.loads(out.read_text())
    assert report["payload"]["sha256"] == sha
    assert report["payload"]["pinned"] is False
    assert set(report["code"]) >= {"scripts/f8_floor.py", "scripts/a5_simulation.py"}
    for v in report["code"].values():
        assert len(v) == 64
    assert report["environment"]["numpy"] == np.__version__
    assert report["units"].startswith("nats per valid next token")
    assert report["rng"]["seed"] == ff.FLOOR_SEED and report["rng"]["draws"] == 4000
    assert len(report["rules"]) >= 8
    assert report["h_star"] == pytest.approx(
        ff.compute_floor(json.loads(path.read_text()))["h_star"]
    )
    assert "h*" in capsys.readouterr().out
    # never NaN/Infinity in the report
    json.loads(out.read_text(), parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))


@pytest.mark.skipif(not COMMITTED.exists(), reason="committed floor payload copy absent")
def test_committed_payload_reproduces_design_preview(tmp_path):
    out = tmp_path / "floor.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/f8_floor.py"),
            "--payload",
            str(COMMITTED),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    report = json.loads(out.read_text())
    assert report["payload"]["sha256"] == ff.PINNED_PAYLOAD_SHA256
    assert report["payload"]["pinned"] is True
    assert report["counts"]["pairs"] == 141 and report["counts"]["components"] == 63
    assert report["counts"]["excluded_masked_or_nonfinite"] == 0
    # Board A: compare the preview at its stated printed precision.
    assert f"{report['h_star']:.7f}" == "0.0049397"
    assert report["preview_comparison"]["h_star"]["agrees_at_printed_precision"] is True
    assert "h*" in proc.stdout
