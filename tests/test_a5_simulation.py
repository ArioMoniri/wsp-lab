"""Synthetic calibration only: no development or project data access."""

import numpy as np
import pytest

from scripts.a5_simulation import (
    SCALES,
    contrasts,
    cp_interval,
    donor_block,
    donor_cells,
    ordered_decision,
    resolution,
    selection_cells,
    simulate,
)


def toy(n=40, *, noise_sd=0.0, seed=0, mean_range=(1.0, 1.0)):
    """Proxy payload: centred edit-minus-predecessor noise [n, 64], masks and pinned donor means."""
    rng = np.random.default_rng(seed)
    rows = [
        {"id": f"e{i:03}", "component": f"c{i:03}", "family": "LoRA", "norm": i} for i in range(n)
    ]
    raw = rng.normal(size=(n, 64)) * noise_sd + rng.uniform(*mean_range, size=(n, 1))
    payload = {
        "inventory": rows,
        "donors": [r | {"audit": [True] * 3, "audit_evidence": ["computed"] * 3} for r in rows],
        "centred_noise": (raw - raw.mean(axis=1, keepdims=True)).tolist(),
        "window_mask": np.ones((n, 64), dtype=bool).tolist(),
        "donor_mean": raw.mean(axis=1).tolist(),
    }
    return payload, {r["id"]: [1, 1, 1] for r in rows}


def test_cp_thresholds_and_boundaries():
    assert cp_interval(12, 15)[0] == pytest.approx(0.5191, abs=5e-5)
    assert cp_interval(27, 40)[0] == pytest.approx(0.5087, abs=5e-5)
    assert cp_interval(11, 15)[0] < 0.5
    assert cp_interval(26, 40)[0] < 0.5
    assert cp_interval(0, 0) == (0, 1)
    assert cp_interval(0, 15)[0] == 0
    assert cp_interval(15, 15)[1] == 1
    assert ordered_decision([40, 0, 40], [40, 40, 40]) == "alternative"
    assert ordered_decision([27, 30, 30], [40, 40, 40]) == "go"


def test_proxy_mapping_shapes_and_sign_sharing():
    rng = np.random.default_rng(3)
    u = rng.normal(size=64)
    u -= u.mean()
    mask = np.ones(64, dtype=bool)
    block = donor_block(u, mask)
    assert block.shape == (32,)
    np.testing.assert_allclose(block, u.reshape(32, 2).mean(axis=1))
    x = contrasts(block, 0.7)
    assert x.shape == (32, 3)
    # One donor block (mean and noise) shared by the three contrasts, scaled (1, .25, .5).
    np.testing.assert_allclose(x / np.asarray(SCALES), np.tile((0.7 + block)[:, None], 3))
    # The component sign flips mean and noise of all three contrasts together.
    np.testing.assert_allclose(contrasts(block, -0.7, -1), -x)
    # A missing window makes its block missing; it is never imputed.
    mask[5] = False
    assert np.isnan(donor_block(u, mask)[2]) and np.isfinite(donor_block(u, mask)[3])
    with pytest.raises(ValueError, match=r"\[64\]"):
        donor_block(u[:63], mask[:63])

    payload, _ = toy(40, noise_sd=1.0, mean_range=(0.0, 0.3))
    result = simulate(payload, replicates=30, draws=200, quotas={"LoRA": (13, 13, 14)})
    assert result["proxy"]["contrast_scales"] == [1.0, 0.25, 0.5]
    for trace in result["reports"]["null_size"]["traces"]:
        resolved = np.asarray(trace["resolved"])
        assert resolved.shape == (40, 3)
        assert (resolved == resolved[:, :1]).all()  # shared sign and block: identical resolution
        assert len(set(trace["agreement_counts"])) == 1
        assert set(trace["component_signs"]) <= {-1, 1}


def test_null_size_on_synthetic_centred_donors():
    # Centred noise with pinned nonzero donor means: resolution runs the paired bootstrap.
    payload, _ = toy(54, noise_sd=0.5, mean_range=(0.5, 1.0), seed=1)
    null = simulate(payload, replicates=2000, draws=200, quotas={"LoRA": (18, 18, 18)})
    report = null["reports"]["null_size"]
    for rate in report["two_sided_rejection_rates"]:
        assert 0.03 <= rate <= 0.07
    assert min(report["yield_rates"]) > 0.99
    assert report["accepted"]


def test_fair_null_size_and_determinism():
    payload, _ = toy()
    kwargs = {"replicates": 2000, "quotas": {"LoRA": (13, 13, 14)}}
    first = simulate(payload, **kwargs)
    assert first == simulate(payload, **kwargs)
    null = first["reports"]["null_size"]
    assert 0.03 <= null["two_sided_rejection_rates"][0] <= 0.07
    assert null["accepted"]


def test_strong_alternative_power():
    payload, fields = toy(120)
    result = simulate(payload, fields, replicates=1000, quotas={"LoRA": (40, 40, 40)})
    assert result["reports"]["power_yield"]["ordered_go_rate"] > 0.9


def test_masks_and_unavailable_audit_policies():
    payload, _ = toy(40)
    payload["window_mask"][0][7] = False
    payload["centred_noise"][0][7] = None
    payload["donors"][1]["audit_evidence"] = ["unavailable"] * 3
    payload["donors"][1]["audit"] = [False] * 3  # never consulted when evidence is unavailable
    kwargs = {"replicates": 100, "draws": 200, "quotas": {"LoRA": (13, 13, 14)}}
    passed = simulate(payload, **kwargs)["reports"]["null_size"]
    failed = simulate(payload, unavailable_audit="assume_fail", **kwargs)["reports"]["null_size"]
    # Donor e000 (masked window) is missing in every contrast; e001 only under assume_fail.
    for trace in passed["traces"]:
        for donor, ok in zip(trace["donor_representatives"], trace["audit_finite"]):
            assert all(ok) == (donor != "e000")
    for trace in failed["traces"]:
        for donor, ok in zip(trace["donor_representatives"], trace["audit_finite"]):
            assert all(ok) == (donor not in ("e000", "e001"))
    assert sum(failed["audit_or_nonfinite_counts"]) > sum(passed["audit_or_nonfinite_counts"])
    with pytest.raises(ValueError, match="unavailable_audit"):
        simulate(payload, unavailable_audit="ignore", **kwargs)


def test_donor_cell_mapping_and_shortage():
    payload, _ = toy()
    _, edges = selection_cells(payload["inventory"], {"LoRA": (13, 13, 14)})
    extra = dict(payload["donors"][0], id="a", component="c000")
    cells = donor_cells([*payload["donors"], extra], edges)
    assert cells["LoRA", 0][0]["id"] == "a"
    with pytest.raises(ValueError, match="empty donor"):
        donor_cells([extra], edges)


def test_resolution_matches_direct_paired_bootstrap():
    blocks = np.random.default_rng(8).normal(size=(32, 2, 3))
    actual = resolution(blocks, np.random.default_rng(9))
    indices = np.random.default_rng(9).integers(0, 32, size=(4000, 32))
    expected = np.quantile(blocks[indices].mean(axis=1), [0.025, 0.975], axis=0)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_constant_positive_mean_scenario_replaces_means_only():
    from scripts.a5_simulation import SCENARIOS, family_constants

    payload, _ = toy(40, noise_sd=0.3, mean_range=(-1.0, 1.0), seed=4)
    kwargs = {"replicates": 50, "draws": 200, "quotas": {"LoRA": (13, 13, 14)}}
    base = simulate(payload, **kwargs)
    const = simulate(payload, **kwargs, **SCENARIOS["constant_positive_mean"])
    expected = float(np.mean(np.abs(payload["donor_mean"])))
    assert const["sensitivity"]["family_constant_means"]["LoRA"] == pytest.approx(expected)
    assert "sensitivity" not in base  # earlier raw outputs keep their exact schema
    # Same streams: identical donor draws and component signs; only the means differ.
    for a, b in zip(
        base["reports"]["null_size"]["traces"], const["reports"]["null_size"]["traces"]
    ):
        assert a["donor_representatives"] == b["donor_representatives"]
        assert a["component_signs"] == b["component_signs"]
    _, edges = selection_cells(payload["inventory"], {"LoRA": (13, 13, 14)})
    cells = donor_cells(payload["donors"], edges)
    zero = {r["id"]: (None, 0.0, None) for r in payload["donors"]}
    with pytest.raises(ValueError, match="positive donor mean"):
        family_constants(cells, zero)
    with pytest.raises(ValueError, match="mean_mode"):
        simulate(payload, mean_mode="median", **kwargs)


def test_cell_mask_noise_scenario_masks_extra_slots_and_scales_cell_noise():
    payload, _ = toy(40, noise_sd=0.3, mean_range=(0.2, 0.4), seed=5)
    quotas = {"LoRA": (13, 13, 14)}
    kwargs = {"replicates": 40, "draws": 200, "quotas": quotas}
    override = {("LoRA", 0): {"active_slots": 1, "noise_scale": 2.0}}
    base = simulate(payload, **kwargs)
    cell = simulate(payload, cell_overrides=override, **kwargs)
    params = cell["sensitivity"]["cell_overrides"][0]
    assert params["slots"] == 13 and params["active_slots"] == 1
    assert cell["sensitivity"]["active_slots"] == 40 - 12
    for a, b in zip(base["reports"]["null_size"]["traces"], cell["reports"]["null_size"]["traces"]):
        assert a["donor_representatives"] == b["donor_representatives"]
        assert a["component_signs"] == b["component_signs"]
        ok = np.asarray(b["audit_finite"])
        assert ok[0].all() and not ok[1:13].any() and ok[13:].all()
        assert np.asarray(a["resolved"])[13:].tolist() == np.asarray(b["resolved"])[13:].tolist()
    assert cell["reports"]["null_size"]["audit_or_nonfinite_counts"] == [12 * 40] * 3
    with pytest.raises(ValueError, match="not a selection cell"):
        simulate(payload, cell_overrides={("chain", 0): {"active_slots": 1}}, **kwargs)


def test_donor_ids_restrict_power_pool_only():
    payload, fields = toy(60, noise_sd=0.3, mean_range=(0.2, 0.4), seed=6)
    quotas = {"LoRA": (5, 5, 5)}
    kwargs = {"replicates": 30, "draws": 200, "quotas": quotas}
    _, edges = selection_cells(payload["inventory"], quotas)
    cells = donor_cells(payload["donors"], edges)
    # Keep two donors per cell (one cell keeps a single donor).
    keep = [r["id"] for t in range(3) for r in cells["LoRA", t][: 1 if t == 0 else 2]]
    full = simulate(payload, fields, **kwargs)
    part = simulate(payload, fields, donor_ids=keep, **kwargs)
    # NULL-SIZE uses the full pool and is reproduced exactly; only POWER-AND-YIELD is restricted.
    assert part["reports"]["null_size"] == full["reports"]["null_size"]
    assert "donor_pool" not in full
    for trace in part["reports"]["power_yield"]["traces"]:
        assert set(trace["donor_representatives"]) <= set(keep)
    pool = part["donor_pool"]
    assert pool["n_ids"] == 5 and pool["applies_to"] == "power_yield"
    assert {k: v["donor_rows"] for k, v in pool["cells"].items()} == {
        "LoRA:0": 1,
        "LoRA:1": 2,
        "LoRA:2": 2,
    }
    assert pool["cells"]["LoRA:0"]["full_pool_components"] == len(cells["LoRA", 0])
    # The restriction is deterministic and the ID order does not matter.
    assert simulate(payload, fields, donor_ids=keep[::-1], **kwargs) == part


def test_donor_ids_cell_components_and_validation():
    payload, fields = toy(60, seed=7)
    quotas = {"LoRA": (5, 5, 5)}
    kwargs = {"replicates": 5, "draws": 50, "quotas": quotas}
    _, edges = selection_cells(payload["inventory"], quotas)
    cells = donor_cells(payload["donors"], edges)
    keep = [cells["LoRA", t][0]["id"] for t in range(3)]
    # A second row of an already listed component in the same cell adds a row, not a component.
    twin = dict(payload["donors"][0], id="e000b", component=cells["LoRA", 0][0]["component"])
    twin_row = dict(payload["inventory"][0], id="e000b", component=twin["component"])
    payload["inventory"].append(twin_row)
    payload["donors"].append(twin)
    payload["centred_noise"].append(payload["centred_noise"][0])
    payload["window_mask"].append(payload["window_mask"][0])
    payload["donor_mean"].append(payload["donor_mean"][0])
    fields["e000b"] = [1, 1, 1]
    result = simulate(payload, fields, donor_ids=[*keep, "e000b"], **kwargs)
    assert result["donor_pool"]["cells"]["LoRA:0"] == {
        "donor_rows": 2,
        "donor_components": 1,
        "full_pool_components": len(cells["LoRA", 0]),
    }
    with pytest.raises(ValueError, match=r"empty donor cell\(s\): LoRA:2"):
        simulate(payload, fields, donor_ids=keep[:2], **kwargs)
    with pytest.raises(ValueError, match="not in the payload"):
        simulate(payload, fields, donor_ids=[*keep, "zzz"], **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        simulate(payload, fields, donor_ids=[*keep, keep[0]], **kwargs)
    with pytest.raises(ValueError, match="without a field prediction"):
        simulate(
            payload, {k: v for k, v in fields.items() if k != keep[1]}, donor_ids=keep, **kwargs
        )
    with pytest.raises(ValueError, match="requires fields"):
        simulate(payload, donor_ids=keep, **kwargs)
    payload["donors"][1]["eligible"] = False
    with pytest.raises(ValueError, match="ineligible"):
        simulate(payload, fields, donor_ids=[*keep, payload["donors"][1]["id"]], **kwargs)


def two_family_toy(n_lora=30, n_chain=12):
    """Registered-quota toy: LoRA rows e*, chain rows k*, one component per row."""
    payload, fields = toy(n_lora, seed=9)
    rows = [
        {"id": f"k{i:03}", "component": f"kc{i:03}", "family": "chain", "norm": i}
        for i in range(n_chain)
    ]
    payload["inventory"] += rows
    payload["donors"] += [
        r | {"audit": [True] * 3, "audit_evidence": ["computed"] * 3} for r in rows
    ]
    payload["centred_noise"] += [[0.0] * 64] * n_chain
    payload["window_mask"] += [[True] * 64] * n_chain
    payload["donor_mean"] += [1.0] * n_chain
    fields |= {r["id"]: [1, 1, 1] for r in rows}
    return payload, fields


def test_donor_ids_cli_records_pool_and_provenance(tmp_path, monkeypatch):
    import functools
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path

    from scripts import a5_simulation

    payload, fields = two_family_toy()
    _, edges = selection_cells(payload["inventory"])
    cells = donor_cells(payload["donors"], edges)
    keep = sorted(r["id"] for key in cells for r in cells[key][:2])
    paths = {}
    for name, obj in (("inputs", payload), ("fields", fields), ("ids", {"donor_ids": keep})):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(obj))
    out = tmp_path / "o.json"
    script = Path(__file__).resolve().parents[1] / "scripts/a5_simulation.py"
    base = [sys.executable, str(script), str(paths["inputs"]), "--output", str(out)]
    # Without --fields, --donor-ids is refused before any simulation.
    refused = subprocess.run(
        [*base, "--donor-ids", str(paths["ids"])], capture_output=True, check=False
    )
    assert refused.returncode != 0 and b"requires --fields" in refused.stderr
    # An empty cell is a hard error raised before the registered-scale simulation starts.
    paths["bad"] = tmp_path / "bad.json"
    paths["bad"].write_text(json.dumps([i for i in keep if i.startswith("e")]))
    bad = subprocess.run(
        [*base, "--fields", str(paths["fields"]), "--donor-ids", str(paths["bad"])],
        capture_output=True,
        check=False,
    )
    assert bad.returncode != 0 and b"empty donor cell(s): chain:0, chain:1, chain:2" in bad.stderr
    assert not out.exists()
    # A short in-process run (synthetic scale) records the pool, its file and its hash.
    small = functools.partial(a5_simulation.simulate, replicates=3, draws=20)
    monkeypatch.setattr(a5_simulation, "simulate", small)
    argv = [*base[1:], "--fields", str(paths["fields"]), "--donor-ids", str(paths["ids"])]
    monkeypatch.setattr(sys, "argv", argv)
    assert a5_simulation.main() == 0
    result = json.loads(out.read_text())
    assert result["donor_pool"]["path"] == str(paths["ids"])
    assert result["donor_pool"]["n_ids"] == len(keep) == 12
    assert (
        result["input_provenance"][str(paths["ids"])]
        == hashlib.sha256(paths["ids"].read_bytes()).hexdigest()
    )
    for trace in result["reports"]["power_yield"]["traces"]:
        assert set(trace["donor_representatives"]) <= set(keep)
