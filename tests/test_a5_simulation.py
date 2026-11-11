"""Synthetic calibration only; the F11.4 tests also read two committed docs/prereg manifests."""

from pathlib import Path

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


# --- F10.6 slot mask (Board A F10 condition 5) ------------------------------------------------


def selected_ids(payload, quotas=None):
    """Selected edit ids in frozen slot order (the ID mapping of ``--slot-mask``)."""
    selected, _ = selection_cells(payload["inventory"], quotas)
    return [r["id"] for rows in selected.values() for r in rows]


def rotation_mask(ids, masked):
    from scripts.a5_simulation import SLOT_MASK_SCHEMA

    full = ["rotation", "ladder_0.25", "ladder_0.5"]
    return {
        "schema": SLOT_MASK_SCHEMA,
        "available": {i: full[1:] if i in masked else full for i in ids},
    }


def test_slot_mask_masks_rotation_slots_and_keeps_ladders():
    payload, fields = toy(40, noise_sd=0.3, mean_range=(0.2, 0.4), seed=11)
    quotas = {"LoRA": (13, 13, 14)}
    kwargs = {"replicates": 30, "draws": 200, "quotas": quotas}
    ids = selected_ids(payload, quotas)
    masked = set(ids[::4]) | set(ids[1::4]) | set(ids[2::4])  # 30 of 40 rotation slots
    assert len(masked) == 30
    positions = np.asarray([i in masked for i in ids])
    base = simulate(payload, fields, **kwargs)
    slot = simulate(payload, fields, slot_mask=rotation_mask(ids, masked), **kwargs)
    for mode in ("null_size", "power_yield"):
        b, s = base["reports"][mode], slot["reports"][mode]
        for tb, ts in zip(b["traces"], s["traces"]):
            # Same streams, draws and signs: the mask removes pairs, it never resamples.
            assert tb["donor_representatives"] == ts["donor_representatives"]
            assert tb["component_signs"] == ts["component_signs"]
            rb, rs = np.asarray(tb["resolved"]), np.asarray(ts["resolved"])
            # Ladder slots of masked edits are retained unchanged.
            assert (rb[:, 1:] == rs[:, 1:]).all()
            assert tb["resolved_counts"][1:] == ts["resolved_counts"][1:]
            assert tb["agreement_counts"][1:] == ts["agreement_counts"][1:]
            # Masked rotation slots contribute no pair; kept rotation slots are unchanged.
            assert not rs[positions, 0].any()
            assert (rs[~positions, 0] == rb[~positions, 0]).all()
            assert ts["resolved_counts"][0] <= 10
            ok = np.asarray(ts["audit_finite"])
            assert not ok[positions, 0].any() and ok[:, 1:].all()
        assert s["yield_rates"][0] == 0.0 and s["ordered_go_rate"] == 0.0
        assert s["yield_rates"][1:] == b["yield_rates"][1:]
        assert s["two_sided_rejection_rates"][1:] == b["two_sided_rejection_rates"][1:]
        assert s["decisions"]["alternative"] == b["decisions"]["alternative"]  # step 2 stays active
        assert s["audit_or_nonfinite_counts"] == [
            b["audit_or_nonfinite_counts"][0] + 30 * 30,
            *b["audit_or_nonfinite_counts"][1:],
        ]
    report = slot["slot_mask"]
    assert report["label"] == "slot-masked (F10.6)"
    assert report["available_slots"] == {"rotation": 10, "ladder_0.25": 40, "ladder_0.5": 40}
    assert report["masked_ids"]["rotation"] == sorted(masked)
    assert report["masked_ids"]["ladder_0.25"] == report["masked_ids"]["ladder_0.5"] == []
    assert report["applies_to"] == ["null_size", "power_yield"]
    assert [row["edit_id"] for row in report["slots"]] == ids
    assert [row["slot"] for row in report["slots"]] == list(range(40))
    assert report["slots"][0]["cell"] == "LoRA:0"
    assert report["cells"]["LoRA:0"]["slots"] == 13
    assert sum(c["available"]["rotation"] for c in report["cells"].values()) == 10
    gates = report["gates"]
    assert gates["label"] == "slot-masked (F10.6)"
    assert gates["power"]["ordered_go_rate"] == 0.0 and gates["power"]["threshold"] == 0.80
    assert gates["power"]["passed"] is False and gates["passed"] is False


def test_slot_mask_all_available_reproduces_unmasked_run():
    payload, fields = toy(40, noise_sd=0.3, mean_range=(0.2, 0.4), seed=12)
    quotas = {"LoRA": (13, 13, 14)}
    kwargs = {"replicates": 25, "draws": 200, "quotas": quotas}
    ids = selected_ids(payload, quotas)
    for f in (None, fields):
        base = simulate(payload, f, **kwargs)
        slot = simulate(payload, f, slot_mask=rotation_mask(ids, set()), **kwargs)
        assert "slot_mask" not in base  # earlier raw outputs keep their exact schema
        report = slot.pop("slot_mask")
        assert slot == base
        assert report["available_slots"] == {"rotation": 40, "ladder_0.25": 40, "ladder_0.5": 40}
    # The mask composes with the F7 donor-pool restriction without changing it.
    _, edges = selection_cells(payload["inventory"], quotas)
    keep = [r["id"] for t in range(3) for r in donor_cells(payload["donors"], edges)["LoRA", t][:2]]
    part = simulate(payload, fields, donor_ids=keep, **kwargs)
    both = simulate(payload, fields, donor_ids=keep, slot_mask=rotation_mask(ids, set()), **kwargs)
    both.pop("slot_mask")
    assert both == part


def test_slot_mask_gates_follow_masked_reports():
    payload, fields = toy(120)
    quotas = {"LoRA": (40, 40, 40)}
    ids = selected_ids(payload, quotas)
    kwargs = {"replicates": 200, "draws": 200, "quotas": quotas}
    result = simulate(payload, fields, slot_mask=rotation_mask(ids, set(ids[:5])), **kwargs)
    null, power = result["reports"]["null_size"], result["reports"]["power_yield"]
    gates = result["slot_mask"]["gates"]
    assert gates["size"]["threshold"] == 0.07
    assert gates["size"]["ordered_go_rate"] == null["ordered_go_rate"]
    assert gates["size"]["two_sided_rejection_rates"] == null["two_sided_rejection_rates"]
    assert gates["size"]["alternative_rate"] == null["decisions"]["alternative"] / 200
    assert gates["size"]["passed"] is null["accepted"]
    assert gates["power"]["ordered_go_rate"] == power["ordered_go_rate"] > 0.9
    assert gates["power"]["passed"] is True
    assert gates["passed"] is (null["accepted"] and True)
    # NULL-SIZE only (no fields): the power gate is not evaluated and the F10.6 gates do not pass.
    null_only = simulate(payload, slot_mask=rotation_mask(ids, set(ids[:5])), **kwargs)
    assert null_only["slot_mask"]["gates"]["power"]["status"] == "not run (no fields)"
    assert null_only["slot_mask"]["gates"]["passed"] is False


def test_slot_mask_validation():
    from scripts.a5_simulation import SLOT_MASK_SCHEMA

    payload, fields = toy(40, seed=13)
    quotas = {"LoRA": (13, 13, 14)}
    kwargs = {"replicates": 2, "draws": 20, "quotas": quotas}
    ids = selected_ids(payload, quotas)
    good = rotation_mask(ids, {ids[0]})

    def bad(mutate, match):
        mask = {"schema": good["schema"], "available": dict(good["available"])}
        mutate(mask)
        with pytest.raises(ValueError, match=match):
            simulate(payload, fields, slot_mask=mask, **kwargs)

    bad(lambda m: m.update(schema="v0"), "schema")
    bad(lambda m: m["available"].pop(ids[3]), r"missing selected edit id\(s\)")
    bad(lambda m: m["available"].update(zzz=["rotation"]), r"not selected edit id\(s\)")
    bad(lambda m: m["available"].update({ids[2]: ["rotation", "ladder_1"]}), "unknown contrast")
    bad(lambda m: m["available"].update({ids[2]: ["ladder_0.25"] * 2}), "duplicate")
    bad(lambda m: m["available"].update({ids[2]: "rotation"}), "list")
    bad(lambda m: m["available"].update({ids[2]: ["rotation", "ladder_0.5"]}), "ladder")
    bad(lambda m: m.update(extra=1), "unknown key")
    bad(lambda m: m.pop("available"), "available")
    assert SLOT_MASK_SCHEMA == "a5-slot-mask-v1"


def test_slot_mask_cli_records_hashes_and_matches_unmasked(tmp_path, monkeypatch):
    import functools
    import hashlib
    import json
    import sys

    from scripts import a5_simulation

    payload, fields = two_family_toy()
    ids = selected_ids(payload)
    paths = {}
    masks = {"none": rotation_mask(ids, set()), "some": rotation_mask(ids, set(ids[:3]))}
    for name, obj in (("inputs", payload), ("fields", fields), *masks.items()):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(obj))
    small = functools.partial(a5_simulation.simulate, replicates=4, draws=20)
    monkeypatch.setattr(a5_simulation, "simulate", small)

    def run(name, *extra):
        out = tmp_path / f"{name}.out.json"
        argv = ["a5_simulation.py", str(paths["inputs"]), "--fields", str(paths["fields"])]
        monkeypatch.setattr(sys, "argv", [*argv, "--output", str(out), *extra])
        assert a5_simulation.main() == 0
        return json.loads(out.read_text())

    plain = run("plain")
    none = run("none", "--slot-mask", str(paths["none"]))
    some = run("some", "--slot-mask", str(paths["some"]))
    assert "slot_mask" not in plain
    digest = hashlib.sha256(paths["some"].read_bytes()).hexdigest()
    assert some["slot_mask"]["path"] == str(paths["some"])
    assert some["slot_mask"]["sha256"] == digest
    assert some["input_provenance"][str(paths["some"])] == digest
    assert some["slot_mask"]["available_slots"]["rotation"] == 37
    # An all-available mask leaves every non-mask field of the output unchanged.
    none.pop("slot_mask")
    none["input_provenance"].pop(str(paths["none"]))
    assert none == plain
    # An invalid mask is refused before the simulation starts, and no output is written.
    paths["bad"] = tmp_path / "bad.json"
    paths["bad"].write_text(json.dumps(rotation_mask(ids[1:], set())))
    out = tmp_path / "bad.out.json"
    argv = [str(paths["inputs"]), "--output", str(out), "--slot-mask", str(paths["bad"])]
    monkeypatch.setattr(sys, "argv", ["a5_simulation.py", *argv])
    with pytest.raises(ValueError, match="missing selected edit id"):
        a5_simulation.main()
    assert not out.exists()


# --- F11.4 supplement (52-slot calibration) ----------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
F11_CELLS = (("lora_sft", "LoRA", 2), ("chain", "chain", 0), ("chain", "chain", 2))


def supplement_toy():
    """Registered quotas with spare representatives: LoRA 14/14/14, chain 8/8/8 (ids sort by norm)."""
    return two_family_toy(n_lora=42, n_chain=24)


def make_supplement(payload):
    """Synthetic ``a5-f11-supplement-v1`` object: the next four representatives in each F11 cell."""
    import hashlib
    import json

    selected, edges = selection_cells(payload["inventory"])
    cells = []
    for fam, family, t in F11_CELLS:
        picks = [r["id"] for r in selected[family, t]]
        group = sorted(
            r["id"]
            for r in payload["inventory"]
            if r["family"] == family and sum((r["norm"], r["id"]) >= e for e in edges[family]) == t
        )
        cells.append(
            {
                "cell": f"{fam}:{t}",
                "family": fam,
                "tertile": t,
                "quota": len(picks),
                "registered_picks": picks,
                "selected": [i for i in group if i not in picks][:4],
            }
        )
    ids = [i for c in cells for i in c["selected"]]
    return {
        "schema": "a5-f11-supplement-v1",
        "k": 12,
        "per_cell": 4,
        "frozen_cells": [c["cell"] for c in cells],
        "cells": cells,
        "replaces_registered": False,
        "no_top_up": True,
        "registered_edit_ids_sorted": sorted(r["id"] for v in selected.values() for r in v),
        "supplement_ids": ids,
        "supplement_ids_sorted": sorted(ids),
        "supplement_ids_sorted_sha256": hashlib.sha256(
            json.dumps(sorted(ids)).encode()
        ).hexdigest(),
    }


SUPPLEMENT_TOY_EXTRA = [
    *(f"e{i:03}" for i in range(38, 42)),
    *(f"k{i:03}" for i in range(4, 8)),
    *(f"k{i:03}" for i in range(19, 23)),
]


def test_frozen_f11_selection_loads_and_validates():
    import json

    from scripts.a5_simulation import load_supplement

    frozen = json.loads((REPO / "docs/prereg/a5_manifests/selection_f11.json").read_text())
    loaded = load_supplement(frozen)
    assert loaded["ids_sorted_sha256"] == (
        "1848fe4cacbaacc9f643a179835770fefd3027a48cef7cf9d83358f15b85fdd9"
    )
    assert {k: len(v) for k, v in loaded["cells"].items()} == {
        ("LoRA", 2): 4,
        ("chain", 0): 4,
        ("chain", 2): 4,
    }
    ids = [i for v in loaded["cells"].values() for i in v]
    assert len(ids) == len(set(ids)) == 12
    forty = json.loads((REPO / "docs/prereg/a5_manifests/part_a/manifest.json").read_text())
    assert len(forty["edit_ids"]) == 40
    assert not set(ids) & set(forty["edit_ids"])
    assert loaded["registered_ids"] == sorted(forty["edit_ids"])
    assert loaded["cells"]["chain", 0] == [
        "342a1be080049e8f",
        "388347cfb0b7fafa",
        "45cd0be9fea1c139",
        "aaf7a0660c6d5a73",
    ]


def test_load_supplement_rejects_malformed_files():
    import copy

    from scripts.a5_simulation import load_supplement

    payload, _ = supplement_toy()
    good = make_supplement(payload)
    load_supplement(good)

    def bad(mutate, match):
        obj = copy.deepcopy(good)
        mutate(obj)
        with pytest.raises(ValueError, match=match):
            load_supplement(obj)

    bad(lambda o: o.update(schema="v0"), "schema")
    bad(lambda o: o.update(k=9), "k")
    bad(lambda o: o["cells"][0]["selected"].pop(), "four")
    bad(lambda o: o.update(frozen_cells=["lora_sft:2", "chain:0", "chain:1"]), "cells")
    bad(lambda o: o["cells"][1].update(cell="chain:1"), "cell")
    bad(lambda o: o["cells"][2].update(family="LoRA"), "family")
    bad(lambda o: o["cells"].pop(), "cells")
    bad(lambda o: o.update(supplement_ids_sorted_sha256="0" * 64), "sha256")
    bad(lambda o: o["supplement_ids"].reverse(), "supplement_ids")
    bad(
        lambda o: o.update(
            registered_edit_ids_sorted=sorted([*o["registered_edit_ids_sorted"][1:], "e038"])
        ),
        "overlap",
    )
    bad(lambda o: o["registered_edit_ids_sorted"].append("e099"), "40 registered")
    bad(lambda o: o.update(replaces_registered=True), "replace")

    def duplicate(o):
        o["cells"][1]["selected"][0] = o["cells"][0]["selected"][0]
        o["supplement_ids"] = [i for c in o["cells"] for i in c["selected"]]

    bad(duplicate, "duplicate")


def test_supplement_appends_twelve_slots_per_cell():
    payload, fields = supplement_toy()
    kwargs = {"replicates": 6, "draws": 50}
    result = simulate(payload, fields, supplement=make_supplement(payload), **kwargs)
    report = result["supplement"]
    assert report["label"] == "supplemented (F11.4)"
    assert report["n_slots"] == 52
    assert report["cells"] == {
        "LoRA:0": {"registered": 10, "supplement": 0, "slots": 10},
        "LoRA:1": {"registered": 10, "supplement": 0, "slots": 10},
        "LoRA:2": {"registered": 10, "supplement": 4, "slots": 14},
        "chain:0": {"registered": 4, "supplement": 4, "slots": 8},
        "chain:1": {"registered": 3, "supplement": 0, "slots": 3},
        "chain:2": {"registered": 3, "supplement": 4, "slots": 7},
    }
    # Registered slots keep slots 0-39 in the frozen order; supplements are appended in cell order.
    slots = report["slots"]
    assert [s["edit_id"] for s in slots[:40]] == selected_ids(payload)
    assert [s["edit_id"] for s in slots[40:]] == SUPPLEMENT_TOY_EXTRA
    assert {s["origin"] for s in slots[:40]} == {"registered"}
    assert {s["origin"] for s in slots[40:]} == {"supplement"}
    assert [s["cell"] for s in slots[40:]] == ["LoRA:2"] * 4 + ["chain:0"] * 4 + ["chain:2"] * 4
    _, edges = selection_cells(payload["inventory"])
    pools = donor_cells(payload["donors"], edges)
    for mode in ("null_size", "power_yield"):
        for trace in result["reports"][mode]["traces"]:
            assert np.asarray(trace["resolved"]).shape == (52, 3)
            assert len(trace["component_signs"]) == 52
            # Each slot draws from its own cell's (unchanged) donor pool.
            for s, donor in zip(slots, trace["donor_representatives"]):
                family, t = s["cell"].split(":")
                assert donor in {r["id"] for r in pools[family, int(t)]}
    # The quotas recorded stay the registered ones; the design table carries the supplement.
    assert result["quotas"] == {"LoRA": (10, 10, 10), "chain": (4, 3, 3)}


def test_supplement_keeps_donor_pool_and_its_own_donor_rows():
    payload, fields = supplement_toy()
    supplement = make_supplement(payload)
    _, edges = selection_cells(payload["inventory"])
    cells = donor_cells(payload["donors"], edges)
    keep = sorted({r["id"] for key in cells for r in cells[key][:3]} | set(SUPPLEMENT_TOY_EXTRA))
    kwargs = {"replicates": 40, "draws": 50, "donor_ids": keep}
    plain = simulate(payload, fields, **kwargs)
    supp = simulate(payload, fields, supplement=supplement, **kwargs)
    assert supp["donor_pool"] == plain["donor_pool"]  # F11.5: pool unchanged
    overlap = supp["supplement"]["donor_overlap"]
    assert overlap["policy"].startswith("unchanged")
    assert overlap["supplement_ids_in_null_size_pool"] == sorted(SUPPLEMENT_TOY_EXTRA)
    assert overlap["supplement_ids_in_power_pool"] == sorted(SUPPLEMENT_TOY_EXTRA)
    assert overlap["supplement_ids_with_fields"] == sorted(SUPPLEMENT_TOY_EXTRA)
    drawn = {
        d for t in supp["reports"]["power_yield"]["traces"] for d in t["donor_representatives"]
    }
    assert drawn & set(SUPPLEMENT_TOY_EXTRA)  # supplementary donor rows stay drawable


def test_supplement_binds_to_the_payload_selection():
    import copy
    import hashlib
    import json

    payload, fields = supplement_toy()
    good = make_supplement(payload)
    kwargs = {"replicates": 2, "draws": 20}

    def bad(mutate, match):
        obj = copy.deepcopy(good)
        mutate(obj)
        with pytest.raises(ValueError, match=match):
            simulate(payload, fields, supplement=obj, **kwargs)

    def swap(o, cell, old, new):
        o["cells"][cell]["selected"] = [
            new if i == old else i for i in o["cells"][cell]["selected"]
        ]
        o["supplement_ids"] = [i for c in o["cells"] for i in c["selected"]]
        o["supplement_ids_sorted"] = sorted(o["supplement_ids"])
        digest = hashlib.sha256(json.dumps(o["supplement_ids_sorted"]).encode()).hexdigest()
        o["supplement_ids_sorted_sha256"] = digest

    bad(lambda o: swap(o, 0, "e038", "e010"), "not in its cell")  # LoRA:0 row in LoRA:2
    bad(lambda o: swap(o, 0, "e038", "zzz"), "not an eligible inventory")
    bad(lambda o: o["cells"][0]["registered_picks"].reverse(), "registered picks")
    bad(lambda o: o["registered_edit_ids_sorted"].__setitem__(0, "e099"), "registered")
    # A non-representative row (its component is already represented) is refused.
    dup = copy.deepcopy(payload)
    dup["inventory"].append({"id": "e999", "component": "c039", "family": "LoRA", "norm": 39.5})
    obj = copy.deepcopy(good)
    swap(obj, 0, "e039", "e999")
    with pytest.raises(ValueError, match="representative"):
        simulate(dup, fields, supplement=obj, **kwargs)


def test_supplement_slot_mask_requires_the_52_ids():
    from scripts.a5_simulation import SLOT_MASK_SCHEMA

    payload, fields = supplement_toy()
    supplement = make_supplement(payload)
    kwargs = {"replicates": 4, "draws": 20, "supplement": supplement}
    forty = selected_ids(payload)
    fiftytwo = [*forty, *SUPPLEMENT_TOY_EXTRA]
    masked = {forty[0], forty[35], SUPPLEMENT_TOY_EXTRA[5]}
    result = simulate(payload, fields, slot_mask=rotation_mask(fiftytwo, masked), **kwargs)
    report = result["slot_mask"]
    assert report["label"] == "slot-masked (F10.6); supplemented (F11.4)"
    assert report["gates"]["label"] == report["label"]
    assert report["available_slots"] == {"rotation": 49, "ladder_0.25": 52, "ladder_0.5": 52}
    assert [s["edit_id"] for s in report["slots"]] == fiftytwo
    assert report["cells"]["chain:0"] == {
        "slots": 8,
        "available": {"rotation": 7, "ladder_0.25": 8, "ladder_0.5": 8},
    }
    for trace in result["reports"]["power_yield"]["traces"]:
        ok = np.asarray(trace["audit_finite"])
        assert not ok[[0, 35, 45], 0].any()
    # The registered 40-id mask is refused under --supplement (the 12 are missing) ...
    with pytest.raises(ValueError, match=r"missing selected edit id\(s\)"):
        simulate(payload, fields, slot_mask=rotation_mask(forty, set()), **kwargs)
    # ... and the 52-id mask is refused without it.
    with pytest.raises(ValueError, match=r"not selected edit id\(s\)"):
        simulate(payload, fields, slot_mask=rotation_mask(fiftytwo, set()), replicates=2, draws=20)
    # F11.4: a supplementary ladder exclusion may be masked; a registered ladder still may not.
    full = ["rotation", "ladder_0.25", "ladder_0.5"]
    avail = {i: full for i in fiftytwo}
    avail[SUPPLEMENT_TOY_EXTRA[0]] = ["ladder_0.5"]
    ladder = simulate(
        payload, fields, slot_mask={"schema": SLOT_MASK_SCHEMA, "available": avail}, **kwargs
    )
    assert ladder["slot_mask"]["available_slots"] == {
        "rotation": 51,
        "ladder_0.25": 51,
        "ladder_0.5": 52,
    }
    assert ladder["slot_mask"]["masked_ids"]["ladder_0.25"] == [SUPPLEMENT_TOY_EXTRA[0]]
    avail[forty[0]] = ["rotation", "ladder_0.5"]
    with pytest.raises(ValueError, match="ladder slots must stay available"):
        simulate(
            payload, fields, slot_mask={"schema": SLOT_MASK_SCHEMA, "available": avail}, **kwargs
        )


def test_without_supplement_output_is_byte_identical_to_pre_f11_code():
    """Golden sha256s of json.dumps(simulate(...), indent=2) from the pre-F11.4 code (f19e3dc)."""
    import hashlib
    import json

    payload, fields = two_family_toy()
    ids = selected_ids(payload)
    golden = {
        "null": "569636f8e0208f7f424c83f6f67d345235b44f56c8a479aa6e669650b4adecbc",
        "power": "a84232c9c95b20a33f90b67bb4ec8a0f832fefadc3c157fccc7a6b25287dc4d9",
        "mask": "410d1370b86392303b2535d23989bc0c151ef934fa386bc318b97f32d2f462f9",
    }
    runs = {
        "null": simulate(payload, replicates=20, draws=50),
        "power": simulate(payload, fields, replicates=20, draws=50),
        "mask": simulate(
            payload,
            fields,
            slot_mask=rotation_mask(ids, set(ids[::5])),
            replicates=20,
            draws=50,
        ),
    }
    for name, result in runs.items():
        digest = hashlib.sha256(json.dumps(result, indent=2).encode()).hexdigest()
        assert digest == golden[name], name
        assert "supplement" not in result


def test_supplement_cli_records_hash_and_registered_40_sensitivity(tmp_path, monkeypatch):
    import functools
    import hashlib
    import json
    import subprocess
    import sys

    from scripts import a5_simulation

    payload, fields = supplement_toy()
    forty = selected_ids(payload)
    fiftytwo = [*forty, *SUPPLEMENT_TOY_EXTRA]
    masked = {forty[0], forty[35], SUPPLEMENT_TOY_EXTRA[5]}
    paths = {}
    for name, obj in (
        ("inputs", payload),
        ("fields", fields),
        ("supp", make_supplement(payload)),
        ("mask52", rotation_mask(fiftytwo, masked)),
        ("mask40", rotation_mask(forty, masked & set(forty))),
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(obj))
    script = REPO / "scripts/a5_simulation.py"
    base = [sys.executable, str(script), str(paths["inputs"]), "--fields", str(paths["fields"])]
    base += ["--output", str(tmp_path / "refused.json")]
    for extra, message in (
        (["--registered-40-sensitivity"], b"requires --supplement"),
        (["--supplement", str(paths["supp"]), "--registered-40-sensitivity"], b"--slot-mask"),
        (["--supplement", str(paths["supp"]), "--slot-mask", str(paths["mask40"])], b"missing"),
    ):
        refused = subprocess.run([*base, *extra], capture_output=True, check=False)
        assert refused.returncode != 0 and message in refused.stderr, extra
    assert not (tmp_path / "refused.json").exists()
    small = functools.partial(a5_simulation.simulate, replicates=4, draws=20)
    monkeypatch.setattr(a5_simulation, "simulate", small)

    def run(name, *extra):
        out = tmp_path / f"{name}.out.json"
        argv = ["a5_simulation.py", str(paths["inputs"]), "--fields", str(paths["fields"])]
        monkeypatch.setattr(sys, "argv", [*argv, "--output", str(out), *extra])
        assert a5_simulation.main() == 0
        return json.loads(out.read_text())

    supp = run("supp", "--supplement", str(paths["supp"]), "--slot-mask", str(paths["mask52"]))
    digest = hashlib.sha256(paths["supp"].read_bytes()).hexdigest()
    assert supp["supplement"]["path"] == str(paths["supp"])
    assert supp["supplement"]["sha256"] == digest
    assert supp["input_provenance"][str(paths["supp"])] == digest
    assert supp["supplement"]["label"] == "supplemented (F11.4)"
    assert supp["slot_mask"]["available_slots"]["rotation"] == 49
    # Registered-40 sensitivity: the 40-slot design with the frozen 52-id mask restricted to the 40.
    plain40 = run("plain40", "--slot-mask", str(paths["mask40"]))
    sens = run(
        "sens",
        "--supplement",
        str(paths["supp"]),
        "--slot-mask",
        str(paths["mask52"]),
        "--registered-40-sensitivity",
    )
    info = sens.pop("registered_40_sensitivity")
    assert info["label"] == "registered-40 sensitivity (F11.5); descriptive, no decision authority"
    assert info["supplement_sha256"] == digest
    assert info["mask_restricted_to"] == 40
    assert "supplement" not in sens
    assert sens["slot_mask"]["gates"]["decision_authority"] is False
    assert sens["reports"] == plain40["reports"]
    assert sens["slot_mask"]["mapping_sha256"] == plain40["slot_mask"]["mapping_sha256"]
    assert sens["slot_mask"]["available_slots"] == {
        "rotation": 38,
        "ladder_0.25": 40,
        "ladder_0.5": 40,
    }
