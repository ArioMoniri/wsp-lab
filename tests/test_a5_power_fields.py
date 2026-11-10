"""POWER-AND-YIELD field predictions (scripts/a5_power_fields.py) on a toy zoo and toy fields (CPU, no data/)."""

import hashlib
import json
import math

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts import a5_power_fields as pf
from wsp.eval import response_intervention as ri
from wsp.features import response_field as rf
from wsp.features import rotation

OBJ = pf.OBJECTIVE
SHAPES = {"m.a.weight": (6, 4), "m.b.weight": (5, 6), "m.c.weight": (6, 4), "m.n.weight": (4,)}


def _base():
    g = torch.Generator().manual_seed(0)
    return {k: torch.randn(*s, generator=g).to(torch.bfloat16) for k, s in SHAPES.items()}


def _entry(zoo, eid, delta, parent=None, family="chain"):
    d = zoo / eid
    d.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in delta.items()}, str(d / "delta.safetensors"))
    (d / "edit.json").write_text(
        json.dumps({"id": eid, "parent_id": parent, "family": family, "base_revision": "r" * 40})
    )
    return d


def _toy_zoo(tmp_path, base):
    """Root LoRA-like edit r1 (dense), chain root p1, chain step s1 (cumulative from base) and pick k1."""
    g = torch.Generator().manual_seed(1)
    zoo = tmp_path / "zoo"

    def rnd(k, scale=0.5):
        return scale * torch.randn(*SHAPES[k], generator=g)

    r1 = {"m.a.weight": rnd("m.a.weight"), "m.b.weight": rnd("m.b.weight")}
    p1 = {"m.a.weight": rnd("m.a.weight"), "m.c.weight": rnd("m.c.weight")}
    p1_state = {k: pf.stored_state(base[k], v) for k, v in p1.items()}
    s1 = {
        "m.a.weight": p1_state["m.a.weight"].float()
        - base["m.a.weight"].float(),  # unchanged by s1
        "m.c.weight": p1_state["m.c.weight"].float()
        - base["m.c.weight"].float()
        + rnd("m.c.weight"),
        "m.b.weight": rnd("m.b.weight"),
    }
    k1 = {k: v + (0.3 * torch.randn(*v.shape, generator=g)) for k, v in s1.items()}
    _entry(zoo, "r1", r1, family="lora_sft")
    _entry(zoo, "p1", p1)
    _entry(zoo, "s1", s1, parent="p1")
    _entry(zoo, "k1", k1, parent="p1")
    return zoo


def _grads(names, seed=2):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(*SHAPES[n], generator=g) for n in names}


def _field(grads, identity=None):
    cov = {
        n: {"gradient": True, "factors": False, "module": None, "shape": list(t.shape)}
        for n, t in grads.items()
    }
    ident = {"schema_version": rf.SCHEMA_VERSION, "coverage": {OBJ: cov}} | (identity or {})
    return rf.FieldArtifact(ident, {OBJ: grads}, {})


def _entry_plan(zoo, eid, pred, *, pick=False, family="chain"):
    return {
        "id": eid,
        "family": family,
        "predecessor_id": pred,
        "predecessor_is_base": pred == "base",
        "donor_bytes": eid,
        "donor_via_alias": False,
        "predecessor_bytes": None if pred == "base" else pred,
        "predecessor_via_alias": False,
        "registered_chain_pick": pick,
        "gradient_source": pf.SOURCE_EXACT if pick else pf.SOURCE_BASE,
        "canonical_id": eid,
        "rotation_seed": ri.rotation_seed(eid),
        "flags": [],
        "donor_dir": zoo / eid,
        "predecessor_dir": None if pred == "base" else zoo / pred,
    }


def _loader(base):
    from wsp.zoo.apply import load_entry_delta

    return lambda d: load_entry_delta(d, base)


# ------------------------------------------------------------------------------------------ seeds


def test_rotation_seed_is_registered_derivation():
    seeds = json.loads((pf.ROOT / "docs/prereg/a5_manifests/control_seeds.json").read_text())
    reg = seeds["rotation"]
    assert reg["domain"] == pf.ROTATION_DOMAIN == "20261107:rotation:"
    assert (
        ri.rotation_seed("synthetic-test-vector")
        == reg["test_vectors"]["seed('synthetic-test-vector')"]
    )
    digest = hashlib.sha256(b"20261107:rotation:abc").digest()
    assert ri.rotation_seed("abc") == int.from_bytes(digest[:8], "big") % (1 << 63)


# -------------------------------------------------------------------------- exact shortcuts (bitwise)


def test_rotate_block_cached_is_bitwise_rotate_block():
    g = torch.Generator().manual_seed(3)
    seed = ri.rotation_seed("toy")
    shared = torch.Generator()
    shared.manual_seed(seed)
    cache = {}
    for shape, (r0, c0) in [((6, 4), (0, 0)), ((6, 4), (1, 0)), ((5, 6), (0, 1)), ((6, 4), (0, 0))]:
        D = torch.zeros(shape)
        D[r0:, c0:] = torch.randn(shape[0] - r0, shape[1] - c0, generator=g)
        rows, cols = rotation.changed_block(D)
        want = rotation.rotate_block(D, rows, cols, seed, shared_generator=shared)
        got = pf.rotate_block_cached(D, rows, cols, seed, cache)
        assert torch.equal(want, got)
    assert sorted(cache) == [(5, 4), (5, 5), (6, 4)]  # (6, 4) drawn once, reused


def test_exclusion_reason_matches_registered():
    cases = [
        torch.randn(4, 3),
        torch.zeros(4, 3),
        torch.tensor([[float("nan"), 1.0], [0.0, 1.0]]),
        torch.randn(3),
        torch.diag(torch.tensor([1e-30, 0.0, 0.0])),
    ]
    for D in cases:
        if D.ndim == 2:
            rows, cols = rotation.changed_block(D)
            assert pf.exclusion_reason(D, rows, cols) == rotation.exclusion_reason(D, rows, cols)
        else:
            assert pf.exclusion_reason(D, None, None) == "non_2d"


def test_increments_match_build_variants(tmp_path):
    """D, R_eff and the ladders equal what build_variants stages (root and chain step)."""
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    load = _loader(base)
    p1 = load(zoo / "p1")
    pred_state = dict(base) | {k: pf.stored_state(base[k], v) for k, v in p1.items()}
    for eid, pred in (("r1", None), ("s1", "p1")):
        seed = ri.rotation_seed(eid)
        stage = tmp_path / f"stage-{eid}"
        variants = ri.build_variants(
            zoo / eid,
            base if pred is None else pred_state,
            seed,
            stage_root=stage,
            base_state=None if pred is None else base,
            predecessor_id=pred,
        )
        W, D, raw, edited = pf.edit_increments(base, load(zoo / eid), None if pred is None else p1)
        R, exclusions, ladders = pf.variant_increments(W, D, raw, edited, seed)
        assert exclusions == []
        by_kind = {(v.kind, v.strength): v for v in variants}
        assert not by_kind["rotation", 1.0].excluded
        info = json.loads((by_kind["rotation", 1.0].entry_dir / "variant.json").read_text())
        assert info["edited_tensors"] == edited
        for key, inc in [(("rotation", 1.0), R), *((("scale", s), X) for s, X in ladders.items())]:
            d = by_kind[key].entry_dir
            staged = load_file(
                str(d / ("delta.safetensors" if pred is None else "increment.safetensors"))
            )
            assert sorted(staged) == edited
            for k in edited:
                assert torch.equal(staged[k], inc[k]), (eid, key, k)
        if pred is not None:
            assert "m.a.weight" not in edited  # unchanged by the step: zero realised increment


# ------------------------------------------------------------------------------------ predictions


def test_predictions_equal_hand_contraction(tmp_path):
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    grads = _grads(SHAPES)
    view = pf.gradient_view(_field(grads))
    entry = _entry_plan(zoo, "r1", "base", family="LoRA")
    row = pf.donor_record(entry, base, _loader(base), view)
    # hand computation from first principles (registered rotate_block, plain fp64 sums)
    stored = _loader(base)(zoo / "r1")
    keys = sorted(stored)
    D = {k: pf.stored_state(base[k], stored[k]).float() - base[k].float() for k in keys}
    gen = torch.Generator()
    gen.manual_seed(ri.rotation_seed("r1"))
    hand = [0.0, 0.0, 0.0]
    for k in keys:
        rows, cols = rotation.changed_block(D[k])
        R_raw = rotation.rotate_block(D[k], rows, cols, 0, shared_generator=gen)
        R = (base[k].float() + R_raw).to(torch.bfloat16).float() - base[k].float()
        hand[0] += float((grads[k].double() * (D[k] - R).double()).sum())
        for j, s in ((1, 0.25), (2, 0.5)):
            X = (base[k].float() + s * stored[k].float()).to(torch.bfloat16).float() - base[
                k
            ].float()
            hand[j] += float((grads[k].double() * X.double()).sum())
    assert len(row["predictions"]) == 3
    for got, want in zip(row["predictions"], hand):
        assert math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-6)
        assert (got > 0) == (want > 0)
    assert row["flags"] == []
    assert row["rotation_seed"] == ri.rotation_seed("r1")


def test_gradient_view_signed_equals_full_contract():
    grads = _grads(SHAPES)
    full = _field(grads)
    full.identity["coverage"][OBJ]["m.a.weight"] |= {"factors": True, "module": "m.a"}
    full.factors[OBJ] = {
        "m.a": {"A": torch.eye(4), "G": torch.eye(6), "weight": "m.a.weight", "bias": None}
    }
    delta = {k: torch.randn(*SHAPES[k]) for k in ("m.a.weight", "m.b.weight")}
    want = rf.contract(full, delta)[OBJ]
    assert want["m.a.weight"]["q"] is not None
    got = rf.contract(pf.gradient_view(full), delta)[OBJ]
    for k in delta:
        assert got[k]["signed"] == want[k]["signed"] and got[k]["q"] is None
    value, value64 = pf.contraction(pf.gradient_view(full), delta)
    assert math.isclose(value, sum(w["signed"] for w in want.values()), rel_tol=1e-12)
    assert math.isclose(value, value64, rel_tol=1e-5)


def test_exact_identity_checks(tmp_path):
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    load = _loader(base)
    _W, D, _raw, edited = pf.edit_increments(base, load(zoo / "k1"), load(zoo / "p1"))
    nonzero = {k: D[k] for k in edited if torch.count_nonzero(D[k])}
    ident = {
        "pick": {"predecessor_id": "p1"},
        "increment_sha256": pf.tensors_hash(nonzero),
        "support": sorted(nonzero),
    }
    view = pf.gradient_view(_field(_grads(nonzero)))
    entry = _entry_plan(zoo, "k1", "p1", pick=True)
    row = pf.donor_record(entry, base, load, view, exact_identity=ident)
    assert row["hashes"]["D_eff_nonzero"] == ident["increment_sha256"]
    assert row["gradient_source"] == pf.SOURCE_EXACT and row["flags"] == []
    with pytest.raises(ValueError, match="realised increment differs"):
        pf.donor_record(entry, base, load, view, exact_identity=ident | {"increment_sha256": "x"})
    with pytest.raises(ValueError, match="predecessor differs"):
        pf.donor_record(
            entry, base, load, view, exact_identity=ident | {"pick": {"predecessor_id": "q"}}
        )


# ------------------------------------------------------------------------------- plan and flags


def _package(tmp_path):
    comps = [
        {"id": "r1", "parent_id": None, "canonical_id": "r1", "bytes_present": True},
        {"id": "p1", "parent_id": None, "canonical_id": "p1", "bytes_present": True},
        {"id": "s1", "parent_id": "p1", "canonical_id": "s1", "bytes_present": True},
        {"id": "k1", "parent_id": "p1", "canonical_id": "k1", "bytes_present": True},
    ]
    man = tmp_path / "man"
    man.mkdir()
    comp_bytes = "".join(json.dumps(c) + "\n" for c in comps).encode()
    (man / "components.jsonl").write_bytes(comp_bytes)
    selection = {
        "components_jsonl_sha256": hashlib.sha256(comp_bytes).hexdigest(),
        "registered": {
            "selected": [
                {"edit_id": "k1", "family": "chain", "predecessor_id": "p1"},
                {"edit_id": "r1", "family": "lora_sft", "predecessor_id": "base"},
            ]
        },
    }
    (man / "selection.json").write_text(json.dumps(selection))
    inputs = {
        "donors": [
            {"id": "k1", "family": "chain"},
            {"id": "p1", "family": "chain"},
            {"id": "r1", "family": "LoRA"},
            {"id": "s1", "family": "chain"},
        ],
        "provenance": {
            "donors": {
                "k1": {"reference_id": "p1"},
                "p1": {"reference_id": "base"},
                "r1": {"reference_id": "base"},
                "s1": {"reference_id": "p1"},
            }
        },
        "donor_mean": "never read",
    }
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    return inputs, comps, selection, man


def test_plan_sources_and_flags(tmp_path):
    inputs, comps, selection, _ = _package(tmp_path)
    plan = {e["id"]: e for e in pf.plan_donors(inputs, comps, selection)}
    assert [e["id"] for e in pf.plan_donors(inputs, comps, selection)] == ["k1", "p1", "r1", "s1"]
    assert plan["k1"]["gradient_source"] == pf.SOURCE_EXACT and plan["k1"]["flags"] == []
    assert plan["r1"]["gradient_source"] == pf.SOURCE_BASE and plan["r1"]["flags"] == []
    assert plan["p1"]["gradient_source"] == pf.SOURCE_BASE and plan["p1"]["flags"] == []
    assert plan["s1"]["flags"] == [pf.FLAG_UNVALIDATED]
    assert all(e["rotation_seed"] == ri.rotation_seed(i) for i, e in plan.items())
    bad = json.loads(json.dumps(inputs))
    bad["provenance"]["donors"]["s1"]["reference_id"] = "base"
    with pytest.raises(ValueError, match="lineage parent"):
        pf.plan_donors(bad, comps, selection)
    bad = json.loads(json.dumps(inputs))
    bad["donors"] = [d for d in bad["donors"] if d["id"] != "k1"]
    with pytest.raises(ValueError, match="picks without a donor"):
        pf.plan_donors(bad, comps, selection)


def test_run_end_to_end_shape_order_and_manifest(tmp_path, monkeypatch):
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    _inputs, _comps, _selection, man = _package(tmp_path)
    state_hash = rf._hash({n: rf._tensor_hash(t) for n, t in base.items()})
    g2 = tmp_path / "g2"
    grads = _grads(SHAPES)
    fdir = g2 / "fields" / OBJ
    _field(grads, {"state_hash": state_hash}).save(fdir)
    load = _loader(base)
    _W, D, _raw, edited = pf.edit_increments(base, load(zoo / "k1"), load(zoo / "p1"))
    nonzero = {k: D[k] for k in edited if torch.count_nonzero(D[k])}
    exact_grads = _grads(nonzero, seed=5)
    ident = {
        "schema_version": rf.SCHEMA_VERSION,
        "pick": {"predecessor_id": "p1"},
        "increment_sha256": pf.tensors_hash(nonzero),
        "support": sorted(nonzero),
        "base_state_hash": state_hash,
        "predecessor_state_sha256": "toy",
    }
    edir = g2 / "chain_exact" / "k1" / OBJ
    rf.FieldArtifact(ident, {OBJ: exact_grads}, {}).save(edir)

    def files(d):
        return {f: pf.sha256_file(d / f) for f in ("manifest.json", "tensors.pt")}

    exact_c = sum(float((exact_grads[k] * d).sum()) for k, d in nonzero.items())
    (g2 / "g2_manifest.json").write_text(
        json.dumps(
            {
                "fields": {
                    OBJ: {
                        "status": "complete",
                        "files": files(fdir),
                        "state_hash": state_hash,
                        "content_hashes_sha256": "c",
                        "identity_sha256": "i",
                    }
                },
                "chain_exact": {
                    "sha256": "s",
                    "rows": [
                        {
                            "pick_id": "k1",
                            "objective": OBJ,
                            "status": "complete",
                            "predecessor_id": "p1",
                            "dir": str(edir),
                            "files": files(edir),
                            "exact_contraction": exact_c,
                            "identity_sha256": "e",
                        }
                    ],
                },
            }
        )
    )
    monkeypatch.setattr(pf, "load_base_state", lambda h: (base, h) if h == state_hash else None)
    out, mout = tmp_path / "out" / "fields.json", tmp_path / "out" / "manifest.json"
    pf.main(
        [
            "--inputs",
            str(tmp_path / "inputs.json"),
            "--g2",
            str(g2),
            "--selection",
            str(man / "selection.json"),
            "--out",
            str(out),
            "--manifest-out",
            str(mout),
            "--zoo",
            str(zoo),
        ]
    )
    fields = json.loads(out.read_text())
    assert list(fields) == ["k1", "p1", "r1", "s1"]
    assert all(len(v) == 3 and all(math.isfinite(x) for x in v) for v in fields.values())
    manifest = json.loads(mout.read_text())
    assert manifest["fields_json"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["contrast_order"] == ["rotation", "ladder_0.25", "ladder_0.5"]
    donors = manifest["donors"]
    assert donors["k1"]["gradient"]["source"] == pf.SOURCE_EXACT
    assert donors["s1"]["flags"] == [pf.FLAG_UNVALIDATED]
    assert manifest["summary"]["gradient_source_counts"] == {
        pf.SOURCE_BASE: 3,
        pf.SOURCE_EXACT: 1,
    }
    assert manifest["summary"]["flag_counts"] == {pf.FLAG_UNVALIDATED: 1}
    for i, v in fields.items():
        assert donors[i]["predictions"] == v
    # the pick's prediction uses the exact gradient, not the base field
    view = pf.gradient_view(_field(exact_grads))
    entry = _entry_plan(zoo, "k1", "p1", pick=True)
    assert (
        pf.donor_record(entry, base, load, view, exact_identity=ident)["predictions"]
        == fields["k1"]
    )
    # simulation accepts the file shape
    import numpy as np

    arr = np.asarray([fields[i] for i in ["k1", "p1", "r1", "s1"]], float)
    assert arr.shape == (4, 3)
