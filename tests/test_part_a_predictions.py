"""Part A frozen prediction rows (scripts/part_a_predictions.py) on a toy zoo, toy staging and toy fields.

CPU only; nothing under data/ is opened. The toy staging is the real ``build_variants`` (F10 rule), so the staged
files, their layout (root ``delta.safetensors``, chain ``increment.safetensors``) and the manifest hash binding are
those of the Part A staging.
"""

import hashlib
import json
import math

import pytest
import torch

from scripts import a5_power_fields as pf
from scripts import part_a_predictions as pp
from wsp.eval import response_intervention as ri
from wsp.features import response_field as rf

OBJ = pp.OBJECTIVE
SHAPES = {"m.a.weight": (6, 4), "m.b.weight": (5, 6), "m.c.weight": (6, 4), "m.n.weight": (4,)}
MODULES = {"m.a.weight": "m.a", "m.b.weight": "m.b"}  # m.c and m.n have no K-FAC factors


def _base():
    g = torch.Generator().manual_seed(0)
    return {k: torch.randn(*s, generator=g).to(torch.bfloat16) for k, s in SHAPES.items()}


def _entry(zoo, eid, delta, parent=None, family="chain"):
    from safetensors.torch import save_file

    d = zoo / eid
    d.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in delta.items()}, str(d / "delta.safetensors"))
    (d / "edit.json").write_text(
        json.dumps({"id": eid, "parent_id": parent, "family": family, "base_revision": "r" * 40})
    )
    return d


def _toy_zoo(tmp_path, base):
    """Root LoRA-like r1, chain root p1 (predecessor of the pick), chain pick k1 at p1, supplementary root x1."""
    g = torch.Generator().manual_seed(1)
    zoo = tmp_path / "zoo"

    def rnd(k, scale=0.5):
        return scale * torch.randn(*SHAPES[k], generator=g)

    r1 = {"m.a.weight": rnd("m.a.weight"), "m.b.weight": rnd("m.b.weight")}
    p1 = {"m.a.weight": rnd("m.a.weight"), "m.c.weight": rnd("m.c.weight")}
    p1_state = {k: pf.stored_state(base[k], v) for k, v in p1.items()}
    k1 = {
        "m.a.weight": p1_state["m.a.weight"].float() - base["m.a.weight"].float(),  # unchanged
        "m.c.weight": p1_state["m.c.weight"].float()
        - base["m.c.weight"].float()
        + rnd("m.c.weight"),
        "m.b.weight": rnd("m.b.weight"),
    }
    x1 = {"m.a.weight": rnd("m.a.weight"), "m.b.weight": rnd("m.b.weight")}
    _entry(zoo, "r1", r1, family="lora_sft")
    _entry(zoo, "p1", p1)
    _entry(zoo, "k1", k1, parent="p1")
    _entry(zoo, "x1", x1)
    return zoo


def _loader(base):
    from wsp.zoo.apply import load_entry_delta

    return lambda d: load_entry_delta(d, base)


def _spd(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n + 3, generator=g)
    return x @ x.T / (n + 3)


def _field(grads, identity=None, factors=True):
    cov = {}
    for n, t in grads.items():
        has = factors and n in MODULES
        cov[n] = {
            "gradient": True,
            "factors": has,
            "module": MODULES.get(n) if has else None,
            "shape": list(t.shape),
        }
    fac = {}
    if factors:
        fac = {
            m: {
                "A": _spd(SHAPES[n][1], 10 + j),
                "G": _spd(SHAPES[n][0], 20 + j),
                "weight": n,
                "bias": None,
            }
            for j, (n, m) in enumerate(sorted(MODULES.items()))
            if n in grads
        }
    ident = {"schema_version": rf.SCHEMA_VERSION, "coverage": {OBJ: cov}} | (identity or {})
    return rf.FieldArtifact(ident, {OBJ: grads}, {OBJ: fac})


def _grads(names, seed=2):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(*SHAPES[n], generator=g) for n in names}


def _stage(tmp_path, base, zoo, eid, pred):
    load = _loader(base)
    stage = tmp_path / "stage" / "toy"
    pred_state = base
    if pred != "base":
        pred_state = dict(base) | {
            k: pf.stored_state(base[k], v) for k, v in load(zoo / pred).items()
        }
    return ri.build_variants(
        zoo / eid,
        pred_state,
        ri.rotation_seed(eid),
        stage_root=stage,
        base_state=None if pred == "base" else base,
        predecessor_id=pred,
    )


def _manifest(tmp_path, base, zoo, edits):
    """A toy execution manifest over really staged variants: selections, variants and file hashes."""
    m = {
        "selections": [],
        "variants": [],
        "delta_sha256": {},
        "variant_json_sha256": {},
        "rotation_seeds": {},
        "supplement": {"ids": [], "source_bytes_holders": {}},
        "predecessor_bytes_holders": {},
    }
    for eid, pred, family, supp in edits:
        variants = _stage(tmp_path, base, zoo, eid, pred)
        m["selections"].append(
            {"edit_id": eid, "family": family, "predecessor_id": pred, "tertile": 0}
            | ({"supplementary": True} if supp else {})
        )
        if supp:
            m["supplement"]["ids"].append(eid)
        m["rotation_seeds"][eid] = ri.rotation_seed(eid)
        for v in variants:
            m["variants"].append(
                {
                    "variant_id": v.variant_id,
                    "source_id": eid,
                    "kind": v.kind,
                    "strength": v.strength,
                    "excluded": v.excluded,
                    "exclusions": v.exclusions,
                    "predecessor_id": pred,
                }
            )
            d = v.entry_dir
            m["variant_json_sha256"][v.variant_id] = pf.sha256_file(d / "variant.json")
            files = {
                f: pf.sha256_file(d / f)
                for f in ("delta.safetensors", "increment.safetensors")
                if (d / f).exists()
            }
            if files:
                m["delta_sha256"][v.variant_id] = files
    return m


EDITS = [
    ("k1", "p1", "chain", False),
    ("r1", "base", "lora_sft", False),
    ("x1", "base", "chain", True),
]


def _selection_and_components(tmp_path):
    comps = [
        {"id": "r1", "parent_id": None, "bytes_present": True, "bytes_via_alias": None},
        {"id": "p1", "parent_id": None, "bytes_present": True, "bytes_via_alias": None},
        {"id": "k1", "parent_id": "p1", "bytes_present": True, "bytes_via_alias": None},
        {"id": "x1", "parent_id": None, "bytes_present": True, "bytes_via_alias": None},
    ]
    sel = {
        "registered": {
            "selected": [
                {"edit_id": "r1", "family": "lora_sft", "predecessor_id": None},
                {"edit_id": "k1", "family": "chain", "predecessor_id": "p1"},
            ]
        }
    }
    return sel, comps


def _staged_increments(m, stage, eid, root):
    out = {}
    for v in m["variants"]:
        if v["source_id"] != eid:
            continue
        c = "rotation" if v["kind"] == "rotation" else f"ladder_{v['strength']}"
        if v["excluded"]:
            out[c] = None
            continue
        inc, _rec, _vj = pp.load_staged(stage, v["variant_id"], root, m)
        out[c] = {k: t.float() for k, t in inc.items()}
    return out


# ---------------------------------------------------------------------------------- contraction


def test_contract_with_cached_factors_is_bitwise_response_field_contract():
    grads = _grads(SHAPES)
    field = _field(grads)
    g = torch.Generator().manual_seed(3)
    delta = {k: torch.randn(*SHAPES[k], generator=g) for k in SHAPES}
    want = rf.contract(field, delta)[OBJ]
    cache = pp.FactorCache(field.identity["coverage"][OBJ], field.factors[OBJ])
    per, uncovered = pp.contract(grads, field.identity["coverage"][OBJ], cache, delta)
    assert uncovered == {}
    for k in delta:
        assert per[k]["signed"] == want[k]["signed"], k
        assert per[k]["q"] == want[k]["q"], k
    assert want["m.a.weight"]["q"] is not None and want["m.c.weight"]["q"] is None
    # the spectral factors were computed once per module and reused
    cache.raw = None
    again, _ = pp.contract(grads, field.identity["coverage"][OBJ], cache, delta)
    assert again["m.b.weight"]["q"] == want["m.b.weight"]["q"]


def test_factor_cache_drop_raw_keeps_results():
    grads = _grads(SHAPES)
    field = _field(grads)
    cov = field.identity["coverage"][OBJ]
    keep = pp.FactorCache(cov, dict(field.factors[OBJ]))
    drop = pp.FactorCache(cov, dict(field.factors[OBJ]), drop_raw=True)
    for n in MODULES:
        a1, g1 = keep(n)
        a2, g2 = drop(n)
        assert torch.equal(a1, a2) and torch.equal(g1, g2)
    assert all("A" not in p for p in drop.raw.values())


@pytest.mark.parametrize(
    "a,q,partial,kappa,kstatus,tstar,fsign",
    [
        (-2.0, 1.0, False, 0.5, "ok", 2.0, -1),
        (2.0, 1.0, False, 0.5, "ok", None, 1),
        (-1.0, 0.0, False, 0.0, "ok", None, -1),
        (0.0, 1.0, False, None, "edge:a_zero_q_positive", None, 1),
        (0.0, 0.0, False, None, "undefined:a_zero_q_zero", None, 0),
        (-1.0, 2.0, True, None, pp.ABSTAIN_Q_PARTIAL, None, None),
        (None, 1.0, False, None, pp.ABSTAIN_NONFINITE, None, None),
    ],
)
def test_curvature_terms_f8a3_edges(a, q, partial, kappa, kstatus, tstar, fsign):
    out = pp.curvature_terms(a, q, partial)
    assert out["kappa"] == kappa and out["kappa_status"] == kstatus
    assert out["t_star"] == tstar
    assert out["forecast_sign_quadratic"] == fsign
    json.dumps(out, allow_nan=False)


# -------------------------------------------------------------------------------------- staging


def test_load_staged_binds_manifest_hash(tmp_path):
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    m = _manifest(tmp_path, base, zoo, [("r1", "base", "lora_sft", False)])
    stage = tmp_path / "stage" / "toy"
    vid = next(v["variant_id"] for v in m["variants"] if v["kind"] == "scale")
    inc, rec, vj = pp.load_staged(stage, vid, True, m)
    assert rec == {
        "file": "delta.safetensors",
        "sha256": m["delta_sha256"][vid]["delta.safetensors"],
    }
    assert sorted(inc) == vj["edited_tensors"]
    path = stage / vid / "delta.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="differs from the manifest"):
        pp.load_staged(stage, vid, True, m)
    (stage / vid / "variant.json").write_text("{}")
    with pytest.raises(ValueError, match="variant.json differs"):
        pp.load_staged(stage, vid, True, m)


def test_chain_step_reads_increment_file(tmp_path):
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    m = _manifest(tmp_path, base, zoo, [("k1", "p1", "chain", False)])
    stage = tmp_path / "stage" / "toy"
    vid = next(v["variant_id"] for v in m["variants"] if v["kind"] == "rotation")
    _inc, rec, _vj = pp.load_staged(stage, vid, False, m)
    assert rec["file"] == "increment.safetensors"


# ---------------------------------------------------------------------------------- predictions


@pytest.mark.parametrize("eid,pred", [("r1", "base"), ("k1", "p1")])
def test_predictions_equal_power_fields_on_the_same_staged_delta(tmp_path, eid, pred):
    """Same delta -> same value: the staged increments equal a5_power_fields' rebuilt ones, so the predictions
    (rotation <g, D - R>, ladders <g, X_s>) are bitwise the power-field values."""
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    m = _manifest(tmp_path, base, zoo, [(eid, pred, "chain", False)])
    stage = tmp_path / "stage" / "toy"
    load = _loader(base)
    W, D, raw, edited = pf.edit_increments(
        base, load(zoo / eid), None if pred == "base" else load(zoo / pred)
    )
    staged = _staged_increments(m, stage, eid, pred == "base")
    R, excl, ladders = pf.variant_increments(W, D, raw, edited, ri.rotation_seed(eid))
    assert excl == [] and staged["rotation"] is not None
    for k in edited:
        assert torch.equal(staged["rotation"][k], R[k])
    grads = _grads(SHAPES)
    field = _field(grads)
    cov = field.identity["coverage"][OBJ]
    sel, comps = _selection_and_components(tmp_path)
    m["selections"][0]["family"] = "chain" if eid == "k1" else "lora_sft"
    entry = pp.plan_edits(m, sel, comps)[0]
    row, detail = pp.edit_prediction(
        entry, D, edited, staged, grads, cov, pp.FactorCache(cov, field.factors[OBJ])
    )
    want = pf.predictions(pf.gradient_view(field), D, R, ladders, edited)
    assert row["predictions"] == [
        want["rotation"][0],
        want["ladder_0.25"][0],
        want["ladder_0.5"][0],
    ]
    assert row["edit"]["a"] == want["g_dot_D"][0]
    assert row["contrasts"]["rotation"]["a_R"] == want["g_dot_R"][0]
    assert row["contrasts"]["rotation"]["increment_sha256"] == pf.tensors_hash(R)
    assert row["contrasts"]["ladder_0.25"]["increment_sha256"] == pf.tensors_hash(ladders[0.25])
    # q: the per-tensor sum of response_field.contract on the same increment
    full = rf.contract(field, staged["ladder_0.5"])[OBJ]
    assert row["contrasts"]["ladder_0.5"]["q"] == math.fsum(
        r["q"] for r in full.values() if r["q"] is not None
    )
    assert row["contrasts"]["ladder_0.5"]["q_partial"] == any(
        r["q"] is None and torch.count_nonzero(staged["ladder_0.5"][k]) for k, r in full.items()
    )
    assert row["abstentions"] == {}
    assert set(detail) == {"edit", "rotation", "D_minus_R", "ladder_0.25", "ladder_0.5"}
    json.dumps(row, allow_nan=False)


def test_q_partial_and_signs(tmp_path):
    """k1 edits m.c (no factors): q is partial, kappa/t*/forecast abstain, the first-order sign stands."""
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    m = _manifest(tmp_path, base, zoo, [("k1", "p1", "chain", False)])
    stage = tmp_path / "stage" / "toy"
    load = _loader(base)
    _W, D, _raw, edited = pf.edit_increments(base, load(zoo / "k1"), load(zoo / "p1"))
    assert "m.c.weight" in edited
    staged = _staged_increments(m, stage, "k1", False)
    grads = _grads(SHAPES)
    field = _field(grads)
    cov = field.identity["coverage"][OBJ]
    sel, comps = _selection_and_components(tmp_path)
    entry = pp.plan_edits(m, sel, comps)[0]
    row, _ = pp.edit_prediction(
        entry, D, edited, staged, grads, cov, pp.FactorCache(cov, field.factors[OBJ])
    )
    lad = row["contrasts"]["ladder_0.25"]
    assert lad["q_partial"] is True and lad["kappa_status"] == pp.ABSTAIN_Q_PARTIAL
    assert lad["sign"] == pp.sign(lad["prediction"]) != 0
    assert row["contrasts"]["rotation"]["forecast_status"] == pp.ABSTAIN_Q_PARTIAL
    assert row["edit"]["s_star_nominal_status"] == pp.ABSTAIN_Q_PARTIAL


def test_excluded_rotation_abstains_and_ladders_stand(tmp_path):
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    m = _manifest(tmp_path, base, zoo, [("r1", "base", "lora_sft", False)])
    stage = tmp_path / "stage" / "toy"
    load = _loader(base)
    _W, D, _raw, edited = pf.edit_increments(base, load(zoo / "r1"), None)
    staged = _staged_increments(m, stage, "r1", True)
    staged["rotation"] = None
    rot = next(v for v in m["variants"] if v["kind"] == "rotation")
    rot |= {"excluded": True, "exclusions": [{"reason": "replay_failed", "tensor": "m.a.weight"}]}
    grads = _grads(SHAPES)
    field = _field(grads)
    cov = field.identity["coverage"][OBJ]
    sel, comps = _selection_and_components(tmp_path)
    entry = pp.plan_edits(m, sel, comps)[0]
    row, _ = pp.edit_prediction(
        entry, D, edited, staged, grads, cov, pp.FactorCache(cov, field.factors[OBJ])
    )
    assert row["predictions"][0] is None and all(math.isfinite(x) for x in row["predictions"][1:])
    assert row["abstentions"] == {"rotation": pp.ABSTAIN_EXCLUDED}
    assert row["contrasts"]["rotation"]["exclusion_reasons"] == ["replay_failed"]


def test_uncovered_tensor_abstains_only_when_nonzero(tmp_path):
    """chain_exact covers the nonzero-D support only: a zero tensor outside it contributes nothing; a nonzero one
    makes the field abstain for that contrast."""
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    grads = _grads(["m.a.weight"])
    field = _field(_grads(SHAPES))
    cov = field.identity["coverage"][OBJ]
    zero = {"m.a.weight": torch.ones(6, 4), "m.b.weight": torch.zeros(5, 6)}
    row, _ = pp.field_row(grads, cov, None, zero)
    assert row["field_status"] == "ok" and row["n_uncovered_zero_tensors"] == 1
    assert row["a"] == float(grads["m.a.weight"].sum())
    bad = dict(zero, **{"m.b.weight": torch.ones(5, 6)})
    row, _ = pp.field_row(grads, cov, None, bad)
    assert row["field_status"] == pp.ABSTAIN_UNCOVERED and row["a"] is None
    assert row["uncovered_nonzero_tensors"] == ["m.b.weight"]
    nan = {"m.a.weight": torch.full((6, 4), float("nan"))}
    row, _ = pp.field_row(grads, cov, None, nan)
    assert row["field_status"] == pp.ABSTAIN_NONFINITE and row["a"] is None
    del zoo


# ----------------------------------------------------------------------------------------- plan


def test_plan_sources_holders_and_refusals(tmp_path):
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    m = _manifest(tmp_path, base, zoo, EDITS)
    sel, comps = _selection_and_components(tmp_path)
    plan = {e["edit_id"]: e for e in pp.plan_edits(m, sel, comps)}
    assert list(plan) == ["k1", "r1", "x1"]
    assert plan["k1"]["gradient_source"] == pp.SOURCE_EXACT
    assert plan["k1"]["gradient_anchor"] == "predecessor-exact"
    assert plan["r1"]["gradient_source"] == pp.SOURCE_BASE
    assert plan["x1"]["supplementary"] and plan["x1"]["gradient_anchor"] == "base-exact"
    assert plan["k1"]["predecessor_bytes"] == "p1"
    # a nonroot edit that is not a pick has no validated gradient (F6.1)
    sel2 = {
        "registered": {
            "selected": [s for s in sel["registered"]["selected"] if s["edit_id"] != "k1"]
        }
    }
    with pytest.raises(ValueError, match="no validated gradient"):
        pp.plan_edits(m, sel2, comps)
    comps2 = [dict(c, parent_id="r1") if c["id"] == "k1" else c for c in comps]
    with pytest.raises(ValueError, match="lineage parent"):
        pp.plan_edits(m, sel, comps2)
    m2 = dict(m, variants=[v for v in m["variants"] if v["kind"] != "rotation"])
    with pytest.raises(ValueError, match="lacks its rotation"):
        pp.plan_edits(m2, sel, comps)


def test_rows_sha256_and_table():
    rows = [
        {"edit_id": "b", "predictions": [1.0, -2.0, None]},
        {"edit_id": "a", "predictions": [None, 0.5, 0.25]},
    ]
    h = pp.rows_sha256(rows)
    assert h == pp.rows_sha256(list(reversed(rows)))
    assert (
        h
        == hashlib.sha256(
            json.dumps(
                sorted(rows, key=lambda r: r["edit_id"]), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    assert pp.rows_sha256([rows[0], dict(rows[1], predictions=[None, 0.5, 0.26])]) != h
    assert list(pp.predictions_table(rows)) == ["a", "b"]
    assert pp.sign_distribution(rows)["rotation"] == {
        "positive": 1,
        "negative": 0,
        "zero": 0,
        "null": 1,
    }


def test_compare_with_power_fields_causes():
    def row(i, preds, hashes, source=pp.SOURCE_BASE):
        return {
            "edit_id": i,
            "gradient_source": source,
            "predictions": preds,
            "edit": {"increment_sha256": "D"},
            "contrasts": {
                c: {
                    "increment_sha256": h,
                    "field_status": "ok" if p is not None else pp.ABSTAIN_EXCLUDED,
                }
                for c, h, p in zip(pp.CONTRASTS, hashes, preds)
            },
        }

    rows = [
        row("a", [1.0, 2.0, 3.0], ["R", "L1", "L2"]),
        row("b", [-1.0, 2.0, 3.0], ["Rx", "L1", "L2"]),
        row("c", [None, -2.0, 3.0], [None, "L1", "L2"]),
        row("d", [1.0, -2.0, 3.0], ["R", "L1", "L2"]),
    ]
    fields = {
        "a": [1.0, 2.0, 3.0],
        "b": [1.0, 2.0, 3.0],
        "c": [1.0, -2.0, 3.0],
        "d": [1.0, 2.0, 3.0],
        "z": [1, 1, 1],
    }
    prow = {
        i: {
            "gradient_source": pp.SOURCE_BASE,
            "hashes": {"D_eff": "D", "R_eff": "R", "ladder_0.25": "L1", "ladder_0.5": "L2"},
        }
        for i in fields
    }
    out = pp.compare_with_power_fields(rows, fields, prow)
    assert out["n_shared"] == 4
    assert out["by_contrast"]["rotation"] == {
        "compared": 3,
        "sign_agree": 2,
        "bitwise_equal_value": 2,
    }
    assert out["by_contrast"]["ladder_0.25"]["sign_agree"] == 3
    causes = {(d["edit_id"], d["contrast"]): d["cause"] for d in out["disagreements"]}
    assert causes[("b", "rotation")].startswith("different staged R_eff")
    assert causes[("d", "ladder_0.25")].startswith("UNEXPLAINED")
    assert out["unexplained"] == 1
    assert out["not_comparable"][0]["cause"] == pp.ABSTAIN_EXCLUDED


def test_out_path_guard():
    pp.check_out_path(pp.ALLOWED_OUT / "x.json")
    with pytest.raises(ValueError):
        pp.check_out_path(pp.DATA_ROOT / "runs/a5/part_a/x.json")


# ------------------------------------------------------------------------------------ end to end


def test_run_end_to_end(tmp_path, monkeypatch):
    base = _base()
    zoo = _toy_zoo(tmp_path, base)
    m = _manifest(tmp_path, base, zoo, EDITS)
    sel, comps = _selection_and_components(tmp_path)
    man = tmp_path / "man"
    man.mkdir()
    comp_bytes = "".join(json.dumps(c) + "\n" for c in comps).encode()
    (man / "components.jsonl").write_bytes(comp_bytes)
    sel["components_jsonl_sha256"] = hashlib.sha256(comp_bytes).hexdigest()
    (man / "selection.json").write_text(json.dumps(sel))
    (man / "manifest.json").write_text(json.dumps(m))
    state_hash = rf._hash({n: rf._tensor_hash(t) for n, t in base.items()})
    g2 = tmp_path / "g2"
    field = _field(_grads(SHAPES), {"state_hash": state_hash})
    fdir = g2 / "fields" / OBJ
    field.save(fdir)
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

    exact_c = math.fsum(float((exact_grads[k] * d).sum()) for k, d in nonzero.items())
    (g2 / "g2_manifest.json").write_text(
        json.dumps(
            {
                "fields": {
                    OBJ: {
                        "status": "complete",
                        "files": files(fdir),
                        "state_hash": state_hash,
                        "content_hashes_sha256": "c",
                        "identity_sha256": rf._hash(field.identity),
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
    monkeypatch.setattr(pp, "G2_MANIFEST_SHA256", pf.sha256_file(g2 / "g2_manifest.json"))
    monkeypatch.setattr(pp, "WIKITEXT_IDENTITY_SHA256", rf._hash(field.identity))
    monkeypatch.setattr(pf, "load_base_state", lambda h: (base, h) if h == state_hash else None)
    out = tmp_path / "out" / "predictions.json"
    common = [
        "--manifest", str(man / "manifest.json"),
        "--selection", str(man / "selection.json"),
        "--g2", str(g2),
        "--zoo", str(zoo),
        "--stage", str(tmp_path / "stage" / "toy"),
        "--threads", "1",
    ]  # fmt: skip
    pp.main(["--out", str(out), *common])
    doc = json.loads(out.read_text())
    assert doc["status"] == "complete"
    assert list(doc["predictions"]) == ["k1", "r1", "x1"]
    assert all(
        len(v) == 3 and all(math.isfinite(x) for x in v) for v in doc["predictions"].values()
    )
    assert doc["rows_sha256"] == pp.rows_sha256(doc["rows"])
    rows = {r["edit_id"]: r for r in doc["rows"]}
    assert rows["k1"]["gradient"]["source"] == pp.SOURCE_EXACT
    assert rows["k1"]["edit"]["a"] == pytest.approx(exact_c, rel=1e-6)
    assert rows["x1"]["supplementary"] and rows["x1"]["gradient"]["source"] == pp.SOURCE_BASE
    assert all(
        r["ladder_rebuilt_bitwise"] == {"ladder_0.25": True, "ladder_0.5": True}
        for r in rows.values()
    )
    assert doc["inputs"]["manifest"]["sha256"] == pf.sha256_file(man / "manifest.json")
    assert doc["summary"]["pricing"]["first_edit_id"] == "k1"  # exact picks first
    # the --fields mapping is what the analysis consumes (A5.3.2 order, ids sorted)
    assert doc["predictions"] == {r["edit_id"]: r["predictions"] for r in doc["rows"]}
    # the same predictions with the detail and rows files
    assert len(out.with_name("predictions.rows.jsonl").read_text().splitlines()) == 3
    # pricing guard: a cap below the projection stops after the first edit
    with pytest.raises(SystemExit, match="price_exceeds_cap"):
        pp.main(["--out", str(tmp_path / "out2" / "p.json"), *common, "--max-wall-hours", "0"])
    # a tampered staged file refuses the run
    vid = next(
        v["variant_id"] for v in m["variants"] if v["source_id"] == "r1" and v["kind"] == "scale"
    )
    path = tmp_path / "stage" / "toy" / vid / "delta.safetensors"
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="differs from the manifest"):
        pp.main(["--out", str(tmp_path / "out3" / "p.json"), *common])


# ------------------------------------------------------------------------- synthetic binding helpers


def test_binding_tamper_helpers():
    from scripts import part_a_synthetic_binding as sb

    raw = json.dumps({"a": 1, "code_sha": "887a5146", "b": [1, 2]}, indent=1).encode()
    flipped, at = sb.flip_byte_in_value(raw)
    assert len(flipped) == len(raw) and sum(x != y for x, y in zip(raw, flipped)) == 1
    assert json.loads(flipped)["code_sha"] != "887a5146" and flipped[at] != raw[at]
    doc = {"predictions": {"a": [1.0, 2.0, 3.0], "b": [None, 1.0, 2.0]}, "rows": []}
    out = sb.drop_row(doc, "a")
    assert list(out["predictions"]) == ["b"] and list(doc["predictions"]) == ["a", "b"]
    rep = {
        "registered": {
            "pairs": {
                "a": {c: {"prediction": p} for c, p in zip(pp.CONTRASTS, [1.0, 2.0, 3.0])},
                "b": {c: {"prediction": p} for c, p in zip(pp.CONTRASTS, [None, 1.0, 2.0])},
            }
        }
    }
    assert sb._predictions_used(rep, doc["predictions"])["all_equal"]
    rep["registered"]["pairs"]["b"]["ladder_0.5"]["prediction"] = 2.5
    used = sb._predictions_used(rep, doc["predictions"])
    assert not used["all_equal"] and used["mismatches"] == [["b", "ladder_0.5"]]
