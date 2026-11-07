"""Part A intervention runner (PREREG A5.3): selection, rotation/ladder variants, manifests, scoring plan.

Synthetic manifest rows and tiny bf16 state dicts only: no model, no data/, CPU.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from wsp.eval import response_intervention as ri
from wsp.eval.batch import effective_config_hash

REV = "r" * 40
NO_FM = object()  # marks a row without a pre-outcome feature manifest


# --------------------------------------------------------------------------------------------- selection


def _inventory(rows):
    """rows: (id, family, component, norm[, extra sidecar fields]) -> (manifest rows, norms, sidecar)."""
    manifest, norms, sidecar = [{"type": "header", "base_revision": REV}], {}, []
    for row in rows:
        i, fam, comp, norm, *extra = row
        manifest.append({"id": i, "family": fam, "parent_id": None, "eligibility": "production"})
        if norm is not NO_FM:
            norms[i] = norm
        sidecar.append(
            {"id": i, "component_id": comp, "closure_certified": True, **(extra or [{}])[0]}
        )
    return manifest, norms, sidecar


HAND = [
    # LoRA. a2/a3/a4 tie at 2.0 and straddle the low/middle cutpoint: ties go by id.
    ("a1", "lora_sft", "C1", 1.0),
    ("a3", "lora_sft", "C3", 2.0),
    ("a2", "lora_sft", "C2", 2.0),
    ("a4", "lora_sft", "C4", 2.0),
    ("a5", "lora_sft", "C5", 6.0),  # high cell is [a6, a5] by norm, but picks by id -> a5
    ("a6", "lora_sft", "C6", 5.0),
    ("b1", "lora_sft", "C1", 9.0),  # C1 spans tertiles: represented only by a1
    ("d1", "lora_sft", "C7", 0.1),  # C7 spans families: represented only by chain c5 < d1
    ("z9", "lora_sft", "C9", 3.0, {"exclusion_reason": "closure_shares_training_run"}),
    ("x1", "task_vector", "C8", 1.0),  # unsupported family
    ("y1", "lora_sft", "C10", NO_FM),  # ineligible: no pre-outcome feature manifest
    # chain: ordered c4(0.5) c1(1) c3(2) c2(3) c5(4); N=5, b1=1, b2=3
    ("c1", "chain", "K1", 1.0),
    ("c2", "chain", "K2", 3.0),
    ("c3", "chain", "K3", 2.0),
    ("c4", "chain", "K4", 0.5),
    ("c5", "chain", "C7", 4.0),
]


def test_quotas_follow_chair_rule():
    assert ri.tertile_quotas(30) == [10, 10, 10]
    assert ri.tertile_quotas(10) == [4, 3, 3]
    assert ri.tertile_quotas(11) == [4, 4, 3]
    assert ri.tertile_quotas(3) == [1, 1, 1]


def test_selection_reproduces_hand_computed_expectation():
    sel, ledger = ri.select_with_ledger(*_inventory(HAND), n_lora=3, n_chain=3)
    got = [(s.family, s.tertile, s.edit_id) for s in sel]
    assert got == [
        ("lora_sft", 0, "a1"),
        ("lora_sft", 1, "a3"),
        ("lora_sft", 2, "a5"),
        ("chain", 0, "c4"),
        ("chain", 1, "c1"),
        ("chain", 2, "c2"),
    ]
    assert ri.select_edits(*_inventory(HAND), n_lora=3, n_chain=3) == sel
    lora = ledger["families"]["lora_sft"]
    assert lora["ordered"] == ["a1", "a2", "a3", "a4", "a6", "a5"]
    assert (lora["N"], lora["b1"], lora["b2"]) == (6, 2, 4)
    assert lora["cutpoints"] == [[2.0, "a3"], [5.0, "a6"]]
    assert ledger["families"]["chain"]["ordered"] == ["c4", "c1", "c3", "c2", "c5"]
    assert {s.edit_id: s.rank for s in sel}["a5"] == 5
    rejected = {r["id"]: r["reason"] for r in ledger["rejected"]}
    assert rejected == {
        "b1": "component_represented_by:a1",
        "d1": "component_represented_by:c5",
        "z9": "closure_shares_training_run",
        "x1": "unsupported_family:task_vector",
        "y1": "no_feature_manifest",
    }
    assert len({s.component_id for s in sel}) == len(sel)


def test_selection_shortage_stops_without_cross_cell_replacement():
    with pytest.raises(ri.SelectionUnresolved, match="lora_sft tertile 0"):
        ri.select_edits(*_inventory(HAND), n_lora=9, n_chain=3)


@pytest.mark.parametrize(
    "bad", [{"closure_certified": False}, {"component_id": None}], ids=["uncertified", "no_comp"]
)
def test_uncertified_source_blocks_selection(bad):
    manifest, norms, sidecar = _inventory(HAND)
    sidecar[0].update(bad)
    with pytest.raises(ri.SelectionBlocked):
        ri.select_edits(manifest, norms, sidecar, n_lora=3, n_chain=3)


@pytest.mark.parametrize("value", [float("nan"), None, float("inf")])
def test_nonfinite_or_missing_norm_blocks_selection(value):
    manifest, norms, sidecar = _inventory(HAND)
    norms["a4"] = value
    with pytest.raises(ri.SelectionBlocked, match="a4"):
        ri.select_edits(manifest, norms, sidecar, n_lora=3, n_chain=3)


def test_registered_quotas_select_40_components():
    rows = [(f"L{i:03d}", "lora_sft", f"CL{i}", float(i % 17)) for i in range(91)]
    rows += [(f"H{i:03d}", "chain", f"CH{i}", float(i % 7)) for i in range(31)]
    sel = ri.select_edits(*_inventory(rows))
    assert len(sel) == 40 and len({s.component_id for s in sel}) == 40
    count = lambda fam: [sum(s.family == fam and s.tertile == t for s in sel) for t in range(3)]
    assert count("lora_sft") == [10, 10, 10]
    assert count("chain") == [4, 3, 3]


# ---------------------------------------------------------------------------------------------- variants


def _w(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)


def _lowrank(shape, rows, cols, rank, seed, scale=0.1):
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(shape)
    b = torch.randn(len(rows), rank, generator=g)
    a = torch.randn(rank, len(cols), generator=g)
    out[torch.tensor(rows)[:, None], torch.tensor(cols)] = scale * (b @ a) / rank
    return out


def _base():
    return {
        "layers.0.q.weight": _w((8, 6), 1),
        "layers.0.v.weight": _w((6, 6), 2),
        "layers.0.norm.weight": _w((6,), 3),
        "layers.1.o.weight": _w((6, 6), 4),
    }


def _entry(zoo, i, delta, *, family="lora_sft", parent=None, cumulative=False):
    d = zoo / i
    d.mkdir(parents=True)
    edit = {
        "id": i,
        "family": family,
        "parent_id": parent,
        "base_revision": REV,
        "provenance": {"cumulative_snapshot": True} if cumulative else {},
    }
    (d / "edit.json").write_text(json.dumps(edit))
    save_file({k: v.contiguous() for k, v in delta.items()}, str(d / "delta.safetensors"))
    return d


def _root_delta():
    return {
        "layers.0.q.weight": _lowrank((8, 6), [1, 2, 3, 5], [0, 1, 2, 4], 2, 10),
        "layers.0.v.weight": _lowrank((6, 6), [0, 2, 3], [1, 3, 4, 5], 2, 11),
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_variants_obey_realised_convention_and_rotation_audit_passes(tmp_path):
    base, delta = _base(), _root_delta()
    entry = _entry(tmp_path / "zoo", "e1", delta)
    variants = ri.build_variants(entry, base, seed=7, stage_root=tmp_path / "stage")
    assert [(v.kind, v.strength) for v in variants] == [
        ("rotation", 1.0),
        ("scale", 0.25),
        ("scale", 0.5),
    ]
    assert len({v.variant_id for v in variants}) == 3
    for v in variants:
        assert not v.excluded and v.entry_dir.parent == tmp_path / "stage"
        x = load_file(v.entry_dir / "delta.safetensors")
        assert set(x) == set(delta)  # keys = the edited tensors
        for k, d in x.items():
            assert d.dtype == torch.float32
            new = (base[k].float() + d).to(torch.bfloat16)
            assert torch.equal(new.float() - base[k].float(), d)  # realised: bf16(W+X) - W
        meta = json.loads((v.entry_dir / "variant.json").read_text())
        assert meta["kind"] == v.kind and meta["source_id"] == "e1"
        assert meta["predecessor_id"] == "base" and meta["seed"] == 7
        assert json.loads((v.entry_dir / "edit.json").read_text())["id"] == "e1"
    rot = variants[0]
    meta = json.loads((rot.entry_dir / "variant.json").read_text())
    assert meta["edit_audit"]["passes"] is True
    assert all(t["passes"] for t in meta["tensor_audits"].values())
    assert meta["coverage"] == {"n_edited": 2, "n_rotated": 2, "n_excluded": 0, "fraction": 1.0}
    assert meta["exclusions"] == []
    assert meta["tensor_audits"]["layers.0.q.weight"]["orthogonality_error"] < 1e-5
    assert meta["tensor_audits"]["layers.0.q.weight"]["reconstruction_residual"] < 1e-5
    # rotation is confined to the changed block and differs from the edit
    r = load_file(rot.entry_dir / "delta.safetensors")["layers.0.q.weight"]
    d_eff = (base["layers.0.q.weight"].float() + delta["layers.0.q.weight"]).to(torch.bfloat16)
    d_eff = d_eff.float() - base["layers.0.q.weight"].float()
    assert torch.all(r[[0, 4, 6, 7]] == 0) and torch.all(r[:, [3, 5]] == 0)
    assert not torch.equal(r, d_eff)
    # scaled edits apply s * Delta at W and round; they do not scale D
    half = load_file(variants[2].entry_dir / "delta.safetensors")["layers.0.v.weight"]
    w = base["layers.0.v.weight"]
    expected = (w.float() + 0.5 * delta["layers.0.v.weight"]).to(torch.bfloat16).float() - w.float()
    assert torch.equal(half, expected)


def test_rotation_shares_one_generator_per_edit(tmp_path):
    """Equal-shaped tensors with the same realised block get the same frames (A5.3.3)."""
    base = {"a.weight": _w((6, 6), 1), "b.weight": _w((6, 6), 1)}
    blk = _lowrank((6, 6), [0, 1, 2, 3], [0, 1, 2, 3], 2, 3)
    entry = _entry(tmp_path / "zoo", "e2", {"a.weight": blk, "b.weight": blk.clone()})
    rot = ri.build_variants(entry, base, seed=5, stage_root=tmp_path / "s", strengths=())[0]
    x = load_file(rot.entry_dir / "delta.safetensors")
    assert torch.equal(x["a.weight"], x["b.weight"])


def test_exclusion_excludes_whole_rotation_pair_and_is_listed(tmp_path):
    base, delta = _base(), _root_delta()
    delta["layers.0.norm.weight"] = torch.full((6,), 0.01)
    entry = _entry(tmp_path / "zoo", "e3", delta)
    rot, *scaled = ri.build_variants(entry, base, seed=7, stage_root=tmp_path / "stage")
    assert rot.excluded and rot.exclusions == [
        {"tensor": "layers.0.norm.weight", "reason": "non_2d"}
    ]
    assert not (rot.entry_dir / "delta.safetensors").exists()
    meta = json.loads((rot.entry_dir / "variant.json").read_text())
    assert meta["excluded"] is True and meta["exclusions"] == rot.exclusions
    assert meta["coverage"] == {"n_edited": 3, "n_rotated": 0, "n_excluded": 1, "fraction": 0.0}
    for v in scaled:  # the ladder is a separate contrast and is still built
        assert not v.excluded and (v.entry_dir / "delta.safetensors").exists()
    staged = [json.loads(x) for x in (tmp_path / "stage/manifest.jsonl").read_text().splitlines()]
    assert rot.variant_id not in {r.get("id") for r in staged}
    assert {v.variant_id for v in scaled} <= {r.get("id") for r in staged}


def test_never_writes_into_canonical_zoo(tmp_path):
    entry = _entry(tmp_path / "zoo", "e4", _root_delta())
    with pytest.raises(ValueError, match="canonical zoo"):
        ri.build_variants(entry, _base(), seed=1, stage_root=tmp_path / "zoo")
    with pytest.raises(ValueError, match="canonical zoo"):
        ri.build_variants(entry, _base(), seed=1, stage_root=tmp_path / "zoo" / "stage")


def test_chain_variant_is_cumulative_for_the_scorer_and_increment_is_realised(tmp_path):
    base = _base()
    cum_p = {"layers.1.o.weight": _lowrank((6, 6), [0, 1], [2, 3], 1, 20)}
    pred = dict(base)
    pred["layers.1.o.weight"] = (base["layers.1.o.weight"].float() + cum_p["layers.1.o.weight"]).to(
        torch.bfloat16
    )
    step = _lowrank((8, 6), [1, 2, 3], [0, 1, 2], 2, 21)
    succ_q = (pred["layers.0.q.weight"].float() + step).to(torch.bfloat16)
    cum_e = {
        "layers.1.o.weight": pred["layers.1.o.weight"].float() - base["layers.1.o.weight"].float(),
        "layers.0.q.weight": succ_q.float() - base["layers.0.q.weight"].float(),
    }
    zoo = tmp_path / "zoo"
    _entry(zoo, "p1", cum_p, family="chain", cumulative=True)
    entry = _entry(zoo, "c1", cum_e, family="chain", parent="p1", cumulative=True)
    with pytest.raises(ValueError, match="base_state"):
        ri.build_variants(entry, pred, seed=3, stage_root=tmp_path / "stage")
    variants = ri.build_variants(
        entry, pred, seed=3, stage_root=tmp_path / "stage", base_state=base, predecessor_id="p1"
    )
    for v in variants:
        assert not v.excluded and v.predecessor_id == "p1"
        inc = load_file(v.entry_dir / "increment.safetensors")
        assert set(inc) == {"layers.0.q.weight"}
        cum = load_file(v.entry_dir / "delta.safetensors")
        assert set(cum) == {"layers.0.q.weight", "layers.1.o.weight"}
        for k, x in cum.items():  # what wsp.eval.batch scores: bf16(base + delta)
            scored = (base[k].float() + x).to(torch.bfloat16)
            want = (
                pred[k]
                if k != "layers.0.q.weight"
                else (pred[k].float() + inc[k]).to(torch.bfloat16)
            )
            assert torch.equal(scored, want)
            assert torch.equal(want.float() - pred[k].float(), inc.get(k, torch.zeros_like(x)))
        meta = json.loads((v.entry_dir / "variant.json").read_text())
        assert meta["delta_convention"] == "realised_cumulative_from_base"
        assert meta["scale_source"] == "realised_increment"
    assert json.loads((variants[0].entry_dir / "variant.json").read_text())["edit_audit"]["passes"]


# --------------------------------------------------------------------------------------------- manifest


def test_manifest_hashes_are_deterministic(tmp_path):
    base, delta = _base(), _root_delta()
    entry = _entry(tmp_path / "zoo", "a1", delta)
    sel = ri.select_edits(*_inventory(HAND), n_lora=3, n_chain=3)
    m = []
    for name in ("s1", "s2"):
        v = ri.build_variants(entry, base, seed=ri.rotation_seed("a1"), stage_root=tmp_path / name)
        m.append(ri.manifest_for(sel, v, code_sha="abc"))
    assert m[0] == m[1]
    assert json.dumps(m[0], sort_keys=True) == json.dumps(m[1], sort_keys=True)
    man = m[0]
    assert man["code_sha"] == "abc"
    assert man["edit_ids"] == sorted(s.edit_id for s in sel)
    assert man["component_ids"] == sorted(s.component_id for s in sel)
    assert man["predecessor_ids"] == ["base"]
    assert len(man["variant_ids"]) == 3
    for vid, files in man["delta_sha256"].items():
        assert files["delta.safetensors"] == _sha(tmp_path / "s1" / vid / "delta.safetensors")
    assert set(man["reference_ids_needed"]) == {"predecessors", "originals"}
    assert man["reference_ids_needed"]["predecessors"] == ["base"]
    assert man["reference_ids_needed"]["originals"] == man["edit_ids"]  # all originals required
    assert man["selected_without_variants"] == [i for i in man["edit_ids"] if i != "a1"]


def test_rotation_seed_follows_a5_3_3():
    digest = hashlib.sha256(b"20261107:rotation:abc").digest()
    assert ri.rotation_seed("abc") == int.from_bytes(digest[:8], "big") % (1 << 63)


# ----------------------------------------------------------------------------------------- scoring plan

CFG = {
    "generation": {"tier": "tf_v1", "max_new_tokens": 64, "batch_size": 1, "seed": 1},
    "benchmarks": {"wikitext_ppl": {"n_windows": 64}, "mmlu_pro": {"n": 1000}},
}


def _record(evals, i, *, hashes=None, cfg=CFG, status="complete", counts=None):
    d = evals / i
    d.mkdir(parents=True)
    payload = {
        "eval_status": status,
        "config_hash": effective_config_hash(cfg),
        "model_revision": REV,
        "counts": counts or {"wikitext_ppl": 64, "mmlu_pro": 1000},
        "entry_hashes": hashes or {},
        "edit_id": i,
        "run_id": f"run-{i}",
    }
    (d / "eval.json").write_text(json.dumps(payload))


def test_plan_scoring_counts_references(tmp_path):
    base = _base()
    zoo, stage, evals = tmp_path / "zoo", tmp_path / "stage", tmp_path / "evals"
    (zoo).mkdir()
    (zoo / "manifest.jsonl").write_text(json.dumps({"type": "header", "base_revision": REV}) + "\n")
    _entry(zoo, "r1", _root_delta())
    p_ok = _entry(
        zoo,
        "pA",
        {"layers.1.o.weight": _lowrank((6, 6), [0], [1], 1, 1)},
        family="chain",
        cumulative=True,
    )
    _entry(
        zoo,
        "pB",
        {"layers.1.o.weight": _lowrank((6, 6), [1], [2], 1, 2)},
        family="chain",
        cumulative=True,
    )
    built = []
    for i, parent in (("cA", "pA"), ("cB", "pB")):
        pdelta = load_file(zoo / parent / "delta.safetensors")
        pred = dict(base)
        pred["layers.1.o.weight"] = (
            base["layers.1.o.weight"].float() + pdelta["layers.1.o.weight"]
        ).to(torch.bfloat16)
        step = _lowrank((8, 6), [1, 2, 3], [0, 1, 2], 2, 30)
        cum = {
            **pdelta,
            "layers.0.q.weight": (pred["layers.0.q.weight"].float() + step)
            .to(torch.bfloat16)
            .float()
            - base["layers.0.q.weight"].float(),
        }
        e = _entry(zoo, i, cum, family="chain", parent=parent, cumulative=True)
        built += ri.build_variants(
            e, pred, seed=1, stage_root=stage, base_state=base, predecessor_id=parent
        )
    built += ri.build_variants(zoo / "r1", base, seed=1, stage_root=stage)
    # base and pA on file with the same scoring identity; pB missing; original cA on file but stale bytes
    _record(evals, "base")
    _record(evals, "pA", hashes={"delta.safetensors": _sha(p_ok / "delta.safetensors")})
    _record(evals, "cA", hashes={"delta.safetensors": "0" * 64})
    _record(
        evals,
        "r1",
        hashes={"delta.safetensors": _sha(zoo / "r1/delta.safetensors")},
        cfg=CFG | {"x": 1},
    )
    plan = ri.plan_scoring(stage, evals, CFG)
    by_source = {v.variant_id: v.predecessor_id for v in built}
    assert len(plan) == 9
    assert {i: need for i, need in plan} == {i: by_source[i] == "pB" for i in by_source}
    summary = ri.scoring_summary(stage, evals, CFG)
    assert summary["missing_predecessors"] == ["pB"]
    assert summary["missing_originals"] == ["cA", "cB", "r1"]  # stale bytes, absent, other config
    assert (summary["n_variants"], summary["n_P"], summary["n_D"]) == (9, 1, 3)
    assert summary["minutes"] == pytest.approx((9 + 1 + 3) * 1.04)
    # no identity proof (missing entry_hashes) counts as needing the reference
    (evals / "pA" / "eval.json").write_text(
        json.dumps(json.loads((evals / "pA/eval.json").read_text()) | {"entry_hashes": {}})
    )
    assert ri.scoring_summary(stage, evals, CFG)["missing_predecessors"] == ["pA", "pB"]


def test_score_invokes_batch_on_staged_zoo(tmp_path):
    entry = _entry(tmp_path / "zoo", "r1", _root_delta())
    built = ri.build_variants(entry, _base(), seed=1, stage_root=tmp_path / "stage")
    cmds = ri.score_commands(tmp_path / "stage", Path("configs/eval_tf_v1.yaml"))
    assert cmds[0][1:5] == ["-m", "wsp.eval.batch", "--zoo", str(tmp_path / "stage")]
    assert cmds[0][5:7] == ["--config", "configs/eval_tf_v1.yaml"]
    assert cmds[0][cmds[0].index("--ids") + 1].split(",") == sorted(v.variant_id for v in built)
