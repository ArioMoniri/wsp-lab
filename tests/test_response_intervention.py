"""Part A intervention runner (PREREG A5.3): selection, rotation/ladder variants, manifests, scoring plan.

Synthetic manifest rows and tiny bf16 state dicts only: no model, no data/, CPU.
"""

import argparse
import hashlib
import json
import math
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


# ------------------------------------------------------ registered selection, alias holders, reuse proofs


def _selection_doc(rows, n_lora, n_chain):
    ids = sorted(r["edit_id"] for r in rows)
    return {
        "registered": {
            "status": "selected",
            "edit_ids": ids,
            "selected": rows,
            "counts": {},
            "quotas": {},
            "registered_exclusions": {},
        },
        "procedural_freeze": {
            "registered_edit_ids_sorted_sha256": hashlib.sha256(
                json.dumps(ids).encode()
            ).hexdigest()
        },
    }


def _sel_row(i, fam, comp, pred="base", tertile=0):
    return {
        "edit_id": i,
        "component_id": comp,
        "family": fam,
        "tertile": tertile,
        "tertile_label": ("low", "middle", "high")[tertile],
        "rank": 0,
        "effective_norm": 1.0,
        "predecessor_id": pred,
    }


def test_registered_selection_is_loaded_and_checked_not_rerun(tmp_path):
    rows = [_sel_row("r1", "lora_sft", "K1"), _sel_row("c1", "chain", "K2", pred="p1")]
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(_selection_doc(rows, 1, 1)))
    sel, ledger = ri.load_registered_selection(path, n_lora=1, n_chain=1)
    assert [s.edit_id for s in sel] == ["c1", "r1"]
    assert sel[0].predecessor_id == "p1" and sel[1].predecessor_id == "base"
    assert ledger["selection_json_sha256"] == _sha(path)
    with pytest.raises(ri.SelectionBlocked, match="family counts"):
        ri.load_registered_selection(path, n_lora=2, n_chain=1)
    doc = _selection_doc(rows, 1, 1)
    doc["procedural_freeze"]["registered_edit_ids_sorted_sha256"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(ri.SelectionBlocked, match="procedural freeze"):
        ri.load_registered_selection(path, n_lora=1, n_chain=1)
    doc = _selection_doc(rows + [_sel_row("r2", "lora_sft", "K1")], 2, 1)
    path.write_text(json.dumps(doc))
    with pytest.raises(ri.SelectionBlocked, match="components not distinct"):
        ri.load_registered_selection(path, n_lora=2, n_chain=1)


def _alias_zoo(tmp_path):
    """Base, a holder H with bytes, its alias A (edit.json only) and chain step c1 with parent A."""
    base = _base()
    zoo = tmp_path / "zoo"
    zoo.mkdir()
    (zoo / "manifest.jsonl").write_text(
        json.dumps({"type": "header", "base_model": "m", "base_revision": REV}) + "\n"
    )
    cum_p = {"layers.1.o.weight": _lowrank((6, 6), [0, 1], [2, 3], 1, 20)}
    holder = _entry(zoo, "H", cum_p, family="random_dir")
    edit = json.loads((holder / "edit.json").read_text()) | {
        "params": {},
        "seed": 0,
        "base_model": "m",
    }  # a full descriptor: prepare loads the predecessor state through wsp.zoo.apply
    (holder / "edit.json").write_text(json.dumps(edit))
    (zoo / "A").mkdir()
    (zoo / "A" / "edit.json").write_text(
        json.dumps({"id": "A", "family": "chain", "parent_id": None, "base_revision": REV})
    )
    pred = dict(base)
    pred["layers.1.o.weight"] = (base["layers.1.o.weight"].float() + cum_p["layers.1.o.weight"]).to(
        torch.bfloat16
    )
    step = _lowrank((8, 6), [1, 2, 3], [0, 1, 2], 2, 21)
    cum_e = {
        "layers.1.o.weight": pred["layers.1.o.weight"].float() - base["layers.1.o.weight"].float(),
        "layers.0.q.weight": (pred["layers.0.q.weight"].float() + step).to(torch.bfloat16).float()
        - base["layers.0.q.weight"].float(),
    }
    entry = _entry(zoo, "c1", cum_e, family="chain", parent="A", cumulative=True)
    return base, pred, zoo, holder, entry


def test_alias_predecessor_uses_holder_bytes_in_the_plan(tmp_path):
    base, pred, _zoo, holder, entry = _alias_zoo(tmp_path)
    stage, evals = tmp_path / "stage", tmp_path / "evals"
    built = ri.build_variants(
        entry,
        pred,
        seed=3,
        stage_root=stage,
        base_state=base,
        predecessor_id="A",
        predecessor_holder="H",
    )
    rows = [json.loads(x) for x in (stage / "manifest.jsonl").read_text().splitlines()]
    assert {r.get("predecessor_bytes_holder") for r in rows if "id" in r} == {"H"}
    meta = json.loads((built[0].entry_dir / "variant.json").read_text())
    assert meta["predecessor_bytes_holder"] == "H"
    assert meta["lineage"]["n_predecessor_changed"] == 1
    assert meta["lineage"]["n_predecessor_changed_carried_bitwise"] == 1
    # the alias's record was scored on the holder's bytes: reuse is proven against the holder
    _record(evals, "A", hashes={"delta.safetensors": _sha(holder / "delta.safetensors")})
    summary = ri.scoring_summary(stage, evals, CFG)
    assert summary["missing_predecessors"] == [] and summary["missing_originals"] == ["c1"]
    assert summary["predecessor_bytes_holders"] == {"A": "H"}
    proof = next(p for p in summary["reuse_proofs"] if p["id"] == "A")
    assert proof["verdict"] and proof["bytes_holder"] == "H"
    assert (summary["n_priced"], summary["references_to_score"]) == (4, 1)
    # stale holder bytes forbid reuse (fail closed)
    (evals / "A" / "eval.json").write_text(
        json.dumps(
            json.loads((evals / "A/eval.json").read_text())
            | {"entry_hashes": {"delta.safetensors": "0" * 64}}
        )
    )
    assert ri.scoring_summary(stage, evals, CFG)["missing_predecessors"] == ["A"]


def _records(evals, i, windows, repeat_id=None):
    d = evals / i
    p = json.loads((d / "eval.json").read_text()) | {"repeat_id": repeat_id}
    (d / "eval.json").write_text(json.dumps(p))
    (d / "records.jsonl").write_text(
        "".join(
            json.dumps({"benchmark": "wikitext_ppl", "item_id": w, "extra": {"token_ids_hash": h}})
            + "\n"
            for w, h in windows
        )
    )


def test_reuse_proof_window_and_primary_checks(tmp_path):
    entry = _entry(tmp_path / "zoo", "r1", _root_delta())
    stage, evals = tmp_path / "stage", tmp_path / "evals"
    ri.build_variants(entry, _base(), seed=1, stage_root=stage)
    windows = [("0", "h0"), ("1", "h1")]
    tg = tmp_path / "text_groups.json"
    tg.write_text(
        json.dumps({"windows": [{"item_id": w, "token_ids_hash": h} for w, h in windows]})
    )
    assert ri.registered_windows(tg) == windows
    _record(evals, "base")
    _record(evals, "r1", hashes={"delta.safetensors": _sha(entry / "delta.safetensors")})
    for i in ("base", "r1"):
        _records(evals, i, windows)
    summary = ri.scoring_summary(stage, evals, CFG, windows)
    assert (summary["n_P"], summary["n_D"], summary["window_check"]) == (0, 0, True)
    _records(evals, "r1", [("0", "h0"), ("1", "other")])
    proof = next(
        p for p in ri.scoring_summary(stage, evals, CFG, windows)["reuse_proofs"] if p["id"] == "r1"
    )
    assert not proof["verdict"] and proof["failed_checks"] == ["windows"]
    _records(evals, "r1", windows, repeat_id="g1")
    assert ri.scoring_summary(stage, evals, CFG, windows)["missing_originals"] == ["r1"]
    assert ri.scoring_summary(stage, evals, CFG)["missing_originals"] == []  # no window check


def test_score_also_runs_missing_references_on_the_source_zoo(tmp_path):
    base, pred, zoo, _holder, entry = _alias_zoo(tmp_path)
    stage = tmp_path / "stage"
    ri.build_variants(
        entry,
        pred,
        seed=3,
        stage_root=stage,
        base_state=base,
        predecessor_id="A",
        predecessor_holder="H",
    )
    cmds = ri.score_commands(stage, "cfg.yaml", references=["c1"])
    assert len(cmds) == 2 and cmds[1][3:] == [
        "--zoo",
        str(zoo.resolve()),
        "--config",
        "cfg.yaml",
        "--ids",
        "c1",
    ]
    with pytest.raises(ValueError, match="without own bytes"):
        ri.score_commands(stage, "cfg.yaml", references=["A"])
    assert len(ri.score_commands(stage, "cfg.yaml")) == 1


def test_prepare_cli_stages_registered_selection_with_alias_predecessor(tmp_path, monkeypatch):
    from wsp.eval import batch
    from wsp.features import extract

    base, _pred, zoo, holder, _chain = _alias_zoo(tmp_path)
    _entry(zoo, "r1", _root_delta())
    sel = tmp_path / "selection.json"
    sel.write_text(
        json.dumps(
            _selection_doc(
                [_sel_row("r1", "lora_sft", "K1"), _sel_row("c1", "chain", "K2", pred="A")], 1, 1
            )
        )
    )
    refs = tmp_path / "references.json"
    refs.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "edit_id": "r1",
                        "predecessor_id": "base",
                        "original": {"id": "r1", "bytes_holder": "r1"},
                        "predecessor": {"id": "base", "bytes_holder": None},
                    },
                    {
                        "edit_id": "c1",
                        "predecessor_id": "A",
                        "original": {"id": "c1", "bytes_holder": "c1"},
                        "predecessor": {"id": "A", "bytes_holder": "H"},
                    },
                ]
            }
        )
    )
    evals = tmp_path / "evals"
    _record(evals, "base")
    _record(evals, "A", hashes={"delta.safetensors": _sha(holder / "delta.safetensors")})
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps(CFG))
    monkeypatch.setattr(extract, "load_base_state", lambda m, r: (dict(base), r))
    monkeypatch.setattr(batch, "provenance", lambda: {"git_sha": "g", "source_hash": "s"})
    out, man = tmp_path / "out", tmp_path / "pkg" / "manifest.json"
    argv = ["prepare", "--zoo", str(zoo), "--selection", str(sel), "--references", str(refs),
            "--out", str(out), "--manifest-out", str(man), "--config", str(cfg),
            "--evals", str(evals), "--n-lora", "1", "--n-chain", "1"]  # fmt: skip
    assert ri.main(argv) == 0
    m = json.loads(man.read_text())
    assert m["edit_ids"] == ["c1", "r1"] and len(m["variant_ids"]) == 6
    assert m["predecessor_bytes_holders"] == {"A": "H"}
    assert m["rotation_seeds"] == {i: ri.rotation_seed(i) for i in ("c1", "r1")}
    assert m["scoring"]["missing_predecessors"] == []
    assert m["scoring"]["missing_originals"] == ["c1", "r1"]
    assert m["scoring"]["n_priced"] == 8 and m["scored"] is False
    assert m["coverage"]["n_rotation_pairs_staged"] == 2
    assert {e["pair_verdict"] for e in m["edit_audits"].values()} == {"pass"}
    audits = [
        json.loads(x) for x in (man.parent / "rotation_audits.jsonl").read_text().splitlines()
    ]
    assert len(audits) == m["rotation_audits"]["n_tensors"] == 3  # q in c1; q, v in r1
    assert all(a["verdict"] == "pass" for a in audits)
    for vid, files in m["delta_sha256"].items():
        assert files["delta.safetensors"] == _sha(out / "zoo" / vid / "delta.safetensors")
    assert not (zoo / "A" / "delta.safetensors").exists()  # canonical zoo untouched


# -------------------------------------------------------------------- F10 audit rule and chain-ladder flag


def _grid_base():
    """Predecessor values at 1.5: bf16 spacing 2^-7 on both sides of every coordinate."""
    return {"a.weight": torch.full((40, 56), 1.5, dtype=torch.bfloat16),
            "b.weight": torch.full((48, 48), 1.5, dtype=torch.bfloat16)}  # fmt: skip


def _low_kappa_delta():
    g = torch.Generator().manual_seed(3)
    a, b = torch.randn(40, 4, generator=g), torch.randn(4, 56, generator=g)
    return {"a.weight": a @ b * (2**-7 * 0.8) / 4}


def _flips():
    perm = torch.randperm(48, generator=torch.Generator().manual_seed(0))
    flips = torch.zeros(48, 48)
    flips[torch.arange(48), perm] = 2**-7
    return flips


def _meta(v):
    return json.loads((v.entry_dir / "variant.json").read_text())


def _staged_rows(stage):
    return {
        r["id"]
        for r in map(json.loads, (stage / "manifest.jsonl").read_text().splitlines())
        if "id" in r
    }


def _assert_pair_excluded(rot, ladders, stage):
    """Excluded pair: ledger entry kept, no bytes, no staged row; its ladders stay eligible."""
    assert rot.excluded and (rot.entry_dir / "variant.json").exists()
    for name in ("delta.safetensors", "increment.safetensors", "edit.json"):
        assert not (rot.entry_dir / name).exists()
    assert rot.variant_id not in _staged_rows(stage)
    assert _meta(rot)["coverage"]["n_rotated"] == 0
    for v in ladders:
        assert not v.excluded and (v.entry_dir / "delta.safetensors").exists()
        assert v.variant_id in _staged_rows(stage)
    sel = [ri.Selection(rot.source_id, "K", "lora_sft", 0, 0, 1.0, rot.predecessor_id)]
    man = ri.manifest_for(sel, [rot, *ladders], code_sha="x")
    assert (
        rot.variant_id in man["excluded_variant_ids"] and rot.variant_id not in man["delta_sha256"]
    )
    assert {v.variant_id for v in ladders} <= set(man["variant_ids"])


def test_audit_rule_switch_records_rule_and_f10_keeps_low_kappa_pair(tmp_path):
    base = _grid_base()
    entry = _entry(tmp_path / "zoo", "k1", _low_kappa_delta())
    old = ri.build_variants(entry, base, seed=11, stage_root=tmp_path / "a", strengths=(),
                            audit_rule="a533")[0]  # fmt: skip
    new = ri.build_variants(entry, base, seed=11, stage_root=tmp_path / "b", strengths=())[0]
    assert old.excluded and old.exclusions == [{"tensor": "a.weight", "reason": "audit_failed"}]
    assert not new.excluded and new.exclusions == []
    m_old, m_new = _meta(old), _meta(new)
    assert (m_old["audit_rule"], m_new["audit_rule"]) == ("a533", "f10")
    assert ri.DEFAULT_AUDIT_RULE == "f10"
    assert set(m_old["tensor_audits"]["a.weight"]) >= set(ri.AUDIT_FIELDS)  # a533 path unchanged
    t = m_new["tensor_audits"]["a.weight"]
    assert t["passes"] is True and t["s31_passes"] is False and t["failed_checks"] == []
    e = m_new["edit_audit"]
    assert e["passes"] and e["sub_resolution_guard"]["passes"] and e["exclusions"] == []
    assert e["log_energy_ratio"] == pytest.approx(math.log(t["energy_ratio"]), rel=1e-9)
    assert e["pooled_norm_mismatch"] == pytest.approx(abs(t["norm_mismatch_signed"]), rel=1e-9)
    assert e["noise_calibrated_passes"] is t["noise_calibrated_passes"]
    assert e["in_subset_pooled_norm_2e-2"] is (e["pooled_norm_mismatch"] <= 2e-2)
    assert e["in_subset_noise_calibrated"] is e["noise_calibrated_passes"]
    # same construction: the rotation bytes do not depend on the audit rule
    assert old.variant_id == new.variant_id
    with pytest.raises(ValueError, match="audit rule"):
        ri.build_variants(entry, base, seed=11, stage_root=tmp_path / "c", audit_rule="b")
    sel = [ri.Selection("k1", "K", "lora_sft", 0, 0, 1.0, "base")]
    man = ri.manifest_for(sel, [new], code_sha="x")
    assert man["audit_rule"] == "f10" and man["variants"][0]["audit_rule"] == "f10"
    assert man["f10_subsets"] == {"pooled_norm_le_2e-2": ["k1"] if e["in_subset_pooled_norm_2e-2"] else [],
                                  "noise_calibrated": ["k1"] if e["noise_calibrated_passes"] else []}  # fmt: skip
    assert ri.manifest_for(sel, [old], code_sha="x")["audit_rule"] == "a533"


def test_f10_logs_the_complete_a533_verdict_without_using_it(tmp_path):
    base = _grid_base()
    delta = _low_kappa_delta() | {"b.weight": _lowrank((48, 48), range(48), range(48), 30, 4, 0.5)}
    entry = _entry(tmp_path / "zoo", "k2", delta)
    old = ri.build_variants(entry, base, seed=11, stage_root=tmp_path / "a", strengths=(),
                            audit_rule="a533")[0]  # fmt: skip
    new = ri.build_variants(entry, base, seed=11, stage_root=tmp_path / "b", strengths=())[0]
    m_old, m_new = _meta(old), _meta(new)
    assert old.excluded and not new.excluded
    for k, a in m_old["tensor_audits"].items():  # the logged verdict is the explicit a533 run's
        logged = m_new["tensor_audits"][k]["a533"]
        for f in ri.A533_FIELDS:
            assert logged[f] == pytest.approx(a[f], rel=1e-6, abs=1e-12), (k, f)
        assert logged["criteria"]["norm"] is (a["norm_mismatch"] <= 2e-2)
    assert m_new["tensor_audits"]["a.weight"]["a533"]["passes"] is False
    assert m_new["tensor_audits"]["b.weight"]["a533"]["passes"] is True
    d = m_new["edit_audit"]["a533_diagnostic"]
    assert d["passes"] is m_old["edit_audit"]["passes"] is False
    assert d["would_exclude"] == old.exclusions and d["failing_tensors"] == ["a.weight"]
    rows = ri.audit_rows([new], {k: v.shape for k, v in base.items()})
    assert [r["a533_passes"] for r in rows] == [False, True] and all(
        r["verdict"] == "pass" for r in rows
    )


def test_original_reconstruction_only_failure_stays_visible_under_f10(tmp_path, monkeypatch):
    """The fp32 SVD reconstruction artefact of the staging (1.05e-5 to 1.40e-5) fails a533 alone."""
    real = ri.rotation.block_svd

    def fp32_artefact(D, rows, cols, rank_rtol=1e-3):
        u, s, vh, rank = real(D, rows, cols, rank_rtol)
        return u, s * (1 + 2e-5), vh, rank

    monkeypatch.setattr(ri.rotation, "block_svd", fp32_artefact)
    base, delta = _base(), _root_delta()
    entry = _entry(tmp_path / "zoo", "e9", delta)
    old = ri.build_variants(entry, base, seed=7, stage_root=tmp_path / "a", strengths=(),
                            audit_rule="a533")[0]  # fmt: skip
    new = ri.build_variants(entry, base, seed=7, stage_root=tmp_path / "b", strengths=())[0]
    assert old.excluded and all(a["s31_passes"] for a in _meta(old)["tensor_audits"].values())
    assert not new.excluded
    for a in _meta(new)["tensor_audits"].values():
        assert a["passes"] and a["s31_passes"] and a["reconstruction_residual"] < 1e-12
        assert a["a533"]["passes"] is False and a["a533"]["s31_passes"] is True
        assert [c for c, ok in a["a533"]["criteria"].items() if not ok] == ["reconstruction"]
    d = _meta(new)["edit_audit"]["a533_diagnostic"]
    assert d["passes"] is False and d["s31_passes"] is True
    assert d["criterion_failures"]["reconstruction"] == 2 and d["criterion_failures"]["norm"] == 0


def test_f10_sub_resolution_guard_excludes_the_pair(tmp_path):
    """All of R in sparse single-ulp flips: every tensor check passes, rounding removes R, the pair goes."""
    base = _grid_base()
    stage = tmp_path / "st"
    rot, *ladders = ri.build_variants(_entry(tmp_path / "zoo", "s1", {"b.weight": _flips()}), base,
                                      seed=4, stage_root=stage)  # fmt: skip
    meta = _meta(rot)
    assert meta["tensor_audits"]["b.weight"]["passes"] is True
    assert rot.exclusions == [{"tensor": None, "reason": "sub_resolution"}]
    g = meta["edit_audit"]["sub_resolution_guard"]
    assert g["mismatch_signed"] < -0.5 and not meta["edit_audit"]["passes"]
    _assert_pair_excluded(rot, ladders, stage)
    t = meta["tensor_audits"]["b.weight"]  # logged on the exclusion too
    assert {"logged_d", "logged_r", "nonzero_d", "nonzero_r"} <= set(t)
    summary = ri.edit_audit_summary([rot])["s1"]
    assert summary["pair_verdict"] == "excluded" and summary["unrotated_tensors"] == 1


def test_f10_guard_pools_unequal_tensors_and_lets_a_small_one_round_away(tmp_path):
    base = _grid_base()
    g = torch.Generator().manual_seed(5)
    big = torch.randn(40, 3, generator=g) @ torch.randn(3, 56, generator=g) * 0.05
    delta = {"a.weight": big, "b.weight": _flips()}
    rot = ri.build_variants(_entry(tmp_path / "zoo", "s2", delta), base, seed=4,
                            stage_root=tmp_path / "st", strengths=())[0]  # fmt: skip
    assert not rot.excluded
    t, e = _meta(rot)["tensor_audits"], _meta(rot)["edit_audit"]
    assert t["b.weight"]["energy_ratio"] < 0.25  # alone, this tensor would fail the guard
    # independent pooled reference from the stored states and the written bytes
    written = load_file(rot.entry_dir / "delta.safetensors")
    D = {
        k: (base[k].float() + delta[k]).to(torch.bfloat16).float() - base[k].float() for k in delta
    }
    d = torch.cat([D[k].flatten() for k in sorted(D)]).double().norm()
    r = torch.cat([written[k].flatten() for k in sorted(D)]).double().norm()
    assert e["sub_resolution_guard"]["mismatch_signed"] == pytest.approx(
        float((r - d) / d), rel=1e-12
    )
    assert e["pooled_energy_ratio"] == pytest.approx(float(r**2 / d**2), rel=1e-12)
    assert e["log_energy_ratio"] == pytest.approx(math.log(float(r**2 / d**2)), rel=1e-12)


def test_f10_guard_excludes_on_the_upper_side(tmp_path, monkeypatch):
    real = ri.rotation.f10_tensor_audit

    def inflated(*a, **k):
        out = real(*a, **k)
        return out | {"r_eff_norm": out["d_norm"] * 1.6}

    monkeypatch.setattr(ri.rotation, "f10_tensor_audit", inflated)
    stage = tmp_path / "st"
    rot, *ladders = ri.build_variants(_entry(tmp_path / "zoo", "u1", _root_delta()), _base(),
                                      seed=7, stage_root=stage)  # fmt: skip
    assert rot.exclusions == [{"tensor": None, "reason": "sub_resolution"}]
    assert _meta(rot)["edit_audit"]["sub_resolution_guard"]["mismatch_signed"] == pytest.approx(0.6)
    _assert_pair_excluded(rot, ladders, stage)


def test_f10_rounding_failure_is_listed_with_its_reason(tmp_path, monkeypatch):
    """A wrong grid for one tensor (its rounding moved by one ulp in construction and replay alike):
    the replay agrees, the rounding bound does not."""
    base, delta = _base(), _root_delta()
    real = ri.rotation.apply_and_round

    def nudged(W, X):
        new, eff = real(W, X)
        if W.shape == (6, 6):
            new = new.clone()
            new[0, 1] = (new[0, 1].float() + ri.rotation.ulp_bf16(new[0, 1])).to(torch.bfloat16)
            eff = ri.rotation.realised_increment(W, new)
        return new, eff

    monkeypatch.setattr(ri.rotation, "apply_and_round", nudged)
    stage = tmp_path / "st"
    rot, *ladders = ri.build_variants(_entry(tmp_path / "zoo", "f1", delta), base, seed=7,
                                      stage_root=stage)  # fmt: skip
    assert rot.exclusions == [{"tensor": "layers.0.v.weight", "reason": "rounding_bound_failed"}]
    t = _meta(rot)["tensor_audits"]
    assert t["layers.0.v.weight"]["failed_checks"] == ["rounding_bound"]
    assert t["layers.0.v.weight"]["rounding_bound_violations"] == 1
    assert t["layers.0.q.weight"]["passes"]
    _assert_pair_excluded(rot, ladders, stage)


# --- replay against the written, reloaded bytes


def _chain_case(tmp_path, *, same_tensor=True):
    """A chain step whose predecessor changed the edited tensor itself (same_tensor) and another tensor
    that the step carries bitwise."""
    base = _base()
    pred = dict(base)
    pred["layers.1.o.weight"] = (base["layers.1.o.weight"].float()
                                 + _lowrank((6, 6), [0, 1], [2, 3], 1, 20)).to(torch.bfloat16)  # fmt: skip
    if same_tensor:
        pred["layers.0.q.weight"] = (base["layers.0.q.weight"].float()
                                     + _lowrank((8, 6), [2, 3, 4], [1, 2], 1, 22)).to(torch.bfloat16)  # fmt: skip
    cum_p = {
        k: pred[k].float() - base[k].float() for k in base if not torch.equal(pred[k], base[k])
    }
    succ_q = (pred["layers.0.q.weight"].float() + _lowrank((8, 6), [1, 2, 3], [0, 1, 2], 2, 21)).to(
        torch.bfloat16
    )
    cum_e = cum_p | {"layers.0.q.weight": succ_q.float() - base["layers.0.q.weight"].float()}
    zoo = tmp_path / "zoo"
    _entry(zoo, "p1", cum_p, family="chain", cumulative=True)
    entry = _entry(zoo, "c1", cum_e, family="chain", parent="p1", cumulative=True)
    kw = {"base_state": base, "predecessor_id": "p1"}
    return base, pred, entry, kw


def test_f10_chain_replay_reloads_written_files_and_checks_the_successor(tmp_path):
    base, pred, entry, kw = _chain_case(tmp_path)
    assert not torch.equal(pred["layers.0.q.weight"], base["layers.0.q.weight"])
    rot = ri.build_variants(entry, pred, seed=3, stage_root=tmp_path / "st", strengths=(), **kw)[0]
    assert not rot.excluded
    meta = _meta(rot)
    assert meta["edited_tensors"] == ["layers.0.q.weight"]
    assert meta["carried_roundtrip"]["layers.1.o.weight"]["passes"]
    a = meta["tensor_audits"]["layers.0.q.weight"]
    assert a["replay_bits_equal"] and a["scorer_delta_bits_equal"] and a["replay_threads"] == 4
    env = meta["replay_environment"]
    assert (
        env["replay_threads"] == 4
        and env["torch_threads"] == 4
        and env["torch"] == torch.__version__
    )
    # independent: fresh seed, rotate, round at the predecessor, compare with the reloaded bytes
    inc = load_file(rot.entry_dir / "increment.safetensors")
    cum = load_file(rot.entry_dir / "delta.safetensors")
    q = "layers.0.q.weight"
    D = pred[q].float()  # recompute D_t from the stored states
    D = (base[q].float() + load_file(entry / "delta.safetensors")[q]).to(torch.bfloat16).float() - D
    rows, cols = ri.rotation.changed_block(D)
    R = ri.rotation.rotate_block(D, rows, cols, 3)
    state, eff = ri.rotation.apply_and_round(pred[q], R)
    assert ri.rotation.bits_equal(inc[q], eff)
    assert ri.rotation.bits_equal((base[q].float() + cum[q]).to(torch.bfloat16), state)
    o = "layers.1.o.weight"
    assert ri.rotation.bits_equal((base[o].float() + cum[o]).to(torch.bfloat16), pred[o])


def _corrupting_save(monkeypatch, target_vid, name, corrupt):
    real = ri.save_file

    def save(tensors, path, *a, **k):
        real(tensors, path, *a, **k)
        p = Path(path)
        if p.parent.name == target_vid and p.name == name:
            corrupt(p)

    monkeypatch.setattr(ri, "save_file", save)


def _signed_zero(key):
    def corrupt(p):
        x = load_file(p)
        t = x[key].clone()
        idx = (t == 0).nonzero()[0]
        t[tuple(idx)] = -0.0
        save_file(x | {key: t}, str(p))

    return corrupt


def _edit_tensor(key, f):
    def corrupt(p):
        x = load_file(p)
        out = f(x[key])
        save_file(
            x | {key: out} if out is not None else {k: v for k, v in x.items() if k != key}, str(p)
        )

    return corrupt


def _nudge(t):
    t = t.clone()
    i = (t != 0).nonzero()[0]
    t[tuple(i)] = t[tuple(i)] * 2
    return t


CORRUPTIONS = {
    "signed_zero": None,  # filled per key
    "value": lambda t: _nudge(t),
    "missing_key": lambda t: None,
    "dtype": lambda t: t.to(torch.bfloat16),
    "shape": lambda t: t[:-1].contiguous(),
}


def _corruption(kind, key):
    if kind == "signed_zero":
        return _signed_zero(key)
    if kind == "missing_file":
        return lambda p: p.unlink()
    return _edit_tensor(key, CORRUPTIONS[kind])


@pytest.mark.parametrize(
    "kind", ["signed_zero", "value", "missing_key", "dtype", "shape", "missing_file"]
)
@pytest.mark.parametrize(
    "target", ["root:delta", "chain:increment", "chain:delta", "chain:carried"]
)
def test_post_save_corruption_excludes_the_pair_as_replay_failed(
    tmp_path, monkeypatch, target, kind
):
    family, name = target.split(":")
    if family == "root":
        base, entry, kw, pred = _base(), _entry(tmp_path / "zoo", "r1", _root_delta()), {}, None
        key, source, file = "layers.0.v.weight", "r1", "delta.safetensors"
        pred = base
    else:
        base, pred, entry, kw = _chain_case(tmp_path)
        source = "c1"
        key = "layers.1.o.weight" if name == "carried" else "layers.0.q.weight"
        file = "increment.safetensors" if name == "increment" else "delta.safetensors"
    if kind == "missing_file" and name == "carried":
        pytest.skip("the carried tensor lives in the cumulative delta file, covered by chain:delta")
    vid = ri.variant_id(source, "rotation", 1.0, 3)
    _corrupting_save(monkeypatch, vid, file, _corruption(kind, key))
    stage = tmp_path / "st"
    rot, *ladders = ri.build_variants(entry, pred, seed=3, stage_root=stage, **kw)
    assert rot.variant_id == vid
    assert rot.exclusions and {x["reason"] for x in rot.exclusions} == {"replay_failed"}
    if name == "carried":
        assert rot.exclusions == [
            {"tensor": key, "reason": "replay_failed", "detail": "carried_roundtrip"}
        ]
    else:
        assert key in {x["tensor"] for x in rot.exclusions}
        a = _meta(rot)["tensor_audits"][key]
        assert {"replay", "scorer_roundtrip"} & set(a["failed_checks"])
        assert a["reason"] == "replay_failed"
    _assert_pair_excluded(rot, ladders, stage)


# --- failing closed on incomplete or nonfinite audit information


def _tamper_audits(monkeypatch, f):
    real = ri.rotation.f10_tensor_audit

    def wrapped(D, *a, **k):
        return f(real(D, *a, **k), D)

    monkeypatch.setattr(ri.rotation, "f10_tensor_audit", wrapped)


def _first_tensor(out, D):
    return D.shape == (8, 6)  # layers.0.q.weight in _root_delta


TAMPERS = {
    "one_audit_missing": lambda out, D: None if _first_tensor(out, D) else out,
    "all_audits_missing": lambda out, D: None,
    "metric_deleted": lambda out, D: (
        {k: v for k, v in out.items() if k != "orthogonality_error"}
        if _first_tensor(out, D)
        else out
    ),
    "metric_nonfinite": lambda out, D: (
        out | {"rounding_bound_violations": float("nan")} if _first_tensor(out, D) else out
    ),
    "norm_nonfinite": lambda out, D: (
        out | {"d_norm": float("inf")} if _first_tensor(out, D) else out
    ),
    "check_not_boolean": lambda out, D: (
        out | {"checks": out["checks"] | {"support": 1}} if _first_tensor(out, D) else out
    ),
}


@pytest.mark.parametrize("tamper", sorted(TAMPERS))
def test_incomplete_or_nonfinite_audits_exclude_the_pair(tmp_path, monkeypatch, tamper):
    _tamper_audits(monkeypatch, TAMPERS[tamper])
    stage = tmp_path / "st"
    rot, *ladders = ri.build_variants(_entry(tmp_path / "zoo", "x1", _root_delta()), _base(),
                                      seed=7, stage_root=stage)  # fmt: skip
    _assert_pair_excluded(rot, ladders, stage)
    q = "layers.0.q.weight"
    reasons = {x["tensor"]: x for x in rot.exclusions}
    want = {
        "one_audit_missing": "construction_failed",
        "all_audits_missing": "construction_failed",
        "metric_deleted": "construction_failed",
        "metric_nonfinite": "rounding_bound_failed",
        "check_not_boolean": "support_failed",
    }
    if tamper == "norm_nonfinite":
        assert reasons[None] == {
            "tensor": None,
            "reason": "sub_resolution",
            "detail": "guard_incomplete",
        }
    else:
        assert reasons[q]["reason"] == want[tamper] and reasons[q]["detail"]
    if tamper == "all_audits_missing":
        assert set(reasons) == {q, "layers.0.v.weight", None}
        assert reasons[None]["detail"] == "guard_incomplete"
    meta = _meta(rot)  # strict JSON: no NaN was written
    json.dumps(meta, allow_nan=False)
    assert meta["edit_audit"]["passes"] is False and meta["exclusions"] == rot.exclusions


# --- F10.4 pooled diagnostics


def test_f10_pooled_diagnostics_are_independent_aggregates(tmp_path):
    base = {"a.weight": _w((40, 56), 3), "b.weight": _w((12, 20), 4)}
    delta = {"a.weight": _lowrank((40, 56), range(40), range(56), 8, 5, 0.02),
             "b.weight": _lowrank((12, 20), range(3, 12), range(20), 2, 6, 0.004)}  # fmt: skip
    rot = ri.build_variants(_entry(tmp_path / "zoo", "p9", delta), base, seed=13,
                            stage_root=tmp_path / "st", strengths=())[0]  # fmt: skip
    assert not rot.excluded
    e, t = _meta(rot)["edit_audit"], _meta(rot)["tensor_audits"]
    written = load_file(rot.entry_dir / "delta.safetensors")
    D, R, ulp2 = {}, {}, 0.0
    for k in sorted(delta):
        D[k] = (base[k].float() + delta[k]).to(torch.bfloat16).float() - base[k].float()
        rows, cols = ri.rotation.changed_block(D[k])
        R[k] = ri.rotation.rotate_block(D[k], rows, cols, 13)
        ulp2 += float((ri.rotation.ulp_bf16(base[k][rows[:, None], cols]) ** 2).sum())
    cat = lambda x: torch.cat([x[k].flatten().double() for k in sorted(delta)])
    dn = float(cat(D).norm())
    assert e["pooled_d_norm"] == pytest.approx(dn, rel=1e-12)
    assert e["pooled_r_eff_norm"] == pytest.approx(float(cat(written).norm()), rel=1e-12)
    assert e["pooled_rounding_error_rel"] == pytest.approx(
        float((cat(written) - cat(R)).norm()) / dn, rel=1e-9
    )
    assert e["pooled_noise_floor"] == pytest.approx(math.sqrt(ulp2 / 12), rel=1e-12)
    assert e["pooled_kappa"] == pytest.approx(dn / math.sqrt(ulp2), rel=1e-12)
    assert e["pooled_eps_hat"] == pytest.approx(min(1, math.sqrt(ulp2 / 12) / dn), rel=1e-12)
    raw = {k: delta[k] for k in sorted(delta)}
    assert e["pooled_eps_D"] == pytest.approx(float((cat(D) - cat(raw)).norm()) / dn, rel=1e-9)
    assert e["eps_D_status"] == "root_stored_delta"
    sd = {k: torch.linalg.svdvals(D[k]) for k in D}
    sr = {k: torch.linalg.svdvals(written[k]) for k in D}
    num = sum(float((sr[k] - sd[k]).double().norm()) ** 2 for k in D)
    den = sum(float(sd[k].double().norm()) ** 2 for k in D)
    assert e["pooled_spectrum_mismatch"] == pytest.approx(math.sqrt(num / den), rel=1e-6)
    for f in (
        "rank_d",
        "rank_r",
        "noise_rank_d",
        "noise_rank_r",
        "prerounding_rank_d",
        "rank_n_near",
    ):
        key = "sum_" + f
        assert e[key] == sum(t[k][f] for k in t)
    assert e["tau_by_tensor"] == {k: t[k]["tau"] for k in sorted(t)}
    assert e["noise_calibrated_passes"] is all(t[k]["noise_calibrated_passes"] for k in t)
    assert set(ri.F10_POOLED_DEFINITIONS) == set(e["aggregation"])
    rows = ri.audit_rows([rot], {k: v.shape for k, v in base.items()})
    for r in rows:
        assert set(ri.F10_AUDIT_FIELDS) - {"fail_closed_detail"} <= set(r)
        assert r["audit_rule"] == "f10" and "a533_passes" in r


def test_f10_non_root_eps_D_is_unavailable(tmp_path):
    _base_state, pred, entry, kw = _chain_case(tmp_path)
    rot = ri.build_variants(entry, pred, seed=3, stage_root=tmp_path / "st", strengths=(), **kw)[0]
    e, t = _meta(rot)["edit_audit"], _meta(rot)["tensor_audits"]
    assert e["pooled_eps_D"] is None and e["eps_D_status"] == "unavailable_non_root"
    assert all(a["eps_D"] is None for a in t.values())


# --- F10.9 ladders


def test_chain_ladders_are_flagged_realised_increment(tmp_path):
    base, pred, entry, kw = _chain_case(tmp_path, same_tensor=False)
    zoo = tmp_path / "zoo"
    chain = ri.build_variants(entry, pred, seed=3, stage_root=tmp_path / "st", **kw)
    root = ri.build_variants(
        _entry(zoo, "r1", _root_delta()), base, seed=3, stage_root=tmp_path / "st"
    )
    q = "layers.0.q.weight"
    root_chain_delta = {q: _lowrank((8, 6), [1, 2, 3], [0, 1, 2], 2, 31)}
    rc = ri.build_variants(_entry(zoo, "rc", root_chain_delta, family="chain", cumulative=True), base,
                           seed=3, stage_root=tmp_path / "st")  # fmt: skip
    for v in chain + root + rc:
        meta = _meta(v)
        if v.kind == "scale":
            assert meta["realised_increment_ladder"] is (v.source_id == "c1")
            assert v.realised_increment_ladder is (v.source_id == "c1")
        else:
            assert "realised_increment_ladder" not in meta
    # non-root: exactly bf16(W_t + s D_t) - W_t per strength, D_t the realised increment
    D = (base[q].float() + load_file(entry / "delta.safetensors")[q]).to(torch.bfloat16).float()
    D = D - pred[q].float()
    for v in chain[1:]:
        inc = load_file(v.entry_dir / "increment.safetensors")[q]
        want = (pred[q].float() + v.strength * D).to(torch.bfloat16).float() - pred[q].float()
        assert ri.rotation.bits_equal(inc, want), v.strength
    # roots scale the stored raw delta, not the realised increment (they differ here)
    for v in root[1:]:
        k = "layers.0.v.weight"
        raw = _root_delta()[k]
        Dk = (base[k].float() + raw).to(torch.bfloat16).float() - base[k].float()
        got = load_file(v.entry_dir / "delta.safetensors")[k]
        want = (base[k].float() + v.strength * raw).to(torch.bfloat16).float() - base[k].float()
        alt = (base[k].float() + v.strength * Dk).to(torch.bfloat16).float() - base[k].float()
        assert ri.rotation.bits_equal(got, want) and not torch.equal(got, alt)
    sel = [ri.Selection("c1", "K1", "chain", 0, 0, 1.0, "p1"),
           ri.Selection("r1", "K2", "lora_sft", 0, 0, 1.0, "base"),
           ri.Selection("rc", "K3", "chain", 0, 0, 1.0, "base")]  # fmt: skip
    man = ri.manifest_for(sel, chain + root + rc, code_sha="x")
    flagged = {v["variant_id"] for v in man["variants"] if v.get("realised_increment_ladder")}
    assert flagged == {v.variant_id for v in chain if v.kind == "scale"}
    assert man["realised_increment_ladder_variant_ids"] == sorted(flagged)
    for v in man["variants"]:
        assert ("realised_increment_ladder" in v) is (v["kind"] == "scale")


def test_f10_chain_rotation_that_cannot_round_trip_is_excluded_not_raised(tmp_path):
    """Chair ruling: a scorer round-trip failure of the `0027efdf` type is an F10 exclusion
    (`replay_failed`), not an error."""
    base = _base()
    q = base["layers.0.q.weight"].clone()
    q[1, 0] = 1e5  # large base value that the predecessor sets to exactly zero
    base["layers.0.q.weight"] = q
    pred = dict(base)
    pq = q.clone()
    pq[1, 0] = 0.0
    pred["layers.0.q.weight"] = pq
    cum_p = {"layers.0.q.weight": pq.float() - q.float()}
    succ = (pq.float() + _lowrank((8, 6), [1, 2, 3], [0, 1, 2], 2, 21)).to(torch.bfloat16)
    zoo = tmp_path / "zoo"
    _entry(zoo, "p1", cum_p, family="chain", cumulative=True)
    entry = _entry(zoo, "c1", {"layers.0.q.weight": succ.float() - q.float()}, family="chain",
                   parent="p1", cumulative=True)  # fmt: skip
    kw = {"stage_root": tmp_path / "st", "base_state": base, "predecessor_id": "p1"}
    rot = ri.build_variants(entry, pred, seed=3, strengths=(), **kw)[0]
    assert rot.exclusions == [{"tensor": "layers.0.q.weight", "reason": "replay_failed"}]
    t = _meta(rot)["tensor_audits"]["layers.0.q.weight"]
    assert t["failed_checks"] == ["scorer_roundtrip"] and t["replay_bits_equal"]
    _assert_pair_excluded(rot, [], tmp_path / "st")
    with pytest.raises(RuntimeError, match="round-trip"):  # the registered path is unchanged
        ri.build_variants(entry, pred, seed=3, strengths=(), audit_rule="a533",
                          **kw | {"stage_root": tmp_path / "st2"})  # fmt: skip


# --- F10.7 rotation-only preparation: no evaluation reads, ladders untouched


def _staged_part_a(tmp_path, monkeypatch):
    """A registered two-edit staging (as Part A was staged: a533, scoring plan from records)."""
    from wsp.eval import batch
    from wsp.features import extract

    base, _pred, zoo, holder, _chain = _alias_zoo(tmp_path)
    _entry(zoo, "r1", _root_delta())
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps(_selection_doc(
        [_sel_row("r1", "lora_sft", "K1"), _sel_row("c1", "chain", "K2", pred="A")], 1, 1)))  # fmt: skip
    refs = tmp_path / "references.json"
    refs.write_text(json.dumps({"rows": [
        {"edit_id": "r1", "predecessor_id": "base", "original": {"id": "r1", "bytes_holder": "r1"},
         "predecessor": {"id": "base", "bytes_holder": None}},
        {"edit_id": "c1", "predecessor_id": "A", "original": {"id": "c1", "bytes_holder": "c1"},
         "predecessor": {"id": "A", "bytes_holder": "H"}}]}))  # fmt: skip
    evals = tmp_path / "evals"
    _record(evals, "base")
    _record(evals, "A", hashes={"delta.safetensors": _sha(holder / "delta.safetensors")})
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps(CFG))
    monkeypatch.setattr(extract, "load_base_state", lambda m, r: (dict(base), r))
    monkeypatch.setattr(batch, "provenance", lambda: {"git_sha": "g", "source_hash": "s"})
    out, man = tmp_path / "runs", tmp_path / "pkg" / "manifest.json"
    common = ["--zoo", str(zoo), "--selection", str(sel), "--references", str(refs), "--out", str(out),
              "--n-lora", "1", "--n-chain", "1"]  # fmt: skip
    argv = ["prepare", *common, "--manifest-out", str(man), "--config", str(cfg), "--evals", str(evals),
            "--audit-rule", "a533"]  # fmt: skip
    assert ri.main(argv) == 0
    return {"zoo": zoo, "evals": evals, "out": out, "man": man, "common": common}


def _forbid_evaluation_access(monkeypatch, evals_dirs):
    import builtins
    import io
    import os
    import subprocess

    roots = [Path(p).resolve() for p in evals_dirs]
    opened = []

    def guard(path):
        if isinstance(path, int):
            return
        p = Path(os.fsdecode(path)).resolve()
        if any(p == r or r in p.parents for r in roots):
            opened.append(p)
            raise AssertionError(f"evaluation path opened: {p}")

    def wrap(f):
        def inner(path, *a, **k):
            guard(path)
            return f(path, *a, **k)

        return inner

    for mod, name in (
        (builtins, "open"),
        (io, "open"),
        (os, "open"),
        (os, "listdir"),
        (os, "scandir"),
    ):
        monkeypatch.setattr(mod, name, wrap(getattr(mod, name)))

    def no_scoring(*a, **k):
        raise AssertionError("scorer or scoring plan invoked")

    for name in (
        "scoring_summary",
        "plan_scoring",
        "_stage",
        "score",
        "score_commands",
        "reuse_proof",
    ):
        if hasattr(ri, name):
            monkeypatch.setattr(ri, name, no_scoring)
    monkeypatch.setattr(subprocess, "run", no_scoring)
    return opened


def _tree_bytes(d):
    return {p.relative_to(d): (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(Path(d).rglob("*"))
            if p.is_file()}  # fmt: skip


def test_rotation_only_prepare_restages_rotations_and_reads_no_evaluations(tmp_path, monkeypatch):
    from wsp.eval import batch

    s = _staged_part_a(tmp_path, monkeypatch)
    staged = json.loads(s["man"].read_text())
    staged_bytes = s["man"].read_bytes()
    staged_audits = (s["man"].parent / "rotation_audits.jsonl").read_bytes()
    stage = s["out"] / "zoo"
    ladders = [v["variant_id"] for v in staged["variants"] if v["kind"] == "scale"]
    before = {vid: _tree_bytes(stage / vid) for vid in ladders}
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    opened = _forbid_evaluation_access(monkeypatch, [s["evals"], batch.REPO / "data" / "evals"])
    argv = ["prepare", "--rotation-only", "--staged-manifest", str(s["man"]), *s["common"],
            "--manifest-out", str(s["man"])]  # fmt: skip
    assert ri.main(argv) == 0
    assert opened == []
    # ladders: bytes and mtimes untouched, re-hashed equal to the staged manifest
    assert {vid: _tree_bytes(stage / vid) for vid in ladders} == before
    m = json.loads(s["man"].read_text())
    re = m["restage"]
    assert re["mode"] == "rotation_only" and re["ladders_rehashed"]["all_equal_staged"]
    assert re["ladders_rehashed"]["n"] == len(ladders) == 4
    # the original staging manifest and audit file are preserved byte for byte, with their hashes
    keep = s["man"].with_name("manifest_a533.json")
    assert keep.read_bytes() == staged_bytes
    assert (
        re["preserved"]["manifest_a533.json"]["sha256"] == hashlib.sha256(staged_bytes).hexdigest()
    )
    assert s["man"].with_name("rotation_audits_a533.jsonl").read_bytes() == staged_audits
    # the new manifest: F10 verdicts, ladders carried, unchanged rotation bytes where both rules pass
    assert m["audit_rule"] == "f10" and m["scored"] is False
    assert all(e["audit_rule"] == "f10" for e in m["edit_audits"].values())
    assert {v["variant_id"] for v in m["variants"]} == {v["variant_id"] for v in staged["variants"]}
    for vid in ladders:
        assert m["delta_sha256"][vid] == staged["delta_sha256"][vid]
        assert m["variant_json_sha256"][vid] == staged["variant_json_sha256"][vid]
    assert re["staged_pass_bitwise"] and all(x["equal"] for x in re["staged_pass_bitwise"].values())
    assert m["scoring"]["missing_originals"] == staged["scoring"]["missing_originals"]
    assert m["scoring"]["reuse_proofs"] == staged["scoring"]["reuse_proofs"]
    assert (
        m["scoring"]["n_priced"]
        == len(m["variant_ids"]) + m["scoring"]["n_P"] + m["scoring"]["n_D"]
    )
    assert re["thread_env"] == {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    audits = [
        json.loads(x) for x in s["man"].with_name("rotation_audits.jsonl").read_text().splitlines()
    ]
    assert {a["audit_rule"] for a in audits} == {"f10"} and m["rotation_audits"]["n_tensors"] == 3
    # idempotent on the preserved files: a second run keeps manifest_a533.json and refuses nothing
    assert ri.main(argv[:3] + [str(keep)] + argv[4:]) == 0
    assert keep.read_bytes() == staged_bytes


def test_rotation_only_prepare_refuses_unsafe_states(tmp_path, monkeypatch, capsys):
    s = _staged_part_a(tmp_path, monkeypatch)
    argv = ["prepare", "--rotation-only", "--staged-manifest", str(s["man"]), *s["common"],
            "--manifest-out", str(tmp_path / "pkg2" / "manifest.json")]  # fmt: skip
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    assert ri.main(argv) == 2 and "OMP_NUM_THREADS" in capsys.readouterr().err
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert ri.main(argv + ["--audit-rule", "a533"]) == 2
    staged = json.loads(s["man"].read_text())
    ladder = next(v["variant_id"] for v in staged["variants"] if v["kind"] == "scale")
    f = s["out"] / "zoo" / ladder / "delta.safetensors"
    good = f.read_bytes()
    f.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))  # one flipped bit in a staged ladder
    assert ri.main(argv) == 2 and "ladder" in capsys.readouterr().err
    assert not (tmp_path / "pkg2" / "manifest.json").exists()
    f.write_bytes(good)
    (tmp_path / "pkg2").mkdir()
    (tmp_path / "pkg2" / "manifest_a533.json").write_text("{}")  # never overwritten
    assert ri.main(argv) == 2 and "overwrite" in capsys.readouterr().err
    assert (tmp_path / "pkg2" / "manifest_a533.json").read_text() == "{}"


def test_dryrun_entrypoint_writes_only_under_its_scratch_root(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "a5_f10_dryrun", Path(__file__).resolve().parents[1] / "scripts" / "a5_f10_dryrun.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = tmp_path / "scratch"
    root.mkdir()
    assert mod.scratch_dir(root / "f10", root) == (root / "f10").resolve()
    for bad in (tmp_path / "elsewhere", root / ".." / "escape", Path("/tmp")):
        with pytest.raises(SystemExit, match="scratch"):
            mod.scratch_dir(bad, root)
    assert mod.DEFAULT_SCRATCH.endswith("runs/a5/f10_dryrun")


# --- F11 supplementary staging: 12 roots merged into the staged 40, no evaluation reads by default

SUPP = {  # (family, tertile) -> next-in-cell candidates after the registered picks, in canonical order
    ("lora_sft", 2): ["l2_x1", "l2_x2", "l2_x3", "l2_x4", "l2_x5"],  # l2_x5: reserve
    ("chain", 0): ["c0_x1", "c0_x2", "c0_x3", "c0_x4", "c0_x5"],  # c0_x1: not a root (skipped)
    ("chain", 2): ["c2_x1", "c2_x2", "c2_x3", "c2_x4", "c2_x5"],  # c2_x4: bytes via alias H2
}
NOT_ROOT = {"c0_x1"}
SUPP_IDS = sorted(i for cand in SUPP.values() for i in [c for c in cand if c not in NOT_ROOT][:4])
FAILS_F10 = "l2_x2"
WINDOWS = [(str(k), f"h{k}") for k in range(4)]


def _registered_40():
    """30 LoRA (10/10/10) and 10 chain (4/3/3) registered ids, by (family, tertile)."""
    picks = {("lora_sft", t): [f"l{t}_{k:02d}" for k in range(10)] for t in range(3)}
    picks |= {("chain", t): [f"c{t}_{k}" for k in range(n)] for t, n in enumerate((4, 3, 3))}
    return picks


def _supp_delta(seed):
    return {
        "layers.0.q.weight": _lowrank((8, 6), [1, 2, 3, 5], [0, 1, 2, 4], 2, seed),
        "layers.0.v.weight": _lowrank((6, 6), [0, 2, 3], [1, 3, 4, 5], 2, seed + 1),
    }


def _fake_staged_40(stage, picks, excluded):
    """A staged Part A of the 40 in the a5-part-a-launch-v1 shape: real files and hashes, fake bytes."""
    stage.mkdir(parents=True)
    sels, variants, rows, audit_lines = [], [], [], []
    for (fam, t), ids in sorted(picks.items()):
        for i in ids:
            sels.append(ri.Selection(i, f"K-{i}", fam, t, 0, 1.0, "base"))
            seed = ri.rotation_seed(i)
            for kind, s in (("rotation", 1.0), ("scale", 0.25), ("scale", 0.5)):
                vid = ri.variant_id(i, kind, s, seed)
                d = stage / vid
                d.mkdir()
                exc = kind == "rotation" and i in excluded
                ex = [{"tensor": None, "reason": "replay_failed"}] if exc else []
                meta = {
                    "kind": kind,
                    "source_id": i,
                    "excluded": exc,
                    "exclusions": ex,
                    "seed": seed,
                }
                if kind == "rotation":
                    meta |= {"audit_rule": "f10", "edited_tensors": ["t"], "tensor_audits": {},
                             "predecessor_bytes_holder": None, "delta_convention": "x", "lineage": {},
                             "coverage": {"n_edited": 1, "n_rotated": 0 if exc else 1,
                                          "n_excluded": int(exc), "fraction": 0.0 if exc else 1.0},
                             "edit_audit": {"passes": not exc, "in_subset_pooled_norm_2e-2": True,
                                            "in_subset_noise_calibrated": not exc}}  # fmt: skip
                    audit_lines.append(
                        json.dumps({"edit_id": i, "verdict": "x" if exc else "pass"})
                    )
                (d / "variant.json").write_text(json.dumps(meta, sort_keys=True))
                if not exc:
                    save_file(
                        {"t": torch.full((2, 2), float(len(variants)))},
                        str(d / "delta.safetensors"),
                    )
                    rows.append({"id": vid, "variant_of": i, "parent_id": None, "base_revision": REV,
                                 "reuse_key": _sha(d / "delta.safetensors")})  # fmt: skip
                variants.append(ri.Variant(vid, kind, s, seed, i, "base", d, exc, ex,
                                           "f10" if kind == "rotation" else None,
                                           False if kind == "scale" else None))  # fmt: skip
    header = {"type": "header", "base_model": "m", "base_revision": REV}
    (stage / "manifest.jsonl").write_text(
        "".join(
            json.dumps(r, sort_keys=True) + "\n"
            for r in [header, *sorted(rows, key=lambda r: r["id"])]
        )
    )
    return sels, variants, "".join(x + "\n" for x in audit_lines)


def _f11_setup(tmp_path, monkeypatch):
    from wsp.eval import batch
    from wsp.features import extract

    base = _base()
    zoo = tmp_path / "zoo"
    zoo.mkdir(parents=True)
    picks = _registered_40()
    excluded = {"l2_00", "c0_0", "c2_0"}  # one lost rotation per depleted cell
    zrows = [{"type": "header", "base_model": "m", "base_revision": REV}]
    for (fam, t), cand in SUPP.items():
        for k, i in enumerate(cand):
            zrows.append({"id": i, "family": fam, "parent_id": "p0" if i in NOT_ROOT else None})
            if i in NOT_ROOT or i not in SUPP_IDS:
                continue
            if i == "c2_x4":
                _entry(zoo, "H2", _supp_delta(300), family="random_dir")
                (zoo / i).mkdir()
                (zoo / i / "edit.json").write_text(json.dumps(
                    {"id": i, "family": "chain", "parent_id": None, "base_revision": REV,
                     "provenance": {"cumulative_snapshot": True}}))  # fmt: skip
            else:
                _entry(
                    zoo, i, _supp_delta(100 + 10 * k + 3 * t), family=fam, cumulative=fam == "chain"
                )
    (zoo / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in zrows))

    out, pkg = tmp_path / "runs", tmp_path / "pkg"
    stage = out / "zoo"
    pkg.mkdir()
    sels, variants, audit_text = _fake_staged_40(stage, picks, excluded)
    (stage / "stage.json").write_text(json.dumps({"source_zoo": str(zoo.resolve())}))
    (pkg / "rotation_audits.jsonl").write_text(audit_text)
    staged = ri.manifest_for(sels, variants, "staged-sha")
    per_edit = ri.edit_audit_summary(variants)
    vids = staged["variant_ids"]
    evals = tmp_path / "evals"
    missing = sorted(s.edit_id for s in sels)[:5]
    staged |= {
        "provenance": {"git_sha": "staged-sha", "source_hash": "s0", "dirty": False},
        "selection_ledger": {"source": "registered"},
        "scoring": {
            "config": "cfg",
            "evals_dir": str(evals),
            "plan": [[v, False] for v in vids],
            "n_variants": len(vids),
            "excluded_variants": staged["excluded_variant_ids"],
            "predecessors": ["base"],
            "originals": staged["edit_ids"],
            "predecessor_bytes_holders": {},
            "missing_predecessors": [],
            "missing_originals": missing,
            "n_P": 0,
            "n_D": 5,
            "references_to_score": 5,
            "n_priced": len(vids) + 5,
            "minutes_per_entry": 1.04,
            "minutes": round((len(vids) + 5) * 1.04, 4),
            "reuse_proofs": [
                {
                    "id": "base",
                    "verdict": True,
                    "bytes_holder": None,
                    "config_hash": effective_config_hash(CFG),
                }
            ],
            "window_check": True,
        },
        "staged_zoo": str(stage),
        "rotation_seeds": {s.edit_id: ri.rotation_seed(s.edit_id) for s in sels},
        "predecessor_bytes_holders": {},
        "edit_audits": per_edit,
        "rotation_audits": {
            "file": "rotation_audits.jsonl",
            "sha256": _sha(pkg / "rotation_audits.jsonl"),
            "n_tensors": 40,
            "n_pass": 37,
        },
        "coverage": ri._coverage(per_edit, sels),
        "variant_json_sha256": {v.variant_id: _sha(v.entry_dir / "variant.json") for v in variants},
        "disk_bytes": 1,
        "scored": False,
        "restage": {"mode": "rotation_only", "audit_rule": "f10", "note": "carried"},
    }
    man = pkg / "manifest.json"
    man.write_text(json.dumps(staged, indent=1, sort_keys=True) + "\n")

    # registered selection.json with the full ledger cells (registered picks first in canonical order)
    reg_rows = [_sel_row(s.edit_id, s.family, s.component_id, tertile=s.tertile) for s in sels]
    doc = _selection_doc(reg_rows, 30, 10)
    cells = {f: [[] for _ in range(3)] for f in ri.FAMILIES}
    for (fam, t), ids in picks.items():
        cells[fam][t] = ids + SUPP.get((fam, t), [])
    doc["registered"]["ledger"] = {"families": {
        f: {"cells": cells[f], "quotas": [len(picks[(f, t)]) for t in range(3)],
            "ordered": [i for c in cells[f] for i in c]} for f in ri.FAMILIES}}  # fmt: skip
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps(doc))

    rows = []
    for (fam, t), cand in SUPP.items():
        for i in cand:
            if i in SUPP_IDS:
                r = _sel_row(i, fam, f"S-{i}", tertile=t)
                r["rank"] = doc["registered"]["ledger"]["families"][fam]["ordered"].index(i)
                r["bytes_via_alias"] = "H2" if i == "c2_x4" else None
                rows.append(r)
    supp = tmp_path / "selection_f11.json"
    supp.write_text(json.dumps({"amendment": "F11", "supplement": {
        "edit_ids": SUPP_IDS, "selected": rows,
        "cells": {f"{f}:{t}": [i for i in c if i in SUPP_IDS] for (f, t), c in SUPP.items()}}}))  # fmt: skip

    # TF-v1 reference records of the supplementary originals (one missing), with the 64-window check
    for i in SUPP_IDS:
        if i == "c0_x3":
            continue
        holder = zoo / ("H2" if i == "c2_x4" else i)
        _record(evals, i, hashes={"delta.safetensors": _sha(holder / "delta.safetensors")})
        _records(evals, i, WINDOWS)
    tg = tmp_path / "text_groups.json"
    tg.write_text(
        json.dumps({"windows": [{"item_id": w, "token_ids_hash": h} for w, h in WINDOWS]})
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps(CFG))

    monkeypatch.setattr(extract, "load_base_state", lambda m, r: (dict(base), r))
    monkeypatch.setattr(batch, "provenance", lambda: {"git_sha": "g11", "source_hash": "s11"})
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    real = ri.rotation.f10_tensor_audit
    bad_seed = ri.rotation_seed(FAILS_F10)

    def audit(*a, **k):  # one supplementary edit fails F10 (upper-side sub-resolution guard)
        out_ = real(*a, **k)
        return out_ | {"r_eff_norm": out_["d_norm"] * 1.6} if a[6] == bad_seed else out_

    monkeypatch.setattr(ri.rotation, "f10_tensor_audit", audit)
    argv = ["prepare", "--supplement", str(supp), "--staged-manifest", str(man), "--zoo", str(zoo),
            "--selection", str(sel), "--out", str(out), "--manifest-out", str(man), "--config", str(cfg),
            "--evals", str(evals), "--text-groups", str(tg)]  # fmt: skip
    return {"argv": argv, "man": man, "stage": stage, "evals": evals, "zoo": zoo, "supp": supp,
            "staged": staged, "vids40": [v.variant_id for v in variants]}  # fmt: skip


def _carried_identical(m, staged):
    for key in ("delta_sha256", "variant_json_sha256", "edit_audits", "rotation_seeds"):
        for k, v in staged[key].items():
            assert json.dumps(m[key][k], sort_keys=True) == json.dumps(v, sort_keys=True), (key, k)
    for key in ("selections", "variants"):
        merged = {json.dumps(r, sort_keys=True) for r in m[key]}
        assert all(json.dumps(r, sort_keys=True) in merged for r in staged[key])
    for key in ("restage", "selection_ledger", "predecessor_bytes_holders", "staged_zoo"):
        assert m[key] == staged[key]


def test_supplement_merges_twelve_roots_into_the_staged_forty(tmp_path, monkeypatch):
    s = _f11_setup(tmp_path, monkeypatch)
    staged, stage = s["staged"], s["stage"]
    staged_bytes, audits_bytes = (
        s["man"].read_bytes(),
        s["man"].with_name("rotation_audits.jsonl").read_bytes(),
    )
    before = {vid: _tree_bytes(stage / vid) for vid in s["vids40"]}
    assert ri.main(s["argv"]) == 0
    m = json.loads(s["man"].read_text())
    # the 40: files untouched, every manifest entry carried byte for byte, pre-F11 copy preserved
    assert {vid: _tree_bytes(stage / vid) for vid in s["vids40"]} == before
    _carried_identical(m, staged)
    pre = s["man"].with_name("manifest_pre_f11.json")
    assert pre.read_bytes() == staged_bytes
    sup = m["supplement"]
    assert sup["pre_f11"]["manifest"] == {
        "file": "manifest_pre_f11.json",
        "sha256": hashlib.sha256(staged_bytes).hexdigest(),
    }
    assert s["man"].with_name("rotation_audits_pre_f11.jsonl").read_bytes() == audits_bytes
    merged_audits = s["man"].with_name("rotation_audits.jsonl").read_bytes()
    assert merged_audits.startswith(audits_bytes) and m["rotation_audits"]["sha256"] == _sha(
        s["man"].with_name("rotation_audits.jsonl")
    )
    assert m["rotation_audits"]["n_tensors"] == 40 + 2 * 12
    assert (
        sup["carried"]["rehash_before"]["all_equal"] and sup["carried"]["rehash_after"]["all_equal"]
    )
    assert sup["carried"]["entries_byte_identical"] is True
    # the supplement record
    assert sup["amendment"] == "F11" and sup["ids"] == SUPP_IDS
    assert sup["sha256"] == _sha(s["supp"]) and sup["selection_file"] == str(s["supp"])
    assert sup["cells"] == {"chain:0": ["c0_x2", "c0_x3", "c0_x4", "c0_x5"],
                            "chain:2": ["c2_x1", "c2_x2", "c2_x3", "c2_x4"],
                            "lora_sft:2": ["l2_x1", "l2_x2", "l2_x3", "l2_x4"]}  # fmt: skip
    assert sup["source_bytes_holders"] == {"c2_x4": "H2"}
    # merged counts over 52
    assert len(m["edit_ids"]) == 52 and m["edit_ids"] == sorted(m["edit_ids"])
    assert len(m["selections"]) == 52 and len(m["component_ids"]) == 52
    assert sorted(r["edit_id"] for r in m["selections"] if r.get("supplementary")) == SUPP_IDS
    assert sorted(i for i, e in m["edit_audits"].items() if e.get("supplementary")) == SUPP_IDS
    assert len(m["variants"]) == 156
    bad_rot = ri.variant_id(FAILS_F10, "rotation", 1.0, ri.rotation_seed(FAILS_F10))
    assert m["excluded_variant_ids"] == sorted(staged["excluded_variant_ids"] + [bad_rot])
    assert len(m["variant_ids"]) == len(staged["variant_ids"]) + 35
    assert m["edit_audits"][FAILS_F10]["pair_verdict"] == "excluded"
    assert m["edit_audits"][FAILS_F10]["exclusions"] == [
        {"tensor": None, "reason": "sub_resolution"}
    ]
    assert all(m["edit_audits"][i]["pair_verdict"] == "pass" for i in SUPP_IDS if i != FAILS_F10)
    c = m["coverage"]
    assert (
        c["n_edits"] == 52
        and c["n_rotation_pairs_staged"] == 48
        and c["n_rotation_pairs_excluded"] == 4
    )
    assert c["by_family"] == {"lora_sft": {"edits": 34, "pairs_excluded": 2},
                              "chain": {"edits": 18, "pairs_excluded": 2}}  # fmt: skip
    assert m["reference_ids_needed"]["originals"] == m["edit_ids"]
    assert m["selected_without_variants"] == [] and m["variants_without_selection"] == []
    assert set(SUPP_IDS) <= set(m["f10_subsets"]["pooled_norm_le_2e-2"]) | {FAILS_F10}
    assert m["rotation_seeds"] == staged["rotation_seeds"] | {
        i: ri.rotation_seed(i) for i in SUPP_IDS
    }
    assert (
        m["scored"] is False and m["schema"] == "a5-part-a-launch-v1" and m["audit_rule"] == "f10"
    )
    assert m["provenance"]["git_sha"] == "g11" and sup["pre_f11"]["code_sha"] == "staged-sha"
    # scoring plan and priced count recomputed over 52 (reference proofs computed)
    sc = m["scoring"]
    assert [v for v, _ in sc["plan"]] == m["variant_ids"] and sc["n_variants"] == len(
        m["variant_ids"]
    )
    assert sc["missing_originals"] == sorted(staged["scoring"]["missing_originals"] + ["c0_x3"])
    assert sc["n_D"] == 6 and sc["n_P"] == 0 and sc["unproven_originals"] == []
    assert sc["n_priced"] == sc["n_variants"] + 6 and sc["minutes"] == round(
        sc["n_priced"] * 1.04, 4
    )
    assert sc["excluded_variants"] == m["excluded_variant_ids"]
    proofs = {p["id"]: p for p in sc["reuse_proofs"]}
    assert proofs["c2_x4"]["verdict"] is True and proofs["c2_x4"]["bytes_holder"] == "H2"
    assert (
        proofs["c0_x3"]["verdict"] is False
        and proofs["base"] == staged["scoring"]["reuse_proofs"][0]
    )
    rp = sup["reference_proofs"]
    assert rp["computed"] is True
    assert rp["paths_touched"] == sorted(
        str(s["evals"] / i / n) for i in SUPP_IDS for n in ("eval.json", "records.jsonl")
    )
    # the new variants: staged rows, files, alias bytes from the holder
    rows = _staged_rows(stage)
    assert set(m["variant_ids"]) <= rows and bad_rot not in rows
    for vid, files in m["delta_sha256"].items():
        assert files["delta.safetensors"] == _sha(stage / vid / "delta.safetensors")
    alias_rot = ri.variant_id("c2_x4", "rotation", 1.0, ri.rotation_seed("c2_x4"))
    meta = json.loads((stage / alias_rot / "variant.json").read_text())
    assert meta["source_bytes_holder"] == "H2" and meta["source_id"] == "c2_x4"
    assert json.loads((stage / alias_rot / "edit.json").read_text())["id"] == "c2_x4"
    assert m["edit_audits"]["c2_x4"]["source_bytes_holder"] == "H2"
    # the slot mask over the 52 ids (schema a5-slot-mask-v1: schema and available only)
    mask_path = tmp_path / "slot_mask_f11.json"
    assert ri.main(["slot-mask", "--manifest", str(s["man"]), "--out", str(mask_path)]) == 0
    mask = json.loads(mask_path.read_text())
    assert set(mask) == {"schema", "available"} and mask["schema"] == "a5-slot-mask-v1"
    assert sorted(mask["available"]) == m["edit_ids"]
    no_rot = sorted(i for i, a in mask["available"].items() if "rotation" not in a)
    assert no_rot == sorted(["l2_00", "c0_0", "c2_0", FAILS_F10])
    assert all(a[-2:] == ["ladder_0.25", "ladder_0.5"] for a in mask["available"].values())


def test_supplement_refuses_when_a_staged_file_of_the_forty_differs(tmp_path, monkeypatch, capsys):
    s = _f11_setup(tmp_path, monkeypatch)
    staged_bytes = s["man"].read_bytes()
    listing = sorted(p.name for p in s["stage"].iterdir())
    ladder = next(v["variant_id"] for v in s["staged"]["variants"] if v["kind"] == "scale")
    f = s["stage"] / ladder / "delta.safetensors"
    good = f.read_bytes()
    f.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    assert ri.main(s["argv"]) == 2 and "differ" in capsys.readouterr().err
    assert s["man"].read_bytes() == staged_bytes
    assert not s["man"].with_name("manifest_pre_f11.json").exists()
    assert sorted(p.name for p in s["stage"].iterdir()) == listing  # nothing staged
    f.write_bytes(good)
    excluded_rot = s["staged"]["excluded_variant_ids"][0]
    vj = s["stage"] / excluded_rot / "variant.json"
    vj.write_text(vj.read_text() + " ")
    assert ri.main(s["argv"]) == 2 and "differ" in capsys.readouterr().err
    assert s["man"].read_bytes() == staged_bytes


def test_supplement_reads_no_evaluation_record_without_reference_proofs(tmp_path, monkeypatch):
    from wsp.eval import batch

    s = _f11_setup(tmp_path, monkeypatch)
    score = ri.score
    opened = _forbid_evaluation_access(monkeypatch, [s["evals"], batch.REPO / "data" / "evals"])
    assert ri.main(s["argv"] + ["--no-reference-proofs"]) == 0
    assert opened == []
    m = json.loads(s["man"].read_text())
    sc, rp = m["scoring"], m["supplement"]["reference_proofs"]
    assert rp["computed"] is False and rp["paths_touched"] == []
    assert sc["unproven_originals"] == SUPP_IDS
    assert sc["missing_originals"] == s["staged"]["scoring"]["missing_originals"]
    assert sc["reuse_proofs"] == s["staged"]["scoring"]["reuse_proofs"]  # carried, none added
    assert sc["n_priced"] == sc["n_variants"] + sc["n_P"] + sc["n_D"] + len(SUPP_IDS)
    with pytest.raises(SystemExit, match="unproven"):  # scoring waits for the proofs
        score(argparse.Namespace(stage=s["stage"], config="c", manifest=s["man"], dry_run=True))


def test_supplement_reference_proofs_touch_only_the_logged_paths(tmp_path, monkeypatch):
    import builtins
    import io
    import os

    s = _f11_setup(tmp_path, monkeypatch)
    root = s["evals"].resolve()
    seen = set()

    def wrap(f):
        def inner(path, *a, **k):
            if not isinstance(path, int):
                p = Path(os.fsdecode(path)).resolve()
                if root in p.parents:
                    seen.add(str(p))
            return f(path, *a, **k)

        return inner

    for mod, name in ((builtins, "open"), (io, "open"), (os, "open")):
        monkeypatch.setattr(mod, name, wrap(getattr(mod, name)))
    assert ri.main(s["argv"]) == 0
    m = json.loads(s["man"].read_text())
    logged = {str(Path(p).resolve()) for p in m["supplement"]["reference_proofs"]["paths_touched"]}
    assert seen and seen <= logged


def test_supplement_seeds_are_the_registered_rule_and_deterministic(tmp_path, monkeypatch):
    assert ri.rotation_seed("synthetic-test-vector") == 3336192590354858683  # control_seeds.json
    runs = []
    for k in range(2):
        s = _f11_setup(tmp_path / f"run{k}", monkeypatch)
        assert ri.main(s["argv"] + ["--no-reference-proofs"]) == 0
        m = json.loads(s["man"].read_text())
        new = {v["variant_id"] for v in m["variants"] if v["source_id"] in SUPP_IDS}
        runs.append((
            {i: m["rotation_seeds"][i] for i in SUPP_IDS},
            {vid: m["delta_sha256"].get(vid) for vid in sorted(new)},
            {i: m["edit_audits"][i]["seed"] for i in SUPP_IDS},
        ))  # fmt: skip
    assert runs[0] == runs[1]
    assert runs[0][0] == {i: ri.rotation_seed(i) for i in SUPP_IDS} == runs[0][2]
    for i in SUPP_IDS:
        seed = ri.rotation_seed(i)
        assert {ri.variant_id(i, k, st, seed) for k, st in (("rotation", 1.0), ("scale", 0.25),
                                                             ("scale", 0.5))} <= set(runs[0][1]) | {
            ri.variant_id(FAILS_F10, "rotation", 1.0, ri.rotation_seed(FAILS_F10))}  # fmt: skip


@pytest.mark.parametrize("change", ["reserve_instead_of_next", "not_root", "already_registered",
                                    "cell_not_depleted", "unequal_cells"])  # fmt: skip
def test_supplement_refuses_a_list_that_is_not_the_next_in_cell_roots(
    tmp_path, monkeypatch, capsys, change
):
    s = _f11_setup(tmp_path, monkeypatch)
    doc = json.loads(s["supp"].read_text())
    rows = {r["edit_id"]: r for r in doc["supplement"]["selected"]}
    swap = {"reserve_instead_of_next": ("l2_x4", "l2_x5", "lora_sft", 2),
            "not_root": ("c0_x5", "c0_x1", "chain", 0),
            "already_registered": ("l2_x4", "l2_09", "lora_sft", 2),
            "cell_not_depleted": ("l2_x4", "l1_x9", "lora_sft", 1)}.get(change)  # fmt: skip
    if swap:
        old, new, fam, t = swap
        rows.pop(old)
        rows[new] = _sel_row(new, fam, f"S-{new}", tertile=t)
    else:
        rows.pop("l2_x4")
    ids = sorted(rows)
    doc["supplement"] = {"edit_ids": ids, "selected": list(rows.values())}
    s["supp"].write_text(json.dumps(doc))
    staged_bytes = s["man"].read_bytes()
    assert ri.main(s["argv"]) == 2 and "supplement" in capsys.readouterr().err
    assert s["man"].read_bytes() == staged_bytes


def test_slot_mask_builder_reproduces_the_frozen_f10_mask():
    pkg = Path(__file__).resolve().parents[1] / "docs" / "prereg" / "a5_manifests" / "part_a"
    m = json.loads((pkg / "manifest.json").read_text())
    if "supplement" in m:
        pytest.skip("manifest already supplemented")
    text = ri.slot_mask_text(ri.slot_mask(m))
    assert hashlib.sha256(text.encode()).hexdigest() == _sha(pkg / "slot_mask_f10.json")


def test_supplement_refuses_a_malformed_list(tmp_path, monkeypatch, capsys):
    s = _f11_setup(tmp_path, monkeypatch)
    doc = json.loads(s["supp"].read_text())
    del doc["supplement"]["selected"][0]["component_id"]
    s["supp"].write_text(json.dumps(doc))
    assert ri.main(s["argv"]) == 2 and "malformed" in capsys.readouterr().err
