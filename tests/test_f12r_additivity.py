"""F12-R remedies in the F12-I construction code (scripts/f12_additivity_prepare.py, scripts/f12_additivity_stream.py;
unexposed adoption A10, A11, A21, A24, A25, A26, A30). Synthetic tensors and temporary directories; the only committed
file read is the F8a floor recompute (by value and hash). Nothing under data/ is opened."""

import json
import time

import pytest
import torch
from safetensors.torch import save_file

from scripts import a5_build_manifests as bm
from scripts import f12_additivity_prepare as fa
from scripts import f12_additivity_stream as fs
from scripts import f12_inputs_guard as fg
from wsp.features import response_field as rf

SHAPES = {
    "model.layers.3.mlp.down_proj.weight": (4, 6),
    "model.layers.11.self_attn.o_proj.weight": (4, 5),
    "model.layers.12.mlp.down_proj.weight": (4, 6),
    "model.layers.20.mlp.up_proj.weight": (6, 4),
}
STATE_HASH = "toy-state-hash"
FLOOR_DIR = fa.ROOT / "docs/prereg/f8_manifests/floor_payload"


def _base(shapes=SHAPES):
    g = torch.Generator().manual_seed(11)
    base = {n: torch.randn(*s, generator=g).to(torch.bfloat16) for n, s in shapes.items()}
    base["model.embed_tokens.weight"] = torch.randn(7, 4, generator=g).to(torch.bfloat16)
    return base


def _write_lora(d, seed, shapes=SHAPES, rank=2, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for n, (o, i) in shapes.items():
        module = n[: -len(".weight")]
        tensors[f"base_model.model.{module}.lora_A.weight"] = torch.randn(rank, i, generator=g)
        tensors[f"base_model.model.{module}.lora_B.weight"] = scale * torch.randn(
            o, rank, generator=g
        )
    (d / "lora").mkdir(parents=True)
    save_file(tensors, str(d / "lora" / "adapter_model.safetensors"))
    (d / "lora" / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": 4, "r": rank, "bias": "none"})
    )
    (d / "edit.json").write_text(json.dumps({"id": d.name}))


def install_governance(monkeypatch, tmp, pop, g2, base, extra_allowed=()):
    """Toy pins and boundary in place of the governed ones (the governed defaults are tested separately)."""
    monkeypatch.setattr(fa, "GOVERNED_POPULATION_SHA256", fg.sha256_bytes(pop.read_bytes()))
    monkeypatch.setattr(
        fa, "G2_MANIFEST_SHA256", fg.sha256_bytes((g2 / "g2_manifest.json").read_bytes())
    )
    ident = json.loads((g2 / "fields" / fa.OBJECTIVE / "manifest.json").read_text())["identity"]
    monkeypatch.setattr(fa, "WIKITEXT_IDENTITY_SHA256", rf._hash(ident))
    monkeypatch.setattr(
        fa,
        "read_boundary",
        lambda *extra: fg.ReadBoundary(allowed=[tmp, FLOOR_DIR, *extra_allowed, *extra]),
    )

    def guard(build, threads, cap_bytes, ruling):
        return fg.ResourceGuard(
            build, threads, cap_bytes, ruling,
            rss_fn=lambda: 10**6, mem_fn=lambda: 10**12, peak_fn=lambda: 10**6,
        )  # fmt: skip

    monkeypatch.setattr(fa, "resource_guard", guard)
    monkeypatch.setattr(fa, "hf_cache_dir", lambda: tmp / "hf")

    def load_base_state(expected):
        assert expected == STATE_HASH
        return {n: t.clone() for n, t in base.items()}, STATE_HASH

    monkeypatch.setattr(fa.pf, "load_base_state", load_base_state)
    monkeypatch.setattr(time, "time", lambda: 1000.0)


def make_g2(g2):
    ident = {"state_hash": STATE_HASH}
    fman = g2 / "fields" / fa.OBJECTIVE / "manifest.json"
    fman.parent.mkdir(parents=True)
    fman.write_text(json.dumps({"identity": ident}))
    man = {
        "fields": {
            fa.OBJECTIVE: {
                "status": "complete",
                "identity_sha256": rf._hash(ident),
                "files": {"manifest.json": fg.sha256_bytes(fman.read_bytes())},
            }
        }
    }
    (g2 / "g2_manifest.json").write_text(json.dumps(man))


@pytest.fixture
def toy(tmp_path, monkeypatch):
    zoo = tmp_path / "zoo"
    ids = ["aaaa000000000001", "aaaa000000000002"]
    rows = []
    for k, i in enumerate(ids):
        d = zoo / f"holder_{i}"
        _write_lora(d, seed=k + 1)
        rows.append(
            {
                "id": i,
                "family": "lora_sft",
                "source": "purchasable",
                "bytes_holder": f"holder_{i}",
                "resolved_delta_hash": bm.entry_delta_hash(bm.artefact_files(d)),
            }
        )
    pop = tmp_path / "population.json"
    pop.write_text(json.dumps({"rows": rows, "f12i_lora_roots": ids}, indent=1))
    g2 = tmp_path / "g2"
    make_g2(g2)
    base = _base()
    install_governance(monkeypatch, tmp_path, pop, g2, base)
    args = ["--population", str(pop), "--g2", str(g2), "--zoo", str(zoo), "--threads", "1"]
    return {
        "tmp": tmp_path,
        "ids": ids,
        "pop": pop,
        "args": args,
        "zoo": zoo,
        "base": base,
        "g2": g2,
    }


# --------------------------------------------------------------------------------------- A26 layers


def test_layer_outside_registered_range_rejected():
    ok = ["model.layers.0.mlp.up_proj.weight", "model.layers.23.mlp.up_proj.weight"]
    assert [lab for lab, _ in fa.layer_split_supports(ok)] == ["early", "late"]
    with pytest.raises(ValueError, match="outside the registered layers 0-23"):
        fa.layer_split_supports([*ok, "model.layers.24.mlp.up_proj.weight"])
    with pytest.raises(ValueError, match="outside the registered layers 0-23"):
        fa.layer_split_supports(["model.language_model.layers.31.mlp.up_proj.weight"])


def test_layer_split_requires_canonical_base_names():
    names = ["model.layers.3.mlp.up_proj.weight"]
    fa.layer_split_supports(names, canonical={"model.layers.3.mlp.up_proj.weight"})
    with pytest.raises(ValueError, match="not a canonical base tensor"):
        fa.layer_split_supports(
            ["model.language_model.layers.3.mlp.up_proj.weight"], canonical=set(names)
        )


def test_stream_records_layer_24_root_as_failure_not_late(toy, monkeypatch):
    """The streaming caller: a root carrying a layer-24 tensor fails closed and is recorded, never split as late."""
    shapes = dict(SHAPES) | {"model.layers.24.mlp.up_proj.weight": (6, 4)}
    base = _base(shapes)
    d = toy["tmp"] / "x" / "holder"
    _write_lora(d, seed=9, shapes=shapes)
    e = {"full_id": "bbbb000000000001", "kind": "lora_layer_split", "entry_dir": d}
    guard = fa.resource_guard("t", 1, fg.DEFAULT_CAP_BYTES, None)
    rec = fs.stream_root(e, base, guard)
    assert rec["bitwise_sum_verified"] is False and "parts" not in rec
    assert "outside the registered layers 0-23" in rec["failure"]


# ------------------------------------------------------------------------------------------ A25 C4


def test_c4_supports_require_exact_shape_and_boundary():
    good = {fa.C4_TENSOR: torch.zeros(1024, 3584)}
    sup = fa.c4_column_supports(good)
    assert sup == [
        ("cols_lo", {fa.C4_TENSOR: (0, 1792)}),
        ("cols_hi", {fa.C4_TENSOR: (1792, 3584)}),
    ]
    for bad in (
        {fa.C4_TENSOR: torch.zeros(1024, 3586)},
        {fa.C4_TENSOR: torch.zeros(3584, 1024)},
        {
            fa.C4_TENSOR: torch.zeros(1024, 3584),
            "model.layers.1.mlp.up_proj.weight": torch.zeros(2, 2),
        },
        {"model.layers.12.mlp.up_proj.weight": torch.zeros(1024, 3584)},
    ):
        with pytest.raises(ValueError, match="C4"):
            fa.c4_column_supports(bad)


def _c4_files(tmp, shape=(1024, 3584), base_hash=STATE_HASH, cid="c4toy"):
    delta = tmp / "c4" / "delta.safetensors"
    delta.parent.mkdir(parents=True, exist_ok=True)
    save_file({fa.C4_TENSOR: 1e-3 * torch.ones(*shape)}, str(delta))
    rec = {
        "schema": fa.C4_RECORD_SCHEMA,
        "entry_id": cid,
        "tensor": fa.C4_TENSOR,
        "shape": list(shape),
        "delta_path": str(delta),
        "delta_sha256": fg.sha256_bytes(delta.read_bytes()),
        "base_state_hash": base_hash,
        "construction_degenerate": False,
        "noop": False,
        "identity_verifier_passed": True,
    }
    rp = tmp / "c4" / "record.json"
    rp.write_text(json.dumps(rec))
    return delta, rp


def test_c4_refused_without_authorised_f8_c1_source(toy):
    delta, rp = _c4_files(toy["tmp"])
    with pytest.raises(ValueError, match="no authorised F8 C1 C4 source is pinned"):
        fa.main(["plan", *toy["args"], "--out", str(toy["tmp"] / "p.json"),
                 "--c4-delta", str(delta), "--c4-id", "c4toy", "--c4-record", str(rp)])  # fmt: skip


def test_c4_authorised_record_wrong_shape_refused_before_construction(toy, monkeypatch):
    delta, rp = _c4_files(toy["tmp"], shape=(1024, 3586))
    monkeypatch.setattr(fa, "C4_SOURCE_PIN", fg.sha256_bytes(rp.read_bytes()))
    built = []
    monkeypatch.setattr(fa, "construct", lambda *a: built.append(a))
    with pytest.raises(ValueError, match="C4"):
        fa.main(["plan", *toy["args"], "--out", str(toy["tmp"] / "p.json"),
                 "--c4-delta", str(delta), "--c4-id", "c4toy", "--c4-record", str(rp)])  # fmt: skip
    assert built == []


def test_c4_authorised_record_builds_column_split(toy, monkeypatch):
    toy["base"][fa.C4_TENSOR] = torch.zeros(1024, 3584).to(torch.bfloat16)
    monkeypatch.setattr(
        fa, "lora_entries", lambda pop, zoo: []
    )  # C4 only (its tensor is the toy roots' too)
    delta, rp = _c4_files(toy["tmp"])
    monkeypatch.setattr(fa, "C4_SOURCE_PIN", fg.sha256_bytes(rp.read_bytes()))
    out = toy["tmp"] / "p.json"
    fa.main(["plan", *toy["args"], "--out", str(out),
             "--c4-delta", str(delta), "--c4-id", "c4toy", "--c4-record", str(rp)])  # fmt: skip
    doc = json.loads(out.read_text())
    c4 = next(e for e in doc["entries"] if e["kind"] == "c4_column_split")
    assert [p["support"]["columns"][fa.C4_TENSOR] for p in c4["parts"]] == [[0, 1792], [1792, 3584]]
    assert c4["source"]["record_sha256"] == fa.C4_SOURCE_PIN
    # the record must name these delta bytes, this id and this base
    for field, value in (("entry_id", "other"), ("base_state_hash", "other-base")):
        rec = json.loads(rp.read_text()) | {field: value}
        rp.write_text(json.dumps(rec))
        monkeypatch.setattr(fa, "C4_SOURCE_PIN", fg.sha256_bytes(rp.read_bytes()))
        with pytest.raises(ValueError, match="C4 record"):
            fa.main(["plan", *toy["args"], "--out", str(toy["tmp"] / f"{field}.json"),
                     "--c4-delta", str(delta), "--c4-id", "c4toy", "--c4-record", str(rp)])  # fmt: skip


def test_stand_in_shape_checked(toy):
    raw = {fa.C4_TENSOR: torch.zeros(4, 6)}
    with pytest.raises(ValueError, match="C4"):
        fa.c4_column_supports(raw)


# ---------------------------------------------------------------------------------------- A24 Astra


def _astra(toy, qualified=True, **over):
    d = toy["zoo"] / "astra_entry"
    names = ["model.layers.3.mlp.down_proj.weight", "model.layers.20.mlp.up_proj.weight"]
    d.mkdir(parents=True, exist_ok=True)
    save_file({n: 1e-2 * torch.ones(*SHAPES[n]) for n in names}, str(d / "delta.safetensors"))
    (d / "edit.json").write_text("{}")
    rec = {
        "schema": fa.ASTRA_RECORD_SCHEMA,
        "entry_id": "astra0000000001",
        "entry_dir": str(d),
        "qualified": qualified,
        "tensors": names,
        "base_state_hash": STATE_HASH,
        "source_files": fa.holder_files(d),
        "authority": "toy F8a.4 qualification disposition",
    } | over
    rp = toy["tmp"] / "astra_record.json"
    rp.write_text(json.dumps(rec))
    return rp


def test_astra_refused_without_pinned_qualification(toy):
    rp = _astra(toy)
    with pytest.raises(ValueError, match="no authorised Astra qualification is pinned"):
        fa.main(
            ["plan", *toy["args"], "--out", str(toy["tmp"] / "p.json"), "--astra-record", str(rp)]
        )


def test_astra_qualified_pinned_builds_two_tensor_parts(toy, monkeypatch):
    rp = _astra(toy)
    monkeypatch.setattr(fa, "ASTRA_QUALIFICATION_PIN", fg.sha256_bytes(rp.read_bytes()))
    out = toy["tmp"] / "p.json"
    fa.main(["plan", *toy["args"], "--out", str(out), "--astra-record", str(rp)])
    doc = json.loads(out.read_text())
    a = next(e for e in doc["entries"] if e["kind"] == "astra_tensor_split")
    assert a["full_id"] == "astra0000000001" and len(a["parts"]) == 2
    assert a["source"]["record_sha256"] == fa.ASTRA_QUALIFICATION_PIN


def test_astra_failed_qualification_is_an_explicit_drop(toy, monkeypatch):
    rp = _astra(toy, qualified=False)
    monkeypatch.setattr(fa, "ASTRA_QUALIFICATION_PIN", fg.sha256_bytes(rp.read_bytes()))
    monkeypatch.setattr(fa, "load_raw", lambda *a: (_ for _ in ()).throw(AssertionError("loaded")))
    monkeypatch.setattr(
        fa, "lora_entries", lambda pop, zoo: []
    )  # isolate the Astra entry: no LoRA root is loaded either
    out = toy["tmp"] / "p.json"
    fa.main(["plan", *toy["args"], "--out", str(out), "--astra-record", str(rp)])
    doc = json.loads(out.read_text())
    a = next(e for e in doc["entries"] if e["kind"] == "astra_tensor_split")
    assert a["status"] == fa.STATUS_DROPPED_Q and "parts" not in a
    assert (
        a["replacement"] is None
        and doc["summary"]["awaiting"]["astra0000000001"] == fa.STATUS_DROPPED_Q
    )


@pytest.mark.parametrize(
    "over, match",
    [
        ({"score": 0.7}, "fields outside the qualification allowlist"),
        ({"tensors": ["model.layers.3.mlp.down_proj.weight"]}, "exactly two tensors"),
        ({"base_state_hash": "other"}, "base"),
        ({"source_files": {"delta.safetensors": "0" * 64}}, "source files"),
        ({"qualified": "yes"}, "qualified"),
    ],
)
def test_astra_record_must_bind_tensors_base_and_source(toy, monkeypatch, over, match):
    rp = _astra(toy, **over)
    monkeypatch.setattr(fa, "ASTRA_QUALIFICATION_PIN", fg.sha256_bytes(rp.read_bytes()))
    with pytest.raises(ValueError, match=match):
        fa.main(
            ["plan", *toy["args"], "--out", str(toy["tmp"] / "p.json"), "--astra-record", str(rp)]
        )


def test_astra_record_must_match_pin(toy, monkeypatch):
    rp = _astra(toy)
    monkeypatch.setattr(fa, "ASTRA_QUALIFICATION_PIN", "0" * 64)
    with pytest.raises(fg.AuthenticationError):
        fa.main(
            ["plan", *toy["args"], "--out", str(toy["tmp"] / "p.json"), "--astra-record", str(rp)]
        )


# ------------------------------------------------------------------------------------------ A30 floor


def test_floor_literal_bound_to_committed_recompute():
    rec = fa.floor_binding(fg.ReadBoundary(allowed=[FLOOR_DIR]))
    assert rec["h_star"] == fa.F8A_FLOOR_H_STAR == 0.00493967211805284
    assert rec["sha256"] == fa.F8A_FLOOR_SHA256
    assert rec["source"] == fa.F8A_FLOOR_SOURCE
    assert rec["payload_sha256"] == fa.F8A_FLOOR_PAYLOAD_SHA256


def test_floor_mismatch_refused(monkeypatch):
    monkeypatch.setattr(fa, "F8A_FLOOR_H_STAR", 0.0049397)
    with pytest.raises(ValueError, match="floor"):
        fa.floor_binding(fg.ReadBoundary(allowed=[FLOOR_DIR]))
    monkeypatch.undo()
    monkeypatch.setattr(fa, "F8A_FLOOR_SHA256", "0" * 64)
    with pytest.raises(fg.AuthenticationError):
        fa.floor_binding(fg.ReadBoundary(allowed=[FLOOR_DIR]))


def test_plan_doc_carries_floor_binding_and_intent(toy):
    out = toy["tmp"] / "s.json"
    fs.main([*toy["args"], "--out", str(out)])
    doc = json.loads(out.read_text())
    assert doc["floor_binding"]["sha256"] == fa.F8A_FLOOR_SHA256
    assert "outcomes_read" not in doc  # A11
    assert doc["outcomes_read_intent"]["statement"] == "none"
    assert "not access telemetry" in doc["outcomes_read_intent"]["label"]
    side = json.loads(out.with_name("s_run.json").read_text())
    assert (
        "outcomes_read" not in side
        and "not access telemetry" in side["outcomes_read_intent"]["label"]
    )


# ------------------------------------------------------------------------------- A10 inputs, boundary


def test_population_must_be_the_governed_one(toy, monkeypatch):
    monkeypatch.setattr(fa, "GOVERNED_POPULATION_SHA256", "0" * 64)
    with pytest.raises(fg.AuthenticationError):
        fs.main([*toy["args"], "--out", str(toy["tmp"] / "s.json")])
    with pytest.raises(fg.AuthenticationError):
        fa.main(["plan", *toy["args"], "--out", str(toy["tmp"] / "p.json")])


def test_holder_bytes_authenticated(toy):
    p = toy["zoo"] / f"holder_{toy['ids'][1]}" / "lora" / "adapter_config.json"
    p.write_text(json.dumps({"lora_alpha": 8, "r": 2, "bias": "none"}))
    with pytest.raises(fg.AuthenticationError, match="entry_delta_hash"):
        fs.main([*toy["args"], "--out", str(toy["tmp"] / "s.json")])
    with pytest.raises(fg.AuthenticationError, match="entry_delta_hash"):
        fa.main(["plan", *toy["args"], "--out", str(toy["tmp"] / "p.json")])


def test_holder_symlink_into_prohibited_refused(toy, tmp_path_factory):
    evals = tmp_path_factory.mktemp("elsewhere") / "evals"
    evals.mkdir()
    src = toy["zoo"] / f"holder_{toy['ids'][0]}" / "lora" / "adapter_model.safetensors"
    target = evals / "adapter_model.safetensors"
    target.write_bytes(src.read_bytes())
    src.unlink()
    src.symlink_to(target)
    with pytest.raises(fg.BoundaryError):
        fs.main([*toy["args"], "--out", str(toy["tmp"] / "s.json")])


def test_g2_manifest_authenticated_in_plan_mode(toy):
    (toy["g2"] / "g2_manifest.json").write_text("{}")
    with pytest.raises(fg.AuthenticationError):
        fs.main([*toy["args"], "--out", str(toy["tmp"] / "s.json")])


def test_governed_boundary_refuses_data_evals():
    b = fa.read_boundary()
    with pytest.raises(fg.BoundaryError, match="prohibited"):
        b.resolve(fa.ROOT / "data/evals/anything/eval.json")
    assert str(fa.DATA_ROOT / "zoo/qwen3.5-0.8b") in b.record()["allowed"]


def test_sidecar_records_exact_reads(toy):
    out = toy["tmp"] / "s.json"
    fs.main([*toy["args"], "--out", str(out)])
    side = json.loads(out.with_name("s_run.json").read_text())
    reads = side["read_boundary"]["reads"]
    assert any(r["path"] == str(toy["pop"]) and r["authenticated"] for r in reads)
    assert any(
        r["path"].endswith("floor_recompute_2026-11-11.json") and r["authenticated"] for r in reads
    )
    assert any(r["path"].endswith("adapter_model.safetensors") for r in reads)


# ------------------------------------------------------------------------------------- A21 resources


def test_stream_records_cap_and_peak(toy):
    out = toy["tmp"] / "s.json"
    fs.main([*toy["args"], "--out", str(out)])
    side = json.loads(out.with_name("s_run.json").read_text())
    res = side["resources"]
    assert res["cap_bytes"] == fg.DEFAULT_CAP_BYTES and res["within_cap"] is True
    assert (
        res["estimate_bytes"] == fs.STREAM_ESTIMATE_BYTES
        and side["peak_rss_bytes"] == res["peak_rss_bytes"]
    )


def test_cap_above_4gb_refused_without_ruling(toy):
    with pytest.raises(fg.ResourceRefused, match="chair ruling"):
        fs.main([*toy["args"], "--out", str(toy["tmp"] / "s.json"), "--rss-limit-gb", "4.5"])
    with pytest.raises(fg.ResourceRefused, match="chair ruling"):
        fa.main(
            [
                "plan",
                *toy["args"],
                "--out",
                str(toy["tmp"] / "p.json"),
                "--rss-cap-bytes",
                "5000000000",
            ]
        )


def test_prepare_refused_at_launch_when_materialisation_exceeds_cap(toy, monkeypatch):
    monkeypatch.setattr(fa, "materialisation_estimate", lambda *a: fg.DEFAULT_CAP_BYTES + 1)
    built = []
    monkeypatch.setattr(fa, "plan_constructions", lambda *a, **k: built.append(1))
    with pytest.raises(fg.ResourceRefused, match="above the cap"):
        fa.main(["plan", *toy["args"], "--out", str(toy["tmp"] / "p.json")])
    assert built == []


def test_materialisation_estimate_from_headers(toy):
    b = fg.ReadBoundary(allowed=[toy["tmp"]])
    pop = json.loads(toy["pop"].read_text())
    dense = sum(o * i * 4 for o, i in SHAPES.values())
    assert (
        fa.materialisation_estimate(b, pop, toy["zoo"]) == 2 * 2 * dense
    )  # 2 roots, raw + realised


def test_no_overwrite_of_existing_output(toy):
    out = toy["tmp"] / "s.json"
    out.write_text("governed pin")
    with pytest.raises(FileExistsError):
        fs.main([*toy["args"], "--out", str(out)])
    with pytest.raises(FileExistsError):
        fa.main(["plan", *toy["args"], "--out", str(out)])
    assert out.read_text() == "governed pin"


# ------------------------------------------------------------------------------ plan requalification


def test_requalify_compares_lora_entries_byte_for_byte(toy):
    a, b = toy["tmp"] / "a.json", toy["tmp"] / "b.json"
    fs.main([*toy["args"], "--out", str(a)])
    fs.main([*toy["args"], "--out", str(b)])
    rep = fs.requalify(json.loads(a.read_text()), json.loads(b.read_text()))
    assert rep["identical"] is True and rep["n_lora_entries"] == 2 and rep["differences"] == []
    ref = json.loads(b.read_text())
    ref["entries"][0]["parts"][0]["increment_sha256"] = "0" * 64
    rep = fs.requalify(json.loads(a.read_text()), ref)
    assert rep["identical"] is False and rep["differences"] == [toy["ids"][0]]


def test_prepare_and_stream_still_byte_identical(toy):
    a, b = toy["tmp"] / "plan.json", toy["tmp"] / "stream.json"
    fa.main(["plan", *toy["args"], "--out", str(a)])
    fs.main([*toy["args"], "--out", str(b)])
    assert a.read_bytes() == b.read_bytes()
