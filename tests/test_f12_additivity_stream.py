"""Streaming F12-I LoRA plan driver (scripts/f12_additivity_stream.py; chair instruction after F12g): byte-identical to
``f12_additivity_prepare.py plan`` on synthetic inputs, one root and one tensor at a time, failures and no-ops recorded,
cross-check against a reference entry, peak-RSS guard. CPU only; nothing under data/ is opened."""

import json
import time

import pytest
import torch
from safetensors.torch import save_file

from scripts import f12_additivity_prepare as fa
from scripts import f12_additivity_stream as fs

SHAPES = {
    "model.layers.3.mlp.down_proj.weight": (4, 6),
    "model.layers.11.self_attn.o_proj.weight": (4, 5),
    "model.layers.12.mlp.down_proj.weight": (4, 6),
    "model.layers.20.mlp.up_proj.weight": (6, 4),
}
STATE_HASH = "toy-state-hash"


def _base():
    g = torch.Generator().manual_seed(11)
    base = {n: torch.randn(*s, generator=g).to(torch.bfloat16) for n, s in SHAPES.items()}
    base["model.embed_tokens.weight"] = torch.randn(7, 4, generator=g).to(torch.bfloat16)
    return base


def _write_lora(d, seed, rank=2, zero_early=False, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for n, (o, i) in SHAPES.items():
        module = n[: -len(".weight")]
        A = torch.randn(rank, i, generator=g)
        B = scale * torch.randn(o, rank, generator=g)
        if zero_early and fa.layer_of(n) < fa.LAYER_BOUNDARY:
            B = torch.zeros_like(B)
        tensors[f"base_model.model.{module}.lora_A.weight"] = A.contiguous()
        tensors[f"base_model.model.{module}.lora_B.weight"] = B.contiguous()
    (d / "lora").mkdir(parents=True)
    save_file(tensors, str(d / "lora" / "adapter_model.safetensors"))
    (d / "lora" / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": 4, "r": rank, "bias": "none"})
    )
    (d / "edit.json").write_text(json.dumps({"id": d.name}))


@pytest.fixture
def toy(tmp_path, monkeypatch):
    zoo = tmp_path / "zoo"
    ids = ["aaaa000000000001", "aaaa000000000002", "aaaa000000000003"]
    for k, i in enumerate(ids):
        _write_lora(zoo / f"holder_{i}", seed=k + 1, zero_early=(k == 2))
    rows = [
        {"id": i, "family": "lora_sft", "source": "purchasable", "bytes_holder": f"holder_{i}"}
        for i in ids
    ]
    pop = tmp_path / "population.json"
    pop.write_text(json.dumps({"rows": rows, "f12i_lora_roots": ids}, indent=1))
    g2 = tmp_path / "g2"
    man = g2 / "fields" / fa.OBJECTIVE / "manifest.json"
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"identity": {"state_hash": STATE_HASH}}))
    base = _base()

    def load_base_state(expected):
        assert expected == STATE_HASH
        return {n: t.clone() for n, t in base.items()}, STATE_HASH

    monkeypatch.setattr(fa.pf, "load_base_state", load_base_state)
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    args = ["--population", str(pop), "--g2", str(g2), "--zoo", str(zoo), "--threads", "1"]
    return {"tmp": tmp_path, "ids": ids, "pop": pop, "args": args, "zoo": zoo}


def _run_both(toy, extra=()):
    a, b = toy["tmp"] / "plan.json", toy["tmp"] / "stream.json"
    fa.main(["plan", *toy["args"], "--out", str(a)])
    fs.main([*toy["args"], "--out", str(b), *extra])
    return a, b


def test_byte_identical_to_plan(toy):
    a, b = _run_both(toy)
    assert a.read_bytes() == b.read_bytes()
    doc = json.loads(b.read_text())
    lora = [e for e in doc["entries"] if e["kind"] == "lora_layer_split"]
    assert [e["full_id"] for e in lora] == toy["ids"]
    assert all(e["bitwise_sum_verified"] for e in lora)
    # the third root's early half rounds to nothing: a recorded no-op, I undefined, not substituted
    noop = lora[2]
    assert [p["noop"] for p in noop["parts"]] == [True, False] and noop["I_defined"] is False
    assert doc["summary"]["n_noop_parts"] == 1 and doc["summary"]["n_built"] == 3
    assert doc["population_json_sha256"] == fa.sha256_bytes(toy["pop"].read_bytes())


def test_sidecar_records_inputs_and_peak(toy):
    _, b = _run_both(toy)
    side = json.loads(b.with_name(b.stem + "_run.json").read_text())
    assert side["schema"] == fs.SIDECAR_SCHEMA
    assert side["plan_sha256"] == fa.sha256_bytes(b.read_bytes())
    assert side["population"]["sha256"] == fa.sha256_bytes(toy["pop"].read_bytes())
    assert side["driver"]["path"] == "scripts/f12_additivity_stream.py"
    assert side["driver"]["sha256"] == fa.pf.sha256_file(
        fs.ROOT / "scripts/f12_additivity_stream.py"
    )
    assert side["peak_rss_bytes"] > 0 and side["threads"] == 1
    assert side["roots_failed"] == [] and side["noop_parts"] == 1


def test_streams_one_tensor_at_a_time(toy, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("whole-root dense delta loaded")

    monkeypatch.setattr(fa, "load_raw", refuse)
    monkeypatch.setattr(fa, "plan_constructions", refuse)
    seen = []
    real = fa.construct

    def spy(W, raw, supports):
        seen.append(sorted(raw))
        return real(W, raw, supports)

    monkeypatch.setattr(fa, "construct", spy)
    fs.main([*toy["args"], "--out", str(toy["tmp"] / "s.json")])
    assert seen and all(len(names) == 1 for names in seen)
    assert len(seen) == 3 * len(SHAPES)


def test_bitwise_failure_recorded_not_substituted(toy, monkeypatch):
    real = fa.construct
    bad = "model.layers.20.mlp.up_proj.weight"
    calls = {"n": 0}

    def flaky(W, raw, supports):
        if bad in raw:
            calls["n"] += 1
            if calls["n"] == 2:  # the second root
                raise ValueError(
                    f"{bad}: realised parts do not sum bitwise to the full realised increment"
                )
        return real(W, raw, supports)

    monkeypatch.setattr(fa, "construct", flaky)
    out = toy["tmp"] / "s.json"
    fs.main([*toy["args"], "--out", str(out)])
    doc = json.loads(out.read_text())
    lora = [e for e in doc["entries"] if e["kind"] == "lora_layer_split"]
    assert [e["full_id"] for e in lora] == toy["ids"]
    failed = lora[1]
    assert failed["bitwise_sum_verified"] is False and "do not sum bitwise" in failed["failure"]
    assert failed["I_defined"] is False and "parts" not in failed
    assert lora[0]["bitwise_sum_verified"] and lora[2]["bitwise_sum_verified"]
    side = json.loads(out.with_name("s_run.json").read_text())
    assert side["roots_failed"] == [toy["ids"][1]]


def test_roots_subset_and_unknown_root(toy):
    out = toy["tmp"] / "one.json"
    fs.main([*toy["args"], "--out", str(out), "--roots", toy["ids"][1]])
    doc = json.loads(out.read_text())
    assert [e["full_id"] for e in doc["entries"] if "parts" in e] == [toy["ids"][1]]
    with pytest.raises(ValueError, match="not an F12-I LoRA root"):
        fs.main([*toy["args"], "--out", str(out), "--roots", "ffffffffffffffff"])


def test_dense_holder_refused(toy):
    holder = toy["zoo"] / f"holder_{toy['ids'][0]}"
    save_file(
        {"model.layers.3.mlp.down_proj.weight": torch.zeros(4, 6)},
        str(holder / "delta.safetensors"),
    )
    with pytest.raises(ValueError, match="not a LoRA holder"):
        fs.main([*toy["args"], "--out", str(toy["tmp"] / "x.json")])


def test_rss_guard(toy, monkeypatch):
    monkeypatch.setattr(fs, "peak_rss_bytes", lambda: 5 * 10**9)
    with pytest.raises(MemoryError, match="peak RSS"):
        fs.main([*toy["args"], "--out", str(toy["tmp"] / "x.json"), "--rss-limit-gb", "4.5"])


def test_cross_check(toy):
    _, b = _run_both(toy)
    doc = json.loads(b.read_text())
    ref = {"entries": [json.loads(json.dumps(e)) for e in doc["entries"]]}
    rid = toy["ids"][0]
    assert fs.cross_check(doc, ref, rid) == []
    e = next(x for x in ref["entries"] if x["full_id"] == rid)
    e["parts"][1]["increment_sha256"] = "0" * 64
    e["parts"][0]["part_id"] = "x"
    e["bitwise_sum_verified"] = False
    diffs = fs.cross_check(doc, ref, rid)
    assert len(diffs) == 3 and any("increment_sha256" in d for d in diffs)
    with pytest.raises(ValueError, match="absent"):
        fs.cross_check(doc, {"entries": []}, rid)
