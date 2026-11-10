"""CPU checks for the F5.3 4B scale diagnostic: kernel, layer scaling, zoo entries, report arithmetic."""

import dataclasses
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import load_file

from wsp.schema import EditDescriptor

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("scale4b_build_edits")
report = _load("scale4b_report")
REV = "a" * 40
LAYERS = [11, 16, 20, 23, 25, 28]


def tiny_qwen(layers=32, hidden=8, ff=24, seed=0):
    """Qwen3.5-like key layout: full attention on every fourth layer, GatedDeltaNet elsewhere."""
    g = torch.Generator().manual_seed(seed)
    state = {}
    for i in range(layers):
        p = f"model.layers.{i}."
        mixer = "self_attn.o_proj" if i % 4 == 3 else "linear_attn.out_proj"
        state[p + mixer + ".weight"] = torch.randn(hidden, 16, generator=g).to(torch.bfloat16)
        state[p + "mlp.down_proj.weight"] = torch.randn(hidden, ff, generator=g).to(torch.bfloat16)
        state[p + "mlp.up_proj.weight"] = torch.randn(ff, hidden, generator=g).to(torch.bfloat16)
        state[p + "input_layernorm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
    state["model.embed_tokens.weight"] = torch.randn(32, hidden, generator=g).to(torch.bfloat16)
    return state


# ---- layer list and support -------------------------------------------------------------------


def test_layer_list_scales_24_to_32_and_matches_the_4b_zoo_config():
    assert build.scale_layers() == LAYERS
    cfg = yaml.safe_load((ROOT / "configs/zoo_qwen3.5-4b.yaml").read_text())
    src = yaml.safe_load((ROOT / "configs/zoo_qwen3.5-0.8b.yaml").read_text())
    assert cfg["families"]["random_dir"]["layers"] == LAYERS
    assert list(build.SOURCE_LAYERS) == src["families"]["random_dir"]["layers"]
    assert list(build.TENSOR_SPECS) == src["families"]["random_dir"]["tensors"]
    assert build.scale_layers([0, 23], 24, 32) == [0, 31]
    with pytest.raises(ValueError):
        build.scale_layers([0, 1], 24, 2)  # collapses onto one layer


def test_support_on_the_checkpoint_fixture_role_and_literal():
    keys = build.fixture_keys()
    role = build.support_keys(keys, LAYERS)
    assert len(role) == 12
    assert sum(k.endswith("mlp.down_proj.weight") for k in role) == 6
    assert {k for k in role if "self_attn.o_proj" in k} == {
        "model.layers.11.self_attn.o_proj.weight",
        "model.layers.23.self_attn.o_proj.weight",
    }
    assert sum("linear_attn.out_proj" in k for k in role) == 4
    literal = build.support_keys(keys, LAYERS, "literal")
    assert len(literal) == 8 and not any("linear_attn" in k for k in literal)


# ---- kernel -----------------------------------------------------------------------------------


class _StateModel:
    """Duck-typed model for ``wsp.eval.run.apply_delta`` (it only calls ``state_dict()``)."""

    def __init__(self, state):
        self.state = {k: v.clone() for k, v in state.items()}

    def state_dict(self):
        return self.state


def test_radius_is_per_tensor_and_realised_radius_is_what_the_evaluator_applies():
    from wsp.eval.run import apply_delta

    base = {k: v for k, v in tiny_qwen(hidden=64, ff=256).items() if "layers.11." in k}
    base = {k: v for k, v in base.items() if v.dim() == 2}
    for r in (1e-3, 5e-2):
        delta, stats = build.gaussian_delta(base, 20261110, r)
        assert set(delta) == set(base) and all(d.dtype == torch.bfloat16 for d in delta.values())
        model = _StateModel(base)
        apply_delta(model, delta)  # the TF-v1 evaluator's merge: fp32 add, cast to bf16
        sq_w = sq_r = sq_s = 0.0
        for k, w in base.items():
            per = stats["per_tensor"][k]
            wn = torch.linalg.vector_norm(w.double()).item()
            realised = model.state[k].float() - w.float()
            rn = torch.linalg.vector_norm(realised.double()).item()
            assert per["intended_norm"] == pytest.approx(r * wn, rel=1e-6)
            assert per["realised_radius"] == pytest.approx(rn / wn, rel=1e-12)
            assert per["changed_fraction"] == pytest.approx((realised != 0).double().mean().item())
            assert 0 <= per["applied_mismatch_fraction"] < 0.05
            sq_w += wn**2
            sq_r += rn**2
            sq_s += torch.linalg.vector_norm(delta[k].double()).item() ** 2
        assert stats["realised_radius"] == pytest.approx(math.sqrt(sq_r / sq_w), rel=1e-12)
        assert stats["delta_fro_norm"] == pytest.approx(math.sqrt(sq_s), rel=1e-12)
        lo, mid, hi = (stats[f"realised_radius{x}"] for x in ("_min", "", "_max"))
        assert lo <= mid <= hi
        if r == 5e-2:  # well above bf16 resolution: realised ~ intended
            assert stats["realised_radius"] == pytest.approx(r, rel=0.05)
            assert stats["changed_fraction"] > 0.9
        else:  # 1e-3 is near half a bf16 ulp: rounding leaves many elements unchanged
            assert stats["changed_fraction"] < 0.6
            assert 0.3 * r < stats["realised_radius"] < 1.5 * r


def test_seed_determinism_and_independence_of_namespace_and_support():
    base = {k: v for k, v in tiny_qwen().items() if k.endswith("down_proj.weight")}
    a, sa = build.gaussian_delta(base, 20261110)
    b, sb = build.gaussian_delta(base, 20261110)
    assert build.delta_bytes(a) == build.delta_bytes(b) and sa == sb
    c, _ = build.gaussian_delta(base, 20261111)
    assert all(not torch.equal(a[k], c[k]) for k in base)
    key = "model.layers.11.mlp.down_proj.weight"
    alone, _ = build.gaussian_delta({key: base[key]}, 20261110)
    assert torch.equal(alone[key], a[key])  # per-tensor seeds: support composition does not matter
    alias = key.replace("model.layers.", "model.language_model.layers.")
    assert build.tensor_seed(20261110, key) == build.tensor_seed(20261110, alias)
    with pytest.raises(ValueError):
        build.gaussian_delta(base, 1, 0.0)


# ---- zoo entries and manifest -----------------------------------------------------------------


def test_entry_id_matches_edit_descriptor(monkeypatch):
    params = build.entry_params(3, LAYERS, "role")
    monkeypatch.setattr(build, "FAMILY", "random_dir")  # a family the shared schema accepts
    want = EditDescriptor(
        family="random_dir",
        params=params,
        seed=20261113,
        base_model=build.BASE_MODEL,
        base_revision=REV,
    ).id
    assert build.entry_id(params, 20261113, REV) == want


def test_build_manifest_schema_entries_and_verify(tmp_path):
    state = tiny_qwen()
    out = tmp_path / "qwen3.5-4b-scale"
    header, rows = build.build(state, out, REV, n=6, sha="f" * 40, log=lambda *_: None)
    lines = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    assert lines[0] == header and lines[1:] == rows and len(rows) == 6
    for key in (
        "base_model",
        "base_revision",
        "layers",
        "support",
        "paired_ids",
        "config_hash",
        "radius",
    ):
        assert key in header
    assert header["layers"] == LAYERS and len(header["support"]) == 12 and header["radius"] == 1e-3
    assert header["seed"] == 20261110 and [r["seed"] for r in rows] == list(
        range(20261110, 20261116)
    )
    assert (
        header["paired_ids"] == build.paired_ids([r["id"] for r in rows])
        and len(header["paired_ids"]) == 5
    )
    assert sum(r["paired"] for r in rows) == 5
    fields = {f.name for f in dataclasses.fields(EditDescriptor)} | {"id"}
    for row in rows:
        assert {
            "id",
            "family",
            "seed",
            "eligibility",
            "reuse_key",
            "delta_sha256",
            "realised_radius",
        } <= set(row)
        assert row["eligibility"] == "production" and row["stored_delta_format"] == "dense"
        entry = out / row["id"]
        edit = json.loads((entry / "edit.json").read_text())
        assert set(edit) == fields and edit["id"] == row["id"]
        payload = (entry / "delta.safetensors").read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["delta_sha256"]
        delta = load_file(entry / "delta.safetensors")
        assert sorted(delta) == header["support"] == edit["touched_tensors"]
        assert edit["provenance"]["radius_stats"]["realised_radius"] == row["realised_radius"]
    # batch.py would select every entry (production rows of a header-pinned manifest)
    from wsp.eval.batch import select_entries

    assert [r["id"] for r in select_entries(lines, out)] == [r["id"] for r in rows]
    # identical rebuild is a no-op; a rebuild check in memory matches; another design is refused
    build.build(state, out, REV, n=6, sha="f" * 40, log=lambda *_: None)
    assert all(build.verify(state, out).values())
    with pytest.raises(FileExistsError):
        build.build(state, out, REV, n=6, support_mode="literal", log=lambda *_: None)


def test_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "zoo"
    assert build.main(["--dry-run", "--out", str(out)]) == 0
    assert not out.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["header"]["dry_run"] is True and len(plan["entries"]) == 20
    assert (
        plan["header"]["layers"] == LAYERS
        and plan["header"]["base_revision"] == build.DEFAULT_REVISION
    )


def test_wikitext_config_is_tf_v1_cut_to_its_wikitext_stage():
    full = yaml.safe_load((ROOT / "configs/eval_tf_v1.yaml").read_text())
    cut = yaml.safe_load((ROOT / "configs/eval_tf_v1_wikitext.yaml").read_text())
    assert cut["generation"] == full["generation"] and cut["tf_v1"] == full["tf_v1"]
    assert cut["benchmarks"] == {"wikitext_ppl": full["benchmarks"]["wikitext_ppl"]}


def test_shell_scripts_parse_and_cap():
    for name in ("lambda_job_scale4b.sh", "scale4b_h0_pair.sh"):
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)
    job = (ROOT / "scripts/lambda_job_scale4b.sh").read_text()
    assert 'CAP_H="${CAP_H:-3.0}"' in job and "instance-operations/terminate" in job
    assert "configs/eval_tf_v1_wikitext.yaml" in job and "--repeat r1" in job


# ---- report arithmetic ------------------------------------------------------------------------


def write_run(path, nll, run_id, device="H9", extra=None, delta_sha=None):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "records.jsonl").open("w") as f:
        for w, v in enumerate(nll):
            row = {
                "edit_id": path.name,
                "benchmark": "wikitext_ppl",
                "item_id": str(w),
                "correct": math.exp(v),
                "n_tokens": 1023,
                "hit_max_tokens": False,
                "extra": {"mean_nll": float(v), "token_ids_hash": f"t{w}"},
            }
            f.write(json.dumps(row) + "\n")
    meta = {
        "run_id": run_id,
        "base_run_id": "base-run",
        "eval_device_id": device,
        "config_hash": "c",
        "model_revision": REV,
        "source_hash": "s",
        "wall_seconds": 30.0,
        "stage_seconds": {"wikitext_ppl": 25.0},
        "model_load_seconds": 40.0,
        "gpu_class": "NVIDIA H100",
        "entry_hashes": {"delta.safetensors": delta_sha} if delta_sha else {},
        **(extra or {}),
    }
    (path / "eval.json").write_text(json.dumps(meta))


def make_device(root, base, shifts, digests, device, repeat_jitter=0.0, ids=None):
    write_run(root / "base", base, "base-run", device)
    write_run(root / "base/repeats/r1", base + repeat_jitter, "rep", device)
    for i, s in shifts.items():
        if ids is not None and i not in ids:
            continue
        write_run(root / i, base + s, "run", device, delta_sha=digests[i])
        write_run(root / i / "repeats/r1", base + s, "rep", device, delta_sha=digests[i])


def make_zoo(root, ids, paired):
    header = {
        "type": "header",
        "base_model": "Qwen/Qwen3.5-4B",
        "base_revision": REV,
        "kernel": "k",
        "radius": 1e-3,
        "layers": LAYERS,
        "target_depth": 32,
        "source_layers": [8, 12, 15, 17, 19, 21],
        "source_depth": 24,
        "tensors": ["mlp.down_proj", "self_attn.o_proj"],
        "support_mode": "role",
        "support": ["x"] * 12,
        "seed": 20261110,
        "n": len(ids),
        "config_hash": "d" * 64,
        "paired_ids": paired,
        "pairing_rule": "rule",
    }
    rows = [
        {
            "id": i,
            "seed": 20261110 + k,
            "delta_sha256": f"sha{i}",
            "realised_radius": 8e-4,
            "realised_radius_min": 7e-4,
            "realised_radius_max": 9e-4,
            "changed_fraction": 0.22,
        }
        for k, i in enumerate(ids)
    ]
    root.mkdir(parents=True)
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [header, *rows]))
    return {r["id"]: r["delta_sha256"] for r in rows}


def test_paired_effect_and_fraction_arithmetic():
    rng = np.random.default_rng(0)
    base = {"nll": 2.0 + rng.normal(0, 0.1, 64), "tokens": np.full(64, 1023.0)}
    shifted = {"nll": base["nll"] - 0.01, "tokens": base["tokens"]}
    e = report.paired_effect(shifted, base)
    assert e["delta"] == pytest.approx(-0.01, abs=1e-12)
    assert e["lo"] == pytest.approx(-0.01, abs=1e-12) and e["hi"] == pytest.approx(-0.01, abs=1e-12)
    assert e["improving"] and e["resolved"] == "improving"
    tokens = np.arange(1, 65, dtype=float)
    noisy = {"nll": base["nll"] + rng.normal(0, 0.05, 64), "tokens": tokens}
    weighted = report.paired_effect(noisy, {"nll": base["nll"], "tokens": tokens})
    want = float(np.sum(tokens * (noisy["nll"] - base["nll"])) / tokens.sum())
    assert weighted["delta"] == pytest.approx(want, rel=1e-12)
    assert (
        weighted["lo"] <= weighted["delta"] <= weighted["hi"]
        and weighted["resolved"] == "unresolved"
    )
    effects = {
        str(k): {"improving": k < 3, "resolved": "improving" if k == 0 else "unresolved"}
        for k in range(20)
    }
    fr = report.improvement_fraction(effects)
    assert (fr["improving"], fr["n"], fr["resolved_improving"]) == (3, 20, 1)
    assert fr["fraction"] == pytest.approx(0.15)
    lo, hi = report.clopper_pearson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 20), rel=1e-9)


def test_report_pairing_pending_pass_and_fail(tmp_path):
    ids = [f"e{k:02d}" for k in range(20)]
    paired = ids[:5]
    digests = make_zoo(tmp_path / "zoo", ids, paired)
    rng = np.random.default_rng(1)
    base = 2.0 + rng.normal(0, 0.1, 64)
    shifts = {i: (k - 6) * 1e-3 for k, i in enumerate(ids)}  # 6 improving, 1 zero, 13 worsening
    make_device(tmp_path / "h100", base, shifts, digests, "H9")

    pending = report.analyse(tmp_path / "zoo", tmp_path / "h100")
    assert pending["pairing"]["status"] == "pending" and not pending["counted"]
    assert pending["fraction"]["improving"] == 6 and pending["fraction"]["n"] == 20
    assert (
        pending["fraction"]["resolved_improving"] == 6
        and pending["fraction"]["resolved_worsening"] == 13
    )

    # H0 half: constant per-device offset (cancels in the paired delta), tiny entry-level noise
    make_device(
        tmp_path / "h0",
        base + 1e-4,
        {i: shifts[i] + 1e-6 * k for k, i in enumerate(paired)},
        digests,
        "H0",
        ids=paired,
    )
    ok = report.analyse(tmp_path / "zoo", tmp_path / "h100", tmp_path / "h0")
    assert ok["pairing"]["status"] == "pass", ok["pairing"]["reasons"]
    assert ok["counted"] and ok["pairing"]["cross_device"]["ratio"] < 0.1

    out = tmp_path / "doc.md"
    assert (
        report.main(
            [
                "--zoo",
                str(tmp_path / "zoo"),
                "--h100",
                str(tmp_path / "h100"),
                "--h0",
                str(tmp_path / "h0"),
                "--date",
                "2026-11-11",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    text = out.read_text()
    assert "diagnostic, not a gate; exposure recorded" in text and "**6 of 20**" in text
    assert "Status: **pass**" in text

    # a non-bitwise same-device repeat on H0 fails the pairing; records are then not counted
    make_device(
        tmp_path / "h0b",
        base,
        {i: shifts[i] for i in paired},
        digests,
        "H0",
        repeat_jitter=1e-7,
        ids=paired,
    )
    bad = report.analyse(tmp_path / "zoo", tmp_path / "h100", tmp_path / "h0b")
    assert bad["pairing"]["status"] == "fail" and not bad["counted"]
    assert any("H0 base: repeat not bitwise" in r for r in bad["pairing"]["reasons"])
    text = report.render(bad, "2026-11-11")
    assert "**not counted**" in text

    # a cross-device spread comparable to the between-entry spread fails section 6
    noisy = {i: shifts[i] + rng.normal(0, 5e-3) for i in paired}
    make_device(tmp_path / "h0c", base, noisy, digests, "H0", ids=paired)
    assert "cross-device SD ratio" in " ".join(
        report.analyse(tmp_path / "zoo", tmp_path / "h100", tmp_path / "h0c")["pairing"]["reasons"]
    )
