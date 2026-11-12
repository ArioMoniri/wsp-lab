"""F12f step 3: the pinned tokens-to-matrix builder (scripts/f12_matrix.py; chair ruling on F12e (b); F12.3; F5 pins
"Augmented comparators"; F12a.2(h)). Every input is fabricated over the committed population: no real token file,
pooled feature file, noise payload, outcome or anything under data/evals is opened."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import f12_matrix as fm
from scripts import f12_qualify as fq
from scripts import f12_simulation as fs
from tests.test_f12e_qualify import POP, POP_SHA, _fake_output

N_STATS, N_SPEC = 1242, 136


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _fake_gauge(tmp_path, tok, token_sha, mutate=None):
    from safetensors.torch import save_file

    out = tmp_path / "gauge"
    out.mkdir()
    Xg = np.array(tok["X"], copy=True)
    pres = tok["presence"].astype(bool)
    Xg[..., 0:16][pres] *= np.float32(1 + 1e-6)
    P = np.array(tok["presence"], copy=True)
    if mutate:
        mutate(Xg, P)
    save_file({"X_gauge": torch.from_numpy(Xg), "presence": torch.from_numpy(P)}, str(out / "gauge_tokens.safetensors"),
              metadata={"row_ids": json.dumps(tok["row_ids"])})  # fmt: skip
    for n in ("rows.jsonl", "detail.jsonl", "pricing.json"):
        (out / n).write_text("{}\n")
    files = {p.name: _sha(p.read_bytes()) for p in sorted(out.iterdir())}
    man = {"schema": "f12-gauge-tokens-v1", "status": "complete", "out_dir": str(out), "files": files,
           "inputs": {"token_manifest": {"sha256": token_sha}},
           "summary": {"n_rows_done": len(tok["row_ids"]), "n_tokens": int(P.sum()),
                       "k_reproduced_exact_rows": len(tok["row_ids"]),
                       "tokens_reproduced_bitwise": int(P.sum()), "masked_tokens": 0}}  # fmt: skip
    return man, _sha(json.dumps(man).encode())


def _fake_features(tmp_path, tok, mutate_meta=None):
    from safetensors.torch import save_file

    root = tmp_path / "features"
    names = [s[2] for s in fs.SLOTS]
    rng = np.random.default_rng(1)
    content = {}
    for i, rid in enumerate(tok["row_ids"]):
        d = root / rid
        d.mkdir(parents=True)
        present = [names[s] for s in np.flatnonzero(tok["presence"][i])]
        meta = {
            "base_commit": "c", "base_model": "m", "base_revision": "main", "config": {
                "buckets": 4, "delta": "effective_bf16", "feature_version": 4, "proj_dim": 64, "rank": 8,
                "seed": 0, "stats_k": 8},
            "config_hash": "h", "delta_key_mapping": None, "delta_relative_to": "base", "edit_id": rid,
            "eligibility": "production", "family": "x", "identity_hash": "i", "n_layers": 24, "norms": {},
            "params": {}, "parent_id": None, "rounding_stats": {}, "seed": 0, "stat_names": [],
            "step_semantics": None, "stored_delta_format": "dense",
            "tensors": [{"name": n} for n in present], "unchanged_tensors": [],
        }  # fmt: skip
        if mutate_meta:
            mutate_meta(rid, meta)
        (d / "features.json").write_text(json.dumps(meta))
        (d / "features.safetensors").write_bytes(b"per-tensor features")
        save_file({"stats": torch.from_numpy(rng.normal(size=N_STATS).astype(np.float32)),
                   "svd": torch.from_numpy(np.abs(rng.normal(size=N_SPEC)).astype(np.float32))},
                  str(d / "pooled.safetensors"))  # fmt: skip
        content[rid] = [_sha((d / n).read_bytes()) for n in fm.FEATURE_FILES]
    return root, content


@pytest.fixture()
def inputs(tmp_path):
    out, man = _fake_output(tmp_path)
    tok = fq.load_output(out)
    tsha = _sha(json.dumps(man).encode())
    gman, gsha = _fake_gauge(tmp_path, tok, tsha)
    froot, content = _fake_features(tmp_path, tok)
    return {"tmp": tmp_path, "out": out, "man": man, "tsha": tsha, "tok": tok, "gman": gman, "gsha": gsha,
            "froot": froot, "content": content}  # fmt: skip


def _build(x, **over):
    kw = {"token_manifest": x["man"], "token_manifest_sha256": x["tsha"], "gauge_manifest": x["gman"],
              "gauge_manifest_sha256": x["gsha"], "population": POP, "population_sha256": POP_SHA,
              "feature_content": x["content"], "features_dir": x["froot"]}  # fmt: skip
    kw.update(over)
    return fm.assemble(**kw)


# --------------------------------------------------------------------------------------------------- v4


def test_v4_vector_is_the_pinned_pooling_layout_stats_then_spectrum(tmp_path):
    from safetensors.torch import save_file

    st = np.arange(N_STATS, dtype=np.float32)
    sv = -np.arange(N_SPEC, dtype=np.float32)
    save_file(
        {"stats": torch.from_numpy(st), "svd": torch.from_numpy(sv)},
        str(tmp_path / "p.safetensors"),
    )
    v = fm.v4_vector(tmp_path / "p.safetensors")
    assert v.shape == (fs.V4_REAL_COLS,) and v.dtype == np.float64
    assert np.array_equal(v[:N_STATS], st) and np.array_equal(v[N_STATS:], sv)
    bad = st.copy()
    bad[7] = np.nan
    save_file(
        {"stats": torch.from_numpy(bad), "svd": torch.from_numpy(sv)},
        str(tmp_path / "q.safetensors"),
    )
    with pytest.raises(fm.SourceMismatch, match="non-finite"):
        fm.v4_vector(tmp_path / "q.safetensors")
    save_file(
        {"stats": torch.from_numpy(st[:-1]), "svd": torch.from_numpy(sv)},
        str(tmp_path / "r.safetensors"),
    )
    with pytest.raises(fm.SourceMismatch, match="width"):
        fm.v4_vector(tmp_path / "r.safetensors")


def test_x0_is_the_pooling_code_on_the_zero_delta(monkeypatch):
    from wsp.features import pool

    x0 = fm.x0_v4()
    s, p = pool.pool_features([], 24, 4)
    assert np.array_equal(x0, np.concatenate([s.numpy(), p.numpy()]).astype(np.float64))
    assert x0.shape == (fs.V4_REAL_COLS,) and not x0.any()
    monkeypatch.setattr(pool, "pool_features", lambda r, n, b: (torch.full((N_STATS,), float("nan")),
                                                                 torch.zeros(N_SPEC)))  # fmt: skip
    with pytest.raises(fm.SourceMismatch, match="non-finite"):
        fm.x0_v4()


# ----------------------------------------------------------------------------------------------- layout


def test_matrix_layout_in_population_and_a583_slot_order(inputs):
    m, facts = _build(inputs)
    tok = inputs["tok"]
    fs.validate_matrix(m)
    assert m["synthetic"] is False and m["schema"] == fs.TOKEN_SCHEMA
    assert list(m["row_ids"]) == tok["row_ids"] == sorted(r["id"] for r in POP["rows"])
    by_id = {r["id"]: r for r in POP["rows"]}
    assert list(m["component"]) == [by_id[i]["component_id"] for i in tok["row_ids"]]
    assert list(m["origin"]) == ["purchasable" if by_id[i]["source"] == "purchasable" else "existing"
                                 for i in tok["row_ids"]]  # fmt: skip
    assert m["eligible"].tolist() == tok["row_eligible"].astype(bool).tolist()
    assert m["sum_a"].tolist() == [tok["rows"][i]["a"] for i in tok["row_ids"]]
    assert m["sum_q"].tolist() == [tok["rows"][i]["q"] for i in tok["row_ids"]]
    # tokens: every present (row, slot) once, row-major then slot order; values are the frozen fp32 tokens
    rr, ss = np.nonzero(tok["presence"])
    assert m["token_row"].tolist() == rr.tolist() and m["token_slot"].tolist() == ss.tolist()
    assert np.array_equal(m["tokens"], tok["X"][rr, ss].astype(np.float64))
    assert m["token_role"].tolist() == [fs.SLOTS[s][1] for s in ss]
    assert np.array_equal(m["tokens_gauge"][:, 16:], m["tokens"][:, 16:])
    assert not np.array_equal(m["tokens_gauge"][:, :16], m["tokens"][:, :16])
    # v4 rows are the pooled files; x0 the zero-edit pooling
    v = fm.v4_vector(inputs["froot"] / tok["row_ids"][3] / "pooled.safetensors")
    assert np.array_equal(m["v4"][3], v) and not m["x0_v4"].any()
    # comparator width: v4 + 186 x (42 + presence), flattened slot-major
    F = fs.statnn_features(m)
    assert F.shape[1] == facts["widths"]["statnn_total"] == 1378 + 186 * 43 == 9376
    i, s = int(rr[0]), int(ss[0])
    off = fs.V4_REAL_COLS + s * 43
    assert np.array_equal(F[i, off : off + 42], tok["X"][i, s]) and F[i, off + 42] == 1.0
    assert facts["widths"] == {"v4": 1378, "v4_stats": 1242, "v4_spectrum": 136, "token_block": 7998,
                               "statnn_total": 9376, "token_columns": 42, "slots": 186}  # fmt: skip


# ------------------------------------------------------------------------------------- source identity


def test_pooled_file_differing_from_the_snapshot_fails_closed(inputs):
    rid = inputs["tok"]["row_ids"][5]
    content = copy.deepcopy(inputs["content"])
    content[rid][2] = "0" * 64
    with pytest.raises(fm.SourceMismatch, match="pooled.safetensors"):
        _build(inputs, feature_content=content)
    missing = copy.deepcopy(inputs["content"])
    del missing[rid]
    with pytest.raises(fm.SourceMismatch, match="snapshot"):
        _build(inputs, feature_content=missing)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda rid, m: m.update(edit_id="other"), "edit_id"),
        (lambda rid, m: m["config"].update(feature_version=3), "feature_version"),
        (lambda rid, m: m.update(n_layers=12), "n_layers"),
        (lambda rid, m: m.update(delta_relative_to="parent"), "delta_relative_to"),
        (lambda rid, m: m.update(wikitext_nll=1.0), "unexpected"),
        (
            lambda rid, m: (
                m["tensors"].append({"name": fs.SLOTS[0][2]})
                if m["tensors"][0]["name"] != fs.SLOTS[0][2]
                else m["tensors"].pop()
            ),
            "presence",
        ),
    ],
)
def test_feature_metadata_must_be_edit_derived_and_match_the_tokens(tmp_path, mutate, match):
    out, man = _fake_output(tmp_path)
    tok = fq.load_output(out)
    tsha = _sha(json.dumps(man).encode())
    gman, gsha = _fake_gauge(tmp_path, tok, tsha)
    target = tok["row_ids"][2]
    froot, content = _fake_features(
        tmp_path, tok, mutate_meta=lambda rid, m: mutate(rid, m) if rid == target else None
    )
    x = {"man": man, "tsha": tsha, "gman": gman, "gsha": gsha, "froot": froot, "content": content}
    with pytest.raises(fm.SourceMismatch, match=match):
        _build(x)


def test_gauge_output_must_match_the_frozen_tokens(inputs):
    tmp, tok = inputs["tmp"], inputs["tok"]
    bad_sha = dict(
        inputs["gman"],
        files=dict(inputs["gman"]["files"], **{"gauge_tokens.safetensors": "0" * 64}),
    )
    with pytest.raises(fm.SourceMismatch, match="gauge_tokens.safetensors"):
        _build(inputs, gauge_manifest=bad_sha)
    other = dict(inputs["gman"], inputs={"token_manifest": {"sha256": "1" * 64}})
    with pytest.raises(fm.SourceMismatch, match="token manifest"):
        _build(inputs, gauge_manifest=other)
    partial = dict(inputs["gman"], status="priced_only")
    with pytest.raises(fm.SourceMismatch, match="complete"):
        _build(inputs, gauge_manifest=partial)

    def tail(Xg, P):
        Xg[0, np.flatnonzero(P[0])[0], 20] += 1.0

    (tmp / "t1").mkdir()
    g2, s2 = _fake_gauge(tmp / "t1", tok, inputs["tsha"], mutate=tail)
    with pytest.raises(fm.SourceMismatch, match="16-41"):
        _build(inputs, gauge_manifest=g2, gauge_manifest_sha256=s2)

    def presence(Xg, P):
        s = np.flatnonzero(P[1] == 0)[0]
        P[1, s] = 1

    (tmp / "t2").mkdir()
    g3, s3 = _fake_gauge(tmp / "t2", tok, inputs["tsha"], mutate=presence)
    with pytest.raises(fm.SourceMismatch, match="presence"):
        _build(inputs, gauge_manifest=g3, gauge_manifest_sha256=s3)


def test_nonfinite_inputs_fail_closed(tmp_path):
    def nan_a(X, P, E, recs, details):
        recs[4]["a"] = None

    out, man = _fake_output(tmp_path, mutate=nan_a)
    tok = fq.load_output(out)
    tsha = _sha(json.dumps(man).encode())
    gman, gsha = _fake_gauge(tmp_path, tok, tsha)
    froot, content = _fake_features(tmp_path, tok)
    x = {"man": man, "tsha": tsha, "gman": gman, "gsha": gsha, "froot": froot, "content": content}
    with pytest.raises(fm.SourceMismatch, match="non-finite"):
        _build(x)


def test_token_files_are_verified_against_the_token_manifest(inputs):
    (inputs["out"] / "pricing.json").write_text('{"tampered": 1}')
    with pytest.raises(fq.TokenFileMismatch):
        _build(inputs)


# --------------------------------------------------------------------------------------------- outputs


def test_write_matrix_and_manifest_bind_the_identity(inputs, tmp_path):
    m, facts = _build(inputs)
    npz = tmp_path / "matrix.npz"
    man_path = tmp_path / "matrix_manifest.json"
    doc = fm.write_outputs(m, facts, npz, man_path, sources={"note": "test"})
    back = fs.load_matrix(npz)
    assert fs.matrix_sha256(back) == doc["matrix"]["matrix_sha256"] == fs.matrix_sha256(m)
    assert doc["matrix"]["file_sha256"] == fs.sha256_file(npz)
    assert doc["matrix"]["x0_v4_sha256"] == _sha(np.ascontiguousarray(m["x0_v4"]).tobytes())
    assert json.loads(man_path.read_text()) == doc
    assert (
        doc["matrix_sha256"] == doc["matrix"]["matrix_sha256"]
        and doc["file_sha256"] == doc["matrix"]["file_sha256"]
    )
    assert doc["widths"]["statnn_total"] == 9376 and doc["outcomes_read"].startswith("none")
