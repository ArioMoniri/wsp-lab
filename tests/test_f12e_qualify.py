"""F12e: real-token qualification and C1 token facts (Board A F12c review, ordered path steps 2-3).

Every token output here is fabricated over the committed population (schema-valid toy values). No test opens the real
F12 token directory, a noise payload, an outcome or anything under data/evals, so later sessions that run the suite are
not exposed to the real field forecasts."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import f12_c1 as c1
from scripts import f12_population as fpop
from scripts import f12_qualify as fq
from scripts import f12_simulation as fs
from scripts import f12_tokens as ft

POP = json.loads(fq.POPULATION.read_text())
POP_SHA = hashlib.sha256(fq.POPULATION.read_bytes()).hexdigest()
DOWN = [i for i, s in enumerate(fs.SLOTS) if fs.ROLES[s[1]] == "mlp.down_proj"]
UP = [i for i, s in enumerate(fs.SLOTS) if fs.ROLES[s[1]] == "mlp.up_proj"]


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _token(slot, rng):
    layer, role, _, full = fs.SLOTS[slot]
    t = np.zeros(42)
    t[0:16] = rng.normal(0, 1e-3, 16)
    sig = np.exp(rng.normal(-4, 0.5)) * 0.5 ** np.arange(8)
    t[16:24] = np.log(sig)
    t[24:32] = (sig > 1e-3 * sig[0]).astype(float)
    t[32] = 0.01
    t[33] = layer / 24
    t[34] = float(full)
    t[35], t[36], t[37], t[38] = rng.normal(0, 1e-3), abs(rng.normal(0, 1e-5)), 0.0, 1.0
    if fs.ROLES[role] == "mlp.down_proj":
        t[39], t[40], t[41] = 0.01, 0.02, 1.0
    return t


def _fake_output(tmp_path, *, ineligible=(), mutate=None):
    """A schema-valid fabricated token output directory and its manifest (same layout as f12_tokens)."""
    from safetensors.torch import save_file

    out = tmp_path / "tokens"
    out.mkdir()
    ids = sorted(r["id"] for r in POP["rows"])
    by_id = {r["id"]: r for r in POP["rows"]}
    rng = np.random.default_rng(0)
    X = np.zeros((len(ids), fs.N_SLOTS, 42), dtype=np.float32)
    P = np.zeros((len(ids), fs.N_SLOTS), dtype=np.uint8)
    E = np.zeros(len(ids), dtype=np.uint8)
    recs, details = [], []
    for i, rid in enumerate(ids):
        r = by_id[rid]
        slots = [DOWN[i % 24], UP[(i + 5) % 24]]
        toks, per = {}, {}
        for s in slots:
            t = _token(s, rng)
            X[i, s] = t.astype(np.float32)
            P[i, s] = 1
            toks[str(s)] = [float(x) for x in X[i, s]]
            name = fs.SLOTS[s][2]
            om = 0.2 if (r["family"] in ("abliteration", "random_dir") and i % 3) else 0.001
            per[name] = {"slot": s, "omitted_energy": om, "flags": [], "r_eff": 1, "sigma1": 1.0,
                         "numerical_rank": 1, "a": float(t[35]), "q": float(t[36])}  # fmt: skip
        elig = rid not in ineligible
        E[i] = int(elig)
        oms = [v["omitted_energy"] for v in per.values()]
        recs.append({
            "id": rid, "family": r["family"], "source": r["source"], "bytes_holder": r["bytes_holder"],
            "rank": 1, "rank_source": "pinned:x", "n_edited_tensors": 2, "n_tokens": 2,
            "unsupported_changed_tensors": [],
            "eligibility": "eligible" if elig else "excluded:no_k_frame:x",
            "a": float(sum(X[i, s, 35] for s in slots)), "a_fp64": None, "q": float(sum(X[i, s, 36] for s in slots)),
            "q_partial": False, "field_status": "ok", "increment_sha256": _sha(rid.encode()),
            "token_flags": {}, "omitted_energy_max": max(oms), "omitted_energy_median": float(np.median(oms)),
            "numerical_rank_max": 1, "n_probe_tokens": 1, "seconds": 1.0,
        })  # fmt: skip
        details.append({"id": rid, "tokens": toks, "per_tensor": per, "field_per_tensor": {}})
    if mutate:
        mutate(X, P, E, recs, details)
    save_file(
        {"X": torch.from_numpy(X), "presence": torch.from_numpy(P), "row_eligible": torch.from_numpy(E)},
        str(out / "tokens.safetensors"),
        metadata={"row_ids": json.dumps(ids), "columns": json.dumps(ft.COLUMNS)},
    )
    (out / "rows.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in recs))
    (out / "detail.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in details))
    (out / "aq.json").write_text(json.dumps({x["id"]: {"a": x["a"]} for x in recs}))
    frames = {s[2]: {"G": {"flags": []}, "A": {"flags": []}} for s in fs.SLOTS}
    (out / "kframes.json").write_text(json.dumps({"schema": "f12-kframes-v1", "tensors": frames, "counts": {}}))
    (out / "kframes.safetensors").write_bytes(b"frames")
    (out / "pricing.json").write_text("{}")
    files = {p.name: _sha(p.read_bytes()) for p in sorted(out.iterdir())}
    from wsp.features import response_field as rf

    man = {
        "schema": "f12-tokens-v1",
        "status": "complete",
        "out_dir": str(out),
        "files": files,
        "columns": ft.COLUMNS,
        "population_frozen": {
            "population_ids_sha256": POP["population_ids_sha256"],
            "population_json_sha256": POP_SHA,
            "fold_manifest_sha256": POP["folds"]["fold_manifest_sha256"],
        },
        "slots": {"n": fs.N_SLOTS, "names_sha256": _sha(json.dumps([s[2] for s in fs.SLOTS]).encode()),
                  "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(24)]},
        "tokens": {"shape": list(X.shape), "row_ids_sha256": _sha(json.dumps(ids).encode()),
                   "X_sha256": rf._tensor_hash(torch.from_numpy(X)),
                   "presence_sha256": rf._tensor_hash(torch.from_numpy(P)), "n_present_tokens": int(P.sum()),
                   "sha256": files["tokens.safetensors"]},
        "rows_sha256": _sha(ft.canonical_json(sorted(recs, key=lambda x: x["id"]))),
        "kframes": {"n_tensors": fs.N_SLOTS, "flag_counts": {}, "sha256": files["kframes.json"],
                    "kframes_safetensors_sha256": files["kframes.safetensors"]},
        "inputs": {"g2_manifest": {"sha256": "g2"}, "probe_cache": {"content_hash": "pc"},
                   "wikitext_field": {"identity_sha256": "wf"}},
        "base": {"state_hash": "base", "model": {"revision": "rev"}},
        "summary": {"n_rows_done": len(ids), "n_rows_population": len(ids), "eligible": int(E.sum()),
                    "excluded": {x["id"]: x["eligibility"] for x in recs if x["eligibility"] != "eligible"},
                    "rows_with_unsupported_changed_tensors": [],
                    "omitted_energy_gt_0.01_rows": sum(x["omitted_energy_max"] > 0.01 for x in recs)},
    }  # fmt: skip
    return out, man


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    out, man = _fake_output(tmp_path_factory.mktemp("good"))
    return out, man, fq.load_output(out)


def _checks(report):
    return {c["check"]: c for c in report["checks"]}


# ------------------------------------------------------------------------------------------ file verification


def test_verify_token_files_requires_every_manifest_hash(tmp_path):
    out, man = _fake_output(tmp_path)
    assert set(fq.verify_token_files(man, out)) == set(fq.REQUIRED_FILES)
    (out / "aq.json").write_text('{"tampered": 1}')
    with pytest.raises(fq.TokenFileMismatch, match="aq.json"):
        fq.verify_token_files(man, out)
    man2 = copy.deepcopy(man)
    del man2["files"]["rows.jsonl"]
    with pytest.raises(fq.TokenFileMismatch, match="rows.jsonl"):
        fq.verify_token_files(man2, out)


def test_load_output_refuses_unverified_files(tmp_path):
    out, man = _fake_output(tmp_path)
    (out / "pricing.json").write_text('{"x": 2}')
    with pytest.raises(fq.TokenFileMismatch):
        fq.qualify(man, POP, POP_SHA, out)


# ---------------------------------------------------------------------------------------------- qualification


def test_consistent_fabricated_output_passes_all_but_gauge_and_matrix(good):
    out, man, tok = good
    rep = fq.qualify(man, POP, POP_SHA, out, tok=tok)
    ch = _checks(rep)
    for name in ("token_files", "population_identity", "counts", "exclusions", "finite_schema", "folds",
                 "nesting"):  # fmt: skip
        assert ch[name]["pass"], (name, ch[name])
    # the frozen output carries no gauge-transformed tokens and the runner has no real-token gauge audit
    assert ch["gauge"]["pass"] is False
    assert ch["gauge"]["detail"]["prerequisites_pass"] is True
    assert ch["gauge"]["detail"]["audit_inputs_in_frozen_output"] is False
    # no v4 pooled statistics in the frozen output: the simulation matrix cannot be assembled as registered
    assert ch["simulation_matrix_inputs"]["pass"] is False
    assert rep["pass"] is False
    assert rep["first_failure"] == "gauge"
    assert rep["counts_expected"] == {"rows": 270, "components": 234}


def test_exclusion_mismatch_fails(tmp_path):
    rid = sorted(r["id"] for r in POP["rows"])[3]

    def mutate(X, P, E, recs, details):
        E[3] = 0  # flag says ineligible, rows.jsonl still says eligible

    out, man = _fake_output(tmp_path, mutate=mutate)
    rep = fq.qualify(man, POP, POP_SHA, out)
    ch = _checks(rep)
    assert ch["exclusions"]["pass"] is False
    assert rid in json.dumps(ch["exclusions"]["detail"])


def test_nonfinite_or_misplaced_columns_fail_the_schema(tmp_path):
    def nonfinite(X, P, E, recs, details):
        s = DOWN[0]
        X[0, s, 5] = np.nan

    out, man = _fake_output(tmp_path, mutate=nonfinite)
    assert _checks(fq.qualify(man, POP, POP_SHA, out))["finite_schema"]["pass"] is False

    def probe_off_down(X, P, E, recs, details):
        s = UP[5]
        X[0, s, 39:42] = [0.1, 0.2, 1.0]
        details[0]["tokens"][str(s)] = [float(x) for x in X[0, s]]

    out2, man2 = _fake_output(tmp_path / "b", mutate=probe_off_down)
    ch = _checks(fq.qualify(man2, POP, POP_SHA, out2))["finite_schema"]
    assert ch["pass"] is False and "probe" in json.dumps(ch["detail"])

    def absent_nonzero(X, P, E, recs, details):
        X[1, DOWN[20], 33] = 0.5  # an absent slot with a nonzero value

    out3, man3 = _fake_output(tmp_path / "c", mutate=absent_nonzero)
    assert _checks(fq.qualify(man3, POP, POP_SHA, out3))["finite_schema"]["pass"] is False


def test_population_identity_and_fold_tampering_fail(good):
    out, man, tok = good
    pop = copy.deepcopy(POP)
    k145 = pop["s_k"][-1]
    c = sorted(k145["folds"])[0]
    k145["folds"][c] = (k145["folds"][c] + 1) % 5
    ch = _checks(fq.qualify(man, pop, POP_SHA, out, tok=tok))
    assert ch["folds"]["pass"] is False
    man2 = copy.deepcopy(man)
    man2["population_frozen"]["population_json_sha256"] = "0" * 64
    assert _checks(fq.qualify(man2, POP, POP_SHA, out, tok=tok))["population_identity"]["pass"] is False


def test_gauge_prerequisites_need_a_k_frame_for_every_present_tensor(good):
    out, man, tok = good
    tok2 = dict(tok)
    tok2["kframes"] = copy.deepcopy(tok["kframes"])
    del tok2["kframes"]["tensors"][fs.SLOTS[DOWN[0]][2]]
    ch = _checks(fq.qualify(man, POP, POP_SHA, out, tok=tok2))["gauge"]
    assert ch["detail"]["prerequisites_pass"] is False


# --------------------------------------------------------------------------------------- folds and nesting


def test_eligible_row_folds_match_population_and_nest():
    elig = {r["id"] for r in POP["rows"]}
    folds = fq.eligible_row_folds(POP, elig)
    assert [f["k"] for f in folds] == list(fs.K_GRID)
    for f, s in zip(folds, POP["s_k"]):
        assert f["rows"] == s["rows"] and f["outer"] == s["folds"]
        assert fpop.fold_manifest_sha256(f["outer"]) == s["fold_manifest_sha256"]
        for o, inner in f["inner"].items():
            train = {c for c, g in f["outer"].items() if g != int(o)}
            assert set(inner) == train and sorted(set(inner.values())) == [0, 1, 2, 3]
        mixed = {m["component_id"] for m in POP["mixed_components"]}
        for fam, split in f["lofo"].items():
            assert not mixed & set(split["eval_components"])
            assert mixed <= set(split["train_components"]) or not mixed & set(f["outer"])


def test_an_excluded_row_leaves_its_s_k_without_top_up():
    rows = {r["id"]: r for r in POP["rows"]}
    single = next(r["id"] for r in POP["rows"] if r["source"] == "purchasable"
                  and sum(x["component_id"] == r["component_id"] for x in POP["rows"]) == 1)  # fmt: skip
    elig = set(rows) - {single}
    folds = fq.eligible_row_folds(POP, elig)
    k145 = folds[-1]
    assert k145["rows"] == 269 and rows[single]["component_id"] not in k145["outer"]
    assert k145["components"] == 233


def test_k_bins_follow_the_purchase_order():
    bins = fq.k_bins(POP)
    order = [p["component_id"] for p in POP["purchase_order"]]
    for r in POP["rows"]:
        if r["source"] != "purchasable":
            assert bins[r["id"]] == 0
        else:
            rank = order.index(r["component_id"])
            assert bins[r["id"]] == min(k for k in fs.K_GRID if k > rank)


def test_omitted_energy_summary_by_family_and_k_bin(good):
    out, man, tok = good
    s = fq.omitted_energy_summary(tok["rows"], POP)
    assert s["rows_gt_threshold"] == man["summary"]["omitted_energy_gt_0.01_rows"]
    assert sum(v["gt_threshold"] for v in s["by_family"].values()) == s["rows_gt_threshold"]
    assert sum(v["gt_threshold"] for v in s["by_k_bin"].values()) == s["rows_gt_threshold"]
    assert set(s["by_k_bin"]) == {str(k) for k in fs.K_GRID}
    assert s["by_family"]["lora_sft"]["gt_threshold"] == 0
    assert "diagnostic, not an input" in s["registered_status"]


# ------------------------------------------------------------------------------------------- facts for C1


def test_token_facts_bind_hashes_identities_folds_and_f12i(good):
    out, man, tok = good
    msha = _sha(json.dumps(man).encode())
    base = {"state_hash": "base", "tensors": {n: "h" + n for n in fq.base_tensor_names(tok, POP)}}
    facts = fq.token_facts(man, msha, POP, POP_SHA, tok, base_hashes=base)
    assert facts["token_manifest"]["sha256"] == msha
    assert facts["eligibility"]["n_eligible"] == 270 and facts["eligibility"]["excluded"] == {}
    assert facts["token_inputs"]["X_sha256"] == man["tokens"]["X_sha256"]
    assert facts["identities"]["g2_manifest"] == man["inputs"]["g2_manifest"]
    assert len(facts["down_proj_base_tensor_hashes"]) == 24
    assert set(facts["f12i_lora_parents"]) == set(POP["f12i_lora_roots"])
    for rid, p in facts["f12i_lora_parents"].items():
        assert p["increment_sha256"] == _sha(rid.encode())
        assert all(fq.layer_of(n) <= 11 for n in p["part1"]) and all(fq.layer_of(n) >= 12 for n in p["part2"])
        assert set(p["support_base_tensor_hashes"]) == set(p["part1"]) | set(p["part2"])
    folds = fq.eligible_row_folds(POP, set(facts["eligibility"]["eligible_ids"]))
    assert facts["folds_sha256"] == fq.folds_sha256(folds)


def test_base_tensor_hashes_check_the_state_hash():
    from wsp.features import response_field as rf

    state = {s[2]: torch.full((2, 3), float(i), dtype=torch.bfloat16) for i, s in enumerate(fs.SLOTS)}
    state["model.norm.weight"] = torch.ones(3, dtype=torch.bfloat16)
    good_hash = rf._hash({n: rf._tensor_hash(t) for n, t in state.items()})
    names = [fs.SLOTS[i][2] for i in DOWN]
    got = fq.base_tensor_hashes(state, names, good_hash)
    assert set(got["tensors"]) == set(names) and got["state_hash"] == good_hash
    with pytest.raises(ValueError, match="state hash"):
        fq.base_tensor_hashes(state, names, "0" * 64)


# ------------------------------------------------------------------------------------- identity freeze


def test_candidate_identities_cover_code_configs_payload_and_tokens():
    ident = fq.candidate_identities()
    for rel in ("scripts/f12_simulation.py", "scripts/f12_budget.py", "scripts/f12_c1.py",
                "scripts/f12_qualify.py", "scripts/tfv1_requalify.py"):  # fmt: skip
        assert ident["code"][rel] == _sha((fs.ROOT / rel).read_bytes())
    assert ident["simulation_code_identity"] == fs.current_code_identity()
    assert ident["configs"] == {k: v for k, v in fs.F12C_CONFIG_SHA256.items()}
    assert ident["restricted_payload"]["sha256"] == fq.RESTRICTED_SHA256
    assert ident["token_manifest"]["sha256"] == fq.TOKEN_MANIFEST_SHA256
    assert set(ident["versions"]) >= {"python", "numpy", "torch", "lightgbm", "scipy"}
    assert fq.identity_drift(ident) == []
    stale = copy.deepcopy(ident)
    stale["code"]["scripts/f12_simulation.py"] = "0" * 64
    assert fq.identity_drift(stale) == ["scripts/f12_simulation.py"]


# --------------------------------------------------------------------------------------------- C1 draft


def test_c1_draft_filled_from_committed_token_facts():
    files = c1.build_draft(fs.ROOT)
    assert list(files) == list(c1.C1_FILES)
    for name, obj in files.items():
        assert "draft" in obj["status"] and "not a C1 file" in obj["status"], name
    for name in ("population.json", "folds.json", "schemas.json"):
        assert c1.AWAITING not in json.dumps(files[name]), name
    facts = json.loads(fq.FACTS_PATH.read_text())
    assert files["population.json"]["eligibility"]["n_eligible"] == facts["eligibility"]["n_eligible"]
    assert files["folds.json"]["eligible_row_folds_sha256"] == facts["folds_sha256"]
    assert len(files["folds.json"]["eligible_row_folds"]) == len(fs.K_GRID)
    sch = files["schemas.json"]
    assert sch["token_input_hashes"]["X_sha256"] == facts["token_inputs"]["X_sha256"]
    assert len(sch["down_proj_tensor_hashes"]["tensors"]) == 24
    gauge = files["models.json"]["gauge_and_control_audits"]
    assert gauge["F5_gauge_audit"]["satisfied_by_frozen_output"] is False
    f12i = files["f12i_c1.json"]["parent_support_tensor_hashes"]
    assert set(f12i["lora"]) == set(POP["f12i_lora_roots"])
    assert c1.AWAITING in json.dumps(f12i["astra_and_c4"])  # not in the token output
    for name, obj in files.items():  # the committed draft equals the builder's output
        p = fs.ROOT / c1.DRAFT_DIR / name
        assert json.loads(p.read_text()) == json.loads(json.dumps(obj)), name
    assert c1.verify_c1(fs.ROOT, code_identity=fs.current_code_identity()) != []
