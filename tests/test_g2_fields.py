"""CPU contracts of the G2 driver (scripts/g2_fields.py).

Tiny random Qwen3.5 (GatedDeltaNet + full attention, reference kernels), synthetic panels in the
panels.json shape, synthetic chain deltas under tmp_path; no network, no dataset or model download.
Nothing here is a G2 measurement.
"""

import copy
import dataclasses
import json
import math
import socket

import pytest
import torch

from scripts import g2_fields as g2
from wsp.eval import tf_v1
from wsp.features import response_field as rf


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise OSError("network disabled in G2 driver tests")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture(scope="module")
def toy_doc():
    return g2.toy_panels_doc()


@pytest.fixture(scope="module")
def toy_ctx(tmp_path_factory):
    return g2.build_toy_context(tmp_path_factory.mktemp("g2ctx"), limit=6)


def wikitext_only(ctx, n_pairs=3):
    """Pipeline-sized context: the WikiText objective (full-rank toy factors) and a few pairs."""
    return with_objectives(ctx, ("wikitext_nll",), n_pairs)


def with_objectives(ctx, names, n_pairs=20):
    keep = [o for o in ctx.objectives if o.spec.name in names]
    return dataclasses.replace(
        ctx, objectives=keep, pairs=ctx.pairs[:n_pairs], probe_items=ctx.probe_items[:2]
    )


TWO = ("wikitext_nll", "hellaswag_gold_char_log_likelihood")


def forced_audit(monkeypatch, errors):
    """Real audit rows, with the approximation replaced so that row i of objective o has relative
    error errors[o](i); everything else (identities, hashes, exact contraction) stays computed."""
    original, calls = g2.audit_one, []

    def patched(ctx, field, objective, pair, base, budget, impl):
        row = original(ctx, field, objective, pair, base, budget, impl)
        name = objective.spec.name
        i = sum(1 for c in calls if c == name)
        calls.append(name)
        error = errors[name](i)
        row["approximate"] = row["exact"] + error * row["denominator"]
        row["relative_error"] = g2.relative_error(row["exact"], row["approximate"])
        return row

    monkeypatch.setattr(g2, "audit_one", patched)
    return calls


def pin_field_walls(monkeypatch, walls):
    """Fixed field wall seconds (and the GPU seconds read from them) after the real fields stage, so
    the audit's ascending objective order does not depend on the toy machine's timings."""
    original = g2.run_fields

    def pinned(ctx, out, manifest, *args, **kwargs):
        result = original(ctx, out, manifest, *args, **kwargs)
        for name, seconds in walls.items():
            res = (manifest.get("fields") or {}).get(name, {}).get("resources")
            if res is not None:
                res |= {
                    "wall_seconds": seconds,
                    "gpu_seconds": seconds,
                    "gpu_hours": seconds / 3600,
                }
        return result

    monkeypatch.setattr(g2, "run_fields", pinned)


REGISTERED_ORDER_WALLS = {"wikitext_nll": 1.0, "hellaswag_gold_char_log_likelihood": 2.0}


def run_main(tmp_path, ctx, *extra):
    out = tmp_path / "out"
    code = g2.main(["--toy", "--out", str(out), *extra], context_builder=lambda o, limit: ctx)
    manifest_path = out / "g2_manifest.json"
    return code, out, json.loads(manifest_path.read_text()) if manifest_path.exists() else None


# ------------------------------------------------------------------------- panel verification


def test_registered_sizes_and_hashes_verify(toy_doc):
    doc, _ = toy_doc
    records = g2.verify_panels_doc(doc)
    sizes = {name: rec["n_field"] for name, rec in records.items()}
    assert sizes == {
        "wikitext_nll": 64,
        "mmlu_gold_log_prob": 64,
        "hellaswag_gold_char_log_likelihood": 64,
        "refusal_margin_harmful": 20,
        "refusal_margin_harmless": 64,
        "probe_response": 64,
    }
    mmlu = doc["panels"]["mmlu_gold_log_prob"]
    assert records["mmlu_gold_log_prob"]["field_ids"] == mmlu["item_ids"][:64]
    assert records["mmlu_gold_log_prob"]["ids_sorted_sha256"] == mmlu["item_ids_sorted_sha256"]


def _swap_id(doc):
    doc["panels"]["mmlu_gold_log_prob"]["item_ids"][3] = "999999"


def _bad_hash(doc):
    doc["panels"]["refusal_margin_harmless"]["item_ids_sorted_sha256"] = "0" * 64


def _window_hash(doc):
    doc["panels"]["wikitext_nll"]["windows"][5]["token_ids_hash"] = "0" * 64


def _dev_overlap(doc):
    cert = doc["panels"]["wikitext_nll"]["certificate"]["computed_checks"]
    cert["wikitext_development_windows"]["dev_touched_articles"] = [
        doc["panels"]["wikitext_nll"]["document_ids"][0]
    ]


def _harmful_64(doc):
    doc["panels"]["refusal_margin_harmful"]["registered_size"] = 64


def _prefix(doc):
    doc["panels"]["hellaswag_gold_char_log_likelihood"]["field_prefix"] = 32


def _not_disjoint(doc):
    doc["panels"]["probe_response"]["certificate"]["computed_disjoint"] = False


def _short(doc):
    doc["panels"]["refusal_margin_harmful"]["short_by"] = 1


def _schema(doc):
    doc["schema"] = "a5-panels-v1"


@pytest.mark.parametrize(
    "mutate",
    [
        _swap_id,
        _bad_hash,
        _window_hash,
        _dev_overlap,
        _harmful_64,
        _prefix,
        _not_disjoint,
        _short,
        _schema,
    ],
)
def test_panel_hash_verification_refuses_mismatch(toy_doc, mutate):
    doc = copy.deepcopy(toy_doc[0])
    mutate(doc)
    with pytest.raises(g2.PanelMismatch):
        g2.verify_panels_doc(doc)


def test_panels_file_sha256_is_checked(tmp_path, toy_doc):
    path = tmp_path / "panels.json"
    path.write_text(json.dumps(toy_doc[0]))
    actual = g2.sha256_file(path)
    assert g2.load_panels_doc(path, actual)[1] == actual
    with pytest.raises(g2.PanelMismatch):
        g2.load_panels_doc(path, "0" * 64)


def test_wikitext_windows_are_retokenised_and_checked(toy_doc):
    from scripts.a5_build_panels import wikitext_articles

    doc, lines = toy_doc
    panel = doc["panels"]["wikitext_nll"]
    tok = g2.ToyTokenizer()
    windows = g2.wikitext_windows(lines, tok.encode, panel, wikitext_articles)
    assert len(windows) == 64 and all(len(w) == 16 for w in windows)
    edited = list(lines)
    edited[3] = edited[3].replace("one", "uno")
    with pytest.raises(g2.PanelMismatch):
        g2.wikitext_windows(edited, tok.encode, panel, wikitext_articles)


def test_real_run_refuses_wrong_panels_hash_before_any_work(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("model or datasets must not be loaded on a panel mismatch")

    monkeypatch.setattr(g2, "build_real_context", boom)
    out = tmp_path / "g2"
    argv = ["--out", str(out), "--panels-sha256", "0" * 64, "--allow-code-drift"]
    assert g2.main([*argv, "--stages", "fields"]) == g2.EXIT["refused"]
    assert not (out / "g2_manifest.json").exists()


def test_cli_guards():
    with pytest.raises(SystemExit):
        g2.parse_args(["--out", "/tmp/x"])  # real run without the A5.1 panels hash
    with pytest.raises(SystemExit):
        g2.parse_args(["--panels-sha256", "a" * 64, "--limit", "3"])  # no coverage cut
    with pytest.raises(SystemExit):
        g2.parse_args(["--toy", "--out", str(g2.ROOT / "data/runs/g2_toy")])  # toy never in data/
    real = ["--panels-sha256", "a" * 64]
    assert g2.parse_args(real).max_chain_exact_gpu_hours == g2.CHAIN_EXACT_CAP_HOURS == 2.2
    with pytest.raises(SystemExit):
        g2.parse_args([*real, "--max-chain-exact-gpu-hours", "2.3"])  # never above the F6.2 cap
    with pytest.raises(SystemExit):
        g2.parse_args([*real, "--stages", "chain_exact,bogus"])
    assert g2.parse_args([*real, "--stages", "chain_exact"]).stages == ("chain_exact",)
    assert g2.parse_args(["--toy", "--out", "/tmp/x"]).stages == g2.STAGES  # default unchanged


# --------------------------------------------------------------------- scorer conventions


class Runner:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer, self.edit_id = model, tokenizer, "toy"


def test_objectives_match_tf_v1_scorer_conventions(toy_ctx):
    model, tok = toy_ctx.model, toy_ctx.tokenizer
    runner = Runner(model, tok)

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": text}])

    row = g2.toy_rows("mmlu_gold_log_prob", "5")
    item, _ = g2.mmlu_item(row, tok)
    choices = [chr(65 + i) for i in range(len(row["options"]))]
    context = (
        row["question"]
        + "\n"
        + "\n".join(f"{c}. {o}" for c, o in zip(choices, row["options"]))
        + "\nAnswer:"
    )
    scores, _ = tf_v1.letter_scores(runner, context, choices)
    reference = tf_v1.choice_record("t", "mmlu_pro", "5", scores, row["answer_index"])
    got = g2.score_mmlu(model, tok, item, "cpu")
    assert got.score_sum.item() == pytest.approx(reference.extra["gold_log_prob"], abs=1e-5)
    assert got.weight == 1.0 and got.valid_mask.sum() == 1

    row = g2.toy_rows("hellaswag_gold_char_log_likelihood", "6")
    item, _ = g2.hellaswag_item(row, tok)
    scores, _ = tf_v1.hellaswag_scores(runner, row["ctx"], row["endings"])
    got = g2.score_hellaswag(model, tok, item, "cpu")
    assert got.score_sum.item() == pytest.approx(scores[row["label"]], abs=1e-5)

    text = g2.toy_rows("refusal_margin_harmful", "train:3")
    item, _ = g2.refusal_item(text, tok, chat, g2.TOY_SETS)
    sets = tf_v1.build_continuations(tok, g2.TOY_SETS)
    prefix = tok.encode(chat(text), add_special_tokens=False)
    phrase_scores, _ = tf_v1.batched_continuation_scores(
        runner, prefix, sets["refusal"] | sets["compliance"]
    )
    margin = tf_v1.phrase_mass_margin(
        [phrase_scores[p] for p in sets["refusal"]], [phrase_scores[p] for p in sets["compliance"]]
    )
    got = g2.score_refusal(model, tok, item, "cpu")
    assert got.score_sum.item() == pytest.approx(margin, abs=1e-5)
    # shared prompt position counted once in the Fisher mask
    assert got.valid_mask[:, item["start"]].sum() == 1

    ids = next(iter(toy_ctx.objectives[0].panel.items()))["input_ids"].clone()
    loss = model(ids[None], labels=ids[None]).loss.item()
    got = g2.score_wikitext(model, tok, {"input_ids": ids}, "cpu")
    assert got.weight == len(ids) - 1
    assert got.score_sum.item() / got.weight == pytest.approx(loss, abs=1e-5)


# ------------------------------------------------------------------- coverage and slots


def test_coverage_is_every_registered_matrix_slot_with_typed_roles(toy_ctx):
    roles = toy_ctx.supported
    linears = [
        n
        for n, m in toy_ctx.model.named_modules()
        if isinstance(m, torch.nn.Linear) and ".layers." in n
    ]
    assert sorted(roles) == sorted(linears)
    assert roles["model.layers.3.self_attn.q_proj"] == "full_attention:self_attn.q_proj"
    assert (
        roles["model.layers.0.linear_attn.in_proj_qkv"]
        == "linear_attention:linear_attn.in_proj_qkv"
    )
    assert roles["model.layers.0.mlp.down_proj"] == "mlp:mlp.down_proj"
    assert toy_ctx.writers[0] == "model.layers.0.mlp.down_proj"


def test_slot_without_linear_is_missing_hook(toy_ctx):
    slots = copy.deepcopy(toy_ctx.slots)
    matrix = next(s for s in slots if s["matrix"])
    matrix["module"] = matrix["module"].rsplit(".", 1)[0]  # the parent block, not a Linear
    with pytest.raises(rf.MissingHookError):
        g2.supported_modules(toy_ctx.model, slots)
    first = next(s for s in toy_ctx.slots if s["matrix"])
    with pytest.raises(rf.MissingHookError, match="outside the slot registry"):
        g2.supported_modules(toy_ctx.model, [s for s in toy_ctx.slots if s is not first])


# --------------------------------------------------------------------------- audit rule


def test_relative_error_uses_the_nats_floor():
    assert g2.relative_error(2.0, 2.2) == pytest.approx(0.1)
    assert g2.relative_error(1e-4, 1.5e-4) == pytest.approx(0.05)  # 5e-5 / 1e-3
    assert g2.relative_error(0.0, 1e-4) == pytest.approx(0.1)


def _rows(errors):
    return [{"exact": 1.0, "approximate": 1.0 + e, "relative_error": e} for e in errors]


def test_audit_rule_is_16_of_20_with_inclusive_threshold():
    assert g2.audit_verdict(_rows([0.1] * 16 + [0.5] * 4))["passed"]
    verdict = g2.audit_verdict(_rows([0.05] * 15 + [0.1000001] * 5))
    assert not verdict["passed"] and verdict["n_passed"] == 15
    nan = g2.audit_verdict(_rows([0.0] * 15) + _rows([float("nan")] * 5))
    assert not nan["passed"]
    errored = _rows([0.0] * 16) + [{"exact": math.nan, "approximate": math.nan, "error": "x"}] * 4
    assert g2.audit_verdict(errored)["passed"]  # 16 still pass; errors count as failures
    short = g2.audit_verdict(_rows([0.0] * 15))
    assert short["short"] and short["n_audited"] == 15 and not short["passed"]


def test_short_audit_never_passes_even_with_16_passes():
    # the registered twenty-row floor: 16 or 19 passing rows of a short audit are still a hard stop
    for n in (16, 19):
        verdict = g2.audit_verdict(_rows([0.0] * n), "wikitext_nll")
        assert verdict["n_passed"] == n >= g2.AUDIT_MIN_PASS
        assert verdict["short"] and not verdict["passed"]
    assert not g2.audit_verdict([], "wikitext_nll")["passed"]
    audit = {
        "status": "complete",
        "objectives": {
            "wikitext_nll": {
                "status": "complete",
                "verdict": g2.audit_verdict(_rows([0.0] * 19), "wikitext_nll"),
            }
        },
        "verdict": {
            "passed": True
        },  # an inconsistent overall flag does not rescue a short objective
    }
    stop = g2.audit_hard_stop(audit)
    assert stop["HARD_STOP"] and stop["short_objectives"] == ["wikitext_nll"]


def test_every_objective_needs_its_own_pass():
    required = list(g2.AUDIT_OBJECTIVES)
    assert len(required) == 5 and required == list(rf.OBJECTIVES)
    good = {o: g2.audit_verdict(_rows([0.0] * 20), o) for o in required}
    assert g2.overall_verdict(good, required)["passed"]
    for bad in required:  # no single objective's pass (e.g. WikiText) stands for another
        verdicts = dict(good) | {bad: g2.audit_verdict(_rows([0.0] * 15 + [1.0] * 5), bad)}
        overall = g2.overall_verdict(verdicts, required)
        assert not overall["passed"] and overall["failed_objectives"] == [bad]
    missing = {o: v for o, v in good.items() if o != "refusal_margin_harmful"}
    overall = g2.overall_verdict(missing, required)
    assert not overall["passed"] and overall["missing_objectives"] == ["refusal_margin_harmful"]


def test_real_run_requires_all_five_audit_objectives(toy_ctx):
    real = dataclasses.replace(toy_ctx, toy=False)
    assert g2.required_audit_objectives(real) == list(rf.OBJECTIVES)
    with pytest.raises(ValueError, match="audit objectives missing"):
        g2.required_audit_objectives(dataclasses.replace(wikitext_only(toy_ctx), toy=False))
    assert g2.required_audit_objectives(wikitext_only(toy_ctx)) == ["wikitext_nll"]


def test_cost_estimate_is_five_objectives_by_twenty_predecessors():
    fields = {o: {"resources": {"wall_seconds": 36.0}} for o in rf.OBJECTIVES}
    est = g2.audit_cost_estimate(fields, list(rf.OBJECTIVES), 1.0, 0.0)
    assert est["n_objectives"] == 5 and est["n_predecessors"] == 20 and est["n_rows"] == 100
    # per row 0.5 x 36 s = 18 s; 100 rows = 0.5 GPU-h <= the 1 GPU-h cap
    assert est["estimated_gpu_hours"] == pytest.approx(0.5) and est["within_cap"]
    assert not g2.audit_cost_estimate(fields, list(rf.OBJECTIVES), 1.0, 0.6)["within_cap"]
    assert not g2.audit_cost_estimate(fields, list(rf.OBJECTIVES), 1.0, 0.0, ratio=2.5)[
        "within_cap"
    ]


def test_audit_pairs_walk_lineages_sort_and_resolve_alias_bytes():
    rows = [
        {"id": "r1", "parent_id": None, "bytes_present": True, "bytes_via_alias": None},
        {"id": "a2", "parent_id": "r1", "bytes_present": False, "bytes_via_alias": "h2"},
        {"id": "p3", "parent_id": "a2", "bytes_present": True, "bytes_via_alias": None},
        {"id": "s4", "parent_id": "p3", "bytes_present": True, "bytes_via_alias": None},
        {"id": "root", "parent_id": None, "bytes_present": True, "bytes_via_alias": None},
    ]
    selection = {
        "registered": {
            "selected": [
                {"edit_id": "s4", "family": "chain", "predecessor_id": "p3"},
                {"edit_id": "root", "family": "chain", "predecessor_id": "base"},
                {"edit_id": "L", "family": "lora_sft", "predecessor_id": "base"},
            ]
        }
    }
    pairs, meta = g2.audit_pairs(selection, rows)
    assert [(p["predecessor_id"], p["successor_id"]) for p in pairs] == [
        ("a2", "p3"),
        ("p3", "s4"),
        ("r1", "a2"),
    ]
    assert pairs[0]["predecessor_bytes"] == "h2" and pairs[2]["successor_bytes"] == "h2"
    assert meta["n_candidates"] == 3 and meta["n_used"] == 3 and meta["short"]
    rows[1]["bytes_via_alias"] = None
    with pytest.raises(ValueError, match="bytes"):
        g2.audit_pairs(selection, rows)


def test_audit_pairs_take_first_20_by_sorted_id():
    rows, selected = [], []
    for c in range(3):
        parent = None
        for s in range(9):
            rows.append({"id": f"{c}{s}", "parent_id": parent, "bytes_present": True})
            parent = f"{c}{s}"
        selected.append({"edit_id": parent, "family": "chain", "predecessor_id": f"{c}7"})
    pairs, meta = g2.audit_pairs({"registered": {"selected": selected}}, rows)
    ids = [p["predecessor_id"] for p in pairs]
    assert meta["n_candidates"] == 24 and len(pairs) == 20 and not meta["short"]
    assert ids == sorted(ids) and ids[-1] == "23"  # 8 per chain: 00-07, 10-17, 20-23


def test_gradient_only_audit_equals_pinned_predecessor_audit(toy_ctx, tmp_path):
    ctx = wikitext_only(toy_ctx)
    obj = ctx.objectives[0]
    field = rf.collect_field(
        ctx.model, ctx.tokenizer, [obj.spec], {obj.spec.panel_id: obj.panel}, ctx.supported, "cpu"
    )
    budget = g2.Budget("audit", 10.0)
    before = {n: p.detach().clone() for n, p in ctx.model.named_parameters()}
    rows = {}
    for impl in ("gradient", "pinned"):
        base = {}
        rows[impl] = g2.audit_one(ctx, field, obj, ctx.pairs[0], base, budget, impl)
        with torch.no_grad():
            for n, t in base.items():
                dict(ctx.model.named_parameters())[n].copy_(t)
    g, p = rows["gradient"], rows["pinned"]
    assert g["exact"] == pytest.approx(p["exact"], rel=1e-6, abs=1e-9)
    assert g["approximate"] == pytest.approx(p["approximate"], rel=1e-6, abs=1e-9)
    assert g["approximate"] == pytest.approx(g["base_contraction"] + g["cross_term"], abs=1e-12)
    assert g["denominator"] == max(abs(g["exact"]), 1e-3)
    assert g["n_displacement_tensors"] > 0 and g["objective"] == "wikitext_nll"
    for k in ("increment_sha256", "displacement_sha256", "predecessor_bytes", "successor_bytes"):
        assert g[k] == p[k]
    for n, p in ctx.model.named_parameters():
        assert torch.equal(p, before[n])


# --------------------------------------------------------------------- pipeline/manifest


def test_pipeline_manifest_schema_and_short_audit_hard_stop(toy_ctx, tmp_path):
    ctx = wikitext_only(toy_ctx)
    code, out, manifest = run_main(tmp_path, ctx)
    assert g2.validate_manifest(manifest) == []
    # three identified predecessors are a short audit: hard stop before any audit compute
    assert code == g2.EXIT["HARD_STOP"] and manifest["status"] == "HARD_STOP"
    assert manifest["hard_stop"]["HARD_STOP"] is True and not manifest["complete"]
    assert manifest["hard_stop"]["short_objectives"] == ["wikitext_nll"]
    assert manifest["audit"]["short_selection"] and manifest["stages"]["audit"] == "complete"
    audit = json.loads((out / "predecessor_audit.json").read_text())
    assert audit["schema"] == g2.AUDIT_SCHEMA and not audit["verdict"]["passed"]
    assert audit["verdict"]["short_objectives"] == ["wikitext_nll"]
    per = json.loads((out / "predecessor_audit" / "wikitext_nll.json").read_text())
    assert per["verdict"]["short"] and per["verdict"]["n_audited"] == 0 and per["rows"] == []
    assert len(per["short_selection"]) == 3
    assert manifest["audit"]["sha256"] == g2.sha256_file(out / "predecessor_audit.json")
    rec = manifest["fields"]["wikitext_nll"]
    root = out / "fields" / "wikitext_nll"
    assert rec["files"]["tensors.pt"] == g2.sha256_file(root / "tensors.pt")
    loaded = rf.FieldArtifact.load(root)
    assert rf._hash(loaded.identity) == rec["identity_sha256"]
    assert rf._hash(loaded.content_hashes) == rec["content_hashes_sha256"]
    assert rec["panel"]["content_hash"] == manifest["panels"]["wikitext_nll"]["content_hash"]
    assert rec["resources"]["bytes_written"] > 0 and rec["resources"]["wall_seconds"] >= 0
    proj = manifest["projectors"]
    assert proj["n_matrix_slots"] == len(ctx.supported) == 31
    assert proj["files"]["projectors.safetensors"] == g2.sha256_file(
        out / "projectors" / "projectors.safetensors"
    )
    probe = manifest["probe_cache"]
    assert probe["resources"]["n_prompts"] == 2 and len(probe["files"]) == 3
    assert manifest["stages"] == dict.fromkeys(g2.STAGES, "complete")
    # an existing manifest is never overwritten without --resume, and a hard stop is terminal
    assert g2.main(["--toy", "--out", str(out)], context_builder=lambda o, n: ctx) == 5
    resume = ["--toy", "--out", str(out), "--resume", "--stages", "audit"]
    assert g2.main(resume, context_builder=lambda o, n: ctx) == 5
    assert json.loads((out / "g2_manifest.json").read_text())["status"] == "HARD_STOP"


def test_per_objective_audit_rows_verdicts_and_manifest_schema(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    assert len(ctx.pairs) == g2.AUDIT_N == 20
    pin_field_walls(monkeypatch, REGISTERED_ORDER_WALLS)
    calls = forced_audit(monkeypatch, {n: (lambda i: 0.09 if i < 16 else 0.9) for n in TWO})
    code, out, manifest = run_main(tmp_path, ctx)
    assert calls == [TWO[0]] * 20 + [TWO[1]] * 20  # one objective gradient per predecessor each
    assert [o["objective"] for o in manifest["audit"]["order"]] == list(TWO)
    assert code == g2.EXIT["complete"] and manifest["status"] == "complete" and manifest["complete"]
    assert g2.validate_manifest(manifest) == []
    audit = manifest["audit"]
    assert audit["required_objectives"] == list(TWO) and not audit["short_selection"]
    assert [p["predecessor_id"] for p in audit["pairs"]] == [p["predecessor_id"] for p in ctx.pairs]
    est = audit["cost_estimate"]
    assert est["n_objectives"] == 2 and est["n_predecessors"] == 20 and est["n_rows"] == 40
    assert est["within_cap"] and est["cap_hours"] == 1.0
    assert set(est["per_row_seconds"]) == set(TWO)
    assert audit["verdict"]["passed"] and audit["verdict"]["failed_objectives"] == []
    for name in TWO:
        entry = audit["objectives"][name]
        assert entry["verdict"]["n_audited"] == 20 and entry["verdict"]["n_passed"] == 16
        assert entry["verdict"]["passed"] and entry["verdict"]["objective"] == name
        path = out / "predecessor_audit" / f"{name}.json"
        assert entry["sha256"] == g2.sha256_file(path)
        per = json.loads(path.read_text())
        assert per["verdict"] == entry["verdict"] and len(per["rows"]) == 20
        for row, pair in zip(per["rows"], ctx.pairs):
            assert set(g2.AUDIT_ROW_KEYS) <= set(row) and row["objective"] == name
            assert row["predecessor_id"] == pair["predecessor_id"]
            assert row["successor_id"] == pair["successor_id"]
            assert row["denominator"] == max(abs(row["exact"]), 1e-3)
            assert math.isfinite(row["base_contraction"] + row["cross_term"])
            assert len(row["increment_sha256"]) == len(row["displacement_sha256"]) == 64
    # identical identities across objectives: the same twenty predecessors are audited per objective
    rows = [json.loads((out / "predecessor_audit" / f"{n}.json").read_text())["rows"] for n in TWO]
    assert [r["increment_sha256"] for r in rows[0]] == [r["increment_sha256"] for r in rows[1]]
    top = json.loads((out / "predecessor_audit.json").read_text())
    assert set(top["objectives"]) == set(TWO) and top["verdict"]["passed"]
    # schema checks catch a short objective claimed as passing and a missing per-objective file
    broken = copy.deepcopy(manifest)
    broken["audit"]["objectives"][TWO[1]]["rows"] = broken["audit"]["objectives"][TWO[1]]["rows"][
        :19
    ]
    broken["audit"]["objectives"][TWO[1]]["verdict"]["n_audited"] = 19
    assert "a short audit cannot pass" in " ".join(g2.validate_manifest(broken))
    broken = copy.deepcopy(manifest)
    del broken["audit"]["objectives"][TWO[0]]
    errors = " ".join(g2.validate_manifest(broken))
    assert "without every required objective" in errors
    real = copy.deepcopy(manifest) | {"toy": False}
    assert "all five field objectives" in " ".join(g2.validate_manifest(real))


def test_any_objective_failure_is_hard_stop(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    # WikiText passes 20/20; HellaSwag has 15/20: G2 hard-stops, and WikiText cannot stand for it
    forced_audit(
        monkeypatch,
        {TWO[0]: lambda i: 0.0, TWO[1]: lambda i: 0.05 if i < 15 else 0.2},
    )
    code, out, manifest = run_main(tmp_path, ctx)
    assert code == g2.EXIT["HARD_STOP"] and manifest["status"] == "HARD_STOP"
    assert not manifest["complete"] and g2.validate_manifest(manifest) == []
    stop = manifest["hard_stop"]
    assert stop["failed_objectives"] == [TWO[1]] and stop["short_objectives"] == []
    verdicts = manifest["audit"]["verdict"]
    assert not verdicts["passed"] and verdicts["failed_objectives"] == [TWO[1]]
    assert manifest["audit"]["objectives"][TWO[0]]["verdict"]["passed"]
    assert manifest["audit"]["objectives"][TWO[1]]["verdict"]["n_passed"] == 15
    resume = ["--toy", "--out", str(out), "--resume", "--stages", "audit"]
    assert g2.main(resume, context_builder=lambda o, n: ctx) == g2.EXIT["refused"]
    # a manifest claiming completion with a failed objective is a schema error
    claimed = copy.deepcopy(manifest) | {"status": "complete", "complete": True}
    errors = " ".join(g2.validate_manifest(claimed))
    assert "without a passing per-objective audit" in errors
    assert "without HARD_STOP status" in errors


def test_audit_resumes_per_objective_without_rerunning_completed(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    pin_field_walls(monkeypatch, REGISTERED_ORDER_WALLS)
    calls = forced_audit(monkeypatch, {n: (lambda i: 0.0) for n in TWO})
    original = g2._audit_objective
    state = {"stopped": False}

    def stop_second(ctx_, out_, manifest_, budget, mpath, impl, objective, base, **kw):
        if objective.spec.name == TWO[1] and not state["stopped"]:
            state["stopped"] = True
            raise g2.CapExceeded("audit_gpu_hours", f"audit:{TWO[1]}", 1.01, 1.0)
        return original(ctx_, out_, manifest_, budget, mpath, impl, objective, base, **kw)

    monkeypatch.setattr(g2, "_audit_objective", stop_second)
    code, out, first = run_main(tmp_path, ctx)
    assert code == g2.EXIT["incomplete"] and first["incomplete_reason"]["cap"] == "audit_gpu_hours"
    assert first["audit"]["objectives"][TWO[0]]["status"] == "complete"
    assert TWO[1] not in first["audit"]["objectives"]
    assert g2.validate_manifest(first) == []
    done = first["audit"]["objectives"][TWO[0]]
    argv = ["--toy", "--out", str(out), "--resume", "--stages", "audit"]
    assert g2.main(argv, context_builder=lambda o, n: ctx) == g2.EXIT["complete"]
    manifest = json.loads((out / "g2_manifest.json").read_text())
    kept = manifest["audit"]["objectives"][TWO[0]]
    assert kept["sha256"] == done["sha256"] and kept["reused_in_attempt"] == 2
    assert calls.count(TWO[0]) == 20 and calls.count(TWO[1]) == 20  # first objective not re-run
    assert manifest["audit"]["verdict"]["passed"] and g2.validate_manifest(manifest) == []
    assert manifest["attempts"][0]["audit"]["objectives"][TWO[0]]["sha256"] == done["sha256"]


def fields_with_wall(seconds):
    return {o: {"resources": {"wall_seconds": w}} for o, w in seconds.items()}


def test_row_cost_guard_is_one_tenth_of_the_audit_cap():
    # the G2 field wall times of 10 Nov 2026 (data/runs/g2/g2_manifest.json), cap 1 GPU-h -> 360 s
    fields = fields_with_wall(
        {"wikitext_nll": 1648.599084167974, "mmlu_gold_log_prob": 613.4179068059893}
    )
    guard = g2.row_cost_guard(fields, "wikitext_nll", 1.0, 1.0)
    assert guard["limit_seconds"] == pytest.approx(360.0)
    assert guard["estimated_row_seconds"] == pytest.approx(824.2995, rel=1e-6)
    assert not guard["within_guard"]
    assert g2.row_cost_guard(fields, "mmlu_gold_log_prob", 1.0, 1.0)["within_guard"]
    # a measured ratio from an objective already audited calibrates the one-row estimate
    calibrated = g2.row_cost_guard(fields, "wikitext_nll", 0.1, 1.0)
    assert calibrated["field_based_row_seconds"] == pytest.approx(824.2995, rel=1e-6)
    assert calibrated["estimated_row_seconds"] == pytest.approx(82.42995, rel=1e-6)
    assert calibrated["within_guard"]


def test_first_row_cost_check_uses_measured_row_and_max_ratio_for_later():
    fields = fields_with_wall({"a": 100.0, "b": 200.0, "c": 40.0})  # rows estimated 50, 100, 20 s
    check = g2.first_row_cost_check(fields, "a", 5.0, 20, ["b", "c"], {"z": 0.3}, 0.01, 1.0)
    assert check["ratios"] == {"z": 0.3, "a": pytest.approx(0.1)}
    assert check["calibration_ratio"] == pytest.approx(0.3)  # the largest measured/estimated ratio
    assert check["remaining_rows"] == 19 and check["remaining_seconds"] == pytest.approx(95.0)
    assert check["later_objectives_seconds"] == {
        "b": pytest.approx(20 * 100 * 0.3),
        "c": pytest.approx(20 * 20 * 0.3),
    }
    total = 0.01 + (95.0 + 600.0 + 120.0) / 3600.0
    assert check["estimated_total_hours"] == pytest.approx(total) and check["within_cap"]
    over = g2.first_row_cost_check(fields, "a", 200.0, 20, [], {}, 0.0, 1.0)
    assert over["calibration_ratio"] == pytest.approx(4.0)
    assert (
        over["estimated_total_hours"] == pytest.approx(19 * 200 / 3600) and not over["within_cap"]
    )


def clocked_audit(monkeypatch, row_seconds):
    """Forced audit rows on a fake clock that only advances inside a row by row_seconds(name, i)."""
    clock, calls = {"t": 0.0}, []
    original = g2.audit_one
    monkeypatch.setattr(g2, "now", lambda: clock["t"])

    def patched(ctx, field, objective, pair, base, budget, impl):
        row = original(ctx, field, objective, pair, base, budget, impl)
        name = objective.spec.name
        clock["t"] += row_seconds(name, sum(1 for c in calls if c == name))
        calls.append(name)
        row["approximate"] = row["exact"]
        row["relative_error"] = 0.0
        return row

    monkeypatch.setattr(g2, "audit_one", patched)
    return calls


def test_cost_check_runs_after_the_first_registered_row(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    # 30 s rows: 19 x 30 s after the first row exceeds a 0.1 GPU-h cap; row 1 alone does not
    calls = clocked_audit(monkeypatch, lambda name, i: 30.0)
    code, out, manifest = run_main(tmp_path, ctx, "--max-audit-gpu-hours", "0.1")
    assert code == g2.EXIT["incomplete"] and calls == [TWO[0]]  # exactly the first registered row
    reason = manifest["incomplete_reason"]
    assert reason["cap"] == "audit_gpu_hours_estimate" and "after row 1" in reason["where"]
    (check,) = manifest["audit"]["cost_checks"]
    assert check["after_first_row_of"] == TWO[0] and check["rows_done"] == 1
    assert check["measured_first_row_seconds"] == pytest.approx(30.0)
    assert check["remaining_rows"] == 19 and set(check["later_objectives_seconds"]) == {TWO[1]}
    assert check["estimated_total_hours"] > 0.1 and not check["within_cap"]
    entry = manifest["audit"]["objectives"][TWO[0]]
    assert entry["status"] == "stopped" and len(entry["rows"]) == 1  # row 1 counts and is kept
    assert entry["rows"][0]["resources"]["wall_seconds"] == pytest.approx(30.0)
    assert entry["cost_guard"]["within_guard"]  # the guard ran before row 1
    assert TWO[1] not in manifest["audit"]["objectives"] and g2.validate_manifest(manifest) == []
    assert not (out / "predecessor_audit" / f"{TWO[0]}.json").exists()


def test_cost_checks_follow_each_objectives_first_row_within_cap(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    calls = clocked_audit(monkeypatch, lambda name, i: 1.0)
    code, _, manifest = run_main(tmp_path, ctx)
    assert code == g2.EXIT["complete"] and calls == [TWO[0]] * 20 + [TWO[1]] * 20
    checks = manifest["audit"]["cost_checks"]
    assert [c["after_first_row_of"] for c in checks] == list(TWO)
    assert all(c["within_cap"] and c["measured_first_row_seconds"] == 1.0 for c in checks)
    assert checks[1]["later_objectives_seconds"] == {}
    assert manifest["audit"]["cost_estimate"]["role"].startswith("recorded field-based estimate")
    assert g2.validate_manifest(manifest) == []


def test_cost_guard_aborts_an_objective_without_running_it(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    pin_field_walls(monkeypatch, REGISTERED_ORDER_WALLS)
    calls = forced_audit(monkeypatch, {n: (lambda i: 0.0) for n in TWO})
    code, _, manifest = run_main(tmp_path, ctx, "--max-audit-gpu-hours", "1e-9")
    assert code == g2.EXIT["incomplete"] and calls == []
    reason = manifest["incomplete_reason"]
    assert reason["cap"] == "audit_cost_guard" and reason["objectives"] == list(TWO)
    assert reason["limit"] == pytest.approx(0.1 * 1e-9)
    for name in TWO:
        entry = manifest["audit"]["objectives"][name]
        assert entry["status"] == "cost_guard" and entry["rows"] == []
        assert not entry["cost_guard"]["within_guard"]
        assert entry["cost_guard"]["limit_seconds"] == pytest.approx(0.1 * 1e-9 * 3600)
    est = manifest["audit"]["cost_estimate"]
    assert est["n_rows"] == 40 and not est["within_cap"]  # recorded, not the stop criterion
    assert manifest["stages"]["audit"] == "stopped" and manifest["status"] == "incomplete"
    assert g2.validate_manifest(manifest) == []


def test_cost_guard_skips_one_objective_and_audits_the_next(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    calls = forced_audit(monkeypatch, {n: (lambda i: 0.0) for n in TWO})
    original = g2.row_cost_guard

    def guard(fields, name, ratio, cap_hours):
        g = original(fields, name, ratio, cap_hours)
        return g | {"within_guard": name != TWO[0]}

    monkeypatch.setattr(g2, "row_cost_guard", guard)
    code, out, manifest = run_main(tmp_path, ctx)
    assert code == g2.EXIT["incomplete"] and calls == [TWO[1]] * 20
    assert manifest["incomplete_reason"]["objectives"] == [TWO[0]]
    assert manifest["audit"]["objectives"][TWO[0]]["status"] == "cost_guard"
    assert manifest["audit"]["objectives"][TWO[1]]["status"] == "complete"
    # the guarded objective is not dropped: a resume audits it, the completed one is kept
    monkeypatch.setattr(g2, "row_cost_guard", original)
    argv = ["--toy", "--out", str(out), "--resume", "--stages", "audit"]
    assert g2.main(argv, context_builder=lambda o, n: ctx) == g2.EXIT["complete"]
    assert calls == [TWO[1]] * 20 + [TWO[0]] * 20


# A5.8.1 note (10 Nov 2026): per-row estimates of the G2 run (0.5 x field wall seconds)
G2_ROW_SECONDS = {
    "refusal_margin_harmful": 262.53534637595294,
    "mmlu_gold_log_prob": 306.70895340299467,
    "refusal_margin_harmless": 344.7181081150193,
    "hellaswag_gold_char_log_likelihood": 367.5022726369789,
    "wikitext_nll": 824.299542083987,
}


def test_audit_order_is_ascending_field_based_row_estimate():
    fields = fields_with_wall({n: 2 * s for n, s in G2_ROW_SECONDS.items()})
    assert g2.audit_order(fields, list(g2.AUDIT_OBJECTIVES)) == list(G2_ROW_SECONDS)
    # ties keep the registered order
    tied = fields_with_wall({"a": 10.0, "b": 4.0, "c": 10.0})
    assert g2.audit_order(tied, ["c", "a", "b"]) == ["b", "c", "a"]


def test_audit_processes_the_cheapest_objective_first(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    pin_field_walls(monkeypatch, {TWO[0]: 4.0, TWO[1]: 2.0})  # registered order is not ascending
    calls = clocked_audit(monkeypatch, lambda name, i: 1.0)
    code, _, manifest = run_main(tmp_path, ctx)
    assert code == g2.EXIT["complete"] and calls == [TWO[1]] * 20 + [TWO[0]] * 20
    audit = manifest["audit"]
    assert audit["order"] == [
        {"objective": TWO[1], "field_based_row_seconds": 1.0},
        {"objective": TWO[0], "field_based_row_seconds": 2.0},
    ]
    assert [c["after_first_row_of"] for c in audit["cost_checks"]] == [TWO[1], TWO[0]]
    assert set(audit["cost_checks"][0]["later_objectives_seconds"]) == {TWO[0]}
    decisions = audit["guard_decisions"]
    assert [(d["phase"], d["objective"], d["within_guard"]) for d in decisions] == [
        ("in order", TWO[1], True),
        ("in order", TWO[0], True),
    ]
    assert decisions[1]["calibration_ratio"] == pytest.approx(1.0)  # 1 s measured / 1 s estimated
    assert manifest["config"]["audit"]["objective_order"].startswith("ascending field-based")
    assert g2.validate_manifest(manifest) == []


def test_cost_guarded_objective_is_rechecked_and_run_after_calibration(
    toy_ctx, tmp_path, monkeypatch
):
    ctx = with_objectives(toy_ctx, TWO)
    # Under the ascending order the in-order guard already sees every calibration, so the end-of-run
    # re-check is exercised with the order reversed: WikiText (824.3 s row estimate > 360 s) comes
    # first and is guarded uncalibrated; HellaSwag (100 s estimate, 10 s measured, ratio 0.1) runs;
    # the re-check then estimates WikiText at 82.4 s and audits it in the same run.
    pin_field_walls(monkeypatch, {TWO[0]: 1648.599084167974, TWO[1]: 200.0})
    monkeypatch.setattr(g2, "audit_order", lambda fields, names: sorted(names, key=TWO.index))
    calls = clocked_audit(monkeypatch, lambda name, i: 10.0)
    code, out, manifest = run_main(tmp_path, ctx)
    assert code == g2.EXIT["complete"] and manifest["status"] == "complete"
    assert calls == [TWO[1]] * 20 + [TWO[0]] * 20  # one run, no resume
    audit = manifest["audit"]
    decisions = [(d["phase"], d["objective"], d["within_guard"]) for d in audit["guard_decisions"]]
    assert decisions == [
        ("in order", TWO[0], False),
        ("in order", TWO[1], True),
        ("end-of-run re-check", TWO[0], True),
    ]
    first, recheck = audit["guard_decisions"][0], audit["guard_decisions"][2]
    assert first["calibration_ratio"] == 1.0 and first["estimated_row_seconds"] > 360.0
    assert recheck["calibration_ratio"] == pytest.approx(0.1)
    assert recheck["estimated_row_seconds"] == pytest.approx(82.4299542, rel=1e-6)
    entry = audit["objectives"][TWO[0]]
    assert entry["status"] == "complete" and len(entry["rows"]) == 20
    assert entry["cost_guard"] == {k: v for k, v in recheck.items() if k != "phase"}
    assert [c["after_first_row_of"] for c in audit["cost_checks"]] == [TWO[1], TWO[0]]
    assert audit["verdict"]["passed"] and g2.validate_manifest(manifest) == []
    assert (out / "predecessor_audit" / f"{TWO[0]}.json").exists()


def test_still_failing_recheck_leaves_cost_guard_and_run_incomplete(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    # the G2 numbers with the cheapest objective guarded: nothing calibrates, both stay guarded
    pin_field_walls(monkeypatch, {TWO[0]: 1648.599084167974, TWO[1]: 735.0045452739578})
    calls = clocked_audit(monkeypatch, lambda name, i: 1.0)
    code, out, manifest = run_main(tmp_path, ctx)
    assert code == g2.EXIT["incomplete"] and calls == []
    assert manifest["status"] == "incomplete" and manifest["stages"]["audit"] == "stopped"
    reason = manifest["incomplete_reason"]
    assert reason["cap"] == "audit_cost_guard" and reason["objectives"] == [TWO[1], TWO[0]]
    audit = manifest["audit"]
    assert [o["objective"] for o in audit["order"]] == [TWO[1], TWO[0]]
    decisions = [(d["phase"], d["objective"], d["within_guard"]) for d in audit["guard_decisions"]]
    assert decisions == [
        ("in order", TWO[1], False),
        ("in order", TWO[0], False),
        ("end-of-run re-check", TWO[1], False),
        ("end-of-run re-check", TWO[0], False),
    ]
    for name in TWO:
        entry = audit["objectives"][name]
        assert entry["status"] == "cost_guard" and entry["rows"] == []
    assert not (out / "predecessor_audit.json").exists() and g2.validate_manifest(manifest) == []


def test_measured_overrun_stops_incomplete_before_the_next_row(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    # rows of 10 s pass the first-row check (10 + 19 x 10 + later 0 s <= 360 s); the 4th row takes
    # 100 s: used 130 s, mean 32.5 s x 16 remaining rows > 360 s -> stop before row 5
    calls = clocked_audit(monkeypatch, lambda name, i: 10.0 if i < 3 else 100.0)
    code, _, manifest = run_main(tmp_path, ctx, "--max-audit-gpu-hours", "0.1")
    assert code == g2.EXIT["incomplete"] and calls == [TWO[0]] * 4
    reason = manifest["incomplete_reason"]
    assert reason["cap"] == "audit_gpu_hours_row_guard" and "(row 5)" in reason["where"]
    assert reason["used"] == pytest.approx((130 + 32.5 * 16) / 3600)
    assert manifest["caps"]["audit"]["used_hours"] < 0.1  # stopped before the cap was spent
    entry = manifest["audit"]["objectives"][TWO[0]]
    assert entry["status"] == "stopped" and len(entry["rows"]) == 4  # no row dropped
    seconds = [r["resources"]["wall_seconds"] for r in entry["rows"]]
    assert seconds == [10.0, 10.0, 10.0, 100.0]
    assert g2.validate_manifest(manifest) == []


def test_peak_memory_and_gpu_seconds_in_field_and_row_resources(toy_ctx, tmp_path, monkeypatch):
    ctx = with_objectives(toy_ctx, TWO)
    forced_audit(monkeypatch, {n: (lambda i: 0.0) for n in TWO})
    peak = {"max_memory_allocated": 1234, "max_memory_reserved": 5678}
    monkeypatch.setattr(g2, "_peak", lambda device: dict(peak))
    code, out, manifest = run_main(tmp_path, ctx)
    assert code == g2.EXIT["complete"]
    for name in TWO:
        res = manifest["fields"][name]["resources"]
        assert res["max_memory_allocated"] == 1234 and res["max_memory_reserved"] == 5678
        assert res["gpu_seconds"] == res["wall_seconds"] >= 0
        entry = manifest["audit"]["objectives"][name]
        assert entry["resources"]["max_memory_allocated"] == 1234
        for row in entry["rows"]:
            assert row["resources"]["max_memory_allocated"] == 1234
            assert row["resources"]["gpu_seconds"] == row["resources"]["wall_seconds"] >= 0
        per = json.loads((out / "predecessor_audit" / f"{name}.json").read_text())
        assert all(r["resources"]["max_memory_allocated"] == 1234 for r in per["rows"])
    for key in ("projectors", "probe_cache"):
        assert manifest[key]["resources"]["max_memory_allocated"] == 1234
        assert "gpu_seconds" in manifest[key]["resources"]
    summary = manifest["resources_summary"]
    assert summary["fields"][TWO[0]]["max_memory_allocated"] == 1234
    assert (
        summary["projectors"]["gpu_seconds"] == manifest["projectors"]["resources"]["wall_seconds"]
    )
    assert len(summary["audit_rows"][TWO[1]]) == 20
    assert g2.NOT_CAPTURED not in json.dumps(summary)


def test_resource_reader_never_fabricates_uncaptured_peaks():
    # the 10 Nov 2026 G2 run recorded projector resources as wall seconds and bytes only, and field
    # GPU time only as gpu_hours
    manifest = {
        "fields": {
            "wikitext_nll": {
                "status": "complete",
                "resources": {
                    "wall_seconds": 1648.599084167974,
                    "gpu_hours": 0.4579441900466595,
                    "max_memory_allocated": 10932988928,
                    "max_memory_reserved": 11911823360,
                },
            }
        },
        "projectors": {
            "status": "complete",
            "resources": {"bytes_written": 10992053, "wall_seconds": 0.12582707998808473},
        },
        "audit": {"objectives": {}},
    }
    summary = g2.resource_summary(manifest)
    field = summary["fields"]["wikitext_nll"]
    assert field["max_memory_allocated"] == 10932988928
    assert field["gpu_seconds"] == pytest.approx(1648.599084167974)
    proj = summary["projectors"]
    assert proj["wall_seconds"] == pytest.approx(0.12582707998808473)
    for k in ("gpu_seconds", "max_memory_allocated", "max_memory_reserved"):
        assert proj[k] == g2.NOT_CAPTURED == "not captured in this run"
    assert "probe_cache" not in summary and summary["audit_rows"] == {}


def test_resume_runs_only_the_audit_stage_on_verified_artefacts(toy_ctx, tmp_path, monkeypatch):
    ctx = wikitext_only(toy_ctx, n_pairs=20)
    code, out, first = run_main(tmp_path, ctx, "--stages", "fields,projectors,probe")
    assert code == g2.EXIT["incomplete"] and first["incomplete_reason"]["pending_stages"] == [
        "audit"
    ]
    files = {
        p.relative_to(out): g2.sha256_file(p)
        for d in ("fields", "projectors", "probe_cache")
        for p in sorted((out / d).rglob("*"))
        if p.is_file()
    }
    mtimes = {p: (out / p).stat().st_mtime_ns for p in files}
    calls = forced_audit(monkeypatch, {"wikitext_nll": lambda i: 0.0})
    monkeypatch.setattr(g2, "run_fields", lambda *a, **k: pytest.fail("fields re-run"))
    monkeypatch.setattr(g2, "run_projectors", lambda *a, **k: pytest.fail("projectors re-run"))
    monkeypatch.setattr(g2, "run_probe", lambda *a, **k: pytest.fail("probe re-run"))
    code, out, manifest = run_main(tmp_path, ctx, "--resume", "--stages", "audit")
    assert code == g2.EXIT["complete"] and calls == ["wikitext_nll"] * 20
    assert manifest["attempt"] == 2 and manifest["stages"] == dict.fromkeys(g2.STAGES, "complete")
    for key in ("fields", "projectors", "probe_cache"):
        assert manifest[key] == first[key]
    assert manifest["resume_verified"]["artefacts"] == [
        "fields/wikitext_nll",
        "projectors",
        "probe_cache",
    ]
    assert {p: g2.sha256_file(out / p) for p in files} == files
    assert {p: (out / p).stat().st_mtime_ns for p in files} == mtimes
    assert manifest["audit"]["verdict"]["passed"] and g2.validate_manifest(manifest) == []


def test_audit_resume_refuses_changed_completed_artefacts(toy_ctx, tmp_path):
    ctx = wikitext_only(toy_ctx, n_pairs=20)
    code, out, first = run_main(tmp_path, ctx, "--stages", "fields,projectors,probe")
    assert code == g2.EXIT["incomplete"]
    path = out / "projectors" / "projectors.json"
    path.write_text(path.read_text() + " ")
    resume = ["--toy", "--out", str(out), "--resume", "--stages", "audit"]
    assert g2.main(resume, context_builder=lambda o, n: ctx) == g2.EXIT["refused"]
    after = json.loads((out / "g2_manifest.json").read_text())
    assert after == first  # refused before anything was written


def test_wall_clock_cap_stops_cleanly_with_incomplete_manifest(toy_ctx, tmp_path, monkeypatch):
    ctx = wikitext_only(toy_ctx)
    clock = {"t": 0.0}

    def fake_now():
        clock["t"] += 0.2 * 3600  # 0.2 GPU-h per clock read
        return clock["t"]

    monkeypatch.setattr(g2, "now", fake_now)
    code, out, manifest = run_main(tmp_path, ctx, "--max-gpu-hours", "1.0")
    assert code == g2.EXIT["incomplete"]
    assert manifest["status"] == "incomplete" and not manifest["complete"]
    assert manifest["incomplete_reason"]["cap"] == "fields_and_cache_gpu_hours"
    assert manifest["stages"]["fields"] == "stopped"
    assert manifest["fields"]["wikitext_nll"]["status"] == "running"
    assert not (out / "fields" / "wikitext_nll").exists()
    assert manifest["caps"]["fields"]["used_hours"] > 1.0
    assert g2.validate_manifest(manifest) == []


def test_audit_cap_on_resume_keeps_fields_and_charges_prior_hours(toy_ctx, tmp_path, monkeypatch):
    ctx = wikitext_only(toy_ctx, n_pairs=20)
    code, out, first = run_main(tmp_path, ctx, "--stages", "fields")
    assert code == g2.EXIT["incomplete"] and first["incomplete_reason"]["pending_stages"]
    clock = {"t": 0.0}

    def fake_now():
        clock["t"] += 0.3 * 3600
        return clock["t"]

    monkeypatch.setattr(g2, "now", fake_now)
    argv = ["--resume", "--stages", "audit", "--max-audit-gpu-hours", "1.0"]
    code, out, manifest = run_main(tmp_path, ctx, *argv)
    assert code == g2.EXIT["incomplete"]
    assert manifest["incomplete_reason"]["cap"].startswith("audit_gpu_hours")
    assert manifest["caps"]["audit"]["used_hours"] > 0
    assert manifest["attempt"] == 2 and manifest["attempts"][0]["status"] == "incomplete"
    assert manifest["fields"]["wikitext_nll"] == first["fields"]["wikitext_nll"]
    assert (
        manifest["audit"]["status"] == "running" and not (out / "predecessor_audit.json").exists()
    )
    assert g2.validate_manifest(manifest) == []


def test_missing_hook_and_factor_audit_failures_are_gate_failures(toy_ctx, tmp_path, monkeypatch):
    ctx = wikitext_only(toy_ctx)
    bad = dataclasses.replace(ctx, supported={**ctx.supported, "model.layers.0.mlp": "mlp:block"})
    code, _, manifest = run_main(tmp_path / "hook", bad, "--stages", "fields")
    assert code == g2.EXIT["GATE_FAILURE"] and manifest["status"] == "GATE_FAILURE"
    assert "MissingHookError" in manifest["gate_failure"]["reason"]
    # A materially negative eigenvalue (below -1e-6 x lambda_max) still fails the gate
    original = rf.spectral_factor

    def shifted(factor):
        return original(factor - 2e-6 * factor.abs().max() * torch.eye(len(factor)))

    monkeypatch.setattr(rf, "spectral_factor", shifted)
    code, _, manifest = run_main(tmp_path / "psd", ctx, "--stages", "fields")
    assert code == g2.EXIT["GATE_FAILURE"]
    assert "materially negative factor eigenvalue" in manifest["gate_failure"]["reason"]
    assert manifest["gate_failure"]["objective"] == "wikitext_nll"


def small_mmlu(ctx, n=3):
    mmlu = next(o for o in ctx.objectives if o.spec.name == "mmlu_gold_log_prob")
    items = list(mmlu.panel.items())[:n]
    panel = rf.FieldPanel(
        mmlu.panel.panel_id,
        lambda: iter(items),
        mmlu.panel.text_groups,
        mmlu.panel.scored_text_groups,
    )
    record = mmlu.record | {"content_hash": rf._hash([rf._hash(rf._hash_item(i)) for i in items])}
    return dataclasses.replace(
        wikitext_only(ctx), objectives=[g2.Objective(mmlu.spec, panel, record)]
    )


def test_rank_deficient_factors_pass_and_record_clamps_per_objective(toy_ctx, tmp_path):
    # 3 MMLU items give rank-3 factors: fp32 rounding leaves small negative eigenvalues that the
    # schema-1 rule (1e-6 x trace/dim) rejected; the A5.8.1 amendment clamps and counts them.
    code, _, manifest = run_main(tmp_path, small_mmlu(toy_ctx), "--stages", "fields")
    assert code == g2.EXIT["incomplete"] and "gate_failure" not in manifest
    assert g2.validate_manifest(manifest) == []
    assert manifest["config"]["field"]["schema_version"] == rf.SCHEMA_VERSION == 2
    assert manifest["config"]["field"]["eigendecomposition"] == "float64"
    assert manifest["config"]["field"]["negative_tolerance_relative_to_lambda_max"] == 1e-6
    rec = manifest["fields"]["mmlu_gold_log_prob"]
    clamps = rec["eigen_clamps"]
    audits = [a for m in rec["factor_audits"].values() for a in (m["A"], m["G"])]
    assert clamps["n_factors"] == len(audits) == 2 * len(rec["factor_audits"])
    assert clamps["n_eigenvalues_clamped"] == sum(a["n_clamped"] for a in audits) > 0
    assert clamps["n_factors_clamped"] == sum(1 for a in audits if a["n_clamped"]) > 0
    assert -1e-6 <= clamps["most_negative_ratio"] == min(a["most_negative_ratio"] for a in audits)
    assert clamps["most_negative_ratio"] < 0 and clamps["most_negative_factor"] is not None
    module, key = clamps["most_negative_factor"].rsplit("/", 1)
    assert rec["factor_audits"][module][key]["most_negative_ratio"] == clamps["most_negative_ratio"]


def test_clamp_summary_without_clamps():
    audit = {"n_clamped": 0, "most_negative_ratio": 0.0, "dimension": 3}
    summary = g2.clamp_summary({"m": {"A_audit": audit, "G_audit": audit}})
    assert summary["n_eigenvalues_clamped"] == 0 and summary["most_negative_factor"] is None
    assert summary["n_eigenvalues"] == 6 and summary["most_negative_ratio"] == 0.0


# ------------------------------------------------------- chain_exact (A5 amendment F6, 11 Nov 2026)

CHAIN = ("wikitext_nll", "mmlu_gold_log_prob")
MASKED = ("hellaswag_gold_char_log_likelihood", "refusal_margin_harmful", "refusal_margin_harmless")


def chain_ctx(ctx):
    return with_objectives(ctx, CHAIN)


def test_chain_exact_picks_resolve_predecessors_aliases_and_roots():
    rows = [
        {"id": "r1", "parent_id": None, "bytes_present": True, "bytes_via_alias": None},
        {"id": "a2", "parent_id": "r1", "bytes_present": False, "bytes_via_alias": "h2"},
        {"id": "p3", "parent_id": "a2", "bytes_present": True, "bytes_via_alias": None},
        {"id": "root", "parent_id": None, "bytes_present": True, "bytes_via_alias": None},
    ]
    selection = {
        "registered": {
            "selected": [
                {"edit_id": "p3", "family": "chain", "predecessor_id": "a2"},
                {"edit_id": "root", "family": "chain", "predecessor_id": "base"},
                {"edit_id": "L", "family": "lora_sft", "predecessor_id": "base"},
            ]
        }
    }
    picks, meta = g2.chain_exact_picks(selection, rows)
    assert [p["pick_id"] for p in picks] == ["p3", "root"]  # chain only, sorted; LoRA unaffected
    assert picks[0]["predecessor_bytes"] == "h2" and picks[0]["predecessor_via_alias"]
    assert picks[1]["predecessor_is_base"] and picks[1]["predecessor_bytes"] is None
    assert meta["n_picks"] == 2 and meta["n_base_predecessor"] == 1
    bad = copy.deepcopy(selection)
    bad["registered"]["selected"][0]["predecessor_id"] = "r1"  # not the lineage parent
    with pytest.raises(ValueError, match="differs from lineage"):
        g2.chain_exact_picks(bad, rows)


def test_chain_exact_picks_on_the_registered_selection():
    sel = json.loads((g2.MANIFESTS / "selection.json").read_text())
    comps = [
        json.loads(x) for x in (g2.MANIFESTS / "components.jsonl").read_text().splitlines() if x
    ]
    picks, meta = g2.chain_exact_picks(sel, comps)
    assert meta["n_picks"] == g2.CHAIN_EXACT_N_PICKS == 10 and meta["n_base_predecessor"] == 2
    assert sum(p["predecessor_via_alias"] for p in picks) == 2


def test_chain_exact_stage_rows_masks_artefacts_and_manifest(toy_ctx, tmp_path):
    ctx = chain_ctx(toy_ctx)
    before = {n: p.detach().clone() for n, p in ctx.model.named_parameters()}
    code, out, manifest = run_main(tmp_path, ctx, "--stages", "chain_exact")
    assert code == g2.EXIT["incomplete"] and g2.validate_manifest(manifest) == []
    assert manifest["incomplete_reason"]["pending_stages"] == ["fields", "projectors", "probe"]
    assert manifest["stages"]["chain_exact"] == "complete"
    block = manifest["chain_exact"]
    picks = [p["pick_id"] for p in ctx.chain_picks]
    assert len(picks) == 4 and block["n_rows"] == 8 == block["n_complete"]
    assert [(r["pick_id"], r["objective"]) for r in block["rows"]] == [
        (p, o) for p in picks for o in CHAIN
    ]
    assert block["objectives"] == list(CHAIN) and block["masked_objectives"] == list(MASKED)
    for p in picks:
        assert all(block["masks"][p][o]["presence"] == 0 for o in MASKED)
        assert all(block["masks"][p][o]["presence"] == 1 for o in CHAIN)
        assert all(block["masks"][p][o]["q_t"] == "base-factor" for o in CHAIN)
    # F6.1: the audit's disposition is recorded and the approximation is not used for chains
    disposition = manifest["audit_disposition"]
    assert disposition["approximation_used_for_chains"] is False
    assert disposition["amendment"].endswith("F6.1")
    # measured first-row check after the first row of each objective
    assert [c["after_first_row_of"] for c in block["cost_checks"]] == list(CHAIN)
    assert all(c["within_cap"] and c["cap_hours"] == 2.2 for c in block["cost_checks"])
    root_pick = next(r for r in block["rows"] if r["predecessor_is_base"])
    assert root_pick["displacement_sha256"] == rf._hash({})
    for row in block["rows"]:
        d = out / "chain_exact" / row["pick_id"] / row["objective"]
        assert row["dir"] == str(d) and row["files"]["tensors.pt"] == g2.sha256_file(
            d / "tensors.pt"
        )
        art = rf.FieldArtifact.load(d)
        assert art.factors == {} and list(art.gradients) == [row["objective"]]
        assert (
            art.identity["anchor"] == "actual predecessor" and art.identity["q_t"] == "base-factor"
        )
        assert art.identity["pick"]["pick_id"] == row["pick_id"]
        assert rf._hash(art.identity) == row["identity_sha256"]
        assert rf._hash(art.content_hashes) == row["content_hashes_sha256"]
        assert len(art.gradients[row["objective"]]) == row["n_tensors"] > 0
        res = row["resources"]
        assert res["gpu_seconds"] == res["wall_seconds"] >= 0 and "max_memory_allocated" in res
    first = block["rows"][0]
    assert first["resources"]["state_load_seconds"] >= 0
    assert block["rows"][1]["resources"]["state_load_seconds"] == 0.0  # one load per pick
    summary = json.loads((out / "chain_exact" / "chain_exact.json").read_text())
    assert block["sha256"] == g2.sha256_file(out / "chain_exact" / "chain_exact.json")
    assert summary["schema"] == g2.CHAIN_EXACT_SCHEMA and len(summary["rows"]) == 8
    assert manifest["caps"]["chain_exact"]["cap_hours"] == 2.2
    for n, p in ctx.model.named_parameters():  # the model is back at the base
        assert torch.equal(p, before[n])


def test_chain_exact_gradient_is_the_pinned_gradient_at_the_predecessor(toy_ctx, tmp_path):
    ctx = chain_ctx(toy_ctx)
    _, _, manifest = run_main(tmp_path, ctx, "--stages", "chain_exact")
    pick = next(p for p in ctx.chain_picks if not p["predecessor_is_base"])
    obj = next(o for o in ctx.objectives if o.spec.name == "wikitext_nll")
    row = next(
        r
        for r in manifest["chain_exact"]["rows"]
        if r["pick_id"] == pick["pick_id"] and r["objective"] == "wikitext_nll"
    )
    base = {}
    state = g2._chain_state(ctx, pick, base)
    try:
        field = rf.collect_field(
            ctx.model,
            ctx.tokenizer,
            [obj.spec],
            {obj.spec.panel_id: obj.panel},
            ctx.supported,
            "cpu",
        )
    finally:
        g2._restore(ctx, base)
    art = rf.FieldArtifact.load(row["dir"])
    grads = art.gradients["wikitext_nll"]
    assert set(grads) == set(state["increment"])
    for n, g in grads.items():
        torch.testing.assert_close(g, field.gradients["wikitext_nll"][n], rtol=1e-5, atol=1e-8)
    exact = sum((grads[n] * d).sum().item() for n, d in state["increment"].items())
    assert row["exact_contraction"] == pytest.approx(exact)
    assert row["increment_sha256"] == state["increment_sha256"]
    assert row["predecessor_state_sha256"] == state["predecessor_state_sha256"]


def clocked_chain(monkeypatch, row_seconds):
    """Real chain_exact rows on a fake clock that advances only inside a row by row_seconds(i)."""
    clock, calls = {"t": 0.0}, []
    original = g2._chain_row
    monkeypatch.setattr(g2, "now", lambda: clock["t"])

    def patched(ctx, out, objective, pick, state, budget, code):
        row = original(ctx, out, objective, pick, state, budget, code)
        clock["t"] += row_seconds(len(calls))
        calls.append((pick["pick_id"], objective.spec.name))
        return row

    monkeypatch.setattr(g2, "_chain_row", patched)
    return calls


def test_chain_exact_first_row_check_stops_and_keeps_the_row(toy_ctx, tmp_path, monkeypatch):
    ctx = chain_ctx(toy_ctx)
    # 10 s rows against 0.01 GPU-h (36 s): 10 + 3 x 10 + 4 x 10 > 36 after the first row
    calls = clocked_chain(monkeypatch, lambda i: 10.0)
    argv = ("--stages", "chain_exact", "--max-chain-exact-gpu-hours", "0.01")
    code, _, manifest = run_main(tmp_path, ctx, *argv)
    assert code == g2.EXIT["incomplete"] and len(calls) == 1
    reason = manifest["incomplete_reason"]
    assert (
        reason["cap"] == "chain_exact_gpu_hours_estimate"
        and "first row of wikitext_nll" in (reason["where"])
    )
    block = manifest["chain_exact"]
    assert block["status"] == "stopped" and manifest["stages"]["chain_exact"] == "stopped"
    (check,) = block["cost_checks"]
    assert check["measured_first_row_seconds"] == pytest.approx(10.0)
    assert check["remaining_rows"] == {"wikitext_nll": 3, "mmlu_gold_log_prob": 4}
    assert check["estimated_total_hours"] == pytest.approx(80.0 / 3600) and not check["within_cap"]
    assert len(block["rows"]) == 1 and block["rows"][0]["status"] == "complete"
    assert block["masks"][ctx.chain_picks[1]["pick_id"]]["wikitext_nll"]["presence"] == "pending"
    assert g2.validate_manifest(manifest) == []


def test_chain_exact_per_row_guard_then_resume_without_rerun(toy_ctx, tmp_path, monkeypatch):
    ctx = chain_ctx(toy_ctx)
    seconds = [1.0, 1.0, 20.0]
    calls = clocked_chain(monkeypatch, lambda i: seconds[i] if i < len(seconds) else 1.0)
    argv = ("--stages", "chain_exact", "--max-chain-exact-gpu-hours", "0.01")
    code, out, first = run_main(tmp_path, ctx, *argv)
    # before row 4: 22 s used + 2 x 10.5 s (WikiText mean) + 3 x 1 s (MMLU) = 46 s > 36 s
    assert code == g2.EXIT["incomplete"] and len(calls) == 3
    reason = first["incomplete_reason"]
    assert reason["cap"] == "chain_exact_gpu_hours_row_guard" and "(row 4)" in reason["where"]
    assert reason["used"] == pytest.approx(46.0 / 3600)
    assert [c["within_cap"] for c in first["chain_exact"]["cost_checks"]] == [True, True]
    done = {(r["pick_id"], r["objective"]) for r in first["chain_exact"]["rows"]}
    # an interrupted row directory without a manifest row is kept aside, not reused
    nxt = ctx.chain_picks[1]["pick_id"]
    stale = out / "chain_exact" / nxt / "mmlu_gold_log_prob"
    stale.mkdir(parents=True)
    (stale / "partial").write_text("x")
    seconds[:] = []
    code, out, manifest = run_main(tmp_path, ctx, "--resume", "--stages", "chain_exact")
    assert code == g2.EXIT["incomplete"]  # fields/projectors/probe never ran in this toy
    assert manifest["attempt"] == 2 and manifest["stages"]["chain_exact"] == "complete"
    assert len(calls) == 8 and set(calls[3:]).isdisjoint(done)  # completed rows never re-run
    block = manifest["chain_exact"]
    assert block["n_complete"] == 8 and len(block["cost_checks"]) == 2
    assert [r["attempt"] for r in block["rows"]] == [1, 1, 1, 2, 2, 2, 2, 2]
    (moved,) = block["stale_dirs"]
    assert moved["to"].endswith("mmlu_gold_log_prob.stale-attempt2")
    assert (stale.parent / "mmlu_gold_log_prob.stale-attempt2" / "partial").exists()
    assert manifest["caps"]["chain_exact"]["prior_hours"] == pytest.approx(22.0 / 3600)
    assert manifest["attempts"][0]["caps"]["chain_exact"]["used_hours"] == pytest.approx(
        22.0 / 3600
    )
    assert manifest["caps"]["chain_exact"]["used_hours"] == pytest.approx(27.0 / 3600)  # 22 + 5 x 1
    assert manifest["caps"]["fields"]["used_hours"] == 0.0  # not charged for chain_exact time
    assert manifest["attempts"][0]["chain_exact"]["status"] == "stopped"
    assert g2.validate_manifest(manifest) == []
    # a completed stage is verified on a further resume and never re-run; tampering refuses
    code, out, again = run_main(tmp_path, ctx, "--resume", "--stages", "chain_exact")
    assert len(calls) == 8 and again["chain_exact"]["rows"] == block["rows"]
    path = out / "chain_exact" / ctx.chain_picks[0]["pick_id"] / "wikitext_nll" / "tensors.pt"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="no longer match"):
        run_main(tmp_path, ctx, "--resume", "--stages", "chain_exact")


def test_chain_exact_failed_row_is_masked_and_not_retried(toy_ctx, tmp_path, monkeypatch):
    ctx = chain_ctx(toy_ctx)
    bad = ctx.chain_picks[2]["pick_id"]
    original = g2.exact_gradient

    def failing(model, tokenizer, spec, panel, device, names):
        if spec.name == "mmlu_gold_log_prob" and failing.pick == bad:
            raise ValueError("nonfinite gradient: toy")
        return original(model, tokenizer, spec, panel, device, names)

    original_state = g2._chain_state

    def track(ctx_, pick, base):
        failing.pick = pick["pick_id"]
        return original_state(ctx_, pick, base)

    monkeypatch.setattr(g2, "exact_gradient", failing)
    monkeypatch.setattr(g2, "_chain_state", track)
    _, _, manifest = run_main(tmp_path, ctx, "--stages", "chain_exact")
    block = manifest["chain_exact"]
    assert block["status"] == "complete" and block["n_error"] == 1 and block["n_complete"] == 7
    assert block["masks"][bad]["mmlu_gold_log_prob"]["presence"] == 0
    assert block["masks"][bad]["wikitext_nll"]["presence"] == 1
    assert g2.validate_manifest(manifest) == []


def test_f6_complete_g2_needs_chain_exact_and_refuses_the_audit(toy_ctx, tmp_path, monkeypatch):
    ctx = chain_ctx(toy_ctx)
    code, out, first = run_main(tmp_path, ctx, "--stages", "fields,projectors,probe")
    assert code == g2.EXIT["incomplete"] and first["incomplete_reason"]["pending_stages"] == [
        "audit"
    ]
    monkeypatch.setattr(g2, "run_fields", lambda *a, **k: pytest.fail("fields re-run"))
    code, out, manifest = run_main(tmp_path, ctx, "--resume", "--stages", "chain_exact")
    assert code == g2.EXIT["complete"] and manifest["complete"] and manifest["status"] == "complete"
    assert manifest["stages"]["audit"] == "pending" and "audit_disposition" in manifest
    assert manifest["audit_disposition"]["audit_status"] is None
    assert g2.validate_manifest(manifest) == []
    # the WikiText/MMLU base fields calibrate the MMLU estimate before its first row
    (check, _) = manifest["chain_exact"]["cost_checks"]
    assert check["row_seconds_source"]["mmlu_gold_log_prob"].startswith("field-based")
    # after F6.1 the audit stage is refused (toy and real alike)
    resume = ["--toy", "--out", str(out), "--resume", "--stages", "audit"]
    assert g2.main(resume, context_builder=lambda o, n: ctx) == g2.EXIT["refused"]
    broken = copy.deepcopy(manifest)
    broken["chain_exact"]["masks"][ctx.chain_picks[0]["pick_id"]]["refusal_margin_harmful"] = {
        "presence": 1
    }
    broken["chain_exact"]["status"] = "stopped"
    errors = g2.validate_manifest(broken)
    assert any("not masked" in e for e in errors)
    assert "complete without a complete chain_exact stage" in errors


def test_real_run_refuses_the_audit_stage_after_f6(tmp_path, monkeypatch):
    monkeypatch.setattr(g2, "code_record", lambda: pytest.fail("no work before the refusal"))
    out = tmp_path / "g2"
    argv = ["--out", str(out), "--panels-sha256", "0" * 64, "--resume", "--stages", "audit"]
    assert g2.main(argv) == g2.EXIT["refused"] and not out.exists()


def test_chain_cost_projection_uses_measured_means_then_field_estimates():
    fields = fields_with_wall({"wikitext_nll": 1648.6, "mmlu_gold_log_prob": 613.4})
    names = list(CHAIN)
    proj = g2.chain_cost_projection(
        fields,
        names,
        {"wikitext_nll": 9, "mmlu_gold_log_prob": 10},
        {"wikitext_nll": [400.0]},
        0.1,
        2.2,
    )
    ratio = 400.0 / (0.5 * 1648.6)
    assert proj["ratios"]["wikitext_nll"] == pytest.approx(ratio)
    assert proj["row_seconds"]["mmlu_gold_log_prob"] == pytest.approx(0.5 * 613.4 * ratio)
    total = 0.1 + (9 * 400.0 + 10 * 0.5 * 613.4 * ratio) / 3600
    assert proj["estimated_total_hours"] == pytest.approx(total) and proj["within_cap"]
    over = g2.chain_cost_projection(
        fields, names, {"wikitext_nll": 19}, {"wikitext_nll": [420.0]}, 0, 2.2
    )
    assert (
        over["estimated_total_hours"] == pytest.approx(19 * 420 / 3600) and not over["within_cap"]
    )
