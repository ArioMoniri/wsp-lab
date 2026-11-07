"""A5.11 pre-G2 companion builder on synthetic toy datasets (no data/, no HF cache, CPU)."""

import hashlib
import itertools
import json
import math

import numpy as np
import pytest

from scripts import a5_build_manifests as bm
from scripts import a5_build_panels as bp
from scripts.a5_simulation import PROXY_SOURCE, validate_inputs
from wsp.eval.response_intervention import rotation_seed
from wsp.features.response_field import registered_field_items


def toy_tokenizer():
    """Whitespace tokens with character offsets; ids are stable small integers."""
    vocab = {}

    def encode_offsets(text):
        ids, offsets, pos = [], [], 0
        for word in text.split():
            start = text.index(word, pos)
            pos = start + len(word)
            ids.append(vocab.setdefault(word, len(vocab)))
            offsets.append((start, pos))
        return ids, offsets

    return (lambda text: encode_offsets(text)[0]), encode_offsets


def toy_wikitext(n_articles=8, words=12):
    lines = [""]
    for a in range(n_articles):
        lines += [f" = Title{a} = \n", ""]
        lines += [" ".join(f"a{a}w{i}" for i in range(words)) + " \n", ""]
        # an in-article formula line that looks like a heading but has a non-empty neighbour
        if a == 2:
            lines += [" = 1 rad / s = \n", " text after formula \n", ""]
    return lines


def toy_text_groups(lines, encode, n=2, window=4):
    ids = encode("\n\n".join(lines))
    windows = [
        {
            "window_id": f"dev:{i}",
            "item_id": str(i),
            "token_start": i * window,
            "token_end": (i + 1) * window,
            "token_ids_hash": hashlib.sha256(
                json.dumps(ids[i * window : (i + 1) * window]).encode()
            ).hexdigest(),
        }
        for i in range(n)
    ]
    return {
        "windows": windows,
        "window_ids_sorted_sha256": bp.sorted_sha([w["window_id"] for w in windows]),
        "source": {"revision": "w" * 40},
    }


def test_articles_skip_formula_headings():
    arts = bp.wikitext_articles(toy_wikitext())
    assert [a["title"] for a in arts] == [f"Title{a}" for a in range(8)]
    assert arts[0]["line_start"] == 0
    assert all(a["line_end"] == b["line_start"] for a, b in itertools.pairwise(arts))


def test_wikitext_panel_is_article_disjoint_and_reproducible():
    lines = toy_wikitext()
    encode, encode_offsets = toy_tokenizer()
    tg = toy_text_groups(lines, encode, window=8)  # dev = tokens [0, 16): articles 0 and 1
    out = bp.build_wikitext(lines, encode, encode_offsets, tg, n_windows=4, window=4)
    assert out["dev_windows_reproduced"]
    assert out["dev_touched_articles"] == ["article:0", "article:5"]
    chosen = {a["doc_id"] for a in out["articles"]}
    assert not chosen & set(out["dev_touched_articles"])
    assert out["article_overlap_with_dev"] == 0 and out["token_overlap_with_dev_windows"] == 0
    assert len(out["windows"]) == 4 and out["stream_tokens"] >= 16
    assert all(set(w["articles"]) <= chosen for w in out["windows"])
    assert set(out["unreserved_articles"]).isdisjoint(chosen)
    assert out == bp.build_wikitext(lines, encode, encode_offsets, tg, n_windows=4, window=4)
    ranks = [a["rank_key"] for a in out["articles"]]
    assert ranks == sorted(ranks)


def test_groups_exclusion_nesting_and_certificate():
    texts = {"0": "Q a", "1": "q  A", "2": "b", "3": "c", "4": "d", "5": "e", "6": "f"}
    rep = bp.group_items(list(texts), lambda i: [bp.normalize(texts[i])])
    assert rep["1"] == "0"  # normalized duplicates form one group, smallest id represents it
    items, counts = bp.select_groups(rep, {"1"}, bp.FIELD_DOMAIN, lambda r: f"x:{r}", 3)
    ids = [it["item_id"] for it in items]
    assert "0" not in ids and "1" not in ids  # a scored member excludes the whole group
    assert counts == {"eligible_groups": 5, "excluded_groups": 1, "groups": 6}
    longer, _ = bp.select_groups(rep, {"1"}, bp.FIELD_DOMAIN, lambda r: f"x:{r}", 5)
    assert [it["item_id"] for it in longer][:3] == ids  # nested ranking
    cert = bp.certificate(items, rep, {"scored": ["1"]})
    assert cert["computed_disjoint"]
    bad = bp.certificate(items, rep, {"scored": [ids[0]]})
    assert not bad["computed_disjoint"] and bad["computed_checks"]["scored"]["item_overlap"] == 1
    block = bp.panel_block(items, rep, {"scored": ["1"]}, [], 4)
    assert block["short_by"] == 1 and block["item_ids_sorted_sha256"] == bp.sorted_sha(ids)


def toy_data():
    lines = toy_wikitext()
    encode, encode_offsets = toy_tokenizer()
    tg = toy_text_groups(lines, encode)
    n = 40
    data = {
        "mmlu_pro": {
            "question": [f"question {i % 30}" for i in range(n)],
            "question_id": list(range(100, 100 + n)),
            "repo": "toy/mmlu",
            "revision": "m" * 40,
            "split": "test",
            "fingerprint": "f",
            "n_rows": n,
        },
        "hellaswag": {
            "ctx": [f"ctx {i}" for i in range(n)],
            "source_id": [f"s{i // 2}" for i in range(n)],
            "repo": "toy/hs",
            "revision": "h" * 40,
            "split": "validation",
            "fingerprint": "f",
            "n_rows": n,
        },
        "rankings": {"mmlu_pro": ["0", "1"], "hellaswag": ["2"], "harmless_alpaca": ["3", "4"]},
        "legacy": {"mmlu_pro": ["5"], "hellaswag": ["7"]},
        "harmful": {
            "train": [f"harm train {i}" for i in range(405)],
            "test": [f"harm test {i}" for i in range(103)],
            "fingerprints": {"train": "a", "test": "b"},
        },
        "harmless": {
            "train": [f"ok {i}" for i in range(400)],
            "test": [f"ok {i}" for i in range(395, 695)],
            "fingerprints": {"train": "c", "test": "d"},
        },
        "wikitext": bp.build_wikitext(lines, encode, encode_offsets, tg, n_windows=2, window=4),
        "wikitext_fingerprint": "w",
    }
    cfg = {
        "tf_v1": {"over_refusal_offset": 1},
        "benchmarks": {"hellaswag": {"n": 1}, "refusal_rate": {"n": 50}, "over_refusal": {"n": 1}},
    }
    return data, cfg, tg


def test_build_panels_toy_end_to_end():
    data, cfg, tg = toy_data()
    out = bp.build_panels(data, cfg, tg, {"toy": True})
    assert out == bp.build_panels(data, cfg, tg, {"toy": True})  # deterministic
    p = out["panels"]
    assert all(v["certificate"]["computed_disjoint"] for v in p.values())
    mm = p["mmlu_gold_log_prob"]
    # 40 rows, 30 question groups; 0, 1, 5 scored -> their groups (incl. 30, 31, 35) excluded.
    assert mm["n_produced"] == 27 and mm["short_by"] == 473
    assert {"0", "1", "5", "30", "31", "35"}.isdisjoint(mm["reserved_item_ids"])
    hs = p["hellaswag_gold_char_log_likelihood"]
    assert {"2", "3", "6", "7"}.isdisjoint(hs["reserved_item_ids"])  # source groups s1, s3
    harm = p["refusal_margin_harmful"]
    # test[:100] and train[:400] are excluded: train 400-404 and test 100-102 remain.
    assert sorted(harm["item_ids"], key=bp.id_order) == [
        *(f"test:{i}" for i in range(100, 103)),
        *(f"train:{i}" for i in range(400, 405)),
    ]
    # Chair ruling (i), registered: the size exception, and the extraction fill is rejected (listed only).
    assert "conditional_fill_not_registered" not in harm
    fill = harm["extraction_fill_rejected"]
    assert fill["status"].startswith("REJECTED") and fill["item_ids"]
    assert all(i.startswith("train:") for i in fill["item_ids"])
    assert set(fill["item_ids"]).isdisjoint(harm["reserved_item_ids"])
    assert (harm["registered_size"], harm["n_requested"], harm["n_generic_clause"]) == (20, 20, 64)
    assert harm["size_rule"].startswith("REGISTERED")
    assert mm["registered_rule"].startswith("REGISTERED") and mm["field_prefix"] == 64
    harmless = p["refusal_margin_harmless"]
    # test ids 0..4 have texts 'ok 395'..'ok 399', equal to direction-extraction prompts: excluded.
    assert all(int(i) >= 5 for i in harmless["reserved_item_ids"])
    probe = p["probe_response"]
    assert set(probe["reserved_item_ids"]).isdisjoint(harmless["reserved_item_ids"])
    assert probe["n_produced"] == 64 and harmless["n_produced"] == 64
    wt = p["wikitext_nll"]
    assert wt["n_produced"] == 2 and wt["certificate"]["future_seal_obligations"]
    assert "deviation_from_a5_8_1" not in wt and wt["registered_rule"].startswith("REGISTERED")
    for v in p.values():  # future sealed panels are an obligation, never a computed certificate
        assert "by_construction" not in v["certificate"]
        assert v["certificate"]["future_seal_status"].startswith("OBLIGATION, NOT A CERTIFICATE")
    assert bp.apply_rulings(json.loads(json.dumps(out))) == out  # idempotent re-labelling


def registered_doc():
    """A panels.json shaped like the registered package (sizes only; ids are synthetic)."""
    cert = {"certificate": {"computed_disjoint": True}}
    ranked = {"item_ids": [str(i) for i in range(500)], "field_prefix": 64, "n_requested": 500}
    return {
        "panels": {
            "mmlu_gold_log_prob": ranked | cert,
            "hellaswag_gold_char_log_likelihood": dict(ranked) | cert,
            "refusal_margin_harmful": {
                "item_ids": [f"test:{i}" for i in range(20)],
                "n_requested": 64,
                "n_produced": 20,
                "conditional_fill_not_registered": {
                    "item_ids": ["train:9"],
                    "item_ids_sorted_sha256": "x",
                },
            }
            | cert,
            "refusal_margin_harmless": {"item_ids": [str(i) for i in range(64)]} | cert,
            "probe_response": {"item_ids": [str(i) for i in range(100, 164)]} | cert,
            "wikitext_nll": {"window_ids": [f"w{i}" for i in range(64)]} | cert,
        }
    }


def test_harmful_panel_size_exception_is_checked_by_consumer():
    doc = registered_doc()
    # Before the ruling labels the 20-prompt panel fails the consumer check (and the fill is active).
    with pytest.raises(ValueError, match="size exception"):
        registered_field_items(doc, "refusal_margin_harmful")
    for name in ("mmlu_gold_log_prob", "hellaswag_gold_char_log_likelihood", "wikitext_nll"):
        doc["panels"][name]["certificate"] |= {"by_construction": ["future"]}
    doc["panels"]["wikitext_nll"] |= {"document_ids": ["article:1"], "deviation_from_a5_8_1": "x"}
    bp.apply_rulings(doc)
    assert bp.validate_panels(doc) == []
    assert registered_field_items(doc, "refusal_margin_harmful") == [f"test:{i}" for i in range(20)]
    assert registered_field_items(doc, "mmlu_gold_log_prob") == [str(i) for i in range(64)]
    assert len(registered_field_items(doc, "wikitext_nll")) == 64
    # 64 harmful prompts, an active fill, a shifted prefix or a failed certificate all fail closed.
    bad = json.loads(json.dumps(doc))
    bad["panels"]["refusal_margin_harmful"]["item_ids"] += [f"train:{i}" for i in range(44)]
    with pytest.raises(ValueError, match="registered 20"):
        registered_field_items(bad, "refusal_margin_harmful")
    bad = json.loads(json.dumps(doc))
    bad["panels"]["refusal_margin_harmful"]["conditional_fill_not_registered"] = {}
    with pytest.raises(ValueError, match="rejected extraction fill"):
        registered_field_items(bad, "refusal_margin_harmful")
    bad = json.loads(json.dumps(doc))
    bad["panels"]["mmlu_gold_log_prob"]["field_prefix"] = 100
    with pytest.raises(ValueError, match="first 64 of 500"):
        registered_field_items(bad, "mmlu_gold_log_prob")
    bad["panels"]["refusal_margin_harmless"]["certificate"]["computed_disjoint"] = False
    assert any("certificate" in e for e in bp.validate_panels(bad))
    with pytest.raises(ValueError, match="not a registered"):
        registered_field_items(doc, "refusal_margin")


def test_control_seed_functions():
    for key in ("0027efdf2eae7405", bp.TEST_KEY):
        assert bp.domain_seed(bp.ROTATION_DOMAIN, key) == rotation_seed(key)
    seed = bp.domain_seed(bp.PERMUTED_FIELD_DOMAIN, bp.TEST_KEY)
    for n in range(2, 9):
        p = bp.derangement(n, seed)
        assert sorted(p) == list(range(n)) and all(p[i] != i for i in range(n))
        assert p == bp.derangement(n, seed)
    assert bp.derangement(1, seed) == [0]
    seeds = bp.build_control_seeds()
    assert [s["name"] for s in seeds["simulation"]["streams"]] == ["panels", "signs", "resolution"]
    assert seeds["permuted_field"]["domain"] == "20261107:permuted-field:"
    assert seeds["random_field"]["domain"] == "20261107:random-field:"


def test_readme_section_roundtrip():
    first = bp.merge_readme(
        "# head\n", "<!-- a5_build_panels:begin -->\nA\n<!-- a5_build_panels:end -->\n"
    )
    second = bp.merge_readme(
        first, "<!-- a5_build_panels:begin -->\nB\n<!-- a5_build_panels:end -->\n"
    )
    assert "A" not in second.split("# head")[1] and "B" in second and second.startswith("# head")
    kept = bm.keep_companion_section(second.encode(), b"# regenerated\n")
    assert kept.startswith(b"# regenerated") and b"\nB\n" in kept
    assert bm.keep_companion_section(b"# no section\n", b"# new\n") == b"# new\n"


def toy_payload():
    rows = []
    for fam, n in (("LoRA", 30), ("chain", 12)):
        for i in range(n):
            rows.append(
                {
                    "id": f"{fam}{i:02}",
                    "component": f"c{fam}{i:02}",
                    "family": fam,
                    "norm": float(i),
                }
            )
    donors = [r | {"audit": [True] * 3, "audit_evidence": ["computed"] * 3} for r in rows]
    raw = np.random.default_rng(0).normal(size=(len(rows), 64))
    prov = {
        r["id"]: {
            "edit_record_sha256": "e",
            "reference_id": "base",
            "reference_record_sha256": "r",
            "run_ids": [],
        }
        for r in rows
    }
    return {
        "inventory": rows,
        "donors": donors,
        "window_ids": [str(i) for i in range(64)],
        "donor_source": PROXY_SOURCE,
        "centred_noise": (raw - raw.mean(axis=1, keepdims=True)).tolist(),
        "window_mask": [[True] * 64 for _ in rows],
        "donor_mean": raw.mean(axis=1).tolist(),
        "provenance": {"donors": prov, "missing": {}},
    }


def test_inputs_spec_validation():
    tg = {
        "windows": [{"window_id": f"w{i}"} for i in range(64)],
        "window_ids_sorted_sha256": bp.sorted_sha([f"w{i}" for i in range(64)]),
    }
    spec = json.loads(json.dumps(bp.build_nullsize_spec(tg)))
    # The registered proxy source is accepted: no blocker, no fabricated rotation column.
    assert spec["proxy"]["allowed_sources"] == [PROXY_SOURCE]
    assert "rotation_column" not in spec and "residual_shape" not in spec["payload_schema"]
    payload = toy_payload()
    report = validate_inputs(payload, spec)
    assert report["status"] == "valid" and not report["errors"] and not report["blockers"]
    # Masked windows are NaN/null, never imputed; masked pairs are counted.
    payload["window_mask"][0][3] = False
    payload["centred_noise"][0][3] = None
    u = np.asarray([math.nan if v is None else v for v in payload["centred_noise"][0]])
    payload["centred_noise"][0] = (u - np.nanmean(u)).tolist()
    payload["centred_noise"][0][3] = None
    ok = validate_inputs(payload, spec)
    assert ok["status"] == "valid" and ok["counts"]["incomplete_pairs"] == 1
    bad = json.loads(json.dumps(payload))
    bad["centred_noise"][1][0] += 1.0  # no longer centred within the pair
    bad["centred_noise"][2][5] = None  # missing value without a mask
    bad["donors"][3]["residuals"] = [[2.0, 1.0, 0.0]] * 64  # an observed rotation column
    bad["donors"][4]["audit_evidence"] = ["assumed", "computed", "computed"]
    del bad["donors"][5]["audit"]  # audit must be explicit, never a default pass
    bad["donor_mean"] = bad["donor_mean"][:-1]
    bad["donor_source"] = "rotation"
    bad["inventory"][2]["family"] = "lora_sft"
    bad["window_ids"] = bad["window_ids"][::-1]
    report = validate_inputs(bad, spec)
    assert report["status"] == "invalid"
    text = " ".join(report["errors"])
    for needle in (
        "donor_mean shape",
        "residuals",
        "audit_evidence",
        "explicit booleans",
        "donor_source",
        "family",
        "window_ids",
    ):
        assert needle in text, needle
    bad = json.loads(json.dumps(payload))
    bad["centred_noise"][1][0] += 1.0
    bad["centred_noise"][2][5] = None
    text = " ".join(validate_inputs(bad, spec)["errors"])
    assert "not centred" in text and "finite centred noise" in text
    bad["centred_noise"] = [row[:63] for row in bad["centred_noise"]]
    assert "centred_noise shape" in " ".join(validate_inputs(bad, spec)["errors"])
    spec["proxy"]["allowed_sources"] = []
    assert validate_inputs(payload, spec)["status"] == "blocked"
