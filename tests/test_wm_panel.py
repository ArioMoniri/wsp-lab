"""wm.panel: lean teacher-forced panel and disjoint hashed construction/validation/dev/test groups (D5, D6, D8)."""

import json
import math

import pytest
import torch

from wsp.wm import panel as P


class ToyTok:
    """Character tokenizer with a chat template that records its kwargs."""

    vocab_size = 128
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.template_calls = []

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % 120) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        self.template_calls.append(kw)
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + ("<a>" if add_generation_prompt else "")


def toy_model():
    from transformers.models.qwen3_5 import configuration_qwen3_5 as c
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    from wsp.wm.hf import use_reference_kernels

    use_reference_kernels()
    cfg = c.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return q.Qwen3_5ForCausalLM(cfg).float().eval()


@pytest.fixture(scope="module")
def model():
    return toy_model()


def _raw():
    wiki = [{"article": f"a{i}", "text": f"Article {i} text " * 30} for i in range(12)]
    gsm = [
        {"question": f"What is {i}+{i}?", "answer": f"{i} plus {i}\n#### {2 * i}"}
        for i in range(30)
    ]
    gsm.append(
        {"question": "WHAT is 3+3?  ", "answer": "dup\n#### 6"}
    )  # normalised duplicate of item 3
    mmlu = [
        {
            "question_id": i,
            "question": f"Q{i}?",
            "options": [f"o{j}" for j in range(4)],
            "answer_index": i % 4,
        }
        for i in range(40)
    ]
    harmful = [f"harmful request {i}" for i in range(20)]
    harmless = [f"harmless request {i}" for i in range(20)]
    return {
        "wikitext": wiki,
        "gsm8k": gsm,
        "mmlu_pro": mmlu,
        "harmful": harmful,
        "harmless": harmless,
    }


SIZES = {"test": 3, "dev": 3, "validation": 3, "construction": 2}


@pytest.fixture
def built(tmp_path):
    return P.build_panel(_raw(), SIZES, salt="wme-test", out_dir=tmp_path), tmp_path


def test_splits_are_disjoint_hashed_and_deduplicated(built):
    man, _root = built
    seen = {}
    for obj, splits in man["objectives"].items():
        for split, rows in splits.items():
            for h in rows["hashes"]:
                assert h not in seen, (obj, split, seen.get(h))
                seen[h] = (obj, split)
    gsm_hashes = [h for s in man["objectives"]["gsm8k_gold_lp"].values() for h in s["hashes"]]
    assert len(gsm_hashes) == len(set(gsm_hashes))
    assert man["dedup"]["gsm8k"] >= 1


def test_mmlu_and_sycophancy_items_never_share_a_question(built):
    man, _ = built
    a = {h for s in man["objectives"]["mmlupro_gold_lp"].values() for h in s["hashes"]}
    b = {h for s in man["objectives"]["sycophancy_gap"].values() for h in s["hashes"]}
    assert not a & b


def test_wikitext_windows_from_one_article_stay_in_one_split(built):
    _man, root = built
    by_article = {}
    for split in ("construction", "validation", "dev"):
        for row in P.read_open(root, split)["wikitext_nll"]:
            by_article.setdefault(row["group"], set()).add(split)
    assert all(len(v) == 1 for v in by_article.values())


def test_assignment_is_salted_and_reproducible(tmp_path):
    a = P.build_panel(_raw(), SIZES, salt="s1", out_dir=tmp_path / "a")
    b = P.build_panel(_raw(), SIZES, salt="s1", out_dir=tmp_path / "b")
    c = P.build_panel(_raw(), SIZES, salt="s2", out_dir=tmp_path / "c")
    assert a["content_hash"] == b["content_hash"] != c["content_hash"]


def test_test_items_are_written_only_under_sealed(built):
    man, root = built
    assert (root / "sealed_test" / "test.jsonl").exists()
    assert not (root / "open" / "test.jsonl").exists()
    for f in (root / "open").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            assert json.loads(line)["hash"] not in set(man["sealed_hashes"])


def test_chat_prompts_disable_thinking(model, built):
    _man, root = built
    tok = ToyTok()
    items = P.read_open(root, "dev")
    P.evaluate(P.LMAdapter(model), tok, {"gsm8k_gold_lp": items["gsm8k_gold_lp"]}, P.PanelConfig())
    assert tok.template_calls and all(k.get("enable_thinking") is False for k in tok.template_calls)


def _manual_lp(model, ids, targets_from):
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float()
    lp = logits.log_softmax(-1)
    return sum(lp[k - 1, ids[k]].item() for k in range(targets_from, len(ids)))


def test_wikitext_nll_matches_manual_teacher_forcing(model):
    tok = ToyTok()
    row = {"id": "w", "group": "a", "kind": "wikitext", "text": "hello world " * 4, "hash": "h"}
    cfg = P.PanelConfig(window_tokens=16)
    out = P.evaluate(P.LMAdapter(model), tok, {"wikitext_nll": [row]}, cfg)["wikitext_nll"][0]
    ids = tok.encode(row["text"])[:16]
    assert out["n_tokens"] == 15
    assert out["value"] == pytest.approx(-_manual_lp(model, ids, 1) / 15, rel=1e-4)


def test_gold_lp_and_refusal_margin_definitions(model):
    tok = ToyTok()
    cfg = P.PanelConfig(refusal_phrases={"refusal": ["No", "Sorry"], "compliance": ["Yes", "Sure"]})
    g = {"id": "g", "kind": "gsm8k", "question": "2+2?", "gold": "4", "hash": "x", "group": "g"}
    out = P.evaluate(P.LMAdapter(model), tok, {"gsm8k_gold_lp": [g]}, cfg)["gsm8k_gold_lp"][0]
    prefix = tok.encode(P.chat_prompt(tok, P.gsm8k_user(g)) + P.GSM8K_ANSWER_PREFIX)
    ids = prefix + tok.encode("4")
    assert out["value"] == pytest.approx(_manual_lp(model, ids, len(prefix)), rel=1e-4, abs=1e-5)
    h = {"id": "r", "kind": "harmful", "prompt": "do x", "hash": "y", "group": "r"}
    r = P.evaluate(P.LMAdapter(model), tok, {"refusal_margin_harmful": [h]}, cfg)[
        "refusal_margin_harmful"
    ][0]
    pre = tok.encode(P.chat_prompt(tok, "do x"))
    s = {
        p: _manual_lp(model, pre + tok.encode(p), len(pre)) for p in ["No", "Sorry", "Yes", "Sure"]
    }
    want = math.log(math.exp(s["No"]) + math.exp(s["Sorry"])) - math.log(
        math.exp(s["Yes"]) + math.exp(s["Sure"])
    )
    assert r["value"] == pytest.approx(want, rel=1e-4, abs=1e-5)


def test_mmlu_letter_lp_is_normalised_over_option_letters(model):
    tok = ToyTok()
    m = {
        "id": "m",
        "kind": "mmlu_pro",
        "question": "Q?",
        "options": ["a", "b", "c"],
        "answer_index": 1,
        "hash": "z",
        "group": "m",
    }
    out = P.evaluate(P.LMAdapter(model), tok, {"mmlupro_gold_lp": [m]}, P.PanelConfig())[
        "mmlupro_gold_lp"
    ][0]
    assert out["value"] <= 0.0
    s = P.evaluate(
        P.LMAdapter(model), tok, {"sycophancy_gap": [dict(m, wrong_index=2)]}, P.PanelConfig()
    )
    assert "neutral" in s["sycophancy_gap"][0] and "suggested" in s["sycophancy_gap"][0]
    assert s["sycophancy_gap"][0]["value"] == pytest.approx(
        s["sycophancy_gap"][0]["neutral"] - s["sycophancy_gap"][0]["suggested"]
    )


def test_batched_evaluation_matches_single_item_and_repeats_bitwise(model, built):
    _man, root = built
    tok = ToyTok()
    items = P.read_open(root, "validation")
    cfg = P.PanelConfig(window_tokens=24, max_batch_tokens=4096)
    a = P.evaluate(P.LMAdapter(model), tok, items, cfg)
    b = P.evaluate(P.LMAdapter(model), tok, items, cfg)
    assert P.digest(a) == P.digest(b)
    one = P.evaluate(
        P.LMAdapter(model), tok, items, P.PanelConfig(window_tokens=24, max_batch_tokens=1)
    )
    for obj in a:
        for x, y in zip(a[obj], one[obj], strict=True):
            assert x["id"] == y["id"] and x["value"] == pytest.approx(
                y["value"], rel=1e-4, abs=1e-5
            )


def test_paired_summary_uses_item_pairing():
    base = {"o": [{"id": str(i), "value": float(i)} for i in range(10)]}
    edit = {"o": [{"id": str(i), "value": float(i) + 0.1 * (i % 2)} for i in range(10)]}
    s = P.paired(base, edit)["o"]
    assert s["n"] == 10 and s["mean"] == pytest.approx(0.05)
    assert s["se"] == pytest.approx(
        torch.tensor([0.1 * (i % 2) for i in range(10)]).std().item() / math.sqrt(10)
    )
    with pytest.raises(ValueError):
        P.paired(base, {"o": edit["o"][:-1]})
