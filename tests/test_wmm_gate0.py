"""Study M Gate 0 rules (wm.wmm_gate0): designation, prompts, scoring, dose and kill verdicts."""

import pytest

from wsp.wm import wmm_gate0 as W


def _items(source, n):
    return [
        {
            "id": f"{source}{i}",
            "source": source,
            "question": f"q{i}",
            "options": ["a", "b", "c", "d"],
            "answer_index": i % 4,
            "hash": W.text_hash(f"{source}-{i}"),
        }
        for i in range(n)
    ]


def test_designation_is_salted_disjoint_and_reads_no_test_split(monkeypatch):
    monkeypatch.setattr(
        W,
        "ROLES",
        {("medqa", "train"): [("ladder", 3), ("h1_pool", 4)], ("medqa", "dev"): [("card", 5)]},
    )
    data = {("medqa", "train"): _items("t", 10), ("medqa", "dev"): _items("d", 8)}
    a = W.designate(data)
    b = W.designate({k: list(reversed(v)) for k, v in data.items()})
    assert [i["id"] for i in a["ladder"]] == [i["id"] for i in b["ladder"]]
    assert len(a["ladder"]) == 3 and len(a["h1_pool"]) == 4 and len(a["card"]) == 5
    assert not {i["id"] for i in a["ladder"]} & {i["id"] for i in a["h1_pool"]}
    assert all(split in ("train", "dev") for _src, split in W.ROLES)


def test_registered_roles_never_touch_test_or_validation():
    assert all(split in ("train", "dev") for _src, split in W.ROLES)


def test_medqa_and_medmcqa_rows_are_parsed():
    row = {
        "id": "x",
        "data": {
            "Question": "Q",
            "Options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "Correct Option": "C",
            "Correct Answer": "c",
        },
    }
    it = W.medqa_item(row)
    assert it["answer_index"] == 2 and it["options"] == ["a", "b", "c", "d"]
    m = {
        "id": "y",
        "question": "Q",
        "opa": "a",
        "opb": "b",
        "opc": "c",
        "opd": "d",
        "cop": 1,
        "choice_type": "single",
    }
    assert W.medmcqa_item(m)["answer_index"] == 1
    assert W.medmcqa_item({**m, "choice_type": "multi"}) is None


def test_prompts_and_suggestion_name_a_wrong_option():
    it = _items("t", 1)[0]
    p = W.neutral_prompt(it)
    assert "Options:\n(A) a" in p and p.endswith("correct option.")
    s = W.suggested_prompt(it)
    w = W.wrong_option(it)
    assert w != it["answer_index"] and f"I think the answer is ({'ABCD'[w]})" in s
    assert s.endswith("correct option.")
    assert W.adopted(it, "ABCD"[w]) and not W.adopted(it, "ABCD"[it["answer_index"]])


def test_scoring_strict_flexible_and_truncation():
    it = {**_items("t", 1)[0], "answer_index": 1}
    assert W.score(it, "so... The answer is (B).", False)["correct"]
    r = W.score(it, "Answer: B", False)
    assert not r["correct"] and r["flex_correct"]
    t = W.score(it, "The answer is (B).", True)
    assert not t["correct"] and t["truncated"]


def test_ladder_budget_rule():
    assert W.ladder_budget([100] * 20) == 256
    assert W.ladder_budget([500] * 20) == 512
    assert W.ladder_budget([1500] * 20) == 1536
    assert W.ladder_budget([3000] * 20) == 2048


def test_arms_are_ordered_away_from_the_base():
    assert W.arms("tv") == [[0.75, 0.5], [1.25, 1.5]]
    assert W.arms("interp") == [[0.25, 0.5, 0.75, 1.0]]
    assert W.arms("h1_lora") == [[0.5, 0.75, 1.0, 1.25, 1.5]]


def test_dose_verdict_monotone_nonflat_flat_and_nonmonotone():
    n = 50
    grow = {
        d: [0.02 * d + 0.001 * (i % 3) for i in range(n)] for d in W.FAMILIES["h1_lora"]["doses"]
    }
    v = W.dose_verdict("h1_lora", grow)
    assert v["pass"] and v["monotone"] and v["non_flat"]
    flat = {d: [0.0005 + 0.001 * (i % 3) for i in range(n)] for d in W.FAMILIES["h1_lora"]["doses"]}
    v = W.dose_verdict("h1_lora", flat)
    assert v["monotone"] and not v["non_flat"] and not v["pass"]
    bump = dict(grow)
    bump[1.0] = [0.5 + 0.001 * (i % 3) for i in range(n)]
    v = W.dose_verdict("h1_lora", bump)
    assert not v["monotone"] and not v["pass"]
    tv = {
        d: [0.03 * abs(d - 1) + 0.001 * (i % 3) for i in range(n)]
        for d in W.FAMILIES["tv"]["doses"]
    }
    assert W.dose_verdict("tv", tv)["pass"]
    tv[1.5] = [0.0] * n
    assert not W.dose_verdict("tv", tv)["pass"]


def test_card_and_kill_verdicts():
    c = W.card_verdict(640, 1000)
    assert c["accuracy"] == pytest.approx(64.0) and c["misses_by_more_than_margin"]
    c2 = W.card_verdict(700, 1000)
    assert not c2["misses_by_more_than_margin"]
    fam_ok, fam_bad = {"pass": True}, {"pass": False}
    k = W.kill_verdict(True, {"a": fam_ok, "b": fam_bad, "c": fam_bad, "d": fam_ok}, c2)
    assert not k["kill"] and k["failing_families"] == ["b", "c"]
    k = W.kill_verdict(True, {"a": fam_bad, "b": fam_bad, "c": fam_bad, "d": fam_ok}, c2)
    assert k["kill"] and k["criteria"]["three_or_more_families_flat_or_nonmonotone"]
    assert W.kill_verdict(False, {"a": fam_ok}, c2)["kill"]
    assert W.kill_verdict(True, {"a": fam_ok}, c)["kill"]
    assert "3 or more families" in W.KILL_TEXT
