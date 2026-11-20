"""Study M Gate 0b harness (RULING-wmm-gate0 M1): prompt, extractor precedence, fresh designation, pass rule."""

import pytest

from wsp.wm import wmm_gate0 as W
from wsp.wm import wmm_gate0b as B

ITEM = {
    "id": "x",
    "source": "medqa",
    "question": "Which drug?",
    "options": ["aa", "bb", "cc", "dd"],
    "answer_index": 2,
    "hash": "h",
}


# ------------------------------------------------------------------ prompt


def test_prompt_uses_the_card_format_instruction():
    p = B.prompt(ITEM)
    assert p.startswith("Which drug?\n\nOptions:\n(A) aa\n(B) bb\n(C) cc\n(D) dd\n\n")
    assert p.endswith(B.INSTRUCTION)
    assert "Final Answer: (X)" in B.INSTRUCTION and "step by step" in B.INSTRUCTION
    assert "The answer is" not in p


# ------------------------------------------------------------------ extractor: accepted forms


@pytest.mark.parametrize(
    "text,letter",
    [
        ("reasoning...\nFinal Answer: (C)", "C"),
        ("reasoning...\nFinal Answer: C", "C"),
        ("reasoning...\n**Final Answer: (B)**", "B"),
        ("reasoning...\nFinal Answer: **(D)**", "D"),
        ("final answer: (a)", "A"),
        ("Final Answer: (A)\n\nWait. Final Answer: (B)", "B"),  # last Final Answer wins
        ("Final Answer: (C) Ceftriaxone", "C"),
        (
            "Final Answer: The final answer is $\\boxed{D}$",
            "D",
        ),  # rule 2 (no letter after the colon)
        ("so it is\n$\\boxed{B}$", "B"),
        ("so it is \\boxed{(A)}.", "A"),
        ("so it is \\boxed{ C }\n", "C"),
        ("lots of text\nC", "C"),
        ("lots of text\n(D)", "D"),
        ("lots of text\n**B**", "B"),
        ("lots of text\nA.", "A"),
    ],
)
def test_accepted_forms(text, letter):
    assert B.extract(text, n_options=4) == letter


@pytest.mark.parametrize(
    "text",
    [
        "The answer is (C).",  # old template phrase: not accepted
        "Answer: C",
        "The correct option is **(B)**",
        "\\boxed{B} is my guess, but on reflection the drug is cc.",  # boxed not trailing
        "lots of text\nThe answer is C",  # final line is not letter-only
        "Final Answer: (E)",  # outside the options
        "lots of text\nE",
        "",
        "Final Answer: Ceftriaxone",
        "Final Answer: a drug that blocks cell wall synthesis",
        "lots of text\nb",
    ],
)
def test_rejected_forms(text):
    assert B.extract(text, n_options=4) is None


def test_precedence_final_answer_beats_boxed_and_last_line():
    t = "Final Answer: (A)\n\\boxed{B}"
    assert B.extract(t, 4) == "A"
    t2 = "Earlier \\boxed{A} text.\nB"
    assert B.extract(t2, 4) == "B"  # boxed is not trailing; last line holds only B
    t3 = "thinking\n\\boxed{C}"
    assert B.extract(t3, 4) == "C"


def test_rule_used_is_reported():
    assert B.extract_rule("x\nFinal Answer: (A)", 4) == ("A", "final_answer")
    assert B.extract_rule("x $\\boxed{B}$", 4) == ("B", "boxed")
    assert B.extract_rule("x\nC", 4) == ("C", "letter_line")
    assert B.extract_rule("x", 4) == (None, None)


def test_scoring_counts_truncation_as_incorrect():
    assert B.score(ITEM, "Final Answer: (C)", False)["correct"]
    s = B.score(ITEM, "Final Answer: (C)", True)
    assert not s["correct"] and s["truncated"] and s["letter"] is None
    assert not B.score(ITEM, "Final Answer: (B)", False)["correct"]


# ------------------------------------------------------------------ designation


def _items(n, src="medqa"):
    return [
        {
            "id": f"{src}{i}",
            "source": src,
            "question": f"q{i}",
            "options": ["a", "b", "c", "d"],
            "answer_index": 0,
            "hash": W.text_hash(f"{src}{i}"),
        }
        for i in range(n)
    ]


def test_fresh_designation_excludes_exposed_and_is_salted():
    items = _items(50)
    exposed = {"hashes": {items[i]["hash"] for i in range(0, 50, 2)}, "ids": set()}
    got = B.designate_fresh(items, "medqa", "dev", 10, exposed)
    assert len(got) == 10
    assert not {g["hash"] for g in got} & exposed["hashes"]
    again = B.designate_fresh(list(reversed(items)), "medqa", "dev", 10, exposed)
    assert [g["id"] for g in got] == [g["id"] for g in again]
    by_id = {"hashes": set(), "ids": {("medqa", items[1]["id"])}}
    assert items[1]["id"] not in {
        g["id"] for g in B.designate_fresh(items, "medqa", "dev", 49, by_id)
    }


def test_fresh_designation_raises_when_too_few_and_on_overlap():
    items = _items(5)
    with pytest.raises(ValueError):
        B.designate_fresh(items, "medqa", "dev", 6, {"hashes": set(), "ids": set()})
    with pytest.raises(AssertionError):
        B.assert_disjoint(items[:2], {"hashes": {items[0]["hash"]}, "ids": set()})


def test_exposed_set_reader(tmp_path):
    import json

    p = tmp_path / "exposure.json"
    p.write_text(
        json.dumps({"items": [{"id": "a", "hash": "h1", "source": "medqa", "split": "dev"}]})
    )
    ex = B.read_exposed(p)
    assert ex == {"hashes": {"h1"}, "ids": {("medqa", "a")}, "n": 1}


# ------------------------------------------------------------------ pass rule


def test_pass_rule():
    assert B.verdict(257, 400, 20, card=69.1)["pass"]  # 64.25 %, 5.0 %
    v = B.verdict(256, 400, 20, card=69.1)  # 64.0 %
    assert not v["pass"] and not v["accuracy_ok"] and v["truncation_ok"]
    v = B.verdict(300, 400, 21, card=69.1)  # 5.25 % truncated
    assert not v["pass"] and v["accuracy_ok"] and not v["truncation_ok"]
    assert B.verdict(220, 400, 0, card=59.8)["pass"]  # 55.0 >= 54.8
    assert not B.verdict(219, 400, 0, card=59.8)["pass"]  # 54.75
    assert B.CARDS == {"medqa": 69.1, "medmcqa": 59.8}
