"""Pinned TriviaQA generation pool with the registered sycophancy_real template (RULING-wme-v2 A4)."""


from wsp.wm import p2data as D
from wsp.wm import triviaqa as T


def _rows():
    mk = lambda i, v, typ, al: {
        "question": f"Question {i}?",
        "question_id": f"q{i}",
        "answer": {"value": v, "aliases": al, "type": typ, "normalized_value": v.lower()},
    }
    return [
        mk(0, "Paris", "WikipediaEntity", ["Paris", "Paris, France"]),
        mk(1, "Rome", "WikipediaEntity", ["Rome"]),
        mk(2, "paris", "WikipediaEntity", ["paris"]),
        mk(3, "42", "Numerical", ["42", "forty-two"]),
        mk(4, "7", "Numerical", ["7", "seven"]),
    ]


def test_pins():
    assert len(T.PIN["revision"]) == 40 and set(T.PIN["files"]) == {"train", "validation"}
    for f in T.PIN["files"].values():
        assert len(f["sha256"]) == 64


def test_items_use_the_registered_template_and_incorrect_rule():
    its = T.items_from_rows(_rows())
    assert len(its) == 5
    by = {i["question_id"]: i for i in its}
    for it in its:
        assert it["neutral_prompt"] == D.NEUTRAL_TEMPLATE.format(question=it["question"])
        assert it["suggested_prompt"] == D.SUGGESTED_TEMPLATE.format(
            question=it["question"], incorrect_answer=it["incorrect_answer"]
        )
        assert T.norm(it["incorrect_answer"]) not in {T.norm(a) for a in it["aliases"]}
    assert (
        by["q0"]["incorrect_answer"] == "Rome"
    )  # same type; "paris" is an alias of q0 and is skipped
    assert by["q3"]["incorrect_answer"] == "7" and by["q4"]["incorrect_answer"] == "42"
    assert T.items_from_rows(_rows()) == its  # deterministic


def test_no_incorrect_answer_available_is_dropped():
    rows = _rows()[:1]
    assert T.items_from_rows(rows) == []
