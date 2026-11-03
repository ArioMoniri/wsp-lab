from pathlib import Path

import pytest
import torch

from wsp.eval.refusal import keyword_refusal, kl_means
from wsp.eval.run import answer, apply_delta, exact_match, write_results
from wsp.eval.subsets import SPECS, subset_ids
from wsp.schema import EvalRecord


def test_committed_subsets_regenerate():
    for name, spec in SPECS.items():
        p = Path("configs/eval_subsets") / f"{name}.txt"
        assert p.read_text().splitlines() == subset_ids(spec[2], spec[3])


@pytest.mark.parametrize(
    ("text", "gold"),
    [
        ("reason 99. Thus \\boxed{42}", "42"),
        ("cost is $1,234.00", "1234"),
        ("<think>2</think>Final answer: -3.5", "-3.5"),
        ("\\boxed{\\frac{1}{2}}", "0.5"),
        ("\\boxed{(1,2)}", "(1,2)"),
    ],
)
def test_math_extraction(text, gold):
    assert exact_match(text, gold)
    assert answer(text)


def test_delta_is_added_and_unknown_tensor_rejected():
    m = torch.nn.Linear(2, 1, bias=False)
    before = m.weight.detach().clone()
    apply_delta(m, {"weight": torch.ones_like(m.weight)})
    assert torch.equal(m.weight, before + 1)
    with pytest.raises(ValueError):
        apply_delta(m, {"bad": torch.zeros(1)})


def test_summary_derived_from_records(tmp_path):
    records = [
        EvalRecord("base", "gsm8k", "0", 1, 5, True),
        EvalRecord("base", "hellaswag", "0", 0),
    ]
    summary = write_results(tmp_path, records, {"max_new_tokens": 32}, {"seed": 0})
    assert summary["gsm8k"] == 1
    assert summary["loop_rate"] == 1
    assert (tmp_path / "manifest.json").exists()


def test_refusals_and_kl():
    assert keyword_refusal("I *can’t* help")
    assert keyword_refusal("")
    assert not keyword_refusal("Here is a cake recipe")
    p = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    assert kl_means(p, p) == (0.0, 0.0)
    first, full = kl_means(p, p.flip(-1))
    assert first > 0 and full > 0


def test_ifeval_strict_all_instructions():
    from wsp.eval.run import strict_ifeval

    row = {
        "key": 1,
        "prompt": "say hello",
        "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
        "kwargs": [{"keywords": ["hello"]}, {}],
    }
    assert strict_ifeval(row, "hello world")
    assert not strict_ifeval(row, "hello, world")


def test_perplexity_aggregate_uses_token_weighted_nll():
    import math

    from wsp.schema import aggregate

    records = [
        EvalRecord("base", "wikitext_ppl", "0", math.exp(1), 2),
        EvalRecord("base", "wikitext_ppl", "1", math.exp(3), 6),
    ]
    assert aggregate(records)["wikitext_ppl"] == pytest.approx(math.exp(2.5))


def test_presence_penalty_ignores_prompt_and_counts_repeats_once():
    from wsp.eval.run import PresencePenalty

    scores = torch.zeros(1, 5)
    result = PresencePenalty(2, 1.5)(torch.tensor([[0, 1, 2, 2, 3]]), scores)
    assert result.tolist() == [[0, 0, -1.5, -1.5, 0]]
