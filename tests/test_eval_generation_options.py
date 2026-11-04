import pytest

from wsp.eval.run import generation_options


def test_sampling_and_static_cache_options():
    cfg = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "cache_implementation": "static"}
    assert generation_options(cfg) == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "cache_implementation": "static",
    }


def test_greedy_omits_sampling_options():
    assert generation_options({"temperature": 0}) == {"do_sample": False}
    with pytest.raises(ValueError):
        generation_options({"temperature": -1})
