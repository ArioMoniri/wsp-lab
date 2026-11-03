"""Small CUDA determinism contract; short cap bounds the smoke test to three minutes."""

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.timeout(180)
def test_gsm8k_eight_items_deterministic(cuda):
    from wsp.eval.run import HFRunner, evaluate

    cfg = {
        "seed": 0,
        "max_new_tokens": 16,
        "batch_size": 8,
        "thinking": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
    }
    runner = HFRunner("Qwen/Qwen3.5-0.8B", cfg)
    first, _ = evaluate(runner, ["gsm8k"], {"gsm8k": {"n": 8}})
    second, _ = evaluate(runner, ["gsm8k"], {"gsm8k": {"n": 8}})
    assert first == second
    assert len(first) == 8
    assert all(r.n_tokens <= 16 and r.extra["output"] and r.item_id for r in first)
