"""Padding/stop accounting regressions plus an opt-in real-model GPU comparison."""

from types import SimpleNamespace

import pytest
import torch

from wsp.eval.run import HFRunner, PresencePenalty


def test_presence_penalty_ignores_left_padding_and_prompt():
    ids = torch.tensor([[0, 0, 3, 4, 2, 2], [3, 4, 5, 6, 1, 2]])
    scores = PresencePenalty(4, 1.5)(ids, torch.zeros(2, 8))
    assert scores[0].tolist() == [0, 0, -1.5, 0, 0, 0, 0, 0]
    assert scores[1].tolist() == [0, -1.5, -1.5, 0, 0, 0, 0, 0]


class Tokens(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return self


class Tokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        rows = [[3] * len(t) for t in texts]
        width = max(map(len, rows))
        return Tokens(
            input_ids=torch.tensor([[0] * (width - len(r)) + r for r in rows]),
            attention_mask=torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows]),
        )

    def decode(self, seq, **kwargs):
        return str([t for t in seq if t not in (0, 9)])


class Model:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=[9, 8])

    def generate(self, input_ids, attention_mask, **kwargs):
        # Continuation depends on unpadded context; EOS and padding must not count as content.
        lengths = attention_mask.sum(1)
        rows = [[int(n), 9, 0] if n == 1 else [4, 5, 6] for n in lengths]
        return torch.cat([input_ids, torch.tensor(rows)], dim=1)


def test_hf_batch_padding_crop_eos_and_partial_batch():
    runner = HFRunner.__new__(HFRunner)
    runner.tokenizer, runner.model = Tokenizer(), Model()
    runner.chat = lambda x: x
    runner.generation = {
        "seed": 0,
        "temperature": 0,
        "presence_penalty": 1.5,
        "max_new_tokens": 3,
        "batch_size": 2,
    }
    prompts = ["a", "abcd", "xy"]
    batched = runner.generate(prompts)
    runner.generation["batch_size"] = 1
    assert runner.generate(prompts) == batched
    assert batched == [("[1]", 2, False), ("[4, 5, 6]", 3, True), ("[4, 5, 6]", 3, True)]
    assert runner.generation_stats["tokens"] == 16


@pytest.mark.gpu
@pytest.mark.slow
def test_real_qwen_greedy_padding_and_static_cache(monkeypatch):
    import os
    from pathlib import Path

    import yaml

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = yaml.safe_load(Path("configs/eval_fast.yaml").read_text())["generation"]
    cfg.update(temperature=0, max_new_tokens=32, batch_size=1)
    runner = HFRunner(os.environ.get("WSP_PERF_MODEL", "Qwen/Qwen3.5-0.8B"), cfg)
    prompts = ["What is 2+2?", "Answer with one word: the capital city of France is?"]
    baseline = runner.generate(prompts)
    cfg.update(batch_size=2, cache_implementation="static")
    assert runner.generate(prompts) == baseline


def test_choice_padding_matches_scalar_scoring():
    class ChoiceModel:
        device = torch.device("cpu")

        def __call__(self, ids, attention_mask):
            # Context-sensitive logits independent of left padding.
            values = (ids * attention_mask).cumsum(1).float()
            logits = values[..., None] * torch.arange(16).float() / 100
            return SimpleNamespace(logits=logits)

    runner = HFRunner.__new__(HFRunner)
    runner.model = ChoiceModel()
    runner.tokenizer = SimpleNamespace(
        pad_token_id=0, encode=lambda s, **kwargs: [ord(c) % 15 + 1 for c in s]
    )
    runner.generation = {"choice_batch_size": 1}
    choices = ["a", "longer answer", "xy"]
    for normalized in (False, True):
        runner.generation["choice_batch_size"] = 1
        scalar = runner.choice_scores("context", choices, normalized)
        runner.generation["choice_batch_size"] = 2
        assert runner.choice_scores("context", choices, normalized) == pytest.approx(scalar)
