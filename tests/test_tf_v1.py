import math
from types import SimpleNamespace

import pytest
import torch

from wsp.eval.tf_v1 import choice_record, frozen_ids, summarize, token_mass_margin
from wsp.schema import EvalRecord


def test_fixed_ids_exclude_legacy_and_are_reproducible():
    ids = frozen_ids("mmlu_pro", 1000)
    from wsp.eval.subsets import ROOT

    assert len(set(ids)) == 1000
    assert not set(ids) & set((ROOT / "mmlu_pro.txt").read_text().splitlines())
    assert ids == frozen_ids("mmlu_pro", 1000)


def test_gold_letter_is_normalized_over_options_and_margin_is_paired():
    row = choice_record("base", "mmlu_pro", "17", [-3.0, -2.0, -4.0], 0)
    assert row.extra["gold_log_prob"] == pytest.approx(
        -3 - torch.logsumexp(torch.tensor([-3.0, -2.0, -4.0]), 0).item()
    )
    assert row.extra["margin"] == -1
    assert row.correct == 0
    assert len(row.extra["option_log_probs"]) == 3
    result = summarize([row])
    assert result["mmlu_pro"]["mean_gold_log_prob"] == row.extra["gold_log_prob"]
    assert result["mmlu_pro"]["accuracy"] == 0


def test_hellaswag_preserves_character_normalized_likelihoods():
    row = choice_record("base", "hellaswag", "8", [-0.3, -0.8], 1)
    assert row.extra["gold_log_prob"] == -0.8
    assert row.extra["margin"] == pytest.approx(-0.5)
    assert row.extra["ending_normalized_log_likelihoods"] == [-0.3, -0.8]


def test_token_weighted_log_ppl_and_refusal_mass():
    rows = [
        EvalRecord("base", "wikitext_ppl", str(i), math.exp(x), n, extra={"mean_nll": x})
        for i, (x, n) in enumerate([(1.0, 2), (3.0, 6)])
    ]
    assert summarize(rows)["wikitext"]["log_ppl"] == 2.5
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert token_mass_margin(logits, [0, 1], [2, 3]) == pytest.approx(-2.0)
    with pytest.raises(ValueError):
        token_mass_margin(logits, [0], [0])


def test_tf_dispatch_never_calls_sampling(monkeypatch):
    from wsp.eval import run, tf_v1

    runner = SimpleNamespace(generation={"tier": "tf_v1"})
    monkeypatch.setattr(tf_v1, "evaluate", lambda *args: ([], {"tf": True}))
    assert run.evaluate(runner, ["mmlu_pro"], {}) == ([], {"tf": True})


def test_base_cache_is_greedy_reused_and_detects_corruption(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from wsp.eval import batch, tf_v1

    class Model:
        device = "cpu"
        calls = 0

        def generate(self, ids, **kwargs):
            assert kwargs["do_sample"] is False
            self.calls += 1
            return torch.tensor([[0, 1, 2, 3]])

    runner = SimpleNamespace(
        model=Model(),
        revision="fixed",
        generation={"batch_size": 1},
        tokenizer=SimpleNamespace(encode=lambda *a, **k: [0, 1], pad_token_id=0),
        chat=lambda x: x,
        base_context=nullcontext,
        tf_cache_root=tmp_path,
        tf_config={"generation": {"batch_size": 1}, "benchmarks": {}},
    )
    monkeypatch.setattr(tf_v1, "runtime_metadata", lambda _: {"eval_device_id": "H0"})
    monkeypatch.setattr(batch, "provenance", lambda: {"source_hash": "fixed"})
    monkeypatch.setattr(tf_v1, "last_logits", lambda *a: torch.zeros(3, 5))
    root, meta = tf_v1.cache_kl(runner, ["prompt"], {"revision": "pinned"})
    assert runner.model.calls == 1
    assert tf_v1.cache_kl(runner, ["prompt"], {"revision": "pinned"}) == (root, meta)
    assert runner.model.calls == 1
    data = torch.load(root / "0.pt", weights_only=True)
    assert data["prefix_length"] == 2
    assert data["logits"].shape == (2, 5)
    (root / "0.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cache hash mismatch"):
        tf_v1.cache_kl(runner, ["prompt"], {"revision": "pinned"})


def test_results_embed_tf_metrics_and_item_hashes_without_keyword_aliases(tmp_path):
    import json

    from wsp.eval.run import write_results

    row = choice_record("base", "mmlu_pro", "17", [-3.0, -2.0], 0)
    refusal = EvalRecord("base", "refusal_rate", "0", -2.0)
    result = write_results(tmp_path, [row, refusal], {"tier": "tf_v1"}, {"seed": 20261105})
    assert result["tf_v1"]["refusal_rate"]["mean_first_token_margin"] == -2.0
    assert "refusal_keyword_rate" not in result
    assert "refusal_rate" not in result
    assert len(result["item_id_hash"]) == 64
    assert json.loads((tmp_path / "records.jsonl").read_text().splitlines()[0])["item_id"] == "17"


def test_tf_prompt_source_reads_full_pinned_split_not_legacy_100(monkeypatch):
    import datasets

    from wsp.eval.tf_v1 import load_prompts

    class Dataset:
        _fingerprint = "fixed"

        def __getitem__(self, column):
            return [f"prompt {i}" for i in range(250)]

    def load(repo, **kwargs):
        assert kwargs["split"] == "test"
        assert len(kwargs["revision"]) == 40
        return Dataset()

    monkeypatch.setattr(datasets, "load_dataset", load)
    texts, meta = load_prompts("kl_to_base", 200)
    assert len(texts) == 200
    assert meta["subset_ids"] == [str(i) for i in range(200)]
