"""wm.wmm.lora_micro: micro-batched LoRA training reproduces wm.lora.train (registered recipe) on CPU."""

import torch

from wsp.wm import lora as LR
from wsp.wm.wmm import lora_micro as LM
from wsp.wm.wmm import roles as R


class Row:
    def __init__(self, ids, score_from):
        self.ids, self.score_from = ids, score_from


def _model(seed=0):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(seed)
    cfg = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=4,
        max_position_embeddings=128,
        query_pre_attn_scalar=16,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
    )
    return Gemma3ForCausalLM(cfg).float().eval()


def _rows():
    g = torch.Generator().manual_seed(1)
    out = []
    for i in range(11):  # 11 rows: the last batch of 8-row batches is short
        n = 6 + (i * 5) % 13
        ids = torch.randint(1, 64, (n,), generator=g).tolist()
        out.append(Row(ids, 2 + i % 3))
    return out


def test_micro_batched_training_matches_registered_train():
    rows = _rows()
    runs = []
    for fn in ("ref", "micro"):
        m = _model()
        rm = R.GemmaRoleMap.from_config({"model_type": "gemma3_text", **m.config.to_dict()})
        ad = LR.LoRA(m, LR.writer_targets(rm), rank=2, alpha=4, seed=3)
        kw = {
            "pad_id": 0,
            "lr": 1e-2,
            "epochs": 2,
            "batch_rows": 8,
            "seed": 5,
            "device": "cpu",
            "warmup": 2,
            "ce_chunk": 7,
        }
        log = (
            LR.train(m, ad, rows, **kw)
            if fn == "ref"
            else LM.train(m, ad, rows, micro_rows=1, **kw)
        )
        runs.append((log, ad.factors()))
    (l1, f1), (l2, f2) = runs
    assert l1["steps"] == l2["steps"] == 4
    assert torch.allclose(torch.tensor(l1["loss"]), torch.tensor(l2["loss"]), rtol=1e-4, atol=1e-6)
    for n in f1:
        assert torch.allclose(f1[n][0], f2[n][0], rtol=1e-4, atol=1e-6)
        assert torch.allclose(f1[n][1], f2[n][1], rtol=1e-4, atol=1e-6)
    assert l2["micro_rows"] == 1
