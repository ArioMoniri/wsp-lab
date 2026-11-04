import torch

from wsp.features.pool import pool_features


def test_pool_is_fixed_size_and_averages_type_bucket():
    records = [
        {
            "name": "model.layers.0.self_attn.q_proj.weight",
            "stats": torch.ones(18),
            "svd": torch.ones(8, 129),
        },
        {
            "name": "model.layers.1.self_attn.q_proj.weight",
            "stats": torch.full((18,), 3.0),
            "svd": torch.full((8, 129), 3.0),
        },
    ]
    stats, svd = pool_features(records, n_layers=8, buckets=4)
    assert stats.shape == (18 * 29,)
    assert svd.shape == (8 * 7,)
    assert torch.all(stats[:18] == 2)
    assert torch.all(svd[:8] == 2)


def test_linear_attention_out_proj_pools_with_attention_output():
    rec = {
        "name": "model.layers.0.linear_attn.out_proj.weight",
        "stats": torch.full((18,), 5.0),
        "svd": torch.full((8, 129), 5.0),
    }
    stats, svd = pool_features([rec], n_layers=8, buckets=4)
    o = 1 + 3 * 4  # global, then q, k, v, o groups of four buckets
    assert torch.all(stats[18 * o : 18 * (o + 1)] == 5)
    assert torch.all(svd[8 * 3 : 8 * 4] == 5)
