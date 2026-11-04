import torch
from p3_toy import tiny_qwen35_shapes

from wsp.features.pool import TYPES, classify, pool_features


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
    assert stats.shape == (18 * (1 + len(TYPES) * 4),)
    assert svd.shape == (8 * len(TYPES),)
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


EXPECTED = {
    "linear_attn.in_proj_qkv": "lin_qkv",
    "linear_attn.in_proj_z": "lin_z",
    "linear_attn.in_proj_b": "lin_b",
    "linear_attn.in_proj_a": "lin_a",
    "linear_attn.out_proj": "o",
    "linear_attn.A_log": "lin_state",
    "linear_attn.dt_bias": "lin_state",
    "linear_attn.conv1d.weight": "lin_state",
    "linear_attn.norm.weight": "lin_state",
    "self_attn.q_proj": "q",
    "self_attn.k_proj": "k",
    "self_attn.v_proj": "v",
    "self_attn.o_proj": "o",
    "self_attn.q_norm": "norm",
    "self_attn.k_norm": "norm",
    "mlp.gate_proj": "gate",
    "mlp.up_proj": "up",
    "mlp.down_proj": "down",
    "input_layernorm": "norm",
    "post_attention_layernorm": "norm",
}


def _assert_lands_in_per_type_pool(names, n_layers=4):
    for name in names:
        kind, _layer = classify(name)
        assert kind != "other", name
        rec = {"name": name, "stats": torch.full((18,), 7.0), "svd": torch.full((8, 129), 7.0)}
        stats, svd = pool_features([rec], n_layers=n_layers, buckets=4)
        t = TYPES.index(kind)
        assert torch.all(svd[8 * t : 8 * (t + 1)] == 7), name  # per-type spectrum pool
        typed = stats[18:].reshape(len(TYPES), 4, 18)[t]
        assert torch.all(typed.sum(0) == 7), name  # exactly one bucket of that type
        assert torch.all(stats[:18] == 7)  # and the global mean


def test_every_qwen35_hybrid_parameter_has_a_per_type_pool():
    shapes = tiny_qwen35_shapes()
    assert any("linear_attn.in_proj_qkv" in k for k in shapes)
    for name in shapes:
        if ".layers." in name:
            matches = [t for key, t in EXPECTED.items() if f".{key}" in name]
            assert len(matches) == 1 and classify(name)[0] == matches[0], name
    assert classify("model.embed_tokens.weight") == ("embed", None)
    assert classify("lm_head.weight") == ("embed", None)
    assert classify("model.norm.weight") == ("norm", None)
    assert classify("model.something.new.weight")[0] == "other"
    _assert_lands_in_per_type_pool(shapes)


def test_hybrid_only_edit_has_nonzero_spectrum_features():
    recs = [
        {"name": f"model.layers.0.linear_attn.{n}.weight", "stats": torch.ones(18)}
        | {"svd": torch.ones(8, 129)}
        for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    ]
    _stats, svd = pool_features(recs, n_layers=4)
    assert (svd.reshape(len(TYPES), 8).sum(1) > 0).sum() == 4


def _real_entry_names():
    from pathlib import Path

    import pytest

    from wsp.zoo.apply import lora_factors

    root = Path(__file__).resolve().parents[1] / "data" / "zoo" / "qwen3.5-0.8b"
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        pytest.skip("real zoo not available")
    import json

    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "id" in r]
    abl = next((r["id"] for r in rows if r["family"] == "abliteration"), None)
    lora = next(
        (r["id"] for r in rows if r["family"] == "lora_sft" and (root / r["id"] / "lora").exists()),
        None,
    )
    if abl is None or lora is None or not (root / abl / "delta.safetensors").exists():
        pytest.skip("real abliteration / lora entries not available")
    from safetensors import safe_open

    with safe_open(root / abl / "delta.safetensors", "pt") as f:
        abl_names = sorted(f.keys())
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        real_cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
        with torch.device("meta"):
            keys = list(AutoModelForCausalLM.from_config(real_cfg.text_config).state_dict())
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5-0.8B config not cached")
    return abl_names, sorted(lora_factors(root / lora / "lora", keys))


def test_real_abliteration_and_lora_changed_tensors_land_in_per_type_pools():
    abl_names, lora_names = _real_entry_names()
    assert abl_names and lora_names
    _assert_lands_in_per_type_pool(abl_names, n_layers=24)
    _assert_lands_in_per_type_pool(lora_names, n_layers=24)
