"""27B engineering calibration helpers (SYNTHESIS-r9 C9; round-9 L1-L4). CPU only; no model weights."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from wsp.wm import actions as A
from wsp.wm import calib27 as K
from wsp.wm import hf

FIX = Path(__file__).parent / "fixtures" / "qwen3_8_27b"
CFG = json.loads((FIX / "config.json").read_text())
NAMES = json.loads((FIX / "weight_names.json").read_text())


def test_target_revision_is_pinned():
    assert K.REPO == "Qwen/Qwen3.8-27B"
    assert hf.TARGETS[K.REPO] == "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
    assert NAMES["revision"] == hf.TARGETS[K.REPO]


def test_role_map_paths_derive_from_27b_config_and_exist():
    rm = A.RoleMap.from_config(CFG, prefix=K.PREFIX)
    assert (rm.n_layers, rm.hidden, rm.ffn, rm.vocab, rm.tied) == (64, 5120, 17408, 248320, False)
    names = set(NAMES["names"])
    for (layer, role), name in rm.writers().items():
        assert name in names, (layer, role, name)
    for i in range(rm.n_layers):
        roles = ("up", "gate", "down", "o")
        roles += ("q", "k", "v") if rm.layer_types[i] == "full_attention" else ("lin_qkv", "lin_z")
        for r in roles:
            assert rm.name(i, r) in names
    assert [i for i, t in enumerate(rm.layer_types) if t == "full_attention"] == list(
        range(3, 64, 4)
    )


def test_vision_and_mtp_are_frozen_text_writers_are_not():
    frozen = [n for n in NAMES["names"] if A.is_frozen(n)]
    assert frozen and all(n.startswith(("model.visual.", "mtp.")) for n in frozen)
    assert all(
        not n.startswith(("model.visual.", "mtp.")) for n in NAMES["names"] if not A.is_frozen(n)
    )
    rm = A.RoleMap.from_config(CFG, prefix=K.PREFIX)
    assert not any(A.is_frozen(n) for n in rm.writers().values())


def test_support_group_mirrors_the_9b_layout():
    rm = A.RoleMap.from_config(CFG, prefix=K.PREFIX)
    mods = K.support_modules(rm)
    assert K.SUPPORT_LAYERS == (24, 25, 26, 27)
    assert len(mods) == 8
    assert sum(m.endswith("linear_attn.out_proj") for m in mods) == 3
    assert sum(m.endswith("self_attn.o_proj") for m in mods) == 1
    assert sum(m.endswith("mlp.down_proj") for m in mods) == 4


def test_hash_only_hides_values():
    res = {
        "wikitext_nll": [{"id": "a", "value": 1.25, "n_tokens": 3}],
        "sycophancy_gap": [{"id": "b", "value": -0.5, "neutral": 0.1, "suggested": 0.6}],
    }
    h = K.hash_only(res)
    flat = json.dumps(h)
    assert "1.25" not in flat and "0.6" not in flat and "value" not in flat
    assert h["n_items"] == {"wikitext_nll": 1, "sycophancy_gap": 1}
    assert len(h["digest"]) == 64 and set(h["per_objective"]) == set(res)
    assert K.hash_only(res) == h


def test_gates():
    m = {
        "peak_device_bytes": 71.9e9,
        "peak_rss_bytes": 30e9,
        "peak_swap_bytes": 0,
        "repeats": {"base": True, "edit": True, "restore_hash": True, "post_restore": True},
        "panel_seconds_per_state": [60.0, 61.0],
    }
    g = K.gates(m)
    assert g["Go-M"]["pass"] and g["Go-N"]["pass"] and g["Go-C"]["pass"]
    assert not K.gates({**m, "peak_device_bytes": 72.1e9})["Go-M"]["pass"]
    assert not K.gates({**m, "peak_swap_bytes": 1})["Go-M"]["pass"]
    assert not K.gates({**m, "repeats": {**m["repeats"], "edit": False}})["Go-N"]["pass"]
    assert not K.gates({**m, "panel_seconds_per_state": [60.0, 241.0]})["Go-C"]["pass"]


def test_bank_hours_is_the_d8_formula():
    # 9B pilot check: (960 x 24.8 + 96 x 21.8 + 2.30e3 + 0) / 3600 = 7.84
    h = K.bank_hours(visits=960, t_ll=24.8, n_gen=96, t_gen=21.8, t_features=2304, t_train=0)
    assert h == pytest.approx(7.84, abs=0.01)


def test_dp_projection():
    prices = {1: 4.89, 2: 9.58, 4: 18.76, 8: 36.72}
    p = K.dp_projection(
        visits=960, t_state=60.0, setup_s=900, load_s=200, t_group=600, n_groups=16, prices=prices
    )
    one = p[1]
    assert one["wall_hours"] == pytest.approx((900 + 200 + 960 * 60 + 16 * 600) / 3600)
    assert one["usd"] == pytest.approx(one["wall_hours"] * 4.89)
    eight = p[8]
    assert eight["wall_hours"] == pytest.approx((900 + 200 + 120 * 60 + 2 * 600) / 3600)
    assert eight["gpu_hours"] == pytest.approx(8 * eight["wall_hours"])
    assert all(math.isfinite(v["usd"]) for v in p.values())
