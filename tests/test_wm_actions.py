"""wm.actions: action families, compositions, bf16 realisation, restore and theta-hash (SYNTHESIS-r8 D8)."""

import random

import pytest
import torch

from wsp.wm import actions as A

CFG_9B_LIKE = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "tie_word_embeddings": False,
    "text_config": {
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 4,
        "vocab_size": 50,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "tie_word_embeddings": False,
    },
}


def _state(seed=0, prefix="model.layers."):
    g = torch.Generator().manual_seed(seed)
    s = {}
    for i in range(4):
        mixer = "self_attn.o_proj" if i == 3 else "linear_attn.out_proj"
        s[f"{prefix}{i}.{mixer}.weight"] = (torch.randn(16, 16, generator=g) * 0.05).bfloat16()
        s[f"{prefix}{i}.mlp.down_proj.weight"] = (
            torch.randn(16, 24, generator=g) * 0.05
        ).bfloat16()
        s[f"{prefix}{i}.mlp.up_proj.weight"] = (torch.randn(24, 16, generator=g) * 0.05).bfloat16()
    s["lm_head.weight"] = (torch.randn(50, 16, generator=g) * 0.05).bfloat16()
    s["mtp.layers.0.mlp.down_proj.weight"] = (torch.randn(16, 24, generator=g) * 0.05).bfloat16()
    s["model.visual.blocks.0.attn.proj.weight"] = (torch.randn(8, 8, generator=g) * 0.05).bfloat16()
    return s


def _session(state, rolemap=None):
    rolemap = rolemap or A.RoleMap.from_config(CFG_9B_LIKE)
    return A.EditSession(state, base_hash=A.state_hash(state), rolemap=rolemap)


# ------------------------------------------------------------------ role map


def test_role_map_is_read_from_config_not_hard_coded():
    rm = A.RoleMap.from_config(CFG_9B_LIKE)
    assert rm.n_layers == 4 and rm.hidden == 16 and rm.ffn == 24 and rm.vocab == 50
    assert rm.layer_types[3] == "full_attention"
    w = rm.writers()
    assert w[(3, "o")] == "model.layers.3.self_attn.o_proj.weight"
    assert w[(0, "o")] == "model.layers.0.linear_attn.out_proj.weight"
    assert w[(2, "down")] == "model.layers.2.mlp.down_proj.weight"
    assert len(w) == 8
    cfg = {**CFG_9B_LIKE, "text_config": {**CFG_9B_LIKE["text_config"], "num_hidden_layers": 8}}
    cfg["text_config"]["layer_types"] = ["full_attention"] * 8
    assert A.RoleMap.from_config(cfg).n_layers == 8
    assert A.RoleMap.from_config(cfg).writers()[(7, "o")].endswith("7.self_attn.o_proj.weight")


def test_role_map_rejects_inconsistent_layer_types():
    cfg = {"text_config": {**CFG_9B_LIKE["text_config"], "layer_types": ["full_attention"]}}
    with pytest.raises(ValueError):
        A.RoleMap.from_config(cfg)


def test_role_map_prefix_and_flat_config():
    flat = dict(CFG_9B_LIKE["text_config"])
    rm = A.RoleMap.from_config(flat, prefix="model.language_model.layers.")
    assert rm.writers()[(1, "down")] == "model.language_model.layers.1.mlp.down_proj.weight"


def test_vision_and_mtp_tensors_are_frozen():
    state = _state()
    s = _session(state)
    for name in ("mtp.layers.0.mlp.down_proj.weight", "model.visual.blocks.0.attn.proj.weight"):
        with pytest.raises(A.FrozenTensorError):
            s.apply(A.scale(name, factor=1.5, dose=1.0))
    assert s.theta_hash() == s.base_hash


# ------------------------------------------------------------------ realisation


def test_realise_zero_delta_is_bitwise_identity_including_negative_zero():
    w = torch.tensor([0.0, -0.0, 1.5, -2.25, 3e-8]).bfloat16()
    out = A.realise(w, torch.zeros(5))
    assert torch.equal(out.view(torch.int16), w.view(torch.int16))


def test_realise_rounds_through_bf16_and_reports_loss():
    w = torch.ones(4).bfloat16()
    delta = torch.full((4,), 1e-4)
    out = A.realise(w, delta)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, w)  # rounded away: bf16 spacing at 1.0 is 2^-7
    rec = A.realised_record(w, delta)
    assert rec["realised_norm"] == 0.0 and rec["intended_norm"] > 0
    assert rec["rounded_away_frac"] == 1.0


def test_zero_action_is_bitwise_identity_for_every_family():
    state = _state()
    rm = A.RoleMap.from_config(CFG_9B_LIKE)
    before = {k: v.clone() for k, v in state.items()}
    s = _session(state, rm)
    name = rm.writers()[(1, "down")]
    u = torch.randn(16)
    v = torch.randn(24)
    for act in (
        A.scale(name, factor=1.7, dose=0.0),
        A.row_scale(name, rows=[0, 3], factor=2.0, dose=0.0),
        A.proj_remove(name, direction=u, alpha=1.0, dose=0.0),
        A.rank_one(name, u=u, v=v, alpha=0.3, dose=0.0),
        A.noise(name, seed=3, rho=0.1, dose=0.0),
        A.additive(name, delta=torch.randn(16, 24), dose=0.0),
    ):
        s.apply(act)
    for k in state:
        assert torch.equal(state[k].view(torch.int16), before[k].view(torch.int16)), k
    assert s.theta_hash() == s.base_hash


# ------------------------------------------------------------------ families


def test_scaling_ladder_endpoints_are_built_from_base():
    state = _state()
    rm = A.RoleMap.from_config(CFG_9B_LIKE)
    name = rm.writers()[(2, "down")]
    w0 = state[name].float().clone()
    s = _session(state, rm)
    for dose in (0.25, -0.5, 2.0):
        s.restore()
        s.apply(A.scale(name, factor=1.4, dose=dose))
        expect = ((1 + dose * 0.4) * w0).bfloat16()
        assert torch.equal(state[name], expect)


def test_projection_removal_removes_the_output_direction():
    state = {"model.layers.0.mlp.down_proj.weight": torch.randn(16, 24).bfloat16()}
    rm = A.RoleMap.from_config(CFG_9B_LIKE)
    u = torch.randn(16)
    s = A.EditSession(state, base_hash=A.state_hash(state), rolemap=rm)
    s.apply(A.proj_remove("model.layers.0.mlp.down_proj.weight", direction=u, alpha=1.0, dose=1.0))
    w = state["model.layers.0.mlp.down_proj.weight"].float()
    uh = u / u.norm()
    assert (uh @ w).abs().max() < 0.05 * w.abs().max()


def test_rank_one_and_additive_and_row_scale_and_noise_intended_deltas():
    w = torch.randn(16, 24)
    u, v = torch.randn(16), torch.randn(24)
    assert torch.allclose(
        A.rank_one("t", u=u, v=v, alpha=0.5, dose=2.0).intended_delta(w), torch.outer(u, v)
    )
    d = torch.randn(16, 24)
    assert torch.allclose(A.additive("t", delta=d, dose=-0.5).intended_delta(w), -0.5 * d)
    rs = A.row_scale("t", rows=[1], factor=3.0, dose=0.5).intended_delta(w)
    assert torch.allclose(rs[1], w[1]) and rs[0].abs().max() == 0
    n1 = A.noise("t", seed=7, rho=0.1, dose=1.0).intended_delta(w)
    n2 = A.noise("t", seed=7, rho=0.1, dose=2.0).intended_delta(w)
    assert torch.allclose(n1.norm(), 0.1 * w.norm(), rtol=1e-5)
    assert torch.allclose(n2, 2 * n1)


def test_action_key_is_content_addressed():
    u = torch.arange(4.0)
    a = A.rank_one("t", u=u, v=u, alpha=1.0, dose=1.0)
    b = A.rank_one("t", u=u.clone(), v=u.clone(), alpha=1.0, dose=1.0)
    c = A.rank_one("t", u=u + 1, v=u, alpha=1.0, dose=1.0)
    assert a.key() == b.key() != c.key()
    assert a.with_dose(2.0).key() != a.key()


# ------------------------------------------------------------------ restore and hashes


def test_restore_from_saved_bytes_returns_base_hash_although_subtract_is_inexact():
    state = _state()
    rm = A.RoleMap.from_config(CFG_9B_LIKE)
    name = rm.writers()[(1, "down")]
    base = A.state_hash(state)
    s = _session(state, rm)
    act = A.noise(name, seed=1, rho=0.37, dose=1.0)
    rec = s.apply(act)
    assert s.theta_hash() != base
    assert rec.intended_norm > 0 and torch.equal(s.original(name), _state()[name])
    s.restore()
    assert A.state_hash(state) == base
    assert s.theta_hash() == base
    assert s.touched == ()


def test_inexact_bf16_subtract_is_demonstrated():
    g = torch.Generator().manual_seed(5)
    w = (torch.randn(4096, generator=g)).bfloat16()
    d = torch.randn(4096, generator=g) * 0.37
    new = A.realise(w, d)
    naive = A.realise(new, -d)  # bf16 add-then-subtract of the intended delta
    assert not torch.equal(naive.view(torch.int16), w.view(torch.int16))


def test_theta_hash_is_invariant_to_tensor_order_and_matches_full_state_semantics():
    rm = A.RoleMap.from_config(CFG_9B_LIKE)
    names = [rm.writers()[(i, r)] for i in range(4) for r in ("o", "down")]
    acts = [A.noise(n, seed=i, rho=0.2, dose=1.0) for i, n in enumerate(names)]
    hashes, states = set(), []
    for perm_seed in range(4):
        state = _state()
        s = _session(state, rm)
        order = list(acts)
        random.Random(perm_seed).shuffle(order)
        s.apply_bundle(order)
        hashes.add(s.theta_hash())
        states.append(A.state_hash(state))
    assert len(hashes) == 1 and len(set(states)) == 1


def test_theta_hash_differs_for_different_states_and_ignores_unchanged_touches():
    state = _state()
    rm = A.RoleMap.from_config(CFG_9B_LIKE)
    s = _session(state, rm)
    name = rm.writers()[(0, "down")]
    s.apply(A.scale(name, factor=1.0, dose=1.0))  # touched but identical bytes
    assert s.theta_hash() == s.base_hash
    s.apply(A.scale(name, factor=2.0, dose=1.0))
    h1 = s.theta_hash()
    s.apply(A.scale(name, factor=2.0, dose=1.0))
    assert s.theta_hash() not in (h1, s.base_hash)


def test_composition_is_sequential_and_records_exact_bookkeeping():
    state = _state()
    rm = A.RoleMap.from_config(CFG_9B_LIKE)
    name = rm.writers()[(2, "o")]
    w0 = state[name].float().clone()
    s = _session(state, rm)
    u = torch.randn(16)
    v = torch.randn(16)
    steps = [
        A.proj_remove(name, direction=u, alpha=1.0, dose=0.5),
        A.rank_one(name, u=u, v=v, alpha=0.1, dose=1.0),
    ]
    recs = s.compose(steps)
    assert len(recs) == 2
    e = state[name].float() - w0
    assert torch.allclose(recs[0].realised + recs[1].realised, e, atol=1e-7, rtol=0)
    assert torch.equal(s.displacement(name), e)
    tok = A.step_tokens(recs, s)
    assert [t["position"] for t in tok] == [0, 1]
    assert tok[0]["overlap_prev_E"] == 0.0  # nothing before the first step
    assert tok[1]["family"] == "rank_one" and tok[1]["dose"] == 1.0
    # order matters for non-commuting actions
    state2 = _state()
    s2 = _session(state2, rm)
    s2.compose(steps[::-1])
    assert s2.theta_hash() != s.theta_hash()


def test_frozen_session_refuses_unknown_tensor():
    s = _session(_state())
    with pytest.raises(KeyError):
        s.apply(A.scale("model.layers.9.mlp.down_proj.weight", factor=2.0, dose=1.0))
