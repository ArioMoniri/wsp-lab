"""wm.svf: writer keys, inert/bitwise edits, LoRA deltas, SVF (z = 1 inert, hook = dense delta), PG loss, H4."""

import pytest
import torch
from test_wm_panel import toy_model

from wsp.wm import svf as V


def test_writer_keys_parse_hf_and_vllm_names():
    assert V.writer_key("model.language_model.layers.3.mlp.down_proj.weight") == (3, "down")
    assert V.writer_key("language_model.model.layers.12.linear_attn.out_proj.weight") == (12, "o")
    assert V.writer_key("language_model.model.layers.15.self_attn.o_proj.weight") == (15, "o")
    assert V.writer_key("model.layers.3.mlp.up_proj.weight") is None
    m = V.writer_map(["a.layers.1.mlp.down_proj.weight", "a.layers.1.self_attn.o_proj.weight", "x"])
    assert set(m) == {(1, "down"), (1, "o")}
    with pytest.raises(ValueError):
        V.writer_map(["a.layers.1.mlp.down_proj.weight", "b.layers.1.mlp.down_proj.weight"])


def test_live_editor_zero_delta_and_restore_are_bitwise():
    torch.manual_seed(0)
    w = torch.randn(8, 6).to(torch.bfloat16)
    w0 = w.clone()
    ed = V.LiveEditor({"w": w})
    ed.apply([("w", lambda: torch.zeros(8, 6))])
    assert torch.equal(w, w0)
    ed.apply([("w", lambda: torch.ones(8, 6))])
    assert not torch.equal(w, w0)
    ed.apply([("w", lambda: torch.ones(8, 6) * 2)])  # deltas are relative to W0, not cumulative
    assert torch.equal(w, (w0.float() + 2).to(torch.bfloat16))
    c = ed.checksum(["w"])
    ed.restore()
    assert torch.equal(w, w0) and c != ed.checksum(["w"])


def test_lora_delta_and_projection():
    b, a = torch.randn(6, 2), torch.randn(2, 5)
    d = V.lora_delta((b, a, 2.0), 0.5, "cpu")
    assert torch.allclose(d, b @ a)
    assert torch.equal(V.lora_delta((b, a, 2.0), 0.0, "cpu"), torch.zeros(6, 5))
    r = torch.randn(6)
    p = V.project_out(d, r)
    assert torch.allclose((r / r.norm()) @ p, torch.zeros(5), atol=1e-5)
    assert torch.equal(V.project_out(d, None), d)
    q = V.project_out(p, r)  # idempotent
    assert torch.allclose(p, q, atol=1e-6)


@pytest.fixture(scope="module")
def tiny():
    m = toy_model()
    for p in m.parameters():
        p.requires_grad_(False)
    names = [n for n, _ in m.named_parameters() if V.writer_key(n) is not None]
    return m, names


def test_svf_inert_at_one_and_hook_equals_dense_delta(tiny):
    m, names = tiny
    s = V.SVF(m, names, store_dtype=torch.float32)
    ids = torch.randint(2, 120, (2, 9))
    base = m(input_ids=ids).logits
    with s.active():
        same = m(input_ids=ids).logits
    assert torch.equal(base, same)
    assert all(torch.equal(s.delta(n), torch.zeros_like(s.delta(n))) for n in names)
    with torch.no_grad():
        for n in names:
            s.zvec(n).mul_(1.1)
    with s.active():
        hooked = m(input_ids=ids).logits
    params = dict(m.named_parameters())
    ed = V.LiveEditor(params)
    ed.apply([(n, lambda n=n: s.delta(n)) for n in names])
    dense = m(input_ids=ids).logits
    ed.restore()
    assert torch.allclose(hooked, dense, atol=1e-4)
    st = s.state()
    s.load_state({n: torch.ones_like(z) for n, z in st.items()})
    assert all(torch.equal(z, torch.ones_like(z)) for z in s.state().values())


def test_svf_gradient_flows_to_z_only(tiny):
    m, names = tiny
    s = V.SVF(m, names[:2], store_dtype=torch.float32)
    ids = torch.randint(2, 120, (2, 7))
    mask = torch.ones_like(ids, dtype=torch.bool)
    with s.active():
        lp = V.token_logps(m, ids, torch.ones_like(ids), mask)
    lp.sum().backward()
    assert all(s.zvec(n).grad is not None and s.zvec(n).grad.abs().sum() > 0 for n in names[:2])
    assert all(p.grad is None for p in m.parameters())


def test_pg_loss_and_advantages():
    adv = V.group_advantages([1, -1, 1, 1, -1, -1, -1, -1], 4)
    assert torch.allclose(adv, torch.tensor([0.5, -1.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))
    logp = torch.tensor([-1.0, -2.0, -1.0], requires_grad=True)
    seq = torch.tensor([0, 0, 1])
    loss, kl = V.pg_loss(logp, logp.detach(), seq, torch.tensor([1.0, -1.0]), beta=0.1)
    assert kl == pytest.approx(0.0)
    loss.backward()
    # positive advantage raises its sequence's log-probs, negative lowers them
    assert logp.grad[0] < 0 and logp.grad[1] < 0 and logp.grad[2] > 0
