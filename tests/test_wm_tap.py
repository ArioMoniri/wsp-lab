"""wm.tap: seeded residual sketches and per-item losses; evaluation outputs bitwise unchanged (D8)."""

import torch
from torch import nn

from wsp.wm import tap as T


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return (x + torch.tanh(self.lin(x)),)  # HF decoder layers return tuples


class Toy(nn.Module):
    def __init__(self, d=12, vocab=20, n=3):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(Block(d) for _ in range(n))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        h = self.emb(ids)
        for layer in self.layers:
            h = layer(h)[0]
        return self.head(h)


def _losses(model, ids):
    logits = model(ids)
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none"
    ).view(ids.shape[0], -1)


def test_tap_leaves_outputs_bitwise_unchanged_and_records_sketches():
    m = Toy().eval()
    ids = torch.randint(0, 20, (2, 7), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = m(ids)
        tap = T.ResidualTap({i: m.layers[i] for i in range(3)}, hidden=12, k=4, seed=11)
        with tap.recording(item_ids=["a", "b"], mask=torch.ones(2, 7, dtype=torch.bool)):
            out = m(ids)
    assert torch.equal(ref.view(torch.int32), out.view(torch.int32))
    sk = tap.sketches()
    assert set(sk) == {"a", "b"} and sk["a"].shape == (3, 4)
    assert not tap.active_handles()


def test_sketch_projection_is_seeded_and_layer_specific():
    p1 = T.sketch_projection(12, 4, seed=11, layer=0)
    p2 = T.sketch_projection(12, 4, seed=11, layer=0)
    p3 = T.sketch_projection(12, 4, seed=11, layer=1)
    assert torch.equal(p1, p2) and not torch.equal(p1, p3)


def test_sketch_is_masked_mean_of_hidden_times_projection():
    m = Toy().eval()
    ids = torch.randint(0, 20, (1, 5), generator=torch.Generator().manual_seed(2))
    mask = torch.tensor([[True, True, False, True, False]])
    tap = T.ResidualTap({0: m.layers[0]}, hidden=12, k=3, seed=5)
    with torch.no_grad(), tap.recording(item_ids=["x"], mask=mask):
        m(ids)
        h = m.layers[0](m.emb(ids))[0][0]
    want = h[mask[0]].float().mean(0) @ T.sketch_projection(12, 3, seed=5, layer=0)
    assert torch.allclose(tap.sketches()["x"][0], want, atol=1e-6)


def test_per_item_losses_are_recorded_and_unchanged():
    m = Toy().eval()
    ids = torch.randint(0, 20, (2, 6), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        ref = _losses(m, ids).sum(1)
        tap = T.ResidualTap({1: m.layers[1]}, hidden=12, k=2, seed=1)
        with tap.recording(item_ids=["p", "q"], mask=torch.ones(2, 6, dtype=torch.bool)):
            got = _losses(m, ids).sum(1)
        tap.record_losses({"p": got[0].item(), "q": got[1].item()})
    assert torch.equal(ref, got)
    obs = tap.observation(theta_hash="a" * 64, base_hash="a" * 64)
    assert obs.tier.name == "R0" and obs.payload["item_losses"]["q"] == got[1].item()
    obs2 = tap.observation(theta_hash="c" * 64, base_hash="a" * 64)
    assert obs2.tier.name == "R2"


def test_tap_restores_on_exception():
    m = Toy().eval()
    tap = T.ResidualTap({0: m.layers[0]}, hidden=12, k=2, seed=1)
    try:
        with tap.recording(item_ids=["a"], mask=torch.ones(1, 3, dtype=torch.bool)):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not tap.active_handles()
