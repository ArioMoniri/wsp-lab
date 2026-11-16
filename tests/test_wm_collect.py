"""wm.collect: grouped, streamed support-only gradient/K-FAC collector, equivalent to response_field on CPU toys.

Representation's memory contract (SYNTHESIS-r8 D8): gradients per layer group only, support-only field, never a
vocabulary-sized covariance, unsupported (fused/recurrent) tensors marked explicitly instead of q = 0.
"""

import pytest
import torch
from torch import nn

from wsp.features import response_field as rf
from wsp.wm import collect as C

VOCAB = 23


class Block(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, ff)
        self.down = nn.Linear(ff, d, bias=False)
        self.mix = nn.Conv1d(
            d, d, 3, padding=2, groups=d
        )  # a fused/recurrent-like operator, no Linear hook

    def forward(self, x):
        y = self.mix(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return x + self.down(torch.tanh(self.up(self.norm(x + y))))


class ToyLM(nn.Module):
    def __init__(self, d=8, ff=12, n=3):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(VOCAB, d)
        self.layers = nn.ModuleList(Block(d, ff) for _ in range(n))
        self.lm_head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, ids):
        h = self.emb(ids)
        for b in self.layers:
            h = b(h)
        return self.lm_head(h)


def nll_scorer(model, tokenizer, item, device):
    ids = item.to(device)
    logits = model(ids)
    mask = torch.ones(ids.shape, dtype=torch.bool, device=device)
    mask[:, -1] = False  # next-token scoring positions; also the supported Linears' leading dims
    lp = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]
    return rf.ScoreBatch(-lp.sum(), float(mask.sum()), logits, mask)


def last_token_scorer(model, tokenizer, item, device):
    ids = item.to(device)
    logits = model(ids)
    mask = torch.zeros(ids.shape, dtype=torch.bool, device=device)
    mask[:, -2] = True
    lp = logits[:, -2].log_softmax(-1).gather(-1, ids[:, -1:])
    return rf.ScoreBatch(lp.sum(), float(ids.shape[0]), logits, mask)


def _panels():
    g = torch.Generator().manual_seed(4)
    items = [torch.randint(0, VOCAB, (2, 6), generator=g) for _ in range(3)]
    groups = tuple(f"t{i}" for i in range(3))
    p = rf.FieldPanel("toy", lambda: iter(items), groups, ("scored",))
    objs = [
        rf.ObjectiveSpec("toy_nll", "lower", nll_scorer, "toy", p.text_group_hash),
        rf.ObjectiveSpec("toy_last", "higher", last_token_scorer, "toy", p.text_group_hash),
    ]
    return objs, {"toy": p}


SUPPORT = [f"layers.{i}.{r}" for i in range(3) for r in ("up", "down")]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    model = ToyLM().eval()
    objs, panels = _panels()
    return rf.collect_field(model, None, objs, panels, SUPPORT, "cpu")


@pytest.mark.parametrize(
    "groups", [[SUPPORT], [SUPPORT[:2], SUPPORT[2:4], SUPPORT[4:]], [[s] for s in SUPPORT]]
)
def test_grouped_collection_equals_response_field(reference, tmp_path, groups):
    model = ToyLM().eval()
    objs, panels = _panels()
    art = C.collect_grouped(model, None, objs, panels, groups, "cpu", out_dir=tmp_path)
    for o in objs:
        for mod in SUPPORT:
            for p in (f"{mod}.weight", f"{mod}.bias"):
                if p not in reference.gradients[o.name]:
                    continue
                torch.testing.assert_close(
                    art.gradient(o.name, p), reference.gradients[o.name][p], rtol=1e-6, atol=1e-7
                )
            f = art.factors(o.name, mod)
            for key in ("A", "G"):
                torch.testing.assert_close(
                    f[key], reference.factors[o.name][mod][key], rtol=1e-6, atol=1e-7
                )
            assert f["count"] == reference.factors[o.name][mod]["count"]


def test_contractions_equal_response_field(reference, tmp_path):
    model = ToyLM().eval()
    objs, panels = _panels()
    art = C.collect_grouped(
        model, None, objs, panels, [SUPPORT[:3], SUPPORT[3:]], "cpu", out_dir=tmp_path
    )
    g = torch.Generator().manual_seed(9)
    deltas = {
        "layers.1.down.weight": torch.randn(8, 12, generator=g) * 0.01,
        "layers.0.up.weight": torch.randn(12, 8, generator=g) * 0.01,
    }
    ref = rf.contract(reference, deltas)
    got = C.contract(art, deltas)
    for o in ref:
        for n in deltas:
            assert got[o][n]["signed"] == pytest.approx(ref[o][n]["signed"], rel=1e-5, abs=1e-9)
            assert got[o][n]["q"] == pytest.approx(ref[o][n]["q"], rel=1e-4, abs=1e-9)


def test_only_the_current_group_requires_grad_and_flags_are_restored(tmp_path):
    model = ToyLM().eval()
    objs, panels = _panels()
    seen = []
    orig = C._group_pass

    def spy(model_, *a, **k):
        seen.append(sorted(n for n, p in model_.named_parameters() if p.requires_grad))
        return orig(model_, *a, **k)

    C._group_pass = spy
    try:
        C.collect_grouped(
            model, None, objs[:1], panels, [SUPPORT[:2], SUPPORT[2:]], "cpu", out_dir=tmp_path
        )
    finally:
        C._group_pass = orig
    assert seen[0] == sorted(
        f"{m}.{s}"
        for m in SUPPORT[:2]
        for s in ("weight", "bias")
        if not (m.endswith("down") and s == "bias")
    )
    assert all(p.requires_grad for p in model.parameters())


def test_unsupported_tensors_are_marked_and_contract_gives_none_not_zero(tmp_path):
    model = ToyLM().eval()
    objs, panels = _panels()
    art = C.collect_grouped(
        model,
        None,
        objs[:1],
        panels,
        [["layers.0.down", "layers.0.mix", "lm_head"]],
        "cpu",
        out_dir=tmp_path,
    )
    cov = art.coverage["toy_nll"]
    assert cov["layers.0.mix.weight"]["factors"] is False
    assert cov["layers.0.mix.weight"]["reason"] == "not_linear"
    assert cov["lm_head.weight"]["factors"] is False
    assert cov["lm_head.weight"]["reason"] == "vocabulary_output"
    assert cov["layers.0.down.weight"]["factors"] is True
    assert "layers.2.up.weight" not in cov or cov["layers.2.up.weight"]["gradient"] is False
    out = C.contract(art, {"layers.0.mix.weight": torch.ones(8, 1, 3) * 1e-3})
    assert out["toy_nll"]["layers.0.mix.weight"]["q"] is None
    assert out["toy_nll"]["layers.0.mix.weight"]["signed"] != 0.0
    with pytest.raises(C.UnsupportedTensor):
        C.contract(art, {"layers.2.up.weight": torch.ones(12, 8)})


def test_vocabulary_sized_covariance_is_never_formed(tmp_path, monkeypatch):
    model = ToyLM().eval()
    objs, panels = _panels()
    biggest = []
    real_mm = torch.Tensor.__matmul__

    def watch(a, b):
        out = real_mm(a, b)
        if out.ndim == 2 and out.shape[0] == out.shape[1]:
            biggest.append(out.shape[0])
        return out

    monkeypatch.setattr(torch.Tensor, "__matmul__", watch)
    C.collect_grouped(
        model, None, objs[:1], panels, [["lm_head", "layers.0.down"]], "cpu", out_dir=tmp_path
    )
    assert biggest and max(biggest) < VOCAB


def test_artifacts_stream_one_file_per_objective_group(tmp_path):
    model = ToyLM().eval()
    objs, panels = _panels()
    art = C.collect_grouped(
        model, None, objs, panels, [SUPPORT[:2], SUPPORT[2:]], "cpu", out_dir=tmp_path
    )
    files = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert files == ["toy_last.g0.pt", "toy_last.g1.pt", "toy_nll.g0.pt", "toy_nll.g1.pt"]
    assert (tmp_path / "manifest.json").exists()
    assert art.identity["groups"] == [SUPPORT[:2], SUPPORT[2:]]
    assert art.identity["objectives"][0]["name"] == "toy_nll"
    re = C.GroupedField.load(tmp_path)
    torch.testing.assert_close(
        re.gradient("toy_nll", "layers.1.down.weight"),
        art.gradient("toy_nll", "layers.1.down.weight"),
    )


def test_gradients_only_without_factors(tmp_path):
    model = ToyLM().eval()
    objs, panels = _panels()
    art = C.collect_grouped(
        model, None, objs[:1], panels, [SUPPORT[:2]], "cpu", out_dir=tmp_path, fisher=False
    )
    assert art.factors("toy_nll", SUPPORT[0]) is None
    assert art.gradient("toy_nll", SUPPORT[0] + ".weight").shape == (12, 8)
