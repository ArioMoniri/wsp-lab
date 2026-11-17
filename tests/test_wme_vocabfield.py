"""wm.vocabfield: field coverage for vocabulary-row roots (row_scale on lm_head / embed_tokens rows).

Declared approximations, never a vocabulary-sized covariance:
* lm_head rows: *projected* K-FAC, A = E[x xᵀ] (hidden²) and G restricted to the declared rows R (|R|²) -- equal to
  the R-block of the full K-FAC G, so q is exact K-FAC for edits supported on R;
* embed_tokens rows: *row-diagonal* per-token Fisher, D[r, i] = Σ_{p: t_p = r} δ_{p,i}² / n.
Checked against brute force on a toy with a 23-token vocabulary, where the full covariance is formable.
"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from wsp.features import response_field as rf
from wsp.wm import vocabfield as VF

VOCAB, D = 23, 8


class ToyLM(nn.Module):
    def __init__(self, tied=False):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(VOCAB, D)
        self.mix = nn.Linear(D, D)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
        if tied:
            self.lm_head.weight = self.emb.weight

    def forward(self, ids):
        h = self.emb(ids)
        h = h + torch.tanh(
            self.mix(torch.cumsum(h, 1) / torch.arange(1, ids.shape[1] + 1)[:, None])
        )
        return self.lm_head(h)


def nll_scorer(model, tokenizer, item, device):
    ids = item.to(device)
    logits = model(ids)
    mask = torch.ones(ids.shape, dtype=torch.bool)
    mask[:, -1] = False
    lp = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]
    return rf.ScoreBatch(-lp.sum(), float(mask.sum()), logits, mask)


def _panels():
    g = torch.Generator().manual_seed(4)
    items = [torch.randint(0, VOCAB, (2, 6), generator=g) for _ in range(3)]
    p = rf.FieldPanel("toy", lambda: iter(items), ("a", "b", "c"), ("scored",))
    objs = [rf.ObjectiveSpec("toy_nll", "lower", nll_scorer, "toy", p.text_group_hash)]
    return objs, {"toy": p}, items


ROWS = (2, 5, 7, 11, 19)


def _declared(rows=ROWS):
    return VF.declare_rows({"lm_head": {"r0": list(rows)}, "embed": {"r1": list(rows)}}, VOCAB)


def _collect(tmp_path, model=None, rows=ROWS):
    model = model or ToyLM().eval()
    objs, panels, _ = _panels()
    return VF.collect_vocab_rows(
        model,
        None,
        objs,
        panels,
        _declared(rows),
        "cpu",
        tmp_path,
        lm_head="lm_head",
        embed="emb",
    )


def _brute():
    """Full (vocabulary-sized) gradient, K-FAC A/G for lm_head and the per-token Fisher diagonal for emb."""
    model = ToyLM().eval()
    _, _, items = _panels()
    gen = torch.Generator().manual_seed(rf.SEED)
    g_lm, g_emb, total = torch.zeros(VOCAB, D), torch.zeros(VOCAB, D), 0.0
    A, G, Demb, n = torch.zeros(D, D), torch.zeros(VOCAB, VOCAB), torch.zeros(VOCAB, D), 0
    for it in items:
        model.zero_grad()
        b = nll_scorer(model, None, it, "cpu")
        b.score_sum.backward()
        g_lm += model.lm_head.weight.grad
        g_emb += model.emb.weight.grad
        total += b.weight
        model.zero_grad()
        cap = {}
        h1 = model.lm_head.register_forward_hook(lambda m, i, o, cap=cap: cap.update(x=i[0], y=o))
        h2 = model.emb.register_forward_hook(lambda m, i, o, cap=cap: cap.update(ids=i[0], e=o))
        b = nll_scorer(model, None, it, "cpu")
        h1.remove()
        h2.remove()
        cap["y"].retain_grad()
        cap["e"].retain_grad()
        mask = b.valid_mask
        logits = b.logits[mask]
        t = torch.multinomial(logits.detach().softmax(-1), 1, generator=gen)
        F.cross_entropy(logits, t[:, 0], reduction="sum").backward()
        x, dy = cap["x"][mask].detach(), cap["y"].grad[mask]
        A += x.T @ x
        G += dy.T @ dy
        n += len(x)
        de, ids = cap["e"].grad, cap["ids"]
        for r in range(VOCAB):
            Demb[r] += (de[ids == r] ** 2).sum(0)
    return {
        "g_lm": g_lm / total,
        "g_emb": g_emb / total,
        "A": A / n,
        "G": G / n,
        "Demb": Demb / n,
        "n": n,
    }


@pytest.fixture(scope="module")
def brute():
    return _brute()


def test_projected_lm_head_equals_the_R_block_of_full_kfac(tmp_path, brute):
    vf = _collect(tmp_path)
    R = list(vf.rows("lm_head"))
    f = vf.factors("toy_nll", "lm_head")
    torch.testing.assert_close(f["A"], brute["A"], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(f["G_R"], brute["G"][R][:, R], rtol=1e-5, atol=1e-7)
    assert f["count"] == brute["n"]
    torch.testing.assert_close(
        vf.gradient("toy_nll", "lm_head"), brute["g_lm"][R], rtol=1e-5, atol=1e-7
    )


def test_embedding_row_diagonal_equals_brute_force(tmp_path, brute):
    vf = _collect(tmp_path)
    R = list(vf.rows("embed"))
    f = vf.factors("toy_nll", "embed")
    torch.testing.assert_close(f["D_R"], brute["Demb"][R], rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(
        vf.gradient("toy_nll", "embed"), brute["g_emb"][R], rtol=1e-5, atol=1e-7
    )


def _row_scale_delta(model, which, rows, mult=1.5):
    W = (model.lm_head if which == "lm_head" else model.emb).weight.detach()
    d = torch.zeros_like(W)
    d[list(rows)] = (mult - 1) * W[list(rows)]
    return d


def test_contract_row_scale_matches_full_kfac_and_diagonal(tmp_path, brute):
    vf = _collect(tmp_path)
    m = ToyLM()
    rows = (5, 11)
    dl = _row_scale_delta(m, "lm_head", rows)
    de = _row_scale_delta(m, "embed", rows)
    out = vf.contract({"lm_head": dl, "embed": de})["toy_nll"]
    q_full = 0.5 * torch.trace(brute["G"] @ dl @ brute["A"] @ dl.T)
    assert out["lm_head"]["supported"] and out["lm_head"]["approximation"] == "projected_kfac"
    assert out["lm_head"]["q"] == pytest.approx(float(q_full), rel=1e-4, abs=1e-10)
    assert out["lm_head"]["signed"] == pytest.approx(float((brute["g_lm"] * dl).sum()), rel=1e-4)
    q_diag = 0.5 * float((brute["Demb"] * de**2).sum())
    assert out["embed"]["approximation"] == "row_diagonal"
    assert out["embed"]["q"] == pytest.approx(q_diag, rel=1e-4, abs=1e-12)
    assert out["embed"]["signed"] == pytest.approx(float((brute["g_emb"] * de).sum()), rel=1e-4)


def test_row_sparse_delta_equals_dense(tmp_path):
    vf = _collect(tmp_path)
    m = ToyLM()
    rows = [5, 11]
    dense = _row_scale_delta(m, "lm_head", rows)
    sparse = {"rows": rows, "values": dense[rows]}
    a = vf.contract({"lm_head": dense})["toy_nll"]["lm_head"]
    b = vf.contract({"lm_head": sparse})["toy_nll"]["lm_head"]
    assert a["q"] == pytest.approx(b["q"], rel=1e-6) and a["signed"] == pytest.approx(
        b["signed"], rel=1e-6
    )


def test_rows_outside_declared_support_are_marked_not_zero(tmp_path):
    vf = _collect(tmp_path)
    m = ToyLM()
    out = vf.contract({"lm_head": _row_scale_delta(m, "lm_head", (3,))})["toy_nll"]["lm_head"]
    assert out == {
        "signed": None,
        "q": None,
        "supported": False,
        "reason": "rows_outside_declared_support",
        "approximation": None,
    }
    out = vf.contract({"lm_head": _row_scale_delta(m, "lm_head", (3, 5))})["toy_nll"]["lm_head"]
    assert out["supported"] is False and out["q"] is None  # partially outside: still unsupported


def test_unknown_tensor_is_marked_unsupported(tmp_path):
    vf = _collect(tmp_path)
    out = vf.contract({"mix.weight": torch.zeros(D, D)})["toy_nll"]["mix.weight"]
    assert (
        out["supported"] is False
        and out["q"] is None
        and out["reason"] == "not_vocabulary_row_tensor"
    )


def test_never_forms_vocabulary_sized_covariance(tmp_path):
    vf = _collect(tmp_path)
    for obj in vf.objectives:
        f = vf.factors(obj, "lm_head")
        assert tuple(f["G_R"].shape) == (len(ROWS), len(ROWS))
        assert tuple(vf.factors(obj, "embed")["D_R"].shape) == (len(ROWS), D)
    with pytest.raises(VF.VocabularySized):
        VF.declare_rows({"lm_head": {"r": list(range(VOCAB))}, "embed": {}}, VOCAB)
    with pytest.raises(VF.VocabularySized):
        VF.declare_rows(
            {"lm_head": {"r": list(range(VF.MAX_DECLARED_ROWS + 1))}, "embed": {}}, 10**6
        )


def test_tied_embeddings_are_refused(tmp_path):
    with pytest.raises(VF.TiedEmbeddingError):
        _collect(tmp_path, model=ToyLM(tied=True))


def test_declared_rows_identity_and_union():
    d = VF.declare_rows({"lm_head": {"a": [5, 2], "b": [2, 9]}, "embed": {"c": [1]}}, VOCAB)
    assert d.rows("lm_head") == (2, 5, 9) and d.rows("embed") == (1,)
    d2 = VF.declare_rows({"lm_head": {"b": [9, 2], "a": [2, 5]}, "embed": {"c": [1]}}, VOCAB)
    assert d.identity() == d2.identity()
    with pytest.raises(ValueError):
        VF.declare_rows({"lm_head": {"a": [VOCAB]}, "embed": {}}, VOCAB)


def test_manifest_roundtrip_and_hash_check(tmp_path):
    vf = _collect(tmp_path)
    again = VF.VocabField.load(tmp_path)
    assert again.identity["declared_rows"] == vf.identity["declared_rows"]
    torch.testing.assert_close(
        again.factors("toy_nll", "lm_head")["G_R"], vf.factors("toy_nll", "lm_head")["G_R"]
    )
    (tmp_path / "toy_nll.vocab.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        VF.VocabField.load(tmp_path)


def test_shared_field_spec_names_both_approximations():
    s = VF.SHARED_FIELD_SPEC
    assert "projected_kfac" in s and "row_diagonal" in s and "wme-grouped-field-v1" in s
    assert VF.APPROXIMATIONS["lm_head"]["name"] == "projected_kfac"
    assert VF.APPROXIMATIONS["embed"]["name"] == "row_diagonal"


def test_planned_declared_rows_follow_the_bank_recipe_and_inert_spec():
    from wsp.wm import bank as B
    from wsp.wm import controls as C

    roots = B.make_roots()
    rs = [r for r in roots.values() if r["family"] == "row_scale"]
    per = B.NOMINALS["row_scale"]["rows"]
    n_lm = sum(per for r in rs if r["tensor"] == "lm_head")
    n_emb = sum(per for r in rs if r["tensor"] == "embed_tokens")
    dead = C.INERT_SPEC["dead_rows"]
    assert VF.PLANNED_ROWS == {
        "lm_head": n_lm,
        "embed": n_emb + dead["n_controls"] * dead["rows_per_control"],
    }
    assert max(VF.PLANNED_ROWS.values()) <= VF.MAX_DECLARED_ROWS
    assert "dead" in VF.DECLARED_ROWS_RULE and "row_scale" in VF.DECLARED_ROWS_RULE
