"""CPU contracts for the A5.8.1 field; no model or dataset downloads."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from wsp.features.response_field import (
    FieldArtifact,
    FieldPanel,
    MissingHookError,
    ObjectiveSpec,
    ScoreBatch,
    collect_field,
    contract,
    cross_term,
    predecessor_audit,
    spectral_factor,
)


def score(model, tokenizer, item, device):
    x, y = (t.to(device) for t in item)
    logits = model(x)
    mask = torch.ones_like(y, dtype=torch.bool)
    return ScoreBatch(
        F.cross_entropy(logits.flatten(0, -2), y.flatten(), reduction="sum"),
        y.numel(),
        logits,
        mask,
    )


@pytest.fixture
def setup():
    torch.manual_seed(41)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 3))
    items = [(torch.randn(1, n, 3), torch.arange(n).reshape(1, n) % 3) for n in (2, 3)]
    panel = FieldPanel("aux", lambda: iter(items), ("a", "b"), ("scored",))
    obj = ObjectiveSpec("synthetic_nll", "lower", score, "aux", panel.text_group_hash)
    return model, items, panel, obj


def collect(setup):
    model, _, panel, obj = setup
    return collect_field(model, None, [obj], {"aux": panel}, ["0", "2"], "cpu")


def test_gradient_matches_finite_difference_and_is_linear(setup):
    model, items, _, _ = setup
    field = collect(setup)
    delta = {n: torch.randn_like(p) * 1e-3 for n, p in model.named_parameters()}
    prediction = sum(t["signed"] for t in contract(field, delta)["synthetic_nll"].values())
    base = sum(score(model, None, i, "cpu").score_sum.item() for i in items) / 5
    edited = copy.deepcopy(model)
    with torch.no_grad():
        for n, p in edited.named_parameters():
            p.add_(delta[n])
    exact = sum(score(edited, None, i, "cpu").score_sum.item() for i in items) / 5 - base
    assert prediction == pytest.approx(exact, abs=4e-6)
    twice = contract(field, {n: 2 * d for n, d in delta.items()})["synthetic_nll"]
    for n, row in contract(field, delta)["synthetic_nll"].items():
        assert twice[n]["signed"] == pytest.approx(2 * row["signed"])
        assert row["q"] >= 0
    assert all(v == 0 for v in cross_term(field, {}, delta)["synthetic_nll"].values())


def test_one_layer_quadratic_and_bias_augmentation(setup):
    _, items, panel, obj = setup
    model = nn.Sequential(nn.Linear(3, 3))
    field = collect_field(model, None, [obj], {"aux": panel}, ["0"], "cpu")
    factors = field.factors[obj.name]["0"]
    a = spectral_factor(factors["A"])[0]
    g = spectral_factor(factors["G"])[0]
    d = torch.randn(3, 3)
    db = torch.randn(3)
    got = contract(field, {"0.weight": d, "0.bias": db})[obj.name]
    assert factors["A"].shape == (4, 4)
    expected = 0.5 * torch.trace(g @ d @ a[:3, :3] @ d.T)
    assert got["0.weight"]["q"] == pytest.approx(expected.item(), rel=1e-5)
    assert got["0.bias"]["q"] == pytest.approx((0.5 * a[-1, -1] * db @ g @ db).item())
    assert factors["count"] == sum(i[1].numel() for i in items)


def test_supported_but_unused_module_fails(setup):
    model, _, panel, obj = setup
    model.unused = nn.Linear(3, 3)

    # Sequential would execute an appended module; use an uncalled registered child instead.
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(3, 3)
            self.unused = nn.Linear(3, 3)

        def forward(self, x):
            return self.used(x)

    with pytest.raises(MissingHookError):
        collect_field(Wrapper(), None, [obj], {"aux": panel}, ["used", "unused"], "cpu")
    with pytest.raises(MissingHookError):
        collect_field(model, None, [obj], {"aux": panel}, ["absent"], "cpu")


def test_coverage_hash_roundtrip_and_tamper(setup, tmp_path):
    field = collect(setup)
    assert set(field.identity["coverage"]["synthetic_nll"]) == {
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
    }
    assert all(
        r["gradient"] and r["factors"] for r in field.identity["coverage"]["synthetic_nll"].values()
    )
    field.save(tmp_path)
    restored = FieldArtifact.load(tmp_path, expected_identity=field.identity)
    assert restored.content_hashes == field.content_hashes
    with pytest.raises(ValueError, match="identity"):
        FieldArtifact.load(tmp_path, expected_identity={})
    payload = torch.load(tmp_path / "tensors.pt", weights_only=True)
    payload["gradients"]["synthetic_nll"]["0.weight"][0, 0] += 1
    torch.save(payload, tmp_path / "tensors.pt")
    with pytest.raises(ValueError, match="hash"):
        FieldArtifact.load(tmp_path)


def test_disjointness_and_precision_fail_closed(setup):
    model, _, panel, obj = setup
    bad = FieldPanel("aux", panel.items, ("a", "b"), ("a",))
    with pytest.raises(ValueError, match="disjoint"):
        collect_field(model, None, [obj], {"aux": bad}, ["0"], "cpu")
    with pytest.raises(ValueError, match="fp32"):
        collect_field(model, None, [obj], {"aux": panel}, ["0"], "cpu", torch.float64)


def test_spectral_tail_zero_and_negative_gate():
    factor, info = spectral_factor(torch.diag(torch.tensor([100.0, 0.1, 0.0])))
    assert info["rank"] == 1 and info["omitted_trace"] > 0
    assert torch.linalg.eigvalsh(factor).min() >= 0
    assert spectral_factor(torch.zeros(2, 2))[1]["zero_trace"]
    with pytest.raises(ValueError, match="negative"):
        spectral_factor(torch.diag(torch.tensor([1.0, -0.01])))


def test_predecessor_exact_contraction_and_floor(setup):
    model, _, panel, obj = setup
    field = collect(setup)
    d = {n: torch.ones_like(p) * 1e-4 for n, p in model.named_parameters()}
    audit = predecessor_audit(model, None, [obj], {"aux": panel}, ["0", "2"], "cpu", field, {}, d)
    assert audit[obj.name]["relative_error"] == pytest.approx(0, abs=1e-6)
    assert audit[obj.name]["passed"]
    assert audit[obj.name]["denominator"] >= 1e-3


def test_fisher_uses_sampled_targets_not_gold_and_is_repeatable(setup):
    _, _, panel, obj = setup
    model = nn.Sequential(nn.Linear(3, 3, bias=False))
    field = collect_field(model, None, [obj], {"aux": panel}, ["0"], "cpu")
    generator = torch.Generator().manual_seed(20261107)
    a, g, count = torch.zeros(3, 3), torch.zeros(3, 3), 0
    for item in panel.items():
        x = item[0]
        probabilities = model(x).detach().softmax(-1).reshape(-1, 3)
        sampled = torch.multinomial(probabilities, 1, generator=generator)[:, 0]
        output_gradient = probabilities - F.one_hot(sampled, 3)
        a += x.reshape(-1, 3).T @ x.reshape(-1, 3)
        g += output_gradient.T @ output_gradient
        count += len(sampled)
    pair = field.factors[obj.name]["0"]
    assert torch.allclose(pair["A"], a / count)
    assert torch.allclose(pair["G"], g / count, atol=1e-7)
    again = collect_field(model, None, [obj], {"aux": panel}, ["0"], "cpu")
    assert again.content_hashes == field.content_hashes


def test_cross_term_includes_augmented_weight_bias_coupling(setup):
    model, _, panel, obj = setup
    field = collect_field(model, None, [obj], {"aux": panel}, ["0", "2"], "cpu")
    e = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    d = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    got = cross_term(field, e, d)[obj.name]
    for module, pair in field.factors[obj.name].items():
        a, g = spectral_factor(pair["A"])[0], spectral_factor(pair["G"])[0]
        em = torch.cat((e[f"{module}.weight"], e[f"{module}.bias"][:, None]), 1)
        dm = torch.cat((d[f"{module}.weight"], d[f"{module}.bias"][:, None]), 1)
        exact = torch.trace(dm.T @ g @ em @ a).item()
        assert got[f"{module}.weight"] + got[f"{module}.bias"] == pytest.approx(exact, abs=2e-6)


def test_coverage_explicitly_reports_unsupported_and_frozen_parameters(setup):
    model, _, panel, obj = setup
    model[0].bias.requires_grad_(False)
    field = collect_field(model, None, [obj], {"aux": panel}, ["2"], "cpu")
    coverage = field.identity["coverage"][obj.name]
    assert not coverage["0.weight"]["factors"]
    assert coverage["0.weight"]["gradient"]
    assert not coverage["0.bias"]["gradient"]
    assert (
        contract(field, {"0.weight": torch.zeros_like(model[0].weight)})[obj.name]["0.weight"]["q"]
        is None
    )
    with pytest.raises(ValueError, match="missing objective gradient"):
        contract(field, {"0.bias": torch.zeros_like(model[0].bias)})


def test_hooks_and_training_mode_restored_on_failure(setup):
    model, _, panel, obj = setup
    model.train()
    model[1].eval()
    modes = [m.training for m in model.modules()]
    bad = ObjectiveSpec(obj.name, obj.sign, score, "aux", "incorrect-hash")
    with pytest.raises(ValueError, match="identity"):
        collect_field(model, None, [bad], {"aux": panel}, ["0"], "cpu")
    assert modes == [m.training for m in model.modules()]
    assert not model[0]._forward_hooks
