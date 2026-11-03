import torch

from wsp.predictor import TrajectoryPredictor, ssm_backend


def test_backend_is_reported():
    assert ssm_backend() in {"torch", "mamba_ssm"}


def test_outputs_bounded_and_causal():
    torch.manual_seed(0)
    m = TrajectoryPredictor(d_in=12, n_targets=3, d_model=32, n_layers=2, backend="torch").eval()
    x = torch.randn(2, 5, 12)
    y = m(x)
    assert y.shape == (2, 5, 3) and y.min() >= 0 and y.max() <= 1
    x2 = x.clone()
    x2[:, 3:] += 10.0  # perturb the future
    y2 = m(x2)
    assert torch.allclose(y[:, :3], y2[:, :3], atol=1e-5)  # past predictions unchanged
    assert not torch.allclose(y[:, 3:], y2[:, 3:])


def test_sequence_length_one_is_stateless():
    torch.manual_seed(0)
    m = TrajectoryPredictor(d_in=8, n_targets=2, d_model=16, n_layers=1, backend="torch").eval()
    x = torch.randn(4, 1, 8)
    assert torch.allclose(
        m(x)[:, 0], torch.cat([m(x[i : i + 1])[:, 0] for i in range(4)]), atol=1e-6
    )


def test_can_overfit_tiny_trajectory_dataset():
    torch.manual_seed(0)
    m = TrajectoryPredictor(d_in=4, n_targets=1, d_model=32, n_layers=2, backend="torch")
    x = torch.randn(8, 6, 4)
    y = torch.sigmoid(x.cumsum(1).sum(-1, keepdim=True) / 3)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(200):
        loss = torch.nn.functional.mse_loss(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.01


def test_chunked_scan_matches_loop():
    from wsp.predictor.scan import selective_scan

    torch.manual_seed(4)
    u = torch.randn(2, 130, 8)
    dt = torch.rand_like(u) * 0.1
    a = -torch.rand(8, 4)
    b = torch.randn(2, 130, 4)
    c = torch.randn_like(b)
    d = torch.randn(8)
    h = torch.zeros(2, 8, 4)
    expected = []
    for t in range(u.shape[1]):
        h = (dt[:, t, :, None] * a).exp() * h + dt[:, t, :, None] * b[:, t, None, :] * u[
            :, t, :, None
        ]
        expected.append((h * c[:, t, None, :]).sum(-1) + d * u[:, t])
    torch.testing.assert_close(
        selective_scan(u, dt, a, b, c, d), torch.stack(expected, 1), atol=1e-4, rtol=1e-4
    )


def test_scan_backend_dispatch_and_gradients():
    from wsp.predictor.ssm_torch import SelectiveSSMBlock

    torch.manual_seed(42)
    loop = SelectiveSSMBlock(8, 4, backend="loop").double()
    chunked = SelectiveSSMBlock(8, 4, backend="chunked").double()
    chunked.load_state_dict(loop.state_dict())
    x = torch.randn(2, 65, 8, dtype=torch.float64, requires_grad=True)
    y = loop(x)
    z = chunked(x)
    torch.testing.assert_close(y, z)
    torch.testing.assert_close(
        torch.autograd.grad(y.sum(), x)[0], torch.autograd.grad(z.sum(), x)[0]
    )
    future = x.detach().clone()
    future[:, 64:] += 10
    torch.testing.assert_close(chunked(future)[:, :64], z[:, :64])


def test_default_backend_is_loop_when_mamba_unavailable(monkeypatch):
    import wsp.predictor.ssm_torch as module

    monkeypatch.setattr(module, "_Mamba2", None)
    assert module.TrajectoryPredictor(8, 2).backend == "loop"
