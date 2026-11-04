import pytest
import torch

from wsp.predictor.ssm_torch import TrajectoryPredictor, _Mamba2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or _Mamba2 is None, reason="CUDA Mamba unavailable")
def test_mamba_predictor_causal_and_deterministic():
    torch.manual_seed(42)
    model = TrajectoryPredictor(8, 3, d_model=32, n_layers=1, backend="mamba_ssm").cuda().eval()
    x = torch.randn(2, 12, 8, device="cuda")
    with torch.no_grad():
        y = model(x)
        repeated = model(x)
        changed = x.clone()
        changed[:, 8:] += 10
        future = model(changed)
        loop = TrajectoryPredictor(8, 3, d_model=32, n_layers=1, backend="loop").cuda()(x)
    assert y.shape == loop.shape == (2, 12, 3)
    assert torch.all((y >= 0) & (y <= 1))
    torch.testing.assert_close(y, repeated, rtol=0, atol=0)
    torch.testing.assert_close(y[:, :8], future[:, :8])
