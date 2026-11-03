import pytest
import torch

from wsp.predictor import TrajectoryPredictor

pytestmark = pytest.mark.gpu


def test_predictor_runs_on_cuda(cuda):
    m = TrajectoryPredictor(d_in=16, n_targets=4, d_model=64, n_layers=2).to(cuda)
    y = m(torch.randn(3, 7, 16, device=cuda))
    assert y.shape == (3, 7, 4) and y.device.type == "cuda"
