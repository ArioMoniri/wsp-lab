import pytest
import torch


@pytest.fixture(scope="session")
def toy_weights():
    """Toy 2-layer 'transformer' weights: dict name -> tensor (~50k params), deterministic."""
    g = torch.Generator().manual_seed(0)
    d, ff = 32, 64
    w = {}
    for layer in range(2):
        w[f"layers.{layer}.attn.o_proj"] = torch.randn(d, d, generator=g) * 0.05
        w[f"layers.{layer}.mlp.down_proj"] = torch.randn(d, ff, generator=g) * 0.05
        w[f"layers.{layer}.mlp.up_proj"] = torch.randn(ff, d, generator=g) * 0.05
    return w


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return torch.device("cuda")
