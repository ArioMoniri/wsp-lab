import pytest
import torch

# Tiny fixtures run faster without oversubscribing the shared build host.
torch.set_num_threads(min(2, torch.get_num_threads()))


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


@pytest.fixture
def toy_model(toy_weights):
    """Executable version of the shared toy weights, suitable for real PEFT updates."""
    from types import SimpleNamespace

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import PretrainedConfig

            self.config = PretrainedConfig(use_cache=True)
            self.layers = torch.nn.ModuleList()
            for i in range(2):
                layer = torch.nn.ModuleDict()
                for group, names in {"attn": ["o_proj"], "mlp": ["up_proj", "down_proj"]}.items():
                    layer[group] = torch.nn.ModuleDict()
                    for name in names:
                        w = toy_weights[f"layers.{i}.{group}.{name}"]
                        module = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
                        module.weight.data.copy_(w)
                        layer[group][name] = module
                self.layers.append(layer)

        def forward(self, input_ids, attention_mask=None, labels=None):
            h = torch.nn.functional.one_hot(input_ids % 32, 32).float()
            for layer in self.layers:
                h = h + layer["attn"]["o_proj"](h)
                h = h + layer["mlp"]["down_proj"](layer["mlp"]["up_proj"](h).tanh())
            loss = torch.nn.functional.cross_entropy(
                h[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1), ignore_index=-100
            )
            return SimpleNamespace(loss=loss)

    return ToyModel()
