import pytest
import torch

from wsp.editor.api import EditorAPI


def test_batch_prefill_decode_hidden_and_over_budget():
    calls = []
    api = EditorAPI({"w": torch.ones(2, 2)}, lambda **kw: calls.append(kw) or {"logits": []}, 10)
    for batch in (4, 3, 2, 1):
        api.forward([[1]] * batch)
    assert api.used == 10
    with pytest.raises(ValueError):
        api.forward([[1]])
    assert len(calls) == 4


def test_failed_prelaunch_free_launched_failure_charged():
    def fail(**kw):
        raise RuntimeError("launched")

    api = EditorAPI({"w": torch.ones(2, 2)}, fail, 25)
    with pytest.raises(ValueError):
        api.forward([])
    assert api.used == 0
    with pytest.raises(RuntimeError):
        api.forward([[1]])
    assert api.used == 1


def test_copies_and_delta_validation():
    w = torch.ones(2, 2)
    api = EditorAPI({"w": w}, None, 50)
    api.read("w").zero_()
    assert w.sum() == 4
    for name, value in [("bad", w), ("w", torch.ones(1)), ("w", w * float("nan"))]:
        with pytest.raises(ValueError):
            api.set(name, value, "reason")
    with pytest.raises(ValueError):
        api.set("w", w, "")
    api.set("w", w * 0.1, "reason")
    assert api.preview()["w"]["norm"] > 0
