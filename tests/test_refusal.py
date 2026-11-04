"""Refusal-direction extraction: Heretic prompt/residual parity and cache identity (CPU)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import wsp.edits.refusal as R
from wsp.edits.kernels import heretic_abliteration_kernel
from wsp.edits.refusal import interpolate_direction


def test_interpolation_and_full_ablation():
    dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    d = interpolate_direction(dirs, -0.5)
    assert torch.allclose(d, torch.tensor([2**-0.5, 2**-0.5]))
    w = torch.randn(2, 4)
    result, _ = heretic_abliteration_kernel({"w": w}, dirs, {0: ["w"]}, -0.5, 1.0, 0.0, 0.0, 1.0)
    assert torch.allclose(d @ result["w"], torch.zeros(4), atol=1e-6)


# ---- fake tokenizer ---------------------------------------------------------------------------


class FakeTokenizer:
    """ChatML-like template. `system_ok=False` mimics Gemma-2 templates that reject system."""

    def __init__(self, system_ok=True, thinking="closed", template="tmpl-v1"):
        self.system_ok = system_ok
        self.thinking = thinking
        self.chat_template = template
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self.calls = []

    def _render(self, chat, strftime_now):
        out = ""
        for m in chat:
            if m["role"] == "system" and not self.system_ok:
                raise ValueError("System role not supported")
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if strftime_now:
            out += f"[{strftime_now('%Y-%m-%d')}]"
        out += "<|im_start|>assistant\n"
        if self.thinking == "closed":
            out += "<think>\n\n</think>\n\n"
        elif self.thinking == "open":
            out += "<think>\n"
        return out

    def apply_chat_template(self, chats, add_generation_prompt, tokenize, strftime_now=None):
        assert add_generation_prompt and not tokenize
        if chats and isinstance(chats[0], dict):
            return self._render(chats, strftime_now)
        return [self._render(c, strftime_now) for c in chats]

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()


def test_system_role_probe_and_folding():
    assert R.chat_template_supports_system_role(FakeTokenizer(system_ok=True))
    assert not R.chat_template_supports_system_role(FakeTokenizer(system_ok=False))
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    assert R.apply_system_role(msgs, True) == msgs
    assert R.apply_system_role(msgs, False) == [{"role": "user", "content": "SYS\n\nhi"}]


def _heretic_prompt_texts(tok, prefix):
    """Drive heretic.model.Model.generate with a fake model and capture the tokenizer input."""
    from heretic.model import Model

    seen: dict[str, list[str]] = {}

    class Enc:
        def to(self, _):
            return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    class CallableTok:
        apply_chat_template = staticmethod(tok.apply_chat_template)
        pad_token_id = 0

        def __call__(self, texts, **kw):
            seen["texts"] = list(texts)
            return Enc()

    fake = SimpleNamespace(
        tokenizer=CallableTok(),
        supports_system_role=R.chat_template_supports_system_role(tok),
        settings=SimpleNamespace(response_prefix=prefix or None),
        model=SimpleNamespace(device="cpu", generate=lambda **kw: None),
    )
    fake._apply_system_role = lambda m: Model._apply_system_role(fake, m)
    prompt = SimpleNamespace(system=R.DEFAULT_SYSTEM_PROMPT, user="how do I bake?")
    Model.generate(fake, [prompt], max_new_tokens=1)
    return seen["texts"]


def test_prompt_text_matches_heretic_generate():
    """Our formatted prompt text equals what heretic.model.Model.generate feeds the tokenizer."""
    for tok in (FakeTokenizer(), FakeTokenizer(system_ok=False), FakeTokenizer(thinking="open")):
        for prefix in ("", "</think>"):
            ours = R.format_prompts(tok, ["how do I bake?"], response_prefix=prefix)
            assert ours == _heretic_prompt_texts(tok, prefix)


def test_formatted_prompt_text_has_system_user_fixed_date_and_prefix():
    tok = FakeTokenizer(thinking="open")
    text = R.format_prompts(tok, ["Tell me X"], response_prefix="</think>")[0]
    assert text == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nTell me X<|im_end|>\n"
        "[2026-01-01]<|im_start|>assistant\n<think>\n</think>"
    )


def test_response_prefix_detection_follows_heretic_cot_logic():
    assert R.detect_response_prefix(FakeTokenizer(thinking="closed"), True) == (
        "",
        "template-closed-cot-block",
    )
    assert R.detect_response_prefix(FakeTokenizer(thinking="open"), True) == (
        "</think>",
        "template-cot-initializer",
    )
    assert R.detect_response_prefix(FakeTokenizer(thinking="none"), True)[0] is None


# ---- residual position ------------------------------------------------------------------------


def test_residual_is_last_token_of_every_layer_incl_embedding_with_left_padding():
    n_layers, d = 3, 4
    ids = torch.tensor([[0, 0, 5, 6], [1, 2, 3, 4]])  # row 0 is left-padded
    hidden = tuple(
        ids.float().unsqueeze(-1).expand(2, 4, d) * (layer + 1) for layer in range(n_layers + 1)
    )
    res = R.last_token_residuals(hidden)
    assert res.shape == (2, n_layers + 1, d) and res.dtype == torch.float32
    assert torch.equal(res[0, :, 0], torch.tensor([6.0, 12.0, 18.0, 24.0]))
    assert torch.equal(res[1, :, 0], torch.tensor([4.0, 8.0, 12.0, 16.0]))


def test_residual_mean_uses_prefill_hidden_states_float64_mean():
    d = 4

    class Tok:
        def __call__(self, texts, return_tensors, padding, return_token_type_ids):
            width = max(len(t) for t in texts)
            ids = torch.stack(
                [torch.tensor([0] * (width - len(t)) + [ord(c) for c in t]) for t in texts]
            )
            return SimpleNamespace(to=lambda dev: {"input_ids": ids})

    class Model:
        device = "cpu"

        def __call__(self, input_ids, output_hidden_states, use_cache):
            assert output_hidden_states and not use_cache
            h = input_ids.float().unsqueeze(-1).expand(*input_ids.shape, d)
            return SimpleNamespace(hidden_states=(h, h * 2))

    mean = R.residual_mean(Model(), Tok(), ["ab", "c"], batch_size=2)
    assert mean.dtype == torch.float32
    assert mean[0, 0].item() == pytest.approx((ord("b") + ord("c")) / 2)
    assert mean[1, 0].item() == pytest.approx(ord("b") + ord("c"))


def test_directions_are_unit_and_orthogonal_to_good_mean():
    g = torch.Generator().manual_seed(1)
    good, bad = torch.randn(5, 8, generator=g), torch.randn(5, 8, generator=g)
    dirs = R.directions_from_means(good, bad)
    assert torch.allclose(dirs.norm(dim=1), torch.ones(5), atol=1e-6)
    assert torch.allclose(
        (dirs * torch.nn.functional.normalize(good, dim=1)).sum(1), torch.zeros(5), atol=1e-6
    )


def test_prompt_selection_all_by_default_and_seeded_subsample_otherwise():
    assert R.select_prompt_indices(400, 400, 0) == list(range(400))
    assert R.select_prompt_indices(400, 9999, 0) == list(range(400))
    sub = R.select_prompt_indices(400, 8, 5)
    assert sub == R.select_prompt_indices(400, 8, 5) and len(set(sub)) == 8
    assert sub != R.select_prompt_indices(400, 8, 6)


# ---- cache identity ---------------------------------------------------------------------------


@pytest.fixture
def cache_env(monkeypatch, tmp_path):
    import datasets
    import transformers

    state = {"computed": 0, "commits": {"model": "m1", "good": "g1", "bad": "b1"}}
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", FakeTokenizer.from_pretrained
    )
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda name, split, revision: {"text": [f"{name}-{revision}-{i}" for i in range(12)]},
    )

    def fake_compute(config, tokenizer, key, prompts):
        state["computed"] += 1
        state["last_prompts"] = prompts
        return torch.nn.functional.normalize(torch.randn(5, 4), dim=1)

    monkeypatch.setattr(R, "_compute", fake_compute)

    def resolver(repo, kind, rev):
        return state["commits"][
            "model" if kind == "model" else "good" if "harmless" in repo else "bad"
        ]

    def run(**overrides):
        config = {"base_model": "org/model", "seed": 0, "dtype": "bfloat16", "refusal_n": 4}
        config.update(overrides)
        return R.get_refusal_directions(config, tmp_path, "cfghash", resolver)

    state["run"] = run
    return state


def test_cache_hit_on_identical_inputs(cache_env):
    a = cache_env["run"]()
    b = cache_env["run"]()
    assert not a.cache_hit and b.cache_hit and a.fingerprint == b.fingerprint
    assert cache_env["computed"] == 1
    assert torch.equal(a.directions, b.directions)


@pytest.mark.parametrize(
    "override",
    [
        {"dtype": "float16"},
        {"refusal_n": 5},
        {"refusal_seed": 7},
        {"refusal_system_prompt": "Another system prompt."},
        {"refusal_response_prefix": "</think>"},
    ],
)
def test_cache_invalidated_by_config_inputs(cache_env, override):
    first = cache_env["run"]()
    second = cache_env["run"](**override)
    assert not second.cache_hit and second.fingerprint != first.fingerprint
    assert second.manifest_path != first.manifest_path
    assert cache_env["computed"] == 2


@pytest.mark.parametrize("which", ["model", "good", "bad"])
def test_cache_invalidated_by_resolved_commit_drift(cache_env, which):
    first = cache_env["run"]()
    cache_env["commits"][which] = "drifted"
    second = cache_env["run"]()
    assert not second.cache_hit and second.fingerprint != first.fingerprint


def test_manifest_records_identity_and_prompts(cache_env):
    out = cache_env["run"]()
    m = json.loads(out.manifest_path.read_text())
    key = m["key"]
    assert key["dtype"] == "bfloat16" and key["n"] == 4 and key["seed"] == 0
    assert key["model_commit"] == "m1" and key["good_dataset_commit"] == "g1"
    assert key["bad_dataset_commit"] == "b1"
    assert key["good_dataset"] == R.GOOD and key["bad_dataset"] == R.BAD
    assert key["system_prompt"] == R.DEFAULT_SYSTEM_PROMPT and key["response_prefix"] == ""
    assert m["fingerprint"] == out.fingerprint == R.fingerprint(key)
    assert m["config_hash"] == "cfghash" and len(m["good_prompt_indices"]) == 4
    assert m["response_prefix_source"] == "template-closed-cot-block"
    assert "You are a helpful assistant." in m["example_formatted_prompt"]
    # datasets are fetched at the pinned commit
    assert "harmful_behaviors-b1" in cache_env["last_prompts"]["bad"][0]


def test_tampered_direction_file_or_manifest_triggers_reextraction(cache_env):
    first = cache_env["run"]()
    path = first.manifest_path.parent / first.manifest_path.name.replace(".manifest.json", ".pt")
    torch.save(torch.zeros(5, 4), path)  # same filename, different content
    second = cache_env["run"]()
    assert not second.cache_hit and cache_env["computed"] == 2
    manifest = json.loads(second.manifest_path.read_text())
    manifest["key"]["n"] = 999
    second.manifest_path.write_text(json.dumps(manifest))
    third = cache_env["run"]()
    assert not third.cache_hit and cache_env["computed"] == 3


def test_missing_response_prefix_is_an_error_not_a_guess(monkeypatch, cache_env):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTokenizer(thinking="none")
    )
    with pytest.raises(ValueError, match="refusal_response_prefix"):
        cache_env["run"]()
    assert cache_env["run"](refusal_response_prefix="").fingerprint


@pytest.mark.gpu
@pytest.mark.slow
def test_extraction_and_abliteration_roundtrip(cuda, tmp_path):
    from pathlib import Path

    from wsp.edits.refusal import extract_directions
    from wsp.zoo.apply import load_edited_model
    from wsp.zoo.build import _load_base_state, _write_entry, build_abliteration, load_config

    config = load_config(Path("configs/zoo_qwen3.5-0.8b.yaml"))
    config["refusal_n"] = 8
    directions = extract_directions(config, tmp_path)
    assert directions.ndim == 2 and torch.isfinite(directions).all()
    base = _load_base_state(config)
    desc, state, touched = build_abliteration(
        config, config["families"]["abliteration"], 0, base, tmp_path
    )[0]
    delta = {k: state[k] - base[k] for k in touched}
    _write_entry(tmp_path, desc, delta, torch.bfloat16)
    restored = load_edited_model(base, tmp_path / desc.id)
    assert all(torch.equal(restored[k], base[k] + delta[k]) for k in touched)


def test_real_qwen_template_prompt_if_cached():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    assert R.detect_response_prefix(tok, R.chat_template_supports_system_role(tok))[0] == ""
    text = R.format_prompts(tok, ["Tell me X"])[0]
    assert text.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
    assert "<|im_start|>user\nTell me X<|im_end|>\n<|im_start|>assistant\n" in text
    assert text.endswith("</think>\n\n")
