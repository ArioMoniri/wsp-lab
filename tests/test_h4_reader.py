"""H4 LLM-reader: input builder, prompt rendering, provider adapters (mocked), CLI and scoring.

Synthetic feature entries use the real on-disk layout of ``wsp.features.extract``; no network is used.
"""

import json
import math
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from wsp.agents import reader, score
from wsp.features.stats import DELTA_STAT_NAMES
from wsp.schema import BENCHMARKS

REAL = Path(__file__).resolve().parents[1] / "data" / "features" / "qwen3.5-0.8b"

NAMES = [
    "model.layers.3.mlp.down_proj.weight",
    "model.layers.3.self_attn.o_proj.weight",
    "model.layers.7.mlp.down_proj.weight",
    "model.layers.7.linear_attn.in_proj_qkv.weight",
    "model.layers.12.mlp.up_proj.weight",
    "model.embed_tokens.weight",
]


def make_entry(root: Path, entry_id: str = "e0", family: str = "abliteration", n: int = 6, seed=0):
    """Write features.json + features.safetensors for ``n`` tensors with distinct, known norms."""
    rng = np.random.default_rng(seed)
    d = root / entry_id
    d.mkdir(parents=True)
    names = (NAMES * (n // len(NAMES) + 1))[:n] if n <= len(NAMES) else None
    if names is None:
        names = [f"model.layers.{i}.mlp.down_proj.weight" for i in range(n)]
    stats = torch.tensor(
        rng.uniform(0.1, 2.0, size=(n, len(DELTA_STAT_NAMES))), dtype=torch.float32
    )
    stats[:, 0] = torch.tensor(rng.permutation(n) + 1.0)  # fro: distinct, shuffled
    save_file({"stats": stats, "svd": torch.zeros(n, 8, 129)}, str(d / "features.safetensors"))
    meta = {
        "edit_id": entry_id,
        "family": family,
        "params": {"direction_index": 9.1, "max_weight": 0.86},
        "parent_id": None,
        "base_model": "Qwen/Qwen3.5-0.8B",
        "n_layers": 24,
        "delta_relative_to": "base",
        "stat_names": list(DELTA_STAT_NAMES),
        "norms": {
            "fro_effective": 12.5,
            "rel_fro_effective": 0.021,
            "fro_base_changed": 105.9,
            "n_changed_tensors": n,
        },
        "rounding_stats": {
            "n_lost": 3,
            "n_nonzero": 1000,
            "fraction_nonzero_lost": 0.003,
            "max_abs_lost": 1.5e-5,
            "delta_norm": 12.4,
            "effective_delta_norm": 12.5,
        },
        "tensors": [{"name": nm, "row": i, "shape": [8, 8]} for i, nm in enumerate(names)],
        "unchanged_tensors": ["model.layers.0.mlp.up_proj.weight", "model.norm.weight"],
    }
    (d / "features.json").write_text(json.dumps(meta))
    return d, stats, names


@pytest.fixture
def feats(tmp_path):
    root = tmp_path / "features"
    _, stats, names = make_entry(root)
    return root, stats, names


GOOD = {
    "predictions": {b: 0.5 for b in BENCHMARKS},
    "mechanism": "tensors in layer 3 carry the edit",
    "confidence": 0.4,
}


# ---------------------------------------------------------------- input / prompt


def test_rows_sorted_by_fro_with_layer_and_type(feats):
    root, stats, names = feats
    entry = reader.load_entry(root, "e0")
    rows = reader.tensor_rows(entry)
    fro = [r["fro"] for r in rows]
    assert fro == sorted(fro, reverse=True)
    top = rows[0]
    assert top["name"] == names[int(torch.argmax(stats[:, 0]))]
    assert set(top) >= {"name", "layer", "type", "fro", "rel_fro", "cos", "sv", "stable_rank"}
    assert len(top["sv"]) == 3
    by_name = {r["name"]: r for r in rows}
    assert by_name["model.layers.3.mlp.down_proj.weight"]["layer"] == 3
    assert by_name["model.layers.3.mlp.down_proj.weight"]["type"] == "down"
    assert by_name["model.embed_tokens.weight"]["layer"] is None
    assert by_name["model.embed_tokens.weight"]["type"] == "embed"
    i = names.index("model.layers.7.mlp.down_proj.weight")
    assert by_name[names[i]]["stable_rank"] == pytest.approx(float(stats[i, 17]))
    assert by_name[names[i]]["sv"] == pytest.approx([float(x) for x in stats[i, 8:11]])


def test_global_summary_counts_layers_types_norms_rounding(feats):
    root, _, _ = feats
    s = reader.global_summary(reader.load_entry(root, "e0"))
    assert s["n_changed"] == 6 and s["n_unchanged"] == 2
    assert s["layers"] == [3, 7, 12]
    assert s["types"] == {"down": 2, "embed": 1, "lin_qkv": 1, "o": 1, "up": 1}
    assert s["fro_total"] == 12.5 and s["rel_fro_total"] == 0.021
    assert s["rounding"]["n_lost"] == 3


def test_render_is_deterministic_and_hash_stable(feats):
    root, _, _ = feats
    a = reader.build_prompt(reader.load_entry(root, "e0"))
    b = reader.build_prompt(reader.load_entry(root, "e0"))
    assert a.text == b.text and a.sha256 == b.sha256
    assert a.n_shown == a.n_total == 6
    # sections in the documented order
    order = [a.text.index(h) for h in ("## base", "## edit_summary", "## per_tensor_table")]
    assert order == sorted(order)
    assert a.text.startswith(reader.PROMPT_PATH.read_text().rstrip()[:40])
    assert "model.layers.3.mlp.down_proj.weight |" in a.text
    assert "showing all 6 changed tensors" in a.text


def test_family_label_hidden_by_default_and_revealed_on_opt_in(feats):
    root, _, _ = feats
    entry = reader.load_entry(root, "e0")
    hidden = reader.build_prompt(entry).text
    assert "abliteration" not in hidden
    assert "direction_index" not in hidden and "max_weight" not in hidden
    assert "## family" not in hidden
    shown = reader.build_prompt(entry, reveal_family=True).text
    assert "## family" in shown and "abliteration" in shown
    assert "direction_index" in shown
    assert (
        reader.build_prompt(entry, reveal_family=True).sha256 != reader.build_prompt(entry).sha256
    )


def test_truncation_keeps_top_n_and_states_n(tmp_path):
    root = tmp_path / "f"
    _, stats, names = make_entry(root, "big", n=60)
    entry = reader.load_entry(root, "big")
    full = reader.build_prompt(entry)
    assert full.n_shown == 60
    budget = full.tokens - 400
    cut = reader.build_prompt(entry, max_prompt_tokens=budget)
    assert 0 < cut.n_shown < 60 and cut.n_total == 60
    assert cut.tokens <= budget
    assert f"top {cut.n_shown} of 60 changed tensors" in cut.text
    top = [names[i] for i in torch.argsort(stats[:, 0], descending=True)[: cut.n_shown].tolist()]
    listed = [nm for nm in names if f"\n{nm} |" in cut.text]
    assert sorted(listed) == sorted(top)
    # one more tensor would not fit the budget
    again = reader.build_prompt(entry, max_prompt_tokens=cut.tokens)
    assert again.n_shown == cut.n_shown
    # global summary is computed from all tensors, not only the shown ones
    assert "changed tensors: 60" in cut.text


def test_truncation_impossible_budget_raises(feats):
    root, _, _ = feats
    with pytest.raises(ValueError, match="token budget"):
        reader.build_prompt(reader.load_entry(root, "e0"), max_prompt_tokens=50)


def test_load_entry_rejects_unsafe_or_missing_ids(feats):
    root, _, _ = feats
    with pytest.raises(ValueError):
        reader.load_entry(root, "../e0")
    with pytest.raises(FileNotFoundError):
        reader.load_entry(root, "nope")


# ---------------------------------------------------------------- parsing


def test_parse_valid_response():
    out = reader.parse_response(json.dumps(GOOD))
    assert out["predictions"]["gsm8k"] == 0.5 and out["confidence"] == 0.4


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "```json\n" + json.dumps(GOOD) + "\n```",
        "Here you go: " + json.dumps(GOOD),
        json.dumps({k: v for k, v in GOOD.items() if k != "mechanism"}),
        json.dumps(GOOD | {"extra": 1}),
        json.dumps(GOOD | {"confidence": 1.5}),
        json.dumps(GOOD | {"predictions": GOOD["predictions"] | {"gsm8k": "3"}}),
        json.dumps(GOOD | {"predictions": GOOD["predictions"] | {"gsm8k": True}}),
        json.dumps(GOOD | {"predictions": {k: 1 for k in BENCHMARKS if k != "ifeval"}}),
        json.dumps(GOOD | {"predictions": GOOD["predictions"] | {"made_up": 1}}),
        json.dumps(GOOD | {"predictions": GOOD["predictions"] | {"kl_to_base": -0.1}}),
        json.dumps(GOOD).replace("0.4", "NaN"),
        json.dumps(["list"]),
        "",
    ],
)
def test_parse_invalid_response_raises(text):
    with pytest.raises(reader.ParseError):
        reader.parse_response(text)


# ---------------------------------------------------------------- adapters (mocked)

SECRET = "sk-test-SECRET-123456"


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_adapter_request_shape_and_usage(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": json.dumps(GOOD)}],
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        )

    env = {
        "ANTHROPIC_API_KEY": SECRET,
        "ANTHROPIC_CUSTOM_HEADERS": "x-team: wsp\nanthropic-beta: foo-1",
        "ANTHROPIC_BASE_URL": "https://gw.example/anthropic/",
    }
    p = reader.make_provider("anthropic", "claude-fable-5-1", env=env, client=_client(handler))
    assert p.ask("PROMPT") == json.dumps(GOOD)
    assert seen["url"] == "https://gw.example/anthropic/v1/messages"
    assert seen["headers"]["x-api-key"] == SECRET
    assert seen["headers"]["x-team"] == "wsp" and seen["headers"]["anthropic-beta"] == "foo-1"
    assert "anthropic-version" in seen["headers"]
    assert seen["body"]["model"] == "claude-fable-5-1"
    assert seen["body"]["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert seen["body"]["max_tokens"] > 0
    assert p.last_usage == {"input_tokens": 11, "output_tokens": 7}
    assert SECRET not in repr(p)


def test_openai_adapter_request_shape_base_url_and_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": json.dumps(GOOD)}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 9},
            },
        )

    env = {"OPENAI_API_KEY": SECRET}
    p = reader.make_provider(
        "openai", "gpt-6-astra", base_url="https://oa.example/v1", env=env, client=_client(handler)
    )
    assert p.ask("PROMPT") == json.dumps(GOOD)
    assert seen["url"] == "https://oa.example/v1/chat/completions"
    assert seen["auth"] == f"Bearer {SECRET}"
    assert seen["body"]["model"] == "gpt-6-astra"
    assert seen["body"]["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert p.last_usage == {"input_tokens": 20, "output_tokens": 9}
    assert SECRET not in repr(p)


def test_missing_key_is_an_error_naming_the_variable_only(tmp_path):
    with pytest.raises(reader.ProviderError, match="ANTHROPIC_API_KEY"):
        reader.make_provider("anthropic", "m", env={}, env_file=tmp_path / "none")
    with pytest.raises(reader.ProviderError, match="OPENAI_API_KEY"):
        reader.make_provider("openai", "m", env={}, env_file=tmp_path / "none")


def test_key_read_from_env_file(tmp_path):
    f = tmp_path / "env"
    f.write_text(f"# c\nexport ANTHROPIC_API_KEY='{SECRET}'\nOTHER=1\n")
    p = reader.make_provider(
        "anthropic", "m", env={}, env_file=f, client=_client(lambda r: httpx.Response(500))
    )
    assert isinstance(p, reader.AnthropicProvider)
    # the process environment wins over the file
    p2 = reader.make_provider(
        "anthropic",
        "m",
        env={"ANTHROPIC_API_KEY": "other-key"},
        env_file=f,
        client=_client(lambda r: httpx.Response(500)),
    )
    assert p2._key == "other-key"


def test_http_error_message_never_contains_the_key():
    def handler(request):
        return httpx.Response(401, text=f"bad key {SECRET} for request")

    p = reader.make_provider(
        "openai", "m", env={"OPENAI_API_KEY": SECRET}, client=_client(handler), sleep=lambda s: None
    )
    with pytest.raises(reader.ProviderError) as e:
        p.ask("x")
    assert SECRET not in str(e.value) and "401" in str(e.value)


def test_retries_on_429_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    p = reader.make_provider(
        "openai", "m", env={"OPENAI_API_KEY": "k"}, client=_client(handler), sleep=lambda s: None
    )
    assert p.ask("x") == "ok" and len(calls) == 3


def test_empty_or_malformed_provider_payload_is_provider_error():
    p = reader.make_provider(
        "anthropic",
        "m",
        env={"ANTHROPIC_API_KEY": "k"},
        client=_client(lambda r: httpx.Response(200, json={"content": []})),
    )
    with pytest.raises(reader.ProviderError):
        p.ask("x")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="provider"):
        reader.make_provider("nope", "m", env={})


def test_dry_provider_returns_nothing_and_never_calls_network():
    p = reader.make_provider("dry", "any")
    assert isinstance(p, reader.DryProvider)
    with pytest.raises(reader.DryRun):
        p.ask("x")


# ---------------------------------------------------------------- CLI run


class FakeProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.last_usage = {"input_tokens": 100, "output_tokens": 10}

    def ask(self, prompt):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_run_writes_per_entry_files_and_manifest(tmp_path):
    root = tmp_path / "f"
    for i in ("a", "b", "c"):
        make_entry(root, i, seed=ord(i))
    out = tmp_path / "run"
    fake = FakeProvider([json.dumps(GOOD), "garbage", reader.ProviderError("HTTP 500")])
    m = reader.run_reader(
        root,
        ["a", "b", "c"],
        out,
        fake,
        provider_name="openai",
        model="gpt-6-astra",
        price_in_per_mtok=2.0,
        price_out_per_mtok=8.0,
    )
    for i in ("a", "b", "c"):
        assert (out / i / "prompt.md").read_text() == fake.prompts[("abc".index(i))]
    assert json.loads((out / "a" / "parsed.json").read_text())["status"] == "ok"
    assert json.loads((out / "a" / "response.json").read_text())["text"] == json.dumps(GOOD)
    pb = json.loads((out / "b" / "parsed.json").read_text())
    assert pb["status"] == "failed" and "error" in pb and "parsed" not in pb
    pc = json.loads((out / "c" / "parsed.json").read_text())
    assert pc["status"] == "failed" and "HTTP 500" in pc["error"]
    assert not (out / "c" / "response.json").exists()
    man = json.loads((out / "manifest.json").read_text())
    assert m["n_ok"] == 1
    assert man["model"] == "gpt-6-astra" and man["provider"] == "openai"
    assert man["n_ok"] == 1 and man["n_failed"] == 2
    assert man["reveal_family"] is False
    assert len(man["prompt_template_sha256"]) == 64 and man["git_sha"]
    assert man["tokens"]["input_reported"] == 200 and man["tokens"]["output_reported"] == 20
    assert man["tokens"]["input_estimated"] > 0
    assert man["cost_estimate_usd"] == pytest.approx((200 * 2.0 + 20 * 8.0) / 1e6)
    assert man["entries"]["a"]["prompt_sha256"] and man["entries"]["a"]["n_shown"] == 6


def test_run_without_prices_has_no_cost_estimate(tmp_path):
    root = tmp_path / "f"
    make_entry(root, "a")
    m = reader.run_reader(
        root, ["a"], tmp_path / "r", FakeProvider([json.dumps(GOOD)]), provider_name="x", model="m"
    )
    assert m["cost_estimate_usd"] is None


def test_run_resumes_without_repeating_paid_calls(tmp_path):
    root = tmp_path / "f"
    make_entry(root, "a")
    make_entry(root, "b", seed=3)
    out = tmp_path / "r"
    first = FakeProvider([json.dumps(GOOD), reader.ProviderError("HTTP 500")])
    reader.run_reader(root, ["a", "b"], out, first, provider_name="x", model="m")
    second = FakeProvider([json.dumps(GOOD)])
    m = reader.run_reader(root, ["a", "b"], out, second, provider_name="x", model="m")
    assert len(second.prompts) == 1  # only the failed entry is asked again
    assert m["n_ok"] == 2 and m["n_failed"] == 0


def test_dry_run_writes_prompts_only(tmp_path):
    root = tmp_path / "f"
    make_entry(root, "a")
    out = tmp_path / "r"
    m = reader.run_reader(
        root, ["a"], out, reader.make_provider("dry", "m"), provider_name="dry", model="m"
    )
    assert (out / "a" / "prompt.md").exists()
    assert not (out / "a" / "response.json").exists() and not (out / "a" / "parsed.json").exists()
    assert m["n_ok"] == 0 and m["n_failed"] == 0 and m["n_dry"] == 1
    assert json.loads((out / "manifest.json").read_text())["provider"] == "dry"


def test_cli_dry(tmp_path, capsys):
    root = tmp_path / "f"
    make_entry(root, "a")
    make_entry(root, "b", seed=2)
    out = tmp_path / "r"
    reader.main(
        [
            "--features",
            str(root),
            "--ids",
            "a,b",
            "--provider",
            "dry",
            "--model",
            "m",
            "--out",
            str(out),
        ]
    )
    assert (out / "a" / "prompt.md").exists() and (out / "b" / "prompt.md").exists()
    assert (out / "manifest.json").exists()


def test_cli_rejects_unknown_id_before_writing(tmp_path):
    root = tmp_path / "f"
    make_entry(root, "a")
    with pytest.raises(SystemExit):
        reader.main(
            [
                "--features",
                str(root),
                "--ids",
                "a,zzz",
                "--provider",
                "dry",
                "--model",
                "m",
                "--out",
                str(tmp_path / "r"),
            ]
        )
    assert not (tmp_path / "r").exists()


@pytest.mark.skipif(
    not REAL.exists() or len(list(REAL.glob("*/features.json"))) < 2, reason="no real features"
)
def test_dry_run_on_two_real_entries(tmp_path):
    ids = sorted(p.parent.name for p in REAL.glob("*/features.json"))[:2]
    out = tmp_path / "run"
    reader.main(
        [
            "--features",
            str(REAL),
            "--ids",
            ",".join(ids),
            "--provider",
            "dry",
            "--model",
            "m",
            "--out",
            str(out),
        ]
    )
    for i in ids:
        text = (out / i / "prompt.md").read_text()
        meta = json.loads((REAL / i / "features.json").read_text())
        assert meta["family"] not in text
        assert "## per_tensor_table" in text and "model.layers." in text
        assert reader.estimate_tokens(text) <= reader.DEFAULT_MAX_PROMPT_TOKENS


# ---------------------------------------------------------------- scoring


def _scored_run(tmp_path, with_evals=True, n=12):
    root = tmp_path / "f"
    out = tmp_path / "run"
    evals = tmp_path / "evals"
    ids = [f"e{i:02d}" for i in range(n)]
    rng = np.random.default_rng(0)
    base = {"gsm8k": 0.10, "wikitext_ppl": 20.0, "kl_to_base": 0.0, "loop_rate": 0.5}
    (evals / "base").mkdir(parents=True)
    (evals / "base" / "eval.json").write_text(json.dumps(base))
    truth = {}
    for i, e in enumerate(ids):
        make_entry(root, e, seed=i)
        d = out / e
        d.mkdir(parents=True)
        dgsm = float(rng.uniform(-0.05, 0.05))
        truth[e] = dgsm
        if with_evals:
            (evals / e).mkdir()
            (evals / e / "eval.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "gsm8k": base["gsm8k"] + dgsm,
                            "wikitext_ppl": base["wikitext_ppl"] * 1.1,
                            "kl_to_base": 0.02,
                            "loop_rate": 0.5,
                        }
                    }
                )
            )
        preds = {b: 0.0 for b in BENCHMARKS}
        preds["gsm8k"] = 100 * dgsm + 1.0  # constant +1 pp error
        preds["wikitext_ppl"] = 10.0
        preds["kl_to_base"] = 0.05
        (d / "parsed.json").write_text(
            json.dumps({"status": "ok", "parsed": GOOD | {"predictions": preds}})
        )
    (out / "e00" / "parsed.json").write_text(json.dumps({"status": "failed", "error": "bad"}))
    (out / "manifest.json").write_text(json.dumps({"ids": ids, "model": "m"}))
    return out, evals


def test_score_computes_metrics_in_reader_units(tmp_path):
    out, evals = _scored_run(tmp_path)
    res = score.score_run(out, evals, n_boot=200)
    g = res["targets"]["gsm8k"]
    assert g["n"] == 11
    assert g["mae"]["value"] == pytest.approx(1.0)
    assert g["rmse"]["value"] == pytest.approx(1.0)
    assert 0.9 < g["spearman"]["value"] <= 1.0
    assert len(g["mae"]["ci95"]) == 2
    assert res["targets"]["wikitext_ppl"]["mae"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert res["targets"]["kl_to_base"]["mae"]["value"] == pytest.approx(0.03)
    assert res["n_entries"] == 12 and res["n_failed"] == 1 and res["n_scored"] == 11
    assert "hellaswag" in res["skipped"]  # no target in any eval.json
    assert json.loads((out / "score.json").read_text())["n_scored"] == 11


def test_score_refuses_without_targets(tmp_path):
    out, evals = _scored_run(tmp_path, with_evals=False)
    with pytest.raises(score.NoTargets):
        score.score_run(out, evals)
    with pytest.raises(SystemExit):
        score.main(["--run", str(out), "--evals", str(evals)])
    assert not (out / "score.json").exists()


def test_score_refuses_without_base_eval(tmp_path):
    out, evals = _scored_run(tmp_path)
    (evals / "base" / "eval.json").unlink()
    with pytest.raises(score.NoTargets, match="base"):
        score.score_run(out, evals)


def test_score_refuses_empty_run(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(score.NoTargets):
        score.score_run(tmp_path / "run", tmp_path / "evals")


def test_to_reader_units():
    base = {"gsm8k": 0.2, "wikitext_ppl": 10.0, "kl_to_base": 0.0, "loop_rate": 0.1}
    cur = {"gsm8k": 0.25, "wikitext_ppl": 12.0, "kl_to_base": 0.03, "loop_rate": 0.05}
    out = score.to_reader_units(cur, base)
    assert out["gsm8k"] == pytest.approx(5.0)
    assert out["wikitext_ppl"] == pytest.approx(20.0)
    assert out["kl_to_base"] == pytest.approx(0.03)
    assert out["loop_rate"] == pytest.approx(-5.0)
    assert math.isfinite(sum(out.values()))
