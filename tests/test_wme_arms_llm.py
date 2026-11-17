"""wm.arms.llm: thin, offline-mockable LLM clients over the delegate providers (Claude, Codex, Gemini via OpenCode),
token accounting under a hard cap, and the truncation-is-failure rule. No network: every CLI call goes through an
injected runner."""

import json
import os

import pytest

from wsp.wm.arms import llm as L


def test_mock_client_counts_tokens_and_truncates():
    m = L.MockClient(["x" * 100, "y" * 10_000], overhead_tokens=7)
    c = m.complete("sys", "user prompt", max_output_tokens=200)
    assert c.text == "x" * 100 and not c.truncated and c.error is None
    assert c.input_tokens == L.estimate_tokens("sys") + L.estimate_tokens("user prompt") + 7
    assert c.output_tokens == L.estimate_tokens("x" * 100)
    assert c.usage_source == "estimate"
    t = m.complete("sys", "u", max_output_tokens=50)
    assert t.truncated and t.output_tokens == 50 and t.stop_reason == "max_tokens"
    assert m.calls == [("sys", "user prompt"), ("sys", "u")]


def test_mock_client_callable_and_exhaustion():
    m = L.MockClient(lambda s, u: f"echo:{u}")
    assert m.complete("s", "hi", 100).text == "echo:hi"
    m2 = L.MockClient([])
    c = m2.complete("s", "u", 10)
    assert c.error == "mock exhausted" and c.text == ""


class Runner:
    def __init__(self, stdout, rc=0, stderr=""):
        self.stdout, self.rc, self.stderr, self.calls = stdout, rc, stderr, []

    def __call__(self, argv, stdin, env, cwd, timeout):
        self.calls.append(
            {"argv": argv, "stdin": stdin, "env": env, "cwd": cwd, "listing": os.listdir(cwd)}
        )
        return self.rc, self.stdout, self.stderr


CLAUDE_OUT = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": '{"ranking": ["c01", "c00"]}',
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 120,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 50,
            "output_tokens": 12,
        },
    }
)


def test_claude_cli_argv_env_and_parse(monkeypatch):
    for k, v in {
        "GIT_DIR": "/x",
        "GIT_INDEX_FILE": "/y",
        "HF_TOKEN": "hf_secret",
        "LAMBDA_API_KEY": "k",
        "CLAUDECODE": "1",
        "CLAUDE_CODE_SESSION_ID": "s",
    }.items():
        monkeypatch.setenv(k, v)
    r = Runner(CLAUDE_OUT)
    c = L.ClaudeCLI("claude-haiku-5-5", runner=r).complete("SYSTEM", "USER", max_output_tokens=256)
    call = r.calls[0]
    a = call["argv"]
    assert a[0] == "claude" and "-p" in a
    assert a[a.index("--model") + 1] == "claude-haiku-5-5"
    assert a[a.index("--system-prompt") + 1] == "SYSTEM"
    assert a[a.index("--tools") + 1] == ""
    assert a[a.index("--output-format") + 1] == "json"
    assert "--no-session-persistence" in a and "--strict-mcp-config" in a
    assert call["stdin"] == "USER"
    env = call["env"]
    assert not any(k.startswith("GIT_") for k in env)
    assert "HF_TOKEN" not in env and "LAMBDA_API_KEY" not in env and "CLAUDECODE" not in env
    assert "CLAUDE_CODE_SESSION_ID" not in env
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "256"
    # no CLAUDE.md or auto-memory may enter a zero-shot reader's context
    assert (
        env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
        and env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    )
    assert call["listing"] == []  # an empty scratch working directory: no repository files to read
    assert c.text == '{"ranking": ["c01", "c00"]}'
    assert c.input_tokens == 200 and c.output_tokens == 12 and c.usage_source == "provider"
    assert not c.truncated and c.error is None and c.provider == "claude"


def test_claude_cli_truncation_and_error():
    out = json.loads(CLAUDE_OUT)
    out["stop_reason"] = "max_tokens"
    c = L.ClaudeCLI("m", runner=Runner(json.dumps(out))).complete("s", "u", 64)
    assert c.truncated
    out["stop_reason"] = "end_turn"
    out["usage"]["output_tokens"] = 64  # hit the limit exactly: counted as truncated
    assert L.ClaudeCLI("m", runner=Runner(json.dumps(out))).complete("s", "u", 64).truncated
    e = L.ClaudeCLI("m", runner=Runner("", rc=1, stderr="boom")).complete("s", "u", 64)
    assert e.error and e.text == "" and e.usage_source == "estimate"
    assert e.input_tokens >= L.estimate_tokens("s") + L.estimate_tokens("u")
    bad = L.ClaudeCLI("m", runner=Runner("not json")).complete("s", "u", 64)
    assert bad.error and bad.error.startswith("unparseable")
    err = dict(out, is_error=True, result="API Error")
    assert L.ClaudeCLI("m", runner=Runner(json.dumps(err))).complete("s", "u", 64).error


CODEX_OUT = "\n".join(
    json.dumps(e)
    for e in (
        {"type": "thread.started", "thread_id": "t"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "i0", "type": "reasoning", "text": "hmm"}},
        {
            "type": "item.completed",
            "item": {"id": "i1", "type": "agent_message", "text": '{"ranking": ["c00"]}'},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 17000, "cached_input_tokens": 12000, "output_tokens": 40},
        },
    )
)


def test_codex_cli_argv_and_parse():
    r = Runner(CODEX_OUT)
    c = L.CodexCLI("gpt-6-astra", runner=r).complete("SYSTEM", "USER", max_output_tokens=128)
    a = r.calls[0]["argv"]
    assert a[:2] == ["codex", "exec"] and "--json" in a and "--skip-git-repo-check" in a
    assert a[a.index("--sandbox") + 1] == "read-only" and a[a.index("-m") + 1] == "gpt-6-astra"
    assert "model_max_output_tokens=128" in a
    assert r.calls[0]["stdin"].startswith("SYSTEM") and r.calls[0]["stdin"].endswith("USER")
    assert c.text == '{"ranking": ["c00"]}' and c.input_tokens == 17000 and c.output_tokens == 40
    failed = CODEX_OUT.replace('"turn.completed"', '"turn.failed"')
    assert L.CodexCLI("m", runner=Runner(failed)).complete("s", "u", 10).error


OPENCODE_OUT = "\n".join(
    json.dumps(e)
    for e in (
        {"type": "step_start", "part": {"type": "step-start"}},
        {"type": "text", "part": {"type": "text", "text": '{"verdict": '}},
        {"type": "text", "part": {"type": "text", "text": '"ACCEPT"}'}},
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "reason": "stop",
                "tokens": {
                    "input": 900,
                    "output": 8,
                    "reasoning": 4,
                    "cache": {"read": 100, "write": 0},
                },
            },
        },
    )
)


def test_opencode_gemini_parse_and_length_stop():
    r = Runner(OPENCODE_OUT)
    c = L.OpenCodeCLI("opencode-go/gemini-4-argon", runner=r).complete("SYS", "USR", 64)
    a = r.calls[0]["argv"]
    assert a[:2] == ["opencode", "run"] and a[a.index("-m") + 1] == "opencode-go/gemini-4-argon"
    assert a[a.index("--format") + 1] == "json"
    assert c.text == '{"verdict": "ACCEPT"}'
    assert c.input_tokens == 1000 and c.output_tokens == 12 and c.provider == "gemini"
    assert (
        L.OpenCodeCLI("m", runner=Runner(OPENCODE_OUT.replace('"stop"', '"length"')))
        .complete("s", "u", 64)
        .truncated
    )


def test_provider_registry_and_declared_overheads():
    assert set(L.PROVIDERS) == {"claude", "codex", "gemini", "mock"}
    assert L.make_client("claude", "claude-haiku-5-5").provider == "claude"
    assert (
        L.make_client("codex", "gpt-6-astra").overhead_tokens >= 17_000
    )  # measured in local Codex logs
    with pytest.raises(KeyError):
        L.make_client("grok", "x")


def test_token_ledger_hard_cap_precheck_and_postcheck():
    led = L.TokenLedger(cap=1000)
    m = L.MockClient(["a" * 250], overhead_tokens=0)
    c, fail = L.guarded_call(m, led, "worker", "s" * 250, "u" * 250, max_output_tokens=300)
    assert fail is None and led.spent == c.total == 200 + 100
    # pre-check: estimated input 400 + max output 400 > remaining 700? no -> admitted; 400 + 400 = 800 > 700 -> refused
    c2, fail2 = L.guarded_call(m, led, "worker", "s" * 500, "u" * 500, max_output_tokens=400)
    assert c2 is None and fail2 == "over_cap_precheck" and m.calls.__len__() == 1
    assert led.records[-1]["refused"] and led.spent == 300
    # post-check: an under-estimated call that lands over the cap is charged and fails
    big = L.MockClient(["z" * 10], overhead_tokens=0)
    big.complete = lambda s, u, n: L.Completion(
        "z", 900, 5, False, "end_turn", "mock", "m", "provider"
    )
    _, fail3 = L.guarded_call(big, led, "verifier", "s", "u", max_output_tokens=10)
    assert fail3 == "over_cap" and led.spent == 300 + 905 and led.exhausted
    rep = led.report()
    assert rep["spent"] == 1205 and rep["by_role"]["worker"]["calls"] == 1 and rep["refusals"] == 1


def test_guarded_call_truncation_and_error_are_failures():
    led = L.TokenLedger(cap=10_000)
    _, f = L.guarded_call(L.MockClient(["w" * 1000]), led, "r", "s", "u", max_output_tokens=10)
    assert f == "truncated"
    _, f = L.guarded_call(L.MockClient([]), led, "r", "s", "u", max_output_tokens=10)
    assert f == "provider_error"
    assert led.report()["failures"] == {"truncated": 1, "provider_error": 1}
