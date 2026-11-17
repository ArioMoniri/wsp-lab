"""Thin, offline-mockable LLM clients for the reader arms, over the delegate providers on this server, with token
accounting under a hard cap (PREREG_WME §6.8.3: token cap ``[SIM: reader_token_cap]``, truncation is failure).

* **Clients** share one call: ``complete(system, user, max_output_tokens) -> Completion``.
  - :class:`MockClient`: scripted responses, no network; tokens estimated (``wsp.agents.reader.estimate_tokens``).
  - :class:`ClaudeCLI` (``claude -p``): the system prompt replaces Claude Code's, every tool is disabled
    (``--tools ""``), no MCP servers, no CLAUDE.md or auto-memory, no session persistence; output limit through ``CLAUDE_CODE_MAX_OUTPUT_TOKENS``.
  - :class:`CodexCLI` (``codex exec --json``, read-only sandbox) and :class:`OpenCodeCLI` (``opencode run``, used for
    Gemini). These are agentic CLIs with a large fixed per-call prompt; their declared ``overhead_tokens`` (Codex:
    17,772 input tokens on the first turn of a local session log, 16 Nov 2026) enter the pre-call cap check.
  Every CLI runs through an injectable ``runner`` in a fresh empty temporary working directory, with an allow-listed
  environment (no ``GIT_*``, no repository or cloud credentials, no parent-session variables), prompt on stdin.
* **Completion**: text, input and output tokens (provider-reported when the CLI reports usage, cache reads and writes
  included; otherwise an estimate, flagged), stop reason, ``truncated`` (a length stop, or output tokens >= the
  limit) and ``error`` (non-zero exit, error result or unparseable output; tokens then estimated from the input).
* **TokenLedger** (one per arm and episode): a hard cap. :func:`guarded_call` refuses a call whose estimated input +
  overhead + output limit exceeds what remains (``over_cap_precheck``; nothing is sent), charges every completed call,
  and returns the failure: ``provider_error``, ``truncated`` or ``over_cap`` (charged usage beyond the cap).
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import tempfile
import time

from wsp.agents.reader import estimate_tokens

TRUNCATION_STOPS = {"max_tokens", "length", "max_output_tokens"}
ENV_ALLOW = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "USER",
    "LOGNAME",
    "TERM",
    "TMPDIR",
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
)


@dataclasses.dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool
    stop_reason: str | None
    provider: str
    model: str
    usage_source: str
    seconds: float = 0.0
    error: str | None = None

    @property
    def total(self) -> int:
        return int(self.input_tokens) + int(self.output_tokens)


class MockClient:
    provider = "mock"

    def __init__(self, responses, model="mock-1", overhead_tokens=0):
        self.model, self.overhead_tokens = model, int(overhead_tokens)
        self._fn = responses if callable(responses) else None
        self._queue = None if callable(responses) else list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, max_output_tokens):
        self.calls.append((system, user))
        inp = estimate_tokens(system) + estimate_tokens(user) + self.overhead_tokens
        if self._fn is None and not self._queue:
            return Completion(
                "",
                inp,
                0,
                False,
                None,
                self.provider,
                self.model,
                "estimate",
                0.0,
                "mock exhausted",
            )
        text = self._fn(system, user) if self._fn else self._queue.pop(0)
        if isinstance(text, Completion):
            return text
        out = estimate_tokens(text)
        if out > max_output_tokens:
            text = text[: int(max_output_tokens * 2.5)]
            return Completion(
                text,
                inp,
                int(max_output_tokens),
                True,
                "max_tokens",
                self.provider,
                self.model,
                "estimate",
            )
        return Completion(text, inp, out, False, "end_turn", self.provider, self.model, "estimate")


def sanitized_env(base=None) -> dict:
    base = os.environ if base is None else base
    return {k: base[k] for k in ENV_ALLOW if k in base}


def _subprocess_runner(argv, stdin, env, cwd, timeout):
    p = subprocess.run(
        argv,
        input=stdin,
        env=env,
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, p.stdout, p.stderr


class _CLIClient:
    provider = "cli"
    binary = ""
    default_overhead = 0

    def __init__(self, model, runner=None, timeout=300, overhead_tokens=None):
        self.model = model
        self.runner = runner or _subprocess_runner
        self.timeout = timeout
        self.overhead_tokens = (
            self.default_overhead if overhead_tokens is None else int(overhead_tokens)
        )

    # subclass hooks
    def argv(self, system, user, max_output_tokens):
        raise NotImplementedError

    def stdin(self, system, user):
        return f"{system}\n\n{user}"

    def extra_env(self, max_output_tokens):
        return {}

    def parse(self, stdout):
        """-> (text, input_tokens | None, output_tokens | None, stop_reason, error | None)"""
        raise NotImplementedError

    def complete(self, system, user, max_output_tokens):
        est_in = estimate_tokens(system) + estimate_tokens(user) + self.overhead_tokens
        env = dict(sanitized_env(), **self.extra_env(max_output_tokens))
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="wme-reader-") as cwd:
            try:
                rc, out, err = self.runner(
                    self.argv(system, user, max_output_tokens),
                    self.stdin(system, user),
                    env,
                    cwd,
                    self.timeout,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                rc, out, err = -1, "", f"{type(e).__name__}"
        secs = time.time() - t0
        if rc != 0:
            return Completion(
                "",
                est_in,
                0,
                False,
                None,
                self.provider,
                self.model,
                "estimate",
                secs,
                f"exit {rc}: {err[-200:]}",
            )
        try:
            text, tin, tout, stop, perr = self.parse(out)
        except (ValueError, KeyError, TypeError) as e:
            return Completion(
                "",
                est_in,
                0,
                False,
                None,
                self.provider,
                self.model,
                "estimate",
                secs,
                f"unparseable: {e}",
            )
        src = "provider" if tin is not None and tout is not None else "estimate"
        tin = est_in if tin is None else int(tin)
        tout = estimate_tokens(text) if tout is None else int(tout)
        truncated = (stop in TRUNCATION_STOPS) or tout >= max_output_tokens
        return Completion(
            text if not perr else "",
            tin,
            tout,
            truncated,
            stop,
            self.provider,
            self.model,
            src,
            secs,
            perr,
        )


class ClaudeCLI(_CLIClient):
    provider, binary = "claude", "claude"

    def argv(self, system, user, max_output_tokens):
        return [
            self.binary,
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--system-prompt",
            system,
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]

    def stdin(self, system, user):
        return user

    def extra_env(self, max_output_tokens):
        return {
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(int(max_output_tokens)),
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",  # no CLAUDE.md enters a reader's context
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",  # nor auto-memory
        }

    def parse(self, stdout):
        obj = json.loads(stdout)
        if not isinstance(obj, dict) or "result" not in obj:
            raise ValueError("no result field")
        u = obj.get("usage") or {}
        tin = None
        if "input_tokens" in u:
            tin = sum(
                int(u.get(k, 0) or 0)
                for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            )
        tout = u.get("output_tokens")
        err = f"provider error: {str(obj.get('result'))[:200]}" if obj.get("is_error") else None
        return str(obj["result"]), tin, tout, obj.get("stop_reason"), err


class CodexCLI(_CLIClient):
    provider, binary = "codex", "codex"
    default_overhead = 17_772  # first-turn input tokens in a local Codex session log (16 Nov 2026)

    def argv(self, system, user, max_output_tokens):
        return [
            self.binary,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-m",
            self.model,
            "-c",
            f"model_max_output_tokens={int(max_output_tokens)}",
            "-",
        ]

    def parse(self, stdout):
        text, usage, err = None, None, None
        for line in stdout.splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            t = ev.get("type")
            if t == "item.completed" and ev.get("item", {}).get("type") == "agent_message":
                text = ev["item"]["text"]
            elif t == "turn.completed":
                usage = ev.get("usage") or {}
            elif t in ("turn.failed", "error"):
                err = f"provider error: {json.dumps(ev)[:200]}"
        if text is None and err is None:
            raise ValueError("no agent message")
        u = usage or {}
        tin = u.get("input_tokens")
        tout = (
            None
            if "output_tokens" not in u
            else int(u["output_tokens"]) + int(u.get("reasoning_output_tokens", 0) or 0)
        )
        return text or "", tin, tout, None, err


class OpenCodeCLI(_CLIClient):
    provider, binary = "gemini", "opencode"
    default_overhead = 12_000  # declared assumption for the OpenCode agent prompt (not measured)

    def argv(self, system, user, max_output_tokens):
        return [self.binary, "run", "-m", self.model, "--format", "json", f"{system}\n\n{user}"]

    def stdin(self, system, user):
        return ""

    def parse(self, stdout):
        parts, tin, tout, stop, err = [], None, None, None, None
        for line in stdout.splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            t = ev.get("type")
            part = ev.get("part") or {}
            if t == "text":
                parts.append(part.get("text", ""))
            elif t == "step_finish":
                tk = part.get("tokens") or {}
                cache = tk.get("cache") or {}
                tin = (
                    (tin or 0)
                    + int(tk.get("input", 0))
                    + int(cache.get("read", 0))
                    + int(cache.get("write", 0))
                )
                tout = (tout or 0) + int(tk.get("output", 0)) + int(tk.get("reasoning", 0))
                stop = part.get("reason")
            elif t == "error":
                err = f"provider error: {json.dumps(ev)[:200]}"
        if not parts and err is None:
            raise ValueError("no text events")
        return "".join(parts), tin, tout, stop, err


PROVIDERS = {"claude": ClaudeCLI, "codex": CodexCLI, "gemini": OpenCodeCLI, "mock": MockClient}


def make_client(provider: str, model: str, **kw):
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider {provider!r}; choose {sorted(PROVIDERS)}")
    if provider == "mock":
        return MockClient(kw.pop("responses", []), model=model, **kw)
    return PROVIDERS[provider](model, **kw)


class TokenLedger:
    def __init__(self, cap: int):
        if int(cap) <= 0:
            raise ValueError("token cap must be positive")
        self.cap = int(cap)
        self.records: list[dict] = []

    @property
    def spent(self) -> int:
        return sum(r["total"] for r in self.records)

    @property
    def remaining(self) -> int:
        return self.cap - self.spent

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.cap

    def admit(self, client, system, user, max_output_tokens) -> bool:
        need = (
            estimate_tokens(system) + estimate_tokens(user) + getattr(client, "overhead_tokens", 0)
        )
        return need + int(max_output_tokens) <= self.remaining

    def record(self, role, completion: Completion | None, failure, client=None):
        c = completion
        self.records.append(
            {
                "role": role,
                "provider": c.provider if c else getattr(client, "provider", None),
                "model": c.model if c else getattr(client, "model", None),
                "input_tokens": c.input_tokens if c else 0,
                "output_tokens": c.output_tokens if c else 0,
                "total": c.total if c else 0,
                "usage_source": c.usage_source if c else None,
                "truncated": bool(c and c.truncated),
                "refused": c is None,
                "failure": failure,
                "seconds": c.seconds if c else 0.0,
            }
        )

    def report(self) -> dict:
        by_role, fails = {}, {}
        for r in self.records:
            b = by_role.setdefault(r["role"], {"calls": 0, "tokens": 0, "providers": set()})
            if not r["refused"]:
                b["calls"] += 1
            b["tokens"] += r["total"]
            b["providers"].add(f"{r['provider']}:{r['model']}")
            if r["failure"] and r["failure"] != "over_cap_precheck":
                fails[r["failure"]] = fails.get(r["failure"], 0) + 1
        for b in by_role.values():
            b["providers"] = sorted(b["providers"])
        return {
            "cap": self.cap,
            "spent": self.spent,
            "calls": sum(1 for r in self.records if not r["refused"]),
            "refusals": sum(1 for r in self.records if r["refused"]),
            "by_role": by_role,
            "failures": fails,
            "estimated_usage_calls": sum(
                1 for r in self.records if r["usage_source"] == "estimate"
            ),
        }


def guarded_call(client, ledger: TokenLedger, role, system, user, max_output_tokens):
    """-> (Completion | None, failure | None). Failures: over_cap_precheck, provider_error, truncated, over_cap."""
    if not ledger.admit(client, system, user, max_output_tokens):
        ledger.record(role, None, "over_cap_precheck", client)
        return None, "over_cap_precheck"
    c = client.complete(system, user, max_output_tokens)
    failure = None
    if c.error:
        failure = "provider_error"
    elif c.truncated:
        failure = "truncated"
    elif ledger.spent + c.total > ledger.cap:
        failure = "over_cap"
    ledger.record(role, c, failure)
    return c, failure
