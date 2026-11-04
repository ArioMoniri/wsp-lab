"""H4 / M4 LLM-as-reader: build the text input for a zoo entry, ask a model, parse strictly.

    python -m wsp.agents.reader --features data/features/qwen3.5-0.8b --ids a,b \
        --provider dry|anthropic|openai --model <id> --out data/runs/h4/<run> \
        [--reveal-family] [--max-prompt-tokens N] [--base-url URL]
        [--price-in-per-mtok X --price-out-per-mtok Y]

Input per entry (from ``<features>/<id>/features.json`` + ``features.safetensors``, docs/DECISIONS.md D1):
a global summary over *all* changed tensors, then a per-tensor table sorted by ‖Δ_eff‖_F with the
columns name, layer, type, fro, rel_fro, cosine, top-3 singular values, stable rank. The edit family
label and the edit parameters are withheld unless ``--reveal-family``. A token-budget guard keeps the
top-N tensors by ``fro`` and says N in the prompt.

Per entry the run directory gets ``prompt.md``, ``response.json`` (raw text, usage; absent when the call
failed or in a dry run) and ``parsed.json`` (``status: ok`` with the parsed object, or ``status: failed``
with the reason). A response is never repaired, defaulted or guessed: anything that does not validate
against ``reader_schema.json`` (plus the contract in ``reader_prompt.md``) is recorded as failed.
``manifest.json`` records the model, prompt hashes, git SHA, token counts and a cost estimate (only when
prices are given on the command line; no price table is baked in).

Secrets: keys come from the process environment or ``~/.config/wsp/env``; they are only ever sent in
request headers and are scrubbed from error messages and ``repr``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import numpy as np
from safetensors.torch import load_file

from wsp.features.pool import classify
from wsp.features.stats import DELTA_STAT_NAMES
from wsp.schema import BENCHMARKS, write_manifest

log = logging.getLogger("wsp.agents.reader")

HERE = Path(__file__).parent
PROMPT_PATH = HERE / "reader_prompt.md"
SCHEMA_PATH = HERE / "reader_schema.json"
DEFAULT_ENV_FILE = Path.home() / ".config" / "wsp" / "env"

# Token counts are estimates (no provider tokenizer is available offline): numeric tables tokenise to
# roughly 2.5-3 characters per token, so the guard assumes the lower figure to stay on the safe side.
CHARS_PER_TOKEN = 2.5
DEFAULT_MAX_PROMPT_TOKENS = 20_000
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}


class ParseError(ValueError):
    """The model reply does not satisfy the reader output contract."""


class ProviderError(RuntimeError):
    """The provider call failed (configuration, transport, HTTP status or payload shape)."""


class DryRun(Exception):
    """Raised by the dry provider: prompts are written, nothing is sent."""


# ------------------------------------------------------------------ input


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Entry:
    id: str
    meta: dict[str, Any]
    names: list[str]
    stats: np.ndarray  # (n, len(stat_names)), rows follow ``names``
    stat_names: list[str]


def _check_id(entry_id: str) -> str:
    if not _SAFE_ID.fullmatch(entry_id) or ".." in entry_id:
        raise ValueError(f"unsafe entry id {entry_id!r}")
    return entry_id


def load_entry(features_dir: str | Path, entry_id: str) -> Entry:
    d = Path(features_dir) / _check_id(entry_id)
    meta_file = d / "features.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"{meta_file} not found (entry not extracted?)")
    meta = json.loads(meta_file.read_text())
    stat_names = list(meta.get("stat_names") or DELTA_STAT_NAMES)
    tensors = sorted(meta["tensors"], key=lambda t: t["row"])
    stats = load_file(str(d / "features.safetensors"))["stats"].float().numpy()
    if stats.shape != (len(tensors), len(stat_names)):
        raise ValueError(
            f"{d}: stats has shape {stats.shape}, expected {(len(tensors), len(stat_names))}"
        )
    if [t["row"] for t in tensors] != list(range(len(tensors))):
        raise ValueError(f"{meta_file}: tensor rows are not 0..n-1")
    return Entry(entry_id, meta, [t["name"] for t in tensors], stats, stat_names)


def tensor_rows(entry: Entry) -> list[dict[str, Any]]:
    """Per-tensor records sorted by ‖Δ_eff‖_F descending (ties by name)."""
    col = {n: i for i, n in enumerate(entry.stat_names)}
    rows = []
    for name, s in zip(entry.names, entry.stats, strict=True):
        kind, layer = classify(name)
        rows.append(
            {
                "name": name,
                "layer": layer,
                "type": kind,
                "fro": float(s[col["fro"]]),
                "rel_fro": float(s[col["rel_fro"]]),
                "cos": float(s[col["cos_prev_new"]]),
                "sv": [float(s[col[f"sv{k}"]]) for k in (1, 2, 3)],
                "stable_rank": float(s[col["stable_rank"]]),
            }
        )
    rows.sort(key=lambda r: (-r["fro"], r["name"]))
    return rows


def global_summary(entry: Entry) -> dict[str, Any]:
    rows = tensor_rows(entry)
    meta = entry.meta
    types: dict[str, int] = {}
    type_fro2: dict[str, float] = {}
    for r in rows:
        types[r["type"]] = types.get(r["type"], 0) + 1
        type_fro2[r["type"]] = type_fro2.get(r["type"], 0.0) + r["fro"] ** 2
    norms = meta.get("norms", {})
    return {
        "n_changed": len(rows),
        "n_unchanged": len(meta.get("unchanged_tensors", [])),
        "layers": sorted({r["layer"] for r in rows if r["layer"] is not None}),
        "n_layers": meta.get("n_layers"),
        "types": dict(sorted(types.items())),
        "type_fro": {k: math.sqrt(v) for k, v in sorted(type_fro2.items())},
        "fro_total": norms.get("fro_effective"),
        "rel_fro_total": norms.get("rel_fro_effective"),
        "fro_base_changed": norms.get("fro_base_changed"),
        "rounding": dict(meta.get("rounding_stats", {})),
        "delta_relative_to": meta.get("delta_relative_to"),
    }


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _ranges(values: Sequence[int]) -> str:
    out, start, prev = [], None, None
    for v in values:
        if start is None:
            start = prev = v
        elif v == prev + 1:  # type: ignore[operator]
            prev = v
        else:
            out.append((start, prev))
            start = prev = v
    if start is not None:
        out.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in out) or "none"


def _summary_lines(s: dict[str, Any]) -> list[str]:
    lines = [
        f"- changed tensors: {s['n_changed']}",
        f"- unchanged tensors (Δ_eff exactly zero): {s['n_unchanged']}",
        f"- model layers: {s['n_layers']}",
        f"- layers with changed tensors: {_ranges(s['layers'])}",
        "- changed tensors by type: " + ", ".join(f"{k}={v}" for k, v in s["types"].items()),
        "- fro of Δ_eff by type: " + ", ".join(f"{k}={_fmt(v)}" for k, v in s["type_fro"].items()),
        f"- total fro of Δ_eff: {_fmt(s['fro_total'])}",
        f"- total rel_fro of Δ_eff (to W over the changed tensors): {_fmt(s['rel_fro_total'])}",
        f"- fro of W over the changed tensors: {_fmt(s['fro_base_changed'])}",
        f"- delta measured relative to: {s['delta_relative_to']}",
    ]
    r = s["rounding"]
    if r:
        lines.append(
            "- bf16 rounding of the stored delta: "
            + ", ".join(f"{k}={_fmt(r[k])}" for k in sorted(r))
        )
    return lines


TABLE_HEADER = "name | layer | type | fro | rel_fro | cos_prev_new | sv1 | sv2 | sv3 | stable_rank"


def _row_line(r: dict[str, Any]) -> str:
    layer = "-" if r["layer"] is None else str(r["layer"])
    vals = [r["fro"], r["rel_fro"], r["cos"], *r["sv"], r["stable_rank"]]
    return " | ".join([r["name"], layer, r["type"], *(_fmt(v) for v in vals)])


@dataclass(frozen=True)
class Prompt:
    text: str
    sha256: str
    tokens: int  # estimate, see CHARS_PER_TOKEN
    n_shown: int
    n_total: int


def _render(entry: Entry, rows: list[dict[str, Any]], n: int, reveal_family: bool) -> str:
    meta = entry.meta
    parts = [
        PROMPT_PATH.read_text().rstrip(),
        "",
        "# Input",
        "",
        "## base",
        str(meta["base_model"]),
    ]
    if reveal_family:
        parts += ["", "## family", str(meta["family"])]
        parts += ["", "## edit_params", json.dumps(meta.get("params", {}), sort_keys=True)]
    parts += ["", "## edit_summary", *_summary_lines(global_summary(entry))]
    total = len(rows)
    if n == total:
        head = f"showing all {total} changed tensors, sorted by fro (largest first)"
    else:
        head = (
            f"showing the top {n} of {total} changed tensors by fro (largest first); the other "
            f"{total - n} have smaller fro, are not listed, and are included in edit_summary"
        )
    parts += ["", "## per_tensor_table", head, TABLE_HEADER, *(_row_line(r) for r in rows[:n])]
    return "\n".join(parts) + "\n"


def build_prompt(
    entry: Entry,
    *,
    reveal_family: bool = False,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> Prompt:
    """Render ``reader_prompt.md`` plus the entry input, keeping the largest N tensors that fit."""
    rows = tensor_rows(entry)
    total = len(rows)

    def tokens(n: int) -> int:
        return estimate_tokens(_render(entry, rows, n, reveal_family))

    lo, hi = (1 if total else 0), total  # largest n in [lo, hi] with tokens(n) <= budget
    if tokens(lo) > max_prompt_tokens:
        raise ValueError(
            f"entry {entry.id}: token budget {max_prompt_tokens} cannot hold the instructions, the "
            f"summary and {lo} tensor(s) (needs ~{tokens(lo)})"
        )
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tokens(mid) <= max_prompt_tokens:
            lo = mid
        else:
            hi = mid - 1
    text = _render(entry, rows, lo, reveal_family)
    return Prompt(text, hashlib.sha256(text.encode()).hexdigest(), estimate_tokens(text), lo, total)


# ------------------------------------------------------------------ parsing

_SCHEMA = json.loads(SCHEMA_PATH.read_text())
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


def _no_constant(name: str) -> Any:
    raise ParseError(f"non-finite number {name} in reply")


def parse_response(text: str) -> dict[str, Any]:
    """Strictly validate a reply against ``reader_schema.json`` and the reader contract."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty reply")
    try:
        obj = json.loads(text.strip(), parse_constant=_no_constant)
    except json.JSONDecodeError as e:
        raise ParseError(f"reply is not a single JSON object: {e.msg} (char {e.pos})") from e
    err = jsonschema.exceptions.best_match(_VALIDATOR.iter_errors(obj))
    if err is not None:
        path = "/".join(str(p) for p in err.absolute_path) or "<root>"
        raise ParseError(f"schema violation at {path}: {err.message[:200]}")
    if set(obj) != {"predictions", "mechanism", "confidence"}:
        raise ParseError(f"unexpected top-level keys {sorted(set(obj) - set(_SCHEMA['required']))}")
    preds = obj["predictions"]
    if set(preds) != set(BENCHMARKS):
        missing, extra = sorted(set(BENCHMARKS) - set(preds)), sorted(set(preds) - set(BENCHMARKS))
        raise ParseError(
            f"predictions must cover every benchmark: missing {missing}, unknown {extra}"
        )
    for k, v in [*preds.items(), ("confidence", obj["confidence"])]:
        if not math.isfinite(v):
            raise ParseError(f"non-finite value for {k}")
    if preds["kl_to_base"] < 0:
        raise ParseError("kl_to_base must be non-negative")
    return obj


# ------------------------------------------------------------------ providers


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.removeprefix("export ").partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        out[key.strip()] = val
    return out


class _HTTPProvider:
    name = "http"
    key_var = ""

    def __init__(
        self,
        model: str,
        key: str,
        url: str,
        headers: Mapping[str, str],
        *,
        client: httpx.Client | None = None,
        sleep=time.sleep,
        timeout: float = 600.0,
        retries: int = 4,
        max_output_tokens: int = 8192,
    ):
        self.model = model
        self._key = key
        self._url = url
        self._headers = dict(headers)
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep
        self._retries = retries
        self.max_output_tokens = max_output_tokens
        self.last_usage: dict[str, int] | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, url={self._url!r})"

    def _scrub(self, text: str) -> str:
        for secret in [self._key, *self._headers.values()]:
            if secret and len(secret) >= 6:
                text = text.replace(secret, "<redacted>")
        return text[:300]

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last = "no attempt made"
        for attempt in range(self._retries):
            try:
                resp = self._client.post(self._url, headers=self._headers, json=body)
            except httpx.HTTPError as e:
                last = f"transport error {type(e).__name__}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as e:
                        raise ProviderError("provider returned a non-JSON body") from e
                last = f"HTTP {resp.status_code}: {self._scrub(resp.text)}"
                if resp.status_code not in _RETRY_STATUS:
                    raise ProviderError(last)
            if attempt + 1 < self._retries:
                self._sleep(min(2.0**attempt, 30.0))
        raise ProviderError(f"{self.name} request failed after {self._retries} attempts: {last}")

    def ask(self, prompt: str) -> str:
        raise NotImplementedError


class AnthropicProvider(_HTTPProvider):
    name = "anthropic"
    key_var = "ANTHROPIC_API_KEY"

    def ask(self, prompt: str) -> str:
        data = self._post(
            {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        blocks = data.get("content")
        text = (
            "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            if isinstance(blocks, list)
            else ""
        )
        if not text:
            raise ProviderError("anthropic reply has no text content")
        u = data.get("usage") or {}
        self.last_usage = {
            "input_tokens": int(u.get("input_tokens", 0)),
            "output_tokens": int(u.get("output_tokens", 0)),
        }
        return text


class OpenAIProvider(_HTTPProvider):
    name = "openai"
    key_var = "OPENAI_API_KEY"
    max_tokens_field = "max_completion_tokens"

    def ask(self, prompt: str) -> str:
        data = self._post(
            {
                "model": self.model,
                self.max_tokens_field: self.max_output_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError("openai reply has no choices[0].message.content") from e
        if not isinstance(text, str) or not text:
            raise ProviderError("openai reply has empty content")
        u = data.get("usage") or {}
        self.last_usage = {
            "input_tokens": int(u.get("prompt_tokens", 0)),
            "output_tokens": int(u.get("completion_tokens", 0)),
        }
        return text


class DryProvider:
    name = "dry"
    last_usage = None

    def __init__(self, model: str = "dry"):
        self.model = model

    def ask(self, prompt: str) -> str:
        raise DryRun


def _parse_custom_headers(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw.replace("\\n", "\n").splitlines():
        if not line.strip():
            continue
        name, sep, value = line.partition(":")
        if not sep or not name.strip():
            raise ProviderError("ANTHROPIC_CUSTOM_HEADERS lines must look like 'Name: value'")
        headers[name.strip()] = value.strip()
    return headers


def make_provider(
    name: str,
    model: str,
    *,
    base_url: str | None = None,
    env: Mapping[str, str] | None = None,
    env_file: Path | None = None,
    client: httpx.Client | None = None,
    sleep=time.sleep,
    max_output_tokens: int = 8192,
    max_tokens_field: str | None = None,
):
    """Build a provider. ``env=None`` means ``os.environ`` then ``~/.config/wsp/env``; an explicit
    ``env`` is used as given (plus ``env_file`` only when that is passed too)."""
    if name == "dry":
        return DryProvider(model)
    if name not in ("anthropic", "openai"):
        raise ValueError(f"unknown provider {name!r}; choose dry, anthropic or openai")
    if env is None:
        merged = {**read_env_file(env_file or DEFAULT_ENV_FILE), **os.environ}
    else:
        merged = {**(read_env_file(env_file) if env_file else {}), **env}
    cls = AnthropicProvider if name == "anthropic" else OpenAIProvider
    key = merged.get(cls.key_var, "")
    if not key:
        raise ProviderError(f"{cls.key_var} is not set (process environment or ~/.config/wsp/env)")
    common: dict[str, Any] = {
        "client": client,
        "sleep": sleep,
        "max_output_tokens": max_output_tokens,
    }
    if name == "anthropic":
        root = (base_url or merged.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip(
            "/"
        )
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            **_parse_custom_headers(merged.get("ANTHROPIC_CUSTOM_HEADERS", "")),
        }
        return AnthropicProvider(model, key, f"{root}/v1/messages", headers, **common)
    root = (base_url or merged.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
    provider = OpenAIProvider(model, key, f"{root}/chat/completions", headers, **common)
    if max_tokens_field:
        provider.max_tokens_field = max_tokens_field
    return provider


# ------------------------------------------------------------------ run


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            or "unknown"
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def run_reader(
    features_dir: str | Path,
    ids: Sequence[str],
    out: str | Path,
    provider: Any,
    *,
    provider_name: str,
    model: str,
    reveal_family: bool = False,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    price_in_per_mtok: float | None = None,
    price_out_per_mtok: float | None = None,
) -> dict[str, Any]:
    """Ask ``provider`` about every id, write per-entry files and the run manifest, return the manifest.

    Re-running into the same directory only asks again for entries without an ``ok`` result for the
    identical prompt, so a crash never repeats paid calls.
    """
    out = Path(out)
    config = {
        "provider": provider_name,
        "model": model,
        "reveal_family": reveal_family,
        "max_prompt_tokens": max_prompt_tokens,
        "chars_per_token": CHARS_PER_TOKEN,
    }
    previous = _read_json(out / "manifest.json")
    if previous and {k: previous.get(k) for k in config} != config:
        raise ValueError(f"{out} holds a run with a different configuration; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    for entry_id in ids:
        entry = load_entry(features_dir, entry_id)
        prompt = build_prompt(
            entry, reveal_family=reveal_family, max_prompt_tokens=max_prompt_tokens
        )
        d = out / entry_id
        d.mkdir(exist_ok=True)
        info = {
            "prompt_sha256": prompt.sha256,
            "n_shown": prompt.n_shown,
            "n_total": prompt.n_total,
            "tokens_estimated": prompt.tokens,
        }
        done = _read_json(d / "parsed.json")
        if (
            done
            and done.get("status") == "ok"
            and (d / "prompt.md").exists()
            and _sha256_file(d / "prompt.md") == prompt.sha256
        ):
            entries[entry_id] = info | {"status": "ok"}
            log.info("%s: already done, skipping", entry_id)
            continue
        (d / "prompt.md").write_text(prompt.text)
        for stale in ("response.json", "parsed.json"):
            (d / stale).unlink(missing_ok=True)
        try:
            started = time.monotonic()
            text = provider.ask(prompt.text)
        except DryRun:
            entries[entry_id] = info | {"status": "dry"}
            continue
        except ProviderError as e:
            _write_json(
                d / "parsed.json", {"status": "failed", "stage": "provider", "error": str(e)}
            )
            entries[entry_id] = info | {"status": "failed"}
            log.warning("%s: provider failed", entry_id)
            continue
        _write_json(
            d / "response.json",
            {
                "text": text,
                "usage": provider.last_usage,
                "model": model,
                "provider": provider_name,
                "latency_s": round(time.monotonic() - started, 3),
            },
        )
        try:
            parsed = parse_response(text)
        except ParseError as e:
            _write_json(d / "parsed.json", {"status": "failed", "stage": "parse", "error": str(e)})
            entries[entry_id] = info | {"status": "failed"}
            log.warning("%s: reply rejected: %s", entry_id, e)
            continue
        _write_json(d / "parsed.json", {"status": "ok", "parsed": parsed})
        entries[entry_id] = info | {"status": "ok"}

    usage_in = usage_out = 0
    have_usage = False
    for entry_id in entries:
        usage = (_read_json(out / entry_id / "response.json") or {}).get("usage")
        if usage:
            have_usage = True
            usage_in += usage["input_tokens"]
            usage_out += usage["output_tokens"]
    cost = None
    if have_usage and price_in_per_mtok is not None and price_out_per_mtok is not None:
        cost = (usage_in * price_in_per_mtok + usage_out * price_out_per_mtok) / 1e6
    status = [e["status"] for e in entries.values()]
    manifest = {
        **config,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "features": str(features_dir),
        "ids": list(ids),
        "git_sha": _git_sha(),
        "prompt_template_sha256": _sha256_file(PROMPT_PATH),
        "schema_sha256": _sha256_file(SCHEMA_PATH),
        "token_estimate": "characters / chars_per_token (no provider tokenizer offline)",
        "tokens": {
            "input_estimated": sum(e["tokens_estimated"] for e in entries.values()),
            "input_reported": usage_in if have_usage else None,
            "output_reported": usage_out if have_usage else None,
        },
        "prices_usd_per_mtok": {"input": price_in_per_mtok, "output": price_out_per_mtok},
        "cost_estimate_usd": cost,
        "n_ok": status.count("ok"),
        "n_failed": status.count("failed"),
        "n_dry": status.count("dry"),
        "entries": entries,
    }
    write_manifest(out / "manifest.json", **manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--ids", required=True, help="comma-separated entry ids")
    ap.add_argument("--provider", required=True, choices=["dry", "anthropic", "openai"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--reveal-family", action="store_true", help="include the family label + params"
    )
    ap.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    ap.add_argument("--base-url", default=None, help="API root (openai: incl. /v1)")
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--max-tokens-field", default=None, help="openai: max_tokens for older servers")
    ap.add_argument("--price-in-per-mtok", type=float, default=None)
    ap.add_argument("--price-out-per-mtok", type=float, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    if not ids or len(set(ids)) != len(ids):
        ap.error("--ids must be a non-empty list without duplicates")
    for i in ids:
        try:
            _check_id(i)
        except ValueError as e:
            ap.error(str(e))
        if not (args.features / i / "features.json").exists():
            ap.error(f"no extracted entry {i!r} under {args.features}")
    try:
        provider = make_provider(
            args.provider,
            args.model,
            base_url=args.base_url,
            max_output_tokens=args.max_output_tokens,
            max_tokens_field=args.max_tokens_field,
        )
    except (ProviderError, ValueError) as e:
        ap.error(str(e))
    m = run_reader(
        args.features,
        ids,
        args.out,
        provider,
        provider_name=args.provider,
        model=args.model,
        reveal_family=args.reveal_family,
        max_prompt_tokens=args.max_prompt_tokens,
        price_in_per_mtok=args.price_in_per_mtok,
        price_out_per_mtok=args.price_out_per_mtok,
    )
    print(
        json.dumps({k: m[k] for k in ("n_ok", "n_failed", "n_dry", "tokens", "cost_estimate_usd")})
    )


if __name__ == "__main__":
    main()
