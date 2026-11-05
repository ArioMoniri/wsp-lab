#!/usr/bin/env python3
"""Regenerate the status block in README.md from artefacts on disk.

The block sits between ``<!-- status:start -->`` and ``<!-- status:end -->``. Everything in it is read from
files that the pipeline itself writes (zoo manifest, feature directory, eval indices, the base-model run
summary, the Lambda ledger) plus the latest GitLab pipeline if ``glab`` is installed and logged in. Nothing
is typed in by hand, and a missing source is reported as missing instead of being filled with a default.

Usage: ``python scripts/update_readme_status.py [--root DIR] [--no-ci] [--print]``
Running it twice on unchanged artefacts gives an identical README except for the timestamp line.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

START, END = "<!-- status:start -->", "<!-- status:end -->"
MODEL = "qwen3.5-0.8b"
BASE_RUN = "p2/base-final-4096"
GITLAB_PROJECT = "ArioMoniri%2Fwsp-lab"
FAMILY_ORDER = ["random_dir", "abliteration", "heretic_ara", "lora_sft", "task_vector", "chain"]


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def zoo_counts(root: Path) -> dict | None:
    path = root / "data" / "zoo" / MODEL / "manifest.jsonl"
    if not path.is_file():
        return None
    entries: dict[str, dict] = {}
    for row in _jsonl(path):
        if "id" in row:  # header rows carry no id; the last row for an id wins
            entries[row["id"]] = row
    by_family: dict[str, Counter] = defaultdict(Counter)
    for row in entries.values():
        by_family[row.get("family", "unknown")][row.get("eligibility") or "unlabelled"] += 1
    return {
        "total": len(entries),
        "by_family": {f: dict(c) for f, c in by_family.items()},
    }


def feature_counts(root: Path) -> dict | None:
    directory = root / "data" / "features" / MODEL
    if not directory.is_dir():
        return None
    entries = sum(1 for p in directory.iterdir() if (p / "features.json").is_file())
    complete = None
    manifest = directory / "manifest.json"
    if manifest.is_file():
        try:
            complete = json.loads(manifest.read_text()).get("complete_run")
        except json.JSONDecodeError:
            pass
    return {"entries": entries, "complete_run": complete}


def eval_counts(root: Path) -> dict | None:
    indices = sorted((root / "data" / "evals").glob("*/index.jsonl"))
    if not indices:
        return None
    rows, ids, complete = 0, set(), set()
    for path in indices:
        for row in _jsonl(path):
            rows += 1
            ids.add(row.get("edit_id"))
            if row.get("eval_status") == "complete":
                complete.add(row.get("edit_id"))
    # The index only records entries evaluated on this host; entries pulled back from other devices land as
    # <zoo>/<id>/eval.json without an index row, so count the artefacts themselves as the source of truth.
    zoo_complete = 0
    for ev in (root / "data" / "zoo").glob("*/*/eval.json"):
        try:
            import json as _json

            if _json.loads(ev.read_text()).get("eval_status") == "complete":
                zoo_complete += 1
        except Exception:
            continue
    return {
        "rows": rows,
        "entries": max(len(ids), zoo_complete),
        "complete_entries": max(len(complete), zoo_complete),
    }


def base_summary(root: Path) -> dict | None:
    path = root / "data" / "runs" / BASE_RUN / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def lambda_open(root: Path) -> list[dict] | None:
    path = root / "board" / "lambda_spend.md"
    if not path.is_file():
        return None
    open_jobs = []
    for line in path.read_text().splitlines():
        if not line.startswith("|") or line.startswith("|---") or "start (UTC)" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and cells[0] and not cells[1]:
            open_jobs.append(
                {"start": cells[0], "instance": cells[2], "type": cells[3], "job": cells[7]}
            )
    return open_jobs


def latest_ci() -> dict | None:
    """Latest pipeline on main via glab; None when glab is missing or the call fails."""
    if shutil.which("glab") is None:
        return None
    try:
        out = subprocess.run(
            ["glab", "api", f"projects/{GITLAB_PROJECT}/pipelines?ref=main&per_page=1"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        pipe = json.loads(out)[0]
        return {"status": pipe["status"], "id": pipe["id"], "sha": pipe["sha"]}
    except (subprocess.SubprocessError, json.JSONDecodeError, IndexError, KeyError, OSError):
        return None


def collect(root: Path, ci: dict | None) -> dict:
    return {
        "zoo": zoo_counts(root),
        "features": feature_counts(root),
        "evals": eval_counts(root),
        "base": base_summary(root),
        "lambda_open": lambda_open(root),
        "ci": ci,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def render(s: dict, now: str) -> str:
    lines = [
        START,
        "",
        f"_Generated by `scripts/update_readme_status.py` from artefacts on {now}._",
        "",
    ]

    zoo = s["zoo"]
    if zoo is None:
        lines.append(f"Zoo: manifest not found (`data/zoo/{MODEL}/manifest.jsonl`).")
    else:
        fams = zoo["by_family"]
        names = [f for f in FAMILY_ORDER if f in fams] + sorted(set(fams) - set(FAMILY_ORDER))
        states = sorted({k for f in fams.values() for k in f})
        lines += [
            f"**Zoo** (`data/zoo/{MODEL}`): {zoo['total']} entries in the manifest.",
            "",
            "| family | " + " | ".join(states) + " | total |",
            "|---|" + "---:|" * (len(states) + 1),
        ]
        for f in names:
            row = [str(fams[f].get(k, 0)) for k in states]
            lines.append(f"| {f} | " + " | ".join(row) + f" | {sum(fams[f].values())} |")
        lines.append("")
        lines.append(
            "`smoke` entries come from short test builds and `unlabelled` rows predate the eligibility field. "
            "The batch evaluator takes only `production` entries by default; feature extraction reads every "
            "valid manifest id, whatever its eligibility, and skips entries whose files are not built yet."
        )
    lines.append("")

    feat = s["features"]
    if feat is None:
        lines.append(f"Features: directory not found (`data/features/{MODEL}`).")
    else:
        state = {True: "finished", False: "extraction not finished", None: "run state unknown"}[
            feat["complete_run"]
        ]
        lines.append(f"**Features**: {feat['entries']} entries extracted ({state}).")
    lines.append("")

    ev = s["evals"]
    if ev is None:
        lines.append("**Evals**: no `data/evals/*/index.jsonl` found.")
    else:
        lines.append(
            f"**Evals**: {ev['rows']} index rows, {ev['entries']} entries, "
            f"{ev['complete_entries']} with a complete result."
        )
    lines.append("")

    base = s["base"]
    if base is None:
        lines.append(f"**Base model**: `data/runs/{BASE_RUN}/summary.json` not found.")
    else:
        gen = base.get("generation", {})
        lines += [
            (
                f"**Base model** (Qwen3.5-0.8B, run `{BASE_RUN}`, "
                f"{gen.get('max_new_tokens', '?')} new tokens, percent):"
            ),
            "",
            "| GSM8K | MATH-500 | IFEval | MMLU-Pro | HellaSwag | loop rate | refusal rate |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            "| "
            + " | ".join(
                _pct(base[k])
                for k in (
                    "gsm8k",
                    "math500",
                    "ifeval",
                    "mmlu_pro",
                    "hellaswag",
                    "loop_rate",
                    "refusal_rate",
                )
            )
            + " |",
        ]
    lines.append("")

    jobs = s["lambda_open"]
    if jobs is None:
        lines.append("**Lambda**: `board/lambda_spend.md` not found.")
    elif not jobs:
        lines.append(
            "**Lambda**: no open rows in `board/lambda_spend.md`. The ledger is filled by hand, so check `scripts/lambda.sh ls` for live instances."
        )
    else:
        desc = "; ".join(
            f"{j['instance']} ({j['type']}, {j['job']}, since {j['start']})" for j in jobs
        )
        lines.append(f"**Lambda**: {len(jobs)} open ledger row(s): {desc}.")
    lines.append("")

    if s["ci"] is not None:
        c = s["ci"]
        lines.append(
            f"**CI** (GitLab, `main`): pipeline {c['id']} on `{c['sha'][:7]}`, status {c['status']}."
        )
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def update(root: Path, ci: dict | None, now: str | None = None) -> str:
    now = now or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8")
    i, j = text.find(START), text.find(END)
    if i < 0 or j < i:
        raise SystemExit(f"{readme}: status markers not found")
    block = render(collect(root, ci), now)
    new = text[:i] + block + text[j + len(END) :]
    if new != text:
        readme.write_text(new, encoding="utf-8")
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--no-ci", action="store_true", help="skip the GitLab pipeline lookup")
    ap.add_argument("--print", action="store_true", help="print the block after updating")
    args = ap.parse_args()
    block = update(args.root, None if args.no_ci else latest_ci())
    if args.print:
        print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
