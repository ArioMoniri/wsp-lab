#!/usr/bin/env python3
"""Check that paper section citations exist in the bibliography and are verified."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "paper" / "sections"
VERIFIED = ROOT / "paper" / "refs" / "verified.yaml"
BIB = ROOT / "paper" / "refs" / "refs.bib"

RECORD_RE = re.compile(r"(?m)^- key:\s*([^\s#]+)(.*?)(?=^- key:|\Z)", re.DOTALL)
STATUS_RE = re.compile(r"(?m)^\s+status:\s*([^\s#]+)\s*(?:#.*)?$")
CITE_RE = re.compile(r"\\cite[A-Za-z]*\*?(?:\s*\[[^\]]*\])*\s*\{([^}]*)\}")
BIB_KEY_RE = re.compile(r"(?m)^\s*@\w+\s*\{\s*([^,\s]+)\s*,")


def main() -> int:
    if not SECTIONS.is_dir() or not VERIFIED.is_file() or not BIB.is_file():
        print(
            "citation check failed: expected sections, verified.yaml, and refs.bib", file=sys.stderr
        )
        return 2

    verified_text = VERIFIED.read_text(encoding="utf-8")
    verified = {
        match.group(1)
        for match in RECORD_RE.finditer(verified_text)
        if (status := STATUS_RE.search(match.group(2))) and status.group(1) == "verified"
    }
    bibliography = set(BIB_KEY_RE.findall(BIB.read_text(encoding="utf-8")))
    problems: list[str] = []

    for path in sorted(SECTIONS.glob("*.tex")):
        text = uncomment_tex(path.read_text(encoding="utf-8"))
        for match in CITE_RE.finditer(text):
            for key in (part.strip() for part in match.group(1).split(",")):
                if not key:
                    continue
                if key not in bibliography:
                    problems.append(f"{path.relative_to(ROOT)}: {key} is missing from refs.bib")
                if key not in verified:
                    problems.append(
                        f"{path.relative_to(ROOT)}: {key} is not verified in verified.yaml"
                    )

    if problems:
        print("citation check failed:", file=sys.stderr)
        print("\n".join(f"- {problem}" for problem in problems), file=sys.stderr)
        return 1

    print(
        f"citation check passed: {len(verified)} verified keys available; scanned {len(list(SECTIONS.glob('*.tex')))} section files"
    )
    return 0


def uncomment_tex(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                line = line[:index]
                break
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
