#!/usr/bin/env python3
"""Exclude the unconstrained ARA cohort without changing identity or measured artifacts."""

import argparse
import json
from pathlib import Path

FLAG = "ara_unconstrained_surrogate_scale"
REASON = "Diagnostic unconstrained ARA surrogate-scale cohort; corrected rebuild pending"


def _flag(record):
    record.setdefault("provenance", {})["diagnostic_flag"] = FLAG
    record["diagnostic_flag"] = FLAG
    record["eligibility"] = "smoke"
    reasons = record.setdefault("eligibility_reasons", [])
    if REASON not in reasons:
        reasons.append(REASON)


def flag_entries(zoo: Path, ids: list[str]) -> None:
    """Validate the entire cohort before updating metadata; safe to repeat."""
    ids = set(ids)
    manifest = zoo / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if ids - {r.get("id") for r in rows}:
        raise ValueError("Requested IDs missing from manifest")
    edits = {}
    for entry_id in sorted(ids):
        if Path(entry_id).name != entry_id:
            raise ValueError("Invalid entry ID")
        path = zoo / entry_id / "edit.json"
        edit = json.loads(path.read_text())
        if edit["family"] != "heretic_ara" or edit["id"] != entry_id:
            raise ValueError(f"Not the expected ARA entry: {entry_id}")
        # Diagnostic flags are observational: do not change params, id, or prior scores.
        edit.setdefault("provenance", {})["diagnostic_flag"] = FLAG
        edit["eligibility"] = "smoke"
        reasons = edit.setdefault("eligibility_reasons", [])
        if REASON not in reasons:
            reasons.append(REASON)
        edits[path] = edit
    for row in rows:
        if row.get("id") in ids:
            if row["family"] != "heretic_ara":
                raise ValueError("Unexpected manifest family")
            _flag(row)
    for path, edit in edits.items():
        path.write_text(json.dumps(edit, indent=2, sort_keys=True) + "\n")
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, default=Path("/tmp/ara_new_ids.json"))
    parser.add_argument("--zoo", type=Path, default=Path("data/zoo/qwen3.5-0.8b"))
    args = parser.parse_args()
    ids = json.loads(args.ids.read_text())
    flag_entries(args.zoo, ids)
    print(f"Flagged {len(set(ids))} ARA diagnostic entries (eligibility: smoke)")


if __name__ == "__main__":
    main()
