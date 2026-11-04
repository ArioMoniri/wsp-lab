#!/usr/bin/env python3
"""Regenerate the bibliography from the verified reference ledger."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main():
    entries = yaml.safe_load((ROOT / "paper/refs/verified.yaml").read_text())
    records = []
    for entry in entries:
        if entry["status"] != "verified":
            continue
        fields = {
            "title": "{" + entry["title"] + "}",
            "author": " and ".join(entry["authors"]),
            "year": str(entry["year"]),
        }
        if entry.get("arxiv_id"):
            fields.update(
                journal="arXiv preprint arXiv:" + str(entry["arxiv_id"]),
                eprint=str(entry["arxiv_id"]),
                archivePrefix="arXiv",
            )
            kind = "article"
        else:
            fields.update(url=entry["url"], note=entry["venue"])
            kind = "misc"
        records.append(
            "@"
            + kind
            + "{"
            + entry["key"]
            + ",\n"
            + ",\n".join("  " + key + "={" + value + "}" for key, value in fields.items())
            + "\n}"
        )
    (ROOT / "paper/refs/refs.bib").write_text("\n\n".join(records) + "\n")


if __name__ == "__main__":
    main()
