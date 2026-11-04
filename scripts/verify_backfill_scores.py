"""Compare archived eval scores, counts and nested metrics; fail on any difference.

Archives created after historical backfills cannot prove those earlier operations.
All non-metadata fields are protected, including future benchmark score keys.
"""

import argparse
import json
from pathlib import Path

from wsp.eval.repeats import REFERENCE_KEYS

METADATA = set(REFERENCE_KEYS) | {
    "eval_host",
    "batch_size_nominal",
    "metadata_backfill",
    "metadata_backfill_utc",
    "metadata_backfill_missing",
    "metadata_backfill_inferred",
    "metadata_backfill_source",
    "metadata_backfill_provenance",
}


def scores(payload):
    return {
        k: v
        for k, v in payload.items()
        if k not in METADATA and not k.startswith("metadata_backfill_")
    }


def verify(root):
    failures = []
    for current in Path(root).rglob("eval*.json"):
        if not current.with_name(current.name + ".orig").exists():
            failures.append(str(current) + ": missing original archive")
    for original in sorted({p.resolve() for p in Path(root).rglob("eval*.json.orig")}):
        current = original.with_name(original.name.removesuffix(".orig"))
        if not current.exists():
            failures.append(str(current) + ": missing")
            continue
        before, after = (scores(json.loads(p.read_text())) for p in (original, current))
        for key in sorted(before.keys() | after.keys()):
            if key not in before or key not in after or before[key] != after[key]:
                failures.append(f"{current}: {key}")
    return failures


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("roots", type=Path, nargs="+", default=None)
    args = parser.parse_args()
    failures = [failure for root in args.roots for failure in verify(root)]
    checked = {p.resolve() for root in args.roots for p in root.rglob("eval*.json.orig")}
    print(
        json.dumps(
            {
                "checked_records": len(checked),
                "score_differences": len(failures),
                "failures": failures,
            }
        )
    )
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
