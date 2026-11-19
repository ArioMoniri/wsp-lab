"""Write the wme-c1 (Qwen3.5-9B) confirmatory report from a results bundle (PREREG_WME v2, FROZEN).

    python scripts/wme_c1_report.py --bundle <dir> [--out-md <path>] [--out-json <path>]

Defaults: docs/reviews/wme/wme-c1-9b-results-<date>.md and .json. CPU only; every resampling count is the
registered value (``wsp.wm.c1.analysis``). The registered run takes minutes (9,999-resample bootstraps, D-HT
refits): run it through ``agent-tmux``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from wsp.wm.c1 import analysis as A


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument(
        "--no-dht",
        action="store_true",
        help="skip the descriptive D-HT refit (reported as not run)",
    )
    args = ap.parse_args(argv)
    out_dir = A.REPO / "docs" / "reviews" / "wme"
    out_md = args.out_md or out_dir / f"wme-c1-9b-results-{args.date}.md"
    out_json = args.out_json or out_dir / f"wme-c1-9b-results-{args.date}.json"
    md, res = A.report(
        args.bundle, date=args.date, manifest_path=args.manifest, run_dht=not args.no_dht
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    out_json.write_text(json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {out_md} and {out_json}")
    print(f"H-E1: {res['h_e1']['statement']['statement']}")
    print(f"H-E2: {res['h_e2']['primary']['statement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
