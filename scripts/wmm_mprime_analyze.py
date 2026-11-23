#!/usr/bin/env python3
"""Study M′ phase analysis (EXPLORATORY). Reads outcomes only through the FC1 read-gate.

wmm_mprime_analyze.py --run data/wmm/mprime-2026-11-21 --phase base|msyc|mte|ma|final --out docs/reviews/wmm/X.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm.wmm import mprime_analysis as MA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase", required=True, choices=["base", "msyc", "mte", "ma", "final"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = MA.analyze(a.run, a.phase)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"phase": a.phase, "ledger": res["ledger"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
