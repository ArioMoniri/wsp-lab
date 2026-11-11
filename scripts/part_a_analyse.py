#!/usr/bin/env python3
"""Part A single analysis CLI: registered A5.3/A5.5 outputs and F10.5 sensitivity reports.

Thin wrapper over ``wsp.eval.part_a_analysis.main``; see that module for the sealed-ledger schema.
Refuses (exit 2, nothing written) on a manifest/NULL-SIZE/fields hash mismatch, an unsealed or
incomplete ledger, a missing frozen prediction row or a frozen id-list mismatch.

    python scripts/part_a_analyse.py --ledger LEDGER.json --fields FIELDS.json \
        --output REPORT.json --markdown REPORT.md [--validity V.json] [--part-b B.json]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from wsp.eval.part_a_analysis import main

if __name__ == "__main__":
    raise SystemExit(main())
