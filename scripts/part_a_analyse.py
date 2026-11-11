#!/usr/bin/env python3
"""Part A single analysis CLI: registered A5.3/A5.5 outputs and F10.5 sensitivity reports.

Thin wrapper over ``wsp.eval.part_a_analysis.main``; see that module for the sealed-ledger,
validity, Part B and masked-calibration input formats (each a choice pending adopting note).
Refuses (exit 2, nothing written) on a specification hash that differs from the F10b.1 adoption
note or from ``--spec-sha256``; a frozen (a533) manifest whose sha256 is not the preserved
staging manifest's; an execution manifest (``--manifest``) whose bytes differ from the required
``--manifest-sha256`` or from the hash the sealed ledger records, that lacks a registered id or an
a533 variant id, or that selects an unflagged non-registered edit; a NULL-SIZE/text-groups/fields
hash mismatch; a ledger window order, block assignment or token identity that differs from
``text_groups.json``; an unsealed or incomplete ledger; a re-stage entry that contradicts itself
or the execution manifest; a missing frozen prediction row; a frozen id-list mismatch; or a
missing/mismatched F10.6 slot-masked calibration report.

    python scripts/part_a_analyse.py --ledger LEDGER.json --fields FIELDS.json \
        --spec-sha256 8624e06343bbb401ff3b6d783572b0c53e1172d0ea3f36aec3768a13701998d4 \
        --manifest docs/prereg/a5_manifests/part_a/manifest.json --manifest-sha256 SHA256 \
        --output REPORT.json --markdown REPORT.md [--validity V.json] [--part-b B.json] \
        [--masked-calibration MASKED.json] \
        [--frozen-manifest docs/prereg/a5_manifests/part_a/manifest_a533.json]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from wsp.eval.part_a_analysis import main

if __name__ == "__main__":
    raise SystemExit(main())
