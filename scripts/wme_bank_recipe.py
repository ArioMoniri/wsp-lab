#!/usr/bin/env python3
"""Emit the WM-E trajectory-bank RECIPE manifest (PREREG_WME §4). Recipes only: no model is loaded, no state is
realised, nothing is evaluated. Output: docs/prereg/wme_bank_recipe_draft.json (with its sha256)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import bank as B


def main():
    out = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/prereg/wme_bank_recipe_draft.json"
    )
    rec = B.bank_recipe()
    data = json.dumps(rec, indent=1, sort_keys=True, default=str)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(data)
    g = rec["components"]
    print(
        json.dumps(
            {
                "recipe_hash": rec["recipe_hash"],
                "file_sha256": hashlib.sha256(data.encode()).hexdigest(),
                "n_roots": len(rec["roots"]),
                "n_states_max": rec["n_states_max"],
                "components": {p: g[p]["n_components"] for p in g},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
