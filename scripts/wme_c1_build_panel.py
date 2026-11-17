#!/usr/bin/env python3
"""Build the confirmatory panel ``data/wme/panels/wme-c1`` (PREREG_WME §3). POST-FREEZE ONLY.

Refuses to run unless ``WME_C1_FREEZE`` names the freeze commit whose ``docs/prereg/wme_manifest*.json`` fixes the
sizes. Pools and exposures as ``scripts/wme_c1_exposure_audit.py``; assignment by ``wm.panel_c1.build_c1_panel``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    freeze = os.environ.get("WME_C1_FREEZE")
    if not freeze:
        raise SystemExit(
            "wme-c1 is built only after the PREREG_WME freeze: set WME_C1_FREEZE=<freeze commit>"
        )
    from wme_c1_exposure_audit import pools

    from wsp.wm import exposure as X
    from wsp.wm import panel_c1 as C1

    man_path = Path(os.environ.get("WME_C1_MANIFEST", ROOT / "docs/prereg/wme_manifest.json"))
    sizes = json.loads(man_path.read_text())["panel"]["sizes"]
    pool, meta = pools()
    man = C1.build_c1_panel(pool, sizes, ROOT / "data/wme/panels/wme-c1", X.load_exposures(ROOT))
    man.update(freeze_commit=freeze, sources=meta)
    (ROOT / "data/wme/panels/wme-c1/manifest.json").write_text(
        json.dumps(man, indent=1, sort_keys=True, default=str)
    )
    print(man["content_hash"])


if __name__ == "__main__":
    main()
