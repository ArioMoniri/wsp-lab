#!/usr/bin/env python3
"""Build the stage-W dev panel ``data/wme/panels/wme-w`` (RULING-wme-stagew §2; EXPLORATORY; CPU).

MMLU-Pro (b189ec7, test + validation) and SuperGPQA (4430d44, ``SuperGPQA-all.jsonl``), loaded exactly as the pilot-3
builder loads them. Every exposure in ``wm.exposure.EXPOSURES`` (now including ``wme-p3s`` and ``wme-p3h``, all
splits) is excluded at near-duplicate-group level from manifest hash lists only; sealed files are never opened.
dev = 500 per benchmark in salted rank order (salt ``wme-w-panel-20261118``). Nothing is scored.

    wme_w_build_panel.py [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _p3_builder():
    spec = importlib.util.spec_from_file_location("p3b", ROOT / "scripts/wme_p3_build_panels.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # sets the pinned HF environment of the pilot-3 builder
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data/wme/panels/wme-w"))
    a = ap.parse_args()
    out = Path(a.out)
    if not a.dry_run and out.exists():
        raise SystemExit(f"{out} exists; panels are frozen once built")
    p3 = _p3_builder()
    from wsp.wm import exposure as X
    from wsp.wm import p3gen as G
    from wsp.wm import p3panel as PP
    from wsp.wm import stagew as W

    spec = importlib.util.spec_from_file_location("b1", ROOT / "scripts/wme_build_panel.py")
    b1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b1)
    raw, meta = b1.load_raw()
    p, sha = p3._hub_file("supergpqa")
    sg = []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        sg.append(
            {
                "question": r["question"],
                "options": list(r["options"]),
                "answer_index": G.LETTERS.index(r["answer_letter"]),
            }
        )
    meta = {
        "mmlu_pro": meta["mmlu_pro"],
        "supergpqa": {"pin": p3.PINS["supergpqa"], "sha256": sha, "rows": len(sg)},
    }
    exposures = X.load_exposures(ROOT, X.EXPOSURES)
    exposed = X.all_exposed(exposures)
    pools = W.panel_pools(raw["mmlu_pro"], sg)
    res = W.panel_select(pools, exposed, n=W.N_DEV)
    # fresh groups left after this panel (for wme-c2-gen sizing: 3,463 MMLU-Pro, 1,784 SuperGPQA)
    left = {pn: res["stats"][pn]["usable_groups"] - W.N_DEV for pn in W.PANELS}
    summary = {
        "audit": res["audit"],
        "stats": res["stats"],
        "fresh_groups_left_after_wme_w": left,
        "n_exposed_hashes": len(exposed),
        "exposures": sorted(X.EXPOSURES),
    }
    print(json.dumps(summary, indent=1))
    if a.dry_run:
        return
    PP.write_panel(
        out,
        res["splits"],
        W.SALT,
        {
            "label": "EXPLORATORY (WM-E stage W)",
            "plan": "docs/council/2026-11-18/RULING-wme-stagew.md",
            "sources": meta,
            "exposures_excluded": sorted(X.EXPOSURES),
            "n_exposed_hashes": len(exposed),
            "audit": res["audit"],
            "stats": res["stats"],
            "fresh_groups_left_after_wme_w": left,
            "sizes": {pn: {"dev": W.N_DEV} for pn in W.PANELS},
        },
    )
    print("written", out)


if __name__ == "__main__":
    main()
