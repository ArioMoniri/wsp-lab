#!/usr/bin/env python3
"""D8 step 5d: reproduce the 28 recoverable per-entry (y, h) rows of docs/reviews/P5/tf-v1-requalification-2026-11-06.md.

    d8_tf_yh.py prepare     # stage a zoo (symlinks; aliases materialised) and write the ids to score
    d8_tf_yh.py audit       # compare y and h of the re-scored entries with the printed 4-digit values

y = mean over the 1,000 MMLU-Pro items of (entry - reference) gold log-prob, reference = scored predecessor (chain) or the
base; h = 95% item-bootstrap half-width from the registered common joint draw (tfv1_requalify.item_halfwidths, seed 20261105,
4,000 draws). Scored ids follow the alias ledger (tfv1_scoring_plan.scored_id). Rows whose own or reference bytes are lost
are not reproducible (14 of 42 listed rows).
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/ubuntu/wsp-data")
ZOO = DATA / "zoo/qwen3.5-0.8b"
STAGE = DATA / "runs/d8/stage_zoo/qwen3.5-0.8b"
IDS = DATA / "runs/d8/tf_yh_ids.txt"
TF = DATA / "evals/qwen3.5-0.8b/tf_v1"
DOC = ROOT / "docs/reviews/P5/tf-v1-requalification-2026-11-06.md"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def printed_rows():
    out = []
    for line in DOC.read_text().splitlines()[22:66]:
        m = re.match(r"\| `([0-9a-f]{16})` \| (\w+) \| (-?[\d.e-]+) \| ([\d.]+) \|", line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    return out


def plan_rows():
    plan = load("tfv1_scoring_plan")
    panel = json.loads((ROOT / "docs/prereg/panel_manifest.json").read_text())
    rows = plan.load_zoo_rows()
    amap = plan.canonical_map(panel["alias_ledger"])
    status = {
        json.loads(x)["id"]: json.loads(x)
        for x in (ROOT / "docs/prereg/artefact_status.jsonl").read_text().splitlines()
    }
    group = {}
    for g in panel["alias_ledger"].values():
        for m in [g["canonical"], *g["aliases"]]:
            group[m] = [g["canonical"], *g["aliases"]]

    def bytes_source(i):  # an entry id whose delta bytes are present and identical to i's
        for m in group.get(i, [i]):
            if status[m]["delta_present"]:
                return m
        return None

    keep = []
    for i, fam, y, h in printed_rows():
        s, r = plan.scored_id(i, amap), plan.reference_of(i, rows, amap)
        need = [s] + ([] if r == "base" else [r])
        if all(bytes_source(n) for n in need):
            keep.append({"entry": i, "family": fam, "y": y, "h": h, "scored": s, "reference": r})
    ids = sorted({n for k in keep for n in (k["scored"], k["reference"]) if n != "base"})
    return keep, ids, bytes_source, status


def prepare():
    keep, ids, bytes_source, _ = plan_rows()
    STAGE.mkdir(parents=True, exist_ok=True)
    for i in ids:
        d = STAGE / i
        src = bytes_source(i)
        if src == i:
            if not d.is_symlink():
                d.symlink_to(ZOO / i)
        else:
            d.mkdir(exist_ok=True)
            for name, target in (("edit.json", ZOO / i / "edit.json"),):
                if not (d / name).is_symlink():
                    (d / name).symlink_to(target)
            t = ZOO / src
            kind = "delta.safetensors" if (t / "delta.safetensors").exists() else "lora"
            if not (d / kind).is_symlink():
                (d / kind).symlink_to(t / kind)
    if not (STAGE / "manifest.jsonl").is_symlink():
        (STAGE / "manifest.jsonl").symlink_to(ZOO / "manifest.jsonl")
    IDS.write_text("\n".join(ids) + "\n")
    print(
        f"{len(keep)} reproducible rows of {len(printed_rows())}; {len(ids)} entries to score; stage {STAGE}"
    )


def audit():
    rq = load("tfv1_requalify")
    keep, _, _, _ = plan_rows()
    out = []
    for k in keep:

        def vals(i):
            ids, v = rq.read_values(TF / i, "mmlu_pro", "gold_log_prob")
            return ids, v

        ie, ve = vals(k["scored"])
        ir, vr = vals("base" if k["reference"] == "base" else k["reference"])
        assert ie == ir
        diff = (ve - vr)[None, :]
        half, _ = rq.item_halfwidths(diff, 1)
        y, h = float(diff.mean()), float(half[0])
        sig = lambda x: float(f"{x:.4g}")  # noqa: E731
        k.update(
            {
                "y_regenerated": y,
                "h_regenerated": h,
                "y_match": sig(y) == float(k["y"]),
                "h_match": sig(h) == float(k["h"]),
            }
        )
        out.append(k)
    res = {
        "rows": out,
        "n": len(out),
        "y_all_match": all(r["y_match"] for r in out),
        "h_all_match": all(r["h_match"] for r in out),
    }
    (DATA / "runs/d8/tf_yh_audit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    for r in out:
        print(
            r["entry"],
            r["family"],
            r["y"],
            f"{r['y_regenerated']:.5g}",
            r["y_match"],
            r["h"],
            f"{r['h_regenerated']:.5g}",
            r["h_match"],
        )


if __name__ == "__main__":
    {"prepare": prepare, "audit": audit}[sys.argv[1]]()
