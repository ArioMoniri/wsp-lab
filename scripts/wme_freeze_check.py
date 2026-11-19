#!/usr/bin/env python3
"""Check the PREREG_WME freeze conditions (Part I preamble (i)-(iv) and Part II's (i)-(iv)) and write the hash-only
evidence report. Reads only docs/ and the manifest; never opens panel, split or data files.

    python scripts/wme_freeze_check.py --out docs/reviews/wme/wme-freeze-check-2026-11-19.json

Exits non-zero if any condition fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/PREREG_WME.md"
MAN = ROOT / "docs/prereg/wme_manifest.json"
PLACEHOLDER = re.compile(r"\[(MANIFEST|SIM):\s*(.*?)\]", re.DOTALL)
CONVENTION = {"<field>", "<name>", "…"}  # the placeholder-convention sentences, not placeholders
ALIASES = {
    ("MANIFEST", "bank.roots[r"): "bank.roots",  # [MANIFEST: bank.roots[r].control_seed]
    ("MANIFEST", "e2.episodes[e"): "e2.episodes",  # [MANIFEST: e2.episodes[e].seed]
    ("MANIFEST", "audit.available_groups.<objective>"): "audit.available_groups",
    ("MANIFEST", "audit.available_groups.*"): "audit.available_groups",
    ("SIM", "n_<split>,<objective>"): "n_bank",
}
STAGE_REPORTS = (
    "docs/reviews/wme/wme-stagew-2026-11-18.json",
    "docs/reviews/wme/wme-stagew2-2026-11-18.json",
)
LINEAGE = (
    "docs/reviews/wme/wme-v2e-panel-audit-2026-11-19.json",
    "docs/reviews/wme/wme-v2f-root-lineage-2026-11-19.json",
)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def placeholders(text: str, offset_line: int = 0):
    out = []
    for m in PLACEHOLDER.finditer(text):
        name = " ".join(m.group(2).split())
        out.append((m.group(1), name, text.count("\n", 0, m.start()) + 1 + offset_line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    text = DOC.read_text()
    lines = text.split("\n")
    p2 = next(i for i, x in enumerate(lines) if x.startswith("# PART II"))
    c0 = next(i for i, x in enumerate(lines) if x.startswith("## Appendix C."))
    c1 = next(i for i, x in enumerate(lines[c0 + 1 :], c0 + 1) if x.startswith("## "))
    part1, appc = "\n".join(lines[:p2]), "\n".join(lines[c0:c1])
    man = json.loads(MAN.read_text())

    # ---- (ii)/(iii): every Part I placeholder is filled in Appendix C; none is open elsewhere
    c_man = set(re.findall(r"^\| `\[MANIFEST: ([^\]]+)\]` \|", appc, re.MULTILINE))
    c_sim = set()
    for row in re.findall(
        r"^\| (`[^|]+) \| ([^|]+) \|$", appc[appc.index("### C.2") :], re.MULTILINE
    ):
        c_sim |= set(re.findall(r"`([^`]+)`", row[0]))
    rows = []
    for kind, name, line in placeholders(part1):
        if name in CONVENTION or name.startswith("<"):
            rows.append(
                {"line": line, "kind": kind, "placeholder": name, "status": "convention text"}
            )
            continue
        key = ALIASES.get((kind, name), name)
        filled = key in (c_man if kind == "MANIFEST" else c_sim)
        in_manifest = key in man if kind == "MANIFEST" else None
        rows.append(
            {
                "line": line,
                "kind": kind,
                "placeholder": name,
                "appendix_c_row": key,
                "filled": filled,
                "manifest_field_present": in_manifest,
            }
        )
    open_rows = [
        r for r in rows if r.get("filled") is False or r.get("manifest_field_present") is False
    ]
    c_fields_missing = sorted(f for f in c_man if f not in man)
    outside = [
        {"kind": k, "placeholder": n, "line": ln}
        for k, n, ln in placeholders("\n".join(lines[p2:]), p2)
        if not (c0 < ln <= c1) and n not in CONVENTION
    ]

    # ---- (iv): manifest, simulation and stage reports, panels
    sums = {}
    for f in subprocess.run(
        ["git", "ls-files", "docs"], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.split():
        p = ROOT / f
        if p.is_file():
            sums.setdefault(sha(p), []).append(f)
    sim = {}
    for k, h in sorted(man["simulation.report_hash"].items()):
        sim[k] = {"sha256": h, "files": sums.get(h, [])}
    sim["full_refit"] = {
        "sha256": man["simulation.full_refit_hash"],
        "files": sums.get(man["simulation.full_refit_hash"], []),
    }
    sim_ok = all(len(v["files"]) == 1 for v in sim.values())
    stage = {f: sha(ROOT / f) for f in STAGE_REPORTS}
    audits = {f: sha(ROOT / f) for f in LINEAGE}
    panels = {
        "wme-c1": {
            k: man["panel.wme_c1.split_hashes"][k] for k in ("manifest_sha256", "content_hash")
        },
        "wme-c2-gen": {
            k: man["panel.wme_c2_gen"]["split_hashes"][k]
            for k in ("manifest_sha256", "content_hash")
        },
    }
    panels_ok = bool(man["audit.panels_v2e"]["clean"]) and all(
        re.fullmatch(r"[0-9a-f]{64}", v) for p in panels.values() for v in p.values()
    )

    # ---- (i): the Board A acceptance chain
    chain = man["prereg_v2"].get("freeze", {}).get("board_a_chain", [])
    chain_ok = bool(chain) and all(sha(ROOT / c["review"]) == c["sha256"] for c in chain)
    last = chain[-1] if chain else {}
    last_verdict = ""
    if last:
        m = re.search(r"\*\*Verdict: ([^*]+?)\.?\*\*", (ROOT / last["review"]).read_text())
        last_verdict = m.group(1) if m else ""
    accept_ok = chain_ok and last_verdict.startswith("ACCEPT")
    conditions_recorded = "## II.28 " in text and all(
        s in text.lower()
        for s in (
            "serialized, including across threads",
            "immutable-prefix storage is an operating assumption",
            "coordinated rollback",
        )
    )
    w2_known = man["prereg_v2"]["claims"]["H-T3"]["type"].startswith("not registered")

    conditions = {
        "I(i)/II(i) Board A acceptance chain": {
            "met": accept_ok and conditions_recorded,
            "chain": chain,
            "last_verdict": last_verdict,
            "freeze3_conditions_recorded_in_II.28": conditions_recorded,
        },
        "I(ii) every SIM placeholder filled": {
            "met": not [r for r in open_rows if r["kind"] == "SIM"],
            "part_I_sim_placeholders": sum(1 for r in rows if r["kind"] == "SIM" and "filled" in r),
        },
        "I(iii) every MANIFEST placeholder filled": {
            "met": not [r for r in open_rows if r["kind"] == "MANIFEST"] and not c_fields_missing,
            "part_I_manifest_placeholders": sum(
                1 for r in rows if r["kind"] == "MANIFEST" and "filled" in r
            ),
            "appendix_c_fields_missing_from_manifest": c_fields_missing,
        },
        "no open placeholder in Part II outside Appendix C": {"met": not outside, "found": outside},
        "II(ii) freeze-build prerequisites (panels built and re-audited after every exposure)": {
            "met": panels_ok,
            "audit.panels_v2e.clean": man["audit.panels_v2e"]["clean"],
            "panels": panels,
            "audit_reports_sha256": audits,
        },
        "II(iii) stage-W2 outcome known": {
            "met": w2_known,
            "H-T3": man["prereg_v2"]["claims"]["H-T3"]["type"],
        },
        "I(iv)/II(iv) hashes recorded in the freeze commit": {
            "met": sim_ok,
            "manifest_sha256": sha(MAN),
            "simulation_reports": sim,
            "stage_reports_sha256": stage,
        },
    }
    out = {
        "label": "PREREG_WME freeze conditions (hash-only; no panel, split or data file opened)",
        "manifest": str(MAN.relative_to(ROOT)),
        "manifest_sha256": sha(MAN),
        "manifest_generated_from_commit": man["generated_from_commit"],
        "conditions": conditions,
        "all_met": all(c["met"] for c in conditions.values()),
        "placeholders": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for k, c in conditions.items():
        print(("MET  " if c["met"] else "OPEN ") + k)
    print("open placeholders:", open_rows or "none", "| outside Appendix C:", outside or "none")
    sys.exit(0 if out["all_met"] else 1)


if __name__ == "__main__":
    main()
