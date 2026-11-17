#!/usr/bin/env python3
"""Real-provider smoke for the WM-E reader arms (track B). ENGINEERING ONLY: a SYNTHETIC pool and SYNTHETIC own
results; no model, no panel item, no bank, no outcome.

One shared ``TokenLedger`` with a hard cap of 20,000 tokens covers every call (provider-reported usage, cache reads
and writes included). Order: (1) a minimal probe call that measures the CLI's fixed per-call overhead and aborts the
smoke if it is too large to fit the plan; (2) one WSR-style reader round; (3) one Fugu-style team round
(Thinker/Worker/Verifier, at most one revision). Every call is pre-checked against the remaining cap with the
measured overhead, so nothing is sent that could cross it. Codex and Gemini are not called: their agent CLIs carry a
large fixed prompt (Codex: 17,772 input tokens per first turn in a local session log; OpenCode: unmeasured), which
does not fit a 20k total with any guarantee.

    python scripts/wme_arms_smoke.py --out docs/reviews/wme/wme-arms-smoke-2026-11-17.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.agents.reader import estimate_tokens
from wsp.wm.arms import llm as L
from wsp.wm.arms import readers as R

CAP = 20_000
PROBE_MAX_OVERHEAD = 4_000


def synthetic_views():
    rows = [
        ("steer", 20, "syc", 0.0625, 1.1e-3, -0.012, 2.1e-4),
        ("steer", 24, "gsm", -0.03125, 0.6e-3, 0.002, 0.4e-4),
        ("proj_remove", 26, "syc", 0.125, 2.4e-3, -0.020, 6.0e-4),
        ("proj_remove", 30, "trivia", -0.25, 3.9e-3, 0.004, None),
        ("steer", 28, "mathcomp", 0.0625, 1.3e-3, 0.001, 1.2e-4),
        ("lora_tv", 16, "syc", 0.5, 4.2e-3, -0.015, 9.0e-4),
    ]
    out = []
    for i, (fam, layer, con, par, rel, a, q) in enumerate(rows):
        out.append(
            {
                "id": f"c{i:02d}",
                "steps": [
                    {
                        "pid": i,
                        "family": fam,
                        "layer": layer,
                        "contrast": con,
                        "param": par,
                        "rel_norm": rel,
                    }
                ],
                "rel_norm_total": rel,
                "field": {
                    "sycophancy_real": {"a": a, "q": q},
                    "wikitext_nll": {"a": abs(a) / 10, "q": None},
                },
            }
        )
    return out


def synthetic_history(views):
    return [
        {
            "candidate": (90,),
            "view": dict(views[0], id="h"),
            "gain": 0.031,
            "se": 0.012,
            "lcb": 0.0075,
            "guard_ok": True,
            "order": 0,
        },
        {
            "candidate": (91,),
            "view": dict(views[2], id="h"),
            "gain": 0.054,
            "se": 0.015,
            "lcb": 0.0246,
            "guard_ok": False,
            "order": 1,
        },
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-haiku-5-5")
    args = ap.parse_args()
    ledger = L.TokenLedger(CAP)
    client = L.ClaudeCLI(args.model, overhead_tokens=PROBE_MAX_OVERHEAD, timeout=180)
    rec = {
        "status": "ENGINEERING SMOKE - synthetic pool and results; no model, item, bank or outcome",
        "cap": CAP,
        "provider": "claude",
        "model": args.model,
        "not_called": {
            "codex": "declared per-call overhead 17,772 tokens (local session log): no room under a 20k total",
            "gemini": "OpenCode agent overhead unmeasured: the 20k cap could not be guaranteed",
        },
        "steps": {},
    }
    # (1) probe: measures the fixed per-call overhead with a minimal prompt
    sys_p, user_p = 'Reply with exactly {"ok": true}.', "ping"
    c, f = L.guarded_call(client, ledger, "probe", sys_p, user_p, 32)
    overhead = None
    if c is not None and c.usage_source == "provider":
        overhead = max(0, c.input_tokens - estimate_tokens(sys_p) - estimate_tokens(user_p))
    rec["steps"]["probe"] = {
        "failure": f,
        "text_ok": bool(c and c.text.strip().startswith("{")),
        "measured_overhead_tokens": overhead,
    }
    if overhead is None or overhead > PROBE_MAX_OVERHEAD:
        rec["aborted"] = "probe failed or per-call overhead above the plan"
    else:
        client.overhead_tokens = int(overhead * 1.1) + 50
        views, hist = synthetic_views(), synthetic_history(synthetic_views())
        task = R.ReaderTask("sycophancy_real")
        res = R.WSRReader(client, ledger, task).rank(views, hist, b=2, round_index=0)
        rec["steps"]["wsr_reader"] = {"failure": res.failure, "ranking": res.order}
        team = R.FuguTeam({r: client for r in R.FuguTeam.ROLES}, ledger, task)
        res = team.rank(views, hist, b=2, round_index=0)
        rec["steps"]["fugu_team"] = {
            "failure": res.failure,
            "ranking": res.order,
            "calls_by_role": [u["role"] for u in res.usage],
        }
    rep = ledger.report()
    rec["ledger"] = rep
    rec["calls"] = [
        {
            k: r[k]
            for k in (
                "role",
                "provider",
                "model",
                "input_tokens",
                "output_tokens",
                "total",
                "usage_source",
                "truncated",
                "refused",
                "failure",
                "seconds",
            )
        }
        for r in ledger.records
    ]
    rec["total_tokens"] = rep["spent"]
    rec["within_cap"] = rep["spent"] <= CAP
    rec["code_sha256"] = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in (
            "src/wsp/wm/arms/llm.py",
            "src/wsp/wm/arms/readers.py",
            "scripts/wme_arms_smoke.py",
        )
    }
    Path(args.out).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print(
        json.dumps(
            {
                "total_tokens": rec["total_tokens"],
                "within_cap": rec["within_cap"],
                "steps": rec["steps"],
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
