"""CPU-only descriptive council audit; run after `. scripts/env.sh`."""

import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

zoo = Path("data/zoo/qwen3.5-0.8b")
manifest = zoo / "manifest.jsonl"
rows = [json.loads(s) for s in manifest.read_text().splitlines()]
entries = [r for r in rows if r.get("family")]
edits = {r["id"]: json.loads((zoo / r["id"] / "edit.json").read_text()) for r in entries}
print("manifest_sha256", hashlib.sha256(manifest.read_bytes()).hexdigest())
print(
    "rows / entries / unique / headers",
    len(rows),
    len(entries),
    len(edits),
    len(rows) - len(entries),
)
for fam in sorted({r["family"] for r in entries}):
    rs = [r for r in entries if r["family"] == fam]
    prod = [edits[r["id"]] for r in rs if r.get("eligibility") == "production"]
    norms = [e["delta_fro_norm"] for e in prod]
    print(
        "family",
        fam,
        "total",
        len(rs),
        "eligibility",
        dict(collections.Counter(r.get("eligibility", "missing") for r in rs)),
        "production_norm_min_median_max",
        np.quantile(norms, [0, 0.5, 1]).tolist() if norms else None,
    )
    if prod:
        params = collections.defaultdict(collections.Counter)
        for e in prod:
            for k, v in e["params"].items():
                if k not in ("inner_params",):
                    params[k][json.dumps(v, sort_keys=True)] += 1
        print(
            "parameters",
            {k: dict(v) if len(v) <= 15 else {"distinct": len(v)} for k, v in params.items()},
        )
chains = [
    edits[r["id"]]
    for r in entries
    if r["family"] == "chain" and r.get("eligibility") == "production"
]
print("chain roots", sum(e.get("parent_id") is None for e in chains))
sft = [e for e in edits.values() if e["family"] == "lora_sft"]
print(
    "sft timing fields",
    sorted(
        {
            k
            for e in sft
            for k in e
            if any(s in k.lower() for s in ("time", "elapsed", "duration", "second"))
        }
    ),
)
print(
    "sft steps sum production",
    sum(e["params"]["steps"] for e in sft if e.get("eligibility") == "production"),
)
run = Path("data/runs/p4/d3v2-m2-trajectory/seed-0")
df = pd.concat([pd.read_parquet(f) for f in sorted(run.glob("fold-*.parquet"))], ignore_index=True)
print(
    "label source",
    str(run),
    "rows",
    len(df),
    "edits",
    df.id.nunique(),
    "clusters",
    df.ancestry_cluster.nunique(),
)
for (fam, t), g in df[df.target.isin(["mmlu_pro", "hellaswag"])].groupby(["family", "target"]):
    a = g.observed.to_numpy()
    print(
        "range",
        fam,
        t,
        "n",
        len(a),
        "min_median_max",
        np.quantile(a, [0, 0.5, 1]).round(8).tolist(),
        "unique",
        len(np.unique(a)),
        "zero_fraction",
        round(float(np.mean(abs(a) < 1e-7)), 5),
        "sd",
        round(float(np.std(a)), 6),
    )
print(
    "length>=4 unique / clusters",
    df[df.trajectory_length >= 4].id.nunique(),
    df[df.trajectory_length >= 4].ancestry_cluster.nunique(),
)
for name in ["m1_matched", "m2", "m3"]:
    scores = collections.defaultdict(list)
    for seed in range(3):
        fs = list(Path(f"data/runs/p4/d3v2-{name}-trajectory/seed-{seed}").glob("fold-*.parquet"))
        if not fs:
            continue
        d = pd.concat([pd.read_parquet(f) for f in fs])
        for t, g in d[d.target.isin(["mmlu_pro", "hellaswag"])].groupby("target"):
            for pred in ["predicted", "statnn", "delta_norm"]:
                scores[t, pred].append(float(g.observed.corr(g[pred], method="spearman")))
    print("rho", name, {str(k): {"seeds": v, "mean": float(np.mean(v))} for k, v in scores.items()})
print("closed ledger sum", sum([7.29, 37.41, 0.60, 0.60, 38.33, 11.48]))
for n in [56, 240, 320, 480, 640]:
    print(
        "budget",
        n,
        "eval_local_h",
        n * 7.5 / 60,
        "eval_h100_h",
        n * 3.1 / 60,
        "features_h",
        n * 66 / 3600,
        "h100_eval_dollars",
        n * 3.1 / 60 * 3.29,
    )
