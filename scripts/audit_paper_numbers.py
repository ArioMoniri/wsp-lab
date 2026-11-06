#!/usr/bin/env python3
"""Independent numerical audit of the v2 paper sections.

Scope: paper/sections/{abstract,introduction,method,zoo,experiments,results,reliability,
discussion}.tex. Every numeric token in those files is classified as one of:

  CHECKED     inside a claim whose expected value is recomputed from a source artefact
              (runs/p4 metrics.json, seed-0 baselines and paired differences, the evidence
              CSV, metrology and residual outputs, the P2 base run, per-entry eval.json
              files, the zoo manifest, the feature manifest), or inside a table cell that a
              table checker parses and recomputes;
  PRESENCE    a quoted threshold or identifier that must appear in the cited document;
  DECLARED    on the explicit list in this script with a reason (hardware, literature values
              from cited papers, the Takehi blog column, plan parameters, dates, method
              labels, seed labels). Listed in the report; does not fail the audit;
  UNCLAIMED   none of the above. Fails the audit.

A checked value MISMATCHes when the displayed number differs from the recomputed value by more
than half a unit in its last displayed decimal. A snippet that is not found in the tex is MISSING.
Locator numbers (line numbers in cited outputs) are checked against the cited file. Every
\\path{...} in the audited sections must exist. The audit exits non-zero on any mismatch, missing
snippet, failed assertion, or unclaimed token.

Usage:  python3 scripts/audit_paper_numbers.py [--report PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "paper" / "sections"
DATA = ROOT / "data"
# The legacy checks read run artefacts that the 6 Nov 2026 incident (D8) removed. They run only when
# these marker artefacts are restored; otherwise only committed-document checks run (--committed-only).
DATA_MARKERS = [
    DATA / "runs" / "p4" / "d3v2-m1-trajectory" / "metrics.json",
    DATA / "runs" / "h1a" / "requalification_report.json",
    DATA / "runs" / "p2" / "base-final-4096",
]
DATA_PRESENT = all(p.exists() for p in DATA_MARKERS)
DOCS = ROOT / "docs"
P4 = DATA / "runs" / "p4"
EV = DATA / "evals" / "qwen3.5-0.8b"
COUNCIL = DOCS / "council" / "2026-11-05"
EVIDENCE_CSV = COUNCIL / "evidence-statnn-importances.csv"
METROLOGY = COUNCIL / "metrology_budget_output.txt"
RESIDUAL = COUNCIL / "round2_residual_estimand_output.txt"
RESIDUAL_METRO = COUNCIL / "round2_metrology_output.txt"
BASE = DATA / "runs" / "p2" / "base-final-4096"
ZOO_MANIFEST = DATA / "zoo" / "qwen3.5-0.8b" / "manifest.jsonl"
FEATURE_MANIFEST = DATA / "features" / "qwen3.5-0.8b" / "manifest.json"
AUDIT_TXT = P4 / "board-a-d3-audit-20261104" / "audit.txt"
LAMBDA = ROOT / "board" / "lambda_spend.md"
RESULTS_MD = DOCS / "RESULTS.md"
DECISIONS = DOCS / "DECISIONS.md"
HYPOTHESIS = DOCS / "HYPOTHESIS.md"
PREREG = DOCS / "PREREG_H1_TF.md"
PREREG_H5 = DOCS / "reviews" / "H5" / "board-a-prereg-2026-11-04.md"
POOL_PY = ROOT / "src" / "wsp" / "features" / "pool.py"
REFUSAL_PY = ROOT / "src" / "wsp" / "edits" / "refusal.py"
SECTIONS = [
    "abstract",
    "introduction",
    "method",
    "zoo",
    "experiments",
    "results",
    "reliability",
    "discussion",
    "appendix",
]

# ------------------------------------------------------------------ number syntax

NEG = r"(?:\$-\$|\u2212|-)"
# A sign is accepted only when not glued to a word, digit, dot or hyphen on its left.
# Unsigned digits are accepted when not glued to a word or dot, nor to "word-" (so MATH-500,
# Qwen3.5-0.8B and d3v2-m1 give no numbers), while a double hyphen before digits is allowed
# ("11.1--21.2"). A number is not followed by a word character or ".word" ("0.8B" gives nothing).
TOKEN_RE = re.compile(
    r"(?:(?<![\w.\-])" + NEG + r"|(?<![\w.])(?<!\w-))"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|\.\w)"
)
SEED_BEFORE = re.compile(r"[Ss]eeds?[ \-]$")
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def parse_shown(tok: str) -> tuple[float, int]:
    t = tok.replace("$-$", "-").replace("\u2212", "-").replace(",", "")
    decimals = len(t.split(".")[1]) if "." in t else 0
    return float(t), decimals


def same(shown: str, value: float) -> bool:
    v, d = parse_shown(shown)
    return abs(v - value) <= 0.5 * 10 ** (-d) + 1e-9


def mask_nonprose(raw: str) -> str:
    """Blank LaTeX that carries identifiers, labels, citations, layout, interval-level labels or
    math sub/superscripts. The output has the same length as the input."""
    out = list(raw)

    def blank(a: int, b: int) -> None:
        for i in range(a, b):
            if out[i] != "\n":
                out[i] = " "

    for m in re.finditer(r"(?<!\\)%[^\n]*", raw):
        blank(m.start(), m.end())
    text = "".join(out)
    patterns = [
        r"\\(?:label|ref|input|path|texttt|url)\{[^{}]*\}",
        r"\\cite[a-zA-Z]*(?:\[[^\]]*\])?\{[^{}]*\}",
        r"\\begin\{tabular\*?\}\{(?:[^{}]|\{[^{}]*\})*\}",
        r"\\(?:begin|end)\{[^{}]*\}",
        r"\\(?:section|subsection|paragraph)\{[^{}]*\}",
        r"\\multicolumn\{\d+\}\{[^{}]*\}",
        r"\\cmidrule(?:\([^)]*\))?\{[^{}]*\}",
        r"\\setlength\{[^{}]*\}\{[^{}]*\}",
        r"\\resizebox\{[^{}]*\}\{[^{}]*\}",
        r"\\(?:top|mid|bottom)rule",
        r"_\{[^{}]*\}",
        r"\^\{[^{}]*\}",
        r"_\\?\w",
        r"\^\\?\w",
        r"ICC\(A,1\)",
        r"95\\%",
        r"\\\\",
        r"\\[a-zA-Z]+",
    ]
    for p in patterns:
        for m in re.finditer(p, text):
            blank(m.start(), m.end())
        text = "".join(out)
    return "".join(out)


def numeric_tokens(masked: str, lo: int = 0, hi: int | None = None) -> list[dict]:
    hi = len(masked) if hi is None else hi
    res = []
    for m in TOKEN_RE.finditer(masked, lo, hi):
        before = masked[max(0, m.start() - 8) : m.start()]
        res.append(
            {
                "start": m.start(),
                "end": m.end(),
                "text": m.group(0),
                "seed": bool(SEED_BEFORE.search(before)),
            }
        )
    return res


# ------------------------------------------------------------------ source loaders

_cache: dict = {}


def jload(path: Path):
    key = ("json", str(path))
    if key not in _cache:
        _cache[key] = json.loads(path.read_text())
    return _cache[key]


def read(path: Path) -> str:
    key = ("text", str(path))
    if key not in _cache:
        _cache[key] = path.read_text()
    return _cache[key]


def metrics(run: str):
    return jload(P4 / run / "metrics.json")


def seed0(run: str):
    return jload(P4 / run / "seed-0" / "metrics.json")


def ss(run: str, target: str, metric: str, stat: str) -> float:
    node = metrics(run)["seed_summary"][target][metric]
    if stat == "mean":
        return node["mean"]
    return node["range"][0] if stat == "min" else node["range"][1]


def bl(run: str, model: str, target: str):
    return seed0(run)["baselines"][model]["targets"][target]["spearman"]["value"]


def own_mae(run: str, target: str) -> float:
    return seed0(run)["targets"][target]["mae"]["value"]


def paired(run: str, model: str, target: str) -> dict:
    return seed0(run)["paired_differences"][model][target]


def n_rows(run: str, target: str = "mmlu_pro") -> int:
    return seed0(run)["targets"][target]["n"]


def n_clusters(run: str, model: str = "statnn", target: str = "mmlu_pro") -> int:
    return paired(run, model, target)["n_clusters"]


def evidence_rows() -> list[list[str]]:
    key = ("csv",)
    if key not in _cache:
        with EVIDENCE_CSV.open() as fh:
            _cache[key] = list(csv.reader(fh))
    return _cache[key]


def ev_perf(target: str, model: str, field: str) -> float:
    idx = {"spearman": 4, "r2": 5}[field]
    for r in evidence_rows():
        if r[0] == "performance" and r[1] == target and r[3] == model:
            return float(r[idx])
    raise KeyError((target, model, field))


def ev_family(target: str, family: str) -> dict:
    for r in evidence_rows():
        if r[0] == "family_distribution" and r[1] == target and r[13] == family:
            return {
                "n": int(r[2]),
                "mean": float(r[14]),
                "sd": float(r[15]),
                "min": float(r[16]),
                "max": float(r[17]),
            }
    raise KeyError((target, family))


def ev_within(target: str, family: str):
    for r in evidence_rows():
        if r[0] == "within_family" and r[1] == target and r[13] == family:
            return float(r[4]) if r[4] != "" else None
    raise KeyError((target, family))


def ev_within_n(target: str, family: str) -> int:
    for r in evidence_rows():
        if r[0] == "within_family" and r[1] == target and r[13] == family:
            return int(r[2])
    raise KeyError((target, family))


def ev_top5_share(target: str) -> float:
    return sum(
        float(r[12])
        for r in evidence_rows()
        if r[0] == "importance" and r[1] == target and r[6] and float(r[6]) <= 5
    )


def base_summary() -> dict:
    return jload(BASE / "summary.json")


def base_records() -> list[dict]:
    key = ("records",)
    if key not in _cache:
        with (BASE / "records.jsonl").open() as fh:
            _cache[key] = [json.loads(line) for line in fh if line.strip()]
    return _cache[key]


def cap_hits(benchmarks: list[str]) -> tuple[int, int]:
    hit = tot = 0
    for r in base_records():
        if r.get("edit_id") == "base" and r.get("benchmark") in benchmarks:
            tot += 1
            hit += bool(r.get("hit_max_tokens"))
    return hit, tot


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def binom_se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def jeval(p: Path) -> dict:
    return jload(p)


_CONTROLS_FILE = DATA / "runs" / "d3_shards" / "controls.txt"
CONTROLS = (
    [l.strip() for l in _CONTROLS_FILE.read_text().split() if l.strip()]
    if _CONTROLS_FILE.exists()
    else []
)
H100_IDS = CONTROLS + ["013b7b61202fe188"]
A10_IDS = ["0a841d4082532e8b", "3db97b535c996265", "af6d608f8651fb46", "dd90d99e2ddc0e77"]
DEV_KEYS = {  # table label -> eval.json key
    "GSM8K": "gsm8k",
    "MATH-500": "math500",
    "IFEval": "ifeval",
    "Over-refusal": "over_refusal",
    "Refusal": "refusal_rate",
    "MMLU-Pro": "mmlu_pro",
    "HellaSwag": "hellaswag",
    "KL": "kl_to_base",
    "WikiText": "wikitext_ppl",
    "Cap-hit": "loop_rate",
}


def local_eval(entry: str) -> Path:
    p = EV / entry / "eval.local-run1.json"
    return p if p.exists() else EV / entry / "eval.json"


def h100_eval(entry: str) -> Path:
    p = EV / entry / "eval.lambda-209-20-158-168.json"
    if p.exists():
        return p
    (p,) = (EV / entry / "devices" / "H1").glob("*/eval.json")
    return p


def a10_eval(entry: str) -> Path:
    return EV / entry / "eval.lambda-129-153-20-232.json"


def avg_ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    ra, rb = avg_ranks(a), avg_ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb)


def cross_host(entries: list[str], other, key: str, drop: str | None = None) -> dict:
    ids = [e for e in entries if e != drop]
    lv = [float(jeval(local_eval(e))[key]) for e in ids]
    ov = [float(jeval(other(e))[key]) for e in ids]
    diffs = [abs(a - b) for a, b in zip(lv, ov)]
    return {"mean": sum(diffs) / len(diffs), "max": max(diffs), "rho": spearman(lv, ov)}


def base_val(key: str) -> float:
    return float(jeval(local_eval("base"))[key])


def base_dev(dev: str, key: str) -> float:
    (p,) = (EV / "base" / "devices" / dev).glob("*/eval.json")
    return float(jeval(p)[key])


try:
    DEVS = {
        "H1": {lab: cross_host(H100_IDS, h100_eval, key) for lab, key in DEV_KEYS.items()},
        "H2": {lab: cross_host(A10_IDS, a10_eval, key) for lab, key in DEV_KEYS.items()},
    }
except FileNotFoundError:  # per-entry eval.json files are lost (D8); legacy checks are not run
    DEVS = {}


def metrology_row(target: str, scope: str) -> list[str]:
    for line in METROLOGY.read_text().splitlines():
        if re.match(rf"\s*{target}\s+{re.escape(scope)}\s+\d+\s", line):
            return line.split()
    raise KeyError((target, scope))


def metrology_seed_sd(target: str) -> float:
    for line in METROLOGY.read_text().splitlines():
        m = re.match(rf"\s*{target}\s+sd_seed=([\d.]+)", line)
        if m:
            return float(m.group(1))
    raise KeyError(target)


def residual_block(target: str = "mmlu_pro") -> dict:
    block = re.search(rf'"{target}":\s*\{{(.*?)\n \}}', read(RESIDUAL), re.DOTALL).group(1)
    ci = re.search(r'"boot_ci":\s*\[\s*([\d.\-]+),\s*([\d.\-]+)\s*\]', block)
    return {
        "theta": float(re.search(r'"theta":\s*(-?[\d.]+)', block).group(1)),
        "lo": float(ci.group(1)),
        "hi": float(ci.group(2)),
        "n": int(re.search(r'"n":\s*(\d+)', block).group(1)),
        "clusters": int(re.search(r'"clusters":\s*(\d+)', block).group(1)),
    }


RESID = residual_block()
RESID_HS = residual_block("hellaswag")
MET_IDX = {"sd_dy": -6, "sd_seed": -5, "sd_itemSE": -4, "sd_signal_pop": -3}
WITHIN = "within family x device"


def met(target: str, scope: str, field: str) -> float:
    """Field of a metrology_budget_output.txt row (token positions counted from the row end)."""
    return float(metrology_row(target, scope)[MET_IDX[field]])


def audit_counter(name: str) -> int:
    return int(re.search(rf"'{name}': (\d+)", read(AUDIT_TXT)).group(1))


try:
    V2_INFERRED = jload(P4 / "d3v2-m1-trajectory" / "seed-1" / "manifest.json")[
        "targets_with_inferred_runtime"
    ]
except FileNotFoundError:  # run directories are lost (D8)
    V2_INFERRED = None
ZOO_EVAL = {c: DATA / "zoo" / "qwen3.5-0.8b" / c / "eval.json" for c in CONTROLS}
BASE_H0_SEED0 = (
    DATA
    / "evals"
    / "qwen3.5-0.8b"
    / "base"
    / "devices"
    / "H0"
    / "20261104-233637-7c43da7c"
    / "eval.json"
)
SEED1_PANEL = DATA / "runs" / "p2-batch" / "20261105-050543-aad6c310"


def seed_panel(key: str) -> tuple[float, float, float]:
    """Seed 0 against seed 1 over the 10 controls and the base (11 records): mean and max absolute
    difference, and Spearman across the records. Recomputed from the raw eval.json files."""
    ids = CONTROLS + ["BASE"]
    s0 = [float(jeval(ZOO_EVAL[c])[key]) for c in CONTROLS] + [float(jeval(BASE_H0_SEED0)[key])]
    s1 = [
        float(jeval(SEED1_PANEL / c / "repeats" / "seed1" / "eval.json")[key]) for c in CONTROLS
    ] + [float(jeval(SEED1_PANEL / "base" / "repeats" / "seed1" / "eval.json")[key])]
    assert len(ids) == len(s0) == len(s1) == 11
    d = [abs(a - b) for a, b in zip(s0, s1)]
    return sum(d) / len(d), max(d), spearman(s0, s1)


SUITE_RUNS = [p.name for p in sorted(P4.glob("d3v2-*")) if (p / "metrics.json").exists()]
LEARNED = {
    "family": [
        "d3v2-m1-family",
        "d3v2-m1_matched-family",
        "d3v2-m2-family",
        "d3v2-m2_matched-family",
        "d3v2-m5-family",
        "d3v2-m6-family",
    ],
    "seed": [
        "d3v2-m1-seed",
        "d3v2-m1_matched-seed",
        "d3v2-m2-seed",
        "d3v2-m2_matched-seed",
        "d3v2-m3-seed",
        "d3v2-m5-seed",
        "d3v2-m6-seed",
    ],
    "trajectory": [
        "d3v2-m1-trajectory",
        "d3v2-m1_matched-trajectory",
        "d3v2-m2-trajectory",
        "d3v2-m2_matched-trajectory",
        "d3v2-m3-trajectory",
        "d3v2-m5-trajectory",
        "d3v2-m6-trajectory",
    ],
}
HOLDOUTS = ["abliteration", "chain", "lora_sft", "random_dir", "task_vector"]
M1M = "d3v2-m1_matched-trajectory"
M1T = "d3v2-m1-trajectory"
WITHIN_FAMILIES = ["abliteration", "chain", "lora_sft", "random_dir", "task_vector"]


def finite(vals) -> list[float]:
    return [v for v in vals if v is not None and not math.isnan(v)]


# ------------------------------------------------------------------ ledger

RESULTS: list[dict] = []
DECLARED: list[dict] = []


def add(kind: str, file: str, label: str, shown: str, expected, ok: bool, src: str) -> None:
    RESULTS.append(
        {
            "kind": kind,
            "file": file,
            "label": label,
            "shown": shown,
            "expected": expected,
            "ok": bool(ok),
            "src": src,
        }
    )


def assertion(file: str, label: str, ok: bool, src: str) -> None:
    add("ASSERT", file, label, "", None, ok, src)


# ------------------------------------------------------------------ claim registry


class Claim:
    def __init__(self, file: str, snippet: str, values, src: str):
        self.file, self.snippet, self.values, self.src = file, snippet, values, src


CLAIMS: list[Claim] = []
DECLARED_RULES: list[tuple[str, str, str]] = []
FAILED_SNIPPETS: list[dict] = []


def claim(file: str, snippet: str, values, src: str) -> None:
    """values: expected numbers, one per numeric token of the snippet, in order.
    values=None: the tokens are covered by a locator or presence check registered elsewhere."""
    CLAIMS.append(Claim(file, snippet, values, src))


def declare(file: str, snippet: str, reason: str) -> None:
    DECLARED_RULES.append((file, snippet, reason))


def presence(file: str, snippet: str, path: Path, pattern: str, desc: str) -> None:
    claim(file, snippet, None, f"quoted from {path.relative_to(ROOT)}")
    assertion(
        file,
        f"quoted in {path.name}: {desc}",
        re.search(pattern, read(path)) is not None,
        str(path.relative_to(ROOT)),
    )


def locator(file: str, snippet: str, src: str) -> None:
    claim(file, snippet, None, src)


def build() -> None:
    # ---------------- abstract
    claim("abstract", "345 matched-reference rows", [n_rows(M1T)], "d3v2-m1-trajectory seed-0 n")
    claim(
        "abstract",
        "mean Spearman 0.45 and 0.46 on the two targets",
        [ss(M1M, "mmlu_pro", "spearman", "mean"), ss(M1M, "hellaswag", "spearman", "mean")],
        "d3v2-m1_matched-trajectory seed_summary (three-seed mean)",
    )
    claim(
        "abstract",
        "Seed-0 StatNN reached 0.57 and 0.65",
        [bl(M1T, "statnn", "mmlu_pro"), bl(M1T, "statnn", "hellaswag")],
        "trajectory seed-0 baselines",
    )
    claim(
        "abstract",
        "reached 0.52 and 0.61 on the same seed and split",
        [bl(M1T, "descriptor_only", "mmlu_pro"), bl(M1T, "descriptor_only", "hellaswag")],
        "trajectory seed-0 baselines, descriptor-only",
    )

    # ---------------- introduction
    declare(
        "introduction",
        "0.016\\% of Qwen3.5-0.8B's weights",
        "literature value from Takehi et al. (2026), cited; not in repository data",
    )
    declare("introduction", "3 and 4 November", "registration dates of H1 to H4 (HYPOTHESIS.md)")
    declare("introduction", "added on 4 November", "addition date of H5 and H6 (HYPOTHESIS.md)")
    declare("introduction", "On 5 November", "amendment date (HYPOTHESIS.md changelog)")
    declare(
        "introduction",
        "from 7.9\\% to 48.1\\%",
        "literature value from Takehi et al. (2026), cited; not in repository data",
    )
    presence(
        "introduction", "$\\rho \\ge 0.6$ on all", HYPOTHESIS, r"ρ ≥ 0\.6", "H1 threshold 0.6"
    ) if False else None
    presence(
        "introduction",
        "per-benchmark Spearman $\\rho \\ge 0.6$",
        HYPOTHESIS,
        r"ρ ≥ 0\.6",
        "H1 threshold 0.6",
    )
    presence(
        "introduction",
        "falsified if $\\rho < 0.3$",
        HYPOTHESIS,
        r"ρ < 0\.3",
        "H1 falsification 0.3",
    )
    presence(
        "introduction",
        "at least 15\\% relative MAE reduction on trajectories of length at least 4",
        HYPOTHESIS,
        r"≥ 15 % relative MAE reduction on trajectories of length ≥ 4",
        "H2 15% and length 4",
    )
    presence("introduction", "transfer $\\rho \\ge 0.5$", HYPOTHESIS, r"\(ρ ≥ 0\.5\)", "H3 0.5")
    presence(
        "introduction", "transfer $\\rho < 0.2$", HYPOTHESIS, r"ρ < 0\.2", "H3 falsification 0.2"
    )

    # ---------------- method
    claim(
        "method",
        "1,242 statistics and 136 singular-spectrum values",
        [seed0(M1T)["statnn"]["n_features"], 136],
        "statnn n_features; 136 = N_SPECTRUM (8) x 17 pooling groups (see assertion)",
    )
    claim(
        "method",
        "feature version 4",
        [jload(FEATURE_MANIFEST)["config"]["feature_version"]],
        "features manifest config.feature_version",
    )
    assertion(
        "method",
        "pooled widths: 18 x (4 buckets x 17 groups + 1) = 1242 and 8 x 17 = 136",
        18 * (4 * 17 + 1) == 1242
        and 8 * 17 == 136
        and "N_STATS, N_SPECTRUM = 18, 8" in read(POOL_PY)
        and jload(FEATURE_MANIFEST)["config"]["buckets"] == 4,
        "src/wsp/features/pool.py; features config",
    )
    presence(
        "method",
        "The pre-registered sequence test sets a bar of at least 15\\% relative MAE reduction on trajectories of length at least four.",
        HYPOTHESIS,
        r"≥ 15 % relative MAE reduction on trajectories of length ≥ 4",
        "H2 bar",
    )

    # ---------------- zoo (count claims live in build_tfv1; these are unchanged)
    claim(
        "zoo",
        "The v2 suite matches 345 of those 389",
        [345, 389],
        "v2 matched rows; eval count",
    )
    claim(
        "zoo",
        "The D3 subset has 300 evaluated entries, and 260 of them have matched-reference targets.",
        [300, 260],
        "board-a-d3-audit audit.txt (test_n 300; matched_reference_targets 260)",
    )
    claim(
        "zoo",
        "verified KL to base of 0.77 to 14.0 nats",
        [0.77, 14.0],
        "docs/DECISIONS.md D6 addendum 2",
    )
    presence(
        "zoo",
        "D6 addendum 2 in \\path{docs/DECISIONS.md}",
        DECISIONS,
        r"D6 addendum 2",
        "D6 addendum 2",
    )
    presence("zoo", "D6 addendum 3 flags them", DECISIONS, r"D6 addendum 3", "D6 addendum 3")
    presence(
        "zoo",
        "\\texttt{ara\\_unconstrained\\_surrogate\\_scale}",
        DECISIONS,
        r"ara_unconstrained_surrogate_scale",
        "flag name",
    )
    presence(
        "zoo",
        "first 400 harmful and harmless dataset prompts",
        REFUSAL_PY,
        r"train\[:400\]",
        "train[:400]",
    )
    claim(
        "zoo",
        "first 400 harmful and harmless dataset prompts",
        [400],
        "src/wsp/edits/refusal.py train[:400]",
    )
    declare("zoo", "5 November 2026", "date")
    declare("zoo", "The builder implements six families.", "")  # word count, no digits
    declare("zoo", "2026", "date")

    # ---------------- experiments
    b_eval = jeval(local_eval("base"))
    claim(
        "experiments",
        "50 GSM8K, 25 MATH-500, 25 IFEval, 100 MMLU-Pro and 100 HellaSwag items",
        [
            b_eval["counts"]["gsm8k"],
            b_eval["counts"]["math500"],
            b_eval["counts"]["ifeval"],
            b_eval["counts"]["mmlu_pro"],
            b_eval["counts"]["hellaswag"],
        ],
        "base per-entry eval counts",
    )
    claim(
        "experiments",
        "plus 50 prompts each for refusal, over-refusal and KL",
        [b_eval["counts"]["refusal_rate"]],
        "base per-entry eval counts (50 prompts)",
    )
    claim(
        "experiments",
        "record a 1,024-token cap",
        [b_eval["generation"]["max_new_tokens"]],
        "zoo evaluation configuration (per-entry eval generation.max_new_tokens)",
    )
    claim(
        "experiments",
        "Generation is capped at 1024 tokens.",
        [b_eval["generation"]["max_new_tokens"]],
        "base per-entry eval generation.max_new_tokens",
    )
    se_fast = [binom_se(0.16, 50), binom_se(0.16, 25), binom_se(0.40, 25)]
    claim(
        "experiments",
        "a single binomial score for GSM8K, MATH-500 or IFEval has a standard error of 0.05 to 0.10",
        [min(se_fast), max(se_fast)],
        "binomial SE at base fast-tier values",
    )
    claim(
        "experiments",
        "(1,024 tokens at the fast tier, 4,096 at the cheap tier)",
        [1024, 4096],
        "per-entry eval max_new_tokens; base-final-4096 generation",
    )
    claim(
        "experiments",
        "with a 4096-token limit",
        [base_summary()["generation"]["max_new_tokens"]],
        "base-final-4096 summary generation.max_new_tokens",
    )
    declare(
        "experiments",
        "except for MMLU-Pro, which has a 1024-token limit",
        "cheap-tier configuration statement; not re-derived here",
    )
    declare(
        "experiments",
        "A faithful 32768-token tier is reserved for the base and at most 10 selected entries.",
        "plan parameters for a tier not yet run",
    )
    declare(
        "experiments",
        "using seed 0 and sorted IDs, with quotas of 60 random direction, 100 abliteration, 60 LoRA SFT, 30 task vector and 50 chain steps",
        "D3 sampling procedure; the quota sum is checked by assertion",
    )
    claim(
        "experiments",
        "these quotas resolve to 300 entries, and all 300 were evaluated",
        [300, 300],
        "quota sum (assertion below); board-a-d3-audit test_n 300",
    )
    declare("experiments", "32 GB", "hardware description")
    declare("experiments", "RTX PRO 4500", "hardware description")
    declare("experiments", "40 GB", "hardware description")
    claim("experiments", "\\$1.99/h", [1.99], "board/lambda_spend.md A100 hourly rate")
    claim(
        "experiments",
        "temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, batch size 64, seed 0",
        [0.6, 0.95, 20, 1.5, 64],
        "base-final-4096 summary generation (seed 0 is a seed label)",
    )
    claim("experiments", "4,096-token cap", [4096], "base-final-4096 generation")
    claim(
        "experiments",
        "The matched set has 345 rows from 389 evaluated entries; the other 44",
        [345, 389, 44],
        "v2 matched rows; eval count; missing_predecessor_targets",
    )
    claim(
        "experiments",
        "lists the 25 complete rows",
        [25],
        "complete d3v2 run directories (assertion below)",
    )
    claim(
        "experiments",
        "since all 60 random-direction edits leave",
        [60],
        "evidence CSV family_distribution n, random_dir",
    )
    claim(
        "experiments",
        "mark 260 of the 345 rows as having inferred runtime metadata",
        [V2_INFERRED, n_rows(M1T)],
        "d3v2-m1-trajectory seed-1 manifest targets_with_inferred_runtime; n",
    )
    claim(
        "experiments",
        "Those 260 are the matched D3 edits counted in Section",
        [audit_counter("matched_reference_targets")],
        "board-a-d3-audit audit.txt COUNTERS matched_reference_targets (D3 subset)",
    )
    claim(
        "experiments",
        "The other 85 rows are chain predecessors",
        [n_rows(M1T) - V2_INFERRED],
        "345 matched rows minus 260 inferred; the 85 are the chain-predecessor rows (run-loader check)",
    )
    claim(
        "experiments",
        "The 345 rows are not the 300-entry D3 subset",
        [n_rows(M1T), 300],
        "v2 n; D3 subset size (300 entries)",
    )
    claim(
        "experiments",
        "the trajectory-split M1-matched row reached mean Spearman 0.45 and 0.46 on MMLU-Pro and HellaSwag",
        [ss(M1M, "mmlu_pro", "spearman", "mean"), ss(M1M, "hellaswag", "spearman", "mean")],
        "d3v2-m1_matched-trajectory seed_summary",
    )
    claim(
        "experiments",
        "Seed-0 StatNN reached 0.57 and 0.65 on the same split",
        [bl(M1T, "statnn", "mmlu_pro"), bl(M1T, "statnn", "hellaswag")],
        "trajectory seed-0 baselines",
    )
    pm = paired(M1M, "statnn", "mmlu_pro")
    ph = paired(M1M, "statnn", "hellaswag")
    claim(
        "experiments",
        "the paired difference is +0.016 [0.012, 0.021] on MMLU-Pro and +0.011 [0.009, 0.014] on HellaSwag",
        [
            pm["mae"]["value"],
            pm["mae"]["ci95"][0],
            pm["mae"]["ci95"][1],
            ph["mae"]["value"],
            ph["mae"]["ci95"][0],
            ph["mae"]["ci95"][1],
        ],
        "d3v2-m1_matched-trajectory seed-0 paired_differences.statnn MAE (cluster bootstrap)",
    )
    above = sum(
        1
        for r in LEARNED["trajectory"]
        for t in ["mmlu_pro", "hellaswag"]
        if paired(r, "statnn", t)["mae"]["ci95"][0] > 0
    )
    claim(
        "experiments",
        "Across the seven learned rows, all 14 intervals lie above zero.",
        [above],
        "paired MAE lower bounds above zero, 7 rows x 2 targets",
    )
    claim(
        "experiments",
        "M5 reaches 0.41 and 0.26, against 0.45 and 0.46 for M1-matched",
        [
            ss("d3v2-m5-trajectory", "mmlu_pro", "spearman", "mean"),
            ss("d3v2-m5-trajectory", "hellaswag", "spearman", "mean"),
            ss(M1M, "mmlu_pro", "spearman", "mean"),
            ss(M1M, "hellaswag", "spearman", "mean"),
        ],
        "d3v2-m5-trajectory and d3v2-m1_matched-trajectory seed_summary",
    )
    claim(
        "experiments",
        "Their test sets hold 98, 98, 59, 60 and 30 rows, drawn from 98, 45, 59, 60 and 27 clusters.",
        [n_rows(f"d3v2-m2-holdout-{h}") for h in HOLDOUTS]
        + [n_clusters(f"d3v2-m2-holdout-{h}") for h in HOLDOUTS],
        "holdout seed-0 targets.n and paired n_clusters",
    )
    declare(
        "experiments",
        "effective rank of at least 2",
        "M6 acceptance gate constant; the gate outcome is asserted below",
    )
    declare(
        "experiments",
        "rank below 2",
        "M6 acceptance gate constant; the gate outcome is asserted below",
    )
    presence(
        "experiments",
        "registered 15\\% MAE gain on adequately powered trajectories",
        HYPOTHESIS,
        r"≥ 15 % relative MAE reduction",
        "H2 15%",
    )
    claim(
        "experiments",
        "registered 15\\% MAE gain on adequately powered trajectories",
        [15],
        "docs/HYPOTHESIS.md H2 test",
    )
    declare(
        "experiments",
        "Experiment 8",
        "blog table reference (Takehi et al.), not in repository data",
    )
    declare(
        "experiments",
        "93\\% is the MATH-500 share at a 32,768-token limit",
        "Takehi et al. blog figure, cited",
    )
    claim("experiments", "and ours is at 4,096 tokens", [4096], "base-final-4096 generation")

    # ---------------- results
    claim(
        "results",
        "On seed 0 of the trajectory split (345 rows, 278 clusters), the gap is 0.042 Spearman on MMLU-Pro and 0.035 on HellaSwag",
        [
            n_rows(M1T),
            n_clusters(M1T),
            bl(M1T, "statnn", "mmlu_pro") - bl(M1T, "descriptor_only", "mmlu_pro"),
            bl(M1T, "statnn", "hellaswag") - bl(M1T, "descriptor_only", "hellaswag"),
        ],
        "trajectory seed-0 baselines, unrounded difference",
    )
    claim(
        "results",
        "0.560 against 0.555 on MMLU-Pro and 0.630 against 0.588 on HellaSwag",
        [
            ev_perf("mmlu_pro", "descriptor_only", "r2"),
            ev_perf("mmlu_pro", "statnn", "r2"),
            ev_perf("hellaswag", "descriptor_only", "r2"),
            ev_perf("hellaswag", "statnn", "r2"),
        ],
        "evidence CSV performance r2 (pooled out-of-fold)",
    )
    claim(
        "results",
        "is 0.70, against 0.49 for the descriptor model, a lead of 0.20",
        [
            ev_perf("kl_to_base", "statnn", "spearman"),
            ev_perf("kl_to_base", "descriptor_only", "spearman"),
            ev_perf("kl_to_base", "statnn", "spearman")
            - ev_perf("kl_to_base", "descriptor_only", "spearman"),
        ],
        "evidence CSV performance spearman, kl_to_base",
    )
    claim(
        "results",
        "4.1\\% [$-$6.3\\%, 14.0\\%] on MMLU-Pro and 5.5\\% [$-$6.3\\%, 16.4\\%] on HellaSwag",
        [4.1, -6.3, 14.0, 5.5, -6.3, 16.4],
        "round2_metrology_output.txt, paired MAE theta and boot_ci (percent)",
    )
    locator(
        "results",
        "round2_metrology_output.txt}, lines 100 and 102",
        "round2_metrology_output.txt lines 100, 102",
    )
    claim("results", "on the 345 seed-0 trajectory rows", [n_rows(M1T)], "trajectory seed-0 n")
    claim(
        "results",
        "the top five features hold 26\\% of total gain",
        [ev_top5_share("mmlu_pro") * 100],
        "evidence CSV importance ranks 1 to 5, mmlu_pro",
    )
    claim(
        "results",
        "Within-family Spearman runs from 0.13 to 0.71 on MMLU-Pro and from $-$0.12 to 0.71 on HellaSwag",
        [
            min(finite([ev_within("mmlu_pro", f) for f in WITHIN_FAMILIES])),
            max(finite([ev_within("mmlu_pro", f) for f in WITHIN_FAMILIES])),
            min(finite([ev_within("hellaswag", f) for f in WITHIN_FAMILIES])),
            max(finite([ev_within("hellaswag", f) for f in WITHIN_FAMILIES])),
        ],
        "evidence CSV within_family spearman ranges",
    )
    claim(
        "results",
        "LoRA SFT (59 rows) on MMLU-Pro and task vector (30 rows)",
        [ev_within_n("mmlu_pro", "lora_sft"), ev_within_n("mmlu_pro", "task_vector")],
        "evidence CSV within_family n",
    )
    claim(
        "results",
        "The 0.71 values come from the two smallest families",
        [ev_within("mmlu_pro", "lora_sft")],
        "evidence CSV within_family: LoRA SFT, MMLU-Pro (0.709)",
    )
    claim(
        "results",
        "unchanged in all 60 rows",
        [ev_family("hellaswag", "random_dir")["n"]],
        "evidence CSV family_distribution n, random_dir",
    )
    claim(
        "results",
        "their MMLU-Pro SD is 0.007",
        [ev_family("mmlu_pro", "random_dir")["sd"]],
        "evidence CSV family_distribution sd, random_dir",
    )
    claim(
        "results",
        "SDs of 0.013 and 0.005",
        [ev_family("mmlu_pro", "abliteration")["sd"], ev_family("hellaswag", "abliteration")["sd"]],
        "evidence CSV family_distribution sd, abliteration",
    )
    claim(
        "results",
        "$-$0.063 on MMLU-Pro and $+$0.034 on HellaSwag",
        [ev_family("mmlu_pro", "lora_sft")["mean"], ev_family("hellaswag", "lora_sft")["mean"]],
        "evidence CSV family_distribution mean, lora_sft",
    )
    declare("results", "ddof 1", "sample SD convention, not a result")

    # ---------------- discussion
    claim("discussion", "Across the 25 complete rows", [25], "complete d3v2 run directories")
    presence(
        "discussion",
        "H1 required a per-benchmark Spearman of at least 0.6",
        HYPOTHESIS,
        r"Spearman ρ ≥ 0\.6",
        "H1 0.6",
    )
    presence(
        "discussion",
        "H1 also has a falsification rule: Spearman below 0.3",
        HYPOTHESIS,
        r"ρ < 0\.3",
        "H1 0.3",
    )
    claim(
        "discussion",
        "steps of one item out of 100",
        [b_eval["counts"]["mmlu_pro"]],
        "base eval counts (100 items)",
    )
    assertion(
        "discussion",
        "no learned row reaches 0.6 on either target (three-seed means)",
        all(
            ss(r, t, "spearman", "mean") < 0.6
            for r in SUITE_RUNS
            if "statnn" not in r
            for t in ["mmlu_pro", "hellaswag"]
            if not math.isnan(ss(r, t, "spearman", "mean"))
        ),
        "suite seed_summary means",
    )
    claim(
        "discussion",
        "the highest three-seed mean on HellaSwag is 0.46",
        [round(max(finite([ss(r, "hellaswag", "spearman", "mean") for r in SUITE_RUNS])), 2)],
        "max over the 25 complete d3v2 suite rows",
    )
    claim(
        "discussion",
        "The highest on MMLU-Pro is 0.45",
        [round(max(finite([ss(r, "mmlu_pro", "spearman", "mean") for r in SUITE_RUNS])), 2)],
        "max over the 25 complete d3v2 suite rows",
    )
    claim(
        "discussion",
        "their MMLU-Pro SD is 0.007",
        [ev_family("mmlu_pro", "random_dir")["sd"]],
        "evidence CSV family_distribution sd, random_dir",
    )
    claim(
        "discussion",
        "Abliteration's SDs are 0.013 and 0.005",
        [ev_family("mmlu_pro", "abliteration")["sd"], ev_family("hellaswag", "abliteration")["sd"]],
        "evidence CSV family_distribution sd, abliteration",
    )
    claim(
        "discussion", "one item out of 100", [base_summary()["counts"]["mmlu_pro"] / 3], ""
    ) if False else None
    claim(
        "discussion",
        "Each target is a 100-item accuracy, with a binomial standard error of 0.045 for MMLU-Pro and 0.049 for HellaSwag",
        [100, binom_se(base_val("mmlu_pro"), 100), binom_se(base_val("hellaswag"), 100)],
        "binomial SE at base fast-tier values (0.28 and 0.39, n = 100)",
    )
    claim(
        "discussion",
        "the residual SD within family and device is 0.0131, no larger than its item SE of 0.0136",
        [met("hellaswag", WITHIN, "sd_dy"), met("hellaswag", WITHIN, "sd_itemSE")],
        "metrology_budget_output.txt line 44 (hellaswag within family x device: sd_dy, sd_itemSE)",
    )
    claim(
        "discussion",
        "the residual SD is 0.0384, above its item SE of 0.0246, and the remaining signal SD is 0.0295",
        [
            met("mmlu_pro", WITHIN, "sd_dy"),
            met("mmlu_pro", WITHIN, "sd_itemSE"),
            met("mmlu_pro", WITHIN, "sd_signal_pop"),
        ],
        "metrology_budget_output.txt line 42 (mmlu_pro within family x device)",
    )
    locator(
        "discussion",
        "metrology\\_budget\\_output.txt}, lines 42 and 44",
        "metrology lines 42 and 44 (caption)",
    )
    claim(
        "discussion",
        "use the 300 D3 entries",
        [300],
        "metrology_budget_output.txt: D3 subset n = 300",
    )
    claim("discussion", "interval on the 345 matched v2 rows", [n_rows(M1T)], "v2 matched rows (n)")
    claim(
        "discussion",
        "mean absolute difference on the 11 edited entries evaluated on the H100 was 0.008 on MMLU-Pro and 0.003 on HellaSwag, with Spearman 0.75 and 0.76",
        [
            11,
            DEVS["H1"]["MMLU-Pro"]["mean"],
            DEVS["H1"]["HellaSwag"]["mean"],
            DEVS["H1"]["MMLU-Pro"]["rho"],
            DEVS["H1"]["HellaSwag"]["rho"],
        ],
        "per-entry eval.json, local vs H100, 11 entries",
    )
    claim(
        "discussion",
        "On the A10, four edited entries gave mean differences of 0.010 and 0.003",
        [DEVS["H2"]["MMLU-Pro"]["mean"], DEVS["H2"]["HellaSwag"]["mean"]],
        "per-entry eval.json, local vs A10, 4 entries",
    )
    claim(
        "discussion",
        "gives a seed-to-seed SD of 0.0000 for MMLU-Pro and HellaSwag",
        [metrology_seed_sd("mmlu_pro")],
        "metrology_budget_output.txt lines 24 and 25 (sd_seed, likelihood-scored)",
    )
    claim(
        "discussion",
        "between 0.0124 and 0.0504 for the five sampled targets",
        [
            min(
                metrology_seed_sd(t)
                for t in ["gsm8k", "math500", "ifeval", "refusal_rate", "over_refusal"]
            ),
            max(
                metrology_seed_sd(t)
                for t in ["gsm8k", "math500", "ifeval", "refusal_rate", "over_refusal"]
            ),
        ],
        "metrology_budget_output.txt lines 19 to 23 (sd_seed, sampled targets: min and max)",
    )
    locator(
        "discussion",
        "metrology_budget_output.txt}, lines 19 to 27",
        "metrology lines 19 to 27 (seed panel block)",
    )
    assertion(
        "discussion",
        "HellaSwag seed-to-seed SD is 0.0000 (metrology line 25)",
        metrology_seed_sd("hellaswag") == 0.0,
        "metrology_budget_output.txt line 25",
    )
    claim(
        "discussion",
        "delta-norm baseline reaches 0.21 and 0.18 on the trajectory split",
        [bl(M1T, "delta_norm", "mmlu_pro"), bl(M1T, "delta_norm", "hellaswag")],
        "trajectory seed-0 baselines",
    )
    claim(
        "discussion",
        "Under family holdout its Spearman falls to $-$0.13 and $-$0.29",
        [
            bl("d3v2-m1-family", "delta_norm", "mmlu_pro"),
            bl("d3v2-m1-family", "delta_norm", "hellaswag"),
        ],
        "family seed-0 baselines",
    )
    claim(
        "discussion",
        "Within-family refits on HellaSwag range from $-$0.12 to 0.71",
        [
            min(finite([ev_within("hellaswag", f) for f in WITHIN_FAMILIES])),
            max(finite([ev_within("hellaswag", f) for f in WITHIN_FAMILIES])),
        ],
        "evidence CSV within_family hellaswag",
    )
    claim(
        "discussion",
        "The 260 matched D3 rows with inferred runtime metadata",
        [V2_INFERRED],
        "targets_with_inferred_runtime (v2 trajectory seed-1 manifest)",
    )
    claim(
        "discussion",
        "at $-$0.02 on MMLU-Pro and 0.11 on HellaSwag",
        [bl("d3v2-m1-family", "statnn", "mmlu_pro"), bl("d3v2-m1-family", "statnn", "hellaswag")],
        "family seed-0 StatNN baselines",
    )
    claim(
        "discussion",
        "with a three-seed mean of 0.31",
        [ss("d3v2-m1_matched-family", "hellaswag", "spearman", "mean")],
        "d3v2-m1_matched-family seed_summary (HellaSwag spearman mean)",
    )
    claim(
        "discussion",
        "the Spearman gap is 0.042 on MMLU-Pro and 0.035 on HellaSwag",
        [
            bl(M1T, "statnn", "mmlu_pro") - bl(M1T, "descriptor_only", "mmlu_pro"),
            bl(M1T, "statnn", "hellaswag") - bl(M1T, "descriptor_only", "hellaswag"),
        ],
        "trajectory seed-0 baselines",
    )
    claim(
        "discussion",
        "$\\theta = 0.146$ [0.069, 0.214] for MMLU-Pro",
        [RESID["theta"], RESID["lo"], RESID["hi"]],
        "round2_residual_estimand_output.txt, mmlu_pro block",
    )
    claim(
        "discussion",
        "$\\theta = -0.034$ [$-$0.118, 0.037] for HellaSwag",
        [RESID_HS["theta"], RESID_HS["lo"], RESID_HS["hi"]],
        "round2_residual_estimand_output.txt, hellaswag block",
    )
    claim(
        "discussion",
        "over 345 rows and 278 clusters",
        [RESID["n"], RESID["clusters"]],
        "round2_residual_estimand_output.txt, mmlu_pro block n and clusters",
    )
    locator(
        "discussion",
        "round2_residual_estimand_output.txt}, lines 2 to 11",
        "residual theta line within 2 to 11",
    )
    locator("discussion", "and 40 to 47", "residual output lines 40 to 47 (hellaswag block)")
    claim("discussion", "$\\theta = 1 - A/B$", [1], "definition of the estimand") if False else None
    declare("discussion", "$\\theta = 1 - A/B$", "definition of the estimand, not a result")
    declare(
        "discussion",
        "No learned row reaches 0.6 on either target.",
        "threshold; covered by the assertion below",
    )
    declare("discussion", "Several rows exceed 0.3", "threshold; covered by the assertion below")
    locator(
        "discussion",
        "round2-metrology.md}, section 1.4",
        "round2 metrology section 1.4 (HellaSwag accuracy retired; negative signal variance)",
    )
    claim(
        "discussion",
        "0.0384 & 0.0246 & 0.0295 & 0.146 [0.069, 0.214]",
        [
            met("mmlu_pro", WITHIN, "sd_dy"),
            met("mmlu_pro", WITHIN, "sd_itemSE"),
            met("mmlu_pro", WITHIN, "sd_signal_pop"),
            RESID["theta"],
            RESID["lo"],
            RESID["hi"],
        ],
        "tab:residual MMLU-Pro row (metrology line 42; residual output)",
    )
    claim(
        "discussion",
        "0.0131 & 0.0136 & 0.0000 & $-$0.034 [$-$0.118, 0.037]",
        [
            met("hellaswag", WITHIN, "sd_dy"),
            met("hellaswag", WITHIN, "sd_itemSE"),
            met("hellaswag", WITHIN, "sd_signal_pop"),
            RESID_HS["theta"],
            RESID_HS["lo"],
            RESID_HS["hi"],
        ],
        "tab:residual HellaSwag row (metrology line 44; residual output)",
    )
    # ---------------- reliability
    claim(
        "reliability",
        "re-evaluated the base model and the 10 control entries on the local GPU",
        [len(CONTROLS)],
        "runs/d3_shards/controls.txt",
    )
    claim(
        "reliability",
        "The cross-host panel covers the 10 control entries",
        [len(CONTROLS)],
        "controls.txt",
    )
    claim("reliability", "the 10 controls and", [len(CONTROLS)], "controls.txt")
    claim("reliability", "is taken across the 11 H100 entries", [len(H100_IDS)], "H100 entries")
    claim(
        "reliability",
        "All records used seed 0, batch size 32",
        [32],
        "per-entry eval.json batch_size (seed 0 is a seed label)",
    )
    claim(
        "reliability",
        "The seed-1 panel reruns the 10 controls and the base",
        [len(CONTROLS)],
        "controls.txt",
    )
    _g = seed_panel("gsm8k")
    _m = seed_panel("math500")
    _i = seed_panel("ifeval")
    _o = seed_panel("over_refusal")
    _k = seed_panel("kl_to_base")
    claim(
        "reliability",
        "Over the 11 records, seed 0 against seed 1 gives a mean and maximum absolute difference of 0.055 and 0.12 with Spearman $\\rho$ 0.15 on GSM8K",
        [11, _g[0], _g[1], _g[2]],
        "seed-1 panel recomputed from eval.json (11 records)",
    )
    claim(
        "reliability",
        "0.036 and 0.12 with $\\rho$ 0.46 on MATH-500",
        [_m[0], _m[1], _m[2]],
        "seed-1 panel recomputed",
    )
    claim(
        "reliability",
        "0.029 and 0.04 with $\\rho$ 0.10 on IFEval",
        [_i[0], _i[1], _i[2]],
        "seed-1 panel recomputed",
    )
    claim(
        "reliability",
        "0.049 and 0.10 with $\\rho$ 0.67 on keyword over-refusal",
        [_o[0], _o[1], _o[2]],
        "seed-1 panel recomputed",
    )
    claim(
        "reliability",
        "KL differs by at most 0.0003 nats",
        [_k[1]],
        "seed-1 panel recomputed (KL max abs difference)",
    )
    claim(
        "reliability",
        "MMLU-Pro and HellaSwag agree exactly on all 11 records",
        [11],
        "seed-1 panel: 11 records (10 controls and base)",
    )
    claim(
        "reliability",
        "IFEval and the pooled share hitting the 1,024-token cap are descriptive",
        [1024],
        "fast-tier token cap (per-entry eval.json max_new_tokens)",
    )
    claim(
        "reliability",
        "Over the 10 controls, the seed noise in the paired change has SD 0.0504 on GSM8K, 0.0268 on MATH-500, 0.0248 on IFEval, 0.0356 on keyword over-refusal and 0.0124 on keyword refusal, and 0.0000 on MMLU-Pro and HellaSwag",
        [
            len(CONTROLS),
            metrology_seed_sd("gsm8k"),
            metrology_seed_sd("math500"),
            metrology_seed_sd("ifeval"),
            metrology_seed_sd("over_refusal"),
            metrology_seed_sd("refusal_rate"),
            0.0,
        ],
        "metrology_budget_output.txt lines 19 to 27 (sd_seed; 10 controls)",
    )
    locator(
        "reliability",
        "metrology_budget_output.txt}, lines 19 to 27",
        "metrology lines 19 to 27 (seed panel)",
    )
    declare("reliability", "4 November protocol", "date")
    claim(
        "reliability",
        "with seed 0, the fast-tier configuration and batch size 32",
        [32],
        "per-entry eval.json batch_size",
    )
    claim(
        "reliability",
        "All 11 pairs matched exactly on every reported metric",
        [len(CONTROLS) + 1],
        "10 controls and the base",
    )
    retest = []
    for e in CONTROLS + ["base"]:
        a = jeval(EV / e / "eval.json")
        b = jeval(EV / e / "eval.local-run1.json")
        retest += [abs(float(a[k]) - float(b[k])) for k in DEV_KEYS.values()]
    claim(
        "reliability",
        "The mean and maximum absolute differences were 0.000 on each.",
        [max(retest)],
        "retest pairs: eval.json vs eval.local-run1.json, 11 pairs, all metrics",
    )
    claim(
        "reliability",
        "On the H100 PCIe (device H1), 11 edited entries",
        [len(H100_IDS)],
        "H100 entries (10 controls and 013b7b61202fe188)",
    )
    claim(
        "reliability",
        "by 0.10 on GSM8K (0.16 and 0.06; 50 items)",
        [
            abs(base_dev("H1", "gsm8k") - base_val("gsm8k")),
            base_val("gsm8k"),
            base_dev("H1", "gsm8k"),
            50,
        ],
        "base local vs devices/H1 eval.json",
    )
    claim(
        "reliability",
        "by 0.08 on MATH-500 (0.16 and 0.08; 25 items)",
        [
            abs(base_dev("H1", "math500") - base_val("math500")),
            base_val("math500"),
            base_dev("H1", "math500"),
            25,
        ],
        "base local vs devices/H1",
    )
    claim(
        "reliability",
        "by 0.10 on keyword over-refusal (0.10 and 0.20; 50 prompts)",
        [
            abs(base_dev("H1", "over_refusal") - base_val("over_refusal")),
            base_val("over_refusal"),
            base_dev("H1", "over_refusal"),
            50,
        ],
        "base local vs devices/H1",
    )
    claim(
        "reliability",
        "The other proportions agreed within 0.01, and WikiText perplexity was 22.90 locally and 22.92 on the H100",
        [0.01, base_val("wikitext_ppl"), base_dev("H1", "wikitext_ppl")],
        "tolerance 0.01 asserted below; base wikitext_ppl local and devices/H1",
    )
    assertion(
        "reliability",
        "other base proportions on the H100 agree within 0.01",
        all(
            abs(base_dev("H1", k) - base_val(k)) <= 0.01 + 1e-9
            for k in ["ifeval", "refusal_rate", "mmlu_pro", "hellaswag"]
        ),
        "base local vs devices/H1: ifeval, refusal, mmlu_pro, hellaswag",
    )
    claim(
        "reliability",
        "0.10 on GSM8K (0.16 and 0.06), 0.12 on MATH-500 (0.16 and 0.28) and 0.04 on IFEval (0.40 and 0.44)",
        [
            abs(base_dev("H2", "gsm8k") - base_val("gsm8k")),
            base_val("gsm8k"),
            base_dev("H2", "gsm8k"),
            abs(base_dev("H2", "math500") - base_val("math500")),
            base_val("math500"),
            base_dev("H2", "math500"),
            abs(base_dev("H2", "ifeval") - base_val("ifeval")),
            base_val("ifeval"),
            base_dev("H2", "ifeval"),
        ],
        "base local vs devices/H2 (A10)",
    )
    claim(
        "reliability",
        "On the base model, MMLU-Pro scored 0.28 locally, 0.29 on the H100 and 0.28 on the A10, and HellaSwag scored 0.39 on all three hosts.",
        [
            base_val("mmlu_pro"),
            base_dev("H1", "mmlu_pro"),
            base_dev("H2", "mmlu_pro"),
            base_val("hellaswag"),
        ],
        "base local, H100 and A10 eval.json (MMLU-Pro and HellaSwag)",
    )
    assertion(
        "reliability",
        "HellaSwag identical on all three hosts; MMLU-Pro differs by one item on the H100",
        all(base_dev(d, "hellaswag") == base_val("hellaswag") for d in ["H1", "H2"])
        and round(abs(base_dev("H1", "mmlu_pro") - base_val("mmlu_pro")) * 100) == 1,
        "base devices",
    )
    claim(
        "reliability",
        "Both are 100-item scores, so the H100 MMLU-Pro gap of 0.01 is one item.",
        [100, abs(base_dev("H1", "mmlu_pro") - base_val("mmlu_pro"))],
        "base item counts; H100 MMLU-Pro gap",
    )
    assertion(
        "reliability",
        "excluding 013b7b61202fe188 leaves every maximum absolute difference unchanged",
        all(
            cross_host(H100_IDS, h100_eval, k, drop="013b7b61202fe188")["max"]
            == cross_host(H100_IDS, h100_eval, k)["max"]
            for k in DEV_KEYS.values()
        ),
        "per-entry eval.json, H100",
    )
    shift = max(
        abs(
            DEVS["H1"][lab]["rho"]
            - cross_host(H100_IDS, h100_eval, key, drop="013b7b61202fe188")["rho"]
        )
        for lab, key in DEV_KEYS.items()
        if not math.isnan(DEVS["H1"][lab]["rho"])
    )
    cap_excl = cross_host(H100_IDS, h100_eval, "loop_rate", drop="013b7b61202fe188")["rho"]
    claim(
        "reliability",
        "moves the Spearman values by up to 0.16; the cap-hit share, for example, rises from 0.72 to 0.89",
        [round(shift, 2), DEVS["H1"]["Cap-hit"]["rho"], cap_excl],
        "H100 Spearman, 11 entries vs 10 without 013b7b61202fe188",
    )
    claim(
        "reliability",
        "Of 300 evaluated D3 edits, 260 had matched-reference delta targets, and all 260 retained labels",
        [300, 260, 260],
        "board-a-d3-audit audit.txt counters and test_n",
    )
    claim(
        "reliability",
        "The other 40 had no predecessor target",
        [40],
        "audit counter missing_predecessor_targets",
    )
    claim(
        "reliability",
        "0.052 for GSM8K ($n = 50$), 0.073 for MATH-500 ($n = 25$), 0.098 for IFEval ($n = 25$), 0.045 for MMLU-Pro ($n = 100$) and 0.049 for HellaSwag ($n = 100$)",
        [
            binom_se(0.16, 50),
            50,
            binom_se(0.16, 25),
            25,
            binom_se(0.40, 25),
            25,
            binom_se(0.28, 100),
            100,
            binom_se(0.39, 100),
            100,
        ],
        "binomial SE at base fast-tier values",
    )
    claim(
        "reliability",
        "0.16 for GSM8K, 0.16 for MATH-500, 0.40 for IFEval, 0.28 for MMLU-Pro and 0.39 for HellaSwag",
        [
            base_val("gsm8k"),
            base_val("math500"),
            base_val("ifeval"),
            base_val("mmlu_pro"),
            base_val("hellaswag"),
        ],
        "base local eval.json",
    )
    claim(
        "reliability",
        "the H1 threshold $\\rho \\ge 0.6$ would need $r \\ge 0.36$",
        [0.6, 0.36],
        "docs/HYPOTHESIS.md (0.6); 0.6 squared = 0.36",
    )
    claim(
        "reliability",
        "a Spearman $\\rho$ of at least 0.8 with a lower 95\\% bound of at least 0.6, and an ICC(A,1) lower bound of at least 0.75",
        [0.8, 0.6, 0.75],
        "docs/reviews/H5/board-a-prereg-2026-11-04.md (thresholds, as written in the file)",
    )
    presence(
        "reliability",
        "a Spearman $\\rho$ of at least 0.8",
        PREREG_H5,
        r"Spearman point ≥ \.8",
        "rho point 0.8",
    )
    presence(
        "reliability",
        "lower 95\\% bound of at least 0.6",
        PREREG_H5,
        r"lower 95% bound ≥ \.6",
        "lower bound 0.6",
    )
    presence(
        "reliability",
        "ICC(A,1) lower bound of at least 0.75",
        PREREG_H5,
        r"ICC lower bound ≥ \.75",
        "ICC 0.75",
    )
    claim(
        "reliability",
        "smallest scientifically relevant change is $\\delta = 0.10$",
        [0.10],
        "docs/reviews/H5/board-a-prereg-2026-11-04.md",
    )
    presence(
        "reliability",
        "smallest scientifically relevant change is $\\delta = 0.10$",
        PREREG_H5,
        r"δ_b = \.10 \(10 percentage",
        "delta 0.10",
    )
    claim(
        "reliability",
        "For log-perplexity it is $\\log 1.05$, and for KL it is 0.01 nats.",
        [1.05, 0.01],
        "docs/reviews/H5/board-a-prereg-2026-11-04.md",
    )
    presence(
        "reliability",
        "For log-perplexity it is $\\log 1.05$",
        PREREG_H5,
        r"δ_b = log\(1\.05\)",
        "log 1.05",
    )
    presence(
        "reliability", "and for KL it is 0.01 nats", PREREG_H5, r"δ_b = \.01 nats", "KL 0.01 nats"
    )
    claim("reliability", "At 50 prompts one item is 0.02", [50, 1 / 50], "1/50 and the 50 prompts")
    claim(
        "reliability",
        "edit KL differed by 0.0009 nats at most on the H100 and 0.0037 on the A10",
        [DEVS["H1"]["KL"]["max"], DEVS["H2"]["KL"]["max"]],
        "per-entry eval.json KL max",
    )
    claim("reliability", "$n = 11$ on the H100, $n = 4$ on the A10", [11, 4], "entry counts")
    claim(
        "reliability",
        "pools GSM8K, MATH-500 and IFEval responses at the 1,024-token cap, 100 responses per entry",
        [1024, 100],
        "per-entry eval.json max_new_tokens; 50 + 25 + 25 responses",
    )
    claim(
        "reliability",
        "at temperature 0.6",
        [base_summary()["generation"]["temperature"]],
        "base-final-4096 generation.temperature",
    )
    claim(
        "reliability",
        "the pooled share hitting the 1,024-token cap",
        [1024],
        "per-entry eval.json max_new_tokens",
    )
    claim(
        "reliability",
        "Keyword refusal sits at 0.99 to 1.00 on the base model",
        [base_summary()["refusal_rate"], base_val("refusal_rate")],
        "cheap-tier base refusal 0.99; fast-tier local refusal 1.0",
    )
    claim(
        "reliability",
        "Keyword refusal change when the base is at 0.99 to 1.00",
        [base_summary()["refusal_rate"], base_val("refusal_rate")],
        "cheap-tier base refusal 0.99; fast-tier local refusal 1.0",
    )
    claim(
        "reliability",
        "Seed 0, batch size 32 and one configuration hash throughout.",
        [32],
        "per-entry eval.json seed and batch_size",
    )
    claim("reliability", "H100 PCIe ($n = 11$)", [11], "H100 entry count (table header)")
    claim("reliability", "A10 ($n = 4$)", [4], "A10 entry count (table header)")
    claim(
        "reliability",
        "Binomial SE 0.05 to 0.10; base host gaps up to 0.12",
        [
            min(se_fast),
            max(se_fast),
            max(
                abs(base_dev(d, k) - base_val(k))
                for d in ["H1", "H2"]
                for k in [
                    "gsm8k",
                    "math500",
                    "ifeval",
                    "over_refusal",
                    "refusal_rate",
                    "mmlu_pro",
                    "hellaswag",
                ]
            ),
        ],
        "binomial SE at base fast-tier values; largest base host gap",
    )
    claim(
        "reliability",
        "Eight windows; base perplexity 22.90 locally and 22.92 on the H100",
        [base_val("wikitext_ppl"), base_dev("H1", "wikitext_ppl")],
        "base eval wikitext_ppl (8 windows asserted below)",
    )
    assertion(
        "reliability",
        "WikiText fast tier uses eight windows",
        b_eval["counts"]["wikitext_ppl"] == 8,
        "base eval counts",
    )
    claim(
        "reliability",
        "Single-entry SE 0.045 to 0.049 at 100 items",
        [binom_se(0.28, 100), binom_se(0.39, 100), 100],
        "binomial SE at base fast-tier values",
    )
    claim("reliability", "(1,024 here", [1024], "per-entry eval max_new_tokens")
    declare("reliability", "32,768 in the blog", "Takehi et al. token limit, cited")
    declare("reliability", "$\\delta/2$", "definition of the reliability bound")
    declare("reliability", "$\\sqrt{p(1-p)/n}$", "formula, not a result")
    declare(
        "reliability",
        "(identifiers in \\path{data/runs/d3\\_shards/controls.txt})",
        "identifier list",
    )
    declare("reliability", "95th percentile", "definition")

    # ---------------- assertions without a numeric token of their own
    fam_above = {
        r
        for r in LEARNED["family"]
        if ss(r, "mmlu_pro", "spearman", "mean") > bl("d3v2-m1-family", "statnn", "mmlu_pro")
    }
    assertion(
        "results",
        "on the family split only M1-matched and M6 exceed seed-0 StatNN on MMLU-Pro",
        fam_above == {"d3v2-m1_matched-family", "d3v2-m6-family"},
        "family seed means vs seed-0 StatNN",
    )
    assertion(
        "discussion",
        "every learned row's three-seed mean is below seed-0 StatNN on both targets (seed and trajectory splits)",
        all(
            ss(r, t, "spearman", "mean") < bl(ref, "statnn", t)
            for ref, runs in [("d3v2-m1-seed", LEARNED["seed"]), (M1T, LEARNED["trajectory"])]
            for r in runs
            for t in ["mmlu_pro", "hellaswag"]
        ),
        "seed-0 StatNN (seed and trajectory splits) vs three-seed means",
    )
    assertion(
        "discussion",
        "the highest HellaSwag three-seed mean belongs to M1-matched on the trajectory split",
        max(SUITE_RUNS, key=lambda r: ss(r, "hellaswag", "spearman", "mean")) == M1M,
        "suite seed_summary",
    )
    assertion(
        "discussion",
        "rows at 0.45 on MMLU-Pro (rounded) are M1-matched trajectory and M2-matched seed",
        {r for r in SUITE_RUNS if round(ss(r, "mmlu_pro", "spearman", "mean"), 2) == 0.45}
        == {M1M, "d3v2-m2_matched-seed"},
        "suite seed_summary",
    )
    assertion(
        "discussion",
        "at least three suite rows exceed 0.3 on some target",
        sum(
            1
            for r in SUITE_RUNS
            if max(
                finite(
                    [ss(r, "mmlu_pro", "spearman", "mean"), ss(r, "hellaswag", "spearman", "mean")]
                )
            )
            > 0.3
        )
        >= 3,
        "suite seed_summary",
    )
    assertion(
        "results",
        "every learned row has a larger seed-0 MAE than StatNN on both targets (trajectory)",
        all(
            own_mae(r, t) > seed0(M1T)["statnn"]["targets"][t]["mae"]["value"]
            for r in LEARNED["trajectory"]
            for t in ["mmlu_pro", "hellaswag"]
        ),
        "trajectory seed-0 targets.mae",
    )
    for split, runs in LEARNED.items():
        vals = {json.dumps(seed0(r)["baselines"], sort_keys=True) for r in runs}
        assertion(
            "experiments",
            f"seed-0 baselines are identical across models in the {split} split",
            len(vals) == 1,
            f"{len(runs)} runs",
        )
    assertion(
        "experiments",
        "M6 fails its collapse gate on all three standard splits (seed 0)",
        all(
            seed0(f"d3v2-m6-{sp}")["m6_collapse_gate"] is False
            for sp in ["family", "seed", "trajectory"]
        ),
        "m6_collapse_gate in seed-0 metrics",
    )
    assertion(
        "experiments",
        "twenty-five d3v2 run directories have metrics.json",
        len(SUITE_RUNS) == 25,
        "d3v2 run directories",
    )
    assertion(
        "experiments",
        "the suite scores eight targets",
        len(metrics(M1T)["seed_summary"]) == 8,
        "seed_summary keys",
    )
    assertion(
        "experiments",
        "the PREREG status line is reproduced exactly",
        "Status: frozen prospective design, pending Board A acceptance and measured execution gates."
        in read(PREREG),
        "docs/PREREG_H1_TF.md line 3",
    )
    assertion(
        "zoo",
        "D3 quotas 60 + 100 + 60 + 30 + 50 sum to 300",
        60 + 100 + 60 + 30 + 50 == 300,
        "quota arithmetic",
    )
    assertion(
        "zoo",
        "audit.txt reports 300 D3 test entries and 260 matched targets",
        "RUN d3-m1-family-level rows 2400 test_n 300" in read(AUDIT_TXT)
        and "'matched_reference_targets': 260" in read(AUDIT_TXT),
        "board-a-d3-audit audit.txt",
    )
    assertion(
        "experiments",
        "lambda spend log: A100 at 1.99 per hour",
        "gpu_1x_a100_sxm4" in read(LAMBDA) and "| 1.99 |" in read(LAMBDA),
        "board/lambda_spend.md",
    )
    assertion(
        "experiments",
        "RESULTS.md blog values 13.8, 7.9 and 21.0 match the table column",
        re.search(r"\| gsm8k \| 15\.5 \| 13\.8 \|", read(RESULTS_MD)) is not None
        and re.search(r"\| math500 \| 6\.0 \| 7\.9 \|", read(RESULTS_MD)) is not None
        and re.search(r"\| ifeval \| 42\.0 \| 21\.0 \|", read(RESULTS_MD)) is not None,
        "docs/RESULTS.md",
    )
    assertion(
        "experiments",
        "base retest record id 20261104-202947-1dcb69a2",
        jeval(EV / "base" / "eval.json")["run_id"] == "20261104-202947-1dcb69a2",
        "base eval.json run_id",
    )
    m = read(METROLOGY).splitlines()
    assertion(
        "discussion",
        "metrology lines 43 and 44 are the HellaSwag within-family row",
        "hellaswag" in m[42] and "within family x device" in m[43],
        "metrology_budget_output.txt",
    )
    assertion(
        "discussion",
        "metrology lines 24 and 25 are the mmlu_pro and hellaswag sd_seed rows",
        "mmlu_pro" in m[23] and "hellaswag" in m[24],
        "metrology_budget_output.txt",
    )
    rm = read(RESIDUAL_METRO).splitlines()
    assertion(
        "results",
        "round2_metrology lines 100 and 102 are the MMLU-Pro and HellaSwag paired rows",
        "mmlu_pro" in rm[99] and "hellaswag" in rm[101],
        "round2_metrology_output.txt",
    )
    assertion(
        "discussion",
        "residual theta sits inside the cited lines 2 to 11",
        any(
            2 <= i + 1 <= 11
            for i, line in enumerate(read(RESIDUAL).splitlines())
            if '"theta"' in line
        ),
        "round2_residual_estimand_output.txt",
    )


def base_gap_max() -> float:
    return max(
        abs(base_dev(d, k) - base_val(k))
        for d in ["H1", "H2"]
        for k in [
            "gsm8k",
            "math500",
            "ifeval",
            "over_refusal",
            "refusal_rate",
            "mmlu_pro",
            "hellaswag",
        ]
    )


# ------------------------------------------------------------------ tables


def table_envs(raw: str):
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", raw, re.DOTALL):
        lab = re.search(r"\\label\{(tab:[^}]+)\}", m.group(0))
        tab = re.search(
            r"\\begin\{tabular\*?\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\*?\}",
            m.group(0),
            re.DOTALL,
        )
        if lab and tab:
            yield lab.group(1), m.start() + tab.start(1), m.start() + tab.end(1), m.start(), m.end()


def cells(row: str) -> list[str]:
    return [c.strip() for c in row.split("&")]


def rows_of(body: str) -> list[str]:
    body = re.sub(r"\\(?:toprule|midrule|bottomrule)", " ", body)
    return [r.strip() for r in re.split(r"\\\\", body) if r.strip()]


def cell_check(file: str, label: str, shown: str, expected: float, src: str) -> None:
    add("CHECKED", file, label, shown, expected, same(shown, expected), src)


def row_tokens(row: str) -> int:
    """Non-seed numeric tokens in a processed table row; the row's checker verifies each cell."""
    return sum(1 for t in numeric_tokens(mask_nonprose(row)) if not t["seed"])


def t_suite(body: str, file: str) -> int:
    n = 0
    for row in rows_of(body):
        if "texttt" not in row:
            continue
        c = cells(row)
        n += row_tokens(row)
        run = re.search(r"\\texttt\{([^}]+)\}", row).group(1).replace("\\_", "_")
        rc = re.findall(r"\d+", c[1])
        cell_check(file, f"suite {run} rows", rc[0], n_rows(run), "targets.n")
        cell_check(file, f"suite {run} clusters", rc[1], n_clusters(run), "paired n_clusters")
        for col, target in [(2, "mmlu_pro"), (3, "hellaswag")]:
            cell = c[col]
            if "undefined" in cell:
                assertion(
                    file,
                    f"{run} {target} undefined (no finite seed)",
                    metrics(run)["seed_summary"][target]["spearman"]["n_seeds"] == 0,
                    run,
                )
                continue
            m = re.match(r"(\S+)\s*\[(\S+),\s*(\S+)\]", cell)
            cell_check(
                file,
                f"suite {run} {target} mean",
                m.group(1),
                ss(run, target, "spearman", "mean"),
                "seed_summary mean",
            )
            cell_check(
                file,
                f"suite {run} {target} min",
                m.group(2).rstrip(","),
                ss(run, target, "spearman", "min"),
                "seed_summary range",
            )
            cell_check(
                file,
                f"suite {run} {target} max",
                m.group(3),
                ss(run, target, "spearman", "max"),
                "seed_summary range",
            )
    return n


def t_baselines(body: str, file: str) -> int:
    n = 0
    for row in rows_of(body):
        if "texttt" not in row:
            continue
        run = re.search(r"\\texttt\{([^}]+)\}", row).group(1).replace("\\_", "_")
        c = cells(row)
        n += row_tokens(row)
        for col, model in [(2, "statnn"), (3, "descriptor_only"), (4, "delta_norm")]:
            parts = [p.strip() for p in c[col].split("/")]
            for part, target in zip(parts, ["mmlu_pro", "hellaswag"]):
                if "undefined" in part:
                    v = bl(run, model, target)
                    assertion(
                        file, f"{run} {model} {target} undefined", v is None or math.isnan(v), run
                    )
                    continue
                cell_check(
                    file,
                    f"baseline {run} {model} {target}",
                    part,
                    bl(run, model, target),
                    f"{run} seed-0 baselines.{model}",
                )
    return n


FAMILY_LABELS = {
    "Abliteration": "abliteration",
    "Chain": "chain",
    "LoRA SFT": "lora_sft",
    "Random direction": "random_dir",
    "Task vector": "task_vector",
}


def t_dynrange(body: str, file: str) -> int:
    n = 0
    for row in rows_of(body):
        c = cells(row)
        if c[0] not in FAMILY_LABELS:
            continue
        f = FAMILY_LABELS[c[0]]
        n += row_tokens(row)
        cell_check(
            file, f"dyn {f} rows", c[1], ev_family("mmlu_pro", f)["n"], "family_distribution n"
        )
        for col, target in [(2, "mmlu_pro"), (3, "hellaswag")]:
            m = re.match(r"(\S+)\s*\((\S+)\)\s*\[(\S+),\s*(\S+)\]", c[col])
            e = ev_family(target, f)
            for shown, key in [
                (m.group(1), "mean"),
                (m.group(2), "sd"),
                (m.group(3).rstrip(","), "min"),
                (m.group(4), "max"),
            ]:
                cell_check(
                    file,
                    f"dyn {f} {target} {key}",
                    shown,
                    e[key],
                    "family_distribution (evidence CSV)",
                )
    return n


def t_within(body: str, file: str) -> int:
    n = 0
    for row in rows_of(body):
        c = cells(row)
        if c[0] not in FAMILY_LABELS:
            continue
        f = FAMILY_LABELS[c[0]]
        n += row_tokens(row)
        cell_check(file, f"within {f} rows", c[1], ev_within_n("mmlu_pro", f), "within_family n")
        for col, target in [
            (2, "mmlu_pro"),
            (3, "hellaswag"),
            (4, "kl_to_base"),
            (5, "wikitext_ppl_log"),
        ]:
            v = ev_within(target, f)
            if "undefined" in c[col]:
                assertion(
                    file,
                    f"within {f} {target} undefined (constant target)",
                    v is None,
                    "within_family empty",
                )
                continue
            cell_check(
                file, f"within {f} {target}", c[col], v, "within_family spearman (evidence CSV)"
            )
    return n


def t_zoo(body: str, file: str) -> int:
    counts: dict = defaultdict(Counter)
    for line in ZOO_MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") == "header" or "family" not in r:
            continue
        counts[r["family"]][
            "unclassified" if r.get("eligibility") is None else r["eligibility"]
        ] += 1
    labels = {
        "Random direction": "random_dir",
        "Abliteration": "abliteration",
        "Heretic ARA": "heretic_ara",
        "LoRA SFT": "lora_sft",
        "Task vector": "task_vector",
        "Chain steps": "chain",
    }
    n = 0
    for row in rows_of(body):
        c = cells(row)
        if c[0] not in labels:
            continue
        f = labels[c[0]]
        n += row_tokens(row)
        for col, key in [(1, "production"), (2, "smoke"), (3, "unclassified")]:
            add(
                "CHECKED",
                file,
                f"zoo {f} {key}",
                c[col],
                counts[f][key],
                int(c[col]) == counts[f][key],
                "zoo manifest eligibility count",
            )
    return n


def t_base(body: str, file: str) -> int:
    s = base_summary()
    spec = {
        "GSM8K": ("gsm8k", 200),
        "MATH-500": ("math500", 100),
        "IFEval": ("ifeval", 100),
        "MMLU-Pro": ("mmlu_pro", 300),
        "HellaSwag": ("hellaswag", 300),
        "Keyword refusal": ("refusal_rate", 100),
    }
    cap = {
        "Cap-hit share, GSM8K": ["gsm8k"],
        "Cap-hit share, MATH-500": ["math500"],
        "Cap-hit share, IFEval": ["ifeval"],
        "Cap-hit share, pooled": ["gsm8k", "math500", "ifeval"],
    }
    blog = {"GSM8K": 13.8, "MATH-500": 7.9, "IFEval": 21.0}
    n = 0
    for row in rows_of(body):
        c = cells(row)
        if not c or not c[0] or c[0] == "Metric":
            continue
        n += row_tokens(row)
        if c[0] in cap:
            k, tot = (int(x) for x in c[1].split("/"))
            hit, count = cap_hits(cap[c[0]])
            assertion(
                file,
                f"{c[0]}: counts from base records",
                (hit, count) == (k, tot),
                f"records {hit}/{count}",
            )
            cell_check(file, f"{c[0]} share", c[2], 100 * k / tot, "base records hit_max_tokens")
            if "pooled" not in c[0]:
                lo_v, hi_v = wilson(k, tot)
                lo_s, hi_s = c[3].split("--")
                cell_check(file, f"{c[0]} CI low", lo_s, 100 * lo_v, "Wilson 95%")
                cell_check(file, f"{c[0]} CI high", hi_s, 100 * hi_v, "Wilson 95%")
            continue
        key, denom = spec[c[0]]
        k, tot = (int(x) for x in c[1].split("/"))
        assertion(file, f"{c[0]}: denominator {tot}", tot == denom, "base-final-4096 counts")
        assertion(
            file,
            f"{c[0]}: numerator {k} equals summary value x n",
            round(s[key] * tot) == k,
            "base summary",
        )
        cell_check(file, f"{c[0]} ours", c[2], 100 * s[key], "base-final-4096 summary")
        lo_v, hi_v = wilson(k, tot)
        lo_s, hi_s = c[3].split("--")
        cell_check(file, f"{c[0]} CI low", lo_s, 100 * lo_v, "Wilson 95%")
        cell_check(file, f"{c[0]} CI high", hi_s, 100 * hi_v, "Wilson 95%")
        if c[0] in blog:
            cell_check(file, f"{c[0]} blog value", c[4], blog[c[0]], "docs/RESULTS.md blog column")
        for tok in numeric_tokens(mask_nonprose(c[5])) if len(c) > 5 else []:
            DECLARED.append(
                {"file": file, "token": tok["text"], "reason": "blog samples per item (external)"}
            )
    return n


def t_devices(body: str, file: str) -> int:
    n = 0
    labels = {
        "GSM8K": "GSM8K",
        "MATH-500 exact match": "MATH-500",
        "IFEval": "IFEval",
        "Keyword over-refusal": "Over-refusal",
        "Keyword refusal": "Refusal",
        "MMLU-Pro": "MMLU-Pro",
        "HellaSwag": "HellaSwag",
        "KL to base, nats": "KL",
        "WikiText perplexity": "WikiText",
        "Cap-hit share, pooled": "Cap-hit",
    }
    item_counts = {
        "GSM8K": 50,
        "MATH-500": 25,
        "IFEval": 25,
        "Over-refusal": 50,
        "Refusal": 50,
        "MMLU-Pro": 100,
        "HellaSwag": 100,
        "KL": 50,
        "WikiText": 8,
        "Cap-hit": 100,
    }
    for row in rows_of(body):
        c = cells(row)
        lab = re.sub(r"\s*\(.*$", "", c[0]).strip()
        if lab not in labels or len(c) < 6:
            continue
        d = labels[lab]
        n += row_tokens(row)
        cell_check(
            file,
            f"{d} item count in label",
            re.search(r"\((\d+)", c[0]).group(1),
            item_counts[d],
            "base per-entry eval counts",
        )
        cell_check(
            file, f"dev H1 mean {d}", c[1], DEVS["H1"][d]["mean"], "per-entry eval.json, 11 entries"
        )
        cell_check(
            file, f"dev H1 max {d}", c[2], DEVS["H1"][d]["max"], "per-entry eval.json, 11 entries"
        )
        if not math.isnan(DEVS["H1"][d]["rho"]):
            cell_check(
                file,
                f"dev H1 rho {d}",
                c[3],
                DEVS["H1"][d]["rho"],
                "per-entry eval.json, 11 entries",
            )
        cell_check(
            file, f"dev H2 mean {d}", c[4], DEVS["H2"][d]["mean"], "per-entry eval.json, 4 entries"
        )
        cell_check(
            file, f"dev H2 max {d}", c[5], DEVS["H2"][d]["max"], "per-entry eval.json, 4 entries"
        )
    return n


# ------------------------------------------------------------------ TF-v1 section (6 Nov 2026)

REQUAL = DATA / "runs" / "h1a" / "requalification_report.json"
ROUND3 = DATA / "evals" / "qwen3.5-0.8b" / "tf_v1" / "round3_check.json"
G1_REPORT = DATA / "evals" / "qwen3.5-0.8b" / "tf_v1" / "g1_report.json"
G2_REPORT = DATA / "evals" / "qwen3.5-0.8b" / "tf_v1" / "bridge_g2" / "g2_report.json"
MINUTES = DATA / "runs" / "h1a" / "d3_stratum_minutes.jsonl"
PLAN_SUMMARY = DATA / "runs" / "h1a" / "d3_plan_summary.json"
PANEL_MANIFEST = DATA / "runs" / "h1a" / "panel_manifest.json"
REVIEW_BA = DOCS / "reviews" / "P5" / "board-a-requalification-adjudication-2026-11-06.md"


def rq() -> dict:
    return jload(REQUAL)


def zoo_counts() -> dict:
    lines = [ln for ln in ZOO_MANIFEST.read_text().splitlines() if ln.strip()]
    rows = [json.loads(ln) for ln in lines]
    rows = [r for r in rows if "family" in r and r.get("type") != "header"]
    smoke = [r for r in rows if r.get("eligibility") == "smoke"]
    corrected = [
        r for r in rows if r.get("family") == "heretic_ara" and r.get("eligibility") == "production"
    ]
    unflagged = [r for r in rows if r.get("eligibility") != "smoke"]
    evald = [
        r
        for r in unflagged
        if (DATA / "zoo" / "qwen3.5-0.8b" / r["id"] / "eval.json").exists()
        or (EV / r["id"] / "eval.json").exists()
    ]
    feat = jload(FEATURE_MANIFEST)
    return {
        "lines": len(lines),
        "rows": len(rows),
        "smoke": len(smoke),
        "corrected": len(corrected),
        "unflagged": len(unflagged),
        "eval_unflagged": len(evald),
        "features": feat["n_entries"],
        "features_failed": len(feat.get("failed", {})),
    }


def d3_minutes() -> list[dict]:
    return [json.loads(ln) for ln in MINUTES.read_text().splitlines() if ln.strip()]


def review_nums() -> dict:
    t = read(REVIEW_BA)
    return {
        "lora": float(re.search(r"lora_sft (0\.\d+)", t).group(1)) * 100,
        "chain": float(re.search(r"chain (0\.\d+) \(60%\)", t).group(1)) * 100,
        "tv": -float(re.search(r"task_vector −(0\.\d+)", t).group(1)) * 100,
        "ab": float(re.search(r"A→B (0\.\d+)", t).group(1)) * 100,
        "ba": float(re.search(r"B→A (0\.\d+)", t).group(1)) * 100,
    }


def power_lower(theta_key: str) -> list[float]:
    res = rq()["gate"]["results"]["empirical"][theta_key]
    return [
        res[p][m]["rejection_wilson_lower95"]
        for p in ("conditional", "joint")
        for m in ("bootstrap_t", "bca", "percentile")
    ]


def tfqual_expect() -> list[tuple[str, list]]:
    q = rq()
    mm = q["qualification"]["mmlu_gold_log_prob"]
    wt = q["qualification"]["wikitext_log_ppl"]
    return [
        (
            "(i) panel-mean",
            [
                mm["i_panel_mean"]["limit_delta_over_2"],
                mm["i_panel_mean"]["upper97_5"],
                wt["i_panel_mean"]["limit_delta_over_2"],
                wt["i_panel_mean"]["upper97_5"],
            ],
        ),
        (
            "(ii) maximum",
            [
                mm["ii_ceiling"]["limit_delta"],
                mm["ii_ceiling"]["max_halfwidth"],
                wt["ii_ceiling"]["limit_delta"],
                wt["ii_ceiling"]["max_halfwidth"],
            ],
        ),
        (
            "(iii) tripwire P, upper",
            [
                mm["iii_tripwire"]["limit"],
                mm["iii_tripwire"]["P_upper97_5"],
                wt["iii_tripwire"]["limit"],
                wt["iii_tripwire"]["P_upper97_5"],
            ],
        ),
        (
            "(iii) tripwire P, point",
            [None, mm["iii_tripwire"]["P_point"], None, wt["iii_tripwire"]["P_point"]],
        ),
        (
            "Repeat, maximum",
            [
                mm["repeat"]["limit_max"],
                mm["repeat"]["max_abs_discrepancy"],
                wt["repeat"]["limit_max"],
                wt["repeat"]["max_abs_discrepancy"],
            ],
        ),
        (
            "Repeat, 95th",
            [
                mm["repeat"]["limit_p95_upper"],
                mm["repeat"]["p95_upper97_5_component_bootstrap"],
                wt["repeat"]["limit_p95_upper"],
                wt["repeat"]["p95_upper97_5_component_bootstrap"],
            ],
        ),
        (
            "Reliability",
            [
                mm["reliability"]["lower_bound_threshold"],
                mm["reliability"]["reliability_ci95"][0],
                wt["reliability"]["lower_bound_threshold"],
                wt["reliability"]["reliability_ci95"][0],
            ],
        ),
    ]


def t_tfqual(body: str, file: str) -> int:
    """Checks every value cell of the qualification table against the requalification report.
    Returns the number of value-cell tokens it checked (label tokens are declared separately)."""
    expect = tfqual_expect()
    n = 0
    for row in rows_of(body):
        c = cells(row)
        for prefix, exps in expect:
            if not c[0].startswith(prefix):
                continue
            for idx, exp in enumerate(exps, start=1):
                if exp is None:
                    continue
                for t in numeric_tokens(mask_nonprose(c[idx])):
                    n += 1
                    add(
                        "CHECKED",
                        file,
                        f"tab:tfqual {prefix} col {idx}",
                        t["text"],
                        exp,
                        same(t["text"], exp),
                        "requalification_report.json",
                    )
            break
    return n


def build_tfv1() -> None:
    q = rq()
    fit = q["gate"]["fit"]
    mm = q["qualification"]["mmlu_gold_log_prob"]
    wt = q["qualification"]["wikitext_log_ppl"]
    fam = q["family_counts"]
    w = q["stratum_weights"]
    n_st = q["n_stratum"]
    n_null = fit["n_null"]
    n_over = mm["n_entries_over_half_delta"]
    theta = fit["theta_hat_stratum"] * 100
    n_eff = fit["n_eff"]
    lo_all = fit["leave_out_sensitivity"]["all_rows"]
    lo_wo = fit["leave_out_sensitivity"]["without_over_half_delta"]
    lo_keys = [k for k in lo_all if k != "theta_hat"]
    lows = [lo_all[k][0] * 100 for k in lo_keys]
    highs = [lo_all[k][1] * 100 for k in lo_keys]
    cond_pct_lo = lo_all["conditional_percentile"][0] * 100
    pw = power_lower("theta_0.15")
    r3 = jload(ROUND3)
    r3_max: dict = {}
    for e in r3["entries"].values():
        for k, v in e["max_abs"].items():
            r3_max[k] = max(r3_max.get(k, 0.0), v)
    probe = r3["probe_limits"]
    g1 = jload(G1_REPORT)
    g2 = jload(G2_REPORT)
    g2m = g2["pooled"]["mmlu_pro"]
    g2h = g2["pooled"]["hellaswag"]
    mins = d3_minutes()
    med_wall = statistics.median([r["wall_minutes"] for r in mins])
    plan = jload(PLAN_SUMMARY)
    n_confirm = len(jload(PANEL_MANIFEST)["confirmatory_component_ids"])
    rv = review_nums()
    zc = zoo_counts()
    pw_lo, pw_hi = min(pw), max(pw)

    # ---------------- abstract
    claim("abstract", "on 229 development entries the MMLU-Pro endpoint", [n_st], "requalification_report.json n_stratum")

    # ---------------- experiments: TF-v1 subsection
    E = "experiments"
    presence(E, "As built, it scores 1,000 MMLU-Pro items", RESULTS_MD, r"MMLU-Pro 1,000", "RESULTS TF-v1 config")
    presence(
        E,
        "250 HellaSwag items (a nested prefix of a 500-item ranking)",
        RESULTS_MD,
        r"HellaSwag 250 \(nested prefix of the 500-item ranking\)",
        "RESULTS TF-v1 config",
    )
    presence(E, "200 KL prompts with 64 greedy tokens each", RESULTS_MD, r"KL 200 prompts x 64 greedy tokens", "RESULTS TF-v1 config")
    presence(E, "64 WikiText windows", RESULTS_MD, r"WikiText 64,", "RESULTS TF-v1 config")
    presence(
        E,
        "50 harmful prompts for refusal and 50 harmless prompts for over-refusal",
        RESULTS_MD,
        r"refusal 50 \+ 50",
        "RESULTS TF-v1 config",
    )
    presence(E, "with 13 continuation phrases scored per prompt", RESULTS_MD, r"13 continuation phrases", "RESULTS TF-v1 config")
    presence(E, "measured a warm median of 1.805 min per entry", RESULTS_MD, r"round 2 was 1\.805", "RESULTS round-2 timing")
    claim(E, "against a 1.40 min gate", [r3["gate_minutes"]], "round3_check gate_minutes")
    claim(E, "measured a warm median of 1.003 min per entry", [r3["warm_median_minutes"]], "round3_check warm_median_minutes")
    claim(E, "a range of 0.994 to 1.040 min", list(r3["warm_range_minutes"]), "round3_check warm_range_minutes")
    claim(E, "records of all 11 pilot entries", [len(r3["entries"])], "round3_check entries")
    claim(E, "keeps MMLU-Pro, KL and WikiText at batch 1", [1], "A2.4 batch configuration (round3 records)")
    claim(E, "Over the 11 pilot records, the worst deviations", [len(r3["entries"])], "round3_check entries")
    claim(E, "were 0.0148 on HellaSwag", [r3_max["hellaswag"]], "round3_check max_abs (worst over entries)")
    claim(E, "(probe limit 0.010)", [probe["hellaswag"]], "round3_check probe_limits")
    claim(E, "0.526 on the refusal phrase log-probability", [r3_max["refusal_rate.phrase_log_prob"]], "round3_check max_abs")
    claim(E, "(limit 0.39)", [probe["phrase_log_prob"]], "round3_check probe_limits")
    claim(E, "0.371 on the refusal margin", [r3_max["refusal_rate.margin"]], "round3_check max_abs")
    claim(E, "(limit 0.30)", [probe["margin"]], "round3_check probe_limits")
    claim(E, "changed on 17 of 2,750 items", [r3["hellaswag_argmax_changes_total"], 11 * 250], "round3_check; 11 records x 250 items")
    claim(E, "G1 repeats the registered composition, 13 edited entries and the base", [g1["n_entries"] - 1], "g1_report n_entries minus base")
    claim(E, "for the 13 edited entries and", [g1["n_entries"] - 1], "g1_report n_entries minus base")
    claim(E, "All 14 repeats match their primaries", [g1["n_entries"]], "g1_report n_entries")
    claim(E, "maximum discrepancy of 0 on each endpoint", [0], "g1_report worst_max_abs")
    claim(
        E,
        "the fast tier's own 100 MMLU-Pro and 100 HellaSwag items on the 11 pilot records",
        [g2["n_items_per_benchmark"], g2["n_items_per_benchmark"], g2m["entries"]],
        "g2_report n_items_per_benchmark, entries",
    )
    presence(E, "(limit 0.02)", RESULTS_MD, r"median \|Δ\| 2\.2e-16 \(limit 0\.02\)", "RESULTS G2 round 3 (PREREG §6 limit)")
    claim(E, "99th percentile 0.066", [g2m["gold_score_abs_diff_p99_max"]], "g2_report pooled mmlu_pro p99 max")
    presence(E, "(limit 0.2)", RESULTS_MD, r"p99 0\.066 \(limit 0\.2\)", "RESULTS G2 round 3 (PREREG §6 limit)")
    claim(E, "argmax agreement of 99\\%", [g2m["argmax_agreement_min"] * 100], "g2_report pooled mmlu_pro argmax min")
    presence(E, "(limit 99\\%)", RESULTS_MD, r"argmax agreement 99% \(limit 99%\)", "RESULTS G2 round 3 (PREREG §6 limit)")
    claim(E, "agrees with the fast tier on 99.9\\% of correctness labels", [g2h["correctness_agreement_mean"] * 100], "g2_report pooled hellaswag")
    claim(
        E,
        "(mean 0.9991 over 11 entries, worst entry 0.99)",
        [g2h["correctness_agreement_mean"], g2h["entries"], g2h["correctness_agreement_min"]],
        "g2_report pooled hellaswag",
    )
    claim(
        E,
        "Of the 381 unique D3 deltas in the scoring plan, 378 were scored",
        [plan["d3_unique_scored_ids"], len(mins)],
        "d3_plan_summary d3_unique_scored_ids; d3_stratum_minutes rows",
    )
    claim(
        E,
        "the 229 stratum entries (139 chain, 60 LoRA SFT and 30 task-vector entries) and 149 null-family controls",
        [n_st, fam["chain"], fam["lora_sft"], fam["task_vector"], len(mins) - n_st],
        "requalification family_counts; minutes rows minus stratum",
    )
    claim(E, "median wall time of 1.02 min per entry", [round(med_wall, 2)], "d3_stratum_minutes wall_minutes median")
    claim(E, "all 378 scored entries", [len(mins)], "d3_stratum_minutes rows")
    claim(E, "use 229 stratum rows and 157 null-family control rows", [n_st, n_null], "requalification_report n_stratum, n_null")
    claim(
        E,
        "uses the 229 stratum entries, the 1,000 MMLU-Pro items and the 64 WikiText windows",
        [n_st, mm["n_items"], wt["n_items"]],
        "requalification_report n_stratum, n_items",
    )
    claim(E, "chain 0.700, LoRA SFT 0.206 and task vector 0.094", [w["chain"], w["lora_sft"], w["task_vector"]], "requalification stratum_weights")
    declare(E, "$\\delta/2$", "threshold: half of delta (0.05 MMLU, 0.024 WikiText); limits are checked in the table")
    declare(E, "97.5\\%", "percentile label (97.5th), not a result")
    claim(E, "on the 229 D3 stratum entries (source", [n_st], "requalification_report n_stratum")
    claim(E, "compared with the 20\\% limit", [mm["iii_tripwire"]["limit"] * 100], "requalification tripwire limit")
    claim(E, "The repeat bounds use 10 entries per endpoint.", [mm["repeat"]["entries"]], "requalification repeat entries")
    claim(
        E,
        "On 229 stratum entries scored under one source hash, repeat scoring on 10 entries was bitwise identical (maximum discrepancy 0)",
        [n_st, mm["repeat"]["entries"], mm["repeat"]["max_abs_discrepancy"]],
        "requalification repeat block",
    )
    claim(
        E,
        "was 0.990 for MMLU-Pro gold-letter log-probability and 0.992 for WikiText log-perplexity",
        [mm["reliability"]["reliability_ci95"][0], wt["reliability"]["reliability_ci95"][0]],
        "requalification reliability_ci95 lower bounds",
    )
    claim(E, "of 0.013 against $\\delta = 0.049$", [wt["ii_ceiling"]["max_halfwidth"], wt["ii_ceiling"]["limit_delta"]], "requalification ii_ceiling")
    claim(E, "was 0.114, against a limit of 0.10", [mm["ii_ceiling"]["max_halfwidth"], mm["ii_ceiling"]["limit_delta"]], "requalification ii_ceiling")
    claim(E, "A family-weighted 17.6\\% of entries exceeded", [mm["iii_tripwire"]["P_point"] * 100], "requalification iii_tripwire P_point")
    claim(
        E,
        "upper bound of 22.6\\% against a 20\\% limit",
        [mm["iii_tripwire"]["P_upper97_5"] * 100, mm["iii_tripwire"]["limit"] * 100],
        "requalification iii_tripwire P_upper97_5",
    )
    claim(
        E,
        "Unweighted, 42 of 229 entries (18.3\\%) exceed it",
        [n_over, n_st, n_over / n_st * 100],
        "requalification n_entries_over_half_delta; n_stratum",
    )
    presence(E, "(Spearman 0.92)", REVIEW_BA, r"Spearman\(h, \\\|y\\\|\) = 0\.92", "board A requalification adjudication §1a and §2")
    claim(
        E,
        "The gate used 4,000 simulated panels per scenario and 4,000 bootstrap draws per interval, with seed 20261105",
        [q["gate"]["panels"], q["gate"]["draws"]],
        "requalification gate panels, draws",
    )
    claim(E, "at $\\theta = 0.15$ was 0.43 to 0.47", [0.15, pw_lo, pw_hi], "requalification gate results empirical theta_0.15 rejection_wilson_lower95")
    presence(E, "against the 0.80 required", PREREG, r"≥ 0\.80 for \*\*both\*\*", "PREREG A2.3 gate (a)")
    presence(E, "null-size limit (0.035)", PREREG, r"≤ 0\.035", "PREREG A2.3 gate (b)")
    presence(E, "coverage limit (0.93)", PREREG, r"≥ 0\.93", "PREREG A2.3 gate (b)")
    claim(
        E,
        "by 12.9\\% (effective sample size 39.7, about 40 component equivalents)",
        [theta, n_eff, 40],
        "requalification theta_hat_stratum, n_eff",
    )
    claim(E, "run from $-$0.8\\% to 24.3\\%", [min(lows), max(highs)], "requalification leave_out_sensitivity all_rows, six intervals")
    claim(E, "is +0.3 points", [cond_pct_lo], "requalification conditional_percentile lower bound")
    claim(
        E,
        "LoRA SFT 23\\%, chain 9\\% and task vector $-$5\\%",
        [rv["lora"], rv["chain"], rv["tv"]],
        "board A requalification adjudication §2 (family theta)",
    )
    claim(E, "With the 42 entries whose item half-width", [n_over], "requalification n_entries_over_half_delta")
    claim(
        E,
        "it falls to 0.7\\% [$-$20\\%, 15\\%]",
        [lo_wo["theta_hat"] * 100, lo_wo["conditional_bootstrap_t"][0] * 100, lo_wo["conditional_bootstrap_t"][1] * 100],
        "requalification leave_out_sensitivity without_over_half_delta",
    )
    claim(E, "fit on all 229 stratum rows and the 157 null-family rows", [n_st, n_null], "requalification_report n_stratum, n_null")
    claim(E, "(10.5\\% and 11.7\\%)", [rv["ab"], rv["ba"]], "board A requalification adjudication §2 (item halves)")
    claim(E, "fit on 229 stratum rows plus 157 null-family controls", [n_st, n_null], "requalification_report n_stratum, n_null")

    # ---------------- discussion: H1a status and closing paragraph
    D = "discussion"
    claim(D, "on the 229 development stratum entries", [n_st], "requalification_report n_stratum")
    claim(D, "gave a lower power bound of 0.43 to 0.47 at $\\theta = 0.15$", [pw_lo, pw_hi, 0.15], "requalification gate results empirical theta_0.15")
    presence(D, "against the 0.80 required. On the same panel", PREREG, r"≥ 0\.80 for \*\*both\*\*", "PREREG A2.3 gate (a)")
    claim(D, "the exploratory residual estimate of 12.9\\%", [theta], "requalification theta_hat_stratum")
    claim(D, "sealed panel of 106 fresh components", [n_confirm], "panel_manifest confirmatory_component_ids")
    claim(D, "with 1,000 paired items per edit", [mm["n_items"]], "requalification mmlu n_items")
    claim(D, "the power gate gave lower bounds of 0.43 to 0.47 at $\\theta = 0.15$", [pw_lo, pw_hi, 0.15], "requalification gate results empirical theta_0.15")
    presence(D, "against the 0.80 required. A fresh outcome", PREREG, r"≥ 0\.80 for \*\*both\*\*", "PREREG A2.3 gate (a)")

    # ---------------- zoo (manifest counts; corrected Heretic ARA entries, D6 addendum 4)
    Z = "zoo"
    claim(Z, "The file holds 1,267 lines", [zc["lines"]], "zoo manifest line count")
    claim(Z, "1,259 entry rows, one per unique ID", [zc["rows"]], "zoo manifest entry rows")
    claim(
        Z,
        "The manifest therefore has 1,259 rows: 1,247 unflagged rows, which include the 12 corrected entries, and the 12 flagged rows.",
        [zc["rows"], zc["unflagged"], zc["corrected"], zc["smoke"]],
        "zoo manifest: rows, unflagged = rows minus smoke, corrected = heretic production, smoke",
    )
    claim(Z, "The feature manifest records 1,259 entries", [zc["features"]], "features manifest n_entries")
    claim(
        Z,
        "Of the 1,247 unflagged rows, 389 have evaluation records, and none of the 24 flagged or corrected Heretic ARA entries does.",
        [zc["unflagged"], zc["eval_unflagged"], zc["smoke"] + zc["corrected"]],
        "unflagged rows with eval.json; 24 = 12 flagged + 12 corrected",
    )
    claim(Z, "1,259 entry rows, counted by recorded eligibility", [zc["rows"]], "zoo manifest entry rows")
    presence(Z, "(D6 addendum 4)", DECISIONS, r"D6 addendum 4", "DECISIONS D6 addendum 4")
    presence(Z, "carry production eligibility under D6 addendum 4", DECISIONS, r"D6 addendum 4", "DECISIONS D6 addendum 4")
    assertion(Z, "zoo features: no failed feature extractions", zc["features_failed"] == 0, "features manifest failed")
    corrected = [
        r["id"]
        for r in zoo_rows()
        if r.get("family") == "heretic_ara" and r.get("eligibility") == "production"
    ]
    listed = set()
    for name in ("d3_ids.stratum.txt", "d3_ids.null.txt", "d3_ids.fresh_spares.txt"):
        listed |= set((DATA / "runs" / "h1a" / name).read_text().split())
    pm_text = read(DATA / "runs" / "h1a" / "panel_manifest.json")
    assertion(
        Z,
        "zoo: corrected Heretic entries absent from the D3 and fresh lists",
        not any(i in listed or i in pm_text for i in corrected),
        "d3_ids.*.txt and panel_manifest.json",
    )
    assertion(Z, "zoo: 12 corrected Heretic entries (D6 addendum 4)", len(corrected) == 12, "zoo manifest")


def zoo_rows() -> list[dict]:
    out = []
    for ln in ZOO_MANIFEST.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            if "family" in r and r.get("type") != "header":
                out.append(r)
    return out



TABLE_CHECKS = {
    "tab:v2-suite": t_suite,
    "tab:v2-baselines": t_baselines,
    "tab:dynrange": t_dynrange,
    "tab:within": t_within,
    "tab:zoo-counts": t_zoo,
    "tab:base-repro": t_base,
    "tab:devices": t_devices,
    "tab:tfqual": t_tfqual,
}


# ------------------------------------------------------------------ D8 artefact loss (6 Nov 2026)
# Text between "% D8-BEGIN" and "% D8-END" was written or rewritten after the incident. Every numeric
# token there must be claimed (from a committed JSON or a committed document), declared, or quoted.
# Legacy text outside those blocks is checked only when data/runs exists (DATA_PRESENT).

H1A_JSON = DOCS / "reviews" / "P5" / "h1a-companions-2026-11-06.json"
H1W_JSON = DOCS / "reviews" / "P5" / "h1w-gate-2026-11-06.json"
INCIDENT = DOCS / "reviews" / "P5" / "board-a-incident-d8-2026-11-06.md"
REGEN = DOCS / "reviews" / "P5" / "d8-regeneration-2026-11-06.md"
D3V2R = DOCS / "reviews" / "P4" / "d3v2r-summary-2026-11-06.md"
D8_BLOCK = re.compile(r"% D8-BEGIN\n(.*?)% D8-END", re.DOTALL)
LEGACY = [0]  # legacy tokens not checked because their sources are absent
MODE_NOTE = ""


def d8_spans(raw: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in D8_BLOCK.finditer(raw)]


def in_d8(spans: list[tuple[int, int]], pos: int) -> bool:
    return any(a <= pos < b for a, b in spans)


def tex2md(s: str) -> str:
    """Render a LaTeX paragraph the way the incident review writes it, for the verbatim check."""
    s = s.replace("``", '"').replace("''", '"').replace("\u2019", "'")
    s = re.sub(r"\(\\ifarxivpreprint(.*?)\\else.*?\\fi\)", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", s)
    s = re.sub(r"\\(?:texttt|path)\{([^{}]*)\}", r"`\1`", s)
    s = s.replace("\\_", "_").replace("\\%", "%").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def quoted(file: str, snippet: str, patterns: list[str], doc: str, desc: str) -> None:
    """A prose snippet whose facts must appear in a committed document (each regex must match)."""
    claim(file, snippet, None, desc)
    for pat in patterns:
        assertion(file, f"{desc}: /{pat}/", re.search(pat, doc) is not None, desc)


def build_d8() -> None:
    a = jload(H1A_JSON)
    w = jload(H1W_JSON)
    inc = read(INCIDENT)
    dec = read(DECISIONS)
    hyp = read(HYPOTHESIS)
    src_a = "docs/reviews/P5/h1a-companions-2026-11-06.json"
    src_w = "docs/reviews/P5/h1w-gate-2026-11-06.json"

    for needle, label in [
        ("untested (not falsified)", "H1a"),
        ("not distinguishable from zero", "H1w"),
        ("not testable on the surviving zoo", "H2"),
    ]:
        assertion("experiments", f"HYPOTHESIS.md status line quoted for {label}", needle in hyp, "docs/HYPOTHESIS.md")

    # ---- H1a companions (experiments): table rows, the whole row text is the snippet
    fx = a["flexible_nuisance"]["family_x_natural_spline"]
    sh = a["flexible_nuisance"]["shared_natural_spline_family_slope"]
    reg = a["registered"]
    eq = a["equal_component_weight"]
    st = a["strata"]
    hv = a["h_variants"]
    t3 = st["tertile"]["T3"]
    lora = st["family"]["lora_sft"]
    tv = st["family"]["task_vector"]
    c1 = st["cross"]["chain|T1"]
    inv = hv["inverse_h_squared_weighted"]
    rst = hv["restricted_h_le_half_delta"]
    rf = hv["refit_without_over_half_delta"]
    splits = a["item_split"]["random_splits"]
    means = [v["mean"] for v in splits["by_combination"].values()]
    lows = [v[0] for v in fx["six_intervals"].values() if isinstance(v, list)]
    n_wide = a["half_width_summary"]["n_over_half_delta"]
    n_cells = sum(len(st[k]) for k in ("family", "tertile", "cross", "within_family_tertile", "pre_declared"))
    F = "experiments"
    rows = [
        (r"Registered estimate (reference) & 229 & 120 & 0.129 & [0.002, 0.239]",
         [a["n_rows"], a["n_components"], reg["theta"], *reg["cluster_percentile95"]]),
        (r"Flexible nuisance, family by natural spline & 229 & 120 & 0.135 & [0.009, 0.253]",
         [a["n_rows"], a["n_components"], fx["theta"], *fx["cluster_percentile95"]]),
        (r"Flexible nuisance, shared spline with family slopes & 229 & 120 & 0.126 & [$-$0.003, 0.242]",
         [a["n_rows"], a["n_components"], sh["theta"], *sh["cluster_percentile95"]]),
        (r"Equal component weights & 229 & 120 & 0.176 & [0.027, 0.301]",
         [a["n_rows"], a["n_components"], eq["theta_row_weight_1_over_nc"], *eq["cluster_percentile95_row_weight_1_over_nc"]]),
        (r"High-norm tertile, all families & 77 & 56 & 0.195 & [0.045, 0.321]",
         [t3["n_rows"], t3["n_components"], t3["theta"], *t3["cluster_percentile95"]]),
        (r"LoRA SFT only & 60 & 60 & 0.230 & [0.014, 0.399]",
         [lora["n_rows"], lora["n_components"], lora["theta"], *lora["cluster_percentile95"]]),
        (r"Task vectors only & 30 & 27 & $-$0.048 & [$-$0.489, 0.225]",
         [tv["n_rows"], tv["n_components"], tv["theta"], *tv["cluster_percentile95"]]),
        (r"Chain, lowest-norm tertile & 42 & 27 & $-$0.776 & [$-$1.956, $-$0.092]",
         [c1["n_rows"], c1["n_components"], c1["theta"], *c1["cluster_percentile95"]]),
        (r"Inverse-square $h_i$ weighting & 229 & 120 & $-$0.257 & [$-$0.666, 0.074]",
         [a["n_rows"], a["n_components"], inv["theta"], *inv["cluster_percentile95"]]),
        (r"Rows with $h_i \le \delta/2$, fit on all rows & 187 & 96 & 0.007 & [$-$0.164, 0.154]",
         [2, rst["n_rows"], rst["n_components"], rst["theta"], *rst["cluster_percentile95"]]),
        (r"Refit without rows with $h_i > \delta/2$ & 187 & 96 & 0.119 & [0.0003, 0.231]",
         [2, rf["n_rows"], rf["n_components"], rf["theta"], *rf["cluster_percentile95"]]),
    ]
    for snip, vals in rows:
        claim(F, snip, vals, f"{src_a}: companion table row")
    claim(F, "percentile intervals with 4,000 draws", [4000], "h1a-companions: draws")

    # ---- H1a companions (experiments): prose
    prose = [
        (r"each uses the 229 stratum rows, and each interval is a component-bootstrap percentile interval over 4,000 draws",
         [a["n_rows"], 4000]),
        (r"is 0.135, and a shared spline with family slopes gives 0.126", [fx["theta"], sh["theta"]]),
        (r"lie between $-$0.005 and 0.008", [min(lows), max(lows)]),
        (r"equal component weights give 0.176, the high-norm tertile gives 0.195, and LoRA SFT alone gives 0.230",
         [eq["theta_row_weight_1_over_nc"], t3["theta"], lora["theta"]]),
        (r"($-$0.048, 30 rows, 27 components)", [tv["theta"], tv["n_rows"], tv["n_components"]]),
        (r"($-$0.776, 42 rows, 27 components), which is one of 27 stratum cells in the companion file",
         [c1["theta"], c1["n_rows"], c1["n_components"], n_cells]),
        (r"which gives $-$0.257 [$-$0.666, 0.074]", [inv["theta"], *inv["cluster_percentile95"]]),
        (r"The 42 rows with $h_i > \delta/2$ carry 0.09\% of that weight",
         [n_wide, 2, inv["share_of_weight_on_rows_over_half_delta"] * 100]),
        (r"Restricting the sum to the 187 rows with $h_i \le \delta/2$, under the fit on all rows, gives 0.007 [$-$0.164, 0.154]",
         [rst["n_rows"], 2, rst["theta"], *rst["cluster_percentile95"]]),
        (r"Refitting the nuisance and StatNN without the 42 wide rows gives 0.119 [0.0003, 0.231] on 187 rows",
         [n_wide, rf["theta"], *rf["cluster_percentile95"], rf["n_rows"]]),
        (r"so the 0.7\% reported above", [rst["theta"] * 100]),
        (r"The 20 random item splits all give positive estimates. Their means, across the four combinations of split halves, run from 0.109 to 0.126.",
         [splits["n_splits"], min(means), max(means)]),
    ]
    for snip, vals in prose:
        claim(F, snip, vals, f"{src_a}: text")

    # ---- H1w (experiments): prose, caption and table
    wc = w["companions"]
    wg = w["gate"]
    iv = wg["interval_table"]
    pw = wg["criteria"]["per_method"]["bootstrap_t"]
    pw_c = pw["(a) power lower95 >= 0.80 at theta 0.15, conditional"]["value"]
    pw_j = pw["(a) power lower95 >= 0.80 at theta 0.15, joint"]["value"]
    fxw = wc["flexible_nuisance"]["family_x_natural_spline"]
    lw = wc["strata"]["family"]["lora_sft"]
    tw = wc["strata"]["family"]["task_vector"]
    lows_w = [x["theta_interval"][0] for x in iv]
    highs_w = [x["theta_interval"][1] for x in iv]
    h1w = [
        (r"WikiText met the A2.1 qualification, with a maximum item half-width of 0.013 against $\delta = 0.049$",
         [wc["half_width_summary"]["max_h"], math.log(1.05)]),
        (r"The estimate is $\hat\theta = 0.0071$", [wc["registered"]["theta"]]),
        (r"Its six lower bounds run from $-$0.094 to $-$0.139", [max(lows_w), min(lows_w)]),
        (r"at $\theta = 0.15$ is 0.73 for the conditional bootstrap-t procedure and 0.60 for the joint procedure, against 0.80",
         [0.15, pw_c, pw_j, 0.80]),
        (r"The flexible-nuisance estimate is $-$0.049 (percentile interval $-$0.151 to 0.037)",
         [wg["criteria"]["flexible_nuisance_theta_hat"], *fxw["cluster_percentile95"]]),
        (r"$-$1.13 for LoRA SFT (60 rows, 60 components) and 0.38 for task vectors (30 rows, 27 components)",
         [lw["theta"], lw["n_rows"], lw["n_components"], tw["theta"], tw["n_rows"], tw["n_components"]]),
        (r"the same 229 stratum rows in 120 components", [w["n_stratum"], wc["n_components"]]),
        (r"with 4,000 panels per scenario and 4,000 bootstrap draws, on 229 stratum rows in 120 components",
         [4000, 4000, w["n_stratum"], wc["n_components"]]),
        (r"with limit 0.035", [0.035]),
        (r"at $\theta = 0.15$, with limit 0.93", [0.15, 0.93]),
        (r"Coverage at $\theta = 0.15$", [0.15]),
    ]
    for snip, vals in h1w:
        claim(F, snip, vals, f"{src_w}: text")
    h1w_rows = [
        "Conditional, bootstrap-t & [$-$0.094, 0.098] & 0.025 & 0.956",
        "Conditional, BCa & [$-$0.098, 0.097] & 0.027 & 0.950",
        "Conditional, percentile & [$-$0.098, 0.097] & 0.022 & 0.959",
        "Joint, bootstrap-t & [$-$0.139, 0.122] & 0.028 & 0.952",
        "Joint, BCa & [$-$0.131, 0.114] & 0.030 & 0.948",
        "Joint, percentile & [$-$0.116, 0.125] & 0.032 & 0.956",
    ]
    for snip, x in zip(h1w_rows, iv):
        claim(F, snip, [x["theta_interval"][0], x["theta_interval"][1], x["null_size"], x["coverage_at_0.15"]],
              f"{src_w}: interval_table")

    # ---- H1 status and dates in the experiments, discussion, abstract and introduction
    quoted(F, "H2 is not testable on the surviving zoo", [r"not testable on the surviving zoo"], hyp,
           "docs/HYPOTHESIS.md H2 status")
    quoted(F, "only three evaluated lineages of that length remain", [r"3 evaluated lineages of length ≥ 4 remain"],
           dec, "docs/DECISIONS.md D8 addendum")
    declare(F, "6 November 2026", "incident date (D8)")
    claim("discussion", "give point estimates from 0.135 to 0.230", [fx["theta"], lora["theta"]], f"{src_a}: text")
    claim("discussion", "Its estimate, 0.0071, is not distinguishable from zero", [wc["registered"]["theta"]],
          f"{src_w}: text")
    quoted("introduction", "H2 is not testable on the surviving zoo", [r"not testable on the surviving zoo"], hyp,
           "docs/HYPOTHESIS.md H2 status")
    quoted("abstract", "H2 is not testable on the surviving zoo", [r"not testable on the surviving zoo"], hyp,
           "docs/HYPOTHESIS.md H2 status")
    quoted("abstract", "The exploratory WikiText claim H1w is not distinguishable from zero",
           [r"not distinguishable from zero"], hyp, "docs/HYPOTHESIS.md H1w status")
    declare("abstract", "6 November 2026", "incident date (D8)")
    declare("method", "6 November 2026", "incident date (D8)")

    # ---- zoo: artefact-loss footnote and the feature-manifest sentence
    declare("zoo", "6 November 2026", "incident date (D8)")
    quoted("zoo", "Delta bytes are missing for 358 of the 675 chain rows",
           [r"358 truly missing", r"317/675 chain deltas usable"], dec, "docs/DECISIONS.md D8 addendum")
    quoted("zoo", "The other 317 chain rows keep bytes (254 stored and 63 byte-identical duplicates)",
           [r"317 of the 675 chain steps \(254 stored, 63 byte-identical duplicates"], inc,
           "incident review section 1.1")
    quoted("zoo", "all 584 non-chain rows keep bytes", [r"all 584 non-chain entries"], inc,
           "incident review section 1.1")
    quoted("zoo", "at most 901 entries can have features: the 838 rows with stored delta bytes and the 63 chain rows",
           [r"at most 901 \(838 \+ 63 aliases\)"], inc, "incident review section 0 (Z3)")

    # ---- appendix: verbatim paragraph, chain status, classification
    app = read(SEC / "appendix.tex")
    m = re.search(r"% D8-VERBATIM-BEGIN\n(.*?)\n% D8-VERBATIM-END", app, re.DOTALL)
    para = m.group(1).strip()
    md_line = next(ln for ln in inc.splitlines() if ln.startswith("> **Artefact loss.**"))
    md_para = re.sub(r"\s+", " ", md_line[2:]).strip()
    claim("appendix", para, None, "verbatim: incident review section 1.1")
    assertion("appendix", "verbatim paragraph equals incident review section 1.1",
              tex2md(para) == md_para, "docs/reviews/P5/board-a-incident-d8-2026-11-06.md")
    quoted("appendix", "Of the 675 chain steps, 317 keep delta bytes and 358 do not.",
           [r"317 of the 675 chain steps", r"358 need replay"], inc, "incident review section 1.1")
    quoted("appendix", "Of the 99 lineages of length four or more, six keep their first four steps, and three of those six are evaluated.",
           [r"99 lineages of length ≥ 4", r"only \*\*6\*\* have ≥ 4 leading steps recovered, \*\*3\*\* of those evaluated"],
           inc, "incident review section 3")
    quoted("appendix", "The registered 229-row TF-v1 panel", [r"229-row panel"], inc, "incident review section 3")
    claim("appendix", "recomputable from surviving raw records & 12", [12], "incident review section 1 counts")
    claim("appendix", "regenerable under the recorded source hash & 21", [21], "incident review section 1 counts")
    claim("appendix", "cannot be regenerated; reported from logs & 4", [4], "incident review section 1 counts")
    claim("appendix", "Total & & 37", [37], "incident review section 1 counts")

    # ---- regeneration outcome (appendix, experiments, reliability): docs/reviews/P5/d8-regeneration-2026-11-06.md
    reg = read(REGEN)
    rs = "regeneration report"
    quoted("appendix", "Of the 1,259 manifest rows, 838 have delta bytes, 63 are byte-identical to a stored delta and 358 are lost",
           [r"1,259 manifest rows: \*\*838 with bytes, 63 aliased, 358 lost\*\*"], reg, f"{rs} summary row 1")
    quoted("appendix", "the 838 stored deltas show no hash, header or norm mismatch",
           [r"0 hash, header or norm misses over the 838 stored deltas"], reg, f"{rs} summary row 1")
    quoted("appendix", "Feature re-extraction covered 901 entries with no failure",
           [r"\*\*901 entries re-extracted\*\* \(838 \+ 63 aliases\), 0 failed"], reg, f"{rs} summary row 2")
    quoted("appendix", "a ten-entry re-extraction in a fresh process was byte-identical",
           [r"10-entry re-extraction byte-identical"], reg, f"{rs} summary row 2")
    quoted("appendix", "all 458 records are bitwise identical across the source trees on every field they share",
           [r"458 of 458 identical on every field both records carry"], reg, f"{rs} step 3")
    quoted("appendix", "KL differs by at most 0.0003 nats between seeds",
           [r"KL differs by up to 3\.3e-4 nats between seeds"], reg, f"{rs} step 4")
    quoted("appendix", "12 of 12 repeatable G1 entries were bitwise identical (2 of the original 14 are chain predecessors whose deltas are lost)",
           [r"G1 \*\*12 of 12\*\* repeats bitwise \(14 of 14 originally; 2 chain predecessors lost\)"], reg, f"{rs} summary row 5")
    quoted("appendix", "all 11 rows of the G2 bridge reproduced",
           [r"G2 \*\*11 of 11 table rows\*\* reproduce"], reg, f"{rs} summary row 5")
    quoted("appendix", "all 28 recoverable rows of the requalification audit reproduced to the printed four digits",
           [r"\*\*28 of 28\*\* recoverable \(y, h\) rows reproduce to the printed 4 digits"], reg, f"{rs} summary row 5")
    quoted("experiments", "regenerated on 6 November 2026 under source hash \\texttt{a224779bf873} (run \\texttt{20261106-062239-29d865d2})",
           [r"Run `20261106-062239-29d865d2`", r"`source_hash a224779bf873"], reg, f"{rs} step 5")
    quoted("reliability", "regenerated on 6 November 2026 as run \\texttt{p2-batch/20261106-045437-03cde0e3} under source hash \\texttt{ec88cfd95d7d}",
           [r"run `p2-batch/20261106-045437-03cde0e3`", r"`ec88cfd95d7d`"], reg, f"{rs} step 4")
    declare("appendix", "6 November 2026", "incident date (D8)")
    declare("experiments", "6 November 2026", "incident date (D8)")
    declare("reliability", "6 November 2026", "incident date (D8)")

    # ---- recovered-set rerun (experiments): docs/reviews/P4/d3v2r-summary-2026-11-06.md
    r = read(D3V2R)
    rd = "d3v2r summary"
    def row(name):
        ln = next(l for l in r.splitlines() if l.startswith(f"| {name} |"))
        return [c.strip() for c in ln.strip("|").split("|")]
    seed = row("m1_matched-seed"); traj = row("m1_matched-trajectory")
    quoted("experiments", "That set has 297 matched rows in 270 ancestry clusters, against 345 rows in 278 clusters above",
           [r"\| 345 → 297 \| 278 → 270 \|"], r, f"{rd} comparison")
    quoted("experiments", "48 chain rows with a lost ancestor delta are absent from every fold, and the chain holdout's test rows fall from 98 to 50",
           [r"48 chain rows with a lost ancestor delta removed from every fold; the chain holdout's test rows fall from 98 to 50"], r, f"{rd} comparison")
    assertion("experiments", f"{rd}: StatNN seed split 0.54 / 0.65", seed[5] == "0.54 / 0.65", rd)
    assertion("experiments", f"{rd}: descriptor-only seed split 0.48 / 0.58", seed[6] == "0.48 / 0.58", rd)
    assertion("experiments", f"{rd}: delta-norm seed split 0.16 / 0.21", seed[7] == "0.16 / 0.21", rd)
    claim("experiments", "StatNN reaches 0.54 on MMLU-Pro and 0.65 on HellaSwag on the seed split, descriptor-only reaches 0.48 and 0.58, and delta norm alone reaches 0.16 and 0.21", None, rd)
    assertion("experiments", f"{rd}: m1_matched-seed cells", seed[1] == "0.47 [0.43,0.49]" and seed[2] == "0.54 [0.46,0.59]", rd)
    assertion("experiments", f"{rd}: m1_matched-trajectory cells", traj[1] == "0.46 [0.44,0.48]" and traj[2] == "0.55 [0.46,0.62]", rd)
    claim("experiments", "M1-matched at 0.47 [0.43, 0.49] and 0.54 [0.46, 0.59]", None, rd)
    claim("experiments", "M1-matched at 0.46 [0.44, 0.48] and 0.55 [0.46, 0.62]", None, rd)
    fam = [row(f"{m}-family") for m in ("m1", "m1_matched", "m2", "m2_matched", "m5", "m6")]
    mm = [float(x[1].split()[0]) for x in fam]; hs = [float(x[2].split()[0]) for x in fam]
    assertion("experiments", f"{rd}: family-split ranges", min(mm) >= -0.07 - 5e-3 and max(mm) <= 0.18 + 5e-3 and min(hs) >= -0.21 - 5e-3 and max(hs) <= 0.16 + 5e-3, rd)
    claim("experiments", "every model lies within $-$0.07 to 0.18 on MMLU-Pro and within $-$0.21 to 0.16 on HellaSwag", None, rd)
    assertion("experiments", f"{rd}: descriptor-only within 0.07 of StatNN", round(0.54 - 0.48, 2) <= 0.07 and round(0.65 - 0.58, 2) <= 0.07, rd)
    claim("experiments", "descriptor-only is within 0.07 of it", None, rd)
    assertion("experiments", f"{rd}: m3-family no metrics", "| m3-family | no metrics |" in r, rd)
    claim("appendix", "Of the 21 B results, 15 depend on chain rows", [21, 15], "incident review section 1 counts")
    for needle in ["**A = 12**", "**B = 21**", "**C = 4 results", "Total 37 result groups"]:
        assertion("appendix", f"incident review count: {needle}", needle in inc, "incident review section 1")


# ------------------------------------------------------------------ runner


def run_claims(
    raw: str, masked: str, file: str, lo: int, hi: int, skip: list[tuple[int, int]]
) -> tuple[int, set[int]]:
    """Run claims whose first occurrence lies in [lo, hi) and outside skip spans.
    Returns (number of tokens they account for, token start offsets covered)."""
    accounted = 0
    covered: set[int] = set()
    for c in CLAIMS:
        if c.file != file:
            continue
        idx = raw.find(c.snippet)
        if idx < 0:
            if c.snippet and not any(
                f["file"] == file and f["snippet"] == c.snippet for f in FAILED_SNIPPETS
            ):
                FAILED_SNIPPETS.append({"file": file, "snippet": c.snippet})
                add("MISSING", file, c.snippet[:90], "", None, False, c.src)
            continue
        if not (lo <= idx < hi) or any(a <= idx < b for a, b in skip):
            continue
        end = idx + len(c.snippet)
        all_toks = numeric_tokens(masked, idx, end)
        covered |= {t["start"] for t in all_toks}
        toks = [t for t in all_toks if not t["seed"]]
        accounted += len(all_toks)
        if c.values is None:
            add("PRESENCE", file, c.snippet[:80], "", None, True, c.src)
            continue
        if len(toks) != len(c.values):
            add("COUNT", file, c.snippet[:80], f"{len(toks)} tokens", len(c.values), False, c.src)
            continue
        for t, val in zip(toks, c.values):
            add("CHECKED", file, c.snippet[:80], t["text"], val, same(t["text"], val), c.src)
    return accounted, covered


def run_declared(
    raw: str, masked: str, file: str, lo: int, hi: int, skip: list[tuple[int, int]]
) -> tuple[int, set[int]]:
    count = 0
    covered: set[int] = set()
    for f, snip, reason in DECLARED_RULES:
        if f != file or not snip:
            continue
        for m in re.finditer(re.escape(snip), raw):
            if not (lo <= m.start() < hi) or any(a <= m.start() < b for a, b in skip):
                continue
            for t in numeric_tokens(masked, m.start(), m.end()):
                covered.add(t["start"])
                count += 1
                DECLARED.append(
                    {"file": file, "token": t["text"], "reason": reason or "(no reason given)"}
                )
    return count, covered


def run() -> int:
    global MODE_NOTE
    if DATA_PRESENT:
        MODE_NOTE = "Mode: full (source artefacts present under data/)."
        build()
        build_tfv1()
    else:
        MODE_NOTE = (
            "Mode: committed-only. data/runs is absent (artefact loss, D8), so the legacy claims that read "
            "data/ are not run. Checked: D8 blocks, the verbatim incident paragraph, the D8 table companions, "
            "quoted facts in committed documents, and every \\path target."
        )
    build_d8()
    unclaimed: list[dict] = []
    total_prose = total_table = seed_labels = 0
    for name in SECTIONS:
        raw = (SEC / f"{name}.tex").read_text()
        masked_full = mask_nonprose(raw)
        spans = d8_spans(raw)
        envs = list(table_envs(raw))
        table_spans = [(e0, e1) for _, _, _, e0, e1 in envs]
        body_spans = [(b0, b1) for _, b0, b1, _, _ in envs]
        # tables: each body must be fully accounted for by its checker, in-table claims and declarations
        for lab, b0, b1, e0, _ in envs:
            body = numeric_tokens(masked_full, b0, b1)
            if not DATA_PRESENT and not in_d8(spans, e0):
                LEGACY[0] += len(body)
                continue
            total_table += len(body)
            checker_count = 0
            fn = TABLE_CHECKS.get(lab)
            if fn is not None:
                checker_count = fn(raw[b0:b1], name)
            _, cov_claims = run_claims(raw, masked_full, name, b0, b1, [])
            _, cov_decl = run_declared(raw, masked_full, name, b0, b1, [])
            positional = cov_claims | cov_decl | {t["start"] for t in body if t["seed"]}
            seed_labels += sum(1 for t in body if t["seed"])
            accounted = checker_count + len(positional)
            if accounted != len(body):
                unclaimed.append(
                    {
                        "file": name,
                        "line": raw[:b0].count("\n") + 1,
                        "token": f"table {lab}",
                        "context": f"accounted {accounted} of {len(body)} tokens in the tabular body; "
                        f"body tokens: {[t['text'] for t in body]}",
                    }
                )
        # prose: blank the table bodies, keep captions
        blanked = list(raw)
        for b0, b1 in body_spans:
            for i in range(b0, b1):
                if blanked[i] != "\n":
                    blanked[i] = " "
        prose_raw = "".join(blanked)
        masked = mask_nonprose(prose_raw)
        _, cov_c = run_claims(raw, masked, name, 0, len(raw), body_spans)
        _, cov_d = run_declared(raw, masked, name, 0, len(raw), body_spans)
        covered = cov_c | cov_d
        for t in numeric_tokens(masked):
            total_prose += 1
            if t["seed"]:
                seed_labels += 1
                DECLARED.append({"file": name, "token": t["text"], "reason": "seed label"})
                continue
            if t["start"] in covered:
                continue
            if not DATA_PRESENT and not in_d8(spans, t["start"]):
                LEGACY[0] += 1
                continue
            line = raw[: t["start"]].count("\n") + 1
            unclaimed.append(
                {
                    "file": name,
                    "line": line,
                    "token": t["text"],
                    "context": raw[max(0, t["start"] - 70) : t["end"] + 30].replace("\n", " "),
                }
            )
    for name in SECTIONS:
        for p in re.findall(r"\\path\{([^}]+)\}", (SEC / f"{name}.tex").read_text()):
            assertion(name, f"path exists: {p}", (ROOT / p.replace("\\_", "_")).exists(), p)
    return report(total_prose, total_table, seed_labels, unclaimed)


def report(total_prose: int, total_table: int, seed_labels: int, unclaimed: list[dict]) -> int:
    failed = [r for r in RESULTS if not r["ok"]]
    checked = [r for r in RESULTS if r["kind"] == "CHECKED" and r["ok"]]
    asserts_ok = [r for r in RESULTS if r["kind"] == "ASSERT" and r["ok"]]
    presence_ok = [r for r in RESULTS if r["kind"] == "PRESENCE"]
    out = ["# Paper v2 numerical audit: output of scripts/audit_paper_numbers.py", ""]
    out.append(f"Sections audited: {', '.join(SECTIONS)}.")
    out.append(MODE_NOTE)
    if not DATA_PRESENT:
        out.append(
            f"Legacy tokens not checked (their sources are absent): {LEGACY[0]} in prose and tables, "
            "outside the D8 blocks. They are listed as not run, not as passing."
        )
    out.append(
        f"Numeric tokens: {total_prose} in prose, {total_table} in tables; {seed_labels} seed labels and "
        f"{sum(1 for d in DECLARED)} declared tokens are listed below, not checked."
    )
    out.append(
        f"Values checked against sources and passing: {len(checked)}. Assertions passing: {len(asserts_ok)}. "
        f"Presence claims: {len(presence_ok)}."
    )
    out.append(
        f"Failures: {len(failed)} (mismatches, missing snippets, failed assertions or counts). "
        f"Unclaimed tokens: {len(unclaimed)}."
    )
    out.append("")
    out.append("## Failures")
    if not failed:
        out.append("None.")
    for r in failed:
        out.append(
            f"- {r['kind']} [{r['file']}] `{r['label']}`: shown {r['shown']!r}, expected {r['expected']!r} ({r['src']})"
        )
    out.append("")
    out.append("## Unclaimed tokens")
    if not unclaimed:
        out.append("None.")
    for u in unclaimed:
        out.append(f"- [{u['file']}:{u['line']}] `{u['token']}`: {u['context']}")
    out.append("")
    out.append("## Declared, not checked (listed; do not fail the audit)")
    tally = Counter((d["file"], d["reason"]) for d in DECLARED)
    for (f, reason), c in sorted(tally.items()):
        out.append(f"- {f}: {c} token(s). {reason}")
    out.append("")
    out.append("## Checked values by section")
    by_file: dict = defaultdict(list)
    for r in RESULTS:
        if r["kind"] == "CHECKED":
            by_file[r["file"]].append(r)
    for f in SECTIONS:
        rows = by_file.get(f, [])
        if not rows:
            continue
        out.append(f"### {f} ({len(rows)})")
        for r in rows:
            exp = r["expected"]
            exp_s = f"{exp:.6g}" if isinstance(exp, float) else str(exp)
            out.append(
                f"- {'PASS' if r['ok'] else 'FAIL'} `{r['label']}`: shown {r['shown']} vs source {exp_s} ({r['src']})"
            )
        out.append("")
    out.append("## Assertions")
    for r in RESULTS:
        if r["kind"] == "ASSERT":
            out.append(f"- {'PASS' if r['ok'] else 'FAIL'} [{r['file']}] {r['label']} ({r['src']})")
    out.append("")
    passed = not failed and not unclaimed
    out.append(f"Result: {'PASS' if passed else 'FAIL'}")
    text_out = "\n".join(out) + "\n"
    sys.stdout.write(text_out)
    if REPORT_PATH is not None:
        REPORT_PATH.write_text(text_out)
    return 0 if passed else 1


REPORT_PATH: Path | None = None


def main() -> int:
    global REPORT_PATH
    ap = argparse.ArgumentParser(
        description="Audit numbers in the v2 paper sections against their sources."
    )
    ap.add_argument("--report", type=Path, default=None, help="also write the output to this path")
    ap.add_argument(
        "--committed-only",
        action="store_true",
        help="run only the checks that need committed documents (skip legacy checks that read data/)",
    )
    args = ap.parse_args()
    global DATA_PRESENT
    if args.committed_only:
        DATA_PRESENT = False
    REPORT_PATH = args.report
    return run()


if __name__ == "__main__":
    sys.exit(main())
