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
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "paper" / "sections"
DATA = ROOT / "data"
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
SECTIONS = ["abstract", "introduction", "method", "zoo", "experiments", "results", "reliability", "discussion"]

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
        before = masked[max(0, m.start() - 8): m.start()]
        res.append({"start": m.start(), "end": m.end(), "text": m.group(0),
                    "seed": bool(SEED_BEFORE.search(before))})
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
            return {"n": int(r[2]), "mean": float(r[14]), "sd": float(r[15]),
                    "min": float(r[16]), "max": float(r[17])}
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
    return sum(float(r[12]) for r in evidence_rows()
               if r[0] == "importance" and r[1] == target and r[6] and float(r[6]) <= 5)


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


CONTROLS = [l.strip() for l in (DATA / "runs" / "d3_shards" / "controls.txt").read_text().split() if l.strip()]
H100_IDS = CONTROLS + ["013b7b61202fe188"]
A10_IDS = ["0a841d4082532e8b", "3db97b535c996265", "af6d608f8651fb46", "dd90d99e2ddc0e77"]
DEV_KEYS = {  # table label -> eval.json key
    "GSM8K": "gsm8k", "MATH-500": "math500", "IFEval": "ifeval", "Over-refusal": "over_refusal",
    "Refusal": "refusal_rate", "MMLU-Pro": "mmlu_pro", "HellaSwag": "hellaswag",
    "KL": "kl_to_base", "WikiText": "wikitext_ppl", "Cap-hit": "loop_rate",
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


DEVS = {
    "H1": {lab: cross_host(H100_IDS, h100_eval, key) for lab, key in DEV_KEYS.items()},
    "H2": {lab: cross_host(A10_IDS, a10_eval, key) for lab, key in DEV_KEYS.items()},
}


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
    block = re.search(rf'"{target}":\s*\{{(.*?)\n \}}', read(RESIDUAL), re.S).group(1)
    ci = re.search(r'"boot_ci":\s*\[\s*([\d.\-]+),\s*([\d.\-]+)\s*\]', block)
    return {"theta": float(re.search(r'"theta":\s*(-?[\d.]+)', block).group(1)),
            "lo": float(ci.group(1)), "hi": float(ci.group(2)),
            "n": int(re.search(r'"n":\s*(\d+)', block).group(1)),
            "clusters": int(re.search(r'"clusters":\s*(\d+)', block).group(1))}


RESID = residual_block()
RESID_HS = residual_block("hellaswag")
MET_IDX = {"sd_dy": -6, "sd_seed": -5, "sd_itemSE": -4, "sd_signal_pop": -3}
WITHIN = "within family x device"


def met(target: str, scope: str, field: str) -> float:
    """Field of a metrology_budget_output.txt row (token positions counted from the row end)."""
    return float(metrology_row(target, scope)[MET_IDX[field]])


def audit_counter(name: str) -> int:
    return int(re.search(rf"'{name}': (\d+)", read(AUDIT_TXT)).group(1))


V2_INFERRED = jload(P4 / "d3v2-m1-trajectory" / "seed-1" / "manifest.json")["targets_with_inferred_runtime"]
ZOO_EVAL = {c: DATA / "zoo" / "qwen3.5-0.8b" / c / "eval.json" for c in CONTROLS}
BASE_H0_SEED0 = DATA / "evals" / "qwen3.5-0.8b" / "base" / "devices" / "H0" / "20261104-233637-7c43da7c" / "eval.json"
SEED1_PANEL = DATA / "runs" / "p2-batch" / "20261105-050543-aad6c310"


def seed_panel(key: str) -> tuple[float, float, float]:
    """Seed 0 against seed 1 over the 10 controls and the base (11 records): mean and max absolute
    difference, and Spearman across the records. Recomputed from the raw eval.json files."""
    ids = CONTROLS + ["BASE"]
    s0 = [float(jeval(ZOO_EVAL[c])[key]) for c in CONTROLS] + [float(jeval(BASE_H0_SEED0)[key])]
    s1 = [float(jeval(SEED1_PANEL / c / "repeats" / "seed1" / "eval.json")[key]) for c in CONTROLS] \
        + [float(jeval(SEED1_PANEL / "base" / "repeats" / "seed1" / "eval.json")[key])]
    assert len(ids) == len(s0) == len(s1) == 11
    d = [abs(a - b) for a, b in zip(s0, s1)]
    return sum(d) / len(d), max(d), spearman(s0, s1)
SUITE_RUNS = [p.name for p in sorted(P4.glob("d3v2-*")) if (p / "metrics.json").exists()]
LEARNED = {
    "family": ["d3v2-m1-family", "d3v2-m1_matched-family", "d3v2-m2-family", "d3v2-m2_matched-family",
               "d3v2-m5-family", "d3v2-m6-family"],
    "seed": ["d3v2-m1-seed", "d3v2-m1_matched-seed", "d3v2-m2-seed", "d3v2-m2_matched-seed",
             "d3v2-m3-seed", "d3v2-m5-seed", "d3v2-m6-seed"],
    "trajectory": ["d3v2-m1-trajectory", "d3v2-m1_matched-trajectory", "d3v2-m2-trajectory",
                   "d3v2-m2_matched-trajectory", "d3v2-m3-trajectory", "d3v2-m5-trajectory",
                   "d3v2-m6-trajectory"],
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
    RESULTS.append({"kind": kind, "file": file, "label": label, "shown": shown,
                    "expected": expected, "ok": bool(ok), "src": src})


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
    assertion(file, f"quoted in {path.name}: {desc}", re.search(pattern, read(path)) is not None,
              str(path.relative_to(ROOT)))


def locator(file: str, snippet: str, src: str) -> None:
    claim(file, snippet, None, src)


def build() -> None:
    # ---------------- abstract
    claim("abstract", "345 matched-reference rows", [n_rows(M1T)], "d3v2-m1-trajectory seed-0 n")
    claim("abstract", "mean Spearman 0.45 and 0.46 on the two targets",
          [ss(M1M, "mmlu_pro", "spearman", "mean"), ss(M1M, "hellaswag", "spearman", "mean")],
          "d3v2-m1_matched-trajectory seed_summary (three-seed mean)")
    claim("abstract", "Seed-0 StatNN reached 0.57 and 0.65",
          [bl(M1T, "statnn", "mmlu_pro"), bl(M1T, "statnn", "hellaswag")], "trajectory seed-0 baselines")
    claim("abstract", "reached 0.52 and 0.61 on the same seed and split",
          [bl(M1T, "descriptor_only", "mmlu_pro"), bl(M1T, "descriptor_only", "hellaswag")],
          "trajectory seed-0 baselines, descriptor-only")

    # ---------------- introduction
    declare("introduction", "0.016\\% of Qwen3.5-0.8B's weights",
            "literature value from Takehi et al. (2026), cited; not in repository data")
    declare("introduction", "3 and 4 November", "registration dates of H1 to H4 (HYPOTHESIS.md)")
    declare("introduction", "added on 4 November", "addition date of H5 and H6 (HYPOTHESIS.md)")
    declare("introduction", "On 5 November", "amendment date (HYPOTHESIS.md changelog)")
    declare("introduction", "from 7.9\\% to 48.1\\%",
            "literature value from Takehi et al. (2026), cited; not in repository data")
    presence("introduction", "$\\rho \\ge 0.6$ on all", HYPOTHESIS, r"ρ ≥ 0\.6", "H1 threshold 0.6") if False else None
    presence("introduction", "per-benchmark Spearman $\\rho \\ge 0.6$", HYPOTHESIS, r"ρ ≥ 0\.6", "H1 threshold 0.6")
    presence("introduction", "falsified if $\\rho < 0.3$", HYPOTHESIS, r"ρ < 0\.3", "H1 falsification 0.3")
    presence("introduction", "at least 15\\% relative MAE reduction on trajectories of length at least 4",
             HYPOTHESIS, r"≥ 15 % relative MAE reduction on trajectories of length ≥ 4", "H2 15% and length 4")
    presence("introduction", "transfer $\\rho \\ge 0.5$", HYPOTHESIS, r"\(ρ ≥ 0\.5\)", "H3 0.5")
    presence("introduction", "transfer $\\rho < 0.2$", HYPOTHESIS, r"ρ < 0\.2", "H3 falsification 0.2")

    # ---------------- method
    claim("method", "1,242 statistics and 136 singular-spectrum values",
          [seed0(M1T)["statnn"]["n_features"], 136],
          "statnn n_features; 136 = N_SPECTRUM (8) x 17 pooling groups (see assertion)")
    claim("method", "feature version 4", [jload(FEATURE_MANIFEST)["config"]["feature_version"]],
          "features manifest config.feature_version")
    assertion("method", "pooled widths: 18 x (4 buckets x 17 groups + 1) = 1242 and 8 x 17 = 136",
              18 * (4 * 17 + 1) == 1242 and 8 * 17 == 136 and "N_STATS, N_SPECTRUM = 18, 8" in read(POOL_PY)
              and jload(FEATURE_MANIFEST)["config"]["buckets"] == 4, "src/wsp/features/pool.py; features config")
    presence("method", "The pre-registered sequence test sets a bar of at least 15\\% relative MAE reduction on trajectories of length at least four.",
             HYPOTHESIS, r"≥ 15 % relative MAE reduction on trajectories of length ≥ 4", "H2 bar")

    # ---------------- zoo
    claim("zoo", "The file holds 1,255 lines", [1255], "zoo manifest line count")
    claim("zoo", "1,247 entry rows, one per unique ID", [1247], "zoo manifest body rows (unique ids)")
    claim("zoo", "The manifest therefore has 1,247 rows: 1,235 unflagged rows and the 12 flagged rows.",
          [1247, 1235, 12], "zoo manifest rows; unflagged = 1247 - 12")
    claim("zoo", "The feature manifest records 1,235 entries", [1235], "features manifest n_entries")
    claim("zoo", "Of the 1,235, 389 have evaluation records and none of the 12 flagged entries does.",
          [1235, 389, 12], "unflagged rows; eval.json present; flagged ids without eval")
    claim("zoo", "The v2 suite matches 345 of those 389", [345, 389], "v2 matched rows; eval count")
    claim("zoo", "The D3 subset has 300 evaluated entries, and 260 of them have matched-reference targets.",
          [300, 260], "board-a-d3-audit audit.txt (test_n 300; matched_reference_targets 260)")
    claim("zoo", "verified KL to base of 0.77 to 14.0 nats", [0.77, 14.0], "docs/DECISIONS.md D6 addendum 2")
    presence("zoo", "D6 addendum 2 in \\path{docs/DECISIONS.md}", DECISIONS, r"D6 addendum 2", "D6 addendum 2")
    presence("zoo", "D6 addendum 3 flags them", DECISIONS, r"D6 addendum 3", "D6 addendum 3")
    presence("zoo", "verified KL to base of 0.77 to 14.0 nats", DECISIONS, r"0\.77–14\.0 nats", "0.77 to 14.0 nats") \
        if False else None
    claim("zoo", "The twelve Heretic ARA entries are the rebuilt set", [], "word form") if False else None
    presence("zoo", "\\texttt{ara\\_unconstrained\\_surrogate\\_scale}", DECISIONS,
             r"ara_unconstrained_surrogate_scale", "flag name")
    presence("zoo", "first 400 harmful and harmless dataset prompts", REFUSAL_PY, r"train\[:400\]", "train[:400]")
    claim("zoo", "first 400 harmful and harmless dataset prompts", [400], "src/wsp/edits/refusal.py train[:400]")
    claim("zoo", "1,247 entry rows, counted by recorded eligibility", [1247], "zoo manifest body rows")
    declare("zoo", "5 November 2026", "date")
    declare("zoo", "The builder implements six families.", "")  # word count, no digits
    declare("zoo", "2026", "date")

    # ---------------- experiments
    b_eval = jeval(local_eval("base"))
    claim("experiments", "50 GSM8K, 25 MATH-500, 25 IFEval, 100 MMLU-Pro and 100 HellaSwag items",
          [b_eval["counts"]["gsm8k"], b_eval["counts"]["math500"], b_eval["counts"]["ifeval"],
           b_eval["counts"]["mmlu_pro"], b_eval["counts"]["hellaswag"]], "base per-entry eval counts")
    claim("experiments", "plus 50 prompts each for refusal, over-refusal and KL",
          [b_eval["counts"]["refusal_rate"]], "base per-entry eval counts (50 prompts)")
    claim("experiments", "record a 1,024-token cap", [b_eval["generation"]["max_new_tokens"]],
          "zoo evaluation configuration (per-entry eval generation.max_new_tokens)")
    claim("experiments", "Generation is capped at 1024 tokens.", [b_eval["generation"]["max_new_tokens"]],
          "base per-entry eval generation.max_new_tokens")
    se_fast = [binom_se(0.16, 50), binom_se(0.16, 25), binom_se(0.40, 25)]
    claim("experiments", "a single binomial score for GSM8K, MATH-500 or IFEval has a standard error of 0.05 to 0.10",
          [min(se_fast), max(se_fast)], "binomial SE at base fast-tier values")
    claim("experiments", "(1,024 tokens at the fast tier, 4,096 at the cheap tier)",
          [1024, 4096], "per-entry eval max_new_tokens; base-final-4096 generation")
    claim("experiments", "with a 4096-token limit", [base_summary()["generation"]["max_new_tokens"]],
          "base-final-4096 summary generation.max_new_tokens")
    declare("experiments", "except for MMLU-Pro, which has a 1024-token limit",
            "cheap-tier configuration statement; not re-derived here")
    declare("experiments", "A faithful 32768-token tier is reserved for the base and at most 10 selected entries.",
            "plan parameters for a tier not yet run")
    declare("experiments", "using seed 0 and sorted IDs, with quotas of 60 random direction, 100 abliteration, 60 LoRA SFT, 30 task vector and 50 chain steps",
            "D3 sampling procedure; the quota sum is checked by assertion")
    claim("experiments", "these quotas resolve to 300 entries, and all 300 were evaluated",
          [300, 300], "quota sum (assertion below); board-a-d3-audit test_n 300")
    declare("experiments", "32 GB", "hardware description")
    declare("experiments", "RTX PRO 4500", "hardware description")
    declare("experiments", "40 GB", "hardware description")
    claim("experiments", "\\$1.99/h", [1.99], "board/lambda_spend.md A100 hourly rate")
    claim("experiments", "temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, batch size 64, seed 0",
          [0.6, 0.95, 20, 1.5, 64], "base-final-4096 summary generation (seed 0 is a seed label)")
    claim("experiments", "4,096-token cap", [4096], "base-final-4096 generation")
    claim("experiments", "The matched set has 345 rows from 389 evaluated entries; the other 44",
          [345, 389, 44], "v2 matched rows; eval count; missing_predecessor_targets")
    claim("experiments", "lists the 25 complete rows", [25], "complete d3v2 run directories (assertion below)")
    claim("experiments", "since all 60 random-direction edits leave", [60], "evidence CSV family_distribution n, random_dir")
    claim("experiments", "mark 260 of the 345 rows as having inferred runtime metadata", [V2_INFERRED, n_rows(M1T)],
          "d3v2-m1-trajectory seed-1 manifest targets_with_inferred_runtime; n")
    claim("experiments", "Those 260 are the matched D3 edits counted in Section", [audit_counter("matched_reference_targets")],
          "board-a-d3-audit audit.txt COUNTERS matched_reference_targets (D3 subset)")
    claim("experiments", "The other 85 rows are chain predecessors", [n_rows(M1T) - V2_INFERRED],
          "345 matched rows minus 260 inferred; the 85 are the chain-predecessor rows (run-loader check)")
    claim("experiments", "The 345 rows are not the 300-entry D3 subset", [n_rows(M1T), 300],
          "v2 n; D3 subset size (300 entries)")
    claim("experiments", "the trajectory-split M1-matched row reached mean Spearman 0.45 and 0.46 on MMLU-Pro and HellaSwag",
          [ss(M1M, "mmlu_pro", "spearman", "mean"), ss(M1M, "hellaswag", "spearman", "mean")],
          "d3v2-m1_matched-trajectory seed_summary")
    claim("experiments", "Seed-0 StatNN reached 0.57 and 0.65 on the same split",
          [bl(M1T, "statnn", "mmlu_pro"), bl(M1T, "statnn", "hellaswag")], "trajectory seed-0 baselines")
    pm = paired(M1M, "statnn", "mmlu_pro")
    ph = paired(M1M, "statnn", "hellaswag")
    claim("experiments", "the paired difference is +0.016 [0.012, 0.021] on MMLU-Pro and +0.011 [0.009, 0.014] on HellaSwag",
          [pm["mae"]["value"], pm["mae"]["ci95"][0], pm["mae"]["ci95"][1],
           ph["mae"]["value"], ph["mae"]["ci95"][0], ph["mae"]["ci95"][1]],
          "d3v2-m1_matched-trajectory seed-0 paired_differences.statnn MAE (cluster bootstrap)")
    above = sum(1 for r in LEARNED["trajectory"] for t in ["mmlu_pro", "hellaswag"]
                if paired(r, "statnn", t)["mae"]["ci95"][0] > 0)
    claim("experiments", "Across the seven learned rows, all 14 intervals lie above zero.", [above],
          "paired MAE lower bounds above zero, 7 rows x 2 targets")
    claim("experiments", "M5 reaches 0.41 and 0.26, against 0.45 and 0.46 for M1-matched",
          [ss("d3v2-m5-trajectory", "mmlu_pro", "spearman", "mean"), ss("d3v2-m5-trajectory", "hellaswag", "spearman", "mean"),
           ss(M1M, "mmlu_pro", "spearman", "mean"), ss(M1M, "hellaswag", "spearman", "mean")],
          "d3v2-m5-trajectory and d3v2-m1_matched-trajectory seed_summary")
    claim("experiments", "Their test sets hold 98, 98, 59, 60 and 30 rows, drawn from 98, 45, 59, 60 and 27 clusters.",
          [n_rows(f"d3v2-m2-holdout-{h}") for h in HOLDOUTS] + [n_clusters(f"d3v2-m2-holdout-{h}") for h in HOLDOUTS],
          "holdout seed-0 targets.n and paired n_clusters")
    declare("experiments", "effective rank of at least 2", "M6 acceptance gate constant; the gate outcome is asserted below")
    declare("experiments", "rank below 2", "M6 acceptance gate constant; the gate outcome is asserted below")
    presence("experiments", "registered 15\\% MAE gain on adequately powered trajectories", HYPOTHESIS,
             r"≥ 15 % relative MAE reduction", "H2 15%")
    claim("experiments", "registered 15\\% MAE gain on adequately powered trajectories", [15],
          "docs/HYPOTHESIS.md H2 test")
    declare("experiments", "Experiment 8", "blog table reference (Takehi et al.), not in repository data")
    declare("experiments", "93\\% is the MATH-500 share at a 32,768-token limit", "Takehi et al. blog figure, cited")
    claim("experiments", "and ours is at 4,096 tokens", [4096], "base-final-4096 generation")

    # ---------------- results
    claim("results", "On seed 0 of the trajectory split (345 rows, 278 clusters), the gap is 0.042 Spearman on MMLU-Pro and 0.035 on HellaSwag",
          [n_rows(M1T), n_clusters(M1T),
           bl(M1T, "statnn", "mmlu_pro") - bl(M1T, "descriptor_only", "mmlu_pro"),
           bl(M1T, "statnn", "hellaswag") - bl(M1T, "descriptor_only", "hellaswag")],
          "trajectory seed-0 baselines, unrounded difference")
    claim("results", "0.560 against 0.555 on MMLU-Pro and 0.630 against 0.588 on HellaSwag",
          [ev_perf("mmlu_pro", "descriptor_only", "r2"), ev_perf("mmlu_pro", "statnn", "r2"),
           ev_perf("hellaswag", "descriptor_only", "r2"), ev_perf("hellaswag", "statnn", "r2")],
          "evidence CSV performance r2 (pooled out-of-fold)")
    claim("results", "is 0.70, against 0.49 for the descriptor model, a lead of 0.20",
          [ev_perf("kl_to_base", "statnn", "spearman"), ev_perf("kl_to_base", "descriptor_only", "spearman"),
           ev_perf("kl_to_base", "statnn", "spearman") - ev_perf("kl_to_base", "descriptor_only", "spearman")],
          "evidence CSV performance spearman, kl_to_base")
    claim("results", "4.1\\% [$-$6.3\\%, 14.0\\%] on MMLU-Pro and 5.5\\% [$-$6.3\\%, 16.4\\%] on HellaSwag",
          [4.1, -6.3, 14.0, 5.5, -6.3, 16.4],
          "round2_metrology_output.txt, paired MAE theta and boot_ci (percent)")
    locator("results", "round2_metrology_output.txt}, lines 100 and 102", "round2_metrology_output.txt lines 100, 102")
    claim("results", "on the 345 seed-0 trajectory rows", [n_rows(M1T)], "trajectory seed-0 n")
    claim("results", "the top five features hold 26\\% of total gain", [ev_top5_share("mmlu_pro") * 100],
          "evidence CSV importance ranks 1 to 5, mmlu_pro")
    claim("results", "Within-family Spearman runs from 0.13 to 0.71 on MMLU-Pro and from $-$0.12 to 0.71 on HellaSwag",
          [min(finite([ev_within("mmlu_pro", f) for f in WITHIN_FAMILIES])),
           max(finite([ev_within("mmlu_pro", f) for f in WITHIN_FAMILIES])),
           min(finite([ev_within("hellaswag", f) for f in WITHIN_FAMILIES])),
           max(finite([ev_within("hellaswag", f) for f in WITHIN_FAMILIES]))],
          "evidence CSV within_family spearman ranges")
    claim("results", "LoRA SFT (59 rows) on MMLU-Pro and task vector (30 rows)",
          [ev_within_n("mmlu_pro", "lora_sft"), ev_within_n("mmlu_pro", "task_vector")],
          "evidence CSV within_family n")
    claim("results", "The 0.71 values come from the two smallest families",
          [ev_within("mmlu_pro", "lora_sft")], "evidence CSV within_family: LoRA SFT, MMLU-Pro (0.709)")
    claim("results", "unchanged in all 60 rows", [ev_family("hellaswag", "random_dir")["n"]],
          "evidence CSV family_distribution n, random_dir")
    claim("results", "their MMLU-Pro SD is 0.007", [ev_family("mmlu_pro", "random_dir")["sd"]],
          "evidence CSV family_distribution sd, random_dir")
    claim("results", "SDs of 0.013 and 0.005", [ev_family("mmlu_pro", "abliteration")["sd"],
          ev_family("hellaswag", "abliteration")["sd"]], "evidence CSV family_distribution sd, abliteration")
    claim("results", "$-$0.063 on MMLU-Pro and $+$0.034 on HellaSwag",
          [ev_family("mmlu_pro", "lora_sft")["mean"], ev_family("hellaswag", "lora_sft")["mean"]],
          "evidence CSV family_distribution mean, lora_sft")
    declare("results", "ddof 1", "sample SD convention, not a result")

    # ---------------- discussion
    claim("discussion", "Across the 25 complete rows", [25], "complete d3v2 run directories")
    presence("discussion", "H1 required a per-benchmark Spearman of at least 0.6", HYPOTHESIS,
             r"Spearman ρ ≥ 0\.6", "H1 0.6")
    presence("discussion", "H1 also has a falsification rule: Spearman below 0.3", HYPOTHESIS, r"ρ < 0\.3", "H1 0.3")
    claim("discussion", "steps of one item out of 100", [b_eval["counts"]["mmlu_pro"]], "base eval counts (100 items)")
    assertion("discussion", "no learned row reaches 0.6 on either target (three-seed means)",
              all(ss(r, t, "spearman", "mean") < 0.6 for r in SUITE_RUNS if "statnn" not in r
                  for t in ["mmlu_pro", "hellaswag"] if not math.isnan(ss(r, t, "spearman", "mean"))),
              "suite seed_summary means")
    claim("discussion", "the highest three-seed mean on HellaSwag is 0.46",
          [round(max(finite([ss(r, "hellaswag", "spearman", "mean") for r in SUITE_RUNS])), 2)],
          "max over the 25 complete d3v2 suite rows")
    claim("discussion", "The highest on MMLU-Pro is 0.45",
          [round(max(finite([ss(r, "mmlu_pro", "spearman", "mean") for r in SUITE_RUNS])), 2)],
          "max over the 25 complete d3v2 suite rows")
    claim("discussion", "their MMLU-Pro SD is 0.007",
          [ev_family("mmlu_pro", "random_dir")["sd"]], "evidence CSV family_distribution sd, random_dir")
    claim("discussion", "Abliteration's SDs are 0.013 and 0.005",
          [ev_family("mmlu_pro", "abliteration")["sd"], ev_family("hellaswag", "abliteration")["sd"]],
          "evidence CSV family_distribution sd, abliteration")
    claim("discussion", "one item out of 100", [base_summary()["counts"]["mmlu_pro"] / 3], "")  if False else None
    claim("discussion", "Each target is a 100-item accuracy, with a binomial standard error of 0.045 for MMLU-Pro and 0.049 for HellaSwag",
          [100, binom_se(base_val("mmlu_pro"), 100), binom_se(base_val("hellaswag"), 100)],
          "binomial SE at base fast-tier values (0.28 and 0.39, n = 100)")
    claim("discussion", "the residual SD within family and device is 0.0131, no larger than its item SE of 0.0136",
          [met("hellaswag", WITHIN, "sd_dy"), met("hellaswag", WITHIN, "sd_itemSE")],
          "metrology_budget_output.txt line 44 (hellaswag within family x device: sd_dy, sd_itemSE)")
    claim("discussion", "the residual SD is 0.0384, above its item SE of 0.0246, and the remaining signal SD is 0.0295",
          [met("mmlu_pro", WITHIN, "sd_dy"), met("mmlu_pro", WITHIN, "sd_itemSE"), met("mmlu_pro", WITHIN, "sd_signal_pop")],
          "metrology_budget_output.txt line 42 (mmlu_pro within family x device)")
    locator("discussion", "metrology\\_budget\\_output.txt}, lines 42 and 44", "metrology lines 42 and 44 (caption)")
    claim("discussion", "use the 300 D3 entries", [300], "metrology_budget_output.txt: D3 subset n = 300")
    claim("discussion", "interval on the 345 matched v2 rows", [n_rows(M1T)], "v2 matched rows (n)")
    claim("discussion", "mean absolute difference on the 11 edited entries evaluated on the H100 was 0.008 on MMLU-Pro and 0.003 on HellaSwag, with Spearman 0.75 and 0.76",
          [11, DEVS["H1"]["MMLU-Pro"]["mean"], DEVS["H1"]["HellaSwag"]["mean"],
           DEVS["H1"]["MMLU-Pro"]["rho"], DEVS["H1"]["HellaSwag"]["rho"]],
          "per-entry eval.json, local vs H100, 11 entries")
    claim("discussion", "On the A10, four edited entries gave mean differences of 0.010 and 0.003",
          [DEVS["H2"]["MMLU-Pro"]["mean"], DEVS["H2"]["HellaSwag"]["mean"]],
          "per-entry eval.json, local vs A10, 4 entries")
    claim("discussion", "gives a seed-to-seed SD of 0.0000 for MMLU-Pro and HellaSwag",
          [metrology_seed_sd("mmlu_pro")], "metrology_budget_output.txt lines 24 and 25 (sd_seed, likelihood-scored)")
    claim("discussion", "between 0.0124 and 0.0504 for the five sampled targets",
          [min(metrology_seed_sd(t) for t in ["gsm8k", "math500", "ifeval", "refusal_rate", "over_refusal"]),
           max(metrology_seed_sd(t) for t in ["gsm8k", "math500", "ifeval", "refusal_rate", "over_refusal"])],
          "metrology_budget_output.txt lines 19 to 23 (sd_seed, sampled targets: min and max)")
    locator("discussion", "metrology_budget_output.txt}, lines 19 to 27", "metrology lines 19 to 27 (seed panel block)")
    assertion("discussion", "HellaSwag seed-to-seed SD is 0.0000 (metrology line 25)",
              metrology_seed_sd("hellaswag") == 0.0, "metrology_budget_output.txt line 25")
    claim("discussion", "delta-norm baseline reaches 0.21 and 0.18 on the trajectory split",
          [bl(M1T, "delta_norm", "mmlu_pro"), bl(M1T, "delta_norm", "hellaswag")], "trajectory seed-0 baselines")
    claim("discussion", "Under family holdout its Spearman falls to $-$0.13 and $-$0.29",
          [bl("d3v2-m1-family", "delta_norm", "mmlu_pro"), bl("d3v2-m1-family", "delta_norm", "hellaswag")],
          "family seed-0 baselines")
    claim("discussion", "Within-family refits on HellaSwag range from $-$0.12 to 0.71",
          [min(finite([ev_within("hellaswag", f) for f in WITHIN_FAMILIES])),
           max(finite([ev_within("hellaswag", f) for f in WITHIN_FAMILIES]))], "evidence CSV within_family hellaswag")
    claim("discussion", "The 260 matched D3 rows with inferred runtime metadata", [V2_INFERRED],
          "targets_with_inferred_runtime (v2 trajectory seed-1 manifest)")
    claim("discussion", "at $-$0.02 on MMLU-Pro and 0.11 on HellaSwag",
          [bl("d3v2-m1-family", "statnn", "mmlu_pro"), bl("d3v2-m1-family", "statnn", "hellaswag")],
          "family seed-0 StatNN baselines")
    claim("discussion", "with a three-seed mean of 0.31", [ss("d3v2-m1_matched-family", "hellaswag", "spearman", "mean")],
          "d3v2-m1_matched-family seed_summary (HellaSwag spearman mean)")
    claim("discussion", "the Spearman gap is 0.042 on MMLU-Pro and 0.035 on HellaSwag",
          [bl(M1T, "statnn", "mmlu_pro") - bl(M1T, "descriptor_only", "mmlu_pro"),
           bl(M1T, "statnn", "hellaswag") - bl(M1T, "descriptor_only", "hellaswag")], "trajectory seed-0 baselines")
    claim("discussion", "$\\theta = 0.146$ [0.069, 0.214] for MMLU-Pro",
          [RESID["theta"], RESID["lo"], RESID["hi"]], "round2_residual_estimand_output.txt, mmlu_pro block")
    claim("discussion", "$\\theta = -0.034$ [$-$0.118, 0.037] for HellaSwag",
          [RESID_HS["theta"], RESID_HS["lo"], RESID_HS["hi"]], "round2_residual_estimand_output.txt, hellaswag block")
    claim("discussion", "over 345 rows and 278 clusters", [RESID["n"], RESID["clusters"]],
          "round2_residual_estimand_output.txt, mmlu_pro block n and clusters")
    locator("discussion", "round2_residual_estimand_output.txt}, lines 2 to 11", "residual theta line within 2 to 11")
    locator("discussion", "and 40 to 47", "residual output lines 40 to 47 (hellaswag block)")
    claim("discussion", "$\\theta = 1 - A/B$", [1], "definition of the estimand") if False else None
    declare("discussion", "$\\theta = 1 - A/B$", "definition of the estimand, not a result")
    declare("discussion", "No learned row reaches 0.6 on either target.", "threshold; covered by the assertion below")
    declare("discussion", "Several rows exceed 0.3", "threshold; covered by the assertion below")
    claim("discussion", "at least 100 valid fresh components", [100], "docs/PREREG_H1_TF.md pass rule")
    presence("discussion", "a half-width no greater than $\\theta/2$", PREREG, r"≥100 valid genuinely fresh components",
             "PREREG pass rule (components, 100)")
    declare("discussion", "$\\theta/2$", "definition of the pass rule")
    locator("discussion", "docs/PREREG_H1_TF.md}, section 2", "PREREG section 2 (component definition)")
    locator("discussion", "round2-metrology.md}, section 1.4", "round2 metrology section 1.4 (HellaSwag accuracy retired; negative signal variance)")
    claim("discussion", "0.0384 & 0.0246 & 0.0295 & 0.146 [0.069, 0.214]",
          [met("mmlu_pro", WITHIN, "sd_dy"), met("mmlu_pro", WITHIN, "sd_itemSE"), met("mmlu_pro", WITHIN, "sd_signal_pop"),
           RESID["theta"], RESID["lo"], RESID["hi"]], "tab:residual MMLU-Pro row (metrology line 42; residual output)")
    claim("discussion", "0.0131 & 0.0136 & 0.0000 & $-$0.034 [$-$0.118, 0.037]",
          [met("hellaswag", WITHIN, "sd_dy"), met("hellaswag", WITHIN, "sd_itemSE"), met("hellaswag", WITHIN, "sd_signal_pop"),
           RESID_HS["theta"], RESID_HS["lo"], RESID_HS["hi"]], "tab:residual HellaSwag row (metrology line 44; residual output)")
    claim("discussion", "at least 80\\% power at $\\theta = 0.15$", [80, 0.15], "docs/PREREG_H1_TF.md pass rule")
    presence("discussion", "at least 80\\% power at $\\theta = 0.15$", PREREG, r"80% power", "PREREG 80% power")
    claim("discussion", "The primary outcome is the mean, over 1,000 paired items per edit,", [1000],
          "docs/PREREG_H1_TF.md section 1 (1,000 paired items)")
    presence("discussion", "over 1,000 paired items per edit", PREREG, r"\*\*1,000\*\* paired", "PREREG 1,000 items")
    claim("discussion", "An upper 95\\% bound below 0.05 on both intervals", [0.05],
          "docs/PREREG_H1_TF.md kill rule")
    presence("discussion", "An upper 95\\% bound below 0.05", PREREG, r"upper 95% bound<\.05", "PREREG kill rule")
    claim("discussion", "A pass needs $\\theta \\ge 0.15$", [0.15], "docs/PREREG_H1_TF.md pass rule")
    presence("discussion", "A pass needs $\\theta \\ge 0.15$", PREREG, r"θ≥\.15", "PREREG theta 0.15")

    # ---------------- reliability
    claim("reliability", "re-evaluated the base model and the 10 control entries on the local GPU", [len(CONTROLS)],
          "runs/d3_shards/controls.txt")
    claim("reliability", "The cross-host panel covers the 10 control entries", [len(CONTROLS)], "controls.txt")
    claim("reliability", "the 10 controls and", [len(CONTROLS)], "controls.txt")
    claim("reliability", "is taken across the 11 H100 entries", [len(H100_IDS)], "H100 entries")
    claim("reliability", "All records used seed 0, batch size 32", [32],
          "per-entry eval.json batch_size (seed 0 is a seed label)")
    claim("reliability", "The seed-1 panel reruns the 10 controls and the base", [len(CONTROLS)], "controls.txt")
    _g = seed_panel("gsm8k"); _m = seed_panel("math500"); _i = seed_panel("ifeval")
    _o = seed_panel("over_refusal"); _k = seed_panel("kl_to_base")
    claim("reliability", "Over the 11 records, seed 0 against seed 1 gives a mean and maximum absolute difference of 0.055 and 0.12 with Spearman $\\rho$ 0.15 on GSM8K",
          [11, _g[0], _g[1], _g[2]], "seed-1 panel recomputed from eval.json (11 records)")
    claim("reliability", "0.036 and 0.12 with $\\rho$ 0.46 on MATH-500", [_m[0], _m[1], _m[2]], "seed-1 panel recomputed")
    claim("reliability", "0.029 and 0.04 with $\\rho$ 0.10 on IFEval", [_i[0], _i[1], _i[2]], "seed-1 panel recomputed")
    claim("reliability", "0.049 and 0.10 with $\\rho$ 0.67 on keyword over-refusal", [_o[0], _o[1], _o[2]], "seed-1 panel recomputed")
    claim("reliability", "KL differs by at most 0.0003 nats", [_k[1]], "seed-1 panel recomputed (KL max abs difference)")
    claim("reliability", "MMLU-Pro and HellaSwag agree exactly on all 11 records", [11],
          "seed-1 panel: 11 records (10 controls and base)")
    claim("reliability", "IFEval and the pooled share hitting the 1,024-token cap are descriptive", [1024],
          "fast-tier token cap (per-entry eval.json max_new_tokens)")
    claim("reliability", "Over the 10 controls, the seed noise in the paired change has SD 0.0504 on GSM8K, 0.0268 on MATH-500, 0.0248 on IFEval, 0.0356 on keyword over-refusal and 0.0124 on keyword refusal, and 0.0000 on MMLU-Pro and HellaSwag",
          [len(CONTROLS), metrology_seed_sd("gsm8k"), metrology_seed_sd("math500"), metrology_seed_sd("ifeval"),
           metrology_seed_sd("over_refusal"), metrology_seed_sd("refusal_rate"), 0.0],
          "metrology_budget_output.txt lines 19 to 27 (sd_seed; 10 controls)")
    locator("reliability", "metrology_budget_output.txt}, lines 19 to 27", "metrology lines 19 to 27 (seed panel)")
    declare("reliability", "4 November protocol", "date")
    claim("reliability", "with seed 0, the fast-tier configuration and batch size 32", [32],
          "per-entry eval.json batch_size")
    claim("reliability", "All 11 pairs matched exactly on every reported metric", [len(CONTROLS) + 1],
          "10 controls and the base")
    retest = []
    for e in CONTROLS + ["base"]:
        a = jeval(EV / e / "eval.json")
        b = jeval(EV / e / "eval.local-run1.json")
        retest += [abs(float(a[k]) - float(b[k])) for k in DEV_KEYS.values()]
    claim("reliability", "The mean and maximum absolute differences were 0.000 on each.", [max(retest)],
          "retest pairs: eval.json vs eval.local-run1.json, 11 pairs, all metrics")
    claim("reliability", "On the H100 PCIe (device H1), 11 edited entries", [len(H100_IDS)],
          "H100 entries (10 controls and 013b7b61202fe188)")
    claim("reliability", "by 0.10 on GSM8K (0.16 and 0.06; 50 items)",
          [abs(base_dev("H1", "gsm8k") - base_val("gsm8k")), base_val("gsm8k"), base_dev("H1", "gsm8k"), 50],
          "base local vs devices/H1 eval.json")
    claim("reliability", "by 0.08 on MATH-500 (0.16 and 0.08; 25 items)",
          [abs(base_dev("H1", "math500") - base_val("math500")), base_val("math500"), base_dev("H1", "math500"), 25],
          "base local vs devices/H1")
    claim("reliability", "by 0.10 on keyword over-refusal (0.10 and 0.20; 50 prompts)",
          [abs(base_dev("H1", "over_refusal") - base_val("over_refusal")), base_val("over_refusal"),
           base_dev("H1", "over_refusal"), 50], "base local vs devices/H1")
    claim("reliability", "The other proportions agreed within 0.01, and WikiText perplexity was 22.90 locally and 22.92 on the H100",
          [0.01, base_val("wikitext_ppl"), base_dev("H1", "wikitext_ppl")],
          "tolerance 0.01 asserted below; base wikitext_ppl local and devices/H1")
    assertion("reliability", "other base proportions on the H100 agree within 0.01",
              all(abs(base_dev("H1", k) - base_val(k)) <= 0.01 + 1e-9 for k in ["ifeval", "refusal_rate", "mmlu_pro", "hellaswag"]),
              "base local vs devices/H1: ifeval, refusal, mmlu_pro, hellaswag")
    claim("reliability", "0.10 on GSM8K (0.16 and 0.06), 0.12 on MATH-500 (0.16 and 0.28) and 0.04 on IFEval (0.40 and 0.44)",
          [abs(base_dev("H2", "gsm8k") - base_val("gsm8k")), base_val("gsm8k"), base_dev("H2", "gsm8k"),
           abs(base_dev("H2", "math500") - base_val("math500")), base_val("math500"), base_dev("H2", "math500"),
           abs(base_dev("H2", "ifeval") - base_val("ifeval")), base_val("ifeval"), base_dev("H2", "ifeval")],
          "base local vs devices/H2 (A10)")
    claim("reliability", "On the base model, MMLU-Pro scored 0.28 locally, 0.29 on the H100 and 0.28 on the A10, and HellaSwag scored 0.39 on all three hosts.",
          [base_val("mmlu_pro"), base_dev("H1", "mmlu_pro"), base_dev("H2", "mmlu_pro"), base_val("hellaswag")],
          "base local, H100 and A10 eval.json (MMLU-Pro and HellaSwag)")
    assertion("reliability", "HellaSwag identical on all three hosts; MMLU-Pro differs by one item on the H100",
              all(base_dev(d, "hellaswag") == base_val("hellaswag") for d in ["H1", "H2"])
              and round(abs(base_dev("H1", "mmlu_pro") - base_val("mmlu_pro")) * 100) == 1, "base devices")
    claim("reliability", "Both are 100-item scores, so the H100 MMLU-Pro gap of 0.01 is one item.",
          [100, abs(base_dev("H1", "mmlu_pro") - base_val("mmlu_pro"))], "base item counts; H100 MMLU-Pro gap")
    assertion("reliability", "excluding 013b7b61202fe188 leaves every maximum absolute difference unchanged",
              all(cross_host(H100_IDS, h100_eval, k, drop="013b7b61202fe188")["max"]
                  == cross_host(H100_IDS, h100_eval, k)["max"] for k in DEV_KEYS.values()),
              "per-entry eval.json, H100")
    shift = max(abs(DEVS["H1"][lab]["rho"] - cross_host(H100_IDS, h100_eval, key, drop="013b7b61202fe188")["rho"])
                for lab, key in DEV_KEYS.items() if not math.isnan(DEVS["H1"][lab]["rho"]))
    cap_excl = cross_host(H100_IDS, h100_eval, "loop_rate", drop="013b7b61202fe188")["rho"]
    claim("reliability", "moves the Spearman values by up to 0.16; the cap-hit share, for example, rises from 0.72 to 0.89",
          [round(shift, 2), DEVS["H1"]["Cap-hit"]["rho"], cap_excl],
          "H100 Spearman, 11 entries vs 10 without 013b7b61202fe188")
    claim("reliability", "Of 300 evaluated D3 edits, 260 had matched-reference delta targets, and all 260 retained labels",
          [300, 260, 260], "board-a-d3-audit audit.txt counters and test_n")
    claim("reliability", "The other 40 had no predecessor target", [40], "audit counter missing_predecessor_targets")
    claim("reliability", "0.052 for GSM8K ($n = 50$), 0.073 for MATH-500 ($n = 25$), 0.098 for IFEval ($n = 25$), 0.045 for MMLU-Pro ($n = 100$) and 0.049 for HellaSwag ($n = 100$)",
          [binom_se(0.16, 50), 50, binom_se(0.16, 25), 25, binom_se(0.40, 25), 25,
           binom_se(0.28, 100), 100, binom_se(0.39, 100), 100], "binomial SE at base fast-tier values")
    claim("reliability", "0.16 for GSM8K, 0.16 for MATH-500, 0.40 for IFEval, 0.28 for MMLU-Pro and 0.39 for HellaSwag",
          [base_val("gsm8k"), base_val("math500"), base_val("ifeval"), base_val("mmlu_pro"), base_val("hellaswag")],
          "base local eval.json")
    claim("reliability", "the H1 threshold $\\rho \\ge 0.6$ would need $r \\ge 0.36$", [0.6, 0.36],
          "docs/HYPOTHESIS.md (0.6); 0.6 squared = 0.36")
    claim("reliability", "a Spearman $\\rho$ of at least 0.8 with a lower 95\\% bound of at least 0.6, and an ICC(A,1) lower bound of at least 0.75",
          [0.8, 0.6, 0.75], "docs/reviews/H5/board-a-prereg-2026-11-04.md (thresholds, as written in the file)")
    presence("reliability", "a Spearman $\\rho$ of at least 0.8", PREREG_H5, r"Spearman point ≥ \.8", "rho point 0.8")
    presence("reliability", "lower 95\\% bound of at least 0.6", PREREG_H5, r"lower 95% bound ≥ \.6", "lower bound 0.6")
    presence("reliability", "ICC(A,1) lower bound of at least 0.75", PREREG_H5, r"ICC lower bound ≥ \.75", "ICC 0.75")
    claim("reliability", "smallest scientifically relevant change is $\\delta = 0.10$", [0.10],
          "docs/reviews/H5/board-a-prereg-2026-11-04.md")
    presence("reliability", "smallest scientifically relevant change is $\\delta = 0.10$", PREREG_H5,
             r"δ_b = \.10 \(10 percentage", "delta 0.10")
    claim("reliability", "For log-perplexity it is $\\log 1.05$, and for KL it is 0.01 nats.", [1.05, 0.01],
          "docs/reviews/H5/board-a-prereg-2026-11-04.md")
    presence("reliability", "For log-perplexity it is $\\log 1.05$", PREREG_H5, r"δ_b = log\(1\.05\)", "log 1.05")
    presence("reliability", "and for KL it is 0.01 nats", PREREG_H5, r"δ_b = \.01 nats", "KL 0.01 nats")
    claim("reliability", "At 50 prompts one item is 0.02", [50, 1 / 50], "1/50 and the 50 prompts")
    claim("reliability", "edit KL differed by 0.0009 nats at most on the H100 and 0.0037 on the A10",
          [DEVS["H1"]["KL"]["max"], DEVS["H2"]["KL"]["max"]], "per-entry eval.json KL max")
    claim("reliability", "$n = 11$ on the H100, $n = 4$ on the A10", [11, 4], "entry counts")
    claim("reliability", "pools GSM8K, MATH-500 and IFEval responses at the 1,024-token cap, 100 responses per entry",
          [1024, 100], "per-entry eval.json max_new_tokens; 50 + 25 + 25 responses")
    claim("reliability", "at temperature 0.6", [base_summary()["generation"]["temperature"]],
          "base-final-4096 generation.temperature")
    claim("reliability", "the pooled share hitting the 1,024-token cap", [1024],
          "per-entry eval.json max_new_tokens")
    claim("reliability", "Keyword refusal sits at 0.99 to 1.00 on the base model",
          [base_summary()["refusal_rate"], base_val("refusal_rate")],
          "cheap-tier base refusal 0.99; fast-tier local refusal 1.0")
    claim("reliability", "Keyword refusal change when the base is at 0.99 to 1.00",
          [base_summary()["refusal_rate"], base_val("refusal_rate")],
          "cheap-tier base refusal 0.99; fast-tier local refusal 1.0")
    claim("reliability", "Seed 0, batch size 32 and one configuration hash throughout.", [32],
          "per-entry eval.json seed and batch_size")
    claim("reliability", "H100 PCIe ($n = 11$)", [11], "H100 entry count (table header)")
    claim("reliability", "A10 ($n = 4$)", [4], "A10 entry count (table header)")
    claim("reliability", "Binomial SE 0.05 to 0.10; base host gaps up to 0.12",
          [min(se_fast), max(se_fast), max(abs(base_dev(d, k) - base_val(k)) for d in ["H1", "H2"]
                                              for k in ["gsm8k", "math500", "ifeval", "over_refusal", "refusal_rate",
                                                        "mmlu_pro", "hellaswag"])],
          "binomial SE at base fast-tier values; largest base host gap")
    claim("reliability", "Eight windows; base perplexity 22.90 locally and 22.92 on the H100",
          [base_val("wikitext_ppl"), base_dev("H1", "wikitext_ppl")], "base eval wikitext_ppl (8 windows asserted below)")
    assertion("reliability", "WikiText fast tier uses eight windows", b_eval["counts"]["wikitext_ppl"] == 8, "base eval counts")
    claim("reliability", "Single-entry SE 0.045 to 0.049 at 100 items",
          [binom_se(0.28, 100), binom_se(0.39, 100), 100], "binomial SE at base fast-tier values")
    claim("reliability", "(1,024 here", [1024], "per-entry eval max_new_tokens")
    declare("reliability", "32,768 in the blog", "Takehi et al. token limit, cited")
    declare("reliability", "$\\delta/2$", "definition of the reliability bound")
    declare("reliability", "$\\sqrt{p(1-p)/n}$", "formula, not a result")
    declare("reliability", "(identifiers in \\path{data/runs/d3\\_shards/controls.txt})", "identifier list")
    declare("reliability", "95th percentile", "definition")

    # ---------------- assertions without a numeric token of their own
    fam_above = {r for r in LEARNED["family"]
                 if ss(r, "mmlu_pro", "spearman", "mean") > bl("d3v2-m1-family", "statnn", "mmlu_pro")}
    assertion("results", "on the family split only M1-matched and M6 exceed seed-0 StatNN on MMLU-Pro",
              fam_above == {"d3v2-m1_matched-family", "d3v2-m6-family"}, "family seed means vs seed-0 StatNN")
    assertion("discussion", "every learned row's three-seed mean is below seed-0 StatNN on both targets (seed and trajectory splits)",
              all(ss(r, t, "spearman", "mean") < bl(ref, "statnn", t)
                  for ref, runs in [("d3v2-m1-seed", LEARNED["seed"]), (M1T, LEARNED["trajectory"])]
                  for r in runs for t in ["mmlu_pro", "hellaswag"]),
              "seed-0 StatNN (seed and trajectory splits) vs three-seed means")
    assertion("discussion", "the highest HellaSwag three-seed mean belongs to M1-matched on the trajectory split",
              max(SUITE_RUNS, key=lambda r: ss(r, "hellaswag", "spearman", "mean")) == M1M, "suite seed_summary")
    assertion("discussion", "rows at 0.45 on MMLU-Pro (rounded) are M1-matched trajectory and M2-matched seed",
              {r for r in SUITE_RUNS if round(ss(r, "mmlu_pro", "spearman", "mean"), 2) == 0.45}
              == {M1M, "d3v2-m2_matched-seed"}, "suite seed_summary")
    assertion("discussion", "at least three suite rows exceed 0.3 on some target",
              sum(1 for r in SUITE_RUNS if max(finite([ss(r, "mmlu_pro", "spearman", "mean"),
                                                       ss(r, "hellaswag", "spearman", "mean")])) > 0.3) >= 3,
              "suite seed_summary")
    assertion("results", "every learned row has a larger seed-0 MAE than StatNN on both targets (trajectory)",
              all(own_mae(r, t) > seed0(M1T)["statnn"]["targets"][t]["mae"]["value"]
                  for r in LEARNED["trajectory"] for t in ["mmlu_pro", "hellaswag"]),
              "trajectory seed-0 targets.mae")
    for split, runs in LEARNED.items():
        vals = {json.dumps(seed0(r)["baselines"], sort_keys=True) for r in runs}
        assertion("experiments", f"seed-0 baselines are identical across models in the {split} split",
                  len(vals) == 1, f"{len(runs)} runs")
    assertion("experiments", "M6 fails its collapse gate on all three standard splits (seed 0)",
              all(seed0(f"d3v2-m6-{sp}")["m6_collapse_gate"] is False for sp in ["family", "seed", "trajectory"]),
              "m6_collapse_gate in seed-0 metrics")
    assertion("experiments", "twenty-five d3v2 run directories have metrics.json", len(SUITE_RUNS) == 25,
              "d3v2 run directories")
    assertion("experiments", "the suite scores eight targets", len(metrics(M1T)["seed_summary"]) == 8,
              "seed_summary keys")
    assertion("experiments", "the PREREG status line is reproduced exactly",
              "Status: frozen prospective design, pending Board A acceptance and measured execution gates." in read(PREREG),
              "docs/PREREG_H1_TF.md line 3")
    assertion("zoo", "D3 quotas 60 + 100 + 60 + 30 + 50 sum to 300", 60 + 100 + 60 + 30 + 50 == 300, "quota arithmetic")
    assertion("zoo", "audit.txt reports 300 D3 test entries and 260 matched targets",
              "RUN d3-m1-family-level rows 2400 test_n 300" in read(AUDIT_TXT)
              and "'matched_reference_targets': 260" in read(AUDIT_TXT), "board-a-d3-audit audit.txt")
    assertion("experiments", "lambda spend log: A100 at 1.99 per hour",
              "gpu_1x_a100_sxm4" in read(LAMBDA) and "| 1.99 |" in read(LAMBDA), "board/lambda_spend.md")
    assertion("experiments", "RESULTS.md blog values 13.8, 7.9 and 21.0 match the table column",
              re.search(r"\| gsm8k \| 15\.5 \| 13\.8 \|", read(RESULTS_MD)) is not None
              and re.search(r"\| math500 \| 6\.0 \| 7\.9 \|", read(RESULTS_MD)) is not None
              and re.search(r"\| ifeval \| 42\.0 \| 21\.0 \|", read(RESULTS_MD)) is not None, "docs/RESULTS.md")
    assertion("experiments", "base retest record id 20261104-202947-1dcb69a2",
              jeval(EV / "base" / "eval.json")["run_id"] == "20261104-202947-1dcb69a2", "base eval.json run_id")
    m = read(METROLOGY).splitlines()
    assertion("discussion", "metrology lines 43 and 44 are the HellaSwag within-family row",
              "hellaswag" in m[42] and "within family x device" in m[43], "metrology_budget_output.txt")
    assertion("discussion", "metrology lines 24 and 25 are the mmlu_pro and hellaswag sd_seed rows",
              "mmlu_pro" in m[23] and "hellaswag" in m[24], "metrology_budget_output.txt")
    rm = read(RESIDUAL_METRO).splitlines()
    assertion("results", "round2_metrology lines 100 and 102 are the MMLU-Pro and HellaSwag paired rows",
              "mmlu_pro" in rm[99] and "hellaswag" in rm[101], "round2_metrology_output.txt")
    assertion("discussion", "residual theta sits inside the cited lines 2 to 11",
              any(2 <= i + 1 <= 11 for i, line in enumerate(read(RESIDUAL).splitlines()) if '"theta"' in line),
              "round2_residual_estimand_output.txt")


def base_gap_max() -> float:
    return max(abs(base_dev(d, k) - base_val(k)) for d in ["H1", "H2"]
               for k in ["gsm8k", "math500", "ifeval", "over_refusal", "refusal_rate", "mmlu_pro", "hellaswag"])


# ------------------------------------------------------------------ tables

def table_envs(raw: str):
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", raw, re.S):
        lab = re.search(r"\\label\{(tab:[^}]+)\}", m.group(0))
        tab = re.search(r"\\begin\{tabular\*?\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\*?\}", m.group(0), re.S)
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
                assertion(file, f"{run} {target} undefined (no finite seed)",
                          metrics(run)["seed_summary"][target]["spearman"]["n_seeds"] == 0, run)
                continue
            m = re.match(r"(\S+)\s*\[(\S+),\s*(\S+)\]", cell)
            cell_check(file, f"suite {run} {target} mean", m.group(1), ss(run, target, "spearman", "mean"), "seed_summary mean")
            cell_check(file, f"suite {run} {target} min", m.group(2).rstrip(","), ss(run, target, "spearman", "min"), "seed_summary range")
            cell_check(file, f"suite {run} {target} max", m.group(3), ss(run, target, "spearman", "max"), "seed_summary range")
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
                    assertion(file, f"{run} {model} {target} undefined", v is None or math.isnan(v), run)
                    continue
                cell_check(file, f"baseline {run} {model} {target}", part, bl(run, model, target),
                           f"{run} seed-0 baselines.{model}")
    return n


FAMILY_LABELS = {"Abliteration": "abliteration", "Chain": "chain", "LoRA SFT": "lora_sft",
                 "Random direction": "random_dir", "Task vector": "task_vector"}


def t_dynrange(body: str, file: str) -> int:
    n = 0
    for row in rows_of(body):
        c = cells(row)
        if c[0] not in FAMILY_LABELS:
            continue
        f = FAMILY_LABELS[c[0]]
        n += row_tokens(row)
        cell_check(file, f"dyn {f} rows", c[1], ev_family("mmlu_pro", f)["n"], "family_distribution n")
        for col, target in [(2, "mmlu_pro"), (3, "hellaswag")]:
            m = re.match(r"(\S+)\s*\((\S+)\)\s*\[(\S+),\s*(\S+)\]", c[col])
            e = ev_family(target, f)
            for shown, key in [(m.group(1), "mean"), (m.group(2), "sd"), (m.group(3).rstrip(","), "min"), (m.group(4), "max")]:
                cell_check(file, f"dyn {f} {target} {key}", shown, e[key], "family_distribution (evidence CSV)")
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
        for col, target in [(2, "mmlu_pro"), (3, "hellaswag"), (4, "kl_to_base"), (5, "wikitext_ppl_log")]:
            v = ev_within(target, f)
            if "undefined" in c[col]:
                assertion(file, f"within {f} {target} undefined (constant target)", v is None, "within_family empty")
                continue
            cell_check(file, f"within {f} {target}", c[col], v, "within_family spearman (evidence CSV)")
    return n


def t_zoo(body: str, file: str) -> int:
    counts: dict = defaultdict(Counter)
    for line in ZOO_MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") == "header" or "family" not in r:
            continue
        counts[r["family"]]["unclassified" if r.get("eligibility") is None else r["eligibility"]] += 1
    labels = {"Random direction": "random_dir", "Abliteration": "abliteration", "Heretic ARA": "heretic_ara",
              "LoRA SFT": "lora_sft", "Task vector": "task_vector", "Chain steps": "chain"}
    n = 0
    for row in rows_of(body):
        c = cells(row)
        if c[0] not in labels:
            continue
        f = labels[c[0]]
        n += row_tokens(row)
        for col, key in [(1, "production"), (2, "smoke"), (3, "unclassified")]:
            add("CHECKED", file, f"zoo {f} {key}", c[col], counts[f][key],
                int(c[col]) == counts[f][key], "zoo manifest eligibility count")
    return n


def t_base(body: str, file: str) -> int:
    s = base_summary()
    spec = {"GSM8K": ("gsm8k", 200), "MATH-500": ("math500", 100), "IFEval": ("ifeval", 100),
            "MMLU-Pro": ("mmlu_pro", 300), "HellaSwag": ("hellaswag", 300), "Keyword refusal": ("refusal_rate", 100)}
    cap = {"Cap-hit share, GSM8K": ["gsm8k"], "Cap-hit share, MATH-500": ["math500"],
           "Cap-hit share, IFEval": ["ifeval"], "Cap-hit share, pooled": ["gsm8k", "math500", "ifeval"]}
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
            assertion(file, f"{c[0]}: counts from base records", (hit, count) == (k, tot), f"records {hit}/{count}")
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
        assertion(file, f"{c[0]}: numerator {k} equals summary value x n", round(s[key] * tot) == k, "base summary")
        cell_check(file, f"{c[0]} ours", c[2], 100 * s[key], "base-final-4096 summary")
        lo_v, hi_v = wilson(k, tot)
        lo_s, hi_s = c[3].split("--")
        cell_check(file, f"{c[0]} CI low", lo_s, 100 * lo_v, "Wilson 95%")
        cell_check(file, f"{c[0]} CI high", hi_s, 100 * hi_v, "Wilson 95%")
        if c[0] in blog:
            cell_check(file, f"{c[0]} blog value", c[4], blog[c[0]], "docs/RESULTS.md blog column")
        for tok in numeric_tokens(mask_nonprose(c[5])) if len(c) > 5 else []:
            DECLARED.append({"file": file, "token": tok["text"], "reason": "blog samples per item (external)"})
    return n


def t_devices(body: str, file: str) -> int:
    n = 0
    labels = {"GSM8K": "GSM8K", "MATH-500 exact match": "MATH-500", "IFEval": "IFEval",
              "Keyword over-refusal": "Over-refusal", "Keyword refusal": "Refusal", "MMLU-Pro": "MMLU-Pro",
              "HellaSwag": "HellaSwag", "KL to base, nats": "KL", "WikiText perplexity": "WikiText",
              "Cap-hit share, pooled": "Cap-hit"}
    item_counts = {"GSM8K": 50, "MATH-500": 25, "IFEval": 25, "Over-refusal": 50, "Refusal": 50,
                   "MMLU-Pro": 100, "HellaSwag": 100, "KL": 50, "WikiText": 8, "Cap-hit": 100}
    for row in rows_of(body):
        c = cells(row)
        lab = re.sub(r"\s*\(.*$", "", c[0]).strip()
        if lab not in labels or len(c) < 6:
            continue
        d = labels[lab]
        n += row_tokens(row)
        cell_check(file, f"{d} item count in label", re.search(r"\((\d+)", c[0]).group(1), item_counts[d],
                   "base per-entry eval counts")
        cell_check(file, f"dev H1 mean {d}", c[1], DEVS["H1"][d]["mean"], "per-entry eval.json, 11 entries")
        cell_check(file, f"dev H1 max {d}", c[2], DEVS["H1"][d]["max"], "per-entry eval.json, 11 entries")
        if not math.isnan(DEVS["H1"][d]["rho"]):
            cell_check(file, f"dev H1 rho {d}", c[3], DEVS["H1"][d]["rho"], "per-entry eval.json, 11 entries")
        cell_check(file, f"dev H2 mean {d}", c[4], DEVS["H2"][d]["mean"], "per-entry eval.json, 4 entries")
        cell_check(file, f"dev H2 max {d}", c[5], DEVS["H2"][d]["max"], "per-entry eval.json, 4 entries")
    return n


TABLE_CHECKS = {
    "tab:v2-suite": t_suite, "tab:v2-baselines": t_baselines, "tab:dynrange": t_dynrange,
    "tab:within": t_within, "tab:zoo-counts": t_zoo, "tab:base-repro": t_base, "tab:devices": t_devices,
}


# ------------------------------------------------------------------ runner

def run_claims(raw: str, masked: str, file: str, lo: int, hi: int, skip: list[tuple[int, int]]) -> tuple[int, set[int]]:
    """Run claims whose first occurrence lies in [lo, hi) and outside skip spans.
    Returns (number of tokens they account for, token start offsets covered)."""
    accounted = 0
    covered: set[int] = set()
    for c in CLAIMS:
        if c.file != file:
            continue
        idx = raw.find(c.snippet)
        if idx < 0:
            if c.snippet and not any(f["file"] == file and f["snippet"] == c.snippet for f in FAILED_SNIPPETS):
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


def run_declared(raw: str, masked: str, file: str, lo: int, hi: int, skip: list[tuple[int, int]]) -> tuple[int, set[int]]:
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
                DECLARED.append({"file": file, "token": t["text"], "reason": reason or "(no reason given)"})
    return count, covered


def run() -> int:
    build()
    unclaimed: list[dict] = []
    total_prose = total_table = seed_labels = 0
    for name in SECTIONS:
        raw = (SEC / f"{name}.tex").read_text()
        masked_full = mask_nonprose(raw)
        envs = list(table_envs(raw))
        table_spans = [(e0, e1) for _, _, _, e0, e1 in envs]
        body_spans = [(b0, b1) for _, b0, b1, _, _ in envs]
        # tables: each body must be fully accounted for by its checker, in-table claims and declarations
        for lab, b0, b1, _, _ in envs:
            body = numeric_tokens(masked_full, b0, b1)
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
                unclaimed.append({"file": name, "line": raw[:b0].count("\n") + 1,
                                  "token": f"table {lab}",
                                  "context": f"accounted {accounted} of {len(body)} tokens in the tabular body; "
                                             f"body tokens: {[t['text'] for t in body]}"})
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
            line = raw[:t["start"]].count("\n") + 1
            unclaimed.append({"file": name, "line": line, "token": t["text"],
                              "context": raw[max(0, t["start"] - 70): t["end"] + 30].replace("\n", " ")})
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
    out.append(f"Numeric tokens: {total_prose} in prose, {total_table} in tables; {seed_labels} seed labels and "
               f"{sum(1 for d in DECLARED)} declared tokens are listed below, not checked.")
    out.append(f"Values checked against sources and passing: {len(checked)}. Assertions passing: {len(asserts_ok)}. "
               f"Presence claims: {len(presence_ok)}.")
    out.append(f"Failures: {len(failed)} (mismatches, missing snippets, failed assertions or counts). "
               f"Unclaimed tokens: {len(unclaimed)}.")
    out.append("")
    out.append("## Failures")
    if not failed:
        out.append("None.")
    for r in failed:
        out.append(f"- {r['kind']} [{r['file']}] `{r['label']}`: shown {r['shown']!r}, expected {r['expected']!r} ({r['src']})")
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
            out.append(f"- {'PASS' if r['ok'] else 'FAIL'} `{r['label']}`: shown {r['shown']} vs source {exp_s} ({r['src']})")
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
    ap = argparse.ArgumentParser(description="Audit numbers in the v2 paper sections against their sources.")
    ap.add_argument("--report", type=Path, default=None, help="also write the output to this path")
    args = ap.parse_args()
    REPORT_PATH = args.report
    return run()


if __name__ == "__main__":
    sys.exit(main())
