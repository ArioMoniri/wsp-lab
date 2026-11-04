"""Score an H4 reader run against measured targets.

    python -m wsp.agents.score --run data/runs/h4/<run> --evals data/evals/<zoo> [--n-boot 1000]

Targets are ``<evals>/<id>/eval.json`` converted to the reader's units with ``<evals>/base/eval.json``
(the reader predicts changes, the evaluator stores absolute scores):

  accuracy benchmarks and rates   100 * (y - y_base)            percentage points
  wikitext_ppl                    100 * (y / y_base - 1)         percent
  kl_to_base                      y                              (already relative to base)

Per benchmark: MAE, RMSE, Spearman ρ with seeded 95 % percentile-bootstrap CIs over edits (the same
code as the P3 baselines, ``wsp.predictor.baselines.metrics_with_ci``). Only entries listed in the run manifest (and in ``<evals>/index.jsonl`` when present) are considered;
other entry directories are reported under ``excluded_ids``. Only entries with an ``ok`` parse
and a target are scored; failed replies are counted and reported, never imputed. The command refuses
when no scored entry has a target (or the base eval is missing) and writes ``score.json`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from wsp.predictor.baselines import load_targets, metrics_with_ci
from wsp.schema import BENCHMARKS

log = logging.getLogger("wsp.agents.score")


class NoTargets(RuntimeError):
    """Nothing to score against (no eval.json for any parsed entry, or no base eval)."""


def to_reader_units(current: Mapping[str, float], base: Mapping[str, float]) -> dict[str, float]:
    """Δy in the units of ``reader_prompt.md``; benchmarks without a usable base value are dropped."""
    out: dict[str, float] = {}
    for k, y in current.items():
        if k == "kl_to_base":
            out[k] = float(y)
        elif k not in base:
            continue
        elif k == "wikitext_ppl":
            if base[k] > 0:
                out[k] = 100.0 * (y / base[k] - 1.0)
        else:
            out[k] = 100.0 * (y - base[k])
    return out


def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _evals_index(evals: Path) -> set[str] | None:
    """Edit ids listed in ``<evals>/index.jsonl`` (``None`` when the evals dir has no index)."""
    f = evals / "index.jsonl"
    if not f.exists():
        return None
    ids = {"base"}
    for line in f.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            ids.add(str(row.get("edit_id") or row.get("id")))
    return ids


def score_run(run: str | Path, evals: str | Path, *, n_boot: int = 1000, seed: int = 0) -> dict:
    run, evals = Path(run), Path(evals)
    manifest = run / "manifest.json"
    if not manifest.exists():
        raise NoTargets(f"{run} has no manifest.json; scoring is restricted to the manifest's ids")
    m = json.loads(manifest.read_text())
    current = list(m.get("ids") or [])
    on_disk: dict[str, dict] = {}
    for f in sorted(run.glob("*/parsed.json")):
        on_disk[f.parent.name] = json.loads(f.read_text())
    # Only predictions for ids of the current manifest AND listed in the evals index (when the evals
    # directory has one) are scored; leftovers from earlier runs in the same directory are excluded.
    indexed = _evals_index(evals)
    excluded: dict[str, list[str]] = {"not_in_manifest": [], "not_in_evals_index": []}
    parsed: dict[str, dict] = {}
    for i, p in on_disk.items():
        if i not in current:
            excluded["not_in_manifest"].append(i)
        elif indexed is not None and i not in indexed:
            excluded["not_in_evals_index"].append(i)
        else:
            parsed[i] = p
    for reason, ids_ in excluded.items():
        if ids_:
            log.warning("excluded from scoring (%s): %s", reason, ids_)
    if not parsed:
        raise NoTargets(f"{run} has no parsed.json for the manifest ids (run the reader first)")
    ok = {i: p["parsed"]["predictions"] for i, p in parsed.items() if p.get("status") == "ok"}
    n_failed = sum(1 for p in parsed.values() if p.get("status") != "ok")
    raw, _ = load_targets(evals, sorted(ok))
    if not raw:
        raise NoTargets(f"no eval.json under {evals} for any of the {len(ok)} parsed entries")
    base_raw, _ = load_targets(evals, ["base"])
    if "base" not in base_raw:
        raise NoTargets(
            f"base eval {evals / 'base' / 'eval.json'} is missing; cannot form Δy targets"
        )
    targets = {i: to_reader_units(t, base_raw["base"]) for i, t in raw.items()}

    result: dict[str, Any] = {
        "run": str(run),
        "evals": str(evals),
        "n_boot": n_boot,
        "seed": seed,
        "n_entries": len(parsed),
        "manifest_ids": sorted(current),
        "excluded_ids": excluded,
        "n_failed": n_failed,
        "n_scored": len([i for i in targets if targets[i]]),
        "n_no_target": len(ok) - len(raw),
        "units": "pp for accuracy/rates, % for wikitext_ppl, absolute for kl_to_base",
        "targets": {},
        "skipped": {},
    }
    result["model"] = m.get("model")
    result["reveal_family"] = m.get("reveal_family")
    for b in BENCHMARKS:
        ids = sorted(i for i in targets if b in targets[i])
        if len(ids) < 2:
            result["skipped"][b] = f"{len(ids)} entries with a target (need at least 2)"
            continue
        y = np.array([targets[i][b] for i in ids])
        p = np.array([ok[i][b] for i in ids])
        result["targets"][b] = metrics_with_ci(y, p, seed, n_boot)
    if not result["targets"]:
        raise NoTargets("no benchmark has targets for at least 2 parsed entries")
    result = _clean(result)
    (run / "score.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _show(name: str, m: Mapping[str, Any]) -> str:
    lo, hi = m["ci95"]
    ci = "[n/a]" if lo is None or hi is None else f"[{lo:.4g}, {hi:.4g}]"
    value = "n/a" if m["value"] is None else f"{m['value']:.4g}"
    return f"{name} {value} {ci}"


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--evals", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        res = score_run(args.run, args.evals, n_boot=args.n_boot, seed=args.seed)
    except NoTargets as e:
        sys.exit(f"refusing to score: {e}")
    print(f"scored {res['n_scored']} of {res['n_entries']} entries ({res['n_failed']} failed)")
    for b, m in res["targets"].items():
        print(
            f"{b:14s} n={m['n']:<4d} "
            + "  ".join(_show(k, m[k]) for k in ("mae", "rmse", "spearman"))
        )


if __name__ == "__main__":
    main()
