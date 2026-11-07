#!/usr/bin/env python3
"""CPU/numpy-only A5 calibration; reads only explicitly supplied JSON/NPZ files.

JSON schema: inventory and donors are lists of {id, component, family, norm,
eligible (default true)}; donor rows additionally have residuals [64,3]
(edit, rotation, predecessor) and audit [3] (default true). NPZ stores this
same JSON object as a scalar Unicode `payload`. Fields JSON maps donor row IDs
to three finite predictions in [rotation, ladder .25, ladder .5] order.
No project data discovery or access is performed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

QUOTAS = {"LoRA": (10, 10, 10), "chain": (4, 3, 3)}


def cp_interval(k, n, alpha=0.05):
    """Exact equal-tail Clopper-Pearson, via monotone binomial-tail inversion."""
    if not 0 <= k <= n or n < 0:
        raise ValueError("invalid binomial counts")
    if n == 0:
        return 0.0, 1.0

    js = np.arange(n + 1)
    log_coeff = np.array(
        [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) for j in range(n + 1)]
    )

    def tail(p, start, stop):
        weights = log_coeff + js * math.log(p) + (n - js) * math.log1p(-p)
        return float(np.exp(weights[start:stop]).sum())

    def bisect(target, start, stop, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (tail(mid, start, stop) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else bisect(alpha / 2, k, n + 1, True)
    upper = 1.0 if k == n else bisect(alpha / 2, 0, k + 1, False)
    return lower, upper


def selection_cells(inventory, quotas=None):
    """Global component deduplication; frozen rank cutpoints include ID tie-breaks."""
    quotas = QUOTAS if quotas is None else quotas
    representatives = {}
    for row in sorted(inventory, key=lambda x: x["id"]):
        if row.get("eligible", True):
            if row["family"] not in quotas or not np.isfinite(row["norm"]):
                raise ValueError("invalid selection metadata")
            representatives.setdefault(row["component"], row)
    selected, boundaries = {}, {}
    for family, counts in quotas.items():
        rows = sorted(
            (r for r in representatives.values() if r["family"] == family),
            key=lambda r: (r["norm"], r["id"]),
        )
        b1, b2 = len(rows) // 3, 2 * len(rows) // 3
        if b1 == 0 or b2 == len(rows):
            raise ValueError("empty selection cell")
        boundaries[family] = [(rows[b]["norm"], rows[b]["id"]) for b in (b1, b2)]
        for t, group in enumerate((rows[:b1], rows[b1:b2], rows[b2:])):
            if len(group) < counts[t]:
                raise ValueError("selection cell shortage")
            selected[family, t] = sorted(group, key=lambda r: r["id"])[: counts[t]]
    return selected, boundaries


def donor_cells(donors, boundaries):
    """One lexicographically first eligible row per donor component AND matching cell."""
    cells = {(family, t): {} for family in boundaries for t in range(3)}
    for row in sorted(donors, key=lambda r: r["id"]):
        if not row.get("eligible", True):
            continue
        family = row["family"]
        if family not in boundaries or not np.isfinite(row["norm"]):
            raise ValueError("invalid donor metadata")
        t = sum((row["norm"], row["id"]) >= edge for edge in boundaries[family])
        cells[family, t].setdefault(row["component"], row)
    result = {key: list(group.values()) for key, group in cells.items()}
    if any(not group for group in result.values()):
        raise ValueError("empty donor cell")
    return result


def contrasts(row):
    residuals = np.asarray(row["residuals"], dtype=float)
    if residuals.shape != (64, 3):
        raise ValueError("residuals must have shape [64,3]")
    centred = residuals[:, :2] - residuals[:, 2, None]
    return (
        np.stack(
            (centred[:, 0] - centred[:, 1], 0.25 * centred[:, 0], 0.5 * centred[:, 0]),
            axis=-1,
        )
        .reshape(32, 2, 3)
        .mean(axis=1)
    )


def resolution(blocks, rng, draws=4000):
    """Common paired block draws across every slot and contrast; linear percentiles."""
    point = blocks.mean(axis=0)
    if np.all(blocks == blocks[:1]):
        # Degenerate bootstrap is exact; consume no resolution variates in this case.
        return np.stack((point, point))
    indices = rng.integers(0, 32, size=(draws, 32))
    counts = np.zeros((draws, 32), dtype=float)
    for b in range(32):
        counts[:, b] = (indices == b).sum(axis=1)
    means = (counts @ blocks.reshape(32, -1) / 32).reshape(draws, *point.shape)
    return np.quantile(means, [0.025, 0.975], axis=0, method="linear")


def ordered_decision(agreements, resolved, minimum=15, *, intervals=None):
    """Part A projection of A5.5: validity, ladder alternative, rotation go."""
    if intervals is None:
        intervals = [cp_interval(int(k), int(n)) for k, n in zip(agreements, resolved)]
    if resolved[1] >= minimum and intervals[1][1] <= 0.6:
        return "alternative"
    if resolved[0] >= minimum and intervals[0][0] > 0.5:
        return "go"
    return "unresolved"


def simulate(payload, fields=None, *, replicates=10000, draws=4000, quotas=None, minimum=15):
    """Null without fields; optional power at the registered .75 sign alternative.

    Controls/Part B are outside this Part A calibration and cannot be inferred here.
    Replicate/draw/quota overrides are for synthetic tests, never registered runs.
    """
    selected, boundaries = selection_cells(payload["inventory"], quotas)
    cells = donor_cells(payload["donors"], boundaries)
    streams = np.random.SeedSequence(20261107).spawn(3)
    panels, signs, res_rng = [np.random.Generator(np.random.PCG64(s)) for s in streams]
    slots = [key for key, rows in selected.items() for _ in rows]
    donors = payload["donors"]
    blocks = {r["id"]: contrasts(r) for r in donors if r.get("eligible", True)}
    cp = {(k, n): cp_interval(k, n) for n in range(len(slots) + 1) for k in range(n + 1)}
    reports = {}
    for mode in ["null_size", "power_yield"] if fields is not None else ["null_size"]:
        if mode == "power_yield" and not reports["null_size"]["accepted"]:
            reports[mode] = {"accepted": False, "status": "blocked by null-size validity"}
            break
        hits = np.zeros(3, dtype=int)
        two_sided = np.zeros(3, dtype=int)
        yields = np.zeros(3, dtype=int)
        decisions = {"go": 0, "alternative": 0, "unresolved": 0}
        failures = np.zeros(3, dtype=int)
        covered = np.zeros(3, dtype=int)
        coverage_n = np.zeros(3, dtype=int)
        traces = []
        for _ in range(replicates):
            # Re-run the frozen metadata selection; no outcome replacement.
            selection_cells(payload["inventory"], quotas)
            chosen = [cells[key][panels.integers(len(cells[key]))] for key in slots]
            x = np.stack([blocks[r["id"]] for r in chosen], axis=1)
            truth = x.mean(axis=0)
            x = x[panels.integers(0, 32, 32)]  # common outer window-block panel
            audit = np.asarray([r.get("audit", [True] * 3) for r in chosen], dtype=bool)
            if audit.shape != (len(slots), 3):
                raise ValueError("audit must have three contrast flags")
            if mode == "null_size":
                prediction = np.ones((len(slots), 3))
                flip = signs.choice([-1, 1], size=(len(slots), 1))
                x = x * flip[None, :, :]
                truth = truth * flip
            else:
                prediction = np.asarray([fields[r["id"]] for r in chosen], float)
                if prediction.shape != (len(slots), 3) or not np.isfinite(prediction).all():
                    raise ValueError("fields must provide three finite predictions per donor")
                prediction = np.sign(prediction)
                # One Bernoulli per slot, shared across all three prediction signs.
                flip = np.where(signs.random((len(slots), 1)) < 0.75, 1, -1)
                mean = truth.copy()
                truth = np.abs(mean) * prediction * flip
                x = x - mean + truth
            finite = np.isfinite(x).all(axis=0)
            ci = resolution(np.where(np.isfinite(x), x, 0), res_rng, draws)
            resolved = audit & finite & ((ci[0] > 0) | (ci[1] < 0))
            n = resolved.sum(axis=0)
            k = (resolved & (prediction != 0) & (np.sign(x.mean(axis=0)) == prediction)).sum(axis=0)
            intervals = [cp[int(a), int(b)] for a, b in zip(k, n)]
            hits += [b >= minimum and interval[0] > 0.5 for b, interval in zip(n, intervals)]
            two_sided += [
                b >= minimum and (interval[0] > 0.5 or interval[1] < 0.5)
                for b, interval in zip(n, intervals)
            ]
            yields += n >= minimum
            failures += (~(audit & finite)).sum(axis=0)
            covered += (resolved & (ci[0] <= truth) & (truth <= ci[1])).sum(axis=0)
            coverage_n += n
            decision = ordered_decision(k, n, minimum, intervals=intervals)
            traces.append(
                {
                    "donor_representatives": [r["id"] for r in chosen],
                    "component_signs": flip[:, 0].tolist(),
                    "audit_finite": (audit & finite).tolist(),
                    "resolved": resolved.tolist(),
                    "resolved_counts": n.tolist(),
                    "agreement_counts": k.tolist(),
                    "decision": decision,
                }
            )
            decisions[decision] += 1
        rate = decisions["go"] / replicates
        reports[mode] = {
            "ordered_go_rate": rate,
            "ordered_go_cp95": cp_interval(decisions["go"], replicates),
            "two_sided_rejection_rates": (two_sided / replicates).tolist(),
            "two_sided_cp95": [cp_interval(int(v), replicates) for v in two_sided],
            "contrast_rejection_rates": (hits / replicates).tolist(),
            "yield_rates": (yields / replicates).tolist(),
            "yield_cp95": [cp_interval(int(v), replicates) for v in yields],
            "conditional_coverage": np.divide(
                covered, coverage_n, out=np.zeros(3), where=coverage_n > 0
            ).tolist(),
            "audit_or_nonfinite_counts": failures.tolist(),
            "decisions": decisions,
            "traces": traces,
            "positive_rejection_cp95": [cp_interval(int(v), replicates) for v in hits],
            "accepted": (
                rate <= 0.07 and bool(np.all(two_sided / replicates <= 0.07))
                if mode == "null_size"
                else rate >= 0.80
            ),
        }
    return {
        "replicates": replicates,
        "resolution_draws": draws,
        "minimum_resolved": minimum,
        "quotas": QUOTAS if quotas is None else quotas,
        "donor_cluster_members": {
            component: [r["id"] for r in donors if r["component"] == component]
            for component in sorted({r["component"] for r in donors})
        },
        "frozen_cell_cutpoints": boundaries,
        "seed": 20261107,
        "stream_order": ["panels", "signs", "resolution"],
        "registered_alternative": 0.75,
        "reports": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.inputs.suffix == ".npz":
        with np.load(args.inputs, allow_pickle=False) as archive:
            payload = json.loads(str(archive["payload"].item()))
    else:
        payload = json.loads(args.inputs.read_text())
    fields = json.loads(args.fields.read_text()) if args.fields else None
    result = simulate(payload, fields)
    result["input_provenance"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [args.inputs, *([args.fields] if args.fields else [])]
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
