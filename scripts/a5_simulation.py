#!/usr/bin/env python3
"""CPU/numpy-only A5 calibration; reads only explicitly supplied JSON/NPZ files.

Registered NULL-SIZE proxy (PREREG_H1_TF A5 freeze-prep amendments, 7 Nov 2026; chair ruling
A5.3.2(ii)): no rotation is scored before G3, so the donor for each development pair is the
edit-minus-predecessor TF-v1 response U_i[w] = y(W_i + D_i)[w] - y(W_i)[w] on the 64 registered
windows (same actual predecessor, scoring identity and windows). The payload carries, per donor
pair and aligned with ``donors``:

* ``centred_noise`` [n_pairs, 64]: U_i centred within the pair over its valid windows
  (U_i[w] - mean_w U_i), NaN (or null) where the window is missing;
* ``window_mask`` [n_pairs, 64]: True where U_i[w] is on file and finite;
* ``donor_mean`` [n_pairs]: the separately pinned mean_w U_i before centring (the donor effect);
* donor rows {id, component, family, norm, eligible, audit[3], audit_evidence[3]} with audit in
  [rotation, ladder .25, ladder .5] order and evidence 'computed' or 'unavailable'.

The three simulated contrasts of a slot are built from that one donor block (``contrasts``):
contrast c has scale a_c in (1, .25, .5); under the null it is a_c * sigma * (mean + noise) with
one fair-coin sigma per simulated component per replicate shared by all three contrasts; under the
registered alternative it is a_c * (|mean| * sign(prediction_c) * B + noise) with one B per
component (P(B=+1) = .75). No observed rotation column exists or is used. Inventory rows keep the
A5.3.1 selection schema {id, component, family, norm, eligible (default true)}. NPZ stores the same
JSON object as a scalar Unicode ``payload``. Fields JSON maps donor row IDs to three finite
predictions in [rotation, ladder .25, ladder .5] order.
`--inputs-spec SPEC` only validates INPUT against the registered NULL-SIZE input
specification (docs/prereg/a5_manifests/nullsize_inputs.json) and writes the
report to --output; it exits non-zero unless the status is "valid".
`--donor-ids FILE` (A5 amendment F7, 11 Nov 2026) restricts the POWER-AND-YIELD donor pool to the
listed donor row IDs (a JSON list, or an object with a ``donor_ids`` list). The IDs are validated
against the payload and the fields; donor cells are rebuilt from the restricted rows with the frozen
cutpoints and the same one-row-per-component rule, and a (family, tertile) cell left without a donor is
a hard error. NULL-SIZE keeps the full payload, so it still reproduces the registered NULL-SIZE run;
seeds, streams, scenarios and decision rules are unchanged. The output records the pool and its cell
counts under ``donor_pool``.
No project data discovery or access is performed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

QUOTAS = {"LoRA": (10, 10, 10), "chain": (4, 3, 3)}
WINDOWS, BLOCKS = 64, 32
SCALES = (1.0, 0.25, 0.5)  # rotation contrast, ladder s = .25, ladder s = .5 (proxy mapping)
PROXY_SOURCE = "tf_v1_edit_minus_actual_predecessor"
AUDIT_EVIDENCE = ("computed", "unavailable")
AUDIT_POLICIES = ("assume_pass", "assume_fail")  # for 'unavailable' audit evidence
CENTRING_TOLERANCE = 1e-9
SCENARIOS = {  # registered run and its frozen sensitivity scenarios (A5 freeze-prep amendments)
    "registered": {"noise_scale": 1.0, "unavailable_audit": "assume_pass"},
    "variance_x2": {"noise_scale": 2.0, "unavailable_audit": "assume_pass"},
    "unavailable_audit_fails": {"noise_scale": 1.0, "unavailable_audit": "assume_fail"},
    # A5.3.2 planned sensitivities, frozen 10 Nov 2026 (Board A freeze review edit 2), outcome-independent:
    # constant positive donor means (constant-sign family effect) ...
    "constant_positive_mean": {
        "noise_scale": 1.0,
        "unavailable_audit": "assume_pass",
        "mean_mode": "family_constant",
    },
    # ... and cell-specific masks/noise (norm-dependent resolution) in the single-donor LoRA-low cell.
    "cell_mask_noise": {
        "noise_scale": 1.0,
        "unavailable_audit": "assume_pass",
        "cell_overrides": {("LoRA", 0): {"active_slots": 1, "noise_scale": 2.0}},
    },
}
MEAN_MODES = ("donor", "family_constant")


def family_constants(cells, inputs):
    """Per family: mean |donor_mean| over its unique donor-cell representatives (``constant_positive_mean``).

    This is the registered alternative's scale (|m_i|, F2) averaged within family into one positive constant;
    it reads only the frozen NULL-SIZE payload (development edit-minus-predecessor means), never G2/G3.
    """
    out = {}
    for family in sorted({f for f, _ in cells}):
        ids = sorted({r["id"] for (f, _), rows in cells.items() if f == family for r in rows})
        values = [abs(inputs[i][1]) for i in ids if i in inputs and np.isfinite(inputs[i][1])]
        if not values or not math.fsum(values) > 0:
            raise ValueError(f"{family}: no positive donor mean for the family constant")
        out[family] = math.fsum(values) / len(values)
    return out


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


def restricted_donor_cells(donors, boundaries, donor_ids, fields=None):
    """Donor cells of the POWER-AND-YIELD pool restricted to ``donor_ids`` (A5 amendment F7).

    Every ID must be a unique, eligible payload donor row (and have a prediction when ``fields`` is
    given). Cells are built by :func:`donor_cells` on the restricted rows with the frozen cutpoints; a
    cell without a donor is a hard error naming the cell. Returns (cells, report).
    """
    ids = list(donor_ids)
    if not ids or not all(isinstance(i, str) for i in ids):
        raise ValueError("donor_ids must be a non-empty list of donor row id strings")
    if len(set(ids)) != len(ids):
        raise ValueError("donor_ids contains duplicate ids")
    by_id = {r["id"]: r for r in donors}
    unknown = sorted(set(ids) - set(by_id))
    if unknown:
        raise ValueError(f"donor_ids not in the payload donors: {unknown}")
    ineligible = sorted(i for i in ids if not by_id[i].get("eligible", True))
    if ineligible:
        raise ValueError(f"donor_ids are ineligible payload donors: {ineligible}")
    if fields is not None:
        missing = sorted(i for i in ids if i not in fields)
        if missing:
            raise ValueError(f"donor_ids without a field prediction: {missing}")
    rows = [by_id[i] for i in sorted(set(ids))]
    rows_per_cell = {key: 0 for key in ((f, t) for f in boundaries for t in range(3))}
    for row in rows:
        family = row["family"]
        if family not in boundaries or not np.isfinite(row["norm"]):
            raise ValueError("invalid donor metadata")
        rows_per_cell[family, sum((row["norm"], row["id"]) >= e for e in boundaries[family])] += 1
    empty = [f"{f}:{t}" for (f, t), n in rows_per_cell.items() if n == 0]
    if empty:
        raise ValueError(f"donor_ids leave empty donor cell(s): {', '.join(empty)}")
    cells = donor_cells(rows, boundaries)
    full = donor_cells(donors, boundaries)
    report = {
        "n_ids": len(ids),
        "ids_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
        "applies_to": "power_yield",
        "cells": {
            f"{f}:{t}": {
                "donor_rows": rows_per_cell[f, t],
                "donor_components": len(cells[f, t]),
                "full_pool_components": len(full[f, t]),
            }
            for (f, t) in cells
        },
    }
    return cells, report


def as_float(values):
    """JSON arrays with null for missing values -> float array with NaN."""
    if isinstance(values, np.ndarray):
        return values.astype(float)

    def convert(v):
        return (
            [convert(x) for x in v] if isinstance(v, list | tuple) else np.nan if v is None else v
        )

    try:
        return np.asarray(convert(values), dtype=float)
    except (TypeError, ValueError):
        return np.zeros(0)


def donor_block(noise, mask, scale=1.0):
    """Two-window block means [32] of one donor's centred noise; a missing window makes its block NaN."""
    noise = np.asarray(noise, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if noise.shape != (WINDOWS,) or mask.shape != (WINDOWS,):
        raise ValueError("centred noise and window mask must have shape [64]")
    values = np.where(mask & np.isfinite(noise), noise, np.nan) * scale
    return values.reshape(BLOCKS, 2).mean(axis=1)


def contrasts(noise_block, means, noise_sign=1):
    """Proxy-to-three-contrast mapping: [32, 3] = SCALES * (means + noise_sign * noise_block).

    ``means`` holds the three signed contrast centres before scaling: sigma * donor mean under the
    null (with noise_sign = sigma), |donor mean| * sign(prediction_c) * B under the alternative
    (noise_sign = 1). One donor noise block is shared by all three contrasts; it is never an
    observed rotation column.
    """
    noise_block = np.asarray(noise_block, dtype=float)
    means = np.broadcast_to(np.asarray(means, dtype=float), (3,))
    if noise_block.shape != (BLOCKS,):
        raise ValueError("noise block must have shape [32]")
    return np.asarray(SCALES)[None, :] * (means[None, :] + noise_sign * noise_block[:, None])


def donor_inputs(payload, noise_scale=1.0, unavailable_audit="assume_pass"):
    """Per eligible donor id: (noise block [32], donor mean, effective audit [3])."""
    if unavailable_audit not in AUDIT_POLICIES:
        raise ValueError(f"unavailable_audit must be one of {AUDIT_POLICIES}")
    donors = payload["donors"]
    noise = as_float(payload["centred_noise"])
    mask = np.asarray(payload["window_mask"], dtype=bool)
    means = as_float(payload["donor_mean"])
    if noise.shape != (len(donors), WINDOWS) or mask.shape != noise.shape:
        raise ValueError("centred_noise and window_mask must have shape [n_pairs, 64]")
    if means.shape != (len(donors),):
        raise ValueError("donor_mean must have shape [n_pairs]")
    out = {}
    for j, row in enumerate(donors):
        if not row.get("eligible", True):
            continue
        audit = np.asarray(row.get("audit", [True] * 3), dtype=bool)
        evidence = row.get("audit_evidence", ["computed"] * 3)
        if audit.shape != (3,) or len(evidence) != 3:
            raise ValueError("audit and audit_evidence must have three contrast entries")
        unavailable = np.asarray([e == "unavailable" for e in evidence])
        audit = np.where(unavailable, unavailable_audit == "assume_pass", audit)
        out[row["id"]] = (donor_block(noise[j], mask[j], noise_scale), float(means[j]), audit)
    return out


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


def simulate(
    payload,
    fields=None,
    *,
    replicates=10000,
    draws=4000,
    quotas=None,
    minimum=15,
    noise_scale=1.0,
    unavailable_audit="assume_pass",
    mean_mode="donor",
    cell_overrides=None,
    donor_ids=None,
):
    """Null without fields; optional power at the registered .75 sign alternative.

    Registered run: noise_scale 1 and unavailable_audit 'assume_pass' (unavailable audit evidence
    counts as a pass, which keeps more pairs and is conservative for size). Registered sensitivity
    scenarios: noise_scale 2 (variance inflation, means unchanged), 'assume_fail',
    ``mean_mode='family_constant'`` (every donor mean replaced by its family's positive constant,
    :func:`family_constants`; noise unchanged) and ``cell_overrides`` {(family, tertile):
    {"active_slots": k, "noise_scale": s}} (cell-specific masks/noise: only the first k slots of the
    cell in slot order stay active, the others are masked as missing in all three contrasts with no
    replacement; the cell's donor noise is multiplied by s). Overrides only change values: every
    random stream is consumed exactly as in the registered run.
    ``donor_ids`` (A5 amendment F7) restricts the POWER-AND-YIELD donor pool only
    (:func:`restricted_donor_cells`); it requires ``fields``. NULL-SIZE draws from the full pool as
    registered, and the random streams are spawned and consumed in the same order.
    Controls/Part B are outside this Part A calibration and cannot be inferred here.
    Replicate/draw/quota overrides are for synthetic tests, never registered runs.
    """
    if mean_mode not in MEAN_MODES:
        raise ValueError(f"mean_mode must be one of {MEAN_MODES}")
    selected, boundaries = selection_cells(payload["inventory"], quotas)
    cells = donor_cells(payload["donors"], boundaries)
    power_cells, donor_pool = cells, None
    if donor_ids is not None:
        if fields is None:
            raise ValueError("donor_ids restricts the POWER-AND-YIELD pool and requires fields")
        power_cells, donor_pool = restricted_donor_cells(
            payload["donors"], boundaries, donor_ids, fields
        )
    streams = np.random.SeedSequence(20261107).spawn(3)
    panels, signs, res_rng = [np.random.Generator(np.random.PCG64(s)) for s in streams]
    slots = [key for key, rows in selected.items() for _ in rows]
    donors = payload["donors"]
    inputs = donor_inputs(payload, noise_scale, unavailable_audit)
    sensitivity = {}
    if mean_mode == "family_constant":
        constants = family_constants(cells, inputs)
        family_of = {r["id"]: r["family"] for r in donors}
        inputs = {
            i: (block, constants[family_of[i]], audit) for i, (block, _, audit) in inputs.items()
        }
        sensitivity["family_constant_means"] = constants
    slot_noise = np.ones(len(slots))
    slot_active = np.ones(len(slots), dtype=bool)
    if cell_overrides:
        for key, override in cell_overrides.items():
            if key not in cells:
                raise ValueError(f"cell override {key} is not a selection cell")
            positions = [s for s, k in enumerate(slots) if k == key]
            slot_noise[positions] = override.get("noise_scale", 1.0)
            slot_active[positions[override.get("active_slots", len(positions)) :]] = False
        sensitivity["cell_overrides"] = [
            {
                "cell": list(key),
                "donor_components": len(cells[key]),
                "slots": sum(k == key for k in slots),
                **override,
            }
            for key, override in cell_overrides.items()
        ]
        sensitivity["active_slots"] = int(slot_active.sum())
    scales = np.asarray(SCALES)
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
        pool = power_cells if mode == "power_yield" else cells
        for _ in range(replicates):
            # Re-run the frozen metadata selection; no outcome replacement.
            selection_cells(payload["inventory"], quotas)
            chosen = [pool[key][panels.integers(len(pool[key]))] for key in slots]
            noise = np.stack([inputs[r["id"]][0] for r in chosen], axis=1)  # [32, slots]
            mean = np.asarray([inputs[r["id"]][1] for r in chosen])
            audit = np.stack([inputs[r["id"]][2] for r in chosen])  # [slots, 3]
            # A pair with any missing window or nonfinite donor mean is missing in all three
            # contrasts, whether or not the outer panel redraws that block.
            complete = np.isfinite(noise).all(axis=0) & np.isfinite(mean) & slot_active
            noise = noise[panels.integers(0, BLOCKS, BLOCKS)] * slot_noise  # common outer panel
            if mode == "null_size":
                prediction = np.ones((len(slots), 3))
                # One fair sign per simulated component, shared by mean and noise of all 3 contrasts.
                flip = signs.choice([-1, 1], size=(len(slots), 1))
                centre = flip * mean[:, None] * np.ones(3)
                noise_sign = flip[:, 0]
            else:
                prediction = np.asarray([fields[r["id"]] for r in chosen], float)
                if prediction.shape != (len(slots), 3) or not np.isfinite(prediction).all():
                    raise ValueError("fields must provide three finite predictions per donor")
                prediction = np.sign(prediction)
                # One Bernoulli per slot, shared across all three prediction signs.
                flip = np.where(signs.random((len(slots), 1)) < 0.75, 1, -1)
                centre = np.abs(mean)[:, None] * prediction * flip
                noise_sign = np.ones(len(slots))
            truth = scales[None, :] * centre  # generating means [slots, 3]
            x = scales[None, None, :] * (
                centre[None, :, :] + (noise_sign[None, :] * noise)[:, :, None]
            )  # [32, slots, 3]; per slot equal to contrasts(noise[:, s], centre[s], noise_sign[s])
            finite = np.isfinite(x).all(axis=0) & complete[:, None]
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
        "proxy": {
            "source": PROXY_SOURCE,
            "donor": "U_i[w] = y(W_i + D_i)[w] - y(W_i)[w], centred within pair over windows",
            "contrast_scales": list(SCALES),
            "null": "a_c * sigma * (donor_mean + centred noise); one sigma per component, all contrasts",
            "alternative": "a_c * (|donor_mean| * sign(prediction_c) * B + centred noise); one B per component",
        },
        "noise_scale": noise_scale,
        "unavailable_audit": unavailable_audit,
        # Present only for the 10 Nov sensitivity scenarios, so earlier raw outputs stay reproducible.
        **({"sensitivity": sensitivity} if sensitivity else {}),
        # Present only with donor_ids (A5 amendment F7), so earlier raw outputs stay reproducible.
        **({"donor_pool": donor_pool} if donor_pool is not None else {}),
        "reports": reports,
    }


def validate_inputs(payload, spec):
    """Check a NULL-SIZE payload against docs/prereg/a5_manifests/nullsize_inputs.json; no simulation.

    Returns {"status": "valid" | "invalid" | "blocked", "errors", "blockers", "counts"}. Errors are schema
    violations of the payload; blockers are open registration issues named by the spec itself.
    """
    schema = spec["payload_schema"]
    errors, blockers = [], []
    missing = [k for k in schema["required_keys"] if k not in payload]
    errors += [f"missing key {k!r}" for k in missing]
    families = set(schema["family_map"].values())
    inventory, donors = payload.get("inventory", []), payload.get("donors", [])
    ids = [r.get("id") for r in inventory]
    if len(set(ids)) != len(ids):
        errors.append("duplicate inventory ids")
    by_id = {r.get("id"): r for r in inventory}
    for r in inventory:
        if not isinstance(r.get("id"), str) or not isinstance(r.get("component"), str):
            errors.append(f"inventory row {r.get('id')!r}: id/component must be strings")
        if r.get("family") not in families:
            errors.append(
                f"inventory row {r.get('id')!r}: family {r.get('family')!r} not in {sorted(families)}"
            )
        eligible = r.get("eligible", True)
        if not isinstance(eligible, bool):
            errors.append(f"inventory row {r.get('id')!r}: eligible must be boolean")
        if eligible and not (isinstance(r.get("norm"), int | float) and math.isfinite(r["norm"])):
            errors.append(f"inventory row {r.get('id')!r}: eligible rows need a finite norm")
    n_pairs, n_windows = len(donors), len(schema["window_item_ids"])
    noise = as_float(payload.get("centred_noise", []))
    try:
        mask = np.asarray(payload.get("window_mask", []))
    except ValueError:
        mask = np.zeros(0)
    means = as_float(payload.get("donor_mean", []))
    shapes_ok = True
    for name, arr, want in (
        ("centred_noise", noise, (n_pairs, n_windows)),
        ("window_mask", mask, (n_pairs, n_windows)),
        ("donor_mean", means, (n_pairs,)),
    ):
        if arr.shape != want:
            errors.append(f"{name} shape {arr.shape} != {want} ([n_pairs, 64] aligned with donors)")
            shapes_ok = False
    if shapes_ok and mask.dtype != bool:
        errors.append("window_mask must hold booleans")
        shapes_ok = False
    masked_windows = incomplete_pairs = 0
    if shapes_ok:
        for j, r in enumerate(donors):
            rid, m, u = r.get("id"), mask[j], noise[j]
            masked_windows += int((~m).sum())
            incomplete_pairs += int(not m.all())
            if not np.isfinite(u[m]).all():
                errors.append(f"donor {rid!r}: valid windows must have finite centred noise")
            if not np.isnan(u[~m]).all():
                errors.append(f"donor {rid!r}: missing windows must be NaN/null, never imputed")
            if m.any():
                scale = max(1.0, float(np.abs(u[m]).max())) if np.isfinite(u[m]).all() else 1.0
                if abs(float(np.mean(u[m]))) > CENTRING_TOLERANCE * scale:
                    errors.append(
                        f"donor {rid!r}: centred_noise is not centred over its valid windows"
                    )
                if not math.isfinite(means[j]):
                    errors.append(
                        f"donor {rid!r}: donor_mean must be finite when a window is valid"
                    )
    order = schema["audit_order"]
    for r in donors:
        rid = r.get("id")
        base = by_id.get(rid)
        if base is None:
            errors.append(f"donor {rid!r} is not an inventory row")
        elif any(base.get(k) != r.get(k) for k in ("component", "family", "norm", "eligible")):
            errors.append(f"donor {rid!r} metadata differs from its inventory row")
        if "residuals" in r:
            errors.append(
                f"donor {rid!r}: 'residuals' (observed rotation column) is not a proxy input"
            )
        audit, evidence = r.get("audit"), r.get("audit_evidence")
        if (
            not isinstance(audit, list)
            or len(audit) != len(order)
            or not all(isinstance(a, bool) for a in audit)
        ):
            errors.append(f"donor {rid!r}: audit must list {len(order)} explicit booleans")
        allowed = schema["audit_evidence_values"]
        if (
            not isinstance(evidence, list)
            or len(evidence) != len(order)
            or any(e not in allowed for e in evidence)
        ):
            errors.append(
                f"donor {rid!r}: audit_evidence must list {len(order)} values in {allowed}"
            )
        prov = payload.get("provenance", {}).get("donors", {}).get(rid)
        if prov is None or any(k not in prov for k in spec["provenance_per_donor"]):
            errors.append(f"donor {rid!r}: provenance needs {spec['provenance_per_donor']}")
    if "window_ids" in payload and payload["window_ids"] != schema["window_item_ids"]:
        errors.append("window_ids differ from the registered development window order")
    proxy = spec["proxy"]
    if not proxy["allowed_sources"]:
        blockers.append(f"proxy source: {proxy['status']}")
    elif payload.get("donor_source") not in proxy["allowed_sources"]:
        errors.append(f"donor_source {payload.get('donor_source')!r} is not registered")
    if not errors:
        try:
            _, boundaries = selection_cells(inventory)
            donor_cells(donors, boundaries)
        except ValueError as exc:
            errors.append(f"selection/donor cells: {exc}")
    return {
        "status": "invalid" if errors else "blocked" if blockers else "valid",
        "errors": errors,
        "blockers": blockers,
        "counts": {
            "inventory": len(inventory),
            "eligible": sum(bool(r.get("eligible", True)) for r in inventory),
            "donors": len(donors),
            "masked_windows": masked_windows,
            "incomplete_pairs": incomplete_pairs,
            "unavailable_audit_entries": sum(
                e == "unavailable" for r in donors for e in (r.get("audit_evidence") or [])
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inputs-spec",
        type=Path,
        help="validate INPUT against this NULL-SIZE spec (nullsize_inputs.json) and exit; no simulation",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="registered",
        help="registered run or a separately labelled frozen sensitivity scenario",
    )
    parser.add_argument(
        "--donor-ids",
        type=Path,
        help="JSON list (or object with 'donor_ids') restricting the POWER-AND-YIELD donor pool "
        "(A5 amendment F7); requires --fields",
    )
    args = parser.parse_args()
    if args.donor_ids and not args.fields:
        parser.error("--donor-ids restricts the POWER-AND-YIELD pool and requires --fields")
    if args.inputs.suffix == ".npz":
        with np.load(args.inputs, allow_pickle=False) as archive:
            payload = json.loads(str(archive["payload"].item()))
    else:
        payload = json.loads(args.inputs.read_text())
    if args.inputs_spec:
        report = validate_inputs(payload, json.loads(args.inputs_spec.read_text()))
        report["input_provenance"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.inputs, args.inputs_spec)
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0 if report["status"] == "valid" else 1
    fields = json.loads(args.fields.read_text()) if args.fields else None
    donor_ids = None
    if args.donor_ids:
        listed = json.loads(args.donor_ids.read_text())
        donor_ids = listed["donor_ids"] if isinstance(listed, dict) else listed
    result = simulate(payload, fields, donor_ids=donor_ids, **SCENARIOS[args.scenario])
    result["scenario"] = args.scenario
    if args.donor_ids:
        result["donor_pool"]["path"] = str(args.donor_ids)
    result["input_provenance"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [args.inputs, *([args.fields] if args.fields else [])]
        + ([args.donor_ids] if args.donor_ids else [])
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
