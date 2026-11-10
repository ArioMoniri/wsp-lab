#!/usr/bin/env python3
"""F8a floor h* from the committed NULL-SIZE payload copy (spec §2.3; PREREG_H1_TF F8a.2).

CPU/numpy only. Reads exactly one JSON payload given by ``--payload`` (default: the committed copy
``docs/prereg/f8_manifests/floor_payload/nullsize_inputs.json``), verifies its sha256 against the pin
``67da223c…`` before use, and computes:

* donor set: every ``donors[j]`` with ``eligible`` true whose 64 windows are all masked-in and finite;
  any other eligible donor is excluded and counted (never imputed); ineligible rows are counted apart;
* blocks: ``a5_simulation.donor_block(centred_noise[j], window_mask[j], 1.0)``, 32 two-window blocks;
* interval: ONE call of ``a5_simulation.resolution`` on the [32, n_donor] block matrix in payload row
  order (4,000 draws, one common [4000, 32] index array, linear .025/.975 quantiles) with
  ``numpy.random.Generator(PCG64(domain_seed("20261111:f8-floor:", "resolution")))``;
* per pair h_j = (q.975 − q.025)/2; per component h_c = numpy.median of h_j; h* = numpy.median of h_c.

Units: WikiText ``extra.mean_nll``, nats per valid next token. Sensitivities (pair-weighted median,
family medians, range, normal approximation) are published and labelled non-binding; they are never
selected among. The result is compared with the spec's design-time preview at its printed precision
and any discrepancy is reported, never tuned away. ``--output`` receives the JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import a5_simulation as sim

PINNED_PAYLOAD_SHA256 = "67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de"
DEFAULT_PAYLOAD = ROOT / "docs/prereg/f8_manifests/floor_payload/nullsize_inputs.json"
SEED_DOMAIN, SEED_KEY = "20261111:f8-floor:", "resolution"
DRAWS = 4000
QUANTILES = (0.025, 0.975)
NOISE_SCALE = 1.0
CODE_FILES = ("scripts/f8_floor.py", "scripts/a5_simulation.py")
# Spec §2.3 design-time preview (10 Nov 2026), printed precision as stated there.
PREVIEW = {
    "h_star": (0.0049397, 7),
    "pair_weighted_median": (0.0011513, 7),
    "lora_median": (0.0091528, 7),
    "chain_median": (0.0006664, 7),
    "h_min": (0.000167, 6),
    "h_max": (0.012924, 6),
    "normal_approx_pair_median": (0.0011536, 7),
}
UNITS = "nats per valid next token (WikiText extra.mean_nll; 1,023 tokens per window)"


def domain_seed(domain: str, key: str) -> int:
    """A5.3.3 convention: sha256(domain + key), first eight bytes big-endian, mod 2^63."""
    return int.from_bytes(hashlib.sha256(f"{domain}{key}".encode()).digest()[:8], "big") % (1 << 63)


FLOOR_SEED = domain_seed(SEED_DOMAIN, SEED_KEY)

RULES = [
    "payload sha256 verified before use against the F8a.2 pin (or an explicit --expected-sha256)",
    (
        "fields: centred_noise [n,64], window_mask [n,64], donors[j].{id, component, family,"
        " eligible}; row j of the arrays is donors[j]; donor_source must be"
        " tf_v1_edit_minus_actual_predecessor"
    ),
    "donor set: eligible rows with all 64 windows masked-in and finite, in payload row order",
    "missingness: eligible donors with any masked or nonfinite window excluded and counted; no imputation",
    "pairing: 32 two-window blocks (2k, 2k+1) via a5_simulation.donor_block(noise, mask, 1.0)",
    (
        "interval: one call of a5_simulation.resolution on the [32, n_donor] block matrix,"
        " 4000 draws, common [4000, 32] index array, linear .025/.975 quantiles"
    ),
    f"rng: numpy.random.Generator(PCG64({FLOOR_SEED})) = domain_seed({SEED_DOMAIN!r}, {SEED_KEY!r})",
    "per pair: h_j = (q.975 - q.025) / 2",
    "per component: h_c = numpy.median of h_j over the component's pairs",
    "floor (binding): h* = numpy.median of h_c over components (even count: mean of the middle two)",
    "comparison downstream: strict, a component qualifies iff Q_c > h*",
    "sensitivities (pair-weighted, LoRA/chain, range, normal approximation) are non-binding",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _median(values) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def compute_floor(payload: dict) -> dict:
    """Floor computation proper (no I/O). Raises ValueError on any schema violation."""
    if payload.get("donor_source") != sim.PROXY_SOURCE:
        raise ValueError(f"donor_source must be {sim.PROXY_SOURCE!r}")
    donors = payload["donors"]
    noise = sim.as_float(payload["centred_noise"])
    mask = np.asarray(payload["window_mask"], dtype=bool)
    n = len(donors)
    if noise.shape != (n, sim.WINDOWS) or mask.shape != (n, sim.WINDOWS):
        raise ValueError(
            f"centred_noise {noise.shape} and window_mask {mask.shape} must be [{n}, {sim.WINDOWS}]"
        )
    for row in donors:
        if not {"id", "component", "family", "eligible"} <= set(row):
            raise ValueError(f"donor row lacks id/component/family/eligible: {row}")
    if len({row["id"] for row in donors}) != n:
        raise ValueError("donor ids are not unique")

    kept, excluded, ineligible = [], [], []
    for j, row in enumerate(donors):
        if row["eligible"] is not True:
            ineligible.append(row["id"])
        elif mask[j].all() and np.isfinite(noise[j]).all():
            kept.append(j)
        else:
            excluded.append(row["id"])
    if not kept:
        raise ValueError("no eligible donor with 64 complete windows")

    blocks = np.stack([sim.donor_block(noise[j], mask[j], NOISE_SCALE) for j in kept], axis=1)
    if not np.isfinite(blocks).all():
        raise ValueError("nonfinite block mean among kept donors")
    rng = np.random.Generator(np.random.PCG64(FLOOR_SEED))
    lo, hi = sim.resolution(blocks, rng, draws=DRAWS)
    half = (hi - lo) / 2

    pairs = [
        {
            "id": donors[j]["id"],
            "component": donors[j]["component"],
            "family": donors[j]["family"],
            "h": float(h),
            "q025": float(a),
            "q975": float(b),
        }
        for j, h, a, b in zip(kept, half, lo, hi, strict=True)
    ]
    by_comp: dict[str, list[float]] = {}
    for r in pairs:
        by_comp.setdefault(r["component"], []).append(r["h"])
    components = {c: _median(v) for c, v in sorted(by_comp.items())}
    h_star = _median(list(components.values()))

    hs = [r["h"] for r in pairs]
    by_family: dict[str, list[float]] = {}
    for r in pairs:
        by_family.setdefault(r["family"], []).append(r["h"])
    sd = blocks.std(axis=0, ddof=1)
    normal = 1.959963984540054 * sd / np.sqrt(sim.BLOCKS)
    sensitivities = {
        "pair_weighted_median": {
            "value": _median(hs),
            "binding": False,
            "definition": "numpy.median of h_j over all kept pairs (each pair weighted equally)",
        },
        **{
            f"{fam.lower()}_median": {
                "value": _median(v),
                "binding": False,
                "n_pairs": len(v),
                "definition": f"numpy.median of h_j over {fam} pairs only",
            }
            for fam, v in sorted(by_family.items())
        },
        "h_range": {"min": float(min(hs)), "max": float(max(hs)), "binding": False},
        "normal_approx_pair_median": {
            "value": _median(normal),
            "binding": False,
            "definition": "numpy.median over pairs of z.975 * SD(32 block means, ddof=1) / sqrt(32)",
        },
    }
    return {
        "h_star": h_star,
        "components": components,
        "component_pairs": {c: len(v) for c, v in sorted(by_comp.items())},
        "pairs": pairs,
        "sensitivities": sensitivities,
        "counts": {
            "donor_rows": n,
            "pairs": len(kept),
            "components": len(components),
            "excluded_masked_or_nonfinite": len(excluded),
            "excluded_ineligible": len(ineligible),
            "excluded_ids": sorted(excluded + ineligible),
            "masked_windows_total": int((~mask).sum()),
        },
    }


def preview_comparison(result: dict) -> dict:
    s = result["sensitivities"]
    observed = {
        "h_star": result["h_star"],
        "pair_weighted_median": s["pair_weighted_median"]["value"],
        "lora_median": s.get("lora_median", {}).get("value"),
        "chain_median": s.get("chain_median", {}).get("value"),
        "h_min": s["h_range"]["min"],
        "h_max": s["h_range"]["max"],
        "normal_approx_pair_median": s["normal_approx_pair_median"]["value"],
    }
    out = {}
    for key, (preview, digits) in PREVIEW.items():
        value = observed[key]
        printed = None if value is None else f"{value:.{digits}f}"
        out[key] = {
            "preview": f"{preview:.{digits}f}",
            "observed_full": value,
            "observed_printed": printed,
            "agrees_at_printed_precision": printed == f"{preview:.{digits}f}",
            "difference": None if value is None else value - preview,
        }
    return out


def build_report(payload_path: Path, expected_sha256: str) -> dict:
    payload_path = Path(payload_path)
    raw = payload_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expected_sha256:
        raise ValueError(f"payload sha256 {sha} != expected {expected_sha256}; refusing to compute")
    result = compute_floor(json.loads(raw))
    return {
        "schema": "f8-floor-v1",
        "status": "computed",
        "binding_rule": "h* = median over components of (median over the component's pairs of h_j)",
        "h_star": result["h_star"],
        "units": UNITS,
        "payload": {
            "path": str(payload_path),
            "sha256": sha,
            "bytes": len(raw),
            "pinned_sha256": PINNED_PAYLOAD_SHA256,
            "pinned": sha == PINNED_PAYLOAD_SHA256,
        },
        "code": {rel: sha256_file(ROOT / rel) for rel in CODE_FILES},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "rng": {
            "bit_generator": "PCG64",
            "seed": FLOOR_SEED,
            "domain": SEED_DOMAIN,
            "key": SEED_KEY,
            "draws": DRAWS,
            "quantiles": list(QUANTILES),
            "quantile_method": "linear",
        },
        "rules": RULES,
        "counts": result["counts"],
        "components": result["components"],
        "component_pairs": result["component_pairs"],
        "pairs": result["pairs"],
        "sensitivities": result["sensitivities"],
        "preview_comparison": preview_comparison(result),
        "spec": "docs/design/F8_EXECUTION_SPEC.md §2.3; docs/PREREG_H1_TF.md F8a.2",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--expected-sha256",
        default=PINNED_PAYLOAD_SHA256,
        help="override the pinned payload sha256 (tests only; the report records pinned=false)",
    )
    args = ap.parse_args(argv)
    try:
        report = build_report(args.payload, args.expected_sha256)
    except (ValueError, KeyError, OSError) as exc:
        print(f"f8_floor: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n")

    c = report["counts"]
    print(f"h* = {report['h_star']:.7f} {report['units']}  (full: {report['h_star']!r})")
    print(f"  binding rule: {report['binding_rule']}")
    print(
        f"  payload: {report['payload']['path']} sha256 {report['payload']['sha256']} "
        f"(pinned: {report['payload']['pinned']})"
    )
    for rel, h in report["code"].items():
        print(f"  code: {rel} sha256 {h}")
    env = report["environment"]
    print(f"  python {env['python']}, numpy {env['numpy']}; rng PCG64({FLOOR_SEED}), {DRAWS} draws")
    print(
        f"  donors: {c['pairs']} pairs in {c['components']} components; excluded "
        f"{c['excluded_masked_or_nonfinite']} masked/nonfinite, {c['excluded_ineligible']} ineligible"
    )
    for rule in report["rules"]:
        print(f"  rule: {rule}")
    print("  non-binding sensitivities:")
    for key, s in report["sensitivities"].items():
        val = s.get("value", (s.get("min"), s.get("max")))
        print(f"    {key}: {val}")
    print("  design-time preview (spec §2.3):")
    for key, cmp in report["preview_comparison"].items():
        verdict = "agrees" if cmp["agrees_at_printed_precision"] else "DIFFERS"
        print(
            f"    {key}: preview {cmp['preview']} observed {cmp['observed_printed']} -> {verdict}"
            + ("" if cmp["agrees_at_printed_precision"] else f" (diff {cmp['difference']!r})")
        )
    print(f"  report: {args.output}")
    return report


if __name__ == "__main__":
    main()
