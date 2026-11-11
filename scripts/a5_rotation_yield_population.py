"""Population summary of the Part A staging rotation audits (design note ROTATION_YIELD_2026-11-11).

Reads only staging metadata (rotation_audits.jsonl, manifest.json, rotation variant.json files); never
reads a score, eval record or contrast. CPU only. Writes a JSON summary to --out.
"""

import argparse
import collections
import json
import math
from pathlib import Path


def classify(r):
    if r["passes"]:
        return "pass"
    norm_bad = r["norm_mismatch"] > 2e-2
    spec_bad = r["spectrum_mismatch"] > 5e-2
    rank_bad = abs(r["rank_difference"]) > 1
    if norm_bad:
        return "N_norm"  # entries rounded away / quantised: norm (and nearly always spectrum) fail
    if spec_bad:
        return "S_spectrum_only"
    if rank_bad:
        return "K_rank_only"
    return "X_numeric_only"  # recon / orth only


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audits", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.audits)]
    man = json.load(open(a.manifest))
    fam = {s["edit_id"]: s["family"] for s in man["selections"]}
    tert = {s["edit_id"]: s["tertile"] for s in man["selections"]}
    # per-tensor ||D|| from the logged moments of D (unbiased std over all entries, mean)
    dnorm = {}
    for vid in {r["variant_id"] for r in rows}:
        v = json.load(open(Path(a.stage) / vid / "variant.json"))
        for t, au in v["tensor_audits"].items():
            n = 1
            for s in next(r["shape"] for r in rows if r["variant_id"] == vid and r["tensor"] == t):
                n *= s
            ld = au["logged_d"]
            dnorm[(v["source_id"], t)] = math.sqrt(max((n - 1) * ld["std"] ** 2 + n * ld["mean"] ** 2, 0.0))
    out = {"classes": {}, "by_edit": {}, "notes": "classes: N_norm = norm>2e-2; S = norm ok, spectrum>5e-2; "
           "K = norm and spectrum ok, |rank diff|>1; X = only recon/orth"}
    by = collections.defaultdict(list)
    for r in rows:
        r["family"] = fam[r["edit_id"]]
        r["cls"] = classify(r)
        r["density_d"] = r["nonzero_d"] / (r["block_shape"][0] * r["block_shape"][1])
        r["d_norm"] = dnorm[(r["edit_id"], r["tensor"])]
        by[(r["family"], r["cls"])].append(r)
    for (f, c), rs in sorted(by.items()):
        out["classes"][f"{f}:{c}"] = {
            "n": len(rs),
            "rounding_error_rel_median": q([r["rounding_error_rel"] for r in rs], .5),
            "rounding_error_rel_p10_p90": [q([r["rounding_error_rel"] for r in rs], .1), q([r["rounding_error_rel"] for r in rs], .9)],
            "norm_mismatch_median": q([r["norm_mismatch"] for r in rs], .5),
            "spectrum_mismatch_median": q([r["spectrum_mismatch"] for r in rs], .5),
            "rank_difference_median": q([r["rank_difference"] for r in rs], .5),
            "rank_difference_min_max": [min(r["rank_difference"] for r in rs), max(r["rank_difference"] for r in rs)],
            "rank_d_median": q([r["rank_d"] for r in rs], .5),
            "density_d_median": q([r["density_d"] for r in rs], .5),
            "nonzero_r_over_d_median": q([r["nonzero_r"] / max(r["nonzero_d"], 1) for r in rs], .5),
            "recon_fail": sum(r["reconstruction_residual"] > 1e-5 for r in rs),
        }
    edits = collections.defaultdict(list)
    for r in rows:
        edits[r["edit_id"]].append(r)
    for e, rs in sorted(edits.items()):
        tot = sum(r["d_norm"] ** 2 for r in rs)
        def frac(pred):
            return sum(r["d_norm"] ** 2 for r in rs if pred(r)) / tot
        out["by_edit"][e] = {
            "family": fam[e], "tertile": tert[e], "n_tensors": len(rs),
            "n_pass": sum(r["passes"] for r in rs),
            "classes": dict(collections.Counter(r["cls"] for r in rs)),
            "d_norm_edit": math.sqrt(tot),
            "frac_sqnorm_pass": frac(lambda r: r["passes"]),
            "frac_sqnorm_N": frac(lambda r: r["cls"] == "N_norm"),
            "frac_sqnorm_norm_ok": frac(lambda r: r["norm_mismatch"] <= 2e-2),
            "frac_sqnorm_norm_spec_ok": frac(lambda r: r["norm_mismatch"] <= 2e-2 and r["spectrum_mismatch"] <= 5e-2),
            "pooled_rounding_error_rel": math.sqrt(sum((r["rounding_error_rel"] * r["d_norm"]) ** 2 for r in rs) / tot),
        }
    out["n_tensors"] = len(rows)
    out["counts"] = {f: dict(collections.Counter(r["cls"] for r in rows if r["family"] == f)) for f in ("lora_sft", "chain")}
    # rank-direction among K and N
    out["rank_direction"] = {f: {"up": sum(r["rank_difference"] > 1 for r in rows if r["family"] == f),
                                 "down": sum(r["rank_difference"] < -1 for r in rows if r["family"] == f)} for f in ("lora_sft", "chain")}
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("counts", "rank_direction", "classes")}, indent=1))


if __name__ == "__main__":
    main()
