"""Remedy yields from a5_rotation_yield_full.py output (design note ROTATION_YIELD_2026-11-11). No outcome."""

import collections
import json
import sys


def main(full, out):
    d = json.load(open(full))
    res = {"n_edits": len(d["edits"]), "n_tensors": sum(len(e["tensors"]) for e in d["edits"].values())}
    res["reproduces_staged"] = sum(t["reproduces"] for e in d["edits"].values() for t in e["tensors"].values())
    res["registered_recomputed_pass_tensors"] = sum(t["reg"]["registered_pass"] for e in d["edits"].values() for t in e["tensors"].values())
    res["staged_pass_tensors"] = sum(t["staged_pass"] for e in d["edits"].values() for t in e["tensors"].values())
    res["eps_R_over_eps_hat_max"] = max(t["eps_R"] / t["eps_hat"] for e in d["edits"].values() for t in e["tensors"].values())

    def b_pass(t, c, rank_tol=1):
        r = t["reg"]
        return (abs(r["norm"]) <= 2e-2 + t["eps_hat"] and r["spectrum"] <= 5e-2 + t["eps_hat"]
                and abs(r[f"nrank_y_c{c:g}"] - r[f"nrank_d_c{c:g}"]) <= rank_tol)

    def a3_pass(t, mode):
        a = t["a3"]
        if mode == "registered":
            return a["registered_pass"]
        c = 3.0
        return abs(a["norm"]) <= 2e-2 and a["spectrum"] <= 5e-2 and abs(a[f"nrank_y_c{c:g}"] - a[f"nrank_d_c{c:g}"]) <= 1

    rules = {
        "b_c2_rank1": lambda t: b_pass(t, 2.0),
        "b_c3_rank1": lambda t: b_pass(t, 3.0),
        "b_c3_rank2": lambda t: b_pass(t, 3.0, 2),
        "b_norm_spec_only": lambda t: abs(t["reg"]["norm"]) <= 2e-2 + t["eps_hat"] and t["reg"]["spectrum"] <= 5e-2 + t["eps_hat"],
        "registered_norm_spec_only": lambda t: abs(t["reg"]["norm"]) <= 2e-2 and t["reg"]["spectrum"] <= 5e-2,
        "registered_rank_only": lambda t: abs(t["reg"]["rank_y"] - t["reg"]["rank_d"]) <= 1,
    }
    res["rules"] = {}
    for name, fn in rules.items():
        per_edit = {eid: all(fn(t) for t in e["tensors"].values()) for eid, e in d["edits"].items()}
        tens = sum(fn(t) for e in d["edits"].values() for t in e["tensors"].values())
        by = collections.Counter((d["edits"][e]["family"], d["edits"][e]["tertile"]) for e, ok in per_edit.items() if ok)
        res["rules"][name] = {"edits": sum(per_edit.values()), "tensors": tens,
                              "by_family_tertile": {f"{k[0]}:{k[1]}": v for k, v in sorted(by.items())},
                              "passing_edits": sorted(e for e, ok in per_edit.items() if ok)}
    for mode in ("registered", "noise_rank_c3"):
        roots = {eid: e for eid, e in d["edits"].items() if e["root"]}
        per_edit = {eid: all(a3_pass(t, mode) for t in e["tensors"].values()) for eid, e in roots.items()}
        by = collections.Counter((d["edits"][e]["family"], d["edits"][e]["tertile"]) for e, ok in per_edit.items() if ok)
        res["rules"][f"a3_{mode}"] = {"edits": sum(per_edit.values()), "of_root_edits": len(roots),
                                      "tensors": sum(a3_pass(t, mode) for e in roots.values() for t in e["tensors"].values()),
                                      "by_family_tertile": {f"{k[0]}:{k[1]}": v for k, v in sorted(by.items())},
                                      "failing_root_edits": sorted(e for e, ok in per_edit.items() if not ok)}
    # noise-referenced rank detail: how many tensors have any signal above 3 tau
    res["tensors_with_nrank_d_c3_zero"] = sum(t["reg"]["nrank_d_c3"] == 0 for e in d["edits"].values() for t in e["tensors"].values())
    res["seconds_total"] = sum(e["seconds"] for e in d["edits"].values())
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: (v if k != "rules" else {n: {kk: vv for kk, vv in r.items() if kk not in ("passing_edits",)} for n, r in v.items()}) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
