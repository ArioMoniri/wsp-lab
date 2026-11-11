"""Summarise a5_rotation_yield_check.py output into the design-note tables (no outcome is read)."""

import argparse
import json
import math
import statistics as st


def cls(a):
    if a["audit_pass"]:
        return "pass"
    if a["audit_norm"] > 2e-2:
        return "N"
    if a["audit_spectrum"] > 5e-2:
        return "S"
    if abs(a["audit_rank_difference"]) > 1:
        return "K"
    return "X"


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def f(x, p=3):
    if x is None:
        return "–"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{p}g}" if abs(x) < 1e-2 and x != 0 else f"{x:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = json.load(open(a.check))
    summary = {"deep": {}, "light": {}, "light_all": {}}
    lines = []
    lines.append("| edit | family/tertile | pred | class | n | κ = rms D/rms ulp (med) | med abs D/ulp, block | D density | frac R < ulp/2 | frac R → 0 | ε_R meas / pred | norm (signed, med) | spectrum (med) | rank_d / rank_r @1e-3 (med) | rank above ‖E_R‖₂ d / r (med) | ε_D (roots) | rank Δ @1e-3 | a3 pass | a1 pass (iters) |")
    lines.append("|" + "---|" * 20)
    for eid, deep in sorted(d["deep"].items()):
        L = d["light"][eid]
        groups = {}
        for k, t in L["tensors"].items():
            groups.setdefault(cls(t), []).append(k)
        for c, ks in sorted(groups.items()):
            dd = [deep[k] for k in ks]
            lt = [L["tensors"][k] for k in ks]
            a3 = [x["a3_rotate_delta"]["pass"] for x in dd if "a3_rotate_delta" in x]
            a1 = [x["a1_error_feedback"] for x in dd if "a1_error_feedback" in x]
            a1s = f"{sum(h[-1]['pass'] for h in a1)}/{len(a1)} ({','.join(str(h[-1]['iter']) for h in a1)})" if a1 else "–"
            row = {
                "n": len(ks), "kappa": med(t["kappa"] for t in lt),
                "absd_ulp_block": med(x.get("median_absD_over_ulp_block") for x in dd),
                "density_d": med(t["density_d"] for t in lt),
                "frac_R_below_half_ulp": med(x["frac_R_below_half_ulp"] for x in dd),
                "frac_R_to_zero": med(x["frac_R_rounded_to_zero"] for x in dd),
                "eps_R": med(x["eps_R"] for x in dd), "eps_R_pred": med(x["eps_R_pred_uniform_noise"] for x in dd),
                "norm_signed": med(x["norm_signed"] for x in dd), "spectrum": med(x["spectrum"] for x in dd),
                "rank_d": med(x["rank_d_1e-3"] for x in dd), "rank_r": med(x["rank_r_1e-3"] for x in dd),
                "rank_d_floor": med(x["rank_d_above_E_R"] for x in dd), "rank_r_floor": med(x["rank_r_above_E_R"] for x in dd),
                "eps_D": med(x.get("eps_D") for x in dd), "rank_delta": med(x.get("rank_delta_1e-3") for x in dd),
                "a3_pass": f"{sum(a3)}/{len(a3)}" if a3 else "–", "a1": a1s,
                "reproduces": all(x["reproduces_staged_audit"] for x in dd),
                "norm_signed_min_max": [min(x["norm_signed"] for x in dd), max(x["norm_signed"] for x in dd)],
            }
            summary["deep"][f"{eid}:{c}"] = row
            lines.append(
                f"| `{eid}` | {L['family']}/{L['tertile']} | {'base' if L['root'] else L['predecessor'][:8]} | {c} | {row['n']} | "
                f"{f(row['kappa'], 2)} | {f(row['absd_ulp_block'], 2)} | {f(row['density_d'], 2)} | {f(row['frac_R_below_half_ulp'], 2)} | "
                f"{f(row['frac_R_to_zero'], 2)} | {f(row['eps_R'], 3)} / {f(row['eps_R_pred'], 3)} | {row['norm_signed']:+.4f} | "
                f"{f(row['spectrum'], 3)} | {f(row['rank_d'], 0)} / {f(row['rank_r'], 0)} | {f(row['rank_d_floor'], 0)} / {f(row['rank_r_floor'], 0)} | "
                f"{f(row['eps_D'], 3)} | {f(row['rank_delta'], 0)} | {row['a3_pass']} | {row['a1']} |"
            )
    # a1 detail
    a1rows = []
    for eid, deep in sorted(d["deep"].items()):
        for k, x in sorted(deep.items()):
            if "a1_error_feedback" in x:
                h = x["a1_error_feedback"]
                a1rows.append({"edit": eid, "tensor": k, "class": x["a1_class"], "iters": h[-1]["iter"],
                               "pass": h[-1]["pass"], "start": {kk: h[0][kk] for kk in ("norm_signed", "spectrum", "rank_d", "rank_y")},
                               "end": {kk: h[-1][kk] for kk in ("norm_signed", "spectrum", "rank_d", "rank_y", "dev_from_R", "realised_dev_from_R")}})
    summary["a1"] = a1rows
    # light pass over all edits: per edit pooled predicted vs measured eps
    allk = []
    for eid, L in sorted(d["light"].items()):
        ts = L["tensors"].values()
        dn2 = sum(t["d_norm"] ** 2 for t in ts)
        nf2 = sum(t["noise_floor"] ** 2 for t in ts)
        meas = math.sqrt(sum((t["audit_rounding_error_rel"] * t["d_norm"]) ** 2 for t in ts) / dn2)
        pred = math.sqrt(nf2 / dn2)
        roots = [t for t in ts if "eps_D" in t]
        epsD = math.sqrt(sum((t["eps_D"] * t["d_norm"]) ** 2 for t in roots) / sum(t["d_norm"] ** 2 for t in roots)) if roots else None
        summary["light"][eid] = {"family": L["family"], "tertile": L["tertile"], "root": L["root"], "n": len(ts),
                                 "d_norm": math.sqrt(dn2), "noise_floor_edit": math.sqrt(nf2),
                                 "eps_R_pooled_meas": meas, "eps_R_pooled_pred": pred, "eps_D_pooled": epsD,
                                 "kappa_median": med(t["kappa"] for t in ts),
                                 "n_pass": sum(t["audit_pass"] for t in ts)}
        for t in ts:
            allk.append((t["audit_rounding_error_rel"], 1 / (math.sqrt(12) * t["kappa"]) if t["kappa"] > 0 else None, t["kappa"]))
    ratios = [m / p for m, p, _ in allk if p]
    summary["light_all"] = {"n_tensors": len(allk), "eps_meas_over_pred_median": med(ratios),
                            "eps_meas_over_pred_p05_p95": [sorted(ratios)[int(.05 * len(ratios))], sorted(ratios)[int(.95 * len(ratios)) - 1]],
                            "kappa_bins": {}}
    for lo, hi in ((0, .5), (.5, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)):
        sel = [r for r in allk if lo <= r[2] < hi]
        if sel:
            summary["light_all"]["kappa_bins"][f"[{lo},{hi})"] = {"n": len(sel), "eps_meas_median": med(r[0] for r in sel),
                                                                 "eps_pred_median": med(r[1] for r in sel)}
    text = "\n".join(lines)
    print(text)
    print(json.dumps({k: summary[k] for k in ("light_all",)}, indent=1))
    for r in a1rows:
        print(r)
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1, sort_keys=True)
        open(a.out.replace(".json", "_table.md"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
