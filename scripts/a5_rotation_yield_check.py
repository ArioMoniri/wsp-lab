"""CPU diagnostic of the Part A rotation-audit yield failure (design note ROTATION_YIELD_2026-11-11).

Reads only stored zoo bytes (edits, predecessors, alias holders, base) and staging metadata. It never reads
an eval record, score, response or contrast, and writes only to ``--out`` (a scratch JSON). No GPU.

Rebuilds D_eff and the registered rotation R exactly as ``wsp.eval.response_intervention.build_variants``
does (same seed, same cloned generator state per tensor, same pinned ``wsp.features.rotation`` functions),
checks the reproduction against the staged audit row, and measures per tensor:

* the bf16 spacing (ulp) of W at every block coordinate, |D_eff| in ulps, and the rms ratio
  kappa = ||D_block|| / sqrt(sum ulp^2) ("D's rms in ulps");
* the fraction of R entries below half an ulp and the fraction actually rounded to zero;
* signed norm mismatch, spectrum mismatch, rank at 1e-3 of D_eff and R_eff, the spectral norm of the
  rounding error E_R = R_eff - R_raw and the ranks above that noise floor;
* for root edits (raw Delta stored): eps_D = ||D_eff - Delta|| / ||D_eff||, rank(Delta) at 1e-3;
* remedy probes (not registered constructions): (a3) rotate Delta instead of D_eff, then round once;
  (a1) error-feedback rounding-aware construction X <- X + (P(Y) - Y), Y = bf16(W + X) - W, P = SVD of Y
  with D_eff's singular values, iterated on a deterministic subset of failing tensors.

Modes: ``--light`` (all selected edits, no SVD: per-tensor norms, kappa, eps_D) and ``--deep ID ...``.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

TOL = {"norm": 2e-2, "spectrum": 5e-2, "rank_rtol": 1e-3, "rank_diff": 1}


def ulp_bf16(w):
    """Spacing of the bf16 grid at each value of w (bf16 tensor); 0 where w == 0."""
    m, e = torch.frexp(w.float())
    u = torch.ldexp(torch.ones_like(m), e - 8)
    return torch.where(w == 0, torch.zeros_like(u), u)


def fro(x):
    return float(torch.linalg.vector_norm(x.double()))


def rank_at(s, tol_abs):
    return int((s > tol_abs).sum())


def realised_audit(sd, Y):
    """Registered realised audit, block form (zeros outside the block do not change the spectrum)."""
    sy = torch.linalg.svdvals(Y)
    nd = float(torch.linalg.vector_norm(sd))
    norm_signed = (fro(Y) - nd) / nd
    k = min(len(sd), len(sy))
    spec = float(torch.linalg.vector_norm(sy[:k] - sd[:k]) / nd)
    rd = int((sd > sd[0] * TOL["rank_rtol"]).sum())
    ry = int((sy > sy[0] * TOL["rank_rtol"]).sum()) if sy[0] > 0 else 0
    ok = abs(norm_signed) <= TOL["norm"] and spec <= TOL["spectrum"] and abs(ry - rd) <= TOL["rank_diff"]
    return {"norm_signed": norm_signed, "spectrum": spec, "rank_d": rd, "rank_y": ry, "pass": ok}, sy


def load_states(zoo, sel, holders, base):
    from wsp.zoo.apply import load_edited_model

    if sel["predecessor_id"] == "base":
        return base, True
    entry = zoo / (holders.get(sel["predecessor_id"]) or sel["predecessor_id"])
    return load_edited_model(base, entry), False


def edit_tensors(zoo, sel, W, base, root):
    """D_eff and raw Delta (roots only) exactly as build_variants computes them."""
    from wsp.features import rotation
    from wsp.zoo.apply import load_entry_delta

    stored = load_entry_delta(zoo / sel["edit_id"], base)
    keys = set(stored)
    pred_changed = set() if root else {k for k in W if not torch.equal(W[k], base[k])}
    D = {}
    for k in sorted(keys | pred_changed):
        S = (base[k].float() + stored[k].float()).to(base[k].dtype) if k in stored else base[k]
        D[k] = rotation.realised_increment(W[k], S)
    if root:
        edited = sorted(keys)
        raw = {k: stored[k].float() for k in edited}
    else:
        edited = sorted(k for k in D if torch.any(D[k] != 0))
        raw = None
    return edited, D, raw


def light_row(W, D, raw, rows, cols):
    Wb = W[rows[:, None], cols]
    Db = D[rows[:, None], cols]
    u = ulp_bf16(Wb)
    nz = Db != 0
    row = {
        "block_shape": [len(rows), len(cols)],
        "d_norm": fro(Db),
        "noise_floor": math.sqrt(float((u.double() ** 2).sum()) / 12.0),  # sqrt(sum ulp^2 / 12)
        "kappa": fro(Db) / math.sqrt(float((u.double() ** 2).sum())),
        "density_d": float(nz.double().mean()),
        "frac_nonzero_d_single_ulp": float(((Db.abs() <= u * 1.0000001) & nz).double().sum() / max(int(nz.sum()), 1)),
        "median_absd_over_ulp_nonzero": float((Db.abs()[nz] / u[nz].clamp_min(1e-45)).median()) if nz.any() else None,
    }
    if raw is not None:
        Dfull = raw
        row["delta_norm"] = fro(Dfull)
        row["eps_D"] = fro(D - Dfull) / max(fro(D), 1e-30)
        row["delta_support_outside_block"] = bool(
            fro(Dfull) ** 2 - fro(Dfull[rows[:, None], cols]) ** 2 > 1e-12 * fro(Dfull) ** 2
        )
        ub = ulp_bf16(W)
        row["frac_delta_below_half_ulp"] = float(((Dfull.abs() < ub / 2) & (ub > 0)).double().mean())
    return row


def error_feedback(W_b, D_b, R_b, sd, iters):
    """Rounding-aware construction probe: returns per-iteration audits and the final deviation from R."""
    X = R_b.clone()
    hist = []
    for it in range(iters + 1):
        Y = (W_b.float() + X).to(torch.bfloat16).float() - W_b.float()
        a, _ = realised_audit(sd, Y)
        a["iter"] = it
        a["dev_from_R"] = fro(X - R_b) / fro(D_b)
        a["realised_dev_from_R"] = fro(Y - R_b) / fro(D_b)
        hist.append(a)
        if a["pass"] or it == iters:
            break
        u, s, vh = torch.linalg.svd(Y, full_matrices=False)
        k = min(len(s), len(sd))
        T = (u[:, :k] * sd[:k]) @ vh[:k]
        X = X + (T - Y)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default="data/zoo/qwen3.5-0.8b")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--deep", nargs="*", default=[])
    ap.add_argument("--ef-iters", type=int, default=12)
    ap.add_argument("--ef-per-class", type=int, default=2)
    a = ap.parse_args()

    from wsp.eval.response_intervention import rotation_seed
    from wsp.features import rotation
    from wsp.features.extract import load_base_state

    torch.set_grad_enabled(False)
    zoo = Path(a.zoo)
    man = json.load(open(a.manifest))
    holders = man["predecessor_bytes_holders"]
    sels = {s["edit_id"]: s for s in man["selections"]}
    audits = {}
    for line in open(a.audits):
        r = json.loads(line)
        audits[(r["edit_id"], r["tensor"])] = r
    rows_m = [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines() if x.strip()]
    header = next(r for r in reversed(rows_m) if r.get("type") == "header")
    rev = json.loads((zoo / next(iter(sels)) / "edit.json").read_text())["base_revision"]
    base, _ = load_base_state(header["base_model"], rev)
    out = {"light": {}, "deep": {}, "tolerances": TOL}
    targets = list(sels) if a.light else []
    for eid in sorted(set(targets) | set(a.deep)):
        t0 = time.time()
        sel = sels[eid]
        W, root = load_states(zoo, sel, holders, base)
        edited, D, raw = edit_tensors(zoo, sel, W, base, root)
        seed = rotation_seed(eid)
        assert seed == man["rotation_seeds"][eid]
        gen = torch.Generator()
        gen.manual_seed(seed)
        light, deep = {}, {}
        for k in edited:
            rows, cols = rotation.changed_block(D[k])
            au = audits[(eid, k)]
            lr = light_row(W[k], D[k], raw[k] if raw is not None else None, rows, cols)
            lr.update({"audit_pass": au["passes"], "audit_rounding_error_rel": au["rounding_error_rel"],
                       "audit_norm": au["norm_mismatch"], "audit_spectrum": au["spectrum_mismatch"],
                       "audit_rank_difference": au["rank_difference"]})
            light[k] = lr
            if eid not in a.deep:
                continue
            R_raw = rotation.rotate_block(D[k], rows, cols, seed, shared_generator=gen)
            _, R_eff = rotation.apply_and_round(W[k], R_raw)
            Wb = W[k][rows[:, None], cols]
            Db = D[k][rows[:, None], cols].float()
            Rb = R_raw[rows[:, None], cols]
            Reb = R_eff[rows[:, None], cols]
            u = ulp_bf16(Wb)
            pos = u > 0
            Er = Reb - Rb
            eps_r = fro(Er) / fro(Db)
            sd = torch.linalg.svdvals(Db)
            ra, sr = realised_audit(sd, Reb)
            e2 = float(torch.linalg.matrix_norm(Er, ord=2))
            d = {
                "reproduces_staged_audit": abs(eps_r - au["rounding_error_rel"]) <= 1e-4 * max(1, au["rounding_error_rel"])
                and ra["rank_d"] == au["rank_d"] and ra["rank_y"] == au["rank_r"],
                "eps_R": eps_r,
                "eps_R_pred_uniform_noise": 1 / (math.sqrt(12) * lr["kappa"]),
                "frac_R_below_half_ulp": float(((Rb.abs() < u / 2) & pos).double().mean()),
                "frac_R_rounded_to_zero": float(((Reb == 0) & (Rb != 0)).double().mean()),
                "median_absR_over_ulp": float((Rb[pos].abs() / u[pos]).median()),
                "median_absD_over_ulp_block": float((Db[pos].abs() / u[pos]).median()),
                "norm_signed": ra["norm_signed"], "spectrum": ra["spectrum"],
                "rank_d_1e-3": ra["rank_d"], "rank_r_1e-3": ra["rank_y"],
                "s1_d": float(sd[0]), "rank_rtol_abs": float(sd[0]) * 1e-3,
                "noise_spec_norm_E_R": e2,
                "rank_d_above_E_R": rank_at(sd, e2), "rank_r_above_E_R": rank_at(sr, e2),
                "registered_pass": au["passes"],
            }
            if raw is not None:
                Xb = raw[k][rows[:, None], cols]
                sx = torch.linalg.svdvals(Xb)
                ED = Db - Xb
                d["rank_delta_1e-3"] = int((sx > sx[0] * 1e-3).sum())
                d["eps_D"] = fro(ED) / fro(Db)
                d["noise_spec_norm_E_D"] = float(torch.linalg.matrix_norm(ED, ord=2))
                d["rank_d_above_E_D"] = rank_at(sd, d["noise_spec_norm_E_D"])
                # (a3) rotate the raw Delta, round once, audit against D_eff
                Rp = rotation.rotate_block(raw[k], rows, cols, seed, shared_generator=gen)
                _, Rpe = rotation.apply_and_round(W[k], Rp)
                a3, _ = realised_audit(sd, Rpe[rows[:, None], cols])
                d["a3_rotate_delta"] = a3
            deep[k] = d
        if eid in a.deep:
            # (a1) error-feedback probe on a deterministic subset of failing tensors
            classes = {}
            for k in edited:
                au = audits[(eid, k)]
                if au["passes"]:
                    continue
                c = ("N" if au["norm_mismatch"] > 2e-2 else "S" if au["spectrum_mismatch"] > 5e-2
                     else "K" if abs(au["rank_difference"]) > 1 else "X")
                classes.setdefault(c, []).append(k)
            for c, ks in sorted(classes.items()):
                for k in ks[: a.ef_per_class]:
                    rows, cols = rotation.changed_block(D[k])
                    R_raw = rotation.rotate_block(D[k], rows, cols, seed, shared_generator=gen)
                    Db = D[k][rows[:, None], cols].float()
                    sd = torch.linalg.svdvals(Db)
                    hist = error_feedback(W[k][rows[:, None], cols], Db, R_raw[rows[:, None], cols], sd, a.ef_iters)
                    deep[k]["a1_class"] = c
                    deep[k]["a1_error_feedback"] = hist
        out["light"][eid] = {"family": sel["family"], "tertile": sel["tertile"], "root": root,
                             "predecessor": sel["predecessor_id"], "tensors": light}
        if deep:
            out["deep"][eid] = deep
        print(f"{eid} {sel['family']} t{sel['tertile']} root={root} tensors={len(edited)} "
              f"{time.time() - t0:.0f}s", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
