"""All-tensor remedy yields for the Part A rotation audit (design note ROTATION_YIELD_2026-11-11).

For every edited tensor of the 40 selected edits: rebuilds the registered rotation R (Haar frames cached per
(edit, block shape); rotate_block draws them from the same cloned generator state for every tensor, so the
cache is exact; checked against the staged audit), rounds it at the actual predecessor and records
* the registered realised audit (norm, spectrum, rank at 1e-3) and its reproduction of the staged row;
* remedy (b) inputs: predicted rounding ratio eps_hat = min(1, 1/(sqrt(12) kappa)), measured eps_R, and the
  noise-referenced rank: singular values above max(1e-3 s1, c tau), tau = sqrt(mean ulp^2/12)(sqrt m + sqrt n),
  c in {2, 3}, for D_eff and R_eff;
* remedy (a3), root edits only: R' = Q_out Delta_block Q_in^T rounded once, same audits.
Zoo bytes and staging metadata only; no eval record, score or contrast is read. CPU only.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from a5_rotation_yield_check import edit_tensors, load_states, ulp_bf16  # noqa: E402

CS = (2.0, 3.0)


def audit_block(sd, Y, tau):
    sy = torch.linalg.svdvals(Y)
    nd = float(torch.linalg.vector_norm(sd.double()))
    ny = float(torch.linalg.vector_norm(Y.double()))
    k = min(len(sd), len(sy))
    spec = float(torch.linalg.vector_norm((sy[:k] - sd[:k]).double())) / nd
    rd = int((sd > sd[0] * 1e-3).sum())
    ry = int((sy > sy[0] * 1e-3).sum()) if sy[0] > 0 else 0
    out = {"norm": (ny - nd) / nd, "spectrum": spec, "rank_d": rd, "rank_y": ry,
           "registered_pass": abs(ny - nd) / nd <= 2e-2 and spec <= 5e-2 and abs(ry - rd) <= 1}
    for c in CS:
        out[f"nrank_d_c{c:g}"] = int((sd > max(c * tau, 1e-3 * float(sd[0]))).sum())
        out[f"nrank_y_c{c:g}"] = int((sy > max(c * tau, 1e-3 * float(sy[0]))).sum()) if sy[0] > 0 else 0
    return out


def main(manifest, audits, out, zoo="data/zoo/qwen3.5-0.8b"):
    from wsp.eval.response_intervention import rotation_seed
    from wsp.features import rotation
    from wsp.features.extract import load_base_state

    torch.set_grad_enabled(False)
    zoo = Path(zoo)
    man = json.load(open(manifest))
    sels = {s["edit_id"]: s for s in man["selections"]}
    staged = {}
    for line in open(audits):
        r = json.loads(line)
        staged[(r["edit_id"], r["tensor"])] = r
    rows_m = [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines() if x.strip()]
    header = next(r for r in reversed(rows_m) if r.get("type") == "header")
    rev = json.loads((zoo / next(iter(sels)) / "edit.json").read_text())["base_revision"]
    base, _ = load_base_state(header["base_model"], rev)
    res = {"c": CS, "edits": {}}
    if Path(out).exists():
        res = json.loads(Path(out).read_text())
    for eid in sorted(sels):
        if eid in res["edits"]:
            continue
        t0 = time.time()
        sel = sels[eid]
        W, root = load_states(zoo, sel, man["predecessor_bytes_holders"], base)
        edited, D, raw = edit_tensors(zoo, sel, W, base, root)
        seed = rotation_seed(eid)
        frames = {}
        per = {}
        for k in edited:
            rows, cols = rotation.changed_block(D[k])
            m, n = len(rows), len(cols)
            if (m, n) not in frames:
                g = torch.Generator()
                g.manual_seed(seed)
                frames[(m, n)] = (rotation.haar_orthogonal(m, g), rotation.haar_orthogonal(n, g))
            qo, qi = frames[(m, n)]
            Wb = W[k][rows[:, None], cols]
            Db = D[k][rows[:, None], cols].float()
            u = ulp_bf16(Wb).double()
            tau = math.sqrt(float((u ** 2).mean()) / 12) * (math.sqrt(m) + math.sqrt(n))
            kappa = float(torch.linalg.vector_norm(Db.double())) / math.sqrt(float((u ** 2).sum()))
            Rb = qo @ Db @ qi.T
            Ye = (Wb.float() + Rb).to(torch.bfloat16).float() - Wb.float()
            sd = torch.linalg.svdvals(Db)
            a = audit_block(sd, Ye, tau)
            eps = float(torch.linalg.vector_norm((Ye - Rb).double()) / torch.linalg.vector_norm(Db.double()))
            st = staged[(eid, k)]
            row = {"kappa": kappa, "eps_hat": min(1.0, 1 / (math.sqrt(12) * kappa)) if kappa > 0 else 1.0,
                   "eps_R": eps, "d_norm": float(torch.linalg.vector_norm(Db.double())), "tau": tau,
                   "reg": a, "staged_pass": st["passes"],
                   "reproduces": abs(eps - st["rounding_error_rel"]) <= 1e-4 * max(1.0, st["rounding_error_rel"])
                   and a["rank_d"] == st["rank_d"] and a["rank_y"] == st["rank_r"]}
            if raw is not None:
                Xb = raw[k][rows[:, None], cols].float()
                Rp = qo @ Xb @ qi.T
                Yp = (Wb.float() + Rp).to(torch.bfloat16).float() - Wb.float()
                row["a3"] = audit_block(sd, Yp, tau)
            per[k] = row
        res["edits"][eid] = {"family": sel["family"], "tertile": sel["tertile"], "root": root, "tensors": per,
                             "seconds": time.time() - t0}
        Path(out).write_text(json.dumps(res))
        print(eid, sel["family"], len(edited), f"{time.time() - t0:.0f}s",
              "repro", all(r["reproduces"] for r in per.values()), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
