"""Noise-referenced rank probe (design note ROTATION_YIELD_2026-11-11). For the deep-check edits, counts
singular values of D_eff, R_eff (registered rotation) and, for roots, R'_eff (rotation of the raw Delta)
above c * tau_t, tau_t = sqrt(mean ulp^2 / 12) * (sqrt(m) + sqrt(n)) (predicted spectral norm of one bf16
rounding of W's block), c in {1, 1.5, 2, 3}; also times fp32 and fp64 SVDs for pricing. Zoo bytes only."""

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from a5_rotation_yield_check import edit_tensors, load_states, ulp_bf16  # noqa: E402

CS = (1.0, 1.5, 2.0, 3.0)


def main(manifest, out, *edits, zoo="data/zoo/qwen3.5-0.8b"):
    from wsp.eval.response_intervention import rotation_seed
    from wsp.features import rotation
    from wsp.features.extract import load_base_state

    torch.set_grad_enabled(False)
    zoo = Path(zoo)
    man = json.load(open(manifest))
    sels = {s["edit_id"]: s for s in man["selections"]}
    rows_m = [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines() if x.strip()]
    header = next(r for r in reversed(rows_m) if r.get("type") == "header")
    rev = json.loads((zoo / edits[0] / "edit.json").read_text())["base_revision"]
    base, _ = load_base_state(header["base_model"], rev)
    res = {"c": CS, "edits": {}, "timing": {}}
    for eid in edits:
        W, root = load_states(zoo, sels[eid], man["predecessor_bytes_holders"], base)
        edited, D, raw = edit_tensors(zoo, sels[eid], W, base, root)
        seed = rotation_seed(eid)
        gen = torch.Generator()
        gen.manual_seed(seed)
        per = {}
        for k in edited:
            rows, cols = rotation.changed_block(D[k])
            Wb = W[k][rows[:, None], cols]
            u = ulp_bf16(Wb).double()
            m, n = len(rows), len(cols)
            tau = math.sqrt(float((u ** 2).mean()) / 12) * (math.sqrt(m) + math.sqrt(n))
            Db = D[k][rows[:, None], cols].float()
            R = rotation.rotate_block(D[k], rows, cols, seed, shared_generator=gen)
            _, Re = rotation.apply_and_round(W[k], R)
            sd = torch.linalg.svdvals(Db)
            sr = torch.linalg.svdvals(Re[rows[:, None], cols])
            row = {"tau": tau, "s1_d": float(sd[0]),
                   "rank_d": [int((sd > max(c * tau, 1e-3 * float(sd[0]))).sum()) for c in CS],
                   "rank_r": [int((sr > max(c * tau, 1e-3 * float(sr[0]))).sum()) for c in CS]}
            if raw is not None:
                Rp = rotation.rotate_block(raw[k], rows, cols, seed, shared_generator=gen)
                _, Rpe = rotation.apply_and_round(W[k], Rp)
                sp = torch.linalg.svdvals(Rpe[rows[:, None], cols])
                row["rank_r_a3"] = [int((sp > max(c * tau, 1e-3 * float(sp[0]))).sum()) for c in CS]
            per[k] = row
        res["edits"][eid] = {"family": sels[eid]["family"], "tensors": per}
        print(eid, len(edited), flush=True)
    # timing on one 1024x3584 block
    x = torch.randn(1024, 3584)
    for name, fn in (("svdvals_fp32", lambda: torch.linalg.svdvals(x)),
                     ("svd_fp32", lambda: torch.linalg.svd(x, full_matrices=False)),
                     ("svd_fp64", lambda: torch.linalg.svd(x.double(), full_matrices=False))):
        t = time.time()
        fn()
        res["timing"][name] = time.time() - t
    print(res["timing"])
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
