"""Re-run the A5.3.3 full-SVD reconstruction residual in fp64 for the staged tensors whose fp32 residual
exceeded 1e-5 (design note ROTATION_YIELD_2026-11-11). Reads zoo bytes and staging metadata only."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from a5_rotation_yield_check import edit_tensors, load_states  # noqa: E402


def main(manifest, audits, out, zoo="data/zoo/qwen3.5-0.8b"):
    from wsp.features import rotation
    from wsp.features.extract import load_base_state

    torch.set_grad_enabled(False)
    zoo = Path(zoo)
    man = json.load(open(manifest))
    sels = {s["edit_id"]: s for s in man["selections"]}
    bad = [json.loads(x) for x in open(audits)]
    bad = [r for r in bad if r["reconstruction_residual"] > 1e-5]
    rows_m = [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines() if x.strip()]
    header = next(r for r in reversed(rows_m) if r.get("type") == "header")
    rev = json.loads((zoo / bad[0]["edit_id"] / "edit.json").read_text())["base_revision"]
    base, _ = load_base_state(header["base_model"], rev)
    res = []
    for r in bad:
        W, root = load_states(zoo, sels[r["edit_id"]], man["predecessor_bytes_holders"], base)
        _, D, _ = edit_tensors(zoo, sels[r["edit_id"]], W, base, root)
        rows, cols = rotation.changed_block(D[r["tensor"]])
        b = D[r["tensor"]][rows[:, None], cols].double()
        u, s, vh = torch.linalg.svd(b, full_matrices=False)
        rec = float(torch.linalg.vector_norm(u @ torch.diag(s) @ vh - b) / torch.linalg.vector_norm(b))
        res.append({"edit_id": r["edit_id"], "tensor": r["tensor"], "fp32_recon": r["reconstruction_residual"], "fp64_recon": rec,
                    "other_criteria_pass": r["norm_mismatch"] <= 2e-2 and r["spectrum_mismatch"] <= 5e-2 and abs(r["rank_difference"]) <= 1})
        print(res[-1], flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
