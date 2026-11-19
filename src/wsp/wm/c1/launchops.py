"""Launch-precondition checks on the pinned engines (PREREG_WME §II.28.3 items 1-3; §II.21.1 (a)). Worker side;
exposed engineering items only (wme-p2 construction, wme-p3h holdout); no wme-c1 or wme-c2-gen item.

* ``inert_identity``: every registered inert construction (``controls.inert``: I1 2-step and 4-step round trips,
  I2 never-indexed rows at each factor, the compensated pairs) applied through the resident session:
  - teacher-forced: I1 endpoints have the base theta-hash; every endpoint's canary-panel values are bitwise equal
    to the base (fp32 bytes);
  - HF generate (batch 128, 128 tokens): I1/I2 endpoints' canary token ids equal the base's;
  - vLLM (stage-W numerics, 2,048 tokens): the I2 rows and the I1 endpoint tensors written into the live engine
    (byte copies), token ids equal the base's.
* ``parity``: 1-vs-2-GPU parity of the teacher-forced evaluation path: the canary-panel chunks (one per objective)
  scored all on GPU 0 versus alternated over GPU 0 and GPU 1 (a second process with ``CUDA_VISIBLE_DEVICES=1``),
  compared bitwise; plus the HF and vLLM canaries on GPU 1 against GPU 0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from wsp.wm.c1 import modelops as MO
from wsp.wm.c1.serve import HANDLERS


def register(name):
    def deco(fn):
        HANDLERS[name] = fn
        return fn

    return deco


def _inert_endpoints(rt, m):
    from wsp.wm.c1 import bankops as BO

    out = []
    for spec in BO.inert_specs(rt.manifest):
        if rt.cfg.smoke and spec["kind"] == "i2":
            continue  # the 0.8B smoke model ties its embeddings: I2 is a 9B (untied) construction
        out.append((spec, BO.inert_steps(spec, m)))
    return out


@register("inert_identity")
def h_inert_identity(rt, req, ctx):
    import torch

    from wsp.wm.c1 import vllmops as VO

    m = rt.hf()
    m.at_base()
    tf_items = MO.tf_canary_items(MO.PANEL_P2)
    gen_items = MO.gen_canary_items()
    base_tf = m.canary_tf(tf_items)
    base_hf = m.canary_hf(gen_items)
    rows, vllm_states = [], []
    for spec, steps in _inert_endpoints(rt, m):
        m.at_base()
        try:
            for b in steps:
                m.sess.apply_bundle(b)
            th = m.theta_hash()
            res, _ = m.evaluate(tf_items)
            tf = MO.sha_values(MO.flatten_values(res, MO.CANARY_TF_OBJECTIVES))
            hf = None
            if spec["kind"] in ("i1", "i2"):
                from wsp.wm import genaudit as GA
                from wsp.wm import p3gen as G

                o = GA.generate_greedy(
                    m.model,
                    m.tok,
                    [G.user_prompt("mmlupro_gen", it) for it in gen_items],
                    MO.CANARY_HF["max_new_tokens"],
                    batch=MO.CANARY_HF["batch"],
                    device="cuda",
                )
                hf = MO.sha_ids([r["ids"] for r in o])
                tensors = {}
                for n in m.sess.touched:
                    if n == m.embed:
                        rows_ = list(range(*spec["rows"])) if spec["kind"] == "i2" else []
                        tensors["embed"] = {
                            "rows": rows_,
                            "values": m.sess.params[n].detach()[rows_].cpu().clone(),
                        }
                    else:
                        L = m.rm.layer_of(n)
                        role = "down" if "down_proj" in n else "o"
                        tensors[(L, role)] = m.sess.params[n].detach().cpu().clone()
                p = ctx.scratch / f"inert_{len(vllm_states)}.pt"
                torch.save(tensors, p)
                vllm_states.append((spec, str(p)))
        finally:
            m.restore()
        rows.append(
            {
                "spec": spec,
                "theta_hash": th,
                "theta_is_base": th == m.base_hash,
                "tf_sha256": tf,
                "tf_bitwise_base": tf == base_tf,
                "hf_sha256": hf,
                "hf_bitwise_base": None if hf is None else hf == base_hf,
            }
        )
        ctx.check()
    vrows = []
    if rt.cfg.vllm:
        from wsp.wm import p3gen as G

        rt.release()
        tok = MO_tokenizer(rt)
        prompts = [G.chat(tok, G.user_prompt("mmlupro_gen", it)) for it in gen_items]
        job = VO.canary_job(rt.cfg.repo, prompts, smoke=rt.cfg.smoke)
        for i, (spec, path) in enumerate(vllm_states):
            job["blocks"].append(
                {
                    "name": f"inert{i}",
                    "prompts": prompts,
                    "max_tokens": VO.CANARY_TOKENS,
                    "state": {"kind": "tensors", "path": path},
                }
            )
        out = VO.spawn(job, ctx.scratch / "vllm")
        base_v = out["blocks"][0]["ids_sha256"]
        for (spec, _p), blk in zip(vllm_states, out["blocks"][1:], strict=True):
            vrows.append(
                {
                    "spec": spec,
                    "vllm_sha256": blk["ids_sha256"],
                    "vllm_bitwise_base": blk["ids_sha256"] == base_v,
                }
            )
    ok = (
        all(r["tf_bitwise_base"] for r in rows)
        and all(r["theta_is_base"] for r in rows if r["spec"]["kind"] == "i1")
        and all(r["hf_bitwise_base"] for r in rows if r["hf_bitwise_base"] is not None)
        and all(r["vllm_bitwise_base"] for r in vrows)
    )
    return {
        "pass": bool(ok),
        "base": {"tf": base_tf, "hf_gen": base_hf},
        "rows": rows,
        "vllm_rows": vrows,
        "n_constructions": len(rows),
    }, {}


def MO_tokenizer(rt):
    from transformers import AutoTokenizer

    from wsp.wm import hf

    return AutoTokenizer.from_pretrained(
        str(hf.snapshot_dir(rt.cfg.repo, hf.TARGETS[rt.cfg.repo])), local_files_only=True
    )


def _chunks():
    items = MO.tf_canary_items(MO.PANEL_P2)
    return [{o: items[o]} for o in MO.CANARY_TF_OBJECTIVES]


def score_chunks(m, idx) -> dict:
    out = {}
    ch = _chunks()
    for i in idx:
        res, _ = m.evaluate(ch[i])
        (o,) = list(ch[i])
        out[i] = MO.sha_values([r["value"] for r in res[o]])
    return out


@register("parity")
def h_parity(rt, req, ctx):
    m = rt.hf()
    m.at_base()
    n = len(_chunks())
    k1 = score_chunks(m, range(n))
    t0 = time.time()
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("WME_C1_PARITY_DEVICE", "1")
    )  # override only for the local engineering smoke
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "wsp.wm.c1.launchops",
            "gpu1",
            json.dumps({"smoke": rt.cfg.smoke, "repo": rt.cfg.repo, "vllm": rt.cfg.vllm}),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=7200,
    )
    if p.returncode != 0:
        raise RuntimeError(f"GPU-1 parity process failed: {p.stderr[-800:]}")
    g1 = json.loads(p.stdout.strip().splitlines()[-1])
    k2 = {i: (k1[i] if i % 2 == 0 else g1["chunks"][str(i)]) for i in range(n)}
    gpu1_all = {i: g1["chunks"][str(i)] for i in range(n)}
    eng0 = {"tf": m.canary_tf(), "hf_gen": m.canary_hf()}
    res = {
        "chunks_k1": k1,
        "chunks_k2": k2,
        "parity_bitwise": k1 == k2,
        "gpu0_vs_gpu1_every_chunk_bitwise": k1 == gpu1_all,
        "canary_gpu0": eng0,
        "canary_gpu1": g1["canary"],
        "canary_gpu0_eq_gpu1": {e: eng0[e] == g1["canary"].get(e) for e in eng0},
        "gpu1_seconds": time.time() - t0,
    }
    if rt.cfg.vllm:
        from wsp.wm import p3gen as G
        from wsp.wm.c1 import vllmops as VO

        rt.release()
        prompts = [
            G.chat(MO_tokenizer(rt), G.user_prompt("mmlupro_gen", it))
            for it in MO.gen_canary_items()
        ]
        v0 = VO.spawn(VO.canary_job(rt.cfg.repo, prompts, smoke=rt.cfg.smoke), ctx.scratch / "v0")[
            "blocks"
        ][0]["ids_sha256"]
        res["vllm_gpu0"] = v0
        res["vllm_gpu1"] = g1.get("vllm")
        res["vllm_gpu0_eq_gpu1"] = v0 == g1.get("vllm")
    return res, {}


def _gpu1_main(cfg):
    from wsp.wm.c1 import serve as S

    c = S.Config.make("/tmp/wme-c1-gpu1", smoke=cfg["smoke"], vllm=cfg["vllm"])
    rt = S.Runtime(c)
    m = rt.hf()
    out = {
        "chunks": {str(i): v for i, v in score_chunks(m, range(len(_chunks()))).items()},
        "canary": {"tf": m.canary_tf(), "hf_gen": m.canary_hf()},
    }
    if cfg["vllm"]:
        from wsp.wm import p3gen as G
        from wsp.wm.c1 import vllmops as VO

        rt.release()
        prompts = [
            G.chat(MO_tokenizer(rt), G.user_prompt("mmlupro_gen", it))
            for it in MO.gen_canary_items()
        ]
        out["vllm"] = VO.spawn(
            VO.canary_job(cfg["repo"], prompts, smoke=cfg["smoke"]), "/tmp/wme-c1-gpu1/v"
        )["blocks"][0]["ids_sha256"]
    print(json.dumps(out))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "gpu1":
    _gpu1_main(json.loads(sys.argv[2]))


@register("timing")
def h_timing(rt, req, ctx):
    """Outcome-blind unit timings for the unpriced bank work (amendment re-pricing, §II.13 item 9): one fp64 GPU
    eigendecomposition of a random 12288 x 12288 PSD matrix (the down_proj A factor size) and of a 4096 one, and one
    factored contraction. Synthetic matrices only."""
    import torch

    from wsp.wm.c1 import field as F

    out = {}
    for n in (4096, 12288):
        g = torch.Generator(device="cpu").manual_seed(n)
        x = torch.randn(n, 2048, generator=g).cuda()
        f = (x @ x.T / 2048).float().cpu()
        torch.cuda.synchronize()
        t0 = time.time()
        UA, lA, aud = F.spectral_factor_gpu(f, "cuda")
        torch.cuda.synchronize()
        out[f"eigh_{n}_s"] = time.time() - t0
        out[f"rank_{n}"] = aud["rank"]
        ctx.check()
    UG, lG, _ = F.spectral_factor_gpu(torch.eye(4096) + 0.01, "cuda")
    D = torch.randn(4096, 12288, device="cuda")
    UA, lA = UA.cuda(), lA.cuda()
    UG, lG = UG.cuda(), lG.cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        M = UG.T @ D @ UA
        _ = float(((lG[:, None] * lA[None, :]) * M.square()).sum())
    torch.cuda.synchronize()
    out["contraction_s"] = (time.time() - t0) / 10
    return out, {}
