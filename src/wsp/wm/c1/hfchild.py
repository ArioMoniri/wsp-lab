"""Fresh child process for the HF canaries and every other non-unit generation check (chair ruling
``docs/council/2026-11-21/RULING-wme-c1-bank-halt.md`` K1). Engineering; no scientific content.

The worker (parent) first releases all its GPU state (``Runtime.release``: the model, the field and R-cache tensors),
then runs ``python -m wsp.wm.c1.hfchild <job.json> <out.json>`` exactly as the vLLM path runs ``wm.c1.vllmops``. The
child loads only what the check needs (the pinned model at theta0, through a fresh ``serve.Runtime``), runs the
listed operations and exits, so no bank-phase tensor ever shares a process with an HF canary. A child failure is
raised in the parent with the child's error text (a CUDA out-of-memory keeps its text, so the worker classifies it
``out_of_memory``).

Operations (``job["ops"]``, in order):

* ``{"op": "canary", "hf": bool}``: the teacher-forced canary and, unless ``hf`` is False (C1(b)), the ``hf_gen``
  canary, at the base state (the registered ``ModelCtx.canary_tf`` / ``canary_hf``);
* ``{"op": "gen_repeat", "n": 3}``: ``n`` deterministic ``hf_gen`` canaries and one registered-path run (timing);
* ``{"op": "inert", "scratch": dir}``: the HF part of the pinned-engine inert identity
  (``launchops.inert_hf_rows``); the vLLM part stays in the parent's vLLM child.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

OPS = ("canary", "gen_repeat", "inert")


def job(rt, ops: list) -> dict:
    """A child job for the parent's runtime configuration."""
    from wsp.wm import hf

    bad = [o.get("op") for o in ops if o.get("op") not in OPS]
    if bad:
        raise ValueError(f"unknown child operations {bad}")
    out = {
        "root": str(rt.cfg.root),
        "smoke": bool(rt.cfg.smoke),
        "repo": rt.cfg.repo,
        "ops": list(ops),
    }
    if rt.cfg.smoke:
        out["revision"] = hf.TARGETS.get(rt.cfg.repo)
    return out


def run_job(j: dict, runtime=None) -> dict:
    from wsp.wm.c1 import genengine as GE
    from wsp.wm.c1 import modelops as MO

    GE.prepare_process()
    if runtime is None:
        from wsp.wm import hf
        from wsp.wm.c1 import serve as S

        if j.get("revision"):
            hf.TARGETS.setdefault(j["repo"], j["revision"])
        runtime = S.Runtime(S.Config.make(j["root"], smoke=j["smoke"], vllm=False))
    rt = runtime
    out = {"pid": os.getpid()}
    for op in j["ops"]:
        name = op["op"]
        if name == "canary":
            m = rt.hf()
            rec = {"tf": m.canary_tf(MO.tf_canary_items(MO.PANEL_P2))}
            if op.get("hf", True):
                rec["hf_gen"] = m.canary_hf(MO.gen_canary_items())
            out["canary"] = rec
        elif name == "gen_repeat":
            m = rt.hf()
            items = MO.gen_canary_items()
            det = []
            for _ in range(int(op.get("n", 3))):
                h, s = m.canary_hf_timed(items, mode="hf_deterministic")
                det.append({"sha256": h, "seconds": s})
            h_reg, s_reg = m.canary_hf_timed(items, mode="hf_registered")
            out["gen_repeat"] = {
                "deterministic": det,
                "registered": {"sha256": h_reg, "seconds": s_reg},
            }
        elif name == "inert":
            from wsp.wm.c1 import launchops as LO

            out["inert"] = LO.inert_hf_rows(rt, Path(op["scratch"]))
        else:
            raise ValueError(f"unknown child operation {name!r}")
    return out


def spawn(j: dict, workdir, python=sys.executable, timeout_s: float = 3 * 3600, argv=None) -> dict:
    """Run a job in a fresh child process (the parent must have released its GPU state)."""
    from wsp.wm.c1 import childmem as CM

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # addendum 6: the parent's free GPU memory, checked and recorded before the child starts (no child on a shortfall)
    mem = CM.require(
        "hf", CM.SMOKE_CHILD_GIB if j.get("smoke") else CM.HF_CHILD_GIB, workdir=workdir
    )
    jp, op = workdir / "hf_job.json", workdir / "hf_out.json"
    jp.write_text(json.dumps(j))
    if op.exists():
        op.unlink()
    cmd = argv or [python, "-m", "wsp.wm.c1.hfchild", str(jp), str(op)]
    p = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, env=dict(os.environ), check=False
    )
    (workdir / "hf_child.log").write_text(
        (p.stdout or "")[-20000:] + "\n---\n" + (p.stderr or "")[-20000:]
    )
    if p.returncode != 0:
        tail = (p.stderr or "")[-1500:]
        raise RuntimeError(f"HF child exit {p.returncode}: {tail[-800:]}")
    out = json.loads(op.read_text())
    out["parent_memory"] = mem
    return out


if __name__ == "__main__":
    res = run_job(json.loads(Path(sys.argv[1]).read_text()))
    Path(sys.argv[2]).write_text(json.dumps(res))
