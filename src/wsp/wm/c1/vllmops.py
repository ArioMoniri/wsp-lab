"""vLLM path (S-TE engine, §II.6.2; canary engine ``vllm``), run in a child process of the worker.

The parent worker releases its Hugging Face model (``ModelCtx.release``) and runs
``python -m wsp.wm.c1.vllmops <job.json> <out.json>``. A job lists generation blocks; each block is generated at one
engine state, in one ``generate`` call with all its prompts (§II.6.2), under the stage-W frozen numerics
(:data:`ENGINE`: prefix caching off, seed 0, ``gpu_memory_utilization`` 0.88, ``max_num_seqs`` 256,
``max_model_len`` 8,192; greedy unless a block asks for the registered sampled settings).

Engine states are applied in place to the live vLLM writer tensors (``svf.writer_map`` names) and restored bitwise
after each block:

* ``{"kind": "base"}``;
* ``{"kind": "lora", "factors": <path>, "lam": 1.0}``: W <- bf16(fp32(W0) + lam * s * B A) (stage-W application);
* ``{"kind": "tensors", "path": <file>}``: exact realised bf16 tensors keyed by writer (layer, role) or ``embed``,
  copied byte for byte (the E1 winner's realised edit, the inert I2 rows).

Every block returns token ids, text and ``hit_max`` per prompt; the canary block also returns the sha256 of the ids.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ENGINE = {"gpu_memory_utilization": 0.88, "max_num_seqs": 256, "max_model_len": 8192}
CANARY_TOKENS = 2048


def engine_config() -> dict:
    return {"engine": dict(ENGINE), "dtype": "bfloat16", "enable_prefix_caching": False, "seed": 0}


class Engine:
    def __init__(self, repo: str, smoke: bool = False):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        from wsp.wm import hf
        from wsp.wm import svf as V

        path = hf.snapshot_dir(repo, hf.TARGETS[repo])
        kw = dict(ENGINE)
        if smoke:
            kw.update(gpu_memory_utilization=0.2, max_num_seqs=32, max_model_len=4096)
        self.kw = kw
        self.llm = LLM(
            model=str(path),
            dtype="bfloat16",
            enable_prefix_caching=False,
            seed=0,
            limit_mm_per_prompt={"image": 0, "video": 0},
            **kw,
        )
        self.tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        vp = self.llm.apply_model(lambda m: dict(m.named_parameters()))
        self.params = vp[0] if isinstance(vp, list) else vp
        self.vmap = V.writer_map(self.params)
        self.embed = next((n for n in self.params if n.endswith("embed_tokens.weight")), None)
        self.ed = V.LiveEditor(self.params)
        self.names = sorted(self.vmap.values())
        self.base_checksum = self.ed.checksum(self.names)
        self.torch = torch

    def stop_ids(self):
        return {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}

    def gen(
        self, prompts, max_tokens, sample=None, stop_token_ids=None, prompt_ids=None, hf_text=False
    ):
        """``prompt_ids`` (§II.31.8 R5): generate from these token ids (the HF encoding), not by re-tokenising the
        text; ``hf_text``: each output's text is ``tok.decode(ids, skip_special_tokens=True)`` with the HF tokenizer,
        not vLLM's incremental detokenisation."""
        from vllm import SamplingParams

        kw = {"max_tokens": int(max_tokens)}
        if stop_token_ids is not None:  # §II.31 C1(b) blocks: the registered HF stop rule
            kw["stop_token_ids"] = [int(t) for t in stop_token_ids]
        if sample is None:
            sp = SamplingParams(temperature=0.0, top_p=1.0, **kw)
        else:
            sp = SamplingParams(
                temperature=float(sample["temperature"]),
                top_p=float(sample["top_p"]),
                seed=int(sample["seed"]),
                **kw,
            )
        if prompt_ids is not None:
            if len(prompt_ids) != len(prompts):
                raise ValueError("prompt_token_ids and prompts differ in length")
            inputs = [{"prompt_token_ids": [int(t) for t in ids]} for ids in prompt_ids]
        else:
            inputs = prompts
        outs = self.llm.generate(inputs, sp, use_tqdm=False)
        stop = self.stop_ids()
        res = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            while ids and ids[-1] in stop:
                ids.pop()
            text = self.tok.decode(ids, skip_special_tokens=True) if hf_text else c.text
            res.append({"text": text, "ids": ids, "hit_max": c.finish_reason == "length"})
        return res

    def apply_state(self, st: dict):
        import torch

        from wsp.wm import svf as V

        k = st.get("kind", "base")
        if k == "base":
            return
        if k == "lora":
            fac = torch.load(st["factors"], map_location="cpu", weights_only=False)
            deltas = []
            for name, entry in fac.items():
                wk = V.writer_key(name)
                if wk is None or wk not in self.vmap:
                    raise KeyError(f"LoRA tensor {name} has no vLLM writer")
                deltas.append(
                    (self.vmap[wk], lambda e=entry: V.lora_delta(e, st.get("lam", 1.0), "cuda"))
                )
            self.ed.apply(deltas)
            return
        if k == "tensors":
            data = torch.load(st["path"], map_location="cpu", weights_only=False)
            for key, w in data.items():
                name = (
                    self.embed
                    if key == "embed"
                    else self.vmap[tuple(key) if isinstance(key, list) else key]
                )
                live = self.params[name]
                if key == "embed":
                    rows = w["rows"]
                    if name not in self.ed.orig:
                        self.ed.orig[name] = live.detach().clone()
                    with torch.no_grad():
                        live[rows] = w["values"].to(live.device, live.dtype)
                    continue
                if tuple(live.shape) != tuple(w.shape) or live.dtype != w.dtype:
                    raise ValueError(f"{name}: realised tensor shape/dtype mismatch")
                if name not in self.ed.orig:
                    self.ed.orig[name] = live.detach().clone()
                with torch.no_grad():
                    live.copy_(w.to(live.device))
            return
        raise KeyError(k)

    def restore(self):
        self.ed.restore()
        if self.ed.checksum(self.names) != self.base_checksum:
            raise RuntimeError("theta-hash after restore: vLLM writers not back at theta0")


def run_job(job: dict) -> dict:
    from wsp.wm.c1 import modelops as MO

    t0 = time.time()
    # optional: the pinned snapshot of a repo not in hf.TARGETS (the 0.8B smoke model)
    if job.get("revision"):
        from wsp.wm import hf

        hf.TARGETS.setdefault(job["repo"], job["revision"])
    eng = Engine(job["repo"], smoke=bool(job.get("smoke")))
    out = {"engine": eng.kw, "load_s": time.time() - t0, "blocks": []}
    for b in job["blocks"]:
        t = time.time()
        eng.apply_state(b.get("state") or {"kind": "base"})
        try:
            prompts = b["prompts"]
            res = eng.gen(
                prompts,
                b["max_tokens"],
                b.get("sample"),
                b.get("stop_token_ids"),
                b.get("prompt_token_ids"),
                bool(b.get("hf_text")),
            )
        finally:
            eng.restore()
        rec = {
            "name": b["name"],
            "seconds": time.time() - t,
            "outputs": res,
            "ids_sha256": MO.sha_ids([r["ids"] for r in res]),
        }
        out["blocks"].append(rec)
        _progress(job.get("progress"), b["name"], rec["seconds"])
    out["seconds"] = time.time() - t0
    return out


def _progress(spec, name, seconds):
    """Optional job field ``progress`` = {"path", "tag": {...}}: one fsync'd line per completed block, appended to the
    worker's progress file (§II.13 item 3 liveness during a long child job). Absent: nothing is written."""
    if not spec:
        return
    rec = {
        "t": time.time(),
        **spec.get("tag", {}),
        "event": "vllm_block",
        "block": name,
        "seconds": seconds,
    }
    with open(spec["path"], "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def canary_job(repo: str, prompts: list, smoke: bool = False) -> dict:
    return {
        "repo": repo,
        "smoke": smoke,
        "blocks": [
            {
                "name": "canary",
                "prompts": prompts,
                "max_tokens": CANARY_TOKENS,
                "state": {"kind": "base"},
            }
        ],
    }


def child_env(base=None) -> dict:
    """The vLLM child's environment: the stage-W numerics, so without the HF deterministic engine's process-wide
    ``CUBLAS_WORKSPACE_CONFIG`` (§II.31)."""
    env = dict(os.environ if base is None else base, VLLM_ENABLE_V1_MULTIPROCESSING="0")
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    return env


def spawn(job: dict, workdir, python=sys.executable, timeout_s: float = 6 * 3600) -> dict:
    """Run a job in a child process (parent must have released its GPU model)."""
    import subprocess

    from wsp.wm.c1 import childmem as CM

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # addendum 6: the parent's free GPU memory, checked and recorded before the child starts (no child on a shortfall)
    mem = CM.require(
        "vllm", util=0.2 if job.get("smoke") else ENGINE["gpu_memory_utilization"], workdir=workdir
    )
    jp, op = workdir / "vllm_job.json", workdir / "vllm_out.json"
    jp.write_text(json.dumps(job))
    env = child_env()
    p = subprocess.run(
        [python, "-m", "wsp.wm.c1.vllmops", str(jp), str(op)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
        check=False,
    )
    (workdir / "vllm_child.log").write_text(
        (p.stdout or "")[-20000:] + "\n---\n" + (p.stderr or "")[-20000:]
    )
    if p.returncode != 0:
        tail = (p.stderr or "")[-1500:]
        if "out of memory" in tail.lower():
            raise MemoryError(f"CUDA out of memory in the vLLM child: {tail[-300:]}")
        raise RuntimeError(f"vLLM child exit {p.returncode}: {tail[-600:]}")
    out = json.loads(op.read_text())
    if isinstance(out, dict):
        out["parent_memory"] = mem
    return out


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text())
    res = run_job(job)
    Path(sys.argv[2]).write_text(json.dumps(res))
