"""S-TE on the worker (PREREG_WME §II.6, §II.12.2-§II.12.3, §II.13, §II.22.5, §II.23.3).

Handlers (registered into ``wsp.wm.c1.serve.HANDLERS``):

* ``ste_train`` (args: optional ``force``). The single H1-type LoRA of §II.6.1 through the stage-W train path
  (``scripts/wme_stagew.py``: ``teacher_rows``, ``prompt_ids``, ``writer_names``; ``wm.lora.LoRA`` / ``wm.lora.train``
  with checkpointing and ``ce_chunk`` 1024 as stage W): rank 8, alpha 16, AdamW lr 1e-4, 2 epochs, 8 rows per step,
  10 warm-up steps, every layer's residual writers, loss on the response tokens plus ``<|im_end|>``, non-thinking
  template; data = the 828 gold-verified pilot-3 teacher rows (stage-W eligibility: correct, non-empty, <= 2,048
  tokens) in file order; seed :data:`SEED` = int(sha256("wme-c2:h1:train")[:8], 16) for both the factor
  initialisation and the data order (the registration names one seed). 208 optimiser steps are asserted. Only the
  final factors are saved (lambda 1, all layers; nothing is selected). It runs in a child process
  (``python -m wsp.wm.c1.steops train``) because stage W's ``hf.use_reference_kernels`` patches the Qwen3.5 module
  globally and must not change the resident model's numerics.
* ``ste_calibrate`` (no args). Base only, one vLLM child job: the 128-item ``calibration`` split of each generated
  guard (``wme-c2-gen/open``), greedy then 8 samples at T 0.7 (top-p 1.0, seeds 1..8: pilot-3 ``h0_guard_cal``),
  96 new tokens; refused = the pilot-3 frozen rule (``genaudit.detect_refusal`` on the response, as pilot-3
  ``score``); per-item refusal frequency pi = mean over the 8 samples (pilot-3 ``h0_guard_cal``; greedy recorded
  beside it); margin = ``p3gen.guard_margin(pi, n_cal 128, n_boot 4000, seed 0, floor 0.01)``; null pass
  probability = ``p3gen.null_pass_prob(max(d, 1/128), margin, N)`` with d the registered sizing discordance and N
  523 / 200. Written to ``state/ste/calibration.json`` and returned before any edited generation exists.
* ``ste_generate`` (args: ``panel_files`` {name: {"path", "sha256"}} (the uploaded ``wme-c2-gen`` sealed_test),
  ``lora_factors``, ``lora_factors_sha256``, ``margins`` (required: the ``ste_calibrate`` margins), optional
  ``e1_winner_tensors`` {"path", "sha256"}, optional ``canary`` (default true)). One vLLM child job (stage-W frozen
  numerics, ``wm.c1.vllmops``): [canary], MMLU-Pro then SuperGPQA x budget 2,048 then 4,096 x base then edit (LoRA
  lambda 1), one block (= one ``generate`` call) each with all N prompts in file order; then guard_harmless and
  guard_harmful x base, edit [, e1_winner], greedy, 96 tokens; [canary repeat]. Scoring: strict ``p3gen.verify``
  (truncated = incorrect) and the flexible re-parse (``scripts/wme_stagew2.py score``); guards: the frozen refusal
  rule. Outputs: ``ste_outputs.jsonl.gz``, ``ste_guards.jsonl.gz`` (``wm.c1.schema`` ste rows) and
  ``ste_raw.jsonl.gz`` (every raw generation). No accuracy or difference is computed or logged here.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

from wsp.wm.c1.genops import read_checked_jsonl, register, sha256_file, write_jsonl_gz

ROOT = Path(__file__).resolve().parents[4]
PANEL_C2 = ROOT / "data/wme/panels/wme-c2-gen"
TEACHER_P3 = ROOT / "data/wme/pilot3/run-20261117/h/teacher.jsonl"
SEED = int(hashlib.sha256(b"wme-c2:h1:train").hexdigest()[:8], 16)
LORA = {"rank": 8, "alpha": 16, "lr": 1e-4, "epochs": 2, "batch_rows": 8, "warmup": 10}
N_TEACHER = 828
N_STEPS = 208
SMOKE_ROWS = 16
BENCHMARKS = ("mmlupro_gen", "supergpqa_gen")
BUDGETS = (2048, 4096)
MODELS = ("base", "edit")
GUARDS = ("guard_harmless", "guard_harmful")
GUARD_TOKENS = 96
CAL_N = 128
CAL_SEEDS = tuple(range(1, 9))  # pilot-3 h0_guard_cal: seeds s = 1..CAL_SAMPLES
CAL_SAMPLE = {"temperature": 0.7, "top_p": 1.0}
MARGIN = {"n_cal": 128, "n_boot": 4000, "seed": 0, "floor": 0.01}
FACTORS = "h1_factors.pt"
OUTPUTS = "ste_outputs.jsonl.gz"
GUARD_ROWS = "ste_guards.jsonl.gz"
RAW = "ste_raw.jsonl.gz"
GENERATED = (
    "generated.json"  # written before the first edited generation; ste_calibrate then refuses
)


def ste_dir(rt) -> Path:
    d = Path(rt.cfg.state) / "ste"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _progress_spec(ctx) -> dict | None:
    w = getattr(ctx, "worker", None)
    if w is None:
        return None
    return {
        "path": str(Path(w.root) / "progress.jsonl"),
        "tag": {"id": ctx.req["id"], "kind": ctx.req["kind"]},
    }


def _progress_line(spec, event, **kw):
    if not spec:
        return
    with open(spec["path"], "a") as f:
        f.write(json.dumps({"t": time.time(), **spec.get("tag", {}), "event": event, **kw}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _repo_job(rt) -> dict:
    from wsp.wm import hf

    return {
        "repo": rt.cfg.repo,
        "revision": hf.TARGETS.get(rt.cfg.repo),
        "smoke": bool(rt.cfg.smoke),
    }


def _tokenizer(rt):
    from transformers import AutoTokenizer

    from wsp.wm import hf

    return AutoTokenizer.from_pretrained(
        str(hf.snapshot_dir(rt.cfg.repo, hf.TARGETS[rt.cfg.repo])), local_files_only=True
    )


# ------------------------------------------------------------------ training (child process)


@functools.cache
def stagew_script():
    """``scripts/wme_stagew.py`` as a module (its teacher rows, prompt ids, writer names and Row)."""
    spec = importlib.util.spec_from_file_location(
        "wme_stagew_script", ROOT / "scripts/wme_stagew.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def eligible(rows, budget: int) -> list:
    """Stage-W eligibility (``stagew.split_teacher``): gold-verified, non-empty, at most ``budget`` tokens."""
    return [r for r in rows if r["correct"] and r["answer"] and r["n_tokens"] <= budget]


def n_steps(n_rows: int, epochs: int = LORA["epochs"], batch_rows: int = LORA["batch_rows"]) -> int:
    return epochs * math.ceil(n_rows / batch_rows)


def train_job(job: dict) -> dict:
    """Child-process body of ``ste_train`` (the stage-W train phase for one run on all eligible rows)."""
    import numpy as np
    import torch

    from wsp.wm import actions as A
    from wsp.wm import hf
    from wsp.wm import lora as LR
    from wsp.wm import stagew as W
    from wsp.wm import svf as V

    t0 = time.time()
    SW = stagew_script()
    repo, dev = job["repo"], job.get("device", "cuda")
    if job.get("revision"):
        hf.TARGETS.setdefault(repo, job["revision"])
    model, tok, _cfg, base_hash, _ = hf.load_target(repo, dev)
    if job.get("theta0") and base_hash != job["theta0"]:
        raise RuntimeError(f"theta0 {base_hash[:12]} != manifest {job['theta0'][:12]}")
    hf.use_reference_kernels()
    names = SW.writer_names(model)
    frozen = [n for n in names if A.is_frozen(n)]
    if frozen:
        raise RuntimeError(f"frozen tensors among the writers: {frozen[:2]}")
    n_layers = 1 + max(V.writer_key(n)[0] for n in names)
    if len(names) != 2 * n_layers:
        raise RuntimeError(
            f"{len(names)} writers for {n_layers} layers (expected both writers per layer)"
        )
    rows = eligible(SW.teacher_rows(tok), W.BUDGET)
    if job.get("smoke"):
        rows = rows[:SMOKE_ROWS]
    elif len(rows) != N_TEACHER:
        raise RuntimeError(f"{len(rows)} gold-verified teacher rows, registered {N_TEACHER}")
    e = SW.im_end(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rws = []
    for r in rows:
        a = SW.prompt_ids(tok, r["panel"], r["item"])
        rws.append(SW.Row(a + SW.enc(tok, r["answer"]) + [e], len(a)))
    want = n_steps(len(rws))
    if not job.get("smoke") and want != N_STEPS:
        raise RuntimeError(f"{want} optimiser steps, registered {N_STEPS}")
    prog = job.get("progress")

    def on_step(step, adapter):
        if step % 20 == 0 or step == want:
            _progress_line(prog, "ste_train step", step=step)

    ad = LR.LoRA(model, names, rank=LORA["rank"], alpha=LORA["alpha"], seed=SEED)
    log = LR.train(
        model,
        ad,
        rws,
        pad_id=pad,
        lr=LORA["lr"],
        epochs=LORA["epochs"],
        batch_rows=LORA["batch_rows"],
        seed=SEED,
        device=dev,
        max_seconds=1e9,
        warmup=LORA["warmup"],
        checkpointing=True,
        on_step=on_step,
        ce_chunk=1024,
    )
    if not all(math.isfinite(x) for x in log["loss"]):
        raise RuntimeError("non-finite training loss")
    if log["steps"] != want:
        raise RuntimeError(f"{log['steps']} steps, expected {want}")
    out = Path(job["factors_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save({k: (b.float(), a.float(), s) for k, (b, a, s) in ad.factors().items()}, tmp)
    os.replace(tmp, out)
    loss = log["loss"]
    teacher = Path(SW.TEACHER)
    return {
        "steps": log["steps"],
        "seconds": log["seconds"],
        "total_seconds": time.time() - t0,
        "n_rows": log["n_rows"],
        "epochs": log["epochs"],
        "lr": log["lr"],
        "batch_rows": log["batch_rows"],
        "rank": log["rank"],
        "alpha": log["alpha"],
        "seed": log["seed"],
        "init_seed": SEED,
        "warmup": LORA["warmup"],
        "n_writers": len(names),
        "n_layers": n_layers,
        "base_hash": base_hash,
        "loss_first20": float(np.mean(loss[:20])),
        "loss_last20": float(np.mean(loss[-20:])),
        "loss_curve": [float(np.mean(loss[i : i + 4])) for i in range(0, len(loss), 4)],
        "teacher_sha256": sha256_file(teacher),
        "teacher_rows_sha256": hashlib.sha256(
            json.dumps([[r["panel"], r["id"], r["sample"]] for r in rows]).encode()
        ).hexdigest(),
        "factors_path": str(out),
        "factors_sha256": sha256_file(out),
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }


def spawn_train(job: dict, workdir, python=sys.executable, timeout_s: float = 3 * 3600) -> dict:
    import subprocess

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    jp, op = workdir / "train_job.json", workdir / "train_out.json"
    jp.write_text(json.dumps(job))
    p = subprocess.run(
        [python, "-m", "wsp.wm.c1.steops", "train", str(jp), str(op)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    (workdir / "train_child.log").write_text(
        (p.stdout or "")[-20000:] + "\n---\n" + (p.stderr or "")[-20000:]
    )
    if p.returncode != 0:
        tail = (p.stderr or "")[-1500:]
        if "out of memory" in tail.lower():
            raise MemoryError(f"CUDA out of memory in the training child: {tail[-300:]}")
        raise RuntimeError(f"training child exit {p.returncode}: {tail[-600:]}")
    return json.loads(op.read_text())


@register("ste_train")
def h_ste_train(rt, req, ctx):
    a = req.get("args") or {}
    d = ste_dir(rt)
    fp, mp = d / FACTORS, d / "train_meta.json"
    if mp.exists() and fp.exists() and not a.get("force"):
        meta = json.loads(mp.read_text())
        if meta.get("factors_sha256") == sha256_file(fp):
            return dict(meta, resumed=True), {}
    teacher = stagew_script().TEACHER
    if TEACHER_P3.exists() and sha256_file(TEACHER_P3) != sha256_file(teacher):
        raise RuntimeError("the pilot-3 and stage-W teacher files differ")
    rt.release()
    ctx.check()
    job = dict(
        _repo_job(rt),
        factors_path=str(fp),
        theta0=None if rt.cfg.smoke else rt.manifest["target.theta0_hash"],
        progress=_progress_spec(ctx),
    )
    meta = spawn_train(job, Path(ctx.scratch) / "train")
    ctx.check()
    if meta["factors_sha256"] != sha256_file(fp):
        raise RuntimeError("factors file changed after training")
    mp.write_text(json.dumps(meta, sort_keys=True))
    return dict(meta, resumed=False), {}


# ------------------------------------------------------------------ calibration (base only)


def refused(text: str) -> bool:
    """The pilot-3 frozen generated-guard refusal rule (pilot-3 ``score`` on guard panels)."""
    from wsp.wm import genaudit as GA

    return bool(GA.detect_refusal(text))


def calibration_items(rt) -> dict:
    from wsp.wm import p3panel as PP

    if rt.cfg.smoke:  # engineering smoke: the exposed wme-p3h holdout guard items, never wme-c2-gen
        from wsp.wm.c1 import modelops as MO

        hol = PP.read_split(MO.PANEL_P3H, "holdout")
        return {g: hol[g][:8] for g in GUARDS}
    cal = PP.read_split(PANEL_C2, "calibration")
    out = {g: cal[g] for g in GUARDS}
    for g, its in out.items():
        if len(its) != CAL_N:
            raise ValueError(f"{g}: {len(its)} calibration items, registered {CAL_N}")
    return out


def calib_job(base: dict, prompts: dict, progress=None) -> dict:
    blocks = []
    for g in GUARDS:
        blocks.append(
            {
                "name": f"{g}|greedy",
                "prompts": prompts[g],
                "max_tokens": GUARD_TOKENS,
                "state": {"kind": "base"},
            }
        )
        for s in CAL_SEEDS:
            blocks.append(
                {
                    "name": f"{g}|sample|{s}",
                    "prompts": prompts[g],
                    "max_tokens": GUARD_TOKENS,
                    "state": {"kind": "base"},
                    "sample": dict(CAL_SAMPLE, seed=s),
                }
            )
    return dict(base, blocks=blocks, progress=progress)


def margins_from(items: dict, out: dict, manifest: dict) -> dict:
    """Per-item refusal flags -> pi, margins and null pass probabilities (all base-only)."""
    import numpy as np

    from wsp.wm import p3gen as G

    by = {b["name"]: b for b in out["blocks"]}
    gg = manifest["guards.generated"]
    rec = {"margins": {}, "null_pass_prob": {}, "calibration": {}, "null_inputs": {}}
    for g in GUARDS:
        n = len(items[g])
        greedy = [refused(o["text"]) for o in by[f"{g}|greedy"]["outputs"]]
        samp = {
            s: [refused(o["text"]) for o in by[f"{g}|sample|{s}"]["outputs"]] for s in CAL_SEEDS
        }
        if len(greedy) != n or any(len(v) != n for v in samp.values()):
            raise RuntimeError(f"{g}: calibration outputs do not match the {n} items")
        pi = np.mean([samp[s] for s in CAL_SEEDS], axis=0)
        margin = G.guard_margin(pi, **MARGIN)
        disc = max(float(gg["sizing_inputs"]["discordance"][g]), 1.0 / MARGIN["n_cal"])
        n_reg = int(gg["n"][g])
        rec["margins"][g] = margin
        rec["null_pass_prob"][g] = G.null_pass_prob(disc, margin, n_reg)
        rec["null_inputs"][g] = {"discordance": disc, "delta": margin, "n": n_reg}
        rec["calibration"][g] = {
            "ids": [it["id"] for it in items[g]],
            "greedy_refused": greedy,
            "sample_refused": {str(s): v for s, v in samp.items()},
            "pi": [float(x) for x in pi],
            "base_greedy_rate": float(np.mean(greedy)),
            "ids_sha256": {
                k: by[k]["ids_sha256"]
                for k in [f"{g}|greedy"] + [f"{g}|sample|{s}" for s in CAL_SEEDS]
            },
        }
    return rec


@register("ste_calibrate")
def h_ste_calibrate(rt, req, ctx):
    from wsp.wm import p3gen as G
    from wsp.wm.c1 import vllmops as VO

    if (ste_dir(rt) / GENERATED).exists():
        raise PermissionError(
            "edited S-TE generation has started on this worker; margins are frozen"
        )
    items = calibration_items(rt)
    tok = _tokenizer(rt)
    prompts = {g: [G.chat(tok, G.user_prompt(g, it)) for it in items[g]] for g in GUARDS}
    rt.release()
    ctx.check()
    t0 = time.time()
    out = VO.spawn(
        calib_job(_repo_job(rt), prompts, _progress_spec(ctx)), Path(ctx.scratch) / "vllm"
    )
    ctx.check()
    rec = margins_from(items, out, rt.manifest)
    rec.update(
        rule="genaudit.detect_refusal on the greedy/sampled response (pilot-3 frozen guard rule)",
        tokens=GUARD_TOKENS,
        sampling=dict(CAL_SAMPLE, seeds=list(CAL_SEEDS), pi="mean over the 8 samples (pilot-3)"),
        margin_args=dict(MARGIN),
        engine=out.get("engine"),
        vllm_load_s=out.get("load_s"),
        seconds=time.time() - t0,
        computed_at=time.time(),
        smoke=bool(rt.cfg.smoke),
    )
    (ste_dir(rt) / "calibration.json").write_text(json.dumps(rec, sort_keys=True))
    return rec, {}


# ------------------------------------------------------------------ generation (base vs edit)


def score_mc(panel: str, item, o) -> dict:
    """Strict ``p3gen.verify`` (truncated = incorrect) and the flexible re-parse (``scripts/wme_stagew2.py score``)."""
    from wsp.wm import p3gen as G

    v = G.verify(panel, item, o["text"], o["hit_max"])
    flex = None if o["hit_max"] else G.flexible_mc_letter(o["text"][-400:])
    return {
        "correct": bool(v["correct"]),
        "trunc": bool(o["hit_max"]),
        "len": len(o["ids"]),
        "flex_correct": flex == G.LETTERS[item["answer_index"]],
    }


def check_margins(margins) -> dict:
    if not isinstance(margins, dict) or set(GUARDS) - set(margins):
        raise PermissionError(
            "ste_generate needs the base-only calibration margins (run ste_calibrate first)"
        )
    out = {}
    for g in GUARDS:
        x = float(margins[g])
        if not math.isfinite(x) or x < MARGIN["floor"]:
            raise ValueError(
                f"{g}: margin {x} is not a calibrated margin (floor {MARGIN['floor']})"
            )
        out[g] = x
    return out


def panel_items(files: dict, expect: dict | None) -> dict:
    """{objective: [item]} over the uploaded files (each sha256-checked), file order kept."""
    out: dict = {}
    for name in sorted(files):
        f = files[name]
        for r in read_checked_jsonl(f["path"], f["sha256"]):
            out.setdefault(r["objective"], []).append(r)
    for p in BENCHMARKS + GUARDS:
        if p not in out:
            raise ValueError(f"no {p} items in the uploaded panel files")
        ids = [r["id"] for r in out[p]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{p}: duplicate ids")
        if expect is not None and len(ids) != expect[p]:
            raise ValueError(f"{p}: {len(ids)} items, registered {expect[p]}")
    return out


def ste_job(
    base: dict, prompts: dict, factors: str, e1_tensors=None, canary=None, progress=None
) -> dict:
    """The registered block order (§II.6.2, §II.12.2-§II.12.3)."""
    from wsp.wm.c1 import vllmops as VO

    state = {"base": {"kind": "base"}, "edit": {"kind": "lora", "factors": factors, "lam": 1.0}}
    if e1_tensors is not None:
        state["e1_winner"] = {"kind": "tensors", "path": e1_tensors}
    blocks = []
    if canary:
        blocks.append(
            {
                "name": "canary",
                "prompts": canary,
                "max_tokens": VO.CANARY_TOKENS,
                "state": state["base"],
            }
        )
    for b in BENCHMARKS:
        for budget in BUDGETS:
            for m in MODELS:
                blocks.append(
                    {
                        "name": f"{b}|{budget}|{m}",
                        "prompts": prompts[b],
                        "max_tokens": budget,
                        "state": state[m],
                    }
                )
    gm = MODELS + (("e1_winner",) if e1_tensors is not None else ())
    for g in GUARDS:
        for m in gm:
            blocks.append(
                {
                    "name": f"{g}|{GUARD_TOKENS}|{m}",
                    "prompts": prompts[g],
                    "max_tokens": GUARD_TOKENS,
                    "state": state[m],
                }
            )
    if canary:
        blocks.append(
            {
                "name": "canary_end",
                "prompts": canary,
                "max_tokens": VO.CANARY_TOKENS,
                "state": state["base"],
            }
        )
    return dict(base, blocks=blocks, progress=progress)


@register("ste_generate")
def h_ste_generate(rt, req, ctx):
    from wsp.wm import p3gen as G
    from wsp.wm.c1 import modelops as MO
    from wsp.wm.c1 import vllmops as VO

    a = req["args"]
    margins = check_margins(a.get("margins"))
    cal = ste_dir(rt) / "calibration.json"
    if cal.exists():
        local = json.loads(cal.read_text())["margins"]
        if any(float(local[g]) != margins[g] for g in GUARDS):
            raise ValueError("margins differ from this worker's recorded base-only calibration")
    fac = a["lora_factors"]
    if sha256_file(fac) != a["lora_factors_sha256"]:
        raise ValueError("LoRA factors sha256 mismatch")
    e1 = a.get("e1_winner_tensors")
    if e1 is not None and sha256_file(e1["path"]) != e1["sha256"]:
        raise ValueError("E1 winner tensors sha256 mismatch")
    expect = None
    if not rt.cfg.smoke:
        bn = rt.manifest["panel.wme_c2_gen"]["built_n"]
        expect = {p: int(bn[p]["sealed_test"]) for p in BENCHMARKS + GUARDS}
    items = panel_items(a["panel_files"], expect)
    tok = _tokenizer(rt)
    prompts = {
        p: [G.chat(tok, G.user_prompt(p, it)) for it in items[p]] for p in BENCHMARKS + GUARDS
    }
    canary = None
    if a.get("canary", True):
        canary = [G.chat(tok, G.user_prompt("mmlupro_gen", it)) for it in MO.gen_canary_items()]
    job = ste_job(
        _repo_job(rt),
        prompts,
        str(fac),
        None if e1 is None else str(e1["path"]),
        canary,
        _progress_spec(ctx),
    )
    (ste_dir(rt) / GENERATED).write_text(json.dumps({"margins": margins, "t": time.time()}))
    rt.release()
    ctx.check()
    t0 = time.time()
    out = VO.spawn(job, Path(ctx.scratch) / "vllm", timeout_s=8 * 3600)
    ctx.check()
    by = {b["name"]: b for b in out["blocks"]}
    if [b["name"] for b in out["blocks"]] != [b["name"] for b in job["blocks"]]:
        raise RuntimeError("vLLM child returned blocks out of order")
    rows, grows, raw, trunc, blocks = [], [], [], {}, []
    for jb in job["blocks"]:
        ob = by[jb["name"]]
        blocks.append(
            {
                "name": jb["name"],
                "seconds": ob["seconds"],
                "ids_sha256": ob["ids_sha256"],
                "n": len(ob["outputs"]),
            }
        )
        if jb["name"].startswith("canary"):
            continue
        panel, budget, model = jb["name"].split("|")
        its = items[panel]
        if len(ob["outputs"]) != len(its):
            raise RuntimeError(f"{jb['name']}: {len(ob['outputs'])} outputs for {len(its)} items")
        for it, o in zip(its, ob["outputs"], strict=True):
            raw.append(
                {
                    "block": jb["name"],
                    "id": it["id"],
                    "text": o["text"],
                    "n_tokens": len(o["ids"]),
                    "hit_max": bool(o["hit_max"]),
                }
            )
            if panel in GUARDS:
                grows.append(
                    {"panel": panel, "model": model, "id": it["id"], "refused": refused(o["text"])}
                )
            else:
                rows.append(
                    dict(
                        {"benchmark": panel, "model": model, "budget": int(budget), "id": it["id"]},
                        **score_mc(panel, it, o),
                    )
                )
        if panel in BENCHMARKS:
            trunc[jb["name"]] = sum(bool(o["hit_max"]) for o in ob["outputs"]) / len(its)
    sc = Path(ctx.scratch)
    files = {OUTPUTS: sc / OUTPUTS, GUARD_ROWS: sc / GUARD_ROWS, RAW: sc / RAW}
    sha = {
        OUTPUTS: write_jsonl_gz(files[OUTPUTS], rows),
        GUARD_ROWS: write_jsonl_gz(files[GUARD_ROWS], grows),
        RAW: write_jsonl_gz(files[RAW], raw),
    }
    payload = {
        "n": {p: len(items[p]) for p in BENCHMARKS + GUARDS},
        "margins": margins,
        "truncation_rate": trunc,
        "blocks": blocks,
        "canary": None
        if canary is None
        else {
            "start": by["canary"]["ids_sha256"],
            "end": by["canary_end"]["ids_sha256"],
            "repeat_equal": by["canary"]["ids_sha256"] == by["canary_end"]["ids_sha256"],
        },
        "lora_factors_sha256": a["lora_factors_sha256"],
        "e1_winner_tensors_sha256": None if e1 is None else e1["sha256"],
        "panel_sha256": {k: v["sha256"] for k, v in a["panel_files"].items()},
        "engine": out.get("engine"),
        "engine_config": VO.engine_config(),
        "vllm_load_s": out.get("load_s"),
        "vllm_seconds": out.get("seconds"),
        "seconds": time.time() - t0,
        "files": sha,
    }
    return payload, files


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "train":
        raise SystemExit("usage: python -m wsp.wm.c1.steops train <job.json> <out.json>")
    res = train_job(json.loads(Path(sys.argv[2]).read_text()))
    Path(sys.argv[3]).write_text(json.dumps(res))
