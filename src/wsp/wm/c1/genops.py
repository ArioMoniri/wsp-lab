"""E1 generation audit on the worker (PREREG_WME Part I §6.7; Part II §II.4 delta 4, §II.12.3, §II.13 items 5-6).

Handler ``e1_gen`` (args: ``items_file``, ``items_sha256``, ``winner`` = {"bundles": :func:`serialise_bundles`
output, "theta_hash", optional "candidate"}, ``decodings`` = ["greedy"] or ["sampled:<seed>", ...], optional
``save_tensors``):

* **Items.** The uploaded wme-c1 ``sealed_gen`` TriviaQA file (sha256 checked before parsing), in file order. The
  prompts are rebuilt from the frozen templates (``wm.p2data`` NEUTRAL_TEMPLATE / SUGGESTED_TEMPLATE with the
  item's ``incorrect_answer``, the ``gen.pool`` rule that ``wm.triviaqa.items_from_rows`` applied at the build) and
  must equal the prompts the builder stored on the item.
* **Decoding** (``gen.config``). Hugging Face ``generate`` through ``genaudit.generate_greedy`` /
  ``generate_sampled`` (non-thinking chat template, left padding), batch 128, 128 new tokens; sampled: T 0.7,
  top-p 1.0, one of the four ``gen.sampled_seeds``.
* **Pairing** (§II.13 item 5). One resident model: for every decoding, the base model (neutral, then suggested) and
  then the winner (applied through ``EditSession.apply_bundle`` per bundle, theta-hash asserted equal to the
  winner's, restored and asserted at theta0 afterwards) on the same prompts, batch and order.
* **Output.** ``gen_e1_outputs.jsonl.gz`` (rows of ``wm.c1.schema`` gen_e1 outputs) and ``gen_e1_items.jsonl.gz``
  (id, group, aliases, incorrect); the payload carries counts, seconds and sha256s only (no scores).

Optionally (``save_tensors``) the winner's realised bf16 tensors are saved, keyed by writer (layer, role) or
``embed``, for the vLLM ``tensors`` state (§II.12.3 generated safety of the E1 winner, run by ``ste_generate``).
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import sys
import time
from pathlib import Path

OUTPUTS = "gen_e1_outputs.jsonl.gz"
ITEMS = "gen_e1_items.jsonl.gz"
OBJECTIVE = "triviaqa_gen"
PROMPTS = ("neutral", "suggested")
MODELS = ("base", "winner")


# ------------------------------------------------------------------ registration


def register(name):
    """Add a handler to ``wsp.wm.c1.serve.HANDLERS``. Under ``python -m wsp.wm.c1.serve`` the running module is
    ``__main__``, a second copy whose ``HANDLERS`` dict is the one ``main()`` serves, so the entry is mirrored there."""
    from wsp.wm.c1 import serve as S

    def deco(fn):
        S.HANDLERS[name] = fn
        main = sys.modules.get("__main__")
        spec = getattr(main, "__spec__", None)
        if main is not S and spec is not None and spec.name == "wsp.wm.c1.serve":
            main.HANDLERS[name] = fn
        return fn

    return deco


# ------------------------------------------------------------------ files


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_checked_jsonl(path, sha256: str) -> list:
    """The uploaded file's rows in file order, after its sha256 matches the controller's."""
    data = Path(path).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise ValueError(f"{path}: sha256 {got} != expected {sha256}")
    return [json.loads(x) for x in data.decode().splitlines() if x.strip()]


def write_jsonl_gz(path, rows) -> str:
    """Deterministic gzip JSON lines (mtime 0, no stored name); returns the sha256 of the file."""
    path = Path(path)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        for r in rows:
            f.write((json.dumps(r, sort_keys=True) + "\n").encode())
    return sha256_file(path)


def read_jsonl_gz(path) -> list:
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


# ------------------------------------------------------------------ action serialisation


def _families() -> dict:
    from wsp.wm import actions as A
    from wsp.wm import lora as LR

    return {
        "scale": A._scale,
        "row_scale": A._row_scale,
        "proj_remove": A._proj_remove,
        "rank_one": A._rank_one,
        "noise": A._noise,
        "additive": A._additive,
        "lora_task_vector": LR._lowrank,
    }


def _enc(v):
    import torch

    if isinstance(v, torch.Tensor):
        t = v.detach().contiguous().cpu()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "__tensor__": base64.b64encode(raw).decode(),
            "dtype": str(t.dtype).removeprefix("torch."),
            "shape": list(t.shape),
        }
    if isinstance(v, tuple):
        return {"__tuple__": [_enc(x) for x in v]}
    if isinstance(v, list):
        return [_enc(x) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise TypeError(f"cannot serialise an action parameter of type {type(v).__name__}")


def _dec(v):
    import torch

    if isinstance(v, dict):
        if "__tensor__" in v:
            dtype = getattr(torch, v["dtype"])
            raw = base64.b64decode(v["__tensor__"])
            if not raw:
                return torch.empty(v["shape"], dtype=dtype)
            return (
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                .view(dtype)
                .reshape(v["shape"])
                .clone()
            )
        if "__tuple__" in v:
            return tuple(_dec(x) for x in v["__tuple__"])
        raise TypeError(f"unknown encoded parameter {sorted(v)}")
    if isinstance(v, list):
        return [_dec(x) for x in v]
    return v


def serialise_action(a) -> dict:
    if a.family not in _families():
        raise KeyError(f"unknown action family {a.family!r}")
    return {
        "family": a.family,
        "tensor": a.tensor,
        "dose": float(a.dose),
        "params": [[k, _enc(v)] for k, v in a.params],
        "key": a.key(),
    }


def deserialise_action(obj: dict):
    from wsp.wm import actions as A

    fam = _families()
    if obj["family"] not in fam:
        raise KeyError(f"unknown action family {obj['family']!r}")
    a = A.Action(
        obj["family"],
        obj["tensor"],
        float(obj["dose"]),
        tuple((k, _dec(v)) for k, v in obj["params"]),
        fam[obj["family"]],
    )
    if "key" in obj and a.key() != obj["key"]:
        raise ValueError(
            f"action key mismatch after deserialisation ({obj['family']} {obj['tensor']})"
        )
    return a


def serialise_bundles(bundles) -> list:
    """[[Action]] (bundles applied in order, each through ``EditSession.apply_bundle``) -> JSON-able lists."""
    return [[serialise_action(a) for a in b] for b in bundles]


def deserialise_bundles(obj) -> list:
    return [[deserialise_action(a) for a in b] for b in obj]


def bundles_digest(obj) -> str:
    """sha256 over the ordered action keys of serialised bundles (identity of the edit, not of its bytes)."""
    keys = [[a["key"] if "key" in a else deserialise_action(a).key() for a in b] for b in obj]
    return hashlib.sha256(json.dumps(keys).encode()).hexdigest()


# ------------------------------------------------------------------ items and prompts


def item_id(it) -> str:
    """The wme-c1 TriviaQA rows carry ``question_id`` (``wm.triviaqa``); a row with an ``id`` keeps it."""
    return str(it["id"]) if "id" in it else str(it["question_id"])


def prompts_for(it) -> dict:
    """The verbatim neutral and suggested-incorrect user prompts (``wm.p2data`` templates, ``gen.pool`` incorrect
    answer). Raises if they differ from the prompts the panel builder stored on the item."""
    from wsp.wm import p2data as D

    q = it["question"]
    out = {
        "neutral": D.NEUTRAL_TEMPLATE.format(question=q),
        "suggested": D.SUGGESTED_TEMPLATE.format(
            question=q, incorrect_answer=it["incorrect_answer"]
        ),
    }
    for p in PROMPTS:
        stored = it.get(f"{p}_prompt")
        if stored is not None and stored != out[p]:
            raise ValueError(f"item {item_id(it)}: {p} prompt differs from the builder's template")
    return out


def load_items(path, sha256: str, expect_n: int | None) -> list:
    items = read_checked_jsonl(path, sha256)
    bad = [item_id(it) for it in items if it.get("objective", OBJECTIVE) != OBJECTIVE]
    if bad:
        raise ValueError(f"{len(bad)} items are not {OBJECTIVE} (e.g. {bad[0]})")
    ids = [item_id(it) for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item ids in the generation items")
    if expect_n is not None and len(items) != expect_n:
        raise ValueError(f"{len(items)} generation items, registered {expect_n}")
    return items


def item_rows(items) -> list:
    return [
        {
            "id": item_id(it),
            "group": it.get("group"),
            "aliases": list(it["aliases"]),
            "incorrect": it["incorrect_answer"],
        }
        for it in items
    ]


# ------------------------------------------------------------------ decoding


def gen_config(manifest: dict) -> dict:
    g = manifest["gen.config"]
    return {
        "batch": int(g["batch"]),
        "max_new_tokens": int(g["max_new_tokens"]["sycophancy_real"]),
        "sampled": dict(g["sampled_audit"]),
        "seeds": [int(s) for s in manifest["gen.sampled_seeds"]],
        "decoding": dict(g["decoding"]),
    }


def parse_decoding(d: str, seeds) -> int | None:
    """ "greedy" -> None; "sampled:<seed>" -> seed (one of ``gen.sampled_seeds``)."""
    if d == "greedy":
        return None
    if d.startswith("sampled:"):
        s = int(d.split(":", 1)[1])
        if s not in seeds:
            raise ValueError(f"sampled seed {s} is not one of gen.sampled_seeds {list(seeds)}")
        return s
    raise ValueError(f"unknown decoding {d!r}")


def generate(m, prompts, cfg, seed, mode=None):
    """HF generation through the recorded engine (``wm.c1.genengine``; §II.31 C1(a))."""
    from wsp.wm.c1 import genengine as GE

    sample = None if seed is None else dict(cfg["sampled"], seed=int(seed))
    return GE.hf_generate(
        m.model,
        m.tok,
        prompts,
        cfg["max_new_tokens"],
        cfg["batch"],
        m.device,
        sample=sample,
        mode=mode,
    )


def winner_tensors(m) -> dict:
    """The realised tensors of the current (winner) state: writer (layer, role) -> bf16 CPU tensor; the embedding as
    {"rows", "values"} over its changed rows (the vLLM ``tensors`` state). Any other tensor raises."""
    from wsp.wm import svf as V

    out = {}
    for name in m.sess.touched:
        live = m.sess.params[name].detach()
        if name == m.embed:
            orig = m.sess.original(name).to(live.device)
            rows = (live != orig).any(dim=1).nonzero().flatten().tolist()
            out["embed"] = {"rows": rows, "values": live[rows].cpu().clone()}
            continue
        k = V.writer_key(name)
        if k is None:
            raise KeyError(
                f"{name}: the vLLM tensors state takes residual writers and the embedding only"
            )
        out[k] = live.cpu().clone()
    if not out:
        raise ValueError("the winner touches no tensor")
    return out


# ------------------------------------------------------------------ handler


@register("e1_gen")
def h_e1_gen(rt, req, ctx):
    a = req["args"]
    cfg = gen_config(rt.manifest)
    decodings = list(a["decodings"])
    if not decodings or len(set(decodings)) != len(decodings):
        raise ValueError("decodings must be a non-empty list without repeats")
    seeds = [parse_decoding(d, cfg["seeds"]) for d in decodings]
    expect = (
        None if rt.cfg.smoke else int(rt.manifest["panel.wme_c1.sizes"][OBJECTIVE]["sealed_gen"])
    )
    items = load_items(a["items_file"], a["items_sha256"], expect)
    prompts = {p: [] for p in PROMPTS}
    for it in items:
        for p, text in prompts_for(it).items():
            prompts[p].append(text)
    ids = [item_id(it) for it in items]
    win = a["winner"]
    bundles = deserialise_bundles(win["bundles"])
    if not bundles or not all(bundles):
        raise ValueError("the winner has no actions")
    from wsp.wm.c1 import genengine as GE

    t_all = time.time()
    if GE.GEN_ENGINE == "vllm":  # §II.31 C1(b), inactive unless selected at the Phase L re-run
        rows, blocks, tensors, theta0 = _e1_vllm(
            rt, ctx, a, cfg, decodings, seeds, prompts, ids, win, bundles
        )
    else:
        rows, blocks, tensors, theta0 = _e1_hf(
            rt, ctx, a, cfg, decodings, seeds, prompts, ids, win, bundles
        )
    return _e1_payload(rt, ctx, a, cfg, items, decodings, win, rows, blocks, tensors, theta0, t_all)


def _e1_hf(rt, ctx, a, cfg, decodings, seeds, prompts, ids, win, bundles):
    from wsp.wm.c1 import modelops as MO

    m = rt.hf()
    m.at_base()
    rows, blocks, tensors = [], [], None
    for dec, seed in zip(decodings, seeds, strict=True):
        for model in MODELS:
            ctx.check()
            try:
                if model == "winner":
                    for b in bundles:
                        m.sess.apply_bundle(b)
                    th = m.theta_hash()
                    if th != win["theta_hash"]:
                        raise RuntimeError(
                            f"winner theta-hash {th[:12]} != designated {win['theta_hash'][:12]}"
                        )
                    if a.get("save_tensors") and tensors is None:
                        tensors = _save_tensors(rt, m, th)
                else:
                    m.at_base()
                    th = m.base_hash
                for p in PROMPTS:
                    t0 = time.time()
                    out = generate(m, prompts[p], cfg, seed)
                    if len(out) != len(ids):
                        raise RuntimeError(f"{len(out)} generations for {len(ids)} prompts")
                    for i, o in zip(ids, out, strict=True):
                        rows.append(
                            {
                                "model": model,
                                "prompt": p,
                                "decoding": dec,
                                "id": i,
                                "ids": [int(x) for x in o["ids"]],
                                "text": o["text"],
                                "hit_max": bool(o["hit_max"]),
                            }
                        )
                    blocks.append(
                        {
                            "model": model,
                            "prompt": p,
                            "decoding": dec,
                            "theta_hash": th,
                            "n": len(out),
                            "seconds": time.time() - t0,
                            "ids_sha256": MO.sha_ids([o["ids"] for o in out]),
                        }
                    )
                    ctx.progress("e1_gen block", model=model, prompt=p, decoding=dec)
                    ctx.check()
            finally:
                if model == "winner":
                    m.restore()
    m.at_base()
    return rows, blocks, tensors, m.base_hash


def _e1_vllm(rt, ctx, a, cfg, decodings, seeds, prompts, ids, win, bundles):
    """§II.31 C1(b): the winner is realised on the resident HF model (theta-hash asserted), its exact bf16 tensors
    saved, the model released; one vLLM child job then generates every (decoding, model, prompt) block at the
    registered settings (``genengine.vllm_block``), the winner as the ``tensors`` state."""
    from wsp.wm import hf
    from wsp.wm.c1 import genengine as GE
    from wsp.wm.c1 import vllmops as VO

    m = rt.hf()
    m.at_base()
    try:
        for b in bundles:
            m.sess.apply_bundle(b)
        th = m.theta_hash()
        if th != win["theta_hash"]:
            raise RuntimeError(
                f"winner theta-hash {th[:12]} != designated {win['theta_hash'][:12]}"
            )
        saved = _save_tensors(rt, m, th)
    finally:
        m.restore()
    m.at_base()
    theta0, tok = m.base_hash, m.tok
    rt.release()
    job = {"repo": rt.cfg.repo, "smoke": rt.cfg.smoke, "blocks": []}
    if rt.cfg.smoke:
        job["revision"] = hf.TARGETS[rt.cfg.repo]
    keys = []
    for dec, seed in zip(decodings, seeds, strict=True):
        sample = None if seed is None else dict(cfg["sampled"], seed=int(seed))
        for model in MODELS:
            state = (
                {"kind": "base"} if model == "base" else {"kind": "tensors", "path": saved["path"]}
            )
            for p in PROMPTS:
                job["blocks"].append(
                    GE.vllm_block(
                        f"{dec}|{model}|{p}", tok, prompts[p], cfg["max_new_tokens"], sample, state
                    )
                )
                keys.append((dec, model, p, th if model == "winner" else theta0))
    ctx.check()
    out = VO.spawn(job, Path(ctx.scratch) / "vllm_e1")
    rows, blocks = [], []
    for (dec, model, p, th_), blk in zip(keys, out["blocks"], strict=True):
        res = blk["outputs"]
        if len(res) != len(ids):
            raise RuntimeError(f"{len(res)} generations for {len(ids)} prompts")
        for i, o in zip(ids, res, strict=True):
            rows.append(
                {
                    "model": model,
                    "prompt": p,
                    "decoding": dec,
                    "id": i,
                    "ids": [int(x) for x in o["ids"]],
                    "text": o["text"],
                    "hit_max": bool(o["hit_max"]),
                }
            )
        blocks.append(
            {
                "model": model,
                "prompt": p,
                "decoding": dec,
                "theta_hash": th_,
                "n": len(res),
                "seconds": blk["seconds"],
                "ids_sha256": blk["ids_sha256"],
            }
        )
    return rows, blocks, (saved if a.get("save_tensors") else None), theta0


def _e1_payload(rt, ctx, a, cfg, items, decodings, win, rows, blocks, tensors, theta0, t_all):
    from wsp.wm.c1 import genengine as GE

    out_p = Path(ctx.scratch) / OUTPUTS
    items_p = Path(ctx.scratch) / ITEMS
    payload = {
        "n_items": len(items),
        "n_rows": len(rows),
        "decodings": decodings,
        "items_sha256": a["items_sha256"],
        "winner": {
            "candidate": win.get("candidate"),
            "theta_hash": win["theta_hash"],
            "bundles_digest": bundles_digest(win["bundles"]),
        },
        "theta0": theta0,
        "blocks": blocks,
        "seconds": time.time() - t_all,
        "gen_engine": GE.config(),
        "config": {
            "engine": GE.config()["engine"],
            "batch": cfg["batch"],
            "padding": "left",
            "max_new_tokens": cfg["max_new_tokens"],
            "decoding": cfg["decoding"],
            "sampled": {k: cfg["sampled"][k] for k in ("temperature", "top_p")},
            "templates": "wm.p2data NEUTRAL_TEMPLATE / SUGGESTED_TEMPLATE",
            "order": "file order (salted rank order of the panel)",
        },
        "files": {
            OUTPUTS: write_jsonl_gz(out_p, rows),
            ITEMS: write_jsonl_gz(items_p, item_rows(items)),
        },
    }
    if tensors is not None:
        payload["winner_tensors"] = tensors
    return payload, {OUTPUTS: out_p, ITEMS: items_p}


def _save_tensors(rt, m, theta_hash: str) -> dict:
    import torch

    d = Path(rt.cfg.state) / "e1"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"winner_tensors_{theta_hash[:16]}.pt"
    buf = io.BytesIO()
    torch.save(winner_tensors(m), buf)
    p.write_bytes(buf.getvalue())
    return {"path": str(p), "sha256": hashlib.sha256(buf.getvalue()).hexdigest()}
