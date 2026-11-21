"""Worker entry point: ``python -m wsp.wm.c1.serve --root <dir> [--smoke]`` (runs on the Lambda instance; the smoke
mode runs locally on Qwen3.5-0.8B with exposed wme-p2 items only, §II.13 item 10).

The worker holds at most one resident engine: the Hugging Face model (teacher-forced scoring, HF generation, R(E))
or nothing while a vLLM child runs. Handlers are registered by request kind; each returns (payload, files).
Large artefacts (field, kept factors, R cache) stay in the worker's ``state/`` directory; payloads carry their
digests. Nothing here reads a sealed split except the sealed-read handlers, which the controller issues only after
the winners manifest is frozen (and which read only the files the controller uploads for that read).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from wsp.wm.c1 import modelops as MO
from wsp.wm.c1 import worker as W

ROOT = MO.ROOT


@dataclasses.dataclass
class Config:
    root: Path
    smoke: bool = False
    repo: str = MO.REPO
    roots_dir: Path = ROOT / "data/wme/bank_roots/roots-9b-20261117"
    shards: Path = ROOT / "docs/reviews/wme/wme-c1-shards-2026-11-19.json"
    bank_panel: Path = MO.PANEL_C1
    vllm: bool = True

    @classmethod
    def make(cls, root, smoke=False, vllm=True):
        c = cls(Path(root), smoke=smoke, vllm=vllm)
        if smoke:
            from huggingface_hub import constants

            from wsp.wm import hf

            snaps = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-0.8B" / "snapshots"
            hf.TARGETS.setdefault(
                MO.SMOKE_REPO, min(snaps.iterdir()).name
            )  # engineering smoke only
            c.repo = MO.SMOKE_REPO
            c.roots_dir = ROOT / "data/wme/bank_roots/smoke"
            c.bank_panel = MO.PANEL_P2
        return c

    @property
    def state(self) -> Path:
        p = self.root / "state"
        p.mkdir(parents=True, exist_ok=True)
        return p


class Runtime:
    """Lazily created resident objects (model, field, R cache, bank world)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.m = None
        self._field = None
        self._rc = None
        self._bank = None
        self.manifest = MO.manifest()

    # -------------------------------------------------- engines
    def hf(self):
        if self.m is None:
            self.m = MO.ModelCtx(self.cfg.repo, "cuda", check_theta0=not self.cfg.smoke)
        return self.m

    def release(self):
        if self.m is not None:
            self.m.release()
        self.m = None
        self._rc = self._bank = None
        if self._field is not None:
            self._field._kept.clear()
            self._field._grad.clear()

    # -------------------------------------------------- data
    def construction(self) -> dict:
        from wsp.wm import panel2 as P2

        return {
            r["hash"]: r
            for v in P2.read_open(self.cfg.bank_panel, "construction").values()
            for r in v
        }

    def shard_items(self):
        """Field items per objective and the e2 shards (production: the committed designation; smoke: the first
        items of the exposed wme-p2 construction split)."""
        from wsp.wm import panel2 as P2

        cons = P2.read_open(self.cfg.bank_panel, "construction")
        if self.cfg.smoke:
            first = lambda o, n: sorted(cons[o], key=lambda r: r["id"])[:n]
            field = {o: first(o, 2) for o in MO.CANARY_TF_OBJECTIVES if o != "mathcomp_gold_lp"}
            e2 = first("sycophancy_real", 8)
            src = {
                "gsm": first("gsm8k_gold_lp", 8),
                "wiki": first("wikitext_nll", 8),
                "mathcomp": first("mathcomp_gold_lp", 8),
                "harmless": first("refusal_margin_harmless", 8),
            }
            return field, e2, src
        rec = json.loads(self.cfg.shards.read_text())
        by = {r["hash"]: r for v in cons.values() for r in v}
        field = {o: [by[x["hash"]] for x in v] for o, v in rec["field"].items()}
        e2 = [by[x["hash"]] for x in rec["e2"]]
        src = {k: [by[x["hash"]] for x in v] for k, v in rec["e2_sources"].items()}
        return field, e2, src

    def bank_items(self) -> dict:
        from wsp.wm import panel2 as P2

        split = "validation" if self.cfg.smoke else "bank"
        its = P2.read_open(self.cfg.bank_panel, split)
        objs = (
            "wikitext_nll",
            "gsm8k_gold_lp",
            "mmlupro_gold_lp",
            "refusal_margin_harmful",
            "refusal_margin_harmless",
            "sycophancy_real",
        )
        if self.cfg.smoke:
            return {o: sorted(its[o], key=lambda r: r["id"])[:4] for o in objs}
        return {o: its[o] for o in objs}

    def scored_hashes(self) -> tuple:
        from wsp.wm import panel2 as P2

        out = set()
        for split in ("validation", "dev") if self.cfg.smoke else ("bank", "dev"):
            try:
                for v in P2.read_open(self.cfg.bank_panel, split).values():
                    out |= {r["hash"] for r in v}
            except FileNotFoundError:
                pass
        if not self.cfg.smoke:
            out |= set(
                json.loads((self.cfg.bank_panel / "manifest.json").read_text())["sealed_hashes"]
            )
        return tuple(sorted(out))

    def roots(self):
        import torch

        rec = json.loads((self.cfg.roots_dir / "roots.json").read_text())
        dirs = torch.load(
            self.cfg.roots_dir / "directions.pt", map_location="cpu", weights_only=False
        )
        return rec, dirs

    def declared_rows(self, m):
        from wsp.wm import vocabfield as VF

        rec, _ = self.roots()
        emb = {r["id"]: list(r["rows"]) for r in rec["roots"] if r["family"] == "row_scale"}
        lo, hi = self.manifest["controls.inert"]["I2_rows"]
        if not self.cfg.smoke:
            emb["inert:I2"] = list(range(int(lo), int(hi)))
        vocab = m.rm.vocab
        return VF.declare_rows({"lm_head": {}, "embed": emb}, vocab)

    # -------------------------------------------------- field and R
    def field_dir(self) -> Path:
        d = self.cfg.state / "field"
        d.mkdir(exist_ok=True)
        return d

    def field(self):
        from wsp.wm.c1 import field as F

        if self._field is None:
            self._field = F.Field(self.field_dir())
        return self._field

    def rcache(self):
        from wsp.wm.c1 import field as F

        if self._rc is None:
            self._rc = F.RCache(self.hf(), self.field())
        return self._rc

    def bank(self, check):
        from wsp.wm.c1 import bankops as B

        if self._bank is None:
            rec, dirs = self.roots()
            self._bank = B.BankWorld(
                self.hf(),
                self.bank_items(),
                self.rcache(),
                rec,
                dirs,
                self.manifest["tap.seed"],
                check,
            )
        self._bank.check = check
        return self._bank


# ------------------------------------------------------------------ handlers


def h_hello(rt: Runtime, req, ctx):
    from wsp.wm.c1 import genengine as GE

    m = rt.hf()
    return (
        {
            "gen_engine": GE.config(),
            "cublas_workspace_config": os.environ.get(GE.CUBLAS_ENV),
            "global_state": GE.global_state(),  # §II.31.9 N4 (b): the worker's process globals at start (R11)
            "code_sha": (MO.ROOT / "CODE_SHA").read_text().strip()
            if (MO.ROOT / "CODE_SHA").exists()
            else None,
            "theta0": m.base_hash,
            "repo": rt.cfg.repo,
            "load_s": m.load_s,
            "versions": MO.versions(),
            "physical": MO.physical_identity(),
            "smoke": rt.cfg.smoke,
        },
        {},
    )


def h_canary(rt: Runtime, req, ctx):
    """Per-engine canaries (``genengine.canary_engines``): ``tf``; ``hf_gen`` (C1(a)) or, under the C1(b) engine,
    ``vllm_gen`` (a second vLLM block: the 64 items at the E1 generation settings); ``vllm`` (stage-W, 2,048).

    RULING-wme-c1-bank-halt K1: the worker first releases all its GPU state (model, field and R-cache tensors);
    the teacher-forced and HF canaries then run in a fresh child process (``wm.c1.hfchild``), the vLLM ones in the
    vLLM child as before. No canary shares a process with bank-phase tensors."""
    from wsp.wm.c1 import genengine as GE
    from wsp.wm.c1 import hfchild as HC
    from wsp.wm.c1 import vllmops as VO

    vllm_gen = GE.GEN_ENGINE == "vllm"
    if vllm_gen and not rt.cfg.vllm:
        raise RuntimeError(
            "generation engine 'vllm' needs the vLLM engine (worker started with --no-vllm)"
        )
    rt.release()
    t0 = time.time()
    child = HC.spawn(HC.job(rt, [{"op": "canary", "hf": not vllm_gen}]), ctx.scratch / "hfchild")
    eng = dict(child["canary"])
    ctx.check()
    t1 = time.time()
    detail = {"tf_hf_seconds": t1 - t0, "gen_engine": GE.GEN_ENGINE, "child": True}
    if rt.cfg.vllm:
        prompts = _gen_canary_prompts(rt)
        job = VO.canary_job(rt.cfg.repo, prompts, smoke=rt.cfg.smoke)
        if vllm_gen:
            job["blocks"].append(
                GE.vllm_block("vllm_gen", _canary_tokenizer(rt), _gen_canary_user_prompts())
            )
        out = VO.spawn(job, ctx.scratch / "vllm")
        eng["vllm"] = out["blocks"][0]["ids_sha256"]
        if vllm_gen:
            eng["vllm_gen"] = out["blocks"][1]["ids_sha256"]
        detail["vllm_seconds"] = out["seconds"]
    return {"engines": eng, "detail": detail}, {}


def _canary_tokenizer(rt: Runtime):
    from transformers import AutoTokenizer

    from wsp.wm import hf

    return AutoTokenizer.from_pretrained(
        str(hf.snapshot_dir(rt.cfg.repo, hf.TARGETS[rt.cfg.repo])), local_files_only=True
    )


def _gen_canary_user_prompts():
    from wsp.wm import p3gen as G

    return [G.user_prompt("mmlupro_gen", it) for it in MO.gen_canary_items()]


def _gen_canary_prompts(rt: Runtime):
    from transformers import AutoTokenizer

    from wsp.wm import hf
    from wsp.wm import p3gen as G

    tok = AutoTokenizer.from_pretrained(
        str(hf.snapshot_dir(rt.cfg.repo, hf.TARGETS[rt.cfg.repo])), local_files_only=True
    )
    return [G.chat(tok, G.user_prompt("mmlupro_gen", it)) for it in MO.gen_canary_items()]


def h_field_group(rt: Runtime, req, ctx):
    from wsp.wm import collect as C
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2
    from wsp.wm import scorers as Sc
    from wsp.wm.c1 import field as F

    g = int(req["args"]["group"])
    m = rt.hf()
    m.at_base()
    field_items, _, _ = rt.shard_items()
    scored = rt.scored_hashes()
    panels = {
        f"construction:{o}": Sc.field_panel(f"construction:{o}", its, scored)
        for o, its in field_items.items()
    }
    objs = [o for o in F.T2_OBJECTIVES if o in field_items]
    specs = P2.objective_specs(objs, panels, P.PanelConfig())
    n_layers = m.rm.n_layers
    layers = [L for L in F.group_layers(g) if L < n_layers]
    if not layers:
        return {"group": g, "skipped": "no layers"}, {}
    mods = [F.module_of(m.rm.name(L, r)) for L in layers for r in ("down", "o")]
    d = rt.field_dir() / f"g{g}"
    if d.exists():
        import shutil

        shutil.rmtree(d)
    t0 = time.time()
    C.collect_grouped(
        m.model, m.tok, specs, panels, [mods], "cuda", d, fisher=True, theta_hash=m.base_hash
    )
    ctx.check()
    t1 = time.time()
    kept = F.compute_kept(d, "cuda")
    ctx.check()
    man = json.loads((d / "manifest.json").read_text())
    rt._field = None
    rt._rc = None
    return {
        "group": g,
        "modules": mods,
        "manifest_sha256": hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest(),
        "file_sha256": man["file_sha256"],
        "kept_sha256": kept["sha256"],
        "kept_ranks": {
            k: {"A": v["A"]["rank"], "G": v["G"]["rank"]} for k, v in kept["audits"].items()
        },
        "collect_seconds": t1 - t0,
        "kept_seconds": time.time() - t1,
    }, {}


def h_field_vocab(rt: Runtime, req, ctx):
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2
    from wsp.wm import scorers as Sc
    from wsp.wm import vocabfield as VF
    from wsp.wm.c1 import field as F

    m = rt.hf()
    m.at_base()
    field_items, _, _ = rt.shard_items()
    scored = rt.scored_hashes()
    panels = {
        f"construction:{o}": Sc.field_panel(f"construction:{o}", its, scored)
        for o, its in field_items.items()
    }
    objs = [o for o in F.T2_OBJECTIVES if o in field_items]
    specs = P2.objective_specs(objs, panels, P.PanelConfig())
    d = rt.field_dir() / "vocab"
    t0 = time.time()
    VF.collect_vocab_rows(
        m.model,
        m.tok,
        specs,
        panels,
        rt.declared_rows(m),
        "cuda",
        d,
        embed=m.embed.rsplit(".", 1)[0],
        theta_hash=m.base_hash,
    )
    rt._field = None
    rt._rc = None
    man = json.loads((d / "manifest.json").read_text())
    return {
        "manifest_sha256": hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest(),
        "file_sha256": man["file_sha256"],  # data-level hashes (the manifest embeds wall times; K3)
        "seconds": time.time() - t0,
    }, {}


def h_bank_base(rt: Runtime, req, ctx):
    return rt.bank(ctx.check).base(), {}


def h_bank_side(rt: Runtime, req, ctx):
    a = req["args"]
    return rt.bank(ctx.check).side(a["root"], a["doses"], a["base_wikitext"]), {}


def h_bank_comp(rt: Runtime, req, ctx):
    a = req["args"]
    return rt.bank(ctx.check).composition(a["order"], a["base_wikitext"]), {}


def h_bank_inert(rt: Runtime, req, ctx):
    a = req["args"]
    return rt.bank(ctx.check).inert(a["spec"], a["base_values"]), {}


def h_rstats(rt: Runtime, req, ctx):
    return {"rcache": rt.rcache().stats() if rt._rc is not None else None}, {}


def h_field_repair(rt: Runtime, req, ctx):
    """RULING-wme-c1-bank-halt K3: re-collect one field unit (a group, or the vocabulary rows) on this instance with
    the registered handler; a non-unit request (the controller verifies the hashes; no unit result)."""
    a = req["args"]
    if "group" in a:
        return h_field_group(rt, req, ctx)
    return h_field_vocab(rt, req, ctx)


def h_bank_verify(rt: Runtime, req, ctx):
    """RULING-wme-c1-bank-halt K3b: re-run a completed bank unit (side / composition / inert) with the
    re-collected field as a verification computation; a non-unit request (no unit result)."""
    a = req["args"]
    fn = {"bank_side": h_bank_side, "bank_comp": h_bank_comp, "bank_inert": h_bank_inert}[a["kind"]]
    return fn(rt, {"args": a["args"]}, ctx)


HANDLERS = {
    "hello": h_hello,
    "field_repair": h_field_repair,
    "bank_verify": h_bank_verify,
    "canary": h_canary,
    "field_group": h_field_group,
    "field_vocab": h_field_vocab,
    "bank_base": h_bank_base,
    "bank_side": h_bank_side,
    "bank_comp": h_bank_comp,
    "bank_inert": h_bank_inert,
    "rstats": h_rstats,
}


def register(name):
    def deco(fn):
        HANDLERS[name] = fn
        return fn

    return deco


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-vllm", action="store_true")
    a = ap.parse_args(argv)
    from wsp.wm.c1 import genengine as GE

    GE.prepare_process()  # §II.31: CUBLAS_WORKSPACE_CONFIG before the first CUDA call
    cfg = Config.make(a.root, smoke=a.smoke, vllm=not a.no_vllm)
    rt = Runtime(cfg)
    import importlib

    from wsp.wm.c1 import serve_more  # noqa: F401  (E2, sealed, generation, S-TE and launch handlers)

    # under ``python -m`` this file runs as __main__; the handler modules register into the importable module
    registry = dict(HANDLERS)
    registry.update(importlib.import_module("wsp.wm.c1.serve").HANDLERS)
    handlers = {k: (lambda req, ctx, fn=fn: fn(rt, req, ctx)) for k, fn in registry.items()}
    from wsp.wm.c1 import health as HL

    W.Worker(cfg.root, handlers, sampler=HL.Sampler(cfg.root)).serve()


if __name__ == "__main__":
    sys.exit(main())
