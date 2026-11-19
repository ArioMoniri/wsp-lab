"""Worker-side model operations shared by every wme-c1 handler (engineering; runs on the Lambda instance).

* :class:`ModelCtx` loads the pinned target text-only in bf16 (``wm.hf.load_target``), checks theta0 against the
  manifest, and holds one resident ``EditSession`` (apply -> evaluate -> restore, ``harness.ResidentServer``).
* :func:`evaluate` scores a panel through ``wm.panel2.evaluate`` (the registered teacher-forced path, batching by
  ``wm.panel.PanelConfig``), optionally with the observation tap (``wm.tap.ResidualTap``; outputs are bitwise
  unchanged by the hooks).
* Canaries (§II.13 item 4, §II.22.6; manifest ``harness.runguard.canary``), one per engine:
  - ``tf``: the P-H100 inert-check items (wme-p2 construction, 16 per objective by id, 7 objectives) scored in one
    ``panel2.evaluate`` call; sha256 of the fp32 value bytes in (objective, item) order;
  - ``hf_gen``: the 64 exposed wme-p3h holdout MMLU-Pro items (stage-W canary set, file order = salted rank order),
    ``genaudit.generate_greedy`` at batch 128, 128 new tokens, non-thinking; sha256 of the token-id lists;
  - ``vllm``: the same items under the stage-W frozen numerics, 2,048 new tokens (``wm.c1.vllmops``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REPO = "Qwen/Qwen3.5-9B"
SMOKE_REPO = "Qwen/Qwen3.5-0.8B"
PANEL_C1 = ROOT / "data/wme/panels/wme-c1"
PANEL_P2 = ROOT / "data/wme/panels/wme-p2"
PANEL_P3H = ROOT / "data/wme/panels/wme-p3h"
CANARY_TF_OBJECTIVES = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
    "mathcomp_gold_lp",
)
CANARY_TF_ITEMS = 16
CANARY_GEN_ITEMS = 64
CANARY_HF = {"batch": 128, "max_new_tokens": 128}
# pilot-1 tap convention (layers, k); the seed is the registered tap.seed (manifest)
TAP_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
TAP_K = 16


def manifest() -> dict:
    return json.loads((ROOT / "docs/prereg/wme_manifest.json").read_text())


def sha_ids(rows) -> str:
    """sha256 of the generated token-id lists (stage-W canary convention)."""
    return hashlib.sha256(json.dumps([list(map(int, r)) for r in rows]).encode()).hexdigest()


def sha_values(values) -> str:
    """sha256 of fp32 value bytes, in the given order."""
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def tf_canary_items(panel=PANEL_P2) -> dict:
    from wsp.wm import panel2 as P2

    cons = P2.read_open(panel, "construction")
    return {o: sorted(cons[o], key=lambda r: r["id"])[:CANARY_TF_ITEMS] for o in CANARY_TF_OBJECTIVES}


def gen_canary_items(panel=PANEL_P3H, n=CANARY_GEN_ITEMS) -> list:
    from wsp.wm import p3panel as PP

    return PP.read_split(panel, "holdout")["mmlupro_gen"][:n]


def flatten_values(res: dict, order) -> list:
    out = []
    for o in order:
        out.extend(float(r["value"]) for r in res[o])
    return out


class ModelCtx:
    def __init__(self, repo: str = REPO, device: str = "cuda", check_theta0: bool = True):
        import torch

        from wsp.wm import actions as A
        from wsp.wm import hf
        from wsp.wm import panel as P

        t = time.time()
        self.repo, self.device = repo, device
        self.model, self.tok, self.cfg, self.base_hash, _ = hf.load_target(repo, device)
        names = [n for n, _ in self.model.named_parameters()]
        prefix = (
            "model.language_model.layers."
            if any(".language_model." in n for n in names)
            else "model.layers."
        )
        self.prefix = prefix
        self.rm = A.RoleMap.from_config(self.cfg, prefix=prefix)
        self.embed = next(n for n in names if n.endswith("embed_tokens.weight"))
        self.sess = A.EditSession(dict(self.model.named_parameters()), base_hash=self.base_hash, rolemap=self.rm)
        self.adapter = P.LMAdapter(self.model)
        self.pcfg = P.PanelConfig()
        if check_theta0 and repo == REPO:
            want = manifest()["target.theta0_hash"]
            if self.base_hash != want:
                raise RuntimeError(f"theta0 {self.base_hash[:12]} != manifest {want[:12]}")
        if device == "cuda":
            torch.cuda.synchronize()
        self.load_s = time.time() - t

    # -------------------------------------------------- state discipline
    def at_base(self):
        if self.sess.theta_hash() != self.base_hash:
            raise RuntimeError("theta drift: worker not at the base state")

    def restore(self):
        self.sess.restore()
        if self.sess.theta_hash() != self.base_hash:
            raise RuntimeError(f"theta-hash after restore {self.sess.theta_hash()[:12]} != base")

    def layers(self):
        m = self.model
        inner = getattr(m, "model", m)
        inner = getattr(inner, "language_model", inner)
        return inner.layers

    # -------------------------------------------------- evaluation
    def evaluate(self, items: dict, tap_seed: int | None = None):
        """``items`` {objective: [item]}: per-item records (panel2.evaluate); with ``tap_seed`` also the tap."""
        import torch

        from wsp.wm import panel2 as P2
        from wsp.wm import tap as T

        tap = None
        if tap_seed is not None:
            lay = self.layers()
            tap = T.ResidualTap({i: lay[i] for i in TAP_LAYERS}, hidden=self.rm.hidden, k=TAP_K, seed=tap_seed)
        with torch.no_grad():
            res = P2.evaluate(self.adapter, self.tok, items, self.pcfg, tap=tap)
        return res, tap

    def theta_hash(self) -> str:
        return self.sess.theta_hash()

    # -------------------------------------------------- canaries (at the base state only)
    def canary_tf(self, items=None) -> str:
        self.at_base()
        items = items or tf_canary_items()
        res, _ = self.evaluate(items)
        return sha_values(flatten_values(res, CANARY_TF_OBJECTIVES))

    def canary_hf(self, items=None) -> str:
        from wsp.wm import genaudit as GA
        from wsp.wm import p3gen as G

        self.at_base()
        items = items or gen_canary_items()
        prompts = [G.user_prompt("mmlupro_gen", it) for it in items]
        out = GA.generate_greedy(
            self.model, self.tok, prompts, CANARY_HF["max_new_tokens"], batch=CANARY_HF["batch"], device=self.device
        )
        return sha_ids([r["ids"] for r in out])

    def release(self):
        """Free the GPU (before a vLLM engine starts in this process)."""
        import gc

        import torch

        self.sess = self.adapter = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def versions() -> dict:
    out = {"python": os.sys.version.split()[0]}
    for mod in ("torch", "transformers", "vllm", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    try:
        import torch

        out["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        pass
    return out


def physical_identity() -> dict:
    """Unique physical-instance evidence (§II.28.3 item 5): sha256 of the GPU UUIDs and of the machine id (raw ids
    are not recorded), driver version and GPU names."""
    import subprocess

    out = {}
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip().splitlines()
        rows = [[x.strip() for x in line.split(",")] for line in q]
        out["gpu_uuid_sha256"] = [hashlib.sha256(r[0].encode()).hexdigest() for r in rows]
        out["gpu_names"] = [r[1] for r in rows]
        out["driver"] = rows[0][2] if rows else None
    except Exception as exc:  # noqa: BLE001
        out["gpu_error"] = str(exc)[:200]
    try:
        out["machine_id_sha256"] = hashlib.sha256(Path("/etc/machine-id").read_bytes().strip()).hexdigest()
    except OSError:
        out["machine_id_sha256"] = None
    return out
