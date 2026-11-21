"""Study M′ run-driver logic (PREREG_WMM, frozen at ``prereg-wmm-exploratory`` = 1904bbe; EXPLORATORY).

* **FC2 ledger and schedule.** :func:`build_tasks` expands ``prereg.COMPUTE["schedule"]`` into set-level tasks in the
  registered order: base → M-SYC → M-TE (without its ARC guard) → M-A minimum (20 units) → M-TE ARC guard → M-A
  remainder (12 units) → TE-I. :class:`Ledger` charges wall time with the model resident; :func:`run_schedule` starts
  no new task once the ledger has reached the cap (42.0 GPU-h). Skipped tasks are recorded "not measured (cap)", so a
  hard stop removes the M-A remainder first, then the M-TE ARC guard. TE-I (descriptive) runs only if its estimate
  fits under the cap.
* **FC1 read-gate.** :func:`load_outcomes` calls ``prereg.assert_provenanced`` on every state's metadata before it
  opens any outcome file.
* **Fixed batch composition.** :class:`BatchRegistry`: batch 32 per set; an OOM switches the set to batch 16 and
  returns every state already generated at 32 for regeneration (both sides of every pair), per PREREG §5.
* **Guards** with unmeasured sets (:func:`guards_with_missing`): an unmeasured guard is ``{"pass": False, "status":
  "not measured"}`` and voids its claim.
* **M-A constructions** (PREREG §4): τ-slice, LoRA-seed, random and sham units, scaled to the type's matched norm;
  R0 weight-space features (:func:`r0_features`).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from wsp.wm import actions as A
from wsp.wm.wmm import prereg as P
from wsp.wm.wmm import provenance as PV

CAP_H = P.COMPUTE["local_cap_gpu_h"]

# ------------------------------------------------------------------ tasks (seconds per item from Gate 1, PREREG §8)

_S = {
    "mq": 6.56,
    "mm": 5.13,
    "harm": 1.32,
    "benign": 1.15,
    "gen": 4.0,
    "short": 1.0,
    "ma_sugg": 5.4,
}
_N = {"mq": 1544, "mm": 2000, "harm": 1700, "benign": 1500, "gen": 2161, "ma": 200}


def _h(n, s):
    return n * s / 3600.0


def _state_sets(prefix, phase, sets):
    return [
        {
            "id": f"{prefix}:{name}",
            "phase": phase,
            "kind": "set",
            "state": prefix,
            "set": name,
            "est_h": est,
        }
        for name, est in sets
    ]


def build_tasks() -> list:
    long_sets = [
        ("mq_neutral", _h(_N["mq"], _S["mq"])),
        ("mq_suggested", _h(_N["mq"], _S["mq"])),
        ("mm_neutral", _h(_N["mm"], _S["mm"])),
        ("gcal", 0.05),
        ("harm", _h(_N["harm"], _S["harm"])),
        ("benign", _h(_N["benign"], _S["benign"])),
        ("gen", _h(_N["gen"], _S["gen"])),
    ]
    tasks = _state_sets("base", "base", long_sets)
    tasks += [
        {
            "id": "msyc:pool",
            "phase": "M-SYC",
            "kind": "pool",
            "state": "base",
            "est_h": _h(128, _S["mq"]),
        },
        {"id": "msyc:train", "phase": "M-SYC", "kind": "train", "state": "msyc", "est_h": 0.05},
    ]
    tasks += _state_sets("msyc", "M-SYC", long_sets)
    tasks += [
        {
            "id": "mte:pool",
            "phase": "M-TE",
            "kind": "pool",
            "state": "base",
            "est_h": _h(P.MTE["pool_n"], _S["mq"]),
        },
        {"id": "mte:rewrite", "phase": "M-TE", "kind": "rewrite", "state": "base", "est_h": 0.15},
        {"id": "mte:train", "phase": "M-TE", "kind": "train", "state": "mte", "est_h": 0.1},
    ]
    tasks += _state_sets(
        "mte",
        "M-TE",
        [
            ("mq_neutral", _h(_N["mq"], _S["short"])),
            ("mm_neutral", _h(_N["mm"], _S["short"])),
            ("gcal", 0.05),
            ("benign", _h(_N["benign"], _S["benign"])),
            ("harm", _h(_N["harm"], _S["harm"])),
        ],
    )
    unit_h = _h(_N["ma"], _S["ma_sugg"]) + 0.01
    tasks += [
        {
            "id": "ma:base",
            "phase": "M-A minimum",
            "kind": "set",
            "state": "base",
            "set": "ma_suggested",
            "est_h": _h(_N["ma"], _S["ma_sugg"]),
        },
        {
            "id": "ma:pool",
            "phase": "M-A minimum",
            "kind": "pool",
            "state": "base",
            "est_h": _h(256, _S["mq"]),
        },
    ]
    mini = [u["id"] for u in P.ma_bank(minimum=True)]
    rest = [u["id"] for u in P.ma_bank() if u["id"] not in set(mini)]
    tasks += [
        {"id": f"ma:{u}", "phase": "M-A minimum", "kind": "ma_unit", "unit": u, "est_h": unit_h}
        for u in mini
    ]
    tasks += _state_sets("mte", "M-TE ARC guard", [("gen", _h(_N["gen"], _S["gen"]))])
    tasks += [
        {"id": f"ma:{u}", "phase": "M-A remainder", "kind": "ma_unit", "unit": u, "est_h": unit_h}
        for u in rest
    ]
    tasks += _state_sets(
        "tei",
        "TE-I",
        [
            ("mq_neutral", _h(_N["mq"], 3.4)),
            ("mm_neutral", _h(_N["mm"], 3.4)),
            ("gcal", 0.05),
            ("harm", _h(_N["harm"], _S["harm"])),
            ("benign", _h(_N["benign"], _S["benign"])),
            ("gen", _h(_N["gen"], _S["gen"])),
        ],
    )
    return tasks


class Ledger:
    """Persistent GPU-hour ledger (wall time with the model resident) and task statuses."""

    def __init__(self, path, cap_h: float = CAP_H, clock=time.time):
        self.path = Path(path)
        self.cap_h = float(cap_h)
        self.clock = clock
        d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.seconds = float(d.get("seconds", 0.0))
        self.tasks = d.get("tasks", {})
        self.events = d.get("events", [])
        self._t = None

    @property
    def used_h(self) -> float:
        return self.seconds / 3600.0

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "cap_h": self.cap_h,
                    "seconds": self.seconds,
                    "used_h": self.used_h,
                    "tasks": self.tasks,
                    "events": self.events,
                },
                indent=1,
            )
        )
        tmp.replace(self.path)

    def add(self, seconds: float, what: str = ""):
        self.seconds += float(seconds)
        if what:
            self.events.append({"what": what, "seconds": float(seconds), "used_h": self.used_h})
        self.save()

    def begin(self):
        self._t = self.clock()

    def end(self, what: str = "overhead"):
        if self._t is not None:
            self.add(self.clock() - self._t, what)
            self._t = None


def run_schedule(tasks, ledger: Ledger, runner) -> dict:
    """Run ``tasks`` in order. A task starts only if the ledger is below the cap (TE-I: only if its estimate fits);
    otherwise it is recorded "not measured (cap)". Done tasks are skipped on resume."""
    for t in tasks:
        st = ledger.tasks.get(t["id"])
        if st and st["status"] == "done":
            continue
        fits = ledger.used_h < ledger.cap_h
        if t["phase"] == "TE-I":  # all-or-nothing: every remaining TE-I set must fit under the cap
            rest = sum(
                x["est_h"]
                for x in tasks
                if x["phase"] == "TE-I" and ledger.tasks.get(x["id"], {}).get("status") != "done"
            )
            fits = ledger.used_h + rest <= ledger.cap_h
        if not fits:
            ledger.tasks[t["id"]] = {
                "status": "not measured (cap)" if t["phase"] != "TE-I" else "not run (TE-I)",
                "phase": t["phase"],
                "at_h": ledger.used_h,
            }
            ledger.save()
            continue
        start_h = ledger.used_h
        t0 = ledger.clock()
        runner(t)
        ledger.add(ledger.clock() - t0, t["id"])
        ledger.tasks[t["id"]] = {
            "status": "done",
            "phase": t["phase"],
            "started_at_h": start_h,
            "ended_at_h": ledger.used_h,
        }
        ledger.save()
    return dict(ledger.tasks)


# ------------------------------------------------------------------ fixed batch composition: OOM policy


class BatchRegistry:
    def __init__(self, d=None):
        d = d or {}
        self.sizes = d.get("sizes", {})
        self.done = d.get("done", {})

    def to_json(self):
        return {"sizes": self.sizes, "done": self.done}

    def batch(self, set_name) -> int:
        return int(self.sizes.get(set_name, P.BATCH))

    def mark_done(self, set_name, state, batch):
        self.done.setdefault(set_name, {})[state] = int(batch)

    def on_oom(self, set_name, state) -> list:
        """Switch the set to batch 16; return the states generated at another batch (to regenerate)."""
        if self.batch(set_name) == 16:
            raise RuntimeError(
                f"{set_name}: OOM at batch 16 (state {state}); not recoverable under PREREG §5"
            )
        self.sizes[set_name] = 16
        return [s for s, b in self.done.get(set_name, {}).items() if b != 16 and s != state]


# ------------------------------------------------------------------ guards with unmeasured sets


def guards_with_missing(g: dict) -> dict:
    out = {}
    for k, kind, margin in (
        ("harm_compliance", "ub", P.MARGINS["harm_compliance"]),
        ("benign_over_refusal", "ub", P.MARGINS["benign_over_refusal"]),
        ("general_accuracy", "lb", P.MARGINS["general_accuracy"]),
    ):
        if g.get(k) is None:
            out[k] = {"pass": False, "status": "not measured"}
            continue
        r = P.paired_binary(*g[k], P.GUARD_ALPHA)
        out[k] = {**r, "pass": (r["ub"] <= margin) if kind == "ub" else (r["lb"] >= margin)}
    out["gcal"] = (
        {"pass": False, "status": "not measured"}
        if g.get("gcal") is None
        else P.gcal_guard(g["gcal"])
    )
    out["pass"] = all(v["pass"] for v in out.values())
    return out


# ------------------------------------------------------------------ FC1: gated outcome loading


def _read_outcome_file(p: Path) -> list:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def load_outcomes(run_dir, tags, set_name) -> dict:
    """Read the outcome files of ``set_name`` for ``tags`` only after every state's edit sha256 is checked against
    the provenance log (``prereg.assert_provenanced``)."""
    run_dir = Path(run_dir)
    states = [json.loads((run_dir / "states" / f"{t}.json").read_text()) for t in tags]
    P.assert_provenanced(states, PV.ProvenanceLog(run_dir / "provenance.jsonl"))
    return {t: _read_outcome_file(run_dir / "outcomes" / f"{t}__{set_name}.jsonl") for t in tags}


# ------------------------------------------------------------------ M-A constructions


def slice_names(rm, layers, comp) -> list:
    roles = ("q", "k", "v", "o") if comp == "attn" else ("gate", "up", "down")
    a, z = layers
    return sorted(rm.name(L, r) for L in range(a, z + 1) for r in roles)


def frob(d: dict) -> float:
    return math.sqrt(sum(float(v.float().pow(2).sum()) for v in d.values()))


def scale_to(d: dict, norm: float) -> dict:
    f = norm / frob(d)
    return {k: v.float() * f for k, v in d.items()}


def random_direction(params, rm, seed: int, norm: float) -> dict:
    out = {}
    for n in sorted(rm.writers().values()):
        w = params[n]
        g = torch.Generator(device="cpu").manual_seed(int(seed) * 100_003 + sum(map(ord, n)))
        x = torch.randn(tuple(w.shape), generator=g).to(w.device)
        out[n] = x / x.norm() * w.detach().float().norm()
    return scale_to(out, norm)


def sham_actions(rm, kind: str, layers) -> list:
    a, z = layers
    if kind == "uniform_o_scale":
        return [A.scale(rm.name(L, "o"), factor=2.0, dose=1.0) for L in range(a, z + 1)]
    if kind == "compensated_mlp_pair":
        from wsp.wm.wmm import controls as C

        return [act for b in C.compensated_pairs(rm, mlp_layers=range(a, z + 1)) for act in b]
    raise KeyError(kind)


def r0_keys(n_groups: int) -> list:
    return [f"cos_tau_group_{i}" for i in range(n_groups)] + [
        "cos_syc_attn",
        "cos_syc_mlp",
        "share_last_third",
        "top1_sv_share",
        "postnorm_cancel_index",
    ]


def _layer(rm, name):
    return rm.layer_of(name)


def _cos(a: dict, b: dict, names) -> float:
    names = [n for n in names if n in a and n in b]
    if not names:
        return 0.0
    dot = sum(float((a[n].float() * b[n].float().to(a[n].device)).sum()) for n in names)
    na = math.sqrt(sum(float(a[n].float().pow(2).sum()) for n in names))
    nb = math.sqrt(sum(float(b[n].float().pow(2).sum()) for n in names))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _top1_share(x: torch.Tensor, iters: int = 30) -> float:
    x = x.float()
    tot = float(x.pow(2).sum())
    if tot == 0 or x.dim() != 2:
        return 0.0
    v = torch.ones(x.shape[1], device=x.device) / math.sqrt(x.shape[1])
    for _ in range(iters):
        u = x @ v
        u = u / (u.norm() + 1e-30)
        v = x.T @ u
        s = float(v.norm())
        v = v / (s + 1e-30)
    return min(1.0, s * s / tot)


def r0_features(delta: dict, tau: dict, syc: dict, params, rm, groups=None) -> dict:
    """R0 (weight space): cos(Δ, τ) per layer group on the tensors Δ touches; cos(Δ, ΔW_syc) on o_proj and down_proj;
    share of ‖Δ‖² in the last third of layers; mean top-1 singular-value share; post-norm cancellation index
    (‖Δ‖²-weighted squared cosine between each writer Δ_t and W_t)."""
    groups = groups or P.LAYER_GROUPS
    out = {}
    for i, (a, z) in enumerate(groups):
        names = [n for n in delta if (L := _layer(rm, n)) is not None and a <= L <= z]
        out[f"cos_tau_group_{i}"] = _cos(delta, tau, names)
    out["cos_syc_attn"] = _cos(
        delta, syc, [n for n in delta if n.endswith("self_attn.o_proj.weight")]
    )
    out["cos_syc_mlp"] = _cos(delta, syc, [n for n in delta if n.endswith("mlp.down_proj.weight")])
    tot = sum(float(v.float().pow(2).sum()) for v in delta.values())
    cut = rm.n_layers - rm.n_layers // 3
    late = sum(
        float(v.float().pow(2).sum()) for n, v in delta.items() if (_layer(rm, n) or 0) >= cut
    )
    out["share_last_third"] = late / tot if tot > 0 else 0.0
    shares = [_top1_share(v) for v in delta.values() if v.dim() == 2]
    out["top1_sv_share"] = sum(shares) / len(shares) if shares else 0.0
    writers = set(rm.writers().values())
    idx = 0.0
    for n, v in delta.items():
        if n in writers and tot > 0:
            w = params[n].detach().float().to(v.device)
            vv = float(v.float().pow(2).sum())
            c2 = float((v.float() * w).sum()) ** 2 / (vv * float(w.pow(2).sum()) + 1e-30)
            idx += vv / tot * c2
    out["postnorm_cancel_index"] = idx
    return {k: out[k] for k in r0_keys(len(groups))}
