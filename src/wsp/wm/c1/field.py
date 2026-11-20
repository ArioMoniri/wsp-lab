"""The base field at theta0 (R1) and the representation R(E) on 9B (PREREG_WME §6.1.2, §II.22.2; manifest
``model.representation``). Worker side.

**Field units.** One unit per 4-layer group g (layers 4g..4g+3; modules ``down`` and ``o`` of each layer), every
trajectory objective (``panel2.T2_OBJECTIVES``), on the 24-item ``construction/field`` shard
(``wm.c1.shards``): ``wm.collect.collect_grouped`` (the Fisher draws are seeded per (objective, group) pass, so a
per-group call replays the single-call estimator). After collection the unit computes every module's kept factors
by the registered rule (``response_field.spectral_factor``: symmetry audit, clamp, trace-relative damping, 99 %
energy) with the fp64 eigendecomposition on the GPU (:func:`spectral_factor_gpu`, the same arithmetic as the CPU
rule) and stores them factored (eigenvectors and kept eigenvalues), because the dense kept matrices of all 64
writers do not fit in memory. The vocabulary-row unit (``wm.vocabfield``) covers the declared ``embed_tokens`` rows
(row_scale roots and the I2 never-indexed rows).

**R(E).** ``r_from_contractions`` sums per-tensor contributions; a contraction depends on one tensor's realised
displacement only. :class:`RCache` therefore computes R exactly by caching each tensor's (a_k, q_k, |D|^2) under
the ordered list of the actions that touch it (the realised bf16 composition of those actions on that tensor,
applied through the live ``EditSession``), and summing. q = 1/2 tr(G D A D^T) is evaluated as
1/2 sum_ij lG_i lA_j (U_G^T D U_A)_ij^2 with the kept eigenpairs (the same value as the dense kept form of
``collect.contract``). The single registered chart column (``chart:schema:CHART_SCHEMA_VERSION=2``) is 0 for every
state (amendment G3a); norms are Frobenius norms of the realised displacement (writers, embedding rows, total).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from wsp.features import response_field as rf

T2_OBJECTIVES = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
)
N_GROUPS = 8
GROUP_LAYERS = 4
CHART_COLUMNS = ("schema:CHART_SCHEMA_VERSION=2",)
NORM_GROUPS = ("writers", "embed_rows")


def group_layers(g: int) -> list:
    return list(range(GROUP_LAYERS * g, GROUP_LAYERS * (g + 1)))


def module_of(param_name: str) -> str:
    return param_name.rsplit(".", 1)[0]


def group_modules(rm, g: int) -> list:
    return [module_of(rm.name(L, r)) for L in group_layers(g) for r in ("down", "o")]


def spec_9b():
    from wsp.wm import wmcp as CP

    return CP.spec_9b(T2_OBJECTIVES, CHART_COLUMNS, NORM_GROUPS)


# ------------------------------------------------------------------ kept factors


def spectral_factor_gpu(factor: torch.Tensor, device="cuda"):
    """``response_field.spectral_factor`` with the fp64 eigendecomposition on ``device``; returns (vectors[:, :r]
    fp32 on CPU, values[:r] fp32 on CPU, audit). vectors @ diag(values) @ vectors.T is the kept factor."""
    f = factor.detach().to(device).double()
    if f.ndim != 2 or f.shape[0] != f.shape[1] or not torch.isfinite(f).all():
        raise ValueError("invalid finite square factor")
    largest = f.abs().max().item() if f.numel() else 0.0
    asym = (f - f.T).abs().max().item() if f.numel() else 0.0
    if asym > rf.SYMMETRY_TOLERANCE * largest:
        raise ValueError(f"factor symmetry audit failed: {asym:.3e} > tol x {largest:.3e}")
    f = (f + f.T) * 0.5
    values, vectors = torch.linalg.eigh(f)
    lmax, lmin = values.max().item(), values.min().item()
    if lmin < 0 and (lmax <= 0 or lmin < -rf.NEGATIVE_TOLERANCE * lmax):
        raise ValueError(f"materially negative factor eigenvalue: {lmin / lmax:.3e}")
    n_clamped = int((values < 0).sum().item())
    values = values.clamp_min(0)
    raw_trace = values.sum().item()
    damping = rf.DAMPING * raw_trace / f.shape[0]
    values = values.flip(0) + damping
    vectors = vectors.flip(1)
    total = values.sum()
    rank = (
        int(torch.searchsorted(values.cumsum(0), rf.ENERGY * total).item()) + 1 if total > 0 else 0
    )
    audit = {
        "raw_trace": raw_trace,
        "damping": damping,
        "rank": rank,
        "dimension": int(f.shape[0]),
        "lambda_max": lmax,
        "n_clamped": n_clamped,
        "omitted_energy": (values[rank:].sum() / total).item() if total > 0 else 0.0,
    }
    return (
        vectors[:, :rank].float().cpu().contiguous(),
        values[:rank].float().cpu().contiguous(),
        audit,
    )


def compute_kept(field_dir: Path, device="cuda", log=None) -> dict:
    """Kept eigenpairs of every (objective, module) of one collected group, saved under ``kept/``."""
    from wsp.wm import collect as C

    fld = C.GroupedField.load(field_dir, verify=False)
    out = Path(field_dir) / "kept"
    out.mkdir(exist_ok=True)
    audits, hashes = {}, {}
    for obj in fld.coverage:
        mods = sorted({r["module"] for r in fld.coverage[obj].values() if r["factors"]})
        for m in mods:
            f = fld.factors(obj, m)
            UA, lA, aA = spectral_factor_gpu(f["A"], device)
            UG, lG, aG = spectral_factor_gpu(f["G"], device)
            fn = f"{obj}.{m}.pt"
            torch.save({"UA": UA, "lA": lA, "UG": UG, "lG": lG}, out / fn)
            hashes[fn] = hashlib.sha256((out / fn).read_bytes()).hexdigest()
            audits[f"{obj}|{m}"] = {"A": aA, "G": aG}
            if log:
                log(
                    f"kept {obj} {m}: rank A {aA['rank']}/{aA['dimension']} G {aG['rank']}/{aG['dimension']}"
                )
        fld.drop_cache()
    (out / "audit.json").write_text(
        json.dumps({"audits": audits, "sha256": hashes}, sort_keys=True)
    )
    return {"audits": audits, "sha256": hashes}


# ------------------------------------------------------------------ field access for contractions


class Field:
    """All collected groups (gradients by parameter, kept eigenpairs by module) plus the vocabulary-row field."""

    def __init__(self, root, device="cuda", cache_gb: float = 24.0):
        from wsp.wm import collect as C

        self.root = Path(root)
        self.device = device
        self.groups = {}
        self.param_group = {}
        for g in range(N_GROUPS):
            d = self.root / f"g{g}"
            if (d / "manifest.json").exists():
                fld = C.GroupedField.load(d, verify=False)
                self.groups[g] = fld
                for cov in fld.coverage.values():
                    for p, row in cov.items():
                        self.param_group[p] = (g, row)
        self.vocab = None
        if (self.root / "vocab" / "manifest.json").exists():
            from wsp.wm import vocabfield as VF

            self.vocab = VF.VocabField.load(self.root / "vocab")
        self._kept = {}
        self._grad = {}
        self.cache_bytes = int(cache_gb * 1e9)
        self._order = []

    def _evict(self):
        size = lambda v: (
            sum(t.numel() * t.element_size() for t in v)
            if isinstance(v, tuple)
            else v.numel() * v.element_size()
        )
        tot = sum(size(v) for v in list(self._kept.values()) + list(self._grad.values()))
        while tot > self.cache_bytes and self._order:
            k = self._order.pop(0)
            v = self._kept.pop(k, None) or self._grad.pop(k, None)
            if v is not None:
                tot -= size(v)

    def grad(self, obj, param):
        k = ("g", obj, param)
        if k not in self._grad:
            g, _row = self.param_group[param]
            self._grad[k] = self.groups[g].gradient(obj, param).to(self.device, torch.float32)
            self._order.append(k)
            self._evict()
        return self._grad[k]

    def kept(self, obj, module):
        k = ("k", obj, module)
        if k not in self._kept:
            g = next(
                gg
                for gg, fld in self.groups.items()
                if any(r.get("module") == module for r in fld.coverage[obj].values())
            )
            d = torch.load(
                self.root / f"g{g}" / "kept" / f"{obj}.{module}.pt",
                map_location="cpu",
                weights_only=True,
            )
            self._kept[k] = tuple(d[x].to(self.device) for x in ("UA", "lA", "UG", "lG"))
            self._order.append(k)
            self._evict()
        return self._kept[k]

    def contract_tensor(self, param: str, D: torch.Tensor, objectives=T2_OBJECTIVES) -> dict:
        """{obj: (signed, q or None)} for one writer tensor's realised displacement D."""
        if param not in self.param_group:
            from wsp.wm.collect import UnsupportedTensor

            raise UnsupportedTensor(f"{param}: not in the collected support")
        _g, row = self.param_group[param]
        d = D.detach().to(self.device, torch.float32)
        out = {}
        for obj in objectives:
            signed = float((self.grad(obj, param) * d).sum())
            q = None
            if row["factors"]:
                UA, lA, UG, lG = self.kept(obj, row["module"])
                M = UG.T @ d[:, : UA.shape[0]] @ UA
                q = max(0.0, 0.5 * float(((lG[:, None] * lA[None, :]) * M.square()).sum()))
            out[obj] = (signed, q)
        return out

    def contract_embed(self, embed_name: str, D: torch.Tensor, objectives=T2_OBJECTIVES) -> dict:
        if self.vocab is None:
            return {o: (None, None) for o in objectives}
        res = self.vocab.contract({embed_name: D.detach().float().cpu()})
        return {o: (res[o][embed_name]["signed"], res[o][embed_name]["q"]) for o in objectives}


# ------------------------------------------------------------------ exact R(E) with per-tensor caching


class RCache:
    """R(E) of realised compositions. ``r_of(bundles)``: ``bundles`` is a list of steps, each a list of
    ``wm.actions.Action`` applied in order from theta0 (``EditSession.compose`` order). Exact: per tensor, the
    realised displacement of the ordered actions touching it (applied through the live session, then restored)."""

    def __init__(self, mctx, field: Field, objectives=T2_OBJECTIVES):
        self.m, self.f, self.objs = mctx, field, tuple(objectives)
        self.spec = spec_9b()
        self.cache: dict = {}
        self.hits = self.misses = 0
        self.seconds = 0.0

    def _tensor_contrib(self, name: str, actions: list):
        key = (name, tuple(a.key() for a in actions))
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        t0 = time.time()
        sess = self.m.sess
        self.m.at_base()
        try:
            for a in actions:
                sess.apply(a)
            D = sess.displacement(name)
            sq = float(D.float().square().sum())
            if name == self.m.embed:
                vals = self.f.contract_embed(name, D, self.objs)
                group = "embed_rows"
            else:
                vals = self.f.contract_tensor(name, D, self.objs)
                group = "writers"
        finally:
            sess.restore()
        rec = {"vals": vals, "sq": sq, "group": group}
        self.cache[key] = rec
        self.seconds += time.time() - t0
        return rec

    def warm(self, bundles_list):
        """Fill the cache tensor-major for many states (each tensor's factors are loaded once)."""
        need: dict = {}
        for bundles in bundles_list:
            by_tensor: dict = {}
            for step in bundles:
                for a in step:
                    by_tensor.setdefault(a.tensor, []).append(a)
            for name, acts in by_tensor.items():
                key = (name, tuple(a.key() for a in acts))
                if key not in self.cache:
                    need.setdefault(name, {})[key] = acts
        for name in sorted(need):
            for _key, acts in sorted(need[name].items()):
                self._tensor_contrib(name, acts)

    def r_of_many(self, bundles_list) -> list:
        self.warm(bundles_list)
        return [self.r_of(b) for b in bundles_list]

    def r_of(self, bundles) -> tuple[np.ndarray, list]:
        """R(E) through the registered builder ``wmcp.r_from_contractions`` (per-tensor contributions in sorted
        tensor order; unsupported q enters as 0 and is flagged by the builder)."""
        from wsp.wm import wmcp as CP

        by_tensor: dict = {}
        for step in bundles:
            for a in step:
                by_tensor.setdefault(a.tensor, []).append(a)
        con = {o: {} for o in self.objs}
        sq = {"writers": 0.0, "embed_rows": 0.0}
        for name in sorted(by_tensor):
            rec = self._tensor_contrib(name, by_tensor[name])
            sq[rec["group"]] += rec["sq"]
            for o in self.objs:
                s, qq = rec["vals"][o]
                if s is None:
                    raise ValueError(f"{name}: no field gradient for {o} (unsupported tensor)")
                con[o][name] = {"signed": s, "q": qq}
        if not by_tensor:
            con = {o: {} for o in self.objs}
        return CP.r_from_contractions(
            self.spec,
            self.objs,
            con,
            chart=[0.0] * len(CHART_COLUMNS),
            group_norms=[math.sqrt(sq["writers"]), math.sqrt(sq["embed_rows"])],
            total_norm=math.sqrt(sq["writers"] + sq["embed_rows"]),
        )

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "seconds": self.seconds,
            "entries": len(self.cache),
        }


def field_digest(root) -> dict:
    """Digest of every collected group manifest and kept-factor audit (the field unit payload)."""
    root = Path(root)
    out = {}
    for p in (
        sorted(root.glob("g*/manifest.json"))
        + sorted(root.glob("g*/kept/audit.json"))
        + sorted(root.glob("vocab/manifest.json"))
    ):
        out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
