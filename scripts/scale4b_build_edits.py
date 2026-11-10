"""F5.3 scale diagnostic: build 20 outcome-blind Gaussian edits of Qwen3.5-4B as dense bf16 zoo entries.

Design (the registered F5.3/FP4 text governs; aligned 10 Nov 2026, "A5 note (10 Nov 2026): 4B builder aligned
to F5.3/FP4" in docs/PREREG_H1_TF.md; nothing here is fitted to an outcome):

* support: the writer tensors of the 0.8B ``random_dir`` family at the same relative depths. The 0.8B
  layer list [8, 12, 15, 17, 19, 21] of 24 scales to round(l * 32 / 24) = [11, 16, 20, 23, 25, 28] of 32
  (the same list as ``configs/zoo_qwen3.5-4b.yaml``). Tensors are the ``random_dir`` specs
  ``mlp.down_proj`` and ``self_attn.o_proj``, which ``wsp.zoo.arch`` resolves as role aliases
  (``mlp_out``, ``attn_out``): ``self_attn.o_proj`` on the full-attention layers (11, 23) and
  ``linear_attn.out_proj`` on the GatedDeltaNet layers, exactly as ``random_dir`` does at 0.8B.
  ``--support literal`` keeps only modules literally named ``o_proj`` (8 tensors instead of 12).
* kernel: for every support tensor W, an independent iid standard Gaussian direction Z (numpy PCG64,
  float64, seed derived per tensor from the draw seed, below), scaled to ||delta||_F = rho * ||W||_F with
  rho = 0.005 (F5.3; per touched tensor, so the pooled intended radius over the support is also rho); the
  stored delta is the bf16 **effective** delta bf16(bf16(W + delta) - W) (DECISIONS D1).
  bf16 rounding changes the edit: the *realised* delta is what the evaluator applies,
  bf16(W + stored) - W, and its radius ||delta_real||_F / ||W||_F per tensor and pooled, the fraction
  of changed elements, the cosine to the intended Gaussian and the fraction of elements where the
  applied weight differs from bf16(W + delta) are recorded in every edit.json and manifest row.
* seeds (F5.3): draw i in 0..19 has seed = int.from_bytes(sha256("20261110:4b-gaussian:" + str(i))[:8],
  "big") mod 2^63 (``draw_seed``). Per tensor (``tensor_seed``): int.from_bytes(sha256(f"{draw seed}:{layer}:
  {module path}")[:8], "big") mod 2^63, where (layer, module path) is the namespace-free key from
  ``wsp.zoo.arch.split_key``; Z = numpy.random.Generator(PCG64(tensor seed)).standard_normal(shape, float64).
  A tensor's draw therefore depends only on the draw index and its own key, never on the other support
  tensors or the checkpoint's key namespace. ids follow ``EditDescriptor.id`` (family ``gaussian``, not in
  the shared schema's family list on purpose: adding it would change ``source_hash`` for every TF-v1 run).
* FP4 paired qualification panel: draw indices 0, 4, 8, 12, 16 (``PAIRED_DRAWS``), fixed in the manifest
  header before any outcome exists; the one registered expansion is draw indices 1, 5, 9, 13, 17
  (``EXPANSION_DRAWS``), recorded in the header and used only under FP4's fail-closed rule.

Usage:
    python scripts/scale4b_build_edits.py --dry-run            # plan only; writes nothing
    python scripts/scale4b_build_edits.py --revision <sha>     # build into data/zoo/qwen3.5-4b-scale
    python scripts/scale4b_build_edits.py --verify data/zoo/qwen3.5-4b-scale [--ids a b]
                                                               # rebuild in memory, compare hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from wsp.zoo import arch as _arch

BASE_MODEL = "Qwen/Qwen3.5-4B"
# Config revision checked against the checkpoint headers (tests/fixtures/arch/qwen3.5-4b.json).
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FAMILY = "gaussian"
KERNEL = "gaussian_relative_frobenius_per_tensor_pcg64_bf16_effective_v2"
RADIUS = 0.005  # F5.3 planned chair default rho (the 1e-3 default is withdrawn)
SEED_DOMAIN = (
    "20261110:4b-gaussian:"  # F5.3: seed = sha256(domain + str(draw index))[:8] big-endian mod 2^63
)
GENERATOR = "numpy.random.PCG64"
N_EDITS = 20
PAIRED_DRAWS = (0, 4, 8, 12, 16)  # FP4 panel
EXPANSION_DRAWS = (1, 5, 9, 13, 17)  # FP4 one-time fail-closed expansion
SOURCE_LAYERS = (8, 12, 15, 17, 19, 21)  # configs/zoo_qwen3.5-0.8b.yaml random_dir.layers
SOURCE_DEPTH = 24
TARGET_DEPTH = 32
TENSOR_SPECS = ("mlp.down_proj", "self_attn.o_proj")  # random_dir tensors (role aliases)
OUT_DIR = Path("data/zoo/qwen3.5-4b-scale")
DTYPE = torch.bfloat16
FIXTURE = REPO / "tests/fixtures/arch/qwen3.5-4b.json"


# ---- plan ------------------------------------------------------------------------------------


def scale_layers(layers=SOURCE_LAYERS, source_depth=SOURCE_DEPTH, target_depth=TARGET_DEPTH):
    """Same relative depth: floor(l * target / source + 0.5) (half up, never banker's rounding)."""
    out = [math.floor(layer * target_depth / source_depth + 0.5) for layer in layers]
    if len(set(out)) != len(out) or not all(0 <= x < target_depth for x in out):
        raise ValueError(f"scaled layer list {out} is not {len(layers)} distinct layers")
    return out


def support_keys(state_keys, layers, mode="role"):
    """Sorted state keys edited by every entry: random_dir's writer specs in ``layers``."""
    if mode not in ("role", "literal"):
        raise ValueError("support mode must be 'role' or 'literal'")
    index = _arch.index_state_keys(state_keys)
    qwen = _arch.ARCHS["qwen3_5"]
    keys: list[str] = []
    for layer in layers:
        if mode == "role":
            keys += _arch.select_keys(TENSOR_SPECS, layer, index, qwen)
        else:
            for spec in TENSOR_SPECS:
                keys += _arch.writer_keys([spec.replace(".", r"\.")], layer, index)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate support key")
    return sorted(keys)


def _sha_seed(text: str) -> int:
    """First eight sha256 bytes of the UTF-8 text, big-endian, modulo 2^63 (F5.3)."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**63


def draw_seed(index: int) -> int:
    """F5.3 draw seed for draw index 0-19: sha256('20261110:4b-gaussian:' + str(index))."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("draw index must be a nonnegative int")
    return _sha_seed(SEED_DOMAIN + str(index))


def tensor_seed(seed: int, key: str) -> int:
    """Per-tensor PCG64 seed from the draw seed and the namespace-free (layer, module path)."""
    parsed = _arch.split_key(key)
    if parsed is None:
        raise ValueError(f"{key} is not a layer weight")
    return _sha_seed(f"{seed}:{parsed[0]}:{parsed[1]}")


def gaussian_direction(seed: int, key: str, shape) -> torch.Tensor:
    """Z for one tensor: iid standard normal, float64, PCG64 seeded by ``tensor_seed``."""
    rng = np.random.Generator(np.random.PCG64(tensor_seed(seed, key)))
    return torch.from_numpy(rng.standard_normal(tuple(shape), dtype=np.float64))


def entry_params(index: int, layers, support_mode: str) -> dict[str, Any]:
    return {
        "kernel": KERNEL,
        "radius": RADIUS,
        "radius_scope": "per_tensor_frobenius",
        "seed_rule": f"sha256('{SEED_DOMAIN}' + str(index))[:8] big-endian mod 2^63",
        "generator": GENERATOR,
        "layers": list(layers),
        "tensors": list(TENSOR_SPECS),
        "support_mode": support_mode,
        "index": index,
    }


def entry_id(params, seed, revision, base_model=BASE_MODEL) -> str:
    """Identical payload and hash to ``wsp.schema.EditDescriptor.id``."""
    payload = json.dumps(
        {"f": FAMILY, "p": params, "s": seed, "b": base_model, "r": revision, "parent": None},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def ids_at(ids, draws):
    """ids of the given draw indices (``ids`` in draw-index order); every index must exist."""
    if any(d >= len(ids) for d in draws):
        raise ValueError(f"draw indices {list(draws)} need at least {max(draws) + 1} draws")
    return [ids[d] for d in draws]


def paired_ids(ids, draws=PAIRED_DRAWS):
    """FP4 paired qualification panel: the ids of draw indices 0, 4, 8, 12, 16."""
    return ids_at(ids, draws)


def plan(revision, n=N_EDITS, support_mode="role"):
    layers = scale_layers()
    rows = []
    for i in range(n):
        params = entry_params(i, layers, support_mode)
        seed = draw_seed(i)
        rows.append(
            {
                "id": entry_id(params, seed, revision),
                "draw_index": i,
                "seed": seed,
                "params": params,
            }
        )
    return layers, rows


# ---- kernel ----------------------------------------------------------------------------------


def _norm(x: torch.Tensor) -> float:
    return math.sqrt(x.double().square().sum().item())


def gaussian_delta(base: dict[str, torch.Tensor], seed: int, radius: float = RADIUS):
    """bf16-effective Gaussian deltas at relative radius ``radius`` of each tensor's Frobenius norm.

    Returns ``(delta, stats)``. The stored delta is bf16(bf16(W + d) - W) (DECISIONS D1, bf16 zoo
    format). The *realised* edit is what the evaluator's ``apply_delta`` produces from it,
    bf16(W + stored) - W; all realised radii are of that tensor. Where a difference of two bf16
    values is not representable in bf16 (an edit crossing a binade or zero) the applied weight can
    differ from bf16(W + d); those elements are counted in ``applied_mismatch_fraction``.
    """
    if not base or not math.isfinite(radius) or radius <= 0:
        raise ValueError("nonempty tensors and positive finite radius required")
    delta: dict[str, torch.Tensor] = {}
    per: dict[str, dict[str, float]] = {}
    base_sq = eff_sq = stored_sq = 0.0
    for key in sorted(base):
        original = base[key].detach().to(device="cpu", dtype=DTYPE)
        w = original.float()
        w_norm = _norm(w)
        if not math.isfinite(w_norm) or w_norm == 0:
            raise ValueError(f"{key}: base norm must be finite and nonzero")
        z = gaussian_direction(seed, key, original.shape)
        intended = (z * (radius * w_norm / _norm(z))).to(torch.float32)
        target = (w + intended).to(DTYPE)
        d = (target.float() - w).to(DTYPE)
        applied = (w + d.float()).to(DTYPE)  # wsp.eval.run.apply_delta: fp32 add, cast to bf16
        realised = applied.float() - w
        e_norm = _norm(realised)
        i_norm = _norm(intended)
        dot = (realised.double() * intended.double()).sum().item()
        per[key] = {
            "base_norm": w_norm,
            "intended_norm": i_norm,
            "realised_norm": e_norm,
            "realised_radius": e_norm / w_norm,
            "changed_fraction": (realised != 0).double().mean().item(),
            "cosine_to_intended": dot / (e_norm * i_norm) if e_norm else 0.0,
            "applied_mismatch_fraction": (applied != target).double().mean().item(),
        }
        delta[key] = d
        base_sq += w_norm**2
        eff_sq += e_norm**2
        stored_sq += _norm(d.float()) ** 2
    realised = math.sqrt(eff_sq / base_sq)
    stats = {
        "radius": radius,
        "realised_radius": realised,
        "realised_radius_min": min(v["realised_radius"] for v in per.values()),
        "realised_radius_max": max(v["realised_radius"] for v in per.values()),
        "changed_fraction": _weighted(per, base, "changed_fraction"),
        "applied_mismatch_fraction": _weighted(per, base, "applied_mismatch_fraction"),
        "realised_fro_norm": math.sqrt(eff_sq),
        "delta_fro_norm": math.sqrt(stored_sq),  # of the stored tensor, as wsp.zoo.build records it
        "per_tensor": per,
    }
    return delta, stats


def _weighted(per, base, field):
    return sum(v[field] * base[k].numel() for k, v in per.items()) / sum(
        base[k].numel() for k in per
    )


def delta_bytes(delta: dict[str, torch.Tensor]) -> bytes:
    from safetensors.torch import save

    return save({k: v.to(DTYPE).contiguous() for k, v in sorted(delta.items())})


# ---- build -----------------------------------------------------------------------------------


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def spec_hash(revision, n, support_mode, keys) -> str:
    spec = {
        "base_model": BASE_MODEL,
        "base_revision": revision,
        "kernel": KERNEL,
        "radius": RADIUS,
        "n": n,
        "seed_domain": SEED_DOMAIN,
        "generator": GENERATOR,
        "paired_draws": list(PAIRED_DRAWS),
        "layers": scale_layers(),
        "tensors": list(TENSOR_SPECS),
        "support_mode": support_mode,
        "support": list(keys),
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def header_row(revision, n, support_mode, keys, ids, sha, dry_run=False):
    return {
        "type": "header",
        "dry_run": dry_run,
        "purpose": "F5.3 4B scale diagnostic (diagnostic, not a gate)",
        "base_model": BASE_MODEL,
        "base_revision": revision,
        "family": FAMILY,
        "kernel": KERNEL,
        "radius": RADIUS,
        "radius_scope": "per_tensor_frobenius",
        "seed_domain": SEED_DOMAIN,
        "seed_rule": f"sha256('{SEED_DOMAIN}' + str(draw index))[:8] big-endian mod 2^63",
        "tensor_seed_rule": "sha256(f'{draw seed}:{layer}:{module path}')[:8] big-endian mod 2^63",
        "generator": GENERATOR,
        "seeds": [draw_seed(i) for i in range(n)],
        "n": n,
        "source_layers": list(SOURCE_LAYERS),
        "source_depth": SOURCE_DEPTH,
        "layers": scale_layers(),
        "target_depth": TARGET_DEPTH,
        "tensors": list(TENSOR_SPECS),
        "support_mode": support_mode,
        "support": list(keys),
        "paired_draw_indices": list(PAIRED_DRAWS),
        "paired_ids": paired_ids(ids),
        "expansion_draw_indices": list(EXPANSION_DRAWS),
        "expansion_ids": ids_at(ids, EXPANSION_DRAWS),
        "pairing_rule": "FP4: draw indices 0, 4, 8, 12, 16 (expansion once: 1, 5, 9, 13, 17)",
        "config_hash": spec_hash(revision, n, support_mode, keys),
        "git_sha": sha,
        "script_sha256": script_sha256(),
        "dtype": "bfloat16",
    }


def make_entry(base_state, keys, row, revision, sha, cfg_hash):
    """(edit.json dict, safetensors bytes, manifest row) for one planned entry."""
    delta, stats = gaussian_delta({k: base_state[k] for k in keys}, row["seed"], RADIUS)
    payload = delta_bytes(delta)
    digest = hashlib.sha256(payload).hexdigest()
    identity = {
        "family": FAMILY,
        "base_model": BASE_MODEL,
        "base_revision": revision,
        "dtype": "bfloat16",
        "stored_format": "dense",
        "kernel_variant": KERNEL,
        "support": list(keys),
    }
    reuse_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    edit = {
        "family": FAMILY,
        "params": row["params"],
        "seed": row["seed"],
        "base_model": BASE_MODEL,
        "base_revision": revision,
        "parent_id": None,
        "git_sha": sha,
        "touched_tensors": sorted(delta),
        "delta_fro_norm": stats["delta_fro_norm"],
        "provenance": {
            "kernel_variant": KERNEL,
            "reuse_identity": identity,
            "reuse_key": reuse_key,
            "delta_sha256": digest,
            "radius_stats": stats,
            "script_sha256": script_sha256(),
            "config_hash": cfg_hash,
        },
        "eligibility": "production",
        "eligibility_reasons": [],
        "id": row["id"],
    }
    line = {
        "id": row["id"],
        "family": FAMILY,
        "parent_id": None,
        "draw_index": row["draw_index"],
        "seed": row["seed"],
        "config_hash": cfg_hash,
        "git_sha": sha,
        "base_revision": revision,
        "eligibility": "production",
        "stored_delta_format": "dense",
        "reuse_key": reuse_key,
        "delta_sha256": digest,
        "realised_radius": stats["realised_radius"],
        "realised_radius_min": stats["realised_radius_min"],
        "realised_radius_max": stats["realised_radius_max"],
        "changed_fraction": stats["changed_fraction"],
        "applied_mismatch_fraction": stats["applied_mismatch_fraction"],
    }
    return edit, payload, line


def _write_entry(out_dir: Path, edit, payload: bytes) -> None:
    """Atomic: stage edit.json + delta.safetensors, then rename; identical rebuilds are no-ops."""
    final = out_dir / edit["id"]
    if final.exists():
        have = json.loads((final / "edit.json").read_text())["provenance"]["delta_sha256"]
        if have != edit["provenance"]["delta_sha256"]:
            raise FileExistsError(f"{final} exists with another delta; move it aside to rebuild")
        if hashlib.sha256((final / "delta.safetensors").read_bytes()).hexdigest() != have:
            raise FileExistsError(f"{final}/delta.safetensors does not match its edit.json")
        return
    stage = out_dir / ".staging" / edit["id"]
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    (stage / "delta.safetensors").write_bytes(payload)
    (stage / "edit.json").write_text(json.dumps(edit, indent=2, sort_keys=True))
    os.replace(stage, final)


def build(
    base_state,
    out_dir: Path,
    revision: str,
    n=N_EDITS,
    support_mode="role",
    dry_run=False,
    sha=None,
    expected_support=None,
    log=print,
):
    """Build (or, with ``dry_run``, only plan) the entries. Returns (header, rows)."""
    if len(revision) != 40:
        raise ValueError("pin the base model to a full 40-character commit")
    sha = sha or git_sha()
    layers, planned = plan(revision, n, support_mode)
    keys = support_keys(base_state, layers, support_mode)
    if expected_support is not None and keys != list(expected_support):
        raise ValueError("support differs from the checkpoint-verified fixture")
    header = header_row(revision, n, support_mode, keys, [r["id"] for r in planned], sha)
    if dry_run:
        header["dry_run"] = True
        return header, [
            {"id": r["id"], "draw_index": r["draw_index"], "seed": r["seed"]} for r in planned
        ]
    out_dir = Path(out_dir)
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists():
        old = json.loads(manifest.read_text().splitlines()[0])
        if old.get("config_hash") != header["config_hash"]:
            raise FileExistsError(f"{manifest} belongs to another design; refusing to mix")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in planned:
        edit, payload, line = make_entry(
            base_state, keys, row, revision, sha, header["config_hash"]
        )
        _write_entry(out_dir, edit, payload)
        line["paired"] = row["id"] in header["paired_ids"]
        rows.append(line)
        log(
            f"entry {row['id']} draw={row['draw_index']} seed={row['seed']} realised_radius={line['realised_radius']:.6g} "
            f"changed={line['changed_fraction']:.3f}"
        )
    tmp = manifest.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in [header, *rows]))
    tmp.replace(manifest)
    shutil.rmtree(out_dir / ".staging", ignore_errors=True)
    return header, rows


def verify(base_state, zoo_dir: Path, ids=None):
    """Rebuild entries in memory and compare their delta hashes with the manifest (writes nothing)."""
    lines = [json.loads(x) for x in (Path(zoo_dir) / "manifest.jsonl").read_text().splitlines()]
    header, rows = lines[0], {r["id"]: r for r in lines[1:]}
    keys = header["support"]
    out = {}
    for i in ids or sorted(rows):
        row = rows[i]
        index = row["draw_index"]
        if row["seed"] != draw_seed(index):
            raise ValueError(f"{i}: seed does not follow from draw index {index}")
        params = entry_params(index, header["layers"], header["support_mode"])
        plan_row = {"id": i, "draw_index": index, "seed": row["seed"], "params": params}
        if entry_id(params, row["seed"], header["base_revision"]) != i:
            raise ValueError(f"{i}: id does not follow from the recorded design")
        _, payload, _ = make_entry(
            base_state, keys, plan_row, header["base_revision"], "", header["config_hash"]
        )
        out[i] = hashlib.sha256(payload).hexdigest() == row["delta_sha256"]
    return out


# ---- CLI -------------------------------------------------------------------------------------


def fixture_keys() -> list[str]:
    return json.loads(FIXTURE.read_text())["params"]


def load_base_state(revision: str) -> dict[str, torch.Tensor]:
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=revision, torch_dtype=DTYPE, device_map="cpu"
    )
    state = {k: v.detach().cpu().to(DTYPE) for k, v in model.state_dict().items()}
    del model
    return state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--revision", default=DEFAULT_REVISION, help="pinned 40-char base commit")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--n", type=int, default=N_EDITS)
    p.add_argument("--support", choices=["role", "literal"], default="role")
    p.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    p.add_argument("--verify", type=Path, metavar="ZOO_DIR", help="rebuild in memory and compare")
    p.add_argument("--ids", nargs="+", help="with --verify: only these ids")
    args = p.parse_args(argv)
    if args.dry_run:
        # Key names from the checkpoint-verified fixture: no weights, no network, no files written.
        names = {k: None for k in fixture_keys()}
        header, rows = build(names, args.out, args.revision, args.n, args.support, True)
        print(json.dumps({"header": header, "entries": rows}, indent=2))
        return 0
    state = load_base_state(args.revision)
    if args.verify:
        result = verify(state, args.verify, args.ids)
        print(json.dumps(result, indent=2))
        return 0 if all(result.values()) else 1
    expected = support_keys(fixture_keys(), scale_layers(), args.support)
    header, rows = build(
        state, args.out, args.revision, args.n, args.support, expected_support=expected
    )
    print(json.dumps({"config_hash": header["config_hash"], "paired_ids": header["paired_ids"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
