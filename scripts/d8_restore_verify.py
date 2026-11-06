#!/usr/bin/env python3
"""D8 step 1: verify the restored zoo against the manifest and write the artefact-loss sidecar.

Board A gate 4 (docs/reviews/P5/board-a-incident-d8-2026-11-06.md section 2):
  * every stored delta / adapter file has a readable safetensors header,
  * every dense delta's Frobenius norm equals ``edit.json:delta_fro_norm`` (LoRA: norm of the dense
    product ``(B @ A) * scaling``, reported but compared with a looser tolerance),
  * every entry's content hash (``entry_delta_hash`` of build_h1a_panel.py) is a member of its
    component's ``source_identities`` in docs/prereg/panel_manifest.json.
Then the sidecar: one row per manifest entry (id, family, delta_present, alias_of, eval_present,
records_present, artefact_lost). ``manifest.jsonl`` and every eligibility are read, never written.

An entry without bytes is an alias when the content-hash alias ledger of panel_manifest.json lists it
in a group with a member whose bytes are present and verified; ``alias_of`` is that member (the
ledger's canonical entry when it is present, else the smallest present id).

    d8_restore_verify.py [--workers 4] [--zoo DIR] [--out DIR] [--sidecar PATH]
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ZOO = Path("/home/ubuntu/wsp-data/zoo/qwen3.5-0.8b")
PANEL = ROOT / "docs/prereg/panel_manifest.json"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def artefact_files(entry_dir):
    dense = entry_dir / "delta.safetensors"
    if dense.exists():
        return [dense]
    return sorted(p for p in (entry_dir / "lora").glob("*") if p.is_file())


def entry_delta_hash(files):  # identical to scripts/build_h1a_panel.py:entry_delta_hash
    return sha("".join(f"{p.name}:{file_sha(p)}\n" for p in files)) if files else None


def header_ok(path):
    """The safetensors header parses and every tensor's byte range lies inside the file."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > size - 8:
            return False, f"header length {n} exceeds file size {size}"
        meta = json.loads(f.read(n))
    end = 0
    for k, v in meta.items():
        if k == "__metadata__":
            continue
        end = max(end, v["data_offsets"][1])
    if 8 + n + end != size:
        return False, f"data end {8 + n + end} != file size {size}"
    return True, None


def dense_norm(path):
    from safetensors.torch import load_file

    total = 0.0
    for t in load_file(path).values():
        total += (t.float() ** 2).sum().item()
    return math.sqrt(total)


def lora_norm(lora_dir):
    from wsp.zoo.apply import lora_factors

    total = 0.0
    for B, A, scale in lora_factors(lora_dir).values():
        # ||(B A) s||_F^2 = s^2 * sum((B^T B) * (A A^T))
        total += float(scale) ** 2 * float(((B.T @ B) * (A @ A.T)).sum())
    return math.sqrt(total)


def verify_entry(args):
    torch.set_num_threads(1)
    entry_id, zoo = args
    d = Path(zoo) / entry_id
    out = {"id": entry_id}
    files = artefact_files(d)
    out["files"] = [p.name for p in files]
    if not files:
        out["bytes"] = False
        return out
    out["bytes"] = True
    try:
        edit = json.loads((d / "edit.json").read_text())
        out["family"] = edit["family"]
        out["edit_norm"] = edit.get("delta_fro_norm")
        hdr = [header_ok(p) for p in files if p.suffix == ".safetensors"]
        out["header_ok"] = bool(hdr) and all(ok for ok, _ in hdr)
        out["header_error"] = next((e for ok, e in hdr if not ok), None)
        out["hash"] = entry_delta_hash(files)
        if out["header_ok"] and out["edit_norm"] is not None:
            dense = (d / "delta.safetensors").exists()
            out["norm"] = dense_norm(d / "delta.safetensors") if dense else lora_norm(d / "lora")
            out["norm_kind"] = "dense" if dense else "lora_product"
            out["norm_rel_err"] = abs(out["norm"] - out["edit_norm"]) / max(out["edit_norm"], 1e-30)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--zoo", default=str(ZOO))
    p.add_argument("--out", default="/home/ubuntu/wsp-data/runs/d8/restore")
    p.add_argument("--sidecar", default=str(ZOO / "artefact_status.jsonl"))
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    zoo, out = Path(args.zoo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in (zoo / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    entries = [r for r in rows if "id" in r]
    ids = [r["id"] for r in entries]
    assert len(ids) == len(set(ids)) == 1259, (len(ids), len(set(ids)))
    family = {r["id"]: r["family"] for r in entries}
    if args.limit:
        ids = ids[: args.limit]

    t0 = time.time()
    results = {}
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for n, r in enumerate(pool.map(verify_entry, [(i, str(zoo)) for i in ids], chunksize=4), 1):
            results[r["id"]] = r
            if n % 100 == 0:
                print(f"verified {n}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    with open(out / "restore_verify.jsonl", "w") as f:
        f.writelines(json.dumps(results[i], sort_keys=True) + "\n" for i in ids)

    panel = json.loads(PANEL.read_text())
    ledger = panel["alias_ledger"]
    identity_of = {}  # entry -> set of source identities of its component
    for comp in panel["components"]:
        for m in comp["members"]:
            identity_of[m] = set(comp["source_identities"])
    hash_of = {i: r.get("hash") for i, r in results.items() if r["bytes"]}

    problems = []
    members_missing_identity = []
    for i, r in results.items():
        if not r["bytes"]:
            continue
        if "error" in r or not r.get("header_ok"):
            problems.append(
                {
                    "id": i,
                    "problem": "header/read",
                    "detail": r.get("error") or r.get("header_error"),
                }
            )
            continue
        rel = r.get("norm_rel_err")
        tol = 1e-3 if r["norm_kind"] == "dense" else 5e-2
        if rel is not None and rel > tol:
            problems.append(
                {
                    "id": i,
                    "problem": "norm",
                    "kind": r["norm_kind"],
                    "norm": r["norm"],
                    "edit_norm": r["edit_norm"],
                    "rel_err": rel,
                }
            )
        if i in identity_of:
            if f"delta:{r['hash']}" not in identity_of[i]:
                problems.append(
                    {"id": i, "problem": "hash_not_in_source_identities", "hash": r["hash"]}
                )
        else:
            members_missing_identity.append(i)

    # alias resolution through the content-hash ledger (groups keyed by entry_delta_hash)
    alias_of = {}
    ledger_unverified = []
    for key, g in ledger.items():
        members = [g["canonical"], *g["aliases"]]
        present = sorted(m for m in members if m in hash_of)
        for m in present:
            if hash_of[m] != key:
                ledger_unverified.append({"id": m, "ledger_key": key, "actual": hash_of[m]})
        ok_present = [m for m in present if hash_of[m] == key]
        target = (
            g["canonical"]
            if g["canonical"] in ok_present
            else (ok_present[0] if ok_present else None)
        )
        for m in members:
            if m not in hash_of and target is not None:
                alias_of[m] = target

    sidecar = []
    for i in (r["id"] for r in entries):
        d = zoo / i
        has_bytes = bool(artefact_files(d)) and i in hash_of
        rec = {
            "id": i,
            "family": family[i],
            "delta_present": has_bytes,
            "alias_of": None if has_bytes else alias_of.get(i),
            "eval_present": (d / "eval.json").exists(),
            "records_present": (d / "records.jsonl").exists(),
        }
        rec["artefact_lost"] = (not rec["delta_present"]) and rec["alias_of"] is None
        sidecar.append(rec)
    if args.limit:
        print("limit set: sidecar not written")
    else:
        Path(args.sidecar).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in sidecar)
        )

    c = Counter()
    for r in sidecar:
        c["delta_present" if r["delta_present"] else "alias" if r["alias_of"] else "lost"] += 1
    by_family = Counter(
        (r["family"], "present" if r["delta_present"] else "alias" if r["alias_of"] else "lost")
        for r in sidecar
    )
    summary = {
        "n_manifest_entries": len(entries),
        "counts": dict(c),
        "by_family": {f"{k[0]}:{k[1]}": v for k, v in sorted(by_family.items())},
        "eval_present": sum(r["eval_present"] for r in sidecar),
        "records_present": sum(r["records_present"] for r in sidecar),
        "evaluated_and_lost": sum(r["eval_present"] and r["artefact_lost"] for r in sidecar),
        "evaluated_and_alias": sum(r["eval_present"] and bool(r["alias_of"]) for r in sidecar),
        "hash_verified": len(hash_of),
        "entries_without_component_identity": len(members_missing_identity),
        "problems": problems,
        "ledger_unverified": ledger_unverified,
        "norm_max_rel_err": {
            k: max(
                (r["norm_rel_err"] for r in results.values() if r.get("norm_kind") == k),
                default=None,
            )
            for k in ("dense", "lora_product")
        },
        "norm_checked": Counter(r.get("norm_kind") for r in results.values() if "norm" in r),
        "seconds": time.time() - t0,
        "manifest_sha256": hashlib.sha256((zoo / "manifest.jsonl").read_bytes()).hexdigest(),
    }
    summary["norm_checked"] = dict(summary["norm_checked"])
    (out / "restore_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(
        json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=1, sort_keys=True)
    )
    print("problems:", len(problems), problems[:10])
    return 0 if not problems and not ledger_unverified else 1


if __name__ == "__main__":
    sys.exit(main())
