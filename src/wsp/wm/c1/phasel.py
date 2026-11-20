"""Phase L re-run logic (PREREG_WME §II.31.1-§II.31.2, §II.31.8 R1, R2, R9). Pure functions; the driver is
``scripts/wme_c1_ctl.py`` (``phasel-parity``, ``phasel-main``). Engineering, outcome-blind: exposed canary items only.

* :func:`run_checks` issues ``parity`` (GPU 0 vs GPU 1), ``inert_identity`` and, under C1(a), ``gen_repeat`` (n = 3,
  GPU 0) through a ``call(kind, args) -> payload`` function.
* :func:`evaluate` computes the §II.31.1 pass rule with all three items explicit (3/3 repeats, GPU 0 = GPU 1 on the
  ``hf_gen`` canary, the deterministic HF inert rows), the §II.31.2 selection (S_L from the dead-man ledger, C from
  the cap in force) and the other launch preconditions, and decides ``proceed``: only if every check passes and the
  selected engine equals the recorded ``GEN_ENGINE``.
* :func:`write_new` writes run-tagged evidence and never overwrites a file; every Phase L evidence object carries the
  engine-config sha256 it was produced under (:func:`check_binding` refuses a mismatch).
* :func:`parser_check` (R9) recomputes the per-function source digests of the five registered ``gen.parsers``
  functions and the file sha256 of ``genaudit.py``.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from wsp.wm.c1 import genengine as GE

ROOT = Path(__file__).resolve().parents[4]
GENAUDIT = ROOT / "src/wsp/wm/genaudit.py"
PARSER_FUNCTIONS = ("squad_norm", "classify_v1", "detect_loop_v1", "audit_v1", "audit_v1_single")
# sha256 of each function's source segment (ast.get_source_segment) in the registered file 43ec3beb… (2737136);
# unchanged in every later version (§II.31.8 D4)
PARSER_DIGESTS = {
    "squad_norm": "b71d71c47c1d96fab27170c35d40de6d7469f01f0652a03fc8afb1969021874a",
    "classify_v1": "5e6e441e4741a9dfe87810bb50df6d08b31699b103dff90f59469d20ff71d0b2",
    "detect_loop_v1": "00fa1f502da4472cc141d6cf19315c8db5b7eb406f5ae978f3992e0837dc2d8a",
    "audit_v1": "da4bcb8844866a588bae0f9648022fadcebe5f53bbbde89beff9f43bfa4cf79a",
    "audit_v1_single": "3a504e36f3fed121fe199794c9f5cdae919aa21499e2eaef5eae47544c5c501d",
}
# the file's known history since registration (§II.31.8 D4); any other file hash fails the launch check
GENAUDIT_FILES = {
    "43ec3beb27bec9bd3dc2429501173094e1b579606becd4803c14e9862a2d9b15": "2737136: the registered gen.parsers hash",
    "52ad2bbddd407d1962f690f37ba3f458e89b53a16d1e30167b7aa0d434c55016": "63ca07f (pre-freeze): _generate / "
    "generate_sampled added; parser functions unchanged",
    "cc13cd377ad1a7e4551a3b6c08562fb99db6aff953fd79cac7837d6927301e0a": "46967f8 (study M Gemma 3 port): the stop "
    "ids of _generate via wm.chat.stop_ids; parser functions unchanged; Qwen-equivalent (§II.31.8 D4)",
}


# the generation file a launch must run (63ca07f; restored on main by d3a2d01 after the 46967f8 deviation)
GENAUDIT_LAUNCH_SHA256 = "52ad2bbddd407d1962f690f37ba3f458e89b53a16d1e30167b7aa0d434c55016"


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------ R9


def parser_digests(src: str) -> dict:
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in PARSER_FUNCTIONS:
            out[node.name] = hashlib.sha256(ast.get_source_segment(src, node).encode()).hexdigest()
    return out


def parser_check(path=GENAUDIT) -> dict:
    data = Path(path).read_bytes()
    file_sha = hashlib.sha256(data).hexdigest()
    got = parser_digests(data.decode())
    funcs_ok = got == PARSER_DIGESTS
    return {
        "functions": got,
        "registered_functions": dict(PARSER_DIGESTS),
        "functions_unchanged": funcs_ok,
        "file_sha256": file_sha,
        "file_history": GENAUDIT_FILES.get(file_sha),
        "launch_file_sha256": GENAUDIT_LAUNCH_SHA256,
        # §II.31.8 note (main d3a2d01 restored the registered bytes): a launch needs exactly the registered file
        "pass": funcs_ok and file_sha == GENAUDIT_LAUNCH_SHA256,
    }


# ------------------------------------------------------------------ R2


def write_new(path, obj) -> str:
    """Write evidence that must never replace committed evidence: refuses an existing path. Returns its sha256."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"{p} exists: Phase L evidence is run-tagged and never overwritten")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_binding(evidence: dict, engine_config_sha256: str, what: str = "evidence"):
    got = evidence.get("engine_config_sha256")
    if got != engine_config_sha256:
        raise RuntimeError(
            f"{what} was produced under engine config {str(got)[:12]}, not the current {engine_config_sha256[:12]}"
        )


# ------------------------------------------------------------------ R1


def run_checks(call, gen_engine: str | None = None) -> dict:
    gen_engine = GE.check_mode(gen_engine)
    out = {"parity": call("parity", {}), "inert_identity": call("inert_identity", {})}
    out["gen_repeat"] = call("gen_repeat", {"n": 3}) if gen_engine == "hf_deterministic" else None
    return out


def inert_hf_ok(inert: dict) -> dict:
    """§II.31.1 item 3: every I1/I2 row's deterministic HF generation is bitwise equal to the base (at least one)."""
    rows = [r for r in inert.get("rows", []) if r["spec"]["kind"] in ("i1", "i2")]
    ok = bool(rows) and all(r.get("hf_sha256") and r.get("hf_bitwise_base") is True for r in rows)
    return {"rows": len(rows), "pass": ok}


def inert_vllm_ok(inert: dict) -> dict:
    rows = inert.get("vllm_rows", [])
    return {"rows": len(rows), "pass": bool(rows) and all(r.get("vllm_bitwise_base") for r in rows)}


def evaluate(checks: dict, gen_engine: str | None, s_l: float, cap: float, g_e1: float) -> dict:
    gen_engine = GE.check_mode(gen_engine)
    par, inert, rep = checks["parity"], checks["inert_identity"], checks.get("gen_repeat")
    c0, c1 = par.get("canary_gpu0") or {}, par.get("canary_gpu1") or {}
    other = {
        "tf_chunks_1_vs_2_gpu_bitwise": par.get("parity_bitwise") is True,
        "tf_gpu0_vs_gpu1_every_chunk_bitwise": par.get("gpu0_vs_gpu1_every_chunk_bitwise") is True,
        "tf_canary_gpu0_eq_gpu1": bool(c0.get("tf")) and c0.get("tf") == c1.get("tf"),
        "vllm_gpu0_eq_gpu1": par.get("vllm_gpu0_eq_gpu1") is True,
        "inert_identity_pass": inert.get("pass") is True,
    }
    if gen_engine == "hf_deterministic":
        hashes = [r["sha256"] for r in rep["deterministic"]]
        gp = GE.gen_pass(hashes, c0.get("hf_gen"), c1.get("hf_gen"))
        item3 = inert_hf_ok(inert)
        passed = {
            "repeat_3_of_3": gp["repeat_3_of_3"],
            "gpu0_eq_gpu1": gp["gpu0_eq_gpu1"],
            "inert_hf_bitwise": item3["pass"],
            "pass": gp["pass"] and item3["pass"],
        }
        sel = GE.select_engine(
            passed, rep["t_det_median_s"], rep["registered"]["seconds"], g_e1, s_l, cap
        )
        selected = sel["engine"]
        inputs = {
            "repeat_hashes": hashes,
            "hf_gen_gpu0": c0.get("hf_gen"),
            "hf_gen_gpu1": c1.get("hf_gen"),
            "inert_hf": item3,
            "t_det_median_s": rep["t_det_median_s"],
            "t_reg_s": rep["registered"]["seconds"],
        }
    else:  # C1(b) active (selected earlier): its own checks; no further fallback exists
        vg = par.get("vllm_gen_gpu0")
        item_v = inert_vllm_ok(inert)
        passed = {
            "vllm_gen_gpu0_eq_gpu1": bool(vg) and vg == par.get("vllm_gen_gpu1"),
            "inert_vllm_bitwise": item_v["pass"],
        }
        passed["pass"] = all(passed.values())
        sel = {
            "engine": "vllm" if passed["pass"] else None,
            "note": "C1(b) active: no further fallback",
        }
        selected = sel["engine"]
        inputs = {
            "vllm_gen_gpu0": vg,
            "vllm_gen_gpu1": par.get("vllm_gen_gpu1"),
            "inert_vllm": item_v,
        }
    return {
        "active_engine": gen_engine,
        "inputs": dict(inputs, S_L=float(s_l), C=float(cap), G_E1=float(g_e1)),
        "pass_rule": passed,
        "selection": sel,
        "selected_engine": selected,
        "other_preconditions": other,
        "proceed": bool(selected == gen_engine and passed["pass"] and all(other.values())),
    }


# ------------------------------------------------------------------ N3: cross-instance canary identity

CANARY_SOURCES = {  # engine -> where the parity evidence holds its GPU-0 hash
    "tf": ("parity", "canary_gpu0", "tf"),
    "hf_gen": ("parity", "canary_gpu0", "hf_gen"),
    "vllm": ("parity", "vllm_gpu0"),
    "vllm_gen": ("parity", "vllm_gen_gpu0"),
}


def _dig(obj, path):
    for k in path:
        obj = (obj or {}).get(k)
    return obj


def cross_instance_canaries(engines: dict, parity_evidence: dict) -> dict:
    """§II.31.9 N3: the main instance's per-engine canaries must equal the GPU-0 canaries of the bound parity evidence
    (an engine reproducible within one instance but not across instances is refused before it becomes the
    committed reference)."""
    pairs = {}
    for eng, h in sorted(engines.items()):
        src = CANARY_SOURCES.get(eng)
        ref = None if src is None else _dig(parity_evidence, src)
        pairs[eng] = {"main": h, "parity_gpu0": ref, "equal": bool(ref) and ref == h}
    return {"pairs": pairs, "pass": bool(pairs) and all(p["equal"] for p in pairs.values())}


# ------------------------------------------------------------------ N4 (b): code identity


def git(*args, cwd=ROOT) -> str:
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def code_identity(worker_code_sha: str | None, bundle_sha256: str | None, cwd=ROOT) -> dict:
    """§II.31.9 N4 (b): the controller commit (clean tracked tree), the commit the worker bundle was built from
    (its CODE_SHA, reported by ``hello``) and the bundle's sha256. ``pass`` only if the tree is clean, the worker runs
    the controller's commit and the bundle hash is known."""
    commit = git("rev-parse", "HEAD", cwd=cwd)
    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=cwd)
    out = {
        "controller_commit": commit,
        "controller_tree_clean": dirty == "",
        "worker_code_sha": worker_code_sha,
        "bundle_sha256": bundle_sha256,
    }
    out["pass"] = (
        out["controller_tree_clean"]
        and worker_code_sha == commit
        and bool(bundle_sha256)
        and len(str(bundle_sha256)) == 64
    )
    return out


# ------------------------------------------------------------------ N6: the registered test gate

GATE_COMMAND = "tests/test_wme_c1_*.py tests/test_wme_runguard.py tests/test_wme_runguard_property.py tests/test_wm_harness.py"


def parse_pytest_summary(text: str) -> dict:
    import re

    last = [ln for ln in text.strip().splitlines() if re.search(r"\b(passed|failed|error)", ln)]
    counts = {}
    if last:
        for n, k in re.findall(
            r"(\d+) (passed|failed|skipped|deselected|errors?|xfailed|xpassed)", last[-1]
        ):
            counts[k.rstrip("s") if k.startswith("error") else k] = int(n)
    return counts


def run_gate(runner=None, cwd=ROOT) -> dict:
    """§II.31.9 N6: the registered wme-c1 test command, independent of the repo-wide ruff state (CPU only). Recorded
    with its counts and exit code in the launch evidence; a launch proceeds only on exit 0."""
    import glob
    import os
    import subprocess
    import sys

    files = sorted(f for pat in GATE_COMMAND.split() for f in glob.glob(str(Path(cwd) / pat)))
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    if runner is None:
        p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, env=env, check=False)
        code, text = p.returncode, (p.stdout or "") + (p.stderr or "")
    else:
        code, text = runner(argv, env)
    counts = parse_pytest_summary(text)
    return {
        "command": f"CUDA_VISIBLE_DEVICES= python -m pytest -q {GATE_COMMAND}",
        "files": [str(Path(f).relative_to(cwd)) for f in files],
        "exit_code": int(code),
        "counts": counts,
        "pass": int(code) == 0 and counts.get("failed", 0) == 0 and counts.get("passed", 0) > 0,
    }


def expected_worker_state(gen_engine: str | None = None) -> dict:
    """The worker's process globals at ``hello`` (R11, N4 (b)): torch defaults, and under C1(a) the CUBLAS workspace
    variable set by ``prepare_process``."""
    gen_engine = GE.check_mode(gen_engine)
    want = json.loads(json.dumps(GE.BASELINE_STATE))
    want[GE.CUBLAS_ENV] = GE.CUBLAS_WORKSPACE if gen_engine == "hf_deterministic" else None
    return want


def worker_state_check(state, gen_engine: str | None = None) -> dict:
    want = expected_worker_state(gen_engine)
    return {"expected": want, "got": state, "pass": state == want}
