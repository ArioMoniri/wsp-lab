"""The wme-c1 instance bundle, built from the committed tree (PREREG_WME §II.31.9 P2; Board A N7).

* **Code** (``src/`` and every tracked bundle file) comes from ``git archive HEAD``: the exact committed bytes, never
  the working tree. The build refuses if ``src/`` or any tracked bundle file is dirty, or ``src/`` has untracked
  files, so ``CODE_SHA`` (HEAD) always describes what was shipped.
* **Data inputs** (``data/``, outside git by design: the open panel splits, bank roots, teacher files) are copied
  from the data tree and recorded with their sha256 in ``BUNDLE_INPUTS.json``. A bundle file that is untracked and
  not under ``data/`` is refused; a sealed path is refused.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ARC = "wmec1"


def _git(root, *args, binary=False):
    p = subprocess.run(["git", *args], cwd=root, capture_output=True, check=True)
    return p.stdout if binary else p.stdout.decode().strip()


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sealed(rel: str) -> bool:
    return "sealed" in rel


def build(out, root, files) -> dict:
    """Write the bundle to ``out``; returns {"bundle", "sha256", "code_sha", "inputs"}. Raises SystemExit (writing
    nothing) on a dirty or untracked source tree, an untracked non-data file or a sealed path."""
    root, out = Path(root), Path(out)
    files = list(files)
    bad = [f for f in files if _sealed(f)]
    if bad:
        raise SystemExit(f"refusing a sealed path in the bundle: {bad[:3]}")
    tracked = set(_git(root, "ls-files", "--", *files).splitlines()) if files else set()
    untracked_code = [f for f in files if f not in tracked and not f.startswith("data/")]
    if untracked_code:
        raise SystemExit(f"bundle files not tracked by git (and not data inputs): {untracked_code}")
    code_paths = ["src", *sorted(tracked)]
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *code_paths)
    if dirty:
        raise SystemExit(
            f"refusing to bundle: dirty or untracked files under the bundled code:\n{dirty}"
        )
    head = _git(root, "rev-parse", "HEAD")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / ARC
        stage.mkdir()
        archive = _git(root, "archive", "--format=tar", "HEAD", "--", *code_paths, binary=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(stage, filter="data")
        inputs = {}
        for rel in sorted(f for f in files if f not in tracked):
            src = root / rel
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            inputs[rel] = _sha_file(dst)
        (stage / "BUNDLE_INPUTS.json").write_text(json.dumps(inputs, indent=1, sort_keys=True))
        (stage / "CODE_SHA").write_text(head)
        leaked = [str(p.relative_to(stage)) for p in stage.rglob("*") if _sealed(p.name)]
        if leaked:
            raise SystemExit(f"sealed path in bundle: {leaked[:3]}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w:gz") as tf:
            tf.add(stage, arcname=ARC)
    return {"bundle": str(out), "sha256": _sha_file(out), "code_sha": head, "inputs": inputs}
