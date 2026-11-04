"""Assert generated smoke artifacts describe the running committed checkout."""

import json
import subprocess
import sys
from pathlib import Path


def check(root: Path) -> None:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for family in ("heretic_ara", "lora_sft"):
        entry = root / family
        edit = json.loads((entry / "edit.json").read_text())
        manifest = json.loads((entry / "manifest.json").read_text())
        assert edit["git_sha"] == manifest["git_sha"] == sha, (family, sha)
        assert edit["params"]["git_dirty"] is False, family
        assert edit["base_revision"] != "main", family
        print(f"{family}: clean committed SHA {sha}; base {edit['base_revision']}")


if __name__ == "__main__":
    check(Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs/p1-generator-smoke"))
