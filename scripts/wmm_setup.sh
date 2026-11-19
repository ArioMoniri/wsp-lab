#!/usr/bin/env bash
# Study M Gate 0 setup: track venv "wmm" and the three pinned snapshots (target medgemma-1.5-4b-it; endpoints medgemma-4b-it, gemma-3-4b-it) in the HF cache under data/hf.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -e "$HOME/projects/wsp-lab/.venvs/wmm" ] || nice -n 10 scripts/venv_track.sh wmm cu132
WSP_VENV=wmm . scripts/env.sh
for repo in google/medgemma-1.5-4b-it google/medgemma-4b-it google/gemma-3-4b-it; do
  nice -n 10 python - "$repo" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.jinja"])
print(sys.argv[1], "->", p)
PY
done
echo SETUP_DONE
