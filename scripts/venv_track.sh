#!/usr/bin/env bash
# Create an isolated venv for one track: scripts/venv_track.sh <name> [cuda-tag e.g. cu132|cu128] [extra uv pip args...]
# It lives at $WSP_CANON/.venvs/<name>, starts from the shared .venv's frozen versions (so results stay comparable),
# and then takes the extra packages (e.g. vllm) without touching the shared .venv. Use it with: WSP_VENV=<name> . scripts/env.sh
set -euo pipefail
name="${1:?usage: venv_track.sh <name> [cuda-tag] [extra pkgs...]}"; shift
tag="${1:-}"; [ $# -gt 0 ] && shift
canon="$HOME/projects/wsp-lab"; dir="$canon/.venvs/$name"
[ -e "$dir" ] && { echo "exists: $dir" >&2; exit 1; }
if [ -z "$tag" ]; then
  tag="cu$("$canon/.venv/bin/python" -c 'import torch;print((torch.version.cuda or "").replace(".",""))')"
fi
cons="$(mktemp)"; trap 'rm -f "$cons"' EXIT
"$canon/.venv/bin/python" -m pip freeze --exclude-editable 2>/dev/null | grep -viE '^(torch|torchvision|torchaudio|triton|nvidia-)' > "$cons" || true
uv venv -q --python 3.12 "$dir"
uv pip install -q --python "$dir/bin/python" torch --index-url "https://download.pytorch.org/whl/$tag"
uv pip install -q --python "$dir/bin/python" -c "$cons" -e "$canon[dev]"
[ $# -gt 0 ] && uv pip install -q --python "$dir/bin/python" "$@"
"$dir/bin/python" -c 'import torch;print("venv ok", torch.__version__, torch.version.cuda)'
