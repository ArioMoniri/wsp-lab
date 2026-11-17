# source this: sets project env; secrets come from ~/.config/wsp/env (chmod 600)
export WSP_CANON="$HOME/projects/wsp-lab"
export WSP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$WSP_CANON/data/hf"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$WSP_ROOT/src:${PYTHONPATH:-}"
[ -f "$HOME/.config/wsp/env" ] && set -a && . "$HOME/.config/wsp/env" && set +a
# Per-track venv isolation: WSP_VENV=<name> selects $WSP_CANON/.venvs/<name> (create with scripts/venv_track.sh).
# The shared $WSP_CANON/.venv is read-only for tracks: never pip/uv install into it; make a track venv instead.
if [ -n "${WSP_VENV:-}" ]; then export WSP_VENV_DIR="$WSP_CANON/.venvs/$WSP_VENV"; else export WSP_VENV_DIR="$WSP_CANON/.venv"; fi
export PATH="$WSP_VENV_DIR/bin:$HOME/.local/bin:$PATH"
# Per-project, per-CUDA-tag LD_LIBRARY_PATH, CUDA_HOME and kernel caches (other projects on this host use other CUDA builds).
export CUDA_ENV_PROJECT="wsp-lab-${WSP_VENV:-shared}"
. "$WSP_ROOT/scripts/cuda_env.sh" "$WSP_VENV_DIR"

# Canonical data lives outside the checkout since the 6 Nov 2026 incident (D8); data/ in the checkout is a symlink.
export WSP_DATA_ROOT="${WSP_DATA_ROOT:-/home/ubuntu/wsp-data}"
