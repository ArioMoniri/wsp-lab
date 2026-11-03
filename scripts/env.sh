# source this: sets project env; secrets come from ~/.config/wsp/env (chmod 600)
export WSP_CANON="$HOME/projects/wsp-lab"
export WSP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$WSP_CANON/data/hf"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$WSP_ROOT/src:${PYTHONPATH:-}"
[ -f "$HOME/.config/wsp/env" ] && set -a && . "$HOME/.config/wsp/env" && set +a
export PATH="$WSP_CANON/.venv/bin:$HOME/.local/bin:$PATH"
