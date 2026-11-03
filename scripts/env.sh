# source this: sets project env; secrets come from ~/.config/wsp/env (chmod 600)
export WSP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$HOME/projects/wsp-lab/data/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$WSP_ROOT/src:${PYTHONPATH:-}"
[ -f "$HOME/.config/wsp/env" ] && set -a && . "$HOME/.config/wsp/env" && set +a
export PATH="$WSP_ROOT/.venv/bin:$HOME/.local/bin:$PATH"
