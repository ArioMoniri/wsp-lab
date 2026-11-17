# shellcheck shell=bash
# Per-project CUDA isolation. Source it AFTER the project's venv is on PATH (or pass the venv dir):
#   source scripts/cuda_env.sh [VENV_DIR]
# What it does, for this shell only (nothing system-wide is changed):
#   1. reads the CUDA tag the venv's torch was built for (from torch/version.py; torch is not imported);
#   2. points CUDA_HOME at the matching /usr/local/cuda-<ver> toolkit if one is installed;
#   3. rebuilds LD_LIBRARY_PATH without any global /usr/local/cuda* entries (the AMI exports cuda-13.2 for every
#      login shell), so the venv's bundled nvidia-* wheels are used and a cu128 project never loads cu132 libs;
#   4. gives every JIT/kernel cache (Triton, Inductor, torch extensions, CUDA ComputeCache, vLLM, FlashInfer) its own
#      directory per project and CUDA tag, so projects built for different CUDA versions never share compiled kernels.
# Override: CUDA_ENV_PROJECT (cache namespace), CUDA_ENV_CACHE_ROOT (default ~/.cache/cuda-isolated).
_ce_venv="${1:-${VIRTUAL_ENV:-}}"
if [ -z "$_ce_venv" ]; then
  _ce_py="$(command -v python 2>/dev/null || true)"
  [ -n "$_ce_py" ] && _ce_venv="$(cd "$(dirname "$_ce_py")/.." && pwd)"
fi
_ce_tag="cpu"
_ce_ver=""
if [ -n "$_ce_venv" ]; then
  for _ce_f in "$_ce_venv"/lib/python3*/site-packages/torch/version.py; do
    [ -f "$_ce_f" ] || continue
    _ce_ver="$(sed -n "s/^cuda[^=]*= *['\"]\([0-9.]*\)['\"].*/\1/p" "$_ce_f" | head -1)"
    break
  done
fi
if [ -n "$_ce_ver" ]; then
  _ce_tag="cu${_ce_ver//./}"
  if [ -d "/usr/local/cuda-$_ce_ver" ]; then
    export CUDA_HOME="/usr/local/cuda-$_ce_ver"
  else
    unset CUDA_HOME
  fi
fi
# Drop every global CUDA toolkit entry; keep anything else (MPI, /usr/local/lib, ...).
_ce_ld=""
IFS=':' read -r -a _ce_parts <<< "${LD_LIBRARY_PATH:-}"
for _ce_p in "${_ce_parts[@]}"; do
  case "$_ce_p" in ""|/usr/local/cuda*) ;; *) _ce_ld="${_ce_ld:+$_ce_ld:}$_ce_p" ;; esac
done
# Only nvcc-built extensions need the toolkit libs; they go last so the venv's wheels win.
[ -n "${CUDA_HOME:-}" ] && _ce_ld="${_ce_ld:+$_ce_ld:}$CUDA_HOME/lib64"
export LD_LIBRARY_PATH="$_ce_ld"
if [ -n "${CUDA_HOME:-}" ]; then
  _ce_path=""
  IFS=':' read -r -a _ce_parts <<< "$PATH"
  for _ce_p in "${_ce_parts[@]}"; do
    case "$_ce_p" in /usr/local/cuda*/bin) ;; *) _ce_path="${_ce_path:+$_ce_path:}$_ce_p" ;; esac
  done
  export PATH="$_ce_path:$CUDA_HOME/bin"
fi
_ce_proj="${CUDA_ENV_PROJECT:-$(basename "$(dirname "${_ce_venv:-$PWD/x}")")}"
_ce_root="${CUDA_ENV_CACHE_ROOT:-$HOME/.cache/cuda-isolated}/$_ce_proj/$_ce_tag"
export TRITON_CACHE_DIR="$_ce_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$_ce_root/inductor"
export TORCH_EXTENSIONS_DIR="$_ce_root/torch_extensions"
export CUDA_CACHE_PATH="$_ce_root/nv-computecache"
export VLLM_CACHE_ROOT="$_ce_root/vllm"
export FLASHINFER_WORKSPACE_BASE="$_ce_root/flashinfer"
export CUDA_ENV_TAG="$_ce_tag"
mkdir -p "$_ce_root" 2>/dev/null || true
unset _ce_venv _ce_py _ce_f _ce_ver _ce_tag _ce_ld _ce_parts _ce_p _ce_path _ce_proj _ce_root
