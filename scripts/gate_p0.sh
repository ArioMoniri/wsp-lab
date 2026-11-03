#!/usr/bin/env bash
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh
python -c "import torch,transformers,safetensors,optuna,lm_eval;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
python -c "import heretic; print('heretic ok')" || echo "heretic import failed"
python -c "import mamba_ssm; print('mamba-ssm ok')" || echo "mamba-ssm absent -> torch fallback (expected until installed)"
pytest -q && pytest -q -m gpu && echo "P0 GATE GREEN"
