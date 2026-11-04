#!/usr/bin/env bash
# Run ON the Lambda instance: bootstrap repo + venv. Usage: curl … | bash -s -- <git-remote-or-rsync-src>
set -euo pipefail
sudo apt-get update -qq && sudo apt-get install -y -qq git rsync tmux >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/projects && cd ~/projects
[ -d wsp-lab ] || git clone "$1" wsp-lab; cd wsp-lab
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,serve]"
mkdir -p ~/.config/wsp && echo "put HF_TOKEN and LAMBDA_API_KEY into ~/.config/wsp/env (chmod 600) then: scripts/hf_sync.sh pull zoo"
echo "bootstrap done; run jobs under tmux; push results with scripts/hf_sync.sh push evals|runs (the AWS server pulls from HF)"
