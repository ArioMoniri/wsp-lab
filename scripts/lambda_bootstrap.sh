#!/usr/bin/env bash
# Run ON the Lambda instance: bootstrap repo + venv. Usage: curl … | bash -s -- <git-remote-or-rsync-src>
set -euo pipefail
sudo apt-get update -qq && sudo apt-get install -y -qq git rsync tmux >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/projects && cd ~/projects
[ -d wsp-lab ] || git clone "$1" wsp-lab; cd wsp-lab
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,serve]"
echo "bootstrap done; run jobs under tmux and rsync data/evals data/features back to the AWS server"
