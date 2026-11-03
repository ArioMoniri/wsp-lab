# Infrastructure

## This server
- AWS `eu-central-1`, instance `<instance-id>` (g7.2xlarge), host `<aws-host>`
- GPU: NVIDIA RTX PRO 4500 Blackwell Server Edition, 32,623 MiB, compute capability 12.0, driver 595.112, CUDA 13.2
- 8 vCPU, 30 GiB RAM, 922 GB free disk
- Nightly DR tarballs + AMIs exist (see `~/DR-RESTORE.md`), so project data under `~/projects` is backed up.

## Conventions
- Project root: `~/projects/wsp-lab`. venv: `.venv` (uv). Activate: `source .venv/bin/activate`.
- Long jobs: `agent-tmux <name> <cmd>`; list `agent-tmux -l`; tail `agent-tmux -t <name>`; logs in `~/agent-logs/`.
- Worktrees: `scripts/wt.sh new <branch>` → `.worktrees/<branch>`; `scripts/wt.sh ls|rm`.
- Sessions: `sessions/<date>-<nnn>.md`, created by `scripts/session.sh start "<title>"`.
- Secrets: `~/.config/wsp/env` (chmod 600), sourced by `scripts/env.sh`. Never commit.
- HF cache: `HF_HOME=~/projects/wsp-lab/data/hf` (set in `scripts/env.sh`) so models stay with the project.

## GPU bursting (Lambda Cloud)
- API: `https://cloud.lambda.ai/api/v1`, `Authorization: Bearer $LAMBDA_API_KEY`. 1 req/s, launch 1 per 12 s.
- Prices (on-demand, Nov 2026): 1× H100 SXM $4.29/h, 1× H100 PCIe $3.29/h, 1× B200 $6.99/h, 1× GH200 $2.29/h,
  1× A6000 48 GB $1.09/h.
- `scripts/lambda.sh types|launch <type> <region>|ls|ssh <id>|terminate <id>`; launch requires `MAX_USD_PER_H`.
- Remote bootstrap: `scripts/lambda_bootstrap.sh` clones this repo, makes the venv, pulls models, and
  `rsync`s `data/evals` and `data/features` back here (the server is the single source of truth).
- Needed from the user: `LAMBDA_API_KEY` (asked when P5 starts) and an SSH key registered in Lambda.

## mamba-ssm

On 2026-11-03, the latest 30 GitHub releases of state-spaces/mamba and
Dao-AILab/causal-conv1d had no torch 2.15 / CUDA 13 wheel. A source build failed in `wsp-mamba2`, log `~/agent-logs/wsp-mamba2-20261103-231214.log`:

```sh
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 CUDA_HOME=/usr/local/cuda
uv pip install --no-build-isolation causal-conv1d mamba-ssm
```

The causal-conv1d build system generates many CUDA architectures despite
`TORCH_CUDA_ARCH_LIST`. The mamba-ssm extension compilation uses C++17,
while the installed PyTorch headers require C++20 (`#error C++20 or later
compatible compiler is required`); `import mamba_ssm` still fails. The previous
build also failed compiling extension objects; see `~/agent-logs/wsp-mamba-20261103-225608.log`.

The reproducible C++20 patch build is `bash scripts/build_mamba.sh`. It fetches
both pinned PyPI sdists into `data/build`, changes every `-std=c++17` in their
setup scripts to `-std=c++20`, and installs each with uv using
`--no-build-isolation --no-deps`, `TORCH_CUDA_ARCH_LIST=12.0` and `MAX_JOBS=4`.
The first attempt found that the shared venv has no pip (log
`~/agent-logs/wsp-mamba3-20261103-235340.log`); the revised uv build is in
`~/agent-logs/wsp-mamba3b-20261103-235407.log`.
