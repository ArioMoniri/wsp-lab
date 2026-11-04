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

On this CUDA 13 / PyTorch 2.15 host, run `bash scripts/build_mamba.sh` after installing PyTorch and build dependencies in the shared venv. The script downloads pinned causal-conv1d 1.7.0 and mamba-ssm 2.3.2.post1 source distributions, patches C++17 flags to C++20, and installs with `--no-build-isolation --no-deps`, `TORCH_CUDA_ARCH_LIST=12.0` and `MAX_JOBS=4`. The successful build took approximately 11 minutes; log: `~/agent-logs/wsp-mamba3b-20261103-235407.log`. Both import and a CUDA Mamba2 forward succeeded.
