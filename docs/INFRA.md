# Infrastructure

## This server
- AWS `eu-central-1`, instance `<instance-id>` (g7.2xlarge), host `<aws-host>`
- GPU: NVIDIA RTX PRO 4500 Blackwell Server Edition, 32,623 MiB, compute capability 12.0, driver 595.112, CUDA 13.2
- 8 vCPU, 30 GiB RAM, 675 GB free disk (`df -h /` on 4 Nov 2026 06:27 UTC: 991G size, 317G used, 675G available)
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
- `scripts/lambda.sh types|ls|launch <type> <region> <ssh-key-name>|terminate <id>`; launch requires `MAX_USD_PER_H`;
  terminate refuses any instance whose name does not start with `wsp-lab`.
- Remote bootstrap: `scripts/lambda_bootstrap.sh <git-url>`, run on the instance, installs git, rsync, tmux and uv,
  clones this repo to `~/projects/wsp-lab`, and creates `.venv` with `.[dev,serve]`. It pulls no models and copies
  nothing back here. It prints the next steps: put `HF_TOKEN` and `LAMBDA_API_KEY` into `~/.config/wsp/env`, then
  `scripts/hf_sync.sh pull zoo`.
- Job script: `scripts/lambda_job_ara.sh <n> <trials>` (defaults 30 30), run on the instance. It pulls the repo,
  reinstalls torch from the CUDA wheel index that matches the host driver (12.6/12.7 → cu126, 12.8/12.9 → cu128,
  13.x → cu132, otherwise cu128) only if torch cannot see CUDA, pulls the zoo from HF, builds the `heretic_ara` family, pushes the zoo back to HF, and
  terminates the `wsp-lab*` instance. Results return through HF, not through rsync.
- Needed from the user: `LAMBDA_API_KEY` (asked when P5 starts) and an SSH key registered in Lambda.

## mamba-ssm

On this CUDA 13 / PyTorch 2.15 host, run `bash scripts/build_mamba.sh` after installing PyTorch and build dependencies in the shared venv. The script downloads pinned causal-conv1d 1.7.0 and mamba-ssm 2.3.2.post1 source distributions, patches C++17 flags to C++20, and installs with `--no-build-isolation --no-deps`, `TORCH_CUDA_ARCH_LIST=12.0` and `MAX_JOBS=4`. The successful build took approximately 11 minutes; log: `~/agent-logs/wsp-mamba3b-20261103-235407.log`. Both import and a CUDA Mamba2 forward succeeded.

## Artefact store: Hugging Face (private datasets)
GitHub rejects files > 100 MB and the zoo will reach tens of GB, so edited-model deltas, LoRA adapters, refusal
caches and eval records live in private HF dataset repos (user decision 4 Nov 2026):
`Aralario/wsp-lab-zoo`, `Aralario/wsp-lab-evals`, `Aralario/wsp-lab-runs`. Sync with `scripts/hf_sync.sh push|pull <kind>`
after each zoo/eval batch. They stay private because abliteration deltas are, in effect, uncensoring edits
(see docs/VENUE.md broader-impact note); only statistics are published. Lambda instances pull from HF, not from this server.

## README status block
`python scripts/update_readme_status.py` rewrites the text between `<!-- status:start -->` and `<!-- status:end -->` in
`README.md` from artefacts: zoo entries by family and eligibility (`data/zoo/qwen3.5-0.8b/manifest.jsonl`, duplicate
ids collapsed, header rows ignored), feature-extraction entries (`data/features/qwen3.5-0.8b/*/features.json`), eval
index rows (`data/evals/*/index.jsonl`), the base-model summary (`data/runs/p2/base-final-4096/summary.json`), open
rows of `board/lambda_spend.md` (no end time), and the latest `main` pipeline from GitLab when `glab` is installed and
logged in (omitted otherwise). A missing source is printed as missing. `--no-ci` skips the GitLab call, `--print`
echoes the block, `--root` points at another checkout. Running it twice changes only the timestamp. `scripts/precommit.sh`
runs it as a last, non-blocking step and stages `README.md` when the refresh succeeds (not after a failure); the test is `tests/test_readme_status.py`.

- Lambda note (4 Nov 2026): on-demand A100 hosts run driver 570 (CUDA 12.8). Install torch from the `cu128`
  index there (the job script does this automatically); mamba-ssm is not needed for zoo/eval jobs.

### Opaque evaluation device identity

Set `~/.config/wsp/device_id` outside the repository to an opaque `H<number>` identifier.
Absent files default to `H0`; allocate distinct IDs privately before cross-device runs.
Only the opaque ID and GPU class enter evaluation artifacts. Keep the mapping to actual
infrastructure private; do not put hostnames or instance IDs in this file or artifacts.
