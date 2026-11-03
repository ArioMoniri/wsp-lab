# AGENTS.md — rules for every agent working in wsp-lab

- Read `README.md`, `docs/HYPOTHESIS.md`, `docs/ROADMAP.md`, `docs/TDD.md`, `docs/DOD.md` before touching code.
- Python: `source .venv/bin/activate` (uv venv, Python 3.12). Package is `wsp` under `src/`.
- Tests first. `pytest -q` must stay under 60 s on CPU. GPU tests are marked `gpu`.
- Long jobs (> 2 min): `agent-tmux <name> <cmd>`; never a bare background process; never another tmux server.
- Never stop/reboot the instance. Never commit weights, data, or secrets. Secrets: `~/.config/wsp/env`.
- Work in a worktree: `scripts/wt.sh new <branch>`; merge to `main` only when `docs/DOD.md` is met.
- Every artefact has a `manifest.json` with seed, config hash, git SHA, model revisions.
- Record what you did in the current `sessions/*.md` (scripts/session.sh current).
- Role definitions: `.claude/agents/*.md`. Orchestration: `docs/ORCHESTRATOR.md`.
- The GitHub repo is PUBLIC-facing: never paste keys, tokens, IPs of rented instances, or raw logs into commits,
  board posts, docs, or sessions. `scripts/secret_scan.sh` runs on commit and push; `board.sh` redacts tokens.
- Lambda instances must be terminated right after a job (`scripts/lambda.sh terminate <id>`); only touch
  instances whose name starts with `wsp-lab`.
