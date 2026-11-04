#!/usr/bin/env bash
# git worktree helper: wt.sh new <branch> | ls | rm <branch> | merge <branch>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
case "${1:-}" in
  new)  b="$2"; git worktree add -b "$b" ".worktrees/$b" main >/dev/null; for d in .venv data; do [ -d ".worktrees/$b/$d" ] && [ ! -L ".worktrees/$b/$d" ] && rm -rf ".worktrees/$b/$d"; ln -sfn "$ROOT/$d" ".worktrees/$b/$d"; done; echo "$ROOT/.worktrees/$b" ;;
  ls)   git worktree list ;;
  check) for d in .worktrees/*/* .worktrees/*; do [ -e "$d/data" ] || continue; if [ -L "$d/data" ]; then echo "ok   $d/data -> $(readlink "$d/data")"; else echo "REAL $d/data (artefacts not in canonical data/; consolidate with: rsync -a $d/data/ data/ && rm -rf $d/data && ln -s $ROOT/data $d/data)"; fi; done ;;
  rm)   b="$2"; git worktree remove --force ".worktrees/$b"; git branch -D "$b" ;;
  merge) b="$2"; git merge --no-ff -m "merge $b" "$b" ;;
  *) echo "usage: wt.sh new|ls|check|rm|merge <branch>"; exit 1 ;;
esac
