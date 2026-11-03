#!/usr/bin/env bash
# git worktree helper: wt.sh new <branch> | ls | rm <branch> | merge <branch>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
case "${1:-}" in
  new)  b="$2"; git worktree add -b "$b" ".worktrees/$b" main >/dev/null; ln -sfn "$ROOT/.venv" ".worktrees/$b/.venv"; ln -sfn "$ROOT/data" ".worktrees/$b/data"; echo "$ROOT/.worktrees/$b" ;;
  ls)   git worktree list ;;
  rm)   b="$2"; git worktree remove --force ".worktrees/$b"; git branch -D "$b" ;;
  merge) b="$2"; git merge --no-ff -m "merge $b" "$b" ;;
  *) echo "usage: wt.sh new|ls|rm|merge <branch>"; exit 1 ;;
esac
