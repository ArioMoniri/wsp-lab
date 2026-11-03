#!/usr/bin/env bash
# Agent message board: GitHub issue (label board) + local mirror board/<phase>.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PHASE="${WSP_PHASE:-P0}"; MIRROR="board/$PHASE.md"; mkdir -p board
REPO="${WSP_GH_REPO:-$(git remote get-url origin 2>/dev/null | sed -E 's#.*github.com[:/]##; s#\.git$##' || true)}"
issue_num() { gh issue list -R "$REPO" --label board --search "in:title [board] $PHASE" --json number --jq '.[0].number' 2>/dev/null || true; }
post() { local body="$1"; printf '%s\n' "$body" >> "$MIRROR"
  if [ -n "$REPO" ]; then n=$(issue_num); if [ -z "$n" ]; then n=$(gh issue create -R "$REPO" --label board --title "[board] $PHASE" --body "Agent message board for $PHASE. See docs/BOARD.md" --json number --jq .number 2>/dev/null || gh issue create -R "$REPO" --label board --title "[board] $PHASE" --body "Agent message board for $PHASE" | grep -oE '[0-9]+$'); fi
    gh issue comment "$n" -R "$REPO" --body "$body" >/dev/null 2>&1 || printf '%s\n' "$body" >> board/.queue; fi; }
ts() { date -u +%FT%TZ; }
case "${1:-}" in
  claim)  post "**CLAIM** $(ts) \`$2\` files: $3 — $4" ;;
  done)   post "**DONE** $(ts) \`$2\` files: $3 — $4" ;;
  release) post "**RELEASE** $(ts) \`$2\` files: $3" ;;
  review) post "**REVIEW** $(ts) task \`$2\` verdict **$3** — $4" ;;
  ask)    post "**ASK** $(ts) \`$2\`: $3" ;;
  show)   cat "$MIRROR" 2>/dev/null || echo "(empty)" ;;
  *) echo "usage: board.sh claim|done|release|review|ask|show …"; exit 1 ;;
esac
