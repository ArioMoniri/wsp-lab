#!/usr/bin/env bash
# session.sh start "<title>" | current | note "<text>"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; S="$ROOT/sessions"; mkdir -p "$S"
case "${1:-}" in
  start) d=$(date -u +%F); n=$(printf '%03d' $(( $(ls "$S" | grep -c "^$d-") + 1 ))); f="$S/$d-$n.md"
         printf '# Session %s-%s — %s\n\nStarted: %s UTC\nHost: %s\nGit: %s\n\n## Log\n' "$d" "$n" "$2" "$(date -u +%FT%TZ)" "$(hostname)" "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo none)" > "$f"; echo "$f" ;;
  current) ls -1 "$S"/*.md | tail -1 ;;
  note) f=$(ls -1 "$S"/*.md | tail -1); printf -- '- %s %s\n' "$(date -u +%H:%M)" "$2" >> "$f" ;;
  *) echo "usage: session.sh start <title>|current|note <text>"; exit 1 ;;
esac
