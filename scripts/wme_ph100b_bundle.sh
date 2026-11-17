#!/usr/bin/env bash
# P-H100b code bundle: src/, the P-H100 scripts, the AWS venv freeze, and ONLY the wme-p2 panel manifest plus its
# open construction split (never validation, dev or sealed_test).   wme_ph100b_bundle.sh <out.tgz>
set -euo pipefail
OUT="${1:?out.tgz}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)/wmeph"
mkdir -p "$STAGE/scripts" "$STAGE/data/wme/panels/wme-p2/open"
cp -r "$ROOT/src" "$STAGE/src"
find "$STAGE/src" -name __pycache__ -prune -exec rm -rf {} +
cp "$ROOT/scripts/wme_ph100.py" "$ROOT/scripts/wme_ph100b_remote.sh" "$STAGE/scripts/"
P="$ROOT/data/wme/panels/wme-p2"
cp "$P/manifest.json" "$STAGE/data/wme/panels/wme-p2/"
cp "$P/open/construction.jsonl" "$STAGE/data/wme/panels/wme-p2/open/"
{ "$ROOT/.venv/bin/python" -m pip freeze 2>/dev/null || uv pip freeze --python "$ROOT/.venv/bin/python"; } \
  | grep -vE '^-e |@ file:' > "$STAGE/constraints.txt"   # local editable and locally built wheels are not portable
grep -q '^torch==' "$STAGE/constraints.txt"
if find "$STAGE" -path '*sealed*' | grep -q .; then echo "sealed path in bundle" >&2; exit 1; fi
tar -C "$(dirname "$STAGE")" -czf "$OUT" wmeph
tar -tzf "$OUT" | grep -v '^wmeph/src/' 
