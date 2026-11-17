#!/usr/bin/env bash
# Pilot-3 code bundle: src/, the pilot-3 scripts, the AWS venv freeze, and the wme-p3s / wme-p3h panels (open splits
# and manifests only; pilot 3 writes no sealed split).   wme_p3_bundle.sh <out.tgz>
set -euo pipefail
OUT="${1:?out.tgz}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)/wmep3"
mkdir -p "$STAGE/scripts" "$STAGE/data/wme/panels"
cp -r "$ROOT/src" "$STAGE/src"
find "$STAGE/src" -name __pycache__ -prune -exec rm -rf {} +
cp "$ROOT/scripts/wme_pilot3_s.py" "$ROOT/scripts/wme_pilot3_h.py" "$ROOT/scripts/wme_p3_remote.sh" "$ROOT/scripts/wme_p3_watch.sh" "$STAGE/scripts/"
for p in wme-p3s wme-p3h; do
  mkdir -p "$STAGE/data/wme/panels/$p/open"
  cp "$ROOT/data/wme/panels/$p/manifest.json" "$STAGE/data/wme/panels/$p/"
  cp "$ROOT/data/wme/panels/$p/open/"*.jsonl "$STAGE/data/wme/panels/$p/open/"
done
{ "$ROOT/.venv/bin/python" -m pip freeze 2>/dev/null || uv pip freeze --python "$ROOT/.venv/bin/python"; } \
  | grep -vE '^-e |@ file:' > "$STAGE/constraints.txt"
grep -q '^torch==' "$STAGE/constraints.txt"
if find "$STAGE" -path '*sealed*' | grep -q .; then echo "sealed path in bundle" >&2; exit 1; fi
tar -C "$(dirname "$STAGE")" -czf "$OUT" wmep3
tar -tzf "$OUT" | grep -v '^wmep3/src/'
