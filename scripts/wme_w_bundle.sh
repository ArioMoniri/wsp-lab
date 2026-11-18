#!/usr/bin/env bash
# Stage-W code bundle (RULING-wme-stagew): src/, the stage-W job scripts, the wme-w dev panel, the exposed wme-p3h
# open splits (canary, timing, teacher prompts) and the pilot-3 teacher rows. No sealed path.   wme_w_bundle.sh <out.tgz>
set -euo pipefail
OUT="${1:?out.tgz}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)/wmew"
mkdir -p "$STAGE/scripts" "$STAGE/data/wme/panels" "$STAGE/data/wme/stagew"
cp -r "$ROOT/src" "$STAGE/src"
find "$STAGE/src" -name __pycache__ -prune -exec rm -rf {} +
cp "$ROOT/scripts/wme_stagew.py" "$ROOT/scripts/wme_w_remote.sh" "$STAGE/scripts/"
for p in wme-w wme-p3h; do
  mkdir -p "$STAGE/data/wme/panels/$p/open"
  cp "$ROOT/data/wme/panels/$p/manifest.json" "$STAGE/data/wme/panels/$p/"
  cp "$ROOT/data/wme/panels/$p/open/"*.jsonl "$STAGE/data/wme/panels/$p/open/"
done
cp "$ROOT/data/wme/stagew/teacher.jsonl" "$STAGE/data/wme/stagew/"
if find "$STAGE" -path '*sealed*' | grep -q .; then echo "sealed path in bundle" >&2; exit 1; fi
tar -C "$(dirname "$STAGE")" -czf "$OUT" wmew
tar -tzf "$OUT" | grep -v '^wmew/src/'
