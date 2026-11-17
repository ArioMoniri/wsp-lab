#!/usr/bin/env bash
# Code-only bundle for the v2b CPU simulation jobs (no data, no panel items).   wme_v2b_bundle.sh <out.tgz>
set -euo pipefail
OUT="${1:?out.tgz}"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"; STAGE="$(mktemp -d)/wmev2b"
mkdir -p "$STAGE/scripts"; cp -r "$ROOT/src" "$STAGE/src"; find "$STAGE/src" -name __pycache__ -prune -exec rm -rf {} +
cp "$ROOT/scripts/wme_v2b_sim.py" "$ROOT/scripts/wme_v2b_remote.sh" "$STAGE/scripts/"
if find "$STAGE" -path '*sealed*' -o -path '*/data/*' | grep -q .; then echo "data path in bundle" >&2; exit 1; fi
tar -C "$(dirname "$STAGE")" -czf "$OUT" wmev2b && echo "$OUT"
