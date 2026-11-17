#!/usr/bin/env bash
# Code-only bundle for the full-refit CPU job (no data, no panel items).   wme_refit_bundle.sh <out.tgz>
set -euo pipefail
OUT="${1:?out.tgz}"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"; STAGE="$(mktemp -d)/wmerefit"
mkdir -p "$STAGE/scripts"; cp -r "$ROOT/src" "$STAGE/src"; find "$STAGE/src" -name __pycache__ -prune -exec rm -rf {} +
cp "$ROOT/scripts/wme_powersim_refit.py" "$ROOT/scripts/wme_refit_remote.sh" "$STAGE/scripts/"
tar -C "$(dirname "$STAGE")" -czf "$OUT" wmerefit && echo "$OUT"
