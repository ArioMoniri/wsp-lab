#!/usr/bin/env bash
# Stage-W2 code bundle (RULING-wme-stagew2): src/, the W2 job scripts, the wme-w2 dev panel, the exposed wme-p3h open
# splits (canary, teacher prompts) and the pilot-3 teacher rows. No sealed path.
#   wme_w2_bundle.sh <out.tgz> [synced out/w2 dir of a slot, for a relaunch or top-up that resumes it (stop rules 4-5)]
set -euo pipefail
OUT="${1:?out.tgz}"; RESUME="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)/wmew2"
mkdir -p "$STAGE/scripts" "$STAGE/data/wme/panels" "$STAGE/data/wme/stagew"
cp -r "$ROOT/src" "$STAGE/src"
find "$STAGE/src" -name __pycache__ -prune -exec rm -rf {} +
cp "$ROOT/scripts/wme_stagew2.py" "$ROOT/scripts/wme_w2_remote.sh" "$STAGE/scripts/"
for p in wme-w2 wme-p3h; do
  mkdir -p "$STAGE/data/wme/panels/$p/open"
  cp "$ROOT/data/wme/panels/$p/manifest.json" "$STAGE/data/wme/panels/$p/"
  cp "$ROOT/data/wme/panels/$p/open/"*.jsonl "$STAGE/data/wme/panels/$p/open/"
done
cp "$ROOT/data/wme/stagew/teacher.jsonl" "$STAGE/data/wme/stagew/"
if [ -n "$RESUME" ]; then
  mkdir -p "$STAGE/out/w2" && cp -r "$RESUME/." "$STAGE/out/w2/"
  rm -f "$STAGE/out/w2/SHA256SUMS" "$STAGE/out/w2/STATUS_eval"
fi
if find "$STAGE" -path '*sealed*' | grep -q .; then echo "sealed path in bundle" >&2; exit 1; fi
tar -C "$(dirname "$STAGE")" -czf "$OUT" wmew2
tar -tzf "$OUT" | grep -v '^wmew2/src/'
