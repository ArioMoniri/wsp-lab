#!/usr/bin/env bash
# Nightly versioned copy of the canonical data root to the account's in-region disaster-recovery bucket (D8 backup control,
# approved by the user on 10 Nov 2026). The bucket is versioned, so an accidental local deletion is NOT propagated as a loss:
# `sync` without --delete never removes objects, and overwritten objects keep their previous versions. Deletes are not denied
# at the IAM level (the instance role cannot change IAM); that part of the D8 plan stays open.
set -uo pipefail
SRC=/home/ubuntu/wsp-data
DST=s3://t3-dr-235229001983-eu-central-1/wsp-lab/wsp-data
LOG=/home/ubuntu/agent-logs/backup-sync-$(date -u +%Y%m%d-%H%M%S).log
{
  date -u +"start %FT%TZ"
  aws s3 sync "$SRC" "$DST" --only-show-errors --no-progress --exclude "runs/d8/stage_zoo/*" --exclude "runs/d8/staging_zoo/*" --exclude "hf/*"
  rc=$?
  date -u +"end %FT%TZ rc=$rc"
  aws s3 ls --summarize --recursive "$DST" 2>/dev/null | tail -2
} > "$LOG" 2>&1
exit ${rc:-0}
