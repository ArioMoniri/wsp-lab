#!/bin/bash
# F12-I streaming LoRA plan (scripts/f12_additivity_stream.py), 2 threads, peak-RSS guard 4.5 GB + external watchdog.
# usage: run_stream.sh <tag> [driver args...]
set -o pipefail
TAG="$1"; shift
OUTD=/home/ubuntu/wsp-data/runs/a5/f12_inputs/c1_freeze
cd /home/ubuntu/projects/wsp-lab/.worktrees/prereg/f12-c1 || exit 1
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "[$(date -u +%FT%TZ)] start $TAG head $(git rev-parse --short HEAD) dirty=[$(git status --porcelain -- scripts src | tr '\n' ' ')]" | tee "$OUTD/$TAG.log"
free -b | tee -a "$OUTD/$TAG.log"
AVAIL=$(free -b | awk '/^Mem:/{print $7}')
if [ "$AVAIL" -lt 6000000000 ]; then echo "ABORT: available memory $AVAIL < 6 GB" | tee -a "$OUTD/$TAG.log"; exit 3; fi
python scripts/f12_additivity_stream.py --threads 2 --rss-limit-gb 4.5 \
  --zoo /home/ubuntu/projects/wsp-lab/data/zoo/qwen3.5-0.8b --g2 /home/ubuntu/projects/wsp-lab/data/runs/g2 "$@" \
  >> "$OUTD/$TAG.log" 2>&1 &
PID=$!
MAXKB=0
while kill -0 $PID 2>/dev/null; do
  KB=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
  if [ -n "$KB" ]; then
    [ "$KB" -gt "$MAXKB" ] && MAXKB=$KB
    if [ "$KB" -gt 4394531 ]; then echo "WATCHDOG: RSS ${KB} KB > 4.5 GB, killing $PID" >> "$OUTD/$TAG.log"; kill $PID; fi
  fi
  sleep 1
done
wait $PID; RC=$?
echo "[$(date -u +%FT%TZ)] EXIT=$RC watchdog_max_sampled_rss_kb=$MAXKB" >> "$OUTD/$TAG.log"
