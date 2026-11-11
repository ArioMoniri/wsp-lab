#!/bin/bash
cd /home/ubuntu/projects/wsp-lab || exit 1
echo "PHASE1 WRAPPER START $(date -u +%FT%TZ) HEAD $(git rev-parse HEAD) dirty $(git status --porcelain | wc -l)"
scripts/part_a_score_launch.sh 71e77a3e1f703451864ce960129399be1579d8aa84631bebc0c8d6324b4790d6
RC=$?; echo "PHASE1 WRAPPER END $(date -u +%FT%TZ)"; echo "PHASE1_EXIT $RC"
