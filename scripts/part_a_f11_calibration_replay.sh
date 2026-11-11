#!/bin/bash
cd /home/ubuntu/projects/wsp-lab || exit 1
echo "START $(date -u +%FT%TZ) HEAD $(git rev-parse HEAD) dirty $(git status --porcelain | wc -l) mask $(sha256sum docs/prereg/a5_manifests/part_a/slot_mask_f11.json | cut -c1-64)"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/projects/wsp-lab/.venv/bin/python scripts/a5_simulation.py data/runs/a5/nullsize/inputs.json --fields data/runs/a5/power/fields.json --donor-ids docs/prereg/a5_manifests/power/registered_donors.json --supplement docs/prereg/a5_manifests/selection_f11.json --slot-mask docs/prereg/a5_manifests/part_a/slot_mask_f11.json --scenario registered --output /home/ubuntu/wsp-data/runs/a5/f11_sim_dryrun/boarda/power_slotmask_f11_replay.json
RC=$?; echo "END $(date -u +%FT%TZ)"; echo "REPLAY_EXIT $RC"
