#!/usr/bin/env bash
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh
ruff check src tests && ruff format --check src tests && pytest -q
if ! python scripts/verify_refs.py; then
  echo 'Warning: reference verification failed; paper review requires it to pass.' >&2
fi
# Non-blocking: refresh the README status block (writes README.md; never stages it, never fails the commit).
python scripts/update_readme_status.py || echo 'Warning: README status update failed.' >&2
