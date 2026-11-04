#!/usr/bin/env bash
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh
ruff check src tests && ruff format --check src tests && pytest -q
if ! python scripts/verify_refs.py; then
  echo 'Warning: reference verification failed; paper review requires it to pass.' >&2
fi
# Non-blocking: refresh the README status block, and stage README.md only if the refresh succeeded.
# Hand edits to README.md that were not staged yet are staged along with it, so commit from a clean README.
if python scripts/update_readme_status.py; then
  git add -- README.md
else
  echo 'Warning: README status update failed; README.md not staged.' >&2
fi
