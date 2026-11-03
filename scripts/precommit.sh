#!/usr/bin/env bash
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh
ruff check src tests && ruff format --check src tests && pytest -q
