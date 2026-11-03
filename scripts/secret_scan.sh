#!/usr/bin/env bash
# Blocks commits/pushes containing secret-looking strings. Used by .githooks/pre-commit and pre-push.
set -e; cd "$(dirname "$0")/.."
PAT='ip-172-31-31-23|i-0c76db91358eee850|172\.31\.[0-9]+\.[0-9]+|secret_wsp_[A-Za-z0-9]+|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|gho_[A-Za-z0-9]{30,}|glpat-[A-Za-z0-9_-]{20}|sk-[A-Za-z0-9]{32,}|-----BEGIN (RSA|OPENSSH) PRIVATE KEY'
if git diff --cached | grep -qE "$PAT"; then echo "secret_scan: staged diff contains a secret-looking token; aborting"; exit 1; fi
if [ "${1:-}" = "--history" ]; then git log -p --all | grep -cE "$PAT" | grep -q '^0$' || { echo "secret_scan: history contains secrets"; exit 1; }; fi
echo "secret_scan: clean"
