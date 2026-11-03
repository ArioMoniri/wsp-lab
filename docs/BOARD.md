# Agent message board

All agents (any provider) talk on one GitHub issue per phase, labelled `board`, plus a local mirror
`board/<phase>.md` (so it works offline and is in git). Protocol:

1. **Before editing**: post `CLAIM` with files you will touch and why. If another agent holds a claim on a
   file (open `board/<phase>.md`, search for `CLAIM` without a later `RELEASE`), coordinate in the thread
   instead of editing.
2. **After finishing**: post `DONE` with files changed, tests run, tmux log paths, and what you learned.
3. **Reviews** post `REVIEW <task> <verdict>`.
4. **Questions** post `ASK @role …`; the orchestrator routes them.

Commands (`scripts/board.sh`): `claim <agent> "<files>" "<why>"`, `done <agent> "<files>" "<summary>"`,
`review <task> <verdict> "<summary>"`, `ask <agent> "<text>"`, `show`. Each posts to GitHub (gh CLI) and
appends to the local mirror; if GitHub is unreachable it appends locally and queues in `board/.queue`.
