# Definition of Done

A task / PR / worktree is done when **all** of the following hold:

- [ ] Tests written first, all tiers that apply are green; new behaviour has at least one test.
- [ ] `ruff check src tests` and `ruff format --check` pass.
- [ ] Any run longer than 2 min was launched through `agent-tmux <name> …` and its log path is referenced
      in the session note.
- [ ] Data artefacts are under `data/` with a `manifest.json` (seed, config hash, git SHA, model revisions).
- [ ] Config used is committed under `configs/` and referenced by the artefact manifest.
- [ ] No demo/mock/placeholder values in paper, docs, figures or analyses (real run artefacts only, run id in caption).
- [ ] Numbers that will be quoted appear in `docs/RESULTS.md` with n, CI method and the run id.
- [ ] Session note in `sessions/YYYY-MM-DD-NNN.md` updated: what changed, what was learned, open questions.
- [ ] If a hypothesis gate was touched, `docs/ROADMAP.md` checkbox updated.
- [ ] No secrets in the repo (`git diff --cached | grep -iE "key|token"` is clean); keys live in `~/.config/wsp/env`.
- [ ] Worktree merged to `main` with a fast-forward or a merge commit whose message names the phase (P1…P6).
