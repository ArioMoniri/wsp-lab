# Orchestrator & specialised sub-agents

Two layers, both configured in this repo:

1. **T3 `delegate_task`** (cross-provider): the main Claude thread delegates scoped tasks to the provider
   best suited to the role. Providers live on this server: Codex (`gpt-6-astra`, `gpt-6.1-sol`),
   Claude (`claude-opus-5-5`, `claude-fable-5-1`, `claude-sonnet-5-5`, `claude-haiku-5-5`),
   OpenCode (`opencode-go/kimi-k2.7-code`, `glm-5.3`, `deepseek-v4-pro`, `gemini-4-argon`),
   Grok (`grok-4.7`), Antigravity (`gemini-3.8-flash`, `gpt-oss-120b`).
2. **Ruflo v3.66** (`ruflo init` done in repo; hierarchical swarm, hybrid memory in `data/memory`) for
   background workers and shared vector memory across sessions.

## Roles ("characters") and default assignment

| Role file | Character | Default provider/model | Scope |
|---|---|---|---|
| `.claude/agents/zoo-engineer.md` | careful, seed-obsessed data engineer | OpenCode `kimi-k2.7-code` | `src/wsp/zoo`, `src/wsp/edits`, Heretic integration |
| `.claude/agents/eval-engineer.md` | sceptical benchmark reproducer | Codex `gpt-6.1-sol` (high) | `src/wsp/eval`, lm-eval wrappers, item-level logs |
| `.claude/agents/features-mathematician.md` | symmetry/linear-algebra nerd | Claude `claude-opus-5-5` | `src/wsp/features`, W2T canonicalisation, invariance tests |
| `.claude/agents/ssm-researcher.md` | Mamba/JEPA researcher | Claude `claude-fable-5-1` | `src/wsp/predictor`, training loops, ablations |
| `.claude/agents/weight-reader.md` | the blog's "Astra" character | Codex `gpt-6-astra` (xhigh) | H4: reads weight summaries, predicts Δy, explains |
| `.claude/agents/reviewer-ml.md`, `reviewer-journal.md`, `reviewer-arch.md` | Boards A, B, C (correctness, claims, architecture alternatives) | OpenCode `deepseek-v4-pro` | reviews every worktree against `docs/DOD.md` |
| `.claude/agents/infra-ops.md` | tmux/Lambda operator | Claude `claude-sonnet-5-5` | `scripts/`, Lambda bursts, rsync back |
| `.claude/agents/scribe.md` | documentation | OpenCode `claude-haiku-5-5` | `sessions/`, `docs/RESULTS.md` |

The same roles are mirrored in `opencode.jsonc` (for `opencode` CLI users) and in `.claude/agents/` (for
Claude Code). Briefs are the only context a child gets: always include the role file path, the phase,
the files to touch, the tests to make green, and the DoD.

## Delegation recipe
```
delegate_task(role="implementation", isolation="worktree", title="P1 zoo: heretic kernel sampler",
  target={providerInstanceId:"opencode", model:"opencode-go/kimi-k2.7-code"},
  task="Read .claude/agents/zoo-engineer.md, docs/ROADMAP.md §P1, docs/TDD.md. Implement … Make tests/… green.")
```
Reviews are always a new `delegate_task` with `role="review"`.
