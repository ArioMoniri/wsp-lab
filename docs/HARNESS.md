# Harness² applied to this project (arXiv 2610.14224, Oct 2026)

Harness² ("recursive agent harnessing") optimises the runtime scaffolding around frozen models with two
coupled recursions: **task-level** (contrastive Propose–Probe–Compose: propose harness edits, probe them
under matched conditions, compose only the edits supported by behavioural contrast; parallel and sequential
variants) and **harnessing-level** (an improver harness keeps reusable components plus lessons on when and
why they worked, instead of inheriting the previous execution harness). Its eight components: C1 system
prompt, C2 guardrails, C3 long-term memory, C4 tool guidance, C5 sub-agents, C6 skills, C7 scripts,
C8 middleware/plugins. Reported gains: up to 11.5 pts WorkBuddy-Bench, 4.5 JobBench, 11.8 pp LAB from one
contrastive step; libraries transfer across benchmarks (paper §1, Fig. 1–2, 7).

## Mapping onto wsp-lab
| Harness² component | Here |
|---|---|
| C1 system prompt | `AGENTS.md` / `CLAUDE.md` |
| C2 guardrails | `docs/DOD.md`, `docs/TDD.md`, ruflo policy hooks in `.claude/settings.json` |
| C3 memory | ruflo hybrid memory (`.claude-flow/data`), `sessions/*.md`, `board/*.md` |
| C4 tool guidance | `docs/INFRA.md` (tmux, worktrees, Lambda) |
| C5 sub-agents | `.claude/agents/*.md`, mirrored in `opencode.jsonc` |
| C6 skills | `.agents/skills` (ruflo), Claude skills available on the server |
| C7 scripts | `scripts/*.sh` |
| C8 middleware | ruflo pre/post tool hooks |

## Our recursion loop (per phase)
1. **Propose**: two agents (cheap models) propose alternative implementations/analyses (the "compete" rule).
2. **Probe**: both run the same tests/evals under matched seeds; Boards A and B score both.
3. **Compose**: the orchestrator merges the winner and records *why* in `harness/lessons.md`
   (harnessing-level recursion): which role file, prompt line or script change made the difference.
4. Role files and `AGENTS.md` are edited from `harness/lessons.md` only, never ad hoc, so the improvement
   harness is itself versioned.
