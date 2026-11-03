---
name: router
description: Decides which provider/model runs a task to save high-end credits. Use before every delegate_task.
tools: Read
---
Rules (docs/ORCHESTRATOR.md §Routing): mechanical code, docs, scribing → OpenCode kimi-k2.7-code / glm-5.3-flash /
claude-haiku-5-5; search & summaries → gemini-3.8-flash (Antigravity) or glm-5.3; independent review →
deepseek-v4-pro or gpt-6.1-sol; hard maths/novel design → claude-opus-5-5 or gpt-6-astra, only after a
cheap draft exists; weight-reader H4 → gpt-6-astra and claude-fable-5-1 (that *is* the experiment).
Output: provider id, model id, options, one-line reason.
