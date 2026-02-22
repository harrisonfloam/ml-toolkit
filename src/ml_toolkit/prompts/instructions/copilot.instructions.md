---
description: Global instructions for all chats.
---
## General guidance
- Prefer the smallest correct change; avoid speculative complexity.
- Do not make architectural decisions before they are necessary; defer decisions to the user when possible.
- Avoid new helper functions and abstractions on initial implementations; suggest refactor once patterns stabilize.
- When uncertain, state assumptions explicitly.
- When using a tool, agent, or strategy, note any issues and attempt to solve them.
- Suggest creating new tools, agents, prompts, or skills when they would be helpful.

## Naming and documentation
- Use clear, descriptive, and concise names and comments
- Avoid numbered comments; write concise, professional prose instead.
- Do not append noisy labels like "v2 (updated with ...)" to names, headings, or comments.

## Tools
- Use available tooling to reduce manual work and avoid re-discovering context; record strategies and issues.
- Use your native To Do tool to break down non-trivial tasks.
- **mem0**: If available, use as system memory; always use 'app_id': 'code'.
- **upstash/context7**: Use for all external documentation queries.
- **oraios/serena**: Enables symbol-first code retrieval and editing; always attempt to use for code changes. Be sure to activate Serena for each new workspace.

## Agents
- **utility**: Use as a subagent for all token-heavy support work that doesn’t require the primary model (e.g., search, doc lookup, summarization, git operations, log digestion).

## Agent-authored files
- When helpful, write intermediate artifacts to workspace-local files so progress is inspectable and resumable.
- Default location: `/.agent/` (create if missing).
- Examples:
  - `/.agent/plan.md`: current intent, decisions, and milestones
  - `/.agent/todo.md`: user-originated backlog, organized and deduplicated
  - `/.agent/research.md`: evidence, links, citations, and relevant repo paths
  - `/.agent/strategies.md`: repeatable workflows, tooling notes, and known edge cases
  - `/.agent/feature-x.md`: design notes and implementation details for feature x
- Keep these as working notes, not canonical requirements. Prefer to keep them untracked unless the user asks otherwise.