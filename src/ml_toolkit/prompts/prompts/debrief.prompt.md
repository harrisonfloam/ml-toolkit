---
name: debrief
description: Produce a concise, structured debrief of the conversation.
---
Review the entire conversation and extract all relevant details.

## Structure
Output a concise debrief with clear structure:
- **User input**: explicit or infered intent, key facts, requirements, constraints, decisions, preferences, open questions (quote/paraphrase only what the user provided).
- **Assistant content**: proposals, recommendations, assumptions made, actions taken, artifacts produced.
- **Current state**: what is implemented/changed vs planned, what remains unresolved.
- **Next steps**: short, ordered list of actionable items.

## Guidelines
- Be terse; Avoid fluff.
- Do not invent details.
- If uncertain, label it explicitly.
- Respond only with the debrief; do not include any meta commentary or process notes.
- If the conversation is long or the context is large, delegate the scan/extraction pass to the `utility` agent, then write the final debrief yourself.