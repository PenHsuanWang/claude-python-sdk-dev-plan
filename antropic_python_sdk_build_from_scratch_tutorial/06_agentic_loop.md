# Chapter 06 — The Agentic Loop

> **Goal:** Build `AgentService` — the heart of the system. Understand exactly how the `tool_use` → `tool_result` cycle works and why it is structured as a loop with a safety cap.

---

## 6.1 The Core Insight

Claude is a **stateless language model**.  It doesn't remember previous API calls.  To create an agent that reasons over multiple steps, you must:

1. Maintain the **full conversation history** yourself (the `session.messages` list)
2. Call Claude in a **loop** until it signals it is done (`stop_reason == "end_turn"`)
3. Between iterations, execute any requested tools and feed results back

This is the "manual tool runner" pattern — you write the loop, Claude drives it with `stop_reason`.

---

## 6.2 The Full `AgentService` Implementation

```python
# app/services/agent.py
"""AgentService — Application layer orchestrating the agentic loop."""

import logging
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message

from app.core.config import settings
from app.domain.exceptions import AgentLoopError
from app.domain.models import AgentSession
from app.services.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

# ── Single client instance (connection pooling) ──────────────────────────── #
_client = AsyncAnthropic(
    api_key=settings.anthropic_api_key.get_secret_value(),
    base_url=settings.anthropic_base_url,
    max_retries=settings.max_retries,
)

# Safety cap: prevents runaway loops on unexpected model behaviour
_MAX_LOOP_ITERATIONS = 10


def _extract_text(message: Message) -> str:
    """Pull the first TextBlock from a completed Claude response."""
    for block in message.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def _serialize_content(content: Any) -> Any:
    """Convert SDK ContentBlock objects to JSON-serialisable dicts.

    The SDK content blocks are Pydantic models. model_dump() produces a plain
    dict that can be stored and re-submitted in subsequent turns.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [
            block.model_dump() if hasattr(block, "model_dump") else block
            for block in content
        ]
    return content


class AgentService:
    """Orchestrates the Claude agentic loop for a single user turn."""

    async def run(self, session: AgentSession) -> str:
        """Drive the tool-use loop until Claude returns a final text answer.

        Args:
            session: An AgentSession whose ``messages`` list already contains
                     the user's current message as the last entry.

        Returns:
            The final natural language answer from Claude.

        Raises:
            AgentLoopError: If the loop exceeds the safety cap or terminates
                            in an unexpected state.
        """
        for iteration in range(_MAX_LOOP_ITERATIONS):
            logger.debug(
                "Agent loop iteration %d for session '%s'",
                iteration,
                session.session_id,
            )

            # ── Call Claude ──────────────────────────────────────────────── #
            response: Message = await _client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.max_tokens,
                tools=TOOL_DEFINITIONS,
                messages=session.messages,
            )
            logger.debug("Claude stop_reason=%s", response.stop_reason)

            # ── Case 1: Final answer ─────────────────────────────────────── #
            if response.stop_reason == "end_turn":
                final_text = _extract_text(response)
                session.add_assistant_message(_serialize_content(response.content))
                return final_text

            # ── Case 2: Tool call(s) requested ──────────────────────────── #
            if response.stop_reason == "tool_use":
                # Record the assistant's tool_use turn in history FIRST
                session.add_assistant_message(_serialize_content(response.content))

                # Execute every tool the model requested in this iteration
                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info(
                            "Executing tool '%s' with input: %s",
                            block.name,
                            block.input,
                        )
                        result = execute_tool(block.name, dict(block.input))
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,    # ← must match exactly
                            "content": result,
                        })

                # Feed ALL results back in a single user turn
                session.add_tool_results(tool_results)
                continue  # ← loop back and call Claude again

            # ── Unexpected stop_reason ───────────────────────────────────── #
            logger.warning(
                "Unexpected stop_reason '%s', aborting loop.",
                response.stop_reason,
            )
            raise AgentLoopError(
                f"Unexpected stop_reason '{response.stop_reason}' from Claude."
            )

        raise AgentLoopError(
            f"Agent loop exceeded the maximum of {_MAX_LOOP_ITERATIONS} iterations."
        )


# Module-level singleton
agent_service = AgentService()
```

---

## 6.3 Step-by-Step Loop Walkthrough

Let's trace through a query: **"Summarize report.txt"**

### Iteration 0

**Input to Claude** (`session.messages`):
```python
[{"role": "user", "content": "Summarize report.txt"}]
```

**Claude's response** (`stop_reason = "tool_use"`):
```python
response.content = [
    ToolUseBlock(
        type="tool_use",
        id="toolu_01abc",
        name="list_local_documents",
        input={}
    )
]
```

**Action:** Execute `list_local_documents()` → `"Available documents:\n- report.txt"`

**`session.messages` after iteration 0:**
```python
[
    {"role": "user", "content": "Summarize report.txt"},
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_01abc", "name": "list_local_documents", "input": {}}
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_01abc",
         "content": "Available documents:\n- report.txt"}
    ]},
]
```

### Iteration 1

**Claude's response** (`stop_reason = "tool_use"`):
```python
response.content = [
    ToolUseBlock(
        type="tool_use",
        id="toolu_02def",
        name="read_local_document",
        input={"file_name": "report.txt"}
    )
]
```

**Action:** Execute `read_local_document("report.txt")` → full file content

### Iteration 2

**Claude's response** (`stop_reason = "end_turn"`):
```python
response.content = [
    TextBlock(text="Here is a summary of report.txt: ...")
]
```

**Action:** Extract text, add to session history, return to caller.

**Total API calls: 3.** The user sees only the final text.

---

## 6.4 Critical Implementation Details

### Why `_serialize_content()`?

The Anthropic SDK returns content blocks as Pydantic model instances (e.g., `ToolUseBlock`).  If you store these objects in `session.messages` and re-submit them to the API, the SDK serializes them correctly.  But if you later need to persist sessions (Redis, PostgreSQL), you need plain dicts.

`_serialize_content()` calls `.model_dump()` on any SDK object to get a serializable dict:

```python
def _serialize_content(content: Any) -> Any:
    if isinstance(content, list):
        return [
            block.model_dump() if hasattr(block, "model_dump") else block
            for block in content
        ]
    return content
```

### Why Check `DOCS_DIR not in requested.parents`?

This is the path traversal check.  Even after `_validate_filename()` in the API layer strips path components, the tool layer does its own check as a **defense in depth** strategy.

### Why the Safety Cap (`_MAX_LOOP_ITERATIONS = 10`)?

In rare edge cases, a model might enter a loop (repeatedly requesting a tool that returns an error, then requesting it again).  The safety cap prevents:
- Runaway costs (10 API calls max per user turn)
- Infinite loops hanging the request
- Unexpected model behavior in production

10 iterations is generous for most tasks (the typical file summarization uses 2–4).

### Why Store the Assistant Turn Before Executing Tools?

```python
# FIRST: record what Claude said (its tool_use request)
session.add_assistant_message(_serialize_content(response.content))

# THEN: execute the tools
for block in response.content:
    if block.type == "tool_use":
        ...
```

If you execute tools first and then crash (exception), the session history would be missing the assistant's tool_use turn.  On retry, Claude would get a malformed history.  Always persist Claude's response before executing side effects.

---

## 6.5 Handling Multiple Parallel Tool Calls

Claude can request **multiple tools in a single turn**.  The loop handles this correctly by collecting all `tool_results` and returning them in one batch:

```python
# Claude may return multiple ToolUseBlocks in one response
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, dict(block.input))
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
        })

# All results fed back in one user turn
session.add_tool_results(tool_results)
```

The Anthropic API requires that **all tool results for a turn are included in a single `user` message**.  If you split them across multiple turns, the API will return a validation error.

---

## 6.6 The `end_turn` vs `tool_use` Decision Tree

```
Claude responds
      │
      ├─ stop_reason == "end_turn"
      │       └── Extract text → return to user ✓
      │
      ├─ stop_reason == "tool_use"  
      │       └── For each ToolUseBlock:
      │               └── execute_tool(name, input) → string
      │           Collect results → add_tool_results()
      │           continue → call Claude again
      │
      ├─ stop_reason == "max_tokens"
      │       └── AgentLoopError (handle by increasing max_tokens)
      │
      └─ anything else
              └── AgentLoopError (unexpected model behavior)
```

---

## Chapter Summary

| Concept | Key Takeaway |
|---|---|
| The loop | Call Claude → check stop_reason → execute tools → repeat |
| `end_turn` | Claude is done — extract text, return to user |
| `tool_use` | Execute all requested tools, feed all results back in one user turn |
| `tool_use_id` | Must exactly match between `tool_use` block and `tool_result` |
| Serialization | Call `.model_dump()` to convert SDK objects to storable dicts |
| Safety cap | Prevent runaway loops with a max iteration limit |
| Persist before execute | Add assistant message to history before running tools |

---

Next: [Chapter 07 → Session Management](./07_session_management.md)
