# Chapter 04 — Domain Layer

> **Goal:** Build the two domain files — `models.py` and `exceptions.py` — that form the zero-dependency core of the agent.

---

## 4.1 Why a Separate Domain Layer?

The Anthropic SDK, FastAPI, and Pydantic all have their own abstractions for data.  By expressing your *business concepts* (a conversation session, an error hierarchy) in plain Python dataclasses, you decouple the core logic from every external library.

This means:
- Domain objects can be unit-tested without mocking FastAPI or Anthropic
- If Anthropic changes their SDK, only the infrastructure layer adapts
- The domain layer serves as living documentation of what the system *is*

---

## 4.2 The `AgentSession` Dataclass

```python
# app/domain/models.py
"""Core domain entities — ZERO external dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSession:
    """Represents a single user conversation thread.

    ``messages`` follows the Anthropic API format:
      [{"role": "user" | "assistant", "content": str | list[ContentBlock]}, ...]

    Content blocks may be plain strings, Anthropic SDK objects, or their dict
    representations — the SDK handles serialisation transparently.
    """

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: Any) -> None:
        """Accept either a plain string or a list of Anthropic ContentBlocks."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        """Wrap tool results in a ``user`` turn as required by the Anthropic API."""
        self.messages.append({"role": "user", "content": tool_results})
```

### Why `@dataclass`?

- No inheritance from FastAPI's `BaseModel` or SQLAlchemy's `Base`
- No external dependencies — only `dataclasses` (stdlib)
- `field(default_factory=list)` gives each instance its own fresh list (avoids the classic mutable default argument bug)

### The Three Message Mutation Methods

Each method encapsulates one kind of Anthropic API turn:

| Method | Role | Content Type | When Called |
|--------|------|-------------|-------------|
| `add_user_message(str)` | `"user"` | plain string | When the human types a message |
| `add_assistant_message(Any)` | `"assistant"` | string or content blocks list | After each Claude response |
| `add_tool_results(list)` | `"user"` | list of `tool_result` dicts | After executing tool calls |

Note that **tool results go in a `user` turn** — this is an Anthropic API requirement, not a design choice.

### Message State After a Typical Multi-Turn Session

```python
session.messages == [
    # User's original question
    {"role": "user", "content": "Summarize report.txt"},

    # Claude's first response (asks to use a tool)
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_01abc", "name": "list_local_documents", "input": {}}
    ]},

    # Tool result fed back as a user turn
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_01abc", "content": "Available documents:\n- report.txt"}
    ]},

    # Claude's second response (asks to read the file)
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_02def", "name": "read_local_document", "input": {"file_name": "report.txt"}}
    ]},

    # Second tool result
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_02def", "content": "Q3 Revenue: $4.2M..."}
    ]},

    # Claude's final answer
    {"role": "assistant", "content": [
        {"type": "text", "text": "Here is a summary of report.txt: Q3 revenue was $4.2M..."}
    ]},
]
```

This entire list is sent to Claude on every new turn — it is the conversation "memory".

---

## 4.3 The Exception Hierarchy

```python
# app/domain/exceptions.py
"""Typed domain error hierarchy.

Inner layers raise these; the Presentation layer catches and maps them to HTTP responses.
Never expose raw stack traces to the client.
"""


class AgentError(Exception):
    """Base error for all agent domain failures."""


class ToolSecurityError(AgentError):
    """A tool call violated path-traversal security constraints."""


class ToolExecutionError(AgentError):
    """A tool failed during execution (non-recoverable at service level)."""


class SessionNotFoundError(AgentError):
    """Requested session_id does not exist in the store."""


class AgentLoopError(AgentError):
    """The agentic loop terminated in an unexpected state."""
```

### Error Propagation Strategy

```
tools.py  ──  NEVER raises — returns "Error: ..." strings for Claude to self-correct
              (tools should be fault-tolerant input for the LLM)

agent.py  ──  raises AgentLoopError on unexpected stop_reason or loop overflow

chat.py   ──  catches AgentError → returns ChatResponse(status="error")
              catches Exception  → logs with full traceback, returns generic error
```

This three-tier strategy means:
1. **Tool errors** let Claude retry with corrected parameters (self-healing)
2. **Agent errors** are surfaced as typed domain exceptions, not HTTP 500s
3. **Unknown errors** are logged but never exposed to clients

---

## 4.4 Quick Tests

Because the domain layer has zero external dependencies, tests are trivial:

```python
# tests/test_domain.py
from app.domain.models import AgentSession
from app.domain.exceptions import AgentLoopError


def test_agent_session_adds_user_message():
    session = AgentSession(session_id="test-1")
    session.add_user_message("Hello")
    assert session.messages == [{"role": "user", "content": "Hello"}]


def test_agent_session_add_tool_results():
    session = AgentSession(session_id="test-2")
    results = [{"type": "tool_result", "tool_use_id": "id1", "content": "file content"}]
    session.add_tool_results(results)
    assert session.messages[0]["role"] == "user"
    assert session.messages[0]["content"] == results


def test_agent_loop_error_is_agent_error():
    from app.domain.exceptions import AgentError
    err = AgentLoopError("loop exceeded")
    assert isinstance(err, AgentError)
```

```bash
uv run pytest tests/test_domain.py -v
# 3 passed in 0.12s
```

No mocks, no fixtures, no HTTP client — pure Python.

---

## Chapter Summary

| File | Purpose | External Deps |
|------|---------|---------------|
| `app/domain/models.py` | `AgentSession` dataclass — conversation thread | None |
| `app/domain/exceptions.py` | Typed error hierarchy | None |

**The `messages` list format is the Anthropic API's wire format** — understand it deeply, because the entire agentic loop depends on building and sending it correctly.

---

Next: [Chapter 05 → Tool System](./05_tool_system.md)
