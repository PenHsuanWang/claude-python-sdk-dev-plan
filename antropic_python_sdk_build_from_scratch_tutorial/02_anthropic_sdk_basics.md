# Chapter 02 — Anthropic SDK Basics

> **Goal:** Understand the Anthropic Python SDK's core primitives — the client, `messages.create()`, content blocks, `stop_reason`, and tool use — before wiring them into the full agent.

---

## 2.1 The Client

The Anthropic Python SDK exposes two client classes:

| Class | When to use |
|---|---|
| `Anthropic` | Synchronous code (scripts, tests) |
| `AsyncAnthropic` | `async/await` code — FastAPI, async scripts |

In a FastAPI application **always use `AsyncAnthropic`** so you don't block the event loop.

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(
    api_key="sk-ant-...",          # required — your Anthropic API key
    base_url=None,                 # optional — override for corporate proxy
    max_retries=2,                 # automatic retry on transient failures (429, 5xx)
)
```

**In the real project** the client is created once at module level (connection-pool reuse):

```python
# app/services/agent.py (excerpt)
from anthropic import AsyncAnthropic
from app.core.config import settings

_client = AsyncAnthropic(
    api_key=settings.anthropic_api_key.get_secret_value(),  # unwrap SecretStr here
    base_url=settings.anthropic_base_url,                   # None → default endpoint
    max_retries=settings.max_retries,
)
```

> ⚠️ **Never** re-instantiate the client inside a request handler — create it once at module level and reuse it. This allows the SDK to maintain an underlying connection pool.

---

## 2.2 The Messages API

Every interaction with Claude goes through `client.messages.create()`.

### Minimal example

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key="sk-ant-...")

async def simple_chat():
    response = await client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )
    print(response.content[0].text)  # → "The capital of France is Paris."

asyncio.run(simple_chat())
```

### Key parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | Which Claude model to use (e.g., `claude-3-7-sonnet-20250219`) |
| `max_tokens` | `int` | Maximum tokens in Claude's **response** (not the prompt) |
| `messages` | `list[dict]` | The conversation history in `[{role, content}, ...]` format |
| `tools` | `list[dict]` | Optional — list of tool schemas Claude can call (covered in Ch. 05) |
| `system` | `str` | Optional — a system prompt placed before the conversation |

---

## 2.3 The Response Object (`Message`)

`messages.create()` returns a `Message` object with these important fields:

```python
response = await client.messages.create(...)

response.id           # "msg_01XFDUDYJgAACzvnptvVoYEL"
response.model        # "claude-3-7-sonnet-20250219"
response.stop_reason  # "end_turn" | "tool_use" | "max_tokens" | "stop_sequence"
response.content      # list of content blocks
response.usage        # Usage(input_tokens=25, output_tokens=128)
```

### `stop_reason` — the most important field for agents

| `stop_reason` | Meaning | Action |
|---|---|---|
| `"end_turn"` | Claude finished its answer naturally | Extract text, return to user |
| `"tool_use"` | Claude wants to call one or more tools | Execute tools, feed results back |
| `"max_tokens"` | Response was cut off (hit `max_tokens`) | Consider increasing limit or chunking |
| `"stop_sequence"` | Hit a custom stop sequence | Depends on your use case |

**The entire agentic loop is built around this `stop_reason` field.** You will see this in Chapter 06.

---

## 2.4 Content Blocks

`response.content` is a **list** because Claude can return multiple blocks in one turn (e.g., a text block followed by a tool_use block).

### `TextBlock`

```python
# stop_reason == "end_turn" → expect TextBlock(s)
for block in response.content:
    if hasattr(block, "text"):
        print(block.text)   # Claude's final answer as a plain string
```

### `ToolUseBlock`

```python
# stop_reason == "tool_use" → expect ToolUseBlock(s)
for block in response.content:
    if block.type == "tool_use":
        print(block.id)     # "toolu_01A09q90qw90lq917835lq9" — needed for tool_result
        print(block.name)   # "read_local_document" — the tool function to call
        print(block.input)  # {"file_name": "report.txt"} — arguments dict
```

---

## 2.5 The Conversation History Format

The `messages` parameter is a list of alternating `user`/`assistant` turns. Understanding this format is critical because **you** are responsible for maintaining it (Claude has no memory between API calls).

```python
messages = [
    # Turn 1: user asks a question
    {"role": "user", "content": "What files are available?"},

    # Turn 2: assistant responds with a tool_use block
    {"role": "assistant", "content": [
        {
            "type": "tool_use",
            "id": "toolu_01A09q90qw90lq917835lq9",
            "name": "list_local_documents",
            "input": {}
        }
    ]},

    # Turn 3: user provides the tool result
    {"role": "user", "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
            "content": "Available documents:\n- report.txt\n- notes.md"
        }
    ]},

    # Turn 4: assistant gives the final answer
    {"role": "assistant", "content": [
        {"type": "text", "text": "You have 2 documents available: report.txt and notes.md."}
    ]},
]
```

**Rules enforced by the Anthropic API:**
1. Messages must alternate `user` / `assistant` / `user` / `assistant` ...
2. Tool results go in a **`user`** turn (not assistant), wrapped in `{"type": "tool_result", ...}`.
3. The `tool_use_id` in the result must exactly match the `id` from the `tool_use` block.

---

## 2.6 A Standalone "Hello Claude" Script

Before building the full agent, verify your API key works:

```python
# scripts/hello_claude.py
import asyncio
import os
from anthropic import AsyncAnthropic

async def main():
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = await client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "Say hello in exactly one sentence."}
        ],
    )

    print("stop_reason:", response.stop_reason)
    print("response:   ", response.content[0].text)
    print("tokens used:", response.usage.input_tokens, "+", response.usage.output_tokens)

asyncio.run(main())
```

```bash
uv run python scripts/hello_claude.py
# stop_reason: end_turn
# response:    Hello! I'm Claude, an AI assistant made by Anthropic — how can I help you today?
# tokens used: 16 + 27
```

---

## 2.7 SDK Versioning Note

The Anthropic Python SDK is under active development.  Pin a minimum version in `pyproject.toml`:

```toml
"anthropic>=0.40.0"
```

Key milestone versions:
- `0.25+` — `Tool` and `ToolUseBlock` types stabilized
- `0.34+` — `model_dump()` available on all content blocks
- `0.40+` — current stable, used in this project

---

## Chapter Summary

| Concept | Key Takeaway |
|---|---|
| `AsyncAnthropic` | Use async client in FastAPI — create once at module level |
| `messages.create()` | The single entry point for all Claude interactions |
| `stop_reason` | `"end_turn"` = done; `"tool_use"` = run a tool and loop back |
| `response.content` | A list of `TextBlock` or `ToolUseBlock` objects |
| Conversation history | A list you maintain yourself — Claude has no memory otherwise |
| Tool results | Go in a `user` turn, referencing the `tool_use_id` |

---

Next: [Chapter 03 → Clean Architecture Overview](./03_clean_architecture.md)
