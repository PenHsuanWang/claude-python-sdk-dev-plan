# Build an AI Agent from Scratch with Anthropic Python SDK + FastAPI

> A step-by-step tutorial that takes you from zero to a production-quality AI Agent — modelled on a real working codebase.

---

## What You Will Build

A **Local File Reader & Summarizer AI Agent**: a FastAPI service where users send natural language questions, and Claude autonomously decides which local files to read, loops through tool calls until it has enough context, and returns a final summarised answer.

```
User ──▶ POST /api/v1/chat ──▶ FastAPI ──▶ AgentService
                                                │
                                    ┌─────────────────────┐
                                    │    AGENTIC LOOP      │
                                    │  1. Call Claude API  │◀───────────────┐
                                    │  2. Check stop_reason│                │
                                    └─────────┬───────────┘                │
                                              │                              │
                                    stop_reason == "tool_use"               │
                                              │ YES                          │
                                    Execute Python tool ── result ──────────┘
                                              │ NO (end_turn)
                                    Return final text ──▶ User
```

---

## Tutorial Chapters

| # | Chapter | Key Concepts |
|---|---------|-------------|
| 01 | [Project Setup](./01_project_setup.md) | `uv`, `pyproject.toml`, `.env`, `pydantic-settings` |
| 02 | [Anthropic SDK Basics](./02_anthropic_sdk_basics.md) | `AsyncAnthropic`, `messages.create()`, `stop_reason`, content blocks |
| 03 | [Clean Architecture Overview](./03_clean_architecture.md) | Layers, dependency rules, why this structure |
| 04 | [Domain Layer](./04_domain_layer.md) | `AgentSession` dataclass, typed exceptions |
| 05 | [Tool System](./05_tool_system.md) | Tool definitions, JSON Schema, tool registry, security |
| 06 | [The Agentic Loop](./06_agentic_loop.md) | `AgentService.run()`, `tool_use` → `tool_result` cycle |
| 07 | [Session Management](./07_session_management.md) | `InMemorySessionStore`, conversation history format |
| 08 | [FastAPI Integration](./08_fastapi_integration.md) | Routers, Pydantic schemas, error handling, lifespan |
| 09 | [Security Patterns](./09_security.md) | Path traversal, `SecretStr`, graceful degradation |
| 10 | [Running & Extending](./10_running_and_extending.md) | Starting the server, curl examples, adding new tools |

---

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- An [Anthropic API key](https://console.anthropic.com/)
- Basic familiarity with Python async/await and FastAPI

---

## Quick Reference: Final Project Structure

```
ai-agent-mvp/
├── .env                        ← your secrets (never commit)
├── .env.example                ← template
├── pyproject.toml              ← dependencies & tool config
├── local_data/                 ← sandboxed document directory
│   └── sample_report.txt
└── app/
    ├── main.py                 ← FastAPI app factory + lifespan
    ├── core/
    │   └── config.py           ← pydantic-settings (SecretStr)
    ├── domain/
    │   ├── models.py           ← AgentSession (zero external deps)
    │   └── exceptions.py       ← typed error hierarchy
    ├── schemas/
    │   ├── chat.py             ← ChatRequest / ChatResponse
    │   └── documents.py        ← ListResponse / UploadResponse / DeleteResponse
    ├── api/v1/
    │   ├── chat.py             ← POST /api/v1/chat
    │   └── documents.py        ← CRUD for local documents
    └── services/
        ├── agent.py            ← AgentService — the agentic loop ★
        ├── memory.py           ← InMemorySessionStore
        └── tools.py            ← tool definitions + registry ★
```

---

## The Two Most Important Files

If you only read two files in the entire codebase, read these:

1. **`app/services/agent.py`** — implements the `tool_use` → `tool_result` loop that is the heart of every Anthropic-powered agent.
2. **`app/services/tools.py`** — shows exactly how tool schemas are structured and why tools must return strings (never raise exceptions).

---

Start with [Chapter 01 → Project Setup](./01_project_setup.md)
